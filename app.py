# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雲端專用版 / 生產環境就緒）
# ✅ 双模式 AI 判断：回档等待型 + 趋势突破型
# ✅ 自动运行 + 手动 RUN 按钮（专业 UX）
# ✅ 中文股票名称自动抓取（TWSE 上市 + 上櫃）
# ✅ 只显示「未来预估买卖点」（不显示历史买卖点）
# ✅ 多指标共振：MACD + KD + 乖离率 + 成交量 + 布林 + 支撑/压力
# ✅ split 修正 4967O 问题
# ✅ .TW/.TWO 双尾码 fallback（只用 Yahoo Finance）
# ✅ Top 10 扫描器（以「共振确认度」排序）
# ✅ 自动更新：随时更新 (每 N 分钟) 或 收盘后更新 (台北时间 13:30)
# ✅ st.toast 相容旧版 Streamlit
# ✅ 自动运行只在完整 4 码代号时触发

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import warnings
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# ---- optional autorefresh component ----
AUTOREFRESH_AVAILABLE = True
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    AUTOREFRESH_AVAILABLE = False

TWSE_ISIN_LISTED = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"  # 上市
TWSE_ISIN_OTC    = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃

TZ_TAIPEI = ZoneInfo("Asia/Taipei")
TW_CLOSE_HOUR = 13
TW_CLOSE_MIN = 30


# -----------------------------
# Time helpers (for cache/refresh)
# -----------------------------
def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)

def seconds_until_next_close_refresh(buffer_minutes: int = 10) -> int:
    """
    回传：距离「下一次收盘后更新点」的秒数（用于 cache ttl）
    预设收盘 13:30，并加 buffer（例如 10 分钟 → 13:40 才更新）
    """
    now = now_taipei()
    today_close = now.replace(hour=TW_CLOSE_HOUR, minute=TW_CLOSE_MIN, second=0, microsecond=0)
    refresh_time = today_close + timedelta(minutes=buffer_minutes)

    if now < refresh_time:
        return max(60, int((refresh_time - now).total_seconds()))
    # 已过今日 refresh_time → 目标明天
    tomorrow = now + timedelta(days=1)
    tomorrow_close = tomorrow.replace(hour=TW_CLOSE_HOUR, minute=TW_CLOSE_MIN, second=0, microsecond=0)
    tomorrow_refresh = tomorrow_close + timedelta(minutes=buffer_minutes)
    return max(60, int((tomorrow_refresh - now).total_seconds()))


# -----------------------------
# Helpers（代号清洗 / 名称查询）
# -----------------------------
def clean_code(code: str) -> str:
    """避免 4967O / .TW / .TWO 等问题：只保留点号前 + 大写"""
    return str(code).strip().upper().split(".")[0]

def to_tw_symbol(code: str) -> str:
    """自动识别上市/上櫃（粗略：6/4 开头→TWO，其它→TW）"""
    c = clean_code(code)
    if not c:
        return ""
    if c.startswith("6") or c.startswith("4"):
        return f"{c}.TWO"
    return f"{c}.TW"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_tw_stock_name_map() -> dict:
    """
    从 TWSE ISIN 公告页抓取「代号→中文名」(上市 + 上櫃)
    """
    def fetch(url: str) -> dict:
        try:
            r = requests.get(url, timeout=20)
            r.encoding = "big5"
            tables = pd.read_html(r.text)
            if not tables:
                return {}
            df = tables[0]
            col0 = df.columns[0]
            out = {}
            for v in df[col0].astype(str).tolist():
                if " " in v:
                    code, name = v.split(" ", 1)
                elif "\u3000" in v:
                    code, name = v.split("\u3000", 1)
                else:
                    continue
                code = clean_code(code)
                name = str(name).strip()
                if code.isdigit() and len(code) == 4 and name:
                    out[code] = name
            return out
        except Exception:
            return {}

    merged = {}
    merged.update(fetch(TWSE_ISIN_LISTED))
    merged.update(fetch(TWSE_ISIN_OTC))
    return merged

def get_stock_name(code: str) -> str:
    """优先用 TWSE 动态对照表；没有就退回代号"""
    c = clean_code(code)
    name_map = load_tw_stock_name_map()
    return name_map.get(c, c)

def safe_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError("Expected single column series")
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return pd.Series(arr)


# -----------------------------
# Data download (Yahoo only)
# -----------------------------
def _download_ohlc_core(stock_no: str, period: str, interval: str) -> pd.DataFrame:
    candidates = [f"{stock_no}.TW", f"{stock_no}.TWO"]
    for sym in candidates:
        try:
            import yfinance as yf
            df = yf.download(
                sym,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                timeout=30,
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(how="all")
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if needed.issubset(set(df.columns)):
                return df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception:
            continue
    return pd.DataFrame()

def download_ohlc(stock_no: str, period: str, interval: str, refresh_mode: str) -> pd.DataFrame:
    """
    依 refresh_mode 决定 cache ttl：
    - 随时自动更新：ttl 固定 10~30 分钟
    - 收盘后更新：ttl = 距离下一个收盘后更新点
    """
    stock_no = clean_code(stock_no)

    if refresh_mode == "每日收盘后更新":
        ttl = seconds_until_next_close_refresh(buffer_minutes=10)
    else:
        ttl = 15 * 60  # 15 minutes

    @st.cache_data(show_spinner=False, ttl=ttl)
    def _cached(stock_no: str, period: str, interval: str) -> pd.DataFrame:
        return _download_ohlc_core(stock_no, period, interval)

    return _cached(stock_no, period, interval)


# -----------------------------
# Indicators
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = safe_series(df["Close"])
    high = safe_series(df["High"])
    low = safe_series(df["Low"])

    df["SMA20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["RSI14"] = RSIIndicator(close=close, window=14).rsi()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR14"] = atr.average_true_range()

    # MACD
    ema12 = EMAIndicator(close=close, window=12).ema_indicator()
    ema26 = EMAIndicator(close=close, window=26).ema_indicator()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = EMAIndicator(close=df["MACD"], window=9).ema_indicator()

    # KD
    kd = StochasticOscillator(high=high, low=low, close=close, window=14)
    df["K"] = kd.stoch()
    df["D"] = df["K"].rolling(3).mean()

    # Bias 乖离率
    ma5 = df["Close"].rolling(5).mean()
    ma10 = df["Close"].rolling(10).mean()
    df["Bias_5"] = (close - ma5) / ma5 * 100
    df["Bias_10"] = (close - ma10) / ma10 * 100
    df["Bias_20"] = (close - df["SMA20"]) / df["SMA20"] * 100

    # 成交量均量
    df["Vol_MA5"] = df["Volume"].rolling(5).mean()
    df["Vol_MA10"] = df["Volume"].rolling(10).mean()

    return df


# -----------------------------
# Support / Resistance
# -----------------------------
def calculate_support_resistance(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 20:
        return {}

    current_price = float(df["Close"].iloc[-1])
    recent_high = float(df["High"].tail(20).max())
    recent_low = float(df["Low"].tail(20).min())

    lookback = min(252, len(df))
    high_52w = float(df["High"].tail(lookback).max())
    low_52w = float(df["Low"].tail(lookback).min())

    bb_high = float(df["BB_High"].iloc[-1]) if pd.notna(df["BB_High"].iloc[-1]) else None
    bb_low = float(df["BB_Low"].iloc[-1]) if pd.notna(df["BB_Low"].iloc[-1]) else None

    def gap(p):
        if p is None or pd.isna(p):
            return None
        return round((p - current_price) / current_price * 100, 2)

    return {
        "压力位": {
            "近期高点": {"价": round(recent_high, 2), "差距": gap(recent_high)},
            "布林上轨": {"价": round(bb_high, 2) if bb_high else None, "差距": gap(bb_high)},
            "52 周高点": {"价": round(high_52w, 2), "差距": gap(high_52w)},
        },
        "支撑位": {
            "近期低点": {"价": round(recent_low, 2), "差距": gap(recent_low)},
            "布林下轨": {"价": round(bb_low, 2) if bb_low else None, "差距": gap(bb_low)},
            "52 周低点": {"价": round(low_52w, 2), "差距": gap(low_52w)},
        },
    }

def nearest_levels(sr: dict, price: float):
    sup = None
    res = None

    for _, v in (sr.get("支撑位", {}) or {}).items():
        if v and v.get("价") is not None and v["价"] < price:
            if sup is None or v["价"] > sup:
                sup = v["价"]

    for _, v in (sr.get("压力位", {}) or {}).items():
        if v and v.get("价") is not None and v["价"] > price:
            if res is None or v["价"] < res:
                res = v["价"]

    return sup, res


# -----------------------------
# 趋势强度判断（双模式核心）
# -----------------------------
def calculate_trend_strength(df: pd.DataFrame) -> dict:
    """
    判断当前市场状态：趋势市 or 震荡市
    返回：trend_type (trend/range), strength_score (0-100)
    """
    if df.empty or len(df) < 50:
        return {"trend_type": "range", "strength_score": 0, "signals": {}}
    
    signals = {}
    score = 0
    
    last = df.iloc[-1]
    
    # 1) ADX 趋势强度（简化版：用均线斜率代替）
    sma20_curr = float(last["SMA20"]) if pd.notna(last["SMA20"]) else 0
    sma20_prev = float(df["SMA20"].iloc[-5]) if len(df) >= 5 else sma20_curr
    adx = abs(sma20_curr - sma20_prev) / sma20_prev * 1000 if sma20_prev != 0 else 0
    adx = min(50, max(0, adx))  # 标准化到 0-50
    
    if adx > 25:
        score += 40
        signals["趋势强度"] = f"✅ 强趋势 (ADX={adx:.1f})"
    elif adx > 20:
        score += 20
        signals["趋势强度"] = f"⚠️ 中等 (ADX={adx:.1f})"
    else:
        signals["趋势强度"] = f"❌ 震荡 (ADX={adx:.1f})"
    
    # 2) 均线排列
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    
    if ma5 > ma10 > ma20:
        score += 30
        signals["均线排列"] = "✅ 多头排列"
    elif ma5 < ma10 < ma20:
        score += 30
        signals["均线排列"] = "✅ 空头排列"
    else:
        signals["均线排列"] = "❌ 混乱/震荡"
    
    # 3) 波动率（ATR/价格）
    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0
    close = float(last["Close"])
    atr_pct = (atr / close * 100) if close > 0 else 0
    
    if atr_pct > 3:
        score += 30
        signals["波动率"] = f"✅ 高波动 (ATR={atr_pct:.2f}%)"
    elif atr_pct > 2:
        score += 15
        signals["波动率"] = f"⚠️ 中等 (ATR={atr_pct:.2f}%)"
    else:
        signals["波动率"] = f"❌ 低波动 (ATR={atr_pct:.2f}%)"
    
    # 判断趋势类型
    trend_type = "trend" if score >= 60 else "range"
    
    return {
        "trend_type": trend_type,
        "strength_score": score,
        "signals": signals,
        "adx": adx,
        "atr_pct": atr_pct
    }


# -----------------------------
# 突破确认检查（趋势突破型专用）
# -----------------------------
def check_breakout_confirmation(df: pd.DataFrame, price: float, breakout_level: float, side: str) -> dict:
    """
    检查突破是否有效
    条件：收盘价突破 + 放量 + 指标配合
    """
    if df.empty:
        return {"confirmed": False, "score": 0, "signals": {}}
    
    last = df.iloc[-1]
    signals = {}
    score = 0
    
    # 1) 价格突破确认
    if side == "BUY":
        if price > breakout_level * 1.01:  # 突破 1% 以上
            score += 40
            signals["价格突破"] = f"✅ 有效突破 {round(breakout_level, 2)}"
        else:
            signals["价格突破"] = f"❌ 未有效突破"
    else:  # SELL
        if price < breakout_level * 0.99:  # 跌破 1% 以上
            score += 40
            signals["价格跌破"] = f"✅ 有效跌破 {round(breakout_level, 2)}"
        else:
            signals["价格跌破"] = f"❌ 未有效跌破"
    
    # 2) 成交量确认
    if "Volume" in df.columns:
        vol5 = float(last["Vol_MA5"]) if pd.notna(last["Vol_MA5"]) else 0
        vol10 = float(last["Vol_MA10"]) if pd.notna(last["Vol_MA10"]) else 0
        vol_ratio = vol5 / vol10 if vol10 > 0 else 1
        
        if vol_ratio > 1.5:  # 放量 50% 以上
            score += 35
            signals["成交量"] = f"✅ 大量突破 (x{vol_ratio:.2f})"
        elif vol_ratio > 1.2:
            score += 20
            signals["成交量"] = f"⚠️ 放量 (x{vol_ratio:.2f})"
        else:
            signals["成交量"] = f"❌ 未放量 (x{vol_ratio:.2f})"
    
    # 3) MACD 配合
    macd = float(last["MACD"]) if pd.notna(last["MACD"]) else 0
    macd_sig = float(last["MACD_Signal"]) if pd.notna(last["MACD_Signal"]) else 0
    
    if side == "BUY":
        if macd > macd_sig and macd > 0:
            score += 25
            signals["MACD"] = "✅ 多头确认"
        else:
            signals["MACD"] = "❌ 未确认"
    else:
        if macd < macd_sig and macd < 0:
            score += 25
            signals["MACD"] = "✅ 空头确认"
        else:
            signals["MACD"] = "❌ 未确认"
    
    return {
        "confirmed": score >= 60,
        "score": score,
        "signals": signals
    }


# -----------------------------
# Confluence check (future points)
# -----------------------------
def check_confluence_signals(df: pd.DataFrame, signal_type: str) -> dict:
    if df.empty or len(df) < 30:
        return {"score": 0, "max_score": 100, "confirmation_rate": 0.0, "confirmed": False, "signals": {}}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    signals = {}
    score = 0
    max_score = 0

    # 1) MACD (25)
    max_score += 25
    macd = float(last.get("MACD", np.nan))
    macd_sig = float(last.get("MACD_Signal", np.nan))
    macd_prev = float(prev.get("MACD", np.nan))
    macd_sig_prev = float(prev.get("MACD_Signal", np.nan))

    if not np.isnan(macd) and not np.isnan(macd_sig):
        if signal_type == "BUY":
            if macd > macd_sig:
                score += 15
                signals["MACD"] = "✅ 多头"
                if macd_prev <= macd_sig_prev and macd > macd_sig:
                    score += 10
                    signals["MACD"] = "✅ 黄金交叉"
            else:
                signals["MACD"] = "❌ 空头"
        else:
            if macd < macd_sig:
                score += 15
                signals["MACD"] = "✅ 空头"
                if macd_prev >= macd_sig_prev and macd < macd_sig:
                    score += 10
                    signals["MACD"] = "✅ 死亡交叉"
            else:
                signals["MACD"] = "❌ 多头"
    else:
        signals["MACD"] = "⚠️ 无资料"
        max_score -= 25

    # 2) KD (25)
    max_score += 25
    k = float(last.get("K", np.nan))
    d = float(last.get("D", np.nan))
    k_prev = float(prev.get("K", np.nan))
    d_prev = float(prev.get("D", np.nan))

    if not np.isnan(k) and not np.isnan(d):
        if signal_type == "BUY":
            if (k > d) and (k < 80):
                score += 15
                signals["KD"] = "✅ 多头"
                if k_prev <= d_prev and k > d:
                    score += 10
                    signals["KD"] = "✅ 黄金交叉"
            else:
                signals["KD"] = "❌ 超买或空头"
        else:
            if (k < d) or (k > 80):
                score += 15
                signals["KD"] = "✅ 空头或超买"
                if k_prev >= d_prev and k < d:
                    score += 10
                    signals["KD"] = "✅ 死亡交叉"
            else:
                signals["KD"] = "❌ 多头"
    else:
        signals["KD"] = "⚠️ 无资料"
        max_score -= 25

    # 3) Bias 乖离率 (25)
    max_score += 25
    bias5 = float(last.get("Bias_5", np.nan))
    bias10 = float(last.get("Bias_10", np.nan))
    bias20 = float(last.get("Bias_20", np.nan))

    if not np.isnan(bias5) and not np.isnan(bias10):
        if signal_type == "BUY":
            if (bias5 < 0) or (-5 <= bias5 <= 5):
                score += 15
                signals["乖离率"] = "✅ 合理或负乖离"
            else:
                signals["乖离率"] = "❌ 正乖离过大"
        else:
            if (bias5 > 5) or (bias20 > 10):
                score += 15
                signals["乖离率"] = "✅ 正乖离偏大"
            else:
                signals["乖离率"] = "❌ 乖离正常"
    else:
        signals["乖离率"] = "⚠️ 无资料"
        max_score -= 25

    # 4) Bollinger (15)
    max_score += 15
    close = float(last.get("Close", np.nan))
    bb_high = float(last.get("BB_High", np.nan))
    bb_low = float(last.get("BB_Low", np.nan))

    if not np.isnan(close) and not np.isnan(bb_high) and not np.isnan(bb_low):
        if signal_type == "BUY":
            if close <= bb_low * 1.02:
                score += 15
                signals["布林通道"] = "✅ 接近下轨"
            else:
                signals["布林通道"] = "❌ 未接近下轨"
        else:
            if close >= bb_high * 0.98:
                score += 15
                signals["布林通道"] = "✅ 接近上轨"
            else:
                signals["布林通道"] = "❌ 未接近上轨"
    else:
        signals["布林通道"] = "⚠️ 无资料"
        max_score -= 15

    # 5) Volume (10)
    max_score += 10
    if "Volume" in df.columns:
        recent_vol = df["Volume"].tail(5).mean()
        prev_vol = df["Volume"].tail(10).head(5).mean()
        ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0

        if signal_type == "BUY":
            if ratio > 1.2:
                score += 10
                signals["成交量"] = "✅ 放大"
            elif ratio > 0.8:
                score += 5
                signals["成交量"] = "⚠️ 正常"
            else:
                signals["成交量"] = "❌ 萎缩"
        else:
            if ratio > 1.5:
                score += 10
                signals["成交量"] = "✅ 大量"
            else:
                signals["成交量"] = "⚠️ 正常"
    else:
        signals["成交量"] = "⚠️ 无资料"
        max_score -= 10

    confirmation_rate = (score / max_score) if max_score > 0 else 0.0
    confirmed = confirmation_rate >= 0.6

    return {
        "score": score,
        "max_score": max_score,
        "confirmation_rate": round(confirmation_rate * 100, 1),
        "confirmed": confirmed,
        "signals": signals,
    }


# -----------------------------
# 双模式未来买卖点预估（核心修改）
# -----------------------------
def estimate_future_buy_sell_points(df: pd.DataFrame, rr: float, atr_mult: float, sr: dict) -> dict:
    """
    双模式 AI 判断：
    - 模式 1：回档等待型（Mean Reversion）- 适合震荡市
    - 模式 2：趋势突破型（Momentum）- 适合趋势市
    
    自动根据市场状态选择主导模式，但两种都显示供参考
    """
    if df.empty or len(df) < 30:
        return {}
    
    # 1) 判断市场状态
    trend_info = calculate_trend_strength(df)
    trend_type = trend_info["trend_type"]
    trend_score = trend_info["strength_score"]
    
    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
    sup, res = nearest_levels(sr, close)
    recent_high = float(df["High"].tail(20).max())
    
    def sl_tp(entry: float, side: str):
        if not np.isnan(atr) and atr > 0:
            if side == "BUY":
                sl = entry - atr_mult * atr
                tp = entry + rr * (entry - sl)
            else:
                sl = entry + atr_mult * atr
                tp = entry - rr * (sl - entry)
        else:
            if side == "BUY":
                sl = entry * 0.95
                tp = entry * 1.10
            else:
                sl = entry * 1.05
                tp = entry * 0.90
        return round(sl, 2), round(tp, 2)
    
    result = {
        "current_price": round(close, 2),
        "market_mode": trend_type,
        "trend_strength": trend_score,
        "trend_signals": trend_info["signals"],
        "future_buy_points": [],
        "future_sell_points": []
    }
    
    # ========================================
    # 🧊 模式 1：回档等待型（Mean Reversion）
    # ========================================
    
    # BUY - 回测支撑
    if sup:
        entry = sup * 1.01
        sl, tp = sl_tp(entry, "BUY")
        dist = (sup - close) / close * 100
        conf = check_confluence_signals(df, "BUY")
        
        result["future_buy_points"].append({
            "模式": "🧊 回档型",
            "情境": "📉 回测支撑买点",
            "预估买点": round(entry, 2),
            "停损": sl,
            "停利": tp,
            "条件": f"价格回测 {round(sup,2)}（距离：{dist:+.1f}%）",
            "共振确认": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通过": conf["confirmed"],
            "指标详情": conf["signals"],
            "优先级": "高" if abs(dist) < 8 and conf["confirmed"] else "中",
            "方向": "BUY",
            "适用市场": "震荡市"
        })
    
    # SELL - 触及压力
    if res:
        entry = res * 0.99
        sl, tp = sl_tp(entry, "SELL")
        dist = (res - close) / close * 100
        conf = check_confluence_signals(df, "SELL")
        
        result["future_sell_points"].append({
            "模式": "🧊 回档型",
            "情境": "🎯 触及压力卖点（获利）",
            "预估卖点": round(entry, 2),
            "停损": sl,
            "停利": tp,
            "条件": f"价格接近压力位 {round(res,2)}（距离：{dist:+.1f}%）",
            "共振确认": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通过": conf["confirmed"],
            "指标详情": conf["signals"],
            "优先级": "🟡 中" if conf["confirmed"] else "🟢 低",
            "类型": "获利",
            "方向": "SELL",
            "适用市场": "震荡市"
        })
    
    # ========================================
    # 🚀 模式 2：趋势突破型（Momentum）
    # ========================================
    
    # BUY - 突破压力
    if res:
        breakout_price = res * 1.01  # 突破 1%
        breakout_conf = check_breakout_confirmation(df, close, res, "BUY")
        conf = check_confluence_signals(df, "BUY")
        
        # 只在趋势市或突破确认时显示
        if trend_type == "trend" or breakout_conf["confirmed"]:
            sl, tp = sl_tp(breakout_price, "BUY")
            dist = (res - close) / close * 100
            
            result["future_buy_points"].append({
                "模式": "🚀 突破型",
                "情境": "🚀 突破压力买点（追涨）",
                "预估买点": round(breakout_price, 2),
                "停损": sl,
                "停利": tp,
                "条件": f"价格突破 {round(res,2)}（距离：{dist:+.1f}%）",
                "突破确认": f"{'✅ 有效' if breakout_conf['confirmed'] else '⚠️ 待确认'} ({breakout_conf['score']}分)",
                "突破细节": breakout_conf["signals"],
                "共振确认": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
                "共振率": conf["confirmation_rate"],
                "共振通过": conf["confirmed"],
                "指标详情": conf["signals"],
                "优先级": "🔴 高" if breakout_conf["confirmed"] and trend_type == "trend" else "🟡 中",
                "方向": "BUY",
                "适用市场": "趋势市"
            })
    
    # SELL - 跌破支撑
    if sup:
        breakdown_price = sup * 0.99  # 跌破 1%
        breakdown_conf = check_breakout_confirmation(df, close, sup, "SELL")
        conf = check_confluence_signals(df, "SELL")
        
        # 只在趋势市或跌破确认时显示
        if trend_type == "trend" or breakdown_conf["confirmed"]:
            sl, tp = sl_tp(breakdown_price, "SELL")
            dist = (sup - close) / close * 100
            
            result["future_sell_points"].append({
                "模式": "🚀 突破型",
                "情境": "🛑 跌破支撑卖点（停损）",
                "预估卖点": round(breakdown_price, 2),
                "停损": sl,
                "停利": "N/A",
                "条件": f"价格跌破 {round(sup,2)}（距离：{dist:+.1f}%）",
                "突破确认": f"{'✅ 有效' if breakdown_conf['confirmed'] else '⚠️ 待确认'} ({breakdown_conf['score']}分)",
                "突破细节": breakdown_conf["signals"],
                "共振确认": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
                "共振率": conf["confirmation_rate"],
                "共振通过": conf["confirmed"],
                "指标详情": conf["signals"],
                "优先级": "🔴 高" if breakdown_conf["confirmed"] and trend_type == "trend" else "🟡 中",
                "类型": "停损",
                "方向": "SELL",
                "适用市场": "趋势市"
            })
    
    # ========================================
    # 🚀 趋势延续买点（多头回踩）
    # ========================================
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    
    if ma5 > ma10 > ma20:  # 多头排列
        # 回踩 EMA20 买点
        entry = ma20 * 1.005
        sl, tp = sl_tp(entry, "BUY")
        conf = check_confluence_signals(df, "BUY")
        
        result["future_buy_points"].append({
            "模式": "🚀 突破型",
            "情境": "📊 趋势延续回踩买点",
            "预估买点": round(entry, 2),
            "停损": sl,
            "停利": tp,
            "条件": f"多头排列，回踩 MA20({round(ma20,2)}) 附近",
            "共振确认": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通过": conf["confirmed"],
            "指标详情": conf["signals"],
            "优先级": "🔴 高" if trend_type == "trend" and conf["confirmed"] else "🟡 中",
            "方向": "BUY",
            "适用市场": "趋势市"
        })
    
    # ========================================
    # 移动停利（两种模式通用）
    # ========================================
    if close < recent_high:
        trail_5 = recent_high * 0.95
        pullback_pct = (recent_high - close) / recent_high * 100
        
        result["future_sell_points"].append({
            "模式": "通用",
            "情境": "📊 移动停利（保护获利）",
            "预估卖点": f"{round(trail_5,2)} (-5%)",
            "停损": "N/A",
            "停利": "N/A",
            "条件": f"20 日高点≈{round(recent_high,2)}；目前回撤 {pullback_pct:.1f}%",
            "优先级": "🟢 中",
            "类型": "保护获利",
            "方向": "SELL",
            "适用市场": "所有市场"
        })
    
    return result


# -----------------------------
# Chart
# -----------------------------
def plot_chart(df: pd.DataFrame, title: str, sr: dict | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB High", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Low", line=dict(dash="dot")))

    if sr:
        for key, value in (sr.get("压力位", {}) or {}).items():
            if value and value.get("价") is not None:
                fig.add_hline(y=value["价"], line_dash="dash", line_color="rgba(255,0,0,0.35)", annotation_text=f"🔴 {key}")
        for key, value in (sr.get("支撑位", {}) or {}).items():
            if value and value.get("价") is not None:
                fig.add_hline(y=value["价"], line_dash="dash", line_color="rgba(0,255,0,0.35)", annotation_text=f"🟢 {key}")

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=600, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# -----------------------------
# Stock pool for Top 10
# -----------------------------
@st.cache_data(show_spinner=False, ttl=24*3600)
def default_stock_pool(limit: int = 200) -> list[str]:
    """
    用 TWSE 名称表当 stock pool（会包含很多，不建议全扫）
    这里预设取前 N 档（你可以改成你要的固定清单，例如 2330/2317/2454...）
    """
    name_map = load_tw_stock_name_map()
    codes = sorted(list(name_map.keys()))
    return codes[:limit]


def summarize_best_future_point(future: dict) -> dict | None:
    """
    从 future_buy_points + future_sell_points 中，挑一个「最值得列入 Top10 的点」
    排序规则：共振通过 > 共振率高 > 优先级
    """
    cands = []

    for b in future.get("future_buy_points", []):
        if "共振率" in b:
            cands.append({
                "方向": "BUY",
                "情境": b.get("情境"),
                "价位": b.get("预估买点"),
                "停损": b.get("停损"),
                "停利": b.get("停利"),
                "共振率": b.get("共振率", 0),
                "共振通过": bool(b.get("共振通过", False)),
                "优先级": b.get("优先级", "中"),
                "条件": b.get("条件", ""),
            })

    for s in future.get("future_sell_points", []):
        if "共振率" in s:
            cands.append({
                "方向": "SELL",
                "情境": s.get("情境"),
                "价位": s.get("预估卖点"),
                "停损": s.get("停损"),
                "停利": s.get("停利"),
                "共振率": s.get("共振率", 0),
                "共振通过": bool(s.get("共振通过", False)),
                "优先级": s.get("优先级", "🟡 中"),
                "条件": s.get("条件", ""),
            })

    if not cands:
        return None

    # 排序：先看 confirmed，再看共振率，再看优先级（高>中>低）
    pr_rank = {"🔴 高": 3, "高": 3, "🟡 中": 2, "中": 2, "🟢 低": 1, "低": 1}
    cands.sort(key=lambda x: (x["共振通过"], x["共振率"], pr_rank.get(x["优先级"], 2)), reverse=True)
    return cands[0]


def scan_top10(stock_list: list[str], period: str, interval: str, rr: float, atr_mult: float,
               refresh_mode: str, min_price: float, min_volume_k: float) -> pd.DataFrame:
    """
    Top10 扫描：只抓「未来预估买卖点」的最佳候选，依共振率排序
    min_volume_k 单位：千股（Yahoo Volume 是股数）
    """
    results = []
    pb = st.progress(0)
    status = st.empty()

    for i, code in enumerate(stock_list):
        c = clean_code(code)
        status.text(f"扫描中：{c} - {get_stock_name(c)} ({i+1}/{len(stock_list)})")

        df = download_ohlc(c, period, interval, refresh_mode)
        if df.empty or len(df) < 30:
            pb.progress((i+1)/len(stock_list)); continue

        df = add_indicators(df)

        last = df.iloc[-1]
        price = float(last["Close"])
        vol = float(last.get("Volume", 0.0))
        vol_k = vol / 1000.0

        if price < min_price or vol_k < min_volume_k:
            pb.progress((i+1)/len(stock_list)); continue

        sr = calculate_support_resistance(df)
        future = estimate_future_buy_sell_points(df, rr, atr_mult, sr)
        if not future:
            pb.progress((i+1)/len(stock_list)); continue

        best = summarize_best_future_point(future)
        if best is None:
            pb.progress((i+1)/len(stock_list)); continue

        results.append({
            "代号": c,
            "名称": get_stock_name(c),
            "现价": round(price, 2),
            "成交量 (千股)": round(vol_k, 1),
            "方向": best["方向"],
            "情境": best["情境"],
            "预估价位": best["价位"],
            "停损": best["停损"],
            "停利": best["停利"],
            "共振率 (%)": best["共振率"],
            "共振通过": "✅" if best["共振通过"] else "⚠️",
            "条件": best["条件"],
        })

        pb.progress((i+1)/len(stock_list))

    status.text("扫描完成！")
    pb.empty()

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # 排序：共振通过优先，再共振率
    out["_c"] = (out["共振通过"] == "✅").astype(int)
    out = out.sort_values(by=["_c", "共振率 (%)"], ascending=[False, False]).drop(columns=["_c"])
    return out.head(10)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant", layout="wide")
st.title("📈 AI Stock Trading Assistant（台股分析专业版 / 双模式 AI 判断）")
st.caption("双模式 AI 判断：🧊 回档等待型 + 🚀 趋势突破型 | 支撑/压力 + 布林 + MACD + KD + 乖离率 + 成交量 共振确认；不做自动下单。")

with st.sidebar:
    st.header("设定")

    mode = st.radio("选择模式", ["单一股票分析", "Top 10 扫描器"])

    refresh_mode = st.selectbox("自动更新模式", ["随时自动更新", "每日收盘后更新"], index=1)
    refresh_minutes = st.slider("随时更新：每几分钟刷新", 1, 60, 5, 1)

    # 启用自动 refresh（如果套件存在）
    if AUTOREFRESH_AVAILABLE:
        if refresh_mode == "随时自动更新":
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")
        else:
            # 收盘后更新：不用一直刷新；给一个低频刷新（例如每 60 分钟）即可
            st_autorefresh(interval=60 * 60 * 1000, key="auto_refresh_close")
    else:
        st.info("⚠️ 未安装 streamlit-autorefresh，因此不会自动刷新。")

    st.divider()
    period = st.selectbox("期间", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("K 线", ["1d", "1wk", "1mo"], index=0)

    st.divider()
    rr = st.slider("风险报酬比（Take Profit）", 1.0, 5.0, 2.0, 0.25)
    atr_mult = st.slider("Stop Loss ATR 倍数", 0.5, 5.0, 1.5, 0.25)

    st.divider()
    broker = st.selectbox("券商（仅显示，不下单）", ["元大", "富邦", "国泰", "凯基", "永丰", "其他"], index=0)

    # ✅ 修改 1：UI 区段（自动运行 + RUN 按钮）
    if mode == "单一股票分析":
        code = st.text_input("台股代号（例：2330、2317、0050）", value="2330")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run = st.button("RUN / 重新计算", type="primary")
        with col_btn2:
            auto_run = st.toggle("自动运行", value=True)

    else:
        st.subheader("Top 10 扫描设定")
        pool_size = st.slider("扫描股票数量", 20, 300, 120, 10)
        min_price = st.number_input("最低价格（元）", min_value=0.0, max_value=100000.0, value=50.0, step=10.0)
        min_volume_k = st.number_input("最低成交量（千股）", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_scan = st.button("🔍 开始扫描", type="primary")
        with col_btn2:
            auto_scan = st.toggle("自动运行", value=False)


# -----------------------------
# Single Stock
# -----------------------------
# ✅ 修改 2：启动条件（自动运行 + 手动 RUN + 4 码代号检查）
if mode == "单一股票分析":
    c_try = clean_code(code)
    auto_ok = c_try.isdigit() and len(c_try) == 4  # 只接受 4 码数字代号
    
    if run or (auto_run and auto_ok):
        # ✅ 修改 3：视觉回馈（相容旧版 Streamlit）
        try:
            st.toast("正在重新计算…", icon="⏳")
        except AttributeError:
            st.info("⏳ 正在重新计算…")
        except Exception:
            pass
        
        c = c_try
        symbol = to_tw_symbol(c)
        stock_name = get_stock_name(c)

        st.subheader("1) 下载股价资料（Yahoo）")
        with st.spinner(f"下载中... {symbol} {stock_name}"):
            df = download_ohlc(c, period=period, interval=interval, refresh_mode=refresh_mode)

        if df.empty:
            st.error(f"❌ 下载不到资料：{symbol}（{stock_name}）。建议换代号/期间/稍后再试。")
            st.stop()

        st.success(f"✅ 已下载：{symbol} {stock_name} / {period} / {interval}（券商：{broker}）")
        st.dataframe(df.tail(5), use_container_width=True)

        st.subheader("2) 指标计算 + 支撑压力")
        df = add_indicators(df)
        sr = calculate_support_resistance(df)

        fig = plot_chart(df, title=f"{symbol} {stock_name} Price + Indicators", sr=sr)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("3) 📊 关键支撑压力位")
        if sr:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🔴 压力位")
                for k, v in (sr.get("压力位", {}) or {}).items():
                    if v and v.get("价") is not None:
                        gap = f"({v['差距']:+.2f}%)" if v.get("差距") is not None else ""
                        st.info(f"**{k}**: {v['价']} {gap}")
            with col2:
                st.markdown("##### 🟢 支撑位")
                for k, v in (sr.get("支撑位", {}) or {}).items():
                    if v and v.get("价") is not None:
                        gap = f"({v['差距']:+.2f}%)" if v.get("差距") is not None else ""
                        st.info(f"**{k}**: {v['价']} {gap}")
        else:
            st.info("资料不足，无法计算支撑压力位")

        st.subheader("4) 🔮 未来预估买卖点（双模式 AI 判断）")
        future = estimate_future_buy_sell_points(df, rr=rr, atr_mult=atr_mult, sr=sr)

        if not future:
            st.info("资料不足，无法预估未来买卖点")
            st.stop()

        # 显示市场状态
        mode_emoji = "🚀" if future.get("market_mode") == "trend" else "🧊"
        mode_text = "趋势市" if future.get("market_mode") == "trend" else "震荡市"
        st.info(f"**当前市场状态**：{mode_emoji} {mode_text}（趋势强度：{future.get('trend_strength', 0)}%）")
        
        with st.expander("📊 查看趋势判断细节"):
            for k, v in future.get("trend_signals", {}).items():
                st.write(f"**{k}**: {v}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🟢 未来潜在买点")
            buys = future.get("future_buy_points", [])
            if not buys:
                st.info("目前无潜在买点")
            else:
                # 按模式分组显示
                reversion_buys = [b for b in buys if "回档型" in b.get("模式", "")]
                momentum_buys = [b for b in buys if "突破型" in b.get("模式", "")]
                
                if momentum_buys:
                    st.markdown("**🚀 趋势突破型**")
                    for i, b in enumerate(momentum_buys, 1):
                        st.success(
                            f"**{i}. {b['情境']}**\n\n"
                            f"预估买点：**{b['预估买点']}**\n\n"
                            f"停损：{b['停损']} | 停利：{b['停利']}\n\n"
                            f"条件：{b['条件']}\n\n"
                            f"突破确认：{b.get('突破确认', 'N/A')}\n\n"
                            f"共振确认：{b.get('共振确认', 'N/A')}\n\n"
                            f"优先级：{b.get('优先级', '中')} | 适用：{b.get('适用市场', '—')}"
                        )
                        with st.expander("📊 指标细节"):
                            for kk, vv in (b.get("突破细节") or {}).items():
                                st.write(f"**{kk}**: {vv}")
                            st.divider()
                            for kk, vv in (b.get("指标详情") or {}).items():
                                st.write(f"**{kk}**: {vv}")
                
                if reversion_buys:
                    st.markdown("**🧊 回档等待型**")
                    for i, b in enumerate(reversion_buys, 1):
                        st.info(
                            f"**{i}. {b['情境']}**\n\n"
                            f"预估买点：**{b['预估买点']}**\n\n"
                            f"停损：{b['停损']} | 停利：{b['停利']}\n\n"
                            f"条件：{b['条件']}\n\n"
                            f"共振确认：{b.get('共振确认', 'N/A')}\n\n"
                            f"优先级：{b.get('优先级', '中')} | 适用：{b.get('适用市场', '—')}"
                        )
                        with st.expander("📊 指标细节"):
                            for kk, vv in (b.get("指标详情") or {}).items():
                                st.write(f"**{kk}**: {vv}")

        with col2:
            st.markdown("##### 🔴 未来潜在卖点")
            sells = future.get("future_sell_points", [])
            if not sells:
                st.warning("目前无潜在卖点")
            else:
                # 按模式分组显示
                momentum_sells = [s for s in sells if "突破型" in s.get("模式", "")]
                reversion_sells = [s for s in sells if "回档型" in s.get("模式", "")]
                universal_sells = [s for s in sells if "通用" in s.get("模式", "")]
                
                if momentum_sells:
                    st.markdown("**🚀 趋势突破型（停损/反转）**")
                    for i, s in enumerate(momentum_sells, 1):
                        st.error(
                            f"**{i}. {s['情境']}**\n\n"
                            f"预估卖点：**{s['预估卖点']}**\n\n"
                            f"停损：{s['停损']} | 停利：{s['停利']}\n\n"
                            f"条件：{s['条件']}\n\n"
                            f"突破确认：{s.get('突破确认', 'N/A')}\n\n"
                            f"共振确认：{s.get('共振确认', 'N/A')}\n\n"
                            f"优先级：{s.get('优先级', '中')} | 适用：{s.get('适用市场', '—')}"
                        )
                        with st.expander("📊 指标细节"):
                            for kk, vv in (s.get("突破细节") or {}).items():
                                st.write(f"**{kk}**: {vv}")
                            st.divider()
                            for kk, vv in (s.get("指标详情") or {}).items():
                                st.write(f"**{kk}**: {vv}")
                
                if reversion_sells:
                    st.markdown("**🧊 回档等待型（获利）**")
                    for i, s in enumerate(reversion_sells, 1):
                        st.warning(
                            f"**{i}. {s['情境']}**\n\n"
                            f"预估卖点：**{s['预估卖点']}**\n\n"
                            f"停损：{s['停损']} | 停利：{s['停利']}\n\n"
                            f"条件：{s['条件']}\n\n"
                            f"共振确认：{s.get('共振确认', 'N/A')}\n\n"
                            f"优先级：{s.get('优先级', '中')} | 适用：{s.get('适用市场', '—')}"
                        )
                        with st.expander("📊 指标细节"):
                            for kk, vv in (s.get("指标详情") or {}).items():
                                st.write(f"**{kk}**: {vv}")
                
                if universal_sells:
                    st.markdown("**📊 通用策略**")
                    for i, s in enumerate(universal_sells, 1):
                        st.info(
                            f"**{i}. {s['情境']}**\n\n"
                            f"预估卖点：**{s['预估卖点']}**\n\n"
                            f"条件：{s['条件']}\n\n"
                            f"优先级：{s.get('优先级', '中')}"
                        )

        st.subheader("5) 指标快照（最近 10 笔）")
        snap_cols = ["Close","SMA20","EMA20","RSI14","BB_High","BB_Low","ATR14","MACD","MACD_Signal","K","D","Bias_5","Bias_10","Bias_20","Volume"]
        snap_cols = [c for c in snap_cols if c in df.columns]
        st.dataframe(df[snap_cols].tail(10), use_container_width=True)


# -----------------------------
# Top 10 Scanner
# -----------------------------
elif mode == "Top 10 扫描器" and (run_scan or auto_scan):
    st.subheader("🏆 Top 10 共振买点/卖点扫描（双模式 AI 判断）")
    st.caption(f"更新模式：{refresh_mode}；期间：{period}/{interval}")

    codes = default_stock_pool(limit=pool_size)

    with st.spinner("扫描中（依股票数量可能需要一些时间）..."):
        top10 = scan_top10(
            stock_list=codes,
            period=period,
            interval=interval,
            rr=rr,
            atr_mult=atr_mult,
            refresh_mode=refresh_mode,
            min_price=float(min_price),
            min_volume_k=float(min_volume_k),
        )

    if top10.empty:
        st.warning("没有扫到符合条件的股票（可能是筛选条件太严格或资料不足）。")
    else:
        st.success(f"找到 Top {len(top10)} 档（依共振通过/共振率排序）")
        st.dataframe(top10, use_container_width=True)

        csv = top10.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载扫描结果 (CSV)",
            data=csv,
            file_name=f"top10_future_points_{now_taipei().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

st.caption("⚠️ 本工具仅做分析提示，不构成投资建议；请自行评估风险。")