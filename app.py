# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雲端專用版）
# ✅ 中文股票名稱自動抓取（TWSE 上市+上櫃）
# ✅ 只顯示「未來預估買賣點」（不顯示歷史買賣點）
# ✅ 多指標共振：MACD + KD + 乖離率 + 成交量 + 布林 + 支撐/壓力
# ✅ split 修正 4967O 問題
# ✅ .TW/.TWO 雙尾碼 fallback（只用 Yahoo Finance）
# ✅ Top 10 掃描器（以「共振確認度」排序）
# ✅ 自動更新：隨時更新(每N分鐘) 或 收盤後更新(台北時間 13:30)

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
    回傳：距離「下一次收盤後更新點」的秒數（用於 cache ttl）
    預設收盤 13:30，並加 buffer（例如 10 分鐘 → 13:40 才更新）
    """
    now = now_taipei()
    today_close = now.replace(hour=TW_CLOSE_HOUR, minute=TW_CLOSE_MIN, second=0, microsecond=0)
    refresh_time = today_close + timedelta(minutes=buffer_minutes)

    if now < refresh_time:
        return max(60, int((refresh_time - now).total_seconds()))

    tomorrow = now + timedelta(days=1)
    tomorrow_close = tomorrow.replace(hour=TW_CLOSE_HOUR, minute=TW_CLOSE_MIN, second=0, microsecond=0)
    tomorrow_refresh = tomorrow_close + timedelta(minutes=buffer_minutes)
    return max(60, int((tomorrow_refresh - now).total_seconds()))


# -----------------------------
# Helpers（代號清洗 / 名稱查詢）
# -----------------------------
def clean_code(code: str) -> str:
    """避免 4967O / .TW / .TWO 等問題：只保留點號前 + 大寫"""
    return str(code).strip().upper().split(".")[0]

def to_tw_symbol(code: str) -> str:
    """自動識別上市/上櫃（粗略：6/4 開頭→TWO，其它→TW）"""
    c = clean_code(code)
    if not c:
        return ""
    if c.startswith("6") or c.startswith("4"):
        return f"{c}.TWO"
    return f"{c}.TW"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_tw_stock_name_map() -> dict[str, str]:
    """
    從 TWSE ISIN 公告頁抓取「代號→中文名」(上市+上櫃)
    - 以 big5 解碼
    - pd.read_html 可能回傳多表；逐表掃描第一欄，找「代號　名稱」格式
    """
    def fetch(url: str) -> dict[str, str]:
        try:
            r = requests.get(url, timeout=30)
            r.encoding = "big5"
            tables = pd.read_html(r.text)
            if not tables:
                return {}

            out: dict[str, str] = {}
            for t in tables:
                if t is None or t.empty:
                    continue
                col0 = t.columns[0]
                for v in t[col0].astype(str).tolist():
                    # TWSE 常見是「2330　台積電」(全形空白)
                    if "　" in v:
                        code, name = v.split("　", 1)
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

    merged: dict[str, str] = {}
    merged.update(fetch(TWSE_ISIN_LISTED))
    merged.update(fetch(TWSE_ISIN_OTC))
    return merged

def get_stock_name(code: str) -> str:
    """優先用 TWSE 動態對照表；沒有就退回代號"""
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
    依 refresh_mode 決定 cache ttl：
    - 隨時自動更新：ttl 固定 15 分鐘
    - 收盤後更新：ttl = 距離下一個收盤後更新點
    """
    stock_no = clean_code(stock_no)

    if refresh_mode == "每日收盤後更新":
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

    # Bias 乖離率
    ma5 = df["Close"].rolling(5).mean()
    ma10 = df["Close"].rolling(10).mean()
    df["Bias_5"] = (close - ma5) / ma5 * 100
    df["Bias_10"] = (close - ma10) / ma10 * 100
    df["Bias_20"] = (close - df["SMA20"]) / df["SMA20"] * 100

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
        "壓力位": {
            "近期高點": {"價": round(recent_high, 2), "差距": gap(recent_high)},
            "布林上軌": {"價": round(bb_high, 2) if bb_high else None, "差距": gap(bb_high)},
            "52周高點": {"價": round(high_52w, 2), "差距": gap(high_52w)},
        },
        "支撐位": {
            "近期低點": {"價": round(recent_low, 2), "差距": gap(recent_low)},
            "布林下軌": {"價": round(bb_low, 2) if bb_low else None, "差距": gap(bb_low)},
            "52周低點": {"價": round(low_52w, 2), "差距": gap(low_52w)},
        },
    }

def nearest_levels(sr: dict, price: float):
    sup = None
    res = None

    for _, v in (sr.get("支撐位", {}) or {}).items():
        if v and v.get("價") is not None and v["價"] < price:
            if sup is None or v["價"] > sup:
                sup = v["價"]

    for _, v in (sr.get("壓力位", {}) or {}).items():
        if v and v.get("價") is not None and v["價"] > price:
            if res is None or v["價"] < res:
                res = v["價"]

    return sup, res


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
                signals["MACD"] = "✅ 多頭"
                if macd_prev <= macd_sig_prev and macd > macd_sig:
                    score += 10
                    signals["MACD"] = "✅ 黃金交叉"
            else:
                signals["MACD"] = "❌ 空頭"
        else:
            if macd < macd_sig:
                score += 15
                signals["MACD"] = "✅ 空頭"
                if macd_prev >= macd_sig_prev and macd < macd_sig:
                    score += 10
                    signals["MACD"] = "✅ 死亡交叉"
            else:
                signals["MACD"] = "❌ 多頭"
    else:
        signals["MACD"] = "⚠️ 無資料"
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
                signals["KD"] = "✅ 多頭"
                if k_prev <= d_prev and k > d:
                    score += 10
                    signals["KD"] = "✅ 黃金交叉"
            else:
                signals["KD"] = "❌ 超買或空頭"
        else:
            if (k < d) or (k > 80):
                score += 15
                signals["KD"] = "✅ 空頭或超買"
                if k_prev >= d_prev and k < d:
                    score += 10
                    signals["KD"] = "✅ 死亡交叉"
            else:
                signals["KD"] = "❌ 多頭"
    else:
        signals["KD"] = "⚠️ 無資料"
        max_score -= 25

    # 3) Bias 乖離率 (25)
    max_score += 25
    bias5 = float(last.get("Bias_5", np.nan))
    bias10 = float(last.get("Bias_10", np.nan))
    bias20 = float(last.get("Bias_20", np.nan))

    if not np.isnan(bias5) and not np.isnan(bias10):
        if signal_type == "BUY":
            if (bias5 < 0) or (-5 <= bias5 <= 5):
                score += 15
                signals["乖離率"] = "✅ 合理或負乖離"
            else:
                signals["乖離率"] = "❌ 正乖離過大"
        else:
            if (bias5 > 5) or (bias20 > 10):
                score += 15
                signals["乖離率"] = "✅ 正乖離偏大"
            else:
                signals["乖離率"] = "❌ 乖離正常"
    else:
        signals["乖離率"] = "⚠️ 無資料"
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
                signals["布林通道"] = "✅ 接近下軌"
            else:
                signals["布林通道"] = "❌ 未接近下軌"
        else:
            if close >= bb_high * 0.98:
                score += 15
                signals["布林通道"] = "✅ 接近上軌"
            else:
                signals["布林通道"] = "❌ 未接近上軌"
    else:
        signals["布林通道"] = "⚠️ 無資料"
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
                signals["成交量"] = "❌ 萎縮"
        else:
            if ratio > 1.5:
                score += 10
                signals["成交量"] = "✅ 大量"
            else:
                signals["成交量"] = "⚠️ 正常"
    else:
        signals["成交量"] = "⚠️ 無資料"
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
# Future points (BUY/SELL)
# -----------------------------
def estimate_future_buy_sell_points(df: pd.DataFrame, rr: float, atr_mult: float, sr: dict) -> dict:
    if df.empty or len(df) < 30:
        return {}

    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["ATR14"]) if pd.notna(last.get("ATR14", np.nan)) else np.nan
    ema20 = float(last["EMA20"]) if pd.notna(last.get("EMA20", np.nan)) else np.nan
    sma20 = float(last["SMA20"]) if pd.notna(last.get("SMA20", np.nan)) else np.nan

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

    result = {"current_price": round(close, 2), "future_buy_points": [], "future_sell_points": []}

    # BUY 1: 突破壓力
    if res:
        entry = res * 1.01
        sl, tp = sl_tp(entry, "BUY")
        dist = (res - close) / close * 100
        conf = check_confluence_signals(df, "BUY")
        result["future_buy_points"].append({
            "情境": "🚀 突破壓力買點",
            "預估買點": round(entry, 2),
            "停損": sl,
            "停利": tp,
            "條件": f"價格突破 {round(res,2)}（距離：{dist:+.1f}%）",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "高" if dist < 10 and conf["confirmed"] else "中",
            "方向": "BUY",
        })

    # BUY 2: 回測支撐
    if sup:
        entry = sup * 1.01
        sl, tp = sl_tp(entry, "BUY")
        dist = (sup - close) / close * 100
        conf = check_confluence_signals(df, "BUY")
        result["future_buy_points"].append({
            "情境": "📉 回檔支撐買點",
            "預估買點": round(entry, 2),
            "停損": sl,
            "停利": tp,
            "條件": f"價格回測 {round(sup,2)}（距離：{dist:+.1f}%）",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "高" if abs(dist) < 10 and conf["confirmed"] else "低",
            "方向": "BUY",
        })

    # BUY 3: 均線偏多
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    if (ma5 > ma10 > ma20):
        entry = close * 1.01
        sl, tp = sl_tp(entry, "BUY")
        conf = check_confluence_signals(df, "BUY")
        result["future_buy_points"].append({
            "情境": "📊 均線多頭確認買點",
            "預估買點": round(entry, 2),
            "停損": sl,
            "停利": tp,
            "條件": "5/10/20MA 多頭排列",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "高" if conf["confirmed"] else "中",
            "方向": "BUY",
        })

    # SELL 1: 跌破支撐
    if sup:
        entry = sup * 0.99
        sl, tp = sl_tp(entry, "SELL")
        dist = (sup - close) / close * 100
        conf = check_confluence_signals(df, "SELL")
        result["future_sell_points"].append({
            "情境": "🛑 跌破支撐賣點（停損）",
            "預估賣點": round(entry, 2),
            "停損": sl,
            "停利": "N/A",
            "條件": f"價格跌破 {round(sup,2)}（距離：{dist:+.1f}%）",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "🔴 高" if conf["confirmed"] else "🟡 中",
            "類型": "停損",
            "方向": "SELL",
        })

    # SELL 2: 接近壓力
    if res:
        entry = res * 0.99
        sl, tp = sl_tp(entry, "SELL")
        dist = (res - close) / close * 100
        conf = check_confluence_signals(df, "SELL")
        result["future_sell_points"].append({
            "情境": "🎯 觸及壓力賣點（獲利）",
            "預估賣點": round(entry, 2),
            "停損": sl,
            "停利": tp,
            "條件": f"價格接近壓力位 {round(res,2)}（距離：{dist:+.1f}%）",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "🟡 中" if conf["confirmed"] else "🟢 低",
            "類型": "獲利",
            "方向": "SELL",
        })

    # SELL 3: 均線反轉警戒
    if not np.isnan(ema20) and not np.isnan(sma20) and ema20 > sma20:
        entry = sma20 * 0.99
        sl, tp = sl_tp(entry, "SELL")
        conf = check_confluence_signals(df, "SELL")
        result["future_sell_points"].append({
            "情境": "📉 均線死亡交叉警戒",
            "預估賣點": round(entry, 2),
            "停損": sl,
            "停利": tp,
            "條件": f"EMA20 若跌破 SMA20（SMA20≈{round(sma20,2)}）",
            "共振確認": f"{conf['confirmation_rate']}% ({'✅' if conf['confirmed'] else '⚠️'})",
            "共振率": conf["confirmation_rate"],
            "共振通過": conf["confirmed"],
            "指標詳情": conf["signals"],
            "優先級": "🔴 高" if conf["confirmed"] else "🟡 中",
            "類型": "趨勢反轉",
            "方向": "SELL",
        })

    # SELL 4: 移動停利
    if close < recent_high:
        trail_5 = recent_high * 0.95
        pullback_pct = (recent_high - close) / recent_high * 100
        result["future_sell_points"].append({
            "情境": "📊 移動停利（保護獲利）",
            "預估賣點": f"{round(trail_5,2)} (-5% from 20D high)",
            "停損": "N/A",
            "停利": "N/A",
            "條件": f"20日高點≈{round(recent_high,2)}；目前回撤 {pullback_pct:.1f}%",
            "優先級": "🟢 中",
            "類型": "保護獲利",
            "方向": "SELL",
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
        for key, value in (sr.get("壓力位", {}) or {}).items():
            if value and value.get("價") is not None:
                fig.add_hline(y=value["價"], line_dash="dash", line_color="rgba(255,0,0,0.35)", annotation_text=f"🔴 {key}")
        for key, value in (sr.get("支撐位", {}) or {}).items():
            if value and value.get("價") is not None:
                fig.add_hline(y=value["價"], line_dash="dash", line_color="rgba(0,255,0,0.35)", annotation_text=f"🟢 {key}")

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=600, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# -----------------------------
# Stock pool for Top 10
# -----------------------------
@st.cache_data(show_spinner=False, ttl=24*3600)
def default_stock_pool(limit: int = 200) -> list[str]:
    name_map = load_tw_stock_name_map()
    codes = sorted(list(name_map.keys()))
    return codes[:limit]


def summarize_best_future_point(future: dict) -> dict | None:
    cands = []

    for b in future.get("future_buy_points", []):
        if "共振率" in b:
            cands.append({
                "方向": "BUY",
                "情境": b.get("情境"),
                "價位": b.get("預估買點"),
                "停損": b.get("停損"),
                "停利": b.get("停利"),
                "共振率": b.get("共振率", 0),
                "共振通過": bool(b.get("共振通過", False)),
                "優先級": b.get("優先級", "中"),
                "條件": b.get("條件", ""),
            })

    for s in future.get("future_sell_points", []):
        if "共振率" in s:
            cands.append({
                "方向": "SELL",
                "情境": s.get("情境"),
                "價位": s.get("預估賣點"),
                "停損": s.get("停損"),
                "停利": s.get("停利"),
                "共振率": s.get("共振率", 0),
                "共振通過": bool(s.get("共振通過", False)),
                "優先級": s.get("優先級", "🟡 中"),
                "條件": s.get("條件", ""),
            })

    if not cands:
        return None

    pr_rank = {"🔴 高": 3, "高": 3, "🟡 中": 2, "中": 2, "🟢 低": 1, "低": 1}
    cands.sort(key=lambda x: (x["共振通過"], x["共振率"], pr_rank.get(x["優先級"], 2)), reverse=True)
    return cands[0]


def scan_top10(stock_list: list[str], period: str, interval: str, rr: float, atr_mult: float,
               refresh_mode: str, min_price: float, min_volume_k: float) -> pd.DataFrame:
    results = []
    pb = st.progress(0)
    status = st.empty()

    for i, code in enumerate(stock_list):
        c = clean_code(code)
        status.text(f"掃描中：{c} - {get_stock_name(c)} ({i+1}/{len(stock_list)})")

        df = download_ohlc(c, period, interval, refresh_mode)
        if df.empty or len(df) < 30:
            pb.progress((i+1)/len(stock_list))
            continue

        df = add_indicators(df)

        last = df.iloc[-1]
        price = float(last["Close"])
        vol = float(last.get("Volume", 0.0))
        vol_k = vol / 1000.0

        if price < min_price or vol_k < min_volume_k:
            pb.progress((i+1)/len(stock_list))
            continue

        sr = calculate_support_resistance(df)
        future = estimate_future_buy_sell_points(df, rr, atr_mult, sr)
        if not future:
            pb.progress((i+1)/len(stock_list))
            continue

        best = summarize_best_future_point(future)
        if best is None:
            pb.progress((i+1)/len(stock_list))
            continue

        results.append({
            "代號": c,
            "名稱": get_stock_name(c),
            "現價": round(price, 2),
            "成交量(千股)": round(vol_k, 1),
            "方向": best["方向"],
            "情境": best["情境"],
            "預估價位": best["價位"],
            "停損": best["停損"],
            "停利": best["停利"],
            "共振率(%)": best["共振率"],
            "共振通過": "✅" if best["共振通過"] else "⚠️",
            "條件": best["條件"],
        })

        pb.progress((i+1)/len(stock_list))

    status.text("掃描完成！")
    pb.empty()

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["_c"] = (out["共振通過"] == "✅").astype(int)
    out = out.sort_values(by=["_c", "共振率(%)"], ascending=[False, False]).drop(columns=["_c"])
    return out.head(10)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant", layout="wide")
st.title("📈 AI Stock Trading Assistant（台股分析專業版 / 不自動下單）")
st.caption("只提供『未來預估買賣點』：支撐/壓力 + 布林 + MACD + KD + 乖離率 + 成交量 共振確認；不做自動下單。")

with st.sidebar:
    st.header("設定")

    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])

    # ✅ 重要：把「料號輸入」放回上方（就在模式下面）
    code = None
    if mode == "單一股票分析":
        code = st.text_input("台股代號（例：2330、2317、0050）", value="2330")

    st.divider()

    refresh_mode = st.selectbox("自動更新模式", ["隨時自動更新", "每日收盤後更新"], index=1)
    refresh_minutes = st.slider("隨時更新：每幾分鐘刷新", 1, 60, 5, 1)

    if AUTOREFRESH_AVAILABLE:
        if refresh_mode == "隨時自動更新":
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")
        else:
            # 收盤後更新：低頻刷新即可（避免一直重跑）
            st_autorefresh(interval=60 * 60 * 1000, key="auto_refresh_close")
    else:
        st.warning("⚠️ 未安裝 streamlit-autorefresh，因此不會自動刷新。")

    st.divider()
    period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("K 線", ["1d", "1wk", "1mo"], index=0)

    st.divider()
    rr = st.slider("風險報酬比（Take Profit）", 1.0, 5.0, 2.0, 0.25)
    atr_mult = st.slider("Stop Loss ATR 倍數", 0.5, 5.0, 1.5, 0.25)

    st.divider()
    broker = st.selectbox("券商（僅顯示，不下單）", ["元大", "富邦", "國泰", "凱基", "永豐", "其他"], index=0)

    if mode == "單一股票分析":
        run = st.button("RUN", type="primary")
    else:
        st.subheader("Top 10 掃描設定")
        pool_size = st.slider("掃描股票數量", 20, 300, 120, 10)
        min_price = st.number_input("最低價格（元）", min_value=0.0, max_value=100000.0, value=50.0, step=10.0)
        min_volume_k = st.number_input("最低成交量（千股）", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0)
        run_scan = st.button("🔍 開始掃描", type="primary")


# -----------------------------
# Single Stock
# -----------------------------
if mode == "單一股票分析" and run:
    c = clean_code(code or "")
    symbol = to_tw_symbol(c)
    stock_name = get_stock_name(c)

    st.subheader("1) 下載股價資料（Yahoo）")
    with st.spinner(f"下載中... {symbol} {stock_name}"):
        df = download_ohlc(c, period=period, interval=interval, refresh_mode=refresh_mode)

    if df.empty:
        st.error(f"❌ 下載不到資料：{symbol}（{stock_name}）。建議換代號/期間/稍後再試。")
        st.stop()

    st.success(f"✅ 已下載：{symbol} {stock_name} / {period} / {interval}（券商：{broker}）")
    st.dataframe(df.tail(5), use_container_width=True)

    st.subheader("2) 指標計算 + 支撐壓力")
    df = add_indicators(df)
    sr = calculate_support_resistance(df)

    fig = plot_chart(df, title=f"{symbol} {stock_name} Price + Indicators", sr=sr)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3) 📊 關鍵支撐壓力位")
    if sr:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔴 壓力位")
            for k, v in (sr.get("壓力位", {}) or {}).items():
                if v and v.get("價") is not None:
                    gap = f"({v['差距']:+.2f}%)" if v.get("差距") is not None else ""
                    st.info(f"**{k}**: {v['價']} {gap}")
        with col2:
            st.markdown("##### 🟢 支撐位")
            for k, v in (sr.get("支撐位", {}) or {}).items():
                if v and v.get("價") is not None:
                    gap = f"({v['差距']:+.2f}%)" if v.get("差距") is not None else ""
                    st.info(f"**{k}**: {v['價']} {gap}")
    else:
        st.info("資料不足，無法計算支撐壓力位")

    st.subheader("4) 🔮 未來預估買賣點（多指標共振）")
    future = estimate_future_buy_sell_points(df, rr=rr, atr_mult=atr_mult, sr=sr)

    if not future:
        st.info("資料不足，無法預估未來買賣點")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🟢 未來潛在買點")
        buys = future.get("future_buy_points", [])
        if not buys:
            st.info("目前無潛在買點")
        else:
            for i, b in enumerate(buys, 1):
                st.info(
                    f"**{i}. {b['情境']}**\n\n"
                    f"預估買點：**{b['預估買點']}**\n\n"
                    f"停損：{b['停損']}\n\n"
                    f"停利：{b['停利']}\n\n"
                    f"條件：{b['條件']}\n\n"
                    f"共振確認：{b.get('共振確認','N/A')}\n\n"
                    f"優先級：{b.get('優先級','中')}"
                )
                with st.expander("📊 指標共振細節"):
                    for kk, vv in (b.get("指標詳情") or {}).items():
                        st.write(f"**{kk}**: {vv}")

    with col2:
        st.markdown("##### 🔴 未來潛在賣點")
        sells = future.get("future_sell_points", [])
        if not sells:
            st.warning("目前無潛在賣點")
        else:
            for i, s in enumerate(sells, 1):
                st.warning(
                    f"**{i}. {s['情境']}**\n\n"
                    f"預估賣點：**{s['預估賣點']}**\n\n"
                    f"停損：{s['停損']}\n\n"
                    f"停利：{s['停利']}\n\n"
                    f"條件：{s['條件']}\n\n"
                    f"共振確認：{s.get('共振確認','N/A')}\n\n"
                    f"優先級：{s.get('優先級','中')}"
                )
                if "指標詳情" in s:
                    with st.expander("📊 指標共振細節"):
                        for kk, vv in (s.get("指標詳情") or {}).items():
                            st.write(f"**{kk}**: {vv}")

    st.subheader("5) 指標快照（最近 10 筆）")
    snap_cols = ["Close","SMA20","EMA20","RSI14","BB_High","BB_Low","ATR14","MACD","MACD_Signal","K","D","Bias_5","Bias_10","Bias_20","Volume"]
    snap_cols = [c for c in snap_cols if c in df.columns]
    st.dataframe(df[snap_cols].tail(10), use_container_width=True)


# -----------------------------
# Top 10 Scanner
# -----------------------------
elif mode == "Top 10 掃描器" and run_scan:
    st.subheader("🏆 Top 10 共振買點/賣點掃描（只看未來預估點）")
    st.caption(f"更新模式：{refresh_mode}；期間：{period}/{interval}")

    codes = default_stock_pool(limit=pool_size)

    with st.spinner("掃描中（依股票數量可能需要一些時間）..."):
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
        st.warning("沒有掃到符合條件的股票（可能是篩選條件太嚴格或資料不足）。")
    else:
        st.success(f"找到 Top {len(top10)} 檔（依共振通過/共振率排序）")
        st.dataframe(top10, use_container_width=True)

        csv = top10.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下載掃描結果 (CSV)",
            data=csv,
            file_name=f"top10_future_points_{now_taipei().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")
