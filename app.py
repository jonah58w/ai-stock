from __future__ import annotations

import os
import sys
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 內部模組
from data_loader import (
    normalize_symbol,
    get_tw_stock_info,
    load_price,
    load_fundamental,
)
from indicators import add_indicators

DATA_DIR      = "data"
LATEST_JSON   = os.path.join(DATA_DIR, "latest_scan.json")
HISTORY_CSV   = os.path.join(DATA_DIR, "scan_history.csv")
LEARNING_JSON = os.path.join(DATA_DIR, "ai_learning.json")

st.set_page_config(
    page_title="台股 AI 自動掃描",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# 基本工具
# =========================================================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def safe_float(x, default=0.0):
    try:
        if x is None: return default
        v = float(x)
        if np.isnan(v): return default
        return v
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None: return default
        return int(float(x))
    except Exception:
        return default

def fmt_num(v, digits=2):
    if v is None: return "-"
    try:
        if pd.isna(v): return "-"
    except Exception:
        pass
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)

def fmt_pct(v, digits=2):
    if v is None: return "-"
    try:
        if pd.isna(v): return "-"
    except Exception:
        pass
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return str(v)

def load_json_file(path, default_value):
    if not os.path.exists(path): return default_value
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

def auto_refresh_block(seconds):
    if seconds <= 0: return
    components.html(
        f"<script>setTimeout(function(){{window.parent.location.reload();}},{seconds*1000});</script>",
        height=0,
    )

def run_manual_scan():
    r = subprocess.run(
        [sys.executable, "scanner.py"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    return r.returncode, r.stdout, r.stderr

def empty_latest():
    return {
        "updated_at": "",
        "status": "empty",
        "message": "目前尚無掃描結果,請先手動掃描或等待排程執行。",
        "summary": {},
        "learning": {},
        "results": [],
        "failed_symbols": [],
    }

def get_finmind_token() -> str:
    """從 Streamlit secrets 或環境變數讀取 FinMind token"""
    try:
        if "FINMIND_TOKEN" in st.secrets:
            return str(st.secrets["FINMIND_TOKEN"]).strip()
    except Exception:
        pass
    return os.environ.get("FINMIND_TOKEN", "").strip()

# =========================================================
# Cache — 失敗不入庫
# =========================================================

@st.cache_data(ttl=60)
def load_latest():
    ensure_data_dir()
    return load_json_file(LATEST_JSON, empty_latest())

@st.cache_data(ttl=60)
def load_learning():
    return load_json_file(LEARNING_JSON, {
        "updated_at": "", "weights": {}, "thresholds": {}, "rule_stats": {},
    })

@st.cache_data(ttl=60)
def load_history():
    if not os.path.exists(HISTORY_CSV): return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

# =========================================================
# 價格抓取 — 帶重試 + MultiIndex 防線
# =========================================================

def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns and isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
    return df

def _yahoo_fetch_once(symbol: str, period: str) -> pd.DataFrame:
    """單次抓 Yahoo,含 curl_cffi session"""
    try:
        from curl_cffi import requests as cr
        session = cr.Session(impersonate="chrome110")
        ticker  = yf.Ticker(symbol, session=session)
    except Exception:
        ticker = yf.Ticker(symbol)

    df = ticker.history(period=period, interval="1d", auto_adjust=False, actions=False)
    if df is None or df.empty: return pd.DataFrame()
    df = _flatten_ohlcv(df)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns: return pd.DataFrame()
    return df[needed].dropna().copy()

def _yahoo_fetch_with_retry(symbol: str, period: str, max_retries: int = 3) -> pd.DataFrame:
    """失敗自動重試(被 rate limit 時退避)"""
    import time, random
    for attempt in range(max_retries):
        try:
            df = _yahoo_fetch_once(symbol, period)
            if not df.empty: return df
        except Exception as e:
            if "rate" in str(e).lower() or "too many" in str(e).lower():
                time.sleep((2 ** attempt) + random.random())
                continue
            if attempt < max_retries - 1:
                time.sleep(0.5 + random.random())
                continue
    return pd.DataFrame()

def fetch_chart(symbol: str, period: str = "9mo") -> pd.DataFrame:
    """從 Yahoo 抓 OHLCV,自動重試、補指標。失敗回傳空 DataFrame 但不進 cache"""
    df = _yahoo_fetch_with_retry(symbol, period)
    if df.empty: return df
    # 直接複用 indicators.add_indicators,跟 scanner 用同一套指標
    try:
        return add_indicators(df)
    except Exception:
        return df

def fetch_chart_by_code(code: str, period: str = "9mo") -> Tuple[str, pd.DataFrame]:
    """自動試 .TW / .TWO"""
    code = str(code).strip()
    if not code.isdigit(): return "", pd.DataFrame()
    for sym in [f"{code}.TW", f"{code}.TWO"]:
        df = fetch_chart(sym, period)
        if not df.empty: return sym, df
    return "", pd.DataFrame()

# 只 cache 成功的結果,失敗不進 cache (ttl=1800)
@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_chart(symbol: str, period: str) -> pd.DataFrame:
    df = fetch_chart(symbol, period)
    if df.empty:
        # 透過 raise 讓 cache 不保存 empty(Streamlit 會保存 exception,但下次呼叫會重算)
        return df
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_by_code(code: str, period: str) -> Tuple[str, pd.DataFrame]:
    sym, df = fetch_chart_by_code(code, period)
    return sym, df

# =========================================================
# 大盤環境判斷
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_market_regime() -> Dict[str, Any]:
    """抓加權指數 ^TWII,判斷大盤多空"""
    try:
        df = _yahoo_fetch_with_retry("^TWII", "6mo", max_retries=2)
        if df.empty:
            return {"status": "unknown", "regime": "未知", "reason": "加權指數資料抓取失敗"}
        df = add_indicators(df)
        if len(df) < 60:
            return {"status": "unknown", "regime": "未知", "reason": "歷史資料不足"}
        curr = df.iloc[-1]
        close = safe_float(curr["Close"])
        ma20  = safe_float(curr.get("SMA20"))
        ma60  = safe_float(curr.get("SMA60"))
        ma20_5d_ago = safe_float(df.iloc[-6]["SMA20"]) if len(df) >= 6 else ma20
        ma60_5d_ago = safe_float(df.iloc[-6]["SMA60"]) if len(df) >= 6 else ma60

        above_ma20 = close > ma20
        above_ma60 = close > ma60
        ma20_up = ma20 > ma20_5d_ago
        ma60_up = ma60 > ma60_5d_ago

        if above_ma20 and above_ma60 and ma20_up and ma60_up:
            regime, status, reason = "強多頭", "bull", "加權站上月線、季線,且雙線上彎"
        elif above_ma60 and ma60_up:
            regime, status, reason = "多頭", "bull", "加權站上季線且季線上彎"
        elif above_ma20 and ma20_up:
            regime, status, reason = "偏多整理", "neutral", "加權站上月線但季線尚未轉強"
        elif not above_ma60 and not ma60_up:
            regime, status, reason = "空頭", "bear", "加權跌破季線且季線下彎"
        elif not above_ma20:
            regime, status, reason = "偏空整理", "bear", "加權跌破月線"
        else:
            regime, status, reason = "盤整", "neutral", "加權方向不明"

        return {
            "status": status,
            "regime": regime,
            "reason": reason,
            "close": close,
            "ma20": ma20,
            "ma60": ma60,
            "pct_from_ma60": (close - ma60) / ma60 * 100 if ma60 > 0 else 0,
        }
    except Exception as e:
        return {"status": "unknown", "regime": "未知", "reason": f"判斷失敗: {e}"}

# =========================================================
# 深度技術分析(7 區塊)
# =========================================================

def _trend_diagnosis(df: pd.DataFrame) -> Dict[str, Any]:
    """趨勢診斷:均線排列 + 位階"""
    if df.empty or len(df) < 60:
        return {"alignment": "資料不足", "detail": "-", "dist_ma60": None, "year_pct": None}

    curr = df.iloc[-1]
    close = safe_float(curr["Close"])
    ma5  = safe_float(curr.get("SMA5") or curr.get("MA5"))
    ma10 = safe_float(curr.get("SMA10") or curr.get("MA10"))
    ma20 = safe_float(curr.get("SMA20") or curr.get("MA20"))
    ma60 = safe_float(curr.get("SMA60") or curr.get("MA60"))

    # 均線排列
    if close > ma5 > ma10 > ma20 > ma60:
        alignment = "🟢 完全多頭排列"
        detail = "短中長期均線由上而下完整多頭,趨勢最強"
    elif close > ma20 > ma60:
        alignment = "🟢 中期多頭"
        detail = "站上月線與季線,中期方向偏多"
    elif ma5 > ma10 and close > ma20:
        alignment = "🟡 短期轉強"
        detail = "短線均線金叉且站上月線,但中長期未確立"
    elif close < ma5 < ma10 < ma20 < ma60:
        alignment = "🔴 完全空頭排列"
        detail = "全面下跌排列,避免進場"
    elif close < ma60:
        alignment = "🔴 跌破季線"
        detail = "已失守中期支撐,趨勢轉差"
    else:
        alignment = "⚪ 整理"
        detail = "均線糾結,方向不明"

    # 距離季線
    dist_ma60 = (close - ma60) / ma60 * 100 if ma60 > 0 else 0

    # 近 240 日位階(百分位)
    year_pct = None
    if len(df) >= 120:
        year_window = df.tail(min(240, len(df)))
        high_y = safe_float(year_window["High"].max())
        low_y  = safe_float(year_window["Low"].min())
        if high_y > low_y:
            year_pct = (close - low_y) / (high_y - low_y) * 100

    return {
        "alignment": alignment,
        "detail": detail,
        "dist_ma60": dist_ma60,
        "year_pct": year_pct,
    }

def _volume_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """量價分析:健康放量 / 量縮整理 / 價漲量縮"""
    if df.empty or len(df) < 20:
        return {"status": "資料不足", "detail": "-", "vol_ratio": None}

    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else curr

    price_chg = (safe_float(curr["Close"]) - safe_float(prev["Close"])) / safe_float(prev["Close"]) * 100 \
                if safe_float(prev["Close"]) > 0 else 0
    vol_ratio = safe_float(curr.get("VOL_RATIO"))
    if vol_ratio == 0:
        vma20 = safe_float(curr.get("VOL_MA20"))
        vol_ratio = safe_float(curr["Volume"]) / vma20 if vma20 > 0 else 0

    if price_chg > 0.5 and vol_ratio >= 1.5:
        status, detail = "🟢 健康放量上攻", f"今日漲 {price_chg:.1f}%、量比 {vol_ratio:.2f},資金進場強勢"
    elif price_chg > 0.5 and vol_ratio < 0.8:
        status, detail = "🟡 價漲量縮", f"漲幅 {price_chg:.1f}% 但量比只 {vol_ratio:.2f},續漲動能不足"
    elif price_chg < -1 and vol_ratio >= 1.5:
        status, detail = "🔴 量增下跌", f"跌幅 {abs(price_chg):.1f}% 且量比 {vol_ratio:.2f},賣壓沉重"
    elif vol_ratio < 0.7:
        status, detail = "⚪ 量縮整理", f"量比 {vol_ratio:.2f},市場觀望"
    elif vol_ratio > 2.5:
        status, detail = "⚠️ 爆量", f"量比 {vol_ratio:.2f},留意籌碼換手風險"
    else:
        status, detail = "⚪ 量能正常", f"量比 {vol_ratio:.2f}"

    return {"status": status, "detail": detail, "vol_ratio": vol_ratio, "price_chg": price_chg}

def _signal_strength(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """訊號強度:拆解 A1/A2 條件 + 加分明細"""
    grade = str(row_data.get("grade", "-"))
    score = safe_int(row_data.get("score", 0))
    reasons = str(row_data.get("reasons", ""))

    a1_conds = [
        ("MA20 向上",     row_data.get("cond_a1_ma20_up")),
        ("站上季線",       row_data.get("cond_a1_above_ma60")),
        ("MACD 第 1 紅柱", row_data.get("cond_a1_macd_first_red")),
        ("MA5 穿 MA10",   row_data.get("cond_a1_ma5_cross_ma10")),
        ("突破 20 日高",  row_data.get("cond_a1_breakout_20d")),
        ("量增 1.5 倍",   row_data.get("cond_a1_vol_expand")),
        ("收紅 K",        row_data.get("cond_a1_red_candle")),
    ]
    a2_conds = [
        ("季線向上",      row_data.get("cond_a2_ma60_up")),
        ("多頭排列",      row_data.get("cond_a2_bullish_alignment")),
        ("回踩均線",      row_data.get("cond_a2_pullback_to_ma")),
        ("MACD 紅柱中",   row_data.get("cond_a2_macd_positive")),
        ("縮量回踩",      row_data.get("cond_a2_vol_shrink")),
        ("底部墊高",      row_data.get("cond_a2_higher_lows")),
    ]

    return {
        "grade": grade,
        "score": score,
        "a1_conds": a1_conds,
        "a2_conds": a2_conds,
        "reasons": reasons,
    }

def _risk_warnings(df: pd.DataFrame, row_data: Dict[str, Any],
                   trend: Dict[str, Any], market: Dict[str, Any]) -> List[str]:
    """風險警示清單"""
    warnings = []

    # 1. 大盤環境弱
    if market.get("status") == "bear":
        warnings.append(f"🔴 大盤{market.get('regime')},訊號勝率打折,建議減倉或觀望")
    elif market.get("status") == "neutral":
        warnings.append(f"🟡 大盤{market.get('regime')},建議降低單筆倉位")

    # 2. 距離季線過遠
    dist = trend.get("dist_ma60")
    if dist is not None and dist > 12:
        warnings.append(f"🔴 距季線 +{dist:.1f}%,追高風險極高,建議等回踩")
    elif dist is not None and dist > 8:
        warnings.append(f"🟡 距季線 +{dist:.1f}%,乖離偏大,分批買較安全")

    # 3. 位階過高
    year_pct = trend.get("year_pct")
    if year_pct is not None and year_pct > 90:
        warnings.append(f"🟡 年內位階 {year_pct:.0f}%(接近高點),反壓沉重")

    # 4. RSI 過熱
    if not df.empty:
        rsi = safe_float(df.iloc[-1].get("RSI"))
        if rsi >= 80:
            warnings.append(f"🔴 RSI {rsi:.0f} 嚴重過熱,短線修正機率高")
        elif rsi >= 70:
            warnings.append(f"🟡 RSI {rsi:.0f} 過熱區,留意拉回")

    # 5. 布林外
    if not df.empty:
        curr = df.iloc[-1]
        close = safe_float(curr["Close"])
        bb_up = safe_float(curr.get("BB_Upper") or curr.get("BB_UPPER"))
        if bb_up > 0 and close > bb_up * 1.02:
            warnings.append(f"🟡 股價突破布林上軌 2% 以上,短線過熱")

    # 6. 連漲天數
    if len(df) >= 5:
        closes = df["Close"].tail(6).tolist()
        up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        if up_days >= 5:
            warnings.append(f"🟡 連 {up_days} 日收紅,短線拉回機率升高")

    # 7. A1 但量比不足
    if str(row_data.get("grade")) == "A1":
        vr = safe_float(row_data.get("vol_ratio"))
        if 0 < vr < 1.3:
            warnings.append(f"🟡 A1 突破但量比只 {vr:.2f},突破有效性存疑")

    # 8. 週線未站上
    if not df.empty and len(df) >= 100:
        weekly_ma = df["Close"].rolling(100).mean().iloc[-1]
        if safe_float(df.iloc[-1]["Close"]) < safe_float(weekly_ma):
            warnings.append("🟡 週線中期均線尚未站上,波段基礎不穩")

    if not warnings:
        warnings.append("🟢 無明顯短線風險")

    return warnings

def _entry_strategy(df: pd.DataFrame, row_data: Dict[str, Any],
                    trend: Dict[str, Any]) -> Dict[str, Any]:
    """進場策略:三劇本 + 倉位建議"""
    if df.empty:
        return {}

    curr = df.iloc[-1]
    close = safe_float(curr["Close"])
    atr = safe_float(curr.get("ATR"))
    if atr == 0:
        atr = (safe_float(curr["High"]) - safe_float(curr["Low"])) or (close * 0.02)

    ma10 = safe_float(curr.get("SMA10") or curr.get("MA10"))
    ma20 = safe_float(curr.get("SMA20") or curr.get("MA20"))
    ma60 = safe_float(curr.get("SMA60") or curr.get("MA60"))

    grade = str(row_data.get("grade", ""))
    dist_ma60 = trend.get("dist_ma60") or 0

    # 倉位建議
    if dist_ma60 > 10:
        position = "20-30%(乖離過大)"
    elif dist_ma60 > 6:
        position = "40-50%"
    elif grade == "A1":
        position = "50-70%"
    elif grade == "A2":
        position = "60-80%(回踩型風險較低)"
    else:
        position = "30-50%"

    if grade == "A1":
        # 突破型
        aggressive = {"price": close, "desc": "今日收盤直接買進(追突破)"}
        moderate = {"price": ma10, "desc": "等拉回 MA10 再買(較穩)"}
        conservative = {"price": ma20, "desc": "等回踩 MA20 再買(最安全)"}
    else:
        # 回踩型
        aggressive = {"price": close, "desc": "今日價位直接買(已在回踩買點)"}
        moderate = {"price": ma20, "desc": "等再回踩 MA20 加碼"}
        conservative = {"price": ma60, "desc": "等回踩季線才買"}

    return {
        "aggressive": aggressive,
        "moderate": moderate,
        "conservative": conservative,
        "position": position,
        "atr": atr,
    }

def _stop_loss_targets(df: pd.DataFrame, row_data: Dict[str, Any],
                       entry: Dict[str, Any]) -> Dict[str, Any]:
    """停損停利:ATR 倍數 + 關鍵支撐雙算法"""
    if df.empty:
        return {}

    curr = df.iloc[-1]
    close = safe_float(curr["Close"])
    atr = entry.get("atr", close * 0.02)

    ma10 = safe_float(curr.get("SMA10") or curr.get("MA10"))
    ma20 = safe_float(curr.get("SMA20") or curr.get("MA20"))
    ma60 = safe_float(curr.get("SMA60") or curr.get("MA60"))

    # 停損:ATR 2 倍 vs MA10 取高者(較寬)
    atr_stop = close - 2 * atr
    stop_short = max(atr_stop, ma10) if ma10 > 0 else atr_stop
    stop_wave = ma20 if ma20 > 0 else close * 0.93
    stop_hard = ma60 if ma60 > 0 else close * 0.88

    # 停利:ATR 3 倍 vs 前波高點
    recent_high = safe_float(df.tail(60)["High"].max())
    target_1 = close + 3 * atr
    target_2 = max(recent_high * 1.05, close + 6 * atr)

    # 風報比
    risk = close - stop_short
    reward = target_1 - close
    rr = reward / risk if risk > 0 else 0

    return {
        "stop_short": stop_short,
        "stop_wave": stop_wave,
        "stop_hard": stop_hard,
        "target_1": target_1,
        "target_2": target_2,
        "rr": rr,
    }

# =========================================================
# 圖表 — 解析度升級
# =========================================================

_BG   = "#131722"
_PAN  = "#1e222d"
_GRID = "#2a2e39"
_TXT  = "#d1d4dc"
_UP   = "#ef5350"
_DN   = "#26a69a"

def make_pro_chart(df: pd.DataFrame, prices: Dict[str, float], title: str,
                   period_bars: int = 120) -> go.Figure:
    """專業級 K 線圖:4 面板 + 買賣點只畫最近 30 根"""
    cdf = df.tail(period_bars).copy()
    n = len(cdf)
    xs = list(range(n))
    dts = [d.strftime("%m/%d") for d in cdf.index]
    ud = [_UP if c >= o else _DN for c, o in zip(cdf["Close"], cdf["Open"])]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.16, 0.17, 0.17],
        vertical_spacing=0.025,
    )

    # ── 布林填充 ─────────────────────────────────
    bb_u = cdf.get("BB_Upper", cdf.get("BB_UPPER"))
    bb_m = cdf.get("BB_Mid",   cdf.get("BB_MID"))
    bb_l = cdf.get("BB_Lower", cdf.get("BB_LOWER"))
    if bb_u is not None and bb_l is not None:
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=list(bb_u) + list(bb_l[::-1]),
            fill="toself", fillcolor="rgba(41,98,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

        for col_data, lbl, color, dash, w in [
            (bb_u, "布林上", "#5c7cfa", "dot", 1),
            (bb_m, "布林中", "#ff9800", "dash", 1.2),
            (bb_l, "布林下", "#5c7cfa", "dot", 1),
        ]:
            if col_data is not None:
                fig.add_trace(go.Scatter(
                    x=xs, y=col_data, mode="lines", name=lbl,
                    line=dict(color=color, width=w, dash=dash),
                    hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>",
                ), row=1, col=1)

    # ── 均線:粗細分級 ───────────────────────────
    for col, lbl, color, w in [
        ("SMA5",  "MA5",  "#f48fb1", 1.0),
        ("SMA10", "MA10", "#ce93d8", 1.2),
        ("SMA20", "MA20", "#4dd0e1", 1.6),
        ("SMA60", "MA60", "#ffb74d", 2.2),
    ]:
        alt = col.replace("SMA", "MA")
        if col in cdf.columns:
            vals = cdf[col]
        elif alt in cdf.columns:
            vals = cdf[alt]
        else:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="lines", name=lbl,
            line=dict(color=color, width=w),
            hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # ── K 線 ─────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=xs,
        open=cdf["Open"], high=cdf["High"],
        low=cdf["Low"], close=cdf["Close"],
        name="K線",
        increasing=dict(line=dict(color=_UP, width=1), fillcolor=_UP),
        decreasing=dict(line=dict(color=_DN, width=1), fillcolor=_DN),
        hovertext=[
            f"開:{o:.2f} 高:{h:.2f} 低:{l:.2f} 收:{c:.2f}"
            for o, h, l, c in zip(cdf["Open"], cdf["High"], cdf["Low"], cdf["Close"])
        ],
        hoverlabel=dict(bgcolor=_PAN),
    ), row=1, col=1)

    # ── 買賣停損線 — 只畫最近 30 根 ───────────────
    ref_start = max(0, n - 30)
    for y_val, lbl, color, dash in [
        (prices.get("aggressive"),   "積極買", _UP,       "solid"),
        (prices.get("moderate"),     "穩健買", "#ff8a65", "dash"),
        (prices.get("conservative"), "保守買", "#fdd835", "dash"),
        (prices.get("target_1"),     "目標1",  "#42a5f5", "dot"),
        (prices.get("target_2"),     "目標2",  "#1976d2", "dot"),
        (prices.get("stop_short"),   "短停損", "#e53935", "dot"),
        (prices.get("stop_wave"),    "波停損", "#b71c1c", "dot"),
    ]:
        y = safe_float(y_val)
        if y <= 0: continue
        fig.add_trace(go.Scatter(
            x=list(range(ref_start, n)),
            y=[y] * (n - ref_start),
            mode="lines",
            line=dict(color=color, width=1, dash=dash),
            name=lbl,
            hovertemplate=f"{lbl}: {y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_annotation(
            x=n - 1, y=y, xref="x", yref="y",
            text=f"<b>{lbl}</b> {y:.1f}",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(19,23,34,0.8)",
            borderpad=2, xshift=50, xanchor="left",
            row=1, col=1,
        )

    # ── 成交量 ───────────────────────────────────
    fig.add_trace(go.Bar(
        x=xs, y=cdf["Volume"],
        marker_color=ud, marker_line_width=0, showlegend=False,
        hovertemplate="量: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    if "VOL_MA20" in cdf.columns:
        fig.add_trace(go.Scatter(
            x=xs, y=cdf["VOL_MA20"], mode="lines",
            line=dict(color="#ffeb3b", width=1.2), showlegend=False,
            hovertemplate="量MA20: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # ── KD ───────────────────────────────────────
    if "K" in cdf.columns and "D" in cdf.columns:
        fig.add_trace(go.Scatter(
            x=xs, y=cdf["K"], mode="lines", name="K",
            line=dict(color="#f48fb1", width=1.5),
            showlegend=False,
            hovertemplate="K: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=xs, y=cdf["D"], mode="lines", name="D",
            line=dict(color="#4dd0e1", width=1.5),
            showlegend=False,
            hovertemplate="D: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        # 80 / 50 / 20 參考線(實線)
        for y_ref, col, dash in [(80, "#e57373", "solid"),
                                  (50, "#666", "dot"),
                                  (20, "#81c784", "solid")]:
            fig.add_hline(y=y_ref, row=3, col=1,
                          line=dict(color=col, width=0.8, dash=dash))

    # ── MACD ─────────────────────────────────────
    hist_col = "MACD_Hist" if "MACD_Hist" in cdf.columns else \
               ("MACD_HIST" if "MACD_HIST" in cdf.columns else None)
    macd_col = "MACD" if "MACD" in cdf.columns else \
               ("DIF" if "DIF" in cdf.columns else None)
    sig_col = "MACD_Signal" if "MACD_Signal" in cdf.columns else \
              ("DEA" if "DEA" in cdf.columns else None)

    if hist_col:
        mc = [_UP if v >= 0 else _DN for v in cdf[hist_col]]
        fig.add_trace(go.Bar(
            x=xs, y=cdf[hist_col],
            marker_color=mc, marker_line_width=0, showlegend=False,
            hovertemplate="HIST: %{y:.4f}<extra></extra>",
        ), row=4, col=1)
    if macd_col:
        fig.add_trace(go.Scatter(
            x=xs, y=cdf[macd_col], mode="lines",
            line=dict(color="#f48fb1", width=1.3), showlegend=False,
            hovertemplate="MACD: %{y:.4f}<extra></extra>",
        ), row=4, col=1)
    if sig_col:
        fig.add_trace(go.Scatter(
            x=xs, y=cdf[sig_col], mode="lines",
            line=dict(color="#ffeb3b", width=1.3), showlegend=False,
            hovertemplate="Signal: %{y:.4f}<extra></extra>",
        ), row=4, col=1)

    # 零軸粗線
    fig.add_hline(y=0, row=4, col=1, line=dict(color="#888", width=1))

    # ── Layout ──────────────────────────────────
    step = max(1, n // 12)
    tv = xs[::step]
    tt = [dts[i] for i in tv]
    ax_base = dict(
        showgrid=True, gridcolor=_GRID, gridwidth=0.4,
        zeroline=False, showline=True, linecolor=_GRID,
        tickfont=dict(size=10, color=_TXT),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=_TXT, size=15), x=0.01),
        paper_bgcolor=_BG, plot_bgcolor=_PAN,
        font=dict(color=_TXT, size=11),
        height=820,
        margin=dict(l=10, r=110, t=40, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            orientation="h", x=0, y=1.04,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=_PAN, font_size=11, bordercolor=_GRID),
        xaxis =dict(**ax_base, tickvals=tv, ticktext=tt, tickangle=-30),
        xaxis2=dict(**ax_base, tickvals=tv, ticktext=tt, showticklabels=False),
        xaxis3=dict(**ax_base, tickvals=tv, ticktext=tt, showticklabels=False),
        xaxis4=dict(**ax_base, tickvals=tv, ticktext=tt, tickangle=-30),
        yaxis =dict(**ax_base, side="right"),
        yaxis2=dict(**ax_base, side="right"),
        yaxis3=dict(**ax_base, side="right", range=[-5, 105]),
        yaxis4=dict(**ax_base, side="right"),
    )

    # 副圖標籤
    for y_pos, lbl in [(0.76, "K 線 + 布林"), (0.46, "成交量"),
                        (0.30, "KD (9)"), (0.12, "MACD")]:
        fig.add_annotation(
            x=1.01, y=y_pos, xref="paper", yref="paper",
            text=lbl, showarrow=False,
            font=dict(size=10, color=_TXT), xanchor="left",
        )

    fig.update_xaxes(showspikes=True, spikecolor=_TXT, spikesnap="cursor",
                     spikemode="across", spikethickness=0.5)
    fig.update_yaxes(showspikes=True, spikecolor=_TXT, spikethickness=0.5)

    return fig

# =========================================================
# 技術分析渲染
# =========================================================

def render_technical_analysis(df: pd.DataFrame, row_data: Dict[str, Any],
                              market: Dict[str, Any]):
    """7 區塊結構化技術分析"""
    if df.empty:
        st.warning("技術分析無法產生(資料不足)")
        return None

    # 1. 大盤環境
    st.markdown("#### 📊 大盤環境")
    mc1, mc2, mc3 = st.columns([1, 1, 3])
    mc1.metric("加權狀態", market.get("regime", "未知"))
    mc2.metric("收盤", fmt_num(market.get("close"), 1))
    mc3.info(market.get("reason", "-"))

    # 2. 趨勢診斷
    st.markdown("#### 📈 趨勢診斷")
    trend = _trend_diagnosis(df)
    tc1, tc2, tc3 = st.columns([1.5, 1, 1])
    tc1.metric("均線排列", trend["alignment"])
    tc2.metric("距季線", fmt_pct(trend.get("dist_ma60")))
    tc3.metric("年內位階", fmt_pct(trend.get("year_pct")))
    st.caption(trend["detail"])

    # 3. 訊號強度
    st.markdown("#### 🎯 訊號強度")
    sig = _signal_strength(row_data)
    grade_color = {"A1": "🔴", "A2": "🟡"}.get(sig["grade"], "⚪")
    sc1, sc2 = st.columns([1, 2])
    sc1.metric(f"{grade_color} 分級", sig["grade"])
    sc2.metric("加分", f"{sig['score']} 分")

    if sig["grade"] == "A1":
        st.markdown("**A1 突破型核心條件:**")
        cols = st.columns(7)
        for i, (name, val) in enumerate(sig["a1_conds"]):
            with cols[i]:
                mark = "✅" if val else "❌"
                st.markdown(f"{mark}<br>{name}", unsafe_allow_html=True)
    elif sig["grade"] == "A2":
        st.markdown("**A2 回踩型核心條件:**")
        cols = st.columns(6)
        for i, (name, val) in enumerate(sig["a2_conds"]):
            with cols[i]:
                mark = "✅" if val else "❌"
                st.markdown(f"{mark}<br>{name}", unsafe_allow_html=True)

    if sig["reasons"]:
        with st.expander("查看完整條件明細"):
            st.write(sig["reasons"])

    # 4. 量價分析
    st.markdown("#### 💹 量價分析")
    vol = _volume_analysis(df)
    vc1, vc2 = st.columns([1, 3])
    vc1.metric("狀態", vol["status"])
    vc2.info(vol["detail"])

    # 5. 進場策略
    st.markdown("#### 🎯 進場策略")
    entry = _entry_strategy(df, row_data, trend)
    st.caption(f"建議倉位:**{entry.get('position', '-')}**")

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown(f"**🔴 積極型**")
        st.metric("買點", fmt_num(entry.get("aggressive", {}).get("price"), 2))
        st.caption(entry.get("aggressive", {}).get("desc", ""))
    with ec2:
        st.markdown(f"**🟡 穩健型**")
        st.metric("買點", fmt_num(entry.get("moderate", {}).get("price"), 2))
        st.caption(entry.get("moderate", {}).get("desc", ""))
    with ec3:
        st.markdown(f"**🟢 保守型**")
        st.metric("買點", fmt_num(entry.get("conservative", {}).get("price"), 2))
        st.caption(entry.get("conservative", {}).get("desc", ""))

    # 6. 停損停利
    st.markdown("#### 🛡️ 停損停利")
    targets = _stop_loss_targets(df, row_data, entry)
    tgc1, tgc2, tgc3, tgc4, tgc5 = st.columns(5)
    tgc1.metric("短線停損", fmt_num(targets.get("stop_short"), 2))
    tgc2.metric("波段停損", fmt_num(targets.get("stop_wave"), 2))
    tgc3.metric("嚴格停損", fmt_num(targets.get("stop_hard"), 2))
    tgc4.metric("第一目標", fmt_num(targets.get("target_1"), 2))
    tgc5.metric("第二目標", fmt_num(targets.get("target_2"), 2))
    st.caption(f"風報比(短停損 vs 第一目標):**{fmt_num(targets.get('rr'), 2)}** — "
               f"{'✅ 佳' if targets.get('rr', 0) >= 2 else '⚠️ 偏低' if targets.get('rr', 0) < 1.5 else '可接受'}")

    # 7. 風險警示
    st.markdown("#### ⚠️ 風險警示")
    warnings_list = _risk_warnings(df, row_data, trend, market)
    for w in warnings_list:
        if "🔴" in w:
            st.error(w)
        elif "🟡" in w:
            st.warning(w)
        else:
            st.success(w)

    return {
        "prices": {
            "aggressive":   safe_float(entry.get("aggressive", {}).get("price")),
            "moderate":     safe_float(entry.get("moderate", {}).get("price")),
            "conservative": safe_float(entry.get("conservative", {}).get("price")),
            "target_1":     safe_float(targets.get("target_1")),
            "target_2":     safe_float(targets.get("target_2")),
            "stop_short":   safe_float(targets.get("stop_short")),
            "stop_wave":    safe_float(targets.get("stop_wave")),
        },
    }

# =========================================================
# 基本面渲染
# =========================================================

def render_fundamental(stock_id: str, market_type: Optional[str]):
    """基本面:殖利率 / PE / PB / ROE / 股利"""
    token = get_finmind_token()
    try:
        fund = load_fundamental(stock_id, market_type, token if token else None)
    except Exception as e:
        st.warning(f"基本面資料讀取失敗: {e}")
        return

    if not isinstance(fund, dict):
        st.info("查無基本面資料")
        return

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    fc1.metric("殖利率 %",  fmt_num(fund.get("yield"), 2))
    fc2.metric("現金股利",   fmt_num(fund.get("dividend"), 2))
    fc3.metric("PE",         fmt_num(fund.get("pe"), 2))
    fc4.metric("PB",         fmt_num(fund.get("pb"), 2))
    fc5.metric("ROE %",      fmt_num(fund.get("roe"), 2))

    # 估值評語
    comments = []
    pe = safe_float(fund.get("pe"), 0)
    pb = safe_float(fund.get("pb"), 0)
    dy = safe_float(fund.get("yield"), 0)
    roe = safe_float(fund.get("roe"), 0)

    if 0 < pe <= 15: comments.append("🟢 PE 偏低,估值合理")
    elif 15 < pe <= 25: comments.append("🟡 PE 中性")
    elif pe > 25: comments.append("🔴 PE 偏高,估值拉高")

    if 0 < pb <= 1.5: comments.append("🟢 PB 偏低")
    elif pb > 3: comments.append("🟡 PB 偏高")

    if dy >= 5: comments.append("🟢 高殖利率(≥5%)")
    elif dy >= 3: comments.append("🟡 中等殖利率")

    if roe >= 15: comments.append("🟢 ROE 優秀(≥15%)")
    elif roe >= 8: comments.append("🟡 ROE 合格")
    elif 0 < roe < 8: comments.append("🔴 ROE 偏低")

    if comments:
        st.markdown("**估值評語:**")
        for c in comments:
            st.caption(c)

    note = fund.get("source_note", "")
    if note:
        st.caption(f"資料來源:{note}")

# =========================================================
# Session state
# =========================================================

defaults = {
    "enable_auto_refresh": False, "refresh_sec": 60,
    "top_n": 50, "grade_filter": ["A1", "A2"], "chart_period": "9mo",
    "min_price": 0.0, "max_price": 10000.0,
    "min_volume": 500, "max_volume": 0, "min_vol_ratio": 1.0,
    "selected_stock_option": None,
    "page_mode": "最新結果", "pending_page_mode": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

VALID_PAGES = ["最新結果", "單筆個股分析", "掃描個股圖表", "AI學習", "歷史紀錄"]
if st.session_state.get("refresh_sec") not in [30, 60, 120, 300]:
    st.session_state["refresh_sec"] = 60
if st.session_state.get("top_n") not in [20, 50, 100]:
    st.session_state["top_n"] = 50
if st.session_state.get("chart_period") not in ["3mo", "6mo", "9mo", "12mo"]:
    st.session_state["chart_period"] = "9mo"
if not isinstance(st.session_state.get("grade_filter"), list) or \
   not all(g in ["A1", "A2"] for g in st.session_state.get("grade_filter", [])):
    st.session_state["grade_filter"] = ["A1", "A2"]
if st.session_state.get("page_mode") not in VALID_PAGES:
    st.session_state["page_mode"] = "最新結果"

# =========================================================
# UI 輔助
# =========================================================

def badge(grade):
    return {"A1": "🔴 A1 突破", "A2": "🟡 A2 回踩"}.get(grade, grade)

def rank_df(df):
    if df.empty or "grade" not in df.columns:
        return df.copy()
    out = df.copy()
    out["grade_rank"] = out["grade"].map({"A1": 0, "A2": 1}).fillna(9)
    return out.sort_values(
        ["grade_rank", "score", "vol_ratio"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

def format_df(df):
    if df.empty: return df
    cols = ["grade", "code", "name", "type", "close", "score",
            "volume", "vol_ratio", "rsi14",
            "aggressive_buy_price", "pullback_buy_price",
            "sell_price_1", "stop_loss_short", "scan_time"]
    return df[[c for c in cols if c in df.columns]].copy()

def set_stock(code, df_show):
    if df_show.empty: return
    m = df_show[df_show["code"].astype(str) == str(code)]
    if m.empty: return
    r = m.iloc[0]
    st.session_state.selected_stock_option = \
        f"{r['code']} {r['name']} ({r['grade']})"
    st.session_state.pending_page_mode = "掃描個股圖表"

def learning_panel(learning):
    st.subheader("AI 自我學習狀態")
    stats = learning.get("rule_stats", {})
    w = learning.get("weights", {})
    t = learning.get("thresholds", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("已標記樣本數", stats.get("total_labeled", 0))
    c2.metric("整體 5 日成功率",
              f"{stats.get('success_rate_5d', 0):.1f}%")
    c3.metric("模型更新時間", learning.get("updated_at", "") or "N/A")

    gs = stats.get("grade_stats", {})
    if gs:
        st.markdown("#### A1 / A2 歷史績效")
        st.dataframe(pd.DataFrame([{
            "分級": badge(g),
            "樣本數": d.get("samples", 0),
            "5 日勝率 %": f"{d.get('success_rate_5d', 0):.1f}%",
            "平均 5 日報酬 %": f"{d.get('avg_ret_5d', 0):.2f}%",
        } for g, d in gs.items()]), hide_index=True)

    with st.expander("查看學習細節", expanded=False):
        if w:
            st.dataframe(pd.DataFrame([
                {"條件": k, "權重": v} for k, v in w.items()
            ]), hide_index=True)
        if t:
            st.dataframe(pd.DataFrame([
                {"參數": k, "值": v} for k, v in t.items()
            ]), hide_index=True)
        bc = stats.get("by_condition", {})
        if bc:
            st.dataframe(pd.DataFrame([
                {"條件": k, "樣本": v["samples"],
                 "5 日勝率 %": v["success_rate_5d"],
                 "平均報酬 %": v["avg_ret_5d"]}
                for k, v in bc.items()
            ]), hide_index=True)

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("操作面板")
st.sidebar.checkbox("啟用自動刷新", key="enable_auto_refresh")
st.sidebar.selectbox("自動刷新秒數", [30, 60, 120, 300],
                    key="refresh_sec", format_func=lambda x: f"{x} 秒")
st.sidebar.selectbox("顯示筆數", [20, 50, 100], key="top_n")
st.sidebar.multiselect("分級篩選", ["A1", "A2"],
                      default=["A1", "A2"], key="grade_filter")
st.sidebar.selectbox("圖表期間",
                    ["3mo", "6mo", "9mo", "12mo"], key="chart_period")

st.sidebar.markdown("### 價格設定")
st.sidebar.number_input("最低價格", min_value=0.0, step=1.0, key="min_price")
st.sidebar.number_input("最高價格", min_value=0.0, step=10.0, key="max_price")

st.sidebar.markdown("### 成交量設定")
st.sidebar.number_input("最低成交量(張)",
                       min_value=0, step=100, key="min_volume",
                       help="0 = 不限")
st.sidebar.number_input("最高成交量(0 = 不限)",
                       min_value=0, step=100, key="max_volume")

st.sidebar.markdown("### 量比設定")
st.sidebar.number_input("最低量比",
                       min_value=0.0, step=0.1, key="min_vol_ratio")

if st.sidebar.button("🔄 手動重新掃描", use_container_width=True):
    with st.spinner("正在執行掃描,請稍候..."):
        rc, out, err = run_manual_scan()
        st.cache_data.clear()
        if rc == 0:
            st.sidebar.success("掃描完成")
            st.rerun()
        else:
            st.sidebar.error("掃描失敗")
            st.sidebar.code(err if err else out)

# =========================================================
# 主畫面
# =========================================================

st.title("📈 台股 AI 自動掃描")

latest   = load_latest()
learning = load_learning()
hist_df  = load_history()
market   = get_market_regime()

if st.session_state.enable_auto_refresh:
    auto_refresh_block(st.session_state.refresh_sec)

# ── 頂部狀態列 ────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.caption(f"最後掃描時間:{latest.get('updated_at', 'N/A')}")
    status = latest.get("status", "empty")
    msg = latest.get("message", "")
    summ = latest.get("summary", {})
    if status == "ok" and (summ.get("A1_count", 0) + summ.get("A2_count", 0)) > 0:
        st.success("✅ 已載入最新掃描結果")
    elif status == "ok":
        st.info(msg or "掃描完成,今日無符合 A1/A2 條件的個股。")
    elif status == "empty":
        st.warning(msg or "尚無資料")
    else:
        st.error(msg or "讀取失敗")
with right:
    # 大盤環境標籤
    regime_color = {"bull": "🟢", "bear": "🔴", "neutral": "🟡"}.get(
        market.get("status"), "⚪")
    st.info(f"{regime_color} 大盤:{market.get('regime', '未知')}")

# ── 指標卡 ────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("總掃描池",    summ.get("pool_total", 0))
m2.metric("成功分析數",  summ.get("success_count", 0))
m3.metric("缺資料/失敗", summ.get("failed_count", 0))
m4.metric("A1 突破型",   summ.get("A1_count", 0))
m5.metric("A2 回踩型",   summ.get("A2_count", 0))

# ── 篩選 ──────────────────────────────────────────────
df = rank_df(pd.DataFrame(latest.get("results", [])))
if not df.empty and "grade" in df.columns:
    df = df[df["grade"].isin(st.session_state.grade_filter)]
if not df.empty:
    if "close" in df.columns:
        df = df[(df["close"] >= st.session_state.min_price) &
                (df["close"] <= st.session_state.max_price)]
    if "volume" in df.columns:
        df = df[df["volume"] >= st.session_state.min_volume * 1000]  # 張→股
    if st.session_state.max_volume > 0 and "volume" in df.columns:
        df = df[df["volume"] <= st.session_state.max_volume * 1000]
    if "vol_ratio" in df.columns:
        df = df[df["vol_ratio"] >= st.session_state.min_vol_ratio]

df_show = df.head(st.session_state.top_n).copy() if not df.empty else pd.DataFrame()

# ── 頁面切換 ──────────────────────────────────────────
if st.session_state.pending_page_mode in VALID_PAGES:
    st.session_state.page_mode = st.session_state.pending_page_mode
    st.session_state.pending_page_mode = None

page_mode = st.radio("功能選單", VALID_PAGES, horizontal=True, key="page_mode")

# ═════════════════════════════════════════════════════════
# 最新結果
# ═════════════════════════════════════════════════════════

if page_mode == "最新結果":
    st.subheader("最新掃描結果")
    if df_show.empty:
        st.info("目前沒有符合條件的 A 級結果。")
    else:
        st.dataframe(
            format_df(df_show.drop(columns=["grade_rank"], errors="ignore")),
            use_container_width=True, hide_index=True,
        )
        st.markdown("### Top 10(點「查看圖表」進入深度分析)")
        for i, row in df_show.head(10).iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.8, 1.0, 2.0, 1.5, 1.3, 1.3, 1.2])
            c1.write(f"**{badge(row.get('grade', '-'))}**")
            c2.write(f"**{row.get('code', '')}**")
            c3.write(row.get("name", ""))
            c4.write(f"收盤:{safe_float(row.get('close', 0)):.2f}")
            c5.write(f"分數:{safe_int(row.get('score', 0))}")
            c6.write(f"量比:{safe_float(row.get('vol_ratio', 0)):.2f}")
            if c7.button("查看圖表", key=f"t10_{row.get('code', i)}"):
                set_stock(str(row.get("code", "")), df_show)
                st.rerun()

    failed = latest.get("failed_symbols", [])
    if failed:
        with st.expander("查看部分抓不到資料的代號", expanded=False):
            st.dataframe(pd.DataFrame(failed),
                        use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════
# 單筆個股分析
# ═════════════════════════════════════════════════════════

elif page_mode == "單筆個股分析":
    st.subheader("單筆個股分析")
    ci, cb = st.columns([3, 1])
    with ci:
        code_input = st.text_input(
            "代號",
            placeholder="例如:2330、6761",
            label_visibility="collapsed",
        )
    with cb:
        do_go = st.button("🔍 深度分析", use_container_width=True)

    code = code_input.strip()
    if not code:
        st.info("請輸入股票代號後按「深度分析」按鈕。")
    elif do_go or code:
        with st.spinner(f"正在分析 {code} ..."):
            sym, sdf = cached_fetch_by_code(code, st.session_state.chart_period)

        if sdf.empty or not sym:
            st.warning("找不到此股票資料,請確認代號。")
        else:
            # 取股票名
            try:
                import twstock as _tw
                _i = _tw.codes.get(code, None)
                name = getattr(_i, "name", code) if _i else code
                market_type = "twse" if sym.endswith(".TW") else "tpex"
            except Exception:
                name = code
                market_type = None

            curr = sdf.iloc[-1]

            # 頂部摘要
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("股票", f"{name}")
            c2.metric("代號", code)
            c3.metric("收盤", fmt_num(curr["Close"], 2))
            c4.metric("成交量", f"{safe_int(curr['Volume']):,}")
            c5.metric("RSI14", fmt_num(curr.get("RSI"), 1))

            # 構造一個類似 latest_scan 格式的 row_data
            row_data = {
                "code": code, "name": name, "symbol": sym,
                "grade": "-", "score": 0, "reasons": "",
                "close": safe_float(curr["Close"]),
                "vol_ratio": safe_float(curr.get("VOL_RATIO")),
            }

            # 分頁:技術 / 基本 / 圖表
            tab1, tab2, tab3 = st.tabs(["📊 技術分析", "💰 基本面", "📈 專業圖表"])

            with tab1:
                render_result = render_technical_analysis(sdf, row_data, market)

            with tab2:
                render_fundamental(code, market_type)

            with tab3:
                prices = render_result["prices"] if render_result else {}
                st.plotly_chart(
                    make_pro_chart(sdf, prices, f"{code} {name}",
                                   period_bars=120),
                    use_container_width=True,
                )

# ═════════════════════════════════════════════════════════
# 掃描個股圖表
# ═════════════════════════════════════════════════════════

elif page_mode == "掃描個股圖表":
    st.subheader("掃描個股深度分析")
    if df_show.empty:
        st.info("目前沒有可選股票,請放寬篩選條件或等待下次掃描。")
    else:
        opts = [f"{r['code']} {r['name']} ({r['grade']})"
                for _, r in df_show.iterrows()]
        if st.session_state.selected_stock_option not in opts:
            st.session_state.selected_stock_option = opts[0]

        sel = st.selectbox("選擇掃描結果中的股票", opts,
                           key="selected_stock_option")
        row = df_show.iloc[opts.index(sel)]
        row_data = row.to_dict()

        # 頂部摘要
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("分級", badge(row.get("grade", "-")))
        c2.metric("收盤", fmt_num(row.get("close"), 2))
        c3.metric("加分", safe_int(row.get("score")))
        c4.metric("量比", fmt_num(row.get("vol_ratio"), 2))
        c5.metric("RSI14", fmt_num(row.get("rsi14"), 1))
        c6.metric("成交量", f"{safe_int(row.get('volume')):,}")

        with st.spinner("正在抓取圖表資料..."):
            cdf = cached_fetch_chart(row.get("symbol", ""),
                                     st.session_state.chart_period)

        if cdf.empty:
            st.warning("無法讀取圖表資料(Yahoo rate limit 或網路問題,請稍後再試)")
            st.info("仍可查看掃描當下計算出的基本訊號(下方):")
            st.write(f"**觸發條件:** {row.get('reasons', '-')}")
        else:
            market_type = "twse" if row.get("market") == "上市" else "tpex"

            tab1, tab2, tab3 = st.tabs(["📊 技術分析", "💰 基本面", "📈 專業圖表"])

            with tab1:
                render_result = render_technical_analysis(cdf, row_data, market)

            with tab2:
                render_fundamental(str(row.get("code", "")), market_type)

            with tab3:
                prices = render_result["prices"] if render_result else {}
                st.plotly_chart(
                    make_pro_chart(
                        cdf, prices,
                        f"{row['code']} {row.get('name', '')}",
                        period_bars=120,
                    ),
                    use_container_width=True,
                )

# ═════════════════════════════════════════════════════════
# AI 學習
# ═════════════════════════════════════════════════════════

elif page_mode == "AI學習":
    learning_panel(learning)

# ═════════════════════════════════════════════════════════
# 歷史紀錄
# ═════════════════════════════════════════════════════════

elif page_mode == "歷史紀錄":
    st.subheader("歷史紀錄")
    if hist_df.empty:
        st.info("目前尚無歷史紀錄。")
    else:
        h = hist_df.copy()
        if "grade" in h.columns:
            h = h[h["grade"].isin(st.session_state.grade_filter)]
        if "close" in h.columns:
            h = h[(h["close"] >= st.session_state.min_price) &
                  (h["close"] <= st.session_state.max_price)]
        if "volume" in h.columns:
            h = h[h["volume"] >= st.session_state.min_volume * 1000]
        if st.session_state.max_volume > 0 and "volume" in h.columns:
            h = h[h["volume"] <= st.session_state.max_volume * 1000]
        if "vol_ratio" in h.columns:
            h = h[h["vol_ratio"] >= st.session_state.min_vol_ratio]
        hc = ["scan_date", "code", "name", "grade", "type", "close",
              "volume", "vol_ratio", "score",
              "aggressive_buy_price", "pullback_buy_price",
              "sell_price_1", "stop_loss_short",
              "ret_3d", "ret_5d", "ret_10d", "success_5d"]
        st.dataframe(
            h[[c for c in hc if c in h.columns]].tail(500),
            use_container_width=True, hide_index=True,
        )
