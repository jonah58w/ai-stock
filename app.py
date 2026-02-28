# app.py
# AI 台股量化專業平台（無 Plotly / 全功能保留 / V23.2+ 修正版）
# ✅ 單一股票分析 + Top10 掃描器
# ✅ 多來源備援：FinMind(首選) → yfinance → Stooq → CSV上傳
# ✅ 逐路診斷（哪一路失敗、為什麼）
# ✅ 指標共振：MA/RSI/MACD/KD/布林/ATR/量能
# ✅ 布林通道 + 均線 + 訊號標記（Matplotlib）
# ✅ Retry + 缺套件提示（側邊欄）
#
# 注意：
# - Streamlit Cloud 若缺套件：requirements.txt 請加上 FinMind / yfinance / pandas_datareader / matplotlib
# - FinMind 需要 token：Secrets 設 FINMIND_TOKEN="xxxxx"
#
# 免責聲明：本工具僅做資訊顯示與風控演算，不構成投資建議，不會自動下單。

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Dependency checks (no plotly)
# -----------------------------
DEPENDENCIES: Dict[str, bool] = {
    "requests": True,
    "matplotlib": True,
    "finmind": False,
    "yfinance": False,
    "pandas_datareader": False,
}

# FinMind
try:
    from FinMind.data import DataLoader  # type: ignore
    DEPENDENCIES["finmind"] = True
except Exception:
    DataLoader = None  # type: ignore

# yfinance
try:
    import yfinance as yf  # type: ignore
    DEPENDENCIES["yfinance"] = True
except Exception:
    yf = None  # type: ignore

# pandas_datareader (optional fallback)
try:
    from pandas_datareader import data as pdr  # type: ignore
    DEPENDENCIES["pandas_datareader"] = True
except Exception:
    pdr = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def _as_str_ticker(x) -> str:
    """防止 tuple/list/set 被誤傳進來造成 yfinance .lower() 爆炸。"""
    if isinstance(x, (tuple, list, set)):
        x = list(x)[0] if len(x) else ""
    x = "" if x is None else str(x)
    return x.strip()


def _retry(fn, times: int = 3, sleep: float = 0.7):
    last = None
    for i in range(times):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep * (i + 1))
    raise last


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _human_pct(p: float) -> str:
    if p is None or np.isnan(p):
        return "N/A"
    return f"{p*100:.1f}%"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: Date, Open, High, Low, Close, Volume
    Date => datetime; sort asc; drop duplicates.
    """
    df = df.copy()

    # Find date col
    if "Date" not in df.columns:
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "time"):
                df = df.rename(columns={c: "Date"})
                break

    # Standardize OHLCV column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("open", "o"):
            rename_map[c] = "Open"
        elif cl in ("high", "h", "max"):
            rename_map[c] = "High"
        elif cl in ("low", "l", "min"):
            rename_map[c] = "Low"
        elif cl in ("close", "c", "adj close", "adjclose"):
            rename_map[c] = "Close"
        elif cl in ("volume", "vol", "trading_volume"):
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    # Keep only required columns if exist
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Numeric conversions
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # Fill Volume if missing
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    return df


# -----------------------------
# Fetch attempts (diagnosis)
# -----------------------------
@dataclass
class FetchAttempt:
    source: str
    result: str
    url: str = ""
    status: str = ""
    snippet: str = ""
    note: str = ""


# -----------------------------
# Data sources
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_finmind(code: str, months_back: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    code = _as_str_ticker(code)
    if not DEPENDENCIES.get("finmind", False) or DataLoader is None:
        return None, FetchAttempt("FinMind", "NO_MODULE", note="缺少套件：pip install FinMind")

    # token from secrets or manual input
    token = None
    try:
        token = st.secrets.get("FINMIND_TOKEN", None)
    except Exception:
        token = None
    token = token or st.session_state.get("FINMIND_TOKEN_INPUT", "")

    if not token:
        return None, FetchAttempt("FinMind", "NO_TOKEN", note="請在 Secrets 設定 FINMIND_TOKEN 或側邊欄手動輸入")

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months_back * 35 + 40)).strftime("%Y-%m-%d")

        dl = DataLoader()
        dl.login_by_token(api_token=token)

        def _do():
            return dl.taiwan_stock_daily(stock_id=code, start_date=start, end_date=end)

        df = _retry(_do, times=3, sleep=0.8)

        if df is None or len(df) == 0:
            return None, FetchAttempt("FinMind", "EMPTY", note="FinMind 回傳空資料")

        # rename based on FinMind daily schema
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
        })

        df = _normalize_ohlcv(df)
        if len(df) < 10:
            return None, FetchAttempt("FinMind", "TOO_SHORT", note=f"rows={len(df)}")

        return df, FetchAttempt("FinMind", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("FinMind", "EXC", note=str(e))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    code = _as_str_ticker(code)
    if not DEPENDENCIES.get("yfinance", False) or yf is None:
        return None, FetchAttempt("YF", "NO_MODULE", note="缺少套件：pip install yfinance")

    tickers: List[str]
    if re.fullmatch(r"\d{4,6}", code):
        tickers = [f"{code}.TW", f"{code}.TWO"]
    else:
        tickers = [code]

    try:
        for tk in tickers:
            tk = _as_str_ticker(tk)
            if not tk:
                continue

            def _do():
                return yf.download(
                    tk,
                    period=f"{max(30, period_days)}d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )

            raw = _retry(_do, times=3, sleep=0.8)

            if raw is None or len(raw) == 0:
                continue

            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df = df.reset_index().rename(columns={"Date": "Date", "Datetime": "Date"})
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Close",
                "Volume": "Volume",
            })

            df = _normalize_ohlcv(df)

            if len(df) >= 10:
                return df, FetchAttempt("YF", "OK", note=f"{tk} rows={len(df)}")

        return None, FetchAttempt("YF", "EMPTY", note="yfinance 無資料（可能被擋或代號不對）")
    except Exception as e:
        return None, FetchAttempt("YF", "EXC", note=str(e))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    code = _as_str_ticker(code)

    try:
        if DEPENDENCIES.get("pandas_datareader", False) and pdr is not None:
            ticker = code.lower()

            def _do():
                return pdr.DataReader(ticker, "stooq")

            df = _retry(_do, times=3, sleep=0.8)
            if df is None or len(df) == 0:
                return None, FetchAttempt("STOOQ", "EMPTY", note="stooq 回傳空資料")

            df = df.reset_index().rename(columns={
                "Date": "Date",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })
            df = _normalize_ohlcv(df)
            if len(df) >= 10:
                return df, FetchAttempt("STOOQ", "OK", note=f"rows={len(df)}")
            return None, FetchAttempt("STOOQ", "TOO_SHORT", note=f"rows={len(df)}")

        return None, FetchAttempt("STOOQ", "NO_PDR", note="缺少 pandas_datareader 或 stooq 不支援該代號")
    except Exception as e:
        return None, FetchAttempt("STOOQ", "EXC", note=str(e))


def _load_csv_fallback(file) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    try:
        if file is None:
            return None, FetchAttempt("CSV_UPLOAD", "NO_FILE")

        df = pd.read_csv(file)
        df = _normalize_ohlcv(df)
        if df is None or len(df) < 10:
            return None, FetchAttempt("CSV_UPLOAD", "TOO_SHORT", note=f"rows={0 if df is None else len(df)}")
        return df, FetchAttempt("CSV_UPLOAD", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("CSV_UPLOAD", "EXC", note=str(e))


def fetch_ohlcv_multi(code: str, months_back: int, csv_upload=None) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:
    code = _as_str_ticker(code)
    attempts: List[FetchAttempt] = []
    period_days = int(max(2, months_back) * 31)

    # 2026 推薦順序：FinMind → yfinance → Stooq → CSV
    df, att = fetch_finmind(code, months_back)
    attempts.append(att)
    if df is not None and len(df) >= 30:
        return df, "FinMind", attempts

    df, att = fetch_yfinance(code, period_days)
    attempts.append(att)
    if df is not None and len(df) >= 30:
        return df, "YF", attempts

    df, att = fetch_stooq(code, period_days)
    attempts.append(att)
    if df is not None and len(df) >= 30:
        return df, "STOOQ", attempts

    if csv_upload is not None:
        df, att = _load_csv_fallback(csv_upload)
        attempts.append(att)
        if df is not None and len(df) >= 30:
            return df, "CSV_UPLOAD", attempts

    return None, "NONE", attempts


# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.bfill()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_line = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d_line = k_line.rolling(d).mean()
    return k_line.bfill(), d_line.bfill()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill()


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return mid, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]

    df["SMA20"] = sma(c, 20)
    df["EMA20"] = ema(c, 20)
    df["SMA60"] = sma(c, 60)
    df["RSI14"] = rsi(c, 14)

    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_SIG"] = macd_sig
    df["MACD_HIST"] = macd_hist

    k, d = stochastic_kd(df["High"], df["Low"], c, 14, 3)
    df["K"] = k
    df["D"] = d

    df["ATR14"] = atr(df["High"], df["Low"], c, 14)

    mid, up, lo = bollinger(c, 20, 2.0)
    df["BB_MID"] = mid
    df["BB_UP"] = up
    df["BB_LO"] = lo

    # Volume signals
    vol = df["Volume"].fillna(0)
    df["VOL_SMA20"] = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["VOL_Z20"] = ((vol - df["VOL_SMA20"]) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# -----------------------------
# Scoring & Signals
# -----------------------------
def resonance_score(latest: pd.Series) -> Tuple[int, Dict[str, Any]]:
    """0-100 共振分數（偏可操作 / 偏順勢）"""
    score = 50.0
    details: Dict[str, Any] = {}

    close = _safe_float(latest.get("Close"))
    sma20 = _safe_float(latest.get("SMA20"))
    ema20 = _safe_float(latest.get("EMA20"))
    sma60 = _safe_float(latest.get("SMA60"))
    rsi14 = _safe_float(latest.get("RSI14"))
    macd_h = _safe_float(latest.get("MACD_HIST"))
    k = _safe_float(latest.get("K"))
    d = _safe_float(latest.get("D"))
    bb_up = _safe_float(latest.get("BB_UP"))
    bb_lo = _safe_float(latest.get("BB_LO"))
    volz = _safe_float(latest.get("VOL_Z20"))

    # Trend
    if not np.isnan(sma20) and not np.isnan(sma60):
        if sma20 > sma60:
            score += 8
            details["trend"] = "up"
        elif sma20 < sma60:
            score -= 6
            details["trend"] = "down"
        else:
            details["trend"] = "flat"

    # Price vs EMA/SMA
    if not np.isnan(ema20) and close > ema20:
        score += 6
    elif not np.isnan(ema20) and close < ema20:
        score -= 4

    if not np.isnan(sma20) and close > sma20:
        score += 4
    elif not np.isnan(sma20) and close < sma20:
        score -= 3

    # RSI
    if not np.isnan(rsi14):
        if rsi14 >= 70:
            score -= 4
            details["rsi_zone"] = "overbought"
        elif rsi14 <= 30:
            score += 6
            details["rsi_zone"] = "oversold"
        else:
            details["rsi_zone"] = "neutral"

    # MACD
    if not np.isnan(macd_h):
        score += 6 if macd_h > 0 else -4

    # KD
    if not np.isnan(k) and not np.isnan(d):
        score += 3 if k > d else -2

    # Bollinger position
    if (not np.isnan(bb_up)) and (not np.isnan(bb_lo)) and (bb_up > bb_lo):
        pos = (close - bb_lo) / (bb_up - bb_lo)
        details["bb_pos"] = float(pos)
        if pos < 0.15:
            score += 6
        elif pos > 0.85:
            score -= 4

    # Volume
    if not np.isnan(volz):
        if volz >= 1.2:
            score += 6
            details["volume"] = "impulse"
        elif volz <= -0.8:
            score -= 1
            details["volume"] = "quiet"
        else:
            details["volume"] = "normal"

    score = float(np.clip(score, 0, 100))
    return int(round(score)), details


def determine_action(df: pd.DataFrame, strategy: str, params: Dict[str, float]) -> Tuple[str, str]:
    """Return (action, reason): BUY / SELL / WATCH"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    close = float(latest["Close"])
    ema20 = float(latest.get("EMA20", np.nan))
    sma20 = float(latest.get("SMA20", np.nan))
    sma60 = float(latest.get("SMA60", np.nan))
    rsi14 = float(latest.get("RSI14", np.nan))
    macd_h = float(latest.get("MACD_HIST", np.nan))
    k = float(latest.get("K", np.nan))
    d = float(latest.get("D", np.nan))
    bb_up = float(latest.get("BB_UP", np.nan))
    bb_lo = float(latest.get("BB_LO", np.nan))
    atr14 = float(latest.get("ATR14", np.nan))
    volz = float(latest.get("VOL_Z20", 0))

    trend_up = (not np.isnan(sma20) and not np.isnan(sma60) and sma20 > sma60)
    above_ema = (not np.isnan(ema20) and close > ema20)
    macd_pos = (not np.isnan(macd_h) and macd_h > 0)
    kd_bull = (not np.isnan(k) and not np.isnan(d) and k > d)

    if strategy == "pullback":
        near_bb_lo = (not np.isnan(bb_lo) and close <= bb_lo * (1 + 0.01))
        near_ema = (not np.isnan(ema20) and abs((close - ema20) / ema20) <= 0.02)
        oversold = (not np.isnan(rsi14) and rsi14 <= 35)
        turning = (macd_h > float(prev.get("MACD_HIST", macd_h))) or kd_bull

        if (near_bb_lo or near_ema) and (oversold or turning):
            return "BUY", "回檔接近支撐（布林下緣/EMA20）且出現指標翻揚跡象"

        near_bb_up = (not np.isnan(bb_up) and close >= bb_up * (1 - 0.005))
        overbought = (not np.isnan(rsi14) and rsi14 >= 70)
        weakening = (macd_h < float(prev.get("MACD_HIST", macd_h))) or (not kd_bull)
        if near_bb_up and (overbought or weakening):
            return "SELL", "接近壓力（布林上緣）且指標偏過熱/轉弱"

        return "WATCH", "條件尚未明確，等待價格進入區間或指標翻轉"

    # breakout
    max_chase = float(params.get("max_chase_distance", 0.06))
    atr_buffer = float(params.get("breakout_atr_buffer", 0.20))

    lookback = 20
    if len(df) < lookback + 2:
        return "WATCH", "資料筆數不足以判定突破條件"

    prev_high = df["High"].rolling(lookback).max().iloc[-2]
    trigger = prev_high + (atr_buffer * atr14 if not np.isnan(atr14) else 0)
    dist = (close - trigger) / trigger if trigger else 0

    confirm = trend_up and above_ema and macd_pos and (volz >= 0.6)

    if close >= trigger and dist <= max_chase and confirm:
        return "BUY", f"突破成立（站上近{lookback}日高點+ATR緩衝）且量價/趨勢/動能同步"

    if close < (ema20 if not np.isnan(ema20) else close) and close < trigger:
        return "SELL", "突破失效（跌回EMA20且低於突破位），偏向風控撤退"

    return "WATCH", "突破型：尚未滿足『突破+確認+追價距離』三條件"


def estimate_zones(df: pd.DataFrame, max_buy_distance: float) -> Dict[str, Any]:
    """提供專業買賣區間（避免買點離現價太遠）"""
    latest = df.iloc[-1]
    close = float(latest["Close"])
    atr14 = float(latest.get("ATR14", np.nan))
    bb_lo = float(latest.get("BB_LO", np.nan))
    bb_up = float(latest.get("BB_UP", np.nan))
    bb_mid = float(latest.get("BB_MID", np.nan))
    ema20 = float(latest.get("EMA20", np.nan))

    sup = df["Low"].rolling(20).min().iloc[-1]
    res = df["High"].rolling(20).max().iloc[-1]

    if np.isnan(atr14):
        atr14 = max(0.001, close * 0.02)

    base_lo = min(
        bb_lo if not np.isnan(bb_lo) else close - 1.2 * atr14,
        ema20 if not np.isnan(ema20) else close - 1.0 * atr14,
        bb_mid if not np.isnan(bb_mid) else close - 0.8 * atr14,
    )
    base_hi = min(
        ema20 if not np.isnan(ema20) else close - 0.5 * atr14,
        bb_mid if not np.isnan(bb_mid) else close - 0.3 * atr14,
        close,
    )

    z1_lo = min(base_lo, base_hi)
    z1_hi = max(base_lo, base_hi)

    min_price_allowed = close * (1 - max_buy_distance)
    z1_lo = max(z1_lo, min_price_allowed)
    z1_hi = max(min(z1_hi, close), z1_lo)

    z2_center = float(sup)
    z2_lo = max(0.01, z2_center - 0.8 * atr14)
    z2_hi = z2_center + 0.5 * atr14

    tp_base = max(res * 0.98, (bb_up * 0.985) if not np.isnan(bb_up) else res * 0.98)
    z3_lo = tp_base
    z3_hi = tp_base + 0.8 * atr14

    def _dist(target: float) -> float:
        return (target - close) / close if close else np.nan

    d1 = _dist((z1_lo + z1_hi) / 2)
    d2 = _dist((z2_lo + z2_hi) / 2)
    d3 = _dist((z3_lo + z3_hi) / 2)

    return {
        "close": close,
        "action_buy_zone": (float(z1_lo), float(z1_hi), float(d1), _human_pct(d1)),
        "deep_buy_zone": (float(z2_lo), float(z2_hi), float(d2), _human_pct(d2)),
        "sell_zone": (float(z3_lo), float(z3_hi), float(d3), _human_pct(d3)),
        "support": float(sup),
        "resistance": float(res),
        "atr": float(atr14),
        "bb_lo": float(bb_lo) if not np.isnan(bb_lo) else np.nan,
        "bb_up": float(bb_up) if not np.isnan(bb_up) else np.nan,
    }


# -----------------------------
# Plot (Matplotlib)
# -----------------------------
def plot_bollinger(df: pd.DataFrame, title: str, marks: Optional[List[Tuple[pd.Timestamp, float, str]]] = None):
    x = df["Date"]
    c = df["Close"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, c, label="Close")

    for col, lab in [("BB_UP", "BB Upper"), ("BB_MID", "BB Mid"), ("BB_LO", "BB Lower"), ("SMA20", "SMA20"), ("EMA20", "EMA20")]:
        if col in df.columns:
            ax.plot(x, df[col], label=lab)

    if "BB_LO" in df.columns and "BB_UP" in df.columns:
        ax.fill_between(x, df["BB_LO"], df["BB_UP"], alpha=0.10)

    if marks:
        for d, p, lab in marks:
            ax.scatter([d], [p], s=60)
            ax.annotate(lab, (d, p), xytext=(5, 8), textcoords="offset points")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Top10 Scanner
# -----------------------------
DEFAULT_POOL_SMALL = [
    "2330", "2317", "2454", "2308", "2412", "2881", "2882", "2891", "2603", "2609",
    "3037", "3711", "2382", "2303", "1301", "1303", "1101", "1216", "3008", "3443",
    "4967", "8046", "6274", "6488", "6643", "3533", "6669"
]


def _unique_codes(codes: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in codes:
        c = _as_str_ticker(c)
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def scan_top10(codes: List[str], months_back: int, max_buy_distance: float, strategy: str, params: Dict[str, float]) -> pd.DataFrame:
    rows = []
    codes = _unique_codes(codes)

    for code in codes:
        df, src, _attempts = fetch_ohlcv_multi(code, months_back, csv_upload=None)
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)
        score, _details = resonance_score(df.iloc[-1])
        action, reason = determine_action(df, strategy, params)
        zones = estimate_zones(df, max_buy_distance)

        close = float(df.iloc[-1]["Close"])
        buy_lo, buy_hi, d1, _d1s = zones["action_buy_zone"]

        # professional distance label
        if np.isnan(d1):
            dist_label = "N/A"
        else:
            dist_label = f"{abs(d1)*100:.1f}% {'回檔空間' if d1 < 0 else '上行空間'}"

        rows.append({
            "股票": code,
            "來源": src,
            "AI分數": score,
            "當下判斷": action,
            "可操作買區": f"{buy_lo:.2f} ~ {buy_hi:.2f}",
            "相對現價偏離": dist_label,
            "摘要理由": reason,
        })

    if not rows:
        return pd.DataFrame(columns=["股票", "來源", "AI分數", "當下判斷", "可操作買區", "相對現價偏離", "摘要理由"])

    out = pd.DataFrame(rows)
    out["__rank"] = out["當下判斷"].map({"BUY": 0, "WATCH": 1, "SELL": 2}).fillna(3)
    out = out.sort_values(["__rank", "AI分數"], ascending=[True, False]).drop(columns="__rank")
    out = out.drop_duplicates(subset=["股票"], keep="first")
    return out.head(10).reset_index(drop=True)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI 台股量化專業平台（無 Plotly）", layout="wide")
st.title("🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）")
st.caption("多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + Top10掃描 + 交易計畫（不自動下單）")

with st.sidebar:
    st.subheader("設定")
    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)
    months_opt = st.selectbox("資料期間", ["3mo", "6mo", "12mo", "24mo"], index=1)
    months_back = {"3mo": 3, "6mo": 6, "12mo": 12, "24mo": 24}[months_opt]

    show_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.markdown("---")
    st.subheader("策略 / 參數")
    strategy_label = st.radio("策略", ["回檔等待型（回檔分批）", "趨勢突破型（突破追價進場）"], index=0)
    strategy = "pullback" if strategy_label.startswith("回檔") else "breakout"

    max_buy_distance = st.slider(
        "可操作買點最大距離（避免買點離現實太遠）",
        0.02, 0.20, 0.12, 0.01
    )

    params: Dict[str, float] = {}
    if strategy == "breakout":
        params["max_chase_distance"] = st.slider("最大可接受追價/偏離距離（%）", 0.01, 0.20, 0.06, 0.01)
        params["breakout_atr_buffer"] = st.slider("突破觸發 buffer（ATR 倍數）", 0.00, 1.00, 0.20, 0.05)

    st.markdown("---")
    st.subheader("⚙️ 系統狀態 / 缺套件提示")
    for pkg, ok in DEPENDENCIES.items():
        if ok:
            st.success(f"✅ {pkg}")
        else:
            st.error(f"❌ {pkg} 缺失（請安裝）")

    if DEPENDENCIES["finmind"]:
        finmind_token_input = st.text_input("FinMind Token（若 Secrets 沒設，可手動填）", type="password", value="")
        if finmind_token_input:
            st.session_state["FINMIND_TOKEN_INPUT"] = finmind_token_input.strip()
    else:
        st.info("建議安裝 FinMind：requirements.txt 加入 FinMind>=1.0.0；並在 Secrets 設 FINMIND_TOKEN")

    st.markdown("---")
    st.subheader("🧪 網路測試（建議先按一次）")
    if st.button("網路測試（建議先按一次）"):
        test_rows = []
        try:
            r = _retry(lambda: requests.get("https://www.google.com", timeout=10), times=2, sleep=0.5)
            test_rows.append({"target": "google.com", "status": r.status_code, "ok": r.status_code == 200})
        except Exception as e:
            test_rows.append({"target": "google.com", "status": "ERR", "ok": False, "note": str(e)})

        if DEPENDENCIES["yfinance"] and yf is not None:
            try:
                r2 = _retry(lambda: yf.download("2330.TW", period="5d", progress=False), times=2, sleep=0.6)
                ok2 = r2 is not None and len(r2) > 0
                test_rows.append({"target": "yfinance 2330.TW", "status": "OK" if ok2 else "EMPTY", "ok": ok2})
            except Exception as e:
                test_rows.append({"target": "yfinance 2330.TW", "status": "ERR", "ok": False, "note": str(e)})

        st.dataframe(pd.DataFrame(test_rows), use_container_width=True)

# Main input + CSV upload
colA, colB = st.columns([1.1, 1.4], gap="large")
with colA:
    code = st.text_input("請輸入股票代號", value="6274", help="可輸入 2330 / 2317 / 6274 等")
    code = _as_str_ticker(code)

    st.caption("（選用）上傳 export.csv 作為備援資料源（Date/Open/High/Low/Close/Volume）")
    csv_upload = st.file_uploader("上傳 export.csv 作為備援資料源", type=["csv"])

with colB:
    st.info(
        "💡 若 Cloud 偶發抓不到資料：\n"
        "- 先按左側「網路測試」\n"
        "- 或直接上傳 export.csv（Date/Open/High/Low/Close/Volume）即可完整分析"
    )

# -----------------------------
# Mode: Single stock analysis
# -----------------------------
if mode == "單一股票分析":
    if not code:
        st.warning("請輸入股票代號")
        st.stop()

    df, src, attempts = fetch_ohlcv_multi(code, months_back, csv_upload=csv_upload)

    if df is None:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試或改用 CSV 上傳備援。")
        if show_debug:
            st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
            st.dataframe(pd.DataFrame([a.__dict__ for a in attempts]), use_container_width=True)
        st.stop()

    df = add_indicators(df)
    latest = df.iloc[-1]
    score, score_details = resonance_score(latest)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{float(latest['Close']):.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", src)
    c4.metric("最後日期 / 筆數", f"{latest['Date'].date()} / {len(df)}")

    action, reason = determine_action(df, strategy, params)
    st.subheader("📌 當下是否為買點/賣點？（可操作判斷）")
    if action == "BUY":
        st.success(f"✅ BUY：{reason}")
    elif action == "SELL":
        st.warning(f"⚠️ SELL：{reason}")
    else:
        st.info(f"⏳ WATCH：{reason}")

    zones = estimate_zones(df, max_buy_distance=max_buy_distance)
    buy_lo, buy_hi, d1, _ = zones["action_buy_zone"]
    deep_lo, deep_hi, d2, _ = zones["deep_buy_zone"]
    sell_lo, sell_hi, d3, _ = zones["sell_zone"]

    def _pro_dist_text(d: float) -> str:
        if np.isnan(d):
            return "N/A"
        if d < 0:
            return f"需回檔約 {abs(d)*100:.1f}% 才進入區間（Pullback distance）"
        return f"距離區間約 {d*100:.1f}%（Upside to zone）"

    st.subheader("🗺️ 未來預估買賣點（區間 + 專業偏離說法）")
    st.success(f"🟢 近端買點（可操作）：{buy_lo:.2f} ~ {buy_hi:.2f} ｜ {_pro_dist_text(d1)}")
    st.info(f"🔵 深回檔買點（等待型）：{deep_lo:.2f} ~ {deep_hi:.2f} ｜ {_pro_dist_text(d2)}")
    st.warning(f"🟡 近端賣點/壓力：{sell_lo:.2f} ~ {sell_hi:.2f} ｜ {_pro_dist_text(d3)}")

    st.subheader("📊 指標共振摘要")
    rsi14 = float(latest.get("RSI14", np.nan))
    macd_h = float(latest.get("MACD_HIST", np.nan))
    k = float(latest.get("K", np.nan))
    d = float(latest.get("D", np.nan))
    bb_pos = score_details.get("bb_pos", np.nan)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("RSI14", f"{rsi14:.1f}" if not np.isnan(rsi14) else "N/A")
    s2.metric("MACD Hist", f"{macd_h:.3f}" if not np.isnan(macd_h) else "N/A")
    s3.metric("KD (K/D)", f"{k:.1f}/{d:.1f}" if (not np.isnan(k) and not np.isnan(d)) else "N/A")
    s4.metric("BB 位置", f"{bb_pos:.2f}" if not np.isnan(bb_pos) else "N/A")

    st.subheader("📈 布林通道走勢圖（專業視覺判讀）")
    view_df = df.tail(200).copy()

    marks: List[Tuple[pd.Timestamp, float, str]] = []
    if "EMA20" in view_df.columns:
        cross_up = (view_df["Close"] > view_df["EMA20"]) & (view_df["Close"].shift(1) <= view_df["EMA20"].shift(1))
        cross_dn = (view_df["Close"] < view_df["EMA20"]) & (view_df["Close"].shift(1) >= view_df["EMA20"].shift(1))

        for idx in view_df.index[cross_up.fillna(False)].tolist()[-5:]:
            marks.append((view_df.loc[idx, "Date"], float(view_df.loc[idx, "Close"]), "▲ EMA上穿"))
        for idx in view_df.index[cross_dn.fillna(False)].tolist()[-5:]:
            marks.append((view_df.loc[idx, "Date"], float(view_df.loc[idx, "Close"]), "▼ EMA下穿"))

    plot_bollinger(view_df, title=f"{code} 布林通道 + 均線（來源：{src}）", marks=marks)

    if show_debug:
        st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
        st.dataframe(pd.DataFrame([a.__dict__ for a in attempts]), use_container_width=True)

# -----------------------------
# Mode: Top10 scanner
# -----------------------------
else:
    st.subheader("🔥 AI 強勢股 Top 10（含可操作買賣判斷）")
    st.caption("建議先用小池測試，避免 Cloud 全市場掃描超時。Top10 結果會自動去重。")

    pool_mode = st.selectbox("掃描股票池", ["小池（推薦）", "自訂（貼代號）"], index=0)
    if pool_mode.startswith("小池"):
        codes = DEFAULT_POOL_SMALL
    else:
        raw = st.text_area("貼上股票代號（用空白/逗號/換行分隔）", value="2330 2317 2454 2303 3037 2382 2603 2609 6274 4967")
        codes = [c for c in re.split(r"[\s,]+", raw.strip()) if c]

    if st.button("開始掃描 Top10"):
        with st.spinner("掃描中...（依股票池大小可能需要一點時間）"):
            topdf = scan_top10(codes, months_back, max_buy_distance, strategy, params)

        if topdf is None or topdf.empty:
            st.warning("掃不到足夠資料（可能 API 被擋或資料不足）。建議：改用小池、或稍後再試。")
        else:
            st.dataframe(topdf, use_container_width=True)
            st.caption("Top10 顯示『當下判斷（BUY/SELL/WATCH）』與『可操作買區』，可直接用來做明日測試。")

st.markdown("---")
st.caption("⚠️ 本工具僅做資訊顯示與風控演算，不構成投資建議，也不會自動下單。")