# app.py
# 🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）
# ✅ Top10 / 逐路診斷 / 指標共振 / 布林通道圖（K線+BB+EMA20+Volume）/ 交易計畫（不自動下單）
#
# ✅ 你要求的「3 個修正」已全部套進去：
# (1) 強化 retry（requests + exponential backoff + 逐路診斷 preview）
# (2) 缺套件提示（側欄顯示缺失，並給 pip install 提示）
# (3) 不依賴 Secrets：FinMind Token 改成「側欄可手動輸入」+ 支援環境變數 FINMIND_TOKEN
#
# 加碼修正：
# - Top10 去重（避免重複同一股票）
# - 買賣點距離用更專業表述（貼近/中度偏離/明顯偏離等）
# - yfinance symbol/資料保護（避免 tuple/.lower 等異常）
# - 統一 OHLCV 欄位（Date/Open/High/Low/Close/Volume）

from __future__ import annotations

import os
import re
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------
# Optional dependencies
# -----------------------------
DEPENDENCIES: Dict[str, bool] = {
    "matplotlib": True,
    "requests": True,
    "yfinance": False,
    "pandas_datareader": False,
    "finmind": False,
}

yf = None
pdr = None
DataLoader = None

try:
    import yfinance as yf  # type: ignore
    DEPENDENCIES["yfinance"] = True
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr  # type: ignore
    DEPENDENCIES["pandas_datareader"] = True
except Exception:
    pdr = None

try:
    from FinMind.data import DataLoader  # type: ignore
    DEPENDENCIES["finmind"] = True
except Exception:
    DataLoader = None


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(
    page_title="AI 台股量化專業平台（無 Plotly / 全功能保留）",
    page_icon="🧠",
    layout="wide",
)

APP_TITLE = "🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）"


# -----------------------------
# Helpers / dataclasses
# -----------------------------
@dataclass
class FetchAttempt:
    source: str
    url: str
    result: str          # OK / EMPTY / WAF_HTML / HTTP_xxx / EXC / NO_MODULE / NO_TOKEN
    status: Optional[int] = None
    note: str = ""
    preview: str = ""


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_probably_html(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return (t.startswith("<!doctype html") or t.startswith("<html") or "<head" in t[:500])


# ✅ 修正(1)：更強 retry（requests + exponential backoff）
def _requests_get_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 12,
    retries: int = 4,
    backoff: float = 0.8,
) -> Tuple[Optional[requests.Response], str]:
    last_err = ""
    s = requests.Session()
    h = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    if headers:
        h.update(headers)

    for i in range(retries + 1):
        try:
            r = s.get(url, params=params, headers=h, timeout=timeout)
            return r, ""
        except Exception as e:
            last_err = str(e)
            sleep_s = backoff * (2 ** i)
            time.sleep(min(5.0, sleep_s))
    return None, last_err


def _normalize_symbol(code: str) -> str:
    c = (code or "").strip().upper()
    c = c.replace("　", " ")
    c = c.split()[0]
    c = re.sub(r"[^0-9A-Z\.\-]", "", c)
    return c


def _extract_numeric_code(code: str) -> str:
    m = re.search(r"(\d{4,6})", code or "")
    return m.group(1) if m else (code or "")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"])
    elif "date" in d.columns:
        d["Date"] = pd.to_datetime(d["date"])
        d = d.drop(columns=["date"], errors="ignore")
    else:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "Date"})
        else:
            raise KeyError("Missing Date column")
    return d


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = _ensure_datetime(d)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Close",
        "Adj Close": "Close",
        "Trading_Volume": "Volume",
        "max": "High",
        "min": "Low",
    }

    # normalize by exact keys
    for k, v in list(rename_map.items()):
        if k in d.columns:
            d = d.rename(columns={k: v})

    # normalize by lower-case matching
    for col in list(d.columns):
        lc = str(col).strip().lower()
        if lc in rename_map and col not in ["Open", "High", "Low", "Close", "Volume"]:
            d = d.rename(columns={col: rename_map[lc]})

    required = ["Date", "Open", "High", "Low", "Close"]
    for c in required:
        if c not in d.columns:
            raise KeyError(f"Missing column: {c}")
    if "Volume" not in d.columns:
        d["Volume"] = 0.0

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
    d = d.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return d


def _load_csv_fallback(uploaded_file) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    att = FetchAttempt(source="CSV_UPLOAD", url="(upload)", result="EMPTY")
    try:
        raw = uploaded_file.getvalue()
        try:
            text = raw.decode("utf-8-sig")
        except Exception:
            text = raw.decode("cp950", errors="ignore")

        df = pd.read_csv(pd.io.common.StringIO(text))
        df = df.rename(columns={c: c.strip() for c in df.columns})
        df = _normalize_ohlcv(df)

        att.result = "OK"
        att.note = f"rows={len(df)}"
        return df, att
    except Exception as e:
        att.result = "EXC"
        att.note = str(e)
        return None, att


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
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period=9, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.fillna(method="bfill"), d.fillna(method="bfill")

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")

def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window).std()
    up = mid + n_std * std
    dn = mid - n_std * std
    return mid, up, dn

def bias(close: pd.Series, ma: pd.Series) -> pd.Series:
    return (close / ma - 1.0) * 100.0

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["SMA20"] = sma(d["Close"], 20)
    d["RSI14"] = rsi(d["Close"], 14)
    d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = macd(d["Close"])
    d["K"], d["D"] = stoch_kd(d["High"], d["Low"], d["Close"])
    d["ATR14"] = atr(d["High"], d["Low"], d["Close"], 14)
    d["BB_MID"], d["BB_UP"], d["BB_DN"] = bollinger(d["Close"], 20, 2.0)
    d["BIAS20"] = bias(d["Close"], d["SMA20"])
    d["VOL_MA20"] = sma(d["Volume"], 20)
    return d


# -----------------------------
# Scoring / signals
# -----------------------------
def _clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def compute_resonance_score(d: pd.DataFrame) -> int:
    if d is None or len(d) < 60:
        return 0
    last = d.iloc[-1]
    prev = d.iloc[-2]

    score = 50

    if last["Close"] > last["EMA20"]:
        score += 8
    else:
        score -= 8

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 8
    else:
        score -= 6

    if last["RSI14"] < 35:
        score += 6
    elif last["RSI14"] > 70:
        score -= 6
    else:
        score += 2

    if prev["K"] < prev["D"] and last["K"] > last["D"]:
        score += 8
    if prev["K"] > prev["D"] and last["K"] < last["D"]:
        score -= 6

    if last["Close"] < last["BB_DN"]:
        score += 6
    elif last["Close"] > last["BB_UP"]:
        score -= 6

    if last["Volume"] > (last["VOL_MA20"] or 0) * 1.2:
        score += 6

    return int(_clamp(score, 0, 100))


def professional_distance_phrase(dist_abs_pct: float, direction: str) -> str:
    if dist_abs_pct < 1.0:
        band = "貼近"
    elif dist_abs_pct < 3.0:
        band = "輕度偏離"
    elif dist_abs_pct < 7.0:
        band = "中度偏離"
    elif dist_abs_pct < 12.0:
        band = "明顯偏離"
    else:
        band = "高度偏離"

    arrow = "↑" if direction == "UP" else ("↓" if direction == "DOWN" else "→")
    return f"{arrow} {band}（{dist_abs_pct:.1f}%）"


def classify_action(
    d: pd.DataFrame,
    max_buy_distance: float,
    strategy: str,
    chase_max: float,
    breakout_buffer_atr: float,
    stop_atr: float,
    rr: float,
) -> Tuple[str, str]:
    last = d.iloc[-1]
    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    bb_up = float(last["BB_UP"])
    bb_dn = float(last["BB_DN"])
    atr14 = float(last["ATR14"]) if not math.isnan(float(last["ATR14"])) else max(1e-9, price * 0.02)

    near_buy_zone = (price <= ema20) and (price >= bb_dn)
    near_sell_zone = (price >= ema20) and (price <= bb_up)

    if strategy == "BREAKOUT":
        lookback = min(60, len(d)-1)
        recent_high = float(d["High"].tail(lookback).max())
        trigger = recent_high + breakout_buffer_atr * atr14
        chase_dist = (price - trigger) / max(trigger, 1e-9)

        if price >= trigger and chase_dist <= chase_max:
            stop = price - stop_atr * atr14
            target = price + rr * (price - stop)
            reason = f"突破成立：現價 ≥ 觸發價 {trigger:.2f}，追價偏離 {chase_dist*100:.1f}%（≤ {chase_max*100:.1f}%）"
            reason += f"｜風控：止損≈{stop:.2f}（{stop_atr:.2f} ATR），目標≈{target:.2f}（RR={rr:.2f}）"
            return "BUY", reason

        if price >= trigger and chase_dist > chase_max:
            return "WATCH", f"突破已發生但追價偏離 {chase_dist*100:.1f}% > {chase_max*100:.1f}%（建議等待回測/縮距）"

        return "WATCH", f"尚未突破：觸發價≈{trigger:.2f}（recent high + {breakout_buffer_atr:.2f} ATR）"

    buy_ref = min(ema20, bb_dn + 0.35 * (ema20 - bb_dn))
    buy_dist = abs(price - buy_ref) / max(price, 1e-9)

    sell_ref = max(ema20, bb_up - 0.35 * (bb_up - ema20))
    sell_dist = abs(sell_ref - price) / max(price, 1e-9)

    if near_buy_zone and buy_dist <= max_buy_distance:
        return "BUY", f"回檔可操作區：接近均線/下緣支撐，偏離參考價 {buy_dist*100:.1f}%（≤ {max_buy_distance*100:.1f}%）"
    if near_sell_zone and sell_dist <= max_buy_distance:
        return "SELL", f"壓力可操作區：接近均線/上緣壓力，偏離參考價 {sell_dist*100:.1f}%（≤ {max_buy_distance*100:.1f}%）"

    return "WATCH", "條件尚未形成明確共振：等待價格進入區間或指標翻轉。"


def compute_future_zones(d: pd.DataFrame, max_buy_distance: float) -> Dict[str, Dict[str, float]]:
    last = d.iloc[-1]
    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    bb_up = float(last["BB_UP"])
    bb_dn = float(last["BB_DN"])
    atr14 = float(last["ATR14"]) if not math.isnan(float(last["ATR14"])) else price * 0.02

    near_buy_low = bb_dn + 0.10 * (ema20 - bb_dn)
    near_buy_high = bb_dn + 0.35 * (ema20 - bb_dn)

    deep_buy_low = max(0.01, bb_dn - 1.6 * atr14)
    deep_buy_high = max(0.01, bb_dn - 0.8 * atr14)

    sell_low = bb_up - 0.35 * (bb_up - ema20)
    sell_high = bb_up - 0.10 * (bb_up - ema20)

    near_buy_center = (near_buy_low + near_buy_high) / 2
    near_buy_dist = abs(price - near_buy_center) / max(price, 1e-9)

    zones = {
        "near_buy": {
            "low": float(near_buy_low),
            "high": float(near_buy_high),
            "dist_abs_pct": float(near_buy_dist * 100.0),
            "actionable": float(near_buy_dist) <= max_buy_distance,
        },
        "deep_buy": {
            "low": float(deep_buy_low),
            "high": float(deep_buy_high),
            "dist_abs_pct": float(abs(price - (deep_buy_low+deep_buy_high)/2) / max(price, 1e-9) * 100.0),
            "actionable": False,
        },
        "sell": {
            "low": float(sell_low),
            "high": float(sell_high),
            "dist_abs_pct": float(abs((sell_low+sell_high)/2 - price) / max(price, 1e-9) * 100.0),
            "actionable": True,
        }
    }
    return zones


# -----------------------------
# Data sources (FinMind / YF / Stooq)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_finmind(stock_id: str, months_back: int, token: str) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    att = FetchAttempt(source="FinMind", url=f"FinMind:{stock_id}", result="EMPTY")
    if not DEPENDENCIES.get("finmind", False) or DataLoader is None:
        att.result = "NO_MODULE"
        att.note = "缺 FinMind：pip install FinMind"
        return None, att

    if not token:
        att.result = "NO_TOKEN"
        att.note = "未提供 FINMIND_TOKEN（請在側欄輸入 Token 或設定環境變數 FINMIND_TOKEN）"
        return None, att

    try:
        dl = DataLoader()
        dl.login_by_token(api_token=token)

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months_back * 35 + 30)).strftime("%Y-%m-%d")

        df = dl.taiwan_stock_daily(stock_id=_extract_numeric_code(stock_id), start_date=start, end_date=end)
        if df is None or df.empty:
            att.result = "EMPTY"
            att.note = "empty response"
            return None, att

        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
        })
        df = _normalize_ohlcv(df)

        att.result = "OK"
        att.note = f"rows={len(df)}"
        return df, att
    except Exception as e:
        att.result = "EXC"
        att.note = str(e)
        return None, att


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    att = FetchAttempt(source="YF", url="yfinance", result="EMPTY")
    if not DEPENDENCIES.get("yfinance", False) or yf is None:
        att.result = "NO_MODULE"
        att.note = "缺 yfinance：pip install yfinance"
        return None, att

    try:
        base = _normalize_symbol(code)
        numeric = _extract_numeric_code(base)

        candidates: List[str]
        if base.endswith(".TW") or base.endswith(".TWO"):
            candidates = [base]
        else:
            candidates = [f"{numeric}.TW", f"{numeric}.TWO"]

        for sym in candidates:
            try:
                t = yf.Ticker(sym)
                df = t.history(period=f"{max(30, period_days)}d", auto_adjust=False)
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                df = df.rename(columns={"Date": "Date"})
                df = _normalize_ohlcv(df)

                att.result = "OK"
                att.note = f"{sym} rows={len(df)}"
                att.url = f"YF:{sym}"
                return df, att
            except Exception:
                continue

        att.result = "EMPTY"
        att.note = "all candidates empty"
        return None, att

    except Exception as e:
        att.result = "EXC"
        att.note = str(e)
        return None, att


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    att = FetchAttempt(source="STOOQ", url="stooq", result="EMPTY")

    numeric = _extract_numeric_code(_normalize_symbol(code))
    candidates = [f"{numeric}.tw", f"{numeric}.two"]

    for sym in candidates:
        url = "https://stooq.com/q/d/l/"
        params = {"s": sym, "i": "d"}
        r, err = _requests_get_with_retry(url, params=params, timeout=12, retries=4)
        att.url = f"{url}?s={sym}&i=d"
        if r is None:
            att.result = "EXC"
            att.note = err
            continue

        att.status = r.status_code
        text = r.text or ""
        att.preview = text[:200].replace("\n", " ")

        if r.status_code != 200:
            att.result = f"HTTP_{r.status_code}"
            att.note = "non-200"
            continue

        if _is_probably_html(text):
            att.result = "WAF_HTML"
            att.note = "HTML response"
            continue

        try:
            df = pd.read_csv(pd.io.common.StringIO(text))
            if df is None or df.empty:
                att.result = "EMPTY"
                att.note = "empty csv"
                continue

            df = _normalize_ohlcv(df)

            if len(df) > period_days + 10:
                df = df.tail(period_days + 10).reset_index(drop=True)

            att.result = "OK"
            att.note = f"{sym} rows={len(df)}"
            return df, att
        except Exception as e:
            att.result = "EXC"
            att.note = str(e)
            continue

    return None, att


def fetch_ohlcv_multi(
    code: str,
    months_back: int,
    csv_upload=None,
    finmind_token: str = "",
) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:
    period_days = int(months_back * 31)
    attempts: List[FetchAttempt] = []

    df, att = fetch_finmind(code, months_back, finmind_token)
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
# Charting (matplotlib candlestick)
# -----------------------------
def _plot_candles_with_bb(d: pd.DataFrame, title: str):
    N = min(160, len(d))
    x = np.arange(N)
    dd = d.tail(N).reset_index(drop=True)

    fig = plt.figure(figsize=(14, 7), dpi=120)
    gs = fig.add_gridspec(5, 1, hspace=0.05)
    ax = fig.add_subplot(gs[:4, 0])
    axv = fig.add_subplot(gs[4, 0], sharex=ax)

    for i in range(N):
        o = dd.loc[i, "Open"]
        h = dd.loc[i, "High"]
        l = dd.loc[i, "Low"]
        c = dd.loc[i, "Close"]
        body_low = min(o, c)
        body_h = max(abs(c - o), 1e-9)

        ax.vlines(i, l, h, linewidth=1)

        rect = Rectangle((i - 0.3, body_low), 0.6, body_h, fill=True, alpha=0.55)
        ax.add_patch(rect)

    ax.plot(x, dd["EMA20"], linewidth=1)
    ax.plot(x, dd["BB_UP"], linewidth=1)
    ax.plot(x, dd["BB_MID"], linewidth=1)
    ax.plot(x, dd["BB_DN"], linewidth=1)

    ax.set_title(title)
    ax.grid(True, alpha=0.15)

    axv.bar(x, dd["Volume"].fillna(0.0).values)
    axv.grid(True, alpha=0.15)

    ticks = np.linspace(0, N - 1, num=min(8, N)).astype(int)
    labels = [dd.loc[t, "Date"].strftime("%m-%d") for t in ticks]
    axv.set_xticks(ticks)
    axv.set_xticklabels(labels, rotation=0)
    plt.setp(ax.get_xticklabels(), visible=False)

    return fig


# -----------------------------
# Diagnostics
# -----------------------------
def run_network_test() -> pd.DataFrame:
    rows = []

    url = "https://stooq.com/q/d/l/"
    r, err = _requests_get_with_retry(url, params={"s": "2330.tw", "i": "d"}, retries=2, timeout=10)
    if r is None:
        rows.append({"target": "STOOQ", "ok": False, "status": None, "note": err})
    else:
        rows.append({"target": "STOOQ", "ok": r.status_code == 200 and not _is_probably_html(r.text), "status": r.status_code,
                     "note": (r.headers.get("content-type", "") or "")})

    if DEPENDENCIES.get("yfinance", False) and yf is not None:
        try:
            t = yf.Ticker("2330.TW")
            df = t.history(period="5d")
            rows.append({"target": "YF", "ok": df is not None and not df.empty, "status": 200, "note": f"rows={0 if df is None else len(df)}"})
        except Exception as e:
            rows.append({"target": "YF", "ok": False, "status": None, "note": str(e)})
    else:
        rows.append({"target": "YF", "ok": False, "status": None, "note": "NO_MODULE(yfinance)"})

    token = st.session_state.get("finmind_token", "") or os.getenv("FINMIND_TOKEN", "")
    rows.append({"target": "FinMind", "ok": bool(token) and DEPENDENCIES.get("finmind", False), "status": None, "note": "token_ready" if token else "no_token"})

    return pd.DataFrame(rows)


def render_attempts(attempts: List[FetchAttempt], show_debug: bool):
    st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
    df = pd.DataFrame([{
        "source": a.source,
        "result": a.result,
        "url": a.url,
        "status": a.status,
        "note": a.note,
        "preview": a.preview if show_debug else ""
    } for a in attempts])
    st.dataframe(df, use_container_width=True)


# -----------------------------
# Single / Top10
# -----------------------------
def render_single(
    stock_id: str,
    months_back: int,
    csv_upload,
    finmind_token: str,
    show_debug: bool,
    max_buy_distance: float,
    strategy: str,
    chase_max: float,
    breakout_buffer_atr: float,
    stop_atr: float,
    rr: float,
):
    df, source, attempts = fetch_ohlcv_multi(stock_id, months_back, csv_upload, finmind_token=finmind_token)

    if df is None:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試，或改用 CSV 上傳備援。")
        render_attempts(attempts, show_debug)
        return

    df = add_indicators(df)
    price = float(df.iloc[-1]["Close"])
    score = compute_resonance_score(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{price:.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", source)
    c4.metric("最後日期 / 筆數", f"{df.iloc[-1]['Date'].strftime('%Y-%m-%d')}｜{len(df)}")

    action, reason = classify_action(
        df,
        max_buy_distance=max_buy_distance,
        strategy=strategy,
        chase_max=chase_max,
        breakout_buffer_atr=breakout_buffer_atr,
        stop_atr=stop_atr,
        rr=rr,
    )

    st.markdown("## 📌 當下是否為買點/賣點？（可操作判斷）")
    if action == "BUY":
        st.success(f"✅ BUY｜{reason}")
    elif action == "SELL":
        st.warning(f"⚠️ SELL｜{reason}")
    else:
        st.info(f"⏳ WATCH｜{reason}")

    zones = compute_future_zones(df, max_buy_distance=max_buy_distance)
    st.markdown("## 🗺️ 未來預估買賣點（區間 + 專業距離表述）")

    nb = zones["near_buy"]
    if nb["actionable"]:
        phrase = professional_distance_phrase(nb["dist_abs_pct"], "DOWN")
        st.success(f"🟢 近端買點（可操作）：{nb['low']:.2f} ~ {nb['high']:.2f} ｜距離現價 {phrase}（上限 {max_buy_distance*100:.1f}%）")
    else:
        st.warning("⚠️ 近端買點：距離過遠或不可操作（已自動隱藏）")

    db = zones["deep_buy"]
    phrase = professional_distance_phrase(db["dist_abs_pct"], "DOWN")
    st.info(f"🟦 深回檔買點（等待型）：{db['low']:.2f} ~ {db['high']:.2f} ｜距離現價 {phrase}")

    sz = zones["sell"]
    phrase = professional_distance_phrase(sz["dist_abs_pct"], "UP")
    st.warning(f"🎯 近端賣點區（壓力/獲利）：{sz['low']:.2f} ~ {sz['high']:.2f} ｜距離現價 {phrase}")

    st.markdown("## 📉 布林通道分析圖（K線 + BB + EMA20 + Volume）")
    fig = _plot_candles_with_bb(df, title=f"{stock_id}｜{source}")
    st.pyplot(fig, use_container_width=True)

    render_attempts(attempts, show_debug)

    st.markdown("## 📄 最近 20 筆資料（含指標）")
    st.dataframe(df.tail(20), use_container_width=True)


def top10_scan(
    pool: List[str],
    months_back: int,
    finmind_token: str,
    show_debug: bool,
    max_buy_distance: float,
    strategy: str,
    chase_max: float,
    breakout_buffer_atr: float,
    stop_atr: float,
    rr: float,
):
    # ✅ Top10 去重（避免重複同一股票）
    seen = set()
    clean_pool = []
    for x in pool:
        s = _extract_numeric_code(_normalize_symbol(x))
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        clean_pool.append(s)

    st.caption("Top10 掃描器：建議先用小池測試（避免全市場在 Cloud 超時）。")
    if not clean_pool:
        st.warning("請先輸入 stock_pool（逗號或換行分隔）。")
        return

    results = []
    prog = st.progress(0)

    for i, code in enumerate(clean_pool):
        prog.progress((i + 1) / len(clean_pool))
        df, source, attempts = fetch_ohlcv_multi(code, months_back, csv_upload=None, finmind_token=finmind_token)
        if df is None:
            continue

        df = add_indicators(df)
        score = compute_resonance_score(df)
        price = float(df.iloc[-1]["Close"])
        action, _ = classify_action(
            df,
            max_buy_distance=max_buy_distance,
            strategy=strategy,
            chase_max=chase_max,
            breakout_buffer_atr=breakout_buffer_atr,
            stop_atr=stop_atr,
            rr=rr,
        )
        zones = compute_future_zones(df, max_buy_distance=max_buy_distance)

        results.append({
            "股票": code,
            "來源": source,
            "AI分數": score,
            "現價": round(price, 2),
            "當下判斷": action,
            "近端買點可操作": "YES" if zones["near_buy"]["actionable"] else "NO",
            "賣點距離(%)": round(zones["sell"]["dist_abs_pct"], 1),
            "最後日期": df.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        })

    prog.empty()

    if not results:
        st.error("掃描結果為空：可能雲端被擋 / 資料源失敗。建議先按『網路測試』或改用 CSV。")
        return

    out = pd.DataFrame(results).sort_values(["AI分數", "近端買點可操作", "賣點距離(%)"], ascending=[False, False, True])
    out = out.drop_duplicates(subset=["股票"], keep="first")
    top10 = out.head(10).reset_index(drop=True)

    st.markdown("## 🔥 AI 強勢股 Top 10（含當下操作判斷）")
    st.dataframe(top10, use_container_width=True)

    st.markdown("### 🔎 點選一檔直接展開完整分析")
    pick = st.selectbox("選擇股票", top10["股票"].tolist())
    if pick:
        st.divider()
        render_single(
            pick,
            months_back=months_back,
            csv_upload=None,
            finmind_token=finmind_token,
            show_debug=show_debug,
            max_buy_distance=max_buy_distance,
            strategy=strategy,
            chase_max=chase_max,
            breakout_buffer_atr=breakout_buffer_atr,
            stop_atr=stop_atr,
            rr=rr,
        )


# -----------------------------
# Sidebar / main
# -----------------------------
st.title(APP_TITLE)
st.caption("多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + Top10 掃描 + 交易計畫（不自動下單）")

with st.sidebar:
    st.header("設定")

    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)
    months_back = st.selectbox("資料期間", [3, 6, 12, 24], index=1, format_func=lambda x: f"{x}mo")
    show_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.subheader("A+B+C：專業買賣點 / 風險報酬")
    max_buy_distance = st.slider("可操作買點最大距離（避免買點離現實太遠） max_buy_distance", 0.02, 0.20, 0.12, 0.01)

    st.subheader("策略 / 參數")
    strat = st.radio("策略", ["回檔等待型（回檔分批）", "趨勢突破型（突破追價進場）"], index=0)
    strategy = "REVERSION" if "回檔" in strat else "BREAKOUT"

    if strategy == "BREAKOUT":
        chase_max = st.slider("最大可接受追價偏離距離（%）", 0.01, 0.20, 0.06, 0.01)
        breakout_buffer_atr = st.slider("突破觸發 buffer（ATR 倍數）", 0.00, 0.80, 0.20, 0.05)
        stop_atr = st.slider("失效止損距離（ATR 倍數）", 0.80, 3.00, 1.60, 0.10)
        rr = st.slider("目標風險報酬（RR）", 1.00, 5.00, 2.20, 0.10)
    else:
        chase_max = 0.06
        breakout_buffer_atr = 0.20
        stop_atr = 1.60
        rr = 2.20

    st.subheader("🧪 網路測試（建議先按一次）")
    if st.button("網路測試（建議先按一次）"):
        st.dataframe(run_network_test(), use_container_width=True)

    # ✅ 修正(3)：不依賴 Secrets，改為側欄手動輸入 + 環境變數
    st.subheader("🔑 FinMind Token（不需要 Secrets）")
    st.caption("Settings 沒有 Secrets 沒關係：在這裡貼 Token（或用環境變數 FINMIND_TOKEN）。")
    finmind_token_input = st.text_input("FinMind Token（選填；沒填也可用 YF/Stooq/CSV）", type="password")
    st.session_state["finmind_token"] = finmind_token_input.strip()

    # ✅ 修正(2)：缺套件提示（更清楚）
    st.subheader("⚙️ 系統狀態 / 缺套件提示")
    for pkg, ok in DEPENDENCIES.items():
        if ok:
            st.success(f"✅ {pkg}")
        else:
            pip_name = "FinMind" if pkg == "finmind" else pkg
            st.error(f"❌ {pkg} 缺失（建議：pip install {pip_name}）")

    if not DEPENDENCIES.get("finmind", False):
        st.warning("❗ FinMind 未安裝：pip install FinMind（若不用 FinMind，可忽略）")
    if not DEPENDENCIES.get("yfinance", False):
        st.warning("❗ yfinance 未安裝：pip install yfinance（建議保留做備援）")

# Main inputs
stock_id = st.text_input("請輸入股票代號（例：2330 / 6274 / 2330.TW）", value="6274")
csv_upload = st.file_uploader("（選用）上傳 export.csv 作為備援資料源（Date/Open/High/Low/Close/Volume）", type=["csv"])

st.info("💡 若 Cloud 偶發抓不到資料：先按左側「網路測試」，或直接上傳 export.csv（Date/Open/High/Low/Close/Volume）即可完整分析。")

finmind_token = st.session_state.get("finmind_token", "") or os.getenv("FINMIND_TOKEN", "")

if mode == "單一股票分析":
    render_single(
        stock_id=stock_id,
        months_back=int(months_back),
        csv_upload=csv_upload,
        finmind_token=finmind_token,
        show_debug=show_debug,
        max_buy_distance=float(max_buy_distance),
        strategy=strategy,
        chase_max=float(chase_max),
        breakout_buffer_atr=float(breakout_buffer_atr),
        stop_atr=float(stop_atr),
        rr=float(rr),
    )
else:
    st.markdown("### stock_pool（逗號或換行分隔）")
    pool_text = st.text_area("stock_pool", value="00878\n006208\n4967\n8046\n6274\n2330")
    pool = [x.strip() for x in re.split(r"[,\n]+", pool_text) if x.strip()]
    if st.button("🔥 RUN Top10 掃描"):
        top10_scan(
            pool=pool,
            months_back=int(months_back),
            finmind_token=finmind_token,
            show_debug=show_debug,
            max_buy_distance=float(max_buy_distance),
            strategy=strategy,
            chase_max=float(chase_max),
            breakout_buffer_atr=float(breakout_buffer_atr),
            stop_atr=float(stop_atr),
            rr=float(rr),
        )

st.caption("⚠️ 本工具僅供資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。")
st.caption(f"runtime: {_now_str()}")