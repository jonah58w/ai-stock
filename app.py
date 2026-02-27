# app.py
# 🧠 AI 台股量化專業平台（最終極全備援 + 逐路診斷 / 不自動下單 / 無 Plotly）
# ✅ 單一股票分析：當下 BUY/SELL/WATCH + 理由 + 未來預估買賣點（近端/深回檔/近端賣點）
# ✅ Top10 掃描：小池 stock_pool、去重、顯示「操作判斷 + 距離%」
# ✅ 逐路診斷：顯示每個資料源的狀態（HTTP/內容/錯誤片段）
# ✅ 指標：EMA20/EMA60、MACD、KD(隨機)、RSI、乖離率、布林通道、成交量均線
# ✅ 圖：Matplotlib K線 + BB + EMA20 + Volume（無 plotly）
# ✅ 資料源備援順序（可改）：TWSE_JSON -> Yahoo (yfinance) -> 使用者上傳CSV
#    - TWSE/TPEX：優先用 TWSE 的 STOCK_DAY JSON（Streamlit Cloud 最穩）
#    - 上櫃若 TWSE 取不到，會自動改用 yfinance .TWO（若 Cloud 擋 Yahoo，仍可用上傳CSV當最後備援）

from __future__ import annotations

import io
import re
import time
import math
import json
import datetime as dt
from dataclasses import dataclass
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
# 基本設定
# -----------------------------
APP_TITLE = "🧠 AI 台股量化專業平台（最終極全備援 + 逐路診斷版 / 不自動下單 / 無 Plotly）"

DEFAULT_POOL = """00878
006208
4967
8046
6274
2330
2317
2454
2382
2303
"""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

PERIOD_MAP = {
    "3mo": 3,
    "6mo": 6,
    "1y": 12,
    "2y": 24,
    "5y": 60,
}


# -----------------------------
# 小工具
# -----------------------------
def _now_taipei() -> dt.datetime:
    # Streamlit Cloud 可能是 UTC，這裡用固定 +8 顯示用
    return dt.datetime.utcnow() + dt.timedelta(hours=8)

def _to_float(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in ("", "--", "—", "-", "None", "nan", "NaN"):
        return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def _clean_stock_id(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(".TW", "").replace(".TWO", "").replace(".two", "").replace(".tw", "")
    return s

def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _is_taiwan_stock_id(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4,6}", s))


# -----------------------------
# 指標計算（不用 ta-lib / plotly）
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 基本欄位保護
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan

    close = df["Close"].astype(float)

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA60"] = close.ewm(span=60, adjust=False).mean()

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI14"] = 100 - (100 / (1 + rs))

    # KD (Stochastic %K(9), %D(3))
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    k = 100 * (close - low9) / (high9 - low9).replace(0, np.nan)
    df["K"] = k.rolling(3).mean()
    df["D"] = df["K"].rolling(3).mean()

    # 乖離率（相對 EMA20）
    df["Bias20"] = (close / df["EMA20"] - 1.0) * 100.0

    # Bollinger(20,2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std

    # 量能均線
    df["VMA20"] = df["Volume"].rolling(20).mean()

    return df


# -----------------------------
# 買賣點區間（讓買點不要離現價太遠）
# -----------------------------
@dataclass
class Zones:
    action: str
    reason: str
    score: int
    near_buy: Optional[Tuple[float, float]]
    deep_buy: Optional[Tuple[float, float]]
    near_sell: Optional[Tuple[float, float]]
    near_buy_dist: Optional[float]
    deep_buy_dist: Optional[float]
    near_sell_dist: Optional[float]

def _pct_dist(curr: float, lo: float, hi: float) -> float:
    center = (lo + hi) / 2.0
    if not np.isfinite(curr) or curr == 0:
        return np.nan
    return abs(center - curr) / curr

def compute_zones(df: pd.DataFrame, max_buy_distance: float = 0.12) -> Zones:
    """
    規則：
    - 近端買點：以 BB_lower、EMA20、近 20 日區間支撐 組合（會更貼近現價）
    - 深回檔買點：以 BB_lower 再往下加一個 std 的保守區
    - 近端賣點：以 BB_upper 與近 20 日壓力區組合
    - 若近端買點距離 > max_buy_distance：自動隱藏近端買點（避免「離現實太遠」）
    """
    if df is None or len(df) < 60:
        return Zones(
            action="WATCH",
            reason="資料不足（至少建議 60 根以上）",
            score=0,
            near_buy=None, deep_buy=None, near_sell=None,
            near_buy_dist=None, deep_buy_dist=None, near_sell_dist=None
        )

    last = df.iloc[-1]
    curr = float(last["Close"])
    ema20 = float(last["EMA20"])
    bb_l = float(last["BB_lower"])
    bb_m = float(last["BB_mid"])
    bb_u = float(last["BB_upper"])

    # 支撐/壓力（用近 20 日低點/高點做簡單區間）
    window = df.iloc[-20:]
    sup = float(window["Low"].min())
    res = float(window["High"].max())

    # 近端買點（貼近現價的「可操作區」）
    # 讓區間更合理：取 max(支撐, BB_lower, EMA20*0.97) 到 min(EMA20*0.995, BB_mid*0.98)
    nb_lo = max(sup, bb_l, ema20 * 0.97)
    nb_hi = min(ema20 * 0.995, bb_m * 0.985)
    if nb_hi <= nb_lo:
        # fallback：以 sup 附近做窄區
        nb_lo = max(sup * 0.995, bb_l)
        nb_hi = nb_lo * 1.01

    # 深回檔買點（更保守）
    # bb_l 再下移一點，但不要離譜：用近 60 日低點下緣附近
    window60 = df.iloc[-60:]
    low60 = float(window60["Low"].min())
    db_lo = max(low60 * 0.995, bb_l * 0.92)
    db_hi = min(bb_l * 0.98, low60 * 1.03)
    if db_hi <= db_lo:
        db_lo = low60 * 0.99
        db_hi = low60 * 1.02

    # 近端賣點（壓力/獲利）
    ns_lo = max(bb_u * 0.985, res * 0.99)
    ns_hi = max(ns_lo * 1.01, bb_u * 1.02)
    # 防止過寬
    ns_hi = min(ns_hi, curr * 1.25)

    nb_dist = _pct_dist(curr, nb_lo, nb_hi)
    db_dist = _pct_dist(curr, db_lo, db_hi)
    ns_dist = _pct_dist(curr, ns_lo, ns_hi)

    # 若近端買點距離太遠，隱藏
    near_buy = (round(nb_lo, 2), round(nb_hi, 2))
    if np.isfinite(nb_dist) and nb_dist > max_buy_distance:
        near_buy = None

    deep_buy = (round(db_lo, 2), round(db_hi, 2))
    near_sell = (round(ns_lo, 2), round(ns_hi, 2))

    # 當下操作判斷（可操作）
    # BUY：價格接近/落入近端買點 + 指標轉強
    # SELL：價格接近/落入近端賣點 + 指標轉弱
    k = float(last["K"]) if np.isfinite(last["K"]) else np.nan
    d = float(last["D"]) if np.isfinite(last["D"]) else np.nan
    macd_h = float(last["MACD_hist"]) if np.isfinite(last["MACD_hist"]) else np.nan
    rsi = float(last["RSI14"]) if np.isfinite(last["RSI14"]) else np.nan

    def in_zone(z):
        if z is None:
            return False
        lo, hi = z
        return (curr >= lo) and (curr <= hi)

    bullish_flip = (np.isfinite(macd_h) and macd_h > 0) or (np.isfinite(k) and np.isfinite(d) and k > d)
    bearish_flip = (np.isfinite(macd_h) and macd_h < 0) or (np.isfinite(k) and np.isfinite(d) and k < d)

    action = "WATCH"
    reason = "條件尚未明確，等待價格進入區間或指標翻轉。"

    if near_buy and (in_zone(near_buy) or (np.isfinite(nb_dist) and nb_dist <= 0.03)) and bullish_flip:
        action = "BUY"
        reason = "價格接近/進入近端買點區，且指標轉強（MACD/KD）。"
    elif in_zone(near_sell) or (np.isfinite(ns_dist) and ns_dist <= 0.03):
        if bearish_flip or (np.isfinite(rsi) and rsi >= 70):
            action = "SELL"
            reason = "價格接近/進入近端賣點區，且指標轉弱或過熱（RSI/MACD/KD）。"
        else:
            action = "WATCH"
            reason = "價格接近壓力區，但指標未轉弱；偏向觀望/分批獲利。"

    # 共振分數（100分）
    score = 0
    # 趨勢
    if curr > ema20:
        score += 15
    if ema20 > float(last["EMA60"]):
        score += 10
    # 動能
    if np.isfinite(macd_h) and macd_h > 0:
        score += 15
    if np.isfinite(k) and np.isfinite(d) and k > d:
        score += 10
    # 位置（靠近 BB_mid~upper 加分）
    if np.isfinite(bb_m) and curr > bb_m:
        score += 10
    # 量能
    v = float(last["Volume"]) if np.isfinite(last["Volume"]) else np.nan
    vma = float(last["VMA20"]) if np.isfinite(last["VMA20"]) else np.nan
    if np.isfinite(v) and np.isfinite(vma) and v > vma:
        score += 10
    # 乖離率不過熱
    bias = float(last["Bias20"]) if np.isfinite(last["Bias20"]) else np.nan
    if np.isfinite(bias) and abs(bias) <= 8:
        score += 10
    # RSI 中性偏強
    if np.isfinite(rsi) and 45 <= rsi <= 70:
        score += 10

    score = int(min(100, max(0, score)))

    return Zones(
        action=action,
        reason=reason,
        score=score,
        near_buy=near_buy,
        deep_buy=deep_buy,
        near_sell=near_sell,
        near_buy_dist=(None if near_buy is None else float(nb_dist)),
        deep_buy_dist=float(db_dist) if np.isfinite(db_dist) else None,
        near_sell_dist=float(ns_dist) if np.isfinite(ns_dist) else None,
    )


# -----------------------------
# Matplotlib K 線 + BB + EMA20 + Volume（無 plotly）
# -----------------------------
def plot_bollinger_candles(df: pd.DataFrame, title: str = ""):
    dfp = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(dfp) < 30:
        fig = plt.figure(figsize=(10, 4))
        plt.title("資料不足，無法繪圖")
        return fig

    # 只畫最近 N 根，避免 Cloud 太慢
    N = min(140, len(dfp))
    dfp = dfp.iloc[-N:]
    x = np.arange(len(dfp))

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(5, 1, hspace=0.05)
    ax = fig.add_subplot(gs[:4, 0])
    axv = fig.add_subplot(gs[4, 0], sharex=ax)

    # Candles
    o = dfp["Open"].values
    h = dfp["High"].values
    l = dfp["Low"].values
    c = dfp["Close"].values

    for i in range(len(dfp)):
        up = c[i] >= o[i]
        color = "#2ecc71" if up else "#e74c3c"
        # wick
        ax.vlines(x[i], l[i], h[i], color=color, linewidth=1)
        # body
        body_low = min(o[i], c[i])
        body_h = max(0.000001, abs(c[i] - o[i]))
        ax.add_patch(Rectangle((x[i] - 0.35, body_low), 0.7, body_h, facecolor=color, edgecolor=color, alpha=0.9))

    # BB + EMA20
    ax.plot(x, dfp["BB_upper"].values, linewidth=1.2, label="BB Upper")
    ax.plot(x, dfp["BB_mid"].values, linewidth=1.2, label="BB Mid")
    ax.plot(x, dfp["BB_lower"].values, linewidth=1.2, label="BB Lower")
    ax.plot(x, dfp["EMA20"].values, linewidth=1.2, label="EMA20")

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    # Volume
    vol = dfp["Volume"].fillna(0).values
    axv.bar(x, vol, width=0.7)
    axv.plot(x, dfp["VMA20"].fillna(0).values, linewidth=1.2, label="VMA20")
    axv.grid(True, alpha=0.25)
    axv.legend(loc="upper left", fontsize=9)

    # x ticks: date strings
    dates = [d.strftime("%Y-%m-%d") for d in dfp.index]
    step = max(1, len(dates) // 6)
    axv.set_xticks(x[::step])
    axv.set_xticklabels(dates[::step], rotation=0, fontsize=9)

    plt.setp(ax.get_xticklabels(), visible=False)
    return fig


# -----------------------------
# 資料下載（逐路備援 + 診斷）
# -----------------------------
@dataclass
class FetchAttempt:
    source: str
    result: str
    url: str
    status: Optional[int]
    snippet: str

def _req_text(url: str, timeout: int = 12) -> Tuple[Optional[int], str, Optional[str]]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        txt = r.text if r is not None else ""
        return (r.status_code, txt, None)
    except Exception as e:
        return (None, "", str(e))

def _req_json(url: str, timeout: int = 12) -> Tuple[Optional[int], Optional[dict], Optional[str], str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        txt = r.text if r is not None else ""
        try:
            j = r.json()
        except Exception:
            j = None
        return (r.status_code, j, None, txt[:200])
    except Exception as e:
        return (None, None, str(e), "")

def _twse_month_list(months_back: int) -> List[dt.date]:
    """回傳每個月的 1 號，用於 STOCK_DAY 按月抓"""
    today = _now_taipei().date().replace(day=1)
    out = []
    y = today.year
    m = today.month
    for _ in range(months_back):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m <= 0:
            m = 12
            y -= 1
    return list(reversed(out))

def _parse_twse_stock_day_json(j: dict) -> pd.DataFrame:
    # TWSE STOCK_DAY JSON 格式：{"stat":"OK","date":"20250201","title":...,"fields":[...],"data":[...]}
    if not isinstance(j, dict):
        return pd.DataFrame()
    if j.get("stat") != "OK":
        return pd.DataFrame()
    fields = j.get("fields", [])
    data = j.get("data", [])
    if not fields or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=fields)

    # 常見欄位（中文）
    # 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
    col_date = None
    for c in df.columns:
        if "日期" in c:
            col_date = c
            break
    if col_date is None:
        return pd.DataFrame()

    def parse_date_tw(x: str) -> Optional[dt.datetime]:
        # 例如 "114/02/03"
        s = str(x).strip()
        m = re.match(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", s)
        if not m:
            return None
        yy = int(m.group(1)) + 1911
        mm = int(m.group(2))
        dd = int(m.group(3))
        return dt.datetime(yy, mm, dd)

    df["Date"] = df[col_date].apply(parse_date_tw)
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date")

    # 抽 OHLCV
    def pick(name_keywords: List[str]) -> Optional[str]:
        for k in name_keywords:
            for c in df.columns:
                if k in c:
                    return c
        return None

    c_open = pick(["開盤"])
    c_high = pick(["最高"])
    c_low = pick(["最低"])
    c_close = pick(["收盤"])
    c_vol = pick(["成交股數", "成交量"])

    if not all([c_open, c_high, c_low, c_close, c_vol]):
        return pd.DataFrame()

    out = pd.DataFrame({
        "Open": df[c_open].map(_to_float),
        "High": df[c_high].map(_to_float),
        "Low": df[c_low].map(_to_float),
        "Close": df[c_close].map(_to_float),
        "Volume": df[c_vol].map(_to_float),
    }, index=pd.to_datetime(df["Date"]))

    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_twse_json(stock_id: str, months_back: int) -> Tuple[pd.DataFrame, List[FetchAttempt]]:
    attempts: List[FetchAttempt] = []
    all_df = []

    for d0 in _twse_month_list(months_back):
        date_str = f"{d0.year}{d0.month:02d}01"
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={stock_id}"
        status, j, err, snip = _req_json(url)
        if err:
            attempts.append(FetchAttempt("TWSE_JSON", "ERROR", url, status, err[:120]))
            continue
        if j is None:
            attempts.append(FetchAttempt("TWSE_JSON", "BAD_JSON", url, status, snip))
            continue
        if j.get("stat") != "OK":
            attempts.append(FetchAttempt("TWSE_JSON", f"STAT_{j.get('stat')}", url, status, snip))
            continue
        dfm = _parse_twse_stock_day_json(j)
        if dfm.empty:
            attempts.append(FetchAttempt("TWSE_JSON", "EMPTY_PARSED", url, status, snip))
            continue
        attempts.append(FetchAttempt("TWSE_JSON", "OK", url, status, ""))
        all_df.append(dfm)

        # 小睡避免太密集
        time.sleep(0.05)

    if all_df:
        df = pd.concat(all_df).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df, attempts
    return pd.DataFrame(), attempts

@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_yfinance(stock_id: str, months_back: int) -> Tuple[pd.DataFrame, List[FetchAttempt], str]:
    """
    回傳 df, attempts, source_tag
    source_tag: YF.TW / YF.TWO
    """
    attempts: List[FetchAttempt] = []
    # yfinance 在 Cloud 可能會被擋，但仍保留備援
    try:
        import yfinance as yf
    except Exception as e:
        attempts.append(FetchAttempt("YF", "NO_MODULE", "import yfinance", None, str(e)[:120]))
        return pd.DataFrame(), attempts, "YF"

    period_days = max(90, int(months_back * 30.5) + 10)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=period_days)

    # 先 .TW 再 .TWO
    for suf, tag in [(".TW", "YF.TW"), (".TWO", "YF.TWO")]:
        ticker = f"{stock_id}{suf}"
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                attempts.append(FetchAttempt("YF", "EMPTY", ticker, 200, ""))
                continue
            # yfinance 欄位可能是多層 or 小寫
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Adj Close": "Adj Close", "Volume": "Volume"
            })
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df = df.dropna(subset=["Close"])
            attempts.append(FetchAttempt("YF", "OK", ticker, 200, ""))
            return df, attempts, tag
        except Exception as e:
            attempts.append(FetchAttempt("YF", "ERROR", ticker, None, str(e)[:120]))
            continue

    return pd.DataFrame(), attempts, "YF"

def parse_user_csv(uploaded_file) -> Tuple[pd.DataFrame, List[FetchAttempt]]:
    attempts: List[FetchAttempt] = []
    if uploaded_file is None:
        return pd.DataFrame(), attempts
    try:
        raw = uploaded_file.getvalue()
        df = pd.read_csv(io.BytesIO(raw))
        # 允許常見欄位：Date/Open/High/Low/Close/Volume（大小寫不拘）
        cols = {c.lower(): c for c in df.columns}
        need = ["date", "open", "high", "low", "close", "volume"]
        if not all(k in cols for k in need):
            attempts.append(FetchAttempt("CSV", "BAD_COLUMNS", "uploaded", None, f"columns={list(df.columns)[:12]}"))
            return pd.DataFrame(), attempts

        out = pd.DataFrame({
            "Open": df[cols["open"]].map(_to_float),
            "High": df[cols["high"]].map(_to_float),
            "Low": df[cols["low"]].map(_to_float),
            "Close": df[cols["close"]].map(_to_float),
            "Volume": df[cols["volume"]].map(_to_float),
        }, index=pd.to_datetime(df[cols["date"]], errors="coerce"))

        out = out.dropna(subset=["Close"])
        out = out.sort_index()
        attempts.append(FetchAttempt("CSV", "OK", "uploaded", 200, ""))
        return out, attempts
    except Exception as e:
        attempts.append(FetchAttempt("CSV", "ERROR", "uploaded", None, str(e)[:140]))
        return pd.DataFrame(), attempts

def load_price_data(
    stock_id: str,
    months_back: int,
    uploaded_csv=None,
    enable_debug: bool = False
) -> Tuple[pd.DataFrame, str, List[FetchAttempt]]:
    """
    回傳：df, source_tag, attempts
    """
    stock_id = _clean_stock_id(stock_id)
    attempts: List[FetchAttempt] = []

    # 1) TWSE JSON（最穩）
    df, att = fetch_twse_json(stock_id, months_back=months_back)
    attempts += att
    if not df.empty:
        return df, "TWSE_JSON", attempts

    # 2) Yahoo（備援）
    df2, att2, tag = fetch_yfinance(stock_id, months_back=months_back)
    attempts += att2
    if not df2.empty:
        return df2, tag, attempts

    # 3) 使用者上傳 CSV（最後備援）
    df3, att3 = parse_user_csv(uploaded_csv)
    attempts += att3
    if not df3.empty:
        return df3, "CSV", attempts

    return pd.DataFrame(), "NONE", attempts


# -----------------------------
# Top10 掃描（去重 + 顯示操作判斷 + 距離%）
# -----------------------------
def scan_top10(
    stock_pool: List[str],
    months_back: int,
    max_buy_distance: float,
    uploaded_csv=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    回傳：
    - result_df: Top10 table
    - diag_df: diagnostics attempts
    """
    rows = []
    diag_rows = []

    # 去重（避免 Top10 出現同一檔重複）
    pool = _unique_keep_order([_clean_stock_id(x) for x in stock_pool if _clean_stock_id(x)])
    for sid in pool:
        if not _is_taiwan_stock_id(sid):
            continue

        df, src, attempts = load_price_data(sid, months_back, uploaded_csv=uploaded_csv)
        for a in attempts:
            diag_rows.append({
                "stock": sid,
                "source": a.source,
                "result": a.result,
                "url": a.url,
                "status": a.status,
                "snippet": a.snippet
            })

        if df.empty:
            continue

        dfi = add_indicators(df)
        z = compute_zones(dfi, max_buy_distance=max_buy_distance)

        last_date = dfi.index.max().strftime("%Y-%m-%d") if len(dfi) else ""
        curr = float(dfi["Close"].iloc[-1])

        rows.append({
            "股票": sid,
            "來源": src,
            "目前價": round(curr, 2),
            "操作判斷": z.action,
            "AI分數": z.score,
            "近端買點距離": (None if z.near_buy_dist is None else f"{z.near_buy_dist*100:.1f}%"),
            "深回檔距離": (None if z.deep_buy_dist is None else f"{z.deep_buy_dist*100:.1f}%"),
            "近端賣點距離": (None if z.near_sell_dist is None else f"{z.near_sell_dist*100:.1f}%"),
            "最後日期": last_date,
        })

    if not rows:
        return pd.DataFrame(), pd.DataFrame(diag_rows)

    result_df = pd.DataFrame(rows)

    # 分數排序 + 同分則優先距離較近的賣點（可操作）
    # 若欄位為 None/字串，先轉數值
    def pct_to_float(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).replace("%", "").strip()
        return _to_float(s) / 100.0

    result_df["_score"] = result_df["AI分數"].map(_to_float)
    result_df["_sell_dist"] = result_df["近端賣點距離"].map(pct_to_float)
    result_df = result_df.sort_values(by=["_score", "_sell_dist"], ascending=[False, True])

    # 只取前 10
    result_df = result_df.drop(columns=["_score", "_sell_dist"]).head(10).reset_index(drop=True)

    return result_df, pd.DataFrame(diag_rows)


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI 台股量化平台", layout="wide")

    st.title(APP_TITLE)
    st.caption("只做資訊顯示，不自動下單。Top10 先用小池測試（避免全市場在 Cloud 超時）。")

    with st.sidebar:
        st.subheader("選擇模式")
        mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0, label_visibility="collapsed")

        st.subheader("資料期間")
        period = st.selectbox("資料期間", list(PERIOD_MAP.keys()), index=1)
        months_back = PERIOD_MAP[period]

        enable_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

        max_buy_distance = st.slider(
            "可操作買點最大距離（避免買點離現實太遠）\nmax_buy_distance",
            min_value=0.05, max_value=0.30, value=0.12, step=0.01
        )

        st.markdown("---")
        st.markdown("🧪 **網路測試（建議先按一次）**")
        do_net_test = st.button("🧪 網路測試（建議先按一次）")

        st.markdown("---")
        uploaded_csv = st.file_uploader("（最後備援）上傳 CSV（Date,Open,High,Low,Close,Volume）", type=["csv"])

    if do_net_test:
        # 簡單測試 TWSE JSON 連線
        test_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20250101&stockNo=2330"
        status, j, err, snip = _req_json(test_url)
        if err:
            st.error(f"TWSE JSON 測試失敗：{err}")
        else:
            ok = (j is not None and isinstance(j, dict) and j.get("stat") == "OK")
            if ok:
                st.success("TWSE JSON 連線正常 ✅（Cloud 通常最穩）")
            else:
                st.warning(f"TWSE JSON 回應非 OK（可能暫時擁塞/封鎖） status={status} snip={snip}")

    if mode == "Top 10 掃描器":
        st.subheader("Top10 掃描器：先用小池測試（避免全市場在 Cloud 超時）。")

        stock_pool_text = st.text_area("stock_pool（逗號或換行分隔）", value=DEFAULT_POOL, height=140)
        stock_pool = re.split(r"[,\n]+", stock_pool_text.strip())
        stock_pool = [x.strip() for x in stock_pool if x.strip()]

        run = st.button("🔥 RUN Top10 掃描", type="primary")

        if run:
            with st.spinner("掃描中..."):
                result_df, diag_df = scan_top10(
                    stock_pool=stock_pool,
                    months_back=months_back,
                    max_buy_distance=max_buy_distance,
                    uploaded_csv=uploaded_csv
                )

            st.markdown("### 🔥 AI 強勢股 Top 10（含當下操作判斷 + 距離%）")
            if result_df.empty:
                st.warning("目前掃描結果為空（代表資料下載失敗）。請稍後再試或換 stock_pool。")
            else:
                st.dataframe(result_df, use_container_width=True)

                st.markdown("### 🔎 點選一檔直接展開完整分析")
                pick = st.selectbox("選擇股票", result_df["股票"].tolist(), index=0)
                render_single_stock(
                    pick, months_back=months_back, max_buy_distance=max_buy_distance,
                    uploaded_csv=uploaded_csv, enable_debug=enable_debug
                )

            if enable_debug and (diag_df is not None) and (not diag_df.empty):
                st.markdown("### 🧩 逐路診斷（哪一路失敗、為什麼）")
                st.dataframe(diag_df, use_container_width=True)

        return

    # 單一股票
    st.subheader("單一股票分析")
    stock_id = st.text_input("請輸入股票代號", value="2330")

    render_single_stock(
        stock_id, months_back=months_back, max_buy_distance=max_buy_distance,
        uploaded_csv=uploaded_csv, enable_debug=enable_debug
    )


def render_single_stock(
    stock_id: str,
    months_back: int,
    max_buy_distance: float,
    uploaded_csv=None,
    enable_debug: bool = False
):
    sid = _clean_stock_id(stock_id)
    if not _is_taiwan_stock_id(sid):
        st.error("請輸入台股代號（例如：2330 / 2317 / 00878）")
        return

    with st.spinner("下載資料中..."):
        df, src, attempts = load_price_data(sid, months_back, uploaded_csv=uploaded_csv, enable_debug=enable_debug)

    if df.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。")
        if enable_debug and attempts:
            st.markdown("### 🧩 逐路診斷（哪一路失敗、為什麼）")
            diag_df = pd.DataFrame([{
                "source": a.source, "result": a.result, "url": a.url, "status": a.status, "snippet": a.snippet
            } for a in attempts])
            st.dataframe(diag_df, use_container_width=True)
        return

    dfi = add_indicators(df)
    z = compute_zones(dfi, max_buy_distance=max_buy_distance)

    last = dfi.iloc[-1]
    curr = float(last["Close"])
    last_date = dfi.index.max().strftime("%Y-%m-%d")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{curr:.2f}")
    c2.metric("AI 共振分數", f"{z.score}/100")
    c3.metric("資料來源", src)
    c4.metric("最後日期 / 筆數", f"{last_date} / {len(dfi)}")

    st.markdown("## 📌 當下是否為買點/賣點？（可操作判斷）")
    if z.action == "BUY":
        st.success(f"✅ BUY：{z.reason}")
    elif z.action == "SELL":
        st.warning(f"⚠️ SELL：{z.reason}")
    else:
        st.info(f"⏳ WATCH：{z.reason}")

    # 若近端買點被隱藏，明確寫出原因（對應你 PDF 第 3 頁的敘述邏輯）:contentReference[oaicite:1]{index=1}
    st.markdown("## 🗺️ 未來預估買賣點（區間 + 距離%）")

    if z.near_buy is None:
        st.warning("⚠️ 近端買點：距離過遠或不可操作（已自動隱藏）")
    else:
        lo, hi = z.near_buy
        dist = z.near_buy_dist or np.nan
        st.success(f"🟢 近端買點（可操作）：{lo:.2f} ~ {hi:.2f}（距離現價：約 {dist*100:.1f}%）")

    if z.deep_buy:
        lo, hi = z.deep_buy
        dist = z.deep_buy_dist or np.nan
        st.info(f"🟦 深回檔買點（等待型）：{lo:.2f} ~ {hi:.2f}（距離現價：約 {dist*100:.1f}%）")

    if z.near_sell:
        lo, hi = z.near_sell
        dist = z.near_sell_dist or np.nan
        st.warning(f"🎯 近端賣點區（壓力/獲利）：{lo:.2f} ~ {hi:.2f}（距離現價：約 {dist*100:.1f}%）")

    st.markdown("## 📉 布林通道分析圖（K線 + BB + EMA20 + Volume）")
    fig = plot_bollinger_candles(dfi, title=f"{sid} | {src}")
    st.pyplot(fig, clear_figure=True)

    with st.expander("📄 最近 20 筆資料（含指標）", expanded=False):
        tail_cols = ["Open", "High", "Low", "Close", "Volume", "EMA20", "EMA60", "BB_upper", "BB_mid", "BB_lower", "MACD", "MACD_signal", "MACD_hist", "K", "D", "RSI14", "Bias20"]
        show = dfi[tail_cols].tail(20).copy()
        st.dataframe(show, use_container_width=True)

    if enable_debug and attempts:
        st.markdown("## 🧩 逐路診斷（哪一路成功/失敗、為什麼）")
        diag_df = pd.DataFrame([{
            "source": a.source, "result": a.result, "url": a.url, "status": a.status, "snippet": a.snippet
        } for a in attempts])
        st.dataframe(diag_df, use_container_width=True)


if __name__ == "__main__":
    main()
