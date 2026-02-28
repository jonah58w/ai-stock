# app.py
# AI 台股量化專業平台（無 Plotly / 全功能保留 / 多源備援 + 逐路診斷 + Top10 + 指標共振 + 布林通道圖）
# - 資料源備援：Yahoo(yfinance) -> Stooq(pandas-datareader) -> TWSE JSON -> TWSE CSV -> TPEX JSON/CSV -> CSV上傳
# - 加強：HTTP retry/backoff、WAF/HTML/空回傳辨識、缺套件提示（requirements）
# - 指標：SMA/EMA/RSI/MACD/KD/ATR/Bollinger/Volume
# - 策略：回檔等待型（分批） / 趨勢突破型（突破追價進場）
# - Top10：避免重複 ticker、顯示當下可操作判斷 + 交易計畫摘要
# - 圖表：matplotlib（Close + Bollinger + SMA/EMA + 重要點位）

from __future__ import annotations

import math
import time
import json
import re
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import requests


# -----------------------------
# Optional imports (with hints)
# -----------------------------
HAS_YFINANCE = True
HAS_PDR = True
try:
    import yfinance as yf
except Exception:
    HAS_YFINANCE = False

try:
    from pandas_datareader import data as pdr
except Exception:
    HAS_PDR = False


# -----------------------------
# UI / Page
# -----------------------------
st.set_page_config(
    page_title="AI 台股量化專業平台（無 Plotly / 全功能保留）",
    layout="wide",
)

st.title("🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）")
st.caption("多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + Top10 掃描 + 交易計畫（不自動下單）")


# -----------------------------
# Helpers
# -----------------------------
def now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def to_date_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def is_html_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # crude WAF/HTML detection
    return ("<html" in t) or ("<!doctype" in t) or ("<head" in t) or ("cloudflare" in t) or ("access denied" in t)


def compact_preview(text: str, n: int = 140) -> str:
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    return (s[:n] + "…") if len(s) > n else s


@dataclass
class FetchDiag:
    source: str
    result: str
    url: str
    status: Optional[int] = None
    reason: str = ""
    preview: str = ""


def make_retry_session(
    timeout_sec: float = 8.0,
    max_tries: int = 3,
    backoff_sec: float = 0.8,
    headers: Optional[dict] = None,
):
    """
    Simple manual retry wrapper (works in Streamlit Cloud reliably).
    """
    sess = requests.Session()
    base_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIStockBot/1.0; +https://streamlit.io)",
        "Accept": "*/*",
    }
    if headers:
        base_headers.update(headers)

    def get(url: str) -> Tuple[Optional[requests.Response], List[FetchDiag]]:
        diags: List[FetchDiag] = []
        last_exc = None
        for i in range(max_tries):
            try:
                resp = sess.get(url, headers=base_headers, timeout=timeout_sec)
                # Some endpoints return 200 with empty/HTML
                txt = resp.text if resp is not None else ""
                if resp.status_code >= 500:
                    diags.append(FetchDiag(
                        source="HTTP",
                        result="RETRY",
                        url=url,
                        status=resp.status_code,
                        reason=f"Server error {resp.status_code} (try {i+1}/{max_tries})",
                        preview=compact_preview(txt),
                    ))
                    time.sleep(backoff_sec * (i + 1))
                    continue
                return resp, diags
            except Exception as e:
                last_exc = e
                diags.append(FetchDiag(
                    source="HTTP",
                    result="RETRY",
                    url=url,
                    status=None,
                    reason=f"Exception: {type(e).__name__}: {e} (try {i+1}/{max_tries})",
                    preview="",
                ))
                time.sleep(backoff_sec * (i + 1))

        # failed
        diags.append(FetchDiag(
            source="HTTP",
            result="FAIL",
            url=url,
            status=None,
            reason=f"All retries failed: {type(last_exc).__name__}: {last_exc}" if last_exc else "All retries failed",
            preview="",
        ))
        return None, diags

    return get


# -----------------------------
# Technical Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 9, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI14"] = rsi(df["Close"], 14)
    m, s, h = macd(df["Close"])
    df["MACD"] = m
    df["MACD_SIGNAL"] = s
    df["MACD_HIST"] = h
    k, d = stochastic_kd(df["High"], df["Low"], df["Close"])
    df["K"] = k
    df["D"] = d
    df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)
    up, mid, low = bollinger(df["Close"], 20, 2.0)
    df["BB_UP"] = up
    df["BB_MID"] = mid
    df["BB_LOW"] = low
    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
    return df


# -----------------------------
# Data Normalization
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: Date index (datetime), Open/High/Low/Close/Volume float.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # If Date is a column, set as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].sort_index()

    # rename common variants
    colmap = {c: c.strip().title() for c in df.columns}
    df.rename(columns=colmap, inplace=True)

    # handle Adj Close
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    required = ["Open", "High", "Low", "Close"]
    for c in required:
        if c not in df.columns:
            # try lowercase
            if c.lower() in df.columns:
                df.rename(columns={c.lower(): c}, inplace=True)

    if "Volume" not in df.columns and "Vol" in df.columns:
        df.rename(columns={"Vol": "Volume"}, inplace=True)

    # coerce numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    # drop duplicates
    df = df[~df.index.duplicated(keep="last")]
    return df


# -----------------------------
# Sources: Yahoo / Stooq / TWSE / TPEX / CSV upload
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_yfinance(symbol: str, period: str) -> Tuple[pd.DataFrame, List[FetchDiag]]:
    diags: List[FetchDiag] = []
    if not HAS_YFINANCE:
        diags.append(FetchDiag(
            source="YFINANCE",
            result="MISSING_PKG",
            url="",
            reason="yfinance not installed. Add to requirements.txt: yfinance lxml html5lib",
        ))
        return pd.DataFrame(), diags

    try:
        # yfinance period mapping
        yf_period = {"3mo": "3mo", "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y"}.get(period, "6mo")
        df = yf.download(symbol, period=yf_period, interval="1d", auto_adjust=False, progress=False)
        df = normalize_ohlcv(df)
        if df.empty or len(df) < 30:
            diags.append(FetchDiag(source="YFINANCE", result="TOO_SHORT", url=symbol, reason="Downloaded data too short/empty", preview=""))
            return pd.DataFrame(), diags
        diags.append(FetchDiag(source="YFINANCE", result="OK", url=symbol, status=200, reason=f"rows={len(df)}"))
        return df, diags
    except Exception as e:
        diags.append(FetchDiag(source="YFINANCE", result="FAIL", url=symbol, reason=f"{type(e).__name__}: {e}"))
        return pd.DataFrame(), diags


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stooq(symbol: str, period: str) -> Tuple[pd.DataFrame, List[FetchDiag]]:
    diags: List[FetchDiag] = []
    if not HAS_PDR:
        diags.append(FetchDiag(
            source="STOOQ",
            result="MISSING_PKG",
            url="",
            reason="pandas-datareader not installed. Add to requirements.txt: pandas-datareader",
        ))
        return pd.DataFrame(), diags

    try:
        # Stooq expects e.g. 2330.TW ; if not, still try
        df = pdr.DataReader(symbol, "stooq")
        df = df.sort_index()
        df = normalize_ohlcv(df)

        # cut by period
        days_map = {"3mo": 80, "6mo": 160, "1y": 260, "2y": 520, "5y": 1300}
        need = days_map.get(period, 160)
        if len(df) > need:
            df = df.iloc[-need:]

        if df.empty or len(df) < 30:
            diags.append(FetchDiag(source="STOOQ", result="TOO_SHORT", url=symbol, reason="Downloaded data too short/empty"))
            return pd.DataFrame(), diags

        diags.append(FetchDiag(source="STOOQ", result="OK", url=symbol, status=200, reason=f"rows={len(df)}"))
        return df, diags
    except Exception as e:
        diags.append(FetchDiag(source="STOOQ", result="FAIL", url=symbol, reason=f"{type(e).__name__}: {e}"))
        return pd.DataFrame(), diags


def twse_json_url(stock_no: str, yyyymmdd: str) -> str:
    return f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={yyyymmdd}&stockNo={stock_no}"


def twse_csv_url(stock_no: str, yyyymmdd: str) -> str:
    return f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date={yyyymmdd}&stockNo={stock_no}"


def tpex_json_url(stock_no: str, roc_yyy_mm_dd: str) -> str:
    # example d=114/02/27
    return f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={roc_yyy_mm_dd}&stkno={stock_no}&response=json"


def roc_date_str(d: dt.date) -> str:
    # ROC year = AD - 1911
    y = d.year - 1911
    return f"{y:03d}/{d.month:02d}/{d.day:02d}"


def parse_twse_json(obj: dict) -> pd.DataFrame:
    # obj['data'] rows: [date, volume, amount, open, high, low, close, change, transactions]
    if not obj or "data" not in obj or not obj["data"]:
        return pd.DataFrame()
    rows = obj["data"]
    out = []
    for r in rows:
        try:
            # date like "114/02/27"
            date_str = r[0]
            parts = date_str.split("/")
            roc_y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
            ad_y = roc_y + 1911
            date = dt.date(ad_y, m, d)
            o = float(str(r[3]).replace(",", ""))
            h = float(str(r[4]).replace(",", ""))
            l = float(str(r[5]).replace(",", ""))
            c = float(str(r[6]).replace(",", ""))
            v = float(str(r[1]).replace(",", ""))
            out.append([date, o, h, l, c, v])
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return normalize_ohlcv(df)


def parse_twse_csv(text: str) -> pd.DataFrame:
    # CSV contains header lines; data lines include ROC date and prices
    if not text or len(text.strip()) < 50:
        return pd.DataFrame()
    if is_html_text(text):
        return pd.DataFrame()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # data rows often start with "114/02/03"
    data_lines = [ln for ln in lines if re.match(r"^\d{3}/\d{2}/\d{2}", ln)]
    if not data_lines:
        return pd.DataFrame()

    out = []
    for ln in data_lines:
        parts = [p.strip().strip('"') for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            date_str = parts[0]
            roc_y, m, d = [int(x) for x in date_str.split("/")]
            ad_y = roc_y + 1911
            date = dt.date(ad_y, m, d)
            o = float(parts[3].replace(",", ""))
            h = float(parts[4].replace(",", ""))
            l = float(parts[5].replace(",", ""))
            c = float(parts[6].replace(",", ""))
            v = float(parts[1].replace(",", ""))
            out.append([date, o, h, l, c, v])
        except Exception:
            continue

    df = pd.DataFrame(out, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return normalize_ohlcv(df)


def parse_tpex_json(obj: dict) -> pd.DataFrame:
    # TPEX returns "aaData" or "data" depending on endpoint version
    rows = None
    for key in ["aaData", "data"]:
        if isinstance(obj, dict) and key in obj and obj[key]:
            rows = obj[key]
            break
    if not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        # commonly: [date, close, change, open, high, low, volume, ...]
        try:
            date_str = r[0]  # "114/02/27"
            roc_y, m, d = [int(x) for x in date_str.split("/")]
            ad_y = roc_y + 1911
            date = dt.date(ad_y, m, d)

            c = float(str(r[1]).replace(",", ""))
            o = float(str(r[3]).replace(",", ""))
            h = float(str(r[4]).replace(",", ""))
            l = float(str(r[5]).replace(",", ""))
            v = float(str(r[6]).replace(",", ""))
            out.append([date, o, h, l, c, v])
        except Exception:
            continue

    df = pd.DataFrame(out, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return normalize_ohlcv(df)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_twse_monthly(stock_no: str, period: str) -> Tuple[pd.DataFrame, List[FetchDiag]]:
    get = make_retry_session(timeout_sec=8, max_tries=3, backoff_sec=0.7)
    diags: List[FetchDiag] = []
    days_map = {"3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60}
    months = days_map.get(period, 6)

    today = dt.date.today()
    # fetch month by month (from oldest to newest)
    frames = []
    for i in range(months, -1, -1):
        d = (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)  # last month first day
        # walk back i months from current month
        mm = today.month - i
        yy = today.year
        while mm <= 0:
            mm += 12
            yy -= 1
        while mm > 12:
            mm -= 12
            yy += 1
        month_date = dt.date(yy, mm, 1)
        yyyymmdd = to_date_yyyymmdd(month_date)

        # JSON first
        url = twse_json_url(stock_no, yyyymmdd)
        resp, retry_diags = get(url)
        diags.extend([FetchDiag(source="TWSE_JSON", result=d.result, url=url, status=d.status, reason=d.reason, preview=d.preview) for d in retry_diags])

        if resp is None:
            diags.append(FetchDiag(source="TWSE_JSON", result="FAIL", url=url, reason="No response"))
        else:
            txt = resp.text or ""
            if is_html_text(txt):
                diags.append(FetchDiag(source="TWSE_JSON", result="WAF_HTML", url=url, status=resp.status_code, reason="HTML/WAF returned", preview=compact_preview(txt)))
            else:
                try:
                    obj = resp.json()
                    df = parse_twse_json(obj)
                    if not df.empty:
                        frames.append(df)
                        diags.append(FetchDiag(source="TWSE_JSON", result="OK", url=url, status=resp.status_code, reason=f"rows={len(df)}"))
                        continue
                    else:
                        diags.append(FetchDiag(source="TWSE_JSON", result="EMPTY_TEXT", url=url, status=resp.status_code, reason="JSON parsed but no data", preview=compact_preview(txt)))
                except Exception as e:
                    diags.append(FetchDiag(source="TWSE_JSON", result="PARSE_FAIL", url=url, status=resp.status_code, reason=f"{type(e).__name__}: {e}", preview=compact_preview(txt)))

        # fallback CSV
        url2 = twse_csv_url(stock_no, yyyymmdd)
        resp2, retry_diags2 = get(url2)
        diags.extend([FetchDiag(source="TWSE_CSV", result=d.result, url=url2, status=d.status, reason=d.reason, preview=d.preview) for d in retry_diags2])

        if resp2 is None:
            diags.append(FetchDiag(source="TWSE_CSV", result="FAIL", url=url2, reason="No response"))
            continue

        txt2 = resp2.text or ""
        if is_html_text(txt2):
            diags.append(FetchDiag(source="TWSE_CSV", result="WAF_HTML", url=url2, status=resp2.status_code, reason="HTML/WAF returned", preview=compact_preview(txt2)))
            continue

        df2 = parse_twse_csv(txt2)
        if not df2.empty:
            frames.append(df2)
            diags.append(FetchDiag(source="TWSE_CSV", result="OK", url=url2, status=resp2.status_code, reason=f"rows={len(df2)}"))
        else:
            diags.append(FetchDiag(source="TWSE_CSV", result="EMPTY_TEXT", url=url2, status=resp2.status_code, reason="CSV empty/invalid", preview=compact_preview(txt2)))

    if not frames:
        return pd.DataFrame(), diags

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    # keep last N
    need_days = {"3mo": 80, "6mo": 160, "1y": 260, "2y": 520, "5y": 1300}.get(period, 160)
    if len(df_all) > need_days:
        df_all = df_all.iloc[-need_days:]
    return df_all, diags


@st.cache_data(ttl=600, show_spinner=False)
def fetch_tpex_monthly(stock_no: str, period: str) -> Tuple[pd.DataFrame, List[FetchDiag]]:
    get = make_retry_session(timeout_sec=8, max_tries=3, backoff_sec=0.7)
    diags: List[FetchDiag] = []
    months_map = {"3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60}
    months = months_map.get(period, 6)

    today = dt.date.today()
    frames = []
    for i in range(months, -1, -1):
        mm = today.month - i
        yy = today.year
        while mm <= 0:
            mm += 12
            yy -= 1
        month_date = dt.date(yy, mm, 1)
        roc = roc_date_str(month_date)
        url = tpex_json_url(stock_no, roc)

        resp, retry_diags = get(url)
        diags.extend([FetchDiag(source="TPEX_JSON", result=d.result, url=url, status=d.status, reason=d.reason, preview=d.preview) for d in retry_diags])

        if resp is None:
            diags.append(FetchDiag(source="TPEX_JSON", result="FAIL", url=url, reason="No response"))
            continue

        txt = resp.text or ""
        if is_html_text(txt):
            diags.append(FetchDiag(source="TPEX_JSON", result="WAF_HTML", url=url, status=resp.status_code, reason="HTML/WAF returned", preview=compact_preview(txt)))
            continue

        try:
            obj = resp.json()
            df = parse_tpex_json(obj)
            if not df.empty:
                frames.append(df)
                diags.append(FetchDiag(source="TPEX_JSON", result="OK", url=url, status=resp.status_code, reason=f"rows={len(df)}"))
            else:
                diags.append(FetchDiag(source="TPEX_JSON", result="EMPTY_TEXT", url=url, status=resp.status_code, reason="JSON parsed but no data", preview=compact_preview(txt)))
        except Exception as e:
            diags.append(FetchDiag(source="TPEX_JSON", result="PARSE_FAIL", url=url, status=resp.status_code, reason=f"{type(e).__name__}: {e}", preview=compact_preview(txt)))

    if not frames:
        return pd.DataFrame(), diags

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    need_days = {"3mo": 80, "6mo": 160, "1y": 260, "2y": 520, "5y": 1300}.get(period, 160)
    if len(df_all) > need_days:
        df_all = df_all.iloc[-need_days:]
    return df_all, diags


def parse_uploaded_csv(uploaded: Any) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()

    # Try common formats
    # expected: Date/Open/High/Low/Close/Volume
    if "Date" not in df.columns:
        # sometimes first column unnamed date
        if df.columns[0].lower() in ["date", "datetime"] or "date" in df.columns[0].lower():
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    return normalize_ohlcv(df)


# -----------------------------
# Symbol helpers
# -----------------------------
def normalize_ticker_input(raw: str) -> str:
    r = (raw or "").strip().upper()
    r = r.replace(" ", "")
    return r


def candidate_symbols(ticker: str) -> List[str]:
    """
    For Taiwan stocks: try .TW then .TWO if numeric 4-5 digits.
    Accept if user already provided suffix.
    """
    t = normalize_ticker_input(ticker)
    if not t:
        return []
    if "." in t:
        return [t]
    if t.isdigit():
        return [f"{t}.TW", f"{t}.TWO", t]  # yfinance/stooq sometimes accept plain
    return [t]


# -----------------------------
# Scoring & Strategy logic
# -----------------------------
def resonance_score(latest: pd.Series) -> int:
    """
    0..100 simple composite score based on multi-indicator alignment.
    """
    score = 0

    # Trend
    if pd.notna(latest.get("SMA20")) and pd.notna(latest.get("EMA20")):
        if latest["Close"] > latest["SMA20"]:
            score += 10
        if latest["EMA20"] > latest["SMA20"]:
            score += 10

    # Momentum
    if pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_SIGNAL")):
        if latest["MACD"] > latest["MACD_SIGNAL"]:
            score += 10
        if latest.get("MACD_HIST", 0) > 0:
            score += 5

    # RSI / KD (avoid extremes)
    r = latest.get("RSI14")
    if pd.notna(r):
        if 40 <= r <= 65:
            score += 10
        elif r < 35:
            score += 6
        elif r > 70:
            score += 3

    k = latest.get("K")
    d = latest.get("D")
    if pd.notna(k) and pd.notna(d):
        if k > d:
            score += 10
        if k < 25:
            score += 6
        if k > 80:
            score += 3

    # Bollinger position
    bb_low = latest.get("BB_LOW")
    bb_up = latest.get("BB_UP")
    if pd.notna(bb_low) and pd.notna(bb_up):
        if latest["Close"] <= bb_low * 1.02:
            score += 12
        if latest["Close"] >= bb_up * 0.98:
            score += 12

    # Volume confirmation
    if pd.notna(latest.get("VOL_SMA20")) and pd.notna(latest.get("Volume")):
        if latest["Volume"] > latest["VOL_SMA20"]:
            score += 8

    return int(min(100, max(0, score)))


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def premium_discount(current: float, zone_mid: float) -> float:
    # + = premium (current above zone), - = discount (current below zone)
    if zone_mid <= 0:
        return np.nan
    return (current / zone_mid) - 1.0


def professional_distance_label(current: float, z1: float, z2: float) -> str:
    lo, hi = (min(z1, z2), max(z1, z2))
    mid = (lo + hi) / 2
    pdiff = premium_discount(current, mid)
    if np.isnan(pdiff):
        return "—"
    # professional wording
    if pdiff > 0:
        return f"相對區間中值溢價：+{abs(pdiff)*100:.1f}%（現價高於觸發區）"
    else:
        return f"相對區間中值折價：-{abs(pdiff)*100:.1f}%（現價低於觸發區）"


def compute_trade_zones(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute buy/sell zones based on Bollinger + ATR + local structure.
    """
    latest = df.iloc[-1]
    close = float(latest["Close"])
    atr14 = float(latest.get("ATR14", np.nan))
    bb_low = float(latest.get("BB_LOW", np.nan))
    bb_mid = float(latest.get("BB_MID", np.nan))
    bb_up = float(latest.get("BB_UP", np.nan))

    # local support/resistance
    lookback = min(60, len(df))
    recent = df.iloc[-lookback:]
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    # near-term buy zone: max(support, BB_LOW) with a small band width tied to ATR
    if np.isnan(atr14) or atr14 <= 0:
        band = close * 0.015
    else:
        band = max(atr14 * 0.6, close * 0.01)

    buy_anchor = np.nanmax([support, bb_low]) if not (np.isnan(bb_low) and np.isnan(support)) else close * 0.95
    near_buy_lo = buy_anchor
    near_buy_hi = buy_anchor + band

    # deep buy zone: below BB_LOW (value area)
    deep_buy_lo = max(0.0, (bb_low - band * 1.6) if not np.isnan(bb_low) else close * 0.88)
    deep_buy_hi = max(deep_buy_lo, (bb_low - band * 0.6) if not np.isnan(bb_low) else close * 0.92)

    # sell zone near BB_UP / resistance
    sell_anchor = np.nanmin([resistance, bb_up]) if not (np.isnan(bb_up) and np.isnan(resistance)) else close * 1.05
    near_sell_lo = max(0.0, sell_anchor - band)
    near_sell_hi = sell_anchor

    # aggressive breakout sell / take profit
    ext_sell_lo = max(0.0, sell_anchor)
    ext_sell_hi = sell_anchor + band * 1.5

    return {
        "close": close,
        "atr": atr14,
        "bb_low": bb_low,
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "support": support,
        "resistance": resistance,
        "near_buy": (near_buy_lo, near_buy_hi),
        "deep_buy": (deep_buy_lo, deep_buy_hi),
        "near_sell": (near_sell_lo, near_sell_hi),
        "ext_sell": (ext_sell_lo, ext_sell_hi),
    }


def strategy_pullback(df: pd.DataFrame, max_buy_distance: float) -> Dict[str, Any]:
    """
    回檔等待型（分批）：偏向「等回到區間」再分批買
    """
    latest = df.iloc[-1]
    zones = compute_trade_zones(df)
    close = zones["close"]
    near_buy_lo, near_buy_hi = zones["near_buy"]
    deep_buy_lo, deep_buy_hi = zones["deep_buy"]

    r = latest.get("RSI14", np.nan)
    k = latest.get("K", np.nan)
    d = latest.get("D", np.nan)

    # condition: near buy zone + oversold confirmation
    in_near = (close >= near_buy_lo) and (close <= near_buy_hi)
    in_deep = (close >= deep_buy_lo) and (close <= deep_buy_hi)

    oversold = (pd.notna(r) and r < 38) or (pd.notna(k) and k < 25) or (pd.notna(d) and d < 25)
    macd_turn = pd.notna(latest.get("MACD_HIST")) and latest.get("MACD_HIST") > (df["MACD_HIST"].iloc[-2] if len(df) >= 2 else -999)

    # distance filter: avoid "buy zone too far from current"
    # We enforce that near_buy mid should not be too far away from current (max_buy_distance)
    mid = (near_buy_lo + near_buy_hi) / 2
    dist = abs((close / mid) - 1.0) if mid > 0 else 999

    if dist > max_buy_distance and not in_near and not in_deep:
        action = "WATCH"
        reason = "買區距離過遠（已超過可接受距離門檻），等待回檔或重新形成區間。"
    else:
        if in_deep and oversold:
            action = "BUY"
            reason = "進入深回檔價值區 + 超賣條件成立，適合分批布局。"
        elif in_near and (oversold or macd_turn):
            action = "BUY"
            reason = "進入近端買區 + 指標出現超賣/動能改善，具可操作性。"
        else:
            action = "WATCH"
            reason = "條件尚未明確，等待價格進入買區或指標翻轉確認。"

    return {
        "strategy": "回檔等待型（分批）",
        "action": action,
        "reason": reason,
        "zones": zones,
    }


def strategy_breakout(df: pd.DataFrame, chase_max: float, breakout_atr_buffer: float, stop_atr: float, rr: float) -> Dict[str, Any]:
    """
    趨勢突破型（突破追價進場）：突破 + 跟隨，控制追價距離
    """
    zones = compute_trade_zones(df)
    latest = df.iloc[-1]
    close = zones["close"]
    atr14 = zones["atr"]

    # breakout reference: recent 20-day high or BB upper
    look = min(60, len(df))
    recent = df.iloc[-look:]
    high20 = float(recent["High"].rolling(20).max().iloc[-1]) if len(recent) >= 20 else float(recent["High"].max())
    bb_up = zones["bb_up"]
    ref = np.nanmax([high20, bb_up]) if not (np.isnan(high20) and np.isnan(bb_up)) else high20

    buffer = (atr14 * breakout_atr_buffer) if (pd.notna(atr14) and atr14 > 0) else (close * 0.01)
    trigger = ref + buffer

    # chase distance: how far above trigger
    chase_dist = (close / trigger) - 1.0 if trigger > 0 else 999

    macd_ok = pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_SIGNAL")) and latest["MACD"] > latest["MACD_SIGNAL"]
    vol_ok = pd.notna(latest.get("Volume")) and pd.notna(latest.get("VOL_SMA20")) and (latest["Volume"] >= latest["VOL_SMA20"])

    if close >= trigger and chase_dist <= chase_max and (macd_ok or vol_ok):
        action = "BUY"
        reason = "突破觸發價成立 + 追價距離在可控範圍內（且動能/量能至少一項確認）。"
    elif close >= trigger and chase_dist > chase_max:
        action = "WATCH"
        reason = "已突破但追價距離過大（風險不對稱），等待回測不破或重新給進場點。"
    else:
        action = "WATCH"
        reason = "尚未突破觸發價，等待突破成立或回測結構。"

    # risk plan (if breakout entry)
    if pd.notna(atr14) and atr14 > 0:
        stop = close - atr14 * stop_atr
    else:
        stop = close * 0.97
    target = close + (close - stop) * rr

    return {
        "strategy": "趨勢突破型（突破追價進場）",
        "action": action,
        "reason": reason,
        "trigger": trigger,
        "chase_dist": chase_dist,
        "stop": stop,
        "target": target,
        "zones": zones,
    }


# -----------------------------
# Visualization (Matplotlib)
# -----------------------------
def plot_bollinger(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = df.index
    ax.plot(x, df["Close"], linewidth=1.6, label="Close")
    if "BB_MID" in df.columns:
        ax.plot(x, df["BB_MID"], linewidth=1.0, label="BB Mid (SMA20)")
    if "BB_UP" in df.columns and "BB_LOW" in df.columns:
        ax.plot(x, df["BB_UP"], linewidth=1.0, label="BB Upper")
        ax.plot(x, df["BB_LOW"], linewidth=1.0, label="BB Lower")
        ax.fill_between(x, df["BB_LOW"].values, df["BB_UP"].values, alpha=0.12)

    if "EMA20" in df.columns:
        ax.plot(x, df["EMA20"], linewidth=1.0, label="EMA20", linestyle="--")
    if "SMA20" in df.columns:
        ax.plot(x, df["SMA20"], linewidth=1.0, label="SMA20", linestyle=":")

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    st.pyplot(fig)


# -----------------------------
# Main fetch orchestration
# -----------------------------
def fetch_with_fallback(ticker: str, period: str, uploaded_csv: Optional[Any]) -> Tuple[pd.DataFrame, str, List[FetchDiag]]:
    """
    Returns df, source_name, diagnostics
    """
    diags: List[FetchDiag] = []
    t = normalize_ticker_input(ticker)
    if not t:
        return pd.DataFrame(), "N/A", [FetchDiag(source="INPUT", result="EMPTY", url="", reason="Empty ticker")]

    # 0) CSV uploaded overrides if provided
    if uploaded_csv is not None:
        df_csv = parse_uploaded_csv(uploaded_csv)
        if not df_csv.empty:
            diags.append(FetchDiag(source="UPLOAD_CSV", result="OK", url="uploaded", status=200, reason=f"rows={len(df_csv)}"))
            return df_csv, "UPLOAD_CSV", diags
        else:
            diags.append(FetchDiag(source="UPLOAD_CSV", result="FAIL", url="uploaded", reason="CSV parsed empty or missing columns"))

    # 1) yfinance (try .TW then .TWO)
    for sym in candidate_symbols(t):
        df_yf, d = fetch_yfinance(sym, period)
        diags.extend(d)
        if not df_yf.empty:
            return df_yf, f"YF:{sym}", diags

    # 2) stooq
    for sym in candidate_symbols(t):
        df_sq, d = fetch_stooq(sym, period)
        diags.extend(d)
        if not df_sq.empty:
            return df_sq, f"STOOQ:{sym}", diags

    # If ticker numeric, try TWSE/TPEX official
    if t.isdigit():
        # 3) TWSE
        df_twse, d1 = fetch_twse_monthly(t, period)
        diags.extend(d1)
        if not df_twse.empty:
            return df_twse, "TWSE_OFFICIAL", diags

        # 4) TPEX
        df_tpex, d2 = fetch_tpex_monthly(t, period)
        diags.extend(d2)
        if not df_tpex.empty:
            return df_tpex, "TPEX_OFFICIAL", diags

    return pd.DataFrame(), "ALL_FAILED", diags


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("設定")

    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    period = st.selectbox("資料期間", ["3mo", "6mo", "1y", "2y", "5y"], index=1)

    debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.divider()
    st.subheader("策略 / 參數")

    strategy_choice = st.radio(
        "策略",
        ["回檔等待型（回檔分批）", "趨勢突破型（突破追價進場）"],
        index=0,
    )

    if strategy_choice.startswith("回檔"):
        max_buy_distance = st.slider(
            "可操作買點最大距離（避免買點離現實太遠） max_buy_distance",
            min_value=0.03, max_value=0.25, value=0.12, step=0.01,
        )
        chase_max = None
        breakout_atr_buffer = None
        stop_atr = None
        rr = None
    else:
        chase_max = st.slider(
            "最大可接受追價/偏離距離（%）",
            min_value=0.02, max_value=0.20, value=0.06, step=0.01,
        )
        breakout_atr_buffer = st.slider(
            "突破觸發 buffer（ATR 倍數）",
            min_value=0.00, max_value=1.50, value=0.20, step=0.05,
        )
        stop_atr = st.slider(
            "失效止損距離（ATR 倍數）",
            min_value=0.80, max_value=4.00, value=1.60, step=0.10,
        )
        rr = st.slider(
            "目標風險報酬（RR）",
            min_value=1.0, max_value=5.0, value=2.2, step=0.1,
        )
        max_buy_distance = None

    st.divider()
    st.subheader("備援 / 測試")
    network_test = st.button("🧪 網路測試（建議先按一次）")

    st.caption("提示：若 Cloud 偶發抓不到資料，可先按「網路測試」，或直接上傳 export.csv 立即可用。")


# -----------------------------
# Network test
# -----------------------------
if network_test:
    st.info("正在測試連線…（TWSE / TPEX / Yahoo）")
    get = make_retry_session(timeout_sec=6, max_tries=2, backoff_sec=0.5)

    test_urls = [
        ("TWSE", "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20250101&stockNo=2330"),
        ("TPEX", "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d=114/01/02&stkno=6274&response=json"),
        ("YAHOO", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=2330.TW"),
    ]

    rows = []
    for name, url in test_urls:
        resp, d = get(url)
        if resp is None:
            rows.append([name, "FAIL", None, "no response", ""])
        else:
            txt = resp.text or ""
            if is_html_text(txt):
                rows.append([name, "WAF_HTML", resp.status_code, "HTML/WAF returned", compact_preview(txt)])
            else:
                rows.append([name, "OK", resp.status_code, "reachable", compact_preview(txt)])
    st.dataframe(pd.DataFrame(rows, columns=["target", "result", "status", "reason", "preview"]))


# -----------------------------
# Input + CSV upload
# -----------------------------
colA, colB = st.columns([2, 3], gap="large")

with colA:
    ticker = st.text_input("請輸入股票代號", value="6274")
    uploaded = st.file_uploader("（選用）上傳 export.csv 作為備援資料源", type=["csv"])

with colB:
    st.info("若 Cloud 偶發抓不到資料：\n- 先按左側「網路測試」\n- 或直接上傳 export.csv（Date/Open/High/Low/Close/Volume）即可完整分析", icon="💡")


# -----------------------------
# Single stock analysis
# -----------------------------
def render_single_stock():
    df, src, diags = fetch_with_fallback(ticker, period, uploaded)

    if df.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試、或改用 CSV 上傳備援。")
        if debug and diags:
            st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
            st.dataframe(pd.DataFrame([d.__dict__ for d in diags]))
        # Missing pkg hints (quick)
        missing = [d for d in diags if d.result == "MISSING_PKG"]
        if missing:
            st.warning("你目前 Cloud 環境缺套件，請把下列套件加入 requirements.txt：\n" +
                       "\n".join({m.reason for m in missing}))
        return

    df = add_indicators(df)
    latest = df.iloc[-1]
    current = float(latest["Close"])
    score = resonance_score(latest)
    last_date = df.index[-1].date().isoformat()

    # Strategy decision
    if strategy_choice.startswith("回檔"):
        out = strategy_pullback(df, max_buy_distance=max_buy_distance or 0.12)
    else:
        out = strategy_breakout(
            df,
            chase_max=chase_max or 0.06,
            breakout_atr_buffer=breakout_atr_buffer or 0.20,
            stop_atr=stop_atr or 1.60,
            rr=rr or 2.2
        )

    action = out["action"]
    reason = out["reason"]
    zones = out["zones"]

    # Header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("目前價格", f"{current:.2f}")
    m2.metric("AI 共振分數", f"{score}/100")
    m3.metric("資料來源", src)
    m4.metric("最後日期 / 筆數", f"{last_date} / {len(df)}")

    st.subheader("📌 當下是否為買點/賣點？（可操作判斷）")
    if action == "BUY":
        st.success(f"✅ BUY：{reason}")
    elif action == "SELL":
        st.error(f"🔻 SELL：{reason}")
    else:
        st.info(f"⏳ WATCH：{reason}")

    # Zones with professional distance
    st.subheader("🗺️ 未來預估買賣點（區間 + 專業距離描述）")

    nb1, nb2 = zones["near_buy"]
    db1, db2 = zones["deep_buy"]
    ns1, ns2 = zones["near_sell"]
    es1, es2 = zones["ext_sell"]

    c1, c2 = st.columns(2)
    with c1:
        st.success(
            f"🟢 近端買點（可操作）: {nb1:.2f} ~ {nb2:.2f}\n\n"
            f"• {professional_distance_label(current, nb1, nb2)}"
        )
        st.info(
            f"🟦 深回檔買點（等待型）: {db1:.2f} ~ {db2:.2f}\n\n"
            f"• {professional_distance_label(current, db1, db2)}"
        )

    with c2:
        st.warning(
            f"🟨 近端賣點區（壓力/獲利）: {ns1:.2f} ~ {ns2:.2f}\n\n"
            f"• {professional_distance_label(current, ns1, ns2)}"
        )
        st.warning(
            f"🟥 延伸賣點區（強勢延伸）: {es1:.2f} ~ {es2:.2f}\n\n"
            f"• {professional_distance_label(current, es1, es2)}"
        )

    # Breakout plan
    if "trigger" in out:
        st.subheader("🚀 趨勢突破型：交易計畫（不自動下單）")
        trig = out["trigger"]
        chase_d = out["chase_dist"]
        stop_ = out["stop"]
        target_ = out["target"]
        st.write(
            f"- 突破觸發價（含 buffer）: **{trig:.2f}**\n"
            f"- 現價相對觸發偏離: **{chase_d*100:.2f}%**（越大越偏追價）\n"
            f"- 失效止損（ATR-based）: **{stop_:.2f}**\n"
            f"- 目標價（RR={rr or 2.2:.1f}）: **{target_:.2f}**"
        )

    st.subheader("📈 布林通道走勢圖（視覺判讀）")
    plot_bollinger(df, f"{ticker} | {src} | Bollinger + SMA/EMA")

    # Indicator snapshot
    st.subheader("🧾 指標快照（最新一根）")
    snap_cols = ["Close", "SMA20", "EMA20", "RSI14", "MACD", "MACD_SIGNAL", "K", "D", "ATR14", "BB_LOW", "BB_MID", "BB_UP", "Volume", "VOL_SMA20"]
    snap = latest.reindex(snap_cols).to_frame("value")
    st.dataframe(snap)

    if debug and diags:
        st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
        st.dataframe(pd.DataFrame([d.__dict__ for d in diags]))

    # Missing pkg hints
    missing = [d for d in diags if d.result == "MISSING_PKG"]
    if missing:
        st.warning(
            "你目前 Cloud 環境缺套件，請把下列套件加入 requirements.txt：\n\n"
            + "\n".join(sorted({m.reason for m in missing}))
        )


# -----------------------------
# Top10
# -----------------------------
DEFAULT_POOL = [
    "2330", "2317", "2454", "2303", "2412", "3037", "2382", "2881", "2882", "1301",
    "1303", "2308", "2379", "3711", "3008", "4967", "3034", "8046", "6274", "2618"
]

def render_top10():
    st.subheader("🔥 AI 強勢股 Top 10（含可操作判斷）")
    st.caption("先用小池測試（避免全市場在 Cloud 超時）。Top10 會自動去重、並顯示當下操作判斷與關鍵區間。")

    pool_text = st.text_area("Stock Pool（每行一檔，留空用預設池）", value="\n".join(DEFAULT_POOL), height=170)
    pool = [normalize_ticker_input(x) for x in pool_text.splitlines() if normalize_ticker_input(x)]
    if not pool:
        pool = DEFAULT_POOL[:]

    # de-duplicate early
    pool = list(dict.fromkeys(pool))

    rows = []
    diag_fail_count = 0
    for t in pool:
        df, src, diags = fetch_with_fallback(t, period, uploaded)
        if df.empty:
            diag_fail_count += 1
            continue
        df = add_indicators(df)
        latest = df.iloc[-1]
        score = resonance_score(latest)
        zones = compute_trade_zones(df)
        close = zones["close"]
        nb1, nb2 = zones["near_buy"]
        ns1, ns2 = zones["near_sell"]

        # Determine action under chosen strategy
        if strategy_choice.startswith("回檔"):
            out = strategy_pullback(df, max_buy_distance=max_buy_distance or 0.12)
        else:
            out = strategy_breakout(
                df,
                chase_max=chase_max or 0.06,
                breakout_atr_buffer=breakout_atr_buffer or 0.20,
                stop_atr=stop_atr or 1.60,
                rr=rr or 2.2
            )

        rows.append({
            "股票": t,
            "來源": src,
            "現價": round(close, 2),
            "AI分數": score,
            "當下判斷": out["action"],
            "近端買區": f"{nb1:.2f}~{nb2:.2f}",
            "買區距離": professional_distance_label(close, nb1, nb2),
            "近端賣區": f"{ns1:.2f}~{ns2:.2f}",
            "賣區距離": professional_distance_label(close, ns1, ns2),
        })

    if not rows:
        st.error("Top10 掃描結果為空（代表資料源暫時抓不到）。建議先按「網路測試」或用 CSV 上傳備援。")
        return

    df_rank = pd.DataFrame(rows)

    # avoid duplicate single stock repeats (extra safeguard)
    df_rank = df_rank.drop_duplicates(subset=["股票"], keep="first")

    df_rank = df_rank.sort_values(by="AI分數", ascending=False).head(10)
    st.dataframe(df_rank, use_container_width=True)

    if diag_fail_count > 0:
        st.info(f"掃描時有 {diag_fail_count} 檔抓不到資料（Cloud 偶發/WAF/缺套件），其餘已成功計分。")

    st.caption("提示：點回「單一股票分析」輸入 Top10 任一檔，可看完整布林通道圖與交易計畫。")


# -----------------------------
# Render
# -----------------------------
if mode == "單一股票分析":
    render_single_stock()
else:
    render_top10()
