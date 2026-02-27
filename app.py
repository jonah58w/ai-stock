from __future__ import annotations

import io
import time
import math
import datetime as dt
from typing import Optional, List, Dict, Tuple

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# App Config
# =========================
st.set_page_config(layout="wide")
APP_TITLE = "🧠 AI 台股量化專業平台（最終極全備援 + 逐路診斷版 / 不自動下單）"

HEADERS_TWSE = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.twse.com.tw/",
    "Connection": "keep-alive",
}
HEADERS_TPEX = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.9",
    "Referer": "https://www.tpex.org.tw/",
    "Connection": "keep-alive",
}
HEADERS_GENERIC = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Connection": "keep-alive",
}


# =========================
# Helpers
# =========================
def _norm_code(code: str) -> str:
    return str(code).strip().upper().replace(".TW", "").replace(".TWO", "")

def _roc_year(ad_year: int) -> int:
    return ad_year - 1911

def _period_to_months(period: str) -> int:
    p = (period or "6mo").strip().lower()
    if p.endswith("mo"):
        return max(1, int(p.replace("mo", "")))
    if p.endswith("y"):
        return max(1, int(p.replace("y", ""))) * 12
    return 6

def _safe_float(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"", "--", "null", "None"}:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def _safe_int(x):
    v = _safe_float(x)
    if math.isnan(v):
        return np.nan
    return int(v)

def _parse_roc_date(s: str):
    p = str(s).split("/")
    if len(p) != 3:
        return pd.NaT
    y = int(p[0]) + 1911
    m = int(p[1])
    d = int(p[2])
    return pd.Timestamp(y, m, d)

def _ensure_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            return None
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None
    return df[need].copy()

def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]

def _req(url: str, headers: dict, retry: int = 3, sleep: float = 0.7):
    last_dbg = {"url": url, "status": None, "snippet": None, "exception": None}
    for i in range(retry):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            last_dbg["status"] = r.status_code
            last_dbg["snippet"] = (r.text or "")[:200]
            if r.status_code == 200:
                return r, last_dbg
            time.sleep(sleep * (i + 1))
        except Exception as e:
            last_dbg["exception"] = str(e)[:200]
            time.sleep(sleep * (i + 1))
    return None, last_dbg

def _pct(a: float, b: float) -> Optional[float]:
    # (a-b)/b
    if a is None or b is None:
        return None
    try:
        if b == 0:
            return None
        return (a - b) / b * 100.0
    except:
        return None


# =========================
# Network test
# =========================
def net_test() -> pd.DataFrame:
    targets = [
        ("TWSE", "https://www.twse.com.tw/"),
        ("TPEX", "https://www.tpex.org.tw/"),
        ("Yahoo", "https://finance.yahoo.com/"),
        ("Stooq", "https://stooq.com/"),
        ("Google204", "https://www.google.com/generate_204"),
    ]
    rows = []
    for name, url in targets:
        try:
            r = requests.get(url, headers=HEADERS_GENERIC, timeout=12)
            rows.append({"Target": name, "URL": url, "Status": r.status_code, "OK": r.status_code in (200, 204)})
        except Exception as e:
            rows.append({"Target": name, "URL": url, "Status": "EXCEPTION", "OK": False, "Error": str(e)[:160]})
    return pd.DataFrame(rows)


# =========================
# Source A: TWSE CSV (上市) ✅已修正：找表頭行再解析（避免 TOO_SHORT）
# =========================
def _twse_csv_month(code: str, yyyymm: str):
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date={date_str}&stockNo={code}"
    r, dbg = _req(url, HEADERS_TWSE, retry=3)
    if not r:
        return None, {"source": "TWSE_CSV", "result": "FAIL", **dbg, "yyyymm": yyyymm}

    try:
        r.encoding = "utf-8"
    except:
        pass

    text = r.text or ""
    if not text.strip():
        return None, {"source": "TWSE_CSV", "result": "EMPTY_TEXT", **dbg, "yyyymm": yyyymm}

    raw_lines = [ln.strip().strip("\ufeff") for ln in text.splitlines() if ln.strip()]
    if len(raw_lines) < 3:
        return None, {"source": "TWSE_CSV", "result": "TOO_SHORT_TEXT", **dbg, "yyyymm": yyyymm}

    header_idx = None
    for i, ln in enumerate(raw_lines):
        if ln.startswith("日期,") and (("成交股數" in ln) or ("開盤價" in ln) or ("收盤價" in ln)):
            header_idx = i
            break
    if header_idx is None:
        for i, ln in enumerate(raw_lines):
            if ("日期" in ln) and ("成交股數" in ln) and ("," in ln):
                header_idx = i
                break

    if header_idx is None:
        return None, {
            "source": "TWSE_CSV",
            "result": "NO_HEADER_FOUND",
            "preview": raw_lines[:6],
            **dbg,
            "yyyymm": yyyymm,
        }

    csv_text = "\n".join(raw_lines[header_idx:])

    try:
        df = pd.read_csv(io.StringIO(csv_text), engine="python")
    except Exception as e:
        return None, {"source": "TWSE_CSV", "result": "PARSE_ERR", "exception": str(e)[:200], **dbg, "yyyymm": yyyymm}

    df.columns = [str(c).strip().replace('"', "") for c in df.columns]
    colmap = {"日期": "Date", "開盤價": "Open", "最高價": "High", "最低價": "Low", "收盤價": "Close", "成交股數": "Volume"}
    df = df.rename(columns=colmap)

    if "Date" not in df.columns:
        return None, {"source": "TWSE_CSV", "result": "NO_DATE_COL", "cols": list(df.columns)[:12], **dbg, "yyyymm": yyyymm}

    df["Date"] = df["Date"].apply(_parse_roc_date)
    df = df.set_index("Date")

    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "").apply(_safe_float)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].astype(str).str.replace(",", "").apply(_safe_int)

    df = _ensure_ohlcv(df)
    if df is None:
        return None, {"source": "TWSE_CSV", "result": "NO_OHLCV", "cols": list(df.columns)[:12], **dbg, "yyyymm": yyyymm}

    return df, {"source": "TWSE_CSV", "result": "OK", **dbg, "yyyymm": yyyymm}


# =========================
# Source B: TWSE JSON (上市)
# =========================
def _twse_json_month(code: str, yyyymm: str):
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={code}"
    r, dbg = _req(url, HEADERS_TWSE, retry=3)
    if not r:
        return None, {"source": "TWSE_JSON", "result": "FAIL", **dbg, "yyyymm": yyyymm}

    try:
        j = r.json()
    except Exception as e:
        return None, {"source": "TWSE_JSON", "result": "JSON_PARSE_ERR", "exception": str(e)[:200], **dbg, "yyyymm": yyyymm}

    if j.get("stat") != "OK":
        return None, {"source": "TWSE_JSON", "result": f"NOT_OK({j.get('stat')})", **dbg, "yyyymm": yyyymm}

    fields = j.get("fields", [])
    data = j.get("data", [])
    if not data:
        return None, {"source": "TWSE_JSON", "result": "EMPTY", **dbg, "yyyymm": yyyymm}

    df = pd.DataFrame(data, columns=fields)
    colmap = {"日期": "Date", "開盤價": "Open", "最高價": "High", "最低價": "Low", "收盤價": "Close", "成交股數": "Volume"}
    df = df.rename(columns=colmap)

    if "Date" not in df.columns:
        return None, {"source": "TWSE_JSON", "result": "NO_DATE_COL", "cols": list(df.columns)[:12], **dbg, "yyyymm": yyyymm}

    df["Date"] = df["Date"].apply(_parse_roc_date)
    df = df.set_index("Date")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].apply(_safe_float)
    df["Volume"] = df["Volume"].apply(_safe_int)

    df = _ensure_ohlcv(df)
    if df is None:
        return None, {"source": "TWSE_JSON", "result": "NO_OHLCV", "cols": list(df.columns)[:12], **dbg, "yyyymm": yyyymm}

    return df, {"source": "TWSE_JSON", "result": "OK", **dbg, "yyyymm": yyyymm}


# =========================
# Source C: TPEX JSON (上櫃)
# =========================
def _tpex_json_month(code: str, yyyymm: str):
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    d_param = f"{_roc_year(y)}/{m:02d}"
    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?d={d_param}&stkno={code}"
    r, dbg = _req(url, HEADERS_TPEX, retry=3)
    if not r:
        return None, {"source": "TPEX_JSON", "result": "FAIL", **dbg, "yyyymm": yyyymm}

    try:
        j = r.json()
    except Exception as e:
        return None, {"source": "TPEX_JSON", "result": "JSON_PARSE_ERR", "exception": str(e)[:200], **dbg, "yyyymm": yyyymm}

    data = j.get("aaData") or j.get("data") or []
    if not data:
        return None, {"source": "TPEX_JSON", "result": "EMPTY", **dbg, "yyyymm": yyyymm}

    rows = []
    for row in data:
        if not row or len(row) < 7:
            continue
        date_s, vol_s = row[0], row[1]
        open_s, high_s, low_s, close_s = row[3], row[4], row[5], row[6]
        date = _parse_roc_date(date_s)
        if pd.isna(date):
            continue
        rows.append({
            "Date": date,
            "Open": _safe_float(open_s),
            "High": _safe_float(high_s),
            "Low": _safe_float(low_s),
            "Close": _safe_float(close_s),
            "Volume": _safe_int(vol_s),
        })

    if not rows:
        return None, {"source": "TPEX_JSON", "result": "ROWS_EMPTY", **dbg, "yyyymm": yyyymm}

    df = pd.DataFrame(rows).set_index("Date")
    df = _ensure_ohlcv(df)
    if df is None:
        return None, {"source": "TPEX_JSON", "result": "NO_OHLCV", **dbg, "yyyymm": yyyymm}

    return df, {"source": "TPEX_JSON", "result": "OK", **dbg, "yyyymm": yyyymm}


# =========================
# Source D: yfinance (備援)
# =========================
def _yahoo_yfinance(code: str, period: str):
    try:
        import yfinance as yf
    except Exception as e:
        return None, {"source": "YFINANCE", "result": "NOT_INSTALLED", "exception": str(e)[:200]}

    tickers = [f"{code}.TW", f"{code}.TWO"]
    last_err = None
    for t in tickers:
        try:
            df = yf.download(t, period=period, progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                last_err = f"{t} empty"
                continue
            df = df.dropna(subset=["Close"])
            if df.empty:
                last_err = f"{t} empty after dropna"
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df = _ensure_ohlcv(df)
            if df is not None:
                return df, {"source": "YFINANCE", "result": "OK", "ticker": t}
            last_err = f"{t} no ohlcv"
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(0.6)
    return None, {"source": "YFINANCE", "result": "FAIL", "exception": last_err}


# =========================
# Source E: Stooq (備援)
# =========================
def _stooq(code: str):
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        return None, {"source": "STOOQ", "result": "NOT_INSTALLED", "exception": str(e)[:200]}

    tickers = [f"{code}.TW", f"{code}.TWO"]
    last_err = None
    for t in tickers:
        try:
            df = pdr.DataReader(t, "stooq")
            if df is None or df.empty:
                last_err = f"{t} empty"
                continue
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = _ensure_ohlcv(df)
            if df is not None:
                return df, {"source": "STOOQ", "result": "OK", "ticker": t}
            last_err = f"{t} no ohlcv"
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(0.6)

    return None, {"source": "STOOQ", "result": "FAIL", "exception": last_err}


# =========================
# Ultimate master downloader (with trace)
# =========================
@st.cache_data(ttl=600)
def download_data_with_trace(code: str, period: str = "6mo", fixed_months: int = 14):
    code = _norm_code(code)
    months_keep = _period_to_months(period)

    today = dt.date.today()
    end = today.replace(day=1)
    month_list = [(end - relativedelta(months=i)).strftime("%Y%m") for i in range(fixed_months)]
    month_list = list(reversed(month_list))

    trace: List[Dict] = []

    # 1) TWSE CSV
    parts = []
    for yyyymm in month_list:
        dfm, info = _twse_csv_month(code, yyyymm)
        trace.append(info)
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TWSE_CSV", trace

    # 2) TWSE JSON
    parts = []
    for yyyymm in month_list:
        dfm, info = _twse_json_month(code, yyyymm)
        trace.append(info)
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TWSE_JSON", trace

    # 3) TPEX JSON
    parts = []
    for yyyymm in month_list:
        dfm, info = _tpex_json_month(code, yyyymm)
        trace.append(info)
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TPEX_JSON", trace

    # 4) yfinance
    df, info = _yahoo_yfinance(code, period=period)
    trace.append(info)
    if df is not None:
        return df, "YFINANCE", trace

    # 5) stooq
    df, info = _stooq(code)
    trace.append(info)
    if df is not None:
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "STOOQ", trace

    return None, None, trace


# =========================
# Indicators
# =========================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # KD(9)
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = (df["Close"] - low9) / (high9 - low9) * 100
    df["K"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1/3, adjust=False).mean()

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    # Bollinger(20,2)
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    # Volume MA
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # Support/Resistance (60)
    df["SUPPORT"] = df["Low"].rolling(60).min()
    df["RESIST"] = df["High"].rolling(60).max()

    # ATR(14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df


# =========================
# Scoring + zones + NOW action
# =========================
def ai_score(df: pd.DataFrame) -> int:
    latest = df.iloc[-1]
    score = 0
    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA60"]) and latest["SMA20"] > latest["SMA60"]:
        score += 25
    if pd.notna(latest["SMA20"]) and latest["Close"] > latest["SMA20"]:
        score += 10
    if pd.notna(latest["RSI"]) and latest["RSI"] > 55:
        score += 15
    if pd.notna(latest["K"]) and pd.notna(latest["D"]) and latest["K"] > latest["D"]:
        score += 10
    if pd.notna(latest["MACD"]) and latest["MACD"] > 0:
        score += 10
    if pd.notna(latest["VOL_MA20"]) and latest["Volume"] > latest["VOL_MA20"]:
        score += 15
    if pd.notna(latest["RESIST"]) and latest["Close"] >= latest["RESIST"]:
        score += 15
    return int(min(score, 100))

def future_buy_sell_zones(df: pd.DataFrame) -> Tuple[Optional[Tuple[float,float]], List[str], Optional[Tuple[float,float]], List[str]]:
    latest = df.iloc[-1]
    close = float(latest["Close"])

    support = float(latest["SUPPORT"]) if pd.notna(latest["SUPPORT"]) else None
    resist = float(latest["RESIST"]) if pd.notna(latest["RESIST"]) else None
    bb_low = float(latest["BB_LOW"]) if pd.notna(latest["BB_LOW"]) else None
    bb_up = float(latest["BB_UP"]) if pd.notna(latest["BB_UP"]) else None
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    k = float(latest["K"]) if pd.notna(latest["K"]) else None
    d = float(latest["D"]) if pd.notna(latest["D"]) else None

    buy_center = None
    if support and bb_low:
        buy_center = min(support, bb_low)
    elif support:
        buy_center = support
    elif bb_low:
        buy_center = bb_low

    buy_zone = None
    buy_reason = []
    if buy_center:
        buy_zone = (buy_center * 0.99, buy_center * 1.01)
        if rsi is not None and rsi <= 40:
            buy_reason.append("RSI偏低")
        if k is not None and d is not None and k <= 25 and d <= 30:
            buy_reason.append("KD偏低")
        if bb_low and close <= bb_low * 1.03:
            buy_reason.append("接近布林下軌")
        if support and close <= support * 1.05:
            buy_reason.append("接近支撐")

    sell_center = None
    if resist and bb_up:
        sell_center = max(resist, bb_up)
    elif resist:
        sell_center = resist
    elif bb_up:
        sell_center = bb_up

    sell_zone = None
    sell_reason = []
    if sell_center:
        sell_zone = (sell_center * 0.99, sell_center * 1.01)
        if rsi is not None and rsi >= 65:
            sell_reason.append("RSI偏高")
        if k is not None and d is not None and k >= 75 and d >= 70:
            sell_reason.append("KD偏高")
        if bb_up and close >= bb_up * 0.97:
            sell_reason.append("接近布林上軌")
        if resist and close >= resist * 0.95:
            sell_reason.append("接近壓力")

    return buy_zone, buy_reason, sell_zone, sell_reason

def now_action(df: pd.DataFrame):
    """
    回傳：
      action: "買點" / "賣點" / "觀望"
      confidence: 0~100
      reasons: list[str]
      dist_to_buy_pct, dist_to_sell_pct: 目前價 到 buy_zone/sell_zone 中心的距離(%)
      in_buy_zone, in_sell_zone: bool
    """
    latest = df.iloc[-1]
    close = float(latest["Close"])
    score = ai_score(df)

    buy_zone, buy_reason, sell_zone, sell_reason = future_buy_sell_zones(df)

    in_buy = False
    in_sell = False
    dist_buy = None
    dist_sell = None

    if buy_zone:
        lo, hi = buy_zone
        in_buy = (close >= lo) and (close <= hi)
        center = (lo + hi) / 2.0
        dist_buy = _pct(close, center)

    if sell_zone:
        lo, hi = sell_zone
        in_sell = (close >= lo) and (close <= hi)
        center = (lo + hi) / 2.0
        dist_sell = _pct(close, center)

    # 共振條件（用來做「當下判斷」）
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    k = float(latest["K"]) if pd.notna(latest["K"]) else None
    d = float(latest["D"]) if pd.notna(latest["D"]) else None
    macd = float(latest["MACD"]) if pd.notna(latest["MACD"]) else None
    vol_ma = float(latest["VOL_MA20"]) if pd.notna(latest["VOL_MA20"]) else None
    vol = float(latest["Volume"]) if pd.notna(latest["Volume"]) else None

    buy_votes = 0
    sell_votes = 0
    reasons = []

    # 買方條件
    if in_buy:
        buy_votes += 2
        reasons.append("價格已進入買點區間")
    if rsi is not None and rsi <= 40:
        buy_votes += 1
        reasons.append("RSI低檔")
    if k is not None and d is not None and k <= 25 and d <= 30:
        buy_votes += 1
        reasons.append("KD低檔")
    if macd is not None and macd > 0:
        buy_votes += 1
        reasons.append("MACD轉強(>0)")
    if vol is not None and vol_ma is not None and vol > vol_ma:
        buy_votes += 1
        reasons.append("量能放大")

    # 賣方條件
    if in_sell:
        sell_votes += 2
        reasons.append("價格已進入賣點區間")
    if rsi is not None and rsi >= 65:
        sell_votes += 1
        reasons.append("RSI高檔")
    if k is not None and d is not None and k >= 75 and d >= 70:
        sell_votes += 1
        reasons.append("KD高檔")
    if macd is not None and macd < 0:
        sell_votes += 1
        reasons.append("MACD轉弱(<0)")

    # 決策邏輯（避免亂跳）
    if (sell_votes >= 3) and (sell_votes > buy_votes):
        action = "賣點"
        confidence = min(95, 55 + sell_votes * 10 + (max(0, (100 - score)) // 10))
    elif (buy_votes >= 3) and (buy_votes > sell_votes):
        action = "買點"
        confidence = min(95, 55 + buy_votes * 10 + (score // 10))
    else:
        action = "觀望"
        confidence = 40 + min(30, abs(buy_votes - sell_votes) * 5)

    # 理由精簡：買/賣 只留對應
    if action == "買點":
        reasons = [r for r in reasons if ("買" in r) or ("低檔" in r) or ("轉強" in r) or ("量能" in r)]
        if buy_reason:
            reasons += [f"（區間條件）{' / '.join(buy_reason)}"]
    elif action == "賣點":
        reasons = [r for r in reasons if ("賣" in r) or ("高檔" in r) or ("轉弱" in r)]
        if sell_reason:
            reasons += [f"（區間條件）{' / '.join(sell_reason)}"]
    else:
        # 觀望：只留中性
        reasons = ["條件尚未形成明確共振（等待價格進入區間/或指標翻轉）"]

    return {
        "action": action,
        "confidence": int(confidence),
        "reasons": reasons[:4],  # 不要太長
        "score": score,
        "close": close,
        "buy_zone": buy_zone,
        "sell_zone": sell_zone,
        "dist_to_buy_pct": dist_buy,
        "dist_to_sell_pct": dist_sell,
        "in_buy_zone": in_buy,
        "in_sell_zone": in_sell,
    }


# =========================
# Top10
# =========================
def scan_top10(stock_pool: List[str], period: str) -> pd.DataFrame:
    rows = []
    for code in stock_pool:
        df, src, _trace = download_data_with_trace(code, period=period)
        if df is None or len(df) < 80:
            continue
        df = calc_indicators(df)
        info = now_action(df)

        bz = info["buy_zone"]
        sz = info["sell_zone"]
        bz_s = "-" if not bz else f"{bz[0]:.2f}~{bz[1]:.2f}"
        sz_s = "-" if not sz else f"{sz[0]:.2f}~{sz[1]:.2f}"

        # Top10更好用：顯示距離買/賣區(%)（越小越接近）
        d_buy = info["dist_to_buy_pct"]
        d_sell = info["dist_to_sell_pct"]

        rows.append({
            "股票": _norm_code(code),
            "Close": round(info["close"], 2),
            "當下建議": info["action"],
            "信心": f'{info["confidence"]}%',
            "BuyZone": bz_s,
            "距離買區%": None if d_buy is None else round(d_buy, 2),
            "SellZone": sz_s,
            "距離賣區%": None if d_sell is None else round(d_sell, 2),
            "AI分數": info["score"],
            "來源": src,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 排序：先把「買點」排前面，再比 AI分數，再比距離買區絕對值（越小越接近）
    def action_rank(x):
        return {"買點": 0, "觀望": 1, "賣點": 2}.get(str(x), 9)

    out["__rank"] = out["當下建議"].apply(action_rank)
    out["__abs_buy"] = out["距離買區%"].abs().fillna(9999)

    out = out.sort_values(["__rank", "AI分數", "__abs_buy"], ascending=[True, False, True])
    out = out.drop(columns=["__rank", "__abs_buy"])
    return out.head(10)


# =========================
# UI
# =========================
st.title(APP_TITLE)

mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])
period = st.sidebar.selectbox("資料期間", ["3mo", "6mo", "12mo"], index=1)
show_debug = st.sidebar.checkbox("顯示下載除錯資訊（Debug）", value=False)

with st.sidebar.expander("🧪 網路測試（建議先按一次）", expanded=False):
    if st.button("Run network test"):
        st.dataframe(net_test(), use_container_width=True)

if mode == "單一股票分析":
    code = _norm_code(st.text_input("請輸入股票代號", "2330"))
    df, src, trace = download_data_with_trace(code, period=period)

    if df is None:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。")
        st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
        st.dataframe(pd.DataFrame(trace), use_container_width=True)
        st.stop()

    df = calc_indicators(df)
    info = now_action(df)

    # Header
    st.success(f"✅ 取得資料成功 | 代號: {code} | Source: {src}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("當下建議", info["action"])
    c2.metric("信心", f'{info["confidence"]}%')
    c3.metric("目前價格", round(info["close"], 2))
    c4.metric("AI 共振分數", f'{info["score"]}/100')

    # 明確可判讀的買賣點資訊
    st.subheader("📌 當下是否為買點/賣點？（可操作判斷）")
    if info["action"] == "買點":
        st.info("✅ **偏買**：目前條件達成共振（或已進入買區），可分批佈局 / 等回測確認。")
    elif info["action"] == "賣點":
        st.warning("✅ **偏賣**：目前條件偏高檔/壓力，適合減碼或等待回檔。")
    else:
        st.write("⏳ **觀望**：條件尚未明確，等待價格進入區間或指標翻轉。")

    if info["reasons"]:
        st.caption("理由： " + " / ".join(info["reasons"]))

    # 未來預估區間 + 距離（重點）
    st.subheader("🗺️ 未來預估買賣點（區間 + 距離%）")
    colA, colB = st.columns(2)

    with colA:
        bz = info["buy_zone"]
        if bz:
            st.success(f"預估買點區間：**{bz[0]:.2f} ~ {bz[1]:.2f}**")
            if info["dist_to_buy_pct"] is not None:
                st.caption(f"目前價距離買區中心：{info['dist_to_buy_pct']:.2f}%（負值=低於中心，正值=高於中心）")
            st.caption("狀態：" + ("✅ 已進入買區" if info["in_buy_zone"] else "尚未進入買區"))
        else:
            st.warning("買點區間：資料不足（需更多K線）")

    with colB:
        sz = info["sell_zone"]
        if sz:
            st.success(f"預估賣點區間：**{sz[0]:.2f} ~ {sz[1]:.2f}**")
            if info["dist_to_sell_pct"] is not None:
                st.caption(f"目前價距離賣區中心：{info['dist_to_sell_pct']:.2f}%（負值=低於中心，正值=高於中心）")
            st.caption("狀態：" + ("✅ 已進入賣區" if info["in_sell_zone"] else "尚未進入賣區"))
        else:
            st.warning("賣點區間：資料不足（需更多K線）")

    # 圖表
    st.subheader("📈 收盤價走勢（視覺判讀）")
    st.line_chart(df["Close"])

    if show_debug:
        st.subheader("🛠 Debug Trace")
        st.dataframe(pd.DataFrame(trace), use_container_width=True)

else:
    st.caption("Top10 掃描器：先用小池測試（避免全市場在 Cloud 超時）。")
    stock_pool = ["2330", "2317", "2303", "2454", "2382", "3037", "8046", "6274", "4967"]

    result = scan_top10(stock_pool, period=period)

    st.subheader("🔥 AI 強勢股 Top 10（含當下買賣判斷 + 未來區間）")
    if result.empty:
        st.warning("目前掃描結果為空（代表 pool 內股票都抓不到或不足K線）。建議切回單股模式看逐路診斷表。")
    else:
        st.dataframe(result, use_container_width=True)

        st.caption("排序邏輯：買點優先 → AI分數高者優先 → 越接近買區者優先。")
