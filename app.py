from __future__ import annotations

import io
import time
import math
import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------
# App Config
# -----------------------
st.set_page_config(layout="wide")
APP_TITLE = "🧠 AI 台股量化專業平台（最終極全備援版 / 不自動下單）"

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

# -----------------------
# Helpers
# -----------------------
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

def _req(url: str, headers: dict, retry: int = 3, sleep: float = 0.7):
    last = None
    for i in range(retry):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            last = (r.status_code, r.text[:180])
            if r.status_code == 200:
                return r, last
            time.sleep(sleep * (i + 1))
        except Exception as e:
            last = ("EXCEPTION", str(e)[:180])
            time.sleep(sleep * (i + 1))
    return None, last

def _parse_roc_date(s: str):
    # 114/02/03
    p = str(s).split("/")
    if len(p) != 3:
        return pd.NaT
    y = int(p[0]) + 1911
    m = int(p[1])
    d = int(p[2])
    return pd.Timestamp(y, m, d)

def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
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

# -----------------------
# Name (optional)
# -----------------------
@st.cache_data(ttl=86400)
def get_stock_name(code: str) -> str:
    code = _norm_code(code)
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        r, _ = _req(url, HEADERS_TWSE, retry=2)
        if not r:
            return ""
        data = r.json()
        for s in data:
            if s.get("Code") == code:
                return s.get("Name", "")
    except:
        pass
    return ""

# -----------------------
# Source A: TWSE CSV (上市)
# -----------------------
def _twse_csv_month(code: str, yyyymm: str):
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date={date_str}&stockNo={code}"
    r, dbg = _req(url, HEADERS_TWSE, retry=3)
    if not r:
        return None, ("TWSE_CSV_FAIL", dbg)

    text = r.text
    if ("很抱歉" in text) or ("查詢日期" in text and "資料" in text and "不存在" in text):
        return None, ("TWSE_CSV_EMPTY", dbg)

    # TWSE CSV 會有很多雜訊行，挑出含引號的資料段
    lines = [ln for ln in text.splitlines() if ln and ('"' in ln)]
    if len(lines) < 3:
        return None, ("TWSE_CSV_TOO_SHORT", dbg)

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception:
        # 再試原文直接讀（有時候可行）
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            return None, ("TWSE_CSV_PARSE_ERR", str(e)[:180])

    df.columns = [str(c).replace('"', "").strip() for c in df.columns]
    # 常見欄位：日期 開盤價 最高價 最低價 收盤價 成交股數
    colmap = {
        "日期": "Date",
        "開盤價": "Open",
        "最高價": "High",
        "最低價": "Low",
        "收盤價": "Close",
        "成交股數": "Volume",
    }
    df = df.rename(columns=colmap)
    if "Date" not in df.columns:
        return None, ("TWSE_CSV_NO_DATE", dbg)

    df["Date"] = df["Date"].apply(_parse_roc_date)
    df = df.set_index("Date")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].astype(str).str.replace(",", "").apply(_safe_float)
    df["Volume"] = df["Volume"].astype(str).str.replace(",", "").apply(_safe_int)

    df = _ensure_ohlcv(df)
    return df, ("TWSE_CSV_OK", None)

# -----------------------
# Source B: TWSE JSON (上市)
# -----------------------
def _twse_json_month(code: str, yyyymm: str):
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={code}"
    r, dbg = _req(url, HEADERS_TWSE, retry=3)
    if not r:
        return None, ("TWSE_JSON_FAIL", dbg)

    try:
        j = r.json()
    except Exception as e:
        return None, ("TWSE_JSON_PARSE_ERR", str(e)[:180])

    if j.get("stat") != "OK":
        return None, ("TWSE_JSON_NOT_OK", dbg)

    fields = j.get("fields", [])
    data = j.get("data", [])
    if not data:
        return None, ("TWSE_JSON_EMPTY", dbg)

    df = pd.DataFrame(data, columns=fields)
    colmap = {
        "日期": "Date",
        "開盤價": "Open",
        "最高價": "High",
        "最低價": "Low",
        "收盤價": "Close",
        "成交股數": "Volume",
    }
    df = df.rename(columns=colmap)
    if "Date" not in df.columns:
        return None, ("TWSE_JSON_NO_DATE", dbg)

    df["Date"] = df["Date"].apply(_parse_roc_date)
    df = df.set_index("Date")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].apply(_safe_float)
    df["Volume"] = df["Volume"].apply(_safe_int)

    df = _ensure_ohlcv(df)
    return df, ("TWSE_JSON_OK", None)

# -----------------------
# Source C: TPEX JSON (上櫃)
# -----------------------
def _tpex_json_month(code: str, yyyymm: str):
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    d_param = f"{_roc_year(y)}/{m:02d}"
    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?d={d_param}&stkno={code}"
    r, dbg = _req(url, HEADERS_TPEX, retry=3)
    if not r:
        return None, ("TPEX_JSON_FAIL", dbg)

    try:
        j = r.json()
    except Exception as e:
        return None, ("TPEX_JSON_PARSE_ERR", str(e)[:180])

    data = j.get("aaData") or j.get("data") or []
    if not data:
        return None, ("TPEX_JSON_EMPTY", dbg)

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
        return None, ("TPEX_JSON_ROWS_EMPTY", dbg)

    df = pd.DataFrame(rows).set_index("Date")
    df = _ensure_ohlcv(df)
    return df, ("TPEX_JSON_OK", None)

# -----------------------
# Source D: Yahoo yfinance (備援)
# -----------------------
def _yahoo_yfinance(code: str, period: str):
    try:
        import yfinance as yf
    except Exception as e:
        return None, ("YFINANCE_NOT_INSTALLED", str(e)[:180])

    # 台股尾碼
    tickers = [f"{code}.TW", f"{code}.TWO"]
    for t in tickers:
        for _ in range(2):
            try:
                df = yf.download(t, period=period, progress=False, threads=False, auto_adjust=False)
                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue
                df = df.rename(columns={"Adj Close": "AdjClose"})
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df = _ensure_ohlcv(df)
                if df is not None:
                    return df, ("YFINANCE_OK", t)
            except Exception as e:
                time.sleep(0.6)
                last = str(e)[:180]
    return None, ("YFINANCE_FAIL", "blocked/empty")

# -----------------------
# Source E: Stooq (備援)
# -----------------------
def _stooq(code: str):
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        return None, ("PDR_NOT_INSTALLED", str(e)[:180])

    # stooq 通常接受 2330.TW
    tickers = [f"{code}.TW", f"{code}.TWO"]
    for t in tickers:
        for _ in range(2):
            try:
                df = pdr.DataReader(t, "stooq")
                if df is None or df.empty:
                    continue
                df = df.sort_index()
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = _ensure_ohlcv(df)
                if df is not None:
                    return df, ("STOOQ_OK", t)
            except Exception:
                time.sleep(0.6)
    return None, ("STOOQ_FAIL", "blocked/empty")

# -----------------------
# Master downloader: Ultimate fallback
# -----------------------
@st.cache_data(ttl=600)
def download_data(code: str, period: str = "6mo", fixed_months: int = 14):
    """
    最終極順序：
    1) TWSE CSV（月）
    2) TWSE JSON（月）
    3) TPEX JSON（月）
    4) Yahoo yfinance（period）
    5) Stooq（近似 period）
    """
    code = _norm_code(code)
    months_keep = _period_to_months(period)

    today = dt.date.today()
    end = today.replace(day=1)
    month_list = [(end - relativedelta(months=i)).strftime("%Y%m") for i in range(fixed_months)]
    month_list = list(reversed(month_list))

    # A) TWSE CSV
    parts = []
    last_dbg = None
    for yyyymm in month_list:
        dfm, dbg = _twse_csv_month(code, yyyymm)
        last_dbg = dbg
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TWSE_CSV", last_dbg

    # B) TWSE JSON
    parts = []
    for yyyymm in month_list:
        dfm, dbg = _twse_json_month(code, yyyymm)
        last_dbg = dbg
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TWSE_JSON", last_dbg

    # C) TPEX JSON
    parts = []
    for yyyymm in month_list:
        dfm, dbg = _tpex_json_month(code, yyyymm)
        last_dbg = dbg
        if dfm is not None and not dfm.empty:
            parts.append(dfm)
    if parts:
        df = _dedup_sort(pd.concat(parts))
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "TPEX_JSON", last_dbg

    # D) yfinance
    df, dbg = _yahoo_yfinance(code, period=period)
    if df is not None:
        return df, "YFINANCE", dbg

    # E) stooq
    df, dbg = _stooq(code)
    if df is not None:
        # stooq 回來可能比 period 多/少，這裡簡單用 cutoff 切
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        df = _ensure_ohlcv(df)
        if df is not None:
            return df, "STOOQ", dbg

    return None, None, last_dbg

# -----------------------
# Indicators & Signals
# -----------------------
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMA
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

    # BIAS(20)
    df["BIAS20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100

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

def ai_score(df: pd.DataFrame) -> int:
    latest = df.iloc[-1]
    score = 0

    # 趨勢
    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA60"]) and latest["SMA20"] > latest["SMA60"]:
        score += 25
    if pd.notna(latest["SMA20"]) and latest["Close"] > latest["SMA20"]:
        score += 10

    # 動能
    if pd.notna(latest["RSI"]) and latest["RSI"] > 55:
        score += 15
    if pd.notna(latest["K"]) and pd.notna(latest["D"]) and latest["K"] > latest["D"]:
        score += 10
    if pd.notna(latest["MACD"]) and latest["MACD"] > 0:
        score += 10

    # 量能
    if pd.notna(latest["VOL_MA20"]) and latest["Volume"] > latest["VOL_MA20"]:
        score += 15

    # 突破
    if pd.notna(latest["RESIST"]) and latest["Close"] >= latest["RESIST"]:
        score += 15

    return int(min(score, 100))

def future_buy_sell_zones(df: pd.DataFrame):
    latest = df.iloc[-1]
    close = float(latest["Close"])

    support = float(latest["SUPPORT"]) if pd.notna(latest["SUPPORT"]) else None
    resist = float(latest["RESIST"]) if pd.notna(latest["RESIST"]) else None
    bb_low = float(latest["BB_LOW"]) if pd.notna(latest["BB_LOW"]) else None
    bb_up = float(latest["BB_UP"]) if pd.notna(latest["BB_UP"]) else None

    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    k = float(latest["K"]) if pd.notna(latest["K"]) else None
    d = float(latest["D"]) if pd.notna(latest["D"]) else None

    # Buy center: support vs bb_low
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

    # Sell center: resist vs bb_up
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

def scan_top10(stock_pool: list[str], period: str) -> pd.DataFrame:
    rows = []
    for code in stock_pool:
        df, src, _dbg = download_data(code, period=period)
        if df is None or len(df) < 80:
            continue
        df = calc_indicators(df)
        rows.append({
            "股票": _norm_code(code),
            "來源": src,
            "AI分數": ai_score(df),
        })
    out = pd.DataFrame(rows, columns=["股票", "來源", "AI分數"])
    if out.empty:
        return out
    return out.sort_values("AI分數", ascending=False).head(10)

# -----------------------
# UI
# -----------------------
st.title(APP_TITLE)

mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])
period = st.sidebar.selectbox("資料期間", ["3mo", "6mo", "12mo"], index=1)
show_debug = st.sidebar.checkbox("顯示下載除錯資訊（Debug）", value=False)

if mode == "單一股票分析":
    code = _norm_code(st.text_input("請輸入股票代號", "2330"))

    df, src, dbg = download_data(code, period=period)
    if df is None:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試或換一檔股票。")
        if show_debug:
            st.write("Debug:", dbg)
        st.stop()

    name = get_stock_name(code)
    df = calc_indicators(df)

    score = ai_score(df)
    last = float(df["Close"].iloc[-1])
    atr = df["ATR14"].iloc[-1]
    stop_loss = None if pd.isna(atr) else (last - 2 * float(atr))

    st.success(f"{name} ({code}) | Source: {src}")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI 共振分數", f"{score}/100")
    c2.metric("目前價格", round(last, 2))
    c3.metric("ATR 停損參考", "-" if stop_loss is None else round(stop_loss, 2))

    buy_zone, buy_reason, sell_zone, sell_reason = future_buy_sell_zones(df)

    st.subheader("📌 未來預估買賣點（只給未來區間，不顯示歷史買賣點）")
    colA, colB = st.columns(2)

    with colA:
        if buy_zone:
            st.info(f"✅ 預估買點區間：{buy_zone[0]:.2f} ~ {buy_zone[1]:.2f}")
            if buy_reason:
                st.caption("條件： " + " / ".join(buy_reason))
        else:
            st.warning("買點區間：資料不足（需更多K線）")

    with colB:
        if sell_zone:
            st.info(f"✅ 預估賣點區間：{sell_zone[0]:.2f} ~ {sell_zone[1]:.2f}")
            if sell_reason:
                st.caption("條件： " + " / ".join(sell_reason))
        else:
            st.warning("賣點區間：資料不足（需更多K線）")

    st.subheader("📈 收盤價走勢")
    st.line_chart(df["Close"])

    if show_debug:
        st.subheader("🛠 Debug")
        st.write("最後一次備援嘗試回傳：", dbg)

else:
    st.caption("Top10 掃描器：會依序用多個資料源抓取；若空，代表該 pool 可能同時抓不到或被擋。")

    # 先用可測小池；你要全市場我再幫你做自動抓清單 + 加速
    stock_pool = ["2330", "2317", "2303", "2454", "2382", "3037", "8046", "6274", "4967"]

    result = scan_top10(stock_pool, period=period)

    st.subheader("🔥 AI 強勢股 Top 10")
    if result.empty:
        st.warning("目前掃描結果為空（代表資料下載全失敗或池內股票都取不到）。可開 Debug 觀察最後回傳。")
    else:
        st.dataframe(result, use_container_width=True)
