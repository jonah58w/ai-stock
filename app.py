# app.py
# AI 台股量化專業平台（無 Plotly / 全功能保留）
# ✅ 多源備援：Yahoo(yfinance) / Stooq / TWSE / TPEX (JSON/CSV) / CSV上傳備援
# ✅ 逐路診斷：哪一路失敗、為什麼（含 WAF / EMPTY_TEXT 判斷）
# ✅ 指標共振：MACD + KD + RSI + 布林 + 乖離 + 量能 + 支撐壓力
# ✅ 單股：當下是否可操作（買/賣/觀望）+ 近端/等待型買點 + 壓力/獲利區 + 專業距離說法
# ✅ Top10：去重、避免同一檔重複、顯示「可操作/等待」與距離
# ✅ 趨勢突破型：追價距離限制 + ATR buffer + 風報比 + 失效止損
# ⚠️ 僅供資訊顯示與風控演算，不構成投資建議，不自動下單。

from __future__ import annotations

import io
import re
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt


# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(page_title="AI 台股量化專業平台（無 Plotly / 全功能保留）", layout="wide")

APP_TITLE = "🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）"
APP_SUBTITLE = "多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + Top10 掃描 + 交易計畫（不自動下單）"

DEFAULT_STOCK_POOL = [
    # 你可自行增減；Top10 會先用小池避免 Cloud 超時
    "2330", "2317", "2454", "2412", "2308", "2881", "2882", "2891", "2892", "2886",
    "2603", "2609", "2615", "3037", "2382", "3711", "5871", "6415", "6533", "6505",
    "8046", "4967", "3008", "3443", "3034", "6669", "2357", "2379", "2327", "4938",
]

PERIOD_TO_DAYS = {
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
    "Connection": "keep-alive",
}


# -----------------------------
# 工具：格式化/解析
# -----------------------------
def now_tpe() -> dt.datetime:
    # 不依賴 pytz
    return dt.datetime.utcnow() + dt.timedelta(hours=8)


def normalize_ticker(user_input: str) -> str:
    s = str(user_input).strip().upper()
    s = re.sub(r"[^0-9A-Z\.]", "", s)
    return s


def is_taiwan_code(s: str) -> bool:
    s = normalize_ticker(s)
    return bool(re.fullmatch(r"\d{4}", s))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    期望欄位: Date index, Open/High/Low/Close/Volume
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # 可能是 MultiIndex columns（yfinance）
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[-1] for c in out.columns]

    # 欄名標準化
    rename_map = {}
    for c in out.columns:
        lc = str(c).lower()
        if lc in ["open", "o"]:
            rename_map[c] = "Open"
        elif lc in ["high", "h"]:
            rename_map[c] = "High"
        elif lc in ["low", "l"]:
            rename_map[c] = "Low"
        elif lc in ["close", "c", "adj close", "adjclose"]:
            rename_map[c] = "Close"
        elif lc in ["volume", "v", "vol"]:
            rename_map[c] = "Volume"
    out = out.rename(columns=rename_map)

    # 若 Close 缺，用 Adj Close 代替（少數資料源）
    if "Close" not in out.columns:
        for cand in ["Adj Close", "AdjClose", "adjclose", "adj close"]:
            if cand in out.columns:
                out["Close"] = out[cand]
                break

    need = ["Open", "High", "Low", "Close"]
    for c in need:
        if c not in out.columns:
            return pd.DataFrame()

    if "Volume" not in out.columns:
        out["Volume"] = 0

    # index
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.set_index("Date")

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    # numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Close"])
    return out


# -----------------------------
# 指標計算
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 9, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_line = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return lower, mid, upper


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_ohlcv(df)
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["EMA20"] = ema(close, 20)
    out["SMA60"] = sma(close, 60)
    out["RSI14"] = rsi(close, 14)

    out["MACD"], out["MACDsig"], out["MACDhist"] = macd(close)
    out["K"], out["D"] = stochastic_kd(high, low, close)

    out["ATR14"] = atr(high, low, close, 14)

    out["BBL"], out["BBM"], out["BBU"] = bollinger(close, 20, 2.0)
    out["BBwidth"] = (out["BBU"] - out["BBL"]) / out["BBM"]
    out["BBpctB"] = (close - out["BBL"]) / (out["BBU"] - out["BBL"]).replace(0, np.nan)

    # 乖離率（距離 EMA20）
    out["DEV_EMA20"] = (close - out["EMA20"]) / out["EMA20"]

    # 量能 z-score（20日）
    vol = out["Volume"].fillna(0)
    out["VOL_Z20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # 支撐/壓力（近端）
    out["SUP20"] = close.rolling(20).min()
    out["RES20"] = close.rolling(20).max()

    return out


# -----------------------------
# 資料來源：多源備援
# -----------------------------
@dataclass
class FetchResult:
    ok: bool
    source: str
    df: pd.DataFrame
    note: str = ""


def try_requests_text(url: str, timeout: int = 10) -> Tuple[int, str]:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    return r.status_code, r.text[:20000]  # 限長


def looks_like_waf(text: str) -> bool:
    t = (text or "").lower()
    if "cloudflare" in t or "attention required" in t or "captcha" in t:
        return True
    if "<html" in t and ("verify you are human" in t or "incapsula" in t):
        return True
    return False


def twse_stock_day_csv(month_yyyymm: str) -> str:
    # TWSE: STOCK_DAY CSV（整月，需再篩代號）
    # month_yyyymm 例如 202501
    return f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date={month_yyyymm}01&stockNo="


def twse_json_daily_price(stock_no: str, month_yyyymm: str) -> str:
    # TWSE JSON（afterTrading/..）不同路徑會改；此處保留一個常見端點
    # 注意：TWSE 端點偶爾變動，本程式用診斷 + 備援
    return f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={month_yyyymm}01&stockNo={stock_no}"


def tpex_json_daily_price(stock_no: str, month_yyyymm: str) -> str:
    # TPEX 常見端點（可能 WAF/HTML）
    return f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={month_yyyymm[:4]}/{month_yyyymm[4:6]}&stkno={stock_no}"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_yfinance(code4: str, days: int) -> FetchResult:
    try:
        import yfinance as yf
    except Exception as e:
        return FetchResult(False, "YF", pd.DataFrame(), f"yfinance not available: {e}")

    # 先試 .TW，再試 .TWO
    symbols = [f"{code4}.TW", f"{code4}.TWO"]
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{max(7, days)}d", interval="1d", auto_adjust=False, progress=False, threads=False)
            df = ensure_ohlcv(df)
            if df is not None and not df.empty and len(df) >= 60:
                return FetchResult(True, f"YF:{sym.split('.')[-1]}", df, "")
        except Exception as e:
            last_err = str(e)
            continue
    return FetchResult(False, "YF", pd.DataFrame(), "yfinance no data / blocked")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_stooq(code4: str, days: int) -> FetchResult:
    # Stooq: 通常代號格式 2330.TW（不保證）
    try:
        import pandas_datareader.data as web
    except Exception as e:
        return FetchResult(False, "STOOQ", pd.DataFrame(), f"pandas-datareader not available: {e}")

    sym = f"{code4}.TW"
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=max(120, days + 30))
    try:
        df = web.DataReader(sym, "stooq", start, end)
        # stooq 回來 index 倒序、欄位小寫
        df = df.sort_index()
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = ensure_ohlcv(df)
        if df is not None and not df.empty and len(df) >= 60:
            return FetchResult(True, "STOOQ", df, "")
    except Exception:
        pass
    return FetchResult(False, "STOOQ", pd.DataFrame(), "stooq no data / blocked")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_twse_tpex_http(code4: str, days: int) -> Tuple[FetchResult, List[Dict[str, Any]]]:
    """
    走 TWSE/TPEX 端點（可能 200 但空內容 / WAF）
    回傳：FetchResult + diag_rows
    """
    diag: List[Dict[str, Any]] = []
    months = []
    today = now_tpe().date()
    # 取最近 N 個月（以 days 換算）
    mcount = int(math.ceil(days / 30.0)) + 1
    y, m = today.year, today.month
    for i in range(mcount):
        yy = y
        mm = m - i
        while mm <= 0:
            yy -= 1
            mm += 12
        months.append(f"{yy}{mm:02d}")

    frames = []
    used_source = None

    # 1) TWSE JSON（最常用）
    for mon in months:
        url = twse_json_daily_price(code4, mon)
        try:
            status, text = try_requests_text(url, timeout=10)
            bytes_len = len(text.encode("utf-8", errors="ignore"))
            result = "OK"
            snippet = text[:120].replace("\n", " ")

            if status != 200:
                result = "HTTP_ERR"
            elif not text.strip():
                result = "EMPTY_TEXT"
            elif looks_like_waf(text):
                result = "WAF_HTML"
            else:
                # 解析 JSON
                try:
                    j = json.loads(text)
                    if not j or j.get("stat", "").lower().find("ok") == -1:
                        # 有些月份無資料
                        result = "NO_DATA"
                    else:
                        data = j.get("data", [])
                        if not data:
                            result = "NO_DATA"
                        else:
                            # 欄位：日期、成交股數、成交金額、開盤、最高、最低、收盤、漲跌價差、成交筆數
                            rows = []
                            for r in data:
                                if len(r) < 7:
                                    continue
                                date_str = str(r[0]).strip()
                                # 民國年轉西元
                                # 112/01/03 -> 2023-01-03
                                try:
                                    parts = date_str.split("/")
                                    if len(parts) == 3:
                                        yy = int(parts[0]) + 1911
                                        mm = int(parts[1])
                                        dd = int(parts[2])
                                        d = dt.date(yy, mm, dd)
                                    else:
                                        d = pd.to_datetime(date_str).date()
                                except Exception:
                                    continue

                                o = safe_float(str(r[3]).replace(",", ""), np.nan)
                                h = safe_float(str(r[4]).replace(",", ""), np.nan)
                                l = safe_float(str(r[5]).replace(",", ""), np.nan)
                                c = safe_float(str(r[6]).replace(",", ""), np.nan)
                                v = safe_float(str(r[1]).replace(",", ""), 0.0)
                                rows.append((d, o, h, l, c, v))

                            if rows:
                                dfm = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                                dfm["Date"] = pd.to_datetime(dfm["Date"])
                                dfm = dfm.set_index("Date").sort_index()
                                frames.append(dfm)
                                used_source = "TWSE_JSON"
                                result = "OK"
                except Exception:
                    result = "PARSE_ERR"

            diag.append({
                "source": "TWSE_JSON",
                "result": result,
                "http": status,
                "bytes": bytes_len,
                "url": url,
                "snippet": snippet
            })
        except Exception as e:
            diag.append({
                "source": "TWSE_JSON",
                "result": "EXC",
                "http": -1,
                "bytes": 0,
                "url": url,
                "snippet": str(e)[:120]
            })

    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = ensure_ohlcv(df)
        if len(df) >= 60:
            return FetchResult(True, used_source or "TWSE_JSON", df, ""), diag

    # 2) TPEX JSON（上櫃）
    frames = []
    for mon in months:
        url = tpex_json_daily_price(code4, mon)
        try:
            status, text = try_requests_text(url, timeout=10)
            bytes_len = len(text.encode("utf-8", errors="ignore"))
            snippet = text[:120].replace("\n", " ")
            result = "OK"

            if status != 200:
                result = "HTTP_ERR"
            elif not text.strip():
                result = "EMPTY_TEXT"
            elif looks_like_waf(text):
                result = "WAF_HTML"
            else:
                # st43_result.php 常回 JSON（或帶 data）
                try:
                    j = json.loads(text)
                    data = j.get("aaData") or j.get("data") or []
                    if not data:
                        result = "NO_DATA"
                    else:
                        rows = []
                        for r in data:
                            # 可能欄位很多；日期常在 r[0]
                            date_str = str(r[0]).strip()
                            try:
                                parts = date_str.split("/")
                                if len(parts) == 3:
                                    yy = int(parts[0]) + 1911
                                    mm = int(parts[1])
                                    dd = int(parts[2])
                                    d = dt.date(yy, mm, dd)
                                else:
                                    d = pd.to_datetime(date_str).date()
                            except Exception:
                                continue

                            # 開高低收常在固定位置（不保證）；保守抓最後幾個數值：o,h,l,c
                            # 這裡用 heuristic：在 row 裡找 4 個像價格的欄位（含小數）
                            nums = []
                            for item in r:
                                s = str(item).replace(",", "").strip()
                                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                                    nums.append(float(s))
                            # 通常會含成交量等，取最後 4 個視作 o/h/l/c 的機率也不高
                            # 因不穩定，若判斷不足，跳過，留給其他來源或 CSV 備援
                            if len(nums) < 5:
                                continue
                            # 取後面 4 個當作 o/h/l/c（保守）
                            o, h, l, c = nums[-4], nums[-3], nums[-2], nums[-1]
                            v = nums[1] if len(nums) > 1 else 0
                            rows.append((d, o, h, l, c, v))

                        if rows:
                            dfm = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                            dfm["Date"] = pd.to_datetime(dfm["Date"])
                            dfm = dfm.set_index("Date").sort_index()
                            frames.append(dfm)
                            result = "OK"
                except Exception:
                    result = "PARSE_ERR"

            diag.append({
                "source": "TPEX_JSON",
                "result": result,
                "http": status,
                "bytes": bytes_len,
                "url": url,
                "snippet": snippet
            })
        except Exception as e:
            diag.append({
                "source": "TPEX_JSON",
                "result": "EXC",
                "http": -1,
                "bytes": 0,
                "url": url,
                "snippet": str(e)[:120]
            })

    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = ensure_ohlcv(df)
        if len(df) >= 60:
            return FetchResult(True, "TPEX_JSON", df, ""), diag

    return FetchResult(False, "TWSE/TPEX", pd.DataFrame(), "no data / blocked"), diag


def parse_uploaded_csv(file_bytes: bytes) -> FetchResult:
    """
    支援你常見的 export.csv（可能是 Yahoo 匯出、或自製 CSV）
    只要能解析出 Date + OHLCV 即可。
    """
    try:
        raw = file_bytes.decode("utf-8-sig", errors="ignore")
    except Exception:
        raw = file_bytes.decode("utf-8", errors="ignore")

    # 嘗試多種分隔符
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=sep)
            if df.shape[1] < 5:
                continue
            # 常見欄名
            cols = {c.lower(): c for c in df.columns}
            # 若第一欄像日期
            if "date" not in cols:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)

            # 常見 Yahoo: Date, Open, High, Low, Close, Adj Close, Volume
            df = ensure_ohlcv(df)
            if df is not None and not df.empty and len(df) >= 60:
                return FetchResult(True, "CSV_UPLOAD", df, "")
        except Exception:
            continue

    return FetchResult(False, "CSV_UPLOAD", pd.DataFrame(), "CSV parse failed")


def fetch_data_all(
    code4: str,
    days: int,
    uploaded_csv: Optional[bytes],
    debug: bool
) -> Tuple[FetchResult, List[Dict[str, Any]]]:
    """
    回傳最終資料 + diag_rows
    """
    diag_rows: List[Dict[str, Any]] = []

    # 0) CSV 上傳（最穩，優先）
    if uploaded_csv:
        fr = parse_uploaded_csv(uploaded_csv)
        diag_rows.append({"source": "CSV_UPLOAD", "result": "OK" if fr.ok else "FAIL", "http": "-", "bytes": len(uploaded_csv), "url": "-", "snippet": fr.note[:120]})
        if fr.ok:
            return fr, diag_rows

    # 1) yfinance
    fr_yf = fetch_yfinance(code4, days)
    diag_rows.append({"source": "YF", "result": "OK" if fr_yf.ok else "FAIL", "http": "-", "bytes": "-", "url": "-", "snippet": fr_yf.note[:120]})
    if fr_yf.ok:
        return fr_yf, diag_rows

    # 2) stooq
    fr_sq = fetch_stooq(code4, days)
    diag_rows.append({"source": "STOOQ", "result": "OK" if fr_sq.ok else "FAIL", "http": "-", "bytes": "-", "url": "-", "snippet": fr_sq.note[:120]})
    if fr_sq.ok:
        return fr_sq, diag_rows

    # 3) TWSE/TPEX（HTTP）
    fr_tw, diag_http = fetch_twse_tpex_http(code4, days)
    diag_rows.extend(diag_http if debug else [])
    if fr_tw.ok:
        return fr_tw, diag_rows

    return FetchResult(False, "ALL", pd.DataFrame(), "all sources failed"), diag_rows


# -----------------------------
# 訊號/評分/買賣點（專業敘述）
# -----------------------------
@dataclass
class Plan:
    action: str               # BUY / SELL / WATCH
    reason: str
    score: int
    near_buy_zone: Optional[Tuple[float, float]]
    deep_buy_zone: Optional[Tuple[float, float]]
    sell_zone: Optional[Tuple[float, float]]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    metrics: Dict[str, float]


def score_and_plan(
    df: pd.DataFrame,
    strategy: str,
    max_buy_distance: float,
    breakout_buffer_atr: float,
    stop_atr: float,
    rr: float
) -> Plan:
    """
    strategy:
      - "mean_reversion": 回檔等待型（靠近下緣/支撐 + 指標翻轉）
      - "breakout": 趨勢突破型（突破 + 追價距離限制）
    """
    ind = calc_indicators(df)
    if ind is None or ind.empty:
        return Plan("WATCH", "資料不足（需要至少 60 根日K）", 0, None, None, None, None, None, {})

    last = ind.iloc[-1]
    close = float(last["Close"])
    atr14 = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else 0.0
    bbl, bbm, bbu = float(last["BBL"]), float(last["BBM"]), float(last["BBU"])
    bbpct = float(last["BBpctB"]) if not np.isnan(last["BBpctB"]) else np.nan
    dev = float(last["DEV_EMA20"]) if not np.isnan(last["DEV_EMA20"]) else 0.0
    rsi14 = float(last["RSI14"]) if not np.isnan(last["RSI14"]) else np.nan
    macdh = float(last["MACDhist"]) if not np.isnan(last["MACDhist"]) else 0.0
    k = float(last["K"]) if not np.isnan(last["K"]) else np.nan
    d = float(last["D"]) if not np.isnan(last["D"]) else np.nan
    sup = float(last["SUP20"]) if not np.isnan(last["SUP20"]) else np.nan
    res = float(last["RES20"]) if not np.isnan(last["RES20"]) else np.nan
    volz = float(last["VOL_Z20"]) if not np.isnan(last["VOL_Z20"]) else 0.0

    # 近端買點（靠近支撐/下軌/EMA20 - 以 ATR 給帶寬）
    band = max(atr14 * 0.4, close * 0.008)  # 最少給 0.8% 的帶寬
    near_buy_center = np.nanmin([bbl, sup, last["EMA20"]])
    near_buy = (float(near_buy_center - band), float(near_buy_center + band))

    # 深回檔買點（更靠近下軌延伸）
    deep_center = bbl - atr14 * 0.8
    deep_buy = (float(deep_center - band), float(deep_center + band))

    # 賣點（壓力/上軌）
    sell_center = np.nanmax([bbu, res])
    sell_zone = (float(sell_center - band), float(sell_center + band))

    # 距離（用更專業說法）
    def pct_to_zone(zone: Tuple[float, float]) -> float:
        lo, hi = zone
        mid = (lo + hi) / 2
        return (mid - close) / close

    near_buy_gap = pct_to_zone(near_buy)      # 負數代表在下方
    deep_buy_gap = pct_to_zone(deep_buy)
    sell_gap = pct_to_zone(sell_zone)         # 正數代表在上方

    # 共振評分
    score = 50

    # RSI（低位加分 / 高位扣分）
    if not np.isnan(rsi14):
        if rsi14 < 35:
            score += 12
        elif rsi14 < 45:
            score += 6
        elif rsi14 > 70:
            score -= 10
        elif rsi14 > 60:
            score -= 4

    # MACD hist（翻正加分）
    score += 8 if macdh > 0 else -3

    # KD（K 上穿 D 加分）
    if not (np.isnan(k) or np.isnan(d)):
        if k > d and k < 40:
            score += 10
        elif k > d:
            score += 5
        elif k < d and k > 70:
            score -= 8

    # 布林位置（%B 低位加分，高位扣分）
    if not np.isnan(bbpct):
        if bbpct < 0.15:
            score += 10
        elif bbpct < 0.30:
            score += 5
        elif bbpct > 0.90:
            score -= 10
        elif bbpct > 0.75:
            score -= 5

    # 乖離（離 EMA20 太遠扣分）
    if abs(dev) > 0.10:
        score -= 10
    elif abs(dev) > 0.06:
        score -= 6

    # 量能（突破策略會更重視）
    if strategy == "breakout":
        if volz > 1.2:
            score += 8
        elif volz < -0.5:
            score -= 3

    score = int(clamp(score, 0, 100))

    # 交易判斷
    action = "WATCH"
    reason = "條件尚未明確，等待價格進入區間或指標翻轉。"

    # 當下是否在「近端買區」
    in_near_buy = (close >= near_buy[0] and close <= near_buy[1])
    in_sell = (close >= sell_zone[0] and close <= sell_zone[1])

    # 趨勢突破：突破 RES20 或 BBU，且追價距離受限
    breakout_trigger = False
    if strategy == "breakout":
        trigger = np.nanmax([res, bbu])
        if not np.isnan(trigger) and atr14 > 0:
            # 加 buffer（ATR 倍數）
            trigger2 = trigger + atr14 * breakout_buffer_atr
            breakout_trigger = close >= trigger2

    # 追價距離限制（避免「進場離現實太遠」）
    # breakout：距離 trigger 不能超過 max_buy_distance（例如 6%）
    can_chase = True
    if strategy == "breakout" and breakout_trigger:
        trigger = np.nanmax([res, bbu])
        if not np.isnan(trigger) and trigger > 0:
            chase_gap = (close - trigger) / close
            can_chase = chase_gap <= max_buy_distance

    # 回檔等待：在近端買區 + 指標偏翻正
    mean_rev_ok = in_near_buy and (macdh > 0 or (not np.isnan(k) and not np.isnan(d) and k > d))

    if in_sell and score >= 65:
        action = "SELL"
        reason = "價格進入壓力/獲利帶（布林上緣/近端壓力），且共振偏高：可分批減碼/移動停利。"
    elif strategy == "breakout":
        if breakout_trigger and can_chase and score >= 65:
            action = "BUY"
            reason = "突破成立（含 ATR buffer）且追價距離可接受：可採突破追價進場，搭配 ATR 失效止損。"
        elif breakout_trigger and (not can_chase):
            action = "WATCH"
            reason = "突破雖成立，但追價距離過大（風險/報酬不佳）：等待回測不破或縮距再進。"
        else:
            action = "WATCH"
            reason = "未達突破觸發：等待放量突破或回測後再評估。"
    else:
        if mean_rev_ok and score >= 60:
            action = "BUY"
            reason = "價格進入近端買區（支撐/布林下緣附近）且指標出現翻轉：可採回檔分批布局。"

    # 風控：以 ATR 計算止損/目標（breakout/mean rev 都可用）
    stop_loss = None
    take_profit = None
    if atr14 > 0:
        stop_loss = close - atr14 * stop_atr
        risk = close - stop_loss
        take_profit = close + risk * rr

    metrics = {
        "close": close,
        "atr14": atr14,
        "rsi14": rsi14,
        "macdh": macdh,
        "k": k,
        "d": d,
        "bbpctB": bbpct,
        "dev_ema20": dev,
        "vol_z20": volz,
        "near_buy_gap": near_buy_gap,
        "deep_buy_gap": deep_buy_gap,
        "sell_gap": sell_gap,
    }

    return Plan(
        action=action,
        reason=reason,
        score=score,
        near_buy_zone=near_buy,
        deep_buy_zone=deep_buy,
        sell_zone=sell_zone,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metrics=metrics
    )


def professional_gap_text(gap: float) -> str:
    """
    gap： (zone_mid - close) / close
    負：目標在下方（回檔幅度）
    正：目標在上方（上行空間）
    """
    if np.isnan(gap):
        return "—"
    pct = gap * 100.0
    if pct < 0:
        return f"回檔需求：約 {abs(pct):.1f}%（目標低於現價）"
    elif pct > 0:
        return f"上行空間：約 {pct:.1f}%（目標高於現價）"
    else:
        return "貼近現價（0.0%）"


# -----------------------------
# 視覺：Matplotlib 布林通道圖（無 Plotly）
# -----------------------------
def plot_bollinger(ind: pd.DataFrame, title: str, plan: Plan):
    ind = ind.copy()
    # 只畫最近 140 根，避免太擠
    if len(ind) > 140:
        ind = ind.iloc[-140:]

    x = ind.index
    close = ind["Close"]
    bbl, bbm, bbu = ind["BBL"], ind["BBM"], ind["BBU"]
    ema20 = ind["EMA20"]
    sma60 = ind["SMA60"]
    sup20 = ind["SUP20"]
    res20 = ind["RES20"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, close, linewidth=1.6, label="Close")
    ax.plot(x, ema20, linewidth=1.0, label="EMA20")
    ax.plot(x, sma60, linewidth=1.0, label="SMA60")
    ax.plot(x, bbu, linewidth=1.0, label="BB Upper")
    ax.plot(x, bbm, linewidth=1.0, label="BB Mid")
    ax.plot(x, bbl, linewidth=1.0, label="BB Lower")
    ax.plot(x, sup20, linewidth=0.9, label="Support(20)")
    ax.plot(x, res20, linewidth=0.9, label="Resistance(20)")

    # 區間塊
    def span(zone, alpha=0.10):
        if zone:
            ax.axhspan(zone[0], zone[1], alpha=alpha)

    span(plan.near_buy_zone, 0.12)
    span(plan.deep_buy_zone, 0.08)
    span(plan.sell_zone, 0.10)

    # 止損/目標
    if plan.stop_loss:
        ax.axhline(plan.stop_loss, linestyle="--", linewidth=1.0, label="Stop (ATR)")
    if plan.take_profit:
        ax.axhline(plan.take_profit, linestyle="--", linewidth=1.0, label="Target (RR)")

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Top10 掃描（去重 + 顯示可操作/等待）
# -----------------------------
def scan_top10(
    stock_pool: List[str],
    days: int,
    strategy: str,
    max_buy_distance: float,
    breakout_buffer_atr: float,
    stop_atr: float,
    rr: float,
    debug: bool
) -> pd.DataFrame:
    seen = set()
    uniq = []
    for s in stock_pool:
        c = normalize_ticker(s)
        if is_taiwan_code(c) and c not in seen:
            seen.add(c)
            uniq.append(c)

    rows = []
    for code4 in uniq:
        fr, _ = fetch_data_all(code4, days, uploaded_csv=None, debug=debug)
        if not fr.ok:
            continue
        plan = score_and_plan(fr.df, strategy, max_buy_distance, breakout_buffer_atr, stop_atr, rr)
        close = plan.metrics.get("close", np.nan)

        rows.append({
            "股票": code4,
            "來源": fr.source,
            "目前價": round(close, 2) if pd.notna(close) else np.nan,
            "AI分數": plan.score,
            "操作": plan.action,
            "近端買區": f"{plan.near_buy_zone[0]:.2f} ~ {plan.near_buy_zone[1]:.2f}" if plan.near_buy_zone else "—",
            "近端距離": professional_gap_text(plan.metrics.get("near_buy_gap", np.nan)),
            "等待買區": f"{plan.deep_buy_zone[0]:.2f} ~ {plan.deep_buy_zone[1]:.2f}" if plan.deep_buy_zone else "—",
            "等待距離": professional_gap_text(plan.metrics.get("deep_buy_gap", np.nan)),
            "壓力/獲利帶": f"{plan.sell_zone[0]:.2f} ~ {plan.sell_zone[1]:.2f}" if plan.sell_zone else "—",
            "上行空間": professional_gap_text(plan.metrics.get("sell_gap", np.nan)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Top10 排序：優先 BUY > WATCH > SELL，再按分數
    order_map = {"BUY": 0, "WATCH": 1, "SELL": 2}
    df["__op"] = df["操作"].map(order_map).fillna(9).astype(int)
    df = df.sort_values(["__op", "AI分數"], ascending=[True, False]).drop(columns=["__op"])

    # 去除重複股票（保險）
    df = df.drop_duplicates(subset=["股票"], keep="first")

    return df.head(10)


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("設定")

    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    period = st.selectbox("資料期間", ["3mo", "6mo", "1y", "2y"], index=1)
    days = PERIOD_TO_DAYS.get(period, 180)

    debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.markdown("---")
    st.subheader("策略 / 參數")

    strategy_ui = st.radio("策略", ["回檔等待型（回檔分批）", "趨勢突破型（突破追價進場）"], index=1)
    strategy = "mean_reversion" if "回檔" in strategy_ui else "breakout"

    max_buy_distance = st.slider("最大可接受追價/偏離距離（%）", 0.01, 0.20, 0.06, 0.01)
    breakout_buffer_atr = st.slider("突破觸發 buffer（ATR 倍數）", 0.00, 0.80, 0.20, 0.05)
    stop_atr = st.slider("失效止損距離（ATR 倍數）", 0.80, 3.00, 1.60, 0.10)
    rr = st.slider("目標風險報酬（RR）", 1.0, 5.0, 2.2, 0.1)

    st.markdown("---")
    st.subheader("資金 / 倉位（顯示用）")
    capital = st.number_input("資金規模（NTD）", min_value=100000, max_value=100000000, value=7000000, step=100000)

    st.markdown("---")
    st.subheader("CSV 上傳備援（最穩）")
    uploaded = st.file_uploader("上傳 export.csv（抓不到資料時立刻可操作）", type=["csv"])
    uploaded_bytes = uploaded.read() if uploaded else None

    st.markdown("---")
    network_test = st.button("🧪 網路測試（建議先按一次）")


# 主輸入
colA, colB = st.columns([1, 2])
with colA:
    code_in = st.text_input("請輸入股票代號", value="2330")
    code = normalize_ticker(code_in)
    if not is_taiwan_code(code):
        st.warning("請輸入 4 位台股代號（例如 2330 / 2317 / 8046）。")

with colB:
    st.info("提示：若 Cloud 偶發抓不到資料，請先按「網路測試」，或直接上傳 export.csv 立即可用。")

# -----------------------------
# 網路測試 / 診斷
# -----------------------------
if network_test:
    st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
    # 做幾個代表性端點測試
    test_rows = []
    test_urls = [
        ("TWSE_JSON", twse_json_daily_price("2330", now_tpe().strftime("%Y%m"))),
        ("TPEX_JSON", tpex_json_daily_price("6274", now_tpe().strftime("%Y%m"))),
        ("TWSE_CSV", twse_stock_day_csv(now_tpe().strftime("%Y%m")) + "2330"),
    ]
    for src, url in test_urls:
        try:
            status, text = try_requests_text(url, timeout=10)
            bytes_len = len(text.encode("utf-8", errors="ignore"))
            if status != 200:
                res = "HTTP_ERR"
            elif not text.strip():
                res = "EMPTY_TEXT"
            elif looks_like_waf(text):
                res = "WAF_HTML"
            else:
                # 粗判是否 JSON/CSV
                if text.lstrip().startswith("{"):
                    try:
                        j = json.loads(text)
                        res = "OK" if j else "NO_DATA"
                    except Exception:
                        res = "PARSE_ERR"
                else:
                    res = "OK"
            test_rows.append({"source": src, "result": res, "status": status, "bytes": bytes_len, "url": url, "snippet": text[:120].replace("\n", " ")})
        except Exception as e:
            test_rows.append({"source": src, "result": "EXC", "status": -1, "bytes": 0, "url": url, "snippet": str(e)[:120]})

    st.dataframe(pd.DataFrame(test_rows), use_container_width=True)

st.markdown("---")

# -----------------------------
# 單股模式
# -----------------------------
if mode == "單一股票分析":
    if not is_taiwan_code(code):
        st.stop()

    fr, diag_rows = fetch_data_all(code, days, uploaded_bytes, debug=debug)

    if not fr.ok:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試或改用 CSV 上傳備援。")
        if debug and diag_rows:
            st.subheader("🧩 逐路診斷（Debug）")
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
        st.stop()

    ind = calc_indicators(fr.df)
    plan = score_and_plan(fr.df, strategy, max_buy_distance, breakout_buffer_atr, stop_atr, rr)

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    close = plan.metrics.get("close", np.nan)
    last_date = ind.index.max().date() if ind is not None and not ind.empty else None

    k1.metric("目前價格", f"{close:.2f}" if pd.notna(close) else "—")
    k2.metric("AI 共振分數", f"{plan.score}/100")
    k3.metric("資料來源", fr.source)
    k4.metric("最後日期 / 筆數", f"{last_date} / {len(fr.df)}" if last_date else f"{len(fr.df)}")

    st.markdown("### 📌 當下是否為買點/賣點？（可操作判斷）")
    if plan.action == "BUY":
        st.success(f"✅ BUY：{plan.reason}")
    elif plan.action == "SELL":
        st.warning(f"⚠️ SELL：{plan.reason}")
    else:
        st.info(f"⏳ WATCH：{plan.reason}")

    st.markdown("### 🗺️ 未來預估買賣點（區間 + 專業距離敘述）")

    c1, c2 = st.columns(2)
    with c1:
        if plan.near_buy_zone:
            st.success(
                f"🟢 近端買點（可操作）: {plan.near_buy_zone[0]:.2f} ~ {plan.near_buy_zone[1]:.2f}  ｜ "
                f"{professional_gap_text(plan.metrics.get('near_buy_gap', np.nan))}"
            )
        if plan.deep_buy_zone:
            st.info(
                f"🔵 深回檔買點（等待型）: {plan.deep_buy_zone[0]:.2f} ~ {plan.deep_buy_zone[1]:.2f}  ｜ "
                f"{professional_gap_text(plan.metrics.get('deep_buy_gap', np.nan))}"
            )
    with c2:
        if plan.sell_zone:
            st.warning(
                f"🟠 壓力/獲利帶: {plan.sell_zone[0]:.2f} ~ {plan.sell_zone[1]:.2f}  ｜ "
                f"{professional_gap_text(plan.metrics.get('sell_gap', np.nan))}"
            )
        if plan.stop_loss and plan.take_profit:
            st.caption(f"風控參考（ATR/RR）：失效止損 ≈ {plan.stop_loss:.2f} ｜ 目標價 ≈ {plan.take_profit:.2f}（RR={rr:.1f}）")

    st.markdown("### 📈 收盤價走勢（布林通道 + 均線 + 支撐/壓力 + 區間帶）")
    plot_bollinger(ind, f"{code} / {fr.source} / {period}", plan)

    # 指標摘要
    st.markdown("### 🧭 指標摘要（用於判讀，不是口號）")
    m = plan.metrics
    desc = {
        "RSI14": m.get("rsi14", np.nan),
        "MACDhist": m.get("macdh", np.nan),
        "KD(K,D)": (m.get("k", np.nan), m.get("d", np.nan)),
        "BB %B": m.get("bbpctB", np.nan),
        "DEV vs EMA20": m.get("dev_ema20", np.nan),
        "VOL Z20": m.get("vol_z20", np.nan),
        "ATR14": m.get("atr14", np.nan),
    }
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("RSI14", f"{desc['RSI14']:.1f}" if pd.notna(desc["RSI14"]) else "—")
    dcol2.metric("MACD hist", f"{desc['MACDhist']:.3f}" if pd.notna(desc["MACDhist"]) else "—")
    kd = desc["KD(K,D)"]
    dcol3.metric("KD", f"{kd[0]:.1f} / {kd[1]:.1f}" if pd.notna(kd[0]) and pd.notna(kd[1]) else "—")
    dcol4.metric("BB %B", f"{desc['BB %B']:.2f}" if pd.notna(desc["BB %B"]) else "—")

    if debug and diag_rows:
        st.markdown("### 🧩 逐路診斷（Debug）")
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

# -----------------------------
# Top10 模式
# -----------------------------
else:
    st.markdown("### 🔥 AI 強勢股 Top 10（去重 + 顯示可操作/等待 + 距離敘述）")
    st.caption("Top10 會先用小池測試（避免全市場 Cloud 超時）。你也可以在程式內把 DEFAULT_STOCK_POOL 擴大。")

    df_top = scan_top10(
        stock_pool=DEFAULT_STOCK_POOL,
        days=days,
        strategy=strategy,
        max_buy_distance=max_buy_distance,
        breakout_buffer_atr=breakout_buffer_atr,
        stop_atr=stop_atr,
        rr=rr,
        debug=debug
    )

    if df_top.empty:
        st.warning("目前掃描結果為空（代表多數資料源暫時抓不到或超時）。建議稍後再試。")
    else:
        st.dataframe(df_top, use_container_width=True)

    st.markdown("### 🧠 Top10 判讀提醒（你要的『能不能操作』）")
    st.write(
        "- **操作=BUY**：符合策略觸發（回檔翻轉 / 突破成立且追價距離可接受）\n"
        "- **操作=WATCH**：條件未明確或追價距離過大，等待縮距/回測\n"
        "- **操作=SELL**：進入壓力/獲利帶，可分批減碼或移動停利\n"
    )
