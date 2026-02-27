# app.py
# AI 台股量化專業平台（V23：趨勢突破型 / 無 Plotly / 全功能保留）
# ✅ 單一股票分析 + Top10 掃描器
# ✅ 逐路診斷（每條資料源的 HTTP / bytes / preview）
# ✅ 指標：MA/EMA、MACD、KD(Stoch)、RSI、ATR、布林通道、量能
# ✅ 趨勢突破策略：突破觸發價 / 回測買點 / 失效止損 / 目標價
# ✅ 專業距離：溢價/折價%、ATR 標準化距離、Entry Efficiency
# ✅ 倉位建議（500–1000萬）：依 ATR 止損距離 + 風險% + 最大持倉% 自動計算
#
# requirements.txt（建議）
# streamlit
# pandas
# numpy
# requests
# matplotlib
# yfinance
# pandas_datareader

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# optional sources
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="AI 台股量化專業平台（V23 Breakout）", layout="wide")
st.markdown("# 🧠 AI 台股量化專業平台（V23 / 趨勢突破型 / 無 Plotly / 全功能保留）")
st.caption("多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + 突破交易計畫（不自動下單）")


# -----------------------------
# Diagnostics model
# -----------------------------
@dataclass
class FetchAttempt:
    source: str
    url: str
    status: str
    http: Optional[int] = None
    bytes: Optional[int] = None
    note: str = ""
    preview: str = ""


# -----------------------------
# HTTP (Session + headers)
# -----------------------------
@st.cache_resource
def get_session() -> requests.Session:
    s = requests.Session()
    return s


def _requests_get(url: str, timeout: int = 12) -> Tuple[Optional[requests.Response], Optional[str], str]:
    """
    Cloud 常被擋：加 headers + session + retry + backoff
    回傳 (response, error, preview_text)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.twse.com.tw/zh/trading/historical/stock-day.html",
        "X-Requested-With": "XMLHttpRequest",
    }

    sess = get_session()
    last_err = None
    preview = ""
    for i in range(3):
        try:
            r = sess.get(url, headers=headers, timeout=timeout)
            txt = r.text or ""
            preview = txt[:160].replace("\n", " ")
            return r, None, preview
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (i + 1))

    return None, last_err, preview


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", "").strip()
            if x in ["", "--", "—", "NA", "N/A", "null", "None"]:
                return None
        return float(x)
    except Exception:
        return None


def _yyyymmdd(dt: datetime) -> str:
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date")

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # normalize column names
    col_map = {c.lower(): c for c in out.columns}

    def pick(names: List[str]) -> Optional[str]:
        for n in names:
            if n.lower() in col_map:
                return col_map[n.lower()]
        return None

    o = pick(["open"])
    h = pick(["high"])
    l = pick(["low"])
    c = pick(["close", "adj close"])
    v = pick(["volume", "成交股數", "成交量"])

    if c is None:
        raise ValueError("No Close column")

    rename = {}
    if o: rename[o] = "Open"
    if h: rename[h] = "High"
    if l: rename[l] = "Low"
    rename[c] = "Close"
    if v: rename[v] = "Volume"

    out = out.rename(columns=rename)

    for k in ["Open", "High", "Low"]:
        if k not in out.columns:
            out[k] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.dropna(subset=["Close"])
    return out


# -----------------------------
# Indicators (no ta-lib / no plotly)
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stoch_kd(df: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return lower, mid, upper


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA60"] = sma(out["Close"], 60)
    out["EMA20"] = ema(out["Close"], 20)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ATR14"] = atr(out, 14)

    macd_line, sig_line, hist = macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_SIG"] = sig_line
    out["MACD_HIST"] = hist

    k, d = stoch_kd(out, 9, 3)
    out["K"] = k
    out["D"] = d

    bb_l, bb_m, bb_u = bollinger(out["Close"], 20, 2.0)
    out["BB_L"] = bb_l
    out["BB_M"] = bb_m
    out["BB_U"] = bb_u
    out["BB_W"] = (bb_u - bb_l) / bb_m

    out["VMA20"] = sma(out["Volume"].fillna(method="ffill"), 20)
    return out


# -----------------------------
# Confluence score
# -----------------------------
def confluence_score(last: pd.Series) -> Tuple[int, Dict[str, bool]]:
    flags = {}
    flags["price_above_ema20"] = bool(last["Close"] > last["EMA20"]) if pd.notna(last["EMA20"]) else False
    flags["ema20_above_sma60"] = bool(last["EMA20"] > last["SMA60"]) if pd.notna(last["SMA60"]) else False
    flags["macd_bull"] = bool(last["MACD"] > last["MACD_SIG"]) if pd.notna(last["MACD_SIG"]) else False
    flags["macd_hist_up"] = bool(last["MACD_HIST"] > 0) if pd.notna(last["MACD_HIST"]) else False
    flags["rsi_ok"] = bool(50 <= last["RSI14"] <= 78) if pd.notna(last["RSI14"]) else False
    flags["rsi_overbought"] = bool(last["RSI14"] > 80) if pd.notna(last["RSI14"]) else False
    flags["kd_bull"] = bool(last["K"] > last["D"]) if pd.notna(last["D"]) else False
    flags["vol_up"] = bool(last["Volume"] > last["VMA20"]) if pd.notna(last["VMA20"]) and pd.notna(last["Volume"]) else False
    flags["break_bb_upper"] = bool(last["Close"] >= last["BB_U"]) if pd.notna(last["BB_U"]) else False

    score = 0
    score += 16 if flags["price_above_ema20"] else 0
    score += 10 if flags["ema20_above_sma60"] else 0
    score += 14 if flags["macd_bull"] else 0
    score += 10 if flags["macd_hist_up"] else 0
    score += 10 if flags["kd_bull"] else 0
    score += 14 if flags["vol_up"] else 0
    score += 12 if flags["rsi_ok"] else 0
    score += 10 if flags["break_bb_upper"] else 0
    score -= 12 if flags["rsi_overbought"] else 0

    return int(np.clip(score, 0, 100)), flags


# -----------------------------
# Data Sources
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if yf is None:
        return None, FetchAttempt("YF", "", "NO_MODULE", note="yfinance not installed")

    tickers = [code] if (code.endswith(".TW") or code.endswith(".TWO")) else [f"{code}.TW", f"{code}.TWO"]
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days + 10)

    last_note = ""
    for t in tickers:
        try:
            df = yf.download(t, start=str(start), end=str(end), progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                last_note = "empty"
                continue
            df = df.reset_index()
            df = _normalize_ohlcv(df)
            return df, FetchAttempt("YF", f"yfinance:{t}", "OK", note=f"rows={len(df)}")
        except Exception as e:
            last_note = str(e)
            continue

    return None, FetchAttempt("YF", "yfinance", "FAIL", note=last_note or "download failed")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_twse_json(code: str, months_back: int) -> Tuple[Optional[pd.DataFrame], List[FetchAttempt]]:
    """
    TWSE JSON: https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date=YYYYMMDD&stockNo=2330
    """
    attempts: List[FetchAttempt] = []
    frames = []

    today = datetime.now()
    for m in range(months_back):
        dt = (today.replace(day=1) - pd.DateOffset(months=m)).to_pydatetime()
        ymd = _yyyymmdd(dt)
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={ymd}&stockNo={code}"

        r, err, preview = _requests_get(url)
        if r is None:
            attempts.append(FetchAttempt("TWSE_JSON", url, "EXC", note=err or "request failed", preview=preview))
            continue

        text = r.text or ""
        if r.status_code != 200:
            attempts.append(FetchAttempt("TWSE_JSON", url, "HTTP_ERR", http=r.status_code, bytes=len(text), preview=preview))
            continue

        if len(text.strip()) < 80:
            attempts.append(FetchAttempt("TWSE_JSON", url, "TOO_SHORT", http=200, bytes=len(text), preview=preview))
            continue

        # if WAF returns HTML, json() will fail; we capture preview
        try:
            j = r.json()
            data = j.get("data", [])
            fields = j.get("fields", [])
            if not data or not fields:
                attempts.append(FetchAttempt("TWSE_JSON", url, "NO_DATA", http=200, bytes=len(text), preview=preview))
                continue

            dfm = pd.DataFrame(data, columns=fields)

            col_date = "日期"
            col_open = "開盤價"
            col_high = "最高價"
            col_low = "最低價"
            col_close = "收盤價"
            col_vol = "成交股數" if "成交股數" in dfm.columns else ("成交量" if "成交量" in dfm.columns else None)

            def parse_roc(s: str) -> datetime:
                p = str(s).strip().split("/")
                y = int(p[0]) + 1911
                return datetime(y, int(p[1]), int(p[2]))

            out = pd.DataFrame({
                "Date": dfm[col_date].map(parse_roc),
                "Open": dfm[col_open].map(_safe_float) if col_open in dfm.columns else None,
                "High": dfm[col_high].map(_safe_float) if col_high in dfm.columns else None,
                "Low": dfm[col_low].map(_safe_float) if col_low in dfm.columns else None,
                "Close": dfm[col_close].map(_safe_float),
                "Volume": dfm[col_vol].map(_safe_float) if col_vol and col_vol in dfm.columns else None,
            }).dropna(subset=["Close"])

            out = _normalize_ohlcv(out)
            frames.append(out)
            attempts.append(FetchAttempt("TWSE_JSON", url, "OK", http=200, bytes=len(text), note=f"rows={len(out)}"))
        except Exception as e:
            attempts.append(FetchAttempt("TWSE_JSON", url, "PARSE_ERR", http=200, bytes=len(text), note=str(e), preview=preview))

    if not frames:
        return None, attempts

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, attempts


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tpex_html(code: str, months_back: int) -> Tuple[Optional[pd.DataFrame], List[FetchAttempt]]:
    """
    TPEX（上櫃）常見頁面會回 HTML table：
    https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d=114/02&stkno=6274

    我們逐月拉，使用 pandas.read_html 解析表格（比 JSON 更耐擋）
    """
    attempts: List[FetchAttempt] = []
    frames = []

    today = datetime.now().replace(day=1)

    for m in range(months_back):
        dt = (today - pd.DateOffset(months=m)).to_pydatetime()
        roc_y = dt.year - 1911
        d_param = f"{roc_y}/{dt.month:02d}"
        url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={d_param}&stkno={code}"

        r, err, preview = _requests_get(url)
        if r is None:
            attempts.append(FetchAttempt("TPEX_HTML", url, "EXC", note=err or "request failed", preview=preview))
            continue

        text = r.text or ""
        if r.status_code != 200:
            attempts.append(FetchAttempt("TPEX_HTML", url, "HTTP_ERR", http=r.status_code, bytes=len(text), preview=preview))
            continue

        if len(text.strip()) < 200:
            attempts.append(FetchAttempt("TPEX_HTML", url, "TOO_SHORT", http=200, bytes=len(text), preview=preview))
            continue

        try:
            tables = pd.read_html(text)
            if not tables:
                attempts.append(FetchAttempt("TPEX_HTML", url, "NO_TABLE", http=200, bytes=len(text), preview=preview))
                continue

            # usually first table contains daily data
            t = tables[0].copy()

            # Try to detect columns
            # Typical columns might include: 日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數...
            cols = [str(c).strip() for c in t.columns]
            # Find date column
            date_col = None
            for c in cols:
                if "日期" in c:
                    date_col = c
                    break
            if date_col is None:
                attempts.append(FetchAttempt("TPEX_HTML", url, "SCHEMA_MISS", http=200, bytes=len(text), preview=preview))
                continue

            def parse_roc_date(s: str) -> datetime:
                # may be "114/02/27"
                p = str(s).strip().split("/")
                y = int(p[0]) + 1911
                return datetime(y, int(p[1]), int(p[2]))

            # close/open/high/low columns
            def find_col(keys: List[str]) -> Optional[str]:
                for k in keys:
                    for c in cols:
                        if k in c:
                            return c
                return None

            c_open = find_col(["開盤"])
            c_high = find_col(["最高"])
            c_low = find_col(["最低"])
            c_close = find_col(["收盤"])
            c_vol = find_col(["成交股數", "成交量"])

            if c_close is None:
                attempts.append(FetchAttempt("TPEX_HTML", url, "SCHEMA_MISS", http=200, bytes=len(text), preview=preview))
                continue

            out = pd.DataFrame({
                "Date": t[date_col].map(parse_roc_date),
                "Open": t[c_open].map(_safe_float) if c_open else None,
                "High": t[c_high].map(_safe_float) if c_high else None,
                "Low": t[c_low].map(_safe_float) if c_low else None,
                "Close": t[c_close].map(_safe_float),
                "Volume": t[c_vol].map(_safe_float) if c_vol else None,
            }).dropna(subset=["Close"])

            out = _normalize_ohlcv(out)
            frames.append(out)
            attempts.append(FetchAttempt("TPEX_HTML", url, "OK", http=200, bytes=len(text), note=f"rows={len(out)}"))
        except Exception as e:
            attempts.append(FetchAttempt("TPEX_HTML", url, "PARSE_ERR", http=200, bytes=len(text), note=str(e), preview=preview))

    if not frames:
        return None, attempts

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, attempts


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if pdr is None:
        return None, FetchAttempt("STOOQ", "", "NO_MODULE", note="pandas_datareader not installed")

    candidates = [f"{code}.TW", f"{code}.TWO", code]
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days + 10)

    last_note = ""
    for sym in candidates:
        try:
            df = pdr.DataReader(sym, "stooq", start, end)
            if df is None or df.empty:
                last_note = "empty"
                continue
            df = df.reset_index()
            df = _normalize_ohlcv(df)
            return df, FetchAttempt("STOOQ", f"stooq:{sym}", "OK", note=f"rows={len(df)}")
        except Exception as e:
            last_note = str(e)
            continue

    return None, FetchAttempt("STOOQ", "stooq", "FAIL", note=last_note or "download failed")


def fetch_ohlcv_multi(code: str, months_back: int) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:
    period_days = int(months_back * 31)
    attempts: List[FetchAttempt] = []

    # 1) TWSE JSON
    df1, att1 = fetch_twse_json(code, months_back)
    attempts += att1
    if df1 is not None and len(df1) >= 30:
        return df1, "TWSE_JSON", attempts

    # 2) TPEX HTML (for OTC)
    df2, att2 = fetch_tpex_html(code, months_back)
    attempts += att2
    if df2 is not None and len(df2) >= 30:
        return df2, "TPEX_HTML", attempts

    # 3) Yahoo
    df3, att3 = fetch_yfinance(code, period_days)
    attempts.append(att3)
    if df3 is not None and len(df3) >= 30:
        return df3, "YF", attempts

    # 4) Stooq
    df4, att4 = fetch_stooq(code, period_days)
    attempts.append(att4)
    if df4 is not None and len(df4) >= 30:
        return df4, "STOOQ", attempts

    return None, "NONE", attempts


# -----------------------------
# Breakout Plan (Trend breakout)
# -----------------------------
def support_resistance(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]:
    d = df.tail(lookback)
    sup = float(np.nanmin(d["Low"].values))
    res = float(np.nanmax(d["High"].values))
    return sup, res


def professional_distance(current: float, ref: float, atr_v: float, max_chase_pct: float) -> Dict[str, str]:
    if ref <= 0 or current <= 0:
        return {"pct": "N/A", "atr": "N/A", "eff": "N/A"}

    dist_pct = (current - ref) / ref
    dist_atr = (current - ref) / atr_v if atr_v and atr_v > 0 else np.nan

    denom = max(max_chase_pct, 1e-6)
    eff = 1 - (abs(dist_pct) / denom)
    eff = float(np.clip(eff, 0, 1))

    pct_text = f"{dist_pct*100:+.2f}%（{'追價' if dist_pct>0 else '回測'}距離）"
    atr_text = f"{dist_atr:+.2f} ATR" if not np.isnan(dist_atr) else "N/A"
    eff_text = f"{eff*100:.0f}/100"
    return {"pct": pct_text, "atr": atr_text, "eff": eff_text}


def build_breakout_plan(
    df: pd.DataFrame,
    max_chase_pct: float,
    trigger_buffer_atr: float,
    stop_atr_mult: float,
    target_rr: float,
) -> Dict[str, Dict]:
    """
    趨勢突破型：
    - Breakout Trigger：近 N 日壓力（high）+ buffer
    - Pullback Buy：突破後回測（EMA20 / BB_M）不破
    - Invalidation：跌回 trigger 下方 + ATR stop
    - Targets：RR 目標 & 參考布林上緣/延伸
    """
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_v = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan

    sup, res = support_resistance(df, lookback=60)
    bb_u = float(last["BB_U"]) if pd.notna(last["BB_U"]) else np.nan
    bb_m = float(last["BB_M"]) if pd.notna(last["BB_M"]) else close
    ema20_v = float(last["EMA20"]) if pd.notna(last["EMA20"]) else close

    if np.isnan(atr_v) or atr_v <= 0:
        atr_v = max(1e-6, close * 0.02)

    trigger = res + (trigger_buffer_atr * atr_v)
    # chase guard: if trigger too far above current, keep it as "wait"
    chase_dist = (trigger - close) / close if close > 0 else 999

    # pullback zone: between max(ema20, bb_m) and that - 0.6 ATR
    pb_center = max(ema20_v, bb_m)
    pb_low = pb_center - 0.6 * atr_v
    pb_high = pb_center + 0.2 * atr_v

    # stop: below pullback low (or below trigger) by ATR multiple
    stop = min(pb_low, trigger) - (stop_atr_mult * atr_v)

    # targets: RR from trigger entry baseline
    entry_ref = min(trigger, close * (1 + max_chase_pct))  # realistic reference entry
    risk = max(entry_ref - stop, 0.01)
    target = entry_ref + risk * target_rr

    # cap target with BB upper extension if available (still allow a bit above)
    if not np.isnan(bb_u) and bb_u > 0:
        target = max(target, bb_u)  # breakout expects upper band expansion

    return {
        "trigger": {"price": float(trigger), "chase_dist": float(chase_dist)},
        "pullback": {"low": float(pb_low), "high": float(pb_high), "center": float(pb_center)},
        "risk": {"entry_ref": float(entry_ref), "stop": float(stop), "target": float(target), "atr": float(atr_v)},
        "sr": {"support": float(sup), "resistance": float(res)},
    }


def breakout_action(df: pd.DataFrame, flags: Dict[str, bool], plan: Dict, max_chase_pct: float) -> Tuple[str, str]:
    last = df.iloc[-1]
    close = float(last["Close"])
    trigger = plan["trigger"]["price"]
    pb = plan["pullback"]
    chase_dist = (close - trigger) / trigger if trigger > 0 else 999
    is_breaking = close >= trigger
    is_in_pullback = (pb["low"] <= close <= pb["high"])

    # Breakout BUY condition: price breaks trigger + volume + macd
    if is_breaking and flags.get("vol_up", False) and flags.get("macd_bull", False):
        if chase_dist <= max_chase_pct:
            return "BUY", "突破成立：價格站上突破觸發價，且量能放大 + MACD 翻多（可分批進場，嚴守失效點）。"
        return "WATCH", "突破成立但追價距離過大：等待回測不破（Pullback）再進場，避免高位追價。"

    # Pullback entry: price in pullback zone + momentum not broken
    if is_in_pullback and (flags.get("macd_hist_up", True) or flags.get("kd_bull", False)):
        return "BUY", "回測買點：突破後回測區間內止跌，動能未破（屬於『回測不破』的二次進場）。"

    # Sell / reduce: overbought / upper band + weakening
    if flags.get("rsi_overbought", False) and (not flags.get("macd_hist_up", True)):
        return "SELL", "過熱轉弱：RSI 過熱且動能減弱（可分批減碼/停利）。"

    return "WATCH", "觀望：等待『突破觸發』或『突破後回測』條件成形。"


# -----------------------------
# Position sizing
# -----------------------------
def position_sizing(
    account_ntd: float,
    entry: float,
    stop: float,
    max_position_pct: float,
    risk_pct: float,
) -> Dict[str, float]:
    capital = float(account_ntd)
    risk_amt = capital * risk_pct
    max_pos_amt = capital * max_position_pct

    per_share_risk = max(entry - stop, 0.0)
    if per_share_risk <= 0:
        return {"risk_amt": risk_amt, "per_share_risk": per_share_risk, "shares": 0.0, "pos_value": 0.0, "pos_pct": 0.0}

    shares_by_risk = risk_amt / per_share_risk
    shares_by_cap = max_pos_amt / entry if entry > 0 else 0.0
    shares = max(0.0, min(shares_by_risk, shares_by_cap))
    shares_int = math.floor(shares)

    pos_value = shares_int * entry
    pos_pct = pos_value / capital if capital > 0 else 0.0

    return {"risk_amt": risk_amt, "per_share_risk": per_share_risk, "shares": float(shares_int), "pos_value": float(pos_value), "pos_pct": float(pos_pct)}


# -----------------------------
# Charts (matplotlib)
# -----------------------------
def plot_bollinger(df: pd.DataFrame, code: str):
    d = df.tail(180).copy()
    fig = plt.figure(figsize=(12, 5.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d.index, d["Close"], linewidth=1.2, label="Close")
    ax.plot(d.index, d["BB_M"], linewidth=1.0, label="BB Mid (SMA20)")
    ax.plot(d.index, d["BB_U"], linewidth=1.0, label="BB Upper")
    ax.plot(d.index, d["BB_L"], linewidth=1.0, label="BB Lower")
    ax.fill_between(d.index, d["BB_L"].values, d["BB_U"].values, alpha=0.12)
    ax.set_title(f"{code} 布林通道（近 180 根）")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)


# -----------------------------
# UI: Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("設定")
    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    months_map = {"3mo": 3, "6mo": 6, "1y": 12}
    period_label = st.selectbox("資料期間", list(months_map.keys()), index=1)
    months_back = months_map[period_label]

    debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.divider()
    st.subheader("策略：趨勢突破型（你選的）")

    max_chase_pct = st.slider("最大可接受追價距離（%）", 0.02, 0.18, 0.06, 0.01)  # 追價超過就不買，等回測
    trigger_buffer_atr = st.slider("突破觸發 buffer（ATR 倍數）", 0.0, 0.8, 0.2, 0.05)
    stop_atr_mult = st.slider("失效止損距離（ATR 倍數）", 0.8, 3.0, 1.6, 0.1)
    target_rr = st.slider("目標風險報酬（RR）", 1.2, 5.0, 2.2, 0.1)

    st.divider()
    st.subheader("資金 500–1000萬（倉位）")
    account_ntd = st.slider("資金規模（NTD）", 5_000_000, 10_000_000, 7_000_000, step=100_000)
    risk_pct = st.slider("每筆最大風險（% of 資金）", 0.2, 2.0, 0.8, 0.1) / 100.0
    max_position_pct = st.slider("單筆最大持倉（% of 資金）", 5.0, 40.0, 20.0, 1.0) / 100.0


# -----------------------------
# Diagnostics renderer
# -----------------------------
def render_diagnostics(attempts: List[FetchAttempt]):
    if not attempts:
        return
    df = pd.DataFrame([{
        "source": a.source,
        "result": a.status,
        "http": a.http,
        "bytes": a.bytes,
        "url": a.url,
        "note": a.note,
        "preview": a.preview
    } for a in attempts])
    st.markdown("### 🧩 逐路診斷（哪一路失敗、為什麼）")
    st.dataframe(df, use_container_width=True, hide_index=True)


# -----------------------------
# Main: Single analysis
# -----------------------------
def analyze_one(code: str):
    code = str(code).strip().upper().replace(" ", "")
    if not code:
        st.warning("請輸入股票代號（例如：2330 / 2317 / 6274）")
        return

    df, src, attempts = fetch_ohlcv_multi(code, months_back=months_back)

    if debug:
        render_diagnostics(attempts)

    if df is None or df.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。")
        st.info("建議：勾選 Debug 觀看 preview（通常是被回傳 HTML / WAF 擋），或先按『網路測試』。")
        return

    df = compute_signals(df)
    last = df.iloc[-1]
    score, flags = confluence_score(last)

    plan = build_breakout_plan(
        df=df,
        max_chase_pct=max_chase_pct,
        trigger_buffer_atr=trigger_buffer_atr,
        stop_atr_mult=stop_atr_mult,
        target_rr=target_rr,
    )
    action, reason = breakout_action(df, flags, plan, max_chase_pct=max_chase_pct)

    # headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{float(last['Close']):,.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", src)
    c4.metric("最後日期/筆數", f"{df.index[-1].date()} / {len(df)}")

    st.markdown("## 📌 當下是否可操作？（趨勢突破型）")
    if action == "BUY":
        st.success(f"🟢 **BUY / 可操作**：{reason}")
    elif action == "SELL":
        st.warning(f"🟠 **SELL / 可操作**：{reason}")
    else:
        st.info(f"🕒 **WATCH / 觀望**：{reason}")

    close = float(last["Close"])
    atr_v = float(plan["risk"]["atr"])
    trigger = float(plan["trigger"]["price"])
    pb = plan["pullback"]
    risk = plan["risk"]

    st.markdown("## 🧭 突破交易計畫（專業描述）")

    dist = professional_distance(close, trigger, atr_v, max_chase_pct)

    st.markdown(
        f"""
**① Breakout Trigger（突破觸發價）**：**{trigger:,.2f}**  
• 現價相對觸發價：**{dist['pct']}** / **{dist['atr']}** / Entry Efficiency：**{dist['eff']}**  
• 追價控管：若追價距離 > **{max_chase_pct*100:.0f}%** → 建議等待回測（Pullback）再進  

**② Pullback Buy（回測買點區）**：**{pb['low']:,.2f} – {pb['high']:,.2f}**（中心：{pb['center']:,.2f}）  
• 定義：突破後回測到均線/中軌附近止跌，屬「二次進場」  

**③ Invalidation（失效止損）**：**{risk['stop']:,.2f}**  
• 定義：跌破此價位代表突破失效（避免套牢）  

**④ Targets（目標價）**：**{risk['target']:,.2f}**（RR≈{target_rr:.1f}）  
"""
    )

    # Position sizing
    st.markdown("## 🧮 倉位建議（500–1000萬 / 風險控管）")
    size = position_sizing(
        account_ntd=account_ntd,
        entry=risk["entry_ref"],
        stop=risk["stop"],
        max_position_pct=max_position_pct,
        risk_pct=risk_pct,
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("每筆可承擔風險（NTD）", f"{size['risk_amt']:,.0f}")
    d2.metric("每股風險（Entry-Stop）", f"{size['per_share_risk']:,.2f}")
    d3.metric("建議股數（shares）", f"{size['shares']:,.0f}")
    d4.metric("部位金額 / 佔資金", f"{size['pos_value']:,.0f} / {size['pos_pct']*100:.1f}%")

    # Indicators quick view
    st.markdown("## 📈 指標狀態（快速判讀）")
    ind = pd.DataFrame([{
        "Close": float(last["Close"]),
        "EMA20": float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan,
        "SMA60": float(last["SMA60"]) if pd.notna(last["SMA60"]) else np.nan,
        "RSI14": float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan,
        "K": float(last["K"]) if pd.notna(last["K"]) else np.nan,
        "D": float(last["D"]) if pd.notna(last["D"]) else np.nan,
        "MACD": float(last["MACD"]) if pd.notna(last["MACD"]) else np.nan,
        "MACD_SIG": float(last["MACD_SIG"]) if pd.notna(last["MACD_SIG"]) else np.nan,
        "ATR14": float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan,
        "BB_L": float(last["BB_L"]) if pd.notna(last["BB_L"]) else np.nan,
        "BB_M": float(last["BB_M"]) if pd.notna(last["BB_M"]) else np.nan,
        "BB_U": float(last["BB_U"]) if pd.notna(last["BB_U"]) else np.nan,
        "VOL>VMA20": bool(flags.get("vol_up", False)),
        "Break BB Upper": bool(flags.get("break_bb_upper", False)),
    }])
    st.dataframe(ind, use_container_width=True, hide_index=True)

    st.markdown("## 🧷 布林通道圖（無 Plotly / Matplotlib）")
    plot_bollinger(df, code)


# -----------------------------
# Top10 scanner (Breakout style)
# -----------------------------
def scan_top10(stock_pool: List[str]):
    seen = set()
    pool = []
    for s in stock_pool:
        s = str(s).strip().upper().replace(" ", "")
        if s and s not in seen:
            pool.append(s)
            seen.add(s)

    if not pool:
        st.warning("股票池為空。")
        return

    hard_limit = 35
    pool = pool[:hard_limit]

    rows = []
    diag_map: Dict[str, List[FetchAttempt]] = {}

    prog = st.progress(0)
    for i, code in enumerate(pool, start=1):
        df, src, attempts = fetch_ohlcv_multi(code, months_back=months_back)
        diag_map[code] = attempts

        if df is None or df.empty:
            prog.progress(i / len(pool))
            continue

        df = compute_signals(df)
        last = df.iloc[-1]
        score, flags = confluence_score(last)

        plan = build_breakout_plan(
            df=df,
            max_chase_pct=max_chase_pct,
            trigger_buffer_atr=trigger_buffer_atr,
            stop_atr_mult=stop_atr_mult,
            target_rr=target_rr,
        )
        action, _ = breakout_action(df, flags, plan, max_chase_pct=max_chase_pct)

        close = float(last["Close"])
        trigger = float(plan["trigger"]["price"])
        chase_pct = (close - trigger) / trigger * 100 if trigger > 0 else np.nan

        rows.append({
            "股票": code,
            "來源": src,
            "AI分數": score,
            "當下判斷": action,
            "現價": close,
            "突破觸發價": trigger,
            "追價距離(%)": chase_pct,
        })

        prog.progress(i / len(pool))

    if not rows:
        st.warning("掃描結果為空（資料源可能暫時抓不到）。")
        return

    out = pd.DataFrame(rows).drop_duplicates(subset=["股票"], keep="first")

    # 排序：BUY > WATCH > SELL；分數高優先；追價距離越小越好（負數=尚未突破）
    rank = {"BUY": 0, "WATCH": 1, "SELL": 2}
    out["rank"] = out["當下判斷"].map(rank).fillna(9).astype(int)
    out["abs_chase"] = out["追價距離(%)"].abs()

    out = out.sort_values(["rank", "AI分數", "abs_chase"], ascending=[True, False, True]).head(10)
    out = out.drop(columns=["rank", "abs_chase"])

    st.markdown("## 🔥 AI 強勢股 Top 10（趨勢突破型排序）")
    st.dataframe(out, use_container_width=True, hide_index=True)

    if debug:
        st.markdown("### 🧩 Top10 逐檔診斷（Debug）")
        pick = st.selectbox("選擇要看診斷的股票", out["股票"].tolist())
        render_diagnostics(diag_map.get(pick, []))


# -----------------------------
# UI routing
# -----------------------------
if mode == "單一股票分析":
    colL, colR = st.columns([1.2, 3.6])
    with colL:
        code = st.text_input("請輸入股票代號", value="6274")
        if st.button("網路測試（建議先按一次）"):
            url = "https://www.twse.com.tw/"
            r, err, preview = _requests_get(url, timeout=8)
            if r is None:
                st.error(f"TWSE ping 失敗：{err}")
            else:
                st.success(f"TWSE ping：HTTP {r.status_code} / bytes {len(r.text or '')}")
                st.caption(f"preview：{preview}")
            url2 = "https://www.tpex.org.tw/"
            r2, err2, preview2 = _requests_get(url2, timeout=8)
            if r2 is None:
                st.error(f"TPEX ping 失敗：{err2}")
            else:
                st.success(f"TPEX ping：HTTP {r2.status_code} / bytes {len(r2.text or '')}")
                st.caption(f"preview：{preview2}")
    with colR:
        analyze_one(code)

else:
    st.markdown("## Top 10 掃描器（趨勢突破型）")
    st.caption("雲端避免一次掃全市場；建議先用 20–35 檔測試。")

    default_pool = "2330,2317,2454,3037,8046,2382,2303,4967,2603,2609,2882,2881,2891,0050,006208,6274"
    pool_text = st.text_area("輸入股票池（逗號分隔）", value=default_pool, height=110)
    stock_pool = [x.strip() for x in pool_text.split(",") if x.strip()]
    scan_top10(stock_pool)

st.caption("⚠️ 本工具僅供資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。")
