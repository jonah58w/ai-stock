# app.py
# AI 台股量化專業平台（最終極全備援 + 逐路診斷 / 不自動下單 / 無 Plotly）
# ✅ 單股分析 + Top10 掃描器
# ✅ 多來源備援：yfinance → TWSE(JSON) → TPEX(JSON)
# ✅ 逐路診斷（哪一路失敗、為什麼）
# ✅ 指標：BB(20,2) / EMA20 / MACD / KD / 乖離率 / 成交量
# ✅ 布林通道分析圖（K線 + BB + EMA20 + Volume）—純 matplotlib，無外掛
# ✅ 修正 date 欄位 KeyError：強制把日期變成 DatetimeIndex
# ⚠️ 僅做資訊顯示，不含自動下單

from __future__ import annotations

import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# 基本設定
# -----------------------------
APP_TITLE = "🧠 AI 台股量化專業平台（最終極全備援 + 逐路診斷 / 不自動下單 / 無 Plotly）"

DEFAULT_POOL = [
    "2330", "2317", "2454", "2382", "3037", "2303", "2408", "2882", "1301", "1101",
    "2603", "2618", "0050", "00878", "006208", "4967", "8046", "6274"
]

PERIOD_TO_DAYS = {
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}

REQ_TIMEOUT = 12
REQ_RETRY = 2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

# -----------------------------
# 資料結構：診斷紀錄
# -----------------------------
@dataclass
class FetchLog:
    source: str
    ok: bool
    result: str
    url: str
    status: Optional[int] = None
    snippet: str = ""


# -----------------------------
# 工具：日期與代號
# -----------------------------
def _today_taipei() -> dt.date:
    # Streamlit Cloud 可能不是台北時區；這裡用 UTC+8 的日期概念處理
    now_utc = dt.datetime.utcnow()
    now_tpe = now_utc + dt.timedelta(hours=8)
    return now_tpe.date()


def normalize_stock_id(s: str) -> str:
    s = (s or "").strip().upper()
    # 允許輸入 2330.TW / 2330.TWO / 0050.TW 等
    s = s.replace(".TW", "").replace(".TWO", "")
    return s


def is_etf(stock_id: str) -> bool:
    # 粗略：ETF 常見為 00xxx / 00xxxx
    return stock_id.startswith("00")


def guess_market_suffix(stock_id: str) -> List[str]:
    # yfinance：上市 .TW，上櫃 .TWO
    # 先猜 .TW，再試 .TWO
    return [".TW", ".TWO"]


# -----------------------------
# 核心：HTTP 取資料（含 retry）
# -----------------------------
def http_get(url: str, params: Optional[dict] = None) -> Tuple[Optional[requests.Response], Optional[str]]:
    last_err = None
    for i in range(REQ_RETRY + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=REQ_TIMEOUT)
            return r, None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.6 * (i + 1))
    return None, last_err


# -----------------------------
# 來源 1：yfinance（若被擋會失敗）
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_yfinance(stock_id: str, days: int) -> Tuple[Optional[pd.DataFrame], List[FetchLog], str]:
    logs: List[FetchLog] = []
    stock_id = normalize_stock_id(stock_id)

    try:
        import yfinance as yf  # 延遲 import：避免環境缺失時整個 app 掛掉
    except Exception as e:
        logs.append(FetchLog("YF", False, f"NO_MODULE: {e}", url="(import yfinance)"))
        return None, logs, "YF"

    for suf in guess_market_suffix(stock_id):
        ticker = f"{stock_id}{suf}"
        try:
            df = yf.download(
                ticker,
                period=f"{max(7, days)}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            # yfinance 有時回 MultiIndex 欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            if df is None or df.empty:
                logs.append(FetchLog("YF", False, "EMPTY", url=f"yfinance:{ticker}"))
                continue

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
            })

            # 強制 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.dropna(subset=["close"])
            df = df.sort_index()
            df = df.tail(days + 5)

            if len(df) < 60:
                logs.append(FetchLog("YF", False, f"TOO_SHORT({len(df)})", url=f"yfinance:{ticker}"))
                continue

            logs.append(FetchLog("YF", True, "OK", url=f"yfinance:{ticker}"))
            return df[["open", "high", "low", "close", "volume"]].copy(), logs, f"YF{suf}"

        except Exception as e:
            logs.append(FetchLog("YF", False, f"EXCEPTION: {e}", url=f"yfinance:{ticker}"))
            continue

    return None, logs, "YF"


# -----------------------------
# 來源 2：TWSE（上市）— JSON
# STOCK_DAY?response=json&date=YYYYMMDD&stockNo=2330
# -----------------------------
def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_twse_json(stock_id: str, days: int) -> Tuple[Optional[pd.DataFrame], List[FetchLog], str]:
    logs: List[FetchLog] = []
    stock_id = normalize_stock_id(stock_id)

    end = _today_taipei()
    start = end - dt.timedelta(days=days + 15)

    # 逐月抓（TWSE 是月資料）
    months = []
    cursor = dt.date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        # next month
        y = cursor.year + (cursor.month // 12)
        m = 1 if cursor.month == 12 else cursor.month + 1
        cursor = dt.date(y, m, 1)

    all_rows = []

    for m0 in months[-10:]:  # 6mo/1y 通常夠用，避免太多請求
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        params = {"response": "json", "date": _yyyymmdd(m0), "stockNo": stock_id}
        r, err = http_get(url, params=params)
        if r is None:
            logs.append(FetchLog("TWSE_JSON", False, f"NET_ERR: {err}", url=f"{url}?{params}"))
            continue

        status = r.status_code
        text = r.text or ""
        snip = text[:120].replace("\n", " ").replace("\r", " ")

        if status != 200:
            logs.append(FetchLog("TWSE_JSON", False, f"HTTP_{status}", url=r.url, status=status, snippet=snip))
            continue

        # 可能回傳空字串、或非 JSON
        try:
            data = r.json()
        except Exception:
            logs.append(FetchLog("TWSE_JSON", False, "NOT_JSON", url=r.url, status=status, snippet=snip))
            continue

        if not isinstance(data, dict) or ("data" not in data):
            logs.append(FetchLog("TWSE_JSON", False, "BAD_SCHEMA", url=r.url, status=status, snippet=snip))
            continue

        rows = data.get("data", [])
        if not rows:
            logs.append(FetchLog("TWSE_JSON", False, "EMPTY_DATA", url=r.url, status=status, snippet=snip))
            continue

        # TWSE 日期是民國：112/01/03
        for row in rows:
            try:
                roc_date = row[0].strip()
                yy, mm, dd = roc_date.split("/")
                y = int(yy) + 1911
                date = dt.date(y, int(mm), int(dd))
                open_ = float(str(row[3]).replace(",", ""))
                high_ = float(str(row[4]).replace(",", ""))
                low_ = float(str(row[5]).replace(",", ""))
                close_ = float(str(row[6]).replace(",", ""))
                vol = float(str(row[1]).replace(",", ""))  # 成交股數
                all_rows.append((date, open_, high_, low_, close_, vol))
            except Exception:
                continue

        logs.append(FetchLog("TWSE_JSON", True, f"OK({len(rows)})", url=r.url, status=status))

    if not all_rows:
        return None, logs, "TWSE_JSON"

    df = pd.DataFrame(all_rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    df = df.tail(days + 5)

    if len(df) < 60:
        logs.append(FetchLog("TWSE_JSON", False, f"TOO_SHORT({len(df)})", url="(merged)"))
        return None, logs, "TWSE_JSON"

    return df, logs, "TWSE_JSON"


# -----------------------------
# 來源 3：TPEX（上櫃）— JSON
# st43_result.php?l=zh-tw&d=YYY/MM/DD&stkno=xxxx
# -----------------------------
def _roc_yyy_mm_dd(d: dt.date) -> str:
    y = d.year - 1911
    return f"{y}/{d.month:02d}/{d.day:02d}"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_tpex_json(stock_id: str, days: int) -> Tuple[Optional[pd.DataFrame], List[FetchLog], str]:
    logs: List[FetchLog] = []
    stock_id = normalize_stock_id(stock_id)

    end = _today_taipei()
    start = end - dt.timedelta(days=days + 15)

    # 逐月抓（TPEX 也常用月查）
    months = []
    cursor = dt.date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        y = cursor.year + (cursor.month // 12)
        m = 1 if cursor.month == 12 else cursor.month + 1
        cursor = dt.date(y, m, 1)

    all_rows = []

    for m0 in months[-10:]:
        url = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php"
        params = {"l": "zh-tw", "d": _roc_yyy_mm_dd(m0), "stkno": stock_id}
        r, err = http_get(url, params=params)
        if r is None:
            logs.append(FetchLog("TPEX_JSON", False, f"NET_ERR: {err}", url=f"{url}?{params}"))
            continue

        status = r.status_code
        text = r.text or ""
        snip = text[:120].replace("\n", " ").replace("\r", " ")
        if status != 200:
            logs.append(FetchLog("TPEX_JSON", False, f"HTTP_{status}", url=r.url, status=status, snippet=snip))
            continue

        try:
            data = r.json()
        except Exception:
            logs.append(FetchLog("TPEX_JSON", False, "NOT_JSON", url=r.url, status=status, snippet=snip))
            continue

        rows = data.get("aaData") or data.get("data") or []
        if not rows:
            logs.append(FetchLog("TPEX_JSON", False, "EMPTY_DATA", url=r.url, status=status, snippet=snip))
            continue

        # TPEX 欄位常見：日期、成交股數、成交金額、開盤、最高、最低、收盤...
        for row in rows:
            try:
                roc_date = str(row[0]).strip()
                yy, mm, dd = roc_date.split("/")
                y = int(yy) + 1911
                date = dt.date(y, int(mm), int(dd))
                open_ = float(str(row[3]).replace(",", ""))
                high_ = float(str(row[4]).replace(",", ""))
                low_ = float(str(row[5]).replace(",", ""))
                close_ = float(str(row[6]).replace(",", ""))
                vol = float(str(row[1]).replace(",", ""))
                all_rows.append((date, open_, high_, low_, close_, vol))
            except Exception:
                continue

        logs.append(FetchLog("TPEX_JSON", True, f"OK({len(rows)})", url=r.url, status=status))

    if not all_rows:
        return None, logs, "TPEX_JSON"

    df = pd.DataFrame(all_rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    df = df.tail(days + 5)

    if len(df) < 60:
        logs.append(FetchLog("TPEX_JSON", False, f"TOO_SHORT({len(df)})", url="(merged)"))
        return None, logs, "TPEX_JSON"

    return df, logs, "TPEX_JSON"


# -----------------------------
# 取資料：多路備援 + 診斷彙總
# -----------------------------
def fetch_data_all(stock_id: str, days: int) -> Tuple[Optional[pd.DataFrame], str, List[FetchLog]]:
    logs_all: List[FetchLog] = []

    # 1) yfinance
    df, logs, src = fetch_yfinance(stock_id, days)
    logs_all += logs
    if df is not None and not df.empty:
        return standardize_ohlcv(df), src, logs_all

    # 2) TWSE JSON
    df, logs, src = fetch_twse_json(stock_id, days)
    logs_all += logs
    if df is not None and not df.empty:
        return standardize_ohlcv(df), src, logs_all

    # 3) TPEX JSON
    df, logs, src = fetch_tpex_json(stock_id, days)
    logs_all += logs
    if df is not None and not df.empty:
        return standardize_ohlcv(df), src, logs_all

    return None, "NONE", logs_all


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # 允許 date 欄或 index 是日期；最後統一成 DatetimeIndex
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).set_index("date")

    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")

    d = d.dropna(subset=["close"]).sort_index()
    # 補齊必要欄位
    for c in ["open", "high", "low"]:
        if c not in d.columns:
            d[c] = d["close"]
    if "volume" not in d.columns:
        d["volume"] = 0.0

    d = d[["open", "high", "low", "close", "volume"]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    return d


# -----------------------------
# 指標計算
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(0)

    k_line = rsv.ewm(alpha=1 / k, adjust=False).mean()
    d_line = k_line.ewm(alpha=1 / d, adjust=False).mean()
    return k_line, d_line


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["sma20"] = sma(d["close"], 20)
    d["bb_u"], d["bb_m"], d["bb_l"] = bollinger(d["close"], 20, 2.0)

    d["macd"], d["macd_sig"], d["macd_hist"] = macd(d["close"])
    d["k"], d["d"] = kd(d["high"], d["low"], d["close"])

    d["bias20"] = (d["close"] / d["sma20"]) - 1.0
    d["vol_ma20"] = sma(d["volume"], 20)
    d["vol_ratio"] = d["volume"] / d["vol_ma20"]
    d["vol_ratio"] = d["vol_ratio"].replace([np.inf, -np.inf], np.nan)

    return d


# -----------------------------
# 判斷：當下操作 + 近端/深回檔買點 + 近端賣點
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resonance_score(d: pd.DataFrame) -> int:
    """0-100：多指標共振（偏實用、偏保守）"""
    if len(d) < 60:
        return 0

    r = d.iloc[-1]
    score = 0.0

    # 1) 布林位置（越接近下軌偏買；越接近上軌偏賣，但也算「有訊號」）
    if pd.notna(r["bb_u"]) and pd.notna(r["bb_l"]) and (r["bb_u"] > r["bb_l"]):
        pos = (r["close"] - r["bb_l"]) / (r["bb_u"] - r["bb_l"])  # 0~1
        # 靠近兩端都給分，中間較少
        edge = 1 - abs(pos - 0.5) * 2  # 中間=0，兩端=1
        score += (edge * 30)

    # 2) EMA20 趨勢（站上加分）
    if pd.notna(r["ema20"]):
        score += (15 if r["close"] >= r["ema20"] else 8)

    # 3) MACD 動能
    if pd.notna(r["macd_hist"]):
        # hist 轉強給分
        score += (15 if r["macd_hist"] > 0 else 7)

    # 4) KD 超買超賣端點加分（有訊號）
    if pd.notna(r["k"]):
        if r["k"] <= 20:
            score += 20
        elif r["k"] >= 80:
            score += 18
        else:
            score += 10

    # 5) 量能（突破或放量）
    if pd.notna(r["vol_ratio"]):
        if r["vol_ratio"] >= 1.8:
            score += 20
        elif r["vol_ratio"] >= 1.2:
            score += 12
        else:
            score += 6

    return int(clamp(score, 0, 100))


def estimate_zones(d: pd.DataFrame, max_buy_dist: float) -> Dict[str, object]:
    """
    輸出：
    - action: BUY / SELL / WATCH
    - reason
    - near_buy / deep_buy / near_sell：區間（low, high）與距離%
    """
    r = d.iloc[-1]
    price = float(r["close"])

    # 支撐/壓力：用 60 日區間
    win = min(60, len(d))
    support = float(d["low"].tail(win).min())
    resist = float(d["high"].tail(win).max())

    bb_l = float(r["bb_l"]) if pd.notna(r["bb_l"]) else support
    bb_u = float(r["bb_u"]) if pd.notna(r["bb_u"]) else resist
    ema20_v = float(r["ema20"]) if pd.notna(r["ema20"]) else price

    # 近端買點：下軌與支撐取較「近」者（避免太遠）
    near_buy_center = max(bb_l, support)
    deep_buy_center = min(bb_l, support)

    # 近端賣點：上軌與壓力取較「近」者
    near_sell_center = min(bb_u, resist)

    def band(center: float, width: float = 0.010) -> Tuple[float, float]:
        return (center * (1 - width), center * (1 + width))

    near_buy = band(near_buy_center, 0.010)
    deep_buy = band(deep_buy_center, 0.012)
    near_sell = band(near_sell_center, 0.011)

    def dist_pct(center: float) -> float:
        return (center - price) / price

    near_buy_dist = dist_pct((near_buy[0] + near_buy[1]) / 2)
    deep_buy_dist = dist_pct((deep_buy[0] + deep_buy[1]) / 2)
    near_sell_dist = dist_pct((near_sell[0] + near_sell[1]) / 2)

    # 「可操作買點」：距離不能太遠（你要求的現實限制）
    # 若太遠，近端買點就改成 None（只保留深回檔作參考）
    near_buy_actionable = abs(near_buy_dist) <= max_buy_dist

    # 當下判斷（偏保守，可操作）
    action = "WATCH"
    reason = "條件尚未明確，等待價格進入區間或指標翻轉。"

    k_now = float(r["k"]) if pd.notna(r["k"]) else 50.0
    hist_now = float(r["macd_hist"]) if pd.notna(r["macd_hist"]) else 0.0
    hist_prev = float(d["macd_hist"].iloc[-2]) if len(d) >= 2 and pd.notna(d["macd_hist"].iloc[-2]) else hist_now

    in_near_buy_zone = (near_buy[0] <= price <= near_buy[1]) if near_buy_actionable else False
    in_near_sell_zone = (near_sell[0] <= price <= near_sell[1])

    macd_turn_up = (hist_now > hist_prev) and (hist_now > -0.5 * abs(hist_prev))
    macd_turn_down = (hist_now < hist_prev)

    # BUY：靠近下軌/支撐 + KD 低檔 + MACD 動能轉強
    if in_near_buy_zone and (k_now <= 30) and macd_turn_up:
        action = "BUY"
        reason = "價格進入『近端買點區』且 KD 低檔、MACD 動能轉強，屬可操作回檔買點。"

    # SELL：靠近壓力/上軌 + KD 高檔 + MACD 轉弱
    elif in_near_sell_zone and (k_now >= 70) and macd_turn_down:
        action = "SELL"
        reason = "價格逼近『近端賣點區』且 KD 高檔、MACD 轉弱，適合減碼/停利/提高警覺。"

    # 特別提示：如果近端買點距離太遠，就明確標示
    if not near_buy_actionable:
        reason += f"（近端買點距離現價約 {abs(near_buy_dist)*100:.1f}%，超過可操作上限，僅保留深回檔區參考）"

    return {
        "price": price,
        "ema20": ema20_v,
        "support": support,
        "resist": resist,
        "action": action,
        "reason": reason,
        "near_buy": near_buy if near_buy_actionable else None,
        "near_buy_dist": near_buy_dist,
        "deep_buy": deep_buy,
        "deep_buy_dist": deep_buy_dist,
        "near_sell": near_sell,
        "near_sell_dist": near_sell_dist,
    }


# -----------------------------
# 圖表：K線 + 布林 + EMA20 + 量（無 plotly）
# -----------------------------
def plot_candles_with_bb(d: pd.DataFrame, title: str = "") -> plt.Figure:
    # 取最後 N 根，避免太密
    N = min(140, len(d))
    dd = d.tail(N).copy()
    dd = dd.dropna(subset=["open", "high", "low", "close"]).copy()

    x = np.arange(len(dd))

    fig = plt.figure(figsize=(12.5, 6.8), dpi=120)
    gs = fig.add_gridspec(5, 1, hspace=0.05)
    ax = fig.add_subplot(gs[:4, 0])
    axv = fig.add_subplot(gs[4, 0], sharex=ax)

    # K線
    for i, (_, r) in enumerate(dd.iterrows()):
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        up = (c >= o)

        # wick
        ax.vlines(i, l, h, linewidth=1)

        # body
        body_low = min(o, c)
        body_h = max(0.5e-6, abs(c - o))
        rect = Rectangle((i - 0.35, body_low), 0.7, body_h, fill=True, alpha=0.85)
        # 不指定顏色：用預設色系，但區分上/下用 alpha+edge
        if up:
            rect.set_edgecolor("black")
        else:
            rect.set_edgecolor("black")
        ax.add_patch(rect)

    # BB + EMA20（線圖）
    if "bb_u" in dd.columns:
        ax.plot(x, dd["bb_u"].values, linewidth=1.2, label="BB Upper")
    if "bb_m" in dd.columns:
        ax.plot(x, dd["bb_m"].values, linewidth=1.0, label="BB Mid")
    if "bb_l" in dd.columns:
        ax.plot(x, dd["bb_l"].values, linewidth=1.2, label="BB Lower")
    if "ema20" in dd.columns:
        ax.plot(x, dd["ema20"].values, linewidth=1.2, label="EMA20")

    # Volume
    vol = dd["volume"].fillna(0).values
    axv.bar(x, vol, width=0.7)

    # X 軸日期標籤（稀疏顯示）
    dates = dd.index.to_pydatetime()
    step = max(1, len(dates) // 6)
    xticks = list(range(0, len(dates), step))
    axv.set_xticks(xticks)
    axv.set_xticklabels([dates[i].strftime("%Y-%m-%d") for i in xticks], rotation=0)

    ax.set_title(title, fontsize=12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    axv.grid(True, linewidth=0.3, alpha=0.5)

    ax.legend(loc="upper left", fontsize=9, frameon=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    return fig


# -----------------------------
# UI：診斷顯示
# -----------------------------
def logs_to_df(logs: List[FetchLog]) -> pd.DataFrame:
    rows = []
    for lg in logs:
        rows.append({
            "source": lg.source,
            "result": ("OK" if lg.ok else lg.result),
            "url": lg.url,
            "status": lg.status,
            "snippet": lg.snippet,
        })
    return pd.DataFrame(rows)


def render_diagnostics(logs: List[FetchLog]):
    st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
    if not logs:
        st.info("尚無診斷紀錄。")
        return
    st.dataframe(logs_to_df(logs), use_container_width=True, hide_index=True)


def network_test_panel():
    st.markdown("### 🧪 網路測試（建議先按一次）")
    if st.button("✅ 立即測試常用來源連線", use_container_width=True):
        tests = [
            ("TWSE_HOME", "https://www.twse.com.tw/"),
            ("TWSE_STOCK_DAY_JSON", "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20250101&stockNo=2330"),
            ("TPEX_HOME", "https://www.tpex.org.tw/"),
            ("TPEX_ST43", "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d=114/01/02&stkno=6488"),
        ]
        out = []
        for name, url in tests:
            r, err = http_get(url, params=None)
            if r is None:
                out.append({"name": name, "ok": False, "status": None, "url": url, "snippet": (err or "")[:120]})
            else:
                out.append({"name": name, "ok": (r.status_code == 200), "status": r.status_code, "url": r.url,
                            "snippet": (r.text or "")[:120].replace("\n", " ")})
        st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)


# -----------------------------
# 單股渲染
# -----------------------------
def render_single(stock_id: str, period_key: str, show_debug: bool, max_buy_dist: float):
    stock_id = normalize_stock_id(stock_id)
    days = PERIOD_TO_DAYS.get(period_key, 180)

    df, src, logs = fetch_data_all(stock_id, days)

    if df is None or df.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。")
        if show_debug:
            render_diagnostics(logs)
        return

    d = compute_indicators(df)
    last_date = d.index[-1].date()
    price = float(d["close"].iloc[-1])

    score = resonance_score(d)
    zones = estimate_zones(d, max_buy_dist=max_buy_dist)

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目前價格", f"{price:.2f}")
    col2.metric("AI 共振分數", f"{score}/100")
    col3.metric("資料來源", f"{src}")
    col4.metric("最後日期 / 筆數", f"{last_date} / {len(d)}")

    st.markdown("### 📌 當下是否為買點/賣點？（可操作判斷）")
    if zones["action"] == "BUY":
        st.success("✅ 當下位於『近端買點區』（可操作）")
    elif zones["action"] == "SELL":
        st.warning("⚠️ 當下位於『近端賣點區』")
    else:
        st.info("⏳ 觀望：條件尚未明確")

    st.caption(f"原因：{zones['reason']}")

    st.markdown("### 🗺️ 未來預估買賣點（區間 + 距離%）")

    if zones["near_buy"] is not None:
        nb0, nb1 = zones["near_buy"]
        st.success(f"✅ 可操作買點（近端回檔）：{nb0:.2f} ~ {nb1:.2f}（距離現價：約 {abs(zones['near_buy_dist'])*100:.1f}%）")
    else:
        st.warning("⚠️ 近端買點：距離過遠或不可操作（已自動隱藏）")

    db0, db1 = zones["deep_buy"]
    st.info(f"🕳️ 深回檔買點（等待型）：{db0:.2f} ~ {db1:.2f}（距離現價：約 {abs(zones['deep_buy_dist'])*100:.1f}%）")

    ns0, ns1 = zones["near_sell"]
    st.warning(f"🎯 近端賣點區（壓力/獲利）：{ns0:.2f} ~ {ns1:.2f}（距離現價：約 {abs(zones['near_sell_dist'])*100:.1f}%）")

    st.markdown("### 📉 布林通道分析圖（K線 + BB + EMA20 + Volume）")
    fig = plot_candles_with_bb(d, title=f"{stock_id}｜{src}")
    st.pyplot(fig, clear_figure=True)

    with st.expander("📄 最近 20 筆資料（含指標）", expanded=False):
        show_cols = ["open", "high", "low", "close", "volume", "ema20", "bb_u", "bb_m", "bb_l", "k", "d", "macd_hist", "bias20", "vol_ratio"]
        st.dataframe(d[show_cols].tail(20), use_container_width=True)

    if show_debug:
        render_diagnostics(logs)


# -----------------------------
# Top10 掃描
# -----------------------------
def scan_top10(stock_pool: List[str], period_key: str, max_buy_dist: float, show_debug: bool) -> Tuple[pd.DataFrame, Dict[str, List[FetchLog]]]:
    days = PERIOD_TO_DAYS.get(period_key, 180)
    rows = []
    debug_map: Dict[str, List[FetchLog]] = {}

    for sid in stock_pool:
        sid = normalize_stock_id(sid)
        df, src, logs = fetch_data_all(sid, days)
        debug_map[sid] = logs

        if df is None or df.empty:
            continue

        d = compute_indicators(df)
        if len(d) < 60:
            continue

        price = float(d["close"].iloc[-1])
        score = resonance_score(d)
        zones = estimate_zones(d, max_buy_dist=max_buy_dist)

        def pct_to_str(p: float) -> str:
            return f"{p*100:.1f}%"

        near_buy_pct = ""
        if zones["near_buy"] is not None:
            near_buy_pct = pct_to_str(abs(zones["near_buy_dist"]))
        deep_buy_pct = pct_to_str(abs(zones["deep_buy_dist"]))
        near_sell_pct = pct_to_str(abs(zones["near_sell_dist"]))

        rows.append({
            "股票": sid,
            "來源": src,
            "目前價": round(price, 2),
            "操作判斷": zones["action"],
            "AI分數": score,
            "近端買點距離": near_buy_pct if near_buy_pct else "—",
            "深回檔距離": deep_buy_pct,
            "近端賣點距離": near_sell_pct,
            "最後日期": str(d.index[-1].date()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, debug_map

    # 先依 AI 分數，再把 SELL/BUY/觀望做次排序
    action_rank = {"BUY": 0, "SELL": 1, "WATCH": 2}
    out["_ar"] = out["操作判斷"].map(lambda x: action_rank.get(x, 9))
    out = out.sort_values(["AI分數", "_ar"], ascending=[False, True]).drop(columns=["_ar"]).head(10)
    return out, debug_map


def render_top10(period_key: str, max_buy_dist: float, show_debug: bool):
    st.markdown("Top10 掃描器：**先用小池測試**（避免全市場在 Cloud 超時）。")

    pool_text = st.text_area(
        "stock_pool（逗號或換行分隔）",
        value="\n".join(DEFAULT_POOL),
        height=140,
    )
    pool = []
    for part in pool_text.replace(",", "\n").split("\n"):
        p = normalize_stock_id(part)
        if p:
            pool.append(p)

    if st.button("🔥 RUN Top10 掃描", use_container_width=True):
        with st.spinner("掃描中…（逐檔取資料＋算指標）"):
            result, debug_map = scan_top10(pool, period_key, max_buy_dist, show_debug)

        if result.empty:
            st.warning("目前掃描結果為空（可能資料下載失敗/來源限制）。稍後再試或換 stock_pool。")
            if show_debug:
                # 顯示前幾檔診斷
                for k in pool[:5]:
                    st.write(f"— {k}")
                    render_diagnostics(debug_map.get(k, []))
            return

        st.subheader("🔥 AI 強勢股 Top 10（含當下操作判斷 + 距離%）")
        st.dataframe(result, use_container_width=True, hide_index=True)

        st.markdown("### 🔎 點選一檔直接展開完整分析")
        pick = st.selectbox("選擇股票", options=result["股票"].tolist())
        st.divider()
        render_single(pick, period_key=period_key, show_debug=show_debug, max_buy_dist=max_buy_dist)


# -----------------------------
# Streamlit 主程式
# -----------------------------
def main():
    st.set_page_config(page_title="AI 台股量化平台", layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.markdown("## 選擇模式")
        mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

        st.markdown("## 資料期間")
        period_key = st.selectbox("資料期間", options=list(PERIOD_TO_DAYS.keys()), index=1)

        show_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

        st.markdown("## 可操作買點最大距離（避免買點離現實太遠）")
        max_buy_dist = st.slider("max_buy_distance", min_value=0.03, max_value=0.30, value=0.12, step=0.01)

        with st.expander("🧪 網路測試（建議先按一次）", expanded=False):
            network_test_panel()

    if mode == "單一股票分析":
        stock = st.text_input("請輸入股票代號", value="2330")
        render_single(stock, period_key=period_key, show_debug=show_debug, max_buy_dist=max_buy_dist)
    else:
        render_top10(period_key=period_key, max_buy_dist=max_buy_dist, show_debug=show_debug)


if __name__ == "__main__":
    main()
