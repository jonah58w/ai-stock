# app.py
# AI 台股量化專業平台（V22：A+B+C+D / 無 Plotly / 全功能保留）
# ✅ 單一股票分析 + Top10 掃描器
# ✅ 逐路診斷（每條資料源的 HTTP 狀態 / bytes / 片段）
# ✅ 指標：MA/EMA、MACD、KD(Stoch)、RSI、ATR、布林通道、量能
# ✅ 買賣點：當下可操作判斷 + 未來預估買賣點（近端/深回檔/賣點）
# ✅ 專業顯示：溢價/折價%、ATR 標準化距離、區域屬性、Entry Efficiency
# ✅ 風險報酬：止損/目標/即時 RR
# ✅ 倉位建議（500–1000萬）：依 ATR 止損距離 + 風險% + 最大持倉% 自動計算
#
# 依賴（建議 requirements.txt）：
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
# UI / Page
# -----------------------------
st.set_page_config(page_title="AI 台股量化專業平台（V22）", layout="wide")

TITLE = "🧠 AI 台股量化專業平台（V22 / 無 Plotly / 全功能保留）"
SUB = "資料多源備援 + 逐路診斷 + 指標共振 + 布林通道視覺 + 專業買賣點/風險/倉位建議（不自動下單）"

st.markdown(f"# {TITLE}")
st.caption(SUB)

# -----------------------------
# Helpers
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


def _requests_get(url: str, timeout: int = 12) -> Tuple[Optional[requests.Response], Optional[str]]:
    """
    Cloud 常被擋：加 headers + retry + backoff
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }

    last_err = None
    for i in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return r, None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.4 * (i + 1))
    return None, last_err


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


def _roc_date(dt: datetime) -> str:
    # TPEX 有些 API 用 ROC 年：民國年 = 西元年 - 1911
    y = dt.year - 1911
    return f"{y}/{dt.month:02d}/{dt.day:02d}"


def _yyyymmdd(dt: datetime) -> str:
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"


def _is_tpex_like(code: str) -> bool:
    # 粗略：多數上櫃是 .TWO；但我們不靠此判斷，會多源嘗試
    return True


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    保證欄位：Date, Open, High, Low, Close, Volume（datetime index）
    """
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # normalize columns
    col_map = {c.lower(): c for c in out.columns}
    def pick(names: List[str]) -> Optional[str]:
        for n in names:
            if n.lower() in col_map:
                return col_map[n.lower()]
        return None

    o = pick(["open", "Open"])
    h = pick(["high", "High"])
    l = pick(["low", "Low"])
    c = pick(["close", "Close", "adj close", "Adj Close"])
    v = pick(["volume", "Volume", "成交股數", "成交量"])

    if c is None:
        raise ValueError("No Close column")

    out = out.rename(columns={
        o: "Open" if o else o,
        h: "High" if h else h,
        l: "Low" if l else l,
        c: "Close",
        v: "Volume" if v else v,
    })

    # if missing OHLC, approximate from Close
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


# -----------------------------
# Data Sources (multi-backup + diagnostics)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if yf is None:
        return None, FetchAttempt("YF", "", "NO_MODULE", note="yfinance not installed")

    # Try both tails
    tickers = []
    if code.endswith(".TW") or code.endswith(".TWO"):
        tickers = [code]
    else:
        tickers = [f"{code}.TW", f"{code}.TWO"]

    # Yahoo blocks sometimes -> still try quickly
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
    TWSE: rwd/zh/afterTrading/STOCK_DAY?date=YYYYMMDD&stockNo=2330&response=json
    逐月拉，合併
    """
    attempts: List[FetchAttempt] = []
    frames = []

    # use first day of each month going backwards
    today = datetime.now()
    for m in range(months_back):
        dt = (today.replace(day=1) - pd.DateOffset(months=m)).to_pydatetime()
        ymd = _yyyymmdd(dt)
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={ymd}&stockNo={code}"
        r, err = _requests_get(url)
        if r is None:
            attempts.append(FetchAttempt("TWSE_JSON", url, "EXC", note=err or "request failed"))
            continue

        text = r.text or ""
        preview = text[:120].replace("\n", " ")
        if r.status_code != 200:
            attempts.append(FetchAttempt("TWSE_JSON", url, "HTTP_ERR", http=r.status_code, bytes=len(text), preview=preview))
            continue

        # empty / blocked patterns
        if len(text.strip()) < 50:
            attempts.append(FetchAttempt("TWSE_JSON", url, "EMPTY_TEXT", http=200, bytes=len(text), preview=preview))
            continue

        try:
            j = r.json()
            data = j.get("data", [])
            fields = j.get("fields", [])
            if not data or not fields:
                attempts.append(FetchAttempt("TWSE_JSON", url, "NO_DATA", http=200, bytes=len(text), preview=preview))
                continue

            dfm = pd.DataFrame(data, columns=fields)

            # TWSE data uses ROC date: 114/02/27
            # Common fields: 日期, 開盤價, 最高價, 最低價, 收盤價, 成交股數 ...
            col_date = "日期"
            col_open = "開盤價"
            col_high = "最高價"
            col_low = "最低價"
            col_close = "收盤價"
            col_vol = "成交股數" if "成交股數" in dfm.columns else ("成交量" if "成交量" in dfm.columns else None)

            if col_date not in dfm.columns or col_close not in dfm.columns:
                attempts.append(FetchAttempt("TWSE_JSON", url, "SCHEMA_MISS", http=200, bytes=len(text), preview=preview))
                continue

            def parse_roc(s: str) -> datetime:
                # "114/02/27"
                p = str(s).strip().split("/")
                if len(p) != 3:
                    raise ValueError("bad roc date")
                y = int(p[0]) + 1911
                return datetime(y, int(p[1]), int(p[2]))

            out = pd.DataFrame({
                "Date": dfm[col_date].map(parse_roc),
                "Open": dfm[col_open].map(_safe_float) if col_open in dfm.columns else None,
                "High": dfm[col_high].map(_safe_float) if col_high in dfm.columns else None,
                "Low": dfm[col_low].map(_safe_float) if col_low in dfm.columns else None,
                "Close": dfm[col_close].map(_safe_float),
                "Volume": dfm[col_vol].map(_safe_float) if col_vol and col_vol in dfm.columns else None,
            })
            out = out.dropna(subset=["Close"])
            out = _normalize_ohlcv(out)
            frames.append(out)
            attempts.append(FetchAttempt("TWSE_JSON", url, "OK", http=200, bytes=len(text), note=f"rows={len(out)}"))
        except Exception as e:
            attempts.append(FetchAttempt("TWSE_JSON", url, "PARSE_ERR", http=200, bytes=len(text), note=str(e), preview=preview))

    if not frames:
        return None, attempts

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Close"])
    return df, attempts


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    """
    Stooq: sometimes available when Yahoo/TWSE fails.
    Taiwan tickers in stooq are not always consistent; try a few patterns.
    """
    if pdr is None:
        return None, FetchAttempt("STOOQ", "", "NO_MODULE", note="pandas_datareader not installed")

    # Common stooq pattern: 2330.TW? not guaranteed
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
            df = df.reset_index().rename(columns={"Date": "Date"})
            df = _normalize_ohlcv(df)
            return df, FetchAttempt("STOOQ", f"stooq:{sym}", "OK", note=f"rows={len(df)}")
        except Exception as e:
            last_note = str(e)
            continue

    return None, FetchAttempt("STOOQ", "stooq", "FAIL", note=last_note or "download failed")


def fetch_ohlcv_multi(code: str, months_back: int, debug: bool) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:
    """
    Multi-backup:
    1) TWSE JSON (official, stable if reachable)
    2) yfinance (may be blocked)
    3) stooq (backup)
    """
    period_days = int(months_back * 31)

    # 1) TWSE JSON
    df1, att1 = fetch_twse_json(code, months_back)
    if df1 is not None and len(df1) >= 30:
        return df1, "TWSE_JSON", att1

    # 2) YF
    df2, att2 = fetch_yfinance(code, period_days)
    atts = att1 + [att2]
    if df2 is not None and len(df2) >= 30:
        return df2, "YF", atts

    # 3) STOOQ
    df3, att3 = fetch_stooq(code, period_days)
    atts = atts + [att3]
    if df3 is not None and len(df3) >= 30:
        return df3, "STOOQ", atts

    return None, "NONE", atts


# -----------------------------
# Signal / Zones / Scoring
# -----------------------------
def support_resistance(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]:
    d = df.tail(lookback)
    sup = float(np.nanmin(d["Low"].values))
    res = float(np.nanmax(d["High"].values))
    return sup, res


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

    # volume MA
    out["VMA20"] = sma(out["Volume"].fillna(method="ffill"), 20)

    return out


def confluence_score(last: pd.Series) -> Tuple[int, Dict[str, bool]]:
    """
    0–100：偏保守、可解釋
    """
    flags = {}

    # trend
    flags["price_above_ema20"] = bool(last["Close"] > last["EMA20"]) if pd.notna(last["EMA20"]) else False
    flags["ema20_above_sma60"] = bool(last["EMA20"] > last["SMA60"]) if pd.notna(last["SMA60"]) else False

    # macd
    flags["macd_bull"] = bool(last["MACD"] > last["MACD_SIG"]) if pd.notna(last["MACD_SIG"]) else False
    flags["macd_hist_up"] = bool(last["MACD_HIST"] > 0) if pd.notna(last["MACD_HIST"]) else False

    # rsi regime
    flags["rsi_ok"] = bool(45 <= last["RSI14"] <= 70) if pd.notna(last["RSI14"]) else False
    flags["rsi_oversold"] = bool(last["RSI14"] < 35) if pd.notna(last["RSI14"]) else False
    flags["rsi_overbought"] = bool(last["RSI14"] > 75) if pd.notna(last["RSI14"]) else False

    # stoch
    flags["kd_bull"] = bool(last["K"] > last["D"]) if pd.notna(last["D"]) else False
    flags["kd_oversold"] = bool(last["K"] < 20) if pd.notna(last["K"]) else False

    # bollinger
    flags["near_bb_l"] = bool(last["Close"] <= last["BB_L"] * 1.01) if pd.notna(last["BB_L"]) else False
    flags["near_bb_u"] = bool(last["Close"] >= last["BB_U"] * 0.99) if pd.notna(last["BB_U"]) else False

    # volume
    flags["vol_up"] = bool(last["Volume"] > last["VMA20"]) if pd.notna(last["VMA20"]) and pd.notna(last["Volume"]) else False

    score = 0
    # weights (sum ~100)
    score += 18 if flags["price_above_ema20"] else 0
    score += 10 if flags["ema20_above_sma60"] else 0
    score += 14 if flags["macd_bull"] else 0
    score += 8 if flags["macd_hist_up"] else 0
    score += 12 if flags["rsi_ok"] else 0
    score += 10 if flags["kd_bull"] else 0
    score += 8 if flags["vol_up"] else 0
    # bonus: oversold reversal setup
    score += 10 if (flags["rsi_oversold"] and flags["kd_oversold"]) else 0
    # penalty: overbought
    score -= 10 if flags["rsi_overbought"] else 0
    score -= 8 if flags["near_bb_u"] else 0

    score = int(np.clip(score, 0, 100))
    return score, flags


def zone_from_reference(ref: float, width_pct: float) -> Tuple[float, float, float]:
    """
    return (low, high, center)
    """
    center = ref
    low = center * (1 - width_pct / 2)
    high = center * (1 + width_pct / 2)
    return float(low), float(high), float(center)


def professional_distance_text(current: float, center: float, atr_v: float, max_buy_dist: float) -> Dict[str, str]:
    """
    A/B：溢價/折價% + ATR distance + Entry Efficiency
    """
    if center <= 0 or current <= 0:
        return {"pct": "N/A", "atr": "N/A", "eff": "N/A"}

    dist_pct = (current - center) / center  # + means current > center (premium)
    dist_atr = (current - center) / atr_v if atr_v and atr_v > 0 else np.nan

    # efficiency: closer to zone center = higher
    # scale by max_buy_dist (e.g. 0.12)
    denom = max(max_buy_dist, 1e-6)
    eff = 1 - (abs(dist_pct) / denom)
    eff = float(np.clip(eff, 0, 1))

    pct_text = f"{dist_pct*100:+.2f}%（現價{'溢價' if dist_pct>0 else '折價'}）"
    atr_text = f"{dist_atr:+.2f} ATR" if not np.isnan(dist_atr) else "N/A"
    eff_text = f"{eff*100:.0f}/100"

    return {"pct": pct_text, "atr": atr_text, "eff": eff_text}


def build_trade_plan(
    df: pd.DataFrame,
    max_buy_distance: float,
    zone_width_pct: float,
    stop_atr_mult: float,
    target_rr: float,
) -> Dict[str, Dict]:
    """
    C：風險報酬（止損/目標）
    """
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_v = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan

    sup, res = support_resistance(df, lookback=80)
    bb_l = float(last["BB_L"]) if pd.notna(last["BB_L"]) else np.nan
    bb_m = float(last["BB_M"]) if pd.notna(last["BB_M"]) else np.nan
    bb_u = float(last["BB_U"]) if pd.notna(last["BB_U"]) else np.nan
    ema20_v = float(last["EMA20"]) if pd.notna(last["EMA20"]) else close

    # --- Buy zones (more realistic / closer to market)
    # near-term: between (EMA20 - 0.8*ATR) and (BB_L + small)
    # but don't exceed max_buy_distance
    if not np.isnan(atr_v) and atr_v > 0:
        near_ref = np.nanmean([ema20_v - 0.6 * atr_v, bb_l])
    else:
        near_ref = np.nanmean([ema20_v, bb_l])

    # clamp: ensure center isn't "too far" from current
    # if it's too far, pull it closer (still keep "wait" concept)
    if near_ref and near_ref > 0:
        dist = (close - near_ref) / near_ref
        if dist > max_buy_distance:
            near_ref = close / (1 + max_buy_distance)  # bring to max distance boundary

    near_low, near_high, near_center = zone_from_reference(float(near_ref), zone_width_pct)

    # deep buy: structural support area (support or BB_L-ATR)
    deep_ref = min(sup * 1.02, (bb_l - (atr_v if (not np.isnan(atr_v) and atr_v > 0) else 0)) * 1.02)
    if deep_ref and deep_ref > 0:
        dist2 = (close - deep_ref) / deep_ref
        # deep zone allowed to be farther, but still show
    deep_low, deep_high, deep_center = zone_from_reference(float(deep_ref), zone_width_pct)

    # sell zone: near resistance / upper band
    sell_ref = np.nanmean([bb_u, res * 0.98])
    sell_low, sell_high, sell_center = zone_from_reference(float(sell_ref), zone_width_pct)

    # --- Stop / Targets (based on near_center as entry baseline)
    entry = near_center
    stop = entry - (stop_atr_mult * atr_v) if (not np.isnan(atr_v) and atr_v > 0) else entry * 0.93
    stop = max(stop, 0.01)
    risk_per_share = entry - stop
    target = entry + (risk_per_share * target_rr) if risk_per_share > 0 else entry * 1.06

    # alternative target: min(target, sell_center) to keep realistic
    if sell_center and sell_center > 0:
        target2 = min(target, sell_center)
    else:
        target2 = target

    # current RR if buy now at close
    entry_now = close
    stop_now = entry_now - (stop_atr_mult * atr_v) if (not np.isnan(atr_v) and atr_v > 0) else entry_now * 0.93
    risk_now = entry_now - stop_now
    target_now = entry_now + (risk_now * target_rr) if risk_now > 0 else entry_now * 1.06
    rr_now = (target_now - entry_now) / risk_now if risk_now > 0 else np.nan

    return {
        "near_buy": {"low": near_low, "high": near_high, "center": near_center},
        "deep_buy": {"low": deep_low, "high": deep_high, "center": deep_center},
        "sell": {"low": sell_low, "high": sell_high, "center": sell_center},
        "rr": {
            "entry_ref": entry,
            "stop": stop,
            "target": target2,
            "entry_now": entry_now,
            "stop_now": stop_now,
            "target_now": target_now,
            "rr_now": rr_now,
            "atr": atr_v,
        },
    }


def position_sizing(
    account_ntd: float,
    fx_ntd_to_usd: float,
    entry: float,
    stop: float,
    max_position_pct: float,
    risk_pct: float,
) -> Dict[str, float]:
    """
    D：倉位計算（台幣資金，回傳建議股數/部位）
    - risk_pct: 每筆最大可承擔損失（例如 0.8%）
    - max_position_pct: 單筆最大部位佔用資金（例如 20%）
    """
    # Using NTD for capital; price is TWD.
    capital = float(account_ntd)
    risk_amt = capital * risk_pct
    max_pos_amt = capital * max_position_pct

    per_share_risk = max(entry - stop, 0.0)
    if per_share_risk <= 0:
        return {
            "risk_amt": risk_amt,
            "max_pos_amt": max_pos_amt,
            "per_share_risk": per_share_risk,
            "shares": 0.0,
            "pos_value": 0.0,
            "pos_pct": 0.0,
        }

    shares_by_risk = risk_amt / per_share_risk
    shares_by_cap = max_pos_amt / entry if entry > 0 else 0.0
    shares = max(0.0, min(shares_by_risk, shares_by_cap))

    # round down to board-lot style? Taiwan often 1 share now; keep integer
    shares_int = math.floor(shares)

    pos_value = shares_int * entry
    pos_pct = pos_value / capital if capital > 0 else 0.0

    return {
        "risk_amt": risk_amt,
        "max_pos_amt": max_pos_amt,
        "per_share_risk": per_share_risk,
        "shares": float(shares_int),
        "pos_value": float(pos_value),
        "pos_pct": float(pos_pct),
    }


def action_decision(df: pd.DataFrame, plan: Dict, flags: Dict[str, bool], max_buy_distance: float) -> Tuple[str, str]:
    """
    當下是否買/賣/觀望（可操作判斷）
    """
    last = df.iloc[-1]
    close = float(last["Close"])

    near = plan["near_buy"]
    sell = plan["sell"]

    # price relative to zone
    near_in = (near["low"] <= close <= near["high"])
    sell_in = (sell["low"] <= close <= sell["high"])

    # additional filters: don't chase far above near zone
    dist_pct = (close - near["center"]) / near["center"] if near["center"] > 0 else 999
    too_far = dist_pct > max_buy_distance

    # Buy condition: in zone + bullish reversal / confluence
    buy_ok = near_in and (flags.get("kd_bull", False) or flags.get("macd_bull", False) or flags.get("rsi_oversold", False))

    # Sell condition: in sell zone + overbought or weakening
    sell_ok = sell_in and (flags.get("rsi_overbought", False) or (not flags.get("macd_hist_up", True)))

    if buy_ok and not too_far:
        return "BUY", "條件符合：價格進入近端佈局區，且指標出現翻多/止跌跡象（可分批布局，嚴守止損）。"

    if sell_ok:
        return "SELL", "條件符合：價格進入壓力/布林上緣區，且出現過熱或動能轉弱（可分批減碼/停利）。"

    if too_far:
        return "WATCH", "觀望：現價相對近端買區『溢價過高』，等待回檔進入區間或指標翻轉再操作。"

    return "WATCH", "觀望：條件尚未形成明確共振，等待價格進入區間或指標翻轉。"


# -----------------------------
# Charts (matplotlib only)
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


def plot_indicators_panel(df: pd.DataFrame, code: str):
    d = df.tail(180).copy()

    fig = plt.figure(figsize=(12, 8.6), dpi=140)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(d.index, d["Close"], linewidth=1.2, label="Close")
    ax1.plot(d.index, d["EMA20"], linewidth=1.0, label="EMA20")
    ax1.plot(d.index, d["SMA60"], linewidth=1.0, label="SMA60")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left")
    ax1.set_title(f"{code} 價格與均線")

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(d.index, d["MACD"], linewidth=1.0, label="MACD")
    ax2.plot(d.index, d["MACD_SIG"], linewidth=1.0, label="Signal")
    ax2.axhline(0, linewidth=0.8)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left")
    ax2.set_title("MACD")

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(d.index, d["RSI14"], linewidth=1.0, label="RSI14")
    ax3.axhline(70, linewidth=0.8)
    ax3.axhline(30, linewidth=0.8)
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper left")
    ax3.set_title("RSI")

    fig.tight_layout()
    st.pyplot(fig)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("設定")

    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    months_map = {"3mo": 3, "6mo": 6, "1y": 12}
    period_label = st.selectbox("資料期間", list(months_map.keys()), index=1)
    months_back = months_map[period_label]

    debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)
    st.caption("若 Cloud 偶發連不到資料，先按一次『網路測試』再重試。")

    st.divider()
    st.subheader("A+B+C：專業買賣點 / 風險報酬")

    max_buy_distance = st.slider(
        "可操作買點最大距離（避免買點離現實太遠） max_buy_distance",
        min_value=0.03, max_value=0.25, value=0.12, step=0.01
    )
    zone_width_pct = st.slider("買賣區間寬度（%）", 0.02, 0.12, 0.05, 0.01)
    stop_atr_mult = st.slider("止損距離（ATR 倍數）", 0.8, 2.8, 1.6, 0.1)
    target_rr = st.slider("目標風險報酬（RR）", 1.2, 4.0, 2.0, 0.1)

    st.divider()
    st.subheader("D：資金 500–1000萬 倉位建議")

    account_ntd = st.slider("資金規模（NTD）", 5_000_000, 10_000_000, 7_000_000, step=100_000)
    risk_pct = st.slider("每筆最大風險（% of 資金）", 0.2, 2.0, 0.8, 0.1) / 100.0
    max_position_pct = st.slider("單筆最大持倉（% of 資金）", 5.0, 40.0, 20.0, 1.0) / 100.0


# -----------------------------
# Main
# -----------------------------
def render_diagnostics(attempts: List[FetchAttempt]):
    if not attempts:
        return
    df = pd.DataFrame([{
        "source": a.source,
        "result": a.status,
        "url": a.url,
        "http": a.http,
        "bytes": a.bytes,
        "note": a.note,
        "preview": a.preview
    } for a in attempts])
    st.markdown("### 🧩 逐路診斷（哪一路失敗、為什麼）")
    st.dataframe(df, use_container_width=True, hide_index=True)


def analyze_one(code: str):
    if not code or not str(code).strip():
        st.warning("請輸入股票代號（例如：2330 / 2317 / 6274）")
        return

    code = str(code).strip().upper().replace(" ", "")
    df, src, attempts = fetch_ohlcv_multi(code, months_back=months_back, debug=debug)

    if debug:
        render_diagnostics(attempts)

    if df is None or df.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試或換一檔股票。")
        return

    df = compute_signals(df)
    last = df.iloc[-1]
    score, flags = confluence_score(last)

    plan = build_trade_plan(
        df=df,
        max_buy_distance=max_buy_distance,
        zone_width_pct=zone_width_pct,
        stop_atr_mult=stop_atr_mult,
        target_rr=target_rr,
    )

    action, reason = action_decision(df, plan, flags, max_buy_distance=max_buy_distance)

    # --- Head metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{float(last['Close']):,.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", src)
    c4.metric("最後日期/筆數", f"{df.index[-1].date()} / {len(df)}")

    st.markdown("## 📌 當下是否為買點/賣點？（可操作判斷）")
    if action == "BUY":
        st.success(f"🟢 **BUY / 可操作**：{reason}")
    elif action == "SELL":
        st.warning(f"🟠 **SELL / 可操作**：{reason}")
    else:
        st.info(f"🕒 **WATCH / 觀望**：{reason}")

    # --- Professional zones
    st.markdown("## 🗺️ 未來預估買賣點（區間 + 專業距離）")

    close = float(last["Close"])
    atr_v = float(plan["rr"]["atr"]) if plan["rr"]["atr"] and not np.isnan(plan["rr"]["atr"]) else np.nan

    def zone_card(title: str, z: Dict, tag: str):
        center = float(z["center"])
        low = float(z["low"])
        high = float(z["high"])

        dist = professional_distance_text(
            current=close,
            center=center,
            atr_v=atr_v if not np.isnan(atr_v) else 0.0,
            max_buy_dist=max_buy_distance
        )
        st.markdown(
            f"""
**{title}**  
區間：**{low:,.2f} – {high:,.2f}**（中心：{center:,.2f}）  
• 現價相對中心：**{dist['pct']}**  
• 波動標準化距離：**{dist['atr']}**  
• Entry Efficiency：**{dist['eff']}**  
• 區域屬性：{tag}
"""
        )

    cA, cB = st.columns(2)
    with cA:
        zone_card("🟢 近端買點（Primary Tactical Entry / 可操作）", plan["near_buy"], "回檔型順勢佈局區（分批進）")
    with cB:
        zone_card("🟦 深回檔買點（Strategic Accumulation / 等待型）", plan["deep_buy"], "結構支撐型累積區（大回檔才到）")

    st.markdown("---")
    zone_card("🟥 近端賣點（Profit-taking / 壓力區）", plan["sell"], "壓力/布林上緣區（分批減碼/停利）")

    # --- Risk/Reward (C)
    st.markdown("## 🎯 風險報酬（C：止損 / 目標 / RR）")
    rr = plan["rr"]

    rr_cols = st.columns(4)
    rr_cols[0].metric("參考進場（近端中心）", f"{rr['entry_ref']:,.2f}")
    rr_cols[1].metric("止損（ATR×倍數）", f"{rr['stop']:,.2f}")
    rr_cols[2].metric("目標（RR×）", f"{rr['target']:,.2f}")
    rr_cols[3].metric("若『現在市價』進場 RR", f"{rr['rr_now']:.2f}" if not np.isnan(rr["rr_now"]) else "N/A")

    st.caption("說明：止損以 ATR 倍數設定（更貼近真實波動），目標以 RR 設定並對齊壓力區，避免不切實際的超遠目標。")

    # --- Position sizing (D)
    st.markdown("## 🧮 倉位建議（D：500–1000萬 / 風險控管）")
    size = position_sizing(
        account_ntd=account_ntd,
        fx_ntd_to_usd=1.0,
        entry=rr["entry_ref"],
        stop=rr["stop"],
        max_position_pct=max_position_pct,
        risk_pct=risk_pct,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("每筆可承擔風險（NTD）", f"{size['risk_amt']:,.0f}")
    d2.metric("單股風險（Entry-Stop）", f"{size['per_share_risk']:,.2f}")
    d3.metric("建議股數（shares）", f"{size['shares']:,.0f}")
    d4.metric("部位金額 / 佔資金", f"{size['pos_value']:,.0f} / {size['pos_pct']*100:.1f}%")

    st.caption("計算邏輯：股數 = min(風險金額/每股風險, 最大部位金額/進場價)。若想更保守，降低『每筆最大風險%』或『最大持倉%』。")

    # --- Indicators summary
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
    }])
    st.dataframe(ind, use_container_width=True, hide_index=True)

    # --- Charts
    st.markdown("## 🧷 布林通道圖（無 Plotly / Matplotlib）")
    plot_bollinger(df, code)

    st.markdown("## 📊 指標面板（趨勢 + MACD + RSI）")
    plot_indicators_panel(df, code)


def scan_top10(stock_pool: List[str]):
    # unique + keep order
    seen = set()
    pool = []
    for s in stock_pool:
        s = str(s).strip().upper().replace(" ", "")
        if not s:
            continue
        if s not in seen:
            pool.append(s)
            seen.add(s)

    if not pool:
        st.warning("stock_pool 為空，請輸入股票池。")
        return

    rows = []
    diag_map: Dict[str, List[FetchAttempt]] = {}

    # limit per run to avoid cloud timeouts
    hard_limit = 35
    pool = pool[:hard_limit]

    prog = st.progress(0)
    for i, code in enumerate(pool, start=1):
        df, src, attempts = fetch_ohlcv_multi(code, months_back=months_back, debug=False)
        diag_map[code] = attempts

        if df is None or df.empty:
            prog.progress(i / len(pool))
            continue

        df = compute_signals(df)
        last = df.iloc[-1]
        score, flags = confluence_score(last)
        plan = build_trade_plan(
            df=df,
            max_buy_distance=max_buy_distance,
            zone_width_pct=zone_width_pct,
            stop_atr_mult=stop_atr_mult,
            target_rr=target_rr,
        )
        action, _ = action_decision(df, plan, flags, max_buy_distance=max_buy_distance)

        # Dedup issue fix: always use code as key; never append duplicates
        rows.append({
            "股票": code,
            "來源": src,
            "AI分數": score,
            "當下判斷": action,
            "現價": float(last["Close"]),
            "近端買點中心": float(plan["near_buy"]["center"]),
            "距離(%)": (float(last["Close"]) - float(plan["near_buy"]["center"])) / float(plan["near_buy"]["center"]) * 100
                      if plan["near_buy"]["center"] > 0 else np.nan,
        })
        prog.progress(i / len(pool))

    if not rows:
        st.warning("掃描結果為空（可能資料源暫時抓不到）。稍後再試或縮小股票池。")
        return

    out = pd.DataFrame(rows)

    # 防止 Top10 重複：以股票唯一
    out = out.drop_duplicates(subset=["股票"], keep="first")

    # Sort: prefer BUY, then high score, then closer to near buy zone
    action_rank = {"BUY": 0, "WATCH": 1, "SELL": 2}
    out["rank_action"] = out["當下判斷"].map(action_rank).fillna(9).astype(int)
    out["abs_dist"] = out["距離(%)"].abs()

    out = out.sort_values(["rank_action", "AI分數", "abs_dist"], ascending=[True, False, True]).head(10)
    out = out.drop(columns=["rank_action", "abs_dist"])

    st.markdown("## 🔥 AI 強勢股 Top 10（可操作優先排序）")
    st.dataframe(out, use_container_width=True, hide_index=True)

    if debug:
        st.markdown("### 🧩 Top10 逐檔診斷（Debug）")
        pick = st.selectbox("選擇要看診斷的股票", out["股票"].tolist())
        render_diagnostics(diag_map.get(pick, []))


# -----------------------------
# Mode switch
# -----------------------------
if mode == "單一股票分析":
    colL, colR = st.columns([1.2, 3.6])
    with colL:
        code = st.text_input("請輸入股票代號", value="2330")
        if st.button("網路測試（建議先按一次）"):
            # simple ping
            url = "https://www.twse.com.tw/"
            r, err = _requests_get(url, timeout=8)
            if r is None:
                st.error(f"TWSE ping 失敗：{err}")
            else:
                st.success(f"TWSE ping：HTTP {r.status_code} / bytes {len(r.text or '')}")
    with colR:
        analyze_one(code)

else:
    st.markdown("## Top 10 掃描器")
    st.caption("提示：雲端環境避免一次掃全市場；建議先用小池（20–35 檔）測試。")

    default_pool = "2330,2317,2454,3037,8046,2382,2303,4967,2603,2609,2882,2881,2891,0050,006208"
    pool_text = st.text_area("輸入股票池（逗號分隔）", value=default_pool, height=110)
    stock_pool = [x.strip() for x in pool_text.split(",") if x.strip()]

    scan_top10(stock_pool)

st.caption("⚠️ 本工具僅供資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。")
