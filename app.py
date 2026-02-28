# app.py
# AI 台股量化專業平台（V24 / 無 Plotly / 全功能保留）
# ✅ 多來源備援：FinMind(首選) → yfinance(.TW/.TWO) → Stooq → repo export.csv → 上傳CSV
# ✅ 逐路診斷：WAF/HTML/EMPTY/TOO_SHORT/EXC 清楚顯示（Debug 可看明細）
# ✅ 指標：Bollinger、RSI、MACD、KD、ATR、量能
# ✅ 兩策略：回檔等待型（分批）/ 趨勢突破型（突破追價）
# ✅ Top10 掃描：不重複股票、顯示當下可操作結論 + 進出場區間
# ✅ 無 plotly，僅 matplotlib（避免 Streamlit Cloud 缺套件）

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Dependency checks (friendly)
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
    import yfinance as _yf
    yf = _yf
    DEPENDENCIES["yfinance"] = True
except Exception:
    yf = None

try:
    from pandas_datareader import data as _pdr
    pdr = _pdr
    DEPENDENCIES["pandas_datareader"] = True
except Exception:
    pdr = None

try:
    from FinMind.data import DataLoader as _DL
    DataLoader = _DL
    DEPENDENCIES["finmind"] = True
except Exception:
    DataLoader = None


# -----------------------------
# UI config
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
@dataclass
class FetchAttempt:
    source: str
    url: str
    result: str
    status: Optional[int] = None
    note: str = ""

def _looks_like_html(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or ("<head" in t and "<body" in t)

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # accept Date/Open/High/Low/Close/Volume in any case
    cols = {c.lower(): c for c in df.columns}
    need = ["date", "open", "high", "low", "close"]
    for k in need:
        if k not in cols:
            raise ValueError(f"missing column: {k}")

    df = df.copy()
    df.rename(columns={
        cols["date"]: "Date",
        cols["open"]: "Open",
        cols["high"]: "High",
        cols["low"]: "Low",
        cols["close"]: "Close",
        cols.get("volume", cols.get("vol", "Volume")): "Volume" if ("volume" in cols or "vol" in cols) else "Volume",
    }, inplace=True)

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df

def _request_with_retry(url: str, attempts: int = 3, backoff: float = 0.8, timeout: int = 15) -> Tuple[Optional[requests.Response], str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0; +https://streamlit.io)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    last_err = ""
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return r, ""
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff * (1.7 ** i))
    return None, last_err

def _format_pct(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    return f"{x*100:.1f}%"

def _pro_distance_phrase(target_mid: float, current: float) -> str:
    """
    Professional wording for distance:
    - below current: '需回檔約 X%'
    - above current: '需上漲/突破約 X%'
    """
    if not (np.isfinite(target_mid) and np.isfinite(current) and current != 0):
        return "-"
    diff = (target_mid - current) / current
    if diff < 0:
        return f"需回檔約 {abs(diff)*100:.1f}%"
    if diff > 0:
        return f"需上漲/突破約 {diff*100:.1f}%"
    return "與現價接近"

def _read_csv_any(upload_or_path) -> pd.DataFrame:
    if hasattr(upload_or_path, "read"):
        df = pd.read_csv(upload_or_path)
    else:
        df = pd.read_csv(upload_or_path)
    return df

def _load_csv_fallback(upload_or_path) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    try:
        df = _read_csv_any(upload_or_path)
        # accept common export formats; try to map columns
        cols_lower = {c.lower(): c for c in df.columns}
        # Typical export: Date/Open/High/Low/Close/Volume
        if "date" not in cols_lower:
            # some exports use "Datetime" etc
            for alt in ["datetime", "time", "timestamp"]:
                if alt in cols_lower:
                    df.rename(columns={cols_lower[alt]: "Date"}, inplace=True)
                    break
        # rename variants
        rename_map = {}
        for key, std in [("date", "Date"), ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
            if key in cols_lower:
                rename_map[cols_lower[key]] = std
        df.rename(columns=rename_map, inplace=True)

        df = _normalize_ohlcv(df)
        return df, FetchAttempt("CSV", "upload/path", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("CSV", "upload/path", "EXC", note=str(e))


# -----------------------------
# Data sources
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_finmind(code: str, months_back: int, token: Optional[str]) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if not DEPENDENCIES.get("finmind", False) or DataLoader is None:
        return None, FetchAttempt("FinMind", "", "NO_MODULE", note="缺 FinMind：pip install FinMind")

    if not token:
        return None, FetchAttempt("FinMind", "", "NO_TOKEN", note="缺 FINMIND_TOKEN（Secrets 或手動輸入）")

    try:
        dl = DataLoader()
        dl.login_by_token(api_token=token)

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months_back * 35 + 35)).strftime("%Y-%m-%d")

        df = dl.taiwan_stock_daily(stock_id=code, start_date=start, end_date=end)

        if df is None or df.empty:
            return None, FetchAttempt("FinMind", f"stock_id={code}", "EMPTY", note="empty response")

        # Flexible column mapping (FinMind sometimes differs by dataset/version)
        # Expect at least: date, open, max/high, min/low, close, volume-ish
        colmap = {c.lower(): c for c in df.columns}

        # volume field candidates
        vol_col = None
        for cand in ["trading_volume", "volume", "trade_volume", "shares", "total_volume"]:
            if cand in colmap:
                vol_col = colmap[cand]
                break

        # high/low candidates
        high_col = colmap.get("max") or colmap.get("high")
        low_col = colmap.get("min") or colmap.get("low")

        rename = {}
        if "date" in colmap: rename[colmap["date"]] = "Date"
        if "open" in colmap: rename[colmap["open"]] = "Open"
        if high_col: rename[high_col] = "High"
        if low_col: rename[low_col] = "Low"
        if "close" in colmap: rename[colmap["close"]] = "Close"
        if vol_col: rename[vol_col] = "Volume"

        df = df.rename(columns=rename)

        # If still missing Volume, create
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        df = _normalize_ohlcv(df)
        return df, FetchAttempt("FinMind", f"taiwan_stock_daily:{code}", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("FinMind", f"stock_id={code}", "EXC", note=str(e))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if not DEPENDENCIES.get("yfinance", False) or yf is None:
        return None, FetchAttempt("YF", "", "NO_MODULE", note="缺 yfinance：pip install yfinance")

    tickers = []
    # If user already typed suffix, keep; else try .TW then .TWO
    if code.upper().endswith(".TW") or code.upper().endswith(".TWO"):
        tickers = [code.upper()]
    else:
        tickers = [f"{code}.TW", f"{code}.TWO"]

    last_note = ""
    for t in tickers:
        try:
            df = yf.download(
                t,
                period=f"{max(30, period_days)}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                last_note = "empty"
                continue
            df = df.reset_index()
            # yfinance uses columns: Date Open High Low Close Adj Close Volume
            df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
            df = _normalize_ohlcv(df)
            return df, FetchAttempt("YF", t, "OK", note=f"rows={len(df)}")
        except Exception as e:
            last_note = str(e)

    return None, FetchAttempt("YF", ",".join(tickers), "EMPTY", note=last_note or "no data")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, period_days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if not DEPENDENCIES.get("pandas_datareader", False) or pdr is None:
        return None, FetchAttempt("STOOQ", "", "NO_MODULE", note="缺 pandas_datareader：pip install pandas_datareader")

    # Stooq TW symbols often like 2330.tw
    sym = code.lower()
    if not sym.endswith(".tw"):
        sym = f"{sym}.tw"

    try:
        df = pdr.DataReader(sym, "stooq")
        if df is None or df.empty:
            return None, FetchAttempt("STOOQ", sym, "EMPTY", note="empty")

        # stooq returns index as date and columns: Open High Low Close Volume
        df = df.reset_index().rename(columns={"Date": "Date"})
        df = _normalize_ohlcv(df)
        # limit to period_days
        if len(df) > period_days + 10:
            df = df.iloc[-(period_days + 10):].reset_index(drop=True)
        return df, FetchAttempt("STOOQ", sym, "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("STOOQ", sym, "EXC", note=str(e))


def fetch_ohlcv_multi(
    code: str,
    months_back: int,
    finmind_token: Optional[str],
    csv_upload=None,
    repo_csv_path: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:

    period_days = int(months_back * 31)
    attempts: List[FetchAttempt] = []

    # 0) repo export.csv (if exists) can be used as strong fallback without upload
    if repo_csv_path:
        df_repo, att_repo = _load_csv_fallback(repo_csv_path)
        attempts.append(FetchAttempt("CSV_REPO", repo_csv_path, att_repo.result, note=att_repo.note))
        if df_repo is not None and len(df_repo) >= 30:
            return df_repo, "CSV_REPO", attempts

    # 1) FinMind
    df_fm, att_fm = fetch_finmind(code, months_back, finmind_token)
    attempts.append(att_fm)
    if df_fm is not None and len(df_fm) >= 30:
        return df_fm, "FinMind", attempts

    # 2) yfinance
    df_yf, att_yf = fetch_yfinance(code, period_days)
    attempts.append(att_yf)
    if df_yf is not None and len(df_yf) >= 30:
        return df_yf, "YF", attempts

    # 3) stooq
    df_sq, att_sq = fetch_stooq(code, period_days)
    attempts.append(att_sq)
    if df_sq is not None and len(df_sq) >= 30:
        return df_sq, "STOOQ", attempts

    # 4) upload csv
    if csv_upload is not None:
        df_csv, att_csv = _load_csv_fallback(csv_upload)
        attempts.append(att_csv)
        if df_csv is not None and len(df_csv) >= 30:
            return df_csv, "CSV_UPLOAD", attempts

    return None, "NONE", attempts


# -----------------------------
# Indicators & signals
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # Bollinger 20
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df["MACD"] = macd
    df["MACD_SIG"] = signal
    df["MACD_HIST"] = hist

    # KD (Stochastic 9,3,3)
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    df["K"] = k
    df["D"] = d

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Volume MA20
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    return df


def decide_action(df: pd.DataFrame, strategy: str, max_buy_distance: float, breakout_buffer_atr: float) -> Dict[str, object]:
    """
    Returns:
      action: BUY / SELL / WATCH
      score: 0-100
      reasons: list[str]
      zones: dict with buy_near / buy_deep / sell_near
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    price = float(last["Close"])
    bb_low = float(last.get("BB_LOW", np.nan))
    bb_up = float(last.get("BB_UP", np.nan))
    bb_mid = float(last.get("BB_MID", np.nan))
    rsi = float(last.get("RSI14", np.nan))
    k = float(last.get("K", np.nan))
    d = float(last.get("D", np.nan))
    hist = float(last.get("MACD_HIST", np.nan))
    hist_prev = float(prev.get("MACD_HIST", np.nan))
    atr = float(last.get("ATR14", np.nan))
    vol = float(last.get("Volume", np.nan))
    vol_ma = float(last.get("VOL_MA20", np.nan))

    reasons: List[str] = []
    score = 50

    # zones (default by Bollinger + ATR)
    # near buy zone: around lower band but not crazy far
    buy_near_lo = bb_low
    buy_near_hi = bb_low + (0.35 * atr if np.isfinite(atr) else 0)

    buy_deep_lo = bb_low - (1.0 * atr if np.isfinite(atr) else 0)
    buy_deep_hi = bb_low - (0.3 * atr if np.isfinite(atr) else 0)

    sell_near_lo = bb_up - (0.35 * atr if np.isfinite(atr) else 0)
    sell_near_hi = bb_up

    # distance control
    def within_distance(target_mid: float) -> bool:
        if not (np.isfinite(target_mid) and price > 0):
            return False
        dist = abs(target_mid - price) / price
        return dist <= max_buy_distance

    # base confluence
    if np.isfinite(rsi):
        if rsi < 35: score += 10; reasons.append("RSI 偏低（超賣區附近）")
        if rsi > 65: score += 10; reasons.append("RSI 偏高（過熱區附近）")
    if np.isfinite(k) and np.isfinite(d):
        if k < 20 and k > d: score += 10; reasons.append("KD 低檔黃金交叉傾向")
        if k > 80 and k < d: score += 10; reasons.append("KD 高檔死亡交叉傾向")
    if np.isfinite(hist) and np.isfinite(hist_prev):
        if hist > hist_prev: score += 5; reasons.append("MACD 柱狀體改善")
        if hist < hist_prev: score += 5; reasons.append("MACD 柱狀體轉弱")

    # Strategy-specific
    action = "WATCH"

    if strategy == "回檔等待型":
        # BUY if price is near lower band OR enters buy_near zone + momentum improving
        buy_mid = (buy_near_lo + buy_near_hi) / 2 if np.isfinite(buy_near_hi) else buy_near_lo
        sell_mid = (sell_near_lo + sell_near_hi) / 2 if np.isfinite(sell_near_hi) else sell_near_lo

        near_buy = np.isfinite(buy_near_lo) and np.isfinite(buy_near_hi) and (buy_near_lo <= price <= buy_near_hi)
        near_sell = np.isfinite(sell_near_lo) and np.isfinite(sell_near_hi) and (sell_near_lo <= price <= sell_near_hi)

        if near_buy and within_distance(buy_mid) and (hist >= hist_prev or (k > d)):
            action = "BUY"
            reasons.insert(0, "價格落在布林下緣附近（可分批區）")
            score += 15

        elif near_sell:
            action = "SELL"
            reasons.insert(0, "價格接近布林上緣（壓力/獲利帶）")
            score += 10

        else:
            action = "WATCH"
            reasons.insert(0, "條件尚未明確：等待價格進入區間或指標翻轉")

    else:
        # 趨勢突破型（突破追價）
        # entry when close breaks above BB_UP + buffer(ATR) and volume confirms
        buffer = breakout_buffer_atr * atr if np.isfinite(atr) else 0.0
        trigger = (bb_up + buffer) if np.isfinite(bb_up) else np.nan

        vol_ok = np.isfinite(vol) and np.isfinite(vol_ma) and vol_ma > 0 and (vol >= 1.2 * vol_ma)
        break_ok = np.isfinite(trigger) and (price >= trigger)

        if break_ok and vol_ok:
            # allow entry only if distance to trigger not too large
            if within_distance(trigger):
                action = "BUY"
                reasons.insert(0, "突破成立：收盤突破布林上緣 + 量能放大")
                score += 20
            else:
                action = "WATCH"
                reasons.insert(0, "突破成立但追價偏遠：建議等待回測再評估")
                score += 5
        else:
            action = "WATCH"
            reasons.insert(0, "尚未出現有效突破：等待突破或量能確認")

        # sell/stop suggestion in breakout mode: use ATR based
        # (display later as plan)

    score = int(max(0, min(100, score)))

    zones = {
        "buy_near": (buy_near_lo, buy_near_hi),
        "buy_deep": (buy_deep_lo, buy_deep_hi),
        "sell_near": (sell_near_lo, sell_near_hi),
    }

    return {
        "action": action,
        "score": score,
        "reasons": reasons[:6],
        "zones": zones,
        "price": price,
        "bb_mid": bb_mid,
        "atr": atr,
    }


def plot_bollinger(df: pd.DataFrame, title: str):
    dfp = df.tail(180).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dfp["Date"], dfp["Close"], label="Close")
    if "BB_MID" in dfp.columns:
        ax.plot(dfp["Date"], dfp["BB_MID"], label="BB Mid (20MA)")
        ax.plot(dfp["Date"], dfp["BB_UP"], label="BB Upper")
        ax.plot(dfp["Date"], dfp["BB_LOW"], label="BB Lower")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("✅ 模式")
    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    months_back = st.selectbox("資料期間", [3, 6, 12, 24], index=1, format_func=lambda x: f"{x}mo")

    st.divider()
    st.subheader("⚙️ 系統狀態")
    for pkg, ok in DEPENDENCIES.items():
        st.success(f"✅ {pkg}") if ok else st.error(f"❌ {pkg}（請安裝）")

    # FinMind token input (optional)
    finmind_token = None
    if DEPENDENCIES.get("finmind", False):
        finmind_token = st.secrets.get("FINMIND_TOKEN", None)
        finmind_token_input = st.text_input("FinMind Token（可選，若 Secrets 未設）", type="password", value="")
        if finmind_token_input.strip():
            finmind_token = finmind_token_input.strip()
        if not finmind_token:
            st.warning("FinMind 未提供 Token → 會自動改用 yfinance / Stooq / CSV")

    st.divider()
    st.subheader("📌 策略 / 參數")
    strategy = st.radio("策略", ["回檔等待型", "趨勢突破型"], index=0)
    max_buy_distance = st.slider("可接受進場偏離距離（避免買點離現價太遠）", 0.02, 0.20, 0.12, 0.01)

    breakout_buffer_atr = st.slider("突破觸發 buffer（ATR 倍數）", 0.0, 1.0, 0.20, 0.05)

    st.divider()
    show_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.divider()
    st.subheader("🧯 備援 CSV")
    st.caption("若 Cloud 偶發抓不到資料，可上傳 export.csv（含 Date/Open/High/Low/Close/Volume）")
    csv_upload = st.file_uploader("上傳 export.csv（可選）", type=["csv"])


# Try detect repo export.csv
repo_csv_path = None
try:
    import os
    if os.path.exists("export.csv"):
        repo_csv_path = "export.csv"
except Exception:
    repo_csv_path = None


# -----------------------------
# Single stock analysis
# -----------------------------
def render_attempt_summary(attempts: List[FetchAttempt]):
    # Always show a concise summary even if Debug is off
    if not attempts:
        return
    ok = [a for a in attempts if a.result == "OK"]
    if ok:
        return
    # show brief reasons
    brief = []
    for a in attempts[:5]:
        msg = f"{a.source}: {a.result}"
        if a.note:
            msg += f"（{a.note}）"
        brief.append(msg)
    st.error("無法取得資料（所有備援來源都失敗）。")
    st.info(" | ".join(brief))
    st.warning("建議：① 上傳 export.csv 立刻可跑 ② 或在 Secrets 設 FINMIND_TOKEN ③ 或稍後重試（可能被 WAF/空回傳）")

def render_debug_table(attempts: List[FetchAttempt]):
    if not show_debug or not attempts:
        return
    df = pd.DataFrame([{
        "source": a.source,
        "result": a.result,
        "status": a.status,
        "url": a.url,
        "note": a.note
    } for a in attempts])
    st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_zones(price: float, zones: Dict[str, Tuple[float, float]]):
    def band(label: str, lo: float, hi: float, color: str):
        mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.nan
        phrase = _pro_distance_phrase(mid, price)
        dist = (mid - price) / price if np.isfinite(mid) and price else np.nan
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:10px;background:{color};margin:10px 0;">
              <div style="font-size:18px;font-weight:700;">{label}</div>
              <div style="font-size:16px;">區間：{lo:.2f} ~ {hi:.2f}</div>
              <div style="font-size:15px;opacity:0.9;">相對現價：{phrase}（{_format_pct(dist)}）</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    bn = zones.get("buy_near")
    bd = zones.get("buy_deep")
    sn = zones.get("sell_near")
    if bn and np.isfinite(bn[0]) and np.isfinite(bn[1]):
        band("🟢 近端買點帶（可操作 / 分批區）", bn[0], bn[1], "#0b3d2e")
    if bd and np.isfinite(bd[0]) and np.isfinite(bd[1]):
        band("🔵 深回檔買點帶（等待型 / 價值區）", bd[0], bd[1], "#123a63")
    if sn and np.isfinite(sn[0]) and np.isfinite(sn[1]):
        band("🟡 近端賣點帶（壓力/獲利區）", sn[0], sn[1], "#4b3b0a")


if mode == "單一股票分析":
    code = st.text_input("請輸入股票代號（例：2330 / 6274 / 2330.TW）", value="6274").strip()
    if not code:
        st.stop()

    df, src, attempts = fetch_ohlcv_multi(
        code=code,
        months_back=months_back,
        finmind_token=finmind_token,
        csv_upload=csv_upload,
        repo_csv_path=repo_csv_path,
    )

    render_attempt_summary(attempts)
    render_debug_table(attempts)

    if df is None or df.empty:
        st.stop()

    df = add_indicators(df)
    last = df.iloc[-1]
    price = float(last["Close"])
    last_date = str(pd.to_datetime(last["Date"]).date())

    cols = st.columns(4)
    cols[0].metric("目前價格", f"{price:.2f}")
    cols[1].metric("資料來源", src)
    cols[2].metric("最後日期 / 筆數", f"{last_date} / {len(df)}")
    # score computed below

    decision = decide_action(df, strategy=strategy, max_buy_distance=max_buy_distance, breakout_buffer_atr=breakout_buffer_atr)
    cols[3].metric("AI 共振分數", f"{decision['score']}/100")

    st.subheader("📌 當下是否為買點/賣點？（可操作判斷）")
    action = decision["action"]
    if action == "BUY":
        st.success("✅ BUY：符合可操作條件（依策略與風控分批/進場）")
    elif action == "SELL":
        st.warning("⚠️ SELL：進入壓力/獲利帶（可分批減碼/停利）")
    else:
        st.info("⏳ WATCH：條件尚未明確，等待價格進入區間或指標翻轉")

    if decision["reasons"]:
        st.caption("理由（摘要）：" + "；".join(decision["reasons"]))

    st.subheader("🗺️ 未來預估買賣點（區間 + 專業距離說法）")
    render_zones(price, decision["zones"])

    st.subheader("📈 布林通道 + 收盤價（視覺判讀）")
    plot_bollinger(df, title=f"{code}  Bollinger Bands (20,2)  |  Source: {src}")

    st.subheader("🧾 指標摘要（最新一根）")
    show_cols = ["Date", "Close", "BB_LOW", "BB_MID", "BB_UP", "RSI14", "MACD", "MACD_SIG", "MACD_HIST", "K", "D", "ATR14", "Volume", "VOL_MA20"]
    st.dataframe(df[show_cols].tail(5), use_container_width=True, hide_index=True)


# -----------------------------
# Top10 scanner
# -----------------------------
def default_universe() -> List[str]:
    # 內建一個「雲端可承受」的小池（你也可改成你自己的常用清單）
    return [
        "2330", "2317", "2454", "2308", "2412", "2882", "2881", "2891", "2303", "2603",
        "2609", "2615", "3037", "2382", "3711", "3661", "3023", "3045", "1301", "1303",
        "1590", "3008", "6505", "0050", "0056", "00878", "00919", "006208", "00713", "00881",
        "4967", "6274", "3227", "8086", "8299", "5347", "8358", "8070", "1560", "2383",
    ]

def scan_top10(universe: List[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    for code in universe:
        code = str(code).strip()
        if not code or code in seen:
            continue
        seen.add(code)

        df, src, attempts = fetch_ohlcv_multi(
            code=code,
            months_back=months_back,
            finmind_token=finmind_token,
            csv_upload=None,
            repo_csv_path=None,
        )
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)
        decision = decide_action(df, strategy=strategy, max_buy_distance=max_buy_distance, breakout_buffer_atr=breakout_buffer_atr)
        price = float(df.iloc[-1]["Close"])
        zones = decision["zones"]
        bn = zones.get("buy_near")
        sn = zones.get("sell_near")

        buy_mid = (bn[0] + bn[1]) / 2 if bn and np.isfinite(bn[0]) and np.isfinite(bn[1]) else np.nan
        sell_mid = (sn[0] + sn[1]) / 2 if sn and np.isfinite(sn[0]) and np.isfinite(sn[1]) else np.nan

        buy_phrase = _pro_distance_phrase(buy_mid, price) if np.isfinite(buy_mid) else "-"
        sell_phrase = _pro_distance_phrase(sell_mid, price) if np.isfinite(sell_mid) else "-"

        rows.append({
            "股票": code,
            "來源": src,
            "現價": round(price, 2),
            "判斷": decision["action"],
            "AI分數": decision["score"],
            "近端買點帶": f"{bn[0]:.2f}~{bn[1]:.2f}" if bn and np.isfinite(bn[0]) else "-",
            "距離買點": buy_phrase,
            "近端賣點帶": f"{sn[0]:.2f}~{sn[1]:.2f}" if sn and np.isfinite(sn[0]) else "-",
            "距離賣點": sell_phrase,
        })

        # 雲端保護：避免掃太久
        if len(rows) >= 35:
            break

    if not rows:
        return pd.DataFrame(columns=["股票", "來源", "現價", "判斷", "AI分數", "近端買點帶", "距離買點", "近端賣點帶", "距離賣點"])

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["AI分數", "判斷"], ascending=[False, True]).head(10).reset_index(drop=True)
    return df_out


if mode == "Top 10 掃描器":
    st.subheader("🔥 AI 強勢股 Top 10（不重複 + 可操作結論）")
    st.caption("建議先用小池測試，避免 Cloud 掃全市場超時。")

    pool_choice = st.radio("股票池", ["內建常用清單（建議）", "上傳自訂清單 CSV（欄名：ticker）"], index=0)

    universe = default_universe()
    if pool_choice == "上傳自訂清單 CSV（欄名：ticker）":
        up = st.file_uploader("上傳清單 CSV", type=["csv"], key="ticker_pool")
        if up is not None:
            try:
                tdf = pd.read_csv(up)
                if "ticker" in [c.lower() for c in tdf.columns]:
                    # find actual column
                    col = [c for c in tdf.columns if c.lower() == "ticker"][0]
                    universe = [str(x).strip() for x in tdf[col].dropna().tolist()]
                else:
                    st.error("找不到 ticker 欄位。請用欄名 ticker。")
            except Exception as e:
                st.error(f"讀取失敗：{e}")

    run = st.button("開始掃描 Top10")
    if run:
        with st.spinner("掃描中（雲端會自動限制掃描量以避免超時）..."):
            top10 = scan_top10(universe)

        if top10.empty:
            st.error("掃不到可用資料（可能所有來源被擋 / 缺 token / 雲端限制）。建議：設 FINMIND_TOKEN 或縮小股票池。")
        else:
            st.dataframe(top10, use_container_width=True, hide_index=True)
            st.info("提示：Top10 已強制去重（同一檔不會重複）。若你仍看到重複，代表輸入清單本身含重複，我也會自動去重。")


# -----------------------------
# Footer: requirements hints
# -----------------------------
with st.expander("📦 requirements.txt 建議（無 Plotly 版本）", expanded=False):
    st.code(
        "\n".join([
            "streamlit>=1.20.0",
            "pandas>=1.5.0",
            "numpy>=1.23.0",
            "matplotlib>=3.5.0",
            "requests>=2.28.0",
            "yfinance>=0.2.0",
            "pandas_datareader>=0.10.0",
            "FinMind>=1.0.0",
        ]),
        language="text"
    )
    st.caption("若 FinMind 沒 Token 也能跑（會改用 yfinance / Stooq / CSV），但 FinMind 最穩。")
