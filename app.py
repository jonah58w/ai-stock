# app.py
# AI 台股量化專業平台（無 Plotly / 全功能保留 / 逐路診斷 + Retry / 不需 Secrets）
# ✅ 多來源備援：FinMind(可選手動Token) → yfinance → Stooq → CSV Upload
# ✅ WAF/HTML 偵測、Retry Backoff、逐路診斷表
# ✅ 單股分析：指標共振 + 當下買賣點判斷 + 未來區間(近端/深回檔/近端賣/延伸賣)
# ✅ 布林通道圖（Matplotlib）
# ✅ Top10 掃描器（去重 + 顯示當下可操作判斷與距離）
# ⚠️ 僅供資訊顯示，不構成投資建議，不自動下單

from __future__ import annotations

import io
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

# ----------------------------
# Optional deps
# ----------------------------
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


# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="AI 台股量化專業平台（無 Plotly）", layout="wide")

TITLE = "🧠 AI 台股量化專業平台（無 Plotly / 全功能保留）"
SUBTITLE = "多源備援 + 逐路診斷 + 指標共振 + 布林通道圖 + Top10 掃描 + 交易計畫（不自動下單）"

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class FetchAttempt:
    source: str
    url: str
    result: str  # OK / EMPTY / WAF_HTML / HTTP_xxx / EXC / NO_MODULE / NO_TOKEN / TOO_SHORT
    status_code: Optional[int] = None
    note: str = ""


# ----------------------------
# Utilities
# ----------------------------
def _is_html_like(text: str) -> bool:
    t = (text or "").lstrip().lower()
    if "<html" in t[:300] or "<!doctype html" in t[:300]:
        return True
    # 常見 WAF 關鍵詞
    waf_markers = ["access denied", "forbidden", "cloudflare", "akamai", "incapsula", "waf"]
    return any(m in t[:1200] for m in waf_markers)


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).replace(",", "").replace("--", "").strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: Date, Open, High, Low, Close, Volume
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("Missing Date")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(_safe_float)
    df = df.dropna(subset=["Date", "Close"])
    df = df.set_index("Date")
    # remove zeros
    df = df[df["Close"] > 0]
    return df


def _http_get_retry(url: str, headers: Optional[dict] = None, timeout: int = 15, retries: int = 3, backoff: float = 0.8) -> Tuple[Optional[requests.Response], str]:
    last_err = ""
    h = headers or {}
    # 加 UA 比較不容易被直接擋
    h.setdefault("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36")
    h.setdefault("Accept", "*/*")
    for i in range(retries):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            return r, ""
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff * (2 ** i))
    return None, last_err


# ----------------------------
# Fetchers
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_finmind(code: str, months_back: int, token: str) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if not DEPENDENCIES.get("finmind", False) or DataLoader is None:
        return None, FetchAttempt("FinMind", "", "NO_MODULE", note="未安裝 FinMind（可跳過）")
    if not token:
        return None, FetchAttempt("FinMind", "", "NO_TOKEN", note="未輸入 Token（已跳過 FinMind）")
    try:
        dl = DataLoader()
        dl.login_by_token(api_token=token)

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=months_back * 35 + 30)).strftime("%Y-%m-%d")

        raw = dl.taiwan_stock_daily(stock_id=code, start_date=start, end_date=end)
        if raw is None or len(raw) == 0:
            return None, FetchAttempt("FinMind", f"{code}", "EMPTY", note="empty response")

        df = pd.DataFrame(raw).rename(columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
        })
        df = _normalize_ohlcv(df)
        if len(df) < 30:
            return None, FetchAttempt("FinMind", f"{code}", "TOO_SHORT", note=f"rows={len(df)}")
        return df, FetchAttempt("FinMind", f"{code}", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("FinMind", f"{code}", "EXC", note=str(e))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(code: str, days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    if not DEPENDENCIES.get("yfinance", False) or yf is None:
        return None, FetchAttempt("YF", "", "NO_MODULE", note="未安裝 yfinance")
    try:
        # 台股：先試 .TW 再試 .TWO
        tickers = []
        if code.isdigit() and len(code) <= 6:
            tickers = [f"{code}.TW", f"{code}.TWO"]
        else:
            tickers = [code]

        for t in tickers:
            df = yf.download(t, period=f"{max(days, 60)}d", interval="1d", auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            df = df.reset_index()
            # yfinance 欄位可能是 Date 或 Datetime
            if "Date" not in df.columns and "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df = _normalize_ohlcv(df)
            if len(df) >= 30:
                return df, FetchAttempt("YF", t, "OK", note=f"rows={len(df)}")
            else:
                return None, FetchAttempt("YF", t, "TOO_SHORT", note=f"rows={len(df)}")
        return None, FetchAttempt("YF", ",".join(tickers), "EMPTY", note="no data")
    except Exception as e:
        return None, FetchAttempt("YF", code, "EXC", note=str(e))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq(code: str, days: int) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    # Stooq 台股常見格式：xxxx.tw / xxxx.t
    # 我們用 requests 抓 CSV，比較穩
    try:
        cands = []
        if code.isdigit():
            # stooq 常見：2330.tw
            cands = [f"{code}.tw"]
        else:
            cands = [code]

        for sym in cands:
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            r, err = _http_get_retry(url, retries=3)
            if r is None:
                return None, FetchAttempt("STOOQ", url, "EXC", note=err)
            txt = r.text or ""
            if _is_html_like(txt):
                return None, FetchAttempt("STOOQ", url, "WAF_HTML", status_code=r.status_code, note="HTML/WAF")
            if len(txt.strip()) < 60:
                return None, FetchAttempt("STOOQ", url, "EMPTY", status_code=r.status_code, note="too short")

            df = pd.read_csv(io.StringIO(txt))
            # columns: Date, Open, High, Low, Close, Volume
            if not set(["Date", "Open", "High", "Low", "Close"]).issubset(df.columns):
                return None, FetchAttempt("STOOQ", url, "EMPTY", status_code=r.status_code, note="unexpected columns")
            df = df.tail(max(days, 120) + 30)
            df = _normalize_ohlcv(df)
            if len(df) < 30:
                return None, FetchAttempt("STOOQ", url, "TOO_SHORT", status_code=r.status_code, note=f"rows={len(df)}")
            return df, FetchAttempt("STOOQ", url, "OK", status_code=r.status_code, note=f"rows={len(df)}")
        return None, FetchAttempt("STOOQ", "", "EMPTY", note="no symbol")
    except Exception as e:
        return None, FetchAttempt("STOOQ", "", "EXC", note=str(e))


def _load_csv_fallback(uploaded_file) -> Tuple[Optional[pd.DataFrame], FetchAttempt]:
    try:
        if uploaded_file is None:
            return None, FetchAttempt("CSV_UPLOAD", "", "EMPTY", note="no upload")
        df = pd.read_csv(uploaded_file)
        # 支援常見欄位大小寫
        cols = {c.lower(): c for c in df.columns}
        # 必要欄位
        need = ["date", "open", "high", "low", "close"]
        if not all(n in cols for n in need):
            return None, FetchAttempt("CSV_UPLOAD", "", "EMPTY", note="CSV 欄位需包含 Date/Open/High/Low/Close/Volume(可選)")

        ren = {
            cols["date"]: "Date",
            cols["open"]: "Open",
            cols["high"]: "High",
            cols["low"]: "Low",
            cols["close"]: "Close",
        }
        if "volume" in cols:
            ren[cols["volume"]] = "Volume"
        else:
            df["Volume"] = np.nan
            ren["Volume"] = "Volume"

        df = df.rename(columns=ren)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = _normalize_ohlcv(df)
        if len(df) < 30:
            return None, FetchAttempt("CSV_UPLOAD", "", "TOO_SHORT", note=f"rows={len(df)}（至少 30 根日K）")
        return df, FetchAttempt("CSV_UPLOAD", "", "OK", note=f"rows={len(df)}")
    except Exception as e:
        return None, FetchAttempt("CSV_UPLOAD", "", "EXC", note=str(e))


def fetch_ohlcv_multi(code: str, months_back: int, finmind_token: str, csv_upload=None) -> Tuple[Optional[pd.DataFrame], str, List[FetchAttempt]]:
    days = int(months_back * 31)
    attempts: List[FetchAttempt] = []

    # 1) FinMind（可選）
    df, att = fetch_finmind(code, months_back, finmind_token.strip())
    attempts.append(att)
    if df is not None:
        return df, "FinMind", attempts

    # 2) yfinance
    df, att = fetch_yfinance(code, days)
    attempts.append(att)
    if df is not None:
        return df, "YF", attempts

    # 3) Stooq
    df, att = fetch_stooq(code, days)
    attempts.append(att)
    if df is not None:
        return df, "STOOQ", attempts

    # 4) CSV upload
    if csv_upload is not None:
        df, att = _load_csv_fallback(csv_upload)
        attempts.append(att)
        if df is not None:
            return df, "CSV_UPLOAD", attempts

    return None, "NONE", attempts


# ----------------------------
# Indicators & Signals
# ----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std
    lo = mid - k * std
    z = (close - mid) / std.replace(0, np.nan)
    return mid, up, lo, z

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA60"] = ema(out["Close"], 60)
    out["RSI14"] = rsi(out["Close"], 14)
    m_line, m_sig, m_hist = macd(out["Close"])
    out["MACD"] = m_line
    out["MACD_SIG"] = m_sig
    out["MACD_HIST"] = m_hist
    mid, up, lo, z = bollinger(out["Close"], 20, 2.0)
    out["BB_MID"] = mid
    out["BB_UP"] = up
    out["BB_LO"] = lo
    out["BB_Z"] = z
    out["ATR14"] = atr(out, 14)
    out["VOL_MA20"] = out["Volume"].rolling(20).mean()
    return out

def classify_action(latest: pd.Series, max_buy_distance: float = 0.12) -> Tuple[str, str]:
    """
    回傳 (action, reason)
    action: BUY / SELL / WATCH
    """
    px = float(latest["Close"])
    ema20 = float(latest.get("EMA20", np.nan))
    ema60 = float(latest.get("EMA60", np.nan))
    r = float(latest.get("RSI14", np.nan))
    hist = float(latest.get("MACD_HIST", np.nan))
    bbz = float(latest.get("BB_Z", np.nan))
    bblo = float(latest.get("BB_LO", np.nan))
    bbup = float(latest.get("BB_UP", np.nan))

    # 近端買點：靠近下軌或 -1.5σ以下 + RSI偏低 + MACD動能回升
    near_buy = (px <= bblo * (1 + max_buy_distance)) and (bbz <= -1.2) and (r <= 45)
    # 近端賣點：靠近上軌或 +1.5σ以上 + RSI偏高 + MACD動能轉弱
    near_sell = (px >= bbup * (1 - 0.02)) and (bbz >= 1.2) and (r >= 55)

    # 趨勢濾網
    trend_up = (not math.isnan(ema20)) and (not math.isnan(ema60)) and (ema20 >= ema60)
    trend_dn = (not math.isnan(ema20)) and (not math.isnan(ema60)) and (ema20 < ema60)

    if near_buy and trend_up and hist > 0:
        return "BUY", "趨勢偏多，價格落在布林下緣/負σ區，且動能回升（MACD_HIST>0）"
    if near_sell and trend_dn and hist < 0:
        return "SELL", "趨勢偏空，價格貼近布林上緣/正σ區，且動能轉弱（MACD_HIST<0）"

    # 若接近買點但未觸發：WATCH
    if px <= (bblo * (1 + max_buy_distance)):
        return "WATCH", "接近下緣區但條件未完整共振（等待 RSI/MACD 確認）"
    if px >= (bbup * (1 - 0.02)):
        return "WATCH", "接近上緣區但條件未完整共振（等待 RSI/MACD 確認）"

    return "WATCH", "條件尚未明確，等待價格進入區間或指標翻轉"

def make_future_zones(latest: pd.Series, max_buy_distance: float = 0.12) -> Dict[str, Dict[str, float]]:
    """
    用布林通道 + σ 概念給出「近端/深回檔」買點、以及「近端/延伸」賣點區間
    """
    mid = float(latest.get("BB_MID", np.nan))
    up = float(latest.get("BB_UP", np.nan))
    lo = float(latest.get("BB_LO", np.nan))
    std = (up - mid) / 2.0 if (not math.isnan(up) and not math.isnan(mid)) else np.nan
    px = float(latest["Close"])

    # 區間設計：更貼近實務（避免離現價太遠）
    # 近端買：下軌到下軌+0.5σ
    near_buy_lo = lo
    near_buy_hi = lo + 0.5 * std if not math.isnan(std) else lo * (1 + 0.02)

    # 深回檔買：下軌 -0.5σ 到 下軌（更深）
    deep_buy_lo = lo - 0.6 * std if not math.isnan(std) else lo * (1 - 0.05)
    deep_buy_hi = lo

    # 近端賣：上軌-0.5σ 到 上軌
    near_sell_lo = up - 0.5 * std if not math.isnan(std) else up * (1 - 0.02)
    near_sell_hi = up

    # 延伸賣：上軌到上軌+0.7σ
    ext_sell_lo = up
    ext_sell_hi = up + 0.7 * std if not math.isnan(std) else up * (1 + 0.03)

    def dist_pct(a: float) -> float:
        return (a / px - 1.0) * 100.0

    zones = {
        "near_buy": {"lo": near_buy_lo, "hi": near_buy_hi, "dist_lo_pct": dist_pct(near_buy_lo), "dist_hi_pct": dist_pct(near_buy_hi)},
        "deep_buy": {"lo": deep_buy_lo, "hi": deep_buy_hi, "dist_lo_pct": dist_pct(deep_buy_lo), "dist_hi_pct": dist_pct(deep_buy_hi)},
        "near_sell": {"lo": near_sell_lo, "hi": near_sell_hi, "dist_lo_pct": dist_pct(near_sell_lo), "dist_hi_pct": dist_pct(near_sell_hi)},
        "ext_sell": {"lo": ext_sell_lo, "hi": ext_sell_hi, "dist_lo_pct": dist_pct(ext_sell_lo), "dist_hi_pct": dist_pct(ext_sell_hi)},
    }
    return zones


def compute_ai_score(latest: pd.Series) -> int:
    score = 50
    r = float(latest.get("RSI14", np.nan))
    hist = float(latest.get("MACD_HIST", np.nan))
    z = float(latest.get("BB_Z", np.nan))
    ema20 = float(latest.get("EMA20", np.nan))
    ema60 = float(latest.get("EMA60", np.nan))
    vol = float(latest.get("Volume", np.nan))
    vma = float(latest.get("VOL_MA20", np.nan))

    if not math.isnan(ema20) and not math.isnan(ema60):
        score += 10 if ema20 >= ema60 else -10
    if not math.isnan(hist):
        score += 8 if hist > 0 else -8
    if not math.isnan(r):
        if r < 35:
            score += 6
        elif r > 70:
            score -= 6
    if not math.isnan(z):
        if z < -1.5:
            score += 6
        elif z > 1.5:
            score -= 6
    if not math.isnan(vol) and not math.isnan(vma) and vma > 0:
        score += 6 if (vol / vma) > 1.2 else 0

    return int(max(0, min(100, score)))


def plot_bollinger(df: pd.DataFrame, code: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["BB_MID"], label="BB Mid")
    ax.plot(df.index, df["BB_UP"], label="BB Upper")
    ax.plot(df.index, df["BB_LO"], label="BB Lower")
    ax.fill_between(df.index, df["BB_LO"].values, df["BB_UP"].values, alpha=0.1)
    ax.set_title(f"{code} - Bollinger Bands (20,2)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.markdown("## 設定")
    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"], index=0)

    months_label = st.selectbox("資料期間", ["3mo", "6mo", "12mo", "24mo"], index=1)
    months_back = {"3mo": 3, "6mo": 6, "12mo": 12, "24mo": 24}[months_label]

    show_debug = st.checkbox("顯示下載除錯資訊（Debug）", value=False)

    st.markdown("---")
    st.markdown("## ⚙️ 系統狀態 / 套件")
    for pkg, ok in DEPENDENCIES.items():
        if ok:
            st.success(f"✅ {pkg}")
        else:
            st.error(f"❌ {pkg} 缺失（見下方 requirements.txt）")

    st.markdown("---")
    st.markdown("## 🔑 FinMind（可選）")
    finmind_token = st.text_input("FinMind Token（可留空，不影響使用）", type="password", value="")
    if DEPENDENCIES.get("finmind", False) and finmind_token.strip() == "":
        st.info("FinMind 已安裝，但你未輸入 Token → 會自動跳過 FinMind，改用 yfinance / Stooq / CSV")
    if (not DEPENDENCIES.get("finmind", False)) and finmind_token.strip():
        st.warning("你輸入了 Token，但 FinMind 未安裝 → 請在 requirements.txt 加入 FinMind")

    st.markdown("---")
    st.markdown("## A+B+C：專業買賣點 / 風險報酬")
    max_buy_distance = st.slider("可操作買點最大距離（避免買點離現實太遠）max_buy_distance", 0.02, 0.20, 0.12, 0.01)

    st.markdown("---")
    net_test = st.button("🧪 網路測試（建議先按一次）")

# ----------------------------
# Main UI
# ----------------------------
st.markdown(f"# {TITLE}")
st.caption(SUBTITLE)

colA, colB = st.columns([2, 3])

with colA:
    code = st.text_input("請輸入股票代號", value="2330" if mode == "Top 10 掃描器" else "6274").strip()

with colB:
    st.info("💡 若 Cloud 偶發抓不到資料：\n- 先按左側「網路測試」\n- 或直接上傳 export.csv（Date/Open/High/Low/Close/Volume）立即可用")

csv_upload = st.file_uploader("（選用）上傳 export.csv 作為備援資料源", type=["csv"])

if net_test:
    test_urls = [
        "https://stooq.com",
        "https://query1.finance.yahoo.com/v7/finance/quote?symbols=2330.TW",
    ]
    rows = []
    for u in test_urls:
        r, err = _http_get_retry(u, retries=2, timeout=10)
        if r is None:
            rows.append([u, "FAIL", "", err[:120]])
        else:
            rows.append([u, "OK", r.status_code, (r.text[:60].replace("\n", " ") if r.text else "")])
    st.subheader("🧪 網路測試結果")
    st.dataframe(pd.DataFrame(rows, columns=["url", "result", "status", "snippet"]))

# ----------------------------
# Single Stock Analysis
# ----------------------------
def render_single(code: str):
    df, src, attempts = fetch_ohlcv_multi(code, months_back, finmind_token, csv_upload)

    if df is None:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試，或改用 CSV 上傳備援。")
        if show_debug:
            st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
            st.dataframe(pd.DataFrame([a.__dict__ for a in attempts]))
        return

    df = compute_indicators(df)
    latest = df.iloc[-1]
    px = float(latest["Close"])
    score = compute_ai_score(latest)
    action, reason = classify_action(latest, max_buy_distance=max_buy_distance)
    zones = make_future_zones(latest, max_buy_distance=max_buy_distance)

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目前價格", f"{px:.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", src)
    c4.metric("最後日期 / 筆數", f"{df.index[-1].date()} / {len(df)}")

    st.markdown("## 📌 當下是否為買點/賣點？（可操作判斷）")
    if action == "BUY":
        st.success(f"🟢 BUY：{reason}")
    elif action == "SELL":
        st.error(f"🔴 SELL：{reason}")
    else:
        st.info(f"🟡 WATCH：{reason}")

    # Professional distance description
    st.markdown("## 🗺️ 未來預估買賣點（區間 + 距離 + σ 語彙）")

    def zone_card(title: str, z: Dict[str, float], color: str, note: str):
        lo, hi = z["lo"], z["hi"]
        dlo, dhi = z["dist_lo_pct"], z["dist_hi_pct"]
        st.markdown(
            f"""
<div style="padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,0.08);background:{color};">
<b>{title}</b><br/>
區間： <b>{lo:.2f} ~ {hi:.2f}</b><br/>
相對現價： <b>{dlo:+.1f}% ~ {dhi:+.1f}%</b>（以現價為基準的位移）<br/>
說明： {note}
</div>
""",
            unsafe_allow_html=True,
        )

    zone_card(
        "🟢 近端買點（可操作）",
        zones["near_buy"],
        "rgba(0,120,60,0.25)",
        "偏向「回檔承接」：靠近布林下緣與負σ區，屬於較貼近現價的進場帶。"
    )
    zone_card(
        "🟦 深回檔買點（等待型）",
        zones["deep_buy"],
        "rgba(0,120,255,0.18)",
        "偏向「極端回檔」：需更深回落才出現，通常搭配恐慌/急跌情境，較不常觸發。"
    )
    zone_card(
        "🟡 近端賣點（壓力/獲利）",
        zones["near_sell"],
        "rgba(255,180,0,0.18)",
        "偏向「壓力帶」：接近布林上緣與正σ區，常見獲利了結或遇壓震盪。"
    )
    zone_card(
        "🔴 延伸賣點（突破延伸）",
        zones["ext_sell"],
        "rgba(255,0,0,0.12)",
        "偏向「趨勢延伸」：若放量突破上緣，可能走延伸段，此區間偏向追蹤停利/分批減碼。"
    )

    st.markdown("## 📈 布林通道走勢圖（專業視覺判讀）")
    plot_bollinger(df, code)

    if show_debug:
        st.markdown("## 🧩 逐路診斷（哪一路成功/失敗）")
        st.dataframe(pd.DataFrame([a.__dict__ for a in attempts]))

    st.markdown("## 📊 指標摘要（最新一筆）")
    show_cols = ["Close", "EMA20", "EMA60", "RSI14", "MACD", "MACD_SIG", "MACD_HIST", "BB_MID", "BB_UP", "BB_LO", "BB_Z", "ATR14"]
    st.dataframe(pd.DataFrame(latest[show_cols]).T)


# ----------------------------
# Top10 Scanner
# ----------------------------
DEFAULT_POOL = [
    # 大型與常用測試池（可自行增減）
    "2330", "2317", "2454", "2308", "2412", "6505", "2881", "2882", "2891",
    "2603", "2609", "2615", "3037", "2382", "2303", "3711", "2327",
    "3008", "2002", "1301", "1303", "1101", "1216", "5880", "2886",
    "8046", "4967", "6274", "9910", "3231"
]

def render_top10():
    st.markdown("## 🔥 AI 強勢股 Top 10（去重 / 顯示可操作判斷）")
    st.caption("Top10 建議先用小池測試（避免 Cloud 超時）。你也可以在下方貼上自訂股票清單。")

    custom = st.text_area("（選用）自訂股票清單（用逗號或換行分隔，例如：2330,2317,2454）", value="")
    if custom.strip():
        raw = custom.replace(",", "\n").splitlines()
        pool = [x.strip() for x in raw if x.strip()]
    else:
        pool = DEFAULT_POOL

    # 去重 + 只留數字代號
    uniq = []
    seen = set()
    for x in pool:
        x = x.strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)

    limit = st.slider("掃描數量上限（避免超時）", 10, 60, 25, 5)

    rows = []
    diag_rows = []

    for code in uniq[:limit]:
        df, src, attempts = fetch_ohlcv_multi(code, months_back, finmind_token, csv_upload)
        if df is None:
            diag_rows.append({"code": code, "ok": False, "src": "NONE", "note": "all failed"})
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]
        px = float(latest["Close"])
        score = compute_ai_score(latest)
        action, reason = classify_action(latest, max_buy_distance=max_buy_distance)
        zones = make_future_zones(latest, max_buy_distance=max_buy_distance)

        # 顯示「最貼近的可操作帶」：近端買 or 近端賣
        nb = zones["near_buy"]
        ns = zones["near_sell"]
        nb_mid = (nb["lo"] + nb["hi"]) / 2
        ns_mid = (ns["lo"] + ns["hi"]) / 2
        nb_dist = abs((nb_mid / px) - 1)
        ns_dist = abs((ns_mid / px) - 1)
        closest = "NearBuy" if nb_dist <= ns_dist else "NearSell"
        closest_band = nb if closest == "NearBuy" else ns

        rows.append({
            "股票": code,
            "來源": src,
            "AI分數": score,
            "目前價": round(px, 2),
            "當下判斷": action,
            "可操作帶": "近端買" if closest == "NearBuy" else "近端賣",
            "帶區間": f"{closest_band['lo']:.2f} ~ {closest_band['hi']:.2f}",
            "相對現價": f"{closest_band['dist_lo_pct']:+.1f}% ~ {closest_band['dist_hi_pct']:+.1f}%",
        })
        diag_rows.append({"code": code, "ok": True, "src": src, "note": f"rows={len(df)}"})

    if not rows:
        st.error("沒有任何股票成功取得資料。建議：先按「網路測試」或改用 CSV 上傳備援。")
        if show_debug and diag_rows:
            st.dataframe(pd.DataFrame(diag_rows))
        return

    out = pd.DataFrame(rows).sort_values(["AI分數", "當下判斷"], ascending=[False, True]).head(10)
    st.dataframe(out, use_container_width=True)

    st.markdown("### ✅ Top10 補充：為什麼是這些？")
    st.write("排序以 **AI分數（趨勢+動能+波動位置+量能）** 為主，並顯示最貼近現價的「可操作帶」。")

    if show_debug:
        st.markdown("### 🧩 掃描診斷（成功/失敗概況）")
        st.dataframe(pd.DataFrame(diag_rows))


# ----------------------------
# Run
# ----------------------------
if mode == "單一股票分析":
    render_single(code)
else:
    render_top10()

st.markdown("---")
st.caption("⚠️ 本工具僅供資訊顯示與風險控管演算，不構成投資建議，亦不會自動下單。")