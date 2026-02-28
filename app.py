# app.py
# V7 短波段突破雷達（80+ 強動能 / 自動掃描 20 檔）
# ✅ FinMind 優先 + yfinance fallback
# ✅ MultiIndex 修正 / 資料清洗 / 每檔 try-continue（不再整頁爆）
# ✅ 80+ 主榜 / 60~79 觀察區
# ✅ 市場風險燈號（0050）
# ✅ 資金控管（依 Top1 強突破自動算張數）
#
# ⚠️ 僅做資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="V7 短波段突破雷達（80+ 強動能）", layout="wide")
st.title("⚡ V7 短波段突破雷達（80+ 強動能）")

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]

LOOKBACK_DAYS = 220          # 抓 220 天（足夠算 60/80 天指標）
CACHE_TTL_SEC = 600          # 10 分鐘快取
SCORE_STRONG = 80
SCORE_WATCH = 60

# -----------------------------
# UI - settings
# -----------------------------
with st.expander("⚙️ 設定（股票池 / 快取 / 門檻）", expanded=False):
    watchlist_text = st.text_area(
        "掃描股票池（用逗號或換行分隔，預設 20 檔）",
        value="\n".join(DEFAULT_WATCHLIST),
        height=160
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        score_strong = st.number_input("強突破門檻（主榜）", value=int(SCORE_STRONG), min_value=60, max_value=100, step=1)
    with col2:
        score_watch = st.number_input("觀察門檻（次強）", value=int(SCORE_WATCH), min_value=0, max_value=int(score_strong), step=1)
    with col3:
        cache_ttl = st.number_input("快取秒數（Cloud 建議 600）", value=int(CACHE_TTL_SEC), min_value=60, max_value=3600, step=60)

WATCHLIST = []
for token in watchlist_text.replace(",", "\n").splitlines():
    t = token.strip().upper()
    if t:
        WATCHLIST.append(t)

if len(WATCHLIST) == 0:
    WATCHLIST = DEFAULT_WATCHLIST.copy()

# -----------------------------
# Helpers
# -----------------------------
def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance 偶爾回 MultiIndex：('Close','2330.TW')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: Date, Open, High, Low, Close, Volume (clean)."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # reset index to Date
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()

    # normalize column names
    # yfinance: Date index + Open/High/Low/Close/Adj Close/Volume
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = _safe_float_series(df[c])

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    df = df.sort_values("Date").reset_index(drop=True)
    return df

def _make_yf_candidates(code: str) -> list[str]:
    code = (code or "").strip().upper()
    if not code:
        return []
    if "." in code:
        return [code]
    if code.isdigit():
        return [f"{code}.TW", f"{code}.TWO", code]
    return [code]

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data(ttl=CACHE_TTL_SEC)
def load_from_finmind(code_digits: str, start: str, end: str) -> pd.DataFrame:
    try:
        from FinMind.data import DataLoader
    except Exception:
        return pd.DataFrame()

    try:
        dl = DataLoader()
        raw = dl.taiwan_stock_daily(stock_id=code_digits, start_date=start, end_date=end)
        if raw is None or raw.empty:
            return pd.DataFrame()

        # FinMind columns: date, open, max, min, close, Trading_Volume...
        df = pd.DataFrame({
            "Date": pd.to_datetime(raw["date"], errors="coerce"),
            "Open": raw["open"],
            "High": raw.get("max", raw.get("high", np.nan)),
            "Low": raw.get("min", raw.get("low", np.nan)),
            "Close": raw["close"],
            "Volume": raw.get("Trading_Volume", raw.get("volume", 0)),
        })
        return _ensure_ohlcv(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_SEC)
def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df)
    df = _ensure_ohlcv(df)
    return df

def load_data_best_effort(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, source_used)
    source_used: 'FinMind' or 'YF' or ''
    """
    code = (code or "").strip().upper()
    if not code:
        return pd.DataFrame(), ""

    # 1) FinMind if digits
    if code.isdigit():
        df_fm = load_from_finmind(code, start, end)
        if not df_fm.empty:
            return df_fm, "FinMind"

    # 2) yfinance candidates
    for t in _make_yf_candidates(code):
        df_yf = load_from_yfinance(t, start, end)
        if not df_yf.empty:
            return df_yf, f"YF({t})"

    return pd.DataFrame(), ""

# -----------------------------
# Breakout score engine (突破型短波段)
# -----------------------------
def breakout_score(df_ohlcv: pd.DataFrame) -> tuple[int, dict | None]:
    """
    Score:
      +20 趨勢：Close>EMA20 且 EMA20>EMA60
      +20 突破：Close > 前 5 日高
      +20 量能：Volume > 1.5 * 5日均量
      +20 RSI：RSI > 55
      +20 MACD柱：MACD diff > 0
    Trade:
      entry = Close
      stop  = entry - ATR*1.2
      target= entry + (entry-stop)*2.2
    """
    if df_ohlcv is None or df_ohlcv.empty:
        return 0, None

    # 僅取必要欄位 & 清洗
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df_ohlcv.columns:
            return 0, None

    df = df_ohlcv.copy()
    df = df.dropna(subset=needed)
    for c in needed:
        df[c] = _safe_float_series(df[c])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # 短波段突破：我們保守要求資料更長，避免指標 NaN
    if len(df) < 80:
        return 0, None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    try:
        ema20 = EMAIndicator(close, 20).ema_indicator()
        ema60 = EMAIndicator(close, 60).ema_indicator()
        rsi = RSIIndicator(close, 14).rsi()
        macd_h = MACD(close).macd_diff()
        atr = AverageTrueRange(high, low, close, 14).average_true_range()
    except Exception:
        return 0, None

    df["EMA20"] = ema20
    df["EMA60"] = ema60
    df["RSI"] = rsi
    df["MACD_H"] = macd_h
    df["ATR"] = atr

    df = df.dropna()
    if len(df) < 80:
        return 0, None

    latest = df.iloc[-1]
    prev5_high = df["High"].iloc[-6:-1].max()
    vol_mean = df["Volume"].iloc[-6:-1].mean()

    score = 0
    # 趨勢
    if latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA60"]:
        score += 20
    # 突破 5 日高
    if latest["Close"] > prev5_high:
        score += 20
    # 量能放大
    if vol_mean > 0 and latest["Volume"] > 1.5 * vol_mean:
        score += 20
    # RSI
    if latest["RSI"] > 55:
        score += 20
    # MACD 柱翻正
    if latest["MACD_H"] > 0:
        score += 20

    entry = float(latest["Close"])
    atr_v = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else np.nan
    if not np.isfinite(entry) or not np.isfinite(atr_v) or atr_v <= 0:
        return score, None

    stop = float(entry - atr_v * 1.2)

    # 防呆：避免 entry<=stop 或 stop 非數字
    if not np.isfinite(stop) or entry <= stop:
        return score, None

    target = float(entry + (entry - stop) * 2.2)
    rr = (target - entry) / (entry - stop)

    trade = {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(float(rr), 2),
    }
    return score, trade

# -----------------------------
# Market risk (0050)
# -----------------------------
def market_risk_label(start: str, end: str) -> str:
    df, src = load_data_best_effort("0050", start, end)
    if df.empty:
        return "🟡 市場資料不足（0050 無法取得）"

    close = df["Close"].astype(float)
    if len(close) < 40:
        return "🟡 市場資料不足（樣本不足）"

    try:
        ema20 = EMAIndicator(close, 20).ema_indicator()
    except Exception:
        return "🟡 市場資料不足（指標失敗）"

    if close.iloc[-1] > ema20.iloc[-1]:
        return f"🟢 市場可積極操作（來源：{src}）"
    return f"🔴 市場偏弱，控制倉位（來源：{src}）"

# -----------------------------
# Run Scan (auto)
# -----------------------------
today = dt.date.today()
end_dt = today + dt.timedelta(days=1)  # include today
start_dt = today - dt.timedelta(days=LOOKBACK_DAYS)
start = start_dt.strftime("%Y-%m-%d")
end = end_dt.strftime("%Y-%m-%d")

st.caption(f"掃描範圍：{start} → {today.strftime('%Y-%m-%d')} | 股票池：{len(WATCHLIST)} 檔 | 快取：{cache_ttl}s")

# Auto scan with progress
progress = st.progress(0)
status = st.empty()

results = []
diag_rows = []

for i, code in enumerate(WATCHLIST):
    status.write(f"掃描中：{code} ({i+1}/{len(WATCHLIST)})")
    df, src = load_data_best_effort(code, start, end)

    if df.empty:
        diag_rows.append({"股票": code, "來源": src or "-", "狀態": "EMPTY", "備註": "無資料"})
        progress.progress(int((i + 1) / len(WATCHLIST) * 100))
        continue

    # 每檔都保護：算不了就跳過，絕不拖垮整頁
    score, trade = breakout_score(df)
    if trade is None:
        diag_rows.append({"股票": code, "來源": src, "狀態": "SKIP", "備註": f"指標/資料不足（score={score}）"})
        progress.progress(int((i + 1) / len(WATCHLIST) * 100))
        continue

    diag_rows.append({"股票": code, "來源": src, "狀態": "OK", "備註": f"score={score}"})

    if score >= int(score_watch):
        results.append({
            "股票": code,
            "分數": int(score),
            "來源": src,
            "進場": trade["entry"],
            "停損": trade["stop"],
            "目標": trade["target"],
            "RR": trade["rr"],
        })

    progress.progress(int((i + 1) / len(WATCHLIST) * 100))

progress.empty()
status.empty()

# -----------------------------
# Display results
# -----------------------------
df_results = pd.DataFrame(results).sort_values("分數", ascending=False) if results else pd.DataFrame(columns=["股票","分數","來源","進場","停損","目標","RR"])

left, right = st.columns([1.35, 1])

with left:
    st.subheader(f"⚡ 今日突破榜（{score_strong}+）")
    strong = df_results[df_results["分數"] >= int(score_strong)].copy()
    if strong.empty:
        st.info("今日沒有達到主榜門檻的突破型機會（80+）。")
    else:
        st.dataframe(strong, use_container_width=True)

    st.subheader(f"🟡 次強觀察區（{score_watch}~{int(score_strong)-1}）")
    watch = df_results[(df_results["分數"] >= int(score_watch)) & (df_results["分數"] < int(score_strong))].copy()
    if watch.empty:
        st.caption("無次強觀察標的。")
    else:
        st.dataframe(watch, use_container_width=True)

with right:
    st.subheader("📊 市場風險燈號")
    st.write(market_risk_label(start, end))

    st.subheader("💰 資金控管（以主榜 Top1 計算）")
    capital = st.number_input("總資金", value=5_000_000, step=100_000)
    risk_pct = st.number_input("單筆風險 %", value=2.0, step=0.5)

    if not strong.empty:
        top1 = strong.iloc[0]
        risk_amount = float(capital) * (float(risk_pct) / 100.0)

        loss_per_share = float(top1["進場"]) - float(top1["停損"])
        if loss_per_share <= 0:
            st.warning("Top1 停損距離異常，無法計算張數。")
        else:
            shares = int(risk_amount / loss_per_share)  # 這裡的 shares 是「股數」概念（你可自行換算張）
            lots = max(int(shares // 1000), 0)          # 台股 1 張 = 1000 股（簡化）
            st.write(f"建議標的：**{top1['股票']}**（分數 {top1['分數']}，來源 {top1['來源']}）")
            st.write(f"單筆風險金額：**{risk_amount:,.0f}**")
            st.write(f"停損距離：**{loss_per_share:.2f}**")
            st.write(f"可買股數：約 **{shares:,} 股**")
            st.write(f"可買張數：約 **{lots} 張**")
            st.caption("（張數以 1000 股/張簡化估算；實務請以券商規則與可交易股數為準）")
    else:
        st.caption("主榜沒有標的時，不計算張數（避免硬做）。")

# -----------------------------
# Diagnostics
# -----------------------------
with st.expander("🧩 逐檔診斷（哪一檔被跳過、原因）", expanded=False):
    if diag_rows:
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
    else:
        st.caption("無診斷資料。")

st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")