# app.py
# AI Stock Trading Assistant（TW）- V9.1 FIX
# ✅ 單股代號統一（不再出現 2330/3017 混在一起）
# ✅ 單股頁面：布林通道分析 + 當下買/賣/等 + 預估未來買賣點（清楚可判斷）
# ✅ 雷達：主榜/觀察 + 逐檔診斷收起
# ✅ FinMind 優先 + yfinance fallback（含 MultiIndex 修正）
# ⚠️ 僅供資訊顯示與風控演算，不構成投資建議、不自動下單

from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant (TW) - V9.1", layout="wide")
st.title("⚡ AI Stock Trading Assistant（V9.1：雷達 + 單股決策 / 盤後型）")
st.caption("BUILD: V9.1-SINGLE-CONSISTENT-BB-POINTS-20260228")

DEFAULT_WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]

# -----------------------------
# Session state：唯一的「單股分析代號」
# -----------------------------
if "analysis_code" not in st.session_state:
    st.session_state["analysis_code"] = "2330"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🎯 單一股票決策（盤後）")

# 用 key 綁定 session_state，確保全站只有一個代號
st.sidebar.text_input("股票代號", key="analysis_code")
single_lookback = st.sidebar.selectbox("回溯天數", [180, 365, 730], index=1)
signal_mode = st.sidebar.checkbox("訊號機模式（更乾淨）", value=True)
run_single = st.sidebar.button("分析單股", type="primary")

st.sidebar.divider()
st.sidebar.header("⚡ 市場雷達設定")
watchlist_text = st.sidebar.text_area(
    "掃描股票池（逗號或換行分隔）",
    value="\n".join(DEFAULT_WATCHLIST),
    height=180
)
c1, c2 = st.sidebar.columns(2)
with c1:
    score_strong = st.sidebar.number_input("主榜門檻", value=80, min_value=60, max_value=100, step=1)
with c2:
    score_watch = st.sidebar.number_input("觀察門檻", value=60, min_value=0, max_value=int(score_strong), step=1)

cache_ttl = st.sidebar.number_input("快取秒數（Cloud 建議 600）", value=600, min_value=60, max_value=3600, step=60)
radar_lookback = st.sidebar.number_input("雷達回溯天數", value=240, min_value=120, max_value=2000, step=10)

WATCHLIST = []
for token in watchlist_text.replace(",", "\n").splitlines():
    t = token.strip().upper()
    if t:
        WATCHLIST.append(t)
if not WATCHLIST:
    WATCHLIST = DEFAULT_WATCHLIST.copy()

# -----------------------------
# Helpers
# -----------------------------
def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()
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
    return df.sort_values("Date").reset_index(drop=True)

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
# Data loaders (FinMind -> YF)
# -----------------------------
@st.cache_data(ttl=cache_ttl)
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
        df = pd.DataFrame({
            "Date": pd.to_datetime(raw["date"], errors="coerce"),
            "Open": raw["open"],
            "High": raw.get("max", np.nan),
            "Low": raw.get("min", np.nan),
            "Close": raw["close"],
            "Volume": raw.get("Trading_Volume", raw.get("volume", 0)),
        })
        return _ensure_ohlcv(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=cache_ttl)
def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=False,
            group_by="column", threads=False
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _flatten_yf_columns(df)
    return _ensure_ohlcv(df)

def load_data_best_effort(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    code = (code or "").strip().upper()
    if not code:
        return pd.DataFrame(), ""

    if code.isdigit():
        df_fm = load_from_finmind(code, start, end)
        if not df_fm.empty:
            return df_fm, "FinMind"

    for t in _make_yf_candidates(code):
        df_yf = load_from_yfinance(t, start, end)
        if not df_yf.empty:
            return df_yf, f"YF({t})"

    return pd.DataFrame(), ""

# -----------------------------
# Indicators (EMA/RSI/MACD/ATR + Bollinger)
# -----------------------------
def add_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    if df_ohlcv is None or df_ohlcv.empty:
        return pd.DataFrame()

    df = df_ohlcv.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = _safe_float_series(df[c])
    df = df.dropna(subset=["Open","High","Low","Close"]).reset_index(drop=True)

    if len(df) < 80:
        return pd.DataFrame()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    try:
        df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA60"] = EMAIndicator(close, 60).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()
        df["MACD_H"] = MACD(close).macd_diff()
        df["ATR"] = AverageTrueRange(high, low, close, 14).average_true_range()
    except Exception:
        return pd.DataFrame()

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    return df.dropna().reset_index(drop=True)

# -----------------------------
# Radar score (突破短波段)
# -----------------------------
def breakout_score(df_ohlcv: pd.DataFrame) -> tuple[int, dict | None]:
    df = add_indicators(df_ohlcv)
    if df.empty:
        return 0, None

    latest = df.iloc[-1]
    prev5_high = df["High"].iloc[-6:-1].max()
    vol_mean = df["Volume"].iloc[-6:-1].mean()

    score = 0
    if latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA60"]:
        score += 20
    if latest["Close"] > prev5_high:
        score += 20
    if vol_mean > 0 and latest["Volume"] > 1.5 * vol_mean:
        score += 20
    if latest["RSI"] > 55:
        score += 20
    if latest["MACD_H"] > 0:
        score += 20

    entry = float(latest["Close"])
    atr = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else np.nan
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return score, None

    stop = float(entry - 1.2 * atr)
    if entry <= stop:
        return score, None

    target = float(entry + (entry - stop) * 2.2)
    rr = (target - entry) / (entry - stop)

    return score, {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(float(rr), 2),
    }

# -----------------------------
# Single-stock decision (清楚判斷 + 點位)
# -----------------------------
def single_decision(df_i: pd.DataFrame) -> dict:
    latest = df_i.iloc[-1]
    prev5_high = df_i["High"].iloc[-6:-1].max()
    vol_mean = df_i["Volume"].iloc[-6:-1].mean()

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema60 = float(latest["EMA60"])
    rsi = float(latest["RSI"])
    macd_h = float(latest["MACD_H"])
    atr = float(latest["ATR"])
    bb_up = float(latest["BB_UP"])
    bb_mid = float(latest["BB_MID"])
    bb_low = float(latest["BB_LOW"])
    vol = float(latest["Volume"])

    # 布林位置
    if price >= bb_up:
        boll_pos = "上軌附近（偏過熱/動能強）"
    elif price <= bb_low:
        boll_pos = "下軌附近（偏超跌/反彈區）"
    elif price >= bb_mid:
        boll_pos = "中軌之上（偏多）"
    else:
        boll_pos = "中軌之下（偏弱）"

    # 布林壓縮/擴張（分位數）
    widths = ((df_i["BB_UP"] - df_i["BB_LOW"]) / df_i["BB_MID"]).replace([np.inf, -np.inf], np.nan).dropna()
    width = (bb_up - bb_low) / bb_mid if bb_mid else np.nan
    boll_state = "不足以判斷"
    if len(widths) >= 60 and np.isfinite(width):
        q20 = float(widths.quantile(0.2))
        q80 = float(widths.quantile(0.8))
        if width <= q20:
            boll_state = "壓縮（可能醞釀爆發）"
        elif width >= q80:
            boll_state = "擴張（趨勢段/波動放大）"
        else:
            boll_state = "正常"

    # 規則（盤後）
    trend_ok = (price > ema20 and ema20 > ema60)
    breakout_ok = (price > float(prev5_high))
    vol_ok = (vol_mean > 0 and vol > 1.3 * vol_mean)
    macd_ok = (macd_h > 0)
    rsi_ok = (rsi > 55)

    # 預估點位（你要的）
    buy_pull_low = round(ema20 * 0.99, 2)
    buy_pull_high = round(ema20 * 1.01, 2)
    buy_break = round(float(prev5_high) + 0.2 * atr, 2)

    stop = round(price - 1.2 * atr, 2)
    target_rr = round(price + (price - stop) * 2.2, 2)
    target_bb = round(bb_up, 2)

    # 一眼結論
    if trend_ok and breakout_ok and rsi_ok and (vol_ok or macd_ok):
        decision = "BUY"
        decision_text = "🟢 BUY（盤後確認突破）"
    elif (price < ema20) and (macd_h < 0):
        decision = "SELL"
        decision_text = "🔴 SELL / 減碼（跌破 EMA20 + 動能轉弱）"
    else:
        decision = "WAIT"
        decision_text = "🟡 WAIT（等回踩買區或再突破確認）"

    reasons = [
        f"趨勢：{'OK' if trend_ok else '未確認'}（Close/EMA20/EMA60）",
        f"突破：{'OK' if breakout_ok else '未突破'}（前5高）",
        f"量能：{'OK' if vol_ok else '未放大'}（>1.3×5日均量）",
        f"MACD：{'OK' if macd_ok else '偏弱'}（柱狀）",
        f"RSI：{'OK' if rsi_ok else '偏弱'}（>55）",
    ]

    return {
        "decision": decision,
        "decision_text": decision_text,
        "price": round(price, 2),
        "buy_pull": (buy_pull_low, buy_pull_high),
        "buy_break": buy_break,
        "stop": stop,
        "target_rr": target_rr,
        "target_bb": target_bb,
        "boll_pos": boll_pos,
        "boll_state": boll_state,
        "ema20": round(ema20, 2),
        "ema60": round(ema60, 2),
        "rsi": round(rsi, 1),
        "macd_h": round(macd_h, 4),
        "atr": round(atr, 2),
        "reasons": reasons,
    }

def plot_single_chart(df_i: pd.DataFrame, info: dict, title: str):
    plot_df = df_i.tail(160).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df["Date"],
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA60"], mode="lines", name="EMA60"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_UP"], mode="lines", name="BB_UP"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_MID"], mode="lines", name="BB_MID"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_LOW"], mode="lines", name="BB_LOW"))

    fig.add_hline(y=float(info["price"]), line_dash="dot", annotation_text="Entry(Close)", annotation_position="top left")
    fig.add_hline(y=float(info["stop"]), line_dash="dot", annotation_text="Stop", annotation_position="bottom left")
    fig.add_hline(y=float(info["target_rr"]), line_dash="dot", annotation_text="Target(RR)", annotation_position="top right")

    fig.update_layout(title=title, height=560, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# UI Tabs
# -----------------------------
tab_radar, tab_single = st.tabs(["⚡ 市場雷達（掃描）", "🎯 單股決策（買/賣/等 + 布林 + 點位）"])

# -----------------------------
# Radar tab
# -----------------------------
with tab_radar:
    today = dt.date.today()
    start = (today - dt.timedelta(days=int(radar_lookback))).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    st.caption(f"掃描範圍：{start} → {today.strftime('%Y-%m-%d')} | 股票池：{len(WATCHLIST)} 檔 | 快取：{cache_ttl}s")

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

    df_results = (
        pd.DataFrame(results).sort_values("分數", ascending=False)
        if results else pd.DataFrame(columns=["股票","分數","來源","進場","停損","目標","RR"])
    )

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader(f"⚡ 今日突破榜（{score_strong}+）")
        strong = df_results[df_results["分數"] >= int(score_strong)].copy()
        if strong.empty:
            st.info("今日沒有達到主榜門檻的突破型機會。")
        else:
            st.dataframe(strong, use_container_width=True)

        st.subheader(f"🟡 次強觀察區（{score_watch}~{int(score_strong)-1}）")
        watch = df_results[(df_results["分數"] >= int(score_watch)) & (df_results["分數"] < int(score_strong))].copy()
        if watch.empty:
            st.caption("無次強觀察標的。")
        else:
            st.dataframe(watch, use_container_width=True)

    with right:
        st.subheader("📌 從雷達選一檔帶入『單股決策』")
        if not df_results.empty:
            pick = st.selectbox("選一檔", options=df_results["股票"].tolist(), index=0)
            if st.button("➡️ 帶入單股決策（只會更新同一個代號）"):
                st.session_state["analysis_code"] = str(pick)
                st.success(f"已帶入：{pick}（切到『單股決策』分頁即可看到）")
        else:
            st.caption("目前雷達沒有可用標的可選。")

        with st.expander("🧩 逐檔診斷（收起不干擾）", expanded=False):
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

# -----------------------------
# Single tab
# -----------------------------
with tab_single:
    st.subheader("🎯 單股決策（盤後）— 清楚判斷 + 布林通道 + 預估未來買賣點")

    code_to_use = (st.session_state["analysis_code"] or "").strip().upper()
    st.write(f"**目前分析代號**：`{code_to_use}`（全站唯一，不會出現兩支股票混在一起）")

    today = dt.date.today()
    start = (today - dt.timedelta(days=int(single_lookback))).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    df_raw, src = load_data_best_effort(code_to_use, start, end)
    if df_raw.empty:
        st.error("❌ 無法取得資料（FinMind / yfinance 均失敗）。請換代號或稍後再試。")
        st.stop()

    df_i = add_indicators(df_raw)
    if df_i.empty:
        st.error("❌ 指標計算失敗或資料不足。請把回溯天數改 365/730 再試。")
        st.stop()

    info = single_decision(df_i)

    # 大字結論
    if info["decision"] == "BUY":
        st.success(info["decision_text"])
    elif info["decision"] == "SELL":
        st.error(info["decision_text"])
    else:
        st.warning(info["decision_text"])

    # 核心：買賣點 & 布林
    if signal_mode:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("預估買點（回踩買區）", f'{info["buy_pull"][0]} ~ {info["buy_pull"][1]}')
            st.metric("預估買點（突破觸發）", f'{info["buy_break"]}')
        with c2:
            st.metric("停損（ATR×1.2）", f'{info["stop"]}')
            st.metric("收盤（Entry）", f'{info["price"]}')
        with c3:
            st.metric("預估賣點（RR 2.2）", f'{info["target_rr"]}')
            st.metric("預估賣點（布林上軌）", f'{info["target_bb"]}')

        st.markdown("### 📉 布林通道分析")
        st.write(f"• 位置：**{info['boll_pos']}**")
        st.write(f"• 狀態：**{info['boll_state']}**")
        st.caption(f"資料來源：{src}")

        plot_single_chart(df_i, info, title=f"{code_to_use} | 來源：{src}")

        with st.expander("🧠 判斷依據（展開）", expanded=False):
            for r in info["reasons"]:
                st.write("• ", r)

    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close(Entry)", f'{info["price"]}')
        m2.metric("EMA20 / EMA60", f'{info["ema20"]} / {info["ema60"]}')
        m3.metric("RSI", f'{info["rsi"]}')
        m4.metric("MACD_H", f'{info["macd_h"]}')
        m5.metric("ATR", f'{info["atr"]}')

        st.markdown("### 🔮 預估未來買賣點")
        st.write(f"**回踩買區（EMA20±1%）**：`{info['buy_pull'][0]} ~ {info['buy_pull'][1]}`")
        st.write(f"**突破觸發（前5高+0.2ATR）**：`{info['buy_break']}`")
        st.write(f"**停損（ATR×1.2）**：`{info['stop']}`")
        st.write(f"**賣點（RR 2.2）**：`{info['target_rr']}`")
        st.write(f"**賣點（布林上軌）**：`{info['target_bb']}`")

        st.markdown("### 📉 布林通道分析")
        st.write(f"• 位置：**{info['boll_pos']}**")
        st.write(f"• 狀態：**{info['boll_state']}**")

        plot_single_chart(df_i, info, title=f"{code_to_use} | 來源：{src}")

st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
