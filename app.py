# app.py
# AI Stock Trading Assistant（TW）- V9 Full（雷達 + 單股決策 / 盤後型）
# ✅ FinMind 優先 + yfinance fallback（MultiIndex 修正）
# ✅ 自動掃描（主榜/觀察）+ 每檔保護不炸版
# ✅ 單股決策：BUY/SELL/WAIT + 預估未來買賣點 + 布林通道分析 + 圖表（只畫未來點位線）
# ⚠️ 僅供資訊顯示與風險控管演算，不構成投資建議，不會自動下單。

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
st.set_page_config(page_title="AI Stock Trading Assistant (TW) - V9", layout="wide")
st.title("⚡ AI Stock Trading Assistant（V9：雷達 + 單股決策 / 盤後型）")
st.caption("BUILD: V9-FULL-SINGLE-DECISION-20260228")  # 用來驗證你線上是不是跑到最新版本

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]
DEFAULT_LOOKBACK_RADAR = 240
DEFAULT_LOOKBACK_SINGLE = 365

# -----------------------------
# Sidebar (單股入口永遠存在)
# -----------------------------
st.sidebar.header("🎯 單一股票決策（盤後）")
single_code = st.sidebar.text_input("股票代號", value="2330")
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
colA, colB = st.sidebar.columns(2)
with colA:
    score_strong = st.sidebar.number_input("主榜門檻", value=80, min_value=60, max_value=100, step=1)
with colB:
    score_watch = st.sidebar.number_input("觀察門檻", value=60, min_value=0, max_value=int(score_strong), step=1)

cache_ttl = st.sidebar.number_input("快取秒數（Cloud 建議 600）", value=600, min_value=60, max_value=3600, step=60)
radar_lookback = st.sidebar.number_input("雷達回溯天數", value=DEFAULT_LOOKBACK_RADAR, min_value=120, max_value=2000, step=10)

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
    # yfinance 偶爾回 MultiIndex：('Close','2330.TW')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: Date, Open, High, Low, Close, Volume (clean)."""
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

    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = _safe_float_series(df[c])

    df = df.dropna(subset=["Open","High","Low","Close"])
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
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df)
    return _ensure_ohlcv(df)

def load_data_best_effort(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, source_used)
    """
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

    # 保守：避免 ta 遇到 NaN 爆掉
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

    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    df = df.dropna().reset_index(drop=True)
    return df

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
# Single-stock decision engine (盤後，清楚 BUY/SELL/WAIT + 未來點位)
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

    # 布林狀態（壓縮/擴張）：用近120天寬度分位數
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

    # 布林位置
    if price >= bb_up:
        boll_pos = "上軌附近（偏過熱/動能強）"
    elif price <= bb_low:
        boll_pos = "下軌附近（偏超跌/反彈區）"
    elif price >= bb_mid:
        boll_pos = "中軌之上（偏多）"
    else:
        boll_pos = "中軌之下（偏弱）"

    # 盤後判斷條件
    trend_ok = (price > ema20 and ema20 > ema60)
    breakout_ok = (price > float(prev5_high))
    vol_ok = (vol_mean > 0 and vol > 1.3 * vol_mean)
    macd_ok = (macd_h > 0)
    rsi_ok = (rsi > 55)

    # ✅ 你要的未來預估點位
    pullback_buy_low = round(ema20 * 0.99, 2)
    pullback_buy_high = round(ema20 * 1.01, 2)
    breakout_trigger = round(float(prev5_high) + 0.2 * atr, 2)

    stop = round(price - 1.2 * atr, 2)
    target_rr = round(price + (price - stop) * 2.2, 2)
    target_bb = round(bb_up, 2)

    # ✅ 一眼結論（BUY/SELL/WAIT）
    reasons = []
    reasons.append("趨勢OK" if trend_ok else "趨勢未確認")
    reasons.append("突破OK" if breakout_ok else "未突破前5高")
    reasons.append("量能OK" if vol_ok else "量能未放大")
    reasons.append("MACD柱翻正" if macd_ok else "MACD柱偏弱")
    reasons.append("RSI>55" if rsi_ok else "RSI未達55")

    if trend_ok and breakout_ok and rsi_ok and (vol_ok or macd_ok):
        decision = "BUY"
        decision_text = "🟢 BUY（盤後確認突破）"
    elif (price < ema20) and (macd_h < 0):
        decision = "SELL"
        decision_text = "🔴 SELL / 減碼（跌破 EMA20 + 動能轉弱）"
    else:
        decision = "WAIT"
        decision_text = "🟡 WAIT（等回踩買區或再突破確認）"

    return {
        "decision": decision,
        "decision_text": decision_text,
        "price": round(price, 2),

        "pullback_buy_low": pullback_buy_low,
        "pullback_buy_high": pullback_buy_high,
        "breakout_trigger": breakout_trigger,

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

    # ✅ 只畫「未來預估」相關線：Entry(收盤)、Stop、Target
    fig.add_hline(y=float(info["price"]), line_dash="dot", annotation_text="Entry (Close)", annotation_position="top left")
    fig.add_hline(y=float(info["stop"]), line_dash="dot", annotation_text="Stop", annotation_position="bottom left")
    fig.add_hline(y=float(info["target_rr"]), line_dash="dot", annotation_text="Target (RR)", annotation_position="top right")

    fig.update_layout(title=title, height=560, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Main UI (Tabs)
# -----------------------------
tab_radar, tab_single = st.tabs(["⚡ 市場雷達（掃描）", "🎯 單股決策（清楚買/賣/等 + 未來點位）"])

# -----------------------------
# Tab: Radar
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

    col1, col2 = st.columns([1.4, 1])

    with col1:
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

    with col2:
        st.subheader("📌 從雷達快速丟到「單股決策」")
        if not df_results.empty:
            pick = st.selectbox("選一檔立即分析", options=df_results["股票"].tolist(), index=0)
            if st.button("➡️ 用這檔做單股決策"):
                st.session_state["single_pick"] = pick
                st.success(f"已選：{pick}（切到『單股決策』分頁即可看到）")
        else:
            st.caption("目前雷達沒有可用標的可選。")

        st.subheader("🧩 逐檔診斷（收起不干擾）")
        with st.expander("展開查看哪一檔被跳過/原因", expanded=False):
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

# -----------------------------
# Tab: Single Decision
# -----------------------------
with tab_single:
    st.subheader("🎯 單股決策（盤後）— 一眼判斷 BUY / SELL / WAIT + 預估未來買賣點")

    # 決定要分析哪一檔：優先用雷達選取、其次用 sidebar 按鈕、否則用 sidebar 文字
    code_to_use = (st.session_state.get("single_pick") or single_code or "").strip().upper()

    # 如果 sidebar 有按分析，仍以 sidebar 的 single_code 為準
    if run_single:
        code_to_use = (single_code or "").strip().upper()

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

    # 1) 超清楚的結論
    if info["decision"] == "BUY":
        st.success(info["decision_text"])
    elif info["decision"] == "SELL":
        st.error(info["decision_text"])
    else:
        st.warning(info["decision_text"])

    # 2) 訊號機模式：只顯示「你要決策的三件事」
    if signal_mode:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("預估買點（回踩買區）", f'{info["pullback_buy_low"]} ~ {info["pullback_buy_high"]}')
            st.metric("預估買點（突破觸發）", f'{info["breakout_trigger"]}')
        with c2:
            st.metric("停損（ATR×1.2）", f'{info["stop"]}')
            st.metric("當前收盤（Entry）", f'{info["price"]}')
        with c3:
            st.metric("預估賣點（RR 2.2）", f'{info["target_rr"]}')
            st.metric("預估賣點（布林上軌）", f'{info["target_bb"]}')

        st.caption(f"布林：{info['boll_pos']}｜狀態：{info['boll_state']}｜來源：{src}")

        # 圖表（仍保留）
        plot_single_chart(df_i, info, title=f"{code_to_use} | 來源：{src}")

        # 原因收起來
        with st.expander("🧠 為什麼是這個結論（展開看規則）", expanded=False):
            st.write("• " + " / ".join(info["reasons"]))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("EMA20", f'{info["ema20"]}')
            m2.metric("EMA60", f'{info["ema60"]}')
            m3.metric("RSI", f'{info["rsi"]}')
            m4.metric("MACD_H", f'{info["macd_h"]}')
            m5.metric("ATR", f'{info["atr"]}')

    # 3) 完整模式：更細節
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close(Entry)", f'{info["price"]}')
        m2.metric("EMA20 / EMA60", f'{info["ema20"]} / {info["ema60"]}')
        m3.metric("RSI", f'{info["rsi"]}')
        m4.metric("MACD_H", f'{info["macd_h"]}')
        m5.metric("ATR", f'{info["atr"]}')

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### 🧠 判斷原因")
            st.write("• " + " / ".join(info["reasons"]))
            st.markdown("### 📉 布林通道")
            st.write(f"• 位置：**{info['boll_pos']}**")
            st.write(f"• 狀態：**{info['boll_state']}**")

        with right:
            st.markdown("### 🔮 預估未來買賣點")
            st.write(f"**回踩買區（EMA20±1%）**：`{info['pullback_buy_low']} ~ {info['pullback_buy_high']}`")
            st.write(f"**突破觸發（前5高+0.2ATR）**：`{info['breakout_trigger']}`")
            st.write(f"**停損（ATR×1.2）**：`{info['stop']}`")
            st.write(f"**賣點（RR 2.2）**：`{info['target_rr']}`")
            st.write(f"**賣點（布林上軌）**：`{info['target_bb']}`")

        plot_single_chart(df_i, info, title=f"{code_to_use} | 來源：{src}")

        with st.expander("📋 最新資料（最後 30 筆）", expanded=False):
            st.dataframe(df_i.tail(30), use_container_width=True)

st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
