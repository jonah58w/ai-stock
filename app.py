# ==================================================
# AI Cycle Trading Engine PRO v5.3
# 多事件主要轉折共振版（Launch + Continuation）
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


# =========================
# 下載資料（TW / TWO fallback）
# =========================
def download_data(code):
    if "." in code:
        df = yf.download(code, period="3y", progress=False)
        return df, code

    for suffix in [".TW", ".TWO"]:
        symbol = code + suffix
        df = yf.download(symbol, period="3y", progress=False)
        if df is not None and not df.empty:
            return df, symbol

    return None, None


# =========================
# 指標計算
# =========================
def calculate_indicators(df):

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()

    df["EMA20"] = df["Close"].ewm(span=20).mean()

    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + df["BB_std"] * 2
    df["BB_lower"] = df["BB_mid"] - df["BB_std"] * 2

    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    df["K"] = 100 * (df["Close"] - low9) / (high9 - low9 + 1e-9)
    df["D"] = df["K"].rolling(3).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["VOL5"] = df["Volume"].rolling(5).mean()
    df["HH20"] = df["High"].rolling(20).max()

    return df


# =========================
# Pivot偵測
# =========================
def detect_pivot_lows(df, left=3, right=3, atr_mult=1.0):

    pivots = []
    for i in range(left, len(df) - right):
        if df["Low"].iloc[i] == df["Low"].iloc[i-left:i+right+1].min():

            bounce = df["High"].iloc[i:i+right+1].max() - df["Low"].iloc[i]

            if bounce >= atr_mult * df["ATR"].iloc[i]:
                pivots.append(i)

    return pivots


# =========================
# 共振條件（延續版）
# =========================
def continuation_resonance(df, i):

    if i < 2:
        return False

    macd_bull = df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]
    kd_bull = df["K"].iloc[i] > df["D"].iloc[i]
    ma_bull = df["SMA5"].iloc[i] > df["SMA10"].iloc[i] > df["SMA20"].iloc[i]

    breakout = (
        (df["Close"].iloc[i] > df["BB_upper"].iloc[i] and
         df["Close"].iloc[i-1] <= df["BB_upper"].iloc[i-1])
        or
        (df["Close"].iloc[i] > df["HH20"].iloc[i-1])
    )

    volume_ok = df["Volume"].iloc[i] > df["VOL5"].iloc[i]

    return macd_bull and kd_bull and ma_bull and breakout and volume_ok


# =========================
# 找最近 N 次主要轉折共振
# =========================
def find_recent_events(df, lookback=200, confirm_window=60, atr_mult=1.0, max_events=5):

    start = max(0, len(df) - lookback)
    df2 = df.iloc[start:].copy()

    pivots = detect_pivot_lows(df2, atr_mult=atr_mult)

    events = []

    for p in pivots:

        for i in range(p+1, min(p+confirm_window, len(df2)-1)):

            if continuation_resonance(df2, i):

                events.append({
                    "PivotDate": df2.index[p],
                    "BreakoutDate": df2.index[i],
                    "EntryPrice": float(df2["Close"].iloc[i])
                })
                break

    events = sorted(events, key=lambda x: x["BreakoutDate"], reverse=True)

    return events[:max_events]


# =========================
# UI
# =========================
def main():

    st.set_page_config(layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO v5.3（多事件轉折共振版）")

    code = st.text_input("股票代碼", "6187")

    lookback = st.sidebar.slider("只看最近幾根K棒", 100, 400, 250)
    confirm_window = st.sidebar.slider("Pivot後幾天內找突破", 30, 100, 60)
    atr_mult = st.sidebar.slider("轉折強度(ATR倍數)", 0.5, 2.0, 0.8)

    df_raw, symbol = download_data(code)

    if df_raw is None or df_raw.empty:
        st.error("無法下載資料")
        return

    st.caption(f"成功取得資料：{symbol}")

    df = calculate_indicators(df_raw).dropna()

    events = find_recent_events(df, lookback, confirm_window, atr_mult)

    if not events:
        st.info("最近區間沒有找到主要轉折共振")
        return

    st.subheader("📌 最近主要轉折共振事件")

    options = [
        f"{e['BreakoutDate'].date()} | 轉折:{e['PivotDate'].date()} | 進場:{e['EntryPrice']:.2f}"
        for e in events
    ]

    selected = st.selectbox("選擇要檢視的事件", options)

    selected_index = options.index(selected)
    event = events[selected_index]

    entry_price = event["EntryPrice"]
    current_price = df["Close"].iloc[-1]
    gain = (current_price / entry_price - 1) * 100

    st.success(
        f"主要轉折日：{event['PivotDate'].date()} ｜ "
        f"突破日：{event['BreakoutDate'].date()} ｜ "
        f"啟動價：{entry_price:.2f} ｜ "
        f"至今漲幅：{gain:.2f}%"
    )

    # 畫圖
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower"))

    fig.add_trace(go.Scatter(
        x=[event["PivotDate"]],
        y=[df.loc[event["PivotDate"], "Low"]],
        mode="markers",
        marker=dict(size=12),
        name="Pivot"
    ))

    fig.add_trace(go.Scatter(
        x=[event["BreakoutDate"]],
        y=[entry_price],
        mode="markers",
        marker=dict(size=12),
        name="Breakout"
    ))

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
