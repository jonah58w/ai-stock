# ==============================
# AI Stock Trading Assistant
# 升級版 A+B 趨勢預測系統
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 資料下載
# ==============================

def download_data(stock_code):
    df = yf.download(
        stock_code,
        period="3y",
        interval="1d",
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=False,
    )
    return df

# ==============================
# 技術指標
# ==============================

def calculate_indicators(df):
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100

    # ADX (簡化版)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(14).mean()

    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()

    return df

# ==============================
# A：趨勢延續機率模型
# ==============================

def calculate_trend_probability(df):
    latest = df.iloc[-1]
    score = 0

    # EMA 斜率
    ema20_slope = (df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]) / df["EMA20"].iloc[-5] * 100
    if ema20_slope > 1:
        score += 25
    elif ema20_slope > 0:
        score += 15

    # ADX
    if latest["ADX"] > 40:
        score += 20
    elif latest["ADX"] > 25:
        score += 10

    # 動量
    roc20 = (latest["Close"] / df["Close"].iloc[-20] - 1) * 100
    if roc20 > 15:
        score += 20
    elif roc20 > 5:
        score += 10

    # 量能
    if latest["VOL_5"] > latest["VOL_10"] * 1.2:
        score += 15

    # 突破結構
    if latest["Close"] > df["High"].rolling(20).max().iloc[-2]:
        score += 20

    return min(score, 100)

# ==============================
# B：ATR 價格區間預測
# ==============================

def forecast_range(df, days):
    latest = df.iloc[-1]
    price = latest["Close"]
    atr_pct = latest["ATR_Pct"] / 100

    expected_move = price * atr_pct * np.sqrt(days)

    return round(price - expected_move, 2), round(price + expected_move, 2)

# ==============================
# 交易區域生成
# ==============================

def generate_trading_zone(df):
    prob = calculate_trend_probability(df)
    lower5, upper5 = forecast_range(df, 5)
    lower10, upper10 = forecast_range(df, 10)

    latest = df.iloc[-1]
    price = latest["Close"]

    if prob > 65:
        buy_zone = (lower5, price)
        risk_line = lower5 * 0.97
        regime = "強趨勢延續"
    elif prob > 50:
        buy_zone = (lower5, price)
        risk_line = lower5 * 0.95
        regime = "中性偏多"
    else:
        buy_zone = None
        risk_line = None
        regime = "震盪/不明朗"

    return prob, lower5, upper5, lower10, upper10, buy_zone, risk_line, regime

# ==============================
# 主程式
# ==============================

def main():
    st.set_page_config(page_title="AI Stock Trading Assistant 升級版", layout="wide")
    st.title("📊 AI Stock Trading Assistant（A+B 趨勢預測升級版）")

    stock_input = st.text_input("股票代碼", "2313")

    if stock_input:
        stock_code = stock_input.strip() + ".TW"

        df = download_data(stock_code)

        if df is None or df.empty:
            st.error("無法下載資料")
            return

        df = calculate_indicators(df)
        df.reset_index(inplace=True)

        latest = df.iloc[-1]
        price = latest["Close"]

        # 基本資訊
        c1, c2, c3 = st.columns(3)
        c1.metric("當前價格", round(price, 2))
        c2.metric("ADX", round(latest["ADX"], 1))
        c3.metric("ATR%", round(latest["ATR_Pct"], 2))

        # A+B 模型
        st.subheader("📈 趨勢預測模型（A+B）")

        prob, lower5, upper5, lower10, upper10, buy_zone, risk_line, regime = generate_trading_zone(df)

        st.metric("趨勢延續機率", f"{prob}%")
        st.write(f"趨勢結構判斷：{regime}")

        st.write("🔮 未來 5 日預測區間：", lower5, "~", upper5)
        st.write("🔮 未來 10 日預測區間：", lower10, "~", upper10)

        if buy_zone:
            st.success(f"建議布局區：{buy_zone[0]} ~ {buy_zone[1]}")
            st.warning(f"風險警戒線：{round(risk_line,2)}")
        else:
            st.info("目前不適合主動布局（機率不足）")

        # 圖表
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["EMA20"],
            name="EMA20"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["EMA50"],
            name="EMA50"
        ))

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ 本模型為概率與波動預測模型，不構成投資建議。")

if __name__ == "__main__":
    main()
