# ==================================================
# AI Stock Trading Assistant - Cycle Trading Engine
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# =============================
# 資料下載
# =============================
def download_data(symbol):
    return yf.download(
        symbol,
        period="3y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

# =============================
# 技術指標計算
# =============================
def calculate_indicators(df):

    df = df.copy()

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # 布林通道
    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + df["BB_std"] * 2
    df["BB_lower"] = df["BB_mid"] - df["BB_std"] * 2

    # MACD
    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD_DIF"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_DIF"].ewm(span=9).mean()

    # KD
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    # ATR
    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["Close"].shift())
    tr3 = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100

    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(14).mean()

    return df

# =============================
# 趨勢週期識別
# =============================
def detect_trend_stage(df):

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    deviation = (latest["Close"]/latest["EMA20"] - 1) * 100

    macd_cross = (
        prev["MACD_DIF"] < prev["MACD_Signal"] and
        latest["MACD_DIF"] > latest["MACD_Signal"]
    )

    kd_cross = (
        prev["KD_K"] < prev["KD_D"] and
        latest["KD_K"] > latest["KD_D"]
    )

    ma_bull = (
        latest["SMA5"] > latest["SMA10"] >
        latest["SMA20"]
    )

    if (
        macd_cross and kd_cross and ma_bull and
        20 < latest["ADX"] < 40 and
        deviation < 12
    ):
        return "啟動期"

    elif (
        ma_bull and latest["ADX"] > 40 and
        12 <= deviation <= 20
    ):
        return "加速期"

    elif (
        deviation > 20 and
        latest["KD_K"] > 90 and
        latest["ADX"] > 55
    ):
        return "末端期"

    else:
        return "盤整期"

# =============================
# 倉位建議引擎
# =============================
def position_engine(stage):

    if stage == "啟動期":
        return "建議初始倉位 40%，回踩加碼"
    elif stage == "加速期":
        return "小倉突破（10~20%），等 0.5~1 ATR 回踩加碼"
    elif stage == "末端期":
        return "禁止追高，考慮分批減碼"
    else:
        return "觀望為主"

# =============================
# 主程式
# =============================
def main():

    st.set_page_config(page_title="Cycle Trading Engine", layout="wide")
    st.title("🚀 AI Cycle Trading Engine（週期交易引擎版）")

    stock = st.text_input("股票代碼", "2313")
    symbol = stock + ".TW"

    df = download_data(symbol)

    if df.empty:
        st.error("資料下載失敗")
        return

    df = calculate_indicators(df)
    df.reset_index(inplace=True)

    latest = df.iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Close", round(latest["Close"],2))
    c2.metric("EMA20", round(latest["EMA20"],2))
    c3.metric("ADX", round(latest["ADX"],1))
    c4.metric("ATR%", round(latest["ATR_pct"],2))

    stage = detect_trend_stage(df)
    st.subheader("📊 趨勢週期判斷")
    st.metric("當前階段", stage)

    st.subheader("💼 倉位建議")
    st.info(position_engine(stage))

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50"))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
