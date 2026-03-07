# =========================================================
# AI 股票量化分析系統 V10.1（繁體中文版）
# FinMind + Yahoo fallback
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go

from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(layout="wide")

# =========================================================
# UI
# =========================================================

st.title("📈 AI 股票量化分析系統 V10.1")
st.caption("FinMind + Yahoo Finance ｜ 技術分析 + AI決策")

mode = st.radio(
    "系統模式",
    ["📊 單一股票分析", "🔎 Top Ten 掃描"],
    horizontal=True
)

# =========================================================
# FinMind Loader
# =========================================================

def load_finmind(symbol):

    url = "https://api.finmindtrade.com/api/v4/data"

    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": "2023-01-01"
    }

    r = requests.get(url, params=params)

    if r.status_code != 200:
        return None

    data = r.json()["data"]

    if len(data) == 0:
        return None

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])

    df.set_index("date", inplace=True)

    df.rename(columns={
        "open":"Open",
        "max":"High",
        "min":"Low",
        "close":"Close",
        "Trading_Volume":"Volume"
    }, inplace=True)

    return df


# =========================================================
# Yahoo fallback
# =========================================================

def load_yahoo(symbol):

    df = yf.download(symbol + ".TW", period="1y")

    if df.empty:
        df = yf.download(symbol + ".TWO", period="1y")

    if df.empty:
        return None

    return df


# =========================================================
# Unified Loader
# =========================================================

def load_data(symbol):

    df = load_finmind(symbol)

    if df is not None:
        return df, "FinMind"

    df = load_yahoo(symbol)

    if df is not None:
        return df, "Yahoo"

    return None, None


# =========================================================
# 指標
# =========================================================

def indicators(df):

    close = df["Close"]

    macd = MACD(close)

    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(df["High"], df["Low"], close)

    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    bb = BollingerBands(close)

    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    rsi = RSIIndicator(close)

    df["RSI"] = rsi.rsi()

    atr = AverageTrueRange(df["High"], df["Low"], close)

    df["ATR"] = atr.average_true_range()

    return df


# =========================================================
# AI score
# =========================================================

def ai_score(df):

    last = df.iloc[-1]

    score = 0

    if last["MACD"] > last["MACD_signal"]:
        score += 30

    if last["K"] > last["D"]:
        score += 20

    if last["RSI"] < 40:
        score += 20

    if last["Close"] < last["BB_low"]:
        score += 30

    return score


# =========================================================
# 買賣點
# =========================================================

def trade_points(df):

    last = df.iloc[-1]

    price = last["Close"]

    buy = min(last["BB_low"], df["Close"].rolling(20).mean().iloc[-1])

    sell = max(last["BB_high"], df["Close"].rolling(60).max().iloc[-1])

    stop = price - last["ATR"]*2

    rr = (sell-price)/(price-stop) if stop<price else None

    return buy, sell, stop, rr


# =========================================================
# Chart
# =========================================================

def chart(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="股價"))

    fig.add_trace(go.Scatter(x=df.index,y=df["BB_high"],name="布林上軌"))

    fig.add_trace(go.Scatter(x=df.index,y=df["BB_low"],name="布林下軌"))

    return fig


# =========================================================
# 單一股票
# =========================================================

if mode == "📊 單一股票分析":

    symbol = st.text_input("股票代碼", "2330")

    if st.button("開始分析"):

        df,source = load_data(symbol)

        if df is None:

            st.error("找不到資料")

        else:

            df = indicators(df)

            score = ai_score(df)

            buy,sell,stop,rr = trade_points(df)

            last = df.iloc[-1]

            price = last["Close"]

            if score > 60:

                decision = "🟢 買點"

            elif score < 30:

                decision = "🔴 賣點"

            else:

                decision = "🟡 觀察"

            st.markdown("## 股票決策總覽")

            c1,c2,c3,c4,c5 = st.columns(5)

            c1.metric("目前價格",f"{price:.2f}")

            c2.metric("AI評分",score)

            c3.metric("操作建議",decision)

            c4.metric("資料來源",source)

            c5.metric("RSI",f"{last['RSI']:.1f}")

            st.markdown("## 買賣點")

            b1,b2,b3,b4 = st.columns(4)

            b1.metric("預估買點",f"{buy:.2f}")

            b2.metric("預估賣點",f"{sell:.2f}")

            b3.metric("停損",f"{stop:.2f}")

            b4.metric("R/R",f"{rr:.2f}" if rr else "-")

            st.plotly_chart(chart(df))


# =========================================================
# Top Ten
# =========================================================

elif mode == "🔎 Top Ten 掃描":

    watch = st.text_area(
        "股票清單",
        """2330
2317
2454
2308
2881
2882
1301
1303
2002
1216"""
    )

    if st.button("開始掃描"):

        symbols = watch.split()

        result = []

        for s in symbols:

            df,source = load_data(s)

            if df is None:

                continue

            df = indicators(df)

            score = ai_score(df)

            last = df.iloc[-1]

            result.append({
                "股票":s,
                "價格":round(last["Close"],2),
                "AI分數":score,
                "RSI":round(last["RSI"],1)
            })

        df = pd.DataFrame(result)

        df.sort_values("AI分數",ascending=False,inplace=True)

        st.markdown("## Top Ten 機會股")

        st.dataframe(df.head(10),use_container_width=True)
