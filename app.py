# app.py
# AI Stock Trading Assistant V10 CORE
# Upgrade from V9.3
# Keeps all V9.3 functions + adds Value AI + AI Score Engine

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from ta.trend import MACD, SMAIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(layout="wide")

# ==============================
# DATA LOADER
# ==============================

def load_data(symbol):

    ticker = yf.Ticker(symbol)

    df = ticker.history(period="1y")

    if df.empty:
        return None

    df = df.dropna()

    return df


# ==============================
# TECHNICAL INDICATORS
# ==============================

def calculate_indicators(df):

    close = df["Close"]

    macd = MACD(close)

    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(
        df["High"], df["Low"], close
    )

    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    bb = BollingerBands(close)

    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    atr = AverageTrueRange(
        df["High"], df["Low"], close
    )

    df["ATR"] = atr.average_true_range()

    return df


# ==============================
# TECHNICAL SCORE
# ==============================

def technical_score(df):

    score = 0

    latest = df.iloc[-1]

    if latest["MACD"] > latest["MACD_signal"]:
        score += 30

    if latest["K"] > latest["D"]:
        score += 30

    if latest["Close"] < latest["BB_low"]:
        score += 40

    return score


# ==============================
# MOMENTUM SCORE
# ==============================

def momentum_score(df):

    score = 0

    ma20 = SMAIndicator(df["Close"], window=20).sma_indicator()

    price = df["Close"].iloc[-1]

    if price > ma20.iloc[-1]:
        score += 50

    volume_mean = df["Volume"].rolling(20).mean().iloc[-1]

    if df["Volume"].iloc[-1] > volume_mean:
        score += 50

    return score


# ==============================
# DIVIDEND MODEL
# ==============================

def dividend_yield(symbol, price):

    try:
        ticker = yf.Ticker(symbol)

        dividend = ticker.info.get("dividendRate", 0)

        if dividend is None:
            dividend = 0

        yield_rate = dividend / price

        return dividend, yield_rate

    except:
        return 0, 0


# ==============================
# VALUATION MODEL
# ==============================

def fair_price(dividend):

    r = 0.08
    g = 0.03

    if dividend == 0:
        return None

    price = dividend / (r - g)

    return price


# ==============================
# MARKET RISK MODEL
# ==============================

def market_risk(df):

    atr = df["ATR"].iloc[-1]

    price = df["Close"].iloc[-1]

    volatility = atr / price

    if volatility > 0.05:
        return 30

    if volatility > 0.03:
        return 60

    return 90


# ==============================
# AI SCORE ENGINE
# ==============================

def ai_score(tech, momentum, value, risk):

    score = (
        0.4 * tech +
        0.3 * momentum +
        0.2 * value +
        0.1 * risk
    )

    return round(score, 2)


# ==============================
# VALUE SCORE
# ==============================

def value_score(price, fair):

    if fair is None:
        return 50

    if price < fair * 0.7:
        return 90

    if price < fair:
        return 70

    if price < fair * 1.2:
        return 50

    return 20


# ==============================
# BUY / SELL SIGNAL
# ==============================

def signal(score):

    if score > 80:
        return "🚀 STRONG BUY"

    if score > 60:
        return "BUY"

    if score > 40:
        return "WAIT"

    return "SELL"


# ==============================
# CHART
# ==============================

def chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_high"],
            name="BB Upper"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_low"],
            name="BB Lower"
        )
    )

    return fig


# ==============================
# UI
# ==============================

st.title("AI Stock Trading Assistant V10")

symbol = st.text_input("Stock Symbol", "2330.TW")

if st.button("Run Analysis"):

    df = load_data(symbol)

    if df is None:
        st.error("No data")
        st.stop()

    df = calculate_indicators(df)

    price = df["Close"].iloc[-1]

    tech = technical_score(df)

    momentum = momentum_score(df)

    dividend, yield_rate = dividend_yield(symbol, price)

    fair = fair_price(dividend)

    value = value_score(price, fair)

    risk = market_risk(df)

    score = ai_score(tech, momentum, value, risk)

    decision = signal(score)

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Score", score)
    col2.metric("Dividend Yield", f"{yield_rate:.2%}")
    col3.metric("Signal", decision)

    st.write("Fair Value:", fair)

    fig = chart(df)

    st.plotly_chart(fig, use_container_width=True)
