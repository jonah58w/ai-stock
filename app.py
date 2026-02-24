# ==========================================
# AI 台股量化專業平台 Ultimate Version
# ==========================================

from __future__ import annotations
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# ==========================================
# 下載資料（自動判斷 TW / TWO）
# ==========================================

@st.cache_data(ttl=600)
def download_data(symbol, period="6mo"):

    symbol = symbol.strip()
    candidates = [f"{symbol}.TW", f"{symbol}.TWO"]

    for ticker in candidates:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                return df, ticker
        except:
            time.sleep(1)

    return None, None


# ==========================================
# 中文名稱
# ==========================================

@st.cache_data(ttl=86400)
def get_stock_name(symbol):

    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        data = requests.get(url, timeout=10).json()

        for stock in data:
            if stock["Code"] == symbol:
                return stock["Name"]
    except:
        pass

    return ""


# ==========================================
# 技術指標
# ==========================================

def calculate_indicators(df):

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Upper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["Lower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()

    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    return df


# ==========================================
# ATR 停損
# ==========================================

def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


# ==========================================
# AI 趨勢評分
# ==========================================

def advanced_ai_score(df):

    latest = df.iloc[-1]
    score = 0

    if latest["SMA20"] > latest["SMA60"]:
        score += 20

    if latest["Close"] > latest["SMA20"]:
        score += 15

    if latest["RSI"] > 60:
        score += 15

    if latest["Close"] > latest["Upper"]:
        score += 15

    if latest["Volume"] > latest["Volume_MA"]:
        score += 15

    if latest["Close"] >= df["High"].rolling(60).max().iloc[-1]:
        score += 20

    return min(score, 100)


# ==========================================
# AI 價格預測（3日後）
# ==========================================

def create_features(df):

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Target"] = df["Close"].shift(-3)
    df = df.dropna()

    return df


def train_ai_model(df):

    df = create_features(df)

    features = ["Close", "SMA20", "SMA60", "RSI",
                "Volume", "Volatility", "Momentum"]

    X = df[features]
    y = df["Target"]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model, features


def predict_price(df):

    model, features = train_ai_model(df)

    latest = df.iloc[-1:][features]
    prediction = model.predict(latest)[0]

    current_price = df["Close"].iloc[-1]
    change_pct = (prediction / current_price - 1) * 100

    return prediction, change_pct


# ==========================================
# Top 10 掃描
# ==========================================

def scan_top10(stock_list):

    results = []

    for symbol in stock_list:

        df, full_symbol = download_data(symbol)

        if df is None or len(df) < 60:
            continue

        df = calculate_indicators(df)
        score = advanced_ai_score(df)

        results.append({
            "股票": symbol,
            "AI分數": score
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("AI分數", ascending=False).head(10)

    return result_df


# ==========================================
# Streamlit UI
# ==========================================

st.title("🧠 AI 台股量化專業平台")

mode = st.sidebar.radio(
    "選擇模式",
    ["單一股票分析", "Top 10 掃描器"]
)

if mode == "單一股票分析":

    stock_input = st.text_input("請輸入股票代號", "2330")

    if stock_input:

        df, full_symbol = download_data(stock_input)

        if df is None:
            st.error("❌ 無法取得資料")
            st.stop()

        stock_name = get_stock_name(stock_input)

        df = calculate_indicators(df)
        df["ATR"] = calculate_atr(df)

        score = advanced_ai_score(df)
        prediction, change_pct = predict_price(df)

        last_price = df["Close"].iloc[-1]
        atr_value = df["ATR"].iloc[-1]
        stop_loss = last_price - (2 * atr_value)

        st.success(f"{stock_name} ({full_symbol})")

        col1, col2, col3 = st.columns(3)

        col1.metric("AI 趨勢分數", f"{score}/100")
        col2.metric("目前價格", round(last_price, 2))
        col3.metric("建議停損", round(stop_loss, 2))

        st.subheader("🧠 AI 3日價格預測")
        st.write("預測價格：", round(prediction, 2))

        if change_pct > 1:
            st.success(f"📈 上漲機率偏高 (+{round(change_pct,2)}%)")
        elif change_pct < -1:
            st.error(f"📉 下跌風險偏高 ({round(change_pct,2)}%)")
        else:
            st.info("🔎 震盪機率高")

        st.line_chart(df["Close"])


elif mode == "Top 10 掃描器":

    stock_pool = ["2330", "2317", "2303", "2454", "8046", "3037", "2382"]

    result = scan_top10(stock_pool)

    st.subheader("🔥 AI 強勢股 Top 10")
    st.dataframe(result)
