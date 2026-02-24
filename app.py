# ==========================================
# AI 台股量化專業平台（最穩定官方資料源版）
# 使用 Data.gov.tw 公開 API
# 不依賴 Yahoo / TPEX 主站
# ==========================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

# ----------------------------------------
# 下載資料（政府開放資料）
# ----------------------------------------

@st.cache_data(ttl=600)
def download_data(stock_id: str):

    stock_id = stock_id.strip()

    try:
        url = "https://data.ntpc.gov.tw/api/datasets/ea9a5b54-66a8-4b2a-9c03-8efebd75c4d7/json"
        data = requests.get(url, timeout=10).json()

        df = pd.DataFrame(data)

        df = df[df["證券代號"] == stock_id]

        if df.empty:
            return None

        df = df.rename(columns={
            "日期": "Date",
            "開盤價": "Open",
            "最高價": "High",
            "最低價": "Low",
            "收盤價": "Close",
            "成交股數": "Volume"
        })

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close"])

        return df

    except:
        return None


# ----------------------------------------
# 技術指標
# ----------------------------------------

def calculate_indicators(df):

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    return df


# ----------------------------------------
# AI 趨勢分數
# ----------------------------------------

def ai_score(df):

    latest = df.iloc[-1]
    score = 0

    if latest["SMA20"] > latest["SMA60"]:
        score += 30

    if latest["RSI"] > 60:
        score += 30

    if latest["Volume"] > latest["Volume_MA"]:
        score += 20

    if latest["Close"] > latest["SMA20"]:
        score += 20

    return min(score, 100)


# ----------------------------------------
# AI 預測
# ----------------------------------------

def predict_price(df):

    df["Return"] = df["Close"].pct_change()
    df["Target"] = df["Close"].shift(-3)
    df = df.dropna()

    features = ["Close", "SMA20", "SMA60", "RSI", "Volume"]

    X = df[features]
    y = df["Target"]

    model = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X, y)

    latest = df.iloc[-1:][features]
    prediction = model.predict(latest)[0]

    current = df["Close"].iloc[-1]
    change_pct = (prediction / current - 1) * 100

    return prediction, change_pct


# ----------------------------------------
# UI
# ----------------------------------------

st.title("🧠 AI 台股量化專業平台（最穩定版）")

mode = st.sidebar.radio("選擇模式", ["單一股票分析"])

if mode == "單一股票分析":

    stock_id = st.text_input("請輸入股票代號", "2330")

    df = download_data(stock_id)

    if df is None or len(df) < 100:
        st.error("❌ 無法取得資料，請確認股票代號。")
        st.stop()

    df = calculate_indicators(df)

    score = ai_score(df)
    prediction, change_pct = predict_price(df)

    last_price = df["Close"].iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric("AI 趨勢分數", f"{score}/100")
    col2.metric("目前價格", round(last_price, 2))

    st.subheader("🧠 AI 3日價格預測")

    st.write("預測價格：", round(prediction, 2))

    if change_pct > 1:
        st.success(f"📈 偏多（+{round(change_pct,2)}%）")
    elif change_pct < -1:
        st.error(f"📉 偏空（{round(change_pct,2)}%）")
    else:
        st.info("🔎 偏震盪")

    st.line_chart(df["Close"])
