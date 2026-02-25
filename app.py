# ==============================================
# AI Stock Trading Assistant PRO
# A+B 趨勢預測 + 分批布局 + 過熱警示 + 進場評分
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 下載資料
# ==============================

def download_data(stock_code):
    df = yf.download(
        stock_code,
        period="3y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return df

# ==============================
# 技術指標
# ==============================

def calculate_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100

    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(14).mean()

    # KD
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()

    return df

# ==============================
# 趨勢機率模型
# ==============================

def trend_probability(df):
    latest = df.iloc[-1]
    score = 0

    slope = (df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]) / df["EMA20"].iloc[-5] * 100
    if slope > 1:
        score += 25
    elif slope > 0:
        score += 15

    if latest["ADX"] > 40:
        score += 20
    elif latest["ADX"] > 25:
        score += 10

    roc20 = (latest["Close"] / df["Close"].iloc[-20] - 1) * 100
    if roc20 > 15:
        score += 20
    elif roc20 > 5:
        score += 10

    if latest["VOL_5"] > latest["VOL_10"] * 1.2:
        score += 15

    if latest["Close"] > df["High"].rolling(20).max().iloc[-2]:
        score += 20

    return min(score, 100)

# ==============================
# ATR 區間預測
# ==============================

def forecast_range(df, days):
    latest = df.iloc[-1]
    price = latest["Close"]
    atr_pct = latest["ATR_Pct"] / 100
    move = price * atr_pct * np.sqrt(days)
    return round(price - move, 2), round(price + move, 2)

# ==============================
# 過熱判斷
# ==============================

def overheat_check(df):
    latest = df.iloc[-1]
    price = latest["Close"]
    ema20 = latest["EMA20"]

    deviation = (price / ema20 - 1) * 100

    if deviation > 20 and latest["KD_K"] > 85:
        return True
    return False

# ==============================
# 分批布局 + 評分
# ==============================

def generate_strategy(df):
    prob = trend_probability(df)
    lower5, upper5 = forecast_range(df, 5)
    lower10, upper10 = forecast_range(df, 10)

    latest = df.iloc[-1]
    price = latest["Close"]
    ema20 = latest["EMA20"]

    safe_zone = round(ema20,2)
    standard_zone = lower5
    aggressive_zone = lower10

    score = prob

    deviation = (price / ema20 - 1) * 100
    if deviation < 10:
        score += 10
    if latest["ADX"] > 40:
        score += 10

    score = min(score,100)

    return {
        "prob": prob,
        "lower5": lower5,
        "upper5": upper5,
        "lower10": lower10,
        "upper10": upper10,
        "safe": safe_zone,
        "standard": standard_zone,
        "aggressive": aggressive_zone,
        "entry_score": score
    }

# ==============================
# 主程式
# ==============================

def main():
    st.set_page_config(page_title="AI Stock Trading Assistant PRO", layout="wide")
    st.title("📊 AI Stock Trading Assistant PRO")

    stock_input = st.text_input("股票代碼", "2313")

    if stock_input:
        stock_code = stock_input.strip() + ".TW"

        df = download_data(stock_code)

        if df.empty:
            st.error("資料下載失敗")
            return

        df = calculate_indicators(df)
        df.reset_index(inplace=True)

        latest = df.iloc[-1]
        price = latest["Close"]

        c1,c2,c3 = st.columns(3)
        c1.metric("當前價格", round(price,2))
        c2.metric("ADX", round(latest["ADX"],1))
        c3.metric("ATR%", round(latest["ATR_Pct"],2))

        strategy = generate_strategy(df)

        st.subheader("📈 趨勢與區間預測")
        st.metric("趨勢延續機率", f"{strategy['prob']}%")
        st.write("5日區間:", strategy["lower5"], "~", strategy["upper5"])
        st.write("10日區間:", strategy["lower10"], "~", strategy["upper10"])

        st.subheader("🎯 分批布局建議")
        st.write("安全布局區（EMA20）:", strategy["safe"])
        st.write("標準布局區（5日下緣）:", strategy["standard"])
        st.write("積極布局區（10日下緣）:", strategy["aggressive"])

        st.subheader("📊 進場評分")
        st.metric("當前進場評分", strategy["entry_score"])

        if overheat_check(df):
            st.error("⚠️ 過熱警示：不建議追高")

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
