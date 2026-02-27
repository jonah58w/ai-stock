# ============================================
# AI 台股量化專業平台 V22 穩定實戰完整版
# ============================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta
from datetime import datetime

st.set_page_config(layout="wide")

# ============================
# 參數
# ============================
MAX_BUY_GAP = 0.12   # 近端買點最大距離 12%
TOP10_POOL = ["2330","2317","2454","2303","2382","3037","8046","4967"]

# ============================
# 工具函數
# ============================

def get_data(symbol: str, period="6mo"):
    try:
        df = yf.download(symbol + ".TW", period=period, progress=False)
        if df.empty:
            df = yf.download(symbol + ".TWO", period=period, progress=False)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None


def compute_zones(df):

    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = float(close.iloc[-1])

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    mid = float(bb.bollinger_mavg().iloc[-1])
    up = float(bb.bollinger_hband().iloc[-1])
    dn = float(bb.bollinger_lband().iloc[-1])

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    atr_now = float(atr.average_true_range().iloc[-1])

    ema20 = float(close.ewm(span=20).mean().iloc[-1])

    swing_low_60 = float(low.rolling(60).min().iloc[-1])
    swing_high_60 = float(high.rolling(60).max().iloc[-1])

    # ===== 近端買點 =====
    near_low = max(dn, ema20 - 1.2*atr_now)
    near_high = min(mid, ema20 + 0.3*atr_now)

    gap_near = (price - ((near_low+near_high)/2)) / price

    if gap_near > MAX_BUY_GAP:
        near_low = ema20 - 0.8*atr_now
        near_high = ema20 + 0.2*atr_now
        gap_near = (price - ((near_low+near_high)/2)) / price

    # ===== 深回檔 =====
    deep_low = swing_low_60 - 0.5*atr_now
    deep_high = swing_low_60 + 0.8*atr_now
    gap_deep = (price - ((deep_low+deep_high)/2)) / price

    # ===== 近端賣點 =====
    sell_low = max(mid, up - 0.3*atr_now)
    sell_high = up + 0.8*atr_now
    gap_sell = (((sell_low+sell_high)/2) - price) / price

    return {
        "price": price,
        "buy_near": (near_low, near_high),
        "buy_deep": (deep_low, deep_high),
        "sell_near": (sell_low, sell_high),
        "gap_near": gap_near,
        "gap_deep": gap_deep,
        "gap_sell": gap_sell,
        "bb_mid": mid,
        "bb_up": up,
        "bb_dn": dn,
        "ema20": ema20
    }


def compute_ai_score(df):

    close = df["close"]
    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
    macd = ta.trend.MACD(close)
    macd_hist = macd.macd_diff().iloc[-1]

    score = 50

    if rsi < 30: score += 20
    if rsi > 70: score -= 20
    if macd_hist > 0: score += 15
    if macd_hist < 0: score -= 15

    return max(0, min(100, int(score)))


def decision(z):
    if abs(z["gap_near"]) < 0.02:
        return "🔥 買點區間"
    if abs(z["gap_sell"]) < 0.02:
        return "⚠️ 賣點區間"
    return "⌛ 觀察"


# ============================
# UI
# ============================

st.title("🧠 AI 台股量化專業平台 V22")

mode = st.sidebar.radio("模式", ["單股分析","Top10掃描"])

period = st.sidebar.selectbox("資料期間",["3mo","6mo","1y"])

# ============================
# 單股分析
# ============================

if mode == "單股分析":

    stock = st.text_input("請輸入股票代號","2330")

    df = get_data(stock, period)

    if df is None:
        st.error("無法取得資料")
        st.stop()

    z = compute_zones(df)
    score = compute_ai_score(df)

    st.subheader("目前價格")
    st.markdown(f"# {z['price']:.2f}")

    st.markdown(f"### AI共振分數 {score}/100")

    st.markdown("## 📌 當下判斷")
    st.success(decision(z))

    st.markdown("## 🎯 可操作買點")
    st.info(f"{z['buy_near'][0]:.2f} ~ {z['buy_near'][1]:.2f}   (距離 {z['gap_near']*100:.1f}%)")

    st.markdown("## 🕰️ 深回檔買點")
    st.info(f"{z['buy_deep'][0]:.2f} ~ {z['buy_deep'][1]:.2f}   (距離 {z['gap_deep']*100:.1f}%)")

    st.markdown("## 🎯 近端賣點")
    st.warning(f"{z['sell_near'][0]:.2f} ~ {z['sell_near'][1]:.2f}   (距離 {z['gap_sell']*100:.1f}%)")

    # ===== 圖表 =====
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K線"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["close"].ewm(span=20).mean(),
                             name="EMA20"))

    bb = ta.volatility.BollingerBands(df["close"])
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(), name="BB上軌"))
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(), name="BB下軌"))

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)


# ============================
# Top10 掃描
# ============================

else:

    data = []

    for s in TOP10_POOL:
        df = get_data(s, period)
        if df is None: continue

        z = compute_zones(df)
        score = compute_ai_score(df)

        data.append({
            "股票": s,
            "價格": round(z["price"],2),
            "AI分數": score,
            "近端買點距離%": round(z["gap_near"]*100,1),
            "近端賣點距離%": round(z["gap_sell"]*100,1),
            "判斷": decision(z)
        })

    df_out = pd.DataFrame(data)
    df_out = df_out.sort_values("AI分數", ascending=False)

    st.dataframe(df_out, use_container_width=True)
