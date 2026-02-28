# app.py
# V7 短波段突破雷達（自動掃描版）
# 80 分以上才顯示

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from pandas_datareader import data as pdr
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="V7 短波段突破雷達", layout="wide")
st.title("⚡ V7 短波段突破雷達（80+ 強動能）")

# ----------------------
# 股票池（可自行修改）
# ----------------------
WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]

# ----------------------
# 快取資料 10 分鐘
# ----------------------
@st.cache_data(ttl=600)
def load_data(code):
    ticker = code + ".TW"
    end = dt.date.today()
    start = end - dt.timedelta(days=180)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, group_by="column")
    except:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 🔥 解決 MultiIndex 問題
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # 強制保留必要欄位
    needed = ["Open","High","Low","Close","Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[["Open","High","Low","Close","Volume"]].copy()
    df = df.dropna()
    df = df.astype(float)

    return df

# ----------------------
# 計算突破分數
# ----------------------
def breakout_score(df):
    if df is None or df.empty:
        return 0, None

    df = df.copy()

    # 強制整理欄位
    df = df[["Open","High","Low","Close","Volume"]]
    df = df.dropna()
    df = df.astype(float)

    if len(df) < 60:
        return 0, None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    try:
        ema20 = EMAIndicator(close, 20).ema_indicator()
        ema60 = EMAIndicator(close, 60).ema_indicator()
        rsi = RSIIndicator(close, 14).rsi()
        macd = MACD(close).macd_diff()
        bb = BollingerBands(close)
        atr = AverageTrueRange(high, low, close, 14).average_true_range()
    except:
        return 0, None

    df["EMA20"] = ema20
    df["EMA60"] = ema60
    df["RSI"] = rsi
    df["MACD_H"] = macd
    df["ATR"] = atr

    df = df.dropna()

    if len(df) < 60:
        return 0, None

    latest = df.iloc[-1]
    prev5_high = df["High"].iloc[-6:-1].max()
    vol_mean = df["Volume"].iloc[-6:-1].mean()

    score = 0

    if latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA60"]:
        score += 20

    if latest["Close"] > prev5_high:
        score += 20

    if latest["Volume"] > 1.5 * vol_mean:
        score += 20

    if latest["RSI"] > 55:
        score += 20

    if latest["MACD_H"] > 0:
        score += 20

    entry = latest["Close"]
    stop = entry - latest["ATR"] * 1.2
    target = entry + (entry - stop) * 2.2

    return score, {
        "entry": round(entry,2),
        "stop": round(stop,2),
        "target": round(target,2),
        "rr": round((target-entry)/(entry-stop),2)
    }

# ----------------------
# 市場風險燈號（0050）
# ----------------------
def market_risk():
    df = load_data("0050")
    if df.empty:
        return "🟡 市場資料不足"

    ema20 = EMAIndicator(df["Close"], 20).ema_indicator()
    if df["Close"].iloc[-1] > ema20.iloc[-1]:
        return "🟢 市場可積極操作"
    else:
        return "🔴 市場偏弱，控制倉位"

# ----------------------
# 自動掃描
# ----------------------
results = []

for code in WATCHLIST:
    df = load_data(code)
    if df.empty:
        continue

    score, trade = breakout_score(df)
    if score >= 60:
        results.append({
            "股票": code,
            "分數": score,
            "進場": trade["entry"],
            "停損": trade["stop"],
            "目標": trade["target"],
            "RR": trade["rr"]
        })

if results:
    df_results = pd.DataFrame(results).sort_values("分數", ascending=False)

    st.subheader("⚡ 今日突破榜（80+）")
    strong = df_results[df_results["分數"] >= 80]
    st.dataframe(strong, use_container_width=True)

    st.subheader("🟡 次強觀察區（60~79）")
    watch = df_results[(df_results["分數"] >= 60) & (df_results["分數"] < 80)]
    st.dataframe(watch, use_container_width=True)
else:
    st.info("今日無突破型機會")

st.subheader("📊 市場風險燈號")
st.write(market_risk())

# ----------------------
# 資金控管
# ----------------------
st.subheader("💰 資金控管計算")

capital = st.number_input("總資金", value=5000000)
risk_pct = st.number_input("單筆風險 %", value=2.0)

if strong.shape[0] > 0:
    first = strong.iloc[0]
    risk_amount = capital * (risk_pct/100)
    loss_per_share = first["進場"] - first["停損"]
    shares = int(risk_amount / loss_per_share) if loss_per_share > 0 else 0

    st.write("建議操作股票：", first["股票"])
    st.write("可買張數：約", shares)