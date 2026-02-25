# ==================================================
# AI Cycle Trading Engine PRO v3
# A + B + C Strategy Engine
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

REQUIRED = ["Open","High","Low","Close","Volume"]

# =============================
# 1. Normalize (Cloud 防爆)
# =============================
def normalize_ohlcv(df):

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    rename = {}
    for c in df.columns:
        s = str(c).lower()
        if "open" in s: rename[c] = "Open"
        elif "high" in s: rename[c] = "High"
        elif "low" in s: rename[c] = "Low"
        elif "close" in s: rename[c] = "Close"
        elif "volume" in s: rename[c] = "Volume"

    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]

    return df

# =============================
# 2. Download
# =============================
def download_data(symbol):
    df = yf.download(
        symbol,
        period="3y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )
    return df

# =============================
# 3. Indicators
# =============================
def calculate_indicators(df):

    df = normalize_ohlcv(df)

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Bollinger
    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + df["BB_std"] * 2
    df["BB_lower"] = df["BB_mid"] - df["BB_std"] * 2

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()

    # KD
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["KD_K"] = 100*(df["Close"]-low_9)/(high_9-low_9+1e-9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    # ATR
    tr1 = df["High"]-df["Low"]
    tr2 = abs(df["High"]-df["Close"].shift())
    tr3 = abs(df["Low"]-df["Close"].shift())
    tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"]/df["Close"]*100

    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm<0]=0
    minus_dm[minus_dm<0]=0
    atr14 = tr.rolling(14).mean()
    plus_di = 100*(plus_dm.rolling(14).mean()/(atr14+1e-9))
    minus_di = 100*(minus_dm.rolling(14).mean()/(atr14+1e-9))
    dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di+1e-9)
    df["ADX"] = dx.rolling(14).mean()

    df["VOL_5"] = df["Volume"].rolling(5).mean()

    return df

# =============================
# 4. Trend Stage
# =============================
def detect_stage(df):

    latest = df.iloc[-1]
    deviation = (latest["Close"]/latest["EMA20"]-1)*100

    if deviation>20 and latest["KD_K"]>90 and latest["ADX"]>55:
        return "末端期"
    elif latest["ADX"]>40 and 12<=deviation<=20:
        return "加速期"
    elif latest["ADX"]>20 and deviation<12:
        return "啟動期"
    else:
        return "盤整期"

# =============================
# 5. Resonance Detect
# =============================
def detect_resonance(df):

    signals=[]

    for i in range(20,len(df)):

        macd = df["MACD_DIF"].iloc[i-1]<df["MACD_Signal"].iloc[i-1] and \
               df["MACD_DIF"].iloc[i]>df["MACD_Signal"].iloc[i]

        kd = df["KD_K"].iloc[i-1]<df["KD_D"].iloc[i-1] and \
             df["KD_K"].iloc[i]>df["KD_D"].iloc[i]

        ma = df["SMA5"].iloc[i]>df["SMA10"].iloc[i]>df["SMA20"].iloc[i]
        bb = df["Close"].iloc[i]>df["BB_upper"].iloc[i]
        vol = df["Volume"].iloc[i]>df["VOL_5"].iloc[i]

        if macd and kd and ma and bb and vol:
            signals.append(i)

    return signals

# =============================
# 6. Backtest
# =============================
def backtest(df,signals):

    results=[]

    for i in signals:
        if i+10>=len(df):
            continue

        entry=df["Close"].iloc[i]
        ret5=(df["Close"].iloc[i+5]/entry-1)*100
        ret10=(df["Close"].iloc[i+10]/entry-1)*100

        results.append([ret5,ret10])

    if len(results)==0:
        return None

    bt=pd.DataFrame(results,columns=["ret5","ret10"])

    return {
        "avg5":bt["ret5"].mean(),
        "avg10":bt["ret10"].mean(),
        "winrate":(bt["ret10"]>0).mean()*100
    }

# =============================
# 7. UI
# =============================
def main():

    st.set_page_config(layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO v3")

    code=st.text_input("股票代碼","2313")
    symbol=code+".TW"

    df=download_data(symbol)

    if df.empty:
        st.error("下載失敗")
        return

    df=calculate_indicators(df).dropna()

    latest=df.iloc[-1]

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Close",round(latest["Close"],2))
    c2.metric("EMA20",round(latest["EMA20"],2))
    c3.metric("ADX",round(latest["ADX"],1))
    c4.metric("ATR%",round(latest["ATR_pct"],2))

    stage=detect_stage(df)
    st.subheader("📊 週期判斷")
    st.metric("當前階段",stage)

    signals=detect_resonance(df)

    if signals:
        last=signals[-1]
        entry=df["Close"].iloc[last]
        gain=(latest["Close"]/entry-1)*100

        st.subheader("🚀 最近共振啟動")
        st.write("啟動價:",round(entry,2))
        st.write("至今漲幅:",round(gain,2),"%")

        bt=backtest(df,signals)
        if bt:
            st.subheader("📊 歷史回測")
            st.write("5日平均:",round(bt["avg5"],2),"%")
            st.write("10日平均:",round(bt["avg10"],2),"%")
            st.write("成功率:",round(bt["winrate"],2),"%")

    # Chart
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))
    fig.add_trace(go.Scatter(x=df.index,y=df["EMA20"],name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index,y=df["EMA50"],name="EMA50"))

    st.plotly_chart(fig,use_container_width=True)

if __name__=="__main__":
    main()
