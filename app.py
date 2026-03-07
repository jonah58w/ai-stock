# ==============================================================
# AI 股票量化分析系統 V11 PRO
# Full Professional Edition
# (單一股票 + 全台股掃描 + 價值分析 + 技術分析 + AI決策)
# Designed for Streamlit Cloud
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go

from datetime import datetime, timedelta

from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ==============================================================
# UI
# ==============================================================

st.set_page_config(layout="wide")

st.title("📈 AI 股票量化分析系統 V11 PRO")
st.caption("FinMind + Yahoo Finance ｜ 技術分析 + 價值分析 + AI決策引擎")

# ==============================================================
# Utility
# ==============================================================

def safe(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except:
        return np.nan

# ==============================================================
# Price Loader
# ==============================================================

def load_price(symbol):

    try:
        df=yf.download(symbol+".TW",period="2y",progress=False)

        if df.empty:
            df=yf.download(symbol+".TWO",period="2y",progress=False)

        if df.empty:
            return None

        return df

    except:
        return None

# ==============================================================
# Fundamentals
# ==============================================================

def load_fundamental(symbol):

    out={
        "pe":np.nan,
        "pb":np.nan,
        "eps":np.nan,
        "roe":np.nan,
        "dividend":np.nan,
        "yield":np.nan
    }

    try:

        tk=yf.Ticker(symbol+".TW")
        info=tk.info

        out["pe"]=safe(info.get("trailingPE"))
        out["pb"]=safe(info.get("priceToBook"))
        out["eps"]=safe(info.get("trailingEps"))
        out["roe"]=safe(info.get("returnOnEquity"))*100
        out["dividend"]=safe(info.get("dividendRate"))

        dy=info.get("dividendYield")

        if dy:
            out["yield"]=safe(dy)*100

    except:
        pass

    return out

# ==============================================================
# Indicators
# ==============================================================

def add_indicators(df):

    close=df["Close"]

    macd=MACD(close)

    df["MACD"]=macd.macd()
    df["MACD_signal"]=macd.macd_signal()

    rsi=RSIIndicator(close)
    df["RSI"]=rsi.rsi()

    stoch=StochasticOscillator(df["High"],df["Low"],close)

    df["K"]=stoch.stoch()
    df["D"]=stoch.stoch_signal()

    bb=BollingerBands(close)

    df["BBH"]=bb.bollinger_hband()
    df["BBL"]=bb.bollinger_lband()

    atr=AverageTrueRange(df["High"],df["Low"],close)

    df["ATR"]=atr.average_true_range()

    sma20=SMAIndicator(close,20)
    df["SMA20"]=sma20.sma_indicator()

    sma50=SMAIndicator(close,50)
    df["SMA50"]=sma50.sma_indicator()

    sma200=SMAIndicator(close,200)
    df["SMA200"]=sma200.sma_indicator()

    return df

# ==============================================================
# Valuation Models
# ==============================================================

def dividend_valuation(div,req):

    if div is None or div==0:
        return None

    return div/(req/100)


def eps_valuation(eps,pe):

    if eps is None or pe is None:
        return None

    return eps*pe


def pb_valuation(book,pb):

    if book is None or pb is None:
        return None

    return book*pb

# ==============================================================
# AI Score
# ==============================================================

def ai_score(df,dy,pe,pb):

    last=df.iloc[-1]

    score=0

    if last["MACD"]>last["MACD_signal"]:
        score+=25

    if last["K"]>last["D"]:
        score+=20

    if last["RSI"]<40:
        score+=15

    if dy>5:
        score+=20

    if pe<20:
        score+=10

    if pb<2:
        score+=10

    return score

# ==============================================================
# Trade Point
# ==============================================================

def trade_point(df):

    last=df.iloc[-1]

    price=last["Close"]

    buy=min(last["BBL"],df["Close"].rolling(20).mean().iloc[-1])

    sell=max(last["BBH"],df["Close"].rolling(60).max().iloc[-1])

    stop=price-last["ATR"]*2

    rr=(sell-price)/(price-stop) if stop<price else None

    return buy,sell,stop,rr

# ==============================================================
# Chart
# ==============================================================

def chart(df):

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="股價"))

    fig.add_trace(go.Scatter(x=df.index,y=df["SMA20"],name="SMA20"))

    fig.add_trace(go.Scatter(x=df.index,y=df["SMA50"],name="SMA50"))

    fig.add_trace(go.Scatter(x=df.index,y=df["SMA200"],name="SMA200"))

    fig.add_trace(go.Scatter(x=df.index,y=df["BBH"],name="布林上軌"))

    fig.add_trace(go.Scatter(x=df.index,y=df["BBL"],name="布林下軌"))

    return fig

# ==============================================================
# Mode Select
# ==============================================================

mode=st.radio("系統模式",
              ["單一股票分析","Top10機會掃描"],
              horizontal=True)

# ==============================================================
# Single Stock
# ==============================================================

if mode=="單一股票分析":

    symbol=st.text_input("股票代碼","2330")

    if st.button("開始分析"):

        df=load_price(symbol)

        if df is None:

            st.error("找不到股票資料")

        else:

            df=add_indicators(df)

            fund=load_fundamental(symbol)

            price=df.iloc[-1]["Close"]

            dy=fund["yield"]

            if dy is None and fund["dividend"]:

                dy=fund["dividend"]/price*100

            pe=fund["pe"]
            pb=fund["pb"]
            eps=fund["eps"]
            roe=fund["roe"]

            fair_div=dividend_valuation(fund["dividend"],5)
            fair_eps=eps_valuation(eps,15)

            buy,sell,stop,rr=trade_point(df)

            score=ai_score(df,dy,pe,pb)

            st.markdown("## 股票決策總覽")

            c1,c2,c3,c4=st.columns(4)

            c1.metric("股票",symbol)
            c2.metric("目前價格",round(price,2))
            c3.metric("AI綜合評分",score)
            c4.metric("AI建議","買進" if score>60 else "觀察")

            st.markdown("## 價值分析")

            v1,v2,v3,v4,v5=st.columns(5)

            v1.metric("殖利率",round(dy,2) if dy else "-")
            v2.metric("本益比",round(pe,2) if pe else "-")
            v3.metric("股價淨值比",round(pb,2) if pb else "-")
            v4.metric("EPS",round(eps,2) if eps else "-")
            v5.metric("ROE",round(roe,2) if roe else "-")

            st.markdown("## 合理價估值")

            a1,a2=st.columns(2)

            a1.metric("股利估值",round(fair_div,2) if fair_div else "-")
            a2.metric("EPS估值",round(fair_eps,2) if fair_eps else "-")

            st.markdown("## 買賣點")

            b1,b2,b3,b4=st.columns(4)

            b1.metric("預估買點",round(buy,2))
            b2.metric("預估賣點",round(sell,2))
            b3.metric("停損",round(stop,2))
            b4.metric("R/R",round(rr,2) if rr else "-")

            st.plotly_chart(chart(df),use_container_width=True)

# ==============================================================
# Top 10 Scan
# ==============================================================

else:

    universe=[

"2330","2317","2454","2308","2382",
"2603","2609","2881","2882","2886",
"2891","0050","0056","00878","00919",
"1301","1303","2002","1216","6488"

]

    if st.button("開始掃描全台股"):

        rows=[]

        for s in universe:

            df=load_price(s)

            if df is None:
                continue

            df=add_indicators(df)

            fund=load_fundamental(s)

            price=df.iloc[-1]["Close"]

            dy=fund["yield"]

            if dy is None and fund["dividend"]:
                dy=fund["dividend"]/price*100

            pe=fund["pe"]
            pb=fund["pb"]

            score=ai_score(df,dy,pe,pb)

            rows.append({

"股票":s,
"股價":round(price,2),
"AI分數":score,
"殖利率":round(dy,2) if dy else None,
"本益比":round(pe,2) if pe else None,
"股價淨值比":round(pb,2) if pb else None

            })

        df=pd.DataFrame(rows)

        df=df.sort_values("AI分數",ascending=False)

        st.dataframe(df.head(10))

# ==============================================================
# END
# ==============================================================\n
