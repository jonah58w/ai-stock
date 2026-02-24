from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import requests

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# stooq fallback
from pandas_datareader import data as pdr

st.set_page_config(layout="wide")
st.title("🧠 AI 台股量化專業平台")

# ---------------------------
# Utils
# ---------------------------

def _normalize_symbol(code: str) -> str:
    return code.strip().upper().replace(".TW", "").replace(".TWO", "")

@st.cache_data(ttl=86400)
def get_stock_name(code: str) -> str:
    code = _normalize_symbol(code)
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        data = requests.get(url, timeout=10).json()
        for s in data:
            if s.get("Code") == code:
                return s.get("Name", "")
    except Exception:
        pass
    return ""

@st.cache_data(ttl=600)
def download_data(code: str, period: str = "6mo", max_retry: int = 3):
    """
    下載股價：先 yfinance（.TW -> .TWO），失敗改用 stooq
    回傳: (df, ticker_used)
    """
    code = _normalize_symbol(code)
    candidates = [f"{code}.TW", f"{code}.TWO"]

    # 1) yfinance (retry)
    for ticker in candidates:
        for _ in range(max_retry):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    threads=False,
                    auto_adjust=False,
                )
                if df is not None and not df.empty:
                    df = df.dropna(subset=["Close"])
                    if not df.empty:
                        return df, ticker
            except Exception:
                time.sleep(1)

    # 2) stooq fallback (retry)
    # stooq 對台股通常可用：2330.TW
    for ticker in candidates:
        for _ in range(max_retry):
            try:
                df = pdr.DataReader(ticker, "stooq")
                # stooq 回來通常是新->舊，翻轉成舊->新
                if df is not None and not df.empty:
                    df = df.sort_index()
                    # 對齊欄位名稱成 yfinance 風格
                    df.rename(
                        columns={
                            "Open": "Open",
                            "High": "High",
                            "Low": "Low",
                            "Close": "Close",
                            "Volume": "Volume",
                        },
                        inplace=True,
                    )
                    df = df.dropna(subset=["Close"])
                    if not df.empty:
                        return df, ticker + " (stooq)"
            except Exception:
                time.sleep(1)

    return None, None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["Upper"] = ma20 + 2 * std20
    df["Lower"] = ma20 - 2 * std20

    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def advanced_ai_score(df: pd.DataFrame) -> int:
    latest = df.iloc[-1]
    score = 0

    if pd.notna(latest.get("SMA20")) and pd.notna(latest.get("SMA60")) and latest["SMA20"] > latest["SMA60"]:
        score += 20
    if pd.notna(latest.get("SMA20")) and latest["Close"] > latest["SMA20"]:
        score += 15
    if pd.notna(latest.get("RSI")) and latest["RSI"] > 60:
        score += 15
    if pd.notna(latest.get("Upper")) and latest["Close"] > latest["Upper"]:
        score += 15
    if pd.notna(latest.get("Volume_MA")) and latest["Volume"] > latest["Volume_MA"]:
        score += 15

    hh60 = df["High"].rolling(60).max().iloc[-1]
    if pd.notna(hh60) and latest["Close"] >= hh60:
        score += 20

    return int(min(score, 100))

# ---------------------------
# AI 3日後價格預測 (RF)
# ---------------------------

def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Target"] = df["Close"].shift(-3)
    return df.dropna()

def predict_price_3d(df: pd.DataFrame):
    # 防呆：資料太少就不預測
    if df is None or len(df) < 120:
        return None, None

    df2 = _create_features(df)

    features = ["Close", "SMA20", "SMA60", "RSI", "Volume", "Volatility", "Momentum"]
    if any(c not in df2.columns for c in features):
        return None, None

    X = df2[features]
    y = df2["Target"]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    latest = df2.iloc[-1:][features]
    pred = float(model.predict(latest)[0])

    current = float(df["Close"].iloc[-1])
    change_pct = (pred / current - 1) * 100
    return pred, change_pct

# ---------------------------
# Top10 掃描（修 KeyError）
# ---------------------------

def scan_top10(stock_list: list[str]) -> pd.DataFrame:
    rows = []

    for code in stock_list:
        df, used = download_data(code, period="6mo")
        if df is None or len(df) < 80:
            continue

        df = calculate_indicators(df)
        score = advanced_ai_score(df)

        rows.append({
            "股票": _normalize_symbol(code),
            "資料來源": used,
            "AI分數": score
        })

    # ✅ 保證欄位存在（就算空也不會 KeyError）
    result_df = pd.DataFrame(rows, columns=["股票", "資料來源", "AI分數"])

    # ✅ 空結果直接回傳，不 sort
    if result_df.empty:
        return result_df

    return result_df.sort_values("AI分數", ascending=False).head(10)

# ---------------------------
# UI
# ---------------------------

mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])

if mode == "單一股票分析":
    code = st.text_input("請輸入股票代號", "2330")

    df, used = download_data(code, period="6mo")

    if df is None:
        st.error("❌ 無法取得資料（Yahoo 可能暫時限制 / 網路不穩）。請稍後再試，或換一檔股票。")
        st.stop()

    name = get_stock_name(code)
    df = calculate_indicators(df)
    df["ATR"] = calculate_atr(df)

    score = advanced_ai_score(df)

    last_price = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else None
    stop_loss = (last_price - 2 * atr) if atr else None

    st.success(f"{name} ({_normalize_symbol(code)})  |  {used}")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI 趨勢分數", f"{score}/100")
    c2.metric("目前價格", round(last_price, 2))
    c3.metric("建議停損", "-" if stop_loss is None else round(stop_loss, 2))

    pred, change_pct = predict_price_3d(df)
    st.subheader("🧠 AI 3日價格預測（RF）")

    if pred is None:
        st.info("資料量不足或指標不足，暫不做預測（建議 period 改成 1y / 2y 會更穩）。")
    else:
        st.write("預測價格：", round(pred, 2))
        if change_pct > 1:
            st.success(f"📈 偏多（約 +{round(change_pct, 2)}%）")
        elif change_pct < -1:
            st.error(f"📉 偏空（約 {round(change_pct, 2)}%）")
        else:
            st.info("🔎 偏震盪")

    st.line_chart(df["Close"])

else:
    # 你可以自己擴充這個池（先用小池測試雲端穩定性）
    stock_pool = ["2330", "2317", "2303", "2454", "8046", "3037", "2382"]

    st.caption("Top10 掃描器：若結果為空，代表資料源暫時抓不到（Yahoo/網路限制），稍後再試即可。")
    result = scan_top10(stock_pool)

    st.subheader("🔥 AI 強勢股 Top 10")

    if result.empty:
        st.warning("目前掃描結果為空（代表資料下載失敗）。請稍後再試或換 stock_pool。")
    else:
        st.dataframe(result, use_container_width=True)
