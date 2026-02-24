# ==========================================
# AI 台股量化專業平台（官方資料源版：TWSE + TPEX）
# - Streamlit Cloud 友善（不依賴 Yahoo / yfinance）
# - 單一股票分析 + Top10 掃描器
# - 技術指標 + AI 趨勢分數 + ATR 停損 + RF 3日預測
# ==========================================

from __future__ import annotations

import math
import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")


# -------------------------
# Helpers
# -------------------------

def _norm_code(code: str) -> str:
    return str(code).strip().upper().replace(".TW", "").replace(".TWO", "")


def _roc_year(ad_year: int) -> int:
    return ad_year - 1911


def _period_to_months(period: str) -> int:
    p = (period or "6mo").strip().lower()
    if p.endswith("mo"):
        return max(1, int(p.replace("mo", "")))
    if p.endswith("y"):
        return max(1, int(p.replace("y", ""))) * 12
    # default
    return 6


def _safe_float(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"", "--", "null", "None"}:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan


def _safe_int(x):
    v = _safe_float(x)
    if math.isnan(v):
        return np.nan
    return int(v)


# -------------------------
# Official data fetch
# -------------------------

@st.cache_data(ttl=86400)
def get_stock_name(code: str) -> str:
    """
    用 TWSE OpenAPI 抓中文名稱（上市+上櫃大多能找到；找不到就回空字串）
    """
    code = _norm_code(code)
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        data = requests.get(url, timeout=10).json()
        for s in data:
            if s.get("Code") == code:
                return s.get("Name", "")
    except:
        pass
    return ""


def _fetch_twse_month(code: str, yyyymm: str) -> pd.DataFrame | None:
    """
    TWSE 上市：抓單月資料（yyyymm = '202601'）
    """
    # TWSE 要 date=YYYYMMDD（DD可給01）
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={code}"
    r = requests.get(url, timeout=15)
    j = r.json()

    if j.get("stat") != "OK":
        return None

    fields = j.get("fields", [])
    data = j.get("data", [])
    if not data:
        return None

    df = pd.DataFrame(data, columns=fields)

    # 常見欄位名稱（TWSE）
    # 日期 / 開盤價 / 最高價 / 最低價 / 收盤價 / 成交股數
    colmap = {
        "日期": "Date",
        "開盤價": "Open",
        "最高價": "High",
        "最低價": "Low",
        "收盤價": "Close",
        "成交股數": "Volume",
    }
    df = df.rename(columns=colmap)

    if "Date" not in df.columns:
        return None

    # 日期是民國：例如 114/02/03
    # 轉成 AD
    def parse_twse_date(s):
        s = str(s)
        parts = s.split("/")
        if len(parts) != 3:
            return pd.NaT
        y = int(parts[0]) + 1911
        m = int(parts[1])
        d = int(parts[2])
        return pd.Timestamp(y, m, d)

    df["Date"] = df["Date"].apply(parse_twse_date)
    df = df.set_index("Date").sort_index()

    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(_safe_int)

    df = df.dropna(subset=["Close"])
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def _fetch_tpex_month(code: str, yyyymm: str) -> pd.DataFrame | None:
    """
    TPEX 上櫃：st43_result.php 抓單月資料（yyyymm = '202601'）
    參數 d 需要民國年/月：例如 114/02
    """
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    d_param = f"{_roc_year(y)}/{m:02d}"

    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?d={d_param}&stkno={code}"
    r = requests.get(url, timeout=15)
    j = r.json()

    # 常見結構：{"aaData":[...], "iTotalRecords":...} 或 {"data":[...]}
    data = j.get("aaData") or j.get("data") or []
    if not data:
        return None

    # 典型欄位順序常見為：
    # [日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數]
    # 這裡只取我們需要的
    rows = []
    for row in data:
        if not row or len(row) < 7:
            continue

        date_s = row[0]
        vol_s = row[1]
        open_s = row[3]
        high_s = row[4]
        low_s = row[5]
        close_s = row[6]

        # 日期為民國：114/02/03
        parts = str(date_s).split("/")
        if len(parts) == 3:
            yy = int(parts[0]) + 1911
            mm = int(parts[1])
            dd = int(parts[2])
            date = pd.Timestamp(yy, mm, dd)
        else:
            continue

        rows.append(
            {
                "Date": date,
                "Open": _safe_float(open_s),
                "High": _safe_float(high_s),
                "Low": _safe_float(low_s),
                "Close": _safe_float(close_s),
                "Volume": _safe_int(vol_s),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df.dropna(subset=["Close"])
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


@st.cache_data(ttl=600)
def download_data(code: str, period: str = "6mo"):
    """
    先抓 TWSE（上市），不行再抓 TPEX（上櫃）。
    由於兩者都是「按月」回傳，因此會把最近 N 個月串起來。
    """
    code = _norm_code(code)
    months = _period_to_months(period)

    end = dt.date.today().replace(day=1)
    month_list = [(end - relativedelta(months=i)).strftime("%Y%m") for i in range(months)]
    month_list = list(reversed(month_list))

    # 1) try TWSE
    twse_parts = []
    for yyyymm in month_list:
        try:
            mdf = _fetch_twse_month(code, yyyymm)
            if mdf is not None and not mdf.empty:
                twse_parts.append(mdf)
        except:
            pass

    if twse_parts:
        df = pd.concat(twse_parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df, "TWSE"

    # 2) try TPEX
    tpex_parts = []
    for yyyymm in month_list:
        try:
            mdf = _fetch_tpex_month(code, yyyymm)
            if mdf is not None and not mdf.empty:
                tpex_parts.append(mdf)
        except:
            pass

    if tpex_parts:
        df = pd.concat(tpex_parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df, "TPEX"

    return None, None


# -------------------------
# Indicators / Scoring
# -------------------------

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


# -------------------------
# AI Prediction (RF)
# -------------------------

def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Target"] = df["Close"].shift(-3)
    return df.dropna()


def predict_price_3d(df: pd.DataFrame):
    if df is None or len(df) < 120:
        return None, None

    df2 = _create_features(df)
    features = ["Close", "SMA20", "SMA60", "RSI", "Volume", "Volatility", "Momentum"]
    if any(c not in df2.columns for c in features):
        return None, None

    X = df2[features]
    y = df2["Target"]

    model = RandomForestRegressor(
        n_estimators=140,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    latest = df2.iloc[-1:][features]
    pred = float(model.predict(latest)[0])

    current = float(df["Close"].iloc[-1])
    change_pct = (pred / current - 1) * 100
    return pred, change_pct


# -------------------------
# Top10 Scanner (safe)
# -------------------------

def scan_top10(stock_list: list[str], period: str = "6mo") -> pd.DataFrame:
    rows = []

    for code in stock_list:
        df, src = download_data(code, period=period)
        if df is None or len(df) < 80:
            continue

        df = calculate_indicators(df)
        score = advanced_ai_score(df)

        rows.append({
            "股票": _norm_code(code),
            "資料來源": src,
            "AI分數": score
        })

    result_df = pd.DataFrame(rows, columns=["股票", "資料來源", "AI分數"])
    if result_df.empty:
        return result_df

    return result_df.sort_values("AI分數", ascending=False).head(10)


# -------------------------
# UI
# -------------------------

st.title("🧠 AI 台股量化專業平台（官方資料源版）")

mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])
period = st.sidebar.selectbox("資料期間", ["3mo", "6mo", "12mo"], index=1)

if mode == "單一股票分析":
    code = st.text_input("請輸入股票代號", "2330")

    df, src = download_data(code, period=period)
    if df is None:
        st.error("❌ 無法取得資料（TWSE/TPEX 皆無回傳）。請確認代號，或稍後再試。")
        st.stop()

    name = get_stock_name(code)
    df = calculate_indicators(df)
    df["ATR"] = calculate_atr(df)

    score = advanced_ai_score(df)
    last_price = float(df["Close"].iloc[-1])

    atr_val = df["ATR"].iloc[-1]
    stop_loss = None if pd.isna(atr_val) else (last_price - 2 * float(atr_val))

    st.success(f"{name} ({_norm_code(code)})  |  Source: {src}")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI 趨勢分數", f"{score}/100")
    c2.metric("目前價格", round(last_price, 2))
    c3.metric("建議停損", "-" if stop_loss is None else round(stop_loss, 2))

    pred, change_pct = predict_price_3d(df)
    st.subheader("🧠 AI 3日價格預測（RandomForest）")

    if pred is None:
        st.info("資料量不足或指標不足，暫不做預測（建議期間選 12mo 會更穩）。")
    else:
        st.write("預測價格：", round(pred, 2))
        if change_pct > 1:
            st.success(f"📈 偏多（約 +{round(change_pct, 2)}%）")
        elif change_pct < -1:
            st.error(f"📉 偏空（約 {round(change_pct, 2)}%）")
        else:
            st.info("🔎 偏震盪")

    st.subheader("收盤價走勢")
    st.line_chart(df["Close"])

else:
    # 先用小池測試，之後你要我再幫你改成「全市場自動抓清單」也可以
    stock_pool = ["2330", "2317", "2303", "2454", "8046", "3037", "2382"]

    st.caption("Top10 掃描器：使用 TWSE/TPEX 官方資料源，雲端穩定度高。")
    result = scan_top10(stock_pool, period=period)

    st.subheader("🔥 AI 強勢股 Top 10")

    if result.empty:
        st.warning("目前掃描結果為空：可能是 stock_pool 代號都不在該市場，或 API 暫時無回傳。")
    else:
        st.dataframe(result, use_container_width=True)
