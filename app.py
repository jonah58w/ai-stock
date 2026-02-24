from __future__ import annotations

import time
import math
import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# 基本設定
# =========================
st.set_page_config(layout="wide")

APP_TITLE = "🧠 AI 台股量化專業平台（官方資料源版）"

HEADERS_TWSE = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.twse.com.tw/",
    "Connection": "keep-alive",
}

HEADERS_TPEX = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.tpex.org.tw/",
    "Connection": "keep-alive",
}


# =========================
# 小工具
# =========================
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


def _req_json(url: str, headers: dict, retry: int = 3, sleep: float = 0.6):
    last = None
    for i in range(retry):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            last = (r.status_code, r.text[:200])
            if r.status_code == 200:
                return r.json(), last
            time.sleep(sleep * (i + 1))
        except Exception as e:
            last = ("EXCEPTION", str(e)[:200])
            time.sleep(sleep * (i + 1))
    return None, last


# =========================
# 中文名稱（可有可無，失敗不影響）
# =========================
@st.cache_data(ttl=86400)
def get_stock_name(code: str) -> str:
    code = _norm_code(code)
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        r = requests.get(url, headers=HEADERS_TWSE, timeout=15)
        data = r.json()
        for s in data:
            if s.get("Code") == code:
                return s.get("Name", "")
    except:
        pass
    return ""


# =========================
# 下載資料：TWSE（月） & TPEX（月）
# =========================
def _fetch_twse_month(code: str, yyyymm: str):
    # TWSE: date=YYYYMMDD (用01即可)
    date_str = f"{yyyymm}01"
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_str}&stockNo={code}"
    j, last = _req_json(url, HEADERS_TWSE)

    if not j or j.get("stat") != "OK":
        return None, last

    fields = j.get("fields", [])
    data = j.get("data", [])
    if not data:
        return None, last

    df = pd.DataFrame(data, columns=fields)
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
        return None, last

    def parse_roc_date(s):
        # 114/02/03
        p = str(s).split("/")
        if len(p) != 3:
            return pd.NaT
        y = int(p[0]) + 1911
        m = int(p[1])
        d = int(p[2])
        return pd.Timestamp(y, m, d)

    df["Date"] = df["Date"].apply(parse_roc_date)
    df = df.set_index("Date").sort_index()

    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].apply(_safe_float)
    df["Volume"] = df["Volume"].apply(_safe_int)

    df = df.dropna(subset=["Close"])
    if df.empty:
        return None, last
    return df[["Open", "High", "Low", "Close", "Volume"]].copy(), last


def _fetch_tpex_month(code: str, yyyymm: str):
    # TPEX: d=ROC/02  (114/02)
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    d_param = f"{_roc_year(y)}/{m:02d}"

    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?d={d_param}&stkno={code}"
    j, last = _req_json(url, HEADERS_TPEX)

    if not j:
        return None, last

    data = j.get("aaData") or j.get("data") or []
    if not data:
        return None, last

    rows = []
    for row in data:
        # 常見欄位: [日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數]
        if not row or len(row) < 7:
            continue
        date_s, vol_s = row[0], row[1]
        open_s, high_s, low_s, close_s = row[3], row[4], row[5], row[6]

        p = str(date_s).split("/")
        if len(p) != 3:
            continue
        yy = int(p[0]) + 1911
        mm = int(p[1])
        dd = int(p[2])
        date = pd.Timestamp(yy, mm, dd)

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
        return None, last

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None, last
    return df[["Open", "High", "Low", "Close", "Volume"]].copy(), last


@st.cache_data(ttl=600)
def download_data(code: str, period: str = "6mo", fixed_months: int = 12):
    """
    ✅ 固定抓 fixed_months（預設 12 個月） → 再依 period 切
    先 TWSE，再 TPEX
    """
    code = _norm_code(code)
    months_keep = _period_to_months(period)

    today = dt.date.today()
    end = today.replace(day=1)
    month_list = [(end - relativedelta(months=i)).strftime("%Y%m") for i in range(fixed_months)]
    month_list = list(reversed(month_list))

    # 先 TWSE
    parts = []
    last_dbg = None
    for yyyymm in month_list:
        dfm, dbg = _fetch_twse_month(code, yyyymm)
        last_dbg = dbg
        if dfm is not None and not dfm.empty:
            parts.append(dfm)

    if parts:
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        return df, "TWSE", last_dbg

    # 再 TPEX
    parts = []
    for yyyymm in month_list:
        dfm, dbg = _fetch_tpex_month(code, yyyymm)
        last_dbg = dbg
        if dfm is not None and not dfm.empty:
            parts.append(dfm)

    if parts:
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        cutoff = df.index.max() - relativedelta(months=months_keep)
        df = df[df.index >= cutoff]
        return df, "TPEX", last_dbg

    return None, None, last_dbg


# =========================
# 指標：RSI / KD / MACD / Bollinger / BIAS / 支撐壓力 / ATR
# =========================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMA
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # KD(9)
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = (df["Close"] - low9) / (high9 - low9) * 100
    df["K"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1/3, adjust=False).mean()

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    # Bollinger(20,2)
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    # BIAS(20)
    df["BIAS20"] = (df["Close"] - df["SMA20"]) / df["SMA20"] * 100

    # Volume MA
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # Support/Resistance (近60日)
    df["SUPPORT"] = df["Low"].rolling(60).min()
    df["RESIST"] = df["High"].rolling(60).max()

    # ATR(14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df


def ai_score(df: pd.DataFrame) -> int:
    """
    0~100：多指標共振（趨勢+動能+量能）
    """
    latest = df.iloc[-1]
    score = 0

    # 趨勢
    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA60"]) and latest["SMA20"] > latest["SMA60"]:
        score += 25
    if pd.notna(latest["SMA20"]) and latest["Close"] > latest["SMA20"]:
        score += 10

    # 動能
    if pd.notna(latest["RSI"]) and latest["RSI"] > 55:
        score += 15
    if pd.notna(latest["K"]) and pd.notna(latest["D"]) and latest["K"] > latest["D"]:
        score += 10
    if pd.notna(latest["MACD"]) and latest["MACD"] > 0:
        score += 10

    # 量能
    if pd.notna(latest["VOL_MA20"]) and latest["Volume"] > latest["VOL_MA20"]:
        score += 15

    # 突破/靠近壓力
    if pd.notna(latest["RESIST"]) and latest["Close"] >= latest["RESIST"]:
        score += 15

    return int(min(score, 100))


def future_buy_sell_zones(df: pd.DataFrame):
    """
    ✅ 只回傳「未來預估」買/賣區間（不標歷史點）
    邏輯：
    - 買點：靠近支撐 or 布林下軌 + RSI/KD 偏低
    - 賣點：靠近壓力 or 布林上軌 + RSI/KD 偏高
    """
    latest = df.iloc[-1]
    close = float(latest["Close"])

    support = float(latest["SUPPORT"]) if pd.notna(latest["SUPPORT"]) else None
    resist = float(latest["RESIST"]) if pd.notna(latest["RESIST"]) else None
    bb_low = float(latest["BB_LOW"]) if pd.notna(latest["BB_LOW"]) else None
    bb_up = float(latest["BB_UP"]) if pd.notna(latest["BB_UP"]) else None

    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    k = float(latest["K"]) if pd.notna(latest["K"]) else None
    d = float(latest["D"]) if pd.notna(latest["D"]) else None

    # Buy zone
    buy_center = None
    if support and bb_low:
        buy_center = min(support, bb_low)
    elif support:
        buy_center = support
    elif bb_low:
        buy_center = bb_low

    buy_zone = None
    buy_reason = []
    if buy_center:
        buy_zone = (buy_center * 0.99, buy_center * 1.01)
        if rsi is not None and rsi <= 40:
            buy_reason.append("RSI偏低")
        if k is not None and d is not None and k <= 25 and d <= 30:
            buy_reason.append("KD偏低")
        if bb_low and close <= bb_low * 1.03:
            buy_reason.append("接近布林下軌")
        if support and close <= support * 1.05:
            buy_reason.append("接近支撐")

    # Sell zone
    sell_center = None
    if resist and bb_up:
        sell_center = max(resist, bb_up)
    elif resist:
        sell_center = resist
    elif bb_up:
        sell_center = bb_up

    sell_zone = None
    sell_reason = []
    if sell_center:
        sell_zone = (sell_center * 0.99, sell_center * 1.01)
        if rsi is not None and rsi >= 65:
            sell_reason.append("RSI偏高")
        if k is not None and d is not None and k >= 75 and d >= 70:
            sell_reason.append("KD偏高")
        if bb_up and close >= bb_up * 0.97:
            sell_reason.append("接近布林上軌")
        if resist and close >= resist * 0.95:
            sell_reason.append("接近壓力")

    return buy_zone, buy_reason, sell_zone, sell_reason


# =========================
# Top10 掃描器（安全）
# =========================
def scan_top10(stock_pool: list[str], period: str) -> pd.DataFrame:
    rows = []
    for code in stock_pool:
        df, src, _dbg = download_data(code, period=period)
        if df is None or len(df) < 80:
            continue
        df = calc_indicators(df)
        score = ai_score(df)
        rows.append({"股票": _norm_code(code), "來源": src, "AI分數": score})

    out = pd.DataFrame(rows, columns=["股票", "來源", "AI分數"])
    if out.empty:
        return out
    return out.sort_values("AI分數", ascending=False).head(10)


# =========================
# UI
# =========================
st.title(APP_TITLE)

mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])
period = st.sidebar.selectbox("資料期間", ["3mo", "6mo", "12mo"], index=1)
show_debug = st.sidebar.checkbox("顯示下載除錯資訊（Debug）", value=False)

if mode == "單一股票分析":
    code = st.text_input("請輸入股票代號", "2330")
    code = _norm_code(code)

    df, src, dbg = download_data(code, period=period)

    if df is None:
        st.error("❌ 無法取得資料（TWSE/TPEX 皆無回傳）。請確認代號，或稍後再試。")
        if show_debug:
            st.write("Debug:", dbg)
        st.stop()

    name = get_stock_name(code)
    df = calc_indicators(df)

    score = ai_score(df)
    last = float(df["Close"].iloc[-1])

    buy_zone, buy_reason, sell_zone, sell_reason = future_buy_sell_zones(df)

    st.success(f"{name} ({code}) | Source: {src}")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI 共振分數", f"{score}/100")
    c2.metric("目前價格", round(last, 2))
    atr = df["ATR14"].iloc[-1]
    stop_loss = None if pd.isna(atr) else (last - 2 * float(atr))
    c3.metric("ATR 停損參考", "-" if stop_loss is None else round(stop_loss, 2))

    st.subheader("📌 未來預估買賣點（只給未來區間）")
    colA, colB = st.columns(2)
    with colA:
        if buy_zone:
            st.info(f"✅ 預估買點區間：{buy_zone[0]:.2f} ~ {buy_zone[1]:.2f}")
            if buy_reason:
                st.caption("條件： " + " / ".join(buy_reason))
        else:
            st.warning("買點區間：資料不足（需更多K線）")

    with colB:
        if sell_zone:
            st.info(f"✅ 預估賣點區間：{sell_zone[0]:.2f} ~ {sell_zone[1]:.2f}")
            if sell_reason:
                st.caption("條件： " + " / ".join(sell_reason))
        else:
            st.warning("賣點區間：資料不足（需更多K線）")

    st.subheader("📈 收盤價走勢")
    st.line_chart(df["Close"])

    if show_debug:
        st.subheader("🛠 Debug")
        st.write("最後一次請求狀態/片段：", dbg)

else:
    st.caption("Top10 掃描器：若結果為空，代表下載全部失敗（通常是來源被擋/403/暫時性）。")

    # 你可以先用小池驗證；要全市場我再幫你做「自動抓清單 + 加速」
    stock_pool = ["2330", "2317", "2454", "2303", "2382", "3037", "8046", "6274"]

    result = scan_top10(stock_pool, period=period)

    st.subheader("🔥 AI 強勢股 Top 10")
    if result.empty:
        st.warning("目前掃描結果為空（代表資料下載失敗）。建議先開啟 Debug 看是 403 還是空資料。")
    else:
        st.dataframe(result, use_container_width=True)
