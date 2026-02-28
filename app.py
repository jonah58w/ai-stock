# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雲端穩定資料版 V6.3）
# ✅ 資料來源優先順序：FinMind -> yfinance -> Stooq -> CSV upload
# ✅ 自動補台股尾碼：數字代號 -> .TW / .TWO fallback
# ✅ 逐路診斷表：告訴你哪一路失敗、為什麼
# ⚠️ 僅做資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。

from __future__ import annotations
import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf
from pandas_datareader import data as pdr

# -----------------------------
# UI / Page
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant (TW) - V6.3", layout="wide")
st.title("📈 AI Stock Trading Assistant（台股分析專業版 / 雲端穩定 V6.3）")

# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV to columns: Date, Open, High, Low, Close, Volume (Date as datetime)."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # If index is datetime, move to Date column
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()

    # Common column name mapping
    col_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ["date", "datetime", "time"]:
            col_map[c] = "Date"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc in ["close", "adj close", "adjclose", "adj_close"]:
            # Prefer Close; if both Close and Adj Close exist, keep Close
            if "Close" not in df.columns:
                col_map[c] = "Close"
            else:
                # leave as-is
                pass
        elif lc in ["volume", "vol"]:
            col_map[c] = "Volume"

    df = df.rename(columns=col_map)

    # If Adj Close present but Close missing
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame()

    # Coerce types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)

    return df[REQUIRED_COLS].reset_index(drop=True)

def _make_ticker_candidates(raw: str) -> list[str]:
    """Given user input, return candidate tickers to try for different sources."""
    raw = (raw or "").strip().upper()
    if not raw:
        return []

    # If user already includes suffix, try as-is first
    if "." in raw:
        return [raw]

    # If digits only => Taiwan stock code; try .TW then .TWO
    if raw.isdigit():
        return [f"{raw}.TW", f"{raw}.TWO", raw]  # raw last (some sources might accept)
    # Non-digit: return as-is
    return [raw]

def _diagnosis_row(source: str, result: str, url: str, note: str = "") -> dict:
    return {"source": source, "result": result, "url": url, "note": note}

# -----------------------------
# Data Loaders
# -----------------------------
def load_finmind(code_digits: str, start: str, end: str):
    """
    FinMind: needs finmind installed.
    Only works for Taiwan codes. Expect digits, e.g., '6274'
    """
    try:
        from FinMind.data import DataLoader
    except Exception as e:
        return pd.DataFrame(), _diagnosis_row("FinMind", "NO_MODULE", f"FinMind:{code_digits}", note=str(e)[:120])

    try:
        dl = DataLoader()
        # FinMind uses TaiwanStockPrice with stock_id as digits
        df = dl.taiwan_stock_daily(stock_id=code_digits, start_date=start, end_date=end)
        if df is None or df.empty:
            return pd.DataFrame(), _diagnosis_row("FinMind", "EMPTY", f"FinMind:{code_digits}", note="No data")
        # FinMind columns: date, open, max, min, close, Trading_Volume...
        df2 = pd.DataFrame({
            "Date": pd.to_datetime(df["date"]),
            "Open": df["open"],
            "High": df.get("max", df.get("high", np.nan)),
            "Low": df.get("min", df.get("low", np.nan)),
            "Close": df["close"],
            "Volume": df.get("Trading_Volume", df.get("Trading_turnover", df.get("volume", 0))),
        })
        df2 = _normalize_ohlcv(df2)
        if df2.empty:
            return pd.DataFrame(), _diagnosis_row("FinMind", "BAD_FORMAT", f"FinMind:{code_digits}", note="Format normalization failed")
        return df2, _diagnosis_row("FinMind", "OK", f"FinMind:{code_digits}")
    except Exception as e:
        return pd.DataFrame(), _diagnosis_row("FinMind", "ERROR", f"FinMind:{code_digits}", note=str(e)[:120])

def load_yfinance(ticker: str, start: str, end: str):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        df = _normalize_ohlcv(df)
        if df.empty:
            return pd.DataFrame(), _diagnosis_row("YF", "EMPTY", "yfinance", note=f"{ticker}")
        return df, _diagnosis_row("YF", "OK", "yfinance", note=f"{ticker}")
    except Exception as e:
        return pd.DataFrame(), _diagnosis_row("YF", "ERROR", "yfinance", note=str(e)[:120])

def load_stooq(ticker: str, start: str, end: str):
    """
    Stooq via pandas-datareader:
    For Taiwan, often works with '6274.TW' or '6274.TWO'
    """
    try:
        # Stooq reader ignores start/end sometimes; we filter after
        df = pdr.DataReader(ticker, "stooq")
        df = df.reset_index().rename(columns={"Date": "Date"})
        df = _normalize_ohlcv(df)
        if df.empty:
            return pd.DataFrame(), _diagnosis_row("STOOQ", "EMPTY", f"https://stooq.com/q/d/l/?s={ticker}", note=ticker)

        # Filter dates
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        df = df[(df["Date"] >= s) & (df["Date"] <= e)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(), _diagnosis_row("STOOQ", "EMPTY", f"https://stooq.com/q/d/l/?s={ticker}", note="filtered empty")
        return df, _diagnosis_row("STOOQ", "OK", f"https://stooq.com/q/d/l/?s={ticker}", note=ticker)
    except Exception as e:
        return pd.DataFrame(), _diagnosis_row("STOOQ", "ERROR", f"https://stooq.com/q/d/l/?s={ticker}", note=str(e)[:120])

# -----------------------------
# Main UI Inputs
# -----------------------------
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    raw_code = st.text_input("輸入台股代號 / 代碼（例：2330 或 6274 或 0050）", value="6274")
with colB:
    days = st.selectbox("回溯天數", [180, 365, 730, 1500], index=1)
with colC:
    run = st.button("RUN 取得資料", type="primary")

# CSV upload backup
st.caption("（選用）上傳 export.csv 作為備援資料源（Date/Open/High/Low/Close/Volume）")
upload = st.file_uploader("Upload export.csv", type=["csv"])

# -----------------------------
# Date range
# -----------------------------
end_dt = dt.date.today()
start_dt = end_dt - dt.timedelta(days=int(days))
start = start_dt.strftime("%Y-%m-%d")
end = (end_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")  # include today

# -----------------------------
# Load on RUN
# -----------------------------
if run:
    diag = []
    data = pd.DataFrame()

    candidates = _make_ticker_candidates(raw_code)
    code_digits = raw_code.strip()
    is_digits = code_digits.isdigit()

    # 1) FinMind first (digits only)
    if is_digits:
        df_fm, d = load_finmind(code_digits, start, end)
        diag.append(d)
        if not df_fm.empty:
            data = df_fm

    # 2) yfinance fallback
    if data.empty:
        for t in candidates:
            df_yf, d = load_yfinance(t, start, end)
            diag.append(d)
            if not df_yf.empty:
                data = df_yf
                break

    # 3) Stooq fallback
    if data.empty:
        for t in candidates:
            df_sq, d = load_stooq(t, start, end)
            diag.append(d)
            if not df_sq.empty:
                data = df_sq
                break

    # 4) CSV upload fallback
    if data.empty and upload is not None:
        try:
            raw = upload.read()
            df_csv = pd.read_csv(io.BytesIO(raw))
            df_csv = _normalize_ohlcv(df_csv)
            if df_csv.empty:
                diag.append(_diagnosis_row("CSV", "BAD_FORMAT", "upload", note="Need Date/Open/High/Low/Close/Volume"))
            else:
                data = df_csv
                diag.append(_diagnosis_row("CSV", "OK", "upload"))
        except Exception as e:
            diag.append(_diagnosis_row("CSV", "ERROR", "upload", note=str(e)[:120]))

    # Summary banner
    if data.empty:
        st.error("❌ 無法取得資料（所有備援來源都失敗）。請稍後再試，或改用 CSV 上傳備援。")
    else:
        st.success(f"✅ 取得資料成功：{len(data)} 筆（{data['Date'].min().date()} → {data['Date'].max().date()}）")

    # Diagnosis table
    st.subheader("🧩 逐路診斷（哪一路失敗、為什麼）")
    st.dataframe(pd.DataFrame(diag), use_container_width=True)

    # Show data
    if not data.empty:
        st.subheader("📊 OHLCV 資料預覽")
        st.dataframe(data.tail(30), use_container_width=True)

        # Simple chart
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price"
            )
        ])
        fig.update_layout(height=520, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.info("按上方 RUN 開始抓資料；若 Cloud 偶發抓不到，請直接上傳 export.csv（Date/Open/High/Low/Close/Volume）。")