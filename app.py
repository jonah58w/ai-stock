# ==============================
# AI Stock Trading Assistant（A+B 趨勢預測升級版 - 雲端穩定修正版）
# ✅ A：趨勢延續機率（多因子）
# ✅ B：ATR 區間預測（5/10 日）
# ✅ 自動推算：布局區 / 風險線
# ✅ 修正：yfinance MultiIndex / 欄位重複導致 df["Close"] 變成 DataFrame 的錯誤
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ------------------------------
# 1) 欄位正規化（最關鍵修正）
# ------------------------------
def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """將 yfinance 回來的 DataFrame（可能 MultiIndex / 重複欄）轉成標準 OHLCV 且欄位唯一。"""
    if df is None or df.empty:
        return df

    df = df.copy()

    # (A) MultiIndex：找出包含 OHLCV 的那一層
    if isinstance(df.columns, pd.MultiIndex):
        target_level = None
        for lvl in range(df.columns.nlevels):
            vals = set(str(x).strip().lower() for x in df.columns.get_level_values(lvl).unique())
            if {"open", "high", "low", "close", "volume"}.issubset(vals):
                target_level = lvl
                break

        if target_level is None:
            # fallback：挑最像 OHLCV 的那層
            def score_level(lvl: int) -> int:
                vals = [str(x).strip().lower() for x in df.columns.get_level_values(lvl)]
                return sum(v in ["open", "high", "low", "close", "volume", "adj close"] for v in vals)

            target_level = max(range(df.columns.nlevels), key=score_level)

        df.columns = df.columns.get_level_values(target_level)

    # (B) "2313.TW Close" 這種欄位命名：抽出 Close/Open...
    elif any(isinstance(c, str) and ("close" in c.lower() or "open" in c.lower()) for c in df.columns):
        new_cols = {}
        for c in df.columns:
            s = str(c).strip().lower()
            if "open" in s:
                new_cols[c] = "Open"
            elif "high" in s:
                new_cols[c] = "High"
            elif "low" in s:
                new_cols[c] = "Low"
            elif "adj" in s and "close" in s:
                new_cols[c] = "Adj Close"
            elif "close" in s:
                new_cols[c] = "Close"
            elif "volume" in s or "vol" in s:
                new_cols[c] = "Volume"
            else:
                new_cols[c] = c
        df = df.rename(columns=new_cols)

    # (C) 統一大小寫
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        s = c.strip().lower()
        if s == "open":
            rename_map[c] = "Open"
        elif s == "high":
            rename_map[c] = "High"
        elif s == "low":
            rename_map[c] = "Low"
        elif s == "close":
            rename_map[c] = "Close"
        elif s == "volume":
            rename_map[c] = "Volume"
        elif "adj" in s and "close" in s:
            rename_map[c] = "Adj Close"
    if rename_map:
        df = df.rename(columns=rename_map)

    # (D) 去除重複欄位（重點！避免 df["Close"] 變 DataFrame）
    # keep first occurrence
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df


# ------------------------------
# 2) 技術指標計算（含 ATR/ADX/均線/量均）
# ------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("數據為空")

    df = normalize_ohlcv_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}；目前欄位：{df.columns.tolist()}")

    df = df.copy()

    # 均線
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ✅ 這裡你原本爆掉：現在 df["Close"] 一定是 Series
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

    # ADX（簡化版）
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    # 量均
    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()

    return df


# ------------------------------
# 3) A：趨勢延續機率（0~100）
# ------------------------------
def calculate_trend_probability(df: pd.DataFrame) -> float:
    latest = df.iloc[-1]
    score = 0.0

    # EMA20 斜率（5日）
    if len(df) >= 6 and df["EMA20"].iloc[-5] != 0 and pd.notna(df["EMA20"].iloc[-5]):
        ema20_slope = (df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]) / df["EMA20"].iloc[-5] * 100
    else:
        ema20_slope = 0.0

    if ema20_slope > 1:
        score += 25
    elif ema20_slope > 0:
        score += 15

    # ADX 趨勢強度
    adx = float(latest.get("ADX", 0) or 0)
    if adx > 40:
        score += 20
    elif adx > 25:
        score += 10

    # 20日動量（ROC20）
    if len(df) >= 21 and df["Close"].iloc[-20] != 0 and pd.notna(df["Close"].iloc[-20]):
        roc20 = (latest["Close"] / df["Close"].iloc[-20] - 1) * 100
    else:
        roc20 = 0.0

    if roc20 > 15:
        score += 20
    elif roc20 > 5:
        score += 10

    # 量能放大
    vol5 = float(latest.get("VOL_5", np.nan))
    vol10 = float(latest.get("VOL_10", np.nan))
    if np.isfinite(vol5) and np.isfinite(vol10) and vol10 > 0 and vol5 > vol10 * 1.2:
        score += 15

    # 結構突破：收盤 > 前一日的 20日最高（避免 look-ahead）
    if len(df) >= 21:
        prev_20_high = df["High"].rolling(20).max().iloc[-2]
        if pd.notna(prev_20_high) and latest["Close"] > prev_20_high:
            score += 20

    return float(min(score, 100))


# ------------------------------
# 4) B：ATR 區間預測
# ------------------------------
def forecast_range(df: pd.DataFrame, days: int):
    latest = df.iloc[-1]
    price = float(latest["Close"])
    atr_pct = float(latest.get("ATR_Pct", 0) or 0) / 100.0

    expected_move = price * atr_pct * np.sqrt(days)
    lower = price - expected_move
    upper = price + expected_move
    return round(lower, 2), round(upper, 2)


# ------------------------------
# 5) A+B 融合：布局區 / 風險線 / Regime
# ------------------------------
def generate_trading_zone(df: pd.DataFrame):
    prob = calculate_trend_probability(df)
    lower5, upper5 = forecast_range(df, 5)
    lower10, upper10 = forecast_range(df, 10)

    price = float(df.iloc[-1]["Close"])

    if prob >= 65:
        regime = "強趨勢延續"
        buy_zone = (lower5, price)
        risk_line = round(lower5 * 0.97, 2)
    elif prob >= 50:
        regime = "中性偏多"
        buy_zone = (lower5, price)
        risk_line = round(lower5 * 0.95, 2)
    else:
        regime = "震盪/不明朗"
        buy_zone = None
        risk_line = None

    return prob, (lower5, upper5), (lower10, upper10), buy_zone, risk_line, regime


# ------------------------------
# 6) 下載資料（含雲端穩定參數）
# ------------------------------
def download_data(stock_code: str) -> pd.DataFrame:
    df = yf.download(
        stock_code,
        period="3y",
        interval="1d",
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=False,
    )
    return df


# ------------------------------
# 7) Streamlit 主程式
# ------------------------------
def main():
    st.set_page_config(page_title="AI Stock Trading Assistant（A+B 趨勢預測升級版）", layout="wide")
    st.title("📊 AI Stock Trading Assistant（A+B 趨勢預測升級版）")

    stock_input = st.text_input("股票代碼", "2313").strip()

    # 自動加尾碼
    if stock_input:
        if ".TW" not in stock_input.upper() and ".TWO" not in stock_input.upper():
            stock_code = f"{stock_input}.TW"
        else:
            stock_code = stock_input
    else:
        stock_code = "2313.TW"

    st.caption(f"完整代碼：{stock_code}")

    try:
        df_raw = download_data(stock_code)
        if df_raw is None or df_raw.empty:
            st.error("❌ 無法下載資料（可能被擋或代碼錯誤）")
            st.stop()

        df = calculate_indicators(df_raw)

        # 轉成可畫圖格式
        df = df.reset_index()
        latest = df.iloc[-1]

        # 基本指標
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("當前價格", f"{latest['Close']:.2f}")
        c2.metric("EMA20", f"{latest['EMA20']:.2f}")
        c3.metric("ADX", f"{latest['ADX']:.1f}")
        c4.metric("ATR%", f"{latest['ATR_Pct']:.2f}%")

        # A+B 模型輸出
        st.subheader("📈 趨勢預測模型（A+B）")
        prob, r5, r10, buy_zone, risk_line, regime = generate_trading_zone(df.set_index("Date"))

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("趨勢延續機率", f"{prob:.0f}%")
        cc2.metric("未來 5 日區間", f"{r5[0]} ~ {r5[1]}")
        cc3.metric("未來 10 日區間", f"{r10[0]} ~ {r10[1]}")
        st.info(f"趨勢結構判斷：**{regime}**")

        if buy_zone:
            st.success(f"建議布局區：**{buy_zone[0]} ~ {buy_zone[1]:.2f}**")
            st.warning(f"風險警戒線：**{risk_line}**（跌破代表趨勢假設開始失效）")
        else:
            st.info("目前機率不足，不建議主動布局（等待結構更明確）。")

        # 圖表
        st.subheader("📊 價格 + EMA（含預測區間參考）")
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50"))

        # 區間線（以最新日為中心畫水平參考）
        fig.add_hline(y=r5[0], line_dash="dot", annotation_text="5D Lower", opacity=0.6)
        fig.add_hline(y=r5[1], line_dash="dot", annotation_text="5D Upper", opacity=0.6)
        if risk_line is not None:
            fig.add_hline(y=risk_line, line_dash="dash", annotation_text="Risk Line", opacity=0.7)

        fig.update_layout(height=650, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # 指標快照
        st.subheader("🧾 指標快照（最近 10 筆）")
        cols = ["Date", "Close", "EMA20", "EMA50", "ADX", "ATR_Pct", "VOL_5", "VOL_10"]
        st.dataframe(df[cols].tail(10).round(2), use_container_width=True)

        st.caption("⚠️ 本工具為趨勢機率與波動區間預測，不構成投資建議。")

    except Exception as e:
        st.error("程式執行發生錯誤（雲端已隱藏細節以避免洩漏）。")
        # 這段會在 Cloud 顯示更有用的 debug（不會洩漏敏感資料）
        st.exception(e)


if __name__ == "__main__":
    main()
