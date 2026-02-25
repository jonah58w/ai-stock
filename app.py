import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

REQUIRED = ["Open","High","Low","Close","Volume"]

# -----------------------
# 1) 超強欄位整理（關鍵）
# -----------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    # A) MultiIndex -> 抽出包含 OHLCV 的那一層
    if isinstance(df.columns, pd.MultiIndex):
        pick = None
        for lvl in range(df.columns.nlevels):
            vals = set(str(x).strip().lower() for x in df.columns.get_level_values(lvl).unique())
            if {"open","high","low","close","volume"}.issubset(vals):
                pick = lvl
                break
        if pick is None:
            pick = df.columns.nlevels - 1
        df.columns = df.columns.get_level_values(pick)

    # B) 2313.TW Close 這種 -> 映射成 Close/Open...
    if any(isinstance(c,str) and ("close" in c.lower() or "open" in c.lower()) for c in df.columns):
        rename = {}
        for c in df.columns:
            s = str(c).strip().lower()
            if "open" in s: rename[c] = "Open"
            elif "high" in s: rename[c] = "High"
            elif "low" in s: rename[c] = "Low"
            elif "close" in s and "adj" not in s: rename[c] = "Close"
            elif "volume" in s or "vol" in s: rename[c] = "Volume"
            elif "adj" in s and "close" in s: rename[c] = "Adj Close"
            else: rename[c] = c
        df = df.rename(columns=rename)

    # C) 去掉重複欄位（避免 df["Close"] 變 DataFrame）
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # D) 若 Close 仍是 DataFrame（極少數狀況），硬轉成 Series
    if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]
    if "Open" in df.columns and isinstance(df["Open"], pd.DataFrame):
        df["Open"] = df["Open"].iloc[:, 0]
    if "High" in df.columns and isinstance(df["High"], pd.DataFrame):
        df["High"] = df["High"].iloc[:, 0]
    if "Low" in df.columns and isinstance(df["Low"], pd.DataFrame):
        df["Low"] = df["Low"].iloc[:, 0]
    if "Volume" in df.columns and isinstance(df["Volume"], pd.DataFrame):
        df["Volume"] = df["Volume"].iloc[:, 0]

    return df


# -----------------------
# 2) 下載資料（Cloud 友好）
# -----------------------
def download_data(symbol: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        period="3y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )


# -----------------------
# 3) 指標計算（含 KD/ADX/ATR）
# -----------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("數據為空")

    df = normalize_ohlcv(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}；目前欄位：{df.columns.tolist()}")

    df = df.copy()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100  # ✅ 這行現在不會炸

    # ADX（簡化）
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    # KD
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9 + 1e-9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    # Volume MA
    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()

    return df


# -----------------------
# 4) A：趨勢機率
# -----------------------
def trend_probability(df: pd.DataFrame) -> float:
    latest = df.iloc[-1]
    score = 0.0

    # EMA20 slope
    if len(df) >= 6 and df["EMA20"].iloc[-5] != 0:
        slope = (df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]) / df["EMA20"].iloc[-5] * 100
    else:
        slope = 0.0

    if slope > 1: score += 25
    elif slope > 0: score += 15

    adx = float(latest.get("ADX", 0) or 0)
    if adx > 40: score += 20
    elif adx > 25: score += 10

    if len(df) >= 21 and df["Close"].iloc[-20] != 0:
        roc20 = (latest["Close"] / df["Close"].iloc[-20] - 1) * 100
    else:
        roc20 = 0.0
    if roc20 > 15: score += 20
    elif roc20 > 5: score += 10

    vol5 = float(latest.get("VOL_5", np.nan))
    vol10 = float(latest.get("VOL_10", np.nan))
    if np.isfinite(vol5) and np.isfinite(vol10) and vol10 > 0 and vol5 > vol10 * 1.2:
        score += 15

    if len(df) >= 21:
        prev_20_high = df["High"].rolling(20).max().iloc[-2]
        if pd.notna(prev_20_high) and latest["Close"] > prev_20_high:
            score += 20

    return float(min(score, 100))


# -----------------------
# 5) B：ATR 區間
# -----------------------
def forecast_range(df: pd.DataFrame, days: int):
    latest = df.iloc[-1]
    price = float(latest["Close"])
    atr_pct = float(latest.get("ATR_Pct", 0) or 0) / 100.0
    move = price * atr_pct * np.sqrt(days)
    return round(price - move, 2), round(price + move, 2)


# -----------------------
# 6) 分批布局 + 過熱 + 評分
# -----------------------
def build_strategy(df: pd.DataFrame):
    prob = trend_probability(df)
    r5 = forecast_range(df, 5)
    r10 = forecast_range(df, 10)

    latest = df.iloc[-1]
    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    adx = float(latest["ADX"])
    kd = float(latest["KD_K"])
    atr_pct = float(latest["ATR_Pct"])

    # 三層布局
    safe = round(ema20, 2)
    standard = r5[0]
    aggressive = r10[0]

    # 過熱判斷：乖離過大 + KD 過熱
    deviation = (price / ema20 - 1) * 100 if ema20 else 0
    overheat = (deviation > 20 and kd > 85)

    # 進場評分（0~100）
    score = prob
    if deviation < 10: score += 10
    if adx > 40: score += 10
    if overheat: score -= 20
    score = max(0, min(100, score))

    # 動態風險線：以 5日下緣再下移一點（更貼近波動）
    risk_line = round(r5[0] * 0.97, 2) if prob >= 50 else None

    return {
        "prob": prob,
        "r5": r5,
        "r10": r10,
        "safe": safe,
        "standard": standard,
        "aggressive": aggressive,
        "risk_line": risk_line,
        "overheat": overheat,
        "entry_score": score,
        "deviation": deviation,
        "adx": adx,
        "kd": kd,
        "atr_pct": atr_pct
    }


# -----------------------
# 7) Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="AI Stock Trading Assistant PRO", layout="wide")
    st.title("📊 AI Stock Trading Assistant（A+B PRO：分批布局/過熱/評分）")

    code = st.text_input("股票代碼", "2313").strip()
    if not code:
        code = "2313"
    symbol = code if (".TW" in code.upper() or ".TWO" in code.upper()) else f"{code}.TW"
    st.caption(f"完整代碼：{symbol}")

    df_raw = download_data(symbol)

    # 🔍 Debug：如果你要看 yfinance 原始欄位，打開這個
    with st.expander("🔍 Debug：原始欄位（yfinance 回來長什麼樣）"):
        st.write(df_raw.columns)

    if df_raw is None or df_raw.empty:
        st.error("❌ 下載不到資料（可能代碼錯或被擋）")
        st.stop()

    df = calculate_indicators(df_raw).reset_index()
    latest = df.iloc[-1]

    # 基本資訊
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Close", f"{latest['Close']:.2f}")
    c2.metric("EMA20", f"{latest['EMA20']:.2f}")
    c3.metric("ADX", f"{latest['ADX']:.1f}")
    c4.metric("ATR%", f"{latest['ATR_Pct']:.2f}%")

    st.subheader("🧠 A+B 趨勢與區間預測 + 交易策略")
    strat = build_strategy(df.set_index("Date"))

    c5,c6,c7 = st.columns(3)
    c5.metric("趨勢延續機率", f"{strat['prob']:.0f}%")
    c6.metric("未來 5 日區間", f"{strat['r5'][0]} ~ {strat['r5'][1]}")
    c7.metric("未來 10 日區間", f"{strat['r10'][0]} ~ {strat['r10'][1]}")

    st.subheader("🎯 分批布局建議（自動）")
    st.write(f"**安全層（EMA20 附近）**：{strat['safe']}")
    st.write(f"**標準層（5日下緣）**：{strat['standard']}")
    st.write(f"**積極層（10日下緣）**：{strat['aggressive']}")
    if strat["risk_line"] is not None:
        st.warning(f"**風險警戒線**：{strat['risk_line']}（跌破代表趨勢假設開始失效）")

    st.subheader("📊 當前是否適合進場")
    st.metric("進場評分（0-100）", f"{strat['entry_score']:.0f}")
    st.caption(f"乖離率：{strat['deviation']:+.2f}%｜KD：{strat['kd']:.1f}｜ADX：{strat['adx']:.1f}")

    if strat["overheat"]:
        st.error("⚠️ 過熱警示：目前偏追高區，建議等回檔到標準/安全層再佈局。")
    elif strat["entry_score"] >= 75:
        st.success("✅ 條件良好：可用『分批』方式執行。")
    elif strat["entry_score"] >= 60:
        st.info("🟡 條件尚可：建議小倉位 + 等回檔確認。")
    else:
        st.info("⚪ 條件不足：以等待為主。")

    # 圖表
    st.subheader("📈 K線 + EMA（含區間/風險線）")
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50"))
    fig.add_hline(y=strat["r5"][0], line_dash="dot", annotation_text="5D Lower", opacity=0.6)
    fig.add_hline(y=strat["r5"][1], line_dash="dot", annotation_text="5D Upper", opacity=0.6)
    if strat["risk_line"] is not None:
        fig.add_hline(y=strat["risk_line"], line_dash="dash", annotation_text="Risk Line", opacity=0.7)
    fig.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # 快照
    st.subheader("🧾 指標快照（最近 10 筆）")
    st.dataframe(df[["Date","Close","EMA20","EMA50","ADX","KD_K","ATR_Pct","VOL_5","VOL_10"]].tail(10).round(2),
                 use_container_width=True)

    st.caption("⚠️ 本工具為趨勢機率與波動區間預測，不構成投資建議。")


if __name__ == "__main__":
    main()
