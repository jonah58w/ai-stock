import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

# =============================
# 0) 欄位正規化（Cloud 防爆核心）
# =============================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    # MultiIndex -> 找到含 OHLCV 的層
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

    # 2313.TW Close 這種欄位 -> 映射成 Close/Open...
    if any(isinstance(c, str) and ("close" in c.lower() or "open" in c.lower()) for c in df.columns):
        rename = {}
        for c in df.columns:
            s = str(c).strip().lower()
            if "open" in s: rename[c] = "Open"
            elif "high" in s: rename[c] = "High"
            elif "low" in s: rename[c] = "Low"
            elif "adj" in s and "close" in s: rename[c] = "Adj Close"
            elif "close" in s: rename[c] = "Close"
            elif "volume" in s or "vol" in s: rename[c] = "Volume"
            else: rename[c] = c
        df = df.rename(columns=rename)

    # 去重複欄位（避免 df["Close"] 變 DataFrame）
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 極端狀況：Close 仍被當成 DataFrame（保底硬轉）
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns and isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]

    return df

# =============================
# 1) 下載資料（Cloud 友好）
# =============================
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

# =============================
# 2) 技術指標計算（MACD/KD/BB/ATR/ADX/MA）
# =============================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("數據為空")

    df = normalize_ohlcv(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}；目前欄位：{df.columns.tolist()}")

    df = df.copy()

    # MA
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
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9 + 1e-9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    # ATR
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ✅ 這行就是你 Cloud 爆點：現在保證 df["Close"] 為 Series
    df["ATR_pct"] = (df["ATR"] / df["Close"]) * 100

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

    # Volume MA
    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()

    return df

# =============================
# 3) 趨勢週期識別（盤整/啟動/加速/末端）
# =============================
def detect_trend_stage(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "資料不足"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    deviation = (latest["Close"] / latest["EMA20"] - 1) * 100

    macd_cross_up = (prev["MACD_DIF"] < prev["MACD_Signal"]) and (latest["MACD_DIF"] > latest["MACD_Signal"])
    kd_cross_up = (prev["KD_K"] < prev["KD_D"]) and (latest["KD_K"] > latest["KD_D"])

    ma_bull = (latest["SMA5"] > latest["SMA10"] > latest["SMA20"])
    bb_break_upper = latest["Close"] > latest["BB_upper"]

    adx = float(latest.get("ADX", 0) or 0)
    kd = float(latest.get("KD_K", 0) or 0)

    # 末端期（Blow-off）
    if deviation > 20 and kd > 90 and adx > 55:
        return "末端期"

    # 啟動期（Breakout Expansion）：交叉 + 匯集後向上 + 突破布林
    ma_converge = abs(latest["SMA5"] - latest["SMA20"]) / (latest["Close"] + 1e-9) < 0.03
    if macd_cross_up and kd_cross_up and ma_bull and ma_converge and bb_break_upper and 20 < adx < 40 and deviation < 12:
        return "啟動期"

    # 加速期（Trend Expansion）：多頭排列 + ADX 強 + 偏離合理
    if ma_bull and adx >= 40 and 12 <= deviation <= 20:
        return "加速期"

    # 盤整期（Range）
    return "盤整期"

# =============================
# 4) 交易引擎（依週期產生行動指令）
# =============================
def cycle_trade_engine(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    stage = detect_trend_stage(df)

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    atr = float(latest["ATR"])
    atr_pct = float(latest["ATR_pct"])
    adx = float(latest["ADX"])
    kd = float(latest["KD_K"])
    macd_dif = float(latest["MACD_DIF"])
    macd_sig = float(latest["MACD_Signal"])

    deviation = (price / (ema20 + 1e-9) - 1) * 100

    # 近端回踩區（避免踏空）
    pullback_05atr = round(price - 0.5 * atr, 2)
    pullback_1atr = round(price - 1.0 * atr, 2)

    # 回檔補貨核心區
    core_zone = round(ema20, 2)

    # 風險線（簡化）：EMA20 下方 1 ATR（可再調）
    risk_line = round(ema20 - 1.0 * atr, 2)

    # 行動建議
    if stage == "啟動期":
        action = "🚀 啟動期：可先買 40%，回踩不破壓力/EMA20 再加碼"
        plan = [
            f"第一筆：突破共振成立時買 40%",
            f"第二筆：回踩 0.5ATR（{pullback_05atr}）不破，買 30%",
            f"第三筆：回踩 EMA20（{core_zone}）附近站回，買 30%",
        ]
    elif stage == "加速期":
        action = "🔥 加速期：不重倉追價；用『小倉突破 + 回踩加碼』"
        plan = [
            f"試單：突破共振成立可買 10~20%",
            f"加碼1：回踩 0.5ATR（{pullback_05atr}）轉強再加",
            f"加碼2：回踩 1ATR（{pullback_1atr}）不破再加",
            f"主防守：跌破風險線（{risk_line}）視為趨勢假設失效",
        ]
    elif stage == "末端期":
        action = "⚠️ 末端期：禁止追高；以減碼/停利為主，等待重新築底"
        plan = [
            f"避免新增多單",
            f"保護利潤：可用 EMA20（{core_zone}）或 1ATR 停損/移動停利",
            f"風險線：{risk_line}",
        ]
    elif stage == "盤整期":
        action = "🟦 盤整期：以等待或區間策略為主（非趨勢追價）"
        plan = [
            f"趨勢單不追價",
            f"等啟動期共振訊號（MACD+KD+均線+布林突破）再切換策略",
        ]
    else:
        action = "資料不足"
        plan = []

    return {
        "stage": stage,
        "price": round(price, 2),
        "ema20": round(ema20, 2),
        "atr": round(atr, 2),
        "atr_pct": round(atr_pct, 2),
        "adx": round(adx, 1),
        "kd": round(kd, 1),
        "macd": f"{macd_dif:.2f}/{macd_sig:.2f}",
        "deviation": round(deviation, 2),
        "pullback_05atr": pullback_05atr,
        "pullback_1atr": pullback_1atr,
        "core_zone": core_zone,
        "risk_line": risk_line,
        "action": action,
        "plan": plan,
    }

# =============================
# 5) Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="AI Cycle Trading Engine PRO", layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO（週期交易引擎版 / Cloud 防爆）")

    code = st.text_input("股票代碼", "2313").strip()
    if not code:
        code = "2313"
    symbol = code if (".TW" in code.upper() or ".TWO" in code.upper()) else f"{code}.TW"
    st.caption(f"完整代碼：{symbol}")

    df_raw = download_data(symbol)
    if df_raw is None or df_raw.empty:
        st.error("❌ 無法下載資料（可能代碼錯誤或 yfinance 被擋）")
        st.stop()

    # Debug：看原始欄位（必要時可打開）
    with st.expander("🔍 Debug：yfinance 原始欄位"):
        st.write(df_raw.columns)

    df = calculate_indicators(df_raw).dropna().copy()
    df = df.reset_index()

    latest = df.iloc[-1]

    # 指標摘要
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{latest['Close']:.2f}")
    c2.metric("EMA20", f"{latest['EMA20']:.2f}")
    c3.metric("ADX", f"{latest['ADX']:.1f}")
    c4.metric("ATR%", f"{latest['ATR_pct']:.2f}%")

    engine = cycle_trade_engine(df.set_index("Date"))

    st.subheader("📊 週期判斷")
    st.metric("當前週期階段", engine["stage"])
    st.caption(f"乖離率：{engine['deviation']:+.2f}%｜KD：{engine['kd']}｜MACD(DIF/Signal)：{engine['macd']}")

    st.subheader("🧠 引擎決策")
    st.info(engine["action"])

    st.subheader("🎯 具體操作計畫（自動）")
    for x in engine["plan"]:
        st.write(f"- {x}")

    st.subheader("📌 關鍵價位（自動）")
    st.write(f"近端回踩 0.5 ATR：**{engine['pullback_05atr']}**")
    st.write(f"近端回踩 1.0 ATR：**{engine['pullback_1atr']}**")
    st.write(f"核心回檔區（EMA20）：**{engine['core_zone']}**")
    st.warning(f"風險線（趨勢假設失效）：**{engine['risk_line']}**")

    # 圖表
    st.subheader("📈 K線 + 均線 + 布林")
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="BB Lower"))

    fig.add_hline(y=engine["pullback_05atr"], line_dash="dot", annotation_text="0.5ATR Pullback", opacity=0.6)
    fig.add_hline(y=engine["pullback_1atr"], line_dash="dot", annotation_text="1ATR Pullback", opacity=0.6)
    fig.add_hline(y=engine["core_zone"], line_dash="dash", annotation_text="EMA20 Core", opacity=0.7)
    fig.add_hline(y=engine["risk_line"], line_dash="dash", annotation_text="Risk Line", opacity=0.8)

    fig.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧾 指標快照（最近 10 筆）")
    cols = ["Date","Close","SMA5","SMA10","SMA20","EMA20","EMA50","ADX","KD_K","ATR_pct","BB_upper","BB_lower"]
    st.dataframe(df[cols].tail(10).round(2), use_container_width=True)

    st.caption("⚠️ 本工具為趨勢週期判斷與交易計畫建議，不構成投資建議。")

if __name__ == "__main__":
    main()
