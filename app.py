# ==================================================
# AI Cycle Trading Engine PRO v4 (A+B+C)
# Cloud-safe / Close-guaranteed / Resonance + Backtest
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

# -----------------------------
# 0) Robust OHLCV Normalizer
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    # 0-1) MultiIndex columns -> pick the level that contains OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        pick_level = None
        for lvl in range(df.columns.nlevels):
            vals = set(str(x).strip().lower() for x in df.columns.get_level_values(lvl).unique())
            if {"open", "high", "low", "close", "volume"}.issubset(vals) or {"adj close", "close"}.issubset(vals):
                pick_level = lvl
                break
        if pick_level is None:
            pick_level = df.columns.nlevels - 1
        df.columns = df.columns.get_level_values(pick_level)

    # 0-2) Normalize column names (string)
    def canon(name: str) -> str:
        s = str(name).strip().lower().replace("_", " ")
        s = " ".join(s.split())
        if s == "open" or s.endswith(" open") or " open" in s:
            return "Open"
        if s == "high" or s.endswith(" high") or " high" in s:
            return "High"
        if s == "low" or s.endswith(" low") or " low" in s:
            return "Low"
        if s == "close" or s.endswith(" close") or (("close" in s) and ("adj" not in s)):
            return "Close"
        if "adj" in s and "close" in s:
            return "Adj Close"
        if s == "volume" or "volume" in s or s.endswith(" vol") or " vol" in s:
            return "Volume"
        return str(name).strip()

    df.columns = [canon(c) for c in df.columns]

    # 0-3) Drop duplicated columns to avoid df["Close"] turning into DataFrame
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 0-4) If Close missing but Adj Close exists -> use it
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # 0-5) If still missing Close, try find any column containing 'close'
    if "Close" not in df.columns:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if close_like:
            df["Close"] = df[close_like[0]]

    # 0-6) Same trick for Open/High/Low/Volume if needed
    def ensure_col(target: str, keywords):
        if target in df.columns:
            return
        candidates = [c for c in df.columns if any(k in str(c).lower() for k in keywords)]
        if candidates:
            df[target] = df[candidates[0]]

    ensure_col("Open", ["open"])
    ensure_col("High", ["high"])
    ensure_col("Low", ["low"])
    ensure_col("Volume", ["volume", "vol"])

    # 0-7) Final guard: required must exist
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要欄位：{missing}；目前欄位：{df.columns.tolist()}")

    # 0-8) Make sure OHLCV are numeric
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# 1) Download (Cloud-friendly)
# -----------------------------
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


# -----------------------------
# 2) Indicators
# -----------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv(df).copy()

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
    df["ATR_pct"] = (df["ATR"] / (df["Close"] + 1e-9)) * 100

    # ADX (simplified)
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


# -----------------------------
# 3) Cycle Stage (A/B)
# -----------------------------
def detect_stage(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "資料不足"

    latest = df.iloc[-1]
    deviation = (latest["Close"] / (latest["EMA20"] + 1e-9) - 1) * 100

    # Stage 3: Blow-off
    if deviation > 20 and latest["KD_K"] > 90 and latest["ADX"] > 55:
        return "末端期"

    # Stage 2: Acceleration
    ma_bull = (latest["SMA5"] > latest["SMA10"] > latest["SMA20"])
    if ma_bull and latest["ADX"] >= 40 and 12 <= deviation <= 20:
        return "加速期"

    # Stage 1: Launch
    if latest["ADX"] > 20 and deviation < 12:
        return "啟動期"

    return "盤整期"


# -----------------------------
# 4) Resonance (A)
#   你定義的共振：MACD金叉 + KD金叉 + 5>10>20 + 突破BB上軌 + 量能確認
# -----------------------------
def detect_resonance_indices(df: pd.DataFrame) -> list[int]:
    idxs = []
    if len(df) < 60:
        return idxs

    for i in range(21, len(df)):
        macd_cross = (df["MACD_DIF"].iloc[i-1] < df["MACD_Signal"].iloc[i-1]) and (df["MACD_DIF"].iloc[i] > df["MACD_Signal"].iloc[i])
        kd_cross = (df["KD_K"].iloc[i-1] < df["KD_D"].iloc[i-1]) and (df["KD_K"].iloc[i] > df["KD_D"].iloc[i])
        ma_bull = (df["SMA5"].iloc[i] > df["SMA10"].iloc[i] > df["SMA20"].iloc[i])
        bb_break = df["Close"].iloc[i] > df["BB_upper"].iloc[i]
        vol_ok = df["Volume"].iloc[i] > df["VOL_5"].iloc[i]
        adx_rise = df["ADX"].iloc[i] > df["ADX"].iloc[i-1]

        if macd_cross and kd_cross and ma_bull and bb_break and vol_ok and adx_rise:
            idxs.append(i)

    # 只抓「每一段」的第一天（避免連續幾天都成立）
    filtered = []
    last = -999
    for i in idxs:
        if i - last > 5:   # 5天內只算一次啟動
            filtered.append(i)
        last = i
    return filtered


# -----------------------------
# 5) Backtest (C)
# -----------------------------
def backtest_resonance(df: pd.DataFrame, signals: list[int]) -> pd.DataFrame:
    rows = []
    for i in signals:
        if i + 10 >= len(df):
            continue
        entry = float(df["Close"].iloc[i])
        ret5 = (float(df["Close"].iloc[i+5]) / entry - 1) * 100
        ret10 = (float(df["Close"].iloc[i+10]) / entry - 1) * 100
        rows.append({
            "EntryDate": df.index[i],
            "EntryPrice": entry,
            "Return5D(%)": ret5,
            "Return10D(%)": ret10
        })
    return pd.DataFrame(rows)


# -----------------------------
# 6) Trade Engine Output (B)
# -----------------------------
def build_trade_plan(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    stage = detect_stage(df)

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    atr = float(latest["ATR"])
    atr_pct = float(latest["ATR_pct"])
    deviation = (price / (ema20 + 1e-9) - 1) * 100

    pb05 = round(price - 0.5 * atr, 2)
    pb10 = round(price - 1.0 * atr, 2)
    risk = round(ema20 - 1.0 * atr, 2)

    if stage == "啟動期":
        action = "🚀 啟動期：可積極（先 40%），回踩不破再加碼"
        plan = [
            f"第一筆：共振突破當天（若成立）買 40%",
            f"第二筆：回踩 0.5ATR（{pb05}）轉強買 30%",
            f"第三筆：回踩 EMA20（{ema20:.2f}）站回買 30%",
            f"防守：跌破風險線（{risk}）趨勢假設失效"
        ]
    elif stage == "加速期":
        action = "🔥 加速期：不重倉追價；用『小倉突破 + 回踩加碼』"
        plan = [
            "試單：共振突破成立可買 10~20%",
            f"加碼1：回踩 0.5ATR（{pb05}）轉強再加",
            f"加碼2：回踩 1ATR（{pb10}）不破再加",
            f"防守：跌破風險線（{risk}）視為趨勢破壞",
        ]
    elif stage == "末端期":
        action = "⚠️ 末端期：禁止追高；以減碼/移動停利為主"
        plan = [
            "避免新增多單",
            f"保護利潤：EMA20（{ema20:.2f}）為重要守線",
            f"風險線：{risk}",
        ]
    else:
        action = "🟦 盤整期：等待共振啟動（不要用趨勢追價邏輯）"
        plan = [
            "等待：MACD金叉 + KD金叉 + 均線多頭 + 突破BB上軌 + 量能確認",
        ]

    return {
        "stage": stage,
        "price": round(price, 2),
        "ema20": round(ema20, 2),
        "atr_pct": round(atr_pct, 2),
        "deviation": round(deviation, 2),
        "pb05": pb05,
        "pb10": pb10,
        "risk": risk,
        "action": action,
        "plan": plan
    }


# -----------------------------
# 7) UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI Cycle Trading Engine PRO v4", layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO v4（A+B+C / Cloud 防爆）")

    code = st.text_input("股票代碼", "2313").strip()
    symbol = code if (".TW" in code.upper() or ".TWO" in code.upper()) else f"{code}.TW"
    st.caption(f"完整代碼：{symbol}")

    df_raw = download_data(symbol)
    if df_raw is None or df_raw.empty:
        st.error("❌ 無法下載資料（yfinance 可能被擋 / 代碼錯誤）")
        st.stop()

    with st.expander("🔍 Debug：yfinance 原始欄位"):
        st.write(df_raw.columns)

    try:
        df = calculate_indicators(df_raw).dropna().copy()
    except Exception as e:
        st.error(f"❌ 指標計算失敗：{e}")
        st.stop()

    # Summary
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{latest['Close']:.2f}")
    c2.metric("EMA20", f"{latest['EMA20']:.2f}")
    c3.metric("ADX", f"{latest['ADX']:.1f}")
    c4.metric("ATR%", f"{latest['ATR_pct']:.2f}%")

    # Engine
    engine = build_trade_plan(df)

    st.subheader("📊 週期判斷（B）")
    st.metric("當前階段", engine["stage"])
    st.caption(f"乖離率：{engine['deviation']:+.2f}%")

    st.subheader("🧠 引擎決策")
    st.info(engine["action"])

    st.subheader("🎯 具體操作計畫（自動）")
    for x in engine["plan"]:
        st.write(f"- {x}")

    st.subheader("📌 關鍵價位（自動）")
    st.write(f"0.5 ATR 回踩：**{engine['pb05']}**")
    st.write(f"1.0 ATR 回踩：**{engine['pb10']}**")
    st.write(f"核心 EMA20：**{engine['ema20']}**")
    st.warning(f"風險線：**{engine['risk']}**（跌破視為趨勢假設失效）")

    # Resonance (A) + Tracking (B)
    st.subheader("🚀 共振啟動（A）+ 追蹤（B）")
    signals = detect_resonance_indices(df)

    if signals:
        last_i = signals[-1]
        entry_price = float(df["Close"].iloc[last_i])
        now_price = float(df["Close"].iloc[-1])
        gain = (now_price / entry_price - 1) * 100
        st.success(f"最近共振啟動日：{df.index[last_i].date()}｜啟動價：{entry_price:.2f}｜至今漲幅：{gain:.2f}%")
    else:
        st.info("近 3 年未偵測到符合你定義的『結構型共振突破』啟動日（條件很嚴格是正常的）。")

    # Backtest (C)
    st.subheader("📊 共振歷史回測（C）")
    bt = backtest_resonance(df, signals)
    if bt is not None and not bt.empty:
        avg5 = bt["Return5D(%)"].mean()
        avg10 = bt["Return10D(%)"].mean()
        win10 = (bt["Return10D(%)"] > 0).mean() * 100
        st.write(f"- 5日平均報酬：**{avg5:.2f}%**")
        st.write(f"- 10日平均報酬：**{avg10:.2f}%**")
        st.write(f"- 成功率（10日>0）：**{win10:.2f}%**")
        with st.expander("查看回測明細"):
            st.dataframe(bt.round(2), use_container_width=True)
    else:
        st.info("回測樣本不足（可能因為共振條件過嚴，命中次數很少）。")

    # Chart
    st.subheader("📈 K線 + 均線 + 布林 + 共振標記")
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower"))

    # mark resonance entry points
    if signals:
        xs = [df.index[i] for i in signals]
        ys = [df["Close"].iloc[i] for i in signals]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Resonance Launch", marker=dict(size=10)))

    fig.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ 本工具為交易策略研究與提示，不構成投資建議。")

if __name__ == "__main__":
    main()
