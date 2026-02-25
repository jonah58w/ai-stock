# ==================================================
# AI Cycle Trading Engine PRO v5.1
# 最近主要轉折共振版 + TW/TWO fallback + Cloud-safe
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
# 0) Robust OHLCV Normalizer (Cloud 防爆)
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    # MultiIndex columns -> pick a level that contains OHLCV
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

    def canon(name: str) -> str:
        s = str(name).strip().lower().replace("_", " ")
        s = " ".join(s.split())
        if s == "open" or (" open" in s and "close" not in s): return "Open"
        if s == "high" or " high" in s: return "High"
        if s == "low" or " low" in s: return "Low"
        if "adj" in s and "close" in s: return "Adj Close"
        if s == "close" or (("close" in s) and ("adj" not in s)): return "Close"
        if s == "volume" or "volume" in s or " vol" in s: return "Volume"
        return str(name).strip()

    df.columns = [canon(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Close fallback
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if close_like:
            df["Close"] = df[close_like[0]]

    # Ensure other columns
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

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要欄位：{missing}；目前欄位：{df.columns.tolist()}")

    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# 1) Download with TW/TWO fallback (最重要修正點)
# -----------------------------
def _yf_download(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="3y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        return None
    return None


def download_data(code_or_symbol: str):
    """
    - 若使用者輸入 6187 -> 依序嘗試 6187.TW / 6187.TWO
    - 若使用者輸入 6187.TWO -> 直接抓
    """
    s = (code_or_symbol or "").strip()
    if not s:
        return None, None

    if "." in s:
        df = _yf_download(s)
        return df, s

    # 先 TW
    sym_tw = f"{s}.TW"
    df = _yf_download(sym_tw)
    if df is not None:
        return df, sym_tw

    # 再 TWO
    sym_two = f"{s}.TWO"
    df = _yf_download(sym_two)
    if df is not None:
        return df, sym_two

    return None, None


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
# 3) Cycle Stage (B)
# -----------------------------
def detect_stage(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "資料不足"

    latest = df.iloc[-1]
    deviation = (latest["Close"] / (latest["EMA20"] + 1e-9) - 1) * 100
    ma_bull = (latest["SMA5"] > latest["SMA10"] > latest["SMA20"])

    if deviation > 20 and latest["KD_K"] > 90 and latest["ADX"] > 55:
        return "末端期"
    if ma_bull and latest["ADX"] >= 40 and 12 <= deviation <= 20:
        return "加速期"
    if latest["ADX"] > 20 and deviation < 12:
        return "啟動期"
    return "盤整期"


# -----------------------------
# 4) Pivot Detection (最近主要轉折點)
# -----------------------------
def detect_pivots(df: pd.DataFrame, left=3, right=3, atr_mult=1.0):
    lows = df["Low"].values
    highs = df["High"].values
    atr = df["ATR"].values

    pivot_low = []
    pivot_high = []

    for i in range(left, len(df) - right):
        # pivot low
        if lows[i] == np.min(lows[i-left:i+right+1]):
            bounce = (np.max(highs[i:i+right+1]) - lows[i])
            if not np.isnan(atr[i]) and bounce >= atr_mult * atr[i]:
                pivot_low.append(i)

        # pivot high
        if highs[i] == np.max(highs[i-left:i+right+1]):
            drop = (highs[i] - np.min(lows[i:i+right+1]))
            if not np.isnan(atr[i]) and drop >= atr_mult * atr[i]:
                pivot_high.append(i)

    return pivot_low, pivot_high


# -----------------------------
# 5) Resonance Confirmation (你定義的共振)
# -----------------------------
def is_resonance_breakout(df: pd.DataFrame, i: int) -> bool:
    if i <= 1:
        return False

    macd_cross = (df["MACD_DIF"].iloc[i-1] < df["MACD_Signal"].iloc[i-1]) and (df["MACD_DIF"].iloc[i] > df["MACD_Signal"].iloc[i])
    kd_cross = (df["KD_K"].iloc[i-1] < df["KD_D"].iloc[i-1]) and (df["KD_K"].iloc[i] > df["KD_D"].iloc[i])
    ma_bull = (df["SMA5"].iloc[i] > df["SMA10"].iloc[i] > df["SMA20"].iloc[i])

    bb_break = df["Close"].iloc[i] > df["BB_upper"].iloc[i]
    vol_ok = df["Volume"].iloc[i] > df["VOL_5"].iloc[i]
    adx_rise = df["ADX"].iloc[i] > df["ADX"].iloc[i-1]

    return bool(macd_cross and kd_cross and ma_bull and bb_break and vol_ok and adx_rise)


# -----------------------------
# 6) Major Pivot Resonance Event (A)
# -----------------------------
def find_recent_major_turning_resonance(
    df: pd.DataFrame,
    lookback_bars=120,
    confirm_window=30,
    left=3,
    right=3,
    atr_mult=1.0
):
    if len(df) < 80:
        return None

    start = max(0, len(df) - lookback_bars)

    piv_lows, _ = detect_pivots(df, left=left, right=right, atr_mult=atr_mult)
    piv_lows = [i for i in piv_lows if i >= start]

    events = []
    for p in piv_lows:
        end = min(len(df) - 1, p + confirm_window)
        for i in range(p + 1, end + 1):
            if is_resonance_breakout(df, i):
                events.append({"pivot_i": p, "signal_i": i})
                break

    if not events:
        return None

    events.sort(key=lambda x: x["signal_i"])
    return events[-1]


# -----------------------------
# 7) Backtest for pivot-resonance (C)
# -----------------------------
def backtest_pivot_resonance(
    df: pd.DataFrame,
    lookback_bars=756,
    confirm_window=30,
    left=3,
    right=3,
    atr_mult=1.0
):
    if len(df) < 120:
        return pd.DataFrame()

    start = max(0, len(df) - lookback_bars)
    df2 = df.iloc[start:].copy()

    piv_lows, _ = detect_pivots(df2, left=left, right=right, atr_mult=atr_mult)

    rows = []
    for p in piv_lows:
        end = min(len(df2) - 1, p + confirm_window)
        sig = None
        for i in range(p + 1, end + 1):
            if is_resonance_breakout(df2, i):
                sig = i
                break
        if sig is None:
            continue
        if sig + 10 >= len(df2):
            continue

        entry = float(df2["Close"].iloc[sig])
        ret5 = (float(df2["Close"].iloc[sig + 5]) / entry - 1) * 100
        ret10 = (float(df2["Close"].iloc[sig + 10]) / entry - 1) * 100

        rows.append({
            "PivotDate": df2.index[p],
            "SignalDate": df2.index[sig],
            "EntryPrice": entry,
            "Return5D(%)": ret5,
            "Return10D(%)": ret10
        })

    return pd.DataFrame(rows)


# -----------------------------
# 8) Trade Plan (B)
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
            "第一筆：最近主要轉折共振成立 → 40%",
            f"第二筆：回踩 0.5ATR（{pb05}）轉強 → 30%",
            f"第三筆：回踩 EMA20（{ema20:.2f}）站回 → 30%",
            f"防守：跌破風險線（{risk}）視為趨勢破壞"
        ]
    elif stage == "加速期":
        action = "🔥 加速期：不重倉追價；用『小倉突破 + 回踩加碼』"
        plan = [
            "試單：主要轉折共振成立可買 10~20%",
            f"加碼1：回踩 0.5ATR（{pb05}）轉強再加",
            f"加碼2：回踩 1ATR（{pb10}）不破再加",
            f"防守：跌破風險線（{risk}）視為趨勢假設失效",
        ]
    elif stage == "末端期":
        action = "⚠️ 末端期：禁止追高；以減碼/移動停利為主"
        plan = [
            "避免新增多單",
            f"保護利潤：EMA20（{ema20:.2f}）為重要守線",
            f"風險線：{risk}",
        ]
    else:
        action = "🟦 盤整期：等待主要轉折 + 共振成立"
        plan = [
            "等待：先出現 pivot 轉折，再出現共振突破（MACD/KD/均線/BB/量能/ADX）",
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
# 9) UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI Cycle Trading Engine PRO v5.1", layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO v5.1（最近主要轉折共振版）")

    code = st.text_input("股票代碼", "6187").strip()

    with st.sidebar:
        st.header("⚙️ 共振/轉折參數")
        lookback_bars = st.slider("只看最近幾根（日K）", 60, 300, 120, 10)
        confirm_window = st.slider("Pivot 後幾天內找共振", 10, 60, 30, 5)
        atr_mult = st.slider("主要轉折強度（ATR倍數）", 0.5, 3.0, 1.0, 0.1)
        show_debug = st.checkbox("顯示 Debug 欄位", value=False)

    with st.spinner("下載資料中..."):
        df_raw, used_symbol = download_data(code)

    if df_raw is None or df_raw.empty:
        st.error("❌ 無法下載資料（yfinance 可能被擋 / 代碼錯誤 / Yahoo 暫時不可用）")
        st.info("建議：\n- 換一支股票測試（如 2330）\n- 或輸入完整尾碼（例如 6187.TWO）\n- 若 Cloud 常被擋，可改用 TWSE/TPEX 官方資料源（我可幫你升級）")
        st.stop()

    st.caption(f"✅ 成功取得資料：{used_symbol}")

    if show_debug:
        with st.expander("🔍 Debug：yfinance 原始欄位"):
            st.write(df_raw.columns)

    try:
        df = calculate_indicators(df_raw).dropna().copy()
    except Exception as e:
        st.error(f"❌ 指標計算失敗：{e}")
        st.stop()

    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{latest['Close']:.2f}")
    c2.metric("EMA20", f"{latest['EMA20']:.2f}")
    c3.metric("ADX", f"{latest['ADX']:.1f}")
    c4.metric("ATR%", f"{latest['ATR_pct']:.2f}%")

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

    # ===== Major Turning Resonance (A) =====
    st.subheader("🚀 最近主要轉折共振（A）")
    event = find_recent_major_turning_resonance(
        df,
        lookback_bars=lookback_bars,
        confirm_window=confirm_window,
        left=3, right=3,
        atr_mult=atr_mult
    )

    pivot_i = None
    sig_i = None

    if event is None:
        st.info("最近區間沒有找到『主要轉折 + 共振突破』事件（可調大 lookback 或降低 ATR 倍數）。")
    else:
        pivot_i = event["pivot_i"]
        sig_i = event["signal_i"]
        entry_price = float(df["Close"].iloc[sig_i])
        now_price = float(df["Close"].iloc[-1])
        gain = (now_price / entry_price - 1) * 100

        st.success(
            f"主要轉折日：{df.index[pivot_i].date()} ｜ "
            f"共振突破日：{df.index[sig_i].date()} ｜ "
            f"啟動價：{entry_price:.2f} ｜ 至今漲幅：{gain:.2f}%"
        )

    # ===== Backtest (C) =====
    st.subheader("📊 主要轉折共振回測（C）")
    bt = backtest_pivot_resonance(
        df,
        lookback_bars=756,
        confirm_window=confirm_window,
        left=3, right=3,
        atr_mult=atr_mult
    )

    if not bt.empty:
        avg5 = bt["Return5D(%)"].mean()
        avg10 = bt["Return10D(%)"].mean()
        win10 = (bt["Return10D(%)"] > 0).mean() * 100
        st.write(f"- 5日平均報酬：**{avg5:.2f}%**")
        st.write(f"- 10日平均報酬：**{avg10:.2f}%**")
        st.write(f"- 成功率（10日>0）：**{win10:.2f}%**")
        with st.expander("查看回測明細"):
            st.dataframe(bt.round(2), use_container_width=True)
    else:
        st.info("回測樣本不足（條件偏嚴格很正常）。")

    # ===== Chart =====
    st.subheader("📈 K線 + 均線 + 布林 + 主要轉折/共振標記")
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower"))

    if pivot_i is not None:
        fig.add_trace(go.Scatter(
            x=[df.index[pivot_i]],
            y=[df["Low"].iloc[pivot_i]],
            mode="markers",
            name="Major Pivot Low",
            marker=dict(size=12, symbol="triangle-up")
        ))
    if sig_i is not None:
        fig.add_trace(go.Scatter(
            x=[df.index[sig_i]],
            y=[df["Close"].iloc[sig_i]],
            mode="markers",
            name="Resonance Breakout",
            marker=dict(size=12)
        ))

    fig.update_layout(height=650, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ 本工具為策略研究與提示，不構成投資建議。")


if __name__ == "__main__":
    main()
