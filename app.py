# ============================================================
# AI Cycle Trading Engine PRO V6.1
# - Cloud-safe OHLCV normalization (fix ValueError/KeyError)
# - TW/TWO fallback
# - Multi-event Pivot + Resonance (Launch + Continuation)
# - V6 Squeeze -> Expansion Breakout model
# - NEW: Pullback Rebound Confirmation Module (回踩轉強確認)
# - A + B + C (Events + Trade Plan + Backtest) + PB signals
# ============================================================

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
        if s == "open" or (("open" in s) and ("close" not in s)): return "Open"
        if "high" in s: return "High"
        if "low" in s: return "Low"
        if ("adj" in s) and ("close" in s): return "Adj Close"
        if ("close" in s) and ("adj" not in s): return "Close"
        if ("volume" in s) or ("vol" == s) or (" vol" in s): return "Volume"
        return str(name).strip()

    df.columns = [canon(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if close_like:
            df["Close"] = df[close_like[0]]

    def ensure_col(target: str, keys):
        if target in df.columns:
            return
        candidates = [c for c in df.columns if any(k in str(c).lower() for k in keys)]
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

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


# -----------------------------
# 1) Download with TW/TWO fallback
# -----------------------------
def _yf_download(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="5y",
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
    s = (code_or_symbol or "").strip()
    if not s:
        return None, None

    if "." in s:
        df = _yf_download(s)
        return df, s

    for suffix in [".TW", ".TWO"]:
        sym = f"{s}{suffix}"
        df = _yf_download(sym)
        if df is not None and not df.empty:
            return df, sym

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

    df["BB_mid"] = df["SMA20"]
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + df["BB_std"] * 2
    df["BB_lower"] = df["BB_mid"] - df["BB_std"] * 2
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (df["BB_mid"] + 1e-9) * 100  # %

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()

    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9 + 1e-9)
    df["KD_D"] = df["KD_K"].rolling(3).mean()

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = (df["ATR"] / (df["Close"] + 1e-9)) * 100

    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    df["ADX"] = dx.rolling(14).mean()

    df["VOL_5"] = df["Volume"].rolling(5).mean()
    df["VOL_10"] = df["Volume"].rolling(10).mean()
    df["HH20"] = df["High"].rolling(20).max()

    return df


# -----------------------------
# 3) Cycle Stage (B)
# -----------------------------
def detect_stage(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "資料不足"

    latest = df.iloc[-1]
    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    adx = float(latest["ADX"]) if not np.isnan(latest["ADX"]) else 0.0
    dev = (price / (ema20 + 1e-9) - 1) * 100
    ma_bull = (latest["SMA5"] > latest["SMA10"] > latest["SMA20"])

    if dev > 22 and price > float(latest["BB_upper"]):
        return "高位盤整/加速前段"
    if dev > 20 and latest["KD_K"] > 90 and adx > 55:
        return "末端期"
    if ma_bull and adx >= 35 and 10 <= dev <= 20:
        return "加速期"
    if adx > 18 and dev < 10:
        return "啟動期"
    return "盤整期"


# -----------------------------
# 4) V6 Squeeze & MA Convergence
# -----------------------------
def is_squeeze(df: pd.DataFrame, i: int, squeeze_lookback=120, squeeze_pct=20) -> bool:
    if i < squeeze_lookback:
        return False
    window = df["BB_width"].iloc[i - squeeze_lookback:i].dropna()
    if window.empty:
        return False
    thr = np.percentile(window.values, squeeze_pct)
    return float(df["BB_width"].iloc[i]) <= float(thr)


def ma_convergence(df: pd.DataFrame, i: int, max_gap_pct=2.0) -> bool:
    a = float(df["SMA5"].iloc[i])
    b = float(df["SMA10"].iloc[i])
    c = float(df["SMA20"].iloc[i])
    if np.isnan(a) or np.isnan(b) or np.isnan(c):
        return False
    mid = float(df["Close"].iloc[i]) + 1e-9
    gap = (max(a, b, c) - min(a, b, c)) / mid * 100
    return gap <= max_gap_pct


def first_break_bb_upper(df: pd.DataFrame, i: int) -> bool:
    if i <= 0:
        return False
    return (df["Close"].iloc[i] > df["BB_upper"].iloc[i]) and (df["Close"].iloc[i-1] <= df["BB_upper"].iloc[i-1])


def break_20d_high(df: pd.DataFrame, i: int) -> bool:
    if i <= 0:
        return False
    prev_hh20 = df["HH20"].iloc[i-1]
    if np.isnan(prev_hh20):
        return False
    return df["Close"].iloc[i] > prev_hh20


def volume_surge(df: pd.DataFrame, i: int, mult=1.2) -> bool:
    v5 = df["VOL_5"].iloc[i]
    if np.isnan(v5) or v5 <= 0:
        return False
    return df["Volume"].iloc[i] >= v5 * mult


# -----------------------------
# 5) Resonance (A1/A2/A3)
# -----------------------------
def is_launch_resonance(df: pd.DataFrame, i: int) -> bool:
    if i <= 1:
        return False
    macd_cross = (df["MACD_DIF"].iloc[i-1] < df["MACD_Signal"].iloc[i-1]) and (df["MACD_DIF"].iloc[i] > df["MACD_Signal"].iloc[i])
    kd_cross = (df["KD_K"].iloc[i-1] < df["KD_D"].iloc[i-1]) and (df["KD_K"].iloc[i] > df["KD_D"].iloc[i])

    ma_bull = (df["SMA5"].iloc[i] > df["SMA10"].iloc[i] > df["SMA20"].iloc[i]) and (df["SMA20"].iloc[i] >= df["SMA20"].iloc[i-1])
    breakout = first_break_bb_upper(df, i) or break_20d_high(df, i)

    vol_ok = df["Volume"].iloc[i] > df["VOL_5"].iloc[i]
    adx_ok = (df["ADX"].iloc[i] >= 15) and (df["ADX"].iloc[i] >= df["ADX"].iloc[i-1])

    return bool(macd_cross and kd_cross and ma_bull and breakout and vol_ok and adx_ok)


def is_continuation_resonance(df: pd.DataFrame, i: int) -> bool:
    if i <= 2:
        return False

    macd_bull = (df["MACD_DIF"].iloc[i] > df["MACD_Signal"].iloc[i]) and (df["MACD_DIF"].iloc[i] > df["MACD_DIF"].iloc[i-1])
    kd_bull = (df["KD_K"].iloc[i] > df["KD_D"].iloc[i]) and (df["KD_K"].iloc[i] > df["KD_K"].iloc[i-1])

    ma_bull = (df["SMA5"].iloc[i] > df["SMA10"].iloc[i] > df["SMA20"].iloc[i])
    ma20_up = df["SMA20"].iloc[i] >= df["SMA20"].iloc[i-1]
    ma_ok = ma_bull or ma20_up

    breakout = first_break_bb_upper(df, i) or break_20d_high(df, i)

    vol_ok = volume_surge(df, i, mult=1.1)
    adx_ok = (df["ADX"].iloc[i] >= 15)

    return bool(macd_bull and kd_bull and ma_ok and breakout and vol_ok and adx_ok)


def is_v6_squeeze_breakout(df: pd.DataFrame, i: int,
                           squeeze_lookback=120,
                           squeeze_pct=20,
                           ma_gap_pct=2.0,
                           vol_mult=1.3) -> bool:
    if i <= squeeze_lookback + 5:
        return False

    squeeze_ok = is_squeeze(df, i-1, squeeze_lookback=squeeze_lookback, squeeze_pct=squeeze_pct)
    conv_ok = ma_convergence(df, i-1, max_gap_pct=ma_gap_pct)

    breakout = first_break_bb_upper(df, i) or break_20d_high(df, i)
    vol_ok = volume_surge(df, i, mult=vol_mult)

    macd_ok = df["MACD_DIF"].iloc[i] > df["MACD_Signal"].iloc[i]
    kd_ok = df["KD_K"].iloc[i] > df["KD_D"].iloc[i]

    return bool(squeeze_ok and conv_ok and breakout and vol_ok and macd_ok and kd_ok)


# -----------------------------
# 6) Pivot Detection (fix scalar issue)
# -----------------------------
def detect_pivot_lows(df: pd.DataFrame, left=3, right=3, atr_mult=0.8):
    if len(df) < left + right + 5:
        return []

    lows = df["Low"].values
    highs = df["High"].values
    atr = df["ATR"].values

    pivots = []
    for i in range(left, len(df) - right):
        low_i = float(lows[i])
        if np.isnan(low_i) or np.isnan(atr[i]):
            continue
        seg = lows[i-left:i+right+1]
        if np.isnan(seg).all():
            continue
        if low_i == float(np.nanmin(seg)):
            bounce = float(np.nanmax(highs[i:i+right+1]) - low_i)
            if bounce >= float(atr_mult * atr[i]):
                pivots.append(i)

    return pivots


# -----------------------------
# 7) Find Recent Events (multi events)
# -----------------------------
def find_recent_events(
    df: pd.DataFrame,
    lookback_bars=250,
    confirm_window=60,
    atr_mult=0.8,
    max_events=8,
    enable_launch=True,
    enable_cont=True,
    enable_squeeze=True,
    squeeze_lookback=120,
    squeeze_pct=20,
    ma_gap_pct=2.0,
    vol_mult=1.3
):
    if len(df) < 120:
        return []

    start = max(0, len(df) - lookback_bars)
    df2 = df.iloc[start:].copy()

    pivots = detect_pivot_lows(df2, left=3, right=3, atr_mult=atr_mult)
    events = []

    for p in pivots:
        end = min(len(df2) - 1, p + confirm_window)
        for i in range(p + 1, end + 1):
            typ = None
            if enable_squeeze and is_v6_squeeze_breakout(
                df2, i,
                squeeze_lookback=squeeze_lookback,
                squeeze_pct=squeeze_pct,
                ma_gap_pct=ma_gap_pct,
                vol_mult=vol_mult
            ):
                typ = "V6_SQUEEZE_BREAKOUT"
            elif enable_launch and is_launch_resonance(df2, i):
                typ = "LAUNCH_RESONANCE"
            elif enable_cont and is_continuation_resonance(df2, i):
                typ = "CONTINUATION_RESONANCE"

            if typ is not None:
                entry = float(df2["Close"].iloc[i])
                events.append({
                    "Type": typ,
                    "PivotDate": df2.index[p],
                    "SignalDate": df2.index[i],
                    "Entry": entry,
                    "BB_Upper": float(df2["BB_upper"].iloc[i]),
                    "HH20_prev": float(df2["HH20"].iloc[i-1]) if not np.isnan(df2["HH20"].iloc[i-1]) else np.nan,
                    "VOLx": float(df2["Volume"].iloc[i] / (df2["VOL_5"].iloc[i] + 1e-9)),
                })
                break

    priority = {"V6_SQUEEZE_BREAKOUT": 0, "LAUNCH_RESONANCE": 1, "CONTINUATION_RESONANCE": 2}
    events.sort(key=lambda x: (x["SignalDate"], priority.get(x["Type"], 9)))
    dedup = []
    seen = set()
    for e in events[::-1]:
        k = e["SignalDate"]
        if k in seen:
            continue
        seen.add(k)
        dedup.append(e)
    dedup = dedup[::-1]

    dedup.sort(key=lambda x: x["SignalDate"], reverse=True)
    return dedup[:max_events]


# -----------------------------
# 8) Backtest (C)
# -----------------------------
def backtest_events(df: pd.DataFrame, events: list, horizon=(5, 10, 20)):
    if not events:
        return pd.DataFrame()

    rows = []
    for e in events:
        if e["SignalDate"] not in df.index:
            continue
        idx = df.index.get_loc(e["SignalDate"])
        entry = float(e["Entry"])
        r = {"Type": e["Type"], "SignalDate": e["SignalDate"], "Entry": entry}
        for h in horizon:
            if idx + h < len(df):
                r[f"R{h}D(%)"] = (float(df["Close"].iloc[idx + h]) / entry - 1) * 100
            else:
                r[f"R{h}D(%)"] = np.nan
        rows.append(r)

    return pd.DataFrame(rows)


# -----------------------------
# 9) Trade Plan (B)
# -----------------------------
def build_trade_plan(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    stage = detect_stage(df)

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    atr = float(latest["ATR"])
    atr_pct = float(latest["ATR_pct"])

    pb05 = round(price - 0.5 * atr, 2)
    pb10 = round(price - 1.0 * atr, 2)
    risk = round(ema20 - 1.0 * atr, 2)

    if stage in ["加速期", "高位盤整/加速前段"]:
        action = "🔥 趨勢突破期：用『突破小倉 + 回踩加碼』，避免一次追滿"
        plan = [
            "試單：出現（V6 壓縮爆發 或 延續共振）→ 10~20%",
            f"加碼1：回踩 0.5ATR（{pb05}）轉強再加",
            f"加碼2：回踩 1.0ATR（{pb10}）不破再加",
            f"防守：跌破風險線（{risk}）視為趨勢假設失效",
        ]
    elif stage == "啟動期":
        action = "🚀 啟動期：可較積極（先 40%），回踩不破再加碼"
        plan = [
            "第一筆：啟動共振成立 → 40%",
            f"第二筆：回踩 0.5ATR（{pb05}）轉強 → 30%",
            f"第三筆：回踩 EMA20（{ema20:.2f}）站回 → 30%",
            f"防守：跌破風險線（{risk}）視為趨勢破壞",
        ]
    elif stage == "末端期":
        action = "⚠️ 末端期：禁止追高；以減碼/移動停利為主"
        plan = [
            "避免新增多單",
            f"守線：EMA20（{ema20:.2f}）",
            f"風險線：{risk}",
        ]
    else:
        action = "🟦 盤整期：等待『壓縮→爆發』或『主要轉折+共振突破』"
        plan = [
            "先出現 pivot 轉折",
            "再出現：V6 壓縮爆發 / 延續共振 / 啟動共振（至少一種）",
        ]

    return {
        "stage": stage,
        "price": round(price, 2),
        "ema20": round(ema20, 2),
        "atr": round(atr, 2),
        "atr_pct": round(atr_pct, 2),
        "pb05": pb05,
        "pb10": pb10,
        "risk": risk,
        "action": action,
        "plan": plan,
    }


# ============================================================
# 10) NEW MODULE: Pullback Rebound Confirmation (回踩轉強確認)
# ============================================================

def _within(x, target, tol):
    return abs(x - target) <= tol

def pullback_rebound_signals(
    df: pd.DataFrame,
    event: dict,
    pb_tol_atr: float = 0.25,
    vol_mult: float = 1.05
):
    """
    針對「選定事件」之後的走勢，偵測回踩加碼點：
    - PB_ADD1：回踩 0.5ATR
    - PB_ADD2：回踩 1.0ATR
    - PB_EMA20：回踩 EMA20
    轉強確認（必須同時滿足）：
      1) 觸及回踩區（以 ATR 容忍帶 pb_tol_atr）
      2) 收盤轉強（收紅 或 收盤 > 前一日收盤）
      3) 多頭續航（MACD_DIF > MACD_Signal）
      4) 量能略放大（Volume > VOL_5 * vol_mult）
    回傳：signals list，每筆包含 Date/Type/Price/Reason
    """
    if event is None or "SignalDate" not in event:
        return []

    sdate = event["SignalDate"]
    if sdate not in df.index:
        return []

    start_i = df.index.get_loc(sdate)
    if start_i >= len(df) - 3:
        return []

    signals = []
    latest = df.iloc[-1]
    # 用「當下 ATR」當容忍帶基準（也可改成事件當天 ATR）
    atr_now = float(latest["ATR"]) if not np.isnan(latest["ATR"]) else 0.0
    if atr_now <= 0:
        return []

    price_now = float(latest["Close"])
    pb05 = price_now - 0.5 * atr_now
    pb10 = price_now - 1.0 * atr_now
    ema20_now = float(latest["EMA20"])

    tol = pb_tol_atr * atr_now

    # 掃描事件後的K棒，找第一個符合 PB_ADD1/PB_ADD2/PB_EMA20 的日期
    found = {"PB_ADD1": False, "PB_ADD2": False, "PB_EMA20": False}

    for i in range(start_i + 1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        low = float(row["Low"])
        close = float(row["Close"])
        open_ = float(row["Open"])
        vol = float(row["Volume"])
        vol5 = float(row["VOL_5"]) if not np.isnan(row["VOL_5"]) else 0.0

        macd_ok = bool(row["MACD_DIF"] > row["MACD_Signal"])
        vol_ok = (vol5 > 0) and (vol >= vol5 * vol_mult)

        # 轉強：收紅 或 收盤 > 前收
        rebound_ok = (close >= open_) or (close > float(prev["Close"]))

        def confirm(tag, target_level):
            if found[tag]:
                return
            touched = (low <= target_level + tol) and (low >= target_level - tol or low <= target_level + tol)
            # touched 的本質：low 掃到容忍帶
            if touched and rebound_ok and macd_ok and vol_ok:
                found[tag] = True
                reasons = {
                    "回踩價位": f"{target_level:.2f} ± {tol:.2f}",
                    "轉強": "收紅/收盤走強",
                    "MACD": "多頭續航",
                    "量能": f">{vol_mult:.2f}x VOL5"
                }
                signals.append({
                    "Date": df.index[i],
                    "Type": tag,
                    "Price": close,
                    "Target": target_level,
                    "Tol": tol,
                    "Reason": reasons
                })

        confirm("PB_ADD1", pb05)
        confirm("PB_ADD2", pb10)
        confirm("PB_EMA20", ema20_now)

        if all(found.values()):
            break

    return signals


# -----------------------------
# 11) Plot (add PB markers)
# -----------------------------
def plot_chart(df: pd.DataFrame, event=None, pb_signals=None):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower"))

    if event is not None:
        pdate = event["PivotDate"]
        sdate = event["SignalDate"]
        if pdate in df.index:
            fig.add_trace(go.Scatter(
                x=[pdate],
                y=[float(df.loc[pdate, "Low"])],
                mode="markers",
                name="Pivot",
                marker=dict(size=12, symbol="triangle-up")
            ))
        if sdate in df.index:
            fig.add_trace(go.Scatter(
                x=[sdate],
                y=[float(df.loc[sdate, "Close"])],
                mode="markers",
                name=f"Signal ({event['Type']})",
                marker=dict(size=12)
            ))

    # PB markers
    if pb_signals:
        for s in pb_signals:
            d = s["Date"]
            if d in df.index:
                fig.add_trace(go.Scatter(
                    x=[d],
                    y=[float(df.loc[d, "Close"])],
                    mode="markers",
                    name=s["Type"],
                    marker=dict(size=11, symbol="diamond")
                ))

    fig.update_layout(height=650, showlegend=True)
    return fig


# -----------------------------
# 12) UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI Cycle Trading Engine PRO V6.1", layout="wide")
    st.title("🚀 AI Cycle Trading Engine PRO V6.1（含 回踩轉強確認模組）")

    code = st.text_input("股票代碼", "6187").strip()

    with st.sidebar:
        st.header("⚙️ 掃描參數")
        lookback_bars = st.slider("只看最近幾根（日K）", 120, 600, 250, 10)
        confirm_window = st.slider("Pivot 後幾天內找突破", 20, 120, 60, 5)
        atr_mult = st.slider("主要轉折強度（ATR倍數）", 0.4, 2.5, 0.8, 0.1)
        max_events = st.slider("顯示最近幾次事件", 3, 15, 8, 1)

        st.divider()
        st.subheader("事件引擎開關")
        enable_squeeze = st.checkbox("V6 壓縮爆發 (A3)", value=True)
        enable_launch = st.checkbox("啟動共振 (A1)", value=True)
        enable_cont = st.checkbox("延續共振 (A2)", value=True)

        st.divider()
        st.subheader("V6 壓縮爆發參數")
        squeeze_lookback = st.slider("壓縮比較窗口", 60, 240, 120, 10)
        squeeze_pct = st.slider("壓縮門檻（低檔%）", 5, 40, 20, 1)
        ma_gap_pct = st.slider("均線匯集最大距離(%)", 0.8, 5.0, 2.0, 0.1)
        vol_mult = st.slider("爆發量能倍數(相對VOL5)", 1.0, 2.5, 1.3, 0.1)

        st.divider()
        st.subheader("🆕 回踩轉強確認參數")
        pb_tol_atr = st.slider("回踩容忍帶（ATR倍數）", 0.10, 0.60, 0.25, 0.05)
        pb_vol_mult = st.slider("回踩轉強量能倍數(相對VOL5)", 1.00, 1.60, 1.05, 0.05)

        st.divider()
        show_debug = st.checkbox("顯示 Debug 欄位", value=False)

    with st.spinner("下載資料中..."):
        df_raw, used_symbol = download_data(code)

    if df_raw is None or df_raw.empty:
        st.error("❌ 無法下載資料（yfinance 可能被擋 / 代碼錯誤 / Yahoo 暫時不可用）")
        st.stop()

    st.caption(f"✅ 成功取得資料：{used_symbol}")

    if show_debug:
        with st.expander("🔍 Debug：原始欄位"):
            st.write(df_raw.columns)

    try:
        df = calculate_indicators(df_raw).dropna().copy()
    except Exception as e:
        st.error(f"❌ 指標計算失敗：{e}")
        st.stop()

    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{float(latest['Close']):.2f}")
    c2.metric("EMA20", f"{float(latest['EMA20']):.2f}")
    c3.metric("ADX", f"{float(latest['ADX']):.1f}" if not np.isnan(latest["ADX"]) else "N/A")
    c4.metric("ATR%", f"{float(latest['ATR_pct']):.2f}%")

    # B
    st.subheader("📊 週期判斷（B）")
    plan = build_trade_plan(df)
    st.metric("當前階段", plan["stage"])
    st.info(plan["action"])

    st.subheader("🎯 具體操作計畫（自動）")
    for x in plan["plan"]:
        st.write(f"- {x}")

    st.subheader("📌 關鍵價位（自動）")
    st.write(f"0.5 ATR 回踩：**{plan['pb05']}**")
    st.write(f"1.0 ATR 回踩：**{plan['pb10']}**")
    st.write(f"核心 EMA20：**{plan['ema20']}**")
    st.warning(f"風險線：**{plan['risk']}**（跌破視為趨勢假設失效）")

    # A
    st.subheader("🚀 最近主要轉折共振事件（A）")

    events = find_recent_events(
        df,
        lookback_bars=lookback_bars,
        confirm_window=confirm_window,
        atr_mult=atr_mult,
        max_events=max_events,
        enable_launch=enable_launch,
        enable_cont=enable_cont,
        enable_squeeze=enable_squeeze,
        squeeze_lookback=squeeze_lookback,
        squeeze_pct=squeeze_pct,
        ma_gap_pct=ma_gap_pct,
        vol_mult=vol_mult
    )

    if not events:
        st.info("最近區間沒有找到事件（可：增加 lookback、增加 confirm_window、降低 ATR倍數、放寬匯集% 或 降低量能倍數）。")
        st.plotly_chart(plot_chart(df.tail(lookback_bars)), use_container_width=True)
        st.stop()

    df_ev = pd.DataFrame(events)
    df_ev["SignalDate"] = df_ev["SignalDate"].astype(str)
    df_ev["PivotDate"] = df_ev["PivotDate"].astype(str)
    with st.expander("查看最近事件列表（不再失真）", expanded=True):
        st.dataframe(df_ev[["Type", "PivotDate", "SignalDate", "Entry", "BB_Upper", "HH20_prev", "VOLx"]].round(2), use_container_width=True)

    options = [
        f"{e['SignalDate'].date()} | {e['Type']} | Entry {e['Entry']:.2f}"
        for e in events
    ]
    sel = st.selectbox("選擇要檢視的事件（用來判斷後續回踩加碼點）", options, index=0)
    idx = options.index(sel)
    chosen = events[idx]

    entry = float(chosen["Entry"])
    now = float(df["Close"].iloc[-1])
    gain = (now / entry - 1) * 100

    st.success(
        f"事件：{chosen['Type']} ｜ Pivot：{chosen['PivotDate'].date()} ｜ Signal：{chosen['SignalDate'].date()} ｜ "
        f"Entry：{entry:.2f} ｜ 至今：{gain:.2f}% ｜ VOLx={chosen['VOLx']:.2f}x"
    )

    # 🆕 PB module
    st.subheader("🧲 回踩轉強確認（加碼買點偵測）")
    pb = pullback_rebound_signals(df, chosen, pb_tol_atr=pb_tol_atr, vol_mult=pb_vol_mult)

    if not pb:
        st.info("目前尚未偵測到『回踩到位 + 轉強確認』的加碼點（或回踩尚未發生）。")
        st.caption("提示：若你覺得太嚴格，可把『回踩容忍帶』調大、或把『回踩量能倍數』調低。")
    else:
        pb_df = pd.DataFrame([{
            "Date": x["Date"].date(),
            "Type": x["Type"],
            "Close": round(float(x["Price"]), 2),
            "Target": round(float(x["Target"]), 2),
            "Tol": round(float(x["Tol"]), 2),
            "Reason": " / ".join([f"{k}:{v}" for k, v in x["Reason"].items()])
        } for x in pb])
        st.dataframe(pb_df, use_container_width=True)

    # C
    st.subheader("📊 事件回測（C）")
    bt = backtest_events(df, events, horizon=(5, 10, 20))
    if bt.empty:
        st.info("回測樣本不足。")
    else:
        def summarize(col):
            x = bt[col].dropna()
            if x.empty:
                return None
            return float(x.mean()), float((x > 0).mean() * 100)

        s5 = summarize("R5D(%)")
        s10 = summarize("R10D(%)")
        s20 = summarize("R20D(%)")

        if s5:
            st.write(f"- 5日平均：**{s5[0]:.2f}%**｜勝率：**{s5[1]:.1f}%**")
        if s10:
            st.write(f"- 10日平均：**{s10[0]:.2f}%**｜勝率：**{s10[1]:.1f}%**")
        if s20:
            st.write(f"- 20日平均：**{s20[0]:.2f}%**｜勝率：**{s20[1]:.1f}%**")

        with st.expander("查看回測明細"):
            st.dataframe(bt.round(2), use_container_width=True)

    # Chart
    st.subheader("📈 K線 + 均線 + 布林 + 事件標記 + 回踩加碼標記")
    df_plot = df.tail(lookback_bars)
    fig = plot_chart(df_plot, event=chosen, pb_signals=pb)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ 本工具為策略研究與提示，不構成投資建議。")


if __name__ == "__main__":
    main()
