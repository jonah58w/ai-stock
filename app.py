from __future__ import annotations

import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="V21.5 單檔精準版（可解釋混合AI）", layout="wide")

# =========================
# Global Settings
# =========================
DEFAULT_TOTAL_CAPITAL = 9_000_000
DEFAULT_STOCK_RATIO = 0.40

ROLLING_WINDOW = 120          # rolling training window
PRED_HORIZON_DAYS = 5         # label horizon
MIN_RR = 1.80                 # RR filter
MAX_RISK_CAP = 0.015          # hard cap on risk ratio
BASE_RISK = 0.012             # base per-trade risk (stock capital)
DATA_LOOKBACK_MONTHS = 26     # approx 2y+

# =========================
# Helpers
# =========================
def _to_float(x) -> float:
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # Ensure columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.sort_index()
    return df

# =========================
# TWSE / TPEX Data Loader (Daily)
# =========================
def _month_starts(end_date: dt.date, months: int) -> List[dt.date]:
    # produce month starts going back "months"
    out = []
    y, m = end_date.year, end_date.month
    for _ in range(months):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out

def _fetch_twse_month(stock_no: str, month_start: dt.date) -> Optional[pd.DataFrame]:
    # TWSE endpoint expects date YYYYMMDD
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    params = {
        "response": "json",
        "date": month_start.strftime("%Y%m%d"),
        "stockNo": stock_no
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        if j.get("stat") != "OK":
            return None
        data = j.get("data", [])
        if not data:
            return None
        # TWSE columns: 日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
        rows = []
        for row in data:
            # date format like "114/02/03" (ROC year)
            d = str(row[0]).strip()
            # ROC to AD
            try:
                roc_y, mm, dd = d.split("/")
                ad_y = int(roc_y) + 1911
                date = dt.date(ad_y, int(mm), int(dd))
            except Exception:
                continue
            rows.append({
                "date": date,
                "open": _to_float(row[3]),
                "high": _to_float(row[4]),
                "low": _to_float(row[5]),
                "close": _to_float(row[6]),
                "volume": _to_float(row[1]),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception:
        return None

def _fetch_tpex_month(stock_no: str, month_start: dt.date) -> Optional[pd.DataFrame]:
    # TPEX endpoint expects d=YYY/MM/DD (ROC year)
    roc_year = month_start.year - 1911
    d_str = f"{roc_year}/{month_start.month:02d}/{month_start.day:02d}"
    url = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php"
    params = {
        "l": "zh-tw",
        "d": d_str,
        "stkno": stock_no
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get("aaData", [])
        if not data:
            return None
        # TPEX columns vary; typical: 日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數
        rows = []
        for row in data:
            d = str(row[0]).strip()
            try:
                roc_y, mm, dd = d.split("/")
                ad_y = int(roc_y) + 1911
                date = dt.date(ad_y, int(mm), int(dd))
            except Exception:
                continue
            rows.append({
                "date": date,
                "open": _to_float(row[3]),
                "high": _to_float(row[4]),
                "low": _to_float(row[5]),
                "close": _to_float(row[6]),
                "volume": _to_float(row[1]),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception:
        return None

def _fetch_official_daily(stock_no: str, months_back: int = DATA_LOOKBACK_MONTHS) -> Optional[pd.DataFrame]:
    """Try TWSE first; if no data, try TPEX. Merge months into one df."""
    end = dt.date.today()
    months = _month_starts(end, months_back)
    frames = []
    twse_hits = 0
    tpex_hits = 0
    for ms in months:
        df_m = _fetch_twse_month(stock_no, ms)
        if df_m is not None and not df_m.empty:
            frames.append(df_m)
            twse_hits += 1
            continue
        df_m = _fetch_tpex_month(stock_no, ms)
        if df_m is not None and not df_m.empty:
            frames.append(df_m)
            tpex_hits += 1
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df

@st.cache_data(ttl=300)
def load_data(symbol: str) -> pd.DataFrame:
    """
    symbol:
      - "2330" / "6274" => try TWSE/TPEX official
      - "2330.TW" / "6274.TWO" => yfinance (fallback)
    """
    sym = symbol.strip().upper()

    # If pure digits => official
    if sym.isdigit():
        df = _fetch_official_daily(sym)
        if df is not None and len(df) > 120:
            return _normalize_ohlcv(df)

        # fallback to yfinance with .TW then .TWO
        for suf in [".TW", ".TWO"]:
            try:
                dfy = yf.download(sym + suf, period="2y", interval="1d", auto_adjust=False, progress=False)
                if dfy is not None and len(dfy) > 120:
                    dfy = dfy.rename(columns=str.lower)
                    dfy = dfy.rename(columns={"adj close": "adj_close"})
                    return _normalize_ohlcv(dfy)
            except Exception:
                pass
        raise ValueError(f"抓不到資料：{symbol}（官方 + Yahoo fallback 都失敗）")

    # Otherwise treat as yfinance ticker
    try:
        dfy = yf.download(sym, period="2y", interval="1d", auto_adjust=False, progress=False)
        if dfy is None or len(dfy) < 120:
            raise ValueError("yfinance data too short")
        dfy = dfy.rename(columns=str.lower)
        dfy = dfy.rename(columns={"adj close": "adj_close"})
        return _normalize_ohlcv(dfy)
    except Exception as e:
        raise ValueError(f"抓不到資料：{symbol}（yfinance 失敗）: {e}")

# =========================
# Feature Engineering (Explainable)
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Trend + momentum
    df["ema20"] = close.ewm(span=20).mean()
    df["ema60"] = close.ewm(span=60).mean()

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["adx"] = ta.trend.ADXIndicator(high, low, close).adx()

    # Volume health
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = (vol / df["vol_ma20"]).replace([np.inf, -np.inf], np.nan)

    # Volatility (ATR)
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(60).mean()

    # Distance / momentum
    df["distance_ema20"] = (close - df["ema20"]) / df["ema20"]
    df["mom_5"] = close.pct_change(5)
    df["mom_10"] = close.pct_change(10)

    # Bollinger width for compression
    bb = ta.volatility.BollingerBands(close)
    df["bb_width"] = bb.bollinger_wband()

    # Slope & acceleration (EMA20)
    df["ema_slope"] = df["ema20"].diff()
    df["ema_accel"] = df["ema_slope"].diff()

    # Precomputed rolling highs
    df["roll60_high"] = high.rolling(60).max()
    df["prev60_high"] = df["roll60_high"].shift(1)

    return df

# =========================
# Regime (Explainable)
# =========================
def market_regime(df: pd.DataFrame) -> str:
    close = df["close"]
    ema20 = df["ema20"]
    ema60 = df["ema60"]
    adx = df["adx"]

    if close.iloc[-1] > ema20.iloc[-1] > ema60.iloc[-1] and adx.iloc[-1] > 20:
        return "EXPANSION"
    if adx.iloc[-1] < 18:
        return "COMPRESSION"
    return "CONTRACTION"

# =========================
# Filters / Signals
# =========================
def volume_filter(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """Avoid no-volume breakouts and distribution."""
    vol_ratio = float(df["vol_ratio"].iloc[-1]) if pd.notna(df["vol_ratio"].iloc[-1]) else 0.0
    healthy = (1.2 < vol_ratio < 2.5)

    # Distribution: very high vol + long upper shadow
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    upper_shadow = h - max(o, c)
    body = abs(c - o) + 1e-9
    distribution = (vol_ratio > 2.5 and upper_shadow > body * 1.5)

    return (healthy and not distribution), {
        "vol_ratio": vol_ratio,
        "distribution": float(distribution),
    }

def compression_energy_signal(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """BB width + ATR compression + duration + near-high."""
    bb_width = df["bb_width"].iloc[-1]
    if pd.isna(bb_width):
        return False, {"bb_width": np.nan, "atr_ratio": np.nan, "compression_days": 0, "near_high": 0}

    bb_q25 = df["bb_width"].quantile(0.25)
    compression = bb_width < bb_q25

    atr_ratio = df["atr_ratio"].iloc[-1]
    low_vol = (pd.notna(atr_ratio) and atr_ratio < 0.8)

    # Compression duration: in last 15 days, how many days width < q25
    compression_days = int((df["bb_width"] < bb_q25).rolling(15).sum().iloc[-1])
    energy_build = compression_days >= 8

    # Near rolling high
    roll_high = df["roll60_high"].iloc[-1]
    near_high = (df["close"].iloc[-1] > roll_high * 0.95) if pd.notna(roll_high) else False

    ok = bool(compression and low_vol and energy_build and near_high)
    return ok, {
        "bb_width": float(bb_width),
        "atr_ratio": float(atr_ratio) if pd.notna(atr_ratio) else np.nan,
        "compression_days": float(compression_days),
        "near_high": float(near_high),
    }

def failed_breakout_signal(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """Breakout yesterday, failed today with volume + bearish candle + RSI rollover."""
    if len(df) < 70:
        return False, {"failed_breakout": 0}

    breakout_yday = df["close"].iloc[-2] > df["prev60_high"].iloc[-2]
    failed_return = df["close"].iloc[-1] < df["prev60_high"].iloc[-1]

    vol_ratio = float(df["vol_ratio"].iloc[-1]) if pd.notna(df["vol_ratio"].iloc[-1]) else 0.0
    bearish = df["close"].iloc[-1] < df["open"].iloc[-1]
    large_bear = bearish and (vol_ratio > 1.3)

    rsi_rollover = df["rsi"].iloc[-1] < df["rsi"].iloc[-2] if pd.notna(df["rsi"].iloc[-2]) else False

    ok = bool(breakout_yday and failed_return and large_bear and rsi_rollover)
    return ok, {
        "breakout_yday": float(breakout_yday),
        "failed_return": float(failed_return),
        "vol_ratio": float(vol_ratio),
        "bearish": float(bearish),
        "rsi_rollover": float(rsi_rollover),
        "failed_breakout": float(ok),
    }

def momentum_acceleration_signal(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """Slope up + acceleration up + short momentum > medium momentum."""
    slope_up = df["ema_slope"].iloc[-1] > 0
    accel_up = df["ema_accel"].iloc[-1] > 0
    mom_accel = df["mom_5"].iloc[-1] > df["mom_10"].iloc[-1]

    ok = bool(slope_up and accel_up and mom_accel)
    return ok, {
        "slope_up": float(slope_up),
        "accel_up": float(accel_up),
        "mom_5": float(df["mom_5"].iloc[-1]) if pd.notna(df["mom_5"].iloc[-1]) else np.nan,
        "mom_10": float(df["mom_10"].iloc[-1]) if pd.notna(df["mom_10"].iloc[-1]) else np.nan,
        "mom_accel": float(mom_accel),
    }

# =========================
# AI Model (Rolling 120) - Explainable Features
# =========================
def train_ai_win_prob(df_feat: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Predict probability that close(t+PRED_HORIZON_DAYS) > close(t).
    Rolling window training only.
    """
    df = df_feat.copy()
    df["future_close"] = df["close"].shift(-PRED_HORIZON_DAYS)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    # Features (explainable)
    feature_cols = [
        "distance_ema20",
        "adx",
        "rsi",
        "vol_ratio",
        "atr_ratio",
        "mom_10",
        "bb_width",
        "ema_slope",
        "ema_accel",
    ]

    df = df.dropna(subset=feature_cols + ["target"])
    if len(df) < (ROLLING_WINDOW + 30):
        # Not enough for stable training
        return 0.5, {"model_status": 0.0}

    train = df.iloc[-(ROLLING_WINDOW + 1):-1]
    latest = df.iloc[-1:]

    X = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train["target"].astype(int)

    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=5,
        min_samples_leaf=6,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)

    X_latest = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    prob = float(model.predict_proba(X_latest)[0][1])

    # Quick explain: top importances (normalized)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    explain = {f"imp_{feature_cols[i]}": float(importances[i]) for i in top_idx}
    explain["model_status"] = 1.0
    explain["raw_prob"] = prob

    return prob, explain

# =========================
# Trade Plan
# =========================
@dataclass
class TradePlan:
    verdict: str
    reason: str
    regime: str

    price: float
    entry_zone: Tuple[float, float]
    stop: float
    target1: float
    target2: float
    rr1: float

    risk_ratio: float
    shares: int
    position_value: float
    risk_amount: float

    ai_prob: float

def calc_position_size(
    total_capital: float,
    stock_ratio: float,
    max_positions: int,
    price: float,
    stop: float,
    risk_ratio: float
) -> Tuple[int, float, float]:
    stock_capital = total_capital * stock_ratio
    max_risk_money = stock_capital * risk_ratio

    risk_per_share = max(price - stop, 0.0)
    if risk_per_share <= 0:
        return 0, 0.0, 0.0

    shares = int(max_risk_money / risk_per_share)

    # concentration cap: max 1/max_positions of stock capital
    cap_value = stock_capital / max_positions
    if shares * price > cap_value:
        shares = int(cap_value / price)

    position_value = shares * price
    return shares, position_value, max_risk_money

def build_trade_plan(
    df_feat: pd.DataFrame,
    total_capital: float,
    stock_ratio: float,
    max_positions: int = 3
) -> TradePlan:
    price = float(df_feat["close"].iloc[-1])

    regime = market_regime(df_feat)

    # Baseline filters
    trend_ok = df_feat["ema20"].iloc[-1] > df_feat["ema60"].iloc[-1]
    rsi_now = float(df_feat["rsi"].iloc[-1]) if pd.notna(df_feat["rsi"].iloc[-1]) else 50.0
    momentum_ok = (45 < rsi_now < 70)

    # Volume / Compression / Failed breakout / Acceleration
    vol_ok, vol_dbg = volume_filter(df_feat)
    comp_ok, comp_dbg = compression_energy_signal(df_feat)
    fail_ok, fail_dbg = failed_breakout_signal(df_feat)
    accel_ok, accel_dbg = momentum_acceleration_signal(df_feat)

    # AI
    ai_prob, ai_dbg = train_ai_win_prob(df_feat)

    # ATR stop (dynamic)
    atr_now = float(df_feat["atr"].iloc[-1]) if pd.notna(df_feat["atr"].iloc[-1]) else 0.0
    stop = price - 2.0 * atr_now
    # also ensure not above EMA60 too tightly (avoid negative risk)
    ema60 = float(df_feat["ema60"].iloc[-1])
    stop = min(stop, ema60)  # conservative: whichever lower

    # Targets
    roll_high = float(df_feat["roll60_high"].iloc[-1]) if pd.notna(df_feat["roll60_high"].iloc[-1]) else price
    target1 = roll_high
    target2 = roll_high + 2.0 * atr_now

    # Entry zone around EMA20 (pullback-friendly)
    ema20 = float(df_feat["ema20"].iloc[-1])
    entry_low = ema20 * 0.99
    entry_high = ema20 * 1.01

    # RR
    risk_per_share = max(price - stop, 1e-9)
    rr1 = (target1 - price) / risk_per_share

    # Risk ratio (Explainable hybrid)
    risk_ratio = BASE_RISK
    # AI scaling
    if ai_prob > 0.75:
        risk_ratio *= 1.40
    elif ai_prob > 0.65:
        risk_ratio *= 1.20
    elif ai_prob < 0.55:
        risk_ratio *= 0.60

    # Regime scaling
    if regime == "COMPRESSION":
        risk_ratio *= 0.70
    if regime == "CONTRACTION":
        risk_ratio = 0.0

    # Strong confluence bonus: compression energy + acceleration
    if comp_ok and accel_ok:
        risk_ratio *= 1.10

    # Hard cap
    risk_ratio = min(risk_ratio, MAX_RISK_CAP)

    # Must-skip conditions
    if fail_ok:
        return TradePlan(
            verdict="❌ 不交易（假突破反轉風險）",
            reason="偵測到突破失敗 + 放量轉弱（避免做多）",
            regime=regime,
            price=price,
            entry_zone=(entry_low, entry_high),
            stop=stop,
            target1=target1,
            target2=target2,
            rr1=float(rr1),
            risk_ratio=0.0,
            shares=0,
            position_value=0.0,
            risk_amount=0.0,
            ai_prob=ai_prob,
        )

    # Core confluence condition (explainable)
    confluence_ok = bool(trend_ok and momentum_ok and vol_ok and accel_ok)

    # RR filter
    if rr1 < MIN_RR:
        return TradePlan(
            verdict="⚠ 觀察（RR 不足）",
            reason=f"RR={rr1:.2f} < {MIN_RR}，不符合高品質交易",
            regime=regime,
            price=price,
            entry_zone=(entry_low, entry_high),
            stop=stop,
            target1=target1,
            target2=target2,
            rr1=float(rr1),
            risk_ratio=0.0,
            shares=0,
            position_value=0.0,
            risk_amount=0.0,
            ai_prob=ai_prob,
        )

    # Decide
    if (regime != "EXPANSION") or (not confluence_ok) or (ai_prob < 0.55):
        return TradePlan(
            verdict="⚠ 觀察",
            reason="尚未達到趨勢/量價/動能/AI 勝率的共振條件",
            regime=regime,
            price=price,
            entry_zone=(entry_low, entry_high),
            stop=stop,
            target1=target1,
            target2=target2,
            rr1=float(rr1),
            risk_ratio=0.0,
            shares=0,
            position_value=0.0,
            risk_amount=0.0,
            ai_prob=ai_prob,
        )

    # Position sizing (max 2–3 stocks concept baked in via cap)
    shares, position_value, risk_amount = calc_position_size(
        total_capital=total_capital,
        stock_ratio=stock_ratio,
        max_positions=max_positions,
        price=price,
        stop=stop,
        risk_ratio=risk_ratio,
    )

    if shares <= 0:
        return TradePlan(
            verdict="⚠ 觀察（無法計算有效股數）",
            reason="停損距離過小或資料異常導致股數=0",
            regime=regime,
            price=price,
            entry_zone=(entry_low, entry_high),
            stop=stop,
            target1=target1,
            target2=target2,
            rr1=float(rr1),
            risk_ratio=0.0,
            shares=0,
            position_value=0.0,
            risk_amount=0.0,
            ai_prob=ai_prob,
        )

    return TradePlan(
        verdict="✅ 可執行（規則 + AI 共振）",
        reason="趨勢結構 + 量能健康 + 壓縮/加速 + AI 勝率通過 + RR 合格",
        regime=regime,
        price=price,
        entry_zone=(entry_low, entry_high),
        stop=stop,
        target1=target1,
        target2=target2,
        rr1=float(rr1),
        risk_ratio=risk_ratio,
        shares=shares,
        position_value=position_value,
        risk_amount=risk_amount,
        ai_prob=ai_prob,
    )

# =========================
# UI
# =========================
with st.sidebar:
    st.header("⚙️ 參數（可解釋混合）")
    symbol = st.text_input("台股代碼（例：2330 / 6274）", "2330")
    total_capital = st.number_input("總資產（NTD）", min_value=100_000, value=DEFAULT_TOTAL_CAPITAL, step=100_000)
    stock_ratio = st.slider("股票投入比例", 0.05, 0.80, float(DEFAULT_STOCK_RATIO), 0.05)
    max_positions = st.selectbox("同時持股上限（集中）", [2, 3], index=1)

    auto_refresh = st.checkbox("盤中自動刷新（每 5 分鐘）", value=True)
    if auto_refresh:
        # refresh (no extra deps)
        st.caption("已啟用自動刷新（5 分鐘）")
        st.experimental_set_query_params(_ts=str(dt.datetime.now().timestamp()))

# Autorefresh mechanism
if auto_refresh:
    # simple JS-free rerun trigger via meta refresh
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

st.title("🧠 V21.5 單檔精準版（可解釋混合AI）")

# Load and compute
try:
    df_raw = load_data(symbol)
except Exception as e:
    st.error(str(e))
    st.stop()

df = add_features(df_raw)
df = df.dropna()

if len(df) < 200:
    st.warning("資料不足（至少需要約 200 根日K 才能穩定運算）")
    st.stop()

plan = build_trade_plan(df, total_capital=total_capital, stock_ratio=stock_ratio, max_positions=max_positions)

# =========
# Summary Row
# =========
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    st.subheader("📌 結論")
    st.write(plan.verdict)
    st.caption(plan.reason)
with c2:
    st.subheader("🧭 Regime")
    st.metric("市場狀態", plan.regime)
with c3:
    st.subheader("🤖 AI")
    st.metric("勝率（5日）", f"{plan.ai_prob:.2f}")
with c4:
    st.subheader("🎯 RR")
    st.metric("RR(目標1)", f"{plan.rr1:.2f}")

# =========
# Trade Plan
# =========
st.subheader("📋 交易計畫（可直接執行）")
tp1, tp2, tp3, tp4 = st.columns(4)
tp1.metric("現價", f"{plan.price:.2f}")
tp2.metric("進場區", f"{plan.entry_zone[0]:.2f} ~ {plan.entry_zone[1]:.2f}")
tp3.metric("停損", f"{plan.stop:.2f}")
tp4.metric("目標1 / 目標2", f"{plan.target1:.2f} / {plan.target2:.2f}")

tp5, tp6, tp7, tp8 = st.columns(4)
tp5.metric("風險比例", f"{plan.risk_ratio*100:.2f}%")
tp6.metric("建議股數", f"{plan.shares:,d} 股")
tp7.metric("投入金額", f"{plan.position_value:,.0f} NTD")
tp8.metric("風險金額(約)", f"{plan.risk_amount:,.0f} NTD")

# =========
# Explainability Panel
# =========
st.subheader("🔍 可解釋檢查（你可以用來判斷真假訊號）")
vol_ok, vol_dbg = volume_filter(df)
comp_ok, comp_dbg = compression_energy_signal(df)
fail_ok, fail_dbg = failed_breakout_signal(df)
acc_ok, acc_dbg = momentum_acceleration_signal(df)

e1, e2, e3, e4 = st.columns(4)
e1.write("**量能健康**")
e1.write(f"- vol_ratio: {vol_dbg['vol_ratio']:.2f}")
e1.write(f"- distribution: {bool(vol_dbg['distribution'])}")
e1.write(f"- ✅通過: {vol_ok}")

e2.write("**壓縮能量**")
e2.write(f"- bb_width: {comp_dbg['bb_width']:.2f}")
e2.write(f"- atr_ratio: {comp_dbg['atr_ratio']:.2f}")
e2.write(f"- 压缩天数: {int(comp_dbg['compression_days'])}")
e2.write(f"- near_high: {bool(comp_dbg['near_high'])}")
e2.write(f"- ✅通過: {comp_ok}")

e3.write("**假突破反轉**")
e3.write(f"- breakout_yday: {bool(fail_dbg.get('breakout_yday',0))}")
e3.write(f"- failed_return: {bool(fail_dbg.get('failed_return',0))}")
e3.write(f"- vol_ratio: {fail_dbg.get('vol_ratio',0):.2f}")
e3.write(f"- rsi_rollover: {bool(fail_dbg.get('rsi_rollover',0))}")
e3.write(f"- ⚠偵測: {fail_ok}")

e4.write("**動能加速**")
e4.write(f"- slope_up: {bool(acc_dbg['slope_up'])}")
e4.write(f"- accel_up: {bool(acc_dbg['accel_up'])}")
e4.write(f"- mom_5: {acc_dbg['mom_5']:.3f}")
e4.write(f"- mom_10: {acc_dbg['mom_10']:.3f}")
e4.write(f"- ✅通過: {acc_ok}")

# =========
# Chart
# =========
st.subheader("📈 走勢（含 EMA / 布林壓縮指標）")
chart_df = pd.DataFrame(index=df.index)
chart_df["Close"] = df["close"]
chart_df["EMA20"] = df["ema20"]
chart_df["EMA60"] = df["ema60"]
st.line_chart(chart_df)

# Additional indicator charts (simple)
i1, i2, i3 = st.columns(3)
with i1:
    st.caption("BB Width（越低越壓縮）")
    st.line_chart(df[["bb_width"]])
with i2:
    st.caption("Volume Ratio（>1.2 代表放量健康）")
    st.line_chart(df[["vol_ratio"]])
with i3:
    st.caption("EMA Acceleration（>0 代表動能加速）")
    st.line_chart(df[["ema_accel"]])

st.info("提示：盤中刷新只是更新你的「風控與計畫」，不是鼓勵頻繁交易。你仍然用日K做決策，盤中只用來提早風控。")