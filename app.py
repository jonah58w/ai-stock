# app.py
# AI Stock Trading Assistant V10 PRO
# Single-file Streamlit edition
# Data Engine: FinMind first, Yahoo fallback
# Focus: TW stocks, technical + valuation + AI score + scanner + crash detector
#
# Suggested requirements.txt:
# streamlit
# pandas
# numpy
# requests
# yfinance
# plotly
# ta

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Stock Trading Assistant V10 PRO",
    page_icon="📈",
    layout="wide",
)

APP_TITLE = "AI Stock Trading Assistant V10 PRO"
APP_SUBTITLE = "FinMind + Yahoo fallback | Technical + Value + Risk + AI Decision"

DEFAULT_START_DAYS = 500
DEFAULT_SCAN_LIST = """
2330
2317
2454
2308
2881
2882
2884
2891
2886
2892
2885
1301
1303
2002
1216
2382
2357
3034
2379
2603
2609
2615
3045
3661
0050
0056
00878
00919
"""

# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
    .big-score {
        font-size: 2rem;
        font-weight: 700;
    }
    .small-muted {
        color: #777;
        font-size: 0.9rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class ValueMetrics:
    dividend: float
    dividend_yield: float
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    eps: Optional[float]
    fair_price: Optional[float]
    value_score: float
    valuation_label: str


@dataclass
class TechnicalMetrics:
    technical_score: float
    momentum_score: float
    risk_score: float
    market_score: float
    ai_score: float
    signal: str
    market_regime: str
    crash_risk: str
    suggested_position_pct: float


# =========================================================
# HELPERS
# =========================================================
def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def start_date_str(days: int = DEFAULT_START_DAYS) -> str:
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def normalize_symbol(symbol: str) -> str:
    """
    Normalize Taiwan stock symbols for Yahoo.
    - 2330 -> 2330.TW
    - 6488 -> 6488.TWO if user explicitly sets OTC later via fallback rules
    - keep ^TWII / US tickers / already suffixed symbols unchanged if suitable
    """
    symbol = symbol.strip().upper()

    if not symbol:
        return symbol

    # Special cases
    if symbol.startswith("^"):
        return symbol

    if "." in symbol:
        return symbol

    # TW ETFs / listed stocks often .TW works
    if symbol.isdigit():
        return f"{symbol}.TW"

    return symbol


def raw_code_from_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol.replace(".TW", "").replace(".TWO", "")


def to_display_signal(score: float) -> str:
    if score >= 85:
        return "🚀 STRONG BUY"
    if score >= 70:
        return "✅ BUY"
    if score >= 55:
        return "⏳ WAIT / WATCH"
    if score >= 40:
        return "⚠️ WEAK"
    return "❌ SELL / AVOID"


def nan_to_none(x):
    try:
        if pd.isna(x):
            return None
        return x
    except Exception:
        return x


# =========================================================
# DATA LOADER
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_finmind_price(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    FinMind TaiwanStockPrice:
    dataset=TaiwanStockPrice
    data_id=<stock code>
    """
    code = raw_code_from_symbol(symbol)
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": code,
        "start_date": start_date,
    }
    if end_date:
        params["end_date"] = end_date
    if token:
        params["token"] = token

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return None

        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "max": "High",
                "min": "Low",
                "close": "Close",
                "Trading_Volume": "Volume",
            }
        )
        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[keep_cols].copy()
        df = df.sort_values("Date").set_index("Date")
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_yahoo_price(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Yahoo fallback loader.
    """
    attempts = [symbol]

    # If Taiwan symbol and .TW fails, try .TWO
    if symbol.endswith(".TW"):
        attempts.append(symbol.replace(".TW", ".TWO"))
    elif symbol.isdigit():
        attempts.append(f"{symbol}.TW")
        attempts.append(f"{symbol}.TWO")

    for s in attempts:
        try:
            df = yf.download(
                s,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                if len(wanted) < 5:
                    continue
                df = df[wanted].copy()
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                return df
        except Exception:
            continue

    return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_stock_data(
    symbol: str,
    start_date: str,
    use_finmind: bool,
    use_yahoo: bool,
    finmind_token: Optional[str],
) -> Tuple[Optional[pd.DataFrame], str]:
    symbol = normalize_symbol(symbol)

    if use_finmind:
        df = load_finmind_price(symbol, start_date=start_date, token=finmind_token)
        if df is not None and not df.empty:
            return df, "FinMind"

    if use_yahoo:
        df = load_yahoo_price(symbol, period="2y")
        if df is not None and not df.empty:
            return df, "Yahoo"

    return None, "None"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_info(symbol: str) -> Dict:
    """
    Try fast_info first, then info.
    """
    result = {}
    try:
        ticker = yf.Ticker(symbol)
        try:
            fi = ticker.fast_info
            if fi:
                result["marketCap"] = fi.get("market_cap")
                result["lastPrice"] = fi.get("last_price")
            # fast_info often lacks dividend / PE
        except Exception:
            pass

        try:
            info = ticker.info or {}
            result["dividendRate"] = info.get("dividendRate")
            result["trailingAnnualDividendRate"] = info.get("trailingAnnualDividendRate")
            result["trailingPE"] = info.get("trailingPE")
            result["forwardPE"] = info.get("forwardPE")
            result["epsTrailingTwelveMonths"] = info.get("epsTrailingTwelveMonths")
            result["shortName"] = info.get("shortName")
            result["longName"] = info.get("longName")
            result["quoteType"] = info.get("quoteType")
        except Exception:
            pass
    except Exception:
        pass
    return result


# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend
    df["SMA20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(close, window=50).sma_indicator()
    df["SMA200"] = SMAIndicator(close, window=200).sma_indicator()
    df["EMA12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA26"] = EMAIndicator(close, window=26).ema_indicator()

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # RSI
    df["RSI14"] = RSIIndicator(close, window=14).rsi()

    # KD / Stochastic
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # Bollinger
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_MID"] = bb.bollinger_mavg()
    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]

    # ATR
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR14"] = atr.average_true_range()
    df["ATR_PCT"] = df["ATR14"] / close * 100.0

    # Volume
    df["VOL_MA20"] = volume.rolling(20).mean()
    df["VOL_RATIO"] = volume / df["VOL_MA20"]

    # Return / drawdown
    df["RET_1D"] = close.pct_change()
    df["RET_5D"] = close.pct_change(5)
    df["RET_20D"] = close.pct_change(20)
    df["HH_60"] = close.rolling(60).max()
    df["DD_FROM_60H"] = (close / df["HH_60"] - 1) * 100

    return df


# =========================================================
# MARKET REGIME & CRASH
# =========================================================
def detect_market_regime(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    price = last["Close"]
    sma50 = last.get("SMA50", np.nan)
    sma200 = last.get("SMA200", np.nan)
    macd_hist = last.get("MACD_hist", np.nan)

    if pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200 and macd_hist > 0:
            return "Bull Market"
        if price < sma50 < sma200 and macd_hist < 0:
            return "Bear Market"
        if price > sma200 and macd_hist < 0:
            return "Correction in Uptrend"
    return "Sideways / Mixed"


def detect_crash_risk(df: pd.DataFrame) -> str:
    last = df.iloc[-1]

    atr_pct = safe_float(last.get("ATR_PCT"), 0)
    vol_ratio = safe_float(last.get("VOL_RATIO"), 1)
    bb_width = safe_float(last.get("BB_WIDTH"), 0)
    dd_60 = safe_float(last.get("DD_FROM_60H"), 0)
    ret_5d = safe_float(last.get("RET_5D"), 0) * 100

    risk_points = 0
    if atr_pct >= 6:
        risk_points += 30
    elif atr_pct >= 4:
        risk_points += 20
    elif atr_pct >= 3:
        risk_points += 10

    if vol_ratio >= 2.5:
        risk_points += 25
    elif vol_ratio >= 1.8:
        risk_points += 15

    if bb_width >= 0.18:
        risk_points += 20
    elif bb_width >= 0.12:
        risk_points += 10

    if dd_60 <= -20:
        risk_points += 20
    elif dd_60 <= -12:
        risk_points += 10

    if ret_5d <= -12:
        risk_points += 20
    elif ret_5d <= -7:
        risk_points += 10

    if risk_points >= 70:
        return "High Crash Risk"
    if risk_points >= 40:
        return "Elevated Risk"
    return "Normal"


# =========================================================
# VALUE ENGINE
# =========================================================
def compute_fair_price(
    dividend: float,
    required_return: float = 0.08,
    growth_rate: float = 0.03,
) -> Optional[float]:
    """
    Simplified DDM fair price.
    """
    if dividend is None or dividend <= 0:
        return None
    spread = required_return - growth_rate
    if spread <= 0:
        return None
    return dividend / spread


def compute_value_metrics(
    symbol: str,
    current_price: float,
    required_return: float,
    growth_rate: float,
) -> ValueMetrics:
    symbol = normalize_symbol(symbol)
    info = fetch_yahoo_info(symbol)

    dividend = safe_float(info.get("dividendRate"), np.nan)
    if np.isnan(dividend) or dividend <= 0:
        dividend = safe_float(info.get("trailingAnnualDividendRate"), 0.0)
    if np.isnan(dividend):
        dividend = 0.0

    trailing_pe = nan_to_none(safe_float(info.get("trailingPE"), np.nan))
    forward_pe = nan_to_none(safe_float(info.get("forwardPE"), np.nan))
    eps = nan_to_none(safe_float(info.get("epsTrailingTwelveMonths"), np.nan))

    dy = 0.0
    if current_price and current_price > 0 and dividend > 0:
        dy = dividend / current_price

    fair = compute_fair_price(dividend, required_return=required_return, growth_rate=growth_rate)

    # Value score
    score = 50.0
    label = "Neutral"

    if fair is not None and current_price > 0:
        ratio = current_price / fair
        if ratio <= 0.70:
            score += 30
            label = "Deep Value"
        elif ratio <= 0.90:
            score += 20
            label = "Undervalued"
        elif ratio <= 1.10:
            score += 5
            label = "Fair"
        elif ratio <= 1.30:
            score -= 10
            label = "Slightly Expensive"
        else:
            score -= 25
            label = "Expensive"

    # Dividend influence
    if dy >= 0.07:
        score += 15
    elif dy >= 0.05:
        score += 10
    elif dy >= 0.03:
        score += 5
    elif dy == 0:
        score -= 5

    # PE influence
    pe_ref = None
    if trailing_pe is not None and trailing_pe > 0:
        pe_ref = trailing_pe
    elif forward_pe is not None and forward_pe > 0:
        pe_ref = forward_pe

    if pe_ref is not None:
        if pe_ref <= 10:
            score += 10
        elif pe_ref <= 15:
            score += 5
        elif pe_ref >= 25:
            score -= 10

    score = max(0, min(100, score))

    return ValueMetrics(
        dividend=float(dividend),
        dividend_yield=float(dy),
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        eps=eps,
        fair_price=fair,
        value_score=float(score),
        valuation_label=label,
    )


# =========================================================
# SCORING ENGINE
# =========================================================
def calc_technical_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    score = 0.0
    price = last["Close"]

    # MACD
    if last["MACD"] > last["MACD_signal"]:
        score += 18
    if last["MACD_hist"] > 0:
        score += 8

    # RSI
    rsi = safe_float(last["RSI14"], 50)
    if 45 <= rsi <= 65:
        score += 12
    elif 30 <= rsi < 45:
        score += 8
    elif rsi < 25:
        score += 6
    elif rsi > 75:
        score -= 6

    # KD
    k = safe_float(last["K"], 50)
    d = safe_float(last["D"], 50)
    if k > d:
        score += 12
    if k < 20 and d < 20:
        score += 8
    if k > 85 and d > 85:
        score -= 5

    # Trend MAs
    if price > last["SMA20"]:
        score += 8
    if price > last["SMA50"]:
        score += 10
    if pd.notna(last["SMA200"]) and price > last["SMA200"]:
        score += 12

    # Bollinger
    if price < last["BB_LOWER"]:
        score += 10  # mean reversion opportunity
    elif price > last["BB_UPPER"]:
        score += 6   # breakout tendency

    return max(0, min(100, score))


def calc_momentum_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    score = 0.0

    ret_5d = safe_float(last.get("RET_5D"), 0)
    ret_20d = safe_float(last.get("RET_20D"), 0)
    vol_ratio = safe_float(last.get("VOL_RATIO"), 1)
    price = safe_float(last.get("Close"), 0)
    sma20 = safe_float(last.get("SMA20"), price)
    sma50 = safe_float(last.get("SMA50"), price)

    if ret_5d > 0:
        score += 15
    if ret_20d > 0:
        score += 20
    if price > sma20:
        score += 15
    if price > sma50:
        score += 15

    if vol_ratio >= 1.5:
        score += 15
    elif vol_ratio >= 1.1:
        score += 8

    # Price acceleration
    if ret_20d > 0.12:
        score += 12
    elif ret_20d > 0.05:
        score += 6

    return max(0, min(100, score))


def calc_risk_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    atr_pct = safe_float(last.get("ATR_PCT"), 0)
    dd = abs(safe_float(last.get("DD_FROM_60H"), 0))
    vol_ratio = safe_float(last.get("VOL_RATIO"), 1)

    score = 100.0

    if atr_pct >= 6:
        score -= 30
    elif atr_pct >= 4:
        score -= 20
    elif atr_pct >= 3:
        score -= 10

    if dd >= 20:
        score -= 25
    elif dd >= 12:
        score -= 12

    if vol_ratio >= 2.5:
        score -= 15
    elif vol_ratio >= 1.8:
        score -= 8

    return max(0, min(100, score))


def calc_market_score(regime: str, crash_risk: str) -> float:
    score = 50.0

    if regime == "Bull Market":
        score += 30
    elif regime == "Correction in Uptrend":
        score += 10
    elif regime == "Sideways / Mixed":
        score += 0
    else:  # Bear
        score -= 20

    if crash_risk == "High Crash Risk":
        score -= 30
    elif crash_risk == "Elevated Risk":
        score -= 15
    else:
        score += 5

    return max(0, min(100, score))


def calc_ai_score(
    technical_score: float,
    momentum_score: float,
    value_score: float,
    market_score: float,
    risk_score: float,
) -> float:
    score = (
        0.35 * technical_score
        + 0.25 * momentum_score
        + 0.20 * value_score
        + 0.10 * market_score
        + 0.10 * risk_score
    )
    return round(float(score), 2)


def suggested_position(ai_score: float, crash_risk: str) -> float:
    if crash_risk == "High Crash Risk":
        if ai_score >= 80:
            return 15
        if ai_score >= 65:
            return 10
        return 0

    if ai_score >= 90:
        return 40
    if ai_score >= 80:
        return 30
    if ai_score >= 70:
        return 20
    if ai_score >= 60:
        return 10
    return 0


def build_technical_metrics(df: pd.DataFrame, value_score: float) -> TechnicalMetrics:
    regime = detect_market_regime(df)
    crash_risk = detect_crash_risk(df)

    technical = calc_technical_score(df)
    momentum = calc_momentum_score(df)
    risk = calc_risk_score(df)
    market = calc_market_score(regime, crash_risk)
    ai = calc_ai_score(technical, momentum, value_score, market, risk)
    sig = to_display_signal(ai)
    pos = suggested_position(ai, crash_risk)

    return TechnicalMetrics(
        technical_score=technical,
        momentum_score=momentum,
        risk_score=risk,
        market_score=market,
        ai_score=ai,
        signal=sig,
        market_regime=regime,
        crash_risk=crash_risk,
        suggested_position_pct=pos,
    )


# =========================================================
# PREDICTION ENGINE
# =========================================================
def predict_future_buy_sell_points(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Heuristic model:
    - Buy point: near SMA20/SMA50 pullback or lower Bollinger
    - Sell point: prior high / upper Bollinger / risk extension
    """
    last = df.iloc[-1]
    price = safe_float(last["Close"], np.nan)
    sma20 = safe_float(last["SMA20"], np.nan)
    sma50 = safe_float(last["SMA50"], np.nan)
    bb_upper = safe_float(last["BB_UPPER"], np.nan)
    bb_lower = safe_float(last["BB_LOWER"], np.nan)
    hh60 = safe_float(last["HH_60"], np.nan)
    atr = safe_float(last["ATR14"], np.nan)
    rsi = safe_float(last["RSI14"], 50)

    buy_point = None
    sell_point = None
    stop_loss = None

    if not np.isnan(sma20) and not np.isnan(sma50):
        buy_point = min(sma20, sma50)
    elif not np.isnan(bb_lower):
        buy_point = bb_lower

    if not np.isnan(bb_upper) and not np.isnan(hh60):
        sell_point = max(bb_upper, hh60)
    elif not np.isnan(bb_upper):
        sell_point = bb_upper

    if not np.isnan(atr) and not np.isnan(price):
        stop_loss = max(price - 2 * atr, 0)

    # Adjust with momentum / RSI
    if rsi > 72 and sell_point is not None:
        sell_point = max(sell_point, price * 1.03)
    if rsi < 35 and buy_point is not None:
        buy_point = min(buy_point, price * 0.98)

    rr_ratio = None
    if buy_point and sell_point and stop_loss is not None:
        reward = max(sell_point - price, 0)
        risk = max(price - stop_loss, 0.0001)
        rr_ratio = reward / risk

    return {
        "buy_point": round(buy_point, 2) if buy_point else None,
        "sell_point": round(sell_point, 2) if sell_point else None,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "rr_ratio": round(rr_ratio, 2) if rr_ratio is not None else None,
    }


# =========================================================
# CHARTS
# =========================================================
def make_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    if "SMA200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB Lower"))

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_indicator_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="MACD Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist"))
    fig.update_layout(
        title=f"{symbol} MACD",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


def make_kd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D"))
    fig.add_hline(y=80, line_dash="dash")
    fig.add_hline(y=20, line_dash="dash")
    fig.update_layout(
        title=f"{symbol} KD",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


def make_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(
        title=f"{symbol} RSI",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


# =========================================================
# ANALYSIS PIPELINE
# =========================================================
def analyze_symbol(
    symbol: str,
    use_finmind: bool,
    use_yahoo: bool,
    finmind_token: Optional[str],
    required_return: float,
    growth_rate: float,
) -> Tuple[Optional[pd.DataFrame], Optional[ValueMetrics], Optional[TechnicalMetrics], Dict, str]:
    symbol = normalize_symbol(symbol)
    df, source = load_stock_data(
        symbol=symbol,
        start_date=start_date_str(DEFAULT_START_DAYS),
        use_finmind=use_finmind,
        use_yahoo=use_yahoo,
        finmind_token=finmind_token,
    )
    if df is None or df.empty:
        return None, None, None, {}, source

    df = add_indicators(df)
    current_price = float(df["Close"].iloc[-1])

    value_metrics = compute_value_metrics(
        symbol=symbol,
        current_price=current_price,
        required_return=required_return,
        growth_rate=growth_rate,
    )
    tech_metrics = build_technical_metrics(df, value_metrics.value_score)
    pred = predict_future_buy_sell_points(df)

    return df, value_metrics, tech_metrics, pred, source


# =========================================================
# SCANNER
# =========================================================
def parse_watchlist_input(text: str) -> List[str]:
    codes = []
    for raw in text.replace(",", "\n").splitlines():
        s = raw.strip().upper()
        if not s:
            continue
        codes.append(normalize_symbol(s))
    return list(dict.fromkeys(codes))


def run_scanner(
    symbols: List[str],
    use_finmind: bool,
    use_yahoo: bool,
    finmind_token: Optional[str],
    required_return: float,
    growth_rate: float,
    progress_placeholder,
) -> pd.DataFrame:
    rows = []
    total = len(symbols)

    for i, sym in enumerate(symbols, start=1):
        progress_placeholder.progress(i / total, text=f"Scanning {i}/{total}: {sym}")
        try:
            df, vm, tm, pred, source = analyze_symbol(
                sym,
                use_finmind=use_finmind,
                use_yahoo=use_yahoo,
                finmind_token=finmind_token,
                required_return=required_return,
                growth_rate=growth_rate,
            )
            if df is None or vm is None or tm is None:
                rows.append(
                    {
                        "Symbol": sym,
                        "Source": source,
                        "Price": np.nan,
                        "AI Score": np.nan,
                        "Signal": "NO DATA",
                        "Market": None,
                        "Crash Risk": None,
                        "Dividend Yield %": np.nan,
                        "Fair Price": np.nan,
                        "Value Score": np.nan,
                        "Technical": np.nan,
                        "Momentum": np.nan,
                        "Risk": np.nan,
                        "Position %": np.nan,
                        "Buy Point": np.nan,
                        "Sell Point": np.nan,
                    }
                )
                continue

            price = float(df["Close"].iloc[-1])
            rows.append(
                {
                    "Symbol": sym,
                    "Source": source,
                    "Price": round(price, 2),
                    "AI Score": tm.ai_score,
                    "Signal": tm.signal,
                    "Market": tm.market_regime,
                    "Crash Risk": tm.crash_risk,
                    "Dividend Yield %": round(vm.dividend_yield * 100, 2),
                    "Fair Price": round(vm.fair_price, 2) if vm.fair_price else np.nan,
                    "Value Score": round(vm.value_score, 2),
                    "Technical": round(tm.technical_score, 2),
                    "Momentum": round(tm.momentum_score, 2),
                    "Risk": round(tm.risk_score, 2),
                    "Position %": round(tm.suggested_position_pct, 2),
                    "Buy Point": pred.get("buy_point"),
                    "Sell Point": pred.get("sell_point"),
                }
            )
        except Exception:
            rows.append(
                {
                    "Symbol": sym,
                    "Source": "Error",
                    "Price": np.nan,
                    "AI Score": np.nan,
                    "Signal": "ERROR",
                    "Market": None,
                    "Crash Risk": None,
                    "Dividend Yield %": np.nan,
                    "Fair Price": np.nan,
                    "Value Score": np.nan,
                    "Technical": np.nan,
                    "Momentum": np.nan,
                    "Risk": np.nan,
                    "Position %": np.nan,
                    "Buy Point": np.nan,
                    "Sell Point": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["AI Score", "Dividend Yield %"], ascending=[False, False], na_position="last")
    progress_placeholder.empty()
    return out


# =========================================================
# UI
# =========================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("⚙️ Settings")

    use_finmind = st.checkbox("Use FinMind (priority)", value=True)
    use_yahoo = st.checkbox("Use Yahoo fallback", value=True)

    finmind_token = ""
    try:
        finmind_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        finmind_token = ""

    finmind_token = st.text_input("FinMind token (optional)", value=finmind_token, type="password")

    required_return_pct = st.slider("Required return r (%)", min_value=5.0, max_value=15.0, value=8.0, step=0.5)
    growth_rate_pct = st.slider("Growth rate g (%)", min_value=0.0, max_value=8.0, value=3.0, step=0.5)

    required_return = required_return_pct / 100.0
    growth_rate = growth_rate_pct / 100.0

    st.markdown("---")
    st.subheader("Scanner universe")
    watchlist_text = st.text_area("Symbols", value=DEFAULT_SCAN_LIST, height=260)
    scan_top_n = st.slider("Show Top N", 5, 30, 10)

tab1, tab2, tab3 = st.tabs(["📊 Single Stock", "🔎 Top Scanner", "ℹ️ Notes"])

# =========================================================
# TAB 1 - SINGLE STOCK
# =========================================================
with tab1:
    c1, c2, c3 = st.columns([1.4, 1, 1])

    with c1:
        symbol = st.text_input("Stock symbol", value="2330")
    with c2:
        auto_run = st.checkbox("Auto run on change", value=True)
    with c3:
        run_btn = st.button("Run Analysis", type="primary")

    do_run = auto_run or run_btn

    if do_run:
        try:
            with st.spinner("Analyzing..."):
                df, vm, tm, pred, source = analyze_symbol(
                    symbol=symbol,
                    use_finmind=use_finmind,
                    use_yahoo=use_yahoo,
                    finmind_token=finmind_token if finmind_token else None,
                    required_return=required_return,
                    growth_rate=growth_rate,
                )

            if df is None or vm is None or tm is None:
                st.error("No data found. Please check symbol or data source settings.")
            else:
                display_symbol = normalize_symbol(symbol)
                last = df.iloc[-1]
                price = float(last["Close"])
                prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
                chg = price - prev_close
                chg_pct = (chg / prev_close * 100) if prev_close else 0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"{price:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
                m2.metric("AI Score", f"{tm.ai_score:.2f}")
                m3.metric("Signal", tm.signal)
                m4.metric("Market", tm.market_regime)
                m5.metric("Source", source)

                st.markdown("### AI Decision Panel")
                x1, x2, x3, x4, x5 = st.columns(5)
                x1.metric("Technical", f"{tm.technical_score:.1f}")
                x2.metric("Momentum", f"{tm.momentum_score:.1f}")
                x3.metric("Value", f"{vm.value_score:.1f}")
                x4.metric("Market/Risk", f"{tm.market_score:.1f} / {tm.risk_score:.1f}")
                x5.metric("Suggested Position", f"{tm.suggested_position_pct:.0f}%")

                st.markdown("### Value / Dividend Panel")
                v1, v2, v3, v4, v5 = st.columns(5)
                v1.metric("Dividend", f"{vm.dividend:.2f}")
                v2.metric("Dividend Yield", f"{vm.dividend_yield * 100:.2f}%")
                v3.metric("Fair Price", "-" if vm.fair_price is None else f"{vm.fair_price:.2f}")
                v4.metric("Trailing PE", "-" if vm.trailing_pe is None else f"{vm.trailing_pe:.2f}")
                v5.metric("Valuation", vm.valuation_label)

                st.markdown("### Future Buy / Sell Prediction")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Buy Point", "-" if pred.get("buy_point") is None else f"{pred['buy_point']:.2f}")
                p2.metric("Sell Point", "-" if pred.get("sell_point") is None else f"{pred['sell_point']:.2f}")
                p3.metric("Stop Loss", "-" if pred.get("stop_loss") is None else f"{pred['stop_loss']:.2f}")
                p4.metric("R/R Ratio", "-" if pred.get("rr_ratio") is None else f"{pred['rr_ratio']:.2f}")

                st.markdown("### Risk Panel")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Crash Risk", tm.crash_risk)
                r2.metric("ATR %", f"{safe_float(last.get('ATR_PCT'), 0):.2f}%")
                r3.metric("Vol Ratio", f"{safe_float(last.get('VOL_RATIO'), 1):.2f}x")
                r4.metric("Drawdown(60d)", f"{safe_float(last.get('DD_FROM_60H'), 0):.2f}%")

                st.plotly_chart(make_price_chart(df.tail(220), display_symbol), use_container_width=True)

                c_left, c_right = st.columns(2)
                with c_left:
                    st.plotly_chart(make_indicator_chart(df.tail(220), display_symbol), use_container_width=True)
                with c_right:
                    st.plotly_chart(make_kd_chart(df.tail(220), display_symbol), use_container_width=True)

                st.plotly_chart(make_rsi_chart(df.tail(220), display_symbol), use_container_width=True)

                st.markdown("### Latest Indicator Snapshot")
                snap = pd.DataFrame(
                    {
                        "Metric": [
                            "Close",
                            "SMA20",
                            "SMA50",
                            "SMA200",
                            "MACD",
                            "MACD Signal",
                            "RSI14",
                            "K",
                            "D",
                            "BB Upper",
                            "BB Lower",
                            "ATR %",
                            "Vol Ratio",
                            "Ret 5D %",
                            "Ret 20D %",
                        ],
                        "Value": [
                            round(safe_float(last.get("Close"), np.nan), 4),
                            round(safe_float(last.get("SMA20"), np.nan), 4),
                            round(safe_float(last.get("SMA50"), np.nan), 4),
                            round(safe_float(last.get("SMA200"), np.nan), 4) if pd.notna(last.get("SMA200")) else np.nan,
                            round(safe_float(last.get("MACD"), np.nan), 4),
                            round(safe_float(last.get("MACD_signal"), np.nan), 4),
                            round(safe_float(last.get("RSI14"), np.nan), 4),
                            round(safe_float(last.get("K"), np.nan), 4),
                            round(safe_float(last.get("D"), np.nan), 4),
                            round(safe_float(last.get("BB_UPPER"), np.nan), 4),
                            round(safe_float(last.get("BB_LOWER"), np.nan), 4),
                            round(safe_float(last.get("ATR_PCT"), np.nan), 4),
                            round(safe_float(last.get("VOL_RATIO"), np.nan), 4),
                            round(safe_float(last.get("RET_5D"), np.nan) * 100, 4),
                            round(safe_float(last.get("RET_20D"), np.nan) * 100, 4),
                        ],
                    }
                )
                st.dataframe(snap, use_container_width=True, hide_index=True)

                st.markdown("### Raw Data")
                st.dataframe(df.tail(120), use_container_width=True)

                csv = df.to_csv(index=True).encode("utf-8-sig")
                st.download_button(
                    "Download latest data CSV",
                    data=csv,
                    file_name=f"{display_symbol.replace('.', '_')}_v10pro_data.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.code(traceback.format_exc())

# =========================================================
# TAB 2 - SCANNER
# =========================================================
with tab2:
    st.markdown("### Top Opportunity Scanner")
    symbols = parse_watchlist_input(watchlist_text)

    s1, s2, s3 = st.columns([1, 1, 3])
    with s1:
        start_scan = st.button("Run Scanner", type="primary")
    with s2:
        show_only_buy = st.checkbox("Only show AI Score >= 70", value=False)

    progress = st.empty()

    if start_scan:
        if not symbols:
            st.warning("Please provide at least one symbol.")
        else:
            with st.spinner("Scanning market list..."):
                result_df = run_scanner(
                    symbols=symbols,
                    use_finmind=use_finmind,
                    use_yahoo=use_yahoo,
                    finmind_token=finmind_token if finmind_token else None,
                    required_return=required_return,
                    growth_rate=growth_rate,
                    progress_placeholder=progress,
                )

            if result_df.empty:
                st.warning("No results.")
            else:
                if show_only_buy:
                    result_df = result_df[result_df["AI Score"] >= 70]

                top_df = result_df.head(scan_top_n).copy()

                st.markdown("### Ranked Results")
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                csv = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download scanner CSV",
                    data=csv,
                    file_name="scanner_v10pro_results.csv",
                    mime="text/csv",
                )

# =========================================================
# TAB 3 - NOTES
# =========================================================
with tab3:
    st.markdown(
        """
### 使用說明

1. 單檔分析  
   輸入台股代碼，例如 `2330`、`2454`、`0050`。  
   系統會先嘗試 **FinMind**，失敗再用 **Yahoo**。

2. AI Score 組成  
   - Technical：MACD / KD / RSI / MA / 布林  
   - Momentum：5日 / 20日動能、量比、均線位置  
   - Value：股息、殖利率、簡化 DDM、公允價值、PE  
   - Market：Market Regime + Crash Detector  
   - Risk：ATR%、量能異常、回撤

3. Future Buy / Sell  
   這是 **規則型預估**，不是神諭。  
   它是用 SMA20 / SMA50 / Bollinger / ATR / RSI 做近端估算。

4. FinMind token  
   可不填；有 token 一般可提高可用性。

5. 你可直接把側欄 Scanner universe 改成自己的觀察名單。

### 建議下一步

你這版先跑。  
之後我建議再做三個 patch：

- V10.1：加入你之前偏好的「回檔等待型 / 趨勢突破型」雙模式
- V10.2：加入更完整的台股清單與自動名稱對照
- V10.3：加入你之前一直要的「更清楚的當下是否買點 / 賣點」燈號版面
        """
    )
