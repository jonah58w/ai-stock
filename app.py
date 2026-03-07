# =========================================================
# AI 股票量化分析系統 V10.1（繁體中文版 / 價值分析完整版）
# 單檔分析 + 全台股 Top 10 掃描 + 價值分析
# 資料來源：FinMind 優先 + Yahoo Finance 備援
# =========================================================

from __future__ import annotations

import math
import traceback
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# =========================================================
# 基本設定
# =========================================================
st.set_page_config(
    page_title="AI 股票量化分析系統 V10.1",
    page_icon="📈",
    layout="wide",
)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_START_DAYS = 420

# =========================================================
# UI 樣式
# =========================================================
st.markdown(
    """
    <style>
    .small-muted {color:#6b7280;font-size:0.9rem;}
    .section-title {font-size:1.15rem;font-weight:700;margin-top:0.4rem;margin-bottom:0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 工具函式
# =========================================================
def safe_float(x, default=np.nan):
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
    s = str(symbol).strip().upper()
    s = s.replace(".TW", "").replace(".TWO", "")
    return s


def yahoo_symbol_from_code(code: str, market_type: Optional[str] = None) -> List[str]:
    code = normalize_symbol(code)
    cands = []
    if market_type == "twse":
        cands.append(f"{code}.TW")
        cands.append(f"{code}.TWO")
    elif market_type == "tpex":
        cands.append(f"{code}.TWO")
        cands.append(f"{code}.TW")
    else:
        cands.append(f"{code}.TW")
        cands.append(f"{code}.TWO")
    return cands


def finmind_headers(token: Optional[str]) -> Dict[str, str]:
    token = token or ""
    if token.strip():
        return {"Authorization": f"Bearer {token.strip()}"}
    return {}


def format_num(v, digits=2) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "-"


def format_pct(v, digits=2) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return "-"


# =========================================================
# FinMind API
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def finmind_get(dataset: str, token: Optional[str] = None, **params) -> pd.DataFrame:
    query = {"dataset": dataset}
    query.update(params)

    resp = requests.get(
        FINMIND_URL,
        headers=finmind_headers(token),
        params=query,
        timeout=30,
    )
    if resp.status_code != 200:
        return pd.DataFrame()

    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_stock_info(token: Optional[str] = None) -> pd.DataFrame:
    df = finmind_get("TaiwanStockInfo", token=token)
    if df.empty:
        return df

    df = df.copy()
    for col in ["stock_id", "stock_name", "type", "industry_category"]:
        if col not in df.columns:
            df[col] = ""

    # 只保留上市 / 上櫃，且代碼為純數字
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["type"].isin(["twse", "tpex"])].copy()
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()

    # 去重
    df = df.drop_duplicates(subset=["stock_id"], keep="last")
    df = df.sort_values(["type", "stock_id"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_latest_per_table(token: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    取最近一段時間的 TaiwanStockPER，並保留各股票最新一筆
    """
    start_date = start_date or (date.today() - timedelta(days=15)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanStockPER", token=token, start_date=start_date)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["dividend_yield", "PER", "PBR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["stock_id", "date"]).groupby("stock_id", as_index=False).tail(1)
    df = df.rename(
        columns={
            "dividend_yield": "殖利率%",
            "PER": "本益比",
            "PBR": "股價淨值比",
        }
    )
    return df[["stock_id", "date", "殖利率%", "本益比", "股價淨值比"]].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def get_latest_dividend_table(token: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    取最近 2 年股利政策表，抓每檔股票最新一筆現金股利
    """
    start_date = start_date or (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanStockDividend", token=token, start_date=start_date)
    if df.empty:
        return df

    for col in ["CashEarningsDistribution", "CashStatutorySurplus"]:
        if col not in df.columns:
            df[col] = 0

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["現金股利"] = pd.to_numeric(df["CashEarningsDistribution"], errors="coerce").fillna(0) + \
                 pd.to_numeric(df["CashStatutorySurplus"], errors="coerce").fillna(0)

    df = df.sort_values(["stock_id", "date"]).groupby("stock_id", as_index=False).tail(1)
    return df[["stock_id", "date", "year", "現金股利"]].copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_finmind_price(stock_id: str, token: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    start_date = start_date or start_date_str(DEFAULT_START_DAYS)
    df = finmind_get(
        "TaiwanStockPrice",
        token=token,
        data_id=normalize_symbol(stock_id),
        start_date=start_date,
    )
    if df.empty:
        return df

    rename_map = {
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_yahoo_price(stock_id: str, market_type: Optional[str] = None, period: str = "2y") -> Tuple[pd.DataFrame, str]:
    for ysym in yahoo_symbol_from_code(stock_id, market_type):
        try:
            df = yf.download(
                ysym,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep_cols].copy()
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                if not df.empty:
                    return df, ysym
        except Exception:
            pass
    return pd.DataFrame(), ""


@st.cache_data(ttl=1800, show_spinner=False)
def load_price(stock_id: str, market_type: Optional[str], token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    df = load_finmind_price(stock_id, token=token)
    if not df.empty:
        return df, "FinMind"

    ydf, ysym = load_yahoo_price(stock_id, market_type=market_type)
    if not ydf.empty:
        return ydf, f"Yahoo ({ysym})"

    return pd.DataFrame(), "無"


# =========================================================
# 技術指標
# =========================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["SMA20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(close, window=50).sma_indicator()
    df["SMA200"] = SMAIndicator(close, window=200).sma_indicator()

    df["EMA12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA26"] = EMAIndicator(close, window=26).ema_indicator()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    rsi = RSIIndicator(close, window=14)
    df["RSI"] = rsi.rsi()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_MID"] = bb.bollinger_mavg()
    df["BB_HIGH"] = bb.bollinger_hband()
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_HIGH"] - df["BB_LOW"]) / df["BB_MID"]

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR"] = atr.average_true_range()
    df["ATR%"] = df["ATR"] / df["Close"] * 100

    df["VOL_MA20"] = vol.rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    df["RET_1D"] = df["Close"].pct_change()
    df["RET_5D"] = df["Close"].pct_change(5)
    df["RET_20D"] = df["Close"].pct_change(20)
    df["HH_60"] = df["Close"].rolling(60).max()
    df["DD_60H%"] = (df["Close"] / df["HH_60"] - 1) * 100

    return df


# =========================================================
# 價值分析
# =========================================================
def compute_fair_price_by_yield(cash_dividend: Optional[float], required_yield_pct: float) -> Optional[float]:
    if cash_dividend is None or pd.isna(cash_dividend) or cash_dividend <= 0:
        return None
    if required_yield_pct <= 0:
        return None
    return cash_dividend / (required_yield_pct / 100.0)


def value_score_from_metrics(
    price: float,
    dividend_yield_pct: Optional[float],
    per: Optional[float],
    pbr: Optional[float],
) -> float:
    score = 50.0

    dy = safe_float(dividend_yield_pct, np.nan)
    pe = safe_float(per, np.nan)
    pb = safe_float(pbr, np.nan)

    if pd.notna(dy):
        if dy >= 7:
            score += 18
        elif dy >= 5:
            score += 12
        elif dy >= 3:
            score += 6
        elif dy < 1:
            score -= 6

    if pd.notna(pe):
        if pe <= 10:
            score += 12
        elif pe <= 15:
            score += 6
        elif pe >= 30:
            score -= 12

    if pd.notna(pb):
        if pb <= 1.2:
            score += 10
        elif pb <= 2.0:
            score += 4
        elif pb >= 5:
            score -= 10

    return max(0, min(100, score))


# =========================================================
# 市場 / 風險 / 技術 / 買賣點
# =========================================================
def detect_market_regime(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    price = safe_float(last["Close"], np.nan)
    sma50 = safe_float(last.get("SMA50"), np.nan)
    sma200 = safe_float(last.get("SMA200"), np.nan)
    hist = safe_float(last.get("MACD_hist"), 0)

    if pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200 and hist > 0:
            return "多頭"
        if price < sma50 < sma200 and hist < 0:
            return "空頭"
        if price > sma200 and hist < 0:
            return "多頭回檔"
    return "盤整 / 混合"


def detect_crash_risk(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    atr_pct = safe_float(last.get("ATR%"), 0)
    vol_ratio = safe_float(last.get("VOL_RATIO"), 1)
    dd = safe_float(last.get("DD_60H%"), 0)
    ret5 = safe_float(last.get("RET_5D"), 0) * 100
    bbw = safe_float(last.get("BB_WIDTH"), 0)

    points = 0
    if atr_pct >= 6:
        points += 30
    elif atr_pct >= 4:
        points += 18

    if vol_ratio >= 2.5:
        points += 20
    elif vol_ratio >= 1.8:
        points += 10

    if dd <= -20:
        points += 20
    elif dd <= -12:
        points += 10

    if ret5 <= -12:
        points += 20
    elif ret5 <= -7:
        points += 10

    if bbw >= 0.18:
        points += 10

    if points >= 60:
        return "高"
    if points >= 35:
        return "中"
    return "低"


def technical_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    score = 0.0
    close = safe_float(last["Close"], np.nan)

    if safe_float(last["MACD"], 0) > safe_float(last["MACD_signal"], 0):
        score += 20
    if safe_float(last["MACD_hist"], 0) > 0:
        score += 8

    rsi = safe_float(last["RSI"], 50)
    if 35 <= rsi <= 65:
        score += 12
    elif rsi < 30:
        score += 8
    elif rsi > 75:
        score -= 6

    if safe_float(last["K"], 50) > safe_float(last["D"], 50):
        score += 12
    if safe_float(last["K"], 50) < 20 and safe_float(last["D"], 50) < 20:
        score += 8

    if close > safe_float(last["SMA20"], close):
        score += 10
    if close > safe_float(last["SMA50"], close):
        score += 12
    if close > safe_float(last["SMA200"], close):
        score += 14

    if close <= safe_float(last["BB_LOW"], close) * 1.02:
        score += 8
    elif close >= safe_float(last["BB_HIGH"], close) * 0.98:
        score += 4

    return max(0, min(100, score))


def momentum_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    score = 0.0
    ret5 = safe_float(last.get("RET_5D"), 0)
    ret20 = safe_float(last.get("RET_20D"), 0)
    vr = safe_float(last.get("VOL_RATIO"), 1)

    if ret5 > 0:
        score += 12
    if ret20 > 0:
        score += 18
    if ret20 > 0.10:
        score += 10
    if vr >= 1.5:
        score += 12
    elif vr >= 1.1:
        score += 6

    price = safe_float(last["Close"], 0)
    if price > safe_float(last.get("SMA20"), price):
        score += 14
    if price > safe_float(last.get("SMA50"), price):
        score += 14

    return max(0, min(100, score))


def risk_score(df: pd.DataFrame, crash_risk: str) -> float:
    last = df.iloc[-1]
    score = 100.0

    atr_pct = safe_float(last.get("ATR%"), 0)
    dd = abs(safe_float(last.get("DD_60H%"), 0))
    vr = safe_float(last.get("VOL_RATIO"), 1)

    if atr_pct >= 6:
        score -= 30
    elif atr_pct >= 4:
        score -= 18

    if dd >= 20:
        score -= 20
    elif dd >= 12:
        score -= 10

    if vr >= 2.5:
        score -= 10

    if crash_risk == "高":
        score -= 15
    elif crash_risk == "中":
        score -= 8

    return max(0, min(100, score))


def market_score(regime: str, crash_risk: str) -> float:
    score = 50.0
    if regime == "多頭":
        score += 25
    elif regime == "多頭回檔":
        score += 10
    elif regime == "空頭":
        score -= 20

    if crash_risk == "低":
        score += 5
    elif crash_risk == "高":
        score -= 20

    return max(0, min(100, score))


def predict_trade_points(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    last = df.iloc[-1]
    price = safe_float(last["Close"], np.nan)
    sma20 = safe_float(last.get("SMA20"), np.nan)
    sma50 = safe_float(last.get("SMA50"), np.nan)
    bb_low = safe_float(last.get("BB_LOW"), np.nan)
    bb_high = safe_float(last.get("BB_HIGH"), np.nan)
    hh60 = safe_float(last.get("HH_60"), np.nan)
    atr = safe_float(last.get("ATR"), np.nan)
    rsi = safe_float(last.get("RSI"), 50)

    buy_point = None
    sell_point = None
    stop_loss = None

    cands_buy = [x for x in [sma20, sma50, bb_low] if pd.notna(x)]
    if cands_buy:
        buy_point = min(cands_buy)

    cands_sell = [x for x in [bb_high, hh60] if pd.notna(x)]
    if cands_sell:
        sell_point = max(cands_sell)

    if pd.notna(atr):
        stop_loss = max(price - 2 * atr, 0)

    if rsi < 35 and buy_point is not None:
        buy_point = min(buy_point, price * 0.985)
    if rsi > 72 and sell_point is not None:
        sell_point = max(sell_point, price * 1.02)

    rr = None
    if buy_point is not None and sell_point is not None and stop_loss is not None:
        reward = max(sell_point - price, 0)
        risk = max(price - stop_loss, 0.0001)
        rr = reward / risk

    return {
        "buy_point": round(buy_point, 2) if buy_point is not None else None,
        "sell_point": round(sell_point, 2) if sell_point is not None else None,
        "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
        "rr_ratio": round(rr, 2) if rr is not None else None,
    }


def current_trade_signal(df: pd.DataFrame, pred: Dict[str, Optional[float]]) -> Dict[str, object]:
    last = df.iloc[-1]
    price = float(last["Close"])

    rsi = safe_float(last.get("RSI"), 50)
    macd = safe_float(last.get("MACD"), 0)
    macd_signal_v = safe_float(last.get("MACD_signal"), 0)
    k = safe_float(last.get("K"), 50)
    d = safe_float(last.get("D"), 50)
    bb_upper = safe_float(last.get("BB_HIGH"), price)
    bb_lower = safe_float(last.get("BB_LOW"), price)

    buy_point = pred.get("buy_point")
    sell_point = pred.get("sell_point")

    reasons_buy = []
    reasons_sell = []

    buy_score = 0
    sell_score = 0

    if buy_point is not None and price <= buy_point * 1.03:
        buy_score += 30
        reasons_buy.append("接近預估買點")

    if macd > macd_signal_v:
        buy_score += 18
        reasons_buy.append("MACD 偏多")
    else:
        sell_score += 18
        reasons_sell.append("MACD 偏弱")

    if k > d:
        buy_score += 14
        reasons_buy.append("KD 偏多")
    else:
        sell_score += 14
        reasons_sell.append("KD 偏弱")

    if rsi < 40:
        buy_score += 14
        reasons_buy.append("RSI 偏低")
    elif rsi > 70:
        sell_score += 14
        reasons_sell.append("RSI 偏高")

    if price <= bb_lower * 1.02:
        buy_score += 16
        reasons_buy.append("接近布林下緣")
    if price >= bb_upper * 0.98:
        sell_score += 16
        reasons_sell.append("接近布林上緣")

    if sell_point is not None and price >= sell_point * 0.97:
        sell_score += 30
        reasons_sell.append("接近預估賣點")

    if buy_score >= 55 and buy_score > sell_score:
        decision = "🟢 買點區"
        reason = " / ".join(reasons_buy) if reasons_buy else "偏多"
    elif sell_score >= 55 and sell_score > buy_score:
        decision = "🔴 賣點區"
        reason = " / ".join(reasons_sell) if reasons_sell else "偏弱"
    else:
        decision = "🟡 觀察"
        combo = reasons_buy[:2] + reasons_sell[:2]
        reason = " / ".join(combo) if combo else "尚無明確訊號"

    return {
        "decision": decision,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reason": reason,
    }


def compute_ai_score(
    df: pd.DataFrame,
    dividend_yield_pct: Optional[float],
    per: Optional[float],
    pbr: Optional[float],
) -> Dict[str, object]:
    regime = detect_market_regime(df)
    crash = detect_crash_risk(df)

    ts = technical_score(df)
    ms = momentum_score(df)
    vs = value_score_from_metrics(
        price=float(df["Close"].iloc[-1]),
        dividend_yield_pct=dividend_yield_pct,
        per=per,
        pbr=pbr,
    )
    rs = risk_score(df, crash)
    mks = market_score(regime, crash)

    ai = 0.35 * ts + 0.25 * ms + 0.20 * vs + 0.10 * mks + 0.10 * rs
    ai = round(float(ai), 2)

    if ai >= 85:
        signal = "🚀 強力買進"
    elif ai >= 70:
        signal = "✅ 買進"
    elif ai >= 55:
        signal = "🟡 觀察"
    elif ai >= 40:
        signal = "⚠️ 偏弱"
    else:
        signal = "❌ 避開 / 賣出"

    if crash == "高":
        pos = 0 if ai < 80 else 10
    else:
        if ai >= 90:
            pos = 40
        elif ai >= 80:
            pos = 30
        elif ai >= 70:
            pos = 20
        elif ai >= 60:
            pos = 10
        else:
            pos = 0

    return {
        "technical_score": round(ts, 2),
        "momentum_score": round(ms, 2),
        "value_score": round(vs, 2),
        "risk_score": round(rs, 2),
        "market_score": round(mks, 2),
        "market_regime": regime,
        "crash_risk": crash,
        "ai_score": ai,
        "signal": signal,
        "position_pct": pos,
    }


# =========================================================
# 圖表
# =========================================================
def make_price_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="股價",
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_HIGH"], name="布林上軌"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="布林下軌"))

    fig.update_layout(
        title=title,
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


def make_macd_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


def make_kd_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D"))
    fig.add_hline(y=80, line_dash="dash")
    fig.add_hline(y=20, line_dash="dash")
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


def make_rsi_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend_orientation="h",
    )
    return fig


# =========================================================
# 併表：價值資料
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def build_value_master(token: Optional[str] = None) -> pd.DataFrame:
    info_df = get_tw_stock_info(token=token)
    per_df = get_latest_per_table(token=token)
    div_df = get_latest_dividend_table(token=token)

    if info_df.empty:
        return pd.DataFrame()

    out = info_df.merge(per_df, how="left", left_on="stock_id", right_on="stock_id")
    out = out.merge(div_df[["stock_id", "現金股利"]], how="left", on="stock_id")
    out["現金股利"] = pd.to_numeric(out["現金股利"], errors="coerce")
    return out


# =========================================================
# 單檔分析
# =========================================================
def analyze_single_stock(
    stock_id: str,
    token: Optional[str],
    required_yield_pct: float,
    value_master: pd.DataFrame,
) -> Dict[str, object]:
    stock_id = normalize_symbol(stock_id)

    meta = value_master[value_master["stock_id"] == stock_id].copy()
    stock_name = meta["stock_name"].iloc[0] if not meta.empty else ""
    market_type = meta["type"].iloc[0] if not meta.empty else None
    industry = meta["industry_category"].iloc[0] if not meta.empty else ""

    price_df, source = load_price(stock_id, market_type, token)
    if price_df.empty:
        return {"ok": False, "message": "找不到股價資料"}

    price_df = add_indicators(price_df)
    last = price_df.iloc[-1]
    price = float(last["Close"])

    dy = meta["殖利率%"].iloc[0] if ("殖利率%" in meta.columns and not meta.empty) else np.nan
    per = meta["本益比"].iloc[0] if ("本益比" in meta.columns and not meta.empty) else np.nan
    pbr = meta["股價淨值比"].iloc[0] if ("股價淨值比" in meta.columns and not meta.empty) else np.nan
    cash_dividend = meta["現金股利"].iloc[0] if ("現金股利" in meta.columns and not meta.empty) else np.nan

    fair_price = compute_fair_price_by_yield(cash_dividend, required_yield_pct)
    pred = predict_trade_points(price_df)
    trade_now = current_trade_signal(price_df, pred)
    ai = compute_ai_score(price_df, dy, per, pbr)

    return {
        "ok": True,
        "stock_id": stock_id,
        "stock_name": stock_name,
        "market_type": market_type,
        "industry": industry,
        "source": source,
        "price_df": price_df,
        "price": price,
        "dividend_yield_pct": dy,
        "per": per,
        "pbr": pbr,
        "cash_dividend": cash_dividend,
        "fair_price": fair_price,
        "pred": pred,
        "trade_now": trade_now,
        "ai": ai,
    }


# =========================================================
# 全台股掃描
# =========================================================
def scan_all_tw_stocks(
    token: Optional[str],
    required_yield_pct: float,
    value_master: pd.DataFrame,
    market_filter: str,
    top_n: int,
    progress_bar,
    status_box,
) -> pd.DataFrame:
    universe = value_master.copy()

    if market_filter == "上市":
        universe = universe[universe["type"] == "twse"].copy()
    elif market_filter == "上櫃":
        universe = universe[universe["type"] == "tpex"].copy()

    universe = universe.reset_index(drop=True)
    total = len(universe)
    rows = []

    for i, row in universe.iterrows():
        stock_id = row["stock_id"]
        stock_name = row.get("stock_name", "")
        market_type = row.get("type", "")
        industry = row.get("industry_category", "")

        progress = (i + 1) / max(total, 1)
        progress_bar.progress(progress)
        status_box.caption(f"掃描中：{i+1}/{total}　{stock_id} {stock_name}")

        try:
            price_df, source = load_price(stock_id, market_type, token)
            if price_df.empty or len(price_df) < 80:
                continue

            price_df = add_indicators(price_df)
            last = price_df.iloc[-1]
            price = float(last["Close"])
            prev_close = float(price_df["Close"].iloc[-2]) if len(price_df) >= 2 else price
            chg_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

            dy = row.get("殖利率%", np.nan)
            per = row.get("本益比", np.nan)
            pbr = row.get("股價淨值比", np.nan)
            cash_dividend = row.get("現金股利", np.nan)

            fair_price = compute_fair_price_by_yield(cash_dividend, required_yield_pct)
            pred = predict_trade_points(price_df)
            trade_now = current_trade_signal(price_df, pred)
            ai = compute_ai_score(price_df, dy, per, pbr)

            rows.append(
                {
                    "股票代碼": stock_id,
                    "股票名稱": stock_name,
                    "市場別": "上市" if market_type == "twse" else "上櫃",
                    "產業": industry,
                    "股價": round(price, 2),
                    "漲跌%": round(chg_pct, 2),
                    "AI分數": ai["ai_score"],
                    "AI建議": ai["signal"],
                    "目前動作": trade_now["decision"],
                    "市場狀態": ai["market_regime"],
                    "崩盤風險": ai["crash_risk"],
                    "殖利率%": round(safe_float(dy, np.nan), 2) if pd.notna(dy) else np.nan,
                    "本益比": round(safe_float(per, np.nan), 2) if pd.notna(per) else np.nan,
                    "股價淨值比": round(safe_float(pbr, np.nan), 2) if pd.notna(pbr) else np.nan,
                    "現金股利": round(safe_float(cash_dividend, np.nan), 2) if pd.notna(cash_dividend) else np.nan,
                    "合理價": round(fair_price, 2) if fair_price is not None else np.nan,
                    "技術分": ai["technical_score"],
                    "動能分": ai["momentum_score"],
                    "價值分": ai["value_score"],
                    "建議倉位%": ai["position_pct"],
                    "預估買點": pred.get("buy_point"),
                    "預估賣點": pred.get("sell_point"),
                    "停損": pred.get("stop_loss"),
                    "R/R": pred.get("rr_ratio"),
                    "資料來源": source,
                }
            )
        except Exception:
            continue

    progress_bar.empty()
    status_box.empty()

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["AI分數", "價值分", "殖利率%", "漲跌%"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return out.head(top_n).copy()


# =========================================================
# 主畫面
# =========================================================
st.title("📈 AI 股票量化分析系統 V10.1")
st.caption("資料來源：FinMind + Yahoo Finance ｜ 技術分析 + 價值分析 + AI決策引擎")

with st.sidebar:
    st.header("⚙️ 系統設定")

    finmind_token = ""
    try:
        finmind_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        finmind_token = ""

    finmind_token = st.text_input("FinMind Token（可留空）", value=finmind_token, type="password")
    required_yield_pct = st.slider("合理殖利率假設（%）", min_value=2.0, max_value=10.0, value=5.0, step=0.5)

    st.markdown("---")
    st.caption("全台股掃描會花一些時間；第一次最慢，之後快取會明顯加速。")

mode = st.radio(
    "系統模式",
    ["📊 單一股票分析", "🔎 Top 10 全台股掃描", "ℹ️ 系統說明"],
    horizontal=True,
)

with st.spinner("載入台股清單 / 價值資料中..."):
    value_master = build_value_master(token=finmind_token if finmind_token else None)

# =========================================================
# 單一股票分析
# =========================================================
if mode == "📊 單一股票分析":
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.text_input("股票代碼", value="2330")
    with c2:
        run_single = st.button("開始分析", type="primary")

    if run_single:
        try:
            with st.spinner("分析中..."):
                result = analyze_single_stock(
                    stock_id=symbol,
                    token=finmind_token if finmind_token else None,
                    required_yield_pct=required_yield_pct,
                    value_master=value_master,
                )

            if not result["ok"]:
                st.error(result["message"])
            else:
                price_df = result["price_df"]
                pred = result["pred"]
                trade_now = result["trade_now"]
                ai = result["ai"]

                st.markdown("## 股票決策總覽")
                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("股票", f'{result["stock_id"]} {result["stock_name"]}')
                a2.metric("目前價格", format_num(result["price"], 2))
                a3.metric("AI 綜合評分", format_num(ai["ai_score"], 2))
                a4.metric("AI 建議", ai["signal"])
                a5.metric("資料來源", result["source"])

                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("目前動作", trade_now["decision"])
                b2.metric("市場狀態", ai["market_regime"])
                b3.metric("崩盤風險", ai["crash_risk"])
                b4.metric("建議倉位", f'{ai["position_pct"]}%')
                b5.metric("產業", result["industry"] if result["industry"] else "-")

                st.markdown("## 買賣點判斷")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("預估買點", format_num(pred.get("buy_point"), 2))
                p2.metric("預估賣點", format_num(pred.get("sell_point"), 2))
                p3.metric("買方訊號強度", f'{trade_now["buy_score"]:.0f}')
                p4.metric("賣方訊號強度", f'{trade_now["sell_score"]:.0f}')
                st.info(f'AI 判斷依據：{trade_now["reason"]}')

                st.markdown("## 價值分析")
                v1, v2, v3, v4, v5 = st.columns(5)
                v1.metric("殖利率", format_pct(result["dividend_yield_pct"], 2))
                v2.metric("本益比", format_num(result["per"], 2))
                v3.metric("股價淨值比", format_num(result["pbr"], 2))
                v4.metric("現金股利", format_num(result["cash_dividend"], 2))
                v5.metric("簡化合理價", format_num(result["fair_price"], 2))

                st.markdown("## 風險與停損")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("停損", format_num(pred.get("stop_loss"), 2))
                r2.metric("風險報酬比", format_num(pred.get("rr_ratio"), 2))
                r3.metric("ATR%", format_pct(price_df["ATR%"].iloc[-1], 2))
                r4.metric("量比", format_num(price_df["VOL_RATIO"].iloc[-1], 2))

                st.plotly_chart(
                    make_price_chart(price_df.tail(220), f'{result["stock_id"]} {result["stock_name"]} 股價走勢'),
                    use_container_width=True,
                )

                c_left, c_right = st.columns(2)
                with c_left:
                    st.plotly_chart(
                        make_macd_chart(price_df.tail(220), f'{result["stock_id"]} MACD 指標'),
                        use_container_width=True,
                    )
                with c_right:
                    st.plotly_chart(
                        make_kd_chart(price_df.tail(220), f'{result["stock_id"]} KD 指標'),
                        use_container_width=True,
                    )

                st.plotly_chart(
                    make_rsi_chart(price_df.tail(220), f'{result["stock_id"]} RSI 指標'),
                    use_container_width=True,
                )

                st.markdown("## 分數拆解")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("技術分", format_num(ai["technical_score"], 1))
                s2.metric("動能分", format_num(ai["momentum_score"], 1))
                s3.metric("價值分", format_num(ai["value_score"], 1))
                s4.metric("市場分", format_num(ai["market_score"], 1))
                s5.metric("風險分", format_num(ai["risk_score"], 1))

                st.markdown("## 最新指標快照")
                last = price_df.iloc[-1]
                snap = pd.DataFrame(
                    {
                        "欄位": [
                            "Close", "SMA20", "SMA50", "SMA200", "MACD", "MACD_signal",
                            "RSI", "K", "D", "BB_HIGH", "BB_LOW", "ATR%", "VOL_RATIO",
                            "RET_5D%", "RET_20D%", "DD_60H%"
                        ],
                        "數值": [
                            safe_float(last.get("Close")),
                            safe_float(last.get("SMA20")),
                            safe_float(last.get("SMA50")),
                            safe_float(last.get("SMA200")),
                            safe_float(last.get("MACD")),
                            safe_float(last.get("MACD_signal")),
                            safe_float(last.get("RSI")),
                            safe_float(last.get("K")),
                            safe_float(last.get("D")),
                            safe_float(last.get("BB_HIGH")),
                            safe_float(last.get("BB_LOW")),
                            safe_float(last.get("ATR%")),
                            safe_float(last.get("VOL_RATIO")),
                            safe_float(last.get("RET_5D")) * 100,
                            safe_float(last.get("RET_20D")) * 100,
                            safe_float(last.get("DD_60H%")),
                        ],
                    }
                )
                st.dataframe(snap, use_container_width=True, hide_index=True)

                csv = price_df.to_csv(index=True).encode("utf-8-sig")
                st.download_button(
                    "下載單檔歷史資料 CSV",
                    data=csv,
                    file_name=f'{result["stock_id"]}_stock_data.csv',
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"分析失敗：{e}")
            st.code(traceback.format_exc())

# =========================================================
# 全台股 Top 10 掃描
# =========================================================
elif mode == "🔎 Top 10 全台股掃描":
    st.markdown("## 全台股 Top 10 機會掃描")

    c1, c2, c3 = st.columns(3)
    with c1:
        market_filter = st.selectbox("掃描範圍", ["全部", "上市", "上櫃"], index=0)
    with c2:
        top_n = st.slider("顯示前 N 名", 5, 30, 10)
    with c3:
        only_high_yield = st.checkbox("只看殖利率 >= 3%", value=False)

    start_scan = st.button("開始掃描全台股", type="primary")

    if start_scan:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        with st.spinner("全台股掃描中，第一次可能較久..."):
            top_df = scan_all_tw_stocks(
                token=finmind_token if finmind_token else None,
                required_yield_pct=required_yield_pct,
                value_master=value_master,
                market_filter=market_filter,
                top_n=top_n * 3,  # 先抓多一點，方便再過濾
                progress_bar=progress_bar,
                status_box=status_box,
            )

        if top_df.empty:
            st.warning("沒有掃描到結果。")
        else:
            if only_high_yield:
                top_df = top_df[top_df["殖利率%"] >= 3].copy()

            top_df = top_df.head(top_n).copy()

            st.markdown("### Top 10 掃描結果")
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            if not top_df.empty:
                best = top_df.iloc[0]
                st.markdown("### 第 1 名摘要")
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("股票", f'{best["股票代碼"]} {best["股票名稱"]}')
                d2.metric("股價", format_num(best["股價"], 2))
                d3.metric("AI分數", format_num(best["AI分數"], 2))
                d4.metric("殖利率", format_pct(best["殖利率%"], 2))
                d5.metric("本益比", format_num(best["本益比"], 2))

            csv = top_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載 Top 10 掃描結果 CSV",
                data=csv,
                file_name="top10_scan_result.csv",
                mime="text/csv",
            )

# =========================================================
# 說明
# =========================================================
else:
    st.markdown("## 系統說明")
    st.write(
        """
這版已補上你要求的三個重點：

1. 單一股票加入價值分析  
   - 殖利率
   - 本益比
   - 股價淨值比
   - 現金股利
   - 簡化合理價

2. Top 10 掃描改為全台股  
   - 以上市 / 上櫃清單為基礎
   - 掃描結果列出股價、殖利率、本益比、PBR、現金股利、買點、賣點

3. 保留技術分析與買賣點  
   - MACD / KD / RSI / 布林
   - 當下買點 / 賣點 / 停損 / R:R
   - AI 綜合評分

注意：
- 全台股掃描第一次會比較慢。
- 沒有 FinMind token 也能跑，但穩定度可能較差。
- 殖利率 / 本益比 / PBR 主要來自 FinMind 的 TaiwanStockPER。
        """
    )
