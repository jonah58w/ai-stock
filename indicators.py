from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from data_loader import normalize_ohlcv, safe_float


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = df.copy()

    close = pd.Series(df["Close"], index=df.index, dtype="float64")
    high = pd.Series(df["High"], index=df.index, dtype="float64")
    low = pd.Series(df["Low"], index=df.index, dtype="float64")

    # ===== MACD =====
    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_signal"] = df["MACD_Signal"]  # 相容舊欄位
    df["MACD_Hist"] = macd.macd_diff()
    df["MACD_hist"] = df["MACD_Hist"]      # 相容舊欄位

    # ===== RSI / KD =====
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # ===== Bollinger Bands =====
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()

    # 相容舊欄位
    df["BBH"] = df["BB_Upper"]
    df["BBL"] = df["BB_Lower"]

    spread = (df["BB_Upper"] - df["BB_Lower"]).replace(0, float("nan"))
    df["BB_pos"] = (df["Close"] - df["BB_Lower"]) / spread

    def _bb_position(row):
        c = safe_float(row.get("Close"))
        up = safe_float(row.get("BB_Upper"))
        mid = safe_float(row.get("BB_Mid"))
        low_ = safe_float(row.get("BB_Lower"))
        if c is None or up is None or mid is None or low_ is None:
            return None
        if c >= up:
            return "上軌附近"
        if c >= mid:
            return "中軌上方"
        if c >= low_:
            return "中軌下方"
        return "下軌下方"

    df["BB_Position"] = df.apply(_bb_position, axis=1)

    # ===== ATR =====
    atr = AverageTrueRange(high, low, close, window=14)
    df["ATR"] = atr.average_true_range()
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100

    # ===== SMA / EMA =====
    df["MA5"] = SMAIndicator(close, 5).sma_indicator()
    df["SMA5"] = df["MA5"]

    df["MA10"] = SMAIndicator(close, 10).sma_indicator()
    df["SMA10"] = df["MA10"]

    df["SMA20"] = SMAIndicator(close, 20).sma_indicator()
    df["MA20"] = df["SMA20"]

    df["SMA50"] = SMAIndicator(close, 50).sma_indicator()
    df["MA50"] = df["SMA50"]

    df["SMA60"] = SMAIndicator(close, 60).sma_indicator()
    df["MA60"] = df["SMA60"]

    # 季線/年線常用別名
    if len(df) >= 240:
        df["SMA240"] = SMAIndicator(close, 240).sma_indicator()
    else:
        df["SMA240"] = pd.NA
    df["MA240"] = df["SMA240"]

    if len(df) >= 200:
        df["SMA200"] = SMAIndicator(close, 200).sma_indicator()
    else:
        df["SMA200"] = pd.NA

    df["EMA12"] = EMAIndicator(close, 12).ema_indicator()
    df["EMA26"] = EMAIndicator(close, 26).ema_indicator()

    # ===== Volume =====
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    # ===== Returns =====
    df["RET_1D"] = df["Close"].pct_change(1)
    df["RET_5D"] = df["Close"].pct_change(5)
    df["RET_10D"] = df["Close"].pct_change(10)
    df["RET_20D"] = df["Close"].pct_change(20)

    # ===== Flow Proxy =====
    df["FLOW_PROXY"] = df["RET_1D"] * df["Volume"]
    df["FLOW20"] = df["FLOW_PROXY"].rolling(20).sum()

    # ===== Relative Position =====
    df["PRICE_TO_SMA20"] = df["Close"] / df["SMA20"] - 1
    df["PRICE_TO_SMA50"] = df["Close"] / df["SMA50"] - 1
    df["PRICE_TO_SMA60"] = df["Close"] / df["SMA60"] - 1

    if "SMA240" in df.columns:
        df["PRICE_TO_SMA240"] = df["Close"] / df["SMA240"] - 1

    # ===== Future label =====
    df["FWD5_UP"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.03).astype(float)

    # ===== 交叉與買點欄位 =====
    df["MACD_Golden_Cross"] = (
        (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
        & (df["MACD"] > df["MACD_Signal"])
    )

    df["MA5_10_Golden_Cross"] = (
        (df["MA5"].shift(1) <= df["MA10"].shift(1))
        & (df["MA5"] > df["MA10"])
    )

    df["Above_Quarter_Line"] = df["Close"] > df["SMA60"]

    df["Above_BB_Mid"] = df["Close"] > df["BB_Mid"]

    df["Resonance_Score"] = (
        df["MACD_Golden_Cross"].fillna(False).astype(int)
        + df["MA5_10_Golden_Cross"].fillna(False).astype(int)
        + df["Above_Quarter_Line"].fillna(False).astype(int)
        + df["Above_BB_Mid"].fillna(False).astype(int)
        + ((df["RSI"] >= 50) & (df["RSI"] <= 70)).fillna(False).astype(int)
    )

    return df


def is_macd_gold_cross(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False

    if "MACD_Golden_Cross" in df.columns:
        last = df.iloc[-1]
        v = last.get("MACD_Golden_Cross")
        if pd.notna(v):
            return bool(v)

    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]

    return (
        safe_float(prev["MACD"]) <= safe_float(prev["MACD_Signal"])
        and safe_float(last["MACD"]) > safe_float(last["MACD_Signal"])
    )


def is_ma5_ma10_gold_cross(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False

    if "MA5_10_Golden_Cross" in df.columns:
        last = df.iloc[-1]
        v = last.get("MA5_10_Golden_Cross")
        if pd.notna(v):
            return bool(v)

    ma5_col = "MA5" if "MA5" in df.columns else "SMA5"
    ma10_col = "MA10" if "MA10" in df.columns else "SMA10"

    if ma5_col not in df.columns or ma10_col not in df.columns:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]

    return (
        safe_float(prev[ma5_col]) <= safe_float(prev[ma10_col])
        and safe_float(last[ma5_col]) > safe_float(last[ma10_col])
    )


def is_above_quarter_line(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False

    if "Above_Quarter_Line" in df.columns:
        last = df.iloc[-1]
        v = last.get("Above_Quarter_Line")
        if pd.notna(v):
            return bool(v)

    if "SMA60" not in df.columns:
        return False

    last = df.iloc[-1]
    return safe_float(last["Close"]) > safe_float(last["SMA60"])


def golden_cross_score(df: pd.DataFrame) -> Tuple[float, Dict[str, bool]]:
    macd_gc = is_macd_gold_cross(df)
    ma_gc = is_ma5_ma10_gold_cross(df)
    above_q = is_above_quarter_line(df)

    score = 0.0
    if macd_gc:
        score += 12
    if ma_gc:
        score += 10
    if above_q:
        score += 8
    if macd_gc and ma_gc and above_q:
        score += 5

    return min(score, 35.0), {
        "macd_gold_cross": macd_gc,
        "ma5_ma10_gold_cross": ma_gc,
        "above_quarter_line": above_q,
    }
