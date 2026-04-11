from __future__ import annotations

import os
import json
import math
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import twstock

DATA_DIR = "data"
LATEST_JSON = os.path.join(DATA_DIR, "latest_scan.json")
HISTORY_CSV = os.path.join(DATA_DIR, "scan_history.csv")
LEARNING_JSON = os.path.join(DATA_DIR, "ai_learning.json")

# =========================================================
# 基本工具
# =========================================================

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default

# =========================================================
# 學習模型
# =========================================================

DEFAULT_LEARNING = {
    "updated_at": "",
    "weights": {
        # A1 加分條件權重
        "ma60_slope_up":       15,
        "vol_ratio_2x":        10,
        "rsi_healthy":          8,
        "boll_expanding":       8,
        "weekly_above_ma20":   12,
    },
    "thresholds": {
        "vol_ratio_min":        1.5,   # A1 量增最低倍數
        "pullback_pct":         0.02,  # A2 回踩均線容忍範圍 ±2%
        "bonus_score_min":      15,    # 加分條件最低分數
        "success_return_5d":    3.0,
    },
    "rule_stats": {
        "total_labeled":        0,
        "success_rate_5d":      0.0,
        "by_condition":         {},
        "combo_stats":          {},
    },
}

def load_learning() -> dict:
    if not os.path.exists(LEARNING_JSON):
        return DEFAULT_LEARNING.copy()
    try:
        with open(LEARNING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = DEFAULT_LEARNING.copy()
        out["weights"]     = DEFAULT_LEARNING["weights"].copy()
        out["thresholds"]  = DEFAULT_LEARNING["thresholds"].copy()
        out["rule_stats"]  = DEFAULT_LEARNING["rule_stats"].copy()
        out["weights"].update(data.get("weights", {}))
        out["thresholds"].update(data.get("thresholds", {}))
        out["rule_stats"].update(data.get("rule_stats", {}))
        out["updated_at"]  = data.get("updated_at", "")
        return out
    except Exception:
        return DEFAULT_LEARNING.copy()

def save_learning(data: dict) -> None:
    with open(LEARNING_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# 掃描池
# =========================================================

def is_excluded_security(code: str, item) -> bool:
    name      = str(getattr(item, "name",   "")).strip()
    sec_type  = str(getattr(item, "type",   "")).strip()
    market    = str(getattr(item, "market", "")).strip()
    if market not in ["上市", "上櫃"]:
        return True
    if not code.isdigit() or len(code) != 4:
        return True
    if code.startswith("00"):   # 排除 ETF（0050、0056 等）
        return True
    banned_keywords = ["ETN", "權證", "牛熊", "指數", "展牛", "展熊"]
    if any(k in name     for k in banned_keywords):
        return True
    if any(k in sec_type for k in banned_keywords):
        return True
    return False

def get_tw_stock_list() -> list[dict]:
    rows = []
    for code, item in twstock.codes.items():
        try:
            if is_excluded_security(code, item):
                continue
            rows.append({
                "code":   code,
                "name":   str(getattr(item, "name",   "")).strip(),
                "market": str(getattr(item, "market", "")).strip(),
                "type":   str(getattr(item, "type",   "")).strip(),
            })
        except Exception:
            continue
    dedup = {}
    for r in rows:
        dedup[r["code"]] = r
    out = list(dedup.values())
    out.sort(key=lambda x: x["code"])
    return out

def tw_ticker(code: str, market: str) -> str:
    return f"{code}.TW" if market == "上市" else f"{code}.TWO"

# =========================================================
# 指標
# =========================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 均線
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # MACD
    df["EMA12"]     = ema(df["Close"], 12)
    df["EMA26"]     = ema(df["Close"], 26)
    df["DIF"]       = df["EMA12"] - df["EMA26"]
    df["DEA"]       = ema(df["DIF"], 9)
    df["MACD_HIST"] = df["DIF"] - df["DEA"]

    # 布林通道
    mid              = df["Close"].rolling(20).mean()
    std              = df["Close"].rolling(20).std(ddof=0)
    df["BB_MID"]     = mid
    df["BB_UPPER"]   = mid + 2 * std
    df["BB_LOWER"]   = mid - 2 * std
    df["BB_WIDTH"]   = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan)

    # 量能
    df["VOL_MA5"]   = df["Volume"].rolling(5).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA5"].replace(0, np.nan)

    # RSI
    df["RSI14"] = rsi(df["Close"], 14)

    # 近期高低點
    df["RECENT_HIGH_20"] = df["High"].rolling(20).max()
    df["RECENT_LOW_20"]  = df["Low"].rolling(20).min()

    # 週線 MA20 近似（100日週線）
    df["WEEKLY_MA20_APPROX"] = df["Close"].rolling(100).mean()

    return df

# =========================================================
# A1 突破型條件
# =========================================================

def detect_a1_conditions(df: pd.DataFrame, thresholds: dict) -> dict:
    """
    A1 突破型：所有條件必須全中才算 A1
    """
    if len(df) < 65:
        return {}

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    vol_ratio_min = safe_float(thresholds.get("vol_ratio_min", 1.5), 1.5)

    # 1. 股價站上 MA20 且 MA20 向上
    ma20_up = (
        safe_float(curr["Close"]) > safe_float(curr["MA20"]) and
        safe_float(df.iloc[-1]["MA20"]) > safe_float(df.iloc[-5]["MA20"])
    ) if len(df) >= 6 else False

    # 2. 股價站上 MA60
    above_ma60 = safe_float(curr["Close"]) > safe_float(curr["MA60"])

    # 3. MACD HIST 由負轉正（第1根紅柱）
    macd_first_red = (
        safe_float(prev["MACD_HIST"]) <= 0 and
        safe_float(curr["MACD_HIST"]) > 0
    )

    # 4. MA5 上穿 MA10
    ma5_cross_ma10 = (
        safe_float(prev["MA5"]) <= safe_float(prev["MA10"]) and
        safe_float(curr["MA5"])  > safe_float(curr["MA10"])
    )

    # 5. 突破近20日最高點
    breakout_20d_high = (
        safe_float(curr["Close"]) >= safe_float(df.iloc[-2]["RECENT_HIGH_20"])
    ) if len(df) >= 21 else False

    # 6. 成交量 ≥ 5日均量 × 1.5 倍
    vol_expand = safe_float(curr["VOL_RATIO"]) >= vol_ratio_min

    # 7. 收紅K（收盤 > 開盤）
    red_candle = safe_float(curr["Close"]) > safe_float(curr["Open"])

    return {
        "a1_ma20_up":           ma20_up,
        "a1_above_ma60":        above_ma60,
        "a1_macd_first_red":    macd_first_red,
        "a1_ma5_cross_ma10":    ma5_cross_ma10,
        "a1_breakout_20d":      breakout_20d_high,
        "a1_vol_expand":        vol_expand,
        "a1_red_candle":        red_candle,
    }

def is_a1(conds: dict) -> bool:
    """A1：全部必要條件都要成立"""
    required = [
        "a1_ma20_up",
        "a1_above_ma60",
        "a1_macd_first_red",
        "a1_ma5_cross_ma10",
        "a1_breakout_20d",
        "a1_vol_expand",
        "a1_red_candle",
    ]
    return all(conds.get(k, False) for k in required)

# =========================================================
# A2 回踩型條件
# =========================================================

def detect_a2_conditions(df: pd.DataFrame, thresholds: dict) -> dict:
    """
    A2 回踩型：所有條件必須全中才算 A2
    """
    if len(df) < 65:
        return {}

    curr         = df.iloc[-1]
    pullback_pct = safe_float(thresholds.get("pullback_pct", 0.02), 0.02)

    close  = safe_float(curr["Close"])
    ma10   = safe_float(curr["MA10"])
    ma20   = safe_float(curr["MA20"])
    ma60   = safe_float(curr["MA60"])

    # 1. 股價站上 MA60 且 MA60 向上
    ma60_up = (
        close > ma60 and
        safe_float(df.iloc[-1]["MA60"]) > safe_float(df.iloc[-10]["MA60"])
    ) if len(df) >= 11 else False

    # 2. 均線多頭排列：MA5 > MA10 > MA20
    bullish_alignment = (
        safe_float(curr["MA5"]) > ma10 > ma20
    )

    # 3. 股價回踩 MA10 或 MA20（±2%以內）
    near_ma10 = abs(close - ma10) / ma10 <= pullback_pct if ma10 > 0 else False
    near_ma20 = abs(close - ma20) / ma20 <= pullback_pct if ma20 > 0 else False
    pullback_to_ma = near_ma10 or near_ma20

    # 4. MACD HIST > 0（仍在紅柱中）
    macd_positive = safe_float(curr["MACD_HIST"]) > 0

    # 5. 縮量回踩（今日量 < 昨日量）
    vol_shrink = safe_float(curr["Volume"]) < safe_float(df.iloc[-2]["Volume"])

    # 6. 近3日低點高於前3日低點（底部墊高）
    if len(df) >= 6:
        recent_low  = df.iloc[-3:]["Low"].min()
        prev_low    = df.iloc[-6:-3]["Low"].min()
        higher_lows = safe_float(recent_low) > safe_float(prev_low)
    else:
        higher_lows = False

    return {
        "a2_ma60_up":           ma60_up,
        "a2_bullish_alignment": bullish_alignment,
        "a2_pullback_to_ma":    pullback_to_ma,
        "a2_macd_positive":     macd_positive,
        "a2_vol_shrink":        vol_shrink,
        "a2_higher_lows":       higher_lows,
    }

def is_a2(conds: dict) -> bool:
    """A2：全部必要條件都要成立"""
    required = [
        "a2_ma60_up",
        "a2_bullish_alignment",
        "a2_pullback_to_ma",
        "a2_macd_positive",
        "a2_vol_shrink",
        "a2_higher_lows",
    ]
    return all(conds.get(k, False) for k in required)

# =========================================================
# 加分條件（A1、A2 共用）
# =========================================================

def calc_bonus_score(df: pd.DataFrame, weights: dict) -> tuple[int, list[str]]:
    """
    加分條件：影響排序，門檻 ≥ 15 分才算 A 級
    """
    if len(df) < 65:
        return 0, []

    curr  = df.iloc[-1]
    score = 0
    tags  = []

    # MA60 斜率向上（近10日持續上升）
    if len(df) >= 11:
        ma60_vals = [safe_float(df.iloc[-(i+1)]["MA60"]) for i in range(10)]
        if all(ma60_vals[i] > ma60_vals[i+1] for i in range(9)):
            w = safe_int(weights.get("ma60_slope_up", 15))
            score += w
            tags.append(f"季線持續上揚(+{w})")

    # 量比 ≥ 2 倍
    if safe_float(curr["VOL_RATIO"]) >= 2.0:
        w = safe_int(weights.get("vol_ratio_2x", 10))
        score += w
        tags.append(f"量比≥2倍(+{w})")

    # RSI 在 50~70（健康動能區）
    rsi_val = safe_float(curr["RSI14"])
    if 50 <= rsi_val <= 70:
        w = safe_int(weights.get("rsi_healthy", 8))
        score += w
        tags.append(f"RSI健康區({rsi_val:.1f})(+{w})")

    # 布林通道寬度擴張（今日 > 5日前）
    if len(df) >= 6:
        bw_now  = safe_float(curr["BB_WIDTH"])
        bw_prev = safe_float(df.iloc[-6]["BB_WIDTH"])
        if bw_now > bw_prev > 0:
            w = safe_int(weights.get("boll_expanding", 8))
            score += w
            tags.append(f"布林擴張(+{w})")

    # 週線近似站上 MA20（100日均線）
    weekly_ma20 = safe_float(curr["WEEKLY_MA20_APPROX"])
    if weekly_ma20 > 0 and safe_float(curr["Close"]) > weekly_ma20:
        w = safe_int(weights.get("weekly_above_ma20", 12))
        score += w
        tags.append(f"週線站上MA20(+{w})")

    return score, tags

# =========================================================
# 分級決定
# =========================================================

def decide_grade(
    a1_conds: dict,
    a2_conds: dict,
    bonus_score: int,
    thresholds: dict,
) -> str:
    bonus_min = safe_int(thresholds.get("bonus_score_min", 15), 15)

    if is_a1(a1_conds) and bonus_score >= bonus_min:
        return "A1"
    if is_a2(a2_conds) and bonus_score >= bonus_min:
        return "A2"
    return "-"

# =========================================================
# 抓價
# =========================================================

# =========================================================
# 抓價（yfinance + Cookie Session 修正）
# =========================================================

import requests as _requests

def _make_yf_session():
    """建立帶有 Yahoo Finance Cookie 的 Session，解決 Streamlit Cloud 被擋問題"""
    session = _requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    try:
        # 先取得 Yahoo Finance Cookie（關鍵步驟）
        session.get("https://finance.yahoo.com", timeout=8)
    except Exception:
        pass
    return session

_YF_SESSION = None   # 全域 session，整個掃描共用

def _get_session():
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = _make_yf_session()
    return _YF_SESSION

def fetch_price_history(symbol: str, period: str = "12mo", market: str = "上市") -> tuple[pd.DataFrame | None, str]:
    try:
        session = _get_session()
        ticker  = yf.Ticker(symbol, session=session)
        df      = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if df is None or df.empty:
            return None, "no_data"
        # history() 欄位標準化
        df = df.rename(columns={"stock splits": "Stock Splits"})
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                return None, "missing_columns"
        df = df[needed].dropna().copy()
        if len(df) < 80:
            return None, "too_short"
        return df, ""
    except Exception as e:
        return None, str(e)

def fetch_forward_df(symbol: str, market: str = "上市") -> pd.DataFrame | None:
    try:
        session = _get_session()
        ticker  = yf.Ticker(symbol, session=session)
        df      = ticker.history(
            period="18mo",
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if df is None or df.empty:
            return None
        if "Close" not in df.columns:
            return None
        out = df[["Close"]].copy().dropna().reset_index()
        out.columns = ["Date", "Close"]
        out["Date"] = pd.to_datetime(out["Date"]).dt.date
        return out
    except Exception:
        return None

# =========================================================
# 買賣點價位分析
# =========================================================

def build_trade_prices(df: pd.DataFrame, grade: str) -> dict:
    curr = df.iloc[-1]

    close       = safe_float(curr["Close"])
    ma10        = safe_float(curr["MA10"])
    ma20        = safe_float(curr["MA20"])
    ma60        = safe_float(curr["MA60"])
    bb_upper    = safe_float(curr["BB_UPPER"])
    bb_mid      = safe_float(curr["BB_MID"])
    recent_high = safe_float(df.tail(20)["High"].max())
    recent_low  = safe_float(df.tail(20)["Low"].min())

    if grade == "A1":
        # 突破型：積極買點＝當前收盤（突破點），回踩買點＝MA10
        aggressive_buy  = round(close, 2)
        pullback_buy    = round(ma10, 2)
        conservative_buy = round(ma20, 2)
        stop_loss_short = round(ma10, 2)
        stop_loss_wave  = round(ma20, 2)
    else:
        # 回踩型：積極買點＝MA10，回踩買點＝MA20
        aggressive_buy  = round(ma10, 2)
        pullback_buy    = round(ma20, 2)
        conservative_buy = round(ma60, 2)
        stop_loss_short = round(ma20, 2)
        stop_loss_wave  = round(ma60, 2)

    return {
        "aggressive_buy_price":  aggressive_buy,
        "pullback_buy_price":    pullback_buy,
        "conservative_buy_price": conservative_buy,
        "sell_price_1":          round(bb_upper, 2),
        "sell_price_2":          round(recent_high * 1.05, 2),  # 突破高點再上5%
        "stop_loss_short":       stop_loss_short,
        "stop_loss_wave":        stop_loss_wave,
        "stop_loss_hard":        round(ma60, 2),
        "support_1":             round(ma10, 2),
        "support_2":             round(ma20, 2),
        "support_3":             round(ma60, 2),
        "resistance_1":          round(bb_upper, 2),
        "resistance_2":          round(recent_high, 2),
        "recent_low_20":         round(recent_low, 2),
    }

def build_text_notes(grade: str, prices: dict) -> tuple[str, str, str]:
    label = "A1 突破進場" if grade == "A1" else "A2 回踩買點"
    buy_note = (
        f"{label}｜"
        f"積極 {prices['aggressive_buy_price']:.2f}、"
        f"回踩 {prices['pullback_buy_price']:.2f}、"
        f"保守 {prices['conservative_buy_price']:.2f}"
    )
    stop_note = (
        f"短線停損 {prices['stop_loss_short']:.2f}、"
        f"波段停損 {prices['stop_loss_wave']:.2f}、"
        f"嚴格停損 {prices['stop_loss_hard']:.2f}"
    )
    target_note = (
        f"第一賣點 {prices['sell_price_1']:.2f}、"
        f"第二賣點 {prices['sell_price_2']:.2f}"
    )
    return buy_note, stop_note, target_note

# =========================================================
# 分析單檔
# =========================================================

def analyze_one(stock: dict, learning: dict) -> tuple[dict | None, str]:
    code   = stock["code"]
    name   = stock["name"]
    market = stock["market"]
    symbol = tw_ticker(code, market)

    raw, err = fetch_price_history(symbol, period="12mo", market=market)
    if raw is None or raw.empty:
        return None, err or "fetch_failed"

    df = enrich_indicators(raw).dropna().copy()
    if len(df) < 65:
        return None, "too_short_after_indicators"

    a1_conds    = detect_a1_conditions(df, learning["thresholds"])
    a2_conds    = detect_a2_conditions(df, learning["thresholds"])
    bonus_score, bonus_tags = calc_bonus_score(df, learning["weights"])

    grade = decide_grade(a1_conds, a2_conds, bonus_score, learning["thresholds"])

    # 只保留 A1 / A2
    if grade == "-":
        return None, "not_qualified"

    curr   = df.iloc[-1]
    prices = build_trade_prices(df, grade)
    buy_note, stop_note, target_note = build_text_notes(grade, prices)

    # 整合所有觸發條件說明
    all_conds = {**a1_conds, **a2_conds}
    cond_labels = {
        "a1_ma20_up":           "MA20向上",
        "a1_above_ma60":        "站上季線",
        "a1_macd_first_red":    "MACD第1紅柱",
        "a1_ma5_cross_ma10":    "MA5穿MA10",
        "a1_breakout_20d":      "突破20日高點",
        "a1_vol_expand":        "量增1.5倍",
        "a1_red_candle":        "收紅K",
        "a2_ma60_up":           "季線向上",
        "a2_bullish_alignment": "均線多頭排列",
        "a2_pullback_to_ma":    "回踩均線",
        "a2_macd_positive":     "MACD紅柱中",
        "a2_vol_shrink":        "縮量回踩",
        "a2_higher_lows":       "底部墊高",
    }
    reasons = [cond_labels[k] for k, v in all_conds.items() if v]
    reasons += bonus_tags

    row = {
        "scan_id":    f"{today_str()}_{code}",
        "scan_date":  today_str(),
        "scan_time":  now_str(),
        "data_date":  str(df.index[-1].date()),
        "code":       code,
        "name":       name,
        "market":     market,
        "type":       stock.get("type", ""),
        "symbol":     symbol,
        "grade":      grade,
        "score":      int(bonus_score),
        "reasons":    "、".join(reasons),
        "close":      round(safe_float(curr["Close"]), 2),
        "open":       round(safe_float(curr["Open"]),  2),
        "high":       round(safe_float(curr["High"]),  2),
        "low":        round(safe_float(curr["Low"]),   2),
        "volume":     safe_int(curr["Volume"]),
        "vol_ratio":  round(safe_float(curr["VOL_RATIO"]), 2),
        "rsi14":      round(safe_float(curr["RSI14"]), 1),
        "dif":        round(safe_float(curr["DIF"]),   4),
        "dea":        round(safe_float(curr["DEA"]),   4),
        "hist":       round(safe_float(curr["MACD_HIST"]), 4),
        "ma5":        round(safe_float(curr["MA5"]),   2),
        "ma10":       round(safe_float(curr["MA10"]),  2),
        "ma20":       round(safe_float(curr["MA20"]),  2),
        "ma60":       round(safe_float(curr["MA60"]),  2),
        "bb_upper":   round(safe_float(curr["BB_UPPER"]), 2),
        "bb_mid":     round(safe_float(curr["BB_MID"]),   2),
        "bb_lower":   round(safe_float(curr["BB_LOWER"]), 2),
        "bb_width":   round(safe_float(curr["BB_WIDTH"]), 4),
        **prices,
        "buy_note":    buy_note,
        "stop_note":   stop_note,
        "target_note": target_note,
        # A1 條件欄位（供學習追蹤）
        "cond_a1_ma20_up":          int(a1_conds.get("a1_ma20_up", False)),
        "cond_a1_above_ma60":       int(a1_conds.get("a1_above_ma60", False)),
        "cond_a1_macd_first_red":   int(a1_conds.get("a1_macd_first_red", False)),
        "cond_a1_ma5_cross_ma10":   int(a1_conds.get("a1_ma5_cross_ma10", False)),
        "cond_a1_breakout_20d":     int(a1_conds.get("a1_breakout_20d", False)),
        "cond_a1_vol_expand":       int(a1_conds.get("a1_vol_expand", False)),
        "cond_a1_red_candle":       int(a1_conds.get("a1_red_candle", False)),
        # A2 條件欄位
        "cond_a2_ma60_up":          int(a2_conds.get("a2_ma60_up", False)),
        "cond_a2_bullish_alignment":int(a2_conds.get("a2_bullish_alignment", False)),
        "cond_a2_pullback_to_ma":   int(a2_conds.get("a2_pullback_to_ma", False)),
        "cond_a2_macd_positive":    int(a2_conds.get("a2_macd_positive", False)),
        "cond_a2_vol_shrink":       int(a2_conds.get("a2_vol_shrink", False)),
        "cond_a2_higher_lows":      int(a2_conds.get("a2_higher_lows", False)),
        # 回測欄位
        "ret_3d":    np.nan,
        "ret_5d":    np.nan,
        "ret_10d":   np.nan,
        "success_5d": np.nan,
    }

    return row, ""

# =========================================================
# AI 學習背景累積
# =========================================================

def load_history_df() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

def fetch_forward_df(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period="18mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if "Close" not in df.columns:
            return None
        out = df[["Close"]].copy().dropna().reset_index()
        out["Date"] = pd.to_datetime(out["Date"]).dt.date
        return out
    except Exception:
        return None

def backfill_history_returns(history_df: pd.DataFrame, learning: dict) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    success_threshold = safe_float(learning["thresholds"].get("success_return_5d", 3.0), 3.0)
    unique_symbols = history_df["symbol"].dropna().astype(str).unique().tolist()
    price_cache = {}
    for symbol in unique_symbols:
        mkt = "上市" if symbol.endswith(".TW") else "上櫃"
        price_cache[symbol] = fetch_forward_df(symbol, market=mkt)

    for idx, row in history_df.iterrows():
        try:
            if not pd.isna(row.get("ret_5d", np.nan)):
                continue
            symbol     = str(row.get("symbol", ""))
            scan_date  = pd.to_datetime(row.get("scan_date")).date()
            entry_close = safe_float(row.get("close"), np.nan)
            if pd.isna(entry_close):
                continue
            fdf = price_cache.get(symbol)
            if fdf is None or fdf.empty:
                continue
            future = fdf[fdf["Date"] > scan_date].sort_values("Date").reset_index(drop=True)
            if len(future) >= 3:
                c3 = safe_float(future.iloc[2]["Close"], np.nan)
                history_df.at[idx, "ret_3d"] = round((c3 / entry_close - 1) * 100, 2)
            if len(future) >= 5:
                c5  = safe_float(future.iloc[4]["Close"], np.nan)
                ret5 = round((c5 / entry_close - 1) * 100, 2)
                history_df.at[idx, "ret_5d"]    = ret5
                history_df.at[idx, "success_5d"] = int(ret5 >= success_threshold)
            if len(future) >= 10:
                c10 = safe_float(future.iloc[9]["Close"], np.nan)
                history_df.at[idx, "ret_10d"] = round((c10 / entry_close - 1) * 100, 2)
        except Exception:
            continue
    return history_df

def rebuild_learning(history_df: pd.DataFrame, old_learning: dict) -> dict:
    learning = DEFAULT_LEARNING.copy()
    learning["weights"]    = DEFAULT_LEARNING["weights"].copy()
    learning["thresholds"] = DEFAULT_LEARNING["thresholds"].copy()
    learning["rule_stats"] = DEFAULT_LEARNING["rule_stats"].copy()
    learning["weights"].update(old_learning.get("weights", {}))
    learning["thresholds"].update(old_learning.get("thresholds", {}))

    if history_df.empty or "success_5d" not in history_df.columns:
        learning["updated_at"] = now_str()
        return learning

    labeled = history_df.dropna(subset=["success_5d"]).copy()
    if labeled.empty:
        learning["updated_at"] = now_str()
        return learning

    labeled["success_5d"] = labeled["success_5d"].astype(int)

    # ── 加分條件勝率分析 ──────────────────────────────
    bonus_map = {
        "cond_a1_vol_expand":        "vol_ratio_2x",
        "cond_a1_breakout_20d":      "ma60_slope_up",
    }
    by_condition = {}
    new_weights  = learning["weights"].copy()

    for hist_col, w_key in bonus_map.items():
        if hist_col not in labeled.columns:
            continue
        sub = labeled[labeled[hist_col] == 1].copy()
        if len(sub) == 0:
            continue
        win_rate = round(sub["success_5d"].mean() * 100, 2)
        avg_ret  = round(sub["ret_5d"].mean(), 2) if "ret_5d" in sub.columns else 0.0
        by_condition[w_key] = {
            "samples":         int(len(sub)),
            "success_rate_5d": win_rate,
            "avg_ret_5d":      avg_ret,
        }
        base = DEFAULT_LEARNING["weights"].get(w_key, 10)
        if len(sub) >= 20:
            # 比例式調整：勝率映射到 0.6~1.4 倍基礎權重
            multiplier = max(0.6, min(1.4, win_rate / 55.0))
            new_weights[w_key] = max(1, round(base * multiplier))

    # ── A1 vs A2 勝率比較 ────────────────────────────
    grade_stats = {}
    for g in ["A1", "A2"]:
        sub = labeled[labeled["grade"] == g] if "grade" in labeled.columns else pd.DataFrame()
        if len(sub) >= 5:
            grade_stats[g] = {
                "samples":         int(len(sub)),
                "success_rate_5d": round(sub["success_5d"].mean() * 100, 2),
                "avg_ret_5d":      round(sub["ret_5d"].mean(), 2) if "ret_5d" in sub.columns else 0.0,
            }

    # ── 動態調整加分門檻 ─────────────────────────────
    # 若 A1 整體勝率 < 50%，提高加分門檻（讓標準更嚴）
    a1_stats = grade_stats.get("A1", {})
    if a1_stats.get("samples", 0) >= 30:
        current_min = safe_int(learning["thresholds"].get("bonus_score_min", 15))
        if a1_stats["success_rate_5d"] < 50:
            learning["thresholds"]["bonus_score_min"] = min(current_min + 3, 35)
        elif a1_stats["success_rate_5d"] > 65:
            learning["thresholds"]["bonus_score_min"] = max(current_min - 2, 10)

    learning["weights"]    = new_weights
    learning["rule_stats"] = {
        "total_labeled":   int(len(labeled)),
        "success_rate_5d": round(labeled["success_5d"].mean() * 100, 2),
        "by_condition":    by_condition,
        "grade_stats":     grade_stats,
    }
    learning["updated_at"] = now_str()
    return learning

# =========================================================
# 儲存
# =========================================================

def save_latest(payload: dict) -> None:
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# =========================================================
# 主掃描
# =========================================================

def summarize(df: pd.DataFrame, pool_total: int, success_count: int, failed_count: int) -> dict:
    return {
        "pool_total":    int(pool_total),
        "success_count": int(success_count),
        "failed_count":  int(failed_count),
        "display_total": int(len(df)),
        "A1_count":      int((df["grade"] == "A1").sum()) if not df.empty else 0,
        "A2_count":      int((df["grade"] == "A2").sum()) if not df.empty else 0,
    }

def run_scan(max_workers: int = 12) -> dict:
    ensure_data_dir()
    learning = load_learning()
    stocks   = get_tw_stock_list()
    pool_total = len(stocks)

    rows          = []
    fail_rows     = []
    not_qualified = 0   # 分析成功但不符 A1/A2 條件

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, s, learning): s for s in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                item, err = future.result()
                if item is not None:
                    rows.append(item)
                elif err == "not_qualified":
                    not_qualified += 1      # 正常分析完，只是沒達標
                else:
                    fail_rows.append({
                        "code":   stock["code"],
                        "name":   stock["name"],
                        "market": stock["market"],
                        "symbol": tw_ticker(stock["code"], stock["market"]),
                        "reason": err or "unknown",
                    })
            except Exception as e:
                fail_rows.append({
                    "code":   stock["code"],
                    "name":   stock["name"],
                    "market": stock["market"],
                    "symbol": tw_ticker(stock["code"], stock["market"]),
                    "reason": str(e),
                })

    # 成功分析數 = 有A結果 + 分析完但未達標（兩者都是正常跑完的）
    success_count = len(rows) + not_qualified
    failed_count  = len(fail_rows)

    if not rows:
        payload = {
            "updated_at": now_str(),
            "status":     "ok",   # 掃描本身成功，只是今天沒有 A 級
            "message":    f"今日掃描完成，{success_count} 檔分析成功，目前沒有符合 A1/A2 條件的個股。",
            "summary":    summarize(pd.DataFrame(), pool_total, success_count, failed_count),
            "learning":   learning,
            "results":    [],
            "failed_symbols": fail_rows[:500],
        }
        save_latest(payload)
        return payload

    df = pd.DataFrame(rows)

    # A1 優先，同級內按加分高低排序
    grade_rank = {"A1": 0, "A2": 1}
    df["grade_rank"] = df["grade"].map(grade_rank).fillna(9)
    df = df.sort_values(
        by=["grade_rank", "score", "vol_ratio"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    today_df   = df.drop(columns=["grade_rank"]).copy()
    history_df = load_history_df()
    history_df = pd.concat([history_df, today_df], ignore_index=True)
    if "scan_id" in history_df.columns:
        history_df = history_df.drop_duplicates(subset=["scan_id"], keep="last")

    history_df  = backfill_history_returns(history_df, learning)
    history_df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    new_learning = rebuild_learning(history_df, learning)
    save_learning(new_learning)

    payload = {
        "updated_at": now_str(),
        "status":     "ok",
        "message":    "",
        "summary":    summarize(today_df, pool_total, success_count, failed_count),
        "learning":   new_learning,
        "results":    today_df.to_dict(orient="records"),
        "failed_symbols": fail_rows[:500],
    }
    save_latest(payload)
    return payload


if __name__ == "__main__":
    try:
        result = run_scan(max_workers=12)
        print(json.dumps({
            "updated_at":  result.get("updated_at"),
            "status":      result.get("status"),
            "summary":     result.get("summary", {}),
            "learning_total_labeled": result.get("learning", {}).get("rule_stats", {}).get("total_labeled", 0),
        }, ensure_ascii=False, indent=2))
    except Exception as e:
        ensure_data_dir()
        payload = {
            "updated_at": now_str(),
            "status":     "error",
            "message":    f"掃描失敗: {e}",
            "traceback":  traceback.format_exc(),
            "summary":    {},
            "learning":   DEFAULT_LEARNING.copy(),
            "results":    [],
            "failed_symbols": [],
        }
        save_latest(payload)
        raise
