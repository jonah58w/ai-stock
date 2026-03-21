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
# 學習模型（保留，不干擾）
# =========================================================
DEFAULT_LEARNING = {
    "updated_at": "",
    "weights": {
        "macd_gc": 28,
        "first_red": 18,
        "ma5_ma10_gc": 18,
        "above_ma60": 12,
        "ma60_up": 10,
        "vol_expand": 12,
        "boll_breakout": 10,
        "price_above_ma20": 6,
        "close_above_recent_high": 12,
    },
    "thresholds": {
        "vol_ratio_min": 1.2,
        "success_return_5d": 3.0,
        "a_score_min": 78,
        "b_score_min": 45,
    },
    "rule_stats": {
        "total_labeled": 0,
        "success_rate_5d": 0.0,
        "by_condition": {},
    },
}


def load_learning() -> dict:
    if not os.path.exists(LEARNING_JSON):
        return DEFAULT_LEARNING.copy()
    try:
        with open(LEARNING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = DEFAULT_LEARNING.copy()
        out["weights"] = DEFAULT_LEARNING["weights"].copy()
        out["thresholds"] = DEFAULT_LEARNING["thresholds"].copy()
        out["rule_stats"] = DEFAULT_LEARNING["rule_stats"].copy()
        out["weights"].update(data.get("weights", {}))
        out["thresholds"].update(data.get("thresholds", {}))
        out["rule_stats"].update(data.get("rule_stats", {}))
        out["updated_at"] = data.get("updated_at", "")
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
    name = str(getattr(item, "name", "")).strip()
    sec_type = str(getattr(item, "type", "")).strip()
    market = str(getattr(item, "market", "")).strip()

    if market not in ["上市", "上櫃"]:
        return True
    if not code.isdigit() or len(code) != 4:
        return True

    banned_keywords = ["ETN", "權證", "牛熊", "指數", "展牛", "展熊"]
    if any(k in name for k in banned_keywords):
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
                "code": code,
                "name": str(getattr(item, "name", "")).strip(),
                "market": str(getattr(item, "market", "")).strip(),
                "type": str(getattr(item, "type", "")).strip(),
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


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    df["EMA12"] = ema(df["Close"], 12)
    df["EMA26"] = ema(df["Close"], 26)
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["DEA"] = ema(df["DIF"], 9)
    df["MACD_HIST"] = df["DIF"] - df["DEA"]

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2 * std
    df["BB_LOWER"] = mid - 2 * std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan)

    df["VOL_MA5"] = df["Volume"].rolling(5).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA5"].replace(0, np.nan)

    df["RECENT_HIGH_20"] = df["High"].rolling(20).max()
    df["RECENT_LOW_20"] = df["Low"].rolling(20).min()

    return df


# =========================================================
# 條件
# =========================================================
def detect_conditions(df: pd.DataFrame, thresholds: dict) -> dict:
    if len(df) < 65:
        return {}

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    vol_ratio_min = safe_float(thresholds.get("vol_ratio_min", 1.2), 1.2)

    return {
        "macd_gc": safe_float(prev["DIF"]) <= safe_float(prev["DEA"]) and safe_float(curr["DIF"]) > safe_float(curr["DEA"]),
        "first_red": safe_float(prev["MACD_HIST"]) <= 0 and safe_float(curr["MACD_HIST"]) > 0,
        "ma5_ma10_gc": safe_float(prev["MA5"]) <= safe_float(prev["MA10"]) and safe_float(curr["MA5"]) > safe_float(curr["MA10"]),
        "above_ma60": safe_float(curr["Close"]) > safe_float(curr["MA60"]),
        "ma60_up": safe_float(df.iloc[-1]["MA60"]) > safe_float(df.iloc[-5]["MA60"]) > 0 if len(df) >= 6 else False,
        "vol_expand": safe_float(curr["VOL_RATIO"]) >= vol_ratio_min,
        "boll_breakout": safe_float(curr["Close"]) > safe_float(curr["BB_UPPER"]),
        "price_above_ma20": safe_float(curr["Close"]) > safe_float(curr["MA20"]),
        "close_above_recent_high": safe_float(curr["Close"]) >= safe_float(df.iloc[-2]["RECENT_HIGH_20"]) if len(df) >= 21 else False,
    }


def calc_score(conditions: dict, weights: dict) -> tuple[int, list[str]]:
    labels = {
        "macd_gc": "MACD黃金交叉",
        "first_red": "MACD第1根紅柱",
        "ma5_ma10_gc": "5日/10日黃金交叉",
        "above_ma60": "站上季線",
        "ma60_up": "季線向上",
        "vol_expand": "量增",
        "boll_breakout": "突破布林上軌",
        "price_above_ma20": "股價站上20日線",
        "close_above_recent_high": "突破近20日高點",
    }

    score = 0
    reasons = []
    for k, v in conditions.items():
        if v:
            score += safe_int(weights.get(k, 0))
            reasons.append(labels.get(k, k))
    return score, reasons


def decide_grade(conditions: dict, score: int, thresholds: dict) -> str:
    a_score_min = safe_int(thresholds.get("a_score_min", 78), 78)
    b_score_min = safe_int(thresholds.get("b_score_min", 45), 45)

    cond_a = (
        conditions.get("macd_gc", False)
        and conditions.get("first_red", False)
        and conditions.get("ma5_ma10_gc", False)
        and conditions.get("above_ma60", False)
        and conditions.get("ma60_up", False)
        and conditions.get("vol_expand", False)
    )

    cond_b = (
        conditions.get("macd_gc", False)
        or (conditions.get("boll_breakout", False) and conditions.get("vol_expand", False))
        or conditions.get("close_above_recent_high", False)
    )

    if cond_a and score >= a_score_min:
        return "A"
    if cond_b and score >= b_score_min:
        return "B"
    if conditions.get("above_ma60", False) or conditions.get("price_above_ma20", False) or score >= 25:
        return "C"
    return "-"


# =========================================================
# 抓價
# =========================================================
def fetch_price_history(symbol: str, period: str = "12mo") -> tuple[pd.DataFrame | None, str]:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None, "no_data"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

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


# =========================================================
# 買賣點價位分析
# =========================================================
def build_trade_prices(df: pd.DataFrame) -> dict:
    curr = df.iloc[-1]

    close = safe_float(curr["Close"])
    ma5 = safe_float(curr["MA5"])
    ma10 = safe_float(curr["MA10"])
    ma20 = safe_float(curr["MA20"])
    ma60 = safe_float(curr["MA60"])
    bb_upper = safe_float(curr["BB_UPPER"])
    bb_mid = safe_float(curr["BB_MID"])
    recent_high = safe_float(df.tail(20)["High"].max())
    recent_low = safe_float(df.tail(20)["Low"].min())

    aggressive_buy_price = max(close, recent_high)
    pullback_buy_price = ma10
    conservative_buy_price = bb_mid
    wave_buy_price = max(ma20, ma60)

    sell_price_1 = bb_upper
    sell_price_2 = recent_high
    stop_loss_short = ma10
    stop_loss_wave = ma20
    stop_loss_hard = ma60

    return {
        "aggressive_buy_price": round(aggressive_buy_price, 2),
        "pullback_buy_price": round(pullback_buy_price, 2),
        "conservative_buy_price": round(conservative_buy_price, 2),
        "wave_buy_price": round(wave_buy_price, 2),
        "sell_price_1": round(sell_price_1, 2),
        "sell_price_2": round(sell_price_2, 2),
        "stop_loss_short": round(stop_loss_short, 2),
        "stop_loss_wave": round(stop_loss_wave, 2),
        "stop_loss_hard": round(stop_loss_hard, 2),
        "support_1": round(ma10, 2),
        "support_2": round(ma20, 2),
        "support_3": round(ma60, 2),
        "resistance_1": round(bb_upper, 2),
        "resistance_2": round(recent_high, 2),
        "recent_low_20": round(recent_low, 2),
    }


def build_text_notes(grade: str, prices: dict) -> tuple[str, str, str]:
    buy_note = (
        f"積極買點 {prices['aggressive_buy_price']:.2f}、"
        f"回踩買點 {prices['pullback_buy_price']:.2f}、"
        f"保守買點 {prices['conservative_buy_price']:.2f}"
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

    if grade == "A":
        buy_note = "A級：" + buy_note
        target_note = "A級：" + target_note
    elif grade == "B":
        buy_note = "B級：" + buy_note

    return buy_note, stop_note, target_note


# =========================================================
# 分析單檔
# =========================================================
def analyze_one(stock: dict, learning: dict) -> tuple[dict | None, str]:
    code = stock["code"]
    name = stock["name"]
    market = stock["market"]
    symbol = tw_ticker(code, market)

    raw, err = fetch_price_history(symbol, period="12mo")
    if raw is None or raw.empty:
        return None, err or "fetch_failed"

    df = enrich_indicators(raw).dropna().copy()
    if len(df) < 65:
        return None, "too_short_after_indicators"

    conditions = detect_conditions(df, learning["thresholds"])
    if not conditions:
        return None, "conditions_empty"

    score, reasons = calc_score(conditions, learning["weights"])
    grade = decide_grade(conditions, score, learning["thresholds"])

    curr = df.iloc[-1]
    prices = build_trade_prices(df)
    buy_note, stop_note, target_note = build_text_notes(grade, prices)

    row = {
        "scan_id": f"{today_str()}_{code}",
        "scan_date": today_str(),
        "scan_time": now_str(),
        "data_date": str(df.index[-1].date()),
        "code": code,
        "name": name,
        "market": market,
        "type": stock.get("type", ""),
        "symbol": symbol,
        "grade": grade,
        "score": int(score),
        "reasons": "、".join(reasons),
        "close": round(safe_float(curr["Close"]), 2),
        "open": round(safe_float(curr["Open"]), 2),
        "high": round(safe_float(curr["High"]), 2),
        "low": round(safe_float(curr["Low"]), 2),
        "volume": safe_int(curr["Volume"]),
        "vol_ratio": round(safe_float(curr["VOL_RATIO"]), 2),
        "dif": round(safe_float(curr["DIF"]), 4),
        "dea": round(safe_float(curr["DEA"]), 4),
        "hist": round(safe_float(curr["MACD_HIST"]), 4),
        "ma5": round(safe_float(curr["MA5"]), 2),
        "ma10": round(safe_float(curr["MA10"]), 2),
        "ma20": round(safe_float(curr["MA20"]), 2),
        "ma60": round(safe_float(curr["MA60"]), 2),
        "bb_upper": round(safe_float(curr["BB_UPPER"]), 2),
        "bb_mid": round(safe_float(curr["BB_MID"]), 2),
        "bb_lower": round(safe_float(curr["BB_LOWER"]), 2),
        "bb_width": round(safe_float(curr["BB_WIDTH"]), 4),
        **prices,
        "buy_note": buy_note,
        "stop_note": stop_note,
        "target_note": target_note,
        "cond_macd_gc": int(conditions["macd_gc"]),
        "cond_first_red": int(conditions["first_red"]),
        "cond_ma5_ma10_gc": int(conditions["ma5_ma10_gc"]),
        "cond_above_ma60": int(conditions["above_ma60"]),
        "cond_ma60_up": int(conditions["ma60_up"]),
        "cond_vol_expand": int(conditions["vol_expand"]),
        "cond_boll_breakout": int(conditions["boll_breakout"]),
        "cond_price_above_ma20": int(conditions["price_above_ma20"]),
        "cond_close_above_recent_high": int(conditions["close_above_recent_high"]),
        "ret_3d": np.nan,
        "ret_5d": np.nan,
        "ret_10d": np.nan,
        "success_5d": np.nan,
    }
    return row, ""


# =========================================================
# AI學習背景累積
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
    price_cache = {symbol: fetch_forward_df(symbol) for symbol in unique_symbols}

    for idx, row in history_df.iterrows():
        try:
            if not pd.isna(row.get("ret_5d", np.nan)):
                continue

            symbol = str(row.get("symbol", ""))
            scan_date = pd.to_datetime(row.get("scan_date")).date()
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
                c5 = safe_float(future.iloc[4]["Close"], np.nan)
                ret5 = round((c5 / entry_close - 1) * 100, 2)
                history_df.at[idx, "ret_5d"] = ret5
                history_df.at[idx, "success_5d"] = int(ret5 >= success_threshold)

            if len(future) >= 10:
                c10 = safe_float(future.iloc[9]["Close"], np.nan)
                history_df.at[idx, "ret_10d"] = round((c10 / entry_close - 1) * 100, 2)
        except Exception:
            continue

    return history_df


def rebuild_learning(history_df: pd.DataFrame, old_learning: dict) -> dict:
    learning = DEFAULT_LEARNING.copy()
    learning["weights"] = DEFAULT_LEARNING["weights"].copy()
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

    condition_map = {
        "cond_macd_gc": "macd_gc",
        "cond_first_red": "first_red",
        "cond_ma5_ma10_gc": "ma5_ma10_gc",
        "cond_above_ma60": "above_ma60",
        "cond_ma60_up": "ma60_up",
        "cond_vol_expand": "vol_expand",
        "cond_boll_breakout": "boll_breakout",
        "cond_price_above_ma20": "price_above_ma20",
        "cond_close_above_recent_high": "close_above_recent_high",
    }

    by_condition = {}
    new_weights = learning["weights"].copy()

    for hist_col, w_key in condition_map.items():
        if hist_col not in labeled.columns:
            continue

        sub = labeled[labeled[hist_col] == 1].copy()
        if len(sub) == 0:
            continue

        win_rate = round(sub["success_5d"].mean() * 100, 2)
        avg_ret = round(sub["ret_5d"].mean(), 2) if "ret_5d" in sub.columns else 0.0

        by_condition[w_key] = {
            "samples": int(len(sub)),
            "success_rate_5d": win_rate,
            "avg_ret_5d": avg_ret,
        }

        base = DEFAULT_LEARNING["weights"][w_key]
        if len(sub) >= 20:
            if win_rate >= 60:
                new_weights[w_key] = min(base + 5, base + 8)
            elif win_rate >= 55:
                new_weights[w_key] = min(base + 2, base + 6)
            elif win_rate < 45:
                new_weights[w_key] = max(base - 4, 1)
            elif win_rate < 50:
                new_weights[w_key] = max(base - 2, 1)
            else:
                new_weights[w_key] = base
        else:
            new_weights[w_key] = base

    learning["weights"] = new_weights
    learning["rule_stats"] = {
        "total_labeled": int(len(labeled)),
        "success_rate_5d": round(labeled["success_5d"].mean() * 100, 2),
        "by_condition": by_condition,
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
        "pool_total": int(pool_total),
        "success_count": int(success_count),
        "failed_count": int(failed_count),
        "display_total": int(len(df)),
        "A_count": int((df["grade"] == "A").sum()) if not df.empty else 0,
        "B_count": int((df["grade"] == "B").sum()) if not df.empty else 0,
        "C_count": int((df["grade"] == "C").sum()) if not df.empty else 0,
        "other_count": int((df["grade"] == "-").sum()) if not df.empty else 0,
    }


def run_scan(max_workers: int = 12) -> dict:
    ensure_data_dir()

    learning = load_learning()
    stocks = get_tw_stock_list()
    pool_total = len(stocks)

    rows = []
    fail_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one, s, learning): s for s in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                item, err = future.result()
                if item is not None:
                    rows.append(item)
                else:
                    fail_rows.append({
                        "code": stock["code"],
                        "name": stock["name"],
                        "market": stock["market"],
                        "symbol": tw_ticker(stock["code"], stock["market"]),
                        "reason": err or "unknown",
                    })
            except Exception as e:
                fail_rows.append({
                    "code": stock["code"],
                    "name": stock["name"],
                    "market": stock["market"],
                    "symbol": tw_ticker(stock["code"], stock["market"]),
                    "reason": str(e),
                })

    success_count = len(rows)
    failed_count = len(fail_rows)

    if not rows:
        payload = {
            "updated_at": now_str(),
            "status": "error",
            "message": "沒有掃描到任何成功結果。",
            "summary": summarize(pd.DataFrame(), pool_total, success_count, failed_count),
            "learning": learning,
            "results": [],
            "failed_symbols": fail_rows[:500],
        }
        save_latest(payload)
        return payload

    df = pd.DataFrame(rows)

    rank_map = {"A": 0, "B": 1, "C": 2, "-": 3}
    df["grade_rank"] = df["grade"].map(rank_map).fillna(9)
    df = df.sort_values(by=["grade_rank", "score", "vol_ratio", "close"], ascending=[True, False, False, False]).reset_index(drop=True)

    today_df = df.drop(columns=["grade_rank"]).copy()

    history_df = load_history_df()
    history_df = pd.concat([history_df, today_df], ignore_index=True)

    if "scan_id" in history_df.columns:
        history_df = history_df.drop_duplicates(subset=["scan_id"], keep="last")

    history_df = backfill_history_returns(history_df, learning)
    history_df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    new_learning = rebuild_learning(history_df, learning)
    save_learning(new_learning)

    payload = {
        "updated_at": now_str(),
        "status": "ok",
        "message": "",
        "summary": summarize(today_df, pool_total, success_count, failed_count),
        "learning": new_learning,
        "results": today_df.to_dict(orient="records"),
        "failed_symbols": fail_rows[:500],
    }
    save_latest(payload)
    return payload


if __name__ == "__main__":
    try:
        result = run_scan(max_workers=12)
        print(json.dumps({
            "updated_at": result.get("updated_at"),
            "status": result.get("status"),
            "summary": result.get("summary", {}),
            "learning_total_labeled": result.get("learning", {}).get("rule_stats", {}).get("total_labeled", 0),
        }, ensure_ascii=False, indent=2))
    except Exception as e:
        ensure_data_dir()
        payload = {
            "updated_at": now_str(),
            "status": "error",
            "message": f"掃描失敗: {e}",
            "traceback": traceback.format_exc(),
            "summary": {},
            "learning": DEFAULT_LEARNING.copy(),
            "results": [],
            "failed_symbols": [],
        }
        save_latest(payload)
        raise