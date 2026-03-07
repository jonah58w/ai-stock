from __future__ import annotations

import io
import math
import traceback
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# =============================
# 可選 ML 套件
# =============================
SKLEARN_OK = True
SKLEARN_ERR = ""
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except Exception as e:  # pragma: no cover
    SKLEARN_OK = False
    SKLEARN_ERR = str(e)

# =============================
# 基本設定
# =============================
st.set_page_config(page_title="AI 股票量化分析系統 V13 單檔版", page_icon="📈", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_START_DAYS = 720
DEFAULT_TOP_N = 10
REQUEST_TIMEOUT = 25
MAX_SCAN_WORKERS = 12
TW_TZ = ZoneInfo("Asia/Taipei")

# =============================
# 工具函式
# =============================
def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "").strip()
            if v in {"", "--", "----", "N/A", "nan", "None"}:
                return default
        return float(v)
    except Exception:
        return default


def format_num(v, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "-"


def format_pct(v, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return "-"


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s.replace(".TW", "").replace(".TWO", "")


def start_date_str(days: int = DEFAULT_START_DAYS) -> str:
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def finmind_headers(token: Optional[str]) -> Dict[str, str]:
    token = token or ""
    if token.strip():
        return {"Authorization": f"Bearer {token.strip()}"}
    return {}


def yahoo_symbol_candidates(stock_id: str, market_type: Optional[str] = None) -> List[str]:
    sid = normalize_symbol(stock_id)
    if market_type == "twse":
        return [f"{sid}.TW", f"{sid}.TWO"]
    if market_type == "tpex":
        return [f"{sid}.TWO", f"{sid}.TW"]
    return [f"{sid}.TW", f"{sid}.TWO"]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    keep = [c for c in wanted if c in df.columns]
    if len(keep) < 4:
        return pd.DataFrame()

    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])
    return df


def is_after_market_close_tw() -> bool:
    now = datetime.now(TW_TZ)
    return (now.hour, now.minute) >= (13, 30)


def today_tw_str() -> str:
    return datetime.now(TW_TZ).strftime("%Y-%m-%d")

# =============================
# 備援股票池
# =============================
def get_fallback_universe() -> pd.DataFrame:
    fallback = [
        ("2330", "台積電", "twse", "半導體"), ("2317", "鴻海", "twse", "電子代工"),
        ("2454", "聯發科", "twse", "IC 設計"), ("2308", "台達電", "twse", "電源供應"),
        ("2382", "廣達", "twse", "電子代工"), ("3711", "日月光投控", "twse", "封測"),
        ("3037", "欣興", "twse", "PCB"), ("8046", "南電", "tpex", "PCB"),
        ("3189", "景碩", "twse", "PCB"), ("2603", "長榮", "twse", "航運"),
        ("2609", "陽明", "twse", "航運"), ("2615", "萬海", "twse", "航運"),
        ("2881", "富邦金", "twse", "金融保險"), ("2882", "國泰金", "twse", "金融保險"),
        ("2891", "中信金", "twse", "金融保險"), ("2886", "兆豐金", "twse", "金融保險"),
        ("2412", "中華電", "twse", "電信"), ("2357", "華碩", "twse", "電腦週邊"),
        ("2379", "瑞昱", "twse", "IC 設計"), ("6669", "緯穎", "twse", "伺服器"),
        ("3008", "大立光", "twse", "光學"), ("4938", "和碩", "twse", "電子代工"),
        ("0050", "元大台灣50", "twse", "ETF"), ("0056", "元大高股息", "twse", "ETF"),
        ("00878", "國泰永續高股息", "twse", "ETF"), ("00919", "群益台灣精選高息", "twse", "ETF"),
        ("1301", "台塑", "twse", "塑化"), ("1303", "南亞", "twse", "塑化"),
        ("2002", "中鋼", "twse", "鋼鐵"), ("1216", "統一", "twse", "食品"),
        ("6488", "環球晶", "tpex", "半導體"),
    ]
    return pd.DataFrame(fallback, columns=["stock_id", "stock_name", "type", "industry_category"])

# =============================
# FinMind API
# =============================
@st.cache_data(ttl=3600, show_spinner=False)
def finmind_get(dataset: str, token: Optional[str] = None, **params) -> pd.DataFrame:
    query = {"dataset": dataset}
    query.update(params)
    try:
        resp = requests.get(FINMIND_URL, headers=finmind_headers(token), params=query, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# =============================
# 全台股清單：FinMind -> ISIN -> fallback
# =============================
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list_from_isin() -> pd.DataFrame:
    rows: List[dict] = []
    sources = [
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "twse"),
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "tpex"),
    ]
    for url, typ in sources:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT)
            resp.encoding = "big5"
            tables = pd.read_html(io.StringIO(resp.text), header=0)
            if not tables:
                continue
            df = tables[0].copy()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "raw"})
            df["raw"] = df["raw"].astype(str)
            parts = df["raw"].str.split("\u3000", n=1, expand=True)
            if parts.shape[1] < 2:
                parts = df["raw"].str.split(" ", n=1, expand=True)
            df["stock_id"] = parts[0].astype(str).str.strip()
            df["stock_name"] = parts[1].astype(str).str.strip() if parts.shape[1] > 1 else ""
            df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()
            df["type"] = typ
            df["industry_category"] = ""
            rows.extend(df[["stock_id", "stock_name", "type", "industry_category"]].to_dict("records"))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["stock_id", "stock_name", "type", "industry_category"])
    return pd.DataFrame(rows).drop_duplicates(subset=["stock_id"], keep="first").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_stock_info(token: Optional[str] = None) -> pd.DataFrame:
    expected = ["stock_id", "stock_name", "type", "industry_category"]
    df = finmind_get("TaiwanStockInfo", token=token)
    if not df.empty:
        df = df.copy()
        for c in expected:
            if c not in df.columns:
                df[c] = ""
        df["stock_id"] = df["stock_id"].astype(str)
        df = df[df["type"].isin(["twse", "tpex"])].copy()
        df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()
        df = df.drop_duplicates(subset=["stock_id"], keep="last")
        return df[expected].sort_values(["type", "stock_id"]).reset_index(drop=True)

    df = get_stock_list_from_isin()
    if not df.empty:
        return df[expected].sort_values(["type", "stock_id"]).reset_index(drop=True)

    return get_fallback_universe()[expected].copy()

# =============================
# 價格資料
# =============================
@st.cache_data(ttl=1800, show_spinner=False)
def load_finmind_price(stock_id: str, token: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    start_date = start_date or start_date_str(DEFAULT_START_DAYS)
    df = finmind_get("TaiwanStockPrice", token=token, data_id=normalize_symbol(stock_id), start_date=start_date)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume",
    })
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index()
    return _normalize_ohlcv(df)


@st.cache_data(ttl=1800, show_spinner=False)
def load_yahoo_price(stock_id: str, market_type: Optional[str] = None, period: str = "2y") -> Tuple[pd.DataFrame, str]:
    for ysym in yahoo_symbol_candidates(stock_id, market_type):
        try:
            df = yf.download(ysym, period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
            df = _normalize_ohlcv(df)
            if not df.empty:
                return df, ysym
        except Exception:
            pass
        try:
            df = yf.Ticker(ysym).history(period=period, interval="1d", auto_adjust=False)
            df = _normalize_ohlcv(df)
            if not df.empty:
                return df, ysym
        except Exception:
            pass
    return pd.DataFrame(), ""


@st.cache_data(ttl=1800, show_spinner=False)
def load_price(stock_id: str, market_type: Optional[str], token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    sid = normalize_symbol(stock_id)
    df = load_finmind_price(sid, token=token)
    if not df.empty:
        return df, "FinMind"
    ydf, ysym = load_yahoo_price(sid, market_type)
    if not ydf.empty:
        return ydf, f"Yahoo ({ysym})"
    return pd.DataFrame(), "無"

# =============================
# TWSE 公開價值表
# =============================
@st.cache_data(ttl=1800, show_spinner=False)
def get_twse_value_table() -> pd.DataFrame:
    base = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
    for n in range(0, 14):
        d = (date.today() - timedelta(days=n)).strftime("%Y%m%d")
        try:
            r = requests.get(base, params={"date": d, "selectType": "ALL", "response": "json"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT)
            js = r.json()
            data = js.get("data", [])
            fields = js.get("fields", [])
            if not data:
                continue
            df = pd.DataFrame(data, columns=fields if fields and len(fields) == len(data[0]) else None)
            colmap = {}
            for c in df.columns:
                sc = str(c)
                if "證券代號" in sc:
                    colmap[c] = "stock_id"
                elif "殖利率" in sc:
                    colmap[c] = "yield"
                elif "本益比" in sc:
                    colmap[c] = "pe"
                elif "股價淨值比" in sc:
                    colmap[c] = "pb"
            df = df.rename(columns=colmap)
            need = [c for c in ["stock_id", "yield", "pe", "pb"] if c in df.columns]
            if "stock_id" not in need:
                continue
            df = df[need].copy()
            df["stock_id"] = df["stock_id"].astype(str).str.strip()
            for c in ["yield", "pe", "pb"]:
                if c in df.columns:
                    df[c] = df[c].apply(safe_float)
            return df
        except Exception:
            continue
    return pd.DataFrame(columns=["stock_id", "yield", "pe", "pb"])

# =============================
# 價值分析
# =============================
@st.cache_data(ttl=1800, show_spinner=False)
def load_fundamental(stock_id: str, market_type: Optional[str] = None, token: Optional[str] = None) -> Dict[str, object]:
    sid = normalize_symbol(stock_id)
    out = {"pe": np.nan, "pb": np.nan, "eps": np.nan, "roe": np.nan, "dividend": np.nan, "yield": np.nan, "symbol_used": "", "source_note": ""}

    try:
        per_df = finmind_get("TaiwanStockPER", token=token, data_id=sid, start_date=(date.today() - timedelta(days=120)).strftime("%Y-%m-%d"))
        if not per_df.empty:
            if "date" in per_df.columns:
                per_df["date"] = pd.to_datetime(per_df["date"], errors="coerce")
                per_df = per_df.sort_values("date")
            row = per_df.iloc[-1]
            if "PER" in per_df.columns:
                out["pe"] = safe_float(row.get("PER"), np.nan)
            if "PBR" in per_df.columns:
                out["pb"] = safe_float(row.get("PBR"), np.nan)
            if "dividend_yield" in per_df.columns:
                out["yield"] = safe_float(row.get("dividend_yield"), np.nan)
            out["source_note"] += "FinMind PER; "
    except Exception:
        pass

    try:
        div_df = finmind_get("TaiwanStockDividend", token=token, data_id=sid, start_date=(date.today() - timedelta(days=1400)).strftime("%Y-%m-%d"))
        if not div_df.empty:
            if "date" in div_df.columns:
                div_df["date"] = pd.to_datetime(div_df["date"], errors="coerce")
                div_df = div_df.sort_values("date")
            if "CashEarningsDistribution" not in div_df.columns:
                div_df["CashEarningsDistribution"] = 0
            if "CashStatutorySurplus" not in div_df.columns:
                div_df["CashStatutorySurplus"] = 0
            div_df["現金股利"] = pd.to_numeric(div_df["CashEarningsDistribution"], errors="coerce").fillna(0) + pd.to_numeric(div_df["CashStatutorySurplus"], errors="coerce").fillna(0)
            out["dividend"] = safe_float(div_df.iloc[-1].get("現金股利"), np.nan)
            out["source_note"] += "FinMind Dividend; "
    except Exception:
        pass

    if market_type == "twse" and (pd.isna(out["pe"]) or pd.isna(out["pb"]) or pd.isna(out["yield"])):
        try:
            val_df = get_twse_value_table()
            hit = val_df[val_df["stock_id"] == sid]
            if not hit.empty:
                row = hit.iloc[0]
                if pd.isna(out["yield"]):
                    out["yield"] = safe_float(row.get("yield"), np.nan)
                if pd.isna(out["pe"]):
                    out["pe"] = safe_float(row.get("pe"), np.nan)
                if pd.isna(out["pb"]):
                    out["pb"] = safe_float(row.get("pb"), np.nan)
                out["source_note"] += "TWSE Value Table; "
        except Exception:
            pass

    for ysym in yahoo_symbol_candidates(stock_id, market_type):
        try:
            info = yf.Ticker(ysym).info or {}
            if pd.isna(out["pe"]):
                pe = safe_float(info.get("trailingPE"), np.nan)
                if pd.isna(pe):
                    pe = safe_float(info.get("forwardPE"), np.nan)
                out["pe"] = pe
            if pd.isna(out["pb"]):
                out["pb"] = safe_float(info.get("priceToBook"), np.nan)
            if pd.isna(out["eps"]):
                out["eps"] = safe_float(info.get("trailingEps"), np.nan)
            if pd.isna(out["roe"]):
                roe = info.get("returnOnEquity")
                if roe is not None:
                    out["roe"] = safe_float(roe, np.nan) * 100
            if pd.isna(out["dividend"]):
                out["dividend"] = safe_float(info.get("dividendRate"), np.nan)
            if pd.isna(out["yield"]):
                dy = info.get("dividendYield")
                if dy is not None:
                    out["yield"] = safe_float(dy, np.nan) * 100
            out["symbol_used"] = ysym
            out["source_note"] += f"Yahoo {ysym}; "
            break
        except Exception:
            continue

    return out

# =============================
# 技術指標
# =============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ohlcv(df)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    close = pd.Series(df["Close"], index=df.index, dtype="float64")
    high = pd.Series(df["High"], index=df.index, dtype="float64")
    low = pd.Series(df["Low"], index=df.index, dtype="float64")

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["RSI"] = RSIIndicator(close).rsi()

    stoch = StochasticOscillator(high, low, close)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    bb = BollingerBands(close)
    df["BBH"] = bb.bollinger_hband()
    df["BBL"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_pos"] = (df["Close"] - df["BBL"]) / (df["BBH"] - df["BBL"])

    atr = AverageTrueRange(high, low, close)
    df["ATR"] = atr.average_true_range()
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100

    df["SMA20"] = SMAIndicator(close, 20).sma_indicator()
    df["SMA50"] = SMAIndicator(close, 50).sma_indicator()
    df["SMA200"] = SMAIndicator(close, 200).sma_indicator()
    df["EMA12"] = EMAIndicator(close, 12).ema_indicator()
    df["EMA26"] = EMAIndicator(close, 26).ema_indicator()

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]
    df["RET_1D"] = df["Close"].pct_change(1)
    df["RET_5D"] = df["Close"].pct_change(5)
    df["RET_10D"] = df["Close"].pct_change(10)
    df["RET_20D"] = df["Close"].pct_change(20)
    df["RET_60D"] = df["Close"].pct_change(60)
    df["FLOW_PROXY"] = df["RET_1D"] * df["Volume"]
    df["FLOW20"] = df["FLOW_PROXY"].rolling(20).sum()
    df["PRICE_TO_SMA20"] = df["Close"] / df["SMA20"] - 1
    df["PRICE_TO_SMA50"] = df["Close"] / df["SMA50"] - 1
    df["FWD5_UP"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.03).astype(float)
    return df

# =============================
# 機器學習預測
# =============================
FEATURE_COLS = [
    "RSI", "MACD", "MACD_hist", "K", "D", "ATR_pct", "VOL_RATIO",
    "RET_5D", "RET_10D", "RET_20D", "PRICE_TO_SMA20", "PRICE_TO_SMA50", "BB_pos",
]


def heuristic_probability(df: pd.DataFrame, dy=np.nan, pe=np.nan, pb=np.nan, roe=np.nan) -> Tuple[float, str]:
    last = df.iloc[-1]
    s = 0.0
    s += 0.8 if safe_float(last.get("MACD_hist"), 0) > 0 else -0.5
    s += 0.6 if safe_float(last.get("K"), 50) > safe_float(last.get("D"), 50) else -0.4
    rsi = safe_float(last.get("RSI"), 50)
    s += 0.7 if 35 <= rsi <= 60 else 0.2 if rsi < 35 else -0.5
    s += 0.5 if safe_float(last.get("VOL_RATIO"), 1) > 1.2 else 0.0
    s += 0.5 if safe_float(last.get("RET_20D"), 0) > 0 else -0.2
    s += 0.3 if not pd.isna(dy) and dy >= 3 else 0.0
    s += 0.3 if not pd.isna(pe) and pe < 18 else 0.0
    prob = 1 / (1 + np.exp(-s))
    return float(prob), "heuristic"


def ml_predict_probability(df: pd.DataFrame) -> Tuple[float, str, Optional[float]]:
    data = df.copy().dropna(subset=FEATURE_COLS + ["FWD5_UP"])
    if len(data) < 120 or not SKLEARN_OK:
        p, src = heuristic_probability(df)
        return p, src, None

    X = data[FEATURE_COLS]
    y = data["FWD5_UP"].astype(int)
    if y.nunique() < 2:
        p, src = heuristic_probability(df)
        return p, src, None

    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred) if len(y_test) > 0 else None
    proba = model.predict_proba(data[FEATURE_COLS].iloc[[-1]])[0, 1]
    return float(proba), "random_forest", acc

# =============================
# 回測系統
# =============================
def backtest_score_strategy(df: pd.DataFrame, dy=np.nan, pe=np.nan, pb=np.nan, roe=np.nan, buy_threshold=70, sell_threshold=50) -> Dict[str, object]:
    bt = df.copy().dropna().copy()
    if len(bt) < 150:
        return {"trades": 0, "win_rate": np.nan, "cum_return": np.nan, "max_drawdown": np.nan, "equity_curve": pd.DataFrame()}

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    in_pos = False
    entry = 0.0
    wins = 0
    trades = 0
    curve_rows = []

    for i in range(210, len(bt) - 1):
        sub = bt.iloc[: i + 1]
        score = ai_score(sub, dy, pe, pb, roe)
        px = float(bt["Close"].iloc[i])
        next_px = float(bt["Close"].iloc[i + 1])
        dt_idx = bt.index[i + 1]

        if (not in_pos) and score >= buy_threshold:
            in_pos = True
            entry = px
            trades += 1
        elif in_pos and score <= sell_threshold:
            ret = px / entry - 1
            equity *= (1 + ret)
            if ret > 0:
                wins += 1
            in_pos = False

        mark = equity * (next_px / entry) if in_pos else equity
        peak = max(peak, mark)
        max_dd = min(max_dd, mark / peak - 1)
        curve_rows.append({"date": dt_idx, "equity": mark})

    if in_pos:
        ret = float(bt["Close"].iloc[-1]) / entry - 1
        equity *= (1 + ret)
        if ret > 0:
            wins += 1

    curve_df = pd.DataFrame(curve_rows)
    return {
        "trades": trades,
        "win_rate": wins / trades * 100 if trades > 0 else np.nan,
        "cum_return": (equity - 1) * 100,
        "max_drawdown": max_dd * 100,
        "equity_curve": curve_df,
    }

# =============================
# AI 自動選股與資金流
# =============================
def dividend_valuation(dividend: Optional[float], req_yield_pct: float) -> Optional[float]:
    if dividend is None or pd.isna(dividend) or dividend <= 0 or req_yield_pct <= 0:
        return None
    return dividend / (req_yield_pct / 100.0)


def eps_valuation(eps: Optional[float], fair_pe: float = 15.0) -> Optional[float]:
    if eps is None or pd.isna(eps) or eps <= 0:
        return None
    return eps * fair_pe


def value_score(dy: float, pe: float, pb: float, roe: float) -> float:
    score = 50.0
    if not pd.isna(dy):
        score += 18 if dy >= 7 else 12 if dy >= 5 else 6 if dy >= 3 else -6 if dy < 1 else 0
    if not pd.isna(pe):
        score += 12 if pe <= 10 else 6 if pe <= 15 else -12 if pe >= 30 else 0
    if not pd.isna(pb):
        score += 10 if pb <= 1.2 else 4 if pb <= 2.0 else -10 if pb >= 5 else 0
    if not pd.isna(roe):
        score += 10 if roe >= 15 else 5 if roe >= 10 else -5 if roe < 5 else 0
    return max(0, min(100, score))


def technical_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    price = safe_float(last["Close"], np.nan)
    score = 0.0
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
    if price > safe_float(last["SMA20"], price):
        score += 10
    if price > safe_float(last["SMA50"], price):
        score += 12
    if price > safe_float(last["SMA200"], price):
        score += 14
    if price <= safe_float(last["BBL"], price) * 1.02:
        score += 8
    elif price >= safe_float(last["BBH"], price) * 0.98:
        score += 4
    return max(0, min(100, score))


def ai_score(df: pd.DataFrame, dy: float, pe: float, pb: float, roe: float) -> float:
    ts = technical_score(df)
    vs = value_score(dy, pe, pb, roe)
    last = df.iloc[-1]
    ms = 50.0
    if safe_float(last.get("RET_5D"), 0) > 0:
        ms += 10
    if safe_float(last.get("RET_20D"), 0) > 0:
        ms += 10
    if safe_float(last.get("VOL_RATIO"), 1) >= 1.2:
        ms += 10
    return round(max(0, min(100, 0.40 * ts + 0.25 * vs + 0.15 * ms + 0.20 * 50)), 2)


def money_flow_strength(df: pd.DataFrame) -> float:
    if df.empty or "FLOW20" not in df.columns:
        return np.nan
    f = safe_float(df["FLOW20"].iloc[-1], np.nan)
    v = safe_float(df["Volume"].rolling(20).mean().iloc[-1], np.nan)
    p = safe_float(df["Close"].iloc[-1], np.nan)
    if pd.isna(f) or pd.isna(v) or pd.isna(p) or v == 0 or p == 0:
        return np.nan
    return f / (v * p)


def selector_bucket(score: float, prob: float, dy: float) -> str:
    if score >= 78 and prob >= 0.62:
        return "強勢成長"
    if dy >= 4 and score >= 65:
        return "價值收益"
    if score >= 68 and prob >= 0.55:
        return "平衡候選"
    return "觀察"


def selector_reason(df: pd.DataFrame, score: float, prob: float, dy: float, pe: float, flow: float) -> str:
    last = df.iloc[-1]
    reasons = []
    if safe_float(last.get("MACD_hist"), 0) > 0:
        reasons.append("MACD翻正")
    if safe_float(last.get("K"), 50) > safe_float(last.get("D"), 50):
        reasons.append("KD偏多")
    rsi = safe_float(last.get("RSI"), 50)
    if rsi < 35:
        reasons.append("RSI偏低反彈區")
    if not pd.isna(dy) and dy >= 4:
        reasons.append("高殖利率")
    if not pd.isna(pe) and pe <= 15:
        reasons.append("本益比偏低")
    if not pd.isna(flow) and flow > 0:
        reasons.append("資金流入")
    if prob >= 0.6:
        reasons.append("ML機率高")
    if score >= 75:
        reasons.append("綜合分數強")
    return " / ".join(reasons[:4]) if reasons else "技術與價值條件中性"

# =============================
# 買賣點 / 圖表
# =============================
def trade_point(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if df.empty:
        return None, None, None, None
    last = df.iloc[-1]
    price = safe_float(last["Close"], np.nan)
    if pd.isna(price):
        return None, None, None, None
    buy_candidates = [x for x in [safe_float(last.get("BBL")), safe_float(last.get("SMA20")), safe_float(last.get("SMA50"))] if not pd.isna(x)]
    sell_candidates = [x for x in [safe_float(last.get("BBH")), safe_float(df["Close"].rolling(60).max().iloc[-1])] if not pd.isna(x)]
    buy = min(buy_candidates) if buy_candidates else None
    sell = max(sell_candidates) if sell_candidates else None
    atr = safe_float(last.get("ATR"), np.nan)
    stop = price - atr * 2 if not pd.isna(atr) else None
    rr = None
    if stop is not None and sell is not None and stop < price:
        rr = max(sell - price, 0) / max(price - stop, 0.0001)
    return buy, sell, stop, rr


def chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="股價"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BBH"], name="布林上軌"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], name="布林下軌"))
    fig.update_layout(height=520, legend_orientation="h", margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
    return fig


def equity_curve_chart(curve_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if curve_df is not None and not curve_df.empty:
        fig.add_trace(go.Scatter(x=pd.to_datetime(curve_df["date"]), y=curve_df["equity"], name="策略淨值"))
    fig.update_layout(height=320, legend_orientation="h", margin=dict(l=20, r=20, t=20, b=20), title="回測資產曲線")
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig.update_layout(height=300, legend_orientation="h", margin=dict(l=20, r=20, t=20, b=20), title="MACD 指標")
    return fig


def kd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D"))
    fig.add_hline(y=80, line_dash="dash")
    fig.add_hline(y=20, line_dash="dash")
    fig.update_layout(height=300, legend_orientation="h", margin=dict(l=20, r=20, t=20, b=20), title="KD 指標")
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(height=280, legend_orientation="h", margin=dict(l=20, r=20, t=20, b=20), title="RSI 指標")
    return fig

# =============================
# 掃描器
# =============================
def scan_one_stock(stock_id: str, stock_name: str, market_type: Optional[str], token: Optional[str]) -> Optional[dict]:
    try:
        df_raw, _ = load_price(stock_id, market_type, token)
        if df_raw.empty:
            return None
        df = add_indicators(df_raw)
        if df.empty:
            return None
        fund = load_fundamental(stock_id, market_type, token)
        price = float(df.iloc[-1]["Close"])
        dy = fund["yield"]
        if pd.isna(dy) and not pd.isna(fund["dividend"]) and price > 0:
            dy = fund["dividend"] / price * 100
        pe, pb, eps, roe = fund["pe"], fund["pb"], fund["eps"], fund["roe"]
        if pd.isna(eps) and not pd.isna(pe) and pe > 0:
            eps = price / pe
        if pd.isna(roe) and not pd.isna(pe) and not pd.isna(pb) and pe > 0:
            roe = pb / pe * 100
        score = ai_score(df, dy, pe, pb, roe)
        prob, model_name, acc = ml_predict_probability(df)
        flow = money_flow_strength(df)
        bucket = selector_bucket(score, prob, dy if not pd.isna(dy) else 0)
        reason = selector_reason(df, score, prob, dy, pe, flow)
        return {
            "股票": stock_id,
            "名稱": stock_name,
            "市場": "上市" if market_type == "twse" else "上櫃",
            "股價": round(price, 2),
            "AI分數": score,
            "預測上漲機率": round(prob * 100, 2),
            "ML模型": model_name,
            "ML測試準確率": round(acc * 100, 2) if acc is not None else None,
            "資金流強度": round(flow, 4) if not pd.isna(flow) else None,
            "殖利率": round(dy, 2) if not pd.isna(dy) else None,
            "本益比": round(pe, 2) if not pd.isna(pe) else None,
            "股價淨值比": round(pb, 2) if not pd.isna(pb) else None,
            "ROE": round(roe, 2) if not pd.isna(roe) else None,
            "選股分類": bucket,
            "推薦理由": reason,
        }
    except Exception:
        return None


def scan_universe(universe: pd.DataFrame, token: Optional[str], top_n: int, progress_bar, status_box) -> pd.DataFrame:
    rows = []
    tasks = []
    total = len(universe)
    with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as ex:
        for _, row in universe.iterrows():
            tasks.append(ex.submit(scan_one_stock, row["stock_id"], row.get("stock_name", ""), row.get("type", None), token))

        done = 0
        for fut in as_completed(tasks):
            done += 1
            progress_bar.progress(done / max(total, 1))
            try:
                result = fut.result()
                if result is not None:
                    status_box.caption(f"掃描中：{done}/{total} {result['股票']} {result['名稱']}")
                    rows.append(result)
            except Exception:
                continue

    progress_bar.empty()
    status_box.empty()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["AI分數", "預測上漲機率", "資金流強度"], ascending=[False, False, False])
    return out.head(top_n).reset_index(drop=True)

# =============================
# 側欄
# =============================
st.title("📈 AI 股票量化分析系統 V13 單檔整合版")
st.caption("已加入：機器學習 AI 預測 / 回測系統 / AI 自動選股 / 收盤後自動 Top10 掃描")

with st.sidebar:
    st.header("⚙️ 系統設定")
    finmind_token = ""
    try:
        finmind_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        finmind_token = ""
    finmind_token = st.text_input("FinMind Token（可留空）", value=finmind_token, type="password")
    req_yield = st.slider("合理殖利率假設（%）", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
    top_n = st.slider("Top 掃描顯示前 N 名", 5, 30, DEFAULT_TOP_N)
    market_filter = st.selectbox("掃描範圍", ["全部", "上市", "上櫃"], index=0)
    auto_after_close = st.toggle("收盤後自動掃描 Top10", value=True)
    st.caption(f"目前掃描執行緒：{MAX_SCAN_WORKERS}")

    if not SKLEARN_OK:
        st.warning(f"sklearn 未安裝，ML 會退回 heuristic。{SKLEARN_ERR}")

mode = st.radio("系統模式", ["📊 單一股票分析", "🔎 Top10機會掃描", "🤖 AI 自動選股", "🧪 回測系統"], horizontal=True)

# =============================
# 收盤後自動掃描
# =============================
if "last_auto_scan_date" not in st.session_state:
    st.session_state.last_auto_scan_date = ""
if "last_auto_scan_df" not in st.session_state:
    st.session_state.last_auto_scan_df = pd.DataFrame()

if auto_after_close and is_after_market_close_tw() and st.session_state.last_auto_scan_date != today_tw_str():
    universe = get_tw_stock_info(finmind_token if finmind_token else None)
    if market_filter == "上市":
        universe = universe[universe["type"] == "twse"].copy()
    elif market_filter == "上櫃":
        universe = universe[universe["type"] == "tpex"].copy()
    p = st.progress(0.0)
    s = st.empty()
    st.session_state.last_auto_scan_df = scan_universe(universe, finmind_token if finmind_token else None, top_n, p, s)
    st.session_state.last_auto_scan_date = today_tw_str()

if not st.session_state.last_auto_scan_df.empty:
    with st.expander(f"📌 今日收盤後自動掃描 Top {len(st.session_state.last_auto_scan_df)}", expanded=False):
        topdf = st.session_state.last_auto_scan_df.copy()
        c1, c2, c3 = st.columns(3)
        if len(topdf) >= 1:
            c1.metric("#1", f"{topdf.iloc[0]['股票']} {topdf.iloc[0]['名稱']}", f"AI {topdf.iloc[0]['AI分數']}")
        if len(topdf) >= 2:
            c2.metric("#2", f"{topdf.iloc[1]['股票']} {topdf.iloc[1]['名稱']}", f"AI {topdf.iloc[1]['AI分數']}")
        if len(topdf) >= 3:
            c3.metric("#3", f"{topdf.iloc[2]['股票']} {topdf.iloc[2]['名稱']}", f"AI {topdf.iloc[2]['AI分數']}")
        st.dataframe(topdf, use_container_width=True, hide_index=True)

# =============================
# 單一股票分析
# =============================
if mode == "📊 單一股票分析":
    symbol = st.text_input("股票代碼", "2330")
    if st.button("開始分析"):
        try:
            stock_id = normalize_symbol(symbol)
            info_df = get_tw_stock_info(finmind_token if finmind_token else None)
            market_type = None
            hit = info_df[info_df["stock_id"] == stock_id]
            if not hit.empty:
                market_type = hit.iloc[0]["type"]

            df_raw, source = load_price(stock_id, market_type, finmind_token if finmind_token else None)
            if df_raw.empty:
                st.error("找不到股票資料")
            else:
                df = add_indicators(df_raw)
                if df.empty:
                    st.error("技術指標計算失敗，請稍後再試。")
                else:
                    fund = load_fundamental(stock_id, market_type, finmind_token if finmind_token else None)
                    price = float(df.iloc[-1]["Close"])
                    dy = fund["yield"]
                    if pd.isna(dy) and not pd.isna(fund["dividend"]) and price > 0:
                        dy = fund["dividend"] / price * 100
                    pe, pb, eps, roe = fund["pe"], fund["pb"], fund["eps"], fund["roe"]
                    if pd.isna(eps) and not pd.isna(pe) and pe > 0:
                        eps = price / pe
                    if pd.isna(roe) and not pd.isna(pe) and not pd.isna(pb) and pe > 0:
                        roe = pb / pe * 100

                    fair_div = dividend_valuation(fund["dividend"], req_yield)
                    fair_eps = eps_valuation(eps, 15)
                    score = ai_score(df, dy, pe, pb, roe)
                    buy, sell, stop, rr = trade_point(df)
                    prob, model_name, acc = ml_predict_probability(df)
                    bt = backtest_score_strategy(df, dy, pe, pb, roe)
                    flow = money_flow_strength(df)
                    bucket = selector_bucket(score, prob, dy if not pd.isna(dy) else 0)

                    st.markdown("## 股票決策總覽")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("股票", stock_id)
                    c2.metric("目前價格", format_num(price, 2))
                    c3.metric("AI綜合評分", format_num(score, 2))
                    c4.metric("AI建議", "買進" if score > 60 else "觀察")
                    c5.metric("AI選股分類", bucket)

                    st.markdown("## 價值分析")
                    v1, v2, v3, v4, v5 = st.columns(5)
                    v1.metric("殖利率", format_pct(dy, 2))
                    v2.metric("本益比", format_num(pe, 2))
                    v3.metric("股價淨值比", format_num(pb, 2))
                    v4.metric("EPS", format_num(eps, 2))
                    v5.metric("ROE", format_pct(roe, 2))

                    with st.container(border=True):
                        st.markdown("### 🤖 機器學習與資金流")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("預測上漲機率", format_pct(prob * 100, 2))
                        m2.metric("ML模型", model_name)
                        m3.metric("ML測試準確率", format_pct(acc * 100, 2) if acc is not None else "-")
                        m4.metric("資金流強度", format_num(flow, 4))

                    st.markdown("## 合理價估值")
                    a1, a2 = st.columns(2)
                    a1.metric("股利估值", format_num(fair_div, 2))
                    a2.metric("EPS估值", format_num(fair_eps, 2))

                    st.markdown("## 買賣點")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("預估買點", format_num(buy, 2))
                    b2.metric("預估賣點", format_num(sell, 2))
                    b3.metric("停損", format_num(stop, 2))
                    b4.metric("R/R", format_num(rr, 2))

                    st.markdown("## 回測結果（AI Score 策略）")
                    bt1, bt2, bt3, bt4 = st.columns(4)
                    bt1.metric("交易次數", format_num(bt["trades"], 0))
                    bt2.metric("勝率", format_pct(bt["win_rate"], 2))
                    bt3.metric("累積報酬", format_pct(bt["cum_return"], 2))
                    bt4.metric("最大回撤", format_pct(bt["max_drawdown"], 2))
                    st.plotly_chart(equity_curve_chart(bt.get("equity_curve", pd.DataFrame())), use_container_width=True)

                    st.markdown("## 趨勢圖與技術分析")
                    st.plotly_chart(chart(df.tail(220)), use_container_width=True)
                    lc, rc = st.columns(2)
                    with lc:
                        st.plotly_chart(macd_chart(df.tail(220)), use_container_width=True)
                    with rc:
                        st.plotly_chart(kd_chart(df.tail(220)), use_container_width=True)
                    st.plotly_chart(rsi_chart(df.tail(220)), use_container_width=True)

                    st.caption(f"價值分析來源：{fund['source_note'] if fund['source_note'] else '無'}｜股價來源：{source}")
        except Exception as e:
            st.error(f"執行失敗：{e}")
            st.code(traceback.format_exc())

# =============================
# Top10 掃描
# =============================
elif mode == "🔎 Top10機會掃描":
    st.markdown("## Top10 機會掃描（全市場）")
    if st.button("開始掃描全市場"):
        try:
            universe = get_tw_stock_info(finmind_token if finmind_token else None)
            if market_filter == "上市":
                universe = universe[universe["type"] == "twse"].copy()
            elif market_filter == "上櫃":
                universe = universe[universe["type"] == "tpex"].copy()
            st.info(f"本次掃描股票數：{len(universe)}")
            p = st.progress(0.0)
            s = st.empty()
            result = scan_universe(universe, finmind_token if finmind_token else None, top_n, p, s)
            if result.empty:
                st.warning("沒有掃描到可用結果，請稍後再試。")
            else:
                c1, c2, c3 = st.columns(3)
                if len(result) >= 1:
                    c1.metric("#1", f"{result.iloc[0]['股票']} {result.iloc[0]['名稱']}", f"AI {result.iloc[0]['AI分數']}")
                if len(result) >= 2:
                    c2.metric("#2", f"{result.iloc[1]['股票']} {result.iloc[1]['名稱']}", f"AI {result.iloc[1]['AI分數']}")
                if len(result) >= 3:
                    c3.metric("#3", f"{result.iloc[2]['股票']} {result.iloc[2]['名稱']}", f"AI {result.iloc[2]['AI分數']}")
                st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"掃描失敗：{e}")
            st.code(traceback.format_exc())

# =============================
# AI 自動選股
# =============================
elif mode == "🤖 AI 自動選股":
    st.markdown("## AI 自動選股")
    if st.button("產生 AI 選股清單"):
        try:
            universe = get_tw_stock_info(finmind_token if finmind_token else None)
            if market_filter == "上市":
                universe = universe[universe["type"] == "twse"].copy()
            elif market_filter == "上櫃":
                universe = universe[universe["type"] == "tpex"].copy()
            p = st.progress(0.0)
            s = st.empty()
            result = scan_universe(universe, finmind_token if finmind_token else None, max(top_n * 3, 20), p, s)
            if result.empty:
                st.warning("沒有產生可用候選。")
            else:
                strong = result[result["選股分類"] == "強勢成長"].head(top_n)
                value = result[result["選股分類"] == "價值收益"].head(top_n)
                balance = result[result["選股分類"] == "平衡候選"].head(top_n)
                t1, t2, t3 = st.tabs(["強勢成長", "價值收益", "平衡候選"])
                with t1:
                    st.dataframe(strong[[c for c in strong.columns if c in ['股票','名稱','市場','股價','AI分數','預測上漲機率','資金流強度','選股分類','推薦理由']]], use_container_width=True, hide_index=True)
                with t2:
                    st.dataframe(value[[c for c in value.columns if c in ['股票','名稱','市場','股價','AI分數','預測上漲機率','資金流強度','選股分類','推薦理由']]], use_container_width=True, hide_index=True)
                with t3:
                    st.dataframe(balance[[c for c in balance.columns if c in ['股票','名稱','市場','股價','AI分數','預測上漲機率','資金流強度','選股分類','推薦理由']]], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"選股失敗：{e}")
            st.code(traceback.format_exc())

# =============================
# 回測系統
# =============================
elif mode == "🧪 回測系統":
    st.markdown("## 回測系統")
    symbol = st.text_input("回測股票代碼", "2330")
    buy_threshold = st.slider("買進門檻（AI分數）", 50, 90, 70)
    sell_threshold = st.slider("賣出門檻（AI分數）", 30, 70, 50)
    if st.button("開始回測"):
        try:
            stock_id = normalize_symbol(symbol)
            info_df = get_tw_stock_info(finmind_token if finmind_token else None)
            market_type = None
            hit = info_df[info_df["stock_id"] == stock_id]
            if not hit.empty:
                market_type = hit.iloc[0]["type"]
            df_raw, source = load_price(stock_id, market_type, finmind_token if finmind_token else None)
            if df_raw.empty:
                st.error("找不到股價資料")
            else:
                df = add_indicators(df_raw)
                fund = load_fundamental(stock_id, market_type, finmind_token if finmind_token else None)
                price = float(df.iloc[-1]["Close"])
                dy = fund["yield"]
                if pd.isna(dy) and not pd.isna(fund["dividend"]) and price > 0:
                    dy = fund["dividend"] / price * 100
                pe, pb, roe = fund["pe"], fund["pb"], fund["roe"]
                if pd.isna(roe) and not pd.isna(pe) and not pd.isna(pb) and pe > 0:
                    roe = pb / pe * 100
                bt = backtest_score_strategy(df, dy, pe, pb, roe, buy_threshold, sell_threshold)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("交易次數", format_num(bt["trades"], 0))
                c2.metric("勝率", format_pct(bt["win_rate"], 2))
                c3.metric("累積報酬", format_pct(bt["cum_return"], 2))
                c4.metric("最大回撤", format_pct(bt["max_drawdown"], 2))
                st.caption(f"股價來源：{source}")
        except Exception as e:
            st.error(f"回測失敗：{e}")
            st.code(traceback.format_exc())
