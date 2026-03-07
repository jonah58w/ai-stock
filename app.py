from __future__ import annotations

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
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

st.set_page_config(page_title="AI 股票量化分析系統 V12 PRO", page_icon="📈", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_START_DAYS = 520
DEFAULT_TOP_N = 10
REQUEST_TIMEOUT = 25
UA = {"User-Agent": "Mozilla/5.0"}


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


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list_from_isin() -> pd.DataFrame:
    rows: List[dict] = []
    sources = [
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "twse"),
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "tpex"),
    ]
    for url, typ in sources:
        try:
            tables = pd.read_html(url, header=0)
            if not tables:
                continue
            df = tables[0].copy()
            df.columns = [str(c).strip() for c in df.columns]
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
    out = pd.DataFrame(rows).drop_duplicates(subset=["stock_id"], keep="first").reset_index(drop=True)
    return out


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

    isin_df = get_stock_list_from_isin()
    if not isin_df.empty:
        return isin_df[expected].sort_values(["type", "stock_id"]).reset_index(drop=True)

    return get_fallback_universe()[expected].copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_finmind_price(stock_id: str, token: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    start_date = start_date or start_date_str(DEFAULT_START_DAYS)
    df = finmind_get("TaiwanStockPrice", token=token, data_id=normalize_symbol(stock_id), start_date=start_date)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
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
            tk = yf.Ticker(ysym)
            df = tk.history(period=period, interval="1d", auto_adjust=False)
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
    ydf, ysym = load_yahoo_price(sid, market_type=market_type)
    if not ydf.empty:
        return ydf, f"Yahoo ({ysym})"
    raw = str(stock_id).strip().upper()
    if raw.endswith(".TW") or raw.endswith(".TWO"):
        try:
            ydf = yf.Ticker(raw).history(period="2y", interval="1d", auto_adjust=False)
            ydf = _normalize_ohlcv(ydf)
            if not ydf.empty:
                return ydf, f"Yahoo ({raw})"
        except Exception:
            pass
    return pd.DataFrame(), "無"


@st.cache_data(ttl=1800, show_spinner=False)
def get_twse_value_table() -> pd.DataFrame:
    # listed stocks public PE/PB/yield
    base = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
    for n in range(0, 14):
        d = (date.today() - timedelta(days=n)).strftime("%Y%m%d")
        try:
            r = requests.get(base, params={"date": d, "selectType": "ALL", "response": "json"}, headers=UA, timeout=REQUEST_TIMEOUT)
            js = r.json()
            data = js.get("data", [])
            fields = js.get("fields", [])
            if data:
                df = pd.DataFrame(data, columns=fields if fields and len(fields) == len(data[0]) else None)
                if df.empty:
                    continue
                # Normalize Chinese field names
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


@st.cache_data(ttl=1800, show_spinner=False)
def load_fundamental(stock_id: str, market_type: Optional[str] = None, token: Optional[str] = None) -> Dict[str, object]:
    sid = normalize_symbol(stock_id)
    out = {"pe": np.nan, "pb": np.nan, "eps": np.nan, "roe": np.nan, "dividend": np.nan, "yield": np.nan, "symbol_used": "", "source_note": ""}

    # FinMind single stock
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

    # Public TWSE value table for listed stocks without token
    if market_type == "twse" and (pd.isna(out["pe"]) or pd.isna(out["pb"]) or pd.isna(out["yield"])):
        try:
            twse_val = get_twse_value_table()
            hit = twse_val[twse_val["stock_id"] == sid]
            if not hit.empty:
                row = hit.iloc[0]
                if pd.isna(out["yield"]) and "yield" in hit.columns:
                    out["yield"] = safe_float(row.get("yield"), np.nan)
                if pd.isna(out["pe"]) and "pe" in hit.columns:
                    out["pe"] = safe_float(row.get("pe"), np.nan)
                if pd.isna(out["pb"]) and "pb" in hit.columns:
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ohlcv(df)
    if df.empty or len(df) < 35:
        return pd.DataFrame()
    df = df.copy()
    close = pd.Series(df["Close"], index=df.index, dtype="float64")
    high = pd.Series(df["High"], index=df.index, dtype="float64")
    low = pd.Series(df["Low"], index=df.index, dtype="float64")

    macd = MACD(close)
    df["MACD"] = pd.Series(macd.macd(), index=df.index)
    df["MACD_signal"] = pd.Series(macd.macd_signal(), index=df.index)
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["RSI"] = pd.Series(RSIIndicator(close).rsi(), index=df.index)
    stoch = StochasticOscillator(high, low, close)
    df["K"] = pd.Series(stoch.stoch(), index=df.index)
    df["D"] = pd.Series(stoch.stoch_signal(), index=df.index)
    bb = BollingerBands(close)
    df["BBH"] = pd.Series(bb.bollinger_hband(), index=df.index)
    df["BBL"] = pd.Series(bb.bollinger_lband(), index=df.index)
    atr = AverageTrueRange(high, low, close)
    df["ATR"] = pd.Series(atr.average_true_range(), index=df.index)
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100
    df["SMA20"] = pd.Series(SMAIndicator(close, 20).sma_indicator(), index=df.index)
    df["SMA50"] = pd.Series(SMAIndicator(close, 50).sma_indicator(), index=df.index)
    df["SMA200"] = pd.Series(SMAIndicator(close, 200).sma_indicator(), index=df.index)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]
    df["RET_5D"] = df["Close"].pct_change(5)
    df["RET_20D"] = df["Close"].pct_change(20)
    return df


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
    return round(max(0, min(100, 0.45 * ts + 0.35 * vs + 0.20 * ms)), 2)


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


st.title("📈 AI 股票量化分析系統 V12 PRO")
st.caption("FinMind + Yahoo Finance ｜ 技術分析 + 價值分析 + AI決策引擎")

with st.sidebar:
    st.header("⚙️ 系統設定")
    finmind_token = ""
    try:
        finmind_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        finmind_token = ""
    finmind_token = st.text_input("FinMind Token（可留空）", value=finmind_token, type="password")
    req_yield = st.slider("合理殖利率假設（%）", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
    top_n = st.slider("掃描顯示前 N 名", 5, 50, DEFAULT_TOP_N)
    st.markdown("---")
    st.caption("Top10 會優先掃描 FinMind 全台股清單；抓不到時會改用公開名單；再不行才用備援股票池。")

mode = st.radio("系統模式", ["📊 單一股票分析", "🔎 Top10機會掃描"], horizontal=True)

if mode == "📊 單一股票分析":
    symbol = st.text_input("股票代碼", "2330")
    if st.button("開始分析"):
        try:
            stock_id = normalize_symbol(symbol)
            market_type = None
            info_df = get_tw_stock_info(finmind_token if finmind_token else None)
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
                    fair_div = dividend_valuation(fund["dividend"], req_yield)
                    fair_eps = eps_valuation(eps, 15)
                    score = ai_score(df, dy, pe, pb, roe)
                    buy, sell, stop, rr = trade_point(df)

                    st.markdown("## 股票決策總覽")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("股票", stock_id)
                    c2.metric("目前價格", format_num(price, 2))
                    c3.metric("AI綜合評分", format_num(score, 2))
                    c4.metric("AI建議", "買進" if score > 60 else "觀察")

                    st.markdown("## 價值分析")
                    v1, v2, v3, v4, v5 = st.columns(5)
                    v1.metric("殖利率", format_pct(dy, 2))
                    v2.metric("本益比", format_num(pe, 2))
                    v3.metric("股價淨值比", format_num(pb, 2))
                    v4.metric("EPS", format_num(eps, 2))
                    v5.metric("ROE", format_pct(roe, 2))

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

                    st.markdown("## 趨勢圖與技術分析")
                    st.plotly_chart(chart(df.tail(220)), use_container_width=True)
                    left_col, right_col = st.columns(2)
                    with left_col:
                        st.plotly_chart(macd_chart(df.tail(220)), use_container_width=True)
                    with right_col:
                        st.plotly_chart(kd_chart(df.tail(220)), use_container_width=True)
                    st.plotly_chart(rsi_chart(df.tail(220)), use_container_width=True)

                    st.caption(f"價值分析來源：{fund['source_note'] if fund['source_note'] else '無'}｜股價來源：{source}")
                    if pd.isna(dy) and pd.isna(pe) and pd.isna(pb) and pd.isna(eps) and pd.isna(roe):
                        st.warning("此股票目前無法抓到可用價值分析欄位。若是上市股，請確認網路可連到 TWSE；若有 FinMind Token 也請填入。")
        except Exception as e:
            st.error(f"執行失敗：{e}")
            st.code(traceback.format_exc())

elif mode == "🔎 Top10機會掃描":
    st.markdown("## Top10 機會掃描（全台股）")
    market_filter = st.selectbox("掃描範圍", ["全部", "上市", "上櫃"], index=0)
    start_scan = st.button("開始掃描全台股")
    if start_scan:
        try:
            universe = get_tw_stock_info(finmind_token if finmind_token else None)
            if market_filter == "上市":
                universe = universe[universe["type"] == "twse"].copy()
            elif market_filter == "上櫃":
                universe = universe[universe["type"] == "tpex"].copy()

            st.info(f"本次掃描股票數：{len(universe)}")
            rows = []
            progress = st.progress(0.0)
            status = st.empty()
            total = len(universe)
            for i, row in universe.iterrows():
                stock_id, stock_name, market_type = row["stock_id"], row.get("stock_name", ""), row.get("type", None)
                status.caption(f"掃描中：{i+1}/{total} {stock_id} {stock_name}")
                progress.progress((i + 1) / max(total, 1))
                try:
                    df_raw, _ = load_price(stock_id, market_type, finmind_token if finmind_token else None)
                    if df_raw.empty:
                        continue
                    df = add_indicators(df_raw)
                    if df.empty:
                        continue
                    fund = load_fundamental(stock_id, market_type, finmind_token if finmind_token else None)
                    price = float(df.iloc[-1]["Close"])
                    dy = fund["yield"]
                    if pd.isna(dy) and not pd.isna(fund["dividend"]) and price > 0:
                        dy = fund["dividend"] / price * 100
                    pe, pb, roe = fund["pe"], fund["pb"], fund["roe"]
                    score = ai_score(df, dy, pe, pb, roe)
                    rows.append({"股票": stock_id, "名稱": stock_name, "市場": "上市" if market_type == "twse" else "上櫃", "股價": round(price, 2), "AI分數": score, "殖利率": round(dy, 2) if not pd.isna(dy) else None, "本益比": round(pe, 2) if not pd.isna(pe) else None, "股價淨值比": round(pb, 2) if not pd.isna(pb) else None, "ROE": round(roe, 2) if not pd.isna(roe) else None})
                except Exception:
                    continue
            progress.empty()
            status.empty()
            if len(rows) == 0:
                st.warning("沒有掃描到可用結果，請稍後再試。")
            else:
                scan_df = pd.DataFrame(rows).sort_values("AI分數", ascending=False).head(top_n)
                st.dataframe(scan_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"掃描失敗：{e}")
            st.code(traceback.format_exc())
