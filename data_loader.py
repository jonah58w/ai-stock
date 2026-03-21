from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
REQUEST_TIMEOUT = 20
DEFAULT_START_DAYS = 720


def safe_float(v, default=float("nan")):
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


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".TW", "").replace(".TWO", "")


def finmind_headers(token: Optional[str]) -> Dict[str, str]:
    if token and token.strip():
        return {"Authorization": f"Bearer {token.strip()}"}
    return {}


def yahoo_symbol_candidates(stock_id: str, market_type: Optional[str] = None) -> List[str]:
    sid = normalize_symbol(stock_id)
    if market_type == "twse":
        return [f"{sid}.TW", f"{sid}.TWO"]
    if market_type == "tpex":
        return [f"{sid}.TWO", f"{sid}.TW"]
    return [f"{sid}.TW", f"{sid}.TWO"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(keep) < 4:
        return pd.DataFrame()

    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def finmind_get(dataset: str, token: Optional[str] = None, **params) -> pd.DataFrame:
    try:
        resp = requests.get(
            FINMIND_URL,
            headers=finmind_headers(token),
            params={"dataset": dataset, **params},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return pd.DataFrame()

        payload = resp.json()
        return pd.DataFrame(payload.get("data", []))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list_from_isin() -> pd.DataFrame:
    rows = []
    sources = [
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "twse"),
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "tpex"),
    ]

    for url, typ in sources:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.encoding = "big5"
            tables = pd.read_html(io.StringIO(resp.text), header=0)
            if not tables:
                continue

            df = tables[0].copy()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "raw"})

            parts = df["raw"].astype(str).str.split("\u3000", n=1, expand=True)
            if parts.shape[1] < 2:
                parts = df["raw"].astype(str).str.split(" ", n=1, expand=True)

            df["stock_id"] = parts[0].astype(str).str.strip()
            df["stock_name"] = parts[1].astype(str).str.strip() if parts.shape[1] > 1 else ""
            df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()
            df["type"] = typ
            df["industry_category"] = ""

            rows.extend(
                df[["stock_id", "stock_name", "type", "industry_category"]].to_dict("records")
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["stock_id", "stock_name", "type", "industry_category"])

    return pd.DataFrame(rows).drop_duplicates(subset=["stock_id"], keep="first")


@st.cache_data(ttl=3600, show_spinner=False)
def get_fallback_universe() -> pd.DataFrame:
    data = [
        ("2330", "台積電", "twse", "半導體"),
        ("2317", "鴻海", "twse", "電子代工"),
        ("2454", "聯發科", "twse", "IC設計"),
        ("2308", "台達電", "twse", "電源供應"),
        ("2382", "廣達", "twse", "電子代工"),
        ("3711", "日月光投控", "twse", "封測"),
        ("3037", "欣興", "twse", "PCB"),
        ("8046", "南電", "tpex", "PCB"),
        ("2603", "長榮", "twse", "航運"),
        ("2609", "陽明", "twse", "航運"),
        ("2881", "富邦金", "twse", "金融"),
        ("2891", "中信金", "twse", "金融"),
    ]
    return pd.DataFrame(data, columns=["stock_id", "stock_name", "type", "industry_category"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_tw_stock_info(token: Optional[str] = None) -> pd.DataFrame:
    expected = ["stock_id", "stock_name", "type", "industry_category"]

    df = finmind_get("TaiwanStockInfo", token=token)
    if not df.empty:
        for c in expected:
            if c not in df.columns:
                df[c] = ""

        df["stock_id"] = df["stock_id"].astype(str)
        df = df[df["type"].isin(["twse", "tpex"])]
        df = df[df["stock_id"].str.fullmatch(r"\d{4}")]

        return (
            df[expected]
            .drop_duplicates(subset=["stock_id"], keep="last")
            .sort_values(["type", "stock_id"])
            .reset_index(drop=True)
        )

    df = get_stock_list_from_isin()
    if not df.empty:
        return df[expected].sort_values(["type", "stock_id"]).reset_index(drop=True)

    return get_fallback_universe()


@st.cache_data(ttl=1800, show_spinner=False)
def load_finmind_price(stock_id: str, token: Optional[str] = None) -> pd.DataFrame:
    start_date = (date.today() - timedelta(days=DEFAULT_START_DAYS)).strftime("%Y-%m-%d")

    df = finmind_get(
        "TaiwanStockPrice",
        token=token,
        data_id=normalize_symbol(stock_id),
        start_date=start_date,
    )

    if df.empty:
        return pd.DataFrame()

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

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return normalize_ohlcv(df.set_index("Date").sort_index())


@st.cache_data(ttl=1800, show_spinner=False)
def load_yahoo_price(stock_id: str, market_type: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    for ysym in yahoo_symbol_candidates(stock_id, market_type):
        try:
            df = yf.download(
                ysym,
                period="2y",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            df = normalize_ohlcv(df)
            if not df.empty:
                return df, ysym
        except Exception:
            pass

        try:
            df = yf.Ticker(ysym).history(period="2y", interval="1d", auto_adjust=False)
            df = normalize_ohlcv(df)
            if not df.empty:
                return df, ysym
        except Exception:
            pass

    return pd.DataFrame(), ""


@st.cache_data(ttl=1800, show_spinner=False)
def load_price(stock_id: str, market_type: Optional[str], token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    df = load_finmind_price(stock_id, token)
    if not df.empty:
        return df, "FinMind"

    ydf, ysym = load_yahoo_price(stock_id, market_type)
    if not ydf.empty:
        return ydf, f"Yahoo ({ysym})"

    return pd.DataFrame(), "無"


@st.cache_data(ttl=1800, show_spinner=False)
def get_twse_value_table() -> pd.DataFrame:
    base = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"

    for n in range(0, 10):
        d = (date.today() - timedelta(days=n)).strftime("%Y%m%d")
        try:
            r = requests.get(
                base,
                params={"date": d, "selectType": "ALL", "response": "json"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=REQUEST_TIMEOUT,
            )
            js = r.json()
            data = js.get("data", [])
            fields = js.get("fields", [])

            if not data:
                continue

            df = pd.DataFrame(
                data,
                columns=fields if fields and len(fields) == len(data[0]) else None,
            )

            colmap = {}
            for c in df.columns:
                s = str(c)
                if "證券代號" in s:
                    colmap[c] = "stock_id"
                elif "殖利率" in s:
                    colmap[c] = "yield"
                elif "本益比" in s:
                    colmap[c] = "pe"
                elif "股價淨值比" in s:
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
def load_fundamental(stock_id: str, market_type: Optional[str], token: Optional[str]) -> Dict[str, object]:
    sid = normalize_symbol(stock_id)
    out = {
        "pe": float("nan"),
        "pb": float("nan"),
        "eps": float("nan"),
        "roe": float("nan"),
        "dividend": float("nan"),
        "yield": float("nan"),
        "source_note": "",
    }

    try:
        df = finmind_get(
            "TaiwanStockPER",
            token=token,
            data_id=sid,
            start_date=(date.today() - timedelta(days=120)).strftime("%Y-%m-%d"),
        )
        if not df.empty:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.sort_values("date")

            row = df.iloc[-1]
            if "PER" in df.columns:
                out["pe"] = safe_float(row.get("PER"))
            if "PBR" in df.columns:
                out["pb"] = safe_float(row.get("PBR"))
            if "dividend_yield" in df.columns:
                out["yield"] = safe_float(row.get("dividend_yield"))

            out["source_note"] += "FinMind PER; "
    except Exception:
        pass

    try:
        df = finmind_get(
            "TaiwanStockDividend",
            token=token,
            data_id=sid,
            start_date=(date.today() - timedelta(days=1400)).strftime("%Y-%m-%d"),
        )
        if not df.empty:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.sort_values("date")

            if "CashEarningsDistribution" not in df.columns:
                df["CashEarningsDistribution"] = 0
            if "CashStatutorySurplus" not in df.columns:
                df["CashStatutorySurplus"] = 0

            df["現金股利"] = (
                pd.to_numeric(df["CashEarningsDistribution"], errors="coerce").fillna(0)
                + pd.to_numeric(df["CashStatutorySurplus"], errors="coerce").fillna(0)
            )

            out["dividend"] = safe_float(df.iloc[-1].get("現金股利"))
            out["source_note"] += "FinMind Dividend; "
    except Exception:
        pass

    if market_type == "twse":
        try:
            val = get_twse_value_table()
            hit = val[val["stock_id"] == sid]
            if not hit.empty:
                row = hit.iloc[0]
                if pd.isna(out["yield"]):
                    out["yield"] = safe_float(row.get("yield"))
                if pd.isna(out["pe"]):
                    out["pe"] = safe_float(row.get("pe"))
                if pd.isna(out["pb"]):
                    out["pb"] = safe_float(row.get("pb"))
                out["source_note"] += "TWSE Value Table; "
        except Exception:
            pass

    for ysym in yahoo_symbol_candidates(stock_id, market_type):
        try:
            info = yf.Ticker(ysym).info or {}

            if pd.isna(out["pe"]):
                out["pe"] = safe_float(info.get("trailingPE"))
            if pd.isna(out["pb"]):
                out["pb"] = safe_float(info.get("priceToBook"))
            if pd.isna(out["eps"]):
                out["eps"] = safe_float(info.get("trailingEps"))

            if pd.isna(out["roe"]):
                roe = info.get("returnOnEquity")
                if roe is not None:
                    out["roe"] = safe_float(roe) * 100

            if pd.isna(out["dividend"]):
                out["dividend"] = safe_float(info.get("dividendRate"))

            if pd.isna(out["yield"]):
                dy = info.get("dividendYield")
                if dy is not None:
                    out["yield"] = safe_float(dy) * 100

            out["source_note"] += f"Yahoo {ysym}; "
            break
        except Exception:
            continue

    return out
