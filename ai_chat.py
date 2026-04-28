
"""
AI Chat 模組 — 為每檔股票提供即時對話分析（透過 OpenRouter）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# 模型清單
# ---------------------------------------------------------------------------
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "DeepSeek V3 (推薦・繁中強・便宜)": {
        "id": "deepseek/deepseek-chat",
        "in_price": 0.27, "out_price": 1.10,
    },
    "Qwen 2.5 72B (台股術語熟)": {
        "id": "qwen/qwen-2.5-72b-instruct",
        "in_price": 0.35, "out_price": 0.40,
    },
    "Hermes 4 70B (Nous Research)": {
        "id": "nousresearch/hermes-4-70b",
        "in_price": 0.13, "out_price": 0.40,
    },
    "Hermes 3 405B (免費・有速率限制)": {
        "id": "nousresearch/hermes-3-llama-3.1-405b:free",
        "in_price": 0.0, "out_price": 0.0,
    },
    "Claude Haiku 4.5 (繁中極佳)": {
        "id": "anthropic/claude-haiku-4.5",
        "in_price": 1.0, "out_price": 5.0,
    },
    "GPT-4o mini (OpenAI)": {
        "id": "openai/gpt-4o-mini",
        "in_price": 0.15, "out_price": 0.60,
    },
    "Gemini 2.5 Flash (Google)": {
        "id": "google/gemini-2.5-flash",
        "in_price": 0.30, "out_price": 2.50,
    },
}
DEFAULT_MODEL_LABEL = "DeepSeek V3 (推薦・繁中強・便宜)"


# ---------------------------------------------------------------------------
# 防禦式輔助函式
# ---------------------------------------------------------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# 上下文打包
# ---------------------------------------------------------------------------
def _summarize_recent_ohlcv(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "(無價格資料)"
    o = _col(df, "open"); h = _col(df, "high"); l = _col(df, "low")
    c = _col(df, "close"); v = _col(df, "volume")
    tail = df.tail(n)
    lines = []
    for idx, row in tail.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        try:
            lines.append(
                f"{date_str}: "
                f"O={_safe_float(row[o]):.2f} H={_safe_float(row[h]):.2f} "
                f"L={_safe_float(row[l]):.2f} C={_safe_float(row[c]):.2f} "
                f"V={_safe_float(row[v]):,.0f}"
            )
        except Exception:
            continue
    return "\n".join(lines) if lines else "(無有效資料)"


def _summarize_indicators(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(無指標資料)"
    last = df.iloc[-1]
    parts: List[str] = []

    for p in [5, 10, 20, 60, 120, 240]:
        col = _col(df, f"MA{p}", f"SMA{p}", f"ma_{p}")
        if col:
            v = _safe_float(last[col])
            if not pd.isna(v):
                parts.append(f"MA{p}={v:.2f}")

    for label, cands in [
        ("RSI", ["RSI", "RSI14", "rsi_14"]),
        ("MACD", ["MACD"]),
        ("MACD_signal", ["MACD_signal", "signal"]),
        ("MACD_hist", ["MACD_hist", "hist", "histogram"]),
        ("K", ["K", "kd_k", "%K"]),
        ("D", ["D", "kd_d", "%D"]),
        ("BB上軌", ["BB_upper", "boll_upper", "upper"]),
        ("BB下軌", ["BB_lower", "boll_lower", "lower"]),
    ]:
        col = _col(df, *cands)
        if col:
            v = _safe_float(last[col])
            if not pd.isna(v):
                parts.append(f"{label}={v:.2f}")

    return ", ".join(parts) if parts else "(無有效指標)"


def _summarize_signals(signals_info: Optional[Dict[str, Any]]) -> str:
    if not signals_info:
        return "(無訊號資料)"
    parts = []
    for k, v in signals_info.items():
        if isinstance(v, bool):
            parts.append(f"{k}: {'✅ 觸發中' if v else '❌ 未觸發'}")
        elif isinstance(v, dict):
            t = v.get("triggered", v.get("active", False))
            parts.append(f"{k}: {'✅' if t else '❌'}")
        elif v is not None:
            parts.append(f"{k}: {v}")
    return "; ".join(parts) if parts else "(無訊號)"


def _summarize_fundamentals(fund: Optional[Dict[str, Any]]) -> str:
    if not fund:
        return "(無基本面資料)"
    label_map = {
        "eps": "EPS", "pe": "本益比", "pb": "股價淨值比", "roe": "ROE",
        "revenue_yoy": "營收年增率", "revenue": "月營收",
        "yield": "現金殖利率", "dividend_yield": "現金殖利率",
        "market_cap": "市值", "industry": "產業",
    }
    parts = []
    for k, v in fund.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        label = label_map.get(k.lower(), k)
        if isinstance(v, float):
            if any(s in k.lower() for s in ["yoy", "yield", "roe"]):
                parts.append(f"{label}: {v:.2f}%")
            else:
                parts.append(f"{label}: {v:.2f}")
        else:
            parts.append(f"{label}: {v}")
    return "; ".join(parts) if parts else "(無基本面)"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_recent_news(symbol: str, days: int = 7) -> str:
    if requests is None:
        return "(requests 套件不可用)"
    try:
        token = st.secrets.get("FINMIND_TOKEN", "")
        if not token:
            return "(未設定 FinMind Token)"
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={
                "dataset": "TaiwanStockNews",
                "data_id": symbol,
                "start_date": start, "end_date": end,
                "token": token,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return f"(FinMind {r.status_code})"
        data = r.json().get("data", []) or []
        if not data:
            return "(近期無新聞)"
        items = sorted(data, key=lambda x: x.get("date", ""), reverse=True)[:5]
        return "\n".join(
            f"[{i.get('date','')}] {i.get('title','')} ({i.get('source','')})"
            for i in items
        )
    except Exception as e:
        return f"(新聞擷取失敗:{type(e).__name__})"


def _build_context_block(
    symbol: str, name: str, df: pd.DataFrame,
    signals_info: Optional[Dict[str, Any]],
    fundamentals: Optional[Dict[str, Any]],
    market_regime: Optional[str],
    include_news: bool,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    news_block = _fetch_recent_news(symbol) if include_news else "(本次未擷取)"
    return f"""
=== 個股資料快照 ({today}) ===

【股票】{symbol} {name}
【市場環境(^TWII)】{market_regime or '未提供'}

【最近 20 日 K 線】
{_summarize_recent_ohlcv(df, n=20)}

【最新技術指標】
{_summarize_indicators(df)}

【系統訊號 (A1/A2 等)】
{_summarize_signals(signals_info)}

【基本面】
{_summarize_fundamentals(fundamentals)}

【近 7 日新聞】
{news_block}
""".strip()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """你是一位資深台股技術分析師,專精台灣上市櫃股票的多空研判。
你的對話風格務實、簡潔、有觀點,不講廢話、不打官腔。

回答原則:
1. 一律使用**繁體中文**回答(台灣用語:壓力位、支撐位、主力、籌碼、量價、跳空、突破、回測 等)
2. **嚴格依據下方提供的真實資料**做分析,不可虛構數字、不可虛構新聞
3. 資料不足以下定論時,明說「資料不足」,不要硬掰
4. 講多空時,給出**關鍵價位**(如「站上 XX 元才確認突破」、「跌破 YY 元轉空」)
5. 不用建議式語言(「我建議買進」),改用分析式(「技術面偏多,但需注意 XX」)
6. 風險提醒一次就好,不要每次都加免責聲明
7. 預設回覆 150~400 字;使用者要求展開時再詳述

【可用資料】
{context_block}

以上資料為事實依據,請以此為準。
""".strip()


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------
def _get_client():
    if OpenAI is None:
        st.error("❌ 缺少 `openai` 套件。請在 requireme
