from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from charts import bollinger_trend_chart
from data_loader import get_tw_stock_info, load_fundamental, load_price, normalize_symbol
from indicators import add_indicators
from scanner import analyze_stock


# =========================================================
# 基本工具
# =========================================================
def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        if pd.isna(v):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)


def _fmt_pct(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        if pd.isna(v):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return str(v)


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _get_market_type(info_df: pd.DataFrame, stock_id: str) -> Optional[str]:
    if info_df is None or info_df.empty or "stock_id" not in info_df.columns:
        return None

    hit = info_df[info_df["stock_id"].astype(str) == str(stock_id)]
    if hit.empty:
        return None

    if "type" in hit.columns:
        return hit.iloc[0]["type"]
    return None


def _get_stock_name(info_df: pd.DataFrame, stock_id: str) -> str:
    if info_df is None or info_df.empty or "stock_id" not in info_df.columns:
        return stock_id

    hit = info_df[info_df["stock_id"].astype(str) == str(stock_id)]
    if hit.empty:
        return stock_id

    for c in ["stock_name", "name", "stock_name_zh", "company_name"]:
        if c in hit.columns:
            v = str(hit.iloc[0][c]).strip()
            if v:
                return v
    return stock_id


def _market_label(market_type: Optional[str]) -> str:
    mapping = {
        "twse": "上市",
        "tpex": "上櫃",
        "rotc": "興櫃",
    }
    if market_type is None:
        return "-"
    return mapping.get(str(market_type), str(market_type))


# =========================================================
# 布林判讀
# =========================================================
def _prepare_bb_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # 欄位名標準化
    rename_map = {}
    for c in work.columns:
        cl = str(c).strip().lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        work = work.rename(columns=rename_map)

    if "Close" not in work.columns:
        return work

    close = work["Close"]

    if "bb_middle" not in work.columns:
        work["bb_middle"] = close.rolling(20, min_periods=20).mean()

    if "bb_upper" not in work.columns or "bb_lower" not in work.columns:
        std = close.rolling(20, min_periods=20).std()
        work["bb_upper"] = work["bb_middle"] + 2 * std
        work["bb_lower"] = work["bb_middle"] - 2 * std

    work["bb_width"] = (work["bb_upper"] - work["bb_lower"]) / work["bb_middle"].replace(0, np.nan)
    work["bb_percent_b"] = (close - work["bb_lower"]) / (work["bb_upper"] - work["bb_lower"]).replace(0, np.nan)
    work["bb_mid_slope"] = work["bb_middle"].diff()

    work["bb_width_q15"] = work["bb_width"].rolling(120, min_periods=30).quantile(0.15)
    work["bb_squeeze"] = work["bb_width"] <= work["bb_width_q15"]

    work["near_upper"] = (
        (work["bb_upper"] > 0)
        & (((work["bb_upper"] - work["Close"]) / work["bb_upper"]) <= 0.03)
    )
    work["near_lower"] = (
        (work["bb_lower"] > 0)
        & (((work["Close"] - work["bb_lower"]) / work["bb_lower"]) <= 0.03)
    )

    work["ride_upper_count_5"] = work["near_upper"].rolling(5, min_periods=1).sum()
    work["ride_lower_count_5"] = work["near_lower"].rolling(5, min_periods=1).sum()

    work["ride_upper"] = (
        (work["ride_upper_count_5"] >= 3)
        & (work["bb_mid_slope"] > 0)
        & (work["Close"] >= work["bb_middle"])
    )

    work["ride_lower"] = (
        (work["ride_lower_count_5"] >= 3)
        & (work["bb_mid_slope"] < 0)
        & (work["Close"] <= work["bb_middle"])
    )

    work["bb_overheat"] = (
        (work["bb_percent_b"] >= 0.95)
        & (work["bb_width"] > work["bb_width"].rolling(20, min_periods=10).mean())
        & (work["Close"] > work["bb_middle"])
    )

    work["bb_weakening"] = (
        (work["Close"] < work["bb_middle"])
        & (work["bb_mid_slope"] < 0)
        & (work["bb_percent_b"] < 0.5)
    )

    def _state(row) -> str:
        if bool(row.get("bb_squeeze", False)):
            return "收縮整理"
        if bool(row.get("ride_upper", False)):
            return "沿上軌強攻"
        if bool(row.get("bb_overheat", False)):
            return "高檔過熱"
        if bool(row.get("bb_weakening", False)):
            return "中軌失守轉弱"
        if bool(row.get("ride_lower", False)):
            return "沿下軌走弱"
        if row.get("bb_mid_slope", 0) > 0 and row.get("Close", np.nan) >= row.get("bb_middle", np.nan):
            return "中軌上彎轉強"
        if row.get("bb_mid_slope", 0) < 0 and row.get("Close", np.nan) < row.get("bb_middle", np.nan):
            return "中軌下彎偏弱"
        return "盤整觀察"

    work["bb_state"] = work.apply(_state, axis=1)

    return work


def _build_bb_summary(df: pd.DataFrame) -> Dict[str, Any]:
    work = _prepare_bb_metrics(df)
    if work is None or work.empty:
        return {
            "bb_state": "-",
            "bb_width": None,
            "bb_percent_b": None,
            "bb_mid_slope": None,
            "bb_comment": "無足夠資料",
        }

    last = work.iloc[-1]

    state = str(last.get("bb_state", "-"))
    bbw = _safe_float(last.get("bb_width"))
    pb = _safe_float(last.get("bb_percent_b"))
    slope = _safe_float(last.get("bb_mid_slope"))

    comments = []
    if state == "收縮整理":
        comments.append("波動收斂，留意後續方向性突破")
    elif state == "沿上軌強攻":
        comments.append("價格沿上軌推進，屬強勢趨勢延續")
    elif state == "高檔過熱":
        comments.append("價格過度靠近上軌，短線不宜追價")
    elif state == "中軌失守轉弱":
        comments.append("跌回中軌下方，趨勢有轉弱跡象")
    elif state == "沿下軌走弱":
        comments.append("價格沿下軌下行，偏空格局明顯")
    elif state == "中軌上彎轉強":
        comments.append("中軌上彎且價格位於中軌上方，結構偏多")
    elif state == "中軌下彎偏弱":
        comments.append("中軌下彎且價格位於中軌下方，結構偏弱")
    else:
        comments.append("布林通道呈整理狀態，宜等待更清楚方向")

    if pd.notna(pb):
        if pb > 1:
            comments.append("%B 大於 1，價格已穿越上軌")
        elif pb >= 0.8:
            comments.append("%B 高於 0.8，價格位於通道上緣")
        elif pb <= 0:
            comments.append("%B 小於 0，價格已跌破下軌")
        elif pb <= 0.2:
            comments.append("%B 低於 0.2，價格位於通道下緣")

    if pd.notna(bbw):
        if bbw < 0.08:
            comments.append("BandWidth 偏低，屬低波動區")
        elif bbw > 0.18:
            comments.append("BandWidth 偏高，波動正在擴大")

    return {
        "bb_state": state,
        "bb_width": bbw,
        "bb_percent_b": pb,
        "bb_mid_slope": slope,
        "bb_comment": "；".join(comments),
    }


# =========================================================
# 分析摘要
# =========================================================
def _build_signal_summary(analysis: Dict[str, Any]) -> str:
    grade = str(analysis.get("buy_grade", "-"))
    buy_type = str(analysis.get("buy_type", "-"))
    signal = str(analysis.get("signal", "-"))
    trend = str(analysis.get("trend", "-"))
    entry_zone = str(analysis.get("entry_zone", "-"))
    risk_note = str(analysis.get("risk_note", "-"))
    sell_signal = str(analysis.get("sell_signal", "-"))

    msg = (
        f"目前屬於 **{grade} / {buy_type}**，"
        f"訊號為 **{signal}**，趨勢偏 **{trend}**。"
        f"建議買進區：**{entry_zone}**。"
    )

    if risk_note and risk_note != "正常":
        msg += f" 風險提醒：**{risk_note}**。"
    else:
        msg += " 目前風險屬正常範圍。"

    if sell_signal and sell_signal != "未出現明確賣點":
        msg += f" 賣點提醒：**{sell_signal}**。"

    return msg


def _render_top_metrics(
    stock_id: str,
    stock_name: str,
    market_label: str,
    analysis: Dict[str, Any],
    fund: Dict[str, Any],
):
    close_v = analysis.get("close")
    chg_pct = analysis.get("change_pct")
    grade = analysis.get("buy_grade")
    trend = analysis.get("trend")
    score = analysis.get("buy_score", analysis.get("score"))
    dy = fund.get("yield")
    pe = fund.get("pe")
    pb = fund.get("pb")
    roe = fund.get("roe")

    st.markdown(f"## {stock_id} {stock_name}")
    st.caption(f"市場別：{market_label}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("收盤價", _fmt_num(close_v, 2))
    c2.metric("漲跌幅", _fmt_pct(chg_pct, 2))
    c3.metric("買點等級", str(grade) if grade else "-")
    c4.metric("趨勢", str(trend) if trend else "-")
    c5.metric("買點分數", _fmt_num(score, 0))
    c6.metric("殖利率%", _fmt_num(dy, 2))

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("PE", _fmt_num(pe, 2))
    c8.metric("PB", _fmt_num(pb, 2))
    c9.metric("ROE%", _fmt_num(roe, 2))
    c10.metric("建議買進區", str(analysis.get("entry_zone", "-")))


def _render_trade_plan(analysis: Dict[str, Any]):
    st.markdown("### 交易規劃")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("建議買進區", str(analysis.get("entry_zone", "-")))
    c2.metric("建議停損", _fmt_num(analysis.get("stop_loss"), 2))
    c3.metric("第一目標", _fmt_num(analysis.get("take_profit_1"), 2))
    c4.metric("第二目標", _fmt_num(analysis.get("take_profit_2"), 2))

    stop_v = _safe_float(analysis.get("stop_loss"))
    tp1_v = _safe_float(analysis.get("take_profit_1"))
    close_v = _safe_float(analysis.get("close"))

    rr = None
    if pd.notna(stop_v) and pd.notna(tp1_v) and pd.notna(close_v):
        risk = close_v - stop_v
        reward = tp1_v - close_v
        if risk > 0:
            rr = reward / risk

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("風報比", _fmt_num(rr, 2))
    with col2:
        st.info(_build_signal_summary(analysis))


def _render_indicator_table(analysis: Dict[str, Any], bb_summary: Dict[str, Any]):
    st.markdown("### 指標摘要")

    rows = [
        ["買點等級", analysis.get("buy_grade", "-")],
        ["買點型態", analysis.get("buy_type", "-")],
        ["signal", analysis.get("signal", "-")],
        ["trend", analysis.get("trend", "-")],
        ["MACD黃金交叉", analysis.get("macd_gc_text", "-")],
        ["MA5_10黃金交叉", analysis.get("ma5_10_gc_text", "-")],
        ["站上季線", analysis.get("above_ma60_text", "-")],
        ["布林位置", analysis.get("bb_pos_text", "-")],
        ["RSI14", _fmt_num(analysis.get("rsi14"), 2)],
        ["K", _fmt_num(analysis.get("k"), 2)],
        ["D", _fmt_num(analysis.get("d"), 2)],
        ["ATR14", _fmt_num(analysis.get("atr14"), 2)],
        ["布林狀態", bb_summary.get("bb_state", "-")],
        ["BandWidth", _fmt_num(bb_summary.get("bb_width"), 4)],
        ["%B", _fmt_num(bb_summary.get("bb_percent_b"), 2)],
        ["中軌斜率", _fmt_num(bb_summary.get("bb_mid_slope"), 4)],
        ["風險提醒", analysis.get("risk_note", "-")],
        ["賣點提醒", analysis.get("sell_signal", "-")],
    ]

    df = pd.DataFrame(rows, columns=["項目", "數值"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_fundamental_table(fund: Dict[str, Any]):
    st.markdown("### 基本面摘要")

    rows = [
        ["殖利率%", _fmt_num(fund.get("yield"), 2)],
        ["現金股利", _fmt_num(fund.get("dividend"), 2)],
        ["PE", _fmt_num(fund.get("pe"), 2)],
        ["PB", _fmt_num(fund.get("pb"), 2)],
        ["ROE%", _fmt_num(fund.get("roe"), 2)],
    ]
    df = pd.DataFrame(rows, columns=["項目", "數值"])
    st.dataframe(df, use_container_width=True, hide_index=True)


# =========================================================
# 對外主函式
# =========================================================
def render_stock_detail(
    symbol: str,
    finmind_token: str = "",
    req_yield: float = 5.0,
    section_title: str = "## 單一股票完整分析",
):
    st.markdown(section_title)

    try:
        stock_id = normalize_symbol(symbol)
    except Exception:
        stock_id = str(symbol).strip()

    try:
        info_df = get_tw_stock_info(finmind_token if finmind_token else None)
    except Exception:
        info_df = pd.DataFrame()

    market_type = _get_market_type(info_df, stock_id)
    stock_name = _get_stock_name(info_df, stock_id)
    market_text = _market_label(market_type)

    # 先抓價格資料
    try:
        df_raw, source = load_price(
            stock_id,
            market_type,
            finmind_token if finmind_token else None,
        )
    except Exception as e:
        st.error(f"載入股價資料失敗：{e}")
        return

    if df_raw is None or df_raw.empty:
        st.error("找不到股價資料")
        return

    # 加指標
    try:
        df_ind = add_indicators(df_raw)
    except Exception:
        df_ind = df_raw.copy()

    # scanner 分析
    try:
        analysis = analyze_stock(stock_id)
    except Exception:
        # 若 scanner 抓不到，就用本地 df 做保底
        try:
            analysis = analyze_stock(stock_id, name=stock_name)
        except Exception:
            analysis = {
                "symbol": stock_id,
                "name": stock_name,
                "buy_grade": "-",
                "buy_type": "-",
                "signal": "-",
                "trend": "-",
                "entry_zone": "-",
                "risk_note": "-",
                "sell_signal": "-",
                "close": df_ind.iloc[-1]["Close"] if "Close" in df_ind.columns else None,
                "change_pct": None,
                "stop_loss": None,
                "take_profit_1": None,
                "take_profit_2": None,
            }

    # 基本面
    try:
        fund = load_fundamental(
            stock_id,
            market_type,
            finmind_token if finmind_token else None,
        )
        if not isinstance(fund, dict):
            fund = dict(fund)
    except Exception:
        fund = {
            "yield": None,
            "dividend": None,
            "pe": None,
            "pb": None,
            "roe": None,
        }

    # 若殖利率缺值，用股利/價格推估
    try:
        price = float(df_ind.iloc[-1]["Close"])
        dy = fund.get("yield")
        if pd.isna(dy) and not pd.isna(fund.get("dividend")) and price > 0:
            fund["yield"] = fund["dividend"] / price * 100
    except Exception:
        pass

    # 若 ROE 缺值，用 PB/PE 粗估
    try:
        pe = fund.get("pe")
        pb = fund.get("pb")
        roe = fund.get("roe")
        if pd.isna(roe) and not pd.isna(pe) and not pd.isna(pb) and float(pe) > 0:
            fund["roe"] = float(pb) / float(pe) * 100
    except Exception:
        pass

    bb_summary = _build_bb_summary(df_ind)

    # 上方摘要
    _render_top_metrics(
        stock_id=stock_id,
        stock_name=stock_name,
        market_label=market_text,
        analysis=analysis,
        fund=fund,
    )

    st.caption(f"股價來源：{source}")

    # 交易規劃
    _render_trade_plan(analysis)

    # 布林說明
    st.markdown("### 布林通道判讀")
    st.info(bb_summary.get("bb_comment", "無"))

    # 圖表
    try:
        fig = bollinger_trend_chart(
            df_ind,
            title=f"{stock_id} {stock_name} 布林通道趨勢圖",
            bb_length=20,
            bb_std=2.0,
            show_volume=False,
            max_bars=180,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"布林圖繪製失敗：{e}")

    # 下方資訊
    left, right = st.columns(2)

    with left:
        _render_indicator_table(analysis, bb_summary)

    with right:
        _render_fundamental_table(fund)
