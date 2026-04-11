from __future__ import annotations

import os
import sys
import json
import subprocess

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go

DATA_DIR    = "data"
LATEST_JSON = os.path.join(DATA_DIR, "latest_scan.json")
HISTORY_CSV = os.path.join(DATA_DIR, "scan_history.csv")
LEARNING_JSON = os.path.join(DATA_DIR, "ai_learning.json")

# =========================================================
# 頁面設定
# =========================================================

st.set_page_config(
    page_title="台股 AI 自動掃描",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# 基本工具
# =========================================================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default

def load_json_file(path: str, default_value):
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

def auto_refresh_block(seconds: int):
    if seconds <= 0:
        return
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        height=0,
    )

def run_manual_scan():
    result = subprocess.run(
        [sys.executable, "scanner.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return result.returncode, result.stdout, result.stderr

def empty_latest():
    return {
        "updated_at": "",
        "status":     "empty",
        "message":    "目前尚無掃描結果，請先手動掃描或等待排程執行。",
        "summary":    {},
        "learning":   {},
        "results":    [],
        "failed_symbols": [],
    }

# =========================================================
# Cache
# =========================================================

@st.cache_data(ttl=60)
def load_latest():
    ensure_data_dir()
    return load_json_file(LATEST_JSON, empty_latest())

@st.cache_data(ttl=60)
def load_learning():
    return load_json_file(
        LEARNING_JSON,
        {"updated_at": "", "weights": {}, "thresholds": {}, "rule_stats": {}},
    )

@st.cache_data(ttl=60)
def load_history():
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

# =========================================================
# session_state 預設值
# =========================================================

defaults = {
    "enable_auto_refresh":   False,
    "refresh_sec":           60,
    "top_n":                 50,
    "grade_filter":          ["A1", "A2"],
    "chart_period":          "9mo",
    "min_price":             100.0,
    "max_price":             1000.0,
    "min_volume":            1000,
    "max_volume":            0,
    "min_vol_ratio":         1.0,
    "selected_stock_option": None,
    "single_stock_code":     "",
    "page_mode":             "最新結果",
    "pending_page_mode":     None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 技術指標
# =========================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def enrich_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["EMA12"]     = ema(df["Close"], 12)
    df["EMA26"]     = ema(df["Close"], 26)
    df["DIF"]       = df["EMA12"] - df["EMA26"]
    df["DEA"]       = ema(df["DIF"], 9)
    df["MACD_HIST"] = df["DIF"] - df["DEA"]
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_MID"]   = mid
    df["BB_UPPER"] = mid + 2 * std
    df["BB_LOWER"] = mid - 2 * std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, pd.NA)
    df["VOL_MA5"]   = df["Volume"].rolling(5).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA5"].replace(0, pd.NA)
    df["RECENT_HIGH_20"] = df["High"].rolling(20).max()
    df["RECENT_LOW_20"]  = df["Low"].rolling(20).min()
    return df

def tw_symbol_from_code(code: str) -> list[str]:
    code = str(code).strip()
    if not code.isdigit():
        return []
    return [f"{code}.TW", f"{code}.TWO"]

@st.cache_data(ttl=1800)
def fetch_stock_chart_data(symbol: str, period: str = "9mo") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    df = df[needed].dropna().copy()
    df = enrich_chart_df(df)
    return df

def fetch_single_stock_by_code(code: str, period: str = "9mo") -> tuple[str | None, pd.DataFrame]:
    for symbol in tw_symbol_from_code(code):
        df = fetch_stock_chart_data(symbol, period=period)
        if not df.empty:
            return symbol, df
    return None, pd.DataFrame()

# =========================================================
# 圖表
# =========================================================

def make_candlestick_bollinger_chart(df: pd.DataFrame, title: str):
    chart_df = df.tail(120).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="K線",
    ))
    for col, name in [("MA5","MA5"),("MA10","MA10"),("MA20","MA20"),("MA60","MA60"),
                      ("BB_UPPER","布林上軌"),("BB_MID","布林中軌"),("BB_LOWER","布林下軌")]:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="價格",
                      height=620, xaxis_rangeslider_visible=False)
    return fig

def make_kline_trend_chart(df: pd.DataFrame, price_pack: dict, title: str):
    chart_df = df.tail(120).copy()
    fig = go.Figure()
    for col, name in [("Close","收盤"),("MA10","MA10"),("MA20","MA20"),("MA60","MA60")]:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=name))
    hlines = [
        (price_pack.get("support_1"),             "支撐1",     "dot"),
        (price_pack.get("support_2"),             "支撐2",     "dot"),
        (price_pack.get("support_3"),             "支撐3",     "dot"),
        (price_pack.get("resistance_1"),          "壓力1",     "dash"),
        (price_pack.get("resistance_2"),          "壓力2",     "dash"),
        (price_pack.get("aggressive_buy_price"),  "積極買點",  "solid"),
        (price_pack.get("pullback_buy_price"),    "回踩買點",  "solid"),
        (price_pack.get("conservative_buy_price"),"保守買點",  "solid"),
        (price_pack.get("sell_price_1"),          "第一賣點",  "dash"),
        (price_pack.get("sell_price_2"),          "第二賣點",  "dash"),
        (price_pack.get("stop_loss_short"),       "短線停損",  "dot"),
        (price_pack.get("stop_loss_wave"),        "波段停損",  "dot"),
    ]
    for y, name, dash in hlines:
        if y and safe_float(y) > 0:
            fig.add_hline(y=safe_float(y), line_dash=dash,
                          annotation_text=name, annotation_position="right")
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="價格", height=500)
    return fig

def make_volume_chart(df: pd.DataFrame, title: str):
    chart_df = df.tail(120).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="成交量"))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="成交量", height=240)
    return fig

def make_macd_chart(df: pd.DataFrame, title: str):
    chart_df = df.tail(120).copy()
    colors = ["red" if v >= 0 else "green" for v in chart_df["MACD_HIST"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_HIST"],
                         name="MACD柱", marker_color=colors))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["DIF"], mode="lines", name="DIF"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["DEA"], mode="lines", name="DEA"))
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="MACD", height=320)
    return fig

# =========================================================
# 操作建議（單筆 / 掃描圖表共用）
# =========================================================

def build_trade_prices_from_df(df: pd.DataFrame) -> dict:
    curr        = df.iloc[-1]
    close       = safe_float(curr["Close"])
    ma10        = safe_float(curr["MA10"])
    ma20        = safe_float(curr["MA20"])
    ma60        = safe_float(curr["MA60"])
    bb_upper    = safe_float(curr["BB_UPPER"])
    bb_mid      = safe_float(curr["BB_MID"])
    recent_high = safe_float(df.tail(20)["High"].max())
    recent_low  = safe_float(df.tail(20)["Low"].min())
    return {
        "aggressive_buy_price":   round(max(close, recent_high), 2),
        "pullback_buy_price":     round(ma10, 2),
        "conservative_buy_price": round(bb_mid, 2),
        "sell_price_1":           round(bb_upper, 2),
        "sell_price_2":           round(recent_high, 2),
        "stop_loss_short":        round(ma10, 2),
        "stop_loss_wave":         round(ma20, 2),
        "stop_loss_hard":         round(ma60, 2),
        "support_1":              round(ma10, 2),
        "support_2":              round(ma20, 2),
        "support_3":              round(ma60, 2),
        "resistance_1":           round(bb_upper, 2),
        "resistance_2":           round(recent_high, 2),
        "recent_low_20":          round(recent_low, 2),
    }

def build_analysis_summary(df: pd.DataFrame, price_pack: dict) -> dict:
    curr      = df.iloc[-1]
    close     = safe_float(curr["Close"])
    ma10      = safe_float(curr["MA10"])
    ma20      = safe_float(curr["MA20"])
    ma60      = safe_float(curr["MA60"])
    bb_upper  = safe_float(curr["BB_UPPER"])
    bb_mid    = safe_float(curr["BB_MID"])
    bb_lower  = safe_float(curr["BB_LOWER"])
    dif       = safe_float(curr["DIF"])
    dea       = safe_float(curr["DEA"])
    vol_ratio = safe_float(curr["VOL_RATIO"])

    if close > ma10 and close > ma20 and dif > dea:
        action = "偏多看待，可觀察回踩不破後再進。"
    elif close > ma20 and close < ma60:
        action = "整理轉強中，適合等確認站穩後再介入。"
    elif close < ma10 and close < ma20:
        action = "轉弱觀望，不宜急著進場。"
    else:
        action = "先觀察，等待更明確方向。"

    kline = (
        "K線仍在短中期均線之上，結構偏強。" if close > ma10 > ma20
        else "K線仍守住中期均線，但短線轉整理。" if close > ma20
        else "K線跌落短中期均線下，走勢轉弱。"
    )
    boll = (
        "股價突破布林上軌，短線偏強但不宜追高。" if close > bb_upper
        else "股價位於布林中軌之上，維持偏多整理。" if close >= bb_mid
        else "股價跌到布林中軌下，偏弱整理。" if close > bb_lower
        else "股價靠近布林下軌，宜保守。"
    )
    macd = (
        "MACD偏多，短線動能仍在。" if dif > dea
        else "MACD轉弱，反彈力道有限。"
    )

    risks = []
    if close > bb_upper and vol_ratio < 1.0:
        risks.append("上軌突破但量不足，需防假突破。")
    if close > bb_upper and vol_ratio >= 1.5:
        risks.append("短線乖離偏大，不宜重壓追高。")
    if close < ma10:
        risks.append("跌破 MA10，短線轉弱。")
    if close < ma20:
        risks.append("跌破 MA20，中短線保守。")
    if close < ma60:
        risks.append("跌破季線，波段結構轉差。")
    if not risks:
        risks.append("目前風險中性，仍需觀察量價。")

    return {
        "action": action,
        "kline":  kline,
        "boll":   boll,
        "macd":   macd,
        "risks":  risks,
        "buy_prices": {
            "積極買點": price_pack["aggressive_buy_price"],
            "回踩買點": price_pack["pullback_buy_price"],
            "保守買點": price_pack["conservative_buy_price"],
        },
        "sell_prices": {
            "第一賣點": price_pack["sell_price_1"],
            "第二賣點": price_pack["sell_price_2"],
        },
        "stop_prices": {
            "短線停損": price_pack["stop_loss_short"],
            "波段停損": price_pack["stop_loss_wave"],
        },
    }

def render_analysis(summary_pack: dict):
    st.markdown("### 操作建議")
    st.write(summary_pack["action"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**買點**")
        for label, val in summary_pack["buy_prices"].items():
            st.write(f"- {label}：{val:.2f}")
    with col2:
        st.markdown("**賣點**")
        for label, val in summary_pack["sell_prices"].items():
            st.write(f"- {label}：{val:.2f}")
    with col3:
        st.markdown("**停損**")
        for label, val in summary_pack["stop_prices"].items():
            st.write(f"- {label}：{val:.2f}")

    st.markdown("### 趨勢摘要")
    st.write(f"- K線：{summary_pack['kline']}")
    st.write(f"- 布林：{summary_pack['boll']}")
    st.write(f"- MACD：{summary_pack['macd']}")

    st.markdown("### 風險提醒")
    for item in summary_pack["risks"]:
        st.write(f"- {item}")

# =========================================================
# 顯示工具
# =========================================================

def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "grade" not in df.columns:
        return df.copy()
    rank_map = {"A1": 0, "A2": 1}
    out = df.copy()
    out["grade_rank"] = out["grade"].map(rank_map).fillna(9)
    out = out.sort_values(
        by=["grade_rank", "score", "vol_ratio"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    return out

def format_result_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "grade", "code", "name", "type", "close", "score",
        "volume", "vol_ratio", "rsi14",
        "aggressive_buy_price", "pullback_buy_price", "conservative_buy_price",
        "sell_price_1", "sell_price_2",
        "stop_loss_short", "stop_loss_wave",
        "reasons", "scan_time",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()

def grade_badge(grade: str) -> str:
    if grade == "A1":
        return "🔴 A1 突破"
    if grade == "A2":
        return "🟡 A2 回踩"
    return grade

def learning_panel(learning: dict):
    st.subheader("AI 自我學習狀態")
    stats      = learning.get("rule_stats", {})
    thresholds = learning.get("thresholds", {})
    weights    = learning.get("weights", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("已標記樣本數",   stats.get("total_labeled", 0))
    c2.metric("整體5日成功率",  f"{stats.get('success_rate_5d', 0):.1f}%")
    c3.metric("模型更新時間",   learning.get("updated_at", "") or "N/A")

    grade_stats = stats.get("grade_stats", {})
    if grade_stats:
        st.markdown("#### A1 / A2 歷史績效")
        gs_rows = []
        for g, d in grade_stats.items():
            gs_rows.append({
                "分級": grade_badge(g),
                "樣本數": d.get("samples", 0),
                "5日勝率%": f"{d.get('success_rate_5d', 0):.1f}%",
                "平均5日報酬%": f"{d.get('avg_ret_5d', 0):.2f}%",
            })
        st.dataframe(pd.DataFrame(gs_rows), hide_index=True)

    with st.expander("查看學習細節", expanded=False):
        if weights:
            st.markdown("**加分條件權重**")
            wdf = pd.DataFrame([{"條件": k, "權重": v} for k, v in weights.items()])
            st.dataframe(wdf, hide_index=True)
        if thresholds:
            st.markdown("**動態門檻**")
            tdf = pd.DataFrame([{"參數": k, "值": v} for k, v in thresholds.items()])
            st.dataframe(tdf, hide_index=True)
        by_cond = stats.get("by_condition", {})
        if by_cond:
            st.markdown("**條件個別勝率**")
            cdf = pd.DataFrame([
                {"條件": k, "樣本": v["samples"],
                 "5日勝率%": v["success_rate_5d"],
                 "平均報酬%": v["avg_ret_5d"]}
                for k, v in by_cond.items()
            ])
            st.dataframe(cdf, hide_index=True)

def set_selected_stock_by_code(code: str, df_show: pd.DataFrame):
    if df_show.empty:
        return
    matches = df_show[df_show["code"].astype(str) == str(code)]
    if matches.empty:
        return
    row    = matches.iloc[0]
    option = f"{row['code']} {row['name']} ({row['grade']})"
    st.session_state.selected_stock_option = option
    st.session_state.pending_page_mode     = "掃描個股圖表"

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("操作面板")

st.sidebar.checkbox("啟用自動刷新", key="enable_auto_refresh")
st.sidebar.selectbox("自動刷新秒數", [30, 60, 120, 300], key="refresh_sec",
                     format_func=lambda x: f"{x} 秒")
st.sidebar.selectbox("顯示筆數", [20, 50, 100], key="top_n")
st.sidebar.multiselect("分級篩選", ["A1", "A2"], default=["A1", "A2"], key="grade_filter")
st.sidebar.selectbox("圖表期間", ["3mo", "6mo", "9mo", "12mo"], key="chart_period")

st.sidebar.markdown("### 價格設定區間")
st.sidebar.number_input("最低價格", min_value=0.0, step=1.0,  key="min_price")
st.sidebar.number_input("最高價格", min_value=0.0, step=10.0, key="max_price")

st.sidebar.markdown("### 成交量設定")
st.sidebar.number_input("最低成交量", min_value=0, step=100, key="min_volume")
st.sidebar.number_input("最高成交量", min_value=0, step=100, key="max_volume")

st.sidebar.markdown("### 量比設定")
st.sidebar.number_input("最低量比", min_value=0.0, step=0.1, key="min_vol_ratio")

st.sidebar.markdown("### 單筆個股分析")
st.sidebar.text_input("輸入股票代號", key="single_stock_code")

if st.sidebar.button("🔄 手動重新掃描", use_container_width=True):
    with st.spinner("正在執行掃描，請稍候..."):
        code_ret, out, err = run_manual_scan()
        st.cache_data.clear()
        if code_ret == 0:
            st.sidebar.success("掃描完成")
            st.rerun()
        else:
            st.sidebar.error("掃描失敗")
            st.sidebar.code(err if err else out)

# =========================================================
# 主畫面
# =========================================================

st.title("📈 台股 AI 自動掃描")

latest   = load_latest()
learning = load_learning()
hist_df  = load_history()

if st.session_state.enable_auto_refresh:
    auto_refresh_block(st.session_state.refresh_sec)

left, right = st.columns([3, 1])
with left:
    st.caption(f"最後掃描時間：{latest.get('updated_at', 'N/A')}")
    status = latest.get("status", "empty")
    if status == "ok":
        st.success("已載入最新掃描結果")
    elif status == "empty":
        st.warning(latest.get("message", "尚無資料"))
    else:
        st.error(latest.get("message", "讀取失敗"))
with right:
    st.caption(f"資料檔：{LATEST_JSON}")

summary = latest.get("summary", {})
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("總掃描池",    summary.get("pool_total",    0))
m2.metric("成功分析數",  summary.get("success_count", 0))
m3.metric("缺資料/失敗", summary.get("failed_count",  0))
m4.metric("A1 突破型",   summary.get("A1_count",      0))
m5.metric("A2 回踩型",   summary.get("A2_count",      0))

# 篩選
results = latest.get("results", [])
df = pd.DataFrame(results)
df = rank_df(df)

if not df.empty and "grade" in df.columns:
    df = df[df["grade"].isin(st.session_state.grade_filter)]
if not df.empty:
    if "close" in df.columns:
        df = df[df["close"] >= st.session_state.min_price]
        df = df[df["close"] <= st.session_state.max_price]
    if "volume" in df.columns:
        df = df[df["volume"] >= st.session_state.min_volume]
        if st.session_state.max_volume > 0:
            df = df[df["volume"] <= st.session_state.max_volume]
    if "vol_ratio" in df.columns:
        df = df[df["vol_ratio"] >= st.session_state.min_vol_ratio]

df_show = df.head(st.session_state.top_n).copy() if not df.empty else pd.DataFrame()

# 頁面切換
page_options = ["最新結果", "單筆個股分析", "掃描個股圖表", "AI學習", "歷史紀錄"]

if st.session_state.pending_page_mode in page_options:
    st.session_state.page_mode     = st.session_state.pending_page_mode
    st.session_state.pending_page_mode = None

page_mode = st.radio("功能選單", page_options, horizontal=True, key="page_mode")

# =========================================================
# 最新結果
# =========================================================

if page_mode == "最新結果":
    st.subheader("最新掃描結果")

    if df_show.empty:
        st.info("目前沒有符合條件的 A 級結果。")
    else:
        display_df = format_result_df(df_show.drop(columns=["grade_rank"], errors="ignore"))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Top 10（可直接查看個股圖表）")
        top10 = df_show.head(10).copy()
        for i, row in top10.iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.8, 1.0, 2.0, 1.5, 1.5, 1.5, 1.2])
            c1.write(f"**{grade_badge(row.get('grade', '-'))}**")
            c2.write(f"**{row.get('code', '')}**")
            c3.write(f"{row.get('name', '')}")
            c4.write(f"收盤：{safe_float(row.get('close', 0)):.2f}")
            c5.write(f"分數：{safe_int(row.get('score', 0))}")
            c6.write(f"量比：{safe_float(row.get('vol_ratio', 0)):.2f}")
            if c7.button("查看圖表", key=f"top10_view_{row.get('code', i)}"):
                set_selected_stock_by_code(str(row.get("code", "")), df_show)
                st.rerun()

    failed = latest.get("failed_symbols", [])
    if failed:
        with st.expander("查看部分抓不到資料的代號", expanded=False):
            st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)

# =========================================================
# 單筆個股分析
# =========================================================

elif page_mode == "單筆個股分析":
    st.subheader("單筆個股分析")
    code = st.session_state.single_stock_code.strip()
    if not code:
        st.info("請在左側輸入股票代號，例如 2330、2317、2454。")
    else:
        symbol, single_df = fetch_single_stock_by_code(code, st.session_state.chart_period)
        if single_df.empty or symbol is None:
            st.warning("找不到此股票資料，請確認代號。")
        else:
            pack = build_trade_prices_from_df(single_df)
            summary_pack = build_analysis_summary(single_df, pack)
            curr = single_df.iloc[-1]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("股票代號", code)
            c2.metric("收盤價",   round(safe_float(curr["Close"]), 2))
            c3.metric("成交量",   int(safe_float(curr["Volume"], 0)))
            c4.metric("量比",     round(safe_float(curr["VOL_RATIO"], 0), 2))
            c5.metric("代號來源", symbol)

            render_analysis(summary_pack)
            st.plotly_chart(make_candlestick_bollinger_chart(single_df, f"{code} K線 / 布林通道"),
                            use_container_width=True)
            st.plotly_chart(make_kline_trend_chart(single_df, pack, f"{code} 支撐壓力與買賣點"),
                            use_container_width=True)
            st.plotly_chart(make_volume_chart(single_df, f"{code} 成交量"),
                            use_container_width=True)
            st.plotly_chart(make_macd_chart(single_df, f"{code} MACD"),
                            use_container_width=True)

# =========================================================
# 掃描個股圖表
# =========================================================

elif page_mode == "掃描個股圖表":
    st.subheader("掃描個股圖表")
    if df_show.empty:
        st.info("目前沒有可選股票。")
    else:
        options = [f"{r['code']} {r['name']} ({r['grade']})" for _, r in df_show.iterrows()]
        if st.session_state.selected_stock_option not in options:
            st.session_state.selected_stock_option = options[0]

        selected = st.selectbox("選擇掃描結果中的股票", options, key="selected_stock_option")
        idx  = options.index(selected)
        row  = df_show.iloc[idx]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("分級",   grade_badge(row.get("grade", "-")))
        c2.metric("收盤價", row.get("close", ""))
        c3.metric("加分",   row.get("score", ""))
        c4.metric("成交量", row.get("volume", ""))
        c5.metric("量比",   row.get("vol_ratio", ""))
        c6.metric("RSI14",  row.get("rsi14", ""))

        st.info(f"觸發條件：{row.get('reasons', '-')}")

        symbol   = row.get("symbol", "")
        chart_df = fetch_stock_chart_data(symbol, period=st.session_state.chart_period)

        if chart_df.empty:
            st.warning("無法讀取圖表資料。")
        else:
            pack = {
                "aggressive_buy_price":   safe_float(row.get("aggressive_buy_price")),
                "pullback_buy_price":     safe_float(row.get("pullback_buy_price")),
                "conservative_buy_price": safe_float(row.get("conservative_buy_price")),
                "sell_price_1":           safe_float(row.get("sell_price_1")),
                "sell_price_2":           safe_float(row.get("sell_price_2")),
                "stop_loss_short":        safe_float(row.get("stop_loss_short")),
                "stop_loss_wave":         safe_float(row.get("stop_loss_wave")),
                "stop_loss_hard":         safe_float(row.get("stop_loss_hard")),
                "support_1":              safe_float(row.get("support_1")),
                "support_2":              safe_float(row.get("support_2")),
                "support_3":              safe_float(row.get("support_3")),
                "resistance_1":           safe_float(row.get("resistance_1")),
                "resistance_2":           safe_float(row.get("resistance_2")),
            }
            summary_pack = build_analysis_summary(chart_df, pack)
            render_analysis(summary_pack)

            name = row.get("name", "")
            st.plotly_chart(make_candlestick_bollinger_chart(chart_df, f"{row['code']} {name} K線 / 布林通道"),
                            use_container_width=True)
            st.plotly_chart(make_kline_trend_chart(chart_df, pack, f"{row['code']} {name} 支撐壓力與買賣點"),
                            use_container_width=True)
            st.plotly_chart(make_volume_chart(chart_df, f"{row['code']} {name} 成交量"),
                            use_container_width=True)
            st.plotly_chart(make_macd_chart(chart_df, f"{row['code']} {name} MACD"),
                            use_container_width=True)

# =========================================================
# AI 學習
# =========================================================

elif page_mode == "AI學習":
    learning_panel(learning)

# =========================================================
# 歷史紀錄
# =========================================================

elif page_mode == "歷史紀錄":
    st.subheader("歷史紀錄")
    if hist_df.empty:
        st.info("目前尚無歷史紀錄。")
    else:
        show_hist = hist_df.copy()
        if "grade" in show_hist.columns:
            show_hist = show_hist[show_hist["grade"].isin(st.session_state.grade_filter)]
        if "close" in show_hist.columns:
            show_hist = show_hist[show_hist["close"] >= st.session_state.min_price]
            show_hist = show_hist[show_hist["close"] <= st.session_state.max_price]
        if "volume" in show_hist.columns:
            show_hist = show_hist[show_hist["volume"] >= st.session_state.min_volume]
            if st.session_state.max_volume > 0:
                show_hist = show_hist[show_hist["volume"] <= st.session_state.max_volume]
        if "vol_ratio" in show_hist.columns:
            show_hist = show_hist[show_hist["vol_ratio"] >= st.session_state.min_vol_ratio]

        hist_cols = [
            "scan_date", "code", "name", "grade", "type", "close",
            "volume", "vol_ratio", "score",
            "aggressive_buy_price", "pullback_buy_price",
            "sell_price_1", "stop_loss_short",
            "ret_3d", "ret_5d", "ret_10d", "success_5d",
        ]
        hist_cols = [c for c in hist_cols if c in show_hist.columns]
        st.dataframe(show_hist[hist_cols].tail(500), use_container_width=True, hide_index=True)
