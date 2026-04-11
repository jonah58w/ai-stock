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
from plotly.subplots import make_subplots

DATA_DIR      = "data"
LATEST_JSON   = os.path.join(DATA_DIR, "latest_scan.json")
HISTORY_CSV   = os.path.join(DATA_DIR, "scan_history.csv")
LEARNING_JSON = os.path.join(DATA_DIR, "ai_learning.json")

st.set_page_config(page_title="台股 AI 自動掃描", page_icon="📈", layout="wide")

# =========================================================
# 基本工具
# =========================================================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def safe_float(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None: return default
        return int(float(x))
    except Exception:
        return default

def load_json_file(path, default_value):
    if not os.path.exists(path): return default_value
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default_value

def auto_refresh_block(seconds):
    if seconds <= 0: return
    components.html(f"<script>setTimeout(function(){{window.parent.location.reload();}},{seconds*1000});</script>", height=0)

def run_manual_scan():
    r = subprocess.run([sys.executable, "scanner.py"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return r.returncode, r.stdout, r.stderr

def empty_latest():
    return {"updated_at":"","status":"empty","message":"目前尚無掃描結果，請先手動掃描或等待排程執行。",
            "summary":{},"learning":{},"results":[],"failed_symbols":[]}

# =========================================================
# Cache
# =========================================================

@st.cache_data(ttl=60)
def load_latest():
    ensure_data_dir()
    return load_json_file(LATEST_JSON, empty_latest())

@st.cache_data(ttl=60)
def load_learning():
    return load_json_file(LEARNING_JSON, {"updated_at":"","weights":{},"thresholds":{},"rule_stats":{}})

@st.cache_data(ttl=60)
def load_history():
    if not os.path.exists(HISTORY_CSV): return pd.DataFrame()
    try: return pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    except Exception: return pd.DataFrame()

# =========================================================
# Session state
# =========================================================

defaults = {
    "enable_auto_refresh": False, "refresh_sec": 60,
    "top_n": 50, "grade_filter": ["A1","A2"], "chart_period": "9mo",
    "min_price": 100.0, "max_price": 1000.0,
    "min_volume": 1000, "max_volume": 0, "min_vol_ratio": 1.0,
    "selected_stock_option": None, "single_stock_code": "",
    "page_mode": "最新結果", "pending_page_mode": None,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

VALID_PAGES = ["最新結果","單筆個股分析","掃描個股圖表","AI學習","歷史紀錄"]
if st.session_state.get("refresh_sec") not in [30,60,120,300]: st.session_state["refresh_sec"] = 60
if st.session_state.get("top_n") not in [20,50,100]: st.session_state["top_n"] = 50
if st.session_state.get("chart_period") not in ["3mo","6mo","9mo","12mo"]: st.session_state["chart_period"] = "9mo"
if not isinstance(st.session_state.get("grade_filter"), list) or not all(g in ["A1","A2"] for g in st.session_state.get("grade_filter",[])):
    st.session_state["grade_filter"] = ["A1","A2"]
if st.session_state.get("page_mode") not in VALID_PAGES: st.session_state["page_mode"] = "最新結果"

# =========================================================
# 技術指標
# =========================================================

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def enrich_chart_df(df):
    df = df.copy()
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["EMA12"] = ema(df["Close"], 12)
    df["EMA26"] = ema(df["Close"], 26)
    df["DIF"]   = df["EMA12"] - df["EMA26"]
    df["DEA"]   = ema(df["DIF"], 9)
    df["MACD_HIST"] = df["DIF"] - df["DEA"]
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std(ddof=0)
    df["BB_MID"]   = mid
    df["BB_UPPER"] = mid + 2*std
    df["BB_LOWER"] = mid - 2*std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, pd.NA)
    df["VOL_MA5"]   = df["Volume"].rolling(5).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA5"].replace(0, pd.NA)
    df["RECENT_HIGH_20"] = df["High"].rolling(20).max()
    df["RECENT_LOW_20"]  = df["Low"].rolling(20).min()
    return df

def tw_symbols(code):
    code = str(code).strip()
    if not code.isdigit(): return []
    return [f"{code}.TW", f"{code}.TWO"]

@st.cache_data(ttl=1800)
def fetch_chart(symbol, period="9mo"):
    try:
        from curl_cffi import requests as cr
        session = cr.Session(impersonate="chrome110")
        ticker  = yf.Ticker(symbol, session=session)
        df = ticker.history(period=period, interval="1d", auto_adjust=False, actions=False)
        if df is None or df.empty: return pd.DataFrame()
        needed = ["Open","High","Low","Close","Volume"]
        for c in needed:
            if c not in df.columns: return pd.DataFrame()
        return enrich_chart_df(df[needed].dropna().copy())
    except Exception:
        return pd.DataFrame()

def fetch_by_code(code, period="9mo"):
    for sym in tw_symbols(code):
        df = fetch_chart(sym, period)
        if not df.empty: return sym, df
    return None, pd.DataFrame()

# =========================================================
# 圖表（TradingView 深色三面板）
# =========================================================

_BG   = "#131722"
_PAN  = "#1e222d"
_GRID = "#2a2e39"
_TXT  = "#d1d4dc"
_UP   = "#ef5350"   # 漲紅（台灣）
_DN   = "#26a69a"   # 跌綠（台灣）

def make_professional_chart(df, price_pack, title):
    chart_df = df.tail(120).copy()
    n   = len(chart_df)
    xs  = list(range(n))
    dts = [d.strftime("%m/%d") for d in chart_df.index]
    ud  = [_UP if c >= o else _DN for c, o in zip(chart_df["Close"], chart_df["Open"])]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.60, 0.22, 0.18], vertical_spacing=0.018)

    # 布林填充
    fig.add_trace(go.Scatter(
        x=xs+xs[::-1], y=list(chart_df["BB_UPPER"])+list(chart_df["BB_LOWER"][::-1]),
        fill="toself", fillcolor="rgba(41,98,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # 布林三軌
    for col, lbl, color, dash in [
        ("BB_UPPER","布林上","#5c7cfa","dot"),
        ("BB_MID",  "布林中","#ff9800","dash"),
        ("BB_LOWER","布林下","#5c7cfa","dot"),
    ]:
        fig.add_trace(go.Scatter(x=xs, y=chart_df[col], mode="lines", name=lbl,
            line=dict(color=color, width=1, dash=dash),
            hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # 均線
    for col, lbl, color in [
        ("MA5","MA5","#f48fb1"), ("MA10","MA10","#ce93d8"),
        ("MA20","MA20","#4dd0e1"), ("MA60","MA60","#ffb74d"),
    ]:
        fig.add_trace(go.Scatter(x=xs, y=chart_df[col], mode="lines", name=lbl,
            line=dict(color=color, width=1.4),
            hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # K線
    fig.add_trace(go.Candlestick(
        x=xs, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name="K線",
        increasing=dict(line=dict(color=_UP, width=1.5), fillcolor=_UP),
        decreasing=dict(line=dict(color=_DN, width=1.5), fillcolor=_DN),
        hovertext=[f"開:{o:.2f}  高:{h:.2f}  低:{l:.2f}  收:{c:.2f}"
                   for o,h,l,c in zip(chart_df["Open"],chart_df["High"],chart_df["Low"],chart_df["Close"])],
        hoverlabel=dict(bgcolor=_PAN),
    ), row=1, col=1)

    # 買賣停損線
    for y_val, lbl, color, dash in [
        (price_pack.get("aggressive_buy_price"), "積極買", _UP,      "solid"),
        (price_pack.get("pullback_buy_price"),   "回踩買", "#ff8a65","dash"),
        (price_pack.get("sell_price_1"),         "賣點1",  "#42a5f5","dot"),
        (price_pack.get("stop_loss_short"),      "停損",   "#ffca28","dot"),
    ]:
        if y_val and safe_float(y_val) > 0:
            fig.add_hline(y=safe_float(y_val), row=1, col=1,
                line=dict(color=color, width=0.9, dash=dash),
                annotation=dict(text=f"<b>{lbl}</b> {safe_float(y_val):.1f}",
                    font=dict(size=10, color=color), bgcolor="rgba(19,23,34,0.75)",
                    borderpad=2, x=1.0, xanchor="right"),
                annotation_position="right")

    # 成交量
    fig.add_trace(go.Bar(x=xs, y=chart_df["Volume"], marker_color=ud, marker_line_width=0,
        name="成交量", showlegend=False, hovertemplate="量: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=chart_df["Volume"].rolling(5).mean(), mode="lines",
        line=dict(color="#ffeb3b", width=1), name="量MA5", showlegend=False,
        hovertemplate="量MA5: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # MACD
    mc = [_UP if v >= 0 else _DN for v in chart_df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=xs, y=chart_df["MACD_HIST"], marker_color=mc, marker_line_width=0,
        showlegend=False, hovertemplate="HIST: %{y:.4f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=xs, y=chart_df["DIF"], mode="lines",
        line=dict(color="#f48fb1", width=1.3), showlegend=False,
        hovertemplate="DIF: %{y:.4f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=xs, y=chart_df["DEA"], mode="lines",
        line=dict(color="#ffeb3b", width=1.3), showlegend=False,
        hovertemplate="DEA: %{y:.4f}<extra></extra>",
    ), row=3, col=1)

    step = max(1, n//12)
    tv   = xs[::step]
    tt   = [dts[i] for i in tv]
    ax   = dict(showgrid=True, gridcolor=_GRID, gridwidth=0.5,
                zeroline=False, showline=True, linecolor=_GRID,
                tickfont=dict(size=10, color=_TXT))

    fig.update_layout(
        title=dict(text=title, font=dict(color=_TXT, size=14), x=0.01),
        paper_bgcolor=_BG, plot_bgcolor=_PAN,
        font=dict(color=_TXT, size=11), height=720,
        margin=dict(l=10, r=100, t=40, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h", x=0, y=1.04),
        hovermode="x unified", hoverlabel=dict(bgcolor=_PAN, font_size=11, bordercolor=_GRID),
        xaxis =dict(**ax, tickvals=tv, ticktext=tt, tickangle=-30),
        xaxis2=dict(**ax, tickvals=tv, ticktext=tt, showticklabels=False),
        xaxis3=dict(**ax, tickvals=tv, ticktext=tt, tickangle=-30),
        yaxis =dict(**ax, side="right"),
        yaxis2=dict(**ax, side="right"),
        yaxis3=dict(**ax, side="right"),
    )
    fig.update_xaxes(showspikes=True, spikecolor=_TXT, spikesnap="cursor", spikemode="across", spikethickness=0.5)
    fig.update_yaxes(showspikes=True, spikecolor=_TXT, spikethickness=0.5)
    return fig

# =========================================================
# 操作建議
# =========================================================

def build_trade_prices(df):
    curr = df.iloc[-1]
    close = safe_float(curr["Close"]); ma10 = safe_float(curr["MA10"])
    ma20  = safe_float(curr["MA20"]);  ma60  = safe_float(curr["MA60"])
    bb_u  = safe_float(curr["BB_UPPER"]); bb_m = safe_float(curr["BB_MID"])
    rh = safe_float(df.tail(20)["High"].max()); rl = safe_float(df.tail(20)["Low"].min())
    return {
        "aggressive_buy_price": round(max(close,rh),2), "pullback_buy_price": round(ma10,2),
        "conservative_buy_price": round(bb_m,2), "sell_price_1": round(bb_u,2),
        "sell_price_2": round(rh,2), "stop_loss_short": round(ma10,2),
        "stop_loss_wave": round(ma20,2), "stop_loss_hard": round(ma60,2),
        "support_1": round(ma10,2), "support_2": round(ma20,2), "support_3": round(ma60,2),
        "resistance_1": round(bb_u,2), "resistance_2": round(rh,2), "recent_low_20": round(rl,2),
    }

def build_summary(df, pack):
    curr = df.iloc[-1]
    close = safe_float(curr["Close"]); ma10 = safe_float(curr["MA10"])
    ma20  = safe_float(curr["MA20"]);  ma60  = safe_float(curr["MA60"])
    bb_u  = safe_float(curr["BB_UPPER"]); bb_m = safe_float(curr["BB_MID"])
    bb_l  = safe_float(curr["BB_LOWER"])
    dif   = safe_float(curr["DIF"]); dea = safe_float(curr["DEA"])
    vr    = safe_float(curr["VOL_RATIO"])

    if close > ma10 and close > ma20 and dif > dea: action = "偏多看待，可觀察回踩不破後再進。"
    elif close > ma20 and close < ma60: action = "整理轉強中，適合等確認站穩後再介入。"
    elif close < ma10 and close < ma20: action = "轉弱觀望，不宜急著進場。"
    else: action = "先觀察，等待更明確方向。"

    kline = ("K線仍在短中期均線之上，結構偏強。" if close>ma10>ma20
             else "K線仍守住中期均線，但短線轉整理。" if close>ma20
             else "K線跌落短中期均線下，走勢轉弱。")
    boll  = ("股價突破布林上軌，短線偏強但不宜追高。" if close>bb_u
             else "股價位於布林中軌之上，維持偏多整理。" if close>=bb_m
             else "股價跌到布林中軌下，偏弱整理。" if close>bb_l
             else "股價靠近布林下軌，宜保守。")
    macd  = "MACD偏多，短線動能仍在。" if dif>dea else "MACD轉弱，反彈力道有限。"
    risks = []
    if close>bb_u and vr<1.0: risks.append("上軌突破但量不足，需防假突破。")
    if close>bb_u and vr>=1.5: risks.append("短線乖離偏大，不宜重壓追高。")
    if close<ma10: risks.append("跌破 MA10，短線轉弱。")
    if close<ma20: risks.append("跌破 MA20，中短線保守。")
    if close<ma60: risks.append("跌破季線，波段結構轉差。")
    if not risks: risks.append("目前風險中性，仍需觀察量價。")
    return {
        "action":action, "kline":kline, "boll":boll, "macd":macd, "risks":risks,
        "buy_prices": {"積極買點":pack["aggressive_buy_price"],"回踩買點":pack["pullback_buy_price"],"保守買點":pack["conservative_buy_price"]},
        "sell_prices":{"第一賣點":pack["sell_price_1"],"第二賣點":pack["sell_price_2"]},
        "stop_prices":{"短線停損":pack["stop_loss_short"],"波段停損":pack["stop_loss_wave"]},
    }

def render_analysis(s):
    st.markdown("### 操作建議"); st.write(s["action"])
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**買點**")
        for l,v in s["buy_prices"].items(): st.write(f"- {l}：{v:.2f}")
    with c2:
        st.markdown("**賣點**")
        for l,v in s["sell_prices"].items(): st.write(f"- {l}：{v:.2f}")
    with c3:
        st.markdown("**停損**")
        for l,v in s["stop_prices"].items(): st.write(f"- {l}：{v:.2f}")
    st.markdown("### 趨勢摘要")
    st.write(f"- K線：{s['kline']}")
    st.write(f"- 布林：{s['boll']}")
    st.write(f"- MACD：{s['macd']}")
    st.markdown("### 風險提醒")
    for item in s["risks"]: st.write(f"- {item}")

def ai_block(curr, pack, code, name=""):
    st.markdown("### 🤖 AI 分析（Gemini）")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.warning("❌ GEMINI_API_KEY 未設定，請到 Streamlit Secrets 設定")
        return
    with st.spinner("AI 分析中..."):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""你是專業的台股技術分析師。請根據以下數據，用繁體中文撰寫一份簡潔的個股分析報告（200字以內）。

股票：{name or code}（{code}）
收盤價：{round(safe_float(curr["Close"]),2)}，量比：{round(safe_float(curr["VOL_RATIO"]),2)}
MA5/10/20/60：{round(safe_float(curr["MA5"]),2)} / {round(safe_float(curr["MA10"]),2)} / {round(safe_float(curr["MA20"]),2)} / {round(safe_float(curr["MA60"]),2)}
布林上/中/下：{round(safe_float(curr["BB_UPPER"]),2)} / {round(safe_float(curr["BB_MID"]),2)} / {round(safe_float(curr["BB_LOWER"]),2)}
積極買點：{pack["aggressive_buy_price"]}，回踩買點：{pack["pullback_buy_price"]}
第一賣點：{pack["sell_price_1"]}，停損：{pack["stop_loss_short"]}

請包含：①技術面簡評 ②操作建議 ③主要風險
格式：直接輸出文字，不要標題或編號"""
            response = model.generate_content(prompt)
            st.info(response.text.strip())
        except Exception as e:
            st.error(f"AI 分析失敗：{type(e).__name__}: {e}")

# =========================================================
# 顯示工具
# =========================================================

def rank_df(df):
    if df.empty or "grade" not in df.columns: return df.copy()
    out = df.copy()
    out["grade_rank"] = out["grade"].map({"A1":0,"A2":1}).fillna(9)
    return out.sort_values(["grade_rank","score","vol_ratio"], ascending=[True,False,False]).reset_index(drop=True)

def format_df(df):
    if df.empty: return df
    cols = ["grade","code","name","type","close","score","volume","vol_ratio","rsi14",
            "aggressive_buy_price","pullback_buy_price","conservative_buy_price",
            "sell_price_1","sell_price_2","stop_loss_short","stop_loss_wave","reasons","scan_time"]
    return df[[c for c in cols if c in df.columns]].copy()

def badge(grade):
    return {"A1":"🔴 A1 突破","A2":"🟡 A2 回踩"}.get(grade, grade)

def learning_panel(learning):
    st.subheader("AI 自我學習狀態")
    stats = learning.get("rule_stats",{}); w = learning.get("weights",{}); t = learning.get("thresholds",{})
    c1,c2,c3 = st.columns(3)
    c1.metric("已標記樣本數",  stats.get("total_labeled",0))
    c2.metric("整體5日成功率", f"{stats.get('success_rate_5d',0):.1f}%")
    c3.metric("模型更新時間",  learning.get("updated_at","") or "N/A")
    gs = stats.get("grade_stats",{})
    if gs:
        st.markdown("#### A1 / A2 歷史績效")
        st.dataframe(pd.DataFrame([{"分級":badge(g),"樣本數":d.get("samples",0),
            "5日勝率%":f"{d.get('success_rate_5d',0):.1f}%","平均5日報酬%":f"{d.get('avg_ret_5d',0):.2f}%"}
            for g,d in gs.items()]), hide_index=True)
    with st.expander("查看學習細節", expanded=False):
        if w: st.dataframe(pd.DataFrame([{"條件":k,"權重":v} for k,v in w.items()]), hide_index=True)
        if t: st.dataframe(pd.DataFrame([{"參數":k,"值":v} for k,v in t.items()]), hide_index=True)
        bc = stats.get("by_condition",{})
        if bc: st.dataframe(pd.DataFrame([{"條件":k,"樣本":v["samples"],"5日勝率%":v["success_rate_5d"],"平均報酬%":v["avg_ret_5d"]} for k,v in bc.items()]), hide_index=True)

def set_stock(code, df_show):
    if df_show.empty: return
    m = df_show[df_show["code"].astype(str)==str(code)]
    if m.empty: return
    r = m.iloc[0]
    st.session_state.selected_stock_option = f"{r['code']} {r['name']} ({r['grade']})"
    st.session_state.pending_page_mode = "掃描個股圖表"

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("操作面板")
st.sidebar.checkbox("啟用自動刷新", key="enable_auto_refresh")
st.sidebar.selectbox("自動刷新秒數", [30,60,120,300], key="refresh_sec", format_func=lambda x: f"{x} 秒")
st.sidebar.selectbox("顯示筆數", [20,50,100], key="top_n")
st.sidebar.multiselect("分級篩選", ["A1","A2"], default=["A1","A2"], key="grade_filter")
st.sidebar.selectbox("圖表期間", ["3mo","6mo","9mo","12mo"], key="chart_period")
st.sidebar.markdown("### 價格設定區間")
st.sidebar.number_input("最低價格", min_value=0.0, step=1.0, key="min_price")
st.sidebar.number_input("最高價格", min_value=0.0, step=10.0, key="max_price")
st.sidebar.markdown("### 成交量設定")
st.sidebar.number_input("最低成交量", min_value=0, step=100, key="min_volume")
st.sidebar.number_input("最高成交量", min_value=0, step=100, key="max_volume")
st.sidebar.markdown("### 量比設定")
st.sidebar.number_input("最低量比", min_value=0.0, step=0.1, key="min_vol_ratio")

if st.sidebar.button("🔄 手動重新掃描", use_container_width=True):
    with st.spinner("正在執行掃描，請稍候..."):
        rc, out, err = run_manual_scan()
        st.cache_data.clear()
        if rc == 0:
            st.sidebar.success("掃描完成"); st.rerun()
        else:
            st.sidebar.error("掃描失敗"); st.sidebar.code(err if err else out)

# =========================================================
# 主畫面
# =========================================================

st.title("📈 台股 AI 自動掃描")
latest  = load_latest(); learning = load_learning(); hist_df = load_history()

if st.session_state.enable_auto_refresh:
    auto_refresh_block(st.session_state.refresh_sec)

left, right = st.columns([3,1])
with left:
    st.caption(f"最後掃描時間：{latest.get('updated_at','N/A')}")
    status = latest.get("status","empty"); msg = latest.get("message",""); summ = latest.get("summary",{})
    if status=="ok" and (summ.get("A1_count",0)+summ.get("A2_count",0))>0: st.success("已載入最新掃描結果")
    elif status=="ok": st.info(msg or "掃描完成，今日無符合 A1/A2 條件的個股。")
    elif status=="empty": st.warning(msg or "尚無資料")
    else: st.error(msg or "讀取失敗")
with right:
    st.caption(f"資料檔：{LATEST_JSON}")

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("總掃描池",    summ.get("pool_total",0))
m2.metric("成功分析數",  summ.get("success_count",0))
m3.metric("缺資料/失敗", summ.get("failed_count",0))
m4.metric("A1 突破型",   summ.get("A1_count",0))
m5.metric("A2 回踩型",   summ.get("A2_count",0))

df = rank_df(pd.DataFrame(latest.get("results",[])))
if not df.empty and "grade" in df.columns: df = df[df["grade"].isin(st.session_state.grade_filter)]
if not df.empty:
    if "close"     in df.columns: df = df[(df["close"]>=st.session_state.min_price)&(df["close"]<=st.session_state.max_price)]
    if "volume"    in df.columns: df = df[df["volume"]>=st.session_state.min_volume]
    if st.session_state.max_volume>0 and "volume" in df.columns: df = df[df["volume"]<=st.session_state.max_volume]
    if "vol_ratio" in df.columns: df = df[df["vol_ratio"]>=st.session_state.min_vol_ratio]

df_show = df.head(st.session_state.top_n).copy() if not df.empty else pd.DataFrame()

if st.session_state.pending_page_mode in VALID_PAGES:
    st.session_state.page_mode = st.session_state.pending_page_mode
    st.session_state.pending_page_mode = None

page_mode = st.radio("功能選單", VALID_PAGES, horizontal=True, key="page_mode")

# ── 最新結果 ──────────────────────────────────────────────

if page_mode == "最新結果":
    st.subheader("最新掃描結果")
    if df_show.empty:
        st.info("目前沒有符合條件的 A 級結果。")
    else:
        st.dataframe(format_df(df_show.drop(columns=["grade_rank"],errors="ignore")),
                     use_container_width=True, hide_index=True)
        st.markdown("### Top 10（可直接查看個股圖表）")
        for i, row in df_show.head(10).iterrows():
            c1,c2,c3,c4,c5,c6,c7 = st.columns([0.8,1.0,2.0,1.5,1.5,1.5,1.2])
            c1.write(f"**{badge(row.get('grade','-'))}**")
            c2.write(f"**{row.get('code','')}**")
            c3.write(row.get("name",""))
            c4.write(f"收盤：{safe_float(row.get('close',0)):.2f}")
            c5.write(f"分數：{safe_int(row.get('score',0))}")
            c6.write(f"量比：{safe_float(row.get('vol_ratio',0)):.2f}")
            if c7.button("查看圖表", key=f"t10_{row.get('code',i)}"):
                set_stock(str(row.get("code","")), df_show); st.rerun()
    failed = latest.get("failed_symbols",[])
    if failed:
        with st.expander("查看部分抓不到資料的代號", expanded=False):
            st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)

# ── 單筆個股分析 ──────────────────────────────────────────

elif page_mode == "單筆個股分析":
    st.subheader("單筆個股分析")
    ci, cb = st.columns([3,1])
    with ci:
        code_input = st.text_input("代號", placeholder="例如：2330、6761", label_visibility="collapsed")
    with cb:
        do_go = st.button("🔍 分析", use_container_width=True)

    code = code_input.strip()
    if not code:
        st.info("請輸入股票代號後按「分析」按鈕。")
    elif do_go or code:
        sym, sdf = fetch_by_code(code, st.session_state.chart_period)
        if sdf.empty or sym is None:
            st.warning("找不到此股票資料，請確認代號。")
        else:
            pack = build_trade_prices(sdf)
            curr = sdf.iloc[-1]
            try:
                import twstock as _tw
                _i = _tw.codes.get(code, None)
                name = getattr(_i,"name",code) if _i else code
            except Exception:
                name = code

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("股票", f"{name}（{code}）")
            c2.metric("收盤價", round(safe_float(curr["Close"]),2))
            c3.metric("成交量", int(safe_float(curr["Volume"],0)))
            c4.metric("量比",   round(safe_float(curr["VOL_RATIO"],0),2))
            c5.metric("來源",   sym)

            render_analysis(build_summary(sdf, pack))
            ai_block(curr, pack, code, name)
            st.plotly_chart(make_professional_chart(sdf, pack, f"{code} {name}"),
                            use_container_width=True)

# ── 掃描個股圖表 ──────────────────────────────────────────

elif page_mode == "掃描個股圖表":
    st.subheader("掃描個股圖表")
    if df_show.empty:
        st.info("目前沒有可選股票。")
    else:
        opts = [f"{r['code']} {r['name']} ({r['grade']})" for _, r in df_show.iterrows()]
        if st.session_state.selected_stock_option not in opts:
            st.session_state.selected_stock_option = opts[0]

        sel  = st.selectbox("選擇掃描結果中的股票", opts, key="selected_stock_option")
        row  = df_show.iloc[opts.index(sel)]

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("分級",   badge(row.get("grade","-")))
        c2.metric("收盤價", row.get("close",""))
        c3.metric("加分",   row.get("score",""))
        c4.metric("成交量", row.get("volume",""))
        c5.metric("量比",   row.get("vol_ratio",""))
        c6.metric("RSI14",  row.get("rsi14",""))

        st.info(f"觸發條件：{row.get('reasons','-')}")

        ai_txt = row.get("ai_analysis","")
        if ai_txt:
            st.markdown("### 🤖 Claude AI 分析"); st.info(ai_txt)
        else:
            st.caption("AI 分析：本次掃描未產生（需設定 ANTHROPIC_API_KEY 並重新掃描）")

        cdf = fetch_chart(row.get("symbol",""), period=st.session_state.chart_period)
        if cdf.empty:
            st.warning("無法讀取圖表資料。")
        else:
            pack = {k: safe_float(row.get(k)) for k in [
                "aggressive_buy_price","pullback_buy_price","conservative_buy_price",
                "sell_price_1","sell_price_2","stop_loss_short","stop_loss_wave",
                "stop_loss_hard","support_1","support_2","support_3","resistance_1","resistance_2"]}
            render_analysis(build_summary(cdf, pack))
            st.plotly_chart(make_professional_chart(cdf, pack, f"{row['code']} {row.get('name','')}"),
                            use_container_width=True)

# ── AI 學習 ───────────────────────────────────────────────

elif page_mode == "AI學習":
    learning_panel(learning)

# ── 歷史紀錄 ──────────────────────────────────────────────

elif page_mode == "歷史紀錄":
    st.subheader("歷史紀錄")
    if hist_df.empty:
        st.info("目前尚無歷史紀錄。")
    else:
        h = hist_df.copy()
        if "grade"     in h.columns: h = h[h["grade"].isin(st.session_state.grade_filter)]
        if "close"     in h.columns: h = h[(h["close"]>=st.session_state.min_price)&(h["close"]<=st.session_state.max_price)]
        if "volume"    in h.columns: h = h[h["volume"]>=st.session_state.min_volume]
        if st.session_state.max_volume>0 and "volume" in h.columns: h = h[h["volume"]<=st.session_state.max_volume]
        if "vol_ratio" in h.columns: h = h[h["vol_ratio"]>=st.session_state.min_vol_ratio]
        hc = ["scan_date","code","name","grade","type","close","volume","vol_ratio","score",
              "aggressive_buy_price","pullback_buy_price","sell_price_1","stop_loss_short",
              "ret_3d","ret_5d","ret_10d","success_5d"]
        st.dataframe(h[[c for c in hc if c in h.columns]].tail(500),
                     use_container_width=True, hide_index=True)
