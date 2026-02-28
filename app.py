# app.py
# V8.1（可視化 + 單檔決策面板清楚版）
# ✅ V7：自動掃描 20 檔（80+ 主榜 / 60~79 觀察 / 逐檔診斷 / 市場風險）
# ✅ V8：單一股票「當下買/賣/等」+ 預估未來買賣點 + 布林通道分析 + 圖表
#
# ⚠️ 僅做資訊顯示與風控演算，不構成投資建議、不自動下單。

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant - V8.1", layout="wide")
st.title("⚡ AI Stock Trading Assistant（V8.1：雷達 + 單股決策面板）")

DEFAULT_WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]

LOOKBACK_DAYS_RADAR = 220
LOOKBACK_DAYS_SINGLE = 365
CACHE_TTL = 600

SCORE_STRONG = 80
SCORE_WATCH = 60

# -----------------------------
# Sidebar：永遠顯示「單股分析」
# -----------------------------
st.sidebar.header("🎯 單一股票分析（盤後）")
single_code = st.sidebar.text_input("股票代號", value="2330")
single_lookback = st.sidebar.selectbox("回溯天數", [180, 365, 730], index=1)
single_run = st.sidebar.button("分析單股", type="primary")

st.sidebar.divider()
st.sidebar.header("⚡ 雷達設定")
watchlist_text = st.sidebar.text_area(
    "掃描股票池（逗號或換行分隔）",
    value="\n".join(DEFAULT_WATCHLIST),
    height=180
)
score_strong = st.sidebar.number_input("主榜門檻（80+）", value=SCORE_STRONG, min_value=60, max_value=100, step=1)
score_watch = st.sidebar.number_input("觀察門檻（60~79）", value=SCORE_WATCH, min_value=0, max_value=int(score_strong), step=1)

WATCHLIST = []
for token in watchlist_text.replace(",", "\n").splitlines():
    t = token.strip().upper()
    if t:
        WATCHLIST.append(t)
if not WATCHLIST:
    WATCHLIST = DEFAULT_WATCHLIST.copy()

# -----------------------------
# Helpers
# -----------------------------
def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()

    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = _safe_float_series(df[c])

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    return df.sort_values("Date").reset_index(drop=True)

def _make_yf_candidates(code: str) -> list[str]:
    code = (code or "").strip().upper()
    if not code:
        return []
    if "." in code:
        return [code]
    if code.isdigit():
        return [f"{code}.TW", f"{code}.TWO", code]
    return [code]

# -----------------------------
# Data loaders (FinMind → YF)
# -----------------------------
@st.cache_data(ttl=CACHE_TTL)
def load_from_finmind(code_digits: str, start: str, end: str) -> pd.DataFrame:
    try:
        from FinMind.data import DataLoader
    except Exception:
        return pd.DataFrame()

    try:
        dl = DataLoader()
        raw = dl.taiwan_stock_daily(stock_id=code_digits, start_date=start, end_date=end)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Date": pd.to_datetime(raw["date"], errors="coerce"),
            "Open": raw["open"],
            "High": raw.get("max", np.nan),
            "Low": raw.get("min", np.nan),
            "Close": raw["close"],
            "Volume": raw.get("Trading_Volume", 0),
        })
        return _ensure_ohlcv(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=False,
            group_by="column", threads=False
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _flatten_yf_columns(df)
    return _ensure_ohlcv(df)

def load_data_best_effort(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    code = (code or "").strip().upper()
    if not code:
        return pd.DataFrame(), ""

    if code.isdigit():
        df_fm = load_from_finmind(code, start, end)
        if not df_fm.empty:
            return df_fm, "FinMind"

    for t in _make_yf_candidates(code):
        df_yf = load_from_yfinance(t, start, end)
        if not df_yf.empty:
            return df_yf, f"YF({t})"

    return pd.DataFrame(), ""

# -----------------------------
# Indicators + Bollinger
# -----------------------------
def add_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    if df_ohlcv is None or df_ohlcv.empty:
        return pd.DataFrame()
    df = df_ohlcv.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = _safe_float_series(df[c])
    df = df.dropna(subset=["Open","High","Low","Close"]).reset_index(drop=True)
    if len(df) < 80:
        return pd.DataFrame()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    try:
        df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA60"] = EMAIndicator(close, 60).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()
        df["MACD_H"] = MACD(close).macd_diff()
        df["ATR"] = AverageTrueRange(high, low, close, 14).average_true_range()
    except Exception:
        return pd.DataFrame()

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    return df.dropna().reset_index(drop=True)

# -----------------------------
# Radar scoring (breakout short swing)
# -----------------------------
def breakout_score(df_ohlcv: pd.DataFrame) -> tuple[int, dict | None]:
    df = add_indicators(df_ohlcv)
    if df.empty:
        return 0, None

    latest = df.iloc[-1]
    prev5_high = df["High"].iloc[-6:-1].max()
    vol_mean = df["Volume"].iloc[-6:-1].mean()

    score = 0
    if latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA60"]:
        score += 20
    if latest["Close"] > prev5_high:
        score += 20
    if vol_mean > 0 and latest["Volume"] > 1.5 * vol_mean:
        score += 20
    if latest["RSI"] > 55:
        score += 20
    if latest["MACD_H"] > 0:
        score += 20

    entry = float(latest["Close"])
    atr = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else np.nan
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return score, None

    stop = entry - 1.2 * atr
    if not np.isfinite(stop) or entry <= stop:
        return score, None

    target = entry + (entry - stop) * 2.2
    rr = (target - entry) / (entry - stop)

    return score, {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(float(rr), 2),
    }

def market_risk_label(start: str, end: str) -> str:
    df, src = load_data_best_effort("0050", start, end)
    df_i = add_indicators(df)
    if df_i.empty:
        return "🟡 市場資料不足"
    if df_i["Close"].iloc[-1] > df_i["EMA20"].iloc[-1]:
        return f"🟢 市場可積極操作（來源：{src}）"
    return f"🔴 市場偏弱，控制倉位（來源：{src}）"

# -----------------------------
# Single stock decision engine (clear output)
# -----------------------------
def single_decision(df_i: pd.DataFrame) -> dict:
    latest = df_i.iloc[-1]
    prev5_high = df_i["High"].iloc[-6:-1].max()
    vol_mean = df_i["Volume"].iloc[-6:-1].mean()

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema60 = float(latest["EMA60"])
    rsi = float(latest["RSI"])
    macd_h = float(latest["MACD_H"])
    atr = float(latest["ATR"])
    bb_up = float(latest["BB_UP"])
    bb_mid = float(latest["BB_MID"])
    bb_low = float(latest["BB_LOW"])

    # Boll width percentile
    widths = ((df_i["BB_UP"] - df_i["BB_LOW"]) / df_i["BB_MID"]).replace([np.inf, -np.inf], np.nan).dropna()
    width = (bb_up - bb_low) / bb_mid if bb_mid else np.nan
    if len(widths) >= 60 and np.isfinite(width):
        q20 = float(widths.quantile(0.2))
        q80 = float(widths.quantile(0.8))
        if width <= q20:
            boll_state = "壓縮（可能醞釀爆發）"
        elif width >= q80:
            boll_state = "擴張（趨勢段/波動放大）"
        else:
            boll_state = "正常"
    else:
        boll_state = "不足以判斷"

    # Boll position
    if price >= bb_up:
        boll_pos = "上軌附近（偏過熱/動能強）"
    elif price <= bb_low:
        boll_pos = "下軌附近（偏超跌/反彈區）"
    elif price >= bb_mid:
        boll_pos = "中軌之上（偏多）"
    else:
        boll_pos = "中軌之下（偏弱）"

    # Forecast points
    pullback_buy = (round(ema20 * 0.99, 2), round(ema20 * 1.01, 2))
    breakout_trigger = round(float(prev5_high) + 0.2 * atr, 2)
    stop = round(price - 1.2 * atr, 2)
    target_rr = round(price + (price - stop) * 2.2, 2)
    target_bb = round(bb_up, 2)

    # Rules (盤後)
    trend_ok = (price > ema20 and ema20 > ema60)
    breakout_ok = (price > prev5_high)
    vol_ok = (vol_mean > 0 and float(latest["Volume"]) > 1.3 * vol_mean)
    macd_ok = (macd_h > 0)
    rsi_ok = (rsi > 55)

    reasons = []
    if trend_ok: reasons.append("趨勢：收盤在 EMA20 上、且 EMA20 > EMA60")
    else: reasons.append("趨勢：不符合強多（未站穩 EMA20 或 EMA20 未高於 EMA60）")

    if breakout_ok: reasons.append("突破：收盤突破前 5 日高")
    else: reasons.append("突破：尚未突破前 5 日高")

    if vol_ok: reasons.append("量能：放大（> 1.3×5日均量）")
    else: reasons.append("量能：未明顯放大")

    if macd_ok: reasons.append("動能：MACD 柱翻正")
    else: reasons.append("動能：MACD 柱偏弱/未翻正")

    if rsi_ok: reasons.append("強度：RSI > 55")
    else: reasons.append("強度：RSI 未達 55")

    # Decision (BIG, clear)
    if trend_ok and breakout_ok and rsi_ok and (vol_ok or macd_ok):
        decision = "BUY"
        decision_text = "🟢 買點（盤後確認突破）"
    elif (price < ema20) and (macd_h < 0):
        decision = "SELL"
        decision_text = "🔴 賣點/減碼（跌破 EMA20 且動能轉弱）"
    else:
        decision = "WAIT"
        decision_text = "🟡 觀察（等待回踩買區或再突破確認）"

    return {
        "price": round(price, 2),
        "decision": decision,
        "decision_text": decision_text,
        "reasons": reasons[:5],
        "boll_pos": boll_pos,
        "boll_state": boll_state,
        "pullback_buy": pullback_buy,
        "breakout_trigger": breakout_trigger,
        "stop": stop,
        "target_rr": target_rr,
        "target_bb": target_bb,
        "ema20": round(ema20, 2),
        "ema60": round(ema60, 2),
        "rsi": round(rsi, 1),
        "macd_h": round(macd_h, 4),
        "atr": round(atr, 2),
    }

def plot_single_chart(df_i: pd.DataFrame, info: dict, title: str):
    plot_df = df_i.tail(160).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df["Date"], open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA60"], mode="lines", name="EMA60"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_UP"], mode="lines", name="BB_UP"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_MID"], mode="lines", name="BB_MID"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_LOW"], mode="lines", name="BB_LOW"))

    fig.add_hline(y=info["price"], line_dash="dot", annotation_text="Close", annotation_position="top left")
    fig.add_hline(y=info["stop"], line_dash="dot", annotation_text="Stop", annotation_position="bottom left")
    fig.add_hline(y=info["target_rr"], line_dash="dot", annotation_text="Target(RR)", annotation_position="top right")

    fig.update_layout(title=title, height=560, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Layout：Tabs（更好觀察）
# -----------------------------
tab1, tab2 = st.tabs(["⚡ 市場雷達（自動掃描）", "🎯 單股決策（清楚買/賣/等 + 預估點位）"])

# -----------------------------
# Tab 1：Radar
# -----------------------------
with tab1:
    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS_RADAR)).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    st.caption(f"掃描範圍：{start} → {today.strftime('%Y-%m-%d')} | 股票池：{len(WATCHLIST)} 檔 | 快取：{CACHE_TTL}s")

    progress = st.progress(0)
    status = st.empty()

    results = []
    diag_rows = []

    for i, code in enumerate(WATCHLIST):
        status.write(f"掃描中：{code} ({i+1}/{len(WATCHLIST)})")
        df, src = load_data_best_effort(code, start, end)

        if df.empty:
            diag_rows.append({"股票": code, "來源": src or "-", "狀態": "EMPTY", "備註": "無資料"})
            progress.progress(int((i + 1) / len(WATCHLIST) * 100))
            continue

        score, trade = breakout_score(df)
        if trade is None:
            diag_rows.append({"股票": code, "來源": src, "狀態": "SKIP", "備註": f"指標/資料不足（score={score}）"})
            progress.progress(int((i + 1) / len(WATCHLIST) * 100))
            continue

        diag_rows.append({"股票": code, "來源": src, "狀態": "OK", "備註": f"score={score}"})

        if score >= int(score_watch):
            results.append({
                "股票": code,
                "分數": int(score),
                "來源": src,
                "進場": trade["entry"],
                "停損": trade["stop"],
                "目標": trade["target"],
                "RR": trade["rr"],
            })

        progress.progress(int((i + 1) / len(WATCHLIST) * 100))

    progress.empty()
    status.empty()

    df_results = (pd.DataFrame(results).sort_values("分數", ascending=False)
                  if results else pd.DataFrame(columns=["股票","分數","來源","進場","停損","目標","RR"]))

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"⚡ 今日突破榜（{score_strong}+）")
        strong = df_results[df_results["分數"] >= int(score_strong)].copy()
        st.dataframe(strong if not strong.empty else df_results.head(0), use_container_width=True)
        if strong.empty:
            st.info("今日沒有達到主榜門檻的突破型機會。")

        st.subheader(f"🟡 次強觀察區（{score_watch}~{int(score_strong)-1}）")
        watch = df_results[(df_results["分數"] >= int(score_watch)) & (df_results["分數"] < int(score_strong))].copy()
        st.dataframe(watch if not watch.empty else df_results.head(0), use_container_width=True)

    with right:
        st.subheader("📊 市場風險燈號")
        st.write(market_risk_label(start, end))

        st.subheader("💰 資金控管（以主榜 Top1 計算）")
        capital = st.number_input("總資金", value=5_000_000, step=100_000)
        risk_pct = st.number_input("單筆風險 %", value=2.0, step=0.5)

        if not strong.empty:
            top1 = strong.iloc[0]
            risk_amount = float(capital) * (float(risk_pct) / 100.0)
            loss_per_share = float(top1["進場"]) - float(top1["停損"])

            if loss_per_share > 0:
                shares = int(risk_amount / loss_per_share)
                lots = max(int(shares // 1000), 0)
                st.write(f"建議標的：**{top1['股票']}**（分數 {top1['分數']}）")
                st.write(f"單筆風險金額：**{risk_amount:,.0f}**")
                st.write(f"停損距離：**{loss_per_share:.2f}**")
                st.write(f"可買：**{shares:,} 股 / 約 {lots} 張**")
            else:
                st.warning("停損距離異常，無法計算張數。")
        else:
            st.caption("主榜沒有標的時，不計算張數（避免硬做）。")

    with st.expander("🧩 逐檔診斷（哪一檔被跳過、原因）", expanded=False):
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

# -----------------------------
# Tab 2：Single stock decision（清楚判斷）
# -----------------------------
with tab2:
    st.subheader("🎯 單股決策（盤後）— 一眼判斷買/賣/等 + 預估未來買賣點")

    # 允許從雷達結果下拉直接選（更快）
    colx, coly = st.columns([1, 2])
    with colx:
        pick = st.selectbox("快速選股（從股票池）", options=WATCHLIST, index=0)
    with coly:
        st.caption("你也可以用左側 Sidebar 直接輸入代號按「分析單股」。")

    code_to_use = single_code.strip().upper() if single_run else pick

    today = dt.date.today()
    start = (today - dt.timedelta(days=int(single_lookback))).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    df_raw, src = load_data_best_effort(code_to_use, start, end)
    if df_raw.empty:
        st.error("❌ 無法取得資料（FinMind / YF 都失敗）。請換代號或稍後再試。")
        st.stop()

    df_i = add_indicators(df_raw)
    if df_i.empty:
        st.error("❌ 指標計算失敗或資料不足（NaN 過多）。請把回溯天數改 365/730 再試。")
        st.stop()

    info = single_decision(df_i)

    # 大字卡片：BUY / SELL / WAIT
    st.markdown("## ✅ 當下結論（盤後）")
    if info["decision"] == "BUY":
        st.success(info["decision_text"])
    elif info["decision"] == "SELL":
        st.error(info["decision_text"])
    else:
        st.warning(info["decision_text"])

    # 指標小卡片
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Close", f'{info["price"]}')
    m2.metric("EMA20 / EMA60", f'{info["ema20"]} / {info["ema60"]}')
    m3.metric("RSI", f'{info["rsi"]}')
    m4.metric("MACD_H", f'{info["macd_h"]}')
    m5.metric("ATR", f'{info["atr"]}')

    # 原因 + 布林
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### 🧠 判斷原因（最重要）")
        for r in info["reasons"]:
            st.write("• ", r)

        st.markdown("### 📉 布林通道分析")
        st.write(f"• 位置：**{info['boll_pos']}**")
        st.write(f"• 狀態：**{info['boll_state']}**")

    with right:
        st.markdown("### 🔮 預估未來買賣點（你要的清楚點位）")
        st.write(f"**預估回踩買區（EMA20±1%）**：`{info['pullback_buy'][0]} ~ {info['pullback_buy'][1]}`")
        st.write(f"**預估突破買點（前5高+0.2ATR）**：`{info['breakout_trigger']}`")
        st.write(f"**預估停損（ATR×1.2）**：`{info['stop']}`")
        st.write(f"**預估賣點（目標 RR 2.2）**：`{info['target_rr']}`")
        st.write(f"**布林上軌賣點參考**：`{info['target_bb']}`")

    st.markdown("### 📈 K線 + 布林 + EMA（含停損/目標線）")
    plot_single_chart(df_i, info, title=f"{code_to_use} | 來源：{src}")

st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")