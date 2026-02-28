# app.py
# V8 雙引擎：V7 市場雷達 + V8 單檔決策引擎（盤後型）
# ✅ FinMind 優先 + yfinance fallback
# ✅ MultiIndex 修正 / 資料清洗 / 每檔 try-continue（不再整頁爆）
# ✅ V7：80+ 主榜 / 60~79 觀察 / 市場風險燈號 / 資金控管 / 逐檔診斷
# ✅ V8：單檔當下買賣點判斷 + 預估未來買賣點 + 布林通道分析 + 圖表
#
# ⚠️ 僅做資訊顯示與風險控管演算，不構成投資建議，也不會自動下單。

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
st.set_page_config(page_title="AI Stock Trading Assistant - V8", layout="wide")
st.title("⚡ AI Stock Trading Assistant（V8：市場雷達 + 單檔決策引擎）")

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_WATCHLIST = [
    "2330","2317","2303","2454","3661","3037","2382","2376",
    "3017","3443","2603","2615","1301","1326","2882","2881",
    "2891","0050","6274","2383"
]

LOOKBACK_DAYS_RADAR = 220     # 雷達抓 220 天（足夠算 60/80 天指標）
LOOKBACK_DAYS_SINGLE = 365    # 單檔分析抓 365 天（視覺化更完整）
CACHE_TTL_SEC_DEFAULT = 600   # 10 分鐘快取
SCORE_STRONG_DEFAULT = 80
SCORE_WATCH_DEFAULT = 60

# -----------------------------
# Mode switch (分頁)
# -----------------------------
mode = st.radio(
    "模式選擇",
    ["⚡ 市場雷達（V7）", "🎯 單檔決策引擎（V8 / 盤後型）"],
    horizontal=True
)

# -----------------------------
# Helpers
# -----------------------------
def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance 偶爾回 MultiIndex：('Close','2330.TW')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: Date, Open, High, Low, Close, Volume (clean)."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # reset index to Date
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()

    # normalize column names
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

    df = df.sort_values("Date").reset_index(drop=True)
    return df

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
# Cache TTL (可調)
# -----------------------------
if mode == "⚡ 市場雷達（V7）":
    with st.expander("⚙️ 設定（股票池 / 快取 / 門檻）", expanded=False):
        watchlist_text = st.text_area(
            "掃描股票池（逗號或換行分隔，預設 20 檔）",
            value="\n".join(DEFAULT_WATCHLIST),
            height=160
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            score_strong = st.number_input("強突破門檻（主榜）", value=int(SCORE_STRONG_DEFAULT), min_value=60, max_value=100, step=1)
        with c2:
            score_watch = st.number_input("觀察門檻（次強）", value=int(SCORE_WATCH_DEFAULT), min_value=0, max_value=int(score_strong), step=1)
        with c3:
            cache_ttl = st.number_input("快取秒數（Cloud 建議 600）", value=int(CACHE_TTL_SEC_DEFAULT), min_value=60, max_value=3600, step=60)

    WATCHLIST = []
    for token in watchlist_text.replace(",", "\n").splitlines():
        t = token.strip().upper()
        if t:
            WATCHLIST.append(t)
    if len(WATCHLIST) == 0:
        WATCHLIST = DEFAULT_WATCHLIST.copy()

else:
    # 單檔頁面固定用預設快取
    cache_ttl = CACHE_TTL_SEC_DEFAULT
    score_strong = SCORE_STRONG_DEFAULT
    score_watch = SCORE_WATCH_DEFAULT
    WATCHLIST = DEFAULT_WATCHLIST.copy()

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data(ttl=cache_ttl)
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
            "High": raw.get("max", raw.get("high", np.nan)),
            "Low": raw.get("min", raw.get("low", np.nan)),
            "Close": raw["close"],
            "Volume": raw.get("Trading_Volume", raw.get("volume", 0)),
        })
        return _ensure_ohlcv(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=cache_ttl)
def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df)
    df = _ensure_ohlcv(df)
    return df

def load_data_best_effort(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, source_used)
    source_used: 'FinMind' or 'YF(...)' or ''
    """
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
# Indicators & scoring
# -----------------------------
def add_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add EMA20/EMA60/RSI/MACD_H/ATR + Bollinger bands."""
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

    # Bollinger (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    df = df.dropna().reset_index(drop=True)
    return df

def breakout_score(df_ohlcv: pd.DataFrame) -> tuple[int, dict | None]:
    """
    Score:
      +20 趨勢：Close>EMA20 且 EMA20>EMA60
      +20 突破：Close > 前 5 日高
      +20 量能：Volume > 1.5 * 5日均量
      +20 RSI：RSI > 55
      +20 MACD柱：MACD_H > 0
    Trade:
      entry = Close
      stop  = entry - ATR*1.2
      target= entry + (entry-stop)*2.2
    """
    df = add_indicators(df_ohlcv)
    if df.empty or len(df) < 80:
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
    atr_v = float(latest["ATR"]) if np.isfinite(latest["ATR"]) else np.nan
    if not np.isfinite(entry) or not np.isfinite(atr_v) or atr_v <= 0:
        return score, None

    stop = float(entry - atr_v * 1.2)
    if not np.isfinite(stop) or entry <= stop:
        return score, None

    target = float(entry + (entry - stop) * 2.2)
    rr = (target - entry) / (entry - stop)

    trade = {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(float(rr), 2),
    }
    return score, trade

def market_risk_label(start: str, end: str) -> str:
    df, src = load_data_best_effort("0050", start, end)
    if df.empty:
        return "🟡 市場資料不足（0050 無法取得）"

    df_i = add_indicators(df)
    if df_i.empty:
        return "🟡 市場資料不足（指標失敗）"

    if df_i["Close"].iloc[-1] > df_i["EMA20"].iloc[-1]:
        return f"🟢 市場可積極操作（來源：{src}）"
    return f"🔴 市場偏弱，控制倉位（來源：{src}）"

# -----------------------------
# V7 Radar page
# -----------------------------
if mode == "⚡ 市場雷達（V7）":
    today = dt.date.today()
    end_dt = today + dt.timedelta(days=1)
    start_dt = today - dt.timedelta(days=LOOKBACK_DAYS_RADAR)
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    st.caption(f"掃描範圍：{start} → {today.strftime('%Y-%m-%d')} | 股票池：{len(WATCHLIST)} 檔 | 快取：{cache_ttl}s")

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

        # 每檔獨立保護：算不了就跳過，不拖垮整頁
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

    df_results = (
        pd.DataFrame(results).sort_values("分數", ascending=False)
        if results else
        pd.DataFrame(columns=["股票","分數","來源","進場","停損","目標","RR"])
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(f"⚡ 今日突破榜（{score_strong}+）")
        strong = df_results[df_results["分數"] >= int(score_strong)].copy()
        if strong.empty:
            st.info("今日沒有達到主榜門檻的突破型機會。")
        else:
            st.dataframe(strong, use_container_width=True)

        st.subheader(f"🟡 次強觀察區（{score_watch}~{int(score_strong)-1}）")
        watch = df_results[(df_results["分數"] >= int(score_watch)) & (df_results["分數"] < int(score_strong))].copy()
        if watch.empty:
            st.caption("無次強觀察標的。")
        else:
            st.dataframe(watch, use_container_width=True)

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

            if loss_per_share <= 0:
                st.warning("Top1 停損距離異常，無法計算張數。")
            else:
                shares = int(risk_amount / loss_per_share)
                lots = max(int(shares // 1000), 0)  # 台股 1 張 = 1000 股（簡化）
                st.write(f"建議標的：**{top1['股票']}**（分數 {top1['分數']}，來源 {top1['來源']}）")
                st.write(f"單筆風險金額：**{risk_amount:,.0f}**")
                st.write(f"停損距離：**{loss_per_share:.2f}**")
                st.write(f"可買股數：約 **{shares:,} 股**")
                st.write(f"可買張數：約 **{lots} 張**")
                st.caption("（張數以 1000 股/張簡化估算；實務請以券商規則與可交易股數為準）")
        else:
            st.caption("主榜沒有標的時，不計算張數（避免硬做）。")

    with st.expander("🧩 逐檔診斷（哪一檔被跳過、原因）", expanded=False):
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

    st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------
# V8 Single stock decision engine (盤後)
# -----------------------------
else:
    st.subheader("🎯 單檔決策引擎（盤後型 / 收盤判斷）")

    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        code = st.text_input("輸入股票代號（例：2330 / 6274）", value="2330")
    with colB:
        lookback = st.selectbox("回溯天數", [180, 365, 730], index=1)
    with colC:
        run_single = st.button("開始分析", type="primary")

    if run_single:
        today = dt.date.today()
        end_dt = today + dt.timedelta(days=1)
        start_dt = today - dt.timedelta(days=int(lookback))
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

        df_raw, src = load_data_best_effort(code, start, end)
        if df_raw.empty:
            st.error("❌ 無法取得資料（FinMind / YF 均失敗）。請換代號或稍後再試。")
            st.stop()

        df = add_indicators(df_raw)
        if df.empty:
            st.error("❌ 資料不足或指標計算失敗（可能 NaN 過多）。請改回溯天數更長或換代號。")
            st.stop()

        latest = df.iloc[-1]
        prev5_high = df["High"].iloc[-6:-1].max()
        vol_mean = df["Volume"].iloc[-6:-1].mean()

        # ---------- Trend ----------
        if latest["Close"] > latest["EMA20"] and latest["EMA20"] > latest["EMA60"]:
            trend = "🟢 多頭延續"
        elif latest["Close"] < latest["EMA20"] and latest["EMA20"] < latest["EMA60"]:
            trend = "🔴 空頭趨勢"
        else:
            trend = "🟡 震盪整理"

        # ---------- Bollinger analysis ----------
        bb_mid = float(latest["BB_MID"])
        bb_up = float(latest["BB_UP"])
        bb_low = float(latest["BB_LOW"])
        price = float(latest["Close"])

        # 位置
        if price >= bb_up:
            boll_pos = "上軌附近（偏過熱/動能強）"
        elif price <= bb_low:
            boll_pos = "下軌附近（偏超跌/反彈區）"
        else:
            # 中間再細分
            if price >= bb_mid:
                boll_pos = "中軌之上（偏多）"
            else:
                boll_pos = "中軌之下（偏弱）"

        # 通道寬度 / 壓縮
        width = (bb_up - bb_low) / bb_mid if bb_mid != 0 else np.nan
        # 以近 120 天寬度分位做判斷（盤後更穩）
        widths = ((df["BB_UP"] - df["BB_LOW"]) / df["BB_MID"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(widths) >= 60 and np.isfinite(width):
            q20 = float(widths.quantile(0.2))
            q80 = float(widths.quantile(0.8))
            if width <= q20:
                boll_state = "通道壓縮（可能醞釀爆發）"
            elif width >= q80:
                boll_state = "通道擴張（波動放大/趨勢段）"
            else:
                boll_state = "通道正常"
        else:
            boll_state = "通道狀態不足以判斷"

        # ---------- Momentum ----------
        rsi = float(latest["RSI"])
        macd_h = float(latest["MACD_H"])
        vol_ok = (vol_mean > 0 and float(latest["Volume"]) > 1.3 * vol_mean)

        if rsi >= 70:
            momentum = "🟡 過熱（需防回檔）"
        elif rsi <= 35:
            momentum = "🟡 過冷（反彈區）"
        else:
            momentum = "🟢 動能正常"

        # ---------- Now Buy/Sell signal (盤後) ----------
        breakout_ok = (price > prev5_high)
        trend_ok = (price > float(latest["EMA20"]) and float(latest["EMA20"]) > float(latest["EMA60"]))
        macd_ok = (macd_h > 0)
        rsi_ok = (rsi > 55)

        # 當下買點：突破型（盤後確認）
        # 條件：趨勢OK + 突破 + (量能或MACD) + RSI OK
        if trend_ok and breakout_ok and rsi_ok and (vol_ok or macd_ok):
            now_action = "🟢 當下偏向『買點』（突破盤後確認）"
        # 當下賣點：跌破 EMA20 且 MACD柱轉負（轉弱盤後確認）
        elif price < float(latest["EMA20"]) and macd_h < 0:
            now_action = "🔴 當下偏向『賣點 / 減碼』（跌破關鍵均線＋動能轉弱）"
        else:
            now_action = "🟡 當下以『等待/觀察』為主"

        # ---------- Forecast future buy/sell points ----------
        atr = float(latest["ATR"])
        if not np.isfinite(atr) or atr <= 0:
            atr = float(df["ATR"].iloc[-10:].median())

        # 1) 回踩買區：EMA20 ±1%
        ema20 = float(latest["EMA20"])
        pullback_buy_low = round(ema20 * 0.99, 2)
        pullback_buy_high = round(ema20 * 1.01, 2)

        # 2) 突破觸發價：前5高 + 0.2*ATR（避免假突破）
        breakout_trigger = round(float(prev5_high) + 0.2 * atr, 2)

        # 3) 停損：ATR*1.2
        stop = round(price - 1.2 * atr, 2)

        # 4) 目標：RR 2.2（或上軌附近）
        target_rr = round(price + (price - stop) * 2.2, 2)
        # 5) 布林上軌作為短線目標參考
        target_bb = round(bb_up, 2)

        # ---------- Display summary ----------
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("### 📌 當下結構判斷（盤後）")
            st.write(f"**代號**：{code}　|　**資料來源**：{src}")
            st.write(f"**趨勢**：{trend}")
            st.write(f"**動能**：{momentum}（RSI={rsi:.1f} / MACD_H={macd_h:.3f}）")
            st.write(f"**布林位置**：{boll_pos}")
            st.write(f"**布林狀態**：{boll_state}")
            st.markdown("### ✅ 當下是否買/賣")
            st.write(f"**判斷**：{now_action}")

            st.markdown("### 🔮 預估未來買點 / 賣點（盤後推估）")
            st.write(f"**預估回踩買區（EMA20±1%）**：{pullback_buy_low} ~ {pullback_buy_high}")
            st.write(f"**預估突破觸發價（前5高+0.2ATR）**：{breakout_trigger}")
            st.write(f"**預估停損（ATR×1.2）**：{stop}")
            st.write(f"**預估目標（RR 2.2）**：{target_rr}")
            st.write(f"**布林上軌參考目標**：{target_bb}")
            st.caption("（預估為模型推算，實務仍需配合隔日開盤/成交量確認）")

        with c2:
            st.markdown("### 📈 K 線 + 布林 + EMA（盤後）")
            plot_df = df.tail(160).copy()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=plot_df["Date"],
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="Price"
            ))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA20"], mode="lines", name="EMA20"))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["EMA60"], mode="lines", name="EMA60"))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_UP"], mode="lines", name="BB_UP"))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_MID"], mode="lines", name="BB_MID"))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_LOW"], mode="lines", name="BB_LOW"))

            # Horizontal lines: entry/stop/target
            fig.add_hline(y=price, line_dash="dot", annotation_text="Close(Entry)", annotation_position="top left")
            fig.add_hline(y=stop, line_dash="dot", annotation_text="Stop", annotation_position="bottom left")
            fig.add_hline(y=target_rr, line_dash="dot", annotation_text="Target(RR)", annotation_position="top right")

            fig.update_layout(height=560, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # ---------- Extra: quick table ----------
        with st.expander("📋 最新數據（最後 30 筆）", expanded=False):
            st.dataframe(df.tail(30), use_container_width=True)

    st.caption(f"runtime: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")