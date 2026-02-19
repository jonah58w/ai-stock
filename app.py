# app.py
# AI Stock Trading Assistant（台股分析專業版 / 不自動下單）
# ✅ 台股代號：2330 / 2317 / 0050（自動加 .TW）
# ✅ 資料下載：yfinance → 失敗自動改用 Stooq（pandas-datareader）
# ✅ 指標：SMA/EMA/RSI/Bollinger/ATR
# ✅ 買賣點：只在 EMA20/SMA20「交叉那一根」出現 + 冷卻期 3 根
# ✅ 停損停利：ATR StopLoss + RR TakeProfit
# ✅ 新增：未來觸發價位預估（從目前點看未來）
# ✅ 回測：只用交叉訊號進出 + 冷卻期 + ATR 停損停利
# ⚠️ 僅做資訊顯示，不含自動下單

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# -----------------------------
# Helpers
# -----------------------------
def to_tw_symbol(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    if code.upper().endswith(".TW"):
        return code.upper()
    return f"{code}.TW"

def safe_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError("Expected single column series, got DataFrame with multiple columns.")
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return pd.Series(arr)

@st.cache_data(show_spinner=False, ttl=60)
def download_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    # 1) yfinance
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(how="all")
            needed = {"Open", "High", "Low", "Close"}
            if needed.issubset(set(df.columns)):
                if "Volume" not in df.columns:
                    df["Volume"] = np.nan
                return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print("yfinance failed:", repr(e))
    
    # 2) Stooq 備援
    try:
        import pandas_datareader.data as web
        df = web.DataReader(symbol, "stooq")
        if df is not None and not df.empty:
            df = df.sort_index()
            df.rename(columns=str.capitalize, inplace=True)
            if "Volume" not in df.columns:
                df["Volume"] = np.nan
            needed = {"Open", "High", "Low", "Close"}
            if needed.issubset(set(df.columns)):
                return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print("stooq failed:", repr(e))
    
    return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = safe_series(df["Close"])
    high = safe_series(df["High"])
    low = safe_series(df["Low"])
    df["SMA20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["RSI14"] = RSIIndicator(close=close, window=14).rsi()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR14"] = atr.average_true_range()
    return df

def compute_signal_points(df: pd.DataFrame, cooldown_bars: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Time", "Signal", "Price"])
    data = df.dropna(subset=["EMA20", "SMA20", "RSI14", "Close"]).copy()
    if len(data) < 2:
        return pd.DataFrame(columns=["Time", "Signal", "Price"])
    
    ema_gt = data["EMA20"] > data["SMA20"]
    ema_gt_prev = ema_gt.shift(1)
    crossover_up = (ema_gt == True) & (ema_gt_prev == False)
    crossover_dn = (ema_gt == False) & (ema_gt_prev == True)
    buy_mask = crossover_up & (data["RSI14"] < 70)
    sell_mask = crossover_dn & (data["RSI14"] > 30)
    
    pts = []
    cooldown = 0
    for t, row in data.iterrows():
        if cooldown > 0:
            cooldown -= 1
            continue
        if bool(buy_mask.loc[t]):
            pts.append((t, "BUY", float(row["Close"])))
            cooldown = cooldown_bars
        elif bool(sell_mask.loc[t]):
            pts.append((t, "SELL", float(row["Close"])))
            cooldown = cooldown_bars
    return pd.DataFrame(pts, columns=["Time", "Signal", "Price"])

def estimate_future_triggers(df: pd.DataFrame) -> dict:
    """
    從目前狀態預估「未來可能觸發 BUY/SELL 的關鍵價位」
    回傳：{ 'buy_trigger': float or None, 'sell_trigger': float or None, 'bb_high', 'bb_low' }
    """
    if df.empty or len(df) < 2:
        return {'buy_trigger': None, 'sell_trigger': None, 'bb_high': None, 'bb_low': None}
    
    last = df.iloc[-1]
    ema = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
    sma = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
    bb_high = float(last["BB_High"]) if pd.notna(last["BB_High"]) else np.nan
    bb_low = float(last["BB_Low"]) if pd.notna(last["BB_Low"]) else np.nan
    
    result = {'buy_trigger': None, 'sell_trigger': None, 'bb_high': bb_high, 'bb_low': bb_low}
    
    if np.isnan(ema) or np.isnan(sma):
        return result
    
    # 目前空頭 → 預估「轉多」的觸發價位（突破 SMA20）
    if ema < sma:
        result['buy_trigger'] = sma
    # 目前多頭 → 預估「轉空」的觸發價位（跌破 SMA20）
    elif ema > sma:
        result['sell_trigger'] = sma
    
    return result

def latest_signal_state(df: pd.DataFrame) -> str:
    if df.empty:
        return "NO_DATA"
    last = df.iloc[-1]
    ema = float(last["EMA20"]) if pd.notna(last.get("EMA20", np.nan)) else np.nan
    sma = float(last["SMA20"]) if pd.notna(last.get("SMA20", np.nan)) else np.nan
    rsi = float(last["RSI14"]) if pd.notna(last.get("RSI14", np.nan)) else np.nan
    if np.isnan(ema) or np.isnan(sma) or np.isnan(rsi):
        return "INSUFFICIENT_DATA"
    if (ema > sma) and (rsi < 70):
        return "BUY"
    if (ema < sma) and (rsi > 30):
        return "SELL"
    return "HOLD"

def risk_levels(df: pd.DataFrame, rr: float, atr_mult: float, side: str):
    last = df.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR14"]) if pd.notna(last.get("ATR14", np.nan)) else np.nan
    if np.isnan(atr) or atr <= 0:
        return price, None, None
    if side == "BUY":
        stop = price - atr_mult * atr
        tp = price + rr * (price - stop)
    elif side == "SELL":
        stop = price + atr_mult * atr
        tp = price - rr * (stop - price)
    else:
        stop, tp = None, None
    return price, stop, tp

def simple_backtest(df: pd.DataFrame, rr: float, atr_mult: float, cooldown_bars: int = 3):
    if df.empty:
        return pd.DataFrame(), {}
    data = df.dropna(subset=["EMA20", "SMA20", "RSI14", "ATR14", "Close"]).copy()
    if len(data) < 30:
        return pd.DataFrame(), {"trades": 0}
    
    ema_gt = data["EMA20"] > data["SMA20"]
    ema_gt_prev = ema_gt.shift(1)
    crossover_up = (ema_gt == True) & (ema_gt_prev == False) & (data["RSI14"] < 70)
    crossover_dn = (ema_gt == False) & (ema_gt_prev == True) & (data["RSI14"] > 30)
    
    pos = 0
    entry = stop = tp = None
    entry_time = None
    trades = []
    cooldown = 0
    
    for i in range(1, len(data)):
        row = data.iloc[i]
        t = row.name
        price = float(row["Close"])
        atr = float(row["ATR14"])
        buy_sig = bool(crossover_up.iloc[i])
        sell_sig = bool(crossover_dn.iloc[i])
        
        if pos == 1:
            if stop is not None and price <= stop:
                trades.append(("LONG", entry_time, entry, t, stop, "STOP", stop - entry))
                pos = 0
            elif tp is not None and price >= tp:
                trades.append(("LONG", entry_time, entry, t, tp, "TP", tp - entry))
                pos = 0
            elif sell_sig:
                trades.append(("LONG", entry_time, entry, t, price, "REVERSE", price - entry))
                pos = 0
        elif pos == -1:
            if stop is not None and price >= stop:
                trades.append(("SHORT", entry_time, entry, t, stop, "STOP", entry - stop))
                pos = 0
            elif tp is not None and price <= tp:
                trades.append(("SHORT", entry_time, entry, t, tp, "TP", entry - tp))
                pos = 0
            elif buy_sig:
                trades.append(("SHORT", entry_time, entry, t, price, "REVERSE", entry - price))
                pos = 0
        
        if pos == 0:
            entry = stop = tp = None
            entry_time = None
            
        if cooldown > 0:
            cooldown -= 1
            continue
            
        if pos == 0:
            if buy_sig:
                pos = 1
                entry = price
                entry_time = t
                stop = entry - atr_mult * atr
                tp = entry + rr * (entry - stop)
                cooldown = cooldown_bars
            elif sell_sig:
                pos = -1
                entry = price
                entry_time = t
                stop = entry + atr_mult * atr
                tp = entry - rr * (stop - entry)
                cooldown = cooldown_bars
                
    if not trades:
        return pd.DataFrame(), {"trades": 0}
    
    tdf = pd.DataFrame(trades, columns=["Side", "EntryTime", "EntryPrice", "ExitTime", "ExitPrice", "Reason", "PnL"])
    stats = {
        "trades": int(len(tdf)),
        "win_rate": float((tdf["PnL"] > 0).mean()),
        "total_pnl": float(tdf["PnL"].sum()),
    }
    eq = tdf["PnL"].cumsum()
    dd = eq - eq.cummax()
    stats["max_drawdown"] = float(dd.min()) if len(dd) else 0.0
    return tdf, stats

def plot_chart(df: pd.DataFrame, title: str, signal_points: pd.DataFrame | None = None, 
               last_stop: float | None = None, last_tp: float | None = None,
               future_triggers: dict | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB High", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Low", line=dict(dash="dot")))
    
    # 歷史買賣點（交叉那根 + 冷卻 3 根）
    if signal_points is not None and not signal_points.empty:
        buys = signal_points[signal_points["Signal"] == "BUY"]
        sells = signal_points[signal_points["Signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Time"], y=buys["Price"], mode="markers", 
                                   name="BUY (crossover)", marker=dict(symbol="triangle-up", size=12)))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Time"], y=sells["Price"], mode="markers", 
                                   name="SELL (crossover)", marker=dict(symbol="triangle-down", size=12)))
    
    # 最後狀態的停損/停利（水平線）
    if last_stop is not None:
        fig.add_hline(y=last_stop, line_dash="dash", line_color="red", 
                     annotation_text=f"🛑 SL {last_stop:.0f}", annotation_position="top left")
    if last_tp is not None:
        fig.add_hline(y=last_tp, line_dash="dash", line_color="green", 
                     annotation_text=f"💰 TP {last_tp:.0f}", annotation_position="bottom left")
    
    # 🔮 未來觸發價位預估（新增）
    if future_triggers:
        # 潛在 BUY 觸發線
        if future_triggers.get('buy_trigger') is not None:
            fig.add_hline(
                y=future_triggers['buy_trigger'],
                line_dash="dot", line_color="lime", line_width=2,
                annotation_text=f"🟢 BUY if > {future_triggers['buy_trigger']:.0f}",
                annotation_position="top right", annotation_font=dict(size=10)
            )
        # 潛在 SELL 觸發線
        if future_triggers.get('sell_trigger') is not None:
            fig.add_hline(
                y=future_triggers['sell_trigger'],
                line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"🔴 SELL if < {future_triggers['sell_trigger']:.0f}",
                annotation_position="bottom right", annotation_font=dict(size=10)
            )
        # 布林通道壓力/支撐
        if future_triggers.get('bb_high') is not None:
            fig.add_hline(
                y=future_triggers['bb_high'],
                line_dash="dot", line_color="blue",
                annotation_text=f"🔵 BB Res", annotation_position="top left", annotation_font=dict(size=9)
            )
        if future_triggers.get('bb_low') is not None:
            fig.add_hline(
                y=future_triggers['bb_low'],
                line_dash="dot", line_color="purple",
                annotation_text=f"🟣 BB Sup", annotation_position="bottom left", annotation_font=dict(size=9)
            )
    
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Price", height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant（台股分析專業版）", layout="wide")
st.title("📈 AI Stock Trading Assistant（台股分析專業版 / 不自動下單）")
st.caption("買賣點只在 EMA20/SMA20 交叉當根顯示，並加入冷卻期 3 根；僅做資訊與分析提示，不做自動下單。")

COOLDOWN_BARS = 3

with st.sidebar:
    st.header("設定")
    code = st.text_input("台股代號（例：2330、2317、0050）", value="2330")
    period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("K 線", ["1d", "1wk", "1mo"], index=0)
    st.divider()
    st.subheader("券商（僅做資訊顯示，不下單）")
    broker = st.selectbox("券商", ["元大", "富邦", "國泰", "凱基", "永豐", "其他"], index=0)
    st.divider()
    rr = st.slider("風險報酬比（Take Profit）", 1.0, 5.0, 2.0, 0.25)
    atr_mult = st.slider("Stop Loss ATR 倍數（越大越保守）", 0.5, 5.0, 1.5, 0.25)
    show_backtest = st.checkbox("顯示回測（交叉訊號 + 冷卻 3 根 + ATR 停損停利）", value=True)
    run = st.button("RUN", type="primary")

symbol = to_tw_symbol(code)

if not run:
    st.info("左側設定好代號與期間後，按 RUN。")
    st.stop()

# 1) Download
st.subheader("1) 下載股價資料")
with st.spinner("下載中..."):
    df = download_ohlc(symbol, period=period, interval=interval)
    if df.empty:
        st.error("下載不到資料。請確認代號（例：2330/2317/0050）或換 interval/period。")
        st.markdown("**若你遇到 SSL/curl(77)：** - 先在 venv 裡安裝備援：`pip install pandas-datareader` - 公司網路擋 SSL 時，請用家用網路/手機熱點測試")
        st.stop()
    st.success(f"已下載：{symbol} / {period} / {interval}（券商：{broker}）")
    st.write(df.tail(5))

# 2) Indicators + Points
st.subheader("2) 技術指標 + 買賣點（只顯示交叉那根 + 冷卻 3 根）")
df = add_indicators(df)
signal_points = compute_signal_points(df, cooldown_bars=COOLDOWN_BARS)

# 🔮 預估未來觸發價位
future_triggers = estimate_future_triggers(df)

# 3) Decision + Risk Levels + Future Triggers
st.subheader("3) AI Trading Decision + 未來觸發預估")
signal_state = latest_signal_state(df)
price, stop, tp = risk_levels(df, rr=rr, atr_mult=atr_mult, side=signal_state)

# 顯示未來觸發預估卡片
st.markdown("##### 🔮 未來觸發價位預估（從目前點看未來）")
c1, c2 = st.columns(2)
with c1:
    if future_triggers.get('buy_trigger'):
        st.info(f"🟢 **BUY 觸發價**：>{future_triggers['buy_trigger']:.0f}\n\n若價格突破此價位 + RSI<70 → 可能觸發買訊號")
    else:
        st.info("🟢 BUY 觸發：目前已在多頭區間")
with c2:
    if future_triggers.get('sell_trigger'):
        st.warning(f"🔴 **SELL 觸發價**：<{future_triggers['sell_trigger']:.0f}\n\n若價格跌破此價位 + RSI>30 → 可能觸發賣訊號")
    else:
        st.warning("🔴 SELL 觸發：目前已在空頭區間")

fig = plot_chart(df, title=f"{symbol} Price + Indicators（含未來觸發預估）", 
                signal_points=signal_points, last_stop=stop, last_tp=tp, 
                future_triggers=future_triggers)
st.plotly_chart(fig, use_container_width=True)

if signal_state == "BUY":
    st.success("BUY state — EMA20 > SMA20 且 RSI14 < 70")
elif signal_state == "SELL":
    st.error("SELL state — EMA20 < SMA20 且 RSI14 > 30")
elif signal_state == "HOLD":
    st.warning("HOLD — 訊號不明確，建議觀望")
elif signal_state == "INSUFFICIENT_DATA":
    st.info("資料不足（指標需要足夠 K 數），請拉長期間或用 1d/1wk。")
else:
    st.info("沒有資料")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"{price:,.2f}")
c2.metric("Stop Loss", "-" if stop is None else f"{stop:,.2f}")
c3.metric("Take Profit", "-" if tp is None else f"{tp:,.2f}")
c4.metric("Risk-Reward", f"1 : {rr:.2f}")

# 4) Snapshot
st.subheader("4) 指標快照（最近 10 筆）")
snap_cols = ["Close", "SMA20", "EMA20", "RSI14", "BB_High", "BB_Low", "ATR14"]
st.dataframe(df[snap_cols].tail(10), use_container_width=True)

# 5) Backtest
if show_backtest:
    st.subheader("5) 簡易回測（交叉訊號 + 冷卻 3 根 + ATR 停損停利，示意）")
    tdf, stats = simple_backtest(df, rr=rr, atr_mult=atr_mult, cooldown_bars=COOLDOWN_BARS)
    if not stats or stats.get("trades", 0) == 0:
        st.info("回測交易數為 0（可能期間太短或交叉訊號未觸發）。")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Trades", f"{stats['trades']}")
        b2.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
        b3.metric("Total PnL", f"{stats['total_pnl']:.2f}")
        b4.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}")
        if tdf is not None and not tdf.empty:
            st.dataframe(tdf.tail(50), use_container_width=True)

st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")


