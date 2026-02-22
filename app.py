# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雲端優化版）
# ✅ 優先使用 Yahoo Finance（雲端穩定）
# ✅ 智能降級策略
# ✅ 支援所有台股代號

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 台股股票名稱對照表
# -----------------------------
TW_STOCK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電",
    "2881": "富邦金", "2882": "國泰金", "1301": "台塑", "1303": "南亞",
    "2603": "長榮", "2615": "萬海", "0050": "元大台灣50", "0056": "元大高股息",
    "3008": "大立光", "3045": "台灣大", "2382": "廣達", "2303": "聯電",
    "2891": "中信金", "2892": "第一金", "2886": "兆豐金", "2885": "元大金",
    "2884": "玉山金", "2883": "開發金", "2880": "永豐金", "2889": "國票金",
    "2890": "王道銀行", "2897": "王道銀行", "2801": "彰銀", "2809": "京城銀",
    "2812": "台中銀", "2820": "華票", "2834": "東元", "2845": "遠東銀",
    "2855": "運彩科技", "2867": "三商銀", "2870": "新光金", "2871": "富邦媒",
    "2872": "中華電", "2873": "國巨", "2874": "華新科", "2875": "華碩",
    "2876": "技嘉", "2877": "微星", "2878": "瑞昱", "2879": "聯詠",
    "2887": "新唐", "2888": "台揚", "2894": "聯陽", "2895": "敦泰",
    "2896": "立積", "2898": "牧德", "2899": "力成", "2901": "欣欣",
    "2902": "遠百", "2903": "遠東新", "2904": "東元", "2905": "三商",
    "2906": "寒舍", "2908": "特力", "2910": "統領", "2911": "愛買",
    "2912": "統一超", "2913": "潤泰全", "2915": "特力", "2917": "新燕",
    "2918": "三商企銀", "2919": "東凌", "2920": "潤泰新", "2921": "統一",
    "2922": "大成", "2923": "卜蜂", "2924": "聯華", "2925": "泰山",
    "2926": "福懋油", "2927": "台塑化", "2928": "中油", "2929": "台汽電",
    "2930": "中租", "2931": "和潤", "2932": "裕融", "2933": "中租",
    "2934": "潤泰新", "2935": "潤泰全", "2936": "晶華", "2937": "王品",
    "2938": "雄獅", "2939": "凱撒", "2940": "美食", "2941": "八方雲集",
    "6274": "台燿", "2449": "京元電", "3711": "日月光投控", "8046": "南電",
    "3163": "波若威"
}

TW_STOCK_POOL = list(TW_STOCK_NAMES.keys())

# -----------------------------
# Helpers
# -----------------------------
def to_tw_symbol(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    if code.upper().endswith(".TW"):
        return code.upper()
    if code.upper().endswith(".TWO"):
        return code.upper()
    if code.startswith("6") or code.startswith("4"):
        return f"{code}.TWO"
    else:
        return f"{code}.TW"

def get_stock_name(code: str) -> str:
    code = str(code).strip().replace(".TW", "").replace(".TWO", "")
    return TW_STOCK_NAMES.get(code, code)

def safe_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError("Expected single column series")
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return pd.Series(arr)

@st.cache_data(show_spinner=False, ttl=3600)
def download_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    雲端優化下載：優先 Yahoo Finance
    """
    stock_no = symbol.replace(".TW", "").replace(".TWO", "")
    
    # 1️⃣ 優先使用 Yahoo Finance（雲端最穩定）
    try:
        import yfinance as yf
        st.write(f"🔄 從 Yahoo Finance 下載：{symbol}")
        
        df = yf.download(
            f"{stock_no}.TW",
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            timeout=30
        )
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(how="all")
            
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if needed.issubset(set(df.columns)):
                st.success(f"✅ Yahoo Finance 下載成功！{len(df)} 筆資料")
                return df[["Open", "High", "Low", "Close", "Volume"]]
            else:
                st.warning(f"⚠️ Yahoo Finance 欄位不完整：{list(df.columns)}")
        else:
            st.warning("⚠️ Yahoo Finance 返回空資料")
    
    except Exception as e:
        st.error(f"❌ Yahoo Finance 失敗：{str(e)[:100]}")
    
    # 2️⃣ 如果 Yahoo Finance 失敗，返回空 DataFrame
    st.error("❌ 所有資料源都失敗，請稍後再試或更換股票")
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
    
    # MACD
    ema12 = EMAIndicator(close=close, window=12).ema_indicator()
    ema26 = EMAIndicator(close=close, window=26).ema_indicator()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = EMAIndicator(close=df["MACD"], window=9).ema_indicator()
    
    # KD
    kd = StochasticOscillator(high=high, low=low, close=close, window=14)
    df["K"] = kd.stoch()
    df["D"] = df["K"].rolling(3).mean()
    
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

def plot_chart(df: pd.DataFrame, title: str, signal_points: pd.DataFrame | None = None, 
               last_stop: float | None = None, last_tp: float | None = None):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB High", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Low", line=dict(dash="dot")))
    
    if signal_points is not None and not signal_points.empty:
        buys = signal_points[signal_points["Signal"] == "BUY"]
        sells = signal_points[signal_points["Signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Time"], y=buys["Price"], mode="markers", 
                                   name="歷史 BUY", marker=dict(symbol="triangle-up", size=10)))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Time"], y=sells["Price"], mode="markers", 
                                   name="歷史 SELL", marker=dict(symbol="triangle-down", size=10)))
    
    if last_stop is not None:
        fig.add_hline(y=last_stop, line_dash="dash", line_color="red", 
                     annotation_text="Stop Loss")
    if last_tp is not None:
        fig.add_hline(y=last_tp, line_dash="dash", line_color="green", 
                     annotation_text="Take Profit")
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", 
                     height=600, margin=dict(l=10, r=10, t=60, b=10))
    return fig

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Stock Trading Assistant", layout="wide")
st.title("📈 AI Stock Trading Assistant（台股分析專業版 / 不自動下單）")
st.caption("買賣點只在 EMA20/SMA20 交叉當根顯示，並加入冷卻期 3 根；僅做資訊與分析提示，不做自動下單。")

COOLDOWN_BARS = 3

with st.sidebar:
    st.header("設定")
    mode = st.radio("選擇模式", ["單一股票分析", "Top 10 掃描器"])
    
    if mode == "單一股票分析":
        code = st.text_input("台股代號（例：2330、2317、0050）", value="2330")
        period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("K 線", ["1d", "1wk", "1mo"], index=0)
        
        st.divider()
        st.subheader("券商（僅做資訊顯示，不下單）")
        broker = st.selectbox("券商", ["元大", "富邦", "國泰", "凱基", "永豐", "其他"], index=0)
        
        st.divider()
        rr = st.slider("風險報酬比（Take Profit）", 1.0, 5.0, 2.0, 0.25)
        atr_mult = st.slider("Stop Loss ATR 倍數（越大越保守）", 0.5, 5.0, 1.5, 0.25)
        
        run = st.button("RUN", type="primary")
    else:
        st.info("Top 10 掃描器功能開發中...")
        run = False

if mode == "單一股票分析" and run:
    symbol = to_tw_symbol(code)
    stock_name = get_stock_name(code)
    
    st.subheader("1) 下載股價資料")
    
    with st.spinner(f"下載中... {symbol} {stock_name}"):
        df = download_ohlc(symbol, period=period, interval=interval)
        
        if df.empty:
            st.error(f"""
            ❌ 下載不到資料。請確認：
            
            **股票代號**：{symbol}
            **股票名稱**：{stock_name}
            
            **可能原因**：
            - 股票代號不存在或已下市
            - 網路連接問題（雲端環境限制）
            - Yahoo Finance 暫時無法訪問
            
            **建議**：
            1. 嘗試其他股票（如 2330、2317、2454）
            2. 更換期間（1y → 6mo）
            3. 稍後再試
            """)
            st.stop()
        
        st.success(f"✅ 已下載：{symbol} {stock_name} / {period} / {interval}（券商：{broker}）")
        st.write(df.tail(5))
    
    st.subheader("2) 技術指標 + 買賣點")
    df = add_indicators(df)
    signal_points = compute_signal_points(df, cooldown_bars=COOLDOWN_BARS)
    
    st.subheader("3) AI Trading Decision")
    signal_state = latest_signal_state(df)
    price, stop, tp = risk_levels(df, rr=rr, atr_mult=atr_mult, side=signal_state)
    
    fig = plot_chart(df, title=f"{symbol} {stock_name} Price + Indicators", 
                    signal_points=signal_points, last_stop=stop, last_tp=tp)
    st.plotly_chart(fig, use_container_width=True)
    
    if signal_state == "BUY":
        st.success("✅ BUY state — EMA20 > SMA20 且 RSI14 < 70")
    elif signal_state == "SELL":
        st.error("❌ SELL state — EMA20 < SMA20 且 RSI14 > 30")
    elif signal_state == "HOLD":
        st.warning("⚠️ HOLD — 訊號不明確，建議觀望")
    else:
        st.info("ℹ️ 資料不足")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{price:,.2f}")
    c2.metric("Stop Loss", "-" if stop is None else f"{stop:,.2f}")
    c3.metric("Take Profit", "-" if tp is None else f"{tp:,.2f}")
    c4.metric("Risk-Reward", f"1 : {rr:.2f}")
    
    st.subheader("4) 歷史買賣點記錄")
    if not signal_points.empty:
        signal_points_display = signal_points.copy()
        signal_points_display["時間"] = signal_points_display["Time"].dt.strftime("%Y-%m-%d")
        signal_points_display["訊號"] = signal_points_display["Signal"]
        signal_points_display["價格"] = signal_points_display["Price"].round(2)
        st.dataframe(signal_points_display[["時間", "訊號", "價格"]], use_container_width=True)
    else:
        st.info("期間內無歷史買賣點訊號")
    
    st.subheader("5) 指標快照（最近 10 筆）")
    snap_cols = ["Close", "SMA20", "EMA20", "RSI14", "BB_High", "BB_Low", "ATR14"]
    st.dataframe(df[snap_cols].tail(10), use_container_width=True)

st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")
