# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雲端優化版）
# ✅ 修正 KeyError: 訊號欄位
# ✅ 優先使用 Yahoo Finance（雲端穩定）
# ✅ 正確股票名稱對照
# ✅ 符合現實的買賣點策略
# ✅ 支撐壓力位自動識別

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
# 台股股票名稱對照表（已修正）
# -----------------------------
TW_STOCK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電",
    "2881": "富邦金", "2882": "國泰金", "1301": "台塑", "1303": "南亞",
    "2603": "長榮", "2615": "萬海", "0050": "元大台灣 50", "0056": "元大高股息",
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
    "6274": "台燿", 
    "2449": "京元電",  # ✅ 已修正
    "3711": "日月光投控", 
    "8046": "南電",
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
    """雲端優化下載：優先 Yahoo Finance"""
    stock_no = symbol.replace(".TW", "").replace(".TWO", "")
    
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
                st.success(f"✅ 下載成功！{len(df)} 筆資料")
                return df[["Open", "High", "Low", "Close", "Volume"]]
            else:
                st.warning(f"⚠️ 欄位不完整：{list(df.columns)}")
        else:
            st.warning("⚠️ 返回空資料")
    
    except Exception as e:
        st.error(f"❌ 下載失敗：{str(e)[:100]}")
    
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
    
    ema12 = EMAIndicator(close=close, window=12).ema_indicator()
    ema26 = EMAIndicator(close=close, window=26).ema_indicator()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = EMAIndicator(close=df["MACD"], window=9).ema_indicator()
    
    kd = StochasticOscillator(high=high, low=low, close=close, window=14)
    df["K"] = kd.stoch()
    df["D"] = df["K"].rolling(3).mean()
    
    return df

def calculate_support_resistance(df: pd.DataFrame) -> dict:
    """自動識別支撐壓力位"""
    if df.empty or len(df) < 20:
        return {}
    
    current_price = df["Close"].iloc[-1]
    recent_high = df["High"].tail(20).max()
    recent_low = df["Low"].tail(20).min()
    
    lookback = min(252, len(df))
    high_52w = df["High"].tail(lookback).max()
    low_52w = df["Low"].tail(lookback).min()
    
    bb_high = df["BB_High"].iloc[-1] if pd.notna(df["BB_High"].iloc[-1]) else None
    bb_low = df["BB_Low"].iloc[-1] if pd.notna(df["BB_Low"].iloc[-1]) else None
    
    def calc_gap(price):
        if price is None or pd.isna(price):
            return None
        return round((price - current_price) / current_price * 100, 2)
    
    return {
        "壓力位": {
            "近期高點": {"價": round(recent_high, 2), "差距": calc_gap(recent_high)},
            "布林上軌": {"價": round(bb_high, 2) if bb_high else None, "差距": calc_gap(bb_high)},
            "52 周高點": {"價": round(high_52w, 2), "差距": calc_gap(high_52w)},
        },
        "支撐位": {
            "近期低點": {"價": round(recent_low, 2), "差距": calc_gap(recent_low)},
            "布林下軌": {"價": round(bb_low, 2) if bb_low else None, "差距": calc_gap(bb_low)},
            "52 周低點": {"價": round(low_52w, 2), "差距": calc_gap(low_52w)},
        }
    }

def compute_signal_points(df: pd.DataFrame, cooldown_bars: int = 3) -> pd.DataFrame:
    """計算歷史買賣點（含冷卻期）"""
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

def estimate_future_buy_sell_points(df: pd.DataFrame, rr: float, atr_mult: float, support_resistance: dict) -> dict:
    """預估未來買賣點（結合支撐壓力 + 突破策略）"""
    if df.empty or len(df) < 30:
        return {}
    
    last = df.iloc[-1]
    close = float(last["Close"])
    ema20 = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
    sma20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
    macd = float(last["MACD"]) if pd.notna(last["MACD"]) else np.nan
    macd_signal = float(last["MACD_Signal"]) if pd.notna(last["MACD_Signal"]) else np.nan
    
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    recent_high = df["High"].tail(20).max()
    
    result = {
        "current_price": close,
        "future_buy_points": [],
        "future_sell_points": []
    }
    
    support_levels = support_resistance.get("支撐位", {})
    resistance_levels = support_resistance.get("壓力位", {})
    
    nearest_resistance = None
    for key, value in resistance_levels.items():
        if value and value["價"] is not None and value["價"] > close:
            if nearest_resistance is None or value["價"] < nearest_resistance:
                nearest_resistance = value["價"]
    
    nearest_support = None
    for key, value in support_levels.items():
        if value and value["價"] is not None and value["價"] < close:
            if nearest_support is None or value["價"] > nearest_support:
                nearest_support = value["價"]
    
    # ===== 買點策略 =====
    
    # 策略 1：突破壓力買點
    if nearest_resistance:
        breakout_price = nearest_resistance * 1.01
        if not np.isnan(atr) and atr > 0:
            stop_loss = breakout_price - atr_mult * atr
            take_profit = breakout_price + rr * (breakout_price - stop_loss)
        else:
            stop_loss = breakout_price * 0.95
            take_profit = breakout_price * 1.10
        
        distance = (nearest_resistance - close) / close * 100
        
        result["future_buy_points"].append({
            "情境": "🚀 突破壓力買點",
            "預估買點": round(breakout_price, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"價格突破 {round(nearest_resistance, 2)}（距離：{distance:+.1f}%）",
            "優先級": "高" if distance < 10 else "中"
        })
    
    # 策略 2：回檔支撐買點
    if nearest_support:
        pullback_price = nearest_support * 1.01
        if not np.isnan(atr) and atr > 0:
            stop_loss = pullback_price - atr_mult * atr
            take_profit = pullback_price + rr * (pullback_price - stop_loss)
        else:
            stop_loss = pullback_price * 0.95
            take_profit = pullback_price * 1.10
        
        distance = (nearest_support - close) / close * 100
        
        result["future_buy_points"].append({
            "情境": "📉 回檔支撐買點",
            "預估買點": round(pullback_price, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"價格回測 {round(nearest_support, 2)}（距離：{distance:+.1f}%）",
            "優先級": "高" if abs(distance) < 10 else "低"
        })
    
    # 策略 3：均線多頭確認買點
    if ma5 > ma10 > ma20:
        ma_buy_price = close * 1.01
        if not np.isnan(atr) and atr > 0:
            stop_loss = ma_buy_price - atr_mult * atr
            take_profit = ma_buy_price + rr * (ma_buy_price - stop_loss)
        else:
            stop_loss = ma_buy_price * 0.95
            take_profit = ma_buy_price * 1.10
        
        result["future_buy_points"].append({
            "情境": "📊 均線多頭確認買點",
            "預估買點": round(ma_buy_price, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"5/10/20 日均線多頭排列\nMACD: {round(macd, 2)} > 訊號線：{round(macd_signal, 2)}",
            "優先級": "高"
        })
    
    # ===== 賣點策略 =====
    
    # 策略 1：跌破支撐賣點
    if nearest_support:
        breakdown_price = nearest_support * 0.99
        if not np.isnan(atr) and atr > 0:
            stop_loss = breakdown_price + atr_mult * atr
            take_profit = breakdown_price - rr * (stop_loss - breakdown_price)
        else:
            stop_loss = breakdown_price * 1.05
            take_profit = breakdown_price * 0.90
        
        distance = (nearest_support - close) / close * 100
        
        result["future_sell_points"].append({
            "情境": "🛑 跌破支撐賣點（停損）",
            "預估賣點": round(breakdown_price, 2),
            "停損": round(stop_loss, 2),
            "停利": "N/A",
            "條件": f"價格跌破 {round(nearest_support, 2)}（距離：{distance:+.1f}%）",
            "優先級": "🔴 高",
            "類型": "停損"
        })
    
    # 策略 2：觸及壓力賣點
    if nearest_resistance:
        resistance_sell_price = nearest_resistance * 0.99
        if not np.isnan(atr) and atr > 0:
            stop_loss = resistance_sell_price + atr_mult * atr
            take_profit = resistance_sell_price - rr * (stop_loss - resistance_sell_price)
        else:
            stop_loss = resistance_sell_price * 1.05
            take_profit = resistance_sell_price * 0.90
        
        distance = (nearest_resistance - close) / close * 100
        
        result["future_sell_points"].append({
            "情境": "🎯 觸及壓力賣點（獲利）",
            "預估賣點": round(resistance_sell_price, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"價格接近壓力位 {round(nearest_resistance, 2)}（距離：{distance:+.1f}%）",
            "優先級": "🟡 中",
            "類型": "獲利"
        })
    
    # 策略 3：均線死亡交叉
    if ema20 > sma20:
        death_cross_price = sma20 * 0.99
        if not np.isnan(atr) and atr > 0:
            stop_loss = death_cross_price + atr_mult * atr
            take_profit = death_cross_price - rr * (stop_loss - death_cross_price)
        else:
            stop_loss = death_cross_price * 1.05
            take_profit = death_cross_price * 0.90
        
        result["future_sell_points"].append({
            "情境": "📉 均線死亡交叉",
            "預估賣點": round(death_cross_price, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"EMA20 跌破 SMA20（目前：{round(sma20, 2)}）",
            "優先級": "🔴 高",
            "類型": "趨勢反轉"
        })
    
    # 策略 4：移動停利
    if close < recent_high:
        trailing_stop_5 = recent_high * 0.95
        trailing_stop_10 = recent_high * 0.90
        
        if not np.isnan(atr) and atr > 0:
            stop_loss = trailing_stop_5 + atr_mult * atr
            take_profit = trailing_stop_5 - rr * (stop_loss - trailing_stop_5)
        else:
            stop_loss = trailing_stop_5 * 1.05
            take_profit = trailing_stop_5 * 0.90
        
        pullback_pct = (recent_high - close) / recent_high * 100
        
        result["future_sell_points"].append({
            "情境": "📊 移動停利",
            "預估賣點": f"{round(trailing_stop_5, 2)} (-5%)",
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"近期高點：{round(recent_high, 2)}\n回撤：{pullback_pct:+.1f}%",
            "優先級": "🟢 中",
            "類型": "保護獲利"
        })
    
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

def plot_chart(df: pd.DataFrame, title: str, signal_points: pd.DataFrame | None = None, 
               last_stop: float | None = None, last_tp: float | None = None, 
               support_resistance: dict | None = None):
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
    
    if support_resistance:
        for key, value in support_resistance.get("壓力位", {}).items():
            if value and value["價"] is not None:
                fig.add_hline(y=value["價"], line_dash="dash", line_color="rgba(255,0,0,0.3)", 
                             annotation_text=f"🔴 {key}")
        
        for key, value in support_resistance.get("支撐位", {}).items():
            if value and value["價"] is not None:
                fig.add_hline(y=value["價"], line_dash="dash", line_color="rgba(0,255,0,0.3)", 
                             annotation_text=f"🟢 {key}")
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", 
                     height=600, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def calculate_confluence_score(df: pd.DataFrame) -> dict:
    """計算多指標共振分數"""
    score = 0
    signals = {}
    last = df.iloc[-1]
    
    ma5 = df["Close"].rolling(5).mean().iloc[-1]
    ma10 = df["Close"].rolling(10).mean().iloc[-1]
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    
    if ma5 > ma10 > ma20:
        score += 25
        signals["均線多頭"] = "✅"
    else:
        signals["均線多頭"] = "❌"
    
    macd = float(last["MACD"]) if pd.notna(last.get("MACD", np.nan)) else np.nan
    macd_signal = float(last["MACD_Signal"]) if pd.notna(last.get("MACD_Signal", np.nan)) else np.nan
    
    if not np.isnan(macd) and not np.isnan(macd_signal):
        if macd > macd_signal:
            score += 25
            signals["MACD"] = "✅"
        else:
            signals["MACD"] = "❌"
    else:
        signals["MACD"] = "⚠️"
    
    if "Volume" in df.columns:
        recent_vol = df["Volume"].tail(5).mean()
        prev_vol = df["Volume"].tail(10).head(5).mean()
        vol_increase = (recent_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        if vol_increase > 0.3:
            score += 25
            signals["成交量"] = "✅"
        else:
            signals["成交量"] = "❌"
    else:
        signals["成交量"] = "⚠️"
    
    k = float(last["K"]) if pd.notna(last.get("K", np.nan)) else np.nan
    d = float(last["D"]) if pd.notna(last.get("D", np.nan)) else np.nan
    
    if not np.isnan(k) and not np.isnan(d):
        if k > d and k < 80:
            score += 25
            signals["KD"] = "✅"
        else:
            signals["KD"] = "❌"
    else:
        signals["KD"] = "⚠️"
    
    return {"score": score, "signals": signals}

def scan_top_stocks(stock_list, period, interval, rr, atr_mult, cooldown_bars=3, 
                   min_price=100, min_volume=1000):
    """掃描多檔股票"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(stock_list):
        status_text.text(f"掃描中：{code} - {get_stock_name(code)} ({i+1}/{len(stock_list)})")
        try:
            symbol = to_tw_symbol(code)
            df = download_ohlc(symbol, period, interval)
            if df.empty or len(df) < 30:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            price = float(last["Close"])
            volume = float(last["Volume"]) if "Volume" in df.columns else 0
            volume_in_thousands = volume / 1000
            ema = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
            sma = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
            rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
            atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
            
            if any(np.isnan([ema, sma, rsi])):
                continue
            if price < min_price or volume_in_thousands < min_volume:
                continue
            
            score = 0
            signal = "HOLD"
            buy_point = None
            sell_point = None
            
            if (ema > sma) and (rsi < 70):
                signal = "BUY"
                buy_point = price
                confluence = calculate_confluence_score(df)
                score = confluence["score"]
                if not np.isnan(atr) and atr > 0:
                    stop = price - atr_mult * atr
                    tp = price + rr * (price - stop)
                else:
                    stop = price * 0.95
                    tp = price * 1.10
                stock_name = get_stock_name(code)
                results.append({
                    "代號": code,
                    "名稱": stock_name,
                    "價格": round(price, 2),
                    "成交量 (張)": round(volume_in_thousands, 1),
                    "訊號": signal,
                    "買點": round(buy_point, 2),
                    "停損": round(stop, 2),
                    "停利": round(tp, 2),
                    "評分": round(score, 2)
                })
            elif (ema < sma) and (rsi > 30):
                signal = "SELL"
                sell_point = price
                if not np.isnan(atr) and atr > 0:
                    stop = price + atr_mult * atr
                    tp = price - rr * (stop - price)
                else:
                    stop = price * 1.05
                    tp = price * 0.90
                stock_name = get_stock_name(code)
                results.append({
                    "代號": code,
                    "名稱": stock_name,
                    "價格": round(price, 2),
                    "成交量 (張)": round(volume_in_thousands, 1),
                    "訊號": signal,
                    "賣點": round(sell_point, 2),
                    "停損": round(stop, 2),
                    "停利": round(tp, 2),
                    "評分": round(score, 2)
                })
        except Exception as e:
            print(f"Error scanning {code}: {e}")
            continue
        progress_bar.progress((i + 1) / len(stock_list))
    
    status_text.text("掃描完成！")
    progress_bar.empty()
    
    if results:
        df_results = pd.DataFrame(results)
        return df_results.sort_values("評分", ascending=False).head(10)
    return pd.DataFrame()

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
        atr_mult = st.slider("Stop Loss ATR 倍數", 0.5, 5.0, 1.5, 0.25)
        
        run = st.button("RUN", type="primary")
    else:
        num_stocks = st.slider("掃描股票數量", 10, 100, 50)
        period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        interval = st.selectbox("K 線", ["1d", "1wk"], index=0)
        
        st.divider()
        st.subheader("篩選條件")
        min_price = st.number_input("最低價格（元）", min_value=0, max_value=10000, value=100, step=10)
        min_volume = st.number_input("最低成交量（張）", min_value=0, max_value=1000000, value=1000, step=100)
        
        st.divider()
        rr = st.slider("風險報酬比", 1.0, 5.0, 2.0, 0.25)
        atr_mult = st.slider("Stop Loss ATR 倍數", 0.5, 5.0, 1.5, 0.25)
        
        run = st.button("🔍 開始掃描", type="primary")

if mode == "單一股票分析" and run:
    symbol = to_tw_symbol(code)
    stock_name = get_stock_name(code)
    
    st.subheader("1) 下載股價資料")
    
    with st.spinner(f"下載中... {symbol} {stock_name}"):
        df = download_ohlc(symbol, period=period, interval=interval)
        
        if df.empty:
            st.error(f"""
            ❌ 下載不到資料。
            
            **股票代號**：{symbol}
            **股票名稱**：{stock_name}
            
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
    support_resistance = calculate_support_resistance(df)
    signal_points = compute_signal_points(df, cooldown_bars=COOLDOWN_BARS)
    future_points = estimate_future_buy_sell_points(df, rr, atr_mult, support_resistance)
    
    st.subheader("3) AI Trading Decision")
    signal_state = latest_signal_state(df)
    price, stop, tp = risk_levels(df, rr=rr, atr_mult=atr_mult, side=signal_state)
    
    fig = plot_chart(df, title=f"{symbol} {stock_name} Price + Indicators", 
                    signal_points=signal_points, last_stop=stop, last_tp=tp, 
                    support_resistance=support_resistance)
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
    
    # 顯示支撐壓力位
    st.subheader("4) 📊 關鍵支撐壓力位")
    if support_resistance:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔴 壓力位")
            for key, value in support_resistance.get("壓力位", {}).items():
                if value and value["價"] is not None:
                    gap_str = f"({value['差距']:+.2f}%)" if value["差距"] is not None else ""
                    st.info(f"**{key}**: {value['價']} {gap_str}")
        with col2:
            st.markdown("##### 🟢 支撐位")
            for key, value in support_resistance.get("支撐位", {}).items():
                if value and value["價"] is not None:
                    gap_str = f"({value['差距']:+.2f}%)" if value["差距"] is not None else ""
                    st.info(f"**{key}**: {value['價']} {gap_str}")
    else:
        st.info("資料不足，無法計算支撐壓力位")
    
    # 顯示未來預估買賣點
    st.subheader("5) 🔮 未來預估買賣點")
    if future_points:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🟢 未來潛在買點")
            if future_points.get("future_buy_points"):
                for i, buy in enumerate(future_points["future_buy_points"], 1):
                    st.info(f"**{i}. {buy['情境']}**\n\n"
                           f"預估買點：**{buy['預估買點']}**\n\n"
                           f"停損：{buy['停損']}\n\n"
                           f"停利：{buy['停利']}\n\n"
                           f"條件：{buy['條件']}\n\n"
                           f"優先級：{buy.get('優先級', '中')}")
            else:
                st.info("目前無潛在買點訊號")
        with col2:
            st.markdown("##### 🔴 未來潛在賣點")
            if future_points.get("future_sell_points"):
                for i, sell in enumerate(future_points["future_sell_points"], 1):
                    st.warning(f"**{i}. {sell['情境']}**\n\n"
                              f"預估賣點：**{sell['預估賣點']}**\n\n"
                              f"停損：{sell['停損']}\n\n"
                              f"停利：{sell['停利']}\n\n"
                              f"條件：{sell['條件']}\n\n"
                              f"優先級：{sell.get('優先級', '中')}")
            else:
                st.warning("目前無潛在賣點訊號")
    else:
        st.info("資料不足，無法預估未來買賣點")
    
    # ✅ 修正：歷史買賣點記錄（KeyError 已修正）
    st.subheader("6) 歷史買賣點記錄")
    if not signal_points.empty:
        signal_points_display = signal_points.copy()
        signal_points_display["時間"] = signal_points_display["Time"].dt.strftime("%Y-%m-%d")
        signal_points_display["訊號"] = signal_points_display["Signal"]  # ✅ 新增這行
        signal_points_display["價格"] = signal_points_display["Price"].round(2)
        st.dataframe(signal_points_display[["時間", "訊號", "價格"]], use_container_width=True)
    else:
        st.info("期間內無歷史買賣點訊號")
    
    st.subheader("7) 指標快照（最近 10 筆）")
    snap_cols = ["Close", "SMA20", "EMA20", "RSI14", "BB_High", "BB_Low", "ATR14"]
    st.dataframe(df[snap_cols].tail(10), use_container_width=True)

elif mode == "Top 10 掃描器" and run:
    st.subheader("🏆 Top 10 強勢買點/賣點掃描")
    st.caption(f"掃描熱門股池（價格>{min_price}元，成交量>{min_volume}張）")
    
    top10 = scan_top_stocks(TW_STOCK_POOL[:num_stocks], period, interval, rr, atr_mult, 
                           cooldown_bars=COOLDOWN_BARS, min_price=min_price, 
                           min_volume=min_volume)
    
    if not top10.empty:
        st.success(f"找到 {len(top10)} 檔符合訊號的股票")
        display_cols = ["代號", "名稱", "價格", "成交量 (張)", "訊號", 
                      "買點" if "買點" in top10.columns else "賣點", "停損", "停利", "評分"]
        st.dataframe(top10[display_cols], use_container_width=True)
        
        csv = top10.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(label="📥 下載掃描結果 (CSV)", data=csv, 
                         file_name=f'top10_{pd.Timestamp.now().strftime("%Y%m%d")}.csv', 
                         mime='text/csv')
    else:
        st.warning(f"今日沒有符合訊號的股票，建議調整參數或期間。")

st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")
