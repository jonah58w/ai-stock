# app.py
# AI Stock Trading Assistant（台股分析專業版 / 雙模系統 / 不自動下單）
# ✅ 模式：單一股票分析 + Top 10 掃描器
# ✅ 台股代號：2330 / 2317 / 0050（自動加 .TW）
# ✅ 股票名稱：自動顯示中文名稱
# ✅ 篩選條件：價格 > 100元，成交量 > 1000張
# ✅ 資料下載：TWSE 官方日線
# ✅ 指標：SMA/EMA/RSI/Bollinger/ATR
# ✅ 買賣點：只在 EMA20/SMA20「交叉那一根」出現 + 冷卻期 3 根
# ✅ 停損停利：ATR StopLoss + RR TakeProfit
# ✅ 回測：只用交叉訊號進出 + 冷卻期 + ATR 停損停利
# ✅ 未來觸發預估：從目前點看未來可能的買賣點
# ⚠️ 僅做資訊顯示，不含自動下單

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime

# -----------------------------
# 台股股票名稱對照表（常用股）
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
    "2938": "雄獅", "2939": "凱撒", "2940": "美食", "2941": "八方雲集"
}

# 熱門股清單（用於 Top 10 掃描）
TW_STOCK_POOL = list(TW_STOCK_NAMES.keys())

# -----------------------------
# Helpers
# -----------------------------
def to_tw_symbol(code: str) -> str:
    code = str(code).strip()
    if not code:
        return ""
    if code.upper().endswith(".TW") or code.upper().endswith(".TWO"):
        return code.upper()
    return f"{code}.TW"

def get_stock_name(code: str) -> str:
    """獲取股票中文名稱"""
    code = str(code).strip()
    return TW_STOCK_NAMES.get(code, "未知")

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

@st.cache_data(show_spinner=False, ttl=3600)
def twse_fetch_month(stock_no: str, ym: datetime) -> pd.DataFrame:
    """TWSE 官方日線資料"""
    stock_no = str(stock_no).strip().zfill(4)
    date_str = ym.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    params = {"response": "json", "date": date_str, "stockNo": stock_no}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("stat") != "OK":
            return pd.DataFrame()
        data = js.get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for row in data:
            d = row[0].strip()
            yy, mm, dd = d.split("/")
            ad_year = int(yy) + 1911
            dt = datetime(ad_year, int(mm), int(dd))
            def to_float(s):
                s = str(s).replace(",", "").strip()
                if s in ("--", "", "nan", "None"):
                    return np.nan
                return float(s)
            def to_int(s):
                s = str(s).replace(",", "").strip()
                if s in ("--", "", "nan", "None"):
                    return 0
                return int(float(s))
            rows.append({
                "Date": dt,
                "Open": to_float(row[3]),
                "High": to_float(row[4]),
                "Low": to_float(row[5]),
                "Close": to_float(row[6]),
                "Volume": to_int(row[1]),
            })
        df = pd.DataFrame(rows).sort_values("Date").set_index("Date")
        return df
    except Exception as e:
        print(f"TWSE failed: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def download_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """專業級下載：先 TWSE，失敗改用 Stooq"""
    stock_no = symbol.replace(".TW", "").replace(".TWO", "")
    months_map = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60}
    months = months_map.get(period, 12)
    end_dt = datetime.today()
    all_df = []
    for k in range(months - 1, -1, -1):
        mm = end_dt.month - k
        yy = end_dt.year
        while mm <= 0:
            mm += 12
            yy -= 1
        try:
            d = twse_fetch_month(stock_no, datetime(yy, mm, 1))
            if not d.empty:
                all_df.append(d)
        except Exception:
            continue
    if not all_df:
        return pd.DataFrame()
    df = pd.concat(all_df).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if interval == "1wk":
        df = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    elif interval == "1mo":
        df = df.resample("M").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    return df

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

def estimate_future_triggers(df: pd.DataFrame) -> dict:
    """從目前狀態預估未來可能觸發 BUY/SELL 的關鍵價位"""
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
    if ema < sma:
        result['buy_trigger'] = sma
    elif ema > sma:
        result['sell_trigger'] = sma
    return result

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

def plot_chart(df: pd.DataFrame, title: str, signal_points: pd.DataFrame | None = None, last_stop: float | None = None, last_tp: float | None = None, future_triggers: dict | None = None):
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
            fig.add_trace(go.Scatter(x=buys["Time"], y=buys["Price"], mode="markers", name="BUY (crossover)", marker=dict(symbol="triangle-up", size=12)))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Time"], y=sells["Price"], mode="markers", name="SELL (crossover)", marker=dict(symbol="triangle-down", size=12)))
    if last_stop is not None:
        fig.add_hline(y=last_stop, line_dash="dash", annotation_text="Stop Loss", annotation_position="top left")
    if last_tp is not None:
        fig.add_hline(y=last_tp, line_dash="dash", annotation_text="Take Profit", annotation_position="bottom left")
    if future_triggers:
        if future_triggers.get('buy_trigger') is not None:
            fig.add_hline(y=future_triggers['buy_trigger'], line_dash="dot", line_color="lime", line_width=2, annotation_text=f"🟢 BUY if > {future_triggers['buy_trigger']:.0f}", annotation_position="top right", annotation_font=dict(size=10))
        if future_triggers.get('sell_trigger') is not None:
            fig.add_hline(y=future_triggers['sell_trigger'], line_dash="dot", line_color="orange", line_width=2, annotation_text=f"🔴 SELL if < {future_triggers['sell_trigger']:.0f}", annotation_position="bottom right", annotation_font=dict(size=10))
        if future_triggers.get('bb_high') is not None:
            fig.add_hline(y=future_triggers['bb_high'], line_dash="dot", line_color="blue", annotation_text=f"🔵 BB Res", annotation_position="top left", annotation_font=dict(size=9))
        if future_triggers.get('bb_low') is not None:
            fig.add_hline(y=future_triggers['bb_low'], line_dash="dot", line_color="purple", annotation_text=f"🟣 BB Sup", annotation_position="bottom left", annotation_font=dict(size=9))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=560, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), margin=dict(l=10, r=10, t=60, b=10))
    return fig

def scan_top_stocks(stock_list, period, interval, rr, atr_mult, cooldown_bars=3, min_price=100, min_volume=1000):
    """掃描多檔股票，返回評分排行榜（含股票名稱 + 價格/成交量過濾）"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, code in enumerate(stock_list):
        status_text.text(f"掃描中：{code} ({i+1}/{len(stock_list)})")
        try:
            symbol = to_tw_symbol(code)
            df = download_ohlc(symbol, period, interval)
            if df.empty or len(df) < 30:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            price = float(last["Close"])
            volume = float(last["Volume"]) if "Volume" in df.columns else 0
            volume_in_thousands = volume / 1000  # 轉換為張數
            ema = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
            sma = float(last["SMA20"]) if pd.notna(last["SMA20"]) else np.nan
            rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
            atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
            if any(np.isnan([ema, sma, rsi])):
                continue
            # ✅ 加入價格和成交量過濾
            if price < min_price or volume_in_thousands < min_volume:
                continue
            score = 0
            signal = "HOLD"
            buy_point = None
            sell_point = None
            if (ema > sma) and (rsi < 70):
                signal = "BUY"
                buy_point = price
                score = (ema - sma) / sma * 100 + (70 - rsi) / 70 * 50 + 50
                if not np.isnan(atr) and atr > 0:
                    stop = price - atr_mult * atr
                    tp = price + rr * (price - stop)
                else:
                    stop = price * 0.95
                    tp = price * 1.10
                # 獲取股票名稱
                stock_name = get_stock_name(code)
                results.append({
                    "代號": code,
                    "名稱": stock_name,
                    "價格": round(price, 2),
                    "成交量(張)": round(volume_in_thousands, 1),
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
                # 獲取股票名稱
                stock_name = get_stock_name(code)
                results.append({
                    "代號": code,
                    "名稱": stock_name,
                    "價格": round(price, 2),
                    "成交量(張)": round(volume_in_thousands, 1),
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
st.set_page_config(page_title="AI Stock Trading Assistant（台股分析專業版）", layout="wide")
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
        show_backtest = st.checkbox("顯示回測（交叉訊號 + 冷卻 3 根 + ATR 停損停利）", value=True)
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

if mode == "單一股票分析":
    symbol = to_tw_symbol(code)
    if not run:
        st.info("左側設定好代號與期間後，按 RUN。")
        st.stop()
    st.subheader("1) 下載股價資料")
    with st.spinner("下載中..."):
        df = download_ohlc(symbol, period=period, interval=interval)
        if df.empty:
            st.error("下載不到資料。請確認代號（例：2330/2317/0050）或換 interval/period。")
            st.stop()
        st.success(f"已下載：{symbol} / {period} / {interval}（券商：{broker}）")
        st.write(df.tail(5))
    st.subheader("2) 技術指標 + 買賣點（只顯示交叉那根 + 冷卻 3 根）")
    df = add_indicators(df)
    signal_points = compute_signal_points(df, cooldown_bars=COOLDOWN_BARS)
    future_triggers = estimate_future_triggers(df)
    st.subheader("3) AI Trading Decision（最後狀態 + 停損/停利 + 未來觸發預估）")
    signal_state = latest_signal_state(df)
    price, stop, tp = risk_levels(df, rr=rr, atr_mult=atr_mult, side=signal_state)
    fig = plot_chart(df, title=f"{symbol} Price + Indicators（含未來觸發預估）", signal_points=signal_points, last_stop=stop, last_tp=tp, future_triggers=future_triggers)
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
    st.subheader("4) 指標快照（最近 10 筆）")
    snap_cols = ["Close", "SMA20", "EMA20", "RSI14", "BB_High", "BB_Low", "ATR14"]
    st.dataframe(df[snap_cols].tail(10), use_container_width=True)
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

elif mode == "Top 10 掃描器":
    st.subheader("🏆 Top 10 強勢買點/賣點掃描")
    st.caption(f"掃描熱門股池（價格>{min_price}元，成交量>{min_volume}張），找出評分最高的 10 檔股票")
    if run:
        top10 = scan_top_stocks(TW_STOCK_POOL[:num_stocks], period, interval, rr, atr_mult, cooldown_bars=COOLDOWN_BARS, min_price=min_price, min_volume=min_volume)
        if not top10.empty:
            st.success(f"找到 {len(top10)} 檔符合訊號的股票（價格>{min_price}元，成交量>{min_volume}張）")
            # 顯示表格（包含名稱）
            display_cols = ["代號", "名稱", "價格", "成交量(張)", "訊號", "買點" if "買點" in top10.columns else "賣點", "停損", "停利", "評分"]
            st.dataframe(top10[display_cols], use_container_width=True)
            csv = top10.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(label="📥 下載掃描結果 (CSV)", data=csv, file_name=f'top10_{pd.Timestamp.now().strftime("%Y%m%d")}.csv', mime='text/csv')
        else:
            st.warning(f"今日沒有符合訊號的股票（價格>{min_price}元，成交量>{min_volume}張），建議調整參數或期間。")
    else:
        st.info("點擊左側「🔍 開始掃描」按鈕開始分析。")

st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")

