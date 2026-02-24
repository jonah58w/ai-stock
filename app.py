import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
import time

warnings.filterwarnings("ignore")

# ==================== 欄位正規化（修正重點） ====================
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把 yfinance 各種可能的欄位格式轉成標準 OHLCV"""
    if df is None or df.empty:
        return df

    df = df.copy()

    # 1) MultiIndex 欄位：自動判斷哪一層是 OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        levels = list(range(df.columns.nlevels))

        # 找到包含 OHLCV 的那一層（最常見是 level=1，但有時會反過來）
        target_level = None
        for lvl in levels:
            vals = set(str(x).strip().lower() for x in df.columns.get_level_values(lvl).unique())
            if {"open", "high", "low", "close", "volume"}.issubset(vals):
                target_level = lvl
                break

        # 若找不到，就用「看起來不像 ticker 的那層」做 fallback
        if target_level is None:
            # 假設 ticker 通常包含 .TW/.TWO 或數字，OHLCV 是文字
            # 這裡用：包含 open/high/low/close/volume 的程度來打分
            def score_level(lvl: int) -> int:
                vals = [str(x).strip().lower() for x in df.columns.get_level_values(lvl)]
                return sum(v in ["open", "high", "low", "close", "volume", "adj close"] for v in vals)

            target_level = max(levels, key=score_level)

        df.columns = df.columns.get_level_values(target_level)

    # 2) 欄位是「2313.TW Close」這種：抓出 Close/Open...
    elif any(isinstance(c, str) and ("close" in c.lower() or "open" in c.lower()) for c in df.columns):
        new_cols = {}
        for c in df.columns:
            s = str(c).strip().lower()
            if "open" in s:
                new_cols[c] = "Open"
            elif "high" in s:
                new_cols[c] = "High"
            elif "low" in s:
                new_cols[c] = "Low"
            elif "adj" in s and "close" in s:
                new_cols[c] = "Adj Close"
            elif "close" in s:
                new_cols[c] = "Close"
            elif "volume" in s or "vol" in s:
                new_cols[c] = "Volume"
            else:
                new_cols[c] = c
        df = df.rename(columns=new_cols)

    # 3) 統一大小寫與常見別名
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        s = c.strip().lower()
        if s == "open":
            rename_map[c] = "Open"
        elif s == "high":
            rename_map[c] = "High"
        elif s == "low":
            rename_map[c] = "Low"
        elif s == "close":
            rename_map[c] = "Close"
        elif s == "volume":
            rename_map[c] = "Volume"
        elif "adj" in s and "close" in s:
            rename_map[c] = "Adj Close"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df

# ==================== 技術指標計算 ====================
def calculate_indicators(df):
    """計算所有技術指標"""
    if df is None or df.empty:
        raise ValueError("數據為空")

    # ✅ 先正規化欄位（修正你目前爆掉的點）
    df = normalize_ohlcv_columns(df)

    # 檢查必需列
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"❌ 缺少必要欄位：{missing}")
        st.error(f"當前可用欄位：{df.columns.tolist()}")
        st.info("**可能原因：**\n1. Yahoo Finance 回傳格式改變（MultiIndex 層級顛倒）\n2. 資料下載不完整/被擋\n3. 股票代碼錯誤或暫時無資料")
        raise ValueError(f"缺少必要欄位：{missing}")

    df = df.copy()

    # 移動平均線
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # 布林帶
    df["BB_middle"] = df["SMA20"]
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_High"] = df["BB_middle"] + (df["BB_std"] * 2)
    df["BB_Low"] = df["BB_middle"] - (df["BB_std"] * 2)

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_DIF"] - df["MACD_Signal"]

    # KD
    low_9 = df["Low"].rolling(window=9).min()
    high_9 = df["High"].rolling(window=9).max()
    df["KD_K"] = 100 * (df["Close"] - low_9) / (high_9 - low_9 + 1e-8)
    df["KD_D"] = df["KD_K"].rolling(window=3).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = true_range
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df["ADX"] = dx.rolling(14).mean()

    # 成交量均線
    df["VOL_5T"] = df["Volume"].rolling(window=5).mean()
    df["VOL_10T"] = df["Volume"].rolling(window=10).mean()

    return df

# ==================== 共振確認機制 ====================
def calculate_resonance_score(df, signal_type="buy"):
    if len(df) < 20:
        return 0.0

    latest = df.iloc[-1]
    score = 0
    weights = {"MACD": 25, "KD": 20, "Volume": 20, "BB": 15, "MA": 10, "ADX": 10}

    if latest["MACD_DIF"] > latest["MACD_Signal"]:
        score += weights["MACD"]

    if signal_type == "buy":
        if latest["KD_K"] > latest["KD_D"] and latest["KD_K"] < 80:
            score += weights["KD"]
    else:
        if latest["KD_K"] < latest["KD_D"] or latest["KD_K"] > 80:
            score += weights["KD"]

    if latest["VOL_5T"] > latest["VOL_10T"] * 1.1:
        score += weights["Volume"]

    if signal_type == "buy":
        if latest["Close"] > latest["BB_Low"]:
            score += weights["BB"]
    else:
        if latest["Close"] < latest["BB_High"]:
            score += weights["BB"]

    if latest["SMA5"] > latest["SMA20"]:
        score += weights["MA"]

    if latest["ADX"] > 40:
        score += weights["ADX"]

    return round(min(score, 100), 1)

# ==================== 精準買點計算 ====================
def calculate_precise_buy_points(df):
    if len(df) < 30:
        return {}

    latest = df.iloc[-1]
    price = latest["Close"]
    ma20 = latest["SMA20"]
    bb_lower = latest["BB_Low"]
    recent_low = df["Low"].rolling(20).min().iloc[-1]
    adx = latest["ADX"]
    atr_pct = latest["ATR_Pct"]

    buy_points = {}

    if adx > 40 and ma20 > 0:
        pullback_buy = ma20 * 1.005
        stop_loss = pullback_buy * (1 - atr_pct / 100 * 2) if atr_pct > 0 else pullback_buy * 0.91
        take_profit = pullback_buy * (1 + atr_pct / 100 * 9) if atr_pct > 0 else pullback_buy * 1.45
        buy_points["趨勢突破型_回踩買點"] = {
            "預估買點": round(pullback_buy, 2),
            "停損": round(stop_loss, 2),
            "停利": round(take_profit, 2),
            "條件": f"多頭排列，回踩 MA20({ma20:.1f}) 附近",
            "共振確認": f'{calculate_resonance_score(df, "buy"):.1f}%',
            "優先級": "中",
            "適用": "趨勢市",
            "指標細節": {
                "ADX": f"{adx:.1f}",
                "乖離率": f"{(price / ma20 - 1) * 100:.1f}%",
                "ATR": f"{atr_pct:.2f}%",
                "KD": f'{latest["KD_K"]:.1f}/{latest["KD_D"]:.1f}',
            },
        }

    if adx <= 40 or price < ma20:
        if recent_low > 0:
            support_buy = recent_low * 1.01
            stop_loss = support_buy * 0.95
            take_profit = support_buy * 1.61
            distance_pct = (recent_low / price - 1) * 100
            buy_points["回檔等待型_支撐買點"] = {
                "預估買點": round(support_buy, 2),
                "停損": round(stop_loss, 2),
                "停利": round(take_profit, 2),
                "條件": f"價格回測{recent_low}（距離：{distance_pct:.1f}%）",
                "共振確認": f'{calculate_resonance_score(df, "buy"):.1f}%',
                "優先級": "中",
                "適用": "震盪市",
                "指標細節": {
                    "距離低點": f"{distance_pct:.1f}%",
                    "布林下軌": f"{bb_lower:.2f}",
                    "KD": f'{latest["KD_K"]:.1f}/{latest["KD_D"]:.1f}',
                    "量能": f'{latest["VOL_5T"]/latest["VOL_10T"]:.2f}x' if latest["VOL_10T"] > 0 else "N/A",
                },
            }

    return buy_points

# ==================== 精準賣點計算 ====================
def calculate_precise_sell_points(df):
    if len(df) < 30:
        return {}

    latest = df.iloc[-1]
    price = latest["Close"]
    ma20 = latest["SMA20"]
    bb_lower = latest["BB_Low"]
    recent_low = df["Low"].rolling(20).min().iloc[-1]
    atr_pct = latest["ATR_Pct"]
    adx = latest["ADX"]

    sell_points = {}

    if ma20 > 0 and recent_low > 0:
        support_level = min(ma20 * 0.95, recent_low)
        sell_price = support_level * 0.99
        stop_loss_sell = support_level * 1.05
        sell_points["跌破支撐賣點（停損）"] = {
            "預估賣點": round(sell_price, 2),
            "停損": round(stop_loss_sell, 2),
            "停利": "N/A",
            "條件": f"價格跌破{support_level}（關鍵支撐）",
            "突破確認": "待確認 (0 分)",
            "共振確認": f'{calculate_resonance_score(df, "sell"):.1f}%',
            "優先級": "中" if adx > 40 else "高",
            "適用": "趨勢市",
            "指標細節": {
                "支撐位": f"{support_level:.2f}",
                "ATR": f"{atr_pct:.2f}%",
                "MACD": f'{latest["MACD_DIF"]:.2f}/{latest["MACD_Signal"]:.2f}',
                "KD": f'{latest["KD_K"]:.1f}/{latest["KD_D"]:.1f}',
            },
        }

    if latest["MACD_DIF"] < latest["MACD_Signal"]:
        sell_points["MACD 死叉賣點"] = {
            "預估賣點": round(price * 0.98, 2),
            "停損": round(price * 1.03, 2),
            "停利": "N/A",
            "條件": "MACD 死叉 + KD 高檔",
            "共振確認": f'{calculate_resonance_score(df, "sell"):.1f}%',
            "優先級": "中",
            "適用": "趨勢/震盪市",
        }

    if latest["KD_K"] > 80 and price > bb_lower:
        sell_points["獲利了結賣點"] = {
            "預估賣點": round(price * 0.99, 2),
            "停損": round(bb_lower * 1.02, 2),
            "停利": "N/A",
            "條件": "KD>80 + 接近布林上軌",
            "共振確認": f'{calculate_resonance_score(df, "sell"):.1f}%',
            "優先級": "高",
            "適用": "趨勢市",
        }

    return sell_points

# ==================== 市場狀態判斷 ====================
def dual_mode_ai_analysis(df):
    latest = df.iloc[-1]
    adx = latest.get("ADX", 0)
    ma5 = latest["SMA5"]
    ma20 = latest["SMA20"]
    atr_pct = latest["ATR_Pct"]

    if adx > 40:
        market_state, trend_strength = "趨勢市", "強趨勢"
    elif adx > 25:
        market_state, trend_strength = "趨勢市", "中趨勢"
    else:
        market_state, trend_strength = "震盪市", "弱趨勢"

    ma_arrangement = "多頭排列" if ma5 > ma20 else "空頭排列"

    return {
        "市場狀態": market_state,
        "趨勢強度": f"{trend_strength}(ADX={adx:.1f})",
        "均線排列": ma_arrangement,
        "波動率": f'{"高" if atr_pct > 4 else "低"}波動 (ATR={atr_pct:.2f}%)',
        "建議模式": "趨勢突破型" if adx > 40 else "回檔等待型",
    }

# ==================== 支撐壓力位計算 ====================
def calculate_support_resistance(df):
    latest = df.iloc[-1]
    price = latest["Close"]

    recent_high = df["High"].rolling(20).max().iloc[-1]
    recent_low = df["Low"].rolling(20).min().iloc[-1]
    bb_high = latest["BB_High"]
    bb_low = latest["BB_Low"]

    if len(df) >= 252:
        high_52w = df["High"].rolling(252).max().iloc[-1]
        low_52w = df["Low"].rolling(252).min().iloc[-1]
    else:
        high_52w = df["High"].max()
        low_52w = df["Low"].min()

    def calc_distance(level, p):
        if p == 0:
            return "N/A"
        return f"{(level / p - 1) * 100:+.2f}%"

    return {
        "壓力位": {
            "近期高點": {"價位": round(recent_high, 2), "距離": calc_distance(recent_high, price)},
            "布林上軌": {"價位": round(bb_high, 2), "距離": calc_distance(bb_high, price)},
            "52 周高點": {"價位": round(high_52w, 2), "距離": calc_distance(high_52w, price)},
        },
        "支撐位": {
            "近期低點": {"價位": round(recent_low, 2), "距離": calc_distance(recent_low, price)},
            "布林下軌": {"價位": round(bb_low, 2), "距離": calc_distance(bb_low, price)},
            "52 周低點": {"價位": round(low_52w, 2), "距離": calc_distance(low_52w, price)},
        },
    }

# ==================== Top 10 篩選引擎 ====================
def screen_top_stocks(stock_list, days=252):
    results = []
    for stock in stock_list:
        try:
            df = yf.download(
                stock,
                period=f"{days}d",
                interval="1d",
                progress=False,
                group_by="column",
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty or len(df) < 30:
                continue

            df = calculate_indicators(df)
            latest = df.iloc[-1]

            ret_20d = (latest["Close"] / df.iloc[-20]["Close"] - 1) * 100 if len(df) >= 20 else 0
            adx = latest["ADX"]
            resonance = calculate_resonance_score(df, "buy")

            score = 0
            score += min(ret_20d, 30)
            score += min(adx * 0.5, 25)
            score += resonance * 0.3
            if latest["VOL_5T"] > latest["VOL_10T"] * 1.2:
                score += 15

            results.append(
                {
                    "股票": stock.replace(".TW", "").replace(".TWO", ""),
                    "價格": round(latest["Close"], 2),
                    "20 日漲幅": f"{ret_20d:.1f}%",
                    "ADX": f"{adx:.1f}",
                    "共振分數": f"{resonance:.1f}%",
                    "綜合分數": round(score, 1),
                }
            )
        except:
            continue

    if results:
        df_res = pd.DataFrame(results)
        return df_res.sort_values("綜合分數", ascending=False).head(10)
    return None

# ==================== 主程式 ====================
def main():
    st.set_page_config(page_title="AI Stock Trading Assistant", layout="wide")
    st.title("📊 AI Stock Trading Assistant（台股分析專業版）")
    st.markdown("**雙模式 AI 判斷**：回檔等待型 + 趨勢突破型 | 支撐/壓力 + 布林 + MACD + KD + 乖離率 + 成交量共振確認")

    st.sidebar.header("🔍 分析模式")
    mode = st.sidebar.radio("選擇模式", ["單一股票分析", "Top 10 潛力股篩選"])

    if mode == "單一股票分析":
        stock_input = st.sidebar.text_input("股票代碼", "2313")

        if stock_input:
            s = stock_input.strip()
            if ".TW" not in s.upper() and ".TWO" not in s.upper():
                stock_code = s + ".TW"
            else:
                stock_code = s
        else:
            stock_code = "2313.TW"

        st.sidebar.caption(f"完整代碼：{stock_code}")

        if st.sidebar.button("開始分析"):
            with st.spinner(f"正在下載 {stock_code} 數據..."):
                try:
                    df = None
                    for attempt in range(3):
                        try:
                            df = yf.download(
                                stock_code,
                                period="1y",
                                interval="1d",
                                progress=False,
                                group_by="column",
                                auto_adjust=False,
                                threads=False,
                            )
                            if df is not None and not df.empty:
                                break
                        except Exception:
                            if attempt < 2:
                                st.warning(f"嘗試 {attempt+1} 失敗，重試中...")
                                time.sleep(2)

                    if df is None or df.empty:
                        st.error(f"❌ 無法獲取 {stock_code} 的數據")
                        st.info("**可能原因：**\n1. 股票代碼錯誤\n2. Yahoo Finance 服務暫時不可用/被擋\n3. 網路連接問題\n4. 該股票已下市")
                        return

                    df = calculate_indicators(df)
                    df = df.reset_index()
                    latest = df.iloc[-1]
                    price = latest["Close"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("當前價格", f"{price:.2f}")
                    c2.metric("MA20", f"{latest['SMA20']:.2f}")
                    c3.metric("乖離率", f"{(price/latest['SMA20']-1)*100:+.2f}%")
                    c4.metric("ADX", f"{latest['ADX']:.1f}")

                    st.subheader("3) 關鍵支撐壓力位")
                    sr = calculate_support_resistance(df)
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.write("**壓力位**")
                        for k, v in sr["壓力位"].items():
                            st.caption(f"{k}: {v['價位']} ({v['距離']})")
                    with cc2:
                        st.write("**支撐位**")
                        for k, v in sr["支撐位"].items():
                            st.caption(f"{k}: {v['價位']} ({v['距離']})")

                    st.subheader("4) 未來預估買賣點（雙模式 AI 判斷）")
                    ai_status = dual_mode_ai_analysis(df)
                    st.info(f"當前市場狀態：{ai_status['市場狀態']}（{ai_status['趨勢強度']}）")

                    st.write("**未來潛在買點**")
                    buy_points = calculate_precise_buy_points(df)
                    for name, info in buy_points.items():
                        with st.expander(name):
                            st.metric("預估買點", info["預估買點"])
                            st.caption(f"停損：{info['停損']} | 停利：{info['停利']}")
                            st.caption(f"條件：{info['條件']}")
                            st.caption(f"共振確認：{info['共振確認']} | 優先級：{info['優先級']}")
                            if "指標細節" in info:
                                with st.expander("🔍 指標細節"):
                                    st.json(info["指標細節"])

                    st.write("**未來潛在賣點**")
                    sell_points = calculate_precise_sell_points(df)
                    for name, info in sell_points.items():
                        with st.expander(name):
                            st.metric("預估賣點", info["預估賣點"])
                            st.caption(f"停損：{info['停損']} | 停利：{info['停利']}")
                            st.caption(f"條件：{info['條件']}")
                            st.caption(f"共振確認：{info['共振確認']} | 優先級：{info['優先級']}")
                            if "指標細節" in info:
                                with st.expander("🔍 指標細節"):
                                    st.json(info["指標細節"])

                    st.subheader("2) 指標計算 + 支撐壓力")
                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.6, 0.2, 0.2]
                    )

                    fig.add_trace(
                        go.Candlestick(
                            x=df["Date"], open=df["Open"], high=df["High"],
                            low=df["Low"], close=df["Close"], name="股價"
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="MA20"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_High"], name="布林上軌"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Low"], name="布林下軌"), row=1, col=1)

                    colors = ["red" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "green" for i in range(len(df))]
                    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="成交量", marker_color=colors), row=2, col=1)

                    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_DIF"], name="DIF"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal"), row=3, col=1)

                    fig.update_layout(height=700, title_text=f"{stock_code} 技術分析", showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("5) 指標快照（最近 10 筆）")
                    cols = ["Date", "Close", "SMA20", "BB_High", "BB_Low", "MACD_DIF", "KD_K", "KD_D", "ADX", "ATR_Pct"]
                    st.dataframe(df[cols].tail(10).round(2), use_container_width=True)

                    st.caption("⚠️ 本工具僅做分析提示，不構成投資建議；請自行評估風險。")

                except Exception as e:
                    st.error(f"錯誤：{str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.subheader("🏆 Top 10 潛力股篩選")
        st.caption("篩選條件：20 日動量 + ADX 趨勢強度 + 共振確認分數 + 量能放大")

        default_stocks = [
            "2330.TW","2317.TW","2454.TW","3163.TW","6770.TW",
            "3491.TW","2313.TW","1560.TW","3105.TW","6187.TW",
            "2395.TW","2382.TW","2353.TW","2303.TW","2308.TW",
        ]

        stock_input = st.text_area("輸入股票代碼（用逗號分隔）", ", ".join(default_stocks), height=100)
        stock_list = []
        for s in stock_input.split(","):
            t = s.strip()
            if not t:
                continue
            tu = t.upper()
            if ".TW" not in tu and ".TWO" not in tu:
                t = t + ".TW"
            stock_list.append(t)

        if st.button("開始篩選"):
            with st.spinner("分析中..."):
                top10 = screen_top_stocks(stock_list)
                if top10 is not None and not top10.empty:
                    st.dataframe(top10, use_container_width=True)
                else:
                    st.warning("未找到符合條件的股票（或資料來源被擋/不完整）")

if __name__ == "__main__":
    main()
