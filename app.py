# =========================================================
# AI 台股量化專業平台 V-FINAL (No Plotly / Full Feature)
# ✅ 單股分析（買賣點 + 指標 + 布林通道 + K線）
# ✅ Top10 掃描器（含可操作買賣判斷 + 距離%）
# ✅ 逐路診斷（TWSE/TPEX/YF 哪一路失敗、為什麼）
# ✅ 多來源備援：TWSE JSON / TPEX JSON → yfinance
# ✅ 不使用 plotly（避免 Streamlit Cloud 缺套件）
# =========================================================

from __future__ import annotations

import ssl
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# yfinance 最後備援（可能被 cloud 擋）
import yfinance as yf

# 技術指標
import ta

# Altair 畫圖（通常 Streamlit Cloud 可用）
try:
    import altair as alt
    ALT_OK = True
except Exception:
    ALT_OK = False


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI 台股量化專業平台", layout="wide")
st.title("🧠 AI 台股量化專業平台（最終極全備援 + 逐路診斷 / 不自動下單 / 無 Plotly）")


# -----------------------------
# Sidebar controls
# -----------------------------
mode = st.sidebar.radio("選擇模式", ["單股分析", "Top 10 掃描器"], index=0)
period = st.sidebar.selectbox("資料期間", ["3mo", "6mo", "1y", "2y"], index=1)
show_debug = st.sidebar.checkbox("顯示下載除錯資訊（Debug）", value=False)

max_buy_gap_pct = st.sidebar.slider(
    "可操作買點最大距離（避免買點離現實太遠）",
    min_value=0.03, max_value=0.25, value=0.12, step=0.01
)

default_pool = ["2330", "2317", "2454", "2303", "2382", "3037", "8046", "4967", "6274"]


# =========================================================
# Networking helpers
# =========================================================
def _ua_headers() -> dict:
    # TWSE/TPEX 常常對無 header 或 bot 類請求回空字串
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/json,text/plain,*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }


def http_get(url: str, timeout: int = 12) -> tuple[int | None, str, str | None]:
    """
    Return: (status_code, text, error)
    """
    try:
        req = urllib.request.Request(url, headers=_ua_headers(), method="GET")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            status = getattr(r, "status", 200)
            raw = r.read()
        # 嘗試 utf-8，失敗再用 big5/latin fallback
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        return status, text, None
    except Exception as e:
        return None, "", f"{type(e).__name__}: {e}"


def net_probe(url: str) -> dict:
    status, text, err = http_get(url, timeout=8)
    return {
        "url": url,
        "status": status,
        "len": len(text or ""),
        "error": err or ""
    }


# =========================================================
# Official data sources (TWSE/TPEX)
# =========================================================
def _period_to_days(p: str) -> int:
    return {"3mo": 110, "6mo": 210, "1y": 420, "2y": 820}.get(p, 210)


def _month_starts(days: int) -> list[str]:
    """
    Return list of YYYYMMDD for month starts needed to cover days.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    cur = datetime(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append(cur.strftime("%Y%m%d"))
        # next month
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return out


def fetch_twse_month(stock_id: str, yyyymmdd: str) -> tuple[pd.DataFrame | None, dict]:
    """
    TWSE: https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=YYYYMMDD&stockNo=2330
    """
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={yyyymmdd}&stockNo={stock_id}"
    status, text, err = http_get(url)
    diag = {"source": "TWSE_JSON", "url": url, "status": status, "len": len(text), "error": err or ""}

    if err or not text:
        return None, diag

    try:
        j = json.loads(text)
        # 有些時候會回 {"stat":"很抱歉..."} 或 data 空
        if not isinstance(j, dict):
            diag["error"] = "JSON_NOT_DICT"
            return None, diag
        if j.get("stat") != "OK":
            diag["error"] = f"STAT_NOT_OK: {j.get('stat')}"
            return None, diag

        data = j.get("data", [])
        if not data:
            diag["error"] = "EMPTY_DATA"
            return None, diag

        # 欄位通常：日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
        rows = []
        for r in data:
            if len(r) < 9:
                continue
            # 日期是民國，例如 113/02/01
            roc = str(r[0]).strip()
            # 轉西元
            y, m, d = roc.split("/")
            y = int(y) + 1911
            dt = f"{y:04d}-{int(m):02d}-{int(d):02d}"

            def _to_float(x):
                s = str(x).replace(",", "").strip()
                if s in ["--", ""]:
                    return np.nan
                try:
                    return float(s)
                except Exception:
                    return np.nan

            def _to_int(x):
                s = str(x).replace(",", "").strip()
                if s in ["--", ""]:
                    return np.nan
                try:
                    return int(s)
                except Exception:
                    return np.nan

            rows.append({
                "date": dt,
                "open": _to_float(r[3]),
                "high": _to_float(r[4]),
                "low": _to_float(r[5]),
                "close": _to_float(r[6]),
                "volume": _to_int(r[1]),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).sort_values("date")
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 60:
            diag["error"] = f"TOO_SHORT({len(df)})"
            return None, diag
        return df, diag

    except Exception as e:
        diag["error"] = f"PARSE_FAIL: {type(e).__name__}: {e}"
        return None, diag


def fetch_tpex_month(stock_id: str, yyyymmdd: str) -> tuple[pd.DataFrame | None, dict]:
    """
    TPEX (上櫃) 常用：st43_result
    https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d=113/02&stkno=6274
    d 是民國年/月
    """
    y = int(yyyymmdd[:4])
    m = int(yyyymmdd[4:6])
    roc_y = y - 1911
    d_param = f"{roc_y}/{m:02d}"

    qs = urllib.parse.urlencode({"l": "zh-tw", "d": d_param, "stkno": stock_id})
    url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?{qs}"

    status, text, err = http_get(url)
    diag = {"source": "TPEX_JSON", "url": url, "status": status, "len": len(text), "error": err or ""}

    if err or not text:
        return None, diag

    try:
        j = json.loads(text)
        aa = j.get("aaData") or j.get("data") or []
        if not aa:
            diag["error"] = "EMPTY_DATA"
            return None, diag

        rows = []
        for r in aa:
            # 常見欄位：日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌, 成交筆數
            if len(r) < 9:
                continue
            roc = str(r[0]).strip()
            y2, m2, d2 = roc.split("/")
            y2 = int(y2) + 1911
            dt = f"{y2:04d}-{int(m2):02d}-{int(d2):02d}"

            def _to_float(x):
                s = str(x).replace(",", "").strip()
                if s in ["--", "", "X"]:
                    return np.nan
                try:
                    return float(s)
                except Exception:
                    return np.nan

            def _to_int(x):
                s = str(x).replace(",", "").strip()
                if s in ["--", "", "X"]:
                    return np.nan
                try:
                    return int(float(s))
                except Exception:
                    return np.nan

            rows.append({
                "date": dt,
                "open": _to_float(r[3]),
                "high": _to_float(r[4]),
                "low": _to_float(r[5]),
                "close": _to_float(r[6]),
                "volume": _to_int(r[1]),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).sort_values("date")
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 60:
            diag["error"] = f"TOO_SHORT({len(df)})"
            return None, diag
        return df, diag

    except Exception as e:
        diag["error"] = f"PARSE_FAIL: {type(e).__name__}: {e}"
        return None, diag


# =========================================================
# yfinance fallback
# =========================================================
def fetch_yfinance(stock_id: str, period: str) -> tuple[pd.DataFrame | None, dict]:
    """
    最後備援：2330.TW / 6274.TWO
    """
    diags = []
    for suffix in [".TW", ".TWO"]:
        sym = stock_id + suffix
        url = f"yfinance:{sym}:{period}"
        diag = {"source": f"YF{suffix}", "url": url, "status": None, "len": 0, "error": ""}
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=False)
            if df is None or df.empty:
                diag["error"] = "EMPTY"
                diags.append(diag)
                continue

            # MultiIndex columns fix
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [str(c).lower().strip() for c in df.columns]

            need = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in need):
                diag["error"] = f"MISSING_COLS({df.columns})"
                diags.append(diag)
                continue

            df = df[need].dropna()
            diag["len"] = len(df)
            diag["status"] = 200
            return df, diag

        except Exception as e:
            diag["error"] = f"{type(e).__name__}: {e}"
            diags.append(diag)

    # return last diag as representative
    return None, (diags[-1] if diags else {"source": "YF", "url": "yfinance", "status": None, "len": 0, "error": "ALL_FAIL"})


# =========================================================
# Master loader (TWSE/TPEX → YF)
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(stock_id: str, period: str) -> tuple[pd.DataFrame | None, str, list[dict]]:
    stock_id = stock_id.strip()
    diags: list[dict] = []
    if not stock_id:
        return None, "EMPTY", [{"source": "INPUT", "url": "", "status": None, "len": 0, "error": "EMPTY_STOCK_ID"}]

    days = _period_to_days(period)
    months = _month_starts(days)

    # 先試 TWSE（月拼接）
    parts = []
    ok_count = 0
    for ms in months:
        dfm, diag = fetch_twse_month(stock_id, ms)
        diags.append(diag)
        if dfm is not None:
            parts.append(dfm)
            ok_count += 1

    if ok_count >= 2 and parts:
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.tail(days)
        if len(df) >= 60:
            return df, "TWSE_JSON", diags

    # 再試 TPEX（月拼接）
    parts = []
    ok_count = 0
    for ms in months:
        dfm, diag = fetch_tpex_month(stock_id, ms)
        diags.append(diag)
        if dfm is not None:
            parts.append(dfm)
            ok_count += 1

    if ok_count >= 2 and parts:
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.tail(days)
        if len(df) >= 60:
            return df, "TPEX_JSON", diags

    # 最後才用 yfinance
    df, diag = fetch_yfinance(stock_id, period)
    diags.append(diag)
    if df is not None and len(df) >= 60:
        return df, diag["source"], diags

    return None, "ALL_FAIL", diags


# =========================================================
# Indicators
# =========================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = close.ewm(span=20).mean()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_up"] = bb.bollinger_hband()
    df["bb_dn"] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # KD(隨機指標)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["k"] = stoch.stoch()
    df["d"] = stoch.stoch_signal()

    # 乖離（close vs ema20）
    df["bias_ema20_pct"] = (df["close"] - df["ema20"]) / df["ema20"] * 100.0

    return df


# =========================================================
# Buy/Sell zones (fix "too far from reality")
# =========================================================
def compute_buy_sell_zones(df: pd.DataFrame, max_buy_gap_pct: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], dict]:
    df = df.copy()
    p = float(df["close"].iloc[-1])
    ema20 = float(df["ema20"].iloc[-1])
    bb_mid = float(df["bb_mid"].iloc[-1])
    bb_up = float(df["bb_up"].iloc[-1])
    bb_dn = float(df["bb_dn"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0

    low60 = float(df["low"].rolling(60).min().iloc[-1])
    high60 = float(df["high"].rolling(60).max().iloc[-1])

    # 近端可操作買點：以 EMA20 / BB_dn 附近為核心（避免離太遠）
    near_low = max(bb_dn, ema20 - 1.0 * atr)
    near_high = min(bb_mid, ema20 + 0.3 * atr)
    if near_low > near_high:
        near_low, near_high = min(near_low, near_high), max(near_low, near_high)

    gap_near = (p - ((near_low + near_high) / 2)) / p

    # 如果仍離現價太遠，硬性收斂到 EMA20 附近
    if gap_near > max_buy_gap_pct:
        near_low = ema20 - 0.7 * atr
        near_high = ema20 + 0.2 * atr
        gap_near = (p - ((near_low + near_high) / 2)) / p

    # 深回檔買點：60日低點附近（等待型）
    deep_low = max(0.0, low60 - 0.5 * atr)
    deep_high = low60 + 0.8 * atr
    gap_deep = (p - ((deep_low + deep_high) / 2)) / p

    # 近端賣點：BB_up 附近
    sell_low = max(bb_mid, bb_up - 0.3 * atr)
    sell_high = bb_up + 0.8 * atr
    gap_sell = (((sell_low + sell_high) / 2) - p) / p

    meta = {
        "price": p,
        "ema20": ema20,
        "atr": atr,
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "bb_dn": bb_dn,
        "low60": low60,
        "high60": high60,
        "gap_near": gap_near,
        "gap_deep": gap_deep,
        "gap_sell": gap_sell,
        "last_date": df.index[-1].date().isoformat(),
    }
    return (near_low, near_high), (deep_low, deep_high), (sell_low, sell_high), meta


def compute_ai_score(df: pd.DataFrame) -> int:
    """
    共振分數（簡潔但可讀）：
    - 趨勢：close > ema20 加分
    - RSI：<35 偏買、>65 偏賣
    - MACD hist：正加分
    - KD：K上穿D加分、超買扣分
    """
    close = float(df["close"].iloc[-1])
    ema20 = float(df["ema20"].iloc[-1])
    rsi = float(df["rsi"].iloc[-1])
    hist = float(df["macd_hist"].iloc[-1])
    k = float(df["k"].iloc[-1])
    d = float(df["d"].iloc[-1])

    score = 50

    # 趨勢
    if close >= ema20:
        score += 10
    else:
        score -= 10

    # RSI
    if rsi < 30:
        score += 18
    elif rsi < 40:
        score += 10
    elif rsi > 70:
        score -= 18
    elif rsi > 60:
        score -= 10

    # MACD 動能
    score += 12 if hist > 0 else -12

    # KD
    if k > d:
        score += 6
    else:
        score -= 6
    if k > 80:
        score -= 6
    if k < 20:
        score += 6

    return int(max(0, min(100, round(score))))


def decision(meta: dict, buy_near: tuple[float, float], sell_near: tuple[float, float]) -> tuple[str, str]:
    p = meta["price"]
    bl, bh = buy_near
    sl, sh = sell_near

    if bl <= p <= bh:
        return "🔥 當下位於『可操作買點區』", "價格已進入近端買點區間，可分批、設停損。"
    if sl <= p <= sh:
        return "⚠️ 當下位於『近端賣點區』", "價格逼近壓力帶，適合減碼/停利/提高警覺。"

    if abs(meta["gap_near"]) < 0.02:
        return "🟢 接近買點（2%內）", "接近近端買點區，等指標翻轉或量縮止跌。"
    if abs(meta["gap_sell"]) < 0.02:
        return "🟠 接近賣點（2%內）", "接近近端賣點區，留意轉弱與爆量。"

    return "⏳ 觀察", "尚未進入可操作買/賣區，等待價格進區或指標翻轉。"


# =========================================================
# Chart (No Plotly): Altair Kline + Bollinger + EMA + Volume
# =========================================================
def render_chart(df: pd.DataFrame, title: str = "K線 + 布林通道 + EMA20"):
    if not ALT_OK:
        st.warning("⚠️ Altair 未可用：改用簡化折線圖（close/BB/EMA）。")
        st.line_chart(df[["close", "ema20", "bb_up", "bb_mid", "bb_dn"]], use_container_width=True)
        return

    d = df.copy().reset_index().rename(columns={"index": "date"})
    d["date"] = pd.to_datetime(d["date"])

    base = alt.Chart(d).encode(
        x=alt.X("date:T", axis=alt.Axis(title=None))
    )

    # K線：用 rule + bar 模擬
    wick = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q"
    )

    body = base.mark_bar().encode(
        y="open:Q",
        y2="close:Q"
    )

    bb_up = base.mark_line().encode(y="bb_up:Q")
    bb_mid = base.mark_line().encode(y="bb_mid:Q")
    bb_dn = base.mark_line().encode(y="bb_dn:Q")
    ema20 = base.mark_line(strokeDash=[6, 3]).encode(y="ema20:Q")

    price_layer = (wick + body + bb_up + bb_mid + bb_dn + ema20).properties(height=360, title=title)

    vol = base.mark_area(opacity=0.25).encode(
        y=alt.Y("volume:Q", title="Volume")
    ).properties(height=120)

    st.altair_chart(price_layer & vol, use_container_width=True)


# =========================================================
# Diagnostics
# =========================================================
def build_diag_table(diags: list[dict]) -> pd.DataFrame:
    if not diags:
        return pd.DataFrame([{"source": "NONE", "url": "", "status": None, "len": 0, "error": "NO_DIAG"}])
    return pd.DataFrame(diags)


# =========================================================
# Single stock UI
# =========================================================
def render_single(stock_id: str):
    df, source, diags = load_price_data(stock_id, period)

    if df is None:
        st.error("❌ 無法取得資料（所有備援來源皆失敗）。")
        if show_debug:
            st.markdown("## 🧩 逐路診斷（哪一路失敗、為什麼）")
            st.dataframe(build_diag_table(diags), use_container_width=True)
            st.markdown("### 🌐 即時網路測試（參考）")
            st.write(net_probe("https://www.twse.com.tw/zh/"))
            st.write(net_probe("https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d=113/01&stkno=6274"))
            st.write(net_probe("https://finance.yahoo.com/"))
        st.stop()

    df = add_indicators(df)
    buy_near, buy_deep, sell_near, meta = compute_buy_sell_zones(df, max_buy_gap_pct)
    score = compute_ai_score(df)
    label, reason = decision(meta, buy_near, sell_near)

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.8])
    c1.metric("目前價格", f"{meta['price']:.2f}")
    c2.metric("AI 共振分數", f"{score}/100")
    c3.metric("資料來源", source)
    c4.write(f"最後日期：{meta['last_date']}｜資料筆數：{len(df)}")

    st.markdown("## 📌 當下是否為買點/賣點？（可操作判斷）")
    st.success(label)
    st.caption(f"原因：{reason}")

    st.markdown("## 🗺️ 未來預估買賣點（區間 + 距離%）")
    st.info(
        f"✅ 可操作買點（近端回檔）：{buy_near[0]:.2f} ~ {buy_near[1]:.2f} "
        f"（距離現價：約 {meta['gap_near']*100:.1f}%）"
    )
    st.info(
        f"🕰️ 深回檔買點（等待型）：{buy_deep[0]:.2f} ~ {buy_deep[1]:.2f} "
        f"（距離現價：約 {meta['gap_deep']*100:.1f}%）"
    )
    st.warning(
        f"🎯 近端賣點區（壓力/獲利）：{sell_near[0]:.2f} ~ {sell_near[1]:.2f} "
        f"（距離現價：約 {meta['gap_sell']*100:.1f}%）"
    )

    st.markdown("## 📉 布林通道分析圖（K線 + BB + EMA20 + Volume）")
    render_chart(df, title=f"{stock_id}｜{source}")

    with st.expander("📊 指標細節（RSI / MACD / KD / ATR / EMA20 / BB / 乖離）", expanded=False):
        last = df.iloc[-1]
        st.write({
            "RSI": round(float(last["rsi"]), 2),
            "MACD_HIST": round(float(last["macd_hist"]), 4),
            "K": round(float(last["k"]), 2),
            "D": round(float(last["d"]), 2),
            "ATR": round(float(last["atr"]), 2),
            "EMA20": round(float(last["ema20"]), 2),
            "BIAS_EMA20_%": round(float(last["bias_ema20_pct"]), 2),
            "BB_UP": round(float(last["bb_up"]), 2),
            "BB_MID": round(float(last["bb_mid"]), 2),
            "BB_DN": round(float(last["bb_dn"]), 2),
        })

    if show_debug:
        st.markdown("## 🧩 逐路診斷（哪一路成功/失敗、為什麼）")
        st.dataframe(build_diag_table(diags), use_container_width=True)


# =========================================================
# Top10 Scanner UI
# =========================================================
def render_top10():
    st.caption("Top10 掃描器：建議先用小池測試（避免全市場在 Cloud 超時）。")

    pool_text = st.text_area("stock_pool（每行一檔代號）", "\n".join(default_pool), height=160)
    pool = [x.strip() for x in pool_text.splitlines() if x.strip()]

    rows = []
    fail = 0

    for sid in pool:
        df, source, diags = load_price_data(sid, period)
        if df is None:
            fail += 1
            continue

        df = add_indicators(df)
        buy_near, buy_deep, sell_near, meta = compute_buy_sell_zones(df, max_buy_gap_pct)
        score = compute_ai_score(df)
        label, _ = decision(meta, buy_near, sell_near)

        rows.append({
            "股票": sid,
            "來源": source,
            "價格": round(meta["price"], 2),
            "AI分數": score,
            "可操作買點距離%": round(meta["gap_near"] * 100, 1),
            "近端賣點距離%": round(meta["gap_sell"] * 100, 1),
            "當下判斷": label,
            "近端買點": f"{buy_near[0]:.2f}~{buy_near[1]:.2f}",
            "深回檔買點": f"{buy_deep[0]:.2f}~{buy_deep[1]:.2f}",
            "近端賣點": f"{sell_near[0]:.2f}~{sell_near[1]:.2f}",
            "最後日期": meta["last_date"],
        })

    if not rows:
        st.warning("目前掃描結果為空（代表資料下載失敗）。請稍後再試或縮小 stock_pool。")
        st.stop()

    out = pd.DataFrame(rows).sort_values(["AI分數", "可操作買點距離%"], ascending=[False, True]).head(10)
    st.markdown("## 🔥 AI 強勢股 Top 10（含買賣點距離% / 當下判斷）")
    st.dataframe(out, use_container_width=True)

    st.caption(f"掃描池：{len(pool)} 檔｜成功：{len(rows)}｜失敗：{fail}")


# =========================================================
# Main
# =========================================================
if mode == "單股分析":
    stock = st.text_input("請輸入股票代號", "2330").strip()
    render_single(stock)
else:
    render_top10()
