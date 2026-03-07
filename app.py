# app.py
# AI Stock Trading Assistant (V11.1 - Production Ready)
# 修復重點：單一股票頁基本面抓取邏輯強化 (FinMind + Yahoo Fallback)
# 適用對象：台股 (2330, 3388 等)

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import Dict, List, Optional, Union
import warnings
import time

# 技術指標庫
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# 忽略警告
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 全局設定與輔助函式 (Utils & Helpers)
# =============================================================================

def safe_float(val, default=np.nan) -> float:
    """安全轉 float，處理 None / str / np.nan"""
    try:
        if val is None:
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def normalize_symbol(stock_id: str) -> str:
    """將台股代號轉為 FinMind 格式 (例：2330 → 2330)"""
    return str(stock_id).strip().upper().split(".")[0]

def yahoo_symbol_candidates(stock_id: str, market_type: Optional[str] = None) -> List[str]:
    """
    產生 Yahoo Finance 可能的代碼候補
    優先順序：使用者指定尾碼 → 自動判斷 (.TW/.TWO) → 純代碼
    """
    sid = str(stock_id).strip().upper()
    if "." in sid:
        return [sid]
    # 自動判斷：6/4 開頭多為上櫃 (.TWO)，其他為上市 (.TW)
    if sid.startswith(("6", "4")):
        return [f"{sid}.TWO", f"{sid}.TW"]
    return [f"{sid}.TW", f"{sid}.TWO"]

def finmind_get(dataset: str, data_id: str, start_date: str, **kwargs) -> pd.DataFrame:
    """
    包裝 FinMind API 呼叫，支援 token 與錯誤處理
    請確認你已安裝 FinMind 並設定 st.secrets["FINMIND_TOKEN"]
    """
    try:
        from FinMind import data as fm_data
        # 嘗試從 secrets 讀取 token，若無則傳 None (FinMind 部分資料免登入)
        token = st.secrets.get("FINMIND_TOKEN", "") if hasattr(st, "secrets") else ""
        return fm_data.data(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            token=token if token else None,
            **kwargs
        )
    except Exception:
        return pd.DataFrame()

# =============================================================================
# 2. 核心修復：多源基本面抓取 (Load Fundamental)
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_fundamental(stock_id: str, market_type: Optional[str] = None) -> Dict[str, object]:
    """
    [V11.1 修復版] 多源基本面抓取：FinMind → Yahoo → 自動計算殖利率
    專為解決 2330 等 Yahoo info 回傳不穩的問題
    """
    out = {
        "pe": np.nan, "pb": np.nan, "eps": np.nan, "roe": np.nan,
        "dividend": np.nan, "yield": np.nan, "symbol_used": "",
    }

    # ── 1) 先試 FinMind 單檔 PER / PBR / 殖利率 ──────────────────
    try:
        per_df = finmind_get(
            "TaiwanStockPER",
            data_id=normalize_symbol(stock_id),
            start_date=(date.today() - timedelta(days=60)).strftime("%Y-%m-%d"),
        )
        if not per_df.empty:
            if "date" in per_df.columns:
                per_df["date"] = pd.to_datetime(per_df["date"], errors="coerce")
                per_df = per_df.sort_values("date")
            row = per_df.iloc[-1]
            if "PER" in per_df.columns:
                out["pe"] = safe_float(row.get("PER"), np.nan)
            if "PBR" in per_df.columns:
                out["pb"] = safe_float(row.get("PBR"), np.nan)
            if "dividend_yield" in per_df.columns:
                out["yield"] = safe_float(row.get("dividend_yield"), np.nan)
    except Exception:
        pass  # FinMind 失敗無妨，繼續下一層

    # ── 2) 再試 FinMind 單檔股利（現金股利加總） ──────────────────
    try:
        div_df = finmind_get(
            "TaiwanStockDividend",
            data_id=normalize_symbol(stock_id),
            start_date=(date.today() - timedelta(days=1100)).strftime("%Y-%m-%d"),
        )
        if not div_df.empty:
            if "date" in div_df.columns:
                div_df["date"] = pd.to_datetime(div_df["date"], errors="coerce")
                div_df = div_df.sort_values("date")
            # 確保欄位存在
            for col in ["CashEarningsDistribution", "CashStatutorySurplus"]:
                if col not in div_df.columns:
                    div_df[col] = 0
            div_df["現金股利"] = (
                pd.to_numeric(div_df["CashEarningsDistribution"], errors="coerce").fillna(0)
                + pd.to_numeric(div_df["CashStatutorySurplus"], errors="coerce").fillna(0)
            )
            row = div_df.iloc[-1]
            out["dividend"] = safe_float(row.get("現金股利"), np.nan)
    except Exception:
        pass

    # ── 3) 用 Yahoo 補足空缺欄位 (PE/PB/EPS/ROE/股利/殖利率) ─────
    for ysym in yahoo_symbol_candidates(stock_id, market_type):
        try:
            import yfinance as yf
            tk = yf.Ticker(ysym)
            info = tk.info or {}

            if pd.isna(out["pe"]):
                pe = safe_float(info.get("trailingPE"), np.nan)
                if pd.isna(pe):
                    pe = safe_float(info.get("forwardPE"), np.nan)
                out["pe"] = pe

            if pd.isna(out["pb"]):
                out["pb"] = safe_float(info.get("priceToBook"), np.nan)

            if pd.isna(out["eps"]):
                out["eps"] = safe_float(info.get("trailingEps"), np.nan)

            if pd.isna(out["roe"]):
                roe = info.get("returnOnEquity")
                if roe is not None:
                    out["roe"] = safe_float(roe, np.nan) * 100  # 轉為百分比

            if pd.isna(out["dividend"]):
                out["dividend"] = safe_float(info.get("dividendRate"), np.nan)

            if pd.isna(out["yield"]):
                dy = info.get("dividendYield")
                if dy is not None:
                    out["yield"] = safe_float(dy, np.nan) * 100  # 轉為百分比

            out["symbol_used"] = ysym
            break  # 成功即跳出
        except Exception:
            continue  # 嘗試下一個 Yahoo 代碼

    return out

# =============================================================================
# 3. 股價下載與指標計算 (Price & Indicators)
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def download_ohlc(stock_id: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """下載 OHLCV 資料，支援 .TW/.TWO 自動 fallback"""
    import yfinance as yf
    
    candidates = yahoo_symbol_candidates(stock_id)
    for sym in candidates:
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                # 處理 MultiIndex 欄位 (yfinance 新版)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # 確保欄位存在
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan
                return df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception:
            continue
    return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標"""
    if df.empty: return df
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    # 均線
    df["SMA20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    
    # RSI
    df["RSI14"] = RSIIndicator(close=close, window=14).rsi()
    
    # 布林通道
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    
    # ATR
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

# =============================================================================
# 4. UI 與主程式 (Streamlit App)
# =============================================================================

def main():
    st.set_page_config(page_title="AI Stock Analyzer V11.1", layout="wide")
    st.title("📈 台股 AI 分析系統 (V11.1)")
    
    # Sidebar: 輸入控制
    with st.sidebar:
        st.header("🔍 股票查詢")
        stock_input = st.text_input("輸入代號 (例: 2330)", value="2330")
        run_btn = st.button("🚀 開始分析", type="primary")
        st.divider()
        st.info("💡 **V11.1 更新**:\n- 修復 2330 價值分析空白問題\n- 強化 FinMind + Yahoo 雙源備援")
    
    # 主邏輯
    if run_btn or stock_input:
        stock_id = stock_input.strip()
        if not stock_id:
            st.warning("⚠️ 請輸入股票代號")
            return
            
        with st.spinner(f"🔄 正在分析 {stock_id} ..."):
            # 1. 下載股價
            df = download_ohlc(stock_id)
            if df.empty:
                st.error(f"❌ 無法下載 {stock_id} 的股價資料，請檢查網路或代號")
                return
            
            # 2. 計算指標
            df = add_indicators(df)
            current_price = float(df["Close"].iloc[-1])
            
            # 3. [關鍵修復] 抓取基本面 + 殖利率兜底計算
            fund = load_fundamental(stock_id)
            
            # 殖利率自動補算：若 Yahoo/FinMind 都沒回傳，但股利有值 → 用 股利/股價 自算
            dy = fund["yield"]
            if pd.isna(dy) and not pd.isna(fund["dividend"]) and current_price > 0:
                dy = fund["dividend"] / current_price * 100
                fund["yield"] = dy  # 寫回 dict，供後續顯示使用

            # 4. 顯示結果
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"{stock_id} 股價走勢")
                # 簡單 K 線圖
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'
                )])
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name='SMA20'))
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 價值分析")
                # 顯示價值數據 (使用修復後的 fund)
                st.metric("EPS (元)", f"{fund['eps']:.2f}" if not pd.isna(fund['eps']) else "-")
                st.metric("本益比 (PE)", f"{fund['pe']:.2f}" if not pd.isna(f['pe']) else "-")
                st.metric("股價淨值比 (PB)", f"{fund['pb']:.2f}" if not pd.isna(fund['pb']) else "-")
                st.metric("ROE (%)", f"{fund['roe']:.2f}" if not pd.isna(fund['roe']) else "-")
            
            with col3:
                st.subheader("💰 股利與殖利率")
                st.metric("現金股利 (元)", f"{fund['dividend']:.2f}" if not pd.isna(fund['dividend']) else "-")
                # 顯示殖利率 (含自動補算結果)
                yield_val = fund['yield']
                if not pd.isna(yield_val):
                    st.metric("預估殖利率 (%)", f"{yield_val:.2f}")
                else:
                    st.metric("預估殖利率 (%)", "-")
                
                st.caption(f"📡 數據來源: {fund['symbol_used'] or 'FinMind/Yahoo'}")

if __name__ == "__main__":
    main()
