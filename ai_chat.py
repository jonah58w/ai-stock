# ============================================================
# AI 分析助手（每檔股票各自獨立的對話）
# ============================================================
render_chat_section(
    symbol=symbol,                    # 你既有變數：股票代碼，例如 "2330"
    name=name,                        # 你既有變數：股票名稱，例如 "台積電"
    df=df,                            # 你既有的 OHLCV+指標 DataFrame
    signals_info={                    # 把你 signals.py 算出的結果傳進來
        "A1": a1_triggered,           # ← 換成你實際的變數名
        "A2": a2_triggered,           # ← 換成你實際的變數名
    },
    fundamentals={                    # 把你 views.py 取到的基本面傳進來
        "eps": eps,                   # ← 換成你實際的變數名
        "pe": pe,
        "pb": pb,
        "roe": roe,
        "revenue_yoy": rev_yoy,
    },
    market_regime=market_regime_str,  # 你既有的 ^TWII 多空判斷字串，沒有就傳 None
)
