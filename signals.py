from __future__ import annotations

import pandas as pd


def _score_row(row, dy, pe, pb, roe):
    score = 0

    close = row.get("Close")
    sma20 = row.get("SMA20")
    sma60 = row.get("SMA60")
    rsi = row.get("RSI")
    macd = row.get("MACD")
    macd_signal = row.get("MACD_Signal")

    if pd.notna(close) and pd.notna(sma20) and close > sma20:
        score += 20
    if pd.notna(sma20) and pd.notna(sma60) and sma20 > sma60:
        score += 20
    if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
        score += 20
    if pd.notna(rsi) and 45 <= rsi <= 70:
        score += 15

    if pd.notna(dy):
        if dy >= 6:
            score += 15
        elif dy >= 4:
            score += 10
        elif dy >= 2:
            score += 5

    if pd.notna(roe):
        if roe >= 15:
            score += 10
        elif roe >= 8:
            score += 5

    if pd.notna(pe) and pe > 0:
        if pe <= 20:
            score += 5

    if pd.notna(pb) and pb > 0:
        if pb <= 3:
            score += 5

    return min(score, 100)


def backtest_score_strategy(df, dy, pe, pb, roe, buy_threshold=70, sell_threshold=50):
    if df is None or df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "cum_return": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": pd.DataFrame(),
        }

    data = df.copy().reset_index(drop=True)

    data["score"] = data.apply(lambda r: _score_row(r, dy, pe, pb, roe), axis=1)
    data["ret"] = data["Close"].pct_change().fillna(0)

    position = 0
    entry_price = None
    trade_returns = []
    strategy_rets = []

    for i in range(len(data)):
        score = data.loc[i, "score"]
        ret = data.loc[i, "ret"]
        close = data.loc[i, "Close"]

        if position == 0 and score >= buy_threshold:
            position = 1
            entry_price = close
            strategy_rets.append(0.0)
        elif position == 1 and score <= sell_threshold:
            if entry_price and entry_price > 0 and pd.notna(close):
                trade_returns.append((close - entry_price) / entry_price)
            position = 0
            entry_price = None
            strategy_rets.append(0.0)
        else:
            strategy_rets.append(ret if position == 1 else 0.0)

    data["strategy_ret"] = strategy_rets
    data["equity"] = (1 + data["strategy_ret"]).cumprod()

    if position == 1 and entry_price and entry_price > 0:
        final_close = data.iloc[-1]["Close"]
        if pd.notna(final_close):
            trade_returns.append((final_close - entry_price) / entry_price)

    cum_return = float(data["equity"].iloc[-1] - 1) if not data.empty else 0.0
    running_max = data["equity"].cummax()
    max_drawdown = float((data["equity"] / running_max - 1).min()) if not data.empty else 0.0

    wins = sum(1 for x in trade_returns if x > 0)
    trades = len(trade_returns)
    win_rate = (wins / trades * 100) if trades > 0 else 0.0

    equity_curve = data.copy()
    keep_cols = [c for c in ["Date", "Close", "score", "strategy_ret", "equity"] if c in equity_curve.columns]
    equity_curve = equity_curve[keep_cols].copy()

    return {
        "trades": trades,
        "win_rate": win_rate,
        "cum_return": cum_return * 100,
        "max_drawdown": max_drawdown * 100,
        "equity_curve": equity_curve,
    }
