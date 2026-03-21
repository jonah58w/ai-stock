from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 基本工具
# =========================================================
def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lowered = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        hit = lowered.get(str(name).lower())
        if hit is not None:
            return hit
    return None


def _ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    rename_map = {}
    for c in work.columns:
        cl = str(c).strip().lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl in ("adj close", "adj_close", "adjclose"):
            rename_map[c] = "Adj Close"
        elif cl == "volume":
            rename_map[c] = "Volume"

    if rename_map:
        work = work.rename(columns=rename_map)

    return work


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _calc_bollinger(
    close: pd.Series,
    length: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    mid = _sma(close, length)
    std = close.rolling(length, min_periods=length).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    width = (upper - lower) / mid.replace(0, np.nan)
    pb = (close - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_middle": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percent_b": pb,
        }
    )


def _add_bollinger_metrics(
    df: pd.DataFrame,
    length: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    work = _ensure_ohlcv_columns(df)
    if "Close" not in work.columns:
        raise ValueError("缺少 Close 欄位，無法繪製布林圖")

    bb = _calc_bollinger(work["Close"], length=length, num_std=num_std)
    for c in bb.columns:
        work[c] = bb[c]

    # 中軌斜率
    work["bb_mid_slope"] = work["bb_middle"].diff()

    # squeeze 門檻：用過去 120 根帶寬的 15 百分位
    rolling_q = work["bb_width"].rolling(120, min_periods=30).quantile(0.15)
    work["bb_width_q15"] = rolling_q
    work["bb_squeeze"] = work["bb_width"] <= work["bb_width_q15"]

    # 上下軌相對位置
    work["near_upper"] = (
        (work["bb_upper"] > 0)
        & (((work["bb_upper"] - work["Close"]) / work["bb_upper"]) <= 0.03)
    )
    work["near_lower"] = (
        (work["bb_lower"] > 0)
        & (((work["Close"] - work["bb_lower"]) / work["bb_lower"]) <= 0.03)
    )

    # 沿上軌 / 沿下軌
    work["ride_upper_count_5"] = work["near_upper"].rolling(5, min_periods=1).sum()
    work["ride_lower_count_5"] = work["near_lower"].rolling(5, min_periods=1).sum()

    work["ride_upper"] = (
        (work["ride_upper_count_5"] >= 3)
        & (work["bb_mid_slope"] > 0)
        & (work["Close"] >= work["bb_middle"])
    )

    work["ride_lower"] = (
        (work["ride_lower_count_5"] >= 3)
        & (work["bb_mid_slope"] < 0)
        & (work["Close"] <= work["bb_middle"])
    )

    # 過熱 / 轉弱
    work["bb_overheat"] = (
        (work["bb_percent_b"] >= 0.95)
        & (work["bb_width"] > work["bb_width"].rolling(20, min_periods=10).mean())
        & (work["Close"] > work["bb_middle"])
    )

    work["bb_weakening"] = (
        (work["Close"] < work["bb_middle"])
        & (work["bb_mid_slope"] < 0)
        & (work["bb_percent_b"] < 0.5)
    )

    # 布林狀態
    def _state(row) -> str:
        if bool(row.get("bb_squeeze", False)):
            return "收縮整理"
        if bool(row.get("ride_upper", False)):
            return "沿上軌強攻"
        if bool(row.get("bb_overheat", False)):
            return "高檔過熱"
        if bool(row.get("bb_weakening", False)):
            return "中軌失守轉弱"
        if bool(row.get("ride_lower", False)):
            return "沿下軌走弱"
        if row.get("bb_mid_slope", 0) > 0 and row.get("Close", np.nan) >= row.get("bb_middle", np.nan):
            return "中軌上彎轉強"
        if row.get("bb_mid_slope", 0) < 0 and row.get("Close", np.nan) < row.get("bb_middle", np.nan):
            return "中軌下彎偏弱"
        return "盤整觀察"

    work["bb_state"] = work.apply(_state, axis=1)

    return work


# =========================================================
# 回測權益曲線
# =========================================================
def equity_curve_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if equity_df is None or equity_df.empty:
        fig.update_layout(
            template="plotly_white",
            title="Equity Curve",
            height=420,
        )
        return fig

    work = equity_df.copy()

    x = work.index
    y_col = None
    for c in ["equity", "Equity", "strategy_equity", "cum_equity", "portfolio_value"]:
        if c in work.columns:
            y_col = c
            break

    if y_col is None:
        if len(work.columns) >= 1:
            y_col = work.columns[0]
        else:
            fig.update_layout(template="plotly_white", title="Equity Curve", height=420)
            return fig

    fig.add_trace(
        go.Scatter(
            x=x,
            y=work[y_col],
            mode="lines",
            name="Equity",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Equity Curve",
        height=420,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    return fig


# =========================================================
# 升級版布林通道圖
# =========================================================
def bollinger_trend_chart(
    df: pd.DataFrame,
    title: str = "布林通道趨勢圖",
    bb_length: int = 20,
    bb_std: float = 2.0,
    show_volume: bool = False,
    max_bars: int = 180,
) -> go.Figure:
    """
    主圖：K線 + 布林通道
    副圖1：BandWidth
    副圖2：%B
    自動標示：squeeze / 沿上軌 / 過熱 / 轉弱
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=title, height=900)
        return fig

    work = _add_bollinger_metrics(df, length=bb_length, num_std=bb_std)
    work = work.tail(max_bars).copy()

    # 準備欄位
    if "Open" not in work.columns or "High" not in work.columns or "Low" not in work.columns or "Close" not in work.columns:
        raise ValueError("缺少 OHLC 欄位，無法繪製 K 線圖")

    # 主圖 + 2 副圖
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.18, 0.18],
        subplot_titles=(
            title,
            "BandWidth",
            "%B",
        ),
    )

    x = work.index

    # -------------------------
    # 主圖：K線 + 布林
    # -------------------------
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=work["Open"],
            high=work["High"],
            low=work["Low"],
            close=work["Close"],
            name="K線",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["bb_upper"],
            mode="lines",
            name="上軌",
            line=dict(width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["bb_middle"],
            mode="lines",
            name="中軌",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["bb_lower"],
            mode="lines",
            name="下軌",
            line=dict(width=1.5),
            fill="tonexty",
            fillcolor="rgba(100,100,180,0.10)",
        ),
        row=1,
        col=1,
    )

    # 可選成交量
    if show_volume and "Volume" in work.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=work["Volume"],
                name="Volume",
                opacity=0.25,
                yaxis="y4",
            ),
            row=1,
            col=1,
        )

    # -------------------------
    # 副圖1：BandWidth
    # -------------------------
    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["bb_width"],
            mode="lines",
            name="BandWidth",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    if "bb_width_q15" in work.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=work["bb_width_q15"],
                mode="lines",
                name="Squeeze門檻",
                line=dict(width=1, dash="dot"),
            ),
            row=2,
            col=1,
        )

    # -------------------------
    # 副圖2：%B
    # -------------------------
    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["bb_percent_b"],
            mode="lines",
            name="%B",
            line=dict(width=2),
        ),
        row=3,
        col=1,
    )

    # %B 參考線
    for y, dash, name in [(0.0, "dot", "%B=0"), (0.5, "dash", "%B=0.5"), (1.0, "dot", "%B=1")]:
        fig.add_hline(
            y=y,
            line_dash=dash,
            line_width=1,
            row=3,
            col=1,
        )

    # -------------------------
    # 自動標示
    # -------------------------
    marker_specs = [
        ("bb_squeeze", "Squeeze", "收縮整理"),
        ("ride_upper", "沿上軌", "沿上軌強攻"),
        ("bb_overheat", "過熱", "高檔過熱"),
        ("bb_weakening", "轉弱", "中軌失守轉弱"),
    ]

    for flag_col, short_text, hover_text in marker_specs:
        if flag_col not in work.columns:
            continue

        sub = work[work[flag_col].fillna(False)].copy()
        if sub.empty:
            continue

        # 標示放在主圖上
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=sub["Close"],
                mode="markers+text",
                text=[short_text] * len(sub),
                textposition="top center",
                name=short_text,
                marker=dict(size=8),
                hovertext=[
                    f"{hover_text}<br>Close={c:.2f}<br>狀態={st}"
                    for c, st in zip(sub["Close"], sub["bb_state"])
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # 最新狀態註解
    last = work.iloc[-1]
    latest_state = str(last.get("bb_state", ""))
    latest_pb = last.get("bb_percent_b", np.nan)
    latest_bw = last.get("bb_width", np.nan)
    latest_close = last.get("Close", np.nan)

    annotate_text = (
        f"最新狀態：{latest_state}"
        f"<br>Close：{latest_close:.2f}" if pd.notna(latest_close) else f"最新狀態：{latest_state}"
    )
    if pd.notna(latest_pb):
        annotate_text += f"<br>%B：{latest_pb:.2f}"
    if pd.notna(latest_bw):
        annotate_text += f"<br>BandWidth：{latest_bw:.4f}"

    fig.add_annotation(
        x=work.index[-1],
        y=work["High"].max() if "High" in work.columns else work["Close"].max(),
        text=annotate_text,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        xshift=-10,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(80,80,80,0.4)",
        row=1,
        col=1,
    )

    # Hover 顯示更多資料
    customdata = np.stack(
        [
            work["bb_middle"].fillna(np.nan),
            work["bb_upper"].fillna(np.nan),
            work["bb_lower"].fillna(np.nan),
            work["bb_width"].fillna(np.nan),
            work["bb_percent_b"].fillna(np.nan),
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=work["Close"],
            mode="lines",
            line=dict(width=0),
            opacity=0,
            showlegend=False,
            hovertemplate=(
                "日期：%{x}<br>"
                "Close：%{y:.2f}<br>"
                "中軌：%{customdata[0]:.2f}<br>"
                "上軌：%{customdata[1]:.2f}<br>"
                "下軌：%{customdata[2]:.2f}<br>"
                "BandWidth：%{customdata[3]:.4f}<br>"
                "%B：%{customdata[4]:.2f}<extra></extra>"
            ),
            customdata=customdata,
        ),
        row=1,
        col=1,
    )

    # 版面
    fig.update_layout(
        template="plotly_white",
        height=980,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=30, r=30, t=80, b=30),
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="帶寬", row=2, col=1)
    fig.update_yaxes(title_text="%B", row=3, col=1)

    return fig


# =========================================================
# 相容別名
# =========================================================
def bollinger_chart(
    df: pd.DataFrame,
    title: str = "布林通道趨勢圖",
) -> go.Figure:
    return bollinger_trend_chart(df=df, title=title)


def bb_chart(
    df: pd.DataFrame,
    title: str = "布林通道趨勢圖",
) -> go.Figure:
    return bollinger_trend_chart(df=df, title=title)
