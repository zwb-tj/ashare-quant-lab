"""Extra indicators required by the personal strategy set.

These are the standard formulas behind the rule set in :mod:`aqlab.rules_zgnb`:

* ``rsl``          — 相对强度定位 ``100*(C-LLV(L,N))/(HHV(C,N)-LLV(L,N))``
* ``kdj``          — KDJ(9,3,3)，``J = 3K - 2D``，K/D 初值 50，平滑 ``2/3 前值 + 1/3 新值``
* ``amplitude``    — 当日振幅 ``(high - low) / 前收盘``
* ``white_line``   — 白线 ``EMA(EMA(close, 10), 10)``
* ``yellow_line``  — 大哥线（黄线）``(MA14 + MA28 + MA57 + MA114) / 4``

All functions are causal: the value at bar ``t`` uses only bars up to ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rsl", "kdj", "amplitude", "pct_change_1d", "white_line", "yellow_line", "ma"]


def ma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average (causal)."""
    return series.astype(float).rolling(window, min_periods=window).mean()


def rsl(df: pd.DataFrame, window: int) -> pd.Series:
    """Relative strength location.

    ``100 * (close_t - LLV(low, N)) / (HHV(close, N) - LLV(low, N))``

    Note the deliberate asymmetry: the low extreme uses ``low`` while the high
    extreme uses ``close`` (the classic formula). Returns 50 when the range is
    degenerate or history is insufficient.
    """
    close = df["close"].astype(float)
    low = df["low"].astype(float)
    llv = low.rolling(window, min_periods=window).min()
    hhv = close.rolling(window, min_periods=window).max()
    span = (hhv - llv).replace(0.0, np.nan)
    out = (close - llv) / span * 100.0
    return out.fillna(50.0)


def kdj(df: pd.DataFrame, period: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """KDJ oscillator as a ``(k, d, j)`` DataFrame.

    ``RSV = (C - LLV(L, n)) / (HHV(H, n) - LLV(L, n)) * 100``;
    ``K = (k_smooth-1)/k_smooth * K_prev + 1/k_smooth * RSV`` (same for D on K);
    ``J = 3K - 2D``. The recursion starts from K = D = 50.
    """
    close = df["close"].astype(float)
    low = df["low"].astype(float)
    high = df["high"].astype(float)

    llv = low.rolling(period, min_periods=period).min()
    hhv = high.rolling(period, min_periods=period).max()
    span = (hhv - llv).replace(0.0, np.nan)
    rsv = ((close - llv) / span * 100.0).fillna(50.0)

    k_values: list[float] = []
    d_values: list[float] = []
    k = d = 50.0
    a_k, a_d = (k_smooth - 1) / k_smooth, (d_smooth - 1) / d_smooth
    for value in rsv.to_numpy():
        k = a_k * k + (1 - a_k) * value
        d = a_d * d + (1 - a_d) * k
        k_values.append(k)
        d_values.append(d)

    out = pd.DataFrame({"k": k_values, "d": d_values}, index=df.index)
    out["j"] = 3 * out["k"] - 2 * out["d"]
    return out


def amplitude(df: pd.DataFrame) -> pd.Series:
    """Intraday amplitude relative to the previous close."""
    prev_close = df["close"].astype(float).shift(1)
    return (df["high"].astype(float) - df["low"].astype(float)) / prev_close


def pct_change_1d(df: pd.DataFrame) -> pd.Series:
    """Same-day close-to-close return (fraction, not percent)."""
    return df["close"].astype(float).pct_change()


def white_line(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """白线: double-smoothed EMA of the close."""
    close = df["close"].astype(float)
    first = close.ewm(span=window, adjust=False).mean()
    return first.ewm(span=window, adjust=False).mean()


def yellow_line(df: pd.DataFrame, windows: tuple[int, ...] = (14, 28, 57, 114)) -> pd.Series:
    """大哥线（黄线）: average of several long moving averages."""
    close = df["close"].astype(float)
    return sum(ma(close, w) for w in windows) / len(windows)
