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

__all__ = ["rsl", "kdj", "amplitude", "pct_change_1d", "white_line", "yellow_line", "ma", "sma_tdx", "brick_chart"]


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


def sma_tdx(series: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 ``SMA(X, N, M)``: ``Y_t = (M*X_t + (N-M)*Y_{t-1}) / N``，``Y_1 = X_1``。

    Equivalent to an EWMA with ``alpha = M/N`` initialised at the first value, which
    is exactly how the platform defines it (this matters for reproducing published
    formulas digit-for-digit).
    """
    if n <= 0 or m <= 0 or m > n:
        raise ValueError("require 0 < m <= n")
    return series.astype(float).ewm(alpha=m / n, adjust=False).mean()


def brick_chart(df: pd.DataFrame, threshold: float = 4.0, hhv_window: int = 4, llv_window: int = 4) -> pd.DataFrame:
    """砖型图（用户自定义公式的等价实现）。

    通达信公式::

        VAR1A := (HHV(H,4) - C) / (HHV(H,4) - LLV(L,4)) * 100 - 90
        VAR2A := SMA(VAR1A, 4, 1) + 100
        VAR3A := (C - LLV(L,4)) / (HHV(H,4) - LLV(L,4)) * 100
        VAR4A := SMA(VAR3A, 6, 1)
        VAR5A := SMA(VAR4A, 6, 1) + 100
        VAR6A := VAR5A - VAR2A
        砖型图 := IF(VAR6A > 4, VAR6A - 4, 0)
        视觉红柱 := IF(砖型图 > 昨砖型图, 砖型图 - 昨砖型图, 0)
        视觉绿柱 := IF(砖型图 < 昨砖型图, 昨砖型图 - 砖型图, 0)
        绿转红   := 昨视觉绿柱 > 0 AND 视觉红柱 > 0
        强度比   := IF(绿转红, ROUND(视觉红柱 / 昨视觉绿柱, 2), 0)
        XG       := 绿转红 AND 视觉红柱 >= 昨视觉绿柱 * 0.6667

    Returns a DataFrame with the intermediate columns (``var1a``…``var6a``), the brick
    value, the red/green bars, ``strength_ratio`` and the ``xg`` flag so every step is
    auditable.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    hhv = high.rolling(hhv_window, min_periods=1).max()
    llv = low.rolling(llv_window, min_periods=1).min()
    span = (hhv - llv).replace(0.0, np.nan)

    var1a = (hhv - close) / span * 100.0 - 90.0
    var2a = sma_tdx(var1a, 4, 1) + 100.0
    var3a = (close - llv) / span * 100.0
    var4a = sma_tdx(var3a, 6, 1)
    var5a = sma_tdx(var4a, 6, 1) + 100.0
    var6a = var5a - var2a
    brick = (var6a - threshold).where(var6a > threshold, 0.0)

    prev_brick = brick.shift(1)
    red = (brick - prev_brick).where(brick > prev_brick, 0.0)
    green = (prev_brick - brick).where(brick < prev_brick, 0.0)
    prev_green = green.shift(1)
    green_to_red = (prev_green > 0) & (red > 0)
    strength_ratio = (red / prev_green).where(green_to_red, 0.0).round(2)
    xg = green_to_red & (red >= prev_green * 0.6667)

    return pd.DataFrame(
        {
            "var1a": var1a,
            "var2a": var2a,
            "var3a": var3a,
            "var4a": var4a,
            "var5a": var5a,
            "var6a": var6a,
            "brick": brick,
            "prev_brick": prev_brick,
            "red": red,
            "green": green,
            "prev_green": prev_green,
            "green_to_red": green_to_red.fillna(False),
            "strength_ratio": strength_ratio.fillna(0.0),
            "xg": xg.fillna(False),
        },
        index=df.index,
    )


def brick_streaks(chart: pd.DataFrame) -> pd.DataFrame:
    """红砖/绿砖的连续块数（红砖第 N 块、绿砖第 N 块）。

    ``red_streak`` 在砖型图连续上升时递增，转绿清零；``green_streak`` 反之。
    砖型图不变（既非红也非绿）时保持前值（既不新增也不清零）。
    """
    red_streak: list[int] = []
    green_streak: list[int] = []
    r = g = 0
    for is_red, is_green in zip(chart["red"].to_numpy() > 0, chart["green"].to_numpy() > 0):
        if is_red:
            r += 1
            g = 0
        elif is_green:
            g += 1
            r = 0
        red_streak.append(r)
        green_streak.append(g)
    out = chart.copy()
    out["red_streak"] = red_streak
    out["green_streak"] = green_streak
    return out
