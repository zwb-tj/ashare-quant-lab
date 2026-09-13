"""Indicator library. Pure pandas/numpy, no lookahead by construction.

Every function here is *causal*: the value at row ``t`` only uses information
available up to and including ``t``. Shifting for execution is handled once,
in :mod:`aqlab.backtest`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "donchian",
    "ema",
    "pct_change_n",
    "realized_vol",
    "rolling_zscore",
    "rsi",
    "sma",
    "true_range",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.astype(float).rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (adjust=False for a recursive definition)."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.astype(float).ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, bounded to [0, 100]."""
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0, 100.0).clip(0.0, 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    high, low = df["high"].astype(float), df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    ranges = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing)."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """(x - rolling mean) / rolling std, causal."""
    s = series.astype(float)
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def donchian(df: pd.DataFrame, window: int = 20):
    """Donchian channel computed on data up to and including the current bar.

    Returns ``(upper, lower)`` as a tuple of Series. Note that a breakout
    strategy must compare *today's close* against the channel built from
    *previous* bars; use ``.shift(1)`` when defining such a rule.
    """
    high = df["high"].astype(float).rolling(window, min_periods=window).max()
    low = df["low"].astype(float).rolling(window, min_periods=window).min()
    return high, low


def realized_vol(series: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized realized volatility of daily returns."""
    returns = series.astype(float).pct_change()
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    """N-bar momentum as a simple return: ``close_t / close_{t-n} - 1``."""
    s = series.astype(float)
    return s / s.shift(n) - 1.0
