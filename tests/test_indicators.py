import numpy as np
import pandas as pd
import pytest

from aqlab.data import generate_synthetic_ohlcv
from aqlab.indicators import atr, ema, pct_change_n, realized_vol, rolling_zscore, rsi, sma, true_range


def test_sma_matches_rolling_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_ema_is_recursive_and_finite():
    s = pd.Series(np.arange(1, 21, dtype=float))
    out = ema(s, 5)
    assert out.notna().all()
    assert out.iloc[-1] > out.iloc[0]


def test_rsi_bounds_and_monotonic_series():
    up = pd.Series(np.arange(1, 40, dtype=float))
    r = rsi(up, 14)
    assert r.dropna().between(0, 100).all()
    assert r.iloc[-1] == 100.0

    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    r_down = rsi(down, 14)
    assert r_down.iloc[-1] == 0.0


def test_atr_positive_on_synthetic():
    df = generate_synthetic_ohlcv(n_days=120, seed=5)
    a = atr(df, 14).dropna()
    assert (a > 0).all()
    assert (true_range(df).dropna() >= 0).all()


def test_rolling_zscore_matches_manual_formula():
    s = pd.Series(np.arange(1.0, 11.0))
    z = rolling_zscore(s, 5)
    window = s.iloc[4:9]
    expected = (s.iloc[8] - window.mean()) / window.std(ddof=0)
    assert z.iloc[8] == pytest.approx(expected)
    assert z.iloc[3] != z.iloc[3]  # NaN before the window is full
    assert z.iloc[:4].isna().all()


def test_realized_vol_positive_and_scaled():
    df = generate_synthetic_ohlcv(n_days=200, seed=6)
    v = realized_vol(df["close"], 20).dropna()
    assert (v > 0).all()
    assert 0.05 < float(v.iloc[-1]) < 1.5


def test_pct_change_n_matches_manual():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = pct_change_n(s, 2)
    assert out.iloc[2] == 3.0 / 1.0 - 1.0
    assert np.isnan(out.iloc[0])
