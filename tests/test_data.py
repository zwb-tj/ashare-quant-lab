import numpy as np
import pandas as pd
import pytest

from aqlab.data import generate_synthetic_ohlcv, make_universe, normalize_ohlcv


def test_synthetic_shape_and_columns():
    df = generate_synthetic_ohlcv(n_days=200, seed=1)
    assert len(df) == 200
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_synthetic_ohlc_consistency():
    df = generate_synthetic_ohlcv(n_days=300, seed=2)
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] > 0).all()


def test_synthetic_is_deterministic():
    a = generate_synthetic_ohlcv(n_days=120, seed=42)
    b = generate_synthetic_ohlcv(n_days=120, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = generate_synthetic_ohlcv(n_days=120, seed=43)
    assert not np.allclose(a["close"], c["close"])


def test_normalize_ohlcv_renames_and_sorts():
    raw = pd.DataFrame(
        {
            "日期": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "开盘": [10.0, 9.0],
            "最高": [11.0, 9.5],
            "最低": [9.5, 8.5],
            "收盘": [10.5, 9.2],
            "成交量": [100, 200],
        }
    )
    df = normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df["close"].iloc[1] == 10.5


def test_normalize_ohlcv_missing_columns_raises():
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame({"date": ["2024-01-02"], "close": [1.0]}))


def test_make_universe_is_stable():
    u1 = make_universe(n_symbols=5, n_days=180, seed=3)
    u2 = make_universe(n_symbols=5, n_days=180, seed=3)
    assert list(u1) == ["SYN001", "SYN002", "SYN003", "SYN004", "SYN005"]
    pd.testing.assert_frame_equal(u1["SYN003"], u2["SYN003"])
