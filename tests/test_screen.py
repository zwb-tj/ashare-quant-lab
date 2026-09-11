import pandas as pd
import pytest

from aqlab.data import generate_synthetic_ohlcv, make_universe
from aqlab.screen import ScreenConfig, factor_table, rank_universe


def test_rank_universe_is_deterministic_and_ordered():
    universe = make_universe(n_symbols=12, n_days=400, seed=11)
    a = rank_universe(universe)
    b = rank_universe(universe)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["rank"]) == list(range(1, len(a) + 1))
    assert a["score"].is_monotonic_decreasing


def test_top_n_respected_and_columns_present():
    universe = make_universe(n_symbols=20, n_days=400, seed=11)
    table = rank_universe(universe, config=ScreenConfig(top_n=5, min_history=130))
    assert {"rank", "symbol", "close", "mom_20", "mom_60", "trend_gap", "vol_20", "rsi_14", "score"}.issubset(table.columns)
    assert len(table) == 20  # ranking returns the whole eligible universe
    assert table.iloc[0]["rank"] == 1


def test_min_history_filters_short_symbols():
    universe = {
        "LONG": generate_synthetic_ohlcv(n_days=400, seed=1),
        "SHORT": generate_synthetic_ohlcv(n_days=60, seed=2),
    }
    table = rank_universe(universe, config=ScreenConfig(min_history=130))
    assert list(table["symbol"]) == ["LONG"]


def test_as_of_uses_no_future_data():
    universe = make_universe(n_symbols=6, n_days=400, seed=11)
    full = factor_table(universe)
    early_stamp = list(universe.values())[0].index[250]
    early = factor_table(universe, as_of=early_stamp)
    assert (early["date"] <= early_stamp).all()
    assert not full["date"].equals(early["date"])
    assert (early["close"] != full["close"]).any()


def test_no_eligible_symbol_raises():
    universe = {"ONLY": generate_synthetic_ohlcv(n_days=50, seed=1)}
    with pytest.raises(ValueError):
        rank_universe(universe, config=ScreenConfig(min_history=130))


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        ScreenConfig(min_history=10)
    with pytest.raises(ValueError):
        ScreenConfig(top_n=0)
