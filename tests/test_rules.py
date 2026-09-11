import numpy as np
import pandas as pd
import pytest

from aqlab.rules import (
    DEFAULT_RULE_BINDINGS,
    ActivityValueGate,
    NeedleBelowMA,
    TieredPullback,
    VolumePriceSurge,
    build_rule,
)


def frame(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)
    return df


def steady(n=80, base=100.0, step=0.3, volume=1000.0):
    """A gently rising series so trend filters and MAs are warm."""
    return frame([[base + i * step, base + i * step + 1, base + i * step - 1, base + i * step, volume] for i in range(n)])


def test_tiered_pullback_scores_shallow_dip_highest():
    df = steady()
    last = len(df) - 1
    # MA20 is computed from closes only, so piercing it via `low` does not move it
    ma_last = df["close"].rolling(20).mean().iloc[last]

    shallow = df.copy()
    shallow.loc[shallow.index[last], "low"] = ma_last * 0.99  # depth 1% -> tier 1
    rule = TieredPullback(ma_window=20, trend_window=60)
    assert rule.score(shallow).iloc[last] == pytest.approx(1.0)

    deep = df.copy()
    deep.loc[deep.index[last], "low"] = ma_last * 0.95        # depth 5% -> tier 3
    assert rule.score(deep).iloc[last] == pytest.approx(0.4)

    untouched = rule.score(df)
    assert untouched.iloc[last] == 0.0
    assert (untouched >= 0).all() and untouched.max() <= 1.0


def test_tiered_pullback_ignores_downtrend():
    df = frame([[100 - i, 101 - i, 99 - i, 100 - i, 1000.0] for i in range(80)])
    rule = TieredPullback(ma_window=20, trend_window=60)
    assert rule.score(df).sum() == 0


def test_needle_below_ma_detects_long_lower_shadow():
    df = steady()
    last = len(df) - 1
    ma = df["close"].rolling(20).mean().iloc[last]
    df.loc[df.index[last], "low"] = ma * 0.95      # pierce below MA
    df.loc[df.index[last], "close"] = ma * 1.02    # close back above MA with a long shadow
    df.loc[df.index[last], "open"] = ma * 1.01
    rule = NeedleBelowMA(ma_window=20, min_shadow_ratio=0.01, full_score_ratio=0.05)
    scores = rule.score(df)
    assert scores.iloc[last] > 0
    assert scores.iloc[last] <= 1.0
    assert scores.iloc[last - 1] == 0


def test_needle_requires_close_recovery():
    df = steady()
    last = len(df) - 1
    ma = df["close"].rolling(20).mean().iloc[last]
    df.loc[df.index[last], "low"] = ma * 0.9
    df.loc[df.index[last], "close"] = ma * 0.9   # never recovers -> no signal
    rule = NeedleBelowMA(ma_window=20)
    assert rule.score(df).iloc[last] == 0


def test_volume_price_surge_single_day_and_confirmation():
    df = steady()
    last = len(df) - 1
    df.loc[df.index[last], "close"] = df["close"].iloc[last - 1] * 1.06
    df.loc[df.index[last], "volume"] = 5000.0

    v1 = VolumePriceSurge(min_price_change=0.03, volume_multiple=1.5, confirm_days=1)
    assert v1.score(df).iloc[last] > 0

    v3 = VolumePriceSurge(min_price_change=0.03, volume_multiple=1.5, confirm_days=3)
    assert v3.score(df).iloc[last] == 0  # only one bar of confirmation


def test_activity_value_gate_hysteresis():
    up = frame([[100 + i, 101 + i, 99 + i, 100 + i, 1000.0 + 10 * i] for i in range(80)])
    gate = ActivityValueGate(fast_window=5, slow_window=20, on_threshold=0.02, off_threshold=-0.01)
    series = gate.gate_series({"A": up})
    assert series.iloc[-1] == 1
    assert gate.state_at({"A": up}) == 1

    down = frame([[100 - 0.5 * i, 101 - 0.5 * i, 99 - 0.5 * i, 100 - 0.5 * i, 1000.0 - 5 * i] for i in range(80)])
    assert gate.state_at({"A": down}) == 0

    # states are only 0/1
    assert set(gate.gate_series({"A": up}).unique()) <= {0, 1}


def test_gate_rejects_bad_windows():
    with pytest.raises(ValueError):
        ActivityValueGate(fast_window=20, slow_window=5)
    with pytest.raises(ValueError):
        ActivityValueGate(on_threshold=-0.01, off_threshold=0.02)


def test_build_rule_registry_and_errors():
    rule = build_rule("needle_below_ma", ma_window=30)
    assert rule.params["ma_window"] == 30
    with pytest.raises(KeyError):
        build_rule("nope")

    assert len(DEFAULT_RULE_BINDINGS) == 3
    weights = [w for _n, _p, w in DEFAULT_RULE_BINDINGS]
    assert all(w > 0 for w in weights)
