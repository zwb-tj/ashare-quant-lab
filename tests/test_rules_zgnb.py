"""Tests for the personal rule set (B1/B2/B3, 单针下20/30, 量价齐升V3, 0AMV 开关)."""

import numpy as np
import pandas as pd
import pytest

from aqlab.indicators_extra import amplitude, kdj, rsl, white_line, yellow_line
from aqlab.profiles import PROFILES, build_gate, list_profiles, load_profile
from aqlab.rules import build_rule
from aqlab.rules_zgnb import (
    ActiveMarketValueGate,
    B1Opportunity,
    B2Confirm,
    B3Confirm,
    NeedleRSL,
    VolumePriceV3,
)


def frame(closes, volumes=None, highs=None, lows=None, opens=None, lows_eq_close=False):
    closes = pd.Series(closes, dtype=float)
    n = len(closes)
    idx = pd.bdate_range("2024-01-01", periods=n)
    opens = closes.shift(1).fillna(closes) if opens is None else pd.Series(opens, dtype=float)
    highs = closes * 1.002 if highs is None else pd.Series(highs, dtype=float)
    if lows is not None:
        lows = pd.Series(lows, dtype=float)
    elif lows_eq_close:
        lows = closes
    else:
        lows = closes * 0.998
    volumes = pd.Series(1000.0, index=range(n)) if volumes is None else pd.Series(volumes, dtype=float)
    return pd.DataFrame(
        {"open": opens.values, "high": highs.values, "low": lows.values, "close": closes.values, "volume": volumes.values},
        index=idx,
    )


# 15 根 -1.5% 的下跌后收平：J ≈ -11（≤ -10），振幅 4%（≤ 7%），涨幅 0% —— 正好命中 B1
B1_CLOSES = [100.0 * 0.98 ** i for i in range(14)]


def b1_frame():
    """-2%/日 的下跌序列：过程中会命中 B1（J ≤ -10、涨幅 -2%、振幅 0.4%）。"""
    return frame(B1_CLOSES)


# --------------------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------------------
def test_kdj_matches_recursive_definition():
    df = frame(list(np.linspace(100, 130, 60)))
    out = kdj(df)
    assert list(out.columns) == ["k", "d", "j"]
    assert out["k"].between(0, 100).all() and out["d"].between(0, 100).all()
    # j = 3k - 2d
    assert np.allclose(out["j"], 3 * out["k"] - 2 * out["d"])
    # flat prices -> RSV undefined -> defaults to 50 -> k,d stay 50, j = 50
    flat = frame([100.0] * 30)
    flat_kdj = kdj(flat)
    assert flat_kdj["k"].iloc[-1] == pytest.approx(50.0)
    assert flat_kdj["j"].iloc[-1] == pytest.approx(50.0)


def test_rsl_formula_and_bounds():
    closes = list(np.linspace(100, 120, 30))
    df = frame(closes)
    value = rsl(df, 21).iloc[-1]
    low = min(df["low"].iloc[-21:])
    high = max(df["close"].iloc[-21:])
    expected = (closes[-1] - low) / (high - low) * 100
    assert value == pytest.approx(expected)
    assert 0 <= value <= 100
    # degenerate window returns the neutral 50
    assert rsl(frame([100.0] * 5, lows_eq_close=True), 5).iloc[-1] == 50.0


def test_white_and_yellow_lines():
    df = frame(list(np.linspace(100, 140, 130)))
    white = white_line(df)
    yellow = yellow_line(df)
    assert white.iloc[-1] > 0 and yellow.notna().sum() > 0
    assert white.iloc[-1] > yellow.iloc[-1]  # steady uptrend: short line above long line
    assert amplitude(df).iloc[-1] == pytest.approx((df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-2])


# --------------------------------------------------------------------------------------
# 单针下 20 / 30
# --------------------------------------------------------------------------------------
def test_needle_rsl_triggers_on_shallow_pullback_from_highs():
    closes = [100.0 * 1.01 ** i for i in range(27)] + [130.3, 130.15, 130.0]
    df = frame(closes, lows_eq_close=True)  # no lower shadow: RSL3 then measures the pullback itself
    rule20 = NeedleRSL(short_max=20.0, long_min=80.0)
    rule30 = NeedleRSL(short_max=30.0, long_min=85.0, long_strict=True)

    short = rsl(df, 3).iloc[-1]
    long = rsl(df, 21).iloc[-1]
    assert short <= 20 and long >= 80
    assert bool(rule20.signal(df).iloc[-1]) is True
    assert bool(rule30.signal(df).iloc[-1]) is True
    assert rule20.score(df).iloc[-1] == 1.0


def test_needle_rsl_does_not_trigger_in_downtrend():
    df = frame([100.0 * 0.99 ** i for i in range(40)])
    assert NeedleRSL().signal(df).sum() == 0


# --------------------------------------------------------------------------------------
# B1 / B2 / B3
# --------------------------------------------------------------------------------------
def test_b1_triggers_after_decline_with_small_candle():
    df = b1_frame()
    rule = B1Opportunity()
    signal = rule.signal(df)
    assert signal.any(), "the declining series must produce at least one B1 bar"
    idx = signal[signal].index[-1]

    assert kdj(df)["j"].loc[idx] <= -10
    assert -0.02 <= df["close"].pct_change().loc[idx] <= 0.018
    assert amplitude(df).loc[idx] <= 0.07

    # the same bar with a +5% move must not qualify (pct_max = 1.8%)
    other = df.copy()
    other.loc[idx, "close"] = df["close"].shift(1).loc[idx] * 1.05
    assert bool(rule.signal(other).loc[idx]) is False


def test_b1_turnover_condition():
    df = b1_frame()
    df["turnover"] = 0.01  # cumulative turnover far below 38%
    signal = B1Opportunity().signal(df)
    assert signal.any()
    idx = signal[signal].index[-1]

    blocked = df.copy()
    blocked["turnover"] = 0.5  # cumulative turnover far above 38% -> blocked
    assert bool(B1Opportunity().signal(blocked).loc[idx]) is False

    # without a turnover column the condition is skipped, or enforced as a hard gate
    plain = b1_frame()
    assert bool(B1Opportunity().signal(plain).loc[idx]) is True
    assert bool(B1Opportunity(require_turnover=True).signal(plain).loc[idx]) is False


def test_b2_requires_gain_j_and_volume():
    closes = [100.0 * 0.98 ** i for i in range(12)]
    closes[-1] = closes[-2] * 0.99
    step1 = closes + [closes[-1] * 1.025, closes[-1] * 1.025 * 1.022]
    volumes = [1000.0] * len(closes) + [1500.0, 2200.0]
    df = frame(step1, volumes=volumes)
    rule = B2Confirm()
    assert bool(rule.signal(df).iloc[-1]) is True

    # not enough gain -> no signal
    flat = frame(closes + [closes[-1] * 1.001, closes[-1] * 1.002], volumes=volumes)
    assert bool(rule.signal(flat).iloc[-1]) is False


def test_b3_requires_doji_after_b2_and_flat_open():
    closes = [100.0 * 0.98 ** i for i in range(12)]
    closes[-1] = closes[-2] * 0.99
    step1 = closes + [closes[-1] * 1.025, closes[-1] * 1.025 * 1.022]
    volumes = [1000.0] * len(closes) + [1500.0, 2200.0]
    step2 = step1 + [step1[-1] * 1.004]                      # doji-ish small body
    df = frame(step2, volumes=volumes + [1000.0])            # keep the volume profile of the B2 bar
    df.loc[df.index[-1], "open"] = step1[-1]                 # flat open
    assert bool(B3Confirm().signal(df).iloc[-1]) is True

    gapped = df.copy()
    gapped.loc[gapped.index[-1], "open"] = step1[-1] * 1.05  # 高开 5% -> not flat
    assert bool(B3Confirm(open_tolerance=0.01).signal(gapped).iloc[-1]) is False


# --------------------------------------------------------------------------------------
# 量价齐升 V3
# --------------------------------------------------------------------------------------
def test_volume_price_v3_logic_via_controlled_lines(monkeypatch):
    """Isolate the five hard conditions by controlling the two lines and J."""
    n = 130
    closes = list(100.0 * 1.002 ** np.arange(n))
    closes[-3] = closes[-4] * 1.01
    closes[-2] = closes[-3] * 1.03
    closes[-1] = closes[-2] * 1.04
    volumes = [1000.0] * n
    volumes[-3] = 900.0
    volumes[-2] = 1500.0
    volumes[-1] = 2400.0
    df = frame(closes, volumes=volumes)
    for i in (-2, -1):
        df.iloc[i, df.columns.get_loc("open")] = closes[i] / 1.01  # 阳线

    import aqlab.rules_zgnb as mod

    monkeypatch.setattr(mod, "white_line", lambda d: pd.Series(150.0, index=d.index))
    monkeypatch.setattr(mod, "yellow_line", lambda d: pd.Series(120.0, index=d.index))
    monkeypatch.setattr(mod, "kdj", lambda d: pd.DataFrame({"k": 40.0, "d": 35.0, "j": 40.0}, index=d.index))

    rule = VolumePriceV3()
    assert bool(rule.signal(df).iloc[-1]) is True
    score = rule.score(df).iloc[-1]
    assert 0.7 <= score <= 1.0

    # J too high -> blocked
    monkeypatch.setattr(mod, "kdj", lambda d: pd.DataFrame({"k": 80.0, "d": 70.0, "j": 95.0}, index=d.index))
    assert bool(rule.signal(df).iloc[-1]) is False

    # white below yellow -> blocked
    monkeypatch.setattr(mod, "kdj", lambda d: pd.DataFrame({"k": 40.0, "d": 35.0, "j": 40.0}, index=d.index))
    monkeypatch.setattr(mod, "white_line", lambda d: pd.Series(100.0, index=d.index))
    assert bool(rule.signal(df).iloc[-1]) is False


def test_volume_price_v3_needs_114_bars_for_the_yellow_line():
    df = frame(list(np.linspace(100, 120, 60)))
    assert VolumePriceV3().signal(df).sum() == 0


# --------------------------------------------------------------------------------------
# 0AMV 活跃市值 + 波段开关
# --------------------------------------------------------------------------------------
class _FakePctGate(ActiveMarketValueGate):
    """Gate whose 0AMV percentage series is injected, so the band logic is testable."""

    def __init__(self, pcts, **kwargs):
        super().__init__(**kwargs)
        self._pcts = pd.Series(pcts, dtype=float, index=pd.bdate_range("2024-01-01", periods=len(pcts)))

    def pct_series(self, universe):  # noqa: D102 - test double
        return self._pcts


def test_active_shares_recursion_with_true_turnover():
    n = 200
    df = frame([100.0] * n, volumes=[100.0] * n)
    df["turnover"] = 0.0  # no churn: A converges to vol_shares / (1 - rho)
    gate = ActiveMarketValueGate(rho=0.92)
    a = gate.active_shares(df)
    expected = 100.0 * 100.0 / (1 - 0.92)  # volume(手)->股 ×100 then /(1-rho)
    assert a.iloc[-1] == pytest.approx(expected, rel=1e-3)
    assert any("真实换手率" in note for note in gate.notes)


def test_active_shares_fallback_estimates_float_shares():
    df = frame([100.0] * 40, volumes=[100.0] * 40)
    gate = ActiveMarketValueGate()
    gate.active_shares(df)
    assert any("估计流通股本" in note for note in gate.notes)


class _ConstantSharesGate(ActiveMarketValueGate):
    """Gate with a constant active-share vector: isolates the index aggregation."""

    def active_shares(self, df):  # noqa: D102 - test double
        return pd.Series(1.0, index=df.index)


def test_amv_pct_series_uses_close_times_active_shares():
    universe = {"A": frame([100.0, 110.0, 121.0], volumes=[100.0] * 3), "B": frame([50.0, 55.0, 60.5], volumes=[10.0] * 3)}
    gate = _ConstantSharesGate()
    pct = gate.pct_series(universe)
    assert len(pct) == 3
    assert pct.iloc[-1] == pytest.approx(0.10, rel=1e-9)  # both symbols +10% -> index +10%


def test_gate_opens_on_strong_single_day_and_closes_on_2_3pct():
    gate = _FakePctGate([0.01, 0.045, 0.01, -0.024])
    series = gate.gate_series({})
    assert list(series["gate"]) == [0, 1, 1, 0]
    assert series["trigger"].iloc[1] == "strong"
    assert series["trigger"].iloc[3] == "close"
    assert gate.state_at({}) == 0
    assert gate.state_at({}, as_of=series.index[2]) == 1


def test_gate_opens_on_two_day_cumulative_but_rejects_a_close_day_inside_the_window():
    gate = _FakePctGate([0.025, 0.02])
    assert gate.state_at({}) == 1
    assert gate.trigger_at({}) == "normal"

    blocked = _FakePctGate([0.03, -0.024, 0.03])
    assert list(blocked.gate_series({})["gate"]) == [0, 0, 0]  # a <= -2.3% day poisons the window


def test_gate_weak_signal_requires_three_days():
    gate = _FakePctGate([0.015, 0.015, 0.015])
    series = gate.gate_series({})
    assert series["gate"].iloc[-1] == 1
    assert series["trigger"].iloc[-1] == "weak"


# --------------------------------------------------------------------------------------
# profiles & registry
# --------------------------------------------------------------------------------------
def test_profiles_are_registered_and_loadable():
    summary = list_profiles()
    for name in ("generic", "b1", "b2", "b3", "needle_20", "needle_30", "volume_price_v3", "zgnb_full", "zgnb_needle30"):
        assert name in summary
        bindings = load_profile(name)
        assert bindings and all(weight > 0 for _r, _p, weight in bindings)

    needle20 = dict((r, p) for r, p, _w in load_profile("needle_20"))
    assert needle20["needle_rsl"]["short_max"] == 20.0
    needle30 = dict((r, p) for r, p, _w in load_profile("needle_30"))
    assert needle30["needle_rsl"]["short_max"] == 30.0 and needle30["needle_rsl"]["long_min"] == 85.0

    # copies are returned, so callers cannot mutate the registry
    mutated = load_profile("b1")
    mutated[0][1]["j_max"] = 123
    assert load_profile("b1")[0][1] == {}

    with pytest.raises(KeyError):
        load_profile("nope")


def test_build_gate_and_build_rule_resolve_personal_names():
    assert isinstance(build_gate("amv"), ActiveMarketValueGate)
    assert build_gate("none") is None
    with pytest.raises(KeyError):
        build_gate("nope")

    rule = build_rule("needle_rsl", short_max=30.0)
    assert rule.params["short_max"] == 30.0
    assert isinstance(build_rule("b1_opportunity"), B1Opportunity)
    with pytest.raises(KeyError):
        build_rule("nope")


def test_pipeline_runs_zgnb_profile_end_to_end():
    from aqlab.agent import ScriptedClient  # noqa: F401 - keep import graph exercised
    from aqlab.pipeline import DailyConfig, DailyPipeline
    from aqlab.profiles import load_profile
    from aqlab.tools import SyntheticDataSource

    source = SyntheticDataSource(n_symbols=10, n_days=300, seed=11)
    pipeline = DailyPipeline(
        source,
        rule_bindings=load_profile("zgnb_full"),
        gate=build_gate("amv"),
        config=DailyConfig(top_n=5),
    )
    report = pipeline.run(push=False)
    assert report.gate_state in (0, 1)
    assert set(report.weights) == {rule for rule, _p, _w in PROFILES["zgnb_full"]}
    for pick in report.picks:
        assert pick["score"] > 0
