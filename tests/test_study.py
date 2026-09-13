"""规则事件研究测试（v0.5）。"""

from typing import ClassVar

import pandas as pd
import pytest

from aqlab.profiles import load_profile
from aqlab.study import (
    baseline_stats,
    format_study,
    forward_returns,
    rule_event_study,
    study_profile,
    write_study,
)
from aqlab.tools import SyntheticDataSource


def frame(closes):
    c = pd.Series(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    return pd.DataFrame(
        {"open": c.values, "high": (c * 1.001).values, "low": (c * 0.999).values, "close": c.values, "volume": 1000.0},
        index=idx,
    )


class StubRule:
    """只在指定 bar 触发的最小规则，用于手算校验。"""

    name = "stub"
    params: ClassVar[dict] = {"at": 0}

    def __init__(self, at: int = 0):
        self.params = {"at": at}

    def score(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        if 0 <= self.params["at"] < len(df):
            out.iloc[self.params["at"]] = 1.0
        return out

    def signal(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > 0


def test_forward_returns_hand_checked():
    df = frame([100.0, 110.0, 121.0, 133.1])
    fwd = forward_returns(df, (1, 2))
    assert fwd["fwd_1"].iloc[0] == pytest.approx(0.10)
    assert fwd["fwd_1"].iloc[2] == pytest.approx(0.10)
    assert fwd["fwd_1"].iloc[3] != fwd["fwd_1"].iloc[3]        # 尾部为 NaN
    assert fwd["fwd_2"].iloc[0] == pytest.approx(0.21)
    assert fwd["fwd_2"].iloc[1] == pytest.approx(0.21)
    with pytest.raises(ValueError):
        forward_returns(df, (0,))


def test_rule_event_study_hand_checked():
    df = frame([100.0, 110.0, 121.0, 133.1, 140.0])
    table = rule_event_study({"A": df}, "stub", rule=StubRule(at=0), horizons=(1, 3), min_history=3)
    row1 = table[table["horizon"] == 1].iloc[0]
    assert row1["signals"] == 1
    assert row1["n"] == 1
    assert row1["mean"] == pytest.approx(0.10)          # 100 -> 110
    row3 = table[table["horizon"] == 3].iloc[0]
    assert row3["mean"] == pytest.approx(0.331)         # 100 -> 133.1


def test_baseline_stats_pools_all_bars():
    df = frame([100.0, 110.0, 121.0])
    baseline = baseline_stats({"A": df}, horizons=(1,))
    row = baseline.iloc[0]
    assert row["symbol"] == "ALL"
    assert row["n"] == 2                                # 最后一根没有未来 1 日收益
    assert row["mean"] == pytest.approx(0.10)


def test_study_profile_rows_columns_and_determinism():
    source = SyntheticDataSource(n_symbols=8, n_days=300, seed=11)
    universe = {sym: source.bars(sym) for sym in source.symbols()}
    bindings = [*load_profile("needle_20"), ("brick_green_to_red", {}, 0.5)]

    table, baseline = study_profile(universe, bindings, horizons=(1, 5), min_history=60)
    assert set(table["rule"]) == {"needle_rsl", "brick_green_to_red"}
    assert len(table) == 2 * 2                       # 2 条规则 × 2 个持有期
    for col in ("signals", "mean", "win_rate", "base_mean", "base_win_rate", "excess_mean", "excess_win_rate"):
        assert col in table.columns

    again, _ = study_profile(universe, bindings, horizons=(1, 5), min_history=60)
    pd.testing.assert_frame_equal(table, again)
    assert set(baseline["horizon"]) == {1, 5}


def test_study_profile_handles_rules_without_signals():
    source = SyntheticDataSource(n_symbols=4, n_days=200, seed=3)
    universe = {sym: source.bars(sym) for sym in source.symbols()}
    # 用一条几乎不可能触发的规则（J <= -999 的简化 B1）
    bindings = [("b1_opportunity", {"j_max": -999.0}, 1.0)]
    table, _ = study_profile(universe, bindings, horizons=(1,), min_history=60)
    row = table.iloc[0]
    assert row["signals"] == 0


def test_study_profile_rejects_empty_bindings():
    with pytest.raises(ValueError):
        study_profile({"A": frame([100.0] * 10)}, [], horizons=(1,))


def test_format_study_renders_percentages_and_notes():
    source = SyntheticDataSource(n_symbols=4, n_days=200, seed=5)
    universe = {sym: source.bars(sym) for sym in source.symbols()}
    table, _ = study_profile(universe, load_profile("needle_20"), horizons=(1, 3))
    text = format_study(table)
    assert "规则事件研究" in text
    assert "均值%" in text and "超额均值%" in text
    assert "基线" in text


def test_write_study_writes_all_artifacts(tmp_path):
    source = SyntheticDataSource(n_symbols=4, n_days=200, seed=7)
    universe = {sym: source.bars(sym) for sym in source.symbols()}
    table, baseline = study_profile(universe, load_profile("needle_20"), horizons=(1,))
    paths = write_study(tmp_path, table, baseline, meta={"profile": "needle_20"})
    for key in ("markdown", "csv", "baseline", "meta"):
        assert paths[key].exists()
    assert "needle_rsl" in paths["csv"].read_text(encoding="utf-8-sig")
