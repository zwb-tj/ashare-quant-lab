"""Walk-forward 滚动窗口校验测试。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.walkforward import (
    WalkForwardConfig,
    benchmark_return,
    composite_scores,
    format_walkforward,
    walk_forward,
    window_bounds,
    write_walkforward,
)


def frame(closes):
    c = pd.Series(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    return pd.DataFrame(
        {"open": c.values, "high": (c * 1.005).values, "low": (c * 0.995).values, "close": c.values, "volume": 1000.0},
        index=idx,
    )


class StubRule:
    """在指定 bar 触发的最小规则（用于把窗口逻辑与真实规则解耦）。"""

    name = "stub"

    def __init__(self, at=(5,), value=1.0):
        self.params = {"at": list(at), "value": value}
        self._at = set(int(i) for i in at)
        self._value = float(value)

    def score(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        for i in self._at:
            if 0 <= i < len(df):
                out.iloc[i] = self._value
        return out

    def signal(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > 0


@pytest.fixture
def stub_build(monkeypatch):
    def fake_build(name, **params):
        return StubRule(**params)

    monkeypatch.setattr("aqlab.walkforward.build_rule", fake_build)
    return fake_build


# --------------------------------------------------------------------------------------
# 窗口与基准
# --------------------------------------------------------------------------------------
def test_window_bounds_splits_by_test_days_and_step():
    index = pd.bdate_range("2024-01-01", periods=10)
    bounds = window_bounds(index, test_days=4, step_days=3)
    assert bounds == [
        (index[0], index[3]),
        (index[3], index[6]),
        (index[6], index[9]),
    ]
    short = window_bounds(pd.bdate_range("2024-01-01", periods=3), test_days=5, step_days=1)
    assert short == []


def test_benchmark_return_hand_checked():
    df = frame([100.0, 110.0, 121.0, 133.0])
    assert benchmark_return(df, df.index[0], df.index[2]) == pytest.approx(0.21)
    assert benchmark_return(df, df.index[0], df.index[0]) == 0.0


def test_composite_scores_weighted_average(stub_build):
    df = frame([100.0] * 20)
    universe = {"A": df}
    scores = composite_scores(universe, [("stub", {"at": [3], "value": 1.0}, 3.0), ("stub", {"at": [3], "value": 0.5}, 1.0)])
    assert scores["A"].iloc[3] == pytest.approx((3 * 1.0 + 1 * 0.5) / 4)
    assert scores["A"].iloc[4] == 0.0


def test_composite_scores_validation(stub_build):
    with pytest.raises(ValueError):
        composite_scores({"A": frame([100.0] * 5)}, [])
    with pytest.raises(ValueError):
        composite_scores({"A": frame([100.0] * 5)}, [("stub", {}, 0.0)])


# --------------------------------------------------------------------------------------
# 固定持有期路径（手算）
# --------------------------------------------------------------------------------------
def test_fixed_horizon_return_hand_checked(stub_build):
    closes = [100.0 + i for i in range(20)]      # 100, 101, ..., 119
    df = frame(closes)
    universe = {"A": df}
    config = WalkForwardConfig(test_days=20, step_days=20, min_history=20, use_position=False, horizon=3)
    result = walk_forward(universe, [("stub", {"at": [5]}, 1.0)], config=config)

    trades = result["trades"]
    assert len(trades) == 1
    row = trades.iloc[0]
    # 信号在 bar 5 -> bar 6 收盘入场(106)，持有 3 根 -> bar 9 收盘离场(109)
    assert row["entry_date"] == str(df.index[6].date())
    assert float(row["entry_price"]) == pytest.approx(106.0)
    assert float(row["return"]) == pytest.approx(round(109.0 / 106.0 - 1.0, 4))  # 交易表按 4 位小数落盘
    assert row["exit_reason"] == "fixed_3d"

    windows = result["windows"]
    assert len(windows) == 1
    assert windows.iloc[0]["trades"] == 1
    assert windows.iloc[0]["top_exit_reason"] == "fixed_3d"
    assert result["summary"]["trades"] == 1


def test_fixed_horizon_skips_trades_running_past_the_end(stub_build):
    df = frame([100.0] * 40)
    config = WalkForwardConfig(test_days=40, step_days=40, min_history=20, use_position=False, horizon=5)
    result = walk_forward({"A": df}, [("stub", {"at": [37]}, 1.0)], config=config)
    assert result["trades"].empty       # 38 + 5 > 40，尾部不完整交易被丢弃


# --------------------------------------------------------------------------------------
# 持仓/离场路径与无未来函数
# --------------------------------------------------------------------------------------
def test_position_path_uses_exit_rules(stub_build):
    closes = [100.0 * 1.004 ** i for i in range(40)]
    df = frame(closes)
    config = WalkForwardConfig(test_days=40, step_days=40, min_history=20, use_position=True)
    result = walk_forward({"A": df}, [("stub", {"at": [5]}, 1.0)], config=config)

    trades = result["trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] != "fixed_5d"
    assert trades.iloc[0]["exit_reason"] in {
        "hard_stop", "stop_loss", "breakeven", "bbi_break", "swing_target",
        "time_stop", "time_stop_after_profit", "end_of_data", "first_target",
    }


def test_trades_are_attributed_to_the_window_containing_their_signal(stub_build):
    df = frame([100.0 * 1.002 ** i for i in range(120)])
    config = WalkForwardConfig(test_days=30, step_days=30, min_history=20, use_position=False, horizon=3)
    result = walk_forward({"A": df}, [("stub", {"at": [10, 45, 80]}, 1.0)], config=config)

    windows = result["windows"]
    trades = result["trades"]
    assert len(windows) == 4                       # 120 / 30
    assert len(trades) == 3
    assert set(trades["exit_reason"]) == {"fixed_3d"}

    # 每笔交易的信号日必须落在某个窗口内
    signal_dates = pd.to_datetime(trades["signal_date"])
    for date in signal_dates:
        match = windows[(pd.to_datetime(windows["start"]) <= date) & (pd.to_datetime(windows["end"]) >= date)]
        assert len(match) == 1
    counts = windows["trades"].tolist()
    assert sum(counts) == 3
    assert counts[:2] == [1, 1]                    # bar 10 在窗口 1，bar 45 在窗口 2


def test_walkforward_is_deterministic(stub_build):
    df = frame([100.0 * 1.001 ** i for i in range(90)])
    config = WalkForwardConfig(test_days=30, step_days=30, min_history=20, use_position=False, horizon=2)
    first = walk_forward({"A": df}, [("stub", {"at": [5, 40]}, 1.0)], config=config)
    second = walk_forward({"A": df}, [("stub", {"at": [5, 40]}, 1.0)], config=config)
    pd.testing.assert_frame_equal(first["windows"], second["windows"])
    pd.testing.assert_frame_equal(first["trades"], second["trades"])


# --------------------------------------------------------------------------------------
# 边界情况
# --------------------------------------------------------------------------------------
def test_no_signals_still_reports_windows_and_benchmark(stub_build):
    df = frame([100.0 * 1.001 ** i for i in range(60)])
    config = WalkForwardConfig(test_days=30, step_days=30, min_history=20, use_position=False, horizon=2)
    result = walk_forward({"A": df}, [("stub", {"at": []}, 1.0)], config=config)

    windows = result["windows"]
    assert len(windows) == 2
    assert (windows["trades"] == 0).all()
    assert windows["window_buy_hold"].notna().all()
    assert result["summary"]["trades"] == 0
    assert result["summary"]["windows_with_trades"] == 0


def test_too_short_history_returns_empty_report(stub_build):
    result = walk_forward({"A": frame([100.0] * 10)}, [("stub", {"at": [1]}, 1.0)], config=WalkForwardConfig(test_days=30))
    assert result["windows"].empty
    assert result["summary"] == {"windows": 0, "trades": 0}


def test_config_validation():
    with pytest.raises(ValueError):
        WalkForwardConfig(test_days=2)
    with pytest.raises(ValueError):
        WalkForwardConfig(step_days=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(min_history=5)
    with pytest.raises(ValueError):
        WalkForwardConfig(horizon=0)


def test_format_and_write_artifacts(stub_build, tmp_path):
    df = frame([100.0 * 1.001 ** i for i in range(60)])
    config = WalkForwardConfig(test_days=30, step_days=30, min_history=20, use_position=False, horizon=2)
    result = walk_forward({"A": df}, [("stub", {"at": [5]}, 1.0)], config=config)

    text = format_walkforward(result)
    assert "Walk-forward 校验报告" in text
    assert "超额%" in text and "同期间基准%" in text
    assert "汇总" in text

    paths = write_walkforward(tmp_path, result)
    for key in ("markdown", "windows", "trades"):
        assert paths[key].exists()
    assert "A" in paths["trades"].read_text(encoding="utf-8-sig")


def test_format_handles_insufficient_data():
    text = format_walkforward({"windows": pd.DataFrame(), "summary": {"windows": 0, "trades": 0}})
    assert "数据不足" in text


def test_per_trade_benchmark_is_like_for_like(stub_build):
    """固定持有期下，单笔收益与其同持有期买入持有收益应当一致（超额≈0）。"""
    closes = [100.0 * 1.003 ** i for i in range(60)]
    df = frame(closes)
    config = WalkForwardConfig(test_days=60, step_days=60, min_history=20, use_position=False, horizon=3)
    result = walk_forward({"A": df}, [("stub", {"at": [5, 20]}, 1.0)], config=config)

    trades = result["trades"]
    assert len(trades) == 2
    assert (trades["bench_return"].astype(float) - trades["return"].astype(float)).abs().max() <= 1e-4
    assert trades["excess_return"].astype(float).abs().max() <= 1e-4
    assert result["summary"]["avg_excess_return"] == pytest.approx(0.0, abs=1e-4)


def test_position_path_reports_same_period_benchmark(stub_build):
    """离场规则路径：每笔都带同持有期基准，且离场越早越可能与买入持有不同。"""
    closes = [100.0 * 1.001 ** i for i in range(90)]
    closes[40:50] = [closes[39] * (1 - 0.01 * k) for k in range(1, 11)]   # 中途一段下跌，触发止损
    df = frame(closes)
    config = WalkForwardConfig(test_days=90, step_days=90, min_history=20, use_position=True)
    result = walk_forward({"A": df}, [("stub", {"at": [39, 60]}, 1.0)], config=config)

    trades = result["trades"]
    assert not trades.empty
    assert trades["bench_return"].notna().all()
    assert "avg_excess_return" in result["windows"].columns
    assert result["summary"]["windows_beating_benchmark"] >= 0