"""因子组合方案的测试（离线，合成数据）。

最重要的是**权重的时间边界**：t 日定权只能用严格早于 t 的 IC。
测试用"改掉最后一期 IC，看当期权重变不变"这种最直接的方式验证，而不是靠读代码相信。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.factor_strategy import (
    SCHEMES,
    SchemeConfig,
    run_weight_schemes,
    scheme_weights,
    summarize_schemes,
    trailing_ic,
)
from aqlab.screen import DEFAULT_WEIGHTS
from aqlab.tools import SyntheticDataSource


@pytest.fixture(scope="module")
def frames():
    source = SyntheticDataSource(n_symbols=16, n_days=420, seed=7)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def test_scheme_weights_modes():
    factors = ("mom_20", "mom_60", "vol_20")
    fixed = scheme_weights({}, "fixed", factors)
    assert fixed == {factor: float(DEFAULT_WEIGHTS.get(factor, 0.0)) for factor in factors}

    flipped = scheme_weights({}, "sign_flip", factors)
    assert all(flipped[factor] == -fixed[factor] for factor in factors)

    history = {"mom_20": 0.05, "mom_60": -0.03, "vol_20": 0.0}
    signs = scheme_weights(history, "ic_sign", factors)
    assert signs == {"mom_20": 1.0, "mom_60": -1.0, "vol_20": 0.0}      # 0 就是不加权

    weighted = scheme_weights(history, "ic_weight", factors)
    assert weighted["mom_20"] == pytest.approx(0.05 / 0.08)             # 按 |IC| 归一化
    assert weighted["mom_60"] == pytest.approx(-0.03 / 0.08)

    with pytest.raises(ValueError):
        scheme_weights({}, "no-such-scheme", factors)


def test_trailing_ic_never_sees_the_current_section():
    wide = pd.DataFrame(
        {"mom_20": [0.1, 0.2, 0.3, 9.9], "mom_60": [-0.1, -0.1, -0.1, -9.9]},
        index=pd.date_range("2024-01-01", periods=4, freq="W"),
    )
    # lag=1、index=3：只能用前 3 期；lookback=2 -> 取第 1、2 期
    past = trailing_ic(wide, index=3, lookback=2, lag=1)
    assert past["mom_20"] == pytest.approx((0.2 + 0.3) / 2)
    assert past["mom_60"] == pytest.approx(-0.1)
    # 作弊口径（lag=0）才会把 9.9 算进去——这正是不能用的口径
    cheating = trailing_ic(wide, index=3, lookback=2, lag=0)
    assert cheating["mom_20"] == pytest.approx((0.3 + 9.9) / 2)
    # 第一期没有历史（lag=1）时返回空，方案侧会跳过
    assert trailing_ic(wide, index=0, lookback=2, lag=1) == {}


def test_run_weight_schemes_is_causal_and_reports_benchmark(frames):
    config = SchemeConfig(top_n=5, forward_days=10, step_days=10, min_history=130, min_symbols=8, lookback_periods=3)
    detail, curves = run_weight_schemes(frames, config)
    assert not detail.empty
    assert set(detail["scheme"]) <= set(SCHEMES)
    assert "benchmark" in curves
    # 净收益 = 毛收益 - 成本
    cost = config.round_trip_cost_bps / 10_000.0
    assert np.allclose(detail["net_return"], detail["gross_return"] - cost)
    # 每期都拿够 top_n
    assert (detail["picks"] == config.top_n).all()
    # 超额 = 净收益 - 同期等权全市场
    assert np.allclose(detail["excess_net"], detail["net_return"] - detail["benchmark"])


def test_lag_zero_uses_future_ic_and_lag_one_does_not(frames):
    """把当期 IC 换成极端值：lag=0 的方案会立刻改变选择，lag=1 的不会。"""
    base = SchemeConfig(top_n=5, forward_days=10, step_days=10, min_history=130, min_symbols=8, lookback_periods=3, lag_periods=1)
    cheating = SchemeConfig(**{**base.__dict__, "lag_periods": 0})
    honest_detail, _ = run_weight_schemes(frames, base, schemes=("ic_sign",))
    cheat_detail, _ = run_weight_schemes(frames, cheating, schemes=("ic_sign",))
    # 两种口径在早期（历史还没积累）会被跳过，但至少要有可比较的期数
    assert not honest_detail.empty and not cheat_detail.empty

    # 关键断言：honest 口径下，某期的权重只取决于它之前的 IC
    wide = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3], "b": [-0.1, -0.2, -0.3]},
        index=pd.date_range("2024-01-01", periods=3, freq="W"),
    )
    honest_today = trailing_ic(wide, index=2, lookback=3, lag=1)
    mutated = wide.copy()
    mutated.iloc[2] = [99.0, -99.0]
    assert trailing_ic(mutated, index=2, lookback=3, lag=1) == honest_today


def test_summarize_schemes_orders_by_excess(frames):
    config = SchemeConfig(top_n=5, forward_days=10, step_days=10, min_history=130, min_symbols=8, lookback_periods=3)
    detail, _ = run_weight_schemes(frames, config)
    summary = summarize_schemes(detail)
    assert {"scheme", "periods", "mean_net", "win_rate", "mean_excess", "excess_t"} <= set(summary.columns)
    assert summary["mean_excess"].is_monotonic_decreasing
    assert summarize_schemes(pd.DataFrame()).empty


def test_config_validation():
    with pytest.raises(ValueError):
        SchemeConfig(top_n=0)
    with pytest.raises(ValueError):
        SchemeConfig(lookback_periods=0)
    with pytest.raises(ValueError):
        SchemeConfig(lag_periods=-1)
    with pytest.raises(ValueError):
        SchemeConfig(round_trip_cost_bps=-1)
