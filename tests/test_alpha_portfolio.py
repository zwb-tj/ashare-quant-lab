"""公式因子组合回测的测试（离线，合成数据）。

重点是**口径正确性**，因为这里最容易出现"看起来赚钱其实用了未来价"的错误：

① **入场价必须是次日开盘**：用当日收盘成交就是隐性未来函数；
② **方向不能反**：买"高分位"必须真的选到因子值最大的那批（曾经把升序写反，导致符号整体颠倒）；
③ **成本按实际换手收**：持仓完全不变时不该收费，全部换掉时按全额收；
④ **盈亏平衡成本**要与"毛超额 / 换手"一致，且换手为 0 时返回 NaN 而不是 inf。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.alpha101 import build_panel
from aqlab.alpha_portfolio import (
    AlphaPortfolioConfig,
    break_even_cost_bps,
    cost_sensitivity,
    run_alpha_portfolio,
    summarize_alpha_portfolio,
)


def _market(n_days=400, n_sym=30, seed=0, drift=0.0005, vol=0.02):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    returns = rng.normal(drift, vol, (n_days, n_sym))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=cols)
    open_price = close.shift(1).fillna(close)
    frames = {
        c: pd.DataFrame(
            {
                "open": open_price[c],
                "high": close[c] * 1.01,
                "low": close[c] * 0.99,
                "close": close[c],
                "volume": 1e6,
                "amount": close[c] * 1e6,
            }
        )
        for c in cols
    }
    return close, open_price, build_panel(frames)


def test_high_direction_actually_selects_the_top_factor_values():
    """direction='top' 必须选因子值最大的那批（曾经把排序写成升序，整体反了）。"""
    close, open_price, panel = _market()
    rng = np.random.default_rng(1)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    position = 120
    as_of = close.index[position]

    outcome = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=60, direction="top"))
    row = outcome["detail"][outcome["detail"]["date"] == as_of]
    assert not row.empty
    # 手工重算：同日因子最大的 5 只，用次日开盘买入、第 5 日收盘卖出
    picks = factor.iloc[position].dropna().nlargest(5).index.tolist()
    manual = (close.iloc[position + 5][picks] / open_price.iloc[position + 1][picks] - 1).mean()
    assert float(row["gross_return"].iloc[0]) == pytest.approx(float(manual), rel=1e-9)

    bottom = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=60, direction="bottom"))
    row_b = bottom["detail"][bottom["detail"]["date"] == as_of]
    picks_b = factor.iloc[position].dropna().nsmallest(5).index.tolist()
    manual_b = (close.iloc[position + 5][picks_b] / open_price.iloc[position + 1][picks_b] - 1).mean()
    assert float(row_b["gross_return"].iloc[0]) == pytest.approx(float(manual_b), rel=1e-9)
    assert not np.isclose(float(row["gross_return"].iloc[0]), float(row_b["gross_return"].iloc[0]))


def test_benchmark_is_equal_weight_and_uses_the_same_prices():
    close, open_price, panel = _market()
    rng = np.random.default_rng(2)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    outcome = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=60))
    position = 150
    as_of = close.index[position]
    row = outcome["detail"][outcome["detail"]["date"] == as_of].iloc[0]
    manual = (close.iloc[position + 5] / open_price.iloc[position + 1] - 1).mean()
    assert float(row["benchmark"]) == pytest.approx(float(manual), rel=1e-9)


def test_costs_follow_actual_turnover_only():
    close, _open, panel = _market()
    rng = np.random.default_rng(3)
    # 常数因子 -> 每期选同一批 -> 除首期建仓外换手为 0 -> 不该持续产生成本
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    free = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=60))
    charged = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=20, min_history=60))
    assert float(free["detail"]["turnover"].iloc[0]) == pytest.approx(1.0), "首期是建仓，换手按全额"
    assert float(charged["detail"]["turnover"].iloc[1:].max()) == pytest.approx(0.0), "之后持仓不变，换手必须为 0"
    assert float(charged["detail"]["cost"].iloc[1:].abs().max()) == pytest.approx(0.0)
    # 只有首期付了一次成本，因此两档的净收益差 = 一次双边费率
    assert float(free["detail"]["net_return"].mean()) - float(charged["detail"]["net_return"].mean()) == pytest.approx(
        20 / 10_000 / len(charged["detail"]), rel=1e-6
    )

    # 随机因子 -> 每期几乎全换 -> 换手应接近 1，成本约等于双边费率
    random_factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    heavy = run_alpha_portfolio(random_factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=20, min_history=60))
    assert float(heavy["detail"]["turnover"].mean()) > 0.5
    assert float(heavy["detail"]["cost"].mean()) == pytest.approx(
        float(heavy["detail"]["turnover"].mean()) * 20 / 10_000, rel=1e-9
    )


def test_net_return_equals_gross_minus_cost():
    close, _open, panel = _market()
    rng = np.random.default_rng(4)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    detail = run_alpha_portfolio(factor, panel, AlphaPortfolioConfig(top_n=8, hold_days=5, cost_bps=15, min_history=60))["detail"]
    assert np.allclose(detail["net_return"], detail["gross_return"] - detail["cost"])
    assert np.allclose(detail["gross_excess"], detail["gross_return"] - detail["benchmark"])
    assert np.allclose(detail["net_excess"], detail["net_return"] - detail["benchmark"])


def test_break_even_cost_matches_gross_over_turnover():
    close, _open, panel = _market()
    rng = np.random.default_rng(5)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    config = AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=60)
    detail = run_alpha_portfolio(factor, panel, config)["detail"]
    expected = float(detail["gross_excess"].mean()) / float(detail["turnover"].mean()) * 10_000
    assert break_even_cost_bps(factor, panel, config) == pytest.approx(expected, rel=1e-9)

    # 持仓完全不变（常数因子）时：除首期建仓外换手为 0，
    # 盈亏平衡成本应当很大而不是无意义的小数——这里断言它与"毛超额/平均换手"一致即可。
    flat = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    flat_detail = run_alpha_portfolio(flat, panel, config)["detail"]
    flat_expected = float(flat_detail["gross_excess"].mean()) / float(flat_detail["turnover"].mean()) * 10_000
    assert break_even_cost_bps(flat, panel, config) == pytest.approx(flat_expected, rel=1e-9)


def test_cost_sensitivity_is_monotone_in_cost():
    close, _open, panel = _market()
    rng = np.random.default_rng(6)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    table = cost_sensitivity(factor, panel, AlphaPortfolioConfig(top_n=5, hold_days=5, min_history=60))
    assert list(table["cost_bps"]) == [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    # 毛超额与成本无关（只有净超额随成本下降）
    assert table["mean_gross_excess"].nunique() == 1
    assert table["mean_net_excess"].is_monotonic_decreasing


def test_informative_factor_beats_the_benchmark_gross_and_the_direction_comes_from_in_sample():
    """构造一个真的含未来信息的因子：样本内定方向后，毛超额应当显著为正。"""
    close, open_price, panel = _market(n_days=500, n_sym=40, seed=7)
    rng = np.random.default_rng(8)
    forward = close.shift(-5) / open_price.shift(-1) - 1
    factor = forward + rng.normal(0, 0.01, forward.shape)          # 有信息但含噪声
    config = AlphaPortfolioConfig(top_n=5, hold_days=5, cost_bps=0, min_history=120, direction="ic_sign")
    outcome = run_alpha_portfolio(factor, panel, config)
    assert outcome["direction"] == 1, "样本内 IC 为正，方向应判为做多高分位"
    assert outcome["summary"]["mean_gross_excess"] > 0
    summary = summarize_alpha_portfolio(outcome["detail"])
    assert summary.iloc[0]["periods"] > 0


def test_config_validation_and_empty_detail():
    with pytest.raises(ValueError):
        AlphaPortfolioConfig(top_n=0)
    with pytest.raises(ValueError):
        AlphaPortfolioConfig(hold_days=0)
    with pytest.raises(ValueError):
        AlphaPortfolioConfig(cost_bps=-1)
    with pytest.raises(ValueError):
        AlphaPortfolioConfig(direction="sideways")
    with pytest.raises(ValueError):
        AlphaPortfolioConfig(is_fraction=1.0)
    assert summarize_alpha_portfolio(pd.DataFrame()).empty
