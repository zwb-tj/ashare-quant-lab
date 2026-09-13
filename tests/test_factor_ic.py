"""因子有效性研究的测试（离线，合成数据）。

守两条最容易出错的前提：

1. **因果性**：因子"整段算完再取当天"必须与"截断到当天再算"完全一致
   （否则优化过的预计算会偷偷用上未来数据）；
2. **IC 的定义**：单调关系应为 +1、反向为 -1、噪声接近 0、常量返回 NaN（而不是 0）。
"""

import pandas as pd
import pytest

from aqlab.factor_ic import (
    ICConfig,
    factor_ic_panel,
    precompute_factors,
    quantile_returns,
    spearman_ic,
    summarize_ic,
)
from aqlab.screen import factor_table
from aqlab.tools import SyntheticDataSource


@pytest.fixture(scope="module")
def frames():
    source = SyntheticDataSource(n_symbols=20, n_days=420, seed=11)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def test_spearman_ic_detects_monotone_reverse_and_noise():
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_ic(factor, pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(1.0)
    assert spearman_ic(factor, pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman_ic(factor, pd.Series([3.0] * 5)) != spearman_ic(factor, pd.Series([3.0] * 5))   # 常量 -> NaN
    assert spearman_ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) != spearman_ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))  # 样本不足 -> NaN
    # 单调但非线性：秩相关仍为 1（这正是用 Spearman 而不是 Pearson 的原因）
    assert spearman_ic(factor, pd.Series([1.0, 10.0, 100.0, 1000.0, 10000.0])) == pytest.approx(1.0)


def test_precomputed_factors_match_the_truncated_factor_table(frames):
    """预计算（整段）与 factor_table（截断到当天）必须给出同样的因子值。"""
    config = ICConfig(min_history=130)
    precomputed = precompute_factors(frames, config)
    symbol = sorted(frames)[0]
    frame = frames[symbol]
    as_of = frame.index[-40]
    table = factor_table({symbol: frame}, as_of=as_of, min_history=config.min_history)
    assert len(table) == 1
    row = table.iloc[0]
    reference = precomputed[symbol].loc[as_of]
    for factor in ("mom_20", "mom_60", "trend_gap", "vol_20", "rsi_14"):
        assert float(row[factor]) == pytest.approx(float(reference[factor]), rel=1e-9, abs=1e-12)


def test_factor_ic_panel_is_a_clean_long_table(frames):
    config = ICConfig(forward_days=20, step_days=5, min_history=130, min_symbols=5)
    panel = factor_ic_panel(frames, config)
    assert set(panel.columns) == {"date", "factor", "ic", "symbols"}
    assert set(panel["factor"]) == set(config.factors)
    assert panel["date"].nunique() > 5
    assert (panel["symbols"] >= config.min_symbols).all()
    # 每个截面上每个因子只有一行
    assert not panel.duplicated(subset=["date", "factor"]).any()
    # IC 必须在 [-1, 1] 内，且不再保留算不出 IC 的截面（NaN 行已被剔除）
    assert not panel["ic"].isna().any()
    assert panel["ic"].between(-1.0, 1.0).all()


def test_summarize_ic_uses_the_overlap_adjusted_t(frames):
    config = ICConfig(forward_days=20, step_days=5, min_history=130, min_symbols=5)
    summary = summarize_ic(factor_ic_panel(frames, config), config)
    assert {"factor", "periods", "ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_adj", "positive_rate"} <= set(summary.columns)
    for row in summary.itertuples(index=False):
        # 前瞻 20 / 步长 5 -> 重叠 4 倍 -> t 值保守打折 2 倍
        assert row.t_stat_adj == pytest.approx(row.t_stat / 2.0, rel=1e-9)
        assert 0.0 <= row.positive_rate <= 1.0
    assert summarize_ic(pd.DataFrame()).empty


def test_summarize_without_config_uses_the_default_overlap(frames):
    config = ICConfig(min_history=130, min_symbols=5)
    summary = summarize_ic(factor_ic_panel(frames, config))          # 不传 config -> 用默认参数
    assert not summary.empty
    assert "t_stat_adj" in summary.columns


def test_quantile_returns_covers_every_group(frames):
    config = ICConfig(forward_days=20, step_days=10, min_history=130, min_symbols=8, quantiles=4)
    table = quantile_returns(frames, config)
    assert set(table["group"]) == {1, 2, 3, 4}
    assert set(table["factor"]) == set(config.factors)
    assert (table["periods"] > 0).all()


def test_config_validation():
    with pytest.raises(ValueError):
        ICConfig(forward_days=0)
    with pytest.raises(ValueError):
        ICConfig(step_days=0)
    with pytest.raises(ValueError):
        ICConfig(min_symbols=2)
    with pytest.raises(ValueError):
        ICConfig(quantiles=1)
