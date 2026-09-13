"""因子评估流水线（样本内/外 + 校正）的测试（离线，小合成票池）。

守住三件容易搞错的事：

① **切分点要留出前瞻窗口**：样本内的最后一个截面不能借用样本外的收益；
② **挑选顺序**：必须"用样本内选、用样本外验"，反过来就是自欺；
③ 样本外同向性判定必须同时要求**显著性**与**方向一致**。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.factor_eval import FactorEvalConfig, evaluate_factor_ics, ic_of_factor
from aqlab.tools import SyntheticDataSource


@pytest.fixture(scope="module")
def universe():
    source = SyntheticDataSource(n_symbols=14, n_days=520, seed=11)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def test_config_validation():
    with pytest.raises(ValueError):
        FactorEvalConfig(horizons=())
    with pytest.raises(ValueError):
        FactorEvalConfig(step_days=0)
    with pytest.raises(ValueError):
        FactorEvalConfig(min_symbols=2)
    with pytest.raises(ValueError):
        FactorEvalConfig(alpha=0.0)
    with pytest.raises(ValueError):
        FactorEvalConfig(is_fraction=1.0)


def test_ic_of_factor_skips_sections_without_enough_symbols():
    index = pd.bdate_range("2024-01-01", periods=40)
    # 每只标的的因子与价格都要**各自变化**，否则 Spearman 秩相关无法定义（返回 NaN 会被过滤）
    factor = pd.DataFrame({f"s{i}": np.arange(40.0) * (i + 1) for i in range(4)}, index=index)
    close = pd.DataFrame({f"s{i}": np.arange(40.0) * (i + 1) + 100 for i in range(4)}, index=index)
    # min_symbols=5 时只有 4 列可用 -> 全部截面被跳过，返回空序列而不是拿 4 个样本凑相关
    empty = ic_of_factor(factor, close, horizon=1, dates=list(index), min_symbols=5)
    assert empty.empty
    four = ic_of_factor(factor, close, horizon=1, dates=list(index), min_symbols=4)
    assert not four.empty


def test_ic_of_factor_respects_the_forward_window_at_the_tail():
    """尾部没有前瞻数据的截面必须被丢掉：n 个截面、前瞻 h 日 -> 最多 n-h 个有效截面。"""
    n, horizon = 10, 3
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(5)
    shocks = rng.normal(size=(n, 6))
    factor = pd.DataFrame({f"s{i}": rng.normal(size=n) for i in range(6)}, index=index)
    close = pd.DataFrame(100 + np.cumsum(shocks, axis=0), index=index, columns=[f"s{i}" for i in range(6)])
    ics = ic_of_factor(factor, close, horizon=horizon, dates=list(index), min_symbols=3)
    assert len(ics) == n - horizon, "前 n-horizon 个截面才有效"


def test_evaluate_returns_three_windows_and_corrections(universe):
    config = FactorEvalConfig(horizons=(5,), step_days=5, min_history=200, min_symbols=5, alpha=0.05, is_fraction=0.5)
    outcome = evaluate_factor_ics(universe, config)
    table = outcome["table"]
    assert not table.empty
    for column in ("ic_mean", "t_stat_adj", "is_ic_mean", "is_t_stat_adj", "os_ic_mean", "os_t_stat_adj"):
        assert column in table.columns, f"缺少 {column}"
    assert {"is_significant_bh", "os_significant", "survives_oos", "os_same_sign"} <= set(table.columns)
    assert outcome["split_date"] is not None
    # 样本内 + 样本外的截面数之和不应超过全样本（步长一致时大致各半）
    total = table["periods"].iloc[0]
    inside = table["is_periods"].iloc[0]
    outside = table["os_periods"].iloc[0]
    assert inside + outside <= total + 2, "两段之和不应显著超过全样本"
    assert inside > 0 and outside > 0


def test_survives_oos_requires_significance_and_same_sign(universe):
    """survives_oos 必须同时满足：样本内 BH 显著、样本外显著、方向一致。"""
    config = FactorEvalConfig(horizons=(5,), step_days=10, min_history=200, min_symbols=5, is_fraction=0.5)
    table = evaluate_factor_ics(universe, config)["table"]
    flagged = table[table["survives_oos"]]
    for row in flagged.itertuples(index=False):
        assert bool(row.is_significant_bh) is True
        assert bool(row.os_significant) is True
        assert bool(row.os_same_sign) is True
    # 反向的例子必须被排除
    reversed_rows = table[(~table["os_same_sign"]) & table["is_significant_bh"]]
    assert not reversed_rows["survives_oos"].any(), "样本外反向的组合不能被判为通过"


def test_evaluate_rejects_unusable_input():
    with pytest.raises(ValueError):
        evaluate_factor_ics({}, FactorEvalConfig(horizons=(5,), min_history=10, min_symbols=3))
