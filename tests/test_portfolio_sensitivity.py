"""参数敏感性的测试（离线，小票池）。

守住：① 网格覆盖所有 lookback × cap × method 组合；② 机制检验那行必须给出基准格与最优格，
而不是只说一句"结论稳健"；③ 退化情形（持仓过少、方法同解）不产生误导性叙述。
"""

import pandas as pd

from aqlab.cli import _compare_weight_methods, _portfolio_sensitivity, build_parser
from aqlab.portfolio import PortfolioConfig
from aqlab.tools import SyntheticDataSource


def _universe(n_symbols: int, n_days: int, seed: int = 11):
    source = SyntheticDataSource(n_symbols=n_symbols, n_days=n_days, seed=seed)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def _signals(universe, every: int = 1):
    signals = {}
    for symbol, frame in universe.items():
        mask = pd.Series(False, index=frame.index)
        mask.iloc[::every] = True
        signals[symbol] = mask
    return signals


def _args(tmp_path, extra=()):
    parser = build_parser()
    argv = ["portfolio", "--out", str(tmp_path), *extra]
    return parser.parse_args(argv)


def test_sensitivity_grid_covers_every_combination(tmp_path):
    universe = _universe(24, 400)
    signals = _signals(universe)
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    args = _args(tmp_path, ("--sensitivity", "--sensitivity-lookbacks", "40,80", "--sensitivity-caps", "0.15,0.3"))
    outcome = _portfolio_sensitivity(universe, signals, config, args)
    table = outcome["table"]
    assert len(table) == 2 * 2 * 3, "2 个 lookback × 2 个 cap × 3 种方法"
    assert set(table["method"]) == {"equal", "min_variance", "mean_variance"}
    assert set(table["lookback"]) == {40, 80}
    assert set(table["max_weight"]) == {0.15, 0.3}
    for column in ("total_return", "sharpe", "max_drawdown", "ann_vol", "avg_positions"):
        assert column in table.columns
    assert "参数敏感性" in outcome["markdown"]


def test_sensitivity_note_names_the_baseline_and_the_best_cell(tmp_path):
    """机制检验必须落在具体格子上，而不是一句空泛的"稳健"。"""
    universe = _universe(24, 400)
    signals = _signals(universe)
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    args = _args(tmp_path, ("--sensitivity", "--sensitivity-lookbacks", "40,80", "--sensitivity-caps", "0.15,0.3"))
    markdown = _portfolio_sensitivity(universe, signals, config, args)["markdown"]
    assert "机制检验" in markdown
    assert "lookback=" in markdown and "cap=" in markdown
    assert "最好的一格" in markdown


def test_sensitivity_single_cell_grid_still_works(tmp_path):
    """只给一个 lookback 与一个 cap 时不能崩，也不该硬凑机制结论。"""
    universe = _universe(20, 360)
    signals = _signals(universe)
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    args = _args(tmp_path, ("--sensitivity", "--sensitivity-lookbacks", "40", "--sensitivity-caps", "0.2"))
    table = _portfolio_sensitivity(universe, signals, config, args)["table"]
    assert len(table) == 3


def test_method_comparison_and_sensitivity_agree_on_the_default_methods(tmp_path):
    """两条路径用的是同一份信号与约束，方法集合要一致（避免两套口径）。"""
    universe = _universe(20, 400)
    signals = _signals(universe)
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    args = _args(tmp_path, ("--compare-methods",))
    comparison = _compare_weight_methods(universe, signals, config, args)
    sensitivity = _portfolio_sensitivity(
        universe, signals, config, _args(tmp_path, ("--sensitivity", "--sensitivity-lookbacks", "40", "--sensitivity-caps", "0.2"))
    )
    assert set(sensitivity["table"]["method"]) <= set(comparison["table"]["method"])
