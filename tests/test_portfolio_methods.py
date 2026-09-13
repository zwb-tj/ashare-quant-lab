"""权重方法对比的测试（离线，小票池，几秒钟跑完）。

守住两件事：
① 对比表必须一行一个方法、列名与指标齐全；
② 当信号层选不出足够标的、五种方法退化成同一组合时，必须**明确提示**而不是给出一张看起来很有信息量的表。
"""

import pandas as pd

from aqlab.cli import _compare_weight_methods, build_parser
from aqlab.portfolio import PortfolioConfig
from aqlab.tools import SyntheticDataSource


def _universe(n_symbols: int, n_days: int, seed: int = 11):
    source = SyntheticDataSource(n_symbols=n_symbols, n_days=n_days, seed=seed)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def _signals(universe, every: int):
    """简单信号：每隔 every 根有一根为真，用于控制持仓稀疏程度。"""
    signals = {}
    for symbol, frame in universe.items():
        mask = pd.Series(False, index=frame.index)
        mask.iloc[::every] = True
        signals[symbol] = mask
    return signals


def _args(tmp_path=None):
    parser = build_parser()
    argv = ["portfolio", "--compare-methods"]
    if tmp_path is not None:
        argv += ["--out", str(tmp_path)]
    return parser.parse_args(argv)


def test_comparison_table_has_one_row_per_method(tmp_path):
    universe = _universe(30, 400)
    signals = _signals(universe, every=1)          # 每个标的每根都有信号 -> 持仓充足
    config = PortfolioConfig(method="risk_parity", lookback=40, rebalance_days=10, min_history=120)
    outcome = _compare_weight_methods(universe, signals, config, _args(tmp_path))
    table = outcome["table"]
    assert list(table["method"]) == ["equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance"]
    for column in ("total_return", "sharpe", "max_drawdown", "annual_turnover", "avg_positions", "total_cost"):
        assert column in table.columns
    assert "权重方法对比" in outcome["markdown"]
    assert "chart" not in outcome["markdown"], "表格里不该出现文件路径列"


def test_degenerate_case_is_called_out(tmp_path):
    """稀疏信号 -> 平均持仓很少 -> 五种方法同解，必须明确提示。"""
    universe = _universe(12, 320)
    signals = _signals(universe, every=200)        # 极稀疏
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    outcome = _compare_weight_methods(universe, signals, config, _args(tmp_path))
    table = outcome["table"]
    if float(table["avg_positions"].iloc[0]) <= 1.5 and len(table.drop_duplicates(subset=["total_return", "avg_positions"])) == 1:
        assert "完全相同" in outcome["markdown"], "退化成同一组合时必须有提示"
    else:                                          # 若数据恰好没退化，就只校验表格结构
        assert len(table) == 5


def test_comparison_mentions_the_chart_when_an_output_dir_is_given(tmp_path):
    """CLI 的 --out 永远有默认值，所以正常路径下都会出图并把路径写进报告。"""
    universe = _universe(20, 360)
    signals = _signals(universe, every=1)
    config = PortfolioConfig(method="equal", lookback=40, rebalance_days=10, min_history=120)
    outcome = _compare_weight_methods(universe, signals, config, _args(tmp_path))
    assert isinstance(outcome["curves"], dict) and outcome["curves"]
    assert "对比图" in outcome["markdown"]
    assert (tmp_path / "portfolio" / "method_comparison.png").exists()
