"""可视化模块的测试（离线）。

matplotlib 是可选依赖，所以：**装了就把图真画出来并校验是有效 PNG；没装就跳过**
——这样 CI 不需要额外的绘图依赖，而本地开发能真的验证渲染路径。
"""

import struct

import pandas as pd
import pytest

from aqlab.backtest import BacktestConfig, run_backtest
from aqlab.data import generate_synthetic_ohlcv
from aqlab.metrics import compute_metrics

matplotlib = pytest.importorskip("matplotlib", reason="charts need the optional 'plot' extra")

from aqlab.charts import plot_drawdown, plot_equity_curves, plot_strategy_comparison


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "不是有效 PNG"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@pytest.fixture(scope="module")
def results():
    df = generate_synthetic_ohlcv(n_days=320, seed=25)
    config = BacktestConfig(fee_bps=3.0, slippage_bps=2.0)
    output = {}
    for name, strategy in (("ma_cross", "ma_cross"), ("mean_reversion", "mean_reversion")):
        from aqlab.strategies import build_strategy

        built = build_strategy(strategy)
        result = run_backtest(df, built.positions(df), config=config, name=built.name)
        metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
        output[name] = (df, result, metrics)
    return output


def test_equity_curves_render_two_series_and_a_benchmark(tmp_path, results):
    curves = {name: result.frame for name, (_df, result, _m) in results.items()}
    df = next(iter(results.values()))[0]
    benchmark = df["close"] / df["close"].iloc[0] * 100_000.0
    path = plot_equity_curves(curves, tmp_path / "equity.png", benchmark=benchmark)
    width, height = png_size(path)
    assert width > 600 and height > 300


def test_drawdown_uses_the_equity_column(tmp_path, results):
    _df, result, _metrics = next(iter(results.values()))
    path = plot_drawdown(result.frame, tmp_path / "dd.png")
    assert path.exists() and png_size(path)[0] > 600


def test_strategy_comparison_accepts_a_summary_frame(tmp_path, results):
    summary = pd.DataFrame(
        [
            {"strategy": "ma_cross", "total_return": 0.12, "sharpe": 0.8, "max_drawdown": -0.15, "win_rate_trade": 0.5},
            {"strategy": "mean_reversion", "total_return": -0.05, "sharpe": -0.3, "max_drawdown": -0.22, "win_rate_trade": 0.4},
        ]
    )
    path = plot_strategy_comparison(summary, tmp_path / "cmp.png")
    assert path.exists() and png_size(path)[0] > 300


def test_missing_equity_column_is_reported(tmp_path):
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError):
        plot_equity_curves({"bad": frame}, tmp_path / "bad.png")


def test_comparison_rejects_unknown_metric_columns(tmp_path):
    summary = pd.DataFrame([{"strategy": "x", "unrelated": 1.0}])
    with pytest.raises(ValueError):
        plot_strategy_comparison(summary, tmp_path / "x.png")
