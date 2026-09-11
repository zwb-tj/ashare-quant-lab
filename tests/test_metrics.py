import numpy as np
import pandas as pd
import pytest

from aqlab.backtest import BacktestConfig, run_backtest
from aqlab.data import generate_synthetic_ohlcv
from aqlab.metrics import compute_metrics, drawdown_series, format_metrics


def _flat_frame(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    rets = pd.Series(values, index=idx, dtype=float)
    equity = (1 + rets).cumprod()
    return pd.DataFrame({"strat_ret": rets, "equity": equity, "pos": 1.0, "turnover": 0.0})


def test_max_drawdown_is_exact():
    frame = _flat_frame([0.10, -0.50, 0.20])
    metrics = compute_metrics(frame, initial_cash=1.0)
    # equity: 1.10 -> 0.55 -> 0.66 ; dd min = 0.55/1.10 - 1 = -0.5
    assert metrics["max_drawdown"] == pytest.approx(-0.5)
    assert (drawdown_series(frame["equity"]) <= 0).all()


def test_cagr_of_doubling_in_one_year():
    n = 252
    rets = pd.Series(np.zeros(n))
    rets.iloc[-1] = 1.0  # total return +100% over one year
    equity = (1 + rets).cumprod()
    frame = pd.DataFrame({"strat_ret": rets, "equity": equity})
    metrics = compute_metrics(frame, initial_cash=1.0, periods_per_year=252)
    assert metrics["total_return"] == pytest.approx(1.0)
    assert metrics["cagr"] == pytest.approx(1.0, rel=1e-6)


def test_sharpe_sign_follows_mean_return():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 500))
    bad = pd.Series(rng.normal(-0.001, 0.01, 500))
    for series, expected_sign in ((good, 1), (bad, -1)):
        equity = (1 + series).cumprod()
        frame = pd.DataFrame({"strat_ret": series, "equity": equity})
        metrics = compute_metrics(frame, initial_cash=1.0)
        assert np.sign(metrics["sharpe"]) == expected_sign
        assert metrics["ann_vol"] > 0


def test_profit_factor_and_win_rate_on_synthetic_backtest():
    df = generate_synthetic_ohlcv(n_days=400, seed=13)
    from aqlab.strategies import MACrossStrategy

    strategy = MACrossStrategy(fast=5, slow=20)
    result = run_backtest(df, strategy.positions(df), config=BacktestConfig())
    metrics = compute_metrics(result.frame, initial_cash=result.config.initial_cash, trades=result.trades)
    assert 0.0 <= metrics["win_rate_daily"] <= 1.0
    assert metrics["profit_factor"] >= 0
    assert metrics["annual_turnover"] >= 0
    assert 0.0 <= metrics["exposure"] <= 1.0
    assert "trades" in metrics


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        compute_metrics(pd.DataFrame({"equity": [1.0, 1.1]}))


def test_format_metrics_renders_percentages():
    frame = _flat_frame([0.01, 0.02, -0.01])
    metrics = compute_metrics(frame, initial_cash=1.0)
    text = format_metrics(metrics)
    assert "| total_return |" in text
    assert "%" in text
