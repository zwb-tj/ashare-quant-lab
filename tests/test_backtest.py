import numpy as np
import pandas as pd
import pytest

from aqlab.backtest import BacktestConfig, run_backtest, run_portfolio
from aqlab.data import generate_synthetic_ohlcv, make_universe
from aqlab.strategies import MACrossStrategy, MomentumStrategy, build_strategy


def _frame(closes):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": np.array(closes) * 1.001, "low": np.array(closes) * 0.999, "close": closes, "volume": 1.0},
        index=idx,
    )


def test_no_lookahead_oracle_loses_money():
    """A 'cheating' signal based on today's return must be worthless after the shift.

    With alternating +1% / -1% returns, sign(ret_t) == -sign(ret_{t-1}); after the
    one-bar execution delay the strategy systematically trades the wrong way.
    """
    n = 60
    rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    closes = 100 * np.cumprod(1 + rets)
    df = _frame(closes)

    realized = pd.Series(rets, index=df.index)
    oracle = np.sign(realized)  # uses today's return -> impossible in real life
    result = run_backtest(df, oracle, config=BacktestConfig(fee_bps=0, slippage_bps=0))
    assert result.equity.iloc[-1] < result.config.initial_cash
    assert result.returns.sum() < 0


def test_full_position_equals_buy_and_hold_minus_costs():
    df = generate_synthetic_ohlcv(n_days=300, seed=8)
    ones = pd.Series(1.0, index=df.index)

    free = run_backtest(df, ones, config=BacktestConfig(fee_bps=0, slippage_bps=0))
    costly = run_backtest(df, ones, config=BacktestConfig(fee_bps=10, slippage_bps=10))

    bh_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    free_return = free.equity.iloc[-1] / free.config.initial_cash - 1.0
    # full position from bar 2 onward (one-bar execution delay) == buy & hold
    assert np.isclose(free_return, bh_return, atol=1e-9)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.frame["cost"].sum() > 0


def test_long_only_clips_short_signals():
    df = generate_synthetic_ohlcv(n_days=200, seed=9)
    shorts = pd.Series(-1.0, index=df.index)
    result = run_backtest(df, shorts, config=BacktestConfig())
    assert (result.positions == 0).all()
    assert np.isclose(result.equity.iloc[-1], result.config.initial_cash)

    allowed = run_backtest(df, shorts, config=BacktestConfig(allow_short=True, fee_bps=0, slippage_bps=0))
    assert (allowed.positions <= 0).all()
    assert allowed.positions.iloc[1] == -1.0


def test_positions_are_shifted_by_one_bar():
    df = generate_synthetic_ohlcv(n_days=50, seed=10)
    signal = pd.Series(0.0, index=df.index)
    signal.iloc[5:] = 1.0
    result = run_backtest(df, signal, config=BacktestConfig())
    assert result.positions.iloc[5] == 0.0
    assert result.positions.iloc[6] == 1.0


def test_trades_blotter_consistency():
    df = generate_synthetic_ohlcv(n_days=400, seed=12)
    strategy = MACrossStrategy(fast=10, slow=30)
    result = run_backtest(df, strategy.positions(df), config=BacktestConfig())
    trades = result.trades
    assert {"entry_date", "exit_date", "direction", "net_return"}.issubset(trades.columns)
    assert len(trades) >= 1
    for _, row in trades.iterrows():
        assert row["entry_date"] <= row["exit_date"]
        assert row["bars_held"] >= 0
        gross = row["exit_price"] / row["entry_price"] - 1.0
        if row["direction"] == "short":
            gross = -gross
        assert row["gross_return"] == pytest.approx(gross)


def test_portfolio_aggregates_equal_weight():
    universe = make_universe(n_symbols=4, n_days=400, seed=21)
    result = run_portfolio(universe, MomentumStrategy(lookback=60, trend_window=120), config=BacktestConfig())
    assert len(result.frame) == 400
    assert result.equity.notna().all()
    assert result.name == "portfolio"


def test_build_strategy_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_strategy("does_not_exist")


def test_ma_cross_validates_parameters():
    with pytest.raises(ValueError):
        MACrossStrategy(fast=30, slow=10)
