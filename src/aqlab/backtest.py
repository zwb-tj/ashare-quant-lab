"""Execution-aware backtest engine.

Key realism rules (these are what make the numbers credible):

1. **No lookahead** — a signal computed on bar ``t`` is executed at bar ``t+1``
   (``positions.shift(1)``). Everything else in the package is written so this
   single shift is sufficient.
2. **Costs** — every unit of position change pays ``fee_bps + slippage_bps``.
3. **Long-only by default** — negative targets are clipped to 0 unless
   ``allow_short`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from aqlab.strategies import Strategy

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "run_portfolio"]


@dataclass
class BacktestConfig:
    """Execution assumptions, in basis points (1 bps = 0.01%)."""

    initial_cash: float = 1_000_000.0
    fee_bps: float = 3.0
    slippage_bps: float = 2.0
    allow_short: bool = False
    max_position: float = 1.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be in (0, 1]")

    @property
    def cost_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    """A backtest output: per-bar frame plus the realised trade list."""

    frame: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig
    name: str = "strategy"

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.frame["strat_ret"]

    @property
    def positions(self) -> pd.Series:
        return self.frame["pos"]

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def _extract_trades(pos: pd.Series, close: pd.Series, cost_rate: float) -> pd.DataFrame:
    """Turn an executed position series into a trade blotter."""
    rows: list[dict] = []
    current = 0.0
    entry_date = None
    entry_price = np.nan

    for date, target in pos.items():
        price = float(close.loc[date])
        if target == current:
            continue
        if current != 0.0 and entry_date is not None:
            gross = current * (price / entry_price - 1.0)
            rows.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long" if current > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "bars_held": int(pos.loc[entry_date:date].shape[0] - 1),
                    "gross_return": gross,
                    "net_return": gross - 2.0 * cost_rate * abs(current),
                }
            )
        current = float(target)
        entry_date, entry_price = (date, price) if current != 0.0 else (None, np.nan)

    if current != 0.0 and entry_date is not None:
        price = float(close.iloc[-1])
        gross = current * (price / entry_price - 1.0)
        rows.append(
            {
                "entry_date": entry_date,
                "exit_date": close.index[-1],
                "direction": "long" if current > 0 else "short",
                "entry_price": entry_price,
                "exit_price": price,
                "bars_held": int(pos.loc[entry_date:].shape[0] - 1),
                "gross_return": gross,
                "net_return": gross - 2.0 * cost_rate * abs(current),
                "open": True,
            }
        )

    columns = [
        "entry_date",
        "exit_date",
        "direction",
        "entry_price",
        "exit_price",
        "bars_held",
        "gross_return",
        "net_return",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    trades = pd.DataFrame(rows)
    if "open" not in trades.columns:
        trades["open"] = False
    trades["open"] = trades["open"].fillna(False).astype(bool)
    return trades[[*columns, "open"]]


def run_backtest(df: pd.DataFrame, positions: pd.Series, config: BacktestConfig | None = None, name: str = "strategy") -> BacktestResult:
    """Run a single-asset backtest.

    Parameters
    ----------
    df:
        Normalized OHLCV frame (see :func:`aqlab.data.normalize_ohlcv`).
    positions:
        Target position series produced by a strategy (in ``[-1, 1]``).
    """
    config = config or BacktestConfig()
    if df.empty:
        raise ValueError("empty price frame")

    close = df["close"].astype(float)
    bar_return = close.pct_change().fillna(0.0)

    lower = -config.max_position if config.allow_short else 0.0
    pos = positions.reindex(df.index).astype(float).clip(lower, config.max_position)
    pos = pos.shift(1).fillna(0.0)  # <- execution delay: no lookahead

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    cost = turnover * config.cost_rate
    strat_ret = pos * bar_return - cost
    equity = config.initial_cash * (1.0 + strat_ret).cumprod()

    frame = pd.DataFrame(
        {
            "close": close,
            "bar_ret": bar_return,
            "pos": pos,
            "turnover": turnover,
            "cost": cost,
            "strat_ret": strat_ret,
            "equity": equity,
        }
    )
    trades = _extract_trades(pos, close, config.cost_rate)
    return BacktestResult(frame=frame, trades=trades, config=config, name=name)


def run_portfolio(
    universe: Mapping[str, pd.DataFrame],
    strategy: Strategy | Callable[[], Strategy],
    config: BacktestConfig | None = None,
    weights: Mapping[str, float] | None = None,
) -> BacktestResult:
    """Equal-weight (or custom-weight) portfolio backtest across symbols.

    Each symbol is backtested independently with the same strategy; portfolio
    returns are the weighted average of per-symbol strategy returns. This is a
    deliberately simple, transparent aggregation — no cross-sectional
    rebalancing fictions.
    """
    config = config or BacktestConfig()
    if not universe:
        raise ValueError("universe is empty")

    frames: dict[str, pd.DataFrame] = {}
    for symbol, df in universe.items():
        strat = strategy() if callable(strategy) else strategy
        res = run_backtest(df, strat.positions(df), config=config, name=getattr(strat, "name", "strategy"))
        frames[symbol] = res.frame["strat_ret"]

    rets = pd.DataFrame(frames).sort_index()
    if weights is None:
        w = pd.Series(1.0 / len(rets.columns), index=rets.columns)
    else:
        w = pd.Series(weights, dtype=float).reindex(rets.columns).fillna(0.0)
        if w.sum() <= 0:
            raise ValueError("weights must sum to a positive number")
        w = w / w.sum()

    port_ret = (rets.fillna(0.0) * w).sum(axis=1)
    equity = config.initial_cash * (1.0 + port_ret).cumprod()
    frame = pd.DataFrame({"strat_ret": port_ret, "equity": equity})
    return BacktestResult(frame=frame, trades=pd.DataFrame(), config=config, name="portfolio")
