"""Performance metrics — one place, explicit formulas, no hidden conventions.

All annualisation uses ``periods_per_year`` (default 252 trading days) and a
zero risk-free rate. Every metric is computed from the *net* strategy returns
already including fees and slippage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["compute_metrics", "format_metrics", "drawdown_series"]

_PCT_KEYS = {
    "total_return",
    "cagr",
    "ann_vol",
    "max_drawdown",
    "win_rate_daily",
    "win_rate_trade",
    "exposure",
    "best_day",
    "worst_day",
    "annual_turnover",
}


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown path (<= 0) of an equity curve."""
    equity = equity.astype(float)
    return equity / equity.cummax() - 1.0


def compute_metrics(
    frame: pd.DataFrame,
    initial_cash: float | None = None,
    periods_per_year: int = 252,
    trades: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute a standard metric bundle from a backtest frame.

    ``frame`` must contain ``strat_ret`` and ``equity`` (see
    :class:`aqlab.backtest.BacktestResult`).
    """
    if "strat_ret" not in frame or "equity" not in frame:
        raise ValueError("frame must contain 'strat_ret' and 'equity' columns")
    rets = frame["strat_ret"].astype(float).fillna(0.0)
    equity = frame["equity"].astype(float)
    start = float(initial_cash) if initial_cash is not None else float(equity.iloc[0])
    if start <= 0:
        raise ValueError("initial equity must be positive")

    n_bars = len(rets)
    years = n_bars / float(periods_per_year) if n_bars else 0.0
    end = float(equity.iloc[-1])
    total_return = end / start - 1.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if years > 0 and end > 0 else np.nan

    std = float(rets.std(ddof=0))
    mean = float(rets.mean())
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = mean / std * np.sqrt(periods_per_year) if std > 0 else np.nan

    downside = rets[rets < 0]
    down_std = float(downside.std(ddof=0)) if len(downside) > 1 else np.nan
    sortino = mean / down_std * np.sqrt(periods_per_year) if down_std and down_std > 0 else np.nan

    dd = drawdown_series(equity)
    max_drawdown = float(dd.min()) if len(dd) else np.nan
    calmar = cagr / abs(max_drawdown) if max_drawdown and max_drawdown < 0 else np.nan

    gains = float(rets[rets > 0].sum())
    losses = float(rets[rets < 0].sum())
    profit_factor = gains / abs(losses) if losses < 0 else np.inf

    metrics: dict[str, Any] = {
        "bars": n_bars,
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "sortino": float(sortino) if sortino == sortino else np.nan,
        "max_drawdown": max_drawdown,
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "win_rate_daily": float((rets > 0).mean()),
        "profit_factor": float(profit_factor),
        "best_day": float(rets.max()) if n_bars else np.nan,
        "worst_day": float(rets.min()) if n_bars else np.nan,
    }

    if "turnover" in frame:
        metrics["annual_turnover"] = float(frame["turnover"].fillna(0.0).mean() * periods_per_year)
    if "pos" in frame:
        metrics["exposure"] = float((frame["pos"].fillna(0.0) != 0).mean())

    if trades is not None and len(trades):
        closed = trades[~trades.get("open", False).astype(bool)] if "open" in trades else trades
        closed = closed if len(closed) else trades
        net = closed["net_return"].astype(float)
        metrics["trades"] = int(len(closed))
        metrics["win_rate_trade"] = float((net > 0).mean())
        metrics["avg_trade"] = float(net.mean())
        metrics["avg_bars_held"] = float(closed["bars_held"].mean())
        metrics["best_trade"] = float(net.max())
        metrics["worst_trade"] = float(net.min())

    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    """Render a metric dict as a compact markdown table."""
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in _PCT_KEYS:
                lines.append(f"| {key} | {value * 100:.2f}% |")
            elif value != value:  # NaN
                lines.append(f"| {key} | n/a |")
            else:
                lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines)
