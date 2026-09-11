"""Offline demo: strategies + screening on deterministic synthetic data.

Run with:  python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aqlab.backtest import BacktestConfig, run_backtest  # noqa: E402
from aqlab.data import generate_synthetic_ohlcv, make_universe  # noqa: E402
from aqlab.metrics import compute_metrics  # noqa: E402
from aqlab.screen import ScreenConfig, rank_universe  # noqa: E402
from aqlab.strategies import build_strategy  # noqa: E402


def main() -> None:
    df = generate_synthetic_ohlcv(n_days=750, seed=7)
    config = BacktestConfig()

    rows = []
    for name in ("buy_and_hold", "ma_cross", "momentum", "mean_reversion"):
        strategy = build_strategy(name)
        result = run_backtest(df, strategy.positions(df), config=config, name=strategy.name)
        metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
        rows.append(
            {
                "策略": name,
                "总收益%": round(metrics["total_return"] * 100, 2),
                "年化%": round(metrics["cagr"] * 100, 2),
                "年化波动%": round(metrics["ann_vol"] * 100, 2),
                "Sharpe": round(metrics["sharpe"], 3),
                "最大回撤%": round(metrics["max_drawdown"] * 100, 2),
                "交易数": metrics.get("trades"),
                "胜率%": round(metrics.get("win_rate_trade", float("nan")) * 100, 2),
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))

    universe = make_universe(n_symbols=30, n_days=500, seed=11)
    ranks = rank_universe(universe, config=ScreenConfig(top_n=10))
    print()
    print(ranks[["rank", "symbol", "mom_60", "vol_20", "score"]].head(10).round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
