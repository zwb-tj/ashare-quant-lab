"""aqlab — A-share Quant Lab.

An original, reproducible toolkit for screening and backtesting A-share
strategies, with a clearly separated LLM/agent layer on the roadmap.

Clean-room implementation: this package contains no third-party code or
content; it was written from scratch for this repository.
"""

from aqlab.backtest import BacktestConfig, BacktestResult, run_backtest, run_portfolio
from aqlab.data import generate_synthetic_ohlcv, load_ohlcv_csv, normalize_ohlcv
from aqlab.metrics import compute_metrics
from aqlab.screen import rank_universe
from aqlab.strategies import STRATEGIES, MACrossStrategy, MeanReversionStrategy, MomentumStrategy

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "run_portfolio",
    "generate_synthetic_ohlcv",
    "load_ohlcv_csv",
    "normalize_ohlcv",
    "compute_metrics",
    "rank_universe",
    "STRATEGIES",
    "MACrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "__version__",
]
