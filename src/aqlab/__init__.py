"""aqlab — A-share Quant Lab.

An original, reproducible toolkit for screening and backtesting A-share
strategies, plus a tool-using LLM research agent with a measurable evaluation
harness (grounding, hallucination and abstention metrics).

Clean-room implementation: this package contains no third-party code or
content; it was written from scratch for this repository.
"""

from aqlab.agent import AgentResult, OpenAICompatClient, ResearchAgent, ScriptedClient
from aqlab.backtest import BacktestConfig, BacktestResult, run_backtest, run_portfolio
from aqlab.data import generate_synthetic_ohlcv, load_ohlcv_csv, normalize_ohlcv
from aqlab.evaluation import default_tasks, run_eval
from aqlab.metrics import compute_metrics
from aqlab.screen import rank_universe
from aqlab.strategies import STRATEGIES, MACrossStrategy, MeanReversionStrategy, MomentumStrategy
from aqlab.tools import ToolRegistry, default_registry

__version__ = "0.2.0"

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
    "ToolRegistry",
    "default_registry",
    "ResearchAgent",
    "OpenAICompatClient",
    "ScriptedClient",
    "AgentResult",
    "run_eval",
    "default_tasks",
    "__version__",
]
