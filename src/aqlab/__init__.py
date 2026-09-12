"""aqlab — A-share Quant Lab.

An original, reproducible toolkit for screening and backtesting A-share
strategies, plus a tool-using LLM research agent with a measurable evaluation
harness (grounding, hallucination and abstention metrics).

Clean-room implementation: this package contains no third-party code or
content; it was written from scratch for this repository.
"""

from aqlab.agent import AgentResult, OpenAICompatClient, ResearchAgent, ScriptedClient
from aqlab.backtest import BacktestConfig, BacktestResult, run_backtest, run_portfolio
from aqlab.data import TushareDataSource, generate_synthetic_ohlcv, load_ohlcv_csv, normalize_ohlcv
from aqlab.evaluation import default_tasks, run_eval
from aqlab.intraday import IntradayConfig, confirm_signals, opening_volume_ratio
from aqlab.metrics import compute_metrics
from aqlab.notify import ConsoleNotifier, FeishuWebhookNotifier, build_feishu_card
from aqlab.pipeline import DailyConfig, DailyPipeline, write_daily_report
from aqlab.portfolio import PortfolioConfig, exposure_report, optimize_weights, simulate_portfolio
from aqlab.position import PositionConfig, defend_score, plan_position, simulate_exit, simulate_signals
from aqlab.profiles import build_gate, list_profiles, load_profile
from aqlab.quality import QualityConfig, audit_universe, cross_source_diff, snapshot_hash
from aqlab.rules import DEFAULT_RULE_BINDINGS, ActivityValueGate, NeedleBelowMA, TieredPullback, VolumePriceSurge, build_rule
from aqlab.screen import rank_universe
from aqlab.strategies import STRATEGIES, MACrossStrategy, MeanReversionStrategy, MomentumStrategy
from aqlab.study import forward_returns, rule_event_study, study_profile
from aqlab.sweep import SweepConfig, parse_grid, pick_best, run_sweep
from aqlab.tools import ToolRegistry, default_registry

__version__ = "0.8.0"

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
    "TushareDataSource",
    "DailyConfig",
    "DailyPipeline",
    "write_daily_report",
    "TieredPullback",
    "NeedleBelowMA",
    "VolumePriceSurge",
    "ActivityValueGate",
    "DEFAULT_RULE_BINDINGS",
    "build_rule",
    "ConsoleNotifier",
    "FeishuWebhookNotifier",
    "build_feishu_card",
    "PositionConfig",
    "plan_position",
    "defend_score",
    "simulate_exit",
    "simulate_signals",
    "load_profile",
    "list_profiles",
    "build_gate",
    "forward_returns",
    "rule_event_study",
    "study_profile",
    "SweepConfig",
    "run_sweep",
    "pick_best",
    "parse_grid",
    "IntradayConfig",
    "confirm_signals",
    "opening_volume_ratio",
    "QualityConfig",
    "audit_universe",
    "cross_source_diff",
    "snapshot_hash",
    "__version__",
]
