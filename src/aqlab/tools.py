"""Read-only tool layer for the LLM research agent.

Design rules (these are the guardrails that make the agent auditable):

* **Read-only.** Tools never write files, never place orders, never mutate state.
* **Schema'd.** Every tool declares a JSON-schema parameter block, so an LLM can
  call it through standard tool-calling protocols.
* **Typed results.** Every tool returns a JSON-serialisable dict with ``ok`` and
  either payload fields or an ``error`` string. Errors are *data*, not crashes —
  a missing symbol must come back as ``ok=False`` so the model can abstain
  instead of hallucinating numbers.
* **Numbers only from tools.** The evaluation harness grounds every number in an
  answer against the concatenated tool outputs; anything else counts as a
  hallucination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from aqlab.backtest import BacktestConfig, run_backtest
from aqlab.data import load_ohlcv_csv, make_universe
from aqlab.indicators import atr, ema, pct_change_n, realized_vol, rolling_zscore, rsi, sma
from aqlab.metrics import compute_metrics
from aqlab.screen import ScreenConfig, rank_universe
from aqlab.strategies import STRATEGIES, build_strategy

__all__ = [
    "DataSource",
    "SyntheticDataSource",
    "CsvDataSource",
    "ToolRegistry",
    "ToolSpec",
    "INDICATORS",
    "default_registry",
]


# --------------------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------------------
class DataSource(Protocol):
    """Minimal read-only interface the tools depend on."""

    def symbols(self) -> list[str]: ...

    def bars(self, symbol: str) -> pd.DataFrame: ...

    def describe(self) -> dict: ...


class SyntheticDataSource:
    """Deterministic offline universe — the default so everything runs without a network."""

    def __init__(self, n_symbols: int = 30, n_days: int = 500, seed: int = 11) -> None:
        self._universe = make_universe(n_symbols=n_symbols, n_days=n_days, seed=seed)
        self._seed = seed

    def symbols(self) -> list[str]:
        return sorted(self._universe)

    def bars(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._universe:
            raise KeyError(f"unknown symbol '{symbol}'")
        return self._universe[symbol]

    def describe(self) -> dict:
        first = min(df.index[0] for df in self._universe.values())
        last = max(df.index[-1] for df in self._universe.values())
        return {
            "source": "synthetic",
            "seed": self._seed,
            "symbols": self.symbols(),
            "bars_per_symbol": int(len(next(iter(self._universe.values())))),
            "first_date": str(first.date()),
            "last_date": str(last.date()),
        }


class CsvDataSource:
    """A directory of OHLCV CSV files; each file stem is a symbol."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"data directory not found: {self.directory}")
        self._cache: dict[str, pd.DataFrame] = {}

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.csv"))

    def bars(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            path = self.directory / f"{symbol}.csv"
            if not path.exists():
                raise KeyError(f"unknown symbol '{symbol}'")
            self._cache[symbol] = load_ohlcv_csv(path)
        return self._cache[symbol]

    def describe(self) -> dict:
        symbols = self.symbols()
        return {
            "source": "csv",
            "directory": str(self.directory),
            "symbols": symbols,
            "bars_per_symbol": int(len(self.bars(symbols[0]))) if symbols else 0,
        }


# --------------------------------------------------------------------------------------
# Indicators exposed to the model
# --------------------------------------------------------------------------------------
INDICATORS: dict[str, Callable[[pd.DataFrame, int], pd.Series]] = {
    "sma": lambda df, w: sma(df["close"], w),
    "ema": lambda df, w: ema(df["close"], w),
    "rsi": lambda df, w: rsi(df["close"], w),
    "atr": lambda df, w: atr(df, w),
    "zscore": lambda df, w: rolling_zscore(df["close"], w),
    "realized_vol": lambda df, w: realized_vol(df["close"], w),
    "momentum": lambda df, w: pct_change_n(df["close"], w),
}


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    func: Callable[..., dict]

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _ok(**payload: Any) -> dict:
    return {"ok": True, **payload}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        if value != value or np.isinf(value):
            return None
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _require(arguments: Mapping[str, Any], keys: list[str]) -> str | None:
    missing = [k for k in keys if k not in arguments or arguments[k] in (None, "")]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    return None


class ToolRegistry:
    """Holds the toolset and dispatches calls with validation + error capture."""

    def __init__(self, source: DataSource | None = None, default_config: BacktestConfig | None = None) -> None:
        self.source = source or SyntheticDataSource()
        self.default_config = default_config or BacktestConfig()
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    # -- public API ---------------------------------------------------------------
    def specs(self) -> list[dict]:
        return [tool.as_openai_tool() for tool in self._tools.values()]

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict:
        if name not in self._tools:
            raise KeyError(f"unknown tool '{name}'; available: {self.tool_names()}")
        args = dict(arguments or {})
        try:
            return self._tools[name].func(**args)
        except TypeError as exc:
            return _err(f"bad arguments for '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must never raise to the agent
            return _err(f"{type(exc).__name__}: {exc}")

    # -- registration -------------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="list_strategies",
                description="List the built-in strategies with their parameter names, types and defaults.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                func=self._list_strategies,
            )
        )
        self.register(
            ToolSpec(
                name="describe_data",
                description="Describe the available data source: which symbols exist, how many bars, and the date range. Call this before referencing any symbol.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                func=self._describe_data,
            )
        )
        self.register(
            ToolSpec(
                name="get_bars",
                description="Return a summary of one symbol's daily bars (row count, date range, last close, recent closes). Raises ok=false for unknown symbols.",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol exactly as listed by describe_data"},
                        "tail": {"type": "integer", "description": "How many recent closes to include (default 10, max 60)"},
                    },
                    "required": ["symbol"],
                    "additionalProperties": False,
                },
                func=self._get_bars,
            )
        )
        self.register(
            ToolSpec(
                name="compute_indicator",
                description="Compute a technical indicator on one symbol and return its latest value plus summary statistics. Available indicators: "
                + ", ".join(sorted(INDICATORS)),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "indicator": {"type": "string", "enum": sorted(INDICATORS)},
                        "window": {"type": "integer", "description": "Lookback window (default 20)"},
                    },
                    "required": ["symbol", "indicator"],
                    "additionalProperties": False,
                },
                func=self._compute_indicator,
            )
        )
        self.register(
            ToolSpec(
                name="run_backtest",
                description="Backtest one strategy on one symbol and return performance metrics (net of fees and slippage). Use this instead of estimating numbers yourself.",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "strategy": {"type": "string", "enum": sorted(STRATEGIES)},
                        "params": {"type": "object", "description": "Strategy parameters, e.g. {\"fast\": 10, \"slow\": 30}"},
                        "fee_bps": {"type": "number", "description": "Commission in basis points (default 3)"},
                        "slippage_bps": {"type": "number", "description": "Slippage in basis points (default 2)"},
                    },
                    "required": ["symbol", "strategy"],
                    "additionalProperties": False,
                },
                func=self._run_backtest,
            )
        )
        self.register(
            ToolSpec(
                name="screen_universe",
                description="Rank all available symbols by the transparent weighted factor score and return the top N.",
                parameters={
                    "type": "object",
                    "properties": {
                        "top_n": {"type": "integer", "description": "How many ranked rows to return (default 5)"},
                        "as_of": {"type": "string", "description": "Optional ISO date; only data up to this date is used"},
                    },
                    "additionalProperties": False,
                },
                func=self._screen_universe,
            )
        )

    # -- tool implementations ------------------------------------------------------
    def _list_strategies(self) -> dict:
        out = []
        for name, cls in sorted(STRATEGIES.items()):
            instance = cls()
            out.append(
                {
                    "name": name,
                    "params": {k: _round(v) for k, v in instance.params.items()},
                    "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
                }
            )
        return _ok(strategies=out)

    def _describe_data(self) -> dict:
        info = self.source.describe()
        return _ok(**info)

    def _get_bars(self, symbol: str, tail: int = 10) -> dict:
        tail = max(1, min(int(tail), 60))
        try:
            df = self.source.bars(symbol)
        except KeyError as exc:
            return _err(str(exc))
        closes = df["close"].astype(float)
        recent = [
            {"date": str(idx.date()), "close": _round(val, 2)} for idx, val in closes.tail(tail).items()
        ]
        return _ok(
            symbol=symbol,
            rows=int(len(df)),
            first_date=str(df.index[0].date()),
            last_date=str(df.index[-1].date()),
            last_close=_round(closes.iloc[-1], 2),
            change_20d=_round(closes.iloc[-1] / closes.iloc[-21] - 1.0) if len(closes) > 21 else None,
            recent=recent,
        )

    def _compute_indicator(self, symbol: str, indicator: str, window: int = 20) -> dict:
        if indicator not in INDICATORS:
            return _err(f"unknown indicator '{indicator}'; available: {sorted(INDICATORS)}")
        window = int(window)
        if window < 2:
            return _err("window must be >= 2")
        try:
            df = self.source.bars(symbol)
        except KeyError as exc:
            return _err(str(exc))
        series = INDICATORS[indicator](df, window).dropna()
        if series.empty:
            return _err(f"not enough history to compute {indicator}({window}) for {symbol}")
        return _ok(
            symbol=symbol,
            indicator=indicator,
            window=window,
            last_value=_round(series.iloc[-1]),
            mean=_round(series.mean()),
            min=_round(series.min()),
            max=_round(series.max()),
            observations=int(len(series)),
        )

    def _run_backtest(
        self,
        symbol: str,
        strategy: str,
        params: Mapping[str, Any] | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> dict:
        if strategy not in STRATEGIES:
            return _err(f"unknown strategy '{strategy}'; available: {sorted(STRATEGIES)}")
        try:
            df = self.source.bars(symbol)
        except KeyError as exc:
            return _err(str(exc))
        try:
            strat = build_strategy(strategy, **dict(params or {}))
        except (TypeError, ValueError, KeyError) as exc:
            return _err(f"invalid strategy parameters: {exc}")

        config = BacktestConfig(
            fee_bps=self.default_config.fee_bps if fee_bps is None else float(fee_bps),
            slippage_bps=self.default_config.slippage_bps if slippage_bps is None else float(slippage_bps),
            initial_cash=self.default_config.initial_cash,
        )
        result = run_backtest(df, strat.positions(df), config=config, name=strategy)
        metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
        keep = (
            "total_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
            "exposure",
            "trades",
            "win_rate_trade",
            "avg_trade",
        )
        payload = {k: _round(metrics.get(k)) for k in keep if k in metrics}
        return _ok(
            symbol=symbol,
            strategy=strategy,
            params={k: _round(v) for k, v in strat.params.items()},
            bars=int(len(df)),
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
            metrics=payload,
        )

    def _screen_universe(self, top_n: int = 5, as_of: str | None = None) -> dict:
        top_n = max(1, int(top_n))
        universe = {sym: self.source.bars(sym) for sym in self.source.symbols()}
        try:
            table = rank_universe(universe, as_of=as_of, config=ScreenConfig(top_n=top_n))
        except ValueError as exc:
            return _err(str(exc))
        rows = []
        for _, row in table.head(top_n).iterrows():
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "symbol": row["symbol"],
                    "score": _round(row["score"], 3),
                    "mom_20": _round(row["mom_20"]),
                    "mom_60": _round(row["mom_60"]),
                    "vol_20": _round(row["vol_20"]),
                    "rsi_14": _round(row["rsi_14"], 2),
                }
            )
        return _ok(
            as_of=str(table["date"].iloc[0].date()) if len(table) else None,
            universe_size=int(len(table)),
            top=rows,
        )


def default_registry(data_dir: str | Path | None = None) -> ToolRegistry:
    """Build a registry backed by a CSV directory when provided, else synthetic data."""
    if data_dir is not None:
        return ToolRegistry(source=CsvDataSource(data_dir))
    return ToolRegistry(source=SyntheticDataSource())


def dumps(payload: Mapping[str, Any]) -> str:
    """Compact, stable JSON used when feeding tool results back to a model."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
