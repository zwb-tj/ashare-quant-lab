"""Strategy library.

A strategy only expresses *intent*: it maps price history to a target position
series in ``[0, 1]`` (long-only) or ``[-1, 1]`` (if shorting is enabled later).
Execution realism — the one-bar delay, fees and slippage — lives entirely in
:mod:`aqlab.backtest`, so a strategy can never accidentally peek at the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import pandas as pd

from aqlab.indicators import pct_change_n, rolling_zscore, sma

__all__ = [
    "STRATEGIES",
    "BuyAndHoldStrategy",
    "MACrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "Strategy",
    "build_strategy",
]


class Strategy(ABC):
    """Base class: implement :meth:`positions`."""

    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params
        self.validate()

    def validate(self) -> None:  # noqa: B027 - 可选钩子：子类不实现也应能实例化
        """Override to validate parameters early (fail fast, with a clear message)."""

    @abstractmethod
    def positions(self, df: pd.DataFrame) -> pd.Series:
        """Return a target position per bar, indexed like ``df``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{type(self).__name__}({args})"

    # -- helpers -----------------------------------------------------------------
    def _clean(self, pos: pd.Series, index: pd.Index) -> pd.Series:
        out = pos.reindex(index).astype(float)
        return out.fillna(0.0).clip(lower=0.0, upper=1.0)


class MomentumStrategy(Strategy):
    """Long while medium-term momentum is positive *and* price is above its trend SMA.

    Parameters
    ----------
    lookback:
        Bars used for the momentum measurement.
    trend_window:
        Window of the trend filter SMA.
    vol_window, target_vol:
        Optional volatility scaling: position size = min(1, target_vol / realized_vol).
    """

    name = "momentum"

    def __init__(self, lookback: int = 60, trend_window: int = 120, vol_window: int = 20, target_vol: float = 0.20) -> None:
        super().__init__(
            lookback=lookback, trend_window=trend_window, vol_window=vol_window, target_vol=target_vol
        )

    def validate(self) -> None:
        if self.params["lookback"] < 1 or self.params["trend_window"] < 1:
            raise ValueError("lookback and trend_window must be >= 1")
        if not 0 < self.params["target_vol"] <= 2:
            raise ValueError("target_vol must be in (0, 2]")

    def positions(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        mom = pct_change_n(close, self.params["lookback"])
        trend = sma(close, self.params["trend_window"])
        signal = ((mom > 0) & (close > trend)).astype(float)

        vol = close.pct_change().rolling(self.params["vol_window"], min_periods=self.params["vol_window"]).std()
        ann_vol = vol * (252 ** 0.5)
        scale = (self.params["target_vol"] / ann_vol.replace(0.0, pd.NA)).astype(float).clip(upper=1.0)
        return self._clean(signal * scale.fillna(1.0), df.index)


class MACrossStrategy(Strategy):
    """Classic dual moving-average crossover (long / flat)."""

    name = "ma_cross"

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        super().__init__(fast=fast, slow=slow)

    def validate(self) -> None:
        if self.params["fast"] >= self.params["slow"]:
            raise ValueError("fast must be smaller than slow")
        if self.params["fast"] < 1:
            raise ValueError("fast must be >= 1")

    def positions(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        fast = sma(close, self.params["fast"])
        slow = sma(close, self.params["slow"])
        return self._clean((fast > slow).astype(float), df.index)


class MeanReversionStrategy(Strategy):
    """Long when price is oversold vs its rolling mean; exit on reversion.

    The state machine holds the position until the exit condition triggers
    (forward-fill), which is what a discretionary trader would actually do.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 20, z_entry: float = -1.5, z_exit: float = 0.0, max_holding: int = 20) -> None:
        super().__init__(window=window, z_entry=z_entry, z_exit=z_exit, max_holding=max_holding)

    def validate(self) -> None:
        if self.params["window"] < 2:
            raise ValueError("window must be >= 2")
        if not self.params["z_entry"] < self.params["z_exit"]:
            raise ValueError("z_entry must be smaller than z_exit")
        if self.params["max_holding"] < 1:
            raise ValueError("max_holding must be >= 1")

    def positions(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        z = rolling_zscore(close, self.params["window"])
        raw = pd.Series(float("nan"), index=df.index)
        raw[z <= self.params["z_entry"]] = 1.0
        raw[z >= self.params["z_exit"]] = 0.0
        held = raw.ffill().fillna(0.0)

        # time stop: force flat after max_holding consecutive bars in a position
        out, streak = [], 0
        for value in held.to_numpy():
            if value > 0:
                streak += 1
                if streak > self.params["max_holding"]:
                    value, streak = 0.0, 0
            else:
                streak = 0
            out.append(value)
        return self._clean(pd.Series(out, index=df.index), df.index)


class BuyAndHoldStrategy(Strategy):
    """Baseline: always fully invested. Useful as the benchmark in reports."""

    name = "buy_and_hold"

    def positions(self, df: pd.DataFrame) -> pd.Series:
        return self._clean(pd.Series(1.0, index=df.index), df.index)


STRATEGIES: dict[str, Callable[..., Strategy]] = {
    MomentumStrategy.name: MomentumStrategy,
    MACrossStrategy.name: MACrossStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BuyAndHoldStrategy.name: BuyAndHoldStrategy,
}


def build_strategy(name: str, **params: Any) -> Strategy:
    """Factory used by the CLI and the screening module."""
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy '{name}'; available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
