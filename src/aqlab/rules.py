"""Scoring rules — an open, parameterised rule engine (v0.3).

Every rule here is an **original, generic implementation** of a well-known
technical pattern, exposed with neutral names and fully configurable thresholds.
That is deliberate: the *idea* of a pullback entry, a long lower shadow, or a
volume-price surge is generic; the specific numbers belong to whoever configures
them. Put your own thresholds in the config (or via the CLI) and the rule is yours.

Concepts implemented
--------------------
* :class:`TieredPullback`      — 回踩均线买点，可按回踩深度分档（T1/T2/T3，可用于 B1/B2/B3 这类分档）
* :class:`NeedleBelowMA`       — 单针下探均线后收回（长下影 + 收盘站回均线上方），可换 20/30 等均线
* :class:`VolumePriceSurge`    — 量价齐升，可设确认天数（1 天 = V1，3 天 = V3 这类版本）
* :class:`ActivityValueGate`   — 活跃市值开关：全市场活跃度（收盘价 × 成交量的加总）双均线 + 滞回开关

A rule maps a symbol's OHLCV frame to a score series in ``[0, 1]`` (NaN → 0).
A gate maps a whole universe to a market on/off series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from aqlab.indicators import sma

__all__ = [
    "DEFAULT_RULE_BINDINGS",
    "RULES",
    "ActivityValueGate",
    "NeedleBelowMA",
    "Rule",
    "RuleBinding",
    "TieredPullback",
    "VolumePriceSurge",
    "build_rule",
]


class Rule(Protocol):
    """A scoring rule: price history in, score in [0, 1] out."""

    name: str
    params: dict[str, Any]

    def score(self, df: pd.DataFrame) -> pd.Series: ...


def _clean(score: pd.Series, index: pd.Index) -> pd.Series:
    out = score.reindex(index).astype(float).replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).clip(0.0, 1.0)


@dataclass
class RuleBinding:
    """A rule plus its weight in the composite score."""

    rule: Rule
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("rule weight must be non-negative")


# --------------------------------------------------------------------------------------
# Trend / pullback rules
# --------------------------------------------------------------------------------------
class TieredPullback:
    """回踩均线买点，按"下影扎入均线的深度"分档打分。

    Parameters
    ----------
    ma_window:
        The moving average being defended (e.g. 20).
    trend_window:
        Longer MA used as a trend filter (only buy pullbacks in an uptrend).
    tiers:
        ``(low, high)`` depth bands as a fraction of the MA. Depth is measured as
        ``(ma - low) / ma`` on the current bar. Band 1 is scored highest.
    tier_scores:
        Score assigned to each band, in the same order as ``tiers``.
    """

    name = "tiered_pullback"

    def __init__(
        self,
        ma_window: int = 20,
        trend_window: int = 60,
        tiers: Sequence[tuple[float, float]] = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10)),
        tier_scores: Sequence[float] = (1.0, 0.7, 0.4),
        require_close_above_ma: bool = True,
    ) -> None:
        if len(tiers) != len(tier_scores):
            raise ValueError("tiers and tier_scores must have the same length")
        if ma_window < 2 or trend_window < 2:
            raise ValueError("windows must be >= 2")
        self.params: dict[str, Any] = {
            "ma_window": ma_window,
            "trend_window": trend_window,
            "tiers": [tuple(t) for t in tiers],
            "tier_scores": list(tier_scores),
            "require_close_above_ma": require_close_above_ma,
        }

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        low = df["low"].astype(float)
        ma = sma(close, self.params["ma_window"])
        trend = sma(close, self.params["trend_window"])

        depth = (ma - low) / ma
        base = (close > trend) & ma.notna() & trend.notna()
        if self.params["require_close_above_ma"]:
            base = base & (close >= ma * 0.995)

        out = pd.Series(0.0, index=df.index)
        for (lo, hi), value in zip(self.params["tiers"], self.params["tier_scores"], strict=False):
            hit = base & (depth > lo) & (depth <= hi)
            out = out.where(~hit, value)
        return _clean(out, df.index)


class NeedleBelowMA:
    """单针下探均线后收回：长下影 + 收盘站回均线上方。

    Parameters
    ----------
    ma_window:
        MA being pierced (20 for 单针下 20, 30 for 单针下 30, …).
    min_shadow_ratio:
        Minimum ``(close - low) / close`` for the bar to count as a needle.
    max_close_below_ma:
        Tolerance for allowing the close slightly below the MA (0 = must close above).
    full_score_ratio:
        Shadow ratio that receives the maximum score.
    """

    name = "needle_below_ma"

    def __init__(
        self,
        ma_window: int = 20,
        min_shadow_ratio: float = 0.02,
        max_close_below_ma: float = 0.0,
        full_score_ratio: float = 0.05,
    ) -> None:
        if ma_window < 2:
            raise ValueError("ma_window must be >= 2")
        if not 0 < min_shadow_ratio <= full_score_ratio:
            raise ValueError("require 0 < min_shadow_ratio <= full_score_ratio")
        self.params: dict[str, Any] = {
            "ma_window": ma_window,
            "min_shadow_ratio": min_shadow_ratio,
            "max_close_below_ma": max_close_below_ma,
            "full_score_ratio": full_score_ratio,
        }

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        low = df["low"].astype(float)
        ma = sma(close, self.params["ma_window"])

        shadow = (close - low) / close
        pierced = low < ma
        recovered = close >= ma * (1.0 - self.params["max_close_below_ma"])
        hit = pierced & recovered & ma.notna() & (shadow >= self.params["min_shadow_ratio"])

        span = self.params["full_score_ratio"] - self.params["min_shadow_ratio"]
        scaled = (shadow - self.params["min_shadow_ratio"]) / span if span > 0 else pd.Series(1.0, index=shadow.index)
        out = scaled.clip(0.0, 1.0).where(hit, 0.0)
        return _clean(out, df.index)


class VolumePriceSurge:
    """量价齐升：当日涨幅达标 + 成交量放大，可选连续确认天数（V1/V2/V3 版本）。

    Parameters
    ----------
    min_price_change:
        Minimum same-day close-to-close return (e.g. 0.03 = +3%).
    volume_multiple:
        Required ``volume / MA(volume, volume_window)`` (e.g. 1.5 = 1.5x average).
    volume_window:
        Lookback for the average volume (default 20).
    confirm_days:
        1 = single-day signal (V1); 3 = signal must persist on 3 consecutive bars (V3).
        ``0`` disables confirmation.
    """

    name = "volume_price_surge"

    def __init__(
        self,
        min_price_change: float = 0.03,
        volume_multiple: float = 1.5,
        volume_window: int = 20,
        confirm_days: int = 1,
    ) -> None:
        if volume_window < 2:
            raise ValueError("volume_window must be >= 2")
        if volume_multiple <= 0 or min_price_change <= 0:
            raise ValueError("min_price_change and volume_multiple must be positive")
        if confirm_days < 0:
            raise ValueError("confirm_days must be >= 0")
        self.params: dict[str, Any] = {
            "min_price_change": min_price_change,
            "volume_multiple": volume_multiple,
            "volume_window": volume_window,
            "confirm_days": confirm_days,
        }

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        ret = close.pct_change()
        vol_ma = sma(volume, self.params["volume_window"])
        vol_ratio = volume / vol_ma.replace(0.0, np.nan)

        price_ok = ret >= self.params["min_price_change"]
        volume_ok = vol_ratio >= self.params["volume_multiple"]
        hit = price_ok & volume_ok

        confirm = int(self.params["confirm_days"])
        if confirm > 1:
            streak = hit.rolling(confirm, min_periods=confirm).sum() >= confirm
            hit = hit & streak

        strength = 0.5 * (ret / (2 * self.params["min_price_change"])).clip(0, 1) + 0.5 * (
            vol_ratio / (2 * self.params["volume_multiple"])
        ).clip(0, 1)
        out = strength.where(hit, 0.0)
        return _clean(out, df.index)


# --------------------------------------------------------------------------------------
# Market-level gate (活跃市值开关)
# --------------------------------------------------------------------------------------
@dataclass
class ActivityValueGate:
    """活跃市值开关：全市场活跃度指数的双均线 + 滞回开关。

    活跃度代理 = ``Σ(close × volume)``（等权加总，避免依赖流通股本数据）。
    当快线高于慢线 ``on_threshold`` 时打开；低于 ``off_threshold`` 时关闭；
    两者之间保持上一个状态（这就是"开关规则"的滞回设计）。

    Parameters
    ----------
    fast_window, slow_window:
        Moving averages on the activity index.
    on_threshold, off_threshold:
        e.g. ``on=0.02`` (快线高于慢线 2% 才开) / ``off=-0.01`` (低 1% 才关)。
    """

    fast_window: int = 5
    slow_window: int = 20
    on_threshold: float = 0.02
    off_threshold: float = -0.01
    name: str = "activity_value_gate"
    params: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.fast_window < 2 or self.slow_window <= self.fast_window:
            raise ValueError("require 2 <= fast_window < slow_window")
        if not self.off_threshold < self.on_threshold:
            raise ValueError("off_threshold must be smaller than on_threshold")
        self.params: dict[str, Any] = {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "on_threshold": self.on_threshold,
            "off_threshold": self.off_threshold,
        }

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def activity_series(universe: Mapping[str, pd.DataFrame]) -> pd.Series:
        if not universe:
            raise ValueError("universe is empty")
        frames = []
        for df in universe.values():
            value = df["close"].astype(float) * df["volume"].astype(float)
            frames.append(value)
        total = pd.concat(frames, axis=1).sum(axis=1, min_count=1)
        return total.dropna().sort_index()

    def gate_series(self, universe: Mapping[str, pd.DataFrame]) -> pd.Series:
        activity = self.activity_series(universe)
        fast = sma(activity, self.fast_window)
        slow = sma(activity, self.slow_window)
        spread = (fast - slow) / slow

        state = 0
        out = []
        for value in spread.to_numpy():
            if value != value:  # NaN while the slow MA warms up
                out.append(state)
                continue
            if value >= self.on_threshold:
                state = 1
            elif value <= self.off_threshold:
                state = 0
            out.append(state)
        return pd.Series(out, index=spread.index, dtype=int)

    def state_at(self, universe: Mapping[str, pd.DataFrame], as_of: str | pd.Timestamp | None = None) -> int:
        series = self.gate_series(universe)
        if series.empty:
            return 0
        if as_of is not None:
            series = series.loc[:as_of]
        return int(series.iloc[-1]) if len(series) else 0


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
RULES: dict[str, Callable[..., object]] = {
    TieredPullback.name: TieredPullback,
    NeedleBelowMA.name: NeedleBelowMA,
    VolumePriceSurge.name: VolumePriceSurge,
}


def build_rule(name: str, **params: object):
    """Build a rule from either the generic or the personal rule registry."""
    if name in RULES:
        return RULES[name](**params)
    from aqlab.rules_zgnb import PERSONAL_RULES

    if name in PERSONAL_RULES:
        return PERSONAL_RULES[name](**params)
    raise KeyError(f"unknown rule '{name}'; available: {sorted(RULES) + sorted(PERSONAL_RULES)}")


DEFAULT_RULE_BINDINGS: list[tuple[str, dict, float]] = [
    ("tiered_pullback", {"ma_window": 20, "trend_window": 60}, 0.4),
    ("needle_below_ma", {"ma_window": 20}, 0.3),
    ("volume_price_surge", {"min_price_change": 0.03, "volume_multiple": 1.5, "confirm_days": 3}, 0.3),
]
