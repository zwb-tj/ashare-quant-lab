"""Cross-sectional screening (选股).

The screener answers one question: *as of a given date, which symbols look
strongest on a transparent, weighted factor score?*

Honesty rules baked in:

* factors are computed from data **up to and including** ``as_of`` only;
* cross-sectional z-scores make the score comparable across dates;
* ties break deterministically by symbol, so the output is reproducible;
* the score is a *ranking device*, not a prediction — see the README disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from aqlab.indicators import pct_change_n, realized_vol, rsi, sma

__all__ = ["DEFAULT_WEIGHTS", "factor_table", "rank_universe"]

DEFAULT_WEIGHTS: dict[str, float] = {
    "mom_60": 0.35,
    "mom_20": 0.20,
    "trend_gap": 0.25,
    "vol_20": -0.20,
}


@dataclass
class ScreenConfig:
    """Screening parameters."""

    min_history: int = 130
    top_n: int = 5
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        if self.min_history < 60:
            raise ValueError("min_history must be >= 60 for stable factors")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = float(s.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def factor_table(universe: Mapping[str, pd.DataFrame], as_of: str | pd.Timestamp | None = None, min_history: int = 130) -> pd.DataFrame:
    """Compute the factor snapshot for every symbol as of ``as_of`` (inclusive)."""
    rows: list[dict] = []
    for symbol, df in universe.items():
        if df is None or df.empty:
            continue
        hist = df.loc[:as_of] if as_of is not None else df
        if len(hist) < min_history:
            continue
        close = hist["close"].astype(float)
        ma60 = sma(close, 60)
        rows.append(
            {
                "symbol": symbol,
                "date": hist.index[-1],
                "close": float(close.iloc[-1]),
                "mom_20": float(pct_change_n(close, 20).iloc[-1]),
                "mom_60": float(pct_change_n(close, 60).iloc[-1]),
                "trend_gap": float(close.iloc[-1] / ma60.iloc[-1] - 1.0) if ma60.iloc[-1] == ma60.iloc[-1] else np.nan,
                "vol_20": float(realized_vol(close, 20).iloc[-1]),
                "rsi_14": float(rsi(close, 14).iloc[-1]),
                "avg_volume_20": float(hist["volume"].astype(float).tail(20).mean()) if "volume" in hist else np.nan,
            }
        )
    if not rows:
        raise ValueError("no symbol has enough history for screening")
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def rank_universe(
    universe: Mapping[str, pd.DataFrame],
    as_of: str | pd.Timestamp | None = None,
    config: ScreenConfig | None = None,
) -> pd.DataFrame:
    """Return the universe ranked by weighted, cross-sectional z-score factors."""
    config = config or ScreenConfig()
    table = factor_table(universe, as_of=as_of, min_history=config.min_history)

    score = pd.Series(0.0, index=table.index)
    used: list[str] = []
    for factor, weight in config.weights.items():
        if factor not in table.columns:
            continue
        score = score + weight * _zscore(table[factor].fillna(table[factor].median()))
        used.append(factor)

    table["score"] = score
    table["factors_used"] = ",".join(used)
    table = table.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table
