"""评估"开盘量比确认"：用真实分钟数据看它是否提升了 B1 的信号质量。

思路：对每个 B1 信号，取**次日开盘前 N 分钟**的量比，按量比阈值分成
「买入（量比达标）」与「观望（量比不足）」，再比较两组之后的 1/3/5 日收益与胜率。

* 决策日与收益都来自信号日**之后**，无未来函数；
* 没有次日分钟数据的信号单独计入「无法判断」，不参与两组比较，也不丢进任一组冒充样本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aqlab.intraday import IntradayConfig, confirm_signals
from aqlab.study import forward_returns

__all__ = ["ConfirmEvalConfig", "evaluate_confirmation", "summarize_confirmation"]


@dataclass
class ConfirmEvalConfig:
    window_minutes: int = 7
    baseline_days: int = 5
    min_ratio: float = 1.0
    horizons: tuple[int, ...] = (1, 3, 5)

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")

    @property
    def intraday(self) -> IntradayConfig:
        return IntradayConfig(window_minutes=self.window_minutes, baseline_days=self.baseline_days, min_ratio=self.min_ratio)


def evaluate_confirmation(
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    signal: pd.Series,
    config: ConfirmEvalConfig | None = None,
    symbol: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    """返回 ``(逐信号明细, 量比序列)``。

    明细列：symbol, signal_date, decision_date, volume_ratio, decision, fwd_1..fwd_N。
    """
    config = config or ConfirmEvalConfig()
    from aqlab.intraday import opening_volume_ratio

    ratio = opening_volume_ratio(minute, config.intraday) if not minute.empty else pd.Series(dtype=float)
    decisions = confirm_signals(signal, ratio, config.intraday)
    if decisions.empty:
        return pd.DataFrame(), ratio

    fwd = forward_returns(daily, config.horizons)
    fwd.index = pd.to_datetime(fwd.index)
    details = decisions.copy()
    details["symbol"] = symbol
    for horizon in config.horizons:
        column = f"fwd_{horizon}"
        details[column] = [
            float(fwd[column].get(pd.Timestamp(row["signal_date"]), np.nan)) if row["signal_date"] else np.nan
            for _, row in details.iterrows()
        ]
    return details.sort_values("signal_date").reset_index(drop=True), ratio


def summarize_confirmation(details: pd.DataFrame, config: ConfirmEvalConfig | None = None) -> pd.DataFrame:
    """按决策分组统计：样本数、各持有期的均值与胜率。"""
    config = config or ConfirmEvalConfig()
    if details.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for decision, group in details.groupby("decision"):
        row: dict = {"decision": decision, "signals": int(len(group))}
        for horizon in config.horizons:
            column = f"fwd_{horizon}"
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"n_{horizon}"] = int(len(values))
            row[f"mean_{horizon}"] = float(values.mean()) if len(values) else np.nan
            row[f"win_{horizon}"] = float((values > 0).mean()) if len(values) else np.nan
        rows.append(row)

    overall: dict = {"decision": "全部", "signals": int(len(details))}
    for horizon in config.horizons:
        values = pd.to_numeric(details[f"fwd_{horizon}"], errors="coerce").dropna()
        overall[f"n_{horizon}"] = int(len(values))
        overall[f"mean_{horizon}"] = float(values.mean()) if len(values) else np.nan
        overall[f"win_{horizon}"] = float((values > 0).mean()) if len(values) else np.nan
    rows.append(overall)
    order = {"买入": 0, "观望": 1, "无法判断": 2, "全部": 3}
    table = pd.DataFrame(rows)
    table["_order"] = table["decision"].map(order).fillna(9)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
