"""评估"开盘量比确认"：用真实分钟数据看它是否提升了 B1 的信号质量。

思路：对每个 B1 信号，取**次日开盘前 N 分钟**的量比，按量比阈值分成
「买入（量比达标）」与「观望（量比不足）」，再比较两组之后的 1/3/5 日收益与胜率。

* 决策日与收益都来自信号日**之后**，无未来函数；
* 没有次日分钟数据的信号单独计入「无法判断」，不参与两组比较，也不丢进任一组冒充样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from aqlab.intraday import IntradayConfig, confirm_signals

__all__ = ["ConfirmEvalConfig", "attach_benchmark", "evaluate_confirmation", "summarize_confirmation"]


@dataclass
class ConfirmEvalConfig:
    window_minutes: int = 7
    baseline_days: int = 5
    min_ratio: float = 1.0
    horizons: tuple[int, ...] = (1, 3, 5)
    entry: str = "window_close"     # window_close | decision_close | signal_close

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.entry not in ("window_close", "decision_close", "signal_close"):
            raise ValueError("entry must be 'window_close', 'decision_close' or 'signal_close'")

    @property
    def intraday(self) -> IntradayConfig:
        return IntradayConfig(window_minutes=self.window_minutes, baseline_days=self.baseline_days, min_ratio=self.min_ratio)


def _entry_returns(
    daily: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    horizons: Sequence[int],
) -> dict[str, float]:
    """从 ``entry_date``（含）之后第 h 个交易日**收盘**卖出的收益。"""
    out: dict[str, float] = {}
    if not np.isfinite(entry_price) or entry_price <= 0 or entry_date not in daily.index:
        return {f"fwd_{h}": np.nan for h in horizons}
    position = daily.index.get_loc(entry_date)
    for horizon in horizons:
        exit_position = position + horizon - 1
        if exit_position < len(daily):
            exit_price = float(daily["close"].iloc[exit_position])
            out[f"fwd_{horizon}"] = exit_price / entry_price - 1.0 if np.isfinite(exit_price) else np.nan
        else:
            out[f"fwd_{horizon}"] = np.nan
    return out


def evaluate_confirmation(
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    signal: pd.Series,
    config: ConfirmEvalConfig | None = None,
    symbol: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    """返回 ``(逐信号明细, 量比序列)``。

    明细列：symbol, signal_date, decision_date, entry_date, entry_price, volume_ratio,
    decision, fwd_1..fwd_N。

    ``entry`` 决定收益从哪天算起（默认 ``window_close``：决策日开盘窗口收盘价买入）：
    信号日收盘 → 决策日窗口之间的涨跌**不参与**统计，因为那时还没法做出决策。
    """
    config = config or ConfirmEvalConfig()
    from aqlab.intraday import opening_volume_ratio, opening_window_price

    ratio = opening_volume_ratio(minute, config.intraday) if not minute.empty else pd.Series(dtype=float)
    decisions = confirm_signals(signal, ratio, config.intraday)
    if decisions.empty:
        return pd.DataFrame(), ratio

    frame = daily.copy()
    frame.index = pd.to_datetime(frame.index)
    window_price = opening_window_price(minute, config.intraday) if not minute.empty else pd.Series(dtype=float)
    if len(window_price):
        window_price.index = pd.to_datetime(window_price.index)

    details = decisions.copy()
    details["symbol"] = symbol
    entry_dates: list[Any] = []
    entry_prices: list[float] = []
    returns: dict[str, list[float]] = {f"fwd_{h}": [] for h in config.horizons}
    for _, row in details.iterrows():
        signal_date = pd.Timestamp(row["signal_date"]) if row["signal_date"] else None
        decision_date = pd.Timestamp(row["decision_date"]) if row["decision_date"] else None
        if decision_date is None or config.entry == "signal_close":
            entry_date = signal_date
            price = float(frame["close"].get(entry_date, np.nan)) if entry_date is not None and entry_date in frame.index else np.nan
        elif config.entry == "decision_close":
            entry_date = decision_date
            price = float(frame["close"].get(entry_date, np.nan)) if entry_date in frame.index else np.nan
        else:  # window_close：决策日开盘窗口收盘价（看到量比后的成交价）
            entry_date = decision_date
            price = float(window_price.get(decision_date, np.nan)) if len(window_price) else np.nan
        entry_dates.append(str(entry_date.date()) if entry_date is not None else None)
        entry_prices.append(price)
        for key, value in _entry_returns(frame, entry_date, price, config.horizons).items():
            returns[key].append(value)
    details["entry_date"] = entry_dates
    details["entry_price"] = [round(p, 3) if np.isfinite(p) else np.nan for p in entry_prices]
    for key, values in returns.items():
        details[key] = values
    return details.sort_values("signal_date").reset_index(drop=True), ratio


def attach_benchmark(details: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """按 ``entry_date`` 把基准接到量比明细上，并给出超额。"""
    if details.empty or benchmark.empty:
        return details.copy()
    merged = details.merge(benchmark, on="entry_date", how="left")
    for column in [c for c in merged.columns if c.startswith("fwd_")]:
        horizon = column.split("_", 1)[1]
        bench_column = f"bench_{horizon}"
        if bench_column in merged.columns:
            merged[f"excess_{horizon}"] = pd.to_numeric(merged[column], errors="coerce") - pd.to_numeric(
                merged[bench_column], errors="coerce"
            )
        else:
            merged[f"excess_{horizon}"] = np.nan
    return merged


def summarize_confirmation(details: pd.DataFrame, config: ConfirmEvalConfig | None = None) -> pd.DataFrame:
    """按决策分组统计：样本数、各持有期的均值与胜率；有基准时附超额（配对子集）。"""
    config = config or ConfirmEvalConfig()
    if details.empty:
        return pd.DataFrame()
    has_benchmark = any(f"excess_{h}" in details.columns for h in config.horizons)

    def fill(row: dict, group: pd.DataFrame) -> dict:
        for horizon in config.horizons:
            column = f"fwd_{horizon}"
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.dropna()
            row[f"n_{horizon}"] = len(valid)
            row[f"mean_{horizon}"] = float(valid.mean()) if len(valid) else np.nan
            row[f"win_{horizon}"] = float((valid > 0).mean()) if len(valid) else np.nan
            if has_benchmark:
                bench_column = f"bench_{horizon}"
                bench = (
                    pd.to_numeric(group[bench_column], errors="coerce")
                    if bench_column in group.columns
                    else pd.Series(np.nan, index=group.index)
                )
                paired = pd.DataFrame({"fwd": values, "bench": bench}).dropna()
                row[bench_column] = float(paired["bench"].mean()) if len(paired) else np.nan
                row[f"excess_{horizon}"] = float((paired["fwd"] - paired["bench"]).mean()) if len(paired) else np.nan
                row[f"excesswin_{horizon}"] = float(((paired["fwd"] - paired["bench"]) > 0).mean()) if len(paired) else np.nan
        return row

    rows: list[dict] = []
    for decision, group in details.groupby("decision"):
        rows.append(fill({"decision": decision, "signals": len(group)}, group))
    overall: dict = {"decision": "全部", "signals": len(details)}
    rows.append(fill(overall, details))
    order = {"买入": 0, "观望": 1, "无法判断": 2, "全部": 3}
    table = pd.DataFrame(rows)
    table["_order"] = table["decision"].map(order).fillna(9)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)

