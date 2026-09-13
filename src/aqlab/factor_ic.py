"""因子有效性研究（v0.17）：IC / IR / 分位收益。

回答一个因子研究里最基础的问题：**这个因子和未来的收益到底有没有关系**，而不是只看总分好不好看。

口径（写清楚，避免自欺）：

* 因子值只用 ``as_of`` **当天及之前**的数据算（复用 :mod:`aqlab.screen` 的快照口径）；
* 前瞻收益 = ``close[t+h] / close[t] - 1``，只用于评估、不参与选股；
* 相关性用 **Spearman 秩相关**（不假设线性、对极值稳健），每个截面单独算，再对时间求统计量；
* **IC_IR = 平均 IC / IC 标准差**，t 值 = ``IC_IR × sqrt(截面数)``——不报告"年化 IC"这类容易被夸大的口径；
* 每个截面要求至少 ``min_symbols`` 个标的，否则该截面作废（不拿三五个样本凑相关性）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.indicators import pct_change_n, realized_vol, rsi, sma

__all__ = [
    "ICConfig",
    "factor_ic_panel",
    "factor_ic_panel_multi",
    "precompute_factors",
    "quantile_returns",
    "spearman_ic",
    "summarize_ic",
    "summarize_ic_by_horizon",
]

DEFAULT_FACTORS = ("mom_20", "mom_60", "trend_gap", "vol_20", "rsi_14")


@dataclass
class ICConfig:
    forward_days: int = 20          # 前瞻收益天数
    step_days: int = 5             # 每隔多少个交易日取一个截面（降低自相关）
    min_history: int = 130
    min_symbols: int = 8
    factors: tuple[str, ...] = DEFAULT_FACTORS
    quantiles: int = 5
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if self.forward_days < 1:
            raise ValueError("forward_days must be >= 1")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.min_symbols < 3:
            raise ValueError("min_symbols must be >= 3")
        if self.quantiles < 2:
            raise ValueError("quantiles must be >= 2")


def precompute_factors(
    universe: Mapping[str, pd.DataFrame],
    config: ICConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """对每只标的**一次算完全序列因子**，后续只做截面切片。

    这样做的依据是：这里用到的指标全部是**回看型**（``sma`` / ``pct_change_n`` / ``realized_vol`` / ``rsi``），
    某一天的取值只依赖当天及之前的数据，因此"整段算完再取当天"与"截断到当天再算"结果相同
    （``tests/test_factor_ic.py`` 里有对照测试守着这条前提）。
    好处是复杂度从「截面数 × 标的数」降到「标的数」，全市场跑得动。
    """
    config = config or ICConfig()
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in universe.items():
        if frame is None or len(frame) < config.min_history:
            continue
        hist = frame.copy()
        hist.index = pd.to_datetime(hist.index)
        close = hist["close"].astype(float)
        ma60 = sma(close, 60)
        table = pd.DataFrame(
            {
                "close": close,
                "mom_20": pct_change_n(close, 20),
                "mom_60": pct_change_n(close, 60),
                "trend_gap": close / ma60 - 1.0,
                "vol_20": realized_vol(close, 20),
                "rsi_14": rsi(close, 14),
            },
            index=hist.index,
        )
        out[symbol] = table
    if not out:
        raise ValueError("no symbol has enough history for factor analysis")
    return out


def _cross_section(precomputed: Mapping[str, pd.DataFrame], as_of: pd.Timestamp, min_history: int) -> pd.DataFrame:
    """把某个截面日所有标的的因子值拼成一张表（只保留历史足够的标的）。"""
    rows: list[dict] = []
    for symbol, table in precomputed.items():
        if as_of not in table.index:
            continue
        position = table.index.get_loc(as_of)
        if position + 1 < min_history:
            continue
        row = table.iloc[position].to_dict()
        row["symbol"] = symbol
        row["date"] = as_of
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_ic(factor: pd.Series, forward: pd.Series) -> float:
    """两个序列的 Spearman 秩相关（只保留两边都有效的样本）。"""
    frame = pd.DataFrame({"factor": factor, "forward": forward}).dropna()
    if len(frame) < 3:
        return np.nan
    if frame["factor"].nunique() < 2 or frame["forward"].nunique() < 2:
        return np.nan       # 全是同一个值时相关系数没有定义，返回 NaN 而不是 0
    return float(frame["factor"].rank().corr(frame["forward"].rank()))


def _forward_return(frame: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> float:
    """``as_of`` 之后第 horizon 个交易日的收益；数据不够则 NaN。"""
    index = frame.index
    if as_of not in index:
        return np.nan
    position = index.get_loc(as_of)
    target = position + horizon
    if target >= len(frame):
        return np.nan
    entry = float(frame["close"].iloc[position])
    exit_price = float(frame["close"].iloc[target])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price):
        return np.nan
    return exit_price / entry - 1.0


def factor_ic_panel(universe: Mapping[str, pd.DataFrame], config: ICConfig | None = None) -> pd.DataFrame:
    """逐截面算 IC，返回长表：date / factor / ic / symbols。"""
    config = config or ICConfig()
    precomputed = precompute_factors(universe, config)

    all_dates = sorted({date for table in precomputed.values() for date in table.index})
    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None
    usable = [date for date in all_dates if (start is None or date >= start) and (end is None or date <= end)]
    usable = usable[:: config.step_days]

    rows: list[dict] = []
    for as_of in usable:
        table = _cross_section(precomputed, as_of, config.min_history)
        if table.empty or len(table) < config.min_symbols:
            continue
        forwards = []
        for symbol in table["symbol"]:
            series = precomputed[symbol]["close"]
            position = series.index.get_loc(as_of)
            target = position + config.forward_days
            if target >= len(series):
                forwards.append(np.nan)
                continue
            entry = float(series.iloc[position])
            exit_price = float(series.iloc[target])
            forwards.append(exit_price / entry - 1.0 if entry > 0 and np.isfinite(exit_price) else np.nan)
        table = table.assign(forward=forwards)
        for factor in config.factors:
            if factor not in table.columns:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(as_of),
                    "factor": factor,
                    "ic": spearman_ic(table[factor], table["forward"]),
                    "symbols": int(table[factor].notna().sum()),
                }
            )
    # 丢掉算不出 IC 的截面（前瞻窗口越过数据末尾，或该因子在该截面上取值全同）
    return pd.DataFrame(rows).dropna(subset=["ic"]).reset_index(drop=True)


def summarize_ic(panel: pd.DataFrame, config: ICConfig | None = None) -> pd.DataFrame:
    """把逐截面 IC 汇总成：均值、标准差、IC_IR、t 值（含重叠窗口修正）、正 IC 占比。

    相邻截面的前瞻窗口会重叠（例如前瞻 20 日、步长 5 日 → 每 4 个截面共享同一段未来行情），
    此时 IC 序列是自相关的，直接按截面数算 t 值会**高估显著性**。这里给出保守修正：
    ``t_adj = t / sqrt(ceil(forward_days / step_days))``，即把有效样本数按重叠倍数打折。
    """
    config = config or ICConfig()
    overlap = max(1, int(np.ceil(config.forward_days / config.step_days)))
    if panel is None or panel.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for factor, group in panel.groupby("factor"):
        values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        ir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
        t_stat = ir * np.sqrt(len(values)) if np.isfinite(ir) else np.nan
        rows.append(
            {
                "factor": factor,
                "periods": len(values),
                "ic_mean": mean,
                "ic_std": std,
                "ic_ir": ir,
                "t_stat": t_stat,
                "t_stat_adj": t_stat / np.sqrt(overlap) if np.isfinite(t_stat) else np.nan,
                "positive_rate": float((values > 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("ic_mean", ascending=False).reset_index(drop=True) if not table.empty else table


def factor_ic_panel_multi(
    universe: Mapping[str, pd.DataFrame],
    horizons: Sequence[int],
    config: ICConfig | None = None,
) -> pd.DataFrame:
    """多持有期 IC：**一次遍历**算出所有期限，避免对每个期限重跑一遍截面。

    多出来的 ``horizon`` 列让"IC 随持有期怎么变"这个问题变得可查——例如短期限是动量、
    长期限是反转，这种期限结构不看就会误判因子。
    """
    config = config or ICConfig()
    horizons = tuple(sorted({int(h) for h in horizons if int(h) >= 1}))
    if not horizons:
        raise ValueError("horizons must contain at least one positive integer")
    precomputed = precompute_factors(universe, config)

    all_dates = sorted({date for table in precomputed.values() for date in table.index})
    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None
    usable = [date for date in all_dates if (start is None or date >= start) and (end is None or date <= end)]
    usable = usable[:: config.step_days]

    rows: list[dict] = []
    for as_of in usable:
        table = _cross_section(precomputed, as_of, config.min_history)
        if table.empty or len(table) < config.min_symbols:
            continue
        forwards: dict[int, list[float]] = {horizon: [] for horizon in horizons}
        for symbol in table["symbol"]:
            close = precomputed[symbol]["close"]
            position = close.index.get_loc(as_of)
            entry = float(close.iloc[position])
            for horizon in horizons:
                target = position + horizon
                if target >= len(close) or not np.isfinite(entry) or entry <= 0:
                    forwards[horizon].append(np.nan)
                    continue
                exit_price = float(close.iloc[target])
                forwards[horizon].append(exit_price / entry - 1.0 if np.isfinite(exit_price) else np.nan)
        for horizon in horizons:
            frame = table.assign(forward=forwards[horizon])
            for factor in config.factors:
                if factor not in frame.columns:
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(as_of),
                        "factor": factor,
                        "horizon": horizon,
                        "ic": spearman_ic(frame[factor], frame["forward"]),
                        "symbols": int(frame[factor].notna().sum()),
                    }
                )
    return pd.DataFrame(rows).dropna(subset=["ic"]).reset_index(drop=True)


def summarize_ic_by_horizon(panel: pd.DataFrame, config: ICConfig | None = None) -> pd.DataFrame:
    """多期限面板的汇总：每个 (因子, 期限) 的 IC 均值、IR、修正 t 值。

    重叠修正按各自的期限算：``overlap = ceil(horizon / step_days)``。
    """
    if panel is None or panel.empty or "horizon" not in panel.columns:
        return pd.DataFrame()
    step = (config or ICConfig()).step_days
    rows: list[dict] = []
    for (factor, horizon), group in panel.groupby(["factor", "horizon"]):
        values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        ir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
        t_stat = ir * np.sqrt(len(values)) if np.isfinite(ir) else np.nan
        overlap = max(1, int(np.ceil(int(horizon) / step)))
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "periods": len(values),
                "ic_mean": mean,
                "ic_ir": ir,
                "t_stat_adj": t_stat / np.sqrt(overlap) if np.isfinite(t_stat) else np.nan,
                "positive_rate": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["factor", "horizon"]).reset_index(drop=True)


def quantile_returns(universe: Mapping[str, pd.DataFrame], config: ICConfig | None = None) -> pd.DataFrame:
    """按每个因子分组（分位）统计未来收益：返回 factor / group / mean_forward / periods。"""
    config = config or ICConfig()
    precomputed = precompute_factors(universe, config)
    all_dates = sorted({date for table in precomputed.values() for date in table.index})
    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None
    usable = [date for date in all_dates if (start is None or date >= start) and (end is None or date <= end)]
    usable = usable[:: config.step_days]

    buckets: dict[str, dict[int, list[float]]] = {
        factor: {group: [] for group in range(1, config.quantiles + 1)} for factor in config.factors
    }
    for as_of in usable:
        table = _cross_section(precomputed, as_of, config.min_history)
        if table.empty or len(table) < config.min_symbols:
            continue
        forwards = []
        for symbol in table["symbol"]:
            series = precomputed[symbol]["close"]
            position = series.index.get_loc(as_of)
            target = position + config.forward_days
            if target >= len(series):
                forwards.append(np.nan)
                continue
            entry = float(series.iloc[position])
            exit_price = float(series.iloc[target])
            forwards.append(exit_price / entry - 1.0 if entry > 0 and np.isfinite(exit_price) else np.nan)
        table = table.assign(forward=forwards).dropna(subset=["forward"])
        if len(table) < config.quantiles:
            continue
        for factor in config.factors:
            if factor not in table.columns:
                continue
            subset = table.dropna(subset=[factor])
            if len(subset) < config.quantiles:
                continue
            try:
                labels = pd.qcut(subset[factor], config.quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            for position, group in enumerate(range(config.quantiles)):
                values = subset["forward"][labels == position]
                if len(values):
                    buckets[factor][group + 1].append(float(values.mean()))

    rows: list[dict] = []
    for factor, groups in buckets.items():
        for group, values in groups.items():
            rows.append(
                {
                    "factor": factor,
                    "group": group,
                    "mean_forward": float(np.mean(values)) if values else np.nan,
                    "periods": len(values),
                }
            )
    return pd.DataFrame(rows)
