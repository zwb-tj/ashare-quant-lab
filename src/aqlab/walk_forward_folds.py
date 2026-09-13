"""滚动多折（walk-forward）因子验证（v0.28）。

**为什么需要它**：上一节的"样本外存活"只基于**一次** 50/50 切分——单次切分的结论可能是
特定时段的运气。这一节把时间轴切成连续多折，每折都做一次"用过去选、用未来验"：

    fold 1: [ 训练 ][ 测试 ]
    fold 2: [   训练   ][ 测试 ]
    fold 3: [     训练     ][ 测试 ]
    ...

**折内做两件事**（顺序不能反）：
1. 用**训练段**的 IC 决定方向（做多高分位还是低分位），并判断是否达标（|t| > 门槛，默认 2）；
2. 在**测试段**用这个方向算 IC 与超额，看是否延续。

**为什么只算一次 IC**：IC 是"某个截面 + 某个持有期"的函数，与折的划分无关。因此每个
(因子, 持有期) 只需扫一遍时间轴得到 IC 序列，再按折的日期区间切片聚合即可——既省时间，
又保证各折口径完全一致（不会因为折边界的取样相位不同而口径漂移）。

**汇总口径**：`folds_selected` 是"训练段达标"的折数，`folds_survived` 是在这些折里
测试段**同向且达标**的折数。真正稳的因子应当在**多数折**里都存活，而不是只在一折里好看。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.alpha101 import build_panel, compute_alphas
from aqlab.factor_ic import spearman_ic

__all__ = ["FoldConfig", "ic_series", "run_walk_forward", "summarize_walk_forward"]


@dataclass
class FoldConfig:
    horizons: tuple[int, ...] = (1, 5, 20)
    step_days: int = 5
    min_history: int = 260
    min_symbols: int = 200
    n_folds: int = 5
    train_fraction: float = 0.6      # 每折内训练段占比（训练段从样本起点滚动到该折边界）
    t_threshold: float = 2.0         # 训练段/测试段的达标门槛
    factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.min_symbols < 3:
            raise ValueError("min_symbols must be >= 3")
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.t_threshold <= 0:
            raise ValueError("t_threshold must be > 0")


def _stats(values: Sequence[float], horizon: int, step_days: int) -> dict:
    """一段 IC 的均值 / IR / 重叠修正 t 值。"""
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return {"periods": 0, "ic_mean": np.nan, "t_stat": np.nan}
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    ir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
    overlap = max(1, int(np.ceil(horizon / step_days)))
    t_stat = ir * np.sqrt(len(series)) / np.sqrt(overlap) if np.isfinite(ir) else np.nan
    return {"periods": len(series), "ic_mean": mean, "t_stat": t_stat}


def ic_series(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int,
    dates: Sequence[pd.Timestamp],
    min_symbols: int,
) -> pd.Series:
    """因子相对未来 ``horizon`` 日收益的 IC 序列（索引为截面日期）。

    与 :func:`aqlab.factor_eval.ic_of_factor` 同一口径，但**保留日期索引**，
    这样才能按折区间切片——多折验证的关键就是"同一份 IC 序列、不同的切法"。
    """
    out: dict[pd.Timestamp, float] = {}
    for as_of in dates:
        if as_of not in factor.index:
            continue
        position = factor.index.get_loc(as_of)
        target = position + horizon
        if target >= len(factor):
            continue
        factor_row = factor.iloc[position]
        if factor_row.notna().sum() < min_symbols:
            continue
        entry = close.iloc[position]
        exit_price = close.iloc[target]
        forward = (exit_price / entry - 1.0).replace([np.inf, -np.inf], np.nan)
        value = spearman_ic(factor_row, forward)
        if value == value:
            out[pd.Timestamp(as_of)] = float(value)
    return pd.Series(out, dtype=float).sort_index()


def run_walk_forward(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    config: FoldConfig | None = None,
) -> dict:
    """多折 walk-forward：每折用训练段选方向与达标，再在测试段验证。"""
    config = config or FoldConfig()
    panel = build_panel(daily_by_symbol)
    values = compute_alphas(panel, names=list(config.factors) or None, min_history=config.min_history)
    if not values:
        raise ValueError("no factor could be computed (check min_history and data length)")

    dates = list(panel.dates)
    sampled = dates[:: config.step_days]

    # 折必须切在**因子真正有值**的时间段上：因子有 min_history 预热期，之前全是 NaN；
    # 若按全部交易日等分，靠前的折会因训练段落在预热期内而被整折跳过
    # （真实数据上 556 个交易日 + 预热 260 根，5 折只剩 2 折）。
    usable = [date for date in sampled if any(date in frame.index and bool(frame.loc[date].notna().any()) for frame in values.values())]
    if not usable:
        raise ValueError("no cross-section has any factor value (check min_history)")
    if len(usable) < config.n_folds * 2:
        raise ValueError(
            f"not enough usable cross-sections for {config.n_folds} folds: {len(usable)} "
            f"(a factor warm-up of {config.min_history} bars consumes part of the history)"
        )
    boundaries = np.linspace(0, len(usable), config.n_folds + 1).astype(int)

    folds: list[dict] = []
    for fold_index in range(config.n_folds):
        start, stop = int(boundaries[fold_index]), int(boundaries[fold_index + 1])
        test_dates = usable[start:stop]
        if not test_dates:
            continue
        # 训练段 = 测试段起点之前的**可用**截面里按 train_fraction 取尾部一段（扩张窗口）；
        # 第一折没有更早的历史可训练，按 walk-forward 的定义跳过。
        history = usable[:start]
        if not history:
            continue
        take = max(int(len(history) * config.train_fraction), 5)
        train_dates = history[-take:]
        folds.append({"fold": fold_index + 1, "train": train_dates, "test": test_dates, "train_start": train_dates[0], "test_start": test_dates[0]})

    rows: list[dict] = []
    for name, frame in values.items():
        for horizon in config.horizons:
            series = ic_series(frame, panel.close, horizon, usable, config.min_symbols)
            if series.empty:
                continue
            for fold_info in folds:
                train = series.reindex([d for d in fold_info["train"] if d in series.index]).dropna()
                test = series.reindex([d for d in fold_info["test"] if d in series.index]).dropna()
                if train.empty or test.empty:
                    continue
                train_stats = _stats(train.to_numpy(), horizon, config.step_days)
                direction = int(np.sign(train_stats["ic_mean"])) or 1
                # 测试段按训练段定的方向衡量：方向为负时，把 IC 取反
                signed_test = test * direction
                test_stats = _stats(signed_test.to_numpy(), horizon, config.step_days)
                signed_train = train * direction
                train_stats = _stats(signed_train.to_numpy(), horizon, config.step_days)
                selected = bool(np.isfinite(train_stats["t_stat"]) and train_stats["t_stat"] > config.t_threshold)
                survived = bool(
                    selected
                    and np.isfinite(test_stats["t_stat"])
                    and test_stats["t_stat"] > config.t_threshold
                )
                rows.append(
                    {
                        "factor": name,
                        "horizon": horizon,
                        "fold": fold_info["fold"],
                        "train_start": fold_info["train_start"],
                        "test_start": fold_info["test_start"],
                        "direction": "top" if direction > 0 else "bottom",
                        "train_periods": train_stats["periods"],
                        "train_ic": train_stats["ic_mean"],
                        "train_t": train_stats["t_stat"],
                        "test_periods": test_stats["periods"],
                        "test_ic": test_stats["ic_mean"],
                        "test_t": test_stats["t_stat"],
                        "selected": selected,
                        "survived": survived,
                    }
                )

    detail = pd.DataFrame(rows)
    return {"detail": detail, "summary": summarize_walk_forward(detail, config), "panel": panel, "folds": folds}


def summarize_walk_forward(detail: pd.DataFrame, config: FoldConfig | None = None) -> pd.DataFrame:
    """按 (因子, 持有期) 汇总：多少折被选中、其中多少折在测试段存活、测试段 IC 的均值与 t。"""
    config = config or FoldConfig()
    if detail is None or detail.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (factor, horizon), group in detail.groupby(["factor", "horizon"]):
        selected = group[group["selected"]]
        survived = selected[selected["survived"]]
        # 测试段 IC 在**所有折**上的合并统计（不只看选中的折），用折内均值再平均，避免折长不等
        test_ic = pd.to_numeric(selected["test_ic"], errors="coerce").dropna() if not selected.empty else pd.Series(dtype=float)
        test_t = pd.to_numeric(selected["test_t"], errors="coerce").dropna() if not selected.empty else pd.Series(dtype=float)
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "folds": int(group["fold"].nunique()),
                "folds_selected": len(selected),
                "folds_survived": len(survived),
                # 只在"被选中的折"里看测试段表现——这是诚实的口径：没被选中的折本来就不会交易
                "mean_test_ic": float(test_ic.mean()) if len(test_ic) else np.nan,
                "mean_test_t": float(test_t.mean()) if len(test_t) else np.nan,
                "median_test_t": float(test_t.median()) if len(test_t) else np.nan,
                "survival_rate": float(len(survived) / len(selected)) if len(selected) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["robust"] = (table["folds_selected"] >= 2) & (
        table["folds_survived"] >= np.ceil(table["folds_selected"] * 0.6)
    )
    return table.sort_values(["folds_survived", "mean_test_t"], ascending=False).reset_index(drop=True)
