"""公式化因子的 IC 评估流水线（v0.26）：把 `cli.py` 里的业务逻辑抽出来，并加上
**样本外切分**与**多重比较校正**。

为什么要抽出来：CLI 层应当只做参数解析与流程编排（见 `docs/ARCHITECTURE.md`），
而"评估一批因子"是实打实的研究逻辑，需要被测试直接覆盖。

为什么要切样本：一次评估 111 个 (因子 × 持有期) 组合，**样本内挑出来的最强因子**
拿到样本外很可能失效。本模块把时间轴按比例切成两段（默认前后各半），分别在两段上算 IC，
并给出第三列结论：**只在样本内显著、样本外不显著** = 大概率是噪声。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.alpha101 import build_panel, compute_alphas
from aqlab.factor_ic import spearman_ic
from aqlab.multiple_testing import p_value_two_sided, summarize_significance

__all__ = ["FactorEvalConfig", "evaluate_factor_ics", "ic_of_factor"]


@dataclass
class FactorEvalConfig:
    horizons: tuple[int, ...] = (1, 5, 20)
    step_days: int = 5
    min_history: int = 260
    min_symbols: int = 200
    alpha: float = 0.05
    is_fraction: float = 0.5           # 样本内占比（按交易日切分）
    factors: tuple[str, ...] = ()      # 空 = 全部已实现因子

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.min_symbols < 3:
            raise ValueError("min_symbols must be >= 3")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < self.is_fraction < 1:
            raise ValueError("is_fraction must be in (0, 1)")


def ic_of_factor(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int,
    dates: Sequence[pd.Timestamp],
    min_symbols: int,
) -> pd.Series:
    """在给定截面上算某因子相对未来 ``horizon`` 日收益的 IC 序列。"""
    values: list[float] = []
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
        if value == value:                      # 跳过 NaN
            values.append(value)
    return pd.Series(values, dtype=float)


def _stats(ics: pd.Series, horizon: int, step_days: int) -> dict:
    """把一段 IC 序列汇总成均值/IR/重叠修正 t 值。"""
    if ics.empty:
        return {"periods": 0, "ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "t_stat_adj": np.nan, "positive_rate": np.nan}
    mean = float(ics.mean())
    std = float(ics.std(ddof=1)) if len(ics) > 1 else np.nan
    ir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
    t_stat = ir * np.sqrt(len(ics)) if np.isfinite(ir) else np.nan
    overlap = max(1, int(np.ceil(horizon / step_days)))
    return {
        "periods": len(ics),
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "t_stat_adj": t_stat / np.sqrt(overlap) if np.isfinite(t_stat) else np.nan,
        "positive_rate": float((ics > 0).mean()),
    }


def evaluate_factor_ics(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    config: FactorEvalConfig | None = None,
) -> dict:
    """算全部因子在**全样本 / 样本内 / 样本外**三段的 IC，并给出多重比较校正结果。"""
    config = config or FactorEvalConfig()
    panel = build_panel(daily_by_symbol)
    values = compute_alphas(panel, names=list(config.factors) or None, min_history=config.min_history)
    if not values:
        raise ValueError("no factor could be computed (check min_history and data length)")

    dates = list(panel.dates)
    split = int(len(dates) * config.is_fraction)
    # 先在**整条时间轴**上按步长取样，再按切分点分成两段——这样两段的截面互不重叠、
    # 合起来正好等于全样本的截面集合。
    # 反例（曾经的写法）：对 in_sample 用 dates[::step]，对 out_sample 用 dates[split::step]，
    # 后者会从 split 处重新起相位，导致两段各自覆盖约一半截面却**相互重叠**。
    sampled = dates[:: config.step_days]
    in_sample = [date for date in sampled if date < dates[split]]
    out_sample = [date for date in sampled if date >= dates[split]]

    rows: list[dict] = []
    for name, frame in values.items():
        for horizon in config.horizons:
            full = ic_of_factor(frame, panel.close, horizon, sampled, config.min_symbols)
            inside = ic_of_factor(frame, panel.close, horizon, in_sample, config.min_symbols)
            outside = ic_of_factor(frame, panel.close, horizon, out_sample, config.min_symbols)
            row = {"factor": name, "horizon": horizon}
            for label, series in (("", full), ("is_", inside), ("os_", outside)):
                for key, value in _stats(series, horizon, config.step_days).items():
                    row[f"{label}{key}" if label else key] = value
            rows.append(row)

    table = pd.DataFrame(rows)
    # 校正列直接挂在**完整表**上（这样报告里 ic_mean / ic_ir 等列都还在）；
    # 注意不能对含 t_stat_adj 的表做 rename，那会把列名搬走留下重复列。
    is_significance = summarize_significance(
        table[["factor", "horizon", "is_t_stat_adj"]].rename(columns={"is_t_stat_adj": "t_stat_adj"}),
        t_column="t_stat_adj",
        alpha=config.alpha,
    )["table"]
    table["is_p_value"] = is_significance["p_value"].to_numpy()
    table["is_significant_raw"] = is_significance["significant_raw"].to_numpy()
    table["is_significant_bh"] = is_significance["significant_bh"].to_numpy()
    table["is_significant_bonferroni"] = is_significance["significant_bonferroni"].to_numpy()
    table["is_q_bh"] = is_significance["q_bh"].to_numpy()

    # 样本外是否同向且同样显著（用同一个显著性水平）
    table["os_same_sign"] = np.sign(table["ic_mean"]) == np.sign(table["os_ic_mean"])
    table["os_p_value"] = [p_value_two_sided(value) for value in table["os_t_stat_adj"]]
    table["os_significant"] = table["os_p_value"] <= config.alpha
    table["survives_oos"] = table["is_significant_bh"] & table["os_significant"] & table["os_same_sign"]

    # 全样本的校正（用于报告"若不切样本会声称多少个显著"）：同样挂在完整表上
    full_significance = summarize_significance(
        table[["factor", "horizon", "t_stat_adj"]],
        t_column="t_stat_adj",
        alpha=config.alpha,
    )
    table["p_value"] = full_significance["table"]["p_value"].to_numpy()
    table["q_bh"] = full_significance["table"]["q_bh"].to_numpy()
    table["significant_raw"] = full_significance["table"]["significant_raw"].to_numpy()
    table["significant_bh"] = full_significance["table"]["significant_bh"].to_numpy()
    table["significant_bonferroni"] = full_significance["table"]["significant_bonferroni"].to_numpy()
    return {
        "table": table,
        "full_significance": full_significance,
        "significance_is": is_significance,
        "panel": panel,
        "config": config,
        "split_date": dates[split] if split < len(dates) else None,
        "factors_skipped": None,
    }
