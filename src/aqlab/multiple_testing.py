"""多重比较校正（v0.26）：Bonferroni 与 Benjamini-Hochberg FDR。

**为什么需要它**：一次评估 111 个 (因子 × 持有期) 组合，即使全部是噪声，5% 显著性水平下
也期望出现约 5.5 个"显著"结果。不做校正就报告"41 个显著"，等于把假阳性当发现。

**为什么不用 scipy**：本项目核心依赖坚持 numpy/pandas + 标准库。这里需要的只是
正态分布双侧 p 值（``math.erfc`` 精确给出，无需近似）与两个排序型校正过程，
自己实现既够用又能把口径写清楚。样本截面数在 50~110 之间，用正态近似替代 t 分布
是常见做法；临界值差异在 1% 量级以内，且**结论的方向不敏感**（见测试里的对照）。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "benjamini_hochberg",
    "bonferroni",
    "expected_false_positives",
    "p_value_two_sided",
    "summarize_significance",
]


def p_value_two_sided(t_stat: float) -> float:
    """正态近似下的双侧 p 值；NaN/inf 输入返回 NaN（不假装是 0 或 1）。"""
    value = float(t_stat)
    if not math.isfinite(value):
        return float("nan")
    return math.erfc(abs(value) / math.sqrt(2.0))


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Bonferroni：把阈值收紧到 ``alpha / m``（控制族错误率 FWER）。"""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray([float(p) for p in p_values], dtype=float)
    m = values.size
    if m == 0:
        return {"method": "bonferroni", "m": 0, "threshold": alpha, "rejected": np.array([], dtype=bool), "adjusted": values}
    threshold = alpha / m
    adjusted = np.minimum(values * m, 1.0)
    return {
        "method": "bonferroni",
        "m": int(m),
        "threshold": threshold,
        "rejected": values <= threshold,
        "adjusted": adjusted,
    }


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg：控制错误发现率（FDR），比 Bonferroni 更有功效。

    排序后找最大的 ``k`` 使 ``p_(k) ≤ k/m·α``，前 ``k`` 个拒绝；
    同时给出单调化的调整后 p 值（q 值）。
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray([float(p) for p in p_values], dtype=float)
    m = values.size
    if m == 0:
        return {"method": "benjamini_hochberg", "m": 0, "threshold": alpha, "rejected": np.array([], dtype=bool), "adjusted": values}

    finite = np.isfinite(values)
    order = np.argsort(np.where(finite, values, np.inf))
    ranked = values[order]
    ranks = np.arange(1, m + 1, dtype=float)
    # 调整后 p 值：q_(k) = min_{j>=k} (m/j) * p_(j)，即从大到小取累积最小
    raw = np.where(finite[order], ranked * m / ranks, np.nan)
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.full(m, np.nan)
    adjusted[order] = adjusted_sorted

    # 拒绝集：排序后最大的 k 使 p_(k) ≤ k/m·α，则前 k 个（按 p 升序）全部拒绝。
    # 关键是用**排序后的 p 值**与秩比较；混用未排序数组会把判定整体错位。
    thresholds = alpha * ranks / m
    candidates = finite[order] & (ranked <= thresholds)
    rejected_sorted = np.zeros(m, dtype=bool)
    if candidates.any():
        k = int(np.max(np.nonzero(candidates)[0])) + 1
        rejected_sorted[:k] = True
    rejected = np.zeros(m, dtype=bool)
    rejected[order] = rejected_sorted
    return {
        "method": "benjamini_hochberg",
        "m": int(m),
        "threshold": float(alpha),
        "rejected": rejected,
        "adjusted": adjusted,
    }


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """不做校正时，期望出现的假阳性个数（用来解释"为什么必须校正"）。"""
    if n_tests < 0:
        raise ValueError("n_tests must be >= 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(n_tests) * float(alpha)


def summarize_significance(
    table: pd.DataFrame,
    t_column: str = "t_stat_adj",
    alpha: float = 0.05,
    label_columns: Iterable[str] = ("factor", "horizon"),
) -> dict:
    """给一张因子表加上 p 值与两种校正结果，返回汇总信息。

    返回 ``{"table": 带校正列的副本, "bonferroni": ..., "bh": ..., "expected_false_positives": ...}``。
    """
    if table is None or table.empty:
        return {"table": pd.DataFrame(), "bonferroni": bonferroni([], alpha), "bh": benjamini_hochberg([], alpha), "expected_false_positives": 0.0}
    if t_column not in table.columns:
        raise ValueError(f"table needs a '{t_column}' column")

    frame = table.copy()
    frame["p_value"] = [p_value_two_sided(value) for value in frame[t_column]]
    p_values = frame["p_value"].to_numpy(dtype=float)
    bh = benjamini_hochberg(p_values, alpha=alpha)
    bonf = bonferroni(p_values, alpha=alpha)
    frame["p_bonferroni"] = bonf["adjusted"]
    frame["q_bh"] = bh["adjusted"]
    frame["significant_raw"] = frame["p_value"] <= alpha
    frame["significant_bonferroni"] = bonf["rejected"]
    frame["significant_bh"] = bh["rejected"]
    return {
        "table": frame,
        "bonferroni": bonf,
        "bh": bh,
        "expected_false_positives": expected_false_positives(len(frame), alpha),
        "label_columns": list(label_columns),
    }
