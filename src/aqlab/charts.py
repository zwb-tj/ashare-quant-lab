"""可视化（v0.9）：净值曲线 / 回撤 / 策略对比。

设计约束：

* **matplotlib 是可选依赖**（``pip install -e ".[plot]"``）——没装时给出明确提示而不是 ImportError 崩塌；
* 图表只画回测产出的事实（净值、回撤、基准），不做视觉修饰性夸大；
* 中文字体在多数 Linux 容器里缺失，因此默认使用**英文标签**，保证 CI 与服务器上渲染一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.metrics import drawdown_series

__all__ = ["plot_equity_curves", "plot_drawdown", "plot_strategy_comparison"]


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as error:  # noqa: BLE001
        raise RuntimeError('charts need the optional "plot" extra: pip install -e ".[plot]"') from error


def _equity_of(frame: pd.DataFrame) -> pd.Series:
    for column in ("equity", "nav", "value"):
        if column in frame.columns:
            return frame[column].astype(float)
    raise ValueError("frame must contain an 'equity', 'nav' or 'value' column")


def plot_equity_curves(
    curves: Mapping[str, pd.DataFrame],
    path: str | Path,
    title: str = "Equity curves",
    benchmark: pd.Series | None = None,
) -> Path:
    """把多条净值曲线画在一张图上（可叠加基准）。"""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 4.5), dpi=140)
    for label, frame in curves.items():
        equity = _equity_of(frame)
        axis.plot(equity.index, equity.to_numpy(), linewidth=1.5, label=label)
    if benchmark is not None and len(benchmark):
        axis.plot(benchmark.index, benchmark.to_numpy(), linewidth=1.2, linestyle="--", color="#888888", label="benchmark")
    axis.set_title(title)
    axis.set_ylabel("Equity (normalised)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_drawdown(frame: pd.DataFrame, path: str | Path, title: str = "Drawdown") -> Path:
    """单独画回撤（水下）曲线。"""
    plt = _pyplot()
    equity = _equity_of(frame)
    drawdown = drawdown_series(equity) * 100.0
    figure, axis = plt.subplots(figsize=(9, 2.8), dpi=140)
    axis.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#c0392b", alpha=0.35)
    axis.plot(drawdown.index, drawdown.to_numpy(), color="#c0392b", linewidth=1.0)
    axis.set_title(title)
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_strategy_comparison(
    summary: pd.DataFrame,
    path: str | Path,
    metric_columns: Sequence[str] = ("total_return", "sharpe", "max_drawdown", "win_rate_trade"),
    label_column: str = "strategy",
    title: str = "Strategy comparison",
) -> Path:
    """把多个策略的核心指标画成条形图（每个指标独立归一，便于横向比较）。"""
    plt = _pyplot()
    metrics = [column for column in metric_columns if column in summary.columns]
    if not metrics:
        raise ValueError("summary has none of the requested metric columns")
    figure, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2), dpi=140)
    if len(metrics) == 1:
        axes = [axes]
    labels = summary[label_column].astype(str).tolist()
    positions = np.arange(len(labels))
    for axis, column in zip(axes, metrics):
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
        colors = ["#2e7d32" if value >= 0 else "#c0392b" for value in np.nan_to_num(values)]
        axis.bar(positions, np.nan_to_num(values), color=colors, alpha=0.85)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        axis.set_title(column, fontsize=9)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target
