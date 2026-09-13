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

__all__ = [
    "monthly_return_matrix",
    "plot_drawdown",
    "plot_equity_curves",
    "plot_factor_ic",
    "plot_ic_term_structure",
    "plot_monthly_heatmap",
    "plot_quantile_returns",
    "plot_scheme_curves",
    "plot_strategy_comparison",
]


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as error:
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
    for axis, column in zip(axes, metrics, strict=False):
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


def monthly_return_matrix(equity: pd.Series) -> pd.DataFrame:
    """把净值序列折成 **年 × 月** 的收益矩阵（单位 %）。

    第一个月相对序列起始值计算，避免把"建仓前"的零点当成一个月。
    """
    series = pd.Series(equity).astype(float).sort_index()
    if len(series) < 2:
        raise ValueError("equity series needs at least two points")
    month_end = series.resample("ME").last()
    base = pd.Series([series.iloc[0]], index=pd.DatetimeIndex([series.index[0] - pd.Timedelta(days=1)]))
    returns = pd.concat([base, month_end]).pct_change().dropna()
    frame = pd.DataFrame({"year": returns.index.year, "month": returns.index.month, "ret": returns.to_numpy()})
    matrix = frame.pivot(index="year", columns="month", values="ret") * 100.0
    return matrix.reindex(columns=range(1, 13))


def plot_monthly_heatmap(matrix: pd.DataFrame, path: str | Path, title: str = "Monthly returns (%)") -> Path:
    """月度收益热力图（红绿对称色标，中心为 0）。"""
    plt = _pyplot()
    if matrix is None or matrix.empty:
        raise ValueError("monthly matrix is empty")
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = limit if limit > 0 else 1.0
    height = max(1.8, 0.5 * len(matrix) + 1.4)
    figure, axis = plt.subplots(figsize=(9.2, height), dpi=140)
    image = axis.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels([f"{month:02d}" for month in matrix.columns], fontsize=8)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([str(year) for year in matrix.index], fontsize=8)
    axis.set_xlabel("month", fontsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=7, color="#222222")
    figure.colorbar(image, ax=axis, shrink=0.85, label="return (%)")
    axis.set_title(title)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_factor_ic(summary: pd.DataFrame, path: str | Path, title: str = "Factor IC (Spearman)") -> Path:
    """两个子图：因子平均 IC（带正负色）与 IC_IR。"""
    plt = _pyplot()
    if summary is None or summary.empty:
        raise ValueError("IC summary is empty")
    frame = summary.copy()
    for column in ("ic_mean", "ic_ir"):
        if column not in frame.columns:
            raise ValueError(f"IC summary needs a '{column}' column")
    labels = frame["factor"].astype(str).tolist()
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=140)
    for axis, column, note in zip(axes, ("ic_mean", "ic_ir"), ("mean IC", "IC_IR (mean / std)"), strict=False):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        colors = ["#2e7d32" if value >= 0 else "#c0392b" for value in np.nan_to_num(values)]
        axis.bar(positions, np.nan_to_num(values), color=colors, alpha=0.85)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axis.set_title(note, fontsize=9)
        axis.grid(alpha=0.2, axis="y")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_quantile_returns(table: pd.DataFrame, path: str | Path, title: str = "Forward return by factor quantile (%)") -> Path:
    """每个因子一条线，横轴为分位组（1 = 因子值最低），纵轴为平均前瞻收益。"""
    plt = _pyplot()
    if table is None or table.empty:
        raise ValueError("quantile table is empty")
    figure, axis = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    for factor, group in table.groupby("factor"):
        ordered = group.sort_values("group")
        axis.plot(ordered["group"], ordered["mean_forward"] * 100.0, marker="o", linewidth=1.4, label=str(factor))
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set_xlabel("quantile (1 = lowest factor value)")
    axis.set_ylabel("mean forward return (%)")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False, ncol=2)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_ic_term_structure(summary: pd.DataFrame, path: str | Path, title: str = "IC by holding horizon") -> Path:
    """因子的 IC 期限结构：横轴持有期，纵轴平均 IC，每个因子一条线。"""
    plt = _pyplot()
    if summary is None or summary.empty or "horizon" not in summary.columns:
        raise ValueError("horizon summary is empty")
    figure, axis = plt.subplots(figsize=(7.6, 3.8), dpi=140)
    for factor, group in summary.groupby("factor"):
        ordered = group.sort_values("horizon")
        axis.plot(ordered["horizon"], ordered["ic_mean"], marker="o", linewidth=1.5, label=str(factor))
    axis.axhline(0, color="#444444", linewidth=0.9)
    axis.set_xlabel("holding horizon (trading days)")
    axis.set_ylabel("mean Spearman IC")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_scheme_curves(
    curves: Mapping[str, pd.Series],
    path: str | Path,
    title: str = "Top-N portfolios by weighting scheme (net of costs)",
) -> Path:
    """多方案的累计净值曲线（含等权全市场基准）。"""
    plt = _pyplot()
    if not curves:
        raise ValueError("no curve to plot")
    figure, axis = plt.subplots(figsize=(9.2, 4.0), dpi=140)
    for label, series in curves.items():
        values = pd.Series(series).astype(float).sort_index()
        style = {"linestyle": "--", "color": "#666666", "linewidth": 1.4} if label == "benchmark" else {"linewidth": 1.6}
        axis.plot(values.index, values.to_numpy(), label=str(label), **style)
    axis.axhline(1.0, color="#bbbbbb", linewidth=0.8)
    axis.set_ylabel("cumulative net value")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target
