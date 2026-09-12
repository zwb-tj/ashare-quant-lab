"""组合层（v0.5）：权重优化、风格暴露、换手约束与再平衡成本显式化。

设计取舍（都写在这里，避免"看起来更专业但算不清"）：

* **长仓、全额约束**：``w ≥ 0``、``Σw ≤ 1 - cash_buffer``、``w ≤ max_weight``；只用 numpy 实现
  投影（不引入 scipy），保证核心依赖仍是 numpy/pandas。
* **权重漂移被真实模拟**：持仓以"份额"记账，价格变动会自动产生漂移权重，再平衡费按
  ``Σ|Δw| × 成本率`` 计——不是"假设权重永远不变"的简化。
* **换手约束**：目标权重与当前漂移权重之间的单边换手超过上限时，按比例向目标插值
  （插值不破坏凸约束），而不是硬砍某一笔。
* **暴露只报能算的东西**：波动率/动量/对等权指数的 beta/集中度；**不做假的行业暴露**
  （没有行业数据就不要编）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.metrics import compute_metrics
from aqlab.tables import markdown_table

__all__ = [
    "PortfolioConfig",
    "project_to_capped_simplex",
    "covariance_matrix",
    "equal_weights",
    "inverse_vol_weights",
    "risk_parity_weights",
    "min_variance_weights",
    "mean_variance_weights",
    "optimize_weights",
    "apply_constraints",
    "exposure_report",
    "simulate_portfolio",
    "format_portfolio_report",
    "write_portfolio_report",
]

METHODS = ("equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance")


@dataclass
class PortfolioConfig:
    """组合构建与再平衡参数。"""

    method: str = "risk_parity"
    lookback: int = 60
    rebalance_days: int = 5
    max_weight: float = 0.20
    cash_buffer: float = 0.20      # 总仓 ≤80%：永远留 20% 现金
    turnover_limit: float = 0.30   # 单边换手上限（比例）
    cost_bps: float = 5.0
    shrinkage: float = 0.10        # 协方差向对角矩阵收缩
    risk_aversion: float = 1.0
    min_history: int = 120
    min_positions: float = 3.0     # 分散度目标：平均持仓低于它就在报告里报警

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.lookback < 5:
            raise ValueError("lookback must be >= 5")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if not 0 < self.turnover_limit <= 2:
            raise ValueError("turnover_limit must be in (0, 2]")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")
        if not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be > 0")
        if self.min_history < 20:
            raise ValueError("min_history must be >= 20")
        if self.min_positions < 1:
            raise ValueError("min_positions must be >= 1")


# --------------------------------------------------------------------------------------
# 数值工具
# --------------------------------------------------------------------------------------
def project_to_capped_simplex(v: np.ndarray, total: float = 1.0, cap: float = 1.0) -> np.ndarray:
    """投影到 ``{0 ≤ w ≤ cap, Σw = min(total, n·cap)}``（二分求阈值，稳健且不越界）。

    注意：当 ``n · cap < total``（单票上限使预算不可行）时，结果会**保留现金**而不是
    偷偷放宽上限——上限是风控约束，不能被算法"优化掉"。
    """
    w = np.nan_to_num(np.asarray(v, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(w)
    if n == 0:
        return w
    target = min(float(total), n * float(cap))
    lo, hi = float(w.min()) - target - 1.0, float(w.max())
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if np.clip(w - mid, 0.0, cap).sum() > target:
            lo = mid
        else:
            hi = mid
    return np.clip(w - (lo + hi) / 2.0, 0.0, cap)


def covariance_matrix(returns: pd.DataFrame, shrinkage: float = 0.10, periods_per_year: int = 252) -> pd.DataFrame:
    """年化协方差矩阵，按 ``(1-δ)Σ + δ·diag(Σ)`` 收缩并加微量 ridge 保证可解。"""
    if returns.shape[1] == 0:
        raise ValueError("returns must have at least one column")
    cov = returns.cov().to_numpy(dtype=float) * periods_per_year
    if cov.shape[0] == 1:
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    diag = np.diag(np.diag(cov))
    shrunk = (1 - shrinkage) * cov + shrinkage * diag
    shrunk = shrunk + np.eye(shrunk.shape[0]) * 1e-10
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


# --------------------------------------------------------------------------------------
# 权重方法
# --------------------------------------------------------------------------------------
def equal_weights(symbols: Sequence[str], total: float = 1.0) -> pd.Series:
    if not symbols:
        raise ValueError("symbols must not be empty")
    return pd.Series(total / len(symbols), index=list(symbols), dtype=float)


def inverse_vol_weights(returns: pd.DataFrame, total: float = 1.0) -> pd.Series:
    vol = returns.std(ddof=0).replace(0.0, np.nan)
    inv = (1.0 / vol).fillna(0.0)
    if inv.sum() <= 0:
        return equal_weights(list(returns.columns), total)
    return (inv / inv.sum() * total).rename(None)


def risk_parity_weights(
    cov: pd.DataFrame,
    total: float = 1.0,
    budget: Sequence[float] | None = None,
    iterations: int = 500,
    tol: float = 1e-10,
) -> pd.Series:
    """风险平价（风险贡献相等）：乘法更新迭代，长仓且不做空。"""
    sigma = cov.to_numpy(dtype=float)
    n = sigma.shape[0]
    target_share = np.ones(n) / n if budget is None else np.asarray(budget, dtype=float) / np.sum(budget)
    w = np.ones(n) / n
    for _ in range(iterations):
        marginal = sigma @ w
        rc = w * marginal
        rc = np.clip(rc, 1e-16, None)
        total_rc = rc.sum()
        desired = target_share * total_rc
        updated = w * np.sqrt(desired / rc)
        updated = np.clip(updated, 1e-12, None)
        updated = updated / updated.sum()
        if np.max(np.abs(updated - w)) < tol:
            w = updated
            break
        w = updated
    return pd.Series(w / w.sum() * total, index=cov.columns, dtype=float)


def min_variance_weights(cov: pd.DataFrame, total: float = 1.0, cap: float = 1.0, steps: int = 800) -> pd.Series:
    """长仓最小方差：投影梯度下降（numpy 实现，无需 QP 求解器）。"""
    sigma = cov.to_numpy(dtype=float)
    n = sigma.shape[0]
    w = np.ones(n) / n
    largest = float(np.linalg.eigvalsh(sigma).max()) if n > 1 else float(sigma[0, 0])
    lr = 1.0 / (2 * largest) if largest > 0 else 0.01
    for _ in range(steps):
        w = project_to_capped_simplex(w - lr * (2 * sigma @ w), total=total, cap=cap)
    return pd.Series(w, index=cov.columns, dtype=float)


def mean_variance_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    total: float = 1.0,
    cap: float = 1.0,
    risk_aversion: float = 1.0,
    steps: int = 800,
) -> pd.Series:
    """均值-方差（效用最大化）：``max μᵀw - λ wᵀΣw``，长仓 + 上限约束。"""
    sigma = cov.to_numpy(dtype=float)
    expected = mu.reindex(cov.columns).fillna(0.0).to_numpy(dtype=float)
    n = sigma.shape[0]
    w = np.ones(n) / n
    largest = float(np.linalg.eigvalsh(sigma).max()) if n > 1 else float(sigma[0, 0])
    lr = 1.0 / (2 * risk_aversion * largest) if largest > 0 else 0.01
    for _ in range(steps):
        grad = 2 * risk_aversion * (sigma @ w) - expected
        w = project_to_capped_simplex(w - lr * grad, total=total, cap=cap)
    return pd.Series(w, index=cov.columns, dtype=float)


def optimize_weights(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    cap: float = 1.0,
    shrinkage: float = 0.10,
    risk_aversion: float = 1.0,
    total: float = 1.0,
) -> pd.Series:
    """按方法名分派；输入是**日收益**（内部年化协方差/均值）。"""
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must not be empty")
    if method == "equal":
        return equal_weights(list(returns.columns), total)
    if method == "inverse_vol":
        return inverse_vol_weights(returns, total)
    cov = covariance_matrix(returns, shrinkage=shrinkage)
    if method == "risk_parity":
        return risk_parity_weights(cov, total=total)
    if method == "min_variance":
        return min_variance_weights(cov, total=total, cap=cap)
    if method == "mean_variance":
        mu = returns.mean() * 252
        return mean_variance_weights(mu, cov, total=total, cap=cap, risk_aversion=risk_aversion)
    raise ValueError(f"unknown method '{method}'")


def apply_constraints(
    target: pd.Series,
    previous: Mapping[str, float] | None,
    config: PortfolioConfig,
    total: float | None = None,
) -> tuple[pd.Series, float]:
    """施加现金缓冲 / 单票上限 / 换手约束，返回 ``(最终权重, 触发的约束说明)``。

    换手约束用"向目标插值"实现：``w = prev + λ(target - prev)``，``λ = limit / 实际换手``。
    """
    budget = (1.0 - config.cash_buffer) if total is None else total
    target = target.astype(float).clip(lower=0.0)
    if target.sum() <= 0:
        return pd.Series(0.0, index=target.index, dtype=float), 0.0
    capped = target / target.sum() * budget
    # 单票上限是风控约束：不可行时保留现金，绝不放宽上限
    capped = pd.Series(
        project_to_capped_simplex(capped.to_numpy(), total=budget, cap=config.max_weight),
        index=capped.index,
    )

    prev = pd.Series(0.0, index=capped.index, dtype=float)
    if previous:
        prev = pd.Series(previous, dtype=float).reindex(capped.index).fillna(0.0)

    turnover = float(0.5 * (capped - prev).abs().sum())
    if turnover > config.turnover_limit and turnover > 0:
        lam = config.turnover_limit / turnover
        capped = prev + lam * (capped - prev)
        turnover = config.turnover_limit
    return capped, turnover


# --------------------------------------------------------------------------------------
# 暴露与集中度
# --------------------------------------------------------------------------------------
def exposure_report(weights: Mapping[str, float], universe: Mapping[str, pd.DataFrame], lookback: int = 60) -> dict:
    """组合的集中度与风格暴露（只报能算的：波动率/动量/beta/集中度）。"""
    active = {s: float(w) for s, w in weights.items() if w > 1e-9 and s in universe}
    gross = float(sum(active.values()))
    report: dict[str, Any] = {
        "gross_exposure": round(gross, 4),
        "positions": len(active),
        "max_weight": round(max(active.values()), 4) if active else 0.0,
        "hhi": round(sum((w / gross) ** 2 for w in active.values()), 4) if gross > 0 else 0.0,
    }
    if gross > 0:
        report["effective_n"] = round(1.0 / report["hhi"], 2) if report["hhi"] > 0 else 0.0

    if not active:
        return report

    # 等权指数作为市场代理，用于计算 beta
    closes = pd.DataFrame({s: universe[s]["close"].astype(float) for s in active}).tail(lookback).ffill()
    index_ret = closes.pct_change().mean(axis=1).dropna()
    weighted_vol = weighted_mom = weighted_beta = 0.0
    for symbol, weight in active.items():
        series = closes[symbol].pct_change().dropna()
        if len(series) >= 5:
            weighted_vol += weight * float(series.std(ddof=0) * np.sqrt(252))
            weighted_mom += weight * float(closes[symbol].iloc[-1] / closes[symbol].iloc[0] - 1.0)
            var = float(index_ret.var(ddof=0))
            if var > 0:
                aligned = pd.concat([series, index_ret], axis=1).dropna()
                if len(aligned) >= 5:
                    cov = float(aligned.cov().iloc[0, 1])
                    weighted_beta += weight * (cov / var)
    report["weighted_ann_vol"] = round(weighted_vol, 4)
    report["weighted_momentum"] = round(weighted_mom, 4)
    report["weighted_beta"] = round(weighted_beta, 3)
    return report


# --------------------------------------------------------------------------------------
# 组合模拟（份额记账 + 再平衡成本）
# --------------------------------------------------------------------------------------
def simulate_portfolio(
    universe: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.Series],
    config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    """按再平衡周期构建组合并模拟净值（含成本、现金缓冲、换手约束）。

    * 权重在再平衡日收盘后用**截至当日**的收益估计（无未来函数），从下一根 K 线开始生效；
    * 持仓用份额记账，价格变动自动产生漂移权重；
    * 每次再平衡按 ``Σ|Δw| × cost_bps/1e4 × 组合市值`` 扣费。
    """
    config = config or PortfolioConfig()
    if not universe:
        raise ValueError("universe is empty")

    dates = sorted({date for df in universe.values() for date in df.index})
    dates = pd.DatetimeIndex(dates)
    if len(dates) < config.min_history + config.rebalance_days:
        return {"frame": pd.DataFrame(), "rebalances": pd.DataFrame(), "metrics": {}, "summary": {"note": "数据不足"}}

    price = pd.DataFrame({s: df["close"].astype(float) for s, df in universe.items()}).reindex(dates).ffill()
    returns = price.pct_change()

    value = 1.0
    cash = 1.0
    units: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    rebalances: list[dict] = []
    rows: list[dict] = []

    signal_frame = pd.DataFrame({s: sig.reindex(dates).fillna(False).astype(bool) for s, sig in signals.items()}).reindex(dates)

    for bar, date in enumerate(dates):
        # 1) 先按当日价格重估组合
        prices_today = price.loc[date]
        holdings_value = sum(u * float(prices_today.get(s, np.nan)) for s, u in units.items() if prices_today.get(s) == prices_today.get(s))
        total_value = cash + holdings_value
        if total_value <= 0:
            total_value = 0.0

        weights_now = {
            s: (u * float(prices_today.get(s, 0.0)) / total_value) if total_value > 0 else 0.0
            for s, u in units.items()
        }

        turnover_today = 0.0
        if bar >= config.min_history and (bar - config.min_history) % config.rebalance_days == 0:
            eligible = [
                s
                for s in universe
                if bool(signal_frame[s].loc[date]) and len(price[s].loc[:date].dropna()) >= config.min_history
            ]
            if eligible:
                window = returns[eligible].loc[:date].tail(config.lookback).dropna(how="all").fillna(0.0)
                if len(window) >= 5:
                    target = optimize_weights(
                        window,
                        method=config.method,
                        cap=config.max_weight,
                        shrinkage=config.shrinkage,
                        risk_aversion=config.risk_aversion,
                        total=1.0,
                    )
                    final_weights, turnover = apply_constraints(target, weights_now, config)
                else:
                    final_weights, turnover = pd.Series(dtype=float), 0.0
            else:
                final_weights, turnover = pd.Series(dtype=float), 0.0

            # 换仓：先卖后买，成本按名义换手计
            traded_notional = float(0.5 * sum(abs(final_weights.get(s, 0.0) - weights_now.get(s, 0.0)) for s in set(final_weights.index) | set(weights_now)))
            if traded_notional <= 1e-12 and not final_weights.any():
                # 空仓 -> 继续空仓：不算一次再平衡，也不记流水
                rows.append(
                    {
                        "date": date,
                        "value": total_value,
                        "gross_exposure": 0.0,
                        "turnover": 0.0,
                        "positions": 0,
                    }
                )
                continue

            cost = traded_notional * config.cost_bps / 10_000.0 * total_value
            turnover_today = traded_notional

            invest_value = max(total_value - cost, 0.0)
            units = {}
            for s, w in final_weights.items():
                if w <= 1e-9:
                    continue
                p = float(prices_today.get(s, np.nan))
                if p == p and p > 0:
                    units[s] = w * invest_value / p
            cash = max(invest_value - sum(u * float(prices_today.get(s, 0.0)) for s, u in units.items()), 0.0)
            current_weights = {s: float(w) for s, w in final_weights.items() if w > 1e-9}
            rebalances.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "positions": len(current_weights),
                    "gross_exposure": round(float(sum(current_weights.values())), 4),
                    "turnover": round(turnover, 4),
                    "cost": round(float(cost), 6),
                    "weights": {s: round(w, 4) for s, w in sorted(current_weights.items())},
                }
            )

        rows.append(
            {
                "date": date,
                "value": total_value,
                "gross_exposure": float(sum(weights_now.values())),
                "turnover": turnover_today,
                "positions": len(weights_now),
            }
        )

    frame = pd.DataFrame(rows).set_index("date")
    frame["strat_ret"] = frame["value"].pct_change().fillna(0.0)
    frame["equity"] = frame["value"]
    frame["pos"] = frame["gross_exposure"]
    metrics = compute_metrics(frame, initial_cash=1.0, trades=None)
    summary = {
        "method": config.method,
        "rebalances": len(rebalances),
        "avg_positions": round(float(np.mean([r["positions"] for r in rebalances])), 2) if rebalances else 0.0,
        "avg_turnover": round(float(np.mean([r["turnover"] for r in rebalances])), 4) if rebalances else 0.0,
        "total_cost": round(float(sum(r["cost"] for r in rebalances)), 6),
        "final_equity": round(float(frame["value"].iloc[-1]), 4),
    }
    # 分散度体检：组合层无法弥补信号层过于稀疏
    if rebalances:
        budget = 1.0 - config.cash_buffer
        avg_positions = float(np.mean([r["positions"] for r in rebalances]))
        avg_gross = float(np.mean([r["gross_exposure"] for r in rebalances]))
        if avg_positions < config.min_positions:
            summary["warning"] = (
                f"平均持仓仅 {avg_positions:.1f} 只，低于目标 {config.min_positions:g} 只："
                "组合层无法弥补信号层过于稀疏，应放宽规则或扩大票池"
            )
        elif avg_gross < 0.5 * budget:
            summary["warning"] = (
                f"平均仓位 {avg_gross:.0%} 远低于预算 {budget:.0%}：单票上限 {config.max_weight:.0%} "
                "限制了小票池的可投比例，考虑放宽上限或扩大票池"
            )
    return {"frame": frame, "rebalances": pd.DataFrame(rebalances), "metrics": metrics, "summary": summary}


def format_portfolio_report(result: Mapping[str, Any], exposure: Mapping[str, Any] | None = None, config: PortfolioConfig | None = None) -> str:
    lines = ["# 组合层报告", ""]
    summary = result.get("summary", {})
    if not summary or summary.get("note"):
        lines.append("数据不足，无法构建组合（检查 --days / --min-history）。")
        return "\n".join(lines)

    lines.append("## 配置")
    lines.append("")
    if config is not None:
        lines.append(
            f"- 方法：`{config.method}`｜再平衡：每 {config.rebalance_days} 根｜单票上限 {config.max_weight:.0%}"
            f"｜现金缓冲 {config.cash_buffer:.0%}｜换手上限 {config.turnover_limit:.0%}｜成本 {config.cost_bps:g} bps"
        )
        lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    for key, value in summary.items():
        if key == "warning":
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")
    if summary.get("warning"):
        lines.append(f"> ⚠️ **体检提示**：{summary['warning']}")
        lines.append("")

    rebalances = result.get("rebalances")
    if rebalances is not None and not rebalances.empty:
        lines.append("## 最近 5 次再平衡")
        lines.append("")
        view = rebalances.tail(5).copy()
        view["weights"] = view["weights"].apply(lambda d: ", ".join(f"{k}:{v}" for k, v in list(d.items())[:6]) or "-")
        lines.append(markdown_table(view[["date", "positions", "gross_exposure", "turnover", "cost", "weights"]]))
        lines.append("")

    metrics = result.get("metrics") or {}
    if metrics:
        lines.append("## 组合绩效（净值口径，已扣再平衡成本）")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | ---: |")
        for key in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "annual_turnover", "exposure"):
            if key in metrics and metrics[key] == metrics[key]:
                value = metrics[key]
                lines.append(f"| {key} | {value * 100:.2f}% |" if key in ("total_return", "cagr", "ann_vol", "max_drawdown", "annual_turnover", "exposure") else f"| {key} | {value:.3f} |")
        lines.append("")

    if exposure:
        lines.append("## 集中度与风格暴露（最近一次再平衡）")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | ---: |")
        for key, value in exposure.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("> 只报能算的暴露：波动率/动量/对等权指数的 beta/集中度。**没有行业数据就不编行业暴露。**")
    lines.append("")
    lines.append("> 提醒：若使用合成数据（未传 `--data-dir`），以上绩效只用于验证流程，**不代表真实市场表现**。")
    return "\n".join(lines)


def write_portfolio_report(outdir: str | Path, result: Mapping[str, Any], exposure: Mapping[str, Any] | None = None, config: PortfolioConfig | None = None) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "portfolio.md"
    equity_path = out / "equity.csv"
    rebalance_path = out / "rebalances.csv"
    json_path = out / "portfolio.json"
    md_path.write_text(format_portfolio_report(result, exposure, config), encoding="utf-8")
    result.get("frame", pd.DataFrame()).to_csv(equity_path, encoding="utf-8-sig")
    result.get("rebalances", pd.DataFrame()).to_csv(rebalance_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({"summary": result.get("summary", {}), "exposure": exposure or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"markdown": md_path, "equity": equity_path, "rebalances": rebalance_path, "json": json_path}
