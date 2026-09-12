"""参数扫描（v0.6）：让"该放宽哪个参数"用数据说话，而不是拍脑袋。

对每一个参数组合，依次跑三件事并汇总成一行：

1. **事件研究**（`:mod:`aqlab.study`）→ 信号数、超额均值、超额胜率（主持有期）；
2. **滚动窗口校验**（`:mod:`aqlab.walkforward`）→ 交易数、胜率、单笔均收益、同期间超额、跑赢窗口数；
3. **组合层**（`:mod:`aqlab.portfolio`）→ 平均持仓、累计成本、期末净值、Sharpe、最大回撤。

最后按目标函数给出建议：**在满足"平均持仓 ≥ 目标"的组合里，取超额最高的那个**；
如果没有任何组合达标，就取平均持仓最高的那个，并明确说明"没有配置达标"。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.portfolio import PortfolioConfig, simulate_portfolio
from aqlab.rules import build_rule
from aqlab.study import study_profile
from aqlab.tables import markdown_table
from aqlab.walkforward import WalkForwardConfig, composite_scores, walk_forward

__all__ = [
    "SweepConfig",
    "GridSpec",
    "parse_grid",
    "override_bindings",
    "run_sweep",
    "pick_best",
    "format_sweep",
    "write_sweep",
]


@dataclass
class SweepConfig:
    """扫描配置（也是三个子分析的公共参数）。"""

    horizons: tuple[int, ...] = (1, 3, 5, 10)
    main_horizon: int = 5
    test_days: int = 60
    step_days: int = 60
    min_history: int = 120
    method: str = "risk_parity"
    max_weight: float = 0.20
    cash_buffer: float = 0.20
    turnover_limit: float = 0.30
    cost_bps: float = 5.0
    rebalance_days: int = 5

    def __post_init__(self) -> None:
        if self.main_horizon not in self.horizons:
            raise ValueError("main_horizon must be one of horizons")


@dataclass
class GridSpec:
    """一个参数的扫描轴：``rule.param=value1,value2``。"""

    rule: str
    param: str
    values: list[Any] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.rule}.{self.param}"


def _coerce(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_grid(specs: Sequence[str]) -> list[GridSpec]:
    """解析 ``--set rule.param=v1,v2``；支持多个 ``--set``（笛卡尔积）。"""
    grids: list[GridSpec] = []
    for spec in specs:
        if "=" not in spec or "." not in spec.split("=", 1)[0]:
            raise ValueError(f"bad --set '{spec}', expected rule.param=value1,value2")
        target, raw_values = spec.split("=", 1)
        rule, param = target.split(".", 1)
        values = [_coerce(v) for v in raw_values.split(",") if v.strip() != ""]
        if not values:
            raise ValueError(f"no values in --set '{spec}'")
        grids.append(GridSpec(rule=rule, param=param, values=values))
    return grids


def override_bindings(
    bindings: Sequence[tuple[str, Mapping[str, Any], float]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, dict, float]]:
    """把 ``{rule: {param: value}}`` 覆盖到档案绑定的副本上（不修改原档案）。"""
    out: list[tuple[str, dict, float]] = []
    for name, params, weight in bindings:
        merged = dict(params)
        if name in overrides:
            merged.update(overrides[name])
        out.append((name, merged, weight))
    return out


def _rule_excess(study_table: pd.DataFrame, horizon: int) -> tuple[float, float, int]:
    """汇总某持有期下所有规则的平均超额均值 / 平均超额胜率 / 总信号数。"""
    view = study_table[study_table["horizon"] == horizon]
    if view.empty:
        return float("nan"), float("nan"), 0
    return (
        float(view["excess_mean"].mean()),
        float(view["excess_win_rate"].mean()),
        int(view["signals"].sum()),
    )


def run_sweep(
    universes: Mapping[str, Mapping[str, pd.DataFrame]],
    base_bindings: Sequence[tuple[str, Mapping[str, Any], float]],
    grids: Sequence[GridSpec],
    config: SweepConfig | None = None,
) -> pd.DataFrame:
    """对 (参数组合 × 票池规模) 逐点运行三项分析，返回对比表。"""
    config = config or SweepConfig()
    if not universes:
        raise ValueError("universes must not be empty")
    if not base_bindings:
        raise ValueError("base_bindings must not be empty")

    axes = [grid.values for grid in grids] if grids else [[]]
    rows: list[dict] = []

    for combination in itertools.product(*axes):
        overrides: dict[str, dict[str, Any]] = {}
        label_parts: list[str] = []
        for grid, value in zip(grids, combination):
            overrides.setdefault(grid.rule, {})[grid.param] = value
            label_parts.append(f"{grid.label}={value}")
        bindings = override_bindings(base_bindings, overrides)
        param_label = ", ".join(label_parts) if label_parts else "baseline"

        for universe_label, universe in universes.items():
            study_table, _baseline = study_profile(
                universe, bindings, horizons=config.horizons, min_history=config.min_history
            )
            excess_mean, excess_win, signals = _rule_excess(study_table, config.main_horizon)

            wf = walk_forward(
                universe,
                bindings,
                config=WalkForwardConfig(
                    test_days=config.test_days,
                    step_days=config.step_days,
                    min_history=config.min_history,
                    use_position=True,
                ),
            )
            wf_summary = wf["summary"]

            scores = composite_scores(universe, bindings)
            signals_map = {sym: score > 0 for sym, score in scores.items()}
            portfolio = simulate_portfolio(
                universe,
                signals_map,
                PortfolioConfig(
                    method=config.method,
                    rebalance_days=config.rebalance_days,
                    max_weight=config.max_weight,
                    cash_buffer=config.cash_buffer,
                    turnover_limit=config.turnover_limit,
                    cost_bps=config.cost_bps,
                    min_history=config.min_history,
                ),
            )
            metrics = portfolio.get("metrics") or {}
            summary = portfolio.get("summary") or {}

            rows.append(
                {
                    "params": param_label,
                    "universe": universe_label,
                    "signals": signals,
                    "excess_mean": excess_mean,
                    "excess_win_rate": excess_win,
                    "trades": wf_summary.get("trades", 0),
                    "trade_win_rate": wf_summary.get("win_rate", float("nan")),
                    "trade_excess": wf_summary.get("avg_excess_return", float("nan")),
                    "windows_beating": wf_summary.get("windows_beating_benchmark", 0),
                    "avg_positions": summary.get("avg_positions", 0.0),
                    "total_cost": summary.get("total_cost", 0.0),
                    "final_equity": summary.get("final_equity", float("nan")),
                    "sharpe": metrics.get("sharpe", float("nan")),
                    "max_drawdown": metrics.get("max_drawdown", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def pick_best(table: pd.DataFrame, objective: str = "excess_mean") -> dict:
    """按目标指标直接取最优（持仓数只作为参考列展示，不设门槛）。"""
    if table.empty:
        return {}
    best = table.sort_values(objective, ascending=False).iloc[0]
    return {"row": best.to_dict(), "objective": objective}


def format_sweep(table: pd.DataFrame, config: SweepConfig | None = None, best: Mapping[str, Any] | None = None) -> str:
    config = config or SweepConfig()
    lines = ["# 参数扫描报告", ""]
    if table.empty:
        lines.append("没有可用的扫描结果。")
        return "\n".join(lines)

    lines.append(
        f"排序目标：主持有期 {config.main_horizon} 日的**超额均值**最大（持仓数只作参考，不设门槛）。"
        f"（同期间基准；未扣滑点外的其他成本）"
    )
    lines.append("")
    view = table.copy()
    for col in ("excess_mean", "excess_win_rate", "trade_win_rate", "trade_excess", "max_drawdown"):
        view[col] = (view[col].astype(float) * 100).round(2)
    view = view.rename(
        columns={
            "params": "参数", "universe": "票池", "signals": "信号数", "excess_mean": "超额均值%",
            "excess_win_rate": "超额胜率%", "trades": "交易数", "trade_win_rate": "交易胜率%",
            "trade_excess": "单笔超额%", "windows_beating": "跑赢窗口", "avg_positions": "平均持仓",
            "total_cost": "累计成本", "final_equity": "期末净值", "sharpe": "Sharpe",
            "max_drawdown": "最大回撤%",
        }
    )
    lines.append(markdown_table(view))
    lines.append("")

    if best:
        row = best.get("row", {})
        lines.append(
            f"**建议配置**：`{row.get('params')}` @ {row.get('universe')} —— "
            f"超额均值 {row.get('excess_mean', float('nan')) * 100:.2f}%，"
            f"交易 {row.get('trades')} 笔，单笔超额 {row.get('trade_excess', float('nan')) * 100:.2f}%，"
            f"平均持仓 {row.get('avg_positions')} 只（仅供参考）。"
        )
    return "\n".join(lines)


def write_sweep(outdir: str | Path, table: pd.DataFrame, meta: Mapping[str, Any] | None = None) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sweep.csv"
    json_path = out / "sweep.json"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(dict(meta or {}), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
