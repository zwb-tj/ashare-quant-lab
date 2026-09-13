"""公式因子的组合回测（v0.27）：把"IC 显著"换算成**扣成本后**的可交易超额。

上一节的结论是"某些公式因子的 IC 在样本外仍然显著"。但 IC 回答的是**整个截面的秩相关**，
不等于**买了能赚**。这个模块补上那一跳，并且刻意把三件事做对：

1. **可成交价**：IC 用收盘价对收盘价算；组合必须用**次日开盘**入场（T 日收盘出信号 →
   T+1 开盘买）。用当日收盘价成交是最常见的隐性未来函数。
2. **换手成本**：按相邻两期持仓的**实际更替比例**收费（``round_trip_cost_bps`` 为整体换手的成本）。
   不做"每期都按全额换手收费"的粗略假设，也不假装零成本。
3. **方向不能事后决定**：因子该买高分位还是低分位，只能用**样本内**的 IC 符号来定，
   然后在**样本外**执行——否则就是拿未来信息选方向。

输出还包含**成本敏感性**与**盈亏平衡成本**：在多高的成本下超额归零。这个数字比"净收益多少"
更能说明一个因子值不值得做。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aqlab.alpha101 import Panel

__all__ = [
    "AlphaPortfolioConfig",
    "break_even_cost_bps",
    "cost_sensitivity",
    "run_alpha_portfolio",
    "summarize_alpha_portfolio",
]


@dataclass
class AlphaPortfolioConfig:
    top_n: int = 20
    hold_days: int = 5                # 持有期（交易日）
    step_days: int | None = None      # 调仓间隔；None = 与持有期相同（不重叠）
    cost_bps: float = 20.0            # 整体换手一次的双边成本（0.2%）
    min_history: int = 260
    direction: str = "ic_sign"        # top | bottom | ic_sign（用样本内 IC 符号决定）
    is_fraction: float = 0.5          # 样本内占比，仅用于 direction="ic_sign"

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.hold_days < 1:
            raise ValueError("hold_days must be >= 1")
        if self.step_days is not None and self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")
        if self.direction not in ("top", "bottom", "ic_sign"):
            raise ValueError("direction must be 'top', 'bottom' or 'ic_sign'")
        if not 0 < self.is_fraction < 1:
            raise ValueError("is_fraction must be in (0, 1)")

    @property
    def effective_step(self) -> int:
        return self.step_days if self.step_days is not None else self.hold_days


def _forward_excess(
    factor: pd.Series,
    open_price: pd.Series,
    close_price: pd.Series,
    top_n: int,
    hold_days: int,
    direction: int,
    min_symbols: int,
) -> tuple[float, float, list[str]] | None:
    """给定截面：返回 ``(组合毛收益, 等权基准收益, 选中标的)``；数据不足返回 None。

    ``direction=+1`` 表示"买因子值最高的一档"，因此排序用 ``ascending = direction < 0``
    （direction=+1 → 降序，最大在前）。

    **同分必须按标的代码兜底排序**：因子取值并列时，若只按因子排序，每期选中的批次会随
    排序稳定性漂移，凭空产生换手与成本——常数因子应当零换手，这一点有测试守着。
    """
    valid = factor.dropna()
    if len(valid) < max(min_symbols, top_n):
        return None
    symbols = [str(name) for name in valid.index]
    ordered = pd.DataFrame({"factor": valid.to_numpy(dtype=float), "symbol": symbols}).sort_values(
        ["factor", "symbol"], ascending=[direction < 0, True]
    )
    picks = ordered["symbol"].head(top_n).tolist()
    entry = open_price.reindex(picks).astype(float)
    exit_price = close_price.reindex(picks).astype(float)
    returns = (exit_price / entry - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return None
    benchmark = (close_price / open_price - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if benchmark.empty:
        return None
    return float(returns.mean()), float(benchmark.mean()), picks


def run_alpha_portfolio(
    factor: pd.DataFrame,
    panel: Panel,
    config: AlphaPortfolioConfig | None = None,
) -> dict:
    """单个因子的组合回测：次日开盘入场、持有 ``hold_days``、按实际换手扣成本。

    返回 ``{"detail": DataFrame, "summary": dict, "direction": int}``。
    """
    config = config or AlphaPortfolioConfig()
    close, open_price = panel.close, panel["open"]
    index = factor.index
    if not index.equals(close.index):
        factor = factor.reindex(close.index)

    # 方向：只允许用**样本内** IC 符号决定（top = 买高分位）
    split = int(len(index) * config.is_fraction)
    direction = 1
    if config.direction == "ic_sign":
        from aqlab.factor_ic import spearman_ic as _ic

        ics: list[float] = []
        for position in range(config.min_history, max(split, config.min_history + 1)):
            target = position + config.hold_days
            if target >= len(index):
                break
            factor_row = factor.iloc[position]
            forward = (close.iloc[target] / open_price.iloc[position] - 1.0).replace([np.inf, -np.inf], np.nan)
            value = _ic(factor_row, forward)
            if value == value:
                ics.append(value)
        direction = 1 if (not ics or np.mean(ics) >= 0) else -1
    elif config.direction == "bottom":
        direction = -1

    positions = list(range(config.min_history, len(index) - config.hold_days, config.effective_step))
    rows: list[dict] = []
    previous: set[str] = set()
    for position in positions:
        as_of = index[position]
        outcome = _forward_excess(
            factor.iloc[position],
            open_price.iloc[position + 1],
            close.iloc[position + config.hold_days],
            config.top_n,
            config.hold_days,
            direction,
            min_symbols=max(config.top_n, 3),
        )
        if outcome is None:
            continue
        gross, benchmark, picks = outcome
        current = set(picks)
        turnover = 1.0 if not previous else 1.0 - len(current & previous) / max(len(current | previous), 1)
        cost = turnover * config.cost_bps / 10_000.0
        rows.append(
            {
                "date": pd.Timestamp(as_of),
                "gross_return": gross,
                "net_return": gross - cost,
                "benchmark": benchmark,
                "gross_excess": gross - benchmark,
                "net_excess": gross - cost - benchmark,
                "turnover": turnover,
                "cost": cost,
                "picks": len(picks),
            }
        )
        previous = current

    detail = pd.DataFrame(rows)
    summary: dict = {"periods": len(detail), "direction": "top" if direction > 0 else "bottom"}
    if not detail.empty:
        summary.update(
            {
                "mean_gross": float(detail["gross_return"].mean()),
                "mean_net": float(detail["net_return"].mean()),
                "mean_benchmark": float(detail["benchmark"].mean()),
                "mean_gross_excess": float(detail["gross_excess"].mean()),
                "mean_net_excess": float(detail["net_excess"].mean()),
                "mean_turnover": float(detail["turnover"].mean()),
                "win_rate": float((detail["net_return"] > 0).mean()),
                "excess_win_rate": float((detail["net_excess"] > 0).mean()),
            }
        )
        excess = detail["net_excess"].dropna()
        t_stat = np.nan
        if len(excess) > 2 and excess.std(ddof=1) > 0:
            t_stat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
        summary["net_excess_t"] = t_stat
    return {"detail": detail, "summary": summary, "direction": direction}


def summarize_alpha_portfolio(detail: pd.DataFrame) -> pd.DataFrame:
    """把逐期明细折成一行指标（便于多个因子横向比较）。"""
    if detail is None or detail.empty:
        return pd.DataFrame()
    excess = detail["net_excess"].dropna()
    gross_excess = detail["gross_excess"].dropna()
    t_stat = np.nan
    if len(excess) > 2 and excess.std(ddof=1) > 0:
        t_stat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
    return pd.DataFrame(
        [
            {
                "periods": len(detail),
                "mean_gross_excess": float(gross_excess.mean()) if len(gross_excess) else np.nan,
                "mean_net_excess": float(excess.mean()) if len(excess) else np.nan,
                "net_excess_t": t_stat,
                "mean_turnover": float(detail["turnover"].mean()),
                "win_rate": float((detail["net_return"] > 0).mean()),
                "excess_win_rate": float((detail["net_excess"] > 0).mean()),
            }
        ]
    )


def cost_sensitivity(
    factor: pd.DataFrame,
    panel: Panel,
    config: AlphaPortfolioConfig | None = None,
    cost_levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0),
) -> pd.DataFrame:
    """同一因子上扫描不同成本水平：毛/净超额随成本怎么变。

    方向只算一次（用样本内 IC 符号 + 零成本那一档），避免每个成本档各选一次方向。
    """
    import dataclasses

    base = config or AlphaPortfolioConfig()
    probe = dataclasses.replace(base, cost_bps=0.0)
    direction = run_alpha_portfolio(factor, panel, probe)["direction"]
    rows: list[dict] = []
    for level in cost_levels:
        outcome = run_alpha_portfolio(factor, panel, dataclasses.replace(base, cost_bps=level, direction="top" if direction > 0 else "bottom"))
        summary = outcome["summary"]
        rows.append(
            {
                "cost_bps": float(level),
                "mean_gross_excess": summary.get("mean_gross_excess", np.nan),
                "mean_net_excess": summary.get("mean_net_excess", np.nan),
                "mean_turnover": summary.get("mean_turnover", np.nan),
                "net_excess_t": summary.get("net_excess_t", np.nan),
            }
        )
    return pd.DataFrame(rows)


def break_even_cost_bps(
    factor: pd.DataFrame,
    panel: Panel,
    config: AlphaPortfolioConfig | None = None,
) -> float:
    """盈亏平衡成本（bps）：净超额归零时的成本水平。

    推导：净超额 = 毛超额 - 换手 × 成本，故 ``成本* = 毛超额 / 换手``（换算成 bps）。
    换手为 0 时无意义，返回 NaN 而不是 inf。
    """
    base = config or AlphaPortfolioConfig()
    import dataclasses

    outcome = run_alpha_portfolio(factor, panel, dataclasses.replace(base, cost_bps=0.0))
    detail = outcome["detail"]
    if detail.empty:
        return float("nan")
    turnover = float(detail["turnover"].mean())
    gross_excess = float(detail["gross_excess"].mean())
    if turnover <= 0:
        return float("nan")
    return gross_excess / turnover * 10_000.0
