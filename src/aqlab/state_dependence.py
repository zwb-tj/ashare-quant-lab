"""因子的市场状态依赖性（v0.29）：解释"折 4 全员失效"。

上一节发现折 4（测试段 2026-03-27 ~ 2026-06-26）里所有被选中的因子组合都失效了。
诊断显示这段市场的三个特征与其它折明显不同：

    折    等权指数累计   平均日收益   横截面离散度   开波段占比
    折 2      +1.73%      +0.035%      2.59%        28.3%
    折 3      +2.81%      +0.046%      2.80%        33.3%
    折 4      -2.15%      -0.005%      3.08%        78.3%   <- 唯一负收益、离散度最高
    折 5      -0.24%      -0.036%      3.16%        33.9%

所以这一节把"因子的 IC"按市场状态分组统计，回答一个具体问题：
**因子的有效性是否依赖市场状态？依赖哪种状态？**

三个状态变量（都能用等权指数与横截面收益算出，不需要额外数据）：

* ``trend``     —— 等权指数的 20 日动量（趋势方向与强度）；
* ``dispersion``—— 当日横截面收益标准差（分化程度，可理解为"选股空间"）；
* ``regime``    —— 0AMV 活跃市值波段开关（已有的市场状态实现）。

**口径**：先算每个截面日每个因子的 IC（与前面完全同一套口径），再把 IC 序列按当日状态分组，
比较各组的平均 IC 与 t 值。这样"状态依赖性"是可量化的，而不是靠某一段的偶然观察。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from aqlab.alpha101 import build_panel, compute_alphas
from aqlab.walk_forward_folds import ic_series

__all__ = ["StateConfig", "factor_state_ic", "market_state_frame", "summarize_state_dependence"]


@dataclass
class StateConfig:
    horizons: tuple[int, ...] = (5, 20)
    step_days: int = 5
    min_history: int = 260
    min_symbols: int = 200
    trend_window: int = 20
    quantiles: int = 3               # 每个状态变量分几档（3 = 低/中/高）
    factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.min_symbols < 3:
            raise ValueError("min_symbols must be >= 3")
        if self.trend_window < 2:
            raise ValueError("trend_window must be >= 2")
        if self.quantiles < 2:
            raise ValueError("quantiles must be >= 2")


def market_state_frame(
    close: pd.DataFrame,
    trend_window: int = 20,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    """市场状态：等权指数动量、横截面离散度、（可选）0AMV 波段状态。

    * ``index``       —— 等权指数净值（每日横截面收益等权平均后累乘）；
    * ``trend``       —— 等权指数过去 ``trend_window`` 日收益；
    * ``dispersion``  —— 当日横截面收益标准差（分化程度）；
    * ``regime``      —— 传入的波段状态（已滞后一天，避免用当天信息）。
    """
    returns = close.pct_change()
    mean_return = returns.mean(axis=1).fillna(0.0)
    index = (1.0 + mean_return).cumprod()
    out = pd.DataFrame(
        {
            "index": index,
            "trend": index.pct_change(trend_window),
            "dispersion": returns.std(axis=1),
        }
    )
    if regime is not None:
        out["regime"] = regime.reindex(out.index).fillna(0).astype(int)
    return out


def _label_quantiles(values: pd.Series, quantiles: int, prefix: str) -> pd.Series:
    """把连续状态变量分成低/中/高若干档（等频），分不出档时全部标为中。"""
    clean = values.dropna()
    if clean.nunique() < quantiles:
        return pd.Series(f"{prefix}中", index=values.index)
    try:
        labels = pd.qcut(values, quantiles, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(f"{prefix}中", index=values.index)
    names = {0: "低", 1: "中", 2: "高"} if quantiles == 3 else {i: str(i) for i in range(quantiles)}
    return labels.map(lambda value: f"{prefix}{names.get(int(value), value)}" if pd.notna(value) else np.nan)


def factor_state_ic(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    config: StateConfig | None = None,
    regime: pd.Series | None = None,
) -> dict:
    """每个因子的 IC 序列 + 对应的市场状态标签，用于分组比较。"""
    config = config or StateConfig()
    panel = build_panel(daily_by_symbol)
    values = compute_alphas(panel, names=list(config.factors) or None, min_history=config.min_history)
    if not values:
        raise ValueError("no factor could be computed")

    state = market_state_frame(panel.close, trend_window=config.trend_window, regime=regime)
    state["trend_bucket"] = _label_quantiles(state["trend"], config.quantiles, "动量")
    state["dispersion_bucket"] = _label_quantiles(state["dispersion"], config.quantiles, "离散")
    if "regime" in state.columns:
        state["regime_bucket"] = state["regime"].map({1: "开波段", 0: "关波段"})

    sampled = list(panel.dates[:: config.step_days])
    usable = [date for date in sampled if any(date in frame.index and bool(frame.loc[date].notna().any()) for frame in values.values())]

    rows: list[dict] = []
    for name, frame in values.items():
        for horizon in config.horizons:
            series = ic_series(frame, panel.close, horizon, usable, config.min_symbols)
            if series.empty:
                continue
            joined = pd.DataFrame({"ic": series}).join(state, how="left")
            for record in joined.reset_index(names="date").to_dict("records"):
                rows.append({"factor": name, "horizon": horizon, **record})
    detail = pd.DataFrame(rows)
    return {"detail": detail, "state": state, "panel": panel, "config": config}


def summarize_state_dependence(detail: pd.DataFrame, state_column: str = "trend_bucket") -> pd.DataFrame:
    """按状态分组汇总 IC：每组的天数、平均 IC、t 值、正 IC 占比。

    这里对**每个 (因子, 持有期, 状态档)** 单独算 t 值（组内 IC 序列的均值 / 标准误），
    并给出组间的极差（最高组 - 最低组），用来衡量"状态依赖性有多强"。
    """
    if detail is None or detail.empty or state_column not in detail.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (factor, horizon, bucket), group in detail.dropna(subset=[state_column]).groupby(["factor", "horizon", state_column]):
        values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        t_stat = mean / std * np.sqrt(len(values)) if std and np.isfinite(std) and std > 0 else np.nan
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "state": str(bucket),
                "periods": len(values),
                "ic_mean": mean,
                "t_stat": t_stat,
                "positive_rate": float((values > 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # 每个 (因子, 持有期) 的组间极差与"最好/最差状态"
    spread = (
        table.groupby(["factor", "horizon"])["ic_mean"]
        .agg(["min", "max"])
        .assign(ic_spread=lambda frame: frame["max"] - frame["min"])
        .reset_index()
    )
    return table.merge(spread[["factor", "horizon", "ic_spread"]], on=["factor", "horizon"], how="left")
