"""因子组合方案回测（v0.18）：把"因子 IC 为负"这个发现落成策略，并做样本外检验。

上一节的研究结论是"这段样本里动量类因子 IC 为负、分位收益单调递减"。那么自然的问题是：
**按 IC 的符号/大小动态定权，能不能比写死的权重更好？** 这个模块回答它，并把"研究结论"变成
可回测的方案。

口径（每条都是防自欺的）：

* **权重只用过去的 IC**：在 ``t`` 日定权时，只用**严格早于 t** 的截面 IC（``lag_periods`` ≥ 1），
  再用最近 ``lookback_periods`` 个截面的均值；有测试断言"把最后一期 IC 改掉不会影响当期权重"。
* **选股只用 t 日及之前的行情**：因子值来自因果指标（与 :mod:`aqlab.factor_ic` 同一套预计算）。
* **换手成本显式**：每次调仓按双边 ``round_trip_cost_bps`` 扣一次（默认 0.2%），不假装零成本。
* **基准是同期等权全市场**，所以看的始终是**超额**，而不是"赚了钱就算赢"。

方案：
    ``fixed``        —— 项目默认权重（动量方向）
    ``sign_flip``    —— 默认权重取反（纯反转）
    ``ic_sign``      —— 每个因子权重 = 过去 IC 均值的**符号**
    ``ic_weight``    —— 每个因子权重 ∝ 过去 IC 均值（再归一化）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.factor_ic import ICConfig, factor_ic_panel, precompute_factors
from aqlab.screen import DEFAULT_WEIGHTS

__all__ = ["SchemeConfig", "run_weight_schemes", "scheme_weights", "summarize_schemes"]

SCHEMES = ("fixed", "sign_flip", "ic_sign", "ic_weight")


@dataclass
class SchemeConfig:
    top_n: int = 10
    forward_days: int = 20
    step_days: int = 5
    min_history: int = 130
    min_symbols: int = 30
    lookback_periods: int = 12          # 用最近多少个截面的 IC 定权
    lag_periods: int = 1               # 权重至少落后多少个截面（1 = 只用严格过去的 IC）
    round_trip_cost_bps: float = 20.0  # 每次调仓的双边成本（0.2%）
    factors: tuple[str, ...] = ("mom_20", "mom_60", "trend_gap", "vol_20", "rsi_14")
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.forward_days < 1 or self.step_days < 1:
            raise ValueError("forward_days and step_days must be >= 1")
        if self.lookback_periods < 1:
            raise ValueError("lookback_periods must be >= 1")
        if self.lag_periods < 0:
            raise ValueError("lag_periods must be >= 0")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be >= 0")

    def ic_config(self) -> ICConfig:
        return ICConfig(
            forward_days=self.forward_days,
            step_days=self.step_days,
            min_history=self.min_history,
            min_symbols=self.min_symbols,
            factors=self.factors,
            start=self.start,
            end=self.end,
        )


def scheme_weights(ic_history: Mapping[str, float], scheme: str, factors: Sequence[str]) -> dict[str, float]:
    """根据**过去的** IC 均值给出权重向量。

    ``ic_history`` 是"因子 -> 过去 IC 均值"，只允许包含严格早于当前截面的信息。
    """
    if scheme == "fixed":
        return {factor: float(DEFAULT_WEIGHTS.get(factor, 0.0)) for factor in factors}
    if scheme == "sign_flip":
        return {factor: -float(DEFAULT_WEIGHTS.get(factor, 0.0)) for factor in factors}
    if scheme in ("ic_sign", "ic_weight"):
        signs = {factor: float(np.sign(ic_history.get(factor, 0.0))) for factor in factors}
        if scheme == "ic_sign":
            return signs
        # 权重正比于 IC 本身（正 IC 正权重、负 IC 负权重），再按绝对值之和归一化
        raw = {factor: float(ic_history.get(factor, 0.0)) for factor in factors}
        total = sum(abs(value) for value in raw.values())
        if total <= 0:
            return dict.fromkeys(factors, 0.0)
        return {factor: value / total for factor, value in raw.items()}
    raise ValueError(f"unknown scheme '{scheme}'; available: {SCHEMES}")


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _forward_return(close: pd.Series, as_of: pd.Timestamp, horizon: int) -> float:
    if as_of not in close.index:
        return np.nan
    position = close.index.get_loc(as_of)
    target = position + horizon
    if target >= len(close):
        return np.nan
    entry = float(close.iloc[position])
    exit_price = float(close.iloc[target])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price):
        return np.nan
    return exit_price / entry - 1.0


def trailing_ic(wide: pd.DataFrame, index: int, lookback: int, lag: int) -> dict[str, float]:
    """取第 ``index`` 个截面**定权时允许看到**的过去 IC 均值。

    ``lag=1`` 表示只用严格早于当期的截面（``0 .. index-1``）；``lag=0`` 才会把当期 IC 算进去
    （那是作弊口径，只在测试里用来对照）。
    """
    cutoff = index - lag + 1
    history = wide.iloc[: max(cutoff, 0)].tail(lookback)
    return {factor: float(history[factor].mean()) for factor in history.columns if history[factor].notna().any()}


def run_weight_schemes(
    universe: Mapping[str, pd.DataFrame],
    config: SchemeConfig | None = None,
    schemes: Sequence[str] = SCHEMES,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """跑多个定权方案，返回 ``(逐期明细, 方案 -> 累计净值)``。"""
    config = config or SchemeConfig()
    ic_config = config.ic_config()
    precomputed = precompute_factors(universe, ic_config)
    panel = factor_ic_panel(universe, ic_config)
    if panel.empty:
        raise ValueError("no cross-section is usable for the scheme backtest")

    # 逐截面 IC 宽表：date × factor
    wide = panel.pivot(index="date", columns="factor", values="ic").sort_index()
    dates = list(wide.index)
    cost = config.round_trip_cost_bps / 10_000.0

    records: list[dict] = []
    for index, as_of in enumerate(dates):
        ic_history = trailing_ic(wide, index, config.lookback_periods, config.lag_periods)

        # 当期截面
        rows: list[dict] = []
        for symbol, table in precomputed.items():
            if as_of not in table.index:
                continue
            position = table.index.get_loc(as_of)
            if position + 1 < config.min_history:
                continue
            row = table.iloc[position].to_dict()
            row["symbol"] = symbol
            rows.append(row)
        if len(rows) < config.min_symbols:
            continue
        cross = pd.DataFrame(rows)
        cross["forward"] = [
            _forward_return(precomputed[symbol]["close"], as_of, config.forward_days) for symbol in cross["symbol"]
        ]
        benchmark = float(cross["forward"].mean()) if cross["forward"].notna().any() else np.nan

        for scheme in schemes:
            weights = scheme_weights(ic_history, scheme, config.factors)
            if all(value == 0 for value in weights.values()):
                continue
            score = pd.Series(0.0, index=cross.index)
            for factor, weight in weights.items():
                if factor in cross.columns and weight != 0:
                    score = score + weight * _zscore(cross[factor].fillna(cross[factor].median()))
            picks = cross.assign(score=score).sort_values(["score", "symbol"], ascending=[False, True]).head(config.top_n)
            gross = float(picks["forward"].mean())
            if not np.isfinite(gross):
                continue
            net = gross - cost
            records.append(
                {
                    "date": pd.Timestamp(as_of),
                    "scheme": scheme,
                    "gross_return": gross,
                    "net_return": net,
                    "benchmark": benchmark,
                    "excess_net": net - benchmark if np.isfinite(benchmark) else np.nan,
                    "picks": len(picks),
                }
            )

    detail = pd.DataFrame(records)
    curves: dict[str, pd.Series] = {}
    for scheme, group in detail.groupby("scheme"):
        ordered = group.sort_values("date")
        curves[scheme] = (1.0 + ordered["net_return"]).cumprod().set_axis(ordered["date"]).rename(scheme)
    benchmark_curve = None
    if not detail.empty:
        first = detail.groupby("scheme").get_group(schemes[0]).sort_values("date")
        benchmark_curve = (1.0 + first["benchmark"]).cumprod().set_axis(first["date"]).rename("benchmark")
        curves["benchmark"] = benchmark_curve
    return detail, curves


def summarize_schemes(detail: pd.DataFrame) -> pd.DataFrame:
    """按方案汇总：期数、平均净收益、胜率、平均超额、超额 t 值。"""
    if detail is None or detail.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for scheme, group in detail.groupby("scheme"):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        excess = pd.to_numeric(group["excess_net"], errors="coerce").dropna()
        t_stat = np.nan
        if len(excess) > 2 and excess.std(ddof=1) > 0:
            t_stat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
        rows.append(
            {
                "scheme": scheme,
                "periods": len(group),
                "mean_net": float(returns.mean()) if len(returns) else np.nan,
                "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
                "mean_excess": float(excess.mean()) if len(excess) else np.nan,
                "excess_t": t_stat,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_excess", ascending=False).reset_index(drop=True)
