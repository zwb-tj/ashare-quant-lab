"""滚动窗口校验（walk-forward）：把"规则 → 入场 → 离场"整条链放进时间窗口验证。

与 :mod:`aqlab.study` 的区别：

* ``study`` 看的是"信号之后的固定持有期收益"，回答"这个信号有没有信息量"；
* ``walkforward`` 把信号交给 :mod:`aqlab.position` 的离场规则跑完整交易，再按时间窗口
  汇总，回答"**在真实持仓管理下**，这套规则在不同阶段赚不赚钱、稳不稳定"。

设计上的两条硬规矩：

1. **窗口内选信号**：一笔交易的信号日必须落在该窗口内（越界一律不计），因此不存在
   "用后面的行情解释前面的表现"；
2. **基准同窗口**：每个窗口都给出票池等权买入持有的收益作为对照，超额才是本事。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.position import PositionConfig, simulate_signals
from aqlab.rules import build_rule
from aqlab.tables import markdown_table

__all__ = [
    "WalkForwardConfig",
    "benchmark_return",
    "composite_scores",
    "format_walkforward",
    "walk_forward",
    "window_bounds",
    "write_walkforward",
]


@dataclass
class WalkForwardConfig:
    """窗口与执行参数。"""

    test_days: int = 60
    step_days: int = 60
    min_history: int = 120
    use_position: bool = True
    horizon: int = 5                      # use_position=False 时的固定持有期
    position_config: PositionConfig | None = None

    def __post_init__(self) -> None:
        if self.test_days < 5:
            raise ValueError("test_days must be >= 5")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")
        if self.min_history < 20:
            raise ValueError("min_history must be >= 20")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def composite_scores(
    universe: Mapping[str, pd.DataFrame],
    bindings: Sequence[tuple[str, Mapping[str, Any], float]],
) -> dict[str, pd.Series]:
    """逐票计算加权综合分（0..1），作为入场信号来源。"""
    if not bindings:
        raise ValueError("bindings must not be empty")
    rules = [(build_rule(name, **dict(params)), weight) for name, params, weight in bindings]
    total_weight = sum(weight for _rule, weight in rules)
    if total_weight <= 0:
        raise ValueError("rule weights must sum to a positive number")

    out: dict[str, pd.Series] = {}
    for symbol, df in universe.items():
        score = pd.Series(0.0, index=df.index)
        for rule, weight in rules:
            score = score + weight * rule.score(df).reindex(df.index).fillna(0.0)
        out[symbol] = (score / total_weight).clip(0.0, 1.0)
    return out


def window_bounds(index: pd.DatetimeIndex, test_days: int, step_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """把时间轴切成若干测试窗口（左闭右开）。"""
    if len(index) < test_days:
        return []
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = 0
    while start + test_days <= len(index):
        bounds.append((index[start], index[start + test_days - 1]))
        start += step_days
    return bounds


def benchmark_return(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """窗口内买入持有收益（用窗口内首尾收盘价）。"""
    window = df.loc[start:end]
    if len(window) < 2:
        return 0.0
    return float(window["close"].iloc[-1] / window["close"].iloc[0] - 1.0)


def _trade_rows(
    symbol: str,
    df: pd.DataFrame,
    signal: pd.Series,
    config: WalkForwardConfig,
) -> list[dict]:
    """产生逐笔交易，并为每笔附加"同持有期买入持有"基准（苹果比苹果）。"""
    close = df["close"].astype(float)
    rows: list[dict] = []

    def same_period_bench(entry_date: str, exit_date: str) -> float:
        window = close.loc[pd.Timestamp(entry_date) : pd.Timestamp(exit_date)]
        if len(window) < 2:
            return 0.0
        return float(window.iloc[-1] / window.iloc[0] - 1.0)

    if config.use_position:
        trades = simulate_signals(df, signal, config=config.position_config)
        if trades.empty:
            return []
        for record in trades.to_dict("records"):
            record["symbol"] = symbol
            record["bench_return"] = round(same_period_bench(record["entry_date"], record["exit_date"]), 4)
            record["excess_return"] = round(float(record["return"]) - record["bench_return"], 4)
            rows.append(record)
        return rows

    # 固定持有期：信号次日收盘入场，持有 horizon 根后收盘离场（不依赖离场规则）
    triggered = signal.reindex(df.index).fillna(False).astype(bool)
    for i in np.flatnonzero(triggered.to_numpy()):
        entry_bar = int(i) + 1
        exit_bar = entry_bar + config.horizon
        if exit_bar >= len(df):
            continue
        entry_price = float(close.iloc[entry_bar])
        exit_price = float(close.iloc[exit_bar])
        trade_return = exit_price / entry_price - 1.0
        bench = same_period_bench(str(df.index[entry_bar].date()), str(df.index[exit_bar].date()))
        rows.append(
            {
                "symbol": symbol,
                "signal_date": str(df.index[int(i)].date()),
                "entry_date": str(df.index[entry_bar].date()),
                "entry_price": round(entry_price, 3),
                "exit_date": str(df.index[exit_bar].date()),
                "exit_price": round(exit_price, 3),
                "exit_reason": f"fixed_{config.horizon}d",
                "return": round(trade_return, 4),
                "bars_held": config.horizon,
                "bench_return": round(bench, 4),
                "excess_return": round(trade_return - bench, 4),
            }
        )
    return rows


def walk_forward(
    universe: Mapping[str, pd.DataFrame],
    bindings: Sequence[tuple[str, Mapping[str, Any], float]],
    config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    """滚动窗口校验，返回 ``{"windows": DataFrame, "trades": DataFrame, "summary": dict}``。"""
    config = config or WalkForwardConfig()
    if not universe:
        raise ValueError("universe is empty")

    scores = composite_scores(universe, bindings)
    all_dates = sorted({date for df in universe.values() for date in df.index})
    bounds = window_bounds(pd.DatetimeIndex(all_dates), config.test_days, config.step_days)
    if not bounds:
        return {"windows": pd.DataFrame(), "trades": pd.DataFrame(), "summary": {"windows": 0, "trades": 0}}

    trades: list[dict] = []
    for symbol, df in universe.items():
        if len(df) < config.min_history:
            continue
        signal = scores[symbol] > 0
        trades.extend(_trade_rows(symbol, df, signal, config))
    trades_df = pd.DataFrame(trades)

    rows: list[dict] = []
    for i, (start, end) in enumerate(bounds, start=1):
        window_trades = trades_df
        if not trades_df.empty:
            signal_dates = pd.to_datetime(trades_df["signal_date"])
            window_trades = trades_df[(signal_dates >= start) & (signal_dates <= end)]
        returns = window_trades["return"].astype(float) if not window_trades.empty else pd.Series(dtype=float)
        benchmarks = [benchmark_return(df, start, end) for df in universe.values() if len(df.loc[start:end]) >= 2]
        window_bench = float(np.mean(benchmarks)) if benchmarks else 0.0
        # 逐笔的"同持有期买入持有"基准：这才是与单笔收益可比的对照
        bench_returns = window_trades["bench_return"].astype(float) if "bench_return" in window_trades else pd.Series(dtype=float)
        excess = window_trades["excess_return"].astype(float) if "excess_return" in window_trades else pd.Series(dtype=float)

        gains = float(returns[returns > 0].sum()) if len(returns) else 0.0
        losses = float(returns[returns < 0].sum()) if len(returns) else 0.0
        rows.append(
            {
                "window": i,
                "start": str(pd.Timestamp(start).date()),
                "end": str(pd.Timestamp(end).date()),
                "trades": len(window_trades),
                "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
                "avg_return": float(returns.mean()) if len(returns) else np.nan,
                "median_return": float(returns.median()) if len(returns) else np.nan,
                "profit_factor": (gains / abs(losses)) if losses < 0 else (np.inf if gains > 0 else np.nan),
                "avg_bars_held": float(window_trades["bars_held"].mean()) if not window_trades.empty else np.nan,
                "top_exit_reason": window_trades["exit_reason"].mode().iloc[0] if not window_trades.empty else "",
                "avg_bench_return": float(bench_returns.mean()) if len(bench_returns) else np.nan,
                "avg_excess_return": float(excess.mean()) if len(excess) else np.nan,
                "window_buy_hold": window_bench,
            }
        )

    windows_df = pd.DataFrame(rows)
    valid = windows_df[windows_df["trades"] > 0]
    summary = {
        "windows": len(windows_df),
        "windows_with_trades": len(valid),
        "trades": len(trades_df),
        "win_rate": float(valid["win_rate"].mean()) if len(valid) else np.nan,
        "avg_return": float(valid["avg_return"].mean()) if len(valid) else np.nan,
        "avg_bench_return": float(valid["avg_bench_return"].mean()) if len(valid) else np.nan,
        "avg_excess_return": float(valid["avg_excess_return"].mean()) if len(valid) else np.nan,
        "positive_windows": int((valid["avg_return"] > 0).sum()) if len(valid) else 0,
        "windows_beating_benchmark": int((valid["avg_excess_return"] > 0).sum()) if len(valid) else 0,
        "best_window": float(valid["avg_return"].max()) if len(valid) else np.nan,
        "worst_window": float(valid["avg_return"].min()) if len(valid) else np.nan,
    }
    return {"windows": windows_df, "trades": trades_df, "summary": summary}


def format_walkforward(result: Mapping[str, Any]) -> str:
    windows = result.get("windows")
    summary = result.get("summary", {})
    lines = ["# Walk-forward 校验报告", ""]
    if windows is None or windows.empty:
        lines.append("数据不足以切分窗口（检查 --days / --test-days）。")
        return "\n".join(lines)

    view = windows.copy()
    for col in ("win_rate", "avg_return", "median_return", "avg_bench_return", "avg_excess_return", "window_buy_hold"):
        view[col] = (view[col].astype(float) * 100).round(2)
    view["profit_factor"] = view["profit_factor"].astype(float).round(2)
    view["avg_bars_held"] = view["avg_bars_held"].astype(float).round(1)
    view = view.rename(
        columns={
            "window": "窗口", "start": "起", "end": "止", "trades": "交易数", "win_rate": "胜率%",
            "avg_return": "均收益%", "median_return": "中位收益%", "profit_factor": "盈亏比",
            "avg_bars_held": "均持有", "top_exit_reason": "主要离场原因",
            "avg_bench_return": "同期间基准%", "avg_excess_return": "超额%",
            "window_buy_hold": "窗口买入持有%",
        }
    )
    view = view[[c for c in ("窗口", "起", "止", "交易数", "胜率%", "均收益%", "中位收益%", "盈亏比", "均持有", "主要离场原因", "同期间基准%", "超额%", "窗口买入持有%") if c in view.columns]]
    lines.append(markdown_table(view))
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    for key, value in summary.items():
        lines.append(f"| {key} | {round(value, 4) if isinstance(value, float) and value == value else value} |")
    lines.append("")
    lines.append(
        "> 读法：**超额%** = 单笔收益 − 该笔**同持有期**的买入持有收益（苹果比苹果）；"
        "**窗口买入持有%** 只是窗口长度的背景参考，不能直接与单笔收益比较。"
        "交易收益未扣手续费与滑点；窗口数或交易数少时结论不稳定。"
    )
    return "\n".join(lines)


def write_walkforward(outdir: str | Path, result: Mapping[str, Any]) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "walkforward.md"
    windows_path = out / "windows.csv"
    trades_path = out / "trades.csv"
    md_path.write_text(format_walkforward(result), encoding="utf-8")
    result.get("windows", pd.DataFrame()).to_csv(windows_path, index=False, encoding="utf-8-sig")
    result.get("trades", pd.DataFrame()).to_csv(trades_path, index=False, encoding="utf-8-sig")
    return {"markdown": md_path, "windows": windows_path, "trades": trades_path}
