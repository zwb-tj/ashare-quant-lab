"""规则证据研究（event study）——把"规则有效"从声称变成可复核的统计。

做法很朴素、也因此可信：

1. 在票池上逐票扫描某条规则，记录它**触发过哪些 bar**；
2. 对每个触发点计算**未来 N 日收益**（1/3/5/10 日，close-to-close，纯因果）；
3. 与"全样本bar"的基线对比，输出信号数、均值、中位数、胜率、超额。

这样一条规则值不值得用，看表就有结论；同时它天然会暴露"看起来能选股、其实只是跟涨"
的规则（均值与基线差不多）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.rules import build_rule
from aqlab.tables import markdown_table

__all__ = [
    "forward_returns",
    "baseline_stats",
    "rule_event_study",
    "study_profile",
    "format_study",
    "write_study",
]

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)


def forward_returns(df: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """未来 N 日收益（小数）：``fwd_h[t] = close[t+h] / close[t] - 1``。

    只使用 t 及之后的价格，不含未来信息（t+h 不存在时为 NaN，天然剔除尾部样本）。
    """
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        if h < 1:
            raise ValueError("horizons must be >= 1")
        out[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return out


def _stats(returns: pd.Series) -> dict:
    clean = returns.dropna()
    if clean.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win_rate": np.nan, "p25": np.nan, "p75": np.nan}
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "win_rate": float((clean > 0).mean()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def baseline_stats(universe: Mapping[str, pd.DataFrame], horizons: Sequence[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """基线：票池中所有 bar 的未来收益分布（用来判断规则是否真有超额）。"""
    rows = []
    for symbol, df in universe.items():
        fwd = forward_returns(df, horizons)
        for h in horizons:
            stat = _stats(fwd[f"fwd_{h}"])
            stat.update({"symbol": symbol, "horizon": h})
            rows.append(stat)
    frame = pd.DataFrame(rows)
    pooled = []
    for h in horizons:
        all_returns = pd.concat([forward_returns(df, horizons)[f"fwd_{h}"] for df in universe.values()])
        stat = _stats(all_returns)
        stat.update({"horizon": h, "symbol": "ALL"})
        pooled.append(stat)
    return pd.DataFrame(pooled)


def rule_event_study(
    universe: Mapping[str, pd.DataFrame],
    rule_name: str,
    rule: Any | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_history: int = 60,
) -> pd.DataFrame:
    """单条规则在所有票上的事件研究（逐 horizon 一行）。"""
    rule = rule or build_rule(rule_name)
    pooled: dict[int, list[pd.Series]] = {h: [] for h in horizons}
    signals_total = 0
    symbols_with_signal = 0

    for _symbol, df in universe.items():
        if len(df) < min_history:
            continue
        score = rule.score(df)
        triggered = score.fillna(0.0) > 0
        if not bool(triggered.any()):
            continue
        symbols_with_signal += 1
        signals_total += int(triggered.sum())
        fwd = forward_returns(df, horizons)
        for h in horizons:
            pooled[h].append(fwd.loc[triggered, f"fwd_{h}"])

    rows = []
    for h in horizons:
        combined = pd.concat(pooled[h]) if pooled[h] else pd.Series(dtype=float)
        stat = _stats(combined)
        stat.update({"rule": rule_name, "horizon": h, "signals": signals_total, "symbols": symbols_with_signal})
        rows.append(stat)
    return pd.DataFrame(rows)


def study_profile(
    universe: Mapping[str, pd.DataFrame],
    bindings: Sequence[tuple[str, Mapping[str, Any], float]],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_history: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对一个规则档案（profile）逐规则做事件研究，并给出基线表。

    Returns ``(study_table, baseline_table)``。
    """
    if not bindings:
        raise ValueError("bindings must not be empty")
    frames = [rule_event_study(universe, name, rule=build_rule(name, **dict(params)), horizons=horizons, min_history=min_history) for name, params, _w in bindings]
    table = pd.concat(frames, ignore_index=True)
    baseline = baseline_stats(universe, horizons)
    merged = table.merge(
        baseline[["horizon", "mean", "median", "win_rate", "n"]].rename(
            columns={"mean": "base_mean", "median": "base_median", "win_rate": "base_win_rate", "n": "base_n"}
        ),
        on="horizon",
        how="left",
    )
    merged["excess_mean"] = merged["mean"] - merged["base_mean"]
    merged["excess_win_rate"] = merged["win_rate"] - merged["base_win_rate"]
    return merged, baseline


def format_study(table: pd.DataFrame) -> str:
    """把事件研究结果渲染成 markdown（百分比列已换算）。"""
    view = table.copy()
    for col in ("mean", "median", "win_rate", "p25", "p75", "base_mean", "base_median", "base_win_rate", "excess_mean", "excess_win_rate"):
        if col in view.columns:
            view[col] = (view[col].astype(float) * 100).round(2)
    ordered = [
        "rule", "horizon", "signals", "symbols", "n",
        "mean", "median", "win_rate", "p25", "p75",
        "base_mean", "base_win_rate", "excess_mean", "excess_win_rate",
    ]
    view = view[[c for c in ordered if c in view.columns]]
    view = view.rename(
        columns={
            "rule": "规则", "horizon": "持有日", "signals": "信号数", "symbols": "涉及标的", "n": "样本数",
            "mean": "均值%", "median": "中位数%", "win_rate": "胜率%", "p25": "P25%", "p75": "P75%",
            "base_mean": "基线均值%", "base_win_rate": "基线胜率%",
            "excess_mean": "超额均值%", "excess_win_rate": "超额胜率%",
        }
    )
    header = "### 规则事件研究（未来 N 日收益，% ）\n"
    note = (
        "\n> 读法：`超额均值` 与 `超额胜率` 才是规则的价值所在——与「票池所有 bar」的基线比较；"
        "信号数太少（<20）时结论不稳定，不要当真。\n"
    )
    return header + markdown_table(view) + note


def write_study(outdir: str | Path, table: pd.DataFrame, baseline: pd.DataFrame, meta: Mapping[str, Any] | None = None) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "study.md"
    csv_path = out / "study.csv"
    base_path = out / "baseline.csv"
    meta_path = out / "study.json"
    md_path.write_text(format_study(table), encoding="utf-8")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    baseline.to_csv(base_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps(dict(meta or {}), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": md_path, "csv": csv_path, "baseline": base_path, "meta": meta_path}
