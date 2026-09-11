"""Report generation: markdown summary + machine-readable artifacts.

A report always lands as a directory containing:

* ``report.md``   — human-readable summary (metrics, params, trade stats)
* ``metrics.json``— machine-readable metrics
* ``equity.csv``  — per-bar equity / position / turnover
* ``trades.csv``  — trade blotter (may be empty)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from aqlab.backtest import BacktestResult
from aqlab.metrics import format_metrics

__all__ = ["format_report", "write_report"]

DISCLAIMER = (
    "> **免责声明**：本项目仅用于量化研究与工程演示，所有回测均为历史模拟，"
    "不构成任何投资建议。历史表现不代表未来收益。\n"
)


def format_report(
    name: str,
    result: BacktestResult,
    metrics: Mapping[str, Any],
    extras: Mapping[str, Any] | None = None,
) -> str:
    """Build the markdown body of a report."""
    lines = [f"# 回测报告 · {name}", ""]
    lines.append(f"- 策略：`{result.name}`")
    lines.append(f"- 区间：{result.frame.index[0].date()} ~ {result.frame.index[-1].date()}（{len(result.frame)} 个交易日）")
    lines.append(
        f"- 成本假设：手续费 {result.config.fee_bps:g} bps + 滑点 {result.config.slippage_bps:g} bps，"
        f"初始资金 {result.config.initial_cash:,.0f}"
    )
    lines.append("")
    lines.append("## 绩效指标")
    lines.append("")
    lines.append(format_metrics(dict(metrics)))
    lines.append("")

    if extras:
        lines.append("## 参数 / 备注")
        lines.append("")
        for key, value in extras.items():
            lines.append(f"- **{key}**：{value}")
        lines.append("")

    trades = result.trades
    if len(trades):
        closed = trades[~trades["open"].astype(bool)] if "open" in trades else trades
        lines.append("## 交易明细（最近 5 笔）")
        lines.append("")
        head = closed.tail(5).copy()
        for col in ("entry_date", "exit_date"):
            head[col] = pd.to_datetime(head[col]).dt.strftime("%Y-%m-%d")
        for col in ("entry_price", "exit_price", "gross_return", "net_return"):
            head[col] = head[col].astype(float).round(4)
        lines.append(head.to_markdown(index=False))
        lines.append("")
        if "open" in trades and trades["open"].any():
            lines.append(f"（另有 {int(trades['open'].sum())} 笔持仓未平仓，未计入交易统计）")
            lines.append("")
    else:
        lines.append("## 交易明细")
        lines.append("")
        lines.append("本次回测没有产生持仓变化（策略全程空仓或数据不足）。")
        lines.append("")

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def write_report(
    outdir: str | Path,
    name: str,
    result: BacktestResult,
    metrics: Mapping[str, Any],
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist a report bundle and return the written paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "report.md"
    metrics_path = out / "metrics.json"
    equity_path = out / "equity.csv"
    trades_path = out / "trades.csv"

    report_path.write_text(format_report(name, result, metrics, extras), encoding="utf-8")
    serializable = {
        k: (None if isinstance(v, float) and v != v else v) for k, v in metrics.items()
    }
    metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    result.frame.to_csv(equity_path, encoding="utf-8-sig")
    result.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    return {
        "report": report_path,
        "metrics": metrics_path,
        "equity": equity_path,
        "trades": trades_path,
    }
