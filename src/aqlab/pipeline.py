"""Daily pipeline: fetch → score with rules → rank → report → notify.

This is the piece that turns the library into an actual daily routine:

```
17:30  数据源（tushare / 本地 CSV / 合成）
         │
         ├─ 活跃市值开关（可选，滞回开关）
         │
         ├─ 逐票逐规则打分 ──► 加权综合分 ──► 横截面排名
         │
         ├─ 报告（markdown + json，落盘 output/daily/）
         │
         └─ 推送（飞书 webhook / 控制台 dry-run）
```

**开关语义**：市场开关关闭时，管线仍然输出"观察名单"，但会在报告的 `gate_state=0`
与 notes 中明确写出"开关关闭，不产生新买点"——研究和推送都不隐瞒这件事。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from aqlab.notify import ConsoleNotifier, Notifier
from aqlab.rules import DEFAULT_RULE_BINDINGS, ActivityValueGate, RuleBinding, build_rule

__all__ = ["DailyConfig", "DailyPipeline", "DailyReport", "write_daily_report"]


@dataclass
class DailyConfig:
    top_n: int = 10
    min_history: int = 60
    as_of: str | None = None
    use_gate: bool = True
    require_signal: bool = False  # if True, only symbols with at least one triggered rule are listed


@dataclass
class DailyReport:
    as_of: str
    universe_size: int
    eligible_size: int
    gate_state: int
    weights: dict
    picks: list[dict]
    gate_trigger: str | None = None
    notes: list[str] = field(default_factory=list)
    notifier_result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "universe_size": self.universe_size,
            "eligible_size": self.eligible_size,
            "gate_state": self.gate_state,
            "gate_trigger": self.gate_trigger,
            "weights": self.weights,
            "notes": self.notes,
            "picks": self.picks,
            "notifier_result": self.notifier_result,
        }

    def summary_lines(self) -> list[str]:
        gate_text = "🟢 开关打开" if self.gate_state else "🔴 开关关闭（不产生新买点）"
        if self.gate_trigger:
            gate_text += f"（{self.gate_trigger}）"
        lines = [
            f"**{self.as_of}** ｜ 票池 {self.universe_size} 只（可用 {self.eligible_size} 只）｜ {gate_text}",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        return lines


class DailyPipeline:
    """Runs the end-to-end daily job against any :class:`~aqlab.tools.DataSource`."""

    def __init__(
        self,
        source,
        rule_bindings: Sequence[tuple[str, Mapping[str, Any], float]] | None = None,
        gate: ActivityValueGate | None = None,
        config: DailyConfig | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.source = source
        self.config = config or DailyConfig()
        bindings = rule_bindings if rule_bindings is not None else DEFAULT_RULE_BINDINGS
        self.bindings: list[RuleBinding] = [
            RuleBinding(rule=build_rule(name, **dict(params)), weight=weight) for name, params, weight in bindings
        ]
        if not self.bindings:
            raise ValueError("at least one rule binding is required")
        self.gate = gate
        self.notifier = notifier or ConsoleNotifier()

    # -- helpers -------------------------------------------------------------------
    @property
    def weights(self) -> dict:
        total = sum(b.weight for b in self.bindings)
        return {b.rule.name: round(b.weight / total, 4) for b in self.bindings}

    def _score_symbol(self, df: pd.DataFrame) -> tuple[dict, dict]:
        rule_scores: dict[str, float] = {}
        triggered: list[str] = []
        composite = 0.0
        total_weight = sum(b.weight for b in self.bindings)
        for binding in self.bindings:
            series = binding.rule.score(df)
            value = float(series.iloc[-1]) if len(series) else 0.0
            rule_scores[binding.rule.name] = round(value, 4)
            composite += binding.weight * value
            if value > 0:
                triggered.append(binding.rule.name)
        return rule_scores, {"composite": composite / total_weight, "triggered": triggered}

    # -- public API ----------------------------------------------------------------
    def run(self, push: bool = True) -> DailyReport:
        symbols = self.source.symbols()
        universe = {sym: self.source.bars(sym) for sym in symbols}
        as_of = pd.Timestamp(self.config.as_of) if self.config.as_of else None

        notes: list[str] = []
        gate_state = 1
        gate_trigger: str | None = None
        if self.gate is not None and self.config.use_gate:
            gate_state = self.gate.state_at(universe, as_of=as_of)
            if hasattr(self.gate, "trigger_at"):
                gate_trigger = self.gate.trigger_at(universe, as_of=as_of)
            for note in getattr(self.gate, "notes", [])[:2]:
                notes.append(f"开关数据口径：{note}")
            if gate_state == 0:
                notes.append("活跃市值开关关闭：本期不产生新买点，以下仅为观察名单。")

        rows: list[dict] = []
        eligible = 0
        for symbol, df in universe.items():
            hist = df.loc[:as_of] if as_of is not None else df
            if len(hist) < self.config.min_history:
                continue
            eligible += 1
            rule_scores, agg = self._score_symbol(hist)
            if self.config.require_signal and not agg["triggered"]:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "close": round(float(hist["close"].iloc[-1]), 2),
                    "score": round(agg["composite"] * 100, 2),
                    "rules": agg["triggered"],
                    "rule_scores": rule_scores,
                    "last_date": str(hist.index[-1].date()),
                }
            )

        rows.sort(key=lambda item: (-item["score"], item["symbol"]))
        positive = [row for row in rows if row["score"] > 0]
        if positive:
            selected = positive[: self.config.top_n]
            if len(positive) < self.config.top_n:
                notes.append(f"本期仅 {len(positive)} 只标的触发规则（不足 top_n={self.config.top_n}）。")
        elif rows:
            selected = rows[: self.config.top_n]
            notes.append("本期没有标的触发规则，以下为按综合分排序的观察名单。")
        else:
            selected = []
            notes.append("没有可用标的（历史长度不足或没有触发信号）。")

        for i, row in enumerate(selected, start=1):
            row["rank"] = i
        picks = selected

        report = DailyReport(
            as_of=str(as_of.date()) if as_of is not None else str(max(df.index[-1] for df in universe.values()).date()) if universe else "",
            universe_size=len(symbols),
            eligible_size=eligible,
            gate_state=gate_state,
            gate_trigger=gate_trigger,
            weights=self.weights,
            picks=picks,
            notes=notes,
        )

        if push:
            title = f"aqlab 每日选股 · {report.as_of}"
            result = self.notifier.send(title, report.summary_lines(), picks)
            report.notifier_result = {"ok": result.ok, "channel": result.channel, "detail": result.detail}
        return report


def write_daily_report(outdir: str | Path, report: DailyReport) -> dict[str, Path]:
    """Persist the daily report (markdown + json)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"daily-{report.as_of}.md"
    json_path = out / f"daily-{report.as_of}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def render_markdown(report: DailyReport) -> str:
    lines = [f"# aqlab 每日选股 · {report.as_of}", ""]
    lines.extend(report.summary_lines())
    lines.append("")
    lines.append(f"规则权重：{report.weights}")
    lines.append("")
    if report.picks:
        lines.append("| 排名 | 代码 | 收盘 | 综合分 | 触发规则 | " + " | ".join(report.weights) + " |")
        lines.append("| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in report.weights) + " |")
        for item in report.picks:
            per_rule = " | ".join(str(item["rule_scores"].get(name, 0)) for name in report.weights)
            lines.append(
                f"| {item['rank']} | {item['symbol']} | {item['close']} | {item['score']} | "
                f"{'、'.join(item['rules']) or '-'} | {per_rule} |"
            )
    else:
        lines.append("（本期无标的）")
    lines.append("")
    lines.append("> 仅用于量化研究，不构成投资建议。")
    return "\n".join(lines)
