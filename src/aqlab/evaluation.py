"""Evaluation harness for the LLM research layer.

The point of this module is to make agent reliability **measurable** instead of
vibes-based. It scores four things for every task:

1. **answer rate** — did the agent produce a final answer at all;
2. **tool success rate** — how many tool calls returned ``ok=true``;
3. **grounding / hallucination rate** — every number in the answer must appear in
   the tool outputs; anything else is counted as an ungrounded (hallucinated) number;
4. **abstention accuracy** — on trap tasks (unknown symbol, invalid parameters)
   the correct behaviour is to say "insufficient evidence", not to invent numbers.

An offline mode runs the harness with deterministic ``ScriptedClient`` scripts, so
the machinery itself is testable in CI without any API key. The live mode points
the same harness at a real OpenAI-compatible endpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from aqlab.agent import LLMClient, ResearchAgent, ScriptedClient
from aqlab.tools import ToolRegistry

__all__ = [
    "EvalReport",
    "EvalTask",
    "default_tasks",
    "extract_numbers",
    "format_eval_report",
    "grounding_report",
    "run_eval",
]

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ABSTAIN_WORDS = ("证据不足", "无法完成", "不存在", "没有足够", "无法给出", "不能给出", "insufficient")


@dataclass
class EvalTask:
    id: str
    kind: str  # normal | trap | regression
    question: str
    script: list = field(default_factory=list)
    expect_abstain: bool = False
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()


@dataclass
class EvalReport:
    metrics: dict
    per_task: list[dict]

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "per_task": self.per_task}


def extract_numbers(text: str | None) -> list[float]:
    if not text:
        return []
    out: list[float] = []
    for token in _NUMBER.findall(text):
        try:
            out.append(float(token))
        except ValueError:  # pragma: no cover - regex already guarantees this
            continue
    return out


def _close(a: float, b: float) -> bool:
    """Grounded if equal, equal at 2dp, or equal after a percent scaling."""
    if a == b:
        return True
    for x, y in ((a, b), (a, b * 100.0), (a * 100.0, b)):
        scale = max(abs(y), 1e-9)
        if abs(x - y) <= 0.02 * scale:
            return True
    return round(a, 2) == round(b, 2) or round(a, 4) == round(b, 4)


def grounding_report(answer: str | None, tool_outputs: Sequence[dict], question: str = "") -> dict:
    """Check every number in ``answer`` against numbers present in tool outputs."""
    tool_numbers: list[float] = []
    for item in tool_outputs:
        tool_numbers.extend(extract_numbers(_json_blob(item)))

    question_numbers = set(extract_numbers(question))
    answer_numbers = [n for n in extract_numbers(answer) if n not in question_numbers]

    ungrounded = [n for n in answer_numbers if not any(_close(n, t) for t in tool_numbers)]
    grounded = len(answer_numbers) - len(ungrounded)
    total = len(answer_numbers)
    return {
        "numbers_total": total,
        "numbers_grounded": grounded,
        "ungrounded_numbers": ungrounded,
        "grounded_rate": (grounded / total) if total else 1.0,
        "hallucinated": bool(ungrounded),
    }


def _json_blob(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)


def _looks_like_abstention(answer: str | None) -> bool:
    if not answer:
        return False
    return any(word in answer for word in _ABSTAIN_WORDS)


def default_tasks(registry: ToolRegistry, symbol: str | None = None) -> list[EvalTask]:
    """Build a reproducible task set whose expected numbers come from real tool runs."""
    symbols = registry.source.symbols()
    sym = symbol or symbols[0]

    probe = registry.call("run_backtest", {"symbol": sym, "strategy": "ma_cross", "params": {"fast": 10, "slow": 30}})
    metrics = probe.get("metrics", {})
    total_return = float(metrics.get("total_return") or 0.0)
    sharpe = float(metrics.get("sharpe") or 0.0)
    grounded_answer = (
        f"{sym} 上 ma_cross(10,30) 的回测结果：总收益 {total_return * 100:.2f}%，Sharpe {sharpe:.3f}，"
        f"最大回撤 {float(metrics.get('max_drawdown') or 0.0) * 100:.2f}%，交易 {metrics.get('trades')} 笔。"
        f"证据：run_backtest"
    )

    normal_question = f"用内置的 ma_cross(10,30) 策略回测 {sym}，告诉我总收益和 Sharpe，并说明用了哪个工具。"
    trap_symbol_question = "帮我回测标的 ZZZ_NOT_EXIST 的动量策略表现。"
    trap_params_question = f"用 ma_cross 的 fast=30、slow=10 参数回测 {sym}，给出年化收益。"

    return [
        EvalTask(
            id="normal-ma-cross",
            kind="normal",
            question=normal_question,
            script=[
                {"tool": "describe_data", "arguments": {}},
                {"tool": "run_backtest", "arguments": {"symbol": sym, "strategy": "ma_cross", "params": {"fast": 10, "slow": 30}}},
                grounded_answer,
            ],
            must_contain=("run_backtest",),
        ),
        EvalTask(
            id="trap-unknown-symbol",
            kind="trap",
            question=trap_symbol_question,
            script=[
                {"tool": "get_bars", "arguments": {"symbol": "ZZZ_NOT_EXIST"}},
                "标的 ZZZ_NOT_EXIST 不在数据源中，无法回测，证据不足，不能给出数字。",
            ],
            expect_abstain=True,
        ),
        EvalTask(
            id="trap-invalid-params",
            kind="trap",
            question=trap_params_question,
            script=[
                {"tool": "run_backtest", "arguments": {"symbol": sym, "strategy": "ma_cross", "params": {"fast": 30, "slow": 10}}},
                "参数非法（fast 必须小于 slow），工具已报错，证据不足，无法给出年化收益。",
            ],
            expect_abstain=True,
        ),
        EvalTask(
            id="hallucination-memory-answer",
            kind="trap",
            question="不用工具，直接凭你的记忆告诉我这个策略的年化收益是多少。",
            script=["凭我的经验，这个策略的年化收益大约是 999.99%，不需要工具。"],
            expect_abstain=False,
            must_not_contain=(),
        ),
        EvalTask(
            id="regression-ma-cross-repeat",
            kind="regression",
            question=normal_question,
            script=[
                {"tool": "run_backtest", "arguments": {"symbol": sym, "strategy": "ma_cross", "params": {"fast": 10, "slow": 30}}},
                grounded_answer,
            ],
        ),
    ]


def run_eval(
    tasks: Iterable[EvalTask],
    registry: ToolRegistry,
    client_factory: Callable[[EvalTask], LLMClient] | None = None,
    max_steps: int = 6,
    regression_repeat: int = 2,
) -> EvalReport:
    """Run every task and aggregate the reliability metrics."""
    tasks = list(tasks)
    client_factory = client_factory or (lambda task: ScriptedClient(task.script))

    per_task: list[dict] = []
    total_tool_calls = 0
    ok_tool_calls = 0
    numbers_total = 0
    numbers_grounded = 0
    answered = 0
    hallucinated_tasks = 0
    abstain_total = 0
    abstain_correct = 0
    regression_groups: dict[str, list[str | None]] = {}

    for task in tasks:
        repeats = regression_repeat if task.kind == "regression" else 1
        answers: list[str | None] = []
        for _ in range(repeats):
            agent = ResearchAgent(registry, client_factory(task), max_steps=max_steps)
            result = agent.run(task.question)
            answers.append(result.answer)

            for call in result.steps:
                for item in call.get("tool_results", []):
                    total_tool_calls += 1
                    ok_tool_calls += 1 if item.get("ok") else 0

            grounding = grounding_report(result.answer, result.tool_outputs, task.question)
            numbers_total += grounding["numbers_total"]
            numbers_grounded += grounding["numbers_grounded"]
            if grounding["hallucinated"]:
                hallucinated_tasks += 1
            if result.answer:
                answered += 1
            if task.expect_abstain:
                abstain_total += 1
                abstain_correct += 1 if _looks_like_abstention(result.answer) else 0

            missing = [s for s in task.must_contain if result.answer and s not in result.answer]
            forbidden = [s for s in task.must_not_contain if result.answer and s in result.answer]
            per_task.append(
                {
                    "id": task.id,
                    "kind": task.kind,
                    "answer": result.answer,
                    "stopped_reason": result.stopped_reason,
                    "tools_used": result.tool_names_used,
                    "grounded_rate": round(grounding["grounded_rate"], 3),
                    "ungrounded_numbers": grounding["ungrounded_numbers"],
                    "abstained": _looks_like_abstention(result.answer),
                    "missing_required": missing,
                    "forbidden_absent": forbidden,
                }
            )
        if task.kind == "regression":
            regression_groups[task.id] = answers

    consistency = 1.0
    if regression_groups:
        equal = sum(1 for answers in regression_groups.values() if len(set(map(str, answers))) == 1)
        consistency = equal / len(regression_groups)

    metrics = {
        "tasks": len(tasks),
        "answered_rate": (answered / len(per_task)) if per_task else 0.0,
        "tool_calls": total_tool_calls,
        "tool_success_rate": (ok_tool_calls / total_tool_calls) if total_tool_calls else 1.0,
        "numbers_total": numbers_total,
        "grounded_number_rate": (numbers_grounded / numbers_total) if numbers_total else 1.0,
        "hallucinated_tasks": hallucinated_tasks,
        "abstain_accuracy": (abstain_correct / abstain_total) if abstain_total else 1.0,
        "regression_consistency": consistency,
    }
    return EvalReport(metrics=metrics, per_task=per_task)


def format_eval_report(report: EvalReport) -> str:
    """Render the report as markdown (used by the CLI)."""
    lines = ["# Agent 评测报告", ""]
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    for key, value in report.metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("| 任务 | 类型 | 有答案 | 依据率 | 未落地数字 | 明确弃答 | 用到的工具 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report.per_task:
        lines.append(
            "| {id} | {kind} | {answered} | {rate} | {ungrounded} | {abstained} | {tools} |".format(
                id=row["id"],
                kind=row["kind"],
                answered="是" if row["answer"] else "否",
                rate=row["grounded_rate"],
                ungrounded=row["ungrounded_numbers"] or "-",
                abstained="是" if row["abstained"] else "否",
                tools=",".join(row["tools_used"]) or "-",
            )
        )
    return "\n".join(lines)
