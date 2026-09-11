import pytest

from aqlab.agent import ScriptedClient
from aqlab.evaluation import (
    EvalTask,
    default_tasks,
    extract_numbers,
    format_eval_report,
    grounding_report,
    run_eval,
)
from aqlab.tools import SyntheticDataSource, ToolRegistry


@pytest.fixture(scope="module")
def registry():
    return ToolRegistry(source=SyntheticDataSource(n_symbols=8, n_days=400, seed=5))


def test_extract_numbers_handles_signs_and_decimals():
    assert extract_numbers("总收益 -12.34%，Sharpe 0.78") == [-12.34, 0.78]
    assert extract_numbers(None) == []
    assert extract_numbers("没有数字") == []


def test_grounding_accepts_exact_and_percent_scaled_numbers():
    outputs = [{"result": {"total_return": 0.1234, "sharpe": 0.7777}}]
    report = grounding_report("总收益 12.34%，Sharpe 0.7777。", outputs)
    assert report["numbers_total"] == 2
    assert report["numbers_grounded"] == 2
    assert report["hallucinated"] is False


def test_grounding_flags_invented_numbers():
    outputs = [{"result": {"total_return": 0.05}}]
    report = grounding_report("我凭记忆认为总收益是 999.99%。", outputs)
    assert report["hallucinated"] is True
    assert report["ungrounded_numbers"] == [999.99]
    assert report["grounded_rate"] == 0.0


def test_default_tasks_are_well_formed(registry):
    tasks = default_tasks(registry)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    kinds = {t.kind for t in tasks}
    assert {"normal", "trap", "regression"}.issubset(kinds)
    assert sum(1 for t in tasks if t.expect_abstain) == 2
    for task in tasks:
        assert task.question
        assert task.script


def test_offline_eval_metrics(registry):
    tasks = default_tasks(registry)
    report = run_eval(tasks, registry, client_factory=None, max_steps=6)

    m = report.metrics
    assert m["tasks"] == len(tasks)
    # 4 successful calls (describe_data + 3 run_backtest) out of 6; the 2 failures
    # are the intentional trap-tasks (unknown symbol / invalid parameters)
    assert m["tool_calls"] == 6
    assert m["tool_success_rate"] == pytest.approx(4 / 6)
    assert m["abstain_accuracy"] == pytest.approx(1.0)
    assert m["regression_consistency"] == pytest.approx(1.0)
    assert m["answered_rate"] == pytest.approx(1.0)
    # the deliberately hallucinating task must be detected
    assert m["hallucinated_tasks"] >= 1
    assert 0.5 <= m["grounded_number_rate"] <= 1.0

    rows = {row["id"]: row for row in report.per_task}
    assert rows["normal-ma-cross"]["grounded_rate"] == pytest.approx(1.0)
    assert rows["hallucination-memory-answer"]["grounded_rate"] == 0.0
    assert rows["trap-unknown-symbol"]["abstained"] is True
    assert rows["trap-invalid-params"]["abstained"] is True
    assert "run_backtest" in rows["normal-ma-cross"]["tools_used"]


def test_eval_detects_a_bad_agent(registry):
    """An agent that answers from memory without tools must score badly."""
    bad_task = EvalTask(
        id="bad-agent",
        kind="trap",
        question="SYN001 的年化收益是多少？",
        script=["大约是 42.0% 吧。"],
    )
    report = run_eval([bad_task], registry)
    assert report.metrics["tool_calls"] == 0
    assert report.metrics["grounded_number_rate"] == 0.0
    assert report.metrics["hallucinated_tasks"] == 1


def test_eval_report_renders_markdown(registry):
    report = run_eval(default_tasks(registry), registry)
    text = format_eval_report(report)
    assert "# Agent 评测报告" in text
    assert "| 指标 | 值 |" in text
    assert "grounded_number_rate" in text
    assert "normal-ma-cross" in text


def test_live_mode_factory_is_used(registry):
    """The harness must run a supplied client (this is how live models are scored)."""
    calls = {"n": 0}

    def factory(task):
        calls["n"] += 1
        return ScriptedClient(task.script)

    report = run_eval(default_tasks(registry), registry, client_factory=factory, regression_repeat=2)
    assert calls["n"] == len(default_tasks(registry)) + 1  # +1 for the regression repeat
    assert report.metrics["tool_calls"] >= 3
