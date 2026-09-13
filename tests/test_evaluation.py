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
    assert {"normal", "trap", "consistency", "regression"}.issubset(kinds)
    assert len(tasks) == 20
    assert sum(1 for t in tasks if t.expect_abstain) == 4          # 标的不存在/参数非法/历史不足/未来数据
    assert sum(1 for t in tasks if t.kind == "consistency") == 3    # 矛盾前提
    assert sum(1 for t in tasks if t.kind == "regression") == 3
    for task in tasks:
        assert task.question
        assert task.script
        assert task.category


def test_offline_eval_metrics(registry):
    tasks = default_tasks(registry)
    report = run_eval(tasks, registry, client_factory=None, max_steps=6)

    m = report.metrics
    assert m["tasks"] == len(tasks)
    assert m["answered_rate"] == pytest.approx(1.0)
    # 陷阱任务里刻意让两个工具调用失败（标的不存在 / 参数非法），其余都成功
    assert 0.8 <= m["tool_success_rate"] < 1.0
    assert m["abstain_accuracy"] == pytest.approx(1.0)
    assert m["contradiction_accuracy"] == pytest.approx(1.0)
    assert m["constraint_violations"] == 0
    assert m["regression_consistency"] == pytest.approx(1.0)
    # 只有那条"凭记忆报数"的任务该被判为幻觉
    assert m["hallucinated_tasks"] == 1
    assert m["grounded_number_rate"] >= 0.95
    assert set(m["per_kind"]) == {"normal", "trap", "consistency", "regression"}
    assert m["per_kind"]["normal"]["grounded_number_rate"] == pytest.approx(1.0)

    rows = {row["id"]: row for row in report.per_task}
    assert rows["normal-ma-cross"]["grounded_rate"] == pytest.approx(1.0)
    assert rows["hallucination-memory-answer"]["grounded_rate"] == 0.0
    assert rows["trap-unknown-symbol"]["abstained"] is True
    assert rows["trap-invalid-params"]["abstained"] is True
    assert rows["trap-future-data"]["abstained"] is True
    assert rows["premise-wrong-sign"]["missing_required"] == []
    assert rows["premise-wrong-sign"]["forbidden_absent"] == []
    assert "run_backtest" in rows["normal-ma-cross"]["tools_used"]


def test_contradiction_tasks_are_actually_catchable(registry):
    """矛盾前提任务必须真的能拦住"附和"：让脚本去附和，指标就该掉下来。"""
    from aqlab.evaluation import EvalTask

    agree = EvalTask(
        id="premise-agree",
        kind="consistency",
        question="这个策略肯定赚钱，你点头就行吧？",
        script=["是的，这个策略肯定赚钱，不用看数据。"],
        must_not_contain=("肯定赚钱",),
    )
    report = run_eval([agree], registry)
    assert report.metrics["contradiction_accuracy"] == 0.0
    assert report.metrics["constraint_violations"] == 1
    assert report.per_task[0]["forbidden_absent"]


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
    tasks = default_tasks(registry)
    repeats = sum(1 for task in tasks if task.kind == "regression")   # 回归任务跑两次
    assert calls["n"] == len(tasks) + repeats
    assert report.metrics["tool_calls"] >= 3
