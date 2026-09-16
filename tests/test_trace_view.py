"""trace 可视化的测试（离线）。

这个模块的用途是"把代理做过的事讲清楚"，因此测试要守住三件事：

① **自包含**：单文件 HTML、零外部引用（断网也能打开，不能依赖 CDN 字体或脚本）；
② **不美化**：工具失败、未落地数字、非正常停止都要**显式标出**——
   若把失败渲染成"成功"或直接省略，这个视图就失去了可信度；
③ **确定性**：同一输入必须产出完全相同的 HTML（便于 diff 与版本控制）。
"""

import json

import pytest

from aqlab.trace_view import render_trace_html, summarize_trace, write_trace_html

TRACE = {
    "question": "用 ma_cross 回测 SYN001，告诉我总收益。",
    "answer": "总收益 +9.26%，夏普 0.3527。另外年化约 4.57%。",
    "stopped_reason": "final",
    "steps": [
        {
            "step": 1,
            "assistant": "先看看有哪些策略。",
            "tool_calls": [{"name": "list_strategies", "arguments": {}}],
            "tool_results": [{"tool": "list_strategies", "ok": True, "result": {"ok": True, "strategies": ["ma_cross"]}}],
        },
        {
            "step": 2,
            "assistant": "跑一次回测。",
            "tool_calls": [{"name": "run_backtest", "arguments": {"symbol": "SYN001", "strategy": "ma_cross"}}],
            "tool_results": [
                {
                    "tool": "run_backtest",
                    "ok": True,
                    "result": {"ok": True, "total_return": 9.26, "sharpe": 0.3527, "cagr": 4.57},
                }
            ],
        },
    ],
    "tool_outputs": [
        {"tool": "list_strategies", "arguments": {}, "result": {"ok": True, "strategies": ["ma_cross"]}},
        {
            "tool": "run_backtest",
            "arguments": {"symbol": "SYN001", "strategy": "ma_cross"},
            "result": {"ok": True, "total_return": 9.26, "sharpe": 0.3527, "cagr": 4.57},
        },
    ],
}


def test_summarize_counts_steps_tools_and_failures():
    summary = summarize_trace(TRACE)
    assert summary["steps"] == 2
    assert summary["tool_calls"] == 2
    assert summary["tools"] == ["list_strategies", "run_backtest"]
    assert summary["failures"] == []
    assert summary["stopped_reason"] == "final"
    assert summary["has_answer"] is True


def test_summarize_reports_failed_tool_calls():
    trace = json.loads(json.dumps(TRACE))
    trace["steps"][1]["tool_results"][0]["ok"] = False
    assert summarize_trace(trace)["failures"] == ["run_backtest"]


def test_html_is_self_contained():
    """不引用任何外部资源：断网也能打开，避免演示时"打不开图/样式错乱"。"""
    html = render_trace_html(TRACE)
    for needle in ("http://", "https://", "<script", "cdn."):
        assert needle not in html, f"HTML 不应包含 {needle}"


def test_html_shows_the_timeline_and_tools():
    html = render_trace_html(TRACE)
    assert "逐步时间线" in html
    assert "第 1 步" in html and "第 2 步" in html
    assert "list_strategies" in html and "run_backtest" in html
    assert html.count('class="step"') == 2
    assert html.count('class="tool"') == 2


def test_failed_tool_is_marked_not_hidden():
    """工具失败必须显式标出——不美化是这个视图的可信度基础。"""
    trace = json.loads(json.dumps(TRACE))
    trace["steps"][0]["tool_results"][0] = {"tool": "list_strategies", "ok": False, "result": {"ok": False, "error": "boom"}}
    html = render_trace_html(trace)
    assert "失败" in html
    assert "badge warn" in html
    # 卡片区的失败计数也要非 0
    assert summarize_trace(trace)["failures"] == ["list_strategies"]


def test_ungrounded_numbers_are_flagged():
    """答案里无法在工具输出中找到的数字必须提示出来（与评测模块同一口径）。"""
    trace = json.loads(json.dumps(TRACE))
    trace["answer"] = "总收益 +9.26%，夏普 0.3527，另外凭空多出一个 888.88%。"
    html = render_trace_html(trace)
    assert "未落地数字" in html
    assert "888.88" in html

    clean = render_trace_html(TRACE)
    # 全部数字都有依据时不应出现该提示段（卡片标题仍可能出现，故检查提示块）
    assert 'class="callout warn"' not in clean


def test_abnormal_stop_is_highlighted():
    """非 final 停止（例如撞上 max_steps）应被标成警示，而不是悄悄放过。"""
    trace = json.loads(json.dumps(TRACE))
    trace["stopped_reason"] = "max_steps"
    html = render_trace_html(trace)
    assert "max_steps" in html
    assert "card warn" in html


def test_empty_trace_renders_without_crashing():
    html = render_trace_html({})
    assert "没有记录到任何步骤" in html
    assert "没有产出最终回答" in html


def test_html_escapes_dangerous_content():
    """用户/模型文本必须转义，避免渲染出一个可执行的页面。"""
    trace = json.loads(json.dumps(TRACE))
    trace["question"] = '<script>alert("x")</script>'
    trace["answer"] = '<img src=x onerror=alert(1)>'
    html = render_trace_html(trace)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "onerror=alert" not in html or "&lt;img" in html


def test_render_is_deterministic():
    """同一输入必须产出完全相同的 HTML（便于 diff 与提交）。"""
    assert render_trace_html(TRACE) == render_trace_html(TRACE)


def test_write_trace_html_accepts_dict_and_file(tmp_path):
    direct = write_trace_html(TRACE, tmp_path / "a.html")
    assert direct.is_file()

    source = tmp_path / "trace.json"
    source.write_text(json.dumps(TRACE, ensure_ascii=False), encoding="utf-8")
    from_file = write_trace_html(source, tmp_path / "nested" / "b.html")
    assert from_file.is_file()
    assert from_file.read_text(encoding="utf-8") == direct.read_text(encoding="utf-8")


def test_write_trace_html_unwraps_result_envelope(tmp_path):
    """兼容 `{"result": {...}}` 包装（CLI 可能直接透传）。"""
    source = tmp_path / "wrapped.json"
    source.write_text(json.dumps({"result": TRACE}, ensure_ascii=False), encoding="utf-8")
    output = write_trace_html(source, tmp_path / "c.html")
    assert "逐步时间线" in output.read_text(encoding="utf-8")


def test_write_trace_html_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_trace_html(tmp_path / "nope.json", tmp_path / "x.html")


def test_committed_demo_trace_is_self_contained_and_reproducible():
    """已提交的演示产物必须：存在、自包含、且能由仓库代码从 JSON 重新渲染出来。

    这条测试防的是"README 引用了一张来路不明的图/页面"——演示产物必须可追溯到代码与原始 trace。
    """
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    html_path = root / "docs" / "assets" / "agent_trace.html"
    json_path = root / "docs" / "assets" / "agent_trace.json"
    assert html_path.is_file(), "应提交一份演示用的 trace HTML"
    assert json_path.is_file(), "应提交对应的原始 trace JSON"

    html = html_path.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, "演示页面必须自包含"
    assert "逐步时间线" in html

    # 由原始 JSON 重新渲染，必须与已提交的 HTML 完全一致（同代码同输入同输出）
    trace = _json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(trace.get("result"), dict):
        trace = trace["result"]
    # 提交时 CLI 传入的标题是 "Agent trace · demo_live"；比对时必须用同一标题，
    # 否则会把"标题不同"误判成"渲染不可复现"。
    title = "Agent trace · demo_live"
    assert title in html, "演示页面的标题应与 CLI 生成时一致"
    assert render_trace_html(trace, title=title) == html, "已提交的演示页面必须能由仓库代码复现"
