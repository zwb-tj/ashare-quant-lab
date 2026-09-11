import pytest

from aqlab.agent import OpenAICompatClient, ResearchAgent, ScriptedClient, parse_text_tool_calls
from aqlab.tools import SyntheticDataSource, ToolRegistry


@pytest.fixture(scope="module")
def registry():
    return ToolRegistry(source=SyntheticDataSource(n_symbols=6, n_days=350, seed=9))


def test_agent_executes_tool_then_answers(registry):
    script = [
        {"tool": "run_backtest", "arguments": {"symbol": "SYN001", "strategy": "ma_cross", "params": {"fast": 10, "slow": 30}}},
        "回测完成，详见工具返回。证据：run_backtest",
    ]
    client = ScriptedClient(script)
    agent = ResearchAgent(registry, client, max_steps=4)
    result = agent.run("回测 SYN001 的 ma_cross")

    assert result.stopped_reason == "final"
    assert len(result.steps) == 2
    assert len(result.tool_outputs) == 1
    assert result.tool_outputs[0]["tool"] == "run_backtest"
    assert result.tool_outputs[0]["result"]["ok"] is True
    assert result.tool_names_used == ["run_backtest"]
    assert "证据" in result.answer


def test_agent_respects_max_steps(registry):
    script = [{"tool": "describe_data", "arguments": {}}] * 5
    client = ScriptedClient(script)
    result = ResearchAgent(registry, client, max_steps=2).run("无限调用工具")
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 2
    assert len(result.tool_outputs) == 2
    assert result.answer is None
    assert len(client.script) == 3  # unconsumed turns prove the loop stopped


def test_tool_errors_are_fed_back_to_the_model(registry):
    script = [
        {"tool": "get_bars", "arguments": {"symbol": "ZZZ_NOT_EXIST"}},
        "标的 ZZZ_NOT_EXIST 不存在，证据不足，无法给出数字。",
    ]
    client = ScriptedClient(script)
    result = ResearchAgent(registry, client).run("回测 ZZZ_NOT_EXIST")
    # the second model call must contain the tool error message
    second_call = client.calls[1]
    tool_messages = [m for m in second_call if m.get("role") == "tool"]
    assert tool_messages and "ZZZ_NOT_EXIST" in tool_messages[0]["content"]
    assert result.tool_outputs[0]["result"]["ok"] is False
    assert "证据不足" in result.answer


def test_text_format_tool_calls_are_parsed_and_executed(registry):
    script = [
        {"content": '```json\n{"tool": "describe_data", "arguments": {}}\n```'},
        "数据源内有 6 个标的。证据：describe_data",
    ]
    result = ResearchAgent(registry, ScriptedClient(script)).run("有哪些标的？")
    assert result.tool_names_used == ["describe_data"]
    assert result.steps[0]["tool_results"][0]["ok"] is True


def test_parse_text_tool_calls_shapes():
    assert parse_text_tool_calls(None) == []
    assert parse_text_tool_calls("没有工具调用") == []

    one = parse_text_tool_calls('{"tool": "get_bars", "arguments": {"symbol": "SYN001"}}')
    assert len(one) == 1 and one[0].name == "get_bars" and one[0].arguments == {"symbol": "SYN001"}

    named = parse_text_tool_calls('prefix {"name": "screen_universe", "arguments": {"top_n": 3}} suffix')
    assert named[0].name == "screen_universe" and named[0].arguments == {"top_n": 3}

    multi = parse_text_tool_calls('{"tool_calls": [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {"x": 1}}]}')
    assert [c.name for c in multi] == ["a", "b"]

    string_args = parse_text_tool_calls('{"tool": "get_bars", "arguments": "{\\"symbol\\": \\"SYN002\\"}"}')
    assert string_args[0].arguments == {"symbol": "SYN002"}


def test_scripted_client_rejects_unknown_item():
    client = ScriptedClient([123])
    with pytest.raises(TypeError):
        client.chat([])


def test_openai_compatible_client_requires_key(monkeypatch):
    for var in ("AQLAB_LLM_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        OpenAICompatClient()
