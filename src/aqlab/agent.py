"""Provider-agnostic research agent: plan → call tools → observe → answer.

The agent is deliberately small and auditable:

* the model may only call **read-only tools** from a :class:`~aqlab.tools.ToolRegistry`;
* every step (assistant message, tool call, tool result) is written to a trace that
  can be replayed and inspected;
* numbers in the final answer are expected to come from tool outputs — the
  evaluation harness scores grounding separately;
* the loop is bounded by ``max_steps`` so a confused model cannot spin forever.

Two clients ship with the project:

* :class:`OpenAICompatClient` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (DeepSeek, Moonshot, vLLM, Ollama gateway, ...)
  using only the standard library;
* :class:`ScriptedClient` — deterministic, offline, used by tests and the
  offline evaluation mode.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from aqlab.tools import ToolRegistry, dumps

__all__ = [
    "SYSTEM_PROMPT",
    "AgentResult",
    "LLMClient",
    "LLMReply",
    "OpenAICompatClient",
    "ResearchAgent",
    "ScriptedClient",
    "ToolCall",
    "parse_text_tool_calls",
]

SYSTEM_PROMPT = """你是一个量化研究助手，只能通过工具获取事实。

规则：
1. 任何数字都必须来自工具返回结果，禁止凭记忆或估算编造。
2. 引用某个标的之前，必须先调用 describe_data 确认它存在。
3. 如果工具返回 ok=false（例如标的不存在、参数非法、历史数据不足），必须直接说明"证据不足/无法完成"，不要猜测。
4. 回测数字一律用 run_backtest 的结果，不要自己计算收益。
5. 最终答案用中文，先给结论，再列关键数字，最后一行写"证据：<用到的工具名>"。
"""


# --------------------------------------------------------------------------------------
# Client protocol
# --------------------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = "call_0"


@dataclass
class LLMReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMReply: ...


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE_JSON = re.compile(r"\{.*\}", re.S)


def parse_text_tool_calls(content: str | None) -> list[ToolCall]:
    """Extract tool calls emitted as text (for models without native tool calling).

    Accepted shapes::

        {"tool": "run_backtest", "arguments": {...}}
        {"name": "...", "arguments": {...}}
        {"tool_calls": [{"name": "...", "arguments": {...}}, ...]}
    """
    if not content:
        return []
    candidates: list[str] = []
    for match in _FENCED_JSON.finditer(content):
        candidates.append(match.group(1))
    if not candidates:
        for match in _BARE_JSON.finditer(content):
            candidates.append(match.group(0))

    calls: list[ToolCall] = []
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        items = data["tool_calls"] if "tool_calls" in data and isinstance(data["tool_calls"], list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("tool") or item.get("name")
            if not name:
                continue
            args = item.get("arguments") or item.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=str(name), arguments=dict(args), id=f"call_{len(calls)}"))
    return calls


class ScriptedClient:
    """Deterministic client: replays a fixed list of assistant turns (offline tests/eval).

    Each script item is either

    * ``str`` — a final answer, or
    * ``{"tool": name, "arguments": {...}}`` — a native tool call, or
    * ``{"content": "...", "tool_calls": [...]}`` — explicit reply.
    """

    def __init__(self, script: Sequence[Any]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    @property
    def remaining(self) -> int:
        return len(self.script)

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMReply:
        self.calls.append(list(messages))
        if not self.script:
            return LLMReply(content="（脚本已用尽，无更多回复）")
        item = self.script.pop(0)
        if isinstance(item, str):
            return LLMReply(content=item)
        if isinstance(item, dict) and ("tool" in item or "name" in item) and "tool_calls" not in item:
            name = item.get("tool") or item.get("name")
            return LLMReply(tool_calls=[ToolCall(name=str(name), arguments=dict(item.get("arguments") or {}))])
        if isinstance(item, dict):
            calls = [
                ToolCall(name=str(c.get("tool") or c.get("name")), arguments=dict(c.get("arguments") or {}), id=str(c.get("id", f"call_{i}")))
                for i, c in enumerate(item.get("tool_calls") or [])
            ]
            content = item.get("content")
            if not calls and content:
                # simulate a model that emits tool calls inside its text
                calls = parse_text_tool_calls(content)
            return LLMReply(content=content, tool_calls=calls)
        raise TypeError(f"unsupported script item: {item!r}")


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client using only the standard library.

    Configure via arguments or environment variables::

        AQLAB_LLM_API_KEY   (falls back to LLM_API_KEY / DEEPSEEK_API_KEY)
        AQLAB_LLM_BASE_URL  (default https://api.deepseek.com/v1)
        AQLAB_LLM_MODEL     (default deepseek-chat)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 90,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("AQLAB_LLM_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("AQLAB_LLM_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
        self.model = model or os.environ.get("AQLAB_LLM_MODEL") or "deepseek-chat"
        self.timeout = timeout
        self.temperature = temperature
        if not self.api_key:
            raise ValueError(
                "no API key found; set AQLAB_LLM_API_KEY (or LLM_API_KEY / DEEPSEEK_API_KEY) to use the live agent"
            )

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"LLM connection error: {exc.reason}") from exc

        message = (body.get("choices") or [{}])[0].get("message") or {}
        calls: list[ToolCall] = []
        for i, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=str(function.get("name", "")), arguments=dict(args), id=str(call.get("id", f"call_{i}"))))

        content = message.get("content")
        if not calls:
            calls = parse_text_tool_calls(content)
        return LLMReply(content=content, tool_calls=calls)


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
@dataclass
class AgentResult:
    question: str
    answer: str | None
    steps: list[dict]
    tool_outputs: list[dict]
    stopped_reason: str
    tool_names_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "stopped_reason": self.stopped_reason,
            "tool_names_used": self.tool_names_used,
            "steps": self.steps,
            "tool_outputs": self.tool_outputs,
        }


class ResearchAgent:
    """Bounded plan-act-observe loop over a read-only tool registry."""

    def __init__(self, registry: ToolRegistry, client: LLMClient, max_steps: int = 6, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.registry = registry
        self.client = client
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt

    def run(self, question: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[dict] = []
        tool_outputs: list[dict] = []
        used: list[str] = []
        answer: str | None = None
        stopped_reason = "max_steps"

        for step_index in range(self.max_steps):
            reply = self.client.chat(messages, tools=self.registry.specs())
            record: dict[str, Any] = {
                "step": step_index + 1,
                "assistant": reply.content,
                "tool_calls": [],
                "tool_results": [],
            }

            if not reply.has_tool_calls:
                answer = reply.content
                stopped_reason = "final"
                steps.append(record)
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                        }
                        for call in reply.tool_calls
                    ],
                }
            )

            for call in reply.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                tool_outputs.append({"tool": call.name, "arguments": call.arguments, "result": result})
                used.append(call.name)
                record["tool_calls"].append({"name": call.name, "arguments": call.arguments})
                record["tool_results"].append({"tool": call.name, "ok": bool(result.get("ok")), "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": dumps(result)})

            steps.append(record)

        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            tool_outputs=tool_outputs,
            stopped_reason=stopped_reason,
            tool_names_used=sorted(set(used)),
        )
