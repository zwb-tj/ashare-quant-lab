"""把 Agent 的 trace 渲染成可读的单文件 HTML（v0.34）。

**为什么需要**：trace 一直是 JSON —— 完整、可机读，但演示时看不出"代理做了什么"。
评审或演示时最想看到的是三件事：**每一步调了什么工具、工具返回是否成功、最终结论有没有依据**。
这个模块把 trace 折叠成一条时间线，并**显式标出未落地的数字**（答案里无法在工具输出中找到的数字）。

设计原则（与项目其余部分一致）：

* **单文件、零外部依赖**：样式内联，不引用 CDN，断网也能打开；
* **不美化**：工具失败、未落地数字、提前停止（`max_steps`）都直接标红显示，
  不做"看起来都很好"的排版；
* **可复现**：输入是 `AgentResult.to_dict()` 的 JSON，输出是确定性 HTML（测试比对同输入同输出）。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["render_trace_html", "summarize_trace", "write_trace_html"]


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _short(value: Any, limit: int = 400) -> str:
    """把长文本裁短（保留头部），用于工具返回的预览。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（共 {len(text)} 字符，已截断）"


def summarize_trace(trace: Mapping[str, Any]) -> dict:
    """从 trace 里抽出汇总信息：步数、工具调用、失败调用、停止原因。"""
    steps = list(trace.get("steps") or [])
    tool_calls: list[str] = []
    failures: list[str] = []
    for step in steps:
        for call in step.get("tool_calls") or []:
            name = str(call.get("name"))
            tool_calls.append(name)
        for result in step.get("tool_results") or []:
            if not result.get("ok"):
                failures.append(str(result.get("tool")))
    return {
        "steps": len(steps),
        "tool_calls": len(tool_calls),
        "tools": sorted(set(tool_calls)),
        "failures": failures,
        "stopped_reason": str(trace.get("stopped_reason") or ""),
        "has_answer": bool(trace.get("answer")),
    }


def _ungrounded(answer: str | None, tool_outputs: Sequence[Mapping[str, Any]]) -> list[float]:
    """答案里无法在工具输出中找到的数字（复用评测模块的口径，避免两套标准）。"""
    from aqlab.evaluation import grounding_report

    # grounding_report 的签名要 list[dict]；这里显式转换，避免把只读 Mapping 传进去。
    payload: list[dict] = [dict(item) for item in tool_outputs]
    report = grounding_report(answer, payload)
    return [float(value) for value in (report.get("ungrounded_numbers") or [])]


def render_trace_html(trace: Mapping[str, Any], title: str = "Agent trace") -> str:
    """把一个 trace 渲染成完整 HTML 字符串。"""
    summary = summarize_trace(trace)
    ungrounded = _ungrounded(trace.get("answer"), trace.get("tool_outputs") or [])
    question = _escape(trace.get("question"))
    answer = _escape(trace.get("answer") or "（没有产出最终回答）")

    stopped = summary["stopped_reason"]
    stopped_flag = "warn" if stopped not in ("final",) else "ok"
    failure_flag = "warn" if summary["failures"] else "ok"
    ground_flag = "warn" if ungrounded else "ok"

    cards = [
        ("停止原因", _escape(stopped or "未知"), stopped_flag),
        ("步数", str(summary["steps"]), "ok"),
        ("工具调用", f"{summary['tool_calls']} 次（{len(summary['tools'])} 种）", "ok"),
        ("失败的调用", str(len(summary["failures"])) if summary["failures"] else "0", failure_flag),
        ("未落地数字", str(len(ungrounded)) if ungrounded else "0", ground_flag),
    ]

    timeline: list[str] = []
    for step in trace.get("steps") or []:
        index = _escape(step.get("step"))
        assistant = _escape(_short(step.get("assistant") or "（无文本，直接调用工具）", 600))
        blocks: list[str] = []
        results_by_name = {str(item.get("tool")): item for item in step.get("tool_results") or []}
        for call in step.get("tool_calls") or []:
            name = _escape(call.get("name"))
            arguments = _escape(_short(call.get("arguments"), 300))
            result = results_by_name.get(str(call.get("name")))
            if result is None:
                blocks.append(
                    f'<div class="tool"><div class="tool-head"><b>{name}</b>'
                    f'<span class="badge warn">无结果</span></div>'
                    f'<pre class="args">{arguments}</pre></div>'
                )
                continue
            ok = bool(result.get("ok"))
            badge = '<span class="badge ok">成功</span>' if ok else '<span class="badge warn">失败</span>'
            payload = _escape(_short(result.get("result"), 700))
            blocks.append(
                f'<div class="tool"><div class="tool-head"><b>{name}</b>{badge}</div>'
                f'<pre class="args">参数：{arguments}</pre>'
                f'<pre class="result">{payload}</pre></div>'
            )
        timeline.append(
            f'<div class="step"><div class="step-head">第 {index} 步</div>'
            f'<div class="assistant">{assistant}</div>{"".join(blocks)}</div>'
        )

    if not timeline:
        timeline.append('<div class="step"><div class="assistant">（没有记录到任何步骤）</div></div>')

    ungrounded_html = ""
    if ungrounded:
        values = ", ".join(_escape(value) for value in ungrounded[:20])
        ungrounded_html = (
            f'<div class="callout warn"><b>未落地数字</b>：{values}'
            "　（这些数字在工具输出里找不到对应值，是「编造数字」的直接信号）</div>"
        )

    # 卡片与时间线先分别拼好：嵌套 f-string（同种引号）只在 Python 3.12+ 合法，
    # 而本项目声明支持 3.10，因此这里显式分步构造。
    card_html = "".join(
        '<div class="card {flag}"><div class="k">{key}</div><div class="v">{value}</div></div>'.format(
            flag=flag, key=_escape(key), value=value
        )
        for key, value, flag in cards
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{_escape(title)}</title>
<style>
  :root {{ --fg:#1b1f24; --muted:#5b6672; --line:#e3e7ec; --ok:#1a7f37; --warn:#b42318; --bg:#fff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:28px; font:14px/1.6 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Microsoft YaHei",sans-serif; color:var(--fg); background:var(--bg); }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .muted {{ color:var(--muted); }}
  .cards {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 22px; }}
  .card {{ border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-width:130px; }}
  .card .k {{ font-size:12px; color:var(--muted); }}
  .card .v {{ font-size:18px; font-weight:600; }}
  .card.warn .v {{ color:var(--warn); }}
  .card.ok .v {{ color:var(--ok); }}
  .qa {{ border-left:3px solid var(--line); padding:2px 0 2px 14px; margin:0 0 20px; }}
  .qa .label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  .step {{ border:1px solid var(--line); border-radius:10px; margin:0 0 14px; overflow:hidden; }}
  .step-head {{ background:#f6f8fa; padding:8px 14px; font-weight:600; border-bottom:1px solid var(--line); }}
  .assistant {{ padding:12px 14px; white-space:pre-wrap; }}
  .tool {{ border-top:1px dashed var(--line); padding:10px 14px; }}
  .tool-head {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
  .badge {{ font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid; }}
  .badge.ok {{ color:var(--ok); border-color:#b7e0c0; background:#f0f9f2; }}
  .badge.warn {{ color:var(--warn); border-color:#f3c2bd; background:#fdf3f2; }}
  pre {{ margin:4px 0 0; padding:8px 10px; background:#f6f8fa; border-radius:6px; font-size:12px; white-space:pre-wrap; word-break:break-word; }}
  pre.args {{ color:var(--muted); }}
  .callout {{ border-radius:10px; padding:10px 14px; margin:0 0 18px; border:1px solid; }}
  .callout.warn {{ border-color:#f3c2bd; background:#fdf3f2; color:var(--warn); }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<div class="muted">由 <code>aqlab trace</code> 从 AgentResult.to_dict() 生成｜单文件、无外部依赖</div>

<div class="cards">
{card_html}
</div>

<div class="qa">
  <div class="label">问题</div>
  <div>{question}</div>
</div>

{ungrounded_html}

<div class="qa">
  <div class="label">最终回答</div>
  <div>{answer}</div>
</div>

<h2 style="font-size:16px;margin:22px 0 10px;">逐步时间线</h2>
{"".join(timeline)}
</body>
</html>
"""


def write_trace_html(trace: Mapping[str, Any] | str | Path, path: str | Path, title: str = "Agent trace") -> Path:
    """把 trace（字典或 JSON 文件路径）写成 HTML 文件，返回输出路径。"""
    loaded: Mapping[str, Any] = (
        json.loads(Path(trace).read_text(encoding="utf-8")) if isinstance(trace, (str, Path)) else trace
    )
    inner = loaded.get("result")
    # 兼容 {"result": {...}} 包装
    data: Mapping[str, Any] = inner if isinstance(inner, Mapping) else loaded
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trace_html(data, title=title), encoding="utf-8")
    return output
