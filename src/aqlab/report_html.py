"""离线自包含 HTML 报告（v0.9）：一个文件带走全部结论。

设计约束：

* **自包含**：图片以 base64 内嵌，样式内联，**不引用任何外部资源**（无 CDN、无字体、无 JS 依赖），
  断网可看、拷给别人也能看、也不会因为外链失效而变形；
* **可复核**：报告头部固定写下数据指纹、样本区间、生成时间与运行参数——结论和"当时的数据"绑在一起；
* **不做视觉夸大**：只用可读的排版与表格，不搞仪表盘式的装饰。
"""

from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

__all__ = ["render_html", "write_html_report"]

_STYLE = """
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       margin: 0 auto; max-width: 960px; padding: 28px 20px 60px; color: #1f2328; line-height: 1.55; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 10px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f6f8fa; }
img { max-width: 100%; height: auto; margin: 8px 0 4px; border: 1px solid #e5e7eb; }
.meta { font-size: 12px; color: #57606a; }
.meta code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
.note { font-size: 12px; color: #57606a; margin-top: 6px; }
"""


def _image_tag(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:{mime};base64,{payload}">'


def _table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "<p class=\"note\">no rows</p>"
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(
    title: str,
    summary: pd.DataFrame,
    images: Iterable[str | Path] = (),
    meta: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> str:
    """把指标表 + 内嵌图片渲染成一个自包含 HTML 字符串。"""
    meta = dict(meta or {})
    parts = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
             f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p class=\"note\">generated at {html.escape(meta.pop('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))}</p>"]
    if meta:
        items = "".join(
            f"<div><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</div>" for key, value in meta.items()
        )
        parts.append(f"<div class=\"meta\">{items}</div>")
    parts.append("<h2>Summary</h2>")
    parts.append(_table(summary))
    image_paths = [Path(path) for path in images]
    if image_paths:
        parts.append("<h2>Charts</h2>")
        for path in image_paths:
            if path.exists():
                parts.append(_image_tag(path))
    if notes:
        parts.append(f"<p class=\"note\">{html.escape(notes)}</p>")
    parts.append("</body></html>")
    return "".join(parts)


def write_html_report(
    path: str | Path,
    title: str,
    summary: pd.DataFrame,
    images: Iterable[str | Path] = (),
    meta: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    """写一个自包含 HTML 报告，返回文件路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(title, summary, images=images, meta=meta, notes=notes), encoding="utf-8")
    return target
