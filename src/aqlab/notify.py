"""Notification targets for the daily pipeline.

Two implementations ship with the project:

* :class:`ConsoleNotifier` — prints the message; used by ``--dry-run`` and tests;
* :class:`FeishuWebhookNotifier` — posts an interactive card to a Feishu (Lark)
  custom-bot webhook using only the standard library.

The notifier never receives credentials from the model or the agent layer: it is
constructed from environment variables at the edge of the pipeline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

__all__ = ["NotifyResult", "Notifier", "ConsoleNotifier", "FeishuWebhookNotifier", "build_feishu_card"]


@dataclass
class NotifyResult:
    ok: bool
    channel: str
    detail: str = ""
    payload: dict = field(default_factory=dict)


class Notifier(Protocol):
    def send(self, title: str, lines: Sequence[str], picks: Sequence[dict] | None = None) -> NotifyResult: ...


class ConsoleNotifier:
    """Dry-run notifier: prints instead of posting."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, title: str, lines: Sequence[str], picks: Sequence[dict] | None = None) -> NotifyResult:
        body = "\n".join(lines)
        print(f"[dry-run 通知] {title}\n{body}")
        self.sent.append({"title": title, "lines": list(lines), "picks": list(picks or [])})
        return NotifyResult(ok=True, channel="console", detail="dry-run: 未实际发送", payload={"title": title, "lines": list(lines)})


def build_feishu_card(title: str, lines: Sequence[str], picks: Sequence[dict] | None = None) -> dict:
    """Build a Feishu interactive-card payload (schema: custom bot 'interactive' message)."""
    elements: list[dict] = []
    if lines:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}})

    picks = list(picks or [])
    if picks:
        header = "| 排名 | 代码 | 综合分 | 触发规则 |\n| --- | --- | --- | --- |"
        rows = [
            "| {rank} | {symbol} | {score} | {rules} |".format(
                rank=item.get("rank", "-"),
                symbol=item.get("symbol", "-"),
                score=item.get("score", "-"),
                rules="、".join(item.get("rules", [])) or "-",
            )
            for item in picks
        ]
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": header + "\n" + "\n".join(rows)}})

    elements.append(
        {
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": "aqlab 自动推送 · 仅研究用途，不构成投资建议"}],
        }
    )
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": elements,
        },
    }


class FeishuWebhookNotifier:
    """Post to a Feishu custom-bot webhook.

    Parameters
    ----------
    webhook_url:
        Falls back to the ``FEISHU_WEBHOOK`` environment variable.
    dry_run:
        When true the payload is built and returned but never posted.
    secret:
        Optional signing secret (``FEISHU_WEBHOOK_SECRET``). Feishu signing is not
        implemented yet; supplying one raises rather than silently posting unsigned.
    """

    def __init__(self, webhook_url: str | None = None, dry_run: bool = False, secret: str | None = None, timeout: int = 15) -> None:
        self.webhook_url = webhook_url or os.environ.get("FEISHU_WEBHOOK")
        self.dry_run = dry_run
        self.secret = secret or os.environ.get("FEISHU_WEBHOOK_SECRET")
        self.timeout = timeout

    def send(self, title: str, lines: Sequence[str], picks: Sequence[dict] | None = None) -> NotifyResult:
        payload = build_feishu_card(title, lines, picks)
        if self.secret:
            return NotifyResult(ok=False, channel="feishu", detail="签名校验(secret)尚未实现，请使用不签名的机器人或先实现签名", payload=payload)
        if self.dry_run:
            print(f"[dry-run 飞书] 将发送卡片：{title}")
            print(json.dumps(payload, ensure_ascii=False)[:600] + ("..." if len(json.dumps(payload)) > 600 else ""))
            return NotifyResult(ok=True, channel="feishu", detail="dry-run: 未实际发送", payload=payload)
        if not self.webhook_url:
            return NotifyResult(ok=False, channel="feishu", detail="未配置 FEISHU_WEBHOOK", payload=payload)

        request = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
            parsed: Any = {}
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                pass
            code = parsed.get("code", parsed.get("StatusCode", 0)) if isinstance(parsed, dict) else 0
            ok = code in (0, None)
            return NotifyResult(ok=ok, channel="feishu", detail=body[:300], payload=payload)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            return NotifyResult(ok=False, channel="feishu", detail=f"HTTP {exc.code}", payload=payload)
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            return NotifyResult(ok=False, channel="feishu", detail=f"connection error: {exc.reason}", payload=payload)
