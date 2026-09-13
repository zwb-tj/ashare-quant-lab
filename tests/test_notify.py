import json
import urllib.request

from aqlab.notify import ConsoleNotifier, FeishuWebhookNotifier, build_feishu_card

PICKS = [
    {"rank": 1, "symbol": "AAA", "score": 88.5, "rules": ["needle_below_ma", "volume_price_surge"]},
    {"rank": 2, "symbol": "BBB", "score": 61.2, "rules": []},
]


def test_card_structure_contains_table_and_disclaimer():
    card = build_feishu_card("测试标题", ["第一行", "第二行"], PICKS)
    assert card["msg_type"] == "interactive"
    assert card["card"]["header"]["title"]["content"] == "测试标题"
    contents = [e.get("text", {}).get("content", "") for e in card["card"]["elements"]]
    joined = "\n".join(contents)
    assert "第一行" in joined
    assert "| 排名 | 代码 | 综合分 | 触发规则 |" in joined
    assert "AAA" in joined and "needle_below_ma" in joined
    note = card["card"]["elements"][-1]
    assert note["tag"] == "note"
    assert "不构成投资建议" in note["elements"][0]["content"]


def test_console_notifier_records(capsys):
    notifier = ConsoleNotifier()
    result = notifier.send("标题", ["内容"], PICKS)
    assert result.ok is True and result.channel == "console"
    assert "dry-run" in result.detail
    assert notifier.sent[0]["title"] == "标题"
    assert "内容" in capsys.readouterr().out


def test_feishu_dry_run_does_not_post(monkeypatch):
    def boom(*_args, **_kwargs):  # pragma: no cover - must never be called
        raise AssertionError("dry-run 不应发起网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    result = FeishuWebhookNotifier(webhook_url="https://example.invalid/hook", dry_run=True).send("t", ["l"], PICKS)
    assert result.ok is True
    assert "dry-run" in result.detail
    assert result.payload["msg_type"] == "interactive"


def test_feishu_missing_webhook_fails_cleanly(monkeypatch):
    monkeypatch.delenv("FEISHU_WEBHOOK", raising=False)
    result = FeishuWebhookNotifier(dry_run=False).send("t", ["l"], PICKS)
    assert result.ok is False
    assert "未配置" in result.detail


def test_feishu_secret_refuses_unsigned_post():
    result = FeishuWebhookNotifier(webhook_url="https://example.invalid/hook", secret="s3cret").send("t", ["l"], PICKS)
    assert result.ok is False
    assert "签名" in result.detail


def test_feishu_post_success_and_failure(monkeypatch):
    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = {"n": 0}

    def ok_urlopen(request, timeout=None):
        calls["n"] += 1
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["msg_type"] == "interactive"
        return FakeResponse(b'{"code": 0, "msg": "success"}')

    monkeypatch.setattr(urllib.request, "urlopen", ok_urlopen)
    result = FeishuWebhookNotifier(webhook_url="https://example.invalid/hook", dry_run=False).send("t", ["l"], PICKS)
    assert result.ok is True and calls["n"] == 1

    def bad_urlopen(request, timeout=None):
        return FakeResponse(b'{"code": 19021, "msg": "sign match fail"}')

    monkeypatch.setattr(urllib.request, "urlopen", bad_urlopen)
    failed = FeishuWebhookNotifier(webhook_url="https://example.invalid/hook", dry_run=False).send("t", ["l"], PICKS)
    assert failed.ok is False
    assert "19021" in failed.detail
