"""密钥不得进入会被提交的文件的测试（离线）。

**真实事故**：用户把真实 API key 填进了 `.env.example`——该文件被 git 跟踪
（`.gitignore` 里有 `!.env.example` 例外），一旦提交就会公开泄露。
当时尚未提交，属于运气；这条测试把"运气"换成"机制"。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"

# 值必须为空的敏感变量（它们会被提交）
MUST_BE_EMPTY = ("AQLAB_LLM_API_KEY", "TUSHARE_TOKEN", "FEISHU_WEBHOOK", "FEISHU_WEBHOOK_SECRET")

# 看起来像真实密钥的形状：足够长且不是明显的占位符
SECRET_LIKE = re.compile(r"^(sk-[A-Za-z0-9_\-]{16,}|[A-Fa-f0-9]{32,}|[A-Za-z0-9]{32,})$")


def _values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_env_example_keeps_sensitive_values_empty():
    """`.env.example` 里的敏感变量必须留空——它会被提交。"""
    values = _values(EXAMPLE)
    filled = {key: len(values[key]) for key in MUST_BE_EMPTY if values.get(key)}
    assert not filled, (
        f".env.example 里有非空的敏感值（长度：{filled}）。"
        "请把密钥放到 .env（已被 .gitignore 忽略），示例文件只留占位符。"
    )


def test_env_example_contains_no_secret_looking_value():
    """任何键的值都不该长得像真实密钥（防止将来加了新变量又在示例里填真值）。"""
    offenders = [
        key
        for key, value in _values(EXAMPLE).items()
        if value and SECRET_LIKE.match(value) and not key.endswith(("_URL", "_MODEL", "_START"))
    ]
    assert not offenders, f"这些变量在示例文件里像真实密钥：{offenders}"


def test_gitignore_keeps_env_ignored_but_example_tracked():
    """`.env` 必须被忽略、`.env.example` 必须可提交——这个组合是安全的前提。"""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^\.env$", gitignore, flags=re.M), ".gitignore 应当忽略 .env"
    assert "!.env.example" in gitignore, ".gitignore 应当为 .env.example 开例外"
