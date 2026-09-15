# -*- coding: utf-8 -*-
"""扫描工作区与 git 历史里是否出现疑似密钥。

**为什么入库**：把真实 key 填进会被提交的文件是个很容易犯的错（曾经发生过一次）。
这里用**形状**判断（前缀 + 长度），因此不依赖具体密钥值，换了 key 也能拦住。

用法::

    python scripts/scan_secrets.py            # 只扫工作区与暂存区
    python scripts/scan_secrets.py --history  # 额外扫描 git 历史的提交前缀

退出码非零表示发现疑似泄露。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 会被提交的文件（示例文件、文档、代码）
SCAN_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".example", ".cfg", ".txt")
SKIP_PARTS = {"__pycache__", ".git", ".venv", "venv"}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
)

# 明显的测试/示例标记：扫描器无法判断"这个值是真的吗"，但这些词足以说明它是故意写的假值。
# 把它们排除掉，否则测试文件里的假密钥会造成固定误报，把真问题的信噪比拉低。
FAKE_MARKERS = ("example", "xxx", "fake", "dummy", "placeholder", "test-secret", "should-never", "not-a-real")



def _looks_fake(value: str) -> bool:
    """判断匹配到的字串是否明显是示例/测试用的假值。"""
    lowered = value.lower()
    return any(marker in lowered for marker in FAKE_MARKERS)


def scan_working_tree() -> list[str]:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if SKIP_PARTS & set(path.parts) or path.name == ".env":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if _looks_fake(match.group(0)):
                    continue
                findings.append(f"{path.relative_to(ROOT)}: 疑似密钥（{len(match.group(0))} 字符）")
    return findings


def scan_history() -> list[str]:
    result = subprocess.run(
        ["git", "log", "--all", "-p", "--format="], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    findings: list[str] = []
    # 不能对整段 diff 输出做匹配：diff 头部的提交哈希（40 位十六进制）与"疑似密钥"形状相同，
    # 会造成成片误报。因此只取新增内容行（`+` 开头且不是 `+++`），并跳过 diff 元信息。
    for line in result.stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        content = line[1:]
        if content.lstrip().startswith(("index ", "diff --git", "@@", "new file", "deleted file")):
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                if _looks_fake(match.group(0)):
                    continue
                findings.append(f"git 历史新增内容：疑似密钥（{len(match.group(0))} 字符）")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scan for secret-looking values")
    parser.add_argument("--history", action="store_true", help="also scan committed history")
    args = parser.parse_args(argv)

    findings = scan_working_tree()
    print(f"工作区扫描完成：{len(findings)} 处疑似")
    for item in findings:
        print("  -", item)

    if args.history:
        history = scan_history()
        print(f"历史扫描完成：{len(history)} 处疑似")
        for item in history[:20]:
            print("  -", item)
        findings += history

    if findings:
        print()
        print("发现疑似密钥：请确认是否需要在写进 git 之前移除（.env 才是本地配置文件）")
        return 1
    print("未发现疑似密钥")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
