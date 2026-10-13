# -*- coding: utf-8 -*-
"""控制台与子进程的编码处理——Windows GBK 控制台的共享解法。

**为什么需要**：Windows 控制台默认 GBK，而项目大量输出 `−`（U+2212）、`≤`、`℃` 这类
字符。这会在两个地方同时出问题，而且是**同一个根因**：

① 自己打印时抛 ``UnicodeEncodeError``；
② 更隐蔽的是**子进程**：它按继承到的 GBK 写 stdout，遇到 `−` 时子进程自己就崩了
   （``returncode=1``、stdout 为空），父进程再解码只会拿到替换字符 U+FFFD，然后
   父进程打印这个替换字符又会撞上自己的 GBK 编码错误。

所以修法必须是"给子进程显式设 ``PYTHONIOENCODING=utf-8``"＋"把自己的输出也切成
UTF-8"，而不是只调整父进程的解码参数——那样只是把子进程的崩溃掩盖成空输出。
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["ensure_utf8_output", "run_utf8"]


def ensure_utf8_output() -> None:
    """把自身 stdout/stderr 切成 UTF-8；切不动就退化成替换字符，保证报告能打印完。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def run_utf8(
    args: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """跑一个子进程，两端都按 UTF-8 处理（见模块文档说明）。"""
    env: dict[str, Any] = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
