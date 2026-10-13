# -*- coding: utf-8 -*-
"""控制台/子进程编码的守卫（Windows GBK 事故的回归测试）。

事故回顾：``verify_fresh_clone.py`` 用 ``subprocess.run(..., encoding="utf-8")`` 读子进程输出，
但子进程按控制台 locale（cp936）写 stdout——遇到 ``−``（U+2212）时**子进程自己就崩了**
（``returncode=1``、stdout 为空），父进程再解码只拿到替换字符 U+FFFD，打印时又撞上自己的
GBK 编码错误。所以真正要守的是"子进程被显式要求用 UTF-8 说话"，而不是父进程的解码参数。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aqlab.console import run_utf8

NON_ASCII = "− 12.34℃ ≤"


def test_run_utf8_round_trips_non_ascii():
    """子进程打印非 ASCII（含 U+2212）必须完整回传，且不被替换字符污染。"""
    completed = run_utf8([sys.executable, "-c", f"print({NON_ASCII!r})"])
    assert completed.returncode == 0, "子进程不应因编码失败而崩溃"
    assert "\ufffd" not in completed.stdout, "输出里不应出现替换字符"
    assert NON_ASCII in completed.stdout


def test_run_utf8_forces_child_stdout_encoding():
    """确认子进程真的继承了 PYTHONIOENCODING=utf-8，而不是碰巧没报错。"""
    completed = run_utf8([sys.executable, "-c", "import sys; print(sys.stdout.encoding)"])
    assert completed.stdout.strip().lower() == "utf-8"


def test_bare_subprocess_run_is_not_used_to_capture_python_output():
    """scripts/ 下捕获子进程输出的地方都必须走 run_utf8（共享解法，不是逐点补丁）。"""
    offenders: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "subprocess.run(" not in text:
            continue
        # 允许出现 subprocess 模块本身，但不允许再出现绕过 run_utf8 的捕获式调用
        if "capture_output=True" in text and "run_utf8" not in text:
            offenders.append(path.name)
    assert not offenders, f"这些脚本仍在直接捕获子进程输出（应用 aqlab.console.run_utf8）：{offenders}"


def test_scripts_that_print_unicode_make_their_stdout_utf8_safe():
    """会打印含非 ASCII 文案的入口脚本，必须在 main 里调用 ensure_utf8_output。"""
    for name in ("verify_fresh_clone.py", "reproduce_all.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "ensure_utf8_output" in text, f"{name} 应调用 ensure_utf8_output()"
