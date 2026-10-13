# -*- coding: utf-8 -*-
"""全新克隆验证：在一个临时目录里 git clone 本仓库，跑通"从零开始"承诺的每一步。

与 `reproduce_all.py --verify` 的区别：那条命令在**本机工作副本**里验证图表可复现，
本脚本验证的是**外部使用者拿到的全新克隆**——没有 `data/`（真实行情缓存，不入库）、
没有 `output/`（研究产物，不入库），因此能暴露"依赖本机残留文件"这类问题。

用法::

    python scripts/verify_fresh_clone.py [--repo URL] [--keep]

退出码非零表示某一步失败（克隆 / 安装 / 测试 / 图表校验 / 研究命令）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqlab.console import ensure_utf8_output, run_utf8

DEFAULT_REPO = "https://github.com/zwb-tj/ashare-quant-lab.git"
EXPECTED_SCRIPTS = (
    "fetch_universe_daily.py",
    "alpha_ic_real.py",
    "alpha_walk_forward.py",
    "alpha_cost_check.py",
    "alpha_state_dependence.py",
    "alpha_evidence_chain.py",
    "reproduce_all.py",
)


def run(label: str, args: list[str], cwd: Path, timeout: int = 1800) -> tuple[bool, str]:
    start = time.time()
    completed = run_utf8(args, cwd=cwd, timeout=timeout)
    elapsed = time.time() - start
    tail = " | ".join(line.strip() for line in (completed.stdout or "").strip().splitlines()[-3:])
    ok = completed.returncode == 0
    detail = f"{elapsed:.1f}s｜exit={completed.returncode}｜{tail[:150]}"
    print(f"{'OK  ' if ok else 'FAIL'} {label}: {detail}", flush=True)
    return ok, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify that a fresh clone reproduces the documented steps")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--keep", action="store_true", help="keep the temporary clone for inspection")
    args = parser.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="aqlab-fresh-"))
    clone = root / "repo"
    venv = root / "venv"
    failures = 0
    ensure_utf8_output()
    try:
        started = time.time()
        completed = run_utf8(["git", "clone", "--quiet", args.repo, str(clone)], timeout=600)
        print(f"{'OK  ' if completed.returncode == 0 else 'FAIL'} git clone: {time.time() - started:.1f}s")
        if completed.returncode != 0:
            print(completed.stderr[-300:])
            return 1

        # 必须在**独立虚拟环境**里安装：对本机解释器做 `pip install -e` 会改写它的 aqlab
        # 指向临时克隆，临时目录删除后本机 import aqlab 就失效（本脚本曾这样破坏过自身环境）。
        # 独立 venv 也更诚实——否则"全新克隆"其实在用一个已装好 aqlab 的解释器跑测试。
        ok, _ = run("venv 创建", [sys.executable, "-m", "venv", str(venv)], root, 300)
        failures += 0 if ok else 1
        if not ok:
            return 1
        venv_python = venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

        # 独立 venv 是空的，依赖要真的从索引下载；慢网络下这一步可能几十分钟（本机实测
        # 单独装 numpy 就要约 440s），所以超时给得比"复用本机已装依赖"的旧写法宽得多。
        ok, _ = run("pip install -e .[dev]", [str(venv_python), "-m", "pip", "install", "-q", "-e", ".[dev]"], clone, 5400)
        failures += 0 if ok else 1
        ok, _ = run("pytest -q", [str(venv_python), "-m", "pytest", "-q"], clone)
        failures += 0 if ok else 1
        ok, _ = run("reproduce_all.py --verify", [str(venv_python), "scripts/reproduce_all.py", "--verify"], clone)
        failures += 0 if ok else 1
        ok, _ = run(
            "factor-ic (synthetic)",
            [str(venv_python), "-m", "aqlab.cli", "factor-ic", "--symbols", "30", "--days", "500", "--out", "output"],
            clone,
            900,
        )
        failures += 0 if ok else 1

        missing = [name for name in EXPECTED_SCRIPTS if not (clone / "scripts" / name).exists()]
        if missing:
            print(f"FAIL 缺少脚本：{missing}")
            failures += 1
        else:
            print(f"OK   脚本齐全（{len(EXPECTED_SCRIPTS)} 个）")

        print()
        print("全新克隆验证：" + ("全部通过" if failures == 0 else f"{failures} 项失败"))
        return 1 if failures else 0
    finally:
        if args.keep:
            print(f"临时克隆保留在：{clone}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
