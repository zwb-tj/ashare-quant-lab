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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_REPO = "https://github.com/zwb-tj/ashare-quant-lab.git"
EXPECTED_SCRIPTS = (
    "fetch_universe_daily.py",
    "alpha_ic_real.py",
    "alpha_walk_forward.py",
    "alpha_cost_check.py",
    "alpha_state_dependence.py",
    "reproduce_all.py",
)


def run(label: str, args: list[str], cwd: Path, timeout: int = 1800) -> tuple[bool, str]:
    start = time.time()
    completed = subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )
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
    failures = 0
    try:
        started = time.time()
        completed = subprocess.run(
            ["git", "clone", "--quiet", args.repo, str(clone)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        print(f"{'OK  ' if completed.returncode == 0 else 'FAIL'} git clone: {time.time() - started:.1f}s")
        if completed.returncode != 0:
            print(completed.stderr[-300:])
            return 1

        ok, _ = run("pip install -e .[dev]", [sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"], clone, 1200)
        failures += 0 if ok else 1
        ok, _ = run("pytest -q", [sys.executable, "-m", "pytest", "-q"], clone)
        failures += 0 if ok else 1
        ok, _ = run("reproduce_all.py --verify", [sys.executable, "scripts/reproduce_all.py", "--verify"], clone)
        failures += 0 if ok else 1
        ok, _ = run(
            "factor-ic (synthetic)",
            [sys.executable, "-m", "aqlab.cli", "factor-ic", "--symbols", "30", "--days", "500", "--out", "output"],
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
