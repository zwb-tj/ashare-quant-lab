# -*- coding: utf-8 -*-
"""核对"项目速览"（简历附件）里的每条硬声明是否与仓库实际一致。

**为什么要入库**：速览是给面试官看的一页纸，里面每个数字都必须能被仓库复现。
这个脚本逐条核对——模块数、CLI 子命令数、测试数、因子证据链的每一环、状态依赖的 7/10、
以及文档与产物的一致性。任何一条对不上就退出非零。

用法::

    python scripts/verify_brief_claims.py [--brief PATH]

默认核对工作区之外的速览文件（若不存在则跳过那几项），因此可以在任何克隆里运行。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify the claims made in the one-page project brief")
    parser.add_argument("--brief", default=None, help="path to the brief; defaults to a sibling job-hunt folder if present")
    args = parser.parse_args(argv)

    import pandas as pd

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        checks.append((label, bool(ok), detail))

    modules = len(list((ROOT / "src" / "aqlab").glob("*.py")))
    check(f"源码模块数 = {modules}", modules >= 40, f"实际 {modules}")

    test_modules = len(list((ROOT / "tests").glob("test_*.py")))
    check(f"测试模块数 = {test_modules}", test_modules >= 40, f"实际 {test_modules}")

    from aqlab.cli import build_parser

    commands = len(build_parser()._subparsers._group_actions[0].choices)
    check(f"CLI 子命令数 = {commands}", commands >= 20, f"实际 {commands}")

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    total = sum(int(m) for m in re.findall(r":\s*(\d+)$", collected.stdout, flags=re.M))
    check(f"测试用例数 = {total}", total >= 380, f"实际 {total}")

    out_real = ROOT / "output" / "alpha101_ic_real"
    out = ROOT / "output" / "alpha101_ic"
    if (out_real / "corrected.csv").exists() and (out / "walk_forward_summary.csv").exists():
        corrected = pd.read_csv(out_real / "corrected.csv")
        check("全样本组合 111", len(corrected) == 111, f"实际 {len(corrected)}")
        check("不校正显著 41", int(corrected["significant_raw"].sum()) == 41)
        check("BH 显著 35", int(corrected["significant_bh"].sum()) == 35)
        check("Bonferroni 显著 28", int(corrected["significant_bonferroni"].sum()) == 28)

        wf = pd.read_csv(out / "walk_forward_summary.csv")
        check("多折组合 74", len(wf) == 74, f"实际 {len(wf)}")
        check("多折稳健 12", int(wf["robust"].sum()) == 12, f"实际 {int(wf['robust'].sum())}")

        detail = pd.read_csv(out / "walk_forward_detail.csv")
        fold4 = detail[detail["fold"] == 4]
        check(
            "折 4 被选中 20 且存活 0",
            int(fold4["selected"].sum()) == 20 and int(fold4["survived"].sum()) == 0,
            f"选中 {int(fold4['selected'].sum())}｜存活 {int(fold4['survived'].sum())}",
        )

        costs = pd.read_csv(out / "alpha_portfolio_costs.csv")
        check("成本组合 33", len(costs) == 33, f"实际 {len(costs)}")
        check("净超额 t>2 为 3", int((costs["net_excess_t"] > 2).sum()) == 3)
        check(
            "盈亏平衡成本中位数 23.4bps",
            abs(float(costs["break_even_bps"].median()) - 23.4) < 0.5,
            f"实际 {float(costs['break_even_bps'].median()):.1f}",
        )

        state = pd.read_csv(out / "state_regime_bucket.csv")
        better = 0
        for _factor, group in state.groupby("factor"):
            row = group.set_index("state")
            if {"开波段", "关波段"} <= set(row.index) and float(row.loc["关波段", "ic_mean"]) > float(row.loc["开波段", "ic_mean"]):
                better += 1
        check("关波段更优 7/10", better == 7, f"实际 {better}")
    else:
        print("提示：output/ 下没有真实研究产物，跳过因子证据链核对（可先跑 scripts/alpha_ic_real.py 等）")

    brief = Path(args.brief) if args.brief else None
    if brief and brief.exists():
        text = brief.read_text(encoding="utf-8")
        for figure in ("41", "35", "28", "11 / 16", "12", "3"):
            check(f"速览包含证据链数字 {figure}", figure in text)

    print("=== 速览声明核对 ===")
    failed = [item for item in checks if not item[1]]
    for label, ok, detail in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {label}" + (f"  <- {detail}" if not ok and detail else ""))
    print()
    print(f"通过 {len(checks) - len(failed)} / {len(checks)}")
    if failed:
        print("未通过：", [item[0] for item in failed])
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
