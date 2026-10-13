# -*- coding: utf-8 -*-
"""因子证据链的可提交清单 + 自查（`--check` 不一致时退出非零）。

背景：README / ALPHA_RESEARCH / CASE_STUDY 的核心数字（41→35→28、11/16、12、3……）
来自 ``output/`` 下的真实全市场产物，而 ``output/`` 是 gitignored 的。
结果是**在全新克隆里那些断言会整体跳过**——文档声称的东西在最需要被检查的地方没人检查。

所以这里把证据链数字落成一份**可提交**的清单（``docs/research/alpha-evidence-chain.json``），
并在仓库内固定下来。清单由真实产物生成，不是手抄的。

用法::

    python scripts/alpha_evidence_chain.py            # 与产物对照，不一致则退出 1
    python scripts/alpha_evidence_chain.py --write    # 用当前产物重新生成清单（需要真实数据产物）
    python scripts/alpha_evidence_chain.py --print     # 只打印当前清单

``--check`` 是默认行为。它核对的是"清单 == 产物"，所以能抓住两类漂移：
产物被重算后清单过期、以及有人手改了清单里的数字。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "docs" / "research" / "alpha-evidence-chain.json"

REAL_DIR = ROOT / "output" / "alpha101_ic_real"

COMMANDS = [
    "python scripts/fetch_universe_daily.py",
    "python scripts/alpha_ic_real.py",
    "python scripts/alpha_walk_forward.py",
    "python scripts/alpha_cost_check.py",
    "python scripts/alpha_state_dependence.py",
]


def build_manifest() -> dict[str, Any]:
    """从真实产物算出证据链清单。产物缺失时抛 FileNotFoundError。"""
    import pandas as pd

    corrected = pd.read_csv(REAL_DIR / "corrected.csv")
    isoos = pd.read_csv(REAL_DIR / "is_oos.csv")
    walkforward = pd.read_csv(REAL_DIR / "walk_forward_summary.csv")
    detail = pd.read_csv(REAL_DIR / "walk_forward_detail.csv")
    costs = pd.read_csv(REAL_DIR / "alpha_portfolio_costs.csv")
    state = pd.read_csv(REAL_DIR / "state_regime_bucket.csv")

    fold4 = detail[detail["fold"] == 4]

    better_in_closed = 0
    compared = 0
    for _factor, group in state.groupby("factor"):
        row = group.set_index("state")
        if {"开波段", "关波段"} <= set(row.index):
            compared += 1
            if float(row.loc["关波段", "ic_mean"]) > float(row.loc["开波段", "ic_mean"]):
                better_in_closed += 1

    # 未实现的因子是**代码的属性**（缺行业/市值数据），不依赖任何产物文件
    sys.path.insert(0, str(ROOT / "src"))
    from aqlab.alpha101 import SKIPPED

    skipped = sorted(SKIPPED)

    return {
        "_note": (
            "由 scripts/alpha_evidence_chain.py --write 生成；真实产物的数字以此为准。"
            "核验：python scripts/alpha_evidence_chain.py"
        ),
        "provenance": {
            "universe": "全市场 A 股日线（stockdb 导出至 data/universe/daily）",
            "reproduce": COMMANDS,
            "artifacts": [
                "output/alpha101_ic_real/corrected.csv",
                "output/alpha101_ic_real/is_oos.csv",
                "output/alpha101_ic_real/walk_forward_summary.csv",
                "output/alpha101_ic_real/walk_forward_detail.csv",
                "output/alpha101_ic_real/alpha_portfolio_costs.csv",
                "output/alpha101_ic_real/state_regime_bucket.csv",
            ],
            "note": "产物不入库（体积与数据许可原因），因此 CI 校验的是本清单与文档，而不是重跑数据。",
        },
        "evidence_chain": {
            "combinations": len(corrected),
            "raw_significant": int(corrected["significant_raw"].sum()),
            "bh_significant": int(corrected["significant_bh"].sum()),
            "bonferroni_significant": int(corrected["significant_bonferroni"].sum()),
            "in_sample_significant": int(isoos["is_significant_bh"].sum()),
            "out_of_sample_survivors": int(isoos[isoos["is_significant_bh"]]["survives_oos"].sum()),
            "walkforward_combinations": len(walkforward),
            "walkforward_robust": int(walkforward["robust"].sum()),
            "costed_combinations": len(costs),
            "net_excess_t_above_2": int((costs["net_excess_t"] > 2).sum()),
        },
        "walk_forward": {
            "folds": sorted(int(value) for value in detail["fold"].unique()),
            "fold4_selected": int(fold4["selected"].sum()),
            "fold4_survived": int(fold4["survived"].sum()),
        },
        "cost_check": {
            "median_break_even_bps": round(float(costs["break_even_bps"].median()), 1),
            "mean_turnover": round(float(costs["mean_turnover"].mean()), 3),
        },
        "state_dependence": {
            "factors_compared": compared,
            "factors_better_in_closed_band": better_in_closed,
        },
        "skipped": {"count": len(skipped), "factors": skipped},
    }


def compare(manifest: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """逐项比较清单与产物，返回不一致描述（空列表 = 一致）。"""
    problems: list[str] = []
    for section in ("evidence_chain", "walk_forward", "cost_check", "state_dependence", "skipped"):
        documented = manifest.get(section, {})
        measured = actual.get(section, {})
        for key, value in measured.items():
            if key not in documented:
                problems.append(f"{section}.{key} 缺失（产物为 {value}）")
            elif documented[key] != value:
                problems.append(f"{section}.{key} 不一致：清单 {documented[key]} vs 产物 {value}")
        for key in documented:
            if key not in measured:
                problems.append(f"{section}.{key} 在产物中不存在（清单为 {documented[key]}）")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="committed evidence-chain manifest for the factor study")
    parser.add_argument("--write", action="store_true", help="用当前产物重新生成清单")
    parser.add_argument("--print", dest="show", action="store_true", help="只打印清单")
    args = parser.parse_args(argv)

    if args.show:
        print(MANIFEST.read_text(encoding="utf-8"))
        return 0

    if args.write:
        try:
            actual = build_manifest()
        except FileNotFoundError as exc:
            print(f"产物缺失，无法重新生成清单：{exc}")
            return 1
        MANIFEST.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
        print(f"已写入 {MANIFEST.relative_to(ROOT)}")
        return 0

    if not MANIFEST.is_file():
        print(f"缺少清单：{MANIFEST}（用 --write 生成，需要真实产物）")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    try:
        actual = build_manifest()
    except FileNotFoundError as exc:
        print(f"提示：产物缺失（{exc}），跳过与产物的对照；清单本身仍由测试与文档核对。")
        return 0

    problems = compare(manifest, actual)
    if problems:
        print("=== 证据链清单与产物不一致 ===")
        for item in problems:
            print(f"FAIL {item}")
        print()
        print("如产物是刚重算的，运行 `python scripts/alpha_evidence_chain.py --write` 更新清单。")
        return 1

    print("=== 证据链清单 ===")
    for key, value in manifest["evidence_chain"].items():
        print(f"OK  {key} = {value}")
    print(f"OK  折 4：选中 {manifest['walk_forward']['fold4_selected']} / 存活 {manifest['walk_forward']['fold4_survived']}")
    print(f"OK  关波段更优 {manifest['state_dependence']['factors_better_in_closed_band']}"
          f" / {manifest['state_dependence']['factors_compared']}")
    print()
    print("清单与产物一致。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
