# -*- coding: utf-8 -*-
"""把"文档里的关键数字必须与产物一致"做成常驻测试。

**为什么值得**：这几轮反复出现"文档数字与产物脱节"（关波段 10 个 vs 实际 7 个、
产物被冒烟运行覆盖、省略号命令）。人工核对不可靠，因此把可核对的数字固定下来：

* 版本号（`pyproject.toml`）必须与最新 git 标签一致；
* README / 速览文档里出现的证据链数字，必须与 `output/` 下的真实产物一致。

**只在产物存在时校验**（CI 里没有跑全市场研究，`output/` 不在版本控制中），
因此测试在缺少产物时会跳过并说明原因——而不是伪造通过。
"""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT_REAL = ROOT / "output" / "alpha101_ic_real"
OUT = ROOT / "output" / "alpha101_ic"


def _has_outputs() -> bool:
    return (OUT_REAL / "corrected.csv").exists() and (OUT / "walk_forward_summary.csv").exists()


requires_outputs = pytest.mark.skipif(
    not _has_outputs(),
    reason="需要先跑真实全市场研究脚本（output/alpha101_ic_real 与 output/alpha101_ic）",
)


def test_version_matches_the_latest_git_tag():
    """pyproject 的版本号不能落后于标签（曾经出现过 0.30.0 对 v0.30.1）。"""
    import tomllib

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.split()
    if not tags:
        pytest.skip("仓库里还没有标签")
    latest = max(tags, key=lambda name: [int(part) for part in re.findall(r"\d+", name)][:3])
    assert f"v{version}" == latest, f"pyproject 是 {version}，最新标签是 {latest}"


@requires_outputs
def test_documented_evidence_chain_matches_the_outputs():
    """证据链的每一环都必须与产物一致。"""
    corrected = pd.read_csv(OUT_REAL / "corrected.csv")
    walkforward = pd.read_csv(OUT / "walk_forward_summary.csv")
    costs = pd.read_csv(OUT / "alpha_portfolio_costs.csv")
    isoos = pd.read_csv(OUT_REAL / "is_oos.csv")

    chain = {
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
    }

    # 文档中必须出现的数字（速览与自述文档）
    docs = {
        ROOT / "docs" / "ALPHA_RESEARCH.md",
        ROOT.parent.parent / "work" / "找工作" / "项目速览-ashare-quant-lab.md",
    }
    for path in docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for key, value in chain.items():
            assert str(value) in text, f"{path.name} 缺少证据链数字 {key}={value}"

    # 关键事实：折 4 全灭而其它折有存活
    detail = pd.read_csv(OUT / "walk_forward_detail.csv")
    fold4 = detail[detail["fold"] == 4]
    assert int(fold4["selected"].sum()) > 0 and int(fold4["survived"].sum()) == 0


@requires_outputs
def test_state_claim_is_seven_of_ten_not_all_ten():
    """关波段更优的因子数是 7/10，文档不得写成"全部"（曾经写错过）。"""
    state = pd.read_csv(OUT / "state_regime_bucket.csv")
    better = 0
    for _factor, group in state.groupby("factor"):
        row = group.set_index("state")
        if {"开波段", "关波段"} <= set(row.index) and float(row.loc["关波段", "ic_mean"]) > float(row.loc["开波段", "ic_mean"]):
            better += 1
    assert better == 7, f"实际 {better} 个因子关波段更优"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "全部 10 个因子" not in readme, "README 不该再声称全部 10 个因子"
    assert "10 个因子里**7 个**" in readme or "7 个" in readme
