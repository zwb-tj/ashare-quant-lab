# -*- coding: utf-8 -*-
"""把因子证据链钉在一份**可提交**的清单上（不依赖 `output/`）。

落差在哪：`docs/ALPHA_RESEARCH.md` / README / `docs/CASE_STUDY.md` 的核心数字
（41 → 35 → 28、11/16、12/74、3/18）来自真实全市场产物，而产物写在 gitignored 的
`output/` 下。原先校验它们的断言都挂在"产物存在"的前提后面，于是在**全新克隆与 CI 里
整体静默跳过** —— 文档最核心的声明，恰好是最需要被检查的地方没人检查。

`docs/research/alpha-evidence-chain.json` 由 `scripts/alpha_evidence_chain.py --write`
从真实产物生成并**入库**（与 `brick-research-results.json` 同一套做法）。本文件不重写那些
数字，而是从清单读、断言文档里出现同值：产物重算导致清单过期、或有人手改清单，都会失败。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "alpha-evidence-chain.json"
SCRIPT = ROOT / "scripts" / "alpha_evidence_chain.py"

ALPHA_DOC = ROOT / "docs" / "ALPHA_RESEARCH.md"
CASE_STUDY = ROOT / "docs" / "CASE_STUDY.md"
README_ZH = ROOT / "README.md"
README_EN = ROOT / "README.en.md"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), (
        "缺少证据链清单——请先跑 `python scripts/alpha_evidence_chain.py --write`（需要真实产物）"
    )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---- 清单自身 ----------------------------------------------------------------------


def test_manifest_records_how_it_was_generated(manifest: dict) -> None:
    """清单必须自带来源与核验方式，否则读者无法判断数字可信度。"""
    note = manifest["_note"]
    assert "python scripts/alpha_evidence_chain.py" in note, "清单应写明核验命令"
    assert "真实产物的数字以此为准" in note, "清单应写明与产物的优先关系"
    assert manifest["provenance"]["note"], "清单应说明产物为何不入库"


def test_provenance_reproduce_commands_all_reference_existing_scripts(manifest: dict) -> None:
    """清单里的复现命令必须都指向真实存在的脚本。"""
    commands = manifest["provenance"]["reproduce"]
    assert commands, "清单应给出完整复现命令"
    for command in commands:
        match = re.match(r"python (scripts/[A-Za-z0-9_]+\.py)", command)
        assert match, f"复现命令格式不对：{command}"
        assert (ROOT / match.group(1)).is_file(), f"复现命令引用了不存在的脚本：{command}"
        assert "..." not in command, f"复现命令不应带省略号：{command}"


def test_manifest_artifacts_point_at_the_single_research_output_dir(manifest: dict) -> None:
    """研究脚本都写 `output/alpha101_ic_real/`；清单不得引用旧的目录名。

    这条守的是一个真实发生过的错：三个研究脚本改了输出目录，
    而清单/文档仍指向旧目录 `output/alpha101_ic/`，导致"数字从哪来"无法对账。
    """
    for artifact in manifest["provenance"]["artifacts"]:
        assert artifact.startswith("output/alpha101_ic_real/"), f"清单引用了过期目录：{artifact}"


# ---- 代码派生的部分：不需要任何产物也能查 ------------------------------------------


def test_skipped_factors_match_the_implementation(manifest: dict) -> None:
    """跳过清单是**代码的属性**（缺行业/市值数据），应在任何环境下都能核对。"""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from aqlab.alpha101 import ALPHAS, SKIPPED

    assert sorted(SKIPPED) == manifest["skipped"]["factors"], "清单的跳过因子与代码不一致"
    assert manifest["skipped"]["count"] == len(SKIPPED)
    assert len(ALPHAS) == 44, f"实现的因子数变了：{len(ALPHAS)}"
    assert len(ALPHAS) + len(SKIPPED) == 66, "实现数 + 跳过数应覆盖 Alpha101 子集全集"


def test_readme_states_the_implemented_and_skipped_counts(manifest: dict) -> None:
    """README（中英）声明的因子数与跳过数必须与代码一致。"""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from aqlab.alpha101 import ALPHAS, SKIPPED

    zh = _doc(README_ZH)
    assert f"Alpha101 子集 {len(ALPHAS)} 个" in zh, f"中文 README 应写明实现 {len(ALPHAS)} 个"
    assert f"{len(SKIPPED)} 个显式跳过" in zh, f"中文 README 应写明跳过 {len(SKIPPED)} 个"

    en = _doc(README_EN)
    assert f"{len(ALPHAS)} implemented" in en, f"英文 README 应写明实现 {len(ALPHAS)} 个"
    assert f"{len(SKIPPED)} needing industry or market-cap data explicitly skipped" in en, (
        f"英文 README 应写明跳过 {len(SKIPPED)} 个"
    )


# ---- 证据链数字：清单 → 文档 --------------------------------------------------------


def test_alpha_research_doc_matches_the_manifest_chain(manifest: dict) -> None:
    """自述文档的每一环都必须与清单一致（这是整条证据链的骨架）。"""
    chain = manifest["evidence_chain"]
    text = _doc(ALPHA_DOC)

    assert f"共 **{chain['combinations']} 个 (因子 × 持有期) 组合**" in text
    for label, value in (
        ("不校正", chain["raw_significant"]),
        ("BH-FDR", chain["bh_significant"]),
        ("Bonferroni", chain["bonferroni_significant"]),
    ):
        assert re.search(rf"\|.*{label}.*\|\s*\**{value}\**\s*\|", text), (
            f"文档的「{label}」一行应写 {value}"
        )

    assert f"样本内 BH 校正后显著 | **{chain['in_sample_significant']}**" in text
    assert f"**{chain['out_of_sample_survivors']} / {chain['in_sample_significant']}**" in text

    wf = manifest["walk_forward"]
    assert f"**{chain['walkforward_robust']}**" in text
    assert f"**{chain['walkforward_combinations']} 个 (因子 × 持有期) 组合**" in text
    assert "折 4（测试段" in text
    assert f"被选中的 {wf['fold4_selected']} 个组合，存活 {wf['fold4_survived']} 个" in text, (
        "折 4 全灭是文档里最重要的一条负面结论，数字必须与清单一致"
    )

    costs = manifest["cost_check"]
    assert f"| **净超额 t > 2** | **{chain['net_excess_t_above_2']} / {chain['costed_combinations']}** |" in text
    assert f"| 平均换手 | **{costs['mean_turnover']}** |" in text
    assert f"| 盈亏平衡成本中位数 | **{costs['median_break_even_bps']} bps** |" in text

    state = manifest["state_dependence"]
    assert f"10 个因子里 **{state['factors_better_in_closed_band']} 个**在**关波段**下" in text


def test_chinese_readme_chain_matches_the_manifest(manifest: dict) -> None:
    chain = manifest["evidence_chain"]
    text = _doc(README_ZH)
    assert f"不校正 {chain['raw_significant']} 个显著" in text
    assert f"BH-FDR {chain['bh_significant']}" in text
    assert f"Bonferroni {chain['bonferroni_significant']}" in text
    assert f"样本外存活 {chain['out_of_sample_survivors']}/{chain['in_sample_significant']}" in text
    assert (
        f"扣 20bps 成本后净超额 t>2 剩 **{chain['net_excess_t_above_2']}/{chain['costed_combinations']}**" in text
    ), "中文 README 亮点表的最后一环数字过期"
    assert f"四者交集 **{manifest['gate_overlap']['all_four_gates']} 个**" in text


def test_english_readme_chain_matches_the_manifest(manifest: dict) -> None:
    chain = manifest["evidence_chain"]
    text = _doc(README_EN)
    assert f"{chain['raw_significant']} raw significant cells" in text
    assert f"{chain['bh_significant']} (BH-FDR)" in text
    assert f"{chain['bonferroni_significant']} (Bonferroni)" in text
    assert f"{chain['out_of_sample_survivors']}/{chain['in_sample_significant']} surviving" in text
    assert f"**{chain['net_excess_t_above_2']}/{chain['costed_combinations']}**" in text


def test_case_study_chain_matches_the_manifest(manifest: dict) -> None:
    chain = manifest["evidence_chain"]
    overlap = manifest["gate_overlap"]
    wf = manifest["walk_forward"]
    costs = manifest["cost_check"]
    text = _doc(CASE_STUDY)

    assert f"| Raw IC significant at 5%, no correction ({chain['combinations']} factor-horizon cells) |" in text
    assert f"| Multiple-comparison control (BH-FDR) | {chain['combinations']} | 1 / 5 / 20 | {chain['bh_significant']} |" in text
    assert f"| Bonferroni | {chain['combinations']} | 1 / 5 / 20 | {chain['bonferroni_significant']} |" in text
    assert (
        f"single 50/50 split) | {chain['combinations']} | 1 / 5 / 20 | "
        f"{chain['out_of_sample_survivors']} of {chain['in_sample_significant']} |" in text
    )
    assert (
        f">=2 folds selected, >=60% survived) | {chain['walkforward_combinations']} | "
        f"**5 / 20 only** | {chain['walkforward_robust']} |" in text
    )
    assert (
        f"net-excess t > 2 | {chain['costed_combinations']} (out-of-sample survivors ∪ walk-forward robust) "
        f"| 1 / 5 / 20 | {chain['net_excess_t_above_2']} |" in text
    )
    assert f"Taking all four gates together leaves **{overlap['all_four_gates']}** combinations" in text
    assert f"median break-even cost {costs['median_break_even_bps']} bps" in text
    assert f"only {chain['net_excess_t_above_2']} of {chain['costed_combinations']} clear" in text
    assert "each layer removed candidates" not in text, "各闸门候选集不嵌套，不能写成层层收紧"
    assert wf["fold4_survived"] == 0, "折 4 全灭是清单的核心事实，测试依赖它"


def test_gate_overlap_is_recorded_and_not_described_as_a_funnel(manifest: dict) -> None:
    """证据链不是层层收紧的漏斗——这条必须被钉住，否则又会漂回箭头叙述。

    真实关系（来自产物）：多折只跑 5/20 日持有期，成本闸门检验前两者的并集，
    所以链条在 11 → 12 处会**变大**，且四个闸门的交集是 0。
    """
    overlap = manifest["gate_overlap"]
    assert overlap["all_four_gates"] == 0
    assert overlap["robust_and_costed"] == 0, "多折稳健的组合没有一个能扣成本后保住 t>2"
    assert overlap["robust_with_cost_data"] == manifest["evidence_chain"]["walkforward_robust"]
    assert overlap["robust_missing_cost_data"] == []
    assert overlap["bh_and_oos"] == manifest["evidence_chain"]["out_of_sample_survivors"]
    assert overlap["horizons_tested"]["walk_forward"] == [5, 20], "多折不跑 1 日持有期"
    assert 1 in overlap["horizons_tested"]["cost_check"]

    for path in (ALPHA_DOC, README_ZH, README_EN, CASE_STUDY):
        text = _doc(path)
        for banned in ("证据链层层收紧", "the evidence chain tightens layer by layer"):
            assert banned not in text, f"{path.name} 仍把各闸门写成漏斗：{banned}"

    # 光禁止旧措辞不够——还得要求文档**正面说清**它们不嵌套，否则删掉一句话就能让守卫变绿。
    assert "不是嵌套" in _doc(ALPHA_DOC) and "不是一条层层收紧的漏斗" in _doc(ALPHA_DOC)
    assert "取交集，剩下的组合是 **0 个**" in _doc(ALPHA_DOC)
    assert "不是层层收紧的漏斗" in _doc(README_ZH) and "交集是 **0 个**" in _doc(README_ZH)
    assert "not a funnel" in _doc(README_EN) and "intersection of all four gates is **0**" in _doc(README_EN)
    assert "do **not** form a funnel" in _doc(CASE_STUDY)


# ---- 清单不许静默失联 ---------------------------------------------------------------


def test_manifest_is_documented_where_a_reader_would_look(manifest: dict) -> None:
    """清单必须被自述文档与 README 引用，否则没人知道它的存在。"""
    for path in (ALPHA_DOC, README_ZH):
        assert "alpha-evidence-chain.json" in _doc(path), f"{path.name} 应引用证据链清单"
    for path in (ALPHA_DOC, README_ZH):
        assert "alpha_evidence_chain.py" in _doc(path), f"{path.name} 应给出核验命令"

    script = SCRIPT.read_text(encoding="utf-8")
    assert "docs" in script and "alpha-evidence-chain.json" in script
    assert "Path(__file__)" in script, "脚本应基于自身位置解析路径"
    assert "kimiwork-z" not in script, "脚本不应写死本机路径"
