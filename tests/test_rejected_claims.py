# -*- coding: utf-8 -*-
"""把"被自己数据否决的结论"钉在一份**可提交**的清单上（不依赖 `output/`）。

落差在哪：`docs/CASE_STUDY.md` 的 "What was rejected and why" 一节，全部数字都来自
gitignored 的 `output/`（`picks_backtest/`、`universe_study/`、`sweep/`、`factor_schemes/`、
`eval/`）。原先没有任何断言覆盖它们 —— 文档最引以为傲的"诚实否定"部分，恰好是唯一
**完全没有守卫**的部分：改数字、改口径、产物被覆盖，都不会失败。

`docs/research/rejected-claims.json` 由 `scripts/rejected_claims.py --write` 从真实产物
生成并**入库**（与 `alpha-evidence-chain.json`、`brick-research-results.json` 同一套做法）。
本文件不重写那些数字，而是从清单读、断言文档里出现同值：产物重算导致清单过期、或有人
手改清单，都会失败。

断言一律跑在**折叠空白后**的文档文本上：markdown 会在句中断行，按原样比对会因换行假失败
（这种假失败最容易诱人放宽断言，而断言一旦放宽就再也抓不到真错误了）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "rejected-claims.json"
SCRIPT = ROOT / "scripts" / "rejected_claims.py"

CASE_STUDY = ROOT / "docs" / "CASE_STUDY.md"
README_ZH = ROOT / "README.md"
README_EN = ROOT / "README.en.md"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), (
        "缺少被否结论清单——请先跑 `python scripts/rejected_claims.py --write`（需要真实产物）"
    )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _flat(path: Path) -> str:
    """读文档并折叠所有空白，避免 markdown 换行造成假失败。"""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _structural(manifest: dict, reason: str) -> dict:
    return manifest["exits"]["structural_only"]["reasons"][reason]


def _sop(manifest: dict, reason: str) -> dict:
    return manifest["exits"]["sop_tight"]["reasons"][reason]


# ---- 清单自身 ----------------------------------------------------------------------


def test_manifest_records_how_it_was_generated(manifest: dict) -> None:
    """清单必须自带来源与核验方式，否则读者无法判断数字可信度。"""
    note = manifest["_note"]
    assert "python scripts/rejected_claims.py" in note, "清单应写明核验命令"
    assert "数字以本清单为准" in note, "清单应写明与产物的优先关系"
    assert manifest["provenance"]["note"], "清单应说明产物为何不入库"


def test_provenance_reproduce_commands_are_runnable(manifest: dict) -> None:
    """复现命令必须能照着跑：脚本引用要真实存在，且任何一条都不许带省略号。"""
    commands = manifest["provenance"]["reproduce"]
    assert commands, "清单应给出完整复现命令"
    for command in commands:
        assert "..." not in command, f"复现命令不应带省略号：{command}"
        match = re.match(r"python (scripts/[A-Za-z0-9_]+\.py)", command)
        if match:
            assert (ROOT / match.group(1)).is_file(), f"复现命令引用了不存在的脚本：{command}"
        else:
            assert command.startswith(("aqlab ", "python -m aqlab")), f"无法识别的复现命令：{command}"


def test_manifest_artifacts_are_relative_output_paths(manifest: dict) -> None:
    """产物路径必须是仓库相对的 `output/` 路径，不得写死本机绝对路径。"""
    artifacts = manifest["provenance"]["artifacts"]
    assert artifacts, "清单应列出产物"
    for artifact in artifacts:
        assert artifact.startswith("output/"), f"产物路径应以 output/ 开头：{artifact}"
        assert ":" not in artifact, f"产物路径不应含盘符：{artifact}"


# ---- 文档与清单一致（不需要产物） ---------------------------------------------------


def test_case_study_reports_the_volume_gate_numbers(manifest: dict) -> None:
    """量比门槛那条：选中/排除的收益差与 Welch t 必须与清单一致。"""
    relative = manifest["volume_gate"]["relative_caliber"]
    selected = relative["selected"]
    excluded = relative["excluded"]
    text = _flat(CASE_STUDY)
    assert f"{selected['n']} of 146 signals at {selected['excess_10_pct']:.2f}%" in text
    assert f"{excluded['excess_10_pct']:.2f}% for the {excluded['n']} excluded" in text
    spread = relative["relative_spread"]
    assert f"({spread['spread_pp']:.2f}pp, t = {spread['welch_t']:.2f})" in text, "文档应给出收益差与 Welch t"


def test_case_study_reports_the_software_caliber_and_scan(manifest: dict) -> None:
    """软件口径那一行、以及不分方向的阈值扫描三格，都要对得上。"""
    software = manifest["volume_gate"]["software_caliber"]["ratio_std_4_up"]
    scan = manifest["volume_gate"]["scan_software_undirected"]
    relative_scan = manifest["volume_gate"]["scan_relative"]
    text = _flat(CASE_STUDY)

    # 量比门槛表按**命中数**（matched）列出该行
    assert f"| {software['matched']} | {software['excess_10_pct']:.2f} |" in text
    # 阈值扫描表的 n 行与 Excess 行：三格逐格出现（该表用有报价数 priced）
    assert "| n | " + " | ".join(str(scan[str(t)]["priced"]) for t in (3, 4, 5)) + " |" in text
    assert (
        "| Excess % | "
        + " | ".join(f"{scan[str(t)]['excess_10_pct']:.2f}" for t in (3, 4, 5))
        + " |"
    ) in text
    relative_values = " / ".join(
        f"{relative_scan[k]['excess_10_pct']:.2f}" for k in ("0.5", "0.8", "1.0", "1.5")
    )
    assert f"{relative_values}%" in text, "文档应给出相对口径自身的扫描"


def test_case_study_reports_the_robustness_grid(manifest: dict) -> None:
    """稳健性网格：格数、全为负、均值与中位数。"""
    grid = manifest["robustness_grid"]
    text = _flat(CASE_STUDY)
    assert f"| cells (threshold 3/4/5 x up/down at 09:37) | {grid['cells']} |" in text
    assert f"| mean excess % | {grid['mean_pct']:.2f} |" in text
    assert f"| median excess % | {grid['median_pct']:.2f} |" in text
    assert grid["all_negative"] is True
    assert len(grid["means_pct"]) == grid["cells"], "互异格点的收益个数应与格数相同"
    assert len(set(grid["means_pct"])) == grid["cells"], "6 个格点应互不相同（不是同一批数了两遍）"


def test_case_study_reports_both_exit_calibers(manifest: dict) -> None:
    """结构出场那条必须**同时**给出两个口径，否则又会互相混读。

    这是文档真实犯过的错：把 SOP 紧档的 33,440 / 15,325 写在描述结构档的条目里。
    """
    structural_break = _structural(manifest, "white_break")
    structural_didi = _structural(manifest, "didi")
    sop_break = _sop(manifest, "white_break")
    sop_didi = _sop(manifest, "didi")
    text = _flat(CASE_STUDY)

    assert f"{structural_break['count']:,} of 58,682 exits" in text
    assert f"averaging {structural_break['mean_return_pct']:.2f}%" in text
    assert (
        f"{structural_didi['count']:,} (33.7%) averaging **+{structural_didi['mean_return_pct']:.2f}%**"
        in text
    )
    # SOP 口径作为"早先版本引用的口径"必须也写明
    assert f"{sop_break['count']:,} breaks at" in text
    assert f"{sop_didi['count']:,} at {sop_didi['mean_return_pct']:.2f}%" in text


def test_exit_calibers_disagree_so_the_label_matters(manifest: dict) -> None:
    """两个口径确实给出相反结论（趋势线在结构档是正、在 SOP 档是负），所以标签不能省。"""
    structural = _structural(manifest, "didi")["mean_return_pct"]
    sop = _sop(manifest, "didi")["mean_return_pct"]
    assert structural > 0 > sop, f"两个口径应给出相反符号：{structural} vs {sop}"


def test_case_study_reports_the_tail_risk_evidence(manifest: dict) -> None:
    """干净的反证是尾部风险，两个数都要在文档里。"""
    tail = manifest["exits"]["tail_risk"]
    text = _flat(CASE_STUDY)
    assert f"worst trade -{abs(tail['full_exit_rules_worst_pct']):.1f}%" in text
    assert f"-{abs(tail['plain_hold_20d_worst_pct']):.1f}% for a plain hold" in text
    assert tail["full_exit_rules_worst_pct"] < tail["plain_hold_20d_worst_pct"], "离场规则应更差才有资格做反证"


def test_case_study_reports_the_entry_threshold_numbers(manifest: dict) -> None:
    """放宽 j_max 那条：持仓数、超额、超额胜率，以及 40 只池子的对照。"""
    pool80 = manifest["entry_threshold"]["pool_80_spike_1_8"]
    low, high = pool80["j_max_13"], pool80["j_max_30"]
    pool40 = manifest["entry_threshold"]["pool_40_spike_2_0_j_max_13"]
    text = _flat(CASE_STUDY)

    assert f"from {low['avg_positions']:.2f} to {high['avg_positions']:.2f}" in text
    assert f"excess from +{low['excess_mean_pct']:.2f}% to +{high['excess_mean_pct']:.2f}%" in text
    assert (
        f"excess win rate from -{abs(low['excess_win_rate_pct']):.2f}% to "
        f"-{abs(high['excess_win_rate_pct']):.2f}%"
    ) in text
    assert f"was +{pool40['excess_win_rate_pct']:.2f}%" in text
    assert high["avg_positions"] > low["avg_positions"], "放宽阈值确实提高了持仓数（这条的一半是对的）"


def test_case_study_reports_the_published_pick_record(manifest: dict) -> None:
    """选股规则那条：记录数、四个持有期的超额与 t 区间、24 格中显著为负的格数。"""
    published = manifest["published_picks"]
    text = _flat(CASE_STUDY)
    values = " / ".join(f"{value:.2f}%" for value in published["all_picks_excess_pct"])
    assert values in text, f"文档应给出四个持有期的超额：{values}"
    # t 区间按持有期顺序写（1 日 → 10 日），即列表首尾，而不是数值大小
    assert f"t = {published['all_picks_t'][0]:.2f} to {published['all_picks_t'][-1]:.2f}" in text
    assert f"{published['negative_t_cells']} of {published['cells']} cells were" in text
    assert published["positive_t_cells"] == 0, "不得出现显著为正的格"


def test_published_cell_detail_covers_every_bucket_and_horizon(manifest: dict) -> None:
    """24 格明细必须自洽：6 个分组 × 4 个持有期，且负 t 的格数等于清单声明。"""
    published = manifest["published_picks"]
    detail = published["cells_detail"]
    assert len(detail) == published["cells"] == 24
    assert len({cell["bucket"] for cell in detail}) == 6
    assert {cell["horizon"] for cell in detail} == {1, 3, 5, 10}
    negative = sum(1 for cell in detail if cell["t"] < -2)
    assert negative == published["negative_t_cells"], "清单里的显著为负格数与明细不符"


def test_case_study_reports_the_sign_flip_result(manifest: dict) -> None:
    """取反那条：取反后的净收益/超额/t，以及默认方向与基准的对照。"""
    sign_flip = manifest["sign_flip"]
    flipped = sign_flip["schemes"]["sign_flip"]
    fixed = sign_flip["schemes"]["fixed"]
    text = _flat(CASE_STUDY)
    assert f"returned {flipped['mean_net_pct']:.2f}% net" in text
    assert f"{flipped['mean_excess_pct']:.2f}% excess (t = {flipped['excess_t']:.2f})" in text
    assert f"+{fixed['mean_net_pct']:.2f}% net for the default direction" in text
    assert f"+{sign_flip['benchmark_mean_pct']:.2f}%" in text
    assert flipped["mean_excess_pct"] < fixed["mean_excess_pct"], "取反应比默认方向更差"


def test_case_study_reports_the_agent_grounding_numbers(manifest: dict) -> None:
    """Agent 那条：检查过的数字个数与被标记为编造的任务数。"""
    agent = manifest["agent_grounding"]
    text = _flat(CASE_STUDY)
    assert f"{agent['numbers_total']} numbers were checked" in text
    assert "one task is flagged as hallucinated" in text
    assert agent["hallucinated_tasks"] == 1


def test_opening_features_claim_is_reported_as_window_dependent(manifest: dict) -> None:
    """高量比那条：短窗口是负、21 个月是正，文档应写成"一个弱窗口"而不是规律。"""
    opening = manifest["opening_features"]
    short = opening["short_window"]["volume_ratio_ge_4"]
    long = opening["long_window"]["volume_ratio_ge_4"]
    text = _flat(CASE_STUDY)
    assert f"{short['excess_pct']:.2f}% short-window figure became +{long['excess_pct']:.2f}%" in text
    assert f"t = {long['excess_t']:.2f}" in text
    assert short["excess_pct"] < 0 < long["excess_pct"], "两个窗口应给出相反符号，这正是「只是窗口」的证据"


def test_readme_highlights_quote_the_published_record(manifest: dict) -> None:
    """README 亮点表引用了同一组数字，不能与清单脱节。"""
    published = manifest["published_picks"]
    zh = _flat(README_ZH)
    assert f"{published['records']} " in zh, "中文 README 应给出记录数"
    assert f"24 格中 {published['negative_t_cells']} 格显著为负" in zh, "中文 README 应给出格数"

    en = _flat(README_EN)
    assert f"{published['records']} published records" in en, "英文 README 应给出记录数"
    assert f"{published['negative_t_cells']} of {published['cells']} cells" in en, "英文 README 应给出格数"


# ---- 清单不许静默失联 ---------------------------------------------------------------


def test_manifest_is_documented_where_a_reader_would_look(manifest: dict) -> None:
    """清单必须被案例研究引用，否则没人知道它的存在。"""
    text = _flat(CASE_STUDY)
    assert "rejected-claims.json" in text, "案例研究应引用被否结论清单"
    assert "rejected_claims.py" in text, "案例研究应给出核验命令"

    script = SCRIPT.read_text(encoding="utf-8")
    assert "docs" in script and "rejected-claims.json" in script
    assert "Path(__file__)" in script, "脚本应基于自身位置解析路径"
    assert "kimiwork-z" not in script, "脚本不应写死本机路径"
    assert "--write" in script and "argparse" in script, "脚本应能重新生成清单"
