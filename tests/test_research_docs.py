# -*- coding: utf-8 -*-
"""把 `docs/research/` 里的实测数字钉在复现脚本的产出上。

写作目的：`docs/research/brick-short-term.md` 与 `b3-dynamic-group.md` 声称数字"可复现"，
但产生它们的脚本原先只存在于临时目录。补上 `scripts/brick_research.py` 后重跑，
**立刻抓出三处错误**：

* 「提升倍数」写成 934x，正确是 1.46% ÷ 0.16% ≈ **9.3x**（差 100 倍）；
* 「前期超跌覆盖率归零」只对一种口径成立（20 日累计跌幅），另一种口径是 9.1%；
* 价格 vs 砖色的四个计数与一致率有轻微偏差（旧脚本样本口径不同）。

所以本文件不重复写死那些数字，而是**从清单读取**、再断言文档里出现同样的值：
文档数字一旦漂移，测试立刻失败。类似 `test_documented_numbers.py` 对因子证据链的做法。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"
MANIFEST = RESEARCH / "brick-research-results.json"

BRICK_DOC = RESEARCH / "brick-short-term.md"
B3_DOC = RESEARCH / "b3-dynamic-group.md"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), (
        "缺少数字清单——请先跑 `python scripts/brick_research.py`（约 30 秒）"
    )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    """把清单里的百分数渲染成文档里的写法（76.03 → '76.03%'）。"""
    return f"{value:.2f}%".rstrip("0").rstrip(".") + "" if value % 1 else f"{value:.1f}%"


def _grouped(value: int) -> str:
    return f"{value:,}"


# ---- 砖型图文档：第 10 节的价格 vs 砖色 -------------------------------------------------


def test_brick_doc_matches_the_manifest_agreement_figures(manifest: dict) -> None:
    """一致率/不一致率/观测数必须与清单一致（这三项最容易在改脚本后过期）。"""
    m = manifest["price_vs_brick"]
    text = BRICK_DOC.read_text(encoding="utf-8")
    for label, value in (
        ("一致率", m["agreement_pct"]),
        ("不一致率", m["disagreement_pct"]),
        ("价格涨但绿砖", m["up_but_green_pct"]),
        ("价格跌但红砖", m["down_but_red_pct"]),
    ):
        assert f"{value}%" in text, f"砖型图文档缺少/过期：{label} 应为 {value}%"
    assert _grouped(m["observations"]) in text, (
        f"砖型图文档里的观测数应为 {_grouped(m['observations'])}"
    )
    assert f"{m['symbols']} 只" in text, f"砖型图文档应写明样本为 {m['symbols']} 只标的"


def test_brick_doc_disagreement_is_the_sum_of_its_two_parts(manifest: dict) -> None:
    """不一致率必须等于"价格涨但绿砖 + 价格跌但红砖"——防止表里各数各改。"""
    m = manifest["price_vs_brick"]
    parts = m["up_but_green_pct"] + m["down_but_red_pct"]
    assert parts == pytest.approx(m["disagreement_pct"], abs=0.02), (
        f"不一致率 {m['disagreement_pct']}% 应等于两部分之和 {parts:.2f}%"
    )


# ---- B3 文档：特征分布 + 候选语句 -------------------------------------------------------


def test_b3_doc_feature_distribution_matches_the_manifest(manifest: dict) -> None:
    """B3 特征分布的关键三项必须与清单一致。"""
    feats = manifest["b3_features"]
    text = B3_DOC.read_text(encoding="utf-8")
    n = manifest["b3_extra"]["n_signals"]

    # 「近 3 日大涨」「量在 10 日线上」用于支撑核心建议，必须准确
    for key in ("近 3 日内有过 >=4% 大涨", "量在 10 日线上"):
        pct = feats[key]["pct"]
        hits = feats[key]["hits"]
        assert f"{hits} / {n}" in text, f"B3 文档缺少 {key} 的 {hits} / {n}"
        assert f"{pct}%" in text, f"B3 文档缺少 {key} 的占比 {pct}%"


def test_b3_doc_sentence_table_matches_the_manifest(manifest: dict) -> None:
    """候选语句的覆盖率与准确率必须与清单一致。"""
    text = B3_DOC.read_text(encoding="utf-8")
    for label, row in manifest["b3_sentences"].items():
        assert f"{row['coverage_pct']}%" in text, (
            f"{label} 的覆盖率 {row['coverage_pct']}% 未出现在 B3 文档中"
        )
        precision = f"{row['precision_pct']}%"
        assert precision in text, (
            f"{label} 的准确率 {precision} 未出现在 B3 文档中"
        )


def test_b3_lift_is_a_single_digit_multiple_not_hundreds(manifest: dict) -> None:
    """提升倍数曾被写成 934x（真值约 9.3x）。

    这条**单独守**：数字写错 100 倍时，其它断言（覆盖率/准确率都对）不会失败。
    期望值取清单里的 ``lift``（由脚本按**全精度**准确率算出），
    不从清单里已四舍五入的准确率反推——那样 1.46/0.16 = 9.125 会和 9.34 差一点。
    """
    sentences = manifest["b3_sentences"]
    lift = sentences["B A + 近 3 日大涨 4%"]["lift"]
    assert 5 <= lift <= 20, f"提升倍数应为个位数倍，清单里是 {lift}x"
    assert sentences["A 平开 + 小实体"]["lift"] == 1.0, "纯形态语句的提升倍数基准应为 1x"

    text = B3_DOC.read_text(encoding="utf-8")
    # 文档不得再出现"几百倍"的说法
    for bogus in re.findall(r"(\d{3,})\s*[xX倍]", text):
        raise AssertionError(f"B3 文档出现夸大的提升倍数：{bogus}x（真值约 {lift}x）")
    assert f"{lift:.1f}x" in text, f"B3 文档应写明提升倍数约 {lift:.1f}x"


def test_b3_doc_reports_zero_overlap_with_the_reversal_sentence(manifest: dict) -> None:
    """核心结论「反包语义与 B3 不相交」必须有数据支撑。"""
    extra = manifest["b3_extra"]
    assert extra["reversal_sentence_overlap_with_b3"] == 0
    assert extra["b3_with_prior3_down"] == 0
    text = B3_DOC.read_text(encoding="utf-8")
    assert "交集 = 0" in text or "重叠 0" in text, "B3 文档应写明交集为 0"
    assert _grouped(extra["reversal_sentence_selected"]) in text, (
        f"B3 文档应写明反包语句选中 {_grouped(extra['reversal_sentence_selected'])} 行"
    )


# ---- 复现脚本与文档本身 ----------------------------------------------------------------


def test_reproduction_script_exists_and_is_referenced() -> None:
    """两份研究文档都要指向真正存在的复现脚本（文档承诺不许落空）。"""
    script = ROOT / "scripts" / "brick_research.py"
    assert script.exists(), "应提供砖型图/B3 数字的复现脚本"
    for doc in (BRICK_DOC, B3_DOC):
        text = doc.read_text(encoding="utf-8")
        assert "scripts/brick_research.py" in text, f"{doc.name} 应给出复现命令"
        assert "brick-research-results.json" in text, f"{doc.name} 应指明数字清单"


def test_research_docs_reference_only_existing_scripts() -> None:
    """研究文档里 `python scripts/*.py` 引用的脚本必须都存在。"""
    missing: list[str] = []
    for doc in sorted(RESEARCH.glob("*.md")):
        for name in re.findall(r"python (scripts/[A-Za-z0-9_]+\.py)", doc.read_text(encoding="utf-8")):
            if not (ROOT / name).exists():
                missing.append(f"{doc.name} -> {name}")
    assert not missing, f"研究文档引用了不存在的脚本：{missing}"


def test_research_docs_have_no_ellipsis_in_commands() -> None:
    """复现命令不许带省略号（带了就等于不可复现）。"""
    bad: list[str] = []
    for doc in sorted(RESEARCH.glob("*.md")):
        for line in doc.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("python ") and "..." in stripped:
                bad.append(f"{doc.name}: {stripped}")
    assert not bad, f"复现命令不应含省略号：{bad}"


def test_brick_research_script_has_no_hardcoded_absolute_paths() -> None:
    """研究脚本必须基于脚本位置解析路径，不能写死作者本机目录。"""
    text = (ROOT / "scripts" / "brick_research.py").read_text(encoding="utf-8")
    assert "kimiwork-z" not in text, "复现脚本里写死了本机路径"
    assert "Path(__file__)" in text, "复现脚本应基于 Path(__file__) 解析仓库根"


def test_manifest_carries_a_provenance_note() -> None:
    """清单要说明自己由谁生成，避免以后被当成手写数据改。"""
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "brick_research.py" in data.get("_note", ""), "清单应注明生成脚本"
