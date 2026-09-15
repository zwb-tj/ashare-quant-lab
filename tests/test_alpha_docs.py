# -*- coding: utf-8 -*-
"""把"文档里的复现命令必须存在"这一条固定下来。

写作目的：`docs/ALPHA_RESEARCH.md` 声称"所有数字都来自可复现的命令"，但脚本一度只存在于临时目录、
命令里还带着省略号。这类"文档承诺 ≠ 仓库现状"的落差靠人工检查不可靠，用测试固定：
文档里提到的 `scripts/*.py` 必须真的存在。
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ALPHA_RESEARCH.md"


def test_alpha_research_doc_exists_and_states_the_chain():
    assert DOC.exists(), "因子研究自述文档应当存在"
    text = DOC.read_text(encoding="utf-8")
    # 证据链的关键数字必须在文档里出现（它们是整条链的骨架）
    for figure in ("41", "35", "28", "11 / 16", "12", "3"):
        assert figure in text, f"文档应包含证据链数字 {figure}"
    # 关键局限必须写明，避免结论被单独引用
    for caveat in ("幸存者偏差", "未来信息", "未建模"):
        assert caveat in text, f"文档应写明局限：{caveat}"


def test_every_script_referenced_by_the_doc_exists():
    """文档复现章节里引用的 scripts/*.py 必须真实存在。"""
    text = DOC.read_text(encoding="utf-8")
    referenced = set(re.findall(r"python (scripts/[A-Za-z0-9_]+\.py)", text))
    assert referenced, "文档应当给出可运行的复现命令"
    missing = sorted(name for name in referenced if not (ROOT / name).exists())
    assert not missing, f"文档引用了不存在的脚本：{missing}"


def test_doc_does_not_leave_ellipsis_in_commands():
    """复现命令里不允许出现 `...` 这类省略——那等于不可复现。"""
    text = DOC.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith(("python ", "#")) and line.strip().startswith("python "):
            assert "..." not in line, f"命令不应带省略号：{line.strip()}"


def test_fresh_clone_verifier_exists_and_is_referenced():
    """全新克隆验证脚本必须存在，且 README 的"从零开始"要提到它。"""
    script = ROOT / "scripts" / "verify_fresh_clone.py"
    assert script.exists(), "应提供全新克隆验证脚本"
    text = script.read_text(encoding="utf-8")
    assert "git clone" in text and "pytest" in text and "reproduce_all.py" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "verify_fresh_clone.py" in readme, "README 的从零开始章节应提到它"


def test_brief_claim_verifier_exists():
    """"速览不许吹牛"的把关脚本必须入库，且能解释它在核对什么。"""
    script = ROOT / "scripts" / "verify_brief_claims.py"
    assert script.exists(), "应提供速览声明核对脚本"
    body = script.read_text(encoding="utf-8")
    for keyword in ("alpha101_ic_real", "walk_forward_summary", "alpha_portfolio_costs", "state_regime_bucket"):
        assert keyword in body, f"核对脚本应检查 {keyword}"


def test_research_scripts_avoid_hardcoded_absolute_paths():
    """研究脚本不应写死作者本机的绝对路径（否则别人跑不通）。"""
    for path in sorted((ROOT / "scripts").glob("alpha_*.py")):
        text = path.read_text(encoding="utf-8")
        assert "kimiwork-z" not in text, f"{path.name} 里写死了本机路径"
        assert "Path(__file__)" in text, f"{path.name} 应基于脚本位置解析路径"


# ---- 案例研究的数字一致性 -------------------------------------------------------------

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 20: "twenty",
}


def _case_study() -> str:
    return (ROOT / "docs" / "CASE_STUDY.md").read_text(encoding="utf-8")


def test_case_study_reports_the_real_cli_subcommand_and_test_module_counts():
    """CLI 子命令数与测试模块数必须与仓库实际一致（都曾经过期）。"""

    text = _case_study()
    modules = len(list((ROOT / "tests").glob("test_*.py")))
    assert f"{modules} test modules" in text, f"案例研究应写明 {modules} 个测试模块"

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from aqlab.cli import build_parser

    commands = len(build_parser()._subparsers._group_actions[0].choices)
    assert f"{commands} subcommands" in text, f"案例研究应写明 {commands} 个 CLI 子命令"


def test_case_study_coverage_figures_match_the_output():
    """覆盖率、语句数、未覆盖数必须与覆盖率产物一致。

    这里**不嵌套调用 pytest**：在测试内部跑 `pytest --cov` 会重新导入并可能重复执行整个套件
    （实测直接把测试拖到超时），而且依赖覆盖率插件在子进程里可用。
    改为读取已有的覆盖率数据文件；没有产物时跳过并说明原因，而不是伪造通过。
    """
    import json
    import subprocess
    import sys

    text = _case_study()
    data_file = ROOT / "coverage.json"
    if not data_file.is_file():
        # 用 `pytest --cov=aqlab --cov-report=json:coverage.json` 生成后再校验
        produced = subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", str(data_file)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if produced.returncode != 0 or not data_file.is_file():
            pytest.skip("没有覆盖率产物（可先跑 pytest --cov=aqlab --cov-report=json:coverage.json）")

    try:
        totals = json.loads(data_file.read_text(encoding="utf-8"))["totals"]
    finally:
        data_file.unlink(missing_ok=True)

    statements = int(totals["num_statements"])
    missed = int(totals["missing_lines"])
    percent = round(float(totals["percent_covered"]))

    assert f"{percent}%" in text, f"案例研究里的覆盖率应为 {percent}%"
    assert f"{statements:,}" in text, f"案例研究里的语句数应为 {statements:,}"
    assert str(missed) in text, f"案例研究里的未覆盖语句数应为 {missed}"


def test_case_study_rejected_hypothesis_count_is_accurate():
    """被否假设的条数必须与列表实际条数一致（英文数字词与数字形式都认）。"""
    text = _case_study()
    start = text.index("## What was rejected and why")
    end = text.index("## Engineering")
    actual = len([line for line in text[start:end].splitlines() if line.startswith("- **")])

    word = _NUMBER_WORDS.get(actual, str(actual))
    assert f"{word} hypotheses" in text or f"{actual} hypotheses" in text, (
        f"Overview 应写明被否假设为 {actual} 条（{word} hypotheses）"
    )
    # 反向检查：不能出现其它数字词的错误表述
    for count, other in _NUMBER_WORDS.items():
        if count != actual and f"{other} hypotheses" in text:
            raise AssertionError(f"Overview 出现错误的假设条数：{other} hypotheses（实际 {actual}）")
