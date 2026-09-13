# -*- coding: utf-8 -*-
"""把"文档里的复现命令必须存在"这一条固定下来。

写作目的：`docs/ALPHA_RESEARCH.md` 声称"所有数字都来自可复现的命令"，但脚本一度只存在于临时目录、
命令里还带着省略号。这类"文档承诺 ≠ 仓库现状"的落差靠人工检查不可靠，用测试固定：
文档里提到的 `scripts/*.py` 必须真的存在。
"""

import re
from pathlib import Path

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


def test_research_scripts_avoid_hardcoded_absolute_paths():
    """研究脚本不应写死作者本机的绝对路径（否则别人跑不通）。"""
    for path in sorted((ROOT / "scripts").glob("alpha_*.py")):
        text = path.read_text(encoding="utf-8")
        assert "kimiwork-z" not in text, f"{path.name} 里写死了本机路径"
        assert "Path(__file__)" in text, f"{path.name} 应基于脚本位置解析路径"
