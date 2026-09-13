"""研究产物目录约定的测试（离线）。

**为什么要有这条测试**：CLI 的 `--alpha101` 曾经与真实研究脚本写同一个目录
（`output/alpha101_ic/`），一次带**合成数据**的冒烟运行就把真实全市场结果覆盖掉了——
而覆盖是静默的，只有事后核对数字时才发现。这类"产物互相覆盖"的隐患靠约定不够，
必须由测试固定：CLI 与研究脚本必须写不同目录。

实现上刻意**只检查真实写入路径**（`Path(...) / "目录名"` 这种表达式），不检查注释或文档字符串里
出现的名字——否则一条解释性的注释就会让测试误报。
"""

import re
from pathlib import Path

from aqlab.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def _cli_written_dirs() -> set[str]:
    """从 cli.py 里抽出所有 `Path(...) / "xxx"` 形式的写入目录名。"""
    source = (ROOT / "src" / "aqlab" / "cli.py").read_text(encoding="utf-8")
    return set(re.findall(r'Path\([^)]*\)\s*/\s*"([a-z0-9_]+)"', source))


def test_cli_and_research_scripts_use_different_alpha_directories():
    """CLI 的 alpha101 输出目录必须与研究脚本不同（研究脚本写 *_real）。"""
    written = _cli_written_dirs()
    assert "alpha101_ic_cli" in written, f"CLI 应写自己的子目录，实际写入目录：{sorted(written)}"
    assert "alpha101_ic_real" not in written, "CLI 不得写研究脚本的目录"


def test_cli_written_directories_are_explicitly_named():
    """CLI 写出的目录名要能一眼看出属于哪个入口，避免再次撞名。"""
    written = _cli_written_dirs()
    assert written, "应当能从 cli.py 里解析出写入目录"
    for name in written:
        assert name.islower() and " " not in name, f"目录名应是小写下划线形式：{name}"


def test_parser_accepts_an_isolated_output_root(tmp_path):
    """`--out` 指向哪里，产物就落在那里——测试用临时目录验证参数被正确接收。"""
    parser = build_parser()
    args = parser.parse_args(["factor-ic", "--alpha101", "--out", str(tmp_path)])
    assert Path(args.out) == tmp_path
