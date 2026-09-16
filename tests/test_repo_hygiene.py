"""仓库卫生检查：防止"隐形字符"再咬人。

背景：本项目已经被 UTF-8 **BOM** 咬过两次——用 shell 重写 `pyproject.toml` 时带上了 ``EF BB BF``，
于是 `pip install -e` / `mypy` 解析 TOML 直接失败（"Invalid statement at line 1"），
而本地的 `pytest` 却照跑不误，排查成本很高。这条测试把这类问题变成一次明确的失败。
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOM = b"\xef\xbb\xbf"

# 文本类文件；二进制与生成物不检查
PATTERNS = ("*.py", "*.toml", "*.md", "*.yml", "*.yaml", "*.cfg", "*.txt", "*.ps1")
SKIP_DIRS = {".git", ".ruff_cache", ".pytest_cache", ".mypy_cache", "data", "output", "build", "dist", "__pycache__", ".venv"}


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for pattern in PATTERNS:
        for path in ROOT.rglob(pattern):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def test_no_text_file_starts_with_a_utf8_bom():
    offenders = [path.relative_to(ROOT).as_posix() for path in _candidate_files() if path.read_bytes().startswith(BOM)]
    assert not offenders, f"这些文件带了 UTF-8 BOM，会让 TOML/配置解析失败：{offenders}"


def test_pyproject_parses_and_declares_expected_sections():
    """pyproject.toml 必须能被标准库解析，且关键配置段都在。"""
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11+")
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert data["project"]["name"] == "ashare-quant-lab"
    assert "dev" in data["project"]["optional-dependencies"]
    # lint / 类型 / 测试策略都钉在仓库里，CI 与本地才会一致
    assert data["tool"]["ruff"]["lint"]["select"]
    assert data["tool"]["mypy"]["ignore_missing_imports"] is True
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_workflow_files_are_plain_yaml_without_tabs():
    """CI 配置不做过度解析，只挡住"用 Tab 缩进"这类低级错误。"""
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "\t" not in text, f"{path.name} 里出现了 Tab 缩进"
        assert text.startswith("name:"), f"{path.name} 应以 name: 开头"


def test_gitattributes_pins_line_endings_and_marks_binaries():
    """.gitattributes 必须存在，并把文本文件的换行符钉成 LF、把图片标为二进制。

    没有它时，Windows 上 48 个文件会在 `git add` 时打印 "LF will be replaced by CRLF"，
    而跨平台协作会让"只改了行尾"的提交看起来像整文件重写。图片若被当文本处理，
    行尾转换会直接损坏文件。
    """
    attrs = ROOT / ".gitattributes"
    assert attrs.is_file(), "仓库应当有 .gitattributes"
    text = attrs.read_text(encoding="utf-8")

    # 默认把所有文本按 LF 存储
    assert "* text=auto eol=lf" in text, "应当声明默认 `* text=auto eol=lf`"
    # 关键文本类型显式钉住
    for pattern in ("*.py", "*.md", "*.toml", "*.yml"):
        assert f"{pattern}" in text, f"应当在 .gitattributes 中声明 {pattern}"
    # 二进制类型必须标记，否则行尾转换会损坏文件
    for pattern in ("*.png", "*.jpg", "*.webp"):
        assert f"{pattern}" in text, f"应当把 {pattern} 标为 binary"
    assert "binary" in text


def test_no_text_file_in_the_index_uses_crlf():
    """索引里不应存在以 CRLF 存储的文本文件。

    索引（`git ls-files --eol`）显示 i/crlf 说明某个文本文件被以 CRLF 提交了，
    在 Linux/macOS 上会表现为多余的回车符。这条用 git 自身的事实来判定。
    """
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--eol"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    offenders = [
        line.split("\t", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("i/crlf")
    ]
    assert not offenders, f"索引里有以 CRLF 提交的文本文件：{offenders[:10]}"
