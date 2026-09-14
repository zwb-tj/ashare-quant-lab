# -*- coding: utf-8 -*-
"""防止"本机 3.12 能跑、CI 的 3.10 挂掉"这类问题再次发生。

**这次的教训**：本机只有 Python 3.12，我在测试里直接写了 `import tomllib`（3.11+ 才有），
而项目声明 `requires-python = ">=3.10"`、CI 矩阵含 3.10 —— 于是 CI 变红，本机却全绿。
干净克隆验证也抓不到，因为克隆用的还是 3.12。

因此加两层防线：
① **静态扫描**：禁止对 3.11+ 才有的标准库做**无保护**导入（必须 importorskip 或 try/except）；
② **CI**：在 3.10 那一档额外跑一次"导入全部模块"，把版本兼容问题暴露在最便宜的环节。
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Python 版本高于 3.10 才进入标准库的模块（项目声明支持 3.10）
NEWER_STDLIB = {
    "tomllib": "3.11",
    "exceptiongroup": "3.11",
}


def _python_files() -> list[Path]:
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    return [path for path in files if "__pycache__" not in path.parts]


def _is_guarded(tree: ast.AST, target: str, source: str) -> bool:
    """判断这个模块的导入是否被保护：importorskip / try-except / 版本判断。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Import) and any(alias.name == target for alias in child.names):
                    return True
                if isinstance(child, ast.ImportFrom) and child.module == target:
                    return True
    # importorskip 形式
    return "importorskip" in source and target in source.split("importorskip", 1)[-1][:200]


def test_no_unguarded_import_of_newer_stdlib():
    """3.11+ 才有的标准库模块必须做保护，否则 3.10 上会 ImportError。"""
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:  # 语法错误由别的测试负责
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for module in imported & set(NEWER_STDLIB):
            if not _is_guarded(tree, module, source):
                offenders.append(f"{path.relative_to(ROOT).as_posix()} 导入 {module}（需要 Python {NEWER_STDLIB[module]}+）但未做保护")
    assert not offenders, "存在未保护的新版本标准库导入：\n" + "\n".join(offenders)


def test_declared_python_version_is_respected_in_ci_matrix():
    """CI 矩阵必须覆盖项目声明的最低 Python 版本。"""
    import re

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', pyproject)
    assert match, "pyproject 应当声明 requires-python"
    minimum = f"{match.group(1)}.{match.group(2)}"

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = set(re.findall(r'"(\d+\.\d+)"', workflow))
    assert minimum in versions, f"CI 矩阵（{sorted(versions)}）未覆盖最低版本 {minimum}"


def test_all_modules_import_on_this_interpreter():
    """当前解释器上所有模块都应可导入（这是最低限度的一致性检查）。"""
    sys.path.insert(0, str(ROOT / "src"))
    import importlib

    modules = sorted(path.stem for path in (ROOT / "src" / "aqlab").glob("*.py") if path.stem != "__init__")
    assert modules, "应当能发现模块"
    for name in modules:
        importlib.import_module(f"aqlab.{name}")


def test_every_file_parses_under_the_minimum_supported_syntax():
    """用 Python 3.10 的语法规则解析全部源码。

    本机只有 3.12、跑不到 3.10；但 ``ast.parse(feature_version=(3, 10))`` 能在**语法层**
    抓出"用了更高版本才支持的写法"。运行期差异（如导入 3.11 才有的标准库）
    由 ``test_no_unguarded_import_of_newer_stdlib`` 负责。
    """
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, feature_version=(3, 10))
        except SyntaxError as error:
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{error.lineno} {error.msg}")
    assert not offenders, "以下文件无法按 Python 3.10 语法解析：\n" + "\n".join(offenders)


def test_tomli_is_declared_for_python_310():
    """3.10 没有 tomllib，TOML 解析需要 tomli 回退——依赖必须显式声明。"""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tomli" in pyproject, "dev 依赖里应当为 Python 3.10 声明 tomli 回退"
    assert "python_version" in pyproject, "tomli 应当带 Python 版本条件"
