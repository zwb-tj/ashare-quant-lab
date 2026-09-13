"""复现脚本的测试（离线，不跑重活）。

只测"清单与校验逻辑"这两件容易出错的事：
① 步骤清单是否自洽（有命令、有产出、快速档与完整档的关系正确）；
② 逐字节校验是否能真的发现差异（相同=通过、改了内容=报差异、缺文件=报缺失）。

真正的跑图流程由 ``python scripts/reproduce_all.py`` 手动执行，不进 CI（否则每次要几分钟）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_all import Step, build_steps, sha256, verify_assets


def test_build_steps_fast_is_well_formed():
    steps = build_steps(full=False)
    assert steps, "快速档至少要有步骤"
    names = [step.name for step in steps]
    assert len(names) == len(set(names)), "步骤名必须唯一"
    for step in steps:
        assert step.command, f"{step.name} 缺命令"
        assert step.outputs, f"{step.name} 缺产出清单"
        assert step.description
        assert not step.needs_real_data, "快速档不该包含需要真实数据的步骤"
        argv = step.argv()
        assert argv[0] == sys.executable and argv[1:3] == ["-m", "aqlab.cli"]
        assert "OUT" in argv, "输出根必须用 OUT 占位，脚本运行时会替换"


def test_build_steps_full_extends_fast_and_marks_real_data():
    fast = {step.name for step in build_steps(full=False)}
    full_steps = build_steps(full=True)
    full = {step.name for step in full_steps}
    assert fast < full
    real = [step for step in full_steps if step.needs_real_data]
    assert real, "完整档必须包含需要真实数据的步骤"
    for step in real:
        assert any("data/universe/daily" in part for part in step.command), f"{step.name} 应引用真实数据目录"


def test_committed_assets_map_only_png_files():
    for step in build_steps(full=True):
        for produced, committed in step.assets.items():
            assert produced.endswith(".png") and committed.endswith(".png")
            assert produced in step.outputs, f"{step.name}: {produced} 应出现在 outputs 里"


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8"))
    return path


def test_verify_assets_detects_match_difference_and_missing(tmp_path):
    produced = tmp_path / "produced"
    assets = tmp_path / "assets"
    _write(produced / "charts" / "a.png", "PNG-A")
    _write(assets / "a.png", "PNG-A")                 # 一致
    _write(produced / "charts" / "b.png", "PNG-B")
    _write(assets / "b.png", "PNG-B-CHANGED")         # 不一致
    _write(assets / "c.png", "PNG-C")                 # 产出缺失
    _write(produced / "charts" / "d.png", "PNG-D")    # 已提交版本缺失

    step = Step(
        name="t",
        command=("plot",),
        outputs=("charts/a.png", "charts/b.png", "charts/c.png", "charts/d.png"),
        assets={
            "charts/a.png": "a.png",
            "charts/b.png": "b.png",
            "charts/c.png": "c.png",
            "charts/d.png": "d.png",
        },
    )
    rows = {row["asset"]: row["status"] for row in verify_assets([step], produced, assets)}
    assert rows == {"a.png": "match", "b.png": "DIFFERS", "c.png": "missing-produced", "d.png": "missing-committed"}


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    first = _write(tmp_path / "one.bin", "same")
    second = _write(tmp_path / "two.bin", "same")
    third = _write(tmp_path / "three.bin", "different")
    assert sha256(first) == sha256(second)
    assert sha256(first) != sha256(third)
    assert len(sha256(first)) == 64


def test_every_generated_chart_function_is_reachable_from_a_step():
    """README 里嵌的图必须都能被某个步骤重新生成——否则"可复现"就是空话。"""
    from aqlab.charts import __all__ as chart_functions  # noqa: F401  (存在性检查)

    produced = {name for step in build_steps(full=True) for name in step.outputs}
    for expected in (
        "charts/equity_curves.png",
        "charts/monthly_heatmap.png",
        "factor_ic/factor_ic.png",
        "factor_ic/quantile_returns.png",
        "factor_ic/ic_term_structure.png",
        "factor_schemes/scheme_curves.png",
    ):
        assert expected in produced, f"{expected} 没有对应的复现步骤"


def test_readme_embedded_assets_are_covered_by_the_manifest():
    """README 里引用到的 docs/assets/*.png 必须在复现清单里，否则校验会漏掉它。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set()
    for line in readme.splitlines():
        if "docs/assets/" in line and line.strip().startswith("!["):
            referenced.add(line.split("docs/assets/")[1].split(")")[0].strip())
    covered = {committed for step in build_steps(full=True) for committed in step.assets.values()}
    assert referenced, "README 里应该至少嵌入一张图"
    assert referenced <= covered, f"这些图没有复现步骤：{sorted(referenced - covered)}"
