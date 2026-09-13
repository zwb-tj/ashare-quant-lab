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

from reproduce_all import Step, build_steps, verify_assets


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


def _png(path: Path, size: tuple[int, int], color: str) -> Path:
    """用 matplotlib 造一张真 PNG（校验逻辑会读它的宽高）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    figure.patch.set_facecolor(color)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _step(names: tuple[str, ...]) -> Step:
    return Step(name="t", command=("plot",), outputs=tuple(f"charts/{name}" for name in names),
                assets={f"charts/{name}": name for name in names})


def test_verify_assets_reports_match_difference_and_missing_in_strict_mode(tmp_path):
    produced = tmp_path / "produced"
    assets = tmp_path / "assets"
    _write(produced / "charts" / "a.png", "PNG-A")
    _write(assets / "a.png", "PNG-A")                 # 一致
    _write(produced / "charts" / "b.png", "PNG-B")
    _write(assets / "b.png", "PNG-B-CHANGED")         # 不一致
    _write(assets / "c.png", "PNG-C")                 # 产出缺失
    _write(produced / "charts" / "d.png", "PNG-D")    # 已提交版本缺失

    rows = {row["asset"]: row["status"] for row in verify_assets([_step(("a.png", "b.png", "c.png", "d.png"))], produced, assets, strict=True)}
    assert rows == {"a.png": "match", "b.png": "DIFFERS", "c.png": "missing-produced", "d.png": "missing-committed"}


def test_non_strict_mode_tolerates_pixel_differences_but_not_size_differences(tmp_path):
    """CI 跑在 Linux、字体不同：同尺寸不同像素只报告；尺寸不同仍然致命。"""
    produced = tmp_path / "produced"
    assets = tmp_path / "assets"
    _png(produced / "charts" / "same.png", (240, 160), "white")
    _png(assets / "same.png", (240, 160), "black")            # 同尺寸、不同像素
    _png(produced / "charts" / "size.png", (240, 160), "white")
    _png(assets / "size.png", (300, 160), "white")            # 尺寸不同

    rows = {row["asset"]: row["status"] for row in verify_assets([_step(("same.png", "size.png"))], produced, assets, strict=False)}
    assert rows["same.png"] == "differs-pixels"
    assert rows["size.png"] == "DIFFERS"
    # 严格模式下，"同尺寸不同像素"同样算不一致
    strict_rows = {row["asset"]: row["status"] for row in verify_assets([_step(("same.png",))], produced, assets, strict=True)}
    assert strict_rows["same.png"] == "DIFFERS"


def test_summarize_verification_sets_the_exit_code_policy():
    """退出码策略：只有尺寸不同/缺失才失败；像素差异在非严格模式下不拦。"""
    from reproduce_all import summarize_verification

    rows = [
        {"asset": "a.png", "status": "match", "detail": ""},
        {"asset": "b.png", "status": "differs-pixels", "detail": "same size"},
    ]
    outcome = summarize_verification(rows)
    assert outcome["exit_code"] == 0 and outcome["exact"] == 1 and outcome["pixels"] == 1

    rows.append({"asset": "c.png", "status": "DIFFERS", "detail": "240x160 vs 300x160"})
    outcome = summarize_verification(rows)
    assert outcome["exit_code"] == 1 and [row["asset"] for row in outcome["broken"]] == ["c.png"]

    rows = [{"asset": "d.png", "status": "missing-produced", "detail": "charts/d.png"}]
    assert summarize_verification(rows)["exit_code"] == 1
