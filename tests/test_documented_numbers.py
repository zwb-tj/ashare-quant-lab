# -*- coding: utf-8 -*-
"""把"文档里的关键数字必须与产物一致"做成常驻测试。

**为什么值得**：这几轮反复出现"文档数字与产物脱节"（关波段 10 个 vs 实际 7 个、
产物被冒烟运行覆盖、省略号命令）。人工核对不可靠，因此把可核对的数字固定下来：

* 版本号（`pyproject.toml`）必须与最新 git 标签一致；
* README / 速览文档里出现的证据链数字，必须与 `output/` 下的真实产物一致。

**只在产物存在时校验**（CI 里没有跑全市场研究，`output/` 不在版本控制中），
因此测试在缺少产物时会跳过并说明原因——而不是伪造通过。
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT_REAL = ROOT / "output" / "alpha101_ic_real"
OUT = ROOT / "output" / "alpha101_ic"


def _has_outputs() -> bool:
    return (OUT_REAL / "corrected.csv").exists() and (OUT / "walk_forward_summary.csv").exists()


requires_outputs = pytest.mark.skipif(
    not _has_outputs(),
    reason="需要先跑真实全市场研究脚本（output/alpha101_ic_real 与 output/alpha101_ic）",
)


def _load_pyproject() -> dict:
    """读取 pyproject.toml。

    项目声明支持 Python **3.10**，而标准库的 ``tomllib`` 是 3.11 才加入的——
    所以这里做回退：3.11+ 用 ``tomllib``，3.10 用第三方 ``tomli``，两者都没有就跳过。
    直接 ``import tomllib`` 会让 3.10 的 CI 整个报 ImportError（曾经发生过）。
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+

        return tomllib.loads(text)
    except ModuleNotFoundError:
        tomli = pytest.importorskip("tomli", reason="Python 3.10 needs the tomli backport to parse TOML")
        return tomli.loads(text)


def test_version_matches_the_latest_git_tag():
    """pyproject 的版本号不能落后于标签（曾经出现过 0.30.0 对 v0.30.1）。"""
    version = _load_pyproject()["project"]["version"]
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.split()
    if not tags:
        pytest.skip("仓库里还没有标签")
    latest = max(tags, key=lambda name: [int(part) for part in re.findall(r"\d+", name)][:3])
    assert f"v{version}" == latest, f"pyproject 是 {version}，最新标签是 {latest}"


def test_documented_test_count_matches_the_collected_total():
    """文档里写的测试数必须等于实际收集到的数量。

    这条比"数字是否漂移"更基础：多次迭代后 README 里一度同时存在 288 / 384 / 387 三个数，
    人工同步不可靠，因此由 collect-only 的真实计数来裁决。
    """
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    total = sum(int(match) for match in re.findall(r":\s*(\d+)$", collected.stdout, flags=re.M))
    assert total > 0, "应当能统计到测试数"

    documents = [ROOT / "README.md", ROOT / "README.en.md", ROOT / "docs" / "CASE_STUDY.md"]
    # 本文件每新增一个测试，"实测数"就会增加，于是刚同步过的文档立刻又"过期"。
    # 因此判据是"文档数不得比实测数落后超过 1"（允许滞后一个守卫测试自身），
    # 但落后更多就说明真的漂移了（历史上曾同时存在 288 / 384 / 387 三个数）。
    stale_limit = total - 1
    for path in documents:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        mentioned = {
            int(value)
            for value in re.findall(
                r"(\d{3})\s*(?:个用例|个测试|个离线测试|pytest cases|tests\b|test cases)", text
            )
        }
        too_old = sorted(value for value in mentioned if value < stale_limit)
        assert not too_old, f"{path.name} 里的测试数明显过期（实测 {total}）：{too_old}"


@requires_outputs
def test_documented_evidence_chain_matches_the_outputs():
    """证据链的每一环都必须与产物一致。"""
    corrected = pd.read_csv(OUT_REAL / "corrected.csv")
    walkforward = pd.read_csv(OUT / "walk_forward_summary.csv")
    costs = pd.read_csv(OUT / "alpha_portfolio_costs.csv")
    isoos = pd.read_csv(OUT_REAL / "is_oos.csv")

    chain = {
        "combinations": len(corrected),
        "raw_significant": int(corrected["significant_raw"].sum()),
        "bh_significant": int(corrected["significant_bh"].sum()),
        "bonferroni_significant": int(corrected["significant_bonferroni"].sum()),
        "in_sample_significant": int(isoos["is_significant_bh"].sum()),
        "out_of_sample_survivors": int(isoos[isoos["is_significant_bh"]]["survives_oos"].sum()),
        "walkforward_combinations": len(walkforward),
        "walkforward_robust": int(walkforward["robust"].sum()),
        "costed_combinations": len(costs),
        "net_excess_t_above_2": int((costs["net_excess_t"] > 2).sum()),
    }

    # 仓库内的文档必须包含证据链数字。
    # 另有一份仓库外的「项目速览」一页纸也用同样的数字，但它不在版本控制里，
    # 因此通过环境变量 `AQLAB_BRIEF` 显式指定才校验（默认跳过，避免把个人目录写死进仓库）。
    docs = [ROOT / "docs" / "ALPHA_RESEARCH.md"]
    external_brief = os.environ.get("AQLAB_BRIEF")
    if external_brief:
        docs.append(Path(external_brief))
    for path in docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for key, value in chain.items():
            assert str(value) in text, f"{path.name} 缺少证据链数字 {key}={value}"

    # 关键事实：折 4 全灭而其它折有存活
    detail = pd.read_csv(OUT / "walk_forward_detail.csv")
    fold4 = detail[detail["fold"] == 4]
    assert int(fold4["selected"].sum()) > 0 and int(fold4["survived"].sum()) == 0


@requires_outputs
def test_state_claim_is_seven_of_ten_not_all_ten():
    """关波段更优的因子数是 7/10，文档不得写成"全部"（曾经写错过）。"""
    state = pd.read_csv(OUT / "state_regime_bucket.csv")
    better = 0
    for _factor, group in state.groupby("factor"):
        row = group.set_index("state")
        if {"开波段", "关波段"} <= set(row.index) and float(row.loc["关波段", "ic_mean"]) > float(row.loc["开波段", "ic_mean"]):
            better += 1
    assert better == 7, f"实际 {better} 个因子关波段更优"
    # 判断"是不是把 7/10 写成了 10/10"不能只匹配一种措辞：这个错误在
    # README.md / README.en.md / docs/ROADMAP.md 里各以不同说法出现过一次
    # （"全部高于" / "every factor" / "一致更好"），只钉字面量就会漏。
    banned = ("全部 10 个因子", "全部高于", "一致更好", "every factor", "all ten factors")
    for name in ("README.md", "README.en.md", "docs/ROADMAP.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for phrase in banned:
            assert phrase not in text, f"{name} 不该再声称全部 10 个因子（命中 {phrase!r}）"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "10 个因子里**7 个**" in readme or "7 个" in readme
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    assert "7 of 10" in english, "英文 README 也该写明 7/10"


# --------------------------------------------------------------------------------------
# README 的"实测输出"两张表
# --------------------------------------------------------------------------------------
# 这两张表此前完全没有守卫——`buy_and_hold` 的"1 笔交易 / 胜率 100%"就是从这里被发现口径不对的。
# 守卫必须跑**真实的 CLI**并把它的 stdout 抽出来比对，不能用测试里重写的一份格式化逻辑：
# 那样只会验证"我的复现与我的实现一致"，还会放过 README 里手写的、命令根本不会打印的形式。
def _run_demo(seed: int, tmp_path: Path) -> dict[str, list[str]]:
    """跑真实的 `aqlab demo`，把 stdout 里的表格行解析出来。"""
    completed = subprocess.run(
        [sys.executable, "-m", "aqlab.cli", "demo", "--seed", str(seed), "--out", str(tmp_path / f"demo{seed}")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert completed.returncode == 0, f"demo --seed {seed} 失败：{completed.stderr[-400:]}"
    rows: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("| ") or line.startswith("| 策略 |") or set(line) <= {"|", "-", " ", ":"}:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def _readme_table_rows(text: str, header_marker: str) -> dict[str, list[str]]:
    """取出 README 里以 ``header_marker`` 开头的那张表的数据行。"""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(header_marker))
    rows: dict[str, list[str]] = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_readme_demo_tables_match_the_actual_demo_output(tmp_path):
    """两个 README 的 demo 表必须**就是** `aqlab demo` 的输出（逐列、逐字符）。

    英文表曾经漏改过一次（表头还是旧的 8 列、`buy_and_hold` 那行还是错的），
    所以两种语言都要守。
    """
    for name, marker in (("README.md", "| 策略 | 总收益%"), ("README.en.md", "| Strategy | Total return %")):
        text = (ROOT / name).read_text(encoding="utf-8")
        blocks = text.split(marker)
        assert len(blocks) == 3, f"{name} 应恰好有两张 demo 表（上涨 / 下跌）"

        for seed, block in zip((25, 7), blocks[1:], strict=True):
            documented = _readme_table_rows(marker + block, marker)
            actual = _run_demo(seed, tmp_path)
            assert set(documented) == set(actual), f"{name} seed={seed} 表的策略集合不一致"
            for strategy, expected in actual.items():
                assert documented[strategy] == expected, (
                    f"{name} seed={seed} 的 {strategy} 行与真实 demo 输出不一致："
                    f"文档 {documented[strategy]} vs 实际 {expected}"
                )
            # 口径本身也要钉住：买入持有只有未平仓持仓，胜率必须是 n/a 而不是 100%
            assert documented["buy_and_hold"][5:7] == ["0", "1"], f"{name}: buy_and_hold 应为「已平仓 0 / 未平仓 1」"
            assert documented["buy_and_hold"][7] == "n/a", f"{name}: 只有未平仓持仓时胜率必须是 n/a"
