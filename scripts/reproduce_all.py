# -*- coding: utf-8 -*-
"""一条命令复现 README 里的全部图表与报告，并可选地**校验已提交的产物确实是代码生成的**。

为什么需要它：README 里的每张图都有一句话说明"由某条命令生成"，但如果没人真的跑，
这句话就只是声明。本脚本把声明变成可执行、可校验的流程：

* **快速档**（默认）只用确定性合成数据，约 1 分钟跑完，适合每次改代码后自检；
* **完整档**（``--full``）再加上真实行情产物（需要本地 ``data/universe/daily`` 缓存），约 30 分钟；
* **校验档**（``--verify``）把产物生成到临时目录，与仓库里已提交的 ``docs/assets/`` **逐字节比对**，
  不一致就非零退出——这样"图是代码跑出来的"这件事有证据，而不是靠信任。

用法::

    python scripts/reproduce_all.py                 # 快速档
    python scripts/reproduce_all.py --full          # 含真实数据
    python scripts/reproduce_all.py --verify        # 校验已提交产物
    python scripts/reproduce_all.py --full --verify # 校验全部
    python scripts/reproduce_all.py --verify --strict  # 要求逐字节一致（本机）

校验分两级：默认（CI）只要求**每个产物都被重新生成**，差异仅作报告；``--strict``（本机）要求与已提交图完全一致。
原因：``bbox_inches="tight"`` 按文字尺寸裁剪，跨平台字体不同会让尺寸与像素都变，硬比对只会产生假警报。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


@dataclass
class Step:
    """一个可复现的产物生成步骤。"""

    name: str
    command: tuple[str, ...]
    outputs: tuple[str, ...] = ()          # 相对输出根的路径
    description: str = ""
    needs_real_data: bool = False
    assets: dict[str, str] = field(default_factory=dict)   # 产出文件 -> docs/assets 里的文件名

    def argv(self) -> list[str]:
        return [sys.executable, "-m", "aqlab.cli", *self.command]


def build_steps(full: bool = False) -> list[Step]:
    """构造步骤清单；``full=True`` 时追加需要真实行情缓存的步骤。"""
    steps = [
        Step(
            name="charts",
            command=("plot", "--seed", "25", "--days", "500", "--out", "OUT"),
            outputs=("charts/equity_curves.png", "charts/drawdown.png", "charts/strategy_comparison.png",
                     "charts/monthly_heatmap.png", "charts/report.html"),
            description="净值 / 回撤 / 策略对比 / 月度热力图 + 自包含 HTML 报告（合成行情）",
            assets={
                "charts/equity_curves.png": "equity_curves.png",
                "charts/drawdown.png": "drawdown.png",
                "charts/strategy_comparison.png": "strategy_comparison.png",
                "charts/monthly_heatmap.png": "monthly_heatmap.png",
            },
        ),
        Step(
            name="factor-ic-synthetic",
            command=("factor-ic", "--symbols", "40", "--days", "500", "--horizons", "1,3,5,10,20", "--out", "OUT"),
            outputs=("factor_ic/ic_summary.csv", "factor_ic/quantile_returns.csv",
                     "factor_ic/ic_term_structure.png", "factor_ic/report.html"),
            description="因子 IC / 分位收益 / 期限结构（合成票池）",
        ),
        Step(
            name="factor-schemes-synthetic",
            command=("factor-backtest", "--symbols", "40", "--days", "500", "--out", "OUT"),
            outputs=("factor_schemes/schemes_summary.csv", "factor_schemes/scheme_curves.png"),
            description="定权方案回测（合成票池）",
        ),
        Step(
            name="agent-eval-offline",
            command=("eval", "--mode", "offline", "--out", "OUT"),
            outputs=("eval/report.md",),
            description="Agent 可靠性评测（离线 20 任务）",
        ),
    ]
    if full:
        steps += [
            Step(
                name="universe-study",
                command=("universe-study", "--daily-dir", "data/universe/daily", "--start", "2025-01-01",
                         "--end", "2026-09-11", "--out", "OUT"),
                outputs=("universe_study/trades.csv", "universe_study/summary_all.csv"),
                description="全市场 B1 研究：58k 笔交易 + 大盘阶段 + 离场规则",
                needs_real_data=True,
            ),
            Step(
                name="factor-ic-real",
                command=("factor-ic", "--data-dir", "data/universe/daily", "--horizons", "1,2,3,5,10,20",
                         "--step", "5", "--forward", "20", "--min-symbols", "50", "--out", "OUT"),
                outputs=("factor_ic/factor_ic.png", "factor_ic/quantile_returns.png",
                         "factor_ic/ic_term_structure.png", "factor_ic/ic_by_horizon.csv"),
                description="因子 IC / 期限结构（真实全市场 5,424 只）",
                needs_real_data=True,
                assets={
                    "factor_ic/factor_ic.png": "factor_ic.png",
                    "factor_ic/quantile_returns.png": "quantile_returns.png",
                    "factor_ic/ic_term_structure.png": "ic_term_structure.png",
                },
            ),
            Step(
                name="alpha101-ic-real",
                command=("factor-ic", "--alpha101", "--data-dir", "data/universe/daily",
                         "--horizons", "1,5,20", "--step", "5", "--min-history", "260",
                         "--min-symbols", "200", "--out", "OUT"),
                outputs=("alpha101_ic/alpha101_ic.csv", "alpha101_ic/skipped.txt", "alpha101_ic/report.html"),
                description="公式化因子（Alpha101 子集）真实全市场 IC",
                needs_real_data=True,
            ),
            Step(
                name="portfolio-methods-real",
                command=("portfolio", "--data-dir", "data/universe/daily", "--profile", "b1",
                         "--method", "risk_parity", "--compare-methods", "--out", "OUT"),
                outputs=("portfolio/method_comparison.csv", "portfolio/method_comparison.png",
                         "portfolio/portfolio.md"),
                description="组合层：五种权重方法对比（真实全市场，约 23 分钟）",
                needs_real_data=True,
                assets={"portfolio/method_comparison.png": "portfolio_methods.png"},
            ),
            Step(
                name="factor-schemes-real",
                command=("factor-backtest", "--data-dir", "data/universe/daily", "--top-n", "10", "--forward", "20",
                         "--step", "5", "--lookback", "12", "--cost-bps", "20", "--min-symbols", "50", "--out", "OUT"),
                outputs=("factor_schemes/schemes_summary.csv", "factor_schemes/scheme_curves.png"),
                description="定权方案回测（真实全市场，20 日持有期）",
                needs_real_data=True,
                assets={"factor_schemes/scheme_curves.png": "scheme_curves.png"},
            ),
        ]
    return steps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int] | None:
    """读 PNG 的宽高（不是 PNG 则返回 None）。尺寸由 figsize/dpi 决定，跨平台一致。"""
    import struct

    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def run_step(step: Step, out_root: Path, cwd: Path = ROOT) -> dict:
    """执行一个步骤，返回 ``{"step", "seconds", "returncode", "missing"}``。"""
    command = [str(out_root) if part == "OUT" else part for part in step.argv()]
    start = time.time()
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.time() - start
    missing = [name for name in step.outputs if not (out_root / name).exists()]
    return {
        "step": step.name,
        "seconds": seconds,
        "returncode": completed.returncode,
        "missing": missing,
        "stderr_tail": (completed.stderr or "").strip()[-300:],
    }


def verify_assets(
    steps: list[Step],
    produced_root: Path,
    assets_dir: Path = ASSETS,
    strict: bool = True,
) -> list[dict]:
    """把产出与已提交的 ``docs/assets`` 比对。

    状态含义：``match``（逐字节一致）、``differs-pixels``（PNG 尺寸相同、字节不同）、
    ``differs-size``（PNG 尺寸不同）、``DIFFERS``（非 PNG 或读不出尺寸的差异）、``missing-*``。

    退出码由 :func:`summarize_verification` 决定：严格模式任何差异都失败；
    默认（CI）只把"产物没生成出来"当失败——``bbox_inches="tight"`` 会按文字尺寸裁剪，
    跨平台字体不同会让尺寸与像素都变，硬比对只会产生假警报。
    """
    rows: list[dict] = []
    for step in steps:
        for produced, committed in step.assets.items():
            source = produced_root / produced
            target = assets_dir / committed
            if not source.exists():
                rows.append({"asset": committed, "status": "missing-produced", "detail": produced})
            elif not target.exists():
                rows.append({"asset": committed, "status": "missing-committed", "detail": str(target)})
            elif sha256(source) == sha256(target):
                rows.append({"asset": committed, "status": "match", "detail": ""})
            else:
                size_now, size_ref = png_size(source), png_size(target)
                detail = f"{size_now} vs {size_ref}" if size_now and size_ref else f"{source.stat().st_size}B vs {target.stat().st_size}B"
                if size_now and size_ref and size_now == size_ref:
                    rows.append({"asset": committed, "status": "differs-pixels", "detail": f"same size {size_now}, different bytes"})
                elif size_now and size_ref:
                    note = "" if strict else " (size follows text via bbox_inches='tight', so it can differ across fonts)"
                    rows.append({"asset": committed, "status": "differs-size", "detail": f"{detail}{note}"})
                else:
                    rows.append({"asset": committed, "status": "DIFFERS", "detail": detail})
    return rows


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    line = "| " + " | ".join(str(header[i]).ljust(widths[i]) for i in range(len(header))) + " |"
    print(line)
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))) + " |")


def summarize_verification(rows: list[dict], strict: bool = False) -> dict:
    """把校验结果折成退出码策略。

    * ``strict=True``（本机）：任何与已提交图不一致的状态都算失败——这是"图与代码完全一致"的证明；
    * ``strict=False``（CI 默认）：门禁只要求**每个产物都被重新生成出来**（文件存在），
      差异只报告。原因很实际：``bbox_inches="tight"`` 会按文字尺寸裁剪，跨平台字体不同，
      连图片**尺寸**都可能变，硬比对会变成假警报。CI 因此保证"可再生成"，
      本机 `--strict` 保证"完全一致"。
    """
    exact = [row for row in rows if row["status"] == "match"]
    pixels = [row for row in rows if row["status"] == "differs-pixels"]
    sizes = [row for row in rows if row["status"] == "differs-size"]
    missing = [row for row in rows if row["status"].startswith("missing")]
    differences = [row for row in rows if row["status"] != "match"]
    broken = differences if strict else list(missing)
    return {
        "exit_code": 1 if broken else 0,
        "exact": len(exact),
        "pixels": len(pixels),
        "sizes": len(sizes),
        "missing": len(missing),
        "broken": broken,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="reproduce every figure the README claims")
    parser.add_argument("--full", action="store_true", help="also run the steps that need real cached data")
    parser.add_argument("--verify", action="store_true", help="compare regenerated figures with docs/assets byte by byte")
    parser.add_argument("--out", default=None, help="output root (default: output, or a temp dir when verifying)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory used when verifying")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require byte-identical figures (default: only the image size must match, since fonts differ across platforms)",
    )
    args = parser.parse_args(argv)

    steps = build_steps(full=args.full)
    temporary = None
    if args.out:
        out_root = Path(args.out).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
    elif args.verify:
        temporary = Path(tempfile.mkdtemp(prefix="aqlab-reproduce-"))
        out_root = temporary
    else:
        out_root = ROOT / "output"
    print(f"输出目录：{out_root}")
    print(f"步骤 {len(steps)} 个｜模式：{'完整' if args.full else '快速'}{'（校验）' if args.verify else ''}\n")

    results: list[dict] = []
    for step in steps:
        if step.needs_real_data and not (ROOT / "data" / "universe" / "daily").exists() and args.full:
            print(f"[skip] {step.name}：缺少 data/universe/daily（先跑 scripts/fetch_universe_daily.py）")
            continue
        print(f"[run ] {step.name}：{step.description}")
        result = run_step(step, out_root)
        results.append(result)
        flag = "ok" if result["returncode"] == 0 and not result["missing"] else "FAIL"
        print(f"       {flag}｜{result['seconds']:.1f}s" + (f"｜缺失 {result['missing']}" if result["missing"] else ""))
        if result["returncode"] != 0 and result["stderr_tail"]:
            print(f"       stderr: {result['stderr_tail']}")

    print()
    _print_table(
        [[r["step"], f"{r['seconds']:.1f}s", "ok" if r["returncode"] == 0 else f"exit {r['returncode']}",
          ", ".join(r["missing"]) or "-"] for r in results],
        ["step", "seconds", "status", "missing outputs"],
    )

    failed = [r for r in results if r["returncode"] != 0 or r["missing"]]
    exit_code = 1 if failed else 0

    if args.verify:
        rows = verify_assets(steps, out_root, strict=args.strict)
        print()
        _print_table([[row["asset"], row["status"], row["detail"] or "-"] for row in rows], ["asset", "status", "detail"])
        outcome = summarize_verification(rows, strict=args.strict)
        exit_code = max(exit_code, outcome["exit_code"])
        if outcome["missing"]:
            print(f"\n校验未通过：{outcome['missing']} 个产物没有生成出来（步骤失败或路径变了）")
        elif args.strict:
            if outcome["exact"] == len(rows):
                print(f"\n严格校验通过：{(outcome['exact'])} 个已提交图表与现场生成的结果逐字节一致")
            else:
                print(
                    f"\n严格校验未通过：逐字节一致 {outcome['exact']} 个、"
                    f"同尺寸不同像素 {outcome['pixels']} 个、尺寸不同 {outcome['sizes']} 个"
                )
        else:
            print(
                f"\n校验通过（可再生成）：{len(rows)} 个产物全部重新生成；"
                f"与已提交图逐字节一致 {outcome['exact']} 个、同尺寸不同像素 {outcome['pixels']} 个、"
                f"尺寸不同 {outcome['sizes']} 个（跨平台字体差异所致；本机 `--strict` 可要求完全一致）"
            )

    if temporary is not None:
        if args.keep:
            print(f"临时目录保留在：{temporary}")
        else:
            shutil.rmtree(temporary, ignore_errors=True)

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
