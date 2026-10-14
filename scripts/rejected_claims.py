# -*- coding: utf-8 -*-
"""被否结论的可提交清单 + 自查（`--check` 不一致时退出非零）。

背景：`docs/CASE_STUDY.md` 的 "What was rejected and why" 一节里每条结论都带数字，而这些数字
来自 ``output/`` 下的真实全市场产物——``output/`` 是 gitignored 的。结果是**在全新克隆里那些
断言会整体跳过**：文档声称可复现的东西，在最需要被检查的地方没人检查。

这里把七条被否结论的数字落成一份**可提交**的清单（``docs/research/rejected-claims.json``），
由真实产物生成而不是手抄，并在仓库内固定下来。守卫测试读这份清单去核对文档，所以文档一旦
漂移就会失败，而无数据的环境也能跑。

核对的是"清单 == 产物"，能抓住两类漂移：产物被重算后清单过期、以及有人手改了清单。

用法::

    python scripts/rejected_claims.py            # 与产物对照，不一致则退出 1
    python scripts/rejected_claims.py --write    # 用当前产物重新生成清单（需要真实数据产物）
    python scripts/rejected_claims.py --print    # 只打印当前清单

``--check`` 是默认行为。产物缺失时打印提示并返回 0——无数据的克隆不应因此失败。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "docs" / "research" / "rejected-claims.json"

OUT = ROOT / "output"

COMMANDS = [
    "python scripts/fetch_universe_daily.py",
    "aqlab universe-study --daily-dir data/universe/daily --rule b1_graded --start 2025-01-01 "
    "--end 2026-09-11 --exit-mode entry_low --stop-pct 0.03 --take-profit 0.15 --min-holding 3 --out output",
    "python scripts/opening_filter_study.py",
    "aqlab picks-backtest --archive ./picks_archive --buckets b1,b2,n20,n30,v3 --dedupe-window 5 "
    "--horizons 1,3,5,10 --benchmark-sample 0 --out output",
    "aqlab picks-backtest --archive ./picks_archive --buckets b1 --horizons 1,3,5,10 --confirm "
    "--confirm-buckets b1 --confirm-entry window_close --benchmark-sample 0 --out output",
    "aqlab sweep --profile zgnb_full --set b1_graded.j_max=13,20,30 --set b1_graded.spike_multiple=2.0,1.8 "
    "--sizes 40,80 --days 900 --main-horizon 5",
    "aqlab factor-backtest --data-dir data/universe/daily "
    "--schemes fixed,sign_flip,ic_sign,ic_weight --forward 20 --step 5 --top-n 10 --lag 1 --cost-bps 20 --out output",
    "aqlab eval --mode offline",
]

ARTIFACTS = [
    "output/picks_backtest/confirm_summary.csv",
    "output/picks_backtest/confirm_ratio_std.csv",
    "output/picks_backtest/picks_evaluated.csv",
    "output/picks_backtest/summary.csv",
    "output/universe_study/trades_A.csv",
    "output/universe_study/trades_D.csv",
    "output/universe_study/risk_comparison.csv",
    "output/universe_study/opening_filter.csv",
    "output/universe_study/long_opening_filter.csv",
    "output/sweep/sweep.csv",
    "output/factor_schemes/schemes_summary.csv",
    "output/factor_schemes/schemes_detail.csv",
    "output/eval/report_offline.json",
]


def _t_stat(series: Any) -> float:
    """单样本 t：mean / (sd / sqrt(n))，NaN 与零方差返回 nan。"""
    values = series.dropna()
    n = len(values)
    if n < 2:
        return float("nan")
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(values.mean() / (sd / math.sqrt(n)))


def _excess(frame: Any, horizon: int = 10) -> dict[str, Any]:
    """一组信号的超额：**命中数**（matched）与**有报价数**（priced）都记，均值按 priced 算。

    文档里两种分母都出现过（量比门槛表用命中数，阈值扫描表用有报价数），所以两个都留下来，
    守卫测试才能把每一处引用钉到正确的分母上。
    """
    column = f"excess_{horizon}"
    values = frame[column].dropna()
    return {
        "matched": len(frame),
        "priced": len(values),
        "excess_10_pct": round(float(values.mean()) * 100.0, 2),
    }


def build_manifest() -> dict[str, Any]:
    """从真实产物算出被否结论清单。产物缺失时抛 FileNotFoundError。"""
    import pandas as pd

    confirm = pd.read_csv(OUT / "picks_backtest" / "confirm_summary.csv")
    ratio = pd.read_csv(OUT / "picks_backtest" / "confirm_ratio_std.csv")
    picks = pd.read_csv(OUT / "picks_backtest" / "picks_evaluated.csv")
    summary = pd.read_csv(OUT / "picks_backtest" / "summary.csv")
    sweep = pd.read_csv(OUT / "sweep" / "sweep.csv")
    schemes = pd.read_csv(OUT / "factor_schemes" / "schemes_summary.csv")
    risk = pd.read_csv(OUT / "universe_study" / "risk_comparison.csv")
    report = json.loads((OUT / "eval" / "report_offline.json").read_text(encoding="utf-8"))

    # ---- 1 / 2 量比门槛：相对口径按 decision 分组，软件口径按 ratio_std 与方向切 ----
    def decision_row(label: str) -> Any:
        return confirm[confirm["decision"] == label].iloc[0]

    selected = decision_row("买入")
    excluded = decision_row("观望")
    everything = decision_row("全部")
    relative_scan = {
        str(thr): _excess(ratio[ratio["volume_ratio"] >= thr])
        for thr in (0.5, 0.8, 1.0, 1.5)
    }

    # 「买入 vs 观望」的差距与 Welch t。注意两者分母不同：摘要表的 n（26/120）是**信号数**，
    # 而可比收益要落在同一天有报价的 127 行上（21/106）——t 只能按后者算。
    def _welch(values_a: Any, values_b: Any) -> float:
        a = values_a.dropna().to_numpy()
        b = values_b.dropna().to_numpy()
        variance = a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
        if variance <= 0.0:
            return float("nan")
        return float((a.mean() - b.mean()) / math.sqrt(variance))

    gate_selected = ratio[ratio["volume_ratio"] >= 1.0]["excess_10"]
    gate_excluded = ratio[ratio["volume_ratio"] < 1.0]["excess_10"]
    relative_spread = {
        "spread_pp": round(float(selected["excess_10"] - excluded["excess_10"]) * 100.0, 2),
        "welch_t": round(_welch(gate_selected, gate_excluded), 2),
        "priced_selected": int(gate_selected.notna().sum()),
        "priced_excluded": int(gate_excluded.notna().sum()),
    }

    # 软件口径的两个扫描都跑在**全 146 行**上：命中数（matched）才是文档量比门槛表的口径，
    # 有报价数（priced）才是阈值扫描表的口径，两个数都得留。
    software_scan = {
        str(thr): _excess(ratio[ratio["ratio_std"] >= thr]) for thr in (3, 4, 5)
    }
    direction = {
        f"{thr}_{side}": _excess(
            ratio[(ratio["ratio_std"] >= thr) & (ratio["ret_vs_open"] > 0 if side == "up" else ratio["ret_vs_open"] <= 0)]
        )
        for thr in (3, 4, 5)
        for side in ("up", "down")
    }

    # ---- 3 结构出场：两个口径的离场原因分布 ----
    def exits(tag: str) -> dict[str, Any]:
        frame = pd.read_csv(OUT / "universe_study" / f"trades_{tag}.csv", usecols=["reason", "return_pct"])
        grouped = frame.groupby("reason")["return_pct"].agg(["count", "mean"])
        return {
            "n": len(frame),
            "reasons": {
                str(reason): {
                    "count": int(row["count"]),
                    "share_pct": round(float(row["count"]) / len(frame) * 100.0, 2),
                    "mean_return_pct": round(float(row["mean"]) * 100.0, 2),
                }
                for reason, row in grouped.iterrows()
            },
        }

    tail = {
        "plain_hold_20d_worst_pct": float(risk.iloc[1]["最差单笔%"]),
        "full_exit_rules_worst_pct": float(risk.iloc[2]["最差单笔%"]),
    }

    # ---- 2 开盘窗口特征：短窗口有优势、放到 21 个月就没了。表是中文列名，两张表列序不同 ----
    def opening_row(relative: str, group: str, excess_index: int) -> dict[str, Any]:
        frame = pd.read_csv(OUT / "universe_study" / relative)
        match = frame[frame[frame.columns[0]] == group].iloc[0]
        return {
            "trades": int(match[frame.columns[1]]),
            "excess_pct": round(float(match[frame.columns[excess_index]]), 2),
            "excess_t": round(float(match[frame.columns[excess_index + 1]]), 2),
        }

    opening = {
        "_claim": "高量比意味着资金在吸筹。",
        "short_window": {
            "volume_ratio_ge_4": opening_row("opening_filter.csv", "量比>=4", 3),
            "rising_volume": opening_row("opening_filter.csv", "量能递增", 3),
        },
        "long_window": {"volume_ratio_ge_4": opening_row("long_opening_filter.csv", "量比≥4", 5)},
    }

    # ---- 4 入场阈值：只取 80 只池子 + spike 1.8 的三档，再看 40 只池子 ----
    def sweep_row(j_max: int, spike: float, universe: str) -> dict[str, Any]:
        match = sweep[
            sweep["params"].str.contains(f"j_max={j_max}", regex=False)
            & sweep["params"].str.endswith(f"spike_multiple={spike}")
            & (sweep["universe"] == universe)
        ].iloc[0]
        return {
            "avg_positions": round(float(match["avg_positions"]), 2),
            "excess_mean_pct": round(float(match["excess_mean"]) * 100.0, 2),
            "excess_win_rate_pct": round(float(match["excess_win_rate"]) * 100.0, 2),
        }

    # ---- 5 发布记录：6 桶（含整表）× 4 持有期 = 24 格；t 只能自己算 ----
    cells: list[dict[str, Any]] = []
    for horizon in (1, 3, 5, 10):
        column = f"excess_{horizon}"
        groups = [(str(name), group) for name, group in picks.groupby("bucket")]
        groups.append(("all", picks))
        for name, group in groups:
            values = group[column].dropna()
            cells.append({
                "bucket": name,
                "horizon": horizon,
                "n": len(values),
                "excess_pct": round(float(values.mean()) * 100.0, 2),
                "t": round(_t_stat(values), 2),
            })
    whole = summary[summary["bucket"] == "全部"].iloc[0]

    published = {
        "records": int(whole["picks"]),
        "cells": len(cells),
        "negative_t_cells": sum(1 for cell in cells if cell["t"] < -2),
        "positive_t_cells": sum(1 for cell in cells if cell["t"] > 2),
        "all_picks_excess_pct": [
            round(float(whole[f"excess_{h}"]) * 100.0, 2) for h in (1, 3, 5, 10)
        ],
        "all_picks_t": [next(cell["t"] for cell in cells if cell["bucket"] == "all" and cell["horizon"] == h) for h in (1, 3, 5, 10)],
        "cells_detail": cells,
    }

    # ---- 6 因子权重方向 ----
    benchmark = float(
        pd.read_csv(OUT / "factor_schemes" / "schemes_detail.csv")["benchmark"].mean()
    )
    sign_flip = {
        "benchmark_mean_pct": round(benchmark * 100.0, 2),
        "schemes": {
            str(row["scheme"]): {
                "periods": int(row["periods"]),
                "mean_net_pct": round(float(row["mean_net"]) * 100.0, 2),
                "mean_excess_pct": round(float(row["mean_excess"]) * 100.0, 2),
                "excess_t": round(float(row["excess_t"]), 2),
            }
            for _, row in schemes.iterrows()
        },
    }

    # ---- 7 Agent 落地率 ----
    metrics = report["metrics"]
    agent = {
        "tasks": int(metrics["tasks"]),
        "numbers_total": int(metrics["numbers_total"]),
        "hallucinated_tasks": int(metrics["hallucinated_tasks"]),
        "grounded_number_rate": round(float(metrics["grounded_number_rate"]), 3),
    }

    return {
        "_note": (
            "由 scripts/rejected_claims.py --write 从 output/ 下的真实全市场产物生成；"
            "数字以本清单为准。核验：python scripts/rejected_claims.py"
        ),
        "provenance": {
            "universe": "全市场 A 股日线（stockdb 导出至 data/universe/daily）+ picks_archive（个人选股日志）",
            "reproduce": COMMANDS,
            "artifacts": ARTIFACTS,
            "note": (
                "产物与 picks_archive 都不入库（体积与数据许可原因），"
                "因此 CI 校验的是本清单与文档，而不是重跑数据。"
            ),
        },
        "volume_gate": {
            "_claim": "高量比 + 9:37 向上意味着资金流入，应当买。",
            "relative_caliber": {
                "selected": {"n": int(selected["signals"]), "excess_10_pct": round(float(selected["excess_10"]) * 100.0, 2)},
                "excluded": {"n": int(excluded["signals"]), "excess_10_pct": round(float(excluded["excess_10"]) * 100.0, 2)},
                "all": {"n": int(everything["signals"]), "excess_10_pct": round(float(everything["excess_10"]) * 100.0, 2)},
                "relative_spread": relative_spread,
            },
            "software_caliber": {"ratio_std_4_up": software_scan["4"] | direction["4_up"]},
            "scan_relative": relative_scan,
            "scan_software_undirected": software_scan,
        },
        "opening_features": opening,
        "robustness_grid": {
            "cells": len(direction),
            "means_pct": sorted(round(value["excess_10_pct"], 2) for value in direction.values()),
            "mean_pct": round(sum(value["excess_10_pct"] for value in direction.values()) / len(direction), 2),
            "median_pct": round(
                float(pd.Series([value["excess_10_pct"] for value in direction.values()]).median()), 2
            ),
            "all_negative": all(value["excess_10_pct"] < 0 for value in direction.values()),
        },
        "exits": {
            "_claim": "结构出场（白线破位 / 趋势线）保护利润。",
            "structural_only": exits("A"),
            "sop_tight": exits("D"),
            "tail_risk": tail,
        },
        "entry_threshold": {
            "_claim": "放宽 j_max 能缓解持仓集中。",
            "pool_80_spike_1_8": {
                "j_max_13": sweep_row(13, 1.8, "80只"),
                "j_max_30": sweep_row(30, 1.8, "80只"),
            },
            "pool_40_spike_2_0_j_max_13": sweep_row(13, 2.0, "40只"),
        },
        "published_picks": {"_claim": "选股规则带有可交易优势。", **published},
        "sign_flip": {
            "_claim": "负 IC 因子把权重取反就能做成多头。",
            **sign_flip,
        },
        "agent_grounding": {
            "_claim": "Agent 的回答可以直接采信。",
            **agent,
        },
    }


def compare(manifest: Any, actual: Any, path: str = "") -> list[str]:
    """递归比较清单与产物，返回不一致描述（空列表 = 一致）。"""
    problems: list[str] = []
    if isinstance(manifest, dict) and isinstance(actual, dict):
        for key, value in actual.items():
            if key not in manifest:
                problems.append(f"{path}{key} 缺失（产物为 {value!r}）")
            else:
                problems.extend(compare(manifest[key], value, f"{path}{key}."))
        for key in manifest:
            if key not in actual:
                problems.append(f"{path}{key} 在产物中不存在（清单为 {manifest[key]!r}）")
        return problems
    if manifest != actual:
        problems.append(f"{path.rstrip('.')} 不一致：清单 {manifest!r} vs 产物 {actual!r}")
    return problems


def _summary(manifest: dict[str, Any]) -> list[str]:
    grid = manifest["robustness_grid"]
    published = manifest["published_picks"]
    agent = manifest["agent_grounding"]
    return [
        f"量比门槛：相对口径差 {manifest['volume_gate']['relative_caliber']['selected']['excess_10_pct']}%"
        f" vs {manifest['volume_gate']['relative_caliber']['excluded']['excess_10_pct']}%",
        f"稳健性网格：{grid['cells']} 格，均值 {grid['mean_pct']}% / 中位 {grid['median_pct']}%，全为负={grid['all_negative']}",
        f"发布记录：{published['records']} 条 / {published['cells']} 格，"
        f"显著为负 {published['negative_t_cells']}、显著为正 {published['positive_t_cells']}",
        f"Agent 落地：{agent['numbers_total']} 个数字，{agent['hallucinated_tasks']} 个任务被标记为编造",
    ]

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="committed manifest for the rejected-conclusion claims")
    parser.add_argument("--write", action="store_true", help="用当前产物重新生成清单")
    parser.add_argument("--print", dest="show", action="store_true", help="只打印清单")
    args = parser.parse_args(argv)

    if args.show:
        print(MANIFEST.read_text(encoding="utf-8"))
        return 0

    if args.write:
        try:
            actual = build_manifest()
        except FileNotFoundError as exc:
            print(f"产物缺失，无法重新生成清单：{exc}")
            return 1
        MANIFEST.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
        )
        print(f"已写入 {MANIFEST.relative_to(ROOT)}")
        return 0

    if not MANIFEST.is_file():
        print(f"缺少清单：{MANIFEST}（用 --write 生成，需要真实产物）")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    try:
        actual = build_manifest()
    except FileNotFoundError as exc:
        print(f"提示：产物缺失（{exc}），跳过与产物的对照；清单本身仍由测试与文档核对。")
        return 0

    problems = compare(manifest, actual, "")
    if problems:
        print("=== 被否结论清单与产物不一致 ===")
        for item in problems:
            print(f"FAIL {item}")
        print()
        print("如产物是刚重算的，运行 `python scripts/rejected_claims.py --write` 更新清单。")
        return 1

    print("=== 被否结论清单 ===")
    for line in _summary(manifest):
        print(f"OK  {line}")
    print()
    print("清单与产物一致。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
