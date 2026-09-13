# -*- coding: utf-8 -*-
"""组合层成本检验 + 盈亏平衡成本（真实全市场）。

用法::

    python scripts/alpha_cost_check.py

前提：已用 ``python scripts/fetch_universe_daily.py`` 拉取全市场日线到 ``data/universe/daily``。
产物写入 ``output/alpha101_ic_real/``（与 CLI 的 ``output/alpha101_ic_cli/`` 分开，
避免带合成数据的冒烟运行覆盖真实结果）。
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.alpha101 import build_panel, compute_alphas  # noqa: E402
from aqlab.alpha_portfolio import (  # noqa: E402
    AlphaPortfolioConfig,
    break_even_cost_bps,
    cost_sensitivity,
    run_alpha_portfolio,
)
from aqlab.study_universe import load_universe_daily  # noqa: E402

start = time.time()
frames = load_universe_daily(ROOT / "data" / "universe" / "daily")
print(f"标的 {len(frames)} 只｜载入 {time.time() - start:.0f}s", flush=True)

panel = build_panel(frames)
values = compute_alphas(panel, min_history=260)
print(f"已算因子 {len(values)} 个｜用时 {time.time() - start:.0f}s", flush=True)

# 上一轮样本外存活 / 样本内显著的因子（来自 v0.26 的真实结果）
candidates = [
    "alpha_006", "alpha_044", "alpha_003", "alpha_026", "alpha_012",
    "alpha_013", "alpha_016", "alpha_008", "alpha_001", "alpha_005", "alpha_054",
]

rows: list[dict] = []
break_even: list[dict] = []
for name in candidates:
    if name not in values:
        continue
    factor = values[name]
    for hold in (1, 5, 20):
        config = AlphaPortfolioConfig(
            top_n=20,
            hold_days=hold,
            cost_bps=20.0,
            min_history=260,
            direction="ic_sign",     # 方向由样本内 IC 符号决定（不允许事后挑方向）
        )
        outcome = run_alpha_portfolio(factor, panel, config)
        summary = outcome["summary"]
        if not summary.get("periods"):
            continue
        be = break_even_cost_bps(factor, panel, config)
        rows.append(
            {
                "factor": name,
                "hold_days": hold,
                "direction": summary["direction"],
                "periods": summary["periods"],
                "mean_gross_excess": summary.get("mean_gross_excess"),
                "mean_net_excess": summary.get("mean_net_excess"),
                "net_excess_t": summary.get("net_excess_t"),
                "mean_turnover": summary.get("mean_turnover"),
                "excess_win_rate": summary.get("excess_win_rate"),
                "break_even_bps": be,
            }
        )
    print(f"  {name} 完成（{time.time() - start:.0f}s）", flush=True)

table = pd.DataFrame(rows)
out_dir = ROOT / "output" / "alpha101_ic_real"
out_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(out_dir / "alpha_portfolio_costs.csv", index=False, encoding="utf-8-sig")

view = table.copy()
for column in ("mean_gross_excess", "mean_net_excess", "excess_win_rate"):
    view[column] = (view[column].astype(float) * 100).round(2)
for column in ("net_excess_t", "mean_turnover", "break_even_bps"):
    view[column] = view[column].astype(float).round(2)
print()
print(view.to_string(index=False))

print()
print("=== 汇总 ===")
print(f"组合数：{len(table)}")
print(f"毛超额为正：{int((table['mean_gross_excess'] > 0).sum())} / {len(table)}")
print(f"净超额（扣 20bps）为正：{int((table['mean_net_excess'] > 0).sum())} / {len(table)}")
print(f"净超额 t > 2：{int((table['net_excess_t'] > 2).sum())} / {len(table)}")
print(f"盈亏平衡成本中位数：{table['break_even_bps'].median():.1f} bps")
print(f"平均换手：{table['mean_turnover'].mean():.3f}")

# 对表现最好的几个因子做成本敏感性
print()
print("=== 成本敏感性（毛超额最好的 3 个组合）===")
best = table.sort_values("mean_gross_excess", ascending=False).head(3)
for row in best.itertuples(index=False):
    factor = values[row.factor]
    config = AlphaPortfolioConfig(top_n=20, hold_days=int(row.hold_days), cost_bps=20.0, min_history=260, direction="ic_sign")
    sensitivity = cost_sensitivity(factor, panel, config)
    view2 = sensitivity.copy()
    for column in ("mean_gross_excess", "mean_net_excess"):
        view2[column] = (view2[column].astype(float) * 100).round(2)
    view2[["net_excess_t"]] = view2[["net_excess_t"]].astype(float).round(2)
    print(f"\n-- {row.factor} / 持有 {row.hold_days} 日 --")
    print(view2.to_string(index=False))

print()
print("已写入", out_dir)
