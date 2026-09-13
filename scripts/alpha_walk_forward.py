# -*- coding: utf-8 -*-
"""多折 walk-forward 验证（真实全市场）。

用法::

    python scripts/alpha_walk_forward.py

前提：已用 ``python scripts/fetch_universe_daily.py`` 拉取全市场日线到 ``data/universe/daily``。
产物写入 ``output/alpha101_ic_real/``（与 CLI 的 ``output/alpha101_ic_cli/`` 分开，
避免带合成数据的冒烟运行覆盖真实结果）。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.study_universe import load_universe_daily  # noqa: E402
from aqlab.walk_forward_folds import FoldConfig, run_walk_forward  # noqa: E402

start = time.time()
frames = load_universe_daily(ROOT / "data" / "universe" / "daily")
print(f"标的 {len(frames)} 只｜载入 {time.time() - start:.0f}s", flush=True)

config = FoldConfig(
    horizons=(5, 20),
    step_days=5,
    min_history=260,
    min_symbols=200,
    n_folds=5,
    train_fraction=0.6,
    t_threshold=2.0,
)
outcome = run_walk_forward(frames, config)
detail, summary = outcome["detail"], outcome["summary"]
print(f"折数 {len(outcome['folds'])}｜明细 {len(detail)} 行｜用时 {time.time() - start:.0f}s", flush=True)

out_dir = ROOT / "output" / "alpha101_ic_real"
out_dir.mkdir(parents=True, exist_ok=True)
detail.to_csv(out_dir / "walk_forward_detail.csv", index=False, encoding="utf-8-sig")
summary.to_csv(out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")

print()
print("=== 折的划分 ===")
for fold in outcome["folds"]:
    print(f"  折 {fold['fold']}: 训练 {fold['train_start'].date()} 起 {len(fold['train'])} 个截面 -> "
          f"测试 {fold['test_start'].date()} 起 {len(fold['test'])} 个截面")

print()
print("=== 按因子汇总（前 20）===")
view = summary.head(20).copy()
for column in ("mean_test_ic", "mean_test_t", "median_test_t", "survival_rate"):
    view[column] = view[column].astype(float).round(4)
print(view.to_string(index=False))

print()
print("=== 汇总统计 ===")
print(f"(因子 × 持有期) 组合数：{len(summary)}")
print(f"至少被选中 1 折：{int((summary['folds_selected'] >= 1).sum())}")
print(f"至少被选中 2 折：{int((summary['folds_selected'] >= 2).sum())}")
print(f"被选中 ≥2 折且存活率 ≥60%（robust）：{int(summary['robust'].sum())}")
if summary["robust"].any():
    print()
    print("robust 因子明细：")
    robust = summary[summary["robust"]].copy()
    for column in ("mean_test_ic", "mean_test_t", "survival_rate"):
        robust[column] = robust[column].astype(float).round(4)
    print(robust.to_string(index=False))

# 逐折看"被选中的因子在测试段的表现"，这是最能说明稳定性的一张表
print()
print("=== 逐折：训练段达标（selected）的因子在测试段是否同向达标 ===")
selected = detail[detail["selected"]].copy()
if selected.empty:
    print("没有任何因子在任何一折的训练段达标")
else:
    pivot = selected.pivot_table(index=["factor", "horizon"], columns="fold", values="survived", aggfunc="first")
    print(pivot.fillna(False).astype(int).to_string())

print()
print("已写入", out_dir)
