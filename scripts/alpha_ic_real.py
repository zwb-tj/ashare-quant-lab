# -*- coding: utf-8 -*-
"""全样本 IC + 多重比较校正 + 单次样本内外（真实全市场）。

用法::

    python scripts/alpha_ic_real.py

前提：已用 ``python scripts/fetch_universe_daily.py`` 拉取全市场日线到 ``data/universe/daily``。
产物写入 ``output/alpha101_ic_real/``（与 CLI 的 ``output/alpha101_ic_cli/`` 分开，
避免带合成数据的冒烟运行覆盖真实结果）。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.factor_eval import FactorEvalConfig, evaluate_factor_ics  # noqa: E402
from aqlab.study_universe import load_universe_daily  # noqa: E402

start = time.time()
frames = load_universe_daily(ROOT / "data" / "universe" / "daily")
print(f"标的 {len(frames)} 只｜载入 {time.time() - start:.0f}s", flush=True)

config = FactorEvalConfig(horizons=(1, 5, 20), step_days=5, min_history=260, min_symbols=200, alpha=0.05, is_fraction=0.5)
outcome = evaluate_factor_ics(frames, config)
table = outcome["table"]
full = outcome["full_significance"]
is_sig = outcome["significance_is"]

out_dir = ROOT / "output" / "alpha101_ic_real"
out_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(out_dir / "is_oos.csv", index=False, encoding="utf-8-sig")
full["table"].to_csv(out_dir / "corrected.csv", index=False, encoding="utf-8-sig")

print(f"因子 {table['factor'].nunique()} 个｜组合 {len(table)} 个｜切分日 {outcome['split_date'].date()}｜用时 {time.time() - start:.0f}s")
print()
print("=== 全样本：不校正 vs 校正 ===")
print(f"期望假阳性（{len(table)} 次检验 @5%）：{full['expected_false_positives']:.2f}")
print(f"不校正显著：{int(full['table']['significant_raw'].sum())}")
print(f"BH 校正后 ：{int(full['table']['significant_bh'].sum())}")
print(f"Bonferroni：{int(full['table']['significant_bonferroni'].sum())}")
print()
print("=== 单次 50/50：样本内 BH 显著 → 样本外 ===")
survivors = table[table["is_significant_bh"]]
print(f"样本内 BH 显著：{len(survivors)} 个组合")
print(f"样本外仍显著且同向：{int(survivors['survives_oos'].sum())} / {len(survivors)}")
print(f"样本外反向：{int((~survivors['os_same_sign']).sum())}")
print()
if not survivors.empty:
    view = survivors[["factor", "horizon", "is_ic_mean", "is_t_stat_adj", "os_ic_mean", "os_t_stat_adj", "os_same_sign", "survives_oos"]].copy()
    for column in ("is_ic_mean", "is_t_stat_adj", "os_ic_mean", "os_t_stat_adj"):
        view[column] = view[column].astype(float).round(4)
    print(view.sort_values("is_t_stat_adj", ascending=False).to_string(index=False))
print()
print("已写入", out_dir)
