# -*- coding: utf-8 -*-
"""市场状态依赖性分析（真实全市场）。

用法::

    python scripts/alpha_state_dependence.py

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

from aqlab.rules_zgnb import ActiveMarketValueGate  # noqa: E402
from aqlab.state_dependence import StateConfig, factor_state_ic, summarize_state_dependence  # noqa: E402
from aqlab.study_universe import load_universe_daily  # noqa: E402

start = time.time()
frames = load_universe_daily(ROOT / "data" / "universe" / "daily")
print(f"标的 {len(frames)} 只｜载入 {time.time() - start:.0f}s", flush=True)

# 0AMV 波段状态（已滞后一天）
gate = ActiveMarketValueGate()
regime_raw = gate.gate_series(dict(frames))
regime = regime_raw["gate"].shift(1).fillna(0).astype(int)
print(f"0AMV 波段序列：{len(regime)} 天｜开波段占比 {float(regime.mean()) * 100:.1f}%", flush=True)

# 上一轮多折里"稳健"或"曾被选中"的因子
factors = (
    "alpha_013", "alpha_016", "alpha_050", "alpha_015", "alpha_003",
    "alpha_044", "alpha_026", "alpha_055", "alpha_006", "alpha_012",
)
config = StateConfig(horizons=(5,), step_days=5, min_history=260, min_symbols=200, quantiles=3, factors=factors)
outcome = factor_state_ic(frames, config, regime=regime)
detail = outcome["detail"]
print(f"明细 {len(detail)} 行｜用时 {time.time() - start:.0f}s", flush=True)

out_dir = ROOT / "output" / "alpha101_ic_real"
out_dir.mkdir(parents=True, exist_ok=True)
detail.to_csv(out_dir / "state_detail.csv", index=False, encoding="utf-8-sig")

for column, label in (("trend_bucket", "市场动量"), ("dispersion_bucket", "横截面离散度"), ("regime_bucket", "0AMV 波段")):
    table = summarize_state_dependence(detail, column)
    if table.empty:
        print(f"\n=== 按{label}分组：无可用数据 ===")
        continue
    table.to_csv(out_dir / f"state_{column}.csv", index=False, encoding="utf-8-sig")
    view = table.copy()
    view["ic_mean"] = view["ic_mean"].astype(float).round(4)
    view["t_stat"] = view["t_stat"].astype(float).round(2)
    view["positive_rate"] = (view["positive_rate"].astype(float) * 100).round(1)
    view["ic_spread"] = view["ic_spread"].astype(float).round(4)
    print(f"\n=== 按{label}分组（{column}）===")
    print(view.sort_values(["factor", "state"]).to_string(index=False))

print()
print("=== 状态依赖强度排名（IC 组间极差）===")
trend = summarize_state_dependence(detail, "trend_bucket")
disp = summarize_state_dependence(detail, "dispersion_bucket")
if not trend.empty:
    ranking = trend.groupby("factor")["ic_spread"].first().sort_values(ascending=False)
    print("按市场动量分组：")
    print(ranking.round(4).to_string())
if not disp.empty:
    ranking = disp.groupby("factor")["ic_spread"].first().sort_values(ascending=False)
    print("按离散度分组：")
    print(ranking.round(4).to_string())

# 直接检验折 4：那段窗口的状态档位 + IC 表现
print()
print("=== 折 4 窗口（2026-03-27 ~ 2026-06-25）的状态与 IC ===")
detail["date"] = pd.to_datetime(detail["date"])
fold4 = detail[(detail["date"] >= "2026-03-27") & (detail["date"] <= "2026-06-25")]
other = detail[(detail["date"] < "2026-03-27") | (detail["date"] > "2026-06-25")]
if not fold4.empty:
    print(f"折 4 截面数 {fold4['date'].nunique()}｜平均 IC {float(fold4['ic'].mean()):+.4f}｜正 IC 占比 {float((fold4['ic'] > 0).mean()) * 100:.1f}%")
    print(f"其它   截面数 {other['date'].nunique()}｜平均 IC {float(other['ic'].mean()):+.4f}｜正 IC 占比 {float((other['ic'] > 0).mean()) * 100:.1f}%")
    print()
    print("折 4 的状态分布：")
    print(fold4["trend_bucket"].value_counts().to_string())
    print(fold4["dispersion_bucket"].value_counts().to_string())
    print(fold4["regime_bucket"].value_counts().to_string())
    print()
    print("其它时段的状态分布：")
    print(other["trend_bucket"].value_counts().to_string())
    print(other["dispersion_bucket"].value_counts().to_string())
    print(other["regime_bucket"].value_counts().to_string())

print()
print("已写入", out_dir)
