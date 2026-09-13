# -*- coding: utf-8 -*-
"""全市场 B1 × 大盘阶段 × 离场规则：一次加载数据，跑多套离场配置做对比。

对照口径：
  * 机械持有 h 日（同一批信号、同一入场价）
  * A 只结构离场（白黄线死叉/白线破位/滴滴，无固定止损止盈）
  * B 结构 + 硬止损止盈（盘中 -7% / +15%）
  * C 只硬止损止盈（不看清仓结构）
  * D SOP 紧档（入场K线最低价 -3% + 盘中 -7% + +15% + 结构）
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.exits import ExitConfig  # noqa: E402
from aqlab.rules_zgnb import ActiveMarketValueGate  # noqa: E402
from aqlab.study_universe import (  # noqa: E402
    UniverseStudyConfig,
    load_universe_daily,
    study_universe,
    summarize_trades,
)

START, END = "2025-01-01", "2026-09-11"
frames = load_universe_daily(ROOT / "data" / "universe" / "daily")
print(f"标的 {len(frames)} 只", flush=True)

gate = ActiveMarketValueGate()
regime_raw = gate.gate_series(frames)
regime = regime_raw["gate"].shift(1).fillna(0).astype(int)
print(f"0AMV 波段：开 {int((regime == 1).sum())} 天 / 关 {int((regime == 0).sum())} 天"
      f"｜区间 {regime.index.min().date()} ~ {regime.index.max().date()}", flush=True)

configs = {
    "A 只结构离场": ExitConfig(mode="entry_low", stop_pct=0.05, take_profit_pct=None,
                            intraday_stop_pct=None, min_holding_days=3, didi_mode="full"),
    "B 结构+硬止损止盈": ExitConfig(mode="entry_low", stop_pct=0.05, take_profit_pct=0.15,
                              intraday_stop_pct=0.07, min_holding_days=3, didi_mode="full"),
    "C 只硬止损止盈": ExitConfig(mode="fixed", stop_pct=0.0, take_profit_pct=0.15,
                            intraday_stop_pct=0.07, min_holding_days=3,
                            use_death_cross=False, use_white_break=False, didi_mode="off"),
    "D SOP紧档(entry_low-3%)": ExitConfig(mode="entry_low", stop_pct=0.03, take_profit_pct=0.15,
                                       intraday_stop_pct=0.07, min_holding_days=3, didi_mode="full"),
}

out_dir = ROOT / "output" / "universe_study"
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
tables = {}
for label, exit_config in configs.items():
    config = UniverseStudyConfig(rule="b1_graded", start=START, end=END, exit=exit_config, use_regime_gate=False)
    table, meta = study_universe(frames, config, gate=gate)
    tables[label] = table
    table.to_csv(out_dir / f"trades_{label.split()[0]}.csv", index=False, encoding="utf-8-sig")
    summary = summarize_trades(table).iloc[0]
    rows.append({
        "离场规则": label,
        "笔数": summary["笔数"],
        "平均收益%": summary["平均收益%"],
        "中位收益%": summary["中位收益%"],
        "胜率%": summary["胜率%"],
        "平均超额%": summary["平均超额%"],
        "超额t": summary["超额t"],
        "平均持有": summary["平均持有"],
        "平均最大浮亏%": summary["平均最大浮亏%"],
    })
    print(f"{label}: {meta['trades']} 笔", flush=True)

# 机械持有基线（取最后一组表的 hold_* 列，入场价与信号集相同）
reference = tables["D SOP紧档(entry_low-3%)"]
for horizon in (1, 3, 5, 10, 20):
    values = reference[f"hold_{horizon}"].dropna()
    rows.append({
        "离场规则": f"机械持有 {horizon} 日",
        "笔数": len(values),
        "平均收益%": round(float(values.mean()) * 100, 2),
        "中位收益%": round(float(values.median()) * 100, 2),
        "胜率%": round(float((values > 0).mean()) * 100, 1),
        "平均超额%": "", "超额t": "", "平均持有": horizon, "平均最大浮亏%": "",
    })

comparison = pd.DataFrame(rows)
print()
print(comparison.to_string(index=False))
comparison.to_csv(out_dir / "exit_comparison.csv", index=False, encoding="utf-8-sig")

# 大盘阶段拆分（用 D 组）
d_table = reference.copy()
d_table["波段"] = d_table["signal_date"].apply(lambda date: "开波段" if int(regime.get(pd.Timestamp(date), 0)) == 1 else "关波段")
print()
print("== D 组按大盘阶段 ==")
print(summarize_trades(d_table, by="波段").to_string(index=False))
summarize_trades(d_table, by="波段").to_csv(out_dir / "regime_split.csv", index=False, encoding="utf-8-sig")

print()
print("== D 组按年份 ==")
print(summarize_trades(d_table, by="year").to_string(index=False))

print()
print("== D 组按离场原因 ==")
print(summarize_trades(d_table, by="reason").to_string(index=False))
