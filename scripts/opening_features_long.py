# -*- coding: utf-8 -*-
"""长样本开盘特征分析：把 2025-01 ~ 2026-09 全期的 B1 交易补上开盘窗口特征。

与短窗口版的区别：样本期从 3.5 个月扩到 21 个月（分层抽样标的），
用于检验「量能递增 + 0AMV 开波段」是不是只在单一窗口成立。

输出：output/universe_study/long_opening_filter.csv / long_opening_quantiles.csv /
      long_regime_slope.csv / trades_with_opening_long.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.intraday import IntradayConfig, opening_features, standard_volume_ratio  # noqa: E402
from aqlab.rules_zgnb import ActiveMarketValueGate  # noqa: E402
from aqlab.study_universe import load_universe_daily  # noqa: E402

DAILY = ROOT / "data" / "universe" / "daily"
MINUTES = ROOT / "data" / "universe" / "minutes"
WINDOW = IntradayConfig(window_minutes=8, baseline_days=5, session_minutes=240)
OUT = ROOT / "output" / "universe_study"

trades = pd.read_csv(OUT / "trades_D.csv", dtype={"symbol": str}, parse_dates=["entry_date", "signal_date", "exit_date"])
trades["symbol"] = trades["symbol"].str.zfill(6)
print(f"全期交易 {len(trades)} 笔｜标的 {trades['symbol'].nunique()} 只", flush=True)

rows = []
skipped = 0
for symbol, group in trades.groupby("symbol"):
    minute_path = MINUTES / f"{symbol}_min.csv"
    daily_path = DAILY / f"{symbol}.csv"
    if not minute_path.exists() or not daily_path.exists():
        skipped += len(group)
        continue
    try:
        minute = pd.read_csv(minute_path, parse_dates=["minute"]).set_index("minute")
        daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date")
    except Exception:
        skipped += len(group)
        continue
    if minute.empty or daily.empty:
        skipped += len(group)
        continue
    features = opening_features(minute, WINDOW)
    features["volume_ratio"] = standard_volume_ratio(minute, daily["volume"], WINDOW).reindex(features.index)
    close_by_date = daily["close"]
    for record in group.to_dict("records"):
        entry_date = pd.Timestamp(record["entry_date"])
        if entry_date not in features.index:
            skipped += 1
            continue
        feature = features.loc[entry_date]
        previous = close_by_date.index[close_by_date.index < entry_date]
        prev_close = float(close_by_date.loc[previous[-1]]) if len(previous) else np.nan
        window_close = float(feature["window_close"])
        row = dict(record)
        row["volume_ratio"] = float(feature["volume_ratio"]) if np.isfinite(feature["volume_ratio"]) else np.nan
        row["up_from_open"] = float(feature["up_from_open"])
        row["up_from_prev"] = window_close / prev_close - 1.0 if np.isfinite(prev_close) and prev_close > 0 else np.nan
        row["volume_slope"] = float(feature["volume_slope"])
        row["window_position"] = float(feature["window_position"])
        row["return_937"] = float(record["exit_price"]) / window_close - 1.0 if window_close > 0 else np.nan
        row["excess_937"] = row["return_937"] - record["bench_pct"] if np.isfinite(record["bench_pct"]) else np.nan
        rows.append(row)

frame = pd.DataFrame(rows)
print(f"补上开盘特征 {len(frame)} 笔｜缺分钟数据 {skipped} 笔", flush=True)
if frame.empty:
    raise SystemExit("没有可用样本")
frame["month"] = frame["entry_date"].dt.to_period("M").astype(str)
frame.to_csv(OUT / "trades_with_opening_long.csv", index=False, encoding="utf-8-sig")

REGIME_CACHE = OUT / "regime_series.csv"
if REGIME_CACHE.exists():
    regime = pd.read_csv(REGIME_CACHE, parse_dates=["date"]).set_index("date")["gate"]
    print(f"0AMV 波段序列：读缓存 {len(regime)} 天", flush=True)
else:
    frames = load_universe_daily(DAILY)
    gate = ActiveMarketValueGate()
    regime = gate.gate_series(frames)["gate"].shift(1).fillna(0).astype(int)
    regime.rename("gate").rename_axis("date").reset_index().to_csv(REGIME_CACHE, index=False, encoding="utf-8-sig")
    print(f"0AMV 波段序列：现算并缓存 {len(regime)} 天（开 {int((regime == 1).sum())} 天）", flush=True)
frame["波段"] = frame["entry_date"].apply(lambda date: "开波段" if int(regime.get(pd.Timestamp(date), 0)) == 1 else "关波段")


def describe(group: pd.DataFrame, label: str) -> dict:
    returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
    excess = pd.to_numeric(group["excess_pct"], errors="coerce").dropna()
    t_stat = np.nan
    if len(excess) > 2 and excess.std(ddof=1) > 0:
        t_stat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
    return {
        "组": label,
        "笔数": len(group),
        "平均收益%": round(float(returns.mean()) * 100, 2) if len(returns) else np.nan,
        "中位%": round(float(returns.median()) * 100, 2) if len(returns) else np.nan,
        "胜率%": round(float((returns > 0).mean()) * 100, 1) if len(returns) else np.nan,
        "平均超额%": round(float(excess.mean()) * 100, 2) if len(excess) else np.nan,
        "超额t": round(float(t_stat), 2) if np.isfinite(t_stat) else np.nan,
        "平均持有": round(float(group["bars"].mean()), 1),
    }


print(f"\n覆盖月份：{frame['month'].nunique()} 个月｜{frame['month'].min()} ~ {frame['month'].max()}", flush=True)

rows_out = [describe(frame, "全部（有分钟特征的样本）")]
for label, mask in (
    ("量能递增（slope>0）", frame["volume_slope"] > 0),
    ("量能衰减（slope<0）", frame["volume_slope"] < 0),
    ("量比≥4", frame["volume_ratio"] >= 4),
    ("量比<2", frame["volume_ratio"] < 2),
    ("相对开盘向上", frame["up_from_open"] > 0),
    ("相对昨收向上", frame["up_from_prev"] > 0),
    ("量比≥4 且 向上", (frame["volume_ratio"] >= 4) & (frame["up_from_open"] > 0)),
    ("量能递增 且 量比<4", (frame["volume_slope"] > 0) & (frame["volume_ratio"] < 4)),
    ("量能递增 且 开波段", (frame["volume_slope"] > 0) & (frame["波段"] == "开波段")),
    ("量能递增 且 关波段", (frame["volume_slope"] > 0) & (frame["波段"] == "关波段")),
):
    rows_out.append(describe(frame[mask], label))
table = pd.DataFrame(rows_out)
print()
print(table.to_string(index=False))
table.to_csv(OUT / "long_opening_filter.csv", index=False, encoding="utf-8-sig")

print("\n=== 分位对照 ===")
quantile_rows = []
for column in ("volume_slope", "volume_ratio", "up_from_open", "up_from_prev"):
    try:
        frame["_bucket"] = pd.qcut(frame[column], 4, labels=["Q1低", "Q2", "Q3", "Q4高"], duplicates="drop")
    except ValueError:
        continue
    for bucket, group in frame.groupby("_bucket", observed=True):
        quantile_rows.append(describe(group, f"{column} {bucket}"))
quantiles = pd.DataFrame(quantile_rows)
print(quantiles.to_string(index=False))
quantiles.to_csv(OUT / "long_opening_quantiles.csv", index=False, encoding="utf-8-sig")

print("\n=== 0AMV 波段 × 量能斜率 ===")
regime_rows = []
for regime_name, regime_group in frame.groupby("波段"):
    regime_rows.append(describe(regime_group, f"{regime_name} 全部"))
    regime_rows.append(describe(regime_group[regime_group["volume_slope"] > 0], f"{regime_name} + 量能递增"))
    regime_rows.append(describe(regime_group[regime_group["volume_slope"] < 0], f"{regime_name} + 量能衰减"))
regime_table = pd.DataFrame(regime_rows)
print(regime_table.to_string(index=False))
regime_table.to_csv(OUT / "long_regime_slope.csv", index=False, encoding="utf-8-sig")

print("\n=== 按年份 ===")
print(pd.DataFrame([describe(group, year) for year, group in frame.groupby(frame["entry_date"].dt.year)]).to_string(index=False))

print("\n=== 量能递增子集：离场规则 vs 机械持有（%）===")
good = frame[frame["volume_slope"] > 0]
rows = [describe(good, "量能递增（离场规则）")]
for horizon in (1, 3, 5, 10, 20):
    values = good[f"hold_{horizon}"].dropna()
    rows.append({"组": f"机械持有{horizon}日", "笔数": len(values),
                 "平均收益%": round(values.mean() * 100, 2), "中位%": round(values.median() * 100, 2),
                 "胜率%": round((values > 0).mean() * 100, 1), "平均超额%": np.nan, "超额t": np.nan, "平均持有": horizon})
print(pd.DataFrame(rows).to_string(index=False))
