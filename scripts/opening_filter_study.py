# -*- coding: utf-8 -*-
"""A 部分：把"向上冲"变成可计算条件，并与 0AMV 阶段、离场规则合起来看。

样本：全市场 B1 信号（2026-06-01 ~ 2026-09-10 入场），离场规则用 D（SOP 紧档）。
对每笔交易补上决策日（= 入场日 T+1）开盘窗口的特征：
  * 量比（软件口径，8 根 = 09:30~09:37，基准 5 日全天均量）
  * 相对昨收 / 相对开盘的窗口涨幅（"向上冲"）
  * 窗口内量能斜率（每分钟量递增 = 资金持续流入）
  * 窗口收盘在窗口高低区间的位置
两种入场价对比：T+1 开盘价 vs 9:37 窗口收盘价（同一套离场价位）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.intraday import IntradayConfig, opening_features, standard_volume_ratio  # noqa: E402

DAILY = ROOT / "data" / "universe" / "daily"
MINUTES = ROOT / "data" / "universe" / "minutes"
WINDOW = IntradayConfig(window_minutes=8, baseline_days=5, session_minutes=240)

trades = pd.read_csv(ROOT / "output" / "universe_study" / "trades_D.csv", dtype={"symbol": str}, parse_dates=["entry_date", "exit_date"])
trades["symbol"] = trades["symbol"].str.zfill(6)
window = trades[(trades["entry_date"] >= "2026-06-01") & (trades["entry_date"] <= "2026-09-10")].copy()
print(f"窗口内交易 {len(window)} 笔｜标的 {window['symbol'].nunique()} 只", flush=True)

rows = []
missing_minute = 0
for symbol, group in window.groupby("symbol"):
    minute_path = MINUTES / f"{symbol}_min.csv"
    daily_path = DAILY / f"{symbol}.csv"
    if not minute_path.exists() or not daily_path.exists():
        missing_minute += len(group)
        continue
    try:
        minute = pd.read_csv(minute_path, parse_dates=["minute"]).set_index("minute")
        daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date")
    except Exception:
        missing_minute += len(group)
        continue
    if minute.empty or daily.empty:
        missing_minute += len(group)
        continue
    features = opening_features(minute, WINDOW)
    ratio = standard_volume_ratio(minute, daily["volume"], WINDOW)
    features["volume_ratio"] = ratio.reindex(features.index)
    close_by_date = daily["close"]
    for record in group.to_dict("records"):
        entry_date = pd.Timestamp(record["entry_date"])
        if entry_date not in features.index:
            missing_minute += 1
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
        row["entry_937"] = window_close
        row["return_937"] = float(record["exit_price"]) / window_close - 1.0 if window_close > 0 else np.nan
        row["excess_937"] = row["return_937"] - record["bench_pct"] if np.isfinite(record["bench_pct"]) else np.nan
        rows.append(row)

frame = pd.DataFrame(rows)
print(f"补上分钟特征 {len(frame)} 笔｜缺分钟数据 {missing_minute} 笔", flush=True)
frame.to_csv(ROOT / "output" / "universe_study" / "trades_with_opening.csv", index=False, encoding="utf-8-sig")
if frame.empty:
    print("没有可用样本")
    raise SystemExit(0)


def describe(group: pd.DataFrame, label: str) -> dict:
    returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
    excess = pd.to_numeric(group["excess_pct"], errors="coerce").dropna()
    intraday_excess = pd.to_numeric(group["excess_937"], errors="coerce").dropna()
    t_stat = np.nan
    if len(excess) > 2 and excess.std(ddof=1) > 0:
        t_stat = excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
    return {
        "组": label,
        "笔数": len(group),
        "开盘入场收益%": round(float(returns.mean()) * 100, 2) if len(returns) else np.nan,
        "开盘入场超额%": round(float(excess.mean()) * 100, 2) if len(excess) else np.nan,
        "开盘超额t": round(float(t_stat), 2) if np.isfinite(t_stat) else np.nan,
        "9:37入场收益%": round(float(pd.to_numeric(group['return_937'], errors='coerce').mean()) * 100, 2),
        "9:37超额%": round(float(intraday_excess.mean()) * 100, 2) if len(intraday_excess) else np.nan,
        "胜率%": round(float((returns > 0).mean()) * 100, 1) if len(returns) else np.nan,
    }


rows_out = [describe(frame, "全部（窗口内）")]
rows_out.append(describe(frame[frame["volume_ratio"] >= 4], "量比>=4"))
rows_out.append(describe(frame[frame["volume_ratio"] < 4], "量比<4"))
rows_out.append(describe(frame[frame["up_from_prev"] > 0], "相对昨收向上"))
rows_out.append(describe(frame[frame["up_from_open"] > 0], "相对开盘向上"))
rows_out.append(describe(frame[frame["volume_slope"] > 0], "量能递增"))
rows_out.append(describe(frame[(frame["volume_ratio"] >= 4) & (frame["up_from_prev"] > 0)], "量比>=4 且 相对昨收向上"))
rows_out.append(describe(frame[(frame["volume_ratio"] >= 4) & (frame["up_from_open"] > 0)], "量比>=4 且 相对开盘向上"))
rows_out.append(describe(frame[(frame["volume_ratio"] >= 4) & (frame["volume_slope"] > 0)], "量比>=4 且 量能递增"))
rows_out.append(describe(frame[(frame["volume_ratio"] >= 2) & (frame["up_from_open"] > 0) & (frame["volume_slope"] > 0)], "量比>=2 + 向上 + 量增"))
table = pd.DataFrame(rows_out)
print()
print(table.to_string(index=False))
table.to_csv(ROOT / "output" / "universe_study" / "opening_filter.csv", index=False, encoding="utf-8-sig")

print()
print("=== 2×2：量比 × 相对开盘方向 ===")
grid = []
for high in (True, False):
    for up in (True, False):
        subset = frame[(frame["volume_ratio"] >= 4) == high]
        subset = subset[((subset["up_from_open"] > 0) == up)]
        label = f"量比{'≥4' if high else '<4'} + {'向上' if up else '向下'}"
        grid.append(describe(subset, label))
print(pd.DataFrame(grid).to_string(index=False))

print()
print("=== 分位对照（避免只用阈值）===")
quantiles = []
for column in ("volume_ratio", "up_from_open", "up_from_prev", "volume_slope"):
    valid = frame[column].dropna()
    if len(valid) < 50:
        continue
    try:
        frame["_bucket"] = pd.qcut(frame[column], 4, labels=["Q1低", "Q2", "Q3", "Q4高"], duplicates="drop")
    except ValueError:
        continue
    for bucket, group in frame.groupby("_bucket", observed=True):
        row = describe(group, f"{column} {bucket}")
        quantiles.append(row)
print(pd.DataFrame(quantiles).to_string(index=False))
pd.DataFrame(quantiles).to_csv(ROOT / "output" / "universe_study" / "opening_quantiles.csv", index=False, encoding="utf-8-sig")
