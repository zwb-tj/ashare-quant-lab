# -*- coding: utf-8 -*-
"""长样本分钟数据抓取：按月份分层抽样标的，覆盖 2025-01 ~ 2026-09 全期。

为什么抽样：全期 58,682 个信号涉及 5,228 只标的，全抓要 4~6 小时、十几 GB。
按月份分层各抽一部分标的，可以在保证**每个月都有样本**的前提下把量降到 1/5 左右。
"""
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqlab.stockdb import fetch_minute  # noqa: E402

TARGET_SYMBOLS = 1000
START, END = "20241215", "20260917"

OUT = ROOT / "data" / "universe" / "minutes"
OUT.mkdir(parents=True, exist_ok=True)

trades = pd.read_csv(ROOT / "output" / "universe_study" / "trades_D.csv", dtype={"symbol": str}, parse_dates=["signal_date"])
trades["symbol"] = trades["symbol"].str.zfill(6)
trades["month"] = trades["signal_date"].dt.to_period("M").astype(str)

random.seed(20260912)
months = sorted(trades["month"].unique())
total_signals = len(trades)
chosen: set[str] = set()
for month in months:
    month_symbols = trades.loc[trades["month"] == month, "symbol"].unique().tolist()
    share = len(trades[trades["month"] == month]) / total_signals
    quota = max(30, int(TARGET_SYMBOLS * share))
    random.shuffle(month_symbols)
    added = 0
    for symbol in month_symbols:
        if added >= quota:
            break
        if symbol not in chosen:
            chosen.add(symbol)
            added += 1

symbols = sorted(chosen)
print(f"分层抽样：{len(symbols)} 只标的（目标 {TARGET_SYMBOLS}，覆盖 {len(months)} 个月）", flush=True)

ok = skipped = failed = 0
start_time = time.time()
for index, symbol in enumerate(symbols, 1):
    path = OUT / f"{symbol}_min.csv"
    if path.exists() and path.stat().st_size > 100000:
        skipped += 1
        continue
    try:
        minute = fetch_minute(symbol, START, END)
    except Exception:  # noqa: BLE001
        failed += 1
        continue
    if minute is None or minute.empty:
        failed += 1
        continue
    frame = minute.rename_axis("minute").reset_index()
    keep = [column for column in ("minute", "close", "high", "low", "volume") if column in frame.columns]
    frame[keep].to_csv(path, index=False, encoding="utf-8-sig")
    ok += 1
    if index % 50 == 0:
        print(f"  {index}/{len(symbols)}｜新增 {ok}｜跳过 {skipped}｜失败 {failed}｜用时 {(time.time()-start_time)/60:.1f} 分钟", flush=True)
print(f"完成：新增 {ok}｜跳过 {skipped}｜失败 {failed}｜总用时 {(time.time()-start_time)/60:.1f} 分钟", flush=True)
