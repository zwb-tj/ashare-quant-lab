# -*- coding: utf-8 -*-
"""全市场日线抓取（后台长任务，可中断续跑）。

- 代码表来自 data/universe/symbols.txt
- 时间范围 2024-06-01 ~ 2026-09-11（覆盖 2025-01 起的回测，并给 MA114 预热）
- 每只票一个 CSV，缓存在 data/universe/daily/<symbol>.csv，已存在且非空则跳过
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\software\数据\stockdb\pybao")

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "universe"
DAILY = UNIVERSE / "daily"
DAILY.mkdir(parents=True, exist_ok=True)

from aqlab.stockdb import fetch_daily  # noqa: E402

START, END = "20240601", "20260911"


def symbol_list() -> list[str]:
    path = UNIVERSE / "symbols.txt"
    if path.exists():
        cached = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(cached) > 3000:
            return cached
    from stock_sdk import rd

    keys = list(rd.keys("复权"))
    symbols = sorted({str(key).split(":")[1] for key in keys if str(key).count(":") >= 2})
    keep = [s for s in symbols if s[:2] in {"60", "68", "00", "30", "83", "87", "43", "92"}]
    path.write_text("\n".join(keep), encoding="utf-8")
    return keep


symbols = symbol_list()
print(f"全市场代码：{len(symbols)}", flush=True)

ok = skipped = failed = 0
start_time = time.time()
for index, symbol in enumerate(symbols, 1):
    path = DAILY / f"{symbol}.csv"
    if path.exists() and path.stat().st_size > 200:
        skipped += 1
        continue
    try:
        frame = fetch_daily(symbol, START, END)
    except Exception:  # noqa: BLE001
        failed += 1
        continue
    if frame is None or frame.empty:
        failed += 1
        continue
    frame.to_csv(path, encoding="utf-8-sig")
    ok += 1
    if index % 500 == 0:
        elapsed = time.time() - start_time
        print(f"  {index}/{len(symbols)}｜新增 {ok}｜跳过 {skipped}｜失败 {failed}｜用时 {elapsed/60:.1f} 分钟", flush=True)
print(f"完成：新增 {ok}｜跳过 {skipped}｜失败 {failed}｜总用时 {(time.time()-start_time)/60:.1f} 分钟", flush=True)
