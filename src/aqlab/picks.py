"""基于「选股日志」的回测（v0.10）：不重新选股，直接评估你实际推出去的名单。

入口数据：``picks_archive/picks_YYYY-MM-DD.json``（每天一个文件，含 b1/b2/b3/v3/n20/n30/fa/fb 分桶）。

口径（写清楚，避免自欺）：

* **入场**：日志在收盘后发布 → 默认取**次日开盘价**入场（``entry="next_open"``，可切 ``next_close``），T+1 合法；
* **持有 h 日**：``h=1`` 表示买入当天收盘，``h=3/5/10`` 依次类推；
* **去重**：同一只票在 ``dedupe_window`` 个交易日内重复出现只算第一次（否则连板/续持会被重复计入统计）；
* **缺失**：数据不够（次日起未上市/超出本地数据范围）记为 NaN 并从该项统计里剔除，同时单独计数，不当成 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PICK_BUCKETS",
    "PickBacktestConfig",
    "load_picks_archive",
    "dedupe_picks",
    "evaluate_picks",
    "benchmark_returns",
    "attach_benchmark",
    "summarize_picks",
    "summary_markdown",
]

PICK_BUCKETS = ("b1", "b2", "b3", "v3", "n20", "n30", "fa", "fb")


@dataclass
class PickBacktestConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    entry: str = "next_open"          # next_open | next_close
    dedupe_window: int = 5            # 同一标的多少天内只算第一次
    buckets: tuple[str, ...] = ("b1", "b2", "n20", "n30", "v3")
    benchmark_sample: int = 0         # >0 时用等权篮子做同期基准（需要自带篮子日线）

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.entry not in ("next_open", "next_close"):
            raise ValueError("entry must be 'next_open' or 'next_close'")
        if self.dedupe_window < 1:
            raise ValueError("dedupe_window must be >= 1")
        if self.benchmark_sample < 0:
            raise ValueError("benchmark_sample must be >= 0")


def benchmark_candidates(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    exclude: Iterable[str] = (),
    sample: int = 0,
) -> list[str]:
    """从给定日线集合里等距抽取基准篮子标的（剔除自己选中的票，避免自我对照）。"""
    blocked = {_plain(symbol) for symbol in exclude}
    pool = sorted(symbol for symbol in daily_by_symbol if _plain(symbol) not in blocked)
    if sample <= 0 or sample >= len(pool):
        return pool
    picks = np.linspace(0, len(pool) - 1, sample).round().astype(int)
    return [pool[i] for i in sorted(set(picks.tolist()))]


def benchmark_returns(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    entry_dates: Iterable[Any],
    config: PickBacktestConfig | None = None,
) -> pd.DataFrame:
    """等权篮子在同一入场日的同期收益：``bench_{h}`` 与样本数 ``benchn_{h}``。

    篮子只取**所有成员都有该持有期数据**的样本平均（缺数据的成员直接不计），
    这样基准与选股的口径（次日开盘入场、第 h 日收盘出场）一致。
    """
    config = config or PickBacktestConfig()
    frames = {symbol: frame for symbol, frame in daily_by_symbol.items() if frame is not None and not frame.empty}
    prepared = {}
    for symbol, frame in frames.items():
        copy = frame.copy()
        copy.index = pd.to_datetime(copy.index)
        prepared[symbol] = copy
    price_column = "open" if config.entry == "next_open" else "close"
    rows: list[dict] = []
    for entry_date in pd.Index(pd.to_datetime(list(entry_dates))).dropna().unique():
        row: dict[str, Any] = {"entry_date": str(pd.Timestamp(entry_date).date())}
        for horizon in config.horizons:
            returns: list[float] = []
            for frame in prepared.values():
                if entry_date not in frame.index:
                    continue
                position = frame.index.get_loc(entry_date)
                exit_position = position + horizon - 1
                if exit_position >= len(frame):
                    continue
                entry_price = float(frame[price_column].iloc[position])
                exit_price = float(frame["close"].iloc[exit_position])
                if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price):
                    returns.append(exit_price / entry_price - 1.0)
            row[f"bench_{horizon}"] = float(np.mean(returns)) if returns else np.nan
            row[f"benchn_{horizon}"] = len(returns)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_benchmark(evaluated: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """按入场日把基准接到明细上，并算出超额（``excess_{h}`` = 个股 - 篮子）。"""
    if evaluated.empty or benchmark.empty:
        return evaluated.copy()
    merged = evaluated.merge(benchmark, on="entry_date", how="left")
    for column in [c for c in merged.columns if c.startswith("fwd_")]:
        horizon = column.split("_", 1)[1]
        bench_column = f"bench_{horizon}"
        if bench_column in merged.columns:
            merged[f"excess_{horizon}"] = pd.to_numeric(merged[column], errors="coerce") - pd.to_numeric(
                merged[bench_column], errors="coerce"
            )
        else:                                  # 基准缺这一期 -> 超额记 NaN，避免被当成 0
            merged[f"excess_{horizon}"] = np.nan
    return merged


def _plain(ts_code: str) -> str:
    return str(ts_code).split(".")[0].zfill(6)


def load_picks_archive(archive_dir: str | Path, buckets: Iterable[str] | None = None) -> pd.DataFrame:
    """读取整个日志目录，返回长表：date / ts_code / symbol / name / bucket / score / industry。"""
    import json

    keep = tuple(buckets) if buckets else PICK_BUCKETS
    rows: list[dict] = []
    for path in sorted(Path(archive_dir).glob("picks_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 坏文件跳过但不静默
            rows.append({"date": path.stem.replace("picks_", ""), "bucket": "_broken", "ts_code": "", "name": path.name, "score": np.nan})
            continue
        date = str(payload.get("date") or path.stem.replace("picks_", ""))
        for bucket in keep:
            for item in payload.get(bucket) or []:
                if not isinstance(item, dict) or not item.get("ts_code"):
                    continue
                rows.append(
                    {
                        "date": date,
                        "bucket": bucket,
                        "ts_code": str(item.get("ts_code")),
                        "symbol": _plain(item.get("ts_code", "")),
                        "name": item.get("name", ""),
                        "industry": item.get("industry", ""),
                        "score": item.get("score", np.nan),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["date", "bucket", "ts_code", "symbol", "name", "industry", "score"])
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values(["date", "bucket", "symbol"]).reset_index(drop=True)


def dedupe_picks(picks: pd.DataFrame, window_days: int = 5) -> pd.DataFrame:
    """同一标的在 ``window_days`` 个交易日内重复出现只保留第一次。"""
    if picks.empty:
        return picks
    keep_rows: list[int] = []
    last_seen: dict[str, pd.Timestamp] = {}
    for idx, row in picks.sort_values("date").iterrows():
        symbol, date = row["symbol"], row["date"]
        previous = last_seen.get(symbol)
        span = pd.bdate_range(previous, date) if previous is not None else []
        if previous is None or len(span) > window_days:
            keep_rows.append(idx)
        last_seen[symbol] = date
    return picks.loc[keep_rows].sort_values(["date", "bucket", "symbol"]).reset_index(drop=True)


def evaluate_picks(
    picks: pd.DataFrame,
    daily_by_symbol: Mapping[str, pd.DataFrame],
    config: PickBacktestConfig | None = None,
) -> pd.DataFrame:
    """给每条选股记录附上入场价与 1/3/5/10 日收益（``entry`` 默认次日开盘）。"""
    config = config or PickBacktestConfig()
    records: list[dict] = []
    for row in picks.to_dict("records"):
        df = daily_by_symbol.get(row["symbol"])
        record = dict(row)
        record["entry_date"] = None
        record["entry_price"] = np.nan
        for horizon in config.horizons:
            record[f"fwd_{horizon}"] = np.nan
        if df is None or df.empty:
            records.append(record)
            continue
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index)
        later = frame.index[frame.index > pd.Timestamp(row["date"])]
        if len(later) == 0:
            records.append(record)
            continue
        entry_date = later[0]
        entry_pos = frame.index.get_loc(entry_date)
        price_column = "open" if config.entry == "next_open" else "close"
        entry_price = float(frame[price_column].iloc[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            records.append(record)
            continue
        record["entry_date"] = str(pd.Timestamp(entry_date).date())
        record["entry_price"] = round(entry_price, 3)
        for horizon in config.horizons:
            exit_pos = entry_pos + horizon - 1
            if exit_pos < len(frame):
                exit_price = float(frame["close"].iloc[exit_pos])
                if np.isfinite(exit_price):
                    record[f"fwd_{horizon}"] = round(exit_price / entry_price - 1.0, 4)
        records.append(record)
    return pd.DataFrame(records)


def summarize_picks(evaluated: pd.DataFrame, config: PickBacktestConfig | None = None) -> pd.DataFrame:
    """按分桶汇总：样本数、各持有期的均值/中位数/胜率、缺失数；有基准时附篮子与超额。"""
    config = config or PickBacktestConfig()
    if evaluated.empty:
        return pd.DataFrame()
    has_benchmark = any(f"excess_{h}" in evaluated.columns for h in config.horizons)

    def fill(row: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
        for horizon in config.horizons:
            values = pd.to_numeric(group[f"fwd_{horizon}"], errors="coerce")
            valid = values.dropna()
            row[f"n_{horizon}"] = int(len(valid))
            row[f"mean_{horizon}"] = float(valid.mean()) if len(valid) else np.nan
            row[f"median_{horizon}"] = float(valid.median()) if len(valid) else np.nan
            row[f"win_{horizon}"] = float((valid > 0).mean()) if len(valid) else np.nan
            if has_benchmark:
                bench_column, excess_column = f"bench_{horizon}", f"excess_{horizon}"
                bench = (
                    pd.to_numeric(group[bench_column], errors="coerce").reindex(valid.index).dropna()
                    if bench_column in group.columns
                    else pd.Series(dtype=float)
                )
                excess = (
                    pd.to_numeric(group[excess_column], errors="coerce").dropna()
                    if excess_column in group.columns
                    else pd.Series(dtype=float)
                )
                row[bench_column] = float(bench.mean()) if len(bench) else np.nan
                row[excess_column] = float(excess.mean()) if len(excess) else np.nan
                row[f"excesswin_{horizon}"] = float((excess > 0).mean()) if len(excess) else np.nan
        return row

    rows: list[dict] = []
    for bucket, group in evaluated.groupby("bucket"):
        rows.append(fill({"bucket": bucket, "picks": int(len(group))}, group))

    overall: dict[str, Any] = {"bucket": "全部", "picks": int(len(evaluated))}
    rows.append(fill(overall, evaluated))
    table = pd.DataFrame(rows)
    table["_order"] = table["bucket"].map({b: i for i, b in enumerate(PICK_BUCKETS)}).fillna(9)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def summary_markdown(summary: pd.DataFrame, config: PickBacktestConfig | None = None) -> str:
    from aqlab.tables import markdown_table

    config = config or PickBacktestConfig()
    if summary.empty:
        return "没有可汇总的选股记录。"
    has_benchmark = any(f"excess_{h}" in summary.columns for h in config.horizons)
    view = summary.copy()
    for horizon in config.horizons:
        view[f"mean_{horizon}"] = (view[f"mean_{horizon}"].astype(float) * 100).round(2)
        view[f"median_{horizon}"] = (view[f"median_{horizon}"].astype(float) * 100).round(2)
        view[f"win_{horizon}"] = (view[f"win_{horizon}"].astype(float) * 100).round(1)
        if has_benchmark:
            view[f"bench_{horizon}"] = (view[f"bench_{horizon}"].astype(float) * 100).round(2) if f"bench_{horizon}" in view.columns else np.nan
            view[f"excess_{horizon}"] = (view[f"excess_{horizon}"].astype(float) * 100).round(2) if f"excess_{horizon}" in view.columns else np.nan
            view[f"excesswin_{horizon}"] = (view[f"excesswin_{horizon}"].astype(float) * 100).round(1) if f"excesswin_{horizon}" in view.columns else np.nan
    view = view.rename(columns={"bucket": "分桶", "picks": "选股数"})
    header = f"> 入场：{config.entry}（次日开盘/收盘）｜去重窗口：{config.dedupe_window} 个交易日｜持有期单位：交易日\n"
    if has_benchmark:
        header += "> 基准：同期等权篮子（票池外样本）｜超额 = 个股收益 - 篮子收益｜单位：%\n"
    return header + markdown_table(view)
