"""本地 stockdb 数据源（free-stockdb / LGDB，默认 127.0.0.1:7899）。

本机实测：``分钟k`` 是 1 分钟频率、2025-01-02 起、每日 241 根；``日k`` 从 2000 年起。
官方 SDK 放在 ``pybao/``，负责内存复权（qfq/hfq）与 5/15/30/60m 合成——本模块优先用它，
并在 ``pybao`` 不可用时退回**原始 HTTP**（无复权，仅够做量比这类与价格无关的计算）。

约定：读取用前缀/批量查询，**不要逐股循环上万次**；导出的小样本写到项目 ``data/raw/``。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from aqlab.data import normalize_ohlcv

__all__ = [
    "DEFAULT_HTTP",
    "DEFAULT_PYBAO",
    "StockDbDataSource",
    "export_sample",
    "fetch_daily",
    "fetch_minute",
    "fetch_minute_http",
    "stockdb_available",
]

DEFAULT_PYBAO = r"D:\software\数据\stockdb\pybao"
DEFAULT_HTTP = "http://127.0.0.1:7899"


@lru_cache(maxsize=1)
def _load_rd(pybao_path: str | None = None):
    """加载 stockdb 的 Python SDK（带缓存）。"""
    path = pybao_path or os.environ.get("AQLAB_STOCKDB_PYBAO") or DEFAULT_PYBAO
    if path not in sys.path:
        sys.path.insert(0, path)
    from stock_sdk import rd

    return rd


def stockdb_available(pybao_path: str | None = None) -> bool:
    try:
        _load_rd(pybao_path)
        return True
    except Exception:
        return False


def _daily_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(list(records))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    out = normalize_ohlcv(df)
    # stockdb 的 turnover 是百分数（1.5 = 1.5%），本项目内部统一用小数（0.015）
    if "turnover" in out.columns:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce") / 100.0
    for extra in ("pct_chg", "amount"):
        if extra in df.columns and extra not in out.columns:
            out[extra] = pd.to_numeric(df[extra], errors="coerce")
    return out


def fetch_daily(symbol: str, start: str, end: str, fq: str = "qfq", rd: Any | None = None) -> pd.DataFrame:
    """日线（默认前复权）。``start``/``end`` 用 ``YYYYMMDD``。"""
    rd = rd or _load_rd()
    records = rd.get_data(symbol, start=start, end=end, frequency="1d", fq=fq)
    return _daily_frame(records)


def fetch_minute(symbol: str, start: str, end: str, fq: str = "qfq", rd: Any | None = None) -> pd.DataFrame:
    """1 分钟线（默认前复权），索引名为 ``minute``。"""
    rd = rd or _load_rd()
    records = rd.get_data(symbol, start=start, end=end, frequency="1m", fq=fq)
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(list(records))
    df["minute"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["minute"]).set_index("minute").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    return out


def fetch_minute_http(symbol: str, day: str, base_url: str | None = None, timeout: int = 20) -> pd.DataFrame:
    """无 SDK 时的退回方案：HTTP 原始分钟数据（**未复权**，量比计算不受影响）。"""
    base = (base_url or os.environ.get("AQLAB_STOCKDB_HTTP") or DEFAULT_HTTP).rstrip("/")
    query = urllib.parse.urlencode({"cmd": "vals", "t": "分钟k", "k1": f"key:{symbol}", "k2": f"qz:{day}"})
    with urllib.request.urlopen(f"{base}/?{query}", timeout=timeout) as response:
        records = json.loads(response.read().decode("utf-8"))
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(records)
    df["minute"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["minute"]).set_index("minute").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def export_sample(
    symbols: Iterable[str],
    daily_start: str,
    daily_end: str,
    outdir: str | os.PathLike,
    minute_start: str | None = None,
    minute_end: str | None = None,
    fq: str = "qfq",
    rd: Any | None = None,
) -> dict[str, list[Path]]:
    """导出小样本到 ``outdir``：``{symbol}.csv``（日线）与 ``{symbol}_min.csv``（分钟）。

    分钟文件列固定为 ``minute,close,volume``（供 ``aqlab decide`` / ``confirm-eval`` 使用）。
    """
    rd = rd or _load_rd()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for symbol in symbols:
        daily = fetch_daily(symbol, daily_start, daily_end, fq=fq, rd=rd)
        daily_path = out / f"{symbol}.csv"
        daily.to_csv(daily_path, encoding="utf-8-sig")

        paths = [daily_path]
        if minute_start and minute_end:
            minute = fetch_minute(symbol, minute_start, minute_end, fq=fq, rd=rd)
            if not minute.empty:
                minute_path = out / f"{symbol}_min.csv"
                minute.rename_axis("minute").reset_index()[["minute", "close", "volume"]].to_csv(
                    minute_path, index=False, encoding="utf-8-sig"
                )
                paths.append(minute_path)
        written[symbol] = paths
    return written


class StockDbDataSource:
    """给管线/工具层用的 stockdb 数据源（只读日线）。"""

    def __init__(self, symbols: Sequence[str], start: str, end: str, fq: str = "qfq", rd: Any | None = None) -> None:
        self._symbols = [str(s) for s in symbols]
        if not self._symbols:
            raise ValueError("symbols must not be empty")
        self.start, self.end, self.fq = start, end, fq
        self._rd = rd or _load_rd()
        self._cache: dict[str, pd.DataFrame] = {}

    def symbols(self) -> list[str]:
        return list(self._symbols)

    def bars(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._symbols:
            raise KeyError(f"unknown symbol '{symbol}'")
        if symbol not in self._cache:
            self._cache[symbol] = fetch_daily(symbol, self.start, self.end, fq=self.fq, rd=self._rd)
        return self._cache[symbol]

    def describe(self) -> dict:
        return {
            "source": "stockdb",
            "symbols": self.symbols(),
            "start": self.start,
            "end": self.end,
            "fq": self.fq,
        }
