"""数据质量审计（v0.7）：在"结论"之前先把"数据"查干净。

检查项（全部可配置）：

* **索引**：是否单调、有无重复交易日；
* **缺口**：相邻交易日间隔超过 ``max_gap_days`` 日历天（停牌/漏数据）；
* **零成交**：零成交量占比超过阈值；
* **异常跳变**：单日涨跌幅超过 ``price_jump_pct``（未复权、拆分、脏数据）；
* **缺口历史**：bar 数少于 ``min_bars``；
* **字段**：OHLC 缺失值、是否带真实换手率（turnover 缺失不是错误，但要如实标注）；
* **多源交叉**：同一标的两个数据源的收盘价/成交量差异率；
* **快照指纹**：规范化后的 sha256，保证"同一份数据 → 同一个结论"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from aqlab.tables import markdown_table

__all__ = [
    "QualityConfig",
    "audit_universe",
    "check_frame",
    "cross_source_diff",
    "format_audit",
    "snapshot_hash",
    "write_audit",
]


@dataclass
class QualityConfig:
    max_gap_days: int = 10
    price_jump_pct: float = 0.11
    min_bars: int = 60
    zero_volume_ratio: float = 0.02

    def __post_init__(self) -> None:
        if self.max_gap_days < 1:
            raise ValueError("max_gap_days must be >= 1")
        if not 0 < self.price_jump_pct < 1:
            raise ValueError("price_jump_pct must be in (0, 1)")
        if self.min_bars < 3:
            raise ValueError("min_bars must be >= 3")
        if not 0 <= self.zero_volume_ratio <= 1:
            raise ValueError("zero_volume_ratio must be in [0, 1]")


def check_frame(symbol: str, df: pd.DataFrame, config: QualityConfig | None = None) -> list[dict]:
    """返回问题清单（每条含 type / detail / severity）。"""
    config = config or QualityConfig()
    issues: list[dict] = []
    if df is None or df.empty:
        return [{"symbol": symbol, "type": "empty", "detail": "数据为空", "severity": "error"}]

    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)
    if not frame.index.is_monotonic_increasing:
        issues.append({"symbol": symbol, "type": "not_sorted", "detail": "索引非单调递增", "severity": "error"})
    duplicated = int(frame.index.duplicated().sum())
    if duplicated:
        issues.append({"symbol": symbol, "type": "duplicate_dates", "detail": f"{duplicated} 个重复交易日", "severity": "error"})
    if len(frame) < config.min_bars:
        issues.append(
            {"symbol": symbol, "type": "insufficient_history", "detail": f"仅 {len(frame)} 根 < {config.min_bars}", "severity": "warning"}
        )

    ordered = frame[~frame.index.duplicated()].sort_index()
    if len(ordered) > 1:
        gaps = ordered.index.to_series().diff().dt.days.dropna()
        worst_gap = float(gaps.max()) if len(gaps) else 0.0
        if worst_gap > config.max_gap_days:
            count = int((gaps > config.max_gap_days).sum())
            issues.append(
                {
                    "symbol": symbol,
                    "type": "calendar_gap",
                    "detail": f"最大间隔 {worst_gap:.0f} 天，共 {count} 处超过 {config.max_gap_days} 天",
                    "severity": "warning",
                }
            )

    if "volume" in ordered.columns:
        zero_ratio = float((ordered["volume"].fillna(0) <= 0).mean())
        if zero_ratio > config.zero_volume_ratio:
            issues.append(
                {"symbol": symbol, "type": "zero_volume", "detail": f"零成交量占比 {zero_ratio:.1%}", "severity": "warning"}
            )

    if "close" in ordered.columns:
        returns = ordered["close"].astype(float).pct_change()
        jumps = returns.abs()
        worst_jump = float(jumps.max()) if len(jumps) else 0.0
        if worst_jump > config.price_jump_pct:
            count = int((jumps > config.price_jump_pct).sum())
            issues.append(
                {
                    "symbol": symbol,
                    "type": "price_jump",
                    "detail": f"最大单日跳变 {worst_jump:.1%}，共 {count} 处超过 {config.price_jump_pct:.0%}（疑似未复权/脏数据）",
                    "severity": "warning",
                }
            )

    missing = int(ordered[["open", "high", "low", "close"]].isna().sum().sum()) if {"open", "high", "low", "close"}.issubset(ordered.columns) else 0
    if missing:
        issues.append({"symbol": symbol, "type": "missing_values", "detail": f"OHLC 缺失 {missing} 个", "severity": "error"})

    if "turnover" not in ordered.columns:
        issues.append({"symbol": symbol, "type": "no_turnover", "detail": "缺少真实换手率列（相关规则会跳过并注明）", "severity": "info"})
    return issues


def audit_universe(universe: Mapping[str, pd.DataFrame], config: QualityConfig | None = None) -> pd.DataFrame:
    """对整个票池做审计，输出每票一行 + 问题类型汇总。"""
    config = config or QualityConfig()
    rows: list[dict] = []
    for symbol, df in universe.items():
        issues = check_frame(symbol, df, config)
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        infos = [i for i in issues if i["severity"] == "info"]
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index)
        rows.append(
            {
                "symbol": symbol,
                "bars": len(frame),
                "first": str(frame.index.min().date()) if len(frame) else "",
                "last": str(frame.index.max().date()) if len(frame) else "",
                "errors": len(errors),
                "warnings": len(warnings),
                "infos": len(infos),
                "issue_types": ",".join(sorted({i["type"] for i in issues if i["severity"] != "info"})) or "-",
                "details": " | ".join(i["detail"] for i in (errors + warnings)[:2]) or "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["errors", "warnings", "symbol"], ascending=[False, False, True]).reset_index(drop=True)


def cross_source_diff(symbol: str, left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 0.005) -> dict:
    """两个数据源在同一标的上的差异报告（收盘价与成交量的相对差异率）。"""
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out[~out.index.duplicated()].sort_index()

    a, b = _prep(left), _prep(right)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"symbol": symbol, "common_days": 0, "close_mismatch_rate": np.nan, "volume_mismatch_rate": np.nan, "examples": []}

    close_a, close_b = a.loc[common, "close"].astype(float), b.loc[common, "close"].astype(float)
    close_rel = (close_a - close_b).abs() / close_b.replace(0.0, np.nan)
    volume_rel = None
    if "volume" in a.columns and "volume" in b.columns:
        vol_a, vol_b = a.loc[common, "volume"].astype(float), b.loc[common, "volume"].astype(float)
        volume_rel = (vol_a - vol_b).abs() / vol_b.replace(0.0, np.nan)

    close_bad = close_rel[close_rel > tolerance].dropna()
    examples = [
        {"date": str(pd.Timestamp(idx).date()), "close_rel_diff": round(float(val), 4)}
        for idx, val in close_bad.sort_values(ascending=False).head(3).items()
    ]
    result = {
        "symbol": symbol,
        "common_days": len(common),
        "close_mismatch_rate": round(float((close_rel > tolerance).mean()), 4),
        "worst_close_diff": round(float(close_rel.max()), 4) if len(close_rel.dropna()) else np.nan,
        "examples": examples,
    }
    if volume_rel is not None:
        result["volume_mismatch_rate"] = round(float((volume_rel > tolerance).mean()), 4)
        result["worst_volume_diff"] = round(float(volume_rel.max()), 4) if len(volume_rel.dropna()) else np.nan
    return result


def snapshot_hash(df: pd.DataFrame) -> str:
    """规范化数据的 sha256 指纹（列排序、索引排序、固定精度）。"""
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()[sorted(frame.columns)]
    canonical = frame.round(6).to_csv(float_format="%.6f")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def format_audit(table: pd.DataFrame, diffs: list[dict] | None = None, hashes: Mapping[str, str] | None = None) -> str:
    lines = ["# 数据质量审计", ""]
    if table.empty:
        lines.append("没有可审计的数据。")
        return "\n".join(lines)
    summary = {
        "标的数": len(table),
        "有 error 的标的": int((table["errors"] > 0).sum()),
        "有 warning 的标的": int((table["warnings"] > 0).sum()),
        "缺少换手率的标的": int((table["infos"] > 0).sum()),
    }
    lines.append("| 项目 | 值 |")
    lines.append("| --- | ---: |")
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append(markdown_table(table.head(15)))
    lines.append("")
    if len(table) > 15:
        lines.append(f"（仅显示前 15 行，共 {len(table)} 行）")
        lines.append("")

    if diffs:
        lines.append("## 多源交叉校验")
        lines.append("")
        lines.append(
            markdown_table(
                pd.DataFrame(
                    [
                        {
                            "symbol": d["symbol"],
                            "common_days": d["common_days"],
                            "close_mismatch_rate": d.get("close_mismatch_rate"),
                            "volume_mismatch_rate": d.get("volume_mismatch_rate"),
                        }
                        for d in diffs
                    ]
                )
            )
        )
        lines.append("")

    if hashes:
        lines.append("## 数据指纹（sha256 前 16 位）")
        lines.append("")
        lines.append(markdown_table(pd.DataFrame([{"symbol": s, "hash": h} for s, h in list(hashes.items())[:10]])))
        lines.append("")
        lines.append("> 同一份数据必须得到同一个指纹；指纹变了说明数据变了，结论需要重新验证。")
    return "\n".join(lines)


def write_audit(outdir: str | Path, table: pd.DataFrame, diffs: list[dict] | None = None, hashes: Mapping[str, str] | None = None) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "audit.md"
    csv_path = out / "audit.csv"
    json_path = out / "audit.json"
    md_path.write_text(format_audit(table, diffs, hashes), encoding="utf-8")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({"diffs": diffs or [], "hashes": dict(hashes or {})}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"markdown": md_path, "csv": csv_path, "json": json_path}
