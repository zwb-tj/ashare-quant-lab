"""全市场 B1 研究（v0.11）：把选股、大盘阶段、离场规则放在同一条流水线上评估。

回答三个问题：

1. **B1 到底有没有用**——不再只看使用者发布的那几十只，而是扫全市场；
2. **大盘阶段（0AMV 波段）是不是决定性的**——把交易按"开波段/关波段"切开分别统计；
3. **离场规则值多少**——同一批信号，机械持有 h 日 vs 止损止盈/白线黄线离场，直接对比。

口径：``T`` 日收盘出信号 → ``T+1`` 日**开盘**买入（T+1 制度）→ 由 :mod:`aqlab.exits` 逐日判定离场。
基准：**全市场等权指数**（当日所有可用标的涨跌幅的等权平均累乘），用于算同期超额。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aqlab.exits import ExitConfig, simulate_trade
from aqlab.indicators_extra import white_line, yellow_line
from aqlab.rules_zgnb import ActiveMarketValueGate, build_personal_rule

__all__ = ["UniverseStudyConfig", "equal_weight_index", "load_universe_daily", "study_universe", "summarize_trades"]


@dataclass
class UniverseStudyConfig:
    rule: str = "b1_graded"
    rule_params: dict[str, Any] = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    exit: ExitConfig = field(default_factory=ExitConfig)
    use_regime_gate: bool = False          # True 时只在"开波段"日允许买入
    limit: int | None = None               # 只扫前 N 只（冒烟测试用）


def load_universe_daily(daily_dir: str | Path, limit: int | None = None) -> dict[str, pd.DataFrame]:
    """读取全市场日线缓存（每只票一个 CSV）。"""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(daily_dir).glob("*.csv")):
        if limit is not None and len(frames) >= limit:
            break
        try:
            frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        except Exception:
            continue
        if len(frame) > 30:
            frames[path.stem] = frame
    return frames


def equal_weight_index(frames: Iterable[pd.DataFrame]) -> pd.Series:
    """全市场等权指数：每日取所有可用标的涨跌幅的等权平均后累乘。"""
    returns_sum: pd.Series | None = None
    counts: pd.Series | None = None
    for frame in frames:
        pct = frame["close"].astype(float).pct_change()
        pct = pct.replace([np.inf, -np.inf], np.nan)
        returns_sum = pct if returns_sum is None else returns_sum.add(pct, fill_value=0.0)
        mask = pct.notna().astype(float)
        counts = mask if counts is None else counts.add(mask, fill_value=0.0)
    if returns_sum is None or counts is None:
        return pd.Series(dtype=float)
    mean_return = (returns_sum / counts.replace(0.0, np.nan)).dropna().sort_index()
    return (1.0 + mean_return).cumprod().rename("equal_weight_index")


def _regime_series(gate: ActiveMarketValueGate, frames: dict[str, pd.DataFrame]) -> pd.Series:
    """开波段=1 / 关波段=0，并**后移一天**：用"昨天收盘已知的状态"决定今天能不能买。"""
    raw = gate.gate_series(frames)
    return raw["gate"].shift(1).fillna(0).astype(int)


def study_universe(
    frames: dict[str, pd.DataFrame],
    config: UniverseStudyConfig | None = None,
    gate: ActiveMarketValueGate | None = None,
    rule: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """扫全市场：找信号 → （可选）大盘阶段过滤 → 模拟离场 → 记录每笔交易与超额。

    ``rule`` 可传入自定义规则对象（需实现 ``signal(df) -> Series[bool]``），便于测试与替换。
    """
    config = config or UniverseStudyConfig()
    rule = rule if rule is not None else build_personal_rule(config.rule, **config.rule_params)
    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None

    index = equal_weight_index(frames.values())
    regime = _regime_series(gate or ActiveMarketValueGate(), frames) if config.use_regime_gate else None

    trades: list[dict[str, Any]] = []
    gated_out = 0
    for symbol, frame in frames.items():
        frame = frame.sort_index()
        try:
            signal = rule.signal(frame)
        except Exception:
            continue
        hits = np.flatnonzero(np.asarray(signal.fillna(False), dtype=bool))
        if len(hits) == 0:
            continue
        white = white_line(frame)
        yellow = yellow_line(frame)
        for hit in hits:
            entry_position = hit + 1
            if entry_position >= len(frame):
                continue
            signal_date = pd.Timestamp(frame.index[hit])
            entry_date = pd.Timestamp(frame.index[entry_position])
            if start is not None and signal_date < start:
                continue
            if end is not None and signal_date > end:
                continue
            if regime is not None:
                state = int(regime.get(entry_date, 0)) if entry_date in regime.index else 0
                if state != 1:
                    gated_out += 1
                    continue
            entry_price = float(frame["open"].iloc[entry_position])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            result = simulate_trade(frame, entry_position, config.exit)
            bench = np.nan
            if len(index) and entry_date in index.index and result.exit_date is not None and result.exit_date in index.index:
                bench = float(index.loc[result.exit_date] / index.loc[entry_date] - 1.0)
            records = {
                "symbol": symbol,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "entry_price": round(entry_price, 3),
                "exit_date": result.exit_date,
                "exit_price": round(result.exit_price, 3),
                "reason": result.reason,
                "bars": result.bars_held,
                "return_pct": result.return_pct,
                "bench_pct": bench,
                "excess_pct": result.return_pct - bench if np.isfinite(bench) else np.nan,
                "max_favorable": result.max_favorable,
                "max_adverse": result.max_adverse,
            }
            for horizon in (1, 3, 5, 10, 20):
                position = entry_position + horizon - 1
                records[f"hold_{horizon}"] = (
                    float(frame["close"].iloc[position]) / entry_price - 1.0 if position < len(frame) else np.nan
                )
            white_value = float(white.iloc[entry_position]) if np.isfinite(white.iloc[entry_position]) else np.nan
            yellow_value = float(yellow.iloc[entry_position]) if np.isfinite(yellow.iloc[entry_position]) else np.nan
            records["white_above_yellow"] = bool(white_value > yellow_value) if np.isfinite(white_value) and np.isfinite(yellow_value) else None
            records["entry_above_white"] = bool(entry_price > white_value) if np.isfinite(white_value) else None
            trades.append(records)

    table = pd.DataFrame(trades)
    if not table.empty:
        table["year"] = pd.to_datetime(table["signal_date"]).dt.year
    meta = {
        "rule": config.rule,
        "rule_params": config.rule_params,
        "trades": len(table),
        "gated_out": gated_out,
        "symbols": len(frames),
        "start": config.start,
        "end": config.end,
        "exit": config.exit.params(),
        "regime_gate": config.use_regime_gate,
    }
    return table, meta


def summarize_trades(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """汇总：笔数、平均收益/基准/超额、胜率、持有天数、超额 t 值。"""
    if table is None or table.empty:
        return pd.DataFrame()

    def describe(group: pd.DataFrame, label: str) -> dict[str, Any]:
        returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
        excess = pd.to_numeric(group["excess_pct"], errors="coerce").dropna()
        t_stat = np.nan
        if len(excess) > 2 and excess.std(ddof=1) > 0:
            t_stat = float(excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
        row = {
            "组": label,
            "笔数": len(group),
            "平均收益%": round(float(returns.mean()) * 100, 2) if len(returns) else np.nan,
            "中位收益%": round(float(returns.median()) * 100, 2) if len(returns) else np.nan,
            "胜率%": round(float((returns > 0).mean()) * 100, 1) if len(returns) else np.nan,
            "平均超额%": round(float(excess.mean()) * 100, 2) if len(excess) else np.nan,
            "超额胜率%": round(float((excess > 0).mean()) * 100, 1) if len(excess) else np.nan,
            "超额t": round(t_stat, 2) if np.isfinite(t_stat) else np.nan,
            "平均持有": round(float(pd.to_numeric(group["bars"], errors="coerce").mean()), 1),
            "平均最大浮亏%": round(float(pd.to_numeric(group["max_adverse"], errors="coerce").mean()) * 100, 2),
        }
        return row

    rows = [describe(table, "全部")]
    if by:
        for key, group in table.groupby(by):
            rows.append(describe(group, str(key)))
    return pd.DataFrame(rows)
