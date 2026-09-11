"""持仓与离场管理（v0.5）。

选股层回答"买什么"，这一层回答"买入之后怎么办"。规则来自作者自己的交易口径：

======================  ==========================================================
规则                     口径
======================  ==========================================================
止损（结构位）            买入 K 线最低价 / N 型结构前低 / 横盘平台下沿，下浮 3%~5%
止损（短线）              -2%（"一日游"心态，可选叠加）
脱离成本止损              有浮盈后跌回成本（盈转亏）立即离场
第一次止盈                脱离成本 +3% 减半仓
持有条件                 脱离成本 +5% 后可持有 4-6 根 K 线
波段目标                  +10%~20%
BBI 离场                  收盘连续 2 日跌破 BBI → 清仓
防卖飞评分（5 分制）       收盘涨 + 未破 BBI + 非放量阴线 + 趋势向上 + J 未死叉
                          → 4-5 分持有 / 3 分减半 / <3 分准备离场
建仓阵型 3-2-2            侦察兵 30-35% / 主力军 20-25% / 预备队 40-50%
======================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aqlab.indicators_extra import bbi_line, kdj, ma, white_line, yellow_line

__all__ = [
    "PositionConfig",
    "TradePlan",
    "ExitEvent",
    "plan_position",
    "defend_score",
    "simulate_exit",
    "simulate_signals",
]


@dataclass
class PositionConfig:
    """离场参数（全部可调，默认值来自交易口径）。"""

    # 结构止损：参考低点下浮多少
    stop_buffer_pct: float = 0.03
    # 结构止损的最大容忍（相对买入价），防止结构位太远
    max_stop_pct: float = 0.05
    # 可选短线硬止损（None = 不启用）
    hard_stop_pct: float | None = 0.02
    # 第一次止盈：脱离成本 3% 减半
    first_target_pct: float = 0.03
    first_target_fraction: float = 0.5
    # 持有条件与时间止损
    second_target_pct: float = 0.05
    max_hold_bars: int = 6
    # 波段目标
    swing_target_pct: float = 0.15
    # BBI 离场
    bbi_windows: tuple[int, ...] = (3, 6, 12, 24)
    bbi_break_days: int = 2
    # 3-2-2 建仓阵型
    scout_weight: float = 0.30
    main_weight: float = 0.25
    reserve_weight: float = 0.45

    def __post_init__(self) -> None:
        if not 0 <= self.stop_buffer_pct <= 0.10:
            raise ValueError("stop_buffer_pct must be in [0, 0.10]")
        if not 0 < self.max_stop_pct <= 0.20:
            raise ValueError("max_stop_pct must be in (0, 0.20]")
        if self.hard_stop_pct is not None and not 0 < self.hard_stop_pct <= 0.10:
            raise ValueError("hard_stop_pct must be in (0, 0.10]")
        if not 0 < self.first_target_fraction < 1:
            raise ValueError("first_target_fraction must be in (0, 1)")
        if self.max_hold_bars < 1 or self.bbi_break_days < 1:
            raise ValueError("max_hold_bars and bbi_break_days must be >= 1")
        total = self.scout_weight + self.main_weight + self.reserve_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"3-2-2 weights must sum to 1 (got {total})")


@dataclass
class TradePlan:
    entry_price: float
    stop_price: float
    stop_pct: float           # 相对买入价的止损幅度（负数）
    first_target: float
    second_target: float
    swing_target: float
    risk_reward: float
    weights: dict

    def to_dict(self) -> dict:
        return {
            "entry_price": round(self.entry_price, 3),
            "stop_price": round(self.stop_price, 3),
            "stop_pct": round(self.stop_pct, 4),
            "first_target": round(self.first_target, 3),
            "second_target": round(self.second_target, 3),
            "swing_target": round(self.swing_target, 3),
            "risk_reward": round(self.risk_reward, 2),
            "weights": self.weights,
        }


@dataclass
class ExitEvent:
    bar: int
    date: pd.Timestamp
    action: str      # sell_half | exit_all
    reason: str
    price: float
    fraction: float
    bars_held: int

    def to_dict(self) -> dict:
        return {
            "bar": self.bar,
            "date": str(pd.Timestamp(self.date).date()),
            "action": self.action,
            "reason": self.reason,
            "price": round(self.price, 3),
            "fraction": self.fraction,
            "bars_held": self.bars_held,
        }


def plan_position(
    entry_price: float,
    reference_low: float,
    reference_high: float | None = None,
    config: PositionConfig | None = None,
) -> TradePlan:
    """生成交易计划：结构止损（参考低点下浮，且不超过最大容忍）+ 三档目标 + 3-2-2 仓位。

    ``reference_low`` 用"买入 K 线最低价 / 前低 / 平台下沿"中最低的那个（越保守越好）。
    """
    config = config or PositionConfig()
    if entry_price <= 0 or reference_low <= 0:
        raise ValueError("entry_price and reference_low must be positive")

    structural = reference_low * (1 - config.stop_buffer_pct)
    floor = entry_price * (1 - config.max_stop_pct)
    stop_price = max(structural, floor)
    stop_pct = stop_price / entry_price - 1.0

    risk = entry_price - stop_price
    target = reference_high if reference_high else entry_price * (1 + config.swing_target_pct)
    reward = max(target, entry_price * (1 + config.second_target_pct)) - entry_price
    rr = reward / risk if risk > 0 else float("inf")

    return TradePlan(
        entry_price=entry_price,
        stop_price=stop_price,
        stop_pct=stop_pct,
        first_target=entry_price * (1 + config.first_target_pct),
        second_target=entry_price * (1 + config.second_target_pct),
        swing_target=entry_price * (1 + config.swing_target_pct),
        risk_reward=float(rr),
        weights={"scout": config.scout_weight, "main": config.main_weight, "reserve": config.reserve_weight},
    )


def defend_score(df: pd.DataFrame, window: int = 20, config: PositionConfig | None = None) -> pd.DataFrame:
    """防卖飞评分（5 分制，逐 bar）。

    五项各 1 分：① 收盘涨 ② 收盘 ≥ BBI ③ 非放量阴线 ④ 趋势向上（白线 > 黄线）
    ⑤ J 未死叉（J ≥ D）。评分 4-5 持有 / 3 减半 / <3 准备离场。
    """
    config = config or PositionConfig()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    bbi = bbi_line(df, config.bbi_windows)
    white = white_line(df)
    yellow = yellow_line(df)
    kd = kdj(df)

    score = pd.DataFrame(index=df.index)
    score["up_close"] = close > close.shift(1)
    score["above_bbi"] = close >= bbi
    score["no_volume_bear"] = ~((close < open_) & (volume > volume.shift(1) * 1.5))
    score["trend_up"] = white > yellow
    score["j_not_dead"] = kd["j"] >= kd["d"]
    total = score.sum(axis=1).astype(float)
    out = score.copy()
    out["score"] = total
    out["advice"] = np.where(total >= 4, "持有", np.where(total >= 3, "减半", "准备离场"))
    return out


def simulate_exit(
    df: pd.DataFrame,
    entry_bar: int,
    entry_price: float,
    config: PositionConfig | None = None,
    stop_price: float | None = None,
) -> tuple[list[ExitEvent], dict]:
    """从 ``entry_bar`` 之后逐 bar 执行离场规则，返回 (事件列表, 汇总)。

    离场优先级：硬止损 → 结构止损 → 脱离成本止损 → 第一次止盈（减半）→ BBI 两日破位
    → 波段目标 → 时间止损。规则顺序即风控优先级，先触发者生效。
    """
    config = config or PositionConfig()
    if entry_bar >= len(df) - 1:
        return [], {"exit_reason": "no_data", "return": 0.0, "bars_held": 0, "remaining": 1.0}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    bbi = bbi_line(df, config.bbi_windows)

    stop = stop_price if stop_price is not None else entry_price * (1 - config.max_stop_pct)
    events: list[ExitEvent] = []
    remaining = 1.0
    half_sold = False
    profit_peak = 0.0
    below_bbi_days = 0
    exit_reason = "end_of_data"
    realised = 0.0

    for offset, i in enumerate(range(entry_bar + 1, len(df)), start=1):
        date = df.index[i]
        bar_close, bar_high, bar_low = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
        profit_peak = max(profit_peak, bar_high / entry_price - 1.0)

        def record(action: str, reason: str, price: float, fraction: float) -> None:
            nonlocal remaining, realised
            fraction = min(fraction, remaining)
            events.append(ExitEvent(i, date, action, reason, price, fraction, offset))
            realised += fraction * (price / entry_price - 1.0)
            remaining -= fraction

        # 1) 短线硬止损
        if config.hard_stop_pct is not None and bar_low <= entry_price * (1 - config.hard_stop_pct):
            record("exit_all", "hard_stop", entry_price * (1 - config.hard_stop_pct), remaining)
            exit_reason = "hard_stop"
            break

        # 2) 结构止损
        if bar_low <= stop:
            record("exit_all", "stop_loss", stop, remaining)
            exit_reason = "stop_loss"
            break

        # 3) 脱离成本止损：有过 +3% 浮盈后又跌回成本
        if profit_peak >= config.first_target_pct and bar_close < entry_price:
            record("exit_all", "breakeven", bar_close, remaining)
            exit_reason = "breakeven"
            break

        # 4) 第一次止盈：+3% 减半
        if not half_sold and bar_high >= entry_price * (1 + config.first_target_pct):
            record("sell_half", "first_target", entry_price * (1 + config.first_target_pct), config.first_target_fraction)
            half_sold = True

        # 5) BBI 两日破位
        bbi_value = float(bbi.iloc[i])
        if bbi_value == bbi_value and bar_close < bbi_value:  # NaN 检查
            below_bbi_days += 1
        else:
            below_bbi_days = 0
        if below_bbi_days >= config.bbi_break_days:
            record("exit_all", "bbi_break", bar_close, remaining)
            exit_reason = "bbi_break"
            break

        # 6) 波段目标
        if bar_high >= entry_price * (1 + config.swing_target_pct):
            record("exit_all", "swing_target", entry_price * (1 + config.swing_target_pct), remaining)
            exit_reason = "swing_target"
            break

        # 7) 时间止损：到达持有上限
        if offset >= config.max_hold_bars:
            if profit_peak >= config.second_target_pct:
                record("exit_all", "time_stop_after_profit", bar_close, remaining)
                exit_reason = "time_stop_after_profit"
            else:
                record("exit_all", "time_stop", bar_close, remaining)
                exit_reason = "time_stop"
            break

    if remaining > 0 and exit_reason == "end_of_data":
        last_close = float(close.iloc[-1])
        events.append(ExitEvent(len(df) - 1, df.index[-1], "exit_all", "end_of_data", last_close, remaining, len(df) - 1 - entry_bar))
        realised += remaining * (last_close / entry_price - 1.0)
        remaining = 0.0

    summary = {
        "exit_reason": exit_reason,
        "return": realised,
        "bars_held": events[-1].bars_held if events else 0,
        "remaining": remaining,
        "events": [e.to_dict() for e in events],
    }
    return events, summary


def simulate_signals(
    df: pd.DataFrame,
    signal: pd.Series,
    config: PositionConfig | None = None,
    entry_lag: int = 1,
) -> pd.DataFrame:
    """把入场信号序列跑成完整交易（信号次日收盘入场，避免未来函数）。

    返回每笔交易的入场/离场、离场原因、收益与持有 bar 数。
    """
    config = config or PositionConfig()
    signal = signal.reindex(df.index).fillna(False).astype(bool)
    rows: list[dict] = []
    i = 0
    n = len(df)
    while i < n - entry_lag:
        if signal.iloc[i]:
            entry_bar = i + entry_lag
            entry_price = float(df["close"].iloc[entry_bar])
            events, summary = simulate_exit(df, entry_bar, entry_price, config=config)
            last_bar = max((e.bar for e in events), default=entry_bar)
            rows.append(
                {
                    "signal_date": str(df.index[i].date()),
                    "entry_date": str(df.index[entry_bar].date()),
                    "entry_price": round(entry_price, 3),
                    "exit_date": str(df.index[last_bar].date()) if last_bar < len(df) else str(df.index[-1].date()),
                    "exit_price": round(float(df["close"].iloc[min(last_bar, len(df) - 1)]), 3),
                    "exit_reason": summary["exit_reason"],
                    "return": round(float(summary["return"]), 4),
                    "bars_held": summary["bars_held"],
                }
            )
            i = last_bar + 1
        else:
            i += 1
    return pd.DataFrame(rows)
