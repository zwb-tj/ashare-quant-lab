"""离场规则引擎（v0.11）：把"止损止盈 + 牵牛绳 + 白线黄线 + 滴滴"落成可回测的机械规则。

规则来源是使用者自己在用的那套（牵牛绳 = 白线别名；白线 = EMA(EMA(C,10),10)；
黄线 = (MA14+MA28+MA57+MA114)/4），这里用**本项目自己的实现**重新表达，只借用规则思想与参数。

判定优先级（每天按顺序检查，先命中先成交）::

    1. 死叉清仓      白线今日 < 黄线 且 昨日 ≥ 黄线            → 收盘清仓（最高优先级）
    2. 白线两日破位   连续 N 日 收盘 < 白线 × (1 - 阈值)        → 收盘清仓
    3. 滴滴          今日收盘 < 昨日最低                      → 收盘清仓
    4. 入场价止损     收盘 < 入场K线最低价 × (1 + 止损%)       → 收盘卖出（前 N 日不触发）
    5. 盘中止损       最低价 ≤ 入场价 × (1 - 止损%)            → 按止损价成交
    6. 盘中止盈       最高价 ≥ 入场价 × (1 + 止盈%)            → 按止盈价成交
    7. ATR 止损       收盘 < 入场收盘 - ATR(14) × 倍数
    8. 到期           max_holding_days（默认不设，原策略里没有强制持有上限）

**只看收盘的规则用收盘价成交，盘中触发的规则用触发价成交**——这是"收盘触发 vs 盘中触发"
的严格区分；用日线回测时盘中止损/止盈只能近似（真实成交还要看分时强弱）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aqlab.indicators import atr
from aqlab.indicators_extra import white_line, yellow_line

__all__ = ["EXIT_REASONS", "ExitConfig", "TradeResult", "simulate_trade"]

EXIT_REASONS = {
    "death_cross": "白线下穿黄线（牵牛绳断）",
    "white_break": "白线连续破位",
    "didi": "滴滴（今收 < 昨低）",
    "stop_entry_low": "跌破入场K线最低价止损",
    "stop_intraday": "盘中固定百分比止损",
    "take_profit": "盘中固定百分比止盈",
    "stop_atr": "ATR 止损",
    "max_holding": "持有到期",
    "no_exit": "数据结束未离场",
}


@dataclass
class ExitConfig:
    """离场参数。默认值对应"偏紧档"（BEAR），可按大盘阶段切换。"""

    mode: str = "entry_low"              # entry_low（按入场K线最低价）| fixed | atr
    stop_pct: float = 0.03               # entry_low: 止损价 = 入场最低价 × (1 - 3%)
    intraday_stop_pct: float | None = None   # 固定百分比盘中止损（如 0.07）
    take_profit_pct: float | None = 0.15     # 盘中止盈（如 0.15）
    atr_window: int = 14
    atr_multiple: float = 2.0
    min_holding_days: int = 3            # 前 N 日不止损（BULL 5 / SIDEWAYS 3 / BEAR 2）
    white_break_days: int = 2
    white_break_threshold: float = 0.01
    use_death_cross: bool = True
    use_white_break: bool = True
    didi_mode: str = "full"              # full（连续两根阴线+破昨低+量能不缩+不在深跌区）| simple | off
    max_holding_days: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("entry_low", "fixed", "atr"):
            raise ValueError("mode must be 'entry_low', 'fixed' or 'atr'")
        if self.stop_pct < 0:
            raise ValueError("stop_pct must be >= 0")
        if self.min_holding_days < 0:
            raise ValueError("min_holding_days must be >= 0")
        if self.white_break_days < 1:
            raise ValueError("white_break_days must be >= 1")
        if self.didi_mode not in ("full", "simple", "off"):
            raise ValueError("didi_mode must be 'full', 'simple' or 'off'")
        if self.intraday_stop_pct is not None and not 0 < self.intraday_stop_pct < 1:
            raise ValueError("intraday_stop_pct must be in (0, 1)")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be > 0")

    def params(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class TradeResult:
    exit_position: int
    exit_date: pd.Timestamp | None
    exit_price: float
    reason: str
    bars_held: int
    return_pct: float
    max_favorable: float = np.nan          # 持仓期内最大浮盈
    max_adverse: float = np.nan            # 持仓期内最大浮亏


def simulate_trade(
    daily: pd.DataFrame,
    entry_position: int,
    config: ExitConfig | None = None,
    max_bars: int = 120,
) -> TradeResult:
    """从 ``entry_position``（入场那根 K 线）开始逐日检查离场规则。

    ``entry_position`` 就是**入场日**在 ``daily`` 里的位置；函数只看该位置**之后**的数据，
    不使用任何未来信息以外的额外输入（离场必然发生在入场之后）。
    """
    config = config or ExitConfig()
    frame = daily
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float) if "open" in frame.columns else close
    low = frame["low"].astype(float)
    high = frame["high"].astype(float)
    volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    white = white_line(frame)
    yellow = yellow_line(frame)
    atr_series = atr(frame, config.atr_window) if config.mode == "atr" else None
    hhv20 = high.rolling(20, min_periods=5).max()

    entry_price = float(close.iloc[entry_position])
    entry_low = float(low.iloc[entry_position])
    stop_price = entry_low * (1.0 - config.stop_pct)

    last = min(len(frame) - 1, entry_position + max_bars)
    white_breaks = 0
    best = -np.inf
    worst = np.inf

    for position in range(entry_position + 1, last + 1):
        price = float(close.iloc[position])
        if not np.isfinite(price) or price <= 0:
            continue
        bars = position - entry_position
        best = max(best, float(high.iloc[position]) / entry_price - 1.0)
        worst = min(worst, float(low.iloc[position]) / entry_price - 1.0)
        can_stop = bars >= config.min_holding_days

        # 0a. 盘中止损：挂在市场里的单子先成交，不管收盘规则怎么说
        if can_stop and config.intraday_stop_pct is not None:
            trigger = entry_price * (1.0 - config.intraday_stop_pct)
            if float(low.iloc[position]) <= trigger:
                return _result(position, frame, trigger, "stop_intraday", bars, entry_price, best, worst)

        # 0b. 盘中止盈
        if config.take_profit_pct is not None:
            trigger = entry_price * (1.0 + config.take_profit_pct)
            if float(high.iloc[position]) >= trigger:
                return _result(position, frame, trigger, "take_profit", bars, entry_price, best, worst)

        # 1. 死叉清仓（牵牛绳断）
        if config.use_death_cross and np.isfinite(white.iloc[position]) and np.isfinite(yellow.iloc[position]):
            previous_white, previous_yellow = white.iloc[position - 1], yellow.iloc[position - 1]
            if (
                np.isfinite(previous_white)
                and np.isfinite(previous_yellow)
                and white.iloc[position] < yellow.iloc[position]
                and previous_white >= previous_yellow
            ):
                return _result(position, frame, price, "death_cross", bars, entry_price, best, worst)

        # 2. 白线两日破位
        if config.use_white_break and np.isfinite(white.iloc[position]) and white.iloc[position] > 0:
            if price < float(white.iloc[position]) * (1.0 - config.white_break_threshold):
                white_breaks += 1
            else:
                white_breaks = 0
            if white_breaks >= config.white_break_days:
                return _result(position, frame, price, "white_break", bars, entry_price, best, worst)
        else:
            white_breaks = 0

        # 3. 滴滴
        if config.didi_mode != "off":
            triggered = False
            previous_low = float(low.iloc[position - 1])
            if np.isfinite(previous_low) and price < previous_low:
                if config.didi_mode == "simple":
                    triggered = True
                else:
                    previous_open = float(open_.iloc[position - 1])
                    this_open = float(open_.iloc[position])
                    previous_volume = float(volume.iloc[position - 1])
                    this_volume = float(volume.iloc[position])
                    two_bearish = price < this_open and float(close.iloc[position - 1]) < previous_open
                    volume_ok = not (np.isfinite(previous_volume) and np.isfinite(this_volume)) or this_volume >= previous_volume * 0.8
                    high_ok = not np.isfinite(hhv20.iloc[position]) or price >= float(hhv20.iloc[position]) * 0.8
                    triggered = two_bearish and volume_ok and high_ok
            if triggered:
                return _result(position, frame, price, "didi", bars, entry_price, best, worst)

        # 4. 入场价止损（收盘，看收盘不看盘中）
        if can_stop and config.mode == "entry_low" and price < stop_price:
            return _result(position, frame, price, "stop_entry_low", bars, entry_price, best, worst)

        # 5. ATR 止损
        if can_stop and config.mode == "atr" and atr_series is not None:
            value = float(atr_series.iloc[entry_position])
            if np.isfinite(value) and price < entry_price - value * config.atr_multiple:
                return _result(position, frame, price, "stop_atr", bars, entry_price, best, worst)

        # 6. 到期
        if config.max_holding_days is not None and bars >= config.max_holding_days:
            return _result(position, frame, price, "max_holding", bars, entry_price, best, worst)

    final_position = last
    final_price = float(close.iloc[final_position])
    reason = "max_holding" if (config.max_holding_days is not None and final_position - entry_position >= config.max_holding_days) else "no_exit"
    return _result(final_position, frame, final_price, reason, final_position - entry_position, entry_price, best, worst)


def _result(
    position: int,
    frame: pd.DataFrame,
    price: float,
    reason: str,
    bars: int,
    entry_price: float,
    best: float,
    worst: float,
) -> TradeResult:
    date = frame.index[position]
    return TradeResult(
        exit_position=position,
        exit_date=pd.Timestamp(date),
        exit_price=float(price),
        reason=reason,
        bars_held=int(bars),
        return_pct=float(price) / entry_price - 1.0,
        max_favorable=best if np.isfinite(best) else np.nan,
        max_adverse=worst if np.isfinite(worst) else np.nan,
    )
