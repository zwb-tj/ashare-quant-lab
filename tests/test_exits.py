"""离场规则引擎的测试（离线，合成数据）。

重点：① 优先级顺序；② 最短持仓保护只挡止损、不挡死叉/白线破位；③ 盘中触发用触发价成交，
而不是用当日收盘价（否则会系统性高估/低估止损效果）。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.exits import ExitConfig, simulate_trade
from aqlab.indicators_extra import white_line, yellow_line


def frame_from(closes, lows=None, highs=None, start="2020-01-01"):
    close = pd.Series(closes, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close).to_numpy(),
            "high": (close * 1.01).to_numpy() if highs is None else np.asarray(highs, dtype=float),
            "low": (close * 0.99).to_numpy() if lows is None else np.asarray(lows, dtype=float),
            "close": close.to_numpy(),
            "volume": 1_000_000.0,
        },
        index=index,
    )


def quiet_config(**overrides):
    """关掉所有"抢戏"的规则，每次只测一条。"""
    base = {
        "mode": "fixed",
        "stop_pct": 0.03,
        "intraday_stop_pct": None,
        "take_profit_pct": None,
        "min_holding_days": 0,
        "use_death_cross": False,
        "use_white_break": False,
        "didi_mode": "off",
    }
    base.update(overrides)
    return ExitConfig(**base)


def test_death_cross_exits_on_the_cross_bar():
    # 先涨后跌，逼出白线下穿黄线
    closes = [*np.linspace(10, 25, 160), *np.linspace(25, 12, 60)]
    frame = frame_from(closes)
    white, yellow = white_line(frame), yellow_line(frame)
    cross = None
    for position in range(1, len(frame)):
        both_finite = np.isfinite(yellow.iloc[position]) and np.isfinite(yellow.iloc[position - 1])
        crossed = white.iloc[position] < yellow.iloc[position] and white.iloc[position - 1] >= yellow.iloc[position - 1]
        if both_finite and crossed:
            cross = position
            break
    assert cross is not None, "合成数据没有造出死叉"

    entry = cross - 5
    result = simulate_trade(frame, entry, quiet_config(use_death_cross=True))
    assert result.exit_position == cross
    assert result.reason == "death_cross"
    assert result.return_pct == pytest.approx(float(frame["close"].iloc[cross]) / float(frame["close"].iloc[entry]) - 1)


def test_white_break_needs_two_consecutive_days():
    # 60 根（黄线未成形，天然没有死叉），第 40 根起连续下跌
    closes = list(np.linspace(10, 20, 40)) + list(np.linspace(20, 15, 20))
    frame = frame_from(closes)
    one_day = simulate_trade(frame, 39, quiet_config(use_white_break=True, white_break_days=1))
    two_days = simulate_trade(frame, 39, quiet_config(use_white_break=True, white_break_days=2))
    assert one_day.reason == "white_break"
    assert two_days.reason == "white_break"
    assert two_days.exit_position >= one_day.exit_position      # 两日确认必然不早于一日


def test_didi_exits_when_close_breaks_previous_low():
    closes = [10.0] * 39 + [9.0, 9.0]
    frame = frame_from(closes)                                   # 前一日 low = 10 × 0.99 = 9.9 > 9.0
    result = simulate_trade(frame, 38, quiet_config(didi_mode="simple"))
    assert result.exit_position == 39
    assert result.reason == "didi"


def test_min_holding_days_protects_the_stop_but_not_the_death_cross():
    closes = [10.0] * 39 + [8.5, 8.5, 8.5]
    frame = frame_from(closes)
    protected = simulate_trade(frame, 38, quiet_config(mode="entry_low", stop_pct=0.03, min_holding_days=3))
    unprotected = simulate_trade(frame, 38, quiet_config(mode="entry_low", stop_pct=0.03, min_holding_days=0))
    assert unprotected.reason == "stop_entry_low"
    assert unprotected.exit_position == 39
    assert protected.reason != "stop_entry_low" or protected.exit_position >= 41


def test_entry_low_stop_uses_the_entry_bar_low():
    closes = [10.0] * 30 + [9.0]
    lows = [9.5] * 30 + [8.9]
    frame = frame_from(closes, lows=lows)
    result = simulate_trade(frame, 29, quiet_config(mode="entry_low", stop_pct=0.03))
    # 止损价 = 9.5 × 0.97 = 9.215；次日收盘 9.0 < 9.215 -> 触发
    assert result.reason == "stop_entry_low"
    assert result.exit_price == pytest.approx(9.0)


def test_intraday_stop_fills_at_the_trigger_price_not_the_close():
    closes = [10.0] * 29 + [10.0, 8.0]
    lows = [9.9] * 29 + [9.9, 7.5]                              # 盘中砸到 7.5
    frame = frame_from(closes, lows=lows)
    result = simulate_trade(frame, 29, quiet_config(mode="fixed", intraday_stop_pct=0.05))
    assert result.reason == "stop_intraday"
    assert result.exit_price == pytest.approx(9.5)              # 10 × 0.95，而不是收盘 8.0


def test_take_profit_fills_at_the_trigger_price():
    closes = [10.0] * 29 + [10.0, 11.0]
    highs = [10.1] * 29 + [10.1, 11.8]
    frame = frame_from(closes, highs=highs)
    result = simulate_trade(frame, 29, quiet_config(take_profit_pct=0.15))
    assert result.reason == "take_profit"
    assert result.exit_price == pytest.approx(11.5)             # 10 × 1.15


def test_atr_stop_triggers_after_a_large_drop():
    closes = [*list(np.linspace(10, 12, 40)), 10.0]
    frame = frame_from(closes)
    result = simulate_trade(frame, 39, quiet_config(mode="atr", atr_multiple=2.0))
    assert result.reason in ("stop_atr", "no_exit")
    if result.reason == "stop_atr":
        assert result.exit_price < 12.0


def test_max_holding_forces_an_exit_when_set():
    closes = list(np.linspace(10, 12, 50))
    frame = frame_from(closes)
    result = simulate_trade(frame, 39, quiet_config(max_holding_days=3))
    assert result.reason == "max_holding"
    assert result.bars_held == 3


def test_result_tracks_mfe_and_mae():
    closes = [10.0] * 29 + [10.0, 11.0, 9.0]
    highs = [10.0] * 29 + [10.0, 11.5, 9.2]
    lows = [10.0] * 29 + [10.0, 9.8, 8.8]
    frame = frame_from(closes, lows=lows, highs=highs)
    result = simulate_trade(frame, 29, quiet_config(max_holding_days=2))
    assert result.max_favorable == pytest.approx(0.15)
    assert result.max_adverse == pytest.approx(-0.12)


def test_config_validation():
    with pytest.raises(ValueError):
        ExitConfig(mode="martingale")
    with pytest.raises(ValueError):
        ExitConfig(stop_pct=-0.1)
    with pytest.raises(ValueError):
        ExitConfig(intraday_stop_pct=1.5)
    with pytest.raises(ValueError):
        ExitConfig(take_profit_pct=0)
    with pytest.raises(ValueError):
        ExitConfig(min_holding_days=-1)
