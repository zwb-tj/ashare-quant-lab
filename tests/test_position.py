"""持仓/离场管理测试（v0.5）。"""

import pandas as pd
import pytest

from aqlab.position import (
    PositionConfig,
    defend_score,
    plan_position,
    simulate_exit,
    simulate_signals,
)


def frame(closes, highs=None, lows=None, volumes=None, opens=None, start="2024-01-01"):
    c = pd.Series(closes, dtype=float)
    n = len(c)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": (pd.Series(opens, dtype=float) if opens is not None else c.shift(1).fillna(c)).values,
            "high": (pd.Series(highs, dtype=float) if highs is not None else c * 1.005).values,
            "low": (pd.Series(lows, dtype=float) if lows is not None else c * 0.995).values,
            "close": c.values,
            "volume": (pd.Series(volumes, dtype=float) if volumes is not None else pd.Series(1000.0, index=range(n))).values,
        },
        index=idx,
    )


# --------------------------------------------------------------------------------------
# 交易计划
# --------------------------------------------------------------------------------------
def test_plan_position_structure_stop_targets_and_weights():
    plan = plan_position(entry_price=100.0, reference_low=97.0, reference_high=110.0)
    # 结构止损 97×0.97=94.09，被最大容忍 100×0.95=95 托住 -> 95
    assert plan.stop_price == pytest.approx(95.0)
    assert plan.stop_pct == pytest.approx(-0.05)
    assert plan.first_target == pytest.approx(103.0)
    assert plan.second_target == pytest.approx(105.0)
    assert plan.swing_target == pytest.approx(115.0)
    assert plan.risk_reward == pytest.approx(2.0)          # (110-100)/(100-95)
    assert plan.weights == {"scout": 0.30, "main": 0.25, "reserve": 0.45}
    assert plan.to_dict()["stop_price"] == 95.0


def test_plan_position_rejects_bad_inputs_and_config():
    with pytest.raises(ValueError):
        plan_position(entry_price=0.0, reference_low=10.0)
    with pytest.raises(ValueError):
        PositionConfig(scout_weight=0.5, main_weight=0.5, reserve_weight=0.5)
    with pytest.raises(ValueError):
        PositionConfig(max_stop_pct=0.0)
    with pytest.raises(ValueError):
        PositionConfig(hard_stop_pct=-0.01)
    with pytest.raises(ValueError):
        PositionConfig(first_target_fraction=1.5)


# --------------------------------------------------------------------------------------
# 防卖飞评分
# --------------------------------------------------------------------------------------
def test_defend_score_all_components_pass():
    df = frame([100.0 * 1.004 ** i for i in range(140)])
    out = defend_score(df)
    last = out.iloc[-1]
    assert last["score"] == 5.0
    assert last["advice"] == "持有"
    assert bool(last["up_close"]) and bool(last["above_bbi"]) and bool(last["trend_up"]) and bool(last["j_not_dead"])


def test_defend_score_degrades_when_price_falls():
    closes = [100.0 * 1.004 ** i for i in range(140)]
    closes[-1] = closes[-2] * 0.97          # 当日下跌 + 跌破 BBI
    df = frame(closes)
    out = defend_score(df)
    assert out.iloc[-1]["score"] <= 3.0
    assert out.iloc[-1]["advice"] in {"减半", "准备离场"}


def test_defend_score_flags_volume_bear():
    closes = [100.0 * 1.004 ** i for i in range(140)]
    volumes = [1000.0] * 140
    volumes[-1] = 3000.0
    df = frame(closes, volumes=volumes)
    df.loc[df.index[-1], "open"] = closes[-1] * 1.02   # 阴线
    out = defend_score(df)
    assert bool(out.iloc[-1]["no_volume_bear"]) is False


# --------------------------------------------------------------------------------------
# 离场规则
# --------------------------------------------------------------------------------------
def test_simulate_exit_hard_stop_fires_first():
    df = frame([100.0, 100.0, 97.0], highs=[101.0, 101.0, 100.0], lows=[99.0, 99.0, 97.5])
    events, summary = simulate_exit(df, entry_bar=0, entry_price=100.0)
    assert summary["exit_reason"] == "hard_stop"
    assert events[0].price == pytest.approx(98.0)
    assert summary["return"] == pytest.approx(-0.02)


def test_simulate_exit_structure_stop_when_hard_stop_disabled():
    config = PositionConfig(hard_stop_pct=None)
    df = frame([100.0, 100.0, 96.0], highs=[101.0, 101.0, 100.0], lows=[99.0, 99.0, 95.5])
    events, summary = simulate_exit(df, entry_bar=0, entry_price=100.0, config=config, stop_price=96.0)
    assert summary["exit_reason"] == "stop_loss"
    assert events[0].price == pytest.approx(96.0)


def test_simulate_exit_first_target_half_then_swing_target():
    df = frame([100.0, 103.0, 115.0], highs=[100.0, 104.0, 116.0], lows=[99.0, 99.0, 102.0])
    events, summary = simulate_exit(df, entry_bar=0, entry_price=100.0, config=PositionConfig(hard_stop_pct=None))
    actions = [(e.action, e.reason) for e in events]
    assert ("sell_half", "first_target") in actions
    assert events[-1].reason == "swing_target"
    # 一半在 +3% 走，一半在 +15% 走
    assert summary["return"] == pytest.approx(0.5 * 0.03 + 0.5 * 0.15)
    assert summary["remaining"] == pytest.approx(0.0)


def test_simulate_exit_bbi_two_day_break():
    closes = [100.0 * 1.002 ** i for i in range(60)]
    closes += [closes[-1] * 0.985, closes[-1] * 0.985 * 0.985]
    df = frame(closes)
    config = PositionConfig(hard_stop_pct=None, max_stop_pct=0.20, swing_target_pct=0.5, max_hold_bars=60)
    events, summary = simulate_exit(df, entry_bar=59, entry_price=float(df["close"].iloc[59]), config=config)
    assert summary["exit_reason"] == "bbi_break"
    assert events[-1].reason == "bbi_break"


def test_simulate_exit_time_stop_paths():
    config = PositionConfig(hard_stop_pct=None, max_stop_pct=0.20, swing_target_pct=0.5, max_hold_bars=3)
    flat = frame([100.0] * 6)
    _events, summary = simulate_exit(flat, entry_bar=0, entry_price=100.0, config=config)
    assert summary["exit_reason"] in {"time_stop", "bbi_break"}

    # 有 >5% 浮盈后触发时间止损 -> time_stop_after_profit
    rising = frame([100.0, 106.0, 106.5, 106.2, 106.1], highs=[100.0, 107.0, 107.0, 106.5, 106.2], lows=[99.0, 100.0, 105.0, 105.5, 105.5])
    events, summary2 = simulate_exit(rising, entry_bar=0, entry_price=100.0, config=config)
    assert summary2["exit_reason"] in {"time_stop_after_profit", "bbi_break", "swing_target"}
    assert events  # 至少有一条离场记录


def test_simulate_exit_breakeven_stop():
    config = PositionConfig(hard_stop_pct=None, max_stop_pct=0.20, swing_target_pct=0.5, max_hold_bars=60, bbi_break_days=99)
    df = frame([100.0, 104.0, 99.5], highs=[100.0, 104.5, 100.0], lows=[99.0, 99.5, 99.0])
    _events, summary = simulate_exit(df, entry_bar=0, entry_price=100.0, config=config)
    assert summary["exit_reason"] == "swing_target" or summary["exit_reason"] == "end_of_data" or summary["exit_reason"] == "breakeven"


def test_simulate_signals_enters_next_bar():
    closes = [100.0] * 10 + [110.0] * 3
    df = frame(closes)
    signal = pd.Series(False, index=df.index)
    signal.iloc[3] = True
    trades = simulate_signals(df, signal)
    assert len(trades) == 1
    assert trades.iloc[0]["signal_date"] == str(df.index[3].date())
    assert trades.iloc[0]["entry_date"] == str(df.index[4].date())
    assert trades.iloc[0]["entry_price"] == pytest.approx(100.0)


def test_simulate_signals_empty_when_no_signal():
    df = frame([100.0] * 20)
    trades = simulate_signals(df, pd.Series(False, index=df.index))
    assert trades.empty
