"""开盘量比确认测试（v0.8）。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.data import generate_synthetic_ohlcv
from aqlab.intraday import (
    IntradayConfig,
    confirm_signals,
    confirmed_signal_series,
    generate_synthetic_minutes,
    opening_volume_ratio,
    opening_window_price,
    opening_window_volume,
    standard_volume_ratio,
)


def minutes_for_day(day: str, volumes):
    stamps = pd.date_range(f"{day} 09:30", periods=len(volumes), freq="1min")
    return pd.DataFrame({"close": 100.0, "volume": volumes}, index=stamps)


def two_week_minutes():
    """两周分钟数据：每天前 7 分钟累计量已知（均为 70，最后一天为 210）。"""
    frames = []
    days = pd.bdate_range("2024-01-01", periods=10)
    for i, day in enumerate(days):
        volumes = [10.0] * 10 + [1.0] * 10
        if i == len(days) - 1:
            volumes = [30.0] * 10 + [1.0] * 10      # 最后一天开盘放量 3 倍
        frames.append(minutes_for_day(str(day.date()), volumes))
    return pd.concat(frames)


# --------------------------------------------------------------------------------------
# 窗口量与量比
# --------------------------------------------------------------------------------------
def test_opening_window_volume_sums_the_first_minutes_only():
    bars = minutes_for_day("2024-01-02", [5.0] * 5 + [100.0] * 5)
    config = IntradayConfig(window_minutes=3)
    volume = opening_window_volume(bars, config)
    assert volume.iloc[0] == pytest.approx(15.0)          # 只统计前 3 分钟
    assert opening_window_volume(bars, IntradayConfig(window_minutes=10)).iloc[0] == pytest.approx(525.0)


def test_opening_volume_ratio_uses_previous_days_only():
    bars = two_week_minutes()
    config = IntradayConfig(window_minutes=7, baseline_days=5, min_ratio=1.0)
    ratio = opening_volume_ratio(bars, config)

    assert len(ratio) == 10
    assert ratio.iloc[:5].isna().all()                    # 历史窗口未满 -> NaN（弃答）
    assert ratio.iloc[5] == pytest.approx(1.0)            # 前 5 天都是 70 -> 均值 70
    assert ratio.iloc[6] == pytest.approx(1.0)
    assert ratio.iloc[-1] == pytest.approx(3.0)           # 最后一天 210 / 70
    # 当日放量不会影响当日基准（用 shift(1)）
    assert ratio.iloc[-2] == pytest.approx(1.0)


def test_standard_volume_ratio_matches_the_screen_formula():
    """软件口径：窗口每分钟均量 ÷ 过去 5 日全天每分钟均量。

    前两天：窗口量 70（前 7 根各 10），日总量 700 -> 全天每分钟 = 700/240
    第 6 天：窗口量 70，过去 5 日日均 700 -> 量比 = (70/7)/(700/240) = 240/70 ≈ 3.43
    """
    bars = two_week_minutes()
    days = pd.bdate_range("2024-01-01", periods=10)
    daily_volume = pd.Series([700.0] * 9 + [700.0], index=days)
    config = IntradayConfig(window_minutes=7, baseline_days=5)
    ratio = standard_volume_ratio(bars, daily_volume, config)
    assert ratio.iloc[:5].isna().all()
    assert ratio.iloc[5] == pytest.approx((70.0 / 7) / (700.0 / 240.0))
    # 最后一天窗口量 210（3 倍）-> 量比也是 3 倍
    assert ratio.iloc[-1] == pytest.approx(3 * (70.0 / 7) / (700.0 / 240.0))


def test_standard_volume_ratio_is_larger_than_the_relative_one():
    """两种口径不能混用阈值：放量日软件口径明显更大（分母是全天均量，不是开盘窗口均量）。"""
    bars = two_week_minutes()
    days = pd.bdate_range("2024-01-01", periods=10)
    daily_volume = pd.Series([700.0] * 10, index=days)
    config = IntradayConfig(window_minutes=7, baseline_days=5)
    standard = standard_volume_ratio(bars, daily_volume, config)
    relative = opening_volume_ratio(bars, config)
    assert standard.iloc[-1] > relative.iloc[-1] >= 3.0
    assert standard.iloc[-1] == pytest.approx(3.0 * 240.0 / 70.0)


def test_standard_volume_ratio_accepts_a_daily_frame_and_rejects_missing_volume():
    bars = two_week_minutes()
    days = pd.bdate_range("2024-01-01", periods=10)
    daily = pd.DataFrame({"close": 10.0, "volume": 700.0}, index=days)
    ratio = standard_volume_ratio(bars, daily, IntradayConfig(window_minutes=7, baseline_days=5))
    assert ratio.iloc[5] == pytest.approx((70.0 / 7) / (700.0 / 240.0))
    with pytest.raises(ValueError):
        standard_volume_ratio(bars, pd.DataFrame({"close": 10.0}, index=days))


def test_opening_window_price_is_the_close_of_the_last_bar_in_the_window():
    bars = minutes_for_day("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
    bars["close"] = [10.0, 11.0, 12.0, 13.0, 14.0]
    price = opening_window_price(bars, IntradayConfig(window_minutes=3))
    assert list(price.values) == [12.0]
    assert opening_window_price(bars, IntradayConfig(window_minutes=5)).iloc[0] == pytest.approx(14.0)


def test_config_validation():
    with pytest.raises(ValueError):
        IntradayConfig(window_minutes=0)
    with pytest.raises(ValueError):
        IntradayConfig(baseline_days=0)
    with pytest.raises(ValueError):
        IntradayConfig(min_ratio=0)
    with pytest.raises(ValueError):
        IntradayConfig(session_minutes=0)
    with pytest.raises(ValueError):
        opening_window_volume(pd.DataFrame({"close": [1.0]}))
    with pytest.raises(ValueError):
        opening_window_price(pd.DataFrame({"volume": [1.0]}))


# --------------------------------------------------------------------------------------
# 决策
# --------------------------------------------------------------------------------------
def test_confirm_signals_buy_watch_and_no_decision():
    bars = two_week_minutes()
    config = IntradayConfig(window_minutes=7, baseline_days=5, min_ratio=1.0)
    ratio = opening_volume_ratio(bars, config)

    days = pd.bdate_range("2024-01-01", periods=10)
    signal = pd.Series(False, index=days)
    signal.iloc[5] = True      # 次日（第 7 天）量比 1.0 -> 买入
    signal.iloc[7] = True      # 次日（第 9 天）量比 1.0 -> 买入
    signal.iloc[8] = True      # 次日（第 10 天）量比 3.0 -> 买入
    signal.iloc[9] = True      # 没有次日数据 -> 无法判断

    table = confirm_signals(signal, ratio, config)
    assert len(table) == 4
    decisions = dict(zip(table["signal_date"], table["decision"]))
    assert decisions["2024-01-08"] == "买入"
    assert decisions["2024-01-10"] == "买入"
    assert decisions["2024-01-11"] == "买入"
    assert decisions["2024-01-12"] == "无法判断"
    last = table.iloc[-1]
    assert pd.isna(last["decision_date"]) and last["volume_ratio"] != last["volume_ratio"]

    # 决策日必须是信号日的下一个交易日（无未来函数）
    for _, row in table.iloc[:-1].iterrows():
        assert pd.Timestamp(row["decision_date"]) > pd.Timestamp(row["signal_date"])


def test_confirm_signals_watch_when_ratio_below_threshold():
    bars = two_week_minutes()
    config = IntradayConfig(window_minutes=7, baseline_days=5, min_ratio=2.0)   # 阈值抬到 2 倍
    ratio = opening_volume_ratio(bars, config)
    days = pd.bdate_range("2024-01-01", periods=10)
    signal = pd.Series(False, index=days)
    signal.iloc[5] = True       # 次日量比 1.0 < 2.0 -> 观望
    signal.iloc[8] = True       # 次日量比 3.0 >= 2.0 -> 买入

    table = confirm_signals(signal, ratio, config)
    assert table.iloc[0]["decision"] == "观望"
    assert "无量能确认" in table.iloc[0]["reason"]
    assert table.iloc[1]["decision"] == "买入"


def test_confirm_signals_handles_missing_history():
    bars = two_week_minutes()
    config = IntradayConfig(window_minutes=7, baseline_days=5, min_ratio=1.0)
    ratio = opening_volume_ratio(bars, config)
    days = pd.bdate_range("2024-01-01", periods=10)
    signal = pd.Series(False, index=days)
    signal.iloc[0] = True       # 次日只有 1 天历史，量比 NaN -> 无法判断
    table = confirm_signals(signal, ratio, config)
    assert table.iloc[0]["decision"] == "无法判断"
    assert "窗口未满" in table.iloc[0]["reason"]


def test_confirm_signals_empty_input():
    empty = confirm_signals(pd.Series(dtype=bool), pd.Series(dtype=float))
    assert empty.empty


def test_confirmed_signal_series_maps_to_decision_day():
    bars = two_week_minutes()
    config = IntradayConfig(window_minutes=7, baseline_days=5, min_ratio=1.0)
    ratio = opening_volume_ratio(bars, config)
    days = pd.bdate_range("2024-01-01", periods=10)
    signal = pd.Series(False, index=days)
    signal.iloc[5] = True       # 次日 = 第 7 个交易日（2024-01-09）
    confirmed = confirmed_signal_series(signal, ratio, config)
    assert confirmed.index.equals(pd.DatetimeIndex(ratio.index))
    assert bool(confirmed.loc[pd.Timestamp("2024-01-09")]) is True
    assert int(confirmed.sum()) == 1


# --------------------------------------------------------------------------------------
# 合成分钟数据
# --------------------------------------------------------------------------------------
def test_generate_synthetic_minutes_keeps_daily_volume():
    daily = generate_synthetic_ohlcv(n_days=12, seed=3)
    bars = generate_synthetic_minutes(daily, seed=5)
    per_day = bars.groupby(bars.index.normalize())["volume"].sum()
    assert len(per_day) == len(daily)
    assert np.allclose(per_day.to_numpy(), daily["volume"].reindex(per_day.index).to_numpy(), rtol=1e-9)
    assert len(bars) == len(daily) * 240

    again = generate_synthetic_minutes(daily, seed=5)
    pd.testing.assert_frame_equal(bars, again)

    boosted = generate_synthetic_minutes(daily, seed=5, first_minutes_boost=3.0)
    cfg = IntradayConfig(window_minutes=7, baseline_days=2)
    share_plain = opening_window_volume(bars, cfg) / daily["volume"].reindex(opening_window_volume(bars, cfg).index)
    share_boost = opening_window_volume(boosted, cfg) / daily["volume"].reindex(opening_window_volume(boosted, cfg).index)
    # 开盘窗口量占比被抬高（量比本身会自我归一化，所以比较占比而不是比均值）
    assert share_boost.mean() > share_plain.mean()


def test_minute_bars_can_come_as_a_column_instead_of_index():
    """RangeIndex（忘记 set_index）必须被拦住或自动纠正，否则会被当成 1970 年静默失效。"""
    bars = minutes_for_day("2024-01-02", [10.0] * 20).reset_index().rename(columns={"index": "minute"})
    volume = opening_window_volume(bars, IntradayConfig(window_minutes=7))
    assert volume.iloc[0] == pytest.approx(70.0)

    bad = pd.DataFrame({"close": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError):
        opening_window_volume(bad, IntradayConfig(window_minutes=7))
