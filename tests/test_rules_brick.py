"""砖型图（自定义公式）与 B1 梯度打分的测试。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.indicators_extra import brick_chart, brick_streaks, kdj, sma_tdx
from aqlab.profiles import list_profiles, load_profile
from aqlab.rules import build_rule
from aqlab.rules_zgnb import B1Graded, BrickGreenToRed, brick_filter_mask


def frame(closes, volumes=None, high_mult=1.004, low_mult=0.996):
    c = pd.Series(closes, dtype=float)
    v = pd.Series(1000.0, index=range(len(c))) if volumes is None else pd.Series(volumes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    return pd.DataFrame(
        {"open": c.shift(1).fillna(c).values, "high": (c * high_mult).values, "low": (c * low_mult).values,
         "close": c.values, "volume": v.values}, index=idx)


# --------------------------------------------------------------------------------------
# 通达信 SMA 与砖型图公式
# --------------------------------------------------------------------------------------
def test_sma_tdx_matches_manual_recursion():
    out = sma_tdx(pd.Series([1.0, 2.0, 3.0, 4.0]), 4, 1)
    assert out.tolist() == pytest.approx([1.0, 1.25, 1.6875, 2.265625])
    with pytest.raises(ValueError):
        sma_tdx(pd.Series([1.0]), 4, 5)


def test_brick_chart_invariants():
    closes = [100.0 * 1.01 ** i for i in range(40)] + [100.0 * 1.01 ** 39 * 0.985 ** i for i in range(1, 10)]
    chart = brick_chart(frame(closes))

    # 砖型图 = max(VAR6A - 4, 0)
    above = chart["var6a"] > 4
    assert (chart["brick"] >= 0).all()
    assert np.allclose(chart.loc[~above, "brick"], 0.0)
    assert np.allclose(chart.loc[above, "brick"], chart.loc[above, "var6a"] - 4.0)

    # 红柱/绿柱互斥，且分别对应砖型图的上升/下降
    assert not ((chart["red"] > 0) & (chart["green"] > 0)).any()
    valid = chart["prev_brick"].notna()
    assert ((chart.loc[valid, "red"] > 0) == (chart.loc[valid, "brick"] > chart.loc[valid, "prev_brick"])).all()
    assert ((chart.loc[valid, "green"] > 0) == (chart.loc[valid, "brick"] < chart.loc[valid, "prev_brick"])).all()

    # 绿转红 / 强度比 / XG 的定义
    assert (chart["green_to_red"] == ((chart["prev_green"] > 0) & (chart["red"] > 0))).all()
    expected_xg = chart["green_to_red"] & (chart["red"] >= chart["prev_green"] * 0.6667)
    assert (chart["xg"] == expected_xg).all()
    in_signal = chart["green_to_red"]
    assert np.allclose(chart.loc[in_signal, "strength_ratio"], (chart.loc[in_signal, "red"] / chart.loc[in_signal, "prev_green"]).round(2))
    assert (chart.loc[~in_signal, "strength_ratio"] == 0).all()

    # 无未来函数：把最后一根之后的数据截掉，前面的砖型图不发生变化
    truncated = brick_chart(frame(closes[:-3]))
    assert np.allclose(truncated["brick"].to_numpy(), chart["brick"].to_numpy()[:-3])


def test_brick_green_to_red_signal_and_graded_score():
    closes = [100.0 * 0.96 ** i for i in range(5)]
    base = closes[-1]
    closes += [base * 1.05 ** k for k in range(1, 4)]
    df = frame(closes)
    chart = brick_chart(df)
    assert bool(chart["xg"].any()) is True

    rule = BrickGreenToRed()
    hit_index = chart.index[chart["xg"]][-1]
    assert bool(rule.signal(df).loc[hit_index]) is True
    assert rule.score(df).loc[hit_index] == pytest.approx(1.0)
    # 强度比不足阈值时给梯度分数（0 < score < 1），没有绿转红则为 0
    ratio = float(chart.loc[hit_index, "strength_ratio"])
    assert ratio >= 0.6667
    assert rule.score(df).iloc[0] == 0.0


def test_brick_filter_mask_gates_late_red_bricks():
    df = frame([100.0 * 1.01 ** i for i in range(40)])
    chart = brick_streaks(brick_chart(df))
    mask = brick_filter_mask(df, entry_max=2, block=4)

    expected = (chart["red_streak"] <= 2) & (chart["red_streak"] < 4)
    assert (mask == expected).all()
    assert bool(mask.any()) is True                       # 早期红砖允许
    assert bool(mask[chart["red_streak"] >= 4].any()) is False  # 门 2：红砖 >= 4 禁买

    with pytest.raises(ValueError):
        brick_filter_mask(df, entry_max=0)


# --------------------------------------------------------------------------------------
# B1 梯度打分（5 硬 + 4 软）
# --------------------------------------------------------------------------------------
def b1_graded_frame():
    """B1Graded 五条硬性条件全部满足的行情（用参数搜索定位）。"""
    closes = [100.0 * 1.002 ** i for i in range(120)]
    base = closes[-1]
    closes += [base * 0.985 ** k for k in range(1, 5)]
    volumes = [1000.0] * len(closes)
    volumes[-4] = 2500.0   # 近 15 日放量日（硬 2）
    volumes[-1] = 300.0    # 极致缩量（硬 3）
    return frame(closes, volumes)


def test_b1_graded_scores_60_plus_soft_bonus():
    df = b1_graded_frame()
    rule = B1Graded()
    assert bool(rule.signal(df).iloc[-1]) is True

    detail = rule.detail(df)
    assert detail["hard_all"] is True
    assert detail["soft_pass"] >= 1
    expected = (60 + 10 * detail["soft_pass"]) / 100.0
    assert rule.score(df).iloc[-1] == pytest.approx(expected)
    assert 0.6 <= rule.score(df).iloc[-1] <= 0.9
    # J 阈值是 13（不是早期版本的 -10）
    assert kdj(df)["j"].iloc[-1] <= 13
    assert rule.params["j_max"] == 13.0


def test_b1_graded_requires_volume_spike_and_shrink():
    no_spike = b1_graded_frame()
    no_spike.loc[no_spike.index[-4], "volume"] = 1000.0
    assert bool(B1Graded().signal(no_spike).iloc[-1]) is False
    assert B1Graded().score(no_spike).iloc[-1] == 0.0

    no_shrink = b1_graded_frame()
    no_shrink.loc[no_shrink.index[-1], "volume"] = 5000.0
    assert bool(B1Graded().signal(no_shrink).iloc[-1]) is False


def test_b1_graded_rejects_overbought_and_short_history():
    rising = frame([100.0 * 1.01 ** i for i in range(130)])
    assert B1Graded().score(rising).iloc[-1] == 0.0        # J 远高于 13
    short = frame([100.0 * 1.002 ** i for i in range(60)])
    assert bool(B1Graded().signal(short).iloc[-1]) is False  # 双线需要 >= 114 根


def test_b1_graded_s1_exclusion_blocks_recent_bearish_volume():
    df = b1_graded_frame()
    idx = df.index[-3]
    df.loc[idx, "close"] = df["close"].shift(1).loc[idx] * 0.95   # -5%
    df.loc[idx, "volume"] = df["volume"].shift(1).loc[idx] * 2.0  # 放量
    detail = B1Graded().detail(df)
    assert detail["hard"]["s1_excluded"] is False
    assert B1Graded().score(df).iloc[-1] == 0.0


def test_b1_graded_brick_filter_blocks_late_red_bricks():
    df = frame([100.0 * 1.01 ** i for i in range(130)])
    assert B1Graded(use_brick_filter=True).score(df).iloc[-1] == 0.0


# --------------------------------------------------------------------------------------
# 档案注册
# --------------------------------------------------------------------------------------
def test_profiles_expose_the_graded_and_brick_rules():
    names = list_profiles()
    for name in ("b1", "b1_simple", "b1_brick", "brick_green_to_red", "zgnb_brick"):
        assert name in names
    assert load_profile("b1")[0][0] == "b1_graded"
    assert load_profile("b1_brick")[0][1]["use_brick_filter"] is True
    assert isinstance(build_rule("b1_graded"), B1Graded)
    assert isinstance(build_rule("brick_green_to_red"), BrickGreenToRed)
