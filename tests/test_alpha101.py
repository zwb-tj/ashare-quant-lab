"""公式化 alpha 因子的测试（离线，合成数据）。

守住三件事：

① **算子方向不能搞反**：时序算子在列方向（每只标的一条序列）、截面算子在行方向（每个交易日一次）。
   这是因子实现里最常见的错误，用能解析算出结果的构造数据来验证。
② **因子与面板对齐**：形状、索引、列都要一致，且不得出现 ±inf（除零要么 NaN 要么被兜住）。
③ **不能靠近似值冒充**：依赖行业/市值数据的因子必须在 `SKIPPED` 里写明原因。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.alpha101 import (
    ALPHAS,
    SKIPPED,
    build_panel,
    compute_alphas,
    cross_rank,
    decay_linear,
    delta,
    scale,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_rank,
    ts_std,
)


def _frame(closes, volume=None, start="2022-01-03"):
    close = pd.Series(closes, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    volume = pd.Series(volume, dtype=float) if volume is not None else pd.Series(1_000_000.0, index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close).to_numpy(),
            "high": (close * 1.01).to_numpy(),
            "low": (close * 0.99).to_numpy(),
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
            "amount": (close.to_numpy() * volume.to_numpy()),
        },
        index=index,
    )


def _panel(symbols_closes):
    return build_panel({name: _frame(closes) for name, closes in symbols_closes.items()})


# --------------------------------------------------------------------------------------
# 算子
# --------------------------------------------------------------------------------------
def test_cross_rank_is_per_row_and_pct():
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], index=["d1", "d2"], columns=["a", "b", "c"])
    ranked = cross_rank(frame)
    assert ranked.loc["d1", "c"] == pytest.approx(1.0)      # 当日最大的排到 1
    assert ranked.loc["d1", "a"] == pytest.approx(1 / 3)
    assert ranked.loc["d2", "a"] == pytest.approx(1.0)      # 换一天，排名反过来
    # 全 NaN 的行保持 NaN，不该被当成 0 参与排名
    frame.loc["d3"] = np.nan
    assert cross_rank(frame).loc["d3"].isna().all()


def test_ts_operators_work_down_the_column_not_across_it():
    """时序算子必须跨时间（行方向），不能把不同标的混在一起。"""
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    summed = frame.rolling(2, min_periods=2).sum()
    assert summed.iloc[-1]["a"] == pytest.approx(7.0)      # 3+4
    assert summed.iloc[-1]["b"] == pytest.approx(70.0)     # 30+40，没串味
    assert delta(frame, 1).iloc[-1]["a"] == pytest.approx(1.0)


def test_ts_argmax_counts_from_the_most_recent_bar():
    """ts_argmax 返回"距最新一根的位置"：1 = 最新，window = 最早。"""
    frame = pd.DataFrame({"a": [5.0, 3.0, 9.0, 1.0]}, index=pd.bdate_range("2024-01-01", periods=4))
    assert ts_argmax(frame, 2).iloc[-1]["a"] == pytest.approx(2.0)    # 窗口 [9,1]，最大在较早位置
    assert ts_argmax(frame, 2).iloc[-2]["a"] == pytest.approx(1.0)    # 窗口 [3,9]，最大是最新一根


def test_ts_rank_and_decay_linear_are_bounded():
    frame = pd.DataFrame({"a": np.arange(1.0, 21.0)}, index=pd.bdate_range("2024-01-01", periods=20))
    ranked = ts_rank(frame, 10)
    assert ((ranked.dropna() >= 0) & (ranked.dropna() <= 1)).all().all()
    assert ranked.iloc[-1]["a"] == pytest.approx(1.0)      # 单调上升序列的最新值必是窗口最大
    decayed = decay_linear(frame, 5).dropna()["a"]
    # 只比较窗口已满的部分（前几根 min_periods 允许数据不足，窗口边界不是真实的 5 根）
    full = decayed.iloc[4:]
    window_min = frame["a"].rolling(5).min().reindex(full.index)
    window_max = frame["a"].rolling(5).max().reindex(full.index)
    assert ((full >= window_min) & (full <= window_max)).all()
    latest = frame["a"].iloc[-5:].to_numpy()
    weights = np.arange(1.0, 6.0)
    expected = float(np.dot(latest, weights / weights.sum()))
    assert full.iloc[-1] == pytest.approx(expected)


def test_ts_std_and_corr_handle_constant_series_without_inf():
    frame = pd.DataFrame(
        {"flat": [2.0] * 10, "trend": np.arange(10, dtype=float)},
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    assert (ts_std(frame, 5).iloc[-1]["flat"] == 0) or np.isnan(ts_std(frame, 5).iloc[-1]["flat"])
    corr = ts_corr(frame["trend"], frame["flat"], 5)
    assert not np.isinf(corr.to_numpy()).any(), "常量序列的相关性要么 NaN 要么 0，不能是 inf"


def test_scale_and_signed_power_behave():
    frame = pd.DataFrame({"a": [1.0, -3.0], "b": [1.0, 1.0]}, index=["d1", "d2"])
    scaled = scale(frame)
    assert scaled.loc["d1"].abs().sum() == pytest.approx(1.0)
    assert scaled.loc["d2"].abs().sum() == pytest.approx(1.0)
    assert np.isnan(scale(pd.DataFrame({"a": [0.0, 0.0]})).iloc[0]["a"]), "全零行缩放后应是 NaN 而不是 0/0"
    powered = signed_power(pd.DataFrame({"a": [-4.0, 4.0]}), 0.5)
    assert powered.iloc[0]["a"] == pytest.approx(-2.0) and powered.iloc[1]["a"] == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# 面板与因子
# --------------------------------------------------------------------------------------
def test_build_panel_derives_vwap_from_amount_and_volume():
    frames = {"X": _frame([10.0, 11.0, 12.0], volume=[100.0, 200.0, 300.0])}
    panel = build_panel(frames)
    assert "vwap" in panel
    expected = 10.0 * 100.0 / 100.0
    assert float(panel["vwap"].iloc[0]["X"]) == pytest.approx(expected)
    assert panel.close.shape == (3, 1)


def test_build_panel_rejects_empty_input():
    with pytest.raises(ValueError):
        build_panel({})
    with pytest.raises(ValueError):
        build_panel({"X": pd.DataFrame()})


def test_every_alpha_returns_an_aligned_panel():
    rng = np.random.default_rng(7)
    panel = _panel({f"S{i}": 100 * np.cumprod(1 + rng.normal(0, 0.02, 320)) for i in range(6)})
    values = compute_alphas(panel, min_history=120)
    assert values, "至少要算出若干因子"
    for name, frame in values.items():
        assert frame.shape == panel.close.shape, f"{name} 形状应与面板一致"
        assert frame.index.equals(panel.close.index)
        assert list(frame.columns) == panel.symbols
        finite_or_nan = frame.to_numpy(dtype=float)
        assert not np.isinf(finite_or_nan).any(), f"{name} 出现 inf"
    # min_history 之前必须是 NaN（避免用不充分的窗口）
    early = {name: frame.iloc[:120].isna().all().all() for name, frame in values.items()}
    assert all(early.values()), "min_history 之内的取值应被置为 NaN"


def test_alphas_are_lagged_free_of_infinity_on_degenerate_input():
    """价格全同、成交量为 0 这类退化输入不能产生 inf 或崩溃。"""
    flat = {f"S{i}": _frame([10.0] * 300, volume=[0.0] * 300) for i in range(3)}
    panel = build_panel(flat)
    values = compute_alphas(panel, min_history=0)
    for name, frame in values.items():
        assert not np.isinf(frame.to_numpy(dtype=float)).any(), f"{name} 在退化输入下出现 inf"


def test_skipped_factors_document_their_missing_input():
    """没实现的因子必须写清原因（缺行业/市值等），不能悄悄省略。"""
    assert SKIPPED, "应当显式列出被跳过的因子"
    for name, reason in SKIPPED.items():
        assert name.startswith("alpha_")
        assert reason, f"{name} 缺少说明"
        assert name not in ALPHAS, f"{name} 既然跳过了，就不该出现在实现里"
    joined = " ".join(SKIPPED.values())
    assert "行业" in joined or "市值" in joined, "跳过原因里应当点明缺的是哪类数据"


def test_alpha_101_matches_its_definition():
    frame = _frame([10.0, 11.0, 12.0])
    frame["open"] = [10.0, 10.5, 11.5]
    frame["high"] = [10.2, 11.2, 12.2]
    frame["low"] = [9.8, 10.3, 11.3]
    value = ALPHAS["alpha_101"](build_panel({"X": frame}))
    expected = (12.0 - 11.5) / ((12.2 - 11.3) + 0.001)
    assert float(value.iloc[-1]["X"]) == pytest.approx(expected, rel=1e-9)
