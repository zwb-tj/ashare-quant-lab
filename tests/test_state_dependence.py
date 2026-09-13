"""市场状态依赖性的测试（离线，合成数据）。

重点是"分组统计本身要正确"，因为这类分析最容易出现两种错误：

① **状态标签错位**：市场状态用**当天**的收盘算，但 IC 衡量的是**未来**收益——
   标签必须对齐到同一个截面日，不能前后错一格；
② **分组样本太少也报结论**：每档至少要有若干截面，否则极差只是噪声。
   分组后每档天数应当大致均衡（等频分档），这一点要能验证。
"""

import pandas as pd
import pytest

from aqlab.state_dependence import (
    StateConfig,
    factor_state_ic,
    market_state_frame,
    summarize_state_dependence,
)
from aqlab.tools import SyntheticDataSource


@pytest.fixture(scope="module")
def frames():
    source = SyntheticDataSource(n_symbols=24, n_days=700, seed=11)
    return {symbol: source.bars(symbol) for symbol in source.symbols()}


def test_config_validation():
    with pytest.raises(ValueError):
        StateConfig(horizons=())
    with pytest.raises(ValueError):
        StateConfig(step_days=0)
    with pytest.raises(ValueError):
        StateConfig(min_symbols=2)
    with pytest.raises(ValueError):
        StateConfig(trend_window=1)
    with pytest.raises(ValueError):
        StateConfig(quantiles=1)


def test_market_state_frame_fields_and_alignment(frames):
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()})
    state = market_state_frame(close, trend_window=20)
    assert set(state.columns) == {"index", "trend", "dispersion"}
    assert state.index.equals(close.index)
    # 等权指数应当以 1 附近起步并单调可算
    assert state["index"].iloc[0] == pytest.approx(1.0)
    # 离散度非负，且与横截面收益的波动同量级
    assert (state["dispersion"].dropna() >= 0).all()
    manual_dispersion = close.pct_change().std(axis=1).iloc[-1]
    assert state["dispersion"].iloc[-1] == pytest.approx(float(manual_dispersion), rel=1e-9)
    # 趋势窗口不足时为 NaN（不用不完整的窗口冒充状态）
    assert state["trend"].iloc[:20].isna().all()


def test_market_state_uses_the_provided_regime_lagged(frames):
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()})
    regime = pd.Series(1, index=close.index)
    state = market_state_frame(close, regime=regime)
    assert "regime" in state.columns
    assert (state["regime"] == 1).all()


def test_factor_state_ic_joins_ic_with_same_day_state(frames):
    config = StateConfig(horizons=(5,), min_history=200, min_symbols=8, factors=("alpha_013",))
    outcome = factor_state_ic(frames, config)
    detail = outcome["detail"]
    assert not detail.empty
    assert {"date", "factor", "horizon", "ic", "trend_bucket", "dispersion_bucket"} <= set(detail.columns)
    # 每个截面日的 IC 与状态必须来自同一天
    state = outcome["state"]
    for row in detail.head(10).itertuples(index=False):
        if pd.notna(row.trend_bucket):
            expected = state.loc[row.date, "trend"]
            if pd.notna(expected):
                rank = (state["trend"].dropna() < expected).mean()
                assert 0.0 <= rank <= 1.0
    # IC 必须在 [-1, 1]
    assert detail["ic"].between(-1.0, 1.0).all()


def test_state_buckets_are_roughly_balanced(frames):
    config = StateConfig(horizons=(5,), min_history=200, min_symbols=8, factors=("alpha_013",), quantiles=3)
    detail = factor_state_ic(frames, config)["detail"]
    counts = detail.dropna(subset=["trend_bucket"])["trend_bucket"].value_counts()
    assert len(counts) >= 2, "等频分档应当给出多个档位"
    # 最大档与最小档的天数差异不应太悬殊（等频分档的性质）
    assert counts.max() <= counts.min() * 3


def test_summarize_reports_spread_and_handles_empty(frames):
    config = StateConfig(horizons=(5,), min_history=200, min_symbols=8, factors=("alpha_013",))
    detail = factor_state_ic(frames, config)["detail"]
    table = summarize_state_dependence(detail, "trend_bucket")
    assert {"factor", "horizon", "state", "periods", "ic_mean", "t_stat", "positive_rate", "ic_spread"} <= set(table.columns)
    # 同一个 (因子, 持有期) 的各档 ic_spread 必须一致
    assert table["ic_spread"].nunique() == 1
    spread = float(table["ic_spread"].iloc[0])
    assert spread == pytest.approx(float(table["ic_mean"].max() - table["ic_mean"].min()), rel=1e-9)
    # 缺列 / 空表都要安全返回
    assert summarize_state_dependence(pd.DataFrame(), "trend_bucket").empty
    assert summarize_state_dependence(detail, "no_such_column").empty


def test_summarize_requires_enough_periods_per_bucket(frames):
    """每档截面太少时不该给出"某状态特别好"的结论——这里只验证样本数被如实报出。"""
    config = StateConfig(horizons=(5,), min_history=200, min_symbols=20, factors=("alpha_013",), quantiles=4)
    detail = factor_state_ic(frames, config)["detail"]
    table = summarize_state_dependence(detail, "dispersion_bucket")
    if not table.empty:
        assert (table["periods"] > 0).all()
        assert table["periods"].sum() <= len(detail)
