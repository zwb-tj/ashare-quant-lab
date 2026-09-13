"""量比确认评估的测试（离线，合成数据）。

重点守住两件事：
1. **入场时点必须在决策之后**——9:37 看到量比才下单，信号日收盘到决策日窗口之间的涨跌不能算进来；
2. 基准与超额只在配对子集上统计，且 ``mean - bench = excess`` 自洽。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.confirm_eval import (
    ConfirmEvalConfig,
    attach_benchmark,
    evaluate_confirmation,
    summarize_confirmation,
)
from aqlab.intraday import IntradayConfig, opening_window_price


def daily_frame(closes, start="2026-06-01"):
    index = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {"open": close.values, "high": close.values, "low": close.values, "close": close.values, "volume": 1000.0},
        index=index,
    )


def minute_frame(dates, window_closes, per_day=6, volume=100.0):
    rows = []
    for date, price in zip(dates, window_closes, strict=False):
        base = pd.Timestamp(date) + pd.Timedelta(minutes=9 * 60 + 30)
        for i in range(per_day):
            rows.append({"minute": base + pd.Timedelta(minutes=i), "close": float(price), "volume": volume})
    return pd.DataFrame(rows).set_index("minute")


def test_opening_window_price_takes_the_last_bar_inside_the_window():
    dates = pd.bdate_range("2026-06-01", periods=2)
    bars = []
    for day, prices in zip(dates, ([10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]), strict=False):
        base = pd.Timestamp(day) + pd.Timedelta(minutes=9 * 60 + 30)
        for i, price in enumerate(prices):
            bars.append({"minute": base + pd.Timedelta(minutes=i), "close": price, "volume": 1.0})
    frame = pd.DataFrame(bars).set_index("minute")
    price = opening_window_price(frame, IntradayConfig(window_minutes=3))
    assert list(price.values) == [12.0, 22.0]           # 窗口内最后一根，不是收盘价
    with pytest.raises(ValueError):
        opening_window_price(frame.drop(columns="close"))


def test_returns_start_at_the_decision_not_at_the_signal():
    # 信号日收盘 100 -> 决策日窗口价 190（跳空）-> 决策日收盘 200 -> 次日 202
    daily = daily_frame([100.0, 200.0, 202.0])
    minute = minute_frame(daily.index[:2], [100.0, 190.0])
    signal = pd.Series(False, index=daily.index)
    signal.iloc[0] = True

    base = {"window_minutes": 3, "baseline_days": 1, "min_ratio": 1.0, "horizons": (1, 2)}
    window_close, _ = evaluate_confirmation(daily, minute, signal, ConfirmEvalConfig(entry="window_close", **base), symbol="X")
    row = window_close.iloc[0]
    assert row["decision"] == "买入"                      # 量比 = 100/100 = 1.0
    assert row["entry_date"] == str(daily.index[1].date())  # 决策日
    assert row["entry_price"] == pytest.approx(190.0)      # 窗口收盘价，不是信号日收盘
    assert row["fwd_1"] == pytest.approx(200.0 / 190.0 - 1)   # 跳空那 90 点没算进来
    assert row["fwd_2"] == pytest.approx(202.0 / 190.0 - 1)

    signal_close, _ = evaluate_confirmation(daily, minute, signal, ConfirmEvalConfig(entry="signal_close", **base))
    old = signal_close.iloc[0]
    assert old["entry_date"] == str(daily.index[0].date())
    assert old["entry_price"] == pytest.approx(100.0)
    assert old["fwd_2"] == pytest.approx(200.0 / 100.0 - 1)   # 老口径把信号日之后的跳空也算进收益


def test_decision_close_entry_uses_the_decision_day_close():
    daily = daily_frame([100.0, 200.0, 202.0])
    minute = minute_frame(daily.index[:2], [100.0, 190.0])
    signal = pd.Series(False, index=daily.index)
    signal.iloc[0] = True
    details, _ = evaluate_confirmation(
        daily, minute, signal,
        ConfirmEvalConfig(window_minutes=3, baseline_days=1, horizons=(1, 2), entry="decision_close"),
    )
    row = details.iloc[0]
    assert row["entry_price"] == pytest.approx(200.0)     # 决策日收盘价
    assert row["fwd_1"] == pytest.approx(0.0)             # 买入当日收盘卖出 = 0（口径与 picks 的 next_close 一致）
    assert row["fwd_2"] == pytest.approx(202.0 / 200.0 - 1)


def test_missing_window_data_is_undecidable_not_a_group_member():
    daily = daily_frame([100.0, 200.0, 202.0])
    minute = minute_frame(daily.index[:1], [100.0])        # 决策日没有分钟数据
    signal = pd.Series(False, index=daily.index)
    signal.iloc[0] = True
    details, _ = evaluate_confirmation(daily, minute, signal, ConfirmEvalConfig(entry="window_close"))
    assert details.iloc[0]["decision"] == "无法判断"
    summary = summarize_confirmation(details, ConfirmEvalConfig(entry="window_close"))
    assert set(summary["decision"]) == {"无法判断", "全部"}
    assert summary[summary["decision"] == "无法判断"].iloc[0]["signals"] == 1


def test_summarize_confirmation_uses_paired_subset_for_benchmark():
    details = pd.DataFrame(
        [
            {"decision": "买入", "fwd_1": 0.10, "bench_1": 0.02, "excess_1": 0.08},
            {"decision": "买入", "fwd_1": -0.10, "bench_1": -0.02, "excess_1": -0.08},
            {"decision": "买入", "fwd_1": 0.50, "bench_1": np.nan, "excess_1": np.nan},   # 无基准 -> 不进基准均值
        ]
    )
    config = ConfirmEvalConfig(horizons=(1,))
    summary = summarize_confirmation(details, config)
    row = summary[summary["decision"] == "买入"].iloc[0]
    assert row["n_1"] == 3
    assert row["mean_1"] == pytest.approx(0.50 / 3)
    assert row["bench_1"] == pytest.approx(0.0)
    assert row["excess_1"] == pytest.approx(0.0)


def test_attach_benchmark_adds_nan_when_horizon_is_missing():
    details = pd.DataFrame([{"decision": "买入", "entry_date": "2026-06-02", "fwd_1": 0.05, "fwd_3": 0.10}])
    bench = pd.DataFrame([{"entry_date": "2026-06-02", "bench_1": 0.01, "benchn_1": 7}])
    merged = attach_benchmark(details, bench)
    assert merged.iloc[0]["excess_1"] == pytest.approx(0.04)
    assert np.isnan(merged.iloc[0]["excess_3"])            # 缺 bench_3 -> NaN，不当 0


def test_config_rejects_unknown_entry():
    with pytest.raises(ValueError):
        ConfirmEvalConfig(entry="next_open")
    with pytest.raises(ValueError):
        ConfirmEvalConfig(horizons=())
