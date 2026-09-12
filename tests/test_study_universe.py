"""全市场 B1 研究的测试（离线，合成数据）。

重点：① 入场是 T+1 开盘价（不能用信号日收盘，否则偷看未来）；② 超额相对全市场等权指数；
③ 大盘阶段门只挡买入、且用"昨天收盘已知的状态"判定。
"""

import numpy as np
import pandas as pd
import pytest

from aqlab.exits import ExitConfig
from aqlab.study_universe import (
    UniverseStudyConfig,
    equal_weight_index,
    study_universe,
    summarize_trades,
)


class StubRule:
    """在指定日期打信号（规则接口只拿到行情帧，所以按日期标记）。"""

    def __init__(self, dates):
        self.dates = {pd.Timestamp(date) for date in dates}

    def signal(self, frame):
        return pd.Series(frame.index.isin(self.dates), index=frame.index)


class StubGate:
    def __init__(self, values):
        self.values = values

    def gate_series(self, frames):
        index = next(iter(frames.values())).index
        return pd.DataFrame({"gate": self.values, "trigger": "none"}, index=index)


def frame_from(closes, start="2024-01-01"):
    close = pd.Series(closes, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close).to_numpy(),
            "high": (close * 1.02).to_numpy(),
            "low": (close * 0.98).to_numpy(),
            "close": close.to_numpy(),
            "volume": 1_000_000.0,
        },
        index=index,
    )


def quiet_exit():
    return ExitConfig(mode="fixed", take_profit_pct=None, use_death_cross=False, use_white_break=False, didi_mode="off")


def test_equal_weight_index_averages_daily_returns():
    first = frame_from([100.0, 110.0, 121.0])
    second = frame_from([100.0, 90.0, 99.0])
    index = equal_weight_index([first, second])
    # 第 2 日：+10% 与 -10% 等权 -> 0%；第 3 日：+10% 与 +10% -> +10%
    assert len(index) == 2                      # 第一天没有涨跌幅，无法算均值 -> 丢弃`n    assert index.iloc[0] == pytest.approx(1.0)   # (+10% -10%) / 2 = 0%`n    assert index.iloc[1] == pytest.approx(1.10)  # (+10% +10%) / 2 = +10%


def test_study_uses_next_open_entry_and_reports_excess():
    frame = frame_from([100.0, 100.0, 110.0, 121.0, 133.1, 146.41])
    signal_date = frame.index[1]
    config = UniverseStudyConfig(exit=quiet_exit())
    table, meta = study_universe(
        {"600000": frame}, config, rule=StubRule([signal_date])
    )
    assert meta["trades"] == 1
    row = table.iloc[0]
    assert row["entry_date"] == frame.index[2]
    assert row["entry_price"] == pytest.approx(float(frame["open"].iloc[2]))     # T+1 开盘
    assert row["hold_1"] == pytest.approx(float(frame["close"].iloc[2]) / row["entry_price"] - 1)
    assert row["hold_3"] == pytest.approx(float(frame["close"].iloc[4]) / row["entry_price"] - 1)
    # 单票组合时等权指数就是它自己 -> 超额≈0（进场当日开盘 vs 前一日收盘有微小差异）
    assert abs(row["excess_pct"]) < 0.02


def test_regime_gate_blocks_entries_and_counts_them():
    frame = frame_from([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    rule = StubRule([frame.index[1]])
    config = UniverseStudyConfig(exit=quiet_exit(), use_regime_gate=True)

    closed = StubGate([0] * len(frame))
    table, meta = study_universe({"600000": frame}, config, gate=closed, rule=rule)
    assert table.empty and meta["gated_out"] == 1

    opened = StubGate([1] * len(frame))
    table_open, meta_open = study_universe({"600000": frame}, config, gate=opened, rule=rule)
    assert meta_open["gated_out"] == 0 and len(table_open) == 1


def test_regime_gate_uses_yesterdays_state():
    """门在第 3 天变成"关"，但用它判定第 3 天买入时必须还看到第 2 天的"开"。"""
    frame = frame_from([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    rule = StubRule([frame.index[1]])       # T+1 = index[2] 买入
    gate = StubGate([1, 1, 0, 0, 0, 0])
    table, meta = study_universe(
        {"600000": frame}, UniverseStudyConfig(exit=quiet_exit(), use_regime_gate=True), gate=gate, rule=rule
    )
    assert len(table) == 1 and meta["gated_out"] == 0


def test_start_end_filter_on_signal_date():
    frame = frame_from([100.0 + i for i in range(20)])
    rule = StubRule([frame.index[2], frame.index[12]])
    config = UniverseStudyConfig(start=str(frame.index[5].date()), end=str(frame.index[15].date()), exit=quiet_exit())
    table, _ = study_universe({"600000": frame}, config, rule=rule)
    assert len(table) == 1
    assert pd.Timestamp(table.iloc[0]["signal_date"]) == frame.index[12]


def test_summarize_trades_reports_core_columns():
    frame = frame_from([100.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    rule = StubRule([frame.index[1], frame.index[3]])
    table, _ = study_universe({"600000": frame}, UniverseStudyConfig(exit=quiet_exit()), rule=rule)
    assert len(table) == 2
    summary = summarize_trades(table)
    row = summary.iloc[0]
    assert row["笔数"] == 2
    for column in ("平均收益%", "胜率%", "平均超额%", "超额t", "平均持有"):
        assert column in summary.columns
    assert summarize_trades(pd.DataFrame()).empty


def test_missing_future_bars_are_skipped_not_faked():
    frame = frame_from([100.0, 101.0, 102.0])
    rule = StubRule([frame.index[-1]])      # 信号在最后一根 -> 没有 T+1
    table, meta = study_universe({"600000": frame}, UniverseStudyConfig(exit=quiet_exit()), rule=rule)
    assert table.empty and meta["trades"] == 0
