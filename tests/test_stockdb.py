"""stockdb 数据源与量比确认评估的测试（离线，用假 rd / 合成数据）。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.confirm_eval import ConfirmEvalConfig, evaluate_confirmation, summarize_confirmation
from aqlab.data import generate_synthetic_ohlcv
from aqlab.stockdb import StockDbDataSource, export_sample, fetch_daily, fetch_minute


class FakeRd:
    """最小桩：只实现 get_data，返回 stockdb 风格的 list[dict]。"""

    def __init__(self, daily_days=40, minute_volume=100.0):
        self.calls: list[tuple] = []
        self.daily_days = daily_days
        self.minute_volume = minute_volume

    def get_data(self, symbol, start=None, end=None, frequency="1d", fq="qfq"):
        self.calls.append((symbol, start, end, frequency, fq))
        if frequency == "1d":
            dates = pd.bdate_range("2024-01-01", periods=self.daily_days)
            return [
                {
                    "date": int(day.strftime("%Y%m%d")),
                    "code": symbol,
                    "name": "TEST",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "volume": 1000.0,
                    "amount": 10200.0,
                    "turnover": 1.5,
                    "pct_chg": 2.0,
                }
                for day in dates
            ]
        stamps = pd.date_range("2024-01-02 09:30", periods=241, freq="1min")
        return [
            {
                "code": symbol,
                "date": int(stamp.strftime("%Y%m%d%H%M%S")),
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.05,
                "volume": self.minute_volume,
                "amount": 1000.0,
            }
            for stamp in stamps
        ]


# --------------------------------------------------------------------------------------
# stockdb 数据源
# --------------------------------------------------------------------------------------
def test_fetch_daily_normalizes_schema():
    df = fetch_daily("600519", "20240101", "20240220", rd=FakeRd())
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert "turnover" in df.columns and "pct_chg" in df.columns


def test_turnover_is_converted_to_fraction():
    """stockdb 的 turnover 是百分数，必须转成小数，否则 B1 的换手条件永远不通过。"""
    df = fetch_daily("600519", "20240101", "20240220", rd=FakeRd())
    assert df["turnover"].max() == pytest.approx(0.015)


def test_fetch_minute_schema_and_size():
    minute = fetch_minute("600519", "20240102", "20240102", rd=FakeRd())
    assert len(minute) == 241
    assert minute.index.name == "minute"
    assert {"close", "volume"}.issubset(minute.columns)


def test_export_sample_writes_daily_and_minute(tmp_path):
    written = export_sample(
        ["600519"], "20240101", "20240220", tmp_path, minute_start="20240102", minute_end="20240102", rd=FakeRd()
    )
    paths = written["600519"]
    assert len(paths) == 2
    assert (tmp_path / "600519.csv").exists() and (tmp_path / "600519_min.csv").exists()
    minute_csv = pd.read_csv(tmp_path / "600519_min.csv")
    assert list(minute_csv.columns) == ["minute", "close", "volume"]
    assert len(minute_csv) == 241


def test_stock_db_data_source_protocol():
    source = StockDbDataSource(["600519", "000001"], "20240101", "20240220", rd=FakeRd())
    assert source.symbols() == ["600519", "000001"]
    df = source.bars("600519")
    assert not df.empty
    assert source.describe()["source"] == "stockdb"
    with pytest.raises(KeyError):
        source.bars("999999")
    with pytest.raises(ValueError):
        StockDbDataSource([], "20240101", "20240220", rd=FakeRd())


# --------------------------------------------------------------------------------------
# 量比确认评估
# --------------------------------------------------------------------------------------
def make_daily_minute(days=30, boost_later_days=False):
    daily = generate_synthetic_ohlcv(n_days=days, seed=11)
    daily["volume"] = 10000.0
    stamps = []
    for day in daily.index:
        minutes = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=241, freq="1min")
        volume = 100.0
        if boost_later_days and day >= daily.index[-5]:
            volume = 500.0                       # 最后几天开盘放量
        for i, stamp in enumerate(minutes):
            stamps.append({"minute": stamp, "close": 10.0, "volume": volume if i < 10 else 10.0})
    return daily, pd.DataFrame(stamps).set_index("minute")


def test_evaluate_confirmation_joins_forward_returns():
    daily, minute = make_daily_minute()
    signal = pd.Series(False, index=daily.index)
    signal.iloc[10] = True
    config = ConfirmEvalConfig(window_minutes=7, baseline_days=3, min_ratio=1.0, horizons=(1, 3))
    details, ratio = evaluate_confirmation(daily, minute, signal, config, symbol="T")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["symbol"] == "T"
    assert row["decision"] in {"买入", "观望", "无法判断"}
    for column in ("fwd_1", "fwd_3"):
        assert column in details.columns
    assert not ratio.empty


def test_summarize_groups_by_decision():
    details = pd.DataFrame(
        [
            {"decision": "买入", "fwd_1": 0.02, "fwd_3": 0.05},
            {"decision": "买入", "fwd_1": -0.01, "fwd_3": 0.01},
            {"decision": "观望", "fwd_1": -0.03, "fwd_3": -0.06},
            {"decision": "无法判断", "fwd_1": np.nan, "fwd_3": np.nan},
        ]
    )
    config = ConfirmEvalConfig(horizons=(1, 3))
    table = summarize_confirmation(details, config)
    buy = table[table["decision"] == "买入"].iloc[0]
    watch = table[table["decision"] == "观望"].iloc[0]
    assert buy["signals"] == 2 and buy["n_1"] == 2
    assert buy["mean_1"] == pytest.approx(0.005)
    assert buy["win_1"] == pytest.approx(0.5)
    assert watch["mean_3"] == pytest.approx(-0.06)
    assert table.iloc[-1]["decision"] == "全部"
    assert table.iloc[-1]["signals"] == 4
    assert summarize_confirmation(pd.DataFrame(), config).empty


def test_confirm_config_validation():
    with pytest.raises(ValueError):
        ConfirmEvalConfig(horizons=())
    with pytest.raises(ValueError):
        ConfirmEvalConfig(horizons=(0,))
    assert ConfirmEvalConfig(window_minutes=5, min_ratio=1.5).intraday.min_ratio == 1.5
