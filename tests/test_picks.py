"""选股日志回测的测试（离线，合成数据）。"""

import json

import numpy as np
import pandas as pd
import pytest

from aqlab.picks import (
    PickBacktestConfig,
    attach_benchmark,
    benchmark_candidates,
    benchmark_returns,
    dedupe_picks,
    evaluate_picks,
    load_picks_archive,
    summarize_picks,
    summary_markdown,
)


def write_archive(tmp_path, payloads):
    for date, payload in payloads.items():
        payload = {"date": date, **payload}
        (tmp_path / f"picks_{date}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def daily_frame(closes, opens=None):
    closes = pd.Series(closes, dtype=float)
    index = pd.bdate_range("2026-06-01", periods=len(closes))
    opens = closes.shift(1).fillna(closes) if opens is None else pd.Series(opens, dtype=float)
    return pd.DataFrame(
        {"open": opens.values, "high": (closes * 1.01).values, "low": (closes * 0.99).values, "close": closes.values, "volume": 1000.0},
        index=index,
    )


# --------------------------------------------------------------------------------------
# 解析与去重
# --------------------------------------------------------------------------------------
def test_load_picks_archive_flattens_buckets(tmp_path):
    write_archive(
        tmp_path,
        {
            "2026-06-17": {"b1": [{"ts_code": "000333.SZ", "name": "美的集团", "score": 63.5}], "n20": [{"ts_code": "600519.SH", "name": "贵州茅台"}]},
            "2026-06-18": {"v3": [{"ts_code": "603376.SH", "name": "大明电子", "industry": "汽车配件"}]},
        },
    )
    picks = load_picks_archive(tmp_path, buckets=("b1", "n20", "v3"))
    assert len(picks) == 3
    assert set(picks["bucket"]) == {"b1", "n20", "v3"}
    row = picks[picks["ts_code"] == "000333.SZ"].iloc[0]
    assert row["symbol"] == "000333"                    # 去掉交易所后缀
    assert str(row["date"].date()) == "2026-06-17"
    assert row["score"] == 63.5
    assert picks[picks["ts_code"] == "600519.SH"].iloc[0]["score"] != picks[picks["ts_code"] == "600519.SH"].iloc[0]["score"]  # 缺失分数为 NaN


def test_load_picks_archive_empty(tmp_path):
    assert load_picks_archive(tmp_path).empty


def test_dedupe_picks_window():
    picks = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-17"), "symbol": "000333", "bucket": "b1"},
            {"date": pd.Timestamp("2026-06-18"), "symbol": "000333", "bucket": "b1"},   # 次日重复 -> 丢弃
            {"date": pd.Timestamp("2026-06-19"), "symbol": "000333", "bucket": "b1"},   # 3 天内 -> 丢弃
            {"date": pd.Timestamp("2026-06-30"), "symbol": "000333", "bucket": "b1"},   # 超过窗口 -> 保留
            {"date": pd.Timestamp("2026-06-18"), "symbol": "600519", "bucket": "b1"},
        ]
    )
    kept = dedupe_picks(picks, window_days=5)
    assert list(kept["date"].dt.strftime("%Y-%m-%d")) == ["2026-06-17", "2026-06-18", "2026-06-30"]
    assert len(dedupe_picks(picks, window_days=1)) == 5          # 窗口 1 天：只有同日才可能去重，跨日保留


# --------------------------------------------------------------------------------------
# 收益计算
# --------------------------------------------------------------------------------------
def test_evaluate_picks_next_open_entry_math():
    df = daily_frame([100, 110, 121, 133.1, 146.41])
    picks = pd.DataFrame([{"date": pd.Timestamp(df.index[0]), "symbol": "X", "bucket": "b1", "ts_code": "X.SZ", "name": "X"}])
    config = PickBacktestConfig(horizons=(1, 3), entry="next_open")
    out = evaluate_picks(picks, {"X": df}, config)

    row = out.iloc[0]
    assert row["entry_date"] == str(df.index[1].date())
    assert row["entry_price"] == pytest.approx(100.0)            # 次日开盘 = 前一日收盘（构造）
    assert row["fwd_1"] == pytest.approx(110.0 / 100.0 - 1)      # 买入当天收盘
    assert row["fwd_3"] == pytest.approx(133.1 / 100.0 - 1)      # 持有 3 个交易日


def test_evaluate_picks_next_close_entry():
    df = daily_frame([100, 110, 121, 133.1])
    picks = pd.DataFrame([{"date": pd.Timestamp(df.index[0]), "symbol": "X", "bucket": "b1"}])
    out = evaluate_picks(picks, {"X": df}, PickBacktestConfig(horizons=(1,), entry="next_close"))
    assert out.iloc[0]["entry_price"] == pytest.approx(110.0)
    assert out.iloc[0]["fwd_1"] == pytest.approx(0.0)            # 次日收盘买、次日收盘算 = 0


def test_evaluate_picks_handles_missing_and_short_data():
    df = daily_frame([100, 110])
    picks = pd.DataFrame(
        [
            {"date": pd.Timestamp(df.index[-1]), "symbol": "X", "bucket": "b1"},     # 之后没有数据
            {"date": pd.Timestamp("2026-06-01"), "symbol": "MISSING", "bucket": "b1"},
        ]
    )
    out = evaluate_picks(picks, {"X": df}, PickBacktestConfig(horizons=(1, 3)))
    assert out["entry_price"].isna().all()
    assert out["fwd_1"].isna().all()
    assert out["fwd_3"].isna().all()


def test_evaluate_picks_truncates_horizon_beyond_data():
    df = daily_frame([100, 110, 121])
    picks = pd.DataFrame([{"date": pd.Timestamp(df.index[1]), "symbol": "X", "bucket": "b1"}])
    out = evaluate_picks(picks, {"X": df}, PickBacktestConfig(horizons=(1, 5)))
    row = out.iloc[0]
    assert row["fwd_1"] == pytest.approx(121.0 / 110.0 - 1)
    assert np.isnan(row["fwd_5"])                                 # 数据不足 -> NaN，不填 0


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------
def test_summarize_and_markdown():
    evaluated = pd.DataFrame(
        [
            {"bucket": "b1", "fwd_1": 0.02, "fwd_3": 0.05, "fwd_5": 0.10, "fwd_10": 0.20},
            {"bucket": "b1", "fwd_1": -0.01, "fwd_3": -0.02, "fwd_5": np.nan, "fwd_10": np.nan},
            {"bucket": "v3", "fwd_1": 0.03, "fwd_3": 0.01, "fwd_5": 0.02, "fwd_10": -0.01},
        ]
    )
    config = PickBacktestConfig(horizons=(1, 3, 5, 10))
    summary = summarize_picks(evaluated, config)
    b1 = summary[summary["bucket"] == "b1"].iloc[0]
    assert b1["picks"] == 2 and b1["n_1"] == 2 and b1["n_5"] == 1
    assert b1["mean_1"] == pytest.approx(0.005)
    assert b1["win_3"] == pytest.approx(0.5)
    assert summary.iloc[-1]["bucket"] == "全部" and summary.iloc[-1]["picks"] == 3

    text = summary_markdown(summary, config)
    assert "分桶" in text and "选股数" in text and "去重窗口" in text
    assert summary_markdown(pd.DataFrame(), config) == "没有可汇总的选股记录。"


def test_summary_benchmark_uses_paired_subset():
    # 只有 2/3 条有基准：mean 用全部 3 条，bench/excess 只能用配对的 2 条
    evaluated = pd.DataFrame(
        [
            {"bucket": "b1", "fwd_1": 0.10, "bench_1": 0.02, "excess_1": 0.08},
            {"bucket": "b1", "fwd_1": -0.10, "bench_1": -0.02, "excess_1": -0.08},
            {"bucket": "b1", "fwd_1": 0.50, "bench_1": np.nan, "excess_1": np.nan},   # 无基准，不能进基准均值
        ]
    )
    config = PickBacktestConfig(horizons=(1,))
    summary = summarize_picks(evaluated, config)
    row = summary[summary["bucket"] == "b1"].iloc[0]
    assert row["n_1"] == 3 and row["npaired_1"] == 2
    assert row["mean_1"] == pytest.approx(0.50 / 3)
    assert row["bench_1"] == pytest.approx(0.0)          # (0.02 - 0.02) / 2
    assert row["excess_1"] == pytest.approx(0.0)         # mean - bench 必须自洽
    assert "npaired_1" not in summary_markdown(summary, config)


def test_config_validation():
    with pytest.raises(ValueError):
        PickBacktestConfig(horizons=())
    with pytest.raises(ValueError):
        PickBacktestConfig(entry="close")
    with pytest.raises(ValueError):
        PickBacktestConfig(dedupe_window=0)
    with pytest.raises(ValueError):
        PickBacktestConfig(benchmark_sample=-1)


# --------------------------------------------------------------------------------------
# 市场基准（等权篮子）
# --------------------------------------------------------------------------------------
def test_benchmark_candidates_excludes_picks_and_samples():
    frames = {f"{i:06d}": daily_frame([100, 101]) for i in range(10)}
    assert benchmark_candidates(frames, sample=0) == sorted(frames)                  # 0 = 全部
    chosen = benchmark_candidates(frames, exclude=["000000"], sample=4)
    assert "000000" not in chosen                                                    # 剔除自己选中的票
    assert len(chosen) == 4 and chosen[0] == "000001" and chosen[-1] == "000009"      # 等距覆盖
    assert benchmark_candidates(frames, sample=99) == sorted(frames)                 # 超出池子就取全部


def test_benchmark_returns_equal_weight_same_entry_date():
    # A: 开盘100 收盘110 (+10%)；B: 开盘100 收盘120 (+20%)；篮子等权 = +15%
    a = daily_frame([100.0, 110.0])
    b = daily_frame([100.0, 120.0])
    config = PickBacktestConfig(horizons=(1,), entry="next_open")
    bench = benchmark_returns({"A": a, "B": b}, [a.index[1]], config)
    row = bench.iloc[0]
    assert row["entry_date"] == str(a.index[1].date())
    assert row["bench_1"] == pytest.approx(0.15)
    assert row["benchn_1"] == 2


def test_benchmark_skips_symbols_lacking_the_entry_date():
    a = daily_frame([100.0, 110.0])
    short = daily_frame([100.0, 999.0]).iloc[:1]        # 只有信号日，没有入场日
    bench = benchmark_returns({"A": a, "S": short}, [a.index[1]], PickBacktestConfig(horizons=(1,)))
    assert bench.iloc[0]["benchn_1"] == 1
    assert bench.iloc[0]["bench_1"] == pytest.approx(0.10)


def test_benchmark_missing_entry_date_is_nan_not_zero():
    a = daily_frame([100.0, 110.0])
    bench = benchmark_returns({"A": a}, [pd.Timestamp("2030-01-01")], PickBacktestConfig(horizons=(1,)))
    assert np.isnan(bench.iloc[0]["bench_1"]) and bench.iloc[0]["benchn_1"] == 0


def test_attach_benchmark_excess_and_summary_columns():
    df = daily_frame([100.0, 110.0, 121.0])
    picks = pd.DataFrame([{"date": pd.Timestamp(df.index[0]), "symbol": "X", "bucket": "b1"}])
    config = PickBacktestConfig(horizons=(1, 3))
    evaluated = evaluate_picks(picks, {"X": df}, config)
    bench = pd.DataFrame([{"entry_date": evaluated.iloc[0]["entry_date"], "bench_1": 0.05, "benchn_1": 5}])
    merged = attach_benchmark(evaluated, bench)
    assert merged.iloc[0]["excess_1"] == pytest.approx(merged.iloc[0]["fwd_1"] - 0.05)
    assert np.isnan(merged.iloc[0]["excess_3"])                  # 没有 bench_3 -> NaN，不当 0

    summary = summarize_picks(merged, config)
    assert "excess_1" in summary.columns and "bench_1" in summary.columns
    assert summary.iloc[0]["excess_1"] == pytest.approx(merged.iloc[0]["excess_1"])
    text = summary_markdown(summary, config)
    assert "基准" in text and "超额" in text
