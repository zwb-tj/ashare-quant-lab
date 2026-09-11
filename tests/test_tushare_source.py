import pandas as pd
import pytest

from aqlab.data import TushareDataSource, generate_synthetic_ohlcv, normalize_ohlcv


@pytest.fixture
def fake_fetch():
    calls = {"n": 0}

    def fetch(symbol, start, end, token=None):
        calls["n"] += 1
        return generate_synthetic_ohlcv(n_days=120, seed=3)

    fetch.calls = calls
    return fetch


def test_fetches_then_caches(tmp_path, fake_fetch):
    source = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=fake_fetch)
    df = source.bars("600519.SH")
    assert len(df) == 120
    assert fake_fetch.calls["n"] == 1
    assert (tmp_path / "600519_SH.csv").exists()

    # a fresh instance must read the cache instead of calling the network again
    again = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=fake_fetch)
    again.bars("600519.SH")
    assert fake_fetch.calls["n"] == 1  # unchanged
    assert again.degraded is False


def test_memory_cache_avoids_repeat_fetch(tmp_path, fake_fetch):
    source = TushareDataSource(["000001.SZ"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=fake_fetch)
    source.bars("000001.SZ")
    source.bars("000001.SZ")
    assert fake_fetch.calls["n"] == 1


def test_fetch_failure_without_cache_marks_degraded(tmp_path):
    def boom(symbol, start, end, token=None):
        raise RuntimeError("network down")

    source = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=boom)
    with pytest.raises(KeyError) as exc:
        source.bars("600519.SH")
    assert "无本地缓存" in str(exc.value)
    assert source.degraded is True
    assert "network down" in (source.last_error or "")


def test_cache_is_used_when_no_fetch_available(tmp_path):
    # seed the cache directly, then use a source whose fetch always fails
    df = generate_synthetic_ohlcv(n_days=90, seed=7)
    df.to_csv(tmp_path / "600519_SH.csv", encoding="utf-8-sig")

    def boom(*_a, **_k):
        raise RuntimeError("should not be called")

    source = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=boom)
    out = source.bars("600519.SH")
    assert len(out) == 90
    assert source.degraded is False


def test_describe_reports_cache_contents(tmp_path, fake_fetch):
    source = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=fake_fetch)
    info = source.describe()
    assert info["source"] == "tushare"
    assert info["symbols"] == ["600519.SH"]
    assert "cache_dir" in info


def test_empty_symbols_rejected(tmp_path):
    with pytest.raises(ValueError):
        TushareDataSource([], "2024-01-01", "2024-12-31", cache_dir=tmp_path)


# --------------------------------------------------------------------------------------
# 真实换手率（daily_basic）接入
# --------------------------------------------------------------------------------------
def make_turnover_fn(values=(0.02, 0.03)):
    calls = {"n": 0}

    def turnover_fn(symbol, start, end, token=None):
        calls["n"] += 1
        df = generate_synthetic_ohlcv(n_days=120, seed=3)
        series = pd.Series(list(values) * 60, index=df.index[: len(list(values) * 60)], dtype=float)
        return series.to_frame("turnover")

    turnover_fn.calls = calls
    return turnover_fn


def test_turnover_merged_and_cached(tmp_path, fake_fetch):
    turnover_fn = make_turnover_fn()
    source = TushareDataSource(
        ["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
        fetch_fn=fake_fetch, turnover_fn=turnover_fn, with_turnover=True,
    )
    df = source.bars("600519.SH")
    assert "turnover" in df.columns
    assert source.turnover_available is True
    assert turnover_fn.calls["n"] == 1
    assert source.describe()["with_turnover"] is True
    assert source.describe()["turnover_available"] is True

    # 缓存里带上了 turnover，第二个实例直接用缓存且仍然标记为可用
    again = TushareDataSource(
        ["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
        fetch_fn=fake_fetch, turnover_fn=turnover_fn, with_turnover=True,
    )
    cached = again.bars("600519.SH")
    assert "turnover" in cached.columns
    assert again.turnover_available is True
    assert turnover_fn.calls["n"] == 1     # 未再调用


def test_turnover_failure_degrades_without_breaking_bars(tmp_path, fake_fetch):
    def boom(*_a, **_k):
        raise RuntimeError("daily_basic unavailable")

    source = TushareDataSource(
        ["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
        fetch_fn=fake_fetch, turnover_fn=boom, with_turnover=True,
    )
    df = source.bars("600519.SH")
    assert "turnover" not in df.columns     # 行情仍然可用
    assert source.turnover_available is False
    assert "turnover" in (source.last_error or "")


def test_turnover_empty_result_marks_unavailable(tmp_path, fake_fetch):
    def empty(*_a, **_k):
        return pd.DataFrame(columns=["turnover"])

    source = TushareDataSource(
        ["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path,
        fetch_fn=fake_fetch, turnover_fn=empty, with_turnover=True,
    )
    df = source.bars("600519.SH")
    assert "turnover" not in df.columns
    assert source.turnover_available is False


def test_turnover_disabled_by_default(tmp_path, fake_fetch):
    source = TushareDataSource(["600519.SH"], "2024-01-01", "2024-12-31", cache_dir=tmp_path, fetch_fn=fake_fetch)
    df = source.bars("600519.SH")
    assert "turnover" not in df.columns
    assert source.turnover_available is None


def test_normalize_ohlcv_keeps_turnover_column():
    df = generate_synthetic_ohlcv(n_days=10, seed=1)
    df["turnover"] = 0.02
    out = normalize_ohlcv(df)
    assert "turnover" in out.columns
    assert list(out.columns)[:5] == ["open", "high", "low", "close", "volume"]
