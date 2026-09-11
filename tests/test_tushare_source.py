import pandas as pd
import pytest

from aqlab.data import TushareDataSource, generate_synthetic_ohlcv


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
