"""数据质量审计测试（v0.7）。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.data import generate_synthetic_ohlcv
from aqlab.quality import (
    QualityConfig,
    audit_universe,
    check_frame,
    cross_source_diff,
    format_audit,
    snapshot_hash,
    write_audit,
)


def frame(n=120, seed=1):
    df = generate_synthetic_ohlcv(n_days=max(n, 10), seed=seed)
    return df.iloc[:n]


def issue_types(issues):
    return {i["type"] for i in issues}


def test_clean_frame_has_only_turnover_info():
    df = frame(120)
    issues = check_frame("CLEAN", df)
    assert issue_types(issues) == {"no_turnover"}
    assert all(i["severity"] == "info" for i in issues)


def test_detects_duplicates_unsorted_and_insufficient_history():
    df = frame(20)                                    # 少于默认 min_bars=60
    broken = pd.concat([df, df.iloc[[5]]]).sort_index(ascending=False)
    issues = check_frame("BROKEN", broken)
    types = issue_types(issues)
    assert "duplicate_dates" in types
    assert "not_sorted" in types
    assert "insufficient_history" in types


def test_detects_calendar_gap_zero_volume_and_price_jump():
    df = frame(120)
    df = df.drop(df.index[50:70])                     # 制造缺口
    df.loc[df.index[10], "volume"] = 0.0
    df.loc[df.index[20], "volume"] = 0.0
    df.loc[df.index[30], "close"] = df["close"].iloc[29] * 1.5   # 制造跳变
    issues = check_frame("DIRTY", df, QualityConfig(max_gap_days=5, zero_volume_ratio=0.01))
    types = issue_types(issues)
    assert "calendar_gap" in types
    assert "zero_volume" in types
    assert "price_jump" in types
    gap = next(i for i in issues if i["type"] == "calendar_gap")
    assert "最大间隔" in gap["detail"]


def test_detects_missing_values():
    df = frame(120)
    df.loc[df.index[5], "close"] = np.nan
    assert "missing_values" in issue_types(check_frame("NAN", df))


def test_empty_frame():
    issues = check_frame("EMPTY", pd.DataFrame())
    assert issues[0]["type"] == "empty" and issues[0]["severity"] == "error"


def test_audit_universe_aggregates_and_sorts():
    clean = frame(120, seed=1)
    dirty = frame(120, seed=2)
    dirty.loc[dirty.index[3], "volume"] = 0.0
    dirty.loc[dirty.index[4], "volume"] = 0.0
    dirty.loc[dirty.index[5], "volume"] = 0.0
    broken = clean.drop(clean.index[40:60])

    table = audit_universe({"CLEAN": clean, "DIRTY": dirty, "BROKEN": broken}, QualityConfig(max_gap_days=5, zero_volume_ratio=0.01))
    assert set(table["symbol"]) == {"CLEAN", "DIRTY", "BROKEN"}
    assert table.iloc[0]["symbol"] == "BROKEN"          # 有 warning 的排在前面
    assert (table["warnings"] >= 1).sum() >= 2
    assert set(table.columns) >= {"symbol", "bars", "first", "last", "errors", "warnings", "infos", "issue_types", "details"}


def test_cross_source_diff_identical_and_perturbed():
    a = frame(120, seed=3)
    identical = cross_source_diff("A", a, a.copy())
    assert identical["common_days"] == len(a)
    assert identical["close_mismatch_rate"] == 0.0

    b = a.copy()
    b.loc[b.index[10], "close"] = b["close"].iloc[10] * 1.05      # 5% 差异
    b.loc[b.index[11], "volume"] = b["volume"].iloc[11] * 2
    diff = cross_source_diff("A", a, b, tolerance=0.005)
    assert diff["close_mismatch_rate"] > 0
    assert diff["volume_mismatch_rate"] > 0
    assert diff["examples"] and diff["examples"][0]["date"] == str(b.index[10].date())


def test_cross_source_diff_no_overlap():
    a = frame(120, seed=4)
    shifted = a.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1000)
    diff = cross_source_diff("A", a, shifted)
    assert diff["common_days"] == 0
    assert diff["examples"] == []


def test_snapshot_hash_stable_and_sensitive():
    a = frame(120, seed=5)
    h1 = snapshot_hash(a)
    h2 = snapshot_hash(a.copy())
    assert h1 == h2 and len(h1) == 16

    b = a.copy()
    b.loc[b.index[10], "close"] = b["close"].iloc[10] * 1.01
    assert snapshot_hash(b) != h1

    shuffled = a[sorted(a.columns, reverse=True)]
    assert snapshot_hash(shuffled) == h1               # 列顺序不影响指纹


def test_config_validation():
    for kwargs in ({"max_gap_days": 0}, {"price_jump_pct": 0}, {"min_bars": 2}, {"zero_volume_ratio": 2}):
        with pytest.raises(ValueError):
            QualityConfig(**kwargs)


def test_format_and_write_artifacts(tmp_path):
    clean = frame(120, seed=6)
    table = audit_universe({"CLEAN": clean})
    diffs = [cross_source_diff("CLEAN", clean, clean)]
    hashes = {"CLEAN": snapshot_hash(clean)}
    text = format_audit(table, diffs, hashes)
    assert "数据质量审计" in text and "多源交叉校验" in text and "数据指纹" in text

    paths = write_audit(tmp_path, table, diffs, hashes)
    assert paths["markdown"].exists() and paths["csv"].exists() and paths["json"].exists()
    assert "CLEAN" in paths["csv"].read_text(encoding="utf-8-sig")
