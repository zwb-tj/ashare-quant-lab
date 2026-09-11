import json

import pytest

from aqlab.cli import _apply_rule_overrides
from aqlab.notify import ConsoleNotifier
from aqlab.pipeline import DailyConfig, DailyPipeline, render_markdown, write_daily_report
from aqlab.rules import DEFAULT_RULE_BINDINGS, ActivityValueGate
from aqlab.tools import SyntheticDataSource


@pytest.fixture(scope="module")
def source():
    return SyntheticDataSource(n_symbols=12, n_days=400, seed=11)


def test_daily_pipeline_produces_sorted_picks(source):
    pipeline = DailyPipeline(source, config=DailyConfig(top_n=5), notifier=ConsoleNotifier())
    report = pipeline.run(push=True)

    assert 0 < report.universe_size == 12
    assert report.eligible_size <= 12
    assert len(report.picks) <= 5
    scores = [p["score"] for p in report.picks]
    assert scores == sorted(scores, reverse=True)
    assert [p["rank"] for p in report.picks] == list(range(1, len(report.picks) + 1))
    assert report.gate_state in (0, 1)
    assert abs(sum(report.weights.values()) - 1.0) < 1e-6
    for pick in report.picks:
        assert set(pick["rule_scores"]) == set(report.weights)


def test_daily_pipeline_is_deterministic(source):
    a = DailyPipeline(source, config=DailyConfig(top_n=5)).run(push=False)
    b = DailyPipeline(source, config=DailyConfig(top_n=5)).run(push=False)
    assert [p["symbol"] for p in a.picks] == [p["symbol"] for p in b.picks]
    assert [p["score"] for p in a.picks] == [p["score"] for p in b.picks]


def test_gate_closed_adds_note_and_keeps_watchlist(source):
    closed_gate = ActivityValueGate(fast_window=5, slow_window=20, on_threshold=50.0, off_threshold=40.0)
    report = DailyPipeline(source, gate=closed_gate, config=DailyConfig(top_n=3)).run(push=False)
    assert report.gate_state == 0
    assert any("开关关闭" in note for note in report.notes)
    # the watchlist is still produced, and every listed pick carries a non-zero score
    assert report.picks
    assert all(pick["score"] > 0 for pick in report.picks)


def test_zero_score_symbols_are_not_listed_as_picks(source):
    report = DailyPipeline(source, config=DailyConfig(top_n=10)).run(push=False)
    assert all(pick["score"] > 0 for pick in report.picks)
    assert all(pick["rules"] for pick in report.picks)


def test_require_signal_filters_untriggered_symbols(source):
    strict = DailyPipeline(source, config=DailyConfig(top_n=10, require_signal=True)).run(push=False)
    assert all(p["rules"] for p in strict.picks)


def test_rule_overrides_change_weights_and_params():
    updated = _apply_rule_overrides(DEFAULT_RULE_BINDINGS, ["needle_below_ma.ma_window=30", "volume_price_surge.confirm_days=3"])
    params = {name: p for name, p, _w in updated}
    assert params["needle_below_ma"]["ma_window"] == 30
    assert params["volume_price_surge"]["confirm_days"] == 3

    with pytest.raises(SystemExit):
        _apply_rule_overrides(DEFAULT_RULE_BINDINGS, ["unknown_rule.x=1"])
    with pytest.raises(SystemExit):
        _apply_rule_overrides(DEFAULT_RULE_BINDINGS, ["bad_item"])


def test_markdown_and_files_written(source, tmp_path):
    report = DailyPipeline(source, config=DailyConfig(top_n=3), notifier=ConsoleNotifier()).run(push=True)
    md = render_markdown(report)
    assert "# aqlab 每日选股" in md
    assert "综合分" in md
    assert "不构成投资建议" in md

    paths = write_daily_report(tmp_path, report)
    assert paths["markdown"].exists() and paths["json"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["as_of"] == report.as_of
    assert len(payload["picks"]) == len(report.picks)


def test_pipeline_rejects_empty_rule_set(source):
    with pytest.raises(ValueError):
        DailyPipeline(source, rule_bindings=[])
