import pytest

from aqlab.tools import INDICATORS, SyntheticDataSource, ToolRegistry, ToolSpec, default_registry


@pytest.fixture(scope="module")
def registry():
    return ToolRegistry(source=SyntheticDataSource(n_symbols=8, n_days=400, seed=5))


def test_specs_are_openai_tool_shaped(registry):
    specs = registry.specs()
    assert {s["function"]["name"] for s in specs} == {
        "list_strategies",
        "describe_data",
        "get_bars",
        "compute_indicator",
        "run_backtest",
        "screen_universe",
    }
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"


def test_list_strategies_returns_params(registry):
    out = registry.call("list_strategies")
    assert out["ok"] is True
    names = {item["name"] for item in out["strategies"]}
    assert {"momentum", "ma_cross", "mean_reversion", "buy_and_hold"}.issubset(names)
    ma = next(item for item in out["strategies"] if item["name"] == "ma_cross")
    assert set(ma["params"]) == {"fast", "slow"}


def test_describe_data_is_ground_truth_for_symbols(registry):
    out = registry.call("describe_data")
    assert out["ok"] is True
    assert out["source"] == "synthetic"
    assert len(out["symbols"]) == 8
    assert out["bars_per_symbol"] == 400


def test_get_bars_ok_and_tail_bounded(registry):
    out = registry.call("get_bars", {"symbol": "SYN001", "tail": 5})
    assert out["ok"] is True
    assert len(out["recent"]) == 5
    assert out["rows"] == 400
    assert out["last_close"] == out["recent"][-1]["close"]

    capped = registry.call("get_bars", {"symbol": "SYN001", "tail": 999})
    assert len(capped["recent"]) == 60


def test_get_bars_unknown_symbol_returns_error_not_exception(registry):
    out = registry.call("get_bars", {"symbol": "ZZZ_NOT_EXIST"})
    assert out["ok"] is False
    assert "ZZZ_NOT_EXIST" in out["error"]


def test_compute_indicator_ok_and_errors(registry):
    out = registry.call("compute_indicator", {"symbol": "SYN002", "indicator": "rsi", "window": 14})
    assert out["ok"] is True
    assert 0 <= out["last_value"] <= 100

    bad_name = registry.call("compute_indicator", {"symbol": "SYN002", "indicator": "nope"})
    assert bad_name["ok"] is False
    assert "unknown indicator" in bad_name["error"]

    bad_window = registry.call("compute_indicator", {"symbol": "SYN002", "indicator": "sma", "window": 1})
    assert bad_window["ok"] is False


def test_run_backtest_matches_direct_engine(registry):
    from aqlab.backtest import BacktestConfig, run_backtest
    from aqlab.metrics import compute_metrics
    from aqlab.strategies import MACrossStrategy

    args = {"symbol": "SYN003", "strategy": "ma_cross", "params": {"fast": 10, "slow": 30}}
    out = registry.call("run_backtest", args)
    assert out["ok"] is True
    assert out["metrics"]["trades"] >= 1

    df = registry.source.bars("SYN003")
    strategy = MACrossStrategy(fast=10, slow=30)
    config = BacktestConfig()
    result = run_backtest(df, strategy.positions(df), config=config)
    expected = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)

    assert out["metrics"]["total_return"] == pytest.approx(round(expected["total_return"], 4))
    assert out["metrics"]["sharpe"] == pytest.approx(round(expected["sharpe"], 4), abs=1e-4)


def test_run_backtest_rejects_bad_strategy_and_params(registry):
    unknown = registry.call("run_backtest", {"symbol": "SYN001", "strategy": "nope"})
    assert unknown["ok"] is False and "unknown strategy" in unknown["error"]

    bad_params = registry.call("run_backtest", {"symbol": "SYN001", "strategy": "ma_cross", "params": {"fast": 30, "slow": 10}})
    assert bad_params["ok"] is False
    assert "invalid strategy parameters" in bad_params["error"]

    missing = registry.call("run_backtest", {"symbol": "SYN001", "strategy": "ma_cross", "params": {"fast": 10, "oops": 3}})
    assert missing["ok"] is False


def test_screen_universe_ranks_top_n(registry):
    out = registry.call("screen_universe", {"top_n": 3})
    assert out["ok"] is True
    assert len(out["top"]) == 3
    assert out["top"][0]["rank"] == 1
    scores = [row["score"] for row in out["top"]]
    assert scores == sorted(scores, reverse=True)
    assert out["universe_size"] == 8


def test_unknown_tool_raises_and_custom_tool_registers(registry):
    with pytest.raises(KeyError):
        registry.call("does_not_exist")

    registry.register(
        ToolSpec(
            name="echo",
            description="test tool",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: {"ok": True, "echo": text},
        )
    )
    assert registry.call("echo", {"text": "hi"})["echo"] == "hi"
    assert "echo" in registry.tool_names()


def test_registry_never_raises_on_bad_arguments(registry):
    # unknown argument name -> TypeError captured as a tool error, not an exception
    out = registry.call("get_bars", {"wrong_arg": 1})
    assert out["ok"] is False
    # missing required argument is captured too
    missing = registry.call("get_bars", {})
    assert missing["ok"] is False
    # only an unknown *tool name* is a programming error and raises
    with pytest.raises(KeyError):
        registry.call("nope_not_a_tool")


def test_default_registry_is_offline_and_has_indicators():
    reg = default_registry()
    assert {"sma", "ema", "rsi", "atr", "zscore", "realized_vol", "momentum"} == set(INDICATORS)
    assert reg.call("describe_data")["ok"] is True
