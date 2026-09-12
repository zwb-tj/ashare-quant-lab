"""参数扫描测试（v0.6）。"""

import pandas as pd
import pytest

from aqlab.sweep import (
    GridSpec,
    SweepConfig,
    format_sweep,
    override_bindings,
    parse_grid,
    pick_best,
    run_sweep,
    write_sweep,
)
from aqlab.tools import SyntheticDataSource


def small_universe(n_symbols=6, n_days=220, seed=11):
    source = SyntheticDataSource(n_symbols=n_symbols, n_days=n_days, seed=seed)
    return {sym: source.bars(sym) for sym in source.symbols()}


# --------------------------------------------------------------------------------------
# 解析与覆盖
# --------------------------------------------------------------------------------------
def test_parse_grid_single_and_multiple():
    grids = parse_grid(["b1_graded.j_max=13,20,30"])
    assert len(grids) == 1
    assert grids[0].rule == "b1_graded" and grids[0].param == "j_max"
    assert grids[0].values == [13, 20, 30]
    assert grids[0].label == "b1_graded.j_max"

    grids2 = parse_grid(["needle_rsl.short_max=20,25", "b1_graded.spike_multiple=2.0,1.8"])
    assert [g.values for g in grids2] == [[20, 25], [2.0, 1.8]]

    # 非数值参数保持字符串（例如把 B2 的前置规则换成简化版 B1）
    str_grids = parse_grid(["b2_confirm.b1_rule=b1_opportunity"])
    assert str_grids[0].values == ["b1_opportunity"]


def test_parse_grid_errors():
    with pytest.raises(ValueError):
        parse_grid(["no_equals_sign"])
    with pytest.raises(ValueError):
        parse_grid(["b1_graded.j_max="])
    with pytest.raises(ValueError):
        parse_grid(["b1_graded=13,20"])


def test_override_bindings_does_not_mutate_source():
    base = [("b1_graded", {"j_max": 13}, 1.0), ("needle_rsl", {"short_max": 20}, 0.5)]
    updated = override_bindings(base, {"b1_graded": {"j_max": 25}})
    assert updated[0][1]["j_max"] == 25
    assert updated[1][1]["short_max"] == 20
    assert base[0][1]["j_max"] == 13                     # 原档案未被修改
    assert updated[0][1] is not base[0][1]                # 是副本

    untouched = override_bindings(base, {"unknown_rule": {"x": 1}})
    assert untouched[0][1] == {"j_max": 13}


# --------------------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------------------
def test_run_sweep_grid_columns_and_determinism():
    universe = small_universe()
    config = SweepConfig(horizons=(1, 5), main_horizon=5, test_days=60, step_days=60, min_history=120, min_positions=3.0)
    grids = parse_grid(["needle_rsl.short_max=20,25"])
    table = run_sweep({"6只": universe}, [("needle_rsl", {"long_min": 80.0}, 1.0)], grids, config)

    assert len(table) == 2
    for col in ("params", "universe", "signals", "excess_mean", "excess_win_rate", "trades",
                "trade_win_rate", "trade_excess", "windows_beating", "avg_positions",
                "total_cost", "final_equity", "sharpe", "max_drawdown", "meets_target"):
        assert col in table.columns
    assert table["signals"].sum() > 0
    assert table["meets_target"].dtype == bool

    again = run_sweep({"6只": universe}, [("needle_rsl", {"long_min": 80.0}, 1.0)], grids, config)
    pd.testing.assert_frame_equal(table, again)


def test_run_sweep_cartesian_product_and_universe_axis():
    universe = small_universe(n_symbols=4)
    config = SweepConfig(horizons=(1,), main_horizon=1, test_days=60, step_days=60, min_history=120)
    grids = parse_grid(["needle_rsl.short_max=20,30", "needle_rsl.long_min=80,85"])
    table = run_sweep({"4只": universe, "8只": small_universe(n_symbols=8)}, [("needle_rsl", {}, 1.0)], grids, config)
    assert len(table) == 2 * 2 * 2
    assert set(table["universe"]) == {"4只", "8只"}


def test_run_sweep_validation():
    universe = small_universe(n_symbols=3)
    with pytest.raises(ValueError):
        run_sweep({}, [("needle_rsl", {}, 1.0)], [])
    with pytest.raises(ValueError):
        run_sweep({"x": universe}, [], [])


def test_sweep_config_validation():
    with pytest.raises(ValueError):
        SweepConfig(horizons=(1, 3), main_horizon=5)
    with pytest.raises(ValueError):
        SweepConfig(min_positions=0)


# --------------------------------------------------------------------------------------
# 选择与渲染
# --------------------------------------------------------------------------------------
def make_table(rows):
    return pd.DataFrame(
        [
            {
                "params": p, "universe": u, "signals": 10, "excess_mean": e, "excess_win_rate": w,
                "trades": 5, "trade_win_rate": 0.5, "trade_excess": 0.0, "windows_beating": 1,
                "avg_positions": pos, "total_cost": 0.01, "final_equity": 1.1, "sharpe": 1.0,
                "max_drawdown": -0.05, "meets_target": pos >= 3,
            }
            for p, u, e, w, pos in rows
        ]
    )


def test_pick_best_prefers_qualified_rows():
    table = make_table(
        [
            ("a", "40只", 0.01, 0.55, 1.0),    # 超额最高但不达标
            ("b", "80只", 0.004, 0.52, 3.5),   # 达标
            ("c", "120只", 0.006, 0.53, 4.0),  # 达标且超额更高
        ]
    )
    best = pick_best(table, objective="excess_mean")
    assert best["qualified"] is True
    assert best["row"]["params"] == "c"


def test_pick_best_reports_when_nothing_qualifies():
    table = make_table([("a", "40只", 0.01, 0.55, 1.0), ("b", "80只", 0.002, 0.51, 2.5)])
    best = pick_best(table)
    assert best["qualified"] is False
    assert best["row"]["avg_positions"] == 2.5       # 取持仓最多的


def test_format_sweep_marks_recommendation_and_warning():
    config = SweepConfig(horizons=(1, 5), main_horizon=5, min_positions=3)
    qualified = make_table([("a", "80只", 0.004, 0.52, 3.5)])
    text = format_sweep(qualified, config, pick_best(qualified))
    assert "参数扫描报告" in text and "建议配置" in text and "超额均值%" in text

    none_ok = make_table([("a", "40只", 0.004, 0.52, 1.2)])
    text2 = format_sweep(none_ok, config, pick_best(none_ok))
    assert "没有任何配置达到" in text2

    assert "没有可用的扫描结果" in format_sweep(pd.DataFrame(), config)


def test_write_sweep_artifacts(tmp_path):
    table = make_table([("a", "40只", 0.004, 0.52, 3.2)])
    paths = write_sweep(tmp_path, table, meta={"profile": "zgnb_full"})
    assert paths["csv"].exists() and paths["json"].exists()
    assert "avg_positions" in paths["csv"].read_text(encoding="utf-8-sig")
    assert "zgnb_full" in paths["json"].read_text(encoding="utf-8")
