"""组合层测试（v0.5）。"""

import numpy as np
import pandas as pd
import pytest

from aqlab.portfolio import (
    PortfolioConfig,
    apply_constraints,
    covariance_matrix,
    equal_weights,
    exposure_report,
    inverse_vol_weights,
    mean_variance_weights,
    min_variance_weights,
    optimize_weights,
    project_to_capped_simplex,
    risk_parity_weights,
    simulate_portfolio,
)


def price_frame(closes_by_symbol):
    index = pd.bdate_range("2024-01-01", periods=len(next(iter(closes_by_symbol.values()))))
    return {
        symbol: pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.002 for c in closes],
                "low": [c * 0.998 for c in closes],
                "close": closes,
                "volume": 1000.0,
            },
            index=index,
        )
        for symbol, closes in closes_by_symbol.items()
    }


def returns_frame(values_by_symbol):
    index = pd.bdate_range("2024-01-01", periods=len(next(iter(values_by_symbol.values()))))
    return pd.DataFrame(values_by_symbol, index=index)


# --------------------------------------------------------------------------------------
# 数值工具
# --------------------------------------------------------------------------------------
def test_project_to_capped_simplex():
    w = project_to_capped_simplex(np.array([0.9, 0.5, -0.2, 0.1]), total=1.0, cap=0.4)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all() and (w <= 0.4 + 1e-12).all()
    uniform = project_to_capped_simplex(np.zeros(3), total=0.5, cap=1.0)
    assert uniform.sum() == pytest.approx(0.5)
    assert np.allclose(uniform, uniform[0])


def test_covariance_matrix_shrinks_off_diagonal():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (300, 2))
    returns = returns_frame({"A": base[:, 0], "B": base[:, 1] + 0.5 * base[:, 0]})
    cov_full = covariance_matrix(returns, shrinkage=0.0).to_numpy()
    cov_shrunk = covariance_matrix(returns, shrinkage=0.5).to_numpy()
    assert abs(cov_shrunk[0, 1]) < abs(cov_full[0, 1])          # 收缩后相关性变弱
    assert cov_shrunk[0, 0] == pytest.approx(cov_full[0, 0])    # 对角（方差）保持
    assert cov_full[0, 0] == pytest.approx(returns["A"].var(ddof=1) * 252, rel=1e-6)  # 允许 ridge 带来的 1e-10 偏差


# --------------------------------------------------------------------------------------
# 权重方法
# --------------------------------------------------------------------------------------
def test_equal_and_inverse_vol_weights():
    returns = returns_frame(
        {"LOWVOL": np.full(100, 0.001), "HIGHVOL": np.where(np.arange(100) % 2 == 0, 0.03, -0.03)}
    )
    equal = equal_weights(["A", "B"])
    assert equal.sum() == pytest.approx(1.0)
    assert equal.iloc[0] == pytest.approx(0.5)

    inverse = inverse_vol_weights(returns)
    assert inverse.sum() == pytest.approx(1.0)
    assert inverse["LOWVOL"] > inverse["HIGHVOL"]      # 低波动拿更高权重


def test_risk_parity_matches_inverse_vol_for_uncorrelated_assets():
    rng = np.random.default_rng(1)
    returns = returns_frame({"A": rng.normal(0, 0.01, 400), "B": rng.normal(0, 0.03, 400)})
    cov = covariance_matrix(returns, shrinkage=0.0)
    rp = risk_parity_weights(cov)
    inv = inverse_vol_weights(returns)
    assert rp.sum() == pytest.approx(1.0)
    # 不相关资产的风险平价 = 逆波动率配权
    assert rp["A"] == pytest.approx(inv["A"], rel=0.05)
    # 风险贡献应近似相等
    sigma = cov.to_numpy()
    w = rp.to_numpy()
    rc = w * (sigma @ w)
    assert rc.max() / rc.min() < 1.05


def test_min_variance_prefers_low_vol_and_respects_cap():
    returns = returns_frame({"LOW": np.random.default_rng(2).normal(0, 0.01, 400), "HIGH": np.random.default_rng(3).normal(0, 0.04, 400)})
    cov = covariance_matrix(returns, shrinkage=0.0)
    w = min_variance_weights(cov, cap=1.0)
    assert w.sum() == pytest.approx(1.0)
    assert w["LOW"] > w["HIGH"]
    capped = min_variance_weights(cov, cap=0.6)
    assert capped["LOW"] <= 0.6 + 1e-9
    assert capped.sum() == pytest.approx(1.0)


def test_mean_variance_tilts_toward_expected_return():
    rng = np.random.default_rng(4)
    returns = returns_frame({"GOOD": rng.normal(0.002, 0.01, 400), "MEH": rng.normal(0.0, 0.01, 400)})
    cov = covariance_matrix(returns, shrinkage=0.1)
    mu = returns.mean() * 252
    w = mean_variance_weights(mu, cov, cap=1.0, risk_aversion=0.5)
    assert w.sum() == pytest.approx(1.0)
    assert w["GOOD"] > w["MEH"]


def test_optimize_weights_dispatch_and_errors():
    returns = returns_frame({"A": np.random.default_rng(5).normal(0, 0.01, 200), "B": np.random.default_rng(6).normal(0, 0.01, 200)})
    for method in ("equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance"):
        w = optimize_weights(returns, method=method, cap=0.8)
        assert w.sum() == pytest.approx(1.0)
        assert (w <= 0.8 + 1e-9).all() and (w >= 0).all()
    with pytest.raises(ValueError):
        optimize_weights(returns, method="nope")
    with pytest.raises(ValueError):
        optimize_weights(pd.DataFrame(), method="equal")


# --------------------------------------------------------------------------------------
# 约束
# --------------------------------------------------------------------------------------
def test_apply_constraints_cash_buffer_cap_and_turnover_limit():
    config = PortfolioConfig(max_weight=0.4, cash_buffer=0.2, turnover_limit=0.5)
    target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    weights, turnover = apply_constraints(target, previous=None, config=config)
    assert weights.sum() == pytest.approx(0.8)          # 20% 现金
    assert (weights <= 0.4 + 1e-9).all()
    assert turnover == pytest.approx(0.4)               # 从空仓建到 80% 仓位：单边换手 = 0.5×0.8

    # 换手上限：从满仓 A 换到 B，单边换手 1.0 被限制到 0.3
    strict = PortfolioConfig(max_weight=1.0, cash_buffer=0.0, turnover_limit=0.3)
    weights2, turnover2 = apply_constraints(pd.Series({"A": 0.0, "B": 1.0}), {"A": 1.0, "B": 0.0}, strict)
    assert turnover2 == pytest.approx(0.3)
    assert weights2["B"] == pytest.approx(0.3)          # 只走了 30% 的路
    assert weights2["A"] == pytest.approx(0.7)


def test_apply_constraints_all_zero_target():
    config = PortfolioConfig()
    weights, turnover = apply_constraints(pd.Series({"A": 0.0, "B": 0.0}), {"A": 0.5}, config)
    assert weights.sum() == 0.0
    assert turnover == 0.0


# --------------------------------------------------------------------------------------
# 暴露
# --------------------------------------------------------------------------------------
def test_exposure_report_concentration_and_style():
    rng = np.random.default_rng(7)
    # 用带噪声的随机游走（完美指数序列的波动率≈0，无法检验暴露计算）
    universe = price_frame(
        {
            "A": list(100 * np.cumprod(1 + rng.normal(0.003, 0.012, 120))),
            "B": list(50 * np.cumprod(1 + rng.normal(0.001, 0.008, 120))),
            "C": list(80 * np.ones(120)),
        }
    )
    report = exposure_report({"A": 0.4, "B": 0.3, "C": 0.1}, universe)
    assert report["gross_exposure"] == pytest.approx(0.8)
    assert report["positions"] == 3
    assert report["max_weight"] == pytest.approx(0.4)
    # 归一化后的 HHI = 0.5^2+0.375^2+0.125^2
    assert report["hhi"] == pytest.approx(0.5 ** 2 + 0.375 ** 2 + 0.125 ** 2, abs=1e-4)
    assert report["effective_n"] == pytest.approx(1 / report["hhi"], rel=1e-3)
    assert report["weighted_ann_vol"] > 0
    assert report["weighted_momentum"] > 0


def test_exposure_report_empty_weights():
    report = exposure_report({}, price_frame({"A": [100.0] * 30}))
    assert report["gross_exposure"] == 0.0
    assert report["positions"] == 0


# --------------------------------------------------------------------------------------
# 组合模拟
# --------------------------------------------------------------------------------------
def test_simulate_portfolio_compounds_without_costs():
    n = 40
    closes = list(100 * 1.01 ** np.arange(n))
    universe = price_frame({"A": closes, "B": closes})
    signals = {s: pd.Series(True, index=universe[s].index) for s in universe}
    config = PortfolioConfig(method="equal", rebalance_days=1, max_weight=1.0, cash_buffer=0.0, turnover_limit=2.0, cost_bps=0.0, min_history=20)
    result = simulate_portfolio(universe, signals, config)

    frame = result["frame"]
    assert not frame.empty
    # 从 min_history 那根开始建仓，两票同涨 1%/日 -> 净值按 1.01 复利
    expected = 1.01 ** (n - 1 - config.min_history)
    assert frame["value"].iloc[-1] == pytest.approx(expected, rel=1e-6)
    assert result["rebalances"]["turnover"].iloc[-1] == pytest.approx(0.0, abs=1e-9)   # 后续再平衡无需换手
    assert result["summary"]["total_cost"] == pytest.approx(0.0, abs=1e-9)
    assert result["metrics"]["total_return"] > 0


def test_simulate_portfolio_charges_costs_on_drift():
    n = 60
    universe = price_frame({"FAST": list(100 * 1.02 ** np.arange(n)), "FLAT": list(100 * np.ones(n))})
    signals = {s: pd.Series(True, index=universe[s].index) for s in universe}
    base = {"method": "equal", "rebalance_days": 1, "max_weight": 1.0, "cash_buffer": 0.0, "turnover_limit": 2.0, "min_history": 20}

    free = simulate_portfolio(universe, signals, PortfolioConfig(**base, cost_bps=0.0))
    costly = simulate_portfolio(universe, signals, PortfolioConfig(**base, cost_bps=100.0))

    assert costly["summary"]["total_cost"] > 0            # 漂移带来换手 -> 收费
    assert costly["frame"]["value"].iloc[-1] < free["frame"]["value"].iloc[-1]
    assert costly["summary"]["avg_turnover"] > 0


def test_simulate_portfolio_respects_turnover_limit():
    n = 60
    universe = price_frame({"FAST": list(100 * 1.02 ** np.arange(n)), "FLAT": list(100 * np.ones(n))})
    signals = {s: pd.Series(True, index=universe[s].index) for s in universe}
    config = PortfolioConfig(
        method="equal", rebalance_days=1, max_weight=1.0, cash_buffer=0.0,
        turnover_limit=0.01, cost_bps=0.0, min_history=20,
    )
    result = simulate_portfolio(universe, signals, config)
    assert (result["rebalances"]["turnover"] <= 0.01 + 1e-9).all()


def test_simulate_portfolio_respects_cash_buffer_and_cap():
    n = 60
    closes = list(100 * 1.001 ** np.arange(n))
    universe = price_frame({"A": closes, "B": closes, "C": closes})
    signals = {s: pd.Series(True, index=universe[s].index) for s in universe}
    config = PortfolioConfig(method="equal", rebalance_days=5, max_weight=0.25, cash_buffer=0.2, min_history=20, cost_bps=0.0)
    result = simulate_portfolio(universe, signals, config)
    for record in result["rebalances"].to_dict("records"):
        assert record["gross_exposure"] <= 0.8 + 1e-6
        assert max(record["weights"].values()) <= 0.25 + 1e-6


def test_simulate_portfolio_goes_flat_without_signals():
    n = 60
    universe = price_frame({"A": list(100 * 1.01 ** np.arange(n))})
    signals = {"A": pd.Series(False, index=universe["A"].index)}
    result = simulate_portfolio(universe, signals, PortfolioConfig(min_history=20))
    assert result["summary"]["rebalances"] == 0
    assert result["frame"]["value"].iloc[-1] == pytest.approx(1.0)


def test_simulate_portfolio_insufficient_data():
    universe = price_frame({"A": [100.0] * 10})
    signals = {"A": pd.Series(True, index=universe["A"].index)}
    result = simulate_portfolio(universe, signals, PortfolioConfig(min_history=20))
    assert result["frame"].empty
    assert "note" in result["summary"]


def test_simulate_portfolio_deterministic():
    n = 50
    universe = price_frame({"A": list(100 * 1.003 ** np.arange(n)), "B": list(100 * 1.001 ** np.arange(n))})
    signals = {s: pd.Series(True, index=universe[s].index) for s in universe}
    config = PortfolioConfig(method="risk_parity", rebalance_days=5, min_history=20)
    first = simulate_portfolio(universe, signals, config)
    second = simulate_portfolio(universe, signals, config)
    pd.testing.assert_frame_equal(first["frame"], second["frame"])
    pd.testing.assert_frame_equal(first["rebalances"], second["rebalances"])


# --------------------------------------------------------------------------------------
# 配置校验
# --------------------------------------------------------------------------------------
def test_config_validation():
    with pytest.raises(ValueError):
        PortfolioConfig(method="nope")
    with pytest.raises(ValueError):
        PortfolioConfig(lookback=2)
    with pytest.raises(ValueError):
        PortfolioConfig(rebalance_days=0)
    with pytest.raises(ValueError):
        PortfolioConfig(max_weight=0.0)
    with pytest.raises(ValueError):
        PortfolioConfig(cash_buffer=1.0)
    with pytest.raises(ValueError):
        PortfolioConfig(turnover_limit=0.0)
    with pytest.raises(ValueError):
        PortfolioConfig(cost_bps=-1)
    with pytest.raises(ValueError):
        PortfolioConfig(shrinkage=1.5)
    with pytest.raises(ValueError):
        PortfolioConfig(risk_aversion=0)
    with pytest.raises(ValueError):
        PortfolioConfig(min_history=5)
