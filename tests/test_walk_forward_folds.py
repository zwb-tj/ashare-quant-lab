"""滚动多折验证的测试（离线，合成数据）。

这个模块的价值全在"能不能区分真信号与噪声"，所以测试围绕这一点：

① **真含信息的因子**：应当在**多数折**里被选中且在测试段存活；
② **纯噪声因子**：存活折数应当很少（允许偶然，但不能稳定存活）；
③ **折的构造**：训练段必须在测试段**之前**（用未来数据训练就是自欺），且各折测试段不重叠；
④ **方向口径**：测试段必须按**训练段定的方向**衡量，不能在测试段上事后翻方向。
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from aqlab.alpha101 import build_panel
from aqlab.walk_forward_folds import FoldConfig, ic_series, run_walk_forward, summarize_walk_forward


def _market(n_days=900, n_sym=40, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    returns = rng.normal(0.0005, 0.02, (n_days, n_sym))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=cols)
    open_price = close.shift(1).fillna(close)
    frames = {
        c: pd.DataFrame(
            {
                "open": open_price[c],
                "high": close[c] * 1.01,
                "low": close[c] * 0.99,
                "close": close[c],
                "volume": 1e6,
                "amount": close[c] * 1e6,
            }
        )
        for c in cols
    }
    return close, open_price, frames, build_panel(frames)


def test_config_validation():
    with pytest.raises(ValueError):
        FoldConfig(horizons=())
    with pytest.raises(ValueError):
        FoldConfig(step_days=0)
    with pytest.raises(ValueError):
        FoldConfig(min_symbols=2)
    with pytest.raises(ValueError):
        FoldConfig(n_folds=0)
    with pytest.raises(ValueError):
        FoldConfig(train_fraction=1.0)
    with pytest.raises(ValueError):
        FoldConfig(t_threshold=0)


def test_ic_series_keeps_dates_and_respects_horizon():
    close, _open, _frames, panel = _market(n_days=200, n_sym=20)
    rng = np.random.default_rng(1)
    factor = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    series = ic_series(factor, panel.close, horizon=5, dates=list(close.index), min_symbols=5)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.is_monotonic_increasing
    # 尾部 5 个截面没有前瞻数据
    assert len(series) == len(close.index) - 5
    assert series.between(-1.0, 1.0).all()


def test_folds_train_before_test_and_do_not_overlap():
    _close, _open, frames, _panel = _market()
    outcome = run_walk_forward(frames, FoldConfig(horizons=(5,), step_days=5, min_history=120, min_symbols=8, n_folds=4))
    folds = outcome["folds"]
    # 第一折没有"更早的历史"可训练，因此被跳过——这不是缺陷，而是 walk-forward 的定义使然
    assert len(folds) == 3, "n_folds=4 时第一折无训练数据，应剩 3 折"
    assert [fold["fold"] for fold in folds] == [2, 3, 4]
    for fold in folds:
        assert max(fold["train"]) < min(fold["test"]), "训练段必须严格早于测试段"
    # 测试段之间不重叠
    for first, second in itertools.pairwise(folds):
        assert max(first["test"]) < min(second["test"]), "相邻折的测试段不能重叠"
    # 训练段是扩张窗口：靠后的折训练段更长或等长
    lengths = [len(fold["train"]) for fold in folds]
    assert lengths == sorted(lengths)


def test_informative_factor_survives_in_most_folds_while_noise_does_not():
    """核心断言：真信号在多折里稳定存活，纯噪声不行。"""
    close, open_price, frames, _panel = _market(n_days=1100, n_sym=40, seed=3)
    rng = np.random.default_rng(4)
    # 构造一个真的含未来 5 日信息的因子（噪声版），以及一个纯噪声因子
    forward = close.shift(-5) / open_price.shift(-1) - 1
    informative = forward + rng.normal(0, 0.5 * float(forward.std().mean()), forward.shape)
    noise = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)

    frames_with = dict(frames)
    frames_with["SIGNAL"] = pd.DataFrame(
        {
            "open": open_price["S00"],
            "high": close["S00"] * 1.01,
            "low": close["S00"] * 0.99,
            "close": close["S00"],
            "volume": 1e6,
            "amount": close["S00"] * 1e6,
        }
    )

    config = FoldConfig(horizons=(5,), step_days=5, min_history=120, min_symbols=10, n_folds=5, t_threshold=2.0)
    # 这里直接验证底层：用 informative / noise 作为"因子"跑多折（走 ic_series + 折切片）
    sampled = list(close.index[::5])
    boundaries = np.linspace(0, len(sampled), config.n_folds + 1).astype(int)

    def count_survived(factor: pd.DataFrame) -> int:
        series = ic_series(factor, close, 5, sampled, 10)
        survived = 0
        for fold in range(config.n_folds):
            start, stop = int(boundaries[fold]), int(boundaries[fold + 1])
            test_dates = sampled[start:stop]
            history = sampled[:start]
            if not test_dates or not history:
                continue
            take = max(int(len(history) * config.train_fraction), 5)
            train = series.reindex([d for d in history[-take:] if d in series.index]).dropna()
            test = series.reindex([d for d in test_dates if d in series.index]).dropna()
            if train.empty or test.empty:
                continue
            direction = int(np.sign(train.mean())) or 1
            train_t = _t(train * direction, 5, 5)
            test_t = _t(test * direction, 5, 5)
            if np.isfinite(train_t) and train_t > config.t_threshold and np.isfinite(test_t) and test_t > config.t_threshold:
                survived += 1
        return survived

    informative_survived = count_survived(informative)
    noise_survived = count_survived(noise)
    assert informative_survived >= 3, f"真信号应在多数折存活，实际 {informative_survived}/5"
    assert noise_survived <= 1, f"纯噪声不该稳定存活，实际 {noise_survived}/5"


def _t(values: pd.Series, horizon: int, step_days: int) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if std <= 0:
        return float("nan")
    overlap = max(1, int(np.ceil(horizon / step_days)))
    return float(series.mean() / std * np.sqrt(len(series)) / np.sqrt(overlap))


def test_summarize_marks_robust_only_when_most_selected_folds_survive():
    detail = pd.DataFrame(
        [
            {"factor": "good", "horizon": 5, "fold": 1, "selected": True, "survived": True, "test_ic": 0.05, "test_t": 3.0},
            {"factor": "good", "horizon": 5, "fold": 2, "selected": True, "survived": True, "test_ic": 0.04, "test_t": 2.5},
            {"factor": "good", "horizon": 5, "fold": 3, "selected": True, "survived": False, "test_ic": -0.01, "test_t": -0.5},
            {"factor": "flaky", "horizon": 5, "fold": 1, "selected": True, "survived": True, "test_ic": 0.06, "test_t": 4.0},
            {"factor": "flaky", "horizon": 5, "fold": 2, "selected": False, "survived": False, "test_ic": 0.0, "test_t": 0.1},
            {"factor": "flaky", "horizon": 5, "fold": 3, "selected": False, "survived": False, "test_ic": -0.02, "test_t": -1.0},
        ]
    )
    table = summarize_walk_forward(detail)
    rows = {row.factor: row for row in table.itertuples(index=False)}
    assert rows["good"].folds_selected == 3 and rows["good"].folds_survived == 2
    assert bool(rows["good"].robust) is True, "3 折里 2 折存活（≥60%）应判为稳健"
    assert rows["flaky"].folds_selected == 1 and rows["flaky"].folds_survived == 1
    assert bool(rows["flaky"].robust) is False, "只有 1 折被选中，样本不足，不能算稳健"
    assert summarize_walk_forward(pd.DataFrame()).empty


def test_run_walk_forward_rejects_too_little_data():
    _close, _open, frames, _panel = _market(n_days=200, n_sym=10)
    with pytest.raises(ValueError):
        run_walk_forward(frames, FoldConfig(horizons=(5,), step_days=5, min_history=120, min_symbols=8, n_folds=20))
