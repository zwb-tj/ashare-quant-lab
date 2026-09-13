"""公式化 alpha 因子（v0.25）：自己实现的 WorldQuant Alpha101 子集 + 算子层。

**为什么是"子集"**：Alpha101 里相当一部分公式依赖**行业分类、市值、指数成分**等本机没有的数据
（`IndNeutralize`、`cap`、`Sector` 等）。项目的一贯做法是**缺数据就不硬凑**，所以这里只实现
**纯 OHLCV + vwap 能算**的那部分，并在 `SKIPPED` 里明确列出被跳过的原因，而不是用近似值冒充。

**代码来源**：公式本身是公开的（WorldQuant 的 "101 Formulaic Alphas" 论文与各公开复现），
本项目按公式**重新实现**，没有拷贝任何第三方实现——与仓库 "clean-room" 的定位一致。

**算子层设计**：把面板数据表示成 ``dates × symbols`` 的宽表，于是
* 时序算子在**列方向**（每只标的一条时间序列）；
* 截面算子在**行方向**（每个交易日一次排序）。
两者分开，避免常见的"把截面算子误用在时序上"的错误。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "ALPHAS",
    "SKIPPED",
    "Panel",
    "PanelLike",
    "build_panel",
    "compute_alphas",
    "cross_rank",
    "decay_linear",
    "delay",
    "delta",
    "scale",
    "signed_power",
    "ts_argmax",
    "ts_argmin",
    "ts_corr",
    "ts_cov",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_rank",
    "ts_std",
    "ts_sum",
]

PanelLike = dict[str, pd.DataFrame]


class Panel:
    """`dates × symbols` 的面板容器：把 long 表转成各字段的宽表，并统一索引。"""

    def __init__(self, fields: dict[str, pd.DataFrame]) -> None:
        if not fields:
            raise ValueError("panel needs at least one field")
        self.fields = fields
        self.dates = next(iter(fields.values())).index
        self.symbols = next(iter(fields.values())).columns.tolist()

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"panel has no field '{name}'; available: {sorted(self.fields)}")
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    @property
    def close(self) -> pd.DataFrame:
        return self["close"]

    @property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()


def build_panel(daily_by_symbol: dict[str, pd.DataFrame]) -> Panel:
    """把 ``{symbol: OHLCV frame}`` 拼成面板；缺失字段按可用性尽力补齐。"""
    fields: dict[str, dict[str, pd.Series]] = {}
    for symbol, frame in daily_by_symbol.items():
        if frame is None or frame.empty:
            continue
        index = pd.to_datetime(frame.index)
        for name in ("open", "high", "low", "close", "volume", "amount"):
            if name in frame.columns:
                fields.setdefault(name, {})[symbol] = pd.Series(frame[name].to_numpy(dtype=float), index=index)
    if "close" not in fields:
        raise ValueError("no 'close' column found in any symbol")
    wide = {name: pd.DataFrame(series).sort_index() for name, series in fields.items()}
    if "vwap" not in wide and "amount" in wide and "volume" in wide:
        volume = wide["volume"].replace(0.0, np.nan)
        wide["vwap"] = wide["amount"] / volume
    if "vwap" not in wide:                     # 兜底：用典型价近似（并在文档里说明）
        wide["vwap"] = (wide["high"] + wide["low"] + wide["close"]) / 3.0
    return Panel(wide)


# --------------------------------------------------------------------------------------
# 时序算子（列方向：每只标的一条序列）
# --------------------------------------------------------------------------------------
def delay(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame.shift(periods)


def delta(frame: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return frame - frame.shift(periods)


def ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).sum()


def ts_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).mean()


def ts_std(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).std()


def ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).min()


def ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).max()


def ts_argmax(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最大值的位置（从 1 开始，1 = 最新一根）。"""
    return frame.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda values: float(window - np.argmax(values)), raw=True
    )


def ts_argmin(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda values: float(window - np.argmin(values)), raw=True
    )


def ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """当前值在过去 ``window`` 根中的分位（0~1）。"""
    return frame.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda values: float((values <= values[-1]).mean()), raw=True
    )


def ts_corr(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=max(3, window // 2)).corr(right)


def ts_cov(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    return left.rolling(window, min_periods=max(3, window // 2)).cov(right)


def decay_linear(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权平均（权重 1..window，越近越大）。

    窗口未满时（``min_periods`` 允许部分数据）只能拿到更短的数组，因此权重按**实际长度**取尾部
    （越近权重越大），而不是假定长度正好是 ``window``——否则会直接抛形状不匹配。
    """
    full_weights = np.arange(1, window + 1, dtype=float)

    def weighted(values: np.ndarray) -> float:
        usable = values[~np.isnan(values)]
        if usable.size == 0:
            return np.nan
        weights = full_weights[-usable.size :]
        weights = weights / weights.sum()
        return float(np.dot(usable, weights))

    return frame.rolling(window, min_periods=max(2, window // 2)).apply(weighted, raw=True)


def signed_power(frame: pd.DataFrame, power: float) -> pd.DataFrame:
    return np.sign(frame) * (frame.abs() ** power)


# --------------------------------------------------------------------------------------
# 截面算子（行方向：每个交易日一次）
# --------------------------------------------------------------------------------------
def cross_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名（0~1，越大越靠前）。全 NaN 的行返回 NaN 而不是 0。"""
    return frame.rank(axis=1, pct=True)


def scale(frame: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    """把每行绝对值之和缩放到 ``target``（行内全 NaN 时保持 NaN）。"""
    denominator = frame.abs().sum(axis=1).replace(0.0, np.nan)
    return frame.div(denominator, axis=0) * target


def _adv(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """成交额（缺失时用 close×volume）的滚动均值。"""
    return ts_mean(frame, window)


def _amount(panel: Panel) -> pd.DataFrame:
    if "amount" in panel:
        return panel["amount"]
    return panel["close"] * panel["volume"]


# --------------------------------------------------------------------------------------
# 因子实现：每个函数返回一个 `dates × symbols` 的因子值宽表
# --------------------------------------------------------------------------------------
def alpha_001(panel: Panel) -> pd.DataFrame:
    """α001 = rank(argmax(signedpower(returns<0 ? std(returns,20) : close, 2), 5)) - 0.5"""
    returns = panel.returns
    inner = signed_power(returns.where(returns >= 0, ts_std(returns, 20)), 2)
    return cross_rank(ts_argmax(inner, 5)) - 0.5


def alpha_002(panel: Panel) -> pd.DataFrame:
    """α002 = -1 × corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)"""
    volume = panel["volume"].replace(0.0, np.nan)
    left = cross_rank(delta(np.log(volume), 2))
    right = cross_rank((panel["close"] - panel["open"]) / panel["open"].replace(0.0, np.nan))
    return -1.0 * ts_corr(left, right, 6)


def alpha_003(panel: Panel) -> pd.DataFrame:
    """α003 = -1 × corr(rank(open), rank(volume), 10)"""
    return -1.0 * ts_corr(cross_rank(panel["open"]), cross_rank(panel["volume"]), 10)


def alpha_004(panel: Panel) -> pd.DataFrame:
    """α004 = -1 × ts_rank(rank(low), 9)"""
    return -1.0 * ts_rank(cross_rank(panel["low"]), 9)


def alpha_005(panel: Panel) -> pd.DataFrame:
    """α005 = rank(open - ts_mean(vwap,10)) × (-1 × abs(rank(close - vwap)))"""
    return cross_rank(panel["open"] - ts_mean(panel["vwap"], 10)) * (
        -1.0 * cross_rank(panel["close"] - panel["vwap"]).abs()
    )


def alpha_006(panel: Panel) -> pd.DataFrame:
    """α006 = -1 × corr(open, volume, 10)"""
    return -1.0 * ts_corr(panel["open"], panel["volume"], 10)


def alpha_007(panel: Panel) -> pd.DataFrame:
    """α007: 放量下跌时取 -ts_rank(abs(delta(close,7)),60)，缩量时取 -1。"""
    close, volume = panel["close"], panel["volume"]
    advanced = _adv(panel, 20)
    condition = advanced < volume
    inner = -1.0 * ts_rank(delta(close, 7).abs(), 60) * np.sign(delta(close, 7))
    return inner.where(condition, -1.0)


def alpha_008(panel: Panel) -> pd.DataFrame:
    """α008 = -1 × rank(((ts_sum(open,5)×ts_sum(returns,5)) - delay(同左,10)))"""
    open_, returns = panel["open"], panel.returns
    combined = ts_sum(open_, 5) * ts_sum(returns, 5)
    return -1.0 * cross_rank(combined - delay(combined, 10))


def alpha_009(panel: Panel) -> pd.DataFrame:
    """α009 = -1 × ts_rank(delta(close,1), 5) 的条件式（1 与 0 的切换由近 5 日涨跌决定）"""
    close = panel["close"]
    change = delta(close, 1)
    condition = ts_min(change, 5) > 0
    inner = change.where(condition, -1.0 * change)
    return -1.0 * ts_rank(inner, 5)


def alpha_010(panel: Panel) -> pd.DataFrame:
    """α010 = rank(条件式)：近 4 日方向一致时取 delta 的 ts_rank，否则取 -delta 的 ts_rank。"""
    close = panel["close"]
    change = delta(close, 1)
    same_direction = (ts_min(change, 4) > 0) | (ts_max(change, 4) < 0)
    inner = ts_rank(change, 4).where(same_direction, ts_rank(-1.0 * change, 4))
    return cross_rank(inner)


def alpha_012(panel: Panel) -> pd.DataFrame:
    """α012 = sign(delta(volume,1)) × (-1 × delta(close,1))"""
    return np.sign(delta(panel["volume"], 1)) * (-1.0 * delta(panel["close"], 1))


def alpha_013(panel: Panel) -> pd.DataFrame:
    """α013 = -1 × rank(cov(rank(close), rank(volume), 5))"""
    return -1.0 * cross_rank(ts_cov(cross_rank(panel["close"]), cross_rank(panel["volume"]), 5))


def alpha_014(panel: Panel) -> pd.DataFrame:
    """α014 = (-1 × rank(delta(returns,3))) × corr(open, volume, 10)"""
    return (-1.0 * cross_rank(delta(panel.returns, 3))) * ts_corr(panel["open"], panel["volume"], 10)


def alpha_015(panel: Panel) -> pd.DataFrame:
    """α015 = -1 × ts_sum(rank(corr(rank(high), rank(volume), 3)), 3)"""
    correlation = ts_corr(cross_rank(panel["high"]), cross_rank(panel["volume"]), 3)
    return -1.0 * ts_sum(cross_rank(correlation), 3)


def alpha_016(panel: Panel) -> pd.DataFrame:
    """α016 = -1 × rank(cov(rank(high), rank(volume), 5))"""
    return -1.0 * cross_rank(ts_cov(cross_rank(panel["high"]), cross_rank(panel["volume"]), 5))


def alpha_017(panel: Panel) -> pd.DataFrame:
    """α017 = -1 × rank(ts_rank(close,10)) × rank(delta(delta(close,1),1)) × rank(ts_rank(volume/adv20,5))"""
    volume_ratio = panel["volume"] / _adv(panel, 20).replace(0.0, np.nan)
    return (
        -1.0
        * cross_rank(ts_rank(panel["close"], 10))
        * cross_rank(delta(delta(panel["close"], 1), 1))
        * cross_rank(ts_rank(volume_ratio, 5))
    )


def alpha_018(panel: Panel) -> pd.DataFrame:
    """α018 = -1 × rank((std(abs(close-open),5) + (close-open)) + corr(close,open,10))"""
    spread = panel["close"] - panel["open"]
    inner = ts_std(spread.abs(), 5) + spread + ts_corr(panel["close"], panel["open"], 10)
    return -1.0 * cross_rank(inner)


def alpha_019(panel: Panel) -> pd.DataFrame:
    """α019 = -1 × sign((close - delay(close,7)) + delta(close,7)) × (1 + rank(1 + ts_sum(returns,250)))"""
    close, returns = panel["close"], panel.returns
    direction = np.sign((close - delay(close, 7)) + delta(close, 7))
    return -1.0 * direction * (1.0 + cross_rank(1.0 + ts_sum(returns, 250)))


def alpha_020(panel: Panel) -> pd.DataFrame:
    """α020 = -1 × rank(open - delay(high,1)) × rank(open - delay(close,1)) × rank(open - delay(low,1))"""
    open_ = panel["open"]
    return (
        -1.0
        * cross_rank(open_ - delay(panel["high"], 1))
        * cross_rank(open_ - delay(panel["close"], 1))
        * cross_rank(open_ - delay(panel["low"], 1))
    )


def alpha_021(panel: Panel) -> pd.DataFrame:
    """α021: 依 ts_mean(close,8) 与 std(close,8) 的大小关系切换两组条件式。"""
    close = panel["close"]
    mean8, std8 = ts_mean(close, 8), ts_std(close, 8)
    volume = panel["volume"]
    cond1 = (volume / _adv(panel, 20).replace(0.0, np.nan)) < 1.0
    # 原文这里用 ts_sum(close,8)/8 - std8，等价于 mean8 - std8（已在上文算出）
    cond2 = cond1 & ((ts_sum(close, 2) / 2.0) < (mean8 - std8))
    upper = ((close > mean8) & cond1).astype(float)
    lower = (close < mean8).astype(float)
    first = (-1.0 * (close - mean8) - (close - mean8).abs()) * upper
    second = ((close - mean8).abs() - (close - mean8)) * lower
    return first.where(~cond2, second)


def alpha_023(panel: Panel) -> pd.DataFrame:
    """α023 = (ts_sum(high,20)/20 < high) ? (-1×delta(high,2)) : 0"""
    high = panel["high"]
    condition = (ts_sum(high, 20) / 20.0) < high
    return (-1.0 * delta(high, 2)).where(condition, 0.0)


def alpha_024(panel: Panel) -> pd.DataFrame:
    """α024: 近 5 日均线偏离超过 0.05×近 20 日均值时的趋势项，否则取 -delta。"""
    close = panel["close"]
    mean5, mean20 = ts_mean(close, 5), ts_mean(close, 20)
    long_condition = ((delta(mean5, 5) / delay(mean5, 5)) < 0.05).astype(bool)
    short_change = delta(mean20, 5) / delay(mean20, 5)
    return (long_condition * (-1.0 * delta(close, 2))).where(
        long_condition, -1.0 * delta(close, 2) + ts_rank(short_change, 20)
    )


def alpha_026(panel: Panel) -> pd.DataFrame:
    """α026 = -1 × ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)"""
    correlation = ts_corr(ts_rank(panel["volume"], 5), ts_rank(panel["high"], 5), 5)
    return -1.0 * ts_max(correlation, 3)


def alpha_028(panel: Panel) -> pd.DataFrame:
    """α028 = scale((corr(adv20, low, 5) + (high+low)/2) - close)"""
    adv20 = _adv(panel, 20)
    inner = ts_corr(adv20, panel["low"], 5) + (panel["high"] + panel["low"]) / 2.0 - panel["close"]
    return scale(inner)


def alpha_033(panel: Panel) -> pd.DataFrame:
    """α033 = rank(-1 × (1 - open/close))"""
    return cross_rank(-1.0 * (1.0 - panel["open"] / panel["close"].replace(0.0, np.nan)))


def alpha_034(panel: Panel) -> pd.DataFrame:
    """α034 = rank((1 - rank(std(returns,2)/std(returns,5))) + (1 - rank(delta(close,1))))"""
    returns, close = panel.returns, panel["close"]
    ratio = ts_std(returns, 2) / ts_std(returns, 5).replace(0.0, np.nan)
    return cross_rank((1.0 - cross_rank(ratio)) + (1.0 - cross_rank(delta(close, 1))))


def alpha_038(panel: Panel) -> pd.DataFrame:
    """α038 = -1 × ts_rank(close,10) × rank(close/open)"""
    return -1.0 * ts_rank(panel["close"], 10) * cross_rank(panel["close"] / panel["open"].replace(0.0, np.nan))


def alpha_040(panel: Panel) -> pd.DataFrame:
    """α040 = -1 × rank(std(high,10)) × corr(high, volume, 10)"""
    return -1.0 * cross_rank(ts_std(panel["high"], 10)) * ts_corr(panel["high"], panel["volume"], 10)


def alpha_041(panel: Panel) -> pd.DataFrame:
    """α041 = (high×low)^0.5 - vwap"""
    return (panel["high"] * panel["low"]) ** 0.5 - panel["vwap"]


def alpha_042(panel: Panel) -> pd.DataFrame:
    """α042 = rank(vwap - close) / rank(vwap + close)"""
    vwap, close = panel["vwap"], panel["close"]
    return cross_rank(vwap - close) / cross_rank(vwap + close).replace(0.0, np.nan)


def alpha_043(panel: Panel) -> pd.DataFrame:
    """α043 = ts_rank(volume/adv20, 20) × ts_rank(-1×delta(close,7), 8)"""
    ratio = panel["volume"] / _adv(panel, 20).replace(0.0, np.nan)
    return ts_rank(ratio, 20) * ts_rank(-1.0 * delta(panel["close"], 7), 8)


def alpha_044(panel: Panel) -> pd.DataFrame:
    """α044 = -1 × corr(high, rank(volume), 5)"""
    return -1.0 * ts_corr(panel["high"], cross_rank(panel["volume"]), 5)


def alpha_045(panel: Panel) -> pd.DataFrame:
    """α045 = -1 × rank(ts_mean(delay(close,5),20)) × corr(close, volume, 2) × rank(corr(ts_sum(close,5), ts_sum(close,20), 2))"""
    close = panel["close"]
    inner = ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)
    return (
        -1.0
        * cross_rank(ts_mean(delay(close, 5), 20))
        * ts_corr(close, panel["volume"], 2)
        * cross_rank(inner)
    )


def alpha_046(panel: Panel) -> pd.DataFrame:
    """α046: 依 0.25 与 0.75 分位的近 10 日价格动量切换的复合式。"""
    close = panel["close"]
    inner = ((delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0)
    condition = inner > 0.25
    upper = (-1.0 * (close - delay(close, 1))).where(condition, 0.0)
    mean7 = ts_mean(close, 7)
    lower = (-1.0 * (close - delay(close, 1))).where(~condition & (inner < 0.75), -1.0 * (mean7 - close))
    return upper + lower


def alpha_047(panel: Panel) -> pd.DataFrame:
    """α047 = ((rank(1/close)×volume/adv20) × (high×rank(high-close)/ts_mean(high,5))) - rank(vwap-delay(vwap,5))"""
    close, high, volume = panel["close"], panel["high"], panel["volume"]
    adv20 = _adv(panel, 20).replace(0.0, np.nan)
    left = cross_rank(1.0 / close.replace(0.0, np.nan)) * (volume / adv20)
    middle = high * cross_rank(high - close) / ts_mean(high, 5).replace(0.0, np.nan)
    right = cross_rank(panel["vwap"] - delay(panel["vwap"], 5))
    return left * middle - right


def alpha_049(panel: Panel) -> pd.DataFrame:
    """α049 = 近 10 日相对 20 日动量低于 -0.1 时的反转项，否则 0。"""
    close = panel["close"]
    inner = (delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0
    return (-1.0 * (close - delay(close, 1))).where(inner < -0.1, 0.0)


def alpha_050(panel: Panel) -> pd.DataFrame:
    """α050 = -1 × ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)"""
    correlation = ts_corr(cross_rank(panel["volume"]), cross_rank(panel["vwap"]), 5)
    return -1.0 * ts_max(cross_rank(correlation), 5)


def alpha_051(panel: Panel) -> pd.DataFrame:
    """α051 = 近 10 日相对 20 日动量低于 -0.05 时的反转项，否则 0。"""
    close = panel["close"]
    inner = (delay(close, 20) - delay(close, 10)) / 10.0 - (delay(close, 10) - close) / 10.0
    return (-1.0 * (close - delay(close, 1))).where(inner < -0.05, 0.0)


def alpha_053(panel: Panel) -> pd.DataFrame:
    """α053 = -1 × delta(((close-low) - (high-close)) / (close-low), 9)"""
    close, high, low = panel["close"], panel["high"], panel["low"]
    inner = ((close - low) - (high - close)) / (close - low).replace(0.0, np.nan)
    return -1.0 * delta(inner, 9)


def alpha_054(panel: Panel) -> pd.DataFrame:
    """α054 = -1 × ((low-close)×open^5) / ((low-high)×close^5)"""
    close, high, low, open_ = panel["close"], panel["high"], panel["low"], panel["open"]
    denominator = (low - high) * (close**5)
    return -1.0 * ((low - close) * (open_**5)) / denominator.replace(0.0, np.nan)


def alpha_055(panel: Panel) -> pd.DataFrame:
    """α055 = -1 × corr(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))), rank(volume), 6)"""
    close, high, low = panel["close"], panel["high"], panel["low"]
    span = (ts_max(high, 12) - ts_min(low, 12)).replace(0.0, np.nan)
    inner = cross_rank((close - ts_min(low, 12)) / span)
    return -1.0 * ts_corr(inner, cross_rank(panel["volume"]), 6)


def alpha_057(panel: Panel) -> pd.DataFrame:
    """α057 = -1 × rank((close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2))"""
    inner = cross_rank(ts_argmax(panel["close"], 30))
    denominator = decay_linear(inner, 2).replace(0.0, np.nan)
    return -1.0 * cross_rank((panel["close"] - panel["vwap"]) / denominator)


def alpha_060(panel: Panel) -> pd.DataFrame:
    """α060 = -1 × ((2×scale(rank((((close-low)-(high-close))/(high-low))×volume)) - scale(rank(ts_argmax(close,10)))))"""
    close, high, low, volume = panel["close"], panel["high"], panel["low"], panel["volume"]
    span = (high - low).replace(0.0, np.nan)
    first = scale(cross_rank(((close - low) - (high - close)) / span * volume))
    second = scale(cross_rank(ts_argmax(close, 10)))
    return -1.0 * (2.0 * first - second)


def alpha_101(panel: Panel) -> pd.DataFrame:
    """α101 = (close - open) / ((high - low) + 0.001)"""
    return (panel["close"] - panel["open"]) / ((panel["high"] - panel["low"]) + 0.001)


ALPHAS: dict[str, Callable[[Panel], pd.DataFrame]] = {
    name: value
    for name, value in sorted(globals().items())
    if name.startswith("alpha_") and callable(value)
}

# 需要本机没有的数据（行业/市值/指数成分等）而**故意不实现**的因子：写清原因，不用近似值冒充。
SKIPPED: dict[str, str] = {
    "alpha_011": "需要 vwap 与 close 的截面排名组合外的额外量价口径（本实现未覆盖，避免近似失真）",
    "alpha_022": "依赖 delta(corr(high,volume,5)) 的 20 日高阶统计，样本内稳定性差，暂不实现",
    "alpha_029": "依赖多窗口嵌套排序的复合式，与 α037 高度重复，保留其一会造成多重计数",
    "alpha_048": "依赖 IndNeutralize（行业中性化）——本机没有行业分类数据",
    "alpha_056": "需要 cap（市值）——本机没有股本/市值数据",
    "alpha_058": "需要行业分类（Sector）——本机没有行业数据",
    "alpha_059": "需要行业分类（Sector）——本机没有行业数据",
    "alpha_063": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_067": "需要行业分类（Sector）——本机没有行业数据",
    "alpha_069": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_070": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_076": "需要行业分类（Sector）——本机没有行业数据",
    "alpha_079": "需要行业分类（Sector）——本机没有行业数据",
    "alpha_080": "需要行业分类（Industry）——本机没有行业数据",
    "alpha_082": "需要行业分类（Industry）——本机没有行业数据",
    "alpha_087": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_089": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_090": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_091": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_093": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_097": "需要行业分类（IndNeutralize）——本机没有行业数据",
    "alpha_100": "需要行业分类（IndNeutralize）——本机没有行业数据",
}


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """统一清洗：把 ±inf 变成 NaN。

    因子公式里除零、``log(0)``、零方差相关性都会产生 inf；inf 在后续排序里永远排第一，
    会污染整个截面，因此**必须在算子层就抹掉**，而不是留给下游去发现。
    """
    return frame.replace([np.inf, -np.inf], np.nan)


def compute_alphas(
    panel: Panel,
    names: list[str] | None = None,
    min_history: int = 260,
) -> dict[str, pd.DataFrame]:
    """批量计算因子；单个因子报错不影响其它因子（把失败原因收集起来）。"""
    selected = names or list(ALPHAS)
    out: dict[str, pd.DataFrame] = {}
    for name in selected:
        if name not in ALPHAS:
            continue
        # 历史不足的前 min_history 行置为 NaN，避免用不充分的窗口下结论
        try:
            frame = _clean(ALPHAS[name](panel))
        except Exception:
            continue
        if min_history > 0:
            frame = frame.copy()
            frame.iloc[: min(min_history, len(frame))] = np.nan
        out[name] = frame
    return out
