"""Personal strategy rules: B1 / B2 / B3, 单针下20/30, 量价齐升V3, 0AMV 活跃市值开关.

These are the parameterised, re-implemented rules from the author's own trading
research (thresholds are configuration, not hard-coded magic):

=====================  ==========================================================
规则                    口径
=====================  ==========================================================
``B1Graded``           B1 梯度打分：J ≤ 13、近15日有放量日、极致缩量、双线多头、前N低点未破，
                       再按 4 条软性条件加分（60 + 10×软通过数）
``B1Opportunity``      简化版 B1：J ≤ -10、涨幅 -2%~+1.8%、振幅 ≤ 7%、累计换手 < 38%（保留兼容）
``B2Confirm``          B1 后 3 个交易日内，涨幅 ≥ 4%，J < 55，且放量（量 > 前一日）
``B3Confirm``          B2 后出现十字星/小阴线，且平开（开盘价 ≈ 前收）
``NeedleRSL``          单针下20：RSL(3) ≤ 20 且 RSL(21) ≥ 80；单针下30：RSL(3) < 30 且 RSL(21) > 85
``VolumePriceV3``      连续 2 日阳线且价格创新高、连续 2 日量增、当日涨幅 2%~6%、
                       白线 > 黄线且收盘 ≥ 黄线×0.97、J < 60；评分 = 70 基础分 + 加分项
``ActiveMarketValueGate``  0AMV 活跃市值：活筹置换衰减模型 ρ=0.92，波段开关（见类文档）
=====================  ==========================================================

Every rule still returns a ``[0, 1]`` score series, so they compose with the
existing pipeline exactly like the generic rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aqlab.indicators_extra import (
    amplitude,
    brick_chart,
    brick_streaks,
    kdj,
    pct_change_1d,
    rsl,
    white_line,
    yellow_line,
)

__all__ = [
    "PERSONAL_RULES",
    "ActiveMarketValueGate",
    "B1Graded",
    "B1Opportunity",
    "B2Confirm",
    "B3Confirm",
    "BrickGreenToRed",
    "NeedleRSL",
    "VolumePriceV3",
    "brick_filter_mask",
    "build_personal_rule",
]


def _clean(score: pd.Series, index: pd.Index) -> pd.Series:
    out = score.reindex(index).astype(float).replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0).clip(0.0, 1.0)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


# --------------------------------------------------------------------------------------
# B1 / B2 / B3
# --------------------------------------------------------------------------------------
class B1Opportunity:
    """B1 买点：超卖 + 小实体 + 低换手。

    Parameters
    ----------
    j_max:
        KDJ J value ceiling (classic: ≤ -10, negative is better).
    pct_min, pct_max:
        Allowed same-day change band (classic: -2% ~ +1.8%).
    amplitude_max:
        Maximum intraday amplitude (classic: 7%).
    cum_turnover_max, turnover_window:
        Cumulative turnover ceiling over the recent window (classic: 38%).
        Applied only when the frame carries a ``turnover`` column; otherwise the
        condition is skipped rather than approximated silently.
    """

    name = "b1_opportunity"

    def __init__(
        self,
        j_max: float = -10.0,
        pct_min: float = -0.02,
        pct_max: float = 0.018,
        amplitude_max: float = 0.07,
        cum_turnover_max: float | None = 0.38,
        turnover_window: int = 20,
        require_turnover: bool = False,
    ) -> None:
        if pct_min >= pct_max:
            raise ValueError("pct_min must be < pct_max")
        self.params: dict[str, Any] = {
            "j_max": j_max,
            "pct_min": pct_min,
            "pct_max": pct_max,
            "amplitude_max": amplitude_max,
            "cum_turnover_max": cum_turnover_max,
            "turnover_window": turnover_window,
            "require_turnover": require_turnover,
        }

    def signal(self, df: pd.DataFrame) -> pd.Series:
        j = kdj(df)["j"]
        pct = pct_change_1d(df)
        amp = amplitude(df)

        hit = (j <= self.params["j_max"]) & pct.between(self.params["pct_min"], self.params["pct_max"]) & (
            amp <= self.params["amplitude_max"]
        )

        max_turnover = self.params["cum_turnover_max"]
        if max_turnover is not None and "turnover" in df.columns:
            cum = df["turnover"].astype(float).rolling(self.params["turnover_window"], min_periods=1).sum()
            hit = hit & (cum <= max_turnover)
        elif max_turnover is not None and self.params["require_turnover"]:
            return pd.Series(False, index=df.index)

        return _to_bool(hit)

    def score(self, df: pd.DataFrame) -> pd.Series:
        return _clean(self.signal(df).astype(float), df.index)


class B1Graded:
    """B1 梯度打分（5 硬性条件 + 4 软性加分）。

    硬性条件（任一不满足 → 0 分）::

        硬1  J ≤ 13（超卖区，越低越好；注意不是 -10）
        硬2  近 15 日内存在放量日（成交量 > 前一日 × 2，"有人玩过"）
        硬3  极致缩量：当日成交量 < 近 15 日最高成交量 / 2.5
        硬4  双线多头：白线 > 黄线 且 收盘 ≥ 黄线 × 0.97（需 ≥114 根 K 线）
        硬5  前 N 低点未破：近 15 日最低 ≥ 近 30 日最低 × 0.97
        排除 S1：近 10 日内出现放量大阴线（跌幅 < -3% 且量 > 前一日 × 1.5）

    软性加分（每个 +10 分）::

        软6  涨跌幅 ∈ [-2%, +1.8%]（小阴小阳）
        软7  振幅 < 4%（(high-low)/low）
        软8  盈亏比 ≥ 3（止损=近15日最低×0.97，目标=近15日最高×1.02）
        软9  双30原则（原始规则中搁置，不计分）

    评分：硬性全过 = 60 分基础分 + 10 × 软性通过数（上限 100）→ 规则输出 /100。
    """

    name = "b1_graded"

    def __init__(
        self,
        j_max: float = 13.0,
        spike_lookback: int = 15,
        spike_multiple: float = 2.0,
        shrink_divisor: float = 2.5,
        yellow_buffer: float = 0.97,
        prev_low_buffer: float = 0.97,
        amplitude_max: float = 4.0,
        rr_min: float = 3.0,
        s1_lookback: int = 10,
        s1_pct: float = -3.0,
        s1_vol_multiple: float = 1.5,
        require_double_line: bool = True,
        min_history: int = 30,
        use_brick_filter: bool = False,
        brick_entry_max: int = 2,
        brick_block: int = 4,
    ) -> None:
        self.params: dict[str, Any] = {
            "j_max": j_max,
            "spike_lookback": spike_lookback,
            "spike_multiple": spike_multiple,
            "shrink_divisor": shrink_divisor,
            "yellow_buffer": yellow_buffer,
            "prev_low_buffer": prev_low_buffer,
            "amplitude_max": amplitude_max,
            "rr_min": rr_min,
            "s1_lookback": s1_lookback,
            "s1_pct": s1_pct,
            "s1_vol_multiple": s1_vol_multiple,
            "require_double_line": require_double_line,
            "min_history": min_history,
            "use_brick_filter": use_brick_filter,
            "brick_entry_max": brick_entry_max,
            "brick_block": brick_block,
        }

    # -- hard / soft conditions ----------------------------------------------------
    def _conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        pct = pct_change_1d(df) * 100.0

        look = int(self.params["spike_lookback"])
        j = kdj(df)["j"]
        white = white_line(df)
        yellow = yellow_line(df)

        vol_max_look = volume.rolling(look, min_periods=1).max()
        spike = volume > volume.shift(1) * self.params["spike_multiple"]
        bear_volume = (pct < self.params["s1_pct"]) & (volume > volume.shift(1) * self.params["s1_vol_multiple"])

        min_low_look = low.rolling(look, min_periods=1).min()
        min_low_prev = low.rolling(2 * look, min_periods=1).min()
        max_high_look = high.rolling(look, min_periods=1).max()

        hard = pd.DataFrame(index=df.index)
        hard["hard1_j"] = j <= self.params["j_max"]
        hard["hard2_spike"] = spike.rolling(look, min_periods=1).max().fillna(0).astype(bool)
        hard["hard3_shrink"] = volume < vol_max_look / self.params["shrink_divisor"]
        if self.params["require_double_line"]:
            hard["hard4_double_line"] = (white > yellow) & (close >= yellow * self.params["yellow_buffer"]) & yellow.notna()
        else:
            hard["hard4_double_line"] = pd.Series(True, index=df.index)
        hard["hard5_prev_low"] = min_low_look >= min_low_prev * self.params["prev_low_buffer"]
        hard["s1_excluded"] = ~bear_volume.rolling(int(self.params["s1_lookback"]), min_periods=1).max().fillna(0).astype(bool)

        stop_loss = min_low_look * self.params["prev_low_buffer"]
        target = max_high_look * 1.02
        risk = close - stop_loss
        reward = target - close
        rr = (reward / risk).where(risk > 0, np.nan).fillna(0.0)

        soft = pd.DataFrame(index=df.index)
        soft["soft6_small_candle"] = pct.between(-2.0, 1.8)
        soft["soft7_low_amplitude"] = ((high - low) / low.replace(0.0, np.nan)) * 100.0 < self.params["amplitude_max"]
        soft["soft8_risk_reward"] = rr >= self.params["rr_min"]
        soft["soft9_double_thirty"] = False  # 原始规则中搁置

        return hard, soft, rr

    def detail(self, df: pd.DataFrame) -> dict:
        """最后一根 K 线的条件明细（用于报告与排查，不参与评分）。"""
        if len(df) < self.params["min_history"]:
            return {"score": 0.0, "hard": {}, "soft": {}, "hard_all": False, "note": "数据不足"}
        hard, soft, rr = self._conditions(df)
        hard_last = {k: bool(v.iloc[-1]) for k, v in hard.items()}
        soft_last = {k: bool(v.iloc[-1]) for k, v in soft.items()}
        hard_all = all(hard_last.values())
        soft_pass = sum(1 for k, v in soft_last.items() if v and not k.startswith("soft9"))
        return {
            "score": (60 + 10 * soft_pass) / 100.0 if hard_all else 0.0,
            "hard": hard_last,
            "soft": soft_last,
            "hard_all": hard_all,
            "soft_pass": soft_pass,
            "risk_reward": float(rr.iloc[-1]) if len(rr) else 0.0,
        }

    # -- rule interface ------------------------------------------------------------
    def signal(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.params["min_history"]:
            return pd.Series(False, index=df.index)
        hard, _soft, _rr = self._conditions(df)
        out = hard.all(axis=1)
        if self.params["use_brick_filter"]:
            out = out & brick_filter_mask(
                df, entry_max=self.params["brick_entry_max"], block=self.params["brick_block"]
            )
        return _to_bool(out)

    def score(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.params["min_history"]:
            return pd.Series(0.0, index=df.index)
        hard, soft, _rr = self._conditions(df)
        hard_all = hard.all(axis=1)
        soft_pass = soft[[c for c in soft.columns if not c.startswith("soft9")]].sum(axis=1)
        score = ((60.0 + 10.0 * soft_pass) / 100.0).where(hard_all, 0.0)
        if self.params["use_brick_filter"]:
            allowed = brick_filter_mask(df, entry_max=self.params["brick_entry_max"], block=self.params["brick_block"])
            score = score.where(allowed, 0.0)
        return _clean(score, df.index)


def brick_filter_mask(df: pd.DataFrame, entry_max: int = 2, block: int = 4) -> pd.Series:
    """砖型图过滤门（来自你原来的门 1 / 门 2 口径）。

    * 门 1（入场）：只允许"红砖第 1~``entry_max`` 块"的位置进（默认 ≤2）
    * 门 2（禁买）：红砖 ≥ ``block`` 块（默认 4）一律禁买

    返回布尔序列：``True`` = 允许。
    """
    if entry_max < 1 or block < 1:
        raise ValueError("entry_max and block must be >= 1")
    chart = brick_streaks(brick_chart(df))
    red = chart["red_streak"]
    return _to_bool((red <= entry_max) & (red < block))


class BrickGreenToRed:
    """砖型图"绿转红"信号（你给的公式）：

    ``XG = 绿转红 AND 视觉红柱 >= 昨视觉绿柱 × 0.6667``

    评分做梯度化：``绿转红`` 成立时给 ``min(1, 强度比 / 0.6667)``，满足 XG 阈值给 1.0，
    这样"刚绿转红但强度不够"也能拿到部分分数，便于排序而不是非黑即白。
    """

    name = "brick_green_to_red"

    def __init__(self, ratio_threshold: float = 0.6667, require_green_to_red: bool = True) -> None:
        if ratio_threshold <= 0:
            raise ValueError("ratio_threshold must be > 0")
        self.params: dict[str, Any] = {"ratio_threshold": ratio_threshold, "require_green_to_red": require_green_to_red}

    def chart(self, df: pd.DataFrame) -> pd.DataFrame:
        return brick_streaks(brick_chart(df))

    def signal(self, df: pd.DataFrame) -> pd.Series:
        return _to_bool(self.chart(df)["xg"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        chart = self.chart(df)
        ratio = chart["strength_ratio"].astype(float)
        graded = (ratio / self.params["ratio_threshold"]).clip(upper=1.0)
        if self.params["require_green_to_red"]:
            graded = graded.where(chart["green_to_red"], 0.0)
        return _clean(graded, df.index)


class B2Confirm:
    """B2 确认：B1 之后的放量上攻。

    Classic reading: within ``b1_window`` bars of a B1 signal, price rises at
    least ``min_gain`` from the B1 close, J is still below ``j_max`` (55) and
    volume expands versus the previous bar.
    """

    name = "b2_confirm"

    def __init__(self, b1_window: int = 3, min_gain: float = 0.04, j_max: float = 55.0, require_volume_up: bool = True, b1_params: Mapping | None = None, b1_rule: str = "b1_graded") -> None:
        if b1_window < 1:
            raise ValueError("b1_window must be >= 1")
        self.params: dict[str, Any] = {
            "b1_window": b1_window,
            "min_gain": min_gain,
            "j_max": j_max,
            "require_volume_up": require_volume_up,
            "b1_rule": b1_rule,
        }
        params = dict(b1_params or {})
        self._b1: Any  # 两种 B1 规则实现交替赋值，统一按 Any 处理
        if b1_rule == "b1_graded":
            self._b1 = B1Graded(**params)
        elif b1_rule == "b1_opportunity":
            self._b1 = B1Opportunity(**params)
        else:
            raise KeyError(f"unknown b1 rule '{b1_rule}'")

    def signal(self, df: pd.DataFrame) -> pd.Series:
        b1 = self._b1.signal(df)
        close = df["close"].astype(float)
        j = kdj(df)["j"]
        volume = df["volume"].astype(float)
        prev_volume = volume.shift(1)

        out = pd.Series(False, index=df.index)
        window = int(self.params["b1_window"])
        b1_values = b1.to_numpy()
        close_values = close.to_numpy()
        j_values = j.to_numpy()
        vol_values = volume.to_numpy()
        prev_vol_values = prev_volume.to_numpy()

        for i in range(len(df)):
            start = max(0, i - window + 1)
            base = None
            for k in range(i, start - 1, -1):  # most recent B1 within the window (excluding today's price action)
                if b1_values[k] and k < i:
                    base = k
                    break
            if base is None:
                continue
            if close_values[i] / close_values[base] - 1.0 < self.params["min_gain"]:
                continue
            if j_values[i] >= self.params["j_max"]:
                continue
            if self.params["require_volume_up"] and not (vol_values[i] > prev_vol_values[i]):
                continue
            out.iloc[i] = True
        return out

    def score(self, df: pd.DataFrame) -> pd.Series:
        return _clean(self.signal(df).astype(float), df.index)


class B3Confirm:
    """B3 确认：B2 之后的十字星/小阴线 + 平开。

    Parameters
    ----------
    b2_window:
        How many bars back a B2 signal still counts.
    open_tolerance:
        ``|open / prev_close - 1|`` must be within this tolerance (平开).
    max_body:
        Maximum candle body ``|close - open| / prev_close`` to count as 十字星/小阴线.
    """

    name = "b3_confirm"

    def __init__(self, b2_window: int = 2, open_tolerance: float = 0.01, max_body: float = 0.02, b2_params: Mapping | None = None) -> None:
        if b2_window < 1:
            raise ValueError("b2_window must be >= 1")
        self.params: dict[str, Any] = {"b2_window": b2_window, "open_tolerance": open_tolerance, "max_body": max_body}
        self._b2 = B2Confirm(**dict(b2_params or {}))

    def signal(self, df: pd.DataFrame) -> pd.Series:
        b2 = self._b2.signal(df)
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        prev_close = close.shift(1)

        flat_open = (open_ / prev_close - 1.0).abs() <= self.params["open_tolerance"]
        small_body = ((close - open_).abs() / prev_close) <= self.params["max_body"]
        small_bear = (close <= open_) & small_body  # 小阴线
        doji = small_body  # 十字星
        candle_ok = doji | small_bear

        recent_b2 = b2.astype(float).rolling(int(self.params["b2_window"]), min_periods=1).max().shift(1).fillna(0.0).astype(bool)
        return _to_bool(recent_b2 & flat_open & candle_ok)

    def score(self, df: pd.DataFrame) -> pd.Series:
        return _clean(self.signal(df).astype(float), df.index)


# --------------------------------------------------------------------------------------
# 单针下 20 / 30
# --------------------------------------------------------------------------------------
class NeedleRSL:
    """单针下20 / 单针下30（相对强度定位口径）。

    ``RSL(N) = 100*(C - LLV(L,N)) / (HHV(C,N) - LLV(L,N))``

    * 单针下20：``RSL(3) ≤ 20`` 且 ``RSL(21) ≥ 80``（白线极低 + 红线极高）
    * 单针下30：``RSL(3) < 30`` 且 ``RSL(21) > 85``（阈值上移，换确定性）
    """

    name = "needle_rsl"

    def __init__(
        self,
        short_window: int = 3,
        long_window: int = 21,
        short_max: float = 20.0,
        long_min: float = 80.0,
        long_strict: bool = False,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.params: dict[str, Any] = {
            "short_window": short_window,
            "long_window": long_window,
            "short_max": short_max,
            "long_min": long_min,
            "long_strict": long_strict,
        }

    def signal(self, df: pd.DataFrame) -> pd.Series:
        short = rsl(df, self.params["short_window"])
        long = rsl(df, self.params["long_window"])
        if self.params["long_strict"]:
            hit = (short < self.params["short_max"]) & (long > self.params["long_min"])
        else:
            hit = (short <= self.params["short_max"]) & (long >= self.params["long_min"])
        return _to_bool(hit)

    def score(self, df: pd.DataFrame) -> pd.Series:
        return _clean(self.signal(df).astype(float), df.index)


# --------------------------------------------------------------------------------------
# 量价齐升 V3
# --------------------------------------------------------------------------------------
class VolumePriceV3:
    """量价齐升 V3：五条硬性条件全满足才给分。

    1. 连续 2 日阳线，且收盘价连续创新高；
    2. 连续 2 日成交量递增；
    3. 当日涨幅 2% ~ 6%；
    4. 白线 > 黄线（双线多头）且 收盘 ≥ 黄线 × 0.97；
    5. J < 60（未超买）。

    评分：硬性全满足得 0.70；涨幅落在 3%~5% 加 0.10；量 > 昨日 1.5 倍加 0.10；
    J < 50 加 0.10，上限 1.0。
    """

    name = "volume_price_v3"

    def __init__(
        self,
        pct_min: float = 0.02,
        pct_max: float = 0.06,
        golden_low: float = 0.03,
        golden_high: float = 0.05,
        j_max: float = 60.0,
        j_strong: float = 50.0,
        yellow_buffer: float = 0.97,
        volume_surge_multiple: float = 1.5,
        base_score: float = 0.70,
    ) -> None:
        if pct_min >= pct_max:
            raise ValueError("pct_min must be < pct_max")
        self.params: dict[str, Any] = {
            "pct_min": pct_min,
            "pct_max": pct_max,
            "golden_low": golden_low,
            "golden_high": golden_high,
            "j_max": j_max,
            "j_strong": j_strong,
            "yellow_buffer": yellow_buffer,
            "volume_surge_multiple": volume_surge_multiple,
            "base_score": base_score,
        }

    def signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        pct = close.pct_change()
        j = kdj(df)["j"]
        white = white_line(df)
        yellow = yellow_line(df)

        bull2 = (close > open_) & (close > open_).shift(1).fillna(False) & (close > close.shift(1)) & (
            close.shift(1) > close.shift(2)
        )
        vol2 = (volume > volume.shift(1)) & (volume.shift(1) > volume.shift(2))
        pct_ok = pct.between(self.params["pct_min"], self.params["pct_max"])
        lines_ok = (white > yellow) & (close >= yellow * self.params["yellow_buffer"])
        j_ok = j < self.params["j_max"]

        return _to_bool(bull2 & vol2 & pct_ok & lines_ok & j_ok)

    def score(self, df: pd.DataFrame) -> pd.Series:
        hit = self.signal(df)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        pct = close.pct_change()
        j = kdj(df)["j"]

        score = pd.Series(0.0, index=df.index)
        score = score.where(~hit, self.params["base_score"])
        bonus_range = hit & pct.between(self.params["golden_low"], self.params["golden_high"])
        bonus_volume = hit & (volume > volume.shift(1) * self.params["volume_surge_multiple"])
        bonus_j = hit & (j < self.params["j_strong"])
        score = score + bonus_range.astype(float) * 0.10 + bonus_volume.astype(float) * 0.10 + bonus_j.astype(float) * 0.10
        return _clean(score, df.index)


# --------------------------------------------------------------------------------------
# 0AMV 活跃市值 + 波段开关
# --------------------------------------------------------------------------------------
@dataclass
class ActiveMarketValueGate:
    """0AMV 活跃市值指数与波段开关。

    活筹置换衰减模型（ρ = 0.92）::

        A_i,t = A_i,t-1 × ρ × (1 - turnover_i,t) + vol_i,t     # 停牌日原样保留
        0AMV_t = Σ_i A_i,t × close_i,t
        日涨跌 = 0AMV_t / 0AMV_t-1 - 1

    开关规则（波段）::

        空仓 → 开仓：单日 ≥ +5%（特别强）/ 单日 ≥ +4%（强）/
                    连续 2 日合计 ≥ +4% 且窗口内无 ≤ -2.3% 日（一般）/
                    连续 3 日合计 ≥ +4% 且窗口内无 ≤ -2.3% 日（较弱）
        持仓 → 关仓：单日 ≤ -2.3%（当日为波段最后一天，次日只卖不买）

    换手率口径：优先使用帧内的 ``turnover`` 列；缺失时按
    ``流通股本 ≈ 250 日最大成交量 × 1.5`` 兜底估计（并在 ``notes`` 中标注为估计值）。
    """

    rho: float = 0.92
    strong: float = 0.04
    very_strong: float = 0.05
    close_threshold: float = -0.023
    normal_days: int = 2
    weak_days: int = 3
    min_history: int = 30
    name: str = "active_market_value_gate"
    params: dict = field(init=False, default_factory=dict)
    notes: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.rho < 1:
            raise ValueError("rho must be in (0, 1)")
        if not self.close_threshold < 0 < self.strong < self.very_strong:
            raise ValueError("require close_threshold < 0 < strong < very_strong")
        if self.normal_days < 1 or self.weak_days < 1:
            raise ValueError("window lengths must be >= 1")
        self.params: dict[str, Any] = {
            "rho": self.rho,
            "strong": self.strong,
            "very_strong": self.very_strong,
            "close_threshold": self.close_threshold,
            "normal_days": self.normal_days,
            "weak_days": self.weak_days,
        }

    # -- index construction --------------------------------------------------------
    def _turnover(self, df: pd.DataFrame, volume_shares: pd.Series) -> pd.Series:
        if "turnover" in df.columns:
            note = "turnover: 使用数据中的真实换手率列"
            if note not in self.notes:
                self.notes.append(note)
            return df["turnover"].astype(float).clip(0.0, 1.0)
        float_shares = max(float(volume_shares.rolling(250, min_periods=20).max().iloc[-1]) * 1.5, 1.0)
        note = "turnover: 数据缺少换手率列，按 250 日最大成交量×1.5 估计流通股本"
        if note not in self.notes:
            self.notes.append(note)
        return (volume_shares / float_shares).clip(0.0, 1.0)

    def active_shares(self, df: pd.DataFrame) -> pd.Series:
        """活筹序列 A_i,t（手 → 股按 ×100 处理，与成交量口径一致）。"""
        volume = df["volume"].astype(float) * 100.0
        turnover = self._turnover(df, volume)
        a = 0.0
        initialized = False
        out: list[float] = []
        for vol, tov in zip(volume.to_numpy(), turnover.to_numpy(), strict=False):
            if np.isnan(vol):
                out.append(a if initialized else np.nan)
                continue
            tov = 0.0 if np.isnan(tov) else min(max(tov, 0.0), 1.0)
            a = a * self.rho * (1.0 - tov) + vol
            initialized = True
            out.append(a)
        return pd.Series(out, index=df.index, dtype=float)

    def amv_series(self, universe: Mapping[str, pd.DataFrame]) -> pd.Series:
        """0AMV 指数序列（单位与 close×volume 一致，仅用于计算涨跌幅）。"""
        if not universe:
            raise ValueError("universe is empty")
        parts = []
        for df in universe.values():
            a = self.active_shares(df)
            close = df["close"].astype(float).ffill()
            parts.append(a * close)
        total = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        return total.dropna().sort_index()

    def pct_series(self, universe: Mapping[str, pd.DataFrame]) -> pd.Series:
        """0AMV 日涨跌幅（小数）。"""
        amv = self.amv_series(universe)
        return amv.pct_change()

    # -- band state machine ---------------------------------------------------------
    def gate_series(self, universe: Mapping[str, pd.DataFrame]) -> pd.Series:
        pct = self.pct_series(universe).dropna()
        values = pct.to_numpy()
        states: list[int] = []
        strength: list[str] = []
        open_band = False
        for i, change in enumerate(values):
            if not open_band:
                window_normal = values[max(0, i - self.normal_days + 1) : i + 1]
                window_weak = values[max(0, i - self.weak_days + 1) : i + 1]
                no_close_day = lambda w: bool(np.all(w > self.close_threshold))  # noqa: E731
                if change >= self.very_strong:
                    open_band, tag = True, "very_strong"
                elif change >= self.strong:
                    open_band, tag = True, "strong"
                elif len(window_normal) >= self.normal_days and window_normal.sum() >= self.strong and no_close_day(window_normal):
                    open_band, tag = True, "normal"
                elif len(window_weak) >= self.weak_days and window_weak.sum() >= self.strong and no_close_day(window_weak):
                    open_band, tag = True, "weak"
                else:
                    tag = "none"
            else:
                if change <= self.close_threshold:
                    open_band, tag = False, "close"
                else:
                    tag = "hold"
            states.append(1 if open_band else 0)
            strength.append(tag)
        out = pd.DataFrame({"gate": states, "trigger": strength}, index=pct.index)
        return out

    def state_at(self, universe: Mapping[str, pd.DataFrame], as_of: str | pd.Timestamp | None = None) -> int:
        series = self.gate_series(universe)
        if series.empty:
            return 0
        if as_of is not None:
            series = series.loc[:as_of]
        return int(series["gate"].iloc[-1]) if len(series) else 0

    def trigger_at(self, universe: Mapping[str, pd.DataFrame], as_of: str | pd.Timestamp | None = None) -> str:
        series = self.gate_series(universe)
        if series.empty:
            return "none"
        if as_of is not None:
            series = series.loc[:as_of]
        return str(series["trigger"].iloc[-1]) if len(series) else "none"


PERSONAL_RULES = {
    B1Opportunity.name: B1Opportunity,
    B1Graded.name: B1Graded,
    B2Confirm.name: B2Confirm,
    B3Confirm.name: B3Confirm,
    NeedleRSL.name: NeedleRSL,
    VolumePriceV3.name: VolumePriceV3,
    BrickGreenToRed.name: BrickGreenToRed,
}


def build_personal_rule(name: str, **params):
    if name not in PERSONAL_RULES:
        raise KeyError(f"unknown personal rule '{name}'; available: {sorted(PERSONAL_RULES)}")
    return PERSONAL_RULES[name](**params)
