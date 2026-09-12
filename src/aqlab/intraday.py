"""开盘量比确认（v0.8）：B1 只说明"超跌"，是否买入还要看有没有增量资金进来。

口径（可配置）：

* **开盘窗口**：开盘后前 ``window_minutes`` 分钟（默认 7 分钟）；
* **量比**：当日开盘窗口累计成交量 ÷ 过去 ``baseline_days`` 日（默认 5）同一窗口累计成交量的均值；
* **判定**：量比 ≥ ``min_ratio``（默认 1.0）→ 有增量资金，放行买入；否则观望；
* **缺失即弃答**：次日分钟数据缺失 / 没有次日数据 → 明确写"无法判断"，**不猜**。

因果性：``t`` 日收盘后产生的 B1 信号，用 ``t+1`` 日开盘前 7 分钟的量比决定是否买入——
信号日与决策日分离，不存在未来函数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "IntradayConfig",
    "opening_window_volume",
    "opening_window_price",
    "opening_volume_ratio",
    "standard_volume_ratio",
    "confirm_signals",
    "confirmed_signal_series",
    "generate_synthetic_minutes",
]

_SESSION_MINUTES = 240  # 09:30-11:30 + 13:00-15:00


@dataclass
class IntradayConfig:
    """开盘窗口与量比阈值。"""

    window_minutes: int = 7
    baseline_days: int = 5
    min_ratio: float = 1.0
    session_minutes: int = 240          # 一个交易日有多少分钟（软件量比的分母口径）

    def __post_init__(self) -> None:
        if self.window_minutes < 1:
            raise ValueError("window_minutes must be >= 1")
        if self.baseline_days < 1:
            raise ValueError("baseline_days must be >= 1")
        if self.min_ratio <= 0:
            raise ValueError("min_ratio must be > 0")
        if self.session_minutes < 1:
            raise ValueError("session_minutes must be >= 1")


def _window_frame(minute_bars: pd.DataFrame, config: IntradayConfig) -> pd.DataFrame:
    """准备分钟数据并切出每日开盘窗口内的那几根 K 线。"""
    frame = minute_bars.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "minute" in frame.columns:
            frame = frame.set_index("minute")
        else:
            raise ValueError("minute_bars must be indexed by datetime (or contain a 'minute' column)")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["_date"] = frame.index.normalize()
    frame["_rank"] = frame.groupby("_date").cumcount()
    return frame[frame["_rank"] < config.window_minutes]


def opening_window_volume(minute_bars: pd.DataFrame, config: IntradayConfig | None = None) -> pd.Series:
    """每个交易日在开盘窗口内的累计成交量。"""
    config = config or IntradayConfig()
    if minute_bars.empty or "volume" not in minute_bars.columns:
        raise ValueError("minute_bars must contain a 'volume' column")
    return _window_frame(minute_bars, config).groupby("_date")["volume"].sum().sort_index()


def opening_window_price(minute_bars: pd.DataFrame, config: IntradayConfig | None = None) -> pd.Series:
    """每个交易日开盘窗口**最后一根**分钟 K 的收盘价（默认 7 分钟即 09:37）。

    这是"看到量比之后立刻下单"的成交价，比用当日收盘价更贴近实盘，也避免用信号日收盘价
    去评估一个次日才做出的决策。
    """
    config = config or IntradayConfig()
    if minute_bars.empty or "close" not in minute_bars.columns:
        raise ValueError("minute_bars must contain a 'close' column")
    return _window_frame(minute_bars, config).groupby("_date")["close"].last().sort_index()


def opening_volume_ratio(minute_bars: pd.DataFrame, config: IntradayConfig | None = None) -> pd.Series:
    """量比序列（**相对口径**）：当日窗口量 ÷ 过去 ``baseline_days`` 日**同一窗口**量均值（不含当日）。

    这是"相对自己最近几个早晨"的增量口径，回答的是"今天的开盘比平时更活跃吗"。
    行情软件上显示的"量比"是另一个口径，见 :func:`standard_volume_ratio`。
    """
    config = config or IntradayConfig()
    window_volume = opening_window_volume(minute_bars, config)
    baseline = window_volume.shift(1).rolling(config.baseline_days, min_periods=config.baseline_days).mean()
    return (window_volume / baseline.replace(0.0, np.nan)).rename("volume_ratio")


def standard_volume_ratio(
    minute_bars: pd.DataFrame,
    daily: pd.DataFrame | pd.Series,
    config: IntradayConfig | None = None,
) -> pd.Series:
    """量比序列（**软件口径**）：当日开盘窗口每分钟均量 ÷ 过去 ``baseline_days`` 日**全天**每分钟均量。

    ``量比 = (窗口成交量 / 窗口分钟数) / (过去 N 日日均成交量 / session_minutes)``

    这个口径与行情软件屏幕上显示的"量比"一致：普通股票在 1.5~2.5 之间，
    有异动的热门股才会到 4 以上。两个口径不能混用阈值。
    ``daily`` 可以是日线 DataFrame（含 ``volume`` 列）或日成交量 Series；索引为交易日。
    """
    config = config or IntradayConfig()
    window_volume = opening_window_volume(minute_bars, config)
    if isinstance(daily, pd.Series):
        daily_volume = daily.astype(float)
    else:
        if "volume" not in daily.columns:
            raise ValueError("daily must contain a 'volume' column")
        daily_volume = daily["volume"].astype(float)
    daily_volume = daily_volume.copy()
    daily_volume.index = pd.to_datetime(daily_volume.index)
    daily_volume = daily_volume.sort_index()
    baseline = daily_volume.rolling(config.baseline_days, min_periods=config.baseline_days).mean().shift(1)
    baseline = baseline.reindex(window_volume.index)
    ratio = (window_volume / config.window_minutes) / (baseline / config.session_minutes)
    return ratio.replace([np.inf, -np.inf], np.nan).rename("standard_volume_ratio")



def confirm_signals(
    daily_signal: pd.Series,
    ratio: pd.Series,
    config: IntradayConfig | None = None,
) -> pd.DataFrame:
    """把日线信号（如 B1）与**次日**开盘量比结合，输出逐信号的决策表。"""
    config = config or IntradayConfig()
    if daily_signal is None or daily_signal.empty:
        return pd.DataFrame(columns=["signal_date", "decision_date", "volume_ratio", "decision", "reason"])
    signal = daily_signal.fillna(False).astype(bool)
    signal.index = pd.to_datetime(signal.index)
    ratio = ratio if ratio is not None else pd.Series(dtype=float)
    ratio = ratio.copy()
    ratio.index = pd.to_datetime(ratio.index)

    rows: list[dict] = []
    for signal_date in signal.index[signal.to_numpy()]:
        later = ratio.index[ratio.index > signal_date]
        if len(later) == 0:
            rows.append(
                {
                    "signal_date": str(signal_date.date()),
                    "decision_date": None,
                    "volume_ratio": np.nan,
                    "decision": "无法判断",
                    "reason": "没有次日分钟数据",
                }
            )
            continue
        decision_date = later[0]
        value = float(ratio.loc[decision_date])
        if value != value:
            rows.append(
                {
                    "signal_date": str(signal_date.date()),
                    "decision_date": str(decision_date.date()),
                    "volume_ratio": np.nan,
                    "decision": "无法判断",
                    "reason": "次日开盘窗口数据不足（历史窗口未满）",
                }
            )
            continue
        buy = value >= config.min_ratio
        rows.append(
            {
                "signal_date": str(signal_date.date()),
                "decision_date": str(decision_date.date()),
                "volume_ratio": round(value, 3),
                "decision": "买入" if buy else "观望",
                "reason": (
                    f"开盘 {config.window_minutes} 分钟量比 {value:.2f} ≥ {config.min_ratio:g}，有增量资金"
                    if buy
                    else f"开盘 {config.window_minutes} 分钟量比 {value:.2f} < {config.min_ratio:g}，无量能确认"
                ),
            }
        )
    return pd.DataFrame(rows)


def confirmed_signal_series(
    daily_signal: pd.Series,
    ratio: pd.Series,
    config: IntradayConfig | None = None,
) -> pd.Series:
    """把"确认后"的信号映射到**决策日**（可直接喂给回测/组合层）。"""
    decisions = confirm_signals(daily_signal, ratio, config)
    index = pd.to_datetime(ratio.index)
    out = pd.Series(False, index=index)
    for record in decisions.to_dict("records"):
        if record["decision"] == "买入" and record["decision_date"]:
            stamp = pd.Timestamp(record["decision_date"])
            if stamp in out.index:
                out.loc[stamp] = True
    return out


def generate_synthetic_minutes(
    daily: pd.DataFrame,
    seed: int = 7,
    noise: float = 0.25,
    first_minutes_boost: float = 1.0,
) -> pd.DataFrame:
    """按日线成交量为每日生成 240 根分钟线（U 形分布 + 噪声），用于离线验证量比逻辑。

    ``first_minutes_boost`` 可放大开盘前几分钟的量（模拟"有增量资金"的场景）。
    """
    if daily.empty or "volume" not in daily.columns:
        raise ValueError("daily must contain a 'volume' column")
    rng = np.random.default_rng(seed)
    minutes = np.arange(_SESSION_MINUTES)
    # U 形：开盘/收盘放量，中间缩量
    base_shape = 0.4 + 1.6 * ((minutes - _SESSION_MINUTES / 2) ** 2) / ((_SESSION_MINUTES / 2) ** 2)
    base_shape = base_shape / base_shape.sum()
    if first_minutes_boost != 1.0:
        base_shape[:10] = base_shape[:10] * first_minutes_boost
        base_shape = base_shape / base_shape.sum()

    rows: list[dict] = []
    for date, bar in daily.iterrows():
        day = pd.Timestamp(date).normalize()
        weights = base_shape * (1 + rng.normal(0, noise, _SESSION_MINUTES)).clip(0.05, None)
        weights = weights / weights.sum()
        volumes = weights * float(bar["volume"])
        stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=_SESSION_MINUTES, freq="1min")
        for stamp, volume in zip(stamps, volumes):
            rows.append({"minute": stamp, "close": float(bar["close"]), "volume": float(volume)})
    out = pd.DataFrame(rows).set_index("minute")
    out.index.name = "minute"
    return out
