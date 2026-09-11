"""Data layer: normalization, deterministic synthetic data, optional live fetchers.

Design goals
------------
* Everything downstream consumes the same normalized schema:
  a DatetimeIndex named ``date`` and float columns ``open, high, low, close, volume``.
* Unit tests and demos run fully offline (synthetic generator).
* Live data (Tushare / AkShare) is *optional* and lazily imported, so the core
  package has no hard dependency on a data vendor.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_REQUIRED = {"open", "high", "low", "close"}

# Common Chinese column names from A-share data vendors / exported spreadsheets.
_CN_RENAME = {
    "日期": "date",
    "时间": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
}


def normalize_ohlcv(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with a sorted DatetimeIndex and lower-case OHLCV columns.

    Recognises the common Chinese column names (开盘/最高/最低/收盘/成交量/日期)
    in addition to the English ones, so exported A-share CSVs work as-is.
    """
    out = df.copy()
    out = out.rename(columns=_CN_RENAME)
    if date_col is not None:
        out = out.rename(columns={date_col: "date"})
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.columns = [str(c).lower() for c in out.columns]
    if "vol" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"vol": "volume"})
    missing = _REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"missing required OHLC columns: {sorted(missing)}")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out[[c for c in OHLCV_COLUMNS if c in out.columns]]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return out


def generate_synthetic_ohlcv(
    n_days: int = 750,
    seed: int = 7,
    start: str = "2022-01-03",
    s0: float = 100.0,
    mu: float = 0.0004,
    sigma: float = 0.018,
    regime_strength: float = 0.6,
    regime_period: int = 180,
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV with a slow trend regime.

    The regime term (a sine wave added to the drift) makes trend/momentum
    strategies behave non-trivially while keeping the series fully reproducible:
    the same ``seed`` always yields exactly the same frame.
    """
    if n_days < 10:
        raise ValueError("n_days must be >= 10")
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    regime = regime_strength * np.sin(2 * np.pi * t / float(regime_period)) * 0.0015
    rets = rng.normal(mu, sigma, n_days) + regime
    close = s0 * np.exp(np.cumsum(rets))

    prev_close = np.concatenate([[s0], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.003, n_days))
    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n_days)))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n_days)))
    volume = rng.lognormal(mean=14.0, sigma=0.3, size=n_days)

    dates = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def load_ohlcv_csv(path: str | os.PathLike, date_col: str | None = None) -> pd.DataFrame:
    """Load a CSV file into the normalized schema."""
    df = pd.read_csv(Path(path))
    return normalize_ohlcv(df, date_col=date_col)


def make_universe(n_symbols: int = 12, n_days: int = 750, seed: int = 11, start: str = "2022-01-03"):
    """Build a deterministic universe of synthetic symbols for screening demos.

    Returns ``{symbol: DataFrame}`` with symbols ``SYN001``, ``SYN002``, ...
    Each symbol gets a different seed and drift so cross-sectional ranking is
    meaningful.
    """
    universe = {}
    for i in range(n_symbols):
        sym = f"SYN{i + 1:03d}"
        universe[sym] = generate_synthetic_ohlcv(
            n_days=n_days,
            seed=seed + 17 * i,
            start=start,
            mu=0.0002 + 0.00012 * i,
            sigma=0.014 + 0.001 * (i % 5),
        )
    return universe


# --------------------------------------------------------------------------------------
# Optional live data. Imported lazily; never required by tests or the demo.
# --------------------------------------------------------------------------------------
def fetch_tushare(
    symbol: str,
    start: str,
    end: str,
    token: str | None = None,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch daily bars from Tushare Pro (requires a token and the ``tushare`` extra).

    ``symbol`` follows the Tushare convention, e.g. ``000001.SZ``.
    The token falls back to the ``TUSHARE_TOKEN`` environment variable.
    """
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the data extra: pip install 'ashare-quant-lab[data]'") from exc

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("a Tushare token is required (argument or TUSHARE_TOKEN env var)")
    pro = ts.pro_api(token)
    df = pro.daily(ts_code=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    if df is None or df.empty:
        raise ValueError(f"no data returned for {symbol}")
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df = normalize_ohlcv(df)
    return df.sort_index()


def fetch_akshare(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch daily bars from AkShare (requires the ``data`` extra).

    ``symbol`` is a plain 6-digit A-share code, e.g. ``600519``.
    """
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the data extra: pip install 'ashare-quant-lab[data]'") from exc

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust=adjust,
    )
    rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    df = df.rename(columns=rename)
    return normalize_ohlcv(df)
