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
    out = out[[c for c in OHLCV_COLUMNS if c in out.columns] + [c for c in ("turnover", "amount", "pct_change") if c in out.columns]]
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


def fetch_tushare_turnover(symbol: str, start: str, end: str, token: str | None = None) -> pd.DataFrame:
    """Fetch the real turnover rate from Tushare ``daily_basic``.

    Returns a frame indexed by ``date`` with a single ``turnover`` column as a
    *fraction* (Tushare reports percent). Needed by the 0AMV active-share model and
    by the simplified B1 rule; both work without it, but then the turnover input is
    an estimate (and the reports say so).
    """
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the data extra: pip install 'ashare-quant-lab[data]'") from exc

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("a Tushare token is required (argument or TUSHARE_TOKEN env var)")
    pro = ts.pro_api(token)
    df = pro.daily_basic(
        ts_code=symbol,
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        fields="trade_date,turnover_rate",
    )
    if df is None or df.empty:
        raise ValueError(f"no turnover data returned for {symbol}")
    df = df.rename(columns={"trade_date": "date", "turnover_rate": "turnover"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0
    return df[["turnover"]].dropna()


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


class TushareDataSource:
    """A cached Tushare-backed data source for the daily pipeline.

    Behaviour that matters for a scheduled job:

    * every symbol is cached as CSV under ``cache_dir`` (so a rerun or a network
      outage never loses yesterday's data);
    * if a fetch fails **and** a cache exists, the cached frame is used and
      ``self.degraded`` becomes ``True`` with ``self.last_error`` set — the daily
      report can then say "数据降级" instead of silently pretending everything is fine;
    * ``fetch_fn`` is injectable, which keeps the whole class testable offline.
    * ``with_turnover=True`` additionally merges the real ``daily_basic.turnover_rate``
      (via ``turnover_fn``); if that call fails the frame is returned without the
      column and ``turnover_available`` becomes ``False`` — reports must then say the
      turnover input is missing, never pretend.
    """

    def __init__(
        self,
        symbols,
        start: str,
        end: str,
        token: str | None = None,
        cache_dir: str | os.PathLike = "data/raw",
        fetch_fn=None,
        turnover_fn=None,
        with_turnover: bool = False,
    ) -> None:
        self._symbols = [str(s) for s in symbols]
        if not self._symbols:
            raise ValueError("symbols must not be empty")
        self.start = start
        self.end = end
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetch = fetch_fn or fetch_tushare
        self._turnover_fn = turnover_fn or (fetch_tushare_turnover if with_turnover else None)
        self.with_turnover = bool(with_turnover)
        self.turnover_available: bool | None = None
        self.degraded = False
        self.last_error: str | None = None
        self._cache: dict[str, pd.DataFrame] = {}

    def symbols(self) -> list[str]:
        return list(self._symbols)

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.replace('.', '_')}.csv"

    def _attach_turnover(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the real turnover rate when enabled; degrade loudly on failure."""
        if not self.with_turnover or self._turnover_fn is None:
            return df
        try:
            turnover = self._turnover_fn(symbol, self.start, self.end, token=self.token)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self.turnover_available = False
            self.last_error = self.last_error or f"{symbol} turnover: {type(exc).__name__}: {exc}"
            return df
        if turnover is None or len(turnover) == 0:
            self.turnover_available = False
            return df
        # 真实接口（daily_basic）返回 DataFrame；测试桩可能返回 Series，两者都要支持
        if isinstance(turnover, pd.DataFrame):
            if "turnover" not in turnover.columns:
                self.turnover_available = False
                return df
            series = turnover["turnover"].astype(float)
        else:
            series = turnover.astype(float).rename("turnover")
        merged = df.join(series.rename("turnover"), how="left")
        self.turnover_available = bool(merged["turnover"].notna().any())
        return merged

    def bars(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol]
        path = self._cache_path(symbol)
        if path.exists():
            df = load_ohlcv_csv(path)
            if "turnover" in df.columns and self.turnover_available is None:
                self.turnover_available = bool(df["turnover"].notna().any())
            self._cache[symbol] = df
            return df
        try:
            df = self._fetch(symbol, self.start, self.end, token=self.token)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self.last_error = f"{symbol}: {type(exc).__name__}: {exc}"
            self.degraded = True
            raise KeyError(f"无法获取 {symbol} 且无本地缓存（{self.last_error}）") from exc
        df = self._attach_turnover(symbol, df)
        df.to_csv(path, encoding="utf-8-sig")
        self._cache[symbol] = df
        return df

    def describe(self) -> dict:
        return {
            "source": "tushare",
            "symbols": self.symbols(),
            "start": self.start,
            "end": self.end,
            "cache_dir": str(self.cache_dir),
            "with_turnover": self.with_turnover,
            "turnover_available": self.turnover_available,
            "cached": sorted(p.stem for p in self.cache_dir.glob("*.csv"))[:20],
        }

