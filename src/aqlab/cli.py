"""Command line interface.

Examples
--------
``python -m aqlab.cli demo``                     offline demo on synthetic data
``python -m aqlab.cli screen --top 5``           cross-sectional screening demo
``python -m aqlab.cli run --csv data/raw/600519.csv --strategy ma_cross --params fast=10,slow=30``
``python -m aqlab.cli fetch --source akshare --symbol 600519 --start 2022-01-01 --end 2024-12-31``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aqlab.backtest import BacktestConfig, run_backtest
from aqlab.data import generate_synthetic_ohlcv, load_ohlcv_csv, make_universe
from aqlab.metrics import compute_metrics, format_metrics
from aqlab.report import write_report
from aqlab.screen import ScreenConfig, rank_universe
from aqlab.strategies import STRATEGIES, build_strategy

DEFAULT_OUT = Path("output")


def _parse_params(text: str | None) -> dict:
    """Parse ``k=v,k2=v2`` into a typed dict."""
    params: dict = {}
    if not text:
        return params
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"bad --params item '{chunk}', expected key=value")
        key, value = chunk.split("=", 1)
        value = value.strip()
        try:
            params[key.strip()] = int(value)
        except ValueError:
            try:
                params[key.strip()] = float(value)
            except ValueError:
                params[key.strip()] = value
    return params


def cmd_demo(args: argparse.Namespace) -> int:
    df = generate_synthetic_ohlcv(n_days=args.days, seed=args.seed)
    config = BacktestConfig(fee_bps=3.0, slippage_bps=2.0)
    out_root = Path(args.out)

    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    scenario = "上涨" if bh > 0 else "下跌"
    print(f"合成数据（seed={args.seed}，{scenario}行情 {bh * 100:+.1f}%）：{df.index[0].date()} ~ {df.index[-1].date()}（{len(df)} 个交易日）")
    print()
    summary = []
    for name in ("buy_and_hold", "ma_cross", "momentum", "mean_reversion"):
        strategy = build_strategy(name)
        result = run_backtest(df, strategy.positions(df), config=config, name=strategy.name)
        metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
        summary.append({"strategy": name, **{k: metrics.get(k) for k in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "trades", "win_rate_trade")}})
        write_report(out_root / "demo" / name, name, result, metrics, extras={"strategy": name, **strategy.params})

    import pandas as pd

    table = pd.DataFrame(summary)
    for col in ("total_return", "cagr", "ann_vol", "max_drawdown", "win_rate_trade"):
        table[col] = (table[col] * 100).round(2)
    table = table.rename(
        columns={
            "strategy": "策略",
            "total_return": "总收益%",
            "cagr": "年化%",
            "ann_vol": "年化波动%",
            "sharpe": "Sharpe",
            "max_drawdown": "最大回撤%",
            "trades": "交易数",
            "win_rate_trade": "胜率%",
        }
    )
    print(table.to_markdown(index=False))
    print(f"\n报告已写入：{out_root / 'demo'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    df = load_ohlcv_csv(args.csv)
    strategy = build_strategy(args.strategy, **_parse_params(args.params))
    config = BacktestConfig(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        initial_cash=args.cash,
        allow_short=args.allow_short,
    )
    result = run_backtest(df, strategy.positions(df), config=config, name=strategy.name)
    metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
    print(f"{args.csv}  {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df)} bars)")
    print()
    print(format_metrics(metrics))
    out = Path(args.out) / f"{strategy.name}_{Path(args.csv).stem}"
    paths = write_report(out, str(args.csv), result, metrics, extras={"strategy": args.strategy, **strategy.params})
    print(f"\n报告已写入：{paths['report']}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    universe = make_universe(n_symbols=args.symbols, n_days=args.days, seed=args.seed)
    table = rank_universe(universe, config=ScreenConfig(top_n=args.top, min_history=args.min_history))
    cols = ["rank", "symbol", "close", "mom_20", "mom_60", "trend_gap", "vol_20", "rsi_14", "score"]
    view = table[cols].copy()
    for col in ("mom_20", "mom_60", "trend_gap", "vol_20"):
        view[col] = view[col].round(4)
    view["rsi_14"] = view["rsi_14"].round(1)
    view["close"] = view["close"].round(2)
    view["score"] = view["score"].round(3)
    print(f"截面筛选（合成票池 {len(universe)} 只，按加权因子打分排序）")
    print()
    print(view.head(args.top).to_markdown(index=False))
    if args.out:
        out = Path(args.out) / "screen"
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "screen.csv", index=False, encoding="utf-8-sig")
        print(f"\n明细已写入：{out / 'screen.csv'}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if args.source == "tushare":
        from aqlab.data import fetch_tushare as fetch
    else:
        from aqlab.data import fetch_akshare as fetch
    df = fetch(args.symbol, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, encoding="utf-8-sig")
    print(f"{args.source} 数据已保存：{out}（{len(df)} 行）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aqlab", description="A-share quant lab: screening + backtesting")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="run the offline synthetic demo for all built-in strategies")
    p_demo.add_argument("--seed", type=int, default=7, help="synthetic data seed (7 = bear scenario, 25 = bull scenario)")
    p_demo.add_argument("--days", type=int, default=750)
    p_demo.add_argument("--out", default=str(DEFAULT_OUT))
    p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="backtest a strategy on a CSV file")
    p_run.add_argument("--csv", required=True)
    p_run.add_argument("--strategy", default="ma_cross", choices=sorted(STRATEGIES))
    p_run.add_argument("--params", default=None, help="e.g. fast=10,slow=30")
    p_run.add_argument("--fee-bps", type=float, default=3.0)
    p_run.add_argument("--slippage-bps", type=float, default=2.0)
    p_run.add_argument("--cash", type=float, default=1_000_000.0)
    p_run.add_argument("--allow-short", action="store_true")
    p_run.add_argument("--out", default=str(DEFAULT_OUT))
    p_run.set_defaults(func=cmd_run)

    p_screen = sub.add_parser("screen", help="cross-sectional screening demo on a synthetic universe")
    p_screen.add_argument("--symbols", type=int, default=30)
    p_screen.add_argument("--days", type=int, default=500)
    p_screen.add_argument("--seed", type=int, default=11)
    p_screen.add_argument("--top", type=int, default=10)
    p_screen.add_argument("--min-history", type=int, default=130)
    p_screen.add_argument("--out", default=str(DEFAULT_OUT))
    p_screen.set_defaults(func=cmd_screen)

    p_fetch = sub.add_parser("fetch", help="download daily bars via tushare/akshare")
    p_fetch.add_argument("--source", choices=["tushare", "akshare"], default="akshare")
    p_fetch.add_argument("--symbol", required=True)
    p_fetch.add_argument("--start", required=True)
    p_fetch.add_argument("--end", required=True)
    p_fetch.add_argument("--out", default="data/raw/bars.csv")
    p_fetch.set_defaults(func=cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
