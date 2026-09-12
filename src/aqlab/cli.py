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
import os
import sys
from pathlib import Path

from aqlab.backtest import BacktestConfig, run_backtest
from aqlab.data import generate_synthetic_ohlcv, load_ohlcv_csv, make_universe
from aqlab.metrics import compute_metrics, format_metrics
from aqlab.report import write_report
from aqlab.screen import ScreenConfig, rank_universe
from aqlab.strategies import STRATEGIES, build_strategy
from aqlab.tables import markdown_table

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
    print(markdown_table(table))
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
    print(markdown_table(view.head(args.top)))
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


def cmd_eval(args: argparse.Namespace) -> int:
    import json

    from aqlab.agent import OpenAICompatClient
    from aqlab.evaluation import default_tasks, format_eval_report, run_eval
    from aqlab.tools import default_registry

    registry = default_registry(args.data_dir)
    tasks = default_tasks(registry, symbol=args.symbol)

    factory = None
    if args.mode == "live":
        try:
            client = OpenAICompatClient(model=args.model)
        except ValueError as exc:
            print(f"[live 模式需要 API key] {exc}", file=sys.stderr)
            return 2
        factory = lambda _task: client  # noqa: E731 - one client reused across tasks

    report = run_eval(tasks, registry, client_factory=factory, max_steps=args.max_steps)
    print(format_eval_report(report))

    if args.out:
        out = Path(args.out) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(format_eval_report(report), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入：{out}")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    import json

    from aqlab.agent import OpenAICompatClient, ResearchAgent
    from aqlab.tools import default_registry

    registry = default_registry(args.data_dir)
    try:
        client = OpenAICompatClient(model=args.model)
    except ValueError as exc:
        print(
            f"[需要 API key] {exc}\n提示：可先用 `python -m aqlab.cli eval --mode offline` 查看离线评测。",
            file=sys.stderr,
        )
        return 2

    agent = ResearchAgent(registry, client, max_steps=args.max_steps)
    result = agent.run(args.question)
    print(result.answer or "（模型没有给出最终答案）")
    print()
    print(f"停止原因：{result.stopped_reason} ｜ 步骤：{len(result.steps)} ｜ 工具：{','.join(result.tool_names_used) or '-'}")
    if args.trace:
        out = Path(args.trace)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"trace 已写入：{out}")
    return 0


def _apply_rule_overrides(bindings, overrides):
    """Apply ``--rule rule.param=value`` overrides onto the default bindings."""
    updated = [(name, dict(params), weight) for name, params, weight in bindings]
    index = {name: i for i, (name, _p, _w) in enumerate(updated)}
    for item in overrides or []:
        if "." not in item or "=" not in item:
            raise SystemExit(f"bad --rule '{item}', expected rule_name.param=value")
        target, value = item.split("=", 1)
        rule_name, param = target.split(".", 1)
        if rule_name not in index:
            raise SystemExit(f"unknown rule '{rule_name}'; available: {sorted(index)}")
        try:
            parsed: object = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        name, params, weight = updated[index[rule_name]]
        params[param] = parsed
        updated[index[rule_name]] = (name, params, weight)
    return updated


def cmd_daily(args: argparse.Namespace) -> int:
    import pandas as pd

    from aqlab.notify import FeishuWebhookNotifier
    from aqlab.pipeline import DailyConfig, DailyPipeline, render_markdown, write_daily_report
    from aqlab.profiles import build_gate, load_profile
    from aqlab.rules import DEFAULT_RULE_BINDINGS
    from aqlab.tools import CsvDataSource, SyntheticDataSource

    if args.tushare:
        from aqlab.data import TushareDataSource

        if not args.symbols:
            print("--tushare 需要同时提供 --symbols 600519.SH,000001.SZ", file=sys.stderr)
            return 2
        source = TushareDataSource(
            symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
            start=args.start or "2020-01-01",
            end=args.end or args.date or str(pd.Timestamp.today().date()),
            with_turnover=bool(args.turnover or os.environ.get("AQLAB_TURNOVER")),
        )
    elif args.data_dir:
        source = CsvDataSource(args.data_dir)
    else:
        source = SyntheticDataSource(n_symbols=args.symbols_count, n_days=args.days, seed=args.seed)

    base_bindings = load_profile(args.profile) if args.profile else list(DEFAULT_RULE_BINDINGS)
    bindings = _apply_rule_overrides(base_bindings, args.rule)
    gate_name = "none" if args.no_gate else args.gate
    pipeline = DailyPipeline(
        source=source,
        rule_bindings=bindings,
        gate=build_gate(gate_name),
        config=DailyConfig(top_n=args.top, as_of=args.date, use_gate=gate_name != "none"),
        notifier=FeishuWebhookNotifier(webhook_url=args.webhook, dry_run=not args.push),
    )
    report = pipeline.run(push=True)
    print(render_markdown(report))

    paths = write_daily_report(Path(args.out) / "daily", report)
    print(f"\n报告已写入：{paths['markdown']}")
    if report.notifier_result:
        status = "已发送" if report.notifier_result["ok"] else "发送失败"
        print(f"推送：{status}（{report.notifier_result['channel']}）{report.notifier_result['detail']}")
    if not args.push:
        print("提示：默认 dry-run；要真正推送到飞书请加 --push 并配置 FEISHU_WEBHOOK。")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from aqlab.profiles import load_profile
    from aqlab.rules import DEFAULT_RULE_BINDINGS
    from aqlab.study import format_study, study_profile, write_study
    from aqlab.tools import CsvDataSource, SyntheticDataSource

    if args.data_dir:
        source = CsvDataSource(args.data_dir)
    else:
        source = SyntheticDataSource(n_symbols=args.symbols_count, n_days=args.days, seed=args.seed)

    universe = {sym: source.bars(sym) for sym in source.symbols()}
    bindings = load_profile(args.profile) if args.profile else list(DEFAULT_RULE_BINDINGS)
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    table, baseline = study_profile(universe, bindings, horizons=horizons, min_history=args.min_history)
    print(format_study(table))

    if args.out:
        paths = write_study(
            Path(args.out) / "study",
            table,
            baseline,
            meta={"profile": args.profile or "generic", "symbols": len(universe), "horizons": list(horizons)},
        )
        print(f"\n报告已写入：{paths['markdown']}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    import pandas as pd

    from aqlab.position import PositionConfig, defend_score, plan_position
    from aqlab.tables import markdown_table
    from aqlab.tools import CsvDataSource, SyntheticDataSource

    if args.data_dir:
        source = CsvDataSource(args.data_dir)
    else:
        source = SyntheticDataSource(n_symbols=args.symbols_count, n_days=args.days, seed=args.seed)

    symbol = args.symbol or source.symbols()[0]
    df = source.bars(symbol)
    recent = df.tail(args.lookback)
    plan = plan_position(
        entry_price=float(df["close"].iloc[-1]),
        reference_low=float(recent["low"].min()),
        reference_high=float(recent["high"].max()),
        config=PositionConfig(),
    )
    print(f"{symbol} 交易计划（参考近 {args.lookback} 根）")
    print()
    print(markdown_table(pd.DataFrame([plan.to_dict()])))

    score = defend_score(df).tail(5)
    print()
    print("近 5 根防卖飞评分：")
    print(markdown_table(score[["score", "advice", "up_close", "above_bbi", "no_volume_bear", "trend_up", "j_not_dead"]].reset_index().rename(columns={"index": "date"})))
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from aqlab.position import PositionConfig
    from aqlab.profiles import load_profile
    from aqlab.rules import DEFAULT_RULE_BINDINGS
    from aqlab.tools import CsvDataSource, SyntheticDataSource
    from aqlab.walkforward import WalkForwardConfig, format_walkforward, walk_forward, write_walkforward

    if args.data_dir:
        source = CsvDataSource(args.data_dir)
    else:
        source = SyntheticDataSource(n_symbols=args.symbols_count, n_days=args.days, seed=args.seed)

    universe = {sym: source.bars(sym) for sym in source.symbols()}
    bindings = load_profile(args.profile) if args.profile else list(DEFAULT_RULE_BINDINGS)
    position_config = PositionConfig(hard_stop_pct=None) if args.no_hard_stop else None
    config = WalkForwardConfig(
        test_days=args.test_days,
        step_days=args.step_days,
        min_history=args.min_history,
        use_position=not args.fixed_horizon,
        horizon=args.horizon,
        position_config=position_config,
    )
    result = walk_forward(universe, bindings, config=config)
    print(format_walkforward(result))

    if args.out:
        paths = write_walkforward(Path(args.out) / "walkforward", result)
        print(f"\n报告已写入：{paths['markdown']}")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    from aqlab.portfolio import (
        METHODS,
        PortfolioConfig,
        exposure_report,
        format_portfolio_report,
        simulate_portfolio,
        write_portfolio_report,
    )
    from aqlab.profiles import load_profile
    from aqlab.rules import DEFAULT_RULE_BINDINGS
    from aqlab.tools import CsvDataSource, SyntheticDataSource
    from aqlab.walkforward import composite_scores

    if args.data_dir:
        source = CsvDataSource(args.data_dir)
    else:
        source = SyntheticDataSource(n_symbols=args.symbols_count, n_days=args.days, seed=args.seed)

    universe = {sym: source.bars(sym) for sym in source.symbols()}
    bindings = load_profile(args.profile) if args.profile else list(DEFAULT_RULE_BINDINGS)
    scores = composite_scores(universe, bindings)
    signals = {symbol: score > 0 for symbol, score in scores.items()}

    config = PortfolioConfig(
        method=args.method,
        lookback=args.lookback,
        rebalance_days=args.rebalance_days,
        max_weight=args.max_weight,
        cash_buffer=args.cash_buffer,
        turnover_limit=args.turnover_limit,
        cost_bps=args.cost_bps,
        min_history=args.min_history,
    )
    result = simulate_portfolio(universe, signals, config)
    exposure = None
    if not result["rebalances"].empty:
        exposure = exposure_report(result["rebalances"].iloc[-1]["weights"], universe, lookback=config.lookback)
    print(format_portfolio_report(result, exposure, config))

    if args.out:
        paths = write_portfolio_report(Path(args.out) / "portfolio", result, exposure, config)
        print(f"\n报告已写入：{paths['markdown']}")
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

    p_eval = sub.add_parser("eval", help="run the agent evaluation harness (offline by default)")
    p_eval.add_argument("--mode", choices=["offline", "live"], default="offline")
    p_eval.add_argument("--model", default=None, help="model name for live mode")
    p_eval.add_argument("--symbol", default=None)
    p_eval.add_argument("--data-dir", default=None, help="CSV directory as data source; default synthetic")
    p_eval.add_argument("--max-steps", type=int, default=6)
    p_eval.add_argument("--out", default=str(DEFAULT_OUT))
    p_eval.set_defaults(func=cmd_eval)

    p_agent = sub.add_parser("agent", help="ask the tool-using research agent a question (needs an LLM API key)")
    p_agent.add_argument("--question", required=True)
    p_agent.add_argument("--model", default=None)
    p_agent.add_argument("--data-dir", default=None)
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.add_argument("--trace", default=None, help="path to write the full step trace as JSON")
    p_agent.set_defaults(func=cmd_agent)

    p_daily = sub.add_parser("daily", help="daily scoring pipeline (synthetic/CSV/tushare) with optional Feishu push")
    p_daily.add_argument("--date", default=None, help="as-of date (YYYY-MM-DD); default = latest bar")
    p_daily.add_argument("--top", type=int, default=10)
    p_daily.add_argument("--data-dir", default=None, help="CSV directory as data source")
    p_daily.add_argument("--tushare", action="store_true", help="use Tushare (needs TUSHARE_TOKEN + --symbols)")
    p_daily.add_argument("--turnover", action="store_true", help="also fetch daily_basic turnover_rate (real 换手率)")
    p_daily.add_argument("--symbols", default=None, help="comma separated symbols for --tushare")
    p_daily.add_argument("--start", default=None)
    p_daily.add_argument("--end", default=None)
    p_daily.add_argument("--rule", action="append", default=None, help="override e.g. --rule needle_below_ma.ma_window=30")
    p_daily.add_argument("--profile", default=None, help="rule set: generic|b1|b2|b3|needle_20|needle_30|volume_price_v3|zgnb_full|zgnb_needle30")
    p_daily.add_argument("--gate", choices=["amv", "activity", "none"], default="amv", help="market gate (default amv = 0AMV 波段开关)")
    p_daily.add_argument("--no-gate", action="store_true", help="disable the market gate entirely")
    p_daily.add_argument("--push", action="store_true", help="really push to Feishu (default is dry-run)")
    p_daily.add_argument("--webhook", default=None, help="Feishu webhook URL (falls back to FEISHU_WEBHOOK)")
    p_daily.add_argument("--symbols-count", type=int, default=30, help="synthetic universe size when no data source is given")
    p_daily.add_argument("--days", type=int, default=500)
    p_daily.add_argument("--seed", type=int, default=11)
    p_daily.add_argument("--out", default=str(DEFAULT_OUT))
    p_daily.set_defaults(func=cmd_daily)

    p_study = sub.add_parser("study", help="event study: forward returns of every rule in a profile vs baseline")
    p_study.add_argument("--profile", default=None, help="rule set; default = generic rules")
    p_study.add_argument("--data-dir", default=None)
    p_study.add_argument("--symbols-count", type=int, default=30)
    p_study.add_argument("--days", type=int, default=600)
    p_study.add_argument("--seed", type=int, default=11)
    p_study.add_argument("--horizons", default="1,3,5,10", help="comma separated forward horizons in trading days")
    p_study.add_argument("--min-history", type=int, default=60)
    p_study.add_argument("--out", default=str(DEFAULT_OUT))
    p_study.set_defaults(func=cmd_study)

    p_plan = sub.add_parser("plan", help="trade plan (stop/targets/size) and defend-score for a symbol")
    p_plan.add_argument("--symbol", default=None)
    p_plan.add_argument("--data-dir", default=None)
    p_plan.add_argument("--symbols-count", type=int, default=10)
    p_plan.add_argument("--days", type=int, default=400)
    p_plan.add_argument("--seed", type=int, default=11)
    p_plan.add_argument("--lookback", type=int, default=30, help="bars used for the structural reference low/high")
    p_plan.add_argument("--out", default=str(DEFAULT_OUT))
    p_plan.set_defaults(func=cmd_plan)

    p_wf = sub.add_parser("walkforward", help="rolling-window validation of a rule set (signals -> trades)")
    p_wf.add_argument("--profile", default=None)
    p_wf.add_argument("--data-dir", default=None)
    p_wf.add_argument("--symbols-count", type=int, default=40)
    p_wf.add_argument("--days", type=int, default=900)
    p_wf.add_argument("--seed", type=int, default=11)
    p_wf.add_argument("--test-days", type=int, default=60)
    p_wf.add_argument("--step-days", type=int, default=60)
    p_wf.add_argument("--min-history", type=int, default=120)
    p_wf.add_argument("--horizon", type=int, default=5, help="fixed holding period when --fixed-horizon is used")
    p_wf.add_argument("--fixed-horizon", action="store_true", help="skip position/exit rules, use a fixed holding period")
    p_wf.add_argument("--no-hard-stop", action="store_true", help="disable the -2%% short-term hard stop (experiment)")
    p_wf.add_argument("--out", default=str(DEFAULT_OUT))
    p_wf.set_defaults(func=cmd_walkforward)

    p_pf = sub.add_parser("portfolio", help="portfolio construction: optimize weights, apply constraints, simulate net value")
    p_pf.add_argument("--profile", default=None)
    p_pf.add_argument("--method", choices=["equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance"], default="risk_parity")
    p_pf.add_argument("--data-dir", default=None)
    p_pf.add_argument("--symbols-count", type=int, default=40)
    p_pf.add_argument("--days", type=int, default=900)
    p_pf.add_argument("--seed", type=int, default=11)
    p_pf.add_argument("--lookback", type=int, default=60)
    p_pf.add_argument("--rebalance-days", type=int, default=5)
    p_pf.add_argument("--max-weight", type=float, default=0.20)
    p_pf.add_argument("--cash-buffer", type=float, default=0.20)
    p_pf.add_argument("--turnover-limit", type=float, default=0.30)
    p_pf.add_argument("--cost-bps", type=float, default=5.0)
    p_pf.add_argument("--min-history", type=int, default=120)
    p_pf.add_argument("--out", default=str(DEFAULT_OUT))
    p_pf.set_defaults(func=cmd_portfolio)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
