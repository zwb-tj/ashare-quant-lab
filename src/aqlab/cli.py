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


def cmd_sweep(args: argparse.Namespace) -> int:
    from aqlab.profiles import load_profile
    from aqlab.rules import DEFAULT_RULE_BINDINGS
    from aqlab.sweep import SweepConfig, format_sweep, parse_grid, pick_best, run_sweep, write_sweep
    from aqlab.tools import CsvDataSource, SyntheticDataSource

    bindings = load_profile(args.profile) if args.profile else list(DEFAULT_RULE_BINDINGS)
    grids = parse_grid(args.set or [])
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    universes: dict[str, dict] = {}
    if args.data_dir:
        source = CsvDataSource(args.data_dir)
        universes[args.data_dir] = {sym: source.bars(sym) for sym in source.symbols()}
    else:
        for n in sizes:
            source = SyntheticDataSource(n_symbols=n, n_days=args.days, seed=args.seed)
            universes[f"{n}只"] = {sym: source.bars(sym) for sym in source.symbols()}

    config = SweepConfig(
        horizons=tuple(int(h) for h in args.horizons.split(",") if h.strip()),
        main_horizon=args.main_horizon,
        test_days=args.test_days,
        step_days=args.step_days,
        min_history=args.min_history,
        method=args.method,
        max_weight=args.max_weight,
        cash_buffer=args.cash_buffer,
        turnover_limit=args.turnover_limit,
        cost_bps=args.cost_bps,
        rebalance_days=args.rebalance_days,
    )
    table = run_sweep(universes, bindings, grids, config)
    best = pick_best(table, objective=args.objective)
    print(format_sweep(table, config, best))

    if args.out:
        out = Path(args.out) / "sweep"
        paths = write_sweep(out, table, meta={"profile": args.profile or "generic", "sets": args.set or [], "sizes": sizes})
        (out / "sweep.md").write_text(format_sweep(table, config, best), encoding="utf-8")
        print(f"\n报告已写入：{out / 'sweep.md'}")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    import pandas as pd

    from aqlab.data import load_ohlcv_csv
    from aqlab.intraday import IntradayConfig, confirm_signals, generate_synthetic_minutes, opening_volume_ratio
    from aqlab.rules import build_rule
    from aqlab.tables import markdown_table

    daily = load_ohlcv_csv(args.daily_csv)
    rule = build_rule(args.rule, **_parse_params(args.params))
    signal = (rule.score(daily) > 0).reindex(daily.index).fillna(False)

    if args.minute_csv:
        minutes = pd.read_csv(args.minute_csv, parse_dates=["minute"]).set_index("minute")
    else:
        minutes = generate_synthetic_minutes(daily, seed=args.seed, first_minutes_boost=args.boost)
        print("说明：未提供分钟数据，已用合成分钟线演示流程（真实使用请传 --minute-csv）")

    config = IntradayConfig(window_minutes=args.window_minutes, baseline_days=args.baseline_days, min_ratio=args.min_ratio)
    ratio = opening_volume_ratio(minutes, config)
    table = confirm_signals(signal, ratio, config)

    print(f"规则 {args.rule}｜信号 {int(signal.sum())} 个｜决策记录 {len(table)} 条")
    if not table.empty:
        print("决策分布：" + "、".join(f"{k} {v}" for k, v in table["decision"].value_counts().items()))
        print(markdown_table(table.tail(args.show)))
        valid = table[table["decision"].isin(["买入", "观望"])]
        if len(valid):
            print(f"确认率：{float((valid['decision'] == '买入').mean()):.1%}（买入 / 有量比数据）")
    if args.out:
        out = Path(args.out) / "decide"
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "decisions.csv", index=False, encoding="utf-8-sig")
        ratio.rename("volume_ratio").to_csv(out / "volume_ratio.csv", encoding="utf-8-sig")
        print(f"明细已写入：{out}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from aqlab.quality import QualityConfig, audit_universe, cross_source_diff, format_audit, snapshot_hash, write_audit
    from aqlab.tools import CsvDataSource

    source = CsvDataSource(args.data_dir)
    universe = {sym: source.bars(sym) for sym in source.symbols()}
    config = QualityConfig(max_gap_days=args.max_gap_days, price_jump_pct=args.price_jump_pct, min_bars=args.min_bars)
    table = audit_universe(universe, config)

    diffs = []
    if args.cross_check:
        other = CsvDataSource(args.cross_check)
        common = sorted(set(source.symbols()) & set(other.symbols()))[: args.cross_limit]
        for symbol in common:
            diffs.append(cross_source_diff(symbol, universe[symbol], other.bars(symbol), tolerance=args.tolerance))
        if not common:
            print("提示：两个目录没有同名标的，跳过交叉校验。")

    hashes = {sym: snapshot_hash(df) for sym, df in list(universe.items())[:10]}
    print(format_audit(table, diffs or None, hashes))
    if args.out:
        paths = write_audit(Path(args.out) / "quality", table, diffs, hashes)
        print(f"报告已写入：{paths['markdown']}")
    return 0


def cmd_stockdb_export(args: argparse.Namespace) -> int:
    from aqlab.stockdb import export_sample

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    written = export_sample(
        symbols,
        args.daily_start,
        args.daily_end,
        args.out,
        minute_start=args.minute_start,
        minute_end=args.minute_end,
        fq=args.fq,
    )
    for symbol, paths in written.items():
        detail = ", ".join(f"{p.name}({p.stat().st_size // 1024}KB)" for p in paths)
        print(f"{symbol}: {detail}")
    print(f"共导出 {len(written)} 只 -> {args.out}")
    return 0


def cmd_confirm_eval(args: argparse.Namespace) -> int:
    import pandas as pd

    from aqlab.confirm_eval import ConfirmEvalConfig, evaluate_confirmation, summarize_confirmation
    from aqlab.data import load_ohlcv_csv
    from aqlab.rules import build_rule
    from aqlab.tables import markdown_table

    rule = build_rule(args.rule, **_parse_params(args.params))
    config = ConfirmEvalConfig(
        window_minutes=args.window_minutes,
        baseline_days=args.baseline_days,
        min_ratio=args.min_ratio,
        horizons=tuple(int(h) for h in args.horizons.split(",") if h.strip()),
    )
    data_dir = Path(args.data_dir)
    details_list = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.stem.endswith("_min"):
            continue
        symbol = path.stem
        daily = load_ohlcv_csv(path)
        minute_path = data_dir / f"{symbol}_min.csv"
        minute = (
            pd.read_csv(minute_path, parse_dates=["minute"]).set_index("minute")
            if minute_path.exists()
            else pd.DataFrame()
        )
        signal = (rule.score(daily) > 0).reindex(daily.index).fillna(False)
        details, _ratio = evaluate_confirmation(daily, minute, signal, config, symbol=symbol)
        if not details.empty:
            details_list.append(details)

    if not details_list:
        print("没有可评估的信号（检查数据目录与规则）。")
        return 0

    merged = pd.concat(details_list, ignore_index=True)
    summary = summarize_confirmation(merged, config)
    view = summary.copy()
    for horizon in config.horizons:
        view[f"mean_{horizon}"] = (view[f"mean_{horizon}"].astype(float) * 100).round(2)
        view[f"win_{horizon}"] = (view[f"win_{horizon}"].astype(float) * 100).round(1)
    view = view.rename(columns={"decision": "决策", "signals": "信号数"})
    print(f"规则 {args.rule}｜窗口 {config.window_minutes} 分钟｜量比阈值 {config.min_ratio:g}")
    print(markdown_table(view))
    if args.out:
        out = Path(args.out) / "confirm_eval"
        out.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out / "signals.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"明细已写入：{out}")
    return 0


def cmd_picks_backtest(args: argparse.Namespace) -> int:
    """不重新选股：直接评估 picks_archive 里你实际推出去的名单。"""
    import pandas as pd

    from aqlab.confirm_eval import ConfirmEvalConfig, evaluate_confirmation, summarize_confirmation
    from aqlab.confirm_eval import attach_benchmark as attach_confirm_benchmark
    from aqlab.data import load_ohlcv_csv
    from aqlab.picks import (
        PickBacktestConfig,
        attach_benchmark,
        benchmark_candidates,
        benchmark_returns,
        dedupe_picks,
        evaluate_picks,
        load_picks_archive,
        summarize_picks,
        summary_markdown,
    )
    from aqlab.stockdb import fetch_daily, fetch_minute
    from aqlab.tables import markdown_table

    buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())
    picks = load_picks_archive(args.archive, buckets=buckets)
    if picks.empty:
        print("选股日志为空（检查 --archive 与 --buckets）。")
        return 0
    config = PickBacktestConfig(
        horizons=tuple(int(h) for h in args.horizons.split(",") if h.strip()),
        entry=args.entry,
        dedupe_window=args.dedupe_window,
        buckets=buckets,
    )
    unique = dedupe_picks(picks, config.dedupe_window)
    by_bucket = picks.groupby("bucket").size().to_dict()
    print(f"日志：{picks['date'].min():%Y-%m-%d} ~ {picks['date'].max():%Y-%m-%d}｜记录 {len(picks)} 条｜分桶 {by_bucket}")
    print(f"去重后（同票 {config.dedupe_window} 日内只算一次）：{len(unique)} 条｜标的 {unique['symbol'].nunique()} 只")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    start = (picks["date"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
    daily: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for symbol in sorted(unique["symbol"].unique()):
        path = cache / f"{symbol}.csv"
        if path.exists() and not args.refresh:
            daily[symbol] = load_ohlcv_csv(path)
            continue
        try:
            frame = fetch_daily(symbol, start, args.data_end)
        except Exception:  # noqa: BLE001 - 记录失败，不让整轮崩
            failed.append(symbol)
            continue
        if frame.empty:
            failed.append(symbol)
            continue
        frame.to_csv(path, encoding="utf-8-sig")
        daily[symbol] = frame
    print(f"日线数据：可用 {len(daily)} 只" + (f"；缺失 {len(failed)} 只（如 {failed[:5]}）" if failed else ""))

    evaluated = evaluate_picks(unique, daily, config)

    benchmark_dir = Path(getattr(args, "benchmark_dir", "") or (cache / "bench"))
    basket_symbols = [s.strip() for s in (getattr(args, "benchmark_symbols", "") or "").split(",") if s.strip()]
    basket_sample = int(getattr(args, "benchmark_sample", 0) or 0)
    basket: dict[str, pd.DataFrame] = {}
    if basket_symbols:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        for symbol in basket_symbols:
            path = benchmark_dir / f"{symbol}.csv"
            try:
                if path.exists() and not args.refresh:
                    frame = load_ohlcv_csv(path)
                else:
                    frame = fetch_daily(symbol, start, args.data_end)
                    if frame.empty:
                        continue
                    frame.to_csv(path, encoding="utf-8-sig")
            except Exception:  # noqa: BLE001
                continue
            if len(frame) > 0:
                basket[symbol] = frame
    elif benchmark_dir.exists():
        for path in sorted(benchmark_dir.glob("*.csv")):
            try:
                if args.refresh:                       # 篮子缓存也要跟着刷新，否则新入场日没有基准
                    frame = fetch_daily(path.stem, start, args.data_end)
                    if frame.empty:
                        continue
                    frame.to_csv(path, encoding="utf-8-sig")
                else:
                    frame = load_ohlcv_csv(path)
            except Exception:  # noqa: BLE001
                continue
            if len(frame) > 0:
                basket[path.stem] = frame

    if basket:
        chosen = benchmark_candidates(basket, exclude=unique["symbol"], sample=basket_sample)
        basket = {symbol: basket[symbol] for symbol in chosen}
        benchmark = benchmark_returns(basket, evaluated["entry_date"].dropna().unique(), config)
        evaluated = attach_benchmark(evaluated, benchmark)
        covered = benchmark[[c for c in benchmark.columns if c.startswith("benchn_")]].max().max() if len(benchmark) else 0
        print(f"基准篮子：{len(basket)} 只（已剔除本轮选中的票）｜单日最多可用成员 {int(covered)}")

    summary = summarize_picks(evaluated, config)
    print()
    print(summary_markdown(summary, config))

    details_tables = []
    if args.confirm:
        ce_config = ConfirmEvalConfig(
            window_minutes=args.window_minutes,
            baseline_days=args.baseline_days,
            min_ratio=args.min_ratio,
            horizons=config.horizons,
            entry=args.confirm_entry,
        )
        window = unique[(unique["date"] >= pd.Timestamp(args.confirm_start)) & (unique["date"] <= pd.Timestamp(args.confirm_end))]
        confirm_buckets = tuple(b.strip() for b in args.confirm_buckets.split(",") if b.strip())
        window = window[window["bucket"].isin(confirm_buckets)]
        print()
        print(f"量比确认评估：窗口 {args.confirm_start} ~ {args.confirm_end}｜分桶 {list(confirm_buckets)}｜候选 {len(window)} 条｜入场 {ce_config.entry}")
        minute_start = (pd.Timestamp(args.confirm_start) - pd.Timedelta(days=20)).strftime("%Y%m%d")
        minute_end = (pd.Timestamp(args.confirm_end) + pd.Timedelta(days=7)).strftime("%Y%m%d")
        skipped: list[str] = []
        for symbol, group in window.groupby("symbol"):
            if symbol not in daily:
                skipped.append(symbol)
                continue
            minute_path = cache / f"{symbol}_min.csv"
            if not minute_path.exists() or args.refresh:
                try:
                    minute = fetch_minute(symbol, minute_start, minute_end)
                except Exception:  # noqa: BLE001
                    skipped.append(symbol)
                    continue
                if minute.empty:
                    skipped.append(symbol)
                    continue
                minute.rename_axis("minute").reset_index()[["minute", "close", "volume"]].to_csv(
                    minute_path, index=False, encoding="utf-8-sig"
                )
            minute = pd.read_csv(minute_path, parse_dates=["minute"]).set_index("minute")
            signal = pd.Series(False, index=daily[symbol].index)
            signal.loc[daily[symbol].index.intersection(group["date"])] = True
            details, _ratio = evaluate_confirmation(daily[symbol], minute, signal, ce_config, symbol=symbol)
            if not details.empty:
                details_tables.append(details)
        if skipped:
            print(f"（{len(skipped)} 只没有可用分钟数据，已跳过：{skipped[:5]}）")
        if details_tables:
            merged = pd.concat(details_tables, ignore_index=True)
            if basket:
                # 口径对齐：个股在决策日 9:37 买入，篮子用**决策日开盘价**买入（差几分钟，量级可忽略），
                # 两侧都在第 h 个交易日的收盘卖出；用 next_close 会让 bench_1 恒等于 0。
                ce_bench = benchmark_returns(
                    basket, merged["entry_date"].dropna().unique(), PickBacktestConfig(horizons=ce_config.horizons, entry="next_open")
                )
                merged = attach_confirm_benchmark(merged, ce_bench)
            confirm_summary = summarize_confirmation(merged, ce_config)
            view = confirm_summary.copy()
            has_ce_bench = any(f"excess_{h}" in view.columns for h in ce_config.horizons)
            for horizon in ce_config.horizons:
                view[f"mean_{horizon}"] = (view[f"mean_{horizon}"].astype(float) * 100).round(2)
                view[f"win_{horizon}"] = (view[f"win_{horizon}"].astype(float) * 100).round(1)
                if has_ce_bench:
                    view[f"bench_{horizon}"] = (view[f"bench_{horizon}"].astype(float) * 100).round(2)
                    view[f"excess_{horizon}"] = (view[f"excess_{horizon}"].astype(float) * 100).round(2)
                    view[f"excesswin_{horizon}"] = (view[f"excesswin_{horizon}"].astype(float) * 100).round(1)
            print(markdown_table(view.rename(columns={"decision": "决策", "signals": "信号数"})))
            if args.out:
                confirm_out = Path(args.out) / "picks_backtest"
                confirm_out.mkdir(parents=True, exist_ok=True)
                merged.to_csv(confirm_out / "confirm_details.csv", index=False, encoding="utf-8-sig")
                confirm_summary.to_csv(confirm_out / "confirm_summary.csv", index=False, encoding="utf-8-sig")
        else:
            print("（窗口内没有可评估的分钟数据）")

    if args.out:
        out = Path(args.out) / "picks_backtest"
        out.mkdir(parents=True, exist_ok=True)
        evaluated.to_csv(out / "picks_evaluated.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
        (out / "summary.md").write_text(summary_markdown(summary, config), encoding="utf-8")
        print(f"\n结果已写入：{out}")
    return 0


def cmd_universe_study(args: argparse.Namespace) -> int:
    """全市场 B1 研究：选股 + 大盘阶段（0AMV）+ 离场规则，同一批信号直接对比。"""
    import pandas as pd

    from aqlab.exits import EXIT_REASONS, ExitConfig
    from aqlab.rules_zgnb import ActiveMarketValueGate
    from aqlab.study_universe import UniverseStudyConfig, load_universe_daily, study_universe, summarize_trades
    from aqlab.tables import markdown_table

    params: dict = {}
    if args.params:
        for item in args.params.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            try:
                params[key.strip()] = float(value) if "." in value else int(value)
            except ValueError:
                params[key.strip()] = value.strip()

    exit_config = ExitConfig(
        mode=args.exit_mode,
        stop_pct=args.stop_pct,
        intraday_stop_pct=args.intraday_stop,
        take_profit_pct=args.take_profit if args.take_profit > 0 else None,
        min_holding_days=args.min_holding,
        white_break_days=args.white_break_days,
        use_death_cross=not args.no_death_cross,
        use_white_break=not args.no_white_break,
        didi_mode="off" if args.no_didi else args.didi_mode,
        max_holding_days=args.max_holding if args.max_holding > 0 else None,
    )
    config = UniverseStudyConfig(
        rule=args.rule,
        rule_params=params,
        start=args.start,
        end=args.end,
        exit=exit_config,
        use_regime_gate=not args.no_regime_gate,
        limit=args.limit,
    )
    frames = load_universe_daily(args.daily_dir, limit=args.limit)
    if not frames:
        print(f"没有读到日线数据（--daily-dir {args.daily_dir}）")
        return 1
    print(f"标的 {len(frames)} 只｜规则 {config.rule}{params}｜离场 {args.exit_mode}｜大盘门 {'开' if config.use_regime_gate else '关'}")

    gate = ActiveMarketValueGate()
    table, meta = study_universe(frames, config, gate=gate)
    if table.empty:
        print(f"没有交易（被大盘阶段挡掉 {meta['gated_out']} 笔）")
        return 0

    print()
    print("## 全部交易")
    print(markdown_table(summarize_trades(table)))
    print()
    print("## 按大盘阶段（0AMV 波段）")
    table["波段"] = table["signal_date"].apply(lambda date: "开波段" if int(_regime_lookup(gate, frames, date)) == 1 else "关波段")
    print(markdown_table(summarize_trades(table, by="波段")))
    print()
    print("## 按离场原因")
    table["离场原因"] = table["reason"].map(lambda code: EXIT_REASONS.get(code, code))
    print(markdown_table(summarize_trades(table, by="离场原因")))
    print()
    print("## 按年份")
    print(markdown_table(summarize_trades(table, by="year")))
    print()
    print("## 对照组：同一批信号机械持有 h 日（%）")
    rows = [{"持有期": f"{h} 日", "平均收益%": round(table[f"hold_{h}"].mean() * 100, 2),
             "中位%": round(table[f"hold_{h}"].median() * 100, 2),
             "胜率%": round((table[f"hold_{h}"] > 0).mean() * 100, 1)}
            for h in (1, 3, 5, 10, 20)]
    rows.append({"持有期": "离场规则", "平均收益%": round(table["return_pct"].mean() * 100, 2),
                 "中位%": round(table["return_pct"].median() * 100, 2),
                 "胜率%": round((table["return_pct"] > 0).mean() * 100, 1)})
    rows.append({"持有期": "平均持有天数", "平均收益%": round(table["bars"].mean(), 1), "中位%": "", "胜率%": ""})
    print(markdown_table(pd.DataFrame(rows)))

    if args.out:
        out = Path(args.out) / "universe_study"
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
        for name, by in (("summary_all", None), ("summary_regime", "波段"), ("summary_reason", "离场原因"), ("summary_year", "year")):
            summarize_trades(table, by=by).to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        (out / "meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()), encoding="utf-8")
        print(f"\n结果已写入：{out}")
    return 0


_REGIME_CACHE: dict = {}


def _regime_lookup(gate, frames, date):
    """按需计算 0AMV 波段状态（缓存一次）。"""
    import pandas as pd

    if "series" not in _REGIME_CACHE:
        raw = gate.gate_series(frames)
        _REGIME_CACHE["series"] = raw["gate"].shift(1).fillna(0).astype(int)
    series = _REGIME_CACHE["series"]
    stamp = pd.Timestamp(date)
    return series.get(stamp, 0)


def cmd_plot(args: argparse.Namespace) -> int:
    """把回测结果画成图：净值曲线、回撤、策略指标对比（可选依赖 matplotlib）。"""
    import pandas as pd

    from aqlab.charts import plot_drawdown, plot_equity_curves, plot_strategy_comparison

    if args.csv:
        from aqlab.data import load_ohlcv_csv

        df = load_ohlcv_csv(args.csv)
    else:
        df = generate_synthetic_ohlcv(n_days=args.days, seed=args.seed)

    config = BacktestConfig(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    default_names = ["buy_and_hold", "ma_cross", "momentum", "mean_reversion"]
    names = [item.strip() for item in args.strategies.split(",") if item.strip()] if args.strategies else default_names
    curves: dict[str, pd.DataFrame] = {}
    summary: list[dict] = []
    for name in names:
        strategy = build_strategy(name)
        result = run_backtest(df, strategy.positions(df), config=config, name=strategy.name)
        metrics = compute_metrics(result.frame, initial_cash=config.initial_cash, trades=result.trades)
        curves[strategy.name] = result.frame
        summary.append(
            {
                "strategy": strategy.name,
                **{key: metrics.get(key) for key in ("total_return", "sharpe", "max_drawdown", "win_rate_trade")},
            }
        )

    benchmark = (df["close"] / df["close"].iloc[0]) * config.initial_cash
    out = Path(args.out) / "charts"
    written = [
        plot_equity_curves(curves, out / "equity_curves.png", title=args.title, benchmark=benchmark),
        plot_drawdown(curves[names[0]], out / "drawdown.png", title=f"Drawdown - {names[0]}"),
        plot_strategy_comparison(pd.DataFrame(summary), out / "strategy_comparison.png", title=args.title),
    ]
    for path in written:
        print(f"图已写入：{path}")
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

    p_sw = sub.add_parser("sweep", help="parameter sweep: event study + walk-forward + portfolio per grid point")
    p_sw.add_argument("--profile", default=None)
    p_sw.add_argument("--set", action="append", default=None, help="rule.param=v1,v2 (repeatable -> cartesian product)")
    p_sw.add_argument("--sizes", default="40", help="comma separated synthetic universe sizes, e.g. 40,80,120")
    p_sw.add_argument("--data-dir", default=None)
    p_sw.add_argument("--days", type=int, default=900)
    p_sw.add_argument("--seed", type=int, default=11)
    p_sw.add_argument("--horizons", default="1,3,5,10")
    p_sw.add_argument("--main-horizon", type=int, default=5)
    p_sw.add_argument("--test-days", type=int, default=60)
    p_sw.add_argument("--step-days", type=int, default=60)
    p_sw.add_argument("--min-history", type=int, default=120)
    p_sw.add_argument("--method", choices=["equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance"], default="risk_parity")
    p_sw.add_argument("--max-weight", type=float, default=0.20)
    p_sw.add_argument("--cash-buffer", type=float, default=0.20)
    p_sw.add_argument("--turnover-limit", type=float, default=0.30)
    p_sw.add_argument("--cost-bps", type=float, default=5.0)
    p_sw.add_argument("--rebalance-days", type=int, default=5)
    p_sw.add_argument("--objective", default="excess_mean", help="column to maximise among rows meeting the target")
    p_sw.add_argument("--out", default=str(DEFAULT_OUT))
    p_sw.set_defaults(func=cmd_sweep)
    p_dec = sub.add_parser("decide", help="confirm daily signals (e.g. B1) with the opening N-minute volume ratio")
    p_dec.add_argument("--daily-csv", required=True)
    p_dec.add_argument("--minute-csv", default=None, help="minute bars CSV with columns: minute,close,volume")
    p_dec.add_argument("--rule", default="b1_graded")
    p_dec.add_argument("--params", default=None)
    p_dec.add_argument("--window-minutes", type=int, default=7)
    p_dec.add_argument("--baseline-days", type=int, default=5)
    p_dec.add_argument("--min-ratio", type=float, default=1.0)
    p_dec.add_argument("--boost", type=float, default=1.0)
    p_dec.add_argument("--seed", type=int, default=7)
    p_dec.add_argument("--show", type=int, default=10)
    p_dec.add_argument("--out", default=str(DEFAULT_OUT))
    p_dec.set_defaults(func=cmd_decide)

    p_q = sub.add_parser("quality", help="audit data quality (gaps, zero volume, price jumps, cross-source, hash)")
    p_q.add_argument("--data-dir", required=True)
    p_q.add_argument("--cross-check", default=None)
    p_q.add_argument("--cross-limit", type=int, default=5)
    p_q.add_argument("--tolerance", type=float, default=0.005)
    p_q.add_argument("--max-gap-days", type=int, default=10)
    p_q.add_argument("--price-jump-pct", type=float, default=0.11)
    p_q.add_argument("--min-bars", type=int, default=60)
    p_q.add_argument("--out", default=str(DEFAULT_OUT))
    p_q.set_defaults(func=cmd_quality)
    p_se = sub.add_parser("stockdb-export", help="export a small daily+minute sample from the local stockdb")
    p_se.add_argument("--symbols", required=True, help="comma separated, e.g. 600519,000001")
    p_se.add_argument("--daily-start", required=True)
    p_se.add_argument("--daily-end", required=True)
    p_se.add_argument("--minute-start", default=None)
    p_se.add_argument("--minute-end", default=None)
    p_se.add_argument("--fq", default="qfq", choices=["qfq", "hfq", "none"])
    p_se.add_argument("--out", default="data/raw")
    p_se.set_defaults(func=cmd_stockdb_export)

    p_ce = sub.add_parser("confirm-eval", help="does the opening volume-ratio gate improve B1 signals? (real minute data)")
    p_ce.add_argument("--data-dir", required=True, help="directory with {symbol}.csv and {symbol}_min.csv")
    p_ce.add_argument("--rule", default="b1_opportunity")
    p_ce.add_argument("--params", default=None)
    p_ce.add_argument("--window-minutes", type=int, default=7)
    p_ce.add_argument("--baseline-days", type=int, default=5)
    p_ce.add_argument("--min-ratio", type=float, default=1.0)
    p_ce.add_argument("--horizons", default="1,3,5")
    p_ce.add_argument("--out", default=str(DEFAULT_OUT))
    p_ce.set_defaults(func=cmd_confirm_eval)
    p_pb = sub.add_parser("picks-backtest", help="backtest the picks you actually published (picks_archive)")
    p_pb.add_argument("--archive", required=True, help="directory with picks_YYYY-MM-DD.json")
    p_pb.add_argument("--cache", default="data/picks", help="cache directory for fetched daily/minute bars")
    p_pb.add_argument("--buckets", default="b1,b2,n20,n30,v3")
    p_pb.add_argument("--entry", default="next_open", choices=["next_open", "next_close"])
    p_pb.add_argument("--dedupe-window", type=int, default=5)
    p_pb.add_argument("--horizons", default="1,3,5,10")
    p_pb.add_argument("--data-end", default="20260930")
    p_pb.add_argument("--refresh", action="store_true", help="re-fetch bars even if cached")
    p_pb.add_argument("--confirm", action="store_true", help="also evaluate the opening volume-ratio gate")
    p_pb.add_argument("--confirm-buckets", default="b1", help="which buckets the volume-ratio gate is evaluated on")
    p_pb.add_argument(
        "--confirm-entry",
        default="window_close",
        choices=["window_close", "decision_close", "signal_close"],
        help="window_close = buy at the close of the opening window (default, no lookahead)",
    )
    p_pb.add_argument("--confirm-start", default="2026-06-17")
    p_pb.add_argument("--confirm-end", default="2026-09-10")
    p_pb.add_argument("--window-minutes", type=int, default=7)
    p_pb.add_argument("--baseline-days", type=int, default=5)
    p_pb.add_argument("--min-ratio", type=float, default=1.0)
    p_pb.add_argument("--benchmark-dir", default="", help="directory of basket daily CSVs (default: <cache>/bench)")
    p_pb.add_argument("--benchmark-symbols", default="", help="comma list of basket symbols to fetch/cache on demand")
    p_pb.add_argument("--benchmark-sample", type=int, default=0, help="evenly sample N basket symbols (0 = all)")
    p_pb.add_argument("--out", default=str(DEFAULT_OUT))
    p_pb.set_defaults(func=cmd_picks_backtest)
    p_us = sub.add_parser("universe-study", help="full-market B1 study: regime gate + exit rules on one signal set")
    p_us.add_argument("--daily-dir", default="data/universe/daily")
    p_us.add_argument("--rule", default="b1_graded")
    p_us.add_argument("--params", default=None)
    p_us.add_argument("--start", default="2025-01-01")
    p_us.add_argument("--end", default="2026-09-11")
    p_us.add_argument("--exit-mode", default="entry_low", choices=["entry_low", "fixed", "atr"])
    p_us.add_argument("--stop-pct", type=float, default=0.03, help="entry_low: 止损价 = 入场K线最低价 × (1 - x)")
    p_us.add_argument("--intraday-stop", type=float, default=None, help="fixed 模式的盘中止损百分比（如 0.07）")
    p_us.add_argument("--take-profit", type=float, default=0.15, help="盘中止盈百分比，<=0 关闭")
    p_us.add_argument("--min-holding", type=int, default=3, help="最短持仓保护天数（前 N 日不止损）")
    p_us.add_argument("--white-break-days", type=int, default=2)
    p_us.add_argument("--no-death-cross", action="store_true", help="关闭白线下穿黄线清仓")
    p_us.add_argument("--no-white-break", action="store_true", help="关闭白线连续破位清仓")
    p_us.add_argument("--no-didi", action="store_true", help="关闭滴滴（今收 < 昨低）")
    p_us.add_argument("--didi-mode", default="full", choices=["full", "simple"], help="full=连续两根阴线+破昨低+量能不缩+不在深跌区")
    p_us.add_argument("--max-holding", type=int, default=0, help="强制持有上限，0 = 不设")
    p_us.add_argument("--no-regime-gate", action="store_true", help="不用 0AMV 大盘阶段过滤买入")
    p_us.add_argument("--limit", type=int, default=None)
    p_us.add_argument("--out", default=str(DEFAULT_OUT))
    p_us.set_defaults(func=cmd_universe_study)
    p_plot = sub.add_parser("plot", help="render equity/drawdown/comparison charts (needs the 'plot' extra)")
    p_plot.add_argument("--csv", default=None, help="optional OHLCV CSV; default uses synthetic bars")
    p_plot.add_argument("--strategies", default=None, help="comma list, default all four built-ins")
    p_plot.add_argument("--days", type=int, default=500)
    p_plot.add_argument("--seed", type=int, default=25)
    p_plot.add_argument("--fee-bps", type=float, default=3.0)
    p_plot.add_argument("--slippage-bps", type=float, default=2.0)
    p_plot.add_argument("--title", default="Built-in strategies on the same synthetic bars")
    p_plot.add_argument("--out", default=str(DEFAULT_OUT))
    p_plot.set_defaults(func=cmd_plot)
    return parser


def _make_output_encoding_safe() -> None:
    """Windows 控制台默认 GBK，遇到 −/≤/℃ 这类字符会直接抛 UnicodeEncodeError。

    优先切 UTF-8；切不动就退化成替换字符，保证报告能完整打印完。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 老终端不支持就退到 replace
            try:
                reconfigure(errors="replace")
            except Exception:  # noqa: BLE001
                pass


def main(argv: list[str] | None = None) -> int:
    _make_output_encoding_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
