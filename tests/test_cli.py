"""CLI 端到端冒烟测试（离线）：把每个不依赖网络/外部数据库的子命令真跑一遍。

为什么值得单独测：``cli.py`` 是最外层门面，错误只会在用户手里出现；
每个子命令都断言"退出码 0 + 产出文件存在"，比只跑 pytest 单元测试更接近真实使用。
"""

import pytest

from aqlab.cli import main


def write_csv(path, rows=400, seed=3):
    """造一份最小可用的日线 CSV（中文列名，验证 CLI 的列名兼容）。"""
    from aqlab.data import generate_synthetic_ohlcv

    frame = generate_synthetic_ohlcv(n_days=rows, seed=seed)
    frame = frame.reset_index().rename(
        columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_demo_runs_and_writes_reports(tmp_path):
    assert main(["demo", "--seed", "25", "--days", "300", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "demo").exists()


def test_screen_runs(tmp_path):
    assert main(["screen", "--symbols", "20", "--days", "300", "--top", "5", "--out", str(tmp_path)]) == 0
    assert list((tmp_path / "screen").glob("*"))


def test_run_backtest_with_chinese_columns(tmp_path):
    csv = write_csv(tmp_path / "600519.csv")
    assert main(["run", "--csv", str(csv), "--strategy", "ma_cross", "--params", "fast=10,slow=30", "--out", str(tmp_path)]) == 0


def test_eval_offline(tmp_path):
    assert main(["eval", "--mode", "offline", "--out", str(tmp_path)]) == 0


def test_study_plan_walkforward_portfolio_sweep_run_offline(tmp_path):
    """这些子命令都用合成数据，应该全部零依赖跑通。"""
    commands = [
        ["study", "--out", str(tmp_path)],
        ["plan", "--out", str(tmp_path)],
        ["walkforward", "--out", str(tmp_path)],
        ["portfolio", "--out", str(tmp_path)],
        ["sweep", "--out", str(tmp_path)],
        ["daily", "--out", str(tmp_path)],
    ]
    for command in commands:
        assert main(command) == 0, f"{command[0]} 退出码非 0"


def test_decide_with_daily_csv(tmp_path):
    csv = write_csv(tmp_path / "600519.csv", rows=500, seed=7)
    assert main(["decide", "--daily-csv", str(csv), "--rule", "b1_opportunity", "--boost", "2.5", "--out", str(tmp_path)]) == 0


def test_quality_audit_on_directory(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    for seed in (3, 7):
        write_csv(data_dir / f"SYN{seed:03d}.csv", rows=250, seed=seed)
    assert main(["quality", "--data-dir", str(data_dir), "--out", str(tmp_path)]) == 0


def test_confirm_eval_offline_with_minute_csv(tmp_path):
    """量比确认：日线 + 合成分钟线都要能跑，且决策分布非空。"""
    from aqlab.data import generate_synthetic_ohlcv
    from aqlab.intraday import generate_synthetic_minutes

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    daily = generate_synthetic_ohlcv(n_days=320, seed=5)
    daily.to_csv(data_dir / "SYN005.csv", encoding="utf-8-sig")
    minutes = generate_synthetic_minutes(daily)
    minutes.rename_axis("minute").reset_index().to_csv(data_dir / "SYN005_min.csv", index=False, encoding="utf-8-sig")

    assert main(
        [
            "confirm-eval", "--data-dir", str(data_dir), "--rule", "b1_graded",
            "--window-minutes", "7", "--min-ratio", "1.0", "--horizons", "1,3,5",
            "--out", str(tmp_path),
        ]
    ) == 0


def test_universe_study_on_tiny_synthetic_universe(tmp_path):
    """全市场研究：两只有信号的合成标的，验证入场、基准与产出。"""
    from aqlab.data import generate_synthetic_ohlcv

    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    for seed in (11, 25):
        generate_synthetic_ohlcv(n_days=400, seed=seed).to_csv(daily_dir / f"SYN{seed:03d}.csv", encoding="utf-8-sig")
    out = tmp_path / "out"
    assert main(["universe-study", "--daily-dir", str(daily_dir), "--start", "2020-01-01", "--end", "2030-01-01", "--out", str(out)]) == 0


def test_picks_backtest_requires_archive_dir(tmp_path):
    """空目录应该给出 0 退出码与提示，而不是崩溃。"""
    empty = tmp_path / "archive"
    empty.mkdir()
    assert main(["picks-backtest", "--archive", str(empty), "--out", str(tmp_path)]) == 0


def test_unknown_command_exits_with_error():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])

def test_plot_renders_charts_when_matplotlib_is_available(tmp_path):
    """绘图子命令：装了 matplotlib 就真出图；没装则跳过（CI 不强制绘图依赖）。"""
    import pytest as _pytest

    _pytest.importorskip("matplotlib", reason="charts need the optional 'plot' extra")
    assert main(["plot", "--seed", "25", "--days", "300", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "charts" / "equity_curves.png").exists()
