# A-Share Quant Lab (`aqlab`)

**一个从零实现、可复现的 A 股选股 + 回测实验室**，包含显式的执行成本模型、无未来函数的信号执行、横截面因子打分，以及一条可测试的 CLI。仓库为 **clean-room 原创实现**：不含任何第三方项目代码或整理内容。

> 不是又一个"翻倍策略"仓库。这是一个把**研究纪律**写在代码里的工具：成本要显式、执行要延迟一根 K 线、策略要能被证伪。

---

## 为什么做这个

做量化研究最常见的三个坑，都能用工程手段提前堵死：

1. **未来函数（lookahead）**——信号用了当天收盘价还算收益，回测自然"很赚"。本项目把所有信号统一延迟一根 K 线执行，并写了一条**专门证明它有延迟**的测试（用一个"偷看今天涨跌"的信号，结果必须是亏的）。
2. **成本幻觉**——不计手续费/滑点的高频信号在纸面上很美。本项目把成本建模进收益，逐笔可查。
3. **不可复现**——数据与随机性没固定，别人跑不出同样结果。本项目用确定性合成数据 + 固定种子，`pytest` 与 demo 在任何机器上输出一致。

## 快速开始

```bash
# 1) 安装（可编辑模式，附开发依赖）
pip install -e ".[dev]"

# 2) 跑测试（33 个用例，全部离线，无需网络/API key）
pytest -q

# 3) 三分钟看结果：内置策略在同一份合成行情上的对比
python -m aqlab.cli demo --seed 25      # 上涨行情
python -m aqlab.cli demo --seed 7       # 下跌行情

# 4) 横截面选股打分（30 只合成票池，加权因子打分排序）
python -m aqlab.cli screen --symbols 30 --days 500 --top 5

# 5) 用你自己的 CSV 回测（支持中文列名：日期/开盘/最高/最低/收盘/成交量）
python -m aqlab.cli run --csv data/raw/600519.csv --strategy ma_cross --params fast=10,slow=30

# 6) 可选：下载真实日线（需要 tushare token 或 akshare）
export TUSHARE_TOKEN=xxxx          # Windows: set TUSHARE_TOKEN=xxxx
python -m aqlab.cli fetch --source akshare --symbol 600519 --start 2022-01-01 --end 2024-12-31 --out data/raw/600519.csv
```

## 实测输出（可复现，合成行情）

**上涨行情（seed=25，标的累计 +128.5%）**

| 策略 | 总收益% | 年化% | 年化波动% | Sharpe | 最大回撤% | 交易数 | 胜率% |
|:--- |---:|---:|---:|---:|---:|---:|---:|
| buy_and_hold | 128.39 | 31.98 | 28.70 | 1.111 | -28.55 | 1 | 100 |
| ma_cross | 55.80 | 16.07 | 22.34 | 0.778 | -25.92 | 12 | 50.00 |
| momentum | 9.83 | 3.20 | 14.81 | 0.287 | -18.23 | 367 | 50.95 |
| mean_reversion | 41.60 | 12.40 | 13.65 | 0.924 | -23.30 | 20 | 85.00 |

**下跌行情（seed=7，标的累计 -65.1%）**

| 策略 | 总收益% | 年化% | 年化波动% | Sharpe | 最大回撤% | 交易数 | 胜率% |
|:--- |---:|---:|---:|---:|---:|---:|---:|
| buy_and_hold | -65.15 | -29.83 | 26.94 | -1.179 | -72.91 | 1 | 0.00 |
| ma_cross | -21.35 | -7.75 | 15.39 | -0.447 | -37.28 | 13 | 23.08 |
| momentum | -10.98 | -3.83 | 9.14 | -0.382 | -19.76 | 155 | 40.65 |
| mean_reversion | -45.53 | -18.46 | 19.00 | -0.979 | -53.88 | 30 | 50.00 |

**怎么读这两张表（这也是我在面试里会讲的点）**

- 牛市里**没有任何策略跑赢买入持有**——这不是 bug，而是回测该暴露的事实：择时策略在单边上涨中天然吃亏。
- 熊市里 `momentum` / `ma_cross` 把 -65% 的回撤压到 -11% / -21%，说明它们的价值在**风控**而不是收益增强。
- 一个只看单一行情就宣称"策略有效"的分析，是不值得信的；**跨越不同行情做对照**才是起点。

## 设计原则

| 原则 | 落地方式 |
| --- | --- |
| 策略只表达意图 | `Strategy.positions()` 返回目标仓位；执行、成本、延迟全部在 `backtest.py`，策略无法偷看未来 |
| 无未来函数 | 引擎统一 `positions.shift(1)`；`tests/test_backtest.py::test_no_lookahead_oracle_loses_money` 用"偷看信号必然亏钱"反证 |
| 成本显式 | `BacktestConfig(fee_bps=3, slippage_bps=2)`，逐 bar 的 `cost`、`turnover` 全部落盘 |
| 可复现 | 合成数据由种子决定；测试与 demo 无需网络 |
| 可解释 | 筛选分数 = 因子横截面 z-score 的加权和，权重写在 `DEFAULT_WEIGHTS` 里，不藏黑箱 |
| 失败要早 | 参数校验放在 `Strategy.validate()` / `BacktestConfig.__post_init__`，错误信息直接指出问题 |

## 目录结构

```
zgnb-skill/
├── src/aqlab/
│   ├── data.py          # 数据归一化（含中文列名）、确定性合成行情、可选 tushare/akshare 抓取
│   ├── indicators.py    # SMA/EMA/RSI/ATR/Donchian/z-score/波动率（全部因果）
│   ├── strategies.py    # 内置策略 + 注册表 + 工厂
│   ├── backtest.py      # 执行引擎（延迟、成本、仓位裁剪）、交易流水提取、组合聚合
│   ├── metrics.py       # 收益/年化/波动/Sharpe/Sortino/回撤/Calmar/换手/胜率
│   ├── screen.py        # 横截面因子表与加权打分排序
│   ├── report.py        # markdown 报告 + metrics.json + equity.csv + trades.csv
│   └── cli.py           # demo / run / screen / fetch
├── tests/               # 33 个用例：数据、指标、回测（含无未来函数反证）、指标计算、筛选
├── examples/            # 离线 demo 脚本 + 示例报告
├── docs/                # 架构说明与路线图
└── .github/workflows/   # CI：多 Python 版本跑 pytest
```

## 路线图

- **v0.2（进行中）LLM / Agent 研究层**：把"数据查询、因子计算、回测执行"暴露成工具（tool calling），让模型生成假设 → 自动回测 → 汇总成可复核的研究笔记；并建立**评测集**（含"看起来合理但其实错"的陷阱任务）来量化代理的可靠性。
- **v0.3** 组合层：权重优化、行业/风格暴露、换手约束。
- **v0.4** 数据质量：缺口/停牌/复权一致性检查，多源交叉校验。
- **v0.5** 可视化报告：净值/回撤/因子贡献图。

详见 [`docs/ROADMAP.md`](docs/ROADMAP.md) 与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 免责声明

本项目仅用于量化研究与软件工程演示。所有回测均为历史模拟，**不构成任何投资建议**，历史表现不代表未来收益。真实交易还需考虑流动性、涨跌停、停牌、冲击成本等本项目未建模的因素。

## 许可

MIT © 2026 Weibin Zhang。本仓库为 clean-room 原创实现。
