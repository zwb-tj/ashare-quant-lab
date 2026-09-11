# A-Share Quant Lab (`aqlab`)

[![ci](https://github.com/zwb-tj/ashare-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/zwb-tj/ashare-quant-lab/actions/workflows/ci.yml)

**一个从零实现、可复现的 A 股选股 + 回测实验室**，包含显式的执行成本模型、无未来函数的信号执行、横截面因子打分、**可审计的 LLM 研究代理（tool calling）**，以及一条可测试的 CLI。仓库为 **clean-room 原创实现**：不含任何第三方项目代码或整理内容。

> 不是又一个"翻倍策略"仓库。这是一个把**研究纪律**写在代码里的工具：成本要显式、执行要延迟一根 K 线、策略要能被证伪、**代理说的每个数字都必须来自工具**。

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

# 2) 跑测试（60 个用例，全部离线，无需网络/API key）
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

# 7) 代理可靠性评测（离线，无需 API key）
python -m aqlab.cli eval --mode offline

# 8) 用真实模型跑研究代理（需 OpenAI 兼容的 API key，如 DeepSeek）
export AQLAB_LLM_API_KEY=sk-xxxx   # 可选：AQLAB_LLM_BASE_URL / AQLAB_LLM_MODEL
python -m aqlab.cli agent --question "用 ma_cross(10,30) 回测 SYN001，给我总收益和 Sharpe" --trace output/trace.json
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

## LLM / Agent 研究层（v0.2，已实现）

这一层的目标不是"让模型随便聊行情"，而是让 **断言可核查、可靠性可量化**：

```
问题 ──► 模型(plan) ──► 工具调用 ──► 只读工具执行 ──► 观察结果 ──► ... ──► 结论文本
                             │                                              │
                             └────────────── 全步骤 trace ──────────────────┘
                                              │
                                    评测：数字是否全部来自工具？
```

**只读工具层**（`aqlab/tools.py`，6 个工具，带 JSON Schema）：
`list_strategies` / `describe_data` / `get_bars` / `compute_indicator` / `run_backtest` / `screen_universe`。
工具不写文件、不下单、不改状态；**错误以 `ok=false` 返回**（例如标的不存在、参数非法），迫使模型选择"弃答"而不是编数字。

**代理循环**（`aqlab/agent.py`）：plan → 调用工具 → 观察 → 再决策，最多 `max_steps` 步；每一步（含工具入参与返回）都写入 trace，可回放审计。支持两种客户端：任何 OpenAI 兼容端点（仅用标准库实现，DeepSeek/Moonshot/vLLM/Ollama 网关均可）与用于离线测试的 `ScriptedClient`。

**评测集**（`aqlab/evaluation.py`）：5 个任务覆盖三类情形——正常任务、**陷阱任务**（标的不存在 / 参数非法，正确行为是弃答）、**回归任务**（重复执行结果必须一致）。指标定义：

| 指标 | 含义 |
| --- | --- |
| `grounded_number_rate` | 答案里的数字能在工具返回中找到的比例（**未落地的数字 = 幻觉**） |
| `hallucinated_tasks` | 至少出现一个未落地数字的任务数 |
| `abstain_accuracy` | 陷阱任务上"明确弃答"的比例 |
| `tool_success_rate` | 工具调用返回 `ok=true` 的比例（差距来自刻意设计的陷阱任务） |
| `regression_consistency` | 同一任务重复执行结果一致的比例 |

**离线评测实测输出**（`python -m aqlab.cli eval --mode offline`）：

| 指标 | 值 |
| --- | --- |
| tasks | 5 |
| answered_rate | 1.000 |
| tool_calls | 6 |
| tool_success_rate | 0.667 |
| numbers_total | 10 |
| grounded_number_rate | 0.900 |
| hallucinated_tasks | 1 |
| abstain_accuracy | 1.000 |
| regression_consistency | 1.000 |

读法：唯一被判定为幻觉的任务，是脚本里**刻意让模型"凭记忆报 999.99%"**的那条，`[999.99]` 被明确标记为未落地数字；两个陷阱任务都被正确识别为"证据不足"；回归任务两次执行结果一致。这套机制的价值在于：**换任何模型进来，都能得到同一把尺子的分数**，而不是靠感觉说"这个模型挺靠谱"。

## 设计原则

| 原则 | 落地方式 |
| --- | --- |
| 策略只表达意图 | `Strategy.positions()` 返回目标仓位；执行、成本、延迟全部在 `backtest.py`，策略无法偷看未来 |
| 无未来函数 | 引擎统一 `positions.shift(1)`；`tests/test_backtest.py::test_no_lookahead_oracle_loses_money` 用"偷看信号必然亏钱"反证 |
| 成本显式 | `BacktestConfig(fee_bps=3, slippage_bps=2)`，逐 bar 的 `cost`、`turnover` 全部落盘 |
| 可复现 | 合成数据由种子决定；测试与 demo 无需网络 |
| 可解释 | 筛选分数 = 因子横截面 z-score 的加权和，权重写在 `DEFAULT_WEIGHTS` 里，不藏黑箱 |
| 失败要早 | 参数校验放在 `Strategy.validate()` / `BacktestConfig.__post_init__`，错误信息直接指出问题 |

## 每日流水线：17:30 打分 → 飞书推送（v0.3）

```bash
# 默认 dry-run（合成数据验证流程，不推送）
python -m aqlab.cli daily --symbols-count 30 --days 500 --top 8

# 用本地 CSV 目录当数据源
python -m aqlab.cli daily --data-dir data/raw --top 10

# 真实数据 + 真推送（需要 TUSHARE_TOKEN 与 FEISHU_WEBHOOK）
set TUSHARE_TOKEN=xxxx
set FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
python -m aqlab.cli daily --tushare --symbols 600519.SH,000001.SZ,300750.SZ --start 2023-01-01 --push

# 覆盖规则参数（如把单针的均线换成 30、量价齐升改成 3 日确认）
python -m aqlab.cli daily --rule needle_below_ma.ma_window=30 --rule volume_price_surge.confirm_days=3

# Windows 定时任务：每周一至周五 17:30 自动跑 + 真实推送
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_daily.ps1 -DryRun   # 先试跑
```

**流程与产物**

```
17:30 ──► 数据源（tushare / 本地 CSV / 合成）
            │  每只票缓存为 CSV（断网/复跑不会丢昨天的数据；取数失败但有缓存 → 用缓存并标记降级）
            ▼
        活跃市值开关（可选，滞回开关：快线≥慢线+on 开；≤慢线+off 关；中间保持）
            ▼
        逐票逐规则打分 ─► 加权综合分（默认 0.4/0.3/0.3）─► 横截面排名
            ▼
        output/daily/daily-YYYY-MM-DD.md + .json   ──►  飞书卡片（不签名自定义机器人）
```

**规则是插件，阈值是配置**。仓库里实现的是通用形态 + 中性命名；你自己的参数（均线、档位、放大倍数、确认天数、开关阈值）通过配置传进去即可：

| 通用规则 | 实现要点 | 可对应的买方概念 |
| --- | --- | --- |
| `tiered_pullback` | 回踩均线买点，按"下影扎入均线的深度"分档打分（tiers/tier_scores 可配 3 档或更多） | B1 / B2 / B3 这类分档买点 |
| `needle_below_ma` | 单针下探均线后收回：长下影 + 收盘站回均线上方（`ma_window` 可设 20 / 30） | 单针下 20 / 单针下 30 |
| `volume_price_surge` | 量价齐升：涨幅达标 + 放量，`confirm_days=1` 为单日、`=3` 为三日确认 | 量价齐升 V1 / V3 |
| `ActivityValueGate` | 活跃市值开关：`Σ(close×volume)` 的双均线 + 滞回开关（on/off 阈值分离） | 0AMV 活跃市值 + 开关规则 |

> 公开仓库里刻意使用**中性命名**（`tiered_pullback` / `needle_below_ma` / `volume_price_surge` / `ActivityValueGate`）：规则逻辑与阈值是你自己的配置，命名与描述也建议用你自己的说法，避免把仓库绑到任何人的品牌或课程内容上。

**本期实测输出（dry-run，合成票池，交易日 2023-12-01）**

```
票池 30 只 ｜ 🔴 活跃市值开关关闭（不产生新买点）
规则权重：{'tiered_pullback': 0.4, 'needle_below_ma': 0.3, 'volume_price_surge': 0.3}

| 排名 | 代码 | 收盘 | 综合分 | 触发规则 | tiered_pullback | needle_below_ma | volume_price_surge |
| 1 | SYN028 | 1145.68 | 40.1 | tiered_pullback、needle_below_ma | 1.0 | 0.0034 | 0.0 |
| 2 | SYN029 |  531.19 | 40.0 | tiered_pullback                  | 1.0 | 0.0    | 0.0 |
```

行为约定（都是有测试的）：**综合分为 0 的标的不会被列进"选股"**（不足 top_n 就如实写"本期仅 N 只触发"）；开关关闭时仍输出观察名单，但明确标注"不产生新买点"。

## 目录结构

```
ashare-quant-lab/
├── src/aqlab/
│   ├── data.py          # 数据归一化（含中文列名）、确定性合成行情、tushare/akshare 抓取与缓存
│   ├── indicators.py    # SMA/EMA/RSI/ATR/Donchian/z-score/波动率（全部因果）
│   ├── strategies.py    # 内置策略 + 注册表 + 工厂
│   ├── backtest.py      # 执行引擎（延迟、成本、仓位裁剪）、交易流水提取、组合聚合
│   ├── metrics.py       # 收益/年化/波动/Sharpe/Sortino/回撤/Calmar/换手/胜率
│   ├── screen.py        # 横截面因子表与加权打分排序
│   ├── rules.py         # 打分规则插件 + 活跃市值滞回开关
│   ├── pipeline.py      # 每日流水线：取数 → 开关 → 打分 → 排名 → 报告 → 推送
│   ├── notify.py        # 飞书卡片 / 控制台 dry-run 通知
│   ├── report.py        # markdown 报告 + metrics.json + equity.csv + trades.csv
│   ├── tools.py         # 只读工具层（JSON Schema，错误以 ok=false 返回）
│   ├── agent.py         # 有界 plan-act-observe 代理循环 + trace + 两种 LLM 客户端
│   ├── evaluation.py    # 评测集与指标：落地率/幻觉/弃答/回归一致性
│   └── cli.py           # demo / run / screen / fetch / eval / agent / daily
├── scripts/             # run_daily.ps1（跑当日任务）、register_task.ps1（注册 17:30 计划任务）
├── tests/               # 87 个用例：数据、指标、回测（含无未来函数反证）、工具、代理、评测、规则、流水线、通知
├── examples/            # 离线 demo 脚本 + 示例报告
├── docs/                # 架构说明与路线图
└── .github/workflows/   # CI：多 Python 版本跑 pytest
```

## 个人策略集：B1/B2/B3 · 单针下20/30 · 量价齐升V3 · 0AMV 活跃市值

这一层是作者自己交易研究中的规则，已按 **参数化 + 档案（profile）** 的方式重新实现（`src/aqlab/rules_zgnb.py`、`profiles.py`），阈值全部可配置：

| 规则 | 命中口径（默认值） |
| --- | --- |
| `b1_graded` | **B1 梯度打分（5 硬 + 4 软）**：硬性 —— J ≤ 13、近 15 日存在放量日（量 > 前一日×2）、极致缩量（当日量 < 近 15 日最高 / 2.5）、双线多头（白线 > 黄线 且 收盘 ≥ 黄线×0.97，需 ≥114 根）、前 N 低点未破（近 15 日最低 ≥ 近 30 日最低×0.97）；排除近 10 日放量大阴线。软性（各 +10 分）—— 涨跌幅 ∈ [-2%,+1.8%]、振幅 < 4%、盈亏比 ≥ 3（止损=近 15 日最低×0.97，目标=近 15 日最高×1.02）；评分 = 60 + 10×软通过数 |
| `b1_opportunity` | 简化版（保留兼容）：J ≤ -10；涨幅 -2%~+1.8%；振幅 ≤ 7%；累计换手 < 38%（有换手率列时启用，缺失则跳过并在报告里说明） |
| `brick_green_to_red` | 砖型图"绿转红"：`XG = 绿转红 AND 视觉红柱 ≥ 昨视觉绿柱 × 0.6667`；评分梯度化（`min(1, 强度比/0.6667)`，满足 XG 给满分） |
| 砖型过滤门 `brick_filter_mask` | 门 1（入场）：只允许红砖第 1~2 块进；门 2（禁买）：红砖 ≥ 4 块一律禁买（`use_brick_filter=True` 打开） |
| `b2_confirm` | B1 后 3 个交易日内，涨幅 ≥ 4%，J < 55，且放量（量 > 前一日） |
| `b3_confirm` | B2 后出现十字星/小阴线（实体 ≤ 2%），且平开（\|开盘/前收-1\| ≤ 1%） |
| `needle_rsl` | 单针下20：RSL(3) ≤ 20 且 RSL(21) ≥ 80；单针下30：RSL(3) < 30 且 RSL(21) > 85（`RSL(N)=100*(C-LLV(L,N))/(HHV(C,N)-LLV(L,N))`） |
| `volume_price_v3` | 连续 2 日阳线且价格连续创新高；连续 2 日量增；当日涨幅 2%~6%；白线 > 黄线且收盘 ≥ 黄线×0.97；J < 60。评分 = 0.70 基础分 + 涨幅 3~5%（+0.10）+ 量 > 昨日 1.5 倍（+0.10）+ J < 50（+0.10） |
| `ActiveMarketValueGate` | 0AMV 活跃市值：活筹置换衰减 `A_t = A_{t-1} × 0.92 × (1-换手率_t) + 成交量_t`，指数 `Σ A×close`；**开仓**：单日 ≥ +5%（特别强）/ ≥ +4%（强）/ 连续 2 日合计 ≥ +4% 且窗口内无 ≤ -2.3% 日（一般）/ 连续 3 日合计 ≥ +4% 同条件（较弱）；**关仓**：持仓状态下单日 ≤ -2.3%（次日只卖不买） |

辅助指标（`src/aqlab/indicators_extra.py`）：`rsl`、`kdj`(9,3,3，J=3K-2D)、`amplitude`、`white_line`=EMA(EMA(C,10),10)、`yellow_line`=(MA14+MA28+MA57+MA114)/4、**`sma_tdx`（通达信 `SMA(X,N,M)`）**、**`brick_chart`（砖型图）**。

### 砖型图（按你的公式逐行等价实现）

```
VAR1A := (HHV(H,4) - C) / (HHV(H,4) - LLV(L,4)) * 100 - 90
VAR2A := SMA(VAR1A,4,1) + 100
VAR3A := (C - LLV(L,4)) / (HHV(H,4) - LLV(L,4)) * 100
VAR4A := SMA(VAR3A,6,1)
VAR5A := SMA(VAR4A,6,1) + 100
VAR6A := VAR5A - VAR2A
砖型图 := IF(VAR6A > 4, VAR6A - 4, 0)
视觉红柱 := IF(砖型图 > 昨砖型图, 砖型图 - 昨砖型图, 0)
视觉绿柱 := IF(砖型图 < 昨砖型图, 昨砖型图 - 砖型图, 0)
绿转红   := 昨视觉绿柱 > 0 AND 视觉红柱 > 0
强度比   := IF(绿转红, ROUND(视觉红柱 / 昨视觉绿柱, 2), 0)
XG       := 绿转红 AND 视觉红柱 >= 昨视觉绿柱 * 0.6667
```

`brick_chart()` 返回全部中间列（`var1a`…`var6a`、`brick`、`red`、`green`、`green_to_red`、`strength_ratio`、`xg`），`brick_streaks()` 再给出**红砖/绿砖连续块数**（红砖第 N 块），供门 1 / 门 2 使用。竖线（红/绿柱）与图标（XG）属于绘图指令，这里不做，值本身完整保留。

```bash
# 用个人档案跑当日流水线（默认 0AMV 开关，dry-run）
python -m aqlab.cli daily --profile zgnb_full --symbols-count 40 --days 600 --top 10

# 单规则档案
python -m aqlab.cli daily --profile needle_20      # 或 needle_30 / b1 / b1_brick / b2 / b3 / volume_price_v3 / brick_green_to_red
python -m aqlab.cli daily --profile zgnb_needle30  # 单针下30 + 量价齐升V3 组合
python -m aqlab.cli daily --profile zgnb_brick     # B1(带砖型过滤) + 绿转红 + V3 + 单针下20

# 参数覆盖 + 切换开关口径
python -m aqlab.cli daily --profile b1 --rule b1_opportunity.j_max=-15 --gate activity
python -m aqlab.cli daily --profile zgnb_full --no-gate      # 不启用市场开关
```

实测输出（合成票池 40 只，dry-run）：

```
2024-04-19 ｜ 票池 40 只（可用 40 只）｜ 🟢 开关打开（hold）
- 开关数据口径：turnover: 数据缺少换手率列，按 250 日最大成交量×1.5 估计流通股本
- 本期仅 3 只标的触发规则（不足 top_n=6）
规则权重：{'b1_opportunity': 0.25, 'b2_confirm': 0.2, 'b3_confirm': 0.15, 'needle_rsl': 0.15, 'volume_price_v3': 0.25}

| 排名 | 代码 | 收盘 | 综合分 | 触发规则 | b1 | b2 | b3 | needle_rsl | v3 |
| 1 | SYN009 | 173.01 | 25.0 | b1_opportunity | 1.0 | 0 | 0 | 0 | 0 |
| 2 | SYN026 | 1079.05 | 20.0 | volume_price_v3 | 0 | 0 | 0 | 0 | 0.8 |
| 3 | SYN002 | 86.22 | 15.0 | needle_rsl | 0 | 0 | 0 | 1.0 | 0 |
```

> ⚠️ **数据依赖说明（不隐藏）**：B1 的"累计换手率 < 38%"需要真实换手率（Tushare `daily_basic.turnover_rate`）。数据里没有 `turnover` 列时本项目**跳过**该条件并写进报告备注；0AMV 的换手率缺失时按 `250 日最大成交量×1.5` 估计流通股本——这是估计值，报告里同样会标注。要精确复用请先同步 `daily_basic`。

## 规则证据研究：把"有效"变成可复核的数字（v0.4.2）

规则谁都会写，**能拿出证据的少**。`aqlab study` 对档案里的每条规则做事件研究：
先找出它历史上触发过的每一根 bar，再统计这些触发点的**未来 1/3/5/10 日收益**，最后与
"票池所有 bar"的基线对比——`超额均值`/`超额胜率` 才是规则的真实价值。

```bash
python -m aqlab.cli study --profile zgnb_full --symbols-count 40 --days 600 --horizons 1,3,5,10
```

实测输出（合成票池 40 只 / 600 交易日，节选）：

| 规则 | 持有日 | 信号数 | 均值% | 胜率% | 基线均值% | 基线胜率% | 超额均值% | 超额胜率% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| needle_rsl | 1 | 506 | 0.44 | 59.60 | 0.27 | 56.58 | **+0.17** | **+3.03** |
| needle_rsl | 3 | 506 | 1.14 | 67.47 | 0.82 | 61.08 | **+0.32** | **+6.38** |
| needle_rsl | 10 | 506 | 3.42 | 73.52 | 2.78 | 68.84 | **+0.63** | **+4.68** |
| b1_graded | 5 | 161 | 1.44 | 66.46 | 1.38 | 64.15 | +0.07 | +2.31 |
| b1_graded | 1 | 161 | 0.20 | 49.07 | 0.27 | 56.58 | **-0.07** | **-7.51** |
| volume_price_v3 | 3 | 22 | 0.60 | 66.67 | 0.82 | 61.08 | -0.23 | +5.58 |
| b2_confirm / b3_confirm | — | **0** | — | — | — | — | — | — |

这张表立刻给出三条结论（比任何"我觉得有效"都硬）：

1. `needle_rsl` 在这份数据上有**稳定正超额**（持有越久超额越明显）——值得继续研究；
2. `b1_graded` 短期（1 日）**跑输基线**，5 日才勉强打平——说明它是"结构型买点"，不是短线信号；
3. `b2_confirm`/`b3_confirm` **一次都没触发**（因为默认前置是严格的梯度 B1），提示"B2/B3 依赖的 B1 太苛刻，需要放宽或改用简化口径"——这是工程上立刻可行动的信息。

> 每个信号点的收益都是 close-to-close 的未来收益，含尾部 NaN 剔除；`信号数 < 20` 时不要当真（表格会照实显示 n）。

## 持仓与离场管理（v0.4.3）

选股回答"买什么"，`position.py` 回答"买了之后怎么办"。规则全部来自作者自己的口径，并按风控优先级排序：

| 优先级 | 规则 | 口径 |
| --- | --- | --- |
| 1 | 短线硬止损（可选） | -2%（"一日游"心态） |
| 2 | 结构止损 | 买入 K 线最低价 / 前低 / 平台下沿，下浮 3%~5%，且不宽于 -5% |
| 3 | 脱离成本止损 | 有过 +3% 浮盈后跌回成本（盈转亏）立即走 |
| 4 | 第一次止盈 | 脱离成本 +3% 减半仓 |
| 5 | BBI 离场 | 收盘连续 2 日跌破 BBI（(MA3+MA6+MA12+MA24)/4）→ 清仓 |
| 6 | 波段目标 | +10%~20% |
| 7 | 时间止损 | 持有到 4-6 根 K 线：有 +5% 浮盈走"盈利时间止损"，否则平价时间止损 |

外加两个工具：

- **`plan_position()`** —— 给出交易计划：结构止损价、三档目标、盈亏比、**3-2-2 建仓阵型**（侦察兵 30% / 主力军 25% / 预备队 45%）；
- **`defend_score()`** —— **防卖飞评分（5 分制）**：收盘涨 + 未破 BBI + 非放量阴线 + 趋势向上 + J 未死叉 → `4-5 分持有 / 3 分减半 / <3 分准备离场`。

```bash
python -m aqlab.cli plan --symbols-count 6 --days 300 --symbol SYN001
```

```
| entry_price | stop_price | stop_pct | first_target | second_target | swing_target | risk_reward | weights |
| 120.894 | 117.094 | -0.0314 | 124.52 | 126.938 | 139.028 | 2.07 | scout 0.3 / main 0.25 / reserve 0.45 |

近 5 根防卖飞评分：4 分（持有）→ 2 分（准备离场）×4
```

`simulate_exit()` / `simulate_signals()` 可以拿任意信号序列跑完整交易（**信号次日收盘入场**，无未来函数），逐笔返回离场原因与收益，方便与回测引擎对账。

## 路线图

- ✅ **v0.2（已完成）LLM / Agent 研究层**：只读工具层（6 个工具，JSON Schema）+ 有界代理循环 + 全步骤 trace + 评测集（含"看起来合理但其实错"的陷阱任务）与落地率/幻觉率/弃答率指标。
- ✅ **v0.3（已完成）每日流水线**：规则插件化（回踩分档 / 单针下均线 / 量价齐升）+ 活跃市值滞回开关 + tushare 取数与本地缓存（断网降级）+ 飞书卡片推送 + Windows 17:30 计划任务 + 全链路离线测试。
- **v0.4** 组合层：权重优化、行业/风格暴露、换手约束。
- **v0.5** 数据质量：缺口/停牌/复权一致性检查，多源交叉校验。
- **v0.6** 可视化报告：净值/回撤/因子贡献图。

详见 [`docs/ROADMAP.md`](docs/ROADMAP.md) 与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 免责声明

本项目仅用于量化研究与软件工程演示。所有回测均为历史模拟，**不构成任何投资建议**，历史表现不代表未来收益。真实交易还需考虑流动性、涨跌停、停牌、冲击成本等本项目未建模的因素。

## 许可

MIT © 2026 Weibin Zhang。本仓库为 clean-room 原创实现。
