# 路线图

## v0.2 · LLM / Agent 研究层（✅ 已完成，2026-09）

**目标**：让模型承担"提出假设 → 调用工具 → 得到证据 → 写成结论"的循环，同时把可靠性**量化**出来。

已交付：

1. **工具层（tool calling）** — `src/aqlab/tools.py`，6 个只读工具，带 JSON Schema：
   `list_strategies` / `describe_data` / `get_bars` / `compute_indicator` / `run_backtest` / `screen_universe`。
   - 只读、幂等、参数校验；**错误以 `ok=false` 返回**（未知标的、非法参数、历史不足）；
   - 可挂 `SyntheticDataSource`（离线）或 `CsvDataSource`（本地 CSV 目录）。
2. **代理循环** — `src/aqlab/agent.py`：plan → 调用工具 → 观察 → 再决策，`max_steps` 硬上限；
   每步写入 trace（含工具入参与返回），可回放审计；支持原生 tool_calls 与文本形式 JSON 两种调用格式。
3. **客户端** — `OpenAICompatClient`（仅标准库，任何 OpenAI 兼容端点：DeepSeek / Moonshot / vLLM / Ollama 网关）
   与 `ScriptedClient`（离线确定性测试）。
4. **评测集与指标** — `src/aqlab/evaluation.py`：5 个任务（正常 / 标的不存在 / 参数非法 / 幻觉诱导 / 回归重复），
   指标为 `grounded_number_rate`、`hallucinated_tasks`、`abstain_accuracy`、`tool_success_rate`、`regression_consistency`。
   离线实测：`grounded_number_rate=0.900`，`hallucinated_tasks=1`（刻意诱导的那条被抓出），`abstain_accuracy=1.000`，`regression_consistency=1.000`。
5. **CLI** — `aqlab eval --mode offline|live`、`aqlab agent --question ... --trace ...`；无 key 时给出可执行提示而非报错崩溃。
6. **测试** — 工具、代理（含 `max_steps` 与小工具错误回灌）、评测（含"坏代理必须低分"）共 23 个新用例，全套 60 个用例离线通过。

### v0.2 遗留（下一轮继续）
- 评测集扩展到 20–30 个任务，加入"结论与指标自相矛盾"类陷阱；
- 代理产出的研究笔记落盘为可复核 markdown（引用每次工具调用的证据编号）；
- 多轮对话与工具结果的上下文压缩策略。
4. **防护**：所有模型产出的数字必须来自工具返回值（禁止模型自行算数）；报告里标注每句话的证据来源。

## v0.3 · 每日流水线（✅ 已完成，2026-09）

**目标**：把"取数 → 打分 → 排名 → 推送"变成可定时、可离线验证、断网也不丢数据的日常任务。

已交付：

1. **规则插件层** — `src/aqlab/rules.py`
   - `TieredPullback`：回踩均线买点，按扎入深度分档打分（档位与分值可配）；
   - `NeedleBelowMA`：单针下探均线后收回（`ma_window` 可设 20/30）；
   - `VolumePriceSurge`：量价齐升，`confirm_days` 支持单日/多日确认；
   - `ActivityValueGate`：活跃市值（`Σ close×volume`）双均线 + **滞回开关**（on/off 阈值分离）；
   - 规则返回 `[0,1]` 分数序列（NaN→0），可用 `--rule name.param=value` 覆盖参数。
2. **每日流水线** — `src/aqlab/pipeline.py`：源 → 开关 → 逐票逐规则打分 → 加权综合分 → 横截面排名 → markdown/JSON 报告 → 推送；**综合分为 0 的标的不进选股列表**，开关关闭时输出观察名单并明确标注。
3. **数据与缓存** — `TushareDataSource`：逐票 CSV 缓存；取数失败但有缓存则用缓存并标记 `degraded`，没有缓存则 `KeyError`（不静默失败）。
4. **通知** — `src/aqlab/notify.py`：飞书交互卡片（不签名自定义机器人，仅标准库 POST）+ 控制台 dry-run；未配置 webhook / 配了 secret 都有明确失败原因，不假装成功。
5. **定时** — `scripts/run_daily.ps1`（读 `.env`、落日志到 `output/logs/`）+ `scripts/register_task.ps1`（注册周一至周五 17:30 的 Windows 计划任务）。
6. **测试** — 规则、流水线（含 0 分不入选、开关关闭、确定性、参数覆盖）、通知（卡片结构、dry-run 不发网络请求、POST 成功/失败分支）、Tushare 缓存与降级，共 27 个新用例，全套 87 个用例离线通过。

### 后续可选增强
- 规则组合的回测校验：把"每日打分"的历史序列做成可回测的信号（避免只看当日榜单的错觉）；
- 报告里加入"本票近 N 日信号历史"，便于人工复核；
- 全市场扫描时的并发取数与限速（Tushare 频率限制）。

## v0.4 · 个人策略集（✅ 已完成，2026-09）

**目标**：把作者自己交易研究里的规则做成可配置、可测试、能进每日流水线的模块。

已交付：

1. **辅助指标** — `src/aqlab/indicators_extra.py`：`rsl`（相对强度定位）、`kdj`(9,3,3)、`amplitude`、`white_line`(EMA(EMA(C,10),10))、`yellow_line`((MA14+MA28+MA57+MA114)/4)。
2. **规则** — `src/aqlab/rules_zgnb.py`：
   - `B1Opportunity`（J ≤ -10、涨幅 -2%~1.8%、振幅 ≤7%、累计换手 <38%，换手缺失时跳过并提示）
   - `B2Confirm`（B1 后 3 日内涨幅 ≥4%、J < 55、放量）
   - `B3Confirm`（B2 后十字星/小阴线 + 平开）
   - `NeedleRSL`（单针下20：RSL3 ≤20 且 RSL21 ≥80；单针下30：RSL3 <30 且 RSL21 >85）
   - `VolumePriceV3`（五条硬性条件 + 加分制评分）
   - `ActiveMarketValueGate`（0AMV 活筹置换衰减模型 ρ=0.92 + 波段开关：单日 ≥+5%/+4%、连续 2/3 日合计 ≥+4% 且窗口无 ≤-2.3% 日；持仓遇 ≤-2.3% 关仓）
3. **档案（profile）** — `src/aqlab/profiles.py`：`generic / b1 / b2 / b3 / needle_20 / needle_30 / volume_price_v3 / zgnb_full / zgnb_needle30`，CLI `--profile` 一键切换；`--gate amv|activity|none` 选择开关口径。
4. **报告增强** — 每日报告输出开关触发档位（strong/very_strong/normal/weak/hold/close）与取数口径提示。
5. **测试** — 20 个新用例：指标公式、单针触发/不触发、B1（含换手条件与硬性开关）、B2、B3、V3（用受控白/黄线与 J 隔离五条件）、0AMV 递推收敛与波段状态机、档案注册与端到端流水线；全套 124 个用例离线通过。

### 后续可选
- ~~砖型图（红/绿砖）过滤门~~ → **已完成，见 v0.4.1**；
- 把"每日打分序列"接回回测引擎，做规则组合的 walk-forward 校验；
- `daily_basic` 换手率同步（让 0AMV 的换手率与简化版 B1 的 38% 都用真实值）。

## v0.4.1 · 砖型图与 B1 梯度打分（✅ 已完成，2026-09）

按作者提供的公式与旧版口径修正：

1. **B1 修正** — 新增 `B1Graded`（旧版 `B1Opportunity` 的 J ≤ -10 过于极端，真实口径为 **J ≤ 13**）：
   - 5 条硬性：J ≤ 13、近 15 日有放量日（>前日×2）、极致缩量（<近 15 日最高/2.5）、双线多头（白线>黄线且收盘≥黄线×0.97，需 ≥114 根）、前 N 低点未破（近 15 日最低 ≥ 近 30 日最低×0.97）；并排除近 10 日放量大阴线（S1）；
   - 4 条软性加分：涨跌幅 ∈[-2%,+1.8%]、振幅 <4%、盈亏比 ≥3、双 30（原始规则搁置）；
   - 评分 = 60 + 10×软通过数（→ /100 输出），并保留 `detail()` 输出每条条件明细便于复核。
2. **砖型图** — `indicators_extra.sma_tdx`（通达信 `SMA(X,N,M)`）与 `brick_chart`（VAR1A…VAR6A、砖型图、视觉红/绿柱、绿转红、强度比、XG 全列），`brick_streaks` 给出红/绿砖连续块数；绘图指令（STICKLINE/DRAWICON）不实现，数值完整。
3. **规则与门** — `BrickGreenToRed`（梯度评分：`min(1, 强度比/0.6667)`，满足 XG 给满分）与 `brick_filter_mask`（门 1：红砖 ≤2 才允许入场；门 2：红砖 ≥4 禁买）；`B1Graded(use_brick_filter=True)` 可把门并入 B1。
4. **档案** — 新增 `b1`（= b1_graded）、`b1_simple`、`b1_brick`、`brick_green_to_red`、`zgnb_brick`；`zgnb_full` 改用 `b1_graded`；`B2Confirm` 默认以梯度 B1 作为前置信号（可切换 `b1_rule="b1_opportunity"`）。
5. **测试** — 新增 10 个用例（SMA 手算校验、砖型图恒等式与无未来函数、XG 触发、红砖门、B1 五硬条件与软性加分、S1 排除、档案注册）；全套 **124** 个用例离线通过。

## v0.4.2 · 规则事件研究（✅ 已完成，2026-09）

**目标**：让"规则有效"变成可复核的统计，而不是口头结论。

- `src/aqlab/study.py`：`forward_returns`（未来 N 日收益，纯因果）、`rule_event_study`（单规则逐 horizon 统计）、`baseline_stats`（票池所有 bar 的基线）、`study_profile`（档案级对比，输出超额均值/超额胜率）、`format_study` / `write_study`。
- CLI：`aqlab study --profile zgnb_full --horizons 1,3,5,10`，产物 `output/study/study.md + study.csv + baseline.csv + study.json`。
- 实测（合成票池 40 只 / 600 天）：`needle_rsl` 全 horizon 正超额（1 日 +0.17pp / 3 日 +6.4pp 胜率）；`b1_graded` 1 日跑输基线、5 日打平（结构型买点，非短线信号）；`b2_confirm`/`b3_confirm` 零触发（提示前置 B1 过严）。**表里照实显示**，不挑选好看的结论。
- 测试：9 个新用例（手算校验的未来收益、桩规则事件研究、基线池化、确定性、零信号规则、产物落盘）。

## v0.4.3 · 持仓与离场管理（✅ 已完成，2026-09）

**目标**：把"买了之后怎么办"从经验变成可执行、可回测的规则。

- `src/aqlab/position.py`：
  - `plan_position()` —— 结构止损（参考低点下浮 3~5%，不宽于 -5%）+ 三档目标（+3%/+5%/+15%）+ 盈亏比 + **3-2-2 建仓阵型**；
  - `defend_score()` —— **防卖飞评分（5 分制）**：收盘涨 / 未破 BBI / 非放量阴线 / 趋势向上 / J 未死叉 → 4-5 持有、3 减半、<3 准备离场；
  - `simulate_exit()` —— 按风控优先级执行：硬止损 → 结构止损 → 脱离成本止损 → +3% 减半 → BBI 两日破位 → 波段目标 → 时间止损；
  - `simulate_signals()` —— 任意信号序列 → 完整交易清单（信号次日收盘入场，无未来函数）。
- CLI：`aqlab plan --symbol SYN001`（输出交易计划 + 近 5 根防卖飞评分）。
- 指标层新增 `bbi_line`（(MA3+MA6+MA12+MA24)/4）。
- 测试：12 个新用例（结构止损/硬止损/减半+波段/脱离成本/BBI 两日破位/时间止损两条路径/防卖飞评分降级/信号次日入场/参数校验）。

## v0.4.4 · 滚动窗口校验与真实换手率（✅ 已完成，2026-09）

1. **`src/aqlab/walkforward.py`** —— 把"规则 → 入场 → 离场"整条链放进滚动窗口：
   - `composite_scores`（加权综合分作为信号）、`window_bounds`、`benchmark_return`；
   - `walk_forward` 输出 `windows`（窗口级指标）+ `trades`（逐笔）+ `summary`；
   - **同持有期基准**：每笔交易都带 `bench_return`（该笔持有区间内的买入持有收益），`超额%` 只有这个口径才可比（早期版本拿单笔 1% 对比窗口买入持有 20%，属于苹果比橘子，已修正）；
   - `format_walkforward` / `write_walkforward`，CLI `aqlab walkforward`（含 `--fixed-horizon`、`--no-hard-stop` 两个实验开关）。
2. **真实换手率接入** —— `fetch_tushare_turnover`（`daily_basic.turnover_rate`，% → 小数）+ `TushareDataSource(with_turnover=True)`：有缓存则复用、抓取失败则**降级但继续**（`turnover_available=False` 并写入 `last_error`），报告照实说明换手率是估计值；CLI `aqlab daily --tushare --turnover`（或环境变量 `AQLAB_TURNOVER=1`）。
3. **实测结论（诚实版）**：默认配置 814 笔 / 胜率 57.1% / 单笔均收益 1.17% / 同期间基准 1.44% / 超额 **-0.28pp**，仅 2/15 窗口跑赢；`--no-hard-stop` 把胜率提到 61.7% 但超额反而 -0.36pp。结论：**信号层有价值（见 v0.4.2），离场层在中性行情中是保险而非收益增强**。
4. **测试**：22 个新用例（窗口切分/手算基准/固定持有期/持仓路径/窗口归属/确定性/无信号/参数校验；换手率合并与缓存、抓取失败降级、空结果、默认关闭、`normalize_ohlcv` 保留 turnover 列）；全套 **166** 个用例离线通过。

## v0.5 · 组合层
- 权重优化（风险平价 / 均值方差 + 收缩估计）
- 行业与风格暴露、换手约束
- 组合层面的成本与再平衡口径显式化

## v0.6 · 数据质量
- 缺口/停牌/涨跌停/复权一致性检查
- 多数据源交叉校验与差异报告
- 数据版本快照（parquet + 校验和）

## v0.7 · 报告与可视化
- 净值 / 回撤 / 月度收益热力图 / 因子贡献分解
- HTML 报告（离线自包含）

## 长期（工程化）
- GitHub Actions：pytest + 类型检查 + 报告产物归档
- 参数稳健性：walk-forward、参数敏感性矩阵
- 把"选股 → 回测 → 复盘"固化为每日可跑的任务，并把每次产出归档
