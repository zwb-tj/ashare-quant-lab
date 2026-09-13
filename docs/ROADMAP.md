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
- ✅ 评测集已扩展到 **20 个任务**（9 正常 / 5 陷阱 / 3 矛盾前提 / 3 回归），新增 `contradiction_accuracy` 与 `constraint_violations` 两项指标；
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

## v0.5 · 组合层（✅ 已完成，2026-09）

**目标**：从"选股名单"走到"权重、约束、成本都算清楚的组合"。

已交付（`src/aqlab/portfolio.py`）：

1. **权重方法**（纯 numpy：投影梯度 + 二分投影，无 QP 依赖）：`equal` / `inverse_vol` / `risk_parity`（风险贡献相等的乘法迭代）/ `min_variance` / `mean_variance`；协方差按 `(1-δ)Σ + δ·diag(Σ)` 收缩并加 ridge。
2. **约束**：现金缓冲（总仓 ≤1-buffer）、单票上限、换手上限（向目标插值）；**单票上限不可行时保留现金而非放宽上限**（测试抓出的真 bug，已修）。
3. **组合模拟**：份额记账（权重随价格自动漂移）+ 再平衡成本 `Σ|Δw|×费率` + 现金账户；输出净值序列并用既有 `metrics` 计算绩效；空仓→空仓不记为一次再平衡。
4. **暴露报告**：集中度（HHI/有效持仓数/最大权重/持仓数）+ 风格（加权年化波动、加权动量、对等权指数的加权 beta）。**没有行业数据就不编行业暴露。**
5. **体检提示**：平均持仓 < 3 只或平均仓位远低于预算时，报告主动给出警告——把"组合层救不了稀疏信号"这件事说出来。
6. **CLI**：`aqlab portfolio --method ... --max-weight ... --cash-buffer ... --turnover-limit ... --cost-bps ...`，产物 `portfolio.md + equity.csv + rebalances.csv + portfolio.json`。
7. **测试**：19 个新用例（投影/收缩协方差/五种方法/约束/暴露/份额记账复利与成本/换手上限/现金缓冲与上限/无信号空仓/数据不足/确定性/参数校验），全套 **185** 个用例离线通过。

**本轮实测结论**：`zgnb_full` 信号过严 → 148 次再平衡平均仅 1.34 只持仓（扩池到 80 只、上限 10% 后 2.9 只），组合层无法弥补 → **下一步应调整信号层宽松度，而不是继续调权重算法**。

## v0.6 · 信号层迭代与参数扫描（✅ 已完成，2026-09）

**目标**：用数据决定"该放宽哪个参数"，而不是拍脑袋；目标是让组合层的平均持仓达到 **≥3 只**。

1. **参数扫描工具** — `src/aqlab/sweep.py` + CLI `aqlab sweep`：
   - `parse_grid` 解析 `rule.param=v1,v2`（多个 `--set` 取笛卡尔积，值自动转 int/float/str）；
   - 每个格点依次跑 **事件研究 + 滚动窗口 + 组合层**，输出一行：信号数、超额均值/胜率、交易数/胜率/单笔超额、跑赢窗口、平均持仓、累计成本、期末净值、Sharpe、最大回撤；
   - `pick_best`：在"平均持仓 ≥ 目标"的格里取超额最高；**没有任何格达标时明确说"没有配置达标"**并给出持仓最多者。
2. **实测扫描**（`zgnb_full`，j_max ∈ {13,20,30} × spike ∈ {2.0,1.8} × 票池 ∈ {40,80}，合成 900 天）：

| 配置 | 票池 | 信号数 | 超额均值% | 超额胜率% | 交易胜率% | 单笔超额% | 平均持仓 | 达标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| j=13, spike=2.0 | 40 | 1063 | 0.56 | +2.26 | 57.14 | -0.28 | 1.34 | ✗ |
| j=13, spike=1.8 | 80 | 2678 | 0.10 | -2.01 | 68.06 | -0.68 | **3.04** | ✓ |
| j=20, spike=1.8 | 80 | 2826 | 0.07 | -2.27 | 67.94 | -0.67 | **3.25** | ✓ |
| j=30, spike=1.8 | 80 | 3032 | 0.08 | -2.22 | 67.07 | -0.65 | **3.51** | ✓ |

3. **三条结论（诚实版）**：
   - **票池规模是关键杠杆**：40 只票池无论怎么调参数，平均持仓最多 1.71 只；扩到 80 只后才普遍 ≥3；
   - **放宽 j_max 是"数量换质量"**：13→30 让持仓 3.04→3.51，但超额均值 0.10%→0.08%、超额胜率 -2.01%→-2.22%；
   - **票池一换，超额符号就变**：40 只票池上 study 超额胜率是 +2.2%，80 只票池上变成 -2.0%~-2.9%，单笔超额在所有格点都是负的（-0.65~-0.69%）。**这说明这些"超额"高度依赖票池构成，不能当真实结论。**
4. **据此新增档案** `zgnb_full_v2`（j_max=20、spike=1.8；建议配合 80 只以上票池），并保留原 `zgnb_full` 以便对比。
5. **组合层目标**：`PortfolioConfig.min_positions`（默认 3）与 CLI `--min-positions`；低于目标时报告报警。
6. **测试**：11 个新用例（网格解析/覆盖不改原档案/笛卡尔积/确定性与列完整性/选择逻辑两条路径/渲染/落盘/参数校验），全套 **196** 个用例离线通过。

## v0.7 · 数据质量审计（✅ 已完成，2026-09）

- `src/aqlab/quality.py`：`check_frame`（索引单调/重复、日历缺口、零成交占比、异常跳变、历史长度、OHLC 缺失、换手率缺失标为 info）、`audit_universe`（按 error/warning 排序）、`cross_source_diff`（两源差异率 + 最差样本）、`snapshot_hash`（规范化 sha256 前 16 位）、`format_audit` / `write_audit`。
- CLI：`aqlab quality --data-dir DIR [--cross-check DIR2]` → `audit.md / audit.csv / audit.json`。
- 测试：11 个新用例（干净数据只有 info、重复/乱序/历史不足、缺口/零成交/跳变、缺失值、空表、票池排序、两源一致/有差异/无重叠、指纹稳定且敏感、参数校验、渲染与落盘）。

## v0.8 · 开盘量比确认（✅ 已完成，2026-09）

**动机**：B1 只说明"超跌"；若次日开盘没有增量资金，可能继续阴跌 → 买入需要第二道确认。

- `src/aqlab/intraday.py`：`opening_window_volume`、`opening_volume_ratio`（shift(1)，不用当日）、`confirm_signals`（买入/观望/无法判断，缺失即弃答）、`confirmed_signal_series`（映射到决策日）、`generate_synthetic_minutes`（U 形分钟线，保持日成交量）。
- CLI：`aqlab decide --daily-csv ... [--minute-csv ...] --window-minutes 7 --min-ratio 1.0`。
- 同步调整：`aqlab sweep` 去掉"平均持仓 ≥3"的达标门槛；`PortfolioConfig` 移除目标字段，改为中性集中度说明。
- 测试：9 个新用例（窗口量、量比只用历史、三条决策分支、决策日晚于信号日、确认序列映射、合成分钟线守恒与确定性、参数校验），全套 **216** 个用例离线通过。
- 诚实边界：合成分钟线全期统一放量 → 量比自我归一化 → **只能验证流程，不能验证有效性**。

## v0.9 · 可视化（✅ 已完成，2026-09）

- `src/aqlab/charts.py`：`plot_equity_curves`（多策略净值 + 基准）、`plot_drawdown`（水下曲线）、`plot_strategy_comparison`（核心指标条形图）。matplotlib 为**可选依赖**（`pip install -e ".[plot]"`），未安装时给出可执行提示而不是 ImportError 崩塌；标签用英文，保证服务器/CI 无中文字体也能渲染。
- CLI：`aqlab plot [--csv ...] [--strategies ...] [--seed 25]` → `charts/*.png`。
- README / README.en.md 中的净值与对比图**由该命令生成**（`docs/assets/`），不是手绘。
- 测试：5 个用例（两张图的有效 PNG 尺寸校验、缺列报错、未知指标列报错），matplotlib 缺失时整组跳过，CI 不因此变红。
- 已完成（本轮）：**月度收益热力图**（`monthly_return_matrix` / `plot_monthly_heatmap`）与**离线自包含 HTML 报告**（`report_html.py`：base64 内嵌图片、内联样式、零外部引用，测试断言不出现 `http(s)://` 与 `<script>`）。

## v0.10 · 选股日志回测（✅ 已完成，2026-09）

- `src/aqlab/picks.py`：读取 `picks_YYYY-MM-DD.json` 日志 → 同票去重 → 次日开盘入场 → 1/3/5/10 日收益；新增**同期等权篮子基准**（票池外标的、同入场日、同持有期），输出 `excess_*`，并把"发布日 ≠ 可买入日"写进报告表头。
- CLI：`aqlab picks-backtest --archive ... [--confirm ...]`。
- 关键修正：基准与超额只在**配对子集**（个股与篮子都有数据）上计算，保证 `均值 − 基准 = 超额` 自洽。

## v0.11 · 全市场检验 + 离场规则（✅ 已完成，2026-09）

- `src/aqlab/exits.py`：白线/黄线死叉（牵牛绳断）、白线连续破位、滴滴（全版：连续两根阴线 + 破昨低 + 量能不缩 + 不在深跌区）、entry_low 止损、盘中止损止盈（按触发价成交）、ATR 止损、最短持仓保护；优先级为"盘中止损止盈 → 收盘结构规则 → 收盘止损"。
- `src/aqlab/study_universe.py`：全市场扫信号 → T+1 开盘入场 → 离场模拟 → 与全市场等权指数比超额，支持 0AMV 波段门（用"昨天收盘已知的状态"判定）。
- `src/aqlab/intraday.py`：新增 `standard_volume_ratio`（**软件口径**量比，与行情软件一致）与 `opening_features`（开盘上冲/量能斜率/窗口位置）。
- CLI：`aqlab universe-study`；脚本：`scripts/fetch_universe_daily.py`、`scripts/universe_compare.py`、`scripts/opening_filter_study.py`。
- 实测规模：5,424 只标的、58,682 笔交易、2025-01 ~ 2026-09。

## v0.12 · 长样本样本外复核（✅ 已完成，2026-09）

- `scripts/fetch_long_minutes.py`（按月份分层抽样分钟数据）+ `scripts/opening_features_long.py`（21 个月、13,492 笔）。
- 结论：短窗口看到的"开盘量能递增"优势从 +0.95pp 缩到 +0.16pp（超额 +0.08%，t=1.00），"量比越高越差"反转为 +0.21%（t=1.38）→ **开盘形态类结论必须样本外验证**；稳定的是 0AMV 波段与持有期。

## v0.13 · 面向展示的工程面（✅ 已完成，2026-09）

- `tests/test_cli.py`：11 个端到端测试真跑离线 CLI（含中文列名 CSV、全市场研究、量比确认），把语句覆盖率从 **74% 提到 89%**。
- CI：`pytest --cov=aqlab --cov-fail-under=85`，Python 3.10 / 3.11 / 3.12 三版本矩阵。
- 文档：[English README](../README.en.md)、[案例研究](../docs/CASE_STUDY.md)（研究闭环与负面结果）、README 顶部亮点表 + Mermaid 架构图 + 徽章。
- 修正：README 的测试数从 216 更新为实测值；`pyproject.toml` 去掉误加的 UTF-8 BOM（会导致 `pip install -e` 解析失败）。

## 长期（工程化）
- 类型检查（mypy）与 lint（ruff）接入 CI；报告产物归档为 artifact。
- 参数稳健性：walk-forward、参数敏感性矩阵（部分已实现，见 v0.4.4 / v0.6）。
- 把"选股 → 回测 → 复盘"固化为每日可跑的任务，并把每次产出归档。
- 可视化待办：因子贡献分解（月度热力图与自包含 HTML 报告已完成）。

