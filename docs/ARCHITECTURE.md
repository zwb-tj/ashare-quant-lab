# 架构说明

## 分层

```
         ┌──────────────┐
         │   cli.py     │  demo / run / screen / fetch
         └──────┬───────┘
                │
   ┌────────────┼─────────────┬───────────────┐
   ▼            ▼             ▼               ▼
data.py    strategies.py   screen.py      report.py
   │            │             │               │
   ▼            ▼             ▼               │
indicators.py ──┘             │               │
                │             │               │
                ▼             ▼               │
             backtest.py ◄────┘               │
                │                            │
                ▼                            ▼
             metrics.py ───────────────► 报告产物
```

## 各模块职责

分成三层看：**交易层**（data → indicators → strategies → backtest → metrics）负责"能不能跑、跑得对不对"；
**研究层**（screen / rules_* / study* / walkforward / sweep / intraday / confirm_eval / picks / exits /
factor_ic / factor_strategy）负责"结论站不站得住"；**展示层**（report / charts / report_html / cli / agent）
负责"别人能不能复核"。研究层的每个模块都必须给出样本数、基准与显著性，否则不予采纳。


| 模块 | 职责 | 明确不负责 |
| --- | --- | --- |
| `data.py` | 统一 OHLCV 模式（含中文列名映射）、确定性合成行情、可选实盘数据抓取 | 不做因子、不做交易决策 |
| `indicators.py` | 纯因果指标计算 | 不做仓位、不做成本 |
| `strategies.py` | 把价格历史映射为目标仓位，参数校验 | 不做执行、不知道成本 |
| `backtest.py` | 执行延迟、成本、仓位裁剪、交易流水、组合聚合 | 不做指标解释、不写文件 |
| `metrics.py` | 由回测帧计算标准指标 | 不读数据、不写文件 |
| `screen.py` | 横截面因子快照 + 加权 z-score 排序 | 不做回测、不做择时 |
| `report.py` | 把一次回测固化为可复核产物 | 不计算指标 |
| `rules.py` / `rules_zgnb.py` | 规则插件（含自定义规则集）：返回 [0,1] 分数序列，参数可覆盖 | 不做执行、不知道成本 |
| `profiles.py` | 命名档案：一次切换整套规则与门槛 | 不含规则逻辑 |
| `position.py` | 持仓与离场计划：止损/分批/防守评分的事件流 | 不做择时信号 |
| `intraday.py` | 开盘窗口量比（相对口径与**软件口径**）、开盘特征、次决策映射 | 不做日内执行细节 |
| `confirm_eval.py` | 量比闸门评估：决策日入场、基准配对 | 不选股 |
| `study.py` / `walkforward.py` / `sweep.py` | 规则事件研究、滚动窗口校验、参数扫描 | 不做组合优化 |
| `portfolio.py` | 组合权重（五种方法）、暴露与换手约束、份额记账 | 不产生信号 |
| `quality.py` | 数据质量审计：缺口/零成交/跳变/多源交叉/快照指纹 | 不修数据 |
| `picks.py` | 已发布名单回测：T+1 开盘入场、去重、同期等权篮子超额 | 不重新选股 |
| `exits.py` | 离场规则引擎：白黄线、牵牛绳、滴滴、止损止盈、ATR | 不决定买什么 |
| `study_universe.py` | 全市场研究：选股 + 大盘阶段（0AMV）+ 离场规则同一流水线 | 不做参数搜索 |
| `factor_ic.py` | 因子有效性：逐截面 IC/IC_IR、重叠修正 t 值、分位收益、多持有期期限结构 | 不做策略 |
| `factor_strategy.py` | 定权方案回测：滚动 IC 定权、显式成本、等权基准 | 不做风险模型 |
| `charts.py` / `report_html.py` | 净值/回撤/热力图/因子图与**离线自包含** HTML 报告 | 不做统计判定 |
| `stockdb.py` | 本地行情库接入（SDK 优先、HTTP 降级） | 不缓存业务结论 |
| `tables.py` | 零依赖 markdown 表格渲染 | 不做数值计算 |
| `cli.py` | 参数解析与流程编排 | 不含业务逻辑 |

## 执行时序（关键设计）

```
bar t:    close[t] 已知 ──► 策略计算 signal[t]
                                │
                                │  shift(1)
                                ▼
bar t+1:  position[t+1] = signal[t] ──► 收益 = position[t+1] * bar_ret[t+1] - 成本
```

* `positions` 由策略给出，语义是"看到 bar t 为止的信息后想持有的仓位"；
* 引擎把它整体 `shift(1)`，因此 **bar t 的信号只能在 t+1 生效**；
* 成本按 `|position[t+1] - position[t]|` 计费，换手越高成本越高；
* 结论：策略作者无法通过在 `positions()` 里"顺手看一眼未来"来作弊——

  唯一能作弊的路径（在 t 使用 t+1 的信息）会被 `test_no_lookahead_oracle_loses_money` 捕获：
  该测试故意构造一个使用当日收益方向的信号，断言它在延迟后必然亏损。

## 如何扩展

**加一个指标**：在 `indicators.py` 写纯函数（只依赖 `t` 及之前的数据），补 `tests/test_indicators.py`。

**加一个策略**：

```python
class MyStrategy(Strategy):
    name = "my_strategy"

    def __init__(self, window: int = 30) -> None:
        super().__init__(window=window)

    def validate(self) -> None:
        if self.params["window"] < 2:
            raise ValueError("window must be >= 2")

    def positions(self, df):
        # 只需要返回目标仓位，执行/成本交给引擎
        ...
```

然后注册到 `STRATEGIES`，CLI 的 `--strategy` 会自动可选。

**换数据源**：实现 `fetch_xxx(symbol, start, end) -> DataFrame`，返回 `normalize_ohlcv(...)` 的结果即可；也可以直接把 CSV 丢给 `load_ohlcv_csv`。

## 为什么组合聚合做得这么"笨"

`run_portfolio` 只做等权（或给定权重）的收益平均，不做"每日再平衡到目标权重"的模拟。原因：再平衡假设会隐式引入额外换手与成本口径，容易在报告里制造虚假的平滑净值。等权平均是**最保守、最容易被审计**的聚合方式；组合优化留给 v0.3 并配独立测试。

---

## LLM / Agent 研究层（v0.2）

```
                    ┌────────────────────────────┐
   问题 ──────────► │  ResearchAgent（有界循环）  │
                    │  plan → act → observe ...  │
                    └───────┬────────────┬───────┘
                            │            │
              tool_calls    │            │  trace（每步入参/返回）
                            ▼            ▼
                    ┌──────────────┐  output/trace.json
                    │ ToolRegistry │  可回放、可审计
                    │  只读 6 工具  │
                    └──────┬───────┘
                           │ ok=true / ok=false（错误即数据）
                           ▼
                    ┌──────────────┐      ┌──────────────────┐
                    │ 数据/引擎/筛选 │ ───► │ EvaluationHarness │
                    └──────────────┘      │ 落地率/幻觉/弃答   │
                                          └──────────────────┘
```

### 信任边界（这个项目最想展示的设计）

| 边界 | 规则 | 违反时的后果 |
| --- | --- | --- |
| 工具 → 系统 | 工具**只读**：不写文件、不下单、不改全局状态 | 代理无法造成副作用（可安全地让它自由探索） |
| 工具 → 模型 | 失败以 `{"ok": false, "error": ...}` 返回，绝不抛给模型 | 模型看到的是"证据不足"，而不是异常堆栈，因此可以选择弃答 |
| 模型 → 数字 | 答案里的每个数字都必须在工具返回中出现（含百分比换算与四舍五入容差） | 未落地数字被计为幻觉，扣 `grounded_number_rate` |
| 循环 → 时间 | `max_steps` 硬上限，超限即停并标记 `stopped_reason=max_steps` | 防止模型陷入无限调用 |
| 客户端 → 供应商 | `LLMClient` 协议：任何 OpenAI 兼容端点都能接；`ScriptedClient` 用于离线测试 | 换模型不需要改代理代码，评测尺子不变 |

### 模块职责

| 模块 | 职责 | 明确不负责 |
| --- | --- | --- |
| `tools.py` | 工具定义（JSON Schema）、参数校验、错误捕获、序列化 | 不产生结论、不写文件 |
| `agent.py` | 代理循环、消息编排、trace、两种客户端实现 | 不定义工具、不打分 |
| `evaluation.py` | 任务集、数字落地判定、指标聚合与报告渲染 | 不调用网络（离线模式）、不修改代理 |

### 为什么用标准库实现 HTTP 客户端

`OpenAICompatClient` 只依赖 `urllib`：这样核心包不引入任何 SDK 依赖，任何兼容端点（DeepSeek / Moonshot / vLLM / Ollama 网关）都能直接接；同时让"换模型"成为配置问题而不是代码问题——评测分数因此可以横向比较。
