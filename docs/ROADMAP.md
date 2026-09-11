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

## v0.3 · 组合层
- 权重优化（风险平价 / 均值方差 + 收缩估计）
- 行业与风格暴露、换手约束
- 组合层面的成本与再平衡口径显式化

## v0.4 · 数据质量
- 缺口/停牌/涨跌停/复权一致性检查
- 多数据源交叉校验与差异报告
- 数据版本快照（parquet + 校验和）

## v0.5 · 报告与可视化
- 净值 / 回撤 / 月度收益热力图 / 因子贡献分解
- HTML 报告（离线自包含）

## 长期（工程化）
- GitHub Actions：pytest + 类型检查 + 报告产物归档
- 参数稳健性：walk-forward、参数敏感性矩阵
- 把"选股 → 回测 → 复盘"固化为每日可跑的任务，并把每次产出归档
