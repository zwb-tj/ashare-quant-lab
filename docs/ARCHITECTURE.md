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

| 模块 | 职责 | 明确不负责 |
| --- | --- | --- |
| `data.py` | 统一 OHLCV 模式（含中文列名映射）、确定性合成行情、可选实盘数据抓取 | 不做因子、不做交易决策 |
| `indicators.py` | 纯因果指标计算 | 不做仓位、不做成本 |
| `strategies.py` | 把价格历史映射为目标仓位，参数校验 | 不做执行、不知道成本 |
| `backtest.py` | 执行延迟、成本、仓位裁剪、交易流水、组合聚合 | 不做指标解释、不写文件 |
| `metrics.py` | 由回测帧计算标准指标 | 不读数据、不写文件 |
| `screen.py` | 横截面因子快照 + 加权 z-score 排序 | 不做回测、不做择时 |
| `report.py` | 把一次回测固化为可复核产物 | 不计算指标 |
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
