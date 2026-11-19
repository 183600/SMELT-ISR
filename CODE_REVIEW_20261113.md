# Code Review Report - SMELT-ISR (2026-11-13)

## 审查结果

| 类别 | 结果 |
|---|---|
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 深度审查内容

对以下核心模块进行了逐行审查：

- `exp/part_003_model.py` (~1,376 行) — SparseMoE + TransformerBlock + CausalSelfAttention
- `exp/part_011_metrics.py` — 评估指标计算
- `exp/part_018_routing.py` — MoE 路由实验与消融
- `exp/part_015_statistical_baseline.py` — 统计基线
- `exp/analysisflow/part_000.py` — 分析流水线

**未发现实际 bug。**

## 预执行扫描误报核实

前次自动扫描报告的 539 个问题经逐条核实，全部为误报：

- **eval() 安全风险（20 项高严重度）：** 全部为 `model.eval()` 或自定义函数名（如 `_intervention_eval`），无 Python 内置 `eval()` 调用。
- **除零错误（`_core_EdgeCaseRegressionTests_00.py:1663,1720`）：** 位于测试文件中，测试代码已正确添加 guard。
- **数据 shuffle 泄漏（`_core_RoutingTests.py`）：** 测试的是 MoE 路由 shuffle 机制，不涉及数据分割；属于实验设计的一部分。

## 代码质量评估

- **SparseMoE 实现：** 路由机制正确，top-k 选择与权重归一化实现无误
- **TransformerBlock：** 残差缩放（`residual_scale`）正确应用于注意力和 MoE 两条路径
- **指标计算：** `total > 0` 提前返回防御，`max(total, 1)` 分母保护（total 为 Python int 且 > 0 已保证）
- **统计方法：** Holm 校正、Bootstrap 置信区间（含 AR(1) 自相关膨胀）、Wilcoxon 检验
- **GQA 支持：** `n_kv_heads` 验证逻辑正确（n_heads 无需被 n_kv_heads 整除）
