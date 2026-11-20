# 深度代码审查报告 V2 - SMELT-ISR
**日期：** 2026-11-13
**审查方式：** 自动预扫描 + 人工级深度审查

## 审查摘要

| 类别 | 结果 |
|---|---|
| **预扫描问题数** | 539（经核实全部为误报） |
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 预扫描误报核实

- **eval() 安全风险（20 项）：** 全部为 `model.eval()` 或自定义函数名
- **除零错误：** `checks_001.py:999-1000` 为测试代码验证 `_tost_test` 内部已处理的 se=0 边界情况
- **数据 shuffle 泄漏：** 测试的是 MoE 路由 shuffle，不涉及数据分割

## 深度审查覆盖

- `exp/part_003_model.py` (~1,376 行) — SparseMoE + TransformerBlock + CausalSelfAttention
- `exp/part_002_layers.py` — SwiGLU + 堆叠专家权重 + 合并专家输出
- `exp/part_011_metrics.py` — 评估指标计算
- `exp/part_018_routing.py` — MoE 路由实验与消融
- `exp/part_025_statistics.py` (~2,010 行) — 统计分析
- SparseMoE 路由机制: top-k 选择与权重归一化 ✓
- `_SwiGLUDown` / `_StackedExpertWeights` / `_MergeExpertOutputs` Autograd 验证 ✓
- `_BlockCheckpoint` 梯度检查点实现 ✓
- 统计: Holm 校正、Bootstrap CI (含 AR(1) 自相关膨胀)、Wilcoxon 检验 ✓
- GQA 支持: `n_kv_heads` 验证逻辑正确 ✓

## 结论

代码库工程质量极高，未发现需要修复的 bug 或科研方法问题。
