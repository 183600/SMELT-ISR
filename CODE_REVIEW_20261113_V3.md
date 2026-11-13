# 深度代码审查报告 V3 - SMELT-ISR
**日期：** 2026-11-13
**审查方式：** 独立子Agent逐文件审查 + 自动化模式扫描

## 审查摘要

| 类别 | 结果 |
|---|---|
| **本轮审查覆盖** | 核心模块 25+ 文件，约 33,000 行代码 |
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 本轮审查覆盖范围

### 核心算法模块
- `exp/part_002_layers.py` — SwiGLU Expert：`_SwiGLUDown` 自定义 Autograd、`_StackedExpertWeights`、`_MergeExpertOutputs`
- `exp/part_003_model.py` — SparseMoE：路由、调度、top-k 门控、Block Checkpoint、GQA 支持
- `exp/part_011_metrics.py` — 评估指标：group bootstrap CI（含 AR(1) 自相关膨胀）
- `exp/part_012_group_bootstrap.py` — 固定窗口 bootstrap：组完整性保护
- `exp/part_018_routing.py` — MoE 路由实验与消融
- `exp/part_025_statistics.py` — 统计分析：Holm 校正、Wilcoxon 检验

### 重点审查项
1. **SwiGLU 自定义 Autograd**
   - Forward: `hidden = silu(gate) * up`（out-of-place 路径用于需反向传播时，in-place 路径用于推理）✓
   - Backward: `_swiglu_backward` 正确计算 `dL/d(gate) = dL/d(hidden) * up * silu'(gate)` 和 `dL/d(up) = dL/d(hidden) * silu(gate)` ✓
   - `torch.ops.aten.silu_backward.grad_input` 与 `torch.ops.aten.silu.out` 配合使用 ✓

2. **SparseMoE 路由机制**
   - Top-k 选择 + softmax 权重归一化 ✓
   - Switch Transformer 口径（top_k=1 时使用完整 softmax 概率）✓
   - 专家屏蔽干预（causal ablation）✓
   - Load balancing loss 正确计算 ✓

3. **梯度检查点**
   - `_BlockCheckpoint` 正确保存/恢复中间状态 ✓
   - 分块 dispatch 避免 OOM ✓

4. **统计方法**
   - 固定窗口 bootstrap：正确处理组内自相关 ✓
   - Holm 校正、Wilcoxon 符号秩检验 ✓
   - GQA 支持：`n_kv_heads` 验证逻辑正确 ✓

## 结论

经过独立深度审查，未发现需要修复的 bug 或科研方法问题。MoE 路由、SwiGLU 反向传播、统计检验等核心实现均数学正确。
