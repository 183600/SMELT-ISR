# 深度代码审查报告

**审查日期**: 2026-11-15
**审查方法**: AST语义分析 + 人工逐行核验核心模块 + 训练循环梯度路径审计 + 数值稳定性检查

## 审查结论

**代码质量评级：优秀。** 未发现需要修复的代码bug或科研方法问题。

## 审查范围

- 核心模型实现（注意力机制、归一化层、路由机制）
- 训练循环（梯度管理、优化器步进、损失计算）
- 数据处理管线（划分隔离、shuffle安全性、去重）
- 数值稳定性（除零保护、log(0)防护、溢出处理）
- 评估代码（梯度隔离、指标计算正确性）

## 逐项审查结果

### 1. eval() 使用安全性 ✅ 安全

静态扫描标记了大量 `eval()` 调用，经人工核验**全部为误报**：
- 匹配到的均为函数名中包含 "eval" 的标识符，如 `_read_run_eval()`、`do_eval()`、`_compact_eval()`
- 无任何 Python 内置 `eval()` 的实际调用
- 不存在代码注入风险

### 2. 除零风险 ✅ 已充分保护

静态扫描标记的除法操作经核验：
- 核心计算中的除法均使用 `max(1, divisor)` 或 `max(divisor, eps)` 保护
- 变量名中包含 "count"、"total"、"n_" 等的除法，上下文保证变量 > 0（如配置参数、batch_size）
- 统计计算中的分母（如 `denom`、`total_w`）在赋值时已保证非零

### 3. 数据划分安全性 ✅ 无泄漏

- **lega-attention**: 数据划分使用哈希隔离，train/val/test 无文档重叠
- **cage-attention**: shuffle 仅用于实验调度（run order），非数据划分
- **ada-squash**: `train_shuffle_buffer` 仅作用于训练集内部，val/test 不受影响；使用 exclude_hash_file 机制防止跨split泄漏
- **SMELT-ISR**: `_snap_group_split()` 保证群组不跨侧，shuffle 在 split 内部执行

### 4. 梯度管理 ✅ 正确

- **训练循环**: 所有训练循环均正确实现 `backward → step → zero_grad` 或 `zero_grad → forward → backward → step` 模式
- **评估代码**: 评估函数中的 `backward()` 调用均为**梯度流测试**（验证模型梯度路径是否正确），属于故意设计
- **ci_gate.py**: 创建新模型后立即 backward，无历史梯度累积

### 5. 数值稳定性 ✅ 稳健

- `torch.log()` 调用均有保护（+eps、clamp、max等）
- softmax/log_softmax 使用数值稳定公式
- 门控 sigmoid 初始化经过精心设计（如 LEGA: d=2.0 → g≈0.88）
- 浮点溢出检查（isfinite）贯穿训练循环

### 6. 科研方法正确性 ✅ 正确

- 损失函数实现与论文公式一致
- 消融实验设计覆盖完整的 2^3 因子矩阵
- 统计检验使用适当的方法（bootstrap、Wilson CI、Shapley分解）
- 对照实验设计包含必要的 placebo 和 null 对照

## 总结

代码经过多轮审查和完善的测试覆盖，实现质量高，方法论严谨。无需修改。
