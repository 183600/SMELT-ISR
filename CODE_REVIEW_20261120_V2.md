# 代码审查报告 — 2026-11-20 独立深度审查

## 审查范围
- 全仓库 Python 源码静态分析 + 人工逐项核实
- 重点审查：数值稳定性（log/division guards）、梯度管理（no_grad/inference_mode）、
  统计方法（multiple comparison correction）、数据泄漏、梯度累积正确性

## 静态分析标记的高优先级项（全部核实为误报）

### lega-attention/curves.py — "P-value without multiple comparison correction"
**结论：误报。** curves.py 不包含任何假设检验或 p 值计算，纯属训练曲线诊断模块
（NaN/Inf 检测、过拟合/欠拟合判定、门控健康诊断）。项目中涉及多重比较的模块
（reparameterization.py、nested.py）已正确使用 Holm-Bonferroni 校正。

### lega-attention/probe.py:341 — "log() on subtraction result"
**结论：误报。** `_logloss` 函数在调用 `np.log()` 之前已使用
`np.clip(p, 1e-12, 1 - 1e-12)` 将输入裁剪到安全范围，
确保 `log(p)` 和 `log(1-p)` 的输入均在 (1e-12, 1-1e-12) 内。

### ada-squash/assumptions.py:144 — "log() on subtraction result"
**结论：误报。** `math.log(1.0 - W)` 执行前已有 `if W >= 1.0: return` 和
`if W <= 0.0: return` 两个早期返回，保证到达该行时 W ∈ (0, 1)，
因此 1.0 - W ∈ (0, 1) 安全。

## 广泛扫描结果

### 数值稳定性
- 所有 `math.log()` 调用均有适当的 guards（max/clip/早期返回）
- 除法运算均已配备零值保护（`if denom:`, `max(x, eps)`, try/except）
- `_regularized_incomplete_beta` 两个实现均在 log 空间中处理溢出路径

### 梯度管理
- 所有评测函数正确使用 `@torch.inference_mode()` 或 `@torch.no_grad()`
- SMELT-ISR 的 `evaluate()` 同时使用了两者（冗余但无害）
- 梯度累积逻辑正确：loss 除以 grad_accum、clip_grad_norm_ 在 step 前调用

### 统计方法
- reparameterization.py 的 paired_t_test 已通过 `_assign_holm` 应用
  Holm-Bonferroni 校正
- nested.py 的 McNemar 检验同样使用 holm_bonferroni 校正
- budget_sensitivity.py 标记为 exploratory，内部已做 within-family 校正

### 数据完整性
- test/val 数据正确隔离（held-out test 机制）
- 训练数据顺序哈希(data_order_hasher)正确维护
- 无数据泄漏风险

## 总结
本轮深度审查未发现新的代码缺陷或科研方法问题。此前多轮审查已修复了真实存在的
bug（如 math.log base-0、ppf sign error、try/except/finally 缩进、
CUDA device probe 日志等），当前代码质量良好。
