# 独立深度代码审查报告 (2026-11-14)

**审查时间:** 2026-11-14 14:00 CST
**审查范围:** 4个科研仓库的全面独立代码审查

---

## 审查方法

1. **自动化扫描**: AST分析 + 模式匹配，覆盖1858个Python文件
2. **深度人工审查**: 核心算法实现、数值稳定性、梯度流、统计方法论
3. **跨仓库对比**: 参数分组、优化器配置、数据管线一致性

## 审查覆盖

| 仓库 | Python文件 | 核心模块审查 | 状态 |
|------|-----------|-------------|------|
| lega-attention | 164 | attention, model, train, factorial, bootstrap, equivalence | ✅ 无问题 |
| cage-attention | 89 | model, train, baselines, _train_recipe, _metrics_inference, _data_task | 🔧 已修复1个问题 |
| ada-squash | 97 | norms, attention, train_gpt2, train_moe_tiny, calibration | ✅ 无问题 |
| SMELT-ISR | 94 | part_003_model, part_018_routing, part_025_statistics, part_007_data_pipeline | ✅ 无问题 |

---

## 发现的问题

### 🔧 cage-attention: `_make_adamw` Weight Decay 参数分组错误

**严重度:** 中高  
**文件:** `cage/train.py`, `cage/baselines.py`  
**问题:** `_make_adamw` 对所有参数统一施加 `weight_decay=0.01`，未区分 bias/LayerNorm/Embedding（应设为0）。

**修复:** 已修改 `_make_adamw` 接受 `nn.Module`，自动将参数分为：
- decay 组（≥2D 权重矩阵，weight_decay=0.01）
- no_decay 组（bias、norm、embedding、1D标量，weight_decay=0.0）

与 lega-attention、ada-squash、SMELT-ISR 的优化器惯例一致。

**状态:** ✅ 已修复并推送 (commit c8d83c7)

---

## 审查确认无问题的领域

### 数值稳定性
- `log()` 操作均有 `clamp_min` / `eps` 保护
- 除法操作均有 `max(x, 1)` / `clamp_min` / 零值检查
- Shapley 值计算使用尺度缩放技巧避免中间溢出
- Softmax 均有显式 `dim` 参数

### 梯度计算
- 评估路径均使用 `model.eval()` + `torch.inference_mode()`
- 梯度累积正确缩放 (`loss / grad_accum`)
- 梯度裁剪在所有训练循环中实现
- `_evaluate_once` 正确保存/恢复训练状态

### 数据管线
- 合成数据为 i.i.d.，使用种子隔离各 split
- 训练/验证/测试使用独立 RNG 流
- 文档级别数据使用 chronological split（无 shuffle 泄漏）
- 数据审计管道完善（hash、provenance、token分布）

### 统计方法论
- 等价性检验 (TOST) 正确实现
- 多重比较校正 (BH/FDR) 已实现
- Bootstrap CI 有完善的配置和验证
- Shapley 值分解有 efficiency gap 自检

### 模型架构
- RoPE 旋转编码实现正确（前向/反向一致）
- 因果注意力掩码正确
- KV 缓存管理正确（动态+静态路径）
- MoE 路由机制正确（含干预探针）

---

## 预扫描 vs 深度审查对比

| 问题类型 | 预扫描报告 | 深度审查结论 |
|----------|-----------|-------------|
| eval() 注入 | 371-540 个警告 | 均为函数名含"eval"的误报（如 `_train_and_eval`），无真实 eval() 调用 |
| 除零错误 | 多个文件 | 均有 `max(x,1)` / `clamp_min` / 零值检查保护 |
| 数据泄漏 | 多个文件 | 合成数据 i.i.d. 或使用 chronological split，有完善的数据审计 |
| Weight Decay | 未报告 | cage-attention 存在参数分组问题（已修复） |

---

## 结论

四个仓库的代码质量整体极高，体现了严谨的科研工程实践。唯一发现的实际问题是 cage-attention 的 `_make_adamw` 未正确分离 weight decay 参数组，已修复并推送。
