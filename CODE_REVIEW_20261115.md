# 代码审查报告 - SMELT-ISR
**审查日期**: 2026-11-15  
**审查范围**: 核心代码（exp/）和模型实现（exp/part_003_model.py）  
**审查深度**: 人工逐行审查核心模块 + AST 静态分析  

## 审查结论

**代码质量评级：优秀**。未发现需要立即修复的关键代码 bug 或科研方法问题。

## 详细审查结果

### 1. MoE 路由机制 (exp/part_003_model.py)

**Top-k 路由** ✅ 正确
- `route()` 正确实现 top-k 专家选择
- 负载均衡损失 `aux_loss = n_experts * dot(hard_load, soft_load)` 正确实现 Switch Transformer 公式
- 排序-散射-聚合路径正确处理溢出情况（容量外的 assignment 回退到逐专家计算）

**延迟合并 (Scheme A/B)** ✅ 正确
- `first_branch_step()` 正确创建 k1 个分支
- `local_two_stage()` 正确实现局部延迟合并
- 分支独立执行时 attention mask 正确复制 k1 份（branch_attn_bias）
- k1=1 时的门控预乘正确处理（避免双重缩放）

**B_early 对照** ✅ 正确
- 提前合并后复制回 k1 分支，正确隔离"合并位置"因素
- 门控加权合并正确实现

### 2. 模型前向 (exp/part_004_model_data.py)

**模式分发** ✅ 正确
- dense/base/smelt/A/B/B_early/local_base/local_base_k2 所有模式正确实现
- 残差缩放正确区分循环区域内外（global_scale vs loop_scale）
- aux_loss 正确取均值（`torch.stack(aux_losses).mean()`）

**总损失** ✅ 正确
- `total_loss = lm_loss + balance_coef * aux_loss` 正确实现
- cross_entropy 正确 reshape logits 和 targets

### 3. 文档级 Segment Attention Mask

**掩码构建** ✅ 正确
- `SegmentAttentionMask` 正确实现文档级因果注意力
- B/B_early 模式正确将掩码按 k1 复制到分支维度
- 掩码正确防止跨文档注意力

### 4. 数据处理

**数据泄漏防护** ✅ 完善
- `_snap_group_split()` 正确保证群组不跨侧
- `_split_isolation_record()` 正确记录 split 边界
- 近重复审计正确检测跨 split 重复

**Shuffle 警告澄清** ✅
- 数据 pipeline 的 shuffle 均在 split 内部进行
- split 边界由群组对齐保证，不受 shuffle 影响

### 5. 自测模块 (selfcheck/)

**结构验证** ✅ 全面
- 梯度流通测试验证各模式的梯度路径
- 因果性测试验证自回归约束
- k=1 等价性测试验证 B 与 SMELT 的边界情况
- 可复现性测试验证确定性

### 6. 静态分析结果

- **eval() 调用**: 全部为 `model.eval()`，无安全风险
- **数据 shuffle**: 均在 split 后执行，无泄漏风险
- **除零风险**: `part_008_data_audit.py` 和 `part_030_factorial.py` 中的除法已有保护

## 审查中发现的非关键观察

1. 溢出回退路径（capacity overflow fallback）设计精巧，正确处理桶大小超过静态容量的情况
2. 零填充缓冲的梯度问题已正确处理（scratch 行 zero_() 避免双重梯度）

## 结论

代码实现与论文方法一致，MoE 路由、延迟合并、对照实验设计均正确实现，无需修改。
