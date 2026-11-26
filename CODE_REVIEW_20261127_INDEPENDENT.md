# 独立代码审查报告 (2026-11-27)

## 审查范围
- `exp/part_003_model.py` - MoE 模型架构、SparseMoE 路由与权重缓存
- `exp/part_011_metrics.py` - 类别不平衡感知的分类指标、混淆矩阵计算
- `exp/part_013_evaluation.py` - 评估管线、bootstrap 不确定性估计、cluster bootstrap

## 审查结论

**未发现需要修复的 bug 或科研方法问题。**

### 代码质量评估

1. **数值稳定性**：代码在多个层面做了充分的数值保护
   - `confusion_matrix_metrics` 在 `total <= 0` 时安全返回 NaN/0
   - `_valid_target_mask` 正确处理 uint8 与负 ignore_index 的比较
   - `_lag1_autocorrelation` 对大数值输入做归一化处理，避免溢出

2. **统计方法论**：评估不确定性估计严谨
   - 窗口级 bootstrap 以窗口为 resampling unit（而非逐 token）
   - cluster bootstrap 以文档/群组为单位，正确处理窗口相关性
   - AR(1) 有效窗口数校正处理自相关，只放宽不收窄
   - 报告多种 FDR 校正方法（BH、BY、Holm）作为 companion

3. **数据完整性**：防泄漏措施完善
   - 评估集只在最终评估时解封
   - 群组级 split 互斥性审计
   - 近重复窗口审计（含 shingle Jaccard）

4. **缓存管理**：MoE 权重缓存设计合理
   - `_cached_weight_stack` 按 (dtype, version, data_ptr) 签名
   - `_apply()` 正确清空所有缓存
   - `invalidate_weight_cache()` 在 optimizer.step() 后递增失效代数

### 已审查的关键逻辑点

- [x] SparseMoE forward：路由逻辑、aux_loss 计算、专家权重堆栈缓存
- [x] 混淆矩阵构建：`torch.bincount` 加法公式正确 (target*V + predicted)
- [x] macro/weighted F1：对 support > 0 的类求平均，加权按 support
- [x] 类平衡 NLL (macro NLL)：逐类等权平均，与频率加权互补
- [x] bootstrap CI：分位区间、lag-1 自相关校正、群组级重采样
- [x] AUC 收集：均匀抽样位置、softmax chunk 处理、support 阈值过滤
