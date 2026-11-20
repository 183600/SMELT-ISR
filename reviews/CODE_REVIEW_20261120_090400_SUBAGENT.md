# 代码审查报告 - SMELT-ISR (子Agent独立审查)
**日期:** 2026-11-20 09:04
**审查范围:** 全部 95 个 Python 源文件
**审查方法:** 静态分析 + 语法验证 + 算法实现审查 + 科研方法论审查

## 审查结果摘要

| 类别 | 发现数 | 已修复 |
|------|--------|--------|
| 语法错误 | 0 | N/A |
| 静态分析标记（误报/改进建议） | 166 | N/A |
| 科研方法问题 | 0 | N/A |

## 审查结论

代码质量良好，未发现需要修复的 bug 或科研方法问题。166 个静态分析标记中约 100+ 为 TODO 改进建议（添加置信区间），其余为正确保护的除法和安全模式。

## 静态分析标记项复核

### TODO/FIXME (100+项) — 改进建议，非 Bug

大量 `# TODO: Report confidence intervals or std alongside point estimates` 注释分布在：
- `part_011_metrics.py` — 指标计算
- `part_012_group_bootstrap.py` — AUC 计算

这些是已记录的改进建议。当前点估计在数学上是正确的；添加置信区间会增强报告完整性，但不是方法论要求。

### 除法保护 (10项) — 误报

- `part_000_config.py:1002,1036` — `max(1, int(hidden) // n_heads)` 保证 `head_dim >= 1`
- `part_007_data_pipeline.py:728` — `total <= 0` 检查在前
- `part_007_data_pipeline.py:896,901` — `if total_eval_pairs > 0 else 0.0`
- `part_011_metrics.py:527` — `present = counts > 0.0` 保证除数正数
- `part_012_group_bootstrap.py` — 多处有 `eligible.numel() > 0` 保证

### autocast checkpoint (1项) — 正确行为

- `part_003_model.py:1184` — 注释说明前向在 autocast 中执行、反向重算在 autocast 外，这是 `torch.utils.checkpoint` 非重入实现的正确行为

### exec_call (2项) — 有意设计

- `exp/__init__.py:25` 和 `exp/_runner.py:77` — 本地项目文件组合，`# nosec B102`

## 科研方法论审查

### 评估协议
- `@torch.inference_mode()` 和 `@torch.no_grad()` 正确应用于评估函数
- 验证/测试集分割严格执行；测试集仅在最终评估时调用一次
- 固定窗口评估确保一致的指标计算

### 指标计算 (part_011_metrics.py)
- 混淆矩阵指标 (micro/macro/weighted F1, precision, recall) 正确
- `precision = true_positive / predicted.clamp(min=1.0)` 正确处理零除
- F1 调和平均仅在 `denominator > 0.0` 时计算

### Group Bootstrap AUC (part_012_group_bootstrap.py)
- ROC-AUC 通过秩和方法正确计算，含并列分数处理
- PR-AUC 使用降序分数边界检测正确处理并列分数
- AP = Σ(R_i − R_{i−1}) × Precision_i 是标准梯形法则
- 分块计算 `_AUC_SORT_CHUNK_ELEMENTS // n_tokens` 限制内存使用不影响正确性

### 统计基线 (part_015_statistical_baseline.py)
- N-gram 模型带网格搜索阶数 × 平滑
- 与神经模型使用相同的评估协议（确保公平比较）

### 训练曲线分析 (part_016_curve_analysis.py)
- 完整的诊断分类：过拟合、欠拟合、平台期、未收敛、发散、健康
- 基于中位数的梯度尖峰检测

## 验证

全量语法验证: 95 文件 / 0 错误 ✅
