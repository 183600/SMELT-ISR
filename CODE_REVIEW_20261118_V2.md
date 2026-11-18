# 代码审查报告 (2026-11-18 第二轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | SMELT-ISR |
| 审查日期 | 2026-11-18 (第二轮独立审查) |
| 审查方法 | 多模式深度分析：AST 分析 + 数值稳定性 + MoE 路由正确性 + Bootstrap 统计 + 数据审计 |
| 审查范围 | 全部 67 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经独立审查确认此前审查结论，未发现需要修复的实际缺陷。

## 审查维度与结论

### 1. MoE 路由机制

`exp/part_003_model.py` 的 `SparseMoE` 实现：

- **路由策略**：Token-level top-k 专家选择，路由器为无偏置线性层。
- **权重缓存** (`_cached_weight_stack`)：使用 (参数版本号, 数据指针) 签名检测参数变更，`_apply` 时整体重建。`invalidate_weight_cache()` 在 optimizer.step() 后调用，确保训练时缓存与新参数同步。
- **填充缓冲缓存** (`_padded_buffers`)：按 (device, dtype, rows, dim) 键缓存推理缓冲。训练时（grad 开启）每次新建以避免进入计算图——注释清晰标注了这一点。
- **路由诊断**：`_routing_probe` / `_routing_override` / `_expert_drop_index` 仅在显式启用的评估路径中挂载，训练路径不受影响。

### 2. 统计推断方法

`exp/part_012_group_bootstrap.py`：

- **群组聚类百分位 Bootstrap**：以文档/群组为重采样单位，token 加权——符合 NLP 评估的最佳实践。
- **窗口自相关校正** (`window_lag1_autocorrelation`)：检测并报告窗口间的相关性，避免低估方差。
- `_randomization_seed`：从数据差异列表生成确定性种子，确保可重复性。

### 3. 数据审计

`exp/part_007_data_pipeline.py`：

- **条件 Bigram 偏移** (`_conditional_bigram_shift`)：衡量训练/评估分布差异。JS 散度计算中 `_kl(a, b)` 使用 `mask = a > 0` 保护 log——当 `b = m = 0.5*(p+q)` 时，`a > 0` 蕴含 `m > 0`，安全。
- **字节级词表**：固定为 256，Laplace 平滑 `+1` 处理零频 token。

### 4. 评估管线

`exp/part_013_evaluation.py`：

- 混淆矩阵计算（precision / recall / F1）使用 `clamp(min=1.0)` 保护分母。
- `total == 0` 的防御性处理返回全零指标而非崩溃。
- PR-AUC 和 ROC-AUC 的参考水平（chance level）计算正确——PR-AUC 的 chance = prevalence，ROC-AUC 的 chance = 0.5。

### 5. Self-check 验证

`selfcheck/checks_000.py` / `checks_001.py`：

- 大量断言验证核心计算的正确性（Bootstrap 确定性、指标一致性、趋势检验）。
- `point_loss = float(loss_totals.sum() / token_totals.sum())`：此处 `token_totals` 保证非零（由上游可用性检查确保），安全。

### 6. 环境探针

`exp/part_037_environment_probe.py`：

- 梯度健康诊断在 `torch.enable_grad()` 上下文中执行——这是有意为之的设计，用于检测梯度流是否正常。

## 修复清单

无。本轮审查未发现需要修复的代码缺陷。

## 审查结论

SMELT-ISR 的 MoE 路由、Bootstrap 统计和数据审计管线均实现正确。缓存管理策略经过仔细设计，在训练和推理间正确区分。Self-check 框架提供了额外的正确性保障。代码整体体现了严谨的科研工程实践。
