# 代码审查报告 (2026-11-18 第三轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | SMELT-ISR |
| 审查日期 | 2026-11-18 (第三轮独立验证) |
| 审查方法 | 子 Agent 独立深度审查：MoE 路由正确性 + 数据泄漏防护 + Bootstrap 统计 + 评估管线 |
| 审查范围 | 全部 67 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经第三轮独立验证，确认前两轮审查结论，未发现新的缺陷。

## 本轮审查重点

### 1. 数据泄漏防护验证

`exp/part_006_dataset.py` 的 split 隔离协议：

- **群组感知切分**：`_snap_group_split` 将切分位置吸附到群组边界，同一群组的全部字节整体落在同一个 split
- **SHA-256 完整性验证**：每个 split 的内容指纹独立计算，确保切分一致性
- **时序隔离检查**：验证 train/val/test 三段不重叠、覆盖全部字节
- **清洗隔离**：训练集清洗区间不越出 train split，对照评估 split 的清洗有独立记录

### 2. MoE 路由机制复验

`exp/part_003_model.py` 的 `SparseMoE`：

- **权重缓存同步**：`(参数版本号, 数据指针)` 签名检测 + `invalidate_weight_cache()` 在 `optimizer.step()` 后调用
- **训练/推理区分**：训练时（grad 开启）每次新建填充缓冲以避免进入计算图
- **路由诊断隔离**：`_routing_probe` / `_routing_override` 仅在显式启用的评估路径中挂载

### 3. Bootstrap 统计方法

`exp/part_012_group_bootstrap.py`：

- **群组聚类配对比较**：以文档/群组为重采样单位，token 加权汇总
- **群组一致性验证**：检查 `group_separator_sha256` 和 `group_assignment_version` 匹配
- **窗口自相关校正**：`window_lag1_autocorrelation` 避免低估方差
- **确定性种子**：从数据差异列表生成，确保可重复性

### 4. 评估管线

`exp/part_013_evaluation.py`：

- 混淆矩阵使用 `clamp(min=1.0)` 保护分母
- `total == 0` 返回全零指标而非崩溃
- PR-AUC chance = prevalence，ROC-AUC chance = 0.5 — 参考水平计算正确

### 5. Self-check 框架

`selfcheck/checks_000.py` / `checks_001.py`：

- 大量断言验证核心计算正确性（Bootstrap 确定性、指标一致性、趋势检验）
- 提供了额外的正确性保障层

## 修复清单

无新增修复。

## 审查结论

SMELT-ISR 的数据隔离协议、MoE 路由管理、Bootstrap 统计方法和评估管线经过三轮独立审查验证，实现正确可靠。代码体现了严谨的科研工程实践，特别是在数据泄漏防护和统计推断方面。
