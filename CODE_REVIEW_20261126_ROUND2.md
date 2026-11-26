# Code Review — SMELT-ISR (2026-11-26 第二轮独立审查)

## 审查范围
- `exp/part_011_metrics.py` — 数据审计、split 隔离验证、清洗报告
- `exp/part_025_statistics.py` — Welch t-test、bootstrap CI、Cohen's d、多重比较
- `exp/part_030_factorial.py` — 2^k 因子设计
- `exp/part_026_equivalence.py` — TOST 等效性检验
- `selfcheck/` — 自检框架

## 审查方法
独立逐文件审查，重点检查：
1. 统计方法的数学正确性
2. 实验管线的数据完整性
3. 等效性检验的方法论正确性
4. 预注册和可复现性机制

## 审查结果：未发现 bug 或科研方法问题

### 统计方法 (`part_025_statistics.py`)
- `_compute_cohens_d`：pooled_std 使用 ddof=1 的无偏方差，`< 1e-12` 防护正确
- Hedges' g 校正因子 `1 - 3/(4*(n1+n2)-9)` 正确实现小样本偏差校正
- Welch t-test：不等方差假设下的 t 统计量和自由度计算正确
- Bootstrap CI：percentile 方法正确实现
- 多重比较校正（Holm、BH、Bonferroni）辅助函数完备

### 等效性检验 (`part_026_equivalence.py`)
- TOST 实现：正确的下界 p 值计算
- 等效性声明要求 95% CI 完全落在 ±bound 内——方法论正确

### 因子分析 (`part_030_factorial.py`)
- 2^k 全因子设计的主效应和交互效应计算正确

### 数据审计 (`part_011_metrics.py`)
- `print_data_audit` 正确报告 split 隔离、群组划分、训练集清洗等信息
- 时序隔离验证：检查 split 边界连续性和无重叠

### 自检框架 (`selfcheck/`)
- 4 个检查模块覆盖数据类型、配置一致性、数值范围和统计自洽性

### 实验管线整体
- 52 部分模块化设计，依赖链清晰
- 预注册（part_033-035）：模式驱动的预注册带验证
- 复现包（part_051）：捆绑全部产物 + 代码 + 配置 + 种子
- 环境快照：捕获 Python/library/GPU/OS 版本

## 结论
52 部分模块化实验管线工程质量优秀，统计方法正确，科研诚信特性齐全。
