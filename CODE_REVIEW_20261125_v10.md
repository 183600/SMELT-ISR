# SMELT-ISR 代码审查报告 v10

**审查时间**: 2026-11-25 23:10 CST  
**审查范围**: 全量 Python 代码（exp/ 目录下 50+ 模块，约 70,000 行）  
**审查方法**: AST 语法检查 + 统计函数逐文件审读 + 数据完整性验证

---

## 1. 总体结论

**未发现需要修复的 bug 或科研方法问题。** SMELT-ISR 是一个大规模实验框架，代码组织良好，预注册、效应量、数据完整性等方面都有完善的实现。

## 2. 审查详情

### 2.1 语法与导入
- ✅ 全量 AST 解析无语法错误
- ✅ `_manifest._compute_cohens_d` 可正常导入和运行
- ✅ 模块化架构（part_000 到 part_051）组织清晰

### 2.2 核心统计方法

#### Cohen's d + Hedges' g 实现验证
```python
# 验证结果
g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
g2 = [1.5, 2.5, 3.5, 4.5, 5.5]
d_raw = -0.3162  # Cohen's d
correction = 0.9032  # Hedges' g 校正因子
result = -0.2856  # Hedges' g
```
- ✅ 公式正确：`d = (mean1 - mean2) / pooled_std`
- ✅ Hedges' g 校正：`g = d * (1 - 3/(4*(n1+n2) - 9))`
- ✅ 12 个模块中的 `_compute_cohens_d` 实现一致
- ✅ 极端情况处理：相同组返回 0.0，小样本(n=2)正常工作

### 2.3 实验管道 (exp/part_*.py)
- ✅ part_000_config: 配置验证、INIT_SCHEMES（module_order/name_hash）
- ✅ part_001_paper: 论文协议审计（FLOPs、steps、save_at_steps）
- ✅ part_006_dataset: 数据集加载，bootstrap 按文档级抽样
- ✅ part_009_data_cleaning: 数据清洗，固定窗口 bootstrap
- ✅ part_010_data_gates: 数据门控，检查 bootstrap 与 checkpoint 选型
- ✅ part_011_metrics: 指标计算，评估窗口 lag-1 自相关校正

### 2.4 预注册与假设检验
- ✅ part_033_preregistration_core: 预注册核心逻辑
- ✅ part_034_preregistration_schema: 预注册 schema 定义
- ✅ part_035_preregistration_record: 预注册记录管理
- ✅ 假设注册（hypothesis_error）：每个对比都有明确的假设关联

### 2.5 数据完整性
- ✅ SHA-256 校验：数据、配置、源码树、检查点均有指纹
- ✅ 逐 split 指纹：`saved_data_split_sha256` 可检测续跑时的划分变更
- ✅ 种子重叠检测：`seed_overlap`、`tuning_seed_overlap` 防止数据泄露
- ✅ 检查点追踪：part_042_checkpoint_tracking 记录所有保存/恢复事件

### 2.6 统计推断
- ✅ part_025_statistics: 完整统计检验套件
- ✅ part_026_equivalence: TOST 等价检验（Hedges' g 校正 + 置信区间）
- ✅ part_027_aggregation: 效应量聚合（cohens_dz, hedges_gz）
- ✅ part_012_group_bootstrap: 窗口级 bootstrap 重采样

### 2.7 代码规模与组织
| 模块 | 行数 | 职责 |
|------|------|------|
| part_029_ablation.py | 2,344 | 消融实验 |
| part_025_statistics.py | 2,052 | 统计检验 |
| part_018_routing.py | 2,013 | 实验路由 |
| part_044_pilot_power.py | 1,913 | 先验功效分析 |
| part_035_preregistration_record.py | 1,866 | 预注册记录 |

### 2.8 异常处理
- ✅ `except Exception` 均有注释说明用途
- ✅ 环境探针（part_037）的异常不中断实验
- ✅ 诊断模块（part_019）的异常被正确捕获并报告

## 3. 运行验证

| 测试 | 结果 |
|------|------|
| `_compute_cohens_d([1,2,3,4,5], [1.5,2.5,3.5,4.5,5.5])` | -0.2856 ✅ |
| 相同组 `_compute_cohens_d(g, g)` | 0.0 ✅ |
| 小样本 `_compute_cohens_d([1,2], [3,4])` | -1.6162 ✅ |

## 4. 代码质量评级

| 维度 | 评级 |
|------|------|
| 语法正确性 | A |
| 数值稳定性 | A |
| 统计方法 | A+ |
| 预注册支持 | A+ |
| 数据完整性 | A+ |
| 代码组织 | A |
| 可重复性 | A+ |
