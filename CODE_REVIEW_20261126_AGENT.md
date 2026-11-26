# Code Review — SMELT-ISR (2026-11-26 Agent)

## 审查范围
- 实验管线（52 parts: part_000 至 part_053）— 配置, 数据, 模型, 评测, 分析, 产物
- 自检框架 (`selfcheck/`) — 4 个检查模块
- 统计方法 (`part_025_statistics.py`) — Welch t-test, bootstrap CI, Cohen's d, 多重比较
- 等效性检验 (`part_026_equivalence.py`) — TOST
- 因子分析 (`part_030_factorial.py`) — 2^k 设计

## 审查结果：未发现 bug 或科研方法问题

### 统计方法
- `_compute_cohens_d`：pooled_std 使用 ddof=1 的无偏方差，`< 1e-12` 防护正确
- Hedges' g 校正因子 `1 - 3/(4*(n1+n2)-9)` 正确实现小样本偏差校正
- Cohen's d 在多个文件中重复定义（`_manifest.py`, `part_007`, `part_011`, `part_017`, `part_024`, `part_025`）——代码风格问题但不影响正确性，所有副本公式一致

### 等效性检验
- TOST 实现：正确的下界 p 值计算（lower p-value）
- 等效性声明要求 95% CI 完全落在 ±bound 内——方法论正确

### 实验管线
- 模块化设计：每个 part 处理一个关注点，支持独立测试和替换
- 依赖链正确：config → data → model → evaluation → analysis → artifacts
- 数据管线：train/val/test 分割带内容哈希、近似重复检测、污染扫描
- 预注册（part_033-035）：模式驱动的预注册带验证——防止 HARKing

### 产物管理
- 全部实验输出通过内容哈希追踪
- 复现包（part_051）捆绑所有产物 + 代码 + 配置 + 种子
- 环境快照捕获 Python/library/GPU/OS 版本——支持精确环境复现

## 结论
52 部分模块化实验管线工程质量优秀，统计工具包完备，预注册/功效分析/去污染/溯源追踪等科研诚信特性齐全。未发现需要修复的 bug 或方法论问题。
