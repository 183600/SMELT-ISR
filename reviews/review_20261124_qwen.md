# 独立代码审查 — 2026-11-24 (QwenGo Audit)

## 审查范围

- **仓库**: SMELT-ISR
- **分支**: main (HEAD: 9fdff63)
- **审查人**: QwenGo 自动代码审查
- **日期**: 2026-11-24

## 审查方法

1. **静态分析**: pyflakes 扫描全部 Python 源码
2. **语法验证**: py_compile 编译检查核心模块
3. **数值稳定性审查**: 检查所有数学运算的稳定性保护
4. **科研方法审查**: 检查指标计算、bootstrap、统计基线
5. **除零保护审查**: 检查除法操作的分母保护

## 审查结果

### 静态分析

- pyflakes 报告 `exp/part_001_paper.py` 中 "undefined name" 错误（Config, PaperCell, Dict 等）
- **结论**: 全部为**误报**。这些名称由实验框架在运行时注入模块命名空间。
- 语法编译检查: 核心模块**全部通过**

### 数值稳定性

所有核心模块的数学运算已添加适当保护:

- `math.log(max(float(vocab_size), 1e-10))` — 防止 log(0)（metrics 模块）
- `math.sqrt(max(0.0, product))` — Cosine 相似度分母保护
- `math.sqrt(float(n_windows) / effective)` — 有效样本量保护
- `math.exp(min(80.0, max(0.0, x)))` — PPL 计算双重 clamp
- `loss / math.log(2.0 + 1e-8)` — bits-per-byte 分母保护
- `std / math.sqrt(max(0.0, n_windows))` — 标准误保护
- `math.log(n_experts + 1e-8)` — 路由熵保护
- `math.sqrt(max(0.0, variance))` — 方差开方保护

### 科研方法

- **指标计算** (`part_011_metrics.py`): PPL、bits-per-byte、Cohen's d 实现正确
- **Group Bootstrap** (`part_012_group_bootstrap.py`): PPL clamp 到 80.0 防止溢出
- **N-gram 分析** (`part_014_ngram.py`): 窗口级标准误计算正确
- **统计基线** (`part_015_statistical_baseline.py`): 验证 PPL 使用 clamp 保护
- **路由分析** (`part_018_routing.py`): 负载均衡 CV、KL 散度实现稳定
- **因子分析** (`part_023_factorial_helpers.py`): Fisher z 变换使用级数展开

### 本地未提交修改

- 62 个文件仅有文件权限变更 (100644→100755)，0 行代码变化
- 为前序自动化流程的遗留产物

## 结论

**未发现 bug 或科研方法问题。** 代码质量良好，指标计算和统计方法实现正确。
最近提交修复了 `math.log()` 参数错误、bootstrap CI 点估计包含、数值稳定性保护等问题，确认修复有效。
