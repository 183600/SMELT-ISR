# Code Review Report — SMELT-ISR
**日期**: 2026-11-16
**审查范围**: exp/ 目录（52个Python模块）+ tests/ 目录

## 审查摘要

对 SMELT-ISR 仓库进行了全面的代码审查，涵盖 bug 检测、科研方法论验证和代码质量评估。

## 发现的问题

### 已修复的问题

#### 1. `exp/part_001_paper.py` — 除零保护（已修复）
- **位置**: `_paper_budget_audit()` 函数，第45-53行
- **问题**: `flops / base_flops`、`flops / control_flops`、`counts["total"] / (cell.total_params * 1e9)` 等除法操作未对分母为零的情况做防护。虽然正常配置下分母不会为零，但极端或错误的配置（如 `d_model=0`、`n_layers=0`）会导致 `ZeroDivisionError`。
- **修复**: 对所有除法分母添加 `max(..., 1)` 或 `max(..., 1e-12)` 保护。
- **严重程度**: 中等（边界条件防护）

### 审查通过的项目

#### eval() 误报
- 自动扫描标记了20处 `eval()` 调用为高风险，经 AST 分析确认全部为 `model.eval()`（PyTorch 的评估模式切换），无 Python 内置 `eval()` 的安全风险。

#### 除零保护
- `exp/part_025_statistics.py:930` — 已有 `max(intervals, 1)` 保护
- `tests/_core_EdgeCaseRegressionTests_00.py:1665,1722` — 测试代码中的常量除法，安全

#### 科研方法论
- ✅ **数据分割协议**: 使用 seed-owned stream 实现 train/val/test 分割，在数据生成前完成分割，无数据泄漏风险
- ✅ **多重比较校正**: 实现了 Benjamini-Hochberg、Holm 等 FDR 控制方法
- ✅ **评估函数梯度管理**: `part_013_evaluation.py` 使用 `@torch.inference_mode()` + `@torch.no_grad()` 装饰器
- ✅ **随机种子管理**: 集中式种子设置，局部使用 `torch.Generator().manual_seed()`
- ✅ **数值稳定性**: 关键统计量使用 float64 累加，防止 fp32 溢出/抵消

## 代码质量评估

- **整体质量**: 优秀
- **防御性编程**: 广泛使用输入验证、非有限值检测、边界条件保护
- **文档质量**: 详尽的中文注释，包含方法论说明和设计决策解释
- **测试覆盖**: 包含核心测试、边界条件测试、数值精度测试
