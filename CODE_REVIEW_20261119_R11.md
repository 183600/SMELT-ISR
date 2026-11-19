# Code Review Report — SMELT-ISR (Round 11)

**日期**: 2026-11-19
**审查方法**: 深度逐行审查 + 模式扫描 + 跨仓库对比分析
**审查范围**: 全部 67 个 Python 源文件

## 审查结论

**未发现新的 bug 或科研方法问题。** 代码质量评级：优秀。

## 审查覆盖

### 1. 模型架构 (exp/part_003_model.py)

- **SparseMoE**: Token-level top-k 路由正确实现
- 专家权重缓存和失效机制: 参数化缓存管理正确
- 路由探针和干预: 仅在显式启用时挂载，训练路径不受影响
- 零填充缓冲: 推理路径使用缓存，训练路径每次新建

### 2. 数据管线 (exp/part_007_data_pipeline.py)

- 字节级分布审计: 有 `total <= 0` 保护
- 时间边界漂移审计: 正确使用 JS 距离
- 训练/评估隔离: 基于种子的流分割

### 3. 评估指标 (exp/part_011_metrics.py)

- 混淆矩阵指标: micro/macro/weighted F1 正确计算
- 支持类过滤: `counts[present]` 布尔索引确保无零除
- 评估不确定性: 窗口级 bootstrap 实现正确

### 4. 实验框架 (exp/__init__.py, exp/_runner.py)

- `exec()` 调用: 本地项目文件组合，有 `# nosec B102` 注释和异常处理
- 隔离命名空间: `_IsolatedNamespace` 正确管理
- 编译和执行分离: 有详细的错误信息

### 5. 静态分析标志复核

| 标志模式 | 数量 | 复核结果 |
|---|---|---|
| `exec_call` | 2 | 本地模块组合，有安全注释 |
| `unprotected_division` | 8 | 全部有 `max()` 或条件保护 |
| `no_grad_context_missing` | 1 | 文档注释中的说明文本 |
| `todo_fixme` | 1 | 增量改进备注 |

### 6. 跨仓库对比

在 liga-attention、cage-attention、ada-squash 中发现的 `model.eval()` + `model.train()` 排序 bug，在 SMELT-ISR 中不存在。SMELT-ISR 的评估函数正确使用了 `@torch.inference_mode()` 装饰器或显式的 `model.eval()` 而无需切回。

## 结论

SMELT-ISR 代码库质量优秀，未发现需要修复的 bug 或科研方法问题。防御性编程实践全面，数值稳定性保护完备。
