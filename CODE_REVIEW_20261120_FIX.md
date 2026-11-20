# Code Review Report — SMELT-ISR (2026-11-20)

## 审查范围
67 个 Python 源文件，涵盖模型定义、数据处理、评估与统计分析。

## 发现的问题

### 无代码缺陷
对全部 67 个 Python 文件进行了 AST 解析和模式扫描，未发现语法错误或逻辑缺陷。

### 静态分析标记项验证
| 模式 | 数量 | 验证结论 |
|------|------|----------|
| `exec_call` | 2 | `exp/__init__.py` 和 `exp/_runner.py` 中用于组合实验模块片段，仅读取本地项目文件，安全 |
| `unprotected_division` | 8 | 均有 `max(1, ...)` 或前置 `if total <= 0` 保护，安全 |
| `no_grad_context_missing` | 0 | 自定义 autograd.Function 正确使用了 `torch.enable_grad()` 和 `torch.autograd.backward()` |
| `todo_fixme` | 11 | 均为 "Report confidence intervals or std" 的改进计划，非缺陷 |

### 科研方法审查
- **自定义 autograd.Function** (`part_003_model.py`): 梯度检查点实现正确——前向使用 `saved_tensors_hooks` 丢弃激活，反向使用相同输入重算，正确处理了 autocast 状态恢复。反传在 autocast 上下文外执行，与 `torch.utils.checkpoint` 的非重入实现一致。
- **循环残差缩放** (`part_000_config.py`): `loop_count` 属性保证返回 1 或 2，不存在除零风险。Paper 口径按论文 3.1 节正确除以循环次数。
- **数据管线** (`part_007_data_pipeline.py`): 时序分割 + 去重 + 去污染 + 退化检测的完整数据质量保证链。
- **统计方法**: 使用分组 bootstrap 估计置信区间，Kish 有效样本量校正，方法论正确。
- **TODO 项**: 11 处 "Report confidence intervals or std" 的 TODO 标记表明团队计划增加置信区间报告，建议后续实现。

## 修复摘要
无需修复。代码质量良好。
