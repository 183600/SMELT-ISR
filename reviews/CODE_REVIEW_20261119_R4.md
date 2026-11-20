# Code Review Report — SMELT-ISR (Round 4)

**日期：** 2026-11-19
**审查范围：** 全部 67 个 Python 源文件
**审查方法：** 静态分析 + 人工逐文件深度审查

---

## 结论

**未发现 bug 或科研方法问题。**

静态分析工具共标记 113 处可疑模式，经逐一人工审查，全部为误报。

---

## 静态分析标记分类与审查结论

### 1. `exec_call` 模式（2 处）

**审查结论：安全。**

- `exp/__init__.py:25`：有意为之的实验代码组合模式，仅从本地项目文件加载，有 `# nosec B102` 安全注释。
- `exp/_runner.py:77`：同上，实验运行器加载实验部件文件。

两处 `exec()` 均只执行来自本地项目目录的编译后代码，不接受用户输入，是实验框架的常见设计模式。

### 2. `unprotected_division` 模式（多处）

**审查结论：全部安全。** 代表性审查：

| 文件 | 行号 | 防护措施 |
|------|------|----------|
| `part_000_config.py:1002,1036` | `hidden // n_heads` | 遍历 `_divisors(hidden)` 保证整除，`max(1, ...)` 兜底 |
| `part_001_paper.py:49` | `control_counts["total"] / counts["total"]` | 由参数计数语义保证 > 0 |
| `part_007_data_pipeline.py:728` | `counts / total` | `total <= 0` 前置返回 None（第 726 行） |
| `part_007_data_pipeline.py:896,901` | `seen_pairs / total_eval_pairs` | `total_eval_pairs <= 0` 前置返回（第 887 行） |
| `part_011_metrics.py:527` | `sums[present] / counts[present]` | `present` 布尔掩码过滤零计数 |

### 3. `no_grad_context_missing` 模式（1 处）

- `part_003_model.py:1184`：**审查结论：安全。** 这是文档注释，解释 autocast 状态在前向和反向之间的不一致问题，而非实际的 `loss.backward()` 调用。

---

## 科研方法审查

### 模型架构
- Head 几何搜索算法正确：遍历 hidden 的因子，确保整除性
- GQA（Grouped Query Attention）的 KV 头映射灵活，不强制整除
- `_paper_smelt_head_geometry` 以论文报告的 KV 比例为第一目标，方法论正确

### 数据管线
- 时间边界分布漂移审计（`_temporal_boundary_shift_audit`）使用 JS 散度检测切分边界偏倚
- 上下文/目标共现矩阵的去污染分析考虑了时间泄漏
- 数据清理（`part_009_data_cleaning.py`）和数据门控（`part_010_data_gates.py`）分层设计

### 评估方法
- 宏平均 NLL（`part_011_metrics.py`）正确处理零计数类别
- Bootstrap 不确定性量化使用窗口级重采样（非 token 级），考虑了序列内相关性
- 曲线分析（`part_016_curve_analysis.py`）和窗口比较（`part_024_window_comparison.py`）方法论严谨

### 统计基线
- 统计基线（`part_015_statistical_baseline.py`）提供合理的性能下界
- 因子实验辅助（`part_023_factorial_helpers.py`）支持消融分析
- 比较协议（`part_022_comparison_protocol.py`）确保公平对比

### 实验编排
- 实验部件（part_000 ~ part_024+）按职责清晰拆分
- 清单系统（`_manifest.py`）确保部件加载顺序正确
- 运行器（`_runner.py`）支持隔离命名空间执行

---

## 总结

SMELT-ISR 是一个结构严谨的大规模科研实验框架。代码质量高，防御性编程一致，科研方法论正确。所有静态分析标记均为正确代码的误报。无需修改。
