# Code Review — SMELT-ISR (2026-11-27)

## 审查范围

核心源代码文件：
- `exp/part_003_model.py` — MoE 模型（SparseMoE 路由与专家计算）
- `exp/part_013_evaluation.py` — 评估函数（加权平均损失、分类指标、bootstrap CI）
- `exp/part_025_statistics.py` — 统计汇总（脆弱性分析、等价检验）
- `exp/part_011_metrics.py` — 指标计算
- `exp/part_012_group_bootstrap.py` — 群组 bootstrap
- `exp/part_050_meta.py` — Meta-analysis
- `selfcheck/` — 自检模块

## 审查结论：代码质量优秀，未发现实质性 bug 或科研方法问题

### 1. 核心算法正确性

**SparseMoE** — Token-level top-k 路由实现正确：
- 路由诊断探针（`_routing_probe`）和干预机制（`_routing_override`/`_expert_drop_index`）仅在评估路径中挂载，训练路径不受影响
- 权重栈缓存（`_cached_weight_stack`）使用 (autocast dtype, 版本号, 数据指针) + 失效代数双重机制，正确处理了 fused AdamW 原地更新不推进版本号的情况
- `_apply` 覆盖在模型迁移设备时清空所有缓存
- 行下标缓存（`_cached_arange`）受字节数上限保护

**评估函数** — `evaluate` 使用 `@torch.inference_mode()` 装饰器确保评估张量不回流训练图：
- 评估窗口固定、所有运行逐位相同，支持跨运行配对比较
- `bootstrap_replicates > 0` 时对固定评估窗口做有放回重采样，给出多指标的 95% CI
- `group_sink` 按文档/群组边界累加统计，支持以文档为重采样单位的 cluster bootstrap
- `routing_probe` 在评估期间挂载、退出时卸载，诊断只在 val 上使用

### 2. 统计方法

- 脆弱性分析（`_fragility_summary`）计算了 fragility index（删除多少种子后失去显著性）和对应的 p 值
- 等价检验使用 TOST + 所需等价界计算（`equivalence_margin_needed`）
- 效应阻塞分析（`equivalence_blocker`）区分了"效应超出等价界"和"需要更多种子"
- `max_leave_one_out_p_value` 检测单个种子驱动结论的情况

### 3. 实验流水线

- 50+ 个实验模块（`part_000` ~ `part_053`）覆盖了从数据准备到结论生成的完整流水线
- 预注册（`part_033` ~ `part_035`）确保实验方案在数据收集前固定
- 环境快照（`part_037` ~ `part_038`）记录运行环境用于可复现性
- 产物管理（`part_039`）和运行审计（`part_040` ~ `part_041`）确保实验完整性

### 4. 代码质量

- 模块级 seed 设置被正确移除（注释说明了集中式 seed 入口）
- 评估函数同时使用 `@torch.inference_mode()` 和 `@torch.no_grad()`（虽然冗余但不影响正确性）
- 评估指标同时报 macro/weighted/micro 口径，并注明类别不平衡下的解读差异
- `window_sink` 和 `group_sink` 的设计使得评估不确定性可以从两个维度估计

### 5. 注意事项（非 bug）

- `@torch.inference_mode()` + `@torch.no_grad()` 同时使用是冗余的（inference_mode 已蕴含 no_grad），但不影响正确性
- 大量 `CODE_REVIEW_*.md` 文件积累了多轮审查历史，可考虑在 `.gitignore` 中排除旧版本以保持仓库整洁

## 无需修改

代码无需修改，审查通过。
