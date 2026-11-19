# 对比实验与消融的科研方法改进指南

本指南在现有 SMELT-ISR 基座上，围绕**受控变量对比**、**全因子消融**、**统计可靠性**与**计算量口径**四方面，明确可操作的改进点与默认检查清单，是 README 使用说明的方法学补充；README、`.github` CI 配置与各文档均保持原有位置不变。

## 1. 受控变量对比的硬门

### 1.1 必声明因子与禁止混淆因子
- 默认可自由变化的实验因子：`mode, k1, k2, balance_coef, seed, dense_fnn_hidden`。
- 其余任何协议项必须逐项一致：优化器（AdamW betas/eps）、学习率策略（schedule/warmup）、训练步数、warmup、weight decay、梯度裁剪、batch size、数据增强、数据指纹与逐 split 内容指纹、模型结构、评估窗口、模型选型口径、AMP dtype、设备类型、梯度检查点、torch.compile 等。
- `--compare` 会自动核对 `comparison_protocol`；未在 `--compare-factors` 声明的差异会以 `[violation]` 标记并阻断主指数声称。

### 1.2 基线合理性守卫
对比前必须通过稻草人守卫：
- 神经基线主指标 > 无上下文 unigram 参考；
- 不劣于调优好的字节级 n-gram 统计基线（`--suite-baseline-tuning`，val 上网格调优，记录 `baseline_tuning` 来源）；
- 种子数 ≥3；
- 假设预注册：`--hypothesis`、`--rq`、`--primary-metric`、`--hypothesis-direction`、`--hypothesis-min-effect`、`--hypothesis-baseline`。
未通过的对比会以 `[straw-man]` 标记，前进需先调强基线。

## 2. 全因子消融完整性

### 2.1 消融设计
延迟合并 × 均衡损失构成 (1+N)×2 全因子：
- 对照：`local_base`（立即合并，等专家计算量）
- 机制对照：`local_base_k2`（只改变合并时机，第二轮宽度仍为 k2）
- 方法臂：A（局部延迟合并）、B（全局延迟合并）
- B 专用对照：`B_early`（分支数/Attention/MoE 执行次数与 B 完全相同，只改变合并位置）
- 均衡损失：`balance_coef=0` 关闭，开启臂取 `--balance-coef`（默认 0.01）。

默认覆盖 A 与 B，缺失任一方案会被 `ablation_coverage` 列为 `missing_cells`。机制对照臂（`local_base_k2` / `B_early`）不是默认单元，需显式加 `--ablation-mechanism-controls`（见 2.3）；缺它们时组件判定会把延迟合并的归因判为 `blocked`。

### 2.2 结构混杂检查
消融单元内必须保持：
- `k1/k2/top_k/residual_scale` 恒定，否则主效应混入分支几何/路由宽度/残差缩放；
- 开启臂 `balance_coef` 必须唯一，多个正值会被视为 `confounded_designs`；
- `k1*k2 > n_experts` 时第二轮宽度会被截断，`local_base` 与 A 的等计算量假设不再严格，需在报告中显式标记。

`--suite-ablation --ablation-modes A B --seeds 42 43 44 45 46 47` 生成六格完整设计。

### 2.3 单因素机制对照（延迟合并归因的硬门）
(1+N)×2 消融里的 `A`/`B` vs `local_base` 是**多因素**对比：合并时机/位置之外，第二轮路由宽度、专家复用方式与分支数也同时变化。主效应显著只能说明「这套配置整体更好」，不能说明「延迟合并机制本身有用」。要把收益归因到延迟合并，必须有只改一个因子的对照臂：

| 方案 | 机制对照 | 唯一变化的因子 |
|---|---|---|
| A（局部延迟合并） | `local_base_k2` | 合并时机（第二轮后合并 → 第一轮后立即合并） |
| B（全局延迟合并） | `B_early` | 合并位置（第二轮后合并 → 第一轮后提前合并再复制回 k1 分支） |

报告口径：
- `mechanism_control_report` 逐方案给出状态：`supported`（校正后显著且延迟合并更好）、`contradicted`（显著更差）、`partial`（方向不一致）、`not_supported`（无支持证据）、`missing_control`（运行集里没有对照臂）、`no_control_defined`（该方案没登记对照）、`confounded`（对照臂存在，但两臂除被检验因子外还有差异）、`unverifiable`（一侧缺协议记录，无法核对两臂是否只差一个因子）、`insufficient_data`（对照臂没有可用运行）、`insufficient_seeds`（共同种子不足）。
- 单因素核对是逐「可比单元对」做的：先按分组键对齐方案臂与对照臂，再逐项核对两臂的 `comparison_protocol`（除 `mode` 外的全部协议项，含 `k1/k2/balance_coef/lr/训练步数` 等）。因为 `--compare-factors` 声明的因子默认不受受控变量检查约束，`k1/k2/balance_coef` 之间的差异只能在这里被拦住：协议项实际不同 → 该对比判 `confounded` 并在 `confounded_pairs` 里点名差异键；一侧缺协议记录 → 判 `unverifiable`；两臂该因子的水平没有交集（例如 A 的 `k2=4`、对照 `k2=2`）→ 判 `confounded` 并在 `unmatched_levels` 里点名因子。
- 报告顶层给出 `single_factor_verified`（所有方案的单因素核对是否都通过）与 `single_factor_check` 汇总；只要有方案未通过，组件判定不会宣称延迟合并的收益可归因到机制本身。
- 机制对照自成一个 Holm/BH 家族（不与全体两两对比混family），但每个对比仍要求「除被检验因子外其余受控变量逐项一致」。
- `ablation_coverage` 单独记账机制对照臂：`mechanism_controls` 给出每个方案的对照臂、两档的种子对齐（`n_common_seeds`）与缺档 `missing_control_levels`；`mechanism_missing_cells` 列出缺失的对照单元（如 `local_base_k2/b=on`），并进入 `blockers`——只跑了一半的对照臂不能当作完整的两档证据。
- 组件级判定（`ablation_component_verdicts`）会把延迟合并方案的门控接到该状态上：缺对照、对照被混杂/无法核对/数据或种子不足 → 该组件判为 `blocked` 并进入 `blockers`；对照反向 → 判为 `contradicted`；对照支持 → 组件结论不再受此阻断。**没有单因素对照时，不得宣称「延迟合并方案 A/B 各自有用」。**
- 缺对照不是「机制无效」，只是归因未就绪：先补跑对照臂，不要用多因素主效应代替。

运行方式（对照臂与消融同 seed、同 balance 两档，保证对照是单因素）：
```bash
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 45 46 47 \
  --suite-ablation --ablation-modes A B --ablation-mechanism-controls \
  --common-step 2000 --budget-match smelt \
  --primary-metric val_loss --hypothesis "延迟合并改善验证损失" \
  --hypothesis-direction decrease --hypothesis-min-effect 0.02 \
  --hypothesis-baseline smelt --rq RQ1
```
- `--ablation-mechanism-controls` 是**显式 opt-in**：默认消融套件不变（避免家族规模与种子门槛被无意抬高）。
- 开启后计划单元从 3 模式 × 2 档 = 6 臂变成 5 模式 × 2 档 = 10 臂；对照臂必须写进预注册声明的模式集合，否则预注册覆盖检查会拒绝该次分析。
- 对照臂进入运行集后，探索性「全体两两对比」的家族规模从 15 升到 45，`--plan-design` 给出的保守所需种子数随之上升（稳健双检验下限约 11 个种子）；预先指定并分别校正的是 2×2 因子家族（每方案 7 个检验）与机制家族（每方案 2 个水平）。

### 2.4 计算量口径
- 使用 `--budget-match smelt` 检查完整计算图：Attention 执行次数 + MoE assignment + 路由。
- 报告解析 FLOPs 与实测比例，默认容差 10%，论文级要求 <4%。
- 成本倍数 ≥1.5× 的显著涨点会被标记 `[efficiency-flag]`，需评估性价比。

## 3. 统计可靠性

### 3.1 种子与功效
- 关键实验至少 6 个种子（稳健双检验下限）；3~5 个仅支持 t 检验。
- `--seeds` 自动给出 Mean ± Std、配对/Welch t、Cohen's d / Hedges g、Holm 校正、80% 功效 MDE、TOST 等价性、所需种子数。
- 伪重复检测：同一 seed 多次运行折叠为一个观测；`pseudo_replication` 记录组内差异。

### 3.2 选型偏差与共同步数
- 默认共同步数取各运行最后验证步最小值，数据依赖且有偏。正式对比必须预注册 `--common-step STEP`。
- 同时报告 `final`、`val_loss_at_common_step`、`val_loss_curve_mean_to_common_step` 三个口径，防止把早期下降误读为最终质量。
- `val_loss_curve_mean_to_common_step` 用于 RQ1 收敛速度，需与端点一起解读。

### 3.3 脆弱性与功效
- 报告符号翻转所需种子数、leave-one-out 稳定性、失败率/脱落率。
- 失败运行不进入均值，但计入 `planned_runs/failed_runs/failure_rate`；失败率必须与成功条件一起解读。

## 4. 路由诊断与功能干预

RQ3/RQ4 不能仅凭利用率：
- 每个 Jaccard 必须同时报告均匀随机与边际负载保持的机会水平、lift 与 adjusted 重合度。
- `--routing-interventions` 执行路由打乱与逐专家屏蔽，记录损失上升与 `importance_cv`，从利用率推进到功能干预证据。
- 结果写入 `routing.json` 与 `routing_interventions.json`，并进入 `--compare` 的可对比指标集。

## 5. 推荐执行流程

```bash
# 1. 试点估算方差与脱落率
python experiment.py --data corpus.txt --out runs/pilot \
  --seeds 42 43 44 --suite-modes smelt local_base A B \
  --steps 2000 --design-pilot-runs runs/pilot

# 2. 预注册并冻结代码
# 生成 design_plan.json，冻结 commit，写入 preregistration.json

# 3. 全因子消融（6 种子，预算匹配，固定共同步数）
# --ablation-mechanism-controls 会额外跑 local_base_k2 / B_early
# 两个只改合并时机/位置的对照臂；没有它们，延迟合并的收益
# 只能算多因素差异，不能归因到机制本身
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 45 46 47 \
  --suite-ablation --ablation-modes A B \
  --ablation-mechanism-controls \
  --common-step 2000 --budget-match smelt \
  --primary-metric val_loss --hypothesis "延迟合并改善验证损失" \
  --hypothesis-direction decrease --hypothesis-min-effect 0.02 \
  --hypothesis-baseline smelt --rq RQ1

# 4. 只读汇总与统计检验
python experiment.py --compare runs/rq1/* --compare-baseline smelt \
  --compare-metric val_loss_at_common_step \
  --compare-report runs/rq1/comparison_analysis.json
```

## 6. 输出与审计

每次运行生成：
- `config.json`：完整配置、数据指纹、comparison_protocol、paper_fidelity、routing_analysis、sanity_check。
- `metrics.jsonl`：训练/验证曲线、final、唯一一次 test。
- `comparison_analysis.json`：受控变量检查、Mean±Std、显著性、MDE/TOST、消融覆盖、单因素机制对照、效率与脆弱性。

归档前运行：
```bash
python experiment.py --write-manifest runs/rq1
python experiment.py --verify-manifest runs/rq1 --verify-manifest-strict
python experiment.py --audit-run runs/rq1 --audit-strict
```

遵循以上清单，对比实验与消融的结论可避免混淆、功效不足与选型偏差的常见陷阱。
