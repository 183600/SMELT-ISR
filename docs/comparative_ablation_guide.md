# 对比实验与消融的科研方法改进指南

本指南在现有 SMELT-ISR 基座上，围绕**受控变量对比**、**全因子消融**、**统计可靠性**与**计算量口径**四方面，明确可操作的改进点与默认检查清单。README 与 .github 目录保持不变，本文仅补充使用规范。

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

默认覆盖 A 与 B，缺失任一方案会被 `ablation_coverage` 列为 `missing_cells`。

### 2.2 结构混杂检查
消融单元内必须保持：
- `k1/k2/top_k/residual_scale` 恒定，否则主效应混入分支几何/路由宽度/残差缩放；
- 开启臂 `balance_coef` 必须唯一，多个正值会被视为 `confounded_designs`；
- `k1*k2 > n_experts` 时第二轮宽度会被截断，`local_base` 与 A 的等计算量假设不再严格，需在报告中显式标记。

`--suite-ablation --ablation-modes A B --seeds 42 43 44 45 46 47` 生成六格完整设计。

### 2.3 计算量口径
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
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 45 46 47 \
  --suite-ablation --ablation-modes A B \
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
- `comparison_analysis.json`：受控变量检查、Mean±Std、显著性、MDE/TOST、消融覆盖、效率与脆弱性。

归档前运行：
```bash
python experiment.py --write-manifest runs/rq1
python experiment.py --verify-manifest runs/rq1 --verify-manifest-strict
python experiment.py --audit-run runs/rq1 --audit-strict
```

遵循以上清单，对比实验与消融的结论可避免混淆、功效不足与选型偏差的常见陷阱。
