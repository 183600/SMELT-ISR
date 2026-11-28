# 结果分析角度的科研方法改进建议

本文件基于当前代码与 `methodology_audit.md` 的审计结果，从**结果分析**阶段提出可进一步强化的科研方法点。所有建议均不涉及 `.github` 与 `README*.md` 文件。

## 已具备的强项

当前实现已在结果分析层面覆盖以下最佳实践：

- 多随机种子聚合（Mean±Std、t 95% CI）+ 配对/Welch 检验、Holm/BH 多重校正、Hedges g 效应量校正。
- 功效与等价性框架：MDE、80% 功效种子数、TOST 等价性检验、`equivalence_margin_needed` 与精度不足预警。
- 脆弱性分析：精确符号检验、leave-one-out / 删除 k 个种子显著性敏感性、fragility_summary 归档。
- 同时置信区间：Bonferroni 同时区间与逐检验区间分离报告，明确区分描述性区间与 Holm 判定。
- 种子×窗口两层不确定性：`_hierarchical_window_bootstrap_ci` 与方差分量分解 `τ²/ω²/σ²`，判定主导不确定来源。
- 失败/缺失处理：attrition_bounds 部分识别界、差分缺失警告、失败率报告。
- 路由干预不确定性：`routing_shuffle` / `expert_drop` 配对 bootstrap 区间，区间分离作为专门化证据。
- 坏案例流式统计：全量 token 归因、位置/频率分层、跨运行混淆对出现频率聚合。
- 受控变量协议与预注册核对，保证结果可归因。

## 可进一步改进的方向

> 实施状态（2026-11-28）：下列 6 项已全部落地，回归测试见
> `tests/_core_ResultAnalysisTests.py`；本节保留原始建议文本以便
> 对照审计，实际行为以各条「已实现」小节为准。

### 1. 结果分析的可重复可视化与审计日志
- 目前 `--compare-csv` 与 `--compare-report` 为机器可读。建议增加 `--compare-report-md` 自动生成人类可读的 Markdown 摘要，包含：
  - 效应方向、显著性、效应量及 95% CI / 同时区间
  - 脆弱性判定、`min_seed_deletions_to_lose_significance`
  - 功效/MDE、TOST 结果与 `equivalence_blocker` 说明
  - 方差分量主导来源建议（加种子 vs 扩评估集）
- 摘要应包含数据指纹、环境 ID、计划指纹，确保结果可追溯。

**已实现**
- `exp.write_comparison_report_markdown(path, analysis)`（`exp/part_036_checks.py`）生成人类可读摘要：判定指标与方向、逐对比的 Δ/Holm p/显著性/等价性/功效/脆弱性/裁决/建议、裁决理由与标记、精度/效率权衡、外推风险、以及「解读边界」说明。
- `exp.write_comparison_report_file` 落盘 JSON 时会在同目录写出同名 `.md` 摘要（JSON 仍是完整证据结构，Markdown 只做转述，不重算统计量）。
- 追溯指纹由 `exp.part_025_statistics._result_fingerprints` 汇总（`data_sha256` / `split_sha256` / `effective_train_sha256` / `seq_len` / `source_tree_sha256` / `script_sha256` / `environment_id` / `git_commit`）；同一键出现多个取值时摘要给出 `[warn]`，避免把跨数据/跨环境的对比伪装成同一份实验。

### 2. 跨指标一致性与稳健性综合评分
- 当前多指标独立报告。建议在 `--compare` 中增加跨指标一致性检查：
  - 对同一对比，多个预注册指标方向是否一致；不一致时标注 `[metric-divergence]`。
  - 提供基于秩的综合效应量，如汇总 Cliff's δ 的中位数与 IQR，避免单指标挑选。
- 对路由诊断指标，当前已报告 Jaccard chance level。建议增加机会水平校准的重采样 p 值，而非仅 lift。

**已实现**
- `exp.part_025_statistics._cross_metric_effect_profile` / `_cross_metric_consistency_report`：在**所有可用的最终评估点指标**上重算效应，方向统一折算为「变体相对基线是改善还是变差」，并给出族级方向、族级秩效应量中位数与跨族综合 Cliff's δ（中位数 + IQR）。
- 指标族划分（`_CROSS_METRIC_FAMILIES`）：`val_loss`/`val_ppl`/`val_bits_per_byte` 等互为单调变换的指标归为同族，综合效应量按**族**汇总，不把同族指标当作多份独立证据；族划分随结果落盘。
- `exp.part_028_contrasts._cross_metric_consistency_analysis` 把剖面挂到对比记录上：`cross_metric`、`cross_metric_consistent`/`cross_metric_divergent`、`significant_metrics`、`metric_directions`、`cross_metric_summary`；分歧时摘要与终端输出 `[metric-divergence]`。

### 3. 后验敏感性与外部效度提示
- 增加对评估窗口选择的敏感性：固定窗口的换窗口 bootstrap 已有，但未报告窗口大小与重叠对区间的影响。建议在 `comparison_analysis.json` 中增加 `window_size_sensitivity`：在当前固定窗口上，对窗口长度进行 ±20% 扰动，观察效应估计稳定性。
- 对数据切分策略，增加外推风险提示：当 `label_coverage.unseen_label_token_share` > 1e-4 时，在对比报告中自动标注 `[extrapolation-risk]`，并提示该部分 NLL 不可约。

**已实现**
- `exp.part_025_statistics._window_size_sensitivity_analysis` 重写为**评估集规模敏感性**（窗口子集重采样，head/tail/middle 80%、head/tail 50%）：报告 `verdict`（stable/sensitive）、`max_abs_drift`、`window_level_se`、方向翻转与显著性翻转的子集。判定阈值：方向翻转或显著性翻转，或漂移同时超过窗口层噪声的 2 倍标准误与效应量的 20%。窗口长度固定在训练期，无法在分析期重算更短/更长窗口，故该字段**不等价于**窗口长度敏感性（后者需要按不同 `seq_len` 重新评估）。
- `exp.part_025_statistics._extrapolation_risk_report`：当 `label_coverage` 记录中的未见标签 token 占比超过 1e-4 时标记 `[extrapolation-risk]`，并列出缺少覆盖记录的运行（不臆造）。
- `exp.part_021_comparison` 已在对比行上带出 `label_coverage`。

### 4. 路由干预的多重比较校正
- `--routing-interventions` 当前逐专家报告区间分离，未对 E 个专家做多重校正。建议：
  - 对 `expert_drop` 的 Δloss 区间，使用 BH 控制 FDR，给出校正后的分离专家数量。
  - 在 `routing_interventions.json` 中增加 `specialization_fdr` 字段。

**已实现**
- `exp.part_028_contrasts._routing_specialization_fdr`：每次运行内的 E 个专家构成一个 BH 族（p 值由配对 bootstrap 95% 区间的正态近似换算，单侧），再在实验单元内按种子做一致性汇总（过半独立种子显著才判定为共识专门化）。输出 `routing_specialization_fdr`（含逐运行 BH 结果与逐专家共识计数）。
- 原 `_routing_fdr_correction` 已移除：它把 p 值跨对比汇总、读取不存在的字段，永远不生效。

### 5. 结果分析的自动化决策树
- 将当前分散的诊断标记整合为决策树：
  1. `significant` 且 `fragile=fragile` → `inconclusive_fragile`
  2. `not_significant` 且 `power_at_effect < 0.5` → `underpowered`
  3. `equivalent` 未通过且 `effect_exceeds_margin` → `reject_equivalence`
  4. `dominant_source=window` → 建议扩大评估集而非加种子
- 决策结果写入 `comparison_analysis.json` 的 `result_verdict`，便于流水线自动判定。

**已实现**
- `exp.part_025_statistics._result_verdict(contrast, alpha)`：把混杂、脆弱显著性、缺失界跨零、功效不足、等价性阻断、方差主导来源合成为机器可读裁决（`result_verdict.verdict` + `result_verdict.action` + 理由列表），逐对比写入 `result_verdict` 字段，汇总写入 `result_extensions.result_verdicts`。
- 动作建议取值：`add_seeds`（加种子）、`expand_eval_set`（扩大评估集，评估窗口方差主导时）、`complete_runs`（补齐缺失运行）、`fix_protocol`（混杂不可归因）。

### 6. 效率与精度的联合可视化
- 当前效率字段带 CI。建议在 CSV 导出时增加帕累托前沿判定：对每个对比，计算 `gain_per_cost` 与不确定区间，若区间跨 1 且成本 ≥1.5×，自动标记 `[cost-inefficient]`。

**已实现**
- `exp.part_025_statistics._cost_efficiency_assessment`：`gain_per_cost = 相对增益 / 额外成本倍率（cost_ratio − 1）`，>1 表示额外代价以 1:1 以上的比例被赚回；区间用最不利增益与最有利成本组合（保守口径），分母与点估计同为「额外成本」。
- 成本倍率 ≥1.5× 时：区间整体 ≤1 → `[cost-inefficient]`；区间跨 1 → `[cost-unproven]`；变体不比基线贵 → 不存在取舍。效应区间缺失时由 `delta`/`se`/`df` 还原（与显著性判定同一 t 临界值）。
- CSV 导出新增列：`cost_inefficient`、`cost_unproven`、`gain_per_cost`、`gain_per_cost_ci95_low/high`、`cost_ratio`（列名只增不改）。

## 实施原则
- 所有新增字段保持列名只增不改，兼容现有 `--compare-csv`。
- 保持只读入口，不在训练/对比阶段修改代码。
- 预注册文件应可声明是否启用上述增强分析，避免事后挑选。

以上改进在不改动 `.github` 与 README 的前提下，可通过扩展 `exp/part_028_contrasts.py`、`exp/part_021_comparison.py` 与 `exp/part_024_window_comparison.py` 实现，并随 `comparison_analysis.json` 归档。
