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

### 1. 结果分析的可重复可视化与审计日志
- 目前 `--compare-csv` 与 `--compare-report` 为机器可读。建议增加 `--compare-report-md` 自动生成人类可读的 Markdown 摘要，包含：
  - 效应方向、显著性、效应量及 95% CI / 同时区间
  - 脆弱性判定、`min_seed_deletions_to_lose_significance`
  - 功效/MDE、TOST 结果与 `equivalence_blocker` 说明
  - 方差分量主导来源建议（加种子 vs 扩评估集）
- 摘要应包含数据指纹、环境 ID、计划指纹，确保结果可追溯。

### 2. 跨指标一致性与稳健性综合评分
- 当前多指标独立报告。建议在 `--compare` 中增加跨指标一致性检查：
  - 对同一对比，多个预注册指标方向是否一致；不一致时标注 `[metric-divergence]`。
  - 提供基于秩的综合效应量，如汇总 Cliff's δ 的中位数与 IQR，避免单指标挑选。
- 对路由诊断指标，当前已报告 Jaccard chance level。建议增加机会水平校准的重采样 p 值，而非仅 lift。

### 3. 后验敏感性与外部效度提示
- 增加对评估窗口选择的敏感性：固定窗口的换窗口 bootstrap 已有，但未报告窗口大小与重叠对区间的影响。建议在 `comparison_analysis.json` 中增加 `window_size_sensitivity`：在当前固定窗口上，对窗口长度进行 ±20% 扰动，观察效应估计稳定性。
- 对数据切分策略，增加外推风险提示：当 `label_coverage.unseen_label_token_share` > 1e-4 时，在对比报告中自动标注 `[extrapolation-risk]`，并提示该部分 NLL 不可约。

### 4. 路由干预的多重比较校正
- `--routing-interventions` 当前逐专家报告区间分离，未对 E 个专家做多重校正。建议：
  - 对 `expert_drop` 的 Δloss 区间，使用 BH 控制 FDR，给出校正后的分离专家数量。
  - 在 `routing_interventions.json` 中增加 `specialization_fdr` 字段。

### 5. 结果分析的自动化决策树
- 将当前分散的诊断标记整合为决策树：
  1. `significant` 且 `fragile=fragile` → `inconclusive_fragile`
  2. `not_significant` 且 `power_at_effect < 0.5` → `underpowered`
  3. `equivalent` 未通过且 `effect_exceeds_margin` → `reject_equivalence`
  4. `dominant_source=window` → 建议扩大评估集而非加种子
- 决策结果写入 `comparison_analysis.json` 的 `result_verdict`，便于流水线自动判定。

### 6. 效率与精度的联合可视化
- 当前效率字段带 CI。建议在 CSV 导出时增加帕累托前沿判定：对每个对比，计算 `gain_per_cost` 与不确定区间，若区间跨 1 且成本 ≥1.5×，自动标记 `[cost-inefficient]`。

## 实施原则
- 所有新增字段保持列名只增不改，兼容现有 `--compare-csv`。
- 保持只读入口，不在训练/对比阶段修改代码。
- 预注册文件应可声明是否启用上述增强分析，避免事后挑选。

以上改进在不改动 `.github` 与 README 的前提下，可通过扩展 `exp/part_028_contrasts.py`、`exp/part_021_comparison.py` 与 `exp/part_024_window_comparison.py` 实现，并随 `comparison_analysis.json` 归档。
