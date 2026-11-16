# 实验设计与准备阶段科研方法改进方案 (2026)

> 本文档在不修改 `.github/` CI 配置与 `README*.md` 说明文档的前提下，针对 SMELT-ISR 的实验设计与准备阶段提出可落地的科研方法增强措施。所有建议均与现有 `docs/design_preparation.md`、`docs/experiment_protocol.md`、`docs/methodology_audit.md` 互补，并复用 `--plan-design`、`--preregistration`、`--design-pilot-runs` 等已实现机制。

## 1. 现状与改进动机

当前项目已具备：
- 预注册 Schema 与训练前校验、假设强制声明
- 先验功效分析、试点校准、随机化区组与 carryover 平衡
- 数据审计、泄露硬门、群组隔离与清洗偏差记录
- 受控变量协议、多种子统计、MDE/TOST、脆弱性检验

改进重点放在**准备阶段的不可回溯性与可审计性**：数据口径、清洗决策、计算预算、随机化顺序与试点证据必须在第一条训练进程启动前冻结，并在子进程启动时逐项核对，防止“先跑后补”导致的选择性报告。

## 2. 数据与材料准备的强化

### 2.1 数据材料版本钉住
- **要求**：`preregistration.json` 必须包含 `data.sha256`、`entity_manifest_sha256`、`time_order_manifest_sha256` 三项指纹。训练入口在 `--plan-design` 阶段即校验语料文件实际 SHA-256 与预注册一致，不一致直接拒绝。
- **理由**：避免语料静默更新导致结果不可复现。
- **实现**：复用 `exp/part_049_data_plan.py` 的 `data_plan_preregistration_block`，在 `design_plan.json` 中增加 `data_fingerprint_verified` 字段。

### 2.2 清洗决策预注册强制化
- **要求**：任何数据清洗开关（`--drop-train-corrupt-bytes`、`--drop-train-near-duplicate-groups` 等）必须在预注册 `data.cleaning` 块中显式声明布尔值与数值阈值。训练入口比对命令行实际值与预注册值，差异即拒绝。
- **理由**：清洗会引入 `deletion_bias`，事后选择清洗策略会产生结果驱动的选择偏倚。
- **现有支撑**：`data_plan_cleaning_block` 已能生成可冻结对象，需在 `exp/part_033_preregistration_core.py` 中加入清洗块校验。

### 2.3 分割策略可证伪性
- **要求**：`--split-policy chronological` 时必须提供 `time-order-manifest`，且设计阶段即执行单调性校验；`group-separator` 切分时要求 `data_audit.group_count` 与预注册声明一致。
- **理由**：文件顺序≠时间顺序，缺证据时不能声称时间外推。

### 2.4 数据泄露扫描覆盖率门槛
- **要求**：设计阶段 `leakage_scan_coverage` 需 ≥95% 或 stride=1；在 `design_plan.json` 中记录覆盖率，未达标时默认拒绝启动，除非显式 `--allow-data-leakage` 并记录豁免理由。
- **理由**：抽样扫描的“未发现”≠不存在。

## 3. 计算预算与模型口径的准备校验

### 3.1 FLOPs/参数量预审计
- **要求**：`--plan-design` 阶段使用 `--budget-match smelt` 解析完整的计算图，输出 `budget_audit.predicted_flops_ratio` 与 `predicted_param_ratio`，并与预注册 `compute_budget` 声明的上限核对。
- **理由**：RQ2 中的 $k_1+k_1k_2$ 计算量常被误算为 $k_1k_2$，预算不匹配会导致“方法收益”实为计算量差异。

### 3.2 残差缩放与均衡强度锁定
- **要求**：跨模式对比时，`residual_scaling` 与 `balance_coef × effective routing width` 必须在计划阶段核对并记录到 `comparison_protocol`；不同值时 `--compare` 发出 `[balance-scale]` 警告并拒绝对比。
- **理由**：宽度变化若未折算均衡系数，会把路由宽度与均衡强度混杂。

## 4. 设计计划与随机化

### 4.1 试点校准上界口径强制
- **要求**：若预注册声明 `power.pilot_calibration = "upper_confidence_bound"`，`--plan-design` 必须在试点样本量满足 `sd_planning` 容差 ≤1.5 时才通过；否则给出 `recommended_pilot_runs_per_cell`。
- **理由**：小试点 sd 点估计不稳定，直接使用会低估所需种子数。

### 4.2 随机化顺序冻结与子进程核对
- **要求**：`run_order.scheme` 与 `run_order.seed` 必须在预注册中声明；父进程生成的 `design_plan.json` 的 `plan_id` 被子进程在启动前重新哈希校验。顺序变更即视为新研究阶段。
- **已有**：`exp/part_045_design_plan.py` 已实现，改进为在 `design_plan.json` 中增加 `order_balance.max_position_imbalance` 与 `carryover.balanced` 的硬门：`counterbalanced`/`williams` 方案若未平衡直接拒绝。

### 4.3 失败策略与脱落预声明
- **要求**：`stopping_rules.max_total_runs` 与 `exclusion_criteria.attrition_definition` 必须在预注册中完整声明。设计阶段按 `expected_attrition_rate` 计算 `required_seeds_with_attrition`，若计划种子数不足则拒绝。
- **理由**：防止“不显著就再加种子”的序贯偷看。

## 5. 假设与指标的准备阶段锁死

### 5.1 多种子运行强制假设与主指标
- **现状**：已强制 `--hypothesis` 声明。建议扩展为强制同时声明 `primary_metric` 与 `hypothesis_direction`，形成最小预注册集。
- **实现**：在 `exp/part_033_preregistration_core.py` 的参数校验中增加 `hypothesis_direction` 必填检查。

### 5.2 等价界与最小效应一致性检查
- **要求**：设计阶段检查 `equivalence_margin <= min_effect`，且 `plan_equivalence=true` 时同时校验显著性与 TOST 功效。若不满足，在 `--plan-design` 阶段即拒绝，不等到分析阶段。
- **理由**：避免等价界事后放宽制造等价结论。

### 5.3 结论卡片预生成模板
- **要求**：`--plan-design` 时预生成结论卡片模板，列出计划对比、指标、功效、所需种子数，便于研究者在运行前核对研究问题是否可被检验。
- **复用**：`exp/part_052_conclusion_cards.py` 可在计划阶段输出占位卡片。

## 6. 操作流程建议

1. **数据冻结**：生成 `data_plan.json`，校验指纹与清洗口径。
2. **预注册**：填写 `preregistration.json` 含数据、清洗、预算、功效、随机化、停止规则。
3. **设计预检**：`--plan-design --design-pilot-runs ...`，检查 `adequate`、`pilot_evidence`、`order_balance`、`budget_audit`。
4. **代码冻结**：记录 `EXPERIMENT_FREEZE` commit。
5. **正式运行**：子进程核对 `plan_id`、指纹、顺序。
6. **统一解封**：完成后统一 `--final-test-comparison`。

## 7. 约束声明

- **未修改** `.github/` 下任何 CI 配置文件。
- **未修改** `README.md`、`README_en.md` 等项目说明文档。
- 所有新增检查均为训练前门控，不影响 `--compare`、`--sanity-only` 等只读入口。
- 新增字段仅追加到 `config.json`、`design_plan.json`、`comparison_analysis.json`，不重命名或删除旧键。

## 8. 验收标准

- `--plan-design` 输出包含 `data_fingerprint_verified`、`cleaning_prereg_match`、`budget_audit`、`order_balance.balanced`、`power.adequate`。
- 清洗开关未预注册时训练入口抛出明确错误。
- `run_order` 未在预注册中声明时拒绝启动。
- `equivalence_margin > min_effect` 时 `--plan-design` 拒绝。

---

本方案遵循“先验锁定、事后不可更改”的科研原则，在现有 SMELT-ISR 方法学基座上，进一步把实验设计与准备阶段的决策从隐式习惯转为显式、可审计、可机器核对的协议。
