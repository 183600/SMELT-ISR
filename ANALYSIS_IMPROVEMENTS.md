# SMELT-ISR 结果分析方法学改进建议

本文档基于对现有代码库（`exp/`、`tests/`、`selfcheck/`）的全面阅读，从**结果分析、统计推断、可复现性、结论稳健性**四个维度提出改进建议。现有框架已具备受控变量协议、多种子聚合、MDE/TOST、曲线时均指标、选择偏差审计、脆弱性/脱落界等先进设施，建议在**不破坏现有接口**的前提下增强分析深度。

---

## 1. 统计推断增强

### 1.1 贝叶斯补充分析（并行而非替代）
- **动机**：频率派 p 值不回答「效应存在的概率」；贝叶斯因子 (BF) 与可信区间 (HDI) 能直接量证据强度。
- **实现**：在 `part_025_statistics.py` 新增 `_bayesian_contrast()`，使用默认 Cauchy(0, 0.707) 先验（Rouder et al., 2009）计算 BF₁₀ 与 95% HDI。
- **输出**：对比报告并行给出 `bayes_factor_10`、`hdi_low/high`、`posterior_prob_superior`；`log_bf > 1` 为实质证据，`> 2` 为强证据。
- **兼容性**：不改变现有 Holm/TOST 判定，仅作为「证据分级」补充；`--compare` 报告新增 `[bayesian]` 小节。

### 1.2 元分析框架（跨实验/数据集/种子汇总）
- **场景**：多个 `--suite-*` 产出的对比需汇总（如 RQ1 在 3 个数据集、6 个种子）。
- **实现**：新增 `exp/part_050_meta.py`，实现随机效应模型：
  - DerSimonian-Laird / REML 估计异质性 τ²
  - 汇总效应量 ĝ 及 95% CI
  - I² 统计量与 Cochran's Q 检验
  - 预测区间 (PI) 量化新实验效应落点范围
- **接口**：`meta_analyze(contrasts_list, metric)` 返回结构化汇总，可被 `--compare-report` 落盘。

### 1.3 效应量解释阈值的领域校准
- **现状**：Cohen's d / Hedges' g 使用通用阈值（0.2/0.5/0.8）。
- **改进**：在 `part_025_statistics.py` 增加 `--effect-size-calibration`，基于历史 MoE 路由消融（如 `base→smelt`、`A→B`）的经验分布，给出**领域特定**的「微小/小/中/大」分位点；报告同时输出 `d_percentile_in_domain`。

---

## 2. 结果分析深度

### 2.1 正式敏感性分析（超越 leave-one-out）
- **现状**：`_fragility_summary` 仅做种子删除敏感性。
- **扩展**：系统性扰动关键超参数（`lr`、`weight_decay`、`balance_coef`、`grad_clip`、`seq_len`）在 ±20% 范围内的效应稳定性：
  - 采用 **Sobol 指数** 或 **Morris 初等效应** 量化各超参数对结论的贡献
  - 输出 `sensitivity_report.json`：主效应/交互效应/总效应指数
- **用途**：识别「结论仅在极窄超参数区间成立」的脆弱发现。

### 2.2 成本-效益帕累托前沿分析
- **现状**：`[efficiency]` 与 `[cost]` 标记仅做成对标记。
- **扩展**：在 `part_020_curve_metrics.py` 新增 `pareto_frontier_analysis()`：
  - 以 `(val_loss, GFLOPs/token)` 或 `(val_loss, latency_ms)` 为目标空间
  - 计算帕累托最优集、超体积指标 (HV)、R2 指标
  - 自动生成 `pareto_frontier.csv` 供绘图
- **判据**：新方法必须在帕累托前沿上或给出明确权衡理由。

### 2.3 曲线形态聚类与代表性轨迹
- **动机**：`val_loss_curve_mean_to_common_step` 丢失形态信息（早快晚慢 vs 晚快早慢）。
- **实现**：在 `part_016_curve_analysis.py` 增加：
  - 动态时间规整 (DTW) 距离矩阵
  - 层次聚类识别典型收敛模式（如 `fast_converge`、`late_surge`、`oscillatory`）
  - 每簇输出代表性轨迹（中位数曲线）与簇内种子分布
- **应用**：`--compare` 报告新增 `[curve_phenotype]`：不同模式是否落入不同收敛表型。

### 2.4 坏案例的因果归因（超越相关性）
- **现状**：`bad_case_comparison_report` 仅统计跨运行稳定混淆对。
- **扩展**：引入**反事实干预**：
  - `--routing-interventions` 已有专家屏蔽/打乱，新增**上下文替换**：将易错 token 的前 K 字节替换为高频上下文，观察错误率变化
  - 计算**平均处理效应 (ATE)**：`E[error | do(context=common)] - E[error | context=original]`
  - 区分「模型能力不足」vs「数据分布偏移」vs「标签噪声」

---

## 3. 可复现性与证据链闭环

### 3.1 标准化复现包
- **现状**：`config.json` + `metrics.jsonl` + `best.pt` 分散存储。
- **改进**：新增 `export_reproducibility_package(out_dir, format="zip|tar.gz")`：
  - 固化依赖：`pip freeze --all` + `conda list --explicit` + `torch.__version__` + CUDA/cuDNN 版本
  - 代码快照：`git archive --prefix=repo/ HEAD` + 未提交改动 diff
  - 运行清单：所有种子的 `config.json`、`metrics.jsonl`、`routing.json`、效率测量
  - 分析复现脚本：`reproduce_analysis.py`（读取包内数据重跑 `--compare` 全流程）
- **用途**：审稿人/复现者一键验证全链路。

### 3.2 预注册协议的机器可验证 Schema
- **现状**：`hypothesis_preregistration` 为自由文本/键值对。
- **改进**：定义 JSON Schema (`schemas/preregistration.v1.json`)：
  - 必填：`primary_metric`、`direction`、`min_effect`、`equivalence_margin`、`baseline_mode`、`analysis_plan_hash`
  - 可选：`sample_size_justification`、`stopping_rules`、`exclusion_criteria`
- **工具**：`validate_preregistration(config.json)` 在训练前/分析期双向校验，**违规直接阻断** `--compare`。
- **实施状态（2026-11-14）**：schema 与训练前校验已实现；
  `stopping_rules`（`max_total_runs` 硬门 + 中期观察点声明）、
  `exclusion_criteria`（预声明排除规则）与
  `min_effect_justification`（最小效应依据，对应本节
  `sample_size_justification` 意图）已作为可选字段落地，
  见 `docs/methodology_audit.md` 第 49/50 节。
  `analysis_plan_hash` 仍以 `frozen_source_tree_sha256` +
  `--compare-preregistration` 覆盖门承担同等职责，独立字段
  未实现。

### 3.3 盲分析支持
- **动机**：防止分析期隐性偏好（如调整等价界、挑选指标）。
- **实现**：
  - `--blind-analysis` 模式：运行目录重命名为 `run_001`...`run_N`，模式标签哈希化
  - 分析脚本仅输出**效应量、p 值、CI**，不显示模式名
  - 解盲需显式 `--unblind-with <mapping.json>` 并记录时间戳
- **落盘**：`config.json` 新增 `blinding: {enabled: true, unblinded_at: "ISO8601", unblinded_by: "user"}`。

---

## 4. 结论稳健性与报告规范

### 4.1 结构化结论卡片
- **格式**：每个 RQ/对比生成标准化 `ConclusionCard`（JSON + Markdown 双格式）：
  ```json
  {
    "rq": "RQ1",
    "contrast": "smelt → B",
    "metric": "val_loss_at_common_step",
    "effect": {"g": -0.32, "ci95": [-0.58, -0.06], "d": -0.35},
    "significance": {"holm_p": 0.018, "significant": true},
    "equivalence": {"margin": 0.01, "tost_holm_p": 0.34, "equivalent": false},
    "power": {"mde_80": 0.12, "power_at_obs": 0.61, "seeds_needed_80": 9},
    "fragility": {"verdict": "stable", "min_deletions": 3},
    "attrition": {"effect_bounds": [-0.41, -0.19], "robust": true},
    "cost": {"flops_ratio": 1.42, "latency_ratio": 1.38, "pareto": true},
    "verdict": "B 在等计算量口径下显著优于 smelt（g=-0.32），但需 9 种子达 80% 功效；当前 6 种子功效 61%，结论方向稳健。"
  }
  ```
- **自动生成**：`--compare` 结束时写入 `conclusions/<rq>_<contrast>.json` 与 `.md`。

### 4.2 负面/零结果的显性归档
- **现状**：`inconclusive` 仅打印警告。
- **改进**：新增 `null_results_registry.jsonl`：
  - 记录每个「未通过 TOST 且未显著」的对比
  - 强制填写：`reason`（功效不足/效应真为零/方向不一致）、`mde_vs_obs`、`recommended_seeds`
  - 防止「文件柜效应」：同一假设的多次零结果可元分析。

### 4.3 多重比较的层级化控制
- **现状**：Holm 在单一指标家族内校正。
- **扩展**：三层级 FWER 控制（Westfall-Young 步进法）：
  1. **主假设层**：预注册的 RQ1 主指标（α=0.05）
  2. **次要假设层**：RQ2-RQ4 + 消融主效应（α=0.025，Holm 校正）
  3. **探索层**：所有其他指标/交互/亚组（α=0.10，BH-FDR）
- **实现**：`part_025_statistics.py` 新增 `_hierarchical_fwer(contrasts, hierarchy)`，报告每层调整后阈值与通过情况。

---

## 5. 实施优先级与里程碑

| 优先级 | 改进项 | 预估工作量 | 依赖 | 验收标准 |
|--------|--------|------------|------|----------|
| P0 | 贝叶斯补充分析 | 2-3 天 | `part_025_statistics.py` | `--compare` 报告新增 `[bayesian]` 小节，BF₁₀ 与 HDI 并行输出 |
| P0 | 结构化结论卡片 | 1-2 天 | `part_021_comparison.py` | 每对比自动生成 `.json`/`.md`，含效应量/功效/脆弱性/成本/裁决 |
| P1 | 元分析框架 | 3-4 天 | 新建 `part_050_meta.py` | 跨数据集/套件汇总给出 τ²、I²、PI、汇总 ĝ |
| P1 | 正式敏感性分析 | 3-4 天 | 需额外运行（网格/采样） | Sobol 指数识别关键超参数，输出 `sensitivity_report.json` |
| P1 | 复现包导出 | 2 天 | 现有 `config.json`/`metrics.jsonl` | `export_reproducibility_package()` 生成自包含 zip，含复现脚本 |
| P2 | 预注册 Schema + 机器校验 | 2 天 | 新建 `schemas/` | `validate_preregistration()` 违规阻断 `--compare` |
| P2 | 帕累托前沿分析 | 1-2 天 | `part_020_curve_metrics.py` | `pareto_frontier_analysis()` 输出 HV/R2 与 CSV |
| P2 | 盲分析模式 | 2 天 | CLI 入口 | `--blind-analysis` 隐藏模式名，解盲留痕 |
| P3 | 负面结果注册表 | 1 天 | `part_021_comparison.py` | `null_results_registry.jsonl` 自动追加 |
| P3 | 曲线表型聚类 | 2-3 天 | `part_016_curve_analysis.py` | DTW 聚类识别收敛模式，报告 `[curve_phenotype]` |

---

## 6. 兼容性原则

所有改进**必须**：
1. **不破坏现有 CLI**：新增参数默认关闭，旧命令行行为完全保持
2. **不修改 `.github/` 与 `README*.md`**：按约束严格遵守
3. **增量落盘**：新字段仅追加到 `config.json`/`comparison_analysis.json`，不重命名/删除旧键
4. **可选依赖**：贝叶斯/元分析/敏感性分析仅在显式开启时引入额外计算依赖（如 `scipy` 可选）
5. **测试覆盖**：每项新增分析在 `tests/` 增加对应回归测试（`_core_AnalysisImprovementTests.py`）

---

## 7. 示例：增强后的 `--compare` 报告结构

```text
[controlled] ... (现有)
[bayesian]  BF10=4.2 (moderate), HDI=[-0.51, -0.09], P(superior)=0.96
[meta]      τ²=0.012, I²=34%, PI=[-0.62, 0.02], ĝ=-0.30 [-0.48, -0.12]
[aggregate] ... (现有)
[compute-matched] ... (现有)
[common-step] ... (现有)
[convergence] ... (现有)
[curve_phenotype]  B: late_surge (n=4/6), smelt: fast_converge (n=5/6)
[stats] ... (现有)
[assumptions] ... (现有)
[fragility] ... (现有)
[attrition-bounds] ... (现有)
[equivalence] ... (现有)
[hierarchical-fwer]  Layer1(primary): α=0.050 → PASS; Layer2: α=0.025 → PASS; Layer3: FDR=0.10 → 3/12
[pareto]     B on frontier (HV=0.87), smelt dominated by B_early
[sensitivity]  lr (S1=0.42), balance_coef (S1=0.18), others <0.05
[conclusion] RQ1: smelt→B g=-0.32 [CI -0.58,-0.06], Holm p=0.018, TOST p=0.34, power=61%, seeds_needed=9, fragility=stable, pareto=true → VERDICT: Significant but underpowered; direction robust.
```

---

> **注**：本文档仅作为方法学改进建议，不直接修改代码。实施时请按优先级分批 PR，每批含实现、测试、文档更新。
