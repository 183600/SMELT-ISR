# SMELT-ISR 科研方法改进实施计划

> 基于 `ANALYSIS_IMPROVEMENTS.md` 与现有代码审计，提出在不修改 `.github/` 与 `README*.md` 前提下增强科研方法的可复现性、统计可靠性与结论稳健性的分阶段方案。

## 1. 现状评估

现有框架已具备：
- 受控变量协议 `comparison_protocol` 与完整元数据落盘
- 多种子聚合、Holm/BH 校正、MDE/TOST 等价性检验
- 脆弱性/脱落界、符号翻转检验、Wilcoxon/Cliff's δ
- 数据审计、泄露硬门、群组隔离、清洗偏差记录
- 预注册 Schema、`--preregistration` 训练前校验
- 论文复现入口 `--paper-cell/--paper-protocol` 与 `paper_fidelity`

方法学审计已修复 23 项外部审计问题，并在 `docs/methodology_audit.md` 中逐项记录。

## 2. P0 优先级改进

### 2.1 贝叶斯补充分析
**目标**：并行提供证据强度度量，不替代频率派判定。

**实现要点**：
- 在 `exp/part_025_statistics.py` 新增 `_bayesian_contrast()`，默认 Cauchy(0,0.707) 先验
- 计算 `bayes_factor_10`、`hdi_low/high`、`posterior_prob_superior`
- `--compare` 报告新增 `[bayesian]` 小节，`log_bf > 1` 实质证据，`> 2` 强证据
- 默认关闭，通过 `--bayesian` 显式开启，避免额外依赖污染 CI

**兼容性**：不改变 Holm/TOST 判定，仅作为证据分级补充。

### 2.2 结构化结论卡片
**目标**：每个 RQ/对比产出机器可读的标准化卡片。

**格式**：
```json
{
  "rq": "RQ1",
  "contrast": "smelt → B",
  "metric": "val_loss_at_common_step",
  "effect": {"g": -0.32, "ci95": [-0.58,-0.06]},
  "significance": {"holm_p": 0.018, "significant": true},
  "equivalence": {"margin": 0.01, "tost_holm_p": 0.34, "equivalent": false},
  "power": {"mde_80": 0.12, "power_at_obs": 0.61, "seeds_needed_80": 9},
  "fragility": {"verdict": "stable", "min_deletions": 3},
  "cost": {"flops_ratio": 1.42, "pareto": true},
  "verdict": "显著但功效不足，方向稳健"
}
```
落盘路径：`conclusions/<rq>_<contrast>.json` 与 `.md`。`--compare` 结束时自动生成。

## 3. P1 优先级改进

### 3.1 元分析框架
新增 `exp/part_050_meta.py`：DerSimonian-Laird/REML 随机效应模型，输出汇总效应 ĝ、τ²、I²、Cochran's Q、预测区间。用于跨数据集/套件汇总。

### 3.2 正式敏感性分析
系统扰动 `lr/weight_decay/balance_coef/grad_clip/seq_len` ±20%：
- Sobol 或 Morris 初等效应指数
- 输出 `sensitivity_report.json`
- 识别结论仅在极窄超参数区间成立的脆弱发现

### 3.3 复现包导出
`export_reproducibility_package(out_dir, format="zip|tar.gz")`：
- 依赖快照 `pip freeze` + 环境 ID
- `git archive` + 未提交 diff
- 所有种子 `config.json/metrics.jsonl/routing.json` + 效率测量
- `reproduce_analysis.py` 一键重跑 `--compare`

## 4. 接口与约束保证

- 所有新增参数默认关闭，旧命令行行为完全保持
- **严禁修改** `.github/` 下任何文件及 `README*.md`、`README_en.md` 等说明文档
- 新字段仅追加到 `config.json`/`comparison_analysis.json`，不重命名/删除旧键
- 可选依赖：贝叶斯/元分析/敏感性分析仅在显式开启时引入计算
- 每项新增分析在 `tests/` 增加回归测试

## 5. 实施里程碑

| 优先级 | 改进项 | 验收标准 |
|--------|--------|----------|
| P0 | 贝叶斯补充分析 | `--compare` 报告出现 `[bayesian]` 小节，BF10 与 HDI 并行输出 |
| P0 | 结构化结论卡片 | 每对比自动生成 JSON/Markdown 卡片，含效应/功效/脆弱性/成本/裁决 |
| P1 | 元分析框架 | `meta_analyze()` 输出 τ²/I²/PI/汇总 ĝ |
| P1 | 复现包导出 | `export_reproducibility_package()` 生成自包含 zip，含复现脚本 |
| P2 | 预注册 Schema 机器校验 | `validate_preregistration()` 违规阻断 `--compare` |
| P2 | 帕累托前沿分析 | `pareto_frontier_analysis()` 输出 HV/R2 与 CSV |

## 6. 使用示例

```bash
# 贝叶斯补充 + 结论卡片
python experiment.py --compare runs/rq1/* --bayesian --conclusion-cards

# 元分析跨套件
python -c "from exp.part_050_meta import meta_analyze; meta_analyze(contrasts, 'val_loss')"

# 导出复现包
python experiment.py --export-repro-package runs/rq1 --out reproducibility.zip
```

## 7. 风险与限制

- 贝叶斯因子对先验敏感，默认 Cauchy 为领域通用选择，需在报告中声明
- 结论卡片的裁决为自动生成，仍需人工审阅 `verdict` 文本
- 敏感性分析需要额外计算资源，建议在试点上先校准
- 复现包体积较大，需配合外部不可变存储（如 Zenodo）长期保存

---

本计划遵循 `ANALYSIS_IMPROVEMENTS.md` 的兼容性原则：不破坏现有 CLI、不触碰 `.github/` 与 README、增量落盘、可选依赖。实施时按优先级分批 PR，每批含实现、测试、文档更新。
