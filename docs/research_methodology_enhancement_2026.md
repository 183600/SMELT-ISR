# 科研方法改进记录 (2026)

## 改进背景
SMELT-ISR 已具备完整的预注册、受控变量、多种子统计、功效分析、脆弱性检验等科研方法基座。为进一步防止“无假设超参数排列组合”这类研究设计缺陷，本次改进强制多种子/套件实验必须声明科学假设。

## 改进内容

### 1. 多种子/套件运行强制假设声明
- **位置**：`exp/mainflow/part_000.py`
- **规则**：
  - 当 `--seeds` 包含 ≥2 个种子，或使用 `--suite-modes` / `--suite-ablation` 时，若 `--hypothesis` 为空且未使用 `--skip-hypothesis-check`，入口拒绝启动。
  - 错误信息明确提示需通过 `--hypothesis` 明确科学假设，豁免需显式使用 `--skip-hypothesis-check` 并会被记录进 `config.json`。
- **科研意义**：确保所有可做统计检验的对比实验都有预先声明的假设，避免事后挑选指标/方向的 p-hacking。单次调试运行不受影响，保持工程灵活性。

### 2. 与现有机制的衔接
- 该检查在参数解析后、训练前执行，不影响 `--compare`、`--sanity-only`、`--plan-design` 等只读入口。
- 豁免行为被记录为 `hypothesis_check_skipped`，可在分析阶段核对。
- 与预注册 `--preregistration` 机制互补：预注册提供更严格的字段级冻结，本检查提供轻量门槛。

### 3. 不修改受保护文件
- 未修改 `.github/` 目录下任何 CI 配置。
- 未修改 `README*.md` 等项目说明文档。

## 验证
```bash
# 拒绝无假设的多种子运行
python experiment.py --seeds 42 43 --mode B
# error: --seeds / --suite-modes / --suite-ablation 多种子/对比实验必须通过 --hypothesis 明确科学假设...

# 允许带假设的计划
python experiment.py --seeds 42 43 --mode B --hypothesis "延迟合并降低验证损失" --plan-design --out /tmp/out
# [design] 计划 2 次运行...

# 显式豁免
python experiment.py --seeds 42 43 --mode B --skip-hypothesis-check --plan-design --out /tmp/out
# 通过
```

## 后续建议
- 可将此门槛扩展为要求 `--primary-metric` 与 `--hypothesis-direction` 同时提供，形成更完整的预注册最小集。
- 在 `--compare` 报告中增加“假设覆盖率”指标：统计对比集合中缺失假设的运行比例。
