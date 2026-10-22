# 实验设计与准备（先验功效、随机化区组、设计计划）

本文件说明训练开始前应完成的三个设计动作：

1. **先验功效分析**：用预期的种子间波动与最小有意义效应，
   在训练前判断计划种子数是否足够；
2. **随机化完整区组**：把「哪个臂先跑」从便利顺序变成预先
   声明的随机化顺序，避免机器热漂移/负载漂移与处理效应混杂；
3. **设计计划冻结**：把运行清单、顺序、功效与预算写成
   `design_plan.json`，子进程开跑前逐项核对。

这三项都属于**准备阶段**，与训练后的显著性检验、等价性检验和
脆弱性分析互补。它们不能替研究者保证结论正确，但能避免两类
常见、无法在分析阶段补救的设计错误：功效不足与顺序混杂。

## 1. 先做设计预检，不烧算力

```bash
python experiment.py \
  --data corpus.txt --out runs/confirm-design \
  --seeds 42 43 44 45 46 47 48 \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --run-order randomized --run-order-seed 20261022 \
  --plan-design \
  --hypothesis "延迟合并在完整计算图匹配下改善验证损失"
```

`--plan-design` 只生成并打印计划，**不加载数据、不训练**：

- 每个运行单元的 label / mode / seed / balance 系数；
- 执行顺序、种子区组与区组内位置；
- `design_plan.json` 的规范化指纹 `plan_id`；
- 先验功效（若预注册文件提供 `power` 块）；
- 解析 token 预算（`steps × batch_size × seq_len × 运行数`，
  不含评估/重算/优化器开销，也不是墙钟时间）。

确认计划后，再冻结代码、提交预注册文件并正式运行。正式套件
会重新生成同一份计划（同一 `--run-order-seed` 得到同一顺序），
子进程再核对计划指纹与自己的分配。

## 2. 先验功效（preregistration 的 `power` 块）

只要求「至少 3 个种子」只能保证 t 检验在数学上可算，不能保证
它能检出预注册的最小效应。`power` 块把预期噪声也写死：

```json
"power": {
  "expected_seed_sd_relative": 0.02,
  "target_power": 0.8,
  "alpha": 0.05,
  "planned_comparisons": 5,
  "underpowered_acknowledged": false
}
```

字段语义：

| 字段 | 必填 | 含义 |
|---|---|---|
| `expected_seed_sd_relative` | 是 | 预期种子间标准差，**相对对照臂均值**（0.02 = 2%）；通常来自试点运行或先前同规模实验 |
| `target_power` | 是 | 目标功效，惯例 0.8 |
| `alpha` | 是 | 未校正的家族错误率，惯例 0.05 |
| `planned_comparisons` | 否 | 检验家族规模，默认等于 `comparisons` 数量；**不得小于**它（低报会高估功效） |
| `underpowered_acknowledged` | 否 | 显式承认功效不足仍要运行（默认 false） |
| `justification` | 条件 | `underpowered_acknowledged=true` 时必填，写明理由 |

计算口径：

- 最小效应与预期标准差都相对基线均值，因此相除后是尺度无关的
  效应量，不需要事先知道基线均值；
- 家族校正用保守的 Bonferroni 下界 `alpha / planned_comparisons`，
  与分析阶段 Holm 首步阈值一致；
- 所需种子数同时受两个约束：
  - 配对 t 检验达到目标功效所需 n（`_seeds_to_detect`）；
  - 精确符号翻转在家族校正下能达到 alpha 所需
    `ceil(log2(2m/alpha))`；
- 取两者较大值作为 `required_seeds`。只主张 t 检验时可单独看
  `required_seeds_for_t`。

种子数不足时**默认拒绝启动预注册运行**，错误信息给出所需种子数
与两个口径。确需在功效不足的设计上探索时，必须显式写
`underpowered_acknowledged: true` 与 `justification`；该承认会随
`config.json` 的 `preregistration.power` 落盘，分析报告不能把它
改写成「计划足够」。

注意：`required_seeds` 把声明的预期标准差当作真实 sigma；它本身
也是估计，规划时应留余量。`adequate=true` 只表示「在声明效应下
有目标功效」，不保证真实效应一定达到最小有意义效应。

## 3. 运行顺序：随机化完整区组

旧行为（`grouped`）按模式分组：

```text
A_seed1, A_seed2, A_seed3, B_seed1, B_seed2, B_seed3
```

如果机器在跑 B 时更热、共享缓存已预热或集群负载变化，B 与时间
完全混杂，方法差异无法与漂移分离。项目现在默认
`randomized`：每个种子是一个**完整区组**，区组内所有臂都跑一次，
臂顺序由 `(order_seed, seed)` 确定性洗牌：

```text
block seed1: B, A
block seed2: A, B
block seed3: B, A
```

三种方案：

| `--run-order` | 行为 | 用途 |
|---|---|---|
| `grouped` | 保留历史顺序（按模式分组） | 只用于兼容旧运行；不能支持时间漂移下的因果归因 |
| `blocked` | 按种子成区组，区组内保持原臂顺序 | 需要固定臂顺序但希望平衡时间片 |
| `randomized` | 按种子成区组，区组内随机化 | **推荐的确证设计** |

`--run-order-seed` 固定洗牌种子（默认 0）；同一计划必须使用同一
种子才能复现顺序。看到结果后换顺序不再是原设计。

在预注册文件中可把顺序一并冻结：

```json
"run_order": {
  "scheme": "randomized",
  "seed": 20261022
}
```

命令行显式给出不同值时会被拒绝，而不是「以命令行覆盖预注册」。

### 3.1 子进程核对与 `run_order.json`

套件父进程在训练前写 `design_plan.json`，并通过环境变量把计划
文件、`plan_id`、文件 SHA-256 与本次分配（`order_index`、区组、
区组内位置）传给子进程。子进程开跑前会：

- 重新哈希计划文件，确认没有被改动；
- 核对 `plan_id`、顺序方案与顺序种子；
- 核对 label / seed / mode / balance 系数与计划条目一致；
- 核对计划位于本次 `--out` 的父目录。

任一不符即拒绝训练。实际开始/结束时间写到
`<out>/run_order.json`，失败条目也会保留；套件结束后父进程再核对
所有子运行的 `design_assignment.plan_id` 一致。

## 4. 分析阶段的顺序漂移诊断

`--compare` 与套件汇总会对带设计记录的非确定性运行做描述性
诊断（`comparison_analysis.json` 的 `run_order`）：

- 计划指纹是否一致（不同研究不能拼在一起）；
- 是否仍在使用 `grouped`（模式与时间完全混杂）；
- 去掉模式均值后，指标与执行顺序的 Spearman 相关 `ρ` 与
  近似 p 值；
- 每个模式内部指标是否随执行顺序单调变化。

`|ρ| ≥ 0.7` 且至少 4 个运行时会给出 `[warn]`，提示检查机器
负载/热漂移并重跑受影响区组。这是**描述性诊断**，不是显著性
检验；随机化区组只能平衡时间漂移，不能消除它。没有设计记录的
旧运行显示 `available=false`，不能把「无法核对」当作「没有漂移」。

## 5. 推荐的确证准备流程

1. 用试点/先前运行估计种子间标准差，写出 `power` 块；
2. 用 `--plan-design` 检查运行清单、随机化顺序、功效与 token
   预算；调整种子数或最小效应后再冻结；
3. 把 `design_plan.json`、预注册文件、`git archive` 与数据清单
   放入同一只读目录；
4. 正式运行时使用同一 `--run-order-seed`；子进程核对计划；
5. 分析时同时报告 `power`、`seed_plan` 与 `run_order`；顺序相关
   强时先调查漂移，而不是直接解释臂间差异；
6. 增加种子/重跑必须作为新研究阶段重新预注册，不能反复换种子
   直到显著。

## 6. 与现有协议的关系

- `docs/experiment_protocol.md` 的冻结 → 预注册 → 运行 → 统一
  解封流程不变；本文件补上「预注册前如何判断设计可执行」；
- `docs/methodology_audit.md` 第 27 节记录本轮修正；
- 设计计划不替代 `artifact_manifest.json`：前者是训练前设计，
  后者是训练后归档内容完整性；两者都应随最终报告保存。
