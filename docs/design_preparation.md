# 实验设计与准备（先验功效、随机化区组、设计计划）

本文件说明训练开始前应完成的三个设计动作：

1. **先验功效分析**：用预期的种子间波动、最小有意义效应与
   预期数据脱落率，在训练前判断计划种子数是否足够，并同时
   报告计划可用种子数下的实际功效与 MDE；
2. **随机化/平衡完整区组**：把「哪个臂先跑」从便利顺序变成
   预先声明的顺序，避免机器热漂移/负载漂移与处理效应混杂；
3. **设计计划冻结**：把运行清单、顺序、位置平衡、功效与预算
   写成 `design_plan.json`，子进程开跑前逐项核对。

这三项都属于**准备阶段**，与训练后的显著性检验、等价性检验和
脆弱性分析互补。它们不能替研究者保证结论正确，但能避免几类
常见、无法在分析阶段补救的设计错误：功效不足、顺序混杂与
运行脱落造成的有效样本量不足。

## 1. 先做设计预检，不烧算力

```bash
python experiment.py \
  --data corpus.txt --out runs/confirm-design \
  --seeds 42 43 44 45 46 47 48 \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --run-order counterbalanced --run-order-seed 20261022 \
  --plan-design \
  --hypothesis "延迟合并在完整计算图匹配下改善验证损失"
```

`--plan-design` 只生成并打印计划，**不加载数据、不训练**：

- 每个运行单元的 label / mode / seed / balance 系数；
- 执行顺序、种子区组与区组内位置，以及位置平衡审计
  （`order_balance`：每个处理在各位置上的计数与最大极差）；
- `design_plan.json` 的规范化指纹 `plan_id`；
- 先验功效（若预注册文件提供 `power` 块），包括按预期脱落率
  折算的可用种子数、实际功效、MDE 与 sd 敏感性；`planned_contrasts`
  声明与实际计划推断的对比类型会取并集，逐项给出 `contrast_power`
  与最吃种子的 `binding_contrast`（消融通常是 `interaction`）；
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
  "expected_attrition_rate": 0.1,
  "underpowered_acknowledged": false
}
```

若设计是 `--suite-ablation`，建议显式写
`"planned_contrasts": ["pairwise", "main", "interaction", "simple"]`：
交互的方差是两两对比的 2 倍，所需种子数也更高；只声明
`pairwise` 会在设计计划阶段被实际计划推断出的 `interaction`
覆盖，从而可能在开跑前被功效硬门拒绝。

字段语义：

| 字段 | 必填 | 含义 |
|---|---|---|
| `expected_seed_sd_relative` | 是 | 预期**两格配对差值**的种子间标准差，相对对照臂均值（0.02 = 2%）；若只有单配置跨种子的标准差 σ_cell，配对差值通常不超过 √2·σ_cell，应乘 √2 或直接用配对差值估计，否则会低估所需种子 |
| `target_power` | 是 | 目标功效，惯例 0.8 |
| `alpha` | 是 | 未校正的家族错误率，惯例 0.05 |
| `planned_comparisons` | 否 | 检验家族规模，默认等于 `comparisons` 数量；**不得小于**它（低报会高估功效） |
| `planned_contrasts` | 否 | 计划主张的对比类型数组，可选 `pairwise` / `simple` / `main` / `interaction`，默认 `["pairwise"]`；消融设计应声明 `["pairwise","main","interaction","simple"]` |
| `expected_attrition_rate` | 否 | 预期因 NaN/Inf、数值停止或数据问题而不可用的运行比例（默认 0）；必须在 `[0, 1)` 内 |
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
- `required_seeds_with_attrition = ceil(required_seeds / (1 - 预期脱落率))`
  才是**启动**种子数；`planned_usable_seeds = floor(计划种子数 ×
  (1 - 预期脱落率))` 是功效公式真正使用的 n。`adequate` 用计划
  启动种子数与 `required_seeds_with_attrition` 比较（等价于要求
  可用种子数 ≥ 所需可用种子数），避免把「计划了 8 个、预计坏
  2 个」当成 8 个独立重复。
- `achieved_power_at_min_effect` 与
  `minimum_detectable_effect_relative` 直接回答「按当前计划，
  即使真实效应等于预注册最小效应，检出概率是多少」「当前设计
  在 80% 功效下最小能检出多大效应」。它们与 `min_effect` 同单位，
  可直接比较。
- `power_sensitivity` 对 `expected_seed_sd_relative` 乘以
  0.5/0.75/1.0/1.25/1.5/2.0 重算所需种子数：先验 sd 本身是估计，
  只报一个 `required_seeds` 会把估计误差藏起来。若 sd 估计偏小
  25% 就需要更多种子，应在冻结前留余量或承认设计脆弱。

### 2.1 对比类型不同，方差也不同

`expected_seed_sd_relative` 是**两格配对差值**的标准差，不是
任意线性组合的标准差。2×2 消融（L=立即合并对照，M=延迟合并，
balance=off/on）中各对比的方差不同：

| 对比类型 | 线性组合 | 方差（相对两两对比） | SE 乘数 |
|---|---|---|---|
| `pairwise` / `simple` | `M_on − L_on` 等两格差 | 1× | 1 |
| `main` | `½[(M_on−M_off)+(L_on−L_off)]` | 0.5× | 1/√2 |
| `interaction` | `(M_on−M_off)−(L_on−L_off)` | 2× | √2 |

所以**交互的标准误是两两对比的 √2 倍，同样效应下约需 2 倍
种子**；主效应反而只需约一半。只按两两对比算「种子够了」、
却要在结果里解释消融交互，是设计阶段最常见的功效高估。

`power.planned_contrasts` 可显式声明本次研究要主张哪些类型
（默认只声明 `pairwise`）。`--plan-design` 与正式套件还会从
实际运行计划推断：`--suite-ablation` 会估计主效应、交互与
简单效应，因此设计计划把推断类型与声明类型取并集，逐项写入
`contrast_power`，并用最吃种子的类型（`binding_contrast`，
通常是 `interaction`）判定 `adequate`。正式套件在第一个子进程
启动前重新计算；仍不足且未显式承认时直接拒绝，而不是等分析
阶段再说「交互不显著」。

种子数不足时**默认拒绝启动预注册运行**，错误信息给出所需启动
种子数（含脱落折算）、最吃种子的对比类型与两个口径。确需在
功效不足的设计上探索时，必须显式写 `underpowered_acknowledged:
true` 与 `justification`；该承认会随 `config.json` 的
`preregistration.power` 落盘，分析报告不能把它改写成「计划足够」。

注意：`required_seeds` 把声明的预期标准差当作真实 sigma；它本身
也是估计，规划时应留余量。`adequate=true` 只表示「在声明效应下
有目标功效」，不保证真实效应一定达到最小有意义效应。预期脱落率
也只能来自试点或历史数据，不能用本轮结果事后反推。

## 3. 运行顺序：随机化/平衡完整区组

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

四种方案：

| `--run-order` | 行为 | 用途 |
|---|---|---|
| `grouped` | 保留历史顺序（按模式分组） | 只用于兼容旧运行；不能支持时间漂移下的因果归因 |
| `blocked` | 按种子成区组，区组内保持原臂顺序 | 需要固定臂顺序但希望平衡时间片 |
| `randomized` | 按种子成区组，区组内随机化 | 默认；区组内顺序随机，但小样本下不保证位置计数均衡 |
| `counterbalanced` | 按种子成区组，用循环拉丁方轮换臂位置 | 种子数较少、担心位置与热漂移混杂时的限制性随机化；要求每个区组包含同一组处理 |

`randomized` 只保证「每个区组都跑每个臂」，不保证每个臂在
区组第 1/2/3… 个位置上的出现次数相等。6 个种子、2 个臂时，
随机化可能让某个臂 5 次都排在后半段；若位置与缓存预热、GPU
温度或集群负载相关，臂效应仍会混入位置效应。`counterbalanced`
先把处理标签用 `--run-order-seed` 确定性随机化，再让第 b 个
种子区组使用基础序列的第 `(start+b)` 个循环移位：任意连续
b 行中每个处理在各位置最多出现一次，位置计数极差不超过 1。
它要求每个种子区组包含同一组处理；若某个候选只在部分种子上
运行，`--plan-design` 会直接拒绝，而不是生成一个看似平衡、
实则缺格的计划。

`design_plan.json` 的 `order_balance` 会记录每个处理在各位置
上的计数、`max_position_imbalance` 与 `balanced`。`randomized`
不均衡时只提示可改用 `counterbalanced`；`counterbalanced`
不均衡则视为设计错误并告警。

`--run-order-seed` 固定洗牌种子（默认 0）；同一计划必须使用同一
种子才能复现顺序。看到结果后换顺序不再是原设计。

在预注册文件中可把顺序一并冻结：

```json
"run_order": {
  "scheme": "counterbalanced",
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
所有子运行的 `design_assignment.plan_id` 一致。`plan_id` 覆盖
`order_balance`，因此事后把不平衡顺序改成平衡顺序（或反过来）
也会改变指纹。

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

1. 用试点/先前运行估计**两格配对差值**的种子间标准差与预期
   脱落率，写出 `power` 块；消融设计还要声明 `planned_contrasts`
   （至少含 `interaction`），并查看 `contrast_power`、
   `power_sensitivity`、实际功效与 MDE；
2. 用 `--plan-design` 检查运行清单、随机化/平衡顺序、位置平衡、
   对比类型功效（`binding_contrast` 通常是交互）与 token 预算；
   调整种子数或最小效应后再冻结；
3. 把 `design_plan.json`、预注册文件、`git archive` 与数据清单
   放入同一只读目录；
4. 正式运行时使用同一 `--run-order-seed`；子进程核对计划；
5. 分析时同时报告 `power`、`seed_plan`、`order_balance` 与
   `run_order`；顺序相关强时先调查漂移，而不是直接解释臂间
   差异；交互不显著时先看它的 MDE/功效，不能把「按两两对比
   够用的种子数」当成「交互也有足够功效」；
6. 增加种子/重跑必须作为新研究阶段重新预注册，不能反复换种子
   直到显著。因数值失败而剔除运行也必须按预注册的脱落/失败率
   口径报告，不能只保留成功运行。

## 6. 与现有协议的关系

- `docs/experiment_protocol.md` 的冻结 → 预注册 → 运行 → 统一
  解封流程不变；本文件补上「预注册前如何判断设计可执行」；
- `docs/methodology_audit.md` 第 29 节记录本轮修正；
- 设计计划不替代 `artifact_manifest.json`：前者是训练前设计，
  后者是训练后归档内容完整性；两者都应随最终报告保存。
