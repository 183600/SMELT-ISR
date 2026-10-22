# 确证实验协议（冻结 → 预注册 → 运行 → 统一解封 → 归档）

本协议把「探索性运行」与「确证性结论」分开。**确证实验一旦开始，
代码、数据切分与主检验不得再改**；任何改动都意味着新研究阶段，
必须重新冻结并重新预注册。

## 1. 冻结

1. 在仓库根目录创建 `EXPERIMENT_FREEZE`（或把仓库变量
   `EXPERIMENT_FREEZE` 设为 `true`）。持续集成循环会跳过 agent
   阶段，不再修改代码或产生新提交。
2. 记录冻结信息：

   ```bash
   git rev-parse HEAD
   sha256sum experiment.py
   git status --porcelain   # 必须为空
   ```

3. 把 `git archive HEAD`、`requirements-ci.txt`、数据清单（SHA-256、
   切分边界、`split_sha256`）与预注册文件放到同一个只读目录；
   正式实验结束后上传到 Zenodo/OSF 等不可变存储并记录 DOI。

## 2. 预注册

在训练前写入 `preregistration.json`，并**先提交该文件**（预注册
入口会拒绝脏工作区）：

```json
{
  "study_id": "smelt-isr-confirm-001",
  "frozen_commit": "<git sha>",
  "frozen_script_sha256": "<可选：experiment.py 的 SHA-256>",
  "primary_metric": "val_loss_at_common_step",
  "common_step": 20000,
  "steps": 20000,
  "min_effect": 0.01,
  "direction": "decrease",
  "baseline_mode": "smelt",
  "comparisons": [
    ["smelt", "A"],
    ["smelt", "B"],
    ["A", "B"],
    ["local_base_k2", "A"],
    ["B_early", "B"]
  ],
  "seeds": [42, 43, 44, 45, 46, 47],
  "budget_match": "smelt",
  "equivalence_margin": 0.01,
  "run_order": {
    "scheme": "randomized",
    "seed": 20261022
  },
  "data": {
    "sha256": "<corpus sha256>",
    "split_policy": "contiguous",
    "val_fraction": 0.05,
    "test_fraction": 0.05,
    "group_separator": "\n\n",
    "cleaning": {
      "deduplicate_train_groups": false,
      "drop_train_shared_chunks": false,
      "drop_train_corrupt_groups": false,
      "drop_train_corrupt_bytes": false,
      "drop_train_degenerate_groups": false,
      "drop_train_degenerate_runs": false,
      "clean_train_against_test": false,
      "allow_eval_content_for_cleaning": false
    }
  }
}
```

字段要求：

- `frozen_commit` 必须是 7~64 位十六进制 git commit。当前
  `HEAD` 可以与它相同，也可以是它的后代（通常是在记录
  `frozen_commit` 之后又提交了预注册文件本身）；但
  `git diff frozen_commit..HEAD -- experiment.py` 必须为空——
  冻结后改代码必须开始新研究阶段。工作区有未提交修改、或
  提供了 `frozen_script_sha256` 而脚本内容不同，同样拒绝
  启动。没有 git 时必须提供 `frozen_script_sha256`。
- `primary_metric` 必须是评估管线真正计算的指标；`common_step`
  必须是正整数且不超过 `steps`；`min_effect` 是相对
  `baseline_mode` 均值的比例；`direction` 为 `decrease` 或
  `increase`。
- `comparisons` 声明的对照臂必须覆盖本次运行实际展开的模式
  （单次 `--mode`，或 `--suite-modes` / `--suite-ablation`
  计划中的全部臂）；`baseline_mode` 必须出现在至少一个对比中；
  重复/自比较会被拒绝。
- `seeds` 至少 3 个且不重复。3~5 个种子只能做 t 检验；
  「配对 t + 符号翻转同时显著」的稳健主张至少需要 6 个，
  家族校正后按 `ceil(log2(2m/α))` 增加。显式 `--seeds` 必须
  与文件集合完全一致；显式 `--seed`、`--resume` 或最终
  `--evaluate-test` 使用其中一个；两者都未给出时自动按文件
  种子集合运行完整套件。
- `power`（可选）是先验功效分析：`expected_seed_sd_relative`
  （预期种子间标准差，相对基线均值）、`target_power`、`alpha`
  与可选 `planned_comparisons`（默认等于 `comparisons` 数量，
  不得更小）。按 Bonferroni 下界 `alpha/m` 与稳健符号翻转
  要求计算所需种子数；当前 `seeds` 不足时默认拒绝启动。确需
  在功效不足的设计上探索时，必须写
  `underpowered_acknowledged: true` 并给出非空
  `justification`，承认会随 `config.json` 的
  `preregistration.power` 落盘。详见
  [design_preparation.md](design_preparation.md)。
- `run_order`（可选）冻结执行顺序：`scheme` 为 `grouped` /
  `blocked` / `randomized`，`seed` 为确定性洗牌种子。推荐
  `randomized`（随机化完整区组）：每个种子区组内所有臂都跑
  一次、臂顺序随机化，避免机器漂移与臂效应混杂。命令行
  `--run-order` / `--run-order-seed` 显式冲突时拒绝启动。
- `budget_match` 为 `null` 或模式名；非 null 时等价于
  `--budget-match` 的完整计算图硬门。它按**本次运行的候选
  模式**解释：同一 study 若既包含 `base`（循环次数不同，
  不能套 `smelt` 门）又包含需要预算门的 `A`/`B`，应拆成两个
  冻结阶段；`comparisons` 含 `A`/`B`/`local_base` 与 `smelt`
  的对照时不允许 `null`。
- `data` 必填 `sha256`、`split_policy`、`val_fraction`、
  `test_fraction`；可选的 `group_separator`、`cleaning`、
  `entity_manifest_sha256`、`time_order_manifest_sha256`
  也会在数据加载后逐项核对。清洗开关会改变实际参与训练/
  去重的字节，必须在预注册中固定。
- 顶层与 `data` 都拒绝未知字段：拼写错误（如把 `seeds`
  写成 `seed`）会让协议静默失效，不能当作「多写了一个
  备注」放行；要写备注请用 `notes` / `metadata`。
- 延迟合并与 `smelt` 的预算对照必须包含
  `local_base_k2`（只改合并时机）、`local_base`（等专家
  计算量）与 `B_early`（B 的等计算量合并位置）；比较
  `A`/`B` 与 `smelt` 时 `budget_match` 不能为 `null`。
  确需研究其它问题时，在 `notes` 中写明并单独预注册，
  不能靠删掉对照让预算/归因检查静默失效。

### 2.1 设计预检（训练前，不烧算力）

若在预注册中声明先验功效，示例（8 个种子、5 个比较，
`expected_seed_sd_relative=0.005`、`min_effect=0.01` 时
`required_seeds=8`，恰好充足；实际数值必须由试点/先前运行
估计，不能照抄）：

```json
"seeds": [42, 43, 44, 45, 46, 47, 48, 49],
"power": {
  "expected_seed_sd_relative": 0.005,
  "target_power": 0.8,
  "alpha": 0.05,
  "planned_comparisons": 5
}
```

写好预注册文件后，先用 `--plan-design` 检查设计是否可执行：

```bash
python experiment.py --data corpus.txt --out runs/confirm-design \
  --preregistration preregistration.json \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --run-order randomized --run-order-seed 20261022 \
  --plan-design --hypothesis "延迟合并在完整计算图匹配下改善验证损失"
```

它不加载数据、不训练，只写出 `<out>/design_plan.json` 并打印
运行清单、随机化区组顺序、先验功效与解析 token 预算。确认后
再冻结代码并正式运行；正式运行会用同一 `--run-order-seed`
重建同一顺序，子进程逐项核对计划指纹与自己的分配。详见
[design_preparation.md](design_preparation.md)。

运行入口：

```bash
python experiment.py --data corpus.txt --out runs/confirm \
  --preregistration preregistration.json \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --hypothesis "延迟合并在完整计算图匹配下改善验证损失" \
  --rq RQ1
```

- 文件会补齐未显式给出的 `--primary-metric` /
  `--hypothesis-direction` / `--hypothesis-min-effect` /
  `--hypothesis-baseline` / `--common-step` / `--budget-match` /
  `--seeds` / `--steps`；命令行显式值若与文件不同，直接拒绝，
  不存在「以命令行覆盖预注册」的路径。
- 数据加载后会核对语料/切分/清洗指纹；不一致时拒绝训练。
  核对结果（文件 SHA-256、study_id、冻结提交、补齐字段、
  实际数据指纹、运行模式/种子）写入 `config.json` 的
  `preregistration` 字段，供归档审计与人工复核。
- 多种子套件结束后，父进程按每个子运行的
  `preregistration.file_sha256` 反查是否仍是同一份文件；
  中途编辑预注册会导致整批被拒绝汇总，而不是把不同协议的
  种子拼成一项研究。
- `--compare` / `--sanity-only` / `--statistical-baseline` /
  `--suite-baseline-tuning` 不接受 `--preregistration`：它们
  不执行同一套训练协议。确证实验若需要新阶段（改代码、改
  数据、改对照），必须重新冻结、写新文件；不能编辑旧文件
  后继续。

## 3. 运行

```bash
export EXPERIMENT_FREEZE=true
python experiment.py --data corpus.txt --out runs/confirm \
  --preregistration preregistration.json \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --hypothesis "延迟合并在完整计算图匹配下改善验证损失" \
  --rq RQ1
```

不加 `--preregistration` 而手工给 `--seeds/--common-step/
--budget-match/--primary-metric/...` 的运行仍可执行，但只能作为
探索性运行：没有独立文件证明这些口径在训练前已固定。

运行期间：

- 不根据验证结果改代码、改种子或改主指标；早停只作为**固定预算
  之外**的辅助结果报告。
- 套件父进程在第一个子进程启动前写 `design_plan.json`；子进程
  核对计划文件 SHA-256、`plan_id`、顺序/区组与自己的 label/seed/
  mode 后才训练。实际开始/结束时间写 `run_order.json`。
- 每臂保留失败目录；`failure_rate` 与成功条件指标一起报告。
- 每臂至少一次 `--routing-interventions`（可选但建议），用于
  RQ4 的功能证据。

## 4. 分析

- 主检验：同种子配对 t（双侧）+ 对称性假设下的符号翻转；两者
  都做 Holm 校正。报告效应量、95% CI、MDE、门槛检验与 TOST。
- 不显著 ≠ 等价；只有 TOST 校正后通过才能说「无实质差异」。
- 共同步数用预注册的 `--common-step`；报告
  `target_step_rule` 与 `target_step_data_dependent`。
- 报告 `failure_rates`、`pseudo_replication`、差分缺失、
  `scale_guard` 与 `budget_match_report`。
- 只读对比用 `--compare ... --compare-report runs/analysis.json`
  把完整分析落盘；默认的 `--compare` 只打印，终端输出不是
  可归档的分析记录。归档时把该 JSON 与输入运行目录一起保存。
- 报告 `fragility_summary`：每个显著对比都要回答「删除
  1~3 个种子后显著性是否仍在」。`fragile` / `direction_mixed`
  的对比不能写成稳健结论，应增加独立种子或调查异常运行后
  重跑；n=3 的设计删除后不足最小种子数，标为
  `not_evaluable`，不能当作「删除后仍显著」。
- 报告 `run_order`：计划指纹是否一致、是否仍用 `grouped`
  顺序、去模式均值后指标与执行顺序的 Spearman 相关。强顺序
  相关先检查机器负载/热漂移并重跑受影响区组，不能直接把臂间
  差异解释为方法效应。该诊断是描述性的，不替代随机化设计。
- 路由结论必须带机会水平（`jaccard_chance_*`、`lift`）与干预
  结果，不能只报负载熵/死专家数。

## 5. 测试集：冻结候选后统一解封

1. 所有候选运行完成并通过 val 分析后，冻结候选清单
   （run 目录、config/metrics 的 SHA-256、代码 commit）。
2. 指定**所有候选共享**的访问账本：

   ```bash
   python experiment.py --data corpus.txt --mode A \
     --out runs/final/A --evaluate-test \
     --test-ledger runs/final/test_access_ledger.jsonl \
     --study-id smelt-isr-confirm-001
   ```

   账本按 `(data_sha256, test split sha256, test_fraction)` 识别
   同一测试集；默认拒绝第二个候选重复解封。确需重复时显式加
   `--allow-repeat-test-access`，并在最终报告中列出全部访问。

3. 测试集只用于最终确认，不参与选型、调参或候选筛选。用
   `--final-test-comparison` 列出**全部**冻结候选的 test 记录
   （包括缺失 test 的候选），不要只报告最好的那个：

   ```bash
   python experiment.py --final-test-comparison \
     runs/final/smelt_seed42 runs/final/A_seed42 ... \
     --test-ledger runs/final/test_access_ledger.jsonl
   ```

## 6. 归档

必须保存（不可变存储）：

- `git archive` 的源码包 + commit SHA + `script_sha256`；
- 全部运行的 `config.json`、`metrics.jsonl`、`best.pt`（如允许）、
  `comparison_analysis.json`、`routing*.json`、失败运行目录；
- 预注册文件、最终报告、分析脚本、依赖版本与环境信息；
- 测试集访问账本与归档校验和（SHA-256）。

所有训练/评估进程结束后，为每个最终运行目录生成可机读清单：

```bash
python experiment.py --write-manifest runs/final/A
python experiment.py --verify-manifest runs/final/A --verify-manifest-strict
```

`artifact_manifest.json` 逐文件记录相对路径、字节数、SHA-256
与符号链接目标，并从 `config.json` 摘出 `script_sha256`、
`git_commit`、数据/切分指纹、预注册文件指纹与
`comparison_protocol_sha256`。校验器拒绝绝对路径、`..` 与
非规范相对路径；默认把清单外新增文件列为 `untracked`（可能
是后续日志），`--verify-manifest-strict` 下才算失败。清单是
写入时刻的快照：`--resume`、补跑 `--evaluate-test` 或追加
分析报告后必须重新生成；上传/解压到归档目的地后再校验一次，
才能证明传输过程没有改动内容。`--write-manifest` 会打印清单
自身的 SHA-256，`--verify-manifest` 也会在报告中返回
`manifest_sha256`；把该指纹记入外部归档记录/DOI 元数据，
才能把「清单未被替换」也纳入审计链。

仓库内只保留代码与文档；结果放外部归档，避免仓库体积与
「清理脚本删除审计记录」的冲突。
