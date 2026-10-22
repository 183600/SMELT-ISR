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

在训练前写入 `preregistration.json`：

```json
{
  "study_id": "smelt-isr-confirm-001",
  "frozen_commit": "<git sha>",
  "primary_metric": "val_loss_at_common_step",
  "common_step": 20000,
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
  "data": {
    "sha256": "<corpus sha256>",
    "split_policy": "contiguous",
    "entity_manifest": null
  }
}
```

要求：

- **至少 6 个种子**：配对 t 与精确符号翻转同时显著才有定义；
  n=3/4/5 时符号翻转的最小双侧 p 分别为 0.25/0.125/0.0625，
  数学上不可能达到 0.05。家族校正后按 `ceil(log2(2m/α))` 增加。
- `common_step` 必须训练前固定；不要用默认的「各运行最后验证步
  最小值」（该步由观察到的端点决定）。
- `--budget-match` 必须通过；报告完整计算图的解析与实测 FLOPs
  比例，而不是只匹配 `k1·k2`。
- 对照必须包含：`local_base_k2`（合并时机机制对照）、
  `local_base`（等专家计算量性能对照）、`B_early`（B 的等计算量
  合并位置对照）。

## 3. 运行

```bash
export EXPERIMENT_FREEZE=true
python experiment.py --data corpus.txt --out runs/confirm \
  --seeds 42 43 44 45 46 47 \
  --suite-modes smelt local_base local_base_k2 A B B_early \
  --steps 20000 --common-step 20000 \
  --budget-match smelt \
  --hypothesis "延迟合并在完整计算图匹配下改善验证损失" \
  --rq RQ1 --primary-metric val_loss_at_common_step \
  --hypothesis-direction decrease --hypothesis-min-effect 0.01 \
  --hypothesis-baseline smelt
```

运行期间：

- 不根据验证结果改代码、改种子或改主指标；早停只作为**固定预算
  之外**的辅助结果报告。
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

仓库内只保留代码与文档；结果放外部归档，避免仓库体积与
「清理脚本删除审计记录」的冲突。
