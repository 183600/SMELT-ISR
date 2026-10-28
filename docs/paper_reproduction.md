# 论文复现口径（arXiv:2609.01343v1）

本文件说明本仓库如何复现 *SMELT: Scaling Laws for Compute-Matched MoE Looped
Transformers*（arXiv:2609.01343v1），以及哪些量**不能**用公开实现逐位复现。
代码入口：

- 网格与预设：`exp/part_000_config.py` 的 `PAPER_CELLS` / `paper_preset()`（经 `experiment.py` 兼容入口重新导出）；
- 命令行：`--paper-cell`、`--paper-arch`、`--paper-protocol`；
- 审计：`exp/part_001_paper.py` 的 `paper_fidelity_report()`（写入 `config.json` 的
  `paper_fidelity`，每次运行开始时也在标准输出打印摘要）。

## 1. 论文披露的网格（附录 A）

Baseline（未循环 MoE，`--paper-arch base`）：

| Scale | H | L | Total experts | Active experts | Active params | S | Total params |
|---|---|---|---|---|---|---|---|
| 100M | 672 | 10 | 8 / 64 / 192 / 336 | 8 | 0.100 / 0.100 / 0.101 / 0.102B | 0 / 86.0 / 95.2 / 97.2% | 0.10 / 0.71 / 2.12 / 3.70B |
| 200M | 1280 | 12 | 8 / 64 / 192 / 336 | 8 | 0.207 / 0.207 / 0.209 / 0.212B | 0 / 84.3 / 94.6 / 96.9% | 0.21 / 1.32 / 3.87 / 6.74B |
| 600M | 1792 | 20 | 8 / 64 / 192 / 336 | 8 | 0.663 / 0.665 / 0.670 / 0.675B | 0 / 84.5 / 94.7 / 96.9% | 0.66 / 4.28 / 12.54 / 21.83B |
| 1.6B | 2304 | 30 | 8 / 64 / 192 / 336 | 8 | 1.629 / 1.632 / 1.641 / 1.651B | 0 / 84.5 / 94.7 / 96.9% | 1.63 / 10.55 / 30.94 / 53.89B |

SMELT（middle 50% 循环两次，`--paper-arch smelt`）：

| Scale | H | L | Total experts | Active experts | Active params | S | Total params | F ratio | KV ratio |
|---|---|---|---|---|---|---|---|---|---|
| 100M | 576 | 10 | 16 / 96 / 288 / 504 | 8 | 0.071 / 0.067 / 0.069 / 0.070B | 0 / 86.8 / 95.5 / 97.4% | 0.10 / 0.71 / 2.12 / 3.70B | 1.081 / 1.024 / 1.031 / 1.039 | 1.018 / 0.964 / 0.964 / 0.964 |
| 200M | 1056 | 12 | 16 / 96 / 288 / 504 | 8 | 0.142 / 0.140 / 0.142 / 0.145B | 0 / 84.6 / 94.7 / 96.9% | 0.21 / 1.32 / 3.89 / 6.78B | 1.039 / 1.021 / 1.029 / 1.037 | 1.031 ×4 |
| 600M | 1408 (S=0,85) / 1512 (S=95,97) | 20 | 16 / 96 / 288 / 504 | 8 | 0.458 / 0.443 / 0.453 / 0.459B | 0 / 85.1 / 94.7 / 96.9% | 0.67 / 4.31 / 12.44 / 21.70B | 1.046 / 1.007 / 1.024 / 1.031 | 1.031 / 1.031 / 0.984 / 0.984 |
| 1.6B | 1848 | 30 | 16 / 96 / 288 / 504 | 8 | 1.101 / 1.081 / 1.091 / 1.103B | 0 / 84.6 / 94.8 / 97.0% | 1.64 / 10.45 / 30.90 / 53.90B | 1.022 / 1.000 / 1.004 / 1.008 | 0.984 ×4 |

每行四列依次对应 S ≈ 0 / 85 / 95 / 97%。

论文 3.1 节给出的其余协议量：

- 优化器 AdamW，warmup-stable-decay 调度；
- 稳定期 196,075 步，全局 batch 256 条序列（256 × 4096 ≈ 1.05M token/步），
  稳定期约 205B token；
- 从稳定期第 10k/20k/50k/100k/150k/196,075 步分叉出六条 cosine-decay
  分支，每条再训练 10B token（≈ 9,537 步）；
- 训练序列打包成 4096-token context，文档级 segment attention mask；
- 验证集覆盖 39 个来源；DCLM Core 22 任务、MMLU 5-shot。

## 2. 论文未披露的量与推导规则

论文附录 A 的实现来自未公开的 Transformer family，以下量没有公开值，
本仓库用固定规则推导并记入 `paper_fidelity` / `config.json`：

| 量 | 规则 |
|---|---|
| attention head 维度 | 在 H 的整除划分中取 head_dim 最接近 64 的组合（32–128） |
| Baseline 的 KV 头数 | Q 头数的一半（2:1 GQA） |
| SMELT 的 KV 头数 | 使 `kv_dim / kv_dim_baseline` 尽量等于论文报告的 KV ratio；不要求整除（非整除时 Q 头 i 使用 KV 头 `floor(i·n_kv/n_heads)`） |
| 专家中间维 | 由论文的**总非 embedding 参数**目标反解，按 8 对齐 |
| 学习率 / warmup / weight decay / betas / 均衡损失系数 / 初始化 | 本仓库默认值（`paper_fidelity.undisclosed`） |
| 预训练语料 | 用户提供的字节级语料（论文为内部语料） |

因此 `paper_preset()` 的构造保证：H、L、专家总数、top-8、循环层位置、
4096 上下文、GQA、KV cache 比例与总参数预算（<1%）对齐论文；
**FLOPs 比例、active params 与 S 是推导后的实测结果**，与论文目标的偏差
逐项写入 `budget_audit`，不隐藏。

## 3. 运行方式

### 3.1 稳定期运行（32 个网格单元任选其一）

```bash
python experiment.py --data corpus.txt --out runs/paper_smelt_200m_s95 \
  --paper-cell 200M:95 --paper-arch smelt \
  --paper-protocol --group-separator '\n\n' \
  --hypothesis "论文 SMELT 200M S≈95% 单元的验证损失" --rq RQ1
```

`--paper-protocol` 会设置 `--lr-schedule wsd`、`--batch-size 256`、
`--steps 196075`、`--seq-len 4096`、`--document-masks`，并在六个分支点
保存 `step_<N>.pt`。论文未披露的学习率/warmup/weight decay 仍取命令行
或仓库默认值；如需与论文口径保持一致，必须在预注册中显式固定。

### 3.2 从分支点做 cosine-decay 分支

```bash
python experiment.py --data corpus.txt --out runs/paper_smelt_200m_s95_b20k \
  --resume runs/paper_smelt_200m_s95/step_20000.pt \
  --steps 29537 --wsd-stable-steps 20000 \
  --lr-schedule wsd --wsd-final-lr-ratio 0.0 \
  --hypothesis "20k 分支的 10B-token cosine decay" --rq RQ1
```

`--wsd-stable-steps` 是绝对步数；`--steps` 是分支终点（分支点 + 9,537）。
显式给出的 `--wsd-*` / `--lr-schedule` 在 resume 时优先于检查点记录值，
否则 resume 会恢复原稳定期参数（普通续跑语义）。

### 3.3 不带预设的小规模实验

默认 `B`/`A`/`local_base` 等延迟合并实验继续使用原型规模；它们的
`paper_fidelity` 会显示 `architecture_faithful=false` 与偏离项。
这不会影响延迟合并实验本身，但**不能**把这部分结果表述为论文复现。

## 4. 审计字段

`config.json → paper_fidelity`：

```json
{
  "paper": "arXiv:2609.01343v1",
  "cell": {"arch": "smelt", "scale": "200M", "sparsity": 95, ...},
  "preset_geometry": {"n_heads": 16, "n_kv_heads": 10, "expert_hidden": 352},
  "checks": {
    "paper_cell_config": true,
    "top_k_8": true,
    "gqa": true,
    "context_4096": true,
    "loop_span_middle_half": true,
    "loop_placement_paper": true,
    "loop_twice": true,
    "loop_residual_scaling_1_over_r": true,
    "preset_geometry_matches": true
  },
  "protocol_checks": {
    "protocol_wsd_schedule": true,
    "protocol_batch_256": true,
    "protocol_stable_phase": true,
    "protocol_decay_branch": true,
    "protocol_document_masks": true,
    "protocol_branch_checkpoints": true
  },
  "architecture_faithful": true,
  "protocol_faithful": true,
  "is_full_reproduction": true,
  "budget_audit": {
    "measured": {"flops_ratio_vs_base": 1.115, "kv_ratio_vs_base": 1.031,
                  "compute_equivalent_sparsity": 0.938, ...},
    "paper":    {"flops_ratio": 1.029, "kv_ratio": 1.031,
                  "compute_equivalent_sparsity": 0.947, ...},
    "within_paper_tolerance": {"total_params_within_1pct": true,
                               "kv_ratio_within_4pct": true,
                               "flops_ratio_within_4pct": false, ...}
  },
  "undisclosed": ["learning_rate", "warmup_steps", ...]
}
```

判定语义：

- `architecture_faithful`：附录 A 的结构（H/L/专家数/top-8）、循环层位置、
  1/r、GQA、4096 上下文、推导几何与预设一致；
- `protocol_faithful`：3.1 节协议（WSD、batch 256、稳定期/分支步数、
  文档级 mask、六个分支点）齐备；
- `is_full_reproduction` = 两者同时成立（recipe 复现）；
- `numerical_reproduction` = `is_full_reproduction` 且 `budget_audit`
  的 FLOPs/参数/KV/S 全部落在论文容差内（数值复现）；
- `budget_audit` 独立报告预算偏差。**公开实现不能保证与论文逐位相同的
  参数化**：FLOPs 依赖论文未公开的平均文档长度与专家中间维，本仓库按
  4096 上下文与总参数目标推导。预算偏差存在时，结论应写成
  「recipe 复现 + 预算偏差 X%」，不能写成「数值复现」；此时
  `is_full_reproduction=true` 而 `numerical_reproduction=false`。

## 5. 仍未覆盖的部分

- 增量 KV 解码（当前 `kv_cache_bytes_per_token` 只是容量估计）；
- DCLM Core / DCLM Completion / MMLU 的官方评测管线（需要 22 个任务与
  外部基准数据）；
- 39 个验证来源的内部语料与 205B token 的算力；
- 论文的 scaling-law 拟合（Eq. 2）、CE Gain 与下游残差分析需要 96 个
  matched pair × 6 个 token horizon 的完整网格；本仓库提供的是可复现的
  单格训练与协议，不包含论文的拟合脚本。

这些限制不影响「单格 recipe 复现」的判定，但意味着论文的 scaling-law
与下游结论不能仅凭本仓库的运行结果直接宣称复现。

## 6. 基线冻结

冻结由仓库变量 `EXPERIMENT_FREEZE=true` 或
`.github/EXPERIMENT_FREEZE` / 根目录 `EXPERIMENT_FREEZE` 文件触发
（两者任一即可）：

- `Continuous Loop` 只保留排队、清理与归档步骤，不再运行 opencode
  agent，因此不会再自动修改结构参数或提交实验结果；
- 进入冻结时应记录 `git rev-parse HEAD`、`sha256sum experiment.py`
  与 `git status --porcelain`（必须为空），并把冻结提交写入预注册
  的 `frozen_commit`；
- 解除冻结需删除上述文件并把仓库变量设为 `false`（或删除变量）；
- 恢复自动循环前应先确认 `paper_fidelity` 审计仍覆盖新增改动，
  否则基线会再次漂移出论文口径。

当前仓库内没有 `.github/EXPERIMENT_FREEZE`：论文复现的冻结已在
提交 `918efd4` 解除；仓库变量的当前值需在 GitHub 仓库设置中核对，
不能只看仓库文件。冻结不是「锁死」：手工提交与运行始终可用，
冻结只停用无人值守的自动改写。
