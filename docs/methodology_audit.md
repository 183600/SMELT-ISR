# 方法学审计与修正记录（对应 23 项问题）

本文件逐项记录外部审计提出的 23 个问题在**当前代码**中的判定与处理。
判定只依据源码与可运行行为，不依据本仓库此前的文档承诺。

> 结论先行：本轮修正补上了 A/B 的机制对照、完整计算图预算核对、
> 循环区域 1/r 残差缩放、GQA/KV 估计、伪重复处理、种子功效下限、
> 最小效应门槛检验、失败率报告、固定共同步数、块 bootstrap、
> 数据切分口径声明、测试内容参与清洗的披露、测试集访问账本与
> 工作流冻结开关。后续复现工作中已经补上论文 3.1 节的 4096 packed
> context、文档级 segment attention mask、WSD 调度与
> `--paper-cell` 网格预设（见 [docs/paper_reproduction.md](paper_reproduction.md)）；
> 仍未实现的是增量 KV 解码、DCLM/MMLU 官方评测管线与 39 来源内部语料。
> 预算偏差（FLOPs/active params/S）如实写入 `paper_fidelity.budget_audit`，
> 不能靠措辞绕过。

## 1. 基线是否忠实复现 arXiv:2609.01343v1

**判定：原实现是 SMELT 风格简化模型，不能直接称为论文复现。**
原代码缺少论文 3.1 节明确要求的「循环区域内每个子层残差更新按
1/r 缩放」，也没有 GQA、未循环 MoE 基线与 KV cache 估计。

**已修正：**

- 新增 `residual_scaling=paper`（默认）：循环区域内子层残差按
  1/r（r=2 → 1/2）缩放，循环区域外仍用 `residual_scale`；
  `global` 保留历史行为。该口径写入 `comparison_protocol`。
- 新增 `base` 模式：论文的**未循环 MoE 基线**（middle 只执行一次）。
  此前仓库只有 `smelt`（循环两次），无法直接做论文的
  Baseline/SMELT 对照。
- 新增 GQA：`--n-kv-heads`（默认等于 `n_heads`，即 MHA）。
  效率报告新增 `kv_cache_bytes_per_token` 与 per-visit 口径。
- `paper_fidelity_report` 把「middle 50% 循环两次、1/r、top-8、
  GQA、4096 上下文」逐项核对并写入 `config.json` 的
  `paper_fidelity`；存在偏离时明确标记为 SMELT 风格架构。
- `--budget-match MODE` 按完整计算图（Attention 执行次数 + MoE
  assignment + Router）核对每 token FLOPs，不匹配时拒绝启动。
- 默认仍是原型规模（top-2/8 专家、128 上下文），因为延迟合并实验
  需要单机可运行；论文规模的运行必须显式使用 `--paper-cell`，否则
  `paper_fidelity.architecture_faithful=false`。
- 已补上：`--paper-cell` 的 32 格预设（附录 A 的 H/L/专家/top-8）、
  4096 packed context 的文档级 segment attention mask、WSD 调度与
  六个分支点；这些由 `paper_fidelity.protocol_checks` 逐项核对。
- 仍有限制：增量 KV 解码与 DCLM/MMLU 官方评测管线未实现。

复现论文口径的最小命令（推荐预设；需要相应算力）：

```bash
python experiment.py --data corpus.txt --out runs/paper_base_100m_s85 \
  --paper-cell 100M:85 --paper-arch base --paper-protocol \
  --group-separator '\n\n' --hypothesis "论文 Baseline 100M S≈85%" --rq RQ1

python experiment.py --data corpus.txt --out runs/paper_smelt_100m_s85 \
  --paper-cell 100M:85 --paper-arch smelt --paper-protocol \
  --group-separator '\n\n' --hypothesis "论文 SMELT 100M S≈85%" --rq RQ1
```

手工等价骨架（供核对推导量；不推荐，容易漏掉 KV/GQA/文档 mask）：

```bash
python experiment.py --data corpus.txt --mode base --top-k 8 \
  --n-experts 64 --d-model 672 --n-layers 10 --seq-len 4096 \
  --out runs/paper_base_100m_s85

python experiment.py --data corpus.txt --mode smelt --top-k 8 \
  --n-experts 96 --d-model 576 --n-layers 10 --seq-len 4096 \
  --n-kv-heads 2 --residual-scaling paper --out runs/paper_smelt_100m_s85
```

## 2. A/B 的归因对照

**判定：原 `local_base` 同时改变了合并时机、第二轮宽度、专家复用
与专家输入；B 还改变了分支数、Attention/MoE 执行次数。原六格
消融无法把收益归给延迟合并。**

**已修正（新增三个对照）：**

| 模式 | 合并时机 | 第二轮宽度 | Attention/MoE 执行次数 | 回答的问题 |
|---|---|---|---|---|
| `A` | 第二轮后 | 每个分支 k2 | 每层 1 次 Attention，MoE 两阶段 | 延迟合并（局部） |
| `local_base_k2` | 第一轮后立即 | k2（与 A 单分支相同） | 与 A 相同 | **只改变合并时机**（机制对照） |
| `local_base` | 第一轮后立即 | min(E, k1·k2) | 与 A 相同 | **等专家计算量**（性能对照） |
| `B` | 第二轮后 | 每分支 k2 | 分支化 Attention，两轮均按分支 | 全局延迟合并 |
| `B_early` | 第一轮后提前合并，再复制回 k1 分支 | k2 | 与 B **完全相同** | **只改变合并位置**（B 专用等计算量对照） |

`B_early` 的重合并不减少计算：第二轮仍执行 k1 个分支的
Attention/MoE，只是各分支状态在第一轮结束后已经相同。因而
`B` 与 `B_early` 的差异不能再归因于「多算」「多轨迹」或
「更多 Attention」——这些在执行次数上被固定了。

`mode_cost_profile` / `budget_match_report` 把每 token 的
Attention 执行次数、MoE pass、Router 调用、专家 assignment 与
解析 FLOPs 全部落盘；`--budget-match` 可按完整计算图拒绝不匹配
的配置。仅令 `k1·k2≈2·top_k` 会漏掉第一阶段成本，已从文档与
CLI 提示中删除该说法。

## 3. RQ2 的预算约束

原说法 `k1×k2≈2×top_k` 只匹配专家组合数，漏掉了第一阶段
`k1` 次执行。对 A，专家 assignment 是 `k1 + k1·k2`；对 B，还要
加分支创建时的 `k1` 次 assignment，以及第一轮其余 `(m−1)` 个
Block 上每个分支各自路由 `k1` 个专家的 `(m−1)·k1·k1` 次
assignment。**已修正：**
`budget_match_report` 用完整计算图搜索整数 k2（A/local 系列）或
(k1,k2) 网格（B 系列），并报告实测/解析 FLOPs 比例；
`--budget-match smelt` 把预算匹配变成启动硬门。

注意：只匹配专家 assignment 仍会让 A 因 Attention 只执行一次而
比 `smelt` 便宜，因此完整匹配推荐的 k2 不一定等于
`(2·top_k−k1)/k1`。这是本修正要暴露的差异，不是 bug。

## 4. 分支「独立」的定义与检验

**判定：B 的分支是「按第一轮路由排名（slot）组成的序列分支」，
不是某个专家贯穿整条序列的独立轨迹。** 原实现只在前向状态上
隔离，参数、输入来源与梯度都共享。

**已修正：**

- `routing_analysis.branch_semantics` 明确区分前向状态隔离、
  参数共享、梯度耦合与统计独立。
- 自测验证分支 Attention 不跨分支（已有），并新增分支顺序
  置换不变性测试（见 `tests/test_core.py`）。
- 报告 `branch_self_reuse_share` 描述「同一 slot 的第二轮是否
  复用第一轮专家」，但不把它解释成专家轨迹。

仍有限制：没有对「同一专家是否在多个位置形成一致功能」做完整
因果分析；这需要下面的干预实验。

## 5. RQ4 的专家专门化证据

**判定：负载均匀、路由熵高、死专家少只是利用率描述，均匀随机
路由也能得到。原指标不能证明专家专门化。**

**已修正：** 新增 `--routing-interventions`：

1. `routing_shuffle`：保持 assignment 数与均匀权重，把专家身份
   随机化；若专家有可区分功能，损失应上升。
2. `expert_drop`：逐个把某专家输出置零（路由/门控不变），记录
   损失上升；报告 `importance_cv`、最大/最小专家与比值。
3. 与 baseline 同一批固定 val 窗口比较。

该实验在 val 上做 E+2 次前向（默认关闭），结果写入
`routing_interventions.json` 与 `config.json`。它仍不是完整的
「专家替换/路由打乱」因果实验（例如没有跨层专家替换），但已把
「专门化」从利用率指标推进到功能干预证据。

## 6. RQ3 的 Jaccard 机会水平

**已修正：** 每个 Jaccard 同时报告：

- `jaccard_chance_uniform`：均匀随机集合的期望
  `E|A∩B|=k1·k2/E`；
- `jaccard_chance_marginal`：保持第二轮专家边际负载、随机化
  组合的期望重合；
- `jaccard_lift_over_*` 与 `jaccard_adjusted_*`。

低重合本身仍不证明路由更有效；必须结合上面的干预实验。

## 7. 伪重复（第 8 项）

**判定：原实现已经把同一 seed 折叠成一个观测，但文档说「退回
Welch、每个运行一个样本」，与实际不符；折叠时也没有检查复制
运行是否真的逐位一致。**

**已修正：**

- 同一 seed 的多个运行先折叠为**一个独立观测**，再按 seed 配对；
  复制运行不增加 n/df，不再因重复而禁用配对。
- `pseudo_replication.within_seed_sd_max` 记录组内差异；同 seed
  复制运行若给出不同值，会显式暴露，不再静默平均。
- 文档改写为「折叠后配对」，并说明 Welch 不能恢复独立性。

## 8. 种子计划与符号翻转（第 9 项）

**已修正：**

- 新增 `_sign_flip_required_seeds`：精确双侧最小 p=2/2^n，
  n=3/4/5 分别为 0.25/0.125/0.0625，**均不能达到 0.05**；
  n=6 才可能，家族校正后按 `ceil(log2(2m/α))` 增加。
- `seed_plan` 增加 `min_seeds_for_robust_sign_flip`、
  `robust_claim_possible` 与公式说明。
- `significant_robust` 在 n≤5 时标记 `not_evaluable_n_le_5`，
  不再把「设计无法判定」说成「效应不稳健」。
- 文档中的「n<5」改为「n<6」，并把「3~5 个种子」改为
  「3~5 个种子只能做 t 检验；稳健双检验至少 6 个」。

## 9. 符号翻转的假设（第 10 项）

**已修正：** 所有报告与文档把它称为「差值分布关于 0 对称」
假设下的符号翻转检验，不再称为「无分布假设」。结果中记录
`randomization_null_assumption="symmetric_paired_differences_about_zero"`。
同一个 seed 编号只保证配对，不保证方法标签可交换。

## 10. 共同训练步数（第 11 项）

**已修正：**

- 新增 `--common-step STEP`：训练前固定共同步数，派生指标
  `*_at_common_step` 不再由观察到的端点决定。
- 默认（未指定）仍取各运行最后验证步的最小值，但报告记录
  `target_step_rule`、`target_step_data_dependent`，并明确
  「不是无偏，只是 conditionally selection-independent」。
- `selection_bias_note` 改写，删除「不受选型偏差影响」的过强表述。

## 11. 窗口 bootstrap（第 12 项）

**已修正：** 新增 `--eval-bootstrap-block N`：N>1 时用循环移动块
bootstrap 重采样连续窗口，报告 `bootstrap_unit` 与
`bootstrap_block_size`。仍有限制：没有按文档/来源的 cluster
bootstrap，也没有把种子层级并入窗口区间；窗口级比较仍只应作为
辅助证据，主检验是种子级。

## 12. 最小有意义效应（第 13 项）

**已修正：** `hypothesis_alignment` 不再仅凭「点估计超门槛 +
相对 0 显著」判 `supported`。新增：

- 方向调整后的门槛检验 `H0: 改善量 ≤ min_effect_abs`，
  记录 `min_effect_p_value`；
- 方向调整后的 95% 置信下界 `directional_ci95_low`；
- `min_effect_verified`；未通过时状态为
  `min_effect_not_verified` 或 `min_effect_unverifiable`，
  并给出专门 advisory。

## 13. 数值失败与幸存者偏差（第 14 项）

**已修正：** 新增 `failure_rate_report`：

- 按模式报告 `planned_runs`、`failed_runs`、`unknown_runs`、
  `success_runs`、`failure_rate`；
- 臂间失败率差 >10 个百分点时给出 `differential_failure` 警告；
- 明确重试规则：不自动重试；同 seed 重跑成功路径是伪重复，
  失败后换新 seed 或修复数值问题后整批重跑。

失败运行仍不进入均值/检验（否则故障被当成效应），但失败率与
成功条件指标必须成对报告。

## 14. 学习率选型的「无法区分」（第 15 项）

**已修正：** `analyze_baseline_tuning` 的并列判定改用**同种子
配对差值 t 检验**（候选 vs 最优均值），不再用
`max(std_best, std_cand)`；记录 `paired_tie_test`、`selection_rule`
与「不显著 ≠ 等价」的说明。无共同种子时才退回旧启发式并标注。
仍有限制：winner 仍需独立种子确认；1% 等价界不是按指标自动
具有科学意义，必须按任务解释。

## 15. 数据切分口径（第 16 项）

**已修正：**

- 新增 `--split-policy {contiguous,chronological}`。默认
  `contiguous` 明确是「按文件顺序的连续前缀切分」，**不声称
  时间切分**。
- `--split-policy chronological` 必须提供
  `--time-order-manifest`（JSONL 或每行 ISO-8601）；时间戳非
  单调不减时拒绝启动；证据写入 `config.data.time_order_evidence`。
- `comparison_protocol` 记录 `split_policy`。
- 仍有限制：没有实现 document-stratified 与 source-extrapolation
  切分；需要来源清单/实体清单时，当前只能做审计并明确披露。

## 16. 实体隔离与近重复（第 17 项）

**已修正：**

- 新增 `--entity-manifest`（按群组序号的 entity_id 数组）与
  `--allow-entity-overlap`；同一实体跨 split 默认拒绝训练。
- 文档明确：分隔符只保证连续片段完整，不识别非相邻片段的同一
  用户/作者/文档版本；内容定义分块是**精确块**扫描，不是近重复
  检测。
- 仍有限制：没有实现 MinHash/SimHash 近重复检测，也没有
  document-stratified 切分；`--group-separator '\n\n'` 仍可能把
  同一文档拆到不同 split，必须用真实文档/实体 ID。

## 17. 测试内容参与训练集删除（第 18 项）

**判定：这是原实现最严重的协议缺陷。** `--deduplicate-train-groups`
与 `--drop-train-shared-chunks` 会用 test 内容决定删除哪些训练群组，
而文档同时声称「测试集全程不读」。

**已修正：**

- 默认**只用 val** 决定训练集删除；test 内容不参与数据集构建。
- 只有显式 `--allow-eval-content-for-cleaning` 才把 test 纳入，
  并把 `eval_content_access`（含 `test_content_used`）写入
  `config.data` 与报告。
- 文档改写为「test 字节不被修改；默认 test 内容不被读取用于
  清洗，启用开关时明确披露」。

## 18. 「标签噪声」术语（第 19 项）

**判定：非法 UTF-8、NUL、重复字节可能是语料损坏，也可能是任务
允许的合法字节；直接称为标签噪声依据不足。**

**已修正：**

- 启用任一清洗开关必须提供 `--domain-justification`，否则拒绝
  启动；依据写入 `config.data.cleaning_domain_justification`。
- 文档与 CLI 帮助改为「格式/语料完整性清洗」，并说明 NUL 可能
  是合法分隔符、重复字节可能是表格边框/填充。
- 仍有限制：没有自动的 `deletion_bias` 直方图审计；正式的删除
  决策应附 kept/dropped 分布对比。

## 19. 类别不平衡指标（第 20 项）

**已修正：** 删除「ROC-AUC 会因负样本多而升高」「排序判定只能看
PR-AUC」「改用 macro F1/PR-AUC 判定」等错误建议。统一口径：

- 字节语言模型主指标是 NLL / bits-per-byte（预注册口径）；
- 类平衡 `val_macro_nll`、macro F1、PR-AUC lift、ROC-AUC 是
  辅助诊断；
- ROC-AUC 对 prevalence 不敏感，但高 ROC-AUC 不保证低
  prevalence 下有可用的 precision/recall；
- PR-AUC 必须报告相对随机水平（≈平均 prevalence）的 lift。

## 20. 实验规模（第 21 项）

**已修正：** 新增 `scale_guard`（tokens_seen、tokens/param、
epochs、seq_len、scale_tier），在单次运行与对比报告中打印
`[scale]` 警告；默认配置约 1.2 token/param、<1 epoch，明确标为
原型级。仍有限制：没有 `--suite-scale` 多来源/多上下文/多规模
扫描；这些结论必须另行验证。

## 21. 工作流与审计链（第 22 项）

**已修正：**

- 工作流新增 `EXPERIMENT_FREEZE` / `.github/EXPERIMENT_FREEZE`
  冻结开关：冻结时跳过 agent 阶段，不修改代码、不产生新提交。
- `PROMPT_CLEAN` 与收尾 shell 不再删除实验结果、审计记录、
  性能/预算/修复记录或 docs 分析文档；只清理临时文件。
- 新增 `actions/upload-artifact` 归档 runs/config/metrics/
  分析报告/日志（90 天）。
- 仍有限制：GitHub artifact 不是永久不可变归档；正式实验应把
  同提交的 `git archive`、config、失败运行与报告上传到 Zenodo/
  OSF 等外部存储，并记录 DOI。

## 22. 测试集访问流程（第 23 项）

**已修正：**

- 新增 `--test-ledger FILE`：按
  `(data_sha256, test split sha256, test_fraction)` 记录每次解封
  的候选目录、代码指纹、配置指纹、study id 与时间；默认拒绝
  同一测试集被其它候选再次解封（`--allow-repeat-test-access`
  显式豁免并留痕）。
- `--final-test-comparison RUN_DIR...`：只读列出全部冻结候选的
  test 记录、缺失 test 的候选与账本重复访问，不重新评估、不挑
  选候选。
- test 记录新增 `study_id`、`test_access`、`model_selection`、
  `selected_step`、`hypothesis`、`primary_metric`。
- 仍有限制：没有强制「先冻结候选再解封」的单一入口；
  研究级流程仍需人工执行 `docs/experiment_protocol.md` 的步骤。

## 23. 代码层三项

**Top-1 路由梯度：已修复。** `top_k==1` 时不再把门控权重设为
常数 1，而是取完整 softmax 中选中专家的概率（Switch 风格），
并让该权重真正乘到专家输出上；`merge`、B/A 的 `k1==1` 路径
同步修改。自测检查 `balance_coef=0, top_k=1` 时 router 仍从
语言建模损失获得非零梯度。

**负载均衡梯度：已披露，未完全解决。** `balance_coef×k` 只
对齐单次路由的数值尺度，不能保证 B 的第一个中间 Block 多一次
aux 项、`k1≠k2` 的单系数、分支 flatten 后的梯度压力相同。
报告新增 `balance_strength_report` 的说明；完整的分层梯度探针
仍建议在论文级实验中补上。

**完整实现一致性：部分修复/部分限制。**

- 位置编码：仍是可学习绝对位置编码，不是 RoPE；4096 packed
  context 与文档级 attention mask 未实现。
- 分支 Attention：隔离语义正确（自测验证）。
- checkpoint 重算：数值等价（自测）；没有随机算子，因此当前
  无需保存 RNG 状态。
- AMP：仍按硬件自动选择，没有 `--amp-dtype`；跨设备比较时
  `amp_dtype` 已作为受控变量记录。
- 自定义反向：SwiGLU/专家 stack/merge 的实现未发现正确性 bug；
  CUDA 路径未在 CPU CI 覆盖。
- FLOPs：实测包含分支乘法；`--grad-checkpoint` 下训练 FLOPs
  仍按 3× 前向估计，未计重算（已披露为限制）。
- 增量解码：未实现 KV cache；`kv_cache_bytes_per_token` 只是
  容量估计。论文的 KV cache 匹配需要真正的解码路径。

## 24. 预注册的可执行性（本轮补充）

**判定：原实现只有「把判定口径写进 config.json」的预注册，
没有训练前独立冻结、事后只能核对的预注册文件。** 文档要求先写
`preregistration.json`，但入口不读取它：同一命令行可以在看到
数据/结果后补写主指标、方向、最小效应、共同步数、对照臂与
种子集合，再把 `config.json` 当作「预注册」证据——这在审计上
无法与真正的预注册区分。

**已修正：**

- 新增 `--preregistration FILE`：结构校验（必需字段、对照臂、
  种子数、数据类型）、重复键/非有限常量拒绝、冻结提交
  `frozen_commit`（或 `frozen_script_sha256`）核对、脏工作区
  拒绝。
- 文件补齐未显式给出的主指标/方向/最小效应/对照基线/
  `common_step`/`steps`/`--budget-match`/`--seeds`/等价界；
  命令行显式值冲突时拒绝启动，不能「以命令行覆盖预注册」。
- 数据加载后核对 `sha256`、`split_policy`、`val_fraction`/
  `test_fraction`、可选 `group_separator`、清洗开关与实体/
  时间清单指纹；不一致时拒绝训练。
- 核对记录写入 `config.json` 的 `preregistration`（文件
  SHA-256、study_id、冻结提交、补齐字段、实际数据指纹与
  运行模式/种子），resume 时保留；`--compare` 等只读入口
  不接受该参数。

仍有限制：`--preregistration` 只覆盖神经训练/多种子套件，
统计基线仍用单独的 `--primary-metric` 口径；没有实现
「先冻结候选、再统一解封 test」的单一状态机，测试集流程
仍按第 5 节人工执行。

## 推荐的确证流程（优先级）

1. 冻结代码：设置 `EXPERIMENT_FREEZE=true`；记录 git commit /
   `script_sha256`；把 `docs/experiment_protocol.md` 与预注册
   假设一起归档。
2. 固定数据协议：`--split-policy chronological` + 时间清单，
   或明确使用 `contiguous` 且不声称时间外推；实体清单审计；
   默认 val-only 清洗。
3. 运行机制对照：`smelt base local_base local_base_k2 A B B_early`，
   **至少 6 个种子**，固定 `--common-step` 与训练预算；
   用 `--budget-match` 报告完整计算图比例。
4. 统计：主检验配对 t + 符号翻转（明确对称性假设）；报告
   失败率、MDE、门槛检验、TOST；不把不显著读成等价。
5. 冻结候选后统一解封 test：指定共享 `--test-ledger`，
   每个候选只解封一次；用 `--final-test-comparison` 报告全部
   候选，而不是只报告最好的那个。
6. 外部归档：把 `git archive`、全部 config/metrics（含失败）、
   分析脚本与最终报告上传到不可变存储，记录 DOI/校验和。
