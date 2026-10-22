# 方法学审计与修正记录（对应 23 项问题）

本文件逐项记录外部审计提出的 23 个问题在**当前代码**中的判定与处理。
判定只依据源码与可运行行为，不依据本仓库此前的文档承诺。

> 结论先行：本轮修正补上了 A/B 的机制对照、完整计算图预算核对、
> 循环区域 1/r 残差缩放、GQA/KV 估计、伪重复处理、种子功效下限、
> 最小效应门槛检验、失败率报告、固定共同步数、块 bootstrap、
> 数据切分口径声明、测试内容参与清洗的披露、测试集访问账本与
> 工作流冻结开关。本轮另补上结果脆弱性/单种子依赖检查与精确
> 符号检验（见第 26 节），以及同时置信区间与种子×窗口两层
> 不确定性/方差分解（见第 28 节）；还补上了按对比类型分层的
> 先验功效（主效应/交互方差不同，交互约需 2 倍种子，见第 30
> 节）。后续复现工作中已经补上论文 3.1 节的 4096 packed
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
bootstrap；但种子层级已并入窗口区间：逐窗口配对的跨种子汇总
另给种子×窗口两层 bootstrap 与方差分解（见第 28 节），
`dominant_source` 区分该补种子还是扩评估集。两层区间仍只
条件于当前固定窗口，窗口级比较仍只应作为辅助证据，主检验是
种子级。

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

## 25. 归档校验的可执行性（本轮补充）

**判定：协议要求归档 SHA-256，但此前只能人工 `sha256sum`。**
人工命令容易漏文件（尤其是失败运行目录、`eval_windows.json`、
账本与后续追加的 `comparison_analysis.json`），也无法回答
「清单内文件被改动」与「清单外新增文件」的区别；归档上传/
解压后的内容完整性没有统一入口复核。

**已修正：**

- 新增 `--write-manifest RUN_DIR`：原子写入
  `artifact_manifest.json`，逐文件记录规范化相对路径、大小、
  SHA-256 与符号链接目标（不跟随链接，避免把目录外内容或
  链接环纳入清单），并记录生成时的 `script_sha256`、git
  状态、Python/torch/NumPy 版本；有 `config.json` 时摘出
  study_id、数据/切分指纹、预注册文件指纹与
  `comparison_protocol_sha256`。
- 新增 `--verify-manifest RUN_DIR`：重新哈希并报告缺失、
  大小/内容不符、类型/链接目标变化；拒绝绝对路径、`..`、
  非规范相对路径与解析后逃逸运行目录的条目。清单外新增
  文件默认只列为 `untracked`，`--verify-manifest-strict`
  下判失败；完整性失败以非零状态码退出，便于归档流水线
  直接使用。
- 清单不包含自身（避免哈希自指）；代码/环境差异单独报告，
  在另一台机器上校验归档时不会被误判成内容损坏。

仍有限制：清单是写入时刻的快照，不提供签名/时间戳服务，
也不能替代不可变存储；续跑、补测或追加分析后必须重新生成，
否则校验会正确报告旧清单已过期。

## 26. 结果脆弱性与单种子依赖（本轮补充）

**判定：原实现报告了显著性、置信区间、符号翻转、leave-one-out
方向稳定性与 TOST，但没有回答「这个显著结论是否只由一两个
随机种子撑起来」。** leave-one-out 只检查删除一个种子后效应
**方向**是否翻转；一个极端差值可以让方向保持不变、却把 p 值
从显著拉回不显著。把「方向稳定」当成「结论稳健」会系统性
高估小样本结果的可靠性。

**已修正：**

- 每个配对对比保存逐种子差值（`paired_differences` /
  `paired_seeds`）与差值分布，结果分析可直接复核哪些种子
  驱动了显著性，而不是只相信汇总 p 值。
- 新增 `_exact_sign_test`：精确二项符号检验，只看差值方向、
  不使用幅度，回答「多数种子是否与均值效应同向」。零差值
  丢弃并记录；最小可达双侧 p=2/2^n，n≤5 时 0.05 水平上
  不可能显著，报告显式标记 `resolution_limited`。
- 新增 `_paired_result_fragility`：逐一/成组删除种子后重做
  配对 t 检验，报告 `max_leave_one_out_p_value`、
  `significance_survives_leave_one_out`、
  `min_seed_deletions_to_lose_significance`（在组合数上限内
  枚举，超限时标 `deletions_tested_exhaustively=false`）与
  `verdict`（`stable` / `fragile` / `direction_mixed` /
  `not_significant` / `not_evaluable`）。删除后剩余观测少于
  最小种子数时判 `not_evaluable`，不伪造「删除后仍显著」。
- 脆弱性比较使用该对比在 Holm 家族中的实际位置阈值
  `α/(m-rank+1)`（`holm_alpha_threshold`），与最终判定同口径；
  删除种子不改变家族规模，沿用原位置阈值是保守近似。
- `comparison_analysis.json` 顶层新增 `fragility_summary`
  （判定分布、fragile 列表、direction_mixed 列表与 advisory）；
  展示层新增 `[fragility]` 小节与 `[fragile]` 点名。全因子
  消融的主效应/交互同样附该分析；组件联合判定遇到 fragile
  的支持效应时降级为 `inconclusive`，不再宣称该组件有用。
- 只读 `--compare` 新增 `--compare-report FILE`：默认仍只
  打印、不写文件；显式给出路径才把完整分析（含
  `fragility_summary`）落盘，避免结果分析只存在于终端输出、
  无法随运行目录归档与复核。

仍有限制：这是描述性敏感性分析，不是新的显著性检验；它
回答「删除种子后显著性是否存活」，不回答「这些种子为什么
异常」。fragile 的正确反应是增加独立种子或调查异常运行，
不是反复换种子直到显著；n=3 的设计无法做删除后仍满足最小
种子数的检验，只能标为不可判定。

## 27. 实验设计与准备（本轮补充）

**判定：原预注册只要求「至少 3 个种子」，但没有训练前的功效
论证；套件执行顺序按模式分组，时间/机器漂移与处理效应完全
混杂。** 3 个种子只保证 t 检验可算，不保证能检出预注册的最小
效应；而把所有 A 的种子跑完再跑 B，会让热漂移、缓存预热与
集群负载变化系统性地落在某个臂上。两类问题都无法在分析阶段
用统计方法补救，必须在准备阶段修。

**已修正：**

- 新增预注册可选 `power` 块：声明预期种子间标准差（相对基线
  均值）、目标功效、alpha 与家族规模，按 Bonferroni 下界
  `alpha/m` 计算配对 t 所需种子数，并同时满足精确符号翻转
  所需 `ceil(log2(2m/alpha))`。种子数不足时默认拒绝启动；
  显式 `underpowered_acknowledged=true` 必须给出 justification，
  承认随 `config.json` 的 `preregistration.power` 落盘。
- 新增 `--plan-design`：不加载数据、不训练，只生成/打印
  `design_plan.json`——运行清单、随机化区组顺序、先验功效与
  解析 token 预算，供冻结前检查。
- 新增 `--run-order {grouped,blocked,randomized}` 与
  `--run-order-seed`，默认 `randomized`：每个种子是完整区组，
  区组内臂顺序确定性随机化；`grouped` 仅保留用于兼容旧运行。
  预注册可用 `run_order` 冻结方案与种子，命令行冲突拒绝启动。
- `design_plan.json` 的 `plan_id` 是去掉时间戳后的规范化指纹；
  子进程通过环境变量收到计划文件、SHA-256、`plan_id` 与
  (order_index, block, position)，开跑前重新哈希并核对
  label/seed/mode/balance。计划被改、指纹不符或分配不符都
  拒绝训练，而不是先跑完再在报告里解释。
- 实际开始/结束时间写入 `<out>/run_order.json`（失败条目也
  保留）；套件结束后父进程核对全部子运行的 `plan_id` 一致。
- `--compare` 与套件汇总新增 `run_order` 诊断：计划指纹是否
  混杂、是否仍用 grouped、去模式均值后指标与执行顺序的
  Spearman 相关与近似 p 值、每个模式内部的顺序相关。强相关
  以 `[warn]` 提示先查机器漂移，不能直接解释臂间差异。

仍有限制：这是**随机化与描述性诊断**，不是把时间作为协变量的
正式区组模型；每个区组内只有一次观测，无法估计区组×处理交互。
`power` 的预期标准差仍由研究者声明，声明过小会低估所需种子数；
`required_seeds` 只是设计阶段近似，不替代分析阶段的 MDE 与
等价性检验。机器/负载漂移若与区组内位置相关，随机化只能平衡、
不能消除，仍需调查或重跑。

## 28. 结果分析：同时置信区间与种子×窗口两层不确定性（本轮补充）

**判定：此前的区间与判定口径不完全一致，且窗口级与种子级
不确定性从未被放进同一个区间。**

1. 每个对比/因子效应报告的是**逐检验** 95% CI，但显著性判定
   用的是 Holm 校正后的 p 值。家族 m>1 时，读者会看到
   「95% CI 不含 0，却 significant=false」；这不是数学矛盾，
   而是「描述单个参数的区间」与「控制族错误率的判定」回答了
   不同问题。原报告没有把这一区别写进结果字段，容易被误读为
   实现错误，或反过来被拿来主张未校正的显著性。
2. `_paired_window_bootstrap_ci` 只在一个种子的窗口内重采样，
   `_cell_window_paired_analysis` 只把各种子的窗口均值再平均；
   两者都没有给出「同时考虑训练种子与评估窗口」的区间，也没有
   回答「下一步该加种子还是该扩大评估集」。这是本文件第 11 节
   已披露、但一直未实现的限制。

**已修正：**

- 新增 `_attach_family_adjusted_intervals`：对单因子对比、
  `--compare-baseline` 直接对比与全因子主效应/交互，在各自
  检验家族内附加 **Bonferroni 同时区间**（置信水平
  `1−α/m`）。字段为 `ci_family_adjusted_low/high`、
  `ci_family_adjusted_alpha`、`ci_family_adjusted_family_size`、
  `ci_family_adjusted_method`、`ci_family_adjusted_excludes_zero`
  与 `ci_family_adjusted_decision_consistent`。逐检验 95% CI
  （`ci95_*`）保留用于描述效应量范围。同时区间不含 0 是
  Holm 校正后显著的**充分条件**，不是必要条件；报告不再把
  两者混成一个判定。展示层在 m>1 时同时打印两套区间，并注明
  `Bonferroni α/m`。
- 新增 `_hierarchical_window_bootstrap_ci`：对 seed × window
  交叉设计做两向 bootstrap——有放回重采样种子，同时有放回
  重采样同一批固定窗口（所有抽中的种子共用同一组窗口下标，
  保留同一窗口跨种子配对的结构），最后对交叉重采样后的窗口
  差值取种子平均。区间反映「换一批训练种子 + 换一批同分布
  评估窗口」的联合不确定性；随机种子由差值内容确定性派生，
  报告可复现。结果写入每个对比的
  `window_paired.hierarchical_bootstrap_ci95`。
- 新增 `_window_variance_components`：用两向 ANOVA 分解
  seed × window 交叉设计的方差分量（种子主效应 τ²、窗口主效应
  ω²、种子×窗口交互 σ²），报告换种子且换窗口时的
  `τ²/S + ω²/W + σ²/(S·W)`、固定窗口时的
  `τ²/S + σ²/(S·W)`，以及三层标准差、方差占比
  （`variance_share_seed/window/interaction`）、
  `seed_mean_reliability`、`dominant_source` 与
  `new_seed_mean_prediction_interval_95`（同一批固定窗口上换一个
  新种子的预测区间）。直接取种子均值的样本方差会把窗口抽样
  噪声算进种子层，这里用 ANOVA 均方扣除。`dominant_source`
  给出可执行建议：种子主效应占主导时增加评估窗口无济于事；
  窗口主效应占主导时盲目加种子不经济；交互占主导时需要同时
  增加两者。
- 两层结果随 `comparison_analysis.json` 落盘；展示层新增
  `[window]` 行内的两层 bootstrap 区间与方差占比，不再只报
  「固定窗口上的点估计 + 单种子窗口 bootstrap」。
- 新增回归测试：同时区间的族规模/宽度/判定一致性、两层
  bootstrap 的可复现性与退化处理、方差占比与预测区间、
  逐窗口分析不把原始差值落进 `per_seed` 明细。

仍有限制：Bonferroni 同时区间比 Holm 保守，家族很大时区间
可能宽到没有实际信息量；两层 bootstrap 在 3~6 个种子时种子
重采样很粗，百分位区间可能偏窄或不稳定，只作描述性证据，不能
替代增加独立种子。方差分解与预测区间都条件于**这批固定评估
窗口**，不覆盖换文档/来源的外推；按文档/来源的 cluster
bootstrap 仍未实现。交叉 bootstrap 把窗口当可交换单位，未做
AR(1) 膨胀；窗口正自相关时两层区间与窗口/交互方差分量都会
偏小，应同时查看逐种子 `window_lag1_autocorrelation`。预测
区间是频率派 t 近似，不是贝叶斯后验预测。

## 29. 实验设计与准备：脱落折算与位置平衡（本轮补充）

**判定：设计预检已能拒绝「种子数不足」，但仍有三个准备阶段
的盲点。**

1. 功效公式把**计划种子数**当成**可用观测数**。数值失败、
   NaN/Inf 或数据问题导致的运行在分析期被排除；若某臂失败率
   更高，实际进入配对检验的种子会更少。计划 8 个种子、预期
   20% 不可用时，功效只相当于约 6 个，但旧报告仍会显示
   `adequate=true`。
2. `randomized` 完整区组只保证「每个区组都跑每个臂」，不保证
   每个臂在各区组位置上的出现次数均衡。小样本下某个臂可能
   5 次里有 4 次排在区组最后；位置若与机器热漂移、缓存预热
   或集群负载相关，臂效应仍会混入时间效应。旧设计计划也没有
   把位置平衡作为可核对字段。
3. 只报一个 `required_seeds` 掩盖了预期 sd 的不确定性；不报
   计划可用种子数下的实际功效与 MDE，研究者看不到设计离目标
   有多远，容易把「刚好达到 required」当成「有足够余量」。

**已修正：**

- `power` 新增可选 `expected_attrition_rate`（`[0,1)`，默认 0）。
  `preregistration_power_plan` 同时报告
  `required_seeds_with_attrition = ceil(required_seeds/(1-r))`
  与 `planned_usable_seeds = floor(planned_seeds×(1-r))`；
  `adequate` 按可用种子数判断，种子不足时错误信息给出所需
  **启动**种子数。脱落率必须在训练前声明，不能用本轮失败
  结果事后反推。
- 同一记录新增 `achieved_power_at_min_effect` 与
  `minimum_detectable_effect_relative`：按计划可用种子数、
  家族校正后的 alpha 计算「真实效应等于预注册最小效应时的
  检出概率」与「80% 功效下能检出的最小相对效应」。两者与
  `min_effect` 同单位，可直接比较，不再只报所需种子数。
- 新增 `power_sensitivity`：把 `expected_seed_sd_relative`
  乘以 0.5/0.75/1.0/1.25/1.5/2.0 重算所需种子数。先验 sd 本身
  是估计；sd 偏小 25% 就可能让设计从充足变成不足，冻结前
  应看整张敏感性表而不是点估计。
- 新增 `--run-order counterbalanced`：在完整区组上用循环拉丁方
  做限制性随机化。基础处理顺序由 `--run-order-seed` 确定性
  随机化，第 b 个种子区组使用第 `(start+b)` 个循环移位；任意
  连续 b 行中每个处理在各位置最多出现一次，位置计数极差
  ≤1。它要求每个区组包含同一组处理；`--plan-design` 遇到
  只在部分种子上运行的候选会直接拒绝，而不是生成缺格计划。
- `design_plan.json` 新增 `order_balance`：每个处理在各位置上的
  计数、`max_position_imbalance`、`complete_blocks` 与
  `balanced`。`plan_id` 覆盖该字段，事后改顺序/位置会破坏
  指纹。`print_design_summary` 在 `counterbalanced` 不均衡时
  告警，在 `randomized` 不均衡时提示可改用平衡方案。
- `comparison_analysis.json` 的 `design` 与基线调参报告都带出
  `order_balance`，分析阶段不必重新猜测执行顺序是否平衡。
- 新增回归测试：attrition 折算与不足拒绝、实际功效/MDE/
  敏感性、counterbalanced 的位置平衡与可复现性、区组处理
  集合不一致时拒绝、位置混杂检测。

**仍有限制：**

- `expected_attrition_rate` 仍是研究者声明的先验值，不是观测
  到的脱落机制。若脱落与臂别相关（某方法更容易 NaN），按可用
  运行做配对分析仍相当于条件在「没有崩掉」上；必须同时报告
  `failure_rates` 与失败率差异，不能只报成功运行的均值。
- `counterbalanced` 平衡的是**位置计数**，不是所有高阶顺序
  效应；循环拉丁方不保证一阶 carryover（前一个处理对后一个
  处理的影响）完全平衡，也不消除机器漂移本身。它只降低小样本
  随机化偶然造成的位置混杂。
- 位置平衡要求区组同质（每个种子跑同一组处理）。如果套件本身
  有缺格，正确做法是补齐运行或改用 `randomized` 并明确承认
  不平衡，不能把缺格计划标成平衡。
- 实际功效、MDE 与敏感性都条件于声明的 sd、脱落率与配对 t
  近似；种子数很少时非中心 t 积分与 sd 估计都不稳定。它们用于
  设计阶段排序风险，不替代分析阶段的 MDE、TOST 与脆弱性报告。

## 30. 消融对比的先验功效：不同对比类型的方差不同（本轮补充）

**判定：原先验功效把「两两配对差值」的方差套到所有对比上，
消融交互被系统性高估功效。**

1. `preregistration_power_plan` 用
   `_seeds_to_detect(min_effect, expected_seed_sd_relative)`，
   即 `se = sd/√n`。这对两格配对对比成立，但 2×2 消融还有
   主效应与交互：若两格配对差值的方差为 `2σ_cell²`，则主效应
   `½[(M_on−M_off)+(L_on−L_off)]` 的方差是 `σ_cell²`（0.5×），
   交互 `(M_on−M_off)−(L_on−L_off)` 的方差是 `4σ_cell²`
   （2×）。交互的 SE 是两两对比的 √2 倍，同样效应下约需 2 倍
   种子。原实现虽然把消融家族规模按 7 个检验计入了
   `planned_comparisons`，却仍用同一个 `sd` 计算所需种子数，
   所以「按两两对比够用」会被当成「交互也够用」。
2. 预注册入口只按文件里的 `comparisons` 数量检查功效；套件实际
   展开的全部臂两两比较、消融主效应/交互/简单效应只在
   `design_plan.json` 里重算家族规模，且重算时仍未区分对比
   类型。设计计划可以显示「不足」，但正式运行不会因此停下。

**已修正：**

- 新增 `_CONTRAST_VARIANCE_FACTORS`：`pairwise`/`simple`=1、
  `main`=0.5、`interaction`=2（方差相对两格配对差值）。
  `preregistration_power_plan` 对每种声明类型分别计算所需种子、
  计划可用种子下的实际功效与 MDE，并在 `contrast_power` 中
  逐项落盘；`binding_contrast` 是最吃种子的类型（同分时优先
  报交互），`adequate` 按它判定，顶层
  `required_seeds_for_t`/`achieved_power_at_min_effect`/
  `minimum_detectable_effect_relative` 也取该类型口径。
  `power_sensitivity` 对每个 sd 乘数同样按绑定类型重算，避免
  敏感性表比主表更乐观。
- 预注册新增可选 `power.planned_contrasts`（`pairwise` /
  `simple` / `main` / `interaction`，非空、不重复、拒绝未知
  类型），默认 `["pairwise"]` 兼容旧文件。`expected_seed_sd_relative`
  明确为**两格配对差值**的标准差；若只有单配置跨种子 sd
  σ_cell，应乘 √2 或直接用配对差值估计。
- 新增 `planned_contrast_types_for_plan`：从实际运行计划推断
  会估计哪些类型。`--suite-ablation`（mode × balance 两轴且含
  `local_base`）返回 `pairwise`/`main`/`interaction`/`simple`；
  `--suite-modes` 返回 `pairwise`；单臂返回空。
- `_design_power_for_family` 改为同时接收实际家族规模与对比
  类型，把「预注册声明」与「计划实际展开」取并集后重算；
  `design_plan.json` 新增 `planned_contrasts`，`power` 记录带
  `declared_planned_contrasts`/`contrasts_adjusted_for_plan`。
- 新增 `design_power_gate_error`：正式套件在第一个子进程启动
  前检查设计计划；绑定对比仍不足且没有
  `underpowered_acknowledged`/`--allow-underpowered` 时直接
  拒绝。`--plan-design` 仍只打印（不训练），便于冻结前调整
  `seeds` 或最小效应。
- 全因子效应记录新增 `contrast_type` 与
  `contrast_variance_factor`，分析报告可核对「设计时按哪种
  对比准备种子」与「现在报的是哪种对比」；分析阶段的 MDE/
  功效仍用该效应的实测 `se`，不使用乘数。
- 新增回归测试：主效应所需种子不多于两两对比、交互严格更多；
  只按两两对比够用但交互不足时 `adequate=false`；预注册拒绝
  未知对比类型；消融设计计划的 `planned_contrasts` 含交互、
  `binding_contrast=interaction`，启动硬门可被显式豁免。

**仍有限制：**

- 方差乘数假设同一批种子上各实验单元的噪声相关结构相同
  （共享 seed 效应在差值中抵消，交互方差为两两差值的 2 倍）。
  若某些单元之间的配对相关性明显不同，设计阶段的乘数只是
  近似；应以试点运行中**该对比自身**的种子间 sd 估计，或按
  最保守的乘数准备种子。分析阶段不受影响：每个效应都用自己
  的实测 `se` 做 t 检验与 MDE。
- `planned_contrasts` 只描述「打算主张哪些类型」，不改变分析
  阶段实际计算的家族；设计计划会把实际展开的类型并进来，所以
  少报类型不能让硬门静默通过。
- 交互功效仍受 `expected_seed_sd_relative` 估计误差影响；sd
  偏小 25% 就会让所需种子数明显上升，冻结前应查看
  `power_sensitivity` 与 `binding_contrast`。

## 31. 可复现性与工程规范：环境身份与重跑前核对（本轮补充）

**判定：原实现记录了 Python/torch/NumPy 版本与部分设备字段，
但「同一份代码 + 同一份数据」仍可能在另一个环境下得到不同
数值：cuBLAS/cuDNN/MKL/Triton 版本、PyTorch 编译选项、CPU
指令集、GPU 算力与驱动版本、TF32/线程/确定性开关都会改变
浮点归约顺序。原归档清单也只把 Python/torch/NumPy 版本作为
信息字段，无法在重跑前回答「现在这套环境是不是产出结果的那
一套」。**

**已修正：**

- 新增 `build_environment_snapshot()`，生成
  `smelt-isr/environment/v1` 快照：解释器实现/版本/构建、
  libc、CPU 型号与指令集指纹、PyTorch 版本/编译配置/并行
  信息、CUDA/cuDNN/MKLDNN、GPU 型号/算力/显存、NVIDIA
  驱动版本、TF32/确定性/线程开关，以及全部已安装发行包的
  规范名与精确版本。`packages.critical` 单独列出直接或间接
  决定数值内核的子集。
- 快照的 `environment_id` 只对关键字段（python/platform/
  cpu/torch/device/driver/determinism + critical packages）做
  规范化 SHA-256：可执行文件路径、主机名、采集时间、设备
  UUID 与每次运行按种子设置的 PYTHONHASHSEED 等不进入身份，
  避免同一套软件环境因虚拟环境路径或种子不同被误判。完整
  包清单仍保存，供人工重建环境。
- 每次运行把完整快照写入 `run_metadata.environment`，并单独
  原子写 `environment.json`；`config.json` 顶层 `environment`
  记录 `environment_id`、文件名与文件 SHA-256。`resume` 时
  `environment.json` 保留最初运行的环境；若当前环境已变，
  摘要写入 `resume_environment_id` 与
  `environment_changed_since_start=true`，不静默覆盖。
- 新增只读 `--check-environment RUN_DIR_OR_JSON`：从
  `environment.json` 或 `config.json` 读取归档快照，与当前
  环境比较；默认要求 `environment_id` 一致，
  `--check-environment-strict` 还要求完整包清单逐项一致。
  不一致时以非零状态码退出，并打印运行时字段与包版本差异；
  入口拒绝与训练/对比/预注册参数混用。读取
  `config.json` 摘要时会核对 `environment.json` 的 SHA-256，
  防止用被替换的“期望环境”通过核对。
- `environment_id` 加入 `--compare` 的可复现性硬核对键与
  `metrics.jsonl` 首行身份字段：同一对比中的运行若产自不同
  关键环境，报告明确列出，而不是只比较 Python/torch 两个
  版本号。
- `artifact_manifest.json` 的生成环境记录 `environment_id` 与
  完整包清单指纹，`run` 摘要带出归档运行的
  `environment_id` / `environment_file_sha256` /
  `environment_changed_since_start`；`--verify-manifest` 另
  报告清单生成环境与当前进程环境是否一致（信息字段，不参与
  归档完整性判定，因为跨机器解压校验时环境不同是预期情况）。
- 新增 `--expect-manifest-sha256 HEX`（配合
  `--verify-manifest`）：把写入时打印、并记录到 DOI/归档元数据
  的清单指纹传入校验；清单被整体替换时，内部文件仍可能全部
  自洽，只有外部锚点能发现替换。该值参与最终 `ok` 判定，
  格式不是 64 位十六进制时在只读入口直接拒绝。
- 新增回归测试：快照跨进程稳定、完整包漂移不改变关键身份、
  关键包变化破坏身份、`config.json` 摘要的文件哈希校验、
  只读入口拒绝训练参数、CLI 一致/不一致的退出码。

**仍有限制：**

- `environment_id` 不覆盖 CPU 微码、BIOS、GPU 固件、内核
  调度、集群负载或共享缓存；即使 id 相同，跨机器仍可能因
  浮点归约顺序产生微小差异。它回答「是不是同一套数值环境」，
  不保证逐位相同。
- NVIDIA 驱动版本通过 `nvidia-smi` 读取；命令不可用或权限
  受限时记为 `null`，此时不能把「无法读取」当成「驱动一致」。
- 完整包清单包含与数值无关的包；默认核对只要求关键环境
  一致，`--check-environment-strict` 才要求逐项一致。冻结
  正式实验时建议同时保存 `environment.json` 与
  `pip freeze`/容器镜像摘要，而不是只依赖 `environment_id`。
- 快照是运行开始时的状态；训练中途升级依赖或替换驱动不会
  被自动检测，必须在运行结束后重新生成并与归档比较。

## 推荐的确证流程（优先级）

1. 冻结代码：设置 `EXPERIMENT_FREEZE=true`；记录 git commit /
   `script_sha256`；把 `docs/experiment_protocol.md` 与预注册
   假设一起归档。
2. 固定数据协议：`--split-policy chronological` + 时间清单，
   或明确使用 `contiguous` 且不声称时间外推；实体清单审计；
   默认 val-only 清洗。
3. 先做设计预检：在预注册中写 `power`（预期种子间标准差、
   最小效应、目标功效、预期脱落率；消融还要写
   `planned_contrasts` 含 `interaction`）与 `run_order`
   （`counterbalanced` + 种子；区组缺格时退回 `randomized`
   并承认位置可能不均衡），用 `--plan-design` 检查所需启动
   种子数、按对比类型分层的实际功效/MDE（`binding_contrast`，
   消融通常是 `interaction`）、sd 敏感性、位置平衡与 token
   预算；功效不足时先加种子，而不是跑完再解释。
4. 运行机制对照：`smelt base local_base local_base_k2 A B B_early`，
   **至少 6 个种子**（按 `power` 计算，家族校正后通常更多），
   固定 `--common-step` 与训练预算；用 `--budget-match` 报告
   完整计算图比例；按同一 `--run-order-seed` 执行随机化/
   平衡区组。
5. 统计：主检验配对 t + 符号翻转（明确对称性假设）；报告
   失败率、MDE、门槛检验、TOST、`fragility_summary`、
   `design.order_balance` 与 `run_order`；逐检验 95% CI 只
   描述效应量，校正后的结论以 Holm 为准，并同时报告
   Bonferroni `α/m` 同时区间；窗口级证据同时报告种子×窗口
   两层 bootstrap 与方差占比，用它决定补种子还是扩评估集。
   删除种子后不再显著的结论不能写成稳健，不把不显著读成等价；
   顺序强相关先查漂移。
6. 冻结候选后统一解封 test：指定共享 `--test-ledger`，
   每个候选只解封一次；用 `--final-test-comparison` 报告全部
   候选，而不是只报告最好的那个。
7. 外部归档：把 `git archive`、全部 config/metrics（含失败）、
   `design_plan.json`/`run_order.json`、分析脚本与最终报告
   上传到不可变存储，记录 DOI/校验和。归档前对每个最终运行
   目录执行 `--write-manifest`，上传/解压后再执行
   `--verify-manifest --verify-manifest-strict`；清单是快照，
   续跑或补测后必须重新生成。
