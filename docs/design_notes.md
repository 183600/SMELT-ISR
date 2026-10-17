# SMELT-ISR 设计说明

## 概述

SMELT-ISR 在 SMELT（中间层循环两次的稀疏 MoE Transformer）的基础上，
研究第一轮路由与第二轮路由之间的合并时机对专家组合表达能力的影响。

## 运行模式

- `smelt`：原始 SMELT 基线。中间区域循环两次，每轮独立路由。
- `A`：局部延迟合并。中间层每个 Block 的第一轮 MoE 保留 `k1` 个分支，
  第二轮每个分支独立路由 `k2` 个专家，同一 Block 内两轮结束后再合并。
- `B`：全局延迟合并。第一轮在中间区域第一个 Block 处创建 `k1` 个分支，
  第一轮其余 Block 与第二轮都在整个中间区域上按分支独立执行，
  最后跨第一轮分支合并。
- `local_base`：立即合并对照。第一轮 MoE 后立即合并，再执行第二轮 MoE；
  第二轮专家宽度取 `min(n_experts, k1 * k2)`，与 A 的第二轮计算量
  （`k1` 个分支各路由 `k2` 个专家）对齐，`k1 * k2 > n_experts` 时受
  专家总数上限截断，两者不再严格等量。

## 关键配置

- `top_k`：普通 Block 的 MoE 路由宽度。
- `k1` / `k2`：第一轮分支数与第二轮路由宽度。
- `balance_coef`：负载均衡损失的系数。辅助损失按 token 数归一化，
  均匀路由时约为 `top_k`，均衡强度随路由宽度线性增大；跨
  `k1`/`k2`/`top_k` 比较实验时应按比例折算系数（保持
  `balance_coef × k` 不变），否则路由宽度会与均衡强度混杂。
- `residual_scale`：残差缩放系数，全局作用于所有区域、所有子层的残差
  更新（Attention 与 MoE 一致，循环区域内/外一致），默认 1.0。SMELT
  论文对循环区域内的子层残差更新按循环次数 $r$ 缩放 $1/r$（循环两次
  即 1/2），以抑制共享权重的反复访问推高残差流；本基座未实现按区域
  区分的 $1/r$ 缩放。跨模式实验应保持该系数一致；把全局系数设为 0.5
  并不等价于论文「仅循环区域 1/r」的口径。

## 数据协议（防止数据泄露）

- **先划分、后处理**：`ByteDataset` 按语料的时间先后顺序切分
  `train` / `val` / `test`（默认 90% / 5% / 5%，`--val-fraction`、
  `--test-fraction` 可调），三段连续、互不重叠，切分不随机打乱；
  训练阶段的批级 shuffle 由随机窗口起点承担，不改变 split 边界。
- **统计量口径**：当前字节级管线不做标准化、归一化或缺失值填充，
  不计算任何全局统计量；若扩展为连续值特征，必须只在 train 上
  `fit`，再 `transform` val/test，禁止用全量数据计算均值方差后再划分。
- **测试集尘封**：训练循环只访问 train，周期性验证只访问 val；
  `batch()` 显式拒绝 `"test"`。test 只在训练完全结束后、
  显式传入 `--evaluate-test` 时解封评估一次
  （`metrics.jsonl` 写入 `{"step": "test", ...}`）；同一输出目录中
  已有 test 记录时会拒绝重复评估，防止测试集变成隐式验证集。
- **验证集选型**：训练期间每次刷新最优验证损失（判据与早停一致）
  就把权重原子写入 `best.pt`；训练结束后的 `final` 评估与唯一一次
  test 评估都在这份验证集选定的模型上进行，而不是训练最后一步的
  参数。选型口径写入 `metrics.jsonl` 的 `final` 记录与 `config.json`
  的 `model_selection` 字段；`best.pt` 缺失或与记录不一致时明确
  提示并回退到最后一步（`last.pt` 仍是断点恢复用的完整状态）。
- **数据审计**：训练前对 train/val 自动做值域校验、文本完整性
  （无法 UTF-8 解码的字节序列与 NUL 字节，即字节级标签噪声/语料
  损坏检查）、字节直方图分布偏移（JS 距离）与滑窗重复内容扫描
  （有界采样），警告与统计写入 `config.json` 的 `data_audit`；解封
  测试集时追加含 test 的审计。恢复训练时若数据指纹、划分比例或
  群组分隔符与存档不一致会明确提示。文本完整性扫描在 split 边界
  处先做解码器状态预热，多字节 UTF-8 字符横跨切分点不会被误报为
  标签噪声；`final` 只在文件真正末尾置 True，截断字符才会报
  U+FFFD。
- **群组数据**：传入 `--group-separator` 时，语料按分隔符切成
  「群组」（同一文档/用户/物体产生的内容），train/val/test 的切分
  位置吸附到群组边界：任何群组整体落在同一个 split，两侧之间零
  交叉（构造时另有防御性断言）。吸附后若无法保证每段至少
  `seq_len + 1` 字节（如单个群组过大）则显式报错，不拆分群组。
  `data_audit.groups` 记录分隔符与每个 split 的完整群组数；
  `comparison_protocol.group_separator` 把分隔符纳入受控变量。
- **严格审计**：`--strict-data-audit` 把泄露级问题（跨 split 重复
  窗口、越界 token、超阈值 UTF-8/NUL 损坏）从警告升级为致命错误：
  训练前审计不过则拒绝开始训练；解封测试集时 test 审计不过则拒绝
  解封且不写 test 记录。默认只警告，便于先诊断再清洗。

## 代码组织

- `Config`：模型结构与损失配置，在入口处校验。
- `RMSNorm`、`CausalSelfAttention`：基础层。
- `Expert`、`SparseMoE`：专家与稀疏路由，含 `route` / `dispatch` / `merge`。
- `TransformerBlock`：注意力步骤、MoE 步骤，以及分支与两轮路由逻辑。
- `LanguageModel`：按模式组装完整前向。
- `ByteDataset`、`evaluate`：字节级数据管线与按 split 的固定窗口评估。
- `parse_group_separator`、`_snap_group_split`、`_count_groups`：
  群组分隔符解析与「群组整体不跨界」的切分位置吸附。
- `audit_data_splits`、`_window_signature_scan`、`_scan_text_integrity`、
  `enforce_strict_data_audit`：数据审计（值域、文本完整性/标签噪声、
  分布偏移、重复窗口）与严格模式下的泄露拦截。
- `run_sanity_overfit`：训练前的过拟合小批量 sanity check。
- `save_checkpoint`、`load_checkpoint`：断点存档与恢复。
- `save_selected_model`、`load_selected_model`、`restore_selected_weights`：
  验证集选定权重（`best.pt`）的存档、读取与恢复（final/test 评估用）。
- `collect_run_metadata`、`write_run_config`：实验追踪信息收集与落盘。
- `run_self_test`：结构、梯度、因果性、模式等价性、数据划分/
  测试集封存与 sanity check 判定自测。
- `analyze_training_curve`、`print_curve_analysis`：从 `metrics.jsonl`
  重建训练/验证曲线并给出过拟合、欠拟合、未收敛、发散等判定。
- `early_stop_state_from_records`：恢复训练时从历史验证点重建早停
  状态（最优值、最优步、连续未改善次数）。
- `diagnose_non_finite`、`print_non_finite_report`：首次出现
  NaN/Inf 损失或梯度时的四步排查（数据 → 前向数值 → 混合精度 →
  梯度/优化器状态），报告写入触发的 `metrics.jsonl` 记录。
- `analyze_bad_cases`、`print_bad_cases`：分组错误案例并从
  歧义、系统性混淆、疑似标签噪声、不确定四类原因中给出归类。
- `measure_efficiency`、`print_efficiency`：参数量与前向 FLOPs
  （线性层实测 + 注意力估计）。
- `compare_runs`、`print_comparison`：汇总多个运行目录的
  config 与最终指标。
- `build_comparison_protocol`、`check_controlled_variables`：
  受控变量协议记录与一致性检查。
- `aggregate_runs`、`factor_contrasts`、`_paired_t_test`、
  `_welch_t_test`、`_two_sided_t_p_value`：按实验单元聚合
  Mean ± Std、单因子显著性检验（无 SciPy 依赖的 t 分布 p 值）。
- `_holm_bonferroni`、`_benjamini_hochberg`、`_t_critical`、
  `_contrast_interval`、`_apply_multiple_comparison_correction`：
  检验家族内的多重比较校正（Holm 控制 FWER、BH 控制 FDR）、
  t 临界值与效应量 95% 置信区间；显著性判定与权衡结论
  以 Holm 校正后为准。
- `ablation_coverage`：2×2 全因子消融覆盖检查。
- `print_comparison_analysis`、`write_comparison_report`：
  完整的对比分析输出与 `comparison_analysis.json` 落盘。
- `strip_sweep_args`、`build_sweep_plan`、`run_experiment_sweep`：
  多种子 / 模式套件 / 全因子消融的逐进程编排。

## 结果分析方法

- **曲线而非单点**：每轮实验结束时从 `metrics.jsonl` 重建曲线，
  在 `{"step": "curve_analysis", ...}` 中记录状态与建议。
  判定需要至少 3 个验证点；验证损失在最优值后又明显回升判为
  过拟合，训练损失仍在下降或验证损失远高于 `ln(vocab_size)` 时
  判为欠拟合/尚未学到（`--steps` 不足），出现非有限事件判为发散。
  训练损失或梯度任一非有限都计为非有限事件并写入记录。
- **NaN/Inf 系统排查**：训练循环检测到非有限损失/梯度时跳过该步
  的优化器与调度器更新，首次事件自动运行 `diagnose_non_finite`：
  1）数据批次是否存在 NaN/Inf 或越界 token（整数比较前提升到
  64 位，前向证据不足时补跑一次诊断前向）；
  2）logits 是否非有限、是否接近低精度上限；3）fp16/bf16 混合精度
  溢出；4）逐参数梯度与优化器状态是否被污染。参数或优化器状态
  已被污染（或连续三次事件）时提前停止训练，不覆盖 `last.pt`，
  提示从污染前检查点恢复并降低学习率/收紧 `--grad-clip`。
- **坏案例分析**：从验证集固定窗口抽取置信度最低的若干 token，
  记录上下文、目标/预测字节、是否命中同窗口重复模式，按四类原因
  归类；完整报告写入 `out_dir/bad_cases.json`。测试集解封的
  唯一一次评估会附带 test 坏案例。
- **效率口径**：FLOPs 线性层部分用 `FlopCounterMode` 实测，
  注意力按 4·B·T²·D 在 Block 级 hook 估计（SMELT 中间层
  执行两轮，按真实执行次数累计）。
- **早停与选型**：`--early-stop-patience` 启用；NaN 验证损失不算
  「改善」；`--resume` 时从 `metrics.jsonl` 重建早停状态，保证续跑
  语义一致。最优验证点同时是模型选择依据（`best.pt`），final/test
  评估都在该模型上进行。
- **运行对比**：`--compare RUN_DIR...` 只读运行目录，汇总各模式
  在相同数据上的指标；跨运行对比只使用 val 口径指标，test 结果
  只在单个运行自身的 `metrics.jsonl` 中报告，避免「挑 test 最好」
  的隐式测试集调参。同时输出受控变量检查、按实验单元聚合的
  Mean ± Std、单因子显著性检验（含 Holm 多重比较校正与
  95% 置信区间）与消融覆盖报告（见下节）。

## 对比实验与消融协议

对比结论只有在「除待验证模块外，其余设置完全一致」时才可信。
本项目把这一纪律落到代码里：

- **受控变量协议**：每次训练把优化器（AdamW 及 betas/eps）、
  学习率与调度、warmup、训练步数、weight decay、梯度裁剪、
  batch size、数据增强（当前为无增强，显式记录）、数据指纹、
  模型结构、评估协议与选型口径（`model_selection`，新运行为
  验证集选型、无记录的旧运行回退标记为最后一步）写入
  `config.json` 的 `comparison_protocol` 字段。`--compare`
  默认要求这些协议项在声明因子之外逐项一致；不一致会以
  `[violation]` 明确列出（如某个运行换了优化器、调了学习率、
  训练轮数，或 final 指标口径不同），提示结论无法归因到
  方法本身。
- **声明因子**：`--compare-factors`（默认
  `mode,k1,k2,balance_coef,seed,dense_fnn_hidden`）显式声明哪些
  键允许跨运行不同；其余一律按受控变量核对。因子拼写错误会
  直接报错，不会静默关闭检查。统计基线（ngram）没有优化器与
  训练轮数，单独记为 `comparison_family=statistical`，不参与
  受控变量检查。
- **统计可靠性**：`--seeds 42 43 44` 对同一命令行配置跑多个
  随机种子，输出写入 `<out>/seed<SEED>`，结束后自动给出每个
  实验单元的 Mean ± Std（样本标准差）与单因子对比检验：种子
  集合相同用配对 t 检验（同种子 = 配对重复），否则用 Welch
  t 检验，并汇报 p 值、「噪声比」（|Δ|/合并标准差）与 Δ 的
  95% 置信区间（t 分布临界值；区间极宽说明现有种子数只能
  排除「差异为 0」，还不足以定位效应大小）。同一检验家族内
  的全部对比（一次分析的单因子对比集合，或 `--compare-baseline`
  的直接对比集合）统一做多重比较校正：显著性判定与权衡
  结论以 Holm-Bonferroni（FWER）校正后的 p 值为准，BH（FDR）
  校正值随报告给出供探索性解读；未校正判定保留在
  `significant_raw` 字段，可追溯校正前后是否翻转（6 个两两
  对比不校正时家族错误率约 26%）。种子数少于 3 的实验单元
  会被标记为统计可靠性不足；关键结论应基于预设对比且种子数
  3~5。
- **基线合理性（稻草人守卫）**：`--compare-baseline MODE`
  在给出相对差值之外，还核对基线本身是否值得战胜：基线在
  主指标上必须真的优于无上下文 unigram 参考（各运行
  `config.json` 的 `reference_baselines`），不能劣于对比集合
  里调优好的 n-gram 统计基线，且种子数应 ≥3；否则以
  `[straw-man]` 明确标记「战胜这样的弱基线不能证明复杂模型
  的收益，先把基线训练/调参到强于参考」。未指定基线或缺少
  参考记录时给出提示而不是虚报通过——没调过参的弱基线只能
  证明复杂模型比稻草人强。
- **假设登记与指标口径提示**：对比分析按模式汇总每个运行在
  `--hypothesis`/`--rq` 登记的假设，缺失假设的运行被点名
  （统计基线除外），避免把排列组合超参数当成假设驱动的实验；
  同时基于 `val_majority_accuracy`（最高频类的 support 占比）
  提示类别不平衡风险：多数类过半时 accuracy/micro 参考价值
  有限，占比 ≥90% 时 accuracy/micro 已被常数预测主导、必须
  看 macro F1 与 PR-AUC（此时 ROC-AUC 因负样本基数巨大、
  FPR 增长缓慢而偏乐观）。三项检查的结构化结果
  （`hypothesis_registry`、`baseline_adequacy`、
  `imbalance_advisory`）随 `comparison_analysis.json` 落盘。
- **全因子消融**：`--suite-ablation` 运行「延迟合并 × 均衡
  损失」2×2 全因子（`local_base`/`A` × `balance_coef`=0/当前
  值），四个组合都覆盖后才能同时给出两个创新点各自的主效应
  与交互；`--suite-modes` 把同一套受控设置应用到多个模式
  （如 `smelt local_base A B`），保证跨模式对比除模式外所有
  设置逐项一致。两个套件都自动执行消融覆盖检查，缺失的组合
  会以 `[missing]` 列出（只报告部分组合不能支撑「各自都有用」
  的结论）；一次变化多个因子的对比标记为 `[confounded]`。
- **汇总报告**：套件运行结束后把受控变量检查、聚合统计、
  单因子对比与消融覆盖写入 `<out>/comparison_analysis.json`；
  `--compare` 只读运行目录、不写文件。

## 可复现性与实验追踪

- `set_seed` 固定 Python / NumPy（若安装）/ PyTorch(CPU+CUDA) / cuDNN
  的随机种子；`CUBLAS_WORKSPACE_CONFIG=:4096:8` 与
  `use_deterministic_algorithms(True, warn_only=True)` 配合使用。
  数据采样不使用 DataLoader worker，而由显式 `torch.Generator`
  承担，其种子与状态随检查点存档。所有入口统一调用 `set_seed`
  （神经训练、统计基线、自测），没有只在部分入口固定种子的盲区。
  `PYTHONHASHSEED` 只能影响之后启动的子进程，本进程的哈希种子在
  启动时已定；进程启动时的实际值被捕获进运行元数据
  （`pythonhashseed_at_start`），为 `null` 说明本进程的 hash
  随机化未被固定，严格逐位复现应在启动命令行显式设置该变量。
- 全新训练开始前默认运行过拟合小批量 sanity check：
  同一模型结构在固定小批次上训练，关闭辅助均衡损失、weight decay、
  梯度裁剪与 warmup，要求训练损失迅速趋近 0 或下一 token 准确率
  接近 100%。检查走与全量训练一致的前向/反向路径：主训练启用
  `--grad-checkpoint` 时检查同样开启 activation checkpointing
  （唯一会改写反传代码路径的开关，报告记录 `grad_checkpoint` 与
  `amp_dtype`），`--compile` 只做内核融合不改写 autograd 语义、
  不镜像。检查内部固定种子并在结束后恢复 RNG 状态，不影响主训练
  轨迹；失败时以非零状态码退出，不开始全量训练。这是必要不充分
  检查，通过后仍需核对中间结果。可用 `--sanity-only` 单独运行、
  `--skip-sanity-check` 跳过。
- 每轮实验的代码版本、运行环境、超参数与随机种子、数据指纹、
  sanity check 结果写入 `config.json`；训练曲线与最终结果写入
  `metrics.jsonl`，全程自动记录。代码版本是三层记录：
  git commit（仓库可用时）、`git_diff_sha256`（工作区脏时对
  `git diff HEAD` 的指纹，区分同一 commit 的不同未提交状态）与
  `script_sha256`（实验脚本自身的 SHA-256，单文件代码库下即完整
  实验代码的指纹，不依赖 git，覆盖压缩包运行与脏工作区两种
  commit 无法定位代码的场景）。

## 训练入口

```
python experiment.py --data <语料文件> --out <输出目录> --mode <模式>
```

常用参数：`--steps`、`--batch-size`、`--lr`、`--warmup-steps`、
`--grad-clip`、`--eval-interval`、`--save-interval`、`--resume`、
`--grad-checkpoint`、`--compile`、`--seed`、`--note`、`--sanity-only`、
`--skip-sanity-check`、`--sanity-batch-size`、`--sanity-steps`、
`--sanity-lr`、`--sanity-loss-threshold`、`--sanity-acc-threshold`、
`--val-fraction`、`--test-fraction`、`--evaluate-test`、
`--group-separator`（群组整体划分）、`--strict-data-audit`
（泄露级审计问题拒绝训练/拒绝解封测试集）。

结果分析相关参数：`--early-stop-patience`（默认 0，关闭早停）、
`--early-stop-min-delta`、`--bad-cases`（默认 10，0 关闭）、
`--skip-efficiency`（跳过结束时的效率测量）、
`--compare RUN_DIR...`（只读对比多个运行目录，不训练；含受控
变量检查、Mean ± Std 与显著性检验）、`--compare-factors`、
`--compare-baseline MODE`、`--compare-metric`。

多种子 / 消融套件：`--seeds SEED [SEED ...]`、
`--suite-modes MODE [MODE ...]`（配合 `--seeds` 做控变量的
模式对比）、`--suite-ablation`（配合 `--seeds` 做 2×2 全因子
消融）。套件逐进程调用自身，结束后自动输出完整对比分析并
写入 `<out>/comparison_analysis.json`。
