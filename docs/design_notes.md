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
- `run_sanity_overfit`、`_probe_gradient_flow`、`_expert_parameter_ids`：
  训练前的过拟合小批量 sanity check 与第一步梯度探针（损失连通性、
  有限性、非零梯度、非专家参数缺失梯度）。
- `save_checkpoint`、`load_checkpoint`：断点存档与恢复
  （Python/torch/CUDA/NumPy RNG 状态一并存档并逐位恢复）。
- `save_selected_model`、`load_selected_model`、`restore_selected_weights`：
  验证集选定权重（`best.pt`）的存档、读取与恢复（final/test 评估用）。
- `collect_run_metadata`、`write_run_config`：实验追踪信息收集与落盘。
- `run_self_test`：结构、梯度、因果性、模式等价性、数据划分/
  测试集封存、sanity check 判定与「同种子两次运行逐位一致」
  的端到端可复现性自测。
- `analyze_training_curve`、`print_curve_analysis`：从 `metrics.jsonl`
  重建训练/验证曲线并给出过拟合、欠拟合、未收敛、发散等判定；
  泛化 gap 持续扩大但验证尚未回升时给出「过拟合前兆」警告与
  早停建议。`monitor_training_curve` 在训练期间的每个验证点
  复用同一判定，只在状态迁移时打印并写入 `curve_monitor` 记录。
- `early_stop_state_from_records`：恢复训练时从历史验证点重建早停
  状态（最优值、最优步、连续未改善次数）。
- `diagnose_non_finite`、`print_non_finite_report`：首次出现
  NaN/Inf 损失或梯度时的四步排查（数据 → 前向数值 → 混合精度 →
  梯度/优化器状态），报告写入触发的 `metrics.jsonl` 记录。
- `analyze_bad_cases`、`print_bad_cases`：在验证集固定窗口上流式
  统计全部目标 token（不做随机下采样），把错误案例按歧义、
  系统性混淆、疑似标签噪声、不确定四类原因归类；top-N 最差样本
  用「每批 top-N + 跨批截断」精确维护，窗口只为入选样本保留。
- `measure_efficiency`、`print_efficiency`：参数量与前向 FLOPs
  （线性层实测 + 注意力估计）。
- `compare_runs`、`print_comparison`：汇总多个运行目录的
  config 与最终指标。
- `build_comparison_protocol`、`check_controlled_variables`：
  受控变量协议记录与一致性检查。
- `aggregate_runs`、`factor_contrasts`、`_paired_t_test`、
  `_welch_t_test`、`_two_sided_t_p_value`：按实验单元聚合
  Mean ± Std、单因子显著性检验（无 SciPy 依赖的 t 分布 p 值）。
- `_t_power`、`_minimum_detectable_effect`、`_seeds_to_detect`：
  非中心 t 分布功效的数值积分（对 chi 分布做自适应 Simpson
  积分，按 df 缓存节点）、80% 功效的最小可检出效应（MDE）
  与按当前种子间波动检出观测效应所需的近似种子数。
- `_tost_test`、`_t_cdf`、`_attach_power_equivalence`：TOST
  等价性检验（等价界相对基线均值，`--equivalence-margin`）、
  t 分布 CDF 与对比记录的功效/等价性字段挂载。
- `_holm_bonferroni`、`_benjamini_hochberg`、`_t_critical`、
  `_contrast_interval`、`_apply_multiple_comparison_correction`：
  检验家族内的多重比较校正（Holm 控制 FWER、BH 控制 FDR）、
  t 临界值与效应量 95% 置信区间；显著性判定与权衡结论
  以 Holm 校正后为准，等价性（TOST）主张做同口径校正。
- `ablation_coverage`：(1+N)×2 全因子消融覆盖检查（默认
  N=A/B 两个延迟合并方案；含消融单元之间
  `k1/k2/top_k/residual_scale` 结构一致性与均衡强度一致性
  的混杂检查）。
- `print_comparison_analysis`、`write_comparison_report`：
  完整的对比分析输出与 `comparison_analysis.json` 落盘。
- `strip_sweep_args`、`build_sweep_plan`、`run_experiment_sweep`：
  多种子 / 模式套件 / 全因子消融的逐进程编排。

## 结果分析方法

- **曲线而非单点**：每轮实验结束时从 `metrics.jsonl` 重建曲线，
  在 `{"step": "curve_analysis", ...}` 中记录状态与建议；
  训练期间每个验证点在线复用同一判定，状态迁移（如
  healthy→overfitting、发散）立即打印并写入 `curve_monitor`。
  判定需要至少 3 个验证点；验证损失在最优值后又明显回升判为
  过拟合，训练损失不降且仍接近随机基线判为欠拟合，两者都不降
  （区分学习率已退火与停滞）指向欠拟合/学习率不当/数据管道，
  出现非有限事件判为发散。泛化 gap 从首个验证点起扩大达到阈值
  且最终 gap 已不低时，即使验证损失尚未回升也会给出过拟合前兆
  警告（`gap_widening_warning`）与早停建议。训练损失或梯度任一
  非有限都计为非有限事件并写入记录。
- **NaN/Inf 系统排查**：训练循环检测到非有限损失/梯度时跳过该步
  的优化器与调度器更新，首次事件自动运行 `diagnose_non_finite`：
  1）数据批次是否存在 NaN/Inf 或越界 token（整数比较前提升到
  64 位，前向证据不足时补跑一次诊断前向）；
  2）logits 是否非有限、是否接近低精度上限；3）fp16/bf16 混合精度
  溢出；4）逐参数梯度与优化器状态是否被污染。参数或优化器状态
  已被污染（或连续三次事件）时提前停止训练，不覆盖 `last.pt`，
  提示从污染前检查点恢复并降低学习率/收紧 `--grad-clip`。
- **坏案例分析**：在验证集固定窗口上流式统计全部 token 的
  损失/置信度/margin（每 token 约 10 字节紧凑缓冲，不物化窗口
  矩阵、不做随机下采样，token 准确率与 `evaluate` 同口径），
  top-N 最差样本用「每批 top-N + 跨批截断」精确维护，记录上下文、
  目标/预测字节与混淆计数，按四类原因归类；全量错误 token 的
  类别份额（`error_category_counts/shares`）作为改进线索的主证据，
  避免「最高损失样本恰好同类」把结论带偏；完整报告写入
  `out_dir/bad_cases.json`。测试集解封的唯一一次评估会附带
  test 坏案例。
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
  模型结构（含分支几何 `k1/k2`、路由宽度 `top_k`、均衡损失
  系数 `balance_coef`、`dense_fnn_hidden`、`residual_scale`）、
  评估协议与选型口径（`model_selection`，新运行为验证集选型、
  无记录的旧运行回退标记为最后一步），以及有效训练长度与
  数值设置（早停容忍度 `early_stop_patience`/
  `early_stop_min_delta`、AMP 精度 `amp_dtype`、设备类型
  `device_type`、梯度检查点 `grad_checkpoint`、
  `compile`：两臂的 AMP/设备/早停不同会让「方法收益」混入
  训练过程差异）写入 `config.json` 的
  `comparison_protocol` 字段。这些键即使没有出现在
  `--compare-factors`（默认因子包含 `k1/k2/balance_coef/`
  `dense_fnn_hidden`）里，也会退化为受控变量逐项核对——把
  因子列表收窄到 `mode,seed` 时，某个运行偷偷改大 `k1` 会被
  立即抓住。旧版本记录的协议缺少后来新增的键时，用
  `config/args/data` 中可唯一确定的值回填，避免新旧运行
  混用时把「未记录」误判成「值不同/缺失」。`--compare`
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
  排除「差异为 0」，还不足以定位效应大小）。显著性检验要求
  每侧至少 3 个随机种子（配对/ Welch t 检验 df ≥ 2；2 个
  种子时 df=1，p 值基本由噪声决定）；少于 3 个种子只汇报
  差值，显式标注「种子数不足，未做显著性检验，不能据此
  声称差异」。同一检验家族内
  的全部对比（一次分析的单因子对比集合，或 `--compare-baseline`
  的直接对比集合）统一做多重比较校正：显著性判定与权衡
  结论以 Holm-Bonferroni（FWER）校正后的 p 值为准，BH（FDR）
  校正值随报告给出供探索性解读；未校正判定保留在
  `significant_raw` 字段，可追溯校正前后是否翻转（6 个两两
  对比不校正时家族错误率约 26%）。种子数少于 3 的实验单元
  会被标记为统计可靠性不足；关键结论应基于预设对比且种子数
  3~5。**不显著 ≠ 等价（功效与等价性口径）**：显著性检验
  只能回答「有没有证据反对差异为 0」。每个检验过的对比同时
  记录 80% 功效下的最小可检出效应 `mde_80`（含相对基线均值的
  `mde_80_relative`）、观测效应自身的功效 `power_at_effect` 与
  按当前种子间波动检出观测效应所需的近似种子数
  `seeds_to_detect`；并对预先声明的等价界 `--equivalence-margin`
  （相对基线均值，默认 1%）做 TOST 两单侧等价性检验。TOST
  p 值（`tost_p_value`，较小者支持等价）与显著性检验共享
  家族校正：Holm 校正后仍通过才能在报告中主张「无实质差异」
  （`equivalent`，未校正判定留档于 `equivalent_raw`）；不显著
  且等价性未确立时结论只能是「不确定」，展示层打印 MDE 与
  所需种子数，区分「效应太小」与「种子太少」。等价界必须
  事先声明：看到数据后再放宽界可以「制造」等价结论。功效
  积分对 chi 分布做自适应 Simpson 数值积分（按 df 缓存），
  δ=0 时功效恰为 alpha，大 df 退化为正态近似，自测覆盖这些
  解析回归点。**伪重复检测**：同一实验单元里同一随机种子出现多次
  （复制运行目录、重复登记同一运行）会被按种子配对时的字典
  静默合并、虚增样本数并低估种子间方差；聚合层显式记录
  `duplicate_seeds`，展示层点名重复种子，检验层禁用配对检验
  并退回 Welch（每个运行各算一个样本），`comparison_analysis.json`
  的 `duplicate_seed_cells` 列出全部伪重复单元。关键套件
  （`--suite-modes`/`--suite-ablation`/`--suite-baseline-tuning`）
  在少于 3 个不同随机种子时直接拒绝运行，冒烟/调试需显式
  加 `--allow-underpowered`；豁免与种子计划（`seed_plan`）
  随报告落盘，避免低功效运行被当作正式结论。
- **基线合理性（稻草人守卫）**：`--compare-baseline MODE`
  在给出相对差值之外，还核对基线本身是否值得战胜：基线在
  主指标上必须真的优于无上下文 unigram 参考（各运行
  `config.json` 的 `reference_baselines`；PR-AUC / ROC-AUC
  同样有常数预测器参考——纳入类平均 prevalence 与 0.5，
  切换对比指标后守卫不会静默失效），不能劣于对比集合
  里调优好的 n-gram 统计基线，种子数应 ≥3，且必须带有调参
  证据（`config.json` 的 `baseline_tuning` 来源记录；统计基线
  自身在 val 上做过 阶数 × 平滑系数 网格调优，视为已调参）；
  否则以 `[straw-man]` 明确标记「战胜没调过参的弱基线不能
  证明复杂模型的收益，先用 `--suite-baseline-tuning` 把基线
  调强」。调参依据的指标与当前对比指标不一致时不打印无条件
  的 OK，而是显式警告「不能证明基线在该指标上被调强」
  （调在 val_loss 上不能证明 macro F1 也强）。
  未指定基线或缺少参考记录时给出提示而不是虚报通过。
- **简单基线的调参协议**：`--suite-baseline-tuning`（配合
  `--seeds`）对简单基线做预注册的学习率网格搜索
  （`--baseline-tuning-lrs`，默认 `1e-4 3e-4 1e-3`；
  `--baseline-tuning-mode` 选模式）：候选之间只允许学习率
  不同（steps/batch/数据/结构/调度等全部继承同一条命令行），
  保证调参与「训练更久或换了协议」不混杂；按 val 指标
  Mean ± Std 选择 winner——先取最优均值候选，再把与最优均值
  差距小于种子波动（两侧 std 的较大者）的候选视为无法区分，
  从并列集合中取更小 lr 的保守解，并列集合与选择方式
  （`winner_selection`/`tied_lrs`）如实写入报告；winner 落在
  网格边界时提示扩展候选后重跑。报告写入
  `<out>/baseline_tuning.json`。后续基线训练用
  `--baseline-tuning-from <报告>` 核对模式 / winner lr / steps /
  数据指纹与完整训练协议（warmup/调度/weight decay/梯度裁剪/
  数据划分/评估窗口等）逐项一致（旧版报告缺协议字段时显式
  拒绝，不默认为通过），并把来源写入 `config.json` 的
  `baseline_tuning` 字段——这是「基线被调过参」的唯一证据；
  报告本身不是最终基线，test 集全程尘封。
- **假设登记与指标口径提示**：对比分析按模式汇总每个运行在
  `--hypothesis`/`--rq` 登记的假设，缺失假设的运行被点名
  （统计基线除外），避免把排列组合超参数当成假设驱动的实验；
  假设没写明判定指标（如「效果更好」而不提验证损失 / macro
  F1 / PR-AUC / 召回率）同样无法与评估口径对齐，单独点名；
  写明的指标必须是评估管线真正计算的（precision / recall 已
  按 macro / weighted 口径输出并进入对比指标集合；路由类
  指标需要该运行有 `routing_analysis` 记录），否则同样点名
  「该指标当前不可判定」。单次训练启动时也会检查假设是否
  写明判定指标，并把 `hypothesis_metric_mentions` 写入
  `config.json`，避免跑完才发现无法判定。
  同时基于 `val_majority_accuracy`（最高频类的 support 占比）
  提示类别不平衡风险：多数类过半时 accuracy/micro 参考价值
  有限，占比 ≥90% 时 accuracy/micro 已被常数预测主导、必须
  看 macro F1 与 PR-AUC（此时 ROC-AUC 因负样本基数巨大、
  FPR 增长缓慢而偏乐观）；占比 ≥99.9%（约 1:1000 以上）按
  极度不平衡单独分级——随机 PR-AUC 只有平均 prevalence
  （≈0.001，评估时记录为 `val_pr_auc_macro_chance`），排序
  判定必须报告 PR-AUC 高出该水平的 lift，而不能把 PR-AUC
  绝对值当准确率读。四项检查的结构化结果
  （`hypothesis_registry`、`baseline_adequacy`、
  `imbalance_advisory`）随 `comparison_analysis.json` 落盘。
- **全因子消融**：`--suite-ablation` 运行「延迟合并 × 均衡
  损失」全因子（`local_base` 与各延迟合并方案 × `balance_coef`
  =0/当前值；(1+N)×2 格分别对应「无 / 仅延迟合并 / 仅均衡
  损失 / 两者」）。默认覆盖项目提出的全部延迟合并方案
  （A 与 B，共 3×2=6 格）：每个创新点都必须证明各自有用，
  只跑 A 的组合不足以支撑 B 的结论；只声明部分方案时用
  `--ablation-modes A|B`（声明后覆盖检查按声明核对，未声明
  时按运行集中实际出现的方案推断）。全部组合覆盖后才能同时
  给出各方案与均衡损失的主效应及交互；`--suite-modes` 把同一
  套受控设置应用到多个模式（如 `smelt local_base A B`），
  保证跨模式对比除模式外所有设置逐项一致。两个套件都自动
  执行消融覆盖检查，缺失的组合会以 `[missing]` 列出（只报告
  部分组合不能支撑「各自都有用」的结论）；一次变化多个因子
  的对比标记为 `[confounded]`。消融覆盖检查还会核对各单元
  之间 `k1/k2/top_k/residual_scale` 与均衡开启臂的
  `balance_coef` 是否一致：组合齐全但结构设置或均衡强度不同
  （如某个单元悄悄用了更大的 `k1`、A 与 B 的开启臂用了不同
  balance）时主效应会与分支几何、路由宽度、残差缩放或均衡
  强度混杂，同样以 `[confounded]` 标出且不声明「全因子齐全」。
  出现 B 时还会提示与 smelt 的计算量不等价，要求同时报告
  等计算对照。
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
  Python / torch / CUDA / NumPy 的 RNG 状态全部随 `last.pt` 存档，
  `--resume` 时逐位恢复（NumPy 状态以只含基本类型的载荷存放，
  兼容 `torch.load(weights_only=True)`），恢复训练不会因随机流
  错位而偏离原轨迹。`PYTHONHASHSEED` 只能影响之后启动的子进程，
  本进程的哈希种子在启动时已定；进程启动时的实际值被捕获进运行
  元数据（`pythonhashseed_at_start`），为 `null` 时 `set_seed`
  会打印一次性提示，严格逐位复现应在启动命令行显式设置该变量。
  自测内对同一配置做两次同种子的 sanity 检查并要求最终损失/准确率
  逐位相等，作为「同一台机器上结果稳定」的端到端证据。
- 全新训练开始前默认运行过拟合小批量 sanity check：
  同一模型结构在固定小批次上训练，关闭辅助均衡损失、weight decay、
  梯度裁剪与 warmup，要求训练损失迅速趋近 0 或下一 token 准确率
  接近 100%。检查走与全量训练一致的前向/反向路径：主训练启用
  `--grad-checkpoint` 时检查同样开启 activation checkpointing
  （唯一会改写反传代码路径的开关，报告记录 `grad_checkpoint` 与
  `amp_dtype`），`--compile` 只做内核融合不改写 autograd 语义、
  不镜像。第一步反向之后先跑梯度探针（`grad_probe`）：损失必须
  连在计算图上（否则立即停止）、全部梯度必须有限、至少一个参数
  收到非零梯度、非专家参数不得缺失梯度（专家参数允许因本批次未
  路由到而缺失/零梯度）；探针失败立即停止并给出排查提示。
  检查内部固定种子并在结束后恢复 RNG 状态，不影响
  主训练轨迹；失败时以非零状态码退出，不开始全量训练。这是必要
  不充分检查，通过后仍需核对中间结果。可用 `--sanity-only` 单独
  运行、`--skip-sanity-check` 跳过。
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
变量检查、Mean ± Std、显著性检验与功效/等价性分析）、
`--compare-factors`、`--compare-baseline MODE`、`--compare-metric`、
`--equivalence-margin REL`（TOST 等价界，相对基线均值，默认
0.01 = 1%；报告以此区分「无实质差异」与「未发现显著差异」，
并给出 80% 功效 MDE 与所需种子数）。

多种子 / 消融套件：`--seeds SEED [SEED ...]`、
`--suite-modes MODE [MODE ...]`（配合 `--seeds` 做控变量的
模式对比）、`--suite-ablation`（配合 `--seeds` 做 (1+N)×2
全因子消融）、`--ablation-modes A [B ...]`（配合
`--suite-ablation` 声明本轮必须覆盖的延迟合并方案，缺省
A B）、`--allow-underpowered`（显式允许套件在 <3 个种子下
运行，豁免随报告落盘，仅限冒烟/调试）、
`--suite-baseline-tuning`（配合 `--seeds` 做简单基线
的学习率网格调参，`--baseline-tuning-lrs` 给候选、
`--baseline-tuning-mode` 选模式）。关键套件至少需要 3 个
不同随机种子，否则直接报错要求改用 3~5 个种子或显式豁免。
套件逐进程调用自身，结束后自动输出完整对比分析并写入
`<out>/comparison_analysis.json`（调参套件写
`<out>/baseline_tuning.json`）。基线训练可用
`--baseline-tuning-from <报告>` 核对并记录调参来源。
