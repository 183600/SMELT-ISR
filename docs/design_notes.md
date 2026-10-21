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
  `local_base` 第二轮为对齐计算量使用宽度
  `min(n_experts, k1*k2)`，实现会把该轮 aux 按
  `k2 / min(n_experts, k1*k2)` 折算回 A/B 的单分支宽度；
  `comparison_protocol.balance_width_normalization` 记录该口径，
  用于区分未折算的旧运行。对比分析另有
  `balance_strength_report` 核对 `balance_coef × 有效路由宽度`，
  宽度不同而强度未抵消时以 `[balance-scale]` 警告。
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
- **验证集选型**：训练期间每次刷新最优选型指标（默认验证
  损失，`--selection-metric` 可改为与预注册主指标一致的分类
  指标；判据与早停一致）就把权重原子写入 `best.pt`；训练结束
  后的 `final` 评估与唯一一次 test 评估都在这份验证集选定的
  模型上进行，而不是训练最后一步的参数。选型指标、选中步与
  选中点的验证损失写入 `metrics.jsonl` 的 `final` 记录与
  `config.json` 的 `model_selection` 字段；`best.pt` 缺失或与
  记录不一致时明确提示并回退到最后一步（`last.pt` 仍是断点
  恢复用的完整状态）。
- **数据审计**：训练前对 train/val 自动做值域校验、文本完整性
  （无法 UTF-8 解码的字节序列与 NUL 字节，即字节级标签噪声/语料
  损坏检查）、字节直方图分布偏移（JS 距离）与滑窗重复内容扫描
  （有界采样），警告与统计写入 `config.json` 的 `data_audit`；解封
  测试集时追加含 test 的审计。恢复训练时若数据指纹、划分比例或
  群组分隔符与存档不一致会明确提示。文本完整性扫描在 split 边界
  处先做解码器状态预热，多字节 UTF-8 字符横跨切分点不会被误报为
  标签噪声；`final` 只在文件真正末尾置 True，截断字符才会报
  U+FFFD。每个 split 的内容指纹写入 `data.split_sha256`，并作为
  受控变量核对：源文件相同但切分/群组吸附/清洗实现不同，仍会被
  识别为不同数据协议。
- **群组数据**：传入 `--group-separator` 时，语料按分隔符切成
  「群组」（同一文档/用户/物体产生的内容），train/val/test 的切分
  位置吸附到群组边界：任何群组整体落在同一个 split，两侧之间零
  交叉（构造时另有防御性断言）。吸附后若无法保证每段至少
  `seq_len + 1` 字节（如单个群组过大）则显式报错，不拆分群组。
  `data_audit.groups` 记录分隔符与每个 split 的完整群组数；
  `comparison_protocol.group_separator` 把分隔符纳入受控变量。
- **训练集去重**：`--deduplicate-train-groups` 只从 train 剔除与
  val/test 或更早 train 群组内容完全相同的群组，val/test 字节与切分
  位置不变；被剔除内容不进入神经窗口、unigram / n-gram 计数或分布
  审计，`data_audit` 的 `effective_bytes` / `effective_windows` 与
  `config.data.split_fractions` 记录清洗后的真实训练口径。训练期只用
  切分边界核对 test 占比（不读 test 内容）；解封前独立探针再核对
  test 固定窗口的输入/标签逐位对齐。
- **跨 split 共享内容块清洗**：`--drop-train-shared-chunks`（需配合
  群组分隔符）用与审计相同的内容定义分块指纹，把与 val/test 共享
  连续字节块的整个训练群组从 train 剔除；val/test 字节、切分位置与
  内容指纹逐位不变，清洗后同一审计必须归零。它补上「精确群组指纹
  抓不到嵌入片段」的空档，与 `--deduplicate-train-groups` 可叠加。
- **连续字节流标签噪声清洗**：无群组分隔符时，
  `--drop-train-corrupt-bytes` 只从 train 剔除非法 UTF-8 字节与
  NUL 字节本身（不修补、不发明标签），val/test 字节与切分位置
  不变；被剔除区间不进入训练窗口、unigram / n-gram 计数、内容
  分块与分布审计，`data_audit.effective_bytes` 与
  `config.data.train_cleaning` 记录清洗口径。有群组边界时应使用
  `--drop-train-corrupt-groups` 以保持群组完整。
- **连续字节流异常值清洗**：无群组分隔符时，
  `--drop-train-degenerate-runs` 只从 train 剔除长度达到阈值
  （64B）的单一字节连续段（填充、重复字符、错误拼接的常见形态），
  同样不修补字节；`data_audit.degenerate_runs` 记录退化段计数、
  字节数与最大段长，清洗后归零。有群组边界时应使用
  `--drop-train-degenerate-groups` 以保持群组完整。
- **分布偏移硬门**：`--max-byte-shift` / `--max-bigram-shift` /
  `--max-conditional-shift` 可把字节、二元组、条件标签 JS 距离
  变成显式阈值；train/val（解封时含 test）任一对 split 超限即
  拒绝训练/拒绝解封。默认不设阈值，只保留警告，避免把正常
  的时序分布变化直接当成错误。
- **严格审计**：`--strict-data-audit` 把泄露级问题（跨 split 重复
  窗口、越界 token、超阈值 UTF-8/NUL 损坏）从警告升级为致命错误：
  训练前审计不过则拒绝开始训练；解封测试集时 test 审计不过则拒绝
  解封且不写 test 记录。默认只警告，便于先诊断再清洗。
- **测试集标签完整性硬门**：test 不参与任何清洗，因此带标签噪声的
  test 上得出的最终指标不可信。解封时（神经与统计基线两条路径）
  只要 test 存在越界 token、非法 UTF-8 或 NUL 就默认拒绝解封；
  只有显式传入 `--allow-corrupt-test-labels` 才会放行，且豁免写入
  `config.json` 与对比协议。分布偏移阈值对统计基线解封同样生效。
- **扫描覆盖与残余风险**：窗口签名扫描在窗口数超过上限时按 stride
  等间距抽样，因此「未发现跨 split 重复窗口」只是采样范围内的负
  结果。`data_audit.leakage_scan_coverage` 记录各 split 的
  采样/总窗口数、stride 与覆盖率；stride>1 时审计与泄露门都会显式
  写出残余风险（「未发现」≠「不存在」）。内容定义分块指纹按局部
  内容穷尽扫描，可兜底被 stride 漏掉的错位重复，但兜底命中与否
  都不改变窗口级负结果的抽样性质。

## 代码组织

- `Config`：模型结构与损失配置，在入口处校验。
- `RMSNorm`、`CausalSelfAttention`：基础层。
- `Expert`、`SparseMoE`：专家与稀疏路由，含 `route` / `dispatch` / `merge`。
- `TransformerBlock`：注意力步骤、MoE 步骤，以及分支与两轮路由逻辑。
- `LanguageModel`：按模式组装完整前向。
- `ByteDataset`、`evaluate`：字节级数据管线与按 split 的固定窗口评估。
- `parse_group_separator`、`_snap_group_split`、`_count_groups`：
  群组分隔符解析与「群组整体不跨界」的切分位置吸附。
- `audit_data_splits`、`_window_signature_scan`、`_leakage_scan_coverage`、
  `_scan_text_integrity`、`enforce_strict_data_audit`、
  `enforce_data_leakage_gate`：数据审计（值域、文本完整性/标签噪声、
  分布偏移、重复窗口及其扫描覆盖）、严格模式下的标签噪声拦截与
  默认启用的跨 split 泄露硬门。
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
- `measure_efficiency`、`print_efficiency`：参数量与前向 FLOPs。
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
- `_attach_sd_uncertainty_note`：登记所需种子数为 sd 点估计
  外推的插件估计，并给出 sd 的相对标准误 1/√(2(n-1))。
- `_hedges_g_correction`：效应量小样本偏倚校正因子
  J ≈ 1−3/(4·df−1)（Cohen's d → Hedges g）。
- `_attach_family_power`、`_observed_seed_sd`：按家族规模
  α/m 补算 MDE/功效/所需种子/所需等价界的保守口径，并由检验
  标准误还原（近似）种子间标准差。
- `_tost_test`、`_t_cdf`、`_attach_power_equivalence`：TOST
  等价性检验（等价界相对基线均值，`--equivalence-margin`）、
  t 分布 CDF 与对比记录的功效/等价性字段挂载，含主张等价所需
  的最小界 `equivalence_margin_needed`、阻塞原因
  `equivalence_blocker` 与精度受限时的近似所需种子数
  `equivalence_seeds_to_achieve`。
- `_bootstrap_eval_intervals`、`_percentile_interval`：对固定
  评估窗口做有放回重采样（默认 500 次，`--eval-bootstrap`，
  0 关闭），给出 loss / ppl / macro NLL / token accuracy /
  macro 与 weighted F1 / macro 与 weighted precision 与
  recall 的百分位 95% 置信区间；假设以 F1 或召回率判定时，
  区间与点估计同口径，而不是只给 loss/accuracy 误差线；
  种子由数据指纹、split、窗口数与
  窗口长度派生，确定性可复现，且不扰动训练/数据 RNG。
  置信区间回答「换一批同分布评估窗口，点估计会漂多少」，
  与跨种子 t 检验（回答「换训练种子」）互补。退化重采样
  （例如 macro F1 抽到的窗口没有任何 support）会产生 NaN；
  区间只用有限重采样估计，并记录 `<metric>_bootstrap_degenerate`
  次数，避免整个分位区间被 NaN 污染。同时记录
  `eval_windows` 与 `eval_window_overlap_fraction`：窗口重叠
  时 bootstrap 的重采样单位并不独立，区间会偏窄，读者不能
  把重叠窗口的区间当作独立样本精度。窗口不重叠时也检测
  lag-1 自相关（`window_lag1_autocorrelation`、
  `window_effective_windows`、`window_ci_inflation`，阈值
  `_EVAL_WINDOW_AUTOCORRELATION`）：沿评估序列的正相关同样
  违反窗口独立前提，达到阈值即按 AR(1) 方差膨胀放宽区间。
- `_paired_sign_flip_test`、`_leave_one_out_effect_stability`：
  配对符号翻转（randomization）检验与 leave-one-out 稳健性。
  n≤12 精确枚举全部 2^n 符号组合，n>12 用固定种子 Monte
  Carlo；不依赖差值正态性，适合 3~5 个种子的实验。结果同时
  记录最小可达 p 值，样本量不足以在 0.05 水平显著时明确
  提示「不显著 ≠ 无效应」。leave-one-out 逐一去掉一个种子
  重算效应，方向翻转或归零时提示结论可能由单个种子驱动。
- `cross_family_protocol_check`：神经运行与 ngram 统计基线
  之间的数据/评估口径核对（数据指纹、切分比例、评估窗口数、
  batch size、数据增强等）；口径不一致时禁止把 ngram 当作
  「基线已足够强」的证据。
- `_holm_bonferroni`、`_benjamini_hochberg`、`_t_critical`、
  `_contrast_interval`、`_apply_multiple_comparison_correction`：
  检验家族内的多重比较校正（Holm 控制 FWER、BH 控制 FDR）、
  t 临界值与效应量 95% 置信区间；显著性判定与权衡结论
  以 Holm 校正后为准，等价性（TOST）主张做同口径校正。
- `ablation_coverage`：(1+N)×2 全因子消融覆盖检查（默认
  N=A/B 两个延迟合并方案；含消融单元之间
  `k1/k2/top_k/residual_scale` 结构一致性、均衡强度一致性、
  共同种子对齐与 B/local_base 计算量倍率的混杂检查）。
- `factorial_effects`、`annotate_factorial_confounds`：2×2
  主效应、交互与简单效应的共同种子配对检验（Holm 校正、
  95% CI、MDE），并按受控变量/结构/计算量混杂降级结论。
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
  非有限都计为非有限事件并写入记录。有限损失下训练与验证的近期
  区间同时上升时单独判为 `diverging`（有限发散），与 NaN 发散
  区分；梯度范数相对历史中位数突增超过 5 倍时给出
  `grad_norm_spike_warning`，提示检查学习率/梯度裁剪。
- **NaN/Inf 系统排查**：训练循环检测到非有限损失/梯度时跳过该步
  的优化器与调度器更新，首次事件自动运行 `diagnose_non_finite`：
  1）数据批次是否存在 NaN/Inf 或越界 token（整数比较前提升到
  64 位，前向证据不足时补跑一次诊断前向）；
  2）logits 是否非有限、是否接近低精度上限；3）fp16/bf16 混合精度
  溢出；4）逐参数梯度与优化器状态是否被污染。参数或优化器状态
  已被污染（或连续三次事件）时提前停止训练，不覆盖 `last.pt`，
  提示从污染前检查点恢复并降低学习率/收紧 `--grad-clip`。前向
  证据还会用同一 logits 在 fp32 重算逐 token 交叉熵：若 logits
  有限而重算损失非有限、目标越界、目标概率为 0（log(0)/下溢）
  或 `aux_loss` 非有限，findings 会分别指向标签越界、数值下溢与
  路由归一化/除零，避免把所有 NaN 都归因于梯度爆炸。
- **坏案例分析**：在验证集固定窗口上流式统计全部 token 的
  损失/置信度/margin（每 token 约 10 字节紧凑缓冲，不物化窗口
  矩阵、不做随机下采样，token 准确率与 `evaluate` 同口径），
  top-N 最差样本用「每批 top-N + 跨批截断」精确维护，记录上下文、
  目标/预测字节与混淆计数，按四类原因归类；全量错误 token 的
  类别份额（`error_category_counts/shares`）作为改进线索的主证据，
  避免「最高损失样本恰好同类」把结论带偏；完整报告写入
  `out_dir/bad_cases.json`。错误率另按序列位置（开头/后段）与
  训练目标频率（稀有/高频）分层，区分前缀建模、数据覆盖与模型
  盲区；top-N 案例带 `position`、`window_repeat_count`，并汇总
  `top_cases_unique_windows` 与 `top_cases_window_concentration`，
  单一窗口占比 ≥60% 且案例数 ≥3 时提示这些案例不是彼此独立的
  失败模式。测试集解封的唯一一次评估会附带 test 坏案例。
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
  batch size、数据增强（当前为无增强，显式记录）、整体数据指纹
  `data_sha256` 与逐 split 内容指纹 `split_sha256`、
  模型结构（含分支几何 `k1/k2`、路由宽度 `top_k`、均衡损失
  系数 `balance_coef`、`dense_fnn_hidden`、`residual_scale`）、
  评估协议与选型口径（`model_selection` 与 `selection_metric`：
  新运行为验证集选型、无记录的旧运行回退标记为最后一步；按
  loss 选型与按宏观 F1 选型的运行不能混在一起比较），以及
  有效训练长度与
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
  优化器类受控变量检查；但数据与评估口径（`data_sha256`、
  `split_sha256`、`seq_len`、切分比例、`batch_size`、
  `eval_batches`、数据增强、
  群组切分/去重/标签噪声清洗设置）仍通过
  `cross_family_protocol_check`
  逐项核对，避免把不同数据/窗口上的 ngram 指标当作基线证据。
- **统计可靠性**：`--seeds 42 43 44` 对同一命令行配置跑多个
  随机种子，输出写入 `<out>/seed<SEED>`，结束后自动给出每个
  实验单元的 Mean ± Std（样本标准差）与单因子对比检验：种子
  集合相同用配对 t 检验（同种子 = 配对重复），否则用 Welch
  t 检验，并汇报 p 值、「噪声比」（|Δ|/合并标准差，即
  Cohen's d）与 Δ 的
  95% 置信区间（t 分布临界值；区间极宽说明现有种子数只能
  排除「差异为 0」，还不足以定位效应大小）。噪声比在小样本下
  系统性高估真实效应（每臂 3 个种子约 13%，配对口径 df=2 约
  43%），因此同时给出 Hedges g = J·d（`hedges_g`，
  J ≈ 1−3/(4·df−1)，`_hedges_g_correction`），展示层写作
  「噪声比=0.20（Hedges g=0.11）」；`--compare-baseline` 的
  直接对比同口径补上噪声比与 g。每个实验单元的
  均值本身也给出 t 分布 95% 置信区间（`<metric>_ci95_low/high`
  与 `ci95` 字段，聚合表中紧跟 Mean ± Std 展示）：std 只描述
  种子间离散度，不是均值的不确定度，两者混用会在 n 很小时
  严重高估精度；独立种子数 < 2 时不给区间，避免把点估计
  伪装成零宽度的精确估计。效率字段（`costs` 里的参数量、FLOPs、
  推理延迟、训练步耗时、峰值显存）同样按 t 分布给出均值的
  95% CI（`ci95_low`/`ci95_high` 与 `ci95` 子块，n<2 或 std
  不可用时不给），报告用 `[efficiency-ci]` 行展示推理延迟与
  训练步耗时的区间：计时字段的区间宽度与均值同量级时，效率
  差异可能只是测量噪声（倍率本身的区间仍由 `ratio_ci95`
  给出：对数尺度 delta 方法 + Welch-Satterthwaite 自由度与
  t 临界值，`ci_method`/`ci_df`/`ci_t_critical` 一并落盘；两臂
  各 3 个种子时 t 临界值约 2.8，而不是正态近似的 1.96）；
  参数量/FLOPs 由结构决定（std=0），区间退化为点估计。
  计时字段若离散度过大或倍率区间跨 1，会被列为
  `timing_noise_fields`，不单独参与 `cost_verdict` 的材料成本
  判定（结构字段仍参与），避免单次计时把显著涨点误判成
  `gain_with_cost`；若全部字段都被标为噪声则退回全部倍率，
  不静默丢掉成本警告。
  聚合表的 `n(seed)` 列在独立种子数少于
  运行数时写成 `4(3)`，读者不必回到警告里去数伪重复。显著性检验要求
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
  3~5。**差分缺失**：缺判定指标的运行被完整案例分析排除，若
  两臂被排除的运行数不同，对比就不再来自同一批运行
  （`differential_missingness` 字段与展示层警告），此时差异
  可能来自缺测机制而非方法本身。数值失败运行（NaN/Inf、参数或
  优化器状态污染、`curve_status=diverged` 或有限发散
  `diverging`）不进入均值、检验与消融覆盖，并单独列出。
  **配对无分布假设检验**：同一批种子上的配对差值还做
  符号翻转（randomization）检验，n≤12 精确枚举、n>12 固定
  种子 Monte Carlo；它对差值正态性不作要求，与配对 t 检验
  互为稳健性对照。两套检验经同一 Holm 家族校正后结论不一致
  时展示层明确提示「显著性依赖正态性假设或检验功效，不能
  作为稳健结论」；`significant_robust` 要求两者同时显著。
  精确检验同时记录最小可达双侧 p 值（2/2^n），n<5 时该值
  本身大于 0.05，展示层提示该设计在数学上无法达到显著，
  不能把「不显著」解读为「无效应」。**leave-one-out 稳健性**：
  逐一去掉一个种子重算效应，若方向翻转或归零，提示结论可能
  由单个种子驱动；这是描述性检查，不替代显著性检验。
  **不显著 ≠ 等价（功效与等价性口径）**：显著性检验
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
  所需种子数，区分「效应太小」与「种子太少」。每个对比还
  记录在当前精度下主张等价所需的最小对称界
  `equivalence_margin_needed` = |Δ̂| + t_{1−α}(df)·se（及相对
  基线均值的 `equivalence_margin_needed_relative`），并用
  `equivalence_blocker` 把「TOST 未通过」拆成两种不可混为一谈
  的情况：`effect_exceeds_margin`（点估计本身已超出声明界，
  增加种子只会更确定地拒绝等价）与 `insufficient_precision`
  （点估计在界内、噪声半宽把置信区间推出界外，属精度不足，
  额外给出近似所需种子数 `equivalence_seeds_to_achieve`）。
  展示层对后者单独发 `[warn]`，明确「TOST 未通过 ≠ 两者不同，
  应加种子而不是放宽等价界」，避免把精度不足的检验失败读成
  方法之间确实存在差异。**功效口径与家族校正一致**：
  `mde_80`、`power_at_effect`、`seeds_to_detect` 与所需等价界
  原本都按未校正的 α=0.05 计算，而 `significant`/`equivalent`
  用的是 Holm 校正后的 p 值——家族越大，「能检出/能主张等价的
  效应」被说得越乐观。`_attach_family_power` 按 Holm 第一步
  阈值 α/m（Bonferroni，Holm 阈值下界）补算同口径保守量：
  `significance_family_size`、`alpha_bonferroni`、
  `mde_80_bonferroni`、`power_at_effect_bonferroni`、
  `seeds_to_detect_bonferroni`、
  `equivalence_margin_needed_bonferroni`（家族 m=1 时与未校正
  口径相等）。报告的 `[power]` 行给出家族规模与 α/m，展示层
  在「不确定」结论后补一句保守口径的 MDE 与所需种子数，种子
  规划按保守口径而不是未校正口径。**所需种子数是插件估计**：
  `seeds_to_detect` / `equivalence_seeds_to_achieve` 都把当前
  种子间标准差 s 当成真实 σ 代入 se=σ/√n 外推，而 s 自身也是
  估计——正态假设下其相对标准误 ≈ 1/√(2(n-1))（n=3 时约 50%），
  所需种子数 ∝ σ² 会把该不确定度平方放大。报告以
  `seeds_to_detect_basis="sd_point_estimate"` 登记口径，并给出
  `seeds_to_detect_sd_relative_se`；展示层给出全局提示，规划
  种子时要留余量。等价界必须
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
  **评估窗口不确定性**：final val 与唯一一次 test 评估默认
  额外报告评估窗口 bootstrap 95% 置信区间（loss / ppl /
  macro NLL / token accuracy / macro 与 weighted F1 /
  macro 与 weighted precision 与 recall，`--eval-bootstrap`
  控制重采样次数，0 关闭）。它量化「换一批同分布评估窗口」的抽样
  波动，与跨种子方差是两种不同的不确定性来源；窗口数为 1
  或指标退化时区间退化为点估计而不是虚构精度。窗口重叠比例
  （`*_eval_window_overlap_fraction`）已随指标落盘；窗口即便
  不重叠也可能沿评估序列正自相关（同主题/同类块相邻），而
  bootstrap 把窗口当 i.i.d. 抽样单位，正自相关会让区间偏窄。
  因此报告额外给出窗口 loss 率的 lag-1 自相关 `window_lag1_`
  `autocorrelation` 与 AR(1) 有效窗口数 `window_effective_`
  `windows` = n·(1−ρ)/(1+ρ)，并在 ρ̂ ≥ 0.3（`_EVAL_WINDOW_`
  `AUTOCORRELATION`，阈值低于此值时样本 ρ̂ 的抽样噪声与调整
  量同量级）时按方差膨胀因子 √(n/n_eff) 以区间中点为中心
  等比放宽全部四个区间（`window_ci_inflation`；只放宽不收窄，
  负相关不用于把区间做窄），展示层对受影响的 split 发
  `[warn]`，避免放宽后的区间仍被当成窗口独立前提下的宽度。
  逐窗口配对比较的 p 值同样不能把重叠/正自相关窗口当独立
  样本：除原始 `window_p_value` 外，另给 AR(1) 有效窗口数
  校正的 `window_p_value_ar1`（标准误乘 √(n/n_eff)，自由度
  n_eff−1，只放宽不收窄）；n_eff<3 时不给校正 p 值并标记
  `effective_windows_below_three`，而不是继续报原始显著性。
  **AUC 覆盖率**：AUC 需要保留逐 token 概率，默认最多
  100,000 个评估位置（`_AUC_MAX_TOKENS`）。超过上限时不再
  截取评估序列前缀（会让后段窗口系统性缺席），而是在全局
  评估位置上等距抽样；`*_auc_tokens_available`、
  `*_auc_coverage_fraction` 与 `*_auc_sampling`
  （`full`/`uniform_stride`）随指标落盘，读者能直接判断
  AUC 覆盖了多大比例的评估位置。
  **选型偏差的无偏口径**：final 的 val 指标来自验证集选中的
  checkpoint，同一批数据既选型又判定会产生 winner's curse。
  `--compare` 对所有周期验证实际计算的指标（loss / ppl / bpb /
  accuracy / micro、macro、weighted F1 / precision / recall）
  生成 `<metric>_at_common_step` 派生指标：在各运行最后一个
  周期验证步的最小值处精确取值或线性插值，不经过选型，可作为
  `--compare-metric` 做聚合、显著性检验与因子效应。PR-AUC /
  lift / ROC-AUC 默认只在最终评估计算；把它们选为
  `--selection-metric` 或显式传 `--periodic-auc` 后，周期验证
  也按相同固定窗口与排序口径记录，因而同样有
  `<metric>_at_common_step` 无偏口径。极度不平衡时项目建议
  以 PR-AUC 判定——若判定指标只有 final 版本，它恰好是最容易
  被 checkpoint 选型抬高、却没有无偏复核的那个。没有任何运行
  记录某个周期排序指标时跳过该项，不把「没算」写成「缺失」；
  路由诊断只在最终评估计算，不伪造该口径；旧运行缺周期指标
  时显式列入未覆盖而不是虚报。
  **确定性基线**：ngram 的计数与固定窗口评估不消耗随机性，
  换 seed 重跑只是同一估计的复制，不是独立重复。聚合层按
  `deterministic` 标记把同一单元的多个 seed 折叠为 1 个观测
  （`effective_seed_count=1`、`val_*_std=None` 而非人为 0），
  检验层不会因零方差制造假显著；展示层提示其不确定性应来自
  评估窗口 bootstrap 或多次时间切分，而不是增加种子数。若多个
  确定性 seed 运行给出不同指标，`deterministic_conflict_cells`
  点名该单元：确定性管线本应逐位一致，差异来自数据/评估而非
  随机性，不能按种子重复解读。
- **基线合理性（稻草人守卫）**：`--compare-baseline MODE`
  在给出相对差值之外，还核对基线本身是否值得战胜：基线在
  主指标上必须真的优于无上下文 unigram 参考（各运行
  `config.json` 的 `reference_baselines`；PR-AUC / ROC-AUC
  同样有常数预测器参考——纳入类平均 prevalence 与 0.5，
  PR-AUC lift 的参考恒为 0，切换对比指标后守卫不会静默
  失效），不能劣于对比集合
  里调优好的 n-gram 统计基线，种子数应 ≥3，且必须带有调参
  证据（`config.json` 的 `baseline_tuning` 来源记录；统计基线
  自身在 val 上做过 阶数 × 平滑系数 网格调优，视为已调参）；
  否则以 `[straw-man]` 明确标记「战胜没调过参的弱基线不能
  证明复杂模型的收益，先用 `--suite-baseline-tuning` 把基线
  调强」。调参依据的指标与当前对比指标不一致时不打印无条件
  的 OK，而是显式警告「不能证明基线在该指标上被调强」
  （调在 val_loss 上不能证明 macro F1 也强）。
  对比集合没有 ngram 时显式提示简单基线守卫退化为只看
  unigram 参考；确定性 ngram 的多个 seed 不计入独立种子，
  也不建议用「多种子重跑」估计其不确定性，其误差线应来自
  评估窗口 bootstrap。ngram 与神经运行的数据指纹、切分
  比例、评估窗口集合或数据增强不一致时，「基线强于 ngram」
  的判断被跳过并显式列出 `cross_family` 违规项。
  ngram 自身的 (order, alpha) 网格调参也会在 val 上产生
  winner's curse：点估计最优若只领先微小一截，可能只是窗口
  抽样噪声。对 loss / ppl / bpb，候选与点估计 winner 做逐窗口
  配对差，正自相关时按 AR(1) 有效窗口数放宽标准误；配对均值
  差不超过一个标准误的候选按「更小 order / 更小 alpha」收敛
  （`ngram_tuning.selection`，记录点估计最优、最终选择、并列
  集合与阈值）。分类/AUC 指标没有同口径逐窗口 loss，仍按点
  估计选择并显式标注。
  **调参对称性**：基线做了学习率网格、挑战臂却使用默认
  lr 且没有独立调参来源时，报告单独提示：
  落败不能排除调参不足，胜出只是保守下界。
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
  报告本身不是最终基线，test 集全程尘封。调参网格只扫学习率，
  不改变 dense 容量；容量是否与挑战模型等参数由
  `baseline_capacity` 另行核对，需要时用 `--dense-fnn-hidden`
  显式补等参数基线。
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
  绝对值当准确率读。lift 以 `val_pr_auc_macro_lift` 作为
  一等的对比/选型/预注册指标落盘（随机水平恒为 0），避免
  跨数据集直接比较随 prevalence 缩放的 PR-AUC 绝对值。
  四项检查的结构化结果
  （`hypothesis_registry`、`baseline_adequacy`、
  `imbalance_advisory`）随 `comparison_analysis.json` 落盘。
- **假设预注册（方向 / 最小效应 / 主指标）**：可检验的假设
  不仅要写判定指标，还应在训练前写死「预期方向」与「最小有
  意义效应」，否则看到结果后可以任意解释方向、或从多个指标
  里挑显著的那个（指标选择偏差）。入口支持：
  - `--primary-metric`：预注册主判定指标（必须是评估管线
    真正计算的对比指标，且在 `--hypothesis` 文本中被点名）；
  - `--hypothesis-direction decrease|increase`：实验臂相对
    对照臂，主指标的**数值**预期下降/上升；
  - `--hypothesis-min-effect REL`：最小有意义相对效应
    （相对对照均值，如 0.02 = 2%）。
  三者写入 `config.json` 的 `hypothesis_preregistration`。
  `--compare` 阶段据此输出 `hypothesis_alignment`：
  观测方向与预注册显著相反 → `refuted`（假设被数据推翻，
  不能事后改写解释）；方向一致但幅度未达最小效应 →
  `below_declared_min_effect`（统计显著也不等于支持假设）；
  声明了相对最小效应但对照均值为零/近零或缺失 →
  `min_effect_undefined`（比例阈值没有可解释的尺度，结论
  不可判定，不能静默算作达到最小效应）；方向一致且达到
  最小效应、Holm 校正后显著 → `supported`；其余为
  `not_significant` / `direction_mismatch_not_significant`。
  分析指标与运行预注册主指标不一致时单独提示
  `[metric-shopping]`（事后换指标），旧运行没有预注册记录时
  只提示「无法排除指标选择偏差」。
  另外，单标签多分类的 F1 / precision / recall 必须注明
  macro / micro / weighted 口径（三者结论可能显著不同），
  假设只写「F1 提升」会被单独点名。
- **选型指标与判定口径对齐**：模型 checkpoint 默认按
  `val_loss` 选型并早停，但若预注册主指标是 macro F1 等
  分类指标，按 loss 选出的模型并不是按假设口径选出的。
  `--selection-metric` 可把验证集选型/早停改为周期验证
  实际计算的任一对比指标（`val_macro_f1`、
  `val_recall_weighted`、`val_pr_auc_macro` /
  `val_pr_auc_macro_lift` / `val_roc_auc_macro` 等）。AUC
  默认仍只在 final/test
  计算；被显式指定为选型指标时，训练循环会在每个验证点
  按与 final/test 相同的固定窗口和排序口径计算，避免
  「以 PR-AUC 判定却按 val_loss 选模型」的口径脱节。
  路由诊断与等计算量/共同步数派生指标仍不能用于选型。
  未显式指定时，若 `--primary-metric` 可在周期验证中
  计算则自动跟随，否则回退 `val_loss` 并警告口径不一致。
  指标方向由
  `metric_higher_is_better` 决定（loss/ppl 越低越好，
  其余越高越好）；选型指标、最优值、选中步与选中点的
  val_loss 一并写入 `best.pt`、`metrics.jsonl` 的 final
  记录与 `config.json` 的 `model_selection`，并作为受控
  变量参与 `--compare` 核对（按 loss 选型与按 F1 选型的
  运行不能混在一起比较）。`--compare` 再次核对运行记录的
  `selection_metric` 与当前 `--compare-metric`：当前指标本身
  可用于选型却不一致时，`selection_metric_mismatch` 明确警告
  「最终模型不是按假设的判定口径选出的」；等计算量 loss
  等只能离线派生的指标不触发该警告。恢复训练时选型指标按
  检查点恢复，避免中途换指标重新解释历史评估点。
- **基线欠训与容量混淆守卫**：`baseline_adequacy` 除核对
  「基线强于 unigram 参考」「基线有调参来源」「不劣于调优好的
  n-gram」「种子数」之外，还检查：
  - 基线运行的曲线诊断（`curve_analysis.status`）为
    `not_converged` / `underfitting` / `diverged` 时点名
    「欠训的弱基线不说明方法更强」，要求在相同训练配置下
    先调强基线；
  - 挑战模型相对基线的参数量 / 前向 FLOPs 超过
    1.5× 阈值时提示容量混淆，要求报告
    等参数（`--dense-fnn-hidden n_experts×expert_hidden`）
    或等计算（`smelt` / `local_base`）对照，区分容量与机制；
  - dense 基线默认 `dense_fnn_hidden=None` 解析为
    `top_k×expert_hidden`，这是等激活计算而非等参数容量；
    挑战模型参数更多时，`baseline_capacity` 给出等参数宽度，
    提示用 `--dense-fnn-hidden n_experts×expert_hidden` 补
    等参数基线；
  - n-gram 是确定性对照：多个 seed 折叠为 1 个观测，
    `ngram_deterministic=True`，不再建议多种子重跑；需要
    误差线时对评估窗口做 bootstrap 或多次时间切分。
  `--suite-baseline-tuning` 的 winner 候选若仍 `not_converged`，
  调参报告本身也会给出「限制基线的是训练配置而非学习率」的
  issue。
- **全因子消融**：`--suite-ablation` 运行「延迟合并 × 均衡
  损失」全因子（`local_base` 与各延迟合并方案 × `balance_coef`
  =0/当前值；(1+N)×2 格分别对应「无 / 仅延迟合并 / 仅均衡
  损失 / 两者」）。默认覆盖项目提出的全部延迟合并方案
  （A 与 B，共 3×2=6 格）：每个创新点都必须证明各自有用，
  只跑 A 的组合不足以支撑 B 的结论；未声明时按项目声明的
  全部方案核对，不用「运行集中实际出现过」替代项目声明。
  确需阶段性部分消融时用 `--ablation-modes A|B` 并在套件中
  显式加 `--allow-partial-ablation`，报告仍记录 `partial` 与
  未覆盖方案，不会把部分覆盖说成完整。全部组合覆盖后才能同时
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
- **主效应与交互的统计推断**：覆盖齐全只证明「组合跑过」，
  不证明「各自有用」。`factorial_effects` 对每个延迟合并方案
  与 `local_base` 的 2×2 设计计算：延迟合并主效应
  `½(M_off−L_off)+½(M_on−L_on)`、均衡损失主效应
  `½(L_on−L_off)+½(M_on−M_off)`、交互
  `(M_on−M_off)−(L_on−L_off)`，以及四个简单效应。所有效应
  在**共同随机种子**上按同一线性组合计算后做配对 t 检验，
  同一家族内做 Holm 校正并给出 95% CI 与 MDE；只有校正后
  显著且方向符合指标改善方向，才标记为支持该组件。每个效应
  还按预先声明的等价界做 TOST：等价界相对该效应所比较的
  对照单元均值（主效应取两侧对照均值，简单效应取对应水平
  的对照格），TOST p 值在因子效应家族内做 Holm/BH 校正，
  只有校正后仍通过才可主张「无实质差异」；显著与等价可以
  同时成立，二者回答的是不同问题。交互显著只说明一个因子的
  效应随另一个因子变化，不自动解释为有益。
  各 2×2 单元格的共同种子数、种子集合是否对齐与重复种子都会
  核验并写入 `comparison_analysis.json`；配对种子不足时只报
  点估计，不能声称显著。
- **消融中的计算量控制**：B 直通整个中间区域，FLOPs 高于
  `local_base`；即使组合齐全，B 的主效应也会混入额外计算。
  覆盖检查从运行记录读取前向 FLOPs（缺省回退参数量），
  倍率达到 `1.5×` 时以 `[confounded]` 点名；A 的
  `k1×k2 > n_experts` 时 `local_base` 第二轮宽度被截断，
  即使实测倍率未达阈值也会以 `width_confounds` 点名。
  两种情况都要求用
  `--compare-metric val_loss_at_matched_flops` 在等训练
  计算量点复核。分析入口会自动在该派生指标上重跑一遍因子
  主效应/交互（单独成族并标注只适用于共同计算量点），
  避免只给警告、不给出可执行的复核结果；同时存在其他受控
  变量违规或结构混杂时，已显著的因子效应会从「支持该组件」
  降级为「不能归因」。
- **汇总报告**：套件运行结束后把受控变量检查、聚合统计、
  单因子对比、消融覆盖、因子主效应/交互与混杂降级结果写入
  `<out>/comparison_analysis.json`；`--compare` 只读运行
  目录、不写文件。

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
（泄露级审计问题拒绝训练/拒绝解封测试集）；假设预注册：
`--hypothesis`（必填）、`--rq`、`--primary-metric`、
`--hypothesis-direction decrease|increase`、
`--hypothesis-min-effect REL`（后三者配套使用）。

结果分析相关参数：`--early-stop-patience`（默认 0，关闭早停）、
`--early-stop-min-delta`、`--selection-metric`（验证集选型/
早停指标，默认 `val_loss`；预注册主指标可周期计算时自动
跟随）、`--bad-cases`（默认 10，0 关闭）、
`--eval-bootstrap N`（final/test 评估窗口 bootstrap 重采样
次数，默认 500，0 关闭；输出 loss/ppl/macro NLL/
token accuracy/macro 与 weighted F1、precision、recall
的 95% 置信区间）、
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
