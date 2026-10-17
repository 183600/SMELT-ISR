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
- **数据审计**：训练前对 train/val 自动做值域校验、字节直方图
  分布偏移（JS 距离）与滑窗重复内容扫描（有界采样），警告与统计
  写入 `config.json` 的 `data_audit`；解封测试集时追加含 test 的
  审计。恢复训练时若数据指纹或划分比例与存档不一致会明确提示。
- **群组数据**：当前语料为单一字节流，不存在群组结构；若替换为
  同一用户/物体产生多条样本的数据集，必须按群组整体划分，
  train/val/test 任意两侧之间不得交叉。

## 代码组织

- `Config`：模型结构与损失配置，在入口处校验。
- `RMSNorm`、`CausalSelfAttention`：基础层。
- `Expert`、`SparseMoE`：专家与稀疏路由，含 `route` / `dispatch` / `merge`。
- `TransformerBlock`：注意力步骤、MoE 步骤，以及分支与两轮路由逻辑。
- `LanguageModel`：按模式组装完整前向。
- `ByteDataset`、`evaluate`：字节级数据管线与按 split 的固定窗口评估。
- `audit_data_splits`、`_window_signature_scan`：数据审计（值域、
  分布偏移、重复窗口）。
- `run_sanity_overfit`：训练前的过拟合小批量 sanity check。
- `save_checkpoint`、`load_checkpoint`：断点存档与恢复。
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
  64 位，避免 uint8 回绕误报，前向证据不足时补跑一次诊断前向）；
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
- **早停**：`--early-stop-patience` 启用；NaN 验证损失不算「改善」；
  `--resume` 时从 `metrics.jsonl` 重建早停状态，保证续跑语义一致。
- **运行对比**：`--compare RUN_DIR...` 只读运行目录，汇总各模式
  在相同数据上的指标；同时输出受控变量检查、按实验单元聚合的
  Mean ± Std、单因子显著性检验与消融覆盖报告（见下节）。

## 对比实验与消融协议

对比结论只有在「除待验证模块外，其余设置完全一致」时才可信。
本项目把这一纪律落到代码里：

- **受控变量协议**：每次训练把优化器（AdamW 及 betas/eps）、
  学习率与调度、warmup、训练步数、weight decay、梯度裁剪、
  batch size、数据增强（当前为无增强，显式记录）、数据指纹、
  模型结构与评估协议写入 `config.json` 的
  `comparison_protocol` 字段。`--compare` 默认要求这些协议项
  在声明因子之外逐项一致；不一致会以 `[violation]` 明确列出
  （如某个运行换了优化器、调了学习率或训练轮数），提示结论
  无法归因到方法本身。
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
  t 检验，并汇报 p 值与「噪声比」（|Δ|/合并标准差）。种子数
  少于 3 的实验单元会被标记为统计可靠性不足；p 值仅供参照，
  关键结论应基于预设对比且种子数 3~5。
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
  承担，其种子与状态随检查点存档。
- 全新训练开始前默认运行过拟合小批量 sanity check：
  同一模型结构在固定小批次上训练，关闭辅助均衡损失、weight decay、
  梯度裁剪与 warmup，要求训练损失迅速趋近 0 或下一 token 准确率
  接近 100%。检查内部固定种子并在结束后恢复 RNG 状态，不影响主训练
  轨迹；失败时以非零状态码退出，不开始全量训练。这是必要不充分
  检查，通过后仍需核对中间结果。可用 `--sanity-only` 单独运行、
  `--skip-sanity-check` 跳过。
- 每轮实验的代码版本、运行环境、超参数与随机种子、数据指纹、
  sanity check 结果写入 `config.json`；训练曲线与最终结果写入
  `metrics.jsonl`，全程自动记录。

## 训练入口

```
python experiment.py --data <语料文件> --out <输出目录> --mode <模式>
```

常用参数：`--steps`、`--batch-size`、`--lr`、`--warmup-steps`、
`--grad-clip`、`--eval-interval`、`--save-interval`、`--resume`、
`--grad-checkpoint`、`--compile`、`--seed`、`--note`、`--sanity-only`、
`--skip-sanity-check`、`--sanity-batch-size`、`--sanity-steps`、
`--sanity-lr`、`--sanity-loss-threshold`、`--sanity-acc-threshold`、
`--val-fraction`、`--test-fraction`、`--evaluate-test`。

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
