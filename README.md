# SMELT-ISR

**中文** | [English](README_en.md)


> **SMELT with Independent Second Round Routing (Delayed-Merge)**

## 📖 项目简介

SMELT-ISR 是基于 [SMELT](https://arxiv.org/html/2609.01343v1) (Sparse MoE Transformer, middle layers Loop Twice) 架构的路由改进实验项目。

原版 SMELT 在循环执行中间层时，每轮 MoE 的专家输出会立即合并。本项目提出了一种 **延迟合并** 机制，核心思想如下：
1. **第一轮路由后不合并**：Top-$k_1$ 专家的输出保留为独立分支，不进行求和或拼接。
2. **第二轮独立路由**：针对每个第一轮分支的状态，分别独立进行第二轮 Top-$k_2$ 专家路由。
3. **最终聚合**：只有在完整第二轮结束后，才跨第一轮分支进行结果合并。

该结构旨在探索延迟合并是否能提高专家组合的表达能力，并验证不同分支是否能形成完全独立的计算轨迹。

## 🚀 核心特性

- **四种模型模式实现**：
  - `smelt`: 原始 SMELT 架构（中间层循环两次，每轮立即合并）。
  - `A`: 局部延迟合并（仅在同一 MoE 层内保持专家分支独立，第二阶段后合并）。
  - `B`: 全局延迟合并（第一轮产生独立分支后，整个中间 Transformer Block 保持分支独立，包含 Attention 和 MoE，第二轮结束后才合并）。
  - `local_base`: 方案 A 的立即合并对照组，用于隔离“延迟合并”本身带来的收益。
- **严谨的自测模块**：验证梯度流通、自回归因果性、分支间的信息隔离，以及边界条件下的数学等价性（如 $k=1$ 时方案 B 与 SMELT 等价）。
- **完整训练基座**：支持断点恢复、梯度裁剪、负载均衡损失、学习率调度和详细的实验指标记录。
- **控变量对比与统计可靠性**：每轮训练记录受控变量协议（优化器、学习率策略、训练步数、数据增强、数据指纹、早停/AMP/设备等数值设置），`--compare` 自动核对「除声明因子外设置是否完全一致」并识别重复种子的伪重复；`--seeds` 支持多随机种子运行并汇报 Mean ± Std 与显著性检验，报告同时给出 80% 功效的最小可检出效应（MDE）与 TOST 等价性判定（`--equivalence-margin`，默认相对基线均值 1%），「不显著」不会被误读成「等价」；`--suite-ablation` 提供 (1+N)×2 全因子消融（默认覆盖 A/B 两个延迟合并方案），自动检查消融组合是否完整。

## 🛠️ 环境依赖

- Python 3.10+
- PyTorch 2.x

```bash
pip install "torch>=2.2"
```

## 📦 代码使用说明

### 1. 运行结构自测
在开始实验前，建议运行内置的自测模块，以确保不同模式的结构正确性、分支独立性与等价性；自测还包含「同一随机种子两次运行逐位一致」的端到端可复现性检查，以及梯度探针的负例校验。
```bash
python experiment.py --self-test
```

### 2. 准备数据
代码默认使用字节级 tokenizer 以方便快速验证。你需要准备一个 UTF-8 编码的纯文本文件（如 `corpus.txt`）。

数据管线严格防止数据泄露：**先按时间先后顺序划分，再做任何处理**。
语料默认切成 `train`（前 90%）/ `val`（随后 5%）/ `test`（最后 5%）
三段，切分连续、互不重叠、全程不随机打乱（时序数据必须用过去训练、
未来验证和测试）；训练阶段的批级 shuffle 由随机窗口起点承担，不改变
split 边界。当前字节级管线不涉及标准化/归一化/缺失值填充，若扩展为
连续值特征，所有统计量（均值、方差、最值等）必须只在 train 上
`fit`，再 `transform` val/test。

`val` 用于周期性验证、挑超参数、早停与模型选型；`test` 训练期间全程
尘封，只有显式传入 `--evaluate-test` 才会在训练完全结束后评估一次，且
同一次运行拒绝重复评估。切分比例可用 `--val-fraction` / `--test-fraction`
调整。每次运行在训练前都会自动做数据审计（token 值域、无法 UTF-8 解码
的字节与 NUL 字节等文本完整性/标签噪声检查、重复窗口、train↔val 分布
偏移），结果与警告写入 `config.json` 的 `data_audit` 字段；解封测试集
时会追加包含 test 的审计结果。

**群组数据必须整体落在同一侧**：如果语料由多个「群组」拼接而成
（同一文档 / 同一用户 / 同一物体产生的多条记录），用 `--group-separator`
指定群组分隔符（如 `--group-separator '<|doc|>'` 或 `--group-separator '\n\n'`，
支持 `\n`、`\xHH`、`\uXXXX` 等转义）。切分位置会吸附到群组边界，
任何群组都不会被拆到两个 split，train/val/test 之间零交叉；`data_audit`
记录每个 split 的完整群组数。吸附后无法保证每段最小长度时（如单个
群组过大）会显式报错，而不是悄悄拆分群组。

**训练集去重**：`--deduplicate-train-groups`（需配合群组分隔符）会从
train 中整体剔除与 val/test 或更早 train 群组内容完全相同的群组，
val/test 的字节与切分位置保持不变。被剔除内容不会进入神经网络窗口，
也不会进入 unigram / n-gram 统计基线的计数或分布审计；`data_audit`
记录 `effective_bytes` / `effective_windows`，`config.data` 记录实际
切分比例与清洗统计。训练期审计只用切分边界核对 test 实际占比，不读取
test 内容；解封前还会独立核对 test 固定窗口的输入/标签是否逐位对齐。

**默认泄露硬门**：审计一旦发现跨 split 重复窗口或重复群组（泄露级
问题），训练入口会直接拒绝启动；解封测试集时发现同类问题也会拒绝
解封（不写入 `test` 记录）。只有显式传入 `--allow-data-leakage` 才会
放行，且豁免会写入 `config.json` 与对比协议，不能把已泄露的运行当成
无泄露证据。窗口签名扫描在窗口数超过上限时按 stride 等间距抽样，
`data_audit.leakage_scan_coverage` 记录各 split 的采样/总窗口数、stride
与覆盖率；stride>1 时审计与泄露门都会显式写出残余风险——抽样范围内的
「未发现跨 split 重复窗口」不等于「不存在重复」，内容定义分块指纹
（穷尽扫描）可兜底被 stride 漏掉的错位重复。越界 token、超阈值的
UTF-8/NUL 损坏等标签噪声由
`--strict-data-audit` 控制：加上后同样拒绝训练/解封；默认仅警告并继续，
方便先看清数据问题再决定清洗策略。

训练期间每次刷新最优验证损失都会把权重写入 `best.pt`。训练结束后的
`final` 与唯一一次测试集评估都在这份「验证集选定」的模型上进行
（`config.json` 的 `model_selection` 记录选型口径）；若 `best.pt` 缺失
（如从旧检查点恢复），会明确提示并回退到最后一步参数。

```bash
# 训练结束后做唯一一次测试集评估
python experiment.py --data corpus.txt --mode B --out runs/B_seed42 --evaluate-test
```
*(注：正式实验应替换为你的 tokenizer 和数据管线)*

### 3. 开始训练
通过 `--mode` 参数指定运行的结构。

**训练原始 SMELT 基线**：
```bash
python experiment.py \
  --data corpus.txt \
  --mode smelt \
  --top-k 2 \
  --steps 2000 \
  --seed 42 \
  --out runs/smelt_seed42
```

**训练全局延迟合并方案 B**：
```bash
python experiment.py \
  --data corpus.txt \
  --mode B \
  --top-k 2 \
  --k1 2 \
  --k2 2 \
  --steps 2000 \
  --seed 42 \
  --out runs/B_seed42
```

**训练局部延迟合并方案 A 及其对照组**：
```bash
python experiment.py --data corpus.txt --mode local_base --k1 2 --k2 2 --out runs/local_base_seed42
python experiment.py --data corpus.txt --mode A --k1 2 --k2 2 --out runs/A_seed42
```

### 4. 训练前的过拟合 sanity check
每次全新训练（非 `--resume`）开始前，脚本会先抽取一个小批次
（`--sanity-batch-size`，默认 4 条序列），在同一模型结构上做过拟合检查：
关闭负载均衡辅助损失、weight decay、梯度裁剪与 warmup 等一切正则化，
要求训练损失迅速趋近 0 或下一 token 准确率接近 100%（阈值由
`--sanity-loss-threshold`、`--sanity-acc-threshold` 控制）。未通过时会
明确报错并以非零状态码退出，避免在存在梯度未反传、损失计算错误等严重
bug 时盲目开始全量训练。

第一步反向之后还会先做梯度探针：损失必须连在计算图上、所有梯度必须
有限、至少一个参数收到非零梯度、非专家参数不得缺失梯度（专家参数允许
因本批次未路由到而缺失/零梯度）。探针失败会立即停止并给出排查提示
（不浪费后续步数），结果写入 `config.json` 的
`sanity_check.grad_probe` 与 `metrics.jsonl`；探针只是第一道筛查，
通过后仍需按下面的必要不充分说明配合其他检查。

注意这是**必要不充分**检查：通过也不代表代码完全正确，数据增强、特征
对齐、标签映射等 bug 仍可能漏网，还需配合中间结果检查。

```bash
# 只跑 sanity check，不开始训练
python experiment.py --data corpus.txt --mode B --sanity-only --out runs/sanity_B

# 显式跳过检查（已充分验证的前提下）
python experiment.py --data corpus.txt --mode B --skip-sanity-check --out runs/B_seed42
```

检查结果会写入 `metrics.jsonl`（`{"step": "sanity", ...}`）与
`config.json` 的 `sanity_check` 字段；检查内部固定种子并在结束后恢复
RNG 状态，因此是否运行检查不影响主训练的随机轨迹。

### 5. 恢复训练
模型配置会从检查点自动读取，恢复时只需保证数据与总训练步数一致：
```bash
python experiment.py \
  --data corpus.txt \
  --resume runs/B_seed42/last.pt \
  --steps 2000 \
  --out runs/B_seed42
```

## 🧪 实验设计与研究问题

本项目旨在通过以下维度验证“延迟合并”的有效性：

- **RQ1**: 延迟合并是否优于原始 SMELT？（比较 validation loss 和收敛速度）
- **RQ2**: 第一轮分支数 ($k_1$) 与第二轮专家数 ($k_2$) 的分配比例如何影响结果？（如 $k_1k_2 \approx 2k$ 下的不同组合）
- **RQ3**: 第二轮是否会选择与第一轮不同的专家？（分析专家重合度与 Jaccard 相似度）
- **RQ4**: 延迟合并是否改善专家专门化？（分析专家负载分布与路由熵）

RQ3/RQ4 的路由诊断（第二轮 Jaccard / slot 复用 / 分支自复用、第一二轮负载熵与路由熵、死专家数、分支余弦）随每次运行写入 `config.json` 的 `routing_analysis`，并在 `--compare` 中按运行展示、按种子聚合（Mean ± Std）；它们同样进入可对比指标集（`--compare-metric routing_jaccard_branch` 等），因此路由结论可以带跨种子统计与显著性检验，而不是只靠单次运行的定性描述。

### 对照口径与可复现性

- **B 的两轮路由与合并权重**：方案 B 的第一轮中间区域全程使用 $k_1$，第二轮完整中间区域才使用 $k_2$；合并使用第一轮路由得到且 Top-$k_1$ 归一化的固定权重，不会用最终分支状态重新计算合并权重。论文分析或消融中应明确这一点。
- **计算量口径**：方案 B 在第一轮分支创建后按 $k_1$ 倍分支数执行后续 Attention 与 MoE，等参数实验并不等价于等 FLOPs 实验。回答 RQ1/RQ2 时应同时报告按训练步数与按 token/FLOPs 对齐的对比，避免把额外计算误解释为路由机制本身的收益。
- **残差缩放口径**：`residual_scale` 全局作用于所有区域、所有子层的残差更新（循环区域内/外一致）。SMELT 论文对循环区域内的子层残差更新按循环次数 $1/r$ 缩放（循环两次即 1/2）；本基座未实现按区域区分的 $1/r$ 缩放。跨模式对比应保持 `residual_scale` 一致，把全局系数设为 0.5 并不等价于论文口径。
- **负载均衡尺度**：辅助损失中 $f_i$ 按 token 数归一化，而不是按 assignment 数归一化；均匀路由时 $E \sum_i f_i P_i \approx k$，均衡强度随路由宽度线性增大。跨 $k_1/k_2/\text{top-}k$ 比较实验时应按比例折算系数（保持 `balance_coef × k` 不变），否则路由宽度会与均衡强度混杂。`local_base` 第二轮为对齐计算量使用宽度 $\min(n_\text{experts}, k_1 k_2)$，实现会把该轮 aux 按 $k_2/\min(n_\text{experts}, k_1 k_2)$ 折算回 A/B 的单分支宽度；`comparison_protocol` 记录 `balance_width_normalization` 以区分新旧运行。`--compare` 还会核对 `balance_coef × 有效路由宽度`，宽度不同而强度未抵消时给出 `[balance-scale]` 警告。
- **验证/测试协议**：每次评估使用跨评估 split 等间隔选取的固定窗口，窗口集合不依赖随机数，可在不同模型和恢复训练之间逐位对齐；训练期间每次刷新最优验证损失都保存 `best.pt`，训练结束后的 `final` 记录与唯一一次测试集评估都在这份验证集选定的模型上进行（`model_selection` 字段记录选型口径；`best.pt` 不可用时明确提示并回退到最后一步）。测试集默认全程尘封，仅在显式传入 `--evaluate-test` 时解封评估一次（`{"step": "test", ...}`），重复传参会被拒绝，避免测试集变成隐式验证集。final 与 test 记录默认附带基于固定评估窗口的 bootstrap 95% 置信区间（loss / ppl / token accuracy / macro F1；`--eval-bootstrap N` 控制重采样次数，默认 500，0 关闭），用于区分「评估窗口抽样波动」与「训练种子波动」。同一批验证集既用于 checkpoint 选型又用于判定会引入 winner's curse；`--compare` 会为全部周期验证指标生成 `<metric>_at_common_step` 无偏派生口径（loss / ppl / bpb / accuracy / micro、macro、weighted F1 / precision / recall），可在共同训练步数处直接作为 `--compare-metric`。所有结果追加到 `metrics.jsonl`。
- **确定性**：训练入口固定 Python / NumPy / PyTorch(CPU+CUDA) / cuDNN 的随机种子，数据采样由显式 `torch.Generator` 承担（种子与状态随检查点存档），Python / torch / CUDA / NumPy 的 RNG 状态都会随 `last.pt` 存档并在 `--resume` 时逐位恢复；设置 `CUBLAS_WORKSPACE_CONFIG`、要求 PyTorch 优先使用确定性算子；若当前 CUDA 算子没有确定性实现，PyTorch 会给出告警。`PYTHONHASHSEED` 只能影响进程启动后的子进程，未在命令行显式设置时启动会打印一次性提示；严格逐位复现可前缀 `PYTHONHASHSEED=<seed>`。跨硬件架构仍可能因浮点归约顺序产生微小差异，同一台机器上同种子运行应逐位一致（`--self-test` 会做双跑逐位校验）。
- **实验追踪**：每轮实验自动记录代码版本（git commit / 是否有未提交改动）、运行环境（Python / torch / CUDA / 设备）、完整超参数与随机种子、数据指纹（SHA-256、字节数）、训练曲线与最终结果，全部写入 `config.json` 与 `metrics.jsonl`，无需手动记录。
- **受控变量检查**：`config.json` 的 `comparison_protocol` 字段记录优化器（AdamW 及 betas/eps）、学习率与调度、warmup、训练步数、weight decay、梯度裁剪、batch size、数据增强、数据指纹、模型结构、评估协议与选型口径（新运行为验证集选型，无记录的旧运行标记为最后一步），以及**有效训练长度与数值设置**（早停容忍度 `early_stop_patience`/`early_stop_min_delta`、AMP 精度 `amp_dtype`、设备类型、梯度检查点、`torch.compile`）。`--compare` 默认要求这些设置在声明因子（`--compare-factors`，默认 `mode,k1,k2,balance_coef,seed,dense_fnn_hidden`）之外逐项一致：某个运行换了优化器、调了学习率或训练轮数、换了 AMP/设备、或 final 指标口径不同，都会被明确列出，避免把调参收益误当成方法本身的收益。
- **统计可靠性**：`--seeds 42 43 44` 对同一配置跑多个随机种子，自动汇报每个实验单元的 Mean ± Std（样本标准差）与均值本身的 t 分布 95% 置信区间（`<metric>_ci95_low/high` 与 `ci95` 字段；std 只描述种子间离散度，独立种子数 <2 时不给区间）；聚合表的 `n(seed)` 列在独立种子数少于运行数时写成 `4(3)`，把伪重复/确定性折叠造成的功效虚增直接暴露出来；种子集合相同用配对 t 检验、否则用 Welch t 检验，给出 p 值与噪声比（|Δ|/合并标准差）。种子数少于 3 会被标记为统计功效不足；只有 0.2% 的提升而种子间方差有 0.5% 时不会被当成有效提升。**不显著 ≠ 等价**：每个检验过的对比同时报告 80% 功效下的最小可检出效应（MDE，含相对基线比例）、观测效应自身的功效（`power_at_effect`）与按当前种子间波动检出该效应所需的种子数；并对预先声明的等价界（`--equivalence-margin`，相对基线均值，默认 1%）做 TOST 等价性检验。TOST p 值与显著性检验同一家族口径做 Holm（FWER）校正：只有校正后仍通过才主张「无实质差异」，不显著且 TOST 未通过时结论只能是「不确定」（展示层同时打印 MDE 与所需种子数，说明是效应太小还是种子太少）。每个未通过 TOST 的对比还会给出在当前精度下主张等价所需的最小对称界（`equivalence_margin_needed` = |Δ̂| + t·se）并归因：点估计本身超出声明界时标记 `effect_exceeds_margin`（加种子只会更确定地拒绝等价），点估计在界内但噪声把置信区间推出界外时标记 `insufficient_precision` 并给出近似所需种子数（`equivalence_seeds_to_achieve`），报告以 `[warn]` 点名——TOST 未通过不能读成「两者不同」，该加种子而不是放宽等价界；等价界必须事先声明，看到数据后再放宽界可以「制造」等价。同一 TOST 口径也覆盖全因子消融的主效应、交互与简单效应：等价界相对该效应所比较的对照单元均值，p 值在因子效应家族内做 Holm/BH 校正，只有校正后仍通过才可主张「该组件无实质差异」。**伪重复检测**：同一实验单元里同一个随机种子出现多次（复制运行目录等事故）不是独立重复实验，报告会点名重复种子、禁用配对检验并退回 Welch；关键套件（`--suite-modes`/`--suite-ablation`/`--suite-baseline-tuning`）少于 3 个种子会直接拒绝运行，冒烟/调试需显式加 `--allow-underpowered`（豁免随种子计划写入报告）。**差分缺失检测**：某臂的运行缺少判定指标时（如曲线缺失、指标未计算），对比只在有该指标的运行上做（完整案例分析）；两臂缺失数不同会被显式点名（`n_missing_baseline`/`n_missing_variant`/`differential_missingness`），因为此时差异可能来自「哪些运行留下了」而不是方法本身。**配对无分布假设检验**：同种子配对对比同时做符号翻转（randomization）检验（n≤12 精确枚举，n>12 固定种子 Monte Carlo），不依赖差值正态性；两套检验经同一 Holm 校正后结论不一致时明确提示，`significant_robust` 要求两者同时显著；精确检验的最小可达 p 值（2/2^n）写进报告，样本量不足以达到 0.05 时提示「不显著 ≠ 无效应」。**leave-one-out 稳健性**：逐一去掉一个种子重算效应，方向翻转或归零时提示结论可能由单个种子驱动。
- **基线合理性与假设登记**：`--compare-baseline MODE` 在给出差值之外还会核对基线是否值得战胜：基线主指标必须强于无上下文 unigram 参考（各运行 `reference_baselines`；PR-AUC 的参考是其常数预测器水平 = 纳入类平均 prevalence，ROC-AUC = 0.5，因此把对比指标切到 PR-AUC 时稻草人守卫不会静默失效）、不劣于调优好的字节级 n-gram 统计基线、种子数 ≥3、且带有调参证据（`config.json` 的 `baseline_tuning` 来源记录；统计基线自身已在 val 上调优，视为已调参），否则以 `[straw-man]` 标记（战胜没调过参的弱基线不能证明复杂模型的收益，先用 `--suite-baseline-tuning` 把基线调强）；调参口径与当前对比指标不一致时不再打印无条件的 OK，而是显式警告「不能证明基线在该指标上被调强」；对比报告同时按模式列出 `--hypothesis`/`--rq` 登记的假设（缺假设、假设没写明判定指标、或提到的指标当前不可判定——如路由类指标而该运行没有 `routing_analysis` 记录——的运行都被点名，统计基线除外；单次训练启动时也会提示「假设未写明判定指标」，并把 `hypothesis_metric_mentions` 写入 `config.json`），并按 `val_majority_accuracy` 提示类别不平衡风险：多数类过半时 accuracy/micro 参考价值有限，≥99.9%（约 1:1000）的极度不平衡按单独分级处理——PR-AUC 必须与随机水平（≈平均 prevalence，评估时随 `val_pr_auc_macro_chance` 记录）对照看 lift，ROC-AUC 的随机水平仍是 0.5、会给出过于乐观的分数。各项检查随 `comparison_analysis.json` 落盘。ngram 与神经运行的数据指纹、切分比例、评估窗口数、batch size 或数据增强不一致时，跨 family 口径检查会列出 `[violation]` 并跳过「基线强于 ngram」的判断。
- **简单基线的调参协议**：`--suite-baseline-tuning`（配合 `--seeds`）对简单基线做预注册的学习率网格搜索（`--baseline-tuning-lrs`，默认 `1e-4 3e-4 1e-3`，`--baseline-tuning-mode` 选模式）：候选之间只允许学习率不同（steps/batch/数据/结构/调度等全部继承同一条命令行），在 val 上按 Mean ± Std 选择 winner——先取最优均值候选，再把与最优均值差距小于种子波动（两侧 std 的较大者）的候选视为无法区分，从并列集合中取更小 lr 的保守解，并列集合与选择方式（`winner_selection`/`tied_lrs`）如实写入报告；winner 落在网格边界（最小/最大候选）时提示最优可能位于网格之外、建议扩展候选后重跑。报告写入 `<out>/baseline_tuning.json`；后续基线训练用 `--baseline-tuning-from <报告>` 核对模式/winner lr/steps/数据指纹与**完整训练协议**（warmup/调度/weight decay/梯度裁剪/数据划分/评估窗口等）逐项一致（旧版报告缺协议字段时显式拒绝，不默认为通过），并把来源写入 `config.json` 的 `baseline_tuning` 字段——这是「基线被调过参」的唯一证据。test 集全程尘封。
- **消融完整性**：`--suite-ablation` 运行「延迟合并 × 均衡损失」全因子（`local_base` 与各延迟合并方案 × `balance_coef`=0/当前值）；默认覆盖项目提出的**全部**延迟合并方案（A 与 B，(1+N)×2 = 3×2 六格），每个创新点都必须证明各自有用——只跑 A 的组合不足以支撑 B 的结论。覆盖检查会列出缺失的组合、点名结构混杂（各单元 k1/k2/top_k/residual_scale 或均衡强度不一致）与伪重复；显式声明本轮必须覆盖哪些方案用 `--ablation-modes A B`。`--suite-modes` 用于严格控变量的跨模式对比（如 `smelt local_base A B` 一起跑）；套件结束后汇总分析写入 `<out>/comparison_analysis.json`。
- **结果分析（曲线 / NaN / 坏案例 / 效率）**：
  - **在线曲线监控**：不只训练结束后看最终数值，每个周期验证都会复用 `analyze_training_curve` 在线判定；状态迁移（如 healthy→overfitting、发散）立即打印建议并写入 `{"step": "curve_monitor", ...}`，过拟合→加正则化/增大数据/`--early-stop-patience` 早停，两者都不降→查学习率、优化器与数据管道是否对齐；验证损失尚未回升但泛化 gap 持续扩大时提前给出「过拟合前兆」警告（`gap_widening_warning`）与早停建议，不必等验证回升后才被动止损。
  - **NaN/Inf 系统排查**：按「数据 NaN/Inf → log(0)/除零/softmax 溢出 → fp16 混合精度溢出 → 梯度爆炸/学习率过大」路径收集证据；`nan_diagnosis` 记录失败步梯度范数与近期中位数的对比，区分梯度爆炸与前向/损失侧根因。
  - **坏案例分析**：在与周期验证相同的固定窗口上、对**全部**评估 token 做流式统计（每 token 仅保留约 10 字节紧凑标量，不做随机下采样，token 准确率与 `evaluate` 同口径），top-N 最差样本用「每批 top-N + 跨批截断」精确维护并归类（ambiguous / systematic_confusion / label_noise_suspect / uncertain，分别对应样本模糊、模型盲区、标注可疑）；全量错误 token 的类别归因（`error_category_counts` / `error_category_shares`）不只统计 top-N，作为提出新改进思路的证据。
  - **效率与精度权衡**：`--compare` 汇总每个实验单元的参数量、FLOPs、推理延迟、训练步耗时（前向+反向）与显存/内存，并在单因子对比里标注成本倍数；显著涨点但成本 ≥1.5×、或指标变差且更贵的对比会被显式标记（须评估性价比 / 严格劣于基线），避免「涨 0.5% 但计算 ×10」被当成有效改进。

### 多随机种子与消融套件示例

```bash
# RQ1：同一套受控设置下比较 smelt / local_base / A / B，每个 3 个种子
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 --suite-modes smelt local_base A B \
  --steps 2000 --hypothesis "延迟合并改善验证损失" --rq RQ1

# 3×2 全因子消融：local_base/A/B × balance_coef=0/0.01
# （默认覆盖全部已提出方案；只跑部分方案时用 --ablation-modes 声明）
python experiment.py --data corpus.txt --out runs/ablation \
  --seeds 42 43 44 --suite-ablation --ablation-modes A B \
  --steps 2000 --hypothesis "各设计各自有用" --rq RQ1

# 先调参简单基线：学习率网格（val 选择），winner 与报告落在 runs/tuning；
# 注意 --baseline-tuning-mode 必须是之后要声明为 baseline 的那个模式
python experiment.py --data corpus.txt --out runs/tuning \
  --seeds 42 43 44 --suite-baseline-tuning --baseline-tuning-mode dense \
  --steps 2000 --hypothesis "基线调参使其值得战胜" --rq RQ1

# 再用 winner 训练最终基线并记录调参来源（config.json 的 baseline_tuning）；
# 把 --lr 换成 runs/tuning/baseline_tuning.json 里的 winner lr
python experiment.py --data corpus.txt --out runs/baseline_dense \
  --mode dense --lr 3e-4 \
  --baseline-tuning-from runs/tuning/baseline_tuning.json \
  --steps 2000 --hypothesis "dense 基线在调参后的表现" --rq RQ1

# 只读汇总已完成运行：受控变量检查 + Mean ± Std + 显著性检验。
# --compare-baseline 必须指向真正调过参的模式（这里是 dense，且
# runs/baseline_dense 带有 baseline_tuning 来源记录）；若要声明
# smelt 为基线，就对 smelt 跑 --suite-baseline-tuning 并同样记录来源，
# 或在对比集合中加入 --statistical-baseline 产出的 ngram 运行。
python experiment.py --compare runs/rq1/smelt_seed42 runs/rq1/A_seed42 \
  runs/ablation/A_bal0_seed42 runs/baseline_dense --compare-baseline dense
```

## 📂 输出结构

每次实验运行会生成以下目录结构：
```text
runs/<exp_name>/
├── config.json      # 完整配置、参数量、随机种子、数据指纹、数据审计、选型口径、代码版本/环境、sanity check 结果
├── metrics.jsonl    # 逐行记录训练/验证指标及最终评估 (loss, ppl, lr, grad_norm等)，--evaluate-test 时含 {"step": "test", ...} 唯一一次测试集结果
├── best.pt          # 验证集选定的模型权重（final / test 评估使用；原子写入）
└── last.pt          # 模型权重、优化器、学习率调度器及随机种子状态（断点恢复用；原子写入）
```

## 📝 注意事项与免责声明

- 本项目提供的是 **研究实验基座**，默认使用 Python 专家循环实现的稀疏 dispatch，适合研究原型使用。
- 方案 B 尚未实现增量解码 KV Cache，推理阶段需要重新设计缓存策略。
