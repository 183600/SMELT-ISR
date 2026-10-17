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
- **控变量对比与统计可靠性**：每轮训练记录受控变量协议（优化器、学习率策略、训练步数、数据增强、数据指纹等），`--compare` 自动核对「除声明因子外设置是否完全一致」；`--seeds` 支持多随机种子运行并汇报 Mean ± Std 与显著性检验，`--suite-ablation` 提供 2×2 全因子消融，自动检查消融组合是否完整。

## 🛠️ 环境依赖

- Python 3.10+
- PyTorch 2.x

```bash
pip install "torch>=2.2"
```

## 📦 代码使用说明

### 1. 运行结构自测
在开始实验前，建议运行内置的自测模块，以确保不同模式的结构正确性、分支独立性与等价性。
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

**严格数据审计**：加上 `--strict-data-audit` 后，审计发现的泄露级问题
（跨 split 重复窗口、越界 token、超阈值的 UTF-8/NUL 损坏）会直接
拒绝开始训练；解封测试集时若测试集本身有同类问题也会拒绝解封（不写入
`test` 记录）。默认只警告并继续，方便先看清数据问题再决定清洗策略。

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

### 对照口径与可复现性

- **B 的两轮路由与合并权重**：方案 B 的第一轮中间区域全程使用 $k_1$，第二轮完整中间区域才使用 $k_2$；合并使用第一轮路由得到且 Top-$k_1$ 归一化的固定权重，不会用最终分支状态重新计算合并权重。论文分析或消融中应明确这一点。
- **计算量口径**：方案 B 在第一轮分支创建后按 $k_1$ 倍分支数执行后续 Attention 与 MoE，等参数实验并不等价于等 FLOPs 实验。回答 RQ1/RQ2 时应同时报告按训练步数与按 token/FLOPs 对齐的对比，避免把额外计算误解释为路由机制本身的收益。
- **残差缩放口径**：`residual_scale` 全局作用于所有区域、所有子层的残差更新（循环区域内/外一致）。SMELT 论文对循环区域内的子层残差更新按循环次数 $1/r$ 缩放（循环两次即 1/2）；本基座未实现按区域区分的 $1/r$ 缩放。跨模式对比应保持 `residual_scale` 一致，把全局系数设为 0.5 并不等价于论文口径。
- **负载均衡尺度**：辅助损失中 $f_i$ 按 token 数归一化，而不是按 assignment 数归一化；均匀路由时 $E \sum_i f_i P_i \approx k$，均衡强度随路由宽度线性增大。跨 $k_1/k_2/\text{top-}k$ 比较实验时应按比例折算系数（保持 `balance_coef × k` 不变），否则路由宽度会与均衡强度混杂。
- **验证/测试协议**：每次评估使用跨评估 split 等间隔选取的固定窗口，窗口集合不依赖随机数，可在不同模型和恢复训练之间逐位对齐；训练期间每次刷新最优验证损失都保存 `best.pt`，训练结束后的 `final` 记录与唯一一次测试集评估都在这份验证集选定的模型上进行（`model_selection` 字段记录选型口径；`best.pt` 不可用时明确提示并回退到最后一步）。测试集默认全程尘封，仅在显式传入 `--evaluate-test` 时解封评估一次（`{"step": "test", ...}`），重复传参会被拒绝，避免测试集变成隐式验证集。所有结果追加到 `metrics.jsonl`。
- **确定性**：训练入口固定 Python / NumPy / PyTorch(CPU+CUDA) / cuDNN 的随机种子，数据采样由显式 `torch.Generator` 承担（种子与状态随检查点存档），并设置 `CUBLAS_WORKSPACE_CONFIG`、要求 PyTorch 优先使用确定性算子；若当前 CUDA 算子没有确定性实现，PyTorch 会给出告警。跨硬件架构仍可能因浮点归约顺序产生微小差异。
- **实验追踪**：每轮实验自动记录代码版本（git commit / 是否有未提交改动）、运行环境（Python / torch / CUDA / 设备）、完整超参数与随机种子、数据指纹（SHA-256、字节数）、训练曲线与最终结果，全部写入 `config.json` 与 `metrics.jsonl`，无需手动记录。
- **受控变量检查**：`config.json` 的 `comparison_protocol` 字段记录优化器（AdamW 及 betas/eps）、学习率与调度、warmup、训练步数、weight decay、梯度裁剪、batch size、数据增强、数据指纹、模型结构、评估协议与选型口径（新运行为验证集选型，无记录的旧运行标记为最后一步）。`--compare` 默认要求这些设置在声明因子（`--compare-factors`，默认 `mode,k1,k2,balance_coef,seed,dense_fnn_hidden`）之外逐项一致：某个运行换了优化器、调了学习率或训练轮数、或 final 指标口径不同，都会被明确列出，避免把调参收益误当成方法本身的收益。
- **统计可靠性**：`--seeds 42 43 44` 对同一配置跑多个随机种子，自动汇报每个实验单元的 Mean ± Std（样本标准差）；种子集合相同用配对 t 检验、否则用 Welch t 检验，给出 p 值与噪声比（|Δ|/合并标准差）。种子数少于 3 会被标记为统计功效不足；只有 0.2% 的提升而种子间方差有 0.5% 时不会被当成有效提升。
- **消融完整性**：`--suite-ablation` 运行「延迟合并 × 均衡损失」2×2 全因子（`local_base`/`A` × `balance_coef`=0/当前值），覆盖全部四个组合后才能支撑两个设计各自有用的结论；覆盖检查会列出缺失的组合。`--suite-modes` 用于严格控变量的跨模式对比（如 `smelt local_base A B` 一起跑）；套件结束后汇总分析写入 `<out>/comparison_analysis.json`。
- **结果分析（曲线 / NaN / 坏案例 / 效率）**：
  - **在线曲线监控**：不只训练结束后看最终数值，每个周期验证都会复用 `analyze_training_curve` 在线判定；状态迁移（如 healthy→overfitting、发散）立即打印建议并写入 `{"step": "curve_monitor", ...}`，过拟合→加正则化/增大数据/`--early-stop-patience` 早停，两者都不降→查学习率、优化器与数据管道是否对齐。
  - **NaN/Inf 系统排查**：按「数据 NaN/Inf → log(0)/除零/softmax 溢出 → fp16 混合精度溢出 → 梯度爆炸/学习率过大」路径收集证据；`nan_diagnosis` 记录失败步梯度范数与近期中位数的对比，区分梯度爆炸与前向/损失侧根因。
  - **坏案例分析**：在与周期验证相同的固定窗口上打印损失最高的样本并归类（ambiguous / systematic_confusion / label_noise_suspect / uncertain，分别对应样本模糊、模型盲区、标注可疑）；同时对**全部错误 token**做类别归因（`error_category_counts` / `error_category_shares`），不只统计 top-N，作为提出新改进思路的证据。
  - **效率与精度权衡**：`--compare` 汇总每个实验单元的参数量、FLOPs、推理延迟与显存/内存，并在单因子对比里标注成本倍数；显著涨点但成本 ≥1.5×、或指标变差且更贵的对比会被显式标记（须评估性价比 / 严格劣于基线），避免「涨 0.5% 但计算 ×10」被当成有效改进。

### 多随机种子与消融套件示例

```bash
# RQ1：同一套受控设置下比较 smelt / local_base / A / B，每个 3 个种子
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 --suite-modes smelt local_base A B \
  --steps 2000 --hypothesis "延迟合并改善验证损失" --rq RQ1

# 2×2 全因子消融：local_base/A × balance_coef=0/0.01
python experiment.py --data corpus.txt --out runs/ablation \
  --seeds 42 43 44 --suite-ablation \
  --steps 2000 --hypothesis "两个设计各自有用" --rq RQ1

# 只读汇总已完成运行：受控变量检查 + Mean ± Std + 显著性检验
python experiment.py --compare runs/rq1/smelt_seed42 runs/rq1/A_seed42 \
  runs/ablation/A_bal0_seed42 --compare-baseline smelt
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
