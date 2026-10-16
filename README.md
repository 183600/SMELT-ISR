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

`val` 用于周期性验证、挑超参数与早停；`test` 训练期间全程尘封，
只有显式传入 `--evaluate-test` 才会在训练完全结束后评估一次，且同一次
运行拒绝重复评估。切分比例可用 `--val-fraction` / `--test-fraction`
调整。每次运行在训练前都会自动做数据审计（token 值域、重复窗口、
train↔val 分布偏移），结果与警告写入 `config.json` 的 `data_audit`
字段；解封测试集时会追加包含 test 的审计结果。

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
python experiment.py --data corpus.txt --mode B --sanity-only

# 显式跳过检查（已充分验证的前提下）
python experiment.py --data corpus.txt --mode B --skip-sanity-check ...
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
- **验证/测试协议**：每次评估使用跨评估 split 等间隔选取的固定窗口，窗口集合不依赖随机数，可在不同模型和恢复训练之间逐位对齐；训练期间的 `final` 记录是验证集口径，测试集默认全程尘封，仅在显式传入 `--evaluate-test` 时解封评估一次（`{"step": "test", ...}`），重复传参会被拒绝，避免测试集变成隐式验证集。所有结果追加到 `metrics.jsonl`。
- **确定性**：训练入口固定 Python / NumPy / PyTorch(CPU+CUDA) / cuDNN 的随机种子，数据采样由显式 `torch.Generator` 承担（种子与状态随检查点存档），并设置 `CUBLAS_WORKSPACE_CONFIG`、要求 PyTorch 优先使用确定性算子；若当前 CUDA 算子没有确定性实现，PyTorch 会给出告警。跨硬件架构仍可能因浮点归约顺序产生微小差异。
- **实验追踪**：每轮实验自动记录代码版本（git commit / 是否有未提交改动）、运行环境（Python / torch / CUDA / 设备）、完整超参数与随机种子、数据指纹（SHA-256、字节数）、训练曲线与最终结果，全部写入 `config.json` 与 `metrics.jsonl`，无需手动记录。

## 📂 输出结构

每次实验运行会生成以下目录结构：
```text
runs/<exp_name>/
├── config.json      # 完整配置、参数量、随机种子、数据指纹、代码版本/环境、sanity check 结果
├── metrics.jsonl    # 逐行记录训练/验证指标及最终评估 (loss, ppl, lr, grad_norm等)，--evaluate-test 时含 {"step": "test", ...} 唯一一次测试集结果
└── last.pt          # 模型权重、优化器、学习率调度器及随机种子状态（原子写入）
```

## 📝 注意事项与免责声明

- 本项目提供的是 **研究实验基座**，默认使用 Python 专家循环实现的稀疏 dispatch，适合研究原型使用。
- 方案 B 尚未实现增量解码 KV Cache，推理阶段需要重新设计缓存策略。
