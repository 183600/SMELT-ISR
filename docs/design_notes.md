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

## 代码组织

- `Config`：模型结构与损失配置，在入口处校验。
- `RMSNorm`、`CausalSelfAttention`：基础层。
- `Expert`、`SparseMoE`：专家与稀疏路由，含 `route` / `dispatch` / `merge`。
- `TransformerBlock`：注意力步骤、MoE 步骤，以及分支与两轮路由逻辑。
- `LanguageModel`：按模式组装完整前向。
- `ByteDataset`、`evaluate`：字节级数据管线与验证。
- `run_sanity_overfit`：训练前的过拟合小批量 sanity check。
- `save_checkpoint`、`load_checkpoint`：断点存档与恢复。
- `collect_run_metadata`、`write_run_config`：实验追踪信息收集与落盘。
- `run_self_test`：结构、梯度、因果性、模式等价性与 sanity check 判定自测。

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
`--sanity-lr`、`--sanity-loss-threshold`、`--sanity-acc-threshold`。
