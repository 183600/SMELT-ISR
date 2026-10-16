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
- `local_base`：立即合并对照。第一轮 MoE 后立即合并，再执行第二轮 MoE。

## 关键配置

- `top_k`：普通 Block 的 MoE 路由宽度。
- `k1` / `k2`：第一轮分支数与第二轮路由宽度。
- `balance_coef`：负载均衡损失的系数。辅助损失按 token 数归一化，
  均匀路由时约为 `top_k`，均衡强度随路由宽度线性增大；跨
  `k1`/`k2`/`top_k` 比较实验时应按比例折算系数（保持
  `balance_coef × k` 不变），否则路由宽度会与均衡强度混杂。
- `residual_scale`：残差缩放系数。

## 代码组织

- `Config`：模型结构与损失配置，在入口处校验。
- `RMSNorm`、`CausalSelfAttention`：基础层。
- `Expert`、`SparseMoE`：专家与稀疏路由，含 `route` / `dispatch` / `merge`。
- `TransformerBlock`：注意力步骤、MoE 步骤，以及分支与两轮路由逻辑。
- `LanguageModel`：按模式组装完整前向。
- `ByteDataset`、`evaluate`：字节级数据管线与验证。
- `save_checkpoint`、`load_checkpoint`：断点存档与恢复。
- `run_self_test`：结构、梯度、因果性与模式等价性自测。

## 训练入口

```
python experiment.py --data <语料文件> --out <输出目录> --mode <模式>
```

常用参数：`--steps`、`--batch-size`、`--lr`、`--warmup-steps`、
`--grad-clip`、`--eval-interval`、`--save-interval`、`--resume`、
`--grad-checkpoint`、`--compile`。
