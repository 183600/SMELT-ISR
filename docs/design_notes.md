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
