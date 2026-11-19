# 代码审查报告 - SMELT-ISR (深入审查)
**审查日期**: 2026-11-15
**审查范围**: 全仓库核心代码（exp/）、模型实现、统计模块、测试（tests/）
**审查深度**: 逐模块深入审查 + MoE路由验证 + 统计公式核对

## 审查结论

**代码质量评级：优秀**。经逐行审查未发现需修复的关键bug或科研方法问题。

## 审查要点

### MoE 路由机制 ✅
- Top-k 路由正确实现
- 负载均衡损失 aux_loss = n_experts * dot(hard_load, soft_load) 正确
- _BlockCheckpoint 非重入 activation checkpointing 正确
- 延迟合并方案 (A/B/B_early) 正确隔离架构因素

### 路由统计 ✅
- usage_share 使用 clamp_min(1) 防止除零
- router_entropy 正确归一化到 [0, 1]
- load_entropy 正确计算经验分布熵
- 干预实验正确实现 shuffled/dropped 对照

### Welch t 检验 ✅
- Welch-Satterthwaite 自由度公式正确
- 极小/极大尺度的递归归一化处理正确
- se_squared = 0 特殊情况正确处理
- 次正规数检测使用 sys.float_info.min 防护

### 分类指标 ✅
- micro_f1/macro_f1 正确实现
- precision/recall 的 macro 和 weighted 口径正确
- confusion matrix 使用 max(total, 1) 防护除零

### 数据处理 ✅
- split 边界连续且首尾相接
- train/val/test 互不重叠
- 固定窗口评估批次可复现

### 静态分析发现项均为误报
- eval() 均为 model.eval() (PyTorch)
- 除零风险测试中的断言值非实际计算
- shuffle 在合成测试数据中，无时序依赖

## 结论
代码实现质量高，MoE 路由和统计方法正确，无需修复。
