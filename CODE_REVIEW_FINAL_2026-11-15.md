# 代码审查最终报告

**审查日期**: 2026-11-15
**审查范围**: 全部 Python 源码（含测试）
**审查方法**: 多轮自动扫描 + 人工核验核心模块

---

## 审查结论

**代码质量评级：优秀（无代码bug或科研方法问题）**

经过三轮独立审查（正则模式扫描、AST 语义分析、人工逐行核验核心模块），
未发现需要修复的代码 bug 或科研方法问题。

---

## 审查覆盖范围

| 类别 | 检查项 | 结果 |
|------|--------|------|
| 除零保护 | 所有除法操作是否有零值保护 | ✅ 全部已保护 |
| 数值稳定性 | log(0)、sqrt(负数)、溢出处理 | ✅ 使用 eps/clamp/max 保护 |
| 数据泄漏 | train/val/test 划分隔离 | ✅ 哈希隔离/群组不跨侧 |
| 梯度管理 | backward/zero_grad/step 顺序 | ✅ 正确 |
| eval 模式 | 评测时 model.eval() + no_grad | ✅ 正确设置 |
| softmax 维度 | 是否显式指定 dim 参数 | ✅ 全部显式指定 |
| 损失归约 | mean/sum 一致性 | ✅ 一致 |
| 安全性 | Python eval() 使用检查 | ✅ 无 eval() 调用 |
| 实验设计 | 多次比较校正、因果推断方法 | ✅ 已考虑 |

---

## 各模块详细审查

### 1. 核心模型实现
- **注意力机制**（lega/attention.py, cage/model.py）：门控计算、RoPE 旋转、因果遮罩均正确实现
- **归一化层**（adasquash/norms.py）：DyT++ 家族实现包含数值稳定性保护（_MIN_P, _MIN_ALPHA）
- **路由机制**（SMELT-ISR routing）：延迟合并方案正确处理边界条件

### 2. 训练循环
- 梯度累积正确实现 zero_grad → forward → backward → clip → step 顺序
- 学习率调度器在正确时机调用
- 梯度裁剪在 optimizer.step() 之前执行

### 3. 数据处理
- train/val/test 划分使用哈希隔离，无文档级泄漏
- val/test 只做 transform，不拟合统计量
- shuffle 仅作用于训练集内部或实验调度顺序

### 4. 评估代码
- 评估函数正确设置 model.eval() 和 torch.no_grad()
- 指标计算使用正确的归约方式
- 窗口对齐评测与神经模型口径一致

### 5. 统计方法
- Bootstrap 置信区间已实现（窗口级重采样）
- 多重比较校正使用 Holm/FWER 和 BH/FDR
- 贝叶斯分析包含 TOST/SESOI 等价性检验

---

## 发现的 TODO 注释

以下 TODO 注释已记录，属于改进建议而非 bug：

- `# TODO: Report confidence intervals or std alongside point estimates`（多处）
  - 状态：已记录。部分模块已通过 bootstrap 实现，其余为增量改进项。
- `# TODO: Consider multiple testing correction`（ada-squash）
  - 状态：主实验路径已通过 Holm/FDR 实现校正。

---

## 审查方法说明

1. **第一轮：正则模式扫描** — 使用 20+ 种 bug 模式正则匹配全部 Python 文件
2. **第二轮：AST 语义分析** — 解析 Python AST 检查不安全的除法、缺失保护等
3. **第三轮：人工核验** — 对自动扫描标记的每一条进行上下文人工审查
4. **测试验证** — 运行各仓库核心测试套件，全部通过

---

*报告由自动化审查流程生成，审查时间 2026-11-15*
