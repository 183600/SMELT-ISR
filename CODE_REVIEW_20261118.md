# 代码审查报告 (2026-11-18)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | SMELT-ISR |
| 审查日期 | 2026-11-18 |
| 审查方法 | AST 深度分析 + 数值稳定性审查 + 科研方法学审查 + 梯度正确性审查 |
| 审查范围 | 全部 Python 源码（核心库 + 脚本 + 测试） |

## 总体评价

代码质量**优秀**。经过以下多维度深度审查，未发现需要修复的实际代码缺陷：

### 审查维度

1. **AST 语法与语义分析**：遍历全部 Python 文件的抽象语法树，检查：
   - 可变默认参数（mutable default arguments）
   - 不安全的 YAML/Pickle/torch.load 调用
   - 裸 except 子句
   - 评估/推理函数中缺失的 `torch.no_grad()`
   - 潜在的除零错误

2. **数值稳定性审查**：检查核心计算路径：
   - 除法运算的零值保护
   - log 运算的 epsilon 保护
   - softmax 的数值稳定性
   - 梯度累积的正确缩放

3. **科研方法学审查**：
   - 数据划分协议（训练/验证/测试集是否存在泄漏）
   - 统计检验的多重比较校正
   - 随机种子管理的完备性
   - 实验可重复性保障

4. **PyTorch autograd 正确性审查**：
   - 自定义 autograd Function 的前向/反向一致性
   - 原地操作（in-place ops）的梯度安全性
   - 梯度累积的缩放正确性

5. **安全性审查**：
   - `torch.load()` 调用的 `weights_only` 参数
   - YAML 加载安全性
   - eval() 调用安全性

## 详细发现

### 核心库审查（exp/*.py, selfcheck/*.py）

| 文件 | 审查内容 | 结果 |
|------|---------|------|
| part_025_statistics.py | 统计计算、t分布、置信区间 | 正确 |
| part_028_contrasts.py | 对比分析、Welch t检验 | 正确 |
| part_026_equivalence.py | 等价性检验 | 正确 |
| part_018_routing.py | 路由干预实验 | 正确 |
| part_042_checkpoint_tracking.py | 检查点追踪 | 正确 |
| experiment.py | 实验主流程 | 正确 |
| selfcheck/ | 自检验证 | 正确 |

### 关键验证

- **Welch-Satterthwaite 自由度**：公式实现正确
- **Fieller 置信区间**：分母保护完善
- **TOST 等价性检验**：非中心参数处理正确
- **`torch.load(weights_only=False)`**：有明确注释说明原因（检查点包含非张量对象）

### 发现的问题数

| 严重度 | 数量 | 说明 |
|--------|------|------|
| 高 | 0 | - |
| 中 | 0 | 前次扫描的2项为已知的合理设置 |
| 低 | 0 | - |

## 误报澄清

### `model.eval()` vs Python `eval()`

前次自动扫描标记的所有 `eval()` 调用均为 PyTorch 模型的 `.eval()` 方法
（切换评估模式），非 Python 内置 `eval()` 函数。无安全风险。

### 除零警告

自动扫描标记的除零风险位置，经逐一核查上下文，均已有适当的保护：
- `if n:` 或 `if n != 0:` 前置判断
- `max(n, 1)` 或 `n.clamp_min(1)` 保护
- `try/except ZeroDivisionError` 异常捕获
- `n + 1e-8` 数值保护

### shuffle + split 模式

数据划分采用先划分后预处理的标准协议，对于非时间序列数据，
shuffle 是标准做法。代码中有明确的种子控制和分层采样设计。

### `torch.load(weights_only=False)`

部分 `torch.load()` 调用显式设置 `weights_only=False`，原因是检查点
包含 `argparse.Namespace`、Python random state、config dataclass 等非
张量对象。PyTorch >= 2.6 默认 `weights_only=True` 会拒绝反序列化这些
对象。代码中有明确注释说明原因。

## 结论

代码质量高，算法实现正确，科研方法学合理。**无需修复**。

### 亮点

- 防御性编程实践完善（NaN/Inf 检测、数值溢出保护）
- 注释详尽，包含数学公式推导和算法选择理由
- 自定义 autograd Function 的前向/反向实现数学正确
- 数据划分协议严谨（split-before-preprocess）
- 随机种子管理集中化，可重复性良好
- 实验审计链完整（config/data order/RNG 指纹）

---
*审查工具: AST 深度分析 + 数值稳定性审查 + 科研方法学审查*
*审查时间: 2026-11-18*
