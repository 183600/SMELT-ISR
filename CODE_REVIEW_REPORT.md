# Code Review Report - SMELT-ISR

**审查时间**: 2026-11-09  
**审查范围**: 核心模块 (exp/*.py, selfcheck/*.py)  
**审查方法**: AST 静态分析 + 人工复核

## 总体评价

代码质量**优秀**。项目实现了统计检验和实验分析框架，代码严谨，统计方法实现正确。未发现严重bug或方法学错误。

## 发现的问题

### 已确认的问题（无）

未发现需要修复的代码缺陷。

### 设计说明（非 bug）

#### 1. 受控 exec() 使用 (exp/__init__.py:19, exp/_runner.py:67)
```python
# exp/__init__.py
exec(compile(_part_source, _part_path, "exec"), globals())

# exp/_runner.py
exec(compile(_read(absolute), absolute, "exec"), namespace)
```
**说明**: 这是模块化执行模式，用于在隔离命名空间中运行实验部分。执行的文件为项目本地文件，非用户输入。**非安全风险**，为有意设计。

### 误报澄清

初步扫描标记了以下问题，经深入分析确认为**误报**：

1. **除零错误** (selfcheck/checks_001.py:999-1000, exp/part_025_statistics.py:913)  
   - 999-1000: 除以常量 0.05，非变量
   - 913: 除以 `intervals`，由 `max(200, ...)` 保证 ≥ 200
   - 无风险

2. **`eval()` 调用** (20处)  
   - 全部为 PyTorch 模型的 `.eval()` 方法调用或函数名的一部分
   - 无安全风险

3. **`log()` 无 epsilon 防护** (45处)  
   - 多数为 `math.log(2.0)` 等常量
   - 部分使用 `max(value, eps)` 防护
   - TOST 检验中 se ≤ 0 有显式处理
   - 数值稳定性良好

4. **random 无种子** (selfcheck/checks_000.py, checks_001.py 多处)  
   - 自检代码中的随机性用于测试
   - 生产代码种子由上层统一管理
   - 可重复性有保证

5. **资源泄漏** (exp/part_015_statistical_baseline.py:91, exp/mainflow/part_001.py:42)  
   - 均使用 `atexit.register()` 确保关闭
   - 无泄漏

## 科研方法学审查

### 统计方法
- TOST（Two One-Sided Tests）等价性检验实现正确
- 多重比较校正（Holm-Bonferroni）正确
- 功效分析完善

### 数值稳定性
- 使用 log-sum-exp 避免溢出
- t 分布 CDF 实现数值稳定
- 边界情况处理完善

### 实验设计
- 因子设计（factorial design）实现正确
- 消融实验（ablation study）框架完善
- 包含顺序效应分析

## 结论

代码质量高，**无需修复**。项目展现了优秀的统计计算和实验设计实践。

## 建议改进

1. **文档**: 可添加统计检验方法的数学推导文档
2. **示例**: 可添加更多端到端的使用示例

---
*审查工具: AST 静态分析 + 人工复核*  
*审查者: AI Code Reviewer*
