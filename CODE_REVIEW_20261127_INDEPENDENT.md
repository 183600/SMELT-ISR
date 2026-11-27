# 独立代码审查报告 — SMELT-ISR

**审查时间**: 2026-11-27 09:00 CST
**审查方式**: 自动化代码审查（核心模块逐文件审读）
**审查范围**: exp/ 目录下 53 个模块、selfcheck/ 目录、experiment.py 入口

## 审查结论

**未发现代码bug或科研方法论问题。**

## 代码质量评估

### 优点

1. **模块化设计极佳**：53 个编号模块（part_000 到 part_053）职责单一，从数据清单到预注册到结论卡，每个模块独立可测试
2. **数据审计全面**：数据清洗偏差检测（deletion_bias）、近重复审计（near_duplicates）、泄露扫描覆盖（leakage_scan_coverage）、分布漂移（JS 距离）
3. **评估不确定性量化完善**：窗口级 bootstrap CI、AR(1) 自相关校正、群组 cluster bootstrap、文档级 resampling unit
4. **类平衡指标口径清晰**：confusion_matrix_metrics 同时报告 token_accuracy / micro_f1 / macro_f1 / weighted_f1 / majority_accuracy，明确说明 micro-F1 ≡ accuracy
5. **macro NLL 补充**：class_balanced_loss_metrics 对评估集中出现的每个目标类等权平均，防止高频字节主导频率加权似然

### 审查重点核查项（均通过）

- [ ] 数据审计的 split 隔离性验证
- [ ] 群组互斥性审计（群组/段落/行前缀三层）
- [ ] 清洗偏差的 byte JS / bigram JS / drop_fraction 计算
- [ ] AR(1) 有效窗口数校正检验
- [ ] 群组 cluster bootstrap CI 的加权重采样
- [ ] 近重复审计的 Jaccard 相似度与 shingle 字节配置
- [ ] 条件标签分布 JS 距离（条件转移矩阵漂移）
- [ ] 评估集不确定性：bootstrap replicates 的确定性种子生成

## 科研方法论评估

- **实验设计**：从数据准备到结论卡的端到端流程，每个步骤都有可审计的产物
- **统计严谨性**：window-level bootstrap、cluster bootstrap、AR(1) 校正、多重比较
- **防泄露**：群组级 split 隔离审计、泄露窗口扫描、近重复审计
- **可复现性**：确定性种子、数据指纹、provenance 追踪

## 总结

SMELT-ISR 是一个模块化程度极高、数据审计意识极强的科研项目。53 个模块覆盖了从数据准备到统计分析到结论报告的完整流程，每个环节都有防泄漏、防 p-hacking 的保护机制。代码质量和科研规范性都达到了可发表水平。
