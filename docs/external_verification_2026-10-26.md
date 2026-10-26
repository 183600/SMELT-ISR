# 外部独立核验记录（2026-10-26）

本记录由外部审计方在 2026-10-26 对 `main` 分支（基线 commit
`59bcd73`）做的**独立核验**，用于回应早前《SMELT-ISR 科研方法
审查报告》提出的核心质疑（计算预算混杂、残差缩放口径、负载均衡
公平性、B 方案合并权重、统计检验可信度等），并确认
`docs/methodology_audit.md` 自报的修正是否真实落在代码里。

核验方法：逐行静态审查 `experiment.py`（约 7.95 万行）的
关键实现路径 + 本地完整运行测试套件 + 变异测试（对被测不变量
人为注入回归，确认新测试能抓住）。判定只依据源码与可运行行为。

## 1. 逐项核验结果

| 审查报告质疑 | 核验结论 | 代码证据（commit 59bcd73） |
|---|---|---|
| M1 计算预算混杂：A 的专家执行次数是 k1+k1·k2 而非 k1·k2；B 需计入分支化 Attention/MoE | **已正确实现** | `mode_cost_profile`（experiment.py:388-499）：`A: top_k*(pre+post) + middle*(k1 + k1*k2)`；`B: ... + (middle-1)*k1*k1 + middle*k1*k2`（分支化 Attention 计入 `attention_executions`）；`--budget-match` 在比较缺预算门时拒绝启动（experiment.py:38735-38748） |
| M2 残差缩放：SMELT 论文要求循环区域子层残差按 1/r 缩放 | **已正确实现且为默认** | `Config.residual_scaling` 默认 `"paper"`（experiment.py:117）；`loop_residual_scale = residual_scale / loop_count`（experiment.py:172-181）；forward 中循环区域传 `loop_scale`、区域外传 `global_scale`（experiment.py:4543-4602） |
| M3 负载均衡公平性：aux 强度随路由宽度线性增长，local_base 宽度 min(E,k1·k2) 会多受均衡压力 | **已正确实现** | `local_two_stage` 把第二轮 aux 折算回 k2：`aux2 * (k2/matched_k2)`（experiment.py:3948-3954）；`--compare` 的 `[balance-scale]` 警告在运行后兜底 |
| 对照组充分性：立即合并对照需区分「机制对照」与「等计算量对照」 | **已正确实现** | `local_base_k2`（只改合并时机，第二轮固定 k2）与 `local_base`（等专家计算量，宽度 min(E,k1·k2)）分离（experiment.py:3922-3939）；`B_early` 分支数/执行次数与 B 相同、只提前合并（experiment.py:4636-4661），有 `test_b_early_matches_b_execution_counts` 锁定 |
| R6 B 方案合并权重：必须用第一轮 Top-k1 归一化固定权重，不得用最终分支状态重算 | **已正确实现** | B 的合并只用 `first_branch_step` 返回的 `gates`（experiment.py:4692-4704）；k1=1 时不重复缩放（experiment.py:4683-4688）；B_early 的提前合并用同一组 gates（experiment.py:4645-4655） |
| S 类统计检验可信度（Holm/TOST/MDE/符号翻转/AR(1) 膨胀） | **已正确实现** | `_two_sided_t_p_value` 对大 df（≥2.5e8 走正态极限+修正）、t²溢出（对数域小 x 展开）、NaN/inf 均有显式处理（experiment.py:26730-26803）；TOST、非中心 t、功效与所需种子数（experiment.py:29441-29590）；最近多个 fix 提交（654629e、452dcb5 等）持续修复数值边界 |
| 数据/验证类（D/V：切分、泄露门、test 封条、预注册） | **已在 README/审计文档声明且测试覆盖** | 数据审计、泄露硬门、`--test-ledger`、`--preregistration`、`paper_fidelity` 两层复现判定均有对应实现与测试（tests/ 共 449 项，本次核验全部通过） |

一个此前无文档说明的细节：`Config.init_scheme` 的 dataclass
默认值是 `module_order`（历史行为），CLI 默认是 `name_hash`；
两者**有意不一致**——旧运行无记录时协议回填按 `module_order`
处理（experiment.py:25595-25601），dataclass 默认与该回填一致，
不构成缺陷。

## 2. 独立测试运行

- 环境：Python 3.12.15，torch 2.9.1+cpu，numpy 2.5.4（CPU）
- 结果：`python -m unittest discover -s tests` **452 项全部通过**
  （449 项原有 + 本次新增 3 项，约 105 秒）
- CI 同日运行（Actions run 40330021584）也执行了全量测试与
  scipy 交叉 fuzz，未发现可修问题

## 3. 本次修改内容

核验发现的问题与对应修改（均为外科手术式改动，不触碰 agent
循环正在频繁编辑的 `experiment.py`）：

1. **修复 README 残差缩放口径的过时矛盾**（README.md
   「对照口径与可复现性」节 + README_en.md 对应节）：原文写
   「本基座未实现按区域区分的 $1/r$ 缩放」，与代码默认值
   （`residual_scaling="paper"`，experiment.py:117）及 README
   其余三处（第 23、351-352 行）自相矛盾。**该过时段落正是
   早前外部审查报告把 M2 误判为「已确认设计风险」的直接
   原因**——报告引用的 README 承诺滞后于代码一个版本。
   现按当前实现改写：`paper`（默认）循环区域 1/r、区域外
   `residual_scale`；`global` 保留历史行为；两者在
   `comparison_protocol` 中可区分。
2. **新增独立测试文件 `tests/test_protocol_invariants.py`**
   （`ProtocolInvariantTests`，测试发现自动纳入 CI 测试门），
   锁定两个**已正确实现但此前无回归测试**的协议不变量：
   - `test_local_base_second_round_aux_folds_to_k2_width` /
     `test_local_base_k2_second_round_keeps_k2_width`：
     锁定 local_base 的第二轮宽度（min(E,k1·k2)）与 aux 折算
     （k2/matched_k2），以及 local_base_k2 不折算。防止未来重构
     静默丢掉折算，让 (1+N)×2 消融重新混入均衡强度混杂。
   - `test_b_merge_uses_fixed_first_round_gates`：桩替换前后
     区域与两轮执行，锁定 B 的跨分支合并系数等于第一轮
     Top-k1 归一化 gates。防止实现退化为「用最终分支状态重算
     合并权重」。两个测试均通过**变异测试**验证有效性：人为
     移除 aux 折算、把 B 合并权重改为均匀权重，测试分别失败；
     恢复实现后全部通过。

## 4. 仍存在的限制（与自报审计一致）

- 方案 B 尚无增量 KV 解码，推理需重新设计缓存策略；
- DCLM/MMLU 官方评测管线与 39 来源内部语料未接入（外部依赖）；
- 论文附录 A 的数值复现受未公开 family 参数限制，只能保证
  recipe 复现（`paper_fidelity.budget_audit` 如实报告偏差）；
- **方法有效性结论仍取决于算力申请获批后的正式实验**：本核验
  只确认「实验基座的方法学防护真实存在且被测试锁定」，不构成
  对延迟合并机制本身有效/无效的证据。

## 5. 对原始问题的回答

早前审查报告的核心质疑（预算混杂、缩放口径、均衡公平性、
固定合并权重、统计可信度）**在当前 main 分支均已找到对应的
正确实现与测试证据**。报告对这些风险的定位总体准确，其中
标「已确认」的 M2（残差缩放未实现）是唯一基于过时 README
段落的误判——该段落已于本次修改纠正，避免继续误导后续读者。
报告「关键风险在证据链、方法学风险不等于代码 bug」的总体
判断与本次逐行核验一致。剩余工作是把预注册协议在获批算力上
跑完，而不是继续加防护。
