from __future__ import annotations

def _MAIN_part_000():
    global parser, shift_flag, shift_label
    parser = argparse.ArgumentParser(description='SMELT-ISR 训练入口')
    parser.add_argument('--data', type=str, default=None, help='训练语料文件（--compare 模式可省略）')
    parser.add_argument('--mode', type=str, default='B', choices=sorted(Config.MODEL_MODES), help='模型模式；base 为论文的未循环 MoE 基线、smelt 为循环两次的 SMELT、local_base_k2 为只改变合并时机的机制对照、B_early 为 B 的等计算量合并位置对照；dense 为无 MoE、无循环的纯稠密 Transformer 基线（先跑简单基线，再谈复杂模型）')
    parser.add_argument('--dense-fnn-hidden', type=int, default=None, help='dense 基线的 FFN 隐藏宽度；默认 top_k * expert_hidden（与 MoE 单次前向激活参数量对齐的等计算口径），等参数口径可设为 n_experts * expert_hidden')
    parser.add_argument('--init-scheme', type=str, default='name_hash', choices=sorted(Config.INIT_SCHEMES), help='参数初始化方案。name_hash（默认）按运行种子与参数名派生独立随机流，使同种子下不同 mode 的共享层拿到相同初始权重，配对对比不会混入初始化差异；module_order 是历史行为（按模块构造顺序消耗全局 RNG），跨 mode 对比时共享层初始化不同，不建议用于方法比较')
    parser.add_argument('--statistical-baseline', action='store_true', help='不训练神经模型：在 train 上计数、在 val 上网格调优（阶数 × 平滑系数）字节级插值 n-gram 统计基线，并用与神经模型完全相同的固定窗口协议评估；输出与训练运行同构的 config.json / metrics.jsonl')
    parser.add_argument('--ngram-max-order', type=int, default=5, help=f'n-gram 统计基线的最大阶数（字节精确打包上限 为 {_NGRAM_MAX_ORDER}）')
    parser.add_argument('--ngram-alpha-grid', nargs='+', type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], help='n-gram 插值平滑系数的候选网格，全部在 val 上调优后取最优（简单基线也必须充分调优）')
    parser.add_argument('--ngram-tuning-metric', type=str, default=None, choices=_NGRAM_TUNING_METRIC_KEYS, metavar='METRIC', help='n-gram 基线选择 (order, alpha) 的 val 指标；缺省时优先跟随 --primary-metric，否则用 val_loss。在 val_loss 上调好的统计基线不能证明它在 macro F1 / PR-AUC 上也已调强')
    parser.add_argument('--hypothesis', type=str, default='', help='本轮实验要验证的科学假设（必填）：写清楚「预期什么改动带来什么效果、依据什么指标判定成败」，带着假设做实验而不是排列组合超参数')
    parser.add_argument('--rq', type=str, default='', help='本 run 对应的研究问题标签（如 RQ1），与 --hypothesis 一起写入 config.json')
    parser.add_argument('--primary-metric', type=str, default=None, metavar='METRIC', help='预注册的主判定指标（--compare-metric 的候选之一）：训练开始前声明本轮以哪个指标判定成败，对比阶段若分析指标与预注册主指标不一致会提示「事后换指标」；未预注册则无法排除指标选择偏差。--statistical-baseline 时仅用于指定 n-gram 的 val 调参口径')
    parser.add_argument('--hypothesis-direction', type=str, default=None, choices=('decrease', 'increase'), help='预注册的预期方向：实验臂相对对照臂，主指标的数值应当 decrease（下降）还是 increase（上升）。必须与 --primary-metric 一起使用；对比阶段会核对观测方向是否与预注册一致（相反且显著时判为假设被推翻）')
    parser.add_argument('--hypothesis-min-effect', type=float, default=None, metavar='REL', help='预注册的最小有意义相对效应（如 0.02 = 2%%，相对对照臂均值）：观测效应达到该幅度才支持假设；必须与 --primary-metric 和 --hypothesis-direction 一起使用。事先声明的最小效应可以区分「方向对但小到没有实际意义」与真正支持假设')
    parser.add_argument('--hypothesis-baseline', type=str, default=None, choices=sorted(Config.ALL_MODES), metavar='MODE', help='预注册的对照基线（训练前声明本轮与哪个臂比较）：对比阶段若 --compare-baseline 指定了别的模式会提示 baseline shopping；未显式指定分析基线时采用该预注册值。缺少它时事后换一个更弱的臂当基线，任何显著性与等价性结论都可以被操纵')
    parser.add_argument('--selection-metric', type=str, default=None, choices=_SELECTION_METRIC_KEYS, metavar='METRIC', help='验证集 checkpoint 选型与早停依据的指标（默认 val_loss；预注册了 --primary-metric 且该指标在周期验证中可计算时默认跟随主指标）：按宏观 F1 判定成败时若仍按 val_loss 选型，最终模型并不是按假设口径选出来的。可选范围限于周期验证实际计算的指标（AUC 默认只在最终评估计算；选为选型指标或加 --periodic-auc 后可在周期验证计算）')
    parser.add_argument('--periodic-auc', action='store_true', help='在每个周期验证点计算与 final/test 同口径的 PR-AUC / lift / ROC-AUC（额外前向开销）。用于生成避开 checkpoint 选型偏差的 <metric>_at_common_step 排序指标；AUC 已被选为--selection-metric 时自动启用，无需重复传')
    parser.add_argument('--skip-hypothesis-check', action='store_true', help='显式跳过假设检查（只建议纯工程目的的临时运行使用；跳过会被记录进 config.json）')
    parser.add_argument('--preregistration', type=str, default=None, metavar='FILE', help='训练前冻结的预注册 JSON（study_id / frozen_commit / 主指标 / 方向 / 最小效应 / 共同步数 / 对照臂 / 种子集合 / 数据指纹）。文件补齐未显式给出的命令行参数；显式冲突、代码提交不一致、工作区脏或数据指纹不一致时拒绝启动。--compare / --sanity-only / --suite-baseline-tuning 不适用')
    parser.add_argument('--out', type=str, default=None, help='输出目录（--compare 模式可省略）')
    parser.add_argument('--allow-overwrite-out', action='store_true', help='显式允许全新训练复用非空 --out 目录：默认拒绝，因为训练会截断 metrics.jsonl 并覆盖 config.json，使上一轮实验的训练曲线与最终结果不可追溯')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.05, help='按时间顺序划给验证集的尾部比例（默认 0.05）；划分不随机打乱')
    parser.add_argument('--test-fraction', type=float, default=0.05, help='按时间顺序划给测试集的尾部比例（默认 0.05）；测试集训练期间尘封，仅在 --evaluate-test 时评估一次')
    parser.add_argument('--entity-manifest', type=str, default=None, metavar='FILE', help='显式实体清单（JSON 数组，按群组序号给 entity_id）：审计同一用户/作者/文档是否跨 split。--group-separator 只保证连续片段完整，不能识别非相邻片段的同一实体')
    parser.add_argument('--allow-entity-overlap', action='store_true', help='显式允许实体跨 split（默认拒绝训练/拒绝解封）：跨 split 实体是群组级泄露，豁免会写入配置')
    parser.add_argument('--split-policy', type=str, default='contiguous', choices=('contiguous', 'chronological'), help='切分口径：contiguous（默认）只是按文件顺序做连续前缀切分，除非语料确实按时间排序，否则不构成时间切分；chronological 需要 --time-order-manifest 证明文件顺序与时间戳单调一致。文档/来源分层或来源外推切分仍需显式群组/来源清单')
    parser.add_argument('--time-order-manifest', type=str, default=None, metavar='FILE', help='时间顺序清单（JSONL 或每行一个 ISO-8601 时间戳，按语料文件顺序）。仅 --split-policy chronological 时必填；时间戳非单调时拒绝启动')
    parser.add_argument('--evaluate-test', action='store_true', help='训练结束后在测试集上做唯一一次最终评估；不设置时测试集完全不被访问')
    parser.add_argument('--test-ledger', type=str, default=None, metavar='FILE', help='研究级测试集访问账本（JSONL，只追加）。建议放在所有候选共享的目录：每次 --evaluate-test 记录数据/测试 split 指纹、候选目录、代码指纹与时间；同一测试集被多个候选访问时显式披露。默认写入输出目录内，只防同目录重复')
    parser.add_argument('--allow-repeat-test-access', action='store_true', help='允许同一测试集被其它候选/目录再次解封（默认拒绝）。启用后最终报告必须列出全部访问；该豁免写入账本')
    parser.add_argument('--study-id', type=str, default=None, help='研究/确认阶段标识，写入 test 记录与访问账本，用于把多次运行归到同一个预注册比较')
    parser.add_argument('--final-test-comparison', nargs='+', default=None, metavar='RUN_DIR', help='只读汇总冻结候选的 test 记录（不重新评估、不选择）：列出全部候选、缺失 test 的候选与访问账本重复次数。最终确认应先冻结候选再一次性解封测试集')
    parser.add_argument('--group-separator', type=str, default=None, help='群组数据协议：把语料按该分隔符切成「群组」（如文档/用户记录分隔符，支持 \\n、\\xHH 等转义），train/val/test 的切分位置吸附到群组边界，任何群组都不会跨 split（防止同一群组的信息同时出现在训练与验证/测试两侧）；不设置时按单一字节流在名义位置连续切分')
    parser.add_argument('--deduplicate-train-groups', action='store_true', help='训练集群组级去重（需要 --group-separator）：只从 train 中剔除与 val/test 或更早训练群组内容完全相同的群组，被剔除的字节不进入任何训练窗口；val/test 的字节与切分位置完全不变。用于清洗跨 split 重复文档造成的泄露与训练集重复样本')
    parser.add_argument('--drop-train-shared-chunks', action='store_true', help='训练集跨 split 近似重复清洗（需要 --group-separator）：用内容定义分块找出与 val （默认；显式 --clean-train-against-test 时含 test）共享至少一个完整窗口长度连续块的训练群组，并整组剔除；精确群组去重抓不到的「长群组里嵌入评估片段/错位复制」泄露也会被清除。val/test 的字节与切分位置完全不变')
    parser.add_argument('--drop-train-near-duplicate-groups', action='store_true', help='训练集近重复清洗（需要 --group-separator）：按字节 k-shingle 集合的 Jaccard 相似度，把与 val （默认；显式 --clean-train-against-test 时含 test）或更早 train 群组相似度达到阈值的训练群组整组剔除；精确群组指纹与内容定义分块抓不到的改写/转载/轻微编辑重复会被这条内容级规则发现。val/test 的字节、切分位置与指纹完全不变')
    parser.add_argument('--near-duplicate-audit', action='store_true', help='只读近重复审计（需要 --group-separator）：不删除任何数据。data_audit.near_duplicates 报告与 val（解封时含 test）或更早 train 近似的训练群组；data_audit.evaluation_near_duplicates 另报告 val/test 内部与 val↔test 的近重复评估群组、独立簇数与有效群组数，用于判断 bootstrap/选型的有效样本是否被近重复内容虚增。两处都给出最佳匹配位置、Jaccard 值与哈希/草图口径。用于先看清重复程度再决定是否启用 --drop-train-near-duplicate-groups')
    parser.add_argument('--near-duplicate-threshold', type=float, default=_NEAR_DUP_DEFAULT_THRESHOLD, metavar='JACCARD', help=f'近重复判定的 Jaccard 阈值（0, 1]，默认 {_NEAR_DUP_DEFAULT_THRESHOLD}；相似度按完整 shingle 集合精确复核，草图只用于候选生成。阈值决定删除比例，必须与清洗开关一起写入预注册')
    parser.add_argument('--near-duplicate-shingle-bytes', type=int, default=_NEAR_DUP_DEFAULT_SHINGLE_BYTES, metavar='BYTES', help=f'近重复相似度的字节 shingle 长度，默认 {_NEAR_DUP_DEFAULT_SHINGLE_BYTES}；越短越敏感于局部改写、候选越多，越长越接近整段复制检测。允许 1..{_NEAR_DUP_MAX_SHINGLE_BYTES}')
    parser.add_argument('--clean-train-against-test', action='store_true', help='允许训练集去重/共享块清洗读取 test 内容（默认禁止）：默认只对照 val 与更早 train 群组，test 保持尘封，train/test 重复留到解封时由泄露门拦截；显式开启会让训练样本集合依赖 test，该事实写入 config.json 并参与受控协议核对')
    parser.add_argument('--drop-train-corrupt-groups', action='store_true', help='训练集标签噪声清洗（需要 --group-separator）：只从 train 中整组剔除包含非法 UTF-8 或 NUL 字节的群组，被剔除的字节不进入训练窗口、n-gram/参考基线或分布审计；val/test 的字节与切分位置完全不变。用于实际清洗标签噪声，而不只是拒绝或告警')
    parser.add_argument('--drop-train-corrupt-bytes', action='store_true', help='训练集标签噪声清洗（无 --group-separator 的连续字节流）：只从 train 中剔除非法 UTF-8 与 NUL 字节本身，被剔除的字节不进入训练窗口、n-gram/参考基线或分布审计；val/test 的字节与切分位置完全不变。有群组边界时请改用 --drop-train-corrupt-groups 以保持群组完整')
    parser.add_argument('--drop-train-degenerate-groups', action='store_true', help=f'训练集异常值清洗（需要 --group-separator）：只从 train 中整组剔除低信息/退化群组（长度 >= {_AUDIT_DEGENERATE_GROUP_MIN_BYTES}B 且单一字节占比 >= {_AUDIT_DEGENERATE_GROUP_TOP_SHARE}），如填充/重复字符污染；val/test 的字节与切分位置完全不变')
    parser.add_argument('--drop-train-degenerate-runs', action='store_true', help=f'训练集异常值清洗（无 --group-separator 的连续字节流）：只从 train 中剔除长度 >= {_AUDIT_DEGENERATE_RUN_MIN_BYTES}B 的单一字节连续段（填充/重复字符污染），val/test 的字节与切分位置完全不变；有群组边界时请改用 --drop-train-degenerate-groups')
    parser.add_argument('--domain-justification', type=str, default=None, metavar='TEXT', help='启用任一训练集清洗开关时必填：说明这些字节在本任务中确实是损坏/噪声（而不是合法字节、结构分隔或低频但有效的标签），并记录清洗规则的选择依据。非法 UTF-8/NUL/重复字节本身不等于标签错误')
    parser.add_argument('--allow-eval-content-for-cleaning', action='store_true', help='允许用 val 与 test 内容共同决定训练集删除哪些群组/共享块（默认只用 val）。启用测试内容参与数据集构建后，不能再声称「测试集全程未被读取」；该事实与 test 内容访问范围写入 data_audit 与config.json，且清洗规则必须在训练前冻结')
    parser.add_argument('--strict-data-audit', action='store_true', help='严格数据审计：训练前的 train/val 审计或解封测试集时的审计只要发现越界 token 或任何 UTF-8/NUL 损坏等标签噪声，就拒绝开始训练/拒绝解封测试集（跨 split 重复内容已由默认泄露门拦截）')
    for shift_flag, shift_label in (('--max-byte-shift', '单字节分布 JS'), ('--max-bigram-shift', '字节二元组分布 JS'), ('--max-conditional-shift', '条件标签分布 JS')):
        parser.add_argument(shift_flag, type=float, default=None, metavar='JS', help=f'分布偏移硬门（0~1）：{shift_label} 超过该值时拒绝开始训练/拒绝解封测试集' + ('；该阈值同时约束 train/val 边界的尾/头块' if shift_flag == '--max-byte-shift' else '') + '；不设置时只警告')
    parser.add_argument('--allow-data-leakage', action='store_true', help='显式允许跨 split 重复内容继续运行：默认情况下只要审计发现 train/val/test 之间的重复窗口或重复群组（泄露级问题），训练入口和测试集解封都会直接拒绝。仅在你确认这些重复在业务上不是泄露时使用；豁免会记录进 config.json')
    parser.add_argument('--allow-corrupt-test-labels', action='store_true', help='显式允许在含非法 UTF-8/NUL/越界 token 的测试集上解封评估：测试集不参与清洗，默认拒绝在这种标签噪声上产出最终结论；仅当这些字节确实是目标分布的一部分时使用，豁免会记录进 config.json')
    parser.add_argument('--allow-eval-duplicates', action='store_true', help='显式允许在 val/test 内部含完全重复群组时继续运行：重复评估样本会虚增有效样本量、让固定窗口 bootstrap 区间偏窄，验证集重复还会放大选型过拟合。默认拒绝训练/拒绝解封测试集；仅当重复确实属于目标分布时使用，豁免会记录进 config.json')
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-kv-heads', type=int, default=None, help='GQA 的 KV 头数（必须整除 n_heads）；默认等于 n_heads（标准 MHA）。SMELT 论文用 GQA 匹配 KV cache，复现论文口径时可设 n_heads/4')
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--n-experts', type=int, default=8)
    parser.add_argument('--expert-hidden', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--k1', type=int, default=2)
    parser.add_argument('--k2', type=int, default=2)
    parser.add_argument('--balance-coef', type=float, default=0.01)
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--residual-scaling', type=str, default='paper', choices=sorted(Config.RESIDUAL_SCALINGS), help='残差缩放口径：paper（默认）在循环区域内把子层残差更新除以循环次数 r（SMELT 论文 3.1 节，r=2 即 1/2）；global 为历史行为（residual_scale 全局统一作用于循环区域内外）。跨模式对比必须一致，且该口径进入 comparison_protocol')
    parser.add_argument('--budget-match', type=str, default=None, metavar='MODE', help='按完整计算图（Attention 执行次数 + MoE 专家assignment + 路由）核对与参考模式的每 token FLOPs 是否匹配，不匹配时拒绝启动并给出建议 k2。例如 --mode A --budget-match smelt。注意仅令 k1*k2≈2*top_k 会漏掉第一阶段成本，不是充分的预算匹配')
    parser.add_argument('--budget-tolerance', type=float, default=0.1, help='--budget-match 的允许相对误差（默认 0.10 = 10%%；论文的预算匹配残差在 FLOPs 上 <4%%，本原型按整数k2 只能取离散值，默认放宽到 10%% 但仍会报告实测比例）')
    parser.add_argument('--paper-cell', type=str, default=None, metavar='SCALE:SPARSITY', help='直接用论文 arXiv:2609.01343v1 附录 A 的网格单元初始化模型结构，例如 200M:95（也接受 200M@95）。H、L、专家数、top-8、循环布局、4096 上下文与KV cache 匹配都由预设给出；显式冲突的 --d-model 等参数会被拒绝，避免「看起来按论文配置、实际被改动」。与 --paper-arch 配合选择 base / smelt')
    parser.add_argument('--paper-arch', type=str, default=None, choices=sorted(PAPER_ARCHS), help='论文网格的架构臂：base（未循环 MoE 基线）或 smelt（middle 循环两次）。未指定时沿用 --mode（若 --mode 为 base/smelt），否则默认 smelt')
    parser.add_argument('--paper-protocol', action='store_true', help='套用论文 3.1 节训练协议：WSD 调度、全局 batch 256、稳定期 196075 步、4096 packed context 的文档级 segment attention mask，并在六个分支点（10k/20k/50k/100k/150k/196075）存档。需要 --group-separator 标注文档边界。论文未披露的学习率/warmup/weight decay 沿用命令行或仓库默认值，并被 paper_fidelity 标为 undisclosed')
    parser.add_argument('--lr-schedule', type=str, default='cosine', choices=('cosine', 'wsd'), help='学习率调度：cosine（历史默认：linear warmup 后余弦退火到 10%%）；wsd（论文 3.1 节的 warmup-stable-decay：warmup 后恒定，--wsd-stable-steps 之后余弦衰减到 --wsd-final-lr-ratio）。WSD 分支实验用 --resume 从分支点检查点续跑，并把 --wsd-stable-steps 设为分支点')
    parser.add_argument('--wsd-stable-steps', type=int, default=None, help='WSD 稳定期结束的绝对步数（从 0 开始计）；默认等于 --steps（即整段训练都处于稳定期，不衰减）。分支实验把它设为分支点')
    parser.add_argument('--wsd-final-lr-ratio', type=float, default=0.1, help='WSD 衰减终点相对峰值学习率的比例（默认 0.1，与历史 cosine 调度的 10%% 终点一致；论文未披露衰减终点，属于 undisclosed 量）')
    parser.add_argument('--save-at-steps', type=int, nargs='+', default=None, metavar='STEP', help='在这些绝对步数额外保存 step_<STEP>.pt 检查点，供 WSD 的六个余弦衰减分支从同一条稳定期轨迹分叉；--paper-protocol 默认使用论文的 10k/20k/50k/100k/150k/196075')
    parser.add_argument('--document-masks', action='store_true', help='在 4096 packed context 内使用文档级 segment attention mask（论文 3.1 节）：需要 --group-separator 给出文档边界；注意力只允许同一文档内的 token 互相可见，跨文档的键被屏蔽')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad-checkpoint', action='store_true', help='对 Block 前向启用 activation checkpointing')
    parser.add_argument('--compile', action='store_true', help='使用 torch.compile 编译模型前向')
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--eval-bootstrap', type=int, default=500, metavar='N', help='最终验证与（唯一一次）测试评估的评估窗口 bootstrap 重采样次数（默认 500，0=关闭）：报告 loss/ppl/token accuracy/macro F1 的 95%% 置信区间，区分方法差异与评估窗口抽样噪声。重采样单位是固定评估窗口（不是 token），使用由数据指纹派生的确定性种子，不扰动训练 RNG')
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--early-stop-patience', type=int, default=0, help='按验证损失早停：连续 N 次周期验证无改善后提前结束训练（0=关闭）；早停状态跨 --resume 从 metrics.jsonl恢复')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0, help='早停判定的最小改善幅度：验证损失需下降超过该值才算改善（默认 0，即任何下降都算）')
    parser.add_argument('--bad-cases', type=int, default=10, help='训练结束后在验证集上输出损失最高的 N 个坏案例（0=关闭）；--evaluate-test 时同一次解封中也输出测试集坏案例')
    parser.add_argument('--skip-efficiency', action='store_true', help='跳过训练结束后的效率测量')
    parser.add_argument('--skip-routing-analysis', action='store_true', help='跳过训练结束后的 RQ3/RQ4 路由诊断（第二轮与第一轮的专家重合度 / Jaccard、专家负载分布与路由熵、A/B 分支表示相似度）；默认定性诊断在验证集固定窗口、验证集选定模型上执行一次，测试集不参与')
    parser.add_argument('--routing-interventions', action='store_true', help='在验证集固定窗口上追加 RQ4 的功能干预实验：路由打乱（随机专家）与逐专家屏蔽（输出置零），比较损失上升。负载均匀/熵高本身不能证明专家专门化；该选项额外做 E+2 次验证前向，默认关闭')
    parser.add_argument('--routing-intervention-batches', type=int, default=None, metavar='N', help='干预实验使用的验证窗口批数；默认 min(eval_batches, 5)，设为 0 表示与 --eval-batches 相同')
    parser.add_argument('--compare', nargs='+', default=None, metavar='RUN_DIR', help='不训练，直接对比多个运行目录的指标后退出（含受控变量检查、按种子聚合的 Mean ± Std 与单因子显著性检验）')
    parser.add_argument('--compare-report', type=str, default=None, metavar='FILE', help='配合 --compare：把完整对比分析（含 fragility_summary、聚合、检验、功效/等价性与消融检查）写入指定 JSON 文件，而不是只打印到终端。只读入口默认不写文件；显式给出该路径才落盘，便于把结果分析随运行一起归档')
    parser.add_argument('--write-manifest', type=str, default=None, metavar='RUN_DIR', help='不训练：为运行/归档目录生成 artifact_manifest.json（逐文件 SHA-256、大小、符号链接目标，外加生成时的代码与解释器版本）。必须在所有训练/评估进程结束后执行，清单才代表最终归档内容；--verify-manifest 可逐项复核')
    parser.add_argument('--verify-manifest', type=str, default=None, metavar='RUN_DIR', help='不训练：复核目录内的 artifact_manifest.json，报告缺失、大小/SHA-256 不符、类型变化与清单外新增文件；完整性失败时以非零状态码退出')
    parser.add_argument('--verify-manifest-strict', action='store_true', help='与 --verify-manifest 配合：清单之外的任何新增文件也算失败（严格归档逐字一致）；默认只报告untracked，不把后续追加的日志误判为损坏')
    parser.add_argument('--check-environment', type=str, default=None, metavar='RUN_DIR_OR_JSON', help='不训练：把当前环境与运行目录的 environment.json（或 config.json 中的环境快照）逐项比较，environment_id 不一致时以非零状态码退出。用于重跑/续跑/换机器前确认关键依赖、编译选项、设备与确定性开关相同；默认只要求关键环境一致，--check-environment-strict 还要求完整包清单一致')
    parser.add_argument('--check-environment-strict', action='store_true', help='与 --check-environment 配合：完整包清单（含与数值无关的包）也必须逐项一致，任何新增/删除/版本变化都判失败')
    parser.add_argument('--expect-manifest-sha256', type=str, default=None, metavar='HEX', help='与 --verify-manifest 配合：要求清单自身的 SHA-256等于外部归档记录中的值（64 位十六进制）。只校验清单内文件不足以防清单被整体替换；把 --write-manifest打印的指纹写入 DOI/归档记录后，用本参数才能验证「被校验的清单就是被记录的那一份」')
    parser.add_argument('--audit-run', type=str, default=None, metavar='RUN_DIR', help='不训练：对已完成的运行目录做只读复现审计，核对 config.json、environment.json、metrics.jsonl、test 封条/账本、sanity 重复运行与 artifact_manifest 之间的身份是否互相一致；发现不一致时以非零状态码退出。用于归档前确认「这些文件确实属于同一次运行」')
    parser.add_argument('--audit-report', type=str, default=None, metavar='FILE', help='配合 --audit-run：把完整审计报告写入指定 JSON 文件（默认只打印）；建议写在运行目录之外，或在归档前重新生成 artifact_manifest')
    parser.add_argument('--audit-source', type=str, default=None, metavar='FILE', help='配合 --audit-run：核对归档源码的 SHA-256 是否等于运行记录中的 script_sha256（防止源码包与结果不匹配）')
    parser.add_argument('--audit-data', type=str, default=None, metavar='FILE', help='配合 --audit-run：核对归档语料的 SHA-256 是否等于 config.data.sha256')
    parser.add_argument('--audit-preregistration', type=str, default=None, metavar='FILE', help='配合 --audit-run：核对预注册文件的 SHA-256 与关键字段是否仍与运行记录一致（发现运行后编辑预注册）')
    parser.add_argument('--audit-strict', action='store_true', help='配合 --audit-run：把可复现性证据不足的警告也判为失败（未固定 PYTHONHASHSEED、未启用确定性算子、git 工作区脏、缺少 artifact_manifest 等），用于正式归档前的硬门')
    parser.add_argument('--compare-factors', type=str, default=','.join(DEFAULT_COMPARISON_FACTORS), help='对比时允许跨运行不同的实验因子（逗号分隔）；其余协议项（优化器/学习率/训练步数/数据增强/数据指纹等）必须完全一致，否则报警。默认: ' + ','.join(DEFAULT_COMPARISON_FACTORS))

def _MAIN_part_001():
    global actual_manifest_sha, alpha, applied_fields, args, audit_error, audit_report, environment_error, error, expected_manifest_sha, group_separator_bytes, hashseed_error, lr, manifest_error, manifest_path, matches_expected, missing_pilot, omitted_modes, order_seed_message, parsed_factors, pilot_conflicts, prereg_power, preregistration_error, preregistration_payload, preregistration_sha256, report, report_path, seed, seed_message, suite_error, tuning_labels
    parser.add_argument('--compare-baseline', type=str, default=None, metavar='MODE', help='以指定模式作为基线，在统计报告中给出其余实验单元相对基线的差值（如 smelt 或 local_base）；同时核对基线是否真的强于无上下文 unigram 参考与调优好的 n-gram（稻草人守卫：战胜弱基线不能证明复杂模型的收益）')
    parser.add_argument('--compare-metric', type=str, default='val_loss', help='显著性检验使用的指标（默认 val_loss）；val_loss 来自验证集选中的 checkpoint，跨运行比较有选型偏差，建议同时/改用 val_loss_at_common_step（共同训练步数处的即时验证损失，不经过选型）、val_loss_curve_mean_to_common_step（共同窗口内的时均验证损失，用于 RQ1 收敛速度）或 val_loss_at_matched_flops（等训练计算量）；只允许 val 口径指标（测试集尘封协议：test 指标不参与跨运行对比与选择，只在单个运行的最终评估中出现）；可选: ' + ', '.join(COMPARISON_METRIC_KEYS))
    parser.add_argument('--common-step', type=float, default=None, metavar='STEP', help='预注册的共同训练步数，用于 <metric>_at_common_step 派生指标。默认取各运行最后一个周期验证步的最小值，该步由观察到的端点决定（早停/不等步数时与结果相关），不是选型无关口径；正式对比应在训练前固定该步')
    parser.add_argument('--eval-bootstrap-block', type=int, default=1, metavar='N', help='评估窗口 bootstrap 的重采样单位：1=逐窗口 i.i.d.（默认，可能低估序列相关的不确定性）；N>1 时使用长度为 N 的循环移动块 bootstrap（连续窗口成块重采样，部分吸收局部自相关）。按文档/来源做 cluster bootstrap 仍是更合适的选择；本选项只缓解窗口级近似')
    parser.add_argument('--equivalence-margin', type=float, default=_DEFAULT_EQUIVALENCE_MARGIN, metavar='REL', help=f'等价性检验（TOST）的等价界，相对基线均值的比例（默认 {_DEFAULT_EQUIVALENCE_MARGIN:g}，即 1%%）：「不显著」只能说明未发现差异，不等于无差异；报告在预先声明的等价界内给出 TOST 判定，以及 80%% 功效下的最小可检出效应（MDE）与所需种子数，避免把功效不足误读成方法等价。若要把等价性作为研究主张，必须在预注册 power 中设 plan_equivalence=true 并在顶层声明该界，设计预检会同时检查 TOST 功效；命令行默认值不能代替声明')
    parser.add_argument('--plan-design', action='store_true', help='不训练：按 --seeds/--suite-modes/--suite-ablation 只生成并打印 design_plan.json（运行清单、随机化区组顺序、先验功效、解析 token 预算），供冻结/预注册前检查设计是否可执行；正式运行时同一份计划由子进程逐项核对')
    parser.add_argument('--design-pilot-runs', nargs='+', default=None, metavar='RUN', help='试点/先前运行目录（必须包含失败目录）：从相同 seed 的配对差值估计实际种子间 sd，并按处理单元估计脱落率，与预注册 power 的 expected_seed_sd_relative / expected_attrition_rate 对照。观测值更大时设计计划按观测值强制重算功效并写入 pilot_evidence；仍不足则在第一个子进程前拒绝。需配合 --preregistration 与 --seeds/--plan-design')
    parser.add_argument('--run-order', type=str, default=None, choices=_RUN_ORDER_SCHEMES, help='套件的执行顺序设计：grouped=按臂分组（历史行为，时间漂移与臂效应完全混杂）；blocked=按种子成完整区组、区组内保持原臂顺序；randomized=区组内随机化臂顺序（默认，随机化完整区组设计）；counterbalanced=在完整区组上用循环拉丁方限制性随机化，使每个处理在各区组位置上的出现次数尽可能相等（要求每个种子区组包含同一组处理）')
    parser.add_argument('--run-order-seed', type=int, default=None, metavar='SEED', help='随机化区组顺序的确定性种子（默认 0）；同一计划必须使用同一种子才能复现执行顺序，看到结果后换顺序不再是原设计')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, metavar='SEED', help='多随机种子运行：每个种子训练一次到 <out>/seed<SEED>，结束后自动给出 Mean ± Std 与显著性检验（关键实验建议 3~5 个种子）')
    parser.add_argument('--suite-modes', nargs='+', choices=sorted(Config.MODEL_MODES), default=None, metavar='MODE', help='配合 --seeds：把同一套受控设置依次应用到多个模式（输出 <out>/<mode>_seed<SEED>），用于严格控变量的模式对比')
    parser.add_argument('--suite-ablation', action='store_true', help='配合 --seeds：运行「延迟合并 × 均衡损失」全因子消融（立即合并对照 local_base 与各延迟合并方案 A/B × balance_coef=0/当前值）；balance 关闭臂为 0，开启臂取 --balance-coef。每个创新点都必须证明各自有用，默认覆盖 A 与 B 两个方案')
    parser.add_argument('--ablation-modes', nargs='+', choices=sorted(_DELAYED_MERGE_MODES), default=None, metavar='MODE', help='配合 --suite-ablation：声明本轮消融必须覆盖的延迟合并方案（默认 A B）；只跑部分方案时结论只能覆盖已跑方案，覆盖检查会按声明核对')
    parser.add_argument('--allow-partial-ablation', action='store_true', help='显式允许 --suite-ablation 只覆盖 --ablation-modes 声明的部分延迟合并方案（阶段性调试用）。报告会记录 partial 与未覆盖的项目方案；正式结论仍须覆盖 A 与 B 全部方案')
    parser.add_argument('--allow-underpowered', action='store_true', help='显式允许套件在少于 3 个随机种子下运行（低统计功效，仅用于冒烟/调试；关键结论不得据此做显著性主张，豁免会记录进报告）')
    parser.add_argument('--suite-baseline-tuning', action='store_true', help='配合 --seeds：对简单基线（--baseline-tuning-mode）做预注册的学习率网格搜索（--baseline-tuning-lrs，训练配置固定），在 val 上选 winner 并写出 baseline_tuning.json；没调过参的弱基线只能证明复杂模型比稻草人强')
    parser.add_argument('--baseline-tuning-lrs', nargs='+', type=float, default=list(_DEFAULT_BASELINE_TUNING_LRS), help='基线调参的学习率候选网格（默认 ' + ' '.join((format(lr, 'g') for lr in _DEFAULT_BASELINE_TUNING_LRS)) + '）；候选之间只有学习率不同，其余设置逐项一致')
    parser.add_argument('--baseline-tuning-mode', type=str, default='dense', choices=sorted(Config.MODEL_MODES), help='基线调参的模式（默认 dense：最简单的神经基线；先调强简单基线，再谈复杂模型）')
    parser.add_argument('--baseline-tuning-dense-hiddens', nargs='+', type=int, default=[], metavar='N', help='基线调参的容量轴候选（dense 模式的 FFN 隐藏宽度）：与学习率轴一起构成坐标式多轴调参——每个候选相对基线配置只改一个轴，逐轴判定 winner。只对 dense 模式生效')
    parser.add_argument('--baseline-tuning-weight-decays', nargs='+', type=float, default=[], metavar='WD', help='基线调参的正则轴候选（权重衰减）：与学习率/容量轴一起构成坐标式多轴调参——每个候选相对基线配置只改一个轴，逐轴判定 winner')
    parser.add_argument('--baseline-tuning-metric', type=str, default=None, choices=_BASELINE_TUNING_METRIC_KEYS, metavar='METRIC', help='基线调参的 val 选择指标；缺省时优先跟随 --primary-metric（预注册主指标），否则用 --compare-metric。在 val_loss 上调强的基线不能证明它在 macro F1 / PR-AUC 上也已调强')
    parser.add_argument('--baseline-tuning-from', type=str, default=None, metavar='PATH', help='把 --suite-baseline-tuning 生成的调参报告作为本次基线训练的调参来源：核对模式/学习率/训练步数/数据指纹及完整训练协议（warmup/调度/weight decay/数据划分等）与调参时逐项一致，并写入 config.json 的 baseline_tuning 字段，供基线合理性检查核对')
    parser.add_argument('--skip-sanity-check', action='store_true', help='跳过全量训练前的过拟合小批量 sanity check（默认在新训练开始时运行）')
    parser.add_argument('--sanity-only', action='store_true', help='只运行 sanity check 并写出结果，不开始全量训练')
    parser.add_argument('--require-pythonhashseed', action='store_true', help='要求进程启动前显式设置 PYTHONHASHSEED（set_seed 的 setdefault 无法追溯地固定当前解释器的哈希随机化）；未设置时直接报错退出，用于严格逐位复现的正式运行')
    parser.add_argument('--strict-determinism', action='store_true', help='遇到没有确定性实现的 PyTorch 算子时直接报错（默认仅告警并继续）；正式复现实验建议开启')
    parser.add_argument('--torch-threads', type=int, default=None, metavar='N', help='固定 PyTorch CPU 线程数（默认沿用环境设置）。CPU 归约顺序可能随线程数变化，严格复现时显式固定；实际生效值写入 config.json 的 run_metadata')
    parser.add_argument('--allow-tf32', action='store_true', help='允许 CUDA fp32 矩阵乘使用 TF32（默认关闭）。TF32 只保留 10 位尾数，会改变数值结果，且不同 GPU 架构上的实现不完全一致；关闭时使用完整 fp32，同一台机器上的同种子结果更稳定。开启状态会写入 run_metadata 并参与 --compare 的复现性核对')
    parser.add_argument('--sanity-batch-size', type=int, default=4)
    parser.add_argument('--sanity-steps', type=int, default=300)
    parser.add_argument('--sanity-lr', type=float, default=0.003)
    parser.add_argument('--sanity-repeats', type=int, default=2, metavar='N', help='sanity check 的重复运行次数（默认 2）：第一次通过后再用完全相同的模型/批次/种子训练一次，要求初始与最终损失、准确率和梯度范数逐位一致，作为「同一台机器上同种子可复现」的运行时证据；不一致会拒绝开始全量训练。设为 1 可关闭重复校验')
    parser.add_argument('--sanity-loss-threshold', type=float, default=0.1)
    parser.add_argument('--sanity-acc-threshold', type=float, default=0.99)
    parser.add_argument('--sanity-min-loss-reduction', type=float, default=0.5, metavar='REL', help='sanity check 要求最终 loss 相对第一次前向至少下降的比例（默认 0.5 = 50%%）：只满足绝对阈值或准确率阈值还不够，必须观察到真实的学习信号，避免恒定小损失等实现错误假通过')
    parser.add_argument('--note', type=str, default='', help='本轮实验备注，随 config.json 一起存档')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    _normalize_clean_against_test(args)
    if args.compare_report is not None and args.compare is None:
        parser.error('--compare-report 只能配合 --compare 使用：它保存的是只读对比分析，不参与训练')
    audit_error = audit_mode_error(args)
    if audit_error is not None:
        parser.error(audit_error)
    if args.audit_run is not None:
        try:
            audit_report = audit_run_directory(Path(args.audit_run), source_path=args.audit_source, data_path=args.audit_data, preregistration_path=args.audit_preregistration, strict=bool(args.audit_strict))
        except (OSError, ValueError) as error:
            parser.error(f'--audit-run 失败: {error}')
        print_run_audit(audit_report)
        if args.audit_report is not None:
            report_path = Path(args.audit_report)
            try:
                _atomic_write_json(report_path, audit_report)
            except OSError as error:
                parser.error(f'--audit-report 写入失败: {error}')
            print(f'[audit] 完整报告已写入 {report_path}（sha256={_sha256_file(report_path)}）')
        else:
            print(json.dumps(_json_safe(audit_report), ensure_ascii=False, indent=2))
        if not audit_report['ok']:
            raise SystemExit(1)
        raise _SC_RETURN(None)
    environment_error = environment_mode_error(args)
    if environment_error is not None:
        parser.error(environment_error)
    if args.check_environment is not None:
        try:
            report = check_environment_snapshot(Path(args.check_environment), strict=bool(args.check_environment_strict))
        except (OSError, ValueError) as error:
            parser.error(f'--check-environment 失败: {error}')
        print_environment_check(report)
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
        if not report['ok']:
            raise SystemExit(1)
        raise _SC_RETURN(None)
    manifest_error = manifest_mode_error(args)
    if manifest_error is not None:
        parser.error(manifest_error)
    if args.write_manifest is not None:
        try:
            manifest_path = write_artifact_manifest(Path(args.write_manifest))
        except (OSError, ValueError) as error:
            parser.error(f'--write-manifest 失败: {error}')
        print(f'[manifest] 已写入 {manifest_path}（sha256={_sha256_file(manifest_path)}）；请把这个清单指纹记入外部归档记录，并确认所有训练/评估进程已结束')
        raise _SC_RETURN(None)
    if args.verify_manifest is not None:
        report = verify_artifact_manifest(Path(args.verify_manifest), strict=bool(args.verify_manifest_strict))
        expected_manifest_sha = args.expect_manifest_sha256
        if expected_manifest_sha is not None:
            expected_manifest_sha = expected_manifest_sha.lower()
            actual_manifest_sha = report.get('manifest_sha256')
            matches_expected = isinstance(actual_manifest_sha, str) and actual_manifest_sha.lower() == expected_manifest_sha
            report['expected_manifest_sha256'] = expected_manifest_sha
            report['manifest_sha256_matches_expected'] = matches_expected
            report['ok'] = bool(report['ok']) and matches_expected
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
        if not report['ok']:
            raise SystemExit(1)
        raise _SC_RETURN(None)
    apply_paper_cli(args, parser)
    if args.preregistration is not None:
        if any((flag for flag in (args.compare is not None, args.final_test_comparison is not None, args.sanity_only, args.statistical_baseline, args.suite_baseline_tuning))):
            parser.error('--preregistration 只适用于神经训练/多种子确证运行：--compare / --final-test-comparison / --sanity-only / --statistical-baseline / --suite-baseline-tuning 不执行预注册的训练协议')
        try:
            preregistration_payload, preregistration_sha256 = load_preregistration(args.preregistration)
        except ValueError as error:
            parser.error(str(error))
        applied_fields, preregistration_error = preregistration_apply(args, preregistration_payload, current_commit=_git_output('rev-parse', 'HEAD'), working_tree_dirty=bool(_git_output('status', '--porcelain')), current_script_sha256=_script_sha256(), frozen_code_matches=preregistration_frozen_commit_status(str(preregistration_payload['frozen_commit'])))
        if preregistration_error is not None:
            parser.error(preregistration_error)
        args.preregistration_record = preregistration_record(preregistration_payload, args.preregistration, preregistration_sha256, applied_fields, args)
        print(f'[prereg] study_id={preregistration_payload['study_id']}，冻结提交 {str(preregistration_payload['frozen_commit'])[:12]}...，种子={preregistration_payload['seeds']}' + ('；由文件补齐: ' + ', '.join(applied_fields) if applied_fields else '；命令行与预注册逐项一致'))
    if args.design_pilot_runs is not None:
        pilot_conflicts = [flag for flag, active in (('--compare', args.compare is not None), ('--final-test-comparison', args.final_test_comparison is not None), ('--evaluate-test', bool(args.evaluate_test)), ('--resume', args.resume is not None), ('--sanity-only', bool(args.sanity_only)), ('--statistical-baseline', bool(args.statistical_baseline)), ('--suite-baseline-tuning', bool(args.suite_baseline_tuning))) if active]
        if pilot_conflicts:
            parser.error('--design-pilot-runs 只用于训练前的设计校准，不能与以下只读/评估/恢复入口同时使用: ' + ', '.join(pilot_conflicts))
        if args.preregistration is None:
            parser.error('--design-pilot-runs 需要 --preregistration：试点证据必须对照预注册 power 中声明的 expected_seed_sd_relative 与 expected_attrition_rate，不能只生成一份未冻结的描述性统计')
        if args.seeds is None:
            parser.error('--design-pilot-runs 需要 --seeds（或由预注册文件补齐 seeds）：试点证据用于设计计划，单次训练没有计划可校准')
        prereg_power = None
        if isinstance(getattr(args, 'preregistration_record', None), dict):
            prereg_power = args.preregistration_record.get('power')
        if not isinstance(prereg_power, dict):
            parser.error('--design-pilot-runs 需要预注册文件包含 power 块：没有 expected_seed_sd_relative / expected_attrition_rate 就没有可对照的先验声明')
        missing_pilot = [path for path in args.design_pilot_runs if not _path_is_dir(path)]
        if missing_pilot:
            parser.error('--design-pilot-runs 的试点运行目录不存在: ' + ', '.join(missing_pilot))
    if args.wsd_stable_steps is not None and args.wsd_stable_steps < 0:
        parser.error('--wsd-stable-steps 不能为负')
    if args.wsd_stable_steps is not None:
        args.wsd_stable_steps = min(int(args.wsd_stable_steps), int(args.steps))
    if not math.isfinite(args.wsd_final_lr_ratio) or not 0.0 <= args.wsd_final_lr_ratio <= 1.0:
        parser.error('--wsd-final-lr-ratio 必须落在 [0, 1]')
    if args.save_at_steps is not None:
        args.save_at_steps = sorted({int(step) for step in args.save_at_steps})
        if any((step <= 0 for step in args.save_at_steps)):
            parser.error('--save-at-steps 的步数必须为正整数')
    hashseed_error = pythonhashseed_error(args.require_pythonhashseed)
    if hashseed_error:
        parser.error(hashseed_error)
    seed_message = seed_error(args.seed)
    if seed_message:
        parser.error('--seed 无效: ' + seed_message)
    try:
        configure_cpu_threads(args.torch_threads)
    except (RuntimeError, ValueError) as error:
        parser.error(f'--torch-threads 无法生效: {error}')
    if args.steps <= 0:
        parser.error('--steps 必须为正整数')
    if args.batch_size <= 0:
        parser.error('--batch-size 必须为正整数')
    if args.eval_interval <= 0:
        parser.error('--eval-interval 必须为正整数')
    if args.eval_batches <= 0:
        parser.error('--eval-batches 必须为正整数')
    if args.eval_bootstrap < 0:
        parser.error('--eval-bootstrap 不能为负')
    if args.eval_bootstrap_block <= 0:
        parser.error('--eval-bootstrap-block 必须为正整数')
    if args.routing_intervention_batches is not None and args.routing_intervention_batches < 0:
        parser.error('--routing-intervention-batches 不能为负')
    if args.save_interval <= 0:
        parser.error('--save-interval 必须为正整数')
    if args.warmup_steps < 0:
        parser.error('--warmup-steps 不能为负')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr 必须是有限正数')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight-decay 必须是有限非负数')
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        parser.error('--grad-clip 必须为正数')
    if not math.isfinite(args.equivalence_margin) or args.equivalence_margin <= 0.0:
        parser.error('--equivalence-margin 必须是有限正数（相对基线均值的比例，如 0.01 = 1%）')
    if args.skip_sanity_check and args.sanity_only:
        parser.error('--skip-sanity-check 与 --sanity-only 不能同时使用')
    if args.sanity_only and args.evaluate_test:
        parser.error('--evaluate-test 需要完整训练后的模型，不能与 --sanity-only 同时使用')
    if args.sanity_batch_size <= 0:
        parser.error('--sanity-batch-size 必须为正整数')
    if args.sanity_steps <= 0:
        parser.error('--sanity-steps 必须为正整数')
    if args.sanity_repeats <= 0:
        parser.error('--sanity-repeats 必须为正整数')
    if not math.isfinite(args.sanity_lr) or args.sanity_lr <= 0:
        parser.error('--sanity-lr 必须是有限正数')
    if not math.isfinite(args.sanity_loss_threshold) or args.sanity_loss_threshold < 0:
        parser.error('--sanity-loss-threshold 必须是非负有限数')
    if not 0.0 < args.sanity_acc_threshold <= 1.0:
        parser.error('--sanity-acc-threshold 必须落在 (0, 1]')
    if not math.isfinite(args.sanity_min_loss_reduction) or not 0.0 <= args.sanity_min_loss_reduction < 1.0:
        parser.error('--sanity-min-loss-reduction 必须落在 [0, 1)')
    if not math.isfinite(args.val_fraction) or not 0.0 < args.val_fraction < 1.0:
        parser.error('--val-fraction 必须落在 (0, 1)（验证集用于选型/早停，不能为空）')
    if not math.isfinite(args.test_fraction) or not 0.0 <= args.test_fraction < 1.0:
        parser.error('--test-fraction 必须落在 [0, 1)')
    if args.val_fraction + args.test_fraction >= 1.0:
        parser.error('--val-fraction + --test-fraction 必须小于 1（训练集不能为空）')
    group_separator_bytes = None
    if args.group_separator is not None:
        try:
            group_separator_bytes = parse_group_separator(args.group_separator)
        except ValueError as error:
            parser.error(f'--group-separator 无效: {error}')
    if args.early_stop_patience < 0:
        parser.error('--early-stop-patience 不能为负')
    if not math.isfinite(args.early_stop_min_delta) or args.early_stop_min_delta < 0:
        parser.error('--early-stop-min-delta 必须是有限非负数')
    if args.bad_cases < 0:
        parser.error('--bad-cases 不能为负')
    if args.dense_fnn_hidden is not None and args.dense_fnn_hidden <= 0:
        parser.error('--dense-fnn-hidden 必须为正整数')
    if not 1 <= args.ngram_max_order <= _NGRAM_MAX_ORDER:
        parser.error(f'--ngram-max-order 必须落在 [1, {_NGRAM_MAX_ORDER}]')
    for alpha in args.ngram_alpha_grid:
        if not math.isfinite(alpha) or alpha <= 0:
            parser.error('--ngram-alpha-grid 的每个取值必须是有限正数（α→0 时未见的 (ctx,c) 对会得到零概率与无穷损失）')
    if not args.ngram_alpha_grid:
        parser.error('--ngram-alpha-grid 至少需要一个候选值')
    if args.ngram_tuning_metric is not None and (not args.statistical_baseline):
        parser.error('--ngram-tuning-metric 只适用于 --statistical-baseline：神经模型的调参口径请用 --baseline-tuning-metric / --primary-metric')
    try:
        parsed_factors = parse_factor_spec(args.compare_factors)
    except ValueError as error:
        parser.error(str(error))
    suite_error = suite_factor_error(parsed_factors, args.suite_modes, bool(args.suite_ablation))
    if suite_error is not None:
        parser.error(suite_error)
    if args.compare_metric not in COMPARISON_METRIC_KEYS:
        parser.error('--compare-metric 必须是以下之一: ' + ', '.join(COMPARISON_METRIC_KEYS))
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('--seeds 存在重复的随机种子')
        for seed in args.seeds:
            seed_message = seed_error(seed)
            if seed_message:
                parser.error(f'--seeds 中的 {seed} 无效: {seed_message}')
        if args.compare is not None:
            parser.error('--seeds 不能与 --compare 同时使用（对比已完成的运行请直接用 --compare）')
        if args.resume is not None:
            parser.error('--seeds 不能与 --resume 同时使用（每个种子都必须是独立的全新训练）')
        if args.statistical_baseline:
            parser.error('--statistical-baseline 是确定性统计基线，不受随机种子影响，不能与 --seeds 同时使用')
        if args.sanity_only:
            parser.error('--sanity-only 不能与 --seeds 同时使用')
        if args.evaluate_test:
            parser.error('--evaluate-test 不能与 --seeds 同时使用（多种子套件会对测试集重复解封，让测试集退化为选择指标；请在模型与超参数冻结后，改用全新目录对单个最终运行评估一次测试集）')
        if args.data is None and (not args.plan_design):
            parser.error('--seeds 需要 --data')
        if args.out is None:
            parser.error('--seeds 需要 --out')
    if (args.suite_modes or args.suite_ablation or args.suite_baseline_tuning) and args.seeds is None:
        parser.error('--suite-modes / --suite-ablation / --suite-baseline-tuning 必须配合 --seeds 使用（关键实验至少 3~5 个种子）')
    if args.run_order_seed is not None:
        order_seed_message = seed_error(args.run_order_seed)
        if order_seed_message:
            parser.error(f'--run-order-seed 无效: {order_seed_message}')
    if sum((bool(flag) for flag in (args.suite_modes, args.suite_ablation, args.suite_baseline_tuning))) > 1:
        parser.error('--suite-modes / --suite-ablation / --suite-baseline-tuning 只能同时使用一个')
    if args.suite_modes is not None and len(set(args.suite_modes)) != len(args.suite_modes):
        parser.error('--suite-modes 存在重复的模式')
    if args.suite_ablation and (not args.balance_coef > 0):
        parser.error('--suite-ablation 需要正的 --balance-coef 作为开启臂（关闭臂固定为 0）')
    if args.ablation_modes is not None and (not args.suite_ablation):
        parser.error('--ablation-modes 只适用于 --suite-ablation（用于声明本轮必须覆盖哪些延迟合并方案）')
    if args.ablation_modes is not None and len(set(args.ablation_modes)) != len(args.ablation_modes):
        parser.error('--ablation-modes 存在重复的延迟合并方案（重复方案会重复运行同一消融单元、覆盖输出目录）')
    if args.allow_partial_ablation and (not args.suite_ablation):
        parser.error('--allow-partial-ablation 只适用于 --suite-ablation（用于显式允许阶段性部分消融）')
    if args.suite_ablation and args.ablation_modes is not None:
        omitted_modes = [mode for mode in _DELAYED_MERGE_MODES if mode not in args.ablation_modes]
        if omitted_modes and (not args.allow_partial_ablation):
            parser.error('--ablation-modes 未覆盖项目声明的延迟合并方案 ' + ', '.join(omitted_modes) + '：只跑部分方案不能支撑「全部创新点各自有用」。如确需阶段性部分消融，请显式加 --allow-partial-ablation（报告会记录 partial 与未覆盖方案）')
    if args.allow_underpowered and (not (args.suite_modes or args.suite_ablation or args.suite_baseline_tuning)):
        parser.error('--allow-underpowered 只适用于种子套件（--suite-modes / --suite-ablation / --suite-baseline-tuning）')
    if args.seeds is not None and (args.suite_modes or args.suite_ablation or args.suite_baseline_tuning) and (len(args.seeds) < _MIN_SEEDS_FOR_STATS) and (not args.allow_underpowered) and (not getattr(args, 'plan_design', False)):
        parser.error(f'关键套件（--suite-modes / --suite-ablation / --suite-baseline-tuning）至少需要 {_MIN_SEEDS_FOR_STATS} 个不同随机种子才能支撑统计结论；种子太少时 Mean ± Std 与显著性检验不可信。冒烟/调试请显式加 --allow-underpowered（豁免会记录进报告），正式实验请用 3~5 个种子')
    if args.statistical_baseline and args.resume is not None:
        parser.error('--statistical-baseline 不能与 --resume 同时使用（统计基线没有可恢复的训练状态，检查点会被静默忽略）')
    for lr in args.baseline_tuning_lrs:
        if not math.isfinite(lr) or lr <= 0:
            parser.error('--baseline-tuning-lrs 的每个候选必须是有限正数')
    if len(set(args.baseline_tuning_lrs)) != len(args.baseline_tuning_lrs):
        parser.error('--baseline-tuning-lrs 存在重复候选')
    tuning_labels = [f'lr{format(lr, 'g')}' for lr in args.baseline_tuning_lrs]
    if len(set(tuning_labels)) != len(tuning_labels):
        parser.error('--baseline-tuning-lrs 的候选在输出目录名中格式化后重名（%g 精度不足）：请改用差异更大的学习率候选，避免调参运行互相覆盖')
    if args.suite_baseline_tuning and len(set(args.baseline_tuning_lrs)) < 2:
        parser.error('--suite-baseline-tuning 至少需要两个不同的学习率候选（--baseline-tuning-lrs），单档网格不构成调参')

def _MAIN_part_002():
    global _, amp_change_error, amp_dtype, candidates_error, cfg, checkpoint_git_commit, checkpoint_script_sha256, code_changed, compare_analysis, compare_conflicts, compare_rows, conflicts, current_git_commit, current_script_sha256, current_value, dense_fnn_hidden, design, device, error, evidence_field, flag, formatted, formatter, hypothesis_error, integer, invalid, invalid_compare_paths, item, known_config_fields, numerical_error, out_dir, overwrite_error, plan, plan_root, preregistration_error, report, rng_payload, same_run_dir, saved_args, saved_config, saved_data_effective_train_sha256, saved_data_sha256, saved_data_split_isolation, saved_data_split_sha256, saved_data_test_fraction, saved_data_val_fraction, saved_masks, saved_numerical_state, saved_out, saved_value, scaler, schedule_field, start_step, unknown_config_fields, values
    for flag, values, formatter, integer in (('--baseline-tuning-dense-hiddens', list(args.baseline_tuning_dense_hiddens), lambda value: f'h{value}', True), ('--baseline-tuning-weight-decays', list(args.baseline_tuning_weight_decays), lambda value: f'wd{format(value, 'g')}', False)):
        if not values:
            continue
        candidates_error = baseline_tuning_axis_candidates_error(flag, values, integer)
        if candidates_error is not None:
            parser.error(candidates_error)
        if args.suite_baseline_tuning and len(set(values)) < 2:
            parser.error(f'{flag} 至少需要两个不同的候选，单档不算调参')
        formatted = [formatter(value) for value in values]
        if len(set(formatted)) != len(formatted):
            parser.error(f'{flag} 的候选在输出目录名中格式化后重名：请改用差异更大的候选，避免调参运行互相覆盖')
    if args.baseline_tuning_dense_hiddens and args.baseline_tuning_mode != 'dense':
        parser.error('--baseline-tuning-dense-hiddens 只对 dense 基线生效（--baseline-tuning-mode=dense）：容量轴在其它模式下不改变模型，白烧算力')
    if args.baseline_tuning_from is not None:
        if args.suite_baseline_tuning:
            parser.error('--baseline-tuning-from 不能在调参套件（--suite-baseline-tuning）中使用：调参结束时才会生成报告')
        if args.statistical_baseline:
            parser.error('--baseline-tuning-from 只适用于神经基线训练，统计基线没有学习率')
        if args.compare is not None:
            parser.error('--baseline-tuning-from 需要训练基线，不能与只读的 --compare 同时使用')
        if args.resume is not None:
            parser.error('--baseline-tuning-from 不能与 --resume 同时使用（恢复训练的学习率由检查点决定，调参来源已记录在原运行的 config.json）')
    hypothesis_error = hypothesis_check_error(args)
    if hypothesis_error is not None:
        parser.error(hypothesis_error)
    preregistration_error = hypothesis_preregistration_error(args)
    if preregistration_error is not None:
        parser.error(preregistration_error)
    if args.final_test_comparison is not None:
        invalid = [path for path in args.final_test_comparison if not _path_is_dir(path)]
        if invalid:
            parser.error('--final-test-comparison 的运行目录不存在: ' + ', '.join(invalid))
        report = print_final_test_comparison(list(args.final_test_comparison), args.test_ledger)
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
        raise _SC_RETURN(None)
    if args.compare_report is not None and args.compare is None:
        parser.error('--compare-report 只能配合 --compare 使用：它保存的是只读对比分析，不参与训练')
    if args.compare is not None:
        compare_conflicts = [flag for flag, active in (('--resume', args.resume is not None), ('--evaluate-test', bool(args.evaluate_test)), ('--sanity-only', bool(args.sanity_only)), ('--statistical-baseline', bool(args.statistical_baseline)), ('--suite-baseline-tuning', bool(args.suite_baseline_tuning)), ('--plan-design', bool(args.plan_design)), ('--design-pilot-runs', args.design_pilot_runs is not None)) if active]
        if compare_conflicts:
            parser.error('--compare 是只读对比入口，不能与以下训练/评估参数同时使用（否则会被静默忽略）: ' + ', '.join(compare_conflicts))
        invalid_compare_paths = [path for path in args.compare if not _path_is_dir(path)]
        if invalid_compare_paths:
            parser.error('--compare 的运行目录不存在或不是目录: ' + ', '.join(invalid_compare_paths))
        compare_rows = compare_runs(args.compare)
        print_comparison(compare_rows)
        compare_analysis = print_comparison_analysis(compare_rows, factors=parse_factor_spec(args.compare_factors), baseline_mode=args.compare_baseline, metric=args.compare_metric, equivalence_margin=args.equivalence_margin, common_step=args.common_step)
        if args.compare_report is not None:
            try:
                write_comparison_report_file(Path(args.compare_report), compare_analysis)
            except OSError as error:
                parser.error(f'--compare-report 无法写入: {error}')
        raise _SC_RETURN(None)
    if args.plan_design:
        conflicts = [flag for flag, active in (('--resume', args.resume is not None), ('--evaluate-test', bool(args.evaluate_test)), ('--sanity-only', bool(args.sanity_only)), ('--statistical-baseline', bool(args.statistical_baseline)), ('--suite-baseline-tuning', bool(args.suite_baseline_tuning)), ('--final-test-comparison', args.final_test_comparison is not None)) if active]
        if conflicts:
            parser.error('--plan-design 只做设计预检，不能与以下训练/评估参数同时使用: ' + ', '.join(conflicts))
        if args.seeds is None:
            parser.error('--plan-design 需要 --seeds（或预注册文件中的 seeds）来确定计划单元与区组')
        if args.out is None:
            parser.error('--plan-design 需要 --out 指定计划目录')
        plan_root = Path(args.out)
        try:
            plan_root.mkdir(parents=True, exist_ok=True)
            plan, design, _ = build_sweep_execution(args, plan_root)
        except (OSError, ValueError) as error:
            parser.error(f'--plan-design 失败: {error}')
        print_design_summary(design)
        print('[design] 计划单元（执行顺序）:')
        for item in plan:
            print(f'  {int(item['order_index']):>3}. 区组 {item.get('block')} 内第 {item.get('position_in_block')} 个: {item.get('label')}（mode={item.get('mode')}, seed={item.get('seed')}）')
        print('[design] 注意：--plan-design 不加载数据，数据指纹/切分协议仍会在正式训练入口核对；确认计划后把该 design_plan.json 与预注册文件一起冻结')
        raise _SC_RETURN(None)
    if args.data is None:
        parser.error('--data 必填（除非使用 --compare/--plan-design）')
    if args.out is None:
        parser.error('--out 必填（除非使用 --compare）')
    if args.seeds is not None:
        run_experiment_sweep(args)
        raise _SC_RETURN(None)
    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        parser.error(f'--device 无效: {error}')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        parser.error(f'--device={args.device} 需要 CUDA，但当前环境不可用（torch.cuda.is_available()=False）；请改用 --device cpu 或在带 GPU 的环境运行')
    configure_cuda_precision(args.allow_tf32)
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
    elif device.type == 'cuda':
        amp_dtype = torch.float16
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None
    args.amp_dtype = str(amp_dtype) if amp_dtype is not None else None
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f'--out 目录无法创建或已存在同名文件: {error}')
    if test_evaluation_sealed(out_dir, getattr(args, 'test_ledger', None)):
        if args.resume is None:
            parser.error('输出目录已有测试集评估记录/封条：全新训练会截断重写 metrics.jsonl，使测试集被重复解封、退化为隐式验证集。请换用全新的 --out 目录。')
        parser.error('输出目录已有测试集评估记录/封条：测试集已解封结束，resume 继续训练会改写模型/选型，使已有 test 记录不再对应最终模型。如需继续实验，请换用全新的 --out 目录。')
    overwrite_error = fresh_out_dir_error(out_dir, resume_path=args.resume, allow_overwrite=args.allow_overwrite_out)
    if overwrite_error is not None:
        parser.error(overwrite_error)
    try:
        args.design_assignment = verify_design_assignment(args)
    except ValueError as error:
        parser.error(f'设计计划核对失败: {error}')
    if args.design_assignment:
        print(f'[design] 执行分配：计划 {str(args.design_assignment.get('plan_id'))[:12]}...，顺序 {args.design_assignment.get('order_index')}（区组 {args.design_assignment.get('block')} 内第 {args.design_assignment.get('position_in_block')} 个）')
    start_step = 0
    rng_payload = None
    saved_data_sha256 = None
    saved_data_val_fraction = None
    saved_data_test_fraction = None
    saved_data_split_sha256 = None
    saved_data_effective_train_sha256 = None
    saved_data_split_isolation = None
    saved_numerical_state = {}
    if args.resume is not None:
        try:
            rng_payload = load_checkpoint(args.resume, torch.device('cpu'))
        except Exception as error:
            parser.error(f'无法读取 --resume 检查点 {args.resume}: {error}')
        if not isinstance(rng_payload, dict) or not {'config', 'model', 'optimizer'} <= set(rng_payload):
            parser.error(f'--resume 检查点 {args.resume} 缺少 config/model/optimizer 必需字段：不是由本脚本 save_checkpoint 产生的存档')
        numerical_error = _checkpoint_numerical_error(rng_payload)
        if numerical_error is not None:
            parser.error(f'--resume 拒绝恢复 {args.resume}: {numerical_error}')
        saved_numerical_state = rng_payload.get('numerical_state') if isinstance(rng_payload.get('numerical_state'), dict) else {}
        if saved_numerical_state.get('numerical_polluted'):
            parser.error(f'--resume 检查点 {args.resume} 标记为数值已污染：不能把受污染运行当作新的训练起点；请使用污染前的检查点或换全新目录重跑')
        checkpoint_script_sha256 = rng_payload.get('script_sha256')
        checkpoint_git_commit = rng_payload.get('git_commit')
        current_script_sha256 = _script_sha256()
        current_git_commit = _git_output('rev-parse', 'HEAD')
        code_changed = bool(isinstance(checkpoint_script_sha256, str) and isinstance(current_script_sha256, str) and checkpoint_script_sha256 and current_script_sha256 and (checkpoint_script_sha256 != current_script_sha256))
        if code_changed:
            print(f'[repro] [warn] resume 检查点由另一份 experiment.py 保存（checkpoint script_sha256={checkpoint_script_sha256[:12]}..., current={current_script_sha256[:12]}...）：两次不同代码版本的参数被接在一起，训练轨迹不是任何单一代码版本的产物；如需可归因的结论，请用全新 --out 目录从头训练。')
        if isinstance(checkpoint_git_commit, str) and isinstance(current_git_commit, str) and checkpoint_git_commit and current_git_commit and (checkpoint_git_commit != current_git_commit) and (not code_changed):
            print(f'[repro] [note] resume checkpoint git commit {checkpoint_git_commit[:12]}... 与当前 {current_git_commit[:12]}... 不同（实验脚本内容一致）；请确认差异来自与本实验无关的文件。')
        args.resume_code_provenance = {'checkpoint_script_sha256': checkpoint_script_sha256, 'checkpoint_git_commit': checkpoint_git_commit, 'current_script_sha256': current_script_sha256, 'current_git_commit': current_git_commit, 'code_changed': code_changed}
        saved_args = rng_payload.get('args', {})
        if isinstance(saved_args, dict):
            saved_out = saved_args.get('out')
            if isinstance(saved_out, str) and saved_out:
                try:
                    same_run_dir = Path(saved_out).resolve() == out_dir.resolve()
                except OSError:
                    same_run_dir = False
                if not same_run_dir:
                    parser.error(f'--resume 检查点属于输出目录 {saved_out!r}，与当前 --out {str(out_dir)!r} 不一致：不能把别的运行目录的检查点续到本目录，否则会继承无关的原始运行身份/预注册/效率记录。请使用该检查点自己的 --out，或在全新目录从头训练')
        if _restore_resume_cleaning_justification(args, saved_args):
            print('resume restored domain_justification from checkpoint (训练集清洗依据是数据协议的一部分)')
        amp_change_error = resume_amp_change_error(saved_args, args.amp_dtype)
        if amp_change_error is not None:
            parser.error(amp_change_error)
        for schedule_field in ('lr_schedule', 'wsd_stable_steps', 'wsd_final_lr_ratio'):
            if schedule_field not in saved_args:
                continue
            if _argv_has_flag('--' + schedule_field.replace('_', '-')):
                continue
            saved_value = saved_args.get(schedule_field)
            current_value = getattr(args, schedule_field, None)
            if saved_value != current_value:
                print(f'resume restored {schedule_field} {current_value!r} -> {saved_value!r} (学习率轨迹是采样/优化轨迹的一部分)')
                setattr(args, schedule_field, saved_value)
        for evidence_field in ('split_policy', 'time_order_manifest'):
            if evidence_field not in saved_args:
                continue
            if _argv_has_flag('--' + evidence_field.replace('_', '-')):
                continue
            saved_value = saved_args.get(evidence_field)
            current_value = getattr(args, evidence_field, None)
            if saved_value != current_value:
                print(f'resume restored {evidence_field} {current_value!r} -> {saved_value!r} (切分声明/时间证据是数据协议的一部分)')
                setattr(args, evidence_field, saved_value)
        if 'document_masks' in saved_args:
            saved_masks = bool(saved_args.get('document_masks'))
            if saved_masks != bool(getattr(args, 'document_masks', False)):
                print(f'resume restored document_masks {getattr(args, 'document_masks', False)} -> {saved_masks} (注意力可见性是模型语义的一部分)')
                args.document_masks = saved_masks
        saved_data_sha256 = saved_args.get('data_sha256')
        saved_data_val_fraction = saved_args.get('data_val_fraction')
        saved_data_test_fraction = saved_args.get('data_test_fraction')
        saved_data_split_sha256 = saved_args.get('data_split_sha256')
        saved_data_effective_train_sha256 = saved_args.get('data_effective_train_sha256')
        saved_data_split_isolation = saved_args.get('data_split_isolation')
        saved_config = rng_payload['config']
        known_config_fields = {field.name for field in fields(Config)}
        unknown_config_fields = sorted(set(saved_config) - known_config_fields)
        if unknown_config_fields:
            print('resume ignored unknown config fields: ' + json.dumps(unknown_config_fields, ensure_ascii=False))
        if 'residual_scaling' not in saved_config:
            saved_config = dict(saved_config)
            saved_config['residual_scaling'] = 'global'
            print('[repro] [warn] 旧检查点缺少 residual_scaling：按历史 global 口径恢复（循环区域不额外除以 r）')
        if 'n_kv_heads' not in saved_config:
            saved_config = dict(saved_config)
            saved_config['n_kv_heads'] = saved_config.get('n_heads')
        cfg = Config(**{key: value for key, value in saved_config.items() if key in known_config_fields})
    else:
        dense_fnn_hidden = args.dense_fnn_hidden
        if dense_fnn_hidden is None and args.mode == 'dense':
            dense_fnn_hidden = args.top_k * args.expert_hidden
            print(f'dense 基线 FFN 隐藏宽度 = {dense_fnn_hidden}（等计算口径 top_k * expert_hidden；等参数口径可用 --dense-fnn-hidden 设为 n_experts * expert_hidden）')
        cfg = Config(seq_len=args.seq_len, d_model=args.d_model, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, n_layers=args.n_layers, n_experts=args.n_experts, expert_hidden=args.expert_hidden, top_k=args.top_k, k1=args.k1, k2=args.k2, mode=args.mode, dense_fnn_hidden=dense_fnn_hidden, balance_coef=args.balance_coef, residual_scale=args.residual_scale, residual_scaling=args.residual_scaling, init_scheme=args.init_scheme, document_masks=bool(getattr(args, 'document_masks', False)), document_separator=list(parse_group_separator(args.group_separator)) if getattr(args, 'document_masks', False) and args.group_separator is not None else None)

def _MAIN_part_003():
    global budget, budget_baseline, budget_report, cleaning_flags, config_fields, current_threshold, data_change_error, data_preregistration_error, dataset, effective, effective_train_change_error, enabled_cleaning, entity_audit, error, fidelity, field, group_separator_bytes, hint, ignored, isolation_change_error, key, measured, message, protocol_state, provided, restored_mode_error, restored_seed_error, saved_allow_tf32, saved_clean_test, saved_corrupt_byte_drop, saved_corrupt_drop, saved_dedup, saved_degenerate_drop, saved_equivalence_margin, saved_group_separator, saved_run_drop, saved_seed, saved_selection_metric, saved_shared_drop, saved_shingle, saved_strict, saved_threads, saved_threshold, saved_value, search, shift_flag, shift_value, split_change_error, time_evidence
    if rng_payload is not None:
        config_fields = {'seq_len', 'd_model', 'n_heads', 'n_kv_heads', 'n_layers', 'n_experts', 'expert_hidden', 'top_k', 'k1', 'k2', 'mode', 'dense_fnn_hidden', 'balance_coef', 'residual_scale', 'residual_scaling', 'init_scheme'}
        ignored = {}
        for field in config_fields:
            effective = getattr(cfg, field)
            provided = getattr(args, field)
            if provided != effective:
                ignored[field] = (provided, effective)
                setattr(args, field, effective)
        if ignored:
            print('resume ignored CLI config overrides: ' + json.dumps(ignored, ensure_ascii=False))
        if saved_data_val_fraction is not None and saved_data_val_fraction != args.val_fraction:
            print(f'resume restored val_fraction {args.val_fraction} -> {saved_data_val_fraction} (数据划分是采样轨迹的一部分)')
            args.val_fraction = saved_data_val_fraction
        if saved_data_test_fraction is not None and saved_data_test_fraction != args.test_fraction:
            print(f'resume restored test_fraction {args.test_fraction} -> {saved_data_test_fraction} (数据划分是采样轨迹的一部分)')
            args.test_fraction = saved_data_test_fraction
        if 'group_separator' in saved_args:
            saved_group_separator = saved_args.get('group_separator')
            if saved_group_separator != args.group_separator:
                print(f'resume restored group_separator {args.group_separator!r} -> {saved_group_separator!r} (群组切分是数据划分的一部分)')
                args.group_separator = saved_group_separator
        elif args.group_separator is not None:
            print('resume checkpoint predates --group-separator; 当前按群组切分，train/val/test 位置可能与原运行不同')
        if 'deduplicate_train_groups' in saved_args:
            saved_dedup = bool(saved_args.get('deduplicate_train_groups'))
            if saved_dedup != bool(args.deduplicate_train_groups):
                print(f'resume restored deduplicate_train_groups {bool(args.deduplicate_train_groups)} -> {saved_dedup} (训练集清洗是采样轨迹的一部分)')
                args.deduplicate_train_groups = saved_dedup
        if 'drop_train_shared_chunks' in saved_args:
            saved_shared_drop = bool(saved_args.get('drop_train_shared_chunks'))
            if saved_shared_drop != bool(args.drop_train_shared_chunks):
                print(f'resume restored drop_train_shared_chunks {bool(args.drop_train_shared_chunks)} -> {saved_shared_drop} (训练集清洗是采样轨迹的一部分)')
                args.drop_train_shared_chunks = saved_shared_drop
        for key in ('drop_train_near_duplicate_groups', 'near_duplicate_audit'):
            if key not in saved_args:
                continue
            saved_value = bool(saved_args.get(key))
            if saved_value != bool(getattr(args, key, False)):
                print(f'resume restored {key} {bool(getattr(args, key, False))} -> {saved_value} (训练集清洗/审计口径是采样轨迹的一部分)')
                setattr(args, key, saved_value)
        if 'near_duplicate_threshold' in saved_args:
            saved_threshold = saved_args.get('near_duplicate_threshold')
            try:
                saved_threshold = float(saved_threshold)
                current_threshold = float(args.near_duplicate_threshold)
            except (TypeError, ValueError, OverflowError):
                saved_threshold = None
                current_threshold = None
            if saved_threshold is not None and current_threshold is not None and math.isfinite(saved_threshold) and (saved_threshold != current_threshold):
                print(f'resume restored near_duplicate_threshold {args.near_duplicate_threshold} -> {saved_threshold} (训练集清洗是采样轨迹的一部分)')
                args.near_duplicate_threshold = saved_threshold
        if 'near_duplicate_shingle_bytes' in saved_args:
            saved_shingle = saved_args.get('near_duplicate_shingle_bytes')
            if isinstance(saved_shingle, int) and (not isinstance(saved_shingle, bool)) and (1 <= saved_shingle <= _NEAR_DUP_MAX_SHINGLE_BYTES) and (saved_shingle != int(args.near_duplicate_shingle_bytes)):
                print(f'resume restored near_duplicate_shingle_bytes {args.near_duplicate_shingle_bytes} -> {saved_shingle} (训练集清洗是采样轨迹的一部分)')
                args.near_duplicate_shingle_bytes = saved_shingle
        saved_clean_test = _saved_clean_against_test_effective(saved_args)
        if saved_clean_test and (not (args.deduplicate_train_groups or args.drop_train_shared_chunks or args.drop_train_near_duplicate_groups)):
            print('resume checkpoint recorded test-dependent train cleaning, but the cleaning switches restored from the checkpoint are off; ignoring the stale flag')
            saved_clean_test = False
        if saved_clean_test != bool(args.clean_train_against_test):
            print(f'resume restored clean_train_against_test {bool(args.clean_train_against_test)} -> {saved_clean_test} (训练集清洗是采样轨迹的一部分)')
            args.clean_train_against_test = saved_clean_test
            args.allow_eval_content_for_cleaning = saved_clean_test
        elif bool(getattr(args, 'allow_eval_content_for_cleaning', False)) != bool(args.clean_train_against_test):
            _normalize_clean_against_test(args)
        if 'drop_train_corrupt_groups' in saved_args:
            saved_corrupt_drop = bool(saved_args.get('drop_train_corrupt_groups'))
            if saved_corrupt_drop != bool(args.drop_train_corrupt_groups):
                print(f'resume restored drop_train_corrupt_groups {bool(args.drop_train_corrupt_groups)} -> {saved_corrupt_drop} (训练集清洗是采样轨迹的一部分)')
                args.drop_train_corrupt_groups = saved_corrupt_drop
        if 'drop_train_corrupt_bytes' in saved_args:
            saved_corrupt_byte_drop = bool(saved_args.get('drop_train_corrupt_bytes'))
            if saved_corrupt_byte_drop != bool(args.drop_train_corrupt_bytes):
                print(f'resume restored drop_train_corrupt_bytes {bool(args.drop_train_corrupt_bytes)} -> {saved_corrupt_byte_drop} (训练集清洗是采样轨迹的一部分)')
                args.drop_train_corrupt_bytes = saved_corrupt_byte_drop
        if 'drop_train_degenerate_groups' in saved_args:
            saved_degenerate_drop = bool(saved_args.get('drop_train_degenerate_groups'))
            if saved_degenerate_drop != bool(args.drop_train_degenerate_groups):
                print(f'resume restored drop_train_degenerate_groups {bool(args.drop_train_degenerate_groups)} -> {saved_degenerate_drop} (训练集清洗是采样轨迹的一部分)')
                args.drop_train_degenerate_groups = saved_degenerate_drop
        if 'drop_train_degenerate_runs' in saved_args:
            saved_run_drop = bool(saved_args.get('drop_train_degenerate_runs'))
            if saved_run_drop != bool(args.drop_train_degenerate_runs):
                print(f'resume restored drop_train_degenerate_runs {bool(args.drop_train_degenerate_runs)} -> {saved_run_drop} (训练集清洗是采样轨迹的一部分)')
                args.drop_train_degenerate_runs = saved_run_drop
        saved_seed = saved_args.get('seed')
        if isinstance(saved_seed, int) and (not isinstance(saved_seed, bool)) and (saved_seed != args.seed):
            print(f'resume restored seed {args.seed} -> {saved_seed} (随机流是采样轨迹的一部分)')
            args.seed = saved_seed
        for key in ('early_stop_patience', 'early_stop_min_delta'):
            saved_value = saved_args.get(key)
            if saved_value is not None and saved_value != getattr(args, key):
                print(f'resume restored {key} {getattr(args, key)} -> {saved_value} (早停判定必须与存档口径一致)')
                setattr(args, key, saved_value)
        saved_selection_metric = saved_args.get('selection_metric')
        if saved_selection_metric is not None and saved_selection_metric != args.selection_metric:
            print(f'resume restored selection_metric {args.selection_metric} -> {saved_selection_metric} (选型/早停必须与存档口径一致)')
            args.selection_metric = saved_selection_metric
        saved_allow_tf32 = saved_args.get('allow_tf32')
        if isinstance(saved_allow_tf32, bool) and saved_allow_tf32 != args.allow_tf32:
            print(f'resume restored allow_tf32 {args.allow_tf32} -> {saved_allow_tf32} (数值精度是训练轨迹的一部分)')
            args.allow_tf32 = saved_allow_tf32
        saved_threads = saved_args.get('torch_threads')
        if saved_threads is not None and saved_threads != args.torch_threads:
            try:
                configure_cpu_threads(saved_threads)
            except (RuntimeError, ValueError) as error:
                parser.error(f'resume 无法恢复检查点中的 CPU 线程数 {saved_threads!r}: {error}')
            print(f'resume restored torch_threads {args.torch_threads} -> {saved_threads} (CPU 归约顺序是训练轨迹的一部分)')
            args.torch_threads = saved_threads
        elif saved_threads is None and args.torch_threads is not None:
            print(f'[repro] [warn] 检查点未固定 CPU 线程数，本次续跑显式固定为 {args.torch_threads}：原运行的归约顺序可能不同，严格复现请用全新目录从头训练。')
        for key in ('grad_checkpoint', 'compile'):
            saved_value = saved_args.get(key)
            if isinstance(saved_value, bool) and saved_value != getattr(args, key):
                print(f'resume restored {key} {getattr(args, key)} -> {saved_value} (实现路径是训练轨迹的一部分)')
                setattr(args, key, saved_value)
        saved_strict = saved_args.get('strict_determinism')
        if isinstance(saved_strict, bool) and saved_strict != args.strict_determinism:
            print(f'resume restored strict_determinism {args.strict_determinism} -> {saved_strict} (确定性检查口径必须与存档一致)')
            args.strict_determinism = saved_strict
        saved_equivalence_margin = _finite_float(saved_args.get('equivalence_margin'))
        if saved_equivalence_margin is not None and saved_equivalence_margin > 0.0 and (saved_equivalence_margin != args.equivalence_margin):
            print(f'resume restored equivalence_margin {args.equivalence_margin} -> {saved_equivalence_margin} (等价界是训练前预注册的一部分)')
            args.equivalence_margin = saved_equivalence_margin
        configure_cuda_precision(args.allow_tf32)
    restored_seed_error = _preregistration_restored_seed_error(args)
    if restored_seed_error is not None:
        parser.error(restored_seed_error)
    restored_mode_error = _preregistration_restored_mode_error(args)
    if restored_mode_error is not None:
        parser.error(restored_mode_error)
    if args.selection_metric is None:
        if args.primary_metric in _SELECTION_METRIC_KEYS:
            args.selection_metric = args.primary_metric
        else:
            args.selection_metric = 'val_loss'
    if args.primary_metric is not None and args.selection_metric != args.primary_metric:
        print(f'[selection] [warn] 预注册主指标为 {args.primary_metric}，但 checkpoint 选型/早停使用 {args.selection_metric}：最终模型不是按假设的判定口径选出的。若该指标无法在周期验证计算，考虑改预注册指标或显式声明 --selection-metric。')
    elif args.selection_metric != 'val_loss':
        print(f'[selection] 选型/早停指标为 {args.selection_metric}（不是 val_loss）')
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(f'模型结构/损失配置无效: {error}')
    group_separator_bytes = None
    if args.group_separator is not None:
        try:
            group_separator_bytes = parse_group_separator(args.group_separator)
        except ValueError as error:
            parser.error(f'--group-separator 无效: {error}')
    for shift_flag, shift_value in (('--max-byte-shift', args.max_byte_shift), ('--max-bigram-shift', args.max_bigram_shift), ('--max-conditional-shift', args.max_conditional_shift)):
        if shift_value is None:
            continue
        if not math.isfinite(shift_value) or shift_value < 0.0 or shift_value > 1.0:
            parser.error(f'{shift_flag} 必须是 [0, 1] 内的有限数')
    if args.deduplicate_train_groups and group_separator_bytes is None:
        parser.error('--deduplicate-train-groups 需要 --group-separator：没有显式群组边界时无法只清洗 train 而不改动 val/test')
    if args.drop_train_shared_chunks and group_separator_bytes is None:
        parser.error('--drop-train-shared-chunks 需要 --group-separator：没有显式群组边界时无法整组清洗跨 split 共享内容而不改动 val/test')
    if args.drop_train_near_duplicate_groups and group_separator_bytes is None:
        parser.error('--drop-train-near-duplicate-groups 需要 --group-separator：近重复按群组定义，没有显式群组边界时无法只清洗 train 而不改动 val/test')
    if args.near_duplicate_audit and group_separator_bytes is None:
        parser.error('--near-duplicate-audit 需要 --group-separator：近重复按群组定义，没有群组边界就无法区分文档/记录')
    if not math.isfinite(args.near_duplicate_threshold) or not 0.0 < args.near_duplicate_threshold <= 1.0:
        parser.error('--near-duplicate-threshold 必须是 (0, 1] 内的有限数')
    if not (isinstance(args.near_duplicate_shingle_bytes, int) and (not isinstance(args.near_duplicate_shingle_bytes, bool)) and (1 <= args.near_duplicate_shingle_bytes <= _NEAR_DUP_MAX_SHINGLE_BYTES)):
        parser.error(f'--near-duplicate-shingle-bytes 必须是 1..{_NEAR_DUP_MAX_SHINGLE_BYTES} 的整数')
    if args.clean_train_against_test and (not (args.deduplicate_train_groups or args.drop_train_shared_chunks or args.drop_train_near_duplicate_groups)):
        parser.error('--clean-train-against-test 需要与 --deduplicate-train-groups、--drop-train-shared-chunks 或 --drop-train-near-duplicate-groups 一起使用：没有训练集清洗时 test 不会被读取')
    if args.drop_train_corrupt_groups and group_separator_bytes is None:
        parser.error('--drop-train-corrupt-groups 需要 --group-separator：没有显式群组边界时无法只清洗 train 而不改动 val/test')
    if args.drop_train_corrupt_bytes and group_separator_bytes is not None:
        parser.error('--drop-train-corrupt-bytes 仅用于无 --group-separator 的连续字节流；有群组边界时请用 --drop-train-corrupt-groups 以保持群组完整')
    if args.drop_train_degenerate_groups and group_separator_bytes is None:
        parser.error('--drop-train-degenerate-groups 需要 --group-separator：没有显式群组边界时无法只清洗 train 而不改动 val/test')
    if args.drop_train_degenerate_runs and group_separator_bytes is not None:
        parser.error('--drop-train-degenerate-runs 仅用于无 --group-separator 的连续字节流；有群组边界时请用 --drop-train-degenerate-groups 以保持群组完整')
    cleaning_flags = (('--drop-train-corrupt-groups', args.drop_train_corrupt_groups), ('--drop-train-corrupt-bytes', args.drop_train_corrupt_bytes), ('--drop-train-degenerate-groups', args.drop_train_degenerate_groups), ('--drop-train-degenerate-runs', args.drop_train_degenerate_runs))
    enabled_cleaning = [name for name, enabled in cleaning_flags if enabled]
    if enabled_cleaning and (not (args.domain_justification and str(args.domain_justification).strip())):
        parser.error('启用训练集清洗（' + ', '.join(enabled_cleaning) + '）需要 --domain-justification：非法 UTF-8、NUL、重复字节或退化段可能是合法字节/结构（二进制记录、表格边框、填充），也可能是任务允许的标签；没有领域依据的删除会选择性改变监督分布。请先确认这是格式损坏并记录依据，或只用审计/严格门而不删除')
    args.cleaning_domain_justification = str(args.domain_justification).strip() if args.domain_justification else None
    args.time_order_evidence = None
    if args.split_policy == 'chronological':
        if not args.time_order_manifest:
            parser.error('--split-policy chronological 需要 --time-order-manifest：仅凭文件顺序不能证明时间顺序')
        try:
            time_evidence = read_time_order_manifest(args.time_order_manifest)
        except (OSError, ValueError) as error:
            parser.error(f'--time-order-manifest 无效: {error}')
        if not time_evidence['monotonic_non_decreasing']:
            parser.error('--time-order-manifest 的时间戳不是单调不减：语料文件顺序与时间顺序不一致，不能按时间前缀切分（违规行: ' + ', '.join((str(item) for item in time_evidence['order_violations'])) + '）')
        args.time_order_evidence = time_evidence
    elif args.time_order_manifest is not None:
        print('[data] [note] 已提供 --time-order-manifest，但 --split-policy=contiguous：清单只作为证据记录，本次结论仍不声称时间切分')
        try:
            args.time_order_evidence = read_time_order_manifest(args.time_order_manifest)
        except (OSError, ValueError) as error:
            parser.error(f'--time-order-manifest 无效: {error}')
    args.cost_profile = mode_cost_profile(cfg)
    protocol_state = {'lr_schedule': str(getattr(args, 'lr_schedule', 'cosine')), 'batch_size': int(args.batch_size), 'steps': int(args.steps), 'stable_steps': int(args.steps if getattr(args, 'wsd_stable_steps', None) is None else args.wsd_stable_steps), 'document_masks': bool(cfg.document_masks), 'save_at_steps': list(getattr(args, 'save_at_steps', None) or [])}
    args.paper_protocol_state = protocol_state
    args.paper_fidelity = paper_fidelity_report(cfg, protocol_state)
    fidelity = args.paper_fidelity
    if fidelity.get('is_full_reproduction'):
        print(f'[paper] 架构与 3.1 节协议均对齐论文披露口径（单元 {fidelity['cell']}）')
    elif not fidelity.get('architecture_faithful'):
        print('[paper][warn] 当前配置不是论文附录 A 的忠实复现：' + ', '.join(fidelity['deviations']))
    elif not fidelity.get('protocol_faithful'):
        print('[paper][warn] 架构对齐论文，但训练协议仍有偏离：' + ', '.join((name for name in fidelity['deviations'] if name.startswith('protocol_'))))
    budget = fidelity.get('budget_audit')
    if isinstance(budget, dict):
        measured = budget['measured']
        print(f'[paper] 预算审计：FLOPs 比 {measured['flops_ratio_vs_base']:.3f}（论文 {budget['paper']['flops_ratio']}），KV 比 {measured['kv_ratio_vs_base']:.3f}（论文 {budget['paper']['kv_ratio']}），S={measured['compute_equivalent_sparsity'] * 100:.1f}%（论文 {budget['paper']['compute_equivalent_sparsity'] * 100:.1f}%）')
    budget_baseline = args.budget_match if args.budget_match is not None else 'base' if cfg.mode != 'base' else 'smelt'
    if budget_baseline not in Config.MODEL_MODES:
        parser.error('--budget-match 必须是模型模式之一: ' + ', '.join(sorted(Config.MODEL_MODES)))
    if not math.isfinite(args.budget_tolerance) or args.budget_tolerance <= 0.0:
        parser.error('--budget-tolerance 必须是有限正数')
    budget_report = budget_match_report(cfg, budget_baseline, args.budget_tolerance)
    args.budget_match_report = budget_report
    print(f'[budget] 解析口径：每 token FLOPs {budget_report['candidate_flops_per_token']:,} vs {budget_report['baseline_flops_per_token']:,} ({budget_baseline})，比例 {budget_report['flops_ratio']:.3f}')
    if args.budget_match is not None:
        search = budget_report.get('search')
        if not budget_report['matched']:
            hint = ''
            if isinstance(search, dict):
                if search.get('kind') == 'k2':
                    hint = f'；按完整计算图最接近预算的 k2={search['recommended_k2']}（比例 {search['flops_ratio_at_recommendation']:.3f}）'
                elif search.get('kind') == 'k1_k2':
                    hint = f'；按完整计算图最接近预算的 k1={search['recommended_k1']}, k2={search['recommended_k2']}（比例 {search['flops_ratio_at_recommendation']:.3f}）'
            parser.error(f'--budget-match 未通过：候选模式与 {budget_baseline} 的解析 FLOPs 比例为 {budget_report['flops_ratio']:.3f}，超出 ±{args.budget_tolerance:.0%} 的允许误差' + hint + '。仅令 k1*k2≈2*top_k 会漏掉第一阶段成本，不是充分的预算匹配；请调整 k1/k2/宽度后重跑')
    try:
        dataset = ByteDataset(args.data, cfg.seq_len, val_fraction=args.val_fraction, test_fraction=args.test_fraction, device=device, group_separator=group_separator_bytes, deduplicate_train_groups=args.deduplicate_train_groups, drop_train_shared_chunks=args.drop_train_shared_chunks, drop_train_near_duplicate_groups=args.drop_train_near_duplicate_groups, near_duplicate_audit=args.near_duplicate_audit, near_duplicate_threshold=args.near_duplicate_threshold, near_duplicate_shingle_bytes=args.near_duplicate_shingle_bytes, drop_train_corrupt_groups=args.drop_train_corrupt_groups, drop_train_corrupt_bytes=args.drop_train_corrupt_bytes, drop_train_degenerate_groups=args.drop_train_degenerate_groups, drop_train_degenerate_runs=args.drop_train_degenerate_runs, clean_train_against_test=bool(getattr(args, 'clean_train_against_test', False) or getattr(args, 'allow_eval_content_for_cleaning', False)), include_test_in_cleaning=bool(getattr(args, 'allow_eval_content_for_cleaning', False)))
    except (OSError, ValueError) as error:
        parser.error(f'数据管线初始化失败: {error}')
    if args.evaluate_test and (not dataset.has_test):
        parser.error('--evaluate-test 需要有效的测试集：请设置正的 --test-fraction 且语料足够长')
    args.data_sha256 = dataset.data_sha256
    args.data_bytes = dataset.num_bytes
    args.data_val_fraction = dataset.val_fraction
    args.data_test_fraction = dataset.test_fraction
    args.data_split_sha256 = dict(dataset.split_sha256)
    args.data_effective_train_sha256 = dataset.effective_train_sha256
    args.data_split_isolation = dict(dataset.split_isolation)
    if args.preregistration is not None:
        data_preregistration_error = preregistration_data_error(args, preregistration_payload, dataset)
        if data_preregistration_error is not None:
            parser.error(data_preregistration_error)
        if isinstance(getattr(args, 'preregistration_record', None), dict):
            args.preregistration_record.update({'data_sha256': dataset.data_sha256, 'data_split_sha256': dict(dataset.split_sha256), 'data_split_policy': args.split_policy, 'data_val_fraction': dataset.val_fraction, 'data_test_fraction': dataset.test_fraction, 'data_group_separator': args.group_separator, 'run_mode': getattr(args, 'mode', None), 'run_seed': getattr(args, 'seed', None), 'run_seeds': getattr(args, 'seeds', None), 'data_verified': True})
        print(f'[prereg] 数据协议核对通过：语料 {dataset.data_sha256[:12]}...，切分 {args.split_policy}，val/test={dataset.val_fraction:g}/{dataset.test_fraction:g}')
    args.data_entity_audit = None
    if args.entity_manifest is not None:
        try:
            entity_audit = audit_entity_manifest(bytes(dataset._raw), group_separator_bytes, dataset.split_bounds, args.entity_manifest)
        except (OSError, ValueError) as error:
            parser.error(f'--entity-manifest 无效: {error}')
        if not entity_audit.get('available'):
            parser.error('--entity-manifest 需要 --group-separator：没有群组边界就无法把实体清单映射到 split，实体隔离审计无法执行' + (f'（{entity_audit.get('reason')}）' if entity_audit.get('reason') else ''))
        args.data_entity_audit = entity_audit
        if entity_audit.get('cross_split_entities'):
            message = f'实体清单显示 {entity_audit['cross_split_entities']} 个实体同时出现在多个 split：分隔符只保证连续片段完整，不保证实体隔离；同一用户/作者/文档版本的非相邻片段仍会泄露'
            if args.allow_entity_overlap:
                print('[data] [warn] ' + message)
            else:
                parser.error(message + '。请重新切分/剔除跨界实体，或显式加 --allow-entity-overlap（会写入配置）')
        else:
            print(f'[data] 实体审计通过：{entity_audit.get('entities')} 个实体均未跨 split')
    data_change_error = resume_data_change_error(saved_data_sha256, dataset.data_sha256)
    if data_change_error is not None:
        parser.error(data_change_error)
    split_change_error = resume_split_change_error(saved_data_split_sha256, dataset.split_sha256)
    if split_change_error is not None:
        parser.error(split_change_error)
    elif saved_data_sha256 is not None and saved_data_split_sha256 is None:
        print('resume checkpoint predates per-split content fingerprints; 仅凭整体 sha256 与比例无法确认 train/val/test 的实际字节段未变')
    effective_train_change_error = resume_effective_train_change_error(saved_data_effective_train_sha256, dataset.effective_train_sha256)
    if effective_train_change_error is not None:
        parser.error(effective_train_change_error)
    elif saved_data_sha256 is not None and saved_data_effective_train_sha256 is None:
        print('resume checkpoint predates the effective train fingerprint; 仅凭 split 指纹与清洗开关无法确认清洗剔除区间未变')
    isolation_change_error = resume_split_isolation_change_error(saved_data_split_isolation, dataset.split_isolation)
    if isolation_change_error is not None:
        parser.error(isolation_change_error)
    elif saved_data_sha256 is not None and saved_data_split_isolation is None:
        print('resume checkpoint predates the temporal split isolation record; 仅凭 split 内容指纹无法确认边界顺序与绝对位置未变')

def _MAIN_part_004():
    global advisory, base_lr, base_lrs, baseline_tuning_record, best_selection_step, best_selection_value, config_sanity_result, cuda_rng_state, curve_baseline_loss, curve_records, data_audit, data_protocol_check, decay_params, early_stopped, effective_weight_decay, embedding_params, entry_ambiguous, entry_mentions, entry_unverifiable, error, evals_without_improvement, failure, fused_adamw, group, last_curve_status, last_curve_warning_flags, lr_lambda, metrics_path, mismatch, model, n_params, no_decay_params, optimizer, param, param_groups, prior_analysis, prior_flags, prior_status, probe, provenance_error, python_rng_state, recent_grad_norms, reference_baselines, reference_window, repeat_index, repeated_result, reproducibility, rng_payload, run_metadata, run_sanity, sanity_record, sanity_reports, sanity_result, sanity_run_metadata, sanity_tracking, saved_base_lr, saved_batch, saved_eval_batches, saved_eval_bootstrap, saved_eval_bootstrap_block, saved_eval_interval, saved_grad_clip, saved_periodic_auc, saved_scheduler, saved_strict, saved_threads, saved_warmup, scale_guard, scheduler, seed_overlap, selected_model_path, start_step, torch_rng_state, tracking_provenance, train_generator, train_generator_state, tuning_path, tuning_payload, unigram_macro_nll
    if saved_data_sha256 is not None and (saved_data_val_fraction is None or saved_data_test_fraction is None):
        print('resume checkpoint predates the train/val/test split; 数据划分位置可能与原运行不同，采样轨迹不再一致')
    elif saved_data_sha256 is not None and (saved_data_val_fraction != dataset.val_fraction or saved_data_test_fraction != dataset.test_fraction):
        print(f'resume data split fractions changed (val {saved_data_val_fraction} -> {dataset.val_fraction}, test {saved_data_test_fraction} -> {dataset.test_fraction}); 采样轨迹与原运行不再一致')
    data_protocol_check = probe_data_pipeline(dataset, device)
    if data_protocol_check['ok']:
        print('[data-protocol] 数据管线检查通过：训练批采样与固定评估窗口的 (x, y) 均与原始字节切片逐位一致')
    else:
        print('[data-protocol] FAILED: 数据管线存在实质错误，在问题排除前拒绝开始训练：')
        for failure in data_protocol_check['failures']:
            print(f'  - {failure}')
        raise SystemExit('数据管线检查未通过：请先检查输入/标签错位、切分越界或不可复现的采样逻辑，再开始全量训练。')
    baseline_tuning_record = None
    if args.baseline_tuning_from is not None:
        tuning_path = Path(args.baseline_tuning_from)
        try:
            tuning_payload = load_baseline_tuning(tuning_path)
        except ValueError as error:
            parser.error(f'--baseline-tuning-from 无效: {error}')
        provenance_error = baseline_tuning_provenance_error(tuning_payload, mode=cfg.mode, lr=args.lr, steps=args.steps, data_sha256=dataset.data_sha256, batch_size=args.batch_size, protocol=build_comparison_protocol(cfg, args))
        if provenance_error is not None:
            parser.error('--baseline-tuning-from 与本次训练不一致: ' + provenance_error)
        baseline_tuning_record = baseline_tuning_provenance_record(tuning_payload, str(tuning_path))
        print(f'[baseline-tuning] 基线调参来源已核对: mode={baseline_tuning_record['mode']}, winner lr={format(float(baseline_tuning_record['winner_lr']), 'g')}, metric={baseline_tuning_record['metric']}')
        seed_overlap = tuning_seed_overlap_report([{'run': str(out_dir), 'mode': cfg.mode, 'seed': args.seed, 'baseline_tuning': baseline_tuning_record}])
        for advisory in seed_overlap['advisories']:
            print('[baseline-tuning] [warn] ' + str(advisory))
    data_audit = audit_data_splits(dataset, include_test=False, cache_scans=args.evaluate_test)
    print_data_audit(data_audit)
    if not args.allow_data_leakage and (not enforce_data_leakage_gate(data_audit, 'train/val')):
        raise SystemExit('跨 split 泄露检查未通过：请先清洗数据（用 --group-separator/--deduplicate-train-groups 去掉重复内容），或确认不是泄露后显式传 --allow-data-leakage。')
    if args.allow_data_leakage and data_audit.get('leakage_violations'):
        print('[data] [warn] 已通过 --allow-data-leakage 显式豁免跨 split 泄露拦截；本次结果不能作为无泄露证据。')
    if not args.allow_eval_duplicates and (not enforce_evaluation_duplicate_gate(data_audit, 'train/val')):
        raise SystemExit('评估集重复样本检查未通过：请先去重后重新切分，或确认重复属于目标分布后显式传 --allow-eval-duplicates。')
    if args.allow_eval_duplicates and data_audit.get('evaluation_violations'):
        print('[data] [warn] 已通过 --allow-eval-duplicates 显式豁免评估集重复样本拦截；评估区间可能偏窄。')
    if args.strict_data_audit and (not enforce_strict_data_audit(data_audit, 'train/val')):
        raise SystemExit('严格数据审计未通过：请先清洗数据（去重、修编码、修正群组划分），或去掉 --strict-data-audit 仅保留警告。')
    data_audit['distribution_shift_gate'] = {'max_byte_js': args.max_byte_shift, 'max_bigram_js': args.max_bigram_shift, 'max_conditional_js': args.max_conditional_shift}
    if not enforce_distribution_shift_gate(data_audit, max_byte_js=args.max_byte_shift, max_bigram_js=args.max_bigram_shift, max_conditional_js=args.max_conditional_shift, context='train/val'):
        raise SystemExit('分布偏移检查未通过：请先检查切分/时间顺序与评估口径，或显式放宽/去掉 --max-byte-shift / --max-bigram-shift / --max-conditional-shift。')
    if args.statistical_baseline:
        run_statistical_baseline(args, dataset, out_dir, device, data_audit, data_protocol_check)
        raise _SC_RETURN(None)
    reference_baselines = compute_reference_baselines(dataset, args.batch_size, args.eval_batches, torch.device('cpu'), split='val')
    reference_window = (args.batch_size, args.eval_batches)
    print(f'[reference] uniform val_loss={reference_baselines['uniform_loss']:.4f} (= ln 256), unigram val_loss={reference_baselines['unigram_loss']:.4f}, unigram token_accuracy={reference_baselines['unigram_token_accuracy']:.4f}（类别不平衡下解读 accuracy 的参考下界）')
    unigram_macro_nll = reference_baselines.get('unigram_macro_nll')
    if isinstance(unigram_macro_nll, (int, float)):
        print(f'[reference] 类平衡似然参考：uniform macro_nll={reference_baselines['uniform_macro_nll']:.4f} (= ln 256), unigram macro_nll={float(unigram_macro_nll):.4f}（按出现类等权的 NLL；频率加权的 val_loss 几乎只反映高频字节，稀有字节的表现看这一项）')
    if args.hypothesis.strip():
        print(f'[hypothesis] {args.hypothesis.strip()}' + (f' (rq={args.rq.strip()})' if args.rq.strip() else ''))
        entry_mentions = hypothesis_metric_mentions(args.hypothesis)
        entry_unverifiable = hypothesis_metric_unverifiable(entry_mentions, routing_available=False if args.mode == 'dense' or args.statistical_baseline or args.skip_routing_analysis else None)
        if not entry_mentions:
            print('[hypothesis] [warn] 假设未写明判定指标（如验证损失 / macro F1 / PR-AUC / 召回率）：无法与评估口径对齐，事后 --compare 也无法判定成败；建议写出「依据什么指标、预期改善多少」')
        elif entry_unverifiable:
            print('[hypothesis] [warn] 假设提到的判定指标当前不可判定: ' + ', '.join(entry_unverifiable) + '（路由诊断类指标需要该运行有 routing_analysis；dense 基线或 --skip-routing-analysis 时无法核对）')
        else:
            entry_ambiguous = hypothesis_ambiguous_averaging_metrics(args.hypothesis, entry_mentions)
            if entry_ambiguous:
                print('[hypothesis] [warn] 假设提到的 ' + ', '.join(entry_ambiguous) + ' 未注明平均口径（macro / micro / weighted 三者结果可能差异很大）：请写明判定口径，并与 --primary-metric / --compare-metric 保持一致')
        if entry_mentions and (not args.primary_metric):
            print(f'[hypothesis] [warn] 未用 --primary-metric 预注册主判定指标：事后 --compare 可以任选 {len(COMPARISON_METRIC_KEYS)} 个指标中的一个，指标选择偏差无法从记录上排除；建议训练前声明主指标（可选加 --hypothesis-direction / --hypothesis-min-effect）')
    run_sanity = args.sanity_only or (args.resume is None and (not args.skip_sanity_check))
    sanity_result = None
    sanity_record = None
    if run_sanity:
        print(f'[sanity] 过拟合小批量检查: batch_size={args.sanity_batch_size}, steps={args.sanity_steps}, lr={args.sanity_lr}, repeats={args.sanity_repeats}, grad_checkpoint={args.grad_checkpoint}, loss<={args.sanity_loss_threshold} 或 acc>={args.sanity_acc_threshold}, 且 loss 相对初始值下降>={args.sanity_min_loss_reduction:.1%}')
        sanity_reports = []
        for repeat_index in range(args.sanity_repeats):
            if repeat_index > 0:
                print(f'[sanity] 复现性重复 {repeat_index + 1}/{args.sanity_repeats}（同模型、同批次、同种子）')
            repeated_result = run_sanity_overfit(cfg, dataset, device, amp_dtype=amp_dtype, batch_size=args.sanity_batch_size, steps=args.sanity_steps, lr=args.sanity_lr, loss_threshold=args.sanity_loss_threshold, acc_threshold=args.sanity_acc_threshold, min_relative_improvement=args.sanity_min_loss_reduction, seed=args.seed, grad_checkpoint=args.grad_checkpoint, strict_determinism=args.strict_determinism)
            sanity_reports.append(repeated_result)
            if not repeated_result['passed']:
                break
        sanity_result = sanity_reports[0]
        reproducibility = sanity_repeat_report(sanity_reports)
        sanity_result['reproducibility'] = reproducibility
        if reproducibility['identical'] is False:
            sanity_result['passed'] = False
            print('[sanity] FAILED: 同一配置 / 同一种子的重复运行结果不一致（同一台机器上不可复现）。差异项：')
            for mismatch in reproducibility['mismatches']:
                print(f'  - repeat {mismatch['repeat']} {mismatch['key']}: {mismatch['first']!r} != {mismatch['repeat_value']!r}')
            print('这通常意味着仍有未固定的随机源、非确定性算子（可加 --strict-determinism 定位）、异步执行顺序或 CUDA 精度设置差异；在解释清楚之前不要开始全量训练。')
        elif reproducibility['identical'] is True:
            print(f'[sanity] 复现性检查通过：{reproducibility['repeats']} 次同种子运行的初始/最终损失、准确率与梯度范数逐位一致')
        sanity_record = {'step': 'sanity', **sanity_result}
        if sanity_result['passed']:
            print(f'[sanity] PASSED: final total_loss={sanity_result['final_total_loss']:.4f}, final lm_loss={sanity_result['final_lm_loss']:.4f}, token_acc={sanity_result['final_token_accuracy']:.4f}')
        elif reproducibility['identical'] is not False:
            if not sanity_result.get('data_sampling_uses_explicit_generator', True):
                print('[sanity] FAILED: 数据采样消耗了 torch 全局 RNG（未完全使用传入的 torch.Generator）。同一份数据会随模型初始化等其他全局随机消耗的顺序漂移，同种子难以稳定复现；请先修正数据管线的随机源再开始全量训练。')
            elif not sanity_result.get('optimizer_coverage_ok', True):
                print('[sanity] FAILED: 优化器未覆盖全部可训练参数（或参数被重复加入多个 param group），这些参数会一直停在初始值；请先修正 optimizer 的参数列表再开始全量训练。')
            elif not sanity_result.get('parameter_update_ok', True):
                print('[sanity] FAILED: 第一步 optimizer.step 未改变任何非专家参数；请检查学习率是否为 0、参数是否被冻结或 GradScaler 是否跳过了更新。')
            elif not sanity_result.get('gradient_flow_ok', True):
                print('[sanity] FAILED: 梯度检查未通过（缺失/全零/非有限梯度或损失未连入计算图），具体失败项见上方 [sanity] 梯度检查与排查建议；请先修复反向传播路径再开始全量训练。')
            else:
                print(f'[sanity] FAILED: 模型未能在小批量上过拟合 (initial total_loss={sanity_result['initial_total_loss']}, final total_loss={sanity_result['final_total_loss']:.4f}, token_acc={sanity_result['final_token_accuracy']:.4f}, relative_reduction={sanity_result['loss_reduction']})。这通常意味着梯度没有反向传播、损失计算错误、优化器或学习率配置有严重问题；请先排查再开始全量训练。确需跳过可用 --skip-sanity-check。')
    if args.sanity_only or (sanity_result is not None and (not sanity_result['passed'])):
        sanity_run_metadata = collect_run_metadata(device)
        write_run_config(out_dir, cfg, args, n_params=int(sanity_result['n_params']), run_metadata=sanity_run_metadata, sanity_result=sanity_result, data_audit=data_audit, reference_baselines=reference_baselines, baseline_tuning=baseline_tuning_record, data_protocol_check=data_protocol_check, dataset=dataset, init_fingerprints=None)
        if args.resume is None:
            sanity_tracking = load_tracking_provenance(out_dir)
            write_jsonl_record(out_dir / 'metrics.jsonl', {'step': 'reference_baselines', **tracking_header_fields(args, dataset, sanity_run_metadata, run_started_utc=sanity_tracking.get('run_started_utc'), comparison_protocol_sha256=sanity_tracking.get('comparison_protocol_sha256')), **reference_baselines}, append=False)
        write_jsonl_record(out_dir / 'metrics.jsonl', sanity_record, append=True)
        print(json.dumps({'sanity_check': sanity_result}, ensure_ascii=False))
        if not sanity_result['passed']:
            raise SystemExit(1)
        raise _SC_RETURN(None)
    set_seed(args.seed, strict_determinism=args.strict_determinism)
    model = LanguageModel(cfg).to(device)
    model.grad_checkpoint = args.grad_checkpoint
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    embedding_params = (model.token_embedding.weight, model.position_embedding.weight)
    decay_params = [p for p in decay_params if all((p is not emb for emb in embedding_params))]
    no_decay_params.extend(embedding_params)
    param_groups = [{'params': decay_params, 'weight_decay': args.weight_decay}, {'params': no_decay_params, 'weight_decay': 0.0}]
    probe = torch.zeros(1, device=device, requires_grad=True)
    probe.grad = torch.zeros_like(probe)
    try:
        torch.optim.AdamW([probe], lr=args.lr, fused=True).step()
        fused_adamw = True
    except (RuntimeError, TypeError):
        fused_adamw = False
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, fused=fused_adamw, foreach=not fused_adamw)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if rng_payload is not None:
        model.load_state_dict(rng_payload['model'])
        rng_payload.pop('model', None)
        optimizer.load_state_dict(rng_payload['optimizer'])
        rng_payload.pop('optimizer', None)
        start_step = int(rng_payload.get('step', 0))
        if 'step' not in rng_payload:
            print('legacy checkpoint has no step; resuming from step 0')
        if scaler is not None and rng_payload.get('scaler') is not None:
            scaler.load_state_dict(rng_payload['scaler'])
            rng_payload.pop('scaler', None)
        train_generator_state = rng_payload.get('train_generator_state')
        if train_generator_state is not None:
            train_generator_state = train_generator_state.cpu()
        torch_rng_state = rng_payload.get('torch_rng_state')
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        python_rng_state = rng_payload.get('python_rng_state')
        if python_rng_state is not None:
            random.setstate(python_rng_state)
        cuda_rng_state = rng_payload.get('cuda_rng_state')
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        _restore_numpy_rng_state(rng_payload.get('numpy_rng_state'))
    else:
        train_generator_state = None
    if rng_payload is not None:
        saved_scheduler = rng_payload.get('scheduler')
        if saved_scheduler is not None:
            base_lrs = saved_scheduler['base_lrs']
        else:
            saved_base_lr = rng_payload.get('args', {}).get('lr', args.lr)
            base_lrs = [saved_base_lr for _ in optimizer.param_groups]
            print(f'legacy checkpoint has no scheduler state; restored base lr {saved_base_lr}')
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr
        if args.lr != base_lrs[0]:
            print(f'resume restored lr {args.lr} -> {base_lrs[0]} (基础 lr 由检查点调度状态决定)')
            args.lr = base_lrs[0]
    if rng_payload is not None:
        saved_batch = rng_payload.get('args', {}).get('batch_size')
        if saved_batch is not None and saved_batch != args.batch_size:
            print(f'resume restored batch_size {args.batch_size} -> {saved_batch} (batch_size 决定采样轨迹)')
            args.batch_size = saved_batch
        effective_weight_decay = optimizer.param_groups[0]['weight_decay']
        if args.weight_decay != effective_weight_decay:
            print(f'resume restored weight_decay {args.weight_decay} -> {effective_weight_decay} (weight_decay 由优化器状态决定)')
            args.weight_decay = effective_weight_decay
        saved_warmup = rng_payload.get('args', {}).get('warmup_steps')
        if saved_warmup is not None and saved_warmup != args.warmup_steps:
            print(f'resume restored warmup_steps {args.warmup_steps} -> {saved_warmup} (warmup 是 LR 调度定义的一部分)')
            args.warmup_steps = saved_warmup
        saved_grad_clip = rng_payload.get('args', {}).get('grad_clip')
        if saved_grad_clip is not None and saved_grad_clip != args.grad_clip:
            print(f'resume restored grad_clip {args.grad_clip} -> {saved_grad_clip} (grad_clip 决定优化轨迹)')
            args.grad_clip = saved_grad_clip
        saved_eval_batches = rng_payload.get('args', {}).get('eval_batches')
        if saved_eval_batches is not None and saved_eval_batches != args.eval_batches:
            print(f'resume restored eval_batches {args.eval_batches} -> {saved_eval_batches} (验证窗口集合由 batch_size*eval_batches 决定)')
            args.eval_batches = saved_eval_batches
        saved_eval_interval = rng_payload.get('args', {}).get('eval_interval')
        if saved_eval_interval is not None and saved_eval_interval != args.eval_interval:
            print(f'resume restored eval_interval {args.eval_interval} -> {saved_eval_interval} (周期验证步点是验证协议的一部分)')
            args.eval_interval = saved_eval_interval
        saved_eval_bootstrap = rng_payload.get('args', {}).get('eval_bootstrap')
        if isinstance(saved_eval_bootstrap, int) and (not isinstance(saved_eval_bootstrap, bool)) and (saved_eval_bootstrap >= 0) and (saved_eval_bootstrap != args.eval_bootstrap):
            print(f'resume restored eval_bootstrap {args.eval_bootstrap} -> {saved_eval_bootstrap} (bootstrap 次数是评估协议的一部分)')
            args.eval_bootstrap = saved_eval_bootstrap
        saved_eval_bootstrap_block = rng_payload.get('args', {}).get('eval_bootstrap_block')
        if isinstance(saved_eval_bootstrap_block, int) and (not isinstance(saved_eval_bootstrap_block, bool)) and (saved_eval_bootstrap_block > 0) and (saved_eval_bootstrap_block != args.eval_bootstrap_block):
            print(f'resume restored eval_bootstrap_block {args.eval_bootstrap_block} -> {saved_eval_bootstrap_block} (bootstrap 重采样单位是评估协议的一部分)')
            args.eval_bootstrap_block = saved_eval_bootstrap_block
        saved_periodic_auc = rng_payload.get('args', {}).get('periodic_auc')
        if isinstance(saved_periodic_auc, bool) and saved_periodic_auc != bool(args.periodic_auc):
            print(f'resume restored periodic_auc {bool(args.periodic_auc)} -> {saved_periodic_auc} (周期排序指标是评估协议的一部分)')
            args.periodic_auc = saved_periodic_auc
        saved_strict = rng_payload.get('args', {}).get('strict_determinism')
        if saved_strict is not None and bool(saved_strict) != bool(args.strict_determinism):
            print(f'resume restored strict_determinism {args.strict_determinism} -> {saved_strict} (确定性实现开关是数值环境的一部分)')
            args.strict_determinism = bool(saved_strict)
            torch.use_deterministic_algorithms(True, warn_only=not args.strict_determinism)
        saved_threads = rng_payload.get('args', {}).get('torch_threads')
        if saved_threads is not None and saved_threads != args.torch_threads:
            print(f'resume restored torch_threads {args.torch_threads} -> {saved_threads} (CPU 线程数影响浮点归约顺序)')
            args.torch_threads = saved_threads
            configure_cpu_threads(saved_threads)
    scheduler = build_lr_schedule_from_args(optimizer, args, start_step=start_step)
    if rng_payload is not None and rng_payload.get('scheduler') is not None:
        scheduler.load_state_dict(rng_payload['scheduler'])
        for group, base_lr, lr_lambda in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
            group['lr'] = base_lr * lr_lambda(scheduler.last_epoch)
        scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
    if rng_payload is not None:
        del rng_payload
    if reference_window != (args.batch_size, args.eval_batches):
        reference_baselines = compute_reference_baselines(dataset, args.batch_size, args.eval_batches, torch.device('cpu'), split='val')
    train_generator = torch.Generator().manual_seed(args.seed)
    if train_generator_state is not None:
        train_generator.set_state(train_generator_state)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    n_params = sum((p.numel() for p in model.parameters()))
    print(f'mode={cfg.mode} params={n_params:,}')
    print(f'device={device} start_step={start_step}')
    scale_guard = build_scale_guard(cfg, args, int(n_params), dataset)
    args.scale_guard = scale_guard
    print(f'[scale] tokens_seen={scale_guard['tokens_seen']:,} tokens/param={scale_guard['tokens_per_param']:.2f} epochs={scale_guard['epochs_over_train']} tier={scale_guard['scale_tier']}')
    if scale_guard['scale_tier'] == 'prototype':
        print('[scale] [warn] 原型级规模：结果只支持可行性/机制验证，不能外推到大规模 MoE、长上下文或推理任务；需要多来源、多上下文长度与多模型规模的复核')
    config_sanity_result = sanity_result
    if config_sanity_result is None and args.resume is not None:
        config_sanity_result = load_previous_sanity_result(out_dir)
    run_metadata = collect_run_metadata(device)
    write_run_config(out_dir, cfg, args, n_params, run_metadata, config_sanity_result, data_audit=data_audit, reference_baselines=reference_baselines, baseline_tuning=baseline_tuning_record, data_protocol_check=data_protocol_check, dataset=dataset, init_fingerprints=getattr(model, 'init_fingerprints', None))
    tracking_provenance = load_tracking_provenance(out_dir)
    metrics_path = out_dir / 'metrics.jsonl'
    curve_baseline_loss = min(math.log(cfg.vocab_size), float(reference_baselines['unigram_loss']))
    curve_records = _records_at_or_before_step(read_metrics_records(metrics_path), start_step) if args.resume is not None else []
    last_curve_status = None
    last_curve_warning_flags = None
    if curve_records:
        prior_analysis = analyze_training_curve(curve_records, baseline_loss=curve_baseline_loss)
        prior_status = str(prior_analysis.get('status'))
        if prior_status != 'insufficient_data':
            last_curve_status = prior_status
            prior_flags = prior_analysis.get('warning_flags')
            if isinstance(prior_flags, dict):
                last_curve_warning_flags = {str(key): bool(value) for key, value in prior_flags.items()}
    recent_grad_norms = []
    best_selection_value = None
    best_selection_step = None
    evals_without_improvement = 0
    if args.resume is not None:
        best_selection_value, best_selection_step, evals_without_improvement = early_stop_state_from_records(_records_at_or_before_step(read_metrics_records(metrics_path), start_step), args.early_stop_min_delta, metric=args.selection_metric)
    selected_model_path = out_dir / _SELECTED_MODEL_FILENAME
    early_stopped = False

