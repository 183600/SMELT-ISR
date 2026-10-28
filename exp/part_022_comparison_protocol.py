from __future__ import annotations


_LOWER_IS_BETTER_METRICS = {
    "val_loss",
    "val_loss_at_matched_flops",
    _CURVE_MEAN_METRIC,
    "val_ppl",
    "val_bits_per_byte",
    # 类平衡似然：同样是 NLL，越低越好。
    "val_macro_nll",
    *(
        base + _COMMON_STEP_METRIC_SUFFIX
        for base in (
            "val_loss",
            "val_ppl",
            "val_bits_per_byte",
            "val_macro_nll",
        )
    ),
    # RQ3：第二轮选择与第一轮不同的专家（重合度越低越独立）；
    # RQ4：死专家越少越好、分支轨迹越分化（cosine 越低）说明
    # 延迟合并确实产生了独立计算。
    "routing_jaccard_branch",
    "routing_jaccard_token",
    "routing_slot_reuse_branch",
    "routing_branch_self_reuse",
    "routing_r2_dead_experts",
    "routing_branch_cosine",
}


# ``final`` 中来源于验证集选中模型的 val 指标：它们都会被
# checkpoint 选择影响。等计算量、全部共同步数指标与曲线时均
# 指标都是曲线派生值，不在其中。
_SELECTION_BIASED_METRICS = frozenset(
    key
    for key in COMPARISON_METRIC_KEYS
    if key != _COMPUTE_MATCHED_METRIC
    and key not in _COMMON_STEP_METRICS
    and key not in _CURVE_DERIVED_METRICS
)


# 周期验证实际会计算的指标才有资格作为 checkpoint 选型与早停
# 依据。AUC 默认只在最终/测试评估时计算；但当它被显式指定为
# 选型指标时，训练循环会在每个验证点计算同口径的 PR-AUC /
# ROC-AUC（见训练入口的 compute_auc 开关），因此可以作为选型
# 依据——极度不平衡场景推荐以 PR-AUC 判定，选型口径必须能跟着
# 判定口径走，而不是固定按 val_loss 选模型。路由诊断要等训练
# 结束后的一次专门前向，等计算量 / 共同步数 / 曲线时均指标更是
# --compare 阶段才从曲线派生，训练过程中拿不到，仍不能用于选型。
_SELECTION_METRIC_KEYS = tuple(
    key
    for key in COMPARISON_METRIC_KEYS
    if key not in _ROUTING_ROW_FIELDS
    and key
    not in {
        "val_loss_at_matched_flops",
        *_COMMON_STEP_METRICS,
        *_CURVE_DERIVED_METRICS,
    }
)


# 排序指标：周期验证默认不算；被选为选型指标或显式
# --periodic-auc 时按与 final/test 相同的固定窗口与排序口径
# 计算。集中定义避免三处字符串列表漂移。
_AUC_METRIC_KEYS = frozenset({
    "val_pr_auc_macro",
    "val_pr_auc_macro_lift",
    "val_roc_auc_macro",
})



def _periodic_auc_enabled(
    periodic_auc: object,
    selection_metric: object,
) -> bool:
    """周期验证是否实际计算排序指标。

    AUC 选型会自动启用周期 AUC，与显式 ``--periodic-auc``
    等价；协议记录生效值，避免两种等价配置被误报为不同口径。
    """
    return bool(periodic_auc) or (
        isinstance(selection_metric, str)
        and selection_metric in _AUC_METRIC_KEYS
    )



# 简单基线调参可用的目标指标：与对比指标同口径（含最终评估
# 才计算的 AUC），但不含路由诊断与共同步数派生指标——调参
# 必须用训练时真实产生、可直接优化/选择的最终 val 指标；
# 派生指标只用于分析期选型无关复核，不能反过来充当调参目标。
_BASELINE_TUNING_METRIC_KEYS = tuple(
    key
    for key in COMPARISON_METRIC_KEYS
    if key not in _ROUTING_ROW_FIELDS
    and key not in _COMMON_STEP_METRICS
    and key not in _CURVE_DERIVED_METRICS
    and key != _COMPUTE_MATCHED_METRIC
)


# n-gram 基线没有训练曲线，无法定义「等训练计算量」或
# 「共同步数」指标；其余与简单神经基线共用同一组口径。
_NGRAM_TUNING_METRIC_KEYS = tuple(
    key
    for key in _BASELINE_TUNING_METRIC_KEYS
    if key != "val_loss_at_matched_flops"
)


# 有逐窗口同口径估计、可对 n-gram (order, alpha) 网格做
# 逐窗口配对 one-SE 选择的分类指标（confusion_matrix_metrics
# 的键，去掉 val_ 前缀）；AUC 不在此列，逐窗口类支持稀疏，
# 逐窗口估计与全局口径不一致。
_NGRAM_PER_WINDOW_CLASSIFICATION_METRICS = frozenset({
    "token_accuracy",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
})


# 因子的自然排序（未循环基线 → 立即合并对照 → 延迟合并），
# 用于让「基线→改进」的差值符号可解释。
_FACTOR_VALUE_ORDER = {
    "base": 0,
    "smelt": 1,
    "local_base": 2,
    "local_base_k2": 3,
    "A": 4,
    "B": 5,
    "B_early": 6,
    "dense": 7,
    "ngram": 8,
}


# 显著性与统计功效的判定阈值。
#
# 统计检验至少需要 3 个随机种子（配对/ Welch t 检验的 df ≥ 2）：
# 2 个种子时 df=1，t 分布尾部极厚，同样的差值既能给出 p≈0.5 也
# 能给出 p≈0.05，显著性判定基本由噪声决定，不足以支撑「方法
# 更好」的结论。少于 3 个种子只汇报差值，不给 p 值与显著性。
_SIGNIFICANCE_LEVEL = 0.05

_MIN_SEEDS_FOR_STATS = 3

_RECOMMENDED_SEEDS = 3

# 精确配对符号翻转检验的最小可达双侧 p 是 2/2^n：n=3/4/5 时
# 分别为 0.25/0.125/0.0625，均大于 0.05。若把「配对 t 与符号
# 翻转同时显著」作为稳健结论的必要条件，n≤5 的设计在数学上
# 不可能满足；至少需要 n=6（2/64=0.03125），家族校正后还要
# 更多。规划种子时必须把这一点与预注册最小效应/主检验一起
# 考虑，而不是先跑 3~5 个种子再解释为什么「不稳健」。
_MIN_SEEDS_FOR_ROBUST_SIGN_FLIP = 6



def _sign_flip_required_seeds(
    alpha: float = _SIGNIFICANCE_LEVEL,
    family_size: int = 1,
) -> int:
    """精确符号翻转在给定家族校正下所需的最小种子数。

    Holm 首步阈值为 alpha/m；要求 2/2^n <= alpha/m，即
    n >= ceil(log2(2m/alpha))。家族规模 m=1、alpha=0.05 时为 6。
    """
    try:
        alpha_finite = math.isfinite(alpha)
    except (TypeError, ValueError, OverflowError):
        return _MIN_SEEDS_FOR_ROBUST_SIGN_FLIP
    if not alpha_finite or alpha <= 0.0:
        return _MIN_SEEDS_FOR_ROBUST_SIGN_FLIP
    m = max(1, int(family_size))
    # 用 log2(m) + 1 - log2(alpha) 而不是 log2(2*m/alpha)：
    # 后者会先把 m 转成浮点，m 很大时溢出；alpha 极小（如
    # 5e-324）时 2*m/alpha 又会变成 inf，使 ceil 抛错。
    # 分开取对数后，正常范围与极端合法数值都能给出整数结果。
    required = math.ceil(
        math.log2(m) + 1.0 - math.log2(float(alpha))
    )
    return max(_MIN_SEEDS_FOR_STATS, int(required))



# 功效分析的目标功效与实际意义等价界。
#
# 显著性检验只能回答「有没有证据反对差异为 0」，回答不了
# 「差异是否小到可以忽略」：p>0.05 只说明现有数据没能证明
# 差异存在，既不等于差异不存在，也不等于差异可以忽略。
# 因此报告额外给出两类口径：
# - 最小可检出效应（MDE）：在当前种子数、当前种子间方差的
#   条件下，达到 _TARGET_POWER 功效（惯例 80%）所能检出的
#   最小真实效应；观测到的 |Δ| 小于 MDE 时不显著属预期，
#   不能当作「无差异」的证据；
# - 等价性检验（TOST）：在预先声明的等价界（相对基线均值的
#   比例，_DEFAULT_EQUIVALENCE_MARGIN = 1%）内做两个单侧
#   检验，显著通过才能主张「无实质差异」，否则结论只能是
#   「未发现显著差异（不确定）」。
_TARGET_POWER = 0.80

_DEFAULT_EQUIVALENCE_MARGIN = 0.01


# 全因子/消融对比的类型 → 方差乘数（相对于两格配对差值）。
#
# 假设单个实验单元的种子间噪声方差为 σ_cell²，两格配对差
# y_A − y_B 的方差是 2σ_cell²；预注册的
# ``expected_seed_sd_relative`` 就是该**配对差值**的标准差。
# 在 2×2 设计（L_off、L_on、M_off、M_on）中：
# - 两两/简单效应：两个格子相减，方差 2σ_cell² → 1×；
# - 主效应 0.5[(M_on−M_off)+(L_on−L_off)]：方差 σ_cell² → 0.5×；
# - 交互 (M_on−M_off)−(L_on−L_off)：方差 4σ_cell² → 2×。
# 因此交互的标准误是两两对比的 √2 倍，同样效应下约需 2 倍
# 种子。只按两两对比算「够不够」会把消融交互的功效系统性
# 高估；预注册的功效门槛必须按实际计划中最吃种子的对比类型
# （通常是 interaction）判定。
_CONTRAST_VARIANCE_FACTORS = {
    "pairwise": 1.0,
    "simple": 1.0,
    "main": 0.5,
    "interaction": 2.0,
}

_CONTRAST_ORDER = ("pairwise", "simple", "main", "interaction")

_CONTRAST_LABELS = {
    "pairwise": "两两对比",
    "simple": "简单效应",
    "main": "主效应",
    "interaction": "交互效应",
}



def _contrast_types_error(value: object) -> Optional[str]:
    """校验 ``power.planned_contrasts`` 的取值；返回错误或 None。"""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return (
            "power.planned_contrasts 必须是字符串数组，可选值: "
            + ", ".join(_CONTRAST_ORDER)
        )
    if not value:
        return "power.planned_contrasts 不能为空"
    seen = set()
    for item in value:
        if not isinstance(item, str):
            return (
                "power.planned_contrasts 的每一项必须是字符串，"
                "可选值: "
                + ", ".join(_CONTRAST_ORDER)
            )
        if item not in _CONTRAST_VARIANCE_FACTORS:
            return (
                "power.planned_contrasts 含未知对比类型 "
                f"{item!r}；可选值: "
                + ", ".join(_CONTRAST_ORDER)
            )
        if item in seen:
            return (
                "power.planned_contrasts 含重复类型: " + item
            )
        seen.add(item)
    return None



def _normalize_contrast_types(
    value: object,
) -> List[str]:
    """把对比类型规范化成去重后的有序列表；非法输入抛 ValueError。"""
    error = _contrast_types_error(value)
    if error is not None:
        raise ValueError(error)
    values = [str(item) for item in value]  # type: ignore[union-attr]
    # 用固定顺序输出，避免预注册里换个书写顺序就得到不同的
    # 计划指纹；同时让交互等较弱的对比在展示时排在后面。
    return [
        contrast
        for contrast in _CONTRAST_ORDER
        if contrast in values
    ]


# 预注册最小效应按对照均值的比例解释。对照均值恰为零（或
# 小到低于该绝对下限）时，比例阈值退化为 0 或远低于数值
# 分辨率，任何非零差值都会被误判成「达到最小效应」。这类
# 相对阈值没有可解释的尺度，必须报告为不可判定，而不是
# 静默判成 supported。
_HYPOTHESIS_MIN_EFFECT_ABS_FLOOR = 1.0e-12


# 非中心 t 功效的数值积分设置：对 u = sqrt(V)（V ~ chi2_df）
# 做复合 Simpson 积分。u 的密度以 sqrt(df-1) 为众数、宽度
# 约 1/√2，右尾为 exp(-u^2/2)：积分上限取众数 + _T_POWER_TAIL
# （右尾 < 1e-31 可截断），节点数随上限自适应，df 很大时
# 也不会把全部质量丢在积分区间之外。
_T_POWER_TAIL = 12.0

_T_POWER_MIN_INTERVALS = 800

# df<1 时 t=u^(1/p) 的幂次至少为 3，800 个区间在 df=0.5、
# δ=100 的转折处仍只有几个节点，非中心 t CDF 会偏高约
# 4.8e-5；这些低自由度单独提高到 1600，避免拖慢 df>=1 的
# 常用功效计算。
_T_POWER_SMALL_DF_MIN_INTERVALS = 1600

_T_POWER_MAX_INTERVALS = 4000

# df 达到该量级时，固定 u 网格上的 Simpson 会因 chi 密度
# 宽约 1/√2 而节点不足；非中心 t 也已充分接近正态极限。
_T_POWER_NORMAL_DF = 1.0e6



def _build_seed_plan(
    seeds: List[int],
    allow_underpowered: bool,
    planned_comparisons: int = 1,
) -> Dict[str, object]:
    """种子计划：同时记录 t 检验与精确符号翻转的功效下限。

    ``min_seeds_for_significance`` 是配对/ Welch t 检验的最低
    种子数（3）；``min_seeds_for_robust_sign_flip`` 是精确符号
    翻转在给定家族规模下能达到 alpha 的最小 n（n=6，家族校正
    后更大）。n<6 时 ``significant_robust`` 在数学上不可能为
    True，报告必须区分「效应不稳健」与「设计无法判定稳健性」。
    """
    # 重复 seed 是伪重复，不是独立重复：种子计划若按列表长度
    # 计数，会在一份 --seeds 1 1 1 1 1 1 的计划上声称具备 6 个
    # 独立种子的稳健性。按稳定键去重后再算功效下限。
    unique_seeds: List[int] = []
    seen_keys: set = set()
    repeated: List[int] = []
    for seed in seeds:
        value = int(seed)
        key = _stable_factor_key(value)
        if key in seen_keys:
            repeated.append(value)
            continue
        seen_keys.add(key)
        unique_seeds.append(value)
    n = len(unique_seeds)
    required_robust = _sign_flip_required_seeds(
        _SIGNIFICANCE_LEVEL, planned_comparisons
    )
    return {
        "seeds": unique_seeds,
        "n_seeds": n,
        "duplicate_seeds": repeated,
        "min_seeds_for_significance": _MIN_SEEDS_FOR_STATS,
        "min_seeds_for_robust_sign_flip": required_robust,
        "planned_comparisons": int(planned_comparisons),
        "robust_claim_possible": n >= required_robust,
        "robust_claim_note": (
            "精确符号翻转双侧最小 p=2/2^n；n=3/4/5 时为 "
            "0.25/0.125/0.0625，均不能达到 0.05。"
            "significant_robust（配对 t 与符号翻转同时显著）"
            "至少需要 n=6，家族校正后按 "
            "ceil(log2(2m/alpha)) 增加。重复 seed 按一个"
            "独立观测计。"
        ),
        "underpowered_override": bool(
            allow_underpowered and n < _MIN_SEEDS_FOR_STATS
        ),
    }



def _independent_seed_count(
    n_runs: int,
    seeds: List[object],
) -> int:
    """
    实验单元的有效独立种子数。

    同一随机种子出现多次的运行是伪重复（复制的同一运行），
    不能算作独立重复实验：记录的种子按去重计数；未记录种子的
    运行无法判断彼此是否重复，按独立计入（只扣除确定的
    伪重复，不虚增种子数）。
    """
    recorded = len(seeds)
    unique = len({_stable_factor_key(seed) for seed in seeds})
    return unique + max(0, int(n_runs) - recorded)


# 效率与精度的权衡：对比实验除指标涨点外还必须报告参数量与
# 计算量（FLOPs）等模型规模指标；效率倍率达到基线的该阈值
# 时，即使统计显著也必须在结论中显式权衡「涨点是否值得」。
_COST_RATIO_MATERIAL = 1.5


# 参与效率权衡的字段（顺序即展示顺序）。各字段分别反映
# 模型结构与训练规模的差异。
_COST_FIELDS: Tuple[str, ...] = (
    "n_params",
    "flops_per_forward",
    "latency_ms_per_forward",
    "training_ms_per_step",
    "peak_memory_mb",
)


_COST_FIELD_LABELS = {
    "n_params": "参数",
    "flops_per_forward": "FLOPs",
    "latency_ms_per_forward": "推理计时",
    "training_ms_per_step": "训练步计时",
    "peak_memory_mb": "峰值占用",
}


# 计时字段：它们的成本倍率可能由单次测量的噪声/负载波动
# 驱动，跨运行解读时必须带离散度；结构字段（参数/FLOPs）
# 对同一实现是确定量。
_COST_TIMING_FIELDS: Tuple[str, ...] = (
    "latency_ms_per_forward",
    "training_ms_per_step",
)


# 成本倍率的 95% 置信区间用 delta 方法近似（对数尺度）。
# 两侧重复测量通常只有 3 个种子，正态 z=1.96 会系统性低估
# 小样本不确定性；改用 Welch-Satterthwaite 自由度与 t 临界值。
# 离散度超过该阈值时明确标注「计时噪声可能主导」。
_COST_RATIO_CI_ALPHA = 0.05

_COST_NOISE_CV_LIMIT = 0.10


# 需要运行结束测量的效率字段（参数量是结构属性，旧运行也
# 常有；这些字段全缺说明该运行没有做效率测量，精度与效率的
# 权衡无法评估，必须显式点出而不是静默跳过）。
# 训练步计时同样是后加入的补充字段：旧运行没有该记录不改变
# 「是否做过效率测量」的判定，避免把所有旧运行都误标成缺效率
# 数据；有该字段时照常参与倍率与权衡判定。
_MEASURED_COST_FIELDS: Tuple[str, ...] = (
    "flops_per_forward",
    "latency_ms_per_forward",
    "peak_memory_mb",
)



def _cell_missing_measured_costs(
    cell: Dict[str, object],
) -> List[str]:
    """列出一个实验单元缺失的效率字段。"""
    costs = cell.get("costs")
    if not isinstance(costs, dict):
        return list(_MEASURED_COST_FIELDS)
    return [
        field
        for field in _MEASURED_COST_FIELDS
        if field not in costs
    ]



def _missing_efficiency_cell_labels(
    aggregates: List[Dict[str, object]],
) -> List[str]:
    """列出真正缺效率测量数据的实验单元（确定性基线除外）。"""
    return [
        str(aggregate["label"])
        for aggregate in aggregates
        if _cell_missing_measured_costs(aggregate)
        and not aggregate.get("deterministic")
    ]



def _cell_cost_mean(
    cell: Dict[str, object],
    field: str,
) -> Optional[float]:
    """取一个实验单元在某效率字段上的平均值（缺失时 None）。"""
    costs = cell.get("costs")
    if not isinstance(costs, dict):
        return None
    entry = costs.get(field)
    if not isinstance(entry, dict):
        return None
    # 旧/损坏的分析记录可能把 mean 写成 10**400；float() 会
    # 溢出并让整份效率对比崩溃。不可用值按缺失处理。
    return _finite_float(entry.get("mean"))



def _cell_cost_std(
    cell: Dict[str, object],
    field: str,
) -> Optional[float]:
    """取一个实验单元在某效率字段上的样本标准差。"""
    costs = cell.get("costs")
    if not isinstance(costs, dict):
        return None
    entry = costs.get(field)
    if not isinstance(entry, dict):
        return None
    return _finite_float(entry.get("std"))



def _cell_cost_n(
    cell: Dict[str, object],
    field: str,
) -> int:
    """取一个实验单元在某效率字段上的有效观测数。"""
    costs = cell.get("costs")
    if not isinstance(costs, dict):
        return 0
    entry = costs.get(field)
    if not isinstance(entry, dict):
        return 0
    count = entry.get("n")
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    ):
        return count
    return 0



def _device_type_of(value) -> Optional[str]:
    """
    把 --device 归一化成设备类型（"cuda:0" → "cuda"），
    供受控变量核对；设备索引不同不代表数值环境不同，但
    CPU / CUDA / MPS 之间浮点行为不同，跨类型对比不可归因。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("mps"):
        return "mps"
    return text.split(":", 1)[0]



def build_comparison_protocol(
    cfg: Config,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """
    汇总一条运行的全部「受控变量」，写入 config.json。

    优化器（AdamW）及其 betas/eps、学习率与调度、训练步数、
    早停（实际训练轮数）、weight decay、梯度裁剪、batch size、
    数据增强、数值环境（设备类型 / 混合精度）与实现开关
    （grad checkpoint / compile）、数据指纹、模型结构与评估
    协议全部显式记录；--compare 时默认要求这些值在声明因子
    之外逐项一致，避免把「换了优化器、调了学习率或训练更久」
    误当成方法本身的收益。
    """
    return {
        "comparison_family": "neural",
        # 模型模式是结构差异本身；默认作为实验因子，但一旦
        # 因子列表没有它，就必须按受控变量核对。
        "mode": cfg.mode,
        # 初始化方案：name_hash 下同种子的共享参数逐位一致，
        # module_order 下不同 mode 的构造顺序会改变共享层初始化。
        "init_scheme": cfg.init_scheme,
        # 优化器固定为 AdamW（betas/eps 使用 PyTorch 默认值），
        # 显式记录以便未来更换优化器时立即被对比检查发现。
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "lr": args.lr,
        "lr_schedule": (
            "linear_warmup_cosine_to_10pct"
            if str(getattr(args, "lr_schedule", "cosine"))
            == "cosine"
            else "wsd"
        ),
        "lr_schedule_name": str(
            getattr(args, "lr_schedule", "cosine")
        ),
        "wsd_stable_steps": getattr(
            args, "wsd_stable_steps", None
        ),
        "wsd_final_lr_ratio": getattr(
            args, "wsd_final_lr_ratio", None
        ),
        "warmup_steps": args.warmup_steps,
        "total_steps": args.steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        # 文档级 segment mask 改变注意力可见性，属于模型语义，
        # 跨运行对比必须一致（包括分隔符字节）。
        "document_masks": bool(cfg.document_masks),
        "document_separator": (
            list(cfg.document_separator)
            if cfg.document_separator is not None
            else None
        ),
        # 当前字节级管线不做数据增强；显式记录空操作，
        # 未来加入增强后不同运行会立刻被对比检查发现。
        "data_augmentation": "none",
        # 早停决定有效训练轮数；数值环境与实现开关决定
        # 浮点行为与反传路径。
        "early_stop_patience": getattr(
            args, "early_stop_patience", None
        ),
        "early_stop_min_delta": getattr(
            args, "early_stop_min_delta", None
        ),
        "amp_dtype": getattr(args, "amp_dtype", None),
        "device_type": _device_type_of(
            getattr(args, "device", None)
        ),
        "allow_tf32": bool(
            getattr(args, "allow_tf32", False)
        ),
        "grad_checkpoint": bool(
            getattr(args, "grad_checkpoint", False)
        ),
        "compile": bool(getattr(args, "compile", False)),
        "seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.effective_n_kv_heads,
        "n_layers": cfg.n_layers,
        "n_experts": cfg.n_experts,
        "expert_hidden": cfg.expert_hidden,
        "top_k": cfg.top_k,
        # 分支路由几何与均衡损失强度：即使调用方缩减了
        # --compare-factors，也必须作为受控变量被核对。
        "k1": cfg.k1,
        "k2": cfg.k2,
        "balance_coef": cfg.balance_coef,
        "balance_width_normalization": (
            _BALANCE_WIDTH_NORMALIZATION
            if cfg.mode == "local_base"
            else "not_applicable"
        ),
        "dense_fnn_hidden": cfg.dense_fnn_hidden,
        "residual_scale": cfg.residual_scale,
        # 循环区域残差缩放口径决定循环内子层的有效残差强度：
        # paper（1/r）与 global 是两种不同架构，不能跨口径比较。
        "residual_scaling": cfg.residual_scaling,
        "loop_residual_scale": cfg.loop_residual_scale,
        "loop_count": cfg.loop_count,
        "val_fraction": getattr(args, "val_fraction", None),
        "test_fraction": getattr(args, "test_fraction", None),
        "split_policy": getattr(
            args, "split_policy", "contiguous"
        ),
        # 群组分隔符决定切分位置（同一文件 + 不同分隔符 =
        # 不同的数据划分），必须作为受控变量核对。
        "group_separator": getattr(args, "group_separator", None),
        "deduplicate_train_groups": bool(
            getattr(args, "deduplicate_train_groups", False)
        ),
        "drop_train_shared_chunks": bool(
            getattr(args, "drop_train_shared_chunks", False)
        ),
        "drop_train_near_duplicate_groups": bool(
            getattr(
                args,
                "drop_train_near_duplicate_groups",
                False,
            )
        ),
        "near_duplicate_threshold": (
            _near_duplicate_threshold_or_default(
                getattr(
                    args,
                    "near_duplicate_threshold",
                    _NEAR_DUP_DEFAULT_THRESHOLD,
                )
            )
        ),
        "near_duplicate_shingle_bytes": (
            _near_duplicate_shingle_or_default(
                getattr(
                    args,
                    "near_duplicate_shingle_bytes",
                    _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
                )
            )
        ),
        # 训练集清洗是否对照 test：默认 False；True 意味着
        # 训练样本选择读取过 test，不能与其他运行混为同一
        # 数据协议。
        # 两个同义入口按 OR 归一化；直接调用本函数（未经 main
        # 的 _normalize_clean_against_test）时也不能只读其中一个
        # 键，否则同一清洗口径会被记成两个不同协议。
        "clean_train_against_test": _clean_against_test_effective(
            args
        ),
        "drop_train_corrupt_groups": bool(
            getattr(args, "drop_train_corrupt_groups", False)
        ),
        "drop_train_corrupt_bytes": bool(
            getattr(args, "drop_train_corrupt_bytes", False)
        ),
        "drop_train_degenerate_groups": bool(
            getattr(args, "drop_train_degenerate_groups", False)
        ),
        "drop_train_degenerate_runs": bool(
            getattr(args, "drop_train_degenerate_runs", False)
        ),
        "allow_eval_content_for_cleaning": (
            _clean_against_test_effective(args)
        ),
        "allow_entity_overlap": bool(
            getattr(args, "allow_entity_overlap", False)
        ),
        "allow_data_leakage": bool(
            getattr(args, "allow_data_leakage", False)
        ),
        "allow_corrupt_test_labels": bool(
            getattr(args, "allow_corrupt_test_labels", False)
        ),
        "allow_eval_duplicates": bool(
            getattr(args, "allow_eval_duplicates", False)
        ),
        "eval_interval": getattr(args, "eval_interval", None),
        "eval_batches": getattr(args, "eval_batches", None),
        # 固定评估窗口 bootstrap 的重采样次数：区间宽度与
        # 蒙特卡洛精度的一部分，不能跨运行静默改变。
        "eval_bootstrap": getattr(
            args, "eval_bootstrap", None
        ),
        # 重采样单位同样是评估协议的一部分。
        "eval_bootstrap_block": getattr(
            args, "eval_bootstrap_block", 1
        ),
        # 周期验证是否计算排序指标决定无偏共同步数 AUC 指标
        # 是否存在。AUC 选型会自动启用，因此记录生效值而非
        # 原始 flag，避免等价配置被误报为协议差异。
        "periodic_auc": _periodic_auc_enabled(
            getattr(args, "periodic_auc", False),
            getattr(args, "selection_metric", "val_loss"),
        ),
        "data_sha256": getattr(args, "data_sha256", None),
        # 逐 split 内容指纹：同源文件但不同群组吸附/清洗/切分
        # 结果的运行不能算同一数据协议。
        "split_sha256": (
            dict(_as_dict(
                getattr(args, "data_split_sha256", None)
            ))
            or None
        ),
        # 清洗后的有效训练集内容指纹：split 与开关相同仍可能
        # 因清洗实现变化得到不同训练样本集合。
        "effective_train_sha256": getattr(
            args,
            "data_effective_train_sha256",
            None,
        ),
        # 时序隔离记录：边界、顺序与重叠检查一起进入受控
        # 协议，防止「同一文件 + 同一比例」掩盖切分顺序变化。
        "split_isolation": getattr(
            args,
            "data_split_isolation",
            None,
        ),
        # 选型协议：final/test 都在验证集选中的模型上评估
        # （best.pt 不可用时回退到最后一步并明确提示）。
        "model_selection": "best_val_with_last_fallback",
        "selection_metric": getattr(
            args, "selection_metric", "val_loss"
        ),
    }



def experiment_protocol_from_payload(
    payload: Dict[str, object],
) -> Dict[str, object]:
    """
    从 config.json 读取受控变量协议；旧运行没有
    comparison_protocol 字段时按 args/config/data 尽力回退
    （能唯一确定的设置才回退，无法确定的留 None，由检查器
    标记为「缺记录」而不是静默视为一致）。

    记录过的协议项优先；记录中缺失的协议项（例如旧版本运行
    没有后来新增的 k1/k2/balance_coef/dense_fnn_hidden）用可
    回退字段补齐，避免新旧运行混用时把「未记录」误判成
    「值不同/缺失」。
    """
    cfg = payload.get("config") or {}
    run_args = payload.get("args") or {}
    data = payload.get("data") or {}
    run_metadata = payload.get("run_metadata") or {}

    if not isinstance(cfg, dict):
        cfg = {}
    # 记录过的模式值可能来自手工编辑/旧版本的 JSON，类型不保证
    # 是字符串；后面的集合成员判断必须先确认可哈希。
    mode_value = cfg.get("mode")
    if not isinstance(run_args, dict):
        run_args = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(run_metadata, dict):
        run_metadata = {}

    def _pick(*values):
        for value in values:
            if value is not None:
                return value
        return None

    has_training = run_args.get("lr") is not None

    fallback = {
        "comparison_family": (
            "statistical"
            if cfg.get("mode") == "ngram"
            else "neural"
        ),
        # 旧运行没有把 mode 写进协议时，config.json 的 mode
        # 是唯一确定的实际模式；用它回填，避免跨模式对比在
        # 因子列表漏掉 mode 时静默通过。
        "mode": cfg.get("mode"),
        # 历史运行只有按模块构造顺序初始化这一种方案；新运行
        # 记录实际方案。缺记录的旧运行不能默认成 name_hash。
        "init_scheme": (
            cfg.get("init_scheme")
            if cfg.get("init_scheme") is not None
            else ("module_order" if has_training else None)
        ),
        # 代码历史上只有一种优化器与一种调度，能唯一确定时
        # 回退；否则留 None。
        "optimizer": "AdamW" if has_training else None,
        "optimizer_betas": (
            [0.9, 0.999] if has_training else None
        ),
        "optimizer_eps": 1e-8 if has_training else None,
        # 缺 comparison_protocol 时按 args 里存档的实际调度回填：
        # 不能一律假定 cosine，否则旧 WSD 运行的调度会被误记成
        # cosine，跨运行协议核对要么误报不一致，要么漏掉真实差异。
        "lr_schedule": (
            (
                "linear_warmup_cosine_to_10pct"
                if str(
                    run_args.get("lr_schedule", "cosine")
                )
                == "cosine"
                else "wsd"
            )
            if has_training
            else None
        ),
        "lr_schedule_name": (
            str(run_args.get("lr_schedule", "cosine"))
            if has_training
            else None
        ),
        "wsd_stable_steps": run_args.get(
            "wsd_stable_steps"
        ),
        "wsd_final_lr_ratio": run_args.get(
            "wsd_final_lr_ratio"
        ),
        "lr": run_args.get("lr"),
        "warmup_steps": run_args.get("warmup_steps"),
        "total_steps": run_args.get("steps"),
        "weight_decay": run_args.get("weight_decay"),
        "grad_clip": run_args.get("grad_clip"),
        "batch_size": run_args.get("batch_size"),
        # 历史运行没有文档级 mask；缺记录时回填 False。
        # config.json 的 Config 字段是旧运行唯一可靠的来源：
        # argparse 没有 document_separator 这个 dest，只读
        # run_args 会把它误报成缺失。
        "document_masks": (
            bool(_pick(
                run_args.get("document_masks"),
                cfg.get("document_masks"),
                False,
            ))
            if has_training
            else None
        ),
        "document_separator": _pick(
            run_args.get("document_separator"),
            cfg.get("document_separator"),
        ),
        "data_augmentation": "none" if has_training else None,
        # 早停 / 数值环境 / 实现开关：旧运行的 args 里已记录
        # 的部分直接回填（无法唯一确定的留 None 由检查器标记
        # 缺记录，例如旧版本未记录 amp_dtype）。
        "early_stop_patience": (
            run_args.get("early_stop_patience")
            if has_training
            else None
        ),
        "early_stop_min_delta": (
            run_args.get("early_stop_min_delta")
            if has_training
            else None
        ),
        "amp_dtype": run_args.get("amp_dtype"),
        "device_type": _device_type_of(run_args.get("device")),
        # 旧运行的 args 没有 allow_tf32；run_metadata 里记录了
        # 当时实际生效的 matmul/cudnn TF32 状态。两者一致时才
        # 能唯一回填，否则留 None 由检查器标记缺记录。
        "allow_tf32": (
            run_args.get("allow_tf32")
            if run_args.get("allow_tf32") is not None
            else (
                bool(run_metadata.get("allow_tf32_matmul"))
                if (
                    run_metadata.get("allow_tf32_matmul")
                    == run_metadata.get("allow_tf32_cudnn")
                    and run_metadata.get("allow_tf32_matmul")
                    is not None
                )
                else None
            )
        ),
        "grad_checkpoint": (
            run_args.get("grad_checkpoint")
            if has_training
            else None
        ),
        "compile": (
            run_args.get("compile") if has_training else None
        ),
        "seq_len": cfg.get("seq_len"),
        "d_model": cfg.get("d_model"),
        "n_heads": cfg.get("n_heads"),
        # 旧运行都是 MHA（没有 n_kv_heads 字段）：按 n_heads
        # 回填，避免把旧 MHA 运行误报成缺记录。
        "n_kv_heads": _pick(
            cfg.get("n_kv_heads"),
            cfg.get("n_heads"),
        ),
        "n_layers": cfg.get("n_layers"),
        "n_experts": cfg.get("n_experts"),
        "expert_hidden": cfg.get("expert_hidden"),
        "top_k": cfg.get("top_k"),
        "k1": cfg.get("k1"),
        "k2": cfg.get("k2"),
        "balance_coef": cfg.get("balance_coef"),
        # 旧运行没有该字段：历史 local_base 不做宽度折算，必须
        # 显式记为 "none" 才能与新运行区分；其余模式没有该
        # 折算问题，记为 "not_applicable"，避免把「旧 A/B 与
        # 新 A/B 语义没变」误报成协议差异。mode 也缺失时留
        # None，由检查器标为未核对。
        "balance_width_normalization": (
            "none"
            if has_training and cfg.get("mode") == "local_base"
            else "not_applicable"
            if has_training and cfg.get("mode") is not None
            else None
        ),
        "dense_fnn_hidden": cfg.get("dense_fnn_hidden"),
        "residual_scale": cfg.get("residual_scale"),
        # 旧运行只有 global 口径（residual_scale 全局生效）。
        "residual_scaling": (
            cfg.get("residual_scaling")
            if cfg.get("residual_scaling") is not None
            else ("global" if has_training else None)
        ),
        "loop_residual_scale": _pick(
            cfg.get("loop_residual_scale"),
            cfg.get("residual_scale") if has_training else None,
        ),
        "loop_count": _pick(
            cfg.get("loop_count"),
            (
                1
                if isinstance(mode_value, str)
                and mode_value in {"base", "dense"}
                else 2
            )
            if has_training and isinstance(mode_value, str)
            else None,
        ),
        "val_fraction": _pick(
            data.get("val_fraction"),
            run_args.get("val_fraction"),
        ),
        "split_policy": _pick(
            data.get("split_policy"),
            run_args.get("split_policy"),
            "contiguous" if has_training else None,
        ),
        "test_fraction": _pick(
            data.get("test_fraction"),
            run_args.get("test_fraction"),
        ),
        "group_separator": _pick(
            data.get("group_separator"),
            run_args.get("group_separator"),
        ),
        # 数据清洗/泄露豁免对统计基线同样生效：ByteDataset 在
        # 两个入口之前统一构建，ngram 也读取清洗后的 train 与
        # 泄露审计结果。旧运行没有记录这些键时才回退 False，
        # 不能因为「ngram 没有优化器」就丢弃它实际使用的数据
        # 口径。
        "deduplicate_train_groups": bool(
            run_args.get("deduplicate_train_groups", False)
        ),
        "drop_train_shared_chunks": bool(
            run_args.get("drop_train_shared_chunks", False)
        ),
        "drop_train_near_duplicate_groups": bool(
            run_args.get(
                "drop_train_near_duplicate_groups", False
            )
        ),
        "near_duplicate_threshold": (
            _near_duplicate_threshold_or_default(
                _pick(
                    run_args.get(
                        "near_duplicate_threshold"
                    ),
                    _NEAR_DUP_DEFAULT_THRESHOLD,
                )
            )
        ),
        "near_duplicate_shingle_bytes": (
            _near_duplicate_shingle_or_default(
                _pick(
                    run_args.get(
                        "near_duplicate_shingle_bytes"
                    ),
                    _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
                )
            )
        ),
        # 旧运行没有该键时按历史行为回填：旧版只要启用去重/
        # 共享块清洗就会读取 test。不能默认成新版的 False，
        # 否则会把「旧运行实际用过 test」误报成与新运行一致。
        "clean_train_against_test": (
            _saved_clean_against_test_effective(run_args)
        ),
        "drop_train_corrupt_groups": bool(
            run_args.get("drop_train_corrupt_groups", False)
        ),
        "drop_train_corrupt_bytes": bool(
            run_args.get("drop_train_corrupt_bytes", False)
        ),
        "drop_train_degenerate_groups": bool(
            run_args.get("drop_train_degenerate_groups", False)
        ),
        "drop_train_degenerate_runs": bool(
            run_args.get("drop_train_degenerate_runs", False)
        ),
        "allow_data_leakage": bool(
            run_args.get("allow_data_leakage", False)
        ),
        # 这两个豁免也在 COMPARISON_PROTOCOL_KEYS 中；旧配置
        # 没有 comparison_protocol 时必须从 args 回填，否则
        # 同一个有效口径会被当成「缺记录」或与新版不一致。
        "allow_eval_content_for_cleaning": (
            _saved_clean_against_test_effective(run_args)
        ),
        "allow_entity_overlap": bool(
            run_args.get("allow_entity_overlap", False)
        ),
        "allow_corrupt_test_labels": bool(
            run_args.get("allow_corrupt_test_labels", False)
        ),
        "allow_eval_duplicates": bool(
            run_args.get("allow_eval_duplicates", False)
        ),
        "eval_interval": run_args.get("eval_interval"),
        "eval_batches": run_args.get("eval_batches"),
        # 旧运行未记录时保持 None（检查器标为缺记录），
        # 不能默认成当前 CLI 的 500。
        "eval_bootstrap": run_args.get("eval_bootstrap"),
        # 旧运行没有记录重采样单位；保持 None 由检查器标为
        # 未核对，不能默认成当前 CLI 的 1 而掩盖块 bootstrap 差异。
        "eval_bootstrap_block": run_args.get(
            "eval_bootstrap_block"
        ),
        # 旧神经运行没有该字段时，历史行为等价于 False；
        # 但若当时已按 AUC 选型，周期验证实际会计算排序指标，
        # 回退为生效值。ngram 没有周期验证，保持 None。
        "periodic_auc": (
            _periodic_auc_enabled(
                run_args.get("periodic_auc", False),
                _pick(
                    run_args.get("selection_metric"),
                    "val_loss",
                ),
            )
            if has_training
            else None
        ),
        "data_sha256": data.get("sha256"),
        # 逐 split 指纹：新运行写在 data.split_sha256；旧运行
        # 没有该字段时保持 None，由检查器列为「未核对」而不是
        # 默认为一致。
        "split_sha256": data.get("split_sha256"),
        # 清洗后的有效训练集指纹：旧运行没有记录时保持 None，
        # 由检查器列为「未核对」，不能默认成一致。
        "effective_train_sha256": _pick(
            data.get("effective_train_sha256"),
            run_args.get("data_effective_train_sha256"),
        ),
        # 时序隔离记录：旧运行没有时保持 None，由检查器标记为
        # 未核对；新运行也可从 data.split_isolation 回填。
        "split_isolation": _pick(
            data.get("split_isolation"),
            run_args.get("data_split_isolation"),
        ),
        # 旧运行没有选型记录，其 final/test 一律基于最后一步
        # 参数；显式回退为 "last_step"，与新版选型协议混用时
        # 会被受控变量检查标记出来。
        "model_selection": (
            "last_step"
            if run_args.get("lr") is not None
            else None
        ),
        # 旧运行（没有该字段）一律按历史行为在 val_loss 上
        # 选型；参数里已有记录的按记录回填。
        "selection_metric": _pick(
            run_args.get("selection_metric"),
            "val_loss" if has_training else None,
        ),
    }

    protocol = payload.get("comparison_protocol")
    if isinstance(protocol, dict):
        result = dict(fallback)
        result.update(protocol)

        # 记录过的协议里 model_selection 是常量
        # "best_val_with_last_fallback"（说明支持回退），但训练
        # 结束写入的顶层 model_selection.policy 才是实际选型结果：
        # best.pt 缺失/不一致时 final/test 指标来自未经验证集选择
        # 的最后一步参数，必须与正常选型区分，否则受控变量检查
        # 无法发现两类运行被混在一起比较。
        selection = payload.get("model_selection")
        if isinstance(selection, dict):
            policy = selection.get("policy")
            if policy == "last":
                result["model_selection"] = "last_step"
            elif policy == "best_val":
                result["model_selection"] = (
                    "best_val_with_last_fallback"
                )

        result.setdefault("comparison_family", "neural")
        return result

    return fallback



def parse_factor_spec(spec: str) -> List[str]:
    """
    解析 --compare-factors 的逗号分隔列表；未知键显式报错，
    避免拼写错误静默关闭受控变量检查。
    """
    keys: List[str] = []
    for token in spec.split(","):
        key = token.strip()
        if key and key not in keys:
            keys.append(key)

    if not keys:
        raise ValueError("对比因子列表不能为空")

    candidates = set(COMPARISON_PROTOCOL_KEYS) | {
        "mode",
        "k1",
        "k2",
        "top_k",
        "balance_coef",
        "dense_fnn_hidden",
        "seed",
    }

    unknown = [key for key in keys if key not in candidates]
    if unknown:
        raise ValueError(
            "未知的对比因子: "
            + ", ".join(unknown)
            + "（可用: "
            + ", ".join(sorted(candidates))
            + "）"
        )

    # 选型协议不是实验因子：把「按 val_loss 选 checkpoint」与
    # 「按 macro F1 选 checkpoint」声明为因子，看起来在比较
    # 方法，实际比较的是选型规则 + 早停轨迹，最终指标差异
    # 无法归因于方法本身。
    locked = [
        key
        for key in keys
        if key
        in {
            "model_selection",
            "selection_metric",
            "init_scheme",
        }
    ]
    if locked:
        raise ValueError(
            "实现/选型协议不能声明为实验因子: "
            + ", ".join(locked)
            + "（初始化方案决定共享层是否同起点，"
            "--selection-metric 决定 checkpoint 怎么选，"
            "跨运行必须一致；需要比较这些规则时请单独做"
            "方法学实验，而不是把它们混进方法对比）"
        )

    return keys



def suite_factor_error(
    factors: List[str],
    suite_modes: Optional[List[str]],
    suite_ablation: bool,
) -> Optional[str]:
    """
    套件启动前的分析口径校验：套件会不会真的分析它声称的轴。

    ``--suite-ablation`` 的主效应/交互以 mode 与 balance_coef
    为分组因子；``--suite-modes`` 的跨模式对比以 mode 为分组
    因子。如果调用方用 ``--compare-factors`` 把这些键收窄成
    受控变量，训练仍然会跑，但分析阶段不会按该轴分组，套件
    会产出一份「看起来完成、实际没有检验该轴」的报告。这里
    在开跑前直接拒绝，避免浪费算力后才发现分析口径不对。
    """
    factor_set = set(factors)
    required: List[Tuple[str, str]] = []

    if suite_ablation:
        required = [
            ("mode", "消融方案轴"),
            ("balance_coef", "均衡损失开/关轴"),
        ]
    elif suite_modes:
        required = [("mode", "跨模式对比轴")]

    missing = [
        f"{key}（{purpose}）"
        for key, purpose in required
        if key not in factor_set
    ]
    if not missing:
        return None

    return (
        "套件的分析因子被 --compare-factors 排除: "
        + ", ".join(missing)
        + "。套件需要这些键作为分组因子才能做对应分析；"
        "请把它们加入 --compare-factors（例如 "
        "--compare-factors mode,balance_coef,seed）"
    )
