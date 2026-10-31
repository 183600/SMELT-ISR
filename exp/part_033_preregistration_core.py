from __future__ import annotations



# ============================================================
# 外部预注册文件（--preregistration）
# ============================================================

# 预注册不是训练后写进 config.json 的一段说明，而应是训练前
# 独立固定、事后只能核对不能改写的协议。文件字段覆盖判定口径、
# 共同步数、对照臂、种子集合与数据协议；入口只允许用文件
# 补齐未显式给出的命令行参数，显式冲突直接拒绝启动。
_PREREGISTRATION_REQUIRED_FIELDS = (
    "study_id",
    "frozen_commit",
    "primary_metric",
    "common_step",
    "min_effect",
    "direction",
    "baseline_mode",
    "comparisons",
    "seeds",
    "budget_match",
    "data",
)


_PREREGISTRATION_DATA_REQUIRED_FIELDS = (
    "sha256",
    "split_policy",
    "val_fraction",
    "test_fraction",
)


# 可选的清洗协议：清洗会改变实际参与训练/去重的字节范围，
# 看到验证结果后再改清洗开关等于事后改数据。字段名与命令行
# dest 一致，便于逐项核对。
_PREREGISTRATION_CLEANING_BOOL_KEYS = (
    "deduplicate_train_groups",
    "drop_train_shared_chunks",
    "drop_train_near_duplicate_groups",
    "drop_train_corrupt_groups",
    "drop_train_corrupt_bytes",
    "drop_train_degenerate_groups",
    "drop_train_degenerate_runs",
    "clean_train_against_test",
    "allow_eval_content_for_cleaning",
)


# 近重复阈值与 shingle 长度直接决定删除哪些训练群组，和清洗
# 开关一样必须在训练前冻结；它们不是布尔量，单独校验。
_PREREGISTRATION_CLEANING_NUMERIC_KEYS = (
    "near_duplicate_threshold",
    "near_duplicate_shingle_bytes",
)


_PREREGISTRATION_CLEANING_KEYS = (
    _PREREGISTRATION_CLEANING_BOOL_KEYS
    + _PREREGISTRATION_CLEANING_NUMERIC_KEYS
)


_PREREGISTRATION_OPTIONAL_FIELDS = (
    "equivalence_margin",
    "steps",
    "frozen_script_sha256",
    "power",
    "run_order",
    "notes",
    "metadata",
)


# 预注册的**先验功效分析**（power）与**运行顺序设计**
# （run_order）。二者回答的是「训练开始前，这个设计有没有
# 能力回答研究问题」，而不是训练结束后的解释：
# - power：声明预期种子间标准差与最小有意义效应，按家族
#   校正后的 alpha 计算所需种子数；种子数不足时默认拒绝，
#   除非显式承认并写明理由；
# - run_order：把「哪个臂先跑」从便利顺序变成预先声明的
#   随机化完整区组（randomized complete block），避免机器
#   热漂移/负载漂移与处理效应混杂。
_POWER_KEYS = (
    "expected_seed_sd_relative",
    "target_power",
    "alpha",
    "planned_comparisons",
    "planned_contrasts",
    "expected_attrition_rate",
    "pilot_calibration",
    "pilot_confidence",
    "plan_equivalence",
    "underpowered_acknowledged",
    "justification",
)


_RUN_ORDER_KEYS = ("scheme", "seed")


# grouped 保留历史顺序（按模式分组，时间漂移与臂完全混杂）；
# blocked 按种子成区组、区组内保持原有臂顺序；randomized 在
# 每个种子区组内随机化臂顺序（默认）；counterbalanced 在完整
# 区组的基础上用循环拉丁方/平衡排列，使每个处理在各区组位置
# 上的出现次数尽可能相等，避免小样本随机化偶然把某个臂系统性
# 地放在更晚/更早的时间片；williams 在完整区组上使用 Williams
# 设计，除位置平衡外还平衡区组内相邻处理的一阶 carryover
# （偶数处理 n 个区组、奇数处理 2n 个区组可完全平衡）。
_RUN_ORDER_SCHEMES = (
    "grouped",
    "blocked",
    "randomized",
    "counterbalanced",
    "williams",
)


_DESIGN_PLAN_SCHEMA_VERSION = 2

_DESIGN_PLAN_FILENAME = "design_plan.json"

_RUN_ORDER_FILENAME = "run_order.json"


_PREREGISTRATION_DATA_OPTIONAL_FIELDS = (
    "group_separator",
    "cleaning",
    "entity_manifest_sha256",
    "time_order_manifest_sha256",
)


_PREREGISTRATION_MISSING = object()



def _preregistration_finite_number(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        return f"{label} 必须是有限数值"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return f"{label} 必须是有限数值"
    if not math.isfinite(number):
        return f"{label} 必须是有限数值"
    if minimum is not None and number < minimum:
        return f"{label} 必须 >= {minimum:g}"
    if maximum is not None and number > maximum:
        return f"{label} 必须 <= {maximum:g}"
    return None



def _preregistration_sha256_text(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", value)
    )



def _contrast_power_entry(
    contrast: str,
    min_effect: float,
    sd: float,
    planned_seeds: int,
    usable_seeds: int,
    alpha_adjusted: float,
    target: float,
    attrition: float,
    required_robust: int,
    *,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """按某一对比类型的方差乘数计算功效/所需种子数。

    ``equivalence_margin`` 给定时，同时按同一对比类型的方差
    计算「真实差异为零时 TOST 能否主张等价」所需的种子数与
    计划可用种子下的等价性功效。显著性（能否检出最小效应）
    与等价性（能否在预注册界内主张无实质差异）是两个不同的
    设计目标：前者防「漏掉真效应」，后者防把「功效不足的
    不显著」读成「等价」。
    """
    factor = _CONTRAST_VARIANCE_FACTORS[contrast]
    sd_contrast = sd * math.sqrt(factor)
    if alpha_adjusted > 0.0:
        required_t = _seeds_to_detect(
            float(min_effect),
            sd_contrast,
            target_power=target,
            alpha=alpha_adjusted,
        )
    else:
        required_t = None
    required = max(required_t or 0, int(required_robust))
    if required_t is None:
        required_with_attrition = None
    else:
        required_with_attrition = int(
            math.ceil(float(required) / (1.0 - attrition))
        )
    achieved_power = None
    mde = None
    if alpha_adjusted > 0.0 and usable_seeds >= _MIN_SEEDS_FOR_STATS:
        se = sd_contrast / math.sqrt(float(usable_seeds))
        df = float(usable_seeds - 1)
        achieved_power = _t_power(
            float(min_effect),
            se,
            df,
            alpha_adjusted,
        )
        mde = _minimum_detectable_effect(
            se,
            df,
            target,
            alpha_adjusted,
        )
    adequate = (
        required_t is not None
        and int(planned_seeds)
        >= int(required_with_attrition or required)
    )
    entry: Dict[str, object] = {
        "contrast": contrast,
        "label": _CONTRAST_LABELS[contrast],
        "variance_factor": factor,
        "se_multiplier": math.sqrt(factor),
        # 该对比实际使用的（配对差值口径）标准差；两两/简单
        # 效应等于声明值，主效应更小，交互更大。
        "expected_seed_sd_relative": sd_contrast,
        "required_seeds_for_t": required_t,
        "required_seeds": required,
        "required_seeds_with_attrition": required_with_attrition,
        "achieved_power_at_min_effect": achieved_power,
        "minimum_detectable_effect_relative": mde,
        "adequate": adequate,
    }

    # 等价性设计：在真实差异为零时，TOST 需要观测到的标准化
    # 差值整体落在 ±(margin - t·se) 内。margin 与 sd 同尺度
    # （相对基线均值）。主效应/交互的方差乘数同样适用：交互
    # 的差值方差更大，主张「交互等价」需要更宽的界或更多种子。
    if (
        equivalence_margin is not None
        and math.isfinite(float(equivalence_margin))
        and float(equivalence_margin) > 0.0
    ):
        margin = float(equivalence_margin)
        required_eq = _seeds_for_tost(
            sd_contrast,
            margin,
            target_power=target,
            alpha=alpha_adjusted if alpha_adjusted > 0.0 else 0.0,
            variance_factor=1.0,
        ) if alpha_adjusted > 0.0 else None
        required_eq_with_attrition = (
            int(
                math.ceil(
                    float(required_eq) / (1.0 - attrition)
                )
            )
            if required_eq is not None
            else None
        )
        eq_power = None
        eq_margin_needed = None
        if (
            alpha_adjusted > 0.0
            and usable_seeds >= _MIN_SEEDS_FOR_STATS
        ):
            eq_se = sd_contrast / math.sqrt(float(usable_seeds))
            eq_df = float(usable_seeds - 1)
            eq_power = _tost_power(
                0.0,
                eq_se,
                eq_df,
                margin,
                alpha_adjusted,
            )
            eq_margin_needed = _tost_margin_for_power(
                eq_se,
                eq_df,
                target,
                alpha_adjusted,
            )
        entry.update({
            "equivalence_margin_relative": margin,
            "equivalence_power_at_null_difference": eq_power,
            "required_seeds_for_equivalence": required_eq,
            "required_seeds_for_equivalence_with_attrition": (
                required_eq_with_attrition
            ),
            "equivalence_margin_for_target_power": (
                eq_margin_needed
            ),
            "equivalence_adequate": (
                required_eq is not None
                and int(planned_seeds)
                >= int(required_eq_with_attrition or required_eq)
            ),
        })
    return entry



def _pick_binding_contrast(
    entries: Dict[str, Dict[str, object]],
    contrasts: List[str],
) -> str:
    """选出最吃种子的对比类型；所需种子相同时优先报交互。"""
    order = {
        contrast: index
        for index, contrast in enumerate(_CONTRAST_ORDER)
    }
    binding = contrasts[0]
    for contrast in contrasts[1:]:
        current = entries[binding]
        candidate = entries[contrast]
        current_required = current.get("required_seeds_for_t")
        candidate_required = candidate.get("required_seeds_for_t")
        if current_required is None:
            # None 表示「有限种子数内无法达到目标功效」，是最
            # 严格的结果，不能被后面的有限值覆盖。
            continue
        if candidate_required is None:
            binding = contrast
            continue
        if (
            int(candidate_required) > int(current_required)
            or (
                int(candidate_required) == int(current_required)
                and order[contrast] > order[binding]
            )
        ):
            binding = contrast
    return binding



def preregistration_power_plan(
    power: Dict[str, object],
    min_effect: float,
    n_seeds: int,
    comparisons_count: int,
    contrast_types: Optional[List[str]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    先验功效分析：在训练前判断预注册的种子数能否检出预注册
    的最小有意义效应。

    预期种子间标准差与最小效应都按**相对基线均值**解释，因此
    二者相除后是尺度无关的效应量（Cohen's d 的配对版本），
    不需要事先知道基线均值。``expected_seed_sd_relative`` 是
    **两格配对差值**的标准差；主效应/交互的方差不同（见
    ``_CONTRAST_VARIANCE_FACTORS``），因此不能只用两两对比的
    公式回答「消融设计是否足够」。

    多重比较按 Bonferroni 下界
    ``alpha / planned_comparisons`` 计算；这与分析阶段的 Holm
    首步阈值一致，是保守口径，不会把功效说得比实际更乐观。

    ``contrast_types`` 声明本次研究会主张哪些对比类型
    （``pairwise`` / ``simple`` / ``main`` / ``interaction``）。
    每种类型分别计算所需种子数，``adequate`` 取最吃种子的
    类型（通常是交互）判定，并在 ``contrast_power`` 中逐项
    报告；不传时退回预注册的 ``planned_contrasts``，再退回
    只按 ``pairwise``（与旧报告兼容）。

    同时给出精确符号翻转在家族校正下所需的最小种子数。稳健
    主张（配对 t 与符号翻转同时显著）要求二者都满足；只用 t
    检验的主张仍可用 ``required_seeds_for_t`` 判断。

    ``equivalence_margin`` 给定时，额外计算等价性（TOST）设计：
    真实差异为零时，在预注册等价界内主张「无实质差异」所需
    的可用配对数、计划可用种子下的等价性功效，以及当前精度
    能支持的最小等价界。``power.plan_equivalence=true`` 表示
    等价性是本次研究的主张之一，此时 ``adequate`` 同时要求
    显著性功效与等价性功效达标；否则只报告信息性字段，不
    改变显著性功效的判定。等价界必须在预注册顶层显式给出
    （``--equivalence-margin`` 的默认值不能代替事先声明）。
    """
    # 入口 schema 会拒绝损坏的 power 块；直接调用时也必须
    # 可判定：超范围整数不能让 float()/int() 抛异常，也不能
    # 把 inf/nan 写进最终 JSON。sd 无有限上界可用时取一个
    # 足够大的有限值，使功效判定为「不可达」而不是乐观通过。
    raw_sd = _finite_float(power.get("expected_seed_sd_relative"))
    sd = raw_sd if raw_sd is not None and raw_sd >= 0.0 else 1.0e300
    target = _finite_float(
        power.get("target_power", _TARGET_POWER)
    )
    if target is None or not (0.0 < target < 1.0):
        target = _TARGET_POWER
    alpha = _finite_float(power.get("alpha", _SIGNIFICANCE_LEVEL))
    if alpha is None or not (0.0 < alpha < 0.5):
        alpha = _SIGNIFICANCE_LEVEL
    try:
        planned_comparisons = int(
            power.get("planned_comparisons", comparisons_count)
        )
    except (TypeError, ValueError, OverflowError):
        planned_comparisons = int(comparisons_count)
    family_size = max(1, planned_comparisons)
    min_effect_value = _finite_float(min_effect)
    if min_effect_value is None or min_effect_value < 0.0:
        min_effect_value = 0.0
    if contrast_types is None:
        raw_contrasts = power.get(
            "planned_contrasts", ["pairwise"]
        )
    else:
        raw_contrasts = contrast_types
    contrasts = _normalize_contrast_types(raw_contrasts)
    try:
        alpha_adjusted = alpha / family_size
    except OverflowError:
        # 家族规模是合法 JSON 整数但大到无法转成浮点：校正后
        # 的 alpha 实际为 0，任何有限 n 都不可能达到目标功效。
        alpha_adjusted = 0.0
    required_robust = _sign_flip_required_seeds(alpha, family_size)
    acknowledged = bool(
        power.get("underpowered_acknowledged", False)
    )
    plan_equivalence = bool(power.get("plan_equivalence", False))
    if equivalence_margin is None:
        equivalence_margin = power.get(
            "equivalence_margin_relative"
        )
    try:
        equivalence_margin_value = (
            None
            if equivalence_margin is None
            else float(equivalence_margin)
        )
    except (TypeError, ValueError, OverflowError):
        equivalence_margin_value = None
    if (
        equivalence_margin_value is not None
        and (
            not math.isfinite(equivalence_margin_value)
            or equivalence_margin_value <= 0.0
        )
    ):
        equivalence_margin_value = None

    # 预期数据脱落/数值失败：功效公式里的 n 是**可用**运行数，
    # 不是启动数。若某个臂历史上约有 20% 的运行会 NaN 发散，
    # 计划 8 个种子只得到约 6.4 个可用配对；按 8 计算会系统性
    # 高估功效。这里把预期脱落率作为预注册的一部分，反推需要
    # 启动多少种子，并把「计划可用种子数」与「达到目标功效所
    # 需启动数」分开报告。
    raw_attrition = power.get("expected_attrition_rate", 0.0)
    try:
        attrition = float(raw_attrition)
    except (TypeError, ValueError, OverflowError):
        attrition = 0.0
    if (
        not math.isfinite(attrition)
        or attrition < 0.0
        or attrition >= 1.0
    ):
        # 直接调用本函数时也要保持可判定；结构错误由
        # preregistration_schema_error 在入口拒绝。
        attrition = 0.0
    # n_seeds 可能来自手工/旧记录里的超大 JSON 整数：
    # float(10**400) 会 OverflowError。保留原始值用于报告，
    # 但功效数值计算用一个足够大的上限，避免 sqrt/df 溢出；
    # 对任何实际设计，1e6 个种子的功效已饱和。
    try:
        raw_n_seeds = int(n_seeds)
    except (TypeError, ValueError, OverflowError):
        raw_n_seeds = 0
    compute_n_seeds = max(1, min(raw_n_seeds, 1_000_000))
    usable_seeds = max(
        1,
        min(
            compute_n_seeds,
            int(
                math.floor(
                    float(compute_n_seeds)
                    * (1.0 - attrition)
                    + 1e-12
                )
            ),
        ),
    )

    contrast_power = {
        contrast: _contrast_power_entry(
            contrast,
            min_effect_value,
            sd,
            raw_n_seeds,
            usable_seeds,
            alpha_adjusted,
            target,
            attrition,
            required_robust,
            equivalence_margin=equivalence_margin_value,
        )
        for contrast in contrasts
    }
    binding = _pick_binding_contrast(contrast_power, contrasts)
    binding_entry = contrast_power[binding]
    required_t = binding_entry["required_seeds_for_t"]
    required = max(
        int(entry["required_seeds"])
        for entry in contrast_power.values()
    )
    required_with_attrition = binding_entry[
        "required_seeds_with_attrition"
    ]
    superiority_adequate = all(
        bool(entry["adequate"])
        for entry in contrast_power.values()
    )

    # 等价性设计：与显著性功效分开报告。若 plan_equivalence=true，
    # 等价性是主张之一， inadequacy 必须进入总体 adequate；
    # 否则只作为信息，提醒「当前设计即使真实差异为零也无法在
    # 声明界内主张等价」。
    equivalence: Optional[Dict[str, object]] = None
    equivalence_adequate: Optional[bool] = None
    if equivalence_margin_value is not None:
        eq_binding = _pick_binding_contrast(
            {
                contrast: {
                    "required_seeds_for_t": entry.get(
                        "required_seeds_for_equivalence"
                    )
                }
                for contrast, entry in contrast_power.items()
            },
            contrasts,
        )
        eq_entry = contrast_power[eq_binding]
        eq_required = eq_entry.get(
            "required_seeds_for_equivalence"
        )
        eq_required_attrition = eq_entry.get(
            "required_seeds_for_equivalence_with_attrition"
        )
        equivalence_adequate = all(
            bool(entry.get("equivalence_adequate"))
            for entry in contrast_power.values()
        )
        equivalence = {
            "claim_planned": plan_equivalence,
            "margin_relative": equivalence_margin_value,
            "binding_contrast": eq_binding,
            "binding_contrast_label": _CONTRAST_LABELS[eq_binding],
            "planned_usable_seeds": usable_seeds,
            "power_at_null_difference": eq_entry.get(
                "equivalence_power_at_null_difference"
            ),
            "required_seeds_for_equivalence": eq_required,
            "required_seeds_with_attrition": (
                eq_required_attrition
            ),
            "margin_for_target_power": eq_entry.get(
                "equivalence_margin_for_target_power"
            ),
            "adequate": equivalence_adequate,
            "note": (
                "等价性功效按真实差异为零、TOST 每个单侧检验的 "
                "alpha/m 与同一配对差值标准差计算；margin 越宽越容易"
                "主张等价，因此界必须在训练前固定。若 "
                "margin_for_target_power 大于声明的界，说明"
                "当前种子数精度不足以在该界内主张等价，应加"
                "种子而不是放宽界。"
            ),
        }
    adequate = bool(superiority_adequate) and (
        not plan_equivalence or bool(equivalence_adequate)
    )

    # 在计划可用种子数下报告实际能达到的功效与 MDE：仅说
    # 「需要 8 个」不够，研究者还需要看到「按当前计划即使
    # 真实效应等于最小有意义效应，也只有 0.6 的功效」这类
    # 直接可执行的诊断。MDE 与 min_effect 同单位（相对基线
    # 均值），可直接比较。绑定对比的方差最大，其功效/MDE
    # 就是整个计划最保守的那一项。
    achieved_power = binding_entry["achieved_power_at_min_effect"]
    mde = binding_entry["minimum_detectable_effect_relative"]

    # 对「预期种子间标准差估计偏小/偏大」做敏感性分析。先验
    # 功效最大的不确定性来自 sd 本身；只报一个 required_seeds
    # 会把估计误差藏在点估计后面。每种 sd 下仍按最吃种子的
    # 对比类型报告，避免敏感性表比主表更乐观。
    sensitivity: List[Dict[str, object]] = []
    for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        scaled_sd = sd * multiplier
        scaled_entries = {
            contrast: _contrast_power_entry(
                contrast,
                min_effect_value,
                scaled_sd,
                raw_n_seeds,
                usable_seeds,
                alpha_adjusted,
                target,
                attrition,
                required_robust,
                equivalence_margin=equivalence_margin_value,
            )
            for contrast in contrasts
        }
        scaled_binding = _pick_binding_contrast(
            scaled_entries, contrasts
        )
        scaled_entry = scaled_entries[scaled_binding]
        scaled_superiority = all(
            bool(entry["adequate"])
            for entry in scaled_entries.values()
        )
        scaled_equivalence = (
            all(
                bool(entry.get("equivalence_adequate"))
                for entry in scaled_entries.values()
            )
            if equivalence_margin_value is not None
            else None
        )
        sensitivity.append({
            "sd_multiplier": multiplier,
            "expected_seed_sd_relative": scaled_sd,
            "binding_contrast": scaled_binding,
            "required_seeds_for_t": scaled_entry[
                "required_seeds_for_t"
            ],
            "required_seeds_with_attrition": scaled_entry[
                "required_seeds_with_attrition"
            ],
            "superiority_adequate_at_planned_seeds": (
                scaled_superiority
            ),
            "equivalence_adequate_at_planned_seeds": (
                scaled_equivalence
            ),
            "adequate_at_planned_seeds": (
                scaled_superiority
                and (
                    not plan_equivalence
                    or bool(scaled_equivalence)
                )
            ),
        })

    return {
        "expected_seed_sd_relative": sd,
        "min_effect_relative": min_effect_value,
        "target_power": target,
        "alpha": alpha,
        "planned_comparisons": family_size,
        "alpha_bonferroni": alpha_adjusted,
        "planned_contrasts": list(contrasts),
        "binding_contrast": binding,
        "binding_contrast_label": _CONTRAST_LABELS[binding],
        "contrast_power": contrast_power,
        "contrast_variance_factor": binding_entry[
            "variance_factor"
        ],
        "required_seeds_for_t": required_t,
        "required_seeds_for_robust_sign_flip": required_robust,
        # required_seeds 保持「无脱落时的启动数」，便于与旧报告
        # 对比；required_seeds_with_attrition 才是本设计的实际
        # 种子预算。
        "required_seeds": required,
        "expected_attrition_rate": attrition,
        "planned_seeds": raw_n_seeds,
        "planned_usable_seeds": usable_seeds,
        "required_seeds_with_attrition": required_with_attrition,
        "achieved_power_at_min_effect": achieved_power,
        "minimum_detectable_effect_relative": mde,
        "power_sensitivity": sensitivity,
        "plan_equivalence": plan_equivalence,
        "equivalence_margin_relative": equivalence_margin_value,
        "equivalence": equivalence,
        "superiority_adequate": superiority_adequate,
        "equivalence_adequate": equivalence_adequate,
        "adequate": adequate,
        "underpowered_acknowledged": acknowledged,
        "justification": power.get("justification"),
        "note": (
            "所需种子数把声明的预期种子间标准差当作真实 sigma；"
            "它本身也是估计，规划时应留余量。adequacy 按计划中"
            "最吃种子的对比类型（binding_contrast）判定：主效应"
            "方差是两两对比的 0.5 倍，交互是 2 倍，因此同一"
            "最小效应下交互约需 2 倍种子。adequacy 只表示设计"
            "在声明效应下有目标功效，不保证真实效应一定达到"
            "最小有意义效应。planned_usable_seeds 按声明的"
            "预期脱落率折算，required_seeds_with_attrition 是"
            "达到目标功效所需的启动种子数。plan_equivalence=true"
            "时 adequate 还要求等价性（TOST）功效达标；等价性"
            "功效按真实差异为零计算，是「若确实无实质差异，"
            "能否在预注册界内主张等价」的概率，不保证真实"
            "差异为零"
        ),
    }
