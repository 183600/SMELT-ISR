from __future__ import annotations



def preregistration_schema_error(
    payload: Dict[str, object],
) -> Optional[str]:
    """
    校验预注册文件的结构与取值域，返回错误消息或 None。

    这里只检查「文件自身是否完整、可判定」；命令行、代码
    提交与数据指纹的一致性由
    ``preregistration_apply`` / ``preregistration_data_error``
    核对。结构上缺字段时不能默认通过：缺 ``common_step``
    会让分析回退到数据依赖端点，缺 ``comparisons`` 则无法
    核对本次运行的臂是否属于同一项研究，二者都会把预注册
    退化成事后说明。
    """
    missing = [
        key
        for key in _PREREGISTRATION_REQUIRED_FIELDS
        if key not in payload
    ]
    if missing:
        return "缺少必需字段: " + ", ".join(missing)
    unknown = sorted(
        set(payload)
        - set(_PREREGISTRATION_REQUIRED_FIELDS)
        - set(_PREREGISTRATION_OPTIONAL_FIELDS)
    )
    if unknown:
        return "含未知字段（拼写错误会让协议静默失效）: " + ", ".join(
            unknown
        )
    if "notes" in payload and not isinstance(payload["notes"], str):
        return "notes 必须是字符串"
    if "metadata" in payload and not isinstance(
        payload["metadata"], dict
    ):
        return "metadata 必须是对象"

    study_id = payload.get("study_id")
    if (
        not isinstance(study_id, str)
        or not study_id.strip()
        or "\n" in study_id
        or "\r" in study_id
    ):
        return "study_id 必须是非空单行字符串"

    frozen_commit = payload.get("frozen_commit")
    if (
        not isinstance(frozen_commit, str)
        or not re.fullmatch(r"[0-9a-fA-F]{7,64}", frozen_commit)
    ):
        return (
            "frozen_commit 必须是 7~64 位十六进制 git commit "
            "（冻结代码版本，不能用分支名或工作区状态代替）"
        )

    primary_metric = payload.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric:
        return "primary_metric 必须是非空字符串"
    if (
        primary_metric not in COMPARISON_METRIC_KEYS
        and primary_metric not in _NGRAM_TUNING_METRIC_KEYS
    ):
        return (
            f"primary_metric={primary_metric} 不是评估管线计算的"
            "指标"
        )

    common_step = payload.get("common_step")
    if (
        isinstance(common_step, bool)
        or not isinstance(common_step, int)
        or common_step <= 0
    ):
        return "common_step 必须是正整数（训练前固定的比较步）"

    min_effect = payload.get("min_effect")
    error = _preregistration_finite_number(
        min_effect,
        "min_effect",
        minimum=0.0,
        maximum=1.0,
    )
    if error is not None:
        return error
    if float(min_effect) <= 0.0:  # type: ignore[arg-type]
        return "min_effect 必须是正数（相对基线均值的比例）"

    direction = payload.get("direction")
    if direction not in ("decrease", "increase"):
        return "direction 必须是 decrease 或 increase"

    baseline_mode = payload.get("baseline_mode")
    if (
        not isinstance(baseline_mode, str)
        or baseline_mode not in Config.ALL_MODES
    ):
        return (
            "baseline_mode 必须是已知模式: "
            + ", ".join(sorted(Config.ALL_MODES))
        )

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        return "comparisons 必须是非空的 [arm_a, arm_b] 列表"
    seen_pairs = set()
    declared_modes = set()
    for index, pair in enumerate(comparisons):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(
                isinstance(arm, str) and arm for arm in pair
            )
        ):
            return (
                f"comparisons[{index}] 必须是长度为 2 的模式名"
                "列表（如 [\"smelt\", \"A\"]）"
            )
        first, second = str(pair[0]), str(pair[1])
        if first == second:
            return (
                f"comparisons[{index}] 的两臂相同（{first}）："
                "自比较没有信息量"
            )
        unknown = [
            arm
            for arm in (first, second)
            if arm not in Config.ALL_MODES
        ]
        if unknown:
            return (
                f"comparisons[{index}] 含未知模式: "
                + ", ".join(unknown)
            )
        canonical = tuple(sorted((first, second)))
        if canonical in seen_pairs:
            return f"comparisons 存在重复对比: {first} vs {second}"
        seen_pairs.add(canonical)
        declared_modes.update((first, second))
    if str(baseline_mode) not in declared_modes:
        return (
            f"baseline_mode={baseline_mode} 没有出现在 "
            "comparisons 中：预注册的对照臂必须参与至少一个"
            "声明的比较"
        )

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        return "seeds 必须是非空的整数列表"
    normalized_seeds = []
    for index, seed in enumerate(seeds):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > MAX_SEED
        ):
            # 与 seed_error/set_seed 使用同一个合法域，避免
            # 预注册文件通过 schema 后到 NumPy 播种才抛错。
            return (
                f"seeds[{index}] 必须是 [0, {MAX_SEED}] 内的"
                "整数"
            )
        normalized_seeds.append(seed)
    if len(set(normalized_seeds)) != len(normalized_seeds):
        return "seeds 存在重复随机种子"
    if len(normalized_seeds) < _MIN_SEEDS_FOR_STATS:
        return (
            f"seeds 至少需要 {_MIN_SEEDS_FOR_STATS} 个独立种子"
            "才能做显著性检验；预注册研究不能把单次运行当作"
            "可推断的重复实验"
        )

    # 先验功效：种子数 ≥3 只保证 t 检验在数学上可算，不保证
    # 能检出预注册的最小效应。power 块把「预期种子间波动」与
    # 「最小有意义效应」一起固定，训练前就能发现「3 个种子
    # 对 1% 效应几乎没有功效」这类注定不确定的设计。
    power = payload.get("power", _PREREGISTRATION_MISSING)
    if power is not _PREREGISTRATION_MISSING:
        if not isinstance(power, dict):
            return "power 必须是对象"
        unknown_power = sorted(
            set(power) - set(_POWER_KEYS)
        )
        if unknown_power:
            return "power 含未知字段: " + ", ".join(
                unknown_power
            )
        for key in (
            "expected_seed_sd_relative",
            "target_power",
            "alpha",
        ):
            if key not in power:
                return f"power 缺少必需字段: {key}"
        error = _preregistration_finite_number(
            power["expected_seed_sd_relative"],
            "power.expected_seed_sd_relative",
            minimum=0.0,
        )
        if error is not None:
            return error
        if float(power["expected_seed_sd_relative"]) <= 0.0:
            return (
                "power.expected_seed_sd_relative 必须是正数"
                "（相对基线均值的种子间标准差）"
            )
        error = _preregistration_finite_number(
            power["target_power"],
            "power.target_power",
            minimum=0.0,
            maximum=1.0,
        )
        if error is not None:
            return error
        if not 0.0 < float(power["target_power"]) < 1.0:
            return "power.target_power 必须在 (0, 1) 内"
        error = _preregistration_finite_number(
            power["alpha"],
            "power.alpha",
            minimum=0.0,
            maximum=0.5,
        )
        if error is not None:
            return error
        if not 0.0 < float(power["alpha"]) < 0.5:
            return "power.alpha 必须在 (0, 0.5) 内"
        planned_comparisons = power.get(
            "planned_comparisons", len(comparisons)
        )
        if (
            isinstance(planned_comparisons, bool)
            or not isinstance(planned_comparisons, int)
            or planned_comparisons < 1
        ):
            return "power.planned_comparisons 必须是正整数"
        if int(planned_comparisons) < len(comparisons):
            return (
                "power.planned_comparisons 小于 comparisons 数量："
                "家族规模被低估会让功效与多重比较校正过于乐观"
            )
        attrition = power.get(
            "expected_attrition_rate", 0.0
        )
        error = _preregistration_finite_number(
            attrition,
            "power.expected_attrition_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if error is not None:
            return error
        if not 0.0 <= float(attrition) < 1.0:
            return (
                "power.expected_attrition_rate 必须在 [0, 1) 内"
                "（预期因数值失败/数据问题而不可用的运行比例；"
                "1 表示全部运行都会脱落，无法设计）"
            )
        pilot_calibration = power.get(
            "pilot_calibration", "point"
        )
        if pilot_calibration not in _PILOT_CALIBRATION_POLICIES:
            return (
                "power.pilot_calibration 必须是 "
                + " / ".join(_PILOT_CALIBRATION_POLICIES)
                + " 之一"
            )
        if "pilot_confidence" in power:
            error = _preregistration_finite_number(
                power["pilot_confidence"],
                "power.pilot_confidence",
                minimum=0.0,
                maximum=1.0,
            )
            if error is not None:
                return error
            if not 0.5 < float(power["pilot_confidence"]) < 1.0:
                return (
                    "power.pilot_confidence 必须在 (0.5, 1) 内"
                )
        plan_equivalence = power.get(
            "plan_equivalence", False
        )
        if not isinstance(plan_equivalence, bool):
            return (
                "power.plan_equivalence 必须是布尔值"
            )
        if plan_equivalence:
            declared_margin = payload.get(
                "equivalence_margin",
                _PREREGISTRATION_MISSING,
            )
            if declared_margin is _PREREGISTRATION_MISSING:
                return (
                    "power.plan_equivalence=true 时必须在预注册"
                    "顶层声明 equivalence_margin：等价界越宽越"
                    "容易主张「无实质差异」，不能用 "
                    "--equivalence-margin 的命令行默认值代替"
                    "训练前声明"
                )
            error = _preregistration_finite_number(
                declared_margin,
                "equivalence_margin",
                minimum=0.0,
                maximum=1.0,
            )
            if error is not None:
                return error
            if float(declared_margin) <= 0.0:
                return "equivalence_margin 必须是正数"
        acknowledged = power.get(
            "underpowered_acknowledged", False
        )
        if not isinstance(acknowledged, bool):
            return (
                "power.underpowered_acknowledged 必须是布尔值"
            )
        justification = power.get("justification")
        if justification is not None and not isinstance(
            justification, str
        ):
            return "power.justification 必须是字符串"
        if acknowledged and not str(justification or "").strip():
            return (
                "power.underpowered_acknowledged=true 时必须提供"
                " non-empty justification，说明为何仍要在功效"
                "不足的设计上运行"
            )
        planned_contrasts = power.get(
            "planned_contrasts", ["pairwise"]
        )
        contrast_error = _contrast_types_error(planned_contrasts)
        if contrast_error is not None:
            return contrast_error
        plan = preregistration_power_plan(
            power,
            float(min_effect),
            len(normalized_seeds),
            len(comparisons),
            contrast_types=_normalize_contrast_types(
                planned_contrasts
            ),
            equivalence_margin=(
                None
                if payload.get(
                    "equivalence_margin",
                    _PREREGISTRATION_MISSING,
                )
                is _PREREGISTRATION_MISSING
                else payload.get("equivalence_margin")
            ),
        )
        if (
            not plan["underpowered_acknowledged"]
            and plan.get("plan_equivalence")
            and not plan.get("equivalence_adequate")
        ):
            equivalence = plan.get("equivalence") or {}
            eq_required = equivalence.get(
                "required_seeds_with_attrition"
            )
            eq_margin_needed = equivalence.get(
                "margin_for_target_power"
            )
            eq_required_text = (
                f"至少需要 {eq_required} 个启动种子"
                if eq_required is not None
                else "无法在 128 个可用种子内达到目标功效"
            )
            return (
                "预注册 power 分析显示等价性（TOST）功效不足："
                "在等价界 ±"
                f"{float(equivalence.get('margin_relative') or 0.0):g}、"
                "真实差异为零、目标功效 "
                f"{plan['target_power']:g}、家族规模 "
                f"{plan['planned_comparisons']} 下，"
                f"{eq_required_text}，计划只有 "
                f"{plan['planned_seeds']} 个（计划可用 "
                f"{plan['planned_usable_seeds']} 个）。"
                + (
                    f"当前精度在目标功效下只能支持约 ±"
                    f"{float(eq_margin_needed):g} 的等价界；"
                    if eq_margin_needed is not None
                    else ""
                )
                + "等价性主张不能靠事后再放宽界来「制造」："
                "请增加 seeds，或把 plan_equivalence 设为 false "
                "并只把 TOST 作为探索性结果；确需在不足设计中"
                "运行，请显式设置 underpowered_acknowledged=true "
                "并写明 justification"
            )
        if not plan["adequate"] and not plan["underpowered_acknowledged"]:
            required_text = (
                "无法在 128 个种子内达到目标功效"
                if plan["required_seeds_for_t"] is None
                else f"至少需要 {plan['required_seeds_with_attrition']} 个启动种子"
            )
            attrition_text = (
                ""
                if not plan["expected_attrition_rate"]
                else (
                    f"（已按预期脱落率 "
                    f"{plan['expected_attrition_rate']:.0%} 折算，"
                    f"计划可用 {plan['planned_usable_seeds']} 个）"
                )
            )
            return (
                "预注册 power 分析显示当前种子数 "
                f"{len(normalized_seeds)} 不足：在预期种子间"
                f"标准差 {plan['expected_seed_sd_relative']:g}、"
                f"最小效应 {plan['min_effect_relative']:g}、"
                f"家族规模 {plan['planned_comparisons']}、目标功效 "
                f"{plan['target_power']:g} 下，{required_text}"
                f"{attrition_text}"
                "（最吃种子的对比类型 "
                f"{plan['binding_contrast']}"
                f"（{plan['binding_contrast_label']}），配对 t 口径 "
                f"{plan['required_seeds_for_t']}，稳健符号翻转口径 "
                f"{plan['required_seeds_for_robust_sign_flip']}）。"
                "主效应方差是两两对比的 0.5 倍、交互是 2 倍；"
                "若计划包含消融交互，请按 interaction 的所需种子"
                "数准备，或增加 seeds。也可以显式设置 "
                "underpowered_acknowledged=true 并写明 justification"
            )

    run_order = payload.get("run_order", _PREREGISTRATION_MISSING)
    if run_order is not _PREREGISTRATION_MISSING:
        if not isinstance(run_order, dict):
            return "run_order 必须是对象"
        unknown_order = sorted(
            set(run_order) - set(_RUN_ORDER_KEYS)
        )
        if unknown_order:
            return "run_order 含未知字段: " + ", ".join(
                unknown_order
            )
        if run_order.get("scheme") not in _RUN_ORDER_SCHEMES:
            return (
                "run_order.scheme 必须是 "
                + " / ".join(_RUN_ORDER_SCHEMES)
            )
        order_seed = run_order.get("seed")
        if (
            isinstance(order_seed, bool)
            or not isinstance(order_seed, int)
            or order_seed < 0
            or order_seed > MAX_SEED
        ):
            return (
                "run_order.seed 必须是 [0, "
                f"{MAX_SEED}] 内的整数（随机化顺序的可复现"
                "种子）"
            )

    budget_match = payload.get("budget_match")
    if budget_match is not None and (
        not isinstance(budget_match, str)
        or budget_match not in Config.MODEL_MODES
    ):
        return (
            "budget_match 必须是 null 或模型模式之一: "
            + ", ".join(sorted(Config.MODEL_MODES))
        )
    # A/B/local_base 与 smelt 的预算并不天然相同；预注册若声明
    # 这类比较却不给 budget_match，就会在分析期把「机制收益」
    # 与「多算」混在一起。base/smelt 的循环次数不同，不能套用
    # 同一个预算门，因此只对延迟合并相关模式硬性要求。
    if budget_match is None:
        for first, second in comparisons:
            arms = {str(first), str(second)}
            if (
                "smelt" in arms
                and arms & {"A", "B", "local_base"}
            ):
                return (
                    "comparisons 声明了延迟合并与 smelt 的对照，"
                    "但 budget_match=null：必须用完整计算图预算门"
                    "（如 budget_match=smelt）报告等计算量口径"
                )

    equivalence_margin = payload.get(
        "equivalence_margin", _PREREGISTRATION_MISSING
    )
    if equivalence_margin is not _PREREGISTRATION_MISSING:
        error = _preregistration_finite_number(
            equivalence_margin,
            "equivalence_margin",
            minimum=0.0,
            maximum=1.0,
        )
        if error is not None:
            return error
        if float(equivalence_margin) <= 0.0:  # type: ignore[arg-type]
            return "equivalence_margin 必须是正数"
        # 等价界是「视为可忽略的最大差异」，min_effect 是「最小
        # 有意义差异」。若前者大于后者，就存在一个既被本研究
        # 称为有意义、又被等价性主张称为可忽略的效应区间，两个
        # 结论会互相矛盾。边界相等可以接受（只是边界点重合）。
        if float(equivalence_margin) > float(min_effect):  # type: ignore[arg-type]
            return (
                "equivalence_margin 大于 min_effect：等价界"
                f"（{float(equivalence_margin):g}）比本研究声明的"
                f"最小有意义效应（{float(min_effect):g}）还宽，"
                "会存在同时被判定为「有意义」和「可忽略」的效应"
                "区间。等价界应小于或等于 min_effect，并在训练前"
                "固定；不能为了提高 TOST 功效而放宽到超过"
                "min_effect"
            )

    steps = payload.get("steps", _PREREGISTRATION_MISSING)
    if steps is not _PREREGISTRATION_MISSING:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
        ):
            return "steps 必须是正整数"
        if int(steps) < int(common_step):
            return (
                "steps 小于 common_step：训练预算内取不到"
                "预注册的共同步数指标"
            )

    frozen_script = payload.get(
        "frozen_script_sha256", _PREREGISTRATION_MISSING
    )
    if (
        frozen_script is not _PREREGISTRATION_MISSING
        and not _preregistration_sha256_text(frozen_script)
    ):
        return "frozen_script_sha256 必须是 64 位十六进制 SHA-256"

    data = payload.get("data")
    if not isinstance(data, dict):
        return "data 必须是对象"
    missing_data = [
        key
        for key in _PREREGISTRATION_DATA_REQUIRED_FIELDS
        if key not in data
    ]
    if missing_data:
        return "data 缺少必需字段: " + ", ".join(missing_data)
    unknown_data = sorted(
        set(data)
        - set(_PREREGISTRATION_DATA_REQUIRED_FIELDS)
        - set(_PREREGISTRATION_DATA_OPTIONAL_FIELDS)
    )
    if unknown_data:
        return (
            "data 含未知字段（拼写错误会让协议静默失效）: "
            + ", ".join(unknown_data)
        )
    if not _preregistration_sha256_text(data.get("sha256")):
        return "data.sha256 必须是 64 位十六进制 SHA-256"
    if data.get("split_policy") not in ("contiguous", "chronological"):
        return "data.split_policy 必须是 contiguous 或 chronological"
    for key in ("val_fraction", "test_fraction"):
        error = _preregistration_finite_number(
            data.get(key),
            f"data.{key}",
            minimum=0.0,
            maximum=1.0,
        )
        if error is not None:
            return error
    if (
        float(data["val_fraction"])  # type: ignore[arg-type]
        + float(data["test_fraction"])  # type: ignore[arg-type]
        >= 1.0
    ):
        return "data.val_fraction + data.test_fraction 必须小于 1"

    group_separator = data.get(
        "group_separator", _PREREGISTRATION_MISSING
    )
    if (
        group_separator is not _PREREGISTRATION_MISSING
        and group_separator is not None
        and not isinstance(group_separator, str)
    ):
        return "data.group_separator 必须是字符串或 null"
    if isinstance(group_separator, str):
        try:
            parse_group_separator(group_separator)
        except ValueError as error:
            return f"data.group_separator 无效: {error}"

    cleaning = data.get("cleaning", _PREREGISTRATION_MISSING)
    # 可选的 cleaning 允许显式 null（等价于未声明/全部关闭），
    # 与 preregistration_data_error 的「None 视为未声明」以及
    # group_separator: null 的口径保持一致；否则同一份文件会
    # 被 schema 拒绝却被数据核对接受。
    if (
        cleaning is not _PREREGISTRATION_MISSING
        and cleaning is not None
    ):
        if not isinstance(cleaning, dict):
            return "data.cleaning 必须是对象或 null"
        unknown = sorted(
            set(cleaning) - set(_PREREGISTRATION_CLEANING_KEYS)
        )
        if unknown:
            return "data.cleaning 含未知开关: " + ", ".join(unknown)
        for key, value in cleaning.items():
            if key == "near_duplicate_threshold":
                try:
                    threshold = float(value)
                except (TypeError, ValueError, OverflowError):
                    threshold = None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or threshold is None
                    or not math.isfinite(threshold)
                    or not 0.0 < threshold <= 1.0
                ):
                    return (
                        "data.cleaning.near_duplicate_threshold "
                        "必须是 (0, 1] 内的有限数"
                    )
                continue
            if key == "near_duplicate_shingle_bytes":
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1
                    <= value
                    <= _NEAR_DUP_MAX_SHINGLE_BYTES
                ):
                    return (
                        "data.cleaning.near_duplicate_shingle_bytes "
                        "必须是 1.."
                        f"{_NEAR_DUP_MAX_SHINGLE_BYTES} 的整数"
                    )
                continue
            if not isinstance(value, bool):
                return f"data.cleaning.{key} 必须是布尔值"
        declared_alias_values = {
            bool(cleaning[key])
            for key in _CLEAN_AGAINST_TEST_ALIASES
            if key in cleaning
        }
        if len(declared_alias_values) > 1:
            return (
                "data.cleaning.clean_train_against_test 与 "
                "data.cleaning.allow_eval_content_for_cleaning "
                "是同义开关，不能给出相反的取值"
            )

    for key in (
        "entity_manifest_sha256",
        "time_order_manifest_sha256",
    ):
        # 显式 null 与缺省同义（数据核对也只把非 None 值当作
        # 已声明清单），不能让两个校验器给出相反结论。
        if (
            key in data
            and data[key] is not None
            and not _preregistration_sha256_text(data[key])
        ):
            return f"data.{key} 必须是 64 位十六进制 SHA-256"

    return None



def load_preregistration(
    path: object,
) -> Tuple[Dict[str, object], str]:
    """
    读取并校验预注册 JSON，返回 ``(payload, file_sha256)``。

    - 拒绝重复键：JSON 中同名键以后一个为准，会让「文件里
      写了什么」产生歧义；
    - 拒绝 NaN/Infinity 常量：它们不是合法 JSON，且会污染
      后续数值比较；
    - 结构校验失败抛 ValueError，由入口转成参数错误。
    """
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as error:
        raise ValueError(f"无法读取 {file_path}: {error}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{file_path} 不是 UTF-8 文本: {error}"
        ) from error

    def reject_constant(value: str) -> object:
        raise ValueError(f"不允许非有限数值常量 {value}")

    def no_duplicate_keys(pairs):
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"存在重复键 {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=no_duplicate_keys,
        )
    except ValueError as error:
        raise ValueError(
            f"{file_path} 不是合法 JSON: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} 必须是 JSON 对象")
    schema_error = preregistration_schema_error(payload)
    if schema_error is not None:
        raise ValueError(
            f"{file_path} 预注册无效: {schema_error}"
        )
    return payload, hashlib.sha256(raw).hexdigest()



def preregistration_frozen_commit_status(
    frozen_commit: str,
) -> Optional[bool]:
    """
    当前 HEAD 是否仍包含冻结提交的代码。

    预注册文件本身通常要在记录 ``frozen_commit`` 之后提交，
    因此 HEAD 必然比冻结提交多一个（或多个只改预注册/文档的）
    提交；要求二者完全相等会让协议无法执行。这里允许
    ``frozen_commit`` 是 HEAD 的祖先，但要求
    ``experiment.py`` 从冻结提交到 HEAD 逐字节未变。任何代码
    变更都返回 False；git 不可用/提交不存在时返回 None。
    """
    head = _git_output("rev-parse", "HEAD")
    if not isinstance(head, str) or not head:
        return None
    frozen = frozen_commit.strip()
    if (
        head.lower() == frozen.lower()
        or head.lower().startswith(frozen.lower())
        or frozen.lower().startswith(head.lower())
    ):
        return True
    repository = str(Path(__file__).resolve().parent)
    try:
        ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                frozen,
                "HEAD",
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ancestor.returncode != 0:
            return False
        unchanged = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                frozen,
                "HEAD",
                "--",
                "experiment.py",
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return unchanged.returncode == 0



def preregistration_apply(
    args: argparse.Namespace,
    payload: Dict[str, object],
    *,
    current_commit: Optional[str] = None,
    working_tree_dirty: Optional[bool] = None,
    current_script_sha256: Optional[str] = None,
    frozen_code_matches: Optional[bool] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    把预注册文件落到本次运行的命令行参数上。

    规则：
    - 命令行为空（None）的字段由文件补齐，并记录
      ``applied_fields``；
    - 命令行显式给出且与文件不同则拒绝启动——预注册不能在
      看到数据后按当前命令「就高」；
    - ``--seed`` 允许只跑预注册中的一个臂/一个种子；``--seeds``
      必须与预注册种子集合完全一致；二者都未给出时按预注册
      种子集合运行完整套件；
    - 冻结提交、工作区脏状态与脚本指纹不一致时拒绝启动。

    返回 ``(applied_fields, error)``。
    """
    applied: List[str] = []

    def assign(
        attr: str,
        key: str,
        expected: object,
        *,
        numeric: bool = False,
        respect_explicit: bool = False,
    ) -> Optional[str]:
        current = getattr(args, attr, None)
        if respect_explicit and not _argv_has_flag(
            "--" + attr.replace("_", "-")
        ):
            current = None
        if current is None:
            setattr(args, attr, expected)
            applied.append(key)
            return None
        if numeric:
            try:
                # 不能用 float 比较：2**53 以上的整数会因浮点
                # 舍入而把不同取值判成相同，从而静默接受与预
                # 注册不一致的 common_step/steps。稳定键保留
                # 整数精度，同时维持 int/float 同值等价。
                same = (
                    _stable_factor_key(current)
                    == _stable_factor_key(expected)
                )
            except (TypeError, ValueError, OverflowError):
                same = False
        else:
            same = current == expected
        if same:
            return None
        return (
            f"命令行 {attr}={current!r} 与预注册文件 {key}="
            f"{expected!r} 不一致：预注册字段不能在训练开始前"
            "被命令行覆盖；如需新研究阶段，请重新冻结并写新的"
            "预注册文件"
        )

    # 代码版本先核对：之后的字段一致性都建立在「同一份代码」
    # 之上。没有 git 时退回脚本指纹；两者都无法核对则拒绝，
    # 而不是默认通过。
    frozen_commit = str(payload["frozen_commit"])
    if current_commit:
        same_commit = (
            current_commit.lower() == frozen_commit.lower()
            or current_commit.lower().startswith(
                frozen_commit.lower()
            )
            or frozen_commit.lower().startswith(
                current_commit.lower()
            )
        )
        if not same_commit and frozen_code_matches is not True:
            return applied, (
                f"当前 git commit {current_commit[:12]}... 与预注册"
                f" frozen_commit {frozen_commit[:12]}... 不一致，"
                "且无法确认冻结提交是当前 HEAD 的祖先并保持 "
                "experiment.py 未变：确证实验必须在冻结的代码"
                "版本上运行（仅允许在冻结提交之后增加预注册/"
                "文档提交）"
            )
    else:
        expected_script = payload.get("frozen_script_sha256")
        if not expected_script:
            return applied, (
                "无法读取当前 git commit，且预注册文件没有 "
                "frozen_script_sha256：不能核对代码版本，"
                "拒绝把未冻结的运行当作确证实验"
            )
        if (
            not current_script_sha256
            or str(expected_script).lower()
            != str(current_script_sha256).lower()
        ):
            return applied, (
                "当前 experiment.py 的 SHA-256 与预注册 "
                "frozen_script_sha256 不一致：代码在冻结后"
                "被修改"
            )
    if working_tree_dirty:
        return applied, (
            "工作区存在未提交修改：预注册要求冻结代码，"
            "git status --porcelain 必须为空；请提交/还原"
            "改动后重跑，而不是把脏工作区当作冻结版本"
        )
    expected_script = payload.get("frozen_script_sha256")
    if expected_script is not None:
        if current_script_sha256 is None:
            return applied, (
                "预注册声明了 frozen_script_sha256，但无法读取"
                "当前 experiment.py 的 SHA-256：不能把未核对的"
                "脚本当作冻结版本"
            )
        if (
            str(expected_script).lower()
            != str(current_script_sha256).lower()
        ):
            return applied, (
                "当前 experiment.py 的 SHA-256 与预注册 "
                "frozen_script_sha256 不一致：同一 commit 的"
                "脚本内容也必须是冻结内容"
            )

    for attr, key in (
        ("primary_metric", "primary_metric"),
        ("hypothesis_direction", "direction"),
        ("hypothesis_min_effect", "min_effect"),
        ("hypothesis_baseline", "baseline_mode"),
        ("common_step", "common_step"),
        ("budget_match", "budget_match"),
    ):
        if key == "budget_match" and payload[key] is None:
            continue
        error = assign(
            attr,
            key,
            payload[key],
            numeric=key in ("min_effect", "common_step"),
        )
        if error is not None:
            return applied, error

    # 等价界有非 None 默认值：只有命令行未显式给出时才允许
    # 用预注册值覆盖，否则默认值会把文件里的界误判成冲突。
    if "equivalence_margin" in payload:
        error = assign(
            "equivalence_margin",
            "equivalence_margin",
            payload["equivalence_margin"],
            numeric=True,
            respect_explicit=True,
        )
        if error is not None:
            return applied, error

    if "steps" in payload:
        # --steps 的 argparse 默认值非 None；只有命令行显式给出
        # 时才与预注册比较，否则文件应能补齐训练预算。
        error = assign(
            "steps",
            "steps",
            payload["steps"],
            numeric=True,
            respect_explicit=True,
        )
        if error is not None:
            return applied, error

    # 运行顺序也是预注册设计的一部分：同一份 study 若在训练前
    # 声明了随机化完整区组，就不能在执行时改回按臂分组（后者
    # 会让机器漂移与处理效应混杂）。
    if "run_order" in payload:
        run_order = payload["run_order"]
        for attr, key in (
            ("run_order", "scheme"),
            ("run_order_seed", "seed"),
        ):
            error = assign(
                attr,
                f"run_order.{key}",
                run_order[key],  # type: ignore[index]
                numeric=(key == "seed"),
                respect_explicit=True,
            )
            if error is not None:
                return applied, error

    common_step = int(payload["common_step"])
    if int(getattr(args, "steps", 0) or 0) < common_step:
        return applied, (
            f"--steps={getattr(args, 'steps', None)} 小于预注册 "
            f"common_step={common_step}：训练结束前取不到该步的"
            "指标，预注册的主分析无法执行"
        )

    # 计划运行的臂必须属于预注册声明的对照；否则这不是同一
    # 项研究，而是一个借着预注册文件运行的新实验。套件要按
    # 将要实际展开的模式集合核对，而不是只看 --mode 默认值：
    # 否则 `--suite-modes A`（默认 mode=B 不在该 study）会被
    # 误拒。
    if (
        not getattr(args, "statistical_baseline", False)
        and getattr(args, "resume", None) is None
    ):
        declared_modes = {
            str(arm)
            for pair in payload["comparisons"]  # type: ignore[union-attr]
            for arm in pair
        }
        planned_modes: set = set()
        if bool(getattr(args, "suite_ablation", False)):
            planned_modes.add("local_base")
            planned_modes.update(
                str(mode)
                for mode in (
                    getattr(args, "ablation_modes", None)
                    or _DELAYED_MERGE_MODES
                )
            )
        elif getattr(args, "suite_modes", None):
            planned_modes.update(
                str(mode) for mode in args.suite_modes
            )
        else:
            planned_modes.add(str(getattr(args, "mode", "")))
        undeclared = sorted(planned_modes - declared_modes)
        if undeclared:
            return applied, (
                "计划运行的臂 "
                + ", ".join(undeclared)
                + " 不在预注册 comparisons 声明的模式中："
                "该运行不是预注册研究的一部分"
            )
    # resume 的实际模式来自检查点，入口看到的只是 --mode 的
    # argparse 默认值（B）；必须等恢复后再核对，否则原运行是
    # base/smelt 的合法续跑会被默认 B 误拒。

    preregistered_seeds = [
        int(seed)
        for seed in payload["seeds"]  # type: ignore[union-attr]
    ]
    current_seeds = getattr(args, "seeds", None)
    if current_seeds is not None:
        if set(int(seed) for seed in current_seeds) != set(
            preregistered_seeds
        ):
            return applied, (
                f"--seeds={list(current_seeds)} 与预注册 seeds="
                f"{preregistered_seeds} 不一致：确证套件的种子"
                "集合必须在训练前固定，不能事后增删"
            )
    elif getattr(args, "resume", None) is not None:
        # 续跑的真实 seed 在检查点里；入口此时看到的只是
        # argparse 默认值，不能拿它做预注册核对。等 resume
        # 恢复 seed 后再核对（见
        # _preregistration_restored_seed_error），否则原运行
        # seed 不是默认值时合法续跑会被误拒。
        pass
    elif (
        _argv_has_flag("--seed")
        or bool(getattr(args, "evaluate_test", False))
    ):
        # 最终 test 解封是单条轨迹，不能展开成多种子套件；
        # 它只能使用预注册种子之一。
        if int(getattr(args, "seed", -1)) not in preregistered_seeds:
            return applied, (
                f"--seed={getattr(args, 'seed', None)} 不在预注册"
                f" seeds={preregistered_seeds} 中：单次运行只能"
                "是预注册种子之一"
            )
    else:
        args.seeds = list(preregistered_seeds)
        applied.append("seeds")

    return applied, None



def _preregistration_restored_seed_error(
    args: argparse.Namespace,
) -> Optional[str]:
    """
    续跑恢复真实 seed 后，再核对它是否属于预注册种子集合。

    入口阶段无法知道检查点里的 seed；若拿 argparse 默认的 42
    去核对，原运行使用 1/2/3 等合法预注册种子的续跑会被误拒。
    """
    if getattr(args, "resume", None) is None:
        return None
    if getattr(args, "seeds", None) is not None:
        return None
    record = getattr(args, "preregistration_record", None)
    if not isinstance(record, dict):
        return None
    seeds = record.get("seeds")
    if not isinstance(seeds, list):
        return None
    restored_seed = getattr(args, "seed", None)
    if isinstance(restored_seed, bool) or not isinstance(
        restored_seed, int
    ):
        return (
            f"续跑恢复的 --seed={restored_seed!r} 不是整数，"
            "无法核对预注册种子集合"
        )
    normalized = []
    for seed in seeds:
        if isinstance(seed, int) and not isinstance(seed, bool):
            normalized.append(seed)
    if restored_seed not in normalized:
        return (
            f"续跑从检查点恢复的 seed={restored_seed} 不在"
            f"预注册 seeds={normalized} 中：确证运行只能使用"
            "预注册种子之一"
        )
    return None



def _preregistration_restored_mode_error(
    args: argparse.Namespace,
) -> Optional[str]:
    """
    续跑恢复真实 mode 后，再核对它是否属于预注册 comparison 臂。

    入口阶段 --mode 只是 argparse 默认值 B；拿它核对会把
    base/smelt 的合法续跑误判成预注册外的新实验。
    """
    if getattr(args, "resume", None) is None:
        return None
    if getattr(args, "seeds", None) is not None:
        return None
    record = getattr(args, "preregistration_record", None)
    if not isinstance(record, dict):
        return None
    comparisons = record.get("comparisons")
    if not isinstance(comparisons, list):
        return None
    declared_modes: set = set()
    for pair in comparisons:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            declared_modes.update(str(arm) for arm in pair)
    if not declared_modes:
        return None
    restored_mode = getattr(args, "mode", None)
    if str(restored_mode) not in declared_modes:
        return (
            f"续跑从检查点恢复的 mode={restored_mode!r} 不在"
            "预注册 comparisons 声明的模式（"
            + ", ".join(sorted(declared_modes))
            + "）中：该续跑不属于预注册研究"
        )
    return None



def _preregistration_file_sha256(
    path: object,
) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None



def preregistration_sweep_error(
    run_paths: List[str],
    expected_file_sha256: str,
) -> Optional[str]:
    """
    核对套件中的每个子运行是否都使用了同一份预注册文件。

    预注册套件由父进程逐个子进程执行；若文件在运行中途被
    编辑，前后种子就会在不同协议下训练，但父进程看到的仍是
    当前文件。这里用每个子运行 config.json 里的
    ``preregistration.file_sha256`` 反查，发现不一致时拒绝
    把整批结果当作同一项预注册研究。
    """
    mismatched: List[str] = []
    missing: List[str] = []
    for path in run_paths:
        config_path = Path(path) / "config.json"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append(Path(path).name)
            continue
        record = (
            config.get("preregistration")
            if isinstance(config, dict)
            else None
        )
        if not isinstance(record, dict):
            missing.append(Path(path).name)
            continue
        if record.get("file_sha256") != expected_file_sha256:
            mismatched.append(
                f"{Path(path).name}"
                f"（{record.get('file_sha256')}）"
            )
    if missing:
        return (
            "以下子运行缺少预注册核对记录（config.json 不可读或"
            "不是 --preregistration 入口产生）: "
            + ", ".join(sorted(missing))
        )
    if mismatched:
        return (
            "以下子运行使用的预注册文件 SHA-256 与父进程不一致"
            f"（预期 {expected_file_sha256}）: "
            + ", ".join(sorted(mismatched))
            + "；预注册文件不能在套件运行中途被修改，"
            "请重新冻结并从全新目录重跑"
        )
    return None
