from __future__ import annotations



def _design_power_for_family(
    power: Optional[Dict[str, object]],
    family_size: int,
    contrast_types: Optional[List[str]] = None,
    *,
    force: bool = False,
) -> Optional[Dict[str, object]]:
    """把先验功效记录对齐到计划实际展开的家族规模与对比类型。

    预注册里的 ``planned_comparisons`` 可能只覆盖声明过的星形
    对照，实际套件却会展开全部臂的两两比较；``planned_contrasts``
    也可能只声明了两两对比，而消融实际还会估计主效应/交互/简单
    效应。设计计划必须按更大的实际家族与更吃种子的对比类型重新
    计算所需种子数，否则会在功效不足的设计上宣称稳健主张可行。

    ``force=True`` 用于试点证据把 ``expected_seed_sd_relative``
    或 ``expected_attrition_rate`` 上调后强制重算：此时家族规模
    与对比类型可能没有变化，但功效输入已经变了，不能沿用旧的
    ``adequate``。
    """
    if not isinstance(power, dict):
        return power
    try:
        family_size = max(1, int(family_size))
    except (TypeError, ValueError, OverflowError):
        # 损坏/手工记录的家族规模不可转换时按单对比处理，
        # 让后续重算走“缺字段”失败路径而不是抛原生异常。
        family_size = 1
    declared = power.get("planned_comparisons")
    try:
        declared_contrasts = _normalize_contrast_types(
            power.get("planned_contrasts", ["pairwise"])
        )
    except ValueError:
        declared_contrasts = ["pairwise"]
    try:
        plan_contrasts = _normalize_contrast_types(
            contrast_types or []
        )
    except ValueError:
        plan_contrasts = []
    effective_contrasts = list(declared_contrasts)
    for contrast in plan_contrasts:
        if contrast not in effective_contrasts:
            effective_contrasts.append(contrast)

    declared_is_int = (
        isinstance(declared, int)
        and not isinstance(declared, bool)
    )
    family_grew = declared_is_int and int(declared) < int(family_size)
    contrasts_grew = set(effective_contrasts) != set(
        declared_contrasts
    )
    if not family_grew and not contrasts_grew and not force:
        return power

    adjusted = dict(power)
    effective_family = (
        max(int(declared), int(family_size))
        if declared_is_int
        else int(family_size)
    )
    adjusted["planned_comparisons"] = effective_family
    adjusted["planned_contrasts"] = _normalize_contrast_types(
        effective_contrasts
    )
    min_effect = power.get("min_effect_relative")
    planned_seeds = power.get("planned_seeds")
    try:
        recomputed = preregistration_power_plan(
            adjusted,
            float(min_effect),  # type: ignore[arg-type]
            int(planned_seeds),  # type: ignore[arg-type]
            effective_family,
            contrast_types=effective_contrasts,
            equivalence_margin=adjusted.get(
                "equivalence_margin_relative"
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        # 手工构造/旧版记录可能没有重算所需的原始字段；
        # 至少保留调整后的家族规模与对比类型。但原有
        # ``adequate=True`` 是按更小的家族/更弱的对比算出来的，
        # 不能继续沿用，否则新增的交互对比会绕过启动门。
        adjusted["adequate"] = None
        adjusted["adequacy_recomputed"] = False
        adjusted["adequacy_recompute_error"] = str(error)
        # 重算失败时，所有依赖旧家族/旧对比类型/旧功效输入的
        # 派生字段都已过期。不能只清 adequate 而把旧
        # required_seeds、achieved_power、MDE 留在报告里；把它
        # 们冻结到 stale_derived_fields 供追溯，有效字段置空。
        stale_keys = (
            "contrast_power",
            "contrast_variance_factor",
            "binding_contrast",
            "binding_contrast_label",
            "required_seeds_for_t",
            "required_seeds_for_robust_sign_flip",
            "required_seeds",
            "required_seeds_with_attrition",
            "achieved_power_at_min_effect",
            "minimum_detectable_effect_relative",
            "power_sensitivity",
            "superiority_adequate",
            "equivalence_adequate",
            "equivalence",
        )
        adjusted["stale_derived_fields"] = {
            key: adjusted.get(key)
            for key in stale_keys
            if key in adjusted
        }
        for key in stale_keys:
            adjusted.pop(key, None)
        return adjusted
    recomputed["declared_planned_comparisons"] = declared
    recomputed["declared_planned_contrasts"] = declared_contrasts
    recomputed["family_size_adjusted_for_plan"] = bool(
        family_grew
    )
    recomputed["contrasts_adjusted_for_plan"] = bool(
        contrasts_grew
    )
    # 试点校准策略是预注册输入，不属于功效公式的重算结果；
    # 重算时不能把它丢掉，否则设计计划正文与 pilot_evidence
    # 会记录不一致的置信口径。
    for key in (
        "pilot_calibration",
        "pilot_confidence",
        "pilot_calibration_unusable",
    ):
        if key in power:
            recomputed[key] = power[key]
    return recomputed



def _atomic_write_json(
    path: Path,
    payload: Dict[str, object],
) -> None:
    """原子写 JSON：中途失败不留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



def write_design_plan(
    out_root: Path,
    plan: List[Dict[str, object]],
    scheme: str,
    order_seed: int,
    args: argparse.Namespace,
    *,
    kind: str = "sweep",
    planned_comparisons: Optional[int] = None,
) -> Dict[str, object]:
    """
    把「准备阶段」的设计写成可核对的 ``design_plan.json``。

    内容包含：执行顺序与区组、每个运行的身份、先验功效分析、
    种子计划与解析训练预算。``plan_id`` 是去掉时间戳后的规范化
    指纹，子进程会在开跑前核对它；看到统计结果后再改顺序/臂
    会让指纹不一致，而不是静默变成另一项研究。
    """
    out_root = Path(out_root)
    prereg = getattr(args, "preregistration_record", None)
    if not isinstance(prereg, dict):
        prereg = {}
    power = prereg.get("power")
    if not isinstance(power, dict):
        power = None
    if planned_comparisons is None:
        actual_comparisons = planned_comparisons_for_plan(plan)
        declared_comparisons = (
            int(power["planned_comparisons"])
            if power is not None
            and isinstance(power.get("planned_comparisons"), int)
            and not isinstance(
                power.get("planned_comparisons"), bool
            )
            else None
        )
        planned_comparisons = (
            max(declared_comparisons, actual_comparisons)
            if declared_comparisons is not None
            else actual_comparisons
        )
    planned_comparisons = int(planned_comparisons)
    if (
        power is not None
        and isinstance(power.get("planned_comparisons"), int)
        and not isinstance(
            power.get("planned_comparisons"), bool
        )
    ):
        planned_comparisons = max(
            planned_comparisons,
            int(power["planned_comparisons"]),
        )
    # 计划实际会估计的对比类型（消融含主效应/交互/简单效应）。
    # 先验功效必须按最吃种子的类型判定，不能只按两两对比。
    planned_contrasts = planned_contrast_types_for_plan(plan)
    pilot_evidence: Optional[Dict[str, object]] = None
    pilot_runs = getattr(args, "design_pilot_runs", None)
    if pilot_runs:
        declared_sd = None
        declared_attrition = None
        if power is not None:
            declared_sd = _finite_float(
                power.get("expected_seed_sd_relative")
            )
            declared_attrition = _finite_float(
                power.get("expected_attrition_rate")
            )
        metric = (
            getattr(args, "primary_metric", None)
            or prereg.get("primary_metric")
            or "val_loss"
        )
        # 直接调用 write_design_plan 时 args.common_step 可能尚未
        # 被 preregistration_apply 补齐；预注册文件里的 common_step
        # 同样是训练前固定的，不能因此退回数据依赖端点。
        pilot_target_step = getattr(args, "common_step", None)
        if pilot_target_step is None:
            pilot_target_step = prereg.get("common_step")
        expected_protocol = (
            _pilot_expected_comparability_protocol(
                args,
                prereg,
                str(metric),
                pilot_target_step,
            )
        )
        _, pilot_confidence = _pilot_calibration_setting(power)
        pilot_evidence = build_pilot_power_evidence(
            list(pilot_runs),
            str(metric),
            target_step=pilot_target_step,
            declared_sd=declared_sd,
            declared_attrition=declared_attrition,
            planned_modes=[
                item.get("mode")
                for item in plan
                if item.get("mode") is not None
            ],
            expected_protocol=expected_protocol,
            calibration_confidence=pilot_confidence,
        )
        comparability = _as_dict(
            pilot_evidence.get("comparability")
        )
        if comparability.get("usable_for_calibration") is False:
            mismatches = [
                item
                for item in (
                    comparability.get("blocking_mismatches") or []
                )
                if isinstance(item, dict)
            ]
            details = "; ".join(
                f"{item.get('group')}.{item.get('key')} "
                f"期望 {item.get('expected')!r}"
                for item in mismatches[:4]
            )
            raise ValueError(
                "试点运行与本次正式计划的代码/数据/训练/评估协议"
                "不可比"
                + (f"（{details}）" if details else "")
                + "：不能把其配对 sd/脱落率用于正式设计的功效"
                "校准。请改用与正式运行同代码、同数据切分、同"
                "训练预算与同评估调度完成的试点；或去掉 "
                "--design-pilot-runs，只使用预注册声明的先验值"
            )
        power, pilot_adjusted = _apply_pilot_evidence_to_power(
            power, pilot_evidence
        )
    else:
        pilot_adjusted = False
    # 内嵌的 power 记录必须与计划实际家族/对比类型同口径，否则
    # 设计文件一边按实际计划算 seed_plan，一边保留低估的
    # required_seeds。试点证据把 sd/脱落率上调后必须强制重算，
    # 即使家族规模与对比类型没有变化。
    power = _design_power_for_family(
        power,
        planned_comparisons,
        planned_contrasts,
        force=pilot_adjusted,
    )
    if pilot_adjusted and isinstance(power, dict):
        # 输入被上调不等于重算成功：旧/手工 power 记录可能缺少
        # min_effect 等重算字段，此时 _design_power_for_family
        # 会把 adequate 置空并标记 adequacy_recomputed=False。
        # 不能再宣称「已按试点观测值重算功效」，否则报告与
        # 启动门会给出错误结论。
        recompute_failed = power.get("adequacy_recomputed") is False
        power["power_recomputed_for_pilot"] = not recompute_failed
        pilot_evidence["power_recompute_ok"] = not recompute_failed
        pilot_inputs = _as_dict(
            pilot_evidence.get("adjusted_power_inputs")
        )
        if pilot_inputs.get("sd_adjusted"):
            power["pilot_declared_expected_seed_sd_relative"] = (
                pilot_inputs.get(
                    "declared_expected_seed_sd_relative"
                )
            )
            power["pilot_adjusted_expected_seed_sd_relative"] = (
                pilot_inputs.get(
                    "effective_expected_seed_sd_relative"
                )
            )
        if pilot_inputs.get("attrition_adjusted"):
            power["pilot_declared_expected_attrition_rate"] = (
                pilot_inputs.get(
                    "declared_expected_attrition_rate"
                )
            )
            power["pilot_adjusted_expected_attrition_rate"] = (
                pilot_inputs.get(
                    "effective_expected_attrition_rate"
                )
            )
        if recompute_failed:
            pilot_evidence["power_recompute_error"] = power.get(
                "adequacy_recompute_error"
            )
    # power 可能来自手工/旧记录（含 inf/NaN），_apply 也会把
    # 原始声明值复制进 pilot_evidence。落盘前统一清洗，保证
    # 返回给调用方的设计对象本身也能按标准 JSON 序列化，而不
    # 只是文件写入时被 _json_safe 兜底。
    power = _json_safe(power)
    if pilot_evidence is not None:
        pilot_evidence = _json_safe(pilot_evidence)
    seeds = [
        int(seed)
        for seed in (getattr(args, "seeds", None) or [])
    ]
    seed_plan = (
        _build_seed_plan(
            seeds,
            bool(getattr(args, "allow_underpowered", False)),
            planned_comparisons=int(planned_comparisons),
        )
        if seeds
        else None
    )
    budget = _training_budget_summary(args, len(plan))

    runs: List[Dict[str, object]] = []
    for item in plan:
        entry: Dict[str, object] = {
            "order_index": item.get("order_index"),
            "block": item.get("block"),
            "position_in_block": item.get("position_in_block"),
            "label": item.get("label"),
            "mode": item.get("mode"),
            "seed": item.get("seed"),
            "balance_coef": item.get("balance_coef"),
        }
        for key in (
            "lr",
            "dense_fnn_hidden",
            "weight_decay",
            "tuning_axis",
            "tuning_value",
        ):
            if key in item:
                entry[key] = item[key]
        runs.append(entry)

    order_balance = run_order_balance_report(plan)
    core: Dict[str, object] = {
        "schema_version": _DESIGN_PLAN_SCHEMA_VERSION,
        "kind": kind,
        "study_id": prereg.get("study_id"),
        "preregistration_file_sha256": prereg.get(
            "file_sha256"
        ),
        "order": {"scheme": scheme, "seed": int(order_seed)},
        "order_balance": order_balance,
        "planned_runs": len(runs),
        "runs": runs,
        "planned_comparisons": int(planned_comparisons),
        "planned_contrasts": list(planned_contrasts),
        "power": power,
        "seed_plan": seed_plan,
        "training_budget": budget,
    }
    if pilot_evidence is not None:
        # 只有实际提供试点运行时才写入该键：旧计划的 plan_id
        # 不含此字段，不能在重新生成时因「多了一个 null」而
        # 被误判成另一份设计、拒绝覆盖。
        core["pilot_evidence"] = pilot_evidence
    plan_id = _design_plan_id(core)
    payload = dict(core)
    payload["plan_id"] = plan_id
    payload["created_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    path = out_root / _DESIGN_PLAN_FILENAME
    if path.exists():
        try:
            existing_raw = path.read_bytes()
            existing = json.loads(
                existing_raw.decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"输出目录已有无法解析的 {path.name}：{error}；"
                "为避免覆盖冻结设计，请先归档/移走该文件"
            ) from error
        if not isinstance(existing, dict):
            raise ValueError(
                f"输出目录已有 {path.name}，但不是 JSON 对象；"
                "为避免覆盖冻结设计，请先归档/移走该文件"
            )
        existing_plan_id = existing.get("plan_id")
        if _design_plan_id(existing) != existing_plan_id:
            raise ValueError(
                f"输出目录已有的 {path.name} 内容与其 plan_id "
                "不一致：文件可能被改写，拒绝覆盖"
            )
        if str(existing_plan_id) != str(plan_id):
            raise ValueError(
                "输出目录已有另一份设计计划"
                f"（plan_id={str(existing_plan_id)[:12]}...，"
                f"本次为 {plan_id[:12]}...）：事后改变运行清单/"
                "顺序会破坏预注册设计，请使用全新的 --out 目录"
                "（或先把旧计划归档/移走）"
            )
        # 同一份设计：保留已冻结的文件字节，不因重新运行而
        # 改写 created_utc；子进程仍会用它的 SHA-256 核对。
        file_sha256 = hashlib.sha256(existing_raw).hexdigest()
    else:
        _atomic_write_json(path, payload)
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "plan_id": plan_id,
        "kind": kind,
        "order_scheme": scheme,
        "order_seed": int(order_seed),
        "order_balance": order_balance,
        "planned_runs": len(runs),
        "planned_comparisons": int(planned_comparisons),
        "planned_contrasts": list(planned_contrasts),
        "power": power,
        "pilot_evidence": pilot_evidence,
        "seed_plan": seed_plan,
        "training_budget": budget,
    }



_DESIGN_ASSIGNMENT_ENV_KEYS = (
    "SMELT_DESIGN_PLAN_FILE",
    "SMELT_DESIGN_PLAN_SHA256",
    "SMELT_DESIGN_PLAN_ID",
    "SMELT_DESIGN_ORDER_SCHEME",
    "SMELT_DESIGN_ORDER_SEED",
    "SMELT_DESIGN_ORDER_INDEX",
    "SMELT_DESIGN_BLOCK",
    "SMELT_DESIGN_BLOCK_POSITION",
)



def verify_design_assignment(
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    """
    子进程开跑前核对父进程写入的设计计划与本次分配。

    套件通过环境变量传入计划文件、指纹与 (order_index, block,
    position)。子进程重新哈希计划文件、确认自己的 label/seed/
    mode/balance 与计划条目一致后才开始训练。这样「父进程把
    计划写进报告、子进程却按另一套顺序跑」不可能静默发生。
    """
    values = {
        key: os.environ.get(key)
        for key in _DESIGN_ASSIGNMENT_ENV_KEYS
    }
    present = [key for key, value in values.items() if value]
    if not present:
        return None
    if len(present) != len(values):
        missing = [
            key for key, value in values.items() if not value
        ]
        raise ValueError(
            "设计计划环境变量不完整，缺少: "
            + ", ".join(missing)
        )

    plan_path = Path(str(values["SMELT_DESIGN_PLAN_FILE"]))
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"无法读取设计计划 {plan_path}: {error}"
        ) from error
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != values["SMELT_DESIGN_PLAN_SHA256"]:
        raise ValueError(
            "设计计划文件在设计后被修改（SHA-256 不一致）："
            "执行顺序/臂集合不能再声称是预注册的设计"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"设计计划 {plan_path} 不是合法 JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"设计计划 {plan_path} 必须是 JSON 对象")
    if payload.get("plan_id") != values["SMELT_DESIGN_PLAN_ID"]:
        raise ValueError(
            "设计计划的 plan_id 与环境变量不一致："
            "计划内容被替换或改写"
        )
    # plan_id 必须真的是计划核心内容的规范化指纹，而不是一个
    # 可以随文件一起被改写的标签。仅比对文件 SHA-256 时，若
    # 有人同时更新了环境变量里的 SHA 与 plan_id，篡改后的
    # runs/顺序仍会被接受。
    canonical_plan_id = _design_plan_id(payload)
    if canonical_plan_id != payload.get("plan_id"):
        raise ValueError(
            "设计计划内容与其 plan_id 不一致："
            "计划被改写或 plan_id 不是规范化指纹"
        )
    if canonical_plan_id != values["SMELT_DESIGN_PLAN_ID"]:
        raise ValueError(
            "设计计划的规范化 plan_id 与环境变量不一致"
        )
    order = payload.get("order")
    if not isinstance(order, dict):
        raise ValueError("设计计划缺少 order 对象")
    if order.get("scheme") != values["SMELT_DESIGN_ORDER_SCHEME"]:
        raise ValueError(
            "设计计划的运行顺序方案与环境变量不一致"
        )
    try:
        order_seed = int(
            str(values["SMELT_DESIGN_ORDER_SEED"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "设计计划环境变量中的顺序种子不是整数"
        ) from error
    plan_order_seed = order.get("seed")
    if isinstance(plan_order_seed, bool) or not isinstance(
        plan_order_seed, int
    ):
        raise ValueError(
            "设计计划中的顺序种子不是整数"
        )
    if int(plan_order_seed) != order_seed:
        raise ValueError(
            "设计计划的运行顺序种子与环境变量不一致"
        )

    try:
        order_index = int(str(values["SMELT_DESIGN_ORDER_INDEX"]))
        block = int(str(values["SMELT_DESIGN_BLOCK"]))
        position = int(str(values["SMELT_DESIGN_BLOCK_POSITION"]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "设计计划环境变量中的序号/区组不是整数"
        ) from error

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("设计计划缺少 runs 列表")
    entry = next(
        (
            item
            for item in runs
            if isinstance(item, dict)
            and item.get("order_index") == order_index
        ),
        None,
    )
    if entry is None:
        raise ValueError(
            f"设计计划中没有 order_index={order_index} 的运行"
        )

    out_path = Path(args.out)
    plan_root = plan_path.parent.resolve()
    if out_path.parent.resolve() != plan_root:
        raise ValueError(
            "设计计划不在本次 --out 的父目录："
            "不能用另一项研究的顺序计划启动本运行"
        )
    if entry.get("label") != out_path.name:
        raise ValueError(
            f"设计计划第 {order_index} 个运行是 "
            f"{entry.get('label')!r}，当前 --out 是 "
            f"{out_path.name!r}"
        )
    entry_seed = entry.get("seed")
    entry_block = entry.get("block")
    entry_position = entry.get("position_in_block")
    entry_order_index = entry.get("order_index")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (
            entry_seed,
            entry_block,
            entry_position,
            entry_order_index,
        )
    ):
        # 不能对 1.9 之类数值做 int() 截断：计划文件里的
        # seed/区组必须是精确整数，否则“分配不符”会被静默
        # 四舍五入成一个合法分配。
        raise ValueError(
            "设计计划条目的 order_index/seed/区组/区组内位置"
            "不是整数：计划内容损坏或被改写"
        )
    entry_seed = int(entry_seed)
    entry_block = int(entry_block)
    entry_position = int(entry_position)
    entry_order_index = int(entry_order_index)
    if entry_seed != int(args.seed):
        raise ValueError(
            "设计计划中的 seed 与当前 --seed 不一致"
        )
    if entry.get("mode") is not None and str(
        entry.get("mode")
    ) != str(args.mode):
        raise ValueError(
            "设计计划中的 mode 与当前 --mode 不一致"
        )
    recorded_balance = entry.get("balance_coef")
    if recorded_balance is not None:
        if isinstance(recorded_balance, bool) or not isinstance(
            recorded_balance, (int, float)
        ):
            same_balance = False
        else:
            try:
                same_balance = (
                    math.isfinite(float(recorded_balance))
                    and math.isfinite(float(args.balance_coef))
                    and abs(
                        float(recorded_balance)
                        - float(args.balance_coef)
                    ) <= 1e-12
                )
            except (TypeError, ValueError, OverflowError):
                same_balance = False
        if not same_balance:
            raise ValueError(
                "设计计划中的 balance_coef 与当前命令行不一致"
            )
    if entry_block != block:
        raise ValueError(
            "设计计划中的区组与当前分配不一致"
        )
    if entry_position != position:
        raise ValueError(
            "设计计划中的区组内位置与当前分配不一致"
        )

    return {
        "plan_file": str(plan_path),
        "plan_file_sha256": file_sha256,
        "plan_id": payload.get("plan_id"),
        "study_id": payload.get("study_id"),
        "kind": payload.get("kind"),
        "order_scheme": order.get("scheme"),
        "order_seed": order_seed,
        "order_index": order_index,
        "block": block,
        "position_in_block": position,
        "label": entry.get("label"),
    }



def design_plan_consistency_error(
    run_paths: List[str],
    plan_id: str,
) -> Optional[str]:
    """核对全部子运行记录的设计计划指纹是否与父进程一致。"""
    missing: List[str] = []
    mismatched: List[str] = []
    for path in run_paths:
        config_path = Path(path) / "config.json"
        try:
            with open(config_path, encoding="utf-8") as handle:
                config_payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append(Path(path).name)
            continue
        assignment = (
            config_payload.get("design_assignment")
            if isinstance(config_payload, dict)
            else None
        )
        recorded = (
            assignment.get("plan_id")
            if isinstance(assignment, dict)
            else None
        )
        if recorded is None:
            missing.append(Path(path).name)
        elif str(recorded) != str(plan_id):
            mismatched.append(Path(path).name)
    if not missing and not mismatched:
        return None
    details = []
    if missing:
        details.append(
            "缺少设计计划记录: " + ", ".join(missing)
        )
    if mismatched:
        details.append(
            "设计计划指纹不一致: " + ", ".join(mismatched)
        )
    return (
        "子运行没有按同一份设计计划执行（"
        + "；".join(details)
        + "）：不能把这些运行合并成一项研究"
    )



def _average_ranks(values: List[float]) -> List[float]:
    """平均秩（并列取平均），供 Spearman 相关使用。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[index]]
        ):
            end += 1
        average = (index + end + 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks



def _spearman_correlation(
    x: List[float],
    y: List[float],
) -> Optional[float]:
    """Spearman 秩相关；样本不足或秩方差为零时返回 None。"""
    if len(x) != len(y) or len(x) < 3:
        return None
    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    covariance = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(rank_x, rank_y)
    )
    variance_x = sum((a - mean_x) ** 2 for a in rank_x)
    variance_y = sum((b - mean_y) ** 2 for b in rank_y)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)



def _drift_block_key(value: object) -> object:
    """区组值的可哈希诊断键；损坏/超大数值也不能让报告崩溃。"""
    try:
        return _stable_factor_key(value)
    except (TypeError, ValueError, OverflowError):
        return ("repr", repr(value))



def run_order_drift_report(
    rows: List[Dict[str, object]],
    metric: str = "val_loss",
) -> Dict[str, object]:
    """
    诊断「执行时间/顺序」是否可能与处理效应混杂。

    设计上的修复是随机化完整区组；这里只做**描述性**核对：
    1. 计划指纹是否一致（不同研究不能混在一起）；
    2. 是否仍使用 grouped 顺序（模式与时间完全混杂）；
    3. 去掉模式均值后，指标与执行顺序的 Spearman 相关。
    正/负强相关提示机器漂移、缓存预热或负载变化可能在臂之间
    制造系统性差异。它不是显著性检验，也不替代随机化设计。
    """
    entries: List[Dict[str, object]] = []
    for row in rows:
        if _is_deterministic_run(row):
            continue
        index = row.get("design_order_index")
        value = row.get(metric)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or value is None
        ):
            continue
        numeric = _finite_float(value)
        if numeric is None:
            continue
        entries.append({
            "run": row.get("run"),
            "mode": str(row.get("mode")),
            "order_index": index,
            "block": row.get("design_block"),
            "position_in_block": row.get(
                "design_block_position"
            ),
            "value": numeric,
            "plan_id": row.get("design_plan_id"),
            "scheme": row.get("design_order_scheme"),
        })

    if not entries:
        return {
            "available": False,
            "verdict": "unavailable",
            "reason": (
                "运行没有设计计划/执行顺序记录（旧运行或非套件"
                "单跑），无法核对顺序漂移"
            ),
        }

    plan_ids = sorted({
        str(entry["plan_id"])
        for entry in entries
        if entry.get("plan_id")
    })
    schemes = sorted({
        str(entry["scheme"])
        for entry in entries
        if entry.get("scheme")
    })
    warnings: List[str] = []
    missing_plan_ids = [
        str(entry.get("run"))
        for entry in entries
        if not entry.get("plan_id")
    ]
    missing_schemes = [
        str(entry.get("run"))
        for entry in entries
        if not entry.get("scheme")
    ]
    if missing_plan_ids:
        warnings.append(
            "部分运行缺少设计计划指纹（plan_id）："
            + ", ".join(missing_plan_ids)
            + "；不能把这些运行与已计划运行当作同一项研究"
        )
    if missing_schemes:
        warnings.append(
            "部分运行缺少执行顺序方案（order scheme）："
            + ", ".join(missing_schemes)
            + "；无法判断时间漂移是否被区组设计平衡"
        )
    if len(plan_ids) > 1:
        warnings.append(
            "运行来自多份设计计划（plan_id 不一致）："
            "不能把不同顺序方案拼成一项研究"
        )
    if len(schemes) > 1:
        warnings.append(
            "运行混合了多种执行顺序方案（"
            + ", ".join(schemes)
            + "）：顺序效应无法用单一区组设计解释"
        )
    if "grouped" in schemes:
        warnings.append(
            "执行顺序是 grouped：同一臂的所有种子先跑完，"
            "时间/机器漂移与臂效应完全混杂；确证实验应使用"
            "randomized / counterbalanced / williams 完整区组"
        )
    if len(entries) < 3:
        return {
            "available": True,
            "verdict": "insufficient",
            "n_runs": len(entries),
            "plan_ids": plan_ids,
            "order_schemes": schemes,
            "warnings": warnings,
            "reason": "带顺序记录的运行少于 3 个，无法估计漂移",
        }

    # 去掉模式均值：臂间真实差异不应被误读成时间漂移。
    mode_means: Dict[str, List[float]] = {}
    for entry in entries:
        mode_means.setdefault(
            str(entry["mode"]), []
        ).append(float(entry["value"]))
    means = {
        mode: sum(values) / len(values)
        for mode, values in mode_means.items()
    }
    # 保留原始整数序号：Spearman 只需要排序与秩，不需要把
    # 序号转成浮点。手工改坏 config 里的超大整数转 float 会
    # 溢出，且顺序大小本身不参与数值运算。
    indices = [entry["order_index"] for entry in entries]
    residuals = [
        float(entry["value"]) - means[str(entry["mode"])]
        for entry in entries
    ]
    rho = _spearman_correlation(indices, residuals)
    p_value = None
    if rho is not None and len(entries) > 2:
        denominator = max(1e-12, 1.0 - rho * rho)
        statistic = rho * math.sqrt(
            (len(entries) - 2) / denominator
        )
        p_value = _two_sided_t_p_value(
            statistic, float(len(entries) - 2)
        )
    if (
        rho is not None
        and abs(rho) >= 0.7
        and len(entries) >= 4
    ):
        warnings.append(
            "去模式均值后指标与执行顺序的 |Spearman ρ|="
            f"{abs(rho):.2f}（p="
            + ("NA" if p_value is None else f"{p_value:.3f}")
            + "）：存在顺序/时间漂移迹象，应检查机器负载与"
            "热漂移，并用随机化区组复核"
        )

    by_mode: List[Dict[str, object]] = []
    for mode, values in mode_means.items():
        mode_entries = [
            entry for entry in entries
            if str(entry["mode"]) == mode
        ]
        mode_rho = _spearman_correlation(
            [entry["order_index"] for entry in mode_entries],
            [float(entry["value"]) for entry in mode_entries],
        )
        mode_p = None
        if mode_rho is not None and len(mode_entries) > 2:
            denominator = max(1e-12, 1.0 - mode_rho * mode_rho)
            statistic = mode_rho * math.sqrt(
                (len(mode_entries) - 2) / denominator
            )
            mode_p = _two_sided_t_p_value(
                statistic, float(len(mode_entries) - 2)
            )
        if (
            mode_rho is not None
            and abs(mode_rho) >= 0.8
            and len(mode_entries) >= 4
        ):
            warnings.append(
                f"模式 {mode} 内部指标随执行顺序单调变化"
                f"（ρ={mode_rho:.2f}）：该臂的结果可能包含"
                "时间漂移"
            )
        by_mode.append({
            "mode": mode,
            "n_runs": len(mode_entries),
            "spearman_rho": mode_rho,
            "p_value": mode_p,
        })

    if rho is None:
        # 所有运行指标相同（或执行顺序没有秩方差）时，Spearman
        # 相关没有定义；不能把它当成「没有发现顺序相关」的
        # 阳性证据。返回 insufficient，让展示层明确说无法估计。
        return {
            "available": True,
            "verdict": "insufficient",
            "n_runs": len(entries),
            "n_blocks": len({
                _drift_block_key(entry["block"])
                for entry in entries
            }),
            "plan_ids": plan_ids,
            "order_schemes": schemes,
            "detrended_spearman": None,
            "detrended_p_value": None,
            "by_mode": by_mode,
            "warnings": warnings,
            "reason": (
                "去模式均值后的指标或执行顺序没有秩方差"
                "（例如所有运行指标相同），无法估计 Spearman "
                "顺序相关；这不是「没有漂移」的证据"
            ),
            "note": (
                "这是描述性诊断，不是显著性检验；随机化完整区组"
                "只能平衡时间漂移，不能消除它。若出现强顺序相关，"
                "应检查机器/负载并重跑受影响区组"
            ),
        }

    return {
        "available": True,
        "verdict": "warn" if warnings else "ok",
        "n_runs": len(entries),
        "n_blocks": len({
            _drift_block_key(entry["block"])
            for entry in entries
        }),
        "plan_ids": plan_ids,
        "order_schemes": schemes,
        "detrended_spearman": rho,
        "detrended_p_value": p_value,
        "by_mode": by_mode,
        "warnings": warnings,
        "note": (
            "这是描述性诊断，不是显著性检验；随机化完整区组"
            "只能平衡时间漂移，不能消除它。若出现强顺序相关，"
            "应检查机器/负载并重跑受影响区组"
        ),
    }



def _sweep_child_env(seed: int) -> Dict[str, str]:
    """
    套件子进程的环境变量：显式固定 PYTHONHASHSEED。

    父进程的 PYTHONHASHSEED 若未设置，每个子进程在启动时会
    各自随机化 hash（set_seed 里的 setdefault 无法追溯地固定
    当前解释器），同一种子的两次运行会因 dict/set 迭代顺序
    不同而产生不可复现的差异。子进程按本次运行的 seed 固定该
    变量，使「同一种子 = 同一随机环境」。父进程已显式设置时
    尊重原值（所有子进程共享同一固定种子）。

    CUBLAS_WORKSPACE_CONFIG 同样在子进程中兜底设置：确定性
    cuBLAS 工作区必须在 CUDA 上下文创建前生效。
    """
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(int(seed)))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env



def _write_run_order_log(
    out_root: Path,
    design: Optional[Dict[str, object]],
    plan: List[Dict[str, object]],
    records: List[Dict[str, object]],
) -> None:
    """把计划顺序与实际开始/结束时间落盘（失败条目也保留）。"""
    payload: Dict[str, object] = {
        "planned_runs": len(plan),
        "plan_id": (
            design.get("plan_id")
            if isinstance(design, dict)
            else None
        ),
        "plan_file_sha256": (
            design.get("file_sha256")
            if isinstance(design, dict)
            else None
        ),
        "order": (
            {
                "scheme": design.get("order_scheme"),
                "seed": design.get("order_seed"),
            }
            if isinstance(design, dict)
            else None
        ),
        "runs": records,
        "updated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }
    _atomic_write_json(out_root / _RUN_ORDER_FILENAME, payload)



def _execute_sweep_plan(
    base_argv: List[str],
    plan: List[Dict[str, object]],
    out_root: Path,
    design: Optional[Dict[str, object]] = None,
) -> List[str]:
    """
    逐运行以独立子进程执行计划中的每个条目；覆盖 --seed/--out
    以及条目声明的 --mode/--balance-coef/--lr。任一运行失败即
    停止并退出（可用 --compare 分析已完成的目录）。

    每个子进程都带上按 seed 固定的 PYTHONHASHSEED 环境
    （见 ``_sweep_child_env``），避免同种子运行因 hash 随机化
    产生差异。

    给出 ``design`` 时，子进程还会收到设计计划文件、指纹与
    (order_index, block, position)。子进程开跑前会重新核对，
    防止报告里的顺序与实际执行顺序不一致；实际开始/结束时间
    写到 ``run_order.json``，供顺序漂移诊断使用。
    """
    run_paths: List[str] = []
    order_records: List[Dict[str, object]] = [
        {
            "order_index": item.get("order_index", index),
            "block": item.get("block"),
            "position_in_block": item.get("position_in_block"),
            "label": item.get("label"),
            "mode": item.get("mode"),
            "seed": item.get("seed"),
            "status": "planned",
            "started_utc": None,
            "finished_utc": None,
            "returncode": None,
        }
        for index, item in enumerate(plan, start=1)
    ]
    if design is not None:
        _write_run_order_log(out_root, design, plan, order_records)

    for index, item in enumerate(plan, start=1):
        run_out = out_root / str(item["label"])
        record = next(
            (
                entry
                for entry in order_records
                if entry["order_index"]
                == item.get("order_index", index)
            ),
            None,
        )

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
        ] + base_argv
        command += [
            "--seed",
            str(item["seed"]),
            "--out",
            str(run_out),
        ]
        if item.get("mode") is not None:
            command += ["--mode", str(item["mode"])]
        if item.get("balance_coef") is not None:
            command += [
                "--balance-coef",
                format(float(item["balance_coef"]), "g"),
            ]
        if item.get("lr") is not None:
            # repr 保证浮点值往返逐位一致（如 1e-4）。
            command += ["--lr", repr(float(item["lr"]))]
        # 多轴（坐标式）调参：容量与权重衰减轴的候选也逐条覆盖，
        # 其余候选继承命令行里的基线值。
        if item.get("dense_fnn_hidden") is not None:
            command += [
                "--dense-fnn-hidden",
                str(int(item["dense_fnn_hidden"])),
            ]
        if item.get("weight_decay") is not None:
            command += [
                "--weight-decay",
                repr(float(item["weight_decay"])),
            ]

        child_env = _sweep_child_env(int(item["seed"]))
        if design is not None:
            child_env.update({
                "SMELT_DESIGN_PLAN_FILE": str(design["path"]),
                "SMELT_DESIGN_PLAN_SHA256": str(
                    design["file_sha256"]
                ),
                "SMELT_DESIGN_PLAN_ID": str(design["plan_id"]),
                "SMELT_DESIGN_ORDER_SCHEME": str(
                    design["order_scheme"]
                ),
                "SMELT_DESIGN_ORDER_SEED": str(
                    design["order_seed"]
                ),
                "SMELT_DESIGN_ORDER_INDEX": str(
                    item.get("order_index", index)
                ),
                "SMELT_DESIGN_BLOCK": str(item.get("block")),
                "SMELT_DESIGN_BLOCK_POSITION": str(
                    item.get("position_in_block")
                ),
            })

        print(
            f"[sweep] ({index}/{len(plan)}) 运行 {item['label']}"
            + (
                f"（区组 {item.get('block')} 内第 "
                f"{item.get('position_in_block')} 个）"
                if design is not None
                else ""
            )
        )
        if record is not None:
            record["status"] = "running"
            record["started_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            if design is not None:
                _write_run_order_log(
                    out_root, design, plan, order_records
                )

        completed = subprocess.run(
            command,
            env=child_env,
        )

        if record is not None:
            record["finished_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            record["returncode"] = int(completed.returncode)
            record["status"] = (
                "completed"
                if completed.returncode == 0
                else "failed"
            )
            if design is not None:
                _write_run_order_log(
                    out_root, design, plan, order_records
                )

        if completed.returncode != 0:
            print(
                f"[sweep] 运行 {item['label']} 失败 "
                f"(exit={completed.returncode})；已停止后续"
                "运行，可用 --compare 分析已完成的目录"
            )
            raise SystemExit(1)

        run_paths.append(str(run_out))

    return run_paths



def baseline_tuning_axis_candidates_error(
    flag: str,
    values: List[object],
    integer: bool,
) -> Optional[str]:
    """
    校验多轴调参候选的取值域：容量轴必须为正整数，正则轴
    必须为有限非负数。返回错误消息或 None。
    """
    if integer:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in values
        ):
            return f"{flag} 的每个候选必须是正整数"
        return None
    for value in values:
        numeric = _finite_float(value)
        if numeric is None or numeric < 0.0:
            return f"{flag} 的每个候选必须是有限非负数"
    return None



def resolve_baseline_tuning_metric(
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """
    决定简单基线调参的目标指标与来源。

    预注册主指标（--primary-metric）优先于对比指标：若在
    val_loss 上调强基线、却用 macro F1 / PR-AUC 判定成败，
    「基线已调优」的前提在判定口径上并不成立。显式
    --baseline-tuning-metric 优先于两者。主指标是路由诊断
    （dense/ngram 基线没有该记录）时回退到 --compare-metric，
    并打印原因而不是静默换指标。

    返回 ``(metric, source)``，source 为 ``explicit`` /
    ``primary_metric`` / ``compare_metric`` / ``fallback_loss``。
    """
    explicit = getattr(args, "baseline_tuning_metric", None)
    if explicit is not None:
        return str(explicit), "explicit"

    primary = getattr(args, "primary_metric", None)
    if primary is not None:
        if primary in _BASELINE_TUNING_METRIC_KEYS:
            return str(primary), "primary_metric"
        print(
            "[baseline-tuning] 预注册主指标 "
            f"{primary} 不是简单基线可计算的指标"
            "（路由诊断需要 MoE 运行），回退到 "
            f"--compare-metric={args.compare_metric}"
        )

    compare_metric = str(args.compare_metric)
    if compare_metric not in _BASELINE_TUNING_METRIC_KEYS:
        print(
            "[baseline-tuning] 对比指标 "
            f"{compare_metric} 不是训练时产生的指标"
            "（等计算量 loss 只在分析期从验证曲线派生，"
            "路由诊断需要 MoE 记录），回退到 val_loss"
        )
        return "val_loss", "fallback_loss"

    return compare_metric, "compare_metric"
