from __future__ import annotations



def load_baseline_tuning(path: Path) -> Dict[str, object]:
    """
    读取并校验基线调参报告（--suite-baseline-tuning 的输出）。

    校验失败抛 ValueError，由入口转为明确的参数错误。
    """
    def reject_constant(value: str) -> object:
        raise ValueError(f"不允许非有限数值常量 {value}")

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(
                f, parse_constant=reject_constant
            )
    except OSError as error:
        raise ValueError(f"无法读取 {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} 不是合法 JSON: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是调参报告对象")
    if payload.get("comparison_family") != "baseline_tuning":
        raise ValueError(
            f"{path} 不是 --suite-baseline-tuning 生成的报告"
            "（缺少 comparison_family=baseline_tuning）"
        )
    if not isinstance(payload.get("mode"), str):
        raise ValueError(f"{path} 缺少 mode 字段")
    if not isinstance(payload.get("metric"), str):
        raise ValueError(f"{path} 缺少 metric 字段")
    winner = payload.get("winner")
    winner_lr = (
        _finite_float(winner.get("lr"))
        if isinstance(winner, dict)
        else None
    )
    if winner_lr is None or winner_lr <= 0.0:
        raise ValueError(
            f"{path} 没有有效 winner（winner.lr 必须是有限"
            "正数）；调参失败的报告不能作为基线来源"
        )
    winner_metric = (
        _finite_float(winner.get("metric_mean"))
        if isinstance(winner, dict)
        else None
    )
    if winner_metric is None:
        raise ValueError(
            f"{path} 的 winner 没有可用指标均值（必须是"
            "有限数值），不能作为基线来源"
        )
    return payload



def _protocol_value_text(value: object) -> str:
    """协议值的稳定文本表示，用于错误消息与逐项比较。"""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )



def baseline_tuning_provenance_error(
    payload: Dict[str, object],
    mode: str,
    lr: float,
    steps: int,
    data_sha256: Optional[str],
    batch_size: Optional[int] = None,
    protocol: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    """
    核对「当前基线训练」是否忠实复用了调参报告的 winner：
    模式、学习率、训练步数、批大小、数据指纹，以及完整的
    受控变量协议（warmup / 调度 / weight decay / 梯度裁剪 /
    数据划分 / 群组划分 / 评估窗口 / 数值环境等）都必须一致。
    返回错误消息或 None。

    学习率与 warmup、调度、weight decay 等耦合：只核对
    steps/batch/数据指纹时，在一种协议下调出 winner、再换
    协议训练「最终基线」也能借用调参证据。因此：
    - ``protocol`` 为当前训练的 comparison_protocol；
    - 调参报告缺少协议字段（旧版报告）时显式拒绝，而不是
      默认为通过；
    - 任一项与调参时不一致即拒绝，并指明具体协议项。
    """
    record_mode = str(payload.get("mode"))
    if mode != record_mode:
        return (
            f"调参报告的基线模式是 {record_mode}，当前训练是 "
            f"{mode}：该报告不能证明当前模式的基线被调过参"
        )

    winner = _as_dict(payload.get("winner"))

    # 完整调参报告必须带 axes：紧凑来源记录（config.json 里的
    # baseline_tuning）虽然也没有 axes，但它不是 --baseline-tuning-
    # from 读取的报告。多轴报告缺 axes 时不能退回只看 winner.lr
    # 的旧单轴逻辑，否则其余轴完全没有被核对。
    if (
        payload.get("multi_axis") is True
        or payload.get("design") == "coordinate"
    ) and "axes" not in payload:
        return (
            "调参报告声明为多轴设计但缺少 axes：无法逐轴核对"
            "推荐配置是否真的复用于基线训练，不能只核对学习率"
        )

    axis_error = _baseline_tuning_axis_error(payload)
    if axis_error is not None:
        return (
            axis_error
            + "。不完整的坐标式调参报告不能作为基线来源；"
            "请补齐所有轴的候选后重新调参"
        )

    # 调参轴与推荐取值：单轴报告（学习率）取 winner.lr；多轴
    # （坐标式）报告逐轴取该轴 winner——推荐配置是各轴 winner
    # 的组合，当前训练必须与它逐项一致。
    tuned_values: Dict[str, object] = {}
    axis_reports = payload.get("axes")
    if isinstance(axis_reports, list):
        for axis_report in axis_reports:
            if not isinstance(axis_report, dict):
                continue
            key = str(axis_report.get("protocol_key", ""))
            axis_winner = axis_report.get("winner")
            if not key or not isinstance(axis_winner, dict):
                continue
            if "value" in axis_winner:
                tuned_values[key] = axis_winner["value"]
    if not tuned_values:
        winner_lr = _finite_float(winner.get("lr"))
        if winner_lr is None or winner_lr <= 0.0:
            return (
                "调参报告缺少有效的 winner.lr（必须是有限正数）："
                "调参失败或损坏的报告不能作为基线来源"
            )
        tuned_values = {"lr": winner_lr}

    for key, expected in tuned_values.items():
        if key == "lr":
            current: object = lr
        else:
            current = (
                protocol.get(key)
                if isinstance(protocol, dict)
                else None
            )
        if current is None or not _axis_value_equal(
            current,
            expected,
        ):
            flag = _axis_cli_flag(key)
            if key == "lr":
                expected_lr = _finite_float(expected)
                expected_text = (
                    format(expected_lr, "g")
                    if expected_lr is not None
                    else _axis_value_text(expected)
                )
                return (
                    f"当前 --lr={format(lr, 'g')} 与调参 winner "
                    f"lr={expected_text} 不一致："
                    "调参来源必须真实复用于基线训练，不能借用"
                    "调参报告"
                )
            return (
                f"当前 {flag}={_axis_value_text(current)} 与调参"
                f"报告该轴的 winner {_axis_value_text(expected)} "
                f"不一致：{flag} 是调参轴，必须真实复用于基线"
                "训练，不能借用调参报告"
            )

    # 统计功效：调参 winner 是种子噪声的产物时，这条「调过参
    # 的基线」不可信。低功效调参（--allow-underpowered 或
    # 种子数 < 3）不能作为「基线已充分调参」的证据。
    seed_plan = payload.get("seed_plan")
    if not isinstance(seed_plan, dict):
        return (
            "调参报告缺少 seed_plan（旧版报告）：无法核对调参"
            "的独立种子数与统计功效，请用新版 "
            "--suite-baseline-tuning 重新生成调参报告"
        )
    tuning_seeds = _baseline_tuning_seed_count(seed_plan)
    if tuning_seeds is None:
        return (
            "调参报告的 seed_plan 缺少有效的 n_seeds：无法核对"
            "调参的独立种子数与统计功效，请用新版 "
            "--suite-baseline-tuning 重新生成调参报告"
        )
    if (
        seed_plan.get("underpowered_override")
        or tuning_seeds < _MIN_SEEDS_FOR_STATS
    ):
        return (
            "调参报告在少于 "
            f"{_MIN_SEEDS_FOR_STATS} 个独立随机种子的情况下"
            "选出 winner（underpowered）：winner 可能只是种子"
            "噪声的产物，不能作为「基线已充分调参」的证据；"
            "请用 3~5 个种子重新调参"
        )

    # 候选的种子集合不一致/存在伪重复时，winner 可能只是
    # 种子差异的产物。字段缺失（旧版报告）保持兼容，但显式
    # 标记为不一致时必须拒绝。
    seed_alignment = payload.get("seed_alignment")
    if (
        isinstance(seed_alignment, dict)
        and seed_alignment.get("aligned") is False
    ):
        return (
            "调参报告的候选种子集合不一致（或同一候选内存在"
            "重复种子）：winner 的均值差可能来自种子/伪重复"
            "而不是学习率，不能作为基线来源；请让所有候选"
            "使用同一组 3~5 个随机种子重新调参"
        )

    # 候选之间训练协议不一致时，调参结论与训练配置混杂，
    # 报告本身已标为不可用。
    tuning_issues = payload.get("issues")
    if isinstance(tuning_issues, list) and any(
        "训练协议不一致" in str(issue)
        for issue in tuning_issues
    ):
        return (
            "调参报告的候选之间存在训练协议不一致：调参结果与"
            "「训练更久 / 换了协议」混杂，不能作为基线来源；"
            "请修正配置后重新调参"
        )

    config = payload.get("training_config")
    if not isinstance(config, dict):
        return (
            "调参报告的 training_config 缺失或格式非法："
            "来源记录不足以核对训练协议一致"
        )

    # 旧版报告只有 total_steps / batch_size / data_sha256，
    # 缺协议字段时不能默认为通过——那等于把「调参来源」当
    # 护身符。显式要求重新生成完整报告。
    missing_keys = [
        key
        for key in _BASELINE_TUNING_PROVENANCE_KEYS
        if key not in config
        or (
            config.get(key) is None
            and isinstance(protocol, dict)
            and protocol.get(key) is not None
        )
    ]
    if missing_keys:
        return (
            "调参报告缺少受控变量协议字段（"
            + ", ".join(missing_keys)
            + "）：来源记录不足以核对当前训练与调参时的"
            "协议一致，请用新版 --suite-baseline-tuning "
            "重新生成调参报告"
        )

    if protocol is None:
        return (
            "当前训练没有 comparison_protocol 记录，无法核对"
            "训练协议与调参时一致：请用当前版本的训练入口"
            "重新运行（--baseline-tuning-from 需要协议核对）"
        )

    config = payload.get("training_config") or {}
    record_steps = config.get("total_steps")
    if record_steps is not None:
        record_steps_value = _finite_float(record_steps)
        if (
            record_steps_value is None
            or not record_steps_value.is_integer()
        ):
            return (
                "调参报告的 training_config.total_steps 不是"
                "有限整数：无法核对训练步数，拒绝把损坏的报告"
                "作为基线来源"
            )
        if int(steps) != int(record_steps_value):
            return (
                f"当前 --steps={int(steps)} 与调参训练配置 "
                f"steps={int(record_steps_value)} 不一致："
                "学习率是在原训练配置下调出来的，训练配置变了"
                "应重新调参"
            )

    record_batch = config.get("batch_size")
    if (
        batch_size is not None
        and isinstance(record_batch, int)
        and not isinstance(record_batch, bool)
        and int(batch_size) != int(record_batch)
    ):
        return (
            f"当前 --batch-size={int(batch_size)} 与调参训练"
            f"配置 batch_size={int(record_batch)} 不一致："
            "学习率与批大小耦合，训练配置变了应重新调参"
        )

    record_sha = config.get("data_sha256")
    if (
        isinstance(record_sha, str)
        and record_sha
        and data_sha256 is not None
        and record_sha != data_sha256
    ):
        return (
            "当前数据指纹与调参时的数据不一致：学习率是在"
            "原数据上调出来的，换数据后应重新调参"
        )

    # 完整协议逐项核对：warmup / lr_schedule / weight_decay /
    # grad_clip / val/test_fraction / group_separator /
    # eval_batches / amp_dtype 等任何一项不同都说明「最终基线」
    # 不是在调参时的受控条件下训练的。调参轴自身的取值已在上
    # 面按各轴 winner 核对过（推荐配置与当前训练必须一致），
    # 不再与调参时的基线取值比较。
    for key in _BASELINE_TUNING_PROVENANCE_KEYS:
        if key in tuned_values:
            continue
        record_value = config.get(key)
        current_value = protocol.get(key)
        if _protocol_value_text(
            record_value
        ) != _protocol_value_text(current_value):
            return (
                f"当前训练的协议项 {key}="
                f"{_protocol_value_text(current_value)} 与调参"
                f"训练配置 {key}="
                f"{_protocol_value_text(record_value)} 不一致："
                "学习率与 warmup/调度/weight decay/数据划分等"
                "耦合，协议变了应重新调参"
            )

    return None



def baseline_tuning_provenance_record(
    payload: Dict[str, object],
    source: str,
) -> Dict[str, object]:
    """
    生成写入 config.json 的紧凑调参来源记录（baseline_tuning
    字段），供对比阶段的基线合理性检查核对。
    """
    winner = payload.get("winner") or {}
    axis_reports = payload.get("axes")
    axis_winners: Dict[str, object] = {}
    if isinstance(axis_reports, list):
        for axis_report in axis_reports:
            if not isinstance(axis_report, dict):
                continue
            axis_winner = axis_report.get("winner")
            if not isinstance(axis_winner, dict):
                continue
            axis_winners[str(axis_report.get("name"))] = (
                axis_winner.get("value")
            )
    recommended = payload.get("recommended_config")
    return {
        "comparison_family": "baseline_tuning",
        "source": str(source),
        "mode": payload.get("mode"),
        "metric": payload.get("metric"),
        "metric_source": payload.get("metric_source"),
        "winner_lr": winner.get("lr"),
        "winner_mean": winner.get("metric_mean"),
        "n_candidates": payload.get("n_candidates"),
        "grid": payload.get("grid"),
        "training_config": payload.get("training_config"),
        # 统计功效与调参质量问题必须随来源一起落盘：低功效、
        # 种子集合不一致或有协议冲突的调参报告不能被下游当成
        # 「基线已充分调参」。
        "seed_plan": payload.get("seed_plan"),
        "seed_alignment": payload.get("seed_alignment"),
        "issues": payload.get("issues"),
        "winner_at_grid_boundary": payload.get(
            "winner_at_grid_boundary"
        ),
        "best_mean_at_grid_boundary": payload.get(
            "best_mean_at_grid_boundary"
        ),
        # 多轴（坐标式）调参：各轴 winner、组合出的推荐配置，
        # 以及「该组合是否在网格里被直接评估过」必须落盘——
        # 未评估过的组合只是各轴结论的拼接，读者据此判断
        # 「调过参」这个说法的强度。
        "design": payload.get("design", "single_axis"),
        "multi_axis": bool(payload.get("multi_axis", False)),
        "axis_winners": axis_winners,
        "recommended_config": recommended,
        "recommended_config_evaluated": payload.get(
            "recommended_config_evaluated"
        ),
    }



def run_baseline_tuning_sweep(args: argparse.Namespace) -> None:
    """
    --suite-baseline-tuning：对简单基线做预注册的学习率网格
    搜索（val 选择，test 尘封），输出调参报告。调参只是把
    基线调到「值得战胜」，报告本身不是最终基线。
    """
    metric, metric_source = resolve_baseline_tuning_metric(args)
    lrs = list(args.baseline_tuning_lrs)
    seeds = list(args.seeds)
    dense_hiddens = list(
        getattr(args, "baseline_tuning_dense_hiddens", None) or []
    )
    weight_decays = list(
        getattr(args, "baseline_tuning_weight_decays", None) or []
    )
    mode = args.baseline_tuning_mode

    # dense 的 FFN 宽度只对 dense 模式生效：非 dense 模式上传
    # 容量轴会得到一串完全相同的运行，白烧算力。
    if dense_hiddens and mode != "dense":
        print(
            "[suite-baseline-tuning] 提示：--baseline-tuning-"
            "dense-hiddens 只对 dense 模式生效（当前 mode="
            f"{mode}），该轴已跳过"
        )
        dense_hiddens = []

    # 基线配置：坐标式设计里所有候选都相对它只改一个轴。
    base_lr = float(args.lr)
    base_weight_decay = float(args.weight_decay)
    base_dense_hidden = (
        int(args.dense_fnn_hidden)
        if args.dense_fnn_hidden is not None
        else (
            int(args.top_k) * int(args.expert_hidden)
            if dense_hiddens
            else None
        )
    )

    base_argv = strip_sweep_args(sys.argv[1:])
    multi_axis = bool(dense_hiddens or weight_decays)
    plan = build_baseline_tuning_plan(
        seeds=seeds,
        mode=mode,
        lrs=lrs,
        balance_coef=args.balance_coef,
        dense_hiddens=dense_hiddens,
        weight_decays=weight_decays,
        # 单轴（旧）路径保持原样：只改学习率，其余训练配置全部
        # 继承命令行；多轴路径才需要显式的基线配置坐标。
        base_lr=base_lr if multi_axis else None,
        base_dense_hidden=(
            base_dense_hidden if multi_axis else None
        ),
        base_weight_decay=(
            base_weight_decay if multi_axis else None
        ),
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # 调参候选同样按随机化完整区组执行：候选按学习率/容量排序
    # 后依次跑完，会让「后跑的候选」同时承受更晚的时间片。
    order_scheme, order_seed = resolve_run_order(args)
    plan = order_sweep_plan(plan, order_scheme, order_seed)
    design = write_design_plan(
        out_root,
        plan,
        order_scheme,
        order_seed,
        args,
        kind="baseline_tuning",
        planned_comparisons=1,
    )
    print_design_summary(design)

    print(
        "[suite-baseline-tuning] 基线调参：mode="
        f"{mode}, 学习率候选="
        + ", ".join(format(lr, "g") for lr in lrs)
        + (
            ", dense 宽度候选="
            + ", ".join(str(value) for value in dense_hiddens)
            if dense_hiddens
            else ""
        )
        + (
            ", 权重衰减候选="
            + ", ".join(format(value, "g") for value in weight_decays)
            if weight_decays
            else ""
        )
        + f", seeds={seeds}, 指标={metric}"
        f"（来源={metric_source}, val 选择，test 集不参与）"
    )
    if multi_axis:
        print(
            "[suite-baseline-tuning] 坐标式多轴设计：每个候选"
            "相对基线配置只改一个轴，其余轴固定在基线值"
            f"（lr={format(base_lr, 'g')}, "
            f"weight_decay={format(base_weight_decay, 'g')}"
            + (
                f", dense_fnn_hidden={base_dense_hidden}"
                if base_dense_hidden is not None
                else ""
            )
            + "）"
        )
    if len(seeds) < _RECOMMENDED_SEEDS:
        print(
            "[suite-baseline-tuning] 提示：调参建议使用 "
            f"3~5 个随机种子评估每个候选，当前只有 "
            f"{len(seeds)} 个"
        )

    run_paths = _execute_sweep_plan(
        base_argv, plan, out_root, design=design
    )

    # 套件必须使用同一份预注册文件：若文件在运行中途被编辑，
    # 各子运行实际执行的协议就不再相同，不能合并成一项研究。
    prereg_record = getattr(
        args, "preregistration_record", None
    )
    if isinstance(prereg_record, dict):
        sweep_prereg_error = preregistration_sweep_error(
            run_paths,
            str(prereg_record.get("file_sha256")),
        )
        if sweep_prereg_error is not None:
            raise SystemExit(sweep_prereg_error)

    design_error = design_plan_consistency_error(
        run_paths, str(design["plan_id"])
    )
    if design_error is not None:
        raise SystemExit(design_error)

    rows = compare_runs(run_paths)
    # --compare-metric 可以是等计算量 loss；调参报告必须与
    # 当前对比指标同口径。调参候选只有学习率不同、每步 FLOPs
    # 相同，等计算量点仍由验证曲线共同计算量决定。
    annotate_compute_matched_metrics(rows)
    if multi_axis:
        axes = [
            {
                "name": "lr",
                "protocol_key": "lr",
                "label": "学习率",
                "values": [float(lr) for lr in lrs],
                "base_value": base_lr,
            },
        ]
        if dense_hiddens:
            axes.append({
                "name": "dense_fnn_hidden",
                "protocol_key": "dense_fnn_hidden",
                "label": "dense FFN 隐藏宽度",
                "values": [int(value) for value in dense_hiddens],
                "base_value": base_dense_hidden,
            })
        if weight_decays:
            axes.append({
                "name": "weight_decay",
                "protocol_key": "weight_decay",
                "label": "权重衰减",
                "values": [
                    float(value) for value in weight_decays
                ],
                "base_value": base_weight_decay,
            })
        report = analyze_baseline_tuning_axes(
            rows,
            metric,
            mode,
            axes,
            seeds,
            {
                "lr": base_lr,
                "dense_fnn_hidden": base_dense_hidden,
                "weight_decay": base_weight_decay,
            },
        )
    else:
        report = analyze_baseline_tuning(
            rows,
            metric,
            mode,
            lrs,
            seeds,
        )
    report["metric_source"] = metric_source
    report["seed_plan"] = _build_seed_plan(
        seeds,
        args.allow_underpowered,
        planned_comparisons=1,
    )
    report["design"] = {
        key: design.get(key)
        for key in (
            "path",
            "file_sha256",
            "plan_id",
            "kind",
            "order_scheme",
            "order_seed",
            "order_balance",
            "planned_runs",
            "training_budget",
        )
    }
    if len(seeds) < _MIN_SEEDS_FOR_STATS:
        report["issues"].append(
            f"每个候选只有 {len(seeds)} 个种子"
            f"（< {_MIN_SEEDS_FOR_STATS}）：winner 可能只是"
            "种子噪声的产物，调参结果不具统计功效"
        )
    print_baseline_tuning_report(report, out_root)
    write_baseline_tuning_report(out_root, report)

    if report.get("winner") is None:
        raise SystemExit(
            "基线调参没有选出 winner（见报告 issues），"
            "不能产出「调过参的基线」；请检查失败原因后重跑"
        )

    if report.get("multi_axis"):
        missing_axes = [
            str(axis_report.get("label", axis_report.get("name")))
            for axis_report in report.get("axes", [])
            if axis_report.get("winner") is None
        ]
        if missing_axes:
            raise SystemExit(
                "基线调参在以下轴上没有选出 winner："
                + ", ".join(missing_axes)
                + "（见报告 issues），不能产出「调过参的基线」"
            )



def build_sweep_execution(
    args: argparse.Namespace,
    out_root: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object], int]:
    """
    构造并落盘套件执行计划：运行清单 → 顺序/区组 → 设计记录。

    训练开始前就固定「跑哪些单元、按什么顺序跑、每个种子区组
    内臂如何排列」，并写入 design_plan.json。之后子进程核对
    同一份计划，事后无法把顺序/臂集合改成更有利的版本。
    """
    plan = build_sweep_plan(
        seeds=list(args.seeds),
        modes=(
            list(args.suite_modes)
            if args.suite_modes
            else None
        ),
        ablation=bool(args.suite_ablation),
        balance_coef=args.balance_coef,
        default_mode=args.mode,
        ablation_modes=(
            list(args.ablation_modes)
            if args.suite_ablation and args.ablation_modes
            else None
        ),
    )
    scheme, order_seed = resolve_run_order(args)
    prereg_record = getattr(args, "preregistration_record", None)
    prereg_power = (
        prereg_record.get("power")
        if isinstance(prereg_record, dict)
        else None
    )
    actual_comparisons = planned_comparisons_for_plan(plan)
    if (
        isinstance(prereg_power, dict)
        and isinstance(
            prereg_power.get("planned_comparisons"), int
        )
        and not isinstance(
            prereg_power.get("planned_comparisons"), bool
        )
    ):
        declared_comparisons = int(
            prereg_power["planned_comparisons"]
        )
        # 预注册声明的家族规模可能小于实际计划展开的对比数
        # （例如 comparisons 只写了星形对照，但套件包含所有
        # 两两臂）。家族校正与种子计划必须按更大的实际规模，
        # 否则稳健符号翻转门槛会被低估。
        planned_comparisons = max(
            declared_comparisons, actual_comparisons
        )
        if planned_comparisons > declared_comparisons:
            print(
                "[design][warn] 预注册 power.planned_comparisons="
                f"{declared_comparisons} 小于计划实际展开的 "
                f"{actual_comparisons} 个处理单元比较；"
                f"设计/种子计划按 {planned_comparisons} 校正"
            )
    else:
        planned_comparisons = actual_comparisons
    ordered = order_sweep_plan(plan, scheme, order_seed)
    design = write_design_plan(
        out_root,
        ordered,
        scheme,
        order_seed,
        args,
        planned_comparisons=planned_comparisons,
    )
    return ordered, design, planned_comparisons



def print_design_summary(design: Dict[str, object]) -> None:
    """打印设计计划摘要（准备阶段可见，不必先烧算力）。"""
    print(
        "[design] 计划 "
        f"{design['planned_runs']} 次运行；顺序="
        f"{design['order_scheme']}（order_seed="
        f"{design['order_seed']}）；计划指纹="
        f"{str(design['plan_id'])[:12]}..."
    )
    budget = design.get("training_budget")
    if isinstance(budget, dict):
        print(
            "[design] 解析训练预算："
            f"{budget.get('total_tokens')} tokens（"
            f"{budget.get('runs')} × {budget.get('tokens_per_run')}）"
        )
    balance = design.get("order_balance")
    if isinstance(balance, dict) and balance.get("available"):
        if balance.get("balanced"):
            print(
                "[design][order] 区组位置平衡：每个处理在各位置上"
                "的计数极差 "
                f"{balance.get('max_position_imbalance')}（"
                f"{balance.get('n_blocks')} 个区组、"
                f"{balance.get('n_treatments')} 个处理）"
            )
        elif design.get("order_scheme") == "counterbalanced":
            print(
                "[design][order][warn] counterbalanced 计划未达到"
                "位置平衡（最大极差 "
                f"{balance.get('max_position_imbalance')}）："
                "请检查区组是否包含同一组处理"
            )
        else:
            print(
                "[design][order] 提示：当前顺序的最大位置计数"
                f"极差为 {balance.get('max_position_imbalance')}；"
                "randomized 只保证区组内随机化，不保证位置计数"
                "均衡。若担心热漂移与位置混杂，可改用 "
                "--run-order counterbalanced"
            )
    power = design.get("power")
    if isinstance(power, dict):
        recompute_failed = (
            power.get("adequacy_recomputed") is False
        )
        if recompute_failed:
            print(
                "[design][power][warn] 无法按实际家族/对比类型或"
                "试点观测值重算先验功效：不沿用旧的 adequate/"
                "所需种子/功效/MDE 记录"
                + (
                    f"（{power.get('adequacy_recompute_error')}）"
                    if power.get("adequacy_recompute_error")
                    else ""
                )
            )
        status = (
            "无法重算"
            if recompute_failed
            else ("充足" if power.get("adequate") else "不足")
        )
        attrition = _finite_float(
            power.get("expected_attrition_rate")
        )
        if attrition is None:
            attrition = 0.0
        required = power.get(
            "required_seeds_with_attrition",
            power.get("required_seeds"),
        )
        required_text = (
            str(required) if required is not None else "NA"
        )
        required_t = power.get("required_seeds_for_t")
        required_robust = power.get(
            "required_seeds_for_robust_sign_flip"
        )
        planned_text = power.get("planned_seeds")
        planned_text = (
            f"{planned_text} 个"
            if planned_text is not None
            else "未知"
        )
        usable_text = power.get("planned_usable_seeds")
        usable_text = (
            f"{usable_text} 个"
            if usable_text is not None
            else "未知"
        )
        print(
            "[design][power] 先验功效："
            f"计划 {planned_text}种子，"
            f"所需启动种子 {required_text}（t="
            f"{required_t if required_t is not None else 'NA'}, "
            "稳健符号翻转="
            f"{required_robust if required_robust is not None else 'NA'}）"
            + (
                f"，预期脱落 {attrition:.0%}、计划可用 "
                f"{usable_text}"
                if attrition > 0.0
                else ""
            )
            + f" → {status}"
            + (
                "；已显式承认功效不足"
                if power.get("underpowered_acknowledged")
                and not power.get("adequate")
                else ""
            )
        )
        achieved = power.get("achieved_power_at_min_effect")
        mde = power.get("minimum_detectable_effect_relative")
        if achieved is not None or mde is not None:
            print(
                "[design][power] 计划可用种子下："
                + (
                    "对最小效应的功效≈"
                    f"{_format_finite(achieved, '.2f')}"
                    if _finite_float(achieved) is not None
                    else "功效无法计算"
                )
                + (
                    "，80% 功效 MDE≈"
                    f"{_format_finite(mde, '.2%')}"
                    if _finite_float(mde) is not None
                    else ""
                )
            )
        contrast_power = power.get("contrast_power")
        if (
            isinstance(contrast_power, dict)
            and len(contrast_power) > 1
        ):
            parts = []
            for contrast in _CONTRAST_ORDER:
                entry = contrast_power.get(contrast)
                if not isinstance(entry, dict):
                    continue
                required_contrast = entry.get(
                    "required_seeds_for_t"
                )
                parts.append(
                    f"{contrast}="
                    + (
                        str(required_contrast)
                        if required_contrast is not None
                        else "NA"
                    )
                )
            binding = power.get("binding_contrast")
            print(
                "[design][power] 各对比类型所需种子（"
                f"binding={binding}）：" + ", ".join(parts)
            )
            if binding == "interaction":
                print(
                    "[design][power] 交互的方差是两两对比的 2 倍，"
                    "所需种子约为 2 倍；消融设计应按该口径准备种子"
                )
        equivalence = power.get("equivalence")
        if isinstance(equivalence, dict):
            eq_power = equivalence.get(
                "power_at_null_difference"
            )
            eq_required = equivalence.get(
                "required_seeds_with_attrition"
            )
            eq_margin_needed = equivalence.get(
                "margin_for_target_power"
            )
            claim = "硬门" if equivalence.get(
                "claim_planned"
            ) else "信息性"
            margin_relative = equivalence.get("margin_relative")
            if margin_relative is None:
                margin_relative = 0.0
            print(
                "[design][equivalence] "
                f"{claim}：等价界 ±"
                f"{_format_finite(margin_relative, '.2%')}，"
                "真实差异为零时 "
                + (
                    "功效≈"
                    f"{_format_finite(eq_power, '.2f')}"
                    if _finite_float(eq_power) is not None
                    else "功效无法计算"
                )
                + (
                    f"，所需启动种子 {eq_required}"
                    if eq_required is not None
                    else "，128 个种子内无法达到目标功效"
                )
                + (
                    "；当前精度可支持的最小界≈"
                    f"±{_format_finite(eq_margin_needed, '.2%')}"
                    if _finite_float(eq_margin_needed) is not None
                    else ""
                )
            )
            if (
                equivalence.get("claim_planned")
                and not equivalence.get("adequate")
                and not power.get("underpowered_acknowledged")
            ):
                print(
                    "[design][equivalence][warn] 等价性主张功效"
                    "不足：请增加种子，或取消 "
                    "plan_equivalence（把 TOST 降为探索性"
                    "结果），不能事后放宽等价界"
                )
    else:
        print(
            "[design] 未提供预注册 power 块：种子数只保证检验"
            "可算，不保证能检出预注册的最小效应；建议在 "
            "preregistration.json 中声明 expected_seed_sd_relative"
        )
    pilot = design.get("pilot_evidence")
    if isinstance(pilot, dict):
        annotation = _as_dict(pilot.get("metric_annotation"))
        print(
            "[design][pilot] 试点证据："
            f"{pilot.get('source_run_count')} 个运行、"
            f"{pilot.get('cell_count')} 个处理单元、"
            f"{pilot.get('pair_count')} 个可配对单元对；指标="
            f"{pilot.get('metric')}（"
            f"{annotation.get('kind')}，target_step="
            f"{_format_finite(annotation.get('target_step'), 'g')}）"
        )
        comparability = _as_dict(
            pilot.get("comparability")
        )
        if comparability:
            status = comparability.get("status")
            print(
                "[design][pilot][compat] 协议可比性="
                f"{status}"
                + (
                    "（已核对）"
                    if comparability.get("verified")
                    else (
                        "（不可比：禁止用于功效校准）"
                        if comparability.get("status")
                        == "incomparable"
                        else "（缺少记录，未核实）"
                    )
                )
                + f"；不可比项 "
                f"{comparability.get('blocking_mismatch_count')}、"
                "未核对项 "
                f"{comparability.get('unverified_key_count')}"
            )
            seed_overlap = _as_dict(
                comparability.get("seed_overlap")
            )
            if seed_overlap.get("reused_seed_count"):
                print(
                    "[design][pilot][warn] 试点 seed 与正式计划"
                    "重叠 "
                    f"{seed_overlap.get('reused_seed_count')} 个："
                    "同一批运行不能同时充当设计校准与独立确认"
                )
        if annotation.get("target_step_data_dependent"):
            print(
                "[design][pilot][warn] 试点指标的共同步数由"
                "观测端点决定（未固定 --common-step）："
                "sd 估计依赖这批试点曲线的长度，正式设计应固定"
                "共同步数后重算"
            )
        sd_check = _as_dict(pilot.get("sd_check"))
        pilot_recompute_ok = pilot.get("power_recompute_ok")
        if pilot_recompute_ok is None:
            pilot_recompute_ok = bool(pilot.get("applied_to_power"))
        if sd_check.get("status") == "declared_below_observed":
            print(
                "[design][pilot][warn] 试点配对差值 sd≈"
                f"{_format_finite(sd_check.get('observed'), 'g')}"
                " 高于预注册 "
                f"{_format_finite(sd_check.get('declared'), 'g')}"
                + (
                    "；设计功效已按试点观测值重算"
                    if pilot_recompute_ok
                    else "；设计功效未能按试点观测值重算"
                )
            )
        elif sd_check.get("status") == "declared_at_least_observed":
            print(
                "[design][pilot] 预注册 sd≈"
                f"{_format_finite(sd_check.get('declared'), 'g')}"
                " 不低于试点观测 "
                f"{_format_finite(sd_check.get('observed'), 'g')}"
                f"（{sd_check.get('n_evaluable_pairs')} 个可评估"
                "单元对）"
            )
        elif sd_check.get("status") == "not_declared":
            print(
                "[design][pilot][warn] 预注册未声明 "
                "expected_seed_sd_relative：试点证据已估计 "
                f"sd≈{_format_finite(sd_check.get('observed'), 'g')}"
                "，但没有可对照的先验声明；请在新一轮预注册中"
                "显式填写后重新生成设计计划"
            )
        else:
            print(
                "[design][pilot][warn] 试点证据无法估计配对差值 "
                "sd（共同 seed 少于 "
                f"{sd_check.get('min_pairs')} 个或指标不可用）："
                "不能据此核对预注册 sd"
            )
        attrition_check = _as_dict(pilot.get("attrition_check"))
        if attrition_check.get("status") == "declared_below_observed":
            print(
                "[design][pilot][warn] 试点脱落率≈"
                f"{_format_finite(attrition_check.get('observed'), '.1%')}"
                " 高于预注册 "
                f"{_format_finite(attrition_check.get('declared'), '.1%')}"
                + (
                    "；所需启动种子已按观测脱落率重算"
                    if pilot_recompute_ok
                    else "；所需启动种子未能按观测脱落率重算"
                )
            )
        elif attrition_check.get("status") == "declared_at_least_observed":
            print(
                "[design][pilot] 预注册脱落率 "
                f"{_format_finite(attrition_check.get('declared'), '.1%')}"
                " 不低于试点观测 "
                f"{_format_finite(attrition_check.get('observed'), '.1%')}"
            )
        elif attrition_check.get("status") == "not_declared":
            print(
                "[design][pilot][warn] 预注册未声明 "
                "expected_attrition_rate：试点证据已估计脱落率≈"
                f"{_format_finite(attrition_check.get('observed'), '.1%')}"
                "，但没有可对照的先验声明；请在新一轮预注册中"
                "显式填写后重新生成设计计划"
            )
        for warning in list(pilot.get("warnings") or [])[:4]:
            print(f"[design][pilot][warn] {warning}")
    print(
        f"[design] 设计计划已写入 {design['path']}"
        "（子进程开跑前会重新核对指纹与自己的分配）"
    )



def design_power_gate_error(
    design: Dict[str, object],
    allow_underpowered: bool = False,
) -> Optional[str]:
    """训练前的硬门：计划实际展开的对比是否达到先验功效。

    预注册入口只能按文件里声明的 ``comparisons``/``planned_contrasts``
    检查；套件真正展开的家族可能更大（全部臂两两对比），消融还
    会估计方差更大的交互。设计计划重算后仍不足时，必须在开跑前
    拒绝，而不是等分析时再说「交互不显著」。

    ``pilot_evidence`` 存在时还检查试点运行是否至少留下一个可用
    单元；某个臂的全部试点运行都失败/缺指标时，声明 sd 与脱落率
    都无从核对，不能当作「功效足够」放行。
    """
    power = design.get("power")
    pilot_evidence = design.get("pilot_evidence")
    pilot_note = _pilot_gate_note(pilot_evidence)
    # 协议不可比与「功效不足」是两类问题：--allow-underpowered
    # 只承认后者，不能把错误数据来源变成可用校准证据，因此该
    # 检查必须放在 allow_underpowered 的早退之前。
    if isinstance(pilot_evidence, dict):
        comparability = _as_dict(
            pilot_evidence.get("comparability")
        )
        if (
            comparability.get("usable_for_calibration") is False
            and pilot_evidence.get("applied_to_power")
        ):
            return (
                pilot_note
                + "设计计划把不可比的试点证据用于功效校准："
                "同一份设计不能再声称 adequate。请用可比试点"
                "重算，或去掉 --design-pilot-runs 只使用预注册"
                "声明的先验值，并用新的 --out 重新生成计划"
            )
    # --allow-underpowered 必须独立于 power 记录生效：即使设计
    # 计划没有（或损坏了）power 块，冒烟/探索运行也应能显式
    # 绕过启动门；否则会在试点证据不可用时比旧行为更严格，
    # 且与错误信息里「或加 --allow-underpowered」矛盾。
    if allow_underpowered:
        return None
    acknowledged = (
        isinstance(power, dict)
        and bool(power.get("underpowered_acknowledged"))
    )
    if acknowledged:
        return None
    if (
        isinstance(pilot_evidence, dict)
        and _as_dict(
            pilot_evidence.get("attrition_check")
        ).get("unusable")
    ):
        return (
            "设计计划显示试点运行证据不可用："
            + pilot_note
            + "至少一个处理单元的全部试点运行都因数值失败/缺"
            "指标而不可用；这不是「功效不足」，而是该臂在当前"
            "口径下没有可估计的方差与脱落率。请先修复该臂或"
            "补齐试点运行，再用新的 --out 重新生成设计计划；"
            "确需在不足设计中探索，请显式设置 "
            "underpowered_acknowledged=true 并写明 justification"
        )
    if isinstance(pilot_evidence, dict):
        pilot_cells = _finite_float(
            pilot_evidence.get("cell_count")
        )
        if pilot_cells is not None and pilot_cells <= 0.0:
            return (
                pilot_note
                + "设计计划显示试点证据没有任何可归入处理单元"
                "的运行（每个运行都必须有 mode 与整数 seed）："
                "无法从这批证据核对声明 sd/脱落率。请补齐 "
                "config.json 中的 mode/seed，或传入正确的运行"
                "目录，再用新的 --out 重新生成设计计划；确需在"
                "不足设计中探索，请显式设置 "
                "underpowered_acknowledged=true 并写明 justification"
            )
    if not isinstance(power, dict):
        return None
    if power.get("adequacy_recomputed") is False:
        reason = power.get("adequacy_recompute_error")
        return (
            pilot_note
            + "设计计划无法按实际家族/对比类型或试点观测值"
            "重算先验功效"
            + (f"（{reason}）" if reason else "")
            + "：不能沿用旧记录里的 adequacy；请补全预注册 "
            "power 的 min_effect/planned_seeds 等重算字段后重新"
            "生成设计计划，或显式设置 "
            "underpowered_acknowledged=true 并写明 justification"
        )
    if power.get("adequate"):
        return None
    if (
        power.get("plan_equivalence")
        and not power.get("equivalence_adequate")
    ):
        # 旧/手工编辑的 design_plan 可能把 equivalence 写成
        # 标量或列表；损坏字段按空对象处理，不能让整份启动门
        # 在标量上调用 .get 崩溃。
        equivalence = _as_dict(power.get("equivalence"))
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
        margin_relative = equivalence.get("margin_relative")
        if margin_relative is None:
            margin_relative = 0.0
        target_text = _format_finite(
            power.get("target_power"), "g"
        )
        if target_text == "-":
            target_text = "未知"
        family_text = power.get("planned_comparisons")
        family_text = (
            str(family_text)
            if family_text is not None
            else "未知"
        )
        planned_text = power.get("planned_seeds")
        planned_text = (
            f"{planned_text} 个"
            if planned_text is not None
            else "未知"
        )
        return (
            pilot_note
            + "设计计划显示等价性（TOST）功效不足：等价界 ±"
            f"{_format_finite(margin_relative, 'g')}、"
            "真实差异为零、目标功效 "
            f"{target_text}、家族规模 "
            f"{family_text} 下，"
            f"{eq_required_text}，计划只有 "
            f"{planned_text}"
            + (
                f"；当前精度可支持的最小界约 ±"
                f"{_format_finite(eq_margin_needed, 'g')}"
                if _finite_float(eq_margin_needed) is not None
                else ""
            )
            + "。等价性主张不能靠事后放宽界来「制造」：请增加 "
            "--seeds，或把 power.plan_equivalence 设为 false "
            "并只把 TOST 作为探索性结果；确需在不足设计中运行，"
            "请显式设置 underpowered_acknowledged=true 并写明 "
            "justification，或加 --allow-underpowered。调整后"
            "请使用新的 --out（被拒绝的设计计划已落盘）"
        )
    binding = str(power.get("binding_contrast") or "pairwise")
    binding_label = str(
        power.get("binding_contrast_label")
        or _CONTRAST_LABELS.get(binding, binding)
    )
    required = power.get("required_seeds_with_attrition")
    if required is None:
        required = power.get("required_seeds")
    planned = power.get("planned_seeds")
    required_text = (
        f"{required} 个启动种子"
        if required is not None
        else "在当前精度/显著性水平下无法计算的启动种子数"
    )
    planned_text = (
        f"{planned} 个" if planned is not None else "未知"
    )
    target_text = _format_finite(
        power.get("target_power"), "g"
    )
    if target_text == "-":
        target_text = "未知"
    family_text = power.get("planned_comparisons")
    family_text = (
        str(family_text) if family_text is not None else "未知"
    )
    return (
        pilot_note
        + "设计计划显示先验功效不足：计划实际展开的最吃种子对比"
        f"类型是 {binding}（{binding_label}），至少需要 "
        f"{required_text}，计划只有 {planned_text}；"
        f"家族规模 {family_text}、目标功效 "
        f"{target_text}。"
        "交互对比的方差是两两对比的 2 倍，不能把「按两两对比"
        "够用」当成「消融交互也够用」。请增加 --seeds，或在预"
        "注册 power 中显式设置 underpowered_acknowledged=true "
        "并写明 justification；冒烟/调试可显式加 "
        "--allow-underpowered。调整后请使用新的 --out（被拒绝"
        "的设计计划已落盘，旧目录会按冻结设计拒绝覆盖）。"
    )
