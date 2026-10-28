from __future__ import annotations



def run_experiment_sweep(args: argparse.Namespace) -> None:
    """
    执行多随机种子 / 模式套件 / 全因子消融 / 基线调参套件，
    并输出受控变量检查 + Mean ± Std + 显著性检验 + 消融
    覆盖报告（基线调参套件输出调参报告）。
    """
    if args.suite_baseline_tuning:
        try:
            run_baseline_tuning_sweep(args)
        except ValueError as error:
            raise SystemExit(f"[design] {error}") from error
        return

    factors = parse_factor_spec(args.compare_factors)

    base_argv = strip_sweep_args(sys.argv[1:])
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        plan, design, planned_comparisons = build_sweep_execution(
            args, out_root
        )
    except ValueError as error:
        raise SystemExit(f"[design] {error}") from error

    # 预注册入口只能按文件声明的家族/对比类型做功效检查；套件
    # 实际展开的家族可能更大（全部臂两两对比），消融还会估计
    # 方差更大的交互。设计计划重算后仍不足时，必须在第一个子
    # 进程启动前拒绝，不能等分析阶段再解释「交互不显著」。
    power_gate_error = design_power_gate_error(
        design,
        allow_underpowered=bool(
            getattr(args, "allow_underpowered", False)
        ),
    )
    if power_gate_error is not None:
        raise SystemExit(f"[design] {power_gate_error}")

    if len(args.seeds) < _RECOMMENDED_SEEDS:
        print(
            "[sweep] 提示：关键实验建议使用 3~5 个不同随机"
            f"种子并汇报 Mean ± Std，当前只有 "
            f"{len(args.seeds)} 个"
        )
        if args.allow_underpowered:
            print(
                "[sweep] [warn] 已通过 --allow-underpowered "
                "显式允许低统计功效运行：种子数 "
                f"{len(args.seeds)} < {_MIN_SEEDS_FOR_STATS}，"
                "本轮结论不得作为显著性主张（记录在 "
                "comparison_analysis.json 的 seed_plan）"
            )

    print_design_summary(design)
    run_paths = _execute_sweep_plan(
        base_argv, plan, out_root, design=design
    )

    # 每个子进程都会重新读取预注册文件；文件在套件运行中途被
    # 修改时，各子运行会基于不同 sha256 训练。父进程必须核对
    # 全部子运行的 file_sha256 与本次冻结文件一致，不能把混合
    # 协议当成同一项预注册研究。
    prereg_record = getattr(
        args, "preregistration_record", None
    )
    if (
        isinstance(prereg_record, dict)
        and prereg_record.get("file_sha256")
    ):
        sweep_prereg_error = preregistration_sweep_error(
            [str(path) for path in run_paths],
            str(prereg_record["file_sha256"]),
        )
        if sweep_prereg_error is not None:
            raise SystemExit(sweep_prereg_error)

    design_error = design_plan_consistency_error(
        [str(path) for path in run_paths],
        str(design["plan_id"]),
    )
    if design_error is not None:
        raise SystemExit(design_error)

    rows = compare_runs(run_paths)
    analysis = print_comparison_analysis(
        rows,
        factors=factors,
        baseline_mode=args.compare_baseline,
        metric=args.compare_metric,
        ablation_modes=(
            list(args.ablation_modes)
            if args.suite_ablation and args.ablation_modes
            else None
        ),
        equivalence_margin=args.equivalence_margin,
        allow_partial_ablation=bool(
            args.allow_partial_ablation
        ),
        common_step=args.common_step,
    )
    # 种子计划随报告落盘：事后能核对本轮结论是在几个种子上
    # 得出的、是否使用了低统计功效豁免。家族规模用预注册/
    # 计划中的比较数，避免把稳健符号翻转所需种子数说少。
    analysis["seed_plan"] = _build_seed_plan(
        args.seeds,
        args.allow_underpowered,
        planned_comparisons=planned_comparisons,
    )
    analysis["design"] = {
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
            "planned_comparisons",
            "planned_contrasts",
            "power",
            "seed_plan",
            "training_budget",
        )
    }
    analysis_power = analysis["design"].get("power")
    if isinstance(analysis_power, dict):
        if (
            analysis_power.get("plan_equivalence")
            and not analysis_power.get("equivalence_adequate")
        ):
            equivalence = (
                analysis_power.get("equivalence") or {}
            )
            print(
                "[analysis][warn] 本轮设计的等价性（TOST）功效"
                "不足（等价界 ±"
                f"{float(equivalence.get('margin_relative') or 0.0):g}，"
                "需要 "
                f"{equivalence.get('required_seeds_with_attrition')} "
                "个启动种子，计划 "
                f"{analysis_power.get('planned_seeds')} 个）："
                "不能把 TOST 未通过读成「两者不同」，也不能"
                "事后放宽等价界；等价结论只能作为探索性证据"
            )
        elif not analysis_power.get("adequate"):
            print(
                "[analysis][warn] 本轮设计功效不足（最吃种子的对比 "
                f"{analysis_power.get('binding_contrast')}，需要 "
                f"{analysis_power.get('required_seeds_with_attrition')} "
                "个启动种子，计划 "
                f"{analysis_power.get('planned_seeds')} 个）："
                "结论只能作为探索性证据，不能把交互/主效应的不显著"
                "读成「没有效应」；该豁免已随 design.power 落盘"
            )
    analysis["run_order"] = run_order_drift_report(
        rows, args.compare_metric
    )
    write_comparison_report(out_root, analysis)

def main():
    from exp._runner import run_isolated
    from exp import _manifest
    _root = os.path.dirname(os.path.abspath(__file__))
    _paths = [os.path.join(_root, 'exp', 'mainflow', _name) for _name in _manifest.MAIN_PARTS]
    return run_isolated(_paths, globals(), _manifest.MAIN_OWNED, _manifest.MAIN_ENTRY, {})
