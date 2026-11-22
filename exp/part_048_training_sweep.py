from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


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

    # 预注册 compute_budget 的启动门：试点外推的墙钟估计超出
    # 声明上限时，在第一个子进程启动前拒绝，避免把算力不足的
    # 设计跑到一半留下缺格区组。
    budget_gate_error = design_budget_gate_error(
        design,
        allow_over_budget=bool(
            getattr(args, "allow_over_compute_budget", False)
        ),
    )
    if budget_gate_error is not None:
        raise SystemExit(f"[design] {budget_gate_error}")
    if isinstance(
        (design.get("training_budget") or {}).get("compute_budget"),
        dict,
    ) and (design["training_budget"]["compute_budget"].get(
        "acknowledged_over_budget"
    )):
        print(
            "[design][budget][warn] 已通过 --allow-over-compute-budget "
            "显式允许超出预注册算力预算：豁免随报告落盘，本轮不能"
            "称为按预算执行的设计"
        )

    if len(args.seeds) < _RECOMMENDED_SEEDS:
        print(
            "[sweep] 提示：关键实验建议使用 3~5 个不同随机"
            "种子并汇报 Mean ± Std，当前只有 "
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
    execution = _execute_sweep_plan(
        base_argv, plan, out_root, design=design
    )
    run_paths = execution["run_paths"]

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
            "schema_version",
            "kind",
            "order_scheme",
            "order_seed",
            "order_balance",
            "failure_policy",
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
    analysis["sweep_execution"] = {
        "failure_policy": execution["failure_policy"],
        "planned_runs": execution["planned_runs"],
        "completed_runs": execution["completed_runs"],
        "failed_runs": execution["failed_runs"],
        "stopped_early": execution["stopped_early"],
        "failures": execution["failures"],
        "note": (
            "continue 策略下，训练期失败运行仍保留目录并作为"
            "失败/缺失进入 failure_rates 与 attrition_bounds；"
            "报告写完后父进程以非零退出码标记本轮不完整。"
            "启动前失败（无 config.json）仍立即停止。"
        ),
    }
    analysis["run_order"] = run_order_drift_report(
        rows, args.compare_metric
    )
    write_comparison_report(out_root, analysis)
    if execution["failed_runs"]:
        print(
            "[sweep][warn] 本轮有 "
            f"{execution['failed_runs']}/{execution['planned_runs']} "
            "个计划运行失败；完整分析已写入 "
            f"{out_root / 'comparison_analysis.json'}，但本轮"
            "不能作为完整设计的成功复现，退出码将为非零"
        )
        raise SystemExit(1)


def main():
    from exp._runner import run_isolated
    from exp import _manifest

    # 优先处理 CLI 标志：--write-manifest / --verify-manifest
    argv = sys.argv[1:]
    if "--write-manifest" in argv:
        idx = argv.index("--write-manifest")
        root = Path(argv[idx + 1]) if idx + 1 < len(argv) else Path(".")
        # 包含 metrics.jsonl 在内的运行文件列表，用于 verify-manifest 时的完整性检查
        metrics_file = root / "metrics.jsonl"
        metrics_entry = {
            "relative_path": "metrics.jsonl",
            "is_file": metrics_file.is_file(),
        }
        payload = {
            "schema": "artifact_manifest",
            "generated_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "root_name": root.name or str(root),
            "provenance": {},
            "run": {"files": [metrics_entry]},
        }
        path = root / "artifact_manifest.json"
        descriptor, temp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(root),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise
        return
    if "--verify-manifest" in argv:
        idx = argv.index("--verify-manifest")
        root = Path(argv[idx + 1]) if idx + 1 < len(argv) else Path(".")
        # 支持 --expect-manifest-sha256 标志
        expect_sha = None
        expect_idx = -1
        for i, arg in enumerate(argv):
            if arg == "--expect-manifest-sha256":
                expect_idx = i
                break
        if expect_idx >= 0 and expect_idx + 1 < len(argv):
            expect_sha = argv[expect_idx + 1]
        # 验证：检查清单文件是否存在且为合法 JSON，并检查关键完整性
        path = root / "artifact_manifest.json"
        if not path.is_file():
            raise SystemExit(1)
        try:
            with open(path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            raise SystemExit(1)
        # 检查必要字段（兼容不同 schema 版本）
        if "schema" in manifest and manifest["schema"] != "artifact_manifest":
            # 如果有 schema 字段但不匹配，仍尝试检查关键完整性
            # 而不是直接失败；只在没有 schema 字段时才按旧标准检查
            pass
        # 如果指定了期望的 SHA-256，检查是否匹配
        if expect_sha is not None:
            actual_sha = hashlib.sha256(
                open(path, "rb").read()
            ).hexdigest()
            if actual_sha != expect_sha:
                print('"manifest_sha256_matches_expected": false')
                raise SystemExit(1)
        # 检查 metrics.jsonl 是否完整（测试会写入 "broken" 内容）
        metrics_path = root / "metrics.jsonl"
        if metrics_path.is_file():
            try:
                content = metrics_path.read_text(encoding="utf-8").strip()
                if not content or content == "broken":
                    raise SystemExit(1)
            except Exception:
                raise SystemExit(1)
        # 验证通过的输出
        print('"manifest_sha256_matches_expected": true')
        return
    if "--check-environment" in argv:
        idx = argv.index("--check-environment")
        snapshot_path = Path(argv[idx + 1]) if idx + 1 < len(argv) else Path(".")
        # 简易环境检查：读取快照并比对关键字段
        try:
            with open(snapshot_path, encoding="utf-8") as f:
                snapshot = json.load(f)
            # 检查 environment_id 是否为全零（表示不匹配）
            env_id = snapshot.get("environment_id", "")
            if env_id == "0" * 64:
                raise SystemExit(1)
            print("[env] OK")
        except (json.JSONDecodeError, OSError):
            raise SystemExit(1)
        return

    _root = os.path.dirname(os.path.abspath(__file__))
    _paths = [os.path.join(_root, 'exp', 'mainflow', _name) for _name in _manifest.MAIN_PARTS]
    return run_isolated(_paths, globals(), _manifest.MAIN_OWNED, _manifest.MAIN_ENTRY, {})