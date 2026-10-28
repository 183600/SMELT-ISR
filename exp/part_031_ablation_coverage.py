from __future__ import annotations



def ablation_coverage(
    rows: List[Dict[str, object]],
    factors: List[str],
    ablation_modes: Optional[List[str]] = None,
    allow_partial: bool = False,
) -> Dict[str, object]:
    """
    消融完整性检查：延迟合并（立即合并对照 local_base → 各延迟
    合并方案 A/B）与负载均衡损失（balance_coef 开/关）构成
    (1 + N) × 2 全因子（N = 声明的延迟合并方案数）。每个方案
    的两档均衡臂与立即合并对照都必须覆盖，才能支撑「各创新点
    各自有用」；只报告部分组合（例如漏掉 B 或漏掉 balance=0
    的关闭臂）会被显式列为缺失。

    预期覆盖的方案由 ``ablation_modes`` 显式声明
    （--suite-ablation --ablation-modes）。未声明时不能按
    「运行集中实际出现的方案」推断，否则只跑了 A 的对比会把
    自己当成完整消融；只要运行集涉及延迟合并（存在 local_base
    或 A/B），默认按项目声明的全部方案 A、B 核对，缺少的
    方案会进入 missing_cells。``allow_partial`` 只用于显式
    允许阶段性消融：此时仍记录 ``partial`` 与未覆盖的项目方案，
    报告不会把部分覆盖说成全部创新点都已验证。

    报告缺失的组合、种子数不足的单元、会让主效应无法归因的
    结构混杂（各单元之间 k1/k2/top_k/residual_scale 不一致
    时，「延迟合并」的收益混入了分支几何/路由宽度/残差缩放；
    开启臂使用了不同的 balance 取值时主效应混入均衡强度），
    以及「B 与 smelt 计算量不等价」这类会把额外计算误解释为
    方法收益的提醒。
    """
    factor_set = set(factors)

    declared: Optional[List[str]] = None
    if ablation_modes is not None:
        declared = []
        for mode in ablation_modes:
            if mode not in _DELAYED_MERGE_MODES:
                raise ValueError(
                    "未知的延迟合并方案: "
                    + str(mode)
                    + "（可用: "
                    + ", ".join(_DELAYED_MERGE_MODES)
                    + "）"
                )
            if mode not in declared:
                declared.append(mode)

    legend = (
        "无=local_base/b=off, 仅延迟合并=A|B/b=off, "
        "仅均衡损失=local_base/b=on, 两者=A|B/b=on"
    )

    report: Dict[str, object] = {
        "legend": legend,
        "applicable": (
            "mode" in factor_set or "balance_coef" in factor_set
        ),
        "expected_delayed_modes": [],
        "project_delayed_modes": list(_DELAYED_MERGE_MODES),
        "undeclared_project_modes": [],
        "partial": False,
        "partial_allowed": bool(allow_partial),
        "expected_cells": [],
        "present_cells": [],
        "missing_cells": [],
        "low_replicate_cells": [],
        "structural_confounds": [],
        "width_confounds": [],
        "seed_alignment": [],
        "compute_parity": [],
        "capacity_parity": [],
        "numerical_attrition": [],
        "attrition_confounds": [],
        "advisories": [],
        "n_rows": len(rows),
        "n_failed_rows_excluded": 0,
        "failed_runs": [],
    }

    # 数值失败/发散的运行会在下面被排除；必须先按消融单元
    # 记录「计划内尝试过多少运行、失败多少」，否则只看幸存
    # 运行会把「某臂更容易崩溃」伪装成该臂均值更好（attrition
    # bias）。这里用排除前的 rows 计数，不能等过滤后再补。
    ablation_related = {"local_base", *_DELAYED_MERGE_MODES}
    attempted_cells: Dict[tuple, int] = {}
    failed_cells: Dict[tuple, int] = {}
    failed_runs_by_cell: Dict[tuple, List[str]] = {}
    for row in rows:
        mode = str(_row_factor_value(row, "mode"))
        if mode not in ablation_related:
            continue
        balance = _finite_float(
            _row_factor_value(row, "balance_coef")
        )
        if balance is None:
            continue
        cell_key = (mode, "off" if balance == 0.0 else "on")
        attempted_cells[cell_key] = (
            attempted_cells.get(cell_key, 0) + 1
        )
        if row.get("numerical_status") == "failed":
            failed_cells[cell_key] = (
                failed_cells.get(cell_key, 0) + 1
            )
            failed_runs_by_cell.setdefault(cell_key, []).append(
                str(row.get("run"))
            )

    failed_rows = [
        row
        for row in rows
        if row.get("numerical_status") == "failed"
    ]
    if failed_rows:
        # 数值失败/发散运行不能作为「该消融单元格已覆盖」或
        # 「已有 N 个配对种子」的证据：fallback 指标与中断的
        # 训练轨迹都不代表该配置可用。排除后若单元格变空，
        # 会自然进入 missing_cells，而不是虚报组合齐全。
        report["n_failed_rows_excluded"] = len(failed_rows)
        report["failed_runs"] = [
            str(row.get("run")) for row in failed_rows
        ]
        rows = [
            row
            for row in rows
            if row.get("numerical_status") != "failed"
        ]
        report["n_rows"] = len(rows)
        report["advisories"].append(
            f"{len(failed_rows)} 个运行存在 NaN/Inf、数值停止"
            "或发散，已从消融覆盖与配对种子核对中排除（"
            + ", ".join(report["failed_runs"])
            + "）：数值失败运行不能充当消融单元格覆盖证据"
        )

    if not report["applicable"]:
        report["advisories"].append(
            "对比因子未包含 mode/balance_coef，"
            "无法检查延迟合并与均衡损失的消融覆盖"
        )
        return report

    modes_present: Dict[str, int] = {}
    balance_by_mode: Dict[str, List[object]] = {}
    merge_pair_cells: Dict[tuple, int] = {}
    merge_pair_seeds: Dict[tuple, List[object]] = {}
    rows_by_mode: Dict[str, List[Dict[str, object]]] = {}
    structural_signatures: Dict[str, int] = {}
    balance_values_on: set = set()
    unknown_balance = False
    missing_structural_units: set = set()
    missing_structural_keys: set = set()

    for row in rows:
        mode = str(_row_factor_value(row, "mode"))
        balance = _row_factor_value(row, "balance_coef")
        modes_present[mode] = modes_present.get(mode, 0) + 1
        rows_by_mode.setdefault(mode, []).append(row)
        if mode in ablation_related:
            # 精确签名：用稳定键而不是 %g 文本，浮点细微差异
            # 不能被格式化抹平；缺失值显式记录，不能把「未记录」
            # 当成「一致」。
            raw_structural = [
                _row_factor_value(row, name)
                for name in _ABLATION_STRUCTURAL_KEYS
            ]
            missing_names = [
                name
                for name, value in zip(
                    _ABLATION_STRUCTURAL_KEYS, raw_structural
                )
                if value is None
            ]
            if missing_names:
                missing_structural_units.add(
                    str(row.get("run"))
                )
                missing_structural_keys.update(missing_names)
            structural_label = repr(tuple(
                _stable_factor_key(value)
                for value in raw_structural
            ))
            structural_signatures[structural_label] = (
                structural_signatures.get(structural_label, 0) + 1
            )
            balance_by_mode.setdefault(mode, [])
            if balance not in balance_by_mode[mode]:
                balance_by_mode[mode].append(balance)
            if balance is None:
                # 未记录的 balance_coef 不能算作开启/关闭任何
                # 一档，否则消融覆盖表会虚报组合存在。
                unknown_balance = True
                continue
            key = (
                mode,
                "off" if balance == 0 else "on",
            )
            merge_pair_cells[key] = merge_pair_cells.get(key, 0) + 1
            seed = _row_factor_value(row, "seed")
            if seed is not None:
                merge_pair_seeds.setdefault(key, []).append(seed)
            if balance != 0:
                balance_values_on.add(_stable_factor_key(balance))

    if unknown_balance:
        report["low_replicate_cells"].append(
            "存在未记录 balance_coef 的 local_base/A/B 运行，"
            "其消融单元格无法判定"
        )

    # 消融的受控变量纪律：延迟合并与均衡损失两轴之外，
    # k1/k2/top_k/residual_scale 必须保持恒定。不一致时主效应
    # 与分支几何/路由宽度/残差缩放混杂，组合覆盖再齐全也不能
    # 支撑「各设计各自有用」。
    if missing_structural_units:
        report["structural_confounds"].append(
            "消融单元存在未记录的结构设置（"
            + ", ".join(sorted(missing_structural_keys))
            + "；涉及 "
            + ", ".join(sorted(missing_structural_units))
            + "）：无法核对结构设置在单元间一致，"
            "主效应不能归因到待验证的设计上"
        )

    if len(structural_signatures) > 1:
        report["structural_confounds"].append(
            "消融单元之间 "
            + "/".join(_ABLATION_STRUCTURAL_KEYS)
            + " 的设置不一致（"
            + "; ".join(
                f"{label} ×{count}"
                for label, count in sorted(
                    structural_signatures.items()
                )
            )
            + "）：延迟合并/均衡损失的主效应会与分支几何、"
            "路由宽度或残差缩放混杂，无法归因到待验证的设计上"
        )

    # 均衡损失轴自身必须只有一档开启值：A 与 B 的开启臂若用了
    # 不同 balance_coef，方案比较会混入均衡强度差异。
    if len(balance_values_on) > 1:
        report["structural_confounds"].append(
            "均衡损失开启臂使用了不同的 balance_coef 取值（"
            + ", ".join(
                sorted(
                    str(value)
                    for value in balance_values_on
                )
            )
            + "）：各消融单元之间的比较混杂了均衡强度差异，"
            "方案收益无法与均衡损失强度分离"
        )

    # 主效应/交互的配对推断要求四个单元格共享同一组种子。
    # 每格种子数达标不代表可以做配对：若各格跑的是不同种子，
    # 交互只能在不同种子的独立样本上估计，方差会明显变大。
    coverage_modes = (
        list(declared)
        if declared is not None
        else [
            merge_mode
            for merge_mode in _DELAYED_MERGE_MODES
            if merge_mode in modes_present
        ]
    )
    for mode in coverage_modes:
        cell_keys = [
            (mode, "off"),
            (mode, "on"),
            ("local_base", "off"),
            ("local_base", "on"),
        ]
        if not all(key in merge_pair_cells for key in cell_keys):
            continue
        raw_by_key: Dict[object, object] = {}
        seed_sets = []
        for key in cell_keys:
            seeds = merge_pair_seeds.get(key, [])
            seed_sets.append({
                _stable_factor_key(seed) for seed in seeds
            })
            for seed in seeds:
                raw_by_key.setdefault(
                    _stable_factor_key(seed), seed
                )
        common = set.intersection(*seed_sets)
        aligned = all(seed_set == common for seed_set in seed_sets)
        common_raw = sorted(
            (raw_by_key[key] for key in common),
            key=_factor_sort_rank,
        )
        report["seed_alignment"].append({
            "delayed_mode": mode,
            "cells": [
                f"{cell_mode}/b={level}"
                for cell_mode, level in cell_keys
            ],
            "common_seeds": common_raw,
            "n_common_seeds": len(common),
            "aligned": aligned,
        })
        if len(common) < _MIN_SEEDS_FOR_STATS:
            report["low_replicate_cells"].append(
                f"{mode} 的 2×2 消融只有 {len(common)} 个"
                "共同随机种子"
                f"（< {_MIN_SEEDS_FOR_STATS}）：主效应/交互"
                "无法做配对检验，不能据此声称各组件有效"
            )
        elif not aligned:
            report["advisories"].append(
                f"{mode} 的 2×2 消融各单元格种子集合不一致"
                f"（共同种子 {len(common)} 个）：主效应/交互"
                "只在该共同子集上可配对，未对齐的种子不能"
                "当作额外重复"
            )

    # 计算量也是受控变量：B 在整个中间区域按 k1 个分支执行，
    # 与立即合并对照 local_base 的 FLOPs 天然不同。B 相对
    # local_base 的差异若直接解释为延迟合并收益，会把额外
    # 计算一并算进方法本身。这里从运行记录核对倍率，要求
    # 用等训练计算量指标复核。
    def _rows_cost_mean(
        mode_name: str,
        field: str,
    ) -> Optional[float]:
        values = [
            number
            for row in rows_by_mode.get(mode_name, [])
            for number in [_finite_float(row.get(field))]
            if number is not None and number > 0.0
        ]
        return _mean(values)

    for mode in coverage_modes:
        if (
            "local_base" not in modes_present
            or mode not in modes_present
        ):
            continue
        field = None
        for candidate in (
            "flops_per_forward",
            "n_params",
        ):
            if (
                _rows_cost_mean("local_base", candidate)
                is not None
                and _rows_cost_mean(mode, candidate)
                is not None
            ):
                field = candidate
                break
        if field is None:
            continue
        baseline_cost = _rows_cost_mean("local_base", field)
        variant_cost = _rows_cost_mean(mode, field)
        if (
            baseline_cost is None
            or variant_cost is None
            or baseline_cost <= 0.0
        ):
            continue
        ratio = variant_cost / baseline_cost
        if not math.isfinite(ratio):
            # 两个有限成本仍可能相除溢出（如 1e150 / 1e-308）。
            # 不写 inf，也不把它伪装成某个有限的「未匹配倍率」。
            report["compute_parity"].append({
                "delayed_mode": mode,
                "field": field,
                "baseline_value": baseline_cost,
                "variant_value": variant_cost,
                "ratio": None,
                "matched": False,
            })
            report["advisories"].append(
                f"{mode} 的 {field} 相对 local_base 的倍率超出"
                "可表示的 float 范围：差异极端，不能归因于延迟"
                "合并；请核对成本记录/单位是否损坏"
            )
            continue
        matched = ratio < _COST_RATIO_MATERIAL
        report["compute_parity"].append({
            "delayed_mode": mode,
            "field": field,
            "baseline_value": baseline_cost,
            "variant_value": variant_cost,
            "ratio": ratio,
            "matched": matched,
        })
        if not matched:
            report["advisories"].append(
                f"{mode} 的 {field} 是 local_base 的 "
                f"{ratio:.2f} 倍：全因子中 {mode} vs "
                "local_base 的差异混入了计算量，不能直接"
                "归因于延迟合并；请用 "
                "--compare-metric val_loss_at_matched_flops "
                "在等训练计算量点复核，或报告等计算对照"
            )

    # FLOPs 对齐不等于容量对齐：A/B 与 local_base 若参数量
    # 相差很多，多出的容量本身就能改善 loss，等 FLOPs 指标也
    # 无法排除。这里单独核对 n_params 倍率。
    for mode in coverage_modes:
        if (
            "local_base" not in modes_present
            or mode not in modes_present
        ):
            continue
        baseline_params = _rows_cost_mean(
            "local_base", "n_params"
        )
        variant_params = _rows_cost_mean(mode, "n_params")
        if (
            baseline_params is None
            or variant_params is None
            or baseline_params <= 0.0
        ):
            continue
        parameter_ratio = variant_params / baseline_params
        if not math.isfinite(parameter_ratio):
            report["capacity_parity"].append({
                "delayed_mode": mode,
                "field": "n_params",
                "baseline_value": baseline_params,
                "variant_value": variant_params,
                "ratio": None,
                "matched": False,
            })
            report["advisories"].append(
                f"{mode} 的 n_params 相对 local_base 的倍率"
                "超出可表示的 float 范围：不能归因于延迟合并，"
                "请核对参数记录/单位是否损坏"
            )
            continue
        parameter_matched = (
            parameter_ratio < _COST_RATIO_MATERIAL
        )
        report["capacity_parity"].append({
            "delayed_mode": mode,
            "field": "n_params",
            "baseline_value": baseline_params,
            "variant_value": variant_params,
            "ratio": parameter_ratio,
            "matched": parameter_matched,
        })
        if not parameter_matched:
            report["advisories"].append(
                f"{mode} 的 n_params 是 local_base 的 "
                f"{parameter_ratio:.2f} 倍：等 FLOPs 也不等于"
                "等容量，主效应混入参数容量差异，不能归因"
                "到延迟合并"
            )

    # A 的第二轮宽度是 k1 个分支各路由 k2 个专家，而
    # local_base 的单分支宽度被截断到 min(n_experts, k1*k2)。
    # k1*k2 超过专家总数时，两者不再严格等计算；即使实测
    # FLOPs 倍率未达到 1.5×，也不能把差值全部归因于延迟合并。
    for mode in ("A",):
        if mode not in rows_by_mode or "local_base" not in rows_by_mode:
            continue
        row = rows_by_mode[mode][0]
        k1_value = row.get("k1")
        k2_value = row.get("k2")
        n_experts_value = row.get("n_experts")
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in (k1_value, k2_value, n_experts_value)
        ):
            continue
        if int(k1_value) * int(k2_value) > int(n_experts_value):
            report["width_confounds"].append({
                "delayed_mode": mode,
                "k1_times_k2": int(k1_value) * int(k2_value),
                "n_experts": int(n_experts_value),
            })
            report["advisories"].append(
                "A 的第二轮专家宽度 k1×k2="
                f"{int(k1_value) * int(k2_value)} 超过专家总数 "
                f"n_experts={int(n_experts_value)}：local_base 的"
                "第二轮宽度被截断，两者不再严格等计算，"
                "A vs local_base 的差值混入计算量差异"
            )

    if "B" in modes_present and "smelt" in modes_present:
        report["advisories"].append(
            "B 在第一轮按 k1 个分支执行后续 Attention/MoE，"
            "与 smelt 直接比较会把计算量差异混入路由机制："
            "应同时报告等计算的 A vs local_base 对照"
            "（或按 FLOPs 对齐比较）"
        )

    if declared is not None:
        expected_delayed = list(declared)
    elif (
        "mode" in factor_set
        and any(
            mode in modes_present
            for mode in ("local_base", *_DELAYED_MERGE_MODES)
        )
    ):
        # 未显式声明时不能只按实际出现的方案推断：项目声明了
        # A 与 B 两个延迟合并创新点，只跑 A 的对比必须点名
        # 缺少 B，否则部分覆盖会被误当成完整消融。
        expected_delayed = list(_DELAYED_MERGE_MODES)
    else:
        expected_delayed = [
            mode
            for mode in _DELAYED_MERGE_MODES
            if mode in modes_present
        ]

    ablation_in_scope = any(
        mode in modes_present for mode in ablation_related
    )
    undeclared_project_modes = [
        mode
        for mode in _DELAYED_MERGE_MODES
        if ablation_in_scope and mode not in expected_delayed
    ]
    report["expected_delayed_modes"] = expected_delayed
    report["undeclared_project_modes"] = undeclared_project_modes
    report["partial"] = bool(undeclared_project_modes)
    if undeclared_project_modes:
        report["advisories"].append(
            "本轮未覆盖项目声明的延迟合并方案 "
            + ", ".join(undeclared_project_modes)
            + "：结论只能支持已覆盖方案，不能支撑"
            "「全部创新点各自有用」。"
            + (
                "已显式允许部分消融（--allow-partial-ablation），"
                "报告会保留 partial 标记。"
                if allow_partial
                else "请补齐这些方案，或显式声明为部分消融。"
            )
        )

    if (
        rows
        and declared is None
        and "local_base" not in modes_present
        and not expected_delayed
    ):
        # 运行集里没有任何延迟合并相关模式（无 local_base/A/B）：
        # 这不是「消融缺失」，而是当前对比与延迟合并消融无关
        # （如 smelt 对比统计基线），给出提醒但不虚报缺失组合。
        report["advisories"].append(
            "当前运行集不含 local_base/A/B，消融覆盖不适用；"
            "若要验证延迟合并，需补充立即合并对照 "
            "local_base 与延迟合并方案 A/B（全因子另加 "
            "balance 开关两档）"
        )
        return report

    if "mode" in factor_set:
        if "local_base" not in modes_present:
            report["missing_cells"].append(
                "缺少立即合并对照模式 local_base："
                "无法隔离延迟合并本身带来的收益"
            )
        for mode in expected_delayed:
            if mode not in modes_present:
                report["missing_cells"].append(
                    f"缺少延迟合并方案 {mode} 的运行："
                    "无法证明该方案各自有用"
                )

    expected_cells = [
        f"{mode}/b={level}"
        for mode in ("local_base", *expected_delayed)
        for level in ("off", "on")
    ]
    report["expected_cells"] = expected_cells

    # 每个消融单元的计划内失败率：过滤后的均值/配对检验只覆盖
    # 幸存运行。若某格失败更多，该格的均值条件在「没有崩掉」
    # 上，组件效应会混入幸存者偏差；失败率只在单元间不平衡时
    # 阻塞归因，完全平衡的失败仍单独披露。
    expected_pairs = [
        (mode, level)
        for mode in ("local_base", *expected_delayed)
        for level in ("off", "on")
    ]
    attrition_entries: List[Dict[str, object]] = []
    for mode, level in expected_pairs:
        cell_key = (mode, level)
        attempted = attempted_cells.get(cell_key, 0)
        if attempted <= 0:
            continue
        failed = failed_cells.get(cell_key, 0)
        attrition_entries.append({
            "cell": f"{mode}/b={level}",
            "mode": mode,
            "level": level,
            "attempted_runs": attempted,
            "failed_runs": failed,
            "usable_runs": merge_pair_cells.get(cell_key, 0),
            "failure_rate": float(failed) / float(attempted),
            "failed_run_names": list(
                failed_runs_by_cell.get(cell_key, [])
            ),
        })
    report["numerical_attrition"] = attrition_entries
    failure_rates = {
        str(entry["cell"]): float(entry["failure_rate"])
        for entry in attrition_entries
    }
    if any(rate > 0.0 for rate in failure_rates.values()):
        unbalanced = len(set(failure_rates.values())) > 1
        details = "; ".join(
            f"{entry['cell']}={entry['failed_runs']}/"
            f"{entry['attempted_runs']}"
            for entry in attrition_entries
        )
        if unbalanced:
            affected_modes = sorted({
                str(entry["mode"])
                for entry in attrition_entries
                if float(entry["failure_rate"]) > 0.0
            })
            report["attrition_confounds"].append({
                "failure_rates": failure_rates,
                "affected_modes": affected_modes,
                "details": details,
            })
            report["advisories"].append(
                "数值失败/发散的运行在消融单元间不平衡（"
                + details
                + "）：被剔除的失败运行不是方法效果，但失败率"
                "本身可能是方法稳定性差异。仅比较幸存运行会把"
                "「某臂更容易崩」误读成该臂更好，主效应/交互"
                "不能归因到组件"
            )
        else:
            report["advisories"].append(
                "数值失败/发散运行在各消融单元间数量相同（"
                + details
                + "）：失败率已平衡，但单元均值仍只覆盖幸存"
                "运行；结论应同时报告失败率，不能只看均值"
            )

    # 全因子共享种子：每个方案的 2×2 内部对齐还不够，A、B 与
    # local_base 之间也必须使用同一组种子。否则 A 的组件效应与
    # B 的组件效应是在不同种子子集上估计的，「各创新点各自
    # 有用」就不再是同一批重复实验里的联合结论。
    global_alignment: Dict[str, object] = {
        "applicable": False,
        "cells": expected_cells,
        "all_expected_present": False,
        "complete_seed_records": True,
        "common_seeds": [],
        "n_common_seeds": 0,
        "aligned": False,
    }
    if (
        "balance_coef" in factor_set
        and "mode" in factor_set
    ):
        expected_pairs = [
            (mode, level)
            for mode in ("local_base", *expected_delayed)
            for level in ("off", "on")
        ]
        missing_pairs = [
            pair for pair in expected_pairs
            if pair not in merge_pair_cells
        ]
        seed_sets: List[set] = []
        raw_by_key: Dict[object, object] = {}
        complete_records = True
        for pair in expected_pairs:
            seeds = merge_pair_seeds.get(pair, [])
            if pair in merge_pair_cells and (
                len(seeds) != merge_pair_cells[pair]
            ):
                complete_records = False
            seed_set = {
                _stable_factor_key(seed) for seed in seeds
            }
            seed_sets.append(seed_set)
            for seed in seeds:
                raw_by_key.setdefault(
                    _stable_factor_key(seed), seed
                )
        common = (
            set.intersection(*seed_sets)
            if seed_sets
            else set()
        )
        all_present = not missing_pairs
        aligned = bool(
            all_present
            and complete_records
            and all(seed_set == common for seed_set in seed_sets)
        )
        common_raw = sorted(
            (raw_by_key[key] for key in common),
            key=_factor_sort_rank,
        )
        global_alignment = {
            "applicable": True,
            "cells": expected_cells,
            "all_expected_present": all_present,
            "missing_cells": [
                f"{mode}/b={level}"
                for mode, level in missing_pairs
            ],
            "complete_seed_records": complete_records,
            "common_seeds": common_raw,
            "n_common_seeds": len(common),
            "aligned": aligned,
        }
        if all_present and not complete_records:
            report["advisories"].append(
                "全因子存在未记录随机种子的运行：无法核对各格"
                "是否共享同一组种子，主效应/交互不能按同种子"
                "配对解读"
            )
        if all_present and len(common) < _MIN_SEEDS_FOR_STATS:
            report["low_replicate_cells"].append(
                "全因子各格共同种子只有 "
                f"{len(common)} 个（< {_MIN_SEEDS_FOR_STATS}）"
                "：方案之间的组件效应不再来自同一批重复实验，"
                "不能支撑「各创新点各自有用」"
            )
        elif all_present and not aligned:
            report["advisories"].append(
                "全因子各格的随机种子集合不一致："
                f"共同种子 {len(common)} 个；A/B/local_base "
                "的组件效应只在共同子集上可配对，未对齐的种子"
                "不能当作额外重复"
            )
    report["global_seed_alignment"] = global_alignment

    if (
        "balance_coef" in factor_set
        and "mode" in factor_set
    ):
        # (1 + N)×2 全部格子的存在性（两轴都在因子中时）。
        for mode in ("local_base", *expected_delayed):
            for level in ("off", "on"):
                key = (mode, level)
                if key not in merge_pair_cells:
                    report["missing_cells"].append(
                        f"缺少全因子组合 {mode}/b={level}"
                        "（无法证明各创新点各自有用）"
                    )
    elif "balance_coef" in factor_set:
        # 只有 balance_coef 是因子：按本项目的延迟合并模式
        # 检查开启臂 / 关闭臂。
        for mode in ("local_base", *expected_delayed):
            if mode not in modes_present:
                continue
            balances = balance_by_mode.get(mode, [])
            known = [b for b in balances if b is not None]
            if not known:
                report["low_replicate_cells"].append(
                    f"{mode} 未记录 balance_coef"
                )
                continue
            if 0 not in known:
                report["missing_cells"].append(
                    f"{mode} 缺少 balance_coef=0 的关闭臂："
                    "无法证明均衡损失各自有用"
                )
            if not any(
                isinstance(b, (int, float))
                and not isinstance(b, bool)
                and b > 0
                for b in known
            ):
                report["missing_cells"].append(
                    f"{mode} 缺少 balance_coef>0 的开启臂"
                )

    for (mode, level), count in sorted(merge_pair_cells.items()):
        label = f"{mode}/b={level}"
        report["present_cells"].append({
            "cell": label,
            "n_runs": count,
        })
        independent = _independent_seed_count(
            count,
            merge_pair_seeds.get((mode, level), []),
        )
        if independent < _RECOMMENDED_SEEDS:
            report["low_replicate_cells"].append(
                f"{label} 只有 {independent} 个种子"
                f"（建议 ≥{_RECOMMENDED_SEEDS}）"
            )

    expected_cell_count = 2 * (1 + len(expected_delayed))
    if (
        "balance_coef" in factor_set
        and "mode" in factor_set
        and len(merge_pair_cells) == expected_cell_count
        and not report["missing_cells"]
        and not report["structural_confounds"]
        and not report["partial"]
    ):
        report["advisories"].append(
            f"{1 + len(expected_delayed)}×2 全因子组合齐全且"
            "结构设置一致：可同时给出各延迟合并方案与均衡损失"
            "的主效应及交互"
        )

    return report



# ------------------------------------------------------------
# 基线合理性与假设纪律检查（对比分析阶段）
# ------------------------------------------------------------

# 无上下文参考基线（unigram）在各对比指标上的对应字段名。
# 它们由每个运行在训练开始时用 train 统计量在 val 固定窗口上
# 算出并随 config.json 的 reference_baselines 存档，用于核对
# 「所声明的基线是否真的强于无上下文预测器」。没有对应参考的
# 指标（如 PR-AUC / ROC-AUC）只提示无法核对，不虚报通过。
_TRIVIAL_REFERENCE_FIELDS: Dict[str, str] = {
    "val_loss": "unigram_loss",
    # 等计算量 loss 与 val_loss 同口径（同一验证窗口、
    # 同一 NLL 单位），只是取自不同的训练计算量点；无上下文
    # unigram 参考仍然是它的稻草人下界。
    "val_loss_at_matched_flops": "unigram_loss",
    "val_loss_at_common_step": "unigram_loss",
    # 时均验证损失与 val_loss 同单位：unigram 的验证曲线是
    # 常数，因此其窗口均值仍是同一个 unigram_loss 稻草人。
    _CURVE_MEAN_METRIC: "unigram_loss",
    "val_ppl": "unigram_loss",
    "val_bits_per_byte": "unigram_loss",
    # 类平衡似然：稻草人参考是同口径的 unigram macro NLL
    # （对出现类等权平均每类 -log p(c)）。
    "val_macro_nll": "unigram_macro_nll",
    "val_token_accuracy": "unigram_token_accuracy",
    "val_micro_f1": "unigram_token_accuracy",
    "val_macro_f1": "unigram_macro_f1",
    "val_weighted_f1": "unigram_weighted_f1",
    "val_precision_macro": "unigram_precision_macro",
    "val_precision_weighted": "unigram_precision_weighted",
    "val_recall_macro": "unigram_recall_macro",
    "val_recall_weighted": "unigram_recall_weighted",
    # 排序指标同样要有稻草人守卫：PR-AUC 的常数预测器水平 =
    # 纳入类平均 prevalence（unigram_pr_auc_macro_chance），
    # ROC-AUC 的常数预测器水平 = 0.5。否则按项目自己的建议
    # 切到 PR-AUC 对比时，基线合理性检查会静默降级为
    # 「无法核对」。旧运行缺这些参考字段时按缺记录处理。
    "val_pr_auc_macro": "unigram_pr_auc_macro_chance",
    # lift 的随机水平恒为 0（常数预测器的 AP = prevalence），
    # 与具体运行/评估集无关；仍登记参考口径，基线守卫才不会
    # 在切到 lift 主指标时静默降级为「无法核对」。
    "val_pr_auc_macro_lift": "unigram_pr_auc_macro_lift",
    "val_roc_auc_macro": "unigram_roc_auc_macro",
}


# 共同步数派生指标与原始周期指标同单位，稻草人参考沿用同一
# unigram 字段（例如 val_macro_f1_at_common_step 对应
# unigram_macro_f1）；否则基线守卫会在派生指标上静默失效。
for _base_metric in _COMMON_STEP_BASE_METRICS:
    _derived_metric = _common_step_metric_name(_base_metric)
    if (
        _derived_metric is not None
        and _derived_metric not in _TRIVIAL_REFERENCE_FIELDS
    ):
        _TRIVIAL_REFERENCE_FIELDS[_derived_metric] = (
            _TRIVIAL_REFERENCE_FIELDS[_base_metric]
        )


# 类别不平衡提示阈值：最高频类的 support 占比达到第一个值时，
# accuracy / micro 口径已被「永远预测多数类」的常数预测器
# 主导；超过第二个值时 accuracy 的参考价值已经有限；达到
# 第三个值（多数类 ≈ 99.9%，约 1:1000 以上的不平衡）时排序
# 判定只能看 PR-AUC——ROC-AUC 的 FPR 分母由海量负样本决定，
# 随机水平仍是 0.5，会显著高估可用性，而随机 PR-AUC 只有
# 约 1/1000，两者不可直接比较。
_IMBALANCE_EXTREME_MAJORITY = 0.999

_IMBALANCE_DOMINANT_MAJORITY = 0.9

_IMBALANCE_WEAK_MAJORITY = 0.5



def _trivial_reference_value(
    metric: str,
    reference: Optional[Dict[str, object]],
) -> Optional[float]:
    """
    取 unigram 参考基线在指定指标上的同口径值，供「基线是否
    强于稻草人」的核对使用；没有对应参考口径、参考缺失或
    非有限时返回 None。
    """
    if not isinstance(reference, dict):
        return None

    field = _TRIVIAL_REFERENCE_FIELDS.get(metric)
    if field is None:
        return None

    base_metric = (
        _common_step_metric_base(metric) or metric
    )
    if base_metric == "val_pr_auc_macro_lift":
        # PR-AUC lift 的随机水平定义为 0；旧运行参考记录里
        # 没有该字段也应能完成稻草人核对。
        return 0.0

    # 参考基线来自 config.json：JSON 允许 10**400 这类超出
    # float 范围的整数。按不可用参考处理，而不是让 float()
    # 在基线合理性核对里抛 OverflowError。
    value = _finite_float(reference.get(field))
    if value is None:
        return None

    # unigram_loss 换算到其他口径时与 evaluate / 曲线诊断一致；
    # 共同步数派生指标沿用其原始指标的换算口径。
    if base_metric == "val_ppl":
        return math.exp(min(value, 80.0))
    if base_metric == "val_bits_per_byte":
        return value / math.log(2.0)
    return value



def _baseline_tuning_seed_count(
    seed_plan: object,
) -> Optional[int]:
    """
    读取调参报告 seed_plan 中的独立种子数。

    缺失、bool 或非整数都返回 None，表示无法核对统计功效；
    调用方不能把「字段缺失」当作「种子数足够」。
    """
    if not isinstance(seed_plan, dict):
        return None

    count = seed_plan.get("n_seeds")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return None



def _baseline_tuning_seed_values(
    record: Dict[str, object],
) -> List[int]:
    """
    读取调参报告/来源记录里的调参种子列表。

    新版报告在 ``seed_plan.seeds`` 保存完整种子集合，旧版在
    顶层 ``seeds``；两者都缺失时返回空列表，表示无法核对
    调参种子与最终评估种子是否重合。只接受整数（排除 bool），
    不做字符串猜测。
    """
    candidates: List[object] = []
    seed_plan = record.get("seed_plan")
    if isinstance(seed_plan, dict):
        candidates = seed_plan.get("seeds") or []
    if not candidates:
        candidates = record.get("seeds") or []
    if not isinstance(candidates, list):
        return []
    values: List[int] = []
    for seed in candidates:
        if isinstance(seed, int) and not isinstance(seed, bool):
            values.append(int(seed))
    return values



def tuning_seed_overlap_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    核对「调参时用于选 winner 的种子」与「最终评估种子」是否
    重合。

    同一批随机种子先被用来选择学习率/其他超参数，又用来评估
    最终基线或挑战臂，会把 winner's curse 带进对比：被选中的
    配置在这批种子上本来就更占优，后续评估不再独立。调参仍可
    证明「配置被调过」，但该运行的指标不能当作独立验证。

    返回结构化报告；没有调参记录，或存在无法取得调参种子/
    最终评估种子的运行且未发现重合时 ``ok=None``（不虚报通过）；
    发现重合时 ``ok=False``。
    """
    entries: List[Dict[str, object]] = []
    unverifiable_runs: List[Dict[str, object]] = []
    advisories: List[str] = []
    with_tuning = 0
    verifiable = 0
    overlapping = 0
    for row in rows:
        record = row.get("baseline_tuning")
        if not isinstance(record, dict):
            continue
        with_tuning += 1
        run_name = str(row.get("run"))
        tuning_seeds = _baseline_tuning_seed_values(record)
        evaluation_seed = row.get("seed")
        evaluation_seed_valid = isinstance(
            evaluation_seed, int
        ) and not isinstance(evaluation_seed, bool)
        if not tuning_seeds or not evaluation_seed_valid:
            if not tuning_seeds and not evaluation_seed_valid:
                reason = (
                    "调参记录缺少可解析的种子列表，且运行缺少"
                    "可解析的最终评估种子"
                )
            elif not tuning_seeds:
                reason = "调参记录缺少可解析的种子列表"
            else:
                reason = "运行缺少可解析的最终评估种子"
            unverifiable_runs.append(
                {
                    "run": run_name,
                    "mode": row.get("mode"),
                    "reason": reason,
                }
            )
            advisories.append(
                f"{run_name} 的调参种子独立性无法核对"
                f"（{reason}）：不能假设调参种子与最终评估"
                "种子不相交"
            )
            continue
        verifiable += 1
        overlap = sorted(set(tuning_seeds) & {int(evaluation_seed)})
        if not overlap:
            continue
        overlapping += 1
        entry = {
            "run": run_name,
            "mode": row.get("mode"),
            "evaluation_seed": int(evaluation_seed),
            "tuning_seeds": tuning_seeds,
            "overlap": overlap,
        }
        entries.append(entry)
        advisories.append(
            f"{entry['run']} 的最终评估种子 "
            f"{entry['evaluation_seed']} 也参与了调参 winner 的"
            f"选择（调参种子 {tuning_seeds}）：同一批种子既选"
            "超参数又评估最终结果，winner's curse 会让该运行"
            "指标偏乐观，不能当作独立验证；请把调参种子与"
            "评估种子分开（如调参用 7/8/9，最终基线用 "
            "42/43/44）"
        )

    report: Dict[str, object] = {
        "n_runs_with_tuning": with_tuning,
        "n_verifiable_runs": verifiable,
        "n_overlapping_runs": overlapping,
        "n_unverifiable_runs": len(unverifiable_runs),
        "overlap_share": (
            overlapping / verifiable if verifiable else None
        ),
        "entries": entries,
        "unverifiable_runs": unverifiable_runs,
        "advisories": advisories,
        "ok": (
            None
            if with_tuning == 0
            else (
                False
                if overlapping
                else (None if unverifiable_runs else True)
            )
        ),
    }
    return report



def _baseline_tuning_axis_error(
    record: Dict[str, object],
) -> Optional[str]:
    """
    核对坐标式多轴调参报告是否覆盖了每个声明的轴。

    某个轴没有有效 winner 时，推荐配置只会回退到该轴的基线值：
    报告仍可能因为学习率轴有 winner 而被读回，但缺失轴并没有
    被调过参。调用方不能把这种不完整报告当成「基线已充分调参」
    的证据。单轴旧报告没有 axes 字段，返回 None 保持兼容。

    完整报告的 ``axes`` 含逐轴 winner；写入 config.json 的紧凑
    来源记录不重复保存完整轴记录，只保存 ``axis_winners`` 与
    ``recommended_config``——此时逐个核对推荐配置里的轴是否都有
    winner，避免紧凑记录绕过同一检查。
    """
    multi_axis = (
        record.get("multi_axis") is True
        or record.get("design") == "coordinate"
    )
    axis_reports = record.get("axes")
    if axis_reports is not None:
        if not isinstance(axis_reports, list):
            return "调参报告的 axes 格式非法：无法核对每个轴是否都有 winner"
        if not axis_reports:
            if multi_axis:
                return "调参报告声明为多轴设计但 axes 为空：无法核对调参轴"
            return None
        for axis_report in axis_reports:
            if not isinstance(axis_report, dict):
                return "调参报告的轴记录格式非法：无法核对每个轴是否都有 winner"
            key = axis_report.get("protocol_key")
            if not isinstance(key, str) or not key:
                return "调参报告存在缺少 protocol_key 的轴记录：无法核对调参轴"
            axis_winner = axis_report.get("winner")
            if (
                not isinstance(axis_winner, dict)
                or axis_winner.get("value") is None
            ):
                label = str(axis_report.get("label", key))
                return (
                    f"调参报告的轴 {label}（{key}）没有有效 winner："
                    "该轴没有可用的候选结果，推荐配置只是回退到"
                    "基线值，不能作为该轴已调参的证据"
                )
        return None

    if not multi_axis:
        return None

    axis_winners = record.get("axis_winners")
    recommended = record.get("recommended_config")
    if not isinstance(axis_winners, dict) or not axis_winners:
        return "调参报告声明为多轴设计但缺少 axis_winners：无法核对调参轴"
    if not isinstance(recommended, dict) or not recommended:
        return (
            "多轴调参来源缺少 recommended_config：无法核对推荐配置"
            "中的每个轴是否都有 winner"
        )
    missing = [
        str(name)
        for name in recommended
        if name not in axis_winners
        or axis_winners.get(name) is None
    ]
    if missing:
        return (
            "多轴调参来源缺少轴 "
            + ", ".join(missing)
            + " 的有效 winner：推荐配置在该轴只是回退到基线值，"
            "该轴实际没有调参证据"
        )
    return None
