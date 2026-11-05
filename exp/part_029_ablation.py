from __future__ import annotations



def factorial_effects(
    aggregates: List[Dict[str, object]],
    group_keys: List[str],
    metric: str,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    全因子消融的主效应、交互与简单效应。

    对每个延迟合并方案 M ∈ {A, B}，与立即合并对照 L =
    ``local_base`` 组成 2×2 设计（mode ∈ {L, M} × balance ∈
    {off, on}）：

    - 延迟合并主效应 = ½(M_off − L_off) + ½(M_on − L_on)；
    - 均衡损失主效应 = ½(L_on − L_off) + ½(M_on − M_off)；
    - 交互 = (M_on − M_off) − (L_on − L_off)；
    - 四个简单效应用于解释交互：M−L 在 off/on 两档、
      on−off 在 L/M 两档。

    所有效应都在共同随机种子上按线性组合后做配对 t 检验，
    并在家族内做 Holm 校正。只有校正后仍显著且方向符合
    指标改善方向时，才把主效应标记为支持该组件；交互显著
    只说明「效应随另一因子变化」，不能自动解释为有益。

    ``equivalence_margin`` 给定时（相对参考单元均值的比例），
    每个效应同时做 TOST 等价性检验：参考均值取该效应所比较
    的对照单元均值（主效应取两侧对照的均值，简单效应取该
    水平的对照格）。等价 p 值在同一效应家族内做 Holm/BH
    校正；``equivalent`` 只在校正后仍显著时成立。

    ``mode`` 或 ``balance_coef`` 不是分组因子（恒定/未变化）时
    返回 ``applicable=False``；组合不完整时把设计列入
    ``incomplete_designs``，不伪造主效应。同一上下文出现多个
    正 ``balance_coef`` 时不能折叠成单一开启臂，设计列入
    ``confounded_designs`` 并跳过效应估计。
    """
    report: Dict[str, object] = {
        "metric": metric,
        "applicable": (
            "mode" in group_keys and "balance_coef" in group_keys
        ),
        "designs": [],
        "incomplete_designs": [],
        "confounded_designs": [],
        "effects": [],
        "advisories": [],
    }

    if not report["applicable"]:
        report["advisories"].append(
            "mode 与 balance_coef 未同时作为分组因子："
            "无法估计延迟合并 × 均衡损失的主效应与交互"
        )
        return report

    context_keys = [
        key
        for key in group_keys
        if key not in ("mode", "balance_coef")
    ]
    contexts: Dict[tuple, List[Dict[str, object]]] = {}
    context_order: List[tuple] = []
    for aggregate in aggregates:
        context = tuple(
            _stable_factor_key(
                aggregate["keys"].get(key)
            )
            for key in context_keys
        )
        if context not in contexts:
            contexts[context] = []
            context_order.append(context)
        contexts[context].append(aggregate)

    higher_is_better = metric_higher_is_better(metric)
    effect_specs = [
        (
            "delayed_merge_main",
            "延迟合并主效应",
            {
                "L_off": -0.5,
                "L_on": -0.5,
                "M_off": 0.5,
                "M_on": 0.5,
            },
            "main",
        ),
        (
            "balance_loss_main",
            "均衡损失主效应",
            {
                "L_off": -0.5,
                "L_on": 0.5,
                "M_off": -0.5,
                "M_on": 0.5,
            },
            "main",
        ),
        (
            "interaction",
            "延迟合并 × 均衡损失交互",
            {
                "L_off": 1.0,
                "L_on": -1.0,
                "M_off": -1.0,
                "M_on": 1.0,
            },
            "interaction",
        ),
        (
            "delayed_merge_at_balance_off",
            "仅延迟合并（balance=off）",
            {"L_off": -1.0, "M_off": 1.0},
            "simple",
        ),
        (
            "delayed_merge_at_balance_on",
            "仅延迟合并（balance=on）",
            {"L_on": -1.0, "M_on": 1.0},
            "simple",
        ),
        (
            "balance_loss_at_local_base",
            "仅均衡损失（立即合并对照）",
            {"L_off": -1.0, "L_on": 1.0},
            "simple",
        ),
        (
            "balance_loss_at_delayed_merge",
            "仅均衡损失（延迟合并方案）",
            {"M_off": -1.0, "M_on": 1.0},
            "simple",
        ),
    ]

    # TOST 的等价界是相对界，必须给出每条效应所比较的对照
    # 均值作为尺度：主效应取两侧对照格，简单效应用对应水平的
    # 对照格。这样 ±margin 在不同效应上保持同一可解释口径。
    reference_aliases = {
        "delayed_merge_main": ("L_off", "L_on"),
        "balance_loss_main": ("L_off", "M_off"),
        "interaction": ("L_off", "L_on", "M_off", "M_on"),
        "delayed_merge_at_balance_off": ("L_off",),
        "delayed_merge_at_balance_on": ("L_on",),
        "balance_loss_at_local_base": ("L_off",),
        "balance_loss_at_delayed_merge": ("M_off",),
    }

    for context in context_order:
        cells_by_key: Dict[tuple, Dict[str, object]] = {}
        positive_balances: Dict[object, float] = {}
        for aggregate in contexts[context]:
            mode = str(aggregate["keys"].get("mode"))
            if mode not in ("local_base", *_DELAYED_MERGE_MODES):
                continue
            balance = _finite_float(
                aggregate["keys"].get("balance_coef")
            )
            if balance is None:
                continue
            if balance > 0.0:
                positive_balances[
                    _stable_factor_key(balance)
                ] = balance
            level = "off" if balance == 0.0 else "on"
            cells_by_key[(mode, level)] = aggregate

        # 「开启臂」必须只有一个明确的均衡强度。若同一上下文里
        # 同时出现多个正 balance_coef，把它们的均值折叠成一个
        # "on" 档会让主效应混入均衡强度差异，而且 cells_by_key
        # 还会用最后出现的单元格静默覆盖前面的值。这里宁可不
        # 估计，也不输出口径错误的数字。
        if len(positive_balances) > 1:
            details = {
                "context": dict(
                    zip(context_keys, context)
                ),
                "positive_balance_values": sorted(
                    positive_balances.values()
                ),
            }
            report["confounded_designs"].append(details)
            report["advisories"].append(
                "消融上下文 "
                + (
                    ", ".join(
                        f"{key}={_format_factor_value(value)}"
                        for key, value in zip(
                            context_keys, context
                        )
                    )
                    or "(无额外上下文)"
                )
                + " 出现多个正 balance_coef 取值（"
                + ", ".join(
                    format(value, "g")
                    for value in sorted(
                        positive_balances.values()
                    )
                )
                + "）：不能把它们折叠成同一个开启臂，"
                "未估计该上下文的 2×2 主效应/交互；"
                "请统一均衡强度后重跑"
            )
            continue

        present_modes = {
            mode
            for mode, _ in cells_by_key
        }
        for mode in _DELAYED_MERGE_MODES:
            required = {
                ("local_base", "off"): "L_off",
                ("local_base", "on"): "L_on",
                (mode, "off"): "M_off",
                (mode, "on"): "M_on",
            }
            missing = [
                f"{cell_mode}/{cell_level}"
                for (cell_mode, cell_level) in required
                if (cell_mode, cell_level) not in cells_by_key
            ]
            if missing and (mode in present_modes):
                report["incomplete_designs"].append({
                    "delayed_mode": mode,
                    "context": dict(zip(context_keys, context)),
                    "missing_cells": missing,
                })
                continue
            if missing:
                continue

            cells = {
                alias: cells_by_key[cell_key]
                for cell_key, alias in required.items()
            }
            design_label = (
                f"{mode} vs local_base"
                + (
                    "（"
                    + ", ".join(
                        f"{key}={_format_factor_value(value)}"
                        for key, value in zip(context_keys, context)
                    )
                    + "）"
                    if context_keys
                    else ""
                )
            )
            report["designs"].append({
                "delayed_mode": mode,
                "context": dict(zip(context_keys, context)),
                "labels": {
                    alias: cell["label"]
                    for alias, cell in cells.items()
                },
            })

            for effect_name, description, coefficients, kind in (
                effect_specs
            ):
                result = _paired_linear_effect(
                    cells,
                    coefficients,
                    metric,
                )
                effect = result.get("effect")
                if effect is None:
                    continue
                effect_value = float(effect)
                mde = None
                if (
                    result.get("se") is not None
                    and result.get("df") is not None
                ):
                    mde = _minimum_detectable_effect(
                        float(result["se"]),
                        float(result["df"]),
                    )
                if kind == "interaction":
                    direction = (
                        "positive"
                        if effect_value > 0.0
                        else "negative"
                        if effect_value < 0.0
                        else "zero"
                    )
                else:
                    improvement = (
                        effect_value
                        if higher_is_better
                        else -effect_value
                    )
                    direction = (
                        "improvement"
                        if improvement > 0.0
                        else "regression"
                        if improvement < 0.0
                        else "tie"
                    )
                reference_values = []
                for alias in reference_aliases.get(
                    effect_name,
                    (),
                ):
                    mean = _finite_float(
                        cells[alias].get(f"{metric}_mean")
                    )
                    if mean is not None:
                        reference_values.append(mean)
                # 有限参考均值直接求和仍可能溢出（如 +1e308
                # 与 -(-1e308)）；用带缩放的均值保留真实相消。
                reference_mean = _mean(reference_values)

                effect_record = {
                    "design": design_label,
                    "delayed_mode": mode,
                    "context": dict(zip(context_keys, context)),
                    "effect_name": effect_name,
                    "description": description,
                    "kind": kind,
                    # 与设计阶段功效口径对接：主效应方差是两两
                    # 对比的 0.5 倍、交互是 2 倍、简单效应相同。
                    # 分析时仍用该效应自己的实测 se，不看乘数；
                    # 记录该字段是为了让报告能核对「设计时按哪种
                    # 对比准备种子」与「现在报的是哪种对比」。
                    "contrast_type": (
                        kind
                        if kind in ("main", "interaction", "simple")
                        else "pairwise"
                    ),
                    "contrast_variance_factor": (
                        _CONTRAST_VARIANCE_FACTORS.get(kind)
                    ),
                    "coefficients": coefficients,
                    "cells": {
                        alias: cell["label"]
                        for alias, cell in cells.items()
                    },
                    "mean_effect": effect_value,
                    "delta": effect_value,
                    "paired_differences": result.get(
                        "paired_differences"
                    ),
                    "paired_seeds": result.get("paired_seeds"),
                    "direction": direction,
                    "mde_80": mde,
                    "test": result.get("test"),
                    "t": result.get("t"),
                    "df": result.get("df"),
                    "se": result.get("se"),
                    "p_value": result.get("p_value"),
                    "randomization_p_value": result.get(
                        "randomization_p_value"
                    ),
                    "randomization_method": result.get(
                        "randomization_method"
                    ),
                    "randomization_permutations": result.get(
                        "randomization_permutations"
                    ),
                    "randomization_min_attainable_p": result.get(
                        "randomization_min_attainable_p"
                    ),
                    "randomization_can_reach_significance": result.get(
                        "randomization_can_reach_significance"
                    ),
                    "randomization_required_seeds": result.get(
                        "randomization_required_seeds"
                    ),
                    # 与两两对比同口径：线性组合效应的差值分布
                    # 诊断与 Wilcoxon 对照必须随效应一起落盘，
                    # 不能只出现在单因子家族。
                    "paired_diagnostics": result.get(
                        "paired_diagnostics"
                    ),
                    "wilcoxon_p_value": result.get(
                        "wilcoxon_p_value"
                    ),
                    "wilcoxon_method": result.get(
                        "wilcoxon_method"
                    ),
                    "wilcoxon_rank_biserial": result.get(
                        "wilcoxon_rank_biserial"
                    ),
                    "wilcoxon_min_attainable_p": result.get(
                        "wilcoxon_min_attainable_p"
                    ),
                    "wilcoxon_can_reach_significance": result.get(
                        "wilcoxon_can_reach_significance"
                    ),
                    "cohens_dz": result.get("cohens_dz"),
                    "hedges_gz": result.get("hedges_gz"),
                    "cohens_dz_ci95": result.get(
                        "cohens_dz_ci95"
                    ),
                    "hedges_gz_ci95": result.get(
                        "hedges_gz_ci95"
                    ),
                    "leave_one_out_effect_min": result.get(
                        "leave_one_out_effect_min"
                    ),
                    "leave_one_out_effect_max": result.get(
                        "leave_one_out_effect_max"
                    ),
                    "leave_one_out_sign_stable": result.get(
                        "leave_one_out_sign_stable"
                    ),
                    "n_pairs": result.get("n_pairs", 0),
                    "aligned_seeds": bool(
                        result.get("aligned_seeds")
                    ),
                    "common_seeds": result.get("common_seeds", []),
                    "complete_seed_records": bool(
                        result.get("complete_seed_records")
                    ),
                    "duplicate_seeds": result.get(
                        "duplicate_seeds", []
                    ),
                }
                effect_record.update(
                    _contrast_interval(result)
                )
                if reference_mean is not None:
                    effect_record["mean_baseline"] = (
                        reference_mean
                    )
                    _attach_power_equivalence(
                        effect_record,
                        effect_value,
                        result,
                        equivalence_margin,
                    )
                # 增强等价性报告：计算 equivalence_margin_needed 与
                # equivalence_blocker，明确「效应超出界限」与「精度不足」的区别。
                se = result.get("se")
                df = result.get("df")
                baseline_mean = _finite_float(
                    effect_record.get("mean_baseline")
                )
                baseline_abs = None
                if baseline_mean is not None and abs(baseline_mean) > 0.0:
                    baseline_abs = abs(baseline_mean)
                if se is not None and df is not None and baseline_abs is not None:
                    try:
                        se_v = float(se)
                        df_v = float(df)
                        bl_abs = float(baseline_abs)
                    except (TypeError, ValueError, OverflowError):
                        se_v = df_v = bl_abs = None
                    if (
                        equivalence_margin is not None
                        and se_v is not None
                        and df_v is not None
                        and se_v > 0
                        and bl_abs > 0
                    ):
                        alpha = _SIGNIFICANCE_LEVEL
                        t_crit = _t_critical(df_v, alpha)
                        if t_crit is not None:
                            contrast_mag = abs(effect_value)
                            equivalence_margin_needed = (
                                contrast_mag + t_crit * se_v
                            )
                            effect_record["equivalence_margin_needed"] = (
                                equivalence_margin_needed
                            )
                            margin_rel = float(equivalence_margin)
                            margin_abs = margin_rel * baseline_abs
                            if not math.isfinite(margin_abs):
                                effect_record["equivalence_blocker"] = (
                                    "margin_abs_infinite"
                                )
                            elif contrast_mag > margin_abs:
                                effect_record[
                                    "equivalence_blocker"
                                ] = "effect_exceeds_margin"
                            else:
                                sd_observed = _observed_seed_sd(
                                    effect_record,
                                    se_v,
                                )
                                if sd_observed is not None and effect_value != 0.0:
                                    design = (
                                        "welch"
                                        if effect_record.get(
                                            "test"
                                        )
                                        == "welch_t"
                                        else "paired"
                                    )
                                    seeds_needed = (
                                        _seeds_to_detect(
                                            effect_value,
                                            sd_observed,
                                            design=design,
                                        )
                                    )
                                    if seeds_needed is not None:
                                        delta_seeds = (
                                            seeds_needed
                                            - int(
                                                result.get(
                                                    "n_pairs", 0
                                                )
                                            )
                                        )
                                        if delta_seeds > 0:
                                            effect_record[
                                                "equivalence_blocker"
                                            ] = f"need_{delta_seeds}_more_seeds"
                                        else:
                                            effect_record[
                                                "equivalence_blocker"
                                            ] = None
                                    else:
                                        effect_record[
                                            "equivalence_blocker"
                                        ] = "cannot_compute_seeds"
                                else:
                                    effect_record[
                                        "equivalence_blocker"
                                    ] = None
                        else:
                            effect_record[
                                "equivalence_margin_needed"
                            ] = None
                            effect_record[
                                "equivalence_blocker"
                            ] = None
                else:
                    effect_record["equivalence_margin_needed"] = None
                    effect_record["equivalence_blocker"] = None
                # 全因子主效应/交互/简单效应同样只覆盖完整案例。
                # 每个单元的缺失运行数量可能不同；把缺失值推到
                # 观测支持的两端，检查该线性组合的方向是否仍
                # 成立，避免某个更容易失败的消融格伪装成主效应。
                effect_record["attrition_bounds"] = (
                    _attrition_bounds(
                        cells,
                        coefficients,
                        metric,
                        effect_value,
                    )
                )
                report["effects"].append(effect_record)

    eligible = [
        effect
        for effect in report["effects"]
        if effect.get("p_value") is not None
    ]
    # 主效应/交互家族的功效口径同步按家族规模校正（与下方
    # Holm 判定同口径），见 _attach_family_power。
    effect_p_values = [
        float(effect["p_value"]) for effect in eligible
    ]
    effect_thresholds = _holm_rank_thresholds(effect_p_values)
    for effect, holm_threshold in zip(
        eligible, effect_thresholds
    ):
        _attach_family_power(effect, len(eligible))
        effect["holm_alpha_threshold"] = holm_threshold
    holm = _holm_bonferroni(effect_p_values)
    for effect, adjusted in zip(eligible, holm):
        effect["p_value_holm"] = adjusted
        effect["significant_raw"] = bool(
            float(effect["p_value"]) <= _SIGNIFICANCE_LEVEL
        )
        corrected = bool(
            adjusted is not None
            and float(adjusted) <= _SIGNIFICANCE_LEVEL
        )
        effect["significant"] = corrected
        if not corrected:
            effect["claim"] = "not_significant"
        elif effect["kind"] == "interaction":
            effect["claim"] = "interaction_detected"
        elif effect["direction"] == "improvement":
            effect["claim"] = "supports_component"
        elif effect["direction"] == "regression":
            effect["claim"] = "harms_component"
        else:
            effect["claim"] = "not_significant"

        # 与单因子对比同口径：Holm step-down 失败后的位置
        # 阈值不再有效，非显著效应不能进入 stable/fragile 计数。
        paired_values = effect.get("paired_differences")
        if isinstance(paired_values, list):
            threshold = effect.get("holm_alpha_threshold")
            fragility = _paired_result_fragility(
                paired_values,
                effect.get("paired_seeds"),
                alpha=float(
                    threshold
                    if threshold is not None
                    else effect.get(
                        "alpha_bonferroni",
                        _SIGNIFICANCE_LEVEL,
                    )
                ),
            )
            if (
                not corrected
                and fragility.get("available")
                and fragility.get("verdict") != "not_significant"
            ):
                fragility["verdict"] = "not_significant"
                fragility["holm_step_down_valid"] = False
            effect["fragility"] = fragility

    # 因子效应与单因子对比同口径：附 Bonferroni 同时区间，
    # 避免「逐检验 95% CI 不含 0」被误读成校正后显著的结论。
    _attach_family_adjusted_intervals(eligible)

    # 符号翻转检验与 t 检验同家族校正；两者结论不一致时
    # 主效应/交互的显著性依赖分布假设，不能作为稳健结论。
    randomization_eligible = [
        item
        for item in report["effects"]
        if item.get("randomization_p_value") is not None
    ]
    randomization_holm = _holm_bonferroni([
        float(item["randomization_p_value"])
        for item in randomization_eligible
    ])
    for item, randomization_adjusted in zip(
        randomization_eligible,
        randomization_holm,
    ):
        item["randomization_p_value_holm"] = (
            randomization_adjusted
        )
        item["randomization_significant"] = bool(
            randomization_adjusted is not None
            and float(randomization_adjusted)
            <= _SIGNIFICANCE_LEVEL
        )
        item["significant_robust"] = bool(
            item.get("significant")
            and item.get("randomization_significant")
        )
        item["significant_robust_possible"] = bool(
            item.get(
                "randomization_can_reach_significance", True
            )
            is not False
        )

    # 等价性主张与显著性同属一个推断家族：多个主效应/交互
    # 同时声称「落在等价界内」也会放大误判概率。TOST p 值做
    # 同口径的 Holm/BH 校正，只有校正后仍 <= alpha 的
    # ``equivalent`` 才能作为「无实质差异」的证据。
    tost_eligible = [
        item
        for item in report["effects"]
        if item.get("tost_p_value") is not None
    ]
    tost_holm = _holm_bonferroni([
        float(item["tost_p_value"])
        for item in tost_eligible
    ])
    tost_bh = _benjamini_hochberg([
        float(item["tost_p_value"])
        for item in tost_eligible
    ])
    for item, holm_p, bh_p in zip(
        tost_eligible,
        tost_holm,
        tost_bh,
    ):
        item["tost_p_value_holm"] = holm_p
        item["tost_p_value_bh"] = bh_p
        item["equivalent_raw"] = bool(
            item.get("equivalent")
        )
        item["equivalent"] = bool(
            holm_p is not None
            and float(holm_p) <= _SIGNIFICANCE_LEVEL
        )
    for effect in report["effects"]:
        if (
            effect.get("claim") == "supports_component"
            and effect.get("equivalent")
        ):
            effect["claim"] = "not_practically_significant"

    for effect in report["effects"]:
        if effect.get("test") == "no_common_seeds":
            report["advisories"].append(
                f"{effect['design']} 的 {effect['description']}"
                "没有共同随机种子：只给出点估计，未做配对"
                "检验，不能据此声称主效应/交互显著"
            )
        elif not effect.get("aligned_seeds"):
            report["advisories"].append(
                f"{effect['design']} 的 {effect['description']}"
                "种子集合不完全对齐：仅在共同种子上估计，"
                "结论只覆盖该子集"
            )
        if effect.get("n_pairs", 0) < _MIN_SEEDS_FOR_STATS:
            report["advisories"].append(
                f"{effect['design']} 的 {effect['description']}"
                f"只有 {effect.get('n_pairs', 0)} 对共同种子"
                f"（< {_MIN_SEEDS_FOR_STATS}）：统计功效不足"
            )
        if effect.get("duplicate_seeds"):
            report["advisories"].append(
                f"{effect['design']} 的 {effect['description']}"
                "所在单元存在重复随机种子（"
                + ", ".join(
                    str(seed)
                    for seed in effect["duplicate_seeds"]
                )
                + "）：复制运行已折叠为一个观测，不能算独立重复"
            )
        if not effect.get("complete_seed_records", True):
            report["advisories"].append(
                f"{effect['design']} 的 {effect['description']}"
                "所在单元存在未记录随机种子的运行：配对检验"
                "只覆盖有种子且共同的子集，效应不能外推到该"
                "单元的全部运行"
            )

    # 缺失敏感性：某个消融格更容易数值失败时，完整案例的
    # 主效应/交互可能由「哪些格留下了」驱动。这里把每个效应
    # 的缺失支持界汇总到因子层，跨 0 的效应不能作为组件证据。
    attrition_sensitivity = _attrition_bounds_summary(
        report["effects"], label="factorial"
    )
    report["attrition_sensitivity"] = attrition_sensitivity
    for message in attrition_sensitivity["advisories"]:
        report["advisories"].append(message)

    if report["incomplete_designs"]:
        report["advisories"].append(
            "存在不完整的 2×2 设计：缺格时不能估计对应"
            "方案的主效应或交互"
        )

    return report



# 每个延迟合并方案的单因素机制对照（mechanism control）。
#
# 「延迟合并有用」是机制归因主张：只有把合并时机/合并位置
# 之外的设置都固定住，才能把收益归因到延迟合并机制本身。
# (1+N)×2 全因子里的 local_base 对照承担不了这个角色：
#
# - A vs local_base：local_base 的第二轮宽度是 min(E, k1·k2)
#   （等专家计算量口径），A 是「每分支 k2」，差值同时含第二轮
#   路由宽度差异；
# - B vs local_base：B 的分支贯穿整个中间区（分支化 Attention/
#   MoE），local_base 是局部立即合并，两者连合并作用域都不同。
#
# 项目已经提供只改一个变量的机制对照：
#
# - A 的对照 local_base_k2：第二轮宽度同为 k2、作用域同为局部，
#   只改变合并时机；
# - B 的对照 B_early：分支数、Attention/MoE 执行次数与 B 完全
#   相同，只改变合并位置。
#
# 缺少这两个单因素对照时，2×2 主效应只能说明「方案整体（含
# 宽度/作用域差异）更好」，不能宣称收益来自延迟合并机制。
_DELAYED_MERGE_MECHANISM_CONTROLS: Dict[str, str] = {
    "A": "local_base_k2",
    "B": "B_early",
}



# 机制对照的两臂之间只允许差「被检验的那一个因子」
# （mode：合并时机/位置）。除它之外的协议项必须逐项一致，
# 否则这次对比就不是单因素实验。
#
# k1/k2/balance_coef/dense_fnn_hidden 默认是实验因子
# （跨运行允许不同），受控变量检查不会为它们生成违规记录；
# 但机制对照恰恰要求它们在两臂之间固定，因此这里单独按
# comparison_protocol 的键逐项核对，不能依赖「声明因子」
# 的默认行为。
_MECHANISM_PROTOCOL_EXCLUDED_KEYS: Tuple[str, ...] = ("mode",)



def _mechanism_row_protocol_value(
    row: Dict[str, object],
    key: str,
) -> Tuple[object, bool]:
    """
    取运行在某个协议键上的取值，以及该取值是否被记录。

    旧运行没有 comparison_protocol 时，协议值由回退逻辑补齐，
    但「补齐后仍为 None」表示该键根本没有记录：显式 null
    （如 cosine 调度下的 wsd_stable_steps）是已记录的可比值，
    与「未记录」必须区分。
    """
    protocol = row.get("protocol")
    if not isinstance(protocol, dict):
        return None, False
    value = protocol.get(key)
    recorded = row.get("protocol_recorded_keys")
    if isinstance(recorded, (list, tuple, set, frozenset)):
        explicit = str(key) in {str(item) for item in recorded}
    else:
        explicit = value is not None
    return value, explicit



def _mechanism_single_factor_check(
    rows_baseline: List[Dict[str, object]],
    rows_variant: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    核对两臂是否「只差一个因子」：除 mode 外的协议项逐项一致。

    返回结构：

    - ``performed``：是否拿到了可核对的运行记录；
    - ``differences``：两臂之间（或同一臂内部）实际不同的
      协议项——这是多因素对比，必须点名；
    - ``unverifiable``：一侧记录缺失、无法确认一致的协议项；
    - ``unchecked_keys``：两侧都没有记录、无法核对的协议项
      （不算违规，但报告必须说明核对范围有限）。

    两侧都没有记录的键不判为差异：旧运行缺记录是「未核对」，
    不是「不一致」；但一侧有、另一侧没有时无法确认一致，
    记入 ``unverifiable``，不能当作单因素证据。
    """
    report: Dict[str, object] = {
        "performed": False,
        "ok": True,
        "checked_keys": [],
        "differences": [],
        "unverifiable": [],
        "unchecked_keys": [],
    }
    if not rows_baseline or not rows_variant:
        report["ok"] = False
        report["unverifiable"].append({
            "key": "comparison_protocol",
            "reason": "missing_runs",
            "missing_runs": [],
            "note": (
                "没有可核对的运行记录：无法确认两臂除被检验"
                "因子外逐项一致"
            ),
        })
        return report
    report["performed"] = True

    for key in COMPARISON_PROTOCOL_KEYS:
        if key in _MECHANISM_PROTOCOL_EXCLUDED_KEYS:
            continue
        sides: List[Tuple[Dict[str, object], List[str]]] = []
        for side_rows in (rows_baseline, rows_variant):
            values: Dict[str, object] = {}
            missing: List[str] = []
            for row in side_rows:
                value, explicit = _mechanism_row_protocol_value(
                    row, key
                )
                name = str(row.get("run"))
                if value is None and not explicit:
                    missing.append(name)
                    continue
                values[name] = value
            sides.append((values, missing))
        values_baseline, missing_baseline = sides[0]
        values_variant, missing_variant = sides[1]

        if not values_baseline and not values_variant:
            report["unchecked_keys"].append(key)
            continue
        if missing_baseline or missing_variant:
            report["unverifiable"].append({
                "key": key,
                "reason": "missing_records",
                "missing_runs": sorted(
                    set(missing_baseline) | set(missing_variant)
                ),
                "note": (
                    "该协议项在一侧没有记录：无法确认两臂一致，"
                    "不能当作单因素证据"
                ),
            })
            continue

        distinct_baseline = {
            _stable_factor_key(value)
            for value in values_baseline.values()
        }
        distinct_variant = {
            _stable_factor_key(value)
            for value in values_variant.values()
        }
        if (
            len(distinct_baseline) > 1
            or len(distinct_variant) > 1
        ):
            report["differences"].append({
                "key": key,
                "reason": "within_arm_variation",
                "baseline_values": sorted(
                    str(value)
                    for value in {
                        _stable_factor_key(value)
                        for value in values_baseline.values()
                    }
                ),
                "variant_values": sorted(
                    str(value)
                    for value in {
                        _stable_factor_key(value)
                        for value in values_variant.values()
                    }
                ),
                "note": (
                    "同一臂内部的该协议项取值不一致：两臂不是"
                    "同一个实验设置，不能当作单因素对比"
                ),
            })
            continue
        if distinct_baseline != distinct_variant:
            report["differences"].append({
                "key": key,
                "reason": "across_arm_difference",
                "baseline_values": sorted(
                    str(value)
                    for value in {
                        _stable_factor_key(value)
                        for value in values_baseline.values()
                    }
                ),
                "variant_values": sorted(
                    str(value)
                    for value in {
                        _stable_factor_key(value)
                        for value in values_variant.values()
                    }
                ),
                "note": (
                    "两臂在该协议项上不同：对比同时改变了多个"
                    "因子，不是单因素证据"
                ),
            })
            continue
        report["checked_keys"].append(key)

    report["ok"] = (
        not report["differences"]
        and not report["unverifiable"]
    )
    # 附加已检查键的汇总，供报告展示
    if report["checked_keys"]:
        report["checked_keys_summary"] = ", ".join(
            str(k) for k in report["checked_keys"]
        )
    return report



def _mechanism_cell_pairs(
    delayed_cells: List[Dict[str, object]],
    control_cells: List[Dict[str, object]],
    group_keys: List[str],
) -> Tuple[
    List[Tuple[Dict[str, object], Dict[str, object]]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """
    把延迟合并方案与其机制对照的实验单元配成「只差 mode」的
    可比单元对。

    - ``comparable``：除 mode 外其余分组键取值相同的单元对，
      每个对都是候选的单因素对比；
    - ``unmatched``：某个分组键在两臂之间没有共同取值
      （例如对照臂换了 k2），此时两臂之间不存在单因素可比
      单元，实际差异必然是多因素的，必须点名；
    - ``partial``：两臂在该键上只有部分共同取值（对照臂没有
      覆盖方案的全部水平）。可比单元仍然成立，但报告要说明
      对照覆盖不完整。

    分组键取值不同的单元对不是「混杂」，只是不同的实验单元
    （例如均衡开关的两档）；只有当两臂在某个键上完全没有
    共同取值时，才说明它们无法构成单因素对比。
    """
    comparable: List[
        Tuple[Dict[str, object], Dict[str, object]]
    ] = []
    for delayed_cell in delayed_cells:
        for control_cell in control_cells:
            differing = [
                name
                for name in group_keys
                if name != "mode"
                and _stable_factor_key(
                    delayed_cell["keys"].get(name)
                ) != _stable_factor_key(
                    control_cell["keys"].get(name)
                )
            ]
            if not differing:
                comparable.append(
                    (delayed_cell, control_cell)
                )
    unmatched: List[Dict[str, object]] = []
    partial: List[Dict[str, object]] = []
    for name in group_keys:
        if name == "mode":
            continue
        delayed_values = {
            _stable_factor_key(cell["keys"].get(name))
            for cell in delayed_cells
        }
        control_values = {
            _stable_factor_key(cell["keys"].get(name))
            for cell in control_cells
        }
        if not delayed_values or not control_values:
            continue
        shared = delayed_values & control_values
        if not shared:
            unmatched.append({
                "factor": name,
                "delayed_values": sorted(
                    str(value) for value in delayed_values
                ),
                "control_values": sorted(
                    str(value) for value in control_values
                ),
            })
        elif delayed_values != control_values:
            partial.append({
                "factor": name,
                "delayed_values": sorted(
                    str(value) for value in delayed_values
                ),
                "control_values": sorted(
                    str(value) for value in control_values
                ),
                "shared_values": sorted(
                    str(value) for value in shared
                ),
            })
    return comparable, unmatched, partial



def _mechanism_cell_rows(
    cell: Dict[str, object],
    rows_by_run: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    取一个实验单元对应的运行记录（按 config.json 里的运行名）。

    聚合单元只保存运行名列表；协议逐项核对需要运行本身的
    comparison_protocol。找不到记录时返回空列表，调用方据此
    判定「无法核对」，而不是当成「一致」。
    """
    runs = cell.get("runs")
    if not isinstance(runs, (list, tuple, set, frozenset)):
        return []
    found: List[Dict[str, object]] = []
    for run in runs:
        row = rows_by_run.get(str(run))
        if row is not None:
            found.append(row)
    return found



def _mechanism_check_summary(
    checks: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    汇总各可比单元对的单因素核对结果。

    ``performed`` 为 False 表示没有运行记录可核对（例如只拿到
    聚合结果）；此时单因素性质只按分组键判断，报告必须说明
    核对范围有限。
    """
    summary: Dict[str, object] = {
        "performed": False,
        "ok": None,
        "n_pairs": len(checks),
        "checked_keys": [],
        "differences": [],
        "unverifiable": [],
        "unchecked_keys": [],
    }
    if not checks:
        summary["note"] = (
            "没有运行记录：两臂是否只差一个因子只能按分组键核对"
        )
        return summary
    performed = [
        check for check in checks if check.get("performed")
    ]
    summary["performed"] = bool(performed)
    if not performed:
        summary["note"] = (
            "没有运行记录：两臂是否只差一个因子只能按分组键核对"
        )
        return summary
    for check in performed:
        for key in check.get("checked_keys") or []:
            if key not in summary["checked_keys"]:
                summary["checked_keys"].append(str(key))
        for key in check.get("unchecked_keys") or []:
            if key not in summary["unchecked_keys"]:
                summary["unchecked_keys"].append(str(key))
        summary["differences"].extend(
            check.get("differences") or []
        )
        summary["unverifiable"].extend(
            check.get("unverifiable") or []
        )
    summary["ok"] = (
        not summary["differences"]
        and not summary["unverifiable"]
    )
    return summary



def _mechanism_contrast_direction(
    contrast: Dict[str, object],
    delayed_mode: str,
) -> Optional[bool]:
    """
    机制对比里「延迟合并方案是否更好」。

    返回 True/False 表示延迟合并方案相对机制对照更好/更差；
    方向无法判定时返回 None。对比的 baseline/variant 顺序由
    因子的自然顺序决定（B 排在 B_early 之前），不能假定变体
    一定是延迟合并方案。
    """
    baseline_value = contrast.get("baseline_value")
    variant_value = contrast.get("variant_value")
    if baseline_value is None or variant_value is None:
        return None
    verdict = str(contrast.get("verdict"))
    if verdict not in ("improvement", "regression"):
        return None
    delayed_is_variant = str(variant_value) == str(delayed_mode)
    delayed_is_baseline = str(baseline_value) == str(delayed_mode)
    if not (delayed_is_variant or delayed_is_baseline):
        return None
    improvement = verdict == "improvement"
    if delayed_is_variant:
        return improvement
    return not improvement



def mechanism_control_report(
    aggregates: List[Dict[str, object]],
    group_keys: List[str],
    metric: str,
    equivalence_margin: Optional[float] = None,
    controlled_variables: Optional[Dict[str, object]] = None,
    expected_modes: Optional[List[str]] = None,
    rows: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    单因素机制对照检查：延迟合并的归因必须由单因素对照支撑。

    对每个延迟合并方案 M，报告它与指定机制对照 C
    （A→local_base_k2，B→B_early）之间「只改一个因子」的
    对比：两臂必须在其余因子（均衡开关、上下文）上完全一致，
    否则该对比不是单因素证据。机制对照单独成族做 Holm/BH
    校正——它是预先指定的关键对照，不与全部两两对比混在
    同一个家族里。

    单因素性质由三道核对共同保证：

    1. **可比单元**：两臂的实验单元必须除 mode 外逐项相同
       （同一均衡档、同一 k1/k2…）。没有共同取值时该对比是
       多因素的（``confounded``）。
    2. **受控变量**：``check_controlled_variables`` 的违规若
       落在参与对比的运行上，该对不是单因素对比。
    3. **协议逐项一致**（需要 ``rows``）：除 mode 外的全部
       comparison_protocol 键在两臂之间必须一致，且不得有
       一侧缺记录。k1/k2/balance_coef 默认是实验因子、受控
       变量检查不会为它们生成违规，因此这一步不能省略。

    每个方案的 status：

    - ``missing_control``：运行集里没有机制对照臂；
    - ``confounded``：对照臂存在，但该对比同时改变了其它
      因子、协议项不一致或受控变量检查不通过；
    - ``unverifiable``：协议项缺记录，无法确认两臂只差一个
      因子；
    - ``insufficient_data``：两臂没有可比的实验单元；
    - ``insufficient_seeds``：共同种子不足，做不了配对检验；
    - ``supported``：校正后显著且延迟合并方案更好；
    - ``contradicted``：校正后显著且延迟合并方案更差；
    - ``partial``：部分水平支持、部分水平变差；
    - ``not_supported``：对比存在但没有支持证据。
    """
    report: Dict[str, object] = {
        "metric": metric,
        "applicable": False,
        "schemes": {},
        "family_size": 0,
        "advisories": [],
        "single_factor_verified": False,
    }
    rows_by_run: Dict[str, Dict[str, object]] = {}
    if isinstance(rows, (list, tuple)):
        for row in rows:
            if isinstance(row, dict):
                rows_by_run[str(row.get("run"))] = row
    modes = [
        str(mode)
        for mode in (expected_modes or _DELAYED_MERGE_MODES)
    ]
    present_modes = {
        str(aggregate["keys"].get("mode"))
        for aggregate in aggregates
        if isinstance(aggregate.get("keys"), dict)
    }
    relevant = [
        mode for mode in modes if mode in present_modes
    ]
    if not relevant:
        report["advisories"].append(
            "运行集里没有延迟合并方案（"
            + ", ".join(modes)
            + "）：无需检查单因素机制对照"
        )
        return report
    report["applicable"] = True

    family: List[Dict[str, object]] = []
    for mode in relevant:
        control = _DELAYED_MERGE_MECHANISM_CONTROLS.get(mode)
        entry: Dict[str, object] = {
            "delayed_mode": mode,
            "control_mode": control,
            "status": "missing_control",
            "reason": "",
            "contrasts": [],
            "n_contrasts": 0,
            "confounded_pairs": [],
            "unverifiable_pairs": [],
            "partial_levels": [],
            "single_factor_check": {
                "performed": False,
                "ok": None,
                "note": "没有运行记录：只能按分组键核对",
            },
        }
        if control is None:
            entry["status"] = "no_control_defined"
            entry["reason"] = (
                f"{mode} 没有登记单因素机制对照：无法把收益"
                "归因到延迟合并机制本身"
            )
            report["schemes"][mode] = entry
            continue
        if control not in present_modes:
            entry["reason"] = (
                f"缺少单因素机制对照 {control}（只改变合并时机/"
                f"位置的对照臂）：{mode} vs local_base 的多因素"
                "差异不能替代单因素证据"
            )
            report["schemes"][mode] = entry
            continue
        delayed_cells = [
            aggregate
            for aggregate in aggregates
            if str(aggregate["keys"].get("mode")) == mode
        ]
        control_cells = [
            aggregate
            for aggregate in aggregates
            if str(aggregate["keys"].get("mode")) == control
        ]
        comparable, unmatched, partial = _mechanism_cell_pairs(
            delayed_cells, control_cells, group_keys
        )
        entry["comparable_cells"] = len(comparable)
        entry["unmatched_levels"] = unmatched
        entry["partial_levels"] = partial
        # 单因素核对：核对两臂是否除 mode 外其余协议项一致
        single_factor = _mechanism_single_factor_check(
            [
                row for row in control_cells
                if str(row.get("mode")) == control
            ],
            [
                row for row in delayed_cells
                if str(row.get("mode")) == mode
            ],
        )
        entry["single_factor_check"] = {
            "performed": single_factor.get("performed", False),
            "ok": single_factor.get("ok"),
            "checked_keys": single_factor.get("checked_keys", []),
            "differences": single_factor.get("differences", []),
            "unverifiable": single_factor.get("unverifiable", []),
            "unchecked_keys": single_factor.get("unchecked_keys", []),
        }
        if single_factor.get("performed") and single_factor.get("ok"):
            entry["single_factor_verified"] = True
        if not comparable:
            if unmatched:
                entry["status"] = "confounded"
                entry["reason"] = (
                    f"{mode} 与机制对照 {control} 在 "
                    + "、".join(
                        str(item.get("factor"))
                        for item in unmatched
                    )
                    + " 上没有共同取值：两臂之间不存在只差一个"
                    "因子的可比单元，该对比是多因素差异"
                )
            else:
                entry["status"] = "insufficient_data"
                entry["reason"] = (
                    f"{mode} 与机制对照 {control} 之间没有可比的"
                    "实验单元（除被检验因子外其余设置完全一致）"
                )
            report["schemes"][mode] = entry
            continue

        mechanism_contrasts: List[Dict[str, object]] = []
        confounded_pairs: List[Dict[str, object]] = []
        unverifiable_pairs: List[Dict[str, object]] = []
        checks: List[Dict[str, object]] = []
        for delayed_cell, control_cell in comparable:
            pair_contrasts = factor_contrasts(
                [delayed_cell, control_cell],
                ["mode"],
                metric,
                equivalence_margin=equivalence_margin,
                controlled_variables=controlled_variables,
            )
            contrast = (
                pair_contrasts[0] if pair_contrasts else None
            )
            if contrast is None:
                continue
            if not contrast.get("factor"):
                # 受控变量检查不通过：这一对不是单因素对比。
                confounded_pairs.append({
                    "delayed_cell": delayed_cell["label"],
                    "control_cell": control_cell["label"],
                    "reason": "controlled_variables",
                    "differing_factors": list(
                        contrast.get("differing_factors") or []
                    ),
                    "controlled_confound_keys": list(
                        contrast.get("controlled_confound_keys")
                        or []
                    ),
                    "note": (
                        "受控变量检查不通过：参与对比的运行之间"
                        "还有其它协议项不同"
                    ),
                })
                continue
            check = _mechanism_single_factor_check(
                _mechanism_cell_rows(delayed_cell, rows_by_run),
                _mechanism_cell_rows(control_cell, rows_by_run),
            )
            checks.append(check)
            if check.get("performed") and not check.get("ok"):
                differences = list(check.get("differences") or [])
                unverifiable = list(check.get("unverifiable") or [])
                record = {
                    "delayed_cell": delayed_cell["label"],
                    "control_cell": control_cell["label"],
                    "reason": (
                        "protocol_mismatch"
                        if differences
                        else "protocol_unverifiable"
                    ),
                    "differences": differences,
                    "unverifiable": unverifiable,
                }
                if differences:
                    record["note"] = (
                        "两臂除被检验因子外还有协议项不同：不是"
                        "单因素证据"
                    )
                    confounded_pairs.append(record)
                else:
                    record["note"] = (
                        "协议项缺记录，无法确认两臂只差一个因子："
                        "不能当作单因素证据"
                    )
                    unverifiable_pairs.append(record)
                continue
            mechanism_contrasts.append(contrast)
        entry["contrasts"] = mechanism_contrasts
        entry["n_contrasts"] = len(mechanism_contrasts)
        entry["confounded_pairs"] = confounded_pairs
        entry["unverifiable_pairs"] = unverifiable_pairs
        entry["single_factor_check"] = _mechanism_check_summary(
            checks
        )
        family.extend(mechanism_contrasts)
        report["schemes"][mode] = entry

    if family:
        _apply_multiple_comparison_correction(family)
        report["family_size"] = len(family)

    for mode, entry in report["schemes"].items():
        if not isinstance(entry, dict):
            continue
        contrasts = entry.get("contrasts") or []
        confounded_pairs = entry.get("confounded_pairs") or []
        unverifiable_pairs = entry.get("unverifiable_pairs") or []
        if confounded_pairs:
            reasons = []
            for pair in confounded_pairs:
                reasons.append(
                    str(pair.get("note") or pair.get("reason"))
                )
            entry["status"] = "confounded"
            entry["reason"] = (
                f"{mode} 与机制对照 {entry.get('control_mode')} 的"
                "单因素对比不成立（" + "；".join(reasons)
                + "）：两臂不是只差一个因子，不能把收益归因到"
                "延迟合并机制"
            )
            continue
        if unverifiable_pairs:
            reasons = []
            for pair in unverifiable_pairs:
                reasons.append(
                    str(pair.get("note") or pair.get("reason"))
                )
            entry["status"] = "unverifiable"
            entry["reason"] = (
                f"{mode} 与机制对照 {entry.get('control_mode')} 的"
                "单因素性质无法核对（" + "；".join(reasons)
                + "）：不能把收益归因到延迟合并机制"
            )
            continue
        if not contrasts:
            if str(entry.get("status")) in {
                "missing_control",
                "no_control_defined",
                "confounded",
            }:
                continue
            entry["status"] = "insufficient_data"
            entry["reason"] = (
                f"{mode} 与机制对照 {entry.get('control_mode')} "
                "之间没有可用的单因素对比"
            )
            continue
        usable = [
            contrast
            for contrast in contrasts
            if isinstance(contrast.get("n_pairs"), int)
            and not isinstance(contrast.get("n_pairs"), bool)
            and int(contrast["n_pairs"]) >= _MIN_SEEDS_FOR_STATS
        ]
        if not usable:
            entry["status"] = "insufficient_seeds"
            entry["reason"] = (
                f"{mode} 与机制对照 {entry.get('control_mode')} "
                "的共同种子不足 "
                f"（< {_MIN_SEEDS_FOR_STATS}）：做不了配对检验"
            )
            continue
        supports: List[Dict[str, object]] = []
        harms: List[Dict[str, object]] = []
        for contrast in usable:
            direction = _mechanism_contrast_direction(
                contrast, mode
            )
            if direction is None:
                continue
            if not contrast.get("significant"):
                continue
            if direction:
                supports.append(contrast)
            else:
                harms.append(contrast)
        if supports and harms:
            entry["status"] = "partial"
            entry["reason"] = (
                f"{mode} 相对机制对照 {entry.get('control_mode')} "
                "在不同水平上方向不一致：不能宣称延迟合并机制"
                "在所有设置下有用"
            )
        elif harms:
            entry["status"] = "contradicted"
            entry["reason"] = (
                f"{mode} 相对机制对照 {entry.get('control_mode')} "
                "校正后显著更差：延迟合并机制本身没有带来收益"
            )
        elif supports:
            entry["status"] = "supported"
            entry["reason"] = (
                f"{mode} 相对机制对照 {entry.get('control_mode')} "
                "的单因素对比校正后显著且方向改善"
            )
        else:
            entry["status"] = "not_supported"
            entry["reason"] = (
                f"{mode} 相对机制对照 {entry.get('control_mode')} "
                "的单因素对比没有支持证据（未达校正后显著）"
            )
        for contrast in entry["contrasts"]:
            direction = _mechanism_contrast_direction(
                contrast, mode
            )
            contrast["delayed_mode"] = mode
            contrast["control_mode"] = entry.get("control_mode")
            contrast["delayed_better"] = direction
            contrast["delta_delayed_minus_control"] = (
                contrast.get("delta")
                if str(contrast.get("variant_value")) == mode
                else (
                    -float(contrast["delta"])
                    if isinstance(
                        contrast.get("delta"), (int, float)
                    )
                    and not isinstance(
                        contrast.get("delta"), bool
                    )
                    else None
                )
            )
    for mode, entry in report["schemes"].items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status")) in {
            "missing_control",
            "no_control_defined",
            "confounded",
            "unverifiable",
            "insufficient_data",
            "insufficient_seeds",
        }:
            report["advisories"].append(
                str(entry.get("reason"))
            )
        for item in entry.get("partial_levels") or []:
            report["advisories"].append(
                f"{mode} 的机制对照 {entry.get('control_mode')} 在 "
                f"{item.get('factor')} 上没有覆盖方案的全部取值"
                f"（方案 {item.get('delayed_values')}，对照 "
                f"{item.get('control_values')}）：单因素对比只在"
                "共同取值上成立"
            )
    report["single_factor_verified"] = all(
        str(entry.get("status"))
        in {"supported", "contradicted", "partial", "not_supported"}
        for entry in report["schemes"].values()
        if isinstance(entry, dict)
    ) if report["schemes"] else False
    return report



# 因子效应出现这些 claim 时，统计显著也不能归因到组件本身：
# 要么受控变量/结构/计算量/同种子随机性混杂，要么控制项根本
# 没记录。组件级联合判定必须把这些情况集中列为 blocker。
_COMPONENT_BLOCKING_CLAIMS = frozenset({
    "confounded",
    "compute_confounded",
    "capacity_confounded",
    "randomness_confounded",
    "attrition_confounded",
    "balance_confounded",
    "unverified_controls",
})



def _component_evidence_summary(
    effects: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    汇总一组因子效应对某个组件的支持程度。

    返回 ``supported`` / ``partial`` / ``contradicted`` /
    ``blocked`` / ``inconclusive`` / ``insufficient_data`` /
    ``missing``。只有所有可用证据都在 Holm 校正后显著、方向
    改善，且没有落在预注册等价界内，才返回 ``supported``；
    任一显著变差即为 ``contradicted``；出现混杂 claim 时一律
    ``blocked``。统计显著但 TOST 已判定为实质等价的效应不能
    当作组件有用的证据：否则极小幅度的“显著”改进会被包装成
    创新点有效。
    """
    if not effects:
        return {
            "status": "missing",
            "reason": "没有该组件的效应估计（组合缺失或指标不可用）",
        }

    claims = [str(effect.get("claim")) for effect in effects]
    summaries = [
        {
            "effect_name": effect.get("effect_name"),
            "delayed_mode": effect.get("delayed_mode"),
            "claim": effect.get("claim"),
            "mean_effect": effect.get("mean_effect"),
            "p_value_holm": effect.get("p_value_holm"),
            "significant": effect.get("significant"),
            "equivalent": effect.get("equivalent"),
            "n_pairs": effect.get("n_pairs", 0),
            "fragility_verdict": (
                _as_dict(effect.get("fragility")).get("verdict")
            ),
            "attrition_verdict": (
                _as_dict(effect.get("attrition_bounds")).get(
                    "verdict"
                )
            ),
            "attrition_sign_robust": (
                _as_dict(effect.get("attrition_bounds")).get(
                    "sign_robust"
                )
            ),
        }
        for effect in effects
    ]
    fragile_effects = [
        summary
        for summary in summaries
        if summary.get("fragility_verdict") == "fragile"
    ]
    attrition_unstable = [
        summary
        for summary in summaries
        if summary.get("attrition_verdict")
        in {"crosses_zero", "unbounded"}
        or summary.get("attrition_sign_robust") is False
    ]

    if any(claim in _COMPONENT_BLOCKING_CLAIMS for claim in claims):
        return {
            "status": "blocked",
            "reason": "效应被混杂/未核对控制项阻塞，不能归因到组件",
            "evidence": summaries,
        }
    if any(claim == "harms_component" for claim in claims):
        return {
            "status": "contradicted",
            "reason": "至少一个校正后显著的效应方向为变差",
            "evidence": summaries,
        }
    if all(claim == "supports_component" for claim in claims):
        if fragile_effects:
            return {
                "status": "inconclusive",
                "reason": (
                    "效应校正后显著，但在删除少量随机种子后"
                    "显著性消失（fragility=fragile）：结论由"
                    "少数运行驱动，不能作为组件有用的稳健证据；"
                    "应增加独立种子或调查异常种子"
                ),
                "evidence": summaries,
                "fragile_effects": fragile_effects,
            }
        if attrition_unstable:
            return {
                "status": "inconclusive",
                "reason": (
                    "效应校正后显著，但在观测支持范围内给缺失/"
                    "失败运行取极端值时方向可能反转，或界的符号"
                    "与配对点估计不一致（attrition_bounds 跨 0、"
                    "无界或 sign_robust=false）：完整案例的结论"
                    "可能由「哪些运行留下了」驱动，不能作为组件"
                    "有用的稳健证据；应补齐运行或报告失败率与"
                    "部分识别界"
                ),
                "evidence": summaries,
                "attrition_unstable_effects": attrition_unstable,
            }
        if all(effect.get("equivalent") for effect in effects):
            return {
                "status": "not_supported",
                "reason": (
                    "效应虽校正后显著，但全部落在预注册等价界内："
                    "统计上非零不等于有实质改善，不能宣称该组件有用"
                ),
                "evidence": summaries,
            }
        if any(effect.get("equivalent") for effect in effects):
            return {
                "status": "partial",
                "reason": (
                    "部分支持效应落在预注册等价界内：这些水平上"
                    "没有达到最小有意义效应，不能宣称组件在所有"
                    "设置下有用"
                ),
                "evidence": summaries,
            }
        return {
            "status": "supported",
            "reason": (
                "全部可用效应校正后显著、方向改善，且未落入"
                "预注册等价界"
            ),
            "evidence": summaries,
        }
    if any(claim == "supports_component" for claim in claims):
        return {
            "status": "partial",
            "reason": "只有部分效应支持，不能宣称组件在所有设置下有用",
            "evidence": summaries,
        }
    if all(effect.get("equivalent") for effect in effects):
        return {
            "status": "not_supported",
            "reason": (
                "效应在预注册等价界内（TOST 校正后成立）："
                "没有证据支持该组件带来实质改善"
            ),
            "evidence": summaries,
        }
    if all(
        isinstance(effect.get("n_pairs"), int)
        and not isinstance(effect.get("n_pairs"), bool)
        and effect.get("n_pairs") < _MIN_SEEDS_FOR_STATS
        for effect in effects
    ):
        return {
            "status": "insufficient_data",
            "reason": "共同种子数不足，无法判断组件是否有效",
            "evidence": summaries,
        }
    return {
        "status": "inconclusive",
        "reason": (
            "效应未达到校正后显著，且未通过等价性检验："
            "既不能宣称有用，也不能宣称无实质差异"
        ),
        "evidence": summaries,
    }



def _mechanism_entry_for_mode(
    mechanism: Optional[Dict[str, object]],
    mode: str,
) -> Optional[Dict[str, object]]:
    """从机制对照报告里取某个延迟合并方案的条目。"""
    if not isinstance(mechanism, dict):
        return None
    schemes = mechanism.get("schemes")
    if not isinstance(schemes, dict):
        return None
    entry = schemes.get(str(mode))
    if isinstance(entry, dict):
        return entry
    return None



def ablation_component_verdicts(
    factorial: Dict[str, object],
    coverage: Dict[str, object],
    metric_preregistration: Optional[Dict[str, object]] = None,
    mechanism: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    把全因子效应汇总成「每个创新点是否各自有用」的联合判定。

    主动机：主效应、简单效应与交互分散在报告各处时，读者容易
    把「部分组合显著」拼成「全部创新点都有用」。这里对每个
    延迟合并方案与均衡损失分别给出单一 verdict，并要求：

    - 主效应（跨均衡开关加权）支持该组件；
    - 简单效应没有显著变差或混杂；
    - 消融组合完整、结构/计算量/同种子随机性/数值失败率
      没有阻塞；
    - 全因子各格共享同一组随机种子（否则不同组件是在不同
      种子子集上比较）；
    - 当前分析指标与训练前预注册的主指标一致（否则在多个
      指标之间挑一个支持结论的，属于指标选择偏差）；
    - 延迟合并方案还有单因素机制对照支撑（见
      ``mechanism_control_report``）：2×2 主效应是多因素
      对比，缺少只改合并时机/位置的机制对照时不能把收益
      归因到延迟合并机制本身。

    只有全部组件均为 ``supported`` 且无 blocker 时，
    ``all_components_supported`` 才为 True；部分显著只返回
    ``partial``，不会被包装成「各自有用」。
    """
    effects = [
        effect
        for effect in factorial.get("effects", [])
        if isinstance(effect, dict)
    ]
    expected_delayed = [
        str(mode)
        for mode in (
            coverage.get("expected_delayed_modes")
            or list(_DELAYED_MERGE_MODES)
        )
    ]
    project_delayed = [
        str(mode)
        for mode in (
            coverage.get("project_delayed_modes")
            or list(_DELAYED_MERGE_MODES)
        )
    ]

    components: List[Dict[str, object]] = []
    for mode in expected_delayed:
        primary = [
            effect
            for effect in effects
            if effect.get("delayed_mode") == mode
            and effect.get("effect_name") == "delayed_merge_main"
        ]
        simple = [
            effect
            for effect in effects
            if effect.get("delayed_mode") == mode
            and effect.get("effect_name")
            in {
                "delayed_merge_at_balance_off",
                "delayed_merge_at_balance_on",
            }
        ]
        primary_summary = _component_evidence_summary(primary)
        simple_summary = _component_evidence_summary(simple)
        interaction = [
            effect
            for effect in effects
            if effect.get("delayed_mode") == mode
            and effect.get("effect_name") == "interaction"
        ]
        interaction_significant = any(
            effect.get("claim") == "interaction_detected"
            and effect.get("significant")
            for effect in interaction
        )
        status = str(primary_summary["status"])
        reasons = [str(primary_summary.get("reason"))]
        if status == "supported":
            if simple_summary["status"] == "contradicted":
                status = "contradicted"
                reasons.append(
                    "主效应支持，但至少一个均衡开关水平的简单"
                    "效应显著变差（交互），不能宣称各水平都有用"
                )
            elif simple_summary["status"] == "blocked":
                status = "blocked"
                reasons.append(
                    "主效应支持，但简单效应受混杂/未核对控制项"
                    "阻塞"
                )
            elif simple_summary.get("attrition_unstable_effects"):
                status = "inconclusive"
                reasons.append(
                    "主效应支持，但至少一个简单效应在缺失/失败"
                    "运行的支持界内方向可能反转：不能宣称组件"
                    "在所有设置下有用；应补齐运行或调查失败原因"
                )
            elif (
                interaction_significant
                and simple_summary["status"] != "supported"
            ):
                status = "partial"
                reasons.append(
                    "交互显著且简单效应未全部支持：主效应可能"
                    "只由其中一个均衡开关水平驱动，不能宣称"
                    "组件在所有设置下有用"
                )
        elif (
            status in {"missing", "insufficient_data", "inconclusive"}
            and simple_summary["status"] == "supported"
        ):
            status = "partial"
            reasons.append(
                "主效应不足，但部分简单效应支持：只算局部证据"
            )
        mechanism_entry = _mechanism_entry_for_mode(
            mechanism, mode
        )
        if mechanism_entry is not None:
            mechanism_status = str(
                mechanism_entry.get("status")
            )
            mechanism_reason = str(
                mechanism_entry.get("reason") or ""
            )
            if mechanism_status != "supported":
                if status == "supported":
                    if mechanism_status in (
                        "missing_control",
                        "confounded",
                        "no_control_defined",
                        "unverifiable",
                        "insufficient_data",
                        "insufficient_seeds",
                    ):
                        status = "blocked"
                    elif mechanism_status == "contradicted":
                        status = "contradicted"
                    else:
                        status = "partial"
                # 无论组件此前是否已有支持结论，机制对照未通过都
                # 必须点名：否则读者只看到「主效应不足」，看不出
                # 归因还缺单因素对照这一环。
                reasons.append(
                    "单因素机制对照未通过："
                    + mechanism_reason
                )
        components.append({
            "component": f"delayed_merge_{mode}",
            "label": f"延迟合并方案 {mode}",
            "status": status,
            "primary": primary_summary,
            "simple": simple_summary,
            "mechanism": mechanism_entry,
            "reasons": reasons,
        })

    balance_main = [
        effect
        for effect in effects
        if effect.get("effect_name") == "balance_loss_main"
    ]
    balance_simple = [
        effect
        for effect in effects
        if effect.get("effect_name")
        in {
            "balance_loss_at_local_base",
            "balance_loss_at_delayed_merge",
        }
    ]
    balance_primary = _component_evidence_summary(balance_main)
    balance_simple_summary = _component_evidence_summary(
        balance_simple
    )
    balance_interaction_significant = any(
        effect.get("effect_name") == "interaction"
        and effect.get("claim") == "interaction_detected"
        and effect.get("significant")
        for effect in effects
    )
    balance_status = str(balance_primary["status"])
    balance_reasons = [str(balance_primary["reason"])]
    if balance_status == "supported":
        if balance_simple_summary["status"] == "contradicted":
            balance_status = "contradicted"
            balance_reasons.append(
                "均衡损失主效应支持，但至少一个简单效应显著"
                "变差（交互）"
            )
        elif balance_simple_summary["status"] == "blocked":
            balance_status = "blocked"
            balance_reasons.append(
                "均衡损失主效应支持，但简单效应受混杂/未核对"
                "控制项阻塞"
            )
        elif balance_simple_summary.get(
            "attrition_unstable_effects"
        ):
            balance_status = "inconclusive"
            balance_reasons.append(
                "均衡损失主效应支持，但至少一个简单效应在缺失/"
                "失败运行的支持界内方向可能反转：不能宣称该组件"
                "在所有设置下有用；应补齐运行或调查失败原因"
            )
        elif (
            balance_interaction_significant
            and balance_simple_summary["status"] != "supported"
        ):
            balance_status = "partial"
            balance_reasons.append(
                "交互显著且简单效应未全部支持：主效应可能"
                "只由其中一个设置驱动，不能宣称均衡损失"
                "在所有设置下有用"
            )
    elif (
        balance_status
        in {"missing", "insufficient_data", "inconclusive"}
        and balance_simple_summary["status"] == "supported"
    ):
        balance_status = "partial"
        balance_reasons.append(
            "主效应不足，但部分简单效应支持：只算局部证据"
        )
    components.append({
        "component": "balance_loss",
        "label": "负载均衡损失",
        "status": balance_status,
        "primary": balance_primary,
        "simple": balance_simple_summary,
        "reasons": balance_reasons,
    })

    sensitivity = coverage.get("common_step_sensitivity")
    component_flips: Dict[str, object] = (
        sensitivity.get("component_flips", {})
        if isinstance(sensitivity, dict)
        and sensitivity.get("applicable")
        else {}
    )
    for component in components:
        flipped = component_flips.get(
            str(component["component"])
        ) or []
        if flipped and component["status"] == "supported":
            component["status"] = "inconclusive"
            component["reasons"].append(
                "final 选型口径支持，但共同步数选型无关口径下"
                "至少一个效应不再支持：结论对 checkpoint "
                "选择敏感"
            )

    blockers: List[str] = []
    if coverage.get("partial"):
        blockers.append(
            "本轮是部分消融，未覆盖项目方案 "
            + ", ".join(
                str(mode)
                for mode in coverage.get(
                    "undeclared_project_modes", []
                )
            )
        )
    if isinstance(mechanism, dict) and mechanism.get("applicable"):
        for mode in sorted(
            str(name)
            for name in (mechanism.get("schemes") or {})
        ):
            entry = _mechanism_entry_for_mode(mechanism, mode)
            if entry is None:
                continue
            mechanism_status = str(entry.get("status"))
            if mechanism_status not in (
                "missing_control",
                "confounded",
                "no_control_defined",
                "unverifiable",
                "insufficient_data",
                "insufficient_seeds",
            ):
                continue
            blockers.append(
                f"{mode} 的单因素机制对照未就绪"
                f"（对照 {entry.get('control_mode')}，"
                f"状态 {mechanism_status}）：2×2 主效应是多因素"
                "对比，不能单独把收益归因到延迟合并机制本身"
            )
    if coverage.get("missing_cells"):
        blockers.extend(
            "缺少消融组合: " + str(message)
            for message in coverage["missing_cells"]
        )
    if coverage.get("mechanism_missing_cells"):
        blockers.extend(
            "缺少单因素机制对照单元: " + str(message)
            for message in coverage["mechanism_missing_cells"]
        )
    for key, label in (
        ("structural_confounds", "结构设置混杂"),
        ("width_confounds", "专家宽度/计算量混杂"),
        ("attrition_confounds", "数值失败/脱落不平衡"),
        ("randomness_confounds", "同种子随机性混杂"),
        ("randomness_unverified", "同种子随机性未核对"),
        ("balance_confounds", "均衡强度混杂"),
    ):
        values = coverage.get(key) or []
        if values:
            blockers.append(
                f"{label}（{len(values)} 项）：主效应不能归因"
            )
    capacity_entries = coverage.get("capacity_parity") or []
    for entry in capacity_entries:
        if not entry.get("matched"):
            ratio = entry.get("ratio")
            ratio_text = (
                f"（{float(ratio):.2f}×）"
                if isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                else ""
            )
            blockers.append(
                f"{entry.get('delayed_mode')} 与 local_base 的"
                f"参数量不匹配" + ratio_text
            )
    parity_entries = coverage.get("compute_parity") or []
    for entry in parity_entries:
        if not entry.get("matched"):
            ratio = entry.get("ratio")
            ratio_text = (
                f"（{float(ratio):.2f}×）"
                if isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                else ""
            )
            blockers.append(
                f"{entry.get('delayed_mode')} 与 local_base 的"
                f" {entry.get('field')} 不匹配"
                + ratio_text
            )
    if factorial.get("metric") != _COMPUTE_MATCHED_METRIC:
        if not parity_entries:
            blockers.append(
                "所有运行都缺 FLOPs/参数量记录：计算量对齐"
                "未核对，不能把差异归因到组件"
            )
        else:
            parity_modes = {
                str(entry.get("delayed_mode"))
                for entry in parity_entries
            }
            # 一个方案有记录不代表其他方案也有：只核对 A 的
            # FLOPs 时，B vs local_base 仍可能是未核对的额外
            # 计算，不能因为 parity_entries 非空就放行。
            missing_parity_modes = [
                mode
                for mode in expected_delayed
                if mode not in parity_modes
            ]
            if missing_parity_modes:
                blockers.append(
                    "以下延迟合并方案缺 FLOPs/参数量记录："
                    + ", ".join(missing_parity_modes)
                    + "：计算量对齐未核对，不能把差异归因"
                    "到组件"
                )
    alignment = coverage.get("global_seed_alignment")
    if isinstance(alignment, dict) and alignment.get(
        "applicable"
    ):
        if not alignment.get("all_expected_present"):
            blockers.append(
                "全因子存在缺失格，无法核对所有组件的共同种子"
            )
        elif not alignment.get("aligned"):
            blockers.append(
                "全因子各格种子集合不一致，组件效应只在"
                "不同种子子集上可配对"
            )
        elif int(alignment.get("n_common_seeds", 0)) < (
            _MIN_SEEDS_FOR_STATS
        ):
            blockers.append(
                "全因子共同种子只有 "
                f"{alignment.get('n_common_seeds', 0)} 个"
                f"（< {_MIN_SEEDS_FOR_STATS}）"
            )
    duplicate_effects = [
        effect
        for effect in effects
        if effect.get("duplicate_seeds")
    ]
    if duplicate_effects:
        blockers.append(
            "存在重复随机种子（复制运行已折叠，不是独立重复）："
            + "; ".join(
                f"{effect.get('delayed_mode')}/"
                f"{effect.get('effect_name')}"
                for effect in duplicate_effects[:5]
            )
        )
    if factorial.get("incomplete_designs"):
        blockers.append(
            "存在不完整的 2×2 设计："
            + "; ".join(
                f"{item.get('delayed_mode')} 缺 "
                + ", ".join(item.get("missing_cells") or [])
                for item in factorial["incomplete_designs"]
            )
        )
    confounded_designs = factorial.get("confounded_designs") or []
    if confounded_designs:
        blockers.append(
            "存在混杂的消融设计："
            + "; ".join(
                "同一上下文含多个正 balance_coef 取值（"
                + ", ".join(
                    format(float(value), "g")
                    for value in item.get(
                        "positive_balance_values", []
                    )
                )
                + "）"
                for item in confounded_designs
            )
        )
    if factorial.get("controlled_variables_ok") is False:
        blockers.append("受控变量检查存在违规")
    if factorial.get("unchecked_control_keys"):
        blockers.append(
            "以下控制项未记录: "
            + ", ".join(
                str(key)
                for key in factorial["unchecked_control_keys"]
            )
        )
    if coverage.get("low_replicate_cells"):
        blockers.append(
            "存在种子数不足的消融单元/未判定单元（"
            + str(len(coverage["low_replicate_cells"]))
            + " 项）"
        )
    if component_flips:
        blockers.append(
            "共同步数敏感性翻转: "
            + ", ".join(sorted(component_flips))
        )
    preregistration = (
        metric_preregistration
        if isinstance(metric_preregistration, dict)
        else {}
    )
    metric_mismatch_runs = [
        str(run)
        for run in (
            preregistration.get("metric_mismatch_runs") or []
        )
    ]
    metric_mismatch_details = [
        str(item)
        for item in (
            preregistration.get("metric_mismatch_details") or []
        )
    ]
    preregistration_missing_runs = [
        str(run)
        for run in (
            preregistration.get("preregistration_missing_runs")
            or []
        )
    ]
    analysis_metric = factorial.get("metric")
    if metric_mismatch_runs:
        blockers.append(
            f"当前分析指标 {analysis_metric} 与 "
            f"{len(metric_mismatch_runs)} 个运行训练前预注册的"
            "主指标不一致（"
            + "; ".join(metric_mismatch_details)
            + "）：在多个指标之间事后挑选支持结论的那个属于"
            "指标选择偏差，组件联合判定只能作为探索性结果"
        )
    if preregistration_missing_runs:
        blockers.append(
            f"{len(preregistration_missing_runs)} 个运行没有"
            "训练前预注册主指标（"
            + ", ".join(preregistration_missing_runs[:5])
            + ("..." if len(preregistration_missing_runs) > 5 else "")
            + "）：无法核对判定指标是否在看结果前声明，"
            "不能排除指标选择偏差，组件联合判定只能作为"
            "探索性结果"
        )

    statuses = {
        str(component["component"]): str(component["status"])
        for component in components
    }
    missing_project = [
        mode for mode in project_delayed if mode not in {
            str(component["component"])[len("delayed_merge_"):]
            for component in components
            if str(component["component"]).startswith(
                "delayed_merge_"
            )
        }
    ]
    if missing_project:
        blockers.append(
            "项目声明的延迟合并方案未进入本轮判定: "
            + ", ".join(missing_project)
        )

    all_supported = bool(components) and all(
        status == "supported" for status in statuses.values()
    ) and not blockers

    if not factorial.get("applicable"):
        advisory = (
            "对比因子未同时包含 mode 与 balance_coef："
            "无法给出组件级联合判定"
        )
    elif all_supported:
        advisory = (
            "全部组件均有校正后显著、方向改善且无混杂的支持"
            "证据：可以宣称各创新点各自有用"
        )
    elif blockers:
        advisory = (
            "存在阻塞条件：先补齐组合/消除混杂/统一共同种子，"
            "不能宣称全部创新点各自有用"
        )
    else:
        advisory = (
            "并非所有组件都获得支持："
            + "；".join(
                f"{component['label']}={component['status']}"
                for component in components
            )
            + "；只能报告有证据的组件，不能宣称「各自有用」"
        )

    return {
        "applicable": bool(factorial.get("applicable")),
        "components": components,
        "statuses": statuses,
        "all_components_supported": all_supported,
        "blockers": blockers,
        "mechanism": mechanism,
        "metric_preregistration": {
            "analysis_metric": analysis_metric,
            "metric_mismatch_runs": metric_mismatch_runs,
            "preregistration_missing_runs": (
                preregistration_missing_runs
            ),
        },
        "advisory": advisory,
    }



def _factorial_effect_key(
    effect: Dict[str, object],
) -> Tuple[str, str, str]:
    """跨口径匹配同一设计、上下文与效应名。"""
    return (
        str(effect.get("delayed_mode")),
        str(
            _stable_factor_key(
                effect.get("context") or {}
            )
        ),
        str(effect.get("effect_name")),
    )
