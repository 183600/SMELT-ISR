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



def ablation_component_verdicts(
    factorial: Dict[str, object],
    coverage: Dict[str, object],
    metric_preregistration: Optional[Dict[str, object]] = None,
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
      指标之间挑一个支持结论的，属于指标选择偏差）。

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
        components.append({
            "component": f"delayed_merge_{mode}",
            "label": f"延迟合并方案 {mode}",
            "status": status,
            "primary": primary_summary,
            "simple": simple_summary,
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
    if coverage.get("missing_cells"):
        blockers.extend(
            "缺少消融组合: " + str(message)
            for message in coverage["missing_cells"]
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
