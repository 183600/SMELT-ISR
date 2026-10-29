from __future__ import annotations



def factorial_common_step_sensitivity(
    aggregates: List[Dict[str, object]],
    group_keys: List[str],
    metric: str,
    primary: Dict[str, object],
    coverage: Dict[str, object],
    controlled: Optional[Dict[str, object]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    在共同步数选型无关口径上重估全因子效应，检查主结论是否翻转。

    判定指标若是验证集选中的 ``final`` 指标，主效应/交互的
    显著性可能来自 winner's curse；共同步数派生指标只取各运行
    最后一个共同评估步的即时值，不经过 checkpoint 选择。这里
    对同一设计重估效应，并逐项比较 claim 与效应方向：

    - ``support_flips``：final 口径支持组件，但共同步数口径
      不再支持（不显著、变差、混杂或缺证据）；
    - ``sign_flips``：两口径的效应方向相反；
    - ``component_flips``：把支持翻转归到具体组件，供联合
      判定把它们降级为阻塞。

    共同步数口径仍受计算量/结构/种子对齐约束，因此沿用同一
    coverage 标注；它替代不了等计算量分析。
    """
    report: Dict[str, object] = {
        "applicable": False,
        "metric": metric,
        "common_step_metric": None,
        "reason": None,
        "effects": [],
        "support_flips": [],
        "sign_flips": [],
        "claim_flips": [],
        "component_flips": {},
        "any_flip": False,
        "advisory": None,
    }
    if metric not in _SELECTION_BIASED_METRICS:
        report["reason"] = (
            "当前指标不是验证集选型后的 final 指标，"
            "无需做共同步数敏感性复核"
        )
        return report
    derived = _common_step_metric_name(metric)
    report["common_step_metric"] = derived
    if derived is None:
        report["reason"] = (
            "当前指标没有共同步数派生口径（只有周期验证会"
            "计算的指标才有），无法复核选型偏差"
        )
        return report
    if not primary.get("applicable"):
        report["reason"] = (
            "主全因子分析不适用，无法做共同步数敏感性复核"
        )
        return report

    common = factorial_effects(
        aggregates,
        group_keys,
        derived,
        equivalence_margin=equivalence_margin,
    )
    report["applicable"] = bool(common.get("applicable"))
    if not common.get("applicable"):
        report["reason"] = (
            "共同步数口径下 mode/balance_coef 未形成可估计的"
            "全因子设计（可能派生指标缺格）"
        )
        return report
    annotate_factorial_confounds(
        common,
        controlled or {},
        coverage,
    )

    primary_effects = {
        _factorial_effect_key(effect): effect
        for effect in primary.get("effects", [])
        if isinstance(effect, dict)
    }
    common_effects = {
        _factorial_effect_key(effect): effect
        for effect in common.get("effects", [])
        if isinstance(effect, dict)
    }
    component_flips: Dict[str, List[Dict[str, object]]] = {}
    for key, primary_effect in primary_effects.items():
        common_effect = common_effects.get(key)
        primary_claim = str(primary_effect.get("claim"))
        common_claim = (
            str(common_effect.get("claim"))
            if common_effect is not None
            else "missing"
        )
        primary_mean = _finite_float(
            primary_effect.get("mean_effect")
        )
        common_mean = (
            _finite_float(common_effect.get("mean_effect"))
            if common_effect is not None
            else None
        )
        sign_flip = bool(
            primary_mean is not None
            and common_mean is not None
            and primary_mean != 0.0
            and common_mean != 0.0
            and (primary_mean > 0.0)
            != (common_mean > 0.0)
        )
        support_flip = bool(
            primary_claim == "supports_component"
            and common_claim != "supports_component"
        )
        claim_flip = primary_claim != common_claim
        record = {
            "delayed_mode": primary_effect.get("delayed_mode"),
            "context": primary_effect.get("context"),
            "effect_name": primary_effect.get("effect_name"),
            "description": primary_effect.get("description"),
            "primary_claim": primary_claim,
            "common_step_claim": common_claim,
            "primary_mean_effect": primary_mean,
            "common_step_mean_effect": common_mean,
            "primary_significant": bool(
                primary_effect.get("significant")
            ),
            "common_step_significant": bool(
                common_effect.get("significant")
                if common_effect is not None
                else False
            ),
            "support_flip": support_flip,
            "sign_flip": sign_flip,
            "claim_flip": claim_flip,
        }
        report["effects"].append(record)
        if support_flip:
            report["support_flips"].append(record)
        if sign_flip:
            report["sign_flips"].append(record)
        if claim_flip:
            report["claim_flips"].append(record)
        if support_flip:
            effect_name = str(
                primary_effect.get("effect_name")
            )
            delayed_mode = str(
                primary_effect.get("delayed_mode")
            )
            if effect_name.startswith("balance_loss"):
                component_name = "balance_loss"
            elif effect_name.startswith("delayed_merge"):
                component_name = f"delayed_merge_{delayed_mode}"
            else:
                component_name = "interaction"
            component_flips.setdefault(
                component_name, []
            ).append(record)

    report["component_flips"] = component_flips
    report["any_flip"] = bool(
        report["support_flips"]
        or report["sign_flips"]
        or report["claim_flips"]
    )
    if report["support_flips"]:
        report["advisory"] = (
            "final 选型口径支持组件的效应在共同步数选型无关口径下"
            "不再支持：主结论依赖 checkpoint 选型，"
            "不能宣称组件有用"
        )
    elif report["sign_flips"]:
        report["advisory"] = (
            "共同步数口径下至少一个效应方向与 final 口径相反："
            "结论对选型偏差敏感"
        )
    elif report["claim_flips"]:
        report["advisory"] = (
            "共同步数口径下至少一个效应的显著性/混杂判定"
            "与 final 口径不同：结论不稳健"
        )
    else:
        report["advisory"] = (
            "共同步数选型无关口径与 final 口径的效应 claim 一致"
        )
    return report



def _current_seed_count(
    contrast: Dict[str, object],
) -> Optional[float]:
    """
    展示「当前种子数」的检验口径：配对检验用配对对数；Welch
    用与检验 se 等价的每臂种子数（两臂样本量的调和均值，两臂
    不等时低于较小臂的直觉计数但不会高估规模）。无法确定时
    返回 None。
    """
    n_pairs = contrast.get("n_pairs")
    n_pairs_value = _finite_float(n_pairs)
    if (
        isinstance(n_pairs, int)
        and not isinstance(n_pairs, bool)
        and n_pairs_value is not None
        and n_pairs_value >= _MIN_SEEDS_FOR_STATS
    ):
        return n_pairs_value

    n_baseline = contrast.get("n_baseline")
    n_variant = contrast.get("n_variant")
    baseline_value = _finite_float(n_baseline)
    variant_value = _finite_float(n_variant)
    if (
        isinstance(n_baseline, int)
        and not isinstance(n_baseline, bool)
        and isinstance(n_variant, int)
        and not isinstance(n_variant, bool)
        and baseline_value is not None
        and variant_value is not None
        and min(baseline_value, variant_value)
        >= _MIN_SEEDS_FOR_STATS
    ):
        return 2.0 / (
            1.0 / baseline_value + 1.0 / variant_value
        )
    return None



def _format_seed_count(value: float) -> str:
    """种子数展示：整数值不补小数，等效调和均值为小数时保留 1 位。"""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"



def _display_count(value: object) -> int:
    """展示用非负计数；损坏/超范围值按 0，避免 int(inf) 崩溃。"""
    number = _finite_float(value)
    if number is None or number < 0.0:
        return 0
    return int(number)



def _format_window_paired(
    contrast: Dict[str, object],
) -> str:
    """
    逐窗口配对证据的一行展示：固定评估窗口上的逐窗口配对差异
    与方向一致性。没有窗口记录时返回空串（不展示，而不是显示
    「无差异」）。
    """
    window_paired = contrast.get("window_paired")
    if not isinstance(window_paired, dict):
        return ""
    if not window_paired.get("available"):
        return ""
    mean_diff = window_paired.get("mean_loss_diff")
    if not isinstance(mean_diff, (int, float)):
        return ""
    favoring = _display_count(
        window_paired.get("n_seeds_favoring_variant", 0)
    )
    against = _display_count(
        window_paired.get("n_seeds_favoring_baseline", 0)
    )
    n_seeds = _display_count(window_paired.get("n_seeds", 0))
    parts = [
        "[window] 固定评估窗口逐窗口配对: Δloss="
        f"{_format_finite(mean_diff, '+.4f')}"
        f"（{favoring}/{n_seeds} 个种子方向偏向变体，"
        f"{against}/{n_seeds} 偏向基线）"
    ]
    per_seed = window_paired.get("per_seed")
    if isinstance(per_seed, list):
        p_values = [
            entry.get("window_p_value")
            for entry in per_seed
            if isinstance(entry, dict)
            and isinstance(entry.get("window_p_value"), (int, float))
        ]
        if p_values:
            parts.append(
                "逐种子窗口级原始 p: "
                + ", ".join(
                    _format_finite(value, ".3g")
                    for value in p_values
                )
            )
        ar1_p_values = []
        for entry in per_seed:
            if not isinstance(entry, dict):
                continue
            if not entry.get("window_ar1_adjustment_applied"):
                continue
            value = entry.get("window_p_value_ar1")
            if isinstance(value, (int, float)):
                ar1_p_values.append(
                    _format_finite(value, ".3g")
                )
            else:
                ar1_p_values.append("有效窗口不足")
        if ar1_p_values:
            parts.append(
                "AR(1) 校正 p: " + ", ".join(ar1_p_values)
            )
    hierarchical = window_paired.get(
        "hierarchical_bootstrap_ci95"
    )
    if isinstance(hierarchical, dict) and hierarchical.get(
        "available"
    ):
        parts.append(
            "两层 bootstrap 95%CI=["
            f"{_format_finite(hierarchical.get('low'), '+.4f')},"
            f"{_format_finite(hierarchical.get('high'), '+.4f')}]"
            f"（种子 {hierarchical.get('n_seeds')} × 窗口，"
            f"{hierarchical.get('replicates')} 次）"
        )
    hierarchical_group = window_paired.get(
        "hierarchical_group_bootstrap_ci95"
    )
    if isinstance(hierarchical_group, dict) and hierarchical_group.get(
        "available"
    ):
        parts.append(
            "种子×文档群组 bootstrap 95%CI=["
            f"{_format_finite(hierarchical_group.get('low'), '+.4f')},"
            f"{_format_finite(hierarchical_group.get('high'), '+.4f')}]"
            f"（种子 {_display_count(hierarchical_group.get('n_seeds', 0))}"
            f" × 群组 {_display_count(hierarchical_group.get('n_groups', 0))}，"
            f"{hierarchical_group.get('replicates')} 次）"
        )
    variance = window_paired.get("variance_components")
    if isinstance(variance, dict) and variance.get("available"):
        share_seed = variance.get("variance_share_seed")
        share_window = variance.get("variance_share_window")
        share_interaction = variance.get(
            "variance_share_interaction"
        )
        dominant = variance.get("dominant_source")
        if (
            isinstance(share_seed, (int, float))
            and isinstance(share_window, (int, float))
            and isinstance(share_interaction, (int, float))
        ):
            source_text = {
                "training_seed": "种子层",
                "evaluation_windows": "窗口层",
                "seed_window_interaction": "种子×窗口交互",
                "mixed": "多层",
            }.get(str(dominant), "多层")
            parts.append(
                "方差占比: 种子 "
                f"{_format_finite(share_seed, '.0f', 100.0)}%"
                " / 窗口 "
                f"{_format_finite(share_window, '.0f', 100.0)}%"
                " / 交互 "
                f"{_format_finite(share_interaction, '.0f', 100.0)}%"
                f"（主要来自{source_text}）"
            )
    notes = window_paired.get("notes")
    if isinstance(notes, list) and notes:
        parts.append("（" + "；".join(str(note) for note in notes) + "）")
    return "  ".join(parts)



def _contrast_display(
    contrast: Dict[str, object],
    metric: str,
) -> str:
    "\"local_base→A\" 等因子变化的人类可读描述。"
    factor = contrast.get("factor")
    baseline = _format_factor_value(
        contrast.get("baseline_value")
    )
    variant = _format_factor_value(
        contrast.get("variant_value")
    )
    delta = contrast.get("delta")
    p_value = contrast.get("p_value")
    p_holm = contrast.get("p_value_holm")
    test = contrast.get("test")
    n_pairs = contrast.get("n_pairs", 0)

    if delta is None:
        delta_text = "Δ=NA"
    else:
        delta_text = f"Δ={_format_finite(delta, '+.4f')}"

    if test == "paired_t":
        test_text = f"配对 t (n={n_pairs})"
    elif test == "welch_t":
        test_text = "Welch t"
    else:
        test_text = "种子不足"

    p_value_number = _finite_float(p_value)
    if p_value is None:
        p_text = "p=NA"
    else:
        p_text = f"p={_format_finite(p_value)}"
        # 家族内做了多重比较校正时，展示校正后的 p 值；
        # 单检验家族（或未参与校正）时两者相同，不重复。
        p_holm_number = _finite_float(p_holm)
        if (
            p_holm_number is not None
            and p_value_number is not None
            and abs(p_holm_number - p_value_number) > 1e-9
        ):
            p_text += f"(Holm {_format_finite(p_holm)})"
        randomization_p = contrast.get(
            "randomization_p_value"
        )
        if randomization_p is not None:
            randomization_holm = contrast.get(
                "randomization_p_value_holm"
            )
            randomization_text = (
                "符号翻转 p="
                f"{_format_finite(randomization_p)}"
            )
            randomization_p_number = _finite_float(
                randomization_p
            )
            randomization_holm_number = _finite_float(
                randomization_holm
            )
            if (
                randomization_holm_number is not None
                and randomization_p_number is not None
                and abs(
                    randomization_holm_number
                    - randomization_p_number
                ) > 1e-9
            ):
                randomization_text += (
                    "(Holm "
                    f"{_format_finite(randomization_holm)})"
                )
            # 只在小样本精确检验或两种检验结论不一致时展示，
            # 避免每行都堆叠两种 p 值；不一致是必须暴露的
            # 方法学风险。
            if (
                contrast.get("randomization_method") == "exact"
                or contrast.get("significant")
                != contrast.get("randomization_significant")
            ):
                p_text += f" [符号翻转/对称性假设: {randomization_text}]"

    star = "*" if contrast.get("significant") else ""

    noise = contrast.get("delta_over_noise")
    noise_number = _finite_float(noise)
    noise_text = (
        f" 噪声比={_format_finite(noise, '.2f')}"
        if noise is not None
        else ""
    )
    # Hedges g：噪声比的小样本无偏近似。二者差距在 n=3 时可
    # 达 43%，同时展示才不会把 d 当成真实效应大小。
    hedges = contrast.get("hedges_g")
    if noise is not None and hedges is not None:
        noise_text += (
            f"（Hedges g={_format_finite(hedges, '.2f')}）"
        )
    # 配对设计的标准化效应必须用差值标准差（dz），不能用
    # 合并标准差；共同种子难度会把合并标准差抬高、把效应
    # 系统性压低。非配对对比没有该字段，不打印。
    dz = contrast.get("cohens_dz")
    if dz is not None:
        dz_text = f"；配对 dz={_format_finite(dz, '.2f')}"
        dz_hedges = contrast.get("hedges_gz")
        if dz_hedges is not None:
            dz_text += (
                "（Hedges gz="
                f"{_format_finite(dz_hedges, '.2f')}）"
            )
        noise_text += dz_text
    # 种子集合不完全相同时，主检验是 Welch；共同种子上的
    # 配对敏感性能说明丢掉配对是否改变了结论，但缺失机制
    # 可能非随机，不能替代主检验。
    partial = contrast.get("partial_pairing")
    if (
        contrast.get("test") == "welch_t"
        and isinstance(partial, dict)
        and partial.get("available")
    ):
        partial_p = partial.get("p_value")
        partial_effect = partial.get("effect")
        partial_text = (
            f"；共同种子配对敏感性"
            f"（n={partial.get('n_common')}"
        )
        if isinstance(partial_effect, (int, float)):
            partial_text += (
                f", Δ={_format_finite(partial_effect, '+.4f')}"
            )
        if isinstance(partial_p, (int, float)):
            partial_text += (
                f", p={_format_finite(partial_p)}"
            )
        partial_text += "，不替代主检验）"
        noise_text += partial_text

    ci_low = contrast.get("ci95_low")
    ci_high = contrast.get("ci95_high")
    ci_text = (
        " 95%CI=["
        f"{_format_finite(ci_low, '+.4f')},"
        f"{_format_finite(ci_high, '+.4f')}]"
        if ci_low is not None and ci_high is not None
        else ""
    )
    # 逐检验 95% CI 与 Holm 判定回答不同问题：家族 m>1 时
    # 另给 Bonferroni 同时区间，避免读者把「CI 不含 0 但
    # significant=false」误解为矛盾。
    adjusted_low = contrast.get("ci_family_adjusted_low")
    adjusted_high = contrast.get("ci_family_adjusted_high")
    adjusted_family = contrast.get(
        "ci_family_adjusted_family_size"
    )
    if (
        adjusted_low is not None
        and adjusted_high is not None
        and isinstance(adjusted_family, int)
        and adjusted_family > 1
    ):
        ci_text += (
            "；同时95%CI=["
            f"{_format_finite(adjusted_low, '+.4f')},"
            f"{_format_finite(adjusted_high, '+.4f')}]"
            f"(Bonferroni α/{adjusted_family}，比Holm保守)"
        )

    verdict_map = {
        "improvement": "改善",
        "regression": "变差",
        "tie": "持平",
        "missing_metric": "缺指标",
    }
    verdict = verdict_map.get(
        str(contrast.get("verdict")),
        "",
    )

    # 同一因子的多个单因子对比（固定其余因子取不同水平）
    # 用「在 … 下」标注上下文，避免重复行无法区分。
    context_parts = [
        part
        for part in str(contrast.get("baseline", "")).split(", ")
        if part and not part.startswith(f"{factor}=")
    ]
    context = (
        "（在 " + ", ".join(context_parts) + " 下）"
        if context_parts
        else ""
    )

    # 统计显著性与实际意义分开：显著但差值小于种子波动
    # （噪声比 < 1）时提示谨慎解读；校正后不显著时禁止
    # 声称提升，并区分「本来就不显著」与「校正后翻转」。
    caveat = ""
    if contrast.get("significant"):
        if noise_number is not None and noise_number < 1.0:
            caveat = "（差值小于典型种子波动，实际意义存疑）"
    elif (
        contrast.get("significant_raw")
        and verdict == "改善"
        and test in ("paired_t", "welch_t")
    ):
        caveat = "（多重比较校正后不显著，不能据此声称提升）"
    elif verdict == "改善" and test in ("paired_t", "welch_t"):
        caveat = "（未达显著水平，不能据此声称提升）"
    elif test == "insufficient_seeds" and verdict in ("改善", "变差"):
        caveat = (
            "（种子数不足，未做显著性检验，"
            "不能据此声称差异）"
        )

    if (
        contrast.get("randomization_p_value") is not None
        and contrast.get("significant")
        != contrast.get("randomization_significant")
    ):
        caveat += (
            "（配对 t 与「差值对称」假设下的符号翻转检验经 Holm "
            "校正后结论不一致：显著性依赖正态性假设或检验"
            "功效，不能作为稳健结论）"
        )

    if contrast.get("leave_one_out_sign_stable") is False:
        caveat += (
            "（逐一去掉一个种子后效应方向翻转或归零："
            "结论可能由单个种子驱动，leave-one-out 稳健性不足）"
        )

    # 方向稳定不等于显著性能在删除种子后存活：一个极端
    # 差值可以让均值方向不变但 p 值被拉过阈值。这里单独
    # 展示脆弱性判定，避免把「LOO 方向稳定」误读成稳健。
    fragility = contrast.get("fragility")
    if isinstance(fragility, dict) and fragility.get("available"):
        fragility_verdict = fragility.get("verdict")
        if (
            fragility_verdict == "fragile"
            and contrast.get("significant")
        ):
            deleted = fragility.get(
                "min_seed_deletions_to_lose_significance"
            )
            deleted_p = fragility.get(
                "min_seed_deletions_p_value"
            )
            alpha_value = fragility.get("alpha")
            if isinstance(deleted, int) and isinstance(
                deleted_p, (int, float)
            ):
                caveat += (
                    f"（脆弱：删除 {deleted} 个种子后 "
                    f"p={_format_finite(deleted_p)}"
                    + (
                        " > alpha="
                        f"{_format_finite(alpha_value)}"
                        if isinstance(
                            alpha_value, (int, float)
                        )
                        else ""
                    )
                    + "，显著性消失；不能作为稳健结论）"
                )
            elif isinstance(deleted, int):
                caveat += (
                    f"（脆弱：删除 {deleted} 个种子后显著性"
                    "消失；不能作为稳健结论）"
                )
            else:
                caveat += (
                    "（脆弱：删除单个种子后效应方向翻转；"
                    "不能作为稳健结论）"
                )
        elif (
            fragility_verdict == "direction_mixed"
            and contrast.get("significant")
        ):
            sign_test = _as_dict(fragility.get("sign_test"))
            caveat += (
                "（方向混合：配对差值并非多数同向，"
                "符号检验 p="
                f"{_format_finite(sign_test.get('p_value'))}"
                if isinstance(sign_test.get("p_value"), (int, float))
                else "（方向混合：配对差值并非多数同向"
            )
            caveat += "，均值显著可能由少数大幅差值驱动）"

    if (
        contrast.get("randomization_method") == "exact"
        and contrast.get("randomization_can_reach_significance")
        is False
    ):
        minimum = contrast.get("randomization_min_attainable_p")
        caveat += (
            "（精确符号翻转检验的最小可达双侧 p="
            f"{_format_finite(minimum, '.3f')}"
            f"（n={contrast.get('n_pairs')}），"
            "该样本量在数学上无法达到 0.05 显著；"
            "不显著不能解读为无效应，请增加独立种子）"
        )

    # 不显著 ≠ 等价：未校正的显著性检验只能回答「有没有
    # 证据反对差异为 0」。等价性必须由预先声明等价界的
    # TOST 支撑；检验过但两个问题都没答上时（不显著且不等价）
    # 结论只能是「不确定」，并把 80% 功效下的最小可检出效应
    # （MDE）摆出来，说明该种子规模本来能/不能检出多大的差异。
    if test in ("paired_t", "welch_t"):
        margin_abs = contrast.get("equivalence_margin_abs")
        if contrast.get("equivalent"):
            margin_text = (
                f"±{_format_finite(margin_abs)}"
                if isinstance(margin_abs, (int, float))
                else "等价界内"
            )
            tost_holm = contrast.get("tost_p_value_holm")
            tost_text = (
                "，Holm p="
                f"{_format_finite(tost_holm, '.3f')}"
                if isinstance(tost_holm, (int, float))
                else ""
            )
            significant_note = (
                "；差异统计显著但落在等价界内（统计显著 ≠ 实际重要）"
                if contrast.get("significant")
                else ""
            )
            caveat += (
                f"（TOST 等价：|Δ| 落在 {margin_text} 等价界内"
                f"{tost_text}，支持无实质差异{significant_note}）"
            )
        elif not contrast.get("significant"):
            mde = contrast.get("mde_80")
            mde_text = (
                "80% 功效下只能检出 |Δ|≥"
                f"{_format_finite(mde)}，"
                if isinstance(mde, (int, float))
                else ""
            )
            caveat += (
                f"（未显著且等价性未确立：{mde_text}"
                "未显著不代表等价，结论不确定）"
            )
            seeds_needed = contrast.get("seeds_to_detect")
            current_seeds = _current_seed_count(contrast)
            if (
                isinstance(seeds_needed, int)
                and current_seeds is not None
                and seeds_needed > current_seeds
            ):
                caveat += (
                    f"（按当前种子间波动，检出该效应约需 "
                    f"{seeds_needed} 个种子，当前 "
                    f"{_format_seed_count(current_seeds)} 个）"
                )
            family_size = contrast.get("significance_family_size")
            mde_family = contrast.get("mde_80_bonferroni")
            if (
                isinstance(family_size, int)
                and not isinstance(family_size, bool)
                and family_size > 1
                and isinstance(mde_family, (int, float))
            ):
                family_text = (
                    f"（家族 m={family_size} 的保守口径：Holm 首步"
                    "alpha/m 下 MDE="
                    f"{_format_finite(mde_family)}"
                )
                seeds_family = contrast.get(
                    "seeds_to_detect_bonferroni"
                )
                if isinstance(seeds_family, int):
                    family_text += f"，约需 {seeds_family} 个种子"
                caveat += family_text + "）"

    # 伪重复：同一单元里同一随机种子出现多次（复制的运行）
    # 不是独立重复实验；检验前已折叠为每 seed 一个观测，仍按
    # 共同种子配对，复制运行不增加 n。
    if contrast.get("duplicate_seeds"):
        caveat += (
            "（实验单元含重复随机种子，已折叠为每 seed 一个观测，"
            "重复运行不增加 n，检验仍按共同种子配对）"
        )

    # 效率与精度权衡：效率倍率必须与涨点一起展示；显著改善
    # 但效率倍率大幅上升、或指标变差且效率倍率更高时给出
    # 明确结论。
    cost_verdict = contrast.get("cost_verdict")
    if cost_verdict == "gain_with_cost":
        caveat += "（改善伴随效率倍率上升，须评估取舍）"
    elif cost_verdict == "dominated_regression":
        caveat += "（指标变差且效率倍率更高，严格劣于基线）"

    cost = contrast.get("cost")
    ratios = cost.get("ratios") if isinstance(cost, dict) else None
    cost_text = ""
    if isinstance(ratios, dict) and ratios:
        uncertainty = (
            cost.get("uncertainty")
            if isinstance(cost, dict)
            else None
        )
        ratio_parts = []
        for field in _COST_FIELDS:
            if field not in ratios:
                continue
            part = (
                f"{_COST_FIELD_LABELS[field]}"
                f"×{_format_finite(ratios[field], '.2f')}"
            )
            detail = (
                uncertainty.get(field)
                if isinstance(uncertainty, dict)
                else None
            )
            if isinstance(detail, dict):
                interval = detail.get("ratio_ci95")
                if isinstance(interval, list):
                    part += (
                        " ["
                        f"{_format_finite(interval[0], '.2f')},"
                        f"{_format_finite(interval[1], '.2f')}]"
                    )
                elif field in _COST_TIMING_FIELDS:
                    part += "（无重复计时）"
            ratio_parts.append(part)
        if ratio_parts:
            cost_text = "  效率倍率: " + " ".join(ratio_parts)

    if isinstance(cost, dict):
        timing_noise = cost.get("timing_noise_fields")
        if timing_noise:
            caveat += (
                "（计时倍率 "
                + "、".join(
                    _COST_FIELD_LABELS.get(
                        str(field), str(field)
                    )
                    for field in timing_noise
                )
                + " 的 95%CI 跨过 1 或相对误差过大："
                "该倍率可能由计时噪声/负载波动驱动，不能单独"
                "作为效率结论）"
            )
        unverifiable = cost.get(
            "unverifiable_timing_fields"
        )
        if unverifiable:
            caveat += (
                "（计时字段缺重复测量，无法估计倍率不确定性："
                + "、".join(
                    _COST_FIELD_LABELS.get(
                        str(field), str(field)
                    )
                    for field in unverifiable
                )
                + "；建议多种子/多次测量后再下效率结论）"
            )
        # 只缺一侧的某个效率字段时，该字段不会进入 ratios；
        # 若只看「是否还有任一倍率」就会漏报这一项不可比，
        # 让残缺的效率证据静默通过。这里按基线/对照分别列出。
        missing_measured = []
        for side_key, side_label in (
            ("baseline_missing_measured_fields", "基线"),
            ("variant_missing_measured_fields", "对照"),
        ):
            fields = cost.get(side_key)
            if not isinstance(fields, list) or not fields:
                continue
            missing_measured.append(
                f"{side_label}缺 "
                + "、".join(
                    _COST_FIELD_LABELS.get(
                        str(field), str(field)
                    )
                    for field in fields
                )
            )
        if missing_measured:
            caveat += (
                "（效率字段不可比："
                + "；".join(missing_measured)
                + "，对应倍率未参与精度/效率权衡）"
            )

    # 涨点/变差但缺效率数据（没有可比倍率）：只报指标差值
    # 等于回避精度与效率的权衡，必须点出权衡缺失，而不是让
    # 结论静默通过。
    if (
        verdict in ("改善", "变差")
        and not (
            isinstance(ratios, dict)
            and any(
                field in ratios
                for field in _MEASURED_COST_FIELDS
            )
        )
    ):
        caveat += "（缺效率数据，无法评估精度/效率权衡）"

    return (
        f"{factor}: {baseline}→{variant}{context}  "
        f"{delta_text}  {verdict}  {test_text} "
        f"{p_text}{star}{noise_text}{ci_text}{cost_text}{caveat}"
    )



def compare_cells_to_baseline(
    aggregates: List[Dict[str, object]],
    baseline_mode: Optional[str],
    metric: str,
    equivalence_margin: Optional[float] = None,
    controlled_variables: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """
    指定基线模式（--compare-baseline），把每个实验单元与
    基线单元直接对比。基线模式在不同上下文（除 mode 外的
    分组因子，如 balance_coef/top_k）有多个单元时，每个
    上下文内分别选运行数最多的一个，且只比较同一上下文的
    单元——否则 mode 因子会与上下文因子混杂。直接对比构成
    自己的检验家族，同样做多重比较校正并附效应量 95% CI，
    以及 80% 功效 MDE 与 TOST 等价性口径
    （``equivalence_margin`` 给定时）。

    ``controlled_variables`` 给定时，声明因子之外仍实际不同的
    协议项会让该直接对比成为混杂对比：只保留双方身份与差异
    键，不再产出可被误读为「只改 mode」的统计显著性。
    """
    if not baseline_mode:
        return []

    context_keys: List[str] = []
    for aggregate in aggregates:
        keys = aggregate.get("keys") or {}
        if not isinstance(keys, dict):
            continue
        for key in keys:
            if key != "mode" and key not in context_keys:
                context_keys.append(key)

    def _cell_context(
        aggregate: Dict[str, object],
    ) -> Tuple[object, ...]:
        keys = aggregate.get("keys") or {}
        return tuple(
            _stable_factor_key(keys.get(key))
            for key in context_keys
        )

    references: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for aggregate in aggregates:
        if str(aggregate["keys"].get("mode")) != baseline_mode:
            continue
        context = _cell_context(aggregate)
        current = references.get(context)
        candidate_rank = (
            # 参考单元按独立种子数选，伪重复不能靠复制目录
            # 把自己「选」成基线；确定性基线（ngram）的多个
            # seed 只是复制同一估计，按 1 个观测计。
            _nonnegative_int(
                aggregate.get("effective_seed_count")
            ),
            _nonnegative_int(aggregate.get("n_runs")),
        )
        if current is None or candidate_rank > (
            _nonnegative_int(
                current.get("effective_seed_count")
            ),
            _nonnegative_int(current.get("n_runs")),
        ):
            references[context] = aggregate
    if not references:
        return []

    contrasts: List[Dict[str, object]] = []

    for aggregate in aggregates:
        reference = references.get(_cell_context(aggregate))
        if reference is None or aggregate is reference:
            continue
        # 同一基线模式的其他单元（如不同的 balance_coef /
        # top_k）与参考单元的差异不是 mode 因子对比：把它们
        # 混入本家族既会膨胀 mode 检验家族，又会给出
        # 「mode: smelt→smelt」这类误导标签。这些差异由
        # factor_contrasts 的单因子对比负责。
        if (
            str(aggregate["keys"].get("mode"))
            == str(reference["keys"].get("mode"))
        ):
            continue
        controlled_confounds = _controlled_confound_keys_for_cells(
            reference,
            aggregate,
            controlled_variables,
        )
        if controlled_confounds:
            contrasts.append({
                "cells": [
                    reference["label"],
                    aggregate["label"],
                ],
                "factor": "mode",
                "baseline": reference["label"],
                "variant": aggregate["label"],
                "baseline_value": baseline_mode,
                "variant_value": aggregate["keys"].get("mode"),
                "differing_factors": ["mode"] + [
                    key
                    for key in controlled_confounds
                    if key != "mode"
                ],
                "confounded": True,
                "controlled_variables_ok": False,
                "controlled_confound_keys": list(
                    controlled_confounds
                ),
            })
            continue
        samples_reference = _as_dict(
            reference.get("samples")
        ).get(metric, [])
        samples_variant = _as_dict(
            aggregate.get("samples")
        ).get(metric, [])
        if not isinstance(samples_reference, list):
            samples_reference = []
        if not isinstance(samples_variant, list):
            samples_variant = []
        result = _contrast_test(
            samples_reference,
            samples_variant,
        )
        mean_reference = _finite_float(
            reference.get(f"{metric}_mean")
        )
        mean_variant = _finite_float(
            aggregate.get(f"{metric}_mean")
        )
        delta = (
            mean_variant - mean_reference
            if mean_reference is not None
            and mean_variant is not None
            else None
        )
        # 与 factor_contrasts 同口径：有限均值之差仍可能溢出，
        # 不能把 ±inf 当作效应量落盘。
        if delta is not None and not math.isfinite(delta):
            delta = None
        p_value = _finite_float(result.get("p_value"))
        significant = (
            p_value is not None
            and p_value <= _SIGNIFICANCE_LEVEL
        )
        if delta is None:
            verdict = "missing_metric"
        elif (
            (-delta if not metric_higher_is_better(metric) else delta)
            > 0
        ):
            verdict = "improvement"
        elif delta != 0.0:
            verdict = "regression"
        else:
            verdict = "tie"
        cost, cost_verdict = _cost_comparison(
            reference,
            aggregate,
            verdict,
            significant,
        )

        # 噪声比与 Hedges g：与单因子对比同口径（|Δ|/合并
        # 标准差 + 小样本偏倚校正），--compare-baseline 的直接
        # 对比同样要能读出「效应相对种子噪声有多大」。
        std_reference = _finite_float(
            reference.get(f"{metric}_std")
        )
        std_current = _finite_float(
            aggregate.get(f"{metric}_std")
        )
        pooled_std = None
        if std_reference is not None and std_current is not None:
            pooled_squared = (
                std_reference * std_reference
                + std_current * std_current
            ) / 2.0
            if math.isfinite(pooled_squared):
                pooled_std = math.sqrt(pooled_squared)
        noise_ratio = None
        if delta is not None and pooled_std:
            candidate = abs(delta) / pooled_std
            if math.isfinite(candidate):
                noise_ratio = candidate
        hedges_correction = _hedges_g_correction(result.get("df"))
        hedges_g = (
            noise_ratio * hedges_correction
            if noise_ratio is not None
            and hedges_correction is not None
            and math.isfinite(
                noise_ratio * hedges_correction
            )
            else None
        )
        paired_effect_size = _paired_effect_size(delta, result)

        entry = {
            "factor": "mode",
            "baseline": reference["label"],
            "variant": aggregate["label"],
            "baseline_value": baseline_mode,
            "variant_value": aggregate["keys"].get("mode"),
            "mean_baseline": mean_reference,
            "mean_variant": mean_variant,
            # 与 factor_contrasts 同口径：std 先转成有限数值，
            # 损坏/超范围记录按缺失处理，不能把 inf 或字符串
            # 原样写进对比报告（还会破坏 JSON 落盘）。
            "std_baseline": std_reference,
            "std_variant": std_current,
            "delta": delta,
            "paired_differences": result.get(
                "paired_differences"
            ),
            "paired_seeds": result.get("paired_seeds"),
            "delta_over_noise": noise_ratio,
            "hedges_g": hedges_g,
            "cohens_dz": paired_effect_size.get("cohens_dz"),
            "hedges_gz": paired_effect_size.get("hedges_gz"),
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
            # samples 已在上面按 _as_dict/list 校验并取出；
            # 不能再直接对 aggregate["samples"] 调 .get，否则
            # 手工编辑/旧版本的损坏记录会让整份对比崩掉。
            "n_baseline": len(
                _collapse_seed_samples(samples_reference)
            ),
            "n_variant": len(
                _collapse_seed_samples(samples_variant)
            ),
            "duplicate_seeds": bool(
                result.get("duplicate_seeds")
            ),
            "partial_pairing": result.get("partial_pairing"),
            "significant": significant,
            "verdict": verdict,
            "cost": cost,
            "cost_verdict": cost_verdict,
            # 逐窗口配对证据：固定评估窗口对所有运行相同，基线
            # 与变体可以按 seed 配对运行、按窗口配对指标。
            "window_paired": _cell_window_paired_analysis(
                reference,
                aggregate,
            ),
        }
        entry.update(_contrast_interval(result))
        _attach_power_equivalence(
            entry,
            delta,
            result,
            equivalence_margin,
        )
        # 与单因子对比同口径：指定基线的直接对比同样只覆盖
        # 完整案例，缺失敏感性界必须随对比一起落盘。
        entry["attrition_bounds"] = _attrition_bounds(
            {"baseline": reference, "variant": aggregate},
            {"baseline": -1.0, "variant": 1.0},
            metric,
            delta,
        )
        contrasts.append(entry)

    # 与单因子对比同理：--compare-baseline 的直接对比构成
    # 自己的检验家族，统一做多重比较校正。
    _apply_multiple_comparison_correction(contrasts)

    return contrasts



# 延迟合并方案：A（局部延迟合并）与 B（全局延迟合并）。项目
# 对两者都声明了「延迟合并提高专家组合表达能力」这一创新点，
# 因此每个方案都必须能证明各自有用——消融覆盖按实际声明的
# 方案逐个核对，不能只报告 A 的组合而漏掉 B。
_DELAYED_MERGE_MODES: Tuple[str, ...] = ("A", "B")


# 消融（延迟合并 × 均衡损失）中除两个消融轴之外必须保持
# 恒定的结构设置：分支几何（k1/k2）、路由宽度（top_k）与残差
# 缩放。它们跨消融单元不一致时，主效应会被结构差异混杂。
_ABLATION_STRUCTURAL_KEYS: Tuple[str, ...] = (
    "k1",
    "k2",
    "top_k",
    "residual_scale",
)
