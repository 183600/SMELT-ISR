from __future__ import annotations



def _cost_comparison(
    cell_a: Dict[str, object],
    cell_b: Dict[str, object],
    verdict: str,
    significant: bool,
) -> Tuple[Dict[str, object], Optional[str]]:
    """
    变体（cell_b）相对基线（cell_a）的效率对比与权衡判定，
    判定逻辑见 ``_cost_verdict_from_ratios``。
    """
    baseline_costs: Dict[str, float] = {}
    variant_costs: Dict[str, float] = {}
    cost_ratios: Dict[str, float] = {}
    cost_uncertainty: Dict[str, Dict[str, object]] = {}
    timing_noise_fields: List[str] = []
    unverifiable_timing_fields: List[str] = []
    ratio_overflow_fields: List[str] = []

    for field in _COST_FIELDS:
        cost_a = _cell_cost_mean(cell_a, field)
        cost_b = _cell_cost_mean(cell_b, field)
        if cost_a is None or cost_b is None:
            continue
        baseline_costs[field] = cost_a
        variant_costs[field] = cost_b
        if cost_a <= 0.0:
            continue
        ratio = cost_b / cost_a
        if not math.isfinite(ratio):
            # 两个成本均值都有限，但真实倍率越过 float 上限。
            # 不写入 inf；单独记录溢出字段，供权衡判定使用。
            ratio_overflow_fields.append(field)
        else:
            cost_ratios[field] = ratio

        std_a = _cell_cost_std(cell_a, field)
        std_b = _cell_cost_std(cell_b, field)
        n_a = _cell_cost_n(cell_a, field)
        n_b = _cell_cost_n(cell_b, field)
        detail: Dict[str, object] = {
            "baseline_n": n_a,
            "variant_n": n_b,
            "baseline_std": std_a,
            "variant_std": std_b,
        }

        relative_se = None
        critical = None
        df = None
        ci_method = None
        low = None
        high = None
        if (
            std_a is not None
            and std_b is not None
            and n_a >= 2
            and n_b >= 2
            and cost_a > 0.0
            and cost_b > 0.0
        ):
            # 对数尺度的 delta 方法：相对标准误由两侧的
            # 变异系数与独立观测数共同决定。先用乘法检查
            # 平方是否落在 float 范围内，避免 **2 抛
            # OverflowError；变体成本为 0 时也不能除零。
            relative_a = std_a / cost_a
            relative_b = std_b / cost_b
            term_a = relative_a * relative_a / n_a
            term_b = relative_b * relative_b / n_b
            relative_variance = term_a + term_b
            if (
                math.isfinite(relative_a)
                and math.isfinite(relative_b)
                and math.isfinite(term_a)
                and math.isfinite(term_b)
                and math.isfinite(relative_variance)
                and relative_variance >= 0.0
            ):
                relative_se = math.sqrt(relative_variance)
                # Welch-Satterthwaite 自由度：用相对项计算，
                # 避免对方差本身再平方而溢出。
                scale_term = max(term_a, term_b)
                if scale_term > 0.0:
                    scaled_a = term_a / scale_term
                    scaled_b = term_b / scale_term
                    denominator = (
                        scaled_a * scaled_a / (n_a - 1)
                        + scaled_b * scaled_b / (n_b - 1)
                    )
                    if (
                        denominator > 0.0
                        and math.isfinite(denominator)
                    ):
                        df = (
                            (scaled_a + scaled_b) ** 2
                            / denominator
                        )
                        critical = _t_critical(
                            df,
                            _COST_RATIO_CI_ALPHA,
                        )
            if critical is None:
                # 零方差时区间退化为点估计；理论上不应
                # 走到这里，保留正态近似仅作数值兜底。
                critical = 1.96
                df = None
                ci_method = "normal_fallback"
            else:
                ci_method = "welch_t"
            if relative_se is not None:
                exponent = critical * relative_se
                if exponent <= 709.0:
                    low = ratio * math.exp(-exponent)
                    high = ratio * math.exp(exponent)
                elif ratio == 0.0:
                    low = 0.0
                    high = 0.0
                if (
                    low is not None
                    and high is not None
                    and math.isfinite(low)
                    and math.isfinite(high)
                ):
                    detail["relative_se"] = round(
                        relative_se, 6
                    )
                    detail["ci_method"] = ci_method
                    detail["ci_df"] = (
                        round(df, 3)
                        if df is not None
                        else None
                    )
                    detail["ci_t_critical"] = round(
                        critical, 6
                    )
                    detail["ratio_ci95"] = [
                        round(low, 6),
                        round(high, 6),
                    ]
                    detail["ci_excludes_one"] = bool(
                        low > 1.0 or high < 1.0
                    )
                    detail["resolvable"] = True
                    if field in _COST_TIMING_FIELDS and (
                        relative_se > _COST_NOISE_CV_LIMIT
                        or not detail["ci_excludes_one"]
                    ):
                        timing_noise_fields.append(field)
                else:
                    relative_se = None
            if relative_se is None:
                detail["relative_se"] = None
                detail["ratio_ci95"] = None
                detail["ci_excludes_one"] = None
                detail["resolvable"] = False
                if field in _COST_TIMING_FIELDS:
                    unverifiable_timing_fields.append(field)
        else:
            detail["relative_se"] = None
            detail["ratio_ci95"] = None
            detail["ci_excludes_one"] = None
            detail["resolvable"] = False
            if field in _COST_TIMING_FIELDS:
                unverifiable_timing_fields.append(field)

        cost_uncertainty[field] = detail

    # 缺重复测量的计时字段同样是不可信成本证据：它没有可估计
    # 的倍率区间，不能让单次测量单独把显著涨点判成
    # gain_with_cost。判定时与噪声字段一并排除，但两者仍分开
    # 落盘，展示层才能区分「统计上不可分辨」与「根本没重复测」。
    low_confidence_timing_fields = (
        timing_noise_fields + unverifiable_timing_fields
    )
    cost_verdict = _cost_verdict_from_ratios(
        cost_ratios,
        verdict,
        significant,
        noisy_timing_fields=low_confidence_timing_fields,
        ratio_overflow_fields=ratio_overflow_fields,
    )

    # 记录哪些「效率字段」缺失：涨点/变差结论无法做精度与
    # 效率权衡时，展示层必须显式提示，而不是静默略过效率。
    return (
        {
            "baseline": baseline_costs,
            "variant": variant_costs,
            "ratios": cost_ratios,
            # 倍率溢出（真实比值 > float 上限）的字段：没有有限
            # 倍率可写，但权衡判定仍把它们视为材料成本。
            "ratio_overflow_fields": ratio_overflow_fields,
            "uncertainty": cost_uncertainty,
            "timing_noise_fields": timing_noise_fields,
            "material_cost_fields": [
                field
                for field, ratio in cost_ratios.items()
                if field not in set(low_confidence_timing_fields)
                and ratio >= _COST_RATIO_MATERIAL
            ]
            + [
                field
                for field in ratio_overflow_fields
                if field not in set(low_confidence_timing_fields)
            ],
            "unverifiable_timing_fields": (
                unverifiable_timing_fields
            ),
            "baseline_missing_measured_fields": (
                _cell_missing_measured_costs(cell_a)
            ),
            "variant_missing_measured_fields": (
                _cell_missing_measured_costs(cell_b)
            ),
        },
        cost_verdict,
    )



def _apply_multiple_comparison_correction(
    contrasts: List[Dict[str, object]],
) -> None:
    """
    对同一分析家族内的全部显著性检验做多重比较校正。

    家族 = 一次分析里实际执行的检验（单因子对比集合，或
    ``--compare-baseline`` 直接对比集合）；同一家族的 p 值
    共享校正，未执行的检验（种子不足、缺指标）不参与家族
    计数。每个对比记录：

    - ``p_value_holm`` / ``p_value_bh``：Holm（FWER）与
      BH（FDR）校正后的 p 值；
    - ``significant_raw``：未校正判定（留档，可追溯「校正
      前后结论翻转」）；
    - ``significant``：以 Holm 校正为准的最终判定；效率的
      权衡结论（``cost_verdict``）同步按该判定重算；
    - ``tost_p_value_holm`` / ``tost_p_value_bh`` 与
      ``equivalent_raw`` / ``equivalent``：等价性检验（TOST）
      的同口径校正，「等价」主张同样受多重比较约束。

    不校正的风险：跑 6 个两两对比，即使每个都用 alpha=0.05，
    至少出现一个假阳性的概率约 26%。只有校正后的显著性才是
    可以在结论中主张的显著性；等价主张同理。
    """
    eligible: List[Dict[str, object]] = []
    p_values: List[float] = []
    for contrast in contrasts:
        # 旧/手工记录的 p 值可能是字符串、NaN/Inf 或超范围整数；
        # 这些不是可校正的检验结果，跳过而不是中断整份报告。
        p_value = _finite_probability(contrast.get("p_value"))
        if p_value is None:
            if "p_value" in contrast:
                if "significant_raw" not in contrast:
                    contrast["significant_raw"] = bool(
                        contrast.get("significant")
                    )
                contrast["significant"] = False
            continue
        eligible.append(contrast)
        p_values.append(p_value)
    holm_values = _holm_bonferroni(p_values)
    bh_values = _benjamini_hochberg(p_values)
    holm_thresholds = _holm_rank_thresholds(p_values)

    # 功效口径与家族校正口径必须一致：MDE/所需种子/所需等价界
    # 按 Holm 第一步阈值 alpha/m 补算，否则报告一边用校正后
    # 判定、一边用未校正功效，会高估现有实验的可判定性。
    for contrast, holm_threshold in zip(
        eligible, holm_thresholds
    ):
        _attach_family_power(contrast, len(eligible))
        contrast["holm_alpha_threshold"] = holm_threshold

    for contrast, holm_p, bh_p in zip(
        eligible,
        holm_values,
        bh_values,
    ):
        contrast["p_value_holm"] = holm_p
        contrast["p_value_bh"] = bh_p
        # 幂等：重复调用时保留首次记录的未校正判定，
        # 不会把校正结果当成「原始」值再记一遍。
        if "significant_raw" not in contrast:
            contrast["significant_raw"] = bool(
                contrast.get("significant")
            )
        corrected = (
            holm_p is not None
            and float(holm_p) <= _SIGNIFICANCE_LEVEL
        )
        contrast["significant"] = corrected

        cost = contrast.get("cost")
        if isinstance(cost, dict):
            ratios = cost.get("ratios")
            low_confidence_timing = []
            for key in (
                "timing_noise_fields",
                "unverifiable_timing_fields",
            ):
                fields = cost.get(key)
                if isinstance(fields, list):
                    low_confidence_timing.extend(fields)
            overflow_fields = cost.get("ratio_overflow_fields")
            contrast["cost_verdict"] = _cost_verdict_from_ratios(
                ratios if isinstance(ratios, dict) else {},
                str(contrast.get("verdict")),
                corrected,
                noisy_timing_fields=(
                    low_confidence_timing
                    if low_confidence_timing
                    else None
                ),
                ratio_overflow_fields=(
                    overflow_fields
                    if isinstance(overflow_fields, list)
                    else None
                ),
            )

        # 脆弱性检查必须与最终 Holm 判定同口径。Holm step-down
        # 在某个秩位失败后就会停止：此时后续对比的
        # alpha/(m-rank+1) 阈值在统计上并不成立，不能仅凭
        # 未校正 p 值就把该对比标成 stable/fragile。因此只在
        # 最终 corrected 为真时保留脆弱性判定；否则显式降级
        # 为 not_significant，避免汇总把已失败检验算进
        # stable/fragile 计数。
        paired_values = contrast.get("paired_differences")
        if isinstance(paired_values, list):
            threshold = _finite_float(
                contrast.get("holm_alpha_threshold")
            )
            if threshold is None:
                threshold = _finite_float(
                    contrast.get("alpha_bonferroni")
                )
            if threshold is None:
                threshold = _SIGNIFICANCE_LEVEL
            fragility = _paired_result_fragility(
                paired_values,
                contrast.get("paired_seeds"),
                alpha=threshold,
            )
            if (
                not corrected
                and fragility.get("available")
                and fragility.get("verdict") != "not_significant"
            ):
                fragility["verdict"] = "not_significant"
                fragility["holm_step_down_valid"] = False
                fragility["note"] = (
                    str(fragility.get("note", ""))
                    + " 该对比未通过 Holm step-down 的最终"
                    "显著性判定，脆弱性结论降级为 not_significant。"
                )
            contrast["fragility"] = fragility

    # 符号翻转检验与 t 检验同属一个显著性家族：它只在配对
    # 对比上执行，但同样要受多重比较约束。展示层用二者是否
    # 一致来提示「结论依赖正态性假设」的风险。
    randomization_eligible: List[Dict[str, object]] = []
    randomization_values: List[float] = []
    for contrast in contrasts:
        value = _finite_probability(
            contrast.get("randomization_p_value")
        )
        if value is None:
            continue
        randomization_eligible.append(contrast)
        randomization_values.append(value)
    randomization_holm = _holm_bonferroni(
        randomization_values
    )
    randomization_bh = _benjamini_hochberg(
        randomization_values
    )
    for contrast, holm_p, bh_p in zip(
        randomization_eligible,
        randomization_holm,
        randomization_bh,
    ):
        contrast["randomization_null_assumption"] = (
            "symmetric_paired_differences_about_zero"
        )
        contrast["randomization_p_value_holm"] = holm_p
        contrast["randomization_p_value_bh"] = bh_p
        if "randomization_significant_raw" not in contrast:
            contrast["randomization_significant_raw"] = bool(
                float(contrast["randomization_p_value"])
                <= _SIGNIFICANCE_LEVEL
            )
        contrast["randomization_significant"] = bool(
            holm_p is not None
            and float(holm_p) <= _SIGNIFICANCE_LEVEL
        )
        # 稳健显著：两种检验在家族校正后都显著。任一方不显著
        # 时，显著性结论依赖分布假设或检验功效，不能当作
        # 稳健结论。
        contrast["significant_robust"] = bool(
            contrast.get("significant")
            and contrast.get("randomization_significant")
        )
        # 精确符号翻转的最小可达 p 本身大于 alpha 时，
        # significant_robust 在数学上不可能为 True；显式标注，
        # 避免把「设计功效不足」误读成「效应不稳健」。
        contrast["significant_robust_possible"] = bool(
            contrast.get(
                "randomization_can_reach_significance", True
            )
            is not False
        )

    # 等价性（TOST）主张同样构成检验家族：对 m 个对比都声称
    # 「差异落在等价界内」，至少一个误判的概率同样随家族规模
    # 上升。TOST p 值越小越支持等价，这里做与显著性同口径的
    # Holm（FWER）/ BH（FDR）校正：只有校正后仍 <= alpha 的
    # 等价结论才是可以在报告中主张的「无实质差异」，其余只能
    # 报告「未发现显著差异（结论不确定）」。
    tost_eligible: List[Dict[str, object]] = []
    tost_values: List[float] = []
    for contrast in contrasts:
        value = _finite_probability(contrast.get("tost_p_value"))
        if value is None:
            continue
        tost_eligible.append(contrast)
        tost_values.append(value)
    tost_holm = _holm_bonferroni(tost_values)
    tost_bh = _benjamini_hochberg(tost_values)

    for contrast, holm_p, bh_p in zip(
        tost_eligible,
        tost_holm,
        tost_bh,
    ):
        contrast["tost_p_value_holm"] = holm_p
        contrast["tost_p_value_bh"] = bh_p
        if "equivalent_raw" not in contrast:
            contrast["equivalent_raw"] = bool(
                contrast.get("equivalent")
            )
        contrast["equivalent"] = bool(
            holm_p is not None
            and float(holm_p) <= _SIGNIFICANCE_LEVEL
        )

    # 多重比较校正后的显著性以 Holm 为准；展示层的逐检验 95%
    # CI 与它可能「看起来矛盾」。另附 Bonferroni 同时区间，
    # 让「校正后仍不含 0」有一个与家族规模一致的区间口径。
    _attach_family_adjusted_intervals(eligible)



def factor_contrasts(
    aggregates: List[Dict[str, object]],
    group_keys: List[str],
    metric: str,
    equivalence_margin: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    单因子对比：只有恰好一个因子取值不同的两个实验单元之间
    做统计检验。全因子消融的每个因子主效应（以及交互）都由
    若干这样的单因子对比给出；一次变化多个因子的组合被标记
    为混杂对比，结论无法归因。返回前对家族内全部检验做
    Holm（FWER）/ BH（FDR）多重比较校正并附效应量 95% CI。

    ``equivalence_margin`` 给定时（相对基线均值的比例，见
    ``--equivalence-margin``），每个对比额外附带 80% 功效的
    最小可检出效应（MDE）与 TOST 等价性检验：不显著 ≠ 等价，
    只有落在预先声明的等价界内才是「无实质差异」的证据。
    """
    contrasts: List[Dict[str, object]] = []
    higher_is_better = metric_higher_is_better(metric)

    for index_a in range(len(aggregates)):
        for index_b in range(index_a + 1, len(aggregates)):
            cell_a = aggregates[index_a]
            cell_b = aggregates[index_b]

            differing = [
                name
                for name in group_keys
                if _stable_factor_key(
                    cell_a["keys"].get(name)
                )
                != _stable_factor_key(
                    cell_b["keys"].get(name)
                )
            ]

            if not differing:
                continue

            contrast: Dict[str, object] = {
                "cells": [cell_a["label"], cell_b["label"]],
                "differing_factors": differing,
                "confounded": len(differing) > 1,
            }

            if len(differing) != 1:
                contrasts.append(contrast)
                continue

            factor = differing[0]

            # 让「基线值 → 新值」的方向与自然顺序一致，
            # delta = mean(新) - mean(基线)。
            rank_a = _factor_sort_rank(
                cell_a["keys"].get(factor)
            )
            rank_b = _factor_sort_rank(
                cell_b["keys"].get(factor)
            )
            if rank_a > rank_b:
                cell_a, cell_b = cell_b, cell_a

            samples_a = _as_dict(
                cell_a.get("samples")
            ).get(metric, [])
            samples_b = _as_dict(
                cell_b.get("samples")
            ).get(metric, [])
            if not isinstance(samples_a, list):
                samples_a = []
            if not isinstance(samples_b, list):
                samples_b = []
            mean_a = _finite_float(
                cell_a.get(f"{metric}_mean")
            )
            mean_b = _finite_float(
                cell_b.get(f"{metric}_mean")
            )
            std_a = _finite_float(
                cell_a.get(f"{metric}_std")
            )
            std_b = _finite_float(
                cell_b.get(f"{metric}_std")
            )

            delta = (
                mean_b - mean_a
                if mean_a is not None and mean_b is not None
                else None
            )
            # 两端均值都有限但符号相反且接近 float 上限时，差值
            # 本身会溢出成 ±inf。效应量必须是有限数，否则会被
            # 原样写进 JSON 报告；按缺失处理并保留两端均值。
            if delta is not None and not math.isfinite(delta):
                delta = None

            pooled_std = None
            if std_a is not None and std_b is not None:
                pooled_squared = (
                    std_a * std_a + std_b * std_b
                ) / 2.0
                if math.isfinite(pooled_squared):
                    pooled_std = math.sqrt(pooled_squared)

            noise_ratio = None
            if delta is not None and pooled_std:
                candidate = abs(delta) / pooled_std
                if math.isfinite(candidate):
                    noise_ratio = candidate

            test = _contrast_test(samples_a, samples_b)

            # 逐窗口配对证据：固定评估窗口对所有运行相同，两个
            # 单元可以按 seed 配对运行、按窗口配对指标。它与种子
            # 级检验互补（评估集层面 vs 训练随机性），但来自同一
            # 批运行，不能当成两次独立验证。
            window_paired = _cell_window_paired_analysis(
                cell_a,
                cell_b,
            )

            # Hedges g：Cohen's d（噪声比）的小样本无偏近似。
            # n=3 时 d 高估约 13%（配对口径约 43%），报告同时
            # 给 g 才不会把小种子数的效应量当成真实效应大小。
            hedges_correction = _hedges_g_correction(test.get("df"))
            hedges_g = (
                noise_ratio * hedges_correction
                if noise_ratio is not None
                and hedges_correction is not None
                and math.isfinite(
                    noise_ratio * hedges_correction
                )
                else None
            )
            # 展示的样本数按独立种子口径（伪重复折叠后），
            # 与检验使用的 n/df 保持同一单位。
            independent_a = _collapse_seed_samples(samples_a)
            independent_b = _collapse_seed_samples(samples_b)

            # 完整案例分析（只用有该指标的运行）默认假设「缺指标
            # 与指标值无关」。两臂缺失数不同时，对比就建立在不同
            # 的运行子集上：差异可能来自「哪些运行留下了」而不是
            # 方法本身，必须显式暴露而不是静默丢弃。缺失按「原始
            # 样本数 vs 运行数」计算：伪重复折叠已经单独标记，
            # 不能混进缺失计数里造成误报。
            deterministic_cell = bool(
                cell_a.get("deterministic")
                or cell_b.get("deterministic")
            )
            expected_a = _nonnegative_int(
                cell_a.get("n_runs")
            )
            expected_b = _nonnegative_int(
                cell_b.get("n_runs")
            )
            missing_a = (
                0
                if deterministic_cell
                else max(0, expected_a - len(samples_a))
            )
            missing_b = (
                0
                if deterministic_cell
                else max(0, expected_b - len(samples_b))
            )

            if delta is None:
                verdict = "missing_metric"
            else:
                improvement = (
                    -delta if not higher_is_better else delta
                )
                if improvement > 0:
                    verdict = "improvement"
                elif improvement < 0:
                    verdict = "regression"
                else:
                    verdict = "tie"

            # 效率与精度的权衡：给出变体相对基线的效率倍率，并
            # 把「统计显著但效率更低」「变差且效率更低」单独
            # 标记，避免只报涨点而不报效率变化。
            test_p_value = _finite_float(test.get("p_value"))
            significant = (
                test_p_value is not None
                and test_p_value <= _SIGNIFICANCE_LEVEL
            )
            cost, cost_verdict = _cost_comparison(
                cell_a,
                cell_b,
                verdict,
                significant,
            )

            paired_effect_size = _paired_effect_size(delta, test)
            contrast.update({
                "factor": factor,
                "baseline": cell_a["label"],
                "variant": cell_b["label"],
                "baseline_value": cell_a["keys"].get(factor),
                "variant_value": cell_b["keys"].get(factor),
                "mean_baseline": mean_a,
                "mean_variant": mean_b,
                "std_baseline": std_a,
                "std_variant": std_b,
                "delta": delta,
                "paired_differences": test.get(
                    "paired_differences"
                ),
                "paired_seeds": test.get("paired_seeds"),
                "delta_over_noise": noise_ratio,
                "hedges_g": hedges_g,
                "cohens_dz": paired_effect_size.get(
                    "cohens_dz"
                ),
                "hedges_gz": paired_effect_size.get(
                    "hedges_gz"
                ),
                "test": test.get("test"),
                "t": test.get("t"),
                "df": test.get("df"),
                "se": test.get("se"),
                "p_value": test.get("p_value"),
                "randomization_p_value": test.get(
                    "randomization_p_value"
                ),
                "randomization_method": test.get(
                    "randomization_method"
                ),
                "randomization_permutations": test.get(
                    "randomization_permutations"
                ),
                "randomization_min_attainable_p": test.get(
                    "randomization_min_attainable_p"
                ),
                "randomization_can_reach_significance": test.get(
                    "randomization_can_reach_significance"
                ),
                "randomization_required_seeds": test.get(
                    "randomization_required_seeds"
                ),
                "leave_one_out_effect_min": test.get(
                    "leave_one_out_effect_min"
                ),
                "leave_one_out_effect_max": test.get(
                    "leave_one_out_effect_max"
                ),
                "leave_one_out_sign_stable": test.get(
                    "leave_one_out_sign_stable"
                ),
                "n_pairs": test.get("n_pairs", 0),
                "n_baseline": len(independent_a),
                "n_variant": len(independent_b),
                "n_expected_baseline": expected_a,
                "n_expected_variant": expected_b,
                "n_missing_baseline": missing_a,
                "n_missing_variant": missing_b,
                "differential_missingness": bool(
                    (missing_a > 0 or missing_b > 0)
                    and missing_a != missing_b
                ),
                "duplicate_seeds": bool(
                    test.get("duplicate_seeds")
                ),
                # 伪重复的折叠口径与组内差异必须随对比结果一起
                # 落盘；只保留 duplicate_seeds 会丢掉
                # within_seed_sd_max，无法判断复制运行是否真的
                # 是确定性复制。
                "pseudo_replication": test.get(
                    "pseudo_replication"
                ),
                "partial_pairing": test.get("partial_pairing"),
                "significant": significant,
                "verdict": verdict,
                "cost": cost,
                "cost_verdict": cost_verdict,
                "window_paired": window_paired,
            })
            contrast.update(_contrast_interval(test))
            _attach_power_equivalence(
                contrast,
                delta,
                test,
                equivalence_margin,
            )
            # 完整案例分析只覆盖「有该指标且未数值失败」的运行。
            # 在观测支持范围内给缺失运行取极端值，检查效应方向
            # 是否仍能成立；跨 0 时显著结论不能当作对缺失机制
            # 稳健。该界与失败率、差分缺失一起读，不能单独用
            # 完整案例均值掩盖幸存者偏差。
            contrast["attrition_bounds"] = _attrition_bounds(
                {"baseline": cell_a, "variant": cell_b},
                {"baseline": -1.0, "variant": 1.0},
                metric,
                delta,
            )
            contrasts.append(contrast)

    # 家族的显著性判定必须在全部对比都生成后统一校正，
    # 否则先创建的对比不知道家族里还有多少检验。
    _apply_multiple_comparison_correction(contrasts)

    return contrasts



def _cell_seed_values(
    cell: Dict[str, object],
    metric: str,
) -> Tuple[
    Dict[object, Tuple[object, float]],
    bool,
    List[object],
]:
    """
    把一个实验单元在某指标上的样本整理成 seed → (原始 seed, 值)。

    返回 ``(mapping, complete, duplicate_seeds)``：``complete``
    为 False 表示该单元存在未记录种子的样本，无法参与按种子
    配对；``duplicate_seeds`` 列出同一 seed 出现多次的种子，
    这些复制运行虽被折叠成一个观测，但不能当作额外独立重复。
    """
    sample_map = cell.get("samples")
    if not isinstance(sample_map, dict):
        sample_map = {}
    raw_samples = sample_map.get(metric, [])
    if not isinstance(raw_samples, list):
        raw_samples = []
    samples = [
        sample
        for sample in raw_samples
        if isinstance(sample, dict)
    ]
    collapsed = _collapse_seed_samples(samples)
    seed_counts: Dict[object, int] = {}
    seed_values: Dict[object, object] = {}
    for sample in samples:
        seed = sample.get("seed")
        if seed is None:
            continue
        key = _stable_factor_key(seed)
        seed_counts[key] = seed_counts.get(key, 0) + 1
        seed_values.setdefault(key, seed)
    duplicate_seeds = [
        # 对外报告原始 seed 值（与 factor_contrasts /
        # _contrast_test 一致），而不是内部稳定键元组。
        seed_values[seed_key]
        for seed_key, count in seed_counts.items()
        if count > 1
    ]
    mapping: Dict[object, Tuple[object, float]] = {}
    complete = (
        len(samples) == len(raw_samples)
        and all(
            _finite_float(sample.get("value")) is not None
            for sample in samples
        )
    )
    for sample in collapsed:
        seed = sample.get("seed")
        if seed is None:
            complete = False
            continue
        value = _finite_float(sample.get("value"))
        if value is None:
            complete = False
            continue
        mapping[_stable_factor_key(seed)] = (seed, value)
    return mapping, complete, duplicate_seeds



def _paired_linear_effect(
    cells: Dict[str, Dict[str, object]],
    coefficients: Dict[str, float],
    metric: str,
) -> Dict[str, object]:
    """
    计算若干实验单元的线性组合效应，并在共同种子上做配对检验。

    主效应与交互都是单元格均值的线性组合（如 A/B 全因子中的
    ``M_on - M_off - L_on + L_off``）。只有在同一组随机种子上
    计算每条线性组合，配对 t 检验才回答「该效应是否稳定地
    偏离 0」；直接在聚合均值上做 Welch 检验会把不同种子的
    方差当成独立噪声，低估交互的不确定性。
    """
    maps: Dict[str, Dict[object, Tuple[object, float]]] = {}
    complete = True
    duplicate_seeds: List[object] = []
    duplicate_seed_keys: set = set()
    for key in coefficients:
        if key not in cells:
            return {"test": "missing_cell"}
        mapping, cell_complete, cell_duplicates = (
            _cell_seed_values(cells[key], metric)
        )
        maps[key] = mapping
        complete = complete and cell_complete
        # 同一个复制种子可能同时出现在多个组成单元里；它只应
        # 在报告中出现一次，并保留原始 seed 值而非内部稳定键。
        for seed in cell_duplicates:
            seed_key = _stable_factor_key(seed)
            if seed_key in duplicate_seed_keys:
                continue
            duplicate_seed_keys.add(seed_key)
            duplicate_seeds.append(seed)

    if not maps:
        return {"test": "missing_cell"}

    common = set.intersection(
        *(set(mapping) for mapping in maps.values())
    )
    aligned = all(
        set(mapping) == common for mapping in maps.values()
    )
    effect_values: List[float] = []
    raw_seeds: List[object] = []
    seed_keys = sorted(common, key=str)
    first_cell = next(iter(maps.values()))
    for seed_key in seed_keys:
        terms: List[float] = []
        total = 0.0
        for cell_key, coefficient in coefficients.items():
            term = (
                coefficient
                * maps[cell_key][seed_key][1]
            )
            terms.append(term)
            total += term
        # 主效应/交互的线性组合可能先出现同号中间和再相消：
        # 例如 +1e308 -(-1e308) -1e308 +(-1e308) 的真实效应
        # 为 0，但顺序累加在第二步就溢出成 inf。此时按最大
        # 绝对值缩放后再求和，保留真实相消；正常量级仍走上面
        # 的顺序累加，不改变历史数值。若缩放后仍不可表示，
        # 保留非有限值，让下游检验按 invalid_values 处理。
        if not math.isfinite(total) and terms:
            scale = max(
                (abs(term) for term in terms),
                default=0.0,
            )
            if scale > 0.0 and math.isfinite(scale):
                candidate = math.fsum(
                    term / scale for term in terms
                ) * scale
                if math.isfinite(candidate):
                    total = candidate
        effect_values.append(total)
        raw_seeds.append(first_cell[seed_key][0])

    if effect_values:
        # 标签与数值必须一起排序：只排 raw_seeds 会让 paired_seeds
        # 与 paired_differences/effect_values 错位，脆弱性分析
        # 的 LOO 标签就会指向错误的种子。
        ordered = sorted(
            zip(raw_seeds, effect_values),
            key=lambda pair: _factor_sort_rank(pair[0]),
        )
        raw_seeds = [seed for seed, _ in ordered]
        effect_values = [value for _, value in ordered]
        result = _paired_t_test(effect_values)
        randomization = _paired_sign_flip_test(effect_values)
        stability = _leave_one_out_effect_stability(
            effect_values
        )
        effect = result.get("effect")
        if effect is None:
            effect = _mean(effect_values)
        effect_test = dict(result)
        effect_test["n_pairs"] = len(effect_values)
        paired_effect_size = _paired_effect_size(
            effect,
            effect_test,
        )
        finite_effect_values = _finite_values(effect_values)
        return {
            "effect": effect,
            # 线性组合本身可能超出 float 范围；此时检验已按
            # invalid_values 处理，逐种子差值也必须标记为不可用，
            # 不能把 inf 写进报告或交给脆弱性重算。
            "paired_differences": (
                list(effect_values)
                if finite_effect_values is not None
                else None
            ),
            "paired_differences_finite": (
                finite_effect_values is not None
            ),
            "paired_seeds": list(raw_seeds),
            "cohens_dz": paired_effect_size.get("cohens_dz"),
            "hedges_gz": paired_effect_size.get("hedges_gz"),
            "test": result.get("test"),
            "leave_one_out_effect_min": stability.get(
                "leave_one_out_effect_min"
            ),
            "leave_one_out_effect_max": stability.get(
                "leave_one_out_effect_max"
            ),
            "leave_one_out_sign_stable": stability.get(
                "leave_one_out_sign_stable"
            ),
            "t": result.get("t"),
            "df": result.get("df"),
            "p_value": result.get("p_value"),
            "randomization_p_value": randomization.get(
                "p_value"
            ),
            "randomization_method": randomization.get(
                "method"
            ),
            "randomization_permutations": randomization.get(
                "n_permutations"
            ),
            "randomization_min_attainable_p": randomization.get(
                "min_attainable_p"
            ),
            "randomization_can_reach_significance": (
                randomization.get("can_reach_significance")
            ),
            "randomization_required_seeds": randomization.get(
                "required_seeds_for_alpha"
            ),
            "se": result.get("se"),
            "n_pairs": len(effect_values),
            "aligned_seeds": aligned,
            "common_seeds": raw_seeds,
            "complete_seed_records": complete,
            "duplicate_seeds": duplicate_seeds,
        }

    # 没有共同种子：仍给出由聚合均值组合的点估计，但明确
    # 标记不能做配对检验（缺失/不同种子不能伪造显著性）。
    means = [
        _finite_float(cells[key].get(f"{metric}_mean"))
        for key in coefficients
    ]
    if any(mean is None for mean in means):
        return {
            "test": "missing_metric",
            "aligned_seeds": False,
            "n_pairs": 0,
        }
    terms = [
        coefficient * float(mean)
        for coefficient, mean in zip(
            coefficients.values(), means
        )
    ]
    # 有限均值乘系数、再求和仍可能溢出（如 +1e308 与 -(-1e308)
    # 在交互效应里相加）。先按最大绝对值归一化再组合，保留真实
    # 的相消；仍不可表示时返回 None，不能把 inf 当点估计。
    if any(not math.isfinite(term) for term in terms):
        effect = None
    else:
        scale = max((abs(term) for term in terms), default=0.0)
        if scale == 0.0:
            effect = 0.0
        else:
            effect = math.fsum(
                term / scale for term in terms
            ) * scale
            if not math.isfinite(effect):
                effect = None
    return {
        "effect": effect,
        "test": (
            "no_common_seeds"
            if effect is not None
            else "invalid_values"
        ),
        "p_value": None,
        "se": None,
        "df": None,
        "n_pairs": 0,
        "aligned_seeds": False,
        "common_seeds": [],
        "complete_seed_records": complete,
        "duplicate_seeds": duplicate_seeds,
    }
