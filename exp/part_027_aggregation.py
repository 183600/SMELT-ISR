from __future__ import annotations



def aggregate_runs(
    rows: List[Dict[str, object]],
    group_keys: List[str],
    metric_keys: Tuple[str, ...] = COMPARISON_METRIC_KEYS,
) -> List[Dict[str, object]]:
    """
    按因子组合分桶，汇报 Mean ± Std（样本标准差）。

    每个桶 = 一个实验单元（cell），桶内不同随机种子是
    重复实验；std 反映种子间方差。桶内只有 1 个种子时 std
    为 None，分析阶段会提示统计可靠性不足。
    """
    # 数值失败/发散运行不参与均值、方差与显著性检验：把 NaN/
    # 停止运行的最终指标混进「方法效果」会把故障当成效应。
    # 它们单独列出，保证被排除这件事可见而不是静默消失。
    failed_rows = [
        row
        for row in rows
        if row.get("numerical_status") == "failed"
    ]
    usable_rows = [
        row
        for row in rows
        if row.get("numerical_status") != "failed"
    ]
    failed_by_key: Dict[tuple, List[Dict[str, object]]] = {}
    for row in failed_rows:
        key = tuple(
            _stable_factor_key(_row_factor_value(row, name))
            for name in group_keys
        )
        failed_by_key.setdefault(key, []).append(row)
    rows = usable_rows

    groups: Dict[tuple, List[Dict[str, object]]] = {}
    order: List[tuple] = []

    for row in rows:
        key = tuple(
            _stable_factor_key(_row_factor_value(row, name))
            for name in group_keys
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    aggregates: List[Dict[str, object]] = []

    for key in order:
        group_rows = groups[key]
        factor_values = [
            _row_factor_value(group_rows[0], name)
            for name in group_keys
        ]

        if group_keys:
            label = ", ".join(
                f"{name}={_format_factor_value(value)}"
                for name, value in zip(group_keys, factor_values)
            )
        else:
            label = "(全部运行)"

        seed_values = [
            r.get("seed")
            for r in group_rows
            if r.get("seed") is not None
        ]
        seed_counts: Dict[object, int] = {}
        seed_originals: Dict[object, object] = {}
        for seed in seed_values:
            seed_key = _stable_factor_key(seed)
            seed_counts[seed_key] = (
                seed_counts.get(seed_key, 0) + 1
            )
            seed_originals.setdefault(seed_key, seed)
        # 伪重复检测：同一实验单元里同一随机种子出现多次
        # （如重复目录、复制运行）会被当成独立重复实验参与
        # 均值/方差/配对检验，种子间方差被低估。这里显式记录，
        # 统计与展示层据此拒绝把伪重复当重复实验。
        # 用稳定键计数、用因子排序键排序：损坏的旧 config 可能
        # 混入 1 与 "1"，直接 sorted 会抛 TypeError。
        duplicate_seeds = sorted(
            (
                seed_originals[key]
                for key, count in seed_counts.items()
                if count > 1
            ),
            key=_factor_sort_rank,
        )

        deterministic_rows = [
            row for row in group_rows if _is_deterministic_run(row)
        ]
        all_deterministic = len(deterministic_rows) == len(group_rows)

        aggregate: Dict[str, object] = {
            "label": label,
            "keys": dict(zip(group_keys, factor_values)),
            "n_runs": len(group_rows),
            "runs": [str(r.get("run")) for r in group_rows],
            "seeds": seed_values,
            "duplicate_seeds": duplicate_seeds,
            # 确定性基线（ngram）换 seed 只是复制同一估计：
            # 多个运行不能算独立重复，统计推断按 1 个观测计。
            "deterministic": all_deterministic,
            "deterministic_repeat_runs": (
                [str(row.get("run")) for row in deterministic_rows]
                if all_deterministic and len(deterministic_rows) > 1
                else []
            ),
            "effective_seed_count": (
                1
                if all_deterministic
                else _independent_seed_count(
                    len(group_rows), seed_values
                )
            ),
            "samples": {},
            "costs": {},
            # 逐窗口评估记录（固定评估窗口在所有运行之间相同），
            # 供跨单元的窗口级配对比较使用。
            "window_payloads": [
                {
                    "seed": row.get("seed"),
                    "run": str(row.get("run")),
                    "payload": row.get("eval_windows"),
                }
                for row in group_rows
                if isinstance(row.get("eval_windows"), dict)
            ],
            "numerical_failures": [
                {
                    "run": str(row.get("run")),
                    "curve_status": row.get("curve_status"),
                    "nonfinite_events": row.get(
                        "nonfinite_events"
                    ),
                    "numerical_stop": row.get(
                        "numerical_stop"
                    ),
                    "numerical_polluted": row.get(
                        "numerical_polluted"
                    ),
                    "prior_numerical_stop": row.get(
                        "prior_numerical_stop"
                    ),
                }
                for row in failed_by_key.get(key, [])
            ],
        }

        # 效率指标随精度指标一起聚合：评价改进必须同时给出
        # 涨点与效率指标，否则微小的涨点可能掩盖数量级的差异。
        # 与精度指标同口径先按独立种子折叠：同一 seed 的复制运行
        # （重复目录）给出的是同一个数，按运行计数会虚增 n、放大
        # 自由度；复制运行的 std=0 还会让成本倍率的置信区间塌成
        # 零宽度，把「只测过一次」伪装成可分辨的精确证据。
        for field in _COST_FIELDS:
            samples = _collapse_seed_samples(
                _metric_samples(group_rows, field)
            )
            values = [
                float(sample["value"]) for sample in samples
            ]
            field_mean = _mean(values)
            if field_mean is not None:
                field_std = _sample_std(values)
                cost_entry: Dict[str, object] = {
                    "mean": field_mean,
                    # 跨种子/跨运行的离散度：成本倍率只看均值
                    # 无法判断差异是否只是计时噪声。n_params/
                    # FLOPs 对同一结构应为 0；计时字段的 std
                    # 直接反映测量稳定性。
                    "std": field_std,
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                    "values": [
                        round(float(value), 6)
                        for value in values
                    ],
                    "coefficient_of_variation": (
                        _safe_cv(field_std, field_mean)
                        if field_std is not None
                        else None
                    ),
                }
                # 成本均值的 95% CI（t 分布，自由度 n-1）：与
                # 精度指标同口径。计时字段的 CI 直接决定「效率
                # 倍率」是否只是测量噪声（n<2 或 std 不可用时
                # 不给区间，避免把点估计伪装成零宽度）。
                if field_std is not None and len(values) >= 2:
                    cost_critical = _t_critical(float(len(values) - 1))
                    if cost_critical is not None:
                        cost_half = (
                            cost_critical
                            * field_std
                            / math.sqrt(len(values))
                        )
                        cost_low = field_mean - cost_half
                        cost_high = field_mean + cost_half
                        # 合法但极大的有限均值/标准差在乘 t 临界值
                        # 或相加时仍可能越过 float 上限。区间无法
                        # 用有限数表示时省略，而不是把 ±inf 写进
                        # JSON（非标准 JSON，读取端会被迫崩溃或
                        # 把无穷宽区间当成真实数值）。
                        if math.isfinite(cost_low) and math.isfinite(
                            cost_high
                        ):
                            cost_entry["ci95_low"] = cost_low
                            cost_entry["ci95_high"] = cost_high
                            cost_entry["ci95"] = {
                                "low": cost_low,
                                "high": cost_high,
                                "t_critical": cost_critical,
                                "df": len(values) - 1,
                                "method": "t_interval",
                            }
                aggregate["costs"][field] = cost_entry

        for metric in metric_keys:
            samples = _metric_samples(group_rows, metric)
            # 确定性基线：不同 seed 的运行是同一估计的复制，
            # 只保留一个观测，避免把确定性结果当成多种子
            # 重复（std=0 会人为放大显著性）。
            if all_deterministic and samples:
                values = sorted({
                    round(float(sample["value"]), 12)
                    for sample in samples
                })
                if len(values) > 1:
                    aggregate.setdefault(
                        "deterministic_repeat_conflicts", []
                    ).append(metric)
                samples = samples[:1]
            # 统计量按独立种子计算：伪重复（同一 seed 的多个
            # 运行）先行折叠，复制运行不能虚增 n、压低方差。
            # 原始样本保留在 samples 里供检验层继续识别伪重复。
            independent = _collapse_seed_samples(samples)
            values = [float(s["value"]) for s in independent]
            mean = _mean(values)
            std = _sample_std(values)

            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_n"] = len(values)
            aggregate["samples"][metric] = samples

            # 单元均值的 95% CI（t 分布，自由度 n-1）：std 只
            # 描述种子间离散度，不是均值本身的不确定性；n 很小
            # 时 t 临界值很宽，区间会把「种子不够」直接暴露出来。
            # n<2 或 std 不可用时不给区间，避免把点估计伪装成
            # 零宽度的精确估计。
            ci: Optional[Dict[str, object]] = None
            if mean is not None and std is not None and len(values) >= 2:
                critical = _t_critical(float(len(values) - 1))
                if critical is not None:
                    half_width = (
                        critical * std / math.sqrt(len(values))
                    )
                    low = mean - half_width
                    high = mean + half_width
                    # 与成本 CI 同理：std/均值合法但极大时，t 区间
                    # 的端点可能越过 float 上限。不能把 ±inf 写进
                    # aggregate（它会被原样落盘到 JSON 报告）；无法
                    # 表示时按「区间不可用」处理，保留点估计。
                    if math.isfinite(low) and math.isfinite(high):
                        ci = {
                            "low": low,
                            "high": high,
                            "t_critical": critical,
                            "df": len(values) - 1,
                            "method": "t_interval",
                        }
                        aggregate[f"{metric}_ci95_low"] = low
                        aggregate[f"{metric}_ci95_high"] = high
            aggregate.setdefault("ci95", {})[metric] = ci

            # 没有样本的指标从 JSON 报告中省略，避免大量 null。
            if mean is None:
                aggregate.pop(f"{metric}_mean", None)
                aggregate.pop(f"{metric}_std", None)
                aggregate.pop(f"{metric}_n", None)
                aggregate.pop(f"{metric}_ci95_low", None)
                aggregate.pop(f"{metric}_ci95_high", None)
                aggregate["ci95"].pop(metric, None)
                aggregate["samples"].pop(metric, None)

        aggregates.append(aggregate)

    # 整格全部数值失败的实验单元也必须出现在聚合结果里：否则
    # 下游表格/对比会把它当成「没有这个方案」，失败率与
    # numerical_failures 证据一起消失，正好把最需要披露的臂
    # 静默隐藏。这里生成一个没有可用样本、但带完整失败明细的
    # 占位单元。
    usable_keys = set(order)
    for key in failed_by_key:
        if key in usable_keys:
            continue
        failures = failed_by_key[key]
        factor_values = [
            _row_factor_value(failures[0], name)
            for name in group_keys
        ]
        if group_keys:
            label = ", ".join(
                f"{name}={_format_factor_value(value)}"
                for name, value in zip(
                    group_keys, factor_values
                )
            )
        else:
            label = "(全部运行)"
        aggregates.append({
            "label": label,
            "keys": dict(zip(group_keys, factor_values)),
            "n_runs": 0,
            "n_planned_runs": len(failures),
            "all_runs_failed": True,
            "runs": [],
            "seeds": [],
            "duplicate_seeds": [],
            "deterministic": False,
            "deterministic_repeat_runs": [],
            "effective_seed_count": 0,
            "samples": {},
            "costs": {},
            "ci95": {},
            "window_payloads": [],
            "numerical_status": "failed",
            "numerical_failures": [
                {
                    "run": str(row.get("run")),
                    "curve_status": row.get("curve_status"),
                    "nonfinite_events": row.get(
                        "nonfinite_events"
                    ),
                    "numerical_stop": row.get(
                        "numerical_stop"
                    ),
                    "numerical_polluted": row.get(
                        "numerical_polluted"
                    ),
                    "prior_numerical_stop": row.get(
                        "prior_numerical_stop"
                    ),
                }
                for row in failures
            ],
        })

    aggregates.sort(
        key=lambda aggregate: [
            _factor_sort_rank(aggregate["keys"].get(name))
            for name in group_keys
        ]
    )

    return aggregates



def metric_higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER_METRICS



def _contrast_test(
    samples_a: List[Dict[str, object]],
    samples_b: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    对比两个实验单元的指标样本：种子集合完全相同时用配对
    t 检验（同种子 = 配对重复），否则用 Welch t 检验。

    同一单元里出现重复随机种子（伪重复）时，推断单位仍是
    「独立随机种子」而不是「运行目录」：先把同一 seed 的多个
    样本折叠为单个观测（取均值），再在折叠后的种子上做配对
    检验。折叠后每个 seed 只贡献一个观测，配对比较不会虚增 n
    或自由度；复制运行若给出不同数值，其组内标准差记录在
    ``pseudo_replication`` 里并提示，而不是悄悄平均掉。
    直接退回 Welch 并保留每个复制运行作为样本才是错误做法：
    伪重复的运行不是独立重复，换检验不会恢复独立性。
    """
    # 用稳定键而不是原始值做集合/字典键：损坏但合法的旧 config
    # 可能把 seed 写成 list/dict，直接 set() 会抛 unhashable。
    seeds_a = [
        _stable_factor_key(sample["seed"])
        for sample in samples_a
        if sample["seed"] is not None
    ]
    seeds_b = [
        _stable_factor_key(sample["seed"])
        for sample in samples_b
        if sample["seed"] is not None
    ]
    duplicate_seeds = (
        len(seeds_a) != len(set(seeds_a))
        or len(seeds_b) != len(set(seeds_b))
    )

    # 推断单位是独立随机种子：伪重复先折叠成每 seed 一个观测，
    # 检验的 n / df / 均值都不再把复制运行当独立重复。
    samples_a = _collapse_seed_samples(samples_a)
    samples_b = _collapse_seed_samples(samples_b)
    pseudo_replication = {
        "detected": bool(duplicate_seeds),
        "collapsed": True,
        "unit": "independent_seed",
        "within_seed_sd_max": max(
            [
                float(sample["within_seed_sd"])
                for sample in samples_a + samples_b
                if sample.get("within_seed_sd") is not None
            ],
            default=None,
        ),
        "note": (
            "同一 seed 的多个运行先折叠为一个观测；折叠后"
            "按种子配对，复制运行不增加 n"
        ),
    }

    by_seed_a = {
        _stable_factor_key(sample["seed"]): (
            sample["seed"],
            float(sample["value"]),
        )
        for sample in samples_a
        if sample["seed"] is not None
    }
    by_seed_b = {
        _stable_factor_key(sample["seed"]): (
            sample["seed"],
            float(sample["value"]),
        )
        for sample in samples_b
        if sample["seed"] is not None
    }

    common_keys = sorted(
        set(by_seed_a) & set(by_seed_b),
        key=lambda key: _factor_sort_rank(
            by_seed_a[key][0]
        ),
    )
    common_seeds = [
        by_seed_a[key][0] for key in common_keys
    ]

    # 配对要求两侧样本**全部**带种子且种子集合相同：只比较
    # 「有种子的样本」会让无种子样本被静默排除出检验，却仍计入
    # 均值/Δ/样本数，检验结果与展示的效应量脱节。重复 seed 已在
    # 上面折叠，因此不再排除配对。
    paired = (
        len(common_seeds) >= _MIN_SEEDS_FOR_STATS
        and len(by_seed_a) == len(samples_a)
        and len(by_seed_b) == len(samples_b)
        and len(common_seeds) == len(by_seed_a)
        and len(common_seeds) == len(by_seed_b)
    )

    if paired:
        differences = [
            by_seed_b[key][1] - by_seed_a[key][1]
            for key in common_keys
        ]
        result = _paired_t_test(differences)
        result["n_pairs"] = len(common_seeds)
        result["n_baseline"] = len(by_seed_a)
        result["n_variant"] = len(by_seed_b)
        result["pseudo_replication"] = pseudo_replication
        result["duplicate_seeds"] = bool(duplicate_seeds)
        # 保留逐种子配对差值与种子标签：结果分析要能复核
        # 「哪些种子驱动了显著性」，而不是只相信汇总 p 值。
        # 两端指标各自有限时，差值仍可能溢出（如 1e308 与
        # -1e308）；这样的配对差值不是可落盘的有限数值，也不能
        # 参与脆弱性重算。保留逐种子标签，但把差值标记为不可用，
        # 而不是写 ±inf。
        if _finite_values(differences) is not None:
            result["paired_differences"] = list(differences)
            result["paired_differences_finite"] = True
        else:
            result["paired_differences"] = None
            result["paired_differences_finite"] = False
        result["paired_seeds"] = list(common_seeds)
        result.update(
            _leave_one_out_effect_stability(differences)
        )
        # 小样本的配对 t 检验依赖差值正态性；符号翻转检验
        # 提供「差值分布关于 0 对称」假设下的稳健性对照，二者
        # 结论不一致时展示层
        # 必须提示，不能只挑支持结论的那个。
        randomization = _paired_sign_flip_test(differences)
        if randomization.get("p_value") is not None:
            result["randomization"] = randomization
            result["randomization_p_value"] = randomization.get(
                "p_value"
            )
            result["randomization_method"] = randomization.get(
                "method"
            )
            result["randomization_permutations"] = (
                randomization.get("n_permutations")
            )
            result["randomization_min_attainable_p"] = (
                randomization.get("min_attainable_p")
            )
            result["randomization_can_reach_significance"] = (
                randomization.get("can_reach_significance")
            )
            result["randomization_required_seeds"] = (
                randomization.get("required_seeds_for_alpha")
            )
            result["significant_robust_possible"] = bool(
                randomization.get(
                    "can_reach_significance", True
                )
                is not False
            )
            result["randomization_null_assumption"] = (
                "symmetric_paired_differences_about_zero"
            )
            result["significant_robust_status"] = (
                "not_evaluable_n_le_5"
                if not result["significant_robust_possible"]
                else "evaluable"
            )
        return result

    values_a = [float(sample["value"]) for sample in samples_a]
    values_b = [float(sample["value"]) for sample in samples_b]
    result = _welch_t_test(values_a, values_b)
    result["n_pairs"] = 0
    result["pseudo_replication"] = pseudo_replication

    # 种子集合部分重叠时，主检验退回 Welch 会丢掉共同种子的
    # 配对信息（共同种子上的难度是共享的）。这里补一个仅用
    # 共同种子的完整案例配对敏感性分析：它是稳健性参考，
    # 不能替代主检验——缺失机制与结果相关时，共同子集不是
    # 随机样本。重复 seed 已在上面折叠成独立种子，不再阻止
    # 共同种子配对。
    if len(common_seeds) >= _MIN_SEEDS_FOR_STATS:
        differences = [
            by_seed_b[key][1] - by_seed_a[key][1]
            for key in common_keys
        ]
        paired_result = _paired_t_test(differences)
        paired_result["n_pairs"] = len(common_seeds)
        randomization = _paired_sign_flip_test(differences)
        paired_effect = _paired_effect_size(
            paired_result.get("effect"),
            paired_result,
        )
        result["partial_pairing"] = {
            "available": True,
            "common_seeds": common_seeds,
            "n_common": len(common_seeds),
            "n_baseline": len(samples_a),
            "n_variant": len(samples_b),
            "common_share": (
                float(len(common_seeds))
                / float(min(len(samples_a), len(samples_b)))
                if samples_a and samples_b
                else None
            ),
            "effect": paired_result.get("effect"),
            "test": paired_result.get("test"),
            "t": paired_result.get("t"),
            "df": paired_result.get("df"),
            "se": paired_result.get("se"),
            "p_value": paired_result.get("p_value"),
            "randomization_p_value": randomization.get(
                "p_value"
            ),
            "randomization_method": randomization.get("method"),
            "randomization_min_attainable_p": randomization.get(
                "min_attainable_p"
            ),
            "randomization_can_reach_significance": (
                randomization.get("can_reach_significance")
            ),
            "randomization_required_seeds": (
                randomization.get("required_seeds_for_alpha")
            ),
            "cohens_dz": paired_effect.get("cohens_dz"),
            "hedges_gz": paired_effect.get("hedges_gz"),
            "note": (
                "仅使用两侧都有记录的共同种子做完整案例配对；"
                "若缺失与结果相关，子集可能有偏，不能替代主检验"
                "（Welch），只作稳健性参考"
            ),
        }
    else:
        result["partial_pairing"] = {
            "available": False,
            "reason": "insufficient_common_seeds",
            "common_seeds": common_seeds,
            "n_common": len(common_seeds),
        }
    return result



_ATTRITION_BOUND_SCHEMA = "smelt-isr/attrition-bounds/v1"



def _cell_attrition_support(
    cell: Dict[str, object],
    metric: str,
) -> Dict[str, object]:
    """
    汇总一个实验单元的「已观测值 + 缺失运行数」。

    缺失包括两类运行：数值失败/发散（``numerical_failures``）
    与成功运行但缺该指标（例如曲线或指标未计算）。两者都不
    进入均值/检验，却会让完整案例分析只覆盖幸存子集；本函数
    把「计划中有多少运行、实际看到多少值」显式记下来，供
    ``_attrition_bounds`` 计算部分识别界。
    """
    if not isinstance(cell, dict):
        cell = {}
    sample_map = _as_dict(cell.get("samples"))
    raw_samples = sample_map.get(metric, [])
    values: List[float] = []
    if isinstance(raw_samples, list):
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            value = _finite_float(sample.get("value"))
            if value is not None:
                values.append(value)
    usable_runs = _nonnegative_int(cell.get("n_runs"))
    failures = cell.get("numerical_failures")
    failed_runs = len(failures) if isinstance(failures, list) else 0
    observed_runs = len(values)
    # n_runs 只统计可用运行；失败运行单独放在 numerical_failures。
    # 取 max 是防御旧/手工数据把已观测值算得比计划还多的情况；
    # 这种数据不会让缺失数变成负数。
    attempted_runs = max(
        usable_runs + failed_runs,
        observed_runs,
    )
    return {
        "label": str(cell.get("label", "")),
        "deterministic": bool(cell.get("deterministic")),
        "observed_runs": observed_runs,
        "usable_runs": usable_runs,
        "failed_runs": failed_runs,
        "attempted_runs": attempted_runs,
        "missing_runs": max(0, attempted_runs - observed_runs),
        "observed_mean": _mean(values) if values else None,
        "observed_min": min(values) if values else None,
        "observed_max": max(values) if values else None,
    }



def _attrition_bounds(
    cells: Dict[str, Dict[str, object]],
    coefficients: Dict[str, float],
    metric: str,
    observed_effect: Optional[float] = None,
) -> Dict[str, object]:
    """
    缺失/失败运行下对比效应的部分识别界（Manski 风格）。

    完整案例分析（complete-case）默认假设「缺失与结果无关」，
    但数值失败、曲线缺失或某臂更容易崩掉时，这个假设通常
    不成立。这里不假装能估计缺失值，而是给出一个**透明的
    有界结果假设**：若某个运行的真实结果落在两臂全部已观测
    结果的联合范围 ``[L, U]`` 内，则把该单元的所有缺失运行
    分别赋到 L 或 U，可得到线性组合效应的下界/上界。

    报告 ``effect_min`` / ``effect_max``、``zero_within_bounds``
    与 ``sign_robust``。若界跨 0，支持组件的显著性结论可能
    只是「失败运行恰好都更差」造成的，不能作为稳健证据；若
    缺失值可能落在观测范围之外，则真实界是无穷宽，本函数
    会明确写明这一限制，不把有界假设当成无假设。

    这是**部分识别**（partial identification），不是置信区间，
    也不替代主检验：它回答「在最强/最弱的缺失结果下，效应
    方向是否可能反转」，不回答「缺失的概率是多少」。
    """
    # observed_effect 可能来自手工/旧报告中的超范围数值；统一
    # 转成有限 float（不可用则为 None），否则下面的差值与 JSON
    # 落盘会把 inf 带出去。
    observed_effect_value = _finite_float(observed_effect)
    report: Dict[str, object] = {
        "schema": _ATTRITION_BOUND_SCHEMA,
        "applicable": False,
        "verdict": "not_applicable",
        "reason": None,
        "assumption": (
            "未观测结果落在参与比较的全部单元已观测结果的联合"
            "范围 [min, max] 内；这是有界结果假设下的部分识别界，"
            "不是置信区间。若缺失结果可能超出该范围，真实界为"
            "无穷宽"
        ),
        "complete_cases": False,
        "observed_effect": observed_effect_value,
        "effect_min": None,
        "effect_max": None,
        "observed_effect_within_bounds": None,
        "zero_within_bounds": None,
        "sign_robust": None,
        "attempted_runs": 0,
        "observed_runs": 0,
        "missing_runs": 0,
        "observed_fraction": None,
        "support": {"min": None, "max": None},
        "cells": [],
        "note": (
            "缺失运行不进入均值/检验；本界只说明在观测支持范围内"
            "缺失结果最多能把效应推到多远"
        ),
    }
    if not cells or not coefficients:
        report["reason"] = "没有可用的实验单元或线性组合系数"
        return report

    supports = {
        alias: _cell_attrition_support(cell, metric)
        for alias, cell in cells.items()
    }
    cell_rows: List[Dict[str, object]] = []
    total_attempted = 0
    total_observed = 0
    total_missing = 0
    support_values: List[float] = []
    deterministic = False
    for alias, coefficient in coefficients.items():
        support = supports.get(alias)
        if support is None:
            report["reason"] = f"缺少单元 {alias}"
            return report
        coefficient_value = _finite_float(coefficient)
        if coefficient_value is None:
            report["reason"] = (
                f"单元 {alias} 的线性组合系数不是有限数值"
            )
            return report
        deterministic = deterministic or bool(
            support.get("deterministic")
        )
        total_attempted += int(support["attempted_runs"])
        total_observed += int(support["observed_runs"])
        total_missing += int(support["missing_runs"])
        for key in ("observed_min", "observed_max"):
            value = _finite_float(support.get(key))
            if value is not None:
                support_values.append(value)
        cell_rows.append({
            "cell": support["label"] or str(alias),
            "alias": str(alias),
            "coefficient": coefficient_value,
            "observed_runs": support["observed_runs"],
            "attempted_runs": support["attempted_runs"],
            "missing_runs": support["missing_runs"],
            "observed_min": support["observed_min"],
            "observed_max": support["observed_max"],
        })

    report["cells"] = cell_rows
    report["attempted_runs"] = total_attempted
    report["observed_runs"] = total_observed
    report["missing_runs"] = total_missing
    report["observed_fraction"] = (
        total_observed / total_attempted
        if total_attempted > 0
        else None
    )

    if deterministic:
        report["reason"] = (
            "确定性基线（如 ngram）换 seed 是同一估计的复制，"
            "不适用按运行缺失的部分识别界"
        )
        return report
    if total_attempted == 0:
        report["reason"] = (
            "没有计划运行/已观测运行：不是缺失问题，"
            "不生成部分识别界"
        )
        return report
    if total_observed == 0:
        report["verdict"] = "unbounded"
        report["reason"] = (
            "参与比较的单元没有任何已观测结果：无法给出有界"
            "结果假设下的有限界"
        )
        return report

    support_min = min(support_values)
    support_max = max(support_values)
    report["support"] = {
        "min": support_min,
        "max": support_max,
    }
    report["applicable"] = True
    report["complete_cases"] = total_missing == 0

    # 对线性组合的每个单元分别取「最不利/最有利」的缺失赋值。
    # 系数为正的单元要最小化就赋 L、最大化就赋 U；系数为负时
    # 方向相反。所有单元同时取各自极值即可得到组合的极值。
    effect_min_terms: List[float] = []
    effect_max_terms: List[float] = []
    for alias, coefficient in coefficients.items():
        support = supports[alias]
        observed = int(support["observed_runs"])
        attempted = int(support["attempted_runs"])
        observed_mean = _finite_float(
            support.get("observed_mean")
        )
        coefficient_value = _finite_float(coefficient)
        if coefficient_value is None:
            report["verdict"] = "not_applicable"
            report["reason"] = (
                f"单元 {alias} 的线性组合系数不是有限数值"
            )
            report["applicable"] = False
            return report
        if attempted <= 0 or observed <= 0 or observed_mean is None:
            # 整个单元都没有观测值时，所有运行都可取 L 或 U。
            mean_low = support_min
            mean_high = support_max
        else:
            # 用「观测均值 × 观测比例 + 极值 × 缺失比例」而不是
            # 先求总和：极大有限值直接相加可能溢出，比例写法
            # 在极端量级下仍不会抛异常（非有限结果由下方统一
            # 标成 unbounded）。
            observed_share = observed / attempted
            missing_share = 1.0 - observed_share
            mean_low = (
                observed_share * observed_mean
                + missing_share * support_min
            )
            mean_high = (
                observed_share * observed_mean
                + missing_share * support_max
            )
        if coefficient_value >= 0.0:
            effect_min_terms.append(
                coefficient_value * mean_low
            )
            effect_max_terms.append(
                coefficient_value * mean_high
            )
        else:
            effect_min_terms.append(
                coefficient_value * mean_high
            )
            effect_max_terms.append(
                coefficient_value * mean_low
            )

    def _bounded_sum(terms: List[float]) -> Optional[float]:
        direct = sum(terms)
        if math.isfinite(direct):
            return direct
        if any(not math.isfinite(term) for term in terms):
            return None
        scale = max(abs(term) for term in terms)
        if scale == 0.0:
            return 0.0
        scaled = sum(term / scale for term in terms)
        result = scaled * scale
        return result if math.isfinite(result) else None

    effect_min = _bounded_sum(effect_min_terms)
    effect_max = _bounded_sum(effect_max_terms)
    if effect_min is None or effect_max is None:
        report["verdict"] = "unbounded"
        report["reason"] = "极端赋值下的效应不是有限值"
        return report
    if effect_min > effect_max:
        effect_min, effect_max = effect_max, effect_min
    report["effect_min"] = effect_min
    report["effect_max"] = effect_max
    shift = (
        max(
            abs(effect_min - observed_effect_value),
            abs(effect_max - observed_effect_value),
        )
        if observed_effect_value is not None
        else None
    )
    report["max_abs_shift_from_observed"] = (
        shift if shift is not None and math.isfinite(shift) else None
    )
    # 容差随效应尺度缩放：固定 1e-12 会把 1e-13 量级的真实
    # 效应误判为 0（既可能伪造方向稳健，也可能把有方向的界
    # 说成跨零）。
    effect_scale = max(
        abs(effect_min),
        abs(effect_max),
        abs(observed_effect_value)
        if observed_effect_value is not None
        else 0.0,
    )
    tolerance = 1e-12 * effect_scale if effect_scale > 0.0 else 0.0
    report["observed_effect_within_bounds"] = bool(
        observed_effect_value is not None
        and effect_min - tolerance
        <= observed_effect_value
        <= effect_max + tolerance
    )
    report["zero_within_bounds"] = bool(
        effect_min - tolerance <= 0.0 <= effect_max + tolerance
    )

    def _bound_sign(value: Optional[float]) -> int:
        if value is not None and value > tolerance:
            return 1
        if value is not None and value < -tolerance:
            return -1
        return 0

    observed_sign = _bound_sign(observed_effect_value)
    low_sign = _bound_sign(effect_min)
    high_sign = _bound_sign(effect_max)
    sign_robust = bool(
        observed_sign != 0
        and low_sign == observed_sign
        and high_sign == observed_sign
    )

    if total_missing == 0:
        report["verdict"] = "complete_cases"
        # 即使没有缺失，完整案例的聚合均值口径也可能与配对
        # 效应不同（种子不完全对齐）。方向稳健性仍须同时核对
        # 观测效应与界的符号；观测效应为 0 时方向无定义。
        report["sign_robust"] = sign_robust
        report["note"] = (
            "没有缺失/失败运行：完整案例即全部计划运行，"
            "无需做缺失敏感性外推"
        )
        return report

    report["sign_robust"] = sign_robust
    if report["zero_within_bounds"]:
        report["verdict"] = "crosses_zero"
        report["reason"] = (
            "在观测支持范围内给缺失运行取极端值时，效应方向"
            "可能反转或变为 0：完整案例的显著结论不能当作对"
            "缺失机制稳健"
        )
    elif observed_sign == 0:
        report["verdict"] = "zero_effect"
        report["reason"] = "观测效应为 0，方向稳健性无定义"
    else:
        report["verdict"] = "bounded_same_sign"
        report["reason"] = (
            "在观测支持范围内取最不利/最有利的缺失赋值，效应"
            "方向仍与完整案例一致；结论对缺失机制稳健（仍以"
            "有界结果假设为前提）"
        )
    return report



def _attrition_bounds_summary(
    contrasts: Sequence[Dict[str, object]],
    label: str = "combined",
) -> Dict[str, object]:
    """
    汇总一组对比/因子效应的缺失敏感性界，供报告顶层落盘。

    只统计 ``attrition_bounds.applicable`` 的条目；确定性基线
    与没有任何观测的条目单独计数，避免把「无法计算」写成
    「没有缺失问题」。
    """
    def _format_bound(value: object) -> str:
        number = _finite_float(value)
        return "n/a" if number is None else f"{number:.4g}"

    entries: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    seen: set = set()
    for contrast in contrasts:
        bounds = contrast.get("attrition_bounds")
        if not isinstance(bounds, dict):
            continue
        baseline = contrast.get("baseline")
        variant = contrast.get("variant")
        design = contrast.get("design")
        effect_name = contrast.get("effect_name")
        if baseline is None and variant is None:
            baseline = design or effect_name
            variant = effect_name or baseline
        key = (
            str(label),
            tuple(sorted((str(baseline), str(variant)))),
            str(design),
            str(effect_name),
        )
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "family": label,
            "baseline": baseline,
            "variant": variant,
            "design": design,
            "effect_name": effect_name,
            "verdict": bounds.get("verdict"),
            "applicable": bool(bounds.get("applicable")),
            "complete_cases": bool(bounds.get("complete_cases")),
            "missing_runs": bounds.get("missing_runs"),
            "attempted_runs": bounds.get("attempted_runs"),
            "observed_runs": bounds.get("observed_runs"),
            "observed_effect": bounds.get("observed_effect"),
            "effect_min": bounds.get("effect_min"),
            "effect_max": bounds.get("effect_max"),
            "zero_within_bounds": bounds.get("zero_within_bounds"),
            "sign_robust": bounds.get("sign_robust"),
            "observed_effect_within_bounds": bounds.get(
                "observed_effect_within_bounds"
            ),
            "significant": bool(contrast.get("significant")),
        }
        if not bounds.get("applicable"):
            skipped.append(entry)
        else:
            entries.append(entry)

    verdict_counts: Dict[str, int] = {}
    for entry in entries:
        verdict = str(entry.get("verdict"))
        verdict_counts[verdict] = (
            verdict_counts.get(verdict, 0) + 1
        )
    with_missing = [
        entry for entry in entries
        if _nonnegative_int(entry.get("missing_runs")) > 0
    ]
    not_robust = [
        entry for entry in entries
        if entry.get("verdict") in {"crosses_zero", "unbounded"}
        or (
            _nonnegative_int(entry.get("missing_runs")) > 0
            and entry.get("sign_robust") is False
        )
    ]
    advisories: List[str] = []
    if not_robust:
        advisories.append(
            "以下对比在观测支持范围内给缺失/失败运行取极端值时，"
            "效应方向可能反转（或无法给出有限界）：完整案例的"
            "显著结论不能当作对缺失机制稳健，应补齐运行或报告"
            "失败率与界："
            + "; ".join(
                f"{entry['baseline']}→{entry['variant']}"
                f"（缺失 {entry.get('missing_runs')}/"
                f"{entry.get('attempted_runs')}，"
                f"界=[{_format_bound(entry.get('effect_min'))}, "
                f"{_format_bound(entry.get('effect_max'))}]）"
                for entry in not_robust
            )
        )
    elif with_missing:
        advisories.append(
            "有缺测/失败运行的对比在观测支持范围内仍保持效应"
            "方向；这以「缺失结果不超出观测范围」为前提，不能"
            "替代补齐运行。"
        )
    elif entries:
        advisories.append(
            "所有可核对对比都没有缺失/失败运行；无需做缺失"
            "敏感性外推。"
        )
    if skipped:
        advisories.append(
            f"{len(skipped)} 个条目无法计算缺失敏感性界"
            "（确定性基线或没有任何已观测结果）：不能把"
            "「未计算」读成「没有缺失问题」。"
        )
    outside_bounds = [
        entry for entry in entries
        if entry.get("observed_effect_within_bounds") is False
    ]
    if outside_bounds:
        advisories.append(
            "以下条目的配对点估计落在按聚合均值计算的缺失界之外"
            "（种子不完全对齐或均值/配对口径不同）：不能把该界"
            "直接套到配对效应上，应同时查看 common_seeds 与"
            "differential_missingness："
            + "; ".join(
                f"{entry['baseline']}→{entry['variant']}"
                for entry in outside_bounds
            )
        )
    return {
        "schema": "smelt-isr/attrition-summary/v1",
        "entries": entries,
        "skipped": skipped,
        "verdict_counts": verdict_counts,
        "with_missing": with_missing,
        "not_robust": not_robust,
        "advisories": advisories,
        "note": (
            "有界结果假设下的部分识别，不是置信区间；缺失值"
            "可能超出观测支持范围时真实界为无穷宽"
        ),
    }



def _cost_verdict_from_ratios(
    cost_ratios: Dict[str, float],
    verdict: str,
    significant: bool,
    noisy_timing_fields: Optional[Sequence[str]] = None,
    ratio_overflow_fields: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    由效率倍率与显著性给出权衡判定，供对比创建时与多重比较
    校正后重算共用（校正可能把显著翻转为不显著，权衡结论
    必须同步降级为 ``gain_not_significant``）。

    ``noisy_timing_fields`` 是不可信的计时字段：离散度过大、
    倍率区间跨 1，或缺少重复测量而无法估计不确定性。单次/
    少量运行的计时噪声可能伪造 10 倍成本，不能单独把结论判成
    ``gain_with_cost``。结构字段（参数量/FLOPs/峰值占用）不受
    此豁免；若全部字段都被标为不可信，退回使用全部倍率，避免
    因为缺证据而静默丢掉成本警告。

    ``cost_verdict``：
    - ``gain_with_cost``：指标显著改善，但效率倍率达到基线的
      ``_COST_RATIO_MATERIAL`` 倍以上，须评估取舍；
    - ``gain_balanced``：显著改善且效率倍率可控；
    - ``gain_not_significant``：改善不显著（说服力有限）；
    - ``dominated_regression``：指标变差且效率倍率更高，
      严格劣于基线；
    - ``neutral``：其余（持平/缺失效率数据等）。
    """
    if (
        not cost_ratios
        and not ratio_overflow_fields
    ) or verdict == "missing_metric":
        return None

    ignored = set(noisy_timing_fields or ())
    # 损坏的旧报告可能把倍率写成字符串/NaN；只接受有限数值。
    finite_ratios = {
        field: ratio
        for field, ratio in (
            (field, _finite_float(value))
            for field, value in cost_ratios.items()
        )
        if ratio is not None
    }
    considered = {
        field: ratio
        for field, ratio in finite_ratios.items()
        if field not in ignored
    }
    # 真实倍率可以越过 float 上限：它没有可表示的有限值，但
    # 「变体成本远高于基线」这一事实仍然成立，不能因为溢出就
    # 把材料成本当成 0/缺失。
    overflow = {
        field
        for field in (ratio_overflow_fields or ())
        if field not in ignored
    }
    if not considered and not overflow:
        considered = finite_ratios

    most_expensive = (
        max(considered.values()) if considered else None
    )
    expensive = bool(overflow) or (
        most_expensive is not None
        and most_expensive >= _COST_RATIO_MATERIAL
    )
    if verdict == "regression" and (
        bool(overflow)
        or (
            most_expensive is not None
            and most_expensive > 1.0
        )
    ):
        return "dominated_regression"
    if verdict == "improvement" and significant:
        return "gain_with_cost" if expensive else "gain_balanced"
    if verdict == "improvement":
        return "gain_not_significant"
    return "neutral"
