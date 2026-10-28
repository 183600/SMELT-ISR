from __future__ import annotations



def paired_window_comparison(
    payload_a: Dict[str, object],
    payload_b: Dict[str, object],
    include_differences: bool = False,
) -> Dict[str, object]:
    """
    两个运行在同一批固定评估窗口上的配对比较。

    固定评估窗口在所有运行之间逐位相同（见
    ``ByteDataset.fixed_eval_batch``），因此两个运行可以按窗口
    配对：差值 = b 的逐窗口指标 - a 的逐窗口指标。这给出
    「在这批固定评估窗口上」的一致性证据，与种子级检验（训练
    随机性）互补但**不独立**：两者来自同一批运行，不能当成两次
    独立验证。

    口径与解释边界：
    - 评估协议指纹（数据指纹/split/窗口数/批大小/seq_len）不
      一致时直接拒绝：窗口集合不同，配对没有意义；
    - 窗口级 p 值与区间只覆盖「评估集抽样」这一层不确定性，
      重叠窗口不是独立抽样单位，区间会偏窄（overlap_fraction
      一并给出）；正自相关时另给 AR(1) 有效窗口数校正后的
      p 值（`window_p_value_ar1`），原始 p 值保留作对照；
      训练随机性仍由种子级检验负责；
    - 差异方向按「b 相对 a」给出：loss 类指标越低越好，
      accuracy 类指标越高越好。

    ``include_differences=True`` 时额外返回逐窗口差值列表，供
    跨种子的两向 bootstrap 使用；默认不返回以控制报告体积。
    """
    fingerprint_a = _eval_window_fingerprint(payload_a)
    fingerprint_b = _eval_window_fingerprint(payload_b)
    if fingerprint_a != fingerprint_b:
        mismatched = [
            key
            for (key, value_a), (_, value_b)
            in zip(fingerprint_a, fingerprint_b)
            if value_a != value_b
        ]
        return {
            "status": "protocol_mismatch",
            "mismatched_keys": mismatched,
            "reason": (
                "两个运行的评估协议指纹不同（"
                + ", ".join(mismatched)
                + "）：不是同一批固定评估窗口，逐窗口配对无效"
            ),
        }

    windows_a = payload_a.get("windows")
    windows_b = payload_b.get("windows")
    if not isinstance(windows_a, list) or not isinstance(
        windows_b, list
    ):
        return {
            "status": "missing_windows",
            "reason": "评估窗口记录缺失，无法逐窗口配对",
        }
    if len(windows_a) != len(windows_b):
        return {
            "status": "window_count_mismatch",
            "n_windows_a": len(windows_a),
            "n_windows_b": len(windows_b),
            "reason": (
                "两个运行的窗口数不同：不是同一批窗口，逐窗口"
                "配对无效"
            ),
        }
    if not windows_a:
        return {
            "status": "missing_windows",
            "reason": "评估窗口记录为空，无法逐窗口配对",
        }

    loss_differences: List[float] = []
    accuracy_differences: List[float] = []
    invalid_loss_windows: List[int] = []
    invalid_accuracy_windows: List[int] = []
    for index, (window_a, window_b) in enumerate(
        zip(windows_a, windows_b)
    ):
        loss_a = _window_per_token_loss(window_a)
        loss_b = _window_per_token_loss(window_b)
        if loss_a is not None and loss_b is not None:
            difference = loss_b - loss_a
            if math.isfinite(difference):
                loss_differences.append(difference)
            else:
                invalid_loss_windows.append(index)
        else:
            invalid_loss_windows.append(index)
        accuracy_a = _window_accuracy(window_a)
        accuracy_b = _window_accuracy(window_b)
        if accuracy_a is not None and accuracy_b is not None:
            accuracy_differences.append(accuracy_b - accuracy_a)
        else:
            invalid_accuracy_windows.append(index)

    if invalid_loss_windows:
        return {
            "status": "missing_windows",
            "invalid_window_indices": invalid_loss_windows,
            "reason": (
                "窗口记录缺少有效损失或损失非有限（窗口下标："
                + ", ".join(
                    str(index) for index in invalid_loss_windows
                )
                + "）：无法完整逐窗口配对，不能只对剩余窗口"
                "选择性比较"
            ),
        }

    n_windows = len(loss_differences)
    mean_diff = _mean(loss_differences)
    if mean_diff is None:
        return {
            "status": "invalid_values",
            "reason": "窗口差值没有可用的有限均值",
        }
    better_fraction = (
        sum(1 for value in loss_differences if value < 0.0)
        / n_windows
    )
    tie_fraction = (
        sum(1 for value in loss_differences if value == 0.0)
        / n_windows
    )
    bootstrap = _paired_window_bootstrap_ci(
        loss_differences,
        _randomization_seed(loss_differences),
    )
    test = _paired_t_test(loss_differences)
    randomization = _paired_sign_flip_test(loss_differences)
    ar1_test = _ar1_adjusted_mean_test(loss_differences)

    result: Dict[str, object] = {
        "status": "ok",
        "n_windows": n_windows,
        "mean_loss_diff": mean_diff,
        "median_loss_diff": _median(loss_differences),
        "fraction_variant_better": better_fraction,
        "fraction_tied": tie_fraction,
        "window_p_value": test.get("p_value"),
        "window_test": test.get("test"),
        "window_randomization_p_value": randomization.get(
            "p_value"
        ),
        "window_lag1_autocorrelation": ar1_test.get(
            "lag1_autocorrelation"
        ),
        "window_effective_windows": ar1_test.get(
            "effective_windows"
        ),
        "window_ar1_adjustment_applied": ar1_test.get("applied"),
        "window_ci_inflation_ar1": ar1_test.get("ci_inflation"),
        "window_p_value_ar1": ar1_test.get("p_value"),
        "window_df_ar1": ar1_test.get("df"),
        "window_ar1_note": ar1_test.get("note"),
        "overlap_fraction": _finite_float(
            payload_a.get("overlap_fraction")
        ),
        "bootstrap_ci95": bootstrap,
    }
    if invalid_accuracy_windows:
        # 准确率只是附加口径：损失窗口完整时仍可比较损失，
        # 但不能把「剩余窗口」的准确率均值伪装成全部窗口的
        # 配对结果。与损失侧一样拒绝选择性缩小样本，并显式
        # 记录哪些窗口无法配对。
        result["accuracy_status"] = "missing_windows"
        result["invalid_accuracy_window_indices"] = (
            invalid_accuracy_windows
        )
        result["accuracy_unavailable_reason"] = (
            "窗口记录缺少有效准确率（窗口下标："
            + ", ".join(
                str(index) for index in invalid_accuracy_windows
            )
            + "）：不能只对剩余窗口计算准确率配对"
        )
    elif accuracy_differences:
        accuracy_mean = (
            sum(accuracy_differences) / len(accuracy_differences)
        )
        result["accuracy_status"] = "ok"
        result["n_accuracy_windows"] = len(
            accuracy_differences
        )
        result["mean_accuracy_diff"] = accuracy_mean
        result["fraction_variant_more_accurate"] = (
            sum(
                1
                for value in accuracy_differences
                if value > 0.0
            ) / len(accuracy_differences)
        )
    # 默认不把逐窗口差值写进结果（报告体积大）；两层 bootstrap
    # 需要按种子聚集的原始差值，调用方显式要求才返回，并在
    # 归档前由调用方取出、不落入 per_seed 明细。
    if include_differences:
        result["loss_differences"] = list(loss_differences)
    return result



def _cell_window_payloads_by_seed(
    cell: Dict[str, object],
) -> Tuple[Dict[int, Dict[str, object]], List[int]]:
    """
    实验单元内「种子 → 逐窗口评估记录」的映射，以及该单元内
    出现多次的随机种子。重复种子不能静默只取第一份运行时
    记录做窗口配对。
    """
    by_seed: Dict[int, Dict[str, object]] = {}
    duplicate_seeds: set = set()
    # window_payloads 可能来自旧版本或手工编辑的聚合记录：只有
    # list 才是逐窗口条目集合；标量/字典不能直接迭代，否则在
    # 对比阶段抛 TypeError，整份报告无法生成。
    raw_payloads = cell.get("window_payloads")
    if not isinstance(raw_payloads, list):
        raw_payloads = []
    for entry in raw_payloads:
        if not isinstance(entry, dict):
            continue
        seed = entry.get("seed")
        payload = entry.get("payload")
        if not isinstance(seed, int) or isinstance(seed, bool):
            continue
        if not isinstance(payload, dict):
            continue
        value = int(seed)
        if value in by_seed:
            duplicate_seeds.add(value)
        else:
            by_seed[value] = payload
    return by_seed, sorted(duplicate_seeds)



def _clean_window_differences(
    per_seed_differences: Dict[int, List[float]],
) -> Dict[int, List[float]]:
    """过滤非有限窗口差值；只保留精确整数种子键。"""
    cleaned: Dict[int, List[float]] = {}
    for seed, values in per_seed_differences.items():
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        if not isinstance(values, list):
            continue
        numeric = _finite_values(values)
        if numeric is None or not numeric:
            continue
        cleaned[int(seed)] = numeric
    # 先清理键再排序：损坏 JSON 可能混入字符串/None 等不可比较
    # 的键，直接 sorted() 会在报告阶段抛 TypeError。
    return {
        seed: cleaned[seed]
        for seed in sorted(cleaned)
    }



def _hierarchical_window_bootstrap_ci(
    per_seed_differences: Dict[int, List[float]],
    replicates: int = _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
) -> Dict[str, object]:
    """
    种子 × 评估窗口交叉设计的 bootstrap 区间。

    现有 ``paired_window_comparison`` 的 bootstrap 只在一个种子的
    窗口内重采样，``_cell_window_paired_analysis`` 又只把各 seed
    的窗口均值平均后当点估计——两者都没有回答「同时考虑训练
    随机性与评估窗口抽样时，总体均值差有多不确定」。

    固定评估窗口在所有种子上是**同一批**，因此设计是
    seed × window 交叉，而不是窗口嵌套在种子里。这里做两向
    （crossed）bootstrap：

    1. 有放回重采样种子；
    2. 有放回重采样同一批固定窗口（所有抽中的种子共用同一组
       窗口下标，保留「同一窗口跨种子配对」的结构）；
    3. 对交叉重采样后的窗口差值取种子平均，得到一次统计量。

    区间反映「换一批训练种子 + 换一批同分布评估窗口」的联合
    不确定性。种子数很少（3~6）时种子重采样很粗，百分位
    区间可能偏窄/不稳定；它只作为两层不确定性的描述，不能
    替代增加独立种子，也不改变种子级主检验的判定。
    """
    if not isinstance(per_seed_differences, dict):
        return {
            "available": False,
            "reason": "invalid_seed_window_differences",
        }
    try:
        replicate_count = int(replicates)
    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "reason": "invalid_bootstrap_replicates",
            "replicates": _finite_float(replicates),
        }
    if replicate_count <= 1:
        return {
            "available": False,
            "reason": "insufficient_bootstrap_replicates",
            "replicates": replicate_count,
        }
    if replicate_count > _MAX_BOOTSTRAP_REPLICATES:
        return {
            "available": False,
            "reason": "invalid_bootstrap_replicates",
            "replicates": None,
        }
    replicates = replicate_count
    cleaned = _clean_window_differences(per_seed_differences)
    usable = {
        seed: values
        for seed, values in cleaned.items()
        if len(values) >= 2
    }
    if len(usable) < 2:
        return {
            "available": False,
            "reason": "insufficient_seeds_or_windows",
            "n_seeds": len(usable),
            "note": (
                "两层 bootstrap 至少需要 2 个种子且每个种子至少 "
                "2 个评估窗口；不足时不伪造区间"
            ),
        }
    seed_order = sorted(usable)
    window_counts = {
        len(usable[seed]) for seed in seed_order
    }
    if len(window_counts) != 1:
        return {
            "available": False,
            "reason": "inconsistent_window_counts",
            "n_seeds": len(seed_order),
            "window_counts": [
                len(usable[seed]) for seed in seed_order
            ],
            "note": (
                "固定评估窗口在各运行间应逐位相同；窗口数不一致"
                "时不能用同一组窗口下标做交叉 bootstrap"
            ),
        }
    seed_count = len(seed_order)
    window_count = len(usable[seed_order[0]])
    flatten = [
        value
        for seed in seed_order
        for value in usable[seed]
    ]
    # 与单层窗口 bootstrap 相同：合法的大差值在重采样求均值时
    # 会把总和溢出成 inf，区间随之变成 NaN。两层 bootstrap 的
    # 均值同样对公共尺度等变，按所有种子/窗口的最大绝对值统一
    # 归一化，最后再还原；随机种子仍由原始值派生。
    scale = max(abs(value) for value in flatten)
    if scale > 1.0e150:
        scaled_usable = {
            seed: [value / scale for value in usable[seed]]
            for seed in seed_order
        }
    else:
        scale = 1.0
        scaled_usable = usable
    generator = torch.Generator().manual_seed(
        _randomization_seed(flatten)
    )
    # (R, S) 的种子重采样下标与 (R, W) 的窗口重采样下标：
    # 所有抽中的种子共用同一组窗口下标，保留跨种子配对结构。
    seed_indices = torch.randint(
        low=0,
        high=seed_count,
        size=(replicate_count, seed_count),
        generator=generator,
    )
    window_indices = torch.randint(
        low=0,
        high=window_count,
        size=(replicate_count, window_count),
        generator=generator,
    )
    resampled_means = torch.empty(
        (replicate_count, seed_count),
        dtype=torch.float64,
    )
    for column, seed in enumerate(seed_order):
        values = torch.tensor(
            scaled_usable[seed], dtype=torch.float64
        )
        resampled_means[:, column] = values[
            window_indices
        ].mean(dim=1)
    draws = torch.gather(
        resampled_means, 1, seed_indices
    ).mean(dim=1)
    low = float(torch.quantile(draws, 0.025).item()) * scale
    high = float(torch.quantile(draws, 0.975).item()) * scale
    return {
        "available": True,
        "low": low,
        "high": high,
        "replicates": replicate_count,
        "n_seeds": seed_count,
        "n_windows": window_count,
        "n_windows_min": window_count,
        "n_windows_max": window_count,
        "method": "crossed_seed_window_percentile_bootstrap",
        "resampling": "seeds_and_windows_with_replacement",
        "note": (
            "同时反映训练种子与固定评估窗口两层不确定性；"
            "窗口按可交换单位重采样，未做 AR(1) 膨胀，"
            "正自相关时区间可能偏窄；种子数少时重采样较粗，"
            "区间只作描述，不能替代增加独立种子或扩大评估集"
        ),
    }



def _window_variance_components(
    per_seed_differences: Dict[int, List[float]],
) -> Dict[str, object]:
    """
    用两向方差分析把 seed × window 交叉设计的方差分开。

    固定评估窗口在所有种子上是同一批，设计是 seed × window
    交叉而非窗口嵌套。对可加模型

        d_{s,w} = μ + a_s + b_w + e_{s,w}

    用 ANOVA 平方和估计三个方差分量：种子主效应 τ²、窗口主效应
    ω² 与种子×窗口交互/残差 σ²。总体均值 d̄ 的方差为

        Var(d̄) = τ²/S + ω²/W + σ²/(S·W)      （换种子且换窗口）

    若只换种子、仍用这批固定窗口，窗口主效应 b_w 被平均成常数，
    只剩

        Var(d̄ | 固定窗口) = τ²/S + σ²/(S·W) 。

    报告两层（三层）方差占比、种子均值的可靠性，以及「换一个
    新种子、仍用这批固定窗口」的预测区间。方差占比决定下一步
    该加种子还是该扩大评估集：种子主效应占主导时增加窗口无济于
    事；窗口主效应占主导时盲目加种子也不经济；交互占主导时
    需要同时增加两者。这是描述性的方差分解，不是新的显著性
    检验。

    直接取种子均值的样本方差会把窗口抽样噪声算进种子层，这里
    用两向 ANOVA 的均方扣除，避免重复计入。
    """
    cleaned = _clean_window_differences(per_seed_differences)
    usable = {
        seed: values
        for seed, values in cleaned.items()
        if len(values) >= 2
    }
    if not usable:
        return {
            "available": False,
            "reason": "no_seed_with_multiple_windows",
        }
    seed_order = sorted(usable)
    window_counts = [
        len(usable[seed]) for seed in seed_order
    ]
    seed_count = len(seed_order)
    if len(set(window_counts)) != 1:
        return {
            "available": False,
            "reason": "unbalanced_window_counts",
            "n_seeds": seed_count,
            "window_counts": window_counts,
            "note": (
                "两向方差分解要求每个种子使用同一批固定窗口；"
                "窗口数不一致时不做该分解"
            ),
        }
    window_count = window_counts[0]
    if seed_count < 2 or window_count < 2:
        return {
            "available": False,
            "reason": "insufficient_seeds_for_variance_components",
            "n_seeds": seed_count,
            "n_windows": window_count,
            "note": (
                "至少需要 2 个种子与 2 个窗口才能分离种子层与"
                "窗口层方差；不足时不伪造分解"
            ),
        }

    matrix = [usable[seed] for seed in seed_order]
    # 先按全局最大绝对值缩放再分解：ANOVA 的平方和在原始尺度
    # 上很容易越过 float 上限（合法输入如 1e200 就会让
    # ``** 2`` 抛 OverflowError）。方差占比与自由度对公共尺度
    # 不变；绝对分量在最后再按 scale² 还原，超出 float 范围时
    # 明确返回 None。
    max_abs = max(
        abs(value) for row in matrix for value in row
    )
    # 正常量级不缩放（scale=1.0），保持历史方差分量的逐位一致；
    # 平方可能溢出（超大）或下溢（极小但非零）时归一化。
    scale = (
        max_abs
        if max_abs > 1.0e150 or 0.0 < max_abs < 1.0e-150
        else 1.0
    )
    if scale == 0.0:
        scaled_matrix = [
            [0.0 for _ in row] for row in matrix
        ]
    else:
        scaled_matrix = [
            [value / scale for value in row]
            for row in matrix
        ]
    grand_mean_norm = (
        sum(sum(row) for row in scaled_matrix)
        / float(seed_count * window_count)
    )
    seed_means_norm = [
        sum(row) / float(window_count)
        for row in scaled_matrix
    ]
    window_means_norm = [
        sum(
            scaled_matrix[index][column]
            for index in range(seed_count)
        )
        / float(seed_count)
        for column in range(window_count)
    ]
    ss_seed_norm = window_count * sum(
        (mean - grand_mean_norm) ** 2
        for mean in seed_means_norm
    )
    ss_window_norm = seed_count * sum(
        (mean - grand_mean_norm) ** 2
        for mean in window_means_norm
    )
    ss_total_norm = sum(
        (value - grand_mean_norm) ** 2
        for row in scaled_matrix
        for value in row
    )
    # 浮点误差可能让三项之和大一点点；截断到 0 而不是产生
    # 负的方差分量。
    ss_error_norm = max(
        0.0, ss_total_norm - ss_seed_norm - ss_window_norm
    )
    df_seed = seed_count - 1
    df_window = window_count - 1
    df_error = df_seed * df_window
    ms_seed_norm = ss_seed_norm / float(df_seed)
    ms_window_norm = ss_window_norm / float(df_window)
    ms_error_norm = (
        ss_error_norm / float(df_error)
        if df_error > 0
        else 0.0
    )

    # 两向随机效应方差分量（可能为负，按惯例截断到 0，并记录）。
    raw_seed_var_norm = (
        ms_seed_norm - ms_error_norm
    ) / float(window_count)
    raw_window_var_norm = (
        ms_window_norm - ms_error_norm
    ) / float(seed_count)
    seed_var_norm = max(0.0, raw_seed_var_norm)
    window_var_norm = max(0.0, raw_window_var_norm)
    error_var_norm = max(0.0, ms_error_norm)

    mean_seed_var_norm = seed_var_norm / float(seed_count)
    mean_window_var_norm = window_var_norm / float(window_count)
    mean_error_var_norm = error_var_norm / float(
        seed_count * window_count
    )
    total_mean_var_norm = (
        mean_seed_var_norm
        + mean_window_var_norm
        + mean_error_var_norm
    )
    same_window_mean_var_norm = (
        mean_seed_var_norm + mean_error_var_norm
    )

    def _rescale_square(
        value: float,
        *,
        signed: bool = False,
    ) -> Optional[float]:
        """把归一化方差还原到原始尺度；溢出时返回 None。

        ``signed=True`` 保留可正可负的原始方差分量（截断前），
        用于诊断「随机效应方差被截断到 0」。
        """
        if not math.isfinite(value):
            return None
        if value < 0.0 and not signed:
            return None
        result = (value * scale) * scale
        if not math.isfinite(result):
            return None
        # 正的归一化分量还原后下溢为 0：真实方差/均方小于
        # float 可表示范围，报告为缺失而不是伪造零方差。
        if value != 0.0 and result == 0.0:
            return None
        return result

    def _rescale_root(value: float) -> Optional[float]:
        """把归一化标准差还原到原始尺度；溢出时返回 None。"""
        if not math.isfinite(value) or value < 0.0:
            return None
        result = math.sqrt(value) * scale
        return result if math.isfinite(result) else None

    grand_mean = grand_mean_norm * scale
    seed_var = _rescale_square(seed_var_norm)
    window_var = _rescale_square(window_var_norm)
    error_var = _rescale_square(error_var_norm)
    raw_seed_var = _rescale_square(
        raw_seed_var_norm, signed=True
    )
    raw_window_var = _rescale_square(
        raw_window_var_norm, signed=True
    )
    mean_seed_var = _rescale_square(mean_seed_var_norm)
    mean_window_var = _rescale_square(mean_window_var_norm)
    mean_error_var = _rescale_square(mean_error_var_norm)
    total_mean_var = _rescale_square(total_mean_var_norm)
    same_window_mean_var = _rescale_square(
        same_window_mean_var_norm
    )

    components: Dict[str, object] = {
        "available": True,
        "n_seeds": seed_count,
        "n_windows": window_count,
        "n_windows_total": seed_count * window_count,
        "n_windows_min": window_count,
        "n_windows_max": window_count,
        "seed_mean_effect": grand_mean,
        "between_seed_sd": _rescale_root(seed_var_norm),
        "window_sd": _rescale_root(window_var_norm),
        "within_seed_sd_mean": _rescale_root(error_var_norm),
        "within_seed_mean_sd": _rescale_root(
            error_var_norm / float(window_count)
        ),
        "variance_components_single_observation": {
            "seed": seed_var,
            "window": window_var,
            "interaction": error_var,
            "raw_seed": raw_seed_var,
            "raw_window": raw_window_var,
            "truncated_at_zero": bool(
                raw_seed_var_norm < 0.0
                or raw_window_var_norm < 0.0
            ),
        },
        "variance_components_of_mean": {
            "seed": mean_seed_var,
            "window": mean_window_var,
            "interaction": mean_error_var,
            "total": total_mean_var,
        },
        "variance_of_mean": total_mean_var,
        "std_error_of_mean": _rescale_root(
            total_mean_var_norm
        ),
        "variance_of_mean_same_windows": same_window_mean_var,
        "std_error_of_mean_same_windows": _rescale_root(
            same_window_mean_var_norm
        ),
        "variance_share_seed": None,
        "variance_share_window": None,
        "variance_share_interaction": None,
        "variance_share_seed_same_windows": None,
        "variance_share_interaction_same_windows": None,
        "seed_mean_reliability": None,
        "dominant_source": None,
        "dominant_source_same_windows": None,
        "mean_squares": {
            "seed": _rescale_square(ms_seed_norm),
            "window": _rescale_square(ms_window_norm),
            "error": _rescale_square(ms_error_norm),
        },
        "degrees_of_freedom": {
            "seed": df_seed,
            "window": df_window,
            "error": df_error,
        },
        "method": "two_way_anova_variance_components",
        "note": (
            "方差分解针对 seed × window 交叉设计；窗口主效应"
            "只在换评估窗口时计入，换种子但仍用固定窗口时不"
            "计入。窗口沿评估序列正自相关时窗口/交互分量会"
            "偏小；种子数少时方差分量估计很粗，占比只作"
            "方向性参考"
        ),
    }

    if total_mean_var_norm > 0.0:
        share_seed = mean_seed_var_norm / total_mean_var_norm
        share_window = (
            mean_window_var_norm / total_mean_var_norm
        )
        share_interaction = (
            mean_error_var_norm / total_mean_var_norm
        )
        components["variance_share_seed"] = share_seed
        components["variance_share_window"] = share_window
        components["variance_share_interaction"] = (
            share_interaction
        )
        if share_seed >= 0.8:
            dominant = "training_seed"
            advisory = (
                "不确定性主要来自训练种子：增加评估窗口不能"
                "显著收窄区间，应增加独立种子或降低种子间"
                "方差"
            )
        elif share_window >= 0.8:
            dominant = "evaluation_windows"
            advisory = (
                "不确定性主要来自评估窗口：盲目增加种子不"
                "经济，应扩大/改进评估集或降低窗口级噪声"
            )
        elif share_interaction >= 0.8:
            dominant = "seed_window_interaction"
            advisory = (
                "不确定性主要来自种子×窗口交互：需要同时增加"
                "独立种子并扩大评估集，单补一层不够"
            )
        else:
            dominant = "mixed"
            advisory = (
                "种子、窗口与交互三层方差都不小：需要同时"
                "增加独立种子并扩大评估集，单补一层不够"
            )
        components["dominant_source"] = dominant
        components["advisory"] = advisory

    if same_window_mean_var_norm > 0.0:
        components["variance_share_seed_same_windows"] = (
            mean_seed_var_norm / same_window_mean_var_norm
        )
        components["variance_share_interaction_same_windows"] = (
            mean_error_var_norm / same_window_mean_var_norm
        )
        if (
            mean_seed_var_norm / same_window_mean_var_norm
            >= 0.8
        ):
            components["dominant_source_same_windows"] = (
                "training_seed"
            )
        elif (
            mean_error_var_norm / same_window_mean_var_norm
            >= 0.8
        ):
            components["dominant_source_same_windows"] = (
                "seed_window_interaction"
            )
        else:
            components["dominant_source_same_windows"] = "mixed"

    reliability_denominator = seed_var_norm + (
        error_var_norm / float(window_count)
    )
    if reliability_denominator > 0.0:
        components["seed_mean_reliability"] = (
            seed_var_norm / reliability_denominator
        )

    if seed_count >= 3:
        critical = _t_critical(float(seed_count - 1))
        if critical is not None:
            predicted_sd = _rescale_root(
                seed_var_norm + error_var_norm / float(window_count)
            )
            if predicted_sd is not None:
                low = grand_mean - critical * predicted_sd
                high = grand_mean + critical * predicted_sd
                if math.isfinite(low) and math.isfinite(high):
                    components[
                        "new_seed_mean_prediction_interval_95"
                    ] = {
                        "low": low,
                        "high": high,
                        "df": seed_count - 1,
                        "t_critical": critical,
                        "method": "t_prediction_new_seed_mean",
                        "note": (
                            "同一批固定评估窗口上「换一个新训练"
                            "种子」的均值差预测区间；不包含窗口"
                            "主效应，不是换评估集的外推"
                        ),
                    }
    return components



def _cell_window_paired_analysis(
    cell_a: Dict[str, object],
    cell_b: Dict[str, object],
) -> Dict[str, object]:
    """
    两个实验单元之间的逐窗口配对分析（按 seed 配对运行）。

    固定评估窗口对所有运行相同，因此同一 seed 的两个运行天然
    可以按窗口配对；跨 seed 汇总时给出每个 seed 的窗口级差异与
    方向一致性，避免把不同 seed 的窗口混成一个假的大样本。
    """
    if not isinstance(cell_a, dict):
        cell_a = {}
    if not isinstance(cell_b, dict):
        cell_b = {}
    payloads_a, duplicate_a = _cell_window_payloads_by_seed(
        cell_a
    )
    payloads_b, duplicate_b = _cell_window_payloads_by_seed(
        cell_b
    )
    duplicate_seeds = sorted(set(duplicate_a) | set(duplicate_b))
    for seed in duplicate_seeds:
        payloads_a.pop(seed, None)
        payloads_b.pop(seed, None)
    common_seeds = sorted(set(payloads_a) & set(payloads_b))

    if not payloads_a and not payloads_b:
        return {
            "available": False,
            "reason": (
                "没有逐窗口评估记录（运行由旧版本产生或未落盘）："
                "无法给出窗口级配对证据"
                if not duplicate_seeds
                else "实验单元内存在重复随机种子（"
                + ", ".join(
                    str(seed) for seed in duplicate_seeds
                )
                + "），剔除后没有可唯一确定运行归属的逐窗口记录："
                "无法给出窗口级配对证据"
            ),
            "duplicate_seeds_excluded": duplicate_seeds,
        }
    if not common_seeds:
        return {
            "available": False,
            "reason": (
                "两个实验单元没有共同的随机种子：无法按 seed 配对"
                "逐窗口记录"
                + (
                    "（重复随机种子 "
                    + ", ".join(
                        str(seed) for seed in duplicate_seeds
                    )
                    + " 已剔除）"
                    if duplicate_seeds
                    else ""
                )
            ),
            "duplicate_seeds_excluded": duplicate_seeds,
        }

    per_seed: List[Dict[str, object]] = []
    seed_loss_differences: Dict[int, List[float]] = {}
    for seed in common_seeds:
        comparison = paired_window_comparison(
            payloads_a[seed],
            payloads_b[seed],
            include_differences=True,
        )
        differences = comparison.pop("loss_differences", None)
        if (
            comparison.get("status") == "ok"
            and isinstance(differences, list)
        ):
            seed_loss_differences[int(seed)] = [
                float(value) for value in differences
            ]
        per_seed.append({"seed": seed, **comparison})

    usable = [
        entry
        for entry in per_seed
        if entry.get("status") == "ok"
    ]
    if not usable:
        return {
            "available": False,
            "reason": (
                "逐窗口比较没有可用结果："
                + "; ".join(
                    str(entry.get("reason", entry.get("status")))
                    for entry in per_seed
                )
            ),
            "per_seed": per_seed,
            "duplicate_seeds_excluded": duplicate_seeds,
        }

    mean_diffs = [
        float(entry["mean_loss_diff"]) for entry in usable
    ]
    # 每个种子的窗口均值各自有限（如 1e308）时，跨种子直接
    # 求和仍可能溢出成 inf；用带缩放的均值保留真实点估计。
    overall_mean = _mean(mean_diffs)
    spread = max(mean_diffs) - min(mean_diffs)
    if not math.isfinite(spread):
        # 正负极大值相减会溢出：无法给出有限离散度时按缺失
        # 处理，不能把 inf 写进报告。
        spread = None
    favoring = sum(1 for value in mean_diffs if value < 0.0)
    against = sum(1 for value in mean_diffs if value > 0.0)
    # 两层不确定性：逐窗口 bootstrap（现有字段）只覆盖一个种子
    # 的窗口抽样；这里再把种子作为 cluster 纳入，并给出方差
    # 分解，回答「该加种子还是该扩大评估集」。
    hierarchical = _hierarchical_window_bootstrap_ci(
        seed_loss_differences
    )
    variance_components = _window_variance_components(
        seed_loss_differences
    )
    overlap_fractions = [
        entry.get("overlap_fraction")
        for entry in usable
        if isinstance(entry.get("overlap_fraction"), (int, float))
    ]
    notes: List[str] = []
    if overlap_fractions:
        notes.append(
            "评估窗口可能重叠（overlap_fraction="
            f"{_format_finite(max(overlap_fractions), '.3f')}）："
            "窗口不是独立抽样"
            "单位，窗口级原始 p 值偏小；已另给 AR(1) 有效窗口"
            "校正 p 值，训练随机性仍由种子级检验覆盖"
        )
    if duplicate_seeds:
        notes.append(
            "同一实验单元内出现重复随机种子（"
            + ", ".join(str(seed) for seed in duplicate_seeds)
            + "）：窗口级配对只使用归属唯一的种子；种子级"
            "检验先把同一 seed 折叠为一个观测，再做配对，"
            "复制运行不增加 n"
        )
    notes.append(
        "窗口级结果与种子级检验来自同一批运行，不是两次独立验证"
    )
    variance_advisory = variance_components.get("advisory")
    if isinstance(variance_advisory, str) and variance_advisory:
        notes.append(variance_advisory)
    return {
        "available": True,
        "n_seeds": len(usable),
        "hierarchical_bootstrap_ci95": hierarchical,
        "variance_components": variance_components,
        "seeds": [int(entry["seed"]) for entry in usable],
        "per_seed": per_seed,
        "mean_loss_diff": overall_mean,
        "mean_loss_diff_spread": (
            spread if len(mean_diffs) > 1 else 0.0
        ),
        "n_seeds_favoring_variant": favoring,
        "n_seeds_favoring_baseline": against,
        "direction_consistent": bool(
            favoring == 0 or against == 0
        ),
        "duplicate_seeds_excluded": duplicate_seeds,
        "notes": notes,
    }



def _leave_one_out_effect_stability(
    values: List[float],
) -> Dict[str, object]:
    """
    逐一去掉一个观测（种子）后重算效应，检查方向是否稳定。

    小样本（3~5 个种子）下，配对 t 检验显著也可能是由单个
    种子驱动的。若去掉某一种子后效应方向翻转或归零，说明
    结论不具备 leave-one-out 稳健性，展示层必须提示；这不是
    新的显著性检验，而是对「效应是否只来自一个运行」的
    描述性稳健性检查。
    """
    n = len(values)
    if n < _MIN_SEEDS_FOR_STATS:
        return {}
    numeric = _finite_values(values)
    if numeric is None:
        return {}

    full_effect = _mean(numeric)
    if full_effect is None:
        return {}
    # 逐个用剩余观测重新求均值，而不是 inf - value：后者在
    # 总和溢出时会把每个 leave-one-out 效应都变成 inf/nan。
    leave_one_out = [
        effect
        for effect in (
            _mean(numeric[:index] + numeric[index + 1:])
            for index in range(n)
        )
        if effect is not None
    ]
    if len(leave_one_out) != n:
        return {}
    # 容差必须随效应量缩放：固定 1e-12 会把 1e-13 量级的
    # 真实方向差异当成零效应，从而把方向翻转误报为稳定。
    tolerance = 1e-12 * abs(full_effect)
    if abs(full_effect) <= tolerance:
        stable = all(
            abs(value) <= tolerance
            for value in leave_one_out
        )
    else:
        # 直接比较符号，而不是 value * full_effect > 0：两个
        # 极小但同向的效应相乘会下溢成 0，把方向稳定误报成
        # 翻转。零效应仍按不稳定处理（与乘积口径一致）。
        stable = all(
            value != 0.0
            and (value > 0.0) == (full_effect > 0.0)
            for value in leave_one_out
        )

    return {
        "leave_one_out_effect_min": min(leave_one_out),
        "leave_one_out_effect_max": max(leave_one_out),
        "leave_one_out_sign_stable": stable,
    }



def _exact_sign_test(
    differences: List[float],
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Dict[str, object]:
    """
    配对差值中位数为 0 的精确符号检验（双侧二项检验）。

    与配对 t 检验（均值）和符号翻转检验（均值、要求差值分布
    关于 0 对称）不同，符号检验只看每个种子上的差值方向，
    完全不使用差值幅度：少数几个幅度极大的种子可以把均值
    拉显著，但只要多数种子方向相反，符号检验就不会显著。
    这正是「显著结论是否只由少数运行驱动」的直接检查。

    零差值按惯例丢弃（不提供方向信息），n 为非零差值个数。
    精确双侧 p 值按 P=0.5 的二项分布计算；最小可达 p 值为
    2/2^n（n 个非零差值同向），因此 n≤5 时该检验在 0.05
    水平上不可能显著，报告必须显式记录这一分辨率限制。
    """
    numeric = _finite_values(differences)
    if numeric is None:
        return {
            "test": "exact_binomial_sign",
            "available": False,
            "reason": "non_finite_differences",
        }
    nonzero = [value for value in numeric if value != 0.0]
    n = len(nonzero)
    positives = sum(1 for value in nonzero if value > 0.0)
    negatives = n - positives
    if n == 0:
        p_value = 1.0
        min_attainable = 1.0
    else:
        # 用整数分母做真除法：n>1023 时 float(2**n) 会溢出，
        # 而 Python 的大整数除法对可表示的结果仍能正确舍入。
        denominator = 2 ** n
        tail = sum(
            math.comb(n, index)
            for index in range(min(positives, negatives) + 1)
        ) / denominator
        p_value = min(1.0, 2.0 * tail)
        # math.ldexp 直接给出 2^(1-n)，n 极大时安全地下溢到 0。
        min_attainable = min(1.0, math.ldexp(2.0, -n))
    alpha_value = float(alpha)
    return {
        "test": "exact_binomial_sign",
        "available": True,
        "null_hypothesis": "median_paired_difference = 0",
        "n_nonzero": n,
        "n_zero": len(numeric) - n,
        "n_positive": positives,
        "n_negative": negatives,
        "p_value": p_value,
        "alpha": alpha_value,
        "significant": bool(
            n > 0 and p_value <= alpha_value
        ),
        "min_attainable_p": min_attainable,
        # n=0（全部差值为 0）时最小可达 p=1，同样不可能达到
        # alpha；它也必须被标为分辨率受限，不能与「有分辨率但
        # 未发现方向」混为一谈。
        "resolution_limited": bool(
            min_attainable > alpha_value
        ),
        "note": (
            "只看差值方向、不使用幅度；零差值被丢弃。"
            "最小可达双侧 p=2/2^n，n≤5 时在 0.05 水平上"
            "不可能显著"
        ),
    }



# 逐一删除种子做敏感性分析时，最多尝试删除多少个种子。
# n=6 时删除 3 个只剩 3 个，仍满足最小种子数；更大的 n
# 只枚举到 3 个，避免组合数失控（超过上限时明确记录
# 「未穷尽」，不能把未检查读成稳健）。
_FRAGILITY_MAX_DELETIONS = 3

_FRAGILITY_MAX_SUBSETS = 5000



def _paired_result_fragility(
    differences: List[float],
    seeds: Optional[Sequence[object]] = None,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Dict[str, object]:
    """
    配对结果的「单种子脆弱性」分析：显著性是否依赖少数种子。

    现有 leave-one-out 只检查**方向**是否翻转，不检查**显著性**
    是否消失。一个由单个极端差值撑起来的显著结果，去掉该种子
    后方向可能仍在，但 p 值已回到不显著；只看方向会把它误报成
    稳健。这里对每个可删除的种子子集重做配对 t 检验，报告：

    - ``max_leave_one_out_p_value``：删除任一种子后的最大 p；
    - ``significance_survives_leave_one_out``：是否所有单删
      结果在给定 alpha 下仍显著；
    - ``min_seed_deletions_to_lose_significance``：最少删除
      几个种子后不再显著（在 ``deletions_tested`` 范围内）；
    - ``verdict``：``stable`` / ``fragile`` /
      ``direction_mixed`` / ``not_significant`` /
      ``insufficient_seeds`` / ``not_evaluable``；
    - ``sign_test``：精确符号检验，单独回答「多数种子方向是否
      与均值效应一致」。

    ``alpha`` 应传**家族校正后**的阈值（该对比在 Holm
    step-down 中的位置阈值 alpha/(m-rank+1)），否则用未校正
    的 0.05 会系统性高估稳健性。删除种子不改变家族规模，
    因此沿用原位置阈值仍适用（删除后 p 排序可能变化，但
    不会放宽判定）。
    """
    n = len(differences)
    if n < _MIN_SEEDS_FOR_STATS:
        return {
            "available": False,
            "reason": "insufficient_seeds",
            "n": n,
            "min_seeds": _MIN_SEEDS_FOR_STATS,
        }
    numeric = _finite_values(differences)
    if numeric is None:
        return {
            "available": False,
            "reason": "non_finite_differences",
            "n": n,
        }

    mean = _mean(numeric)
    if mean is None:
        return {
            "available": False,
            "reason": "non_finite_differences",
            "n": n,
        }
    sd = _sample_std(numeric)
    base_test = _paired_t_test(numeric)
    base_p = base_test.get("p_value")
    if base_p is None:
        return {
            "available": False,
            "reason": "no_p_value",
            "n": n,
            "note": (
                "差值零方差或检验不可用：不能做删除种子的"
                "显著性敏感性分析"
            ),
        }
    alpha_value = float(alpha)
    significant = bool(float(base_p) <= alpha_value)
    # 删除一个种子后仍要有至少 _MIN_SEEDS_FOR_STATS 个观测，
    # 才能重做配对 t 检验；n=3 时单删只剩 2 个，不能伪装成
    # 「删除后仍显著」。
    leave_one_out_evaluable = bool(
        n > _MIN_SEEDS_FOR_STATS
    )

    labels: List[object] = (
        list(seeds)
        if isinstance(seeds, (list, tuple))
        and len(seeds) == n
        else list(range(n))
    )
    # 单删：报告每个被删除种子的影响，而不只给最大 p。
    leave_one_out: List[Dict[str, object]] = []
    for index in range(n):
        reduced = numeric[:index] + numeric[index + 1:]
        reduced_test = _paired_t_test(reduced)
        reduced_p = reduced_test.get("p_value")
        reduced_effect = reduced_test.get("effect")
        if reduced_effect is None:
            reduced_effect = _mean(reduced)
        leave_one_out.append({
            "seed": labels[index],
            "effect": reduced_effect,
            "p_value": reduced_p,
            "significant": bool(
                reduced_p is not None
                and float(reduced_p) <= alpha_value
            ),
        })
    valid_loo_p = [
        float(entry["p_value"])
        for entry in leave_one_out
        if entry.get("p_value") is not None
    ]
    max_loo_p = max(valid_loo_p) if valid_loo_p else None
    survives_loo = (
        bool(
            valid_loo_p
            and all(value <= alpha_value for value in valid_loo_p)
        )
        if leave_one_out_evaluable
        else None
    )
    sign_flip_loo = [
        entry
        for entry in leave_one_out
        if isinstance(entry.get("effect"), (int, float))
        and not isinstance(entry.get("effect"), bool)
        and float(entry["effect"]) != 0.0
        and mean != 0.0
        and (float(entry["effect"]) > 0.0) != (mean > 0.0)
    ]

    # 最少删除几个种子会失去显著性。只在剩余样本仍满足最小
    # 种子数时删除；删除 1 个已足够时不必枚举 2/3 个。
    max_deletions = min(
        _FRAGILITY_MAX_DELETIONS,
        max(0, n - _MIN_SEEDS_FOR_STATS),
    )
    min_deletions: Optional[int] = None
    min_deletions_p_value: Optional[float] = None
    deletions_tested = 0
    # 已实际进入枚举的最大删除个数：组合数超过上限时下一层
    # 从未被测试，不能把配置上限冒充成已覆盖范围。
    max_deletions_tested = 0
    exhaustive = True
    if significant and max_deletions > 0:
        for delete_count in range(1, max_deletions + 1):
            combinations = math.comb(n, delete_count)
            if combinations > _FRAGILITY_MAX_SUBSETS:
                exhaustive = False
                break
            max_deletions_tested = delete_count
            found = False
            for combo in itertools.combinations(
                range(n), delete_count
            ):
                deleted = set(combo)
                reduced = [
                    value
                    for index, value in enumerate(numeric)
                    if index not in deleted
                ]
                reduced_test = _paired_t_test(reduced)
                reduced_p = reduced_test.get("p_value")
                deletions_tested += 1
                if (
                    reduced_p is not None
                    and float(reduced_p) > alpha_value
                ):
                    min_deletions = delete_count
                    min_deletions_p_value = float(reduced_p)
                    found = True
                    break
            if found:
                break

    sign_test = _exact_sign_test(numeric, alpha=alpha_value)
    n_positive = int(sign_test.get("n_positive", 0))
    n_negative = int(sign_test.get("n_negative", 0))
    direction_consistent = bool(
        n_positive == 0 or n_negative == 0
    )
    if not significant:
        verdict = "not_significant"
    elif not leave_one_out_evaluable:
        verdict = "not_evaluable"
    elif min_deletions is not None:
        verdict = "fragile"
    elif sign_flip_loo:
        verdict = "fragile"
    elif not direction_consistent:
        verdict = "direction_mixed"
    else:
        verdict = "stable"

    return {
        "available": True,
        "verdict": verdict,
        "alpha": alpha_value,
        "n": n,
        "mean_difference": mean,
        "median_difference": _median(numeric),
        "sd_difference": sd,
        "min_difference": min(numeric),
        "max_difference": max(numeric),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": int(sign_test.get("n_zero", 0)),
        "fraction_positive": (
            float(n_positive) / float(n_positive + n_negative)
            if n_positive + n_negative > 0
            else None
        ),
        "direction_consistent": direction_consistent,
        "sign_test": sign_test,
        "leave_one_out": leave_one_out,
        "leave_one_out_evaluable": leave_one_out_evaluable,
        "max_leave_one_out_p_value": max_loo_p,
        "significance_survives_leave_one_out": survives_loo,
        "leave_one_out_sign_flip_seeds": [
            entry.get("seed") for entry in sign_flip_loo
        ],
        "min_seed_deletions_to_lose_significance": min_deletions,
        "min_seed_deletions_p_value": min_deletions_p_value,
        "deletions_tested": deletions_tested,
        "deletions_tested_exhaustively": bool(exhaustive),
        "max_deletions_tested": max_deletions_tested,
        "configured_max_deletions": max_deletions,
        "note": (
            "删除种子后重新做配对 t 检验；alpha 为家族校正后"
            "阈值。fragile 表示去掉不超过 max_deletions_tested "
            "个种子后显著性消失，不能作为稳健结论。"
            + (
                " 组合数超过上限，只检查了部分删除组合。"
                if not exhaustive
                else ""
            )
        ),
    }
