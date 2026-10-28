from __future__ import annotations



def _group_cluster_paired_comparison(
    payload_a: Dict[str, object],
    payload_b: Dict[str, object],
    include_differences: bool = False,
) -> Optional[Dict[str, object]]:
    """
    两个运行在同一批固定评估窗口上的文档/群组级配对比较。

    两个运行的 ``group_stats`` 必须来自同一群组定义（分隔符
    指纹与 assignment 版本一致），且共同群组的 token 数逐项
    相同；否则不是同一批评估单位，直接返回不可用原因。差值 =
    b − a，按群组内 token 加权汇总；bootstrap 以群组为单位
    有放回重采样。

    该区间只覆盖「换一批同分布文档」的评估不确定性；训练
    随机性仍由种子级检验负责。群组数很少时百分位区间很粗，
    只作描述性证据。
    """
    parsed_a = _parse_group_stats(payload_a)
    parsed_b = _parse_group_stats(payload_b)
    if parsed_a is None and parsed_b is None:
        return None
    if parsed_a is None or parsed_b is None:
        return {
            "available": False,
            "reason": "group_stats_missing_in_one_run",
        }
    if parsed_a["unit"] != parsed_b["unit"]:
        return {
            "available": False,
            "reason": "group_unit_mismatch",
            "unit_a": parsed_a["unit"],
            "unit_b": parsed_b["unit"],
        }
    if parsed_a["unit"] not in (None, "document_group"):
        return {
            "available": False,
            "reason": "unsupported_group_unit",
            "unit": parsed_a["unit"],
        }
    if (
        parsed_a["group_separator_sha256"]
        != parsed_b["group_separator_sha256"]
    ):
        return {
            "available": False,
            "reason": "group_separator_mismatch",
        }
    mapping_a = parsed_a["mapping"]
    mapping_b = parsed_b["mapping"]
    assert isinstance(mapping_a, dict)
    assert isinstance(mapping_b, dict)
    common = sorted(set(mapping_a) & set(mapping_b))
    if len(common) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_common_groups",
            "n_common_groups": len(common),
        }
    tokens: List[float] = []
    loss_differences: List[float] = []
    accuracy_differences: List[float] = []
    for group_id in common:
        tokens_a, loss_a, correct_a = mapping_a[group_id]
        tokens_b, loss_b, correct_b = mapping_b[group_id]
        if not math.isclose(
            tokens_a,
            tokens_b,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            return {
                "available": False,
                "reason": "group_token_count_mismatch",
                "group_id": group_id,
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
            }
        loss_differences.append(
            loss_b / tokens_b - loss_a / tokens_a
        )
        accuracy_differences.append(
            correct_b / tokens_b - correct_a / tokens_a
        )
        tokens.append(tokens_a)
    weight_tensor = torch.tensor(
        tokens,
        dtype=torch.float64,
    )
    difference_tensor = torch.tensor(
        loss_differences,
        dtype=torch.float64,
    )
    scale = float(difference_tensor.abs().max().item())
    if scale > 1.0e150 or 0.0 < scale < 1.0e-150:
        scaled_differences = difference_tensor / scale
    else:
        scale = 1.0
        scaled_differences = difference_tensor
    seed_values = loss_differences
    if all(math.isfinite(value) for value in loss_differences):
        seed_scale = max(
            abs(value) for value in loss_differences
        )
        if seed_scale > 0.0:
            seed_values = [
                float(f"{value / seed_scale:.12g}")
                for value in loss_differences
            ]
    generator = torch.Generator().manual_seed(
        _randomization_seed(seed_values)
    )
    interval = _group_stats_bootstrap_ci(
        scaled_differences,
        weight_tensor,
        _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
        generator,
    )
    if interval is None:
        return {
            "available": False,
            "reason": "bootstrap_unavailable",
            "n_common_groups": len(common),
        }
    low = interval[0] * scale
    high = interval[1] * scale
    weight_scale = float(weight_tensor.abs().max().item())
    if (
        weight_scale > 1.0e150
        or 0.0 < weight_scale < 1.0e-150
    ):
        mean_weights = weight_tensor / weight_scale
    else:
        mean_weights = weight_tensor
    weighted_mean = float(
        (
            scaled_differences * mean_weights
        ).sum().item()
        / float(mean_weights.sum().item())
    ) * scale
    unweighted_mean = _mean(loss_differences)
    accuracy_weights = weight_tensor
    accuracy_tensor = torch.tensor(
        accuracy_differences,
        dtype=torch.float64,
    )
    accuracy_interval = _group_stats_bootstrap_ci(
        accuracy_tensor,
        accuracy_weights,
        _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
        torch.Generator().manual_seed(
            _randomization_seed(accuracy_differences)
        ),
    )
    token_total = float(weight_tensor.sum().item())
    result: Dict[str, object] = {
        "available": True,
        "n_groups": len(common),
        "tokens": (
            token_total if math.isfinite(token_total) else None
        ),
        "group_separator_sha256": (
            parsed_a["group_separator_sha256"]
        ),
        "mean_loss_diff": weighted_mean,
        "unweighted_mean_loss_diff": unweighted_mean,
        "median_loss_diff": _median(loss_differences),
        "fraction_groups_better": (
            sum(
                1
                for value in loss_differences
                if value < 0.0
            )
            / len(loss_differences)
        ),
        "bootstrap_ci95": {
            "available": bool(
                math.isfinite(low) and math.isfinite(high)
            ),
            "low": low if math.isfinite(low) else None,
            "high": high if math.isfinite(high) else None,
            "replicates": _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
            "method": "group_cluster_percentile_bootstrap",
        },
        "sign_test": _exact_sign_test(loss_differences),
        "method": "document_group_cluster_bootstrap",
        "note": (
            "以文档/群组为单位的配对评估证据；差值按群组内 "
            "token 加权。群组数少时区间很粗，且与种子级检验"
            "来自同一批运行，不是独立验证"
        ),
    }
    if accuracy_interval is not None:
        result["mean_accuracy_diff"] = (
            float(
                (
                    accuracy_tensor * mean_weights
                ).sum().item()
            )
            / float(mean_weights.sum().item())
        )
        result["accuracy_bootstrap_ci95"] = {
            "low": min(
                1.0, max(-1.0, accuracy_interval[0])
            ),
            "high": min(
                1.0, max(-1.0, accuracy_interval[1])
            ),
            "replicates": _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
        }
    if include_differences:
        result["group_differences"] = [
            [
                int(group_id),
                float(difference),
                float(token),
            ]
            for group_id, difference, token in zip(
                common,
                loss_differences,
                tokens,
            )
        ]
    return result



def _hierarchical_group_bootstrap_ci(
    per_seed_group_differences: Dict[
        int, Dict[int, Tuple[float, float]]
    ],
    replicates: int = _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
) -> Dict[str, object]:
    """
    种子 × 文档/群组交叉 design 的两层 cluster bootstrap。

    每个种子给出「群组 → (差值, token 数)」；只保留所有种子
    都有的群组，并核对 token 数一致。先有放回重采样种子，再
    对所有抽中的种子共用同一组群组下标做有放回重采样，最后
    对交叉重采样后的 token 加权差值取种子平均。区间反映
    「换一批训练种子 + 换一批同分布文档」的联合不确定性。
    """
    if not isinstance(per_seed_group_differences, dict):
        return {
            "available": False,
            "reason": "invalid_seed_group_differences",
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
    cleaned: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for seed, mapping in per_seed_group_differences.items():
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(mapping, dict)
        ):
            continue
        entry: Dict[int, Tuple[float, float]] = {}
        for group_id, value in mapping.items():
            if (
                not isinstance(group_id, int)
                or isinstance(group_id, bool)
                or group_id < 0
                or not isinstance(value, (list, tuple))
                or len(value) < 2
            ):
                continue
            difference = _finite_float(value[0])
            tokens = _finite_float(value[1])
            if (
                difference is None
                or tokens is None
                or tokens <= 0.0
            ):
                continue
            entry[int(group_id)] = (difference, tokens)
        if entry:
            cleaned[int(seed)] = entry
    if len(cleaned) < 2:
        return {
            "available": False,
            "reason": "insufficient_seeds_or_groups",
            "n_seeds": len(cleaned),
        }
    seed_order = sorted(cleaned)
    common_groups = set(cleaned[seed_order[0]])
    for seed in seed_order[1:]:
        common_groups &= set(cleaned[seed])
    group_order = sorted(common_groups)
    if len(group_order) < 2:
        return {
            "available": False,
            "reason": "insufficient_seeds_or_groups",
            "n_seeds": len(seed_order),
            "n_groups": len(group_order),
        }
    reference_tokens = [
        cleaned[seed_order[0]][group_id][1]
        for group_id in group_order
    ]
    for seed in seed_order[1:]:
        tokens = [
            cleaned[seed][group_id][1]
            for group_id in group_order
        ]
        if any(
            not math.isclose(
                left,
                right,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            for left, right in zip(reference_tokens, tokens)
        ):
            return {
                "available": False,
                "reason": "group_token_count_mismatch",
                "n_seeds": len(seed_order),
                "n_groups": len(group_order),
            }
    seed_count = len(seed_order)
    group_count = len(group_order)
    values = torch.tensor(
        [
            [
                cleaned[seed][group_id][0]
                for group_id in group_order
            ]
            for seed in seed_order
        ],
        dtype=torch.float64,
    )
    tokens = torch.tensor(
        reference_tokens,
        dtype=torch.float64,
    )
    scale = float(values.abs().max().item())
    if scale > 1.0e150 or 0.0 < scale < 1.0e-150:
        values = values / scale
    else:
        scale = 1.0
    token_scale = max(reference_tokens)
    if token_scale > 1.0e150 or 0.0 < token_scale < 1.0e-150:
        token_weights = tokens / token_scale
    else:
        token_weights = tokens
    value_scale = float(values.abs().max().item())
    if math.isfinite(value_scale) and value_scale > 0.0:
        seed_values = [
            [
                float(f"{value:.12g}")
                for value in row
            ]
            for row in (values / value_scale).tolist()
        ]
    else:
        seed_values = values.tolist()
    token_seed_scale = max(
        abs(value) for value in reference_tokens
    )
    if math.isfinite(token_seed_scale) and token_seed_scale > 0.0:
        seed_tokens = [
            float(f"{value / token_seed_scale:.12g}")
            for value in reference_tokens
        ]
    else:
        seed_tokens = list(reference_tokens)
    generator = torch.Generator().manual_seed(
        _randomization_seed(
            [
                value
                for row in seed_values
                for value in row
            ]
            + seed_tokens
        )
    )
    seed_indices = torch.randint(
        low=0,
        high=seed_count,
        size=(replicate_count, seed_count),
        generator=generator,
    )
    group_indices = torch.randint(
        low=0,
        high=group_count,
        size=(replicate_count, group_count),
        generator=generator,
    )
    sampled_groups = group_indices
    sampled_tokens = token_weights[sampled_groups]
    sampled_total = sampled_tokens.sum(dim=1)
    usable_draws = (
        torch.isfinite(sampled_total) & (sampled_total > 0.0)
    )
    if not bool(usable_draws.any()):
        return {
            "available": False,
            "reason": "bootstrap_unavailable",
            "n_seeds": seed_count,
            "n_groups": group_count,
        }
    sampled_groups = sampled_groups[usable_draws]
    sampled_tokens = sampled_tokens[usable_draws]
    sampled_total = sampled_total[usable_draws]
    seed_indices = seed_indices[usable_draws]
    seed_means = torch.empty(
        (int(sampled_groups.size(0)), seed_count),
        dtype=torch.float64,
    )
    for column in range(seed_count):
        sampled_values = values[column][sampled_groups]
        seed_means[:, column] = (
            (sampled_values * sampled_tokens).sum(dim=1)
            / sampled_total
        )
    draws = (
        torch.gather(seed_means, 1, seed_indices).mean(dim=1)
        * scale
    )
    low = float(torch.quantile(draws, 0.025).item())
    high = float(torch.quantile(draws, 0.975).item())
    if not (math.isfinite(low) and math.isfinite(high)):
        return {
            "available": False,
            "reason": "unrepresentable_bounds",
            "n_seeds": seed_count,
            "n_groups": group_count,
        }
    return {
        "available": True,
        "low": low,
        "high": high,
        "replicates": int(replicates),
        "n_seeds": seed_count,
        "n_groups": group_count,
        "method": "crossed_seed_group_percentile_bootstrap",
        "resampling": "seeds_and_document_groups_with_replacement",
        "note": (
            "同时反映训练种子与文档/群组两层不确定性；"
            "群组按可交换单位重采样。种子或群组数很少时"
            "百分位区间很粗，只作描述性证据"
        ),
    }



def _bootstrap_window_indices(
    n_windows: int,
    replicates: int,
    generator: torch.Generator,
    block_size: int = 1,
) -> torch.Tensor:
    """窗口重采样下标：i.i.d. 或循环移动块。

    block_size>1 时先抽取 n_blocks=ceil(n/block_size) 个块起点，
    再拼接连续窗口（循环取模），最后截断到 n。它把相邻窗口
    成块重采样，部分吸收局部自相关；按文档/来源做 cluster
    bootstrap 仍然更合适，这里只是窗口级近似。
    """
    n = int(n_windows)
    reps = int(replicates)
    try:
        requested_block = int(block_size)
    except (TypeError, ValueError, OverflowError):
        requested_block = 1
    if requested_block <= 1:
        return torch.randint(
            low=0,
            high=n,
            size=(reps, n),
            generator=generator,
        )
    # 块长超过窗口数时，前 n 个循环偏移量本来就恰好是 0..n-1
    # 的一个排列；按 n 截断只改变中间分配大小，不改变抽样结果。
    # 不截断会让 --eval-bootstrap-block 的合法超大整数先分配
    # arange(block) 并耗尽内存。
    block = max(1, min(n, requested_block))
    n_blocks = int(math.ceil(n / block))
    starts = torch.randint(
        low=0,
        high=n,
        size=(reps, n_blocks),
        generator=generator,
    )
    offsets = torch.arange(block)
    indices = (
        starts.unsqueeze(-1) + offsets
    ) % n
    return indices.reshape(reps, -1)[:, :n]



def _bootstrap_eval_intervals(
    window_loss_sums: torch.Tensor,
    window_tokens: torch.Tensor,
    window_true_positive: torch.Tensor,
    window_predicted: torch.Tensor,
    window_support: torch.Tensor,
    replicates: int,
    seed: int,
    ci_level: float = _EVAL_BOOTSTRAP_CI_LEVEL,
    window_class_loss_sums: Optional[torch.Tensor] = None,
    block_size: int = 1,
) -> Dict[str, object]:
    """
    对固定评估窗口做有放回 bootstrap，给出评估集层面的
    95% 置信区间。

    覆盖的指标：
    - 似然口径：loss / ppl；以及传入逐窗口逐类损失时
      的 macro NLL（与点估计同口径，对出现类等权）；
    - 分类口径：token accuracy（micro）、macro / weighted
      F1、macro / weighted precision / recall。

    口径说明：
    - 重采样单位是完整评估窗口，不是 token：同一窗口内的
      token 由同一段上下文产生，逐 token 重采样会把相关样本
      当成独立样本，低估评估不确定性；
    - macro / weighted F1、precision、recall 的类别集合按
      每个重采样中实际出现的类确定（support>0），与点估计
      「只对评估集中出现过的类平均 / 按 support 加权」完全
      同口径。若固定为全量类别集合、把重采样缺席的稀有类按
      F1=0 计入，区间会系统性偏低，甚至不覆盖点估计；
    - macro NLL 同样只对本次重采样中出现过的类等权平均；
      逐窗口逐类损失足够统计量由评估循环与混淆统计共用同一次
      前向，不额外跑模型；
    - 重采样不进入训练、选型或超参数决策，只用于报告
      「测试集只评估一次」时点估计本身的抽样误差；
    - AR(1) 方差膨胀是一阶近似，膨胀后的区间会夹到指标的
      物理取值域（loss≥0、ppl≥1、比例类∈[0,1]）；
    - 只覆盖混淆矩阵/似然口径的指标；AUC 等排序指标未纳入。

    返回的键不带 split 前缀，由 ``evaluate`` 统一加前缀。
    """
    n_windows = int(window_tokens.numel())
    try:
        replicate_count = int(replicates)
    except (TypeError, ValueError, OverflowError):
        return {}
    try:
        requested_block = int(block_size)
    except (TypeError, ValueError, OverflowError):
        requested_block = 1
    if isinstance(ci_level, bool) or not isinstance(
        ci_level, (int, float)
    ):
        return {}
    try:
        level = float(ci_level)
    except (TypeError, ValueError, OverflowError):
        return {}
    try:
        seed_value = int(seed) % (2 ** 64)
    except (TypeError, ValueError, OverflowError):
        return {}
    if (
        replicate_count <= 1
        or replicate_count > _MAX_BOOTSTRAP_REPLICATES
        or n_windows <= 0
        or not math.isfinite(level)
        or not 0.0 < level < 1.0
    ):
        return {}
    block = max(1, min(n_windows, requested_block))

    generator = torch.Generator().manual_seed(seed_value)
    indices = _bootstrap_window_indices(
        n_windows,
        replicate_count,
        generator,
        block_size=block,
    )

    loss_sums = window_loss_sums.to(torch.float64)
    tokens = window_tokens.to(torch.float64)
    sampled_loss = (
        loss_sums[indices].sum(dim=1)
        / tokens[indices].sum(dim=1)
    )
    sampled_ppl = torch.exp(sampled_loss.clamp(max=80.0))

    true_positive = window_true_positive.to(torch.float64)
    predicted = window_predicted.to(torch.float64)
    support = window_support.to(torch.float64)

    sampled_tp = true_positive[indices].sum(dim=1)
    sampled_predicted = predicted[indices].sum(dim=1)
    sampled_support = support[indices].sum(dim=1)
    sampled_total_support = sampled_support.sum(dim=1)

    sampled_accuracy = (
        sampled_tp.sum(dim=1)
        / sampled_total_support.clamp_min(1.0)
    )

    precision = sampled_tp / sampled_predicted.clamp_min(1.0)
    recall = sampled_tp / sampled_support.clamp_min(1.0)
    denominator = precision + recall
    f1 = torch.zeros_like(sampled_tp)
    nonzero = denominator > 0.0
    f1[nonzero] = (
        2.0
        * precision[nonzero]
        * recall[nonzero]
        / denominator[nonzero]
    )
    # 每个重采样都有自己的类别集合：只对本次抽样中 support>0
    # 的类求平均。固定全量类别集合会把「本次抽样恰好没有该
    # 稀有类」当成 F1=0，使 bootstrap 分布整体下移。
    sampled_present = sampled_support > 0.0
    sampled_class_counts = sampled_present.sum(dim=1)
    sampled_macro_f1 = (
        (f1 * sampled_present).sum(dim=1)
        / sampled_class_counts.clamp_min(1.0)
    )
    sampled_precision_macro = (
        (precision * sampled_present).sum(dim=1)
        / sampled_class_counts.clamp_min(1.0)
    )
    sampled_recall_macro = (
        (recall * sampled_present).sum(dim=1)
        / sampled_class_counts.clamp_min(1.0)
    )
    sampled_weighted_f1 = (
        (f1 * sampled_support).sum(dim=1)
        / sampled_total_support.clamp_min(1.0)
    )
    sampled_precision_weighted = (
        (precision * sampled_support).sum(dim=1)
        / sampled_total_support.clamp_min(1.0)
    )
    sampled_recall_weighted = (
        (recall * sampled_support).sum(dim=1)
        / sampled_total_support.clamp_min(1.0)
    )
    degenerate_macro = sampled_class_counts == 0
    if bool(degenerate_macro.any()):
        for values in (
            sampled_macro_f1,
            sampled_precision_macro,
            sampled_recall_macro,
        ):
            values[degenerate_macro] = float("nan")

    # 类平衡似然的逐窗口逐类损失和：按同一组窗口索引重采样，
    # 再对本次重采样中出现过的类等权平均每类 NLL。
    sampled_macro_nll: Optional[torch.Tensor] = None
    if (
        window_class_loss_sums is not None
        and int(window_class_loss_sums.size(0)) == n_windows
    ):
        class_losses = window_class_loss_sums.to(torch.float64)
        sampled_class_loss = torch.empty(
            (int(replicates), int(class_losses.size(1))),
            dtype=torch.float64,
        )
        # 分块累加，避免 replicates × windows × vocab 的
        # 三维中间张量在较大评估集上占用过多内存。
        chunk = 64
        for begin in range(0, int(replicates), chunk):
            stop = min(begin + chunk, int(replicates))
            sampled_class_loss[begin:stop] = (
                class_losses[indices[begin:stop]].sum(dim=1)
            )
        per_class_nll = (
            sampled_class_loss
            / sampled_support.clamp_min(1.0)
        )
        sampled_macro_nll = (
            (per_class_nll * sampled_present).sum(dim=1)
            / sampled_class_counts.clamp_min(1.0)
        )
        sampled_macro_nll[degenerate_macro] = float("nan")

    # 窗口间 lag-1 自相关：bootstrap 把窗口当 i.i.d. 抽样单位，
    # 但固定窗口沿评估序列排列，窗口指标常随数据顺序渐变（同一
    # 类块、主题相邻等），正自相关会让区间偏窄。ρ̂1 达到阈值时
    # 按方差膨胀因子 √(n/n_eff) 放宽全部区间（以区间中点为中心
    # 等比放大，一阶近似；只放宽不收窄，负相关不用于缩窄）。
    per_window_loss = (
        window_loss_sums.to(torch.float64)
        / window_tokens.to(torch.float64).clamp_min(1.0)
    )
    rho1 = _lag1_autocorrelation(per_window_loss)
    effective_windows, inflation = _ar1_effective_windows(
        n_windows,
        rho1,
    )

    result: Dict[str, object] = {
        "bootstrap_replicates": int(replicates),
        "bootstrap_windows": n_windows,
        "bootstrap_ci_level": level,
        "window_lag1_autocorrelation": rho1,
        "window_effective_windows": effective_windows,
        "window_ci_inflation": inflation,
    }

    # 指标的物理取值域：AR(1) 膨胀是一阶近似，膨胀后可能
    # 把区间推到指标不可能出现的范围（如负准确率、loss<0）。
    # 写出报告前夹回域内，避免读者把不可能的边界当成有效
    # 估计；夹取只放宽到域边界，不会把区间缩窄到点估计之外。
    metric_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        "loss": (0.0, None),
        "ppl": (1.0, None),
        "macro_nll": (0.0, None),
        "token_accuracy": (0.0, 1.0),
        "macro_f1": (0.0, 1.0),
        "weighted_f1": (0.0, 1.0),
        "precision_macro": (0.0, 1.0),
        "precision_weighted": (0.0, 1.0),
        "recall_macro": (0.0, 1.0),
        "recall_weighted": (0.0, 1.0),
    }

    metric_series = [
        ("loss", sampled_loss),
        ("ppl", sampled_ppl),
        ("token_accuracy", sampled_accuracy),
        ("macro_f1", sampled_macro_f1),
        ("weighted_f1", sampled_weighted_f1),
        ("precision_macro", sampled_precision_macro),
        ("precision_weighted", sampled_precision_weighted),
        ("recall_macro", sampled_recall_macro),
        ("recall_weighted", sampled_recall_weighted),
    ]
    if sampled_macro_nll is not None:
        metric_series.append(("macro_nll", sampled_macro_nll))

    for name, values in metric_series:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            continue
        # 退化重采样（如 macro F1 抽到的窗口没有任何 support）
        # 会产生 NaN；直接对含 NaN 的张量取分位数会让整个区间
        # 变成 NaN。只用有限重采样估计区间，并显式记录退化数。
        degenerate = int(values.numel() - finite.numel())
        if degenerate > 0:
            result[f"{name}_bootstrap_degenerate"] = degenerate
        low, high = _percentile_interval(finite, level)
        if inflation > 1.0:
            # 所有纳入区间的指标都非负，用 low + (high - low)/2
            # 而不是 (low + high)/2：后者在端点都接近 float
            # 上限时会先溢出成 inf，把合法区间变成 ±inf。
            midpoint = low + (high - low) / 2.0
            half_width = (high - low) / 2.0 * inflation
            low = midpoint - half_width
            high = midpoint + half_width
            if not (
                math.isfinite(low) and math.isfinite(high)
            ):
                # 膨胀后的区间超出 float 可表示范围：不写出
                # inf/NaN 端点，保留点估计与诊断量。
                result[f"{name}_bootstrap_unavailable"] = (
                    "unrepresentable_bounds"
                )
                continue
        lower_bound, upper_bound = metric_bounds[name]
        if lower_bound is not None:
            low = max(low, lower_bound)
        if upper_bound is not None:
            high = min(high, upper_bound)
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high

    result["bootstrap_unit"] = (
        "moving_block" if block > 1 else "window_iid"
    )
    result["bootstrap_block_size"] = int(block)
    result["bootstrap_note"] = (
        "窗口级重采样只近似评估不确定性；连续/重叠窗口应按"
        "文档/来源做 cluster bootstrap，块 bootstrap 仅部分"
        "吸收局部自相关"
    )
    return result



def macro_auc_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    min_support: int = _AUC_MIN_SUPPORT,
) -> Dict[str, object]:
    """
    one-vs-rest 的 macro PR-AUC 与 ROC-AUC。

    probabilities: [n_tokens, vocab] 的预测分布；
    targets: [n_tokens] 的目标类。只统计 support 达到
    min_support 的类，返回值同时记录纳入类数，保证
    「macro」的口径（对哪些类求平均）可追溯。

    ROC-AUC 的 TPR/FPR 都以真实类别为条件，其期望不随
    prevalence 变化；负样本更多只会让 FPR 估计更精确，不会
    把 AUC 本身抬高。真正的问题是：正类极稀有且分数区分度
    一般时，高 ROC-AUC 仍可能对应很低的 precision/recall，
    PR-AUC 的随机水平又等于 prevalence——因此排序指标要
    报告 PR-AUC 相对随机水平的 lift 与 precision@k，而不是
    把绝对 PR-AUC 当归一化准确率。

    同时返回 ``pr_auc_macro_chance``：macro PR-AUC 的随机
    水平（各纳入类 prevalence 的均值），以及
    ``pr_auc_macro_lift`` = ``pr_auc_macro - chance``。PR-AUC
    本身随正类比例缩放，脱离该水平看绝对值没有意义；
    1:1000 级别的不平衡下随机 PR-AUC 只有约 0.001，而随机
    ROC-AUC 仍是 0.5，两者的「及格线」完全不同。lift 的
    随机水平恒为 0，是不随 prevalence 缩放的一等指标，
    可直接用于预注册、跨运行对比与稻草人核对。

    每个类独立做 one-vs-rest：ROC-AUC 为并列分数取平均
    秩的 Mann-Whitney 统计量；PR-AUC 按相异分数阈值分步
    累计（与 sklearn average_precision_score 的阶梯式口径
    一致）。正类或负类为空的类按对应口径不计入 macro。

    实现按「类块 × 全部 token」分块：块内对各类分数一次
    批量排序后向量化统计，逐类结果与逐类单独排序的实现
    逐位一致——组秩与组边界均为整数运算（半整数秩以 2 倍
    整数累计，求和精确且与逐类求和同值），PR-AUC 分段
    求和的加数顺序也保持不变。
    """
    probabilities = probabilities.detach().cpu().float()
    targets = targets.detach().cpu()

    support = torch.bincount(
        targets,
        minlength=vocab_size,
    )
    eligible = torch.nonzero(
        support >= min_support
    ).squeeze(1)

    n_tokens = int(targets.numel())

    pr_aucs: List[float] = []
    roc_aucs: List[float] = []
    pr_prevalence_sum = 0.0

    if eligible.numel() > 0 and n_tokens > 0:
        # 块大小（类数 × 评估位置数）固定，块内中间量规模
        # 不随评估集增大而膨胀。
        chunk_size = max(
            1,
            _AUC_SORT_CHUNK_ELEMENTS // n_tokens,
        )
        index_i32 = torch.arange(n_tokens, dtype=torch.int32)
        invalid_i32 = torch.tensor(-1, dtype=torch.int32)
        n_i32 = torch.tensor(n_tokens, dtype=torch.int32)

        for begin in range(0, eligible.numel(), chunk_size):
            chunk = eligible[begin:begin + chunk_size]
            n_classes = int(chunk.numel())

            # 块内每个类一列：分数升序排序，ROC-AUC 与
            # PR-AUC 复用同一次排序结果。
            scores = probabilities[:, chunk].t().contiguous()
            labels = (
                targets.unsqueeze(0) == chunk.unsqueeze(1)
            )
            n_pos = labels.sum(dim=1)
            n_pos_list = n_pos.tolist()

            sorted_scores, order = torch.sort(
                scores, dim=1
            )
            sorted_labels = labels.gather(1, order)

            # 并列组：change 标记组起点；组内每个位置的
            # 1 基平均秩为 (start + end_excl + 1) / 2。
            change = torch.ones_like(
                sorted_scores, dtype=torch.bool
            )
            change[:, 1:] = (
                sorted_scores[:, 1:] != sorted_scores[:, :-1]
            )
            group_start = torch.cummax(
                torch.where(change, index_i32, invalid_i32),
                dim=1,
            ).values
            next_change = torch.flip(
                torch.cummin(
                    torch.flip(
                        torch.where(change, index_i32, n_i32),
                        dims=[1],
                    ),
                    dim=1,
                ).values,
                dims=[1],
            )
            group_end_excl = torch.cat(
                [
                    next_change[:, 1:],
                    n_i32.expand(n_classes, 1),
                ],
                dim=1,
            )

            # 平均秩恒为整数或半整数，以 2×秩 的整数和
            # 累计，除以 2 后与逐类实现的 fp64 秩和一致。
            rank_half = (
                (group_start << 1)
                + (group_end_excl - group_start)
                + 1
            )
            roc_rank_half_sums = (
                rank_half * sorted_labels
            ).sum(dim=1)

            # PR-AUC：降序排序下每个相异分数阈值组的
            # 结束位置（1 基秩）与累计真正例数。并列分数必须
            # 在整组结束时取累计命中数；若取组首会漏掉同组内
            # 后续正例，PR-AUC 系统性偏低。
            labels_desc = sorted_labels.flip(1)
            hits = torch.cumsum(labels_desc, dim=1)
            scores_desc = sorted_scores.flip(1)
            boundary = torch.ones_like(
                sorted_scores, dtype=torch.bool
            )
            boundary[:, :-1] = (
                scores_desc[:, :-1] != scores_desc[:, 1:]
            )
            boundary_counts = boundary.sum(dim=1)

            # 组边界按行主序压缩成一维：每类的边界连续，
            # split 分段即为逐类顺序。行内顺序与逐类实现
            # 的组枚举顺序相同。
            boundary_flat = (
                boundary.view(-1).nonzero().squeeze(1)
            )
            boundary_class = torch.div(
                boundary_flat,
                n_tokens,
                rounding_mode="floor",
            )
            boundary_hits = hits.view(-1).gather(
                0, boundary_flat
            )
            boundary_ranks_desc = (
                boundary_flat
                - boundary_class * n_tokens
            ).double() + 1.0

            recall = (
                boundary_hits.double()
                / n_pos.gather(
                    0, boundary_class
                ).double()
            )
            precision = (
                boundary_hits.double()
                / boundary_ranks_desc
            )

            # AP = Σ (R_i − R_{i−1}) × Precision_i；段首
            # 的 R_{i−1} 取 0，增益即该组 recall。
            recall_gain = recall.clone()
            if recall_gain.numel() > 1:
                recall_gain[1:] = recall[1:] - recall[:-1]
                boundary_start = torch.ones(
                    recall_gain.numel(), dtype=torch.bool
                )
                boundary_start[1:] = (
                    boundary_class[1:] != boundary_class[:-1]
                )
                recall_gain[boundary_start] = recall[
                    boundary_start
                ]
            ap_terms = recall_gain * precision
            ap_sums = torch.split(
                ap_terms, boundary_counts.tolist()
            )

            for class_offset in range(n_classes):
                n_pos_class = int(n_pos_list[class_offset])
                if n_pos_class > 0:
                    pr_aucs.append(
                        float(ap_sums[class_offset].sum())
                    )
                    pr_prevalence_sum += (
                        float(n_pos_class) / n_tokens
                    )
                if 0 < n_pos_class < n_tokens:
                    n_neg_class = n_tokens - n_pos_class
                    roc_aucs.append(
                        float(
                            (
                                roc_rank_half_sums[
                                    class_offset
                                ].double()
                                / 2.0
                                - n_pos_class
                                * (n_pos_class + 1)
                                / 2.0
                            )
                            / float(n_pos_class * n_neg_class)
                        )
                    )

    pr_auc_macro = (
        sum(pr_aucs) / len(pr_aucs)
        if pr_aucs
        else None
    )
    pr_auc_macro_chance = (
        pr_prevalence_sum / len(pr_aucs)
        if pr_aucs
        else None
    )
    return {
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_macro_chance": pr_auc_macro_chance,
        # 相对随机水平的 lift 才是不随 prevalence 缩放的一等
        # 指标：PR-AUC 绝对值在 1:1000 不平衡下从 0.001 起步，
        # 跨数据集/不同类别构成不能直接比较；lift 的随机水平
        # 恒为 0，可直接预注册、对比与做稻草人核对。
        "pr_auc_macro_lift": (
            pr_auc_macro - pr_auc_macro_chance
            if pr_auc_macro is not None
            and pr_auc_macro_chance is not None
            else None
        ),
        "roc_auc_macro": (
            sum(roc_aucs) / len(roc_aucs)
            if roc_aucs
            else None
        ),
        # 三类计数分开记录：单类语料等退化情形下 PR 与 ROC 的
        # 可计算集合不同（全正类的 ROC 无定义），单一
        # auc_n_classes 无法同时描述两条 macro 曲线的口径。
        "auc_n_classes": len(eligible),
        "pr_auc_n_classes": len(pr_aucs),
        "roc_auc_n_classes": len(roc_aucs),
        "auc_min_support": min_support,
        "auc_tokens_evaluated": n_tokens,
    }
