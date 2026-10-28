from __future__ import annotations



def _ngram_extended_metrics(
    model: NgramBaseline,
    windows: List[torch.Tensor],
    alpha: float,
    order: int,
    split: str,
    include_auc: bool = True,
    bootstrap_replicates: int = 0,
    bootstrap_seed: Optional[int] = None,
    bootstrap_block: int = 1,
    window_sink: Optional[List[Dict[str, object]]] = None,
    window_classification: bool = False,
) -> Dict[str, object]:
    """
    用调优好的 (order, alpha) 在固定窗口上计算与神经模型
    evaluate() 完全同口径的指标：loss / ppl / bits_per_byte、
    token accuracy（micro）、macro / weighted F1、多数类
    参考，以及 one-vs-rest 的 macro PR-AUC / ROC-AUC。

    ``include_auc=False`` 时跳过 AUC（网格调参只需要
    accuracy/F1 等分类指标时不收集整张概率矩阵，显著降低
    候选评估的 CPU/内存开销）；最终评估必须保持默认 True。

    ``window_classification=True`` 时每个窗口记录额外附带
    该窗口的 accuracy / micro、macro、weighted F1 与
    precision / recall（与全局指标同一 ``confusion_matrix_metrics``
    口径），供分类调参指标做逐窗口配对 one-SE 选择。默认
    False，避免把逐窗口分类明细写进 eval_windows.json。

    ``bootstrap_replicates > 1`` 时额外给出与神经模型同口径的
    逐窗口 bootstrap 置信区间（loss / ppl / macro NLL /
    token accuracy / macro 与 weighted F1 / macro 与 weighted
    precision / recall）。统计基线是确定性的，换 seed 重复运行
    不是独立重复；它的评估不确定性只能来自评估窗口抽样，因此
    这个区间是 ngram 对照唯一的误差线来源。``window_sink`` 非 None 时
    逐窗口记录损失和、token 数与命中数，供跨运行逐窗口配对
    比较使用（固定评估窗口对所有运行相同）。
    """
    vocab_size = model.vocab_size

    total_nll = 0.0
    evaluated_tokens = 0

    confusion = torch.zeros(
        (vocab_size, vocab_size),
        dtype=torch.float64,
    )

    # 类平衡（macro）NLL 的逐类充分统计量：与神经模型
    # evaluate() 同口径，保证统计基线与神经模型可以直接在
    # val_macro_nll 上对比（类平衡似然不能只给神经模型算）。
    class_loss_sums = torch.zeros(
        vocab_size,
        dtype=torch.float64,
    )
    class_counts = torch.zeros(
        vocab_size,
        dtype=torch.float64,
    )

    bootstrap_enabled = int(bootstrap_replicates) > 1
    window_loss_sums: List[float] = []
    window_tokens: List[float] = []
    window_true_positive: List[torch.Tensor] = []
    window_predicted: List[torch.Tensor] = []
    window_support: List[torch.Tensor] = []
    window_class_loss_sums: List[torch.Tensor] = []

    # 与神经模型评估同口径：按窗口总位置数分配收集缓冲
    # （不超过上限），逐窗口写入。不做 AUC 时不分配缓冲区。
    auc_capacity = 0
    auc_probs = None
    auc_targets = None
    auc_tokens = 0
    auc_total_positions = 0
    auc_sample_positions: Optional[torch.Tensor] = None
    auc_processed_positions = 0
    if include_auc:
        auc_total_positions = sum(
            int(window.size(0))
            * (int(window.size(1)) - 1)
            for window in windows
        )
        auc_capacity = min(
            _AUC_MAX_TOKENS,
            auc_total_positions,
        )
        auc_sample_positions = _auc_uniform_sample_positions(
            auc_total_positions,
            auc_capacity,
        )
        auc_probs = torch.empty(
            auc_capacity,
            vocab_size,
            dtype=torch.float32,
        )
        auc_targets = torch.empty(
            auc_capacity,
            dtype=torch.int64,
        )

    for window in windows:
        distributions = model.position_distributions(
            window,
            alpha,
            order,
        )

        targets = window[:, 1:].reshape(-1).to(
            torch.int64
        )
        probabilities = distributions

        target_probs = probabilities[
            torch.arange(targets.numel()),
            targets,
        ]

        token_nll = -target_probs.clamp_min(1e-300).log()

        total_nll += float(token_nll.sum())
        evaluated_tokens += targets.numel()

        class_loss_sums.index_add_(
            0,
            targets,
            token_nll.to(torch.float64),
        )
        class_counts.index_add_(
            0,
            targets,
            torch.ones_like(
                token_nll,
                dtype=torch.float64,
            ),
        )

        predicted = probabilities.argmax(dim=1)
        correct = targets == predicted

        # 逐窗口（batch 内每一行）统计，而不是把整个 batch
        # 当成一个窗口：神经模型的 evaluate() 以单个固定评估
        # 窗口为抽样单位，ngram 若按 batch 聚合，bootstrap 的
        # 重采样单位与窗口数都会与神经运行不一致，逐窗口配对
        # 比较也会因窗口数不同而直接失效。
        seq_len = int(window.size(1)) - 1
        row_losses = token_nll.view(int(window.size(0)), seq_len)
        row_correct = correct.view(int(window.size(0)), seq_len)
        row_targets = targets.view(int(window.size(0)), seq_len)
        row_predicted = predicted.view(int(window.size(0)), seq_len)

        for row in range(int(window.size(0))):
            row_loss_sum = float(row_losses[row].sum())
            row_target = row_targets[row]
            row_pred = row_predicted[row]
            row_hits = int(row_correct[row].sum())
            if window_sink is not None:
                window_record: Dict[str, object] = {
                    "loss_sum": row_loss_sum,
                    "tokens": seq_len,
                    "correct": row_hits,
                }
                if window_classification:
                    row_confusion = torch.bincount(
                        row_target * vocab_size + row_pred,
                        minlength=vocab_size ** 2,
                    ).view(vocab_size, vocab_size)
                    window_record["classification"] = (
                        confusion_matrix_metrics(
                            row_confusion.double()
                        )
                    )
                window_sink.append(window_record)

            if bootstrap_enabled:
                window_loss_sums.append(row_loss_sum)
                window_tokens.append(float(seq_len))
                window_true_positive.append(
                    torch.bincount(
                        row_target[row_correct[row]],
                        minlength=vocab_size,
                    )
                )
                window_predicted.append(
                    torch.bincount(
                        row_pred,
                        minlength=vocab_size,
                    )
                )
                row_support = torch.bincount(
                    row_target,
                    minlength=vocab_size,
                )
                window_support.append(row_support)
                window_class_loss_sums.append(
                    torch.zeros(
                        vocab_size,
                        dtype=torch.float64,
                    ).index_add_(
                        0,
                        row_target,
                        row_losses[row].to(torch.float64),
                    )
                )

        confusion += torch.bincount(
            targets * vocab_size + predicted,
            minlength=vocab_size ** 2,
        ).view(vocab_size, vocab_size).double()

        if (
            auc_probs is not None
            and auc_targets is not None
        ):
            if auc_sample_positions is None:
                selected_probabilities = probabilities
                selected_targets = targets
            else:
                start_position = auc_processed_positions
                end_position = (
                    start_position + targets.numel()
                )
                lower = int(
                    torch.searchsorted(
                        auc_sample_positions,
                        start_position,
                    )
                )
                upper = int(
                    torch.searchsorted(
                        auc_sample_positions,
                        end_position,
                    )
                )
                local_positions = (
                    auc_sample_positions[lower:upper]
                    - start_position
                )
                selected_probabilities = probabilities[
                    local_positions
                ]
                selected_targets = targets[local_positions]
            num_new = int(selected_targets.numel())
            auc_probs[
                auc_tokens:auc_tokens + num_new
            ].copy_(selected_probabilities)
            auc_targets[
                auc_tokens:auc_tokens + num_new
            ].copy_(selected_targets)
            auc_tokens += num_new
            auc_processed_positions += int(targets.numel())

    loss = total_nll / evaluated_tokens

    metrics: Dict[str, object] = {
        f"{split}_loss": loss,
        f"{split}_ppl": math.exp(min(loss, 80.0)),
        f"{split}_bits_per_byte": loss / math.log(2.0),
    }

    class_metrics = confusion_matrix_metrics(confusion)

    for key, value in class_metrics.items():
        metrics[f"{split}_{key}"] = value

    # 类平衡似然（macro NLL）：与神经模型同口径。
    balanced_metrics = class_balanced_loss_metrics(
        class_loss_sums,
        class_counts,
        vocab_size,
    )
    for key, value in balanced_metrics.items():
        metrics[f"{split}_{key}"] = value

    if include_auc:
        metrics[f"{split}_auc_tokens_available"] = (
            auc_total_positions
        )
        metrics[f"{split}_auc_coverage_fraction"] = (
            auc_tokens / auc_total_positions
            if auc_total_positions > 0
            else 0.0
        )
        metrics[f"{split}_auc_sampling"] = (
            "uniform_stride"
            if auc_sample_positions is not None
            else "full"
        )

    if (
        include_auc
        and auc_tokens > 0
        and auc_probs is not None
        and auc_targets is not None
    ):
        auc_metrics = macro_auc_metrics(
            auc_probs[:auc_tokens],
            auc_targets[:auc_tokens],
            vocab_size,
        )
        for key, value in auc_metrics.items():
            metrics[f"{split}_{key}"] = value

    if bootstrap_enabled and window_loss_sums:
        if bootstrap_seed is not None:
            seed_value = int(bootstrap_seed)
        else:
            # 调用方未给种子时由窗口内容派生：同一批固定窗口
            # 在重复评估中得到同一区间，不受全局 RNG 影响。
            material = b"|".join(
                window.numpy().tobytes() for window in windows
            )
            seed_value = int.from_bytes(
                hashlib.blake2b(
                    material,
                    digest_size=8,
                ).digest(),
                "big",
            )
        bootstrap_metrics = _bootstrap_eval_intervals(
            torch.tensor(window_loss_sums, dtype=torch.float64),
            torch.tensor(window_tokens, dtype=torch.float64),
            torch.stack(window_true_positive),
            torch.stack(window_predicted),
            torch.stack(window_support),
            int(bootstrap_replicates),
            seed_value,
            window_class_loss_sums=torch.stack(
                window_class_loss_sums
            ),
            block_size=int(bootstrap_block),
        )
        for key, value in bootstrap_metrics.items():
            metrics[f"{split}_{key}"] = value

    return metrics



def resolve_ngram_tuning_metric(
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """
    决定 n-gram 基线选择 (order, alpha) 的目标指标。

    与简单神经基线同一纪律：预注册主指标优先于默认的
    val_loss；显式 --ngram-tuning-metric 优先于两者。主指标
    是路由诊断等 n-gram 无法计算的指标时回退到 val_loss 并
    说明原因，而不是静默换口径。
    """
    explicit = getattr(args, "ngram_tuning_metric", None)
    if explicit is not None:
        return str(explicit), "explicit"

    primary = getattr(args, "primary_metric", None)
    if primary is not None:
        if primary in _NGRAM_TUNING_METRIC_KEYS:
            return str(primary), "primary_metric"
        print(
            "[ngram] 预注册主指标 "
            f"{primary} 不是 n-gram 基线可计算的指标，"
            "回退到 val_loss 选择 (order, alpha)；"
            "如需按分类指标调参请显式传 "
            "--ngram-tuning-metric"
        )

    return "val_loss", "default_loss"



def _ngram_tuning_diagnostics(
    best: Dict[str, object],
    runner_up: Optional[Dict[str, object]],
    tuning_metric: str,
    alpha_grid: List[float],
    max_order: int,
    candidate_count: int,
    valid_candidate_count: int,
    selection: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    汇总 n-gram (order, alpha) 网格调参的选择不确定性。

    n-gram 的最终 val 指标是在同一批 val 窗口上从整个网格里
    选出的 winner，和神经模型的 checkpoint 选型一样存在乐观
    偏差；只看 winner 的分数无法判断它是否只是网格内的微小
    噪声优势。这里报告：

    - ``metric_gap`` / ``metric_gap_relative``：winner 与
      runner-up 在调参指标上的差距；差距为 0 表示并列，当前
      选择只是「更简单优先」的确定性 tie-break；
    - ``boundary_hits``：winner 落在 alpha / order 网格边界时
      提示网格可能没有覆盖更优区域；
    - ``selection_bias_note``：val 网格调参后的同一 val 指标
      带乐观偏差，test 全程尘封，评估不确定性应看窗口
      bootstrap 而不是重复 seed；
    - ``selection``：可选的 one-SE 选择记录；给出点估计最优、
      最终选择、并列候选与配对标准误，避免把微小噪声优势
      当成唯一最优。
    """
    best_value_float = _finite_float(best.get(tuning_metric))

    runner_value_float: Optional[float] = None
    if runner_up is not None:
        runner_value_float = _finite_float(
            runner_up.get(tuning_metric)
        )

    gap: Optional[float] = None
    gap_relative: Optional[float] = None
    if (
        best_value_float is not None
        and runner_value_float is not None
    ):
        gap = abs(best_value_float - runner_value_float)
        if abs(best_value_float) > 0.0:
            gap_relative = gap / abs(best_value_float)

    best_order = int(best["order"])
    best_alpha = float(best["alpha"])
    alpha_min = min(float(value) for value in alpha_grid)
    alpha_max = max(float(value) for value in alpha_grid)
    boundary_hits: List[str] = []
    if best_order >= int(max_order):
        boundary_hits.append("order=网格上界")
    if best_alpha <= alpha_min:
        boundary_hits.append("alpha=网格下界")
    if best_alpha >= alpha_max:
        boundary_hits.append("alpha=网格上界")

    advisories: List[str] = []
    if gap is not None:
        # 并列判定按指标尺度取相对容差；固定 1e-12 会把
        # 1e-13 量级的真实 gap 误报成并列。
        tolerance = 1e-12 * abs(best_value_float or 0.0)
        if gap <= tolerance and runner_up is not None:
            advisories.append(
                "winner 与 runner-up 在调参指标上并列：当前仅是"
                "按「更小 order / 更小 alpha」的确定性 tie-break，"
                "不能把选中配置的 val 指标当成唯一最优证据"
            )
    if boundary_hits:
        advisories.append(
            "winner 落在调参网格边界（"
            + ", ".join(boundary_hits)
            + "）：网格可能未覆盖更优区域，扩展候选后应重跑"
        )
    if selection and not selection.get(
        "selected_is_point_best", True
    ):
        advisories.append(
            "one-SE 规则未选择点估计最优配置："
            "与点估计 winner 的逐窗口配对差在 1 个标准误内，"
            "按更简单优先选择 order="
            f"{best_order}/alpha={best_alpha:g}，"
            "避免把网格内微小噪声优势当成稳定收益"
        )

    return {
        "tuning_metric": tuning_metric,
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(valid_candidate_count),
        "best": {
            "order": best_order,
            "alpha": best_alpha,
            "value": best_value_float,
        },
        "runner_up": (
            {
                "order": int(runner_up["order"]),
                "alpha": float(runner_up["alpha"]),
                "value": runner_value_float,
            }
            if runner_up is not None
            else None
        ),
        "metric_gap": gap,
        "metric_gap_relative": gap_relative,
        "boundary_hits": boundary_hits,
        "selection_bias_note": (
            "最终 val 指标是在同一批 val 窗口上从整个网格选出的 "
            "winner，和 checkpoint 选型一样带乐观偏差；test 全程"
            "尘封。评估不确定性应看 --eval-bootstrap 的窗口区间，"
            "不能用重复 seed 冒充独立重复"
        ),
        "selection": selection,
        "advisories": advisories,
    }



def _ngram_window_tuning_value(
    loss: float,
    tuning_metric: str,
) -> float:
    """把逐窗口 loss 变换到调参指标尺度（仅损失类指标）。"""
    if tuning_metric == "val_ppl":
        return math.exp(min(loss, 80.0))
    if tuning_metric == "val_bits_per_byte":
        return loss / math.log(2.0)
    return loss



def _select_ngram_candidate_one_se(
    point_best: Dict[str, object],
    candidates: List[Dict[str, object]],
    window_values: Dict[Tuple[int, float], List[float]],
    tuning_metric: str,
) -> Optional[Dict[str, object]]:
    """
    对 n-gram 网格用逐窗口配对 one-SE 规则做最终选择。

    点估计最优只说明「在这批 val 窗口上最小/最大」，网格里
    相邻配置的微小优势可能只是窗口抽样噪声。这里以点估计
    winner 为参照，对每个候选计算逐窗口配对差（候选 - 点估计
    winner）的均值与标准误；正自相关时按 AR(1) 有效窗口数
    放宽标准误。均值差不超过一个标准误的候选视为统计上
    不可分辨，再按「更小 order / 更小 alpha」选择更简单配置。

    损失类指标用逐窗口 loss 变换到调参指标尺度；accuracy /
    micro、macro、weighted F1 与 precision / recall 用
    ``_ngram_extended_metrics`` 记录的逐窗口同口径分类指标，
    并按同一规则配对（越高越好时方向取反）。AUC 需要整张
    概率矩阵且逐窗口类支持稀疏，逐窗口估计与原口径不一致，
    仍按点估计选择。
    """
    loss_like = tuning_metric in (
        "val_loss",
        "val_ppl",
        "val_bits_per_byte",
    )
    higher_is_better = metric_higher_is_better(tuning_metric)
    if (
        not loss_like
        and tuning_metric[len("val_"):]
        not in _NGRAM_PER_WINDOW_CLASSIFICATION_METRICS
    ):
        return None

    def _key(row: Dict[str, object]) -> Tuple[int, float]:
        return int(row["order"]), float(row["alpha"])

    def _values(raw: List[float]) -> Optional[List[float]]:
        values: List[float] = []
        for value in raw:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(number):
                return None
            values.append(
                _ngram_window_tuning_value(
                    number, tuning_metric
                )
                if loss_like
                else number
            )
        return values

    point_key = _key(point_best)
    point_raw = window_values.get(point_key)
    if not point_raw or len(point_raw) < 3:
        return {
            "rule": "paired_one_se",
            "available": False,
            "reason": "insufficient_window_values",
        }
    point_values = _values(point_raw)
    if point_values is None:
        return {
            "rule": "paired_one_se",
            "available": False,
            "reason": "nonfinite_window_values",
        }

    n_windows = len(point_values)
    point_mean = _mean(point_values)
    if point_mean is None:
        return {
            "rule": "paired_one_se",
            "available": False,
            "reason": "nonfinite_point_mean",
        }
    # 容差必须相对于指标本身，而不是固定 1e-12：极小但非零的
    # loss 差异（如 1e-13）否则会被当成数值噪声而误判并列。
    tolerance = 1e-12 * abs(point_mean)

    tied: List[Dict[str, object]] = []
    for candidate in candidates:
        candidate_raw = window_values.get(_key(candidate))
        if (
            candidate_raw is None
            or len(candidate_raw) != n_windows
        ):
            continue
        candidate_values = _values(candidate_raw)
        if candidate_values is None:
            continue
        differences = [
            value - base
            for value, base in zip(
                candidate_values, point_values
            )
        ]
        mean_diff = _mean(differences)
        if mean_diff is None:
            continue
        std = _sample_std(differences)
        if std is None:
            continue
        se = std / math.sqrt(n_windows)
        rho1 = _lag1_autocorrelation(
            torch.tensor(differences, dtype=torch.float64)
        )
        _effective, inflation = _ar1_effective_windows(
            n_windows,
            rho1,
        )
        threshold = se * inflation
        within_one_se = (
            mean_diff >= -(threshold + tolerance)
            if higher_is_better
            else mean_diff <= threshold + tolerance
        )
        if within_one_se:
            value = candidate.get(tuning_metric)
            tied.append({
                "order": int(candidate["order"]),
                "alpha": float(candidate["alpha"]),
                "value": _finite_float(value),
                "paired_mean_diff": mean_diff,
                "paired_se": se,
                "ar1_inflation": inflation,
                "threshold": threshold,
            })

    if not tied:
        return {
            "rule": "paired_one_se",
            "available": False,
            "reason": "no_candidates_with_window_values",
        }

    selected = min(
        tied,
        key=lambda row: (row["order"], row["alpha"]),
    )
    selected_row = next(
        (
            candidate
            for candidate in candidates
            if _key(candidate)
            == (selected["order"], selected["alpha"])
        ),
        point_best,
    )
    selected_is_point_best = (
        _key(selected_row) == point_key
    )
    point_value = point_best.get(tuning_metric)
    return {
        "rule": "paired_one_se",
        "available": True,
        "n_windows": n_windows,
        "point_best": {
            "order": point_key[0],
            "alpha": point_key[1],
            "value": (
                float(point_value)
                if isinstance(point_value, (int, float))
                and not isinstance(point_value, bool)
                and math.isfinite(float(point_value))
                else None
            ),
        },
        "selected": {
            "order": int(selected_row["order"]),
            "alpha": float(selected_row["alpha"]),
        },
        "selected_is_point_best": selected_is_point_best,
        "tied": tied,
        "note": (
            "候选与点估计 winner 的逐窗口配对差不超过一个"
            "标准误时按更简单优先选择；正自相关按 AR(1) "
            "有效窗口数放宽标准误"
        ),
    }
