from __future__ import annotations



def analyze_training_curve(
    records: List[Dict[str, object]],
    baseline_loss: Optional[float] = None,
) -> Dict[str, object]:
    """
    基于完整训练曲线（metrics.jsonl）而不是最终单个数值做诊断。

    对每个周期验证点，取「自上次验证以来训练损失的平均值」与
    val_loss 配对，按以下规则给出结论：

    - ``overfitting``：验证损失最优点早于最后一次验证、最终
      验证损失较最优值回升超过阈值，且训练损失仍在下降、
      或最终训练损失已显著低于验证损失（泛化 gap 大——
      记忆化 regime 下训练损失常在低位走平而不是继续下降）；
    - ``underfitting``：训练损失整体几乎不降，且最终仍接近随
      机基线（ln(vocab_size)）；
    - ``plateau``：训练与验证损失最近一个验证间隔内都没有
      有效下降（排除过拟合后，两者都不降指向欠拟合、学习率
      不当或数据管道问题；学习率已退火殆尽时则更像正常
      收敛平台，报告会结合记录中的 lr 曲线区分两种情形）；
    - ``not_converged``：最近一次间隔内训练与验证损失仍同步
      明显下降，说明训练步数/学习率还没到平台；
    - ``diverging``：最近一次间隔内训练与验证损失同时明显
      上升（有限发散/训练不稳定），需要降低学习率、收紧
      梯度裁剪并从稳定检查点恢复；
    - ``diverged``：最近一步损失非有限（NaN/Inf）；
    - ``healthy``：验证仍在改善（或训练验证同步缓降）且
      验证未回升；
    - ``insufficient_data``：验证点不足以做结论。

    除状态外还检查「尚未 NaN 的前兆」：近期梯度范数达到此前
    中位数的 ``_CURVE_GRAD_SPIKE_RATIO`` 倍时置
    ``grad_norm_spike_warning``，即使曲线仍被判为 healthy/
    plateau 也会在 warnings 中提示排查梯度爆炸/学习率过大。

    返回的诊断会追加为 ``{"step": "curve_analysis", ...}``
    记录，并打印对应建议（过拟合→正则化/增大数据/早停，
    欠拟合/双平台→查学习率、优化器与数据管道，有限发散→
    降学习率/收紧裁剪/从稳定检查点恢复）。
    """
    train_points: List[Tuple[int, float]] = []
    grad_points: List[Tuple[int, float]] = []
    nonfinite_steps: List[int] = []
    eval_points: List[Dict[str, object]] = []
    pending: List[float] = []
    peak_lr: Optional[float] = None
    last_lr: Optional[float] = None

    for record in records:
        step = record.get("step")

        if not isinstance(step, int):
            continue

        if "grad_norm" in record or "train_loss" in record:
            train_loss = record.get("train_loss")
            grad_norm = record.get("grad_norm")
            # 非有限事件既可能落在损失上（写盘为 null，在线记录
            # 保留 NaN/Inf），也可能只落在梯度上。判据必须覆盖
            # 原值非有限的情况，否则 NaN 会流进 _sparkline 的
            # int() 转换。
            # JSON 允许任意大整数（如 10**400）；float() 会溢出，
            # 必须与 NaN/Inf 一样按非有限事件处理，而不是让整份
            # 曲线诊断崩溃。
            train_value = _finite_float(train_loss)
            train_finite = train_value is not None
            grad_value = _finite_float(grad_norm)
            grad_nonfinite = (
                "grad_norm" in record and grad_value is None
            )
            nonfinite = not train_finite or grad_nonfinite
            if nonfinite:
                if (
                    not nonfinite_steps
                    or nonfinite_steps[-1] != step
                ):
                    nonfinite_steps.append(step)
            if train_finite:
                train_points.append((step, train_value))
                pending.append(train_value)

            # 有限梯度范数序列：用于在损失尚未变成 NaN 时发现
            # 梯度尖峰（爆炸/学习率过大的前兆）。
            if "grad_norm" in record and not grad_nonfinite:
                grad_points.append((step, grad_value))

            # 学习率曲线（每步记录）：用于区分「双平台」是调度
            # 末期的正常收敛，还是高学习率下的训练停滞。
            record_lr = _finite_float(record.get("lr"))
            if record_lr is not None:
                last_lr = record_lr
                if (
                    peak_lr is None
                    or record_lr > peak_lr
                ):
                    peak_lr = record_lr

        if "val_loss" not in record:
            continue

        # 盘上记录里的 NaN/Inf 会被 _json_safe 写成 null，在线
        # 记录保留原值；超出 float 范围的整数同样按非有限事件
        # 处理，避免 NaN/溢出值流入 _sparkline 的 int() 转换。
        val_loss = _finite_float(record.get("val_loss"))
        if val_loss is None:
            if (
                not nonfinite_steps
                or nonfinite_steps[-1] != step
            ):
                nonfinite_steps.append(step)
            eval_points.append({
                "step": step,
                "val_loss": None,
                "train_loss": None,
            })
            pending = []
            continue

        # 用带缩放的均值而不是直接 sum：两个各自有限的大损失
        # （如 1e308）相加会溢出成 inf，让 eval 点的 train_loss
        # 以及由它派生的 gap/发散判据全部失真。正常量级下
        # _mean 仍走原来的直接求和路径，历史数值不变。
        train_at_eval = (
            _mean(pending)
            if pending
            else (train_points[-1][1] if train_points else None)
        )

        eval_points.append({
            "step": step,
            "val_loss": val_loss,
            "train_loss": train_at_eval,
        })
        pending = []

    finite_evals = [
        point
        for point in eval_points
        if point["val_loss"] is not None
    ]

    # 「最近仍是非有限」才判定发散；早先出现过 NaN 但后续恢复
    # 的运行保留警告但不直接判发散。
    recent_nonfinite = bool(nonfinite_steps) and (
        not train_points
        or nonfinite_steps[-1] >= train_points[-1][0]
    )
    last_eval_nonfinite = bool(
        eval_points and eval_points[-1]["val_loss"] is None
    )

    warnings: List[str] = []

    # 梯度尖峰：与「此前中位数」比较而不是与绝对阈值比较，
    # 不同模型/模式的正常梯度量级差异很大。只把近期窗口纳入
    # 判断，早期 warmup 的高梯度不应让最终报告永久报警。
    grad_spike = False
    grad_spike_ratio: Optional[float] = None
    grad_spike_recent_max: Optional[float] = None
    grad_prior_median: Optional[float] = None
    if len(grad_points) >= _CURVE_GRAD_SPIKE_MIN_HISTORY:
        grad_values = [value for _, value in grad_points]
        recent_count = max(
            _CURVE_GRAD_SPIKE_MIN_RECENT,
            int(
                len(grad_values)
                * _CURVE_GRAD_SPIKE_RECENT_FRACTION
            ),
        )
        recent_count = min(recent_count, len(grad_values) - 1)
        prior_values = sorted(grad_values[:-recent_count])
        grad_spike_recent_max = max(
            grad_values[-recent_count:]
        )
        grad_prior_median = _median(prior_values)
        if grad_prior_median > 0.0:
            ratio = (
                grad_spike_recent_max / grad_prior_median
            )
            if math.isfinite(ratio):
                grad_spike_ratio = ratio
                grad_spike = (
                    ratio >= _CURVE_GRAD_SPIKE_RATIO
                )
            else:
                # 中位数极小、近期最大值极大时比值会溢出成 inf；
                # 尖峰仍然真实存在，但报告字段必须是有限数或
                # None，不能把 inf 写进 metrics/config。
                grad_spike = True

    analysis: Dict[str, object] = {
        "n_train_steps": len(train_points),
        "n_eval_points": len(finite_evals),
        "n_nonfinite_events": len(nonfinite_steps),
        "recent_grad_norm_max": (
            round(grad_spike_recent_max, 6)
            if grad_spike_recent_max is not None
            else None
        ),
        "prior_grad_norm_median": (
            round(grad_prior_median, 6)
            if grad_prior_median is not None
            else None
        ),
        "grad_norm_spike_ratio": (
            round(grad_spike_ratio, 6)
            if grad_spike_ratio is not None
            else None
        ),
        "grad_norm_spike_warning": bool(grad_spike),
        "nonfinite_steps": nonfinite_steps[-10:],
        "peak_lr": peak_lr,
        "last_lr": last_lr,
        "last_lr_over_peak": (
            _safe_ratio(last_lr, peak_lr)
            if peak_lr is not None
            and peak_lr > 0.0
            and last_lr is not None
            else None
        ),
        "warnings": warnings,
        "status": "insufficient_data",
        "recommendation": (
            "验证点不足，继续训练后再看曲线；"
            "至少需要 3 个周期验证点。"
        ),
    }

    if nonfinite_steps:
        warnings.append(
            f"训练/验证损失出现 {len(nonfinite_steps)} 次非有限"
            "（NaN/Inf）记录；按排查路径逐项检查：数据是否含 "
            "NaN/Inf → log(0)/除零/softmax 溢出 → fp16 混合"
            "精度溢出 → 梯度爆炸/学习率过大（见 "
            "diagnose_non_finite 输出的 nan_diagnosis 记录）。"
        )

    if grad_spike:
        ratio_text = (
            f"{grad_spike_ratio:.1f} 倍"
            if grad_spike_ratio is not None
            else "超出可表示的倍数范围"
        )
        warnings.append(
            "近期梯度范数最大值 "
            f"{grad_spike_recent_max:.4g} 是此前中位数 "
            f"{grad_prior_median:.4g} 的 "
            f"{ratio_text}：即使损失尚未变成 "
            "NaN，也可能是梯度爆炸/学习率过大的前兆。"
            "建议降低学习率、收紧 --grad-clip，并检查是否"
            "已出现参数/优化器状态污染。"
        )

    # 警告签名用布尔类别而不是完整文本：文本包含计数/百分比，
    # 每次验证都会变化，在线监控会退化成每点重复打印。
    analysis["warning_flags"] = {
        "nonfinite": bool(nonfinite_steps),
        "gap_widening": False,
        "diverging": False,
        "grad_spike": bool(grad_spike),
    }

    if (recent_nonfinite or last_eval_nonfinite) and (
        eval_points or train_points
    ):
        analysis["status"] = "diverged"
        analysis["recommendation"] = (
            "损失已发散为 NaN/Inf，不要继续加载该检查点训练；"
            "先按 diagnose_non_finite 的四步路径排查，"
            "再考虑降低学习率/开启梯度裁剪后重跑。"
        )
        analysis["curve_sparkline_train"] = _sparkline(
            [value for _, value in train_points]
        )
        return analysis

    if len(finite_evals) < _CURVE_MIN_EVAL_POINTS:
        analysis["curve_sparkline_train"] = _sparkline(
            [value for _, value in train_points]
        )
        analysis["curve_sparkline_val"] = _sparkline(
            [point["val_loss"] for point in finite_evals]
        )
        return analysis

    first = finite_evals[0]
    last = finite_evals[-1]
    previous = finite_evals[-2]

    best_index = min(
        range(len(finite_evals)),
        key=lambda index: finite_evals[index]["val_loss"],
    )
    best = finite_evals[best_index]

    regression = (
        (last["val_loss"] - best["val_loss"])
        / max(float(best["val_loss"]), 1e-12)
    )
    # 两端损失各自有限，但差值/相对值仍可能越过 float 上限；
    # 无法表示时按缺失处理，不能让 inf 进入状态判定与报告。
    if not math.isfinite(regression):
        regression = None

    def _relative_drop(newer, older):
        if older is None or newer is None:
            return None
        older = float(older)
        newer = float(newer)
        if older <= 0.0:
            return None
        try:
            value = (older - newer) / older
        except OverflowError:
            return None
        return value if math.isfinite(value) else None

    train_drop_total = _relative_drop(
        last["train_loss"],
        first["train_loss"],
    )
    train_still_down = (
        best_index < len(finite_evals) - 1
        and last["train_loss"] is not None
        and best["train_loss"] is not None
        and float(last["train_loss"])
        < float(best["train_loss"])
        * (1.0 - _CURVE_TRAIN_STILL_DESCENDING)
    )
    recent_train_drop = _relative_drop(
        last["train_loss"],
        previous["train_loss"],
    )
    recent_val_drop = _relative_drop(
        last["val_loss"],
        previous["val_loss"],
    )
    recent_train_rise = (
        None
        if recent_train_drop is None
        else -recent_train_drop
    )
    recent_val_rise = (
        None
        if recent_val_drop is None
        else -recent_val_drop
    )

    # 有限发散：最近一个验证间隔内训练与验证损失同时明显
    # 上升。它不是过拟合（过拟合要求训练损失仍在下降），也不是
    # 普通平台；继续训练通常只会更糟，必须先处理数值/优化
    # 稳定性。
    finite_diverging = (
        recent_train_drop is not None
        and recent_val_drop is not None
        and recent_train_drop <= -_CURVE_DIVERGENCE_RISE
        and recent_val_drop <= -_CURVE_DIVERGENCE_RISE
    )

    # 泛化 gap：最终训练损失比验证损失低出的相对比例。记忆化
    # regime 下训练损失常在低位走平而不再「仍在下降」，但
    # 大 gap + 验证回升同样是典型过拟合证据。
    def _gap_ratio(point):
        if (
            point["train_loss"] is None
            or float(point["val_loss"]) <= 0.0
        ):
            return None
        try:
            value = (
                float(point["val_loss"])
                - float(point["train_loss"])
            ) / float(point["val_loss"])
        except OverflowError:
            return None
        return value if math.isfinite(value) else None

    generalization_gap_ratio = _gap_ratio(last)
    generalization_gap_ratio_first = _gap_ratio(first)
    generalization_gap_trend = (
        generalization_gap_ratio - generalization_gap_ratio_first
        if generalization_gap_ratio is not None
        and generalization_gap_ratio_first is not None
        else None
    )
    if (
        generalization_gap_trend is not None
        and not math.isfinite(generalization_gap_trend)
    ):
        generalization_gap_trend = None
    train_below_val = (
        generalization_gap_ratio is not None
        and generalization_gap_ratio >= _CURVE_GENERALIZATION_GAP
    )

    analysis.update({
        "first_train_loss": first["train_loss"],
        "last_train_loss": last["train_loss"],
        "first_val_loss": first["val_loss"],
        "last_val_loss": last["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_step": best["step"],
        "val_regression_from_best": (
            round(regression, 6)
            if regression is not None
            else None
        ),
        "train_val_gap_first": _finite_gap(
            first["train_loss"], first["val_loss"]
        ),
        "train_val_gap_last": _finite_gap(
            last["train_loss"], last["val_loss"]
        ),
        "generalization_gap_ratio_last": (
            None
            if generalization_gap_ratio is None
            else round(generalization_gap_ratio, 6)
        ),
        "generalization_gap_ratio_first": (
            None
            if generalization_gap_ratio_first is None
            else round(generalization_gap_ratio_first, 6)
        ),
        "generalization_gap_trend": (
            None
            if generalization_gap_trend is None
            else round(generalization_gap_trend, 6)
        ),
        "curve_sparkline_train": _sparkline(
            [value for _, value in train_points]
        ),
        "curve_sparkline_val": _sparkline(
            [point["val_loss"] for point in finite_evals]
        ),
    })

    # 过拟合前兆：验证损失尚未明显回升（未触发 overfitting），
    # 但泛化 gap 从首个验证点起持续扩大且已不低。此时训练损失
    # 仍在降而验证收益停滞，继续训练大概率只会加深过拟合；
    # 在状态判定后统一追加警告，不改变既有 status 语义。
    gap_widening = (
        generalization_gap_trend is not None
        and generalization_gap_ratio is not None
        and generalization_gap_trend >= _CURVE_GAP_WIDENING
        and generalization_gap_ratio >= _CURVE_GENERALIZATION_GAP
    )
    analysis["gap_widening_warning"] = bool(gap_widening)
    analysis["warning_flags"]["gap_widening"] = bool(
        gap_widening
    )
    analysis["recent_train_rise"] = (
        None
        if recent_train_rise is None
        else round(recent_train_rise, 6)
    )
    analysis["recent_val_rise"] = (
        None
        if recent_val_rise is None
        else round(recent_val_rise, 6)
    )
    analysis["finite_diverging"] = bool(finite_diverging)
    analysis["warning_flags"]["diverging"] = bool(
        finite_diverging
    )
    if gap_widening:
        warnings.append(
            "泛化 gap 相对首个验证点扩大 "
            f"{generalization_gap_trend * 100:.1f} 个百分点"
            "（最终 gap "
            f"{generalization_gap_ratio * 100:.1f}%）："
            "训练损失继续下降但验证收益停滞，是过拟合前兆。"
        )

    if finite_diverging:
        analysis["status"] = "diverging"
        analysis["divergence_signal"] = "both_losses_rising"
        analysis["recommendation"] = (
            "最近一次验证间隔内训练损失上升 "
            f"{recent_train_rise * 100:.1f}%、验证损失上升 "
            f"{recent_val_rise * 100:.1f}%：这不是过拟合"
            "（过拟合时训练损失仍在下降），而是训练不稳定/"
            "有限发散的信号。继续训练通常只会更糟。建议："
            "降低学习率、收紧 --grad-clip、检查最近的梯度"
            "范数尖峰与数据批次，并从发散前的最优检查点恢复；"
            "若发生在 warmup 阶段，先确认 warmup 步数与峰值"
            "学习率是否匹配。"
        )
    elif (
        best_index <= len(finite_evals) - 2
        and regression is not None
        and regression >= _CURVE_OVERFIT_REGRESSION
        and (train_still_down or train_below_val)
    ):
        analysis["status"] = "overfitting"
        if train_still_down:
            analysis["overfit_signal"] = "train_still_descending"
            evidence_text = "训练损失仍在下降"
        else:
            analysis["overfit_signal"] = "generalization_gap"
            evidence_text = (
                "训练损失已不再下降但比验证损失低 "
                f"{generalization_gap_ratio * 100:.1f}%"
                "（泛化 gap 大，记忆化特征）"
            )
        analysis["recommendation"] = (
            f"验证损失在第 {best['step']} 步达到最优 "
            f"{float(best['val_loss']):.4f} 后回升 "
            f"{regression * 100:.1f}%，{evidence_text}："
            "典型过拟合。建议按序尝试：加正则化"
            "（调大 --weight-decay / 适度增大 balance_coef）、"
            "增大训练数据、用 --early-stop-patience 早停"
            "（不要把验证损失回升后的步数计入结论）。"
        )
    elif (
        train_drop_total is not None
        and train_drop_total < _CURVE_NO_LEARN_DROP
        and (
            baseline_loss is None
            or (
                last["train_loss"] is not None
                and float(last["train_loss"])
                > _CURVE_NO_LEARN_BASELINE * baseline_loss
            )
        )
    ):
        analysis["status"] = "underfitting"
        analysis["recommendation"] = (
            f"训练损失整体仅下降 "
            f"{max(0.0, train_drop_total) * 100:.1f}%，"
            "接近随机基线，训练与验证都没有有效下降："
            "欠拟合或学习率/数据管道有问题。建议检查："
            "学习率是否过小或过大、优化器配置、标签与输入是否"
            "对齐（可先跑 --sanity-only 小批量过拟合检查）、"
            "模型容量与训练步数是否足够。"
        )
    elif (
        recent_train_drop is not None
        and recent_val_drop is not None
        and recent_train_drop < _CURVE_STILL_DESCENDING
        and recent_val_drop < _CURVE_STILL_DESCENDING
    ):
        # 两者都不降：学习率已退火殆尽时是正常的末期收敛平台；
        # 学习率仍高时则指向欠拟合（容量不足）、学习率不当或
        # 数据管道问题。
        analysis["status"] = "plateau"
        lr_annealed = (
            peak_lr is not None
            and peak_lr > 0.0
            and last_lr is not None
            and last_lr
            < _CURVE_LR_ANNEALED_FRACTION * peak_lr
        )
        drops_text = (
            f"训练 {max(0.0, recent_train_drop) * 100:.1f}% / "
            f"验证 {max(0.0, recent_val_drop) * 100:.1f}%"
        )
        if lr_annealed:
            analysis["plateau_kind"] = "schedule_end"
            analysis["recommendation"] = (
                f"最近一次验证间隔内训练与验证损失都没有有效下降"
                f"（{drops_text}），且学习率已退火至峰值的 "
                f"{analysis['last_lr_over_peak'] * 100:.0f}%："
                "更像调度末期的正常收敛平台。若结果已够用可就此"
                "结束；仍需改善时应调整学习率调度/模型容量后重跑，"
                "而不是在已退火的学习率下继续加步数。"
            )
        else:
            analysis["plateau_kind"] = "stalled"
            lr_text = (
                f"（当前 lr={last_lr:.3e}，峰值 {peak_lr:.3e}）"
                if last_lr is not None and peak_lr is not None
                else ""
            )
            analysis["recommendation"] = (
                f"最近一次验证间隔内训练与验证损失都没有有效下降"
                f"（{drops_text}），且已排除过拟合：两者都不降"
                f"通常指向欠拟合（模型容量不足）、学习率不当"
                f"{lr_text}或数据管道本身有问题。建议检查："
                "学习率是否过小或过大、模型容量"
                "（--d-model / --n-layers）是否足够，并用 "
                "--sanity-only 的小批量过拟合检查排除数据管道"
                "问题（标签错位、输入输出不对齐）；定位原因前"
                "不要盲目增加步数。"
            )
    elif (
        recent_train_drop is not None
        and recent_val_drop is not None
        and recent_train_drop > _CURVE_STILL_DESCENDING
        and recent_val_drop > _CURVE_STILL_DESCENDING
    ):
        analysis["status"] = "not_converged"
        analysis["recommendation"] = (
            "最近一次验证间隔内训练与验证损失仍在同步明显下降，"
            "尚未收敛。建议增大 --steps（或调大学习率后重跑），"
            "并参考 learning-rate 曲线确认退火是否太慢。"
        )
    else:
        analysis["status"] = "healthy"
        analysis["recommendation"] = (
            "训练与验证损失同步下降、验证未回升或仅轻微波动，"
            "曲线健康；如需进一步改善，可关注仍未收敛时的"
            "更多步数与模型容量。"
        )

    # 过拟合前兆的行动建议：状态仍判 healthy/not_converged（验证
    # 尚未回升），但必须提示现在就设置早停/加强正则化，而不是等
    # 验证损失回升后才行动。
    if gap_widening and analysis["status"] in (
        "healthy",
        "not_converged",
    ):
        analysis["recommendation"] = (
            str(analysis["recommendation"])
            + " 注意泛化 gap 正在扩大（训练损失下降、验证收益"
            "停滞）：建议立即设定或收紧早停"
            "（--early-stop-patience），并加强正则化/增大数据，"
            "避免验证损失回升后才被动止损。"
        )

    analysis["recent_train_drop"] = (
        None
        if recent_train_drop is None
        else round(recent_train_drop, 6)
    )
    analysis["recent_val_drop"] = (
        None
        if recent_val_drop is None
        else round(recent_val_drop, 6)
    )
    analysis["train_still_down"] = train_still_down

    return analysis



def print_curve_analysis(analysis: Dict[str, object]) -> None:
    """打印训练曲线诊断摘要、走势条与建议。"""
    print(
        "[curve] 状态: "
        f"{analysis.get('status')} | "
        f"验证点 {analysis.get('n_eval_points')} 个 | "
        f"非有限事件 {analysis.get('n_nonfinite_events')} 次"
    )

    if analysis.get("best_val_loss") is not None:
        # 最后一个验证点之前可能一个有限训练损失都没有（早期
        # NaN/Inf 后恢复的运行），此时 last_train_loss 为 None，
        # 与其它字段一致按缺失值打印。
        last_train_loss = analysis.get("last_train_loss")
        last_train_text = (
            _format_finite(last_train_loss)
            if last_train_loss is not None
            else "-"
        )
        regression = analysis.get("val_regression_from_best")
        if regression is None:
            regression = 0.0
        print(
            "[curve] 最优验证损失 "
            f"{_format_finite(analysis.get('best_val_loss'))} @ step "
            f"{analysis.get('best_val_step')}，"
            "最终验证损失 "
            f"{_format_finite(analysis.get('last_val_loss'))}"
            "（相对最优回升 "
            f"{_format_finite(regression, '.1f', 100.0)}%），"
            "最终训练损失 "
            f"{last_train_text}"
        )

    if analysis.get("curve_sparkline_train"):
        print(
            "[curve] train " + str(analysis["curve_sparkline_train"])
        )

    if analysis.get("curve_sparkline_val"):
        print(
            "[curve] val   " + str(analysis["curve_sparkline_val"])
        )

    for warning in analysis.get("warnings", []):
        print(f"[curve] 警告: {warning}")

    print(f"[curve] 建议: {analysis.get('recommendation')}")



def monitor_training_curve(
    records: List[Dict[str, object]],
    baseline_loss: Optional[float],
    previous_status: Optional[str],
    previous_warning_flags: Optional[Dict[str, bool]] = None,
) -> Tuple[Dict[str, object], Optional[str]]:
    """
    训练期间的在线曲线监控：每个周期验证点复用
    analyze_training_curve 的判定，在状态迁移或警告前兆出现时
    报告。

    返回 ``(analysis, new_status)``；状态与上一次相同且警告
    签名没有新增时 ``new_status`` 为 None，表示不需要重复记录/
    打印。只比较状态会漏掉「状态仍为 healthy、但泛化 gap 已
    快速扩大」这类前兆——那正是最值得在训练中途介入的时刻，
    因此 warning_flags 的新增 True 也触发一次报告
    （``analysis["monitor_trigger"]`` 标为 warning_change）。
    验证点不足以判断（``insufficient_data``）时不触发。
    """
    analysis = analyze_training_curve(
        records,
        baseline_loss=baseline_loss,
    )
    status = str(analysis.get("status"))

    if status == "insufficient_data":
        return analysis, None

    warning_flags = analysis.get("warning_flags")
    if not isinstance(warning_flags, dict):
        warning_flags = {}
    analysis["warning_flags"] = {
        str(key): bool(value)
        for key, value in warning_flags.items()
    }

    if status != previous_status:
        analysis["monitor_trigger"] = "status_change"
        return analysis, status

    # 调用方未提供上一次签名时（旧记录/旧调用）不推断新增
    # 警告，保持「仅状态迁移」的兼容语义。
    if previous_warning_flags is None:
        return analysis, None

    previous_flags = previous_warning_flags
    appeared = [
        key
        for key, value in analysis["warning_flags"].items()
        if value and not bool(previous_flags.get(key, False))
    ]
    if appeared:
        analysis["monitor_trigger"] = "warning_change"
        analysis["new_warning_flags"] = appeared
        return analysis, status

    return analysis, None



def selection_improved(
    value: float,
    best: Optional[float],
    higher_is_better: bool,
    min_delta: float,
) -> bool:
    """
    判断一次验证观测是否刷新了选型指标的最优值。

    与早停共用同一口径：``min_delta`` 是必须超过的改善幅度，
    方向由指标本身决定（loss/ppl 越低越好，F1/accuracy 越高
    越好）。非有限值一律不算改善。
    """
    value = _finite_float(value)
    if value is None:
        return False
    best_value = _finite_float(best) if best is not None else None
    # 损坏的 CLI/旧记录可能把 min_delta 写成 None/字符串/NaN；
    # 不可用时按 0 处理（不额外要求改善幅度），而不是让 resume
    # 在减法处崩溃。
    delta = _finite_float(min_delta)
    if delta is None:
        delta = 0.0
    if best_value is None:
        return True
    if higher_is_better:
        return value > best_value + delta
    return value < best_value - delta



def early_stop_state_from_records(
    records: List[Dict[str, object]],
    min_delta: float,
    metric: str = "val_loss",
) -> Tuple[Optional[float], Optional[int], int]:
    """
    从已有 metrics.jsonl 记录恢复早停状态（最优选型指标、最优
    步数、连续未改善次数），保证 resume 后早停判定与原运行一致。

    ``metric`` 必须与本次运行的 ``--selection-metric`` 一致；
    方向由 ``metric_higher_is_better`` 决定（分类指标越高越好，
    loss/ppl 越低越好）。旧记录缺少该指标时跳过，不把「没算」
    当成「变差」。
    """
    best_value: Optional[float] = None
    best_step: Optional[int] = None
    evals_without_improvement = 0
    higher_is_better = metric_higher_is_better(metric)

    for record in records:
        step = record.get("step")
        if not isinstance(step, int):
            continue
        if metric not in record:
            continue

        # 非有限值不算「改善」也不触发「未改善」计数，
        # 与在线训练循环的判定保持一致；否则 NaN 会把 best
        # 永久污染成 NaN，resume 后的选型与早停全部失准。
        # 记录来自 JSON，超范围整数同样按不可用值跳过，
        # 不能让它以 OverflowError 中断 resume。
        value = _finite_float(record.get(metric))
        if value is None:
            continue

        if selection_improved(
            value,
            best_value,
            higher_is_better,
            min_delta,
        ):
            best_value = value
            best_step = step
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1

    return best_value, best_step, evals_without_improvement



# ============================================================
# Bad-case analysis
# ============================================================

# top1 概率与目标 token 概率之差小于该值时视为「样本模糊」：
# 正确答案与模型首选几乎并列，不是明确的模型错误。
_BAD_CASE_AMBIGUOUS_MARGIN = 0.1


# 模型以该置信度以上的概率给出错误答案时，优先怀疑模型盲区
# 或标注噪声。
_BAD_CASE_CONFIDENT = 0.9


# 目标字节在评估 split 中的占比低于该值且模型高置信度答错时，
# 提示标注本身可能有误（需人工复核）。
_BAD_CASE_RARE_TARGET_SHARE = 0.005


# 同一 (target, predicted) 混淆对在错误列表中出现该次数以上时，
# 视为系统性混淆（模型盲区），而不是孤立个例。
_BAD_CASE_SYSTEMATIC_COUNT = 3


# 置信度校准：ECE 的分箱数与「高置信度错误」的阈值。ECE 衡量
# 模型自报概率与真实正确率之间的平均差距；只看 accuracy 会把
# 「自信地答错」与「不确定地答错」混为一谈。
_BAD_CASE_CALIBRATION_BINS = 10

_BAD_CASE_OVERCONFIDENT_THRESHOLD = 0.9


# 标签噪声核验：在训练集里用「目标字节前 K 个字节」作为上下文
# 查找同一上下文的后续字节。只有训练集中同一上下文反复出现、
# 且评估目标不是该上下文的主流后续字节时，才有证据把错误标为
# 疑似标注错误；否则「稀有目标 + 高置信度」更可能是模型盲区
# 或评估抽样波动，不能仅凭此归因于标注。
_BAD_CASE_LABEL_CONTEXT = 8

_BAD_CASE_LABEL_MIN_SUPPORT = 2

_BAD_CASE_LABEL_TARGET_MAX_SHARE = 0.1

# 「训练上下文的主流后续字节」必须有实质优势，否则同一上下文
# 本来就多峰（如空格后的多个合法词首），把「非主流」当标注
# 错误会把正常歧义样本误报为标签噪声。只有主流占比达到该
# 阈值时，评估标签与之矛盾才算可核验的标注可疑。
_BAD_CASE_LABEL_MAJORITY_MIN_SHARE = 0.5


# top-N 案例多样性：最差 token 可能集中在同一窗口的不同位置，
# 此时它们不是 N 个独立证据。至少要有若干案例、且单一窗口
# 占比达到阈值时才提示，避免短报告误报。
_BAD_CASE_WINDOW_CONCENTRATION = 0.6

_BAD_CASE_WINDOW_MIN_CASES = 3


# 全量错误归因与 top-N 案例归因共用同一组类别（顺序即编码）。
_BAD_CASE_CATEGORIES: Tuple[str, ...] = (
    "ambiguous",
    "systematic_confusion",
    "label_noise_suspect",
    "uncertain",
)


# 逐 token 错误率的上下文位置分层：自回归模型在序列开头的
# 有效上下文最短，错误率若显著高于后段，改进方向应是位置/
# 前缀建模而不是继续堆参数。区间为 [lower, upper)。
_BAD_CASE_POSITION_BUCKETS: Tuple[
    Tuple[str, int, Optional[int]], ...
] = (
    ("position_0", 0, 1),
    ("position_1_3", 1, 4),
    ("position_4_7", 4, 8),
    ("position_8_15", 8, 16),
    ("position_16_31", 16, 32),
    ("position_32_plus", 32, None),
)


# 按目标字节在训练集中的频率分层：稀有目标的错误占比高，
# 说明问题在数据不平衡/覆盖不足；高频目标错误率高则更像
# 模型盲区。区间为 [lower, upper)，share 是训练集中该字节
# 的 token 占比。
_BAD_CASE_FREQUENCY_BUCKETS: Tuple[
    Tuple[str, float, Optional[float]], ...
] = (
    ("train_share_lt_0.1pct", 0.0, 0.001),
    ("train_share_0.1pct_to_1pct", 0.001, 0.01),
    ("train_share_1pct_to_5pct", 0.01, 0.05),
    ("train_share_ge_5pct", 0.05, None),
)



def _stratified_error_rows(
    values: torch.Tensor,
    wrong_mask: torch.Tensor,
    buckets: Sequence[Tuple[str, object, Optional[object]]],
    total_wrong: int,
) -> List[Dict[str, object]]:
    """
    按一维分组变量统计 token 数、错误数与错误率。

    ``values`` 与 ``wrong_mask`` 逐 token 对齐；只返回实际出现
    过的分组（空分组不写入报告），避免短序列下大量零计数桶
    淹没信息。``share_of_errors`` 是该组错误占全部错误的比例，
    用于回答「错误主要集中在哪一类 token」。
    """
    rows: List[Dict[str, object]] = []
    for name, lower, upper in buckets:
        mask = values >= lower
        if upper is not None:
            mask = mask & (values < upper)
        tokens = int(mask.sum().item())
        if tokens <= 0:
            continue
        wrong = int((mask & wrong_mask).sum().item())
        rows.append({
            "name": name,
            "tokens": tokens,
            "wrong": wrong,
            "error_rate": round(wrong / tokens, 6),
            "share_of_errors": (
                round(wrong / total_wrong, 6)
                if total_wrong > 0
                else None
            ),
        })
    return rows



def _render_byte(value: int) -> str:
    """把单个字节渲染为可打印文本（不可打印字节用 \\xNN）。"""
    value = int(value)
    if 32 <= value < 127:
        return chr(value)
    return f"\\x{value:02x}"



def _render_bytes(values) -> str:
    """把字节序列渲染为可打印文本。"""
    return "".join(_render_byte(value) for value in values)



def _calibration_report(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int = _BAD_CASE_CALIBRATION_BINS,
) -> Dict[str, object]:
    """
    计算 top1 置信度的可靠性（ECE）与过度自信程度。

    ``confidence`` 是每 token 的 top1 概率，``correct`` 是
    top1 是否命中目标。ECE = Σ (n_bin / N) · |acc_bin − conf_bin|，
    按等宽概率分箱；它直接回答「模型说 90% 时是否真的对 90%」，
    这是 accuracy/loss 都看不出来的性质。
    """
    confidence = confidence.float().reshape(-1).clamp(0.0, 1.0)
    correct = correct.float().reshape(-1)
    total = int(confidence.numel())

    if total == 0:
        return {
            "ece": None,
            "mean_confidence": None,
            "accuracy": None,
            "overconfidence": None,
            "bins": [],
            "note": "没有可校准的 token。",
        }

    ece = 0.0
    bins = []
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        if index == num_bins - 1:
            mask = (confidence >= lower) & (
                confidence <= upper
            )
        else:
            mask = (confidence >= lower) & (
                confidence < upper
            )
        count = int(mask.sum().item())
        if count == 0:
            bins.append({
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "gap": None,
            })
            continue
        mean_confidence = float(
            confidence[mask].mean().item()
        )
        accuracy = float(correct[mask].mean().item())
        gap = accuracy - mean_confidence
        ece += (count / total) * abs(gap)
        bins.append({
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "count": count,
            "mean_confidence": round(mean_confidence, 6),
            "accuracy": round(accuracy, 6),
            "gap": round(gap, 6),
        })

    mean_confidence = float(confidence.mean().item())
    accuracy = float(correct.mean().item())
    return {
        "ece": round(ece, 6),
        "mean_confidence": round(mean_confidence, 6),
        "accuracy": round(accuracy, 6),
        # 正值 = 过度自信（自报概率高于实际正确率）。
        "overconfidence": round(
            mean_confidence - accuracy,
            6,
        ),
        "bins": bins,
        "note": (
            "ECE 为 0 表示置信度与正确率完全一致；"
            "ECE 越大，模型自报概率越不可用于阈值决策。"
        ),
    }



def _train_next_byte_evidence(
    dataset: "ByteDataset",
    contexts: List[bytes],
) -> Dict[bytes, Dict[str, object]]:
    """
    在训练集中查找每个上下文之后实际出现的字节分布。

    只扫描训练段（绝不读取 val/test），并返回支持数、主流后续
    字节及其占比。用于核验「评估标签与训练监督是否矛盾」：
    相同上下文在训练中反复以另一个字节为目标，才构成标注
    可疑的证据；上下文未在训练中出现时只能记为无法核验。
    """
    evidence: Dict[bytes, Dict[str, object]] = {}
    unique_contexts = sorted(set(contexts))

    if not unique_contexts:
        return evidence

    train_end = dataset.split_bounds["train"][1]
    train_bytes = bytes(dataset._raw_buffer[:train_end])
    # 被清洗的群组区间不参与窗口采样，其字节不构成训练监督：
    # 与 train_excluded_spans 相交的命中必须跳过，否则清洗掉的
    # 重复内容会冒充「训练里见过同一上下文」的核验证据（与
    # effective_train_byte_counts / 数据审计同一口径）。
    excluded_spans = _merge_byte_intervals(
        dataset.train_excluded_spans
    )

    for context in unique_contexts:
        counts: Dict[int, int] = {}
        support = 0
        start = 0
        while True:
            position = train_bytes.find(context, start)
            if position < 0:
                break
            next_position = position + len(context)
            if next_position < len(train_bytes):
                # 命中区间为 [position, next_position]（上下文加
                # 目标字节，闭区间），半开化后与排除区间求交。
                if excluded_spans and any(
                    low <= next_position and high > position
                    for low, high in excluded_spans
                ):
                    start = position + 1
                    continue
                next_byte = train_bytes[next_position]
                counts[next_byte] = (
                    counts.get(next_byte, 0) + 1
                )
                support += 1
            start = position + 1

        if support == 0:
            evidence[context] = {
                "support": 0,
                "majority_byte": None,
                "majority_share": None,
                "counts": {},
            }
            continue

        majority_byte, majority_count = max(
            counts.items(),
            key=lambda item: item[1],
        )
        evidence[context] = {
            "support": support,
            "majority_byte": int(majority_byte),
            "majority_share": round(
                majority_count / support,
                6,
            ),
            "counts": {
                int(byte): int(count)
                for byte, count in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
        }

    return evidence



def _label_noise_context_verdict(
    evidence: Optional[Dict[str, object]],
    target: int,
) -> Tuple[bool, Optional[float]]:
    """
    由训练上下文证据判断评估标签是否「可核验地可疑」。

    返回 ``(verified, target_share)``。只要求「多数类不等于
    目标」不够：多数类占比可能只有 11%，其余是大量合法分支，
    这更像上下文歧义而不是标注错误。只有支持数足够、目标占比
    低、且主流后续字节有实质优势时，才把它当作标注噪声证据。
    """
    if not isinstance(evidence, dict):
        return False, None
    support = int(evidence.get("support") or 0)
    counts = evidence.get("counts")
    if support <= 0 or not isinstance(counts, dict):
        return False, None
    target_share = float(counts.get(target, 0)) / support
    majority_share = evidence.get("majority_share")
    if majority_share is None:
        return False, target_share
    majority_byte = evidence.get("majority_byte")
    verified = bool(
        support >= _BAD_CASE_LABEL_MIN_SUPPORT
        and target_share <= _BAD_CASE_LABEL_TARGET_MAX_SHARE
        and float(majority_share)
        >= _BAD_CASE_LABEL_MAJORITY_MIN_SHARE
        and majority_byte is not None
        and int(majority_byte) != int(target)
    )
    return verified, target_share



def _empty_bad_case_report(
    split: str,
    note: str = "评估窗口为空，未生成坏案例。",
) -> Dict[str, object]:
    """
    空评估集的坏案例报告骨架。

    即使没有可评估 token，报告也必须保留与正常路径相同的
    键集合：调用方会无条件读取 wrong_tokens / 类别计数等
    字段，缺键会让「评估集为空」从可解释的空结果变成
    KeyError 崩溃。
    """
    return {
        "split": split,
        "tokens_evaluated": 0,
        "token_accuracy": None,
        "wrong_tokens": 0,
        "top_cases": [],
        "top_cases_unique_windows": 0,
        "top_cases_window_concentration": None,
        "top_cases_concentration_warning": False,
        "top_confusions": [],
        "error_rate_by_target": [],
        "error_rate_by_position": [],
        "error_rate_by_train_frequency": [],
        "error_frequency_basis": None,
        "category_counts": {},
        "error_category_counts": {
            name: 0 for name in _BAD_CASE_CATEGORIES
        },
        "error_category_shares": {
            name: None for name in _BAD_CASE_CATEGORIES
        },
        "calibration": {
            "ece": None,
            "mean_confidence": None,
            "accuracy": None,
            "overconfidence": None,
            "bins": [],
            "note": "没有可校准的 token。",
        },
        "label_noise_verification": {
            "context_length": _BAD_CASE_LABEL_CONTEXT,
            "min_train_support": _BAD_CASE_LABEL_MIN_SUPPORT,
            "max_target_share_in_context": (
                _BAD_CASE_LABEL_TARGET_MAX_SHARE
            ),
            "min_majority_share_in_context": (
                _BAD_CASE_LABEL_MAJORITY_MIN_SHARE
            ),
            "candidates": 0,
            "verified": 0,
            "unverified": 0,
            "unverifiable_no_train_context": 0,
            "note": note,
        },
        "summary": note,
    }
