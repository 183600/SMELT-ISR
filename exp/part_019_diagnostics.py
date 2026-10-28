from __future__ import annotations


_NONFINITE_CHECKLIST = [
    "1) 数据：批次内是否存在 NaN/Inf 或越界 token"
    "（字节管线的整数 token 天然无 NaN，越界/负值可能来自"
    "上游 dtype 或编码错误；扩展到连续值特征后必须查 NaN/Inf）。",
    "2) 前向数值：logits 是否非有限、绝对值是否接近低精度"
    "上限；有限 logits + 非有限损失通常是 log(0)/除零/目标"
    "越界等损失侧问题。",
    "3) 混合精度：fp16 autocast 下 matmul/softmax 溢出会先"
    "出现大绝对值再变 NaN；梯度下溢则表现为小梯度被 flush 成"
    "0、更新丢失（grad_scale 被持续上调、梯度范数异常小、"
    "训练停滞但不发散）。可对比关闭 AMP（--device cpu 或"
    "换 bf16）判断。",
    "4) 梯度与优化：逐参数梯度是否爆炸/非有限、参数与优化器"
    "状态是否已被污染；若是则降低学习率、开启/收紧梯度裁剪"
    "（--grad-clip）并从污染前的检查点恢复。",
]



def _check_batch_integrity(
    x: torch.Tensor,
    y: torch.Tensor,
    vocab_size: int,
) -> Dict[str, object]:
    """检查一个训练批次的值域与非有限元素。"""
    info: Dict[str, object] = {
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "nonfinite_elements": 0,
        "out_of_range_elements": 0,
        "ok": True,
    }

    for name, tensor in (("x", x), ("y", y)):
        if tensor.numel() == 0:
            continue

        if tensor.is_floating_point():
            finite_mask = torch.isfinite(tensor)
            info["nonfinite_elements"] = int(
                info["nonfinite_elements"]
            ) + int((~finite_mask).sum())
            finite_values = tensor[finite_mask]
            if finite_values.numel() > 0:
                info[f"{name}_min"] = float(
                    finite_values.min()
                )
                info[f"{name}_max"] = float(
                    finite_values.max()
                )
            # NaN/Inf 已在上面单独计数；值域检查只看有限元素。
            comparison = finite_values.to(torch.int64)
        else:
            info[f"{name}_min"] = int(tensor.min())
            info[f"{name}_max"] = int(tensor.max())
            # 值域比较前统一提升到 int64。
            comparison = tensor.to(torch.int64)

        info["out_of_range_elements"] = int(
            info["out_of_range_elements"]
        ) + int(
            ((comparison < 0) | (comparison >= vocab_size)).sum()
        )

    info["ok"] = (
        int(info["nonfinite_elements"]) == 0
        and int(info["out_of_range_elements"]) == 0
    )
    return info



def diagnose_non_finite(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    result: Dict[str, torch.Tensor],
    grad_norm: torch.Tensor,
    *,
    step: int,
    lr: float,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[object] = None,
    amp_scale_before_step: Optional[float] = None,
    vocab_size: int = 256,
    recent_grad_norms: Optional[List[Tuple[int, float]]] = None,
) -> Dict[str, object]:
    """
    非有限损失/梯度出现时的系统排查。

    按固定路径收集证据：数据值域与非有限元素 → logits/损失
    数值（log(0)、除零、softmax 溢出）→ 低精度上限 → 逐参数
    梯度、参数与优化器状态污染 → 学习率/梯度缩放状态；并用
    失败步之前的有限梯度范数（``recent_grad_norms``）判断
    「梯度爆炸/学习率过大」是否有证据，还是更像前向/损失侧
    问题。返回的报告会附加到触发该步的 metrics 记录
    （nan_diagnosis 字段）并打印，之后按报告中的 next_steps
    逐项排除。
    """
    findings: List[str] = []
    checks: Dict[str, object] = {}

    report: Dict[str, object] = {
        "step": step,
        "lr": lr,
        "checks": checks,
        "findings": findings,
        "next_steps": list(_NONFINITE_CHECKLIST),
    }

    # ---- 1) 数据 ----
    data_check = _check_batch_integrity(x, y, vocab_size)
    checks["data"] = data_check
    if not data_check["ok"]:
        findings.append(
            "数据批次包含非有限或越界元素"
            f"（nonfinite={data_check['nonfinite_elements']}, "
            f"out_of_range={data_check['out_of_range_elements']}）"
        )

    # ---- 2) 前向数值 ----
    logits = result.get("logits")
    loss = result.get("loss")
    lm_loss = result.get("lm_loss")
    aux_loss = result.get("aux_loss")

    # 训练循环在反向之前就释放了 logits；诊断时补跑一次前向。
    # 只在确认出错后执行一次。
    probe_source = "training result"
    # torch.compile 返回的 OptimizedModule 不是 LanguageModel 的
    # isinstance，必须解包后再判断，否则 --compile 运行时前向
    # 证据（logits 有限性/量级）永远缺失。
    if logits is None and isinstance(
        _unwrap_model(model),
        LanguageModel,
    ):
        was_training = model.training
        try:
            model.eval()
            probe_use_amp = (
                amp_dtype is not None
                and x.is_cuda
            )
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=probe_use_amp,
                ):
                    probe = model(x)
            probe_logits = probe.get("logits")
            if probe_logits is not None:
                logits = probe_logits
                probe_source = "diagnostic forward probe"
        except Exception:
            pass
        finally:
            model.train(was_training)

    def _finite(value) -> Optional[bool]:
        if value is None or not torch.is_tensor(value):
            return None
        return bool(torch.isfinite(value).all())

    logits_absmax = None
    logits_dtype = None
    if logits is not None and torch.is_tensor(logits):
        logits_dtype = str(logits.dtype)
        if logits.numel() > 0:
            logits_absmax = float(
                logits.detach().abs().max()
            )

    # 用同一份 logits 在 fp32 下重算逐 token LM 损失：这是
    # 区分「前向 logits 已坏」与「损失侧 log(0)/除零/低精度
    # 归约」的直接证据，也能量化目标概率下溢为 0 的 token 数。
    recomputed_loss: Dict[str, object] = {
        "available": False,
        "lm_loss": None,
        "finite": None,
        "nonfinite_tokens": None,
        "log_zero_probability_tokens": None,
        "target_range_ok": None,
    }
    if (
        logits is not None
        and torch.is_tensor(logits)
        and y is not None
        and torch.is_tensor(y)
        and logits.size(-1) > 0
    ):
        try:
            flat_logits = logits.detach().reshape(
                -1, int(logits.size(-1))
            ).float()
            flat_targets = y.reshape(-1).to(torch.int64)
            if flat_logits.size(0) == flat_targets.numel():
                target_range_ok = bool(
                    (
                        (flat_targets >= 0)
                        & (flat_targets < logits.size(-1))
                    ).all()
                )
                recomputed_loss["available"] = True
                recomputed_loss["target_range_ok"] = (
                    target_range_ok
                )
                if target_range_ok:
                    with torch.no_grad():
                        token_losses = F.cross_entropy(
                            flat_logits,
                            flat_targets,
                            reduction="none",
                        )
                    finite_tokens = torch.isfinite(
                        token_losses
                    )
                    recomputed_loss["finite"] = bool(
                        finite_tokens.all()
                    )
                    recomputed_loss["nonfinite_tokens"] = int(
                        (~finite_tokens).sum().item()
                    )
                    recomputed_loss["log_zero_probability_tokens"] = int(
                        torch.isinf(token_losses).sum().item()
                    )
                    finite_values = token_losses[
                        finite_tokens
                    ]
                    if finite_values.numel() > 0:
                        recomputed_loss["lm_loss"] = float(
                            finite_values.mean().item()
                        )
        except Exception as exc:
            recomputed_loss["error"] = str(exc)[:200]

    overflow_limit = None
    if amp_dtype == torch.float16:
        overflow_limit = 65504.0
    elif amp_dtype == torch.bfloat16:
        overflow_limit = float(
            torch.finfo(torch.bfloat16).max
        )

    checks["forward"] = {
        "logits_source": probe_source,
        "logits_dtype": logits_dtype,
        "logits_finite": _finite(logits),
        "logits_absmax": logits_absmax,
        "lm_loss_finite": _finite(lm_loss),
        "aux_loss_finite": _finite(aux_loss),
        "total_loss_finite": _finite(loss),
        "amp_dtype": (
            str(amp_dtype) if amp_dtype is not None else None
        ),
        "low_precision_overflow_limit": overflow_limit,
        "recompute_loss_fp32": recomputed_loss,
    }

    if _finite(logits) is False:
        findings.append(
            "logits 已含 NaN/Inf：优先怀疑低精度 matmul/softmax "
            "溢出或模型内部数值爆炸"
        )
    if recomputed_loss["available"]:
        if recomputed_loss["target_range_ok"] is False:
            findings.append(
                "目标 token 越界：损失侧 gather/cross_entropy 会"
                "产生非有限值或未定义行为，先修复数据/标签对齐"
            )
        elif (
            recomputed_loss["finite"] is False
            and _finite(logits) is True
        ):
            findings.append(
                "logits 有限，但用同一 logits 在 fp32 重算仍有 "
                f"{recomputed_loss['nonfinite_tokens']} 个 token "
                "损失非有限：优先查目标概率下溢为 0（log(0)）"
                "与低精度 logits 的动态范围，而不是梯度侧"
            )
        if (
            recomputed_loss["log_zero_probability_tokens"]
            or 0
        ) > 0:
            findings.append(
                "重算发现 "
                f"{recomputed_loss['log_zero_probability_tokens']}"
                " 个 token 的目标概率为 0（对数发散）：这是"
                "log(0)/softmax 下溢的直接证据"
            )
    if _finite(aux_loss) is False:
        findings.append(
            "辅助负载均衡损失非有限：检查路由计数归一化的"
            "除零（空 token/batch）、softmax 溢出与 "
            "balance_coef 数值范围"
        )
    if (
        logits_absmax is not None
        and overflow_limit is not None
        and logits_absmax > overflow_limit / 2.0
    ):
        findings.append(
            f"logits 绝对值 {logits_absmax:.3e} 已接近 "
            f"{amp_dtype} 上限 {overflow_limit:.3e}，"
            "低精度溢出风险高"
        )
    if (
        _finite(logits) is True
        and _finite(loss) is False
    ):
        findings.append(
            "logits 有限但总损失非有限：检查损失侧（目标越界、"
            "log(0)、除零、混合精度下的归约）"
        )

    # ---- 3) 逐参数梯度 ----
    nonfinite_grad_params: List[str] = []
    gradient_absmax: List[Tuple[float, str]] = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if bool(torch.isfinite(gradient).all()):
            gradient_absmax.append(
                (float(gradient.abs().max()), name)
            )
        else:
            nonfinite_grad_params.append(name)

    gradient_absmax.sort(reverse=True)

    grad_norm_value = None
    if torch.is_tensor(grad_norm) and bool(
        torch.isfinite(grad_norm)
    ):
        grad_norm_value = float(grad_norm)

    checks["gradients"] = {
        "grad_norm": grad_norm_value,
        "nonfinite_grad_params": nonfinite_grad_params[:5],
        "n_nonfinite_grad_params": len(nonfinite_grad_params),
        "largest_grad_absmax": [
            [round(value, 6), name]
            for value, name in gradient_absmax[:5]
        ],
    }

    if nonfinite_grad_params:
        findings.append(
            f"{len(nonfinite_grad_params)} 个参数的梯度非有限"
            "（梯度爆炸/损失侧 NaN 反传）"
        )
    elif grad_norm_value is None and torch.is_tensor(grad_norm):
        findings.append(
            "梯度总体范数非有限但逐参数梯度均为有限："
            "可能是裁剪/缩放链路上的数值问题"
        )

    # ---- 4) 参数与优化器状态污染 ----
    nonfinite_params = [
        name
        for name, parameter in model.named_parameters()
        if not bool(
            torch.isfinite(parameter.detach()).all()
        )
    ]
    checks["parameters_nonfinite"] = nonfinite_params[:5]
    if nonfinite_params:
        findings.append(
            f"{len(nonfinite_params)} 个参数已含 NaN/Inf："
            "参数已被污染，应从污染前的检查点恢复"
        )

    nonfinite_state_keys = []
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and not bool(
                torch.isfinite(value).all()
            ):
                nonfinite_state_keys.append(key)
    nonfinite_state_keys = sorted(set(nonfinite_state_keys))
    checks["optimizer_state_nonfinite_keys"] = (
        nonfinite_state_keys[:5]
    )
    if nonfinite_state_keys:
        findings.append(
            "优化器状态（"
            + ", ".join(nonfinite_state_keys)
            + "）已含 NaN/Inf：状态已被污染，需要从检查点恢复"
        )

    # ---- 5) 学习率、梯度缩放与梯度趋势 ----
    checks["lr"] = lr
    amp_state: Dict[str, object] = {
        "grad_scale_before_step": None,
        "grad_scale_at_diagnosis": None,
        "scale_reduced_on_failure": False,
    }
    if scaler is not None:
        try:
            scale_at_diagnosis = float(scaler.get_scale())
        except Exception:
            scale_at_diagnosis = None
        checks["grad_scale"] = scale_at_diagnosis
        amp_state["grad_scale_at_diagnosis"] = (
            scale_at_diagnosis
        )
        if (
            amp_scale_before_step is not None
            and math.isfinite(float(amp_scale_before_step))
            and float(amp_scale_before_step) > 0.0
        ):
            before = float(amp_scale_before_step)
            amp_state["grad_scale_before_step"] = before
            if scale_at_diagnosis is not None:
                amp_state["scale_ratio_to_before"] = round(
                    scale_at_diagnosis / before,
                    6,
                )
                # GradScaler 在 unscale 时发现 inf/nan 梯度会
                # 立即回退 scale（默认 ×0.5）；失败步出现回退
                # 是混合精度反传溢出的直接证据，比让用户
                # 「去查 AMP」更可行动。
                reduced = (
                    scale_at_diagnosis < before * 0.999
                )
                amp_state["scale_reduced_on_failure"] = reduced
                if reduced:
                    findings.append(
                        "AMP GradScaler 在失败步把 grad scale "
                        f"从 {before:.4g} 回退到 "
                        f"{scale_at_diagnosis:.4g}："
                        "反传梯度发生 fp16/bf16 溢出（inf/nan），"
                        "这是混合精度数值问题的直接证据。"
                        "优先检查 logits/损失量级、开启梯度裁剪、"
                        "降低学习率，或改用 bf16/fp32 复现确认。"
                    )
    checks["amp"] = amp_state

    # 用失败步之前的有限梯度范数做对照：非有限通常是「梯度爆炸/
    # 学习率过大」的特征；梯度水平正常则更像前向/损失侧问题，
    # 避免把两类根因混在一起排查。
    gradient_trend: Dict[str, object] = {}
    history_values = [
        float(value)
        for _, value in (recent_grad_norms or [])
        if math.isfinite(float(value))
    ]
    if history_values:
        ordered = sorted(history_values)
        median = float(_median(ordered))
        gradient_trend = {
            "window": len(ordered),
            "recent_median": round(median, 6),
            "recent_max": round(ordered[-1], 6),
            "fail_step_grad_norm": grad_norm_value,
        }
        if grad_norm_value is None:
            findings.append(
                f"失败步梯度范数非有限，而此前 {len(ordered)} 步"
                f"梯度范数均为有限（中位 {median:.4g}）："
                "符合梯度爆炸/学习率过大的特征，优先降低学习率、"
                "收紧 --grad-clip 并从污染前的检查点恢复"
            )
        elif (
            median > 0.0
            and grad_norm_value
            >= median * _NAN_GRAD_EXPLOSION_RATIO
        ):
            findings.append(
                f"失败步梯度范数 {grad_norm_value:.4g} 约为近期"
                f"中位数 {median:.4g} 的 "
                f"{grad_norm_value / median:.1f} 倍：梯度放大明显，"
                "符合梯度爆炸/学习率过大特征"
            )
        elif _finite(loss) is False:
            findings.append(
                "失败步梯度范数与近期水平相当：更像前向/损失侧"
                "（数据、log(0)/除零或低精度溢出）问题，"
                "而非梯度爆炸"
            )
    checks["gradient_trend"] = gradient_trend

    if not findings:
        findings.append(
            "本轮检查未定位到单一根因：按 next_steps 顺序逐项"
            "复现排查（数据→损失→混合精度→梯度/学习率）"
        )

    return report



def print_non_finite_report(report: Dict[str, object]) -> None:
    """打印非有限值的系统排查报告。"""
    print(
        f"[nan] step {report.get('step')} 检测到非有限损失/梯度，"
        f"lr={report.get('lr')}"
    )
    for finding in report.get("findings", []):
        print(f"[nan] 发现: {finding}")
    for line in report.get("next_steps", []):
        print(f"[nan] 排查: {line}")



# ============================================================
# Efficiency measurement
# ============================================================

def _unwrap_model(model: nn.Module) -> nn.Module:
    """取出 torch.compile 包装前的原始模型。"""
    return getattr(model, "_orig_mod", model)



def _count_non_embedding_params(model: LanguageModel) -> int:
    """统计不含 token/position embedding 的参数量。"""
    excluded = {
        id(model.token_embedding.weight),
        id(model.position_embedding.weight),
    }
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in excluded
    )



def _count_forward_flops(
    model: LanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, Optional[int]]:
    """
    统计一次前向的理论计算量。

    - 线性层（mm/addmm，含专家与路由）用
      torch.utils.flop_counter.FlopCounterMode 测量；
    - SDPA 内部的注意力矩阵乘不计入 FlopCounter，测量期间
      包装每个 Block 的 attention_step 按 4·B·T²·D 补上
      （乘加各计 1 次浮点运算）。

    返回的是一次前向的总 FLOPs，用于跨模式比较计算量。
    """
    flops: Dict[str, Optional[int]] = {
        "linear_flops": None,
        "attention_flops": None,
        "total_flops_per_forward": None,
        "flops_note": (
            "linear(mm/addmm) 测量 + attention(4·B·T²·D) 估计"
        ),
    }

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return flops

    attention_flops = [0]

    # 训练路径会直接调用 first_branch_step / local_two_stage 等
    # Block 方法而不经过 __call__，模块级 hook 会漏计；所有路径
    # 最终都要走 attention_step，因此在测量期间按实例包装它。
    wrapped_modules: List[TransformerBlock] = []
    for module in model.modules():
        if not isinstance(module, TransformerBlock):
            continue
        original_attention_step = module.attention_step

        def _wrapped_attention_step(
            inputs: torch.Tensor,
            *args,
            _original=original_attention_step,
            **kwargs,
        ) -> torch.Tensor:
            if inputs.dim() == 3:
                # numel = B·T·D，乘 4·T 即 4·B·T²·D。
                attention_flops[0] += (
                    4 * int(inputs.size(-2)) * int(inputs.numel())
                )
            # scale 等区域参数必须原样透传，否则测量前向会走
            # 默认全局残差口径、与被测模型的实际计算图不一致。
            return _original(inputs, *args, **kwargs)

        module.attention_step = _wrapped_attention_step
        wrapped_modules.append(module)

    try:
        try:
            counter_context = FlopCounterMode(display=False)
        except TypeError:
            counter_context = FlopCounterMode()

        with torch.no_grad():
            with counter_context as counter:
                model(x, y)

        linear_flops = int(counter.get_total_flops())
    except Exception:
        return flops
    finally:
        for module in wrapped_modules:
            del module.attention_step

    flops["linear_flops"] = linear_flops
    flops["attention_flops"] = int(attention_flops[0])
    flops["total_flops_per_forward"] = (
        linear_flops + int(attention_flops[0])
    )
    return flops



def _process_peak_rss_mb() -> Optional[float]:
    """
    进程峰值常驻内存（MB）：Linux 读 /proc/self/status 的
    VmHWM，其他平台退回 resource.getrusage。
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(
                        int(line.split()[1]) / 1024.0,
                        3,
                    )
    except (OSError, ValueError, IndexError):
        pass

    try:
        import resource

        peak_kb = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        if sys.platform == "darwin":
            return round(peak_kb / (1024.0 * 1024.0), 3)
        return round(peak_kb / 1024.0, 3)
    except Exception:
        return None



def _percentile(
    sorted_values: List[float],
    fraction: float,
) -> Optional[float]:
    """线性插值分位数；空列表返回 None。"""
    if not sorted_values:
        return None
    try:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = (len(sorted_values) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        weight = position - lower
        lower_value = float(sorted_values[lower])
        upper_value = float(sorted_values[upper])
    except (IndexError, OverflowError, TypeError, ValueError):
        return None
    return (
        lower_value * (1.0 - weight)
        + upper_value * weight
    )



def _safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """两个有限数的比值；缺失、除零或结果不可表示时返回 None。"""
    if numerator is None or denominator is None:
        return None
    if (
        isinstance(numerator, int)
        and not isinstance(numerator, bool)
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
    ):
        if denominator == 0:
            return None
        try:
            # 两个超大但同量级的整数计数可以直接相除；
            # 先各自转 float 会无谓地丢掉可表示的倍率。
            ratio = numerator / denominator
        except (OverflowError, ValueError):
            return None
    else:
        try:
            left = float(numerator)
            right = float(denominator)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or right == 0.0
        ):
            return None
        try:
            ratio = left / right
        except OverflowError:
            return None
    if not math.isfinite(ratio):
        return None
    return round(ratio, 6)



def _safe_cv(std: float, mean: float) -> Optional[float]:
    """变异系数；均值非正或比值超出 float 范围时返回 None。"""
    if mean <= 0.0 or not math.isfinite(mean):
        return None
    try:
        ratio = std / mean
    except OverflowError:
        return None
    if not math.isfinite(ratio):
        return None
    return round(ratio, 6)



def _timing_summary(
    timings: List[float],
    unit_scale: float = 1e3,
) -> Dict[str, object]:
    """
    计时分布摘要：中位数之外还报告 p95/p99、标准差与变异
    系数。只报单次中位数会把「偶发长尾」隐藏掉——对部署而言
    p99 延迟和均值稳定性往往比中位数更重要，跨运行的成本
    倍率也需要离散度才能判断是否只是计时噪声。
    """
    if not timings:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
            "coefficient_of_variation": None,
        }

    # 记录来自 JSON 时可能是 10**400 这类合法但超出 float 范围
    # 的整数；直接 float() 会让整份计时摘要崩溃。按不可用值
    # 处理，保留 n 并让各统计量为 None。
    numeric = _finite_values(list(timings))
    count = len(timings)
    if numeric is None:
        return {
            "n": count,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
            "coefficient_of_variation": None,
        }
    ordered = sorted(numeric)
    mean = _mean(ordered)
    if mean is None:
        return {
            "n": count,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
            "coefficient_of_variation": None,
        }
    std = _sample_std(ordered)
    if std is None:
        std = 0.0

    def scaled(value: Optional[float]) -> Optional[float]:
        """换算单位并取整；结果超出 float 范围时返回 None。"""
        if value is None:
            return None
        result = value * unit_scale
        if not math.isfinite(result):
            return None
        return round(result, 6)

    return {
        "n": count,
        "mean": scaled(mean),
        "median": scaled(_percentile(ordered, 0.5)),
        "p95": scaled(_percentile(ordered, 0.95)),
        "p99": scaled(_percentile(ordered, 0.99)),
        "min": scaled(ordered[0]),
        "max": scaled(ordered[-1]),
        "std": scaled(std),
        "coefficient_of_variation": _safe_cv(std, mean),
    }



def measure_efficiency(
    model: nn.Module,
    dataset: ByteDataset,
    device: torch.device,
    batch_size: int,
    amp_dtype: Optional[torch.dtype] = None,
    warmup: int = 2,
    iters: int = 5,
) -> Dict[str, object]:
    """
    效率测量：参数量、前向 FLOPs 等模型规模指标。

    计时同时报告中位数与 p95/p99、标准差、变异系数及原始
    样本，避免用单次中位数掩盖长尾或把计时噪声当成方法差异。

    本测量在训练结束时对当前模型测一次，与精度指标一起写入
    config.json 与 metrics.jsonl（``{"step": "efficiency", ...}``）。
    """
    was_training = model.training
    raw_model = _unwrap_model(model)
    cfg = raw_model.cfg

    generator = torch.Generator().manual_seed(0)
    x, y = dataset.batch(
        "train",
        batch_size,
        generator,
        device,
    )

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )
    autocast = torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=use_amp,
    )

    flops = _count_forward_flops(raw_model, x, y)

    latency_seconds = None
    training_step_seconds = None
    inference_peak_mb = None
    training_peak_mb = None
    inference_timing_summary = _timing_summary([])
    training_timing_summary = _timing_summary([])

    try:
        model.eval()

        # 纯推理测量不回流训练图，与 evaluate 一致使用
        # inference_mode。
        with torch.inference_mode(), autocast:
            for _ in range(max(0, warmup)):
                model(x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings: List[float] = []
        with torch.inference_mode(), autocast:
            for _ in range(max(1, iters)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(x, y)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)

        timings.sort()
        latency_seconds = float(_percentile(timings, 0.5))
        inference_timing_summary = _timing_summary(timings)

        # 训练步计时（前向 + 反向，不含优化器更新）与推理
        # 计时使用同一 warmup/iters 口径，取中位数。
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        with autocast:
            for _ in range(max(0, warmup)):
                output = model(x, y)
                output["loss"].backward()
                model.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        training_timings: List[float] = []
        with autocast:
            for _ in range(max(1, iters)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = model(x, y)
                output["loss"].backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                training_timings.append(
                    time.perf_counter() - start
                )
                model.zero_grad(set_to_none=True)

            if device.type == "cuda":
                training_peak_mb = round(
                    torch.cuda.max_memory_allocated(device)
                    / 1e6,
                    3,
                )

        training_timings.sort()
        training_step_seconds = float(
            _percentile(training_timings, 0.5)
        )
        training_timing_summary = _timing_summary(
            training_timings
        )
        del output

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), autocast:
                model(x, y)
            inference_peak_mb = round(
                torch.cuda.max_memory_allocated(device)
                / 1e6,
                3,
            )
    finally:
        model.train(was_training)

    tokens = int(x.numel())

    result: Dict[str, object] = {
        "n_params_total": int(
            sum(
                parameter.numel()
                for parameter in raw_model.parameters()
            )
        ),
        "n_params_non_embedding": int(
            _count_non_embedding_params(raw_model)
        ),
        "flops_per_forward": flops[
            "total_flops_per_forward"
        ],
        "flops_linear": flops["linear_flops"],
        "flops_attention": flops["attention_flops"],
        "flops_note": flops["flops_note"],
        "latency_ms_per_forward": (
            round(latency_seconds * 1e3, 4)
            if latency_seconds is not None
            else None
        ),
        "training_ms_per_step": (
            round(training_step_seconds * 1e3, 4)
            if training_step_seconds is not None
            else None
        ),
        "training_ms_note": (
            "前向+反向的中位计时；不含优化器更新/梯度裁剪"
        ),
        # 完整计时分布（毫秒）：median 之外给 p95/p99、std 与
        # 变异系数，并保留逐次原始样本供复核。单次中位数无法
        # 区分「稳定快」与「偶发长尾」。
        "latency_ms": inference_timing_summary,
        "training_ms": training_timing_summary,
        "timing_samples_seconds": {
            "inference": [
                round(float(value), 9)
                for value in timings
            ],
            "training": [
                round(float(value), 9)
                for value in training_timings
            ],
        },
        "throughput_tokens_per_sec": (
            round(tokens / latency_seconds, 2)
            if latency_seconds
            else None
        ),
        "inference_peak_memory_mb": inference_peak_mb,
        # 该值只覆盖前向+反向的合成训练步，不含优化器状态
        # （AdamW 一阶/二阶矩）与参数更新；真实训练循环的峰值
        # 另由 actual_training_peak_memory_mb 记录，跨运行
        # 对比优先使用后者。
        "training_peak_memory_mb": training_peak_mb,
        "training_peak_memory_note": (
            "前向+反向合成步峰值；不含优化器状态/更新，"
            "真实训练峰值见 actual_training_peak_memory_mb"
        ),
        "process_peak_rss_mb": _process_peak_rss_mb(),
        "batch_size": int(batch_size),
        "seq_len": int(x.size(1)),
        "tokens_per_forward": tokens,
        # 完整计算图口径与 KV cache 估计：论文要求 FLOPs、总参数
        # 与 KV cache 三者共同匹配；这里把解析成本与容量一起
        # 落盘，供 --compare / --budget-match 复核。
        "cost_profile": mode_cost_profile(cfg),
        "kv_cache_bytes_per_token": mode_cost_profile(cfg)[
            "kv_cache_bytes_per_token"
        ],
        "measured_flops_per_token": (
            round(
                float(flops["total_flops_per_forward"])
                / max(1, tokens),
                3,
            )
            if flops["total_flops_per_forward"] is not None
            else None
        ),
        "analytic_flops_per_token": mode_cost_profile(cfg)[
            "analytic_flops_per_token"
        ],
        "measurement": {
            "device": str(device),
            "amp_dtype": (
                str(amp_dtype)
                if amp_dtype is not None
                else None
            ),
            "warmup": int(warmup),
            "iters": int(iters),
            "mode": cfg.mode,
        },
    }

    return result



def print_efficiency(efficiency: Dict[str, object]) -> None:
    """打印效率测量摘要。"""
    def _format(value, unit="", spec=".3f"):
        if value is None:
            return "-"
        return _format_finite(value, spec) + unit

    print(
        "[efficiency] "
        f"params={efficiency.get('n_params_total')} "
        f"(non-embedding="
        f"{efficiency.get('n_params_non_embedding')}) | "
        f"FLOPs/forward="
        f"{_format(efficiency.get('flops_per_forward'), '', '.3e')} | "
        f"latency={_format(efficiency.get('latency_ms_per_forward'), 'ms')} | "
        f"train_step="
        f"{_format(efficiency.get('training_ms_per_step'), 'ms')} | "
        f"throughput="
        f"{_format(efficiency.get('throughput_tokens_per_sec'), 'tok/s')}"
    )
    for label, key in (
        ("inference", "latency_ms"),
        ("train_step", "training_ms"),
    ):
        summary = efficiency.get(key)
        if not isinstance(summary, dict) or not summary.get("n"):
            continue
        print(
            f"[efficiency] {label} 分布（n={summary['n']}）: "
            f"p50={_format(summary.get('median'), 'ms')} "
            f"p95={_format(summary.get('p95'), 'ms')} "
            f"p99={_format(summary.get('p99'), 'ms')} "
            f"std={_format(summary.get('std'), 'ms')} "
            "CV="
            f"{_format(summary.get('coefficient_of_variation'), '', '.1%')}"
            f" max={_format(summary.get('max'), 'ms')}"
        )
    print(
        "[efficiency] "
        f"inference_peak="
        f"{_format(efficiency.get('inference_peak_memory_mb'), 'MB')} | "
        f"fwd_bwd_peak="
        f"{_format(efficiency.get('training_peak_memory_mb'), 'MB')} | "
        f"actual_train_peak="
        f"{_format(efficiency.get('actual_training_peak_memory_mb'), 'MB')} | "
        f"process_peak_rss="
        f"{_format(efficiency.get('process_peak_rss_mb'), 'MB')}"
    )



# 等计算量对比：训练一步的 FLOPs 近似为 3×前向 FLOPs
# （前向 + 反向约等于 2 次前向）。该常数只用于把计算量换算成
# 「训练 FLOPs」的绝对量级，各运行共享同一常数，不改变共同
# 计算量点的位置。
_TRAIN_FLOPS_MULTIPLIER = 3.0


_COMPUTE_MATCHED_METRIC = "val_loss_at_matched_flops"


# 选型偏差的无偏对照：从周期验证曲线取共同训练步数处的即时
# 指标。``final`` 的 val 指标来自验证集选中的 checkpoint
# （best_val），同一批数据既选型又判定，最优值噪声会被当成
# 方法差异；周期验证点是「当时模型」的即时指标，不经过
# checkpoint 选择。与等计算量 loss 一样，它是分析期派生指标。
#
# 只给 loss 提供选型无关口径是不够的：类别不平衡时项目建议用
# macro F1 / PR-AUC 判定，假设也可能预注册 accuracy、precision
# 或 recall；这些指标同样会被 checkpoint 选择抬高。因此凡是
# 周期验证实际计算的 val 指标，都生成一份共同步数派生指标。
_COMMON_STEP_METRIC_SUFFIX = "_at_common_step"

_COMMON_STEP_CORE_BASE_METRICS: Tuple[str, ...] = (
    "val_loss",
    "val_ppl",
    "val_bits_per_byte",
    # 类平衡似然同样会被 checkpoint 选择抬高（选型指标若按
    # 频率加权 loss，稀有字节的表现不影响选型结果），因此
    # 也要有避开选型偏差的共同步数派生口径。
    "val_macro_nll",
    "val_token_accuracy",
    "val_micro_f1",
    "val_macro_f1",
    "val_weighted_f1",
    "val_precision_macro",
    "val_precision_weighted",
    "val_recall_macro",
    "val_recall_weighted",
)

# 排序指标（PR-AUC / lift / ROC-AUC）默认只在 final/test 计算；
# 显式选为选型指标或传 --periodic-auc 时，周期验证也会写出这些
# 键。它们同样需要避开 checkpoint 选型偏差的共同步数口径——
# 极度不平衡时项目建议以 PR-AUC 判定，若只有 final 版本，
# 判定指标恰恰是最容易被选型抬高、又没有选型无关复核的那个。
_COMMON_STEP_OPTIONAL_BASE_METRICS: Tuple[str, ...] = (
    "val_pr_auc_macro",
    "val_pr_auc_macro_lift",
    "val_roc_auc_macro",
)

_COMMON_STEP_BASE_METRICS: Tuple[str, ...] = (
    _COMMON_STEP_CORE_BASE_METRICS
    + _COMMON_STEP_OPTIONAL_BASE_METRICS
)

_COMMON_STEP_METRICS: Tuple[str, ...] = tuple(
    base + _COMMON_STEP_METRIC_SUFFIX
    for base in _COMMON_STEP_BASE_METRICS
)

_COMMON_STEP_METRIC = "val_loss_at_common_step"


# RQ1 里的「收敛速度」不能只由共同步数处的单点 loss 回答：
# 单点只描述某一时刻的质量，两个方法可以在该点相同、但整个
# 训练过程中的损失轨迹明显不同。这里新增一个分析期派生指标：
# 在共同的训练步数窗口上对周期验证曲线做梯形积分，再除以窗口
# 长度，得到「时均验证损失」（越低说明窗口内整体损失更低，
# 常与更早收敛一致，但不能单独分离速度与最终水平）。
#
# 它仍使用周期验证记录（不经过 checkpoint 选型），窗口起点取
# 各运行**第一个**周期验证步的最大值、终点取训练前固定的
# ``--common-step``（未固定时退回各运行最后验证步的最小值），
# 两端都在曲线范围内做线性插值，不做外推。窗口起点也是观测
# 端点：首段（0 到第一次验证）没有观测，不能凭空假设，因此
# 该指标回答的是「共同可观测窗口内谁的整体损失更低」，不是
# 从第 0 步开始的完整曲线面积。
_CURVE_MEAN_METRIC = "val_loss_curve_mean_to_common_step"

_CURVE_DERIVED_METRICS: Tuple[str, ...] = (
    _CURVE_MEAN_METRIC,
)



def _common_step_metric_name(base_metric: str) -> Optional[str]:
    """周期指标名 → 共同步数派生指标名；非周期指标返回 None。"""
    if base_metric not in _COMMON_STEP_BASE_METRICS:
        return None
    return base_metric + _COMMON_STEP_METRIC_SUFFIX



def _common_step_metric_base(metric: str) -> Optional[str]:
    """共同步数派生指标名 → 原始周期指标名。"""
    if not metric.endswith(_COMMON_STEP_METRIC_SUFFIX):
        return None
    base = metric[: -len(_COMMON_STEP_METRIC_SUFFIX)]
    return base if base in _COMMON_STEP_BASE_METRICS else None


# local_base 的第二轮为了与 A/B 对齐计算量，把 k1 个分支合并
# 成一次宽度 min(n_experts, k1*k2) 的路由；均衡损失约随路由
# 宽度线性增长，直接沿用会让「立即合并」臂的均衡强度高于
# 延迟合并臂。该常量记录本轮实现已把 local_base 第二轮的
# aux loss 折算回单分支宽度 k2，供均衡强度审计区分新旧运行。
_BALANCE_WIDTH_NORMALIZATION = (
    "local_base_second_round_scaled_to_k2"
)



def _is_deterministic_run(row: Dict[str, object]) -> bool:
    """
    该运行是否不使用随机性。

    统计基线 ngram 的计数、调参与固定窗口评估都是确定性的：
    换一个 --seed 重跑不会产生新的独立重复，只是同一估计的
    复制。把不同 seed 的 ngram 运行当成独立样本会虚增 n、
    把种子方差压成 0，甚至让检验出现假阳性。未来若有其他
    确定性基线，可在 config.json 写 deterministic=true。
    """
    if bool(row.get("deterministic")):
        return True
    return str(row.get("mode")) == "ngram"



def _finite_float(value: object) -> Optional[float]:
    """把数值/布尔以外的有限数值转成 float，否则返回 None。"""
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        # JSON 允许任意大整数；float(10**400) 会溢出。分析
        # 管线应把这种值当作不可用，而不是让整份报告崩溃。
        return None
    if not math.isfinite(numeric):
        return None
    return numeric



def _format_finite(
    value: object,
    spec: str = ".4f",
    scale: float = 1.0,
) -> str:
    """格式化有限数值用于终端展示；损坏/超范围值返回 "-"。

    报告里的数值可能来自手工编辑或旧版 JSON（含超出 float
    范围的整数与 NaN/Inf），单行坏记录不能中断整张对比表。
    """
    number = _finite_float(value)
    if number is None:
        return "-"
    scaled = number * scale
    if not math.isfinite(scaled):
        return "-"
    return format(scaled, spec)



def _nonnegative_int(value: object) -> int:
    """非负整数计数；bool/损坏值按 0 处理，供旧报告兼容。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, int(value))



def _as_dict(value: object) -> Dict[str, object]:
    """dict 原样返回；其余值（旧/损坏 config 字段）当空 dict。"""
    return value if isinstance(value, dict) else {}



def _path_is_dir(value: object) -> bool:
    """Path.is_dir 的容错版本：损坏/超长路径不能中断分析。

    ``Path.exists()/is_dir()`` 在文件名过长（ENAMETOOLONG）、
    权限或 IO 错误时会抛 OSError；来自手工编辑的 config 或
    命令行路径不应让整份对比/设计报告崩溃，按「不存在」处理。
    """
    try:
        return Path(value).is_dir()
    except (OSError, TypeError, ValueError):
        return False



def _path_exists(value: object) -> bool:
    """Path.exists 的容错版本；不可判定的路径按「不存在」处理。"""
    try:
        return Path(value).exists()
    except (OSError, TypeError, ValueError):
        return False
