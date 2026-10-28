from __future__ import annotations



def _probe_gradient_flow(model: nn.Module) -> Dict[str, object]:
    """
    在第一次反向后检查梯度是否健康。

    比「能否过拟合」更早、更直接地定位严重问题：
    - 至少一个参数收到非零梯度（梯度未反传、参数未进入优化器
      会让训练空转）；
    - 全部梯度必须有限（NaN/Inf 会静默污染所有参数）；
    - 非专家参数不允许缺失梯度（detach、意外 no_grad、分支
      写死等造成的计算图断链）；
    - 非专家参数也不允许整个张量的梯度恒为零：缺失梯度会被
      单独拦截，但「分支执行了却乘了 0 / 被条件性地短路」只会
      表现为全零梯度，单看全局梯度范数仍可能被其他参数掩盖。

    专家参数允许因本批次未路由到而缺失或为零梯度。调用方需在
    反向之后、优化器 step 之前调用（AMP 下先手动
    scaler.unscale_）；返回的 failures 列表为空才视为通过。
    """
    expert_ids = _expert_parameter_ids(model)

    probe: Dict[str, object] = {
        "n_params": 0,
        "n_with_grad": 0,
        "n_missing_grad": 0,
        "n_zero_grad": 0,
        "n_nonfinite_grad": 0,
        "missing_non_expert_params": [],
        "nonfinite_params": [],
        "zero_non_expert_params": [],
        "grad_norm": None,
        "failures": [],
    }

    missing_non_expert: List[str] = probe[
        "missing_non_expert_params"
    ]  # type: ignore[assignment]
    nonfinite: List[str] = probe[
        "nonfinite_params"
    ]  # type: ignore[assignment]
    zero_non_expert: List[str] = probe[
        "zero_non_expert_params"
    ]  # type: ignore[assignment]

    total_sq = 0.0

    # 逐参数的有限性判定与平方和打包后再一次性取回。
    pending: List[Tuple[str, bool, torch.Tensor]] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        probe["n_params"] = int(probe["n_params"]) + 1
        grad = parameter.grad

        if grad is None:
            probe["n_missing_grad"] = (
                int(probe["n_missing_grad"]) + 1
            )
            if id(parameter) not in expert_ids:
                missing_non_expert.append(name)
            continue

        probe["n_with_grad"] = int(probe["n_with_grad"]) + 1

        # 有限性掩码与平方和打包成 [finite, squared] 的对，
        # 与逐参数处理的分支顺序保持一致。第三项记录是否为
        # 专家参数，供后续区分「合法未路由」与「断链零梯度」。
        pending.append((
            name,
            id(parameter) in expert_ids,
            torch.stack([
                torch.isfinite(grad).all().to(torch.float64),
                grad.detach().double().pow(2).sum(),
            ]),
        ))

    if pending:
        values = torch.stack(
            [packed for _, _, packed in pending]
        ).tolist()
    else:
        values = []

    for (name, is_expert, _), packed in zip(pending, values):
        finite_value = packed[0]
        squared = packed[1]

        if finite_value == 0.0:
            probe["n_nonfinite_grad"] = (
                int(probe["n_nonfinite_grad"]) + 1
            )
            nonfinite.append(name)
            continue

        total_sq += squared

        if squared == 0.0:
            probe["n_zero_grad"] = int(probe["n_zero_grad"]) + 1
            if not is_expert:
                zero_non_expert.append(name)

    probe["grad_norm"] = math.sqrt(total_sq)

    failures: List[str] = probe["failures"]  # type: ignore[assignment]

    if int(probe["n_nonfinite_grad"]) > 0:
        failures.append("nonfinite_gradients")
    if int(probe["n_with_grad"]) == 0 or total_sq == 0.0:
        failures.append("no_nonzero_gradient")
    if missing_non_expert:
        failures.append("missing_non_expert_gradients")
    if zero_non_expert:
        failures.append("zero_non_expert_gradients")

    return probe



def _probe_optimizer_coverage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, object]:
    """
    核对优化器参数组是否覆盖全部可训练参数。

    梯度探针只能证明参数「收到了梯度」；如果优化器在模型
    构造完成前创建、漏掉某个子模块或传入了错误参数列表，
    这些参数会一直停在初始值，而损失仍可能因其余参数更新
    而下降。这里直接比较参数对象的 id 集合，给出缺失参数名，
    并检查重复参数（同一参数被多个 param group 更新两次）。
    """
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for _, parameter in trainable}

    optimizer_ids = set()
    duplicate_params: List[str] = []
    id_to_name = {
        id(parameter): name
        for name, parameter in trainable
    }

    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            parameter_id = id(parameter)
            if parameter_id in optimizer_ids:
                duplicate_params.append(
                    id_to_name.get(
                        parameter_id,
                        f"<unknown:{parameter_id}>",
                    )
                )
            optimizer_ids.add(parameter_id)

    missing_params = [
        name
        for name, parameter in trainable
        if id(parameter) not in optimizer_ids
    ]
    foreign_param_count = sum(
        1
        for parameter_id in optimizer_ids
        if parameter_id not in trainable_ids
    )

    return {
        "n_trainable_params": len(trainable),
        "n_optimizer_params": len(optimizer_ids),
        "missing_params": missing_params,
        "duplicate_params": sorted(set(duplicate_params)),
        "foreign_param_count": foreign_param_count,
        "ok": (
            bool(trainable)
            and not missing_params
            and not duplicate_params
            and foreign_param_count == 0
        ),
    }



def _non_expert_parameter_digest(
    model: nn.Module,
) -> str:
    """
    非专家可训练参数的稳定内容指纹。

    sanity check 第一步 optimizer.step 前后各取一次：内容不变
    说明优化器没有真正更新任何非专家参数（lr=0、参数未加入
    optimizer、GradScaler 跳步或被冻结），即使损失曲线仍可能
    因专家参数更新而下降。专家参数允许因本批次未路由而不变，
    因此不纳入该指纹。
    """
    expert_ids = _expert_parameter_ids(model)
    digest = hashlib.sha256()

    for name, parameter in model.named_parameters():
        if (
            not parameter.requires_grad
            or id(parameter) in expert_ids
        ):
            continue

        digest.update(name.encode("utf-8"))
        raw = parameter.detach().to(
            device="cpu",
            dtype=parameter.dtype,
        ).contiguous()
        digest.update(str(raw.dtype).encode("ascii"))
        digest.update(str(tuple(raw.shape)).encode("ascii"))
        raw_bytes = raw.reshape(-1).view(torch.uint8)
        try:
            payload = raw_bytes.numpy().tobytes()
        except (TypeError, RuntimeError):
            payload = bytes(raw_bytes.tolist())
        digest.update(payload)

    return digest.hexdigest()



# 梯度探针失败原因到排查建议的映射。
_GRADIENT_FAILURE_HINTS = {
    "loss_not_connected": (
        "损失未连入计算图（requires_grad=False 或 grad_fn 为空）："
        "检查是否误用 torch.no_grad / detach / 原地操作改写 logits"
    ),
    "nonfinite_gradients": (
        "梯度含 NaN/Inf：检查损失计算中的 log(0)/除零/softmax "
        "溢出，再查混合精度与学习率/梯度爆炸"
    ),
    "no_nonzero_gradient": (
        "没有任何参数收到非零梯度：检查损失是否用错了输出、"
        "参数是否被 detach，或优化器是否拿到的不是模型参数"
    ),
    "missing_non_expert_gradients": (
        "非专家参数缺失梯度：计算图在这些模块处断链"
        "（detach / no_grad / 条件分支未执行）"
    ),
    "zero_non_expert_gradients": (
        "非专家参数整个张量的梯度恒为零：该模块虽然执行了，"
        "但输出被乘 0、被条件分支短路或未参与损失；请检查"
        "gate/掩码/分支合并处是否把有效路径清零"
    ),
}



def _batch_sha256(
    x: torch.Tensor,
    y: torch.Tensor,
) -> str:
    """
    (x, y) 固定小批次的稳定指纹。

    重复运行只比较损失无法区分「同一批次」与「碰巧得到相同
    损失的不同批次」。把 dtype / 形状 / 原始字节一起哈希，可
    在 sanity 报告中直接核对两次重复是否真的消费了同一份
    数据；数据管线若暗中依赖未固定状态，指纹会先变化。
    """
    digest = hashlib.sha256()

    for label, tensor in (("x", x), ("y", y)):
        cpu = tensor.detach().to("cpu").contiguous()
        digest.update(label.encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        raw = cpu.view(torch.uint8)
        if np is not None:
            digest.update(raw.numpy().tobytes())
        else:
            digest.update(bytes(raw.tolist()))

    return digest.hexdigest()



def run_sanity_overfit(
    cfg: Config,
    dataset: ByteDataset,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    *,
    batch_size: int,
    steps: int,
    lr: float,
    loss_threshold: float,
    acc_threshold: float,
    min_relative_improvement: float = 0.5,
    seed: int,
    grad_checkpoint: bool = False,
    strict_determinism: bool = False,
) -> Dict[str, object]:
    """
    过拟合小批量数据的 sanity check。

    在开始全量训练前，用同一模型结构在一个固定小批次上训练：
    关闭负载均衡辅助损失、weight decay、梯度裁剪、warmup 等
    一切正则化（数据管线本身没有 dropout / 数据增强），要求训练
    损失迅速趋近 0 或下一 token 准确率接近 100%。

    通过条件同时要求「确实学到了东西」：最终损失相对第一次
    前向（任何优化器更新之前）至少下降 min_relative_improvement，
    且满足绝对阈值或准确率阈值之一，二者同时成立才算通过。

    第一步反向之后还会先做梯度探针（见 _probe_gradient_flow）：
    损失必须连在计算图上、梯度必须有限、至少一个参数收到非零
    梯度、非专家参数不得缺失梯度；探针失败时立即停止并给出
    排查提示。

    检查必须走与全量训练一致的前向/反向路径：主训练启用
    --grad-checkpoint 时本检查同样开启 activation
    checkpointing（其反向会对 Block 重算前向，是唯一会改变
    反传代码路径的开关），避免「sanity check 验证了一条
    路径、全量训练跑的是另一条路径」。--compile 只做内核
    融合、不改写 autograd 语义，不镜像。

    这是必要不充分的检查：连小批量都无法过拟合，通常意味着
    梯度没有正确反传、损失计算错误、优化器配置错误等问题；
    但通过也不代表代码完全正确，数据增强、特征对齐、标签映射
    等问题仍可能漏网，仍需配合中间结果检查。

    函数内部固定 seed 并在结束后恢复调用前的全部 RNG 状态，
    因此是否运行本检查不影响主训练的随机轨迹。
    """
    saved_rng_state = _capture_rng_state()

    try:
        set_seed(
            seed,
            strict_determinism=strict_determinism,
        )

        # balance_coef=0：关闭辅助负载均衡损失（正则项）。
        sanity_cfg = replace(cfg, balance_coef=0.0)
        model = LanguageModel(sanity_cfg).to(device)
        model.grad_checkpoint = grad_checkpoint
        model.train()

        n_params = sum(
            p.numel() for p in model.parameters()
        )

        # 固定一个小批次：整个检查期间重复使用同一批数据。
        # 采样必须只消费显式 generator：若数据管线暗中使用
        # torch 全局 RNG，任何在前面的全局随机消耗（模型初始
        # 化、其他模块）都会改变数据轨迹，同种子结果在真实
        # 训练中就不再稳定。这里在采样前后核对全局 torch RNG
        # 状态未被触碰（DataLoader worker 场景对应 worker 端
        # 的 generator 隔离）。
        global_rng_before_batch = torch.get_rng_state()
        generator = torch.Generator().manual_seed(seed)
        x, y = dataset.batch(
            "train",
            batch_size,
            generator,
            device,
        )
        data_sampling_uses_explicit_generator = torch.equal(
            torch.get_rng_state(),
            global_rng_before_batch,
        )
        batch_sha256 = _batch_sha256(x, y)
        print(
            "[sanity] fixed batch sha256="
            f"{batch_sha256[:12]}..."
        )
        if not data_sampling_uses_explicit_generator:
            print(
                "[sanity] [warn] 训练批采样消耗了 torch 全局 "
                "RNG：同一批数据会随全局随机消耗顺序漂移，"
                "请让数据管线只使用传入的 torch.Generator"
            )

        # weight_decay=0、无 warmup、无梯度裁剪：全部正则化关闭。
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
        )

        # 优化器覆盖检查与「第一步是否真的更新权重」检查：
        # 梯度存在但参数没进优化器、lr=0 或 GradScaler 跳步，
        # 都会让训练空转，损失/准确率曲线可能暂时看不出来。
        optimizer_coverage = _probe_optimizer_coverage(
            model,
            optimizer,
        )
        if not optimizer_coverage["ok"]:
            print(
                "[sanity] [warn] 优化器参数覆盖不完整: "
                f"missing={optimizer_coverage['missing_params']}, "
                "duplicate="
                f"{optimizer_coverage['duplicate_params']}, "
                "foreign="
                f"{optimizer_coverage['foreign_param_count']}"
            )

        use_amp = (
            amp_dtype is not None
            and device.type == "cuda"
        )
        scaler = None
        if use_amp and amp_dtype == torch.float16:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()

        report_every = max(1, steps // 5)

        initial_total_loss: Optional[float] = None
        initial_lm_loss: Optional[float] = None
        final_total_loss = float("nan")
        final_lm_loss = float("nan")
        final_acc = float("nan")
        grad_probe: Optional[Dict[str, object]] = None
        pre_step_parameter_digest: Optional[str] = None
        post_step_parameter_digest: Optional[str] = None

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)

                result = model(x, y)
                # 与全量训练使用同一条损失聚合路径。sanity_cfg
                # 已把正则系数设为 0，数值上 total_loss 就是 lm_loss，
                # 但代码路径仍会执行 lm + coef*aux 的求和；若 aux
                # 侧出现 NaN/Inf 或聚合实现错误，全量训练会坏而
                # 只看 lm_loss 的检查会漏掉。
                loss = result["loss"]
                lm_loss = result["lm_loss"]

                # 在任何优化器更新之前记录初始损失：它是「模型
                # 是否真的在学习」的参照。恒定/错误的损失实现
                # 无法靠绝对阈值区分，必须看到相对下降。
                if step == 1:
                    initial_total_loss = float(loss.detach())
                    initial_lm_loss = float(lm_loss.detach())

                # 损失必须连在计算图上；否则 backward 必然失败
                # 或训练空转，直接停下。
                loss_connected = True
                if step == 1 and (
                    not loss.requires_grad
                    or loss.grad_fn is None
                ):
                    loss_connected = False
                    grad_probe = {
                        "n_params": None,
                        "n_with_grad": 0,
                        "n_missing_grad": None,
                        "n_zero_grad": None,
                        "n_nonfinite_grad": None,
                        "missing_non_expert_params": [],
                        "nonfinite_params": [],
                        "zero_non_expert_params": [],
                        "grad_norm": None,
                        "failures": ["loss_not_connected"],
                    }

                if scaler is not None:
                    if loss_connected:
                        scaler.scale(loss).backward()

                        if step == 1:
                            # 手动反缩放后再检查梯度；随后
                            # scaler.step 不会重复反缩放。
                            scaler.unscale_(optimizer)
                            grad_probe = _probe_gradient_flow(
                                model
                            )
                            pre_step_parameter_digest = (
                                _non_expert_parameter_digest(
                                    model
                                )
                            )

                        scaler.step(optimizer)
                        scaler.update()

                        if step == 1:
                            post_step_parameter_digest = (
                                _non_expert_parameter_digest(
                                    model
                                )
                            )
                    # 损失未连接时没有 scale/backward，GradScaler
                    # 没有 inf 检查记录，调用 step 会直接抛
                    # RuntimeError；此时跳过优化器步进，保留
                    # grad_probe 的失败证据供上层报告。
                else:
                    if loss_connected:
                        loss.backward()

                        if step == 1:
                            grad_probe = _probe_gradient_flow(
                                model
                            )
                            pre_step_parameter_digest = (
                                _non_expert_parameter_digest(
                                    model
                                )
                            )

                    optimizer.step()

                    if step == 1 and loss_connected:
                        post_step_parameter_digest = (
                            _non_expert_parameter_digest(model)
                        )

                # fused AdamW 不推进参数版本号，显式使专家权重
                # stack 缓存失效，确保下一步前向使用新参数。
                model.invalidate_weight_caches()

                # 仅在需要报告或收尾的步取回标量，取值来源
                # 与时刻均与原实现一致。
                if (
                    step == 1
                    or step % report_every == 0
                    or step == steps
                ):
                    final_total_loss = float(loss.detach())
                    final_lm_loss = float(lm_loss.detach())
                    final_acc = float(
                        (
                            result["logits"].argmax(dim=-1) == y
                        ).float().mean()
                    )

                    print(
                        f"[sanity] step {step}/{steps} "
                        f"total_loss={final_total_loss:.4f} "
                        f"lm_loss={final_lm_loss:.4f} "
                        f"token_acc={final_acc:.4f}"
                    )

                # 探针失败：立即停止训练并给出排查提示。
                if step == 1 and (
                    grad_probe is not None
                    and grad_probe["failures"]
                ):
                    print(
                        "[sanity] 梯度检查失败: "
                        + "; ".join(
                            _GRADIENT_FAILURE_HINTS.get(
                                failure, failure
                            )
                            for failure in grad_probe["failures"]
                        )
                    )
                    break

        # 相对下降证据：绝对阈值/准确率只能说明最终状态，
        # 不能排除「损失实现错误导致恒定小值」这类假通过。
        loss_reduction: Optional[float] = None
        if (
            initial_total_loss is not None
            and math.isfinite(initial_total_loss)
            and initial_total_loss > 0.0
            and math.isfinite(final_total_loss)
        ):
            loss_reduction = (
                initial_total_loss - final_total_loss
            ) / initial_total_loss

        learned = bool(
            loss_reduction is not None
            and loss_reduction >= min_relative_improvement
        )

        metric_ok = (
            final_total_loss <= loss_threshold
            or final_acc >= acc_threshold
        )

        parameter_update_ok = bool(
            pre_step_parameter_digest is not None
            and post_step_parameter_digest is not None
            and pre_step_parameter_digest
            != post_step_parameter_digest
        )

        passed = metric_ok and learned

        if not data_sampling_uses_explicit_generator:
            passed = False

        if not optimizer_coverage["ok"]:
            passed = False

        if not parameter_update_ok:
            passed = False

        if initial_total_loss is not None:
            reduction_text = (
                f"{loss_reduction:.1%}"
                if loss_reduction is not None
                else "无法计算（初始/最终损失非有限或不正）"
            )
            print(
                "[sanity] 学习证据: "
                f"initial_total_loss={initial_total_loss:.4f}, "
                f"final_total_loss={final_total_loss:.4f}, "
                f"initial_lm_loss={initial_lm_loss:.4f}, "
                f"final_lm_loss={final_lm_loss:.4f}, "
                f"relative_reduction={reduction_text}, "
                f"required>={min_relative_improvement:.1%}"
            )

        if grad_probe is not None:
            if grad_probe["failures"]:
                passed = False

            print(
                "[sanity] 梯度检查: "
                f"grad_norm={grad_probe['grad_norm']}, "
                f"missing_grad="
                f"{grad_probe['n_missing_grad']}, "
                f"zero_grad={grad_probe['n_zero_grad']}, "
                f"nonfinite_grad="
                f"{grad_probe['n_nonfinite_grad']}"
                + (
                    ", failures="
                    + ",".join(grad_probe["failures"])
                    if grad_probe["failures"]
                    else ""
                )
            )

        print(
            "[sanity] 优化器检查: "
            f"coverage_ok={optimizer_coverage['ok']} "
            f"({optimizer_coverage['n_optimizer_params']}/"
            f"{optimizer_coverage['n_trainable_params']}), "
            f"parameter_update_ok={parameter_update_ok}"
        )
        if not optimizer_coverage["ok"]:
            print(
                "[sanity] 优化器未覆盖全部可训练参数：请检查"
                " optimizer 是否在模型构建完成后用 "
                "model.parameters() 创建，参数组是否遗漏子模块"
            )
        if not parameter_update_ok:
            print(
                "[sanity] 第一步 optimizer.step 未改变任何非专家"
                "参数：请检查 lr 是否为 0、参数是否被冻结、"
                "GradScaler 是否因子梯度溢出跳过了更新"
            )

        return {
            "passed": passed,
            "n_params": n_params,
            "steps": steps,
            "batch_size": batch_size,
            "batch_sha256": batch_sha256,
            "data_sampling_uses_explicit_generator": (
                data_sampling_uses_explicit_generator
            ),
            "lr": lr,
            "seed": seed,
            "grad_checkpoint": grad_checkpoint,
            "strict_determinism": strict_determinism,
            "amp_dtype": (
                str(amp_dtype) if amp_dtype is not None else None
            ),
            # lm_loss 仅用于报告；passed/loss_reduction/阈值都基于
            # 与全量训练相同的 total_loss 聚合路径。
            "loss_source": (
                "result['loss'] with balance_coef=0"
            ),
            "initial_lm_loss": initial_lm_loss,
            "final_lm_loss": final_lm_loss,
            "initial_total_loss": initial_total_loss,
            "final_total_loss": final_total_loss,
            "final_token_accuracy": final_acc,
            "loss_reduction": loss_reduction,
            "min_relative_improvement": min_relative_improvement,
            "learned": learned,
            "loss_threshold": loss_threshold,
            "acc_threshold": acc_threshold,
            "gradient_flow_ok": (
                grad_probe is not None
                and not grad_probe["failures"]
            ),
            "optimizer_coverage": optimizer_coverage,
            "optimizer_coverage_ok": optimizer_coverage["ok"],
            "parameter_update_ok": parameter_update_ok,
            "pre_step_parameter_sha256": (
                pre_step_parameter_digest
            ),
            "post_step_parameter_sha256": (
                post_step_parameter_digest
            ),
            "grad_probe": grad_probe,
        }
    finally:
        # 恢复调用前的 RNG 状态：sanity check 不改变主训练轨迹。
        _restore_rng_state(saved_rng_state)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()



_SANITY_REPEAT_KEYS: Tuple[str, ...] = (
    "batch_sha256",
    "data_sampling_uses_explicit_generator",
    "initial_lm_loss",
    "final_lm_loss",
    "initial_total_loss",
    "final_total_loss",
    "final_token_accuracy",
    "loss_reduction",
    "loss_source",
    "learned",
    "gradient_flow_ok",
    "optimizer_coverage_ok",
    "parameter_update_ok",
)



def sanity_repeat_report(
    reports: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    核对同一配置 / 同一种子的多次 sanity 运行是否逐位一致。

    固定种子只说明「随机流的起点相同」；若数据采样、模型
    初始化或反传路径仍然依赖未固定的全局状态、非确定性
    算子或异步执行顺序，第二次训练就会偏离。这里以第一次
    运行为基准，逐项比较初始/最终损失、准确率、相对下降与
    梯度探针的范数；任何差异都视为可复现性问题，正式训练
    不应在缺少解释的情况下开始。

    只比较确定性检查关心的标量，不比较路径/时间等元数据。
    """
    if not reports:
        return {
            "repeats": 0,
            "identical": None,
            "mismatches": [],
        }

    first = reports[0]
    mismatches: List[Dict[str, object]] = []

    for repeat_index, report in enumerate(
        reports[1:],
        start=2,
    ):
        for key in _SANITY_REPEAT_KEYS:
            first_value = first.get(key)
            repeat_value = report.get(key)
            if first_value != repeat_value:
                mismatches.append({
                    "repeat": repeat_index,
                    "key": key,
                    "first": first_value,
                    "repeat_value": repeat_value,
                })

        first_probe = first.get("grad_probe")
        repeat_probe = report.get("grad_probe")
        first_norm = (
            first_probe.get("grad_norm")
            if isinstance(first_probe, dict)
            else None
        )
        repeat_norm = (
            repeat_probe.get("grad_norm")
            if isinstance(repeat_probe, dict)
            else None
        )
        if first_norm != repeat_norm:
            mismatches.append({
                "repeat": repeat_index,
                "key": "grad_probe.grad_norm",
                "first": first_norm,
                "repeat_value": repeat_norm,
            })

    return {
        "repeats": len(reports),
        "identical": (
            None if len(reports) < 2 else not mismatches
        ),
        "mismatches": mismatches,
    }



# ============================================================
# Experiment tracking
# ============================================================

def _git_output(*args: str) -> Optional[str]:
    """
    在仓库根目录运行 git 命令并返回 stdout；失败时返回 None。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parent),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()



def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



def _script_sha256() -> Optional[str]:
    """
    实验脚本自身的 SHA-256 指纹。

    本项目是单文件代码库，脚本指纹即完整实验代码的指纹：
    git 不可用（如压缩包直接运行）时它是唯一的代码版本记录，
    工作区有未提交修改时它能把「同一 commit 的不同脏状态」
    区分开——只有 commit + dirty 布尔位做不到这一点。
    """
    try:
        source = Path(__file__).resolve().read_bytes()
    except OSError:
        return None
    return hashlib.sha256(source).hexdigest()



def _cpu_model_name() -> Optional[str]:
    """
    读取 CPU 型号，供跨机器复现时核对硬件差异。

    优先解析 Linux 的 /proc/cpuinfo；其他平台回退到
    platform.processor()/machine()。读取失败不影响实验。
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, UnicodeDecodeError):
        pass

    return platform.processor() or platform.machine() or None



# ============================================================
# Runtime environment snapshot / reproducibility check
# ============================================================

# environment.json：把「这次运行实际使用了哪套数值环境」变成
# 可机读、可比较、可归档的记录。config.json 里只记 Python/torch/
# NumPy 版本不足以复现浮点结果：cuBLAS/cuDNN/MKL/Triton 的版本、
# PyTorch 编译选项、CPU 指令集与线程配置都会改变归约顺序。这里
# 额外记录关键依赖的精确版本与运行时编译信息，并给出规范化
# environment_id；--check-environment 可在重跑/续跑/换机器前核对。
_ENVIRONMENT_SCHEMA = "smelt-isr/environment/v1"

_ENVIRONMENT_FILENAME = "environment.json"

_ENVIRONMENT_CONFIG_SCHEMA = "smelt-isr/run-config/v1"


# 「关键依赖」：直接或间接决定数值内核的包。完整包清单也落盘，
# 但 environment_id 只纳入这一子集，避免把 pip/setuptools 等与
# 数值无关、且常随环境漂移的包算进严格身份。
_ENVIRONMENT_CRITICAL_PREFIXES: Tuple[str, ...] = (
    "torch",
    "pytorch-",
    "numpy",
    "triton",
    "nvidia-",
    "cuda-",
    "cudnn",
    "nccl",
    "mkl",
    "intel-",
    "openmp",
)



def _canonical_distribution_name(name: object) -> Optional[str]:
    """PEP 503 规范化的发行包名；非法输入返回 None。"""
    if not isinstance(name, str):
        return None
    value = re.sub(r"[-_.]+", "-", name.strip()).strip("-").lower()
    return value or None



def _is_critical_distribution(name: str) -> bool:
    return any(
        name == prefix.rstrip("-")
        or name.startswith(prefix)
        for prefix in _ENVIRONMENT_CRITICAL_PREFIXES
    )



def _installed_distributions() -> List[Dict[str, str]]:
    """
    返回已安装发行包的规范名与版本（按名字排序）。

    环境记录不能因为某个包的元数据损坏而让训练失败：无法读取
    的条目跳过，其余仍进入快照。同一规范名出现多份 dist-info
    时取字典序最大的版本，保证结果稳定且不重复。
    """
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python >= 3.8
        return []

    try:
        distributions = list(metadata.distributions())
    except Exception:
        return []

    versions: Dict[str, set] = {}
    for dist in distributions:
        try:
            raw_name = (
                dist.metadata.get("Name")
                if dist.metadata is not None
                else None
            )
            raw_version = dist.version
        except Exception:
            continue
        name = _canonical_distribution_name(raw_name)
        if name is None or raw_version is None:
            continue
        version = str(raw_version).strip()
        if not version:
            continue
        versions.setdefault(name, set()).add(version)

    return [
        {
            "name": name,
            "version": sorted(versions[name])[-1],
        }
        for name in sorted(versions)
    ]



def _cpu_flags_digest() -> Dict[str, object]:
    """记录 CPU 指令集指纹（影响 PyTorch 选用的内核）。"""
    flags = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags"):
                    _, _, value = line.partition(":")
                    flags.update(value.split())
                    break
    except (OSError, UnicodeDecodeError):
        pass

    if not flags:
        return {
            "source": None,
            "flags_sha256": None,
            "flag_count": 0,
        }
    canonical = " ".join(sorted(flags))
    return {
        "source": "/proc/cpuinfo",
        "flags_sha256": _sha256_text(canonical),
        "flag_count": len(flags),
    }



def _torch_environment() -> Dict[str, object]:
    """PyTorch 版本、编译选项与数值后端。"""
    config_text = None
    parallel_info = None
    try:
        config_text = str(torch.__config__.show())
    except Exception:
        pass
    try:
        parallel_info = str(torch.__config__.parallel_info())
    except Exception:
        pass
    return {
        "version": str(torch.__version__),
        "git_version": getattr(torch.version, "git_version", None),
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "mkldnn": bool(torch.backends.mkldnn.is_available()),
        "config_sha256": (
            _sha256_text(config_text)
            if config_text is not None
            else None
        ),
        "parallel_info_sha256": (
            _sha256_text(parallel_info)
            if parallel_info is not None
            else None
        ),
    }



def _environment_device_info(
    device: Optional[torch.device],
) -> Dict[str, object]:
    """设备类型/型号/算力；不记录 UUID 等会随会话变化的标识。"""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif isinstance(device, str):
        try:
            device = torch.device(device)
        except (RuntimeError, ValueError):
            device = torch.device("cpu")

    info: Dict[str, object] = {
        "type": device.type,
        "index": device.index,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(device)
            info.update({
                "name": torch.cuda.get_device_name(device),
                "capability": (
                    f"{properties.major}.{properties.minor}"
                ),
                "total_memory_mb": round(
                    float(properties.total_memory) / (1024 ** 2),
                    1,
                ),
                "multi_processor_count": getattr(
                    properties, "multi_processor_count", None
                ),
            })
        except (AssertionError, RuntimeError, ValueError):
            pass
    elif device.type == "mps":
        info["name"] = "mps"
    else:
        info["name"] = _cpu_model_name()
    return info



@functools.lru_cache(maxsize=1)
def _nvidia_driver_version() -> Optional[str]:
    """
    NVIDIA 驱动版本（若可用）。

    CUDA runtime 版本相同而驱动不同，仍可能选择不同的内核实现。
    这里只读取版本号，不依赖 pynvml；命令不可用时返回 None，
    不把「无法读取」当成「没有差异」。
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=str(Path(__file__).resolve().parent),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return None



def _determinism_environment() -> Dict[str, object]:
    """实际生效的确定性/线程开关。"""
    return {
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allow_fp16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
            None,
        ),
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
            None,
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "pythonhashseed_at_start": _PYTHONHASHSEED_AT_START,
    }



def _canonical_json_sha256(value: object) -> str:
    return _sha256_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )



def _environment_identity_payload(
    snapshot: Dict[str, object],
) -> Dict[str, object]:
    """
    参与 environment_id 的字段：只保留决定数值结果的量。

    可执行文件路径、主机名、采集时间、设备 UUID 等不进入身份，
    否则同一套软件环境在同一台机器上换个虚拟环境路径就会被
    误判为不可复现。
    """
    python = snapshot.get("python")
    python = python if isinstance(python, dict) else {}
    platform_info = snapshot.get("platform")
    platform_info = (
        platform_info if isinstance(platform_info, dict) else {}
    )
    cpu = snapshot.get("cpu")
    cpu = cpu if isinstance(cpu, dict) else {}
    torch_info = snapshot.get("torch")
    torch_info = torch_info if isinstance(torch_info, dict) else {}
    device = snapshot.get("device")
    device = device if isinstance(device, dict) else {}
    determinism = snapshot.get("determinism")
    determinism = (
        determinism if isinstance(determinism, dict) else {}
    )
    driver = snapshot.get("driver")
    driver = driver if isinstance(driver, dict) else {}

    packages = snapshot.get("packages")
    critical: List[Dict[str, str]] = []
    if isinstance(packages, dict):
        raw_critical = packages.get("critical")
        if isinstance(raw_critical, list):
            critical = [
                item
                for item in raw_critical
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("version"), str)
            ]

    identity = {
        "python": {
            key: python.get(key)
            for key in (
                "implementation",
                "version",
                "compiler",
                "build",
            )
        },
        "platform": {
            key: platform_info.get(key)
            for key in (
                "system",
                "release",
                "machine",
                "libc",
            )
        },
        "cpu": {
            key: cpu.get(key)
            for key in ("model", "flags_sha256")
        },
        "torch": {
            key: torch_info.get(key)
            for key in (
                "version",
                "git_version",
                "cuda",
                "cudnn",
                "mkldnn",
                "config_sha256",
                "parallel_info_sha256",
            )
        },
        "device": {
            key: device.get(key)
            for key in ("type", "name", "capability")
        },
        "driver": {
            "nvidia": driver.get("nvidia"),
        },
        # PYTHONHASHSEED 是每次运行/每个种子显式设置的解释器
        # 启动参数，不是安装环境的一部分：套件子进程会按种子
        # 固定不同的值。若把它算进 environment_id，同一环境的
        # 多种子套件会被误报为“环境不同”。完整快照仍保留它，
        # 供逐运行核对随机化是否固定。
        "determinism": {
            key: value
            for key, value in determinism.items()
            if key != "pythonhashseed_at_start"
        },
        "packages": critical,
    }
    # 兼容只含少量字段的旧/手工快照：如果上面全是空值，退回
    # 对整份快照（去掉时间与身份字段）做指纹，避免所有无效
    # 快照共享同一个空对象哈希。注意不能直接
    # ``any(identity.values())``：identity 的每一项都是非空
    # dict（即使其中所有叶子都是 None），那样判断永远为真，
    # 回退分支永远不可达。
    def _has_identity_value(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                _has_identity_value(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                _has_identity_value(item) for item in value
            )
        if isinstance(value, (set, frozenset)):
            return any(
                _has_identity_value(item) for item in value
            )
        return value is not None and value != ""

    if not _has_identity_value(identity):
        return {
            key: value
            for key, value in snapshot.items()
            if key not in ("captured_utc", "environment_id")
        }
    return identity
