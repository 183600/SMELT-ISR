from __future__ import annotations



def load_tracking_provenance(
    out_dir: Path,
) -> Dict[str, object]:
    """
    读取运行目录的原始追踪身份。

    resume 会重写 config.json 的 run_metadata；final/test 记录若
    只写当前段的启动时间与代码指纹，会把多次续跑的历史压缩成
    最后一次。这里优先取 original_run_metadata（首次运行快照），
    并返回 resume_count 与受控变量协议指纹，供 metrics.jsonl
    脱离 config.json 归档时仍能定位原始运行。
    """
    path = out_dir / "config.json"
    empty: Dict[str, object] = {
        "run_started_utc": None,
        "original_script_sha256": None,
        "original_git_commit": None,
        "original_git_dirty": None,
        "original_environment_id": None,
        "resume_count": 0,
        "comparison_protocol_sha256": None,
    }
    if not _path_exists(path):
        return empty

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return empty

    if not isinstance(payload, dict):
        return empty

    original = payload.get("original_run_metadata")
    if not isinstance(original, dict):
        original = payload.get("run_metadata")
    if not isinstance(original, dict):
        original = {}

    resume_count = payload.get("resume_count")
    if (
        not isinstance(resume_count, int)
        or isinstance(resume_count, bool)
        or resume_count < 0
    ):
        resume_count = 0

    protocol_sha = payload.get("comparison_protocol_sha256")
    if not isinstance(protocol_sha, str):
        protocol_sha = None

    return {
        "run_started_utc": original.get("timestamp_utc"),
        "original_script_sha256": original.get("script_sha256"),
        "original_git_commit": original.get("git_commit"),
        "original_git_dirty": original.get("git_dirty"),
        "original_environment_id": original.get(
            "environment_id"
        ),
        "resume_count": resume_count,
        "comparison_protocol_sha256": protocol_sha,
    }



def tracking_header_fields(
    args: argparse.Namespace,
    dataset: ByteDataset,
    run_metadata: Dict[str, object],
    *,
    run_started_utc: Optional[str] = None,
    comparison_protocol_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """
    metrics.jsonl 首行的运行身份字段。

    config.json 是完整运行卡片，但 metrics.jsonl 经常被单独复制
    进图表/归档；首行带上种子、数据版本、代码版本与协议指纹，
    即使脱离 config.json 也能核对曲线属于哪一轮实验。
    """
    return {
        "seed": getattr(args, "seed", None),
        "data_sha256": getattr(dataset, "data_sha256", None),
        "data_split_sha256": dict(
            getattr(dataset, "split_sha256", {}) or {}
        ),
        "data_effective_train_sha256": getattr(
            dataset, "effective_train_sha256", None
        ),
        "script_sha256": run_metadata.get("script_sha256"),
        "environment_id": run_metadata.get("environment_id"),
        "git_commit": run_metadata.get("git_commit"),
        "git_dirty": run_metadata.get("git_dirty"),
        "pythonhashseed_at_start": run_metadata.get(
            "pythonhashseed_at_start"
        ),
        "rng_seed_manifest": run_metadata.get(
            "rng_seed_manifest"
        ),
        "deterministic_algorithms": run_metadata.get(
            "deterministic_algorithms"
        ),
        "run_started_utc": (
            run_started_utc
            or run_metadata.get("timestamp_utc")
        ),
        "comparison_protocol_sha256": (
            comparison_protocol_sha256
        ),
    }



def load_previous_sanity_result(
    out_dir: Path,
) -> Optional[Dict[str, object]]:
    """
    读取已有 config.json 中的 sanity check 结果。

    resume 会重写 config.json；保留原始运行的 sanity 结果，
    避免追踪信息在恢复训练后丢失（metrics.jsonl 首行也有记录）。
    """
    path = out_dir / "config.json"

    if not _path_exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    result = payload.get("sanity_check")
    return result if isinstance(result, dict) else None



# ============================================================
# Checkpoint
# ============================================================

def _snapshot_to_cpu(value: object) -> object:
    """
    递归把 state_dict 类负载中的张量拷成 CPU 快照。

    快照与训练设备解耦（值、dtype、键结构不变）。

    同一段底层存储常被多个键共享（如绑定权重
    ``lm_head.weight`` 与 ``token_embedding.weight``），快照
    按（存储地址、偏移、dtype、形状、步幅）去重：共享张量
    使用同一 CPU 副本，torch.save 仍可对存储去重。
    """
    memo: Dict[Tuple[object, ...], torch.Tensor] = {}

    def snapshot(item: object) -> object:
        if torch.is_tensor(item):
            key = (
                item.untyped_storage().data_ptr(),
                item.storage_offset(),
                item.dtype,
                tuple(item.size()),
                tuple(item.stride()),
            )
            cached = memo.get(key)
            if cached is None:
                cached = item.detach().to("cpu", copy=True)
                memo[key] = cached
            return cached
        if isinstance(item, dict):
            return {
                key: snapshot(entry)
                for key, entry in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [snapshot(entry) for entry in item]
        return item

    return snapshot(value)



def _nonfinite_named_tensors(
    value: object,
    prefix: str = "",
    limit: Optional[int] = None,
) -> List[str]:
    """
    递归列出负载中的非有限浮点/复数张量路径。

    整数张量（如 step、num_batches_tracked）天然有限，不参与
    检查；只扫描浮点/复数张量，避免把合法整数状态误判为污染。
    ``limit`` 限制返回条数，检查大模型时不必构造完整列表。
    """
    found: List[str] = []

    def visit(item: object, path: str) -> None:
        if limit is not None and len(found) >= limit:
            return
        if torch.is_tensor(item):
            if not (
                item.is_floating_point()
                or item.is_complex()
            ):
                return
            if not bool(torch.isfinite(item.detach()).all()):
                found.append(path or "<tensor>")
            return
        if isinstance(item, dict):
            for key, entry in item.items():
                visit(entry, f"{path}.{key}" if path else str(key))
            return
        if isinstance(item, (list, tuple)):
            for index, entry in enumerate(item):
                visit(entry, f"{path}[{index}]")

    visit(value, prefix)
    return found



def _nonfinite_checkpoint_parts(
    payload: Dict[str, object],
    limit: int = 5,
) -> Dict[str, List[str]]:
    """
    检查检查点中模型/优化器/缩放器状态的数值完整性。

    返回非空列表的部件名 → 前几个非有限张量路径。参数或优化器
    动量/方差已被 NaN/Inf 污染时，续跑只会把污染带进新轨迹；
    评估也会失去意义，因此这类检查点必须被明确拒绝，而不是
    靠后续 loss 偶然发现。
    """
    parts: Dict[str, List[str]] = {}

    for key in ("model", "optimizer", "scaler"):
        value = payload.get(key)
        if value is None:
            continue
        nonfinite = _nonfinite_named_tensors(
            value,
            limit=limit,
        )
        if nonfinite:
            parts[key] = nonfinite

    return parts



def _checkpoint_numerical_error(
    payload: Dict[str, object],
) -> Optional[str]:
    """把检查点数值污染转换为可直接展示的拒绝理由。"""
    parts = _nonfinite_checkpoint_parts(payload)
    if not parts:
        return None

    details = "; ".join(
        f"{name}: " + ", ".join(paths)
        for name, paths in parts.items()
    )
    return (
        "检查点包含 NaN/Inf（"
        + details
        + "）：受污染的参数/优化器状态不能恢复训练或用于"
        "最终评估；请从污染前的检查点恢复，并先处理数据、"
        "数值稳定性或学习率问题"
    )



def save_checkpoint(
    filename: str,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    train_generator_state: torch.Tensor,
    args: argparse.Namespace,
    numerical_state: Optional[Dict[str, object]] = None,
) -> bool:
    """
    保存断点检查点；模型/优化器/缩放器状态任何非有限值都会
    拒绝写入并返回 False。

    写入前做数值完整性检查：一次异常的参数更新或优化器状态
    溢出若被存进 last.pt，后续 --resume 会把污染当作正常起点，
    并使最终评估/测试集解封失去意义。拒绝时保留旧检查点不变。
    """
    # torch.compile 包装后 state_dict 带有 "_orig_mod." 前缀，
    # 存档时解包。
    raw_model = getattr(model, "_orig_mod", model)

    payload = {
        "config": asdict(raw_model.cfg),
        "model": _snapshot_to_cpu(raw_model.state_dict()),
        "optimizer": _snapshot_to_cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "scaler": (
            _snapshot_to_cpu(scaler.state_dict())
            if scaler is not None
            else None
        ),
        "step": step,
        "train_generator_state": train_generator_state,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        # NumPy 全局随机状态：只含基本类型，torch.load 在
        # weights_only=True 下也能安全读取。
        "numpy_rng_state": _numpy_rng_state_payload(),
        # 代码指纹：resume 时与当前脚本/commit 比对，跨代码
        # 版本续跑会明确警告并写入 config.json，避免把两次
        # 不同实现的训练轨迹冒充成同一实验。
        "script_sha256": _script_sha256(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "args": vars(args),
        # 数值事件与停止/污染标记随检查点存档：resume 不能把
        # 原运行的 nonfinite 历史清零，测试集解封也必须能看见
        # 「这次运行曾在数值上失败」。
        "numerical_state": numerical_state,
    }

    numerical_error = _checkpoint_numerical_error(payload)
    if numerical_error is not None:
        print(
            "[checkpoint] 拒绝写出受污染的检查点（保留旧文件）："
            f"{numerical_error}"
        )
        return False

    path = Path(filename)
    temporary = path.with_name(
        path.name + ".tmp"
    )

    try:
        # 先写完整临时文件再原子替换。
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return True



def load_checkpoint(
    filename: str,
    device: torch.device,
) -> dict:
    return torch.load(
        filename,
        map_location=device,
    )



# 验证集选中的权重存档名（与断点恢复用的 last.pt 分离：
# last.pt 是「继续训练」的完整状态，best.pt 是「最终评估/
# 测试」应使用的模型，两者语义不同，不能混用）。
_SELECTED_MODEL_FILENAME = "best.pt"



def save_selected_model(
    filename: str,
    model: LanguageModel,
    step: int,
    val_loss: float,
    args: argparse.Namespace,
    selection_metric: Optional[str] = None,
    selection_value: Optional[float] = None,
) -> bool:
    """
    保存「验证集选定」的模型权重（不含优化器状态）。

    每次周期验证刷新最优选型指标时覆盖写入；选型指标默认是
    val_loss，可用 ``--selection-metric`` 改为与预注册主指标
    一致的分类指标。最终评估与唯一一次测试集评估都在该模型上
    进行，保证测试集看到的是经验证集选择的模型，而不是训练
    结束时恰好排在上一步的参数。与 save_checkpoint 一样先写
    临时文件再原子替换。
    """
    raw_model = getattr(model, "_orig_mod", model)

    if selection_metric is None:
        selection_metric = getattr(
            args, "selection_metric", "val_loss"
        )
    if selection_value is None:
        selection_value = val_loss

    payload = {
        "config": asdict(raw_model.cfg),
        "model": _snapshot_to_cpu(raw_model.state_dict()),
        "step": step,
        "val_loss": (
            float(val_loss)
            if val_loss is not None
            else None
        ),
        "selection_metric": str(selection_metric),
        "selection_value": float(selection_value),
        "args": vars(args),
    }

    numerical_error = _checkpoint_numerical_error(payload)
    if numerical_error is not None:
        print(
            "[select] 拒绝把受污染的权重写入 best.pt"
            "（保留旧文件）：" + numerical_error
        )
        return False

    path = Path(filename)
    temporary = path.with_name(
        path.name + ".tmp"
    )

    try:
        # 先写完整临时文件再原子替换。
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return True



def load_selected_model(
    filename: str,
    device: torch.device,
) -> Optional[dict]:
    """
    读取 best.pt；缺失或损坏时返回 None（调用方回退到最后
    一步参数并明确提示）。
    """
    path = Path(filename)

    if not _path_exists(path):
        return None

    try:
        payload = torch.load(
            path,
            map_location=device,
        )
    except Exception:
        # best.pt 损坏时回退到最后一步参数，不能中断训练。
        return None

    if not isinstance(payload, dict) or "model" not in payload:
        return None
    if not isinstance(payload.get("model"), dict):
        return None
    step = payload.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
    ):
        return None

    return payload



def restore_selected_weights(
    model: LanguageModel,
    payload: dict,
) -> None:
    """把 best.pt 中的权重加载回模型（兼容 torch.compile 包装）。"""
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(payload["model"])

def run_self_test():
    """Run the independent self-check routines."""
    from selfcheck import run_self_test as _impl

    return _impl(globals())



# ============================================================
# Training
# ============================================================

def _restore_resume_cleaning_justification(
    args: argparse.Namespace,
    saved_args: Dict[str, object],
) -> bool:
    """
    resume 时恢复检查点里记录的清洗领域依据。

    清洗开关本身会按检查点恢复（否则训练样本集合会变），
    但 ``--domain-justification`` 若也必须重新手打，合法的
    续跑会被入口校验拒绝；或者用户随手写一句新依据，把原
    审计文本静默替换掉。显式命令行优先；否则按存档恢复。

    返回是否恢复了值。
    """
    if not isinstance(saved_args, dict):
        return False
    current = getattr(args, "domain_justification", None)
    if current is not None and str(current).strip():
        return False
    for key in (
        "domain_justification",
        "cleaning_domain_justification",
    ):
        saved = saved_args.get(key)
        if isinstance(saved, str) and saved.strip():
            args.domain_justification = saved
            return True
    return False



def hypothesis_check_error(
    args: argparse.Namespace,
) -> Optional[str]:
    """
    假设纪律检查：全新训练必须带明确的科学假设。

    返回错误消息（应交给 parser.error 报告）或 None。

    --compare（只读对比）、--sanity-only（结构检查）、
    --statistical-baseline（建立对照坐标系）、--plan-design
    （只生成设计计划、不训练）与 --resume（假设已记录在原
    运行 config.json）不受此约束；纯工程目的的临时运行可用
    --skip-hypothesis-check 显式跳过，跳过本身会随 config.json
    存档。
    """
    if (
        args.compare is None
        and getattr(args, "final_test_comparison", None) is None
        and not args.sanity_only
        and not args.statistical_baseline
        and not getattr(args, "plan_design", False)
        and args.resume is None
        and not args.hypothesis.strip()
        and not args.skip_hypothesis_check
    ):
        return (
            "全新训练必须通过 --hypothesis 说明本轮实验"
            "的科学假设（预期什么改动带来什么效果、依据"
            "什么指标判定成败），并可用 --rq 关联研究问题；"
            "例如：--hypothesis \"延迟合并让第二轮专家"
            "组合独立于第一轮，验证损失相对调优 dense 基线"
            "至少改善 2%\" --rq RQ1。"
            "带着假设做实验，而不是排列组合超参数。"
            "纯工程调试可用 --skip-hypothesis-check 跳过。"
        )

    return None



def build_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    start_step: int = 0,
    schedule: str = "cosine",
    stable_steps: Optional[int] = None,
    final_lr_ratio: float = 0.1,
):
    """构造学习率调度。

    schedule="cosine"（历史默认）：linear warmup 后余弦退火到
    峰值 lr 的 10%，与引入 WSD 之前的运行逐位一致。

    schedule="wsd"（论文 3.1 节 warmup-stable-decay）：
    warmup 后保持恒定，``stable_steps`` 之后按余弦衰减到
    ``final_lr_ratio``。``stable_steps=None`` 表示整段训练都
    处于稳定期（不衰减）；分支实验把 ``stable_steps`` 设为
    分支点，并用 ``--resume`` 从该检查点续跑。
    """
    if schedule not in ("cosine", "wsd"):
        raise ValueError(
            f"未知 lr schedule: {schedule!r}（cosine|wsd）"
        )
    # last_epoch >= 0 时 LRScheduler 要求 param group 中存在
    # initial_lr；该键缺失时用当前的基准 lr 补齐。
    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    if schedule == "cosine":
        def lr_lambda(step: int) -> float:
            # step 从 0 开始计数。
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)

            progress = (step - warmup_steps) / max(
                1,
                total_steps - warmup_steps,
            )
            progress = min(1.0, progress)

            # 余弦退火到原 lr 的 10%。
            return 0.1 + 0.9 * (
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
    else:
        stable_end = int(
            total_steps if stable_steps is None else stable_steps
        )
        stable_end = max(
            int(warmup_steps),
            min(stable_end, int(total_steps)),
        )
        decay_steps = max(1, int(total_steps) - stable_end)
        final_ratio = float(final_lr_ratio)
        if not math.isfinite(final_ratio) or not (
            0.0 <= final_ratio <= 1.0
        ):
            raise ValueError(
                "WSD final_lr_ratio 必须落在 [0, 1]"
            )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            # 稳定期：恒定峰值 lr。
            if step < stable_end:
                return 1.0
            progress = (
                step - stable_end
            ) / max(1, decay_steps)
            progress = min(1.0, progress)
            # 论文 3.1 节的分支是 cosine-decay。
            cosine = 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            return final_ratio + (
                1.0 - final_ratio
            ) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
        # 断点恢复时直接同步到 start_step。
        last_epoch=start_step - 1,
    )



def build_lr_schedule_from_args(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    start_step: int = 0,
):
    """按命令行/检查点恢复后的参数构造调度。"""

    return build_lr_schedule(
        optimizer,
        int(args.warmup_steps),
        int(args.steps),
        start_step=start_step,
        schedule=str(
            getattr(args, "lr_schedule", "cosine")
        ),
        stable_steps=getattr(
            args, "wsd_stable_steps", None
        ),
        final_lr_ratio=float(
            getattr(args, "wsd_final_lr_ratio", 0.1)
        ),
    )



# ============================================================
# Multi-seed / ablation sweeps
# ============================================================

# 多随机种子与消融套件由父进程逐个调用自身完成：每次运行都是
# 独立进程，避免跨运行的 RNG / CUDA 缓存 / 优化器状态耦合，
# 同时复用现有的断点、早停与追踪基础设施。
_SWEEP_SINGLE_VALUE_FLAGS = (
    "--out",
    "--seed",
    "--mode",
    "--balance-coef",
    "--run-order",
    "--run-order-seed",
)

_SWEEP_MULTI_VALUE_FLAGS = (
    "--seeds",
    "--suite-modes",
    "--design-pilot-runs",
    "--baseline-tuning-lrs",
    "--baseline-tuning-dense-hiddens",
    "--baseline-tuning-weight-decays",
    "--ablation-modes",
)

_SWEEP_BOOL_FLAGS = (
    "--suite-ablation",
    "--suite-baseline-tuning",
    "--allow-underpowered",
    "--allow-partial-ablation",
    "--plan-design",
)


# 基线调参协议：简单基线（默认 dense）先在 val 上做预注册的
# 学习率网格搜索，各候选训练配置（steps/batch/数据）完全一致，
# 只允许学习率这一个轴变化。调参报告落盘为
# <out>/baseline_tuning.json，后续基线训练用
# --baseline-tuning-from 把「这条基线的超参数来自那次调参」
# 写进 config.json，供基线合理性检查核对。
_BASELINE_TUNING_FILENAME = "baseline_tuning.json"

_DEFAULT_BASELINE_TUNING_LRS: Tuple[float, ...] = (
    1e-4,
    3e-4,
    1e-3,
)


# 调参来源核对所用的完整受控变量协议：学习率是调参因子本身，
# 不在核对范围内；其余协议项（warmup / 调度 / weight decay /
# 梯度裁剪 / 数据划分 / 群组划分 / 评估窗口等）必须与调参时
# 逐项一致。学习率与 warmup/调度/weight decay 耦合，只核对
# steps/batch/数据指纹不足以证明「受控变量」没被悄悄改掉。
_BASELINE_TUNING_PROTOCOL_KEYS: Tuple[str, ...] = tuple(
    key for key in COMPARISON_PROTOCOL_KEYS if key != "lr"
)


# 调参来源核对时无法在训练前知道、由训练结果注入的协议项：
# trained_steps / optimizer_steps 取决于早停与非有限事件，
# model_selection 取决于 best.pt 是否存在；它们属于结果而非
# 训练配置，不能要求最终基线在开跑前就与调参报告一致。
_BASELINE_TUNING_RUNTIME_KEYS: Tuple[str, ...] = (
    "trained_steps",
    "optimizer_steps",
    "model_selection",
)

_BASELINE_TUNING_PROVENANCE_KEYS: Tuple[str, ...] = tuple(
    key
    for key in _BASELINE_TUNING_PROTOCOL_KEYS
    if key not in _BASELINE_TUNING_RUNTIME_KEYS
)



def strip_sweep_args(argv: List[str]) -> List[str]:
    """
    从子命令参数中移除由套件逐运行覆盖的参数
    （--out/--seed/--mode/--balance-coef 与套件开关本身），
    其余参数原样透传，保证各次运行的受控设置完全一致。
    """
    result: List[str] = []
    index = 0

    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]

        if name in _SWEEP_BOOL_FLAGS:
            index += 1
            continue

        if name in _SWEEP_MULTI_VALUE_FLAGS:
            index += 1
            while (
                index < len(argv)
                and not argv[index].startswith("--")
            ):
                index += 1
            continue

        if name in _SWEEP_SINGLE_VALUE_FLAGS:
            index += 1 if "=" in token else 2
            continue

        result.append(token)
        index += 1

    return result



def build_sweep_plan(
    seeds: List[int],
    modes: Optional[List[str]],
    ablation: bool,
    balance_coef: float,
    default_mode: Optional[str] = None,
    ablation_modes: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    构造运行计划。每个条目包含输出子目录名与逐运行覆盖的
    参数；其余设置全部继承同一条命令行。

    - 普通多种子：seed{S}
    - 模式套件：{mode}_seed{S}
    - (1+N)×2 消融：{mode}_bal{coef}_seed{S}
      （{local_base} ∪ 延迟合并方案 × {0, balance_coef}）
      延迟合并方案默认覆盖全部已提出的实现 A 与 B；每个创新
      点都必须证明各自有用，只跑 A 的组合不足以支撑 B 的结论。
    """
    plan: List[Dict[str, object]] = []

    if ablation:
        delayed_modes = (
            # 防御性去重：重复方案会让同一消融单元被运行两次、
            # 同一输出目录被覆盖或重复登记。
            list(dict.fromkeys(ablation_modes))
            if ablation_modes
            else list(_DELAYED_MERGE_MODES)
        )
        for mode in ("local_base", *delayed_modes):
            for coefficient in (0.0, balance_coef):
                label_coefficient = format(coefficient, "g")
                for seed in seeds:
                    plan.append({
                        "label": (
                            f"{mode}_bal{label_coefficient}"
                            f"_seed{seed}"
                        ),
                        "mode": mode,
                        "balance_coef": coefficient,
                        "seed": seed,
                    })
        return plan

    if modes:
        for mode in modes:
            for seed in seeds:
                plan.append({
                    "label": f"{mode}_seed{seed}",
                    "mode": mode,
                    "balance_coef": balance_coef,
                    "seed": seed,
                })
        return plan

    for seed in seeds:
        plan.append({
            "label": f"seed{seed}",
            "mode": default_mode,
            "balance_coef": balance_coef,
            "seed": seed,
        })

    return plan



def build_baseline_tuning_plan(
    seeds: List[int],
    mode: str,
    lrs: List[float],
    balance_coef: float,
    dense_hiddens: Optional[List[int]] = None,
    weight_decays: Optional[List[float]] = None,
    base_lr: Optional[float] = None,
    base_dense_hidden: Optional[int] = None,
    base_weight_decay: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    基线调参网格的运行计划。

    单轴（默认）时是「学习率 × 随机种子」：候选之间只允许学习
    率不同，其余（steps/batch/数据/模型结构）全部继承同一条命令
    行，保证调参比较是在固定训练配置下做的「超参数搜索」，而不
    是偷跑更多步。

    传入 dense_hiddens / weight_decays 时启用**坐标式**（每候选
    相对基线只改一个轴）多轴设计：每个轴的候选都相对同一条基
    线配置只改该轴一个超参，其余轴固定在基线值。这样每个轴的
    效应都能单独归因，不会与「顺手换了别的超参」混杂；多出来的
    代价是运行数按轴相乘（坐标式设计本身就是这个代价）。

    基线配置始终作为共同参照点：多轴设计若没有基线运行，就无法
    回答「改这个轴到底有没有变好」——基线配置已由某个轴的候选覆
    盖时复用该运行，不额外重复训练。
    """
    plan: List[Dict[str, object]] = []
    labels = [f"lr{format(lr, 'g')}" for lr in lrs]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "基线调参候选的学习率在运行目录名中格式化后重名，"
            "会互相覆盖输出目录；请使用差异更大的候选"
        )
    for name, values in (
        ("h", dense_hiddens or []),
        ("wd", weight_decays or []),
    ):
        formatted = [
            f"{name}{format(value, 'g')}" for value in values
        ]
        if len(set(formatted)) != len(formatted):
            raise ValueError(
                f"基线调参候选的 {name} 在运行目录名中格式化后"
                "重名，会互相覆盖输出目录；请使用差异更大的候选"
            )

    axis_specs: List[Dict[str, object]] = [
        {
            "name": "lr",
            "protocol_key": "lr",
            "label": "学习率",
            "values": [float(lr) for lr in lrs],
        },
    ]
    if dense_hiddens:
        axis_specs.append({
            "name": "dense_fnn_hidden",
            "protocol_key": "dense_fnn_hidden",
            "label": "dense FFN 隐藏宽度",
            "values": [int(value) for value in dense_hiddens],
        })
    if weight_decays:
        axis_specs.append({
            "name": "weight_decay",
            "protocol_key": "weight_decay",
            "label": "权重衰减",
            "values": [float(value) for value in weight_decays],
        })

    base_values: Dict[str, object] = {
        "lr": None if base_lr is None else float(base_lr),
        "dense_fnn_hidden": (
            None
            if base_dense_hidden is None
            else int(base_dense_hidden)
        ),
        "weight_decay": (
            None
            if base_weight_decay is None
            else float(base_weight_decay)
        ),
    }

    def _signature(config: Dict[str, object]) -> Tuple[str, ...]:
        return tuple(
            f"{key}={_protocol_value_text(value)}"
            for key, value in sorted(config.items())
        )

    def _label_for(axis: str, value: object, seed: int) -> str:
        if axis == "base":
            return f"base_seed{seed}"
        if axis == "lr":
            return f"lr{format(float(value), 'g')}_seed{seed}"
        if axis == "dense_fnn_hidden":
            return f"h{int(value)}_seed{seed}"
        return f"wd{format(float(value), 'g')}_seed{seed}"

    # 每个轴的候选条目（相对基线只改该轴）；同一配置只保留一条，
    # 避免同一个运行被重复训练。
    seen_signatures = set()
    axis_entries: List[Dict[str, object]] = []
    for spec in axis_specs:
        axis = str(spec["name"])
        for value in spec["values"]:
            config = dict(base_values)
            config[axis] = value
            signature = _signature(config)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            axis_entries.append({
                "axis": axis,
                "value": value,
                "config": config,
            })

    # 基线配置：多轴设计必须有共同参照点（基线运行即该参照）。
    if base_lr is not None:
        base_signature = _signature(base_values)
        if base_signature not in seen_signatures:
            axis_entries.append({
                "axis": "base",
                "value": None,
                "config": dict(base_values),
            })

    for entry in axis_entries:
        axis = str(entry["axis"])
        value = entry["value"]
        config = entry["config"]
        for seed in seeds:
            item: Dict[str, object] = {
                "label": _label_for(axis, value, seed),
                "mode": mode,
                "balance_coef": balance_coef,
                "seed": seed,
                # 调参轴与取值随计划落盘：分析阶段按轴分组时，
                # 不依赖运行目录名的解析。
                "tuning_axis": axis,
                "tuning_value": value,
            }
            if config.get("lr") is not None:
                item["lr"] = config["lr"]
            if config.get("dense_fnn_hidden") is not None:
                item["dense_fnn_hidden"] = config[
                    "dense_fnn_hidden"
                ]
            if config.get("weight_decay") is not None:
                item["weight_decay"] = config["weight_decay"]
            plan.append(item)
    return plan



def resolve_run_order(args: argparse.Namespace) -> Tuple[str, int]:
    """
    解析预注册/命令行声明的运行顺序设计。

    默认 ``randomized``：每个种子是一个完整区组，区组内的臂
    顺序随机化。``counterbalanced`` 在同一区组结构上进一步用
    循环拉丁方约束位置计数，适合种子数较少、担心热漂移与
    区组内位置混杂的确证运行。旧行为 ``grouped`` 会先把一个
    臂的所有种子跑完再跑下一个臂，机器热漂移、共享缓存与
    集群负载随时间变化时，这些漂移与臂效应完全混杂。
    """
    scheme = getattr(args, "run_order", None)
    if scheme is None:
        scheme = "randomized"
    if scheme not in _RUN_ORDER_SCHEMES:
        raise ValueError(
            "运行顺序方案必须是 "
            + " / ".join(_RUN_ORDER_SCHEMES)
        )
    order_seed = getattr(args, "run_order_seed", None)
    if order_seed is None:
        order_seed = 0
    if (
        isinstance(order_seed, bool)
        or not isinstance(order_seed, int)
        or order_seed < 0
        or order_seed > MAX_SEED
    ):
        raise ValueError(
            f"运行顺序种子必须是 [0, {MAX_SEED}] 内的整数"
        )
    return str(scheme), int(order_seed)



def _order_block_rng(
    order_seed: int,
    block_seed: int,
) -> random.Random:
    """
    由 (order_seed, block_seed) 派生确定性的区组洗牌 RNG。

    不使用 Python 内置 hash()：它受 PYTHONHASHSEED 影响，会让
    「同一个设计计划」在不同进程/机器上得到不同顺序，破坏
    预注册顺序的可复现性。
    """
    digest = hashlib.sha256(
        f"smelt-isr-run-order:{int(order_seed)}:{int(block_seed)}".encode(
            "utf-8"
        )
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))



def _order_block_seed(value: object) -> int:
    """区组 RNG 的整数种子；损坏计划里的非整数 seed 也不能崩。"""
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    # 非整数 seed（手工编辑/旧格式损坏）没有 int() 可用的值；
    # 用稳定键派生确定性种子，既避免 TypeError，也保证同一
    # 计划重复生成时顺序不变。
    return int.from_bytes(
        hashlib.sha256(
            repr(_stable_factor_key(value)).encode("utf-8")
        ).digest()[:8],
        "big",
    )



def _order_block_key(value: object) -> object:
    """区组分组的哈希键；list/dict 等不可哈希 seed 用稳定包装。"""
    try:
        hash(value)
    except TypeError:
        return ("__unhashable_block__", repr(_stable_factor_key(value)))
    return value



def _order_block_sort_key(value: object):
    """区组的确定性全序键，避免 None/str/数值混合比较崩溃。"""
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value, "")
    if isinstance(value, float) and math.isfinite(value):
        return (0, value, "")
    # 其余（字符串、None、不可哈希包装）统一按稳定表示的文本
    # 排序；这不会改变正常整数种子的区组顺序。
    return (1, repr(_stable_factor_key(value)), "")



def _counterbalance_rng(order_seed: int) -> random.Random:
    """平衡区组排列的确定性 RNG（与区组洗牌分开命名空间）。"""
    digest = hashlib.sha256(
        f"smelt-isr-counterbalance:{int(order_seed)}".encode(
            "utf-8"
        )
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))



_PLAN_IDENTITY_KEYS = frozenset({
    "label",
    "seed",
    "order_index",
    "block",
    "position_in_block",
})



def _plan_treatment_key(item: Dict[str, object]):
    """
    处理单元的身份：去掉运行标签/种子/顺序等非处理轴后的
    稳定表示。counterbalanced 用它判断「每个区组是否包含同一
    组处理」，balance 报告用它统计每个处理的位置分布。
    """
    return tuple(
        (str(key), _stable_factor_key(item.get(key)))
        for key in sorted(item, key=str)
        if str(key) not in _PLAN_IDENTITY_KEYS
    )



def _counterbalanced_order(
    blocks: Dict[object, List[Dict[str, object]]],
    order_seed: int,
) -> List[Dict[str, object]]:
    """
    构造平衡的完整区组顺序（循环拉丁方）。

    每个种子区组必须包含同一组处理单元各一次。先把处理标签
    用 order_seed 确定性随机化，再让第 b 个区组使用基础序列的
    第 (start + b) 个循环移位：任意连续 b 行中，每个处理在
    每个位置上最多出现一次，因此位置计数之差不超过 1。相比
    纯随机化，这避免了小样本下某个臂偶然总在同一位置、与
    机器热漂移/缓存预热混杂。
    """
    if not blocks:
        return []
    block_keys = sorted(
        blocks, key=_order_block_sort_key
    )
    reference: Optional[List] = None
    keyed_blocks: Dict[object, Dict[object, Dict[str, object]]] = {}
    for block_key in block_keys:
        entries = blocks[block_key]
        keys = [_plan_treatment_key(item) for item in entries]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "counterbalanced 顺序要求同一区组内的处理单元"
                "互不相同；当前区组 "
                f"{block_key!r} 有重复处理单元"
            )
        block_reference = sorted(set(keys), key=repr)
        if reference is None:
            reference = block_reference
        elif block_reference != reference:
            raise ValueError(
                "counterbalanced 顺序要求每个种子区组包含完全"
                "相同的处理单元集合；当前区组 "
                f"{block_key!r} 与其它区组不同。请检查套件是否"
                "有只在部分种子上运行的候选；这种情况不能声称"
                "位置已平衡，应改用 randomized 或补齐运行。"
            )
        keyed_blocks[block_key] = {
            key: item for key, item in zip(keys, entries)
        }
    if not reference:
        # 空区组（损坏/手工计划）没有可排列的处理单元：返回空
        # 顺序，而不是在 len(base) 取模时抛 ZeroDivisionError。
        return []

    base = list(reference)
    rng = _counterbalance_rng(order_seed)
    rng.shuffle(base)
    start = rng.randrange(len(base)) if base else 0

    ordered: List[Dict[str, object]] = []
    for block_index, block_key in enumerate(block_keys):
        rotation = (start + block_index) % len(base)
        row = base[rotation:] + base[:rotation]
        mapping = keyed_blocks[block_key]
        ordered.extend(dict(mapping[key]) for key in row)
    return ordered
