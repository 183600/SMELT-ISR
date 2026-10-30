from __future__ import annotations



def _paper_budget_audit(
    cfg: Config,
    cell: PaperCell,
) -> Dict[str, object]:
    """把实测的 FLOPs / 总参数 / KV cache / S 与论文目标对照。

    论文附录 A 的参数化来自未公开的 Transformer family；本仓库
    实现（SwiGLU 专家、RMSNorm、学习式位置编码）只能保证公开
    披露量的结构一致，不能保证逐位相同的参数/FLOPs。该审计
    如实报告偏差，不据此伪装成数值复现。
    """

    counts = _analytic_param_counts(cfg)
    flops = int(
        mode_cost_profile(cfg)["analytic_flops_per_token"]
    )
    base_cfg = Config(
        **paper_preset(cell.scale, cell.sparsity, "base")["config"]
    )
    base_counts = _analytic_param_counts(base_cfg)
    base_flops = int(
        mode_cost_profile(base_cfg)["analytic_flops_per_token"]
    )
    control_cfg = Config(
        **paper_preset(cell.scale, 0, "base")["config"]
    )
    control_counts = _analytic_param_counts(control_cfg)
    control_flops = int(
        mode_cost_profile(control_cfg)["analytic_flops_per_token"]
    )
    flops_ratio = flops / base_flops
    kv_ratio = counts["kv_dim"] / base_counts["kv_dim"]
    sparsity = 1.0 - (
        flops / control_flops
    ) * (control_counts["total"] / counts["total"])
    total_ratio = counts["total"] / (cell.total_params * 1e9)
    active_ratio = counts["active"] / (
        cell.active_params * 1e9
    )
    within: Dict[str, Optional[bool]] = {
        "total_params_within_1pct": bool(
            abs(total_ratio - 1.0) <= 0.01
        ),
        "active_params_within_5pct": bool(
            abs(active_ratio - 1.0) <= 0.05
        ),
        "sparsity_within_2pp": bool(
            abs(sparsity - cell.measured_sparsity) <= 0.02
        ),
        "flops_ratio_within_4pct": (
            None
            if cell.flops_ratio is None
            else bool(abs(flops_ratio - cell.flops_ratio) <= 0.04)
        ),
        "kv_ratio_within_4pct": (
            None
            if cell.kv_ratio is None
            else bool(abs(kv_ratio - cell.kv_ratio) <= 0.04)
        ),
    }
    return {
        "measured": {
            "total_params": int(counts["total"]),
            "active_params": int(counts["active"]),
            "flops_per_token": int(flops),
            "flops_ratio_vs_base": float(flops_ratio),
            "kv_ratio_vs_base": float(kv_ratio),
            "compute_equivalent_sparsity": float(sparsity),
            "total_params_ratio_vs_paper": float(total_ratio),
            "active_params_ratio_vs_paper": float(active_ratio),
        },
        "paper": {
            "total_params": float(cell.total_params),
            "active_params": float(cell.active_params),
            "flops_ratio": cell.flops_ratio,
            "kv_ratio": cell.kv_ratio,
            "compute_equivalent_sparsity": cell.measured_sparsity,
        },
        "tolerances": {
            "total_params_relative": 0.01,
            "active_params_relative": 0.05,
            "flops_ratio_absolute": 0.04,
            "kv_ratio_absolute": 0.04,
            "sparsity_absolute": 0.02,
        },
        "within_paper_tolerance": within,
        "all_within_tolerance": all(
            value is not False for value in within.values()
        ),
        "note": (
            "论文附录 A 的参数化未公开；本实现的参数统计口径为"
            "「非 embedding 参数」，total/active 与 FLOPs 由"
            "本仓库结构精确计算，S 按论文 Eq.1 由实测 FLOPs "
            "与总参数推出。偏差不是实现缺陷的充分证据，也不"
            "能把偏差忽略后声称数值复现"
        ),
    }



def _paper_protocol_checks(
    protocol: Dict[str, object],
) -> Dict[str, bool]:
    saved_steps = list(
        protocol.get("save_at_steps") or []
    )
    stable = int(protocol.get("stable_steps", 0))
    steps = int(protocol.get("steps", stable))
    stable_target = int(PAPER_PROTOCOL["stable_steps"])
    decay_steps = int(PAPER_PROTOCOL["decay_steps"])
    branch_points = {
        int(step) for step in PAPER_PROTOCOL["branch_steps"]
    }
    # 稳定期跑满 196075 步；或者从六个分支点之一分叉、
    # 再跑满 10B token（9537 步）的 cosine-decay 分支。
    # 196075 本身也是合法分支点（从完整稳定期末端进入 decay），
    # 不能把这条分支误判成「缺少稳定期」的稳定训练。
    branch_run = (
        stable in branch_points
        and steps == stable + decay_steps
    )
    checks = {
        "protocol_wsd_schedule": (
            str(protocol.get("lr_schedule", "")) == "wsd"
        ),
        "protocol_batch_256": (
            int(protocol.get("batch_size", 0)) == 256
        ),
        "protocol_stable_phase": (
            stable == stable_target and steps == stable_target
        ),
        "protocol_decay_branch": bool(
            stable == stable_target or branch_run
        ),
        "protocol_document_masks": bool(
            protocol.get("document_masks")
        ),
        "protocol_branch_checkpoints": bool(
            branch_points.issubset(
                {int(step) for step in saved_steps}
            )
            or branch_run
        ),
    }
    if branch_run:
        # 分支运行本身不包含稳定期；稳定期由它的父运行负责，
        # 不能把「这条分支没跑稳定期」记成协议偏离。
        checks.pop("protocol_stable_phase")
    return checks



def paper_fidelity_report(
    cfg: Config,
    protocol: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """对照 arXiv:2609.01343v1 审计当前配置的复现保真度。

    复现分为两层：

    1. **recipe（架构 + 协议）**：论文正文/附录 A 明确披露的量，
       包括 H/L/专家数/top-8、middle-50% 循环两次的层位置、循环
       区域子层残差 1/r、GQA、4096 上下文，以及 3.1 节的 WSD
       协议、全局 batch 256、稳定期 196 075 步、六个分支点与
       文档级 segment attention mask。``architecture_faithful``
       与 ``protocol_faithful`` 分别给出结论，``is_full_reproduction``
       仅在两者都成立时为真。
    2. **budget（预算）**：论文 3.2 节要求 FLOPs / 总参数 / KV
       cache 三者匹配，但附录 A 的参数化来自未公开 family，
       本实现只能报告实测偏差（``budget_audit``），不把偏差
       忽略后声称数值复现。

    未被论文披露的学习率、warmup、weight decay、betas、均衡
    损失系数、初始化、head 几何、专家中间维与语料列在
    ``undisclosed`` 中；沿用本仓库默认值不等于论文原值。
    """

    cell = paper_cell_for_config(cfg)
    arch = cfg.mode if cfg.mode in PAPER_ARCHS else None
    loop_span = int(cfg.middle_end - cfg.middle_start)
    expected_start = int(cfg.n_layers - loop_span) // 2
    checks: Dict[str, bool] = {
        "paper_cell_config": cell is not None,
        "top_k_8": int(cfg.top_k) == 8,
        "gqa": (
            cfg.n_kv_heads is not None
            and int(cfg.effective_n_kv_heads) < int(cfg.n_heads)
        ),
        "context_4096": int(cfg.seq_len) == int(
            PAPER_PROTOCOL["context_length"]
        ),
        "loop_span_middle_half": (
            loop_span == int(cfg.n_layers) // 2
        ),
        "loop_placement_paper": (
            int(cfg.middle_start) == expected_start
            and loop_span == int(cfg.n_layers) // 2
        ),
    }
    if cfg.mode == "smelt":
        checks["loop_twice"] = cfg.loop_count == 2
        checks["loop_residual_scaling_1_over_r"] = (
            cfg.residual_scaling == "paper"
        )
    elif cfg.mode == "base":
        checks["unlooped_baseline"] = cfg.loop_count == 1
    else:
        checks["paper_arch_mode"] = False

    preset_geometry: Optional[Dict[str, object]] = None
    if cell is not None:
        preset = paper_preset(
            cell.scale,
            cell.sparsity,
            cell.arch,
        )
        derived = preset["derived"]
        preset_geometry = {
            "n_heads": derived["n_heads"],
            "n_kv_heads": derived["n_kv_heads"],
            "expert_hidden": derived["expert_hidden"],
        }
        checks["preset_geometry_matches"] = bool(
            int(cfg.n_heads) == int(derived["n_heads"])
            and int(cfg.effective_n_kv_heads)
            == int(derived["n_kv_heads"])
            and int(cfg.expert_hidden)
            == int(derived["expert_hidden"])
        )

    protocol_checks: Dict[str, bool] = {}
    if protocol is not None:
        protocol_checks = _paper_protocol_checks(protocol)

    architecture_faithful = all(checks.values())
    protocol_faithful: Optional[bool] = (
        all(protocol_checks.values())
        if protocol is not None
        else None
    )
    is_full_reproduction = bool(
        architecture_faithful and protocol_faithful is True
    )
    deviations = [
        name for name, ok in checks.items() if not ok
    ] + [
        name for name, ok in protocol_checks.items() if not ok
    ]

    budget_audit: Optional[Dict[str, object]] = None
    budget_all_within: Optional[bool] = None
    if cell is not None and arch is not None:
        budget_audit = _paper_budget_audit(cfg, cell)
        budget_all_within = bool(
            budget_audit["all_within_tolerance"]
        )
    # 「recipe 复现」与「数值复现」分开：recipe 指附录 A 的结构
    # 与 3.1 节协议；数值复现还要求 FLOPs/参数/KV/S 都落在论文
    # 容差内，而附录 A 的参数化来自未公开 family，公开实现通常
    # 达不到。两个字段都不允许靠措辞合并。
    numerical_reproduction = bool(
        is_full_reproduction and budget_all_within
    )

    if numerical_reproduction:
        interpretation = (
            "recipe（架构+协议）与论文附录 A 的预算容差均对齐；"
            "未被论文披露的超参与语料仍不等价于原实现，结论"
            "应写成复现而非逐位相同"
        )
    elif is_full_reproduction:
        interpretation = (
            "recipe（附录 A 结构 + 3.1 节协议）复现完成；"
            "budget_audit 仍有偏差，numerical_reproduction=false，"
            "不能声称数值复现（偏差来源与推导规则见 "
            "docs/paper_reproduction.md）"
        )
    elif architecture_faithful and protocol is None:
        interpretation = (
            "架构对齐论文附录 A 的公开披露项，但训练协议未按"
            "3.1 节设置；不能称为完整复现"
        )
    elif cell is None:
        interpretation = (
            "当前配置不是论文附录 A 的任何网格单元；只能表述为"
            "「SMELT 风格架构」的原型实验，不能声称复现论文"
        )
    else:
        interpretation = (
            "存在偏离项："
            + ", ".join(deviations)
            + "；不能声称复现论文 SMELT"
        )

    return {
        "paper": PAPER_ID,
        "paper_url": PAPER_URL,
        "cell": (
            None
            if cell is None
            else {
                "arch": cell.arch,
                "scale": cell.scale,
                "sparsity": cell.sparsity,
                "hidden": cell.hidden,
                "layers": cell.layers,
                "total_experts": cell.total_experts,
                "active_experts": cell.active_experts,
            }
        ),
        "preset_geometry": preset_geometry,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "deviations": deviations,
        "architecture_faithful": bool(architecture_faithful),
        "protocol_faithful": protocol_faithful,
        "is_full_reproduction": is_full_reproduction,
        "numerical_reproduction": numerical_reproduction,
        "budget_audit": budget_audit,
        "undisclosed": list(PAPER_UNDISCLOSED),
        "protocol": dict(PAPER_PROTOCOL),
        "interpretation": interpretation,
        "scale_note": (
            "论文标度 100M–1.6B、4096 上下文、top-8 + GQA；"
            "本仓库默认是小规模原型，规模外推需要另行验证"
        ),
    }



def _preset_summary(preset: Dict[str, object]) -> str:
    cell = preset["cell"]
    derived = preset["derived"]
    measured = preset["measured"]
    return (
        f"{cell.arch} {cell.scale} S≈{cell.sparsity}% | "
        f"H={cell.hidden} L={cell.layers} "
        f"E={cell.total_experts} top-{cell.active_experts} | "
        f"heads={derived['n_heads']}/"
        f"{derived['n_kv_heads']} "
        f"expert_hidden={derived['expert_hidden']} | "
        f"total={measured['total_params']/1e9:.3f}B "
        f"(paper {cell.total_params:.3f}B) "
        f"active={measured['active_params']/1e9:.3f}B "
        f"(paper {cell.active_params:.3f}B)"
    )



def _argv_has_flag(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(flag + "="):
            return True
    return False



def apply_paper_cli(args, parser=None) -> None:
    """把 ``--paper-cell`` / ``--paper-protocol`` 落到 args 上。

    - 预设覆盖模型结构参数；若命令行显式给了冲突值则报错，
      避免「看起来按论文配置、实际被命令行改动」。
    - ``--paper-protocol`` 只在用户未显式给出对应命令行参数时
      套用 3.1 节的协议值，方便小规模冒烟测试但仍记录偏离。
    """

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    preset_report: Optional[Dict[str, object]] = None
    if getattr(args, "paper_cell", None):
        try:
            scale, sparsity = parse_paper_cell_label(
                args.paper_cell
            )
        except ValueError as error:
            fail(str(error))
            return
        arch = getattr(args, "paper_arch", None)
        if arch is None:
            arch = (
                args.mode
                if args.mode in PAPER_ARCHS
                else "smelt"
            )
        if arch not in PAPER_ARCHS:
            fail(
                "--paper-arch 必须是 base 或 smelt"
            )
        if (
            args.mode in PAPER_ARCHS
            and args.mode != arch
        ):
            fail(
                f"--mode {args.mode} 与 --paper-arch {arch} 冲突"
            )
        if args.mode not in PAPER_ARCHS and args.mode != "B":
            fail(
                "--paper-cell 只支持 base / smelt 两种论文架构"
            )
        if (
            args.mode == "B"
            and _argv_has_flag("--mode")
        ):
            fail(
                "--paper-cell 与显式 --mode B（延迟合并）冲突；"
                "论文网格只有 base / smelt"
            )
        try:
            preset = paper_preset(scale, sparsity, arch)
        except ValueError as error:
            fail(str(error))
            return
        conflict_keys = {
            "seq_len": "--seq-len",
            "d_model": "--d-model",
            "n_heads": "--n-heads",
            "n_kv_heads": "--n-kv-heads",
            "n_layers": "--n-layers",
            "n_experts": "--n-experts",
            "expert_hidden": "--expert-hidden",
            "top_k": "--top-k",
            "k1": "--k1",
            "k2": "--k2",
            # 论文网格的循环区域残差 1/r 是架构语义的一部分；
            # 显式 --residual-scaling global 不能被预设静默覆盖。
            "residual_scaling": "--residual-scaling",
            # residual_scale 会在 1/r 之外再乘一次；显式改成
            # 0.5 会让 loop_residual_scale 变成 0.25，必须拒绝。
            "residual_scale": "--residual-scale",
        }
        conflicts = []
        for key, flag in conflict_keys.items():
            if not _argv_has_flag(flag):
                continue
            current = getattr(args, key, None)
            expected = preset["config"][key]
            if current != expected:
                conflicts.append(
                    f"{flag}={current!r}（预设要求 {expected!r}）"
                )
        if conflicts:
            fail(
                "--paper-cell 的结构参数由论文附录 A 决定，"
                "不能同时覆盖：" + "; ".join(conflicts)
            )
        for key, value in preset["config"].items():
            setattr(args, key, value)
        args.mode = arch
        args.paper_cell = f"{scale}:{sparsity}"
        args.paper_arch = arch
        # 只保留 JSON 可序列化的字段（PaperCell dataclass 不能
        # 直接写进 config.json / 检查点元数据）。
        preset_report = {
            "paper": preset["paper"],
            "arch": preset["arch"],
            "scale": preset["scale"],
            "sparsity": preset["sparsity"],
            "config": dict(preset["config"]),
            "derived": dict(preset["derived"]),
            "measured": dict(preset["measured"]),
            "targets": dict(preset["targets"]),
        }
        args.paper_preset = preset_report
        print(f"[paper] {_preset_summary(preset)}")
        if args.seq_len != int(PAPER_PROTOCOL["context_length"]):
            print(
                "[paper][warn] 上下文长度不是论文的 "
                f"{PAPER_PROTOCOL['context_length']}；"
                "paper_fidelity 会记为偏离"
            )

    if getattr(args, "paper_protocol", False):
        protocol_defaults = {
            "seq_len": int(PAPER_PROTOCOL["context_length"]),
            "batch_size": int(
                PAPER_PROTOCOL["global_batch_sequences"]
            ),
            "steps": int(PAPER_PROTOCOL["stable_steps"]),
            # warmup / 评估间隔论文未披露（PAPER_UNDISCLOSED），
            # 不能借 --paper-protocol 静默改成任意常数；继续
            # 使用本仓库的命令行默认值，与文档和 fidelity 报告
            # 的「未披露项沿用默认」保持一致。
            "lr_schedule": "wsd",
            "save_at_steps": list(
                PAPER_PROTOCOL["branch_steps"]
            ),
            "document_masks": True,
        }
        for key, value in protocol_defaults.items():
            flag = "--" + key.replace("_", "-")
            if _argv_has_flag(flag):
                continue
            setattr(args, key, value)
        if getattr(args, "wsd_stable_steps", None) is None:
            args.wsd_stable_steps = int(args.steps)
        if not getattr(args, "group_separator", None):
            if not _argv_has_flag("--document-masks"):
                # 没有文档边界时无法构造 segment mask；不阻断
                # 运行，但把 protocol_document_masks 记为偏离，
                # 由 paper_fidelity 审计如实报告。
                args.document_masks = False
            print(
                "[paper][warn] 未提供 --group-separator：无法构造"
                "文档级 segment attention mask，论文 3.1 节协议"
                "不完整（protocol_document_masks=false）；"
                "忠实复现请用 --group-separator 标注文档边界"
            )
        print(
            "[paper][protocol] WSD + batch "
            f"{args.batch_size} + stable {args.wsd_stable_steps} "
            f"steps + ctx {args.seq_len} + document masks；"
            "学习率/warmup/weight decay 论文未披露，沿用本仓库"
            "默认值"
        )

    args.paper_preset_report = preset_report



# ============================================================
# Reproducibility
# ============================================================

# NumPy 的 legacy RandomState 只接受 [0, 2**32 - 1]。把 --seed
# 的合法域统一到这个范围，保证 Python / NumPy / torch 三个随机
# 流拿到的是同一个种子；否则负值或更大值会在 NumPy 播种时抛错，
# 而此前 Python/torch 已经被播种，进程会停在部分初始化的状态。
MAX_SEED = 2**32 - 1



def resume_data_change_error(
    saved_sha256: Optional[str],
    current_sha256: Optional[str],
) -> Optional[str]:
    """
    检查续跑检查点与当前数据文件是否仍是同一份内容。

    数据指纹不同意味着 train/val/test 的切分内容已经变化，
    检查点里的优化器状态、采样 RNG 与「验证集选出的最优步」
    都对应旧数据；继续训练会把两个数据版本混成一条训练轨迹，
    任何指标都无法归因。旧检查点没有数据指纹时无法核对，
    返回 None 由调用方保留兼容提示。
    """
    if saved_sha256 is None or saved_sha256 == "":
        return None
    if not isinstance(saved_sha256, str):
        # 非字符串只可能来自损坏的检查点/config；不能因为
        # 当前指纹缺失（None/空串）就当成「没有变化」继续
        # 续跑，格式检查必须先于「无法核对」的兼容放行。
        return (
            "resume 数据指纹记录格式损坏（不是字符串）："
            "无法核对检查点与当前数据是否同一份，拒绝续跑；"
            "请修复检查点或换用全新的 --out 目录从头训练。"
        )
    if current_sha256 is None or current_sha256 == "":
        return None
    if not isinstance(current_sha256, str):
        return (
            "resume 数据指纹记录格式损坏（不是字符串）："
            "无法核对检查点与当前数据是否同一份，拒绝续跑；"
            "请修复检查点或换用全新的 --out 目录从头训练。"
        )
    if saved_sha256 == current_sha256:
        return None
    return (
        "resume 数据文件内容已变化（数据指纹 "
        f"{saved_sha256[:12]}... -> {current_sha256[:12]}...）："
        "检查点的参数/优化器状态与采样轨迹对应旧数据，"
        "续跑会把两个数据版本混成一条训练轨迹。请换用全新的 "
        "--out 目录与检查点从头训练。"
    )



def resume_split_change_error(
    saved_split_sha256: Optional[object],
    current_split_sha256: Optional[Dict[str, str]],
) -> Optional[str]:
    """
    检查续跑检查点记录的逐 split 内容指纹与当前划分是否一致。

    数据文件整体 sha256 相同、划分比例也相同，仍可能因为群组
    吸附、去重/清洗开关或切分实现变化而产生不同的 train/val/
    test 字节段。整体 sha256 无法发现这种情况；逐 split 指纹
    不同意味着检查点的 RNG/优化器状态对应另一批训练样本，必须
    拒绝续跑而不是把两套划分混成一条轨迹。
    """
    if saved_split_sha256 is None:
        return None
    if not isinstance(saved_split_sha256, dict):
        # 旧检查点缺字段是 None；其余非对象取值只可能来自损坏
        # 的检查点。不能把「无法核对」当成「没有变化」放行。
        return (
            "resume 数据划分内容指纹记录格式损坏（不是对象）："
            "无法核对检查点与当前数据是否同一划分，拒绝续跑；"
            "请修复检查点或换用全新的 --out 目录从头训练。"
        )
    if current_split_sha256 is None:
        # 当前数据可能来自不支持逐 split 指纹的旧版本，此时无法
        # 比较；但检查点自身记录损坏（非字符串/空指纹）不能借
        # 「当前缺失」绕过格式检查，否则损坏记录会被静默放行。
        if any(
            not isinstance(saved_hash, str) or not saved_hash
            for saved_hash in saved_split_sha256.values()
        ):
            return (
                "resume 数据划分内容指纹记录格式损坏（存在非字符串"
                "或空指纹）：无法核对检查点与当前数据是否同一"
                "划分，拒绝续跑；请修复检查点或换用全新的 --out "
                "目录从头训练。"
            )
        return None
    if not isinstance(current_split_sha256, dict):
        # 当前数据是新加载的；旧数据集只会缺字段（None）。
        # 其余非对象取值只可能来自损坏的数据/缓存，不能把
        # 「无法核对」当成「没有变化」放行续跑。
        return (
            "resume 当前数据划分内容指纹记录格式损坏（不是对象）："
            "无法核对检查点与当前数据是否同一划分，拒绝续跑；"
            "请重建数据缓存或换用全新的 --out 目录从头训练。"
        )

    mismatched = []
    invalid = []
    for split, saved_hash in saved_split_sha256.items():
        if not isinstance(saved_hash, str) or not saved_hash:
            # 损坏的旧记录不能当作「没有变化」：无法核对时
            # 续跑会把两套划分混在一起。
            invalid.append(split)
            continue
        # 旧检查点记录过的 split 在当前划分中消失，同样说明
        # 划分口径已经变化；不能因为缺少可比对象就静默放行。
        if (
            split not in current_split_sha256
            or saved_hash != current_split_sha256[split]
        ):
            mismatched.append(split)
    if not mismatched and not invalid:
        return None

    details = []
    if mismatched:
        details.append(
            "变化的 split: "
            + ", ".join(
                sorted(
                    (str(split) for split in mismatched)
                )
            )
        )
    if invalid:
        details.append(
            "记录损坏的 split: "
            + ", ".join(
                sorted(str(split) for split in invalid)
            )
        )
    return (
        "resume 数据划分内容指纹已变化（"
        + "；".join(details)
        + "）：整体数据 sha256 虽相同，但 train/val/test 的"
        "实际字节段不同（群组吸附、清洗开关或切分实现变化）。"
        "检查点的参数/优化器状态对应旧划分，续跑会把两套"
        "数据混成一条训练轨迹。请换用全新的 --out 目录与"
        "检查点从头训练。"
    )



def resume_effective_train_change_error(
    saved_sha256: Optional[object],
    current_sha256: Optional[str],
) -> Optional[str]:
    """
    检查续跑检查点的有效训练集指纹是否与当前一致。

    整体数据 sha256 与 split 指纹相同、清洗开关也相同，仍可能
    因为清洗实现变化而剔除不同区间：网络窗口、unigram/n-gram
    计数与数据采样的有效范围都会变。此时续跑等于把旧检查点的
    优化器/RNG 轨迹接到另一套训练样本上，必须拒绝。
    """
    if saved_sha256 is None or saved_sha256 == "":
        return None
    if not isinstance(saved_sha256, str):
        return (
            "resume 有效训练集指纹记录格式损坏（不是字符串）："
            "无法核对检查点与当前清洗后的训练内容，拒绝续跑；"
            "请修复检查点或换用全新的 --out 目录从头训练。"
        )
    if current_sha256 is None or current_sha256 == "":
        return None
    if not isinstance(current_sha256, str):
        # 当前数据是新加载的；旧数据集只会缺字段（None/空串）。
        # 其余非字符串取值只可能来自损坏的数据/缓存，不能把
        # 「无法核对」当成「没有变化」放行续跑。
        return (
            "resume 当前有效训练集指纹记录格式损坏（不是字符串）："
            "无法核对检查点与当前清洗后的训练内容，拒绝续跑；"
            "请重建数据缓存或换用全新的 --out 目录从头训练。"
        )
    if saved_sha256 == current_sha256:
        return None

    return (
        "resume 有效训练集指纹已变化（"
        f"{saved_sha256[:12]}... -> "
        f"{current_sha256[:12]}...）：整体数据与 split 边界"
        "可能相同，但清洗剔除区间/实际参与训练的字节不同。"
        "检查点对应旧的训练样本集合，续跑会把两套数据混成"
        "一条轨迹。请换用全新的 --out 目录与检查点从头训练。"
    )



def resume_split_isolation_change_error(
    saved_isolation: Optional[object],
    current_isolation: Optional[object],
) -> Optional[str]:
    """
    检查续跑检查点的时序隔离记录是否一致。

    split 内容指纹相同仍可能掩盖边界顺序/绝对位置变化（例如
    重复内容使不同边界产生相同片段）；把边界、顺序与重叠
    记录一起核对，避免续跑接到另一套时间切分上。旧检查点
    没有该字段时返回 None，由调用方给出兼容提示。
    """
    if saved_isolation is None:
        return None
    if not isinstance(saved_isolation, dict) or not isinstance(
        current_isolation, dict
    ):
        return (
            "resume 时序隔离记录格式损坏（不是对象）：无法核对"
            "检查点与当前 train/val/test 边界，拒绝续跑；请修复"
            "检查点或换用全新的 --out 目录从头训练。"
        )

    def canonical(value):
        """把记录转成键类型安全的可比较结构。

        旧检查点可能混入 int/str 键；json.dumps(sort_keys=True)
        会直接抛 TypeError。字典按键的稳定文本排序后转成
        [键, 值] 列表，既保留键的类型信息，又保证比较确定性。
        """
        if isinstance(value, dict):
            return [
                [repr(key), canonical(item)]
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: repr(pair[0]),
                )
            ]
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(
                (canonical(item) for item in value),
                key=repr,
            )
        return value

    saved_key = json.dumps(
        canonical(saved_isolation),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    current_key = json.dumps(
        canonical(current_isolation),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if saved_key == current_key:
        return None

    return (
        "resume 时序隔离记录已变化：train/val/test 的顺序、"
        "绝对边界或重叠检查与检查点不一致。检查点对应另一套"
        "时间切分，续跑会把两套划分混成一条轨迹。请换用全新"
        "的 --out 目录与检查点从头训练。"
    )



def resume_amp_change_error(
    saved_args: object,
    current_amp_dtype: Optional[object],
) -> Optional[str]:
    """
    检查续跑检查点与当前的混合精度环境是否一致。
    """
    if (
        not isinstance(saved_args, dict)
        or "amp_dtype" not in saved_args
    ):
        return None

    saved_dtype = saved_args.get("amp_dtype")
    saved_text = (
        str(saved_dtype)
        if saved_dtype is not None
        else "float32"
    )
    current_text = (
        str(current_amp_dtype)
        if current_amp_dtype is not None
        else "float32"
    )
    if saved_text == current_text:
        return None

    return (
        "resume 检查点的混合精度环境为 "
        f"{saved_text}，当前设备会使用 {current_text}："
        "autocast dtype 与 GradScaler 状态会改变浮点路径和"
        "优化器轨迹。请在同一设备能力/同一 AMP 设置下续跑，"
        "或换用全新的 --out 目录从头训练。"
    )



def seed_error(seed: int) -> Optional[str]:
    """
    校验随机种子的统一合法域；返回错误消息或 None。
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        return "随机种子必须是整数"
    if seed < 0 or seed > MAX_SEED:
        return (
            f"随机种子必须落在 [0, {MAX_SEED}]（NumPy "
            "RandomState 的合法范围；Python/torch/CUDA 也使用"
            "同一个值以保持随机流一致）"
        )
    return None



def _seed_optional_accelerators(seed: int) -> List[str]:
    """
    固定 PyTorch 可选的 XPU / MPS 后端随机源。

    CUDA 由 set_seed 直接处理；XPU（Intel GPU）与 MPS（Apple
    GPU）在对应硬件上同样是训练随机性的来源。旧版 torch 没有
    这些命名空间或后端不可用时静默跳过，不影响 CPU/CUDA 路径。
    返回值只列出真正播种成功的后端，写进运行元数据。
    """
    seeded: List[str] = []

    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                xpu.manual_seed_all(seed)
                seeded.append("xpu")
        except (AttributeError, RuntimeError):
            pass

    mps = getattr(torch, "mps", None)
    if mps is not None:
        try:
            mps_available = getattr(
                getattr(torch, "backends", None),
                "mps",
                None,
            )
            if (
                mps_available is not None
                and mps_available.is_available()
                and hasattr(mps, "manual_seed")
            ):
                mps.manual_seed(seed)
                seeded.append("mps")
        except (AttributeError, RuntimeError):
            pass

    return seeded



def set_seed(
    seed: int,
    *,
    strict_determinism: bool = False,
):
    """
    固定 Python / NumPy / PyTorch(CPU+CUDA) / cuDNN 的随机种子。

    数据采样不经过 DataLoader worker，而由显式的
    torch.Generator 承担（见 ByteDataset.batch 与 main 中的
    train_generator）：其初始种子同样来自 --seed，当前状态随
    检查点一起存档，恢复训练时逐位还原。

    strict_determinism=True 时，遇到没有确定性实现的算子直接
    报错（而不是只告警），供正式复现实验 fail-fast；默认
    warn_only 以兼容尚缺确定性内核的算子。

    跨硬件架构仍可能因浮点归约顺序产生微小差异；同一台机器
    上的多次运行应保持一致。
    """
    error = seed_error(seed)
    if error is not None:
        raise ValueError(error)

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # manual_seed_all 覆盖当前可见的所有 CUDA 设备。
        torch.cuda.manual_seed_all(seed)

    # XPU / MPS 等可选加速后端同样属于训练随机性来源。
    optional_backends = _seed_optional_accelerators(seed)

    cuda_initial_seeds: List[int] = []
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_index):
                    cuda_initial_seeds.append(
                        int(torch.cuda.initial_seed())
                    )
            except (AssertionError, RuntimeError):
                # 设备不可访问时不伪造读数；device_count 本身
                # 仍记录在清单里。
                continue

    global _LAST_SEED_MANIFEST
    _LAST_SEED_MANIFEST = {
        "seed": int(seed),
        "python_random": True,
        "numpy": np is not None,
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "cuda_initial_seeds": cuda_initial_seeds,
        "optional_backends": optional_backends,
    }

    # PYTHONHASHSEED 只对之后启动的子进程（如 torch.compile 的
    # 编译进程）生效；当前解释器的哈希种子在启动时已经确定，
    # 需要严格复现时应在命令行显式设置该环境变量。
    #
    # 若启动时未显式设置，本函数就是该变量的所有者：之后再次
    # set_seed 必须把值更新为当前 seed，否则先 set_seed(42) 再
    # set_seed(7) 时子进程仍继承 42，和当前随机流不一致。用户
    # 启动时显式设置的值则不能被静默覆盖（尊重外部固定口径）。
    if _PYTHONHASHSEED_AT_START is None:
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    global _PYTHONHASHSEED_WARNED
    if (
        _PYTHONHASHSEED_AT_START is None
        and not _PYTHONHASHSEED_WARNED
    ):
        _PYTHONHASHSEED_WARNED = True
        print(
            "[repro] 提示: 进程启动时未设置 PYTHONHASHSEED，当前"
            "解释器的 hash 随机化未被固定（只影响依赖 hash() "
            "的代码与之后启动的子进程，其余随机流已按 --seed "
            "固定）。严格逐位复现可在命令前加 "
            f"PYTHONHASHSEED={seed}。"
        )

    # 要求 PyTorch 优先选用确定性实现。bincount、attention
    # 反向等 CUDA 算子若不存在确定性实现会明确告警，而不是
    # 静默产生跨重启不可复现的微小差异。严格模式直接失败，
    # 避免「带告警继续跑」的实验被误当成可复现结果。
    torch.use_deterministic_algorithms(
        True,
        warn_only=not strict_determinism,
    )

    # 本文件所有 torch.empty 缓冲在使用前都会被完整写入，
    # 不依赖未初始化缓冲的 NaN 填充来保证结果确定性。
    torch.utils.deterministic.fill_uninitialized_memory = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def configure_cuda_precision(allow_tf32: bool = False) -> None:
    """
    配置 CUDA 的 fp32 矩阵乘精度。

    TF32 用 10 位尾数代替 fp32 的 23 位尾数，能加速 Ampere 及
    更新架构上的矩阵乘，但会改变数值结果，且不同 GPU 架构 /
    cuBLAS 版本的 TF32 实现不完全一致。默认关闭（完整 fp32），
    需要吞吐量时用 --allow-tf32 显式开启；两种状态都会写入
    run_metadata 的 allow_tf32_matmul / allow_tf32_cudnn，供
    --compare 的复现性来源检查核对。

    CPU 或 MPS 路径同样可以安全调用：这两个后端标志始终存在，
    只是不生效。
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    # float32_matmul_precision 与上面的 TF32 开关在部分版本里
    # 联动，但环境变量/用户代码可能把它留成别的档位。这里显式
    # 固定：关闭 TF32 时用 highest（完整 fp32），开启时用 high
    #（允许 TensorFloat32）。实际生效值仍写入 run_metadata。
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(
            "high" if allow_tf32 else "highest"
        )



def pythonhashseed_error(
    require_pythonhashseed: bool,
) -> Optional[str]:
    """
    --require-pythonhashseed 的合法性检查：返回错误消息或
    None。当前解释器的哈希种子在进程启动时已确定，set_seed
    的 setdefault 无法追溯固定；需要严格逐位复现的正式运行
    必须由用户在命令行显式设置。
    """
    if not require_pythonhashseed:
        return None
    if _PYTHONHASHSEED_AT_START is None:
        return (
            "--require-pythonhashseed 要求进程启动前显式设置 "
            "PYTHONHASHSEED（例如 "
            "PYTHONHASHSEED=42 python experiment.py ...）；"
            "当前进程的 hash 随机化未被固定，严格逐位复现"
            "无法保证"
        )
    text = str(_PYTHONHASHSEED_AT_START).strip()
    # "random"/空字符串与超范围整数要么显式随机化、要么被
    # 解释器拒绝/回退；它们都不构成一个固定的 hash 种子。
    if (
        not text
        or text.lower() == "random"
        or not text.isdigit()
        or int(text) > MAX_SEED
    ):
        return (
            "--require-pythonhashseed 要求 PYTHONHASHSEED 是 "
            f"[0, {MAX_SEED}] 内的整数（当前值 "
            f"{_PYTHONHASHSEED_AT_START!r} 不能固定本进程的 "
            "hash 随机化）；请用如 PYTHONHASHSEED=42 重新启动"
        )
    return None



def configure_cpu_threads(num_threads: Optional[int]) -> None:
    """
    固定 PyTorch CPU 线程数（None=保持环境默认）。

    CPU 上的并行归约顺序可能随线程数变化，同一台机器上
    改变 OMP/MKL 线程数也可能让「同种子」结果出现末位差异。
    正式复现实验可显式固定线程数；实际生效值随 run_metadata
    存档，即使不固定也能追溯到当时的线程环境。
    """
    if num_threads is None:
        return

    if isinstance(num_threads, bool) or num_threads <= 0:
        raise ValueError("CPU 线程数必须为正整数")

    torch.set_num_threads(int(num_threads))



def _capture_rng_state() -> Dict[str, object]:
    """
    捕获当前进程的全部随机数状态，供临时代码（如 sanity check）
    运行后原样恢复，保证是否运行临时代码不影响主训练轨迹。
    """
    state: Dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    return state



def _restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None and np is not None:
        np.random.set_state(numpy_state)



def _numpy_rng_state_payload() -> Optional[Dict[str, object]]:
    """
    把 NumPy 全局随机状态转成只含基本类型的可存档形式。

    检查点经 torch.load 读取；新版 PyTorch 默认
    weights_only=True，直接存 numpy ndarray 会被拒绝，因此
    key 数组转成 int 列表（恢复时按 uint32 重建）。
    """
    if np is None:
        return None

    name, keys, pos, has_gauss, cached = np.random.get_state()

    return {
        "name": str(name),
        "keys": [int(value) for value in keys.tolist()],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }



def _restore_numpy_rng_state(
    payload: Optional[Dict[str, object]],
) -> None:
    if payload is None or np is None:
        return

    np.random.set_state((
        str(payload.get("name", "MT19937")),
        np.asarray(
            payload.get("keys", []),
            dtype=np.uint32,
        ),
        int(payload.get("pos", 0)),
        int(payload.get("has_gauss", 0)),
        float(payload.get("cached_gaussian", 0.0)),
    ))



# ============================================================
# Basic layers
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

        # torch>=2.4 提供内置实现，内部使用高精度累加，
        # 语义与手动实现一致。
        self._fused = (
            F.rms_norm if hasattr(F, "rms_norm") else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused is not None and (x.is_cuda or x.dtype == torch.float32):
            return self._fused(
                x,
                self.weight.shape,
                self.weight,
                self.eps,
            )

        input_dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(
            y.square().mean(dim=-1, keepdim=True) + self.eps
        )
        # 在 fp32 下乘权重后再转回输入 dtype，
        # 与融合路径的输出 dtype 保持一致；
        # 否则 bf16 * fp32 权重会提升为 fp32 输出。
        return (y * self.weight).to(input_dtype)



class SegmentAttentionMask:
    """文档级 segment attention mask（论文 3.1 节）。

    输入是每个 token 的文档编号 ``[batch, time]``；注意力只允许
    同一文档内、且 key 位置不晚于 query 位置的 token 可见。

    实现有两条路径：

    - CUDA：``torch.nn.attention.flex_attention`` 的 BlockMask，
      不物化 [batch, time, time] 的掩码矩阵，能支撑论文的
      batch=256、context=4096 规模；
    - CPU（或 flex_attention 不可用）：物化 bool 掩码，仅用于
      小规模测试与评估；超过 1 GiB 时显式报错，而不是悄悄
      退化成无掩码的因果注意力。

    掩码在每次前向构造一次，随后共享给所有 Block（同一 batch
    的所有层/分支使用同一份文档边界）。
    """

    # 物化 bool 掩码的内存上限（字节）。B*T*T 超过它时
    # CPU 路径直接报错；论文规模请在 CUDA 上运行。
    DENSE_LIMIT_BYTES = 1 << 30

    def __init__(self, segment_ids: torch.Tensor):
        if (
            segment_ids.dim() != 2
            or segment_ids.dtype
            not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "segment_ids 必须是 [batch, time] 的整数张量"
            )
        self.segment_ids = segment_ids
        self.batch, self.length = segment_ids.shape
        self.dense: Optional[torch.Tensor] = None
        self.block_mask = None
        self.path = "dense"
        if segment_ids.device.type == "cuda":
            if self._build_flex():
                return
        self._build_dense()

    def _build_flex(self) -> bool:
        try:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
            )
        except ImportError:
            return False

        segment = self.segment_ids

        def mask_mod(batch, head, q_index, kv_index):
            doc = segment[batch]
            return (q_index >= kv_index) & (
                doc[q_index] == doc[kv_index]
            )

        try:
            self.block_mask = create_block_mask(
                mask_mod,
                B=int(self.batch),
                H=None,
                Q_LEN=int(self.length),
                KV_LEN=int(self.length),
                device=segment.device,
            )
        except Exception:
            return False
        self.path = "flex"
        return True

    def _build_dense(self) -> None:
        needed = (
            int(self.batch)
            * int(self.length)
            * int(self.length)
        )
        if needed > self.DENSE_LIMIT_BYTES:
            raise RuntimeError(
                "文档级 segment mask 需要物化 "
                f"{self.batch}x{self.length}x{self.length} 的布尔"
                "掩码（超过 1 GiB）；CPU 上无法运行该规模的"
                "论文协议。请在 CUDA 上运行（走 flex_attention "
                "BlockMask 的稀疏路径），或减小 --batch-size / "
                "--seq-len"
            )
        segment = self.segment_ids
        causal = torch.ones(
            (self.length, self.length),
            dtype=torch.bool,
            device=segment.device,
        ).tril()
        same_doc = segment[:, None, :, None] == segment[
            :, None, None, :
        ]
        # [batch, 1, time, time]：可直接广播到 SDPA 的
        # (batch, heads, time, time)。
        self.dense = same_doc & causal
        self.path = "dense"

    def attention(self, q, k, v):
        if self.path == "flex":
            from torch.nn.attention.flex_attention import (
                flex_attention,
            )
            return flex_attention(
                q, k, v, block_mask=self.block_mask
            )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.dense,
            is_causal=False,
        )
