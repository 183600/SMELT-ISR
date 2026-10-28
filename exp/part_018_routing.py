from __future__ import annotations



class RoutingProbe:
    """
    RQ3/RQ4 的路由探针：在分析前向中按调用顺序记录每次 route()。

    只由 analyze_routing 挂载/卸载；探针为 None 时 route()
    不记录。
    """

    def __init__(self, cfg: Config, batch_size: int):
        self.cfg = cfg
        self.batch_size = max(1, int(batch_size))
        self.events: List[Dict[str, object]] = []
        self._modules: List[SparseMoE] = []
        self._block_index: Dict[int, int] = {}
        self._call_counts: Dict[int, int] = {}
        self._wrapped_model: Optional[nn.Module] = None
        self._original_forward = None

    def attach(self, model: nn.Module) -> None:
        core = _unwrap_model(model)

        # 轮次计数必须在每次前向开始时清零：第 1 次调用是
        # 第一轮、第 2 次是第二轮。包装 model.forward（nn.Module
        # 的 __call__ 会在实例上查找 forward），分析循环只通过
        # core(x) 调用前向，因此每次前向都从干净计数开始。
        self._wrapped_model = core
        self._original_forward = core.forward
        probe = self

        def _forward_with_probe_reset(*args, **kwargs):
            probe._call_counts.clear()
            return probe._original_forward(*args, **kwargs)

        core.forward = _forward_with_probe_reset

        for index, block in enumerate(getattr(core, "blocks", [])):
            moe = getattr(block, "moe", None)
            if not isinstance(moe, SparseMoE):
                continue
            self._block_index[id(moe)] = index
            self._modules.append(moe)
            moe._routing_probe = self

    def detach(self) -> None:
        for moe in self._modules:
            moe._routing_probe = None
        self._modules.clear()
        if self._wrapped_model is not None:
            self._wrapped_model.forward = self._original_forward
            self._wrapped_model = None
            self._original_forward = None
        self._call_counts.clear()

    def _branch_cosine_stats(
        self,
        x: torch.Tensor,
        branch_dim: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        分支表示的两两余弦相似度（mean 与最小 pair）。

        余弦接近 1 说明分支轨迹几乎重合（延迟合并没有带来
        独立计算）；明显小于 1 说明分支确实在分化。逐 pair
        统计先在设备上堆叠；均值/最小值保持为设备标量，由
        build_report 与其他路由统计一起一次性取回。
        """
        branches = int(x.size(branch_dim))
        if branches < 2:
            return None, None

        base = x.detach().float()
        base = base / base.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        moved = base.movedim(branch_dim, 0).reshape(
            branches,
            -1,
            base.size(-1),
        )

        pair_tensors: List[torch.Tensor] = []
        for i in range(branches):
            for j in range(i + 1, branches):
                cosine = (moved[i] * moved[j]).sum(dim=-1)
                pair_tensors.append(cosine.mean())

        pair_means = torch.stack(pair_tensors)

        # 保持为 0 维设备张量，由 build_report 取回。
        return pair_means.mean(), pair_means.min()

    def record(
        self,
        module: SparseMoE,
        x: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        block = self._block_index.get(id(module))
        if block is None:
            return

        key = id(module)
        call = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = call

        with torch.no_grad():
            indices = top_indices.detach().reshape(
                -1,
                top_indices.size(-1),
            )
            if indices.numel() == 0:
                return

            n_experts = int(module.n_experts)
            flat_indices = indices.to(torch.int64).clamp(
                min=0,
                max=n_experts - 1,
            ).reshape(-1)
            weights = top_weights.detach().float().reshape(-1)

            usage = torch.bincount(
                flat_indices,
                minlength=n_experts,
            )
            usage_share = usage.float() / usage.sum().clamp_min(1)

            gate_mass = torch.zeros(
                n_experts,
                dtype=torch.float32,
                device=flat_indices.device,
            )
            gate_mass.scatter_add_(0, flat_indices, weights)

            probs = router_logits.detach().float().softmax(dim=-1)
            log_e = (
                math.log(n_experts)
                if n_experts > 1
                else 0.0
            )

            if log_e > 0.0:
                router_entropy = (
                    -(
                        probs.clamp_min(1e-12).log()
                        * probs
                    ).sum(-1).mean()
                    / log_e
                )
                nonzero = usage_share[usage_share > 0]
                load_entropy = (
                    -(nonzero * nonzero.log()).sum() / log_e
                )
            else:
                router_entropy = probs.new_zeros(())
                load_entropy = probs.new_zeros(())

            branch_mean: Optional[torch.Tensor] = None
            branch_min: Optional[torch.Tensor] = None
            cfg = self.cfg
            shape = x.shape
            is_middle = cfg.middle_start <= block < cfg.middle_end
            branch_dim: Optional[int] = None

            if cfg.mode == "A" and len(shape) == 4:
                branch_dim = 2
            elif (
                cfg.mode in ("B", "B_early")
                and is_middle
                and cfg.k1 >= 2
                and len(shape) == 3
                and not (
                    block == cfg.middle_start and call == 1
                )
                and shape[0] % cfg.k1 == 0
            ):
                rows = int(shape[0])
                batch = rows // cfg.k1
                if 0 < batch <= self.batch_size:
                    x = x.reshape(
                        batch,
                        cfg.k1,
                        shape[1],
                        shape[2],
                    )
                    branch_dim = 1

            if branch_dim is not None:
                branch_mean, branch_min = (
                    self._branch_cosine_stats(x, branch_dim)
                )

            # 统计量留在设备上，由 build_report 取回。
            scalar_values = [
                router_entropy.to(torch.float64),
                load_entropy.to(torch.float64),
                usage_share.max().to(torch.float64),
                (usage == 0).sum().to(torch.float64),
                probs.max(-1).values.mean().to(torch.float64),
            ]
            if branch_mean is None:
                nan = router_entropy.new_tensor(
                    float("nan")
                ).to(torch.float64)
                scalar_values.extend([nan, nan])
            else:
                scalar_values.extend([
                    branch_mean.to(torch.float64),
                    branch_min.to(torch.float64),
                ])
            scalars = torch.stack(scalar_values)

            self.events.append({
                "block": block,
                "round": call,
                "top_k": int(indices.size(-1)),
                "rows": int(indices.size(0)),
                "input_shape": [int(size) for size in shape],
                "usage": usage.to(torch.int64),
                "gate_mass": gate_mass,
                "scalars": scalars,
                # int16 在 n_experts>32767 时会静默溢出成负下标，
                # 后续集合指标会从错误的专家上计算；int32 足够
                # 覆盖任何可实际分配的路由表规模。
                "indices": indices.to(torch.int32),
            })

    @staticmethod
    def _accumulate(
        accumulator: Dict[str, object],
        metrics: Optional[Dict[str, float]],
    ) -> None:
        if not metrics:
            return
        for name, value in metrics.items():
            if name == "units":
                accumulator[name] = (
                    int(accumulator.get(name, 0)) + int(value)
                )
            elif isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                accumulator.setdefault(name, []).append(float(value))
            # 非数值字段（如 chance_note 说明文字）不参与均值
            # 聚合，直接忽略而不是 float() 崩溃。

    @staticmethod
    def _finalize(
        accumulator: Dict[str, object],
    ) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for name, value in accumulator.items():
            if name == "units":
                result[name] = int(value)
            else:
                values = [float(v) for v in value]
                result[name] = (
                    sum(values) / len(values)
                    if values
                    else None
                )
        return result

    def _round_summary(
        self,
        events: List[Dict[str, object]],
        n_experts: int,
    ) -> Optional[Dict[str, object]]:
        if not events:
            return None

        usage = [0] * n_experts
        gate = [0.0] * n_experts
        token_count = 0
        for event in events:
            token_count += int(event["rows"])
            for index, count in enumerate(event["usage"]):
                usage[index] += int(count)
            for index, mass in enumerate(event["gate_mass"]):
                gate[index] += float(mass)

        # usage 是 assignment 计数（token × top_k），token 数单独
        # 记录，避免 token 量与 assignment 量混淆。
        total = sum(usage)
        usage_share = (
            [count / total for count in usage]
            if total > 0
            else [0.0] * n_experts
        )
        gate_total = sum(gate)
        gate_share = (
            [mass / gate_total for mass in gate]
            if gate_total > 0
            else [0.0] * n_experts
        )

        shares = torch.tensor(usage_share, dtype=torch.float64)
        log_e = (
            math.log(n_experts)
            if n_experts > 1
            else 0.0
        )
        if log_e > 0.0 and total > 0:
            nonzero = shares[shares > 0]
            load_entropy = float(
                -(nonzero * nonzero.log()).sum() / log_e
            )
        else:
            load_entropy = 0.0

        mean_share = (
            sum(usage_share) / n_experts
            if n_experts > 0
            else 0.0
        )
        if mean_share > 0:
            variance = sum(
                (share - mean_share) ** 2
                for share in usage_share
            ) / n_experts
            load_cv = math.sqrt(variance) / mean_share
        else:
            load_cv = 0.0

        def _mean_events(field: str) -> Optional[float]:
            values = [
                float(event[field])
                for event in events
                if event.get(field) is not None
            ]
            return sum(values) / len(values) if values else None

        return {
            "events": len(events),
            "tokens": token_count,
            "assignments": total,
            "expert_usage_share": [
                round(share, 6) for share in usage_share
            ],
            "expert_gate_mass_share": [
                round(share, 6) for share in gate_share
            ],
            "load_entropy": load_entropy,
            "load_max_share": max(usage_share) if usage_share else 0.0,
            "load_cv": load_cv,
            "dead_experts": sum(1 for count in usage if count == 0),
            "router_entropy_mean": _mean_events("router_entropy"),
            "router_max_prob_mean": _mean_events("router_max_prob"),
        }

    def _pair_overlap(
        self,
        first: Dict[str, object],
        second: Dict[str, object],
        block: int,
    ) -> Dict[str, object]:
        """
        一对（第一轮、第二轮）route 记录的重合指标。

        返回 branch_level（逐分支对齐）与 token_level（跨分支
        并集）两种口径；smelt/local_base 无分支结构，两者相同。
        """
        cfg = self.cfg
        n_experts = cfg.n_experts
        k1 = cfg.k1
        k2 = cfg.k2

        a = first["indices"].to(torch.int64)
        b = second["indices"].to(torch.int64)
        shape_first = list(first["input_shape"])
        shape_second = list(second["input_shape"])

        result: Dict[str, object] = {
            "branch_level": None,
            "token_level": None,
            "branch_self_reuse_share": None,
        }

        if cfg.mode in ("base", "smelt", "local_base", "local_base_k2"):
            metrics = _routing_set_metrics(a, b, n_experts)
            result["branch_level"] = metrics
            result["token_level"] = (
                dict(metrics) if metrics is not None else None
            )
            return result

        if cfg.mode == "A":
            # 第一轮 [B*T, k1]；第二轮 [B*T*k1, k2]，
            # 行序为 (batch, time, branch)。
            if len(shape_second) != 4 or a.size(0) * k1 != b.size(0):
                return result
            rows = a.size(0)
            b_branch = b.reshape(rows, k1, k2)
            a_expanded = a.repeat_interleave(k1, dim=0)
            result["branch_level"] = _routing_set_metrics(
                a_expanded,
                b_branch.reshape(-1, k2).contiguous(),
                n_experts,
            )
            a_bool = _routing_set_bool(a, n_experts)
            b_bool = _routing_set_bool(
                b,
                n_experts,
            ).reshape(rows, k1, n_experts).any(dim=1)
            result["token_level"] = _routing_set_metrics_bool(
                a_bool,
                b_bool,
            )
            b_branch_bool = _routing_set_bool(
                b,
                n_experts,
            ).reshape(rows, k1, n_experts)
            own = b_branch_bool.gather(
                2,
                a.unsqueeze(2),
            ).float()
            result["branch_self_reuse_share"] = float(own.mean())
            return result

        if cfg.mode in ("B", "B_early") and block == cfg.middle_start:
            # 分支创建：[B, T, k1]（第 j 个分支的专家）；第二轮
            # 经 run_independent_blocks 把分支维并入批维后为
            # [B*k1*T, k2]，行序为 (batch, branch, time)。
            batch = int(shape_first[0])
            time_steps = int(shape_first[1])
            if batch * time_steps != a.size(0):
                return result
            if batch * k1 * time_steps != b.size(0):
                return result
            a_tokens = a.reshape(batch, time_steps, k1)
            b_branch = b.reshape(
                batch,
                k1,
                time_steps,
                k2,
            )
            # 分支 j 的第二轮只应和「该分支携带的第一轮专家」
            # 比较，而不是和父 token 的整个 top-k 集合比较：
            # a_tokens[..., j] 才是分支 j 的起点。行序必须与
            # b_branch 的 (batch, branch, time) 一致。
            a_expanded = (
                a_tokens.permute(0, 2, 1)
                .unsqueeze(-1)
                .reshape(-1, 1)
            )
            result["branch_level"] = _routing_set_metrics(
                a_expanded,
                b_branch.reshape(-1, k2).contiguous(),
                n_experts,
            )
            b_bool = _routing_set_bool(
                b_branch.reshape(-1, k2),
                n_experts,
            ).reshape(batch, k1, time_steps, n_experts).any(dim=1)
            result["token_level"] = _routing_set_metrics_bool(
                _routing_set_bool(
                    a_tokens.reshape(batch * time_steps, k1),
                    n_experts,
                ),
                b_bool.reshape(batch * time_steps, n_experts),
            )
            own = _routing_set_bool(
                b_branch.reshape(-1, k2),
                n_experts,
            ).reshape(
                batch,
                k1,
                time_steps,
                n_experts,
            ).gather(
                3,
                a_tokens.permute(0, 2, 1).unsqueeze(-1),
            ).float()
            result["branch_self_reuse_share"] = float(own.mean())
            return result

        if cfg.mode in ("B", "B_early"):
            # 第一轮其余 Block 与第二轮：x 为 [B*k1, T, D]，
            # 行序为 (batch, branch)；两轮逐分支对齐。
            time_steps = int(shape_first[1])
            if time_steps <= 0:
                return result
            rows = a.size(0)
            if rows % (k1 * time_steps) != 0:
                return result
            batch = rows // (k1 * time_steps)
            if b.size(0) != rows:
                return result
            a_branch = a.reshape(batch, k1, time_steps, k1)
            b_branch = b.reshape(batch, k1, time_steps, k2)
            result["branch_level"] = _routing_set_metrics(
                a_branch.reshape(-1, k1),
                b_branch.reshape(-1, k2).contiguous(),
                n_experts,
            )
            a_bool = _routing_set_bool(
                a_branch.reshape(-1, k1),
                n_experts,
            ).reshape(
                batch,
                k1,
                time_steps,
                n_experts,
            ).any(dim=1)
            b_bool = _routing_set_bool(
                b_branch.reshape(-1, k2),
                n_experts,
            ).reshape(
                batch,
                k1,
                time_steps,
                n_experts,
            ).any(dim=1)
            result["token_level"] = _routing_set_metrics_bool(
                a_bool.reshape(batch * time_steps, n_experts),
                b_bool.reshape(batch * time_steps, n_experts),
            )
            return result

        return result

    def _materialize_events(self) -> None:
        """
        把 record 阶段留在设备上的小张量取回并写回事件。

        重复调用是幂等的。
        """
        if not self.events:
            return
        if isinstance(self.events[0].get("usage"), list):
            return

        scalars = torch.stack(
            [event["scalars"] for event in self.events]
        ).cpu()
        usages = torch.stack(
            [event["usage"] for event in self.events]
        ).cpu()
        gate_masses = torch.stack(
            [event["gate_mass"] for event in self.events]
        ).cpu()
        flat_indices = torch.cat(
            [
                event["indices"].reshape(-1)
                for event in self.events
            ]
        ).cpu()

        offset = 0
        for position, event in enumerate(self.events):
            values = scalars[position].tolist()
            event["router_entropy"] = values[0]
            event["load_entropy"] = values[1]
            event["load_max_share"] = values[2]
            event["dead_experts"] = int(values[3])
            event["router_max_prob"] = values[4]
            event["branch_cosine_mean"] = (
                None
                if math.isnan(values[5])
                else values[5]
            )
            event["branch_cosine_min_pair"] = (
                None
                if math.isnan(values[6])
                else values[6]
            )
            event["usage"] = usages[position].to(
                torch.int64
            ).tolist()
            event["gate_mass"] = gate_masses[
                position
            ].tolist()

            shape = event["indices"].shape
            count = event["indices"].numel()
            event["indices"] = flat_indices[
                offset:offset + count
            ].view(shape)
            offset += count
            del event["scalars"]

    def build_report(self, split: str = "val") -> Dict[str, object]:
        cfg = self.cfg
        report: Dict[str, object] = {
            "mode": cfg.mode,
            "split": split,
            "top_k": cfg.top_k,
            "k1": cfg.k1,
            "k2": cfg.k2,
            "n_experts": cfg.n_experts,
            "events": len(self.events),
            "round_semantics": (
                "round = 同一 Block 内 route() 的调用序号："
                "1=第一轮，2=第二轮"
            ),
            "branch_semantics": (
                "B 的分支是「按第一轮路由排名（slot）组成的序列"
                "分支」：同一 slot 在不同位置可以对应不同专家。"
                "分支间的前向状态彼此隔离（各自做 Attention），"
                "但全部专家/router/embedding 参数共享，梯度通过"
                "共享参数耦合；它们不是参数独立的专家轨迹，也"
                "不保证统计独立。'独立'只在「前向状态不混合」"
                "这一意义上成立"
            ),
            "notes": [
                "诊断只使用验证集固定窗口，测试集不参与。",
                "路由诊断不参与模型选择、早停或指标优化。",
                "重合度在同一 token、同一 Block 的两轮路由之间"
                "比较；A/B 的分支结构另有逐分支与 token 级"
                "（跨分支并集）口径。",
                "Jaccard 必须与 jaccard_chance_uniform / "
                "jaccard_chance_marginal 及 lift 一起读；低重合"
                "本身不证明路由更有效，需要干预实验验证。",
            ],
        }

        if cfg.mode == "dense":
            report["skipped"] = (
                "dense 基线没有 MoE 路由，无 RQ3/RQ4 诊断"
            )
            return report

        if not self.events:
            report["skipped"] = (
                "分析前向未记录到任何 route 调用"
            )
            return report

        self._materialize_events()

        grouped: Dict[int, Dict[int, List[Dict[str, object]]]] = {}
        for event in self.events:
            grouped.setdefault(event["block"], {}).setdefault(
                event["round"],
                [],
            ).append(event)

        middle_blocks = list(
            range(cfg.middle_start, cfg.middle_end)
        )

        per_block: Dict[str, object] = {}
        rq3_summary_values: Dict[str, List[float]] = {}
        rq4_round_values: Dict[str, Dict[str, List[float]]] = {
            "round1": {},
            "round2": {},
        }
        divergence_values: List[float] = []
        tokens_total = 0

        for block in middle_blocks:
            rounds = grouped.get(block, {})
            round1 = rounds.get(1, [])
            round2 = rounds.get(2, [])

            round1_summary = self._round_summary(
                round1,
                cfg.n_experts,
            )
            round2_summary = self._round_summary(
                round2,
                cfg.n_experts,
            )
            if round1_summary is not None:
                tokens_total += int(round1_summary["tokens"])

            overlap_accumulators: Dict[str, Dict[str, object]] = {
                "branch_level": {},
                "token_level": {},
            }
            self_reuse_values: List[float] = []
            cosine_values: List[float] = []
            cosine_min_values: List[float] = []

            for first, second in zip(round1, round2):
                overlap = self._pair_overlap(
                    first,
                    second,
                    block,
                )
                self._accumulate(
                    overlap_accumulators["branch_level"],
                    overlap.get("branch_level"),
                )
                self._accumulate(
                    overlap_accumulators["token_level"],
                    overlap.get("token_level"),
                )
                if overlap.get("branch_self_reuse_share") is not None:
                    self_reuse_values.append(
                        float(overlap["branch_self_reuse_share"])
                    )

            for event in round1 + round2:
                if event.get("branch_cosine_mean") is not None:
                    cosine_values.append(
                        float(event["branch_cosine_mean"])
                    )
                if event.get("branch_cosine_min_pair") is not None:
                    cosine_min_values.append(
                        float(event["branch_cosine_min_pair"])
                    )

            branch_metrics = self._finalize(
                overlap_accumulators["branch_level"]
            )
            token_metrics = self._finalize(
                overlap_accumulators["token_level"]
            )

            block_report: Dict[str, object] = {
                "round1": round1_summary,
                "round2": round2_summary,
                "expert_overlap_branch_level": (
                    branch_metrics or None
                ),
                "expert_overlap_token_level": (
                    token_metrics or None
                ),
                "branch_self_reuse_share": (
                    sum(self_reuse_values) / len(self_reuse_values)
                    if self_reuse_values
                    else None
                ),
                "branch_cosine_mean": (
                    sum(cosine_values) / len(cosine_values)
                    if cosine_values
                    else None
                ),
                "branch_cosine_min_pair": (
                    sum(cosine_min_values) / len(cosine_min_values)
                    if cosine_min_values
                    else None
                ),
            }
            per_block[str(block)] = block_report

            for name in (
                "second_round_jaccard_mean",
                "second_round_exact_match_share",
                "second_round_slot_reuse_mean",
                "first_round_recall_mean",
                "second_round_unique_reuse_mean",
                "expected_intersection_uniform",
                "jaccard_chance_uniform",
                "jaccard_chance_marginal",
                "jaccard_lift_over_uniform",
                "jaccard_lift_over_marginal",
                "jaccard_adjusted_uniform",
                "jaccard_adjusted_marginal",
            ):
                value = branch_metrics.get(name)
                if isinstance(value, (int, float)):
                    rq3_summary_values.setdefault(
                        name,
                        [],
                    ).append(float(value))
                token_value = token_metrics.get(name)
                if isinstance(token_value, (int, float)):
                    rq3_summary_values.setdefault(
                        f"token_{name}",
                        [],
                    ).append(float(token_value))
            if self_reuse_values:
                rq3_summary_values.setdefault(
                    "branch_self_reuse_share",
                    [],
                ).append(
                    sum(self_reuse_values) / len(self_reuse_values)
                )

            for round_name, summary in (
                ("round1", round1_summary),
                ("round2", round2_summary),
            ):
                if summary is None:
                    continue
                for field in (
                    "load_entropy",
                    "load_max_share",
                    "dead_experts",
                    "router_entropy_mean",
                    "router_max_prob_mean",
                ):
                    value = summary.get(field)
                    if isinstance(value, (int, float)):
                        rq4_round_values[round_name].setdefault(
                            field,
                            [],
                        ).append(float(value))

            if cosine_values:
                divergence_values.append(
                    sum(cosine_values) / len(cosine_values)
                )

        rq3_summary = {
            name: sum(values) / len(values)
            for name, values in rq3_summary_values.items()
            if values
        }

        rq4_summary = {}
        for round_name, field_values in rq4_round_values.items():
            rq4_summary[round_name] = {
                field: sum(values) / len(values)
                for field, values in field_values.items()
                if values
            }

        report.update({
            "tokens_evaluated": tokens_total,
            "middle_blocks": middle_blocks,
            "per_block": per_block,
            "rq3_expert_overlap": {
                "definition": (
                    "第二轮相对第一轮的专家集合重合：jaccard = "
                    "|A∩B|/|A∪B|；chance_uniform = 均匀随机集合"
                    "的期望 Jaccard（E|A∩B|=k1*k2/E）；"
                    "chance_marginal = 保持第二轮专家边际负载、"
                    "随机化组合的期望 Jaccard；lift = 观测/chance；"
                    "adjusted = (obs−chance)/(1−chance)。"
                    "slot_reuse = 第二轮每个 assignment 的专家落在"
                    "第一轮集合中的比例；branch 级 = 同一分支两轮"
                    "对齐，token 级 = 第二轮跨分支并集对第一轮集合"
                ),
                "summary": rq3_summary,
            },
            "rq4_load_and_entropy": {
                "definition": (
                    "专家负载分布与路由熵：load_entropy 为逐专家"
                    "使用率香农熵（按 ln(n_experts) 归一，1=完全"
                    "均匀）；router_entropy_mean 为路由 softmax "
                    "分布熵；dead_experts 为无 assignment 的专家数"
                ),
                "summary": rq4_summary,
            },
            "branch_divergence": {
                "definition": (
                    "A/B 分支表示的跨分支平均余弦相似度：接近 1 "
                    "说明分支轨迹几乎重合（延迟合并未产生独立"
                    "计算），明显小于 1 说明分支在分化"
                ),
                "summary": {
                    "branch_cosine_mean": (
                        sum(divergence_values)
                        / len(divergence_values)
                        if divergence_values
                        else None
                    ),
                    "blocks_with_branches": len(divergence_values),
                },
            },
        })
        return report



def analyze_routing(
    model: nn.Module,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    split: str = "val",
) -> Dict[str, object]:
    """
    在固定评估窗口上做一次无梯度前向，汇总 RQ3/RQ4 路由诊断。

    split 只允许 val/test；调用方（训练入口）只传 val——路由
    诊断不接触尘封的测试集。返回结构化报告，供 metrics.jsonl、
    config.json 与 routing.json 落盘。
    """
    was_training = model.training
    model.eval()

    core = _unwrap_model(model)
    cfg = core.cfg
    probe = RoutingProbe(cfg, batch_size)
    probe.attach(core)

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            with torch.no_grad():
                for batch_index in range(batches):
                    x, _ = dataset.fixed_eval_batch(
                        split,
                        batch_index,
                        batches,
                        batch_size,
                        device,
                    )
                    if x.size(0) == 0:
                        break
                    core(x)
    finally:
        probe.detach()
        model.train(was_training)

    report = probe.build_report(split=split)
    report["batches_requested"] = batches
    report["batch_size"] = batch_size
    return report



def print_routing_analysis(report: Dict[str, object]) -> None:
    """打印 RQ3/RQ4 路由诊断摘要。"""
    mode = report.get("mode")
    if "skipped" in report:
        print(
            f"[routing] mode={mode} 跳过路由诊断: "
            f"{report.get('skipped')}"
        )
        return

    print(
        f"[routing] mode={mode} split={report.get('split')} "
        f"tokens={report.get('tokens_evaluated')} "
        f"blocks={report.get('middle_blocks')}"
    )

    rq4 = report.get("rq4_load_and_entropy") or {}
    for round_name in ("round1", "round2"):
        summary = (rq4.get("summary") or {}).get(round_name) or {}
        if not summary:
            continue
        print(
            f"[routing] {round_name} 专家负载: "
            "load_entropy="
            f"{_format_finite(summary.get('load_entropy'))} "
            "max_share="
            f"{_format_finite(summary.get('load_max_share'))} "
            f"dead={summary.get('dead_experts')} "
            "router_entropy="
            f"{_format_finite(summary.get('router_entropy_mean'))}"
        )

    rq3 = (report.get("rq3_expert_overlap") or {}).get(
        "summary"
    ) or {}
    if rq3:
        def _show(name: str) -> str:
            value = rq3.get(name)
            if value is None:
                return "n/a"
            return _format_finite(value)

        print(
            "[routing] RQ3 两轮专家重合（branch 级）："
            f"jaccard={_show('second_round_jaccard_mean')} "
            f"exact_match="
            f"{_show('second_round_exact_match_share')} "
            f"slot_reuse="
            f"{_show('second_round_slot_reuse_mean')} | "
            "token 级："
            f"jaccard={_show('token_second_round_jaccard_mean')} "
            f"exact_match="
            f"{_show('token_second_round_exact_match_share')}"
        )
        if rq3.get("branch_self_reuse_share") is not None:
            print(
                "[routing] 分支自身专家在第二轮复现比例（A/B）: "
                f"{_format_finite(rq3.get('branch_self_reuse_share'))}"
            )
        # 机会水平校正：不同 k1/k2 与负载分布下 Jaccard 的
        # 零点不同，必须并列展示。
        print(
            "[routing] RQ3 机会水平："
            f"uniform={_show('jaccard_chance_uniform')} "
            f"marginal={_show('jaccard_chance_marginal')} "
            f"lift_uniform={_show('jaccard_lift_over_uniform')} "
            f"lift_marginal={_show('jaccard_lift_over_marginal')} "
            f"adjusted_marginal="
            f"{_show('jaccard_adjusted_marginal')}"
        )

    divergence = (report.get("branch_divergence") or {}).get(
        "summary"
    ) or {}
    if divergence.get("branch_cosine_mean") is not None:
        print(
            "[routing] 跨分支表示平均余弦相似度: "
            f"{_format_finite(divergence.get('branch_cosine_mean'))}"
            f"（blocks_with_branches="
            f"{divergence.get('blocks_with_branches')}）"
        )



def _routing_shuffle_seed(
    seed: int,
    module_index: int,
) -> int:
    """
    路由打乱干预的稳定子种子。

    只用 (运行 seed, MoE 在模型中的固定序号) 派生，绝不使用
    ``id(module)`` 或对象地址：后者随进程、内存分配和模型
    构造顺序变化，会让同一 ``--seed`` 的 RQ4 干预在不同运行
    中消费不同的随机路由，功能证据不可复现。
    """
    material = (
        f"routing_shuffle_v1|{int(seed)}|{int(module_index)}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(),
        "big",
    )



def _intervention_eval(
    model: nn.Module,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """在固定验证窗口上做一次前向，返回损失与 token 准确率。"""
    core = _unwrap_model(model)
    was_training = core.training
    core.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    try:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            with torch.no_grad():
                for batch_index in range(batches):
                    x, y = dataset.fixed_eval_batch(
                        "val",
                        batch_index,
                        batches,
                        batch_size,
                        device,
                    )
                    if x.size(0) == 0:
                        break
                    logits = core(x)["logits"].float()
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction="sum",
                    )
                    total_loss += float(loss)
                    total_tokens += int(y.numel())
                    correct += int(
                        (logits.argmax(dim=-1) == y).sum()
                    )
    finally:
        core.train(was_training)
    if total_tokens <= 0:
        return {
            "loss": float("nan"),
            "token_accuracy": float("nan"),
            "tokens": 0,
        }
    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": correct / total_tokens,
        "tokens": total_tokens,
    }



def _routing_shuffle_indices(
    n_experts: int,
    shape: torch.Size,
    generator: torch.Generator,
) -> torch.Tensor:
    """为每个 token 无放回抽取 top-k 个随机专家。

    原来的 `torch.randint` 有放回抽样会把同一专家分配给一个
    token 的多个 slot（E=4、k=2 时约 1/4 的行），模型随后把
    重复 slot 合并成单专家满权重，随机路由的分布已经偏离
    「均匀随机 top-k」。这里对每行随机打分再取前 k 个不同的
    专家，保持 assignment 数与均匀权重口径。
    """
    rows = int(shape[0]) if len(shape) > 0 else 0
    slots = int(shape[-1]) if len(shape) > 0 else 0
    if rows <= 0 or slots <= 0:
        return torch.empty(shape, dtype=torch.int64)
    if slots > n_experts:
        raise ValueError(
            "routing_shuffle 的 top-k 不能超过专家数"
        )
    scores = torch.rand(
        (rows, int(n_experts)),
        generator=generator,
    )
    selected = torch.topk(
        scores,
        k=slots,
        dim=-1,
    ).indices
    return selected.reshape(shape)



def analyze_routing_interventions(
    model: nn.Module,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """
    RQ4 的干预验证：负载均匀/熵高本身不能证明专家专门化。

    三类反事实只用验证集固定窗口，不参与选型：

    1. ``routing_shuffle``：把每个 token 的顶层路由替换为均匀
       随机专家（保留 k 个 assignment，权重均匀）。若专家存在
       可区分的功能，随机化路由应显著抬高损失；
    2. ``expert_drop``：逐个把某专家的输出置零（路由/门控保持
       不变），记录损失上升。若专家专门化，不同专家的 drop
       损失上升应有明显差异，而不是所有专家都差不多；
    3. ``baseline``：同一批窗口、同一模型的原损失。

    报告给出损失差、准确率差、专家重要性分布与「最重/最轻
    专家」的比值，并把路由分布诊断降级为「利用率描述」。
    代价是 E+2 次验证前向，默认不启用（``--routing-interventions``）。
    """
    core = _unwrap_model(model)
    cfg = core.cfg
    moe_modules = [
        block.moe
        for block in core.blocks
        if isinstance(getattr(block, "moe", None), SparseMoE)
    ]
    report: Dict[str, object] = {
        "mode": cfg.mode,
        "split": "val",
        "batches_requested": batches,
        "batch_size": batch_size,
        "n_experts": cfg.n_experts,
        "baseline": _intervention_eval(
            core,
            dataset,
            batch_size,
            batches,
            device,
            amp_dtype,
        ),
        "interventions": {
            "routing_shuffle": {},
            "expert_drop": {},
        },
        "notes": [
            "只使用验证集固定窗口；测试集不参与。",
            "routing_shuffle 保持 assignment 数与均匀权重，仅把"
            "专家身份随机化；expert_drop 保持路由与门控，仅把"
            "被屏蔽专家的输出置零。",
            "这是功能干预证据；负载熵/死专家数只是利用率描述，"
            "均匀随机路由也能得到高熵，不能单独证明专门化。",
            "代价：E+2 次验证前向，大 E 时开销明显。",
        ],
    }

    baseline_loss = float(report["baseline"]["loss"])
    baseline_acc = float(report["baseline"]["token_accuracy"])

    # 路由打乱：每个 MoE 用独立、固定的随机流；保留原 k。
    # 种子必须由「运行 seed + 模型内 MoE 序号」派生，不能用
    # id(module)：对象地址随进程/分配顺序变化，同一 --seed 的
    # 干预实验会跨进程得到不同路由，RQ4 的功能证据不可复现。
    def _make_override(module: SparseMoE, module_index: int):
        generator = torch.Generator(device="cpu")
        # 只在创建 override 时播种一次，让随机流在后续每次
        # route 调用（两轮、多个 batch）之间自然推进。原先在
        # 闭包里每调用一次就 manual_seed，所有同形状调用拿到
        # 完全相同的 assignment，round1/round2 与不同 batch 被
        # 人为绑定，随机化失去意义。
        generator.manual_seed(
            _routing_shuffle_seed(seed, module_index)
        )

        def _override(indices, weights, logits):
            new_indices = _routing_shuffle_indices(
                int(module.n_experts),
                indices.shape,
                generator,
            ).to(indices.device)
            new_weights = torch.full_like(
                weights,
                1.0 / max(1, indices.size(-1)),
            )
            return new_indices, new_weights

        return _override

    overrides = {}
    for module_index, module in enumerate(moe_modules):
        overrides[id(module)] = _make_override(
            module,
            module_index,
        )
        module._routing_override = overrides[id(module)]
    try:
        shuffled = _intervention_eval(
            core,
            dataset,
            batch_size,
            batches,
            device,
            amp_dtype,
        )
    finally:
        for module in moe_modules:
            module._routing_override = None
    report["interventions"]["routing_shuffle"] = {
        **shuffled,
        "loss_delta": float(shuffled["loss"]) - baseline_loss,
        "loss_ratio": (
            float(shuffled["loss"]) / baseline_loss
            if baseline_loss > 0
            else None
        ),
        "accuracy_delta": float(shuffled["token_accuracy"]) - baseline_acc,
    }

    # 专家屏蔽：逐个专家置零。E 较大时也只在 val 固定窗口上
    # 做一次前向/专家，成本可控但会明显变慢。
    drop_rows: Dict[str, object] = {}
    drop_deltas: List[float] = []
    for expert_index in range(cfg.n_experts):
        for module in moe_modules:
            module._expert_drop_index = expert_index
        try:
            dropped = _intervention_eval(
                core,
                dataset,
                batch_size,
                batches,
                device,
                amp_dtype,
            )
        finally:
            for module in moe_modules:
                module._expert_drop_index = None
        delta = float(dropped["loss"]) - baseline_loss
        drop_deltas.append(delta)
        drop_rows[str(expert_index)] = {
            **dropped,
            "loss_delta": delta,
        }
    finite_deltas = [
        value for value in drop_deltas if math.isfinite(value)
    ]
    if finite_deltas:
        mean_delta = sum(finite_deltas) / len(finite_deltas)
        variance = sum(
            (value - mean_delta) ** 2 for value in finite_deltas
        ) / len(finite_deltas)
        max_index = max(
            range(len(drop_deltas)),
            key=lambda index: drop_deltas[index],
        )
        min_index = min(
            range(len(drop_deltas)),
            key=lambda index: drop_deltas[index],
        )
        max_delta = drop_deltas[max_index]
        min_delta = drop_deltas[min_index]
        report["interventions"]["expert_drop"] = {
            "per_expert": drop_rows,
            "mean_loss_delta": mean_delta,
            "std_loss_delta": math.sqrt(variance),
            "max_loss_delta": max_delta,
            "max_expert": max_index,
            "min_loss_delta": min_delta,
            "min_expert": min_index,
            "max_over_min_abs_ratio": (
                abs(max_delta) / abs(min_delta)
                if abs(min_delta) > 1e-12
                else None
            ),
            "importance_cv": (
                math.sqrt(variance) / abs(mean_delta)
                if abs(mean_delta) > 1e-12
                else None
            ),
        }
    else:
        report["interventions"]["expert_drop"] = {
            "per_expert": drop_rows,
            "error": "所有专家屏蔽后的损失都不是有限值",
        }
    return report



def print_routing_interventions(report: Dict[str, object]) -> None:
    """打印 RQ4 干预实验摘要。"""
    baseline = report.get("baseline") or {}
    if not baseline or "loss" not in baseline:
        return
    print(
        "[routing-intervention] mode="
        f"{report.get('mode')} baseline_loss="
        f"{_format_finite(baseline.get('loss'), '.6f')}"
    )
    shuffle = (report.get("interventions") or {}).get(
        "routing_shuffle"
    ) or {}
    if shuffle.get("loss_delta") is not None:
        print(
            "[routing-intervention] 路由打乱："
            "Δloss="
            f"{_format_finite(shuffle.get('loss_delta'), '+.6f')} "
            f"ratio={shuffle.get('loss_ratio')} "
            "Δacc="
            f"{_format_finite(shuffle.get('accuracy_delta'), '+.4f')}"
        )
    drop = (report.get("interventions") or {}).get(
        "expert_drop"
    ) or {}
    if drop.get("max_expert") is not None:
        print(
            "[routing-intervention] 专家屏蔽："
            f"max_expert={drop.get('max_expert')} "
            "Δloss="
            f"{_format_finite(drop.get('max_loss_delta'), '+.6f')}；"
            f"min_expert={drop.get('min_expert')} "
            "Δloss="
            f"{_format_finite(drop.get('min_loss_delta'), '+.6f')}；"
            "mean="
            f"{_format_finite(drop.get('mean_loss_delta'), '+.6f')}"
            f"±{_format_finite(drop.get('std_loss_delta'), '.6f')} "
            f"CV={drop.get('importance_cv')}"
        )



# ============================================================
# Non-finite (NaN/Inf) diagnosis
# ============================================================

# 非有限损失/梯度的系统排查路径（顺序固定，先查证据更简单
# 的环节，再查混合精度与优化环节）。
# 诊断时保留的近期有限梯度范数窗口：用失败步之前的梯度水平
# 判断「梯度爆炸/学习率过大」是否有证据。
_NAN_GRAD_TREND_WINDOW = 50


# 失败步梯度范数达到近期中位数的该倍数时，判定为梯度放大明显。
_NAN_GRAD_EXPLOSION_RATIO = 5.0
