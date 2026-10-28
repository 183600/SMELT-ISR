from __future__ import annotations



class SparseMoE(nn.Module):
    """
    Token-level top-k MoE。

    输入:
        x: [batch, time, dim]

    输出:
        merged: [batch, time, dim]
        aux_loss: scalar
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.n_experts = cfg.n_experts
        self.balance_coef = cfg.balance_coef
        self.router = nn.Linear(
            cfg.d_model,
            cfg.n_experts,
            bias=False,
        )

        self.experts = nn.ModuleList([
            Expert(
                cfg.d_model,
                cfg.expert_hidden,
            )
            for _ in range(cfg.n_experts)
        ])

        # 路由诊断探针（RQ3/RQ4）：默认 None，仅在
        # analyze_routing 的诊断前向中挂载。
        self._routing_probe = None

        # 路由干预（RQ4 功能验证）：默认 None。
        #   _routing_override(indices, weights, logits) 返回改写的
        #       (indices, weights)，用于「路由打乱」等反事实；
        #   _expert_drop_index 把指定专家的输出置零，用于
        #       「专家屏蔽」的因果重要性测量。
        # 只在显式启用的评估路径中挂载，训练路径不受影响。
        self._routing_override = None
        self._expert_drop_index: Optional[int] = None

        # 推理路径的零填充缓冲：grad 关闭时按
        # (device, dtype, rows, dim) 取用；训练（grad 开启）时
        # 填充缓冲会经 bmm 进入计算图，必须每次新建。
        self._padded_buffers: Dict[tuple, torch.Tensor] = {}

        # 批量专家路径的权重 stack：键为 "gate_up" / "down"，
        # 值为 (参数版本签名, 失效代数, stack 缓冲)。
        self._weight_stacks: Dict[
            str, Tuple[tuple, int, torch.Tensor]
        ] = {}

        # 失效代数：invalidate_weight_cache 每次递增。
        self._weight_cache_generation = 0

        # 专家权重参数引用。
        self._expert_gate_up_weights: List[torch.Tensor] = []
        self._expert_down_weights: List[torch.Tensor] = []
        self._refresh_expert_weight_lists()

        # dispatch 的行下标 arange：键 (device, n)。
        self._aranges: Dict[
            Tuple[torch.device, int], torch.Tensor
        ] = {}
        self._arange_bytes = 0

    def _refresh_expert_weight_lists(self) -> None:
        self._expert_gate_up_weights = [
            expert.gate_up.weight for expert in self.experts
        ]
        self._expert_down_weights = [
            expert.down.weight for expert in self.experts
        ]

    def _cached_arange(
        self,
        device: torch.device,
        n: int,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        """
        返回 dispatch 的行下标 arange。

        张量只被读取（消费它的算术均为 out-of-place），总量受
        _MOE_ARANGE_CACHE_MAX_BYTES 限制，_apply 迁移设备时
        清空。CUDA 上用 int32 下标，CPU 保持 int64。
        """
        key = (device, dtype, n)
        cached = self._aranges.get(key)
        if cached is not None:
            return cached

        tensor = torch.arange(n, device=device, dtype=dtype)
        bytes_needed = n * tensor.element_size()
        if (
            self._arange_bytes + bytes_needed
            <= _MOE_ARANGE_CACHE_MAX_BYTES
        ):
            self._aranges[key] = tensor
            self._arange_bytes += bytes_needed
        return tensor

    def _apply(self, fn, recurse=True):
        # .to() / .float() 等可能整体替换参数张量（版本号不递增），
        # 相关的 stack、填充缓冲与行下标都必须重建。
        self._weight_stacks.clear()
        self.clear_padded_buffer_cache()
        self._aranges.clear()
        self._arange_bytes = 0
        result = super()._apply(fn, recurse=recurse)
        self._refresh_expert_weight_lists()
        return result

    def clear_padded_buffer_cache(self) -> None:
        """释放当前模块的推理填充缓冲并归还全局额度。"""
        self._padded_buffers.clear()
        _MOE_PADDED_CACHE_REGISTRY.pop(self, None)

    def invalidate_weight_cache(self) -> None:
        """声明 stack 已过期，后续前向在首次使用时刷新。"""
        self._weight_cache_generation += 1

    def _cached_weight_stack(
        self,
        key: str,
        weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        返回 [n_experts, ...] 的专家权重 stack。

        stack 以 (autocast dtype, 版本号, 数据指针) 为签名：普通
        原地更新（如 load_state_dict）会推进版本号并触发刷新；
        但 fused AdamW 的原地更新不会推进版本号，因此训练循环
        在 optimizer.step() 后经 invalidate_weight_cache 递增
        失效代数，首次使用时把新参数写回既有缓冲。参数张量被
        替换或迁移设备时由 _apply 重建。CUDA autocast 下 stack
        使用 autocast dtype。需要反向时经 _StackedExpertWeights
        把梯度按专家散回参数，与「torch.stack + bmm」在 autocast
        下的计算图语义相同。
        """
        first = weights[0]
        target_dtype = self._autocast_target_dtype(first)
        if target_dtype is None:
            target_dtype = first.dtype

        shape = (len(weights), *first.shape)
        signature = (
            target_dtype,
            *(
                (weight._version, weight.data_ptr())
                for weight in weights
            ),
        )

        generation = self._weight_cache_generation
        cached = self._weight_stacks.get(key)

        # 反向路径不能使用 inference_mode 下创建的张量（不能
        # 作为保存张量入图），需要时重建。
        reusable_buffer = cached is not None and (
            not torch.is_grad_enabled()
            or not torch.is_inference(cached[2])
        )

        if (
            reusable_buffer
            and cached[0] == signature
            and cached[1] == generation
        ):
            stacked = cached[2]
        else:
            refreshable = (
                reusable_buffer
                and cached[2].shape == shape
                and cached[2].dtype == target_dtype
                and cached[2].device == first.device
            )

            # inference_mode(False) 保证缓冲不携带 inference 标记，
            # 之后可在 no_grad 与训练路径上原地刷新。
            with torch.inference_mode(False), torch.no_grad():
                if refreshable:
                    stacked = cached[2]
                else:
                    stacked = torch.empty(
                        shape,
                        dtype=target_dtype,
                        device=first.device,
                    )
                # 转换语义与 stack 后 .to() 相同；_foreach_copy_
                # 逐专家拷贝（不可用时回退循环）。
                if hasattr(torch, "_foreach_copy_"):
                    torch._foreach_copy_(
                        list(stacked.unbind(0)),
                        list(weights),
                    )
                else:
                    for index, weight in enumerate(weights):
                        stacked[index].copy_(weight)

            self._weight_stacks[key] = (
                signature,
                generation,
                stacked,
            )

        if not torch.is_grad_enabled() or not any(
            weight.requires_grad for weight in weights
        ):
            return stacked

        return _StackedExpertWeights.apply(stacked, *weights)

    def route(
        self,
        x: torch.Tensor,
        top_k: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat = self._cast_for_autocast(
            x.reshape(-1, x.size(-1))
        )

        router_logits = F.linear(
            flat,
            self.router.weight,
        ).float()

        top_values, top_indices = router_logits.topk(
            top_k,
            dim=-1,
        )

        # top-k 内部的 softmax 等价于完整 softmax 概率的
        # 重新归一化，直接对 top-k logits 求 softmax。
        #
        # top_k == 1 时不能把权重设成常数 1：离散 argmax 的选择
        # 本身不可导，常数权重会让 router 完全收不到语言建模损失
        # 的梯度（只有负载均衡损失还能训练它；balance_coef=0 的
        # 消融臂里 router 就永远停在上一步/初始化）。这里改用
        # Switch Transformer 的口径：取完整 softmax 中选中专家的
        # 概率作为门控权重，概率对 router logits 可导，梯度可以
        # 通过门控缩放回流。
        if top_k == 1:
            full_probs = router_logits.softmax(dim=-1)
            top_weights = full_probs.gather(-1, top_indices)
        else:
            top_weights = top_values.softmax(dim=-1)

        # 路由干预（只用于反事实评估）：在统计 counts / 计算
        # aux loss 之前改写选择，保证后续 dispatch 与负载统计
        # 都使用干预后的 assignment。
        if self._routing_override is not None:
            top_indices, top_weights = self._routing_override(
                top_indices,
                top_weights,
                router_logits,
            )

        # Top-k 版本的 Switch 风格负载均衡损失。
        # f_i 是「被分配给专家 i 的 token 比例」，以 token
        # 数量为分母；均匀路由时 E * sum(f_i * P_i) 恰好等于
        # top_k，均衡强度随路由宽度线性增大，跨 k 比较时
        # 应按比例折算 balance_coef（保持 balance_coef * k 不变）。
        # counts 用 bincount 统计（无 ones_like 分配），输出为
        # int64，后续 dispatch 的索引算术语义与逐 token 计数一致。
        assignment_indices = top_indices.reshape(-1)
        counts = torch.bincount(
            assignment_indices,
            minlength=self.n_experts,
        )

        # 辅助负载均衡项只在梯度开启时可能被消费（总损失、
        # 训练指标）；评估/推理路径（evaluate、measure_efficiency、
        # no_grad 自测）不读取该标量，直接返回零。
        # balance_coef=0 的消融臂仍在 no_grad 下计算同一数值，
        # 保持与训练指标一致。
        if torch.is_grad_enabled():
            if self.balance_coef > 0.0:
                router_probs = router_logits.softmax(dim=-1)
            else:
                with torch.no_grad():
                    router_probs = router_logits.softmax(dim=-1)

            # counts 显式转 fp32，与 soft_load（router_logits 已
            # .float()）保持同一 dtype；归一化就地完成。
            hard_load = counts.float()
            hard_load.div_(flat.size(0))

            soft_load = router_probs.mean(dim=0)

            # 逐专家 f_i * P_i 求和。
            aux_loss = self.n_experts * torch.dot(
                hard_load,
                soft_load,
            )
        else:
            aux_loss = router_logits.new_zeros(())

        # 路由诊断（RQ3/RQ4）：只在显式挂载探针的分析前向中记录；
        # 训练路径上 _routing_probe 为 None。
        if self._routing_probe is not None:
            self._routing_probe.record(
                self,
                x,
                top_indices,
                top_weights,
                router_logits,
            )

        return (
            flat,
            top_indices,
            top_weights,
            counts,
            aux_loss,
        )

    def _autocast_target_dtype(
        self,
        tensor: torch.Tensor,
    ) -> Optional[torch.dtype]:
        """
        CUDA autocast 下返回该张量应转换到的 autocast dtype；
        其余情况返回 None（无需转换）。
        """
        if not tensor.is_cuda or not torch.is_autocast_enabled():
            return None

        try:
            return torch.get_autocast_dtype("cuda")
        except AttributeError:
            return torch.get_autocast_gpu_dtype()

    def _cast_for_autocast(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        CUDA autocast 下把张量转换为 autocast dtype；其余情况
        原样返回。
        """
        dtype = self._autocast_target_dtype(tensor)
        if dtype is None:
            return tensor
        return tensor.to(dtype)

    def _padded_buffer(
        self,
        dtype: torch.dtype,
        device: torch.device,
        rows: int,
        dim: int,
    ) -> Optional[torch.Tensor]:
        """
        grad 关闭时返回填充缓冲，否则返回 None（训练时填充缓冲
        会进入计算图，必须新建）。

        评估路径只读取被 index_copy_ 写入的行（其余行的 bmm
        输出不会被收集），旧缓冲中的陈旧值（含 NaN/Inf）不会
        进入任何结果。
        """
        if torch.is_grad_enabled():
            return None

        key = (device, dtype, rows, dim)
        buffer = self._padded_buffers.get(key)
        if buffer is not None:
            return buffer

        bytes_needed = (
            rows * dim * _dtype_element_size(dtype)
        )
        cached_total = sum(_MOE_PADDED_CACHE_REGISTRY.values())
        if (
            cached_total + bytes_needed
            > _MOE_PADDED_CACHE_MAX_BYTES
        ):
            return None

        # inference_mode(False) 保证缓冲不携带 inference 标记，
        # 可在 inference_mode 与 no_grad 下使用。
        with torch.inference_mode(False), torch.no_grad():
            buffer = torch.zeros(
                rows,
                dim,
                dtype=dtype,
                device=device,
            )

        self._padded_buffers[key] = buffer
        _MOE_PADDED_CACHE_REGISTRY[self] = (
            _MOE_PADDED_CACHE_REGISTRY.get(self, 0) + bytes_needed
        )
        return buffer

    def _batched_experts_forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 [n_experts, rows, dim] 的批量输入执行专家计算。

        各专家权重 stack 成批量矩阵后用 bmm 完成两层 GEMM；
        反向按切片把梯度散回各专家参数，参数与检查点格式不变。
        SwiGLU 语义与 Expert.forward 完全一致。填充行的输入是
        精确的 0：其前向输出为 0，对权重梯度的贡献也恒为 0。
        """
        gate_up_weight = self._cached_weight_stack(
            "gate_up",
            self._expert_gate_up_weights,
        )

        gate_up = torch.bmm(
            inputs,
            gate_up_weight.transpose(1, 2),
        )

        down_weight = self._cached_weight_stack(
            "down",
            self._expert_down_weights,
        )

        if gate_up.requires_grad:
            # SwiGLU 与下投影的组合节点。
            return _SwiGLUDown.apply(gate_up, down_weight)

        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate, inplace=True).mul_(up)

        return torch.bmm(
            hidden,
            down_weight.transpose(1, 2),
        )

    def dispatch(
        self,
        flat: torch.Tensor,
        top_indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        对每个 token-expert assignment 执行专家计算。

        一般路径（n_experts > 1 且 top_k != n_experts）给每个
        专家分配容量 cap（均衡路由下即实际最大桶，倾斜路由
        下为有界的均值上界，见下方注释）：排序后的专家段散射
        进 [n_experts, cap, dim] 的零填充缓冲，经批量 bmm 完成
        容量内的全部专家；超出容量的 assignment 由对应专家的
        2D 前向单独计算。语义与逐专家循环一致（容量内
        为 bmm、溢出为逐专家线性层，均为精确结果）。

        返回:
            [num_tokens, top_k, dim]
        """
        num_tokens, top_k = top_indices.shape
        dim = flat.size(-1)

        # 单专家时所有 token 都路由到同一个专家，
        # 排序等价于恒等排列。
        if self.n_experts == 1:
            expert_flat = self._cast_for_autocast(flat)

            if top_k == 1:
                return self.experts[0].forward(expert_flat).unsqueeze(1)

            # 所有 assignment 指向同一专家：对 [N, dim] 计算后
            # expand 出 slot 维。
            # expand 是 stride=0 视图，后面用 reshape 而不能用 view。
            return self.experts[0].forward(expert_flat).unsqueeze(
                1
            ).expand(
                num_tokens,
                top_k,
                dim,
            )

        # 稠密路由（top_k == n_experts）：每个 token 被分配给
        # 所有专家。所有专家共用同一份输入，expand 出
        # 批量维后经批量 bmm 完成全部专家计算，再按路由下标
        # gather 回 [token, slot] 位置。
        if top_k == self.n_experts:
            expert_flat = self._cast_for_autocast(flat)

            per_expert = self._batched_experts_forward(
                expert_flat.unsqueeze(0).expand(
                    self.n_experts,
                    num_tokens,
                    dim,
                )
            ).transpose(0, 1)
            gather_index = top_indices.unsqueeze(-1).expand(
                num_tokens,
                top_k,
                dim,
            )
            return per_expert.gather(1, gather_index)

        assignment_experts = top_indices.reshape(-1)

        # 收窄仅作用于排序键本身：torch.sort 返回的排序下标
        # 始终是 int64（与其键 dtype 无关），后续 index_select
        # 收集与 index_copy_ 散射的下标不受此处影响；CPU 侧
        # 下标须保持 int64，因此仅在 CUDA 上收窄。
        if assignment_experts.is_cuda:
            assignment_experts = assignment_experts.to(torch.int32)

        # 按 expert id 排序：每个专家的输入在排序后连续，
        # 按段切片分发。同一专家内部的行序不影响结果
        #（专家按行独立计算，散射下标互不重复），因此无需
        # 稳定排序；torch.sort 在相同输入上是确定性的。
        # 排序值即排序后每行所属专家（seg_id）。
        seg_id, perm = torch.sort(assignment_experts)

        # 展平后的 assignment 下标 j 对应 token j // top_k；
        # top_k == 1 时排序下标本身就是 token 下标。
        if top_k == 1:
            sorted_token_ids = perm
        elif top_k == 2:
            # 非负 int64 整除 2 与算术右移等价（精确、向下取整）。
            sorted_token_ids = perm >> 1
        else:
            sorted_token_ids = perm.div(
                top_k,
                rounding_mode="floor",
            )

        # 按排序下标收集输入。autocast 下先转换再收集：
        # 收集结果直接就是目标 dtype，与收集后转换的取值一致。
        gather_source = self._cast_for_autocast(flat)
        gathered = gather_source.index_select(0, sorted_token_ids)

        total = gathered.size(0)
        device = gathered.device
        n_experts = self.n_experts

        # 容量上界：基础容量取均值上界 ceil(total / n_experts)，
        # 再加 1 倍标准差的静态余量（只依赖 assignment 总数）。
        # 实际容量再按本步最大桶收紧，并按 32 对齐。多出的行是
        # 精确 0 填充，不改变任何输出；最大桶超过静态上界时，
        # 下面的逐专家回退路径保证语义不变。
        static_cap = -(-total // n_experts)
        static_cap += math.isqrt(max(1, static_cap)) + 1

        # 各专家段在排序序列中的起始行（exclusive 前缀和）。
        starts = torch.cumsum(counts, dim=0)
        starts.sub_(counts)

        # CUDA 上下标收窄为 int32（与上面的排序键一致）。
        # 行数达到 int32 上限的极端情况仍用 int64，避免溢出。
        index_dtype = (
            torch.int32
            if device.type == "cuda" and total < 2 ** 31
            else torch.int64
        )
        if index_dtype == torch.int32:
            starts = starts.to(index_dtype)

        # 溢出判定只需要看每专家的 assignment 数：同一专家的段在
        # 排序序列中连续，段内局部行号为 0..counts[e]-1，因此
        # 「存在溢出」等价于 counts 的最大值超过容量。
        counts_list = counts.tolist()
        bucket = max(counts_list) if counts_list else 0
        cap = static_cap
        if bucket > 0:
            cap = min(
                static_cap,
                -(-bucket // 32) * 32,
            )
        has_overflow = bucket > cap

        # 行下标。
        row_ids = self._cached_arange(
            device,
            total,
            index_dtype,
        )

        # 每个专家在填充缓冲中的跨度。有溢出时额外留一行
        # scratch：溢出行写入该专家的 scratch 行（有效行不受
        # 影响），其输出随后被回退结果覆盖。
        stride = cap + 1 if has_overflow else cap

        # 目标行号 dest[j] = seg_id[j] * stride + local[j]；
        # 溢出路径把 local 夹到 cap，让溢出行落到 scratch 行。
        if has_overflow:
            local = (
                row_ids
                - starts.index_select(0, seg_id)
            )
            dest = (
                seg_id.to(index_dtype) * stride
                + local.clamp_(max=cap)
            )
        else:
            expert_offsets = (
                self._cached_arange(
                    device,
                    n_experts,
                    index_dtype,
                ) * stride
                - starts
            ).index_select(0, seg_id)
            dest = row_ids + expert_offsets

        padded = self._padded_buffer(
            gathered.dtype,
            device,
            n_experts * stride,
            dim,
        )
        if padded is None:
            padded = gathered.new_zeros(
                n_experts * stride,
                dim,
            )

        padded.index_copy_(0, dest, gathered)

        if not has_overflow:
            # 常规路径到此 gathered 的数据不再被任何一方引用
            #（index_select / index_copy_ 的反向只保存下标）；
            # 溢出路径下方回退仍需读取 gathered，保持引用。
            del gathered

        out = self._batched_experts_forward(
            padded.view(n_experts, stride, dim)
        ).view(n_experts * stride, dim)

        # 散射回原始（未排序）assignment 顺序：perm[j] 是排序
        # 行 j 的原始位置，因此 source[perm[j]] = dest[j]。
        source = dest.new_empty(total)
        source.scatter_(0, perm, dest)
        final = _GatherPaddedExpertOutputs.apply(
            out,
            source,
        )

        # 溢出回退：容量外的 assignment 由对应专家的 2D 前向计算
        # 后写入同一缓冲。排序后每个专家的段连续，溢出行恰为段
        # 尾的 count-cap 行。
        if has_overflow:
            experts = self.experts
            segment_start = 0
            for expert_index, count in enumerate(counts_list):
                if count > cap:
                    rows = slice(
                        segment_start + cap,
                        segment_start + count,
                    )
                    final.index_copy_(
                        0,
                        perm[rows],
                        experts[expert_index].forward(
                            gathered[rows]
                        ),
                    )
                segment_start += count

        return final.view(num_tokens, top_k, dim)

    def unmerged(
        self,
        x: torch.Tensor,
        top_k: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat, top_indices, top_weights, counts, aux_loss = self.route(
            x,
            top_k,
        )

        expert_outputs = self.dispatch(
            flat,
            top_indices,
            counts,
        )

        # dispatch 的 n_experts == 1 分支返回 expand 视图，
        # 此处用 reshape 处理非连续输入。
        expert_outputs = expert_outputs.reshape(
            *x.shape[:-1],
            top_k,
            x.size(-1),
        )

        top_weights = top_weights.view(
            *x.shape[:-1],
            top_k,
        )

        top_indices = top_indices.view(
            *x.shape[:-1],
            top_k,
        )

        # 专家屏蔽干预：把指定专家的输出置零（路由与门控保持
        # 不变），用于测量「该专家对当前 token 的因果贡献」。
        # 不做均值替换，以免引入额外的分布外输入；屏蔽后的
        # 损失上升幅度是专家功能重要性的直接证据。
        if self._expert_drop_index is not None:
            drop = int(self._expert_drop_index)
            mask = (top_indices == drop).to(
                expert_outputs.dtype
            )
            expert_outputs = expert_outputs * (
                1.0 - mask.unsqueeze(-1)
            )

        return (
            expert_outputs,
            top_weights,
            top_indices,
            aux_loss,
        )

    def merge(
        self,
        expert_outputs: torch.Tensor,
        top_weights: torch.Tensor,
        out_shape: torch.Size,
        residual: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        门控加权合并 […, top_k, dim] 的专家输出。

        加权和以逐 slot 的 mul + addcmul 累加实现，
        等价于广播乘后沿 top_k 维求和；dtype 不匹配
        （autocast 残差流精度更高）时同一路径靠类型提升
        合并残差，语义不变。所有路径均为 out-of-place，
        不修改调用方传入的张量。
        """
        top_k = expert_outputs.size(-2)
        dim = expert_outputs.size(-1)

        # top_k == 1 时门控权重是选中专家的完整 softmax 概率
        # （Switch 风格，见 route 注释），不能当成恒等 1.0；
        # 加权和退化为逐行乘概率后再加残差。
        if top_k == 1:
            output = expert_outputs.reshape(-1, dim)
            weights = top_weights.reshape(-1, 1).to(output.dtype)
            output = output * weights

            if residual is not None:
                flat_residual = residual.reshape(-1, dim)

                # dtype 一致时「缩放 + 残差相加」仍保持
                # out-of-place：不修改调用方传入的张量。
                if output.dtype == flat_residual.dtype:
                    if residual_scale != 1.0:
                        output = output.mul(residual_scale)
                    return output.add(flat_residual).view(out_shape)

                return torch.add(
                    flat_residual,
                    output,
                    alpha=residual_scale,
                ).view(out_shape)

            return output.view(out_shape)

        # 权重补出 slot 维并转换 dtype。
        weights = top_weights.to(
            expert_outputs.dtype
        ).unsqueeze(-1)

        # 对 expert_outputs / weights 保持 out-of-place；
        # unbind 出的 slot 视图与逐次 select 等价。
        slots = expert_outputs.unbind(-2)
        weight_slots = weights.unbind(-2)

        merged = slots[0].mul(weight_slots[0])
        for index in range(1, top_k):
            merged.addcmul_(
                slots[index],
                weight_slots[index],
            )

        merged = merged.reshape(out_shape)

        if residual is not None:
            # dtype 一致时原地累加；dtype 不匹配（autocast）时
            # 用 add(alpha=) 完成「缩放 + 残差相加」。
            if merged.dtype == residual.dtype:
                if residual_scale != 1.0:
                    merged.mul_(residual_scale)
                return merged.add_(residual)

            merged = torch.add(
                residual,
                merged,
                alpha=residual_scale,
            )

        return merged

    def forward(
        self,
        x: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_outputs, top_weights, _, aux_loss = self.unmerged(
            x,
            top_k,
        )

        merged = self.merge(
            expert_outputs,
            top_weights,
            x.shape,
        )

        return merged, aux_loss



# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

        self.norm_moe = RMSNorm(cfg.d_model)

        # dense 基线模式：用单个稠密 SwiGLU FFN 替代 SparseMoE，
        # 其余结构（注意力、残差、归一化）与 MoE 模式完全一致，
        # 用于隔离「MoE + 中间层循环」本身带来的收益。
        self.is_dense = cfg.mode == "dense"

        if self.is_dense:
            self.ffn = Expert(
                cfg.d_model,
                cfg.effective_dense_fnn_hidden,
            )
            self.moe = None
        else:
            self.ffn = None
            self.moe = SparseMoE(cfg)

        self.residual_scale = cfg.residual_scale

    def attention_step(
        self,
        x: torch.Tensor,
        scale: Optional[float] = None,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> torch.Tensor:
        # scale 由 LanguageModel 按区域传入：循环区域内使用
        # loop_residual_scale（paper 口径为 residual_scale/r），
        # 区域外与历史行为一致使用 residual_scale。
        scale = self.residual_scale if scale is None else scale
        core = self.attn.core(self.norm_attn(x), attn_mask)

        # dtype 一致（非 autocast，如 CPU/fp32）时以 addmm
        # 计算「投影 + 缩放 + 残差相加」；autocast 下 dtype
        # 不匹配，改用普通投影加残差。
        if core.dtype == x.dtype:
            batch, length, dim = x.shape
            return torch.addmm(
                x.reshape(-1, dim),
                core.reshape(-1, dim),
                self.attn.out.weight.t(),
                beta=1.0,
                alpha=scale,
            ).view(batch, length, dim)

        # add(alpha=) 完成「缩放 + 相加」。
        return torch.add(
            x,
            F.linear(core, self.attn.out.weight),
            alpha=scale,
        )

    def moe_step(
        self,
        x: torch.Tensor,
        top_k: int,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self.residual_scale if scale is None else scale
        # dense 基线：无路由，无辅助损失；top_k 参数被忽略，
        # 保持与 MoE 模式一致的调用签名。
        if self.is_dense:
            y = self.ffn(self.norm_moe(x))
            return (
                torch.add(
                    x,
                    y,
                    alpha=scale,
                ),
                x.new_zeros(()),
            )

        expert_outputs, top_weights, _, aux_loss = (
            self.moe.unmerged(
                self.norm_moe(x),
                top_k,
            )
        )

        x = self.moe.merge(
            expert_outputs,
            top_weights,
            x.shape,
            residual=x,
            residual_scale=scale,
        )
        return x, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        top_k: int,
        scale: Optional[float] = None,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.attention_step(x, scale, attn_mask)
        return self.moe_step(x, top_k, scale)

    def first_branch_step(
        self,
        x: torch.Tensor,
        k1: int,
        scale: Optional[float] = None,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        B 的第一轮第一个 Block。

        输入:
            x: [batch, time, dim]

        输出:
            branches:
                [batch, k1, time, dim]

            gates:
                [batch, time, k1]

            indices:
                [batch, time, k1]

            aux_loss:
                scalar
        """
        scale = self.residual_scale if scale is None else scale
        x = self.attention_step(x, scale, attn_mask)

        expert_outputs, gates, indices, aux_loss = self.moe.unmerged(
            self.norm_moe(x),
            k1,
        )

        # expert_outputs:
        # [batch, time, k1, dim]
        #
        # 每个分支保留完整残差：x + scale * expert。
        # gate 只在这里为 k1==1 预乘：单分支最终收缩是恒等，
        # 不预乘会丢掉 Switch 风格的门控概率。k1>1 时最终
        # 跨分支收缩已经按同一组 gate 加权，这里再乘会让第一
        # 轮专家贡献变成 p_i²·expert（门控被应用两次）。
        branch_experts = expert_outputs
        if int(k1) == 1:
            branch_experts = expert_outputs * gates.to(
                expert_outputs.dtype
            ).unsqueeze(-1)
        branches = torch.add(
            x.unsqueeze(1),
            branch_experts.permute(0, 2, 1, 3),
            alpha=scale,
        )

        return (
            branches,
            gates,
            indices,
            aux_loss,
        )

    def local_two_stage(
        self,
        x: torch.Tensor,
        k1: int,
        k2: int,
        delayed: bool,
        scale: Optional[float] = None,
        second_round_width: Optional[int] = None,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A/local_base/local_base_k2 的局部对照。

        返回值固定为张量三元组 (x, aux1, aux2)，与
        _BlockCheckpoint 的「输出是张量元组」约定一致。

        second_round_width:
            None  —— local_base 的等专家计算量口径：第二轮
                     宽度取 min(n_experts, k1*k2)；
            显式值 —— 机制对照 local_base_k2 取 k2，只改变
                     合并时机，不把第二轮宽度一起改掉。
        """
        scale = self.residual_scale if scale is None else scale
        x = self.attention_step(x, scale, attn_mask)

        if not delayed:
            # 等专家计算量对照：A 在 k1 个分支上各路由 k2 个
            # 专家，单分支的第二轮相应取 min(E, k1*k2) 个专家。
            # 机制对照则固定第二轮为 k2（见 second_round_width）。
            matched_k2 = (
                min(self.moe.n_experts, k1 * k2)
                if second_round_width is None
                else int(second_round_width)
            )
            x, aux1 = self.moe_step(x, k1, scale)
            x, aux2 = self.moe_step(x, matched_k2, scale)
            # 第二轮宽度是计算量对齐的产物；aux loss 约随
            # 路由宽度线性增长（见 SparseMoE.route 注释）。
            # 把 local_base 的 aux2 折算回 A/B 的单分支宽度
            # k2，保证全因子消融中 balance_coef 代表同一均衡
            # 强度，而不是让「立即合并」臂意外多受均衡压力。
            # local_base_k2 的第二轮本来就是 k2，无需折算。
            if (
                second_round_width is None
                and matched_k2 != k2
            ):
                aux2 = aux2 * (
                    float(k2) / float(matched_k2)
                )
            return x, aux1, aux2

        expert_outputs, gates, _, aux1 = self.moe.unmerged(
            self.norm_moe(x),
            k1,
        )

        # k1==1 时最终收缩是恒等，必须在这里乘第一轮门控
        # （top_k=1 时是选中概率，不是 1）；k1>1 时最终收缩
        # 会按同一组 gate 加权，预乘会让第一轮专家贡献变成
        # p_i²·expert（门控被应用两次）。
        if int(k1) == 1:
            expert_outputs = expert_outputs * gates.to(
                expert_outputs.dtype
            ).unsqueeze(-1)

        # 残差并入 expert_outputs：dtype 一致时原地完成
        # x + scale * (gate*expert)；dtype 不匹配时用
        # add(alpha=) 合并。
        if expert_outputs.dtype == x.dtype:
            if scale != 1.0:
                expert_outputs.mul_(
                    scale,
                )
            expert_outputs.add_(
                x.unsqueeze(-2),
            )
            branches = expert_outputs
        else:
            branches = torch.add(
                x.unsqueeze(-2),
                expert_outputs,
                alpha=scale,
            )

        # branches:
        # [batch, time, k1, dim]
        #
        # SparseMoE 支持任意前置维度，因此每个 branch
        # 的 token 会独立经过 Router。
        expert_outputs2, gates2, _, aux2 = self.moe.unmerged(
            self.norm_moe(branches),
            k2,
        )

        branches = self.moe.merge(
            expert_outputs2,
            gates2,
            branches.shape,
            residual=branches,
            residual_scale=scale,
        )

        # 第一轮门控权重在第二轮结束后使用。
        if branches.size(-2) == 1:
            # k1 == 1 时分支已包含完整残差，第一轮专家输出
            # 已按 p1 缩放；不能再乘一次 p1（会缩放残差）。
            x = branches.squeeze(-2)
        else:
            # 跨分支的门控加权和：逐分支 mul + addcmul 累加，
            # 对 branches / gates 保持 out-of-place。
            weights = gates.to(branches.dtype).unsqueeze(-1)

            x = branches.select(-2, 0).mul(
                weights.select(-2, 0)
            )
            for index in range(1, branches.size(-2)):
                x.addcmul_(
                    branches.select(-2, index),
                    weights.select(-2, index),
                )

        return x, aux1, aux2



# ============================================================
# Model
# ============================================================

def _checkpoint_discard(tensor: torch.Tensor):
    """
    前向期间丢弃全部中间激活：占位符不持有数据，前向图在
    _BlockCheckpoint.forward 返回时连同占位符一起释放。
    """
    return None



def _checkpoint_restore(packed):
    # 占位符不会被解包：反向走 _BlockCheckpoint 自己的重算
    # 路径；若被解包说明有代码试图使用被丢弃的前向图。
    raise RuntimeError(
        "_BlockCheckpoint 的中间激活占位符不可解包"
    )



def _autocast_snapshot() -> List[Tuple[str, bool, torch.dtype, bool]]:
    """
    记录当前进程的 autocast 状态（CUDA / CPU 各一份）。

    _BlockCheckpoint 的前向可能在外层 autocast 上下文中执行，
    而反向重算发生在 loss.backward()（autocast 上下文之外）。
    必须按前向时记录的状态重入 autocast，否则 AMP 训练开启
    grad checkpoint 后，重算图走了 fp32 的 dtype 路径，梯度与
    不开启 checkpoint 的同一训练不同（torch.utils.checkpoint
    的非重入实现同样保存并恢复 autocast 状态）。
    """
    states: List[Tuple[str, bool, torch.dtype, bool]] = []

    for device_type in ("cuda", "cpu"):
        try:
            available = torch.amp.is_autocast_available(device_type)
        except AttributeError:
            # 旧版 torch 没有 torch.amp.is_autocast_available。
            available = (
                device_type == "cpu" or torch.cuda.is_available()
            )
        if not available:
            continue

        try:
            enabled = torch.is_autocast_enabled(device_type)
        except TypeError:
            # 旧版 torch 的 is_autocast_enabled 不接受设备参数。
            enabled = (
                torch.is_autocast_cpu_enabled()
                if device_type == "cpu"
                else torch.is_autocast_enabled()
            )

        try:
            dtype = torch.get_autocast_dtype(device_type)
        except AttributeError:
            dtype = (
                torch.get_autocast_cpu_dtype()
                if device_type == "cpu"
                else torch.get_autocast_gpu_dtype()
            )

        states.append(
            (
                device_type,
                enabled,
                dtype,
                torch.is_autocast_cache_enabled(),
            )
        )

    return states



class _BlockCheckpoint(torch.autograd.Function):
    """
    Block 前向的非重入 activation checkpointing。

    与 torch.utils.checkpoint(use_reentrant=False) 语义一致：
    前向在梯度开启下按原样执行——依赖梯度状态的算子（如
    route 的辅助负载均衡损失）取值与直接调用完全一致——
    但经 saved_tensors_hooks 把全部中间激活替换为占位符，
    因此前向不保留任何反向所需的激活；反向用同一输入、
    同一算子顺序重算前向，并把梯度同时回传给输入与重算图
    中的参数叶子。

    所有 Block 内都没有 dropout 等随机算子，输出固定为张量
    元组。
    """

    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.args = args
        ctx.autocast_states = _autocast_snapshot()
        ctx.save_for_backward(
            *(arg for arg in args if torch.is_tensor(arg))
        )

        # autograd.Function.forward 默认在 no_grad 下执行；Block
        # 内存在依赖梯度状态的算子（route 的辅助负载均衡损失
        # 按 torch.is_grad_enabled() 分支返回零），必须显式开启
        # 梯度，否则前向数值与被记录的训练损失和直接调用不一致
        #（torch.utils.checkpoint(use_reentrant=False) 同样如此）。
        with torch.enable_grad(), (
            torch.autograd.graph.saved_tensors_hooks(
                _checkpoint_discard,
                _checkpoint_restore,
            )
        ):
            outputs = fn(*args)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # 与前向图脱钩后只保留数值；requires_grad 由 autograd
        # 依据输入设置，梯度完全来自反向的本节点重算。
        return tuple(
            output.detach()
            if torch.is_tensor(output)
            else output
            for output in outputs
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = iter(ctx.saved_tensors)
        rebuilt: List[object] = []
        grad_flags: List[bool] = []
        inputs: List[torch.Tensor] = []

        # 需要梯度的输入张量重建成新的叶子，其余参数原样传入；
        # 重算路径与直接调用逐位一致。
        for arg in ctx.args:
            if not torch.is_tensor(arg):
                rebuilt.append(arg)
                grad_flags.append(False)
                continue

            tensor = next(saved)
            if tensor.requires_grad:
                tensor = tensor.detach().requires_grad_(True)
                inputs.append(tensor)
                grad_flags.append(True)
            else:
                grad_flags.append(False)
            rebuilt.append(tensor)

        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad())
            # 恢复前向时的 autocast 状态：AMP 下重算图的 dtype
            # 路径必须与前向一致，梯度才与不开启 checkpoint 的
            # 同一训练相同。
            for (
                device_type,
                enabled,
                dtype,
                cache_enabled,
            ) in ctx.autocast_states:
                stack.enter_context(
                    torch.autocast(
                        device_type=device_type,
                        dtype=dtype,
                        enabled=enabled,
                        cache_enabled=cache_enabled,
                    )
                )
            recomputed = ctx.fn(*rebuilt)

        # 反传放在 autocast 上下文之外：与直接调用路径一致
        # （训练循环的 loss.backward() 就在 autocast 之外执行），
        # 也与 torch.utils.checkpoint 的非重入实现一致——重算在
        # autocast 内、重算图的反向在 autocast 外。
        if not isinstance(recomputed, tuple):
            recomputed = (recomputed,)

        # 只对「需要梯度且收到了上游梯度」的输出反传；其余
        # 输出（如整数下标张量）没有梯度项。
        pairs = [
            (output, grad)
            for output, grad in zip(recomputed, grad_outputs)
            if grad is not None
            and torch.is_tensor(output)
            and output.requires_grad
        ]

        if pairs:
            torch.autograd.backward(
                [output for output, _ in pairs],
                grad_tensors=[grad for _, grad in pairs],
            )

        grad_iter = iter(inputs)
        arg_grads = tuple(
            next(grad_iter).grad if flag else None
            for flag in grad_flags
        )

        return (None,) + arg_grads



def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    """张量内容的稳定 sha256（先转 CPU fp32 连续布局）。"""
    array = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    try:
        payload = array.numpy().tobytes()
    except (TypeError, RuntimeError):
        payload = bytes(
            array.reshape(-1).view(torch.uint8).tolist()
        )
    return hashlib.sha256(payload).hexdigest()
