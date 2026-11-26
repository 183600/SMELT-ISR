from __future__ import annotations



def document_segment_ids(
    input_ids: torch.Tensor,
    separator: Sequence[int],
) -> torch.Tensor:
    """从 token 流中检测文档分隔符，返回每 token 的文档编号。

    分隔符的起始字节处开启新文档（文档 0 从窗口起点开始）。
    多字节分隔符按完整字节串匹配；窗口边界处被截断的分隔符
    不计入（窗口外的文档边界本来就不影响窗口内可见性）。
    """

    batch, length = input_ids.shape
    byte_values = [int(byte) for byte in separator]
    if not byte_values or length <= 0:
        return torch.zeros(
            (batch, length),
            dtype=torch.long,
            device=input_ids.device,
        )
    span = length - len(byte_values) + 1
    if span <= 0:
        return torch.zeros(
            (batch, length),
            dtype=torch.long,
            device=input_ids.device,
        )
    starts = torch.ones(
        (batch, span),
        dtype=torch.bool,
        device=input_ids.device,
    )
    for offset, value in enumerate(byte_values):
        starts &= (
            input_ids[:, offset:offset + span] == value
        )
    separator_bytes = bytes(byte_values)
    # 分隔符自身有非空 border（如 "\n\n"、"aba"）时，候选匹配
    # 可能相互重叠；群组扫描（bytes.find）采用不重叠的贪心
    # 匹配，文档 mask 必须同口径，否则会在同一群组内部凭空
    # 插入边界。无 border 时任意两个匹配都不可能重叠，继续
    # 走全向量化路径。
    self_overlapping = any(
        separator_bytes[:border]
        == separator_bytes[-border:]
        for border in range(1, len(separator_bytes))
    )
    if self_overlapping and bool(starts.any()):
        starts_cpu = starts.cpu()
        selected_cpu = torch.zeros_like(starts_cpu)
        for row in range(batch):
            next_allowed = 0
            for position in torch.nonzero(
                starts_cpu[row], as_tuple=False
            ).flatten().tolist():
                if position < next_allowed:
                    continue
                selected_cpu[row, position] = True
                next_allowed = position + len(byte_values)
        starts = selected_cpu.to(input_ids.device)
    boundary = torch.zeros(
        (batch, length),
        dtype=torch.long,
        device=input_ids.device,
    )
    boundary[:, :span] = starts.to(torch.long)
    # 窗口首个 token 即使匹配分隔符也属于本窗口的第一篇
    # 文档（编号从 0 起），因此把第一个位置清零。
    boundary[:, 0] = 0
    return torch.cumsum(boundary, dim=1)



class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.effective_n_kv_heads
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # n_kv_heads == n_heads 时保留融合 qkv 参数（历史检查点
        # 兼容，且 MHA 时更省一次 GEMM）；GQA 时 K/V 的投影宽度
        # 小于 Q，必须分开投影。
        if self.n_kv_heads == self.n_heads:
            self.qkv = nn.Linear(
                cfg.d_model,
                3 * cfg.d_model,
                bias=False,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = nn.Linear(
                cfg.d_model,
                cfg.d_model,
                bias=False,
            )
            self.k_proj = nn.Linear(
                cfg.d_model,
                self.kv_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                cfg.d_model,
                self.kv_dim,
                bias=False,
            )

        self.out = nn.Linear(
            cfg.d_model,
            cfg.d_model,
            bias=False,
        )

    def core(
        self,
        x: torch.Tensor,
        attn_mask: Optional["SegmentAttentionMask"] = None,
    ) -> torch.Tensor:
        """
        Attention 主体，返回输出投影之前的 [batch, length, dim]。
        调用方负责完成输出投影与残差相加。
        """
        batch, length, dim = x.shape

        if self.qkv is not None:
            q, k, v = F.linear(x, self.qkv.weight).view(
                batch, length, 3, self.n_heads, self.head_dim
            ).unbind(2)
        else:
            q = self.q_proj(x).view(
                batch,
                length,
                self.n_heads,
                self.head_dim,
            )
            k = self.k_proj(x).view(
                batch,
                length,
                self.n_kv_heads,
                self.head_dim,
            )
            v = self.v_proj(x).view(
                batch,
                length,
                self.n_kv_heads,
                self.head_dim,
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA：整除时每个 KV 头被 kv_repeat 个连续 Q 头共享；
        # 不整除时按 floor(i * n_kv / n_heads) 均匀映射，保证
        # 任意 n_kv_heads 都能精确控制 KV cache（论文未披露
        # head 几何，预算匹配需要这一自由度）。
        if self.kv_repeat > 1 and self.n_heads % self.n_kv_heads == 0:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
        elif self.n_kv_heads != self.n_heads:
            index = (
                (torch.arange(self.n_heads) * self.n_kv_heads)
                // self.n_heads
            ).to(device=k.device)
            k = k.index_select(1, index)
            v = v.index_select(1, index)

        if attn_mask is not None:
            # 文档级 segment mask 已包含因果约束；走掩码路径时
            # 不能再叠加 is_causal=True（否则两者不一致）。
            y = attn_mask.attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous()
        return y.view(batch, length, dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> torch.Tensor:
        return F.linear(
            self.core(x, attn_mask), self.out.weight
        )



def _swiglu_backward(
    grad_hidden: torch.Tensor,
    gate_up: torch.Tensor,
    return_hidden: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    SwiGLU 的共用反向实现（供 _SwiGLUDown 使用）。

    gate_up 是 [.., 2*hidden] 的拼接张量（gate 在前、up 在
    后）。mul 反向给出两路梯度，gate 路再过 silu 反向；两路
    梯度分别写回 grad_gate_up 的对应半区（两半区互不相交且
    并集完整）。

    ``return_hidden=True`` 时同时返回由同一份 ``silu(gate)``
    重算的隐藏层，供下投影权重梯度使用。
    """
    gate, up = gate_up.chunk(2, dim=-1)

    grad_gate_up = torch.empty_like(gate_up)
    grad_gate, grad_up = grad_gate_up.chunk(2, dim=-1)

    # grad_up 半区先写入 grad_hidden * up 作为 silu 反向的输入。
    torch.mul(grad_hidden, up, out=grad_up)
    torch.ops.aten.silu_backward.grad_input(
        grad_up,
        gate,
        grad_input=grad_gate,
    )

    # grad_up = grad_hidden * silu(gate)。silu(gate) 先原地
    # 写入 grad_up 半区；需要隐藏层时在同一份 silu 结果上
    # 乘 up 得到 hidden。
    torch.ops.aten.silu.out(gate, out=grad_up)
    hidden = None
    if return_hidden:
        hidden = torch.mul(grad_up, up)
    torch.mul(grad_hidden, grad_up, out=grad_up)
    if return_hidden:
        return grad_gate_up, hidden
    return grad_gate_up



class _SwiGLUDown(torch.autograd.Function):
    """
    SwiGLU 与下投影的组合节点。

    前向与「silu(gate) * up 后接下投影」逐位一致；hidden
    中间结果不进入 autograd 的保存集合，反向时由 gate_up
    按前向同一算子重新算出。
    down_weight 支持两种布局：
    - [E, D, H]：批量专家，下投影为 bmm；
    - [D, H]：单专家，下投影为普通线性。
    """

    @staticmethod
    def forward(
        ctx,
        gate_up: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        # gate/up 是 gate_up 的切片视图，保存 gate_up 即可同时
        # 覆盖两者且供反向重算隐藏层。forward 内 grad mode 默认
        # 关闭，因此以 requires_grad 判断是否需要保存。
        gate, up = gate_up.chunk(2, dim=-1)

        if gate_up.requires_grad:
            ctx.save_for_backward(gate_up, down_weight)

            # gate_up 需要保存供反向使用，SiLU 保持 out-of-place，
            # mul_ 原地写入其输出。
            hidden = F.silu(gate).mul_(up)
        else:
            # 无需反向时 gate_up 只是临时张量，SiLU 直接原地
            # 写入 gate（gate_up 的前半区），数值上与上面一致。
            hidden = F.silu(gate, inplace=True).mul_(up)

        # autocast 下单专家路径的权重与激活可能不同 dtype；与
        # 线性层的 autocast 语义一致，先按激活 dtype 对齐权重
        #（批量路径的权重 stack 本就缓存为 autocast dtype，
        # 此处为空操作）。
        if down_weight.dtype != hidden.dtype:
            down_weight = down_weight.to(hidden.dtype)

        if down_weight.dim() == 3:
            return torch.bmm(hidden, down_weight.transpose(1, 2))
        return F.linear(hidden, down_weight)

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple:
        gate_up, down_weight = ctx.saved_tensors

        # autocast 下单专家路径的权重与激活可能不同 dtype，计算
        # 前对齐，返回梯度时再还原为权重的 dtype。
        compute_weight = down_weight
        if compute_weight.dtype != grad_out.dtype:
            compute_weight = compute_weight.to(grad_out.dtype)

        if down_weight.dim() == 3:
            # 与 bmm(hidden, W^T) 的反向同序：grad_hidden 来自
            # grad_out @ W；grad_W 来自 hidden^T @ grad_out 的
            # 转置视图。
            grad_hidden = torch.bmm(grad_out, compute_weight)
            grad_gate_up, hidden = _swiglu_backward(
                grad_hidden,
                gate_up,
                return_hidden=True,
            )
            grad_weight = torch.bmm(
                hidden.transpose(1, 2),
                grad_out,
            ).transpose(1, 2)
        else:
            # 与 F.linear(hidden, W) 的反向同序；输入允许带任意
            # 前导维（如 dense 模式的 [batch, time, dim]），权重的
            # 梯度按全部前导维求和，与线性层反向的展平口径一致。
            grad_hidden = torch.matmul(grad_out, compute_weight)
            grad_gate_up, hidden = _swiglu_backward(
                grad_hidden,
                gate_up,
                return_hidden=True,
            )
            grad_weight = torch.matmul(
                grad_out.reshape(
                    -1,
                    grad_out.size(-1),
                ).transpose(0, 1),
                hidden.reshape(-1, hidden.size(-1)),
            )

        if grad_weight.dtype != down_weight.dtype:
            grad_weight = grad_weight.to(down_weight.dtype)

        return grad_gate_up, grad_weight



class Expert(nn.Module):
    """
    SwiGLU Expert.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()

        # [dim, 2*hidden] 线性层在最后一维按 [gate; up] 切分，
        # 每个输出仍是独立的点积。
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate 与 up 的融合输出直接进入 SwiGLU + 下投影的组合节点。
        gate_up = F.linear(x, self.gate_up.weight)

        # 无反向需求（no_grad/inference_mode 下的评估路径）时直接
        # 原地计算，数值上与组合节点的前向完全一致。
        if gate_up.requires_grad:
            return _SwiGLUDown.apply(gate_up, self.down.weight)

        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate, inplace=True).mul_(up)
        return F.linear(hidden, self.down.weight)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args,
        **kwargs,
    ):
        # 分开保存的 gate.weight / up.weight 按 [gate; up] 顺序
        # 拼接后并入 gate_up.weight，与 forward 中 chunk 的
        # 切分顺序一致。
        old_gate = prefix + "gate.weight"
        old_up = prefix + "up.weight"

        if old_gate in state_dict and old_up in state_dict:
            state_dict[prefix + "gate_up.weight"] = torch.cat(
                [
                    state_dict.pop(old_gate),
                    state_dict.pop(old_up),
                ],
                dim=0,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            *args,
            **kwargs,
        )



# ============================================================
# Sparse MoE
# ============================================================

# 填充缓冲的全局额度。
_MOE_PADDED_CACHE_MAX_BYTES = 64 * 1024 * 1024


# 填充缓冲账本：{模块: 已用字节数}。
_MOE_PADDED_CACHE_REGISTRY: (
    "weakref.WeakKeyDictionary[object, int]"
) = weakref.WeakKeyDictionary()



# dispatch 行下标的额度（每模块）。
_MOE_ARANGE_CACHE_MAX_BYTES = 1024 * 1024



@functools.lru_cache(maxsize=None)
def _dtype_element_size(dtype: torch.dtype) -> int:
    """dtype 的每元素字节数。"""
    return torch.empty(0, dtype=dtype).element_size()



class _StackedExpertWeights(torch.autograd.Function):
    """
    专家权重 stack 的梯度出口。

    forward 原样返回 stack；backward 按专家把上游梯度切成
    逐专家切片返回，与 torch.stack 反向的语义一致。autocast
    下 stack 为 autocast dtype（bf16/fp16），反向先把上游梯度
    转换回参数 dtype 再切分，与「参数 -> 转换 -> stack」
    计算图中转换节点反向的数值完全一致。
    """

    @staticmethod
    def forward(ctx, stacked: torch.Tensor, *weights: torch.Tensor):
        # 逐专家切片的形状；stack 要求各专家权重同形。
        ctx.shapes = [weight.shape for weight in weights]
        # 参数 dtype；stack 可能已是 autocast dtype。
        ctx.weight_dtype = (
            weights[0].dtype if weights else stacked.dtype
        )
        return stacked

    @staticmethod
    def backward(ctx, grad_stacked: torch.Tensor) -> tuple:
        if grad_stacked.dtype != ctx.weight_dtype:
            grad_stacked = grad_stacked.to(ctx.weight_dtype)

        return (None,) + tuple(
            grad_stacked[index].reshape(shape)
            for index, shape in enumerate(ctx.shapes)
        )



class _MergeExpertOutputs(torch.autograd.Function):
    """
    dispatch 的输出侧：把各专家输出散射回原始 assignment 顺序。

    forward 把各专家输出写回原始 assignment 顺序的缓冲；
    backward 按 perm 收集梯度，再按计数切成各专家输出的
    梯度视图。
    """

    @staticmethod
    def forward(
        ctx,
        perm: torch.Tensor,
        counts: list,
        *expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.counts = counts

        if any(t.requires_grad for t in expert_outputs):
            ctx.save_for_backward(perm)

        first = expert_outputs[0]

        # 输出缓冲按非空专家输出分配 dtype；空段是 dispatch
        # 输入的空切片占位，不参与 dtype 判定。
        output_dtype = next(
            (
                tensor.dtype
                for tensor, count in zip(expert_outputs, counts)
                if count > 0
            ),
            first.dtype,
        )

        outputs = torch.empty(
            perm.numel(),
            first.size(-1),
            dtype=output_dtype,
            device=first.device,
        )

        # 散射下标按计数切分，与各专家输出、计数对齐；
        # 空段仍按计数占位，不产生 index_copy_。
        for out, index, count in zip(
            expert_outputs,
            perm.split(counts),
            counts,
        ):
            if count == 0:
                continue

            outputs.index_copy_(0, index, out)

        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_outputs: torch.Tensor,
    ) -> tuple:
        (perm,) = ctx.saved_tensors

        grad_sorted = grad_outputs.index_select(0, perm)

        segment_grads = grad_sorted.split(ctx.counts, dim=0)

        return (None, None) + tuple(segment_grads)



class _GatherPaddedExpertOutputs(torch.autograd.Function):
    """
    把填充缓冲中的各专家有效输出按原始 assignment 顺序取回。

    ``source[a]`` 是原始 assignment a 在填充缓冲中的行号；
    反向按同一映射把上游梯度写回填充缓冲，未使用的填充行
    梯度恒为 0。溢出路径中 ``source`` 的溢出行指向专家
    scratch 行，这些行的上游梯度在回退结果覆盖后恒为 0。
    """

    @staticmethod
    def forward(
        ctx,
        padded: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        ctx.shape = padded.shape
        ctx.dtype = padded.dtype
        ctx.device = padded.device

        final = padded.index_select(0, source)

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(source)
        return final

    @staticmethod
    def backward(
        ctx,
        grad_final: torch.Tensor,
    ) -> tuple:
        (source,) = ctx.saved_tensors

        grad_padded = torch.zeros(
            ctx.shape,
            dtype=ctx.dtype,
            device=ctx.device,
        )
        grad_padded.index_copy_(0, source, grad_final)

        return grad_padded, None
