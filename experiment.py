from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    vocab_size: int = 256
    seq_len: int = 128

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 8

    n_experts: int = 8
    expert_hidden: int = 256

    # SMELT 每轮的 top-k
    top_k: int = 2

    # 延迟合并模型
    k1: int = 2
    k2: int = 2

    # smelt / local_base / A / B
    mode: str = "B"

    balance_coef: float = 0.01
    residual_scale: float = 1.0

    def validate(self):
        if self.mode not in {"smelt", "local_base", "A", "B"}:
            raise ValueError(f"未知 mode: {self.mode}")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正整数")
        if self.seq_len <= 0:
            raise ValueError("seq_len 必须为正整数")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被正的 n_heads 整除")
        if self.n_layers < 4 or self.n_layers % 4 != 0:
            raise ValueError("n_layers 必须是大于等于 4 的 4 的倍数")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k 必须满足 1 <= top_k <= n_experts")
        if not 1 <= self.k1 <= self.n_experts:
            raise ValueError("k1 必须满足 1 <= k1 <= n_experts")
        if not 1 <= self.k2 <= self.n_experts:
            raise ValueError("k2 必须满足 1 <= k2 <= n_experts")

    @property
    def middle_start(self) -> int:
        return self.n_layers // 4

    @property
    def middle_end(self) -> int:
        return 3 * self.n_layers // 4


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 要求 PyTorch 优先选用确定性实现。bincount、attention
    # 反向等 CUDA 算子若不存在确定性实现会明确告警，而不是
    # 静默产生跨重启不可复现的微小差异。
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 本文件所有 torch.empty 缓冲在使用前都会被完整写入，
    # 不依赖未初始化内存的 NaN 填充来保证结果确定性。
    torch.utils.deterministic.fill_uninitialized_memory = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        if self._fused is not None:
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


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(
            cfg.d_model,
            3 * cfg.d_model,
            bias=False,
        )
        self.out = nn.Linear(
            cfg.d_model,
            cfg.d_model,
            bias=False,
        )

    def core(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention 主体，返回输出投影之前的 [batch, length, dim]。
        调用方负责完成输出投影与残差相加。
        """
        batch, length, dim = x.shape

        q, k, v = self.qkv(x).view(
            batch, length, 3, self.n_heads, self.head_dim
        ).unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous()
        return y.view(batch, length, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.core(x))


class _SwiGLU(torch.autograd.Function):
    """
    对 [.., 2*hidden] 的线性输出做 SwiGLU：
    前一半过 SiLU 后与后一半相乘。
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor) -> torch.Tensor:
        gate, up = h.chunk(2, dim=-1)

        if h.requires_grad:
            # gate/up 是 h 的切片视图，保存 h 即可同时覆盖两者。
            # forward 内 grad mode 默认关闭，不能用
            # torch.is_grad_enabled() 判断是否需要保存；
            # requires_grad 为 False 时不会建反向节点。
            ctx.save_for_backward(h)

        # mul_ 原地写入 SiLU 的输出；SiLU 本身必须保持
        # out-of-place（其反向依赖输入或输出缓冲，若再做
        # in-place 会破坏 saved tensor）。
        return F.silu(gate).mul_(up)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (h,) = ctx.saved_tensors
        gate, up = h.chunk(2, dim=-1)

        # 与逐算子反向相同的算子序列：mul 反向给出两路梯度，
        # gate 路再过 silu 反向；两路梯度分别直接写回融合输出的
        # 对应半区（两半区互不重叠且并集完整，空缓冲会被完整覆盖）。
        # silu_backward 的 out= 变体与函数式版本逐位一致。
        grad_h = torch.empty_like(h)
        grad_h_gate, grad_h_up = grad_h.chunk(2, dim=-1)

        # grad_up 半区先写入 grad_out * up 作为 silu 反向的输入，
        # 该半区最终会被 grad_up 完整覆盖。
        torch.mul(grad_out, up, out=grad_h_up)
        torch.ops.aten.silu_backward.grad_input(
            grad_h_up,
            gate,
            grad_input=grad_h_gate,
        )

        # grad_up = grad_out * silu(gate)：σ(gate) 与 gate 相乘得到
        # silu(gate)，与 F.silu 的 x * σ(x) 运算序列和舍入一致；
        # 最后 grad_out * silu 的乘法结合顺序相同，数值逐位一致。
        torch.sigmoid(gate, out=grad_h_up)
        torch.mul(gate, grad_h_up, out=grad_h_up)
        torch.mul(grad_out, grad_h_up, out=grad_h_up)
        return grad_h


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
        # gate 与 up 的融合输出直接进入 SwiGLU。
        return self.down(_SwiGLU.apply(self.gate_up(x)))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args,
        **kwargs,
    ):
        # 兼容旧检查点中分开保存的 gate.weight / up.weight：
        # 按 [gate; up] 顺序拼接后并入 gate_up.weight，
        # 与 forward 中 chunk 的切分顺序一致。
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

class _SplitExpertInputs(torch.autograd.Function):
    """
    dispatch 的输入侧：按每专家计数切分排序后的 assignment。

    forward 等价于 index_select 后按 counts split；
    backward 把各段梯度直接 index_add 回 flat。各段互不
    重叠且并集完整，累加顺序与「先还原整块梯度再散射」
    一致，数值逐位相同。
    """

    @staticmethod
    def forward(
        ctx,
        flat: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        counts: list,
    ) -> Tuple[torch.Tensor, ...]:
        ctx.counts = counts
        ctx.flat_shape = flat.shape

        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(sorted_token_ids)

        gathered = flat.index_select(0, sorted_token_ids)
        return tuple(gathered.split(counts, dim=0))

    @staticmethod
    def backward(
        ctx,
        *segment_grads: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None]:
        (sorted_token_ids,) = ctx.saved_tensors

        grad_flat = None
        start = 0

        for grad, count in zip(segment_grads, ctx.counts):
            if count == 0:
                continue

            end = start + count

            if grad_flat is None:
                grad_flat = torch.zeros(
                    ctx.flat_shape,
                    dtype=grad.dtype,
                    device=grad.device,
                )

            grad_flat.index_add_(
                0,
                sorted_token_ids[start:end],
                grad,
            )
            start = end

        return grad_flat, None, None


class _MergeExpertOutputs(torch.autograd.Function):
    """
    dispatch 的输出侧：把各专家输出散射回原始 assignment 顺序。

    forward 与逐专家 index_copy_ 写入同一缓冲一致；
    backward 只需一次按 perm 的 index_select，再按计数切成
    各专家输出的梯度视图。
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

        # 输出缓冲按非空专家的 dtype 分配：空段占位张量来自
        # fp32 残差流的切片，不代表专家输出的真实 dtype。
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

        start = 0
        for out, count in zip(expert_outputs, counts):
            if count == 0:
                continue

            end = start + count
            outputs.index_copy_(0, perm[start:end], out)
            start = end

        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_outputs: torch.Tensor,
    ) -> Tuple[None, None, Tuple[torch.Tensor, ...]]:
        (perm,) = ctx.saved_tensors

        grad_sorted = grad_outputs.index_select(0, perm)

        segment_grads = grad_sorted.split(ctx.counts, dim=0)

        return (None, None) + tuple(segment_grads)


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

        # counts 回读的暂存与完成事件。
        self._counts_stage = None
        self._counts_event = None

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
        flat = x.reshape(-1, x.size(-1))

        router_logits = self.router(flat).float()
        router_probs = router_logits.softmax(dim=-1)

        top_values, top_indices = router_logits.topk(
            top_k,
            dim=-1,
        )

        # top-k 内部的 softmax 等价于完整 softmax 概率的
        # 重新归一化，直接对 top-k logits 求 softmax。
        # top_k == 1 时单元素 softmax 恒为 1.0，且其反向
        # w*(1-w) 按位为 0；直接以 ones 构造与 softmax 路径的
        # 前向数值和参数梯度逐位一致。
        if top_k == 1:
            top_weights = torch.ones_like(top_values)
        else:
            top_weights = top_values.softmax(dim=-1)

        # Top-k 版本的 Switch 风格负载均衡损失。
        # f_i 是「被分配给专家 i 的 token 比例」，必须以 token
        # 数量而不是 assignment 数量为分母；否则 top-k 改变时
        # 同一 balance_coef 对应的实际均衡强度会随 k 成反比变化，
        # 使不同 k 配置的实验不再可比。
        counts = torch.bincount(
            top_indices.reshape(-1),
            minlength=self.n_experts,
        )

        hard_load = counts.to(router_probs.dtype) / flat.size(0)

        soft_load = router_probs.mean(dim=0)

        aux_loss = self.n_experts * (
            hard_load * soft_load
        ).sum()

        return (
            flat,
            top_indices,
            top_weights,
            counts,
            aux_loss,
        )

    def _issue_counts_readback(self, counts: torch.Tensor) -> None:
        """
        启动 counts 回读：先拷入暂存并记录完成事件，
        到需要逐专家计数时再读取；与 _read_counts 配对使用。
        """
        if not counts.is_cuda:
            return

        if self._counts_stage is None:
            with torch.inference_mode(False):
                self._counts_stage = torch.empty(
                    self.n_experts,
                    dtype=counts.dtype,
                    pin_memory=True,
                )
            self._counts_event = torch.cuda.Event()

        with torch.cuda.device(counts.device):
            self._counts_stage.copy_(counts, non_blocking=True)
            self._counts_event.record()

    def _read_counts(self, counts: torch.Tensor) -> list:
        if counts.is_cuda:
            self._counts_event.synchronize()
            return self._counts_stage.tolist()
        return counts.tolist()

    def dispatch(
        self,
        flat: torch.Tensor,
        top_indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        对每个 token-expert assignment 执行专家计算。

        返回:
            [num_tokens, top_k, dim]
        """
        num_tokens, top_k = top_indices.shape
        dim = flat.size(-1)

        # 单专家时所有 token 都路由到同一个专家，排序等价于
        # 恒等排列；直接对 repeat_interleave 展开后的输入跑
        # 一次专家即可，数值与排序路径逐位一致。
        if self.n_experts == 1:
            # top_k == 1 时 repeat_interleave 是纯拷贝，
            # 直接跑一次专家再补回 slot 维，数值逐位一致。
            if top_k == 1:
                return self.experts[0](flat).unsqueeze(1)

            # 所有 assignment 指向同一专家：对 [N, dim] 计算
            # 一次再 expand 出 slot 维，数值逐位一致。
            # expand 只是 stride=0 视图，后面用 reshape
            # （按需物化）而不能用 view。
            return self.experts[0](flat).unsqueeze(1).expand(
                num_tokens,
                top_k,
                dim,
            )

        # 稠密路由（top_k == n_experts）：每个 token 被分配给
        # 所有专家，分组排序退化为恒等。让每位专家处理完整输入，
        # 再按路由下标 gather 回 [token, slot] 位置，
        # 数值与排序路径逐位一致。
        if top_k == self.n_experts:
            per_expert = torch.stack(
                [expert(flat) for expert in self.experts],
                dim=1,
            )
            gather_index = top_indices.unsqueeze(-1).expand(
                num_tokens,
                top_k,
                dim,
            )
            return per_expert.gather(1, gather_index)

        assignment_experts = top_indices.reshape(-1)

        self._issue_counts_readback(counts)

        # 按 expert id 做稳定排序：
        # 每个专家的输入在排序后连续，按段切片即可分发；
        # 稳定排序保持同一专家内 token 顺序不变，
        # 数值结果与逐专家索引完全一致。
        perm = torch.argsort(
            assignment_experts,
            stable=True,
        )

        # 展平后的 assignment 下标 j 对应 token j // top_k；
        # top_k == 1 时排序下标本身就是 token 下标。
        if top_k == 1:
            sorted_token_ids = perm
        else:
            sorted_token_ids = perm.div(
                top_k,
                rounding_mode="floor",
            )

        counts = self._read_counts(counts)

        # 每专家连续输入段由单个自定义节点切出，反向按段
        # 直接 index_add 回 flat。
        segments = _SplitExpertInputs.apply(
            flat,
            sorted_token_ids,
            counts,
        )

        # 逐专家计算其连续输入段；空段原样占位，不产生
        # 专家调用。
        expert_outputs = [
            (
                expert(segment)
                if count > 0
                else segment
            )
            for count, expert, segment in zip(
                counts,
                self.experts,
                segments,
            )
        ]

        # 各专家输出由单个自定义节点散射回原始（未排序）
        # 位置；各段互不重叠且并集恰好覆盖全部 assignment，
        # 空缓冲区会被完整写入。
        return _MergeExpertOutputs.apply(
            perm,
            counts,
            *expert_outputs,
        ).view(
            num_tokens,
            top_k,
            dim,
        )

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
        # 此处用 reshape 兼容非连续输入。
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

        加权和以广播乘 + 沿 top_k 维求和实现；dtype 不匹配
        （autocast 残差流精度更高）时同一路径靠类型提升
        合并残差，语义不变。
        """
        top_k = expert_outputs.size(-2)
        dim = expert_outputs.size(-1)

        # top_k == 1 时门控权重来自单元素 softmax，
        # 恒等于 1.0（精确值），其对 router logits 的反向
        # 梯度也恒为 0（softmax 反向的 w*(1-w) 按位为 0），
        # 加权和退化为恒等；
        # 有残差时用 add(alpha=) 完成「缩放 + 残差相加」。
        if top_k == 1:
            output = expert_outputs.reshape(-1, dim)

            if residual is not None:
                return torch.add(
                    residual.reshape(-1, dim),
                    output,
                    alpha=residual_scale,
                ).view(out_shape)

            return output.view(out_shape)

        weights = top_weights.to(expert_outputs.dtype)

        # 必须保持对 expert_outputs / weights 的 out-of-place：
        # 原地乘法会让 AOTAutograd 的版本检查失败
        #（torch.compile 下反向报错）。
        if top_k == 2:
            # top_k == 2 时用 mul + addcmul 累加，等价于
            # 广播乘后沿 top_k 维求和（w0*x0 + w1*x1），
            # 但不产生 […, top_k, dim] 的中间张量。
            # addcmul 只原地写入加权乘产生的新张量。
            merged = expert_outputs.select(-2, 0).mul(
                weights.select(-1, 0).unsqueeze(-1)
            )
            merged.addcmul_(
                expert_outputs.select(-2, 1),
                weights.select(-1, 1).unsqueeze(-1),
            )
        else:
            # […, top_k, 1]，按最后一维广播相乘后沿 top_k 求和。
            merged = (
                expert_outputs * weights.unsqueeze(-1)
            ).sum(dim=-2)

        merged = merged.reshape(out_shape)

        # add(alpha=) 完成「缩放 + 残差相加」，dtype 不匹配时
        # 由类型提升合并（输出取更高精度），语义不变。
        if residual is not None:
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
        self.moe = SparseMoE(cfg)

        self.residual_scale = cfg.residual_scale

    def attention_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        core = self.attn.core(self.norm_attn(x))

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
                alpha=self.residual_scale,
            ).view(batch, length, dim)

        # add(alpha=) 完成「缩放 + 相加」。
        return torch.add(
            x,
            self.attn.out(core),
            alpha=self.residual_scale,
        )

    def moe_step(
        self,
        x: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            residual_scale=self.residual_scale,
        )
        return x, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.attention_step(x)
        return self.moe_step(x, top_k)

    def first_branch_step(
        self,
        x: torch.Tensor,
        k1: int,
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
        x = self.attention_step(x)

        expert_outputs, gates, indices, aux_loss = self.moe.unmerged(
            self.norm_moe(x),
            k1,
        )

        # expert_outputs:
        # [batch, time, k1, dim]
        #
        # 每个分支都保留完整残差：x + residual_scale * expert。
        #
        # dtype 一致时原地完成缩放与残差相加；缩放系数为 1.0 时
        # 乘以 1.0 是逐位恒等（含 NaN/Inf）。
        # autocast 下残差流精度更高时走 out-of-place 的类型提升路径。
        if expert_outputs.dtype == x.dtype:
            if self.residual_scale != 1.0:
                expert_outputs.mul_(
                    self.residual_scale,
                )
            branches = torch.add(
                x.unsqueeze(1),
                expert_outputs.permute(0, 2, 1, 3),
            )
        else:
            branches = torch.add(
                x.unsqueeze(1),
                expert_outputs.permute(0, 2, 1, 3),
                alpha=self.residual_scale,
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        A/local_base 的局部对照。

        local_base:
            第一轮 MoE 后立即合并。

        A:
            第一轮专家输出先保留为分支；
            第二轮每个分支独立路由；
            第二轮结束后再合并。
        """
        x = self.attention_step(x)

        if not delayed:
            matched_k2 = min(
                self.moe.n_experts,
                k1 * k2,
            )
            x, aux1 = self.moe_step(x, k1)
            x, aux2 = self.moe_step(x, matched_k2)
            return x, [aux1, aux2]

        expert_outputs, gates, _, aux1 = self.moe.unmerged(
            self.norm_moe(x),
            k1,
        )

        # 残差并入 expert_outputs：dtype 一致时原地完成
        # x + residual_scale * expert；缩放系数为 1.0 时
        # 乘以 1.0 是逐位恒等。autocast 下残差流精度更高时
        # 走 out-of-place 的类型提升路径。
        if expert_outputs.dtype == x.dtype:
            if self.residual_scale != 1.0:
                expert_outputs.mul_(
                    self.residual_scale,
                )
            expert_outputs.add_(
                x.unsqueeze(-2),
            )
            branches = expert_outputs
        else:
            branches = torch.add(
                x.unsqueeze(-2),
                expert_outputs,
                alpha=self.residual_scale,
            )

        # branches:
        # [batch, time, k1, dim]
        #
        # SparseMoE 支持任意前置维度，因此每个 branch
        # 的 token 会独立经过 Router。
        # 第二轮合并时直接把残差融进 merge 的
        # 加权和 + 残差相加路径。
        expert_outputs2, gates2, _, aux2 = self.moe.unmerged(
            self.norm_moe(branches),
            k2,
        )

        branches = self.moe.merge(
            expert_outputs2,
            gates2,
            branches.shape,
            residual=branches,
            residual_scale=self.residual_scale,
        )

        # 第一轮门控权重只在第二轮结束后使用。
        if branches.size(-2) == 1:
            # k1 == 1 时单元素 softmax 权重恒为精确 1.0
            #（且是无梯度常数），加权和退化为恒等，数值逐位一致。
            x = branches.squeeze(-2)
        else:
            # 跨分支的门控加权和：广播乘 + 沿分支维求和。
            # 必须保持对 branches / gates 的 out-of-place：
            # 门控权重需要梯度，原地乘法会让 AOTAutograd 的
            # 版本检查失败（torch.compile 下反向报错）。
            weights = gates.to(branches.dtype)

            if branches.size(-2) == 2:
                # k1 == 2 时用 mul + addcmul 累加
                #（w0*b0 + w1*b1），不产生 […, k1, dim] 的
                # 中间张量；addcmul 只原地写入加权乘产生的新张量。
                x = branches.select(-2, 0).mul(
                    weights.select(-1, 0).unsqueeze(-1)
                )
                x.addcmul_(
                    branches.select(-2, 1),
                    weights.select(-1, 1).unsqueeze(-1),
                )
            else:
                x = (
                    branches * weights.unsqueeze(-1)
                ).sum(dim=-2)

        return x, [aux1, aux2]


# ============================================================
# Model
# ============================================================

class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
        )

        self.position_embedding = nn.Embedding(
            cfg.seq_len,
            cfg.d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        self.final_norm = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
        )

        # 共享输入 embedding 和输出 projection。
        self.lm_head.weight = self.token_embedding.weight

        self.middle_start = cfg.middle_start
        self.middle_end = cfg.middle_end

        # 中间区域 Block 的元组。
        self.middle_blocks = tuple(
            self.blocks[i]
            for i in range(self.middle_start, self.middle_end)
        )

        # 与 position_embedding.weight 指向同一参数（共享存储）。
        # 经 object.__setattr__ 写入实例字典，避免重复注册到
        # _parameters 而污染 state_dict 与检查点兼容。
        object.__setattr__(
            self,
            "position_embedding_weight",
            self.position_embedding.weight,
        )

        self.middle_blocks_rest = self.middle_blocks[1:]

        # 是否对每个 Block 使用 activation checkpointing：
        # 开启后反向时重算前向激活。
        # 由训练入口根据命令行参数开启，默认关闭。
        self.grad_checkpoint = False

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _call(self, fn, *args):
        """
        统一所有 Block 调用的入口。

        开启 grad_checkpoint 时对 Block 前向使用
        use_reentrant=False 的 activation checkpointing，
        数值结果与直接调用一致。
        """
        # 梯度关闭时（evaluate 的 inference_mode / no_grad、以及
        # 纯前向探测）checkpoint 只会原样调用 fn，此时直接调用，
        # 数值结果完全一致。
        if self.grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                fn,
                *args,
                use_reentrant=False,
                # Block 内部没有 dropout 等随机算子，无需保存/恢复 RNG。
                preserve_rng_state=False,
            )
        return fn(*args)

    def run_independent_blocks(
        self,
        block: TransformerBlock,
        branches: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每个 branch 独立执行完整 Transformer Block。

        输入:
            [batch, n_branches, time, dim]

        将 branch 维并入 batch 维以后：
            [batch * n_branches, time, dim]

        因而不同 branch 不会互相做 self-attention。
        """
        batch, n_branches, time, dim = branches.shape

        flat = branches.reshape(
            batch * n_branches,
            time,
            dim,
        )

        flat, aux_loss = block(
            flat,
            top_k,
        )

        branches = flat.view(
            batch,
            n_branches,
            time,
            dim,
        )

        return branches, aux_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        batch, time = input_ids.shape

        if time > cfg.seq_len:
            raise ValueError(
                f"序列长度 {time} 超过 cfg.seq_len={cfg.seq_len}"
            )

        # token embedding 与按序列长度切片的位置 embedding 相加。
        x = self.token_embedding(input_ids)
        x.add_(
            self.position_embedding_weight[:time]
        )

        aux_losses: List[torch.Tensor] = []

        blocks = self.blocks
        call = self._call

        # 中间区域之前的层。
        for index in range(self.middle_start):
            x, aux = call(
                blocks[index],
                x,
                cfg.top_k,
            )
            aux_losses.append(aux)

        middle = self.middle_blocks

        # ----------------------------------------------------
        # Original SMELT
        # ----------------------------------------------------
        if cfg.mode == "smelt":
            for _ in range(2):
                for block in middle:
                    x, aux = call(
                        block,
                        x,
                        cfg.top_k,
                    )
                    aux_losses.append(aux)

        # ----------------------------------------------------
        # Scheme B
        # ----------------------------------------------------
        elif cfg.mode == "B":
            if len(middle) == 0:
                raise RuntimeError(
                    "中间层不能为空"
                )

            # 第一轮：
            # 在第一个中间 Block 的 MoE 处创建 k1 个分支。
            branches, gates, _, aux = call(
                middle[0].first_branch_step,
                x,
                cfg.k1,
            )
            aux_losses.append(aux)

            # 第一轮剩余中间 Block。
            for block in self.middle_blocks_rest:
                branches, aux = call(
                    self.run_independent_blocks,
                    block,
                    branches,
                    cfg.k1,
                )
                aux_losses.append(aux)

            # 第二轮完整执行整个中间区域。
            for block in middle:
                branches, aux = call(
                    self.run_independent_blocks,
                    block,
                    branches,
                    cfg.k2,
                )
                aux_losses.append(aux)

            # 只有第二轮结束后，才跨第一轮分支合并。
            #
            # gates:
            # [batch, time, k1]
            #
            # branches:
            # [batch, k1, time, dim]
            #
            if branches.size(1) == 1:
                # k1 == 1 时 gate 来自单元素 softmax，恒为精确 1.0
                #（且是无梯度常数），分支维收缩退化为恒等；
                # 直接去掉分支维，与 matmul 收缩结果逐位一致。
                x = branches.squeeze(1)
            else:
                # 归一化后的 top-k gate 在每个 token 上和为 1。
                # 跨分支收缩按分支顺序加权累加。
                branch_weights = gates.to(branches.dtype).unbind(-1)

                x = branches[:, 0].mul(
                    branch_weights[0].unsqueeze(-1)
                )

                for index in range(1, len(branch_weights)):
                    x.addcmul_(
                        branches[:, index],
                        branch_weights[index].unsqueeze(-1),
                    )

        # ----------------------------------------------------
        # Local base / Scheme A
        # ----------------------------------------------------
        else:
            delayed = cfg.mode == "A"

            for block in middle:
                x, block_aux = call(
                    block.local_two_stage,
                    x,
                    cfg.k1,
                    cfg.k2,
                    delayed,
                )
                aux_losses.extend(block_aux)

        # 中间区域之后的层。
        for index in range(self.middle_end, cfg.n_layers):
            x, aux = call(
                blocks[index],
                x,
                cfg.top_k,
            )
            aux_losses.append(aux)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        aux_loss = torch.stack(aux_losses).mean()

        output = {
            "logits": logits,
            "aux_loss": aux_loss,
        }

        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    cfg.vocab_size,
                ),
                targets.reshape(-1),
            )

            total_loss = (
                lm_loss
                + cfg.balance_coef * aux_loss
            )

            output.update({
                "lm_loss": lm_loss,
                "loss": total_loss,
            })

        return output


# ============================================================
# Dataset
# ============================================================

class ByteDataset:
    """
    字节级语言模型数据集。

    这是一个可复现的最小数据管线：
    - token id 范围固定为 0..255；
    - 训练/验证按原始字节流切分；
    - 验证窗口固定，避免不同实验之间采样噪声过大。
    """

    def __init__(
        self,
        filename: str,
        seq_len: int,
        val_fraction: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        data_path = Path(filename)

        # 文件在读取期间被截断时按实际读入长度处理。
        buf = bytearray(data_path.stat().st_size)
        with open(data_path, "rb") as handle:
            filled = handle.readinto(buf)

        if filled != len(buf):
            del buf[filled:]

        if len(buf) < seq_len + 2:
            raise ValueError(
                "数据文件太小，至少需要 seq_len + 2 个字节"
            )

        tokens = torch.frombuffer(buf, dtype=torch.uint8)

        _ = device

        split = int(
            len(tokens) * (1.0 - val_fraction)
        )

        self.train = tokens[:split]
        self.val = tokens[split:]
        self.seq_len = seq_len

        # 窗口偏移。
        self._offsets = torch.arange(
            seq_len + 1,
            device=tokens.device,
        )

        # 窗口暂存槽按调用方的 device 决定是否启用。
        self._stage_slots = None
        self._stage_index = 0

        # 每个 split 至少需要一个完整窗口，否则采样/验证
        # 没有合法起点（fixed_val_batch 依赖 num_starts >= 1，
        # evaluate 依赖至少一个验证 token 作分母）。
        if len(self.train) < seq_len + 1:
            raise ValueError(
                "训练集长度不足"
            )

        if len(self.val) < seq_len + 1:
            raise ValueError(
                "验证集长度不足"
            )

    def _gather_windows(
        self,
        data: torch.Tensor,
        starts: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        按窗口起点收集 [num_starts, seq_len + 1] 的字节窗口，
        并搬运到目标设备（尚未转 int64）。

        CUDA 下经暂存槽搬运，CPU 下走普通索引路径。
        """
        # starts 与 _offsets 均为 CPU 张量，窗口切片在 CPU 完成。
        numel = starts.numel() * (self.seq_len + 1)
        index_stage = getattr(self, "_index_stage", None)
        if index_stage is None or index_stage.numel() < numel:
            with torch.inference_mode(False):
                index_stage = torch.empty(numel, dtype=torch.int64)
            self._index_stage = index_stage

        flat_index_2d = index_stage[:numel].view(
            starts.numel(),
            self.seq_len + 1,
        )
        torch.add(
            starts[:, None],
            self._offsets[None, :],
            out=flat_index_2d,
        )
        flat_index = flat_index_2d.reshape(-1)

        if device.type != "cuda" or not torch.cuda.is_available():
            # index_select 连续收集，收集到的字节内容完全一致。
            return torch.index_select(
                data,
                0,
                flat_index,
            # 空批（0 窗口）时不能用 -1 推导列数。
            ).view(-1, self.seq_len + 1).to(device)

        if self._stage_slots is None:
            # 每槽: [缓冲区或 None, 拷贝完成 Event 或 None]
            self._stage_slots = [[None, None], [None, None]]

        slot = self._stage_slots[self._stage_index]
        self._stage_index ^= 1

        # 复用前确认上一次拷贝已完成。
        event = slot[1]
        if event is None:
            event = torch.cuda.Event()
            slot[1] = event
        else:
            event.synchronize()

        buffer = slot[0]
        if buffer is None or buffer.numel() < numel:
            with torch.inference_mode(False):
                buffer = torch.empty(
                    numel,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
            slot[0] = buffer

        staged = buffer[:numel]
        torch.index_select(data, 0, flat_index, out=staged)
        # 空批（0 窗口）时不能用 -1 推导列数。
        staged = staged.view(-1, self.seq_len + 1)

        chunks = staged.to(device, non_blocking=True)

        # 复用该槽位的事件记录完成时点，下次复用前先等待。
        with torch.cuda.device(device):
            event.record()

        return chunks

    def batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train
        elif split == "val":
            data = self.val
        else:
            raise ValueError(f"未知 split: {split}")

        max_start = len(data) - self.seq_len

        starts = torch.randint(
            low=0,
            high=max_start,
            size=(batch_size,),
            generator=generator,
        )

        # 窗口先以 uint8 搬运，到目标设备后再转 int64。
        chunks = self._gather_windows(
            data,
            starts,
            device,
        ).long()

        x = chunks[:, :-1]
        y = chunks[:, 1:]

        return x, y

    def fixed_val_batch(
        self,
        batch_index: int,
        batches: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回跨验证集均匀分布的固定窗口。

        不用随机数，因此验证估计不随运行进程、CUDA 状态或
        eval_batches 的调用顺序而变化；不同实验看到的窗口集合
        完全相同。请求窗口多于合法起点时仅取一遍无重复起点，
        避免同一验证样本因 round 重叠而以不同权重计入均值。
        """
        # 合法起点共 len(val) - seq_len 个。
        num_starts = len(self.val) - self.seq_len
        total_batches = min(
            batches * batch_size,
            num_starts,
        )

        if getattr(self, "_starts_total", None) != total_batches:
            # 均匀覆盖整个验证流。total_batches <= num_starts，
            # linspace 的间距不小于 1，round 后不会重复。
            self._starts = torch.linspace(
                0,
                num_starts - 1,
                steps=total_batches,
                dtype=torch.float64,
            ).round().long()
            self._starts_total = total_batches

        begin = batch_index * batch_size
        # _starts 为 CPU 张量，窗口收集与搬运与训练批一致。
        starts = self._starts[begin:begin + batch_size]

        chunks = self._gather_windows(
            self.val,
            starts,
            device,
        ).long()

        return chunks[:, :-1], chunks[:, 1:]


# ============================================================
# Evaluation
# ============================================================

# 评估张量不会回流到训练图，此处使用 inference_mode 是安全的。
@torch.inference_mode()
def evaluate(
    model: LanguageModel,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    # 按目标 token 数加权累加；最后一批不足 batch_size 时，
    # 也不会与满批获得相同的聚合权重。
    total = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    evaluated_tokens = 0

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            for batch_index in range(batches):
                x, y = dataset.fixed_val_batch(
                    batch_index,
                    batches,
                    batch_size,
                    device,
                )

                # 请求窗口超过验证集合法起点时，后继批次为空；
                # 已收集的窗口均不重复，不能用原 batches 作分母。
                if x.size(0) == 0:
                    break

                result = model(x, y)

                # logits 在前向返回后不再使用。
                result.pop("logits", None)

                # cross_entropy 返回批均值，乘回目标 token 数。
                # add_(alpha=) 完成「乘 token 数 + 累加」。
                total.add_(
                    result["lm_loss"].float(),
                    alpha=float(y.numel()),
                )
                evaluated_tokens += y.numel()
                del result, x, y
    finally:
        # 无论评估是否中途出错，都恢复原来的训练模式。
        model.train(was_training)

    # ByteDataset 保证至少有一个合法验证起点。
    loss = (total / evaluated_tokens).item()

    return {
        "val_loss": loss,
        "val_ppl": math.exp(min(loss, 80.0)),
        "val_bits_per_byte": loss / math.log(2.0),
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    filename: str,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    train_generator_state: torch.Tensor,
    args: argparse.Namespace,
):
    # torch.compile 包装后 state_dict 带有 "_orig_mod." 前缀，
    # 存档时解包，保证编译版与原始版产出的检查点可以互换。
    raw_model = getattr(model, "_orig_mod", model)

    payload = {
        "config": asdict(raw_model.cfg),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": (
            scaler.state_dict()
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
        "args": vars(args),
    }

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


def load_checkpoint(
    filename: str,
    device: torch.device,
) -> dict:
    return torch.load(
        filename,
        map_location=device,
    )


# ============================================================
# Tests
# ============================================================

def run_self_test():
    """
    结构、梯度、因果性和等价性测试。
    """

    set_seed(123)

    try:
        Config(
            vocab_size=256,
            seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=5,
            n_experts=3,
            expert_hidden=24,
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "非法 n_layers 必须在 -O 模式下仍然报错"
        )

    common = dict(
        vocab_size=256,
        seq_len=8,
        d_model=16,
        n_heads=4,
        n_layers=4,
        n_experts=3,
        expert_hidden=24,
        top_k=2,
        k1=2,
        k2=2,
    )

    x = torch.randint(
        0,
        256,
        (2, 8),
    )

    y = torch.randint(
        0,
        256,
        (2, 8),
    )

    # 所有模式都必须可以前向和反向。
    for mode in (
        "smelt",
        "local_base",
        "A",
        "B",
    ):
        model = LanguageModel(
            Config(
                **common,
                mode=mode,
            )
        )

        result = model(x, y)

        assert result["logits"].shape == (
            2,
            8,
            256,
        )

        assert torch.isfinite(result["loss"])

        result["loss"].backward()

        gradients = [
            p.grad
            for p in model.parameters()
            if p.grad is not None
        ]

        assert len(gradients) > 0

        for grad in gradients:
            assert torch.isfinite(grad).all()

    # B 的第一轮必须全程使用 k1，第二轮使用 k2。若在
    # 第一轮后续 Block 中混用 k2，就不再是两个轮次不同宽度
    # 路由的延迟合并基线。
    class RouteWidthProbe(LanguageModel):
        def __init__(self, cfg: Config):
            super().__init__(cfg)
            self.route_widths = []

        def run_independent_blocks(
            self,
            block: TransformerBlock,
            branches: torch.Tensor,
            top_k: int,
        ):
            self.route_widths.append(top_k)
            return super().run_independent_blocks(
                block,
                branches,
                top_k,
            )

    probe = RouteWidthProbe(
        Config(
            **{
                **common,
                "mode": "B",
                "k1": 1,
                "k2": 2,
            }
        )
    )

    with torch.no_grad():
        probe(x)

    middle_count = len(probe.middle_blocks)
    assert probe.route_widths == (
        [1] * (middle_count - 1)
        + [2] * middle_count
    )

    local_probe = LanguageModel(
        Config(
            **{
                **common,
                "n_experts": 4,
                "mode": "local_base",
                "k1": 2,
                "k2": 2,
            }
        )
    )
    local_widths = []
    for block in local_probe.blocks:
        original_route = block.moe.route

        def record_route(
            route_x,
            route_k,
            *,
            original_route=original_route,
        ):
            local_widths.append(route_k)
            return original_route(route_x, route_k)

        block.moe.route = record_route

    with torch.no_grad():
        local_probe(x)

    middle_count = len(local_probe.middle_blocks)
    assert local_widths == (
        [common["top_k"]]
        + [2, 4] * middle_count
        + [common["top_k"]]
    )

    # Top-k 负载均衡损失的规模不能随 k 改变而归一化。
    # 对均匀 Router，f_i 与 P_i 的乘积求和恰好等于 k。
    aux_moe = SparseMoE(
        Config(
            **common,
            mode="smelt",
        )
    )
    with torch.no_grad():
        aux_moe.router.weight.zero_()
    _, uniform_aux = aux_moe(
        torch.randn(2, 4, common["d_model"]),
        top_k=2,
    )
    torch.testing.assert_close(
        uniform_aux,
        torch.tensor(2.0, device=uniform_aux.device),
        rtol=1e-5,
        atol=1e-6,
    )

    # 空段占位张量为 fp32，非空专家输出为 bf16；
    # 散射缓冲按非空专家的 dtype 分配。
    merge_perm = torch.tensor([1, 0, 2])
    merge_counts = [0, 2, 1]
    merged = _MergeExpertOutputs.apply(
        merge_perm,
        merge_counts,
        torch.empty(0, 4, dtype=torch.float32),
        torch.tensor(
            [[1.0] * 4, [2.0] * 4],
            dtype=torch.bfloat16,
        ),
        torch.tensor(
            [[3.0] * 4],
            dtype=torch.bfloat16,
        ),
    )
    assert merged.dtype == torch.bfloat16
    assert torch.equal(
        merged,
        torch.tensor(
            [
                [2.0] * 4,
                [1.0] * 4,
                [3.0] * 4,
            ],
            dtype=torch.bfloat16,
        ),
    )

    # B 在 k1=k2=top_k=1 时应与 SMELT 数值等价。
    single_cfg = dict(
        common,
        top_k=1,
        k1=1,
        k2=1,
    )

    smelt = LanguageModel(
        Config(
            **single_cfg,
            mode="smelt",
        )
    ).eval()

    branch = LanguageModel(
        Config(
            **single_cfg,
            mode="B",
        )
    ).eval()

    branch.load_state_dict(
        smelt.state_dict()
    )

    with torch.no_grad():
        a = smelt(x)["logits"]
        b = branch(x)["logits"]

    torch.testing.assert_close(
        a,
        b,
        rtol=1e-5,
        atol=1e-6,
    )

    # A 在 k1=1 时应与立即合并局部结构等价。
    local_cfg = dict(
        common,
        k1=1,
    )

    local_base = LanguageModel(
        Config(
            **local_cfg,
            mode="local_base",
        )
    ).eval()

    delayed = LanguageModel(
        Config(
            **local_cfg,
            mode="A",
        )
    ).eval()

    delayed.load_state_dict(
        local_base.state_dict()
    )

    with torch.no_grad():
        a = local_base(x)["logits"]
        b = delayed(x)["logits"]

    torch.testing.assert_close(
        a,
        b,
        rtol=1e-5,
        atol=1e-6,
    )

    # grad checkpointing 的前向数值必须与直接调用一致，
    # 且梯度可以正常反传。
    ckpt_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    ckpt_model.grad_checkpoint = True

    plain_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    plain_model.load_state_dict(
        ckpt_model.state_dict()
    )

    with torch.no_grad():
        a = ckpt_model(x)["logits"]
        b = plain_model(x)["logits"]

    torch.testing.assert_close(
        a,
        b,
        rtol=1e-5,
        atol=1e-6,
    )

    ckpt_model.grad_checkpoint = True
    ckpt_model.train()
    ckpt_loss = ckpt_model(x, y)["loss"]
    ckpt_loss.backward()

    assert torch.isfinite(
        ckpt_model.blocks[0].attn.qkv.weight.grad
    ).all()

    # 因果性：改动末尾 token 不能影响之前位置的 logits。
    causal_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    x_changed = x.clone()
    x_changed[:, -1] = (x_changed[:, -1] + 1) % 256

    with torch.no_grad():
        logits_original = causal_model(x)["logits"]
        logits_changed = causal_model(x_changed)["logits"]

    torch.testing.assert_close(
        logits_original[:, :-1],
        logits_changed[:, :-1],
        rtol=1e-5,
        atol=1e-6,
    )

    # 分支之间不能通过 Attention 互相影响。
    model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    block = model.blocks[
        model.middle_start
    ]

    branches = torch.randn(
        2,
        2,
        8,
        16,
    )

    changed = branches.clone()
    changed[:, 1] += torch.randn_like(
        changed[:, 1]
    )

    with torch.no_grad():
        out1, _ = model.run_independent_blocks(
            block,
            branches,
            2,
        )
        out2, _ = model.run_independent_blocks(
            block,
            changed,
            2,
        )

    torch.testing.assert_close(
        out1[:, 0],
        out2[:, 0],
        rtol=1e-5,
        atol=1e-6,
    )

    assert not torch.allclose(
        out1[:, 1],
        out2[:, 1],
    )

    # 验证窗口必须与全局随机数状态无关且逐次调用完全一致。
    with tempfile.NamedTemporaryFile() as data_file:
        data_file.write(bytes(range(256)))
        data_file.flush()

        eval_dataset = ByteDataset(
            data_file.name,
            seq_len=8,
        )
        eval_device = torch.device("cpu")

        first = eval_dataset.fixed_val_batch(
            0,
            batches=2,
            batch_size=2,
            device=eval_device,
        )
        second = eval_dataset.fixed_val_batch(
            0,
            batches=2,
            batch_size=2,
            device=eval_device,
        )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

    # 请求窗口数超过验证起点数时，评估必须只使用每个
    # 起点一次；后继批次为空，而不是让 round 造成不等权重复。
    overfull_first = eval_dataset.fixed_val_batch(
        0,
        batches=3,
        batch_size=4,
        device=eval_device,
    )
    overfull_last = eval_dataset.fixed_val_batch(
        1,
        batches=3,
        batch_size=4,
        device=eval_device,
    )
    overfull_empty = eval_dataset.fixed_val_batch(
        2,
        batches=3,
        batch_size=4,
        device=eval_device,
    )

    assert (
        len(overfull_first[1]) + len(overfull_last[1])
        == len(eval_dataset._starts)
    )
    assert torch.equal(
        eval_dataset._starts.sort().values,
        eval_dataset._starts.unique(sorted=True),
    )
    assert overfull_empty[0].numel() == 0
    assert overfull_empty[1].numel() == 0

    # 模拟 checkpoint/resume：恢复时先从 scheduler.base_lrs
    # 还原基准 lr，再加载 scheduler 状态。
    old_param = nn.Parameter(torch.randn(2))
    old_optimizer = torch.optim.AdamW(
        [old_param],
        lr=0.1,
    )
    old_scheduler = build_lr_schedule(
        old_optimizer,
        warmup_steps=0,
        total_steps=8,
    )
    old_optimizer.step()
    old_scheduler.step()
    old_optimizer.step()
    old_scheduler.step()
    expected_lr = old_scheduler.get_last_lr()[0]
    scheduler_state = old_scheduler.state_dict()

    new_param = nn.Parameter(torch.randn(2))
    new_optimizer = torch.optim.AdamW(
        [new_param],
        lr=999.0,
    )
    new_optimizer.load_state_dict(
        old_optimizer.state_dict()
    )

    for group, base_lr in zip(
        new_optimizer.param_groups,
        scheduler_state["base_lrs"],
    ):
        group["lr"] = base_lr

    resumed_scheduler = build_lr_schedule(
        new_optimizer,
        warmup_steps=0,
        total_steps=8,
        start_step=2,
    )
    resumed_scheduler.load_state_dict(scheduler_state)

    torch.testing.assert_close(
        new_optimizer.param_groups[0]["lr"],
        expected_lr,
    )

    # resume 时必须用存档中的 warmup_steps 重建 lr_lambda。
    warmup_param = nn.Parameter(torch.randn(2))
    warmup_optimizer = torch.optim.AdamW(
        [warmup_param],
        lr=0.1,
    )
    original = build_lr_schedule(
        warmup_optimizer,
        warmup_steps=3,
        total_steps=10,
    )
    original_lrs = []
    for _ in range(10):
        original_lrs.append(
            warmup_optimizer.param_groups[0]["lr"]
        )
        warmup_optimizer.step()
        original.step()

    # 反例：用错误的 warmup（例如 CLI 默认值）重建，
    # lr 序列与原运行不同。
    wrong_param = nn.Parameter(torch.randn(2))
    wrong_optimizer = torch.optim.AdamW(
        [wrong_param],
        lr=0.1,
    )
    wrong = build_lr_schedule(
        wrong_optimizer,
        warmup_steps=5,
        total_steps=10,
    )
    wrong_lrs = []
    for _ in range(10):
        wrong_lrs.append(
            wrong_optimizer.param_groups[0]["lr"]
        )
        wrong_optimizer.step()
        wrong.step()

    assert not torch.allclose(
        torch.tensor(original_lrs),
        torch.tensor(wrong_lrs),
    ), "warmup mismatch must change the LR schedule"

    # 用存档 warmup 重建后逐位一致。
    rebuilt_param = nn.Parameter(torch.randn(2))
    rebuilt_optimizer = torch.optim.AdamW(
        [rebuilt_param],
        lr=0.1,
    )
    rebuilt = build_lr_schedule(
        rebuilt_optimizer,
        warmup_steps=3,
        total_steps=10,
    )
    rebuilt_lrs = []
    for _ in range(10):
        rebuilt_lrs.append(
            rebuilt_optimizer.param_groups[0]["lr"]
        )
        rebuilt_optimizer.step()
        rebuilt.step()

    torch.testing.assert_close(
        torch.tensor(original_lrs),
        torch.tensor(rebuilt_lrs),
    )

    print("self-test passed")


# ============================================================
# Training
# ============================================================

def build_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    start_step: int = 0,
):

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

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
        # 断点恢复时直接同步到 start_step。
        last_epoch=start_step - 1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SMELT-ISR 训练入口",
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="B",
        choices=["smelt", "local_base", "A", "B"],
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--k1", type=int, default=2)
    parser.add_argument("--k2", type=int, default=2)
    parser.add_argument("--balance-coef", type=float, default=0.01)
    parser.add_argument("--residual-scale", type=float, default=1.0)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="对 Block 前向启用 activation checkpointing",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="使用 torch.compile 编译模型前向",
    )

    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    # 在入口处校验参数合法性。
    if args.steps <= 0:
        parser.error("--steps 必须为正整数")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    if args.eval_interval <= 0:
        parser.error("--eval-interval 必须为正整数")
    if args.eval_batches <= 0:
        parser.error("--eval-batches 必须为正整数")
    if args.save_interval <= 0:
        parser.error("--save-interval 必须为正整数")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps 不能为负")
    if args.grad_clip <= 0:
        parser.error("--grad-clip 必须为正数")

    device = torch.device(args.device)

    if device.type == "cuda":
        # 需在首次 CUDA 分配之前设置。
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    scaler = None
    rng_payload = None

    if args.resume is not None:
        # 先在 CPU 上载入再拷入模型/优化器设备：检查点（含
        # 优化器状态）不再整体驻留显存，降低恢复时的显存峰值。
        rng_payload = load_checkpoint(
            args.resume,
            torch.device("cpu"),
        )
        # 模型结构配置从检查点读取；忽略未知的配置键。
        saved_config = rng_payload["config"]
        known_config_fields = {
            field.name for field in fields(Config)
        }
        unknown_config_fields = sorted(
            set(saved_config) - known_config_fields
        )
        if unknown_config_fields:
            print(
                "resume ignored unknown config fields: "
                + json.dumps(
                    unknown_config_fields,
                    ensure_ascii=False,
                )
            )
        cfg = Config(**{
            key: value
            for key, value in saved_config.items()
            if key in known_config_fields
        })
    else:
        cfg = Config(
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_experts=args.n_experts,
            expert_hidden=args.expert_hidden,
            top_k=args.top_k,
            k1=args.k1,
            k2=args.k2,
            mode=args.mode,
            balance_coef=args.balance_coef,
            residual_scale=args.residual_scale,
        )

    if rng_payload is not None:
        # 结构与损失配置已由检查点决定。把被忽略的命令行值
        # 改为实际生效值。
        config_fields = {
            "seq_len",
            "d_model",
            "n_heads",
            "n_layers",
            "n_experts",
            "expert_hidden",
            "top_k",
            "k1",
            "k2",
            "mode",
            "balance_coef",
            "residual_scale",
        }
        ignored = {}
        for field in config_fields:
            effective = getattr(cfg, field)
            provided = getattr(args, field)
            if provided != effective:
                ignored[field] = (provided, effective)
                setattr(args, field, effective)

        if ignored:
            print(
                "resume ignored CLI config overrides: "
                + json.dumps(ignored, ensure_ascii=False)
            )

    cfg.validate()
    set_seed(args.seed)

    model = LanguageModel(cfg).to(device)
    model.grad_checkpoint = args.grad_checkpoint

    # weight decay 只作用于 2D 权重矩阵；
    # 1D 参数（RMSNorm 增益）与 embedding（含与其共享的
    # lm_head）不做 weight decay，这是 LLM 训练的标准做法，
    # 否则会同时压缩输入嵌入与输出 logit 的尺度。
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # embedding 与 lm_head 共享同一个 2D 参数，
    # 单独排除，避免上面的 ndim 规则将其归入 decay 组；
    # position embedding 同理，也属于上面声明不做
    # weight decay 的 embedding。
    embedding_params = (
        model.token_embedding.weight,
        model.position_embedding.weight,
    )

    decay_params = [
        p for p in decay_params
        if all(p is not emb for emb in embedding_params)
    ]
    no_decay_params.extend(embedding_params)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused_adamw = device.type == "cuda"

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=use_fused_adamw,
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # bf16 无需 GradScaler；fp16 需要梯度缩放。
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype: Optional[torch.dtype] = torch.bfloat16
        scaler = None
    elif device.type == "cuda":
        amp_dtype = torch.float16
        # torch.amp.GradScaler(device) 自 torch 2.3 起可用；
        # 兼容 README 声明的 torch>=2.2 下限。
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None

    if rng_payload is not None:
        model.load_state_dict(rng_payload["model"])
        optimizer.load_state_dict(rng_payload["optimizer"])
        start_step = int(rng_payload.get("step", 0))
        if "step" not in rng_payload:
            print(
                "legacy checkpoint has no step; "
                "resuming from step 0"
            )

        if (
            scaler is not None
            and rng_payload.get("scaler") is not None
        ):
            scaler.load_state_dict(rng_payload["scaler"])

        # 检查点中的 RNG 状态统一先拉回 CPU 再恢复
        #（set_rng_state / set_state 只接受 CPU ByteTensor）。
        train_generator_state = rng_payload.get(
            "train_generator_state"
        )
        if train_generator_state is not None:
            train_generator_state = train_generator_state.cpu()

        torch_rng_state = rng_payload.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())

        python_rng_state = rng_payload.get("python_rng_state")
        if python_rng_state is not None:
            random.setstate(python_rng_state)

        cuda_rng_state = rng_payload.get("cuda_rng_state")
        if (
            cuda_rng_state is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state in cuda_rng_state
                ]
            )
    else:
        train_generator_state = None

    if rng_payload is not None:
        saved_scheduler = rng_payload.get("scheduler")

        if saved_scheduler is not None:
            base_lrs = saved_scheduler["base_lrs"]
        else:
            # 旧检查点没有 scheduler 状态时，使用存档中记录
            # 的原始 lr 作为 base_lr 重建调度。
            saved_base_lr = rng_payload.get(
                "args",
                {},
            ).get("lr", args.lr)
            base_lrs = [
                saved_base_lr
                for _ in optimizer.param_groups
            ]
            print(
                "legacy checkpoint has no scheduler state; "
                f"restored base lr {saved_base_lr}"
            )

        for group, base_lr in zip(
            optimizer.param_groups,
            base_lrs,
        ):
            group["lr"] = base_lr

    if rng_payload is not None:
        # batch-size 决定数据消费序列，是采样轨迹的一部分；
        # resume 时与 warmup_steps 一样按存档恢复并提示。
        saved_batch = rng_payload.get(
            "args",
            {},
        ).get("batch_size")

        if (
            saved_batch is not None
            and saved_batch != args.batch_size
        ):
            print(
                "resume restored batch_size "
                f"{args.batch_size} -> {saved_batch} "
                "(batch_size 决定采样轨迹)"
            )
            args.batch_size = saved_batch

        # lr_lambda 的 warmup 段是调度定义的一部分，但不在
        # scheduler 的 state_dict 里。resume 时若用当前 CLI 的
        # --warmup-steps 重建 lambda，共享区间内每一步的 lr 都会
        # 与原运行不一致（例如原 warmup=5、resume 默认 100，
        # 余弦段会被错当成 warmup 段）。因此恢复存档中的
        # warmup_steps；--steps 仍允许用户调整以延长/缩短训练。
        saved_warmup = rng_payload.get(
            "args",
            {},
        ).get("warmup_steps")

        if (
            saved_warmup is not None
            and saved_warmup != args.warmup_steps
        ):
            print(
                "resume restored warmup_steps "
                f"{args.warmup_steps} -> {saved_warmup} "
                "(warmup 是 LR 调度定义的一部分)"
            )
            args.warmup_steps = saved_warmup

        # 梯度裁剪直接改变每次更新后的参数，是优化轨迹的
        # 一部分。resume 时若用当前 CLI 的 --grad-clip（默认
        # 1.0），与原运行不同就会让恢复训练偏离原轨迹。
        saved_grad_clip = rng_payload.get(
            "args",
            {},
        ).get("grad_clip")

        if (
            saved_grad_clip is not None
            and saved_grad_clip != args.grad_clip
        ):
            print(
                "resume restored grad_clip "
                f"{args.grad_clip} -> {saved_grad_clip} "
                "(grad_clip 决定优化轨迹)"
            )
            args.grad_clip = saved_grad_clip

    scheduler = build_lr_schedule(
        optimizer,
        args.warmup_steps,
        args.steps,
        start_step=start_step,
    )

    if (
        rng_payload is not None
        and rng_payload.get("scheduler") is not None
    ):
        # 恢复 scheduler 的 base_lrs 与 last_epoch。
        scheduler.load_state_dict(
            rng_payload["scheduler"]
        )

        # state_dict 里的 _last_lr 是存档时旧调度的计算结果；
        # resume 后按当前调度同步日志视图。
        scheduler._last_lr = [
            group["lr"] for group in optimizer.param_groups
        ]

    if rng_payload is not None:
        del rng_payload

    train_generator = torch.Generator().manual_seed(
        args.seed
    )
    if train_generator_state is not None:
        train_generator.set_state(train_generator_state)

    if args.compile:
        # 在状态加载之后编译。MoE 专家分段长度由数据决定。
        model = torch.compile(model)

    dataset = ByteDataset(
        args.data,
        cfg.seq_len,
        device=device,
    )

    n_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"mode={cfg.mode} params={n_params:,}")
    print(f"device={device} start_step={start_step}")

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": asdict(cfg),
                "args": vars(args),
                "n_params": n_params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_path = out_dir / "metrics.jsonl"

    # metrics_file 在整个训练期间持续打开。
    # resume 时追加到原日志末尾以保持断点续训的完整记录；
    # 全新训练必须截断重写：否则复用旧 --out 目录时，
    # config.json 已被覆盖成新配置，metrics.jsonl 却混入
    # 旧实验的行，实验产物口径将不再对应同一次运行。
    metrics_file = open(
        metrics_path,
        "a" if args.resume is not None else "w",
        encoding="utf-8",
    )

    model.train()

    # 参数集合在训练期间不会变化（torch.compile 只包装调用，
    # 不增删参数）。
    train_params = list(model.parameters())

    # autocast 上下文的开关状态由已固定的 amp_dtype 决定。
    train_autocast = torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    )

    # 每步的标量指标先以设备上的张量暂存，写盘时统一取回。
    pending_records: List[Dict[str, object]] = []
    pending_scalars: List[torch.Tensor] = []

    def flush_records() -> List[Dict[str, object]]:
        if not pending_scalars:
            return []

        # 暂存项按 [lm, aux, grad_norm] 逐组平铺，堆叠后
        # reshape 成 [N, 3] 行。
        values = torch.stack(
            pending_scalars
        ).view(-1, 3).tolist()

        records = []
        for meta, (lm, aux, grad_norm) in zip(
            pending_records,
            values,
        ):
            # 发散步的 inf/nan 标量写盘前统一转成 null，
            # 保留该步记录本身。
            record = {
                "step": meta["step"],
                "train_loss": lm if math.isfinite(lm) else None,
                "aux_loss": aux if math.isfinite(aux) else None,
                "lr": meta["lr"],
                "grad_norm": (
                    grad_norm if math.isfinite(grad_norm) else None
                ),
            }
            extra = meta.get("extra")
            if extra is not None:
                record.update(extra)
            metrics_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            records.append(record)

        pending_records.clear()
        pending_scalars.clear()

        return records

    last_saved_step: Optional[int] = None

    # 采样始终按步序进行（消费 train_generator 的顺序不变）。
    prefetched_batch: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None

    # 每步采样下一批之前记录 generator 状态，即
    # 「从 step+1 恢复训练」所需的采样起点。
    checkpoint_gen_state: Optional[torch.Tensor] = None

    for step in range(start_step, args.steps):
        # 优化器更新成功时会立即推进 scheduler；先采样本步
        # 实际使用的 lr，保证日志语义不变。
        current_lr = scheduler.get_last_lr()[0]

        optimizer.zero_grad(
            set_to_none=True,
        )

        if prefetched_batch is None:
            x, y = dataset.batch(
                "train",
                args.batch_size,
                train_generator,
                device,
            )
        else:
            x, y = prefetched_batch
            prefetched_batch = None

        with train_autocast:
            result = model(x, y)
            loss = result["loss"]

        # 训练循环不再使用 logits；loss 是已算好的标量，
        # 反向图持有自己保存的中间量，与 Python 侧的张量
        # 引用无关。
        del result["logits"]

        # 记录本步存档应保存的 generator 状态，
        # 对应「从 step+1 恢复」的采样起点。
        checkpoint_gen_state = train_generator.get_state()

        # 提前采样并搬运下一批：CPU 侧窗口构造与 H2D 拷贝
        # 与随后的反向计算重叠；采样顺序不变。
        if step + 1 < args.steps:
            prefetched_batch = dataset.batch(
                "train",
                args.batch_size,
                train_generator,
                device,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params,
                args.grad_clip,
            )

            if grad_norm.isfinite():
                scaler.step(optimizer)
                scheduler.step()
            else:
                print(
                    f"step {step + 1}: non-finite gradient; "
                    "optimizer and LR schedule skipped"
                )
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params,
                args.grad_clip,
            )

            # 任何发散步都不应推进优化器或 LR schedule，
            # 否则训练步数与调度进度会随溢出次数漂移。
            if grad_norm.isfinite():
                optimizer.step()
                scheduler.step()
            else:
                print(
                    f"step {step + 1}: non-finite gradient; "
                    "optimizer and LR schedule skipped"
                )

        # detach 后只保留标量值；三者均为 fp32，无需额外转换。
        pending_scalars.append(
            result["lm_loss"].detach()
        )
        pending_scalars.append(
            result["aux_loss"].detach()
        )
        pending_scalars.append(
            grad_norm.detach()
        )

        del result, loss, x, y

        pending_records.append({
            "step": step + 1,
            "lr": current_lr,
        })

        if (step + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(
                model,
                dataset,
                args.batch_size,
                args.eval_batches,
                device,
                amp_dtype=amp_dtype,
            )
            pending_records[-1]["extra"] = eval_metrics

            records = flush_records()
            print(records[-1])
            metrics_file.flush()
            model.train()

        if (step + 1) % args.save_interval == 0:
            # 存档不一定位于 eval 点上，先落盘保证顺序。
            flush_records()
            save_checkpoint(
                str(out_dir / "last.pt"),
                model,
                optimizer,
                scaler,
                scheduler,
                step + 1,
                # 本步存档应记录的采样起点（见循环顶部的注释）。
                checkpoint_gen_state,
                args,
            )
            last_saved_step = step + 1

    # 先把暂存的逐步记录落盘，保持文件内 step 顺序。
    flush_records()

    # 结尾存档的 step 取优化器真实前进到的步数；
    # 循环未执行时（start_step >= args.steps）也取 start_step。
    end_step = max(start_step, args.steps)

    if last_saved_step != end_step:
        save_checkpoint(
            str(out_dir / "last.pt"),
            model,
            optimizer,
            scaler,
            scheduler,
            end_step,
            # 循环结束时最后一次采样前记录的状态即
            # 「从 end_step 恢复」的采样起点；
            # 循环未执行时（start_step >= steps）没有
            # 任何采样发生，直接取当前状态。
            (
                checkpoint_gen_state
                if checkpoint_gen_state is not None
                else train_generator.get_state()
            ),
            args,
        )

    final_metrics = evaluate(
        model,
        dataset,
        args.batch_size,
        args.eval_batches,
        device,
        amp_dtype=amp_dtype,
    )

    final_record = {
        "step": "final",
        **final_metrics,
    }
    metrics_file.write(
        json.dumps(
            final_record,
            ensure_ascii=False,
        )
        + "\n"
    )
    metrics_file.flush()

    print(
        json.dumps(
            {"final": final_metrics},
            ensure_ascii=False,
        )
    )

    metrics_file.close()


if __name__ == "__main__":
    if "--self-test" in __import__("sys").argv:
        run_self_test()
    else:
        main()
