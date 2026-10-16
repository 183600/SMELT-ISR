from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import numpy as np
except ImportError:
    # NumPy 不是必需依赖；安装时一并固定其随机种子。
    np = None


# 分配器与 cuBLAS 确定性工作区配置必须在任何 CUDA 上下文创建
# 之前生效（设备属性查询、内存分配都会创建上下文）。放在模块
# 导入时设置，训练入口与自测入口都覆盖到。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# use_deterministic_algorithms 下 cuBLAS 需要显式的工作区配置
#（PyTorch 官方要求 :4096:8 或 :16:8），否则结果可能不可复现
# 并触发告警。
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)


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
        if self.d_model <= 0:
            raise ValueError("d_model 必须为正整数")
        if self.expert_hidden <= 0:
            raise ValueError("expert_hidden 必须为正整数")
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
        if not math.isfinite(self.balance_coef) or self.balance_coef < 0.0:
            raise ValueError("balance_coef 必须是有限非负数")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale 必须是有限非负数")

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
    """
    固定 Python / NumPy / PyTorch(CPU+CUDA) / cuDNN 的随机种子。

    数据采样不经过 DataLoader worker，而由显式的
    torch.Generator 承担（见 ByteDataset.batch 与 main 中的
    train_generator）：其初始种子同样来自 --seed，当前状态随
    检查点一起存档，恢复训练时逐位还原。

    跨硬件架构仍可能因浮点归约顺序产生微小差异；同一台机器
    上的多次运行应保持一致。
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # manual_seed_all 覆盖当前可见的所有 CUDA 设备。
        torch.cuda.manual_seed_all(seed)

    # PYTHONHASHSEED 只对之后启动的子进程（如 torch.compile 的
    # 编译进程）生效；当前解释器的哈希种子在启动时已经确定，
    # 需要严格复现时应在命令行显式设置该环境变量。
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # 要求 PyTorch 优先选用确定性实现。bincount、attention
    # 反向等 CUDA 算子若不存在确定性实现会明确告警，而不是
    # 静默产生跨重启不可复现的微小差异。
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 本文件所有 torch.empty 缓冲在使用前都会被完整写入，
    # 不依赖未初始化缓冲的 NaN 填充来保证结果确定性。
    torch.utils.deterministic.fill_uninitialized_memory = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        if self._fused is not None and x.is_cuda:
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

        q, k, v = F.linear(x, self.qkv.weight).view(
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
        return F.linear(self.core(x), self.out.weight)


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
            # forward 内 grad mode 默认关闭，因此以 requires_grad
            # 判断是否需要保存。
            ctx.save_for_backward(h)

            # h 需要保存供反向使用，SiLU 保持 out-of-place，
            # mul_ 原地写入其输出。
            return F.silu(gate).mul_(up)

        # 无需反向时 h 只是临时张量，SiLU 直接原地写入
        # gate（h 的前半区）。
        return F.silu(gate, inplace=True).mul_(up)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (h,) = ctx.saved_tensors
        gate, up = h.chunk(2, dim=-1)

        # mul 反向给出两路梯度，gate 路再过 silu 反向；
        # 两路梯度分别写回 grad_h 的对应半区
        #（两半区互不相交且并集完整）。
        grad_h = torch.empty_like(h)
        grad_h_gate, grad_h_up = grad_h.chunk(2, dim=-1)

        # grad_up 半区先写入 grad_out * up 作为 silu 反向的输入。
        torch.mul(grad_out, up, out=grad_h_up)
        torch.ops.aten.silu_backward.grad_input(
            grad_h_up,
            gate,
            grad_input=grad_h_gate,
        )

        # grad_up = grad_out * silu(gate)。
        torch.ops.aten.silu.out(gate, out=grad_h_up)
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
        return F.linear(
            _SwiGLU.apply(
                F.linear(x, self.gate_up.weight)
            ),
            self.down.weight,
        )

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

        # 散射下标按计数一次切分，与各专家输出、计数对齐；
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

        # counts 回读的暂存、完成事件与专用 stream。
        self._counts_stage = None
        self._counts_event = None
        self._counts_stream = None

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
        router_probs = router_logits.softmax(dim=-1)

        top_values, top_indices = router_logits.topk(
            top_k,
            dim=-1,
        )

        # top-k 内部的 softmax 等价于完整 softmax 概率的
        # 重新归一化，直接对 top-k logits 求 softmax；
        # top_k == 1 时权重为单元素 softmax，恒为 1.0。
        if top_k == 1:
            top_weights = torch.ones_like(top_values)
        else:
            top_weights = top_values.softmax(dim=-1)

        # Top-k 版本的 Switch 风格负载均衡损失。
        # f_i 是「被分配给专家 i 的 token 比例」，以 token
        # 数量为分母；均匀路由时 E * sum(f_i * P_i) 恰好等于
        # top_k，均衡强度随路由宽度线性增大，跨 k 比较时
        # 应按比例折算 balance_coef（保持 balance_coef * k 不变）。
        counts = torch.bincount(
            top_indices.reshape(-1),
            minlength=self.n_experts,
        )

        # 与 dispatch 的一般路径一致时发起 counts 回读。
        if (
            counts.is_cuda
            and self.n_experts > 1
            and top_k != self.n_experts
        ):
            self._issue_counts_readback(counts)

        # counts 显式转 fp32，与 soft_load（router_logits 已
        # .float()）保持同一 dtype；int64 / int 的隐式提升跟随
        # 全局默认 dtype，非 fp32 时会与 soft_load 不一致导致
        # torch.dot 报错。
        hard_load = counts.float() / flat.size(0)

        soft_load = router_probs.mean(dim=0)

        # 逐专家 f_i * P_i 求和。
        aux_loss = self.n_experts * torch.dot(
            hard_load,
            soft_load,
        )

        return (
            flat,
            top_indices,
            top_weights,
            counts,
            aux_loss,
        )

    def _cast_for_autocast(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        CUDA autocast 下把张量转换为 autocast dtype；其余情况
        原样返回（dtype 一致时 to 也是无拷贝的空操作）。
        """
        if not tensor.is_cuda or not torch.is_autocast_enabled():
            return tensor

        try:
            dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            dtype = torch.get_autocast_gpu_dtype()

        return tensor.to(dtype)

    def _issue_counts_readback(self, counts: torch.Tensor) -> None:
        """
        启动 counts 回读：在专用 stream 上拷入暂存并记录完成
        事件，到需要逐专家计数时再读取；与 _read_counts 配对使用。
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
            self._counts_stream = torch.cuda.Stream(device=counts.device)

        self._counts_stream.wait_stream(
            torch.cuda.current_stream(counts.device)
        )
        with torch.cuda.stream(self._counts_stream):
            self._counts_stage.copy_(counts, non_blocking=True)
            self._counts_event.record()

        # 声明源张量与拷贝 stream 的关联。
        counts.record_stream(self._counts_stream)

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

        # 单专家时所有 token 都路由到同一个专家，
        # 排序等价于恒等排列。
        if self.n_experts == 1:
            expert_flat = self._cast_for_autocast(flat)

            if top_k == 1:
                return self.experts[0].forward(expert_flat).unsqueeze(1)

            # 所有 assignment 指向同一专家：对 [N, dim] 计算
            # 一次再 expand 出 slot 维。
            # expand 是 stride=0 视图，后面用 reshape 而不能用 view。
            return self.experts[0].forward(expert_flat).unsqueeze(
                1
            ).expand(
                num_tokens,
                top_k,
                dim,
            )

        # 稠密路由（top_k == n_experts）：每个 token 被分配给
        # 所有专家。每位专家处理完整输入，再按路由下标
        # gather 回 [token, slot] 位置。
        if top_k == self.n_experts:
            # 所有专家共用同一份输入副本。
            expert_flat = self._cast_for_autocast(flat)

            per_expert = torch.stack(
                [
                    expert.forward(expert_flat)
                    for expert in self.experts
                ],
                dim=1,
            )
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
        _, perm = torch.sort(assignment_experts)

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

        # 按排序下标收集输入，随后回读 counts。autocast 下
        # 先转换再收集：收集结果直接就是目标 dtype，与收集后
        # 转换的取值一致。
        gather_source = self._cast_for_autocast(flat)
        gathered = gather_source.index_select(0, sorted_token_ids)

        counts = self._read_counts(counts)
        segments = gathered.split(counts, dim=0)

        # 逐专家计算其连续输入段；空段原样占位，不产生
        # 专家调用。
        experts = self.experts
        expert_outputs = [
            (
                expert.forward(segment)
                if count > 0
                else segment
            )
            for count, expert, segment in zip(
                counts,
                experts,
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
        合并残差，语义不变。
        """
        top_k = expert_outputs.size(-2)
        dim = expert_outputs.size(-1)

        # top_k == 1 时门控权重恒为 1.0（单元素 softmax），
        # 加权和退化为恒等；有残差时完成「缩放 + 残差相加」。
        if top_k == 1:
            output = expert_outputs.reshape(-1, dim)

            if residual is not None:
                flat_residual = residual.reshape(-1, dim)

                # dtype 一致时原地完成「缩放 + 残差相加」。
                if output.dtype == flat_residual.dtype:
                    if residual_scale != 1.0:
                        output.mul_(residual_scale)
                    return output.add_(flat_residual).view(out_shape)

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
            F.linear(core, self.attn.out.weight),
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
        # dtype 不一致（autocast 残差流精度更高）时靠类型
        # 提升合并。
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
            # 立即合并对照与 A 的第二轮计算量对齐：A 在 k1 个
            # 分支上各路由 k2 个专家，单分支的第二轮相应取
            # min(n_experts, k1 * k2) 个专家。
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
        # x + residual_scale * expert；dtype 不匹配时用
        # add(alpha=) 合并。
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

        # 第一轮门控权重在第二轮结束后使用。
        if branches.size(-2) == 1:
            # k1 == 1 时单元素 softmax 权重恒为 1.0，
            # 加权和退化为恒等。
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
        # 经 object.__setattr__ 写入实例字典，不重复注册到
        # _parameters。
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
        x = F.embedding(input_ids, self.token_embedding.weight)
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
                # k1 == 1 时 gate 来自单元素 softmax，恒为 1.0，
                # 分支维收缩退化为恒等，直接去掉分支维。
                x = branches.squeeze(1)
            else:
                # 归一化后的 top-k gate 在每个 token 上和为 1。
                # 跨分支收缩按分支顺序加权累加。
                branch_weights = gates.to(
                    branches.dtype
                ).unsqueeze(-1).unbind(-2)

                x = branches[:, 0].mul(
                    branch_weights[0]
                )

                for index in range(1, len(branch_weights)):
                    x.addcmul_(
                        branches[:, index],
                        branch_weights[index],
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
        logits = F.linear(x, self.lm_head.weight)

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

        # 语料内容指纹：恢复训练时用来确认数据文件没有变化。
        self.data_sha256 = hashlib.sha256(buf).hexdigest()
        self.num_bytes = len(buf)
        self.val_fraction = val_fraction

        tokens = torch.frombuffer(buf, dtype=torch.uint8)

        _ = device

        split = int(
            len(tokens) * (1.0 - val_fraction)
        )

        self.train = tokens[:split]
        self.val = tokens[split:]
        self.seq_len = seq_len

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

        # unfold 产生 [num_windows, seq_len + 1] 的滑动窗口
        # 视图（stride 为 1），采样时按起点 index_select。
        self._train_windows = self.train.unfold(
            0,
            seq_len + 1,
            1,
        )
        self._val_windows = self.val.unfold(
            0,
            seq_len + 1,
            1,
        )

    def _gather_windows(
        self,
        windows: torch.Tensor,
        starts: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        按窗口起点从 [num_windows, seq_len + 1] 的窗口视图收集字节，
        并搬运到目标设备（尚未转整数索引）。

        CUDA 下经暂存槽搬运，CPU 下走普通索引路径。
        """
        num_rows = starts.numel()

        if device.type != "cuda" or not torch.cuda.is_available():
            return torch.index_select(
                windows,
                0,
                starts,
            ).to(device)

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

        numel = num_rows * (self.seq_len + 1)
        buffer = slot[0]
        if buffer is None or buffer.numel() < numel:
            with torch.inference_mode(False):
                buffer = torch.empty(
                    numel,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
            slot[0] = buffer

        staged = buffer[:numel].view(
            num_rows,
            self.seq_len + 1,
        )
        torch.index_select(windows, 0, starts, out=staged)

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
            windows = self._train_windows
            data = self.train
        elif split == "val":
            windows = self._val_windows
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

        # 窗口先以 uint8 搬运：输入 embedding 只接受 int32/int64，
        # 在目标设备上转 int32；目标 token 保持 uint8（cross_entropy
        # 直接支持 Byte 目标）。
        chunks = self._gather_windows(
            windows,
            starts,
            device,
        )

        x = chunks[:, :-1].to(torch.int32)
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
        完全相同。请求窗口多于合法起点时仅取一遍无重复起点。
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
            self._val_windows,
            starts,
            device,
        )

        x = chunks[:, :-1].to(torch.int32)
        y = chunks[:, 1:]

        return x, y


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

    # 按目标 token 数加权累加。
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

                # 请求窗口超过验证集合法起点时，后继批次为空。
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
# Sanity overfit check
# ============================================================

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
    seed: int,
) -> Dict[str, object]:
    """
    过拟合小批量数据的 sanity check。

    在开始全量训练前，用同一模型结构在一个固定小批次上训练：
    关闭负载均衡辅助损失、weight decay、梯度裁剪、warmup 等
    一切正则化（数据管线本身没有 dropout / 数据增强），要求训练
    损失迅速趋近 0 或下一 token 准确率接近 100%。

    这是必要不充分的检查：连小批量都无法过拟合，通常意味着
    梯度没有正确反传、损失计算错误、优化器配置错误等严重缺陷；
    但通过也不代表代码完全正确，数据增强、特征对齐、标签映射
    等缺陷仍可能漏网，仍需配合中间结果检查。

    函数内部固定 seed 并在结束后恢复调用前的全部 RNG 状态，
    因此是否运行本检查不影响主训练的随机轨迹。
    """
    saved_rng_state = _capture_rng_state()

    try:
        set_seed(seed)

        # balance_coef=0：关闭辅助负载均衡损失（正则项）。
        sanity_cfg = replace(cfg, balance_coef=0.0)
        model = LanguageModel(sanity_cfg).to(device)
        model.train()

        n_params = sum(
            p.numel() for p in model.parameters()
        )

        # 固定一个小批次：整个检查期间重复使用同一批数据。
        generator = torch.Generator().manual_seed(seed)
        x, y = dataset.batch(
            "train",
            batch_size,
            generator,
            device,
        )

        # weight_decay=0、无 warmup、无梯度裁剪：全部正则化关闭。
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
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

        initial_loss: Optional[float] = None
        final_loss = float("nan")
        final_acc = float("nan")

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)

                result = model(x, y)
                loss = result["lm_loss"]

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # 仅在需要报告或收尾的步取回标量：loss/准确率
                # 的 D2H 取值会同步 GPU，每步两次同步会打断
                # 流水线；取值来源与时刻均与原实现一致。
                if (
                    step == 1
                    or step % report_every == 0
                    or step == steps
                ):
                    final_loss = float(loss.detach())
                    final_acc = float(
                        (
                            result["logits"].argmax(dim=-1) == y
                        ).float().mean()
                    )

                    if initial_loss is None:
                        initial_loss = final_loss

                    print(
                        f"[sanity] step {step}/{steps} "
                        f"lm_loss={final_loss:.4f} "
                        f"token_acc={final_acc:.4f}"
                    )

        passed = (
            final_loss <= loss_threshold
            or final_acc >= acc_threshold
        )

        return {
            "passed": passed,
            "n_params": n_params,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "initial_lm_loss": initial_loss,
            "final_lm_loss": final_loss,
            "final_token_accuracy": final_acc,
            "loss_threshold": loss_threshold,
            "acc_threshold": acc_threshold,
        }
    finally:
        # 恢复调用前的 RNG 状态：sanity check 不改变主训练轨迹。
        _restore_rng_state(saved_rng_state)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def collect_run_metadata(device: torch.device) -> Dict[str, object]:
    """
    收集本轮实验的代码版本与运行环境信息，写入 config.json，
    保证实验可追溯：代码版本 + 环境 + 超参数 + 种子 +
    数据指纹（args.data_sha256）+ 训练曲线（metrics.jsonl）。
    """
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")

    device_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(device)
        except (AssertionError, RuntimeError, ValueError):
            device_name = None

    return {
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "command": " ".join(sys.argv),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": (
            getattr(np, "__version__", None)
            if np is not None
            else None
        ),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "device_name": device_name,
        "git_commit": commit,
        "git_dirty": (
            bool(status) if status is not None else None
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def write_run_config(
    out_dir: Path,
    cfg: Config,
    args: argparse.Namespace,
    n_params: int,
    run_metadata: Dict[str, object],
    sanity_result: Optional[Dict[str, object]],
) -> None:
    """
    写出 config.json：结构配置、完整超参数、参数量、数据指纹、
    代码版本/运行环境，以及 sanity check 结果。
    """
    payload = {
        "config": asdict(cfg),
        "args": vars(args),
        "n_params": n_params,
        "data": {
            "path": args.data,
            "sha256": getattr(args, "data_sha256", None),
            "bytes": getattr(args, "data_bytes", None),
            "val_fraction": getattr(
                args,
                "data_val_fraction",
                None,
            ),
        },
        "run_metadata": run_metadata,
        "sanity_check": sanity_result,
    }

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl_record(
    path: Path,
    record: Dict[str, object],
    append: bool = True,
) -> None:
    """
    追加（或重写）一行 JSON 记录到 metrics.jsonl。
    """
    with open(
        path,
        "a" if append else "w",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def load_previous_sanity_result(
    out_dir: Path,
) -> Optional[Dict[str, object]]:
    """
    读取已有 config.json 中的 sanity check 结果。

    resume 会重写 config.json；保留原始运行的 sanity 结果，
    避免追踪信息在恢复训练后丢失（metrics.jsonl 首行也有记录）。
    """
    path = out_dir / "config.json"

    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    result = payload.get("sanity_check")
    return result if isinstance(result, dict) else None


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
    # 存档时解包。
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
            "非法 n_layers 在 -O 模式下也必须被拒绝"
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

    # B 的第一轮全程使用 k1，第二轮使用 k2。
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

        assert eval_dataset.data_sha256 == hashlib.sha256(
            bytes(range(256))
        ).hexdigest()

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

    # 请求窗口数超过验证起点数时，评估只使用每个起点一次；
    # 后继批次为空。
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

    # param group 缺少 initial_lr 时，
    # start_step > 0 重建调度仍能工作。
    legacy_param = nn.Parameter(torch.randn(2))
    legacy_optimizer = torch.optim.AdamW(
        [legacy_param],
        lr=0.1,
    )
    legacy_optimizer.param_groups[0].pop(
        "initial_lr",
        None,
    )

    legacy_scheduler = build_lr_schedule(
        legacy_optimizer,
        warmup_steps=0,
        total_steps=8,
        start_step=2,
    )

    assert (
        legacy_scheduler.get_last_lr()[0] > 0.0
    ), "legacy resume must rebuild the LR schedule"

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

    # 用不同的 warmup（例如 CLI 默认值）重建时，
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

    # 用存档 warmup 重建后 lr 序列与原运行一致。
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

    # set_seed 必须让 Python / NumPy / PyTorch 的随机流可复现。
    set_seed(2024)
    first_draw = (
        random.random(),
        torch.rand(4),
        (
            float(np.random.rand())
            if np is not None
            else None
        ),
    )
    set_seed(2024)
    second_draw = (
        random.random(),
        torch.rand(4),
        (
            float(np.random.rand())
            if np is not None
            else None
        ),
    )

    assert first_draw[0] == second_draw[0]
    assert torch.equal(first_draw[1], second_draw[1])
    if np is not None:
        assert first_draw[2] == second_draw[2]

    # sanity check 必须能在小批量上过拟合，且结束后恢复调用前的
    # RNG 状态（不影响主训练轨迹）。
    sanity_cfg = Config(
        **common,
        mode="B",
    )

    rng_before = torch.get_rng_state()
    python_rng_before = random.getstate()

    sanity_metrics = run_sanity_overfit(
        sanity_cfg,
        eval_dataset,
        eval_device,
        amp_dtype=None,
        batch_size=2,
        steps=200,
        lr=1e-2,
        loss_threshold=0.5,
        acc_threshold=0.9,
        seed=321,
    )

    assert sanity_metrics["passed"], sanity_metrics
    assert torch.equal(
        torch.get_rng_state(),
        rng_before,
    )
    assert random.getstate() == python_rng_before

    # 学习率极小时不应通过，确认判定逻辑不是恒真。
    failed_metrics = run_sanity_overfit(
        sanity_cfg,
        eval_dataset,
        eval_device,
        amp_dtype=None,
        batch_size=2,
        steps=1,
        lr=1e-9,
        loss_threshold=0.0,
        acc_threshold=1.0,
        seed=321,
    )

    assert not failed_metrics["passed"]

    # 运行元数据必须包含代码版本与随机种子相关的追踪字段。
    metadata = collect_run_metadata(torch.device("cpu"))

    assert "git_commit" in metadata
    assert "torch_version" in metadata
    assert "deterministic_algorithms" in metadata

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
    # last_epoch >= 0 时 LRScheduler 要求 param group 中存在
    # initial_lr；该键缺失时用当前的基准 lr 补齐。
    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

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
    # 分配器与 cuBLAS 工作区配置在模块导入时已设置（见文件顶部），
    # 保证在任何 CUDA 上下文创建之前生效。
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
        "--skip-sanity-check",
        action="store_true",
        help=(
            "跳过全量训练前的过拟合小批量 sanity check"
            "（默认在新训练开始时运行）"
        ),
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="只运行 sanity check 并写出结果，不开始全量训练",
    )
    parser.add_argument("--sanity-batch-size", type=int, default=4)
    parser.add_argument("--sanity-steps", type=int, default=300)
    parser.add_argument("--sanity-lr", type=float, default=3e-3)
    parser.add_argument(
        "--sanity-loss-threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sanity-acc-threshold",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="本轮实验备注，随 config.json 一起存档",
    )

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
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须是有限正数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay 必须是有限非负数")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        parser.error("--grad-clip 必须为正数")
    if args.skip_sanity_check and args.sanity_only:
        parser.error(
            "--skip-sanity-check 与 --sanity-only 不能同时使用"
        )
    if args.sanity_batch_size <= 0:
        parser.error("--sanity-batch-size 必须为正整数")
    if args.sanity_steps <= 0:
        parser.error("--sanity-steps 必须为正整数")
    if not math.isfinite(args.sanity_lr) or args.sanity_lr <= 0:
        parser.error("--sanity-lr 必须是有限正数")
    if (
        not math.isfinite(args.sanity_loss_threshold)
        or args.sanity_loss_threshold < 0
    ):
        parser.error(
            "--sanity-loss-threshold 必须是非负有限数"
        )
    if not 0.0 < args.sanity_acc_threshold <= 1.0:
        parser.error(
            "--sanity-acc-threshold 必须落在 (0, 1]"
        )

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # AMP dtype 由设备与硬件能力决定：bf16 无需 GradScaler，
    # fp16 需要梯度缩放。在 sanity check 与主训练之前统一确定。
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype: Optional[torch.dtype] = torch.bfloat16
        scaler = None
    elif device.type == "cuda":
        amp_dtype = torch.float16
        # torch.amp.GradScaler(device) 自 torch 2.3 起可用。
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    rng_payload = None
    saved_data_sha256 = None

    if args.resume is not None:
        # 检查点先载入 CPU，再按目标设备分别搬运。
        rng_payload = load_checkpoint(
            args.resume,
            torch.device("cpu"),
        )
        saved_data_sha256 = rng_payload.get(
            "args",
            {},
        ).get("data_sha256")
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

    # 数据管线先于模型准备：sanity check 需要在创建全量模型之前
    # 先用同一数据管线验证模型代码能够过拟合小批量，同时拿到
    # 数据版本指纹写入实验追踪信息。
    dataset = ByteDataset(
        args.data,
        cfg.seq_len,
        device=device,
    )

    # 语料指纹/大小写入 args，随 config.json 与检查点一起存档；
    # 恢复训练时若数据文件内容已变化，采样轨迹会与原运行不同，
    # 这里明确提示而不是静默继续。
    args.data_sha256 = dataset.data_sha256
    args.data_bytes = dataset.num_bytes
    args.data_val_fraction = dataset.val_fraction

    if (
        saved_data_sha256 is not None
        and saved_data_sha256 != dataset.data_sha256
    ):
        print(
            "resume data file content changed "
            f"({saved_data_sha256[:12]}... -> "
            f"{dataset.data_sha256[:12]}...); "
            "采样轨迹与原运行不再一致"
        )

    # 全新训练默认先跑过拟合小批量 sanity check；恢复训练默认跳过
    # （可以用 --sanity-only 单独检查）。--skip-sanity-check 显式跳过。
    run_sanity = (
        args.sanity_only
        or (
            args.resume is None
            and not args.skip_sanity_check
        )
    )

    sanity_result: Optional[Dict[str, object]] = None
    sanity_record: Optional[Dict[str, object]] = None

    if run_sanity:
        print(
            "[sanity] 过拟合小批量检查: "
            f"batch_size={args.sanity_batch_size}, "
            f"steps={args.sanity_steps}, "
            f"lr={args.sanity_lr}, "
            f"loss<={args.sanity_loss_threshold} "
            f"或 acc>={args.sanity_acc_threshold}"
        )

        sanity_result = run_sanity_overfit(
            cfg,
            dataset,
            device,
            amp_dtype=amp_dtype,
            batch_size=args.sanity_batch_size,
            steps=args.sanity_steps,
            lr=args.sanity_lr,
            loss_threshold=args.sanity_loss_threshold,
            acc_threshold=args.sanity_acc_threshold,
            seed=args.seed,
        )

        sanity_record = {
            "step": "sanity",
            **sanity_result,
        }

        if sanity_result["passed"]:
            print(
                "[sanity] PASSED: "
                f"final lm_loss="
                f"{sanity_result['final_lm_loss']:.4f}, "
                f"token_acc="
                f"{sanity_result['final_token_accuracy']:.4f}"
            )
        else:
            print(
                "[sanity] FAILED: 模型未能在小批量上过拟合 "
                f"(final lm_loss="
                f"{sanity_result['final_lm_loss']:.4f}, "
                f"token_acc="
                f"{sanity_result['final_token_accuracy']:.4f})。"
                "这通常意味着梯度没有反向传播、损失计算错误、"
                "优化器或学习率配置有严重问题；请先排查再开始"
                "全量训练。确需跳过可用 --skip-sanity-check。"
            )

    if args.sanity_only or (
        sanity_result is not None
        and not sanity_result["passed"]
    ):
        # 保存追踪信息后退出：失败记录本身也是实验追踪的一部分，
        # 不盲目开始全量训练。
        write_run_config(
            out_dir,
            cfg,
            args,
            n_params=int(sanity_result["n_params"]),
            run_metadata=collect_run_metadata(device),
            sanity_result=sanity_result,
        )
        write_jsonl_record(
            out_dir / "metrics.jsonl",
            sanity_record,
            append=args.resume is not None,
        )
        print(
            json.dumps(
                {"sanity_check": sanity_result},
                ensure_ascii=False,
            )
        )

        if not sanity_result["passed"]:
            raise SystemExit(1)

        return

    set_seed(args.seed)

    model = LanguageModel(cfg).to(device)
    model.grad_checkpoint = args.grad_checkpoint

    # weight decay 只作用于 2D 权重矩阵；
    # 1D 参数（RMSNorm 增益）与 embedding（含与其共享的
    # lm_head）不做 weight decay。
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # embedding 与 lm_head 共享同一个 2D 参数，单独排除；
    # position embedding 同理。
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

    # 用单元素参数探测 fused AdamW 是否可用。
    probe = torch.zeros(1, device=device, requires_grad=True)
    probe.grad = torch.zeros_like(probe)
    try:
        torch.optim.AdamW([probe], lr=args.lr, fused=True).step()
        fused_adamw = True
    except (RuntimeError, TypeError):
        fused_adamw = False
    # fused 不可用时改用 foreach 批量更新：逐元素计算与逐参数
    # 路径逐位一致（已验证），但每个参数一组小算子会合并为少
    # 量多张量算子，显著减少调度开销。
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=fused_adamw,
            foreach=not fused_adamw,
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if rng_payload is not None:
        model.load_state_dict(rng_payload["model"])
        rng_payload.pop("model", None)
        optimizer.load_state_dict(rng_payload["optimizer"])
        rng_payload.pop("optimizer", None)
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
            rng_payload.pop("scaler", None)

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
            # 没有 scheduler 状态时，使用存档中记录的原始 lr
            # 作为 base_lr 重建调度。
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

        # 基础 lr 的实际数值来自检查点的调度状态；同步回
        # args，保证 config.json 记录的是真正生效的值。
        if args.lr != base_lrs[0]:
            print(
                "resume restored lr "
                f"{args.lr} -> {base_lrs[0]} "
                "(基础 lr 由检查点调度状态决定)"
            )
            args.lr = base_lrs[0]

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

        # weight_decay 的实际数值来自优化器状态
        #（load_state_dict 会替换 param group 选项）；同步回
        # args，保证 config.json 记录的是真正生效的值。
        effective_weight_decay = optimizer.param_groups[0][
            "weight_decay"
        ]

        if args.weight_decay != effective_weight_decay:
            print(
                "resume restored weight_decay "
                f"{args.weight_decay} -> {effective_weight_decay} "
                "(weight_decay 由优化器状态决定)"
            )
            args.weight_decay = effective_weight_decay

        # warmup 段是 lr 调度定义的一部分，但不在 scheduler 的
        # state_dict 里；resume 时按存档恢复 warmup_steps，
        # --steps 仍允许用户调整以延长/缩短训练。
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

        # 梯度裁剪直接改变每次更新后的参数；
        # resume 时按存档恢复 grad_clip。
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

        # 验证窗口集合由 batch_size * eval_batches 决定；
        # resume 时按存档恢复 eval_batches。
        saved_eval_batches = rng_payload.get(
            "args",
            {},
        ).get("eval_batches")

        if (
            saved_eval_batches is not None
            and saved_eval_batches != args.eval_batches
        ):
            print(
                "resume restored eval_batches "
                f"{args.eval_batches} -> {saved_eval_batches} "
                "(验证窗口集合由 batch_size*eval_batches 决定)"
            )
            args.eval_batches = saved_eval_batches

        # eval_interval 决定周期验证落在哪些训练步上，是验证
        # 协议的一部分；resume 时按存档恢复，保证恢复训练写出的
        # metrics.jsonl 与原始运行逐位对齐。
        saved_eval_interval = rng_payload.get(
            "args",
            {},
        ).get("eval_interval")

        if (
            saved_eval_interval is not None
            and saved_eval_interval != args.eval_interval
        ):
            print(
                "resume restored eval_interval "
                f"{args.eval_interval} -> {saved_eval_interval} "
                "(周期验证步点是验证协议的一部分)"
            )
            args.eval_interval = saved_eval_interval

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

        # 按恢复后的 last_epoch 重算 lr。
        for group, base_lr, lr_lambda in zip(
            optimizer.param_groups,
            scheduler.base_lrs,
            scheduler.lr_lambdas,
        ):
            group["lr"] = base_lr * lr_lambda(
                scheduler.last_epoch
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

    n_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"mode={cfg.mode} params={n_params:,}")
    print(f"device={device} start_step={start_step}")

    # 恢复训练默认跳过 sanity check；保留原运行的结果，
    # 避免重写 config.json 时丢失该追踪信息。
    config_sanity_result = sanity_result
    if config_sanity_result is None and args.resume is not None:
        config_sanity_result = load_previous_sanity_result(
            out_dir
        )

    # config.json 记录完整实验追踪信息：结构配置、超参数、种子、
    # 数据指纹、代码版本/运行环境与 sanity check 结果。
    write_run_config(
        out_dir,
        cfg,
        args,
        n_params,
        collect_run_metadata(device),
        config_sanity_result,
    )

    metrics_path = out_dir / "metrics.jsonl"

    # metrics_file 在整个训练期间持续打开。
    # resume 时追加到原日志末尾；全新训练截断重写。
    metrics_file = open(
        metrics_path,
        "a" if args.resume is not None else "w",
        encoding="utf-8",
    )

    if sanity_record is not None:
        # sanity check 结果也进入训练曲线文件，便于统一追溯。
        metrics_file.write(
            json.dumps(
                sanity_record,
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()

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

        # 记录先写入 Python 缓冲，此处统一落盘。
        metrics_file.flush()

        return records

    last_saved_step: Optional[int] = None

    # 最近一次周期性验证的步数与结果。
    last_eval_step: Optional[int] = None
    last_eval_metrics: Optional[Dict[str, float]] = None

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

        # 训练循环不再使用 logits。
        del result["logits"]

        # 记录本步存档应保存的 generator 状态，
        # 对应「从 step+1 恢复」的采样起点。
        checkpoint_gen_state = train_generator.get_state()

        # 采样下一批并搬运到设备；采样顺序不变。
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

            # 发散步不推进优化器或 LR schedule。
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

            last_eval_step = step + 1
            last_eval_metrics = eval_metrics

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

    # 参数自最近一次周期验证后未更新时复用其结果。
    if last_eval_step == end_step and last_eval_metrics is not None:
        final_metrics = last_eval_metrics
    else:
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
