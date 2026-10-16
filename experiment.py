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
import time
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
        # torch.dot 报错。归一化就地完成。
        hard_load = counts.float()
        hard_load.div_(flat.size(0))

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

    数据协议（防止数据泄露）：
    - 先划分、后处理：按原始字节流的时间先后顺序切成
      train / val / test 三段，三个 split 在原语料中前后相邻、
      互不重叠，切分全程不随机打乱（时序数据用过去训练、用未来
      验证和测试）；训练阶段的批级 shuffle 由随机采样窗口起点
      承担，不改变 split 边界；
    - 训练期间 test 保持尘封：训练循环只访问 train，周期性验证
      只访问 val，test 仅在最终评估（--evaluate-test）时访问一次；
    - 本管线不做任何依赖全局统计量的预处理（标准化、归一化、
      缺失值填充等）；若后续扩展为连续值特征，必须只在 train 上
      fit，再 transform val/test，禁止在划分前用全量数据统计。

    token id 范围固定为 0..255；验证/测试窗口固定，避免不同实验
    之间采样噪声过大。
    """

    def __init__(
        self,
        filename: str,
        seq_len: int,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        data_path = Path(filename)

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction 必须落在 [0, 1)")

        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction 必须落在 [0, 1)")

        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction 必须小于 1"
                "（训练集不能为空）"
            )

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
        self.test_fraction = test_fraction

        tokens = torch.frombuffer(buf, dtype=torch.uint8)

        # 数据审计（重复窗口扫描）直接读取原始字节；
        # memoryview 零拷贝共享同一块缓冲区。
        self._raw = memoryview(buf)

        _ = device

        # 仅做时间顺序上的连续切分，不计算统计量、不打乱：
        # [0, n_train) 训练，[n_train, n_train + n_val) 验证，
        # 其余为测试。
        n_val = int(len(tokens) * val_fraction)
        n_test = int(len(tokens) * test_fraction)
        n_train = len(tokens) - n_val - n_test

        self.split_bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, len(tokens)),
        }

        self.train = tokens[:n_train]
        self.val = tokens[n_train:n_train + n_val]
        self.test = tokens[n_train + n_val:]
        self.seq_len = seq_len

        # 窗口暂存槽按调用方的 device 决定是否启用。
        self._stage_slots = None
        self._stage_index = 0

        # 每个 split 至少需要一个完整窗口，否则采样/验证
        # 没有合法起点（fixed_eval_batch 依赖 num_starts >= 1，
        # evaluate 依赖至少一个评估 token 作分母）。
        if len(self.train) < seq_len + 1:
            raise ValueError(
                "训练集长度不足（检查 val/test fraction 或语料大小）"
            )

        if len(self.val) < seq_len + 1:
            raise ValueError(
                "验证集长度不足（检查 val_fraction 或语料大小）"
            )

        if test_fraction > 0.0 and len(self.test) < seq_len + 1:
            raise ValueError(
                "测试集长度不足（检查 test_fraction 或语料大小）"
            )

        # unfold 产生 [num_windows, seq_len + 1] 的滑动窗口
        # 视图（stride 为 1），采样时按起点 index_select。
        # 窗口只在各自 split 内部滑动，不跨越 split 边界。
        self._windows = {
            "train": self.train.unfold(
                0,
                seq_len + 1,
                1,
            ),
            "val": self.val.unfold(
                0,
                seq_len + 1,
                1,
            ),
        }
        if len(self.test) >= seq_len + 1:
            self._windows["test"] = self.test.unfold(
                0,
                seq_len + 1,
                1,
            )

        # 固定评估起点缓存按 split 分开保存。
        self._fixed_starts: Dict[str, Dict[str, object]] = {}

    @property
    def has_test(self) -> bool:
        """
        是否存在可用的测试 split（test_fraction > 0 且长度足够）。
        """
        return "test" in self._windows

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
            windows = self._windows["train"]
            data = self.train
        elif split == "val":
            windows = self._windows["val"]
            data = self.val
        elif split == "test":
            # 训练期间的随机批次绝不能碰测试集，
            # 否则测试集会在无感知的情况下被反复消费。
            raise ValueError(
                "测试集在最终评估前必须尘封，"
                "不能通过 batch() 访问"
            )
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

    def _eval_windows(
        self,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        取出指定评估 split 的窗口视图与底层 token 张量。

        batch() 只允许 train/val；test 只能经此处的固定窗口评估
        访问。
        """
        if split not in ("val", "test"):
            raise ValueError(f"未知评估 split: {split}")

        if split == "test" and not self.has_test:
            raise ValueError(
                "测试集不可用（test_fraction 为 0 或语料不足）"
            )

        windows = self._windows[split]
        data = self.val if split == "val" else self.test
        return windows, data

    def fixed_eval_batch(
        self,
        split: str,
        batch_index: int,
        batches: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回跨指定评估 split（val/test）均匀分布的固定窗口。

        不用随机数，因此评估估计不随运行进程、CUDA 状态或
        eval_batches 的调用顺序而变化；不同实验看到的窗口集合
        完全相同。请求窗口多于合法起点时仅取一遍无重复起点。
        """
        windows, data = self._eval_windows(split)

        # 合法起点共 len(data) - seq_len 个。
        num_starts = len(data) - self.seq_len
        total_batches = min(
            batches * batch_size,
            num_starts,
        )

        cache = self._fixed_starts.setdefault(
            split,
            {"total": None, "starts": None},
        )

        if cache["total"] != total_batches:
            # 均匀覆盖整个评估流。total_batches <= num_starts，
            # linspace 的间距不小于 1，round 后不会重复。
            cache["starts"] = torch.linspace(
                0,
                num_starts - 1,
                steps=total_batches,
                dtype=torch.float64,
            ).round().long()
            cache["total"] = total_batches

        begin = batch_index * batch_size
        # starts 为 CPU 张量，窗口收集与搬运与训练批一致。
        starts = cache["starts"][begin:begin + batch_size]

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
        """fixed_eval_batch("val", ...) 的兼容包装。"""
        return self.fixed_eval_batch(
            "val",
            batch_index,
            batches,
            batch_size,
            device,
        )


# ============================================================
# Data audit (leakage / distribution checks)
# ============================================================

# 重复窗口扫描的采样上限：窗口数超过该值时按等间距抽样。
_AUDIT_MAX_WINDOWS = 100000

# train/val(/test) 字节分布 JS 距离超过该值时报分布偏移警告。
_AUDIT_SHIFT_WARN = 0.1

# 单个字节占比超过该阈值时疑似填充/退化数据（异常值检查）。
_AUDIT_TOP_BYTE_WARN = 0.5


def _js_distance(
    p: torch.Tensor,
    q: torch.Tensor,
) -> float:
    """
    两个离散分布之间的 Jensen-Shannon 距离
    （JS 散度开方，取值 [0, 1]；0 表示分布相同）。
    """
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return (a[mask] * (a[mask] / b[mask]).log()).sum()

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return math.sqrt(max(0.0, float(js)))


def _window_signature_scan(
    raw: memoryview,
    offset: int,
    length: int,
    window_len: int,
    max_windows: int = _AUDIT_MAX_WINDOWS,
) -> Dict[str, object]:
    """
    对 [offset, offset + length) 内的滑动窗口做有界签名扫描。

    对每个采样窗口取 blake2b-128 摘要，用于发现跨 split 的重复
    内容（数据泄露的常见来源）与 split 内部重复。窗口数超过
    max_windows 时等间距抽样，返回实际采样规模。该扫描只做
    数据审计，不参与特征计算或模型决策。
    """
    total = length - window_len + 1

    if total <= 0:
        return {
            "sampled": 0,
            "total": 0,
            "stride": 1,
            "within_duplicates": 0,
            "signatures": {},
        }

    stride = max(1, math.ceil(total / max_windows))

    signatures: Dict[bytes, List[int]] = {}
    within_duplicates = 0
    sampled = 0

    for start in range(0, total, stride):
        digest = hashlib.blake2b(
            raw[
                offset + start:
                offset + start + window_len
            ],
            digest_size=16,
        ).digest()
        sampled += 1

        bucket = signatures.get(digest)
        if bucket is None:
            signatures[digest] = [start]
        else:
            within_duplicates += 1
            if len(bucket) < 4:
                bucket.append(start)

    return {
        "sampled": sampled,
        "total": total,
        "stride": stride,
        "within_duplicates": within_duplicates,
        "signatures": signatures,
    }


def audit_data_splits(
    dataset: ByteDataset,
    include_test: bool = False,
    max_windows: int = _AUDIT_MAX_WINDOWS,
) -> Dict[str, object]:
    """
    数据审计：值域校验、分布偏移与重复窗口检测。

    - 校验每个 split 的 token 值域（字节管线为 0..255；整数
      token 中不存在 NaN/Inf，值域校验用于捕获上游 dtype 或
      编码错误）；
    - 统计每个 split 的字节直方图，并计算 train↔val（可选包含
      train↔test、val↔test）之间的 JS 距离，用于发现分布偏移；
    - 对滑动窗口做有界签名扫描，检测跨 split 重复内容与 split
      内部重复。

    test 只在 include_test=True（最终评估解封时）才参与统计，
    训练期间默认只审计 train/val。审计只做诊断，不参与任何
    特征计算或模型决策。
    """
    split_names = ["train", "val"]
    if include_test and dataset.has_test:
        split_names.append("test")

    histograms: Dict[str, torch.Tensor] = {}
    scans: Dict[str, Dict[str, object]] = {}
    splits_report: Dict[str, object] = {}
    warnings: List[str] = []

    for split in split_names:
        tokens = getattr(dataset, split)

        # 字节管线固定为 uint8，直方图直接在原始 dtype 上
        # 统计（uint8 天然落在 0..255，越界恒为 0）。其他
        # dtype（若日后扩展管线）回退到 int64 校验路径，
        # 保持诊断语义不变。
        if tokens.dtype == torch.uint8:
            try:
                hist = torch.bincount(
                    tokens,
                    minlength=256,
                )
            except RuntimeError:
                # 旧版本 bincount 仅接受 int64 输入，退回转换。
                hist = torch.bincount(
                    tokens.to(torch.int64),
                    minlength=256,
                )
            hist = hist.to(torch.float64)
            invalid = 0
        else:
            tokens_long = tokens.to(torch.int64)
            hist = torch.bincount(
                tokens_long,
                minlength=256,
            ).to(torch.float64)
            invalid = int(
                ((tokens_long < 0) | (tokens_long > 255)).sum()
            )

        histograms[split] = hist / hist.sum()

        top_share, top_byte = torch.topk(
            histograms[split],
            k=5,
        )

        begin, end = dataset.split_bounds[split]
        scans[split] = _window_signature_scan(
            dataset._raw,
            begin,
            end - begin,
            dataset.seq_len + 1,
            max_windows,
        )

        splits_report[split] = {
            "bytes": end - begin,
            "windows": scans[split]["total"],
            "invalid_tokens": invalid,
            "top_bytes": [
                [int(byte), round(float(share), 6)]
                for share, byte in zip(
                    top_share.tolist(),
                    top_byte.tolist(),
                )
            ],
        }

        if invalid > 0:
            warnings.append(
                f"{split}: 检测到 {invalid} 个越界 token"
            )

        top_byte_share = float(top_share[0])
        if top_byte_share > _AUDIT_TOP_BYTE_WARN:
            warnings.append(
                f"{split}: 最高频字节占比 "
                f"{top_byte_share:.3f}，疑似填充/退化数据"
            )

    distribution_shift: Dict[str, float] = {}

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        if left not in histograms or right not in histograms:
            continue

        distance = _js_distance(
            histograms[left],
            histograms[right],
        )
        key = f"{left}_{right}"
        distribution_shift[key] = round(distance, 6)

        if distance > _AUDIT_SHIFT_WARN:
            warnings.append(
                f"{key}: 字节分布偏移较大"
                f"（JS 距离 {distance:.4f}）"
            )

    duplicate_windows: Dict[str, object] = {}

    for split, scan in scans.items():
        duplicate_windows[split] = {
            "sampled": scan["sampled"],
            "total": scan["total"],
            "stride": scan["stride"],
            "within_duplicates": scan["within_duplicates"],
        }

        if scan["within_duplicates"] > 0:
            warnings.append(
                f"{split}: 采样窗口中有"
                f"{scan['within_duplicates']}个 split 内部重复"
            )

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        if left not in scans or right not in scans:
            continue

        left_sigs = scans[left]["signatures"]
        right_sigs = scans[right]["signatures"]
        common = left_sigs.keys() & right_sigs.keys()

        duplicate_windows[f"{left}_{right}"] = {
            "shared_signatures": len(common),
            "examples": [
                [left_sigs[digest][0], right_sigs[digest][0]]
                for digest in list(common)[:3]
            ],
        }

        if common:
            warnings.append(
                f"{left}_{right}: 检测到重复窗口内容"
                "（跨 split 泄露风险）"
            )

    return {
        "max_windows": max_windows,
        "splits": splits_report,
        "distribution_shift": distribution_shift,
        "duplicate_windows": duplicate_windows,
        "warnings": warnings,
    }


def print_data_audit(report: Dict[str, object]) -> None:
    """打印数据审计摘要与全部警告。"""
    splits = report.get("splits", {})
    parts = [
        f"{name}={info['bytes']}B"
        for name, info in splits.items()
    ]
    print("[data] 数据审计: " + ", ".join(parts))

    for key, distance in report.get(
        "distribution_shift",
        {},
    ).items():
        print(f"[data]   {key} 字节分布 JS 距离 = {distance:.4f}")

    for key, info in report.get(
        "duplicate_windows",
        {},
    ).items():
        if "shared_signatures" in info:
            print(
                f"[data]   {key} 共享窗口签名数 = "
                f"{info['shared_signatures']}"
            )

    for warning in report.get("warnings", []):
        print(f"[data] 警告: {warning}")


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
    split: str = "val",
) -> Dict[str, float]:
    """
    在指定评估 split 上计算加权平均语言模型损失。

    split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    test 只在最终评估时调用一次；返回指标键带 split 前缀
    （val_loss/test_loss 等）。
    """
    was_training = model.training
    model.eval()

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    # split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    # test 只在最终评估时调用一次。

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
                x, y = dataset.fixed_eval_batch(
                    split,
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

    # ByteDataset 保证至少有一个合法评估起点。
    loss = (total / evaluated_tokens).item()

    return {
        f"{split}_loss": loss,
        f"{split}_ppl": math.exp(min(loss, 80.0)),
        f"{split}_bits_per_byte": loss / math.log(2.0),
    }


# ============================================================
# Training-curve analysis
# ============================================================

# 曲线诊断最少需要的验证点数：少于该数量时只报告原始曲线，
# 不给出过拟合/欠拟合结论，避免用一两个点下判断。
_CURVE_MIN_EVAL_POINTS = 3

# 验证损失从最优点回升超过该比例、且训练损失仍在下降时
# 判定为过拟合。
_CURVE_OVERFIT_REGRESSION = 0.02

# 判定「训练损失仍在下降」的最小相对降幅。
_CURVE_TRAIN_STILL_DESCENDING = 0.02

# 最近一次验证间隔内训练/验证损失仍下降超过该比例时，
# 判定为尚未收敛（继续训练可能还有收益）。
_CURVE_STILL_DESCENDING = 0.05

# 整体相对下降小于该比例、且最终训练损失仍接近随机基线
# （baseline_loss）时判定为欠拟合/几乎没有学到东西。
_CURVE_NO_LEARN_DROP = 0.05
_CURVE_NO_LEARN_BASELINE = 0.9

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values, width=48):
    """
    把一列数值渲染成等宽 Unicode 走势条，便于在终端快速扫
    训练曲线走势；数值越小时块越低（损失越小越靠下）。
    """
    values = [float(value) for value in values]

    if not values:
        return ""

    if len(values) > width:
        stride = len(values) / width
        values = [
            values[min(len(values) - 1, int(index * stride))]
            for index in range(width)
        ]

    low = min(values)
    high = max(values)

    if high - low <= 0.0:
        return _SPARK_BLOCKS[0] * len(values)

    return "".join(
        _SPARK_BLOCKS[
            min(
                len(_SPARK_BLOCKS) - 1,
                int(
                    (value - low)
                    / (high - low)
                    * (len(_SPARK_BLOCKS) - 1)
                    + 0.5
                ),
            )
        ]
        for value in values
    )


def analyze_training_curve(
    records: List[Dict[str, object]],
    baseline_loss: Optional[float] = None,
) -> Dict[str, object]:
    """
    基于完整训练曲线（metrics.jsonl）而不是最终单个数值做诊断。

    对每个周期验证点，取「自上次验证以来训练损失的平均值」与
    val_loss 配对，按以下规则给出结论：

    - ``overfitting``：验证损失最优点早于最后一次验证、最终
      验证损失较最优值回升超过阈值，而训练损失仍在下降；
    - ``underfitting``：训练损失整体几乎不降，且最终仍接近随
      机基线（ln(vocab_size)）；
    - ``not_converged``：最近一次间隔内训练与验证损失仍同步
      明显下降，说明训练步数/学习率还没到平台；
    - ``diverged``：最近一步损失非有限（NaN/Inf）；
    - ``healthy``：训练与验证损失同步下降且验证未回升；
    - ``insufficient_data``：验证点不足以做结论。

    返回的诊断会追加为 ``{"step": "curve_analysis", ...}``
    记录，并打印对应建议（过拟合→正则化/增大数据/早停，
    欠拟合→查学习率、优化器与数据管道）。
    """
    train_points: List[Tuple[int, float]] = []
    nonfinite_steps: List[int] = []
    eval_points: List[Dict[str, object]] = []
    pending: List[float] = []

    for record in records:
        step = record.get("step")

        if not isinstance(step, int):
            continue

        if "grad_norm" in record or "train_loss" in record:
            train_loss = record.get("train_loss")
            # 非有限事件既可能落在损失上（写盘为 null），也可能
            # 只落在梯度上（损失仍有限但 grad_norm 为 null）。
            nonfinite = train_loss is None or (
                "grad_norm" in record
                and record.get("grad_norm") is None
            )
            if nonfinite:
                if (
                    not nonfinite_steps
                    or nonfinite_steps[-1] != step
                ):
                    nonfinite_steps.append(step)
            if train_loss is not None:
                train_loss = float(train_loss)
                train_points.append((step, train_loss))
                pending.append(train_loss)

        if "val_loss" not in record:
            continue

        val_loss = record.get("val_loss")

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

        train_at_eval = (
            sum(pending) / len(pending)
            if pending
            else (train_points[-1][1] if train_points else None)
        )

        eval_points.append({
            "step": step,
            "val_loss": float(val_loss),
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

    analysis: Dict[str, object] = {
        "n_train_steps": len(train_points),
        "n_eval_points": len(finite_evals),
        "n_nonfinite_events": len(nonfinite_steps),
        "nonfinite_steps": nonfinite_steps[-10:],
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

    def _relative_drop(newer, older):
        if older is None or newer is None:
            return None
        older = float(older)
        newer = float(newer)
        if older <= 0.0:
            return None
        return (older - newer) / older

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

    analysis.update({
        "first_train_loss": first["train_loss"],
        "last_train_loss": last["train_loss"],
        "first_val_loss": first["val_loss"],
        "last_val_loss": last["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_step": best["step"],
        "val_regression_from_best": round(regression, 6),
        "train_val_gap_first": (
            round(
                float(first["train_loss"])
                - float(first["val_loss"]),
                6,
            )
            if first["train_loss"] is not None
            else None
        ),
        "train_val_gap_last": (
            round(
                float(last["train_loss"])
                - float(last["val_loss"]),
                6,
            )
            if last["train_loss"] is not None
            else None
        ),
        "curve_sparkline_train": _sparkline(
            [value for _, value in train_points]
        ),
        "curve_sparkline_val": _sparkline(
            [point["val_loss"] for point in finite_evals]
        ),
    })

    if (
        best_index <= len(finite_evals) - 2
        and regression >= _CURVE_OVERFIT_REGRESSION
        and train_still_down
    ):
        analysis["status"] = "overfitting"
        analysis["recommendation"] = (
            f"验证损失在第 {best['step']} 步达到最优 "
            f"{float(best['val_loss']):.4f} 后回升 "
            f"{regression * 100:.1f}%，训练损失仍在下降："
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
        print(
            "[curve] 最优验证损失 "
            f"{float(analysis['best_val_loss']):.4f} @ step "
            f"{analysis.get('best_val_step')}，"
            "最终验证损失 "
            f"{float(analysis['last_val_loss']):.4f}"
            "（相对最优回升 "
            f"{float(analysis.get('val_regression_from_best') or 0.0) * 100:.1f}%），"
            "最终训练损失 "
            f"{float(analysis['last_train_loss']):.4f}"
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


def early_stop_state_from_records(
    records: List[Dict[str, object]],
    min_delta: float,
) -> Tuple[Optional[float], Optional[int], int]:
    """
    从已有 metrics.jsonl 记录恢复早停状态（最优验证损失、最优
    步数、连续未改善次数），保证 resume 后早停判定与原运行一致。
    """
    best_loss: Optional[float] = None
    best_step: Optional[int] = None
    evals_without_improvement = 0

    for record in records:
        if "val_loss" not in record:
            continue

        val_loss = record.get("val_loss")
        step = record.get("step")

        if val_loss is None:
            continue

        val_loss = float(val_loss)

        if (
            best_loss is None
            or val_loss < best_loss - min_delta
        ):
            best_loss = val_loss
            best_step = step if isinstance(step, int) else None
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1

    return best_loss, best_step, evals_without_improvement


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


def _render_byte(value: int) -> str:
    """把单个字节渲染为可打印文本（不可打印字节用 \\xNN）。"""
    value = int(value)
    if 32 <= value < 127:
        return chr(value)
    return f"\\x{value:02x}"


def _render_bytes(values) -> str:
    """把字节序列渲染为可打印文本。"""
    return "".join(_render_byte(value) for value in values)


@torch.inference_mode()
def analyze_bad_cases(
    model: LanguageModel,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    split: str = "val",
    top_n: int = 10,
    context_before: int = 16,
    context_after: int = 8,
) -> Dict[str, object]:
    """
    坏案例分析：把预测错误的样本打印出来并归类错误原因。

    使用与周期验证完全相同的固定窗口（protocol 口径一致），
    对每个目标 token 计算：

    - loss / 置信度（top1 概率）/ margin（top1 概率 − 目标概率）；
    - 污染最严重的 top_n 个 token，附上下文文本、目标与预测
      字节、错误类别；
    - 高频混淆对（target→predicted）与逐目标字节错误率；
    - 错误类别统计：ambiguous（样本极其模糊）、
      systematic_confusion（同类混淆反复出现，模型盲区）、
      label_noise_suspect（目标字节稀少但模型高置信度答错，
      提示标注可能有误）、uncertain（其余）。

    仅访问 val（或显式解封后的 test），绝不接触训练期间尘封的
    测试集；输出既打印，也写回 metrics.jsonl 的
    ``{"step": "bad_cases", ...}`` 记录，完整报告另存
    bad_cases.json，作为提出新改进思路的证据来源。
    """
    was_training = model.training
    model.eval()

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    vocab_size = _unwrap_model(model).cfg.vocab_size

    losses: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    predictions: List[torch.Tensor] = []
    confidences: List[torch.Tensor] = []
    margins: List[torch.Tensor] = []
    windows: List[torch.Tensor] = []

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            for batch_index in range(batches):
                x, y = dataset.fixed_eval_batch(
                    split,
                    batch_index,
                    batches,
                    batch_size,
                    device,
                )

                if x.size(0) == 0:
                    break

                result = model(x)

                # 坏案例统计只在 fp32 下做，避免半精度下
                # softmax/gather 的数值误差影响错误归因。
                logits = result["logits"].float()
                del result

                log_normalizer = torch.logsumexp(logits, dim=-1)
                target_logits = logits.gather(
                    -1,
                    y.unsqueeze(-1).to(torch.int64),
                ).squeeze(-1)
                loss_tokens = log_normalizer - target_logits

                top2 = logits.topk(2, dim=-1)

                predicted = top2.indices[..., 0]
                confidence = (top2.values[..., 0] - log_normalizer).exp()
                target_probability = (
                    target_logits - log_normalizer
                ).exp()

                # 还原完整窗口：x = tokens[:-1]，y = tokens[1:]。
                window = torch.cat(
                    [x[:, :1].to(torch.uint8), y],
                    dim=1,
                )

                losses.append(loss_tokens.reshape(-1).cpu())
                targets.append(y.reshape(-1).cpu())
                predictions.append(
                    predicted.reshape(-1).to(torch.int16).cpu()
                )
                confidences.append(
                    confidence.reshape(-1).to(torch.float16).cpu()
                )
                margins.append(
                    (confidence - target_probability)
                    .reshape(-1)
                    .to(torch.float16)
                    .cpu()
                )
                windows.append(window.cpu())

        if not losses:
            return {
                "split": split,
                "tokens_evaluated": 0,
                "token_accuracy": None,
                "top_cases": [],
                "top_confusions": [],
                "error_rate_by_target": [],
                "category_counts": {},
                "summary": "评估窗口为空，未生成坏案例。",
            }

        loss_all = torch.cat(losses)
        target_all = torch.cat(targets).to(torch.int64)
        predicted_all = torch.cat(predictions).to(torch.int64)
        confidence_all = torch.cat(confidences).float()
        margin_all = torch.cat(margins).float()
        window_all = torch.cat(windows)

        num_tokens = loss_all.numel()
        sequence_length = window_all.size(1) - 1

        wrong_mask = target_all != predicted_all
        token_accuracy = float(
            (1.0 - wrong_mask.float().mean()).item()
        )

        # 逐目标字节在评估 split 中的占比，用于标注噪声提示。
        target_counts = torch.bincount(
            target_all,
            minlength=vocab_size,
        ).to(torch.float64)
        target_share = target_counts / max(1.0, float(num_tokens))

        # 高频混淆对：target * vocab + predicted。
        confusion_index = (
            target_all * vocab_size + predicted_all
        )
        confusion_counts = torch.bincount(
            confusion_index[wrong_mask],
            minlength=vocab_size * vocab_size,
        )
        confusion_count_by_token = confusion_counts[
            confusion_index
        ]

        top_confusion_values, top_confusion_indices = torch.topk(
            confusion_counts,
            k=min(10, int((confusion_counts > 0).sum().item()) or 1),
        )

        top_confusions = []
        for count, index in zip(
            top_confusion_values.tolist(),
            top_confusion_indices.tolist(),
        ):
            if count <= 0:
                break
            top_confusions.append([
                int(index) // vocab_size,
                int(index) % vocab_size,
                int(count),
            ])

        # 逐目标字节错误率（只保留出现过的目标字节，按错误率排序）。
        error_counts = torch.bincount(
            target_all[wrong_mask],
            minlength=vocab_size,
        )
        error_rate = torch.where(
            target_counts > 0,
            error_counts.to(torch.float64)
            / target_counts.clamp(min=1),
            torch.zeros(vocab_size, dtype=torch.float64),
        )
        seen = target_counts > 0
        seen_indices = torch.nonzero(seen).reshape(-1)
        seen_rates = error_rate[seen_indices]
        order = torch.argsort(seen_rates, descending=True)
        error_rate_by_target = []
        for index in order[:10].tolist():
            byte = int(seen_indices[index])
            error_rate_by_target.append([
                byte,
                round(float(seen_rates[index]), 6),
                int(target_counts[byte]),
            ])

        # 污染最严重的 top_n token。
        count = min(top_n, num_tokens)
        top_losses, top_indices = torch.topk(loss_all, k=count)

        top_cases = []
        for rank, flat_index in enumerate(
            top_indices.tolist()
        ):
            row = flat_index // sequence_length
            position = flat_index % sequence_length

            target = int(target_all[flat_index])
            predicted = int(predicted_all[flat_index])
            confidence = float(confidence_all[flat_index])
            margin = float(margin_all[flat_index])
            pair_count = int(
                confusion_count_by_token[flat_index]
            )

            if margin < _BAD_CASE_AMBIGUOUS_MARGIN:
                category = "ambiguous"
            elif (
                pair_count >= _BAD_CASE_SYSTEMATIC_COUNT
                and confidence >= 0.5
            ):
                category = "systematic_confusion"
            elif (
                confidence >= _BAD_CASE_CONFIDENT
                and float(target_share[target])
                < _BAD_CASE_RARE_TARGET_SHARE
            ):
                category = "label_noise_suspect"
            else:
                category = "uncertain"

            window = window_all[row]
            target_position = position + 1
            begin = max(0, target_position - context_before)
            end = min(
                window.numel(),
                target_position + 1 + context_after,
            )
            fragment = window[begin:end].tolist()
            marker = target_position - begin

            context = (
                _render_bytes(fragment[:marker])
                + "["
                + _render_byte(target)
                + "]"
                + _render_bytes(fragment[marker + 1:])
            )

            top_cases.append({
                "rank": rank + 1,
                "target": target,
                "target_text": _render_byte(target),
                "predicted": predicted,
                "predicted_text": _render_byte(predicted),
                "loss": round(float(loss_all[flat_index]), 6),
                "confidence": round(confidence, 6),
                "margin": round(margin, 6),
                "category": category,
                "confusion_count": pair_count,
                "context": context,
                "window_text": _render_bytes(
                    window.tolist()
                ),
            })

        category_counts: Dict[str, int] = {}
        for case in top_cases:
            key = str(case["category"])
            category_counts[key] = (
                category_counts.get(key, 0) + 1
            )

        top1 = top_cases[0] if top_cases else None
        summary = (
            f"评估 {num_tokens} 个 token，token 准确率 "
            f"{token_accuracy:.4f}；最差样本 "
            + (
                "target="
                f"'{_render_byte(int(top1['target']))}', "
                "predicted="
                f"'{_render_byte(int(top1['predicted']))}', "
                f"loss={float(top1['loss']):.4f}, "
                f"category={top1['category']}。"
                if top1 is not None
                else "无。"
            )
        )

        report: Dict[str, object] = {
            "split": split,
            "tokens_evaluated": int(num_tokens),
            "token_accuracy": round(token_accuracy, 6),
            "top_cases": top_cases,
            "top_confusions": top_confusions,
            "error_rate_by_target": error_rate_by_target,
            "category_counts": category_counts,
            "summary": summary,
        }

        return report
    finally:
        model.train(was_training)


def print_bad_cases(report: Dict[str, object]) -> None:
    """打印坏案例报告：最差样本、混淆对与错误分类统计。"""
    split = report.get("split")
    print(
        f"[bad-case] split={split} tokens="
        f"{report.get('tokens_evaluated')} "
        f"token_acc={report.get('token_accuracy')}"
    )

    for case in report.get("top_cases", []):
        print(
            f"[bad-case]   #{case['rank']} "
            f"loss={case['loss']:.4f} "
            f"conf={case['confidence']:.3f} "
            f"margin={case['margin']:.3f} "
            f"target='{case['target_text']}' "
            f"pred='{case['predicted_text']}' "
            f"category={case['category']} | "
            f"{case['context']}"
        )

    if report.get("top_confusions"):
        pairs = " ".join(
            f"'{_render_byte(target)}'→"
            f"'{_render_byte(predicted)}'({count})"
            for target, predicted, count in report[
                "top_confusions"
            ][:5]
        )
        print(f"[bad-case] 高频混淆对: {pairs}")

    if report.get("category_counts"):
        print(
            "[bad-case] 错误类别统计: "
            + json.dumps(
                report["category_counts"],
                ensure_ascii=False,
            )
        )

    print(f"[bad-case] 小结: {report.get('summary')}")


# ============================================================
# Non-finite (NaN/Inf) diagnosis
# ============================================================

# 非有限损失/梯度的系统排查路径（顺序固定，先查证据更简单
# 的环节，再查混合精度与优化环节）。
_NONFINITE_CHECKLIST = [
    "1) 数据：批次内是否存在 NaN/Inf 或越界 token"
    "（字节管线的整数 token 天然无 NaN，越界/负值可能来自"
    "上游 dtype 或编码错误；扩展到连续值特征后必须查 NaN/Inf）。",
    "2) 前向数值：logits 是否非有限、绝对值是否接近低精度"
    "上限；有限 logits + 非有限损失通常是 log(0)/除零/目标"
    "越界等损失侧问题。",
    "3) 混合精度：fp16 autocast 下 matmul/softmax 溢出会先"
    "出现大绝对值再变 NaN；可对比关闭 AMP（--device cpu 或"
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
            if bool(finite_mask.any()):
                finite_values = tensor[finite_mask]
                info[f"{name}_min"] = float(
                    finite_values.min()
                )
                info[f"{name}_max"] = float(
                    finite_values.max()
                )
        else:
            info[f"{name}_min"] = int(tensor.min())
            info[f"{name}_max"] = int(tensor.max())

        # uint8 与 Python 整数比较会按 uint8 回绕（如 255 >= 256
        # 会溢出成 255 >= 0 而误报），先提升到 int64 再判值域。
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
    vocab_size: int = 256,
) -> Dict[str, object]:
    """
    非有限损失/梯度出现时的系统排查。

    按固定路径收集证据：数据值域与非有限元素 → logits/损失
    数值（log(0)、除零、softmax 溢出）→ 低精度上限 → 逐参数
    梯度、参数与优化器状态污染 → 学习率/梯度缩放状态。返回的
    报告会附加到触发该步的 metrics 记录（nan_diagnosis 字段）
    并打印，之后按报告中的 next_steps 逐项排除。
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

    # 训练循环在反向之前就释放了 logits；诊断时补跑一次前向，
    # 否则前向数值检查拿不到证据。只在确认出错后执行一次。
    probe_source = "training result"
    if logits is None and isinstance(model, LanguageModel):
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
    }

    if _finite(logits) is False:
        findings.append(
            "logits 已含 NaN/Inf：优先怀疑低精度 matmul/softmax "
            "溢出或模型内部数值爆炸"
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

    # ---- 5) 学习率与梯度缩放 ----
    checks["lr"] = lr
    if scaler is not None:
        try:
            checks["grad_scale"] = float(
                scaler.get_scale()
            )
        except Exception:
            checks["grad_scale"] = None

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
      torch.utils.flop_counter.FlopCounterMode 实测；
    - SDPA 内部的注意力矩阵乘不计入 FlopCounter，用模块
      pre-hook 按 4·B·T²·D 补上（乘加各计 1 次浮点运算）。

    返回的是一次前向的总 FLOPs，用于跨模式比较计算量。
    """
    flops: Dict[str, Optional[int]] = {
        "linear_flops": None,
        "attention_flops": None,
        "total_flops_per_forward": None,
        "flops_note": (
            "linear(mm/addmm) 实测 + attention(4·B·T²·D) 估计"
        ),
    }

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return flops

    attention_flops = [0]

    def _attention_hook(module, args):
        inp = args[0]
        if inp.dim() == 3:
            # numel = B·T·D，乘 4·T 即 4·B·T²·D。
            attention_flops[0] += (
                4 * int(inp.size(-2)) * int(inp.numel())
            )

    # 注意：TransformerBlock 直接调用 self.attn.core()，不会经过
    # CausalSelfAttention.__call__，因此 hook 必须挂在 Block 上
    # （smelt 模式的中间层会被执行两轮，hook 自然只统计真实次数）。
    handles = []
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            handles.append(
                module.register_forward_pre_hook(
                    _attention_hook
                )
            )

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
        for handle in handles:
            handle.remove()

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
    inference_peak_mb = None
    training_peak_mb = None

    try:
        model.eval()

        with torch.no_grad(), autocast:
            for _ in range(max(0, warmup)):
                model(x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings: List[float] = []
        with torch.no_grad(), autocast:
            for _ in range(max(1, iters)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(x, y)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)

        timings.sort()
        latency_seconds = timings[len(timings) // 2]

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            model.train()
            with autocast:
                output = model(x, y)
                output["loss"].backward()
            training_peak_mb = round(
                torch.cuda.max_memory_allocated(device)
                / 1e6,
                3,
            )
            model.zero_grad(set_to_none=True)

            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), autocast:
                model(x, y)
            inference_peak_mb = round(
                torch.cuda.max_memory_allocated(device)
                / 1e6,
                3,
            )
            del output
        else:
            # CPU 路径：跑一次训练模式前反向。
            model.train()
            with autocast:
                output = model(x, y)
                output["loss"].backward()
            model.zero_grad(set_to_none=True)
            del output
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
        "throughput_tokens_per_sec": (
            round(tokens / latency_seconds, 2)
            if latency_seconds
            else None
        ),
        "inference_peak_memory_mb": inference_peak_mb,
        "training_peak_memory_mb": training_peak_mb,
        "process_peak_rss_mb": _process_peak_rss_mb(),
        "batch_size": int(batch_size),
        "seq_len": int(x.size(1)),
        "tokens_per_forward": tokens,
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
        return format(float(value), spec) + unit

    print(
        "[efficiency] "
        f"params={efficiency.get('n_params_total')} "
        f"(non-embedding="
        f"{efficiency.get('n_params_non_embedding')}) | "
        f"FLOPs/forward="
        f"{_format(efficiency.get('flops_per_forward'), '', '.3e')} | "
        f"latency={_format(efficiency.get('latency_ms_per_forward'), 'ms')} | "
        f"throughput="
        f"{_format(efficiency.get('throughput_tokens_per_sec'), 'tok/s')}"
    )
    print(
        "[efficiency] "
        f"inference_peak="
        f"{_format(efficiency.get('inference_peak_memory_mb'), 'MB')} | "
        f"training_peak="
        f"{_format(efficiency.get('training_peak_memory_mb'), 'MB')} | "
        f"process_peak_rss="
        f"{_format(efficiency.get('process_peak_rss_mb'), 'MB')}"
    )


def compare_runs(paths: List[str]) -> List[Dict[str, object]]:
    """
    汇总多个运行目录的 config 与最终指标。

    读取每个目录的 config.json（结构、参数量、efficiency）与
    metrics.jsonl（final/test/curve_analysis），输出统一口径的
    行。
    """
    rows: List[Dict[str, object]] = []

    for path in paths:
        run_dir = Path(path)
        row: Dict[str, object] = {
            "run": run_dir.name,
            "path": str(run_dir),
        }

        config_payload: Dict[str, object] = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                with open(
                    config_path,
                    encoding="utf-8",
                ) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_payload = loaded
            except (OSError, json.JSONDecodeError):
                config_payload = {}

        run_config = config_payload.get("config") or {}
        if isinstance(run_config, dict):
            row["mode"] = run_config.get("mode")
            row["top_k"] = run_config.get("top_k")
            row["k1"] = run_config.get("k1")
            row["k2"] = run_config.get("k2")

        row["n_params"] = config_payload.get("n_params")

        efficiency = config_payload.get("efficiency") or {}
        if not isinstance(efficiency, dict):
            efficiency = {}
        row["latency_ms_per_forward"] = efficiency.get(
            "latency_ms_per_forward"
        )
        row["flops_per_forward"] = efficiency.get(
            "flops_per_forward"
        )
        row["peak_memory_mb"] = (
            efficiency.get("training_peak_memory_mb")
            if efficiency.get("training_peak_memory_mb")
            is not None
            else efficiency.get("process_peak_rss_mb")
        )

        records = read_metrics_records(
            run_dir / "metrics.jsonl"
        )
        final_record = None
        test_record = None
        curve_record = None

        for record in records:
            step = record.get("step")
            if step == "final":
                final_record = record
            elif step == "test":
                test_record = record
            elif step == "curve_analysis":
                curve_record = record

        row["val_loss"] = (
            final_record.get("val_loss")
            if final_record is not None
            else None
        )
        row["val_ppl"] = (
            final_record.get("val_ppl")
            if final_record is not None
            else None
        )
        row["test_loss"] = (
            test_record.get("test_loss")
            if test_record is not None
            else None
        )
        row["curve_status"] = (
            curve_record.get("status")
            if curve_record is not None
            else None
        )
        row["early_stopped"] = (
            final_record.get("early_stopped")
            if final_record is not None
            else None
        )

        rows.append(row)

    return rows


def print_comparison(rows: List[Dict[str, object]]) -> None:
    """以对齐表格打印多个运行的汇总指标。"""
    header = (
        f"{'run':<20} {'mode':<9} {'params':>9} "
        f"{'val_loss':>9} {'test_loss':>9} {'lat(ms)':>9} "
        f"{'GFLOPs':>8} {'mem(MB)':>9} {'curve':<18}"
    )
    print(header)
    print("-" * len(header))

    def _value(value, spec=".4f", scale=1.0):
        if value is None:
            return "-"
        return format(float(value) * scale, spec)

    for row in rows:
        flops = row.get("flops_per_forward")
        gflops = (
            f"{float(flops) / 1e9:.3f}"
            if flops is not None
            else "-"
        )
        print(
            f"{str(row.get('run'))[:19]:<20} "
            f"{str(row.get('mode'))[:8]:<9} "
            f"{str(row.get('n_params')):>9} "
            f"{_value(row.get('val_loss')):>9} "
            f"{_value(row.get('test_loss')):>9} "
            f"{_value(row.get('latency_ms_per_forward'), '.4f'):>9} "
            f"{gflops:>8} "
            f"{_value(row.get('peak_memory_mb'), '.1f'):>9} "
            f"{str(row.get('curve_status')):<18}"
        )


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

                # 仅在需要报告或收尾的步取回标量，取值来源
                # 与时刻均与原实现一致。
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


def _json_safe(value):
    """
    把记录中的非有限浮点值（NaN/Inf）递归转成 None。

    json.dumps 默认会把 NaN/Inf 写成非标准 JSON 字面量
    （NaN / Infinity），其他语言的解析器会拒绝；统一转成 null
    以保留该字段但明确标记为「非有限」，避免污染整份日志。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_run_config(
    out_dir: Path,
    cfg: Config,
    args: argparse.Namespace,
    n_params: int,
    run_metadata: Dict[str, object],
    sanity_result: Optional[Dict[str, object]],
    data_audit: Optional[Dict[str, object]] = None,
) -> None:
    """
    写出 config.json：结构配置、完整超参数、参数量、数据指纹、
    数据划分与审计结果、代码版本/运行环境，以及 sanity check
    结果。
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
            "test_fraction": getattr(
                args,
                "data_test_fraction",
                None,
            ),
        },
        "data_audit": data_audit,
        "run_metadata": run_metadata,
        "sanity_check": sanity_result,
    }

    # resume 会重写 config.json；保留上次运行结束时补充的
    # efficiency 测量结果。
    config_path = out_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict) and "efficiency" in previous:
            payload.setdefault(
                "efficiency",
                previous["efficiency"],
            )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(payload),
            f,
            ensure_ascii=False,
            indent=2,
        )


def update_run_config(out_dir: Path, key: str, value) -> None:
    """
    训练结束后把补充的分析结果（如 efficiency）并入 config.json。

    读出已有 config.json、合并单个键后整体重写；config.json 的
    主体在训练开始前已经写出，这里只追加事后测量结果。
    """
    path = out_dir / "config.json"
    payload: Dict[str, object] = {}

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}

    payload[key] = value

    previous_mode = 0o644
    if path.exists():
        try:
            previous_mode = path.stat().st_mode & 0o777
        except OSError:
            previous_mode = 0o644

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(out_dir),
        prefix=".config.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        temp_path = Path(handle.name)

    os.replace(temp_path, path)

    try:
        os.chmod(path, previous_mode)
    except OSError:
        pass


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
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
            )
            + "\n"
        )


def read_metrics_records(path: Path) -> List[Dict[str, object]]:
    """
    读取 metrics.jsonl 的全部记录；跳过损坏的行。

    训练结束后的曲线诊断、坏案例报告与 --compare 汇总都基于
    这份记录，保证分析与训练写盘使用同一份数据来源。
    """
    if not path.exists():
        return []

    records: List[Dict[str, object]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)

    return records


def test_record_exists(path: Path) -> bool:
    """
    检查 metrics.jsonl 中是否已有 test 评估记录。

    测试集在一次运行中只允许解封评估一次：恢复训练后重复传入
    --evaluate-test 会被拒绝，避免测试集在反复查看中变成隐式
    验证集。
    """
    if not path.exists():
        return False

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(record, dict)
                    and record.get("step") == "test"
                ):
                    return True
    except OSError:
        return False

    return False


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

    # 数据协议：按时间顺序先切成 train/val/test 三段再处理，
    # 切分不随机打乱、互不重叠；训练期不允许访问 test。
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

        # train = [0, 232), val = [232, 244), test = [244, 256)。
        assert len(eval_dataset.train) == 232
        assert int(eval_dataset.train[0]) == 0
        assert int(eval_dataset.train[-1]) == 231
        assert int(eval_dataset.val[0]) == 232
        assert int(eval_dataset.val[-1]) == 243
        assert int(eval_dataset.test[0]) == 244
        assert int(eval_dataset.test[-1]) == 255
        assert eval_dataset.has_test

        # batch() 是训练期唯一的数据入口，必须拒绝 test。
        try:
            eval_dataset.batch(
                "test",
                2,
                torch.Generator().manual_seed(0),
                eval_device,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "batch() 不允许访问测试集"
            )

        # 随机训练批只落在训练 split 内，且输入/标签按位对齐
        # （下一 token 监督，无错位标签噪声）。
        train_x, train_y = eval_dataset.batch(
            "train",
            4,
            torch.Generator().manual_seed(7),
            eval_device,
        )
        assert int(train_x.min()) >= 0
        assert int(train_x.max()) <= 231
        assert int(train_y.max()) <= 231
        assert torch.equal(
            train_x[:, 1:],
            train_y[:, :-1].to(torch.int32),
        )

        # 测试集的固定评估窗口只能来自测试 split。
        test_x, test_y = eval_dataset.fixed_eval_batch(
            "test",
            0,
            1,
            2,
            eval_device,
        )
        assert int(test_x.min()) >= 244
        assert int(test_y.max()) <= 255

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

    val_starts = eval_dataset._fixed_starts["val"]["starts"]
    assert (
        len(overfull_first[1]) + len(overfull_last[1])
        == len(val_starts)
    )
    assert torch.equal(
        val_starts.sort().values,
        val_starts.unique(sorted=True),
    )
    assert overfull_empty[0].numel() == 0
    assert overfull_empty[1].numel() == 0

    # evaluate 的 split 参数：val 指标带 val_ 前缀，test 指标带
    # test_ 前缀；两者窗口互不相同。
    eval_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    val_metrics = evaluate(
        eval_model,
        eval_dataset,
        2,
        1,
        eval_device,
    )
    assert "val_loss" in val_metrics
    assert "val_ppl" in val_metrics

    test_metrics = evaluate(
        eval_model,
        eval_dataset,
        2,
        1,
        eval_device,
        split="test",
    )
    assert "test_loss" in test_metrics
    assert "val_loss" not in test_metrics

    # 数据审计：顺序无重复语料不应报告跨 split 重复；
    # 重复内容必须被捕获并给出警告。
    clean_audit = audit_data_splits(
        eval_dataset,
        include_test=True,
    )
    assert (
        clean_audit["duplicate_windows"]["train_val"][
            "shared_signatures"
        ]
        == 0
    )
    assert (
        clean_audit["duplicate_windows"]["train_test"][
            "shared_signatures"
        ]
        == 0
    )
    assert not any(
        "重复" in warning
        for warning in clean_audit["warnings"]
    )

    with tempfile.NamedTemporaryFile() as repeated_file:
        repeated_file.write(bytes(range(64)) * 4)
        repeated_file.flush()

        repeated_dataset = ByteDataset(
            repeated_file.name,
            seq_len=8,
        )

        repeated_audit = audit_data_splits(
            repeated_dataset,
            include_test=True,
        )

    assert (
        repeated_audit["duplicate_windows"]["train_val"][
            "shared_signatures"
        ]
        > 0
    )
    assert (
        repeated_audit["duplicate_windows"]["train_test"][
            "shared_signatures"
        ]
        > 0
    )
    assert any(
        "重复" in warning
        for warning in repeated_audit["warnings"]
    )

    # 单个字节占主导的退化语料必须被标记（异常值检查）。
    with tempfile.NamedTemporaryFile() as padded_file:
        padded_file.write(
            b"\x00" * 220 + bytes(range(36))
        )
        padded_file.flush()

        padded_dataset = ByteDataset(
            padded_file.name,
            seq_len=8,
        )

        padded_audit = audit_data_splits(
            padded_dataset,
            include_test=True,
        )

    assert any(
        "退化" in warning
        for warning in padded_audit["warnings"]
    )

    # 分布偏移统计量必须是 [0, 1] 的有限数。
    for distance in clean_audit[
        "distribution_shift"
    ].values():
        assert math.isfinite(distance)
        assert 0.0 <= distance <= 1.0

    # metrics.jsonl 中的测试记录只能出现一次：已有记录时必须
    # 拒绝重复解封。
    with tempfile.TemporaryDirectory() as metrics_dir:
        metrics_file = Path(metrics_dir) / "metrics.jsonl"

        assert not test_record_exists(metrics_file)

        write_jsonl_record(
            metrics_file,
            {"step": "final", "val_loss": 1.0},
        )
        assert not test_record_exists(metrics_file)

        write_jsonl_record(
            metrics_file,
            {"step": "test", "test_loss": 1.0},
        )
        assert test_record_exists(metrics_file)

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

    # ---- 结果分析工具自测 ----
    # NaN/Inf 写盘前必须转成 null，保证 JSON 严格可解析。
    assert json.dumps(
        _json_safe({"a": float("nan"), "b": [float("inf"), 1.0]})
    ) == '{"a": null, "b": [null, 1.0]}'

    with tempfile.TemporaryDirectory() as safe_dir:
        safe_path = Path(safe_dir) / "metrics.jsonl"
        write_jsonl_record(
            safe_path,
            {"step": 1, "train_loss": float("nan")},
        )
        safe_records = read_metrics_records(safe_path)
    assert safe_records[0]["train_loss"] is None

    # 曲线诊断必须区分过拟合 / 欠拟合 / 未收敛 / 发散 / 健康。
    baseline = math.log(256.0)

    overfit_analysis = analyze_training_curve(
        [
            {"step": 10, "train_loss": 5.0, "val_loss": 5.10},
            {"step": 20, "train_loss": 4.0, "val_loss": 4.50},
            {"step": 30, "train_loss": 3.0, "val_loss": 4.40},
            {"step": 40, "train_loss": 2.0, "val_loss": 4.60},
            {"step": 50, "train_loss": 1.5, "val_loss": 4.90},
        ],
        baseline_loss=baseline,
    )
    assert overfit_analysis["status"] == "overfitting", (
        overfit_analysis
    )
    assert overfit_analysis["best_val_step"] == 30

    underfit_analysis = analyze_training_curve(
        [
            {"step": 10, "train_loss": 5.50, "val_loss": 5.55},
            {"step": 20, "train_loss": 5.49, "val_loss": 5.54},
            {"step": 30, "train_loss": 5.48, "val_loss": 5.53},
        ],
        baseline_loss=baseline,
    )
    assert underfit_analysis["status"] == "underfitting", (
        underfit_analysis
    )

    descending_analysis = analyze_training_curve(
        [
            {"step": 10, "train_loss": 5.0, "val_loss": 5.2},
            {"step": 20, "train_loss": 4.2, "val_loss": 4.4},
            {"step": 30, "train_loss": 3.4, "val_loss": 3.6},
        ],
        baseline_loss=baseline,
    )
    assert descending_analysis["status"] == "not_converged", (
        descending_analysis
    )

    healthy_analysis = analyze_training_curve(
        [
            {"step": 10, "train_loss": 5.0, "val_loss": 5.1},
            {"step": 20, "train_loss": 4.0, "val_loss": 4.3},
            {"step": 30, "train_loss": 3.18, "val_loss": 3.45},
            {"step": 40, "train_loss": 3.15, "val_loss": 3.46},
        ],
        baseline_loss=baseline,
    )
    assert healthy_analysis["status"] == "healthy", (
        healthy_analysis
    )

    diverged_analysis = analyze_training_curve(
        [
            {"step": 10, "train_loss": 5.0, "val_loss": 5.1},
            {"step": 20, "train_loss": None, "val_loss": None},
            {
                "step": 30,
                "train_loss": 4.9,
                "grad_norm": None,
            },
        ],
        baseline_loss=baseline,
    )
    assert diverged_analysis["status"] == "diverged", (
        diverged_analysis
    )
    assert diverged_analysis["n_nonfinite_events"] == 2

    # 早停状态恢复：最优值与连续未改善次数必须与历史一致。
    (
        early_best,
        early_best_step,
        early_no_improve,
    ) = early_stop_state_from_records(
        [
            {"step": 10, "val_loss": 5.0},
            {"step": 20, "val_loss": 4.0},
            {"step": 30, "val_loss": 4.1},
            {"step": 40, "val_loss": 4.05},
        ],
        min_delta=0.0,
    )
    assert early_best == 4.0
    assert early_best_step == 20
    assert early_no_improve == 2

    # NaN/Inf 排查：越界数据必须被发现；非有限梯度必须被定位。
    bad_data_check = _check_batch_integrity(
        torch.tensor([[0, 300]], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.uint8),
        256,
    )
    assert not bad_data_check["ok"]

    # uint8 与 Python 整数比较会回绕（255 >= 256 误判为越界），
    # 上限内的 uint8 token 必须判定为合法。
    uint8_range_check = _check_batch_integrity(
        torch.tensor([[0, 255]], dtype=torch.uint8),
        torch.tensor([[0, 255]], dtype=torch.uint8),
        256,
    )
    assert uint8_range_check["ok"], uint8_range_check

    diag_cfg = Config(
        **common,
        mode="B",
    )
    diag_model = LanguageModel(diag_cfg)
    diag_model.train()
    diag_optimizer = torch.optim.AdamW(
        diag_model.parameters(),
        lr=1e-3,
    )

    with torch.enable_grad():
        diag_result = diag_model(x, y)
        diag_result["loss"].backward()

    diag_param = next(diag_model.parameters())
    assert diag_param.grad is not None
    diag_param.grad.fill_(float("nan"))

    diag_report = diagnose_non_finite(
        diag_model,
        diag_optimizer,
        x,
        y,
        diag_result,
        torch.tensor(float("nan")),
        step=1,
        lr=1e-3,
        amp_dtype=None,
        vocab_size=256,
    )
    assert diag_report["checks"]["data"]["ok"]
    assert (
        diag_report["checks"]["gradients"][
            "n_nonfinite_grad_params"
        ]
        >= 1
    )

    # 坏案例分析：结构完整、类别合法，且只使用 val 固定窗口。
    bad_case_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    ).eval()

    bad_case_report = analyze_bad_cases(
        bad_case_model,
        eval_dataset,
        batch_size=2,
        batches=2,
        device=eval_device,
        amp_dtype=None,
        split="val",
        top_n=3,
    )

    valid_categories = {
        "ambiguous",
        "systematic_confusion",
        "label_noise_suspect",
        "uncertain",
    }

    assert bad_case_report["tokens_evaluated"] > 0
    assert bad_case_report["split"] == "val"
    assert bad_case_report["token_accuracy"] is not None
    assert 0.0 <= float(
        bad_case_report["token_accuracy"]
    ) <= 1.0
    assert len(bad_case_report["top_cases"]) == min(
        3,
        int(bad_case_report["tokens_evaluated"]),
    )
    for case in bad_case_report["top_cases"]:
        assert case["category"] in valid_categories
        assert case["context"]
        assert 0.0 <= float(case["confidence"]) <= 1.0

    assert set(
        bad_case_report["category_counts"].keys()
    ) <= valid_categories

    # 效率测量：参数量与模型一致，各测量值有效或明确不可用。
    efficiency_model = LanguageModel(
        Config(
            **common,
            mode="B",
        )
    )
    efficiency = measure_efficiency(
        efficiency_model,
        eval_dataset,
        eval_device,
        batch_size=2,
        amp_dtype=None,
        warmup=0,
        iters=1,
    )

    assert efficiency["n_params_total"] == sum(
        parameter.numel()
        for parameter in efficiency_model.parameters()
    )
    assert (
        efficiency["n_params_non_embedding"]
        <= efficiency["n_params_total"]
    )
    assert efficiency["latency_ms_per_forward"] is not None
    assert float(efficiency["latency_ms_per_forward"]) > 0.0
    assert (
        efficiency["flops_per_forward"] is None
        or int(efficiency["flops_per_forward"]) > 0
    )
    # attention 的 FLOPs 由 Block 级 hook 估计；Block 被真实
    # 执行时该值必须为正，否则说明 hook 挂错了对象。
    assert (
        efficiency["flops_attention"] is None
        or int(efficiency["flops_attention"]) > 0
    )
    assert efficiency["measurement"]["mode"] == "B"

    # 运行对比：从 config.json / metrics.jsonl 读出汇总行。
    with tempfile.TemporaryDirectory() as compare_dir:
        for name, val_loss in (
            ("run_a", 3.0),
            ("run_b", 2.9),
        ):
            run_dir = Path(compare_dir) / name
            run_dir.mkdir()
            with open(
                run_dir / "config.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "config": {"mode": "B"},
                        "n_params": 100,
                        "efficiency": {
                            "latency_ms_per_forward": 1.5,
                        },
                    },
                    f,
                )
            write_jsonl_record(
                run_dir / "metrics.jsonl",
                {
                    "step": "final",
                    "val_loss": val_loss,
                    "val_ppl": 10.0,
                },
            )

        comparison_rows = compare_runs([
            str(Path(compare_dir) / "run_a"),
            str(Path(compare_dir) / "run_b"),
        ])

    assert len(comparison_rows) == 2
    assert comparison_rows[0]["val_loss"] == 3.0
    assert comparison_rows[1]["mode"] == "B"
    assert comparison_rows[0][
        "latency_ms_per_forward"
    ] == 1.5

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

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="训练语料文件（--compare 模式可省略）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="B",
        choices=["smelt", "local_base", "A", "B"],
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出目录（--compare 模式可省略）",
    )
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help=(
            "按时间顺序划给验证集的尾部比例（默认 0.05）；"
            "划分不随机打乱"
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help=(
            "按时间顺序划给测试集的尾部比例（默认 0.05）；"
            "测试集训练期间尘封，仅在 --evaluate-test 时评估一次"
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "训练结束后在测试集上做唯一一次最终评估；"
            "不设置时测试集完全不被访问"
        ),
    )

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
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "按验证损失早停：连续 N 次周期验证无改善后提前结束"
            "训练（0=关闭）；早停状态跨 --resume 从 metrics.jsonl"
            "恢复"
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help=(
            "早停判定的最小改善幅度：验证损失需下降超过该值"
            "才算改善（默认 0，即任何下降都算）"
        ),
    )
    parser.add_argument(
        "--bad-cases",
        type=int,
        default=10,
        help=(
            "训练结束后在验证集上输出损失最高的 N 个坏案例"
            "（0=关闭）；--evaluate-test 时同一次解封中"
            "也输出测试集坏案例"
        ),
    )
    parser.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="跳过训练结束后的效率测量",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "不训练，直接对比多个运行目录的指标"
            "后退出"
        ),
    )

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
    if args.sanity_only and args.evaluate_test:
        parser.error(
            "--evaluate-test 需要完整训练后的模型，"
            "不能与 --sanity-only 同时使用"
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
    if (
        not math.isfinite(args.val_fraction)
        or not 0.0 <= args.val_fraction < 1.0
    ):
        parser.error("--val-fraction 必须落在 [0, 1)")
    if (
        not math.isfinite(args.test_fraction)
        or not 0.0 <= args.test_fraction < 1.0
    ):
        parser.error("--test-fraction 必须落在 [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        parser.error(
            "--val-fraction + --test-fraction 必须小于 1"
            "（训练集不能为空）"
        )
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience 不能为负")
    if (
        not math.isfinite(args.early_stop_min_delta)
        or args.early_stop_min_delta < 0
    ):
        parser.error(
            "--early-stop-min-delta 必须是有限非负数"
        )
    if args.bad_cases < 0:
        parser.error("--bad-cases 不能为负")

    # --compare 模式只读运行目录，不需要 --data / --out。
    if args.compare is not None:
        print_comparison(compare_runs(args.compare))
        return

    if args.data is None:
        parser.error("--data 必填（除非使用 --compare）")
    if args.out is None:
        parser.error("--out 必填（除非使用 --compare）")

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
    saved_data_val_fraction = None
    saved_data_test_fraction = None

    if args.resume is not None:
        # 检查点先载入 CPU，再按目标设备分别搬运。
        rng_payload = load_checkpoint(
            args.resume,
            torch.device("cpu"),
        )
        saved_args = rng_payload.get("args", {})
        saved_data_sha256 = saved_args.get("data_sha256")
        saved_data_val_fraction = saved_args.get(
            "data_val_fraction"
        )
        saved_data_test_fraction = saved_args.get(
            "data_test_fraction"
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

        # 数据划分比例决定 train/val/test 的字节位置，是采样
        # 轨迹的一部分；resume 时按存档恢复，避免数据管线换了
        # 位置而检查点的 RNG 状态仍在旧轨迹上。
        if (
            saved_data_val_fraction is not None
            and saved_data_val_fraction != args.val_fraction
        ):
            print(
                "resume restored val_fraction "
                f"{args.val_fraction} -> "
                f"{saved_data_val_fraction} "
                "(数据划分是采样轨迹的一部分)"
            )
            args.val_fraction = saved_data_val_fraction

        if (
            saved_data_test_fraction is not None
            and saved_data_test_fraction != args.test_fraction
        ):
            print(
                "resume restored test_fraction "
                f"{args.test_fraction} -> "
                f"{saved_data_test_fraction} "
                "(数据划分是采样轨迹的一部分)"
            )
            args.test_fraction = saved_data_test_fraction

    cfg.validate()

    # 数据管线先于模型准备：sanity check 需要在创建全量模型之前
    # 先用同一数据管线验证模型代码能够过拟合小批量，同时拿到
    # 数据版本指纹写入实验追踪信息。
    dataset = ByteDataset(
        args.data,
        cfg.seq_len,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        device=device,
    )

    if args.evaluate_test and not dataset.has_test:
        parser.error(
            "--evaluate-test 需要有效的测试集："
            "请设置正的 --test-fraction 且语料足够长"
        )

    # 语料指纹/大小/划分比例写入 args，随 config.json 与检查点
    # 一起存档；恢复训练时若数据文件内容或划分位置已变化，采样
    # 轨迹会与原运行不同，这里明确提示而不是静默继续。
    args.data_sha256 = dataset.data_sha256
    args.data_bytes = dataset.num_bytes
    args.data_val_fraction = dataset.val_fraction
    args.data_test_fraction = dataset.test_fraction

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
    elif saved_data_sha256 is not None and (
        saved_data_val_fraction is None
        or saved_data_test_fraction is None
    ):
        print(
            "resume checkpoint predates the train/val/test "
            "split; 数据划分位置可能与原运行不同，"
            "采样轨迹不再一致"
        )
    elif saved_data_sha256 is not None and (
        saved_data_val_fraction != dataset.val_fraction
        or saved_data_test_fraction != dataset.test_fraction
    ):
        print(
            "resume data split fractions changed "
            f"(val {saved_data_val_fraction} -> "
            f"{dataset.val_fraction}, "
            f"test {saved_data_test_fraction} -> "
            f"{dataset.test_fraction}); "
            "采样轨迹与原运行不再一致"
        )

    # 数据审计在训练之前完成：重复内容、分布偏移与 token 值域
    # 问题会被记录并在发现风险时给出警告。test 此时保持尘封，
    # 只在 --evaluate-test 的最终评估时参与审计。
    data_audit = audit_data_splits(dataset, include_test=False)
    print_data_audit(data_audit)

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
            data_audit=data_audit,
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
    # 路径逐位一致（已验证）。
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
    # 数据指纹与划分、数据审计、代码版本/运行环境与 sanity check
    # 结果。
    write_run_config(
        out_dir,
        cfg,
        args,
        n_params,
        collect_run_metadata(device),
        config_sanity_result,
        data_audit=data_audit,
    )

    metrics_path = out_dir / "metrics.jsonl"

    # 早停状态：resume 时从原运行的 metrics.jsonl 恢复「最优验证
    # 损失 / 最优步 / 连续未改善次数」，保证早停判定跨恢复一致；
    # 全新训练不读旧文件（文件会被截断重写）。
    best_val_loss: Optional[float] = None
    best_val_step: Optional[int] = None
    evals_without_improvement = 0

    if (
        args.early_stop_patience > 0
        and args.resume is not None
    ):
        (
            best_val_loss,
            best_val_step,
            evals_without_improvement,
        ) = early_stop_state_from_records(
            read_metrics_records(metrics_path),
            args.early_stop_min_delta,
        )

    early_stopped = False
    nonfinite_events = 0
    consecutive_nonfinite = 0
    numerical_stop = False
    numerical_polluted = False
    nan_diagnosis: Optional[Dict[str, object]] = None
    nan_record_pending = False

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
                _json_safe(sanity_record),
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
                    _json_safe(record),
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

    # 真正完成（前向+反向+更新判定）的最后一步；早停提前退出
    # 时用它作为存档与最终评估的步数。
    last_completed_step = start_step

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

            # 损失与梯度范数都必须有限才推进优化器：任一非有限
            # 都意味着这一步的参数更新不可信。
            finite_step = bool(
                torch.isfinite(grad_norm)
                & torch.isfinite(loss)
            )

            if finite_step:
                scaler.step(optimizer)
                scheduler.step()
                consecutive_nonfinite = 0
            else:
                nonfinite_events += 1
                consecutive_nonfinite += 1
                print(
                    f"step {step + 1}: non-finite loss/gradient; "
                    "optimizer and LR schedule skipped"
                )
                if nan_diagnosis is None:
                    nan_diagnosis = diagnose_non_finite(
                        model,
                        optimizer,
                        x,
                        y,
                        result,
                        grad_norm,
                        step=step + 1,
                        lr=current_lr,
                        amp_dtype=amp_dtype,
                        scaler=scaler,
                        vocab_size=cfg.vocab_size,
                    )
                    print_non_finite_report(nan_diagnosis)
                    nan_record_pending = True
                    nan_checks = nan_diagnosis.get("checks", {})
                    if (
                        nan_checks.get("parameters_nonfinite")
                        or nan_checks.get(
                            "optimizer_state_nonfinite_keys"
                        )
                    ):
                        # 参数/优化器状态已被污染，继续训练毫无
                        # 意义（每次更新都被跳过）。
                        numerical_polluted = True
                        numerical_stop = True
                if consecutive_nonfinite >= 3:
                    numerical_stop = True
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params,
                args.grad_clip,
            )

            # 发散步不推进优化器或 LR schedule。
            finite_step = bool(
                torch.isfinite(grad_norm)
                & torch.isfinite(loss)
            )

            if finite_step:
                optimizer.step()
                scheduler.step()
                consecutive_nonfinite = 0
            else:
                nonfinite_events += 1
                consecutive_nonfinite += 1
                print(
                    f"step {step + 1}: non-finite loss/gradient; "
                    "optimizer and LR schedule skipped"
                )
                if nan_diagnosis is None:
                    nan_diagnosis = diagnose_non_finite(
                        model,
                        optimizer,
                        x,
                        y,
                        result,
                        grad_norm,
                        step=step + 1,
                        lr=current_lr,
                        amp_dtype=amp_dtype,
                        scaler=scaler,
                        vocab_size=cfg.vocab_size,
                    )
                    print_non_finite_report(nan_diagnosis)
                    nan_record_pending = True
                    nan_checks = nan_diagnosis.get("checks", {})
                    if (
                        nan_checks.get("parameters_nonfinite")
                        or nan_checks.get(
                            "optimizer_state_nonfinite_keys"
                        )
                    ):
                        numerical_polluted = True
                        numerical_stop = True
                if consecutive_nonfinite >= 3:
                    numerical_stop = True

        # 本步已完成（无论是否推进了优化器）。
        last_completed_step = step + 1

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

        # 首次非有限事件的排查报告附着到触发它的那一步，
        # 保持 metrics.jsonl 的 step 顺序与证据链完整。
        if nan_record_pending:
            pending_records[-1]["extra"] = {
                "nan_diagnosis": nan_diagnosis,
            }
            nan_record_pending = False

        # 参数/优化器状态已被污染或连续非有限达到阈值时停止：
        # 后续更新全部会被跳过，继续训练只会浪费算力。
        if numerical_stop:
            consecutive_text = (
                "参数/优化器状态已被 NaN/Inf 污染"
                if numerical_polluted
                else "连续多次出现非有限损失/梯度"
            )
            print(
                f"[nan] {consecutive_text}，在第 {step + 1} 步"
                "停止训练；请按排查路径处理后从污染前的检查点"
                "恢复（降低学习率、收紧 --grad-clip）。"
                + (
                    "本次不会用受污染的参数覆盖 last.pt。"
                    if numerical_polluted
                    else ""
                )
            )
            break

        if (step + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(
                model,
                dataset,
                args.batch_size,
                args.eval_batches,
                device,
                amp_dtype=amp_dtype,
            )
            # 同一步已有 nan_diagnosis 时合并而不是覆盖。
            pending_records[-1].setdefault(
                "extra",
                {},
            ).update(eval_metrics)

            last_eval_step = step + 1
            last_eval_metrics = eval_metrics

            records = flush_records()
            print(records[-1])
            metrics_file.flush()
            model.train()

            # 早停：验证损失连续无改善达到 patience 时提前结束。
            # NaN/Inf 验证损失不计入「未改善」，避免故障顺带触发
            # 早停而掩盖问题。
            if args.early_stop_patience > 0:
                val_loss = eval_metrics.get("val_loss")

                if val_loss is None or not math.isfinite(
                    float(val_loss)
                ):
                    pass
                elif (
                    best_val_loss is None
                    or float(val_loss)
                    < best_val_loss
                    - args.early_stop_min_delta
                ):
                    best_val_loss = float(val_loss)
                    best_val_step = step + 1
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                if (
                    evals_without_improvement
                    >= args.early_stop_patience
                ):
                    early_stopped = True
                    print(
                        "[early-stop] 验证损失连续 "
                        f"{evals_without_improvement} 次未改善"
                        "（最优 "
                        f"{best_val_loss:.4f} @ step "
                        f"{best_val_step}），"
                        f"在第 {step + 1} 步提前结束训练；"
                        "如需继续训练可调大 --steps 后 resume。"
                    )
                    break

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

    # 结尾存档的 step 取优化器真实前进到的步数：正常结束时等于
    # args.steps，早停时等于提前结束的步数；循环未执行时
    # （start_step >= args.steps）取 start_step。
    end_step = last_completed_step

    if last_saved_step != end_step and not numerical_polluted:
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
        "early_stopped": early_stopped,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "nonfinite_events": nonfinite_events,
        "numerical_stop": numerical_stop,
        "numerical_polluted": numerical_polluted,
    }
    metrics_file.write(
        json.dumps(
            _json_safe(final_record),
            ensure_ascii=False,
        )
        + "\n"
    )
    metrics_file.flush()

    print(
        json.dumps(
            _json_safe({"final": final_metrics}),
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # 结果分析 1：训练曲线诊断（不能只看最终单个数值）。
    # --------------------------------------------------------
    curve_analysis = analyze_training_curve(
        read_metrics_records(metrics_path),
        baseline_loss=math.log(cfg.vocab_size),
    )
    metrics_file.write(
        json.dumps(
            _json_safe({
                "step": "curve_analysis",
                **curve_analysis,
            }),
            ensure_ascii=False,
        )
        + "\n"
    )
    metrics_file.flush()
    print_curve_analysis(curve_analysis)

    # --------------------------------------------------------
    # 结果分析 2：坏案例分析（验证集，测试集保持尘封）。
    # 错误的样本往往是新改进思路的最重要来源。
    # --------------------------------------------------------
    bad_cases_path = out_dir / "bad_cases.json"
    bad_case_payload: Dict[str, object] = {}

    if args.bad_cases > 0:
        bad_case_report = analyze_bad_cases(
            model,
            dataset,
            args.batch_size,
            args.eval_batches,
            device,
            amp_dtype=amp_dtype,
            split="val",
            top_n=args.bad_cases,
        )
        print_bad_cases(bad_case_report)

        bad_case_payload["val"] = bad_case_report
        metrics_file.write(
            json.dumps(
                _json_safe({
                    "step": "bad_cases",
                    "split": "val",
                    "tokens_evaluated": bad_case_report[
                        "tokens_evaluated"
                    ],
                    "token_accuracy": bad_case_report[
                        "token_accuracy"
                    ],
                    "category_counts": bad_case_report[
                        "category_counts"
                    ],
                    "top_confusions": bad_case_report[
                        "top_confusions"
                    ],
                    "top_cases": bad_case_report["top_cases"],
                    "summary": bad_case_report["summary"],
                }),
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()

    # --------------------------------------------------------
    # 结果分析 3：效率测量。
    # --------------------------------------------------------
    if not args.skip_efficiency:
        efficiency = measure_efficiency(
            model,
            dataset,
            device,
            args.batch_size,
            amp_dtype=amp_dtype,
        )
        update_run_config(out_dir, "efficiency", efficiency)
        metrics_file.write(
            json.dumps(
                _json_safe({
                    "step": "efficiency",
                    **efficiency,
                }),
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()
        print_efficiency(efficiency)

    # 最终评估口径：上面的 final 记录始终是验证集口径。
    # 测试集默认全程尘封，只有显式 --evaluate-test 时才在训练
    # 完全结束后解封，且同一次运行只允许评估一次；重复传入也
    # 不会让测试集变成隐式验证集，需要新的结论应使用全新目录。
    if args.evaluate_test:
        if test_record_exists(metrics_path):
            print(
                "[test] metrics.jsonl 中已有测试集评估记录；"
                "测试集在该目录的运行中已解封过一次，拒绝重复评估。"
                "如需新的最终结论，请使用全新的 --out 目录，"
                "并在确认模型与超参数冻结后再加 --evaluate-test。"
            )
        else:
            # 解封测试集：先做包含 test 的数据审计，再做唯一一次
            # 最终评估。
            test_audit = audit_data_splits(
                dataset,
                include_test=True,
            )
            print_data_audit(test_audit)

            test_metrics = evaluate(
                model,
                dataset,
                args.batch_size,
                args.eval_batches,
                device,
                amp_dtype=amp_dtype,
                split="test",
            )

            # 坏案例分析随唯一一次解封一起做（val 分析照常）。
            if args.bad_cases > 0:
                test_bad_cases = analyze_bad_cases(
                    model,
                    dataset,
                    args.batch_size,
                    args.eval_batches,
                    device,
                    amp_dtype=amp_dtype,
                    split="test",
                    top_n=args.bad_cases,
                )
                print_bad_cases(test_bad_cases)
                bad_case_payload["test"] = test_bad_cases
                metrics_file.write(
                    json.dumps(
                        _json_safe({
                            "step": "bad_cases",
                            "split": "test",
                            "tokens_evaluated": test_bad_cases[
                                "tokens_evaluated"
                            ],
                            "token_accuracy": test_bad_cases[
                                "token_accuracy"
                            ],
                            "category_counts": test_bad_cases[
                                "category_counts"
                            ],
                            "top_confusions": test_bad_cases[
                                "top_confusions"
                            ],
                            "top_cases": test_bad_cases[
                                "top_cases"
                            ],
                            "summary": test_bad_cases["summary"],
                        }),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                metrics_file.flush()

            test_record = {
                "step": "test",
                **test_metrics,
                "data_audit": test_audit,
            }
            metrics_file.write(
                json.dumps(
                    _json_safe(test_record),
                    ensure_ascii=False,
                )
                + "\n"
            )
            metrics_file.flush()

            print(
                json.dumps(
                    _json_safe({"test": test_metrics}),
                    ensure_ascii=False,
                )
            )
            print(
                "[test] 测试集已解封并完成唯一一次评估；"
                "请不要根据该结果继续调参，否则测试集将退化为"
                "隐式验证集。"
            )

    if bad_case_payload:
        with open(
            bad_cases_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_safe(bad_case_payload),
                f,
                ensure_ascii=False,
                indent=2,
            )

    metrics_file.close()


if __name__ == "__main__":
    if "--self-test" in __import__("sys").argv:
        run_self_test()
    else:
        main()
