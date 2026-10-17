from __future__ import annotations

import argparse
import codecs
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

    # smelt / local_base / A / B / dense / ngram。
    # dense：无 MoE、无循环的纯稠密 Transformer 基线；
    # ngram：仅用于统计基线运行的配置标记，LanguageModel
    # 不支持该模式。
    mode: str = "B"

    # dense 模式的 FFN 隐藏宽度。None 时默认取 top_k * expert_hidden，
    # 即与 MoE 模式单次前向激活的专家参数量对齐（等计算口径）；
    # 要做等参数口径对比时可显式设为 n_experts * expert_hidden。
    dense_fnn_hidden: Optional[int] = None

    balance_coef: float = 0.01
    residual_scale: float = 1.0

    # LanguageModel 可以构建的模式；ngram 仅供统计基线运行写
    # config.json 使用。
    MODEL_MODES = {"smelt", "local_base", "A", "B", "dense"}
    ALL_MODES = MODEL_MODES | {"ngram"}

    @property
    def effective_dense_fnn_hidden(self) -> int:
        if self.dense_fnn_hidden is not None:
            return self.dense_fnn_hidden
        # 等计算口径：top_k 个专家的激活参数量。
        return self.top_k * self.expert_hidden

    def validate(self):
        if self.mode not in self.ALL_MODES:
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
        if self.mode == "dense":
            hidden = self.effective_dense_fnn_hidden
            if hidden <= 0:
                raise ValueError(
                    "dense 模式的 FFN 隐藏宽度必须为正整数"
                    f"（当前解析为 {hidden}）"
                )
        if self.dense_fnn_hidden is not None and (
            not isinstance(self.dense_fnn_hidden, int)
            or self.dense_fnn_hidden <= 0
        ):
            raise ValueError(
                "dense_fnn_hidden 必须是正整数或 None"
            )

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
        gate_up = F.linear(x, self.gate_up.weight)

        # 无反向需求（no_grad/inference_mode 下的评估路径）时
        # 直接原地计算，跳过 autograd.Function 的调度开销；
        # 数值上与 _SwiGLU 的无反向分支完全一致。
        if gate_up.requires_grad:
            hidden = _SwiGLU.apply(gate_up)
        else:
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

        # 辅助负载均衡项只在 balance_coef > 0 且梯度开启时
        # 参与总损失；其余情况下（balance_coef=0 的消融臂、
        # 评估与 sanity check）不必为其建图与保存中间量，
        # 在 no_grad 下计算同一数值即可。
        if (
            self.balance_coef > 0.0
            and torch.is_grad_enabled()
        ):
            router_probs = router_logits.softmax(dim=-1)
        else:
            with torch.no_grad():
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
        合并残差，语义不变。所有路径均为 out-of-place，
        不修改调用方传入的张量。
        """
        top_k = expert_outputs.size(-2)
        dim = expert_outputs.size(-1)

        # top_k == 1 时门控权重恒为 1.0（单元素 softmax），
        # 加权和退化为恒等；有残差时完成「缩放 + 残差相加」。
        if top_k == 1:
            output = expert_outputs.reshape(-1, dim)

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
        # dense 基线：无路由，无辅助损失；top_k 参数被忽略，
        # 保持与 MoE 模式一致的调用签名。
        if self.is_dense:
            y = self.ffn(self.norm_moe(x))
            return (
                torch.add(
                    x,
                    y,
                    alpha=self.residual_scale,
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

        if cfg.mode not in Config.MODEL_MODES:
            raise ValueError(
                f"LanguageModel 不支持 mode={cfg.mode}"
                "（ngram 等统计基线不走本类）"
            )

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

        # dense 模式没有中间循环区域，全部 Block 由下面的 dense
        # 分支统一执行；跳过前后区域循环，避免边界 Block 重复执行。
        pre_end = 0 if cfg.mode == "dense" else self.middle_start

        # 中间区域之前的层。
        for index in range(pre_end):
            x, aux = call(
                blocks[index],
                x,
                cfg.top_k,
            )
            aux_losses.append(aux)

        middle = self.middle_blocks

        # ----------------------------------------------------
        # Dense baseline: 纯稠密 Transformer，每个 Block 只
        # 执行一次，无 MoE、无中间层循环。
        # ----------------------------------------------------
        if cfg.mode == "dense":
            for index in range(cfg.n_layers):
                x, aux = call(
                    blocks[index],
                    x,
                    cfg.top_k,
                )
                aux_losses.append(aux)

        # ----------------------------------------------------
        # Original SMELT
        # ----------------------------------------------------
        elif cfg.mode == "smelt":
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

        # 中间区域之后的层（dense 模式已在上面统一执行）。
        post_start = (
            cfg.n_layers if cfg.mode == "dense" else self.middle_end
        )
        for index in range(post_start, cfg.n_layers):
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

# 群组分隔符允许的反斜杠转义：\n / \t / \r / \0 / \\ / \xHH / \uXXXX。
_GROUP_SEPARATOR_SIMPLE_ESCAPES = {
    "n": 10,
    "t": 9,
    "r": 13,
    "0": 0,
    "\\": 92,
    "'": 39,
    '"': 34,
}


def parse_group_separator(text: str) -> bytes:
    """
    把命令行传入的群组分隔符解析为字节序列。

    支持常见反斜杠转义（\\n、\\t、\\r、\\0、\\\\、\\xHH、\\uXXXX），
    其余字符按 UTF-8 编码；解析结果不能为空。分隔符用于把语料
    切成「群组」（如同一文档/同一用户的一段内容），群组是划分
    的最小单位，绝不允许跨 train/val/test 边界。
    """
    result = bytearray()
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char != "\\":
            result.extend(char.encode("utf-8"))
            index += 1
            continue

        if index + 1 >= length:
            raise ValueError("分隔符不能以单个反斜杠结尾")

        escape = text[index + 1]

        if escape in _GROUP_SEPARATOR_SIMPLE_ESCAPES:
            result.append(_GROUP_SEPARATOR_SIMPLE_ESCAPES[escape])
            index += 2
            continue

        if escape in ("x", "u"):
            digits = 2 if escape == "x" else 4
            hex_part = text[index + 2:index + 2 + digits]
            if (
                len(hex_part) != digits
                or any(
                    digit not in "0123456789abcdefABCDEF"
                    for digit in hex_part
                )
            ):
                raise ValueError(
                    f"\\{escape} 转义需要 {digits} 位十六进制数字"
                )
            codepoint = int(hex_part, 16)
            if escape == "u":
                result.extend(chr(codepoint).encode("utf-8"))
            else:
                result.append(codepoint)
            index += 2 + digits
            continue

        raise ValueError(
            f"不支持的分隔符转义: \\{escape}"
        )

    if not result:
        raise ValueError("群组分隔符不能为空")

    return bytes(result)


def _iter_group_boundaries(
    buf: bytearray,
    separator: bytes,
    start: int = 0,
):
    """
    依次产出群组边界位置（每个分隔符之后的字节位置）与文件末尾。

    边界是群的起始字节；文件末尾 N 永远是一个合法边界。
    只产出 > start 的边界。
    """
    position = start

    while True:
        found = buf.find(separator, position)
        if found < 0:
            break
        end = found + len(separator)
        if start < end < len(buf):
            yield end
        position = end

    if len(buf) > start:
        yield len(buf)


def _count_groups(
    buf: bytearray,
    separator: bytes,
    cuts: List[int],
) -> List[int]:
    """
    统计每个 split 内部的完整群组数量（cuts 必须落在群组边界上）。

    群组起点集合 = {0} ∪ {分隔符之后 < N 的边界}；[a, b) 内的
    群组数 = 该区间内的群组起点数。cuts 本身是群组起点：
    恰好等于切分点的边界计入右侧 split 的基础 1，不能再重复
    计入左侧。
    """
    total = len(buf)
    first, second = cuts[0], cuts[-1]
    counts = [1, 1, 1]

    for end in _iter_group_boundaries(buf, separator, 0):
        if end >= total:
            break
        if end < first:
            counts[0] += 1
        elif end == first or end == second:
            # 切分点本身：作为右侧 split 的起始群组计入基础 1。
            continue
        elif end < second:
            counts[1] += 1
        else:
            counts[2] += 1

    if second >= total:
        # test split 为空（test_fraction = 0 时 val 右端即文件末尾）。
        counts[2] = 0

    return counts


def _snap_group_boundary(
    buf: bytearray,
    separator: bytes,
    target: int,
    lower: int,
) -> List[int]:
    """
    返回 > lower 的、离 target 最近的左右两个群组边界候选。
    """
    left: Optional[int] = None
    right: Optional[int] = None

    for end in _iter_group_boundaries(buf, separator, lower):
        if end <= lower:
            continue
        if end <= target:
            left = end
        else:
            right = end
            break

    candidates = [
        candidate
        for candidate in (left, right)
        if candidate is not None
    ]
    return candidates


def _snap_group_split(
    buf: bytearray,
    separator: bytes,
    nominal_train: int,
    nominal_val_end: int,
    min_len: int,
    has_test: bool,
) -> Tuple[int, int]:
    """
    把名义切分位置吸附到群组边界：train/val/test 的分界点都是
    某个群组的起始字节，任何群组整体落在一个 split 内。

    在邻近候选边界中穷举所有可行组合（每个目标左右各一个候选），
    取偏离名义位置总和最小的一组；找不到合法组合时抛出
    ValueError（宁可报错，也不产生一个拆散群组的划分）。
    """
    if buf.find(separator) < 0:
        raise ValueError(
            "语料中没有出现群组分隔符，无法按群组划分；"
            "请检查 --group-separator 是否正确"
        )

    train_candidates = _snap_group_boundary(
        buf,
        separator,
        nominal_train,
        0,
    )

    if not train_candidates:
        raise ValueError(
            "语料中没有可用的群组边界，无法按群组划分；"
            "请检查 --group-separator 是否正确"
        )

    best: Optional[Tuple[int, int, int]] = None

    for cut_train in sorted(
        train_candidates,
        key=lambda end: (abs(end - nominal_train), end),
    ):
        if cut_train < min_len:
            continue

        val_candidates = _snap_group_boundary(
            buf,
            separator,
            nominal_val_end,
            cut_train,
        )

        for cut_val in sorted(
            val_candidates,
            key=lambda end: (abs(end - nominal_val_end), end),
        ):
            if cut_val - cut_train < min_len:
                continue
            if has_test and len(buf) - cut_val < min_len:
                continue

            score = abs(cut_train - nominal_train) + abs(
                cut_val - nominal_val_end
            )
            if best is None or score < best[0]:
                best = (score, cut_train, cut_val)

    if best is None:
        raise ValueError(
            "语料按群组分隔符划分后，无法同时保证 train/val/test "
            f"各自至少 {min_len} 字节（seq_len + 1）；"
            "请调整 --val-fraction / --test-fraction、减小 "
            "--seq-len、检查分隔符，或合并过小的群组"
        )

    return best[1], best[2]


class ByteDataset:
    """
    字节级语言模型数据集。

    数据协议（防止数据泄露）：
    - 先划分、后处理：按原始字节流的时间先后顺序切成
      train / val / test 三段，三个 split 在原语料中前后相邻、
      互不重叠，切分全程不随机打乱（时序数据用过去训练、用未来
      验证和测试）；训练阶段的批级 shuffle 由随机采样窗口起点
      承担，不改变 split 边界；
    - 群组整体不跨界：传入 group_separator 时，切分位置吸附到
      群组边界（同一文档/同一用户的内容不会被拆到两个 split），
      train/val/test 任意两侧之间都不共享半个群组；
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
        group_separator: Optional[bytes] = None,
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

        if group_separator is not None and len(group_separator) == 0:
            raise ValueError("group_separator 不能为空字节串")

        self.group_separator = group_separator
        self.group_cuts: Optional[Dict[str, int]] = None
        self.groups_per_split: Optional[Dict[str, int]] = None

        # 仅做时间顺序上的连续切分，不计算统计量、不打乱：
        # [0, n_train) 训练，[n_train, n_train + n_val) 验证，
        # 其余为测试。
        n_val = int(len(tokens) * val_fraction)
        n_test = int(len(tokens) * test_fraction)
        n_train = len(tokens) - n_val - n_test
        n_val_end = n_train + n_val

        if group_separator is not None:
            # 群组数据的防泄露协议：切分位置吸附到群组边界，
            # 同一群组的全部字节整体落在同一个 split，任何两侧
            # 之间都不交叉。吸附后会验证三段长度，宁可在划分
            # 阶段显式报错（避免无法对齐的评估协议）。
            cut_train, cut_val = _snap_group_split(
                buf,
                group_separator,
                n_train,
                n_val_end,
                min_len=seq_len + 1,
                has_test=test_fraction > 0.0,
            )
            n_train = cut_train
            n_val_end = cut_val
            self.group_cuts = {
                "train": cut_train,
                "val": cut_val,
            }
            self.groups_per_split = dict(
                zip(
                    ("train", "val", "test"),
                    _count_groups(
                        buf,
                        group_separator,
                        [cut_train, cut_val],
                    ),
                )
            )
            # 群组完整性由构造保证：分界点要么是分隔符之后的群组
            # 起始字节，要么是文件末尾（仅在 test_fraction=0 时
            # 作为 val 的右端）。这里再做一次防御性断言，任何
            # 重构错误的划界都会在数据加载阶段立刻暴露，而不是
            # 悄悄泄露。
            for cut in (cut_train, cut_val):
                if 0 < cut < len(buf):
                    if (
                        buf[cut - len(group_separator):cut]
                        != group_separator
                    ):
                        raise AssertionError(
                            "群组切分边界不在分隔符之后："
                            "内部错误，拒绝继续"
                        )

        self.split_bounds = {
            "train": (0, n_train),
            "val": (n_train, n_val_end),
            "test": (n_val_end, len(tokens)),
        }

        self.train = tokens[:n_train]
        self.val = tokens[n_train:n_val_end]
        self.test = tokens[n_val_end:]
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

# UTF-8 解码失败（替换字符）占比超过该阈值时报语料损坏警告：
# 字节级语言模型直接以字节为标签，无法解码的字节就是
# 「标签噪声」的语料来源，必须先发现再决定是否清洗。
_AUDIT_UTF8_REPLACE_WARN = 1e-3

# NUL 字节占比超过该阈值时报二进制/填充污染警告
# （文本语料中 NUL 基本只来自错误编码或文件拼接损坏）。
_AUDIT_NUL_WARN = 1e-4


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


def _scan_text_integrity(
    raw: memoryview,
    offset: int,
    length: int,
    chunk: int = 1 << 20,
    final: bool = False,
) -> Dict[str, int]:
    """
    扫描一个 split 的文本完整性（数据清洗的标签噪声检查）。

    字节级语言模型的标签就是原始字节，损坏字节等价于标签
    噪声，必须在训练前可见：

    - utf8_replacements：以 ``errors="replace"`` 增量解码时产生
      的 U+FFFD 数量，即无法按 UTF-8 解码的字节序列数；
    - nul_bytes：NUL（0x00）字节数量，文本语料中出现即疑似
      二进制污染或文件拼接错误。

    分块解码；只做诊断，不修改任何数据。

    切分边界处不制造假阳性：切分点之前最多 3 个字节先喂给
    解码器做状态预热（多字节 UTF-8 字符可以横跨 split 边界，
    文件整体仍然完全合法）；``final`` 只在扫描的是文件真正
    的末尾时为 True，否则末尾不完整的多字节序列留给下一个
    split 续接（其解码计数落在后续扫描里）。
    """
    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace"
    )

    # 预热：把解码器带到「从 offset 前几个字节开始」的状态。
    # 预热输出（可能包含边界前的字符或无效序列）全部丢弃，
    # 不计入任何统计。
    if offset > 0:
        decoder.decode(
            bytes(raw[max(0, offset - 3):offset]),
            final=False,
        )

    replacements = 0
    nul_bytes = 0
    position = offset
    end = offset + length

    while position < end:
        stop = min(position + chunk, end)
        block = bytes(raw[position:stop])
        text = decoder.decode(
            block,
            final=final and stop >= end,
        )
        replacements += text.count("\ufffd")
        nul_bytes += block.count(0)
        position = stop

    return {
        "utf8_replacements": replacements,
        "nul_bytes": nul_bytes,
    }


def audit_data_splits(
    dataset: ByteDataset,
    include_test: bool = False,
    max_windows: int = _AUDIT_MAX_WINDOWS,
) -> Dict[str, object]:
    """
    数据审计：值域校验、文本完整性、分布偏移与重复窗口检测。

    - 校验每个 split 的 token 值域（字节管线为 0..255；整数
      token 中不存在 NaN/Inf，值域校验用于捕获上游 dtype 或
      编码错误）；
    - 扫描文本完整性（无法 UTF-8 解码的字节序列、NUL 字节），
      作为字节级标签噪声与语料损坏的清洗检查；
    - 统计每个 split 的字节直方图，并计算 train↔val（可选包含
      train↔test、val↔test）之间的 JS 距离，用于发现分布偏移；
    - 对滑动窗口做有界签名扫描，检测跨 split 重复内容与 split
      内部重复；
    - 群组划分（传入 group_separator 时）报告每个 split 的完整
      群组数与分隔符，供追踪「群组整体不跨界」的协议。

    返回的 ``warnings`` 是全部提醒；``violations`` 是其中会
    被 ``--strict-data-audit`` 视为致命的问题（越界 token、
    跨 split 重复内容、超出阈值的 UTF-8/NUL 损坏）——这些
    问题直接对应数据泄露或标签噪声，严格模式下拒绝训练。

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
    violations: List[str] = []

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
            # 先统计越界，再对 clamp 后的值做直方图：
            # bincount 不接受负值，且 clamp 保证直方图长度
            # 恰为 256，跨 split 的 JS 距离形状一致。
            invalid = int(
                ((tokens_long < 0) | (tokens_long > 255)).sum()
            )
            hist = torch.bincount(
                tokens_long.clamp(0, 255),
                minlength=256,
            ).to(torch.float64)

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
        integrity = _scan_text_integrity(
            dataset._raw,
            begin,
            end - begin,
            final=(end == dataset.num_bytes),
        )

        splits_report[split] = {
            "bytes": end - begin,
            "windows": scans[split]["total"],
            "invalid_tokens": invalid,
            "utf8_replacements": integrity["utf8_replacements"],
            "nul_bytes": integrity["nul_bytes"],
            "top_bytes": [
                [int(byte), round(float(share), 6)]
                for share, byte in zip(
                    top_share.tolist(),
                    top_byte.tolist(),
                )
            ],
        }

        if invalid > 0:
            message = f"{split}: 检测到 {invalid} 个越界 token"
            warnings.append(message)
            violations.append(message)

        split_bytes = max(1, end - begin)
        replacement_share = (
            integrity["utf8_replacements"] / split_bytes
        )
        if (
            integrity["utf8_replacements"] > 0
            and replacement_share > _AUDIT_UTF8_REPLACE_WARN
        ):
            message = (
                f"{split}: 检测到 "
                f"{integrity['utf8_replacements']} 处无法按"
                "UTF-8 解码的字节序列"
                f"（占比 {replacement_share:.4f}），"
                "疑似语料损坏（字节级标签噪声）"
            )
            warnings.append(message)
            violations.append(message)

        nul_share = integrity["nul_bytes"] / split_bytes
        if (
            integrity["nul_bytes"] > 0
            and nul_share > _AUDIT_NUL_WARN
        ):
            message = (
                f"{split}: 检测到 {integrity['nul_bytes']} 个"
                f"NUL 字节（占比 {nul_share:.4f}），"
                "疑似二进制/填充污染"
            )
            warnings.append(message)
            violations.append(message)

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
            message = (
                f"{left}_{right}: 检测到"
                f"{len(common)} 个重复窗口内容"
                "（跨 split 泄露风险）"
            )
            warnings.append(message)
            violations.append(message)

    groups_report: Optional[Dict[str, object]] = None

    if dataset.group_separator is not None:
        # 群组划分协议：分界点都是群组起始字节，因此每个群组
        # 整体落在一个 split 内；这里把群组规模写进审计，供
        # 事后核对（数据集构造时还做了防御性断言）。
        groups_report = {
            "separator_hex": dataset.group_separator.hex(),
            "separator_length": len(dataset.group_separator),
            "cuts": dataset.group_cuts,
            "groups_per_split": dataset.groups_per_split,
            "groups_total": (
                sum(dataset.groups_per_split.values())
                if dataset.groups_per_split is not None
                else None
            ),
        }

    return {
        "max_windows": max_windows,
        "splits": splits_report,
        "distribution_shift": distribution_shift,
        "duplicate_windows": duplicate_windows,
        "groups": groups_report,
        "warnings": warnings,
        "violations": violations,
    }


def enforce_strict_data_audit(
    report: Dict[str, object],
    context: str = "",
) -> bool:
    """
    严格数据审计：报告中存在致命问题（数据泄露/标签噪声）时
    打印明细并返回 False，否则返回 True。

    调用方据此拒绝开始训练或拒绝解封测试集：宁可停在这一步，
    也不在已泄露的数据上产出「看起来有效」的结论。
    """
    violations = report.get("violations") or []

    if not violations:
        return True

    where = f"（{context}）" if context else ""
    print(f"[data] 严格数据审计未通过{where}：")
    for violation in violations:
        print(f"[data]   严重: {violation}")

    return False


def print_data_audit(report: Dict[str, object]) -> None:
    """打印数据审计摘要与全部警告。"""
    splits = report.get("splits", {})
    parts = [
        f"{name}={info['bytes']}B"
        for name, info in splits.items()
    ]
    print("[data] 数据审计: " + ", ".join(parts))

    groups = report.get("groups")
    if isinstance(groups, dict):
        per_split = groups.get("groups_per_split") or {}
        print(
            "[data]   群组划分: 分隔符长度 "
            f"{groups.get('separator_length')}B, "
            "groups="
            + ", ".join(
                f"{name}:{count}"
                for name, count in per_split.items()
            )
        )

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

    violation_set = set(report.get("violations", []))

    for warning in report.get("warnings", []):
        # 致命问题在下面以「严重」单独打印，避免重复两遍。
        if warning not in violation_set:
            print(f"[data] 警告: {warning}")

    for violation in report.get("violations", []):
        print(f"[data] 严重: {violation}")


# ============================================================
# Class-imbalance-aware metrics
# ============================================================
#
# 指标口径说明（多分类、类别不平衡场景）：
# - 下一 token 预测是 256 类单标签多分类问题，字节分布天然
#   高度不平衡（空格/换行/常见字母 vs 罕见控制字节）；
# - token accuracy 单独看会被高频字节抬高，必须与
#   majority_accuracy（常数预测器参考）对照解读，并用
#   macro / weighted F1 补充逐类视角；
# - 单标签多分类下 micro-F1 数值恒等于 token accuracy，
#   两者都保留是为了让键名明确交代「micro」口径；
# - macro F1 只对评估 split 中出现过的类（support > 0）
#   求平均，避免给从未出现的字节类记 0 分拉低均值；
# - one-vs-rest 的 macro PR-AUC / ROC-AUC 只统计 support
#   不小于 _AUC_MIN_SUPPORT 的类。极度不平衡时优先看
#   PR-AUC：ROC-AUC 会因负样本基数巨大、FPR 增长缓慢而
#   给出过于乐观的分数。

# macro AUC 只纳入 support 达到该阈值的字节类。
_AUC_MIN_SUPPORT = 20

# AUC 计算最多保留的评估位置数（positions × 256 类的
# 概率矩阵会随该上限线性增长；默认验证窗口约 8 万位置，
# 不会触顶）。超出时按窗口顺序截断并在结果中注明。
_AUC_MAX_TOKENS = 100_000


def confusion_matrix_metrics(
    confusion: torch.Tensor,
) -> Dict[str, float]:
    """
    从 [n_classes, n_classes] 混淆矩阵（行 = 目标类，
    列 = 预测类）计算 token accuracy 与 micro / macro /
    weighted F1，以及多数类参考 accuracy。

    返回键全部不带 split 前缀，由调用方加上
    （val_ / test_），并保证口径在文档中显式说明：
    - ``token_accuracy`` / ``micro_f1``：micro 口径
      （单标签多分类下两者恒等）；
    - ``macro_f1``：对 support > 0 的类求算术平均；
    - ``weighted_f1``：按各类 support 加权平均；
    - ``majority_accuracy``：把所有位置都预测成评估集
      中最频繁字节时的 accuracy，是类别不平衡下
      解读 accuracy 的下界参考。
    """
    confusion = confusion.double()
    total = confusion.sum()

    if float(total) <= 0.0:
        return {
            "token_accuracy": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "majority_accuracy": float("nan"),
            "classes_present": 0.0,
        }

    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positive = confusion.diagonal()

    precision = true_positive / predicted.clamp(min=1.0)
    recall = true_positive / support.clamp(min=1.0)

    denominator = precision + recall
    f1 = torch.zeros_like(true_positive)
    nonzero = denominator > 0.0
    f1[nonzero] = (
        2.0
        * precision[nonzero]
        * recall[nonzero]
        / denominator[nonzero]
    )

    present = support > 0.0

    return {
        "token_accuracy": float(
            true_positive.sum() / total
        ),
        "micro_f1": float(true_positive.sum() / total),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float(
            (f1 * support).sum() / total
        ),
        "majority_accuracy": float(
            support.max() / total
        ),
        "classes_present": float(present.sum()),
    }


def _binary_roc_auc(
    sorted_scores: torch.Tensor,
    order: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[float]:
    """
    精确处理并列分数的 Mann-Whitney ROC-AUC。

    sorted_scores / order 是 scores 的升序排序结果及其索引，
    由调用方统一计算并在 ROC-AUC 与 PR-AUC 之间复用；
    labels 为 0/1 的目标向量；正类或负类为空时返回
    None（该类不计入 macro 平均）。
    """
    n = sorted_scores.numel()
    n_pos = int(labels.sum())

    if n_pos == 0 or n_pos == n:
        return None

    n_neg = n - n_pos

    # 并列分数取平均秩，保证 AUC 与并列组内的随机排序
    # 期望一致。
    change = torch.ones(n, dtype=torch.bool)
    change[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group = torch.cumsum(change, 0) - 1

    starts = torch.nonzero(change).squeeze(1)
    ends = torch.cat(
        [
            starts[1:],
            torch.tensor([n], dtype=starts.dtype),
        ]
    )

    # 排序后第 i 个位置（0 基）的 1 基秩为 i+1；同组并列
    # 分数共享平均秩 (start + end) / 2（1 基、右开）。
    average_rank = (
        starts[group] + 1 + ends[group]
    ).double() / 2.0

    positive_ranks = average_rank[
        labels[order].bool()
    ]

    auc = (
        positive_ranks.sum()
        - n_pos * (n_pos + 1) / 2.0
    ) / float(n_pos * n_neg)

    return float(auc)


def _binary_average_precision(
    sorted_scores: torch.Tensor,
    order: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[float]:
    """
    单类 one-vs-rest 的 average precision（PR-AUC）。

    sorted_scores / order 是 scores 的升序排序结果及其索引，
    由调用方统一计算并在 ROC-AUC 与 PR-AUC 之间复用；按
    相异分数阈值分步累计（与 sklearn average_precision_score
    的阶梯式口径一致）；正类为空时返回 None。
    """
    n_pos = int(labels.sum())

    if n_pos == 0:
        return None

    # 升序排序翻转即降序；并列组仍然连续，且阈值组的
    # 枚举顺序（分数从高到低）与组内顺序无关，结果不变。
    order = order.flip(0)
    sorted_scores = sorted_scores.flip(0)
    sorted_labels = labels[order].double()

    n = labels.numel()
    positions = torch.arange(
        1,
        n + 1,
        dtype=torch.float64,
    )

    # 每个相异分数阈值组结束位置的累计真正例数。
    hits = torch.cumsum(sorted_labels, 0)

    # 组边界：分数变化处的下一个位置是该组的结束位置
    # （1 基）；最后一组以 n 结束。
    boundary = torch.ones(n, dtype=torch.bool)
    boundary[:-1] = (
        sorted_scores[1:] != sorted_scores[:-1]
    )
    boundary_indices = torch.nonzero(boundary).squeeze(1)

    tp = hits[boundary_indices]
    ranks = positions[boundary_indices]

    precision = tp / ranks
    recall = tp / float(n_pos)

    # AP = Σ (R_i − R_{i−1}) × Precision_i。
    recall_gain = torch.cat(
        [
            recall[:1],
            recall[1:] - recall[:-1],
        ]
    )

    return float((recall_gain * precision).sum())


def macro_auc_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    min_support: int = _AUC_MIN_SUPPORT,
) -> Dict[str, object]:
    """
    one-vs-rest 的 macro PR-AUC 与 ROC-AUC。

    probabilities: [n_tokens, vocab] 的预测分布；
    targets: [n_tokens] 的目标类。只统计 support 达到
    min_support 的类，返回值同时记录纳入类数，保证
    「macro」的口径（对哪些类求平均）可追溯。

    极度不平衡时优先报告 PR-AUC：负样本基数巨大时
    ROC-AUC 的 FPR 分母被拉大，同样的假阳性绝对数量下
    分数增长缓慢，容易给出过于乐观的结论。
    """
    probabilities = probabilities.detach().cpu().float()
    targets = targets.detach().cpu()

    support = torch.bincount(
        targets,
        minlength=vocab_size,
    )
    eligible = torch.nonzero(
        support >= min_support
    ).squeeze(1)

    pr_aucs: List[float] = []
    roc_aucs: List[float] = []

    for class_index in eligible.tolist():
        labels = (targets == class_index).double()
        scores = probabilities[:, class_index]

        # ROC-AUC 与 PR-AUC 复用同一次排序结果。
        sorted_scores, order = torch.sort(scores)

        pr = _binary_average_precision(
            sorted_scores,
            order,
            labels,
        )
        if pr is not None:
            pr_aucs.append(pr)

        roc = _binary_roc_auc(
            sorted_scores,
            order,
            labels,
        )
        if roc is not None:
            roc_aucs.append(roc)

    return {
        "pr_auc_macro": (
            sum(pr_aucs) / len(pr_aucs)
            if pr_aucs
            else None
        ),
        "roc_auc_macro": (
            sum(roc_aucs) / len(roc_aucs)
            if roc_aucs
            else None
        ),
        "auc_n_classes": len(roc_aucs),
        "auc_min_support": min_support,
        "auc_tokens_evaluated": int(targets.numel()),
    }


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
    compute_auc: bool = False,
) -> Dict[str, float]:
    """
    在指定评估 split 上计算加权平均语言模型损失与
    类别不平衡感知的分类指标。

    split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    test 只在最终评估时调用一次；返回指标键带 split 前缀
    （val_loss/test_loss 等）。

    每次评估都基于同一批固定窗口累计混淆矩阵，并报告：

    - ``{split}_loss`` / ``{split}_ppl`` /
      ``{split}_bits_per_byte``：主指标（与 LM 研究目标
      对齐的似然口径）；
    - ``{split}_token_accuracy``（= micro 口径）与
      ``{split}_micro_f1``：单标签多分类下两者恒等；
    - ``{split}_macro_f1`` / ``{split}_weighted_f1``：
      必须注明平均口径，macro 只对评估集中出现过的类
      求平均；
    - ``{split}_majority_accuracy``：常数预测器（永远
      猜最高频字节）的 accuracy，类别不平衡下解读
      accuracy 的参考下界。

    compute_auc=True 时（最终/测试评估）额外计算
    one-vs-rest 的 ``{split}_pr_auc_macro`` 与
    ``{split}_roc_auc_macro``，只纳入 support 达到
    _AUC_MIN_SUPPORT 的类；极度不平衡时优先看 PR-AUC。
    """
    was_training = model.training
    model.eval()

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    # split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    # test 只在最终评估时调用一次。

    vocab_size = _unwrap_model(model).cfg.vocab_size

    # 按目标 token 数加权累加。
    total = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    evaluated_tokens = 0

    # 混淆矩阵 [目标类, 预测类]，在评估设备上以 int64 累积
    #（逐批只做整数加法），收尾时一次性转 fp64。
    confusion = torch.zeros(
        (vocab_size, vocab_size),
        device=device,
        dtype=torch.int64,
    )

    # AUC 需要全部位置的预测概率；只在最终/测试评估时收集，
    # 并按 _AUC_MAX_TOKENS 封顶。
    collected_probs: List[torch.Tensor] = []
    collected_targets: List[torch.Tensor] = []
    auc_tokens = 0

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

                # argmax 直接取模型输出 dtype（bf16/fp16 -> fp32
                # 是精确升位，不改变数值与并列顺序），避免整批
                # logits 的 fp32 临时副本；AUC 的 softmax 仍在
                # fp32 下统计，避免半精度数值噪声影响逐类指标。
                raw_logits = result.pop("logits")

                predicted = raw_logits.argmax(dim=-1)
                flat_target = y.reshape(-1).to(
                    torch.int64
                )
                flat_predicted = predicted.reshape(-1)

                confusion += torch.bincount(
                    flat_target * vocab_size
                    + flat_predicted,
                    minlength=vocab_size ** 2,
                ).view(
                    vocab_size,
                    vocab_size,
                )

                if compute_auc and (
                    auc_tokens < _AUC_MAX_TOKENS
                ):
                    # 最后一批可能超出封顶：按剩余上限截断，
                    # 保证收集的 token 数严格不超过上限。
                    remaining = (
                        _AUC_MAX_TOKENS - auc_tokens
                    )
                    probs = F.softmax(
                        raw_logits.float(),
                        dim=-1,
                    ).reshape(
                        -1,
                        vocab_size,
                    )[:remaining].cpu()
                    collected_probs.append(probs)
                    collected_targets.append(
                        flat_target[:remaining].cpu()
                    )
                    auc_tokens += probs.size(0)

                # cross_entropy 返回批均值，乘回目标 token 数。
                # add_(alpha=) 完成「乘 token 数 + 累加」。
                total.add_(
                    result["lm_loss"].float(),
                    alpha=float(y.numel()),
                )
                evaluated_tokens += y.numel()
                del result, raw_logits, x, y
    finally:
        # 无论评估是否中途出错，都恢复原来的训练模式。
        model.train(was_training)

    # ByteDataset 保证至少有一个合法评估起点。
    loss = (total / evaluated_tokens).item()

    metrics = {
        f"{split}_loss": loss,
        f"{split}_ppl": math.exp(min(loss, 80.0)),
        f"{split}_bits_per_byte": loss / math.log(2.0),
    }

    class_metrics = confusion_matrix_metrics(confusion.double())

    for key, value in class_metrics.items():
        metrics[f"{split}_{key}"] = value

    if compute_auc:
        if collected_probs:
            auc_metrics = macro_auc_metrics(
                torch.cat(collected_probs),
                torch.cat(collected_targets),
                vocab_size,
            )
            for key, value in auc_metrics.items():
                metrics[f"{split}_{key}"] = value
        else:
            # 空评估（理论上不可达）：显式记录而不是
            # 静默缺键。
            metrics[f"{split}_pr_auc_macro"] = None
            metrics[f"{split}_roc_auc_macro"] = None
            metrics[f"{split}_auc_n_classes"] = 0
            metrics[f"{split}_auc_min_support"] = (
                _AUC_MIN_SUPPORT
            )
            metrics[f"{split}_auc_tokens_evaluated"] = 0

    return metrics


# ============================================================
# Trivial reference baselines (uniform / unigram)
# ============================================================

@torch.inference_mode()
def compute_reference_baselines(
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    split: str = "val",
) -> Dict[str, object]:
    """
    在与模型评估完全相同的固定窗口上计算两个「稻草人之下」
    的参考基线，作为解读指标的坐标系：

    - uniform：每个字节等概率 1/256，NLL = ln(256)，是
      任何模型必须显著低于的绝对下限；
    - unigram：只用 train split 的字节频率做无上下文预测
      （Laplace 平滑，统计量只在 train 上 fit），对应
      「类别不平衡下常数/多数类预测器」的视角——它的
      token accuracy 已经有相当份额，单看 accuracy 无法
      判断模型是否真的学到了上下文信息。

    返回 uniform / unigram 在指定 split 上的 loss、
    accuracy、micro/macro/weighted F1 与多数类参考
    accuracy，随 config.json / metrics.jsonl 存档。
    训练曲线诊断把「是否学到东西」的欠拟合参照从
    uniform 换成更强的 unigram，避免只战胜稻草人。
    """
    # 字节级管线的词表固定为 256。
    vocab_size = 256

    # unigram 统计量只在 train 上 fit。
    train_counts = torch.bincount(
        dataset.train.to(torch.int64),
        minlength=vocab_size,
    ).double()
    train_total = float(train_counts.sum())

    unigram_probs = (
        train_counts + 1.0
    ) / (train_total + vocab_size)
    uniform_probs = torch.full(
        (vocab_size,),
        1.0 / vocab_size,
        dtype=torch.float64,
    )

    unigram_prediction = int(train_counts.argmax())

    uniform_total = 0.0
    unigram_total = 0.0
    evaluated_tokens = 0

    confusion = torch.zeros(
        (vocab_size, vocab_size),
        dtype=torch.float64,
    )

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

        flat_target = y.reshape(-1).to(torch.int64)

        uniform_total += float(
            -uniform_probs[flat_target].log().sum()
        )
        unigram_total += float(
            -unigram_probs[flat_target].log().sum()
        )
        evaluated_tokens += flat_target.numel()

        # unigram 预测器对所有位置输出同一个字节。
        predicted = torch.full_like(
            flat_target,
            unigram_prediction,
        )
        confusion += torch.bincount(
            flat_target * vocab_size + predicted,
            minlength=vocab_size ** 2,
        ).view(vocab_size, vocab_size).double()

    loss_uniform = uniform_total / evaluated_tokens
    loss_unigram = unigram_total / evaluated_tokens

    class_metrics = confusion_matrix_metrics(confusion)

    return {
        "split": split,
        "tokens_evaluated": evaluated_tokens,
        "uniform_loss": loss_uniform,
        "unigram_loss": loss_unigram,
        "unigram_train_bytes": int(train_total),
        "majority_byte": unigram_prediction,
        "unigram_token_accuracy": class_metrics[
            "token_accuracy"
        ],
        "unigram_majority_accuracy": class_metrics[
            "majority_accuracy"
        ],
        "unigram_macro_f1": class_metrics["macro_f1"],
        "unigram_weighted_f1": class_metrics[
            "weighted_f1"
        ],
    }


# ============================================================
# N-gram statistical baseline (最简基线，先于任何神经模型)
# ============================================================
#
# 方法论要求：复杂模型必须显著战胜「精心调优过的简单基线」
# 才有说服力。本节实现字节级 n-gram 统计基线：
#
# - 计数只在 train split 上做（无泄露），评估用与神经模型
#   完全相同的固定窗口协议；
# - 插值平滑 P_o(c|ctx) =
#       (count(ctx,c) + α·P_{o-1}(c|ctx')) / (count(ctx) + α)，
#   一路回退到 uniform；阶数 o 与平滑系数 α 在 val 上
#   网格调优（简单基线同样要充分调优，而不是随便跑一档）；
# - test 只在显式 --evaluate-test 时用 val 上选出的
#   (o, α) 评估一次，绝不在 test 上调参。

# 上下文 + 目标按字节精确打包进 int64，7 个字节（56 bit）
# 是不越界的上限；更大的阶数需要哈希，超出即报错。
_NGRAM_MAX_ORDER = 7

# 每阶计数使用的最大训练窗口数；超过时按等距步长抽样
#（抽样是确定性的）。
_NGRAM_COUNT_CAP = 4_000_000


class NgramBaseline:
    """
    字节级插值 n-gram 基线（含 unigram）。

    每个阶 o 存三张有序表（torch.unique 保证升序）：
    - pair_keys / pair_counts：(ctx, c) 对的键
      （ctx*256 + c）与计数；
    - ctx_uniq / ctx_counts：ctx 键与出现次数。

    键按升序排列使得同一 ctx 的所有 (ctx, c) 对在
    pair_keys 中连续，查询用 searchsorted 完成。
    """

    def __init__(
        self,
        train_bytes: torch.Tensor,
        max_order: int,
        count_cap: int = _NGRAM_COUNT_CAP,
    ):
        if not 1 <= max_order <= _NGRAM_MAX_ORDER:
            raise ValueError(
                "n-gram 阶数必须落在 "
                f"[1, {_NGRAM_MAX_ORDER}]（字节精确打包"
                "的 int64 上限）"
            )

        self.max_order = max_order
        self.vocab_size = 256

        data = train_bytes
        n_train = data.numel()

        self.levels: Dict[int, Dict[str, torch.Tensor]] = {}

        for order in range(1, max_order + 1):
            if n_train < order:
                # 语料比阶数还短：该阶没有可计数的窗口，
                # 查询时自然回退到低阶。
                self.levels[order] = {
                    "pair_keys": torch.empty(
                        0,
                        dtype=torch.int64,
                    ),
                    "pair_counts": torch.empty(
                        0,
                        dtype=torch.int64,
                    ),
                    "ctx_uniq": torch.empty(
                        0,
                        dtype=torch.int64,
                    ),
                    "ctx_counts": torch.empty(
                        0,
                        dtype=torch.int64,
                    ),
                }
                continue

            num_windows = n_train - order + 1
            stride = max(
                1,
                math.ceil(num_windows / count_cap),
            )

            # data.unfold(0, order, 1) 是 [num_windows, order]
            # 的滑动窗口视图；按步长取行仍是视图，打包时
            # 才产生 int64 张量。
            windows = data.unfold(0, order, 1)[::stride]

            keys = windows[:, 0].to(torch.int64)
            for column in range(1, order):
                keys = keys * 256 + windows[
                    :, column
                ].to(torch.int64)

            pair_keys, pair_counts = torch.unique(
                keys,
                return_counts=True,
            )

            ctx_keys = pair_keys // 256

            ctx_uniq, inverse = torch.unique(
                ctx_keys,
                return_inverse=True,
            )
            ctx_counts = torch.zeros(
                ctx_uniq.numel(),
                dtype=torch.int64,
            ).scatter_add_(
                0,
                inverse,
                pair_counts,
            )

            self.levels[order] = {
                "pair_keys": pair_keys,
                "pair_counts": pair_counts,
                "ctx_uniq": ctx_uniq,
                "ctx_counts": ctx_counts,
            }

        # unigram 总计数（level 1 的 ctx 计数），用于
        # 参考/归一。
        first = self.levels[1]
        if first["ctx_uniq"].numel():
            self.train_total = int(
                first["ctx_counts"][0]
            )
        else:
            self.train_total = 0

    def _pack_context(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        [n, ctx_len] 字节矩阵打包成 [n] 的 int64 键。
        """
        keys = context[:, 0].to(torch.int64)
        for column in range(1, context.size(1)):
            keys = keys * 256 + context[
                :, column
            ].to(torch.int64)
        return keys

    def _valid_contexts(
        self,
        windows: torch.Tensor,
        order: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """
        返回给定阶数下每个有效位置的 (ctx 键, 扁平位置)。

        目标 y[t] = w[t+1]；长度 order-1 的上下文需要
        t+1-(order-1) >= 0，即 t >= order-2。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        if order == 1:
            return (
                torch.zeros(
                    n_positions,
                    dtype=torch.int64,
                ),
                torch.arange(n_positions),
                n_positions,
            )

        context_len = order - 1

        if context_len > seq_len:
            # 阶数高于窗口可提供的最大上下文长度：没有任何
            # 有效位置，调用方将整阶回退到低阶模型。此处直接
            # 返回空结果，避免 unfold 的 size 越界报错。
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty, 0

        t_min = context_len - 1
        n_valid = seq_len - t_min

        # windows.unfold(1, ctx_len, 1) 的第 j 列是
        # w[:, j : j+ctx_len]；目标 t 对应 j = t+1-ctx_len，
        # j 取值范围 [0, seq_len - ctx_len] = [0, n_valid-1]。
        contexts = windows.unfold(
            1,
            context_len,
            1,
        )[:, :n_valid].reshape(-1, context_len)

        batch_index = torch.arange(
            batch,
        ).repeat_interleave(n_valid)
        offsets = (
            torch.arange(n_valid) + t_min
        ).repeat(batch)

        positions = batch_index * seq_len + offsets

        return (
            self._pack_context(contexts),
            positions,
            n_valid,
        )

    @staticmethod
    def _lookup(
        sorted_keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        在有序键值表中查询，未命中返回 0。
        """
        if sorted_keys.numel() == 0:
            return torch.zeros_like(queries)

        locations = torch.searchsorted(
            sorted_keys,
            queries,
        )
        locations = locations.clamp(
            max=sorted_keys.numel() - 1,
        )
        found = sorted_keys[locations] == queries
        return torch.where(
            found,
            values[locations],
            torch.zeros_like(values[locations]),
        )

    def target_log_probs_per_level(
        self,
        windows: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        对一批窗口 [batch, seq_len + 1] 计算每个阶数
        （1..max_order）下目标字节的 log 概率。

        返回 [max_order, batch*seq_len]：第 o-1 行是
        「最高可用阶不超过 o」的回退模型在每个位置目标
        字节的 log 概率——上下文不足或从未见过的位置
        自动回退到低阶，供网格调优一次性拿到所有候选
        阶数的损失。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        targets = windows[
            :, 1:
        ].reshape(-1).to(torch.int64)

        log_probs = torch.zeros(
            (self.max_order, n_positions),
            dtype=torch.float64,
        )

        # level 0 的目标概率：uniform。
        probability = torch.full(
            (n_positions,),
            1.0 / self.vocab_size,
            dtype=torch.float64,
        )

        for order in range(1, self.max_order + 1):
            level = self.levels[order]

            ctx_keys, positions, _ = (
                self._valid_contexts(windows, order)
            )

            if positions.numel() == 0:
                # 窗口比阶还短：整阶保持低阶回退值。
                if order > 1:
                    log_probs[order - 1] = log_probs[
                        order - 2
                    ]
                continue

            n_ctx = self._lookup(
                level["ctx_uniq"],
                level["ctx_counts"],
                ctx_keys,
            ).double()

            pair_keys = ctx_keys * 256 + targets[
                positions
            ]
            count_pair = self._lookup(
                level["pair_keys"],
                level["pair_counts"],
                pair_keys,
            ).double()

            interpolated = (
                count_pair
                + alpha * probability[positions]
            ) / (n_ctx + alpha)

            row = log_probs[order - 1]
            row[positions] = interpolated.log()

            if order > 1:
                # 上下文不足的位置保持低阶值。
                mask = torch.ones(
                    n_positions,
                    dtype=torch.bool,
                )
                mask[positions] = False
                row[mask] = log_probs[order - 2][mask]

            probability = probability.clone()
            probability[positions] = interpolated

        return log_probs

    def position_distributions(
        self,
        windows: torch.Tensor,
        alpha: float,
        order: int,
    ) -> torch.Tensor:
        """
        返回「最高可用阶不超过 order」的回退模型在每个
        位置的完整 [batch*seq_len, vocab] 概率分布，
        用于最终评估的 argmax / F1 / AUC 指标。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        probabilities = torch.full(
            (n_positions, self.vocab_size),
            1.0 / self.vocab_size,
            dtype=torch.float64,
        )

        for current in range(1, order + 1):
            level = self.levels[current]

            ctx_keys, positions, _ = (
                self._valid_contexts(
                    windows,
                    current,
                )
            )

            if positions.numel() == 0:
                continue

            n_ctx = self._lookup(
                level["ctx_uniq"],
                level["ctx_counts"],
                ctx_keys,
            ).double()

            if n_ctx.numel() and (
                level["pair_keys"].numel() == 0
            ):
                # 该阶没有任何计数：分布保持低阶回退。
                continue

            denominator = n_ctx + alpha

            updated = probabilities[
                positions
            ] * (alpha / denominator.unsqueeze(1))

            # 同一 ctx 的 (ctx, c) 对在升序 pair_keys 中
            # 连续：散射该 ctx 的全部非零计数行。
            starts = torch.searchsorted(
                level["pair_keys"],
                ctx_keys * 256,
            )
            ends = torch.searchsorted(
                level["pair_keys"],
                (ctx_keys + 1) * 256,
            )
            lengths = (ends - starts).clamp(min=0)

            total_nnz = int(lengths.sum())

            if total_nnz > 0:
                row_of = torch.repeat_interleave(
                    torch.arange(
                        positions.numel(),
                        dtype=torch.int64,
                    ),
                    lengths,
                )
                offsets = torch.cumsum(lengths, 0) - (
                    lengths
                )
                intra = torch.arange(total_nnz) - (
                    offsets[row_of]
                )
                flat_index = starts[row_of] + intra

                classes = level["pair_keys"][
                    flat_index
                ] % 256
                counts = level["pair_counts"][
                    flat_index
                ].double()

                updated.index_put_(
                    (row_of, classes),
                    counts
                    / denominator[row_of],
                    accumulate=True,
                )

            probabilities[positions] = updated

        return probabilities


def _ngram_eval_batches(
    dataset: ByteDataset,
    split: str,
    batch_size: int,
    batches: int,
) -> List[torch.Tensor]:
    """
    收集指定评估 split 的全部固定窗口（CPU uint8，
    [batch, seq_len + 1]），供 n-gram 评估复用。
    """
    cpu = torch.device("cpu")
    windows: List[torch.Tensor] = []

    for batch_index in range(batches):
        x, y = dataset.fixed_eval_batch(
            split,
            batch_index,
            batches,
            batch_size,
            cpu,
        )

        if x.size(0) == 0:
            break

        windows.append(
            torch.cat(
                [
                    x[:, :1].to(torch.uint8),
                    y,
                ],
                dim=1,
            )
        )

    return windows


def _ngram_extended_metrics(
    model: NgramBaseline,
    windows: List[torch.Tensor],
    alpha: float,
    order: int,
    split: str,
) -> Dict[str, object]:
    """
    用调优好的 (order, alpha) 在固定窗口上计算与神经模型
    evaluate() 完全同口径的指标：loss / ppl / bits_per_byte、
    token accuracy（micro）、macro / weighted F1、多数类
    参考，以及 one-vs-rest 的 macro PR-AUC / ROC-AUC。
    """
    vocab_size = model.vocab_size

    total_nll = 0.0
    evaluated_tokens = 0

    confusion = torch.zeros(
        (vocab_size, vocab_size),
        dtype=torch.float64,
    )

    collected_probs: List[torch.Tensor] = []
    collected_targets: List[torch.Tensor] = []
    auc_tokens = 0

    for window in windows:
        distributions = model.position_distributions(
            window,
            alpha,
            order,
        )

        targets = window[:, 1:].reshape(-1).to(
            torch.int64
        )
        probabilities = distributions

        target_probs = probabilities[
            torch.arange(targets.numel()),
            targets,
        ]

        total_nll += float(
            -target_probs.clamp_min(1e-300).log().sum()
        )
        evaluated_tokens += targets.numel()

        predicted = probabilities.argmax(dim=1)

        confusion += torch.bincount(
            targets * vocab_size + predicted,
            minlength=vocab_size ** 2,
        ).view(vocab_size, vocab_size).double()

        if auc_tokens < _AUC_MAX_TOKENS:
            # 最后一个窗口可能超出封顶：按剩余上限截断，
            # 保证收集的 token 数严格不超过上限。
            remaining = _AUC_MAX_TOKENS - auc_tokens
            collected_probs.append(
                probabilities[:remaining].to(torch.float32)
            )
            collected_targets.append(targets[:remaining])
            auc_tokens += min(targets.numel(), remaining)

    loss = total_nll / evaluated_tokens

    metrics: Dict[str, object] = {
        f"{split}_loss": loss,
        f"{split}_ppl": math.exp(min(loss, 80.0)),
        f"{split}_bits_per_byte": loss / math.log(2.0),
    }

    class_metrics = confusion_matrix_metrics(confusion)

    for key, value in class_metrics.items():
        metrics[f"{split}_{key}"] = value

    if collected_probs:
        auc_metrics = macro_auc_metrics(
            torch.cat(collected_probs),
            torch.cat(collected_targets),
            vocab_size,
        )
        for key, value in auc_metrics.items():
            metrics[f"{split}_{key}"] = value

    return metrics


def run_statistical_baseline(
    args: argparse.Namespace,
    dataset: ByteDataset,
    out_dir: Path,
    device: torch.device,
    data_audit: Optional[Dict[str, object]],
) -> None:
    """
    统计基线入口：计数（train）→ val 网格调优
    （阶数 × 平滑系数）→ 选出最优配置 → 与神经模型同口径
    的 val（及一次性 test）指标，写入与训练运行相同结构的
    config.json / metrics.jsonl，可直接进入 --compare。
    """
    del device  # n-gram 全部在 CPU 上完成。

    alpha_grid: List[float] = list(args.ngram_alpha_grid)

    print(
        "[ngram] 在 train split 上计数 "
        f"({dataset.train.numel():,} 字节, "
        f"max_order={args.ngram_max_order}, "
        f"count_cap={_NGRAM_COUNT_CAP:,})"
    )

    model = NgramBaseline(
        dataset.train,
        max_order=args.ngram_max_order,
    )

    val_windows = _ngram_eval_batches(
        dataset,
        "val",
        args.batch_size,
        args.eval_batches,
    )

    if not val_windows:
        raise SystemExit(
            "验证集固定窗口为空，无法调优统计基线"
        )

    # ---- 参考基线（uniform / unigram），与训练运行同款 ----
    reference = compute_reference_baselines(
        dataset,
        args.batch_size,
        args.eval_batches,
        torch.device("cpu"),
        split="val",
    )

    print(
        "[ngram] 参考基线: uniform_loss="
        f"{reference['uniform_loss']:.4f}, "
        "unigram_loss="
        f"{reference['unigram_loss']:.4f}"
    )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = open(
        metrics_path,
        "w",
        encoding="utf-8",
    )

    def _write(record: Dict[str, object]) -> None:
        metrics_file.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()

    _write({
        "step": "reference_baselines",
        **reference,
    })

    # ---- 网格调优：α 每档一遍窗口，同时拿到所有阶数损失 ----
    candidates: List[Dict[str, object]] = []

    for alpha in alpha_grid:
        totals = torch.zeros(model.max_order)
        tokens = 0

        for window in val_windows:
            log_probs = (
                model.target_log_probs_per_level(
                    window,
                    alpha,
                )
            )
            totals += -log_probs.sum(dim=1)
            tokens += log_probs.size(1)

        for order in range(1, model.max_order + 1):
            val_loss = float(
                totals[order - 1] / tokens
            )
            record = {
                "step": "baseline_sweep",
                "order": order,
                "alpha": alpha,
                "val_loss": val_loss,
                "val_ppl": math.exp(
                    min(val_loss, 80.0)
                ),
                "val_bits_per_byte": val_loss
                / math.log(2.0),
                "tokens_evaluated": tokens,
            }
            _write(record)
            candidates.append(record)

        print(
            "[ngram] alpha="
            f"{alpha:g}: "
            + ", ".join(
                f"order {row['order']}="
                f"{row['val_loss']:.4f}"
                for row in candidates[
                    -model.max_order:
                ]
            )
        )

    # val 上选最优；并列时偏向更小的阶数与更小的 alpha
    # （简单优先，避免在验证集上过拟合出复杂配置）。
    best = min(
        candidates,
        key=lambda row: (
            row["val_loss"],
            row["order"],
            row["alpha"],
        ),
    )

    best_order = int(best["order"])
    best_alpha = float(best["alpha"])

    print(
        "[ngram] val 调优结果: order="
        f"{best_order}, alpha={best_alpha:g}, "
        f"val_loss={best['val_loss']:.4f}"
    )

    # ---- 用调优好的配置算同口径完整指标 ----
    final_metrics = _ngram_extended_metrics(
        model,
        val_windows,
        best_alpha,
        best_order,
        split="val",
    )

    final_record = {
        "step": "final",
        "model": "ngram",
        "ngram_order": best_order,
        "ngram_alpha": best_alpha,
        "ngram_max_order": args.ngram_max_order,
        **final_metrics,
    }

    _write(final_record)
    print(
        json.dumps(
            _json_safe({"final": final_record}),
            ensure_ascii=False,
        )
    )

    # ---- 一次性 test 评估（沿用训练运行同一封存协议）----
    # 注意：本入口每次运行都会新建输出目录语义（metrics.jsonl
    # 截断重写），test 只在显式 --evaluate-test 时评估一次，
    # 且 (order, alpha) 完全沿用 val 上调优的结果，绝不在
    # test 上二次调参。
    if args.evaluate_test:
        if not dataset.has_test:
            raise SystemExit(
                "--evaluate-test 需要有效的测试集："
                "请设置正的 --test-fraction 且语料足够长"
            )

        test_audit = audit_data_splits(
            dataset,
            include_test=True,
        )
        print_data_audit(test_audit)

        if args.strict_data_audit and not enforce_strict_data_audit(
            test_audit,
            "test",
        ):
            # 统计基线同样不能在泄露/损坏的测试集上报告指标；
            # 不写 test 记录，名额未消耗。
            print(
                "[test] 严格数据审计在测试集上发现泄露/损坏"
                "问题：拒绝解封测试集（不写入 test 记录）。"
                "请先清洗数据后重跑。"
            )
            test_windows = None
        else:
            test_windows = _ngram_eval_batches(
                dataset,
                "test",
                args.batch_size,
                args.eval_batches,
            )

        if test_windows is not None:
            test_metrics = _ngram_extended_metrics(
                model,
                test_windows,
                best_alpha,
                best_order,
                split="test",
            )

            _write({
                "step": "test",
                "model": "ngram",
                "ngram_order": best_order,
                "ngram_alpha": best_alpha,
                **test_metrics,
                "data_audit": test_audit,
            })

            print(
                json.dumps(
                    _json_safe({"test": test_metrics}),
                    ensure_ascii=False,
                )
            )

    metrics_file.close()

    # ---- config.json：与训练运行相同结构，可直接 --compare ----
    payload = {
        "config": {
            "mode": "ngram",
            "vocab_size": 256,
            "seq_len": dataset.seq_len,
            "ngram_max_order": args.ngram_max_order,
            "ngram_alpha_grid": alpha_grid,
            "ngram_order": best_order,
            "ngram_alpha": best_alpha,
        },
        # 统计基线没有优化器与训练轮数；comparison_family 让
        # 受控变量检查只在神经网络运行之间进行。
        "comparison_protocol": {
            "comparison_family": "statistical",
            "optimizer": "none",
            "lr": None,
            "lr_schedule": "none",
            "warmup_steps": None,
            "total_steps": None,
            "weight_decay": None,
            "grad_clip": None,
            "batch_size": args.batch_size,
            "data_augmentation": "none",
            "seq_len": dataset.seq_len,
            "val_fraction": dataset.val_fraction,
            "test_fraction": dataset.test_fraction,
            "group_separator": getattr(args, "group_separator", None),
            "eval_interval": None,
            "eval_batches": args.eval_batches,
            "data_sha256": dataset.data_sha256,
        },
        "args": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "data_sha256",
                "data_bytes",
                "data_val_fraction",
                "data_test_fraction",
            }
        },
        "n_params": 0,
        "data": {
            "path": args.data,
            "sha256": dataset.data_sha256,
            "bytes": dataset.num_bytes,
            "val_fraction": dataset.val_fraction,
            "test_fraction": dataset.test_fraction,
            "group_separator": getattr(args, "group_separator", None),
        },
        "data_audit": data_audit,
        "reference_baselines": reference,
        "run_metadata": collect_run_metadata(
            torch.device("cpu")
        ),
        "sanity_check": None,
    }

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(payload),
            f,
            ensure_ascii=False,
            indent=2,
        )


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
        step = record.get("step")
        if not isinstance(step, int):
            continue
        if "val_loss" not in record:
            continue

        val_loss = record.get("val_loss")
        if val_loss is None:
            continue

        val_loss = float(val_loss)

        if (
            best_loss is None
            or val_loss < best_loss - min_delta
        ):
            best_loss = val_loss
            best_step = step
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
            "linear(mm/addmm) 实测 + attention(4·B·T²·D) 估计"
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
            _original=original_attention_step,
        ) -> torch.Tensor:
            if inputs.dim() == 3:
                # numel = B·T·D，乘 4·T 即 4·B·T²·D。
                attention_flops[0] += (
                    4 * int(inputs.size(-2)) * int(inputs.numel())
                )
            return _original(inputs)

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
            with torch.inference_mode(), autocast:
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
    metrics.jsonl（final/curve_analysis），输出统一口径的行。

    只汇总 val 口径指标：test 只在单个运行内解封评估一次，
    不进入跨运行对比（否则会诱发「挑 test 最好」的隐式
    测试集调参）。
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
            row["balance_coef"] = run_config.get("balance_coef")
            row["dense_fnn_hidden"] = run_config.get(
                "dense_fnn_hidden"
            )

        run_args = config_payload.get("args") or {}
        if isinstance(run_args, dict):
            row["seed"] = run_args.get("seed")

        # 受控变量协议：对比分析用它检查「除声明因子外，
        # 优化器 / 学习率 / 训练轮数 / 数据增强等设置是否一致」。
        row["protocol"] = experiment_protocol_from_payload(
            config_payload
        )

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
        curve_record = None

        for record in records:
            step = record.get("step")
            if step == "final":
                final_record = record
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
        row["val_bits_per_byte"] = (
            final_record.get("val_bits_per_byte")
            if final_record is not None
            else None
        )
        row["val_token_accuracy"] = (
            final_record.get("val_token_accuracy")
            if final_record is not None
            else None
        )
        row["val_macro_f1"] = (
            final_record.get("val_macro_f1")
            if final_record is not None
            else None
        )
        row["val_pr_auc_macro"] = (
            final_record.get("val_pr_auc_macro")
            if final_record is not None
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
    """
    以对齐表格打印多个运行的汇总指标。

    只展示 val 口径指标（尘封协议：test 结果只随单个运行的
    metrics.jsonl 报告，绝不并排对比）；除 val loss 外同时
    给出 token accuracy（micro 口径）与 macro F1：字节分布
    高度不平衡时 accuracy 会被高频字节抬高，必须两类指标
    一起看；完整口径（micro / macro / weighted、PR-AUC 等）
    见各运行 metrics.jsonl。
    """
    header = (
        f"{'run':<20} {'mode':<9} {'params':>9} "
        f"{'val_loss':>9} {'val_bpb':>8} {'acc':>6} "
        f"{'macroF1':>8} "
        f"{'lat(ms)':>9} {'GFLOPs':>8} {'mem(MB)':>9} "
        f"{'curve':<18}"
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
            f"{_value(row.get('val_bits_per_byte'), '.4f'):>8} "
            f"{_value(row.get('val_token_accuracy'), '.4f'):>6} "
            f"{_value(row.get('val_macro_f1'), '.4f'):>8} "
            f"{_value(row.get('latency_ms_per_forward'), '.4f'):>9} "
            f"{gflops:>8} "
            f"{_value(row.get('peak_memory_mb'), '.1f'):>9} "
            f"{str(row.get('curve_status')):<18}"
        )


# ============================================================
# Comparison analysis: controlled variables / seeds / statistics
# ============================================================

# 受控变量协议：对比 A 方法与 B 方法时，除声明为实验因子的
# 模块外，这些设置必须完全一致。每轮训练把这些值写进
# config.json 的 comparison_protocol 字段；--compare 逐项核对。
#
# comparison_family 把「神经网络运行」与「统计基线运行」
# （ngram，无优化器、无训练轮数）分开：受控变量检查只在
# 同一 family 内部进行。
COMPARISON_PROTOCOL_KEYS: Tuple[str, ...] = (
    "optimizer",
    "optimizer_betas",
    "optimizer_eps",
    "lr",
    "lr_schedule",
    "warmup_steps",
    "total_steps",
    "weight_decay",
    "grad_clip",
    "batch_size",
    "data_augmentation",
    "seq_len",
    "d_model",
    "n_heads",
    "n_layers",
    "n_experts",
    "expert_hidden",
    "top_k",
    "residual_scale",
    "val_fraction",
    "test_fraction",
    "group_separator",
    "eval_interval",
    "eval_batches",
    "data_sha256",
    # final/test 评估所用模型的选型口径：旧运行（无记录）是
    # 最后一步参数，新运行是验证集选中的最优验证点；两者混在
    # 一起对比不是同一口径，必须被检查发现。
    "model_selection",
)

# 允许声明为实验因子（跨运行允许不同）的键：行级字段 +
# 全部协议键。默认因子覆盖本项目的结构差异（模式、k1/k2、
# 分支宽度、balance 系数、种子）；学习率、训练步数、数据增强、
# 数据指纹等不在其中，因此默认就会被当作受控变量检查。
DEFAULT_COMPARISON_FACTORS = (
    "mode",
    "k1",
    "k2",
    "balance_coef",
    "seed",
    "dense_fnn_hidden",
)

# 聚合与显著性检验使用的指标。
#
# 尘封协议：这里只允许 val 口径的指标。测试集在每个运行中
# 只解封评估一次，其结果只能随该运行自身的 metrics.jsonl
# 报告；如果把 test 指标纳入跨运行对比/排序/显著性检验，
# 「在多个候选里挑 test 最好」会让测试集退化成隐式验证集
# （多重比较 + 选择偏差）。因此 --compare 系列工具一律
# 不读取、不聚合、不展示 test 指标。
COMPARISON_METRIC_KEYS: Tuple[str, ...] = (
    "val_loss",
    "val_ppl",
    "val_bits_per_byte",
    "val_token_accuracy",
    "val_macro_f1",
    "val_pr_auc_macro",
)

_LOWER_IS_BETTER_METRICS = {
    "val_loss",
    "val_ppl",
    "val_bits_per_byte",
}

# 因子的自然排序（立即合并对照在前，延迟合并在后），
# 用于让「基线→改进」的差值符号可解释。
_FACTOR_VALUE_ORDER = {
    "smelt": 0,
    "local_base": 1,
    "A": 2,
    "B": 3,
    "dense": 4,
    "ngram": 5,
}

# 显著性与统计功效的判定阈值。
_SIGNIFICANCE_LEVEL = 0.05
_MIN_SEEDS_FOR_STATS = 2
_RECOMMENDED_SEEDS = 3


def build_comparison_protocol(
    cfg: Config,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """
    汇总一条运行的全部「受控变量」，写入 config.json。

    优化器（AdamW）及其 betas/eps、学习率与调度、训练步数、
    weight decay、梯度裁剪、batch size、数据增强、数据指纹、
    模型结构与评估协议全部显式记录；--compare 时默认要求
    这些值在声明因子之外逐项一致，避免把「换了优化器或调了
    学习率」误当成方法本身的收益。
    """
    return {
        "comparison_family": "neural",
        # 优化器固定为 AdamW（betas/eps 使用 PyTorch 默认值），
        # 显式记录以便未来更换优化器时立即被对比检查发现。
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "lr": args.lr,
        "lr_schedule": "linear_warmup_cosine_to_10pct",
        "warmup_steps": args.warmup_steps,
        "total_steps": args.steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        # 当前字节级管线不做数据增强；显式记录空操作，
        # 未来加入增强后不同运行会立刻被对比检查发现。
        "data_augmentation": "none",
        "seq_len": cfg.seq_len,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "n_experts": cfg.n_experts,
        "expert_hidden": cfg.expert_hidden,
        "top_k": cfg.top_k,
        "residual_scale": cfg.residual_scale,
        "val_fraction": getattr(args, "val_fraction", None),
        "test_fraction": getattr(args, "test_fraction", None),
        # 群组分隔符决定切分位置（同一文件 + 不同分隔符 =
        # 不同的数据划分），必须作为受控变量核对。
        "group_separator": getattr(args, "group_separator", None),
        "eval_interval": getattr(args, "eval_interval", None),
        "eval_batches": getattr(args, "eval_batches", None),
        "data_sha256": getattr(args, "data_sha256", None),
        # 选型协议：final/test 都在验证集选中的模型上评估
        # （best.pt 不可用时回退到最后一步并明确提示）。
        "model_selection": "best_val_with_last_fallback",
    }


def experiment_protocol_from_payload(
    payload: Dict[str, object],
) -> Dict[str, object]:
    """
    从 config.json 读取受控变量协议；旧运行没有
    comparison_protocol 字段时按 args/config/data 尽力回退
    （能唯一确定的设置才回退，无法确定的留 None，由检查器
    标记为「缺记录」而不是静默视为一致）。
    """
    protocol = payload.get("comparison_protocol")
    if isinstance(protocol, dict):
        result = dict(protocol)
        result.setdefault("comparison_family", "neural")
        return result

    cfg = payload.get("config") or {}
    run_args = payload.get("args") or {}
    data = payload.get("data") or {}

    if not isinstance(cfg, dict):
        cfg = {}
    if not isinstance(run_args, dict):
        run_args = {}
    if not isinstance(data, dict):
        data = {}

    def _pick(*values):
        for value in values:
            if value is not None:
                return value
        return None

    has_training = run_args.get("lr") is not None

    return {
        "comparison_family": (
            "statistical"
            if cfg.get("mode") == "ngram"
            else "neural"
        ),
        # 代码历史上只有一种优化器与一种调度，能唯一确定时
        # 回退；否则留 None。
        "optimizer": "AdamW" if has_training else None,
        "optimizer_betas": (
            [0.9, 0.999] if has_training else None
        ),
        "optimizer_eps": 1e-8 if has_training else None,
        "lr_schedule": (
            "linear_warmup_cosine_to_10pct"
            if has_training
            else None
        ),
        "lr": run_args.get("lr"),
        "warmup_steps": run_args.get("warmup_steps"),
        "total_steps": run_args.get("steps"),
        "weight_decay": run_args.get("weight_decay"),
        "grad_clip": run_args.get("grad_clip"),
        "batch_size": run_args.get("batch_size"),
        "data_augmentation": "none" if has_training else None,
        "seq_len": cfg.get("seq_len"),
        "d_model": cfg.get("d_model"),
        "n_heads": cfg.get("n_heads"),
        "n_layers": cfg.get("n_layers"),
        "n_experts": cfg.get("n_experts"),
        "expert_hidden": cfg.get("expert_hidden"),
        "top_k": cfg.get("top_k"),
        "residual_scale": cfg.get("residual_scale"),
        "val_fraction": _pick(
            data.get("val_fraction"),
            run_args.get("val_fraction"),
        ),
        "test_fraction": _pick(
            data.get("test_fraction"),
            run_args.get("test_fraction"),
        ),
        "group_separator": _pick(
            data.get("group_separator"),
            run_args.get("group_separator"),
        ),
        "eval_interval": run_args.get("eval_interval"),
        "eval_batches": run_args.get("eval_batches"),
        "data_sha256": data.get("sha256"),
        # 旧运行没有选型记录，其 final/test 一律基于最后一步
        # 参数；显式回退为 "last_step"，与新版选型协议混用时
        # 会被受控变量检查标记出来。
        "model_selection": (
            "last_step"
            if run_args.get("lr") is not None
            else None
        ),
    }


def parse_factor_spec(spec: str) -> List[str]:
    """
    解析 --compare-factors 的逗号分隔列表；未知键显式报错，
    避免拼写错误静默关闭受控变量检查。
    """
    keys: List[str] = []
    for token in spec.split(","):
        key = token.strip()
        if key and key not in keys:
            keys.append(key)

    if not keys:
        raise ValueError("对比因子列表不能为空")

    candidates = set(COMPARISON_PROTOCOL_KEYS) | {
        "mode",
        "k1",
        "k2",
        "top_k",
        "balance_coef",
        "dense_fnn_hidden",
        "seed",
    }

    unknown = [key for key in keys if key not in candidates]
    if unknown:
        raise ValueError(
            "未知的对比因子: "
            + ", ".join(unknown)
            + "（可用: "
            + ", ".join(sorted(candidates))
            + "）"
        )

    return keys


def _stable_factor_key(value):
    """
    把因子值转换成可哈希、可比较的稳定表示。
    """
    if isinstance(value, (list, tuple)):
        return tuple(_stable_factor_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _stable_factor_key(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    return ("value", value)


def _factor_sort_rank(value):
    """
    因子值的排序键：数值 < 已知模式顺序 < 其他字符串。
    """
    if value is None:
        return (3, 0.0, "")
    if isinstance(value, bool):
        return (2, float(value), "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    if isinstance(value, str) and value in _FACTOR_VALUE_ORDER:
        return (1, float(_FACTOR_VALUE_ORDER[value]), value)
    if isinstance(value, str):
        return (2, 0.0, value)
    return (4, 0.0, repr(value))


def _row_factor_value(
    row: Dict[str, object],
    key: str,
):
    """
    读取因子值：优先行级字段（mode/k1/k2/seed/...），
    回退到受控变量协议（lr/batch_size/...）。
    """
    if key in row:
        return row.get(key)

    protocol = row.get("protocol") or {}
    if isinstance(protocol, dict):
        return protocol.get(key)

    return None


def _format_factor_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def check_controlled_variables(
    rows: List[Dict[str, object]],
    factors: List[str],
) -> Dict[str, object]:
    """
    受控变量检查：声明为因子的键允许跨运行不同，其余协议项
    必须逐项一致。

    返回:
        {
          "factors": [...],
          "checked_keys": [...],
          "violations": [{"key", "values": {run: value}, "missing_runs": [...]}],
          "n_neural_runs": int,
          "statistical_runs": [run, ...],
          "ok": bool,
        }
    """
    factor_set = set(factors)

    neural: List[Dict[str, object]] = []
    statistical: List[str] = []

    for row in rows:
        protocol = row.get("protocol") or {}
        family = "neural"
        if isinstance(protocol, dict):
            family = protocol.get("comparison_family") or "neural"
        if family == "statistical":
            statistical.append(str(row.get("run")))
        else:
            neural.append(row)

    checked: List[str] = []
    violations: List[Dict[str, object]] = []

    for key in COMPARISON_PROTOCOL_KEYS:
        if key in factor_set:
            continue

        values = [
            (
                str(row.get("run")),
                row.get("protocol", {}).get(key)
                if isinstance(row.get("protocol"), dict)
                else None,
            )
            for row in neural
        ]

        present = [
            (name, value)
            for name, value in values
            if value is not None
        ]
        missing_runs = [
            name for name, value in values if value is None
        ]

        if not present:
            # 所有运行都缺记录：无法判定，不计入检查也不报违规。
            continue

        checked.append(key)

        distinct = {
            repr(_stable_factor_key(value))
            for _, value in present
        }

        if len(distinct) > 1 or missing_runs:
            violations.append({
                "key": key,
                "values": {
                    name: value for name, value in values
                },
                "missing_runs": missing_runs,
            })

    return {
        "factors": list(factors),
        "checked_keys": checked,
        "violations": violations,
        "n_neural_runs": len(neural),
        "statistical_runs": statistical,
        "ok": not violations,
    }


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _sample_std(values: List[float]) -> Optional[float]:
    """
    样本标准差（ddof=1）；少于 2 个观测时返回 None。
    """
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum(
        (value - mean) ** 2 for value in values
    ) / (len(values) - 1)
    return math.sqrt(variance)


def _betacf(a: float, b: float, x: float) -> float:
    """
    正则化不完全 Beta 函数的连分式（Lentz 算法），
    用于无 SciPy 依赖地计算 t 分布 p 值。
    """
    max_iterations = 200
    epsilon = 3.0e-12
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration

        numerator = (
            iteration
            * (b - iteration)
            * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + numerator / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        numerator = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + numerator / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return h


def _regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
) -> float:
    """
    I_x(a, b)：正则化不完全 Beta 函数。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
    )
    front = math.exp(
        log_beta + a * math.log(x) + b * math.log1p(-x)
    )

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a

    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _two_sided_t_p_value(t: float, df: float) -> Optional[float]:
    """
    Student t 分布的双侧 p 值 P(|T| >= |t|)。
    """
    if df <= 0.0:
        return None
    if not math.isfinite(t):
        return 0.0

    x = df / (df + t * t)
    return _regularized_incomplete_beta(x, df / 2.0, 0.5)


def _paired_t_test(
    differences: List[float],
) -> Dict[str, object]:
    """
    配对 t 检验（同一组种子上的差值）：检验均值是否为 0。
    """
    n = len(differences)
    if n < _MIN_SEEDS_FOR_STATS:
        return {"test": "insufficient_seeds"}

    mean = sum(differences) / n
    std = _sample_std(differences)

    if std == 0.0:
        if mean == 0.0:
            return {
                "test": "paired_t",
                "t": 0.0,
                "df": float(n - 1),
                "p_value": 1.0,
                "note": "zero_variance",
            }
        return {
            "test": "paired_t",
            "t": None,
            "df": float(n - 1),
            "p_value": 0.0,
            "note": "noise_free_separation",
        }

    t = mean / (std / math.sqrt(n))
    return {
        "test": "paired_t",
        "t": t,
        "df": float(n - 1),
        "p_value": _two_sided_t_p_value(abs(t), float(n - 1)),
    }


def _welch_t_test(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, object]:
    """
    Welch t 检验（种子不重叠或数量不等时使用，不假设等方差）。
    """
    n_a = len(values_a)
    n_b = len(values_b)
    if n_a < _MIN_SEEDS_FOR_STATS or n_b < _MIN_SEEDS_FOR_STATS:
        return {"test": "insufficient_seeds"}

    mean_a = sum(values_a) / n_a
    mean_b = sum(values_b) / n_b
    var_a = _sample_std(values_a) ** 2
    var_b = _sample_std(values_b) ** 2

    se_squared = var_a / n_a + var_b / n_b
    delta = mean_b - mean_a

    if se_squared == 0.0:
        if delta == 0.0:
            return {
                "test": "welch_t",
                "t": 0.0,
                "df": None,
                "p_value": 1.0,
                "note": "zero_variance",
            }
        return {
            "test": "welch_t",
            "t": None,
            "df": None,
            "p_value": 0.0,
            "note": "noise_free_separation",
        }

    t = delta / math.sqrt(se_squared)
    df = se_squared ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1)
        + (var_b / n_b) ** 2 / (n_b - 1)
    )
    return {
        "test": "welch_t",
        "t": t,
        "df": df,
        "p_value": _two_sided_t_p_value(abs(t), df),
    }


def _metric_samples(
    group_rows: List[Dict[str, object]],
    metric: str,
) -> List[Dict[str, object]]:
    """
    提取一组运行在某指标上的 (seed, run, value) 样本。
    """
    samples: List[Dict[str, object]] = []
    for row in group_rows:
        value = row.get(metric)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            samples.append({
                "seed": row.get("seed"),
                "run": row.get("run"),
                "value": float(value),
            })
    return samples


def aggregate_runs(
    rows: List[Dict[str, object]],
    group_keys: List[str],
    metric_keys: Tuple[str, ...] = COMPARISON_METRIC_KEYS,
) -> List[Dict[str, object]]:
    """
    按因子组合分桶，汇报 Mean ± Std（样本标准差）。

    每个桶 = 一个实验单元（cell），桶内不同随机种子是
    重复实验；std 反映种子间方差。桶内只有 1 个种子时 std
    为 None，分析阶段会提示统计可靠性不足。
    """
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    order: List[tuple] = []

    for row in rows:
        key = tuple(
            _stable_factor_key(_row_factor_value(row, name))
            for name in group_keys
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    aggregates: List[Dict[str, object]] = []

    for key in order:
        group_rows = groups[key]
        factor_values = [
            _row_factor_value(group_rows[0], name)
            for name in group_keys
        ]

        if group_keys:
            label = ", ".join(
                f"{name}={_format_factor_value(value)}"
                for name, value in zip(group_keys, factor_values)
            )
        else:
            label = "(全部运行)"

        aggregate: Dict[str, object] = {
            "label": label,
            "keys": dict(zip(group_keys, factor_values)),
            "n_runs": len(group_rows),
            "runs": [str(r.get("run")) for r in group_rows],
            "seeds": [
                r.get("seed")
                for r in group_rows
                if r.get("seed") is not None
            ],
            "samples": {},
        }

        for metric in metric_keys:
            samples = _metric_samples(group_rows, metric)
            values = [float(s["value"]) for s in samples]
            mean = _mean(values)
            std = _sample_std(values)

            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_n"] = len(values)
            aggregate["samples"][metric] = samples

            # 没有样本的指标从 JSON 报告中省略，避免大量 null。
            if mean is None:
                aggregate.pop(f"{metric}_mean", None)
                aggregate.pop(f"{metric}_std", None)
                aggregate.pop(f"{metric}_n", None)
                aggregate["samples"].pop(metric, None)

        aggregates.append(aggregate)

    aggregates.sort(
        key=lambda aggregate: [
            _factor_sort_rank(aggregate["keys"].get(name))
            for name in group_keys
        ]
    )

    return aggregates


def metric_higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER_METRICS


def _contrast_test(
    samples_a: List[Dict[str, object]],
    samples_b: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    对比两个实验单元的指标样本：种子集合完全相同时用配对
    t 检验（同种子 = 配对重复），否则用 Welch t 检验。
    """
    by_seed_a = {
        sample["seed"]: float(sample["value"])
        for sample in samples_a
        if sample["seed"] is not None
    }
    by_seed_b = {
        sample["seed"]: float(sample["value"])
        for sample in samples_b
        if sample["seed"] is not None
    }

    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))

    paired = (
        len(common_seeds) >= _MIN_SEEDS_FOR_STATS
        and len(common_seeds) == len(by_seed_a)
        and len(common_seeds) == len(by_seed_b)
    )

    if paired:
        differences = [
            by_seed_b[seed] - by_seed_a[seed]
            for seed in common_seeds
        ]
        result = _paired_t_test(differences)
        result["n_pairs"] = len(common_seeds)
        return result

    values_a = [float(sample["value"]) for sample in samples_a]
    values_b = [float(sample["value"]) for sample in samples_b]
    result = _welch_t_test(values_a, values_b)
    result["n_pairs"] = 0
    return result


def factor_contrasts(
    aggregates: List[Dict[str, object]],
    group_keys: List[str],
    metric: str,
) -> List[Dict[str, object]]:
    """
    单因子对比：只有恰好一个因子取值不同的两个实验单元之间
    做统计检验。全因子消融的每个因子主效应（以及交互）都由
    若干这样的单因子对比给出；一次变化多个因子的组合被标记
    为混杂对比，结论无法归因。
    """
    contrasts: List[Dict[str, object]] = []
    higher_is_better = metric_higher_is_better(metric)

    for index_a in range(len(aggregates)):
        for index_b in range(index_a + 1, len(aggregates)):
            cell_a = aggregates[index_a]
            cell_b = aggregates[index_b]

            differing = [
                name
                for name in group_keys
                if cell_a["keys"].get(name)
                != cell_b["keys"].get(name)
            ]

            if not differing:
                continue

            contrast: Dict[str, object] = {
                "cells": [cell_a["label"], cell_b["label"]],
                "differing_factors": differing,
                "confounded": len(differing) > 1,
            }

            if len(differing) != 1:
                contrasts.append(contrast)
                continue

            factor = differing[0]

            # 让「基线值 → 新值」的方向与自然顺序一致，
            # delta = mean(新) - mean(基线)。
            rank_a = _factor_sort_rank(
                cell_a["keys"].get(factor)
            )
            rank_b = _factor_sort_rank(
                cell_b["keys"].get(factor)
            )
            if rank_a > rank_b:
                cell_a, cell_b = cell_b, cell_a

            samples_a = cell_a["samples"].get(metric, [])
            samples_b = cell_b["samples"].get(metric, [])
            mean_a = cell_a.get(f"{metric}_mean")
            mean_b = cell_b.get(f"{metric}_mean")
            std_a = cell_a.get(f"{metric}_std")
            std_b = cell_b.get(f"{metric}_std")

            delta = (
                float(mean_b) - float(mean_a)
                if mean_a is not None and mean_b is not None
                else None
            )

            pooled_std = None
            if std_a is not None and std_b is not None:
                pooled_std = math.sqrt(
                    (float(std_a) ** 2 + float(std_b) ** 2) / 2.0
                )

            noise_ratio = None
            if delta is not None and pooled_std:
                noise_ratio = abs(delta) / pooled_std

            test = _contrast_test(samples_a, samples_b)

            if delta is None:
                verdict = "missing_metric"
            else:
                improvement = (
                    -delta if not higher_is_better else delta
                )
                if improvement > 0:
                    verdict = "improvement"
                elif improvement < 0:
                    verdict = "regression"
                else:
                    verdict = "tie"

            contrast.update({
                "factor": factor,
                "baseline": cell_a["label"],
                "variant": cell_b["label"],
                "baseline_value": cell_a["keys"].get(factor),
                "variant_value": cell_b["keys"].get(factor),
                "mean_baseline": mean_a,
                "mean_variant": mean_b,
                "std_baseline": std_a,
                "std_variant": std_b,
                "delta": delta,
                "delta_over_noise": noise_ratio,
                "test": test.get("test"),
                "t": test.get("t"),
                "df": test.get("df"),
                "p_value": test.get("p_value"),
                "n_pairs": test.get("n_pairs", 0),
                "significant": (
                    test.get("p_value") is not None
                    and float(test["p_value"]) <= _SIGNIFICANCE_LEVEL
                ),
                "verdict": verdict,
            })
            contrasts.append(contrast)

    return contrasts


def _contrast_display(
    contrast: Dict[str, object],
    metric: str,
) -> str:
    "\"local_base→A\" 等因子变化的人类可读描述。"
    factor = contrast.get("factor")
    baseline = _format_factor_value(
        contrast.get("baseline_value")
    )
    variant = _format_factor_value(
        contrast.get("variant_value")
    )
    delta = contrast.get("delta")
    p_value = contrast.get("p_value")
    test = contrast.get("test")
    n_pairs = contrast.get("n_pairs", 0)

    if delta is None:
        delta_text = "Δ=NA"
    else:
        delta_text = f"Δ={float(delta):+.4f}"

    if test == "paired_t":
        test_text = f"配对 t (n={n_pairs})"
    elif test == "welch_t":
        test_text = "Welch t"
    else:
        test_text = "种子不足"

    if p_value is None:
        p_text = "p=NA"
    else:
        p_text = f"p={float(p_value):.4f}"

    star = "*" if contrast.get("significant") else ""

    noise = contrast.get("delta_over_noise")
    noise_text = (
        f" 噪声比={float(noise):.2f}"
        if noise is not None
        else ""
    )

    verdict_map = {
        "improvement": "改善",
        "regression": "变差",
        "tie": "持平",
        "missing_metric": "缺指标",
    }
    verdict = verdict_map.get(
        str(contrast.get("verdict")),
        "",
    )

    # 同一因子的多个单因子对比（固定其余因子取不同水平）
    # 用「在 … 下」标注上下文，避免重复行无法区分。
    context_parts = [
        part
        for part in str(contrast.get("baseline", "")).split(", ")
        if part and not part.startswith(f"{factor}=")
    ]
    context = (
        "（在 " + ", ".join(context_parts) + " 下）"
        if context_parts
        else ""
    )

    # 统计显著性与实际意义分开：显著但差值小于种子波动
    # （噪声比 < 1）时提示谨慎解读；不显著时禁止声称提升。
    caveat = ""
    if contrast.get("significant") and noise is not None:
        if float(noise) < 1.0:
            caveat = "（差值小于典型种子波动，实际意义存疑）"
    elif verdict == "改善" and test in ("paired_t", "welch_t"):
        caveat = "（未达显著水平，不能据此声称提升）"

    return (
        f"{factor}: {baseline}→{variant}{context}  "
        f"{delta_text}  {verdict}  {test_text} "
        f"{p_text}{star}{noise_text}{caveat}"
    )


def compare_cells_to_baseline(
    aggregates: List[Dict[str, object]],
    baseline_mode: Optional[str],
    metric: str,
) -> List[Dict[str, object]]:
    """
    指定基线模式（--compare-baseline），把每个实验单元与
    基线单元直接对比。基线模式有多个单元（如 balance_coef
    不同）时选运行数最多的一个。
    """
    if not baseline_mode:
        return []

    matching = [
        aggregate
        for aggregate in aggregates
        if str(aggregate["keys"].get("mode")) == baseline_mode
    ]
    if not matching:
        return []

    reference = max(
        matching,
        key=lambda aggregate: int(aggregate.get("n_runs", 0)),
    )

    contrasts: List[Dict[str, object]] = []

    for aggregate in aggregates:
        if aggregate is reference:
            continue
        result = _contrast_test(
            reference["samples"].get(metric, []),
            aggregate["samples"].get(metric, []),
        )
        mean_reference = reference.get(f"{metric}_mean")
        mean_variant = aggregate.get(f"{metric}_mean")
        delta = (
            float(mean_variant) - float(mean_reference)
            if mean_reference is not None
            and mean_variant is not None
            else None
        )
        contrasts.append({
            "factor": "mode",
            "baseline": reference["label"],
            "variant": aggregate["label"],
            "baseline_value": baseline_mode,
            "variant_value": aggregate["keys"].get("mode"),
            "mean_baseline": mean_reference,
            "mean_variant": mean_variant,
            "std_baseline": reference.get(f"{metric}_std"),
            "std_variant": aggregate.get(f"{metric}_std"),
            "delta": delta,
            "delta_over_noise": None,
            "test": result.get("test"),
            "t": result.get("t"),
            "df": result.get("df"),
            "p_value": result.get("p_value"),
            "n_pairs": result.get("n_pairs", 0),
            "significant": (
                result.get("p_value") is not None
                and float(result["p_value"])
                <= _SIGNIFICANCE_LEVEL
            ),
            "verdict": (
                "missing_metric"
                if delta is None
                else "improvement"
                if (
                    (-delta if not metric_higher_is_better(metric) else delta)
                    > 0
                )
                else "regression"
                if delta != 0.0
                else "tie"
            ),
        })

    return contrasts


def ablation_coverage(
    rows: List[Dict[str, object]],
    factors: List[str],
) -> Dict[str, object]:
    """
    消融完整性检查：延迟合并（local_base→A）与负载均衡损失
    （balance_coef 开/关）构成 2×2 全因子。报告缺失的组合、
    种子数不足的单元，以及「B 与 smelt 计算量不等价」这类
    会把额外计算误解释为方法收益的提醒。
    """
    factor_set = set(factors)
    legend = "local_base/b=off, local_base/b=on, A/b=off, A/b=on"

    report: Dict[str, object] = {
        "legend": legend,
        "applicable": (
            "mode" in factor_set or "balance_coef" in factor_set
        ),
        "present_cells": [],
        "missing_cells": [],
        "low_replicate_cells": [],
        "advisories": [],
        "n_rows": len(rows),
    }

    if not report["applicable"]:
        report["advisories"].append(
            "对比因子未包含 mode/balance_coef，"
            "无法检查延迟合并与均衡损失的消融覆盖"
        )
        return report

    modes_present: Dict[str, int] = {}
    balance_by_mode: Dict[str, List[object]] = {}
    merge_pair_cells: Dict[tuple, int] = {}
    unknown_balance = False

    for row in rows:
        mode = str(_row_factor_value(row, "mode"))
        balance = _row_factor_value(row, "balance_coef")
        modes_present[mode] = modes_present.get(mode, 0) + 1
        if mode in {"local_base", "A"}:
            balance_by_mode.setdefault(mode, [])
            if balance not in balance_by_mode[mode]:
                balance_by_mode[mode].append(balance)
            if balance is None:
                # 未记录的 balance_coef 不能算作开启/关闭任何
                # 一档，否则 2×2 覆盖表会虚报组合存在。
                unknown_balance = True
                continue
            key = (
                mode,
                "off" if balance == 0 else "on",
            )
            merge_pair_cells[key] = merge_pair_cells.get(key, 0) + 1

    if unknown_balance:
        report["low_replicate_cells"].append(
            "存在未记录 balance_coef 的 local_base/A 运行，"
            "其消融单元格无法判定"
        )

    if "B" in modes_present and "smelt" in modes_present:
        report["advisories"].append(
            "B 在第一轮按 k1 个分支执行后续 Attention/MoE，"
            "与 smelt 直接比较会把计算量差异混入路由机制："
            "应同时报告等计算的 A vs local_base 对照"
            "（或按 FLOPs 对齐比较）"
        )

    if rows and not ({"local_base", "A"} & set(modes_present)):
        # 运行集里没有 local_base/A：这不是「消融缺失」，而是
        # 当前对比与延迟合并消融无关（如 smelt 对比 B、统计
        # 基线对比），给出提醒但不虚报缺失组合。
        report["advisories"].append(
            "当前运行集不含 local_base/A，2×2 消融覆盖不适用；"
            "若要验证延迟合并，需补充立即合并对照 "
            "local_base 与延迟合并 A（全因子另加 balance "
            "开关两档）"
        )
        return report

    if "mode" in factor_set:
        if "local_base" not in modes_present:
            report["missing_cells"].append(
                "缺少立即合并对照模式 local_base："
                "无法隔离延迟合并本身带来的收益"
            )
        if "A" not in modes_present:
            report["missing_cells"].append(
                "缺少局部延迟合并模式 A"
            )

    if (
        "balance_coef" in factor_set
        and "mode" in factor_set
    ):
        # 2×2 四个格子的存在性（两轴都在因子中时）。
        for mode in ("local_base", "A"):
            for level in ("off", "on"):
                key = (mode, level)
                if key not in merge_pair_cells:
                    report["missing_cells"].append(
                        f"缺少全因子组合 {mode}/b={level}"
                        "（无法证明各创新点各自有用）"
                    )
    elif "balance_coef" in factor_set:
        # 只有 balance_coef 是因子：按本项目的合并模式检查
        # 开启臂 / 关闭臂。
        for mode in ("local_base", "A"):
            if mode not in modes_present:
                continue
            balances = balance_by_mode.get(mode, [])
            known = [b for b in balances if b is not None]
            if not known:
                report["low_replicate_cells"].append(
                    f"{mode} 未记录 balance_coef"
                )
                continue
            if 0 not in known:
                report["missing_cells"].append(
                    f"{mode} 缺少 balance_coef=0 的关闭臂："
                    "无法证明均衡损失各自有用"
                )
            if not any(
                isinstance(b, (int, float))
                and not isinstance(b, bool)
                and b > 0
                for b in known
            ):
                report["missing_cells"].append(
                    f"{mode} 缺少 balance_coef>0 的开启臂"
                )

    for (mode, level), count in sorted(merge_pair_cells.items()):
        label = f"{mode}/b={level}"
        report["present_cells"].append({
            "cell": label,
            "n_runs": count,
        })
        if count < _RECOMMENDED_SEEDS:
            report["low_replicate_cells"].append(
                f"{label} 只有 {count} 个种子"
                f"（建议 ≥{_RECOMMENDED_SEEDS}）"
            )

    if (
        "balance_coef" in factor_set
        and len(merge_pair_cells) == 4
        and not report["missing_cells"]
    ):
        report["advisories"].append(
            "2×2 全因子组合齐全："
            "可同时给出延迟合并与均衡损失的主效应及交互"
        )

    return report


def print_comparison_analysis(
    rows: List[Dict[str, object]],
    factors: Optional[List[str]] = None,
    baseline_mode: Optional[str] = None,
    metric: str = "val_loss",
) -> Dict[str, object]:
    """
    打印完整的对比分析：受控变量检查、按种子聚合的
    Mean ± Std、单因子显著性检验与消融覆盖检查。

    返回结构化结果，供写入汇总报告。
    """
    if factors is None:
        factors = list(DEFAULT_COMPARISON_FACTORS)

    if metric not in COMPARISON_METRIC_KEYS:
        raise ValueError(
            f"未知的对比指标: {metric}（可用: "
            + ", ".join(COMPARISON_METRIC_KEYS)
            + "）"
        )

    controlled = check_controlled_variables(rows, factors)

    # 只有实际跨运行变化的因子才定义实验单元；恒定因子
    # （例如所有运行都是 k1=2）不参与分组，cell 标签保持
    # 简洁可读。seed 是重复实验维度，不进分组。
    group_keys = []
    for key in factors:
        if key == "seed":
            continue
        distinct = {
            repr(_stable_factor_key(_row_factor_value(row, key)))
            for row in rows
        }
        if len(distinct) > 1:
            group_keys.append(key)

    aggregates = aggregate_runs(rows, group_keys)
    contrasts = factor_contrasts(aggregates, group_keys, metric)
    reference = compare_cells_to_baseline(
        aggregates,
        baseline_mode,
        metric,
    )
    coverage = ablation_coverage(rows, factors)

    higher_is_better = metric_higher_is_better(metric)

    print()
    print(
        "[controlled] 受控变量检查：除声明因子外，"
        "优化器 / 学习率 / 训练轮数 / 数据增强等协议项"
        "必须逐项一致"
    )
    print("  声明因子: " + ", ".join(factors))

    if controlled["statistical_runs"]:
        print(
            "  统计基线（ngram，无优化器与训练轮数）"
            "不参与受控变量检查: "
            + ", ".join(controlled["statistical_runs"])
        )

    if not rows:
        print("  无运行目录可分析")
    elif not controlled["violations"]:
        print(
            f"  OK：{len(controlled['checked_keys'])} 项协议"
            "在声明因子之外完全一致"
        )
    else:
        for violation in controlled["violations"]:
            values_text = ", ".join(
                f"{run}={_format_factor_value(value)}"
                for run, value in violation["values"].items()
            )
            print(
                f"  [violation] {violation['key']}: {values_text}"
            )
        print(
            "  以上差异会使对比结论无法归因到待验证模块；"
            "请统一这些设置后重跑，或在 --compare-factors 中"
            "显式声明为因子"
        )

    print()
    print(
        "[aggregate] 按实验单元聚合（Mean ± Std，样本标准差；"
        "std 反映不同随机种子的方差）"
    )

    if aggregates:
        # 聚合表只展示 val 口径：test 指标不参与跨运行
        # 对比（尘封协议）。
        header = (
            f"{'cell':<40} {'n':>3} {'val_loss':>18}"
        )
        print(header)
        print("-" * len(header))

        def _stat(aggregate, base_metric: str) -> str:
            mean = aggregate.get(f"{base_metric}_mean")
            if mean is None:
                return "-"
            std = aggregate.get(f"{base_metric}_std")
            if std is None:
                return f"{float(mean):.4f}"
            return f"{float(mean):.4f}±{float(std):.4f}"

        for aggregate in aggregates:
            print(
                f"{str(aggregate['label'])[:39]:<40} "
                f"{int(aggregate['n_runs']):>3} "
                f"{_stat(aggregate, 'val_loss'):>18}"
            )

        underpowered = [
            aggregate["label"]
            for aggregate in aggregates
            if int(aggregate["n_runs"]) < _RECOMMENDED_SEEDS
        ]
        if underpowered:
            print(
                "  统计可靠性不足：以下实验单元种子数 < "
                f"{_RECOMMENDED_SEEDS}（建议 3~5 个不同随机"
                "种子并汇报 Mean ± Std）: "
                + "; ".join(underpowered)
            )
    else:
        print("  无有效指标可聚合")

    print()
    direction = "越低越好" if not higher_is_better else "越高越好"
    print(
        f"[stats] 指标 {metric}（{direction}）；单因子对比"
        "（每个对比只改变一个因子）"
    )

    if not contrasts:
        print("  无单因子对比（至少需要两个不同的实验单元）")
    else:
        for contrast in contrasts:
            if contrast.get("confounded"):
                print(
                    f"  [confounded] {contrast['cells'][0]} vs "
                    f"{contrast['cells'][1]}: 同时改变了 "
                    + ", ".join(contrast["differing_factors"])
                    + "，结论无法归因"
                )
                continue
            print(
                "  " + _contrast_display(contrast, metric)
            )

    if baseline_mode:
        print(
            f"  指定基线 mode={baseline_mode} 的直接对比:"
        )
        if not reference:
            print(
                f"    未找到 mode={baseline_mode} 的运行"
            )
        for contrast in reference:
            print(
                "    " + _contrast_display(contrast, metric)
            )

    print(
        "  注：未做多重比较校正，p 值仅作参考；"
        "结论以预设的关键对比为准，且种子数应 ≥"
        f"{_RECOMMENDED_SEEDS}"
    )

    print()
    print(
        "[ablation] 消融覆盖检查（延迟合并 × 均衡损失 2×2："
        f"{coverage['legend']}）"
    )
    if coverage["missing_cells"]:
        for message in coverage["missing_cells"]:
            print(f"  [missing] {message}")
    if coverage["present_cells"]:
        present = ", ".join(
            f"{cell['cell']}(n={cell['n_runs']})"
            for cell in coverage["present_cells"]
        )
        print(f"  已覆盖组合: {present}")
    if coverage["low_replicate_cells"]:
        for message in coverage["low_replicate_cells"]:
            print(f"  [warn] {message}")
    for advisory in coverage["advisories"]:
        print(f"  [note] {advisory}")
    if (
        not coverage["missing_cells"]
        and coverage["present_cells"]
        and not coverage["low_replicate_cells"]
    ):
        print("  OK：2×2 全因子组合齐全且种子数达标")

    return {
        "factors": factors,
        "metric": metric,
        "controlled": controlled,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "reference_contrasts": reference,
        "ablation_coverage": coverage,
        "n_runs": len(rows),
    }


def write_comparison_report(
    out_dir: Path,
    analysis: Dict[str, object],
) -> Path:
    """
    把对比分析写入 out_dir/comparison_analysis.json。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "comparison_analysis.json"

    payload = {
        "generated_at": datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        **analysis,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe(payload),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[report] 对比分析已写入 {path}")
    return path


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
    reference_baselines: Optional[Dict[str, object]] = None,
) -> None:
    """
    写出 config.json：结构配置、完整超参数、参数量、数据指纹、
    数据划分与审计结果、代码版本/运行环境、sanity check 结果、
    本轮实验的科学假设（--hypothesis/--rq）与参考基线
    （uniform / unigram）。
    """
    payload = {
        "config": asdict(cfg),
        "args": vars(args),
        # 受控变量协议：--compare 用它核对「除声明因子外，
        # 优化器 / 学习率 / 训练轮数 / 数据增强等设置是否一致」。
        "comparison_protocol": build_comparison_protocol(cfg, args),
        "n_params": n_params,
        "hypothesis": args.hypothesis,
        "research_question": args.rq,
        "hypothesis_check_skipped": bool(
            args.skip_hypothesis_check
        ),
        "reference_baselines": reference_baselines,
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
            "group_separator": getattr(
                args,
                "group_separator",
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

    # config.json 是实验追踪的唯一来源，写入中途崩溃不能留下
    # 半截文件：与 update_run_config 一样走临时文件 + 原子替换。
    previous_mode = 0o644
    if config_path.exists():
        try:
            previous_mode = config_path.stat().st_mode & 0o777
        except OSError:
            previous_mode = 0o644

    temp_path: Optional[Path] = None
    try:
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

        os.replace(temp_path, config_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    try:
        os.chmod(config_path, previous_mode)
    except OSError:
        pass


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

    temp_path: Optional[Path] = None
    try:
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
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

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

    if not isinstance(payload, dict):
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
) -> None:
    """
    保存「验证集选定」的模型权重（不含优化器状态）。

    每次周期验证刷新最优验证损失时覆盖写入。最终评估与唯一
    一次测试集评估都在该模型上进行，保证测试集看到的是经
    验证集选择的模型，而不是训练结束时恰好排在上一步的参数。
    与 save_checkpoint 一样先写临时文件再原子替换。
    """
    raw_model = getattr(model, "_orig_mod", model)

    payload = {
        "config": asdict(raw_model.cfg),
        "model": raw_model.state_dict(),
        "step": step,
        "val_loss": float(val_loss),
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


def load_selected_model(
    filename: str,
    device: torch.device,
) -> Optional[dict]:
    """
    读取 best.pt；缺失或损坏时返回 None（调用方回退到最后
    一步参数并明确提示）。
    """
    path = Path(filename)

    if not path.exists():
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

    return payload


def restore_selected_weights(
    model: LanguageModel,
    payload: dict,
) -> None:
    """把 best.pt 中的权重加载回模型（兼容 torch.compile 包装）。"""
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(payload["model"])


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

    # 所有模式都必须可以前向和反向（含 dense 简单基线）。
    for mode in (
        "smelt",
        "local_base",
        "A",
        "B",
        "dense",
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

    # dense 基线：无 MoE、无循环——每个 Block 的 attention
    # 与 FFN 恰好各执行一次；smelt 的中间区域则执行两轮。
    def _count_block_calls(cfg: Config) -> int:
        probe_model = LanguageModel(cfg)
        calls = [0]

        for block in probe_model.blocks:
            original_step = block.attention_step

            def probe_attention_step(
                inputs: torch.Tensor,
                _original=original_step,
            ) -> torch.Tensor:
                calls[0] += 1
                return _original(inputs)

            block.attention_step = probe_attention_step

        with torch.no_grad():
            probe_model(x)

        return calls[0]

    n_layers = common["n_layers"]
    middle_count = n_layers // 2

    assert (
        _count_block_calls(
            Config(**common, mode="dense")
        )
        == n_layers
    )
    assert (
        _count_block_calls(
            Config(**common, mode="smelt")
        )
        == n_layers + middle_count
    )

    # dense 基线的默认 FFN 宽度 = top_k * expert_hidden
    # （等计算口径），显式传参时以显式值为准。
    dense_default = Config(**common, mode="dense")
    assert (
        dense_default.effective_dense_fnn_hidden
        == common["top_k"] * common["expert_hidden"]
    )
    dense_explicit = Config(
        **common,
        mode="dense",
        dense_fnn_hidden=123,
    )
    assert (
        dense_explicit.effective_dense_fnn_hidden
        == 123
    )
    try:
        Config(
            **common,
            mode="dense",
            dense_fnn_hidden=-1,
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "非法 dense_fnn_hidden 必须被拒绝"
        )

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
    # 这份合成语料覆盖 0..255 全部字节值（含大量非 UTF-8
    # 字节），严格审计会把它标记为标签噪声来源；这不影响
    # 上面的重复检测断言。

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
    # 跨 split 重复内容是泄露级问题：严格审计必须失败，
    # 且条目进入 violations 而不是只有 warnings。
    assert repeated_audit["violations"]
    assert any(
        "重复" in violation
        for violation in repeated_audit["violations"]
    )
    assert not enforce_strict_data_audit(repeated_audit)

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

    # 文本完整性审计：无法 UTF-8 解码的字节（标签噪声来源）与
    # NUL 字节必须被检测到并给出警告。
    with tempfile.NamedTemporaryFile() as corrupt_file:
        corrupt_file.write(
            b"valid utf-8 text " * 30
            + b"\xff\xfe"
            + b"\x00" * 4
            + b"tail text " * 30
        )
        corrupt_file.flush()

        corrupt_dataset = ByteDataset(
            corrupt_file.name,
            seq_len=8,
        )

        corrupt_audit = audit_data_splits(
            corrupt_dataset,
            include_test=True,
        )

    assert (
        corrupt_audit["splits"]["train"]["utf8_replacements"]
        > 0
    )
    assert corrupt_audit["splits"]["train"]["nul_bytes"] == 4
    assert any(
        "UTF-8" in warning
        for warning in corrupt_audit["warnings"]
    )
    assert any(
        "NUL" in warning
        for warning in corrupt_audit["warnings"]
    )
    assert corrupt_audit["violations"]
    assert not enforce_strict_data_audit(corrupt_audit)

    # 多字节 UTF-8 字符横跨 split 边界不能产生假阳性「标签
    # 噪声」：文件整体合法，切分只是诊断视角。字符「€」的
    # 三个字节中，前两个落在 train 末尾、最后一个落在 val 开头。
    boundary_payload = (
        b"a" * 161 + "\u20ac".encode("utf-8") + b"b" * 16
    )
    assert len(boundary_payload) == 180

    with tempfile.NamedTemporaryFile() as boundary_file:
        boundary_file.write(boundary_payload)
        boundary_file.flush()

        # val = [162, 171)，train/val 边界正好落在「€」中间。
        boundary_dataset = ByteDataset(
            boundary_file.name,
            seq_len=8,
        )
        assert boundary_dataset.split_bounds["train"] == (0, 162)
        assert boundary_dataset.split_bounds["val"] == (162, 171)

        boundary_audit = audit_data_splits(
            boundary_dataset,
            include_test=True,
        )

    for split_name in ("train", "val", "test"):
        assert (
            boundary_audit["splits"][split_name][
                "utf8_replacements"
            ]
            == 0
        )
    assert not any(
        "UTF-8" in warning
        for warning in boundary_audit["warnings"]
    )
    # 合法 UTF-8、无重复的语料没有任何严格级别的致命问题。
    assert boundary_audit["violations"] == []
    assert enforce_strict_data_audit(boundary_audit)

    # 群组分隔符解析：转义、字面量、Unicode 与非法输入。
    assert parse_group_separator("\\n\\n") == b"\n\n"
    assert parse_group_separator("<|end|>") == b"<|end|>"
    assert parse_group_separator("\\x00") == b"\x00"
    assert parse_group_separator("\\u2028") == "\u2028".encode(
        "utf-8"
    )
    assert parse_group_separator("\\\\n") == b"\\n"

    for invalid in ("", "\\", "\\x1", "\\q"):
        try:
            parse_group_separator(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"非法群组分隔符必须报错: {invalid!r}"
            )

    # 群组数据协议：切分位置吸附到群组边界，任何群组都整体
    # 落在同一个 split 内，train/val/test 之间不交叉。
    separator = b"\n\n"
    groups = [
        bytes([97 + index]) * 10 for index in range(5)
    ]
    grouped_payload = separator.join(groups) + separator
    assert len(grouped_payload) == 5 * 12

    with tempfile.NamedTemporaryFile() as grouped_file:
        grouped_file.write(grouped_payload)
        grouped_file.flush()

        grouped_dataset = ByteDataset(
            grouped_file.name,
            seq_len=8,
            val_fraction=0.2,
            test_fraction=0.2,
            group_separator=separator,
        )

        # 名义位置 (36, 48) 恰好是群组边界，无需吸附。
        assert grouped_dataset.split_bounds["train"] == (0, 36)
        assert grouped_dataset.split_bounds["val"] == (36, 48)
        assert grouped_dataset.split_bounds["test"] == (48, 60)
        assert grouped_dataset.groups_per_split == {
            "train": 3,
            "val": 1,
            "test": 1,
        }

        # 每个 split 自身都以完整的群组为单位：train 以分隔符
        # 结尾，val/test 是完整群组的字节。
        assert bytes(grouped_dataset.train[-2:]) == separator
        assert bytes(grouped_dataset.val) == groups[3] + separator
        assert bytes(grouped_dataset.test) == groups[4] + separator

        # 训练批只来自训练群组（内容字节 97、98、99 与分隔符
        # 0x0a），不会出现 val/test 群组的内容字节 100/101；
        # val 的固定窗口只来自 val 群组。
        grouped_x, grouped_y = grouped_dataset.batch(
            "train",
            4,
            torch.Generator().manual_seed(3),
            eval_device,
        )
        assert int(grouped_x.min()) >= 10
        assert int(grouped_x.max()) <= 99
        assert int(grouped_y.max()) <= 99

        grouped_val = grouped_dataset.fixed_eval_batch(
            "val",
            0,
            1,
            2,
            eval_device,
        )[0]
        assert int(grouped_val.min()) == 10
        assert int(grouped_val.max()) == 100

        grouped_audit = audit_data_splits(
            grouped_dataset,
            include_test=True,
        )

    assert grouped_audit["groups"]["groups_total"] == 5
    assert grouped_audit["groups"]["groups_per_split"] == {
        "train": 3,
        "val": 1,
        "test": 1,
    }
    assert grouped_audit["violations"] == []

    # 名义位置不落在边界上时必须吸附到最近的群组边界。
    separator_five = b"<END>"
    groups_five = [
        bytes([65 + index * 5]) * 15 for index in range(4)
    ]
    aligned_payload = separator_five.join(groups_five) + separator_five
    assert len(aligned_payload) == 80

    with tempfile.NamedTemporaryFile() as aligned_file:
        aligned_file.write(aligned_payload)
        aligned_file.flush()

        aligned_dataset = ByteDataset(
            aligned_file.name,
            seq_len=8,
            val_fraction=0.2,
            test_fraction=0.2,
            group_separator=separator_five,
        )

    # 名义 (48, 64) → 吸附到 (40, 60)：2 / 1 / 1 个群组。
    assert aligned_dataset.split_bounds["train"] == (0, 40)
    assert aligned_dataset.split_bounds["val"] == (40, 60)
    assert aligned_dataset.split_bounds["test"] == (60, 80)
    assert aligned_dataset.groups_per_split == {
        "train": 2,
        "val": 1,
        "test": 1,
    }

    # 群组过大、无法在保证每段最小长度的情况下整体划分：
    # 必须显式报错，而不是拆分群组或产生空 split。
    with tempfile.NamedTemporaryFile() as oversized_file:
        oversized_file.write(b"x" * 100 + b"SEP" + b"y" * 5)
        oversized_file.flush()

        try:
            ByteDataset(
                oversized_file.name,
                seq_len=8,
                group_separator=b"SEP",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "群组无法整体划分时必须报错"
            )

    # 语料中根本没有分隔符时也要显式报错。
    with tempfile.NamedTemporaryFile() as no_sep_file:
        no_sep_file.write(b"z" * 120)
        no_sep_file.flush()

        try:
            ByteDataset(
                no_sep_file.name,
                seq_len=8,
                group_separator=b"<MISSING>",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "分隔符不存在时必须报错"
            )

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

    # 验证集选中的模型存档：保存 / 读取 / 恢复必须逐位一致；
    # best.pt 缺失时安全返回 None（调用方回退到最后一步并提示）。
    with tempfile.TemporaryDirectory() as selected_dir:
        selected_path = Path(selected_dir) / "best.pt"

        selection_model = LanguageModel(
            Config(**common, mode="B")
        )
        with torch.no_grad():
            for parameter in selection_model.parameters():
                parameter.add_(0.25)

        expected_state = {
            key: value.clone()
            for key, value in selection_model.state_dict().items()
        }

        selected_args = argparse.Namespace(seed=1, mode="B")
        save_selected_model(
            str(selected_path),
            selection_model,
            7,
            1.25,
            selected_args,
        )

        assert selected_path.exists()
        assert not selected_path.with_name(
            selected_path.name + ".tmp"
        ).exists()

        with torch.no_grad():
            for parameter in selection_model.parameters():
                parameter.zero_()

        selected_payload = load_selected_model(
            selected_path,
            torch.device("cpu"),
        )
        assert selected_payload is not None
        assert int(selected_payload["step"]) == 7
        assert abs(
            float(selected_payload["val_loss"]) - 1.25
        ) < 1e-12

        restore_selected_weights(
            selection_model,
            selected_payload,
        )

        restored_state = selection_model.state_dict()
        assert all(
            torch.equal(expected_state[key], restored_state[key])
            for key in expected_state
        )

        assert (
            load_selected_model(
                Path(selected_dir) / "missing.pt",
                torch.device("cpu"),
            )
            is None
        )

        # 损坏的 best.pt（截断/非存档内容）同样安全返回 None，
        # 由调用方回退到最后一步参数。
        corrupt_selected_path = (
            Path(selected_dir) / "corrupt.pt"
        )
        corrupt_selected_path.write_bytes(
            b"\x00\x01not a torch archive"
        )
        assert (
            load_selected_model(
                corrupt_selected_path,
                torch.device("cpu"),
            )
            is None
        )

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

    # 非整数 step 的记录（如最终汇总）不参与早停判定。
    (
        early_best,
        early_best_step,
        early_no_improve,
    ) = early_stop_state_from_records(
        [
            {"step": 10, "val_loss": 5.0},
            {"step": 20, "val_loss": 4.0},
            {"step": "final", "val_loss": 9.9},
        ],
        min_delta=0.0,
    )
    assert early_best == 4.0
    assert early_best_step == 20
    assert early_no_improve == 0

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
    # attention 的 FLOPs 按每个 Block 的 attention_step 实际
    # 执行次数估计；独立插桩同一批次做对照，两者必须一致。
    assert (
        efficiency["flops_attention"] is None
        or int(efficiency["flops_attention"]) > 0
    )

    expected_attention_flops = [0]
    probe_modules = []
    for probe_module in efficiency_model.modules():
        if not isinstance(probe_module, TransformerBlock):
            continue
        original_probe_step = probe_module.attention_step

        def probe_attention_step(
            inputs: torch.Tensor,
            _original=original_probe_step,
        ) -> torch.Tensor:
            expected_attention_flops[0] += (
                4 * int(inputs.size(-2)) * int(inputs.numel())
            )
            return _original(inputs)

        probe_module.attention_step = probe_attention_step
        probe_modules.append(probe_module)

    probe_generator = torch.Generator().manual_seed(0)
    probe_x, probe_y = eval_dataset.batch(
        "train",
        2,
        probe_generator,
        eval_device,
    )
    try:
        with torch.no_grad():
            efficiency_model(probe_x, probe_y)
    finally:
        for probe_module in probe_modules:
            del probe_module.attention_step

    assert (
        int(efficiency["flops_attention"])
        == expected_attention_flops[0]
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

    # ---- 对比分析：t 分布 p 值（无 SciPy 依赖） ----
    # 已知分位点：df=4 时 t_{0.975}=2.776445 对应双侧 p=0.05，
    # t_{0.995}=4.604095 对应 p=0.01。
    p_at_050 = _two_sided_t_p_value(2.7764451051977987, 4.0)
    p_at_010 = _two_sided_t_p_value(4.604094871349996, 4.0)
    assert p_at_050 is not None
    assert abs(p_at_050 - 0.05) < 1e-6, p_at_050
    assert p_at_010 is not None
    assert abs(p_at_010 - 0.01) < 1e-6, p_at_010
    assert _two_sided_t_p_value(0.0, 4.0) == 1.0
    # I_x(1,1) 是均匀分布的 CDF，应等于 x。
    assert abs(
        _regularized_incomplete_beta(0.3, 1.0, 1.0) - 0.3
    ) < 1e-9
    # 对称性：P(|T|>=t) 在 t 与 -t 上相同。
    assert abs(
        (_two_sided_t_p_value(2.0, 7.0) or 0.0)
        - (_two_sided_t_p_value(-2.0, 7.0) or 0.0)
    ) < 1e-12

    # ---- 对比分析：受控变量 / 聚合 / 显著性 / 消融覆盖 ----
    with tempfile.TemporaryDirectory() as analysis_dir:
        def _write_analysis_run(
            name: str,
            mode: str,
            seed: int,
            lr: float,
            steps: int,
            balance: float,
            val_loss: float,
        ) -> str:
            run_dir = Path(analysis_dir) / name
            run_dir.mkdir()
            protocol = {
                "comparison_family": "neural",
                "optimizer": "AdamW",
                "optimizer_betas": [0.9, 0.999],
                "optimizer_eps": 1e-8,
                "lr": lr,
                "lr_schedule": "linear_warmup_cosine_to_10pct",
                "warmup_steps": 10,
                "total_steps": steps,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "batch_size": 8,
                "data_augmentation": "none",
                "seq_len": 8,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 4,
                "n_experts": 3,
                "expert_hidden": 24,
                "top_k": 2,
                "residual_scale": 1.0,
                "val_fraction": 0.05,
                "test_fraction": 0.05,
                "eval_interval": 10,
                "eval_batches": 2,
                "data_sha256": "deadbeef",
            }
            config_payload = {
                "config": {
                    "mode": mode,
                    "top_k": 2,
                    "k1": 2,
                    "k2": 2,
                    "balance_coef": balance,
                    "dense_fnn_hidden": None,
                    "seq_len": 8,
                    "d_model": 16,
                },
                "args": {
                    "seed": seed,
                    "lr": lr,
                    "steps": steps,
                },
                "data": {"sha256": "deadbeef"},
                "comparison_protocol": protocol,
            }
            with open(
                run_dir / "config.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(config_payload, f)

            final_record: Dict[str, object] = {
                "step": "final",
                "val_loss": val_loss,
            }
            write_jsonl_record(
                run_dir / "metrics.jsonl",
                final_record,
            )
            return str(run_dir)

        run_paths = [
            _write_analysis_run(
                "local_base_seed42",
                "local_base",
                42,
                3e-4,
                100,
                0.01,
                3.00,
            ),
            _write_analysis_run(
                "local_base_seed43",
                "local_base",
                43,
                3e-4,
                100,
                0.01,
                3.10,
            ),
            _write_analysis_run(
                "local_base_seed44",
                "local_base",
                44,
                3e-4,
                100,
                0.01,
                3.20,
            ),
            _write_analysis_run(
                "A_seed42",
                "A",
                42,
                3e-4,
                100,
                0.01,
                2.80,
            ),
            _write_analysis_run(
                "A_seed43",
                "A",
                43,
                3e-4,
                100,
                0.01,
                2.85,
            ),
            _write_analysis_run(
                "A_seed44",
                "A",
                44,
                3e-4,
                100,
                0.01,
                2.90,
            ),
        ]

        factors = parse_factor_spec(
            "mode,k1,k2,balance_coef,seed,dense_fnn_hidden"
        )
        analysis_rows = compare_runs(run_paths)

        assert analysis_rows[0]["seed"] == 42
        assert analysis_rows[0]["balance_coef"] == 0.01
        assert analysis_rows[0]["protocol"]["optimizer"] == "AdamW"

        controlled = check_controlled_variables(
            analysis_rows,
            factors,
        )
        assert controlled["ok"], controlled["violations"]
        assert "lr" in controlled["checked_keys"]

        aggregates = aggregate_runs(analysis_rows, ["mode"])
        assert len(aggregates) == 2

        by_label = {
            aggregate["label"]: aggregate
            for aggregate in aggregates
        }
        local_base_cell = by_label["mode=local_base"]
        a_cell = by_label["mode=A"]
        assert local_base_cell["n_runs"] == 3
        assert (
            abs(
                local_base_cell["val_loss_mean"]
                - 3.10
            ) < 1e-9
        )
        assert (
            abs(
                local_base_cell["val_loss_std"] - 0.10
            ) < 1e-9
        )
        assert abs(a_cell["val_loss_mean"] - 2.85) < 1e-9
        assert abs(a_cell["val_loss_std"] - 0.05) < 1e-9

        contrasts = factor_contrasts(
            aggregates,
            ["mode"],
            "val_loss",
        )
        assert len(contrasts) == 1
        contrast = contrasts[0]
        assert contrast["test"] == "paired_t"
        assert contrast["n_pairs"] == 3
        assert contrast["factor"] == "mode"
        assert contrast["baseline_value"] == "local_base"
        assert contrast["variant_value"] == "A"
        # delta = mean(A) - mean(local_base) = -0.25（越低越好）。
        assert abs(contrast["delta"] - (-0.25)) < 1e-9
        assert contrast["verdict"] == "improvement"
        assert contrast["significant"]
        assert (
            contrast["p_value"] is not None
            and 0.005 < float(contrast["p_value"]) < 0.05
        )

        # 尘封协议：test 指标绝不能作为跨运行对比口径，
        # 否则「挑 test 最好」会让测试集退化成隐式验证集。
        assert "test_loss" not in COMPARISON_METRIC_KEYS
        assert all(
            "test_loss" not in row for row in analysis_rows
        ), "对比行不允许携带 test 指标"
        try:
            print_comparison_analysis(
                analysis_rows,
                factors=factors,
                metric="test_loss",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "test_loss 必须被拒绝作为对比指标"
                "（测试集不参与跨运行选择）"
            )

        # 显著且差值远大于种子波动时不加额外警告；显著但
        # 差值小于噪声比 1 时提示谨慎解读；不显著时禁止
        # 声称提升。
        display = _contrast_display(contrast, "val_loss")
        assert "存疑" not in display
        assert "不能据此声称" not in display

        weak_display = _contrast_display(
            {
                "factor": "mode",
                "baseline_value": "local_base",
                "variant_value": "A",
                "baseline": "mode=local_base",
                "delta": -0.001,
                "p_value": 0.60,
                "test": "paired_t",
                "n_pairs": 3,
                "verdict": "improvement",
                "significant": False,
                "delta_over_noise": 0.10,
            },
            "val_loss",
        )
        assert "不能据此声称提升" in weak_display

        small_effect_display = _contrast_display(
            {
                "factor": "mode",
                "baseline_value": "local_base",
                "variant_value": "A",
                "baseline": "mode=local_base",
                "delta": -0.001,
                "p_value": 0.01,
                "test": "paired_t",
                "n_pairs": 3,
                "verdict": "improvement",
                "significant": True,
                "delta_over_noise": 0.30,
            },
            "val_loss",
        )
        assert "实际意义存疑" in small_effect_display

        # 学习率不同必须被受控变量检查抓住；声明为因子后
        # 则不再报违规。
        bad_path = _write_analysis_run(
            "A_seed45_badlr",
            "A",
            45,
            1e-3,
            100,
            0.01,
            2.50,
        )
        bad_controlled = check_controlled_variables(
            compare_runs(run_paths + [bad_path]),
            factors,
        )
        assert not bad_controlled["ok"]
        assert any(
            violation["key"] == "lr"
            for violation in bad_controlled["violations"]
        )
        declared = check_controlled_variables(
            compare_runs(run_paths + [bad_path]),
            factors + ["lr"],
        )
        assert all(
            violation["key"] != "lr"
            for violation in declared["violations"]
        )

        # 训练步数不同同样必须被抓住（训练轮数是受控变量）。
        steps_path = _write_analysis_run(
            "A_seed46_badsteps",
            "A",
            46,
            3e-4,
            200,
            0.01,
            2.60,
        )
        steps_controlled = check_controlled_variables(
            compare_runs(run_paths + [steps_path]),
            factors,
        )
        assert any(
            violation["key"] == "total_steps"
            for violation in steps_controlled["violations"]
        )

        # 消融覆盖：只跑了 balance=0.01 的开启臂，缺 2×2 的
        # 关闭臂组合，必须被标记为缺失。
        coverage = ablation_coverage(analysis_rows, factors)
        assert coverage["applicable"]
        assert len(coverage["missing_cells"]) == 2, coverage
        assert any(
            "b=off" in message
            for message in coverage["missing_cells"]
        )

        # 补齐 balance=0 的两格后，2×2 组合齐全（但种子
        # 数不足会被单独提示，不会误报为缺失）。
        full_paths = run_paths + [
            _write_analysis_run(
                "local_base_bal0_seed42",
                "local_base",
                42,
                3e-4,
                100,
                0.0,
                3.05,
            ),
            _write_analysis_run(
                "A_bal0_seed42",
                "A",
                42,
                3e-4,
                100,
                0.0,
                2.95,
            ),
        ]
        full_coverage = ablation_coverage(
            compare_runs(full_paths),
            factors,
        )
        assert not full_coverage["missing_cells"], full_coverage
        assert any(
            "只有 1 个种子" in message
            for message in full_coverage["low_replicate_cells"]
        )

        # 与延迟合并无关的对比（smelt vs B）不虚报消融缺失，
        # 只提示需补 local_base/A 并给出计算量提醒。
        unrelated = [
            _write_analysis_run(
                "smelt_s1",
                "smelt",
                1,
                3e-4,
                100,
                0.01,
                3.00,
            ),
            _write_analysis_run(
                "B_s1",
                "B",
                1,
                3e-4,
                100,
                0.01,
                2.90,
            ),
        ]
        unrelated_coverage = ablation_coverage(
            compare_runs(unrelated),
            factors,
        )
        assert not unrelated_coverage["missing_cells"]
        assert any(
            "2×2 消融覆盖不适用" in message
            for message in unrelated_coverage["advisories"]
        )
        assert any(
            "计算量差异" in message
            for message in unrelated_coverage["advisories"]
        )

    # ---- 多种子 / 套件编排：计划与参数透传 ----
    seed_plan = build_sweep_plan(
        seeds=[1, 2],
        modes=None,
        ablation=False,
        balance_coef=0.01,
        default_mode="B",
    )
    assert [item["label"] for item in seed_plan] == [
        "seed1",
        "seed2",
    ]
    assert {item["mode"] for item in seed_plan} == {"B"}
    assert {item["balance_coef"] for item in seed_plan} == {0.01}

    mode_plan = build_sweep_plan(
        seeds=[7],
        modes=["local_base", "A"],
        ablation=False,
        balance_coef=0.01,
        default_mode="B",
    )
    assert [item["label"] for item in mode_plan] == [
        "local_base_seed7",
        "A_seed7",
    ]

    ablation_plan = build_sweep_plan(
        seeds=[7],
        modes=None,
        ablation=True,
        balance_coef=0.01,
        default_mode="B",
    )
    assert [item["label"] for item in ablation_plan] == [
        "local_base_bal0_seed7",
        "local_base_bal0.01_seed7",
        "A_bal0_seed7",
        "A_bal0.01_seed7",
    ]
    assert [
        item["balance_coef"] for item in ablation_plan
    ] == [0.0, 0.01, 0.0, 0.01]

    stripped = strip_sweep_args([
        "--data",
        "corpus.txt",
        "--out",
        "runs/rq1",
        "--mode",
        "B",
        "--seeds",
        "1",
        "2",
        "3",
        "--suite-ablation",
        "--steps",
        "10",
        "--balance-coef",
        "0.02",
        "--skip-sanity-check",
    ])
    assert stripped == [
        "--data",
        "corpus.txt",
        "--steps",
        "10",
        "--skip-sanity-check",
    ]

    stripped_modes = strip_sweep_args([
        "--suite-modes",
        "A",
        "B",
        "--out=runs/x",
        "--seed=5",
        "--k1",
        "2",
    ])
    assert stripped_modes == ["--k1", "2"]

    for unknown_spec in ("", "mode,not_a_factor"):
        try:
            parse_factor_spec(unknown_spec)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "非法对比因子必须被拒绝: "
                + repr(unknown_spec)
            )

    print("self-test passed")


# ============================================================
# Training
# ============================================================

def hypothesis_check_error(
    args: argparse.Namespace,
) -> Optional[str]:
    """
    假设纪律检查：全新训练必须带明确的科学假设。

    返回错误消息（应交给 parser.error 报告）或 None。

    --compare（只读对比）、--sanity-only（结构检查）、
    --statistical-baseline（建立对照坐标系）与 --resume
    （假设已记录在原运行 config.json）不受此约束；纯工程
    目的的临时运行可用 --skip-hypothesis-check 显式跳过，
    跳过本身会随 config.json 存档。
    """
    if (
        args.compare is None
        and not args.sanity_only
        and not args.statistical_baseline
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
)
_SWEEP_MULTI_VALUE_FLAGS = (
    "--seeds",
    "--suite-modes",
)
_SWEEP_BOOL_FLAGS = (
    "--suite-ablation",
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
) -> List[Dict[str, object]]:
    """
    构造运行计划。每个条目包含输出子目录名与逐运行覆盖的
    参数；其余设置全部继承同一条命令行。

    - 普通多种子：seed{S}
    - 模式套件：{mode}_seed{S}
    - 2×2 消融：{mode}_bal{coef}_seed{S}
      （{local_base, A} × {0, balance_coef}）
    """
    plan: List[Dict[str, object]] = []

    if ablation:
        for mode in ("local_base", "A"):
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


def run_experiment_sweep(args: argparse.Namespace) -> None:
    """
    执行多随机种子 / 模式套件 / 全因子消融套件，并输出
    受控变量检查 + Mean ± Std + 显著性检验 + 消融覆盖报告。
    """
    factors = parse_factor_spec(args.compare_factors)

    base_argv = strip_sweep_args(sys.argv[1:])
    plan = build_sweep_plan(
        seeds=list(args.seeds),
        modes=(
            list(args.suite_modes)
            if args.suite_modes
            else None
        ),
        ablation=bool(args.suite_ablation),
        balance_coef=args.balance_coef,
        default_mode=args.mode,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if len(args.seeds) < _RECOMMENDED_SEEDS:
        print(
            "[sweep] 提示：关键实验建议使用 3~5 个不同随机"
            f"种子并汇报 Mean ± Std，当前只有 "
            f"{len(args.seeds)} 个"
        )

    run_paths: List[str] = []

    for index, item in enumerate(plan, start=1):
        run_out = out_root / str(item["label"])

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
        ] + base_argv
        command += [
            "--seed",
            str(item["seed"]),
            "--out",
            str(run_out),
        ]
        if item["mode"] is not None:
            command += ["--mode", str(item["mode"])]
        if item["balance_coef"] is not None:
            command += [
                "--balance-coef",
                format(float(item["balance_coef"]), "g"),
            ]

        print(
            f"[sweep] ({index}/{len(plan)}) 运行 {item['label']}"
        )

        completed = subprocess.run(command)

        if completed.returncode != 0:
            print(
                f"[sweep] 运行 {item['label']} 失败 "
                f"(exit={completed.returncode})；已停止后续"
                "运行，可用 --compare 分析已完成的目录"
            )
            raise SystemExit(1)

        run_paths.append(str(run_out))

    rows = compare_runs(run_paths)
    analysis = print_comparison_analysis(
        rows,
        factors=factors,
        baseline_mode=args.compare_baseline,
        metric=args.compare_metric,
    )
    write_comparison_report(out_root, analysis)


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
        choices=["smelt", "local_base", "A", "B", "dense"],
        help=(
            "模型模式；dense 为无 MoE、无循环的纯稠密"
            " Transformer 基线（先跑简单基线，再谈复杂模型）"
        ),
    )
    parser.add_argument(
        "--dense-fnn-hidden",
        type=int,
        default=None,
        help=(
            "dense 基线的 FFN 隐藏宽度；默认 top_k * "
            "expert_hidden（与 MoE 单次前向激活参数量对齐"
            "的等计算口径），等参数口径可设为 n_experts * "
            "expert_hidden"
        ),
    )
    parser.add_argument(
        "--statistical-baseline",
        action="store_true",
        help=(
            "不训练神经模型：在 train 上计数、在 val 上网格"
            "调优（阶数 × 平滑系数）字节级插值 n-gram 统计"
            "基线，并用与神经模型完全相同的固定窗口协议评估；"
            "输出与训练运行同构的 config.json / metrics.jsonl"
        ),
    )
    parser.add_argument(
        "--ngram-max-order",
        type=int,
        default=5,
        help=(
            "n-gram 统计基线的最大阶数（字节精确打包上限 "
            f"为 {_NGRAM_MAX_ORDER}）"
        ),
    )
    parser.add_argument(
        "--ngram-alpha-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        help=(
            "n-gram 插值平滑系数的候选网格，全部在 val 上"
            "调优后取最优（简单基线也必须充分调优）"
        ),
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        help=(
            "本轮实验要验证的科学假设（必填）：写清楚"
            "「预期什么改动带来什么效果、依据什么指标"
            "判定成败」，带着假设做实验而不是排列组合"
            "超参数"
        ),
    )
    parser.add_argument(
        "--rq",
        type=str,
        default="",
        help=(
            "本 run 对应的研究问题标签（如 RQ1），"
            "与 --hypothesis 一起写入 config.json"
        ),
    )
    parser.add_argument(
        "--skip-hypothesis-check",
        action="store_true",
        help=(
            "显式跳过假设检查（只建议纯工程目的的临时运行"
            "使用；跳过会被记录进 config.json）"
        ),
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
    parser.add_argument(
        "--group-separator",
        type=str,
        default=None,
        help=(
            "群组数据协议：把语料按该分隔符切成「群组」"
            "（如文档/用户记录分隔符，支持 \\n、\\xHH 等转义），"
            "train/val/test 的切分位置吸附到群组边界，"
            "任何群组都不会跨 split（防止同一群组的信息"
            "同时出现在训练与验证/测试两侧）；不设置时按"
            "单一字节流在名义位置连续切分"
        ),
    )
    parser.add_argument(
        "--strict-data-audit",
        action="store_true",
        help=(
            "严格数据审计：训练前的 train/val 审计或解封测试集"
            "时的审计只要发现越界 token、跨 split 重复内容、"
            "超阈值的 UTF-8/NUL 损坏，就拒绝开始训练/拒绝"
            "解封测试集（默认只警告并照常继续）"
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
            "后退出（含受控变量检查、按种子聚合的 "
            "Mean ± Std 与单因子显著性检验）"
        ),
    )
    parser.add_argument(
        "--compare-factors",
        type=str,
        default=",".join(DEFAULT_COMPARISON_FACTORS),
        help=(
            "对比时允许跨运行不同的实验因子（逗号分隔）；"
            "其余协议项（优化器/学习率/训练步数/数据增强/"
            "数据指纹等）必须完全一致，否则报警。"
            "默认: " + ",".join(DEFAULT_COMPARISON_FACTORS)
        ),
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        metavar="MODE",
        help=(
            "以指定模式作为基线，在统计报告中给出其余"
            "实验单元相对基线的差值（如 smelt 或 local_base）"
        ),
    )
    parser.add_argument(
        "--compare-metric",
        type=str,
        default="val_loss",
        help=(
            "显著性检验使用的指标（默认 val_loss）；"
            "只允许 val 口径指标（测试集尘封协议：test "
            "指标不参与跨运行对比与选择，只在单个运行的"
            "最终评估中出现）；可选: "
            + ", ".join(COMPARISON_METRIC_KEYS)
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "多随机种子运行：每个种子训练一次到 "
            "<out>/seed<SEED>，结束后自动给出 Mean ± Std "
            "与显著性检验（关键实验建议 3~5 个种子）"
        ),
    )
    parser.add_argument(
        "--suite-modes",
        nargs="+",
        choices=sorted(Config.MODEL_MODES),
        default=None,
        metavar="MODE",
        help=(
            "配合 --seeds：把同一套受控设置依次应用到多个"
            "模式（输出 <out>/<mode>_seed<SEED>），用于"
            "严格控变量的模式对比"
        ),
    )
    parser.add_argument(
        "--suite-ablation",
        action="store_true",
        help=(
            "配合 --seeds：运行「延迟合并 × 均衡损失」的 "
            "2×2 全因子消融（local_base/A × balance_coef="
            "0/当前值）；balance 关闭臂为 0，开启臂取 "
            "--balance-coef"
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

    # 群组分隔符在解析阶段就转成字节；非法转义直接报错，
    # 而不是等到切分阶段才失败。
    group_separator_bytes: Optional[bytes] = None
    if args.group_separator is not None:
        try:
            group_separator_bytes = parse_group_separator(
                args.group_separator
            )
        except ValueError as error:
            parser.error(f"--group-separator 无效: {error}")
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
    if args.dense_fnn_hidden is not None and (
        args.dense_fnn_hidden <= 0
    ):
        parser.error("--dense-fnn-hidden 必须为正整数")
    if not 1 <= args.ngram_max_order <= _NGRAM_MAX_ORDER:
        parser.error(
            "--ngram-max-order 必须落在 "
            f"[1, {_NGRAM_MAX_ORDER}]"
        )
    for alpha in args.ngram_alpha_grid:
        if not math.isfinite(alpha) or alpha <= 0:
            parser.error(
                "--ngram-alpha-grid 的每个取值必须是"
                "有限正数（α→0 时未见的 (ctx,c) 对会得到"
                "零概率与无穷损失）"
            )
    if not args.ngram_alpha_grid:
        parser.error(
            "--ngram-alpha-grid 至少需要一个候选值"
        )

    # 对比因子与对比指标必须显式合法，避免拼写错误静默
    # 关闭受控变量检查。
    try:
        parse_factor_spec(args.compare_factors)
    except ValueError as error:
        parser.error(str(error))
    if args.compare_metric not in COMPARISON_METRIC_KEYS:
        parser.error(
            "--compare-metric 必须是以下之一: "
            + ", ".join(COMPARISON_METRIC_KEYS)
        )

    # 多种子 / 套件参数组合校验。
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds 存在重复的随机种子")
        if any(seed < 0 for seed in args.seeds):
            parser.error("--seeds 必须是非负整数")
        if args.compare is not None:
            parser.error(
                "--seeds 不能与 --compare 同时使用"
                "（对比已完成的运行请直接用 --compare）"
            )
        if args.resume is not None:
            parser.error(
                "--seeds 不能与 --resume 同时使用"
                "（每个种子都必须是独立的全新训练）"
            )
        if args.statistical_baseline:
            parser.error(
                "--statistical-baseline 是确定性统计基线，"
                "不受随机种子影响，不能与 --seeds 同时使用"
            )
        if args.sanity_only:
            parser.error(
                "--sanity-only 不能与 --seeds 同时使用"
            )
        if args.evaluate_test:
            parser.error(
                "--evaluate-test 不能与 --seeds 同时使用"
                "（多种子套件会对测试集重复解封，"
                "让测试集退化为选择指标；请在模型与"
                "超参数冻结后，改用全新目录对单个"
                "最终运行评估一次测试集）"
            )
        if args.data is None:
            parser.error("--seeds 需要 --data")
        if args.out is None:
            parser.error("--seeds 需要 --out")
    if (args.suite_modes or args.suite_ablation) and (
        args.seeds is None
    ):
        parser.error(
            "--suite-modes / --suite-ablation 必须配合 "
            "--seeds 使用（关键实验至少 3~5 个种子）"
        )
    if args.suite_modes and args.suite_ablation:
        parser.error(
            "--suite-modes 与 --suite-ablation 不能同时使用"
        )
    if args.suite_modes is not None and (
        len(set(args.suite_modes)) != len(args.suite_modes)
    ):
        parser.error("--suite-modes 存在重复的模式")
    if args.suite_ablation and not args.balance_coef > 0:
        parser.error(
            "--suite-ablation 需要正的 --balance-coef 作为"
            "开启臂（关闭臂固定为 0）"
        )
    if args.statistical_baseline and args.resume is not None:
        parser.error(
            "--statistical-baseline 不能与 --resume 同时使用"
            "（统计基线没有可恢复的训练状态，"
            "检查点会被静默忽略）"
        )

    # 假设纪律：全新训练必须带明确的科学假设，避免
    # 「盲目排列组合超参数」（检查逻辑见
    # hypothesis_check_error）。
    hypothesis_error = hypothesis_check_error(args)

    if hypothesis_error is not None:
        parser.error(hypothesis_error)

    # --compare 模式只读运行目录，不需要 --data / --out。
    if args.compare is not None:
        compare_rows = compare_runs(args.compare)
        print_comparison(compare_rows)
        print_comparison_analysis(
            compare_rows,
            factors=parse_factor_spec(args.compare_factors),
            baseline_mode=args.compare_baseline,
            metric=args.compare_metric,
        )
        return

    if args.data is None:
        parser.error("--data 必填（除非使用 --compare）")
    if args.out is None:
        parser.error("--out 必填（除非使用 --compare）")

    # 多种子 / 套件运行：父进程逐个调用自身完成每次训练，
    # 结束后自动做受控变量检查、Mean ± Std 与显著性检验。
    if args.seeds is not None:
        run_experiment_sweep(args)
        return

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

    # 测试集尘封协议不只约束单次运行，也约束输出目录：
    # 目录里已有 test 记录时，全新训练会把 metrics.jsonl
    # 截断重写，使测试集在无感知的情况下被重复解封
    # （旧记录被抹掉，之后可再次 --evaluate-test）。
    # 直接拒绝复用该目录。
    if args.resume is None and test_record_exists(
        out_dir / "metrics.jsonl"
    ):
        parser.error(
            "输出目录的 metrics.jsonl 中已有测试集评估记录："
            "全新训练会截断重写该文件，使测试集被重复解封、"
            "退化为隐式验证集。请换用全新的 --out 目录。"
        )

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
        # dense 基线的 FFN 宽度在构造 Config 时就解析成
        # 显式数值，config.json / 检查点记录的是实际生效值。
        dense_fnn_hidden = args.dense_fnn_hidden

        if (
            dense_fnn_hidden is None
            and args.mode == "dense"
        ):
            dense_fnn_hidden = (
                args.top_k * args.expert_hidden
            )
            print(
                "dense 基线 FFN 隐藏宽度 = "
                f"{dense_fnn_hidden}（等计算口径 "
                "top_k * expert_hidden；等参数口径可用 "
                "--dense-fnn-hidden 设为 n_experts * "
                "expert_hidden）"
            )

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
            dense_fnn_hidden=dense_fnn_hidden,
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
            "dense_fnn_hidden",
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

        # 群组分隔符决定吸附后的实际切分位置，与划分比例一样
        # 是数据划分协议的一部分；resume 时按存档恢复。
        saved_group_separator = saved_args.get(
            "group_separator"
        )

        if (
            saved_group_separator is not None
            and saved_group_separator != args.group_separator
        ):
            print(
                "resume restored group_separator "
                f"{args.group_separator!r} -> "
                f"{saved_group_separator!r} "
                "(群组切分是数据划分的一部分)"
            )
            args.group_separator = saved_group_separator
        elif (
            saved_group_separator is None
            and args.group_separator is not None
        ):
            print(
                "resume checkpoint predates --group-separator; "
                "当前按群组切分，train/val/test 位置可能与"
                "原运行不同"
            )

    cfg.validate()

    # resume 可能把 group_separator 恢复成存档值，
    # 统一在构造数据管线前重新解析。
    if args.group_separator is not None:
        try:
            group_separator_bytes = parse_group_separator(
                args.group_separator
            )
        except ValueError as error:
            parser.error(
                f"--group-separator 无效: {error}"
            )

    # 数据管线先于模型准备：sanity check 需要在创建全量模型之前
    # 先用同一数据管线验证模型代码能够过拟合小批量，同时拿到
    # 数据版本指纹写入实验追踪信息。
    try:
        dataset = ByteDataset(
            args.data,
            cfg.seq_len,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            device=device,
            group_separator=group_separator_bytes,
        )
    except ValueError as error:
        parser.error(f"数据管线初始化失败: {error}")

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

    # 严格数据审计：发现泄露/标签噪声时拒绝开始训练。检查放在
    # 所有入口（神经训练与统计基线）之前，两种运行都不能在
    # 已泄露的数据上产出结论。
    if args.strict_data_audit and not enforce_strict_data_audit(
        data_audit,
        "train/val",
    ):
        raise SystemExit(
            "严格数据审计未通过：请先清洗/修复数据"
            "（去重、修编码、修正群组划分），或去掉 "
            "--strict-data-audit 仅保留警告。"
        )

    # 统计基线入口：计数、调优、评估、写盘后直接结束，
    # 不进入神经模型训练流程。
    if args.statistical_baseline:
        run_statistical_baseline(
            args,
            dataset,
            out_dir,
            device,
            data_audit,
        )
        return

    # 参考基线（uniform / unigram）：与模型评估同一批固定窗口，
    # 给「accuracy 被高频字节抬高多少」提供解读坐标系，也把
    # 曲线诊断的「是否学到东西」参照从 uniform 稻草人换成
    # 更公平的 unigram。
    reference_baselines = compute_reference_baselines(
        dataset,
        args.batch_size,
        args.eval_batches,
        torch.device("cpu"),
        split="val",
    )

    print(
        "[reference] uniform val_loss="
        f"{reference_baselines['uniform_loss']:.4f} "
        "(= ln 256), unigram val_loss="
        f"{reference_baselines['unigram_loss']:.4f}, "
        "unigram token_accuracy="
        f"{reference_baselines['unigram_token_accuracy']:.4f}"
        "（类别不平衡下解读 accuracy 的参考下界）"
    )

    if args.hypothesis.strip():
        print(
            f"[hypothesis] {args.hypothesis.strip()}"
            + (
                f" (rq={args.rq.strip()})"
                if args.rq.strip()
                else ""
            )
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
            data_audit=data_audit,
            reference_baselines=reference_baselines,
        )
        if args.resume is None:
            # 全新运行：参考基线记录是 metrics.jsonl 的首行；
            # resume 时保留原运行记录，不重复追加。
            write_jsonl_record(
                out_dir / "metrics.jsonl",
                {
                    "step": "reference_baselines",
                    **reference_baselines,
                },
                append=False,
            )
        write_jsonl_record(
            out_dir / "metrics.jsonl",
            sanity_record,
            append=True,
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
    # 数据指纹与划分、数据审计、代码版本/运行环境、sanity check
    # 结果、科学假设与参考基线。
    write_run_config(
        out_dir,
        cfg,
        args,
        n_params,
        collect_run_metadata(device),
        config_sanity_result,
        data_audit=data_audit,
        reference_baselines=reference_baselines,
    )

    metrics_path = out_dir / "metrics.jsonl"

    # 验证集模型选择 + 早停状态：resume 时从原运行的
    # metrics.jsonl 恢复「最优验证损失 / 最优步 / 连续未改善
    # 次数」，保证早停判定与模型选择跨恢复一致；全新训练不读
    # 旧文件（文件会被截断重写）。
    best_val_loss: Optional[float] = None
    best_val_step: Optional[int] = None
    evals_without_improvement = 0

    if args.resume is not None:
        (
            best_val_loss,
            best_val_step,
            evals_without_improvement,
        ) = early_stop_state_from_records(
            read_metrics_records(metrics_path),
            args.early_stop_min_delta,
        )

    # 训练期间每个刷新最优验证损失的评估点覆盖写入 best.pt；
    # 训练结束后用它做 final / test 评估（test 只在这一份
    # 经验证集选择的模型上解封一次）。
    selected_model_path = out_dir / _SELECTED_MODEL_FILENAME

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

    if args.resume is None:
        # 全新训练的首行：参考基线（uniform / unigram），
        # 为后续所有指标提供解读坐标系。
        metrics_file.write(
            json.dumps(
                _json_safe({
                    "step": "reference_baselines",
                    **reference_baselines,
                }),
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()

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
        # 后续更新全部会被跳过，继续训练没有意义。
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

            records = flush_records()
            print(records[-1])
            metrics_file.flush()
            model.train()

            # 验证集模型选择：无论是否启用早停，都记录最优验证
            # 点并把权重写入 best.pt；训练结束后的 final / test
            # 评估使用该模型（验证集用于选型，测试集只在选定的
            # 模型上解封一次）。NaN/Inf 验证损失不作数，避免故障
            # 顺带触发早停或污染选型。
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
                save_selected_model(
                    str(selected_model_path),
                    model,
                    step + 1,
                    best_val_loss,
                    args,
                )
            else:
                evals_without_improvement += 1

            # 早停：验证损失连续无改善达到 patience 时提前结束。
            if (
                args.early_stop_patience > 0
                and evals_without_improvement
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

    # 按验证集选定模型：final / test 评估之前，把权重恢复为
    # best.pt（验证损失最优的评估点）。这样「最终模型」由
    # 验证集决定，而不是由最后一步恰好停在哪里决定；测试集
    # 也只在这份模型上解封一次。
    selection_policy = "last"
    selection_note: Optional[str] = None

    if best_val_step is not None:
        selected_payload = load_selected_model(
            str(selected_model_path),
            torch.device("cpu"),
        )

        if (
            selected_payload is not None
            and int(selected_payload.get("step", -1)) == best_val_step
        ):
            restore_selected_weights(model, selected_payload)
            selection_policy = "best_val"
            print(
                "[select] 使用验证集选中的模型做最终评估："
                f"step={best_val_step}, "
                f"val_loss={float(best_val_loss):.4f}"
            )
        else:
            selection_note = (
                "best.pt 缺失或与验证记录不一致"
                "（如从旧检查点恢复）；最终评估回退到"
                "最后一步参数，模型未经验证集选择"
            )
            print(f"[select] {selection_note}")

    # 最终评估总是重新跑一遍：与周期验证不同，它还要计算
    # one-vs-rest 的 macro PR-AUC / ROC-AUC（周期验证只算
    # 混淆矩阵口径的轻量指标），不能直接复用上次周期结果。
    final_metrics = evaluate(
        model,
        dataset,
        args.batch_size,
        args.eval_batches,
        device,
        amp_dtype=amp_dtype,
        compute_auc=True,
    )

    final_record = {
        "step": "final",
        **final_metrics,
        "early_stopped": early_stopped,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "selected_model": selection_policy,
        "selected_step": (
            best_val_step
            if selection_policy == "best_val"
            else end_step
        ),
        "selected_val_loss": (
            best_val_loss
            if selection_policy == "best_val"
            else None
        ),
        "selection_note": selection_note,
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

    # 选型协议写入 config.json：最终模型由验证集选定（best_val）
    # 或回退到最后一步（last），以及是否用于测试集评估。
    update_run_config(
        out_dir,
        "model_selection",
        {
            "policy": selection_policy,
            "selected_step": (
                best_val_step
                if selection_policy == "best_val"
                else end_step
            ),
            "selected_val_loss": (
                best_val_loss
                if selection_policy == "best_val"
                else None
            ),
            "note": selection_note,
            "evaluate_test_requested": bool(
                args.evaluate_test
            ),
        },
    )

    # --------------------------------------------------------
    # 结果分析 1：训练曲线诊断（不能只看最终单个数值）。
    # 「是否学到东西」的参照从 uniform（ln 256，稻草人）
    # 升级为 train 拟合的 unigram：连字节边际分布都没打赢
    # 才算欠拟合，避免「只比随机猜强」的虚假健康曲线。
    # --------------------------------------------------------
    curve_baseline_loss = min(
        math.log(cfg.vocab_size),
        float(reference_baselines["unigram_loss"]),
    )

    curve_analysis = analyze_training_curve(
        read_metrics_records(metrics_path),
        baseline_loss=curve_baseline_loss,
    )
    metrics_file.write(
        json.dumps(
            _json_safe({
                "step": "curve_analysis",
                "reference_baseline_loss": curve_baseline_loss,
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
    if args.evaluate_test and numerical_polluted:
        # 参数已被 NaN/Inf 污染时模型输出不可信，评估测试集只会
        # 白白消耗该目录唯一一次解封名额；保持尘封，修复后从
        # 污染前的检查点恢复或换全新目录再评估。
        print(
            "[test] 参数/优化器状态已被 NaN/Inf 污染，"
            "本次不消耗唯一的测试集评估名额：测试集保持尘封。"
            "请从污染前的检查点恢复并修复数值问题后，"
            "再显式 --evaluate-test。"
        )
    elif args.evaluate_test:
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

            if args.strict_data_audit and not enforce_strict_data_audit(
                test_audit,
                "test",
            ):
                # 在已泄露/损坏的测试集上报告指标毫无意义，还会
                # 留下唯一的 test 记录；直接拒绝解封。
                print(
                    "[test] 严格数据审计在测试集上发现泄露/"
                    "损坏问题：为避免在污染数据上产出最终结论，"
                    "本次拒绝解封测试集（不写入 test 记录，"
                    "名额未消耗）。请先清洗数据后重跑。"
                )
            else:
                test_metrics = evaluate(
                    model,
                    dataset,
                    args.batch_size,
                    args.eval_batches,
                    device,
                    amp_dtype=amp_dtype,
                    split="test",
                    compute_auc=True,
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
