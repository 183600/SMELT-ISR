from __future__ import annotations


import argparse

import codecs

from collections import OrderedDict, deque

import contextlib

import datetime

import functools

import hashlib

import heapq

import itertools

import json

import math

import os

import platform

import random

import re

import shlex

import subprocess

import sys

import tempfile

import time

import weakref

from dataclasses import asdict, dataclass, fields, replace

from pathlib import Path, PurePosixPath

from typing import Dict, List, Optional, Sequence, Tuple


import torch

import torch.nn as nn

import torch.nn.functional as F


try:
    import numpy as np
except ImportError:
    # NumPy 不是必需依赖；安装时一并固定其随机种子。
    np = None



# 分配器与 cuBLAS 确定性工作区配置必须在任何 CUDA 上下文创建
# 之前生效。放在模块导入时设置，训练入口与自测入口都覆盖到。
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


# 进程启动时的 PYTHONHASHSEED（在 set_seed 里被 setdefault 之前
# 捕获）：当前解释器的哈希种子在启动时已经确定，set_seed 事后
# 设置的环境变量只影响之后启动的子进程。此处的捕获值随运行
# 元数据存档，值为 None 说明本进程的 hash 随机化未被固定，
# 严格逐位复现应在启动命令行显式设置该变量。
_PYTHONHASHSEED_AT_START = os.environ.get("PYTHONHASHSEED")


# set_seed 里的一次性提示标志：避免每个进程重复打印。
_PYTHONHASHSEED_WARNED = False


# 最近一次 set_seed 实际播种的随机源与种子。写入
# run_metadata，供实验追踪核对「参数里的 --seed 是否真的
# 作用到了每个后端」，而不只是相信调用点。
_LAST_SEED_MANIFEST: Optional[Dict[str, object]] = None



# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    vocab_size: int = 256
    seq_len: int = 128

    d_model: int = 128
    n_heads: int = 4
    # GQA / KV 头数：None 等价于 n_heads（标准 MHA）。SMELT 论文
    # 使用 grouped-query attention 以匹配 KV cache；n_kv_heads <
    # n_heads 时每个 KV 头被 n_heads // n_kv_heads 个 Q 头共享。
    n_kv_heads: Optional[int] = None
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

    # 残差缩放口径：
    #   paper  —— 循环区域内的子层残差更新额外除以循环次数 r
    #             （SMELT 论文 3.1 节；本基座 r=2，即 1/2），
    #             循环区域外的子层仍只用 residual_scale；
    #   global —— 历史行为：residual_scale 全局作用于所有子层。
    # 旧运行 / global 口径在 comparison_protocol 中可区分。
    residual_scaling: str = "paper"

    # 论文 3.1 节：训练序列打包进 4096-token 的 packed context，
    # 并用文档级 segment attention mask 限制注意力只在同一篇
    # 文档内（跨文档的 key 被屏蔽）。document_separator 是文档
    # 边界字节串（通常与 --group-separator 相同），以字节列表
    # 形式存档，保证 config.json 可 JSON 序列化。
    document_masks: bool = False
    document_separator: Optional[List[int]] = None

    # 初始化方案。module_order 是历史行为：按模块构造顺序消耗
    # 全局 RNG，不同 mode 的模块数量/构造顺序不同，同一种子下
    # 共享层会得到不同初始权重。name_hash 按参数名与运行种子
    # 派生独立随机流，使共享参数在不同 mode 间逐位一致，避免
    # 把初始化差异混进 A/B/local_base 的配对比较。
    init_scheme: str = "module_order"

    # LanguageModel 可以构建的模式；ngram 仅供统计基线运行写
    # config.json 使用。
    #   base           —— 论文的未循环 MoE 基线（middle 只执行一次）；
    #   smelt          —— 论文方法（middle 循环两次，循环区域残差 1/r）；
    #   local_base     —— 立即合并 + 第二轮宽度 min(E,k1*k2) 的等专家
    #                     计算量对照（性能对照）；
    #   local_base_k2  —— 立即合并 + 第二轮宽度 k2 的机制对照
    #                     （只改变合并时机，不改变第二轮宽度）；
    #   A              —— 局部延迟合并；
    #   B              —— 全局延迟合并；
    #   B_early        —— B 的等计算量机制对照：分支数与 Attention/MoE
    #                     执行次数不变，只在第一轮结束后提前合并并
    #                     重新展开为 k1 个分支（只改变合并位置）；
    #   dense          —— 无 MoE、无循环的纯稠密 Transformer 基线。
    MODEL_MODES = {
        "base",
        "smelt",
        "local_base",
        "local_base_k2",
        "A",
        "B",
        "B_early",
        "dense",
    }
    ALL_MODES = MODEL_MODES | {"ngram"}
    INIT_SCHEMES = {"module_order", "name_hash"}
    RESIDUAL_SCALINGS = {"paper", "global"}

    @property
    def effective_n_kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def loop_count(self) -> int:
        """中间区域每层在完整前向中执行的次数。"""
        return 1 if self.mode in {"base", "dense"} else 2

    @property
    def loop_residual_scale(self) -> float:
        """循环区域内子层残差更新的系数。

        paper 口径按 SMELT 论文 3.1 节对循环区域内的每个子层
        残差更新除以循环次数 r；global 口径保留历史行为，即
        与循环区域外使用同一个 residual_scale。
        """
        if self.residual_scaling == "paper":
            return self.residual_scale / float(self.loop_count)
        return self.residual_scale


    @property
    def effective_dense_fnn_hidden(self) -> int:
        if self.dense_fnn_hidden is not None:
            return self.dense_fnn_hidden
        # 等计算口径：top_k 个专家的激活参数量。
        return self.top_k * self.expert_hidden

    def validate(self):
        if self.mode not in self.ALL_MODES:
            raise ValueError(f"未知 mode: {self.mode}")

        def _positive_int(name: str, value: object) -> None:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} 必须为正整数")

        for _name, _value in (
            ("vocab_size", self.vocab_size),
            ("seq_len", self.seq_len),
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("n_layers", self.n_layers),
            ("n_experts", self.n_experts),
            ("expert_hidden", self.expert_hidden),
            ("top_k", self.top_k),
            ("k1", self.k1),
            ("k2", self.k2),
        ):
            _positive_int(_name, _value)
        _positive_int(
            "n_kv_heads",
            self.effective_n_kv_heads,
        )
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if self.effective_n_kv_heads > self.n_heads:
            raise ValueError(
                "n_kv_heads 必须小于等于 n_heads"
            )
        # GQA 不要求 n_heads 能被 n_kv_heads 整除：整除时按
        # n_heads // n_kv_heads 个连续 Q 头共享一个 KV 头
        # （repeat_interleave，与历史行为逐位一致）；不整除时
        # 按 Q 头 i 使用 KV 头 floor(i * n_kv / n_heads) 做
        # 均匀映射（见 CausalSelfAttention.core）。论文没有披露
        # head 几何，预算匹配需要在任意 GQA 比例下精确控制
        # KV cache，因此这里不能强制整除。
        if self.n_layers < 4:
            raise ValueError("n_layers 必须大于等于 4")
        # 中间区域由 middle_start=n//4、middle_end=3n//4 定义，
        # 对 L=10/12/20/30 等论文深度都能取到约 50% 的循环跨度；
        # 不再强制 L 为 4 的倍数（论文的 L=10/20/30 就不是）。
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
        if self.residual_scaling not in self.RESIDUAL_SCALINGS:
            raise ValueError(
                "residual_scaling 必须是 "
                + " / ".join(sorted(self.RESIDUAL_SCALINGS))
            )
        if self.document_masks:
            if not self.document_separator:
                raise ValueError(
                    "document_masks 需要非空的 document_separator"
                    "（文档边界字节串），否则无法构造 segment mask"
                )
            if any(
                not isinstance(byte, int)
                or not 0 <= byte <= 255
                for byte in self.document_separator
            ):
                raise ValueError(
                    "document_separator 必须是 0..255 的字节列表"
                )
        elif self.document_separator:
            raise ValueError(
                "提供了 document_separator 但未启用 document_masks；"
                "请同时设置 document_masks=True 或清空分隔符"
            )
        if self.dense_fnn_hidden is not None and (
            isinstance(self.dense_fnn_hidden, bool)
            or not isinstance(self.dense_fnn_hidden, int)
            or self.dense_fnn_hidden <= 0
        ):
            raise ValueError(
                "dense_fnn_hidden 必须是正整数或 None"
            )
        if self.mode == "dense":
            hidden = self.effective_dense_fnn_hidden
            if hidden <= 0:
                raise ValueError(
                    "dense 模式的 FFN 隐藏宽度必须为正整数"
                    f"（当前解析为 {hidden}）"
                )
        if self.init_scheme not in self.INIT_SCHEMES:
            raise ValueError(
                "init_scheme 必须是 "
                + " / ".join(sorted(self.INIT_SCHEMES))
            )

    @property
    def middle_start(self) -> int:
        return self.n_layers // 4

    @property
    def middle_end(self) -> int:
        return 3 * self.n_layers // 4



# ============================================================
# Cost model / budget matching
# ============================================================

def mode_cost_profile(cfg: Config) -> Dict[str, object]:
    """
    按完整计算图给出每 token 的执行次数与解析 FLOPs。

    只数路由宽度会漏掉第一阶段成本：A 的每 token 专家
    执行次数是 ``k1 + k1*k2``，而不是 ``k1*k2``；B 还要计入
    分支创建位置、第一轮其余层按分支执行的 Attention/MoE，
    以及分支化 Attention 的开销。该函数把四类成本分开：

    - ``attention_executions``：每 token 经过 Attention 子层的
      次数（B 的分支 Attention 按 k1 倍计）；
    - ``moe_passes``：每 token 经过 Router/MoE 子层的次数；
    - ``expert_assignments``：每 token 的专家执行次数（k 个
      专家算 k 次）；
    - ``router_calls``：每 token 的路由打分次数。

    ``analytic_flops_per_token`` 是统一口径的解析下界
    （矩阵乘 2·m·n·k，Attention 矩阵 4·T·D），用于跨模式
    预算匹配；实测 FLOPs 仍以 FlopCounter 为准。两者不会
    假装相等：报告同时给出 ``measurement`` 与实际测量的
    比例（见 ``measure_efficiency``）。
    """
    layers = int(cfg.n_layers)
    middle = int(cfg.middle_end - cfg.middle_start)
    pre = int(cfg.middle_start)
    post = int(layers - cfg.middle_end)
    dim = int(cfg.d_model)
    head_dim = dim // int(cfg.n_heads)
    kv_dim = int(cfg.effective_n_kv_heads) * head_dim
    top_k = int(cfg.top_k)
    k1 = int(cfg.k1)
    k2 = int(cfg.k2)
    experts = int(cfg.n_experts)
    expert_hidden = int(cfg.expert_hidden)
    seq_len = int(cfg.seq_len)

    mode = cfg.mode
    branch_factor = 1
    if mode == "dense":
        attention_executions = layers
        moe_passes = 0
        expert_assignments = 0
        router_calls = 0
    elif mode == "base":
        attention_executions = layers
        moe_passes = layers
        expert_assignments = top_k * layers
        router_calls = layers
    elif mode == "smelt":
        attention_executions = pre + 2 * middle + post
        moe_passes = pre + 2 * middle + post
        expert_assignments = top_k * (pre + 2 * middle + post)
        router_calls = moe_passes
    elif mode == "B":
        branch_factor = k1
        branch_exec = 1 + (middle - 1) * k1 + middle * k1
        attention_executions = pre + branch_exec + post
        moe_passes = pre + branch_exec + post
        # 第一轮第一个 Block 由 k1 个专家创建 k1 个分支；之后
        # 第一轮剩余 m-1 个 Block 的每个分支都独立路由 k1 个
        # 专家（分支内 top-k 合并），第二轮 m 个 Block 的每个
        # 分支各路由 k2 个专家。
        expert_assignments = (
            top_k * (pre + post)
            + k1
            + (middle - 1) * k1 * k1
            + middle * k1 * k2
        )
        router_calls = moe_passes
    elif mode == "B_early":
        # 与 B 完全相同的分支数、Attention/MoE 执行次数；只有
        # 跨分支合并位置提前到第一轮结束。
        branch_factor = k1
        branch_exec = 1 + (middle - 1) * k1 + middle * k1
        attention_executions = pre + branch_exec + post
        moe_passes = pre + branch_exec + post
        expert_assignments = (
            top_k * (pre + post)
            + k1
            + (middle - 1) * k1 * k1
            + middle * k1 * k2
        )
        router_calls = moe_passes
    else:
        # A / local_base / local_base_k2：Attention 由所有分支
        # 共享（每层一次），MoE 两阶段。
        attention_executions = layers
        if mode == "A":
            # A 的第二轮在 k1 个分支上各自路由，因此每个中间
            # Block 的 Router/MoE 执行 1 + k1 次（local_base*
            # 第二轮已合并为单分支，仍是 2 次）。
            moe_passes = (
                pre + post + middle * (1 + k1)
            )
            router_calls = moe_passes
        else:
            moe_passes = layers + middle
            router_calls = moe_passes
        if mode == "A":
            expert_assignments = (
                top_k * (pre + post) + middle * (k1 + k1 * k2)
            )
        elif mode == "local_base_k2":
            expert_assignments = (
                top_k * (pre + post) + middle * (k1 + k2)
            )
        else:
            expert_assignments = (
                top_k * (pre + post)
                + middle * (k1 + min(experts, k1 * k2))
            )

    # 每 token 经过一次 Attention 子层的解析 FLOPs。
    # Q/K/V/out 投影：Q 为 D×D，K/V 为 D×kv_dim，out 为 D×D；
    # 乘加计 2 次。Attention 矩阵（因果）按 4·T·D 计，与
    # _count_forward_flops 的估计口径一致。
    attention_flops = (
        2 * dim * dim
        + 2 * 2 * dim * kv_dim
        + 2 * dim * dim
        + 4 * seq_len * dim
    )
    # 每个 MoE/Router pass 的路由打分。
    router_flops = 2 * dim * experts
    # 每个专家 assignment：gate_up 为 D×(2·H)，down 为 H×D。
    expert_flops = 2 * dim * (2 * expert_hidden) + 2 * expert_hidden * dim
    # dense 模式的稠密 SwiGLU FFN 与 Expert 结构相同，但每层
    # 都执行且没有路由开销；漏掉这一项会把 dense 基线低估为
    # 「只有 Attention」，预算匹配与解析/实测 FLOPs 对照都会失真。
    dense_ffn_flops = (
        layers
        * (
            2 * dim * (2 * cfg.effective_dense_fnn_hidden)
            + 2 * cfg.effective_dense_fnn_hidden * dim
        )
        if mode == "dense"
        else 0
    )
    analytic_flops = (
        attention_executions * attention_flops
        + router_calls * router_flops
        + expert_assignments * expert_flops
        + dense_ffn_flops
    )
    # KV cache 每 token 字节数（fp16 存储 K 和 V）：每个物理层
    # 为 K、V 各存一个 kv_dim 向量，fp16 每元素 2 字节。
    #
    # 分支只存在于 middle 区域（B/B_early 的第一轮第一个 Block
    # 在 Attention 之后才创建分支，其余 middle Block 与第二轮按
    # 分支执行）；pre/post 始终只有一份 KV。因此物理容量按
    # ``pre + post + middle * branch_factor`` 计，不能把分支数
    # 乘到全部层上。
    #
    # per-visit 口径是「每个 token 实际执行多少次 Attention」，
    # 即 attention_executions：smelt 只有 middle 访问两次，
    # pre/post 各一次；B/B_early 的分支 Attention 也逐次计入。
    # 直接拿物理层数乘 loop_count 会把 pre/post 也算成两次访问，
    # 高估容量。
    kv_bytes_per_layer = 2 * 2 * kv_dim  # K+V × fp16 bytes
    kv_cache_layers = (
        pre + post + middle * branch_factor
        if mode in ("B", "B_early")
        else layers
    )
    kv_bytes_physical = int(
        kv_cache_layers * kv_bytes_per_layer
    )
    kv_bytes_per_visit = int(
        attention_executions * kv_bytes_per_layer
    )
    return {
        "mode": mode,
        "n_layers": layers,
        "middle_layers": middle,
        "loop_count": int(cfg.loop_count),
        "branch_factor": int(branch_factor),
        "attention_executions_per_token": int(attention_executions),
        "moe_passes_per_token": int(moe_passes),
        "router_calls_per_token": int(router_calls),
        "expert_assignments_per_token": int(expert_assignments),
        "analytic_flops_per_token": int(analytic_flops),
        "analytic_flops_breakdown": {
            "attention": int(attention_executions * attention_flops),
            "router": int(router_calls * router_flops),
            "experts": int(expert_assignments * expert_flops),
            "dense_ffn": int(dense_ffn_flops),
        },
        "analytic_formula": {
            "expert_assignments": {
                "smelt": "top_k * (pre + 2*middle + post)",
                "A": "top_k*(pre+post) + middle*(k1 + k1*k2)",
                "local_base": (
                    "top_k*(pre+post) + middle*(k1 + "
                    "min(n_experts, k1*k2))"
                ),
                "local_base_k2": (
                    "top_k*(pre+post) + middle*(k1 + k2)"
                ),
                "B": (
                    "top_k*(pre+post) + k1 + "
                    "(middle-1)*k1*k1 + middle*k1*k2"
                ),
                "B_early": (
                    "top_k*(pre+post) + k1 + "
                    "(middle-1)*k1*k1 + middle*k1*k2 "
                    "(与 B 相同)"
                ),
                "base": "top_k * n_layers",
                "dense": "0（无 MoE）",
            }.get(mode, "n/a"),
            "attention_executions": {
                "smelt": "pre + 2*middle + post",
                "A": "n_layers (Attention 跨分支共享)",
                "B": (
                    "pre + [1 + (middle-1)*k1 + middle*k1] + post"
                ),
                "B_early": (
                    "pre + [1 + (middle-1)*k1 + middle*k1] + post"
                ),
                "base": "n_layers",
                "dense": "n_layers",
            }.get(
                mode,
                "n_layers (Attention 跨分支共享)",
            ),
            "dense_ffn": (
                "n_layers * (4*D*H_dense + 2*D*H_dense)，"
                "H_dense=dense_fnn_hidden 或 top_k*expert_hidden"
                if mode == "dense"
                else "0（无稠密 FFN）"
            ),
            "note": (
                "只匹配 k1*k2≈2*top_k 会漏掉第一阶段成本；"
                "请用 --budget-match 对完整计算图核对"
            ),
        },
        "kv_cache_bytes_per_token": kv_bytes_physical,
        "kv_cache_bytes_per_token_per_visit": (
            kv_bytes_per_visit
        ),
        "kv_cache_physical_layers": int(kv_cache_layers),
        "kv_cache_note": (
            "按物理层各缓存一份 K/V 的 fp16 估计；B/B_early "
            "只在 middle 区域按分支数乘 k1，pre/post 保持一份。"
            "per_visit 口径为每个 token 实际执行的 Attention "
            "次数（含循环与分支），不是物理层数乘循环次数。"
            "增量解码尚未实现，这里只是容量估计/口径说明"
        ),
    }



def _budget_flops_ratio(
    cfg: Config,
    baseline_mode: str,
    k1: Optional[int] = None,
    k2: Optional[int] = None,
) -> float:
    """候选配置相对同维度参考模式的解析 FLOPs 比例。"""
    candidate = replace(
        cfg,
        mode=cfg.mode,
        k1=cfg.k1 if k1 is None else int(k1),
        k2=cfg.k2 if k2 is None else int(k2),
    )
    baseline = replace(cfg, mode=baseline_mode)
    baseline_flops = int(
        mode_cost_profile(baseline)["analytic_flops_per_token"]
    )
    candidate_flops = int(
        mode_cost_profile(candidate)["analytic_flops_per_token"]
    )
    if baseline_flops <= 0:
        return float("inf")
    return candidate_flops / baseline_flops



def budget_match_report(
    cfg: Config,
    baseline_mode: str,
    tolerance: float,
) -> Dict[str, object]:
    """
    按完整计算图核对候选模式与参考模式的每 token FLOPs。

    返回比例、允许误差、是否匹配，并为 A/local_base*/
    local_base_k2 搜索最接近预算的整数 k2（固定其它结构）；
    为 B/B_early 搜索 (k1, k2) 网格中的最接近组合。搜索只是
    建议，不会静默改配置。
    """
    profile = mode_cost_profile(cfg)
    baseline_profile = mode_cost_profile(
        replace(cfg, mode=baseline_mode)
    )
    ratio = (
        float(profile["analytic_flops_per_token"])
        / float(baseline_profile["analytic_flops_per_token"])
        if int(baseline_profile["analytic_flops_per_token"]) > 0
        else float("inf")
    )
    report: Dict[str, object] = {
        "baseline_mode": baseline_mode,
        "candidate_mode": cfg.mode,
        "candidate_flops_per_token": profile[
            "analytic_flops_per_token"
        ],
        "baseline_flops_per_token": baseline_profile[
            "analytic_flops_per_token"
        ],
        "flops_ratio": ratio,
        "tolerance": tolerance,
        "matched": abs(ratio - 1.0) <= tolerance,
        "matched_costs": {
            "candidate": profile,
            "baseline": baseline_profile,
        },
        "search": None,
    }

    if cfg.mode in {
        "A",
        "local_base",
        "local_base_k2",
    }:
        best = None
        for candidate_k2 in range(1, cfg.n_experts + 1):
            candidate_ratio = _budget_flops_ratio(
                cfg,
                baseline_mode,
                k2=candidate_k2,
            )
            score = abs(candidate_ratio - 1.0)
            if best is None or score < best[0]:
                best = (score, candidate_k2, candidate_ratio)
        if best is not None:
            report["search"] = {
                "kind": "k2",
                "recommended_k2": int(best[1]),
                "flops_ratio_at_recommendation": best[2],
                "grid": "1..n_experts",
            }
    elif cfg.mode in {"B", "B_early"}:
        best = None
        for candidate_k1 in range(1, cfg.n_experts + 1):
            for candidate_k2 in range(1, cfg.n_experts + 1):
                candidate_ratio = _budget_flops_ratio(
                    cfg,
                    baseline_mode,
                    k1=candidate_k1,
                    k2=candidate_k2,
                )
                score = abs(candidate_ratio - 1.0)
                if best is None or score < best[0]:
                    best = (
                        score,
                        candidate_k1,
                        candidate_k2,
                        candidate_ratio,
                    )
        if best is not None:
            report["search"] = {
                "kind": "k1_k2",
                "recommended_k1": int(best[1]),
                "recommended_k2": int(best[2]),
                "flops_ratio_at_recommendation": best[3],
                "grid": "1..n_experts x 1..n_experts",
            }
    return report



# ============================================================
# 论文复现口径：arXiv:2609.01343v1
# ============================================================

PAPER_ID = "arXiv:2609.01343v1"

PAPER_URL = "https://arxiv.org/html/2609.01343v1"


# 论文 3.1 / 3.2 / 4.1 节明确给出的训练协议量。这里只收录论文明
# 文写出的数字；学习率、warmup、weight decay、betas、均衡损失系数、
# 初始化、head 几何、专家中间维与数据语料论文均未披露，单列在
# PAPER_UNDISCLOSED 中，不伪装成已复现。
PAPER_PROTOCOL: Dict[str, object] = {
    "optimizer": "AdamW",
    "lr_schedule": "wsd",
    "context_length": 4096,
    "packing": "document_packed_with_segment_attention_masks",
    "global_batch_sequences": 256,
    "tokens_per_step": 256 * 4096,
    "stable_steps": 196075,
    "stable_tokens": 196075 * 256 * 4096,
    "decay_tokens": 10_000_000_000,
    "decay_steps": math.ceil(10_000_000_000 / (256 * 4096)),
    "branch_steps": (10000, 20000, 50000, 100000, 150000, 196075),
    "validation_sources": 39,
}


PAPER_UNDISCLOSED = (
    "learning_rate",
    "warmup_steps",
    "weight_decay",
    "adam_betas",
    "balance_loss_coef",
    "init_scheme",
    "attention_head_dim",
    "n_kv_heads",
    "expert_intermediate_dim",
    "pretraining_corpus",
)



@dataclass(frozen=True)
class PaperCell:
    """论文附录 A 的一个网格单元（基线或 SMELT）。"""

    arch: str
    scale: str
    sparsity: int
    hidden: int
    layers: int
    total_experts: int
    active_experts: int
    active_params: float
    total_params: float
    measured_sparsity: float
    flops_ratio: Optional[float] = None
    kv_ratio: Optional[float] = None



# 附录 A：Baseline 表。字段含义见 table：
# (scale, sparsity label, H, L, total experts, active experts,
#  active params [B], total params [B], measured S)
_PAPER_BASE_ROWS = (
    ("100M", 0, 672, 10, 8, 8, 0.100, 0.10, 0.000),
    ("100M", 85, 672, 10, 64, 8, 0.100, 0.71, 0.860),
    ("100M", 95, 672, 10, 192, 8, 0.101, 2.12, 0.952),
    ("100M", 97, 672, 10, 336, 8, 0.102, 3.70, 0.972),
    ("200M", 0, 1280, 12, 8, 8, 0.207, 0.21, 0.000),
    ("200M", 85, 1280, 12, 64, 8, 0.207, 1.32, 0.843),
    ("200M", 95, 1280, 12, 192, 8, 0.209, 3.87, 0.946),
    ("200M", 97, 1280, 12, 336, 8, 0.212, 6.74, 0.969),
    ("600M", 0, 1792, 20, 8, 8, 0.663, 0.66, 0.000),
    ("600M", 85, 1792, 20, 64, 8, 0.665, 4.28, 0.845),
    ("600M", 95, 1792, 20, 192, 8, 0.670, 12.54, 0.947),
    ("600M", 97, 1792, 20, 336, 8, 0.675, 21.83, 0.969),
    ("1.6B", 0, 2304, 30, 8, 8, 1.629, 1.63, 0.000),
    ("1.6B", 85, 2304, 30, 64, 8, 1.632, 10.55, 0.845),
    ("1.6B", 95, 2304, 30, 192, 8, 1.641, 30.94, 0.947),
    ("1.6B", 97, 2304, 30, 336, 8, 1.651, 53.89, 0.969),
)


# 附录 A：SMELT 表，额外给出 F_SMELT/F_Base 与 KV_SMELT/KV_Base
# 两列（论文报告的匹配后残余比例）。
# (scale, sparsity label, H, L, total experts, active experts,
#  active params [B], total params [B], measured S,
#  F ratio, KV ratio)
_PAPER_SMELT_ROWS = (
    ("100M", 0, 576, 10, 16, 8, 0.071, 0.10, 0.000, 1.081, 1.018),
    ("100M", 85, 576, 10, 96, 8, 0.067, 0.71, 0.868, 1.024, 0.964),
    ("100M", 95, 576, 10, 288, 8, 0.069, 2.12, 0.955, 1.031, 0.964),
    ("100M", 97, 576, 10, 504, 8, 0.070, 3.70, 0.974, 1.039, 0.964),
    ("200M", 0, 1056, 12, 16, 8, 0.142, 0.21, 0.000, 1.039, 1.031),
    ("200M", 85, 1056, 12, 96, 8, 0.140, 1.32, 0.846, 1.021, 1.031),
    ("200M", 95, 1056, 12, 288, 8, 0.142, 3.89, 0.947, 1.029, 1.031),
    ("200M", 97, 1056, 12, 504, 8, 0.145, 6.78, 0.969, 1.037, 1.031),
    ("600M", 0, 1408, 20, 16, 8, 0.458, 0.67, 0.000, 1.046, 1.031),
    ("600M", 85, 1408, 20, 96, 8, 0.443, 4.31, 0.851, 1.007, 1.031),
    ("600M", 95, 1512, 20, 288, 8, 0.453, 12.44, 0.947, 1.024, 0.984),
    ("600M", 97, 1512, 20, 504, 8, 0.459, 21.70, 0.969, 1.031, 0.984),
    ("1.6B", 0, 1848, 30, 16, 8, 1.101, 1.64, 0.000, 1.022, 0.984),
    ("1.6B", 85, 1848, 30, 96, 8, 1.081, 10.45, 0.846, 1.000, 0.984),
    ("1.6B", 95, 1848, 30, 288, 8, 1.091, 30.90, 0.948, 1.004, 0.984),
    ("1.6B", 97, 1848, 30, 504, 8, 1.103, 53.90, 0.970, 1.008, 0.984),
)


PAPER_SCALES = ("100M", "200M", "600M", "1.6B")

PAPER_SPARSITIES = (0, 85, 95, 97)

PAPER_ARCHS = ("base", "smelt")


PAPER_CELLS: Dict[Tuple[str, str, int], PaperCell] = {}

for _row in _PAPER_BASE_ROWS:
    _cell = PaperCell("base", *_row)
    PAPER_CELLS[(_cell.arch, _cell.scale, _cell.sparsity)] = _cell

for _row in _PAPER_SMELT_ROWS:
    _cell = PaperCell("smelt", *_row)
    PAPER_CELLS[(_cell.arch, _cell.scale, _cell.sparsity)] = _cell

del _row, _cell



def _divisors(value: int) -> List[int]:
    return [d for d in range(1, value + 1) if value % d == 0]



def _normalize_scale(scale: str) -> str:
    text = str(scale).strip()
    if text.lower().endswith("b") or text.lower().endswith("m"):
        text = text[:-1] + text[-1].upper()
    for candidate in PAPER_SCALES:
        if candidate.lower() == text.lower():
            return candidate
    raise ValueError(
        f"未知的论文规模 {scale!r}；可选："
        + ", ".join(PAPER_SCALES)
    )



def _normalize_sparsity(sparsity: object) -> int:
    text = str(sparsity).strip().rstrip("%")
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(
            f"未知的稀疏度 {sparsity!r}；可选：0, 85, 95, 97"
        ) from error
    if value not in PAPER_SPARSITIES:
        raise ValueError(
            f"未知的稀疏度 {sparsity!r}；可选：0, 85, 95, 97"
        )
    return value



def parse_paper_cell_label(label: str) -> Tuple[str, int]:
    """解析 ``200M:95`` / ``200M@95`` / ``200M-95`` / ``200M/95``。"""

    text = str(label).strip()
    if not text:
        raise ValueError("--paper-cell 不能为空")
    parts = re.split(r"[:@/\s-]+", text, maxsplit=1)
    if len(parts) != 2:
        # 兼容 "200M95" 这种紧凑写法。
        match = re.fullmatch(r"([0-9.]+[MmBb])([0-9]{1,2})%?", text)
        if not match:
            raise ValueError(
                f"--paper-cell 格式应为 规模:稀疏度（如 200M:95），"
                f"收到 {label!r}"
            )
        parts = [match.group(1), match.group(2)]
    return _normalize_scale(parts[0]), _normalize_sparsity(parts[1])



def paper_cell(
    scale: str,
    sparsity: object,
    arch: str = "smelt",
) -> PaperCell:
    arch_name = str(arch).strip().lower()
    if arch_name not in PAPER_ARCHS:
        raise ValueError(
            f"论文网格只有 {', '.join(PAPER_ARCHS)}，收到 {arch!r}"
        )
    key = (
        arch_name,
        _normalize_scale(scale),
        _normalize_sparsity(sparsity),
    )
    if key not in PAPER_CELLS:
        raise ValueError(f"论文附录 A 中没有单元 {key}")
    return PAPER_CELLS[key]



def _analytic_param_counts(cfg: Config) -> Dict[str, int]:
    """按本实现的结构精确统计非 embedding 参数。

    total  = L * (attn + 2*norm + router + E * expert) + final_norm
    active = L * (attn + 2*norm + router + top_k * expert) + final_norm

    其中 attn = 2*D^2 + 2*D*kv_dim（q/out 各 D^2，k/v 各 D*kv_dim），
    expert = 3*D*H_exp（SwiGLU 的 gate_up 为 D×2H、down 为 H×D）。
    ``_analytic_param_count_roundtrip`` 在自测里与本实现的
    ``nn.Module`` 参数计数逐位核对。
    """

    dim = int(cfg.d_model)
    head_dim = dim // int(cfg.n_heads)
    kv_dim = int(cfg.effective_n_kv_heads) * head_dim
    layers = int(cfg.n_layers)
    experts = int(cfg.n_experts)
    top_k = int(cfg.top_k)
    hidden = int(cfg.expert_hidden)
    per_layer_fixed = (
        2 * dim * dim
        + 2 * dim * kv_dim
        + 2 * dim
        + experts * dim
    )
    expert_params = 3 * dim * hidden
    total = layers * (
        per_layer_fixed + experts * expert_params
    ) + dim
    active = layers * (
        per_layer_fixed + top_k * expert_params
    ) + dim
    return {
        "total": int(total),
        "active": int(active),
        "kv_dim": int(kv_dim),
        "head_dim": int(head_dim),
        "expert_params": int(expert_params),
    }



def _paper_base_head_geometry(hidden: int) -> Tuple[int, int, int]:
    """Baseline 的 head 几何（论文未披露，按可复现规则选定）。

    取 head_dim 最接近 64 的整除组合；KV 头数取 Q 头数的一半
    （2:1 GQA）。论文只要求 GQA 与 KV cache 匹配，没有给出
    head 维度；该选择写入 ``derived`` 字段并在审计中标注来源。
    """

    best: Optional[Tuple[int, int, int]] = None
    for n_heads in _divisors(int(hidden)):
        if n_heads < 2:
            continue
        head_dim = int(hidden) // n_heads
        if not 32 <= head_dim <= 128:
            continue
        score = (abs(head_dim - 64), -head_dim, n_heads)
        if best is None or score < best[0]:
            best = (score, n_heads, head_dim)
    if best is None:
        raise ValueError(
            f"H={hidden} 找不到 32≤head_dim≤128 的整除 head 几何"
        )
    n_heads = best[1]
    head_dim = best[2]
    n_kv_heads = max(1, n_heads // 2)
    return n_heads, n_kv_heads, n_kv_heads * head_dim



def _paper_smelt_head_geometry(
    hidden: int,
    target_kv_dim: int,
) -> Tuple[int, int, int]:
    """SMELT 的 head 几何：以论文报告的 KV 比例为第一目标。

    在 H 的整除 head 划分与 1..n_heads 的 KV 头数中搜索，优先
    最小化 ``|kv_dim / target - 1|``，其次让 head_dim 接近 64。
    不强制 n_heads 整除 n_kv_heads：论文没有披露 head 几何，
    预算匹配需要任意 GQA 比例（见 CausalSelfAttention.core 的
    均匀映射）。
    """

    best: Optional[Tuple[Tuple[float, int, int], int, int, int]] = None
    for n_heads in _divisors(int(hidden)):
        if n_heads < 2:
            continue
        head_dim = int(hidden) // n_heads
        if not 32 <= head_dim <= 128:
            continue
        for n_kv_heads in range(1, n_heads + 1):
            kv_dim = n_kv_heads * head_dim
            error = abs(kv_dim / float(target_kv_dim) - 1.0)
            score = (round(error, 6), abs(head_dim - 64), n_heads)
            if best is None or score < best[0]:
                best = (score, n_heads, n_kv_heads, kv_dim)
    if best is None:
        raise ValueError(
            f"H={hidden} 找不到可用的 GQA head 几何"
        )
    return best[1], best[2], best[3]



def _paper_expert_hidden(
    hidden: int,
    layers: int,
    experts: int,
    total_params: float,
    kv_dim: int,
    active_experts: int = 8,
) -> int:
    """由论文的总参数目标反解专家中间维（按 8 对齐）。

    论文未披露专家中间维；预算匹配把它作为自由量，这里按
    「总非 embedding 参数与论文一致」求解，取 8 的倍数。
    """

    dim = int(hidden)
    per_layer_fixed = (
        2 * dim * dim
        + 2 * dim * int(kv_dim)
        + 2 * dim
        + int(experts) * dim
    )
    budget = float(total_params) * 1e9
    raw = (
        budget - dim - int(layers) * per_layer_fixed
    ) / (int(layers) * int(experts) * 3 * dim)
    if not math.isfinite(raw) or raw < 8.0:
        return 8
    return int(round(raw / 8.0)) * 8



def _paper_config_kwargs(
    cell: PaperCell,
    n_heads: int,
    n_kv_heads: int,
    expert_hidden: int,
) -> Dict[str, object]:
    return {
        "seq_len": int(PAPER_PROTOCOL["context_length"]),
        "d_model": int(cell.hidden),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "n_layers": int(cell.layers),
        "n_experts": int(cell.total_experts),
        "expert_hidden": int(expert_hidden),
        "top_k": int(cell.active_experts),
        "k1": int(cell.active_experts),
        "k2": int(cell.active_experts),
        "mode": cell.arch,
        "residual_scaling": "paper",
        # 论文的 1/r 循环残差语义以全局 residual_scale=1.0 为
        # 前提；显式改成其它值会再乘一次，必须在预设冲突检查
        # 中拒绝，而不能让 fidelity 仍报告架构忠实。
        "residual_scale": 1.0,
    }



def paper_preset(
    scale: str,
    sparsity: object,
    arch: str = "smelt",
) -> Dict[str, object]:
    """构造论文附录 A 某个单元的可运行配置（含推导量）。

    论文只披露 H、L、专家总数、top-8、循环布局与参数/预算目标；
    head 几何与专家中间维未披露，本函数按固定规则推导并记录在
    ``derived`` 中，同时给出与论文目标相比的 measured 值，供
    ``paper_fidelity_report`` 审计。返回值中的 ``config`` 可直接
    作为 ``Config(**config)`` 的输入。
    """

    cell = paper_cell(scale, sparsity, arch)
    base_cell = paper_cell(cell.scale, cell.sparsity, "base")
    base_n_heads, base_n_kv, base_kv_dim = _paper_base_head_geometry(
        base_cell.hidden
    )
    if cell.arch == "base":
        n_heads, n_kv_heads, kv_dim = (
            base_n_heads,
            base_n_kv,
            base_kv_dim,
        )
        kv_target = None
    else:
        if cell.kv_ratio is None:
            raise ValueError(f"{cell} 缺少 KV 比例")
        kv_target = int(round(cell.kv_ratio * base_kv_dim))
        n_heads, n_kv_heads, kv_dim = _paper_smelt_head_geometry(
            cell.hidden,
            kv_target,
        )
    expert_hidden = _paper_expert_hidden(
        cell.hidden,
        cell.layers,
        cell.total_experts,
        cell.total_params,
        kv_dim,
        active_experts=cell.active_experts,
    )
    config = _paper_config_kwargs(
        cell,
        n_heads,
        n_kv_heads,
        expert_hidden,
    )
    cfg = Config(**config)
    counts = _analytic_param_counts(cfg)
    flops = int(
        mode_cost_profile(cfg)["analytic_flops_per_token"]
    )
    return {
        "paper": PAPER_ID,
        "arch": cell.arch,
        "scale": cell.scale,
        "sparsity": cell.sparsity,
        "cell": cell,
        "config": config,
        "derived": {
            "n_heads": int(n_heads),
            "n_kv_heads": int(n_kv_heads),
            "head_dim": int(cell.hidden) // int(n_heads),
            "kv_dim": int(kv_dim),
            "base_kv_dim": int(base_kv_dim),
            "kv_target": kv_target,
            "expert_hidden": int(expert_hidden),
            "top_k": int(cell.active_experts),
            "context_length": int(cfg.seq_len),
            "loop_span_layers": (
                cfg.middle_start + 1,
                cfg.middle_end,
            ),
        },
        "measured": {
            "total_params": int(counts["total"]),
            "active_params": int(counts["active"]),
            "flops_per_token": flops,
        },
        "targets": {
            "total_params": float(cell.total_params),
            "active_params": float(cell.active_params),
            "flops_ratio": cell.flops_ratio,
            "kv_ratio": cell.kv_ratio,
            "compute_equivalent_sparsity": cell.measured_sparsity,
        },
    }



def paper_cell_for_config(cfg: Config) -> Optional[PaperCell]:
    """把运行配置映射回论文附录 A 的网格单元（匹配不到返回 None）。"""

    if cfg.mode not in PAPER_ARCHS:
        return None
    for cell in PAPER_CELLS.values():
        if cell.arch != cfg.mode:
            continue
        if (
            cell.hidden == int(cfg.d_model)
            and cell.layers == int(cfg.n_layers)
            and cell.total_experts == int(cfg.n_experts)
            and cell.active_experts == int(cfg.top_k)
        ):
            return cell
    return None
