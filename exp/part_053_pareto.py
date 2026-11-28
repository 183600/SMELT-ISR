from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CARD_SCHEMA = "smelt-isr/conclusion-card/v1"


def _finite_float(value: object) -> Optional[float]:
    """把数值/布尔以外的有限数值转成 float，否则返回 None。"""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (OverflowError, TypeError, ValueError):
        # JSON 允许任意大整数；float(10**400) 会溢出。分析
        # 管线应把这种值当作不可用，而不是让整份报告崩溃。
        return None
    if not math.isfinite(f):
        return None
    return f


def _pareto_frontier(
    performances: List[float],
    costs: List[float],
    minimize_performance: bool = True,
    minimize_cost: bool = True,
) -> List[int]:
    """
    计算帕累托前沿，返回满足条件的点的索引。

    参数:
        performances: 性能指标列表（如 val_loss 或效应量），较小通常更好
        costs: 成本指标列表（如 FLOPs/参数），较小通常更好
        minimize_performance: 是否最小化性能指标（默认True，val_loss越小越好）
        minimize_cost: 是否最小化成本指标（默认True，FLOPs/参数越小越好）

    返回:
        满足帕累托前沿条件的点的原始索引列表
    """
    n = len(performances)
    if n == 0:
        return []

    # 将性能和成本统一为“越小越好”方向
    # 如果是要最大化（如效应量），则取相反数
    perf = [
        p if minimize_performance else -p for p in performances
    ]
    cost = [
        c if minimize_cost else -c for c in costs
    ]

    # 找出帕累托前沿点：点 i 满足没有其他点 j 统治 i
    # 对于最小化问题：点 j 统治 i 的条件是
    #   perf[j] <= perf[i] 且 cost[j] <= cost[i] 且 (perf[j] < perf[i] 或 cost[j] < cost[i])
    # 即：j 在 both 维度上不比 i 差，且至少在 one 维度上严格优于 i
    frontier_indices: List[int] = []

    for i in range(n):
        is_pareto = True
        for j in range(n):
            if i == j:
                continue
            # 如果点 j 在性能和成本上都不比 i 差（更小/更好），
            # 并且至少在 one 维度上严格优于 i，则 j 统治 i
            if perf[j] <= perf[i] and cost[j] <= cost[i]:
                if perf[j] < perf[i] or cost[j] < cost[i]:
                    is_pareto = False
                    break
        if is_pareto:
            frontier_indices.append(i)

    return frontier_indices


def _hypervolume(
    frontier_performances: List[float],
    frontier_costs: List[float],
    ref_performance: float = 1.0,
    ref_cost: float = 100.0,
) -> float:
    """
    计算超体积（Hypervolume）。
    使用参考点 (ref_performance, ref_cost) 作为基准。
    前沿点构成的区域面积即为 HV。
    简单的矩形近似：每个前沿点贡献 (ref_performance - perf) * (ref_cost - cost)
    """
    if not frontier_performances:
        return 0.0

    hv = 0.0
    # 按成本从小到大排序，避免重复计算区域
    points = list(
        zip(frontier_performances, frontier_costs)
    )
    points.sort(key=lambda p: p[1])  # 按成本升序排序

    current_best_perf = ref_performance
    for perf, cost in points:
        if perf < current_best_perf:
            hv += (current_best_perf - perf) * (ref_cost - cost)
            current_best_perf = perf

    return max(hv, 0.0)


def _r_squared(
    frontier_performances: List[float],
    frontier_costs: List[float],
    all_performances: List[float],
    all_costs: List[float],
) -> float:
    """
    计算 R²：将帕累托前沿视为模型，计算剩余点到前沿的加权残差。
    简化版：将前沿点拟合为代价函数，计算解释方差。
    这里使用简单的线性回归 R² 作为近似。
    """
    if len(frontier_performances) < 2:
        return 0.0

    # 简单的线性拟合：performance = a * cost + b
    x = np.array(frontier_costs, dtype=float)
    y = np.array(frontier_performances, dtype=float)

    if len(x) < 2 or np.var(x) == 0:
        return 0.0

    try:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        r2 = 1.0 - ss_res / ss_tot
        return max(min(r2, 1.0), 0.0)
    except np.linalg.LinAlgError:
        return 0.0


def pareto_frontier_analysis(
    contrasts: List[Dict[str, object]],
    perf_metric: str = "val_loss_at_common_step",
    cost_metric: str = "flops_per_forward",
    reference_performance: Optional[float] = None,
    reference_cost: Optional[float] = None,
    output_csv: Optional[Path] = None,
) -> Dict[str, object]:
    """
    对对比集进行帕累托前沿分析。

    参数:
        contrasts: 对比条目列表，每个条目应包含 perf_metric 与 cost_metric
        perf_metric: 性能指标在 contrast 字典中的键名（默认 val_loss_at_common_step）
        cost_metric: 成本指标在 contrast 字典中的键名（默认 flops_per_forward）
        reference_performance: 参考性能值（默认从所有对比的最优值推导）
        reference_cost: 参考成本值（默认从所有对比的最差值推导）
        output_csv: 可选的 CSV 输出路径，将把前沿点写入该文件

    返回:
        字典包含：
        - "pareto_indices": 满足帕累托前沿的对比在 input 列表中的索引
        - "pareto_performances": 前沿点对应的性能值
        - "pareto_costs": 前沿点对应的成本值
        - "hypervolume": 计算得到的超体积
        - "r_squared": 线性拟合的 R² 作为模型拟合优度
        - "pareto_frontier": 包含 front_idx, performance, cost 等字段的列表
    """
    n = len(contrasts)
    if n == 0:
        return {
            "pareto_indices": [],
            "pareto_performances": [],
            "pareto_costs": [],
            "hypervolume": 0.0,
            "r_squared": 0.0,
            "pareto_frontier": [],
        }

    # 收集性能与成本值
    performances: List[Optional[float]] = []
    costs: List[Optional[float]] = []

    for i, contrast in enumerate(contrasts):
        perf = _finite_float(contrast.get(perf_metric))
        cost = _finite_float(contrast.get(cost_metric))
        performances.append(perf)
        costs.append(cost)

    # 确定参考点：性能取最小（越小越好），成本取最大（作为基准）
    # 如果有 reference 参数则使用，否则从数据中推导
    if reference_performance is None:
        valid_perfs = [p for p in performances if p is not None]
        reference_performance = min(valid_perfs) * 1.5 if valid_perfs else 1.0

    if reference_cost is None:
        valid_costs = [c for c in costs if c is not None]
        reference_cost = max(valid_costs) * 1.5 if valid_costs else 100.0

    # 计算帕累托前沿
    frontier_indices = _pareto_frontier(
        performances, costs, minimize_performance=True, minimize_cost=True
    )

    frontier_perfs = [performances[i] for i in frontier_indices]
    frontier_costs = [costs[i] for i in frontier_indices]

    # 计算超体积
    hv = _hypervolume(frontier_perfs, frontier_costs, reference_performance, reference_cost)

    # 计算 R²
    r2 = _r_squared(frontier_perfs, frontier_costs, performances, costs)

    # 构建前沿详情
    frontier_detail: List[Dict[str, object]] = []
    for idx in frontier_indices:
        frontier_detail.append(
            {
                "front_idx": idx,
                "performance": performances[idx],
                "cost": costs[idx],
                "contrast_label": contrasts[idx].get("family", f"contrast_{idx}"),
            }
        )

    # 输出 CSV 如果指定
    if output_csv is not None:
        try:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["front_idx", "performance", "cost", "contrast_label"])
                for entry in frontier_detail:
                    writer.writerow(
                        [
                            entry["front_idx"],
                            entry["performance"]
                            if entry["performance"] is not None
                            else "",
                            entry["cost"]
                            if entry["cost"] is not None
                            else "",
                            entry["contrast_label"],
                        ]
                    )
        except OSError:
            pass  # CSV 写入失败不中断主流程

    return {
        "pareto_indices": frontier_indices,
        "pareto_performances": frontier_perfs,
        "pareto_costs": frontier_costs,
        "hypervolume": hv,
        "r_squared": r2,
        "pareto_frontier": frontier_detail,
    }