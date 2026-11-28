from __future__ import annotations

import hashlib
import math
from typing import Optional, List, Tuple, Dict, Set, FrozenSet


def _finite_float(value: object) -> Optional[float]:
    """把数值/布尔以外的有限数值转成 float，否则返回 None。"""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite_values(values: List[float]) -> Optional[List[float]]:
    """把数值列表统一转成有限 float；任一值不可用则返回 None。"""
    numeric: List[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            return None
        numeric.append(number)
    return numeric


def _mean(values: List[float]) -> Optional[float]:
    """有限数值的算术平均；用缩放求和避免大数值溢出。"""
    numeric = _finite_values(values)
    if not numeric:
        return None
    # 正常量级走原公式，保持历史结果的逐位一致性；只有直接
    # 求和溢出/非有限时才退回缩放计算。
    direct = sum(numeric) / max(len(numeric), 1)
    if math.isfinite(direct):
        return direct
    scale = max(abs(value) for value in numeric)
    if scale == 0.0:
        return 0.0
    total = math.fsum(value / scale for value in numeric)
    result = (total / max(len(numeric), 1)) * scale
    return result if math.isfinite(result) else None


def _median(values: List[float]) -> Optional[float]:
    """中位数（偶数个观测取中间两者的平均）。"""
    numeric = _finite_values(values)
    if not numeric:
        return None
    ordered = sorted(numeric)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    direct = (ordered[middle - 1] + ordered[middle]) / 2.0
    if math.isfinite(direct):
        return direct
    # 两个同号 1e308 直接相加会溢出：各自折半后再相加。
    return ordered[middle - 1] / 2.0 + ordered[middle] / 2.0


def _sample_std(values: List[float]) -> Optional[float]:
    """
    样本标准差（ddof=1）；少于 2 个观测或结果超出 float 范围时
    返回 None。

    正常量级先用原公式（保持历史结果一致）；平方溢出时再按
    最大绝对值缩放计算，避免 ``(value - mean) ** 2`` 抛
    OverflowError。
    """
    numeric = _finite_values(values)
    if numeric is None or len(numeric) < 2:
        return None
    try:
        mean = (sum(numeric) / max(len(numeric), 1)) if numeric else float("nan")
        variance = sum(
            (value - mean) ** 2 for value in numeric
        ) / (len(numeric) - 1)
        direct = math.sqrt(max(0.0, variance))
    except OverflowError:
        direct = None
    # direct == 0 只有在观测完全相等时才是真零标准差：极小但
    # 非零的差值在平方时会下溢为 0，必须同样回退到缩放计算，
    # 否则 1e-200 量级的种子间波动会被报告成零方差。
    if (
        direct is not None
        and math.isfinite(direct)
        and (
            direct > 0.0
            or all(value == numeric[0] for value in numeric)
        )
    ):
        return direct

    scale = max(abs(value) for value in numeric)
    if scale == 0.0:
        return 0.0
    normalized = [value / scale for value in numeric]
    mean = (math.fsum(normalized) / max(len(normalized), 1)) if normalized else float("nan")
    variance = math.fsum(
        (value - mean) ** 2 for value in normalized
    ) / (len(normalized) - 1)
    if not math.isfinite(variance) or variance < 0.0:
        return None
    result = math.sqrt(max(0.0, variance)) * scale
    if math.isfinite(result):
        return result
    # 所有缩放方案都失败，回退到一个保守的估计
    # 对于两个值，std = |a-b|/sqrt(2) 是不可能的（ddof=1时std=|a-b|/sqrt(2)）
    # 对于一般情况，返回范围的半部作为保守估计
    n = len(numeric)
    if n >= 2:
        conservative = (max(numeric) - min(numeric)) / (2.0 * math.sqrt(max(n - 1, 1)))
        if math.isfinite(conservative):
            return conservative
    return None


def _stable_factor_key(value) -> Tuple[str, ...]:
    """
    把因子值转换成可哈希、可比较的稳定表示。

    正常因子值来自 JSON（标量、列表、字典），但旧配置/手工
    编辑的计划可能混入 set 或其它不可哈希对象。后者若原样
    返回，会让 ``planned_comparisons_for_plan`` 等把稳定键
    放进 set/dict 的地方抛 TypeError；这里统一降级为可哈希
    且确定性的表示。
    """
    if isinstance(value, (list, tuple)):
        return tuple(_stable_factor_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        # set 本身无序；排序后再转 tuple，保证同一集合在不同
        # 进程/插入顺序下得到同一个因子键。
        return (
            "set",
            tuple(
                sorted(
                    (_stable_factor_key(item) for item in value),
                    key=repr,
                )
            ),
        )
    if isinstance(value, dict):
        # 键可能是损坏记录里的混合类型；按 repr 排序而不是让
        # Python 直接比较 int/str 而抛 TypeError。
        return tuple(
            sorted(
                (
                    (key, _stable_factor_key(item))
                    for key, item in value.items()
                ),
                key=lambda pair: repr(pair[0]),
            )
        )
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        # 保留整数本身而不是转成 float：2**53 以上的相邻整数
        # 转 float 后会折成同一个值，两个不同的因子水平会被
        # 静默合并到同一实验单元（或同一种子被当成伪重复）。
        return ("number", value)
    if isinstance(value, float):
        # 整数值的 float 与同值 int 必须仍属于同一个因子水平；
        # 转成精确整数可同时保留 int/float 等价性与大整数精度。
        if math.isfinite(value) and value.is_integer():
            return ("number", int(value))
        if math.isnan(value):
            # float('nan') != float('nan') 且不同对象的 hash 不同；
            # 原样返回会让同一损坏/手工配置里的 NaN 因子水平被
            # 拆成多个不可复现的桶。所有 NaN 统一成稳定的哨兵。
            return ("nonfinite", "nan")
        return ("number", value)
    try:
        hash(value)
    except TypeError:
        # 自定义不可哈希对象无法保留身份；使用稳定表示作为
        # 兜底，至少不让整份计划/报告崩溃。
        return ("unhashable", repr(value))
    return ("value", value)


def _factor_sort_rank(value) -> Tuple:
    """
    因子值的排序键：数值 < 已知模式顺序 < 其他字符串。
    """
    if value is None:
        return (3, 0.0, "")
    if isinstance(value, bool):
        return (2, float(value), "")
    if isinstance(value, int):
        # 与 _stable_factor_key 同一口径：整数直接参与排序，
        # 不经过会丢精度的 float 转换。
        return (0, value, "")
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return (0, int(value), "")
        if math.isnan(value):
            # 排序键也必须构成全序：含 NaN 的元组比较既不小于
            # 也不大于其它数值，排序结果会依赖输入顺序。用稳定
            # 的无穷大占位（第三项区分 +inf）保持确定性。
            return (0, float("inf"), "nan")
        return (0, value, "")
    if isinstance(value, str) and value in _FACTOR_VALUE_ORDER:
        return (1, float(_FACTOR_VALUE_ORDER[value]), value)
    if isinstance(value, str):
        return (2, 0.0, value)
    return (4, 0.0, repr(value))


_FACTOR_VALUE_ORDER: Dict[str, str] = {
    "constant": "0",
    "linear": "1",
    "quadratic": "2",
    "exponential": "3",
    "logarithmic": "4",
}


def _row_factor_value(
    row: Dict[str, object],
    key: str,
) -> object:
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


def _randomization_seed(differences: List[float]) -> int:
    """由配对差值生成稳定的随机化种子（不触碰全局 RNG）。"""
    material = repr(
        [_finite_float(value) for value in differences]
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(),
        "big",
    )


_MIN_SEEDS_FOR_STATS = 3
_RANDOMIZATION_EXACT_MAX_N = 12
_RANDOMIZATION_MONTE_CARLO_REPLICATES = 20000
_SIGNIFICANCE_LEVEL = 0.05
_TARGET_POWER = 0.80


__all__ = [
    "_finite_float",
    "_finite_values",
    "_mean",
    "_median",
    "_sample_std",
    "_stable_factor_key",
    "_factor_sort_rank",
    "_row_factor_value",
    "_format_factor_value",
    "_randomization_seed",
    "_MIN_SEEDS_FOR_STATS",
    "_RANDOMIZATION_EXACT_MAX_N",
    "_RANDOMIZATION_MONTE_CARLO_REPLICATES",
    "_SIGNIFICANCE_LEVEL",
    "_TARGET_POWER",
]