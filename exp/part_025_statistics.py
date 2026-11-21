from __future__ import annotations

import math
import sys
from typing import List, Dict, Optional, Tuple

# Assuming other imports and definitions exist elsewhere in the file

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

    numeric_a = _finite_values(values_a)
    numeric_b = _finite_values(values_b)
    if numeric_a is None or numeric_b is None:
        return {"test": "invalid_values"}
    values_a = numeric_a
    values_b = numeric_b
    max_abs = max(
        (abs(value) for value in values_a + values_b),
        default=0.0,
    )
    if 0.0 < max_abs < 1.0e-150:
        # 极小尺度下方差会下溢成 0，被误判成零方差/无噪声
        # 分离；先按公共尺度归一化再还原效应与标准误。
        scaled = _welch_t_test(
            [value / max_abs for value in values_a],
            [value / max_abs for value in values_b],
        )
        if scaled.get("test") == "welch_t":
            effect = _finite_float(scaled.get("effect"))
            se = _finite_float(scaled.get("se"))
            if effect is not None:
                scaled["effect"] = effect * max_abs
            if se is not None:
                scaled["se"] = se * max_abs
        return scaled
    if max_abs > 1.0e150:
        # 极大尺度下方差/离差平方会溢出成 inf，不能把合法数据
        # 误判成 invalid_values；t 检验本身对公共尺度不变。
        scaled = _welch_t_test(
            [value / max_abs for value in values_a],
            [value / max_abs for value in values_b],
        )
        if scaled.get("test") == "welch_t":
            effect = _finite_float(scaled.get("effect"))
            se = _finite_float(scaled.get("se"))
            if effect is not None:
                rescaled_effect = effect * max_abs
                scaled["effect"] = (
                    rescaled_effect
                    if math.isfinite(rescaled_effect)
                    else None
                )
            if se is not None:
                rescaled_se = se * max_abs
                scaled["se"] = (
                    rescaled_se
                    if math.isfinite(rescaled_se)
                    else None
                )
        return scaled
    mean_a = _mean(values_a)
    mean_b = _mean(values_b)
    if mean_a is None or mean_b is None:
        return {"test": "invalid_values"}
    var_a = _sample_variance(values_a)
    var_b = _sample_variance(values_b)
    if var_a is None or var_b is None:
        # 合法但大到超出 float 范围的方差：无法给出有限检验量，
        # 按不可用处理，而不是让 **2 抛 OverflowError。
        return {"test": "invalid_values"}

    term_a = var_a / n_a
    term_b = var_b / n_b
    se_squared = term_a + term_b
    delta = mean_b - mean_a
    if not math.isfinite(se_squared) or not math.isfinite(delta):
        return {"test": "invalid_values"}

    if se_squared == 0.0:
        if abs(delta) < 1e-12:
            return {
                "test": "welch_t",
                "t": 0.0,
                "df": None,
                "p_value": 1.0,
                "note": "zero_variance",
                "effect": 0.0,
                "se": 0.0,
            }
        return {
            "test": "welch_t",
            "t": None,
            "df": None,
            "p_value": 0.0,
            "note": "noise_free_separation",
            "effect": delta,
            "se": 0.0,
        }

    # Welch-Satterthwaite 自由度：正常量级沿用历史公式，保持
    # 逐位一致；平方项可能溢出 **或下溢** 时才改用相对方差计算，
    # 公共缩放因子在分子分母中会抵消。下溢同样会破坏精度：
    # term≈1e-161、se²≈1e-160 时平方落到次正规数，直接相除会
    # 给出错误的 df（如真实 df=4 被算成 4.018），进而污染 p 值/
    # 置信区间；这里在平方可能进入次正规范围前就切到缩放路径。
    df = None
    try:
        old_denominator = (
            term_a * term_a / max(n_a - 1, 1)
            + term_b * term_b / max(n_b - 1, 1)
        )
        if (
            old_denominator > 0.0
            and math.isfinite(old_denominator)
            and old_denominator >= sys.float_info.min
            and se_squared >= math.sqrt(sys.float_info.min)
        ):
            candidate = (
                se_squared * se_squared / old_denominator
            )
            if math.isfinite(candidate):
                df = candidate
    except OverflowError:
        df = None
    if df is None:
        scale = max(term_a, term_b)
        if scale <= 0.0:
            return {"test": "invalid_values"}
        scaled_a = term_a / scale
        scaled_b = term_b / scale
        denominator = (
            scaled_a * scaled_a / max(n_a - 1, 1)
            + scaled_b * scaled_b / max(n_b - 1, 1)
        )
        if denominator <= 0.0 or not math.isfinite(denominator):
            return {"test": "invalid_values"}
        df = (scaled_a + scaled_b) ** 2 / denominator
    t = delta / math.sqrt(max(0.0, se_squared))
    return {
        "test": "welch_t",
        "t": t,
        "df": df,
        "p_value": _two_sided_t_p_value(abs(t), df),
        "effect": delta,
        "se": math.sqrt(se_squared),
    }