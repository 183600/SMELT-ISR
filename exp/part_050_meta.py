from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 元分析框架（P1，跨数据集/套件汇总）
#
# 把若干独立对比（同一效应在不同数据集/种子套件/评估口径下的
# 估计）合并为一个随机效应汇总。与 ``--compare`` 的单次分析不同，
# 元分析的输入是「已经算好的效应量与标准误」，用于回答：
#
#   - 跨套件的汇总效应是多少？（随机效应 DerSimonian-Laird）
#   - 各套件的效应是否一致？（Cochran's Q、I²、H²、τ²）
#   - 新研究里效应可能落在什么范围？（预测区间）
#
# 局限（必须随结果一起报告）：
#   - DL 的 τ² 估计在套件数很少（k<5）时非常不稳定，I²/τ² 只作
#     描述性证据；
#   - 预测区间假设各套件的真效应近似服从正态分布；
#   - 汇总永远不能修复系统性偏差：如果各套件共享同一混杂
#     （同一份有泄露的数据、同一个有偏的基线），汇总只会把
#     偏差平均得更体面。
# ---------------------------------------------------------------------------

_META_Z_975 = 1.959963984540054  # 标准正态 97.5% 分位数
_META_MIN_STUDIES = 2


def _meta_lgamma(value: float) -> float:
    if value <= 0.0:
        raise ValueError("lgamma 参数必须为正")
    return math.lgamma(value)


def _meta_gamma_upper_regularized(
    shape: float, value: float
) -> float:
    """
    正则化上不完全 Gamma 函数 Q(shape, value)（NR 算法，
    gser/gcf 双路，无 SciPy 依赖）。用于 χ² 上尾 p 值。
    """
    if shape <= 0.0 or value < 0.0:
        raise ValueError("gamma 参数非法")
    if value == 0.0:
        return 1.0
    if value < shape + 1.0:
        # 级数计算 P，再取补。
        total = 1.0 / shape
        term = total
        index = 1
        while index <= 1000:
            term *= value / (shape + index)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            index += 1
        p_value = total * math.exp(
            -value + shape * math.log(value + 1e-8) - _meta_lgamma(shape)
        )
        return max(0.0, min(1.0, 1.0 - p_value))
    # 连分式计算 Q。
    tiny = 1e-300
    b_value = value + 1.0 - shape
    c_value = 1.0 / tiny
    d_value = 1.0 / b_value
    h_value = d_value
    for index in range(1, 1001):
        an = -index * (index - shape)
        b_value += 2.0
        d_value = an * d_value + b_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = b_value + an / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q_value = math.exp(
        -value + shape * math.log(value + 1e-8) - _meta_lgamma(shape)
    ) * h_value
    return max(0.0, min(1.0, q_value))


def _meta_betai(
    a_value: float, b_value: float, x_value: float
) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)（NR 算法）。"""
    if a_value <= 0.0 or b_value <= 0.0:
        raise ValueError("beta 参数必须为正")
    if x_value <= 0.0:
        return 0.0
    if x_value >= 1.0:
        return 1.0
    ln_beta = _meta_lgamma(a_value + b_value) - _meta_lgamma(
        a_value
    ) - _meta_lgamma(b_value)
    front = math.exp(
        ln_beta + a_value * math.log(x_value + 1e-8) + b_value * math.log1p(-x_value)
    )
    if x_value < (a_value + 1.0) / (a_value + b_value + 2.0):
        return front * _meta_beta_continued(
            a_value, b_value, x_value
        ) / a_value
    return 1.0 - front * _meta_beta_continued(
        b_value, a_value, 1.0 - x_value
    ) / b_value


def _meta_beta_continued(
    a_value: float, b_value: float, x_value: float
) -> float:
    """Lentz 连分式计算 betacf。"""
    tiny = 1e-300
    qab = a_value + b_value
    qap = a_value + 1.0
    qam = a_value - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x_value / qap
    if abs(d_value) < tiny:
        d_value = tiny
    d_value = 1.0 / d_value
    h_value = d_value
    for index in range(1, 301):
        m2 = 2 * index
        aa = index * (b_value - index) * x_value / (
            (qam + m2) * (a_value + m2)
        )
        d_value = 1.0 + aa * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + aa / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        h_value *= d_value * c_value
        aa = -(a_value + index) * (qab + index) * x_value / (
            (a_value + m2) * (qap + m2)
        )
        d_value = 1.0 + aa * d_value
        if abs(d_value) < tiny:
            d_value = tiny
        c_value = 1.0 + aa / c_value
        if abs(c_value) < tiny:
            c_value = tiny
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h_value


def _meta_student_t_sf(t_value: float, df: int) -> float:
    """Student t 上尾概率（整数自由度，经不完全 Beta）。"""
    if df < 1:
        raise ValueError("t 分布自由度必须 >= 1")
    x_value = df / (df + t_value * t_value)
    tail = 0.5 * _meta_betai(0.5 * df, 0.5, x_value)
    return max(0.0, min(1.0, tail))


def _meta_t_critical_975(df: int) -> float:
    """双侧 5% 的 t 临界值（t_{0.975, df}，二分求逆）。"""
    if df < 1:
        raise ValueError("t 分布自由度必须 >= 1")
    low = 0.0
    high = 1000.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _meta_student_t_sf(middle, df) > 0.025:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _meta_chi_square_sf(value: float, df: int) -> float:
    """χ² 上尾概率 P(χ²_df >= value)。"""
    if df < 1:
        raise ValueError("卡方自由度必须 >= 1")
    if value <= 0.0:
        return 1.0
    return max(0.0, min(1.0, _meta_gamma_upper_regularized(0.5 * df, 0.5 * value)))


def _meta_normal_sf(z_value: float) -> float:
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def meta_analyze(
    effects: Sequence[float],
    variances: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    随机效应（DerSimonian-Laird）元分析。

    参数是各套件（研究）的效应量估计与**方差**（标准误的平方）。
    返回固定效应与随机效应两套汇总、异质性统计量（Q / I² / H² /
    τ²）与预测区间。套件数 < 2、效应或方差非有限、方差 <= 0 时
    抛 ``ValueError``——调用方应先把不可用的套件剔除并记录原因，
    而不是让不可用输入静默参与汇总。

    字段含义：
    - ``fixed``：共同真效应假设下的逆方差加权汇总（套件间无
      异质性时与随机效应一致）；
    - ``random_dl``：允许套件间真效应不同（τ² ≥ 0）的 DL 汇总；
    - ``heterogeneity``：Q（Cochran）、df=k−1、p、I²、H²、τ²；
      I² = max(0, (Q−df)/Q) 描述观测变异中超出抽样噪声的比例；
    - ``prediction_interval``：θ̂ ± t_{0.975,k−1}·√(se²+τ²)，
      估计「再做一个同类套件」的效应落点（Higgins 等），套件
      数少或 τ² 估计不稳时只能当粗略参考。

    注意：DL 假定各套件估计的是「相关但不必然相同」的效应；
    若各套件其实共享同一份有偏数据/同一实现，汇总不会修复偏差。
    """
    if len(effects) != len(variances):
        raise ValueError(
            f"effects 与 variances 长度不一致: {len(effects)} vs {len(variances)}"
        )
    k = len(effects)
    if k < _META_MIN_STUDIES:
        raise ValueError(
            f"元分析至少需要 {_META_MIN_STUDIES} 个套件（收到 {k}）；"
            "单个对比的显著性请直接看 --compare 的结果"
        )
    if labels is not None and len(labels) != k:
        raise ValueError(
            f"labels 长度与 effects 不一致: {len(labels)} vs {k}"
        )
    clean_effects: List[float] = []
    clean_variances: List[float] = []
    clean_labels: List[str] = []
    for index in range(k):
        effect = float(effects[index])
        variance = float(variances[index])
        if not math.isfinite(effect) or not math.isfinite(variance):
            raise ValueError(
                f"第 {index} 个套件的效应/方差不是有限数"
            )
        if variance <= 0.0:
            raise ValueError(
                f"第 {index} 个套件的方差必须为正（收到 {variance}）；"
                "没有标准误的估计无法参与逆方差加权"
            )
        clean_effects.append(effect)
        clean_variances.append(variance)
        clean_labels.append(
            str(labels[index]) if labels is not None else f"study_{index}"
        )
    weights_fixed = [1.0 / v for v in clean_variances]
    total_weight = sum(weights_fixed)
    theta_fixed = sum(
        w * y for w, y in zip(weights_fixed, clean_effects)
    ) / total_weight
    se_fixed = math.sqrt(1.0 / total_weight)
    q_value = sum(
        w * (y - theta_fixed) ** 2
        for w, y in zip(weights_fixed, clean_effects)
    )
    df = k - 1
    q_p_value = _meta_chi_square_sf(q_value, df)
    c_value = total_weight - sum(
        w * w for w in weights_fixed
    ) / total_weight
    tau_squared = 0.0
    if c_value > 0.0:
        tau_squared = max(0.0, (q_value - df) / c_value)
    weights_random = [1.0 / (v + tau_squared) for v in clean_variances]
    total_weight_random = sum(weights_random)
    theta_random = sum(
        w * y for w, y in zip(weights_random, clean_effects)
    ) / total_weight_random
    se_random = math.sqrt(1.0 / total_weight_random)
    i_squared = max(0.0, (q_value - df) / q_value) if q_value > 0 else 0.0
    h_squared = q_value / df if df > 0 else float("inf")
    z_random = (
        theta_random / se_random if se_random > 0 else None
    )
    t_critical = _meta_t_critical_975(df)
    prediction_half_width = t_critical * math.sqrt(
        se_random * se_random + tau_squared
    )
    if z_random is not None:
        pooled_p_two_sided = 2.0 * _meta_normal_sf(abs(z_random))
    else:
        pooled_p_two_sided = None
    return {
        "schema": "smelt-isr/meta-analysis/v1",
        "method": "derSimonian-Laird random effects",
        "k": k,
        "labels": clean_labels,
        "effects": clean_effects,
        "variances": clean_variances,
        "fixed": {
            "estimate": theta_fixed,
            "se": se_fixed,
            "ci95_low": theta_fixed - _META_Z_975 * se_fixed,
            "ci95_high": theta_fixed + _META_Z_975 * se_fixed,
        },
        "random_dl": {
            "estimate": theta_random,
            "se": se_random,
            "ci95_low": theta_random - _META_Z_975 * se_random,
            "ci95_high": theta_random + _META_Z_975 * se_random,
            "z": z_random,
            "p_value_two_sided": pooled_p_two_sided,
            "tau_squared": tau_squared,
            "tau": math.sqrt(tau_squared),
        },
        "heterogeneity": {
            "q": q_value,
            "df": df,
            "p_value": q_p_value,
            "i_squared": i_squared,
            "h_squared": h_squared,
            "tau_squared": tau_squared,
        },
        "prediction_interval": {
            "low": theta_random - prediction_half_width,
            "high": theta_random + prediction_half_width,
            "method": "theta_hat +/- t_{0.975,k-1} * sqrt(se^2 + tau^2)",
            "t_critical": t_critical,
        },
        "note": (
            "元分析汇总的是效应量与标准误，不重新训练、也不修复"
            "系统性偏差：各套件共享同一混杂（同一数据泄露、同一"
            "有偏基线）时，汇总会把偏差平均而不是消除。k<5 时 "
            "τ²/I² 的估计很不稳定，只作描述性证据；预测区间假设"
            "套件真效应近似正态。"
        ),
    }


def meta_analyze_contrasts(
    contrasts: Sequence[Dict[str, object]],
    effect_key: str = "delta",
) -> Dict[str, object]:
    """
    从 ``--compare`` 的对比记录构造元分析输入。

    每个对比贡献一个效应量（``delta``，或 ``effect_key`` 指定的
    字段）与标准误（``se``）。缺效应/缺标准误/混杂的对比会被
    剔除并记录在 ``excluded`` 里，而不是悄悄丢弃或中断。
    剩余套件不足时抛 ``ValueError``。
    """
    effects: List[float] = []
    variances: List[float] = []
    labels: List[str] = []
    excluded: List[Dict[str, object]] = []
    for index, contrast in enumerate(contrasts):
        if not isinstance(contrast, dict):
            excluded.append({"index": index, "reason": "not_a_dict"})
            continue
        label = (
            f"{contrast.get('baseline')}→{contrast.get('variant')}"
            if contrast.get("baseline") is not None
            else f"contrast_{index}"
        )
        if contrast.get("confounded"):
            excluded.append(
                {"label": label, "reason": "confounded_contrast"}
            )
            continue
        effect = contrast.get(effect_key)
        se = contrast.get("se")
        if not isinstance(effect, (int, float)) or isinstance(
            effect, bool
        ):
            excluded.append({"label": label, "reason": "missing_effect"})
            continue
        if not isinstance(se, (int, float)) or isinstance(se, bool):
            excluded.append({"label": label, "reason": "missing_se"})
            continue
        effect_value = float(effect)
        se_value = float(se)
        if not math.isfinite(effect_value) or not math.isfinite(
            se_value
        ) or se_value <= 0.0:
            excluded.append(
                {"label": label, "reason": "non_finite_or_bad_se"}
            )
            continue
        effects.append(effect_value)
        variances.append(se_value * se_value)
        labels.append(str(label))
    if len(effects) < _META_MIN_STUDIES:
        raise ValueError(
            f"可用套件不足（{len(effects)} < {_META_MIN_STUDIES}）；"
            f"被剔除: {excluded or '无'}"
        )
    result = meta_analyze(effects, variances, labels)
    result["excluded"] = excluded
    result["effect_key"] = effect_key
    return result


def format_meta_summary(result: Dict[str, object]) -> str:
    """把元分析结果渲染成可打印的简短文本。"""
    random_effects = result.get("random_dl") or {}
    heterogeneity = result.get("heterogeneity") or {}
    prediction = result.get("prediction_interval") or {}
    fixed = result.get("fixed") or {}
    lines = [
        f"元分析（{result.get('method')}，k={result.get('k')}）",
        f"  随机效应汇总: θ̂={_format_finite(random_effects.get('estimate'))} "
        f"(SE={_format_finite(random_effects.get('se'))}, "
        f"95% CI [{_format_finite(random_effects.get('ci95_low'))}, "
        f"{_format_finite(random_effects.get('ci95_high'))}])",
        f"  固定效应汇总: θ̂={_format_finite(fixed.get('estimate'))} "
        f"(SE={_format_finite(fixed.get('se'))})",
        f"  异质性: Q={_format_finite(heterogeneity.get('q'))} "
        f"(df={heterogeneity.get('df')}, p={_format_finite(heterogeneity.get('p_value'), '.4f')}), "
        f"I²={_format_finite(heterogeneity.get('i_squared'), '.1%', 100.0)}, "
        f"H²={_format_finite(heterogeneity.get('h_squared'))}, "
        f"τ²={_format_finite(heterogeneity.get('tau_squared'))}",
        f"  预测区间: [{_format_finite(prediction.get('low'))}, "
        f"{_format_finite(prediction.get('high'))}]",
    ]
    excluded = result.get("excluded") or []
    if excluded:
        lines.append(
            "  已剔除套件: "
            + "; ".join(
                f"{entry.get('label', entry.get('index'))}"
                f"（{entry.get('reason')}）"
                for entry in excluded
            )
        )
    lines.append("  " + str(result.get("note")))
    return "\n".join(lines)
