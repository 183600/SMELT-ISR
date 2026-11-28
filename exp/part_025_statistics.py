from __future__ import annotations

import math
import functools
from typing import List, Tuple, Optional, Sequence, Dict

from .part_023_factorial_helpers import _two_sided_t_p_value

import numpy as np
import torch

_SIGNIFICANCE_LEVEL = 0.05
_TARGET_POWER = 0.80
_MIN_SEEDS_FOR_STATS = 3

# 跨指标一致性分析用的「同源指标族」划分。族内指标互相高度
# 相关（例如 val_loss / val_ppl / val_bits_per_byte 互为单调
# 变换），不能当作多份独立证据；跨族汇总才近似独立证据。
_CROSS_METRIC_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "loss",
        (
            "val_loss",
            "val_ppl",
            "val_bits_per_byte",
            "val_macro_nll",
        ),
    ),
    (
        "accuracy",
        (
            "val_token_accuracy",
            "val_micro_f1",
            "val_macro_f1",
            "val_weighted_f1",
        ),
    ),
    (
        "precision",
        ("val_precision_macro", "val_precision_weighted"),
    ),
    (
        "recall",
        ("val_recall_macro", "val_recall_weighted"),
    ),
    ("auc", ("val_pr_auc_macro", "val_roc_auc_macro")),
    ("pr_auc_lift", ("val_pr_auc_macro_lift",)),
)


def _fragility_summary(
    contrasts: Sequence[Dict[str, object]],
    label: str = "single_factor",
) -> Dict[str, object]:
    """
    汇总一组对比的脆弱性判定，供报告顶层落盘与展示。
    """
    entries: List[Dict[str, object]] = []
    seen: set = set()
    for contrast in contrasts:
        fragility = contrast.get("fragility")
        if not isinstance(fragility, dict):
            continue
        if not fragility.get("available"):
            continue
        # 单因子/直接对比用 baseline→variant；全因子消融的
        # 主效应/交互没有这两个字段，用 design + effect_name
        # 作为可读身份，不能把它们汇总成 None→None。
        baseline = contrast.get("baseline")
        variant = contrast.get("variant")
        factor = contrast.get("factor")
        design = contrast.get("design")
        effect_name = contrast.get("effect_name")
        if baseline is None and variant is None:
            baseline = design or effect_name
            variant = effect_name or baseline
        if factor is None:
            factor = effect_name or "effect"
        # 同一个对比可能同时出现在单因子对比与
        # --compare-baseline 直接对比里（方向也可能相反，
        # 例如 local_base→A 与 A→local_base）；脆弱性结论
        # 相同，不能因此重复计数或重复打印。按无序的对比
        # 双方 + 因子去重。
        key = (
            tuple(sorted((str(baseline), str(variant)))),
            str(factor),
        )
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, object] = {
            "family": label,
            "baseline": baseline,
            "variant": variant,
            "factor": factor,
            "design": design,
            "effect_name": effect_name,
            "verdict": fragility.get("verdict"),
            "significant": bool(contrast.get("significant")),
            "fragility_index": fragility.get(
                "min_seed_deletions_to_lose_significance"
            ),
            # 删除 fragility_index 个种子后实际得到的 p 值；
            # 不能拿 max_leave_one_out_p_value（单删最大值）
            # 冒充成组删除后的 p。
            "fragility_index_p_value": fragility.get(
                "min_seed_deletions_p_value"
            ),
            "n": fragility.get("n"),
            "n_positive": fragility.get("n_positive"),
            "n_negative": fragility.get("n_negative"),
            "sign_test_p_value": (
                _as_dict(fragility.get("sign_test")).get(
                    "p_value"
                )
            ),
            "max_leave_one_out_p_value": fragility.get(
                "max_leave_one_out_p_value"
            ),
            "alpha": fragility.get("alpha"),
        }
        # 附加 MDE 与 TOST 信息，帮助阅读者判定「脆弱性」是由
        # 样本量不足还是效应真实超出等价界。
        se = contrast.get("se")
        df = contrast.get("df")
        baseline_mean = contrast.get("mean_effect") or contrast.get(
            "delta"
        )
        if se is not None and df is not None and baseline_mean is not None:
            try:
                se_v = float(se)
                df_v = float(df)
                bl_abs = float(abs(baseline_mean))
                if se_v > 0 and bl_abs > 0:
                    alpha = _SIGNIFICANCE_LEVEL
                    t_crit = _t_critical(df_v, alpha)
                    if t_crit is not None:
                        contrast_mag = abs(float(baseline_mean))
                        equivalence_margin_needed = (
                            contrast_mag + t_crit * se_v
                        )
                        entry["equivalence_margin_needed"] = (
                            equivalence_margin_needed
                        )
                        margin_rel = 0.01  # default 1%
                        margin_abs = margin_rel * bl_abs
                        if contrast_mag > margin_abs:
                            entry["equivalence_blocker"] = (
                                "effect_exceeds_margin"
                            )
                        else:
                            sd_observed = _observed_seed_sd(
                                entry,
                                se_v,
                            )
                            if sd_observed is not None and bl_abs != 0.0:
                                design = (
                                    "welch"
                                    if contrast.get("test") == "welch_t"
                                    else "paired"
                                )
                                seeds_needed = (
                                    _seeds_to_detect(
                                        abs(float(baseline_mean)),
                                        sd_observed,
                                        design=design,
                                    )
                                )
                                if seeds_needed is not None:
                                    delta_seeds = (
                                        seeds_needed
                                        - int(contrast.get("n_pairs", 0))
                                    )
                                    if delta_seeds > 0:
                                        entry[
                                            "equivalence_blocker"
                                        ] = f"need_{delta_seeds}_more_seeds"
                                    else:
                                        entry[
                                            "equivalence_blocker"
                                        ] = None
                                else:
                                    entry[
                                        "equivalence_blocker"
                                    ] = "cannot_compute_seeds"
                            else:
                                entry[
                                    "equivalence_blocker"
                                ] = None
                    else:
                        entry[
                            "equivalence_margin_needed"
                        ] = None
                        entry["equivalence_blocker"] = None
            except (TypeError, ValueError, OverflowError):
                entry["equivalence_margin_needed"] = None
                entry["equivalence_blocker"] = None
        else:
            entry["equivalence_margin_needed"] = None
            entry["equivalence_blocker"] = None
        entries.append(entry)

    verdict_counts: Dict[str, int] = {}
    for entry in entries:
        verdict = str(entry.get("verdict"))
        verdict_counts[verdict] = (
            verdict_counts.get(verdict, 0) + 1
        )
    fragile = [
        entry for entry in entries
        if entry.get("verdict") == "fragile"
        and entry.get("significant")
    ]
    direction_mixed = [
        entry for entry in entries
        if entry.get("verdict") == "direction_mixed"
    ]
    not_evaluable = [
        entry for entry in entries
        if entry.get("verdict") == "not_evaluable"
    ]
    advisories: List[str] = []
    if fragile:
        advisories.append(
            "以下显著对比在删除少量种子后不再显著（结论由"
            "少数运行驱动，不能作为稳健结论）；需要更多独立"
            "种子或先调查这些种子的异常来源："
            + "; ".join(
                f"{entry['baseline']}→{entry['variant']}"
                f"（删 {entry['fragility_index']} 个种子后"
                + (
                    f"p={float(entry['fragility_index_p_value']):.4f}"
                    if isinstance(
                        entry.get("fragility_index_p_value"),
                        (int, float),
                    )
                    else "显著性消失"
                )
                + f"，alpha={float(entry['alpha']):.3f}"
                + (
                    f"，MDE={float(entry.get('equivalence_margin_needed', 0)):.4f}"
                    if entry.get("equivalence_margin_needed") is not None
                    else ""
                )
                + f"，等需种子≈{entry.get('equivalence_blocker', 'N/A')}）："
                "少量种子删除即丧失显著性，表明结论高度依赖少数运行；"
                "若效应已超出等价界，则需要更多数据而非仅仅更多种子。"
                for entry in fragile
            )
        )
    if not_evaluable:
        advisories.append(
            "以下显著对比无法评估删除种子后的显著性是否存活"
            "（剩余观测不足最小种子数）：不能把「未发现脆弱性」"
            "读成稳健结论，也不能当作「删除后仍显著」："
            + "; ".join(
                f"{entry['baseline']}→{entry['variant']}"
                f"（n={entry.get('n')}"
                + (
                    f", MDE≈{entry.get('equivalence_margin_needed', '?')}"
                    if entry.get("equivalence_margin_needed") is not None
                    else ""
                )
                + "）："
                "剩余种子不足以进行脆弱性判定；"
                "建议增加种子数后重新评估脆弱性。"
                for entry in not_evaluable
            )
        )
    if direction_mixed:
        advisories.append(
            "以下显著对比的配对差值方向不一致（均值显著，但"
            "并非多数种子同向；精确符号检验与 t 检验口径"
            "不同）："
            + "; ".join(
                f"{entry['baseline']}→{entry['variant']}"
                f"（正/负={entry['n_positive']}/"
                f"{entry['n_negative']}"
                + (
                    f"，MDE≈{entry.get('equivalence_margin_needed', '?')}"
                    if entry.get("equivalence_margin_needed") is not None
                    else ""
                )
                + ")："
                "效应方向在种子间不一致，结论不稳健；"
                "建议增加种子数或检查种子异常来源。"
                for entry in direction_mixed
            )
        )
    if not advisories and entries:
        significant_entries = [
            entry for entry in entries
            if entry.get("significant")
        ]
        advisories.append(
            "已检查的显著配对结果未发现单种子脆弱性；"
            "该检查是描述性的，不替代显著性检验"
            if significant_entries
            else "没有校正后显著的配对结果可供脆弱性判定："
            "不要把「未发现脆弱性」读成「存在稳健效应」"
        )
    return {
        "entries": entries,
        "verdict_counts": verdict_counts,
        "fragile": fragile,
        "direction_mixed": direction_mixed,
        "not_evaluable": not_evaluable,
        "advisories": advisories,
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
        # 还原效应与标准误时若越过 float 上限则置空，不能写 inf。
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
        if delta == 0.0:
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



def _t_critical(
    df: float,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Optional[float]:
    """
    Student t 分布在显著性水平 alpha（双侧）下的临界值
    t_{1-alpha/2}：对 p 值函数做二分求根，仍然不依赖 SciPy。
    """
    try:
        df_value = float(df)
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        math.isnan(df_value)
        or df_value <= 0.0
        or not math.isfinite(alpha_value)
        or not 0.0 < alpha_value < 1.0
    ):
        return None


    # Special case: df=1 (Cauchy distribution)
    # P(|T| > |t|) = 1 - (2/pi) * arctan(|t|), so critical = cot(pi*alpha/2)
    if df_value == 1.0:
        critical = 1.0 / math.tan(math.pi * alpha_value / 2.0)
        return critical
    # Special case: df=2
    # t = (1-alpha)*sqrt(max(2/(alpha*(2-alpha, 0.0))))
    if df_value == 2.0:
        critical = (1.0 - alpha_value) * math.sqrt(
            2.0 / (alpha_value * (2.0 - alpha_value))
        )
        return critical
            
    low = 0.0
    high = 1.0
    while True:
        p_high = _two_sided_t_p_value(high, df_value)
        if p_high is None:
            return None
        if p_high <= alpha_value:
            break
        high *= 2.0
        # 临界值可以合法地远大于 1e6（如 df=0.5、
        # alpha=1e-100 时约 4e199），旧的 1e6 上限会让这些
        # 输入直接变成 None；而且旧代码在扩大后不先算 p 值
        # 就返回，连临界值仍小于 1e6 的 df=1、alpha=1e-6
        # 也会被误判为不可用。按 double 可表示范围设上限，
        # 下一轮先比较 p 值再决定是否继续。
        if not math.isfinite(high) or high > 1.0e300:
            return None

    for _ in range(200):
        mid = (low + high) / 2.0
        p_mid = _two_sided_t_p_value(mid, df_value)
        if p_mid is None:
            return None
        if p_mid > alpha_value:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0



def _contrast_interval(
    test: Dict[str, object],
) -> Dict[str, object]:
    """
    由 t 检验结果计算效应量的 95% 置信区间。

    区间不包含 0 与 p <= 0.05 在数学上等价（同一个 t 检验），
    但 CI 直接给出效应量的可行范围：极宽的区间说明现有种子数
    只能排除「差异为 0」，还不足以定位效应大小。
    """
    effect = _finite_float(test.get("effect"))
    se = _finite_float(test.get("se"))
    df = _finite_float(test.get("df"))
    if effect is None or se is None or df is None or se < 0.0:
        return {}

    critical = _t_critical(df)
    if critical is None:
        return {}

    half_width = critical * se
    low = effect - half_width
    high = effect + half_width
    # 合法但极大的有限 effect/se 会让区间端点越过 float 上限。
    # 此时没有可落盘的有限区间；省略 95% CI，而不是写入
    # ±inf（非标准 JSON，也会把「数值上无穷宽」伪装成一个
    # 可以计算的端点）。
    if not math.isfinite(low) or not math.isfinite(high):
        return {}
    return {
        "effect": effect,
        "se": se,
        "t_critical": critical,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_excludes_zero": (low > 0.0) or (high < 0.0),
    }



def _attach_family_adjusted_intervals(
    items: Sequence[Dict[str, object]],
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> None:
    """
    为同一检验家族附加 Bonferroni 同时（simultaneous）置信区间。

    逐条对比报告的点估计 95% CI 是**逐检验**区间：它对应未校正
    的 p 值，而不是 Holm 校正后的判定。家族里有 m 个检验时，
    读者可能看到「95% CI 不含 0，但 significant=false」——这不是
    实现矛盾，而是逐检验区间与族错误率控制回答了两个不同问题。
    为了不把这种差异留给读者猜测，这里另给一套同时区间：

    - 置信水平 ``1 - alpha/m``（Bonferroni），保证整个家族参数
      同时被覆盖的概率至少 ``1 - alpha``；
    - 区间不含 0 ⇔ Bonferroni 校正后显著，因此它是比 Holm 更
      保守的**充分条件**，而不是与 Holm 判定冲突的第二套判定；
    - 逐检验 95% CI 继续保留，用于描述效应量的可行范围，不能
      拿它替代多重比较校正后的显著性。

    记录 ``ci_family_adjusted_*``；缺少 effect/se/df 的条目跳过。
    """
    eligible: List[
        Tuple[Dict[str, object], float, float, float]
    ] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        effect = _finite_float(item.get("effect"))
        if effect is None:
            effect = _finite_float(item.get("mean_effect"))
        if effect is None:
            effect = _finite_float(item.get("delta"))
        se = _finite_float(item.get("se"))
        df = _finite_float(item.get("df"))
        if (
            effect is None
            or se is None
            or df is None
            or se <= 0.0
            or df <= 0.0
        ):
            continue
        eligible.append((item, effect, se, df))

    family_size = sum(
        1
        for item in items
        if isinstance(item, dict)
        and _finite_float(item.get("p_value")) is not None
    )
    if family_size == 0:
        family_size = len(eligible)
    if family_size == 0:
        return
    adjusted_alpha = float(alpha) / float(family_size)
    for item, effect, se, df in eligible:
        critical = _t_critical(df, adjusted_alpha)
        if critical is None:
            continue
        low = effect - critical * se
        high = effect + critical * se
        # 同时区间端点也可能因超大有限 se 溢出。无法用有限数
        # 表示时不写该套区间，避免把 ±inf 落盘；逐检验区间与
        # 原始检验结果仍保留。
        if not math.isfinite(low) or not math.isfinite(high):
            continue
        excludes_zero = bool(low > 0.0 or high < 0.0)
        item["ci_family_adjusted_low"] = low
        item["ci_family_adjusted_high"] = high
        item["ci_family_adjusted_alpha"] = adjusted_alpha
        item["ci_family_adjusted_family_size"] = family_size
        item["ci_family_adjusted_critical"] = critical
        item["ci_family_adjusted_method"] = (
            "bonferroni_simultaneous_t"
        )
        item["ci_family_adjusted_excludes_zero"] = excludes_zero
        # 同时区间不含 0 必然蕴含 Holm 显著（Bonferroni 比
        # Holm 保守）；反向不成立。若出现反向，说明区间与
        # 判定用了不同口径，必须在报告里暴露而不是静默。
        item["ci_family_adjusted_decision_consistent"] = bool(
            (not excludes_zero) or bool(item.get("significant"))
        )
        item["ci_family_adjusted_note"] = (
            "同时区间按 Bonferroni α/m 构造（m="
            f"{family_size}），比 Holm step-down 保守；"
            "不含 0 是校正后显著的充分条件，但不是必要条件。"
            "逐检验 95% CI 仍只用于描述效应量范围。"
        )



def _normal_cdf(value: float) -> float:
    """标准正态分布 CDF（math.erfc 实现，无 SciPy 依赖）。

    负半轴直接用 erfc 计算左尾：``0.5 * (1 + erf(z))`` 在
    z 略小于 -6 时会把 1 + erf(z) 抵消成 0，把仍然可表示的
    尾概率（如 z=-10 时约 7.6e-24）误报为 0。功效积分在极小
    alpha 或大非中心参数下正依赖这一尾概率，必须保留。
    """
    scaled = value / math.sqrt(2.0)
    if scaled < 0.0:
        return 0.5 * math.erfc(-scaled)
    return 1.0 - 0.5 * math.erfc(scaled)



def _normal_interval_probability(low: float, high: float) -> float:
    """P(low < N(0,1) <= high)，在两端同为正值时避免相减抵消。"""
    if low >= high:
        return 0.0
    if low > 0.0:
        return (
            0.5 * math.erfc(low / math.sqrt(2.0))
            - 0.5 * math.erfc(high / math.sqrt(2.0))
        )
    return _normal_cdf(high) - _normal_cdf(low)



@functools.lru_cache(maxsize=64)
def _t_power_nodes(
    df: float,
    u_max: Optional[float] = None,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    t 分布功效积分的 Simpson 节点与权重（按 df 缓存）。

    T = (Z + δ)/sqrt(max(V/df, 0.0))，Z ~ N(0,1)，V ~ chi2_df。令
    u = sqrt(V)，则 u 的密度为
    g(u) = 2^(1-df/2) u^(df-1) exp(-u^2/2) / Γ(df/2)，
    且 P(|T| > c | u) = Φ(-c·u/√df - δ) + 1 - Φ(c·u/√df - δ)。

    默认积分区间上界取 chi 分布众数 sqrt(max(df-1, 0.0)) 加
    _T_POWER_TAIL（chi 分布的宽度约 1/√2，右尾可安全截断）；
    节点数随区间长度自适应，保证 df 从 0.5 到数千都足够精确。
    ``u_max`` 可显式扩展上界：低自由度、大非中心参数的非中心 t
    CDF 其正态尾转折可能落在默认截断之外，需要按转折位置扩界
    而不是误把整段概率质量截掉。

    对 df<3，chi 密度在 u=0 附近以 u^(df-1) 变化；df 接近 1
    时导数无界，固定 u 网格上的 Simpson 会少积概率质量
    （df=1.1 时功效误差可达 4%）。此时改用 u=t^p：积分变为
    t^(p·df-1)·exp(-t^(2p)/2)·p 的形式，选 p 使 p·df-1≥2，
    被积函数在 0 附近足够光滑，再在 t 网格上做 Simpson。
    df≥3 时原网格误差已很小，保持历史路径与数值。

    返回 (u_i, simpson_weight_i · g(u_i))；密度用对数域计算，
    df 较大时也不会溢出。
    """
    mode = math.sqrt(df - 1.0) if df > 1.0 else 0.0
    if u_max is None:
        u_max = mode + _T_POWER_TAIL
    log_gamma = math.lgamma(df / 2.0)
    log_const = (1.0 - df / 2.0) * math.log(max(2.0, 1e-10)) - log_gamma

    if df < 3.0:
        exponent = max(
            5, min(1000, int(math.ceil(3.0 / df)))
        )
        t_max = u_max ** (1.0 / exponent)
        minimum_intervals = (
            _T_POWER_SMALL_DF_MIN_INTERVALS
            if df < 1.0
            else _T_POWER_MIN_INTERVALS
        )
        intervals = max(
            minimum_intervals,
            math.ceil(t_max * 24.0),
        )
        intervals = min(intervals, _T_POWER_MAX_INTERVALS)
        if intervals % 2:
            intervals += 1

        step = t_max / intervals
        u_values: List[float] = []
        weighted: List[float] = []
        for index in range(intervals + 1):
            t = index * step
            log_u = exponent * math.log(max(t, 1e-10)) if t > 0.0 else float("-inf")
            u = math.exp(log_u) if log_u >= -745.0 else 0.0
            if index == 0 or index == intervals:
                simpson = step / 3.0
            elif index % 2 == 0:
                simpson = 2.0 * step / 3.0
            else:
                simpson = 4.0 * step / 3.0

            if t <= 0.0:
                weight = 0.0
            else:
                log_weight = (
                    math.log(max(simpson, 1e-10))
                    + log_const
                    + (df - 1.0) * log_u
                    - u * u / 2.0
                    + math.log(max(exponent, 1e-10))
                    + (exponent - 1.0) * math.log(max(t, 1e-10))
                )
                if log_weight <= -745.0:
                    weight = 0.0
                elif log_weight >= 709.0:
                    weight = float("inf")
                else:
                    weight = math.exp(log_weight)
            u_values.append(u)
            weighted.append(weight)

        return tuple(u_values), tuple(weighted)

    intervals = max(
        _T_POWER_MIN_INTERVALS,
        math.ceil(u_max * 24.0),
    )
    intervals = min(intervals, _T_POWER_MAX_INTERVALS)
    if intervals % 2:
        intervals += 1

    step = u_max / intervals
    u_values = []
    weighted = []
    for index in range(intervals + 1):
        u = index * step
        if index == 0 or index == intervals:
            simpson = step / 3.0
        elif index % 2 == 0:
            simpson = 2.0 * step / 3.0
        else:
            simpson = 4.0 * step / 3.0

        if u <= 0.0:
            # u^(df-1)：df>1 时为 0；df=1 时 g(0)=sqrt(max(2/π, 0.0))。
            density = math.sqrt(2.0 / math.pi) if df == 1.0 else 0.0
        else:
            density = math.exp(
                log_const + (df - 1.0) * math.log(max(u, 1e-10)) - u * u / 2.0
            )
        u_values.append(u)
        weighted.append(simpson * density)

    return tuple(u_values), tuple(weighted)



def _t_power_large_shift_normal_mix(
    df_value: float,
    scale: float,
    noncentrality: float,
    intervals: int = 2400,
) -> float:
    """
    小自由度下的功效：对正态位移积分 chi CDF。

    令 T'=(Z+δ)/sqrt(max(V/df, 0.0))、w=scale·u 且 u=sqrt(V)，则
    |T'|>critical 等价于 u<|Z+δ|/scale。因此

        power = E_Z[F_chi2_df((|Z+δ|/scale)^2)]。

    这比对 w 做等距 Simpson 更适合 δ 很大（如 1e5）的情形：
    w 积分的转折宽度约为 1，而区间长度约为 δ；固定 20000
    个区间时步长可达数个单位，会把 δ 附近的概率质量积错。
    这里对 Z 在 ±12 内积分，转折宽度不变，步长始终为 0.01。
    df 在 3–10 且 scale 不大时也走这里：u 网格在 0 附近受
    u^(df-1) 曲率影响，固定 Simpson 节点的误差可达 1e-5。
    """
    log_scale = math.log(max(scale, 1e-10))
    shape = df_value / 2.0
    log_gamma_plus_one = math.lgamma(shape + 1.0)

    def _chi_square_cdf_at_log(log_u: float) -> float:
        if not math.isfinite(log_u):
            return 0.0
        log_x = 2.0 * log_u - math.log(max(2.0, 1e-10))
        if log_x >= 700.0:
            # x = exp(log_x) 已经大到超出 float 范围；对任何
            # 正自由度，chi-square CDF 在该点都等于 1 到双精度。
            # 不能继续做 math.exp(log_x)（会抛 OverflowError
            # 中断整份功效/等价性报告）。
            return 1.0
        if log_x < -700.0:
            # u 极小，u² 可能下溢为 0；用 x^a/Γ(a+1) 的
            # 对数形式保留仍可表示的概率。
            log_probability = (
                -math.exp(log_x)
                + shape * log_x
                - log_gamma_plus_one
            )
            return (
                math.exp(log_probability)
                if log_probability > -745.0
                else 0.0
            )
        value = _chi_square_cdf(
            2.0 * math.exp(log_x), df_value
        )
        return 0.0 if value is None else value

    total = 0.0
    low = -12.0
    high = 12.0
    step = (high - low) / intervals
    for index in range(intervals + 1):
        t = low + index * step
        if index == 0 or index == intervals:
            simpson = step / 3.0
        elif index % 2 == 0:
            simpson = 2.0 * step / 3.0
        else:
            simpson = 4.0 * step / 3.0
        y = noncentrality + t
        if y == 0.0:
            continue
        log_u = math.log(abs(y)) - log_scale
        density = math.exp(-0.5 * t * t) / math.sqrt(
            2.0 * math.pi
        )
        total += (
            simpson * density * _chi_square_cdf_at_log(log_u)
        )
    return min(1.0, max(0.0, total))



def _t_power_small_df_large_scale(
    df_value: float,
    scale: float,
    noncentrality: float,
) -> float:
    """
    df<3 且 scale=critical/√df>10 时的功效积分。

    固定 u 网格的分辨率是 u_max/N；alpha 极小时 critical 可达
    1e19 以上，被积函数的质量集中在 u<1/scale，固定网格会直接
    漏掉整段概率质量并返回 0。这里换到 w=scale·u：先对
    w∈[0,1] 用 w=t^p 吸收 chi 密度在 0 附近的弱奇异项，再对
    w∈[1,w_max] 用等距网格解析正态尾概率的转折。

    δ 很大时 w 积分在 δ 附近的转折会被 20000 个区间的上限
    稀释（区间长度约 δ，步长可达数个 w 单位）；此时改用对
    正态位移的 chi-CDF 积分。
    """
    if not math.isfinite(noncentrality):
        return 1.0
    exponent = max(5, min(1000, int(math.ceil(3.0 / df_value))))
    log_const = (
        (1.0 - df_value / 2.0) * math.log(max(2.0, 1e-10))
        - math.lgamma(df_value / 2.0)
    )
    log_scale = math.log(max(scale, 1e-10))

    def _power_factor(w: float) -> float:
        return (
            _normal_cdf(-w - noncentrality)
            + _normal_cdf(noncentrality - w)
        )

    terms = []
    # Use higher resolution for df < 1.0 where the chi density singularity
    # at u=0 (g(u) ~ u^{df-1}) requires more intervals to resolve;
    # increased from _T_POWER_SMALL_DF_MIN_INTERVALS (1600) to provide
    # sufficient precision for tiny-alpha cases.
    if df_value < 1.0:
        intervals = _T_POWER_MAX_INTERVALS  # 4000, for better resolution
    else:
        intervals = _T_POWER_MIN_INTERVALS
    step = 1.0 / intervals
    for index in range(intervals + 1):
        t = index * step
        if index == 0 or index == intervals:
            simpson = step / 3.0
        elif index % 2 == 0:
            simpson = 2.0 * step / 3.0
        else:
            simpson = 4.0 * step / 3.0
        if t <= 0.0:
            # Integrand is 0 at t=0 for df>0 (exponent*df-1 > 0 after transform);
            # include the endpoint with weight 0 to keep Simpson's rule correct.
            continue
        log_w = exponent * math.log(max(t, 1e-10))
        log_u = log_w - log_scale
        w = math.exp(log_w) if log_w >= -745.0 else 0.0
        factor = _power_factor(w)
        if factor <= 0.0:
            continue
        log_weight = (
            math.log(max(simpson, 1e-10))
            + log_const
            + (df_value - 1.0) * log_u
            - math.exp(2.0 * log_u) / 2.0
            - log_scale
            + math.log(max(exponent, 1e-10))
            + (exponent - 1.0) * math.log(max(t, 1e-10))
            + math.log(max(factor, 1e-10))
        )
        if log_weight > -745.0:
            terms.append(math.exp(log_weight))

    u_max = (
        math.sqrt(df_value - 1.0) + _T_POWER_TAIL
        if df_value > 1.0
        else _T_POWER_TAIL
    )
    w_max = min(
        scale * u_max,
        noncentrality + _T_POWER_TAIL,
    )
    if not math.isfinite(w_max):
        return 1.0
    if w_max > 1.0:
        intervals = max(
            200,
            int(math.ceil((w_max - 1.0) / 0.02)),
        )
        intervals = min(intervals, 20000)
        if intervals % 2:
            intervals += 1
        step = (w_max - 1.0) / max(intervals, 1)  # guard against zero intervals
        log_w_max = math.log(w_max)
        for index in range(intervals + 1):
            w = 1.0 + index * step
            if index == 0 or index == intervals:
                simpson = step / 3.0
            elif index % 2 == 0:
                simpson = 2.0 * step / 3.0
            else:
                simpson = 4.0 * step / 3.0
            factor = _power_factor(w)
            if factor <= 0.0:
                continue
            # Compute log_u in log space to avoid underflow when scale is huge
            log_u = math.log(w) - log_scale
            log_weight = (
                math.log(max(simpson, 1e-10))
                + log_const
                + (df_value - 1.0) * log_u
                - math.exp(2.0 * log_u) / 2.0
                - log_scale
                + math.log(max(factor, 1e-10))
            )
            if log_weight > -745.0:
                terms.append(math.exp(log_weight))
    total = math.fsum(terms)
    return min(1.0, max(0.0, total))



def _t_power(
    effect: float,
    se: float,
    df: float,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Optional[float]:
    """
    双侧 t 检验在给定效应量下的统计功效（非中心 t 分布）。

    非中心参数 δ = |effect|/se；功效 = P(|T| > t 临界值)。
    效应量与 df 不可用时返回 None。数值积分与 SciPy 的
    ``nct`` 口径一致：δ=0 时功效恰为 alpha；δ 增大时单调
    趋近 1。
    """
    try:
        effect_value = abs(float(effect))
        se_value = float(se)
        df_value = float(df)
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(effect_value)
        or not math.isfinite(se_value)
        or se_value < 0.0
        or math.isnan(df_value)
        or not (0.0 < alpha_value < 0.5)
    ):
        return None
    if df_value <= 0.0:
        return None
    if se_value <= 0.0:
        # 零标准误：非零效应必然被检出；零效应必然不显著。
        return 1.0 if effect_value > 0.0 else alpha_value
    if effect_value == 0.0:
        # 非中心参数恰为 0 时低功效积分在数值上仍可能因网格
        # 截断而偏离 alpha；这是可精确知道的值，直接返回。
        return alpha_value

    noncentrality = effect_value / se_value
    if math.isinf(noncentrality):
        # se 小到 δ 溢出：非中心参数趋于无穷，任何有限临界值
        # 都必然被超过。不能把 inf 带进正态混合积分（那里的
        # chi CDF 在 inf 处会返回 0，反而给出零功效）。
        return 1.0
    if (
        not math.isfinite(df_value)
        or df_value >= _T_POWER_NORMAL_DF
    ):
        # 自由度趋于无穷时非中心 t 退化为均值 δ 的正态分布；
        # 同时避开大 df 下固定 u 网格漏掉 chi 密度尖峰的问题。
        critical = _t_critical(float("inf"), alpha_value)
        if critical is None:
            return None
        power = (
            _normal_cdf(-critical - noncentrality)
            + _normal_cdf(noncentrality - critical)
        )
        return min(1.0, max(0.0, power))

    critical = _t_critical(df_value, alpha_value)
    if critical is None:
        return None

    scale = critical / math.sqrt(max(0.0, df_value))

    if scale > 10.0 and noncentrality > 16.0:
        return _t_power_large_shift_normal_mix(
            df_value, scale, noncentrality
        )

    if (
        3.0 <= df_value < 10.0
        and 2.0 < scale <= 10.0
    ):
        # df 在 3–10、临界值较大时，u 网格上的正态转折宽度
        # 约为 1/scale，800–4000 个 Simpson 节点只能把功效
        # 算到约 1e-5 相对误差。改对正态位移积分 chi CDF，
        # 1200 个区间即可稳定到约 1e-9。
        return _t_power_large_shift_normal_mix(
            df_value,
            scale,
            noncentrality,
            intervals=1200,
        )

    if scale > 10.0 and df_value >= 3.0:
        # alpha 很小 / df 较小时临界值巨大，固定 u 网格在
        # u≈0 附近完全分辨不出被积函数的尖峰，会低估功效。
        # 换到 w = scale·u 后按 w 做 Simpson：被积函数在
        # w≈0 附近变成多项式尺度，且 w > δ+12 后正态尾概率
        # 已可忽略，可安全截断。
        #
        # df<3 时改走 ``_t_power_nodes`` 的 t=u^(1/p) 变换：
        # chi 密度在 0 附近的弱奇异项已被变换吸收，既不依赖
        # 固定 u 网格，也不需要尖峰尺度重参数化。
        u_max = (
            math.sqrt(df_value - 1.0) + _T_POWER_TAIL
            if df_value > 1.0
            else _T_POWER_TAIL
        )
        w_max = min(
            scale * u_max,
            noncentrality + _T_POWER_TAIL,
        )
        target_step = 0.02 * min(1.0, scale)
        intervals = max(
            200,
            int(math.ceil(w_max / target_step)),
        )
        if intervals % 2:
            intervals += 1
        intervals = min(intervals, 20000)
        step = w_max / intervals
        log_const = (
            (1.0 - df_value / 2.0) * math.log(max(2.0, 1e-10))
            - math.lgamma(df_value / 2.0)
        )
        total = 0.0
        for index in range(intervals + 1):
            w = index * step
            if index == 0 or index == intervals:
                simpson = step / 3.0
            elif index % 2 == 0:
                simpson = 2.0 * step / 3.0
            else:
                simpson = 4.0 * step / 3.0
            if w <= 0.0:
                # w=0 corresponds to u=0; density is 0 for df>=3
                density = 0.0
            else:
                # Compute log_u in log space to avoid underflow when scale is huge
                log_u = math.log(w) - log_scale
                density = math.exp(
                    log_const
                    + (df_value - 1.0) * log_u
                    - math.exp(2.0 * log_u) / 2.0
                )
            total += (
                simpson
                * density
                * (
                    _normal_cdf(-w - noncentrality)
                    + _normal_cdf(noncentrality - w)
                )
                / scale
            )
        return min(1.0, max(0.0, total))

    if scale > 10.0:
        return _t_power_small_df_large_scale(
            df_value,
            scale,
            noncentrality,
        )

    u_values, weighted = _t_power_nodes(df_value)
    total = 0.0
    for u, weight in zip(u_values, weighted):
        if weight == 0.0:
            continue
        shift = scale * u
        total += weight * (
            _normal_cdf(-shift - noncentrality)
            + _normal_cdf(noncentrality - shift)
        )
    return min(1.0, max(0.0, total))



# 求解最小可检出效应时的初始非中心参数上界。alpha 经家族
# 校正后可能很小，δ=100 未必足够（如 df=1、alpha=1e-6 时
# 临界值本身约 6.4e5），因此还会按临界值扩大并继续倍增。
_MDE_MAX_NONCENTRALITY = 100.0



def _minimum_detectable_effect(
    se: float,
    df: float,
    target_power: float = _TARGET_POWER,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Optional[float]:
    """
    当前设计（se、df 给定）在目标功效下能检出的最小真实
    效应量（绝对口径，与指标同单位）。

    对非中心参数 δ 二分：δ 越大功效越高。观测到的 |Δ| 小于
    MDE 时，「不显著」是功效不足的预期结果，不能解读为
    「无差异」；要在当前种子数下缩小 MDE，需要增加随机种子
    或降低种子间方差。
    """
    try:
        se_value = float(se)
        df_value = float(df)
        target = float(target_power)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(se_value)
        or se_value < 0.0
        or math.isnan(df_value)
        or not math.isfinite(target)
        or df_value <= 0.0
        or not (0.0 < target < 1.0)
    ):
        return None
    if se_value <= 0.0:
        # 零标准误：任意非零效应都以功效 1 被检出。
        return 0.0

    critical = _t_critical(df_value, alpha)
    if critical is None:
        return None

    # 功效只取决于 δ=effect/se，统一在 se=1 的尺度上计算，
    # 避免 effect=mid·se 在大 se 下先溢出成 inf。初始上界取
    # 临界值的两倍；目标功效更高时再倍增到足够大。
    low = 0.0
    high = max(_MDE_MAX_NONCENTRALITY, 2.0 * critical)
    if not math.isfinite(high):
        return None
    while True:
        power_high = _t_power(high, 1.0, df_value, alpha)
        if power_high is None:
            return None
        if power_high >= target:
            break
        low = high
        high *= 2.0
        if not math.isfinite(high) or high > 1.0e300:
            return None

    for _ in range(80):
        mid = (low + high) / 2.0
        power_mid = _t_power(mid, 1.0, df_value, alpha)
        if power_mid is None:
            return None
        if power_mid < target:
            low = mid
        else:
            high = mid
    mde = ((low + high) / 2.0) * se_value
    return mde if math.isfinite(mde) else None



def _seeds_to_detect(
    effect: float,
    sd: float,
    target_power: float = _TARGET_POWER,
    alpha: float = _SIGNIFICANCE_LEVEL,
    max_seeds: int = 128,
    design: str = "paired",
) -> Optional[int]:
    """
    要在目标功效下检出给定效应量所需的种子数（近似：把观测
    到的种子间标准差当作真值）。

    ``design="paired"``：同一种子上的配对比较，n 为配对数，
    se = sd/√n、df = n-1。

    ``design="welch"``：两臂独立种子、每臂 n 个（等分配），
    se = sd·√(2/ max(n, 1))、df ≈ 2(n-1)。配对公式会低估所需种子数，
    不能把配对口径直接套到 Welch 对比上。

    用于回答「不显著到底是没效应还是种子太少」：若当前种子
    数远小于所需种子数，结论只能是「该规模下无法判定」。
    sd=0（无种子间方差）返回 1；效应为 0 时返回 None。
    """
    try:
        effect_value = abs(float(effect))
        sd_value = float(sd)
        target = float(target_power)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(effect_value)
        or not math.isfinite(sd_value)
        or sd_value < 0.0
        or not math.isfinite(target)
        or not (0.0 < target < 1.0)
    ):
        return None
    if effect_value <= 0.0:
        return None
    if sd_value <= 0.0:
        return 1
    if design not in ("paired", "welch"):
        raise ValueError(
            "所需种子数的设计口径必须是 paired 或 welch"
        )

    for n in range(_MIN_SEEDS_FOR_STATS, max_seeds + 1):
        if design == "welch":
            se = sd_value * math.sqrt(2.0 / float(n))
            df = float(2 * (n - 1))
        else:
            se = sd_value / math.sqrt(float(n))
            df = float(n - 1)
        power = _t_power(
            effect_value,
            se,
            df,
            alpha,
        )
        if power is not None and power >= target:
            return n
    return None



def _t_cdf(t: float, df: float) -> Optional[float]:
    """
    Student t 分布 CDF F(t)：由双侧尾概率换算（无 SciPy 依赖）。

    P(|T| >= |t|) = 2·(1 - F(|t|))，因此 t>0 时
    F(t) = 1 - p(|t|)/2，t<0 时 F(t) = p(|t|)/2。
    """
    two_sided = _two_sided_t_p_value(t, df)
    if two_sided is None:
        return None
    if t >= 0.0:
        return 1.0 - two_sided / 2.0
    return two_sided / 2.0



def _noncentral_t_cdf_large_shift(
    value: float,
    df: float,
    noncentrality: float,
) -> float:
    """大 ``|value|/sqrt(df)`` 下按正态参数积分非中心 t CDF。"""
    a = value / math.sqrt(max(0.0, df))
    # 积分变量换成 x=a*u-δ 后，正态 CDF 只在 x∈[-40,40] 内
    # 有可表示的质量；对应 u 的上界是 (40+δ)/a（a>0）或
    # (40-δ)/(-a)（a<0）。默认 u_max=众数+12 只覆盖 chi 密度
    # 本身，非中心参数很大时转折点会落在更远的 chi 尾部：
    # 例如 df=766、δ≈97、t≈52.6 时主导区间在 u≈44，而默认
    # 上界仅 39.7，会把 CDF 低估十几个数量级。按转折点上界
    # 扩界后再做 x 积分，剩余尾部仍由下面的 chi CDF/生存函数
    # 精确补上。
    u_max = math.sqrt(max(df - 1.0, 0.0)) + _T_POWER_TAIL
    if a > 0.0:
        tail_cutoff = (40.0 + noncentrality) / a
    elif a < 0.0:
        tail_cutoff = (40.0 - noncentrality) / (-a)
    else:
        tail_cutoff = None
    if (
        tail_cutoff is not None
        and math.isfinite(tail_cutoff)
        and tail_cutoff > u_max
    ):
        u_max = tail_cutoff
    log_const = (
        (1.0 - df / 2.0) * math.log(max(2.0, 1e-10))
        - math.lgamma(df / 2.0)
    )

    def density(u: float) -> float:
        if u <= 0.0:
            return 0.0
        log_density = (
            log_const
            + (df - 1.0) * math.log(max(u, 1e-10))
            - u * u / 2.0
        )
        return (
            math.exp(log_density)
            if log_density > -745.0
            else 0.0
        )

    def normal_cdf(argument: float) -> float:
        return 0.5 * math.erfc(-argument / math.sqrt(2.0))

    total = 0.0
    if a > 0.0:
        low = max(-noncentrality, -40.0)
        high = min(a * u_max - noncentrality, 40.0)
        if high > low:
            width = high - low
            step = min(0.02, a / 50.0)
            intervals = max(2, math.ceil(width / step))
            if intervals % 2:
                intervals += 1
            step = width / intervals
            for index in range(intervals + 1):
                argument = low + index * step
                u = (argument + noncentrality) / a
                integrand = (
                    density(u) / a * normal_cdf(argument)
                )
                if index == 0 or index == intervals:
                    total += step / 3.0 * integrand
                elif index % 2 == 0:
                    total += 2.0 * step / 3.0 * integrand
                else:
                    total += 4.0 * step / 3.0 * integrand

        cutoff = (40.0 + noncentrality) / a
        if cutoff < u_max:
            if cutoff <= 0.0:
                total += 1.0
            elif cutoff >= math.sqrt(sys.float_info.max):
                # cutoff² 超出 float 范围；df < _T_POWER_NORMAL_DF
                # 时 chi-square 在该点的上尾已远低于可表示范围。
                total += 0.0
            else:
                survival = _chi_square_survival(
                    cutoff * cutoff,
                    df,
                )
                if survival is not None:
                    total += survival
    else:
        scale = -a
        low = max(
            -scale * u_max - noncentrality,
            -40.0,
        )
        high = min(-noncentrality, 40.0)
        if high > low:
            width = high - low
            step = min(0.02, scale / 50.0)
            intervals = max(2, math.ceil(width / step))
            if intervals % 2:
                intervals += 1
            step = width / intervals
            for index in range(intervals + 1):
                argument = low + index * step
                u = (argument + noncentrality) / a
                integrand = (
                    density(u) / scale
                    * normal_cdf(argument)
                )
                if index == 0 or index == intervals:
                    total += step / 3.0 * integrand
                elif index % 2 == 0:
                    total += 2.0 * step / 3.0 * integrand
                else:
                    total += 4.0 * step / 3.0 * integrand

        cutoff = (-40.0 - noncentrality) / scale
        if cutoff > 0.0:
            if cutoff >= math.sqrt(sys.float_info.max):
                # cutoff² 超出 float 范围；df < _T_POWER_NORMAL_DF
                # 时该点以下的 chi-square 概率在双精度下为 1。
                total += 1.0
            else:
                lower = _chi_square_cdf(
                    min(cutoff, u_max) ** 2,
                    df,
                )
                if lower is not None:
                    total += lower
    return min(1.0, max(0.0, total))


# ---------------------------------------------------------------------------
# 贝叶斯补充分析（P0，--bayesian）
#
# 目标：在频率派判定（Holm 校正 + TOST）之外，并行提供证据强度
# 度量。默认关闭、不改变任何现有判定，只在显式开启时计算。
#
# 模型：JZS（Jeffreys-Zellner-Siow）默认先验，标准化效应
# δ ~ Cauchy(0, r)（等价地 g ~ InvGamma(1/2, r²/2)）。单侧/配对
# 设计下 Bayes 因子有 Rouder et al. (2009) 的一维积分形式：
#
#   BF10 = ∫₀^∞ (1+Ng)^(-1/2)·(1+t²/((1+Ng)ν))^(-N/2)·π(g) dg
#          / (1+t²/ν)^(-N/2)
#
# 其中 N 为有效样本量（配对：种子对数；Welch：n₁n₂/(n₁+n₂)），
# ν 为 t 检验自由度，π(g) = r/√(2π)·g^(-3/2)·exp(-r²/(2g))。
# 作代换 g = 1/u² 后积分化为 [0,∞) 上光滑函数
#
#   (2r/√(2π))·∫₀^∞ exp(-r²u²/2)·L(1/u²) du
#
# （u→0 时 L ∝ u/√N → 0，u→∞ 时高斯项迅速衰减），用复合
# Simpson 求积即可精确到机器精度量级，无 SciPy 依赖。
#
# 后验量（P(δ>0|D)、分位数、HDI）不用近似 σ²：给定 g 时
# σ²|D,g ~ InvGamma(N/2, S_g/2)（注意形状是 N/2，因为 Jeffreys
# 先验比样本方差多一个自由度），μ|σ²,D,g ~ N(m_g, v_g·σ²)，
# 因此
#
#   δ = μ/σ | D, g = c_g·√(Y/N) + √(v_g)·Z，  Y ~ χ²_N，Z ~ N(0,1)
#
# 是 (g, Y) 上的正态混合。对 g 用后验权重 π(g)L(g)、对 Y 用
# χ²_N 密度做两层确定性求积，得到精确到网格分辨率的混合表示；
# P(δ>0) 与 CDF 由混合直接累加，HDI 用 δ 网格上的最短覆盖
# 区间计算（后验混合一般不对称，等尾区间不等于 HDI）。
# ---------------------------------------------------------------------------

_JZS_DEFAULT_PRIOR_SCALE = 0.7071067811865476  # Cauchy(0, 1/√2)「中等」先验
_BAYES_QUAD_NODES = 513  # u 空间 Simpson 节点数（奇数）
_BAYES_TAIL_LOG_CUT = 41.45  # exp(-r²u²/2) < e^-41.45 ≈ 1e-18 处截断
_BAYES_Y_SIGMA_CUT = 12.0  # χ²_N 网格覆盖到均值 + 12 个标准差
_BAYES_DELTA_SPAN_SD = 10.0  # δ 网格覆盖后验均值 ± 10 个后验标准差
_BAYES_DELTA_GRID = 401  # δ 网格点数（CDF/HDI/密度）
_BAYES_EVIDENCE_THRESHOLDS = (
    (4.6, "decisive"),
    (2.3, "strong"),
    (1.1, "substantial"),
    (0.69, "weak"),
)
_BAYES_MAX_T = 1.0e6


def _bayes_evidence_category(
    log_bf10: float,
) -> str:
    """按 |ln BF10| 给出 Jeffreys 风格的证据分级。"""
    magnitude = abs(float(log_bf10))
    target = "alternative" if log_bf10 > 0.0 else "null"
    for threshold, label in _BAYES_EVIDENCE_THRESHOLDS:
        if magnitude >= threshold:
            return f"{label}_evidence_for_{target}"
    return f"anecdotal_evidence_for_{target}"


def _jzs_u_nodes_kernel(
    t_value: float,
    n_value: float,
    df_value: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 u 空间求积节点、被积函数值与 Simpson 权重。

    被积函数 = exp(-r²u²/2)·(1+N/u²)^(-1/2)·(1+t²/((1+N/u²)ν))^(-N/2)，
    它同时是 BF10 分子积分的被积函数与 g 后验权重的（未归一化）
    核：π(g)L(g) 在 g=1/u² 代换下就是这个函数乘以常数 2r/√(2π)。

    积分区间分两段复合：[0, u_transition] 用细网格解析
    L(1/u²) 从 0 过渡到渐近值的区域（过渡尺度 ~√N），其余到
    u_max（高斯截断）用粗网格。先验尺度 r 很小时 u_max 远大于
    √N，单段网格会完全跳过过渡区，使 BF 产生百分之几的误差；
    两段复合后该误差消失。u_max ≤ 过渡点时退化为单段。
    """
    r = float(scale)
    u_max = math.sqrt(2.0 * _BAYES_TAIL_LOG_CUT) / r
    u_transition = 12.0 * math.sqrt(float(n_value))
    nodes = int(_BAYES_QUAD_NODES)
    if nodes % 2 == 0:
        nodes += 1

    def _integrand(u: "np.ndarray") -> "np.ndarray":
        values = np.zeros_like(u, dtype=np.float64)
        positive = u > 0.0
        inner = u[positive]
        one_plus_ng = 1.0 + n_value / np.square(inner)
        lam2 = 1.0 + t_value * t_value / (one_plus_ng * df_value)
        values[positive] = np.exp(-0.5 * (r * r) * np.square(inner)) * (
            one_plus_ng ** -0.5
        ) * np.power(lam2, -0.5 * float(n_value))
        return values

    def _simpson_weights(span: float, count: int) -> "np.ndarray":
        weights = np.ones(count, dtype=np.float64)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return weights * (span / max(count - 1, 1e-12) / 3.0)

    if u_max <= u_transition:
        u = np.linspace(0.0, u_max, nodes)
        kernel = _integrand(u)
        return u, kernel, _simpson_weights(u_max, nodes)
    first = np.linspace(0.0, u_transition, nodes)
    second = np.linspace(u_transition, u_max, nodes)
    u = np.concatenate((first, second[1:]))
    kernel = np.concatenate(
        (_integrand(first), _integrand(second[1:]))
    )
    weights = np.concatenate(
        (
            _simpson_weights(u_transition, nodes),
            _simpson_weights(u_max - u_transition, nodes)[1:],
        )
    )
    return u, kernel, weights


def _jzs_log_bayes_factor_10(
    t_value: float,
    n_value: float,
    df_value: float,
    scale: float = _JZS_DEFAULT_PRIOR_SCALE,
) -> Optional[float]:
    """
    计算 JZS Bayes 因子 BF10 的自然对数（配对/单样本 t 口径）。

    返回 None 表示数值上不可计算（t 非有限、自由度/样本量非法
    或积分下溢），调用方应把该对比标记为不可用而不是写入
    非有限数值。
    """
    t_value = _finite_float(t_value)
    n_value = _finite_float(n_value)
    df_value = _finite_float(df_value)
    scale = _finite_float(scale)
    if t_value is None or n_value is None or df_value is None:
        return None
    if scale is None or scale <= 0.0:
        return None
    if n_value <= 0.0 or df_value <= 0.0:
        return None
    if abs(t_value) > _BAYES_MAX_T:
        return None
    _u, kernel, weights = _jzs_u_nodes_kernel(
        t_value, n_value, df_value, scale
    )
    integral = float(np.dot(kernel, weights)) * (
        2.0 * scale / math.sqrt(2.0 * math.pi)
    )
    if not math.isfinite(integral) or integral <= 0.0:
        return None
    # 分母 (1+t²/ν)^(-N/2) 在 log 域计算，避免大 t 下溢。
    # 指数按 JZS 口径用有效样本量 N（n_value），而非自由度 ν（df_value）；
    # 括号内 t²/ν 的 ν 仍是自由度，保持不变。
    log_denominator = -0.5 * float(n_value) * math.log1p(
        t_value * t_value / df_value
    )
    log_numerator = math.log(max(integral, 1e-10))
    log_bf = log_numerator - log_denominator
    if not math.isfinite(log_bf):
        return None
    return log_bf


def _jzs_posterior_mixture(
    t_value: float,
    n_value: float,
    df_value: float,
    scale: float = _JZS_DEFAULT_PRIOR_SCALE,
) -> Optional[Dict[str, object]]:
    """
    构造标准化效应 δ = μ/σ 的后验正态混合表示。

    返回 dict：
    - ``mixture_loc`` / ``mixture_scale`` / ``mixture_weight``：
      (g, Y) 网格上每个正态分量的位置、标准差与权重（和为 1）；
      δ = loc + scale·Z。
    - ``mean`` / ``sd``：后验均值与标准差（由混合矩直接计算）。

    需要 N ≥ 2（配对设计即 ≥2 个种子对；Welch 口径由调用方
    传入 n₁n₂/(n₁+n₂)）且 t/df 有限；不满足时返回 None。
    """
    t_value = _finite_float(t_value)
    n_value = _finite_float(n_value)
    df_value = _finite_float(df_value)
    scale = _finite_float(scale)
    if (
        t_value is None
        or n_value is None
        or df_value is None
        or scale is None
        or scale <= 0.0
    ):
        return None
    if n_value < 2.0 or df_value <= 0.0:
        return None
    if abs(t_value) > _BAYES_MAX_T:
        return None
    _u, kernel, weights = _jzs_u_nodes_kernel(
        t_value, n_value, df_value, scale
    )
    total_kernel = float(np.dot(kernel, weights))
    if not math.isfinite(total_kernel) or total_kernel <= 0.0:
        return None
    # g 后验权重：u 空间 Simpson 权重 × 被积函数。u=0 节点核为
    # 0，对应 g→∞ 的零测度端点，自然被排除。
    g_weights = kernel * weights
    g_nodes = np.empty_like(kernel)
    positive = np.abs(_u) > 0.0
    g_nodes[positive] = 1.0 / np.square(_u[positive])
    keep = g_weights > 0.0
    if not bool(np.any(keep)):
        return None
    g_nodes = g_nodes[keep]
    g_weights = g_weights[keep]
    g_weights = g_weights / float(np.dot(g_weights, np.ones_like(g_weights)))
    one_plus_ng = 1.0 + n_value * g_nodes
    v_g = g_nodes / one_plus_ng
    lam2 = 1.0 + t_value * t_value / (one_plus_ng * df_value)
    c_g = (
        t_value
        * n_value
        * g_nodes
        / (math.sqrt(max(0.0, df_value)) * one_plus_ng * np.sqrt(lam2))
    )
    # σ²|D,g ~ InvGamma(N/2, S_g/2)：Y ~ χ²_N，σ² = S_g/Y，
    # m_g/σ = c_g·√(Y/N)。
    chi_df = n_value
    y_max = chi_df + _BAYES_Y_SIGMA_CUT * math.sqrt(2.0 * chi_df)
    y_nodes_count = 129
    y = np.linspace(0.0, y_max, y_nodes_count)
    y_weights = np.empty(y_nodes_count, dtype=np.float64)
    y_weights[0] = 1.0
    y_weights[-1] = 1.0
    y_weights[1:-1:2] = 4.0
    y_weights[2:-1:2] = 2.0
    y_weights = y_weights * (y_max / max(y_nodes_count - 1, 1e-12) / 3.0)
    log_density = np.full(y_nodes_count, -np.inf)
    positive_y = y > 0.0
    if chi_df > 2.0:
        log_density[positive_y] = (
            (0.5 * chi_df - 1.0) * np.log(max(y[positive_y], 1e-10))
            - 0.5 * y[positive_y]
        )
    elif chi_df == 2.0:
        log_density[positive_y] = -0.5 * y[positive_y]
        # y→0 时 χ²₂ 密度收敛到 1/2（形状参数恰为 1）。
        log_density[0] = math.log(max(0.5, 1e-10))
    else:
        # N < 2 已在上游拒绝；防御性处理，不生成有限密度。
        return None
    log_density[positive_y] = log_density[positive_y] - (
        0.5 * chi_df * math.log(max(2.0, 1e-10)) + math.lgamma(0.5 * chi_df)
    )
    y_density = np.exp(log_density)
    y_density = np.maximum(y_density, 0.0)
    y_mass = y_density * y_weights
    y_total = float(np.dot(y_mass, np.ones_like(y_mass)))
    if not math.isfinite(y_total) or y_total <= 0.0:
        return None
    y_mass = y_mass / y_total
    # 混合分量：外积 (g 节点 × Y 节点)。loc_ij = c_g·√(Y/N)，
    # scale_i = √v_g（与 Y 无关），weight_ij = w_g·p_j。
    sqrt_y_over_n = np.sqrt(y / chi_df)
    loc = np.outer(c_g, sqrt_y_over_n)
    scale_vector = np.sqrt(v_g)
    weight = np.outer(g_weights, y_mass)
    mean = float(np.sum(weight * loc))
    second_moment = float(
        np.sum(weight * (loc * loc + np.square(scale_vector)[:, None]))
    )
    variance = second_moment - mean * mean
    if not math.isfinite(mean) or not math.isfinite(variance):
        return None
    sd = math.sqrt(max(variance, 0.0))
    return {
        "mixture_loc": loc,
        "mixture_scale": scale_vector,
        "mixture_weight": weight,
        "mean": mean,
        "sd": sd,
        "n_effective": n_value,
        "df": df_value,
        "prior_scale": scale,
    }


def _normal_cdf_vector(values: "np.ndarray") -> "np.ndarray":
    """向量版标准正态 CDF（torch.special.erfc，float64）。"""
    tensor = torch.as_tensor(
        np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    )
    result = 0.5 * torch.special.erfc(-tensor / math.sqrt(2.0))
    return result.numpy()


def _jzs_posterior_summaries(
    t_value: float,
    n_value: float,
    df_value: float,
    scale: float = _JZS_DEFAULT_PRIOR_SCALE,
) -> Optional[Dict[str, object]]:
    """
    从后验混合中计算 P(δ>0|D)、后验均值/标准差与 95% HDI。

    HDI 在 δ 网格上用滑动窗口求最短 95% 覆盖区间；后验混合在
    t≠0 时不对称，等尾区间一般不是 HDI，这里直接算最短区间。
    分位数同时给出（等尾 95% 区间），供与 HDI 对照。
    """
    mixture = _jzs_posterior_mixture(
        t_value, n_value, df_value, scale
    )
    if mixture is None:
        return None
    loc = mixture["mixture_loc"]
    scale_vector = mixture["mixture_scale"]
    weight = mixture["mixture_weight"]
    mean = float(mixture["mean"])
    sd = float(mixture["sd"])
    if sd <= 0.0:
        return None
    # P(δ>0|D)：逐 (g, Y) 分量对 loc/sd 求正态 CDF 后加权求和。
    probabilities = _normal_cdf_vector(loc / scale_vector[:, None])
    prob_superior = float(np.sum(weight * probabilities))
    prob_superior = min(1.0, max(0.0, prob_superior))
    # δ 网格上的密度与 CDF：按 g 节点分块累加，避免大矩阵。
    # 后验质量通常集中在少数 g 节点上；质量可忽略的节点直接
    # 跳过（阈值远低于任何报告精度），把每次对比的计算量从
    # 全部 ~1000 个节点降到实际有效的几十个。
    low = mean - _BAYES_DELTA_SPAN_SD * sd
    high = mean + _BAYES_DELTA_SPAN_SD * sd
    delta_grid = np.linspace(low, high, _BAYES_DELTA_GRID)
    density = np.zeros_like(delta_grid)
    cdf = np.zeros_like(delta_grid)
    n_g = loc.shape[0]
    for index in range(n_g):
        component_weight = weight[index]
        component_mass = float(np.sum(component_weight))
        if component_mass <= 1e-13:
            continue
        component_loc = loc[index]
        component_scale = float(scale_vector[index])
        standardized = (delta_grid[:, None] - component_loc[None, :]) / (
            component_scale
        )
        density += (
            component_weight[None, :]
            * np.exp(-0.5 * np.square(standardized))
            / (component_scale * math.sqrt(2.0 * math.pi))
        ).sum(axis=1)
        cdf += _normal_cdf_vector(standardized) @ component_weight
    density_total = _trapz_integral(density, delta_grid)
    if not math.isfinite(density_total) or density_total <= 0.0:
        return None
    density = density / density_total
    cdf = cdf / density_total
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)
    # 等尾 95% 分位区间（线性插值反演 CDF）。
    quantile_low = float(np.interp(0.025, cdf, delta_grid))
    quantile_high = float(np.interp(0.975, cdf, delta_grid))
    # HDI：均匀 δ 网格上的滑动窗口求最短 95% 质量区间。窗口
    # [start, stop) 的质量对应网格点 start 到 stop 之间的梯形
    # 质量与宽度，端点口径一致。
    window_mass = 0.95 * _trapz_integral(density, delta_grid)
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (density[1:] + density[:-1])
                * np.diff(delta_grid)
            ),
        )
    )
    best_low = quantile_low
    best_high = quantile_high
    best_width = None
    start = 0
    for stop in range(1, _BAYES_DELTA_GRID):
        while (
            start < stop
            and cumulative[stop] - cumulative[start] >= window_mass
        ):
            width = delta_grid[stop] - delta_grid[start]
            if best_width is None or width < best_width:
                best_width = width
                best_low = float(delta_grid[start])
                best_high = float(delta_grid[stop])
            start += 1
    if best_width is None:
        # 网格总质量不足 95%（极端平坦）：退回等尾区间。
        best_low = quantile_low
        best_high = quantile_high
    elif abs(mean) <= 1e-12:
        # t=0 时后验关于 0 对称，最短窗口在网格对齐下可能取到
        # 两个等宽的非对称镜像；对称化以匹配解析事实。
        half_width = 0.5 * (best_high - best_low)
        best_low = mean - half_width
        best_high = mean + half_width
    return {
        "posterior_prob_superior": prob_superior,
        "posterior_mean": mean,
        "posterior_sd": sd,
        "posterior_hdi95_low": best_low,
        "posterior_hdi95_high": best_high,
        "posterior_eti95_low": quantile_low,
        "posterior_eti95_high": quantile_high,
        "posterior_method": "exact_jzs_mixture_quadrature",
    }


def _trapz_integral(
    values: "np.ndarray",
    grid: "np.ndarray",
) -> float:
    """均匀/非均匀网格上的梯形积分（不依赖 np.trapz 版本差异）。"""
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(
        np.dot(0.5 * (values[1:] + values[:-1]), np.diff(grid))
    )


def _bayesian_contrast(
    contrast: Dict[str, object],
    prior_scale: float = _JZS_DEFAULT_PRIOR_SCALE,
) -> Optional[Dict[str, object]]:
    """
    为单个对比记录构造 JZS 贝叶斯补充分析。

    只读：不改写对比记录本身。配对设计（paired_t）用种子对数
    作为有效样本量；Welch 设计用 n₁n₂/(n₁+n₂)。任一必要字段
    缺失或非有限时返回带 ``available=False`` 与原因的记录，
    不抛异常，保证报告其余部分不受影响。
    """
    if not isinstance(contrast, dict):
        return None
    if contrast.get("confounded"):
        return {
            "available": False,
            "reason": "confounded_contrast",
        }
    test_name = str(contrast.get("test") or "")
    t_value = _finite_float(contrast.get("t"))
    df_value = _finite_float(contrast.get("df"))
    if t_value is None or df_value is None or df_value <= 0.0:
        return {
            "available": False,
            "reason": "missing_or_invalid_t_statistic",
        }
    if test_name == "paired_t":
        n_pairs = _finite_float(contrast.get("n_pairs"))
        if n_pairs is None or n_pairs < 2.0:
            return {
                "available": False,
                "reason": "insufficient_paired_seeds",
            }
        n_effective = n_pairs
        design = "paired"
    elif test_name == "welch_t":
        n_baseline = _finite_float(contrast.get("n_baseline"))
        n_variant = _finite_float(contrast.get("n_variant"))
        if (
            n_baseline is None
            or n_variant is None
            or n_baseline < 1.0
            or n_variant < 1.0
        ):
            return {
                "available": False,
                "reason": "missing_group_sizes",
            }
        n_effective = (n_baseline * n_variant) / (
            n_baseline + n_variant
        )
        design = "welch"
        if n_effective < 1.0:
            return {
                "available": False,
                "reason": "insufficient_effective_sample_size",
            }
    else:
        return {
            "available": False,
            "reason": f"unsupported_test:{test_name or 'unknown'}",
        }
    log_bf10 = _jzs_log_bayes_factor_10(
        t_value, n_effective, df_value, prior_scale
    )
    if log_bf10 is None:
        return {
            "available": False,
            "reason": "bayes_factor_not_finite",
        }
    bf10 = math.exp(log_bf10)
    entry: Dict[str, object] = {
        "available": True,
        "baseline": contrast.get("baseline"),
        "variant": contrast.get("variant"),
        "factor": contrast.get("factor"),
        "design": design,
        "test": test_name,
        "t": t_value,
        "df": df_value,
        "n_effective": n_effective,
        "delta": _finite_float(contrast.get("delta")),
        "prior": {
            "family": "cauchy",
            "scale": float(prior_scale),
            "standardized_effect": "delta = variant - baseline",
        },
        "log_bayes_factor_10": log_bf10,
        "bayes_factor_10": bf10,
        "evidence_category": _bayes_evidence_category(log_bf10),
        # 等先验几率（P(H1)=P(H0)=0.5）下的后验模型概率：
        # P(H1|D) = BF10/(1+BF10)。这是模型层面的概率，不是
        # 「效应为正」的概率，后者见 posterior_prob_superior。
        "posterior_prob_alternative": bf10 / (1.0 + bf10),
    }
    summaries = _jzs_posterior_summaries(
        t_value, n_effective, df_value, prior_scale
    )
    if summaries is None:
        entry["posterior_summaries_available"] = False
        entry["posterior_summaries_reason"] = (
            "posterior_quadrature_failed"
        )
    else:
        entry["posterior_summaries_available"] = True
        entry.update(summaries)
    return entry


def _result_fingerprints(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """收集本轮结果的追溯指纹（数据/切分/代码/环境）。

    人类可读的结果摘要必须能回答「这份结论对应哪份数据、哪份
    代码、哪个环境」；指纹不一致时不能只报一个值——那会把
    不可归因的对比伪装成同一份实验。
    """
    keys = (
        "data_sha256",
        "split_sha256",
        "effective_train_sha256",
        "seq_len",
        "source_tree_sha256",
        "script_sha256",
        "environment_id",
        "git_commit",
    )
    fingerprints: Dict[str, object] = {}
    for key in keys:
        values: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            protocol = row.get("protocol")
            value = None
            if isinstance(protocol, dict):
                value = protocol.get(key)
            if value is None:
                value = row.get(key)
            if value is None:
                continue
            text = str(value)
            if text not in values:
                values.append(text)
        if not values:
            continue
        fingerprints[key] = (
            values[0] if len(values) == 1 else values
        )
    fingerprints["consistent"] = all(
        not isinstance(value, list)
        for key, value in fingerprints.items()
        if key != "consistent"
    )
    fingerprints["note"] = (
        "同一键出现多个取值时列出全部：该轮对比涉及多份数据/"
        "代码/环境，结论不能整体归因"
    )
    return fingerprints


def _extrapolation_risk_report(
    rows: List[Dict[str, object]],
    threshold: float = 1e-4,
) -> Dict[str, object]:
    """评估集「未见标签」占比与外推风险标记。

    字节级 LM 只能通过上下文降低**已见**标签的 NLL；训练集从未
    出现的字节类在评估 token 上的 NLL 存在不可约的下界（只能
    退到平滑/先验）。这部分误差与方法优劣无关，却会直接进入
    指标均值：覆盖率不同的两臂，指标差里混着「谁的评估集更
    可预测」，不能直接当成方法收益。

    判定：任一运行/任一 split 的 ``unseen_label_token_share``
    超过阈值即标记 ``[extrapolation-risk]``，并给出受影响运行
    与占比。没有标签覆盖记录（旧运行）时只报告「不可核对」，
    不假装没有风险。
    """
    affected: List[Dict[str, object]] = []
    unknown_runs: List[str] = []
    max_share: Optional[float] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        coverage = row.get("label_coverage")
        if not isinstance(coverage, dict) or not coverage:
            unknown_runs.append(str(row.get("run")))
            continue
        for split, info in coverage.items():
            if not isinstance(info, dict):
                continue
            share = _finite_float(
                info.get("unseen_label_token_share")
            )
            if share is None:
                continue
            if max_share is None or share > max_share:
                max_share = share
            if share > threshold:
                affected.append({
                    "run": str(row.get("run")),
                    "split": str(split),
                    "unseen_label_token_share": share,
                    "unseen_label_classes": info.get(
                        "unseen_label_classes"
                    ),
                })
    return {
        "available": bool(affected) or bool(unknown_runs),
        "threshold": float(threshold),
        "max_unseen_label_token_share": max_share,
        "affected_runs": affected,
        "n_affected": len(affected),
        "runs_without_coverage_record": unknown_runs,
        "extrapolation_risk": bool(affected),
        "marker": "[extrapolation-risk]" if affected else None,
        "note": (
            "未见标签的 NLL 含不可约的外推误差：覆盖率不同的两臂"
            "不能直接比较指标。阈值以上的占比会被标出；没有标签"
            "覆盖记录（旧运行）时只能报告「不可核对」"
        ),
    }


def build_result_analysis_extensions(
    rows: List[Dict[str, object]],
    aggregates: List[Dict[str, object]],
    contrasts: List[Dict[str, object]],
    reference: List[Dict[str, object]],
    metric: str,
    group_keys: List[str],
) -> Dict[str, object]:
    """把结果分析层面的稳健性诊断合成为一个可落盘的扩展块。

    扩展块内容：

    - ``cross_metric``：单因子对比与直接基线对比的跨指标一致性
      与族级综合秩效应量（见 ``_cross_metric_effect_profile``）；
    - ``result_verdicts``：每个对比的机器可读裁决与下一步建议
      （见 ``_result_verdict``）；
    - ``cost_efficiency``：精度增益与效率代价的联合判定
      （见 ``_cost_efficiency_assessment``）；
    - ``routing_specialization_fdr``：路由专家分离的 BH-FDR 与
      跨种子一致性（见 ``_routing_specialization_fdr``）；
    - ``extrapolation_risk``：评估集未见标签占比带来的外推风险
      （见 ``_extrapolation_risk_report``）。

    这些字段全部是**追加**的：既有键名与含义不变，老的下游脚本
    仍然可用。
    """
    higher_is_better = metric_higher_is_better(metric)
    reference = _cross_metric_consistency_analysis(
        reference, metric, aggregates
    )

    verdict_entries: List[Dict[str, object]] = []
    cost_entries: List[Dict[str, object]] = []
    verdict_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    for family, family_contrasts in (
        ("single_factor", contrasts),
        ("baseline", reference),
    ):
        for contrast in family_contrasts:
            if not isinstance(contrast, dict):
                continue
            verdict = _result_verdict(contrast)
            contrast["result_verdict"] = verdict
            cost = _cost_efficiency_assessment(
                contrast, higher_is_better=higher_is_better
            )
            contrast["cost_efficiency"] = cost
            label = (
                f"{contrast.get('baseline')}→{contrast.get('variant')}"
            )
            entry = {
                "family": family,
                "baseline": contrast.get("baseline"),
                "variant": contrast.get("variant"),
                "label": label,
                "verdict": verdict.get("verdict"),
                "action": verdict.get("action"),
                "reasons": verdict.get("reasons"),
                "significant": verdict.get("significant"),
                "equivalent": verdict.get("equivalent"),
                "cross_metric_divergent": bool(
                    contrast.get("cross_metric_divergent")
                ),
                "cost_inefficient": bool(
                    cost.get("cost_inefficient")
                ),
            }
            verdict_entries.append(entry)
            key = str(verdict.get("verdict"))
            verdict_counts[key] = verdict_counts.get(key, 0) + 1
            action = str(verdict.get("action"))
            action_counts[action] = action_counts.get(action, 0) + 1
            cost_entries.append({
                "family": family,
                "baseline": contrast.get("baseline"),
                "variant": contrast.get("variant"),
                "label": label,
                "available": bool(cost.get("available")),
                "basis": cost.get("basis"),
                "cost_ratio": cost.get("cost_ratio"),
                "relative_gain": cost.get("relative_gain"),
                "gain_per_cost": cost.get("gain_per_cost"),
                "gain_per_cost_ci95": cost.get("gain_per_cost_ci95"),
                "cost_inefficient": bool(
                    cost.get("cost_inefficient")
                ),
                "cost_unproven": bool(
                    cost.get("cost_unproven")
                ),
                "marker": cost.get("marker"),
                "reason": cost.get("reason"),
            })

    routing_fdr = _routing_specialization_fdr(rows, group_keys)
    extrapolation = _extrapolation_risk_report(rows)
    return {
        "schema": "smelt-isr/result-analysis-extensions/v1",
        "metric": metric,
        "fingerprints": _result_fingerprints(rows),
        "cross_metric": {
            "n_contrasts": len(contrasts),
            "n_divergent": sum(
                1
                for contrast in contrasts
                if isinstance(contrast, dict)
                and contrast.get("cross_metric_divergent")
            ),
        },
        "result_verdicts": {
            "entries": verdict_entries,
            "verdict_counts": verdict_counts,
            "action_counts": action_counts,
        },
        "cost_efficiency": cost_entries,
        "routing_specialization_fdr": routing_fdr,
        "extrapolation_risk": extrapolation,
    }


def _bayesian_analysis_summary(
    entries: List[Dict[str, object]],
    prior_scale: float,
) -> Dict[str, object]:
    """汇总贝叶斯补充分析条目，供展示与 --compare-report 落盘。"""
    available_entries = [
        entry for entry in entries if entry.get("available")
    ]
    unavailable_entries = [
        {
            "baseline": entry.get("baseline"),
            "variant": entry.get("variant"),
            "reason": entry.get("reason"),
        }
        for entry in entries
        if not entry.get("available")
    ]
    advisories: List[str] = []
    for entry in unavailable_entries:
        advisories.append(
            f"{entry.get('baseline')}→{entry.get('variant')}: "
            f"贝叶斯补充分析不可用（{entry.get('reason')}）"
        )
    summary = {
        "schema": "smelt-isr/bayesian-supplement/v1",
        "enabled": True,
        "prior": {
            "family": "cauchy",
            "scale": float(prior_scale),
        },
        "n_contrasts": len(entries),
        "n_available": len(available_entries),
        "entries": entries,
        "unavailable": unavailable_entries,
        "advisories": advisories,
        "note": (
            "贝叶斯因子是证据强度的补充度量，不替代 Holm 校正后的"
            "显著性判定，也不改变 TOST 等价性结论；证据分级按 "
            "|ln BF10|（≥1.1 实质、≥2.3 强、≥4.6 决定性，负值表示"
            "支持零假设）。posterior_prob_superior 是 "
            "P(δ>0|D)，δ = 变体 − 基线；对「越低越好」的指标，"
            "δ>0 对应变体更差。先验尺度是分析者选择，换尺度会"
            "改变 BF 数值，报告必须随结果一起声明先验。"
        ),
    }
    if available_entries:
        categories: Dict[str, int] = {}
        for entry in available_entries:
            category = str(entry.get("evidence_category"))
            categories[category] = categories.get(category, 0) + 1
        summary["evidence_category_counts"] = dict(
            sorted(categories.items())
        )
    return summary


def _window_size_sensitivity_analysis(
    per_seed_differences: Dict[int, List[float]],
) -> Dict[str, object]:
    """评估集规模敏感性：固定窗口集合下换用不同规模的窗口子集。

    固定评估窗口的长度在训练期就定好了，分析期无法重算更短/
    更长的窗口；能做且有意义的是**换用不同规模的窗口子集**，
    观察配对效应估计随评估集规模变化的漂移。它回答的是
    「结论是不是靠某一批评估窗口撑起来的」以及「扩大评估集
    能把区间收窄多少」，不能回答「窗口长度本身变了会怎样」
    （后者需要按不同 seq_len 重新评估）。

    漂移与噪声比较：完整窗口集的逐种子窗口均值差给出效应
    点估计；窗口层噪声用「合并后的窗口均值标准误」衡量
    （每个种子内窗口差值的样本标准差 / √窗口数，再对种子
    取均方根）。漂移超过该噪声尺度才算「对评估集规模敏感」，
    否则只是评估集抽样噪声，不能读成结论不稳。

    方向稳定性单独判定：任一子集的均值差与完整集反号时，
    结论的方向依赖于具体用了哪批窗口，必须显式披露。
    """
    cleaned = _clean_window_differences(per_seed_differences)
    usable = {
        seed: values
        for seed, values in cleaned.items()
        if len(values) >= 4
    }
    if len(usable) < 2:
        return {
            "available": False,
            "reason": "insufficient_seeds_or_windows",
            "n_seeds": len(usable),
            "n_windows": (
                len(next(iter(usable.values()))) if usable else 0
            ),
            "note": (
                "评估集规模敏感性至少需要 2 个种子且每个种子至少 "
                "4 个评估窗口；不足时不伪造敏感性结论"
            ),
        }

    seed_order = sorted(usable)
    window_counts = {len(usable[seed]) for seed in seed_order}
    if len(window_counts) != 1:
        return {
            "available": False,
            "reason": "unbalanced_window_counts",
            "n_seeds": len(seed_order),
            "window_counts": sorted(window_counts),
            "note": (
                "固定评估窗口在各运行间应逐位相同；窗口数不一致"
                "时不能用同一组窗口下标做子集敏感性分析"
            ),
        }

    window_count = window_counts.pop()
    seed_means = {
        seed: float(_mean(usable[seed])) for seed in seed_order
    }
    full_mean = float(_mean(list(seed_means.values())))
    if full_mean is None or not math.isfinite(full_mean):
        return {
            "available": False,
            "reason": "invalid_mean_difference",
            "note": "窗口差值均值不是有限数，无法做敏感性分析",
        }

    # 窗口层噪声：每个种子内窗口差值的标准误，再对种子取
    # 均方根（种子间差异由 seed_means 单独承担）。
    window_variances: List[float] = []
    for seed in seed_order:
        values = usable[seed]
        std = _sample_std(values)
        if std is None:
            continue
        variance = (std * std) / float(max(len(values), 1))
        if math.isfinite(variance):
            window_variances.append(variance)
    window_level_se = (
        math.sqrt(sum(window_variances) / len(window_variances))
        if window_variances
        else None
    )
    window_level_variance = (
        window_level_se * window_level_se
        if window_level_se is not None
        else None
    )

    def _subset_mean(seed: int, start: int, stop: int) -> Optional[float]:
        values = usable[seed][start:stop]
        if not values:
            return None
        return _mean(values)

    subsets: List[Dict[str, object]] = []
    for fraction, anchors in (
        (0.8, ("head", "tail", "middle")),
        (0.5, ("head", "tail")),
    ):
        size = max(2, int(round(fraction * window_count)))
        if size >= window_count:
            continue
        for anchor in anchors:
            if anchor == "head":
                start, stop = 0, size
            elif anchor == "tail":
                start, stop = window_count - size, window_count
            else:
                start = (window_count - size) // 2
                stop = start + size
            subset_means = {
                seed: _subset_mean(seed, start, stop)
                for seed in seed_order
            }
            usable_means = [
                value
                for value in subset_means.values()
                if value is not None
            ]
            if not usable_means:
                continue
            subset_mean = float(_mean(usable_means))
            drift = subset_mean - full_mean
            direction_same = bool(
                (subset_mean > 0.0 and full_mean > 0.0)
                or (subset_mean < 0.0 and full_mean < 0.0)
                or (subset_mean == 0.0 and full_mean == 0.0)
            )
            subset_se = None
            subset_significant = None
            if window_level_variance is not None:
                subset_se = math.sqrt(
                    window_level_variance / float(max(size, 1))
                )
                if subset_se > 0.0:
                    subset_significant = bool(
                        abs(subset_mean) > 1.959963984540054 * subset_se
                    )
            full_significant = None
            if window_level_se is not None and window_level_se > 0.0:
                full_significant = bool(
                    abs(full_mean) > 1.959963984540054 * window_level_se
                )
            significance_same = None
            if (
                subset_significant is not None
                and full_significant is not None
            ):
                significance_same = bool(
                    subset_significant == full_significant
                )
            relative_drift = None
            if abs(full_mean) > 1e-12:
                relative_drift = abs(drift) / abs(full_mean)
            subsets.append({
                "subset": f"{anchor}_{int(round(fraction * 100))}pct",
                "fraction": float(fraction),
                "anchor": anchor,
                "n_windows": size,
                "mean_diff": subset_mean,
                "drift": drift,
                "relative_drift": relative_drift,
                "direction_same_as_full": direction_same,
                "subset_se": subset_se,
                "subset_significant": subset_significant,
                "significance_same_as_full": significance_same,
            })

    drifts = [
        abs(float(entry["drift"]))
        for entry in subsets
        if _finite_float(entry.get("drift")) is not None
    ]
    max_abs_drift = max(drifts) if drifts else None
    direction_flipped = [
        str(entry["subset"])
        for entry in subsets
        if not entry.get("direction_same_as_full")
    ]
    significance_flipped = [
        str(entry["subset"])
        for entry in subsets
        if entry.get("significance_same_as_full") is False
    ]
    exceeds_noise = None
    material_drift = None
    if max_abs_drift is not None and window_level_se is not None:
        # 阈值取两者的较大者：漂移要同时超过普通抽样波动的 2 倍
        # 标准误**并且**相对效应本身不可忽略，才算「结论随评估集
        # 规模改变」。只看标准误会把正常抽样波动误判成不稳定；
        # 只看相对量会把与效应无关的小噪声判成敏感。
        exceeds_noise = bool(max_abs_drift > 2.0 * window_level_se)
    if max_abs_drift is not None and abs(full_mean) > 1e-12:
        material_drift = bool(
            max_abs_drift > 0.2 * abs(full_mean)
        )
    if direction_flipped or significance_flipped:
        verdict = "sensitive"
    elif exceeds_noise and material_drift:
        verdict = "sensitive"
    else:
        verdict = "stable"
    return {
        "available": True,
        "verdict": verdict,
        "method": "window_subset_resampling",
        "n_seeds": len(seed_order),
        "n_windows": window_count,
        "full_mean_diff": full_mean,
        "seed_mean_differences": {
            str(seed): seed_means[seed] for seed in seed_order
        },
        "window_level_se": window_level_se,
        "max_abs_drift": max_abs_drift,
        "drift_exceeds_window_noise": exceeds_noise,
        "drift_material_relative_to_effect": material_drift,
        "direction_flipped_subsets": direction_flipped,
        "significance_flipped_subsets": significance_flipped,
        "subsets": subsets,
        "note": (
            "评估集规模敏感性：固定窗口长度下改用不同规模的窗口"
            "子集重算配对效应。方向翻转、显著性翻转，或漂移同时"
            "超过窗口层噪声的 2 倍标准误与效应量的 20% 时，"
            "结论对评估集敏感，应扩大评估集而不是只加种子；"
            "它不等价于窗口长度敏感性（后者需要按不同 seq_len "
            "重新评估）"
        ),
    }


def _cross_metric_effect_profile(
    cell_a: Dict[str, object],
    cell_b: Dict[str, object],
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """同一对比在多个预注册指标上的效应剖面。

    单因子对比只在**一个**判定指标上做检验，因此「换指标看
    结论是否翻转」无法从对比记录本身读出。这里对两个实验单元
    在所有可用的最终评估点指标上重算效应，方向统一折算成
    「变体相对基线是改善还是变差」，并给出：

    - 每个指标的 Δ、相对变化、秩效应量（Cliff's δ，按改善
      方向折算）、p 值与显著性；
    - 按**同源指标族**（互相高度相关、不能当作独立证据的
      指标归为一族）聚合后的族级方向与族级秩效应量中位数；
    - 跨族综合秩效应量（族级中位数的中位数与 IQR），避免
      只挑一个「最好看」的指标下结论。

    指标族划分是分析口径的一部分，必须随结果一起落盘：把
    val_loss / val_ppl / val_bits_per_byte 这类互为单调变换
    的指标当作三份独立证据，会人为放大「一致性」。
    """
    if metrics is None:
        metrics = tuple(
            name
            for _family, names in _CROSS_METRIC_FAMILIES
            for name in names
        )
    samples_a = _as_dict(cell_a.get("samples"))
    samples_b = _as_dict(cell_b.get("samples"))
    family_of = {
        name: family
        for family, names in _CROSS_METRIC_FAMILIES
        for name in names
    }

    entries: List[Dict[str, object]] = []
    for metric in metrics:
        raw_a = samples_a.get(metric)
        raw_b = samples_b.get(metric)
        if not isinstance(raw_a, list) or not isinstance(raw_b, list):
            continue
        if not raw_a or not raw_b:
            continue
        collapsed_a = _collapse_seed_samples(raw_a)
        collapsed_b = _collapse_seed_samples(raw_b)
        values_a = [
            float(sample["value"]) for sample in collapsed_a
        ]
        values_b = [
            float(sample["value"]) for sample in collapsed_b
        ]
        if not values_a or not values_b:
            continue
        mean_a = _mean(values_a)
        mean_b = _mean(values_b)
        if mean_a is None or mean_b is None:
            continue
        delta = mean_b - mean_a
        if not math.isfinite(delta):
            continue
        higher_is_better = metric_higher_is_better(metric)
        improvement = delta if higher_is_better else -delta
        if improvement > 0.0:
            direction = "improves"
        elif improvement < 0.0:
            direction = "degrades"
        else:
            direction = "tie"
        relative_delta = (
            delta / abs(mean_a)
            if abs(mean_a) > 1e-12
            else None
        )
        cliffs = _cliffs_delta(values_a, values_b)
        # Cliff's δ 以「基线 - 变体」为符号：正值表示基线更大。
        # 折算成「变体是否更好」：越低越好的指标取原值，越高越
        # 好的指标取相反数。
        improvement_cliffs = (
            cliffs if not higher_is_better else (
                None if cliffs is None else -cliffs
            )
        )
        test = _contrast_test(list(raw_a), list(raw_b))
        p_value = _finite_float(test.get("p_value"))
        hedges = _compute_cohens_d(values_a, values_b)
        improvement_hedges = (
            hedges if not higher_is_better else (
                None if hedges is None else -hedges
            )
        )
        entries.append({
            "metric": metric,
            "family": family_of.get(metric, "other"),
            "higher_is_better": bool(higher_is_better),
            "n_baseline": len(values_a),
            "n_variant": len(values_b),
            "mean_baseline": mean_a,
            "mean_variant": mean_b,
            "delta": delta,
            "relative_delta": relative_delta,
            "improvement": improvement,
            "improvement_direction": direction,
            "cliffs_delta": cliffs,
            "improvement_cliffs_delta": improvement_cliffs,
            "hedges_g": hedges,
            "improvement_hedges_g": improvement_hedges,
            "test": test.get("test"),
            "p_value": p_value,
            "significant": bool(
                p_value is not None and p_value <= _SIGNIFICANCE_LEVEL
            ),
        })

    families: List[Dict[str, object]] = []
    for family, names in _CROSS_METRIC_FAMILIES:
        family_entries = [
            entry for entry in entries if entry["family"] == family
        ]
        if not family_entries:
            continue
        cliffs_values = [
            float(entry["improvement_cliffs_delta"])
            for entry in family_entries
            if _finite_float(entry.get("improvement_cliffs_delta"))
            is not None
        ]
        family_cliffs = (
            float(_median(cliffs_values)) if cliffs_values else None
        )
        significant_entries = [
            entry
            for entry in family_entries
            if entry.get("significant")
        ]
        significant_directions = {
            str(entry["improvement_direction"])
            for entry in significant_entries
        }
        if "improves" in significant_directions and (
            "degrades" in significant_directions
        ):
            family_direction = "mixed"
        elif significant_directions == {"improves"}:
            family_direction = "improves"
        elif significant_directions == {"degrades"}:
            family_direction = "degrades"
        elif significant_directions == {"tie"}:
            family_direction = "tie"
        elif significant_entries:
            family_direction = "undetermined"
        elif family_cliffs is not None:
            if family_cliffs > 0.0:
                family_direction = "improves"
            elif family_cliffs < 0.0:
                family_direction = "degrades"
            else:
                family_direction = "tie"
        else:
            family_direction = "undetermined"
        families.append({
            "family": family,
            "metrics": [str(entry["metric"]) for entry in family_entries],
            "n_metrics": len(family_entries),
            "median_improvement_cliffs_delta": family_cliffs,
            "improvement_direction": family_direction,
            "significant_metrics": [
                str(entry["metric"]) for entry in significant_entries
            ],
            "n_significant": len(significant_entries),
        })

    return {
        "schema": "smelt-isr/cross-metric-profile/v1",
        "available": bool(entries),
        "n_metrics": len(entries),
        "metrics": entries,
        "families": families,
        "metric_families": {
            family: list(names)
            for family, names in _CROSS_METRIC_FAMILIES
        },
        "note": (
            "所有指标都在**最终评估点**上计算（不含共同步数/等"
            "计算量等派生口径）；方向统一折算成「变体相对基线是"
            "改善还是变差」。同源指标族内的指标互相高度相关，"
            "跨族汇总才近似独立证据"
        ),
    }


def _cross_metric_consistency_report(
    profile: Dict[str, object],
) -> Dict[str, object]:
    """把跨指标效应剖面合成一致性与综合秩效应量判定。"""
    entries = profile.get("metrics")
    families = profile.get("families")
    if not isinstance(entries, list) or not isinstance(families, list):
        entries = []
        families = []

    metric_directions: Dict[str, str] = {}
    significant_metrics: List[str] = []
    improving_metrics: List[str] = []
    degrading_metrics: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("metric"))
        direction = str(entry.get("improvement_direction"))
        metric_directions[metric] = direction
        if entry.get("significant"):
            significant_metrics.append(metric)
            if direction == "improves":
                improving_metrics.append(metric)
            elif direction == "degrades":
                degrading_metrics.append(metric)

    family_directions: Dict[str, str] = {}
    for family in families:
        if not isinstance(family, dict):
            continue
        family_directions[str(family.get("family"))] = str(
            family.get("improvement_direction")
        )

    # 分歧只看「有显著证据」的族：没有显著证据的族方向只是噪声，
    # 拿它判定分歧会把正常的无效应指标误报成结论冲突。
    informative = {
        name: direction
        for name, direction in family_directions.items()
        if direction in ("improves", "degrades", "mixed")
    }
    divergent = bool(
        "mixed" in informative.values()
        or ("improves" in informative.values()
            and "degrades" in informative.values())
    )
    divergent_families = [
        name
        for name, direction in informative.items()
        if direction == "mixed"
    ]
    if divergent and not divergent_families:
        divergent_families = sorted(informative)

    family_cliffs = [
        float(family["median_improvement_cliffs_delta"])
        for family in families
        if isinstance(family, dict)
        and _finite_float(
            family.get("median_improvement_cliffs_delta")
        )
        is not None
    ]
    composite = (
        float(_median(family_cliffs)) if family_cliffs else None
    )
    iqr = None
    if len(family_cliffs) >= 2:
        ordered = sorted(family_cliffs)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            low = ordered[middle]
            high = ordered[middle]
        else:
            low = ordered[middle - 1]
            high = ordered[middle]
        iqr = high - low

    return {
        "available": bool(entries),
        "cross_metric_consistent": bool(entries) and not divergent,
        "cross_metric_divergent": divergent,
        "significant_metrics": significant_metrics,
        "improving_metrics": improving_metrics,
        "degrading_metrics": degrading_metrics,
        "metric_directions": metric_directions,
        "family_directions": family_directions,
        "divergent_families": divergent_families,
        "composite_cliffs_delta": composite,
        "composite_cliffs_delta_iqr": iqr,
        "composite_n_families": len(family_cliffs),
        "marker": "[metric-divergence]" if divergent else None,
        "note": (
            "跨指标一致性：方向统一折算为「变体相对基线是改善还是"
            "变差」，只在有显著证据的同源指标族之间判定分歧。"
            "综合秩效应量是族级 Cliff's δ 的中位数（中位数与其"
            "IQR），正号表示变体整体更优；它用于防止只挑单个指标"
            "下结论，不替代 Holm 校正后的显著性判定"
        ),
    }


def _result_verdict(
    contrast: Dict[str, object],
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Dict[str, object]:
    """把一个对比的既有判定合成为机器可读的裁决与下一步建议。

    判定树（按优先级）：

    1. 混杂对比（同时改了多个因子）→ ``confounded``：先统一协议
       再重跑，任何统计量都不能归因；
    2. 显著但脆弱（删掉少量种子即失效）→ ``inconclusive_fragile``；
    3. 显著但缺失/失败运行的极端赋值可改变方向 →
       ``inconclusive_attrition``；
    4. 显著但功效低于目标 → ``significant_underpowered``；
    5. 显著且稳健 → ``significant``；
    6. TOST 通过 → ``equivalent``（支持无实质差异）；
    7. 不显著且点估计已越出等价界 → ``reject_equivalence``
       （确有差异，不能读成「无差异」）；
    8. 不显著且卡在精度 → ``inconclusive_precision``（加种子，
       不要事后放宽等价界）；
    9. 不显著且功效不足 → ``underpowered``；
    10. 其余 → ``not_significant``。

    ``recommended_action`` 给出可执行的下一步：``add_seeds`` /
    ``expand_eval_set`` / ``add_seeds_and_expand_eval_set`` /
    ``complete_runs`` / ``fix_protocol`` / ``none``。方差分解显示
    不确定性主要来自评估窗口时，优先建议扩大评估集而不是加种子。
    """
    reasons: List[str] = []
    action = "none"

    if contrast.get("confounded"):
        return {
            "verdict": "confounded",
            "action": "fix_protocol",
            "reasons": [
                "同时改变了多个因子，统计显著也不能归因到待验证模块"
            ],
            "significant": bool(contrast.get("significant")),
            "equivalent": bool(contrast.get("equivalent")),
            "note": (
                "混杂对比必须先统一协议重跑；本裁决不解读统计量"
            ),
        }

    significant = bool(contrast.get("significant"))
    equivalent = bool(contrast.get("equivalent"))
    fragility = _as_dict(contrast.get("fragility"))
    attrition = _as_dict(contrast.get("attrition_bounds"))
    power_at_effect = _finite_float(contrast.get("power_at_effect"))
    blocker = contrast.get("equivalence_blocker")

    if significant:
        verdict = "significant"
        fragility_verdict = str(fragility.get("verdict") or "")
        attrition_verdict = str(attrition.get("verdict") or "")
        if fragility_verdict == "fragile":
            verdict = "inconclusive_fragile"
            action = "add_seeds"
            reasons.append(
                "删除少量随机种子后显著性即消失：结论依赖个别种子"
            )
        elif attrition_verdict in ("crosses_zero", "unbounded") or (
            attrition.get("sign_robust") is False
        ):
            verdict = "inconclusive_attrition"
            action = "complete_runs"
            reasons.append(
                "缺失/失败运行的极端赋值可改变效应方向：显著结论"
                "可能由「哪些运行留下了」驱动"
            )
        elif (
            power_at_effect is not None
            and power_at_effect < _TARGET_POWER
        ):
            verdict = "significant_underpowered"
            action = "add_seeds"
            reasons.append(
                f"当前功效 {power_at_effect:.0%} 低于 "
                f"{_TARGET_POWER:.0%} 目标：结论方向可用，强度不足"
            )
        else:
            reasons.append(
                "Holm 校正后显著，且未触发脆弱性/缺失/低功效标记"
            )
    elif equivalent:
        verdict = "equivalent"
        reasons.append("TOST 通过：支持两臂无实质差异")
    else:
        if blocker == "effect_exceeds_margin":
            verdict = "reject_equivalence"
            reasons.append(
                "点估计已超出等价界：不显著不是「无差异」，而是确有"
                "差异，增加种子只会让这一点更确定"
            )
        elif blocker == "insufficient_precision":
            verdict = "inconclusive_precision"
            action = "add_seeds"
            reasons.append(
                "点估计落在等价界内但精度不足：TOST 未通过不能读成"
                "两者不同，应增加种子而不是事后放宽等价界"
            )
        elif power_at_effect is not None and power_at_effect < 0.5:
            verdict = "underpowered"
            action = "add_seeds"
            reasons.append(
                f"观测效应的功效仅 {power_at_effect:.0%}：本轮"
                "没有能力区分「无效应」与「未检出」"
            )
        else:
            verdict = "not_significant"
            reasons.append(
                "未检出显著差异，且等价性与功效均未给出额外判定"
            )

    window_paired = _as_dict(contrast.get("window_paired"))
    components = _as_dict(window_paired.get("variance_components"))
    dominant = components.get("dominant_source")
    if dominant == "evaluation_windows":
        reasons.append(
            "方差分解显示不确定性主要来自评估窗口：扩大评估集比"
            "加种子更经济"
        )
        action = "expand_eval_set"
    elif dominant in ("seed_window_interaction", "mixed"):
        reasons.append(
            "方差分解显示种子×窗口交互或混合来源占主导：需要同时"
            "增加独立种子并扩大评估集"
        )
        action = "add_seeds_and_expand_eval_set"

    return {
        "verdict": verdict,
        "action": action,
        "reasons": reasons,
        "significant": significant,
        "equivalent": equivalent,
        "power_at_effect": power_at_effect,
        "fragility_verdict": fragility.get("verdict"),
        "attrition_verdict": attrition.get("verdict"),
        "equivalence_blocker": blocker,
        "dominant_uncertainty_source": dominant,
        "alpha": float(alpha),
        "note": (
            "裁决由既有判定字段机械合成，不替代人工审阅；"
            "recommended_action 只提示性价比最高的下一步"
        ),
    }


def _cost_efficiency_assessment(
    contrast: Dict[str, object],
    higher_is_better: bool = False,
) -> Dict[str, object]:
    """精度增益与效率代价的联合判定（gain per cost）。

    只报「涨点 + 效率倍率」不足以支持选型：涨点幅度要与**额外**
    代价成比例才划算。这里用相对增益除以额外成本倍率
    （``cost_ratio − 1``）得到 ``gain_per_cost``：>1 表示相对增益
    以 1:1 以上的比例赚回多花的那部分成本，≤1 表示代价没有赚回。
    区间用效应区间与成本倍率区间给出保守口径（增益取最不利端、
    成本取最有利端）。

    成本倍率 ≥1.5× 时：

    - 区间整体 ≤1 → ``cost_inefficient``（额外代价明确没赚回）；
    - 区间跨过 1 → ``cost_unproven``（无法证明代价被赚回）。

    变体不比基线贵时不存在取舍，判定为 ``no_extra_cost``。
    """
    cost = _as_dict(contrast.get("cost"))
    ratios = _as_dict(cost.get("ratios"))
    uncertainty = _as_dict(cost.get("uncertainty"))
    if not ratios:
        return {
            "available": False,
            "reason": "no_cost_measurement",
            "note": "缺少效率测量，无法做精度/效率联合判定",
        }

    basis = None
    for field in ("flops_per_forward", "n_params"):
        if field in ratios:
            basis = field
            break
    if basis is None:
        basis = sorted(ratios)[0] if ratios else None
    if basis is None:
        return {
            "available": False,
            "reason": "no_cost_measurement",
            "note": "缺少效率测量，无法做精度/效率联合判定",
        }
    cost_ratio = _finite_float(ratios.get(basis))
    if cost_ratio is None or cost_ratio <= 0.0:
        return {
            "available": False,
            "reason": "invalid_cost_ratio",
            "basis": basis,
            "note": "成本倍率不可用，无法做精度/效率联合判定",
        }

    mean_baseline = _finite_float(contrast.get("mean_baseline"))
    mean_variant = _finite_float(contrast.get("mean_variant"))
    delta = _finite_float(contrast.get("delta"))
    if mean_baseline is None or mean_variant is None or delta is None:
        return {
            "available": False,
            "reason": "missing_effect_estimate",
            "basis": basis,
            "note": "缺少效应估计，无法做精度/效率联合判定",
        }
    if abs(mean_baseline) <= 1e-12:
        return {
            "available": False,
            "reason": "degenerate_baseline_mean",
            "basis": basis,
            "note": "基线均值为 0，相对增益无定义",
        }

    relative_gain = (
        (mean_variant - mean_baseline) / abs(mean_baseline)
    )
    if not higher_is_better:
        relative_gain = -relative_gain
    extra_cost_ratio = cost_ratio - 1.0
    if extra_cost_ratio <= 0.0:
        # 变体不比基线贵：没有额外代价，也就没有效率取舍。
        return {
            "available": True,
            "basis": basis,
            "cost_ratio": cost_ratio,
            "material_cost": False,
            "extra_cost_ratio": 0.0,
            "relative_gain": relative_gain,
            "gain_per_cost": None,
            "gain_per_cost_ci95": None,
            "ci_crosses_one": False,
            "cost_inefficient": False,
            "cost_unproven": False,
            "marker": None,
            "reason": (
                f"成本倍率 {cost_ratio:.2f}× ≤1：变体没有额外成本，"
                "不存在效率取舍"
            ),
            "note": (
                "gain_per_cost = 相对增益 / 额外成本倍率"
                "（cost_ratio − 1）；>1 表示额外代价被赚回"
            ),
        }
    gain_per_cost = relative_gain / extra_cost_ratio

    ci_low = _finite_float(contrast.get("ci95_low"))
    ci_high = _finite_float(contrast.get("ci95_high"))
    if ci_low is None or ci_high is None:
        delta_ci = _as_dict(contrast.get("delta_ci95"))
        ci_low = _finite_float(delta_ci.get("low"))
        ci_high = _finite_float(delta_ci.get("high"))
    if ci_low is None or ci_high is None:
        # 对比载荷里常规只有 delta/se/df：效应区间按同一个 t 分布
        # 还原（与显著性判定同一个 t 临界值），不另造口径。
        se = _finite_float(contrast.get("se"))
        df = _finite_float(contrast.get("df"))
        critical = (
            _t_critical(df) if df is not None else None
        )
        if se is not None and critical is not None:
            ci_low = delta - critical * se
            ci_high = delta + critical * se
    detail = _as_dict(uncertainty.get(basis))
    ratio_ci = detail.get("ratio_ci95")
    ratio_low = None
    ratio_high = None
    if isinstance(ratio_ci, (list, tuple)) and len(ratio_ci) == 2:
        ratio_low = _finite_float(ratio_ci[0])
        ratio_high = _finite_float(ratio_ci[1])
    # 区间要与点估计同一口径：分母是**额外**成本倍率，不是总倍率。
    extra_low = None
    extra_high = None
    if ratio_low is not None:
        extra_low = max(ratio_low - 1.0, 0.0)
    if ratio_high is not None:
        extra_high = max(ratio_high - 1.0, 0.0)

    gain_low = None
    gain_high = None
    if ci_low is not None and ci_high is not None:
        # 区间端点是**差值**（variant − baseline），换算成相对基线
        # 的相对增益；越低越好的指标取反后区间要重新排序。
        raw_low = ci_low / abs(mean_baseline)
        raw_high = ci_high / abs(mean_baseline)
        if not higher_is_better:
            raw_low, raw_high = -raw_low, -raw_high
        gain_low = min(raw_low, raw_high)
        gain_high = max(raw_low, raw_high)

    interval_low = None
    interval_high = None
    if gain_low is not None and extra_high is not None and extra_high > 0:
        interval_low = gain_low / extra_high
    if gain_high is not None and extra_low is not None and extra_low > 0:
        interval_high = gain_high / extra_low

    material = bool(cost_ratio >= _COST_RATIO_MATERIAL)
    crosses_one = None
    below_one = None
    if interval_low is not None and interval_high is not None:
        crosses_one = bool(interval_low < 1.0 < interval_high)
        below_one = bool(interval_high <= 1.0)
    elif interval_low is not None:
        # 成本区间下端 ≤1：变体可能根本不更贵，效率取舍无法判定
        # 为「不划算」，只能记为「没有被证明」。
        crosses_one = bool(interval_low < 1.0)
        below_one = False
    cost_inefficient = bool(material and below_one)
    cost_unproven = bool(material and crosses_one)
    if cost_inefficient:
        reason = (
            f"成本倍率 {cost_ratio:.2f}×（≥1.5×）且增益/额外成本区间"
            "整体不超过 1：额外代价没有被赚回"
        )
    elif cost_unproven:
        reason = (
            f"成本倍率 {cost_ratio:.2f}×（≥1.5×）且增益/额外成本区间"
            "跨过 1：额外代价没有被证明换来成比例的收益"
        )
    elif material:
        reason = (
            f"成本倍率 {cost_ratio:.2f}×（≥1.5×），但增益/额外成本区间"
            "仍在 1 以上：代价与收益成比例"
        )
    else:
        reason = (
            f"成本倍率 {cost_ratio:.2f}× 未达 1.5× 门槛：效率代价"
            "相对收益不构成取舍问题"
        )
    marker = None
    if cost_inefficient:
        marker = "[cost-inefficient]"
    elif cost_unproven:
        marker = "[cost-unproven]"
    return {
        "available": True,
        "basis": basis,
        "cost_ratio": cost_ratio,
        "material_cost": material,
        "extra_cost_ratio": extra_cost_ratio,
        "relative_gain": relative_gain,
        "gain_per_cost": gain_per_cost,
        "gain_per_cost_ci95": (
            [interval_low, interval_high]
            if interval_low is not None and interval_high is not None
            else None
        ),
        "ci_crosses_one": crosses_one,
        "ci_below_one": below_one,
        "cost_inefficient": cost_inefficient,
        "cost_unproven": cost_unproven,
        "marker": marker,
        "reason": reason,
        "note": (
            "gain_per_cost = 相对增益 / 额外成本倍率（cost_ratio − 1）；"
            ">1 表示额外代价被赚回。区间用最不利增益与最有利成本"
            "组合，属保守口径。该判定只描述精度与效率的取舍，"
            "不参与显著性判定"
        ),
    }


def _benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Benjamini-Hochberg FDR correction.
    
    Returns (adjusted_p_values, rejected_hypotheses).
    """
    import numpy as np
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return [], []
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]
    # BH step-up procedure
    ranks = np.arange(1, n + 1, dtype=float)
    thresholds = alpha * ranks / n
    # Find largest k where p_(k) <= alpha * k / n
    rejected_sorted = sorted_p <= thresholds
    # Find the largest k that is rejected
    k_max = 0
    for i in range(n):
        if rejected_sorted[i]:
            k_max = i + 1  # 1-indexed
    
    # Compute adjusted p-values
    adjusted = np.empty(n)
    for i in range(n):
        # adj_p_i = min_{j >= i} (n * p_(j) / j), then cap at 1。
        # 该逐元素取 min 已含 step-up 单调化；单调性定义在排序
        # 秩上，映射回原顺序后不能再按原索引顺序做 accumulate
        # （那会把无关假设的校正 p 值错误地串改，如 p=[0.5, 0.01]
        # 会被改成 [0.02, 0.02]）。
        min_ratio = np.min(p[sorted_indices[i:]] * n / (np.arange(i, n) + 1))
        adjusted[sorted_indices[i]] = min(min_ratio, 1.0)

    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    if k_max > 0:
        rejected[sorted_indices[:k_max]] = True
    
    return list(adjusted), list(rejected)


def _compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size with Hedges' g correction for small samples.
    
    Added during code review to complement significance testing with effect size reporting.
    """
    import numpy as np
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    # Hedges' g correction for small sample bias
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * correction


def welch_t_test(
    values_a: List[float],
    values_b: List[float],
) -> Tuple[float, float]:
    """Welch t-test returning (t_stat, p_value)."""
    result = _welch_t_test(values_a, values_b)
    if result.get("test") != "welch_t":
        return 0.0, 1.0
    t_stat = result.get("t")
    p_value = result.get("p_value")
    return float(t_stat) if t_stat is not None else 0.0, float(p_value) if p_value is not None else 1.0


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d effect size with Hedges' g correction."""
    return _compute_cohens_d(group1, group2)


def hedges_g(group1: List[float], group2: List[float]) -> float:
    """Hedges' g effect size (same as cohens_d with correction)."""
    return _compute_cohens_d(group1, group2)


def paired_t_test(
    differences: List[float],
) -> Tuple[float, float]:
    """Paired t-test returning (t_stat, p_value)."""
    result = _paired_t_test(differences)
    if result.get("test") != "paired_t":
        return 0.0, 1.0
    t_stat = result.get("t")
    p_value = result.get("p_value")
    return float(t_stat) if t_stat is not None else 0.0, float(p_value) if p_value is not None else 1.0
