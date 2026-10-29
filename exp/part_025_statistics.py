from __future__ import annotations



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
        entries.append({
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
        })

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
                + f"，alpha={float(entry['alpha']):.3f}）"
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
                f"（n={entry.get('n')}）"
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
                f"{entry['n_negative']}）"
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
            term_a * term_a / (n_a - 1)
            + term_b * term_b / (n_b - 1)
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
            scaled_a * scaled_a / (n_a - 1)
            + scaled_b * scaled_b / (n_b - 1)
        )
        if denominator <= 0.0 or not math.isfinite(denominator):
            return {"test": "invalid_values"}
        df = (scaled_a + scaled_b) ** 2 / denominator
    t = delta / math.sqrt(se_squared)
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

    T = (Z + δ)/sqrt(V/df)，Z ~ N(0,1)，V ~ chi2_df。令
    u = sqrt(V)，则 u 的密度为
    g(u) = 2^(1-df/2) u^(df-1) exp(-u^2/2) / Γ(df/2)，
    且 P(|T| > c | u) = Φ(-c·u/√df - δ) + 1 - Φ(c·u/√df - δ)。

    默认积分区间上界取 chi 分布众数 sqrt(df-1) 加
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
    log_const = (1.0 - df / 2.0) * math.log(2.0) - log_gamma

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
            log_u = exponent * math.log(t) if t > 0.0 else float("-inf")
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
                    math.log(simpson)
                    + log_const
                    + (df - 1.0) * log_u
                    - u * u / 2.0
                    + math.log(exponent)
                    + (exponent - 1.0) * math.log(t)
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
            # u^(df-1)：df>1 时为 0；df=1 时 g(0)=sqrt(2/π)。
            density = math.sqrt(2.0 / math.pi) if df == 1.0 else 0.0
        else:
            density = math.exp(
                log_const + (df - 1.0) * math.log(u) - u * u / 2.0
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

    令 T'=(Z+δ)/sqrt(V/df)、w=scale·u 且 u=sqrt(V)，则
    |T'|>critical 等价于 u<|Z+δ|/scale。因此

        power = E_Z[F_chi2_df((|Z+δ|/scale)^2)]。

    这比对 w 做等距 Simpson 更适合 δ 很大（如 1e5）的情形：
    w 积分的转折宽度约为 1，而区间长度约为 δ；固定 20000
    个区间时步长可达数个单位，会把 δ 附近的概率质量积错。
    这里对 Z 在 ±12 内积分，转折宽度不变，步长始终为 0.01。
    df 在 3–10 且 scale 不大时也走这里：u 网格在 0 附近受
    u^(df-1) 曲率影响，固定 Simpson 节点的误差可达 1e-5。
    """
    log_scale = math.log(scale)
    shape = df_value / 2.0
    log_gamma_plus_one = math.lgamma(shape + 1.0)

    def _chi_square_cdf_at_log(log_u: float) -> float:
        if not math.isfinite(log_u):
            return 0.0
        log_x = 2.0 * log_u - math.log(2.0)
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
        (1.0 - df_value / 2.0) * math.log(2.0)
        - math.lgamma(df_value / 2.0)
    )
    log_scale = math.log(scale)

    def _power_factor(w: float) -> float:
        return (
            _normal_cdf(-w - noncentrality)
            + _normal_cdf(noncentrality - w)
        )

    total = 0.0
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
            continue
        log_w = exponent * math.log(t)
        log_u = log_w - log_scale
        w = math.exp(log_w) if log_w >= -745.0 else 0.0
        factor = _power_factor(w)
        if factor <= 0.0:
            continue
        log_weight = (
            math.log(simpson)
            + log_const
            + (df_value - 1.0) * log_u
            - math.exp(2.0 * log_u) / 2.0
            - log_scale
            + math.log(exponent)
            + (exponent - 1.0) * math.log(t)
            + math.log(factor)
        )
        if log_weight > -745.0:
            total += math.exp(log_weight)

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
        step = (w_max - 1.0) / intervals
        for index in range(intervals + 1):
            w = 1.0 + index * step
            if index == 0 or index == intervals:
                simpson = step / 3.0
            elif index % 2 == 0:
                simpson = 2.0 * step / 3.0
            else:
                simpson = 4.0 * step / 3.0
            u = w / scale
            if u <= 0.0:
                continue
            factor = _power_factor(w)
            if factor <= 0.0:
                continue
            log_weight = (
                math.log(simpson)
                + log_const
                + (df_value - 1.0) * math.log(u)
                - u * u / 2.0
                - log_scale
                + math.log(factor)
            )
            if log_weight > -745.0:
                total += math.exp(log_weight)
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

    scale = critical / math.sqrt(df_value)

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
            (1.0 - df_value / 2.0) * math.log(2.0)
            - math.lgamma(df_value / 2.0)
        )
        total = 0.0
        for index in range(intervals + 1):
            w = index * step
            u = w / scale
            if index == 0 or index == intervals:
                simpson = step / 3.0
            elif index % 2 == 0:
                simpson = 2.0 * step / 3.0
            else:
                simpson = 4.0 * step / 3.0
            if u <= 0.0:
                density = (
                    math.sqrt(2.0 / math.pi)
                    if df_value == 1.0
                    else 0.0
                )
            else:
                density = math.exp(
                    log_const
                    + (df_value - 1.0) * math.log(u)
                    - u * u / 2.0
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
    se = sd·√(2/n)、df ≈ 2(n-1)。配对公式会低估所需种子数，
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
    a = value / math.sqrt(df)
    # 积分变量换成 x=a*u-δ 后，正态 CDF 只在 x∈[-40,40] 内
    # 有可表示的质量；对应 u 的上界是 (40+δ)/a（a>0）或
    # (40-δ)/(-a)（a<0）。默认 u_max=众数+12 只覆盖 chi 密度
    # 本身，非中心参数很大时转折点会落在更远的 chi 尾部：
    # 例如 df=766、δ≈97、t≈52.6 时主导区间在 u≈44，而默认
    # 上界仅 39.7，会把 CDF 低估十几个数量级。按转折点上界
    # 扩界后再做 x 积分，剩余尾部仍由下面的 chi CDF/生存函数
    # 精确补上。
    u_max = math.sqrt(df - 1.0) + _T_POWER_TAIL
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
        (1.0 - df / 2.0) * math.log(2.0)
        - math.lgamma(df / 2.0)
    )

    def density(u: float) -> float:
        if u <= 0.0:
            return 0.0
        log_density = (
            log_const
            + (df - 1.0) * math.log(u)
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
