from __future__ import annotations



def _noncentral_t_cdf(
    value: float,
    df: float,
    noncentrality: float,
) -> Optional[float]:
    """
    非中心 t 分布 CDF：T' = (Z + δ) / sqrt(V/df)。

    给定 V=u²（u 服从 chi_df）时 T' 条件于正态分布
    N(δ, 1)/sqrt(V/df)；对 u 的 chi 密度积分，与 ``_t_power``
    使用同一组 Simpson 节点。δ=0 时退化为中心 t，直接复用
    ``_t_cdf`` 以保持中心情形的数值路径不变；df 趋于无穷时
    退化为 N(δ, 1)。
    """
    # 先排除未定义输入：NaN 自由度/非中心参数/取值都没有 CDF。
    # 原实现把 df=NaN/-inf 误送进「大 df 正态极限」分支，返回
    # 一个看似合法的正态 CDF，调用方会把无效结果当成证据。
    if (
        math.isnan(value)
        or math.isnan(df)
        or math.isnan(noncentrality)
    ):
        return None
    if df <= 0.0:
        return None
    if noncentrality == 0.0:
        return _t_cdf(value, df)
    if math.isinf(noncentrality):
        # δ→±∞ 时 T' 几乎必然同号发散；两端都无定义时返回
        # None，避免把 +inf-+inf 之类的未定式猜成 1 或 0。
        if math.isinf(value):
            return None
        return 1.0 if noncentrality > 0.0 else 0.0
    if not math.isfinite(df) or df >= _T_POWER_NORMAL_DF:
        return _normal_cdf(value - noncentrality)
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0
    if value == 0.0:
        # T'=0 当且仅当 Z+δ<=0，与 V 无关；直接由正态 CDF 给出，
        # 避免低自由度截断积分少算 chi 尾部概率。
        return _normal_cdf(-noncentrality)
    if df >= 3.0 and abs(value) / math.sqrt(df) >= 0.05:
        return _noncentral_t_cdf_large_shift(
            value,
            df,
            noncentrality,
        )

    scale = 1.0 / math.sqrt(df)
    if df < 3.0 and value > 0.0 and noncentrality > 0.0:
        # 正态尾在 u 轴的转折位置约为
        # a*δ/(1+a²)（a=value/√df）。df<3 时默认 u 上界只有
        # √(df-1)+12，δ 较大而 value 较小时转折会落在截断之外，
        # 从而丢掉整个可表示的正尾概率。
        a = value * scale
        peak_u = a * noncentrality / (1.0 + a * a)
        default_u_max = (
            math.sqrt(df - 1.0) + _T_POWER_TAIL
            if df > 1.0
            else _T_POWER_TAIL
        )
        if peak_u > default_u_max - 2.0:
            extended_u_max = peak_u + _T_POWER_TAIL
            if math.isfinite(extended_u_max):
                u_values, weighted = _t_power_nodes(
                    df, extended_u_max
                )
                total = 0.0
                for u, weight in zip(u_values, weighted):
                    if weight == 0.0:
                        continue
                    total += weight * _normal_cdf(
                        value * u * scale - noncentrality
                    )
                return min(1.0, max(0.0, total))

    u_values, weighted = _t_power_nodes(df)
    total = 0.0
    for u, weight in zip(u_values, weighted):
        if weight == 0.0:
            continue
        total += weight * _normal_cdf(
            value * u * scale - noncentrality
        )
    return min(1.0, max(0.0, total))



def _tost_test(
    effect: float,
    se: float,
    df: float,
    margin: float,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Dict[str, object]:
    """
    等价性检验（TOST，two one-sided tests）。

    H0_upper: effect >= +margin（上单侧，p_upper = P(T <= (Δ̂-margin)/se)）
    H0_lower: effect <= -margin（下单侧，p_lower = P(T >= (Δ̂+margin)/se)）
    p_tost = max(p_upper, p_lower)；p_tost <= alpha 才成立
    「差异落在 ±margin 等价界内」的结论。

    margin 必须是预先声明的实际意义界（见
    _DEFAULT_EQUIVALENCE_MARGIN 与 --equivalence-margin）；
    margin 越宽越容易「等价」，因此其取值必须写进报告、
    不能看到数据后再调。返回 p 值为未校正口径，家族内
    多重比较校正由 _apply_multiple_comparison_correction 统一
    处理（不等价的误判同样会被多重比较放大）。
    """
    try:
        effect_value = float(effect)
        se_value = float(se)
        df_value = float(df)
        margin_value = abs(float(margin))
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return {}

    if (
        not math.isfinite(effect_value)
        or not math.isfinite(se_value)
        or se_value < 0.0
        or math.isnan(df_value)
        or not math.isfinite(margin_value)
        or not (0.0 < alpha_value < 0.5)
    ):
        return {}
    if margin_value <= 0.0 or df_value <= 0.0:
        return {}

    if se_value <= 0.0:
        # 零标准误：差异精确已知。界内即等价（p=0），
        # 界外即明显不等价（p=1）；恰好落在边界上保守
        # 地判为不等价。
        equivalent = abs(effect_value) < margin_value
        return {
            "tost_p_lower": 0.0 if equivalent else 1.0,
            "tost_p_upper": 0.0 if equivalent else 1.0,
            "tost_p_value": 0.0 if equivalent else 1.0,
            "tost_margin": margin_value,
        }

    t_upper = (effect_value - margin_value) / se_value
    t_lower = (effect_value + margin_value) / se_value

    cdf_upper = _t_cdf(t_upper, df_value)
    if cdf_upper is None:
        return {}

    p_upper = cdf_upper
    if t_lower >= 0.0:
        two_sided_lower = _two_sided_t_p_value(
            t_lower, df_value
        )
        if two_sided_lower is None:
            return {}
        p_lower = two_sided_lower / 2.0
    else:
        cdf_lower = _t_cdf(t_lower, df_value)
        if cdf_lower is None:
            return {}
        p_lower = 1.0 - cdf_lower
    return {
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p_value": max(p_lower, p_upper),
        "tost_margin": margin_value,
    }



def _tost_power_normal_mix(
    df_value: float,
    noncentrality: float,
    upper: float,
) -> float:
    """
    小自由度、大非中心参数下的 TOST 功效。

    令 T'=(Z+δ)/sqrt(V/df)、Y=Z+δ，则 |T'|≤U 等价于
    u=sqrt(V)≥|Y|·sqrt(df)/U。因此

        power = E_Y[Q_chi2_df(df·(|Y|/U)^2)]。

    直接对 Y 积分避开了非中心 t CDF 的固定 u 网格：δ 很大时
    条件正态尾的转折在 u 上非常窄，固定节点会系统性高估。
    """
    if math.isinf(upper):
        return 1.0
    # 等价界相对 sqrt(df) 很小时，Q_chi2(df·(|Y|/U)²) 在 Y=±U
    # 附近的转折宽度约为 U/sqrt(df)。原来的固定 0.01 网格会
    # 跨过整段转折（df=3026、U≈0.027 时约 0.0005 宽），
    # Simpson 严重低估功效（相对误差可达 14%）。此时改在
    # u=sqrt(V) 的 chi 节点上积分条件正态区间概率：
    #   P(|T'|≤U) = E_u[Φ(U·u/√df - δ) - Φ(-U·u/√df - δ)]，
    # 被积函数对 u 平滑，节点权重已按 chi 密度归一。
    if upper < math.sqrt(df_value):
        a = upper / math.sqrt(df_value)
        u_values, weighted = _t_power_nodes(df_value)
        total = 0.0
        for u, weight in zip(u_values, weighted):
            if weight == 0.0:
                continue
            total += weight * _normal_interval_probability(
                -a * u - noncentrality,
                a * u - noncentrality,
            )
        return min(1.0, max(0.0, total))

    total = 0.0
    intervals = 2400
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
        ratio = abs(y) / upper
        threshold = df_value * ratio * ratio
        if not math.isfinite(threshold):
            survival = 0.0
        else:
            survival = _chi_square_survival(
                threshold, df_value
            )
            if survival is None:
                survival = 0.0
        density = math.exp(-0.5 * t * t) / math.sqrt(
            2.0 * math.pi
        )
        total += simpson * density * survival
    return min(1.0, max(0.0, total))



def _tost_power(
    effect: float,
    se: float,
    df: float,
    margin: float,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Optional[float]:
    """
    等价性检验（TOST）在给定真实效应下的统计功效。

    TOST 的两个单侧检验同时通过，等价于标准化后的观测差值
    落在 ``±(margin - t_{1-alpha,df}·se)`` 内：

        P(TOST 通过) = F_nct(upper; df, δ)
                       - F_nct(lower; df, δ)

    其中 ``upper = (margin - c·se) / se``、
    ``lower = -upper``、``δ = effect / se``、
    ``c = t_{1-alpha,df}``。真实差异非零时观测差值服从非中心
    t 分布；只有 ``effect=0`` 时才退化为中心 t。``margin``
    相对基线均值时，``effect`` 与 ``se`` 也必须用同一相对
    尺度。

    ``margin <= c·se`` 时区间为空，功效为 0。返回 None 表示
    输入不可用，而不是「功效为零」。
    """
    try:
        effect_value = float(effect)
        se_value = float(se)
        df_value = float(df)
        margin_value = abs(float(margin))
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(effect_value)
        or not math.isfinite(se_value)
        or se_value < 0.0
        or math.isnan(df_value)
        or not math.isfinite(margin_value)
        or not (0.0 < alpha_value < 0.5)
        or df_value <= 0.0
        or margin_value <= 0.0
    ):
        return None
    if se_value <= 0.0:
        # 零标准误：差异精确已知，界内必然通过、界外必然不通过。
        return 1.0 if abs(effect_value) < margin_value else 0.0

    critical = _t_critical(df_value, 2.0 * alpha_value)
    if critical is None:
        return None
    half_width = margin_value - critical * se_value
    if half_width <= 0.0:
        return 0.0

    lower = -half_width / se_value
    upper = half_width / se_value
    noncentrality = effect_value / se_value
    if not math.isfinite(noncentrality):
        # se 极小导致 δ 溢出：取 δ→∞ 的极限。条件
        # |T'|≤U 等价于 (Z+δ)/sqrt(V/df) 落在 ±U 内；
        # 除以 δ 后只剩 sqrt(V/df) ≥ 1/r（r=U/δ），
        # 即 V ≥ df·(effect/half_width)²。忽略下界（极限为 0）。
        try:
            threshold = (
                df_value
                * (effect_value / half_width) ** 2
            )
        except OverflowError:
            return 0.0
        if not math.isfinite(threshold):
            return 0.0
        survival = _chi_square_survival(threshold, df_value)
        return survival if survival is not None else 0.0
    if df_value >= _T_POWER_NORMAL_DF:
        # 自由度趋于无穷时非中心 t 退化为 N(δ, 1)。不能继续
        # 走 _tost_power_normal_mix：df 很大时 chi 密度宽度仍
        # 约为 1，而 u 轴长度约为 sqrt(df)，固定节点上限会把
        # 整段密度跨过，甚至在小效应下返回 1。
        power = _normal_interval_probability(
            -upper - noncentrality,
            upper - noncentrality,
        )
        return min(1.0, max(0.0, power))
    if (
        abs(noncentrality) > 16.0
        or (df_value >= 2.0 and upper < 10.0)
    ):
        return _tost_power_normal_mix(
            df_value, noncentrality, upper
        )
    cdf_upper = _noncentral_t_cdf(
        upper, df_value, noncentrality
    )
    cdf_lower = _noncentral_t_cdf(
        lower, df_value, noncentrality
    )
    if cdf_upper is None or cdf_lower is None:
        return None
    return min(1.0, max(0.0, cdf_upper - cdf_lower))



def _seeds_for_tost(
    sd: float,
    margin: float,
    target_power: float = _TARGET_POWER,
    alpha: float = _SIGNIFICANCE_LEVEL,
    max_seeds: int = 128,
    variance_factor: float = 1.0,
) -> Optional[int]:
    """
    在真实差异为零、等价界为 ``margin`` 时达到目标 TOST 功效
    所需的可用配对数。

    与 ``_seeds_to_detect`` 一样把声明的种子间标准差当作真实
    sigma；``sd`` 是两格配对差值的标准差，``variance_factor``
    把主效应/交互的方差差异纳入（与 ``_CONTRAST_VARIANCE_FACTORS``
    同口径）。``margin`` 与 ``sd`` 必须是同一相对尺度。
    """
    try:
        sd_value = float(sd)
        margin_value = abs(float(margin))
        target = float(target_power)
        factor = float(variance_factor)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(sd_value)
        or sd_value < 0.0
        or not math.isfinite(margin_value)
        or not math.isfinite(target)
        or not math.isfinite(factor)
        or not (0.0 < target < 1.0)
        or margin_value <= 0.0
        or factor <= 0.0
    ):
        return None
    if sd_value <= 0.0:
        return _MIN_SEEDS_FOR_STATS

    sd_contrast = sd_value * math.sqrt(factor)
    for n in range(_MIN_SEEDS_FOR_STATS, max_seeds + 1):
        se = sd_contrast / math.sqrt(float(n))
        power = _tost_power(
            0.0, se, float(n - 1), margin_value, alpha
        )
        if power is not None and power >= target:
            return n
    return None



def _tost_margin_for_power(
    se: float,
    df: float,
    target_power: float = _TARGET_POWER,
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> Optional[float]:
    """
    当前精度下、真实差异为零时达到目标 TOST 功效所需的最小
    等价界。

    它回答「按当前种子数，我最多能主张多宽的等价性」：若该
    值大于预注册的 ``equivalence_margin``，说明当前设计即使
    真实差异为零也无法在声明界内主张等价，属于精度不足，
    应增加种子而不是放宽等价界。返回 None 表示不可计算。
    """
    try:
        se_value = float(se)
        df_value = float(df)
        target = float(target_power)
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(se_value)
        or se_value < 0.0
        or math.isnan(df_value)
        or not math.isfinite(target)
        or not (0.0 < target < 1.0)
        or not (0.0 < alpha_value < 0.5)
        or df_value <= 0.0
    ):
        return None
    if se_value <= 0.0:
        return 0.0

    # 功效只取决于 margin/se 的比值；直接在比值尺度上二分，
    # 避免极小 se 下固定区间 [0, 1] 的绝对分辨率不足。
    low = 0.0
    high = 1.0
    # alpha 极小时所需界约为 t 临界值，可能远大于 2^64；按
    # double 的完整指数范围扩展，而不是固定 64 次倍增后误报
    # 「不可计算」。
    for _ in range(2048):
        power = _tost_power(0.0, 1.0, df_value, high, alpha_value)
        if power is not None and power >= target:
            break
        high *= 2.0
        if not math.isfinite(high):
            return None
    else:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        power = _tost_power(
            0.0, 1.0, df_value, mid, alpha_value
        )
        if power is None:
            return None
        if power < target:
            low = mid
        else:
            high = mid
    # 返回二分上界：极小 alpha 下临界值本身已达 1e60 量级，
    # 中点可能因舍入落回 critical 而给出功效 0 的界；上界始终
    # 满足目标功效，代价只是至多一个 ulp 的保守性。
    margin = high * se_value
    return margin if math.isfinite(margin) else None



def _attach_sd_uncertainty_note(
    contrast: Dict[str, object],
) -> None:
    """
    标注「所需种子数」是按种子间标准差点估计外推的插件估计。

    ``seeds_to_detect`` / ``equivalence_seeds_to_achieve`` 都把
    当前 s 当成真实 σ 代入 se = σ/√n 外推，但 s 自身也是估计：
    正态假设下其相对标准误 ≈ 1/√(2(n-1))（n=3 时约 50%），
    而所需种子数 ∝ σ²，σ 的不确定度会被平方放大。不标注会让
    「约需 12 个种子」读起来像精确规划值。

    字段：``seeds_to_detect_basis`` = "sd_point_estimate"，
    ``seeds_to_detect_sd_relative_se`` = 1/√(2(n-1))（n<2 时
    省略）。重复调用只补缺失字段，保持幂等。
    """
    contrast["seeds_to_detect_basis"] = "sd_point_estimate"
    if contrast.get("seeds_to_detect_sd_relative_se") is not None:
        return
    current_n = _current_seed_count(contrast)
    if current_n is None or current_n <= 1.0:
        return
    contrast["seeds_to_detect_sd_relative_se"] = 1.0 / math.sqrt(
        2.0 * (current_n - 1.0)
    )



def _hedges_g_correction(
    df: object,
) -> Optional[float]:
    """
    Hedges g 的小样本偏倚校正因子 J ≈ 1 - 3/(4·df - 1)。

    Cohen's d（= |Δ|/合并标准差）在小样本下系统性高估真实效应：
    每臂 3 个种子时偏倚约 13%，配对口径 df=2 时约 43%。校正后
    的 g 是 d 的无偏近似，报告必须同时给两者，否则小种子数的
    效应量会被当成真实效应大小。df 不可用或校正因子非正时
    返回 None。
    """
    try:
        df_value = float(df)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(df_value) or df_value <= 1.0:
        return None
    correction = 1.0 - 3.0 / (4.0 * df_value - 1.0)
    if correction <= 0.0:
        return None
    return correction



def _paired_effect_size(
    delta: object,
    test: Dict[str, object],
) -> Dict[str, object]:
    """
    配对设计的标准化效应量：Cohen's dz 与 Hedges gz。

    ``delta_over_noise`` / Hedges g 用合并标准差，适合两臂独立
    的情形；同种子配对时种子间的共同难度会进入合并标准差，
    使标准化效应被低估。配对设计应以差值标准差为尺度：
    dz = |Δ| / sd(差值)，并由配对 t 的 se 反推
    sd(差值) = se·√n。只对 ``paired_t`` 且 n_pairs≥2 的结果
    计算；非配对/零方差返回空字典，不伪造效应量。
    """
    try:
        delta_value = abs(float(delta))
    except (TypeError, ValueError, OverflowError):
        return {}
    if test.get("test") != "paired_t":
        return {}
    n_pairs = test.get("n_pairs")
    if (
        not isinstance(n_pairs, int)
        or isinstance(n_pairs, bool)
        or n_pairs < 2
    ):
        return {}
    se_value = _finite_float(test.get("se"))
    if se_value is None or se_value < 0.0:
        return {}
    if se_value == 0.0:
        if delta_value == 0.0:
            return {"cohens_dz": 0.0}
        return {}
    sd_difference = se_value * math.sqrt(float(n_pairs))
    if sd_difference <= 0.0 or not math.isfinite(sd_difference):
        return {}
    dz = delta_value / sd_difference
    # 差值标准差可以合法地极小（如 1e-300），此时 |Δ|/sd 会越过
    # float 上限。效应量没有可表示的有限值时按缺失处理，不能把
    # inf 写进报告（JSON 无法严格解析）。
    if not math.isfinite(dz):
        return {}
    result: Dict[str, object] = {"cohens_dz": dz}
    correction = _hedges_g_correction(test.get("df"))
    if correction is not None:
        result["hedges_gz"] = dz * correction
    return result



def _observed_seed_sd(
    contrast: Dict[str, object],
    se_value: float,
) -> Optional[float]:
    """
    由检验口径的标准误还原观测到的（近似）种子间标准差。

    配对设计 se = sd/√n，直接 sd = se·√n。Welch 的
    se² = s_a²/n_a + s_b²/n_b；等方差下 se = s·√(1/n_a+1/n_b)，
    因此 s = se·√(n_a·n_b/(n_a+n_b))。用调和均值
    n_eq = 2·n_a·n_b/(n_a+n_b) 表示即 s = se·√(n_eq/2)；
    直接把 n_eq 当配对 n 会把 sd 高估 √2，进而把所需种子数
    高估约 2 倍。无法确定口径时返回 None。
    """
    n_pairs = contrast.get("n_pairs")
    n_baseline = contrast.get("n_baseline")
    n_variant = contrast.get("n_variant")
    if (
        isinstance(n_pairs, int)
        and not isinstance(n_pairs, bool)
        and n_pairs >= _MIN_SEEDS_FOR_STATS
    ):
        return se_value * math.sqrt(n_pairs)
    if (
        isinstance(n_baseline, int)
        and not isinstance(n_baseline, bool)
        and isinstance(n_variant, int)
        and not isinstance(n_variant, bool)
        and min(n_baseline, n_variant) >= _MIN_SEEDS_FOR_STATS
    ):
        n_equivalent = (
            2.0 * n_baseline * n_variant
            / (n_baseline + n_variant)
        )
        # Welch se 已包含两臂合并后的 √(1/n_a+1/n_b)，
        # 不能把 n_eq 当作配对样本数直接开方。
        return se_value * math.sqrt(n_equivalent / 2.0)
    return None



def _attach_family_power(
    contrast: Dict[str, object],
    family_size: int,
) -> None:
    """
    把家族校正口径下的功效量挂到对比记录上（原地更新）。

    MDE、观测效应功效、检出所需种子数与「主张等价所需的最小
    界」在 ``_attach_power_equivalence`` 里都按未校正的
    alpha=0.05 计算，而报告的 ``significant`` / ``equivalent``
    判定用的是 Holm 校正后的 p 值。二者口径不一致会系统性地
    高估现有实验的可判定性：Holm 第一步的阈值是 alpha/m，
    家族越大，能被判定为显著/等价的效应就必须越大。

    这里按最保守的第一步阈值 alpha/m（Bonferroni，Holm 阈值
    的下界）补算一组对应口径的量：

    - ``significance_family_size``：家族规模 m（实际执行的
      检验数）；``alpha_bonferroni`` = alpha/m（m=1 时退化为
      alpha）。
    - ``mde_80_bonferroni`` / ``power_at_effect_bonferroni`` /
      ``seeds_to_detect_bonferroni``：同一 se/df 下按
      alpha/m 计算的 MDE、观测效应功效与所需种子数。
    - ``equivalence_margin_needed_bonferroni``：按
      alpha/m 主张等价所需的最小对称界。

    家族规模为 1 时这些量与未校正口径相同，报告不会因此变长。
    """
    contrast["significance_family_size"] = int(family_size)
    size = int(family_size)
    alpha_family = (
        _SIGNIFICANCE_LEVEL / size if size > 1 else _SIGNIFICANCE_LEVEL
    )
    contrast["alpha_bonferroni"] = alpha_family

    se = contrast.get("se")
    df = contrast.get("df")
    try:
        se_value = float(se)  # type: ignore[arg-type]
        df_value = float(df)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return
    if (
        not math.isfinite(se_value)
        or not math.isfinite(df_value)
        or se_value <= 0.0
        or df_value <= 0.0
    ):
        return

    mde = _minimum_detectable_effect(
        se_value, df_value, alpha=alpha_family
    )
    if mde is not None:
        contrast["mde_80_bonferroni"] = mde

    delta = contrast.get("delta")
    try:
        delta_value = abs(float(delta))
    except (TypeError, ValueError, OverflowError):
        delta_value = None
    if delta_value is not None and math.isfinite(delta_value):
        power = _t_power(
            delta_value, se_value, df_value, alpha_family
        )
        if power is not None:
            contrast["power_at_effect_bonferroni"] = power
        sd_observed = _observed_seed_sd(contrast, se_value)
        if sd_observed is not None and delta_value > 0.0:
            design = (
                "welch"
                if contrast.get("test") == "welch_t"
                else "paired"
            )
            seeds_needed = _seeds_to_detect(
                delta_value,
                sd_observed,
                alpha=alpha_family,
                design=design,
            )
            if seeds_needed is not None:
                contrast["seeds_to_detect_bonferroni"] = seeds_needed
                contrast["seeds_to_detect_design"] = design

    if contrast.get("equivalence_margin_abs") is None:
        return
    if delta_value is None:
        return
    critical = _t_critical(df_value, 2.0 * alpha_family)
    if critical is None:
        return
    margin_needed = delta_value + critical * se_value
    if not math.isfinite(margin_needed):
        # 极大但有限的 se/delta 相加后仍可能越过 float 上限。
        # 无法表示所需等价界时省略该字段，不把 inf 落盘。
        return
    contrast["equivalence_margin_needed_bonferroni"] = margin_needed
    baseline_abs = None
    baseline_mean = _finite_float(
        contrast.get("mean_baseline")
    )
    if baseline_mean is not None and abs(baseline_mean) > 0.0:
        baseline_abs = abs(baseline_mean)
    if baseline_abs is not None:
        relative_needed = margin_needed / baseline_abs
        if math.isfinite(relative_needed):
            contrast[
                "equivalence_margin_needed_bonferroni_relative"
            ] = relative_needed



def _attach_power_equivalence(
    contrast: Dict[str, object],
    delta: Optional[float],
    test: Dict[str, object],
    equivalence_margin: Optional[float],
) -> None:
    """
    把功效与等价性口径挂到对比记录上（原地更新）。

    字段含义：
    - ``mde_80``：当前 se、df 下 80% 功效能检出的最小效应
      （绝对口径）；``mde_80_relative`` 为相对 |基线均值| 的
      比例。观测到的 |Δ| 小于 MDE 时，「不显著」没有信息量。
    - ``power_at_effect``：观测效应量自身的功效；远低于目标
      功效时，即使点估计看似差异不小也不能据此下结论。
    - ``seeds_to_detect``：按观测到的种子间波动，要在 80%
      功效下检出观测效应所需的（近似）种子数；远大于当前
      种子数说明实验规模不足以判定。
    - ``equivalence_margin`` / ``tost_*``：预先声明的等价界
      （相对基线均值）与 TOST 结果；``equivalent`` 为未校正
      的原始判定，家族校正由
      ``_apply_multiple_comparison_correction`` 统一完成。
    - ``equivalence_margin_needed``：在当前精度下主张等价所需的
      最小对称界（|Δ̂| + t_{1-α}·se）；``equivalence_blocker``
      区分「效应本身超界」与「精度不足」，后者额外给出
      ``equivalence_seeds_to_achieve``（近似所需种子数），
      避免把「TOST 未通过」一律读成「两者不同」。
    """
    se = test.get("se")
    df = test.get("df")
    if se is None or df is None:
        return
    try:
        se_value = float(se)
        df_value = float(df)
    except (TypeError, ValueError, OverflowError):
        return

    baseline_mean = _finite_float(
        contrast.get("mean_baseline")
    )
    baseline_abs = None
    if baseline_mean is not None and abs(baseline_mean) > 0.0:
        baseline_abs = abs(baseline_mean)

    mde = _minimum_detectable_effect(se_value, df_value)
    if mde is not None:
        contrast["mde_80"] = mde
        if baseline_abs is not None:
            mde_relative = mde / baseline_abs
            # |基线均值| 可以合法地极小：相对 MDE 会越过 float
            # 上限。无法表示的相对量按缺失处理，不能写 inf。
            if math.isfinite(mde_relative):
                contrast["mde_80_relative"] = mde_relative

    if delta is not None and math.isfinite(float(delta)):
        delta_value = float(delta)
        power = _t_power(abs(delta_value), se_value, df_value)
        if power is not None:
            contrast["power_at_effect"] = power

        sd_observed = _observed_seed_sd(contrast, se_value)
        if sd_observed is not None and delta_value != 0.0:
            design = (
                "welch"
                if contrast.get("test") == "welch_t"
                else "paired"
            )
            seeds_needed = _seeds_to_detect(
                delta_value,
                sd_observed,
                design=design,
            )
            if seeds_needed is not None:
                contrast["seeds_to_detect"] = seeds_needed
                contrast["seeds_to_detect_design"] = design
                _attach_sd_uncertainty_note(contrast)

    if equivalence_margin is None or delta is None:
        return
    try:
        margin_relative = float(equivalence_margin)
    except (TypeError, ValueError, OverflowError):
        return
    if (
        not math.isfinite(margin_relative)
        or margin_relative <= 0.0
    ):
        return
    contrast["equivalence_margin"] = margin_relative
    if baseline_abs is None:
        # 基线均值为 0 时相对等价界没有定义；不虚构口径。
        contrast["equivalence_margin_abs"] = None
        return

    margin_abs = margin_relative * baseline_abs
    if not math.isfinite(margin_abs):
        # 相对界与基线均值都有限，但绝对界可能超出 float 范围；
        # 此时无法构造 TOST 的实际意义界，按不可用记录。
        contrast["equivalence_margin_abs"] = None
        return
    contrast["equivalence_margin_abs"] = margin_abs
    tost = _tost_test(
        float(delta),
        se_value,
        df_value,
        margin_abs,
    )
    if not tost:
        return
    contrast.update(tost)
    contrast["equivalent"] = bool(
        float(tost["tost_p_value"]) <= _SIGNIFICANCE_LEVEL
    )

    # 主张「等价」所需的最小对称界：TOST 通过当且仅当
    # (1-2α) 置信区间整体落在 ±margin 内，因此
    # margin_needed = |Δ̂| + t_{1-α}(df)·se（_t_critical 的
    # alpha 是双侧口径，故这里传 2α 取到单侧 t_{1-α}）。它把
    # 「TOST 未通过」
    # 拆成两种完全不同的情况：点估计本身已超出界（效应真实，
    # 加种子只会让「不等价」更确定），还是点估计在界内但噪声
    # 太大（精度不足，加种子可以判定）。不给这个量时，
    # 「TOST 未通过」会被一律读成「两者不同」。
    critical = _t_critical(df_value, 2.0 * _SIGNIFICANCE_LEVEL)
    if critical is not None:
        noise_half_width = critical * se_value
        margin_needed = abs(float(delta)) + noise_half_width
        if not math.isfinite(margin_needed):
            # 所需界超出 float 上限：只能判定为精度不足，且不写
            # 入任何非有限数值（否则 JSON 报告无法严格解析）。
            contrast["equivalence_blocker"] = (
                "insufficient_precision"
            )
        else:
            contrast["equivalence_margin_needed"] = margin_needed
            if baseline_abs is not None:
                relative_needed = margin_needed / baseline_abs
                if math.isfinite(relative_needed):
                    contrast[
                        "equivalence_margin_needed_relative"
                    ] = relative_needed
            if abs(float(delta)) >= margin_abs:
                # 点估计已经在界外：这不是功效问题，是效应本身
                # 超出实际意义界。
                contrast["equivalence_blocker"] = (
                    "effect_exceeds_margin"
                )
            elif margin_needed > margin_abs:
                contrast["equivalence_blocker"] = (
                    "insufficient_precision"
                )
                # 在种子间标准差不变的近似下，se ∝ 1/√n，要把噪声
                # 半宽从 (needed-|Δ|) 压到 (margin-|Δ|)，需要
                # n_needed = n·((needed-|Δ|)/(margin-|Δ|))^2。
                current_n = _current_seed_count(contrast)
                denominator = margin_abs - abs(float(delta))
                if (
                    current_n is not None
                    and current_n > 0
                    and denominator > 0.0
                    and math.isfinite(noise_half_width)
                ):
                    # 合法但极大的 se 会让 ratio² 溢出；此时无法给
                    # 出有限的所需种子数，显式省略该字段，而不是让
                    # int(math.ceil(inf)) 抛 OverflowError 中断整份
                    # 对比报告。正常量级保留原公式以维持历史数值。
                    ratio = noise_half_width / denominator
                    try:
                        required = current_n * ratio ** 2
                    except OverflowError:
                        required = None
                    if required is not None and math.isfinite(required):
                        contrast["equivalence_seeds_to_achieve"] = int(
                            math.ceil(required)
                        )
                        _attach_sd_uncertainty_note(contrast)
            else:
                contrast["equivalence_blocker"] = None



def _finite_probability(value: object) -> Optional[float]:
    """有限且位于 [0, 1] 的 p 值；损坏记录返回 None。"""
    numeric = _finite_float(value)
    if numeric is None or not 0.0 <= numeric <= 1.0:
        return None
    return numeric



def _holm_bonferroni(
    p_values: List[Optional[float]],
) -> List[Optional[float]]:
    """
    Holm step-down 法校正 p 值（控制族错误率 FWER）。

    第 i 小（0 起）的 p 值乘以 m - i 并做单调化；None
    （未执行检验，如种子不足）不参与家族计数。
    """
    numeric = [_finite_probability(p) for p in p_values]
    total = sum(1 for p in numeric if p is not None)
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if total == 0:
        return adjusted

    ranked = sorted(
        (index for index, p in enumerate(numeric) if p is not None),
        key=lambda index: float(numeric[index]),
    )

    running = 0.0
    for position, index in enumerate(ranked):
        candidate = min(
            1.0,
            float(numeric[index]) * (total - position),
        )
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted



def _holm_rank_thresholds(
    p_values: List[Optional[float]],
    alpha: float = _SIGNIFICANCE_LEVEL,
) -> List[Optional[float]]:
    """
    Holm step-down 每个检验位置对应的显著性阈值 alpha/(m-rank+1)。

    脆弱性分析要按「该对比在家族中的实际位置」比较删除种子后
    的 p 值；统一用 alpha/m 会把 Holm 后段本应显著的对比误判为
    不显著。删除种子不会改变家族规模，因此沿用原位置阈值是
    保守近似（删除后 p 排序可能变化，但不会放宽判定）。
    """
    numeric = [_finite_probability(p) for p in p_values]
    total = sum(1 for p in numeric if p is not None)
    thresholds: List[Optional[float]] = [None] * len(p_values)
    if total == 0:
        return thresholds
    ranked = sorted(
        (index for index, p in enumerate(numeric) if p is not None),
        key=lambda index: float(numeric[index]),
    )
    for position, index in enumerate(ranked):
        thresholds[index] = float(alpha) / float(total - position)
    return thresholds



def _benjamini_hochberg(
    p_values: List[Optional[float]],
) -> List[Optional[float]]:
    """
    Benjamini-Hochberg 法校正 p 值（控制错误发现率 FDR）。

    比 Holm 更有功效，适合探索性解读；结论性的显著性判定
    以 Holm（FWER）为准，BH 值随报告给出以标明「更宽松的
    口径下也不显著」的稳健性。
    """
    numeric = [_finite_probability(p) for p in p_values]
    total = sum(1 for p in numeric if p is not None)
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if total == 0:
        return adjusted

    ranked = sorted(
        (index for index, p in enumerate(numeric) if p is not None),
        key=lambda index: float(numeric[index]),
    )

    running = 1.0
    for position in range(len(ranked) - 1, -1, -1):
        index = ranked[position]
        candidate = min(
            1.0,
            float(numeric[index]) * total / (position + 1),
        )
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted



def _metric_samples(
    group_rows: List[Dict[str, object]],
    metric: str,
) -> List[Dict[str, object]]:
    """
    提取一组运行在某指标上的 (seed, run, value) 样本。
    """
    samples: List[Dict[str, object]] = []
    for row in group_rows:
        value = _finite_float(row.get(metric))
        if value is not None:
            samples.append({
                "seed": row.get("seed"),
                "run": row.get("run"),
                "value": value,
            })
    return samples



def _collapse_seed_samples(
    samples: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    把同一随机种子的多个样本折叠为一个独立观测（取均值）。

    伪重复（复制的同一运行 / 同一 seed 跑了多个目录）不是独立
    重复实验：聚合 Mean ± Std 与 Welch 检验都必须以「独立种子」
    为单位，否则复制运行会虚增样本数、压低种子间方差并制造
    假阳性。未记录种子的样本无法判断是否重复，保持各自独立
    （只扣除确定的伪重复，不凭空减少样本）。
    """
    grouped: Dict[object, List[Dict[str, object]]] = {}
    order: List[object] = []
    unseeded: List[Dict[str, object]] = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        seed = sample.get("seed")
        # 非有限/超大整数值无法参与均值与方差；提前跳过而不是让
        # 单个损坏样本在 float() 处炸掉整份聚合报告。
        if _finite_float(sample.get("value")) is None:
            continue
        if seed is None:
            unseeded.append(sample)
            continue
        key = _stable_factor_key(seed)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(sample)

    collapsed: List[Dict[str, object]] = []
    for key in order:
        group = grouped[key]
        values = [
            float(_finite_float(sample.get("value")))
            for sample in group
        ]
        collapsed.append({
            "seed": group[0].get("seed"),
            "run": ", ".join(
                str(sample.get("run")) for sample in group
            ),
            "value": _mean(values),
            "n_runs": len(group),
            # 同一 seed 的复制运行若给出不同值，说明它们并非
            # 逐位相同的确定性复制；均值折叠可以避免虚增 n，
            # 但必须把这份差异显式记录，不能悄悄抹平。
            "within_seed_sd": (
                _sample_std(values)
                if len(group) > 1
                else None
            ),
        })
    collapsed.extend(unseeded)
    return collapsed
