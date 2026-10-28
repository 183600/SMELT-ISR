from __future__ import annotations



def run_order_balance_report(
    plan: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    检查区组内位置是否平衡。

    完整区组设计只保证「每个区组都跑每个处理」，不保证每个
    处理在各位置上的出现次数相等。纯随机化在小样本下可能让
    某个臂 5 次里有 4 次排在区组最后；若位置与机器热漂移/
    缓存状态相关，臂效应就仍会混入时间效应。这里报告每个
    处理在各位置上的计数与最大失衡，供 `design_plan.json`
    在设计阶段留证，而不是等分析期才从结果反推。
    """
    if not plan:
        return {
            "available": False,
            "balanced": None,
            "reason": "计划为空",
        }
    blocks: Dict[object, List[Dict[str, object]]] = {}
    for item in plan:
        block_key = item.get("block")
        if block_key is None:
            block_key = item.get("seed")
        blocks.setdefault(_stable_factor_key(block_key), []).append(
            item
        )

    counts: Dict[object, Dict[int, int]] = {}
    block_treatment_sets: List[set] = []
    malformed = 0
    max_position = 0
    for entries in blocks.values():
        seen = set()
        per_block = set()
        for item in entries:
            treatment = _plan_treatment_key(item)
            position = item.get("position_in_block")
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 1
            ):
                malformed += 1
                continue
            max_position = max(max_position, position)
            per_block.add(treatment)
            if treatment in seen:
                malformed += 1
                continue
            seen.add(treatment)
            position_counts = counts.setdefault(treatment, {})
            position_counts[position] = (
                position_counts.get(position, 0) + 1
            )
        block_treatment_sets.append(per_block)

    if not counts:
        return {
            "available": False,
            "balanced": None,
            "reason": "计划没有可识别的区组内位置",
        }
    common = set.intersection(*block_treatment_sets) if block_treatment_sets else set()
    union = set().union(*block_treatment_sets) if block_treatment_sets else set()
    complete_blocks = (
        not malformed
        and bool(block_treatment_sets)
        and all(
            block_set == common for block_set in block_treatment_sets
        )
        and len(common) == len(union)
    )
    max_imbalance = 0
    details: List[Dict[str, object]] = []
    for treatment in sorted(counts, key=repr):
        values = [
            counts[treatment].get(position, 0)
            for position in range(1, max_position + 1)
        ]
        if not values:
            continue
        span = max(values) - min(values)
        max_imbalance = max(max_imbalance, span)
        details.append({
            "treatment": repr(treatment),
            "position_counts": {
                str(position): count
                for position, count in enumerate(values, start=1)
            },
            "max_position_imbalance": span,
        })
    balanced = bool(complete_blocks and max_imbalance <= 1)
    return {
        "available": True,
        "n_blocks": len(blocks),
        "n_treatments": len(common),
        "complete_blocks": complete_blocks,
        "max_position_imbalance": max_imbalance,
        "balanced": balanced,
        "treatments": details,
        "note": (
            "均衡定义为每个区组包含同一组处理各一次，且每个"
            "处理在各位置上的计数极差 <= 1。counterbalanced "
            "按循环拉丁方保证该性质；randomized 只保证区组内"
            "随机化，不保证位置计数均衡"
        ),
    }



def order_sweep_plan(
    plan: List[Dict[str, object]],
    scheme: str = "randomized",
    order_seed: int = 0,
) -> List[Dict[str, object]]:
    """
    给运行计划加上执行顺序与区组信息。

    - ``grouped``：保留原顺序（历史行为；模式与时间混杂）；
    - ``blocked``：按种子组成完整区组，区组内保持原臂顺序；
    - ``randomized``：按种子组成完整区组，区组内随机化臂
      顺序（随机化完整区组设计）；
    - ``counterbalanced``：按种子组成完整区组，并用确定性
      循环拉丁方让每个处理在各区组位置上的出现次数尽可能
      相等（位置平衡/限制性随机化）。它要求每个区组包含同一
      组处理单元，否则无法定义平衡排列并直接拒绝。

    同一区组内每个臂出现一次，因此「先跑/后跑」在臂之间被
    平衡，而不是让某个臂系统性地占据更晚的时间片。
    """
    if scheme not in _RUN_ORDER_SCHEMES:
        raise ValueError(f"未知的运行顺序方案: {scheme}")

    if scheme == "grouped":
        ordered = [dict(item) for item in plan]
    else:
        blocks: Dict[object, List[Dict[str, object]]] = {}
        for item in plan:
            # seed 来自损坏/手工编辑的计划时可能是 list/dict；
            # 直接当 dict 键会抛 unhashable。稳定键只用于分组，
            # item["seed"] 仍保留原始值供报告/子进程使用。
            block_key = _order_block_key(item.get("seed"))
            blocks.setdefault(block_key, []).append(dict(item))
        if scheme == "counterbalanced":
            ordered = _counterbalanced_order(blocks, order_seed)
        else:
            ordered = []
            for block_key, block in blocks.items():
                if scheme == "randomized":
                    block_seed = (
                        block[0].get("seed")
                        if block
                        else block_key
                    )
                    rng = _order_block_rng(
                        order_seed,
                        _order_block_seed(block_seed),
                    )
                    rng.shuffle(block)
                ordered.extend(block)

    positions: Dict[object, int] = {}
    for index, item in enumerate(ordered, start=1):
        block_seed = item.get("seed")
        block_key = _order_block_key(block_seed)
        positions[block_key] = positions.get(block_key, 0) + 1
        item["order_index"] = index
        item["block"] = block_seed
        item["position_in_block"] = positions[block_key]
    return ordered



def planned_comparisons_for_plan(
    plan: List[Dict[str, object]],
) -> int:
    """
    计划中可能执行的比较数（用于保守的家族校正）。

    单臂/多种子设计没有跨臂比较，家族规模按 1 计；多处理单元
    设计按不同处理单元的两两组合数计。处理单元不能只看
    ``mode``：消融计划里同一模式还有 ``balance_coef``=0/当前
    值两档，忽略它会系统性少算家族规模，让所需种子数显得更
    少。用全部处理轴去重后再取两两组合数，是全因子主效应/
    交互与单因子对比的保守上界。
    """
    identity_keys = {
        "label",
        "seed",
        "order_index",
        "block",
        "position_in_block",
    }
    # 收集真实键而不是 str(key)：旧/手工计划的键可能不是字符串，
    # 转成字符串后再 item.get("1") 会把所有值读成 None，导致
    # 不同处理单元被静默合并。
    treatment_keys = sorted(
        {
            key
            for item in plan
            for key in item
            if str(key) not in identity_keys
        },
        key=str,
    )
    cells = {
        tuple(
            _stable_factor_key(item.get(key))
            for key in treatment_keys
        )
        for item in plan
    }
    pairwise = (
        len(cells) * (len(cells) - 1) // 2
        if len(cells) >= 2
        else 1
    )
    # 消融还包含每个延迟合并方案的 2×2 主效应/交互家族：
    # 主效应 2 + 交互 1 + 简单效应 4 = 7 个检验。两模式
    # （local_base + 一个延迟方案）时 7 大于 4 个单元的 6 个
    # 两两组合；只取两两组合仍会低估，因此显式取上界。
    varying_keys = {
        key
        for position, key in enumerate(treatment_keys)
        if len({cell[position] for cell in cells}) > 1
    }
    mode_values = {
        str(item.get("mode"))
        for item in plan
        if item.get("mode") is not None
    }
    factorial_upper = 0
    if (
        "mode" in varying_keys
        and "balance_coef" in varying_keys
        and "local_base" in mode_values
    ):
        factorial_upper = 7 * len(
            mode_values & set(_DELAYED_MERGE_MODES)
        )
    return max(pairwise, factorial_upper, 1)



def planned_contrast_types_for_plan(
    plan: List[Dict[str, object]],
) -> List[str]:
    """计划实际会估计哪些类型的对比（用于先验功效，不用于显著性）。

    - 普通多种子/单臂：没有跨臂对比；
    - ``--suite-modes``：只有两两对比；
    - ``--suite-ablation``（mode × balance 两轴且含 local_base）：
      除两两对比外还会估计两条主效应、一条交互与四条简单效应。
    功效门槛必须按这些类型中最吃种子的那一种判定，不能只按
    两两对比的方差。
    """
    if not plan:
        return []
    identity_keys = {
        "label",
        "seed",
        "order_index",
        "block",
        "position_in_block",
    }
    # 收集真实键而不是 str(key)：旧/手工计划的键可能不是字符串，
    # 转成字符串后再 item.get("1") 会把所有值读成 None，导致
    # 不同处理单元被静默合并。
    treatment_keys = sorted(
        {
            key
            for item in plan
            for key in item
            if str(key) not in identity_keys
        },
        key=str,
    )
    cells = {
        tuple(
            _stable_factor_key(item.get(key))
            for key in treatment_keys
        )
        for item in plan
    }
    if len(cells) < 2:
        return []
    varying_keys = {
        key
        for position, key in enumerate(treatment_keys)
        if len({cell[position] for cell in cells}) > 1
    }
    mode_values = {
        str(item.get("mode"))
        for item in plan
        if item.get("mode") is not None
    }
    if (
        "mode" in varying_keys
        and "balance_coef" in varying_keys
        and "local_base" in mode_values
        and mode_values & set(_DELAYED_MERGE_MODES)
    ):
        return ["pairwise", "main", "interaction", "simple"]
    return ["pairwise"]



_DESIGN_PILOT_SCHEMA_VERSION = 1

# 试点证据只为「两格配对差值」的 sd 提供校准；少于 3 个配对
# 种子时样本 sd 太不稳定，不作为硬门依据（仍报告点估计）。
_DESIGN_PILOT_MIN_PAIRS = _MIN_SEEDS_FOR_STATS

# 试点校准的默认单侧置信水平只用于报告上界；是否按上界做设计
# 必须由预注册的 power.pilot_calibration 声明，不能看到试点后
# 再在 point / upper_confidence_bound 之间挑选。
_PILOT_DEFAULT_CONFIDENCE = 0.95

_PILOT_CALIBRATION_POLICIES = (
    "point",
    "upper_confidence_bound",
)

_PILOT_UCB_SENSITIVITY_LEVELS = (0.8, 0.9, 0.95, 0.99)



_GAMMA_TEMME_MIN_A = 1.0e5

_GAMMA_TEMME_ETA_TERMS = (
    1.0 / 2.0,
    -1.0 / 3.0,
    1.0 / 4.0,
    -1.0 / 5.0,
    1.0 / 6.0,
    -1.0 / 7.0,
    1.0 / 8.0,
    -1.0 / 9.0,
    1.0 / 10.0,
    -1.0 / 11.0,
    1.0 / 12.0,
    -1.0 / 13.0,
    1.0 / 14.0,
    -1.0 / 15.0,
)

_GAMMA_TEMME_C0 = (
    474671.0 / 493807104000.0,
    -33113.0 / 17635968000.0,
    -571.0 / 235146240.0,
    1.0 / 25515.0,
    -139.0 / 777600.0,
    1.0 / 2835.0,
    1.0 / 864.0,
    -2.0 / 135.0,
    1.0 / 12.0,
    -1.0 / 3.0,
)

_GAMMA_TEMME_C1 = (
    33113.0 / 211631616000.0,
    1092991.0 / 123451776000.0,
    -5759.0 / 314928000.0,
    -353.0 / 167961600.0,
    1.0 / 4860.0,
    -77.0 / 77760.0,
    1.0 / 378.0,
    -1.0 / 288.0,
    -1.0 / 540.0,
)



def _gamma_parts_temme(a: float, x: float) -> Tuple[float, float]:
    delta = x - a
    if delta == 0.0:
        eta = 0.0
    else:
        mu = delta / a
        if abs(mu) < 1.0e-3:
            series = 0.0
            for coefficient in reversed(
                _GAMMA_TEMME_ETA_TERMS
            ):
                series = series * mu + coefficient
            eta_squared = 2.0 * mu * mu * series
        else:
            ratio = x / a
            if ratio == 0.0:
                return 0.0, 1.0
            eta_squared = 2.0 * (
                ratio - 1.0 - math.log(ratio)
            )
        if eta_squared < 0.0:
            eta_squared = 0.0
        eta = math.copysign(math.sqrt(eta_squared), delta)

    if abs(eta) < 1.0:
        c0 = 0.0
        for coefficient in _GAMMA_TEMME_C0:
            c0 = c0 * eta + coefficient
        c1 = 0.0
        for coefficient in _GAMMA_TEMME_C1:
            c1 = c1 * eta + coefficient
    else:
        mu = delta / a
        inverse_mu = 1.0 / mu
        inverse_eta = 1.0 / eta
        c0 = inverse_mu - inverse_eta
        c1 = (
            inverse_eta ** 3
            - inverse_mu ** 3
            - inverse_mu ** 2
            - inverse_mu / 12.0
        )

    z = eta * math.sqrt(a / 2.0)
    if z > 40.0:
        return 1.0, 0.0
    if z < -40.0:
        return 0.0, 1.0
    correction = (
        math.exp(-0.5 * a * eta * eta)
        / (math.sqrt(2.0 * math.pi) * math.sqrt(a))
        * (c0 + c1 / a)
    )
    lower = 0.5 * math.erfc(-z) - correction
    upper = 0.5 * math.erfc(z) + correction
    return (
        min(1.0, max(0.0, lower)),
        min(1.0, max(0.0, upper)),
    )



def _regularized_incomplete_gamma_parts(
    a: float,
    x: float,
) -> Optional[Tuple[float, float]]:
    """同时返回正则化不完全 Gamma 的下尾 P(a,x) 与上尾 Q(a,x)。

    P 与 Q 互补，但靠近 1 的一侧若用 ``1 - 另一侧`` 计算会发生
    灾难性抵消。这里在小 x 时由级数直接得到 P、在大 x 时由连
    分式直接得到 Q，只把远离 1 的一侧取补。
    """
    a_value = _finite_float(a)
    x_value = _finite_float(x)
    if (
        a_value is None
        or x_value is None
        or a_value <= 0.0
        or x_value < 0.0
    ):
        return None
    a = a_value
    x = x_value
    if x == 0.0:
        return 0.0, 1.0
    if a >= _GAMMA_TEMME_MIN_A:
        return _gamma_parts_temme(a, x)
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for index in range(1, 10000):
            term *= x / (a + index)
            total += term
            if abs(term) <= abs(total) * 1.0e-15:
                break
        lower = total * math.exp(log_prefactor)
        upper = 1.0 - lower
    else:
        # Lentz 连分式计算 Q(a, x)。
        fpmin = 1.0e-300
        b = x + 1.0 - a
        c = 1.0 / fpmin
        d = 1.0 / b
        h = d
        for index in range(1, 10000):
            numerator = -index * (index - a)
            b += 2.0
            d = numerator * d + b
            if abs(d) < fpmin:
                d = fpmin
            c = b + numerator / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= 1.0e-15:
                break
        upper = math.exp(log_prefactor) * h
        lower = 1.0 - upper
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    return (
        min(1.0, max(0.0, lower)),
        min(1.0, max(0.0, upper)),
    )



def _regularized_incomplete_gamma(
    a: float,
    x: float,
) -> Optional[float]:
    """正则化下不完全 Gamma 函数 P(a, x)，无 SciPy 依赖。"""
    parts = _regularized_incomplete_gamma_parts(a, x)
    return None if parts is None else parts[0]



def _regularized_incomplete_gamma_upper(
    a: float,
    x: float,
) -> Optional[float]:
    """正则化上不完全 Gamma 函数 Q(a, x)，无 SciPy 依赖。"""
    parts = _regularized_incomplete_gamma_parts(a, x)
    return None if parts is None else parts[1]



def _chi_square_cdf(value: float, df: float) -> Optional[float]:
    value_value = _finite_float(value)
    df_value = _finite_float(df)
    if (
        value_value is None
        or df_value is None
        or value_value < 0.0
        or df_value <= 0.0
    ):
        return None
    return _regularized_incomplete_gamma(
        df_value / 2.0,
        value_value / 2.0,
    )



def _chi_square_survival(
    value: float,
    df: float,
) -> Optional[float]:
    """卡方分布的上尾概率 P(X > value)。"""
    value_value = _finite_float(value)
    df_value = _finite_float(df)
    if (
        value_value is None
        or df_value is None
        or value_value < 0.0
        or df_value <= 0.0
    ):
        return None
    return _regularized_incomplete_gamma_upper(
        df_value / 2.0,
        value_value / 2.0,
    )



def _chi_square_quantile(
    probability: float,
    df: float,
) -> Optional[float]:
    """卡方分布的下侧分位数（0 < p < 1），二分求逆。"""
    probability_value = _finite_float(probability)
    df_value = _finite_float(df)
    if (
        probability_value is None
        or df_value is None
        or not 0.0 < probability_value < 1.0
        or df_value <= 0.0
    ):
        return None
    probability = probability_value
    df = df_value
    # 上侧概率直接由 Q(a,x) 计算；若仍用 1-P(a,x)，p 接近 1
    # 时会把 Q 的绝对舍入误差放大成相对误差，使高分位数偏移
    # （df=2、p=1-1e-15 时约 0.16%）。
    upper_tail = probability > 0.5
    target = 1.0 - probability if upper_tail else probability

    def tail(value: float) -> Optional[float]:
        if upper_tail:
            survival = _chi_square_survival(value, df)
            return survival
        return _chi_square_cdf(value, df)

    low = 0.0
    high = max(1.0, df)
    for _ in range(100):
        value = tail(high)
        if value is None:
            return None
        reached = (
            value <= target if upper_tail else value >= target
        )
        if reached:
            break
        high *= 2.0
        if high > 1.0e300:
            return None
    else:
        return None
    # 每一步二分只增加 1 bit 分辨率；双精度浮点的指数跨度约
    # 2100 bit（从最大有限值到最小次正规数）。固定 200 步会把
    # 下侧分位数的分辨率卡在 2^-200 左右：p=1e-100、df=2 时
    # 会把 2e-100 算成 6.2e-61。迭代上限按完整指数范围设置，
    # 相对容差仍会让常规概率在几十步内提前收敛。
    for _ in range(4096):
        middle = (low + high) / 2.0
        value = tail(middle)
        if value is None:
            return None
        if upper_tail:
            if value > target:
                low = middle
            else:
                high = middle
        else:
            if value < target:
                low = middle
            else:
                high = middle
        # 只用相对容差：下侧分位数可能远小于 1e-12
        # （例如 confidence=1-1e-15、df=1 时约 1e-30），
        # 绝对容差会在这里提前停止并高估几个数量级。
        if high - low <= max(
            math.ulp(high) * 4.0, high * 1.0e-12
        ):
            break
    return (low + high) / 2.0



def _sd_upper_confidence_bound(
    sd: Optional[float],
    n: int,
    confidence: float,
) -> Optional[float]:
    """正态配对差值的 sd 单侧置信上界。

    ``sd`` 为样本 sd（ddof=1），``n`` 为配对数。按
    ``sigma <= s * sqrt((n-1) / chi2_{1-confidence, n-1})``
    计算；小试点时上界会很宽，这正是需要披露的抽样不确定性。
    """
    value = _finite_float(sd)
    confidence_value = _finite_float(confidence)
    if (
        value is None
        or value < 0.0
        or isinstance(n, bool)
        or not isinstance(n, int)
        or n < 2
        or confidence_value is None
        or not 0.0 < confidence_value < 1.0
    ):
        return None
    degrees = _finite_float(n - 1)
    if degrees is None:
        return None
    quantile = _chi_square_quantile(
        1.0 - confidence_value,
        degrees,
    )
    if quantile is None or quantile <= 0.0:
        return None
    try:
        upper = value * math.sqrt(degrees / quantile)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(upper):
        return None
    return max(0.0, upper)



def _binomial_lower_tail_probability(
    failures: int,
    trials: int,
    probability: float,
) -> float:
    """P(X <= failures) for X~Binomial(trials, probability)。

    只用于 failures 很小的情形：逐项累加对数 PMF，避免
    Clopper-Pearson 在 p 极小时走不完全 Beta 互补分支的
    舍入误差。项数由调用方限制。
    """
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0 if failures < trials else 1.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_term = trials * log_q
    total = math.exp(log_term) if log_term > -745.0 else 0.0
    for successes in range(1, failures + 1):
        log_term += (
            math.log((trials - successes + 1) / successes)
            + log_p
            - log_q
        )
        if log_term > -745.0:
            total += math.exp(log_term)
    return min(1.0, max(0.0, total))



def _binomial_upper_confidence_bound(
    failures: int,
    trials: int,
    confidence: float,
) -> Optional[float]:
    """二项比例的单侧 Clopper-Pearson 精确置信上界。

    解 ``P(X <= failures; trials, p) = 1-confidence``。全部
    运行失败时上界为 1；该单元本身已由 ``unusable`` 标记处理。
    """
    confidence_value = _finite_float(confidence)
    if (
        isinstance(failures, bool)
        or not isinstance(failures, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials <= 0
        or failures < 0
        or failures > trials
        or confidence_value is None
        or not 0.0 < confidence_value < 1.0
    ):
        return None
    if failures >= trials:
        return 1.0
    alpha = 1.0 - confidence_value
    if failures == 0:
        try:
            return 1.0 - alpha ** (1.0 / float(trials))
        except (OverflowError, ValueError):
            return None
    remaining = _finite_float(trials - failures)
    successes = _finite_float(failures + 1)
    if remaining is None or successes is None:
        return None
    low = 0.0
    high = 1.0
    # failures 很小时根在 p≈1e-12 量级，不完全 Beta 的互补
    # 分支会被 1-p 的舍入放大；直接累加二项下尾更稳。
    if failures <= 4096:
        for _ in range(200):
            middle = (low + high) / 2.0
            cdf = _binomial_lower_tail_probability(
                failures,
                trials,
                middle,
            )
            if cdf > alpha:
                low = middle
            else:
                high = middle
            tolerance = max(
                math.ulp(middle) * 4.0,
                middle * 1.0e-12,
            )
            if high - low <= tolerance:
                break
        return (low + high) / 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if middle <= 0.5:
            # I_p(successes, remaining) = 1-alpha 等价于
            # I_{1-p}(remaining, successes) = alpha。用 p 侧
            # 计算可避免 p 很小时 1-p 的舍入误差。
            cdf = _regularized_incomplete_beta(
                middle,
                successes,
                remaining,
            )
            if cdf is None or not math.isfinite(cdf):
                return None
            if cdf < confidence_value:
                low = middle
            else:
                high = middle
        else:
            cdf = _regularized_incomplete_beta(
                1.0 - middle,
                remaining,
                successes,
            )
            if cdf is None or not math.isfinite(cdf):
                return None
            if cdf > alpha:
                low = middle
            else:
                high = middle
        # 上界本身可能远小于 1e-12（如 trials=1e12、failures=1
        # 时约 5e-12）。固定绝对容差会在量级相当的位置提前
        # 停止，产生数个百分点误差；按估计值取相对容差。
        tolerance = max(
            math.ulp(middle) * 4.0,
            middle * 1.0e-12,
        )
        if high - low <= tolerance:
            break
    return (low + high) / 2.0



def _pilot_calibration_setting(
    power: Optional[Dict[str, object]],
) -> Tuple[str, float]:
    """读取预注册的试点校准策略，损坏值退回 point/0.95。

    结构错误由 ``preregistration_schema_error`` 在入口拒绝；
    这里保底是为了让旧/手工 power 记录不会让设计计划崩溃。
    """
    if not isinstance(power, dict):
        return "point", _PILOT_DEFAULT_CONFIDENCE
    policy = power.get("pilot_calibration", "point")
    if policy not in _PILOT_CALIBRATION_POLICIES:
        policy = "point"
    confidence = _finite_float(
        power.get("pilot_confidence", _PILOT_DEFAULT_CONFIDENCE)
    )
    if (
        confidence is None
        or not 0.5 < confidence < 1.0
    ):
        confidence = _PILOT_DEFAULT_CONFIDENCE
    return str(policy), confidence



_PILOT_TREATMENT_FIELDS = (
    "mode",
    "balance_coef",
    "k1",
    "k2",
    "top_k",
    "dense_fnn_hidden",
    "n_experts",
    "expert_hidden",
)



def _pilot_treatment_key(row: Dict[str, object]) -> tuple:
    """试点运行的处理单元身份：结构/处理轴，不含 seed/路径。"""
    return tuple(
        _stable_factor_key(row.get(field))
        for field in _PILOT_TREATMENT_FIELDS
    )



def _pilot_cell_label(cell: Dict[str, object]) -> str:
    """处理单元的可读标签（mode + 均衡系数）。"""
    mode = cell.get("mode")
    label = "?" if mode is None else str(mode)
    balance = _finite_float(cell.get("balance_coef"))
    if balance is not None:
        label += f"@bal={balance:g}"
    return label



def _pilot_declared_check(
    declared: Optional[float],
    observed: Optional[float],
) -> str:
    """声明值 vs 试点观测值的判定（相对容差避免浮点边界误判）。"""
    if declared is None:
        return "not_declared"
    declared_value = _finite_float(declared)
    observed_value = _finite_float(observed)
    if declared_value is None or observed_value is None:
        return "not_evaluable"
    tolerance = 1e-12 + abs(declared_value) * 1e-9
    if observed_value <= declared_value + tolerance:
        return "declared_at_least_observed"
    return "declared_below_observed"



_PILOT_COMPARABILITY_SCHEMA_VERSION = 1

_PILOT_PROVENANCE_KEYS = frozenset(_REPRODUCIBILITY_HARD_KEYS)

_PILOT_SHA256_KEYS = frozenset({
    "script_sha256",
    "data_sha256",
    "split_sha256",
    "effective_train_sha256",
    "environment_id",
})

_PILOT_CLEANING_KEYS = (
    "deduplicate_train_groups",
    "drop_train_shared_chunks",
    "drop_train_near_duplicate_groups",
    "near_duplicate_threshold",
    "near_duplicate_shingle_bytes",
    "clean_train_against_test",
    "drop_train_corrupt_groups",
    "drop_train_corrupt_bytes",
    "drop_train_degenerate_groups",
    "drop_train_degenerate_runs",
    "allow_eval_content_for_cleaning",
    "allow_entity_overlap",
    "allow_data_leakage",
    "allow_corrupt_test_labels",
    "allow_eval_duplicates",
)

_PILOT_COMPARABILITY_GROUPS: Tuple[
    Tuple[str, Tuple[str, ...], str], ...
] = (
    ("code", ("script_sha256",), "blocking"),
    (
        "data",
        (
            "data_sha256",
            "split_sha256",
            "effective_train_sha256",
            "val_fraction",
            "test_fraction",
            "split_policy",
            "group_separator",
            "split_isolation",
            *_PILOT_CLEANING_KEYS,
        ),
        "blocking",
    ),
    (
        "structure",
        (
            "init_scheme",
            "seq_len",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "n_layers",
            "residual_scale",
            "residual_scaling",
        ),
        "blocking",
    ),
    (
        "training",
        (
            "optimizer",
            "optimizer_betas",
            "optimizer_eps",
            "lr",
            "lr_schedule",
            "wsd_stable_steps",
            "wsd_final_lr_ratio",
            "warmup_steps",
            "total_steps",
            "weight_decay",
            "grad_clip",
            "batch_size",
            "document_masks",
            "data_augmentation",
            "early_stop_patience",
            "early_stop_min_delta",
            "grad_checkpoint",
            "compile",
        ),
        "blocking",
    ),
    (
        "evaluation",
        (
            "eval_interval",
            "eval_batches",
            "eval_bootstrap",
            "eval_bootstrap_block",
            "selection_metric",
        ),
        "blocking",
    ),
    (
        "environment",
        (
            "device_type",
            "amp_dtype",
            "allow_tf32",
            "environment_id",
            "python_version",
            "torch_version",
            "cuda_version",
            "cudnn_version",
            "platform",
            "device_name",
            "device_uuid",
            "deterministic_algorithms",
            "deterministic_algorithms_warn_only",
            "torch_num_threads",
            "torch_num_interop_threads",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cublas_workspace_config",
            "allow_tf32_matmul",
            "allow_tf32_cudnn",
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ),
        "advisory",
    ),
)



def _pilot_expected_comparability_protocol(
    args: argparse.Namespace,
    prereg: Dict[str, object],
    metric: str,
    target_step: Optional[float],
) -> Dict[str, object]:
    """构造正式运行的可比协议快照，供试点证据逐项核对。

    只放入能从预注册或本次命令行唯一确定的值；无法确定的值
    保持缺失，由比较器标为 ``not_recorded``，而不是默认为
    「一致」。显式 null（如未启用群组分隔符）在预注册 data
    块中确实出现时仍作为已知期望值参与比较。
    """
    prereg = _as_dict(prereg)
    data_block = _as_dict(prereg.get("data"))
    cleaning = _as_dict(data_block.get("cleaning"))
    expected: Dict[str, object] = {}
    planned_seeds: List[int] = []
    seed_plan = getattr(args, "seeds", None)
    if not seed_plan:
        seed_plan = prereg.get("seeds")
    for seed in _pilot_seed_values(seed_plan):
        number = _pilot_seed_int(seed)
        if number is not None:
            planned_seeds.append(number)

    script_sha = prereg.get("frozen_script_sha256")
    if not _preregistration_sha256_text(script_sha):
        script_sha = _script_sha256()
    if isinstance(script_sha, str):
        expected["script_sha256"] = script_sha.lower()

    data_values = {
        "data_sha256": (
            getattr(args, "data_sha256", None)
            or data_block.get("sha256")
            or prereg.get("data_sha256")
        ),
        "split_sha256": (
            dict(_as_dict(
                getattr(args, "data_split_sha256", None)
            ))
            or None
        ),
        "effective_train_sha256": getattr(
            args, "data_effective_train_sha256", None
        ),
        "split_isolation": getattr(
            args, "data_split_isolation", None
        ),
    }
    for key, value in data_values.items():
        if value is None:
            continue
        # SHA-256 十六进制大小写不影响内容同一性；预注册
        # 校验按不区分大小写比较，试点协议核对也必须一致，
        # 否则合法的大写指纹会被误报为不可比。
        if key in ("data_sha256", "effective_train_sha256"):
            if isinstance(value, str):
                value = value.lower()
        elif key == "split_sha256" and isinstance(value, dict):
            value = {
                name: (
                    digest.lower()
                    if isinstance(digest, str)
                    else digest
                )
                for name, digest in value.items()
            }
        expected[key] = value
    # 显式 null 也是已知的正式协议值（例如未启用群组分隔符）；
    # 只有 args 中根本没有该字段时才保持缺失。不能用「值是否为
    # None」区分这两者，否则正式计划不使用分隔符时，试点用了
    # 分隔符也不会被判为不可比。
    missing = object()
    for key in (
        "val_fraction",
        "test_fraction",
        "split_policy",
        "group_separator",
    ):
        if key in data_block:
            expected[key] = data_block.get(key)
        else:
            value = getattr(args, key, missing)
            if value is not missing:
                expected[key] = value
    cleaning_bool_keys = set(
        _PREREGISTRATION_CLEANING_BOOL_KEYS
    ) | {
        "allow_entity_overlap",
        "allow_data_leakage",
        "allow_corrupt_test_labels",
        "allow_eval_duplicates",
    }
    for key in _PILOT_CLEANING_KEYS:
        if key in cleaning:
            value = cleaning.get(key)
        else:
            value = getattr(args, key, missing)
            if value is missing:
                continue
        # 清洗开关在 comparison_protocol 中统一写成 bool；
        # 直接调用方传入 0/1/None 时也必须用同一口径。
        expected[key] = (
            bool(value)
            if key in cleaning_bool_keys
            else value
        )
    # 两个同义清洗开关在 comparison_protocol 中按 OR 归一化；
    # 直接构造 args 时可能只设置了其中一个，期望值必须使用同一
    # 口径，否则合法的试点会被误报为协议不一致。
    alias_keys = (
        "clean_train_against_test",
        "allow_eval_content_for_cleaning",
    )
    alias_present = [
        key
        for key in alias_keys
        if key in cleaning or hasattr(args, key)
    ]
    if alias_present:
        alias_value = bool(
            _clean_against_test_effective(args)
            or any(bool(cleaning.get(key)) for key in alias_keys)
        )
        for key in alias_keys:
            expected[key] = alias_value

    protocol_attrs = (
        "init_scheme",
        "lr",
        "wsd_stable_steps",
        "wsd_final_lr_ratio",
        "warmup_steps",
        "weight_decay",
        "grad_clip",
        "batch_size",
        "document_masks",
        "early_stop_patience",
        "early_stop_min_delta",
        "grad_checkpoint",
        "compile",
        "residual_scale",
        "residual_scaling",
        "seq_len",
        "d_model",
        "n_heads",
        "n_layers",
        "eval_interval",
        "eval_batches",
        "eval_bootstrap",
        "eval_bootstrap_block",
        "selection_metric",
        "amp_dtype",
        "allow_tf32",
    )
    bool_protocol_attrs = {
        "document_masks",
        "grad_checkpoint",
        "compile",
        "allow_tf32",
    }
    for key in protocol_attrs:
        value = getattr(args, key, missing)
        if value is not missing:
            # 这些开关在 comparison_protocol 中统一写成 bool；
            # 直接调用方传入 0/1/None 时也要用同一口径，否则
            # 合法运行会被误报为协议不一致。
            expected[key] = (
                bool(value)
                if key in bool_protocol_attrs
                else value
            )
    # comparison_protocol 把 --lr-schedule 的历史取值 cosine 归一化
    # 成描述性的 linear_warmup_cosine_to_10pct；试点核对必须用同一
    # 口径，否则默认 cosine 运行会被误判为与正式计划不可比。
    schedule = str(getattr(args, "lr_schedule", "cosine"))
    expected["lr_schedule"] = (
        "linear_warmup_cosine_to_10pct"
        if schedule == "cosine"
        else "wsd"
    )
    n_kv_heads = getattr(args, "n_kv_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(args, "n_heads", None)
    if n_kv_heads is not None:
        expected["n_kv_heads"] = n_kv_heads
    steps = getattr(args, "steps", None)
    if steps is None:
        steps = prereg.get("steps")
    if steps is not None:
        expected["total_steps"] = steps
    device = getattr(args, "device", missing)
    if device is not missing:
        expected["device_type"] = _device_type_of(device)
    if getattr(args, "lr", None) is not None:
        expected["optimizer"] = "AdamW"
        expected["optimizer_betas"] = [0.9, 0.999]
        expected["optimizer_eps"] = 1e-8
        expected["data_augmentation"] = "none"
    expected["primary_metric"] = metric
    expected["target_step"] = target_step
    expected["planned_seeds"] = planned_seeds
    return expected



def _pilot_row_protocol_value(
    row: Dict[str, object],
    key: str,
) -> Tuple[object, bool]:
    """读取试点运行的一个协议值及「是否确有记录」。

    旧运行的 comparison_protocol 可能缺键；回退补齐出的非空
    值仍可作为实际值，但显式 null 只有在原始记录中确实出现
    时才算已知，避免把「未记录」当成「没有启用」。
    """
    if key in _PILOT_PROVENANCE_KEYS:
        source = _as_dict(row.get("provenance"))
        recorded_keys = row.get("provenance_recorded_keys")
        value = source.get(key)
        if recorded_keys is None:
            return value, key in source
        recorded = {
            str(item) for item in recorded_keys
        } if isinstance(
            recorded_keys, (list, tuple, set, frozenset)
        ) else set()
        # provenance 没有「按 args 回退补齐」这条路径：值非空
        # 但原始 run_metadata 未记录该键时仍属未核对，不能和
        # 受控变量协议的回退口径混用。
        return value, key in recorded
    protocol = _as_dict(row.get("protocol"))
    recorded_keys = row.get("protocol_recorded_keys")
    value = protocol.get(key)
    if recorded_keys is None:
        return value, value is not None
    recorded = {
        str(item) for item in recorded_keys
    } if isinstance(
        recorded_keys, (list, tuple, set, frozenset)
    ) else set()
    return value, key in recorded or value is not None



def _pilot_row_dicts(value: object) -> List[Dict[str, object]]:
    """把试点行容器归一化为 dict 列表，损坏标量不崩溃。"""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return []
    return [item for item in items if isinstance(item, dict)]



def _pilot_comparability_run_labels(
    rows: Sequence[Dict[str, object]],
) -> List[str]:
    """给每个试点行生成唯一标签，避免同名运行在 values 中折叠。"""
    rows = _pilot_row_dicts(rows)
    counts: Dict[str, int] = {}
    for row in rows:
        name = str(row.get("run"))
        counts[name] = counts.get(name, 0) + 1
    seen: Dict[str, int] = {}
    used: set = set()
    labels: List[str] = []
    for row in rows:
        base = str(row.get("run"))
        if counts.get(base) == 1 and base not in used:
            label = base
        else:
            ordinal = seen.get(base, 0) + 1
            label = (
                base
                if ordinal == 1 and base not in used
                else f"{base}#{ordinal}"
            )
            while label in used:
                ordinal += 1
                label = f"{base}#{ordinal}"
            seen[base] = ordinal
        used.add(label)
        labels.append(label)
    return labels



def _pilot_comparability_factor_key(
    key: str,
    value: object,
):
    """协议值比较键：SHA-256 指纹按大小写不敏感归一化。"""
    if key in _PILOT_SHA256_KEYS:
        if isinstance(value, str):
            value = value.lower()
        elif key == "split_sha256" and isinstance(value, dict):
            value = {
                name: (
                    digest.lower()
                    if isinstance(digest, str)
                    else digest
                )
                for name, digest in value.items()
            }
    return _stable_factor_key(value)



def _pilot_comparability_item(
    key: str,
    group: str,
    severity: str,
    expected: Dict[str, object],
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    expected = _as_dict(expected)
    rows = _pilot_row_dicts(rows)
    values: Dict[str, object] = {}
    missing_runs: List[str] = []
    for row, name in zip(
        rows, _pilot_comparability_run_labels(rows)
    ):
        value, recorded = _pilot_row_protocol_value(row, key)
        values[name] = value
        if not recorded:
            missing_runs.append(name)
    present = [
        (name, value)
        for name, value in values.items()
        if name not in set(missing_runs)
    ]
    distinct = {
        repr(_pilot_comparability_factor_key(key, value))
        for _, value in present
    }
    if not present:
        status = "not_recorded"
    elif len(distinct) > 1:
        status = "mismatch"
    elif key not in expected:
        # 期望值缺失表示正式计划无法唯一确定该协议项；即使
        # 所有试点值彼此一致，也不能据此声称已与正式计划核对。
        status = "not_recorded"
    else:
        expected_key = repr(
            _pilot_comparability_factor_key(
                key, expected.get(key)
            )
        )
        if expected_key not in distinct:
            status = "mismatch"
        elif missing_runs:
            status = "partial"
        else:
            status = "match"
    return {
        "key": key,
        "group": group,
        "severity": severity,
        "expected": expected.get(key) if key in expected else None,
        "expected_recorded": key in expected,
        "values": values,
        "missing_runs": missing_runs,
        "status": status,
    }
