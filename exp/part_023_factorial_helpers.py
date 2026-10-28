from __future__ import annotations



def _stable_factor_key(value):
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



def _factor_sort_rank(value):
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



def cross_family_protocol_check(
    rows: List[Dict[str, object]],
    factors: List[str],
) -> Dict[str, object]:
    """
    核对神经网络运行与统计基线（ngram）是否使用同一份数据与
    同一组评估窗口。

    COMPARISON_PROTOCOL_KEYS 里的优化器/训练轮数对 ngram 不
    适用，因此主受控变量检查按 family 分开；但「基线是否强于
    ngram」这一判断仍然要求 data_sha256、切分比例、评估窗口
    数、批量大小、数据增强等口径一致。这里只比较这些跨 family
    共用项，不一致时返回 violation，供对比报告拒绝归因。
    """
    neural: List[Tuple[str, object, Optional[set]]] = []
    statistical: List[Tuple[str, object, Optional[set]]] = []

    for row in rows:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            protocol = {}
        family = protocol.get("comparison_family") or "neural"
        recorded = row.get("protocol_recorded_keys")
        recorded_keys = (
            {str(key) for key in recorded}
            if isinstance(recorded, (list, tuple, set, frozenset))
            else None
        )
        target = statistical if family == "statistical" else neural
        target.append(
            (str(row.get("run")), protocol, recorded_keys)
        )

    checked: List[str] = []
    unchecked: List[str] = []
    violations: List[Dict[str, object]] = []

    if not neural or not statistical:
        return {
            "checked_keys": checked,
            "unchecked_keys": unchecked,
            "violations": violations,
            "ok": True,
        }

    for key in _CROSS_FAMILY_PROTOCOL_KEYS:
        if key in factors:
            continue

        def _values(entries):
            return [
                (
                    run,
                    protocol.get(key),
                    recorded is not None and key in recorded,
                )
                for run, protocol, recorded in entries
            ]

        neural_values = _values(neural)
        statistical_values = _values(statistical)
        all_values = neural_values + statistical_values
        present = [
            (run, value)
            for run, value, explicit_none in all_values
            if value is not None or explicit_none
        ]
        missing_runs = [
            run
            for run, value, explicit_none in all_values
            if value is None and not explicit_none
        ]

        if not present:
            unchecked.append(key)
            continue
        if not any(
            value is not None or explicit_none
            for _, value, explicit_none in neural_values
        ) or not any(
            value is not None or explicit_none
            for _, value, explicit_none in statistical_values
        ):
            unchecked.append(key)
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
                    run: value
                    for run, value, _ in all_values
                },
                "missing_runs": missing_runs,
                "note": (
                    "神经运行与 ngram 统计基线使用了不同的"
                    "数据/评估口径，两者不能直接比较"
                ),
            })

    return {
        "checked_keys": checked,
        "unchecked_keys": unchecked,
        "violations": violations,
        "ok": not violations,
    }



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
    # 所有神经运行都缺记录的协议项：无法判定一致性。必须单独
    # 列为「未核对」，不能让「没记录」静默混进 OK 结论。
    unchecked: List[str] = []

    # 原 config.json 的 comparison_protocol 显式写出的键。显式
    # null（如 cosine 调度下的 wsd_stable_steps、未启用分组时的
    # group_separator）本身是一个可比较的协议值；只有回退补齐
    # 后仍为 None 的键才是真正「未记录」。旧运行没有该清单时
    # 保持 None，沿用「None 视为缺记录」的保守口径。
    recorded_by_row: Dict[int, Optional[set]] = {}
    for row in neural:
        recorded = row.get("protocol_recorded_keys")
        if isinstance(recorded, (list, tuple, set, frozenset)):
            recorded_by_row[id(row)] = {
                str(key) for key in recorded
            }
        else:
            recorded_by_row[id(row)] = None

    for key in COMPARISON_PROTOCOL_KEYS:
        if key in factor_set:
            continue

        values = []
        for row in neural:
            name = str(row.get("run"))
            protocol = row.get("protocol")
            value = (
                protocol.get(key)
                if isinstance(protocol, dict)
                else None
            )
            recorded = recorded_by_row[id(row)]
            explicit_none = (
                value is None
                and recorded is not None
                and str(key) in recorded
            )
            values.append((name, value, explicit_none))

        present = [
            (name, value)
            for name, value, explicit_none in values
            if value is not None or explicit_none
        ]
        missing_runs = [
            name
            for name, value, explicit_none in values
            if value is None and not explicit_none
        ]

        if not present:
            # 所有运行都缺记录：无法判定，列为未核对。
            if neural:
                unchecked.append(key)
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
                    name: value
                    for name, value, _ in values
                },
                "missing_runs": missing_runs,
            })

    cross_family = cross_family_protocol_check(rows, factors)

    return {
        "factors": list(factors),
        "checked_keys": checked,
        "unchecked_keys": unchecked,
        "violations": violations,
        "n_neural_runs": len(neural),
        "statistical_runs": statistical,
        "cross_family_checked_keys": cross_family[
            "checked_keys"
        ],
        "cross_family_unchecked_keys": cross_family[
            "unchecked_keys"
        ],
        "cross_family_violations": cross_family[
            "violations"
        ],
        "ok": not violations
        and not cross_family["violations"],
    }



def check_run_provenance(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    可复现性来源核对：同一次对比的运行必须来自同一份实验
    代码与同一数值环境（Python/torch/CUDA/cuDNN/设备与
    确定性开关）。

    与受控变量检查一样，只在有记录可核对的运行之间判定；
    旧运行缺记录时列为「未核对」，不虚报一致。git commit /
    dirty 状态单独作为信息项：同一脚本指纹下的 commit 差异
    可能只是文档/CI 改动，不直接判违规。
    """
    provenance_values: List[
        Tuple[str, Dict[str, object], set]
    ] = []
    for row in rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        recorded = row.get("provenance_recorded_keys")
        if isinstance(recorded, (list, tuple, set, frozenset)):
            recorded_keys = {
                str(key) for key in recorded if isinstance(key, str)
            }
        else:
            recorded_keys = set(provenance)
        provenance_values.append(
            (str(row.get("run")), provenance, recorded_keys)
        )

    checked: List[str] = []
    unchecked: List[str] = []
    mismatches: List[Dict[str, object]] = []

    for key in _REPRODUCIBILITY_HARD_KEYS:
        values = [
            (
                name,
                provenance.get(key),
                key in recorded_keys,
            )
            for name, provenance, recorded_keys in provenance_values
        ]
        present = [
            (name, value)
            for name, value, recorded in values
            if recorded
        ]
        missing_runs = [
            name for name, _, recorded in values if not recorded
        ]

        if not present:
            if rows:
                unchecked.append(key)
            continue

        checked.append(key)

        distinct = {
            repr(_stable_factor_key(value))
            for _, value in present
        }

        if len(distinct) > 1 or missing_runs:
            mismatches.append({
                "key": key,
                "values": {
                    name: value
                    for name, value, _ in values
                },
                "missing_runs": missing_runs,
            })

    hashseed_values = [
        (
            name,
            provenance.get("pythonhashseed_at_start"),
            "pythonhashseed_at_start" in recorded_keys,
        )
        for name, provenance, recorded_keys in provenance_values
    ]
    hashseed_unset_runs = [
        name
        for name, value, recorded in hashseed_values
        if recorded and value is None
    ]
    hashseed_unrecorded_runs = [
        name for name, _, recorded in hashseed_values if not recorded
    ]

    git_commit_values = [
        (
            name,
            provenance.get("git_commit"),
            "git_commit" in recorded_keys,
        )
        for name, provenance, recorded_keys in provenance_values
    ]
    git_commits = {
        repr(_stable_factor_key(value))
        for _, value, recorded in git_commit_values
        if recorded and value is not None
    }
    git_commit_missing_runs = [
        name
        for name, value, recorded in git_commit_values
        if not recorded or value is None
    ]

    script_hash_values = [
        (
            name,
            provenance.get("script_sha256"),
            "script_sha256" in recorded_keys,
        )
        for name, provenance, recorded_keys in provenance_values
    ]
    script_hashes = {
        repr(_stable_factor_key(value))
        for _, value, recorded in script_hash_values
        if recorded and value is not None
    }
    script_hash_missing_runs = [
        name
        for name, value, recorded in script_hash_values
        if not recorded or value is None
    ]

    return {
        "keys": list(_REPRODUCIBILITY_HARD_KEYS),
        "checked_keys": checked,
        "unchecked_keys": unchecked,
        "mismatches": mismatches,
        "hashseed_unset_runs": hashseed_unset_runs,
        "hashseed_unrecorded_runs": hashseed_unrecorded_runs,
        "commits_differ": (
            len(git_commits) > 1
            and not git_commit_missing_runs
            and len(script_hashes) == 1
            and not script_hash_missing_runs
        ),
        "ok": not mismatches,
    }



def _finite_values(values: List[float]) -> Optional[List[float]]:
    """把数值列表统一转成有限 float；任一值不可用则返回 None。

    报告/分析函数会读取手工修复或旧版本写下的 JSON，其中可能
    含超过 float 范围的整数。集中在这里转换，避免 ``sum`` 的
    真除法或 ``float()`` 在单个坏值上让整份报告崩溃。
    """
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
    direct = sum(numeric) / len(numeric)
    if math.isfinite(direct):
        return direct
    scale = max(abs(value) for value in numeric)
    if scale == 0.0:
        return 0.0
    total = math.fsum(value / scale for value in numeric)
    result = (total / len(numeric)) * scale
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
        mean = sum(numeric) / len(numeric)
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
    mean = math.fsum(normalized) / len(normalized)
    variance = math.fsum(
        (value - mean) ** 2 for value in normalized
    ) / (len(normalized) - 1)
    if not math.isfinite(variance) or variance < 0.0:
        return None
    result = math.sqrt(variance) * scale
    return result if math.isfinite(result) else None



def _sample_variance(values: List[float]) -> Optional[float]:
    """样本方差（ddof=1）；溢出/不可用时返回 None。"""
    std = _sample_std(values)
    if std is None:
        return None
    variance = std * std
    return variance if math.isfinite(variance) else None



def _betacf(a: float, b: float, x: float) -> float:
    """
    正则化不完全 Beta 函数的连分式（Lentz 算法），
    用于无 SciPy 依赖地计算 t 分布 p 值。
    """
    # 连分式的收敛步数随 sqrt(a+b) 增长：a=b=1e6 时需要约
    # 460 步，200 步会在 x≈1/2 处把 I_x 算成 0.4997，并让
    # 二项分布上置信界偏移数万分之一。正常量级仍在前几十步
    # 收敛，提高上限不会拖慢常见路径。
    max_iterations = 20000
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



_LOG_BETA_ASYMPTOTIC_MIN = 1.0e4



def _log_gamma_ratio_half(z: float) -> float:
    """
    计算 log(Γ(z+1/2)/Γ(z))，避免大 z 下两个 lgamma 相减。
    """
    inverse = 1.0 / z
    series = inverse * (
        -1.0 / 8.0
        + inverse * (
            1.0 / 128.0
            + inverse * (
                5.0 / 1024.0
                - inverse * (21.0 / 32768.0)
            )
        )
    )
    return 0.5 * math.log(z) + math.log1p(series)



def _log_beta_front(a: float, b: float) -> float:
    """
    返回 log(1/B(a, b))，即不完全 Beta 前因子中的常数项。

    大参数下直接做 lgamma(a+b)-lgamma(a)-lgamma(b) 会发生
    灾难性相消：例如 a=1.5e8、b=0.5 时两项都约为 2.5e9，
    相减后只剩约 1e-7 的相对精度，足以让 t 分布尾概率产生
    1e-6 量级的误差。这里利用
        Γ(z+1/2)/Γ(z) ~ z^(1/2) (1 - 1/(8z) + ...)
    精确计算 b=1/2 的情形。
    """
    if a == 0.5 and b >= _LOG_BETA_ASYMPTOTIC_MIN:
        return (
            _log_gamma_ratio_half(b)
            - math.lgamma(0.5)
        )
    if b == 0.5 and a >= _LOG_BETA_ASYMPTOTIC_MIN:
        return (
            _log_gamma_ratio_half(a)
            - math.lgamma(0.5)
        )
    return (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
    )



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

    log_beta = _log_beta_front(a, b)
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
    if math.isnan(df) or df <= 0.0:
        # 自由度必须是正有限数，或 +inf（正态极限）。NaN/负值
        # 没有定义；必须先校验 df，否则 t=inf 会被误报为 p=0。
        return None
    if math.isnan(t):
        # NaN 的 p 值没有定义：返回 0.0 会被当成“极显著”，
        # 必须显式返回 None 让调用方走「无检验结果」分支。
        return None
    if math.isinf(df):
        # df→+inf 时 t 分布退化为标准正态；直接代公式会得到
        # inf/inf=NaN，必须显式走正态尾概率。
        return math.erfc(abs(t) / math.sqrt(2.0))
    if not math.isfinite(t):
        return 0.0

    # 大自由度下不完全 Beta 的 lgamma 差分仍足够精确，直到
    # 约 1e9 才出现 1e-6 量级的舍入误差；而一阶正态尾修正
    # 在 df=1e6、t=30 时误差近 2%。取两者误差交叉点附近
    # 2.5e8 为界：其下走不完全 Beta，其上走正态极限。
    if df >= 2.5e8:
        magnitude = abs(t)
        if magnitude > 40.0:
            return 0.0
        normal_tail = math.erfc(magnitude / math.sqrt(2.0))
        density = (
            math.exp(-0.5 * magnitude * magnitude)
            / math.sqrt(2.0 * math.pi)
        )
        correction = (
            (magnitude * magnitude * magnitude + magnitude)
            * density
            / (2.0 * df)
        )
        return min(1.0, normal_tail + correction)

    if not math.isfinite(t * t):
        # t 的平方溢出时 x=df/(df+t²) 会被舍入成 0，直接走
        # 不完全 Beta 会返回 0，把仍然可表示的 p 值（如
        # df=1、t=1e200 时约 6e-201）误报成 0。这里在
        # x 的对数域使用小 x 展开：
        #   I_x(a, 1/2) ≈ x^a (1-x)^(1/2) / (a·B(a, 1/2))
        # 该分支的 x < 1e-300，级数修正可忽略；结果按
        # float 可表示范围下溢为 0 或截断到 1。
        a = df / 2.0
        if a <= 0.0:
            return 0.0
        log_x = math.log(df) - 2.0 * math.log(abs(t))
        log_beta = (
            math.lgamma(a + 0.5)
            - math.lgamma(a)
            - math.lgamma(0.5)
        )
        log_p = log_beta + a * log_x - math.log(a)
        if log_p >= 0.0:
            return 1.0
        if log_p < -745.0:
            return 0.0
        return math.exp(log_p)

    x = df / (df + t * t)
    # 上尾等于 I_x(df/2, 1/2)，也可用对称性写成
    # 1 - I_complement_x(1/2, df/2)。哪种口径更稳取决于 x：
    # x<=1/2 时直接分支只用到 x，精度不受影响；x>1/2 时
    # complement_x<1/2，1-complement_x 也能精确还原 x。
    #
    # 反过来在 x 很小时（t 很大）先算 complement_x 再取
    # 1-complement_x 会踩到表示误差：例如 df=1/2、
    # t=4.24e7 时 x≈2.78e-16，而 complement_x 舍入后
    # 1-complement_x≈3.33e-16（相对偏大 20%），上尾因此
    # 被高估约 5%。这里按 x 的大小选择稳定的那一侧，而不是
    # 等到上尾低于固定阈值才回退。
    if x <= 0.5:
        tail = _regularized_incomplete_beta(x, df / 2.0, 0.5)
        return None if tail is None else max(0.0, min(1.0, tail))

    complement_x = t * t / (df + t * t)
    complement = _regularized_incomplete_beta(
        complement_x,
        0.5,
        df / 2.0,
    )
    if complement is not None:
        upper_tail = 1.0 - complement
        if upper_tail >= 1.0e-6:
            return max(0.0, min(1.0, upper_tail))
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

    values = _finite_values(differences)
    if values is None:
        return {"test": "invalid_values"}
    # 用带缩放的均值：直接 sum 对合法的 1e308 会溢出成 inf，
    # 让 effect 字段（以及后续落盘）丢失真实点估计。正常量级
    # 仍走完全相同的直接求和路径。
    mean = _mean(values)
    std = _sample_std(values)
    if mean is None or std is None:
        return {"test": "invalid_values"}

    if std == 0.0:
        if mean == 0.0:
            return {
                "test": "paired_t",
                "t": 0.0,
                "df": float(n - 1),
                "p_value": 1.0,
                "note": "zero_variance",
                "effect": 0.0,
                "se": 0.0,
            }
        return {
            "test": "paired_t",
            "t": None,
            "df": float(n - 1),
            "p_value": 0.0,
            "note": "noise_free_separation",
            "effect": mean,
            "se": 0.0,
        }

    # 先算 mean/std 再乘 √n：极小尺度下 std/√n 可能下溢为 0，
    # 直接做 mean / se 会抛 ZeroDivisionError；该写法对公共尺度
    # 不变，也不改变正常量级的逐位结果。
    try:
        t = (mean / std) * math.sqrt(n)
    except (OverflowError, ZeroDivisionError):
        t = None
    se = std / math.sqrt(n)
    if t is None or not math.isfinite(t):
        return {
            "test": "paired_t",
            "t": None,
            "df": float(n - 1),
            "p_value": 0.0,
            "note": "extreme_separation",
            "effect": mean,
            "se": se,
        }
    return {
        "test": "paired_t",
        "t": t,
        "df": float(n - 1),
        "p_value": _two_sided_t_p_value(abs(t), float(n - 1)),
        "effect": mean,
        "se": se,
    }



# 配对符号翻转检验：n 不超过该值时穷举全部 2^n 种符号
# 组合（精确 p 值），否则用固定种子的 Monte Carlo 近似。
_RANDOMIZATION_EXACT_MAX_N = 12

_RANDOMIZATION_MONTE_CARLO_REPLICATES = 20000



def _randomization_seed(differences: List[float]) -> int:
    """由配对差值生成稳定的随机化种子（不触碰全局 RNG）。"""
    material = repr(
        [_finite_float(value) for value in differences]
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(),
        "big",
    )



def _paired_sign_flip_test(
    differences: List[float],
) -> Dict[str, object]:
    """
    配对符号翻转（randomization/sign-flip）检验。

    零假设：配对差值的分布关于 0 对称。检验统计量为差值的
    均值；穷举/抽样全部符号组合，得到双侧精确 p 值
    P(|mean(s_i·d_i)| >= |mean(d_i)|)。与配对 t 检验相比，
    它不要求差值服从正态分布，适合种子数很少（3~5）的
    机器学习实验重复。

    n <= 12 时穷举 2^n 种组合（n=12 为 4096 次）；
    更大样本用 20000 次固定种子的 Monte Carlo，并采用
    (count+1)/(R+1) 的保守估计避免零 p 值。符号翻转以
    「配对」为前提：非配对比较不适用。
    """
    n = len(differences)
    if n < _MIN_SEEDS_FOR_STATS:
        return {"test": "insufficient_seeds"}

    values = _finite_values(differences)
    if values is None:
        return {"test": "invalid_values"}

    # 符号翻转统计量对公共尺度不变：大数值先按最大绝对值缩放到
    # [-1, 1]，避免合法的大浮点数在求和时溢出。正常/极小量级
    # 保持原值（不改变历史点估计的末位），但容差必须按观测均值
    # 取相对值，否则 1e-13 量级的真实差值会被固定 1e-12 误判
    # 成退化。随机化种子仍用原始值，保证结果可复现。
    raw_values = values
    max_abs = max(abs(value) for value in values)
    scale = max_abs if max_abs > 1.0e150 else 1.0
    if scale != 1.0:
        values = [value / scale for value in values]
    mean_scaled = sum(values) / n
    mean = mean_scaled * scale
    observed = abs(mean_scaled)
    tolerance = 1e-12 * observed

    if observed <= tolerance:
        return {
            "test": "paired_sign_flip",
            "p_value": 1.0,
            "method": "degenerate",
            "n_permutations": 1,
            "effect": mean,
        }

    if n <= _RANDOMIZATION_EXACT_MAX_N:
        count = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            statistic = abs(
                sum(
                    sign * value
                    for sign, value in zip(signs, values)
                )
            ) / n
            if statistic >= observed - tolerance:
                count += 1
        permutations = 2 ** n
        p_value = count / permutations
        method = "exact"
    else:
        generator = torch.Generator().manual_seed(
            _randomization_seed(raw_values)
        )
        replicates = _RANDOMIZATION_MONTE_CARLO_REPLICATES
        signs = torch.randint(
            low=0,
            high=2,
            size=(replicates, n),
            generator=generator,
            dtype=torch.int8,
        ).to(torch.float64) * 2.0 - 1.0
        tensor = torch.tensor(values, dtype=torch.float64)
        statistics = (
            signs * tensor
        ).sum(dim=1).abs() / n
        count = int(
            (statistics >= observed - tolerance).sum().item()
        )
        permutations = replicates
        p_value = (count + 1) / (replicates + 1)
        method = "monte_carlo"

    # 最小可达双侧 p 值：精确枚举时两个极端符号组合（全正/
    # 全负）对应 |mean| 最大，n 很小时 2/2^n 可能本身就大于
    # 0.05，此时设计在数学上不可能得到显著结果，必须提示，
    # 不能把「不显著」解读为「无效应」。Monte Carlo 分支采用
    # (count+1)/(R+1)，count 可以为 0（极端组合未必被抽中），
    # 因此最小可达值是 1/(R+1)，不是精确分支的 2/R。
    if method == "exact":
        min_attainable_p = (
            2.0 / permutations if permutations > 0 else None
        )
        # 所需种子数是全局最小值，与当前 n 无关：n 已超过
        # 6 时仍应报告 6，而不是把当前样本量误报成「所需量」。
        required_seeds = _sign_flip_required_seeds(
            _SIGNIFICANCE_LEVEL
        )
    else:
        min_attainable_p = (
            1.0 / (permutations + 1)
            if permutations > 0
            else None
        )
        required_seeds = None

    return {
        "test": "paired_sign_flip",
        "p_value": p_value,
        "method": method,
        "n_permutations": permutations,
        "min_attainable_p": min_attainable_p,
        "can_reach_significance": (
            min_attainable_p is not None
            and min_attainable_p <= _SIGNIFICANCE_LEVEL
        ),
        "required_seeds_for_alpha": required_seeds,
        "effect": mean,
    }



# ============================================================
# 逐窗口配对比较（固定评估窗口）
# ============================================================

# 逐窗口比较的评估协议指纹：这些字段必须逐项相同，两个运行
# 才是在**同一批固定评估窗口**上比较。任何一项不同（例如换了
# 数据、换了窗口数/批大小、换了 seq_len）都意味着窗口集合不是
# 同一批，逐窗口配对没有意义，必须拒绝而不是硬算。
# split_sha256 不能省：整体 data_sha256 相同只说明源文件相同，
# val/test 划分比例、群组吸附或切分实现变化仍会改变 val 的
# 实际字节段与固定窗口起点。
_EVAL_WINDOW_FINGERPRINT_KEYS = (
    "split",
    "data_sha256",
    "split_sha256",
    "seq_len",
    "batch_size",
    "batches",
)



def _eval_window_fingerprint(
    payload: Dict[str, object],
) -> Tuple[object, ...]:
    return tuple(
        (
            key,
            json.dumps(payload.get(key), sort_keys=True),
        )
        for key in _EVAL_WINDOW_FINGERPRINT_KEYS
    )



def _window_per_token_loss(
    window: Dict[str, object],
) -> Optional[float]:
    if not isinstance(window, dict):
        return None
    tokens = _finite_float(window.get("tokens"))
    loss_sum = _finite_float(window.get("loss_sum"))
    if tokens is None or loss_sum is None or tokens <= 0.0:
        return None
    value = loss_sum / tokens
    return value if math.isfinite(value) else None



def _window_accuracy(
    window: Dict[str, object],
) -> Optional[float]:
    if not isinstance(window, dict):
        return None
    tokens = _finite_float(window.get("tokens"))
    correct = _finite_float(window.get("correct"))
    if tokens is None or correct is None or tokens <= 0.0:
        return None
    if correct < 0.0 or correct > tokens:
        return None
    value = correct / tokens
    return value if math.isfinite(value) else None



def _paired_window_bootstrap_ci(
    differences: List[float],
    seed: int,
) -> Dict[str, object]:
    """
    配对差值均值的百分位 bootstrap 区间（固定种子，可复现）。

    窗口级区间的独立性假设比种子级更弱（窗口可能重叠），
    因此该区间只描述「在这批固定评估窗口上」的差异，不解释为
    对「换一批评估窗口」的泛化区间。

    与评估级 bootstrap 同一口径：逐窗口差值沿评估序列可能正
    自相关（同类块/主题相邻），普通 percentile bootstrap 把
    窗口当 i.i.d. 会让区间偏窄。ρ̂1 达到阈值时按 AR(1) 有效
    窗口数做方差膨胀（只放宽不收窄），并把 ρ̂1、有效窗口数与
    膨胀因子随区间一起报告。
    """
    n = len(differences)
    if n < 2:
        return {"available": False, "reason": "insufficient_windows"}
    numeric = _finite_values(differences)
    if numeric is None:
        return {
            "available": False,
            "reason": "non_finite_differences",
        }
    generator = torch.Generator().manual_seed(int(seed))
    values = torch.tensor(numeric, dtype=torch.float64)
    # 合法的大差值（如 1e308）在重采样求均值时会把总和溢出成
    # inf，percentile 区间随之变成 NaN。均值对公共尺度等变，
    # 先按最大绝对值归一化，算出区间后再还原；正常量级保持
    # 原路径，避免改变历史区间。
    scale = float(values.abs().max().item())
    if scale > 1.0e150:
        values = values / scale
    else:
        scale = 1.0
    indices = torch.randint(
        low=0,
        high=n,
        size=(_EVAL_WINDOW_BOOTSTRAP_REPLICATES, n),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    low = float(torch.quantile(means, 0.025).item())
    high = float(torch.quantile(means, 0.975).item())
    rho1 = _lag1_autocorrelation(values)
    effective_windows, inflation = _ar1_effective_windows(
        n,
        rho1,
    )
    if inflation > 1.0:
        midpoint = (low + high) / 2.0
        half_width = (high - low) / 2.0 * inflation
        low = midpoint - half_width
        high = midpoint + half_width
    low *= scale
    high *= scale
    if not (math.isfinite(low) and math.isfinite(high)):
        # AR(1) 膨胀可能把归一化区间放大到超过 float 上限：
        # 乘回原始尺度后端点溢出为 inf。此时没有可落盘的有限
        # 区间，不能把 inf 写进报告（也不是合法的标准 JSON）；
        # 明确标记不可用，保留诊断量供追溯。
        return {
            "available": False,
            "reason": "unrepresentable_bounds",
            "replicates": _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
            "lag1_autocorrelation": rho1,
            "effective_windows": effective_windows,
            "ci_inflation": inflation,
            "note": (
                "AR(1) 膨胀后的区间端点超出 float 可表示范围；"
                "不写出 inf 端点，逐窗口点估计与检验结果仍保留"
            ),
        }
    return {
        "available": True,
        "low": low,
        "high": high,
        "replicates": _EVAL_WINDOW_BOOTSTRAP_REPLICATES,
        "method": (
            "percentile_bootstrap_ar1_adjusted"
            if inflation > 1.0
            else "percentile_bootstrap"
        ),
        "lag1_autocorrelation": rho1,
        "effective_windows": effective_windows,
        "ci_inflation": inflation,
    }
