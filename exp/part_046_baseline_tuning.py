from __future__ import annotations



def analyze_baseline_tuning(
    rows: List[Dict[str, object]],
    metric: str,
    mode: str,
    lrs: List[float],
    seeds: List[int],
) -> Dict[str, object]:
    """
    汇总基线调参网格：按学习率聚合各候选的 val 指标
    Mean ± Std，按方向选出 winner。

    选择规则（与 ``selection_rule`` 字段一致）：先取指标
    均值最优的候选，再把与最优均值的差距小于种子波动
    （两侧 std 的较大者）的候选全部视为「无法区分」，从这个
    并列集合中取更小的学习率作为 winner（更保守、更接近
    默认值）；并列集合的成员与保守选择都会写进报告。
    自测必须覆盖「均值不同但差距落在噪声内」的分支，否则
    规则只在均值恰好相等时才被验证。

    候选必须使用同一组随机种子：若候选的种子集合不同或
    同一候选内出现重复种子，Mean ± Std 的比较会混入种子
    差异/伪重复，winner 不能作为「调优过的基线」证据。

    同时核对调参协议本身：每个候选的全部受控变量
    （steps / batch / 数据指纹 / warmup / 调度 / weight decay /
    梯度裁剪 / 数据划分 / 评估窗口等）必须完全一致，否则
    「调参」会与「训练得更久或换了协议」混杂；候选缺指标时
    显式记录 issue。winner 落在网格边界（最小值或最大值）
    时提示最优可能落在网格之外，应扩展候选后重跑。
    """
    higher_is_better = metric_higher_is_better(metric)

    report: Dict[str, object] = {
        "comparison_family": "baseline_tuning",
        "mode": mode,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "grid": [float(lr) for lr in lrs],
        "seeds": [int(seed) for seed in seeds],
        "n_candidates": len(lrs) * len(seeds),
        "n_runs": len(rows),
        # 完整受控变量协议（除学习率这个调参因子以外）：记录
        # 下来供 --baseline-tuning-from 逐项核对，避免「在
        # 一种协议下调参、在另一种协议下训练最终基线」。
        "training_config": {
            key: None for key in _BASELINE_TUNING_PROTOCOL_KEYS
        },
        "protocol_keys": list(_BASELINE_TUNING_PROTOCOL_KEYS),
        "candidates": [],
        "winner": None,
        "winner_selection": None,
        "tied_lrs": [],
        "winner_at_grid_boundary": False,
        "best_mean_at_grid_boundary": False,
        "winner_unconverged_runs": [],
        # 种子计划随报告落盘：调参的统计功效（独立种子数、
        # 是否用了低功效豁免）是「基线已充分调参」的一部分，
        # 下游来源核对与基线合理性检查都必须能看到。
        "seed_plan": _build_seed_plan(
            seeds,
            False,
        ),
        # 候选的种子集合必须一致：不同候选跑不同种子时，
        # 均值差异可能是种子难度差异而不是学习率差异；
        # 同一候选内重复种子则是伪重复，会低估方差。
        "seed_alignment": {
            "candidate_seed_sets": {},
            "common_seeds": [],
            "aligned": None,
            "duplicate_seeds": {},
            "missing_seed_runs": [],
        },
        "selection_rule": (
            "在 val 上先取指标均值最优的候选；与最优均值的差距"
            "小于种子波动（两侧 std 的较大者）的候选视为并列"
            "（无法在统计上区分），从并列集合中取更小的学习率"
            "作为 winner（更保守）"
        ),
        "issues": [],
        "caveats": [
            "调参只看 val：test 集全程尘封，不参与候选排序",
            "各候选的全部受控变量（steps/batch/数据/调度/"
            "warmup/weight decay/梯度裁剪/数据划分等）完全"
            "一致，调参不会与「训练更久或换了协议」混杂",
            "调参后应固定 winner 配置训练最终基线，并用 "
            "--baseline-tuning-from 记录来源；该来源会逐项"
            "核对训练协议，协议不一致时拒绝；复杂模型必须"
            "战胜这条调过参的基线才有意义",
        ],
    }

    # 训练配置一致性：调参网格的受控变量。逐项核对完整协议；
    # 只统计有协议记录的运行，缺失记录不能被当成「训练配置
    # 不同」虚报（但仍会在来源核对阶段要求补齐）。
    protocol_values: Dict[str, set] = {
        key: set() for key in _BASELINE_TUNING_PROTOCOL_KEYS
    }
    protocol_value_counts = {
        key: 0 for key in _BASELINE_TUNING_PROTOCOL_KEYS
    }
    recorded_protocol = 0
    for row in rows:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            continue
        recorded_protocol += 1
        for key in _BASELINE_TUNING_PROTOCOL_KEYS:
            value = protocol.get(key)
            if value is None:
                continue
            protocol_value_counts[key] += 1
            protocol_values[key].add(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
            )

    inconsistent_keys = [
        key
        for key, values in protocol_values.items()
        if len(values) > 1
    ]
    if inconsistent_keys:
        report["issues"].append(
            "候选之间的训练协议不一致（"
            + ", ".join(inconsistent_keys)
            + "）：调参结果与「训练得更久 / 换了优化协议或"
            "数据划分」混杂，不可用于选择基线超参数"
        )
    partial_protocol_keys = []
    if recorded_protocol:
        # 记录每个键的（唯一）取值；缺记录或候选间不一致的
        # 键保持 None，由来源核对阶段显式拒绝。
        for key, values in protocol_values.items():
            if (
                len(values) == 1
                and protocol_value_counts[key] == recorded_protocol
            ):
                report["training_config"][key] = json.loads(
                    next(iter(values))
                )
            elif len(values) == 1:
                partial_protocol_keys.append(key)
    if partial_protocol_keys:
        report["issues"].append(
            "部分候选运行缺少训练协议字段（"
            + ", ".join(partial_protocol_keys)
            + "）：无法确认所有候选使用同一训练协议，"
            "winner 不能作为「调优过的基线」证据"
        )

    def _row_lr(row: Dict[str, object]) -> Optional[float]:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            return None
        return _finite_float(protocol.get("lr"))

    candidates: List[Dict[str, object]] = []
    seed_alignment = report["seed_alignment"]
    assert isinstance(seed_alignment, dict)
    candidate_seed_sets = seed_alignment[
        "candidate_seed_sets"
    ]
    duplicate_seeds_by_candidate = seed_alignment[
        "duplicate_seeds"
    ]
    missing_seed_runs = seed_alignment["missing_seed_runs"]
    assert isinstance(candidate_seed_sets, dict)
    assert isinstance(duplicate_seeds_by_candidate, dict)
    assert isinstance(missing_seed_runs, list)

    for lr in lrs:
        lr_rows = [
            row for row in rows if _row_lr(row) == _finite_float(lr)
        ]
        values: List[float] = []
        seed_metric: Dict[int, float] = {}
        for row in lr_rows:
            value = _finite_float(row.get(metric))
            if value is None:
                continue
            values.append(value)
            seed = row.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                seed_metric[int(seed)] = value
        mean = _mean(values)

        seed_values: List[int] = []
        for row in lr_rows:
            seed = row.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                seed_values.append(int(seed))
            else:
                missing_seed_runs.append(str(row.get("run")))

        label = format(lr, "g")
        unique_seeds = sorted(set(seed_values))
        candidate_seed_sets[label] = unique_seeds
        duplicate_seeds = sorted({
            seed
            for seed in unique_seeds
            if seed_values.count(seed) > 1
        })
        if duplicate_seeds:
            duplicate_seeds_by_candidate[label] = duplicate_seeds

        candidate = {
            "lr": float(lr),
            "n_runs": len(lr_rows),
            "n_metric_runs": len(values),
            "metric_mean": mean,
            "metric_std": _sample_std(values),
            "runs": [str(row.get("run")) for row in lr_rows],
            "seeds": unique_seeds,
            # 按种子对齐的指标值：调参并列判定必须利用配对
            # 差值（std 不是均值差的标准误，且忽略配对）。
            "seed_metric": seed_metric,
        }
        if len(values) < len(lr_rows):
            report["issues"].append(
                f"lr={format(lr, 'g')} 有 "
                f"{len(lr_rows) - len(values)} 个候选运行缺少 "
                f"{metric}，未计入均值"
            )
        candidates.append(candidate)

    report["candidates"] = candidates

    # 种子集合核对：只有所有候选都在同一组种子上评估，
    # 均值差才能归因到学习率；否则 winner 可能只是抽到了
    # 更容易/更难的种子。重复种子是伪重复，会低估方差。
    nonempty_sets = [
        tuple(seeds)
        for seeds in candidate_seed_sets.values()
        if seeds
    ]
    if duplicate_seeds_by_candidate:
        seed_alignment["aligned"] = False
        report["issues"].append(
            "同一学习率候选内出现重复随机种子（伪重复）："
            + "; ".join(
                f"lr={label} 种子 "
                + ", ".join(str(seed) for seed in seeds)
                for label, seeds in sorted(
                    duplicate_seeds_by_candidate.items()
                )
            )
            + "。重复运行不是独立样本，会低估种子间方差，"
            "winner 不能作为「调优过的基线」证据"
        )
    elif missing_seed_runs:
        seed_alignment["aligned"] = None
        report["issues"].append(
            "部分调参运行缺少随机种子记录，无法核对候选是否"
            "在同一组种子上评估（" + ", ".join(
                missing_seed_runs
            ) + "）：均值差可能混入种子差异，不能作为"
            "「调优过的基线」证据"
        )
    elif len(set(nonempty_sets)) > 1 or len(
        nonempty_sets
    ) != len(lrs):
        seed_alignment["aligned"] = False
        report["issues"].append(
            "各学习率候选使用的随机种子集合不一致："
            + "; ".join(
                f"lr={label} 种子=["
                + ", ".join(str(seed) for seed in seeds)
                + "]"
                for label, seeds in sorted(
                    candidate_seed_sets.items()
                )
            )
            + "。均值差可能是种子难度差异而不是学习率差异，"
            "winner 不能作为「调优过的基线」证据"
        )
    else:
        seed_alignment["aligned"] = True
        seed_alignment["common_seeds"] = (
            list(nonempty_sets[0]) if nonempty_sets else []
        )

    usable = [
        candidate
        for candidate in candidates
        if candidate["metric_mean"] is not None
        and int(candidate["n_runs"]) > 0
    ]
    if not usable:
        report["issues"].append(
            f"没有任何候选提供可用的 {metric}，无法选出"
            "基线学习率"
        )
        return report

    def _mean_order(candidate: Dict[str, object]) -> float:
        mean = float(candidate["metric_mean"])
        return -mean if higher_is_better else mean

    best = min(usable, key=_mean_order)
    best_mean = float(best["metric_mean"])
    best_std = float(best["metric_std"] or 0.0)

    # 并列/无法区分集合：与最优均值的差距小于种子波动
    # （两侧 std 的较大者）的候选都算无法在统计上区分，或
    # 均值恰好相等（噪声为 0 时的精确并列）。winner 必须从
    # 该并列集合中取更小的学习率——这正是 selection_rule
    # 声明的协议，不能只在均值精确相等时才做保守选择。
    tied: List[Dict[str, object]] = []
    indistinguishable: List[float] = []
    for candidate in usable:
        mean = float(candidate["metric_mean"])
        # gap = 候选比最优均值「差」多少（方向上非负；
        # 精确并列时为 0，浮点比较可能得到极小负值）。
        gap = (
            best_mean - mean
            if higher_is_better
            else mean - best_mean
        )
        # 并列判定优先用**同种子配对差值**的 t 检验：std 是
        # 种子间离散度，不是均值差的标准误，且忽略配对。
        # 配对不显著时只能说明当前种子数无法区分，这是预先
        # 声明的简化选型规则（不是等价性证明）；没有可比配对
        # 时才退回旧的 std 启发式，并在报告中标明规则。
        best_seed_metric = best.get("seed_metric") or {}
        cand_seed_metric = candidate.get("seed_metric") or {}
        common_train_seeds = sorted(
            set(best_seed_metric) & set(cand_seed_metric)
        )
        paired_rule = None
        is_tied = gap <= 0.0
        if (
            not is_tied
            and len(common_train_seeds) >= _MIN_SEEDS_FOR_STATS
        ):
            # 方向统一成「候选相对最优的改善量」，越大越好。
            diffs = [
                (
                    cand_seed_metric[seed]
                    - best_seed_metric[seed]
                    if higher_is_better
                    else best_seed_metric[seed]
                    - cand_seed_metric[seed]
                )
                for seed in common_train_seeds
            ]
            paired_test = _paired_t_test(diffs)
            paired_rule = {
                "test": "paired_t_on_seed_differences",
                "n_pairs": len(common_train_seeds),
                "effect": paired_test.get("effect"),
                "se": paired_test.get("se"),
                "df": paired_test.get("df"),
                "p_value": paired_test.get("p_value"),
                "alpha": _SIGNIFICANCE_LEVEL,
                "basis": (
                    "不显著只能说明当前种子数无法区分候选，"
                    "不是等价性证明；这是预声明的选型简化规则"
                ),
            }
            # 配对差值不显著 → 无法区分 → 进入保守并列集合。
            is_tied = bool(
                paired_test.get("p_value") is not None
                and float(paired_test["p_value"])
                > _SIGNIFICANCE_LEVEL
            )
        if paired_rule is None and not is_tied:
            noise = max(
                best_std,
                float(candidate["metric_std"] or 0.0),
            )
            is_tied = noise > 0.0 and gap < noise
        candidate["paired_tie_test"] = paired_rule
        if is_tied:
            tied.append(candidate)
            if candidate is not best:
                indistinguishable.append(float(candidate["lr"]))

    winner = min(tied, key=lambda c: float(c["lr"]))
    # 只要存在并列集合（>1 个候选无法区分/精确并列），winner
    # 就是按 selection_rule 取更小 lr 的结果，如实记录。
    conservative_tie_break = len(tied) > 1
    report["winner"] = {
        "lr": float(winner["lr"]),
        "metric": metric,
        "metric_mean": float(winner["metric_mean"]),
        "metric_std": winner["metric_std"],
        "n_runs": int(winner["n_runs"]),
        "runs": winner["runs"],
    }
    report["tied_lrs"] = sorted(
        float(candidate["lr"]) for candidate in tied
    )
    report["winner_selection"] = (
        "conservative_tie_break"
        if conservative_tie_break
        else "best_mean"
    )
    report["selection_rule"] = {
        "tie_break": "smallest_lr",
        "tie_test": (
            "paired_t_on_seed_differences"
            if any(c.get("paired_tie_test") for c in usable)
            else "max_std_heuristic_fallback"
        ),
        "alpha": _SIGNIFICANCE_LEVEL,
        "note": (
            "并列 = 候选与最优均值的同种子配对差值不显著"
            "（或无法配对时差距小于较大 std 的旧启发式）；"
            "不显著不是等价性证明，winner 仍需独立种子确认"
        ),
    }

    if indistinguishable:
        report["issues"].append(
            "以下学习率候选与最优均值候选无法在统计上区分"
            "（同种子配对 t 检验不显著；已按 selection_rule "
            "从并列集合中取更小 lr 作为保守 winner）: "
            + ", ".join(
                format(lr, "g") for lr in indistinguishable
            )
        )
    elif conservative_tie_break:
        # 均值精确相等的并列情形（噪声为 0 或无候选 std）。
        report["issues"].append(
            "存在与最优均值恰好并列的学习率候选，已按 "
            "selection_rule 取更小的学习率（保守解）"
        )

    # 收敛性：winner 的学习率候选若在曲线诊断里仍是
    # not_converged / underfitting / diverged，说明限制基线的
    # 不是学习率而是训练配置/数据/实现——在欠训曲线上选出的
    # lr 不能让基线变成「调强过的基线」。
    winner_lr = float(winner["lr"])
    winner_unconverged_runs = [
        str(row.get("run"))
        for row in rows
        if _row_lr(row) == winner_lr
        and row.get("curve_status") in (
            "not_converged",
            "underfitting",
            "diverged",
            "diverging",
        )
    ]
    report["winner_unconverged_runs"] = winner_unconverged_runs
    if winner_unconverged_runs:
        report["issues"].append(
            "winner lr="
            f"{format(winner_lr, 'g')} 的部分候选运行没有训练到"
            "收敛/稳定（曲线诊断）: "
            + ", ".join(winner_unconverged_runs)
            + "。限制基线的可能不是学习率而是训练配置/数据/"
            "实现：请增加 --steps（或核查数据与实现）后重新"
            "调参，否则 winner 不能代表「调强过的基线」"
        )

    # 网格边界：winner 或**最优均值候选**恰是候选网格的最小/
    # 最大值时，最优可能落在网格之外，「调过参的基线」仍可能是
    # 被网格边界限制的弱基线；必须提示扩展候选后重跑。只检查
    # 保守并列 winner 会漏掉「最优均值在边界、保守 winner 在
    # 内部」的情形。
    sorted_grid = sorted(float(lr) for lr in lrs)
    boundary_lrs = (
        (sorted_grid[0], sorted_grid[-1]) if sorted_grid else ()
    )
    winner_at_boundary = bool(boundary_lrs) and (
        float(winner["lr"]) in boundary_lrs
    )
    best_at_boundary = bool(boundary_lrs) and (
        float(best["lr"]) in boundary_lrs
    )
    report["winner_at_grid_boundary"] = winner_at_boundary
    report["best_mean_at_grid_boundary"] = best_at_boundary
    if (winner_at_boundary or best_at_boundary) and len(
        sorted_grid
    ) > 1:
        reason = (
            f"winner lr={format(float(winner['lr']), 'g')}"
            if winner_at_boundary
            else "最优均值候选 lr="
            f"{format(float(best['lr']), 'g')}"
        )
        report["issues"].append(
            reason
            + " 位于候选网格边界（"
            + ", ".join(
                format(lr, "g") for lr in sorted_grid
            )
            + "）：最优可能落在网格之外，当前「调过参的"
            "基线」仍可能被网格边界限制；建议扩展 "
            "--baseline-tuning-lrs（如加入更小/更大的"
            "候选）后重跑调参"
        )

    return report



def _axis_value_equal(left: object, right: object) -> bool:
    """轴取值比较：浮点按相对误差比较，其余按等值比较。"""
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    if left_number is not None and right_number is not None:
        # 只用相对容差：固定 1e-12 绝对容差会把 1e-13 量级的
        # 两个不同学习率/超参判成同一个轴取值，从而静默合并
        # 候选配置。正常量级下相对容差与原行为一致。
        return math.isclose(
            left_number,
            right_number,
            rel_tol=1e-9,
            abs_tol=0.0,
        )
    # 超出 float 范围的整数仍可精确相等比较；不能因为无法转成
    # float 就把两个完全相同的合法 JSON 大整数判成不一致。
    return left == right



def _axis_value_sort_key(value: object) -> Tuple[int, object, str]:
    """轴取值的排序键：数值升序在前，其余按文本。"""
    if isinstance(value, bool):
        return (1, 0.0, str(value))
    if isinstance(value, int):
        # 整数保留原值：与 float 可直接比较，且不会在超大整数
        # 上溢出；同一轴内混合 int/float 时仍按数值大小排序。
        return (0, value, "")
    number = _finite_float(value)
    if number is not None:
        return (0, number, "")
    return (1, 0.0, str(value))



def _axis_value_text(value: object) -> str:
    """轴取值的可读文本（错误消息与表格用）。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)



# 调参轴 → 命令行开关（推荐配置与错误消息用）。
_BASELINE_TUNING_AXIS_CLI_FLAGS: Dict[str, str] = {
    "lr": "--lr",
    "dense_fnn_hidden": "--dense-fnn-hidden",
    "weight_decay": "--weight-decay",
}



def _axis_cli_flag(axis_name: str) -> str:
    return _BASELINE_TUNING_AXIS_CLI_FLAGS.get(
        axis_name,
        f"--{axis_name.replace('_', '-')}",
    )



def analyze_baseline_tuning_axes(
    rows: List[Dict[str, object]],
    metric: str,
    mode: str,
    axes: List[Dict[str, object]],
    seeds: List[int],
    base_config: Dict[str, object],
) -> Dict[str, object]:
    """
    坐标式（多轴）基线调参分析：逐轴判定 winner，并给出各轴
    winner 组合出来的**推荐配置**。

    为什么必须逐轴而不是直接比所有组合：坐标式设计里每个候选
    相对基线只改一个轴，跨轴的候选之间容量/正则都不同，直接比较
    会把「轴的效应」与「组合效应」混在一起。逐轴分析保证每个轴
    的结论只归因于该轴。

    推荐配置是各轴 winner 的组合，**本轮网格通常没有直接评估过
    这个组合**（坐标式设计的固有性质）。报告必须如实标注这一点
    （recommended_config_evaluated），并要求最终基线训练用该组合
    再跑一遍作为确认——否则「调过参的基线」只是一个未经验证的
    组合猜测。

    其它检查与单轴调参一致：候选必须用同一组随机种子（否则均
    值差混着种子难度差）、同一候选内不得有重复种子（伪重复）、
    组内训练协议必须逐项一致（除该轴自身）、winner 未收敛时
    提示限制基线的可能不是超参、winner/最优均值落在网格边界时
    提示扩展候选。
    """
    higher_is_better = metric_higher_is_better(metric)

    axis_reports: List[Dict[str, object]] = []
    global_issues: List[str] = []

    for axis in axes:
        key = str(axis["protocol_key"])
        label = str(axis.get("label", key))
        base_value = axis.get("base_value")
        candidate_values = list(axis.get("values", []))
        axis_issues: List[str] = []

        # 该轴的候选组：其它调参轴必须都停在基线值（坐标式设计
        # 的定义——每个候选相对基线只改一个轴）。
        group: List[Tuple[object, Dict[str, object]]] = []
        for row in rows:
            protocol = row.get("protocol")
            if not isinstance(protocol, dict):
                continue
            if key not in protocol:
                continue
            others_at_base = True
            for other in axes:
                other_key = str(other["protocol_key"])
                if other_key == key:
                    continue
                if not _axis_value_equal(
                    protocol.get(other_key),
                    other.get("base_value"),
                ):
                    others_at_base = False
                    break
            if not others_at_base:
                continue
            group.append((protocol.get(key), row))

        # 观察到的取值：候选值 + 基线值（基线运行是共同参照点）。
        observed_values: List[object] = []
        for value in list(candidate_values) + [base_value]:
            if value is None:
                continue
            if not any(
                _axis_value_equal(value, other)
                for other in observed_values
            ):
                observed_values.append(value)
        observed_values.sort(key=_axis_value_sort_key)

        candidates: List[Dict[str, object]] = []
        candidate_seed_sets: Dict[str, List[int]] = {}
        duplicate_seeds_by_value: Dict[str, List[int]] = {}
        missing_seed_runs: List[str] = []

        for value in observed_values:
            value_rows = [
                row
                for observed, row in group
                if _axis_value_equal(observed, value)
            ]
            metric_values = [
                number
                for row in value_rows
                for number in [_finite_float(row.get(metric))]
                if number is not None
            ]
            seed_values: List[int] = []
            for row in value_rows:
                seed = row.get("seed")
                if isinstance(seed, int) and not isinstance(seed, bool):
                    seed_values.append(int(seed))
                else:
                    missing_seed_runs.append(str(row.get("run")))
            unique_seeds = sorted(set(seed_values))
            value_key = _axis_value_text(value)
            candidate_seed_sets[value_key] = unique_seeds
            duplicates = sorted({
                seed
                for seed in unique_seeds
                if seed_values.count(seed) > 1
            })
            if duplicates:
                duplicate_seeds_by_value[value_key] = duplicates
            candidates.append({
                "value": value,
                "is_base": _axis_value_equal(value, base_value),
                "n_runs": len(value_rows),
                "n_metric_runs": len(metric_values),
                "metric_mean": _mean(metric_values),
                "metric_std": _sample_std(metric_values),
                "runs": [str(row.get("run")) for row in value_rows],
                "seeds": unique_seeds,
            })
            if len(metric_values) < len(value_rows):
                axis_issues.append(
                    f"{label}={value_key} 有 "
                    f"{len(value_rows) - len(metric_values)} 个候选"
                    f"运行缺少 {metric}，未计入均值"
                )

        # 种子对齐：各候选必须用同一组种子，否则均值差可能混着
        # 种子难度差；同一候选内的重复种子是伪重复。
        seed_alignment: Dict[str, object] = {
            "candidate_seed_sets": candidate_seed_sets,
            "common_seeds": [],
            "aligned": None,
            "duplicate_seeds": duplicate_seeds_by_value,
            "missing_seed_runs": missing_seed_runs,
        }
        nonempty_sets = [
            tuple(value_seeds)
            for value_seeds in candidate_seed_sets.values()
            if value_seeds
        ]
        if duplicate_seeds_by_value:
            seed_alignment["aligned"] = False
            axis_issues.append(
                f"同一{label}候选内出现重复随机种子（伪重复）："
                + "; ".join(
                    f"{value_key} 种子 "
                    + ", ".join(str(seed) for seed in value_seeds)
                    for value_seeds, value_key in sorted(
                        (
                            (value_seeds, value_key)
                            for value_key, value_seeds
                            in duplicate_seeds_by_value.items()
                        )
                    )
                )
                + "。重复运行不是独立样本，会低估种子间方差，"
                "winner 不能作为「调优过的基线」证据"
            )
        elif missing_seed_runs:
            seed_alignment["aligned"] = None
            axis_issues.append(
                f"部分{label}调参运行缺少随机种子记录，无法核对"
                "候选是否在同一组种子上评估：均值差可能混入种子"
                "差异，不能作为「调优过的基线」证据"
            )
        elif len(set(nonempty_sets)) > 1 or len(
            nonempty_sets
        ) != len(observed_values):
            seed_alignment["aligned"] = False
            axis_issues.append(
                f"各{label}候选使用的随机种子集合不一致："
                + "; ".join(
                    f"{value_key} 种子=["
                    + ", ".join(str(seed) for seed in value_seeds)
                    + "]"
                    for value_key, value_seeds in sorted(
                        candidate_seed_sets.items()
                    )
                )
                + "。均值差可能是种子难度差异而不是该轴的差异，"
                "winner 不能作为「调优过的基线」证据"
            )
        else:
            seed_alignment["aligned"] = True
            seed_alignment["common_seeds"] = (
                list(nonempty_sets[0]) if nonempty_sets else []
            )

        # 组内训练协议一致性：除该轴自身外必须逐项相同，否则
        # 「该轴更优」可能与「换了别的训练配置」混杂。
        protocol_keys = [
            protocol_key
            for protocol_key in _BASELINE_TUNING_PROTOCOL_KEYS
            if protocol_key != key
        ]
        protocol_values: Dict[str, set] = {
            protocol_key: set() for protocol_key in protocol_keys
        }
        protocol_value_counts = {
            protocol_key: 0 for protocol_key in protocol_keys
        }
        recorded_protocol = 0
        for _, row in group:
            protocol = row.get("protocol")
            if not isinstance(protocol, dict):
                continue
            recorded_protocol += 1
            for protocol_key in protocol_keys:
                value = protocol.get(protocol_key)
                if value is None:
                    continue
                protocol_value_counts[protocol_key] += 1
                protocol_values[protocol_key].add(
                    _protocol_value_text(value)
                )
        inconsistent_keys = [
            protocol_key
            for protocol_key, values in protocol_values.items()
            if len(values) > 1
        ]
        if inconsistent_keys:
            axis_issues.append(
                f"该{label}候选之间的训练协议不一致（"
                + ", ".join(inconsistent_keys)
                + "）：调参结果与「训练得更久 / 换了优化协议或"
                "数据划分」混杂，不可用于选择基线超参数"
            )
        partial_protocol_keys = []
        axis_training_config: Dict[str, object] = {}
        if recorded_protocol:
            for protocol_key, values in protocol_values.items():
                if (
                    len(values) == 1
                    and protocol_value_counts[protocol_key]
                    == recorded_protocol
                ):
                    axis_training_config[protocol_key] = json.loads(
                        next(iter(values))
                    )
                elif len(values) == 1:
                    partial_protocol_keys.append(protocol_key)
        if partial_protocol_keys:
            axis_issues.append(
                "部分候选运行缺少训练协议字段（"
                + ", ".join(partial_protocol_keys)
                + "）：无法确认所有候选使用同一训练协议，"
                "winner 不能作为「调优过的基线」证据"
            )

        # winner 选择：均值最优；与最优均值的差距落在种子波动内
        # 的候选视为无法区分，并列时优先取基线值（保守），否则取
        # 数值更小的候选。
        usable = [
            candidate
            for candidate in candidates
            if candidate["metric_mean"] is not None
            and int(candidate["n_runs"]) > 0
        ]
        winner: Optional[Dict[str, object]] = None
        tied: List[Dict[str, object]] = []
        best: Optional[Dict[str, object]] = None
        winner_selection = "none"
        if not usable:
            axis_issues.append(
                f"没有任何候选提供可用的 {metric}，无法选出"
                f"{label}的 winner"
            )
        else:
            def _mean_order(
                candidate: Dict[str, object],
            ) -> float:
                mean = float(candidate["metric_mean"])
                return -mean if higher_is_better else mean

            best = min(usable, key=_mean_order)
            best_mean = float(best["metric_mean"])
            best_std = float(best["metric_std"] or 0.0)
            for candidate in usable:
                mean = float(candidate["metric_mean"])
                gap = (
                    best_mean - mean
                    if higher_is_better
                    else mean - best_mean
                )
                noise = max(
                    best_std,
                    float(candidate["metric_std"] or 0.0),
                )
                if gap <= 0.0 or (noise > 0.0 and gap < noise):
                    tied.append(candidate)
            winner = min(
                tied,
                key=lambda candidate: (
                    0 if candidate["is_base"] else 1,
                    _axis_value_sort_key(candidate["value"]),
                ),
            )
            winner_selection = (
                "conservative_tie_break" if len(tied) > 1 else "best_mean"
            )
            indistinguishable = [
                candidate
                for candidate in tied
                if candidate is not winner
            ]
            if indistinguishable:
                axis_issues.append(
                    f"以下{label}候选与最优均值候选的差距小于种子"
                    "波动（无法在统计上区分，已按 selection_rule "
                    "从并列集合中取保守 winner）: "
                    + ", ".join(
                        _axis_value_text(candidate["value"])
                        for candidate in indistinguishable
                    )
                )
            elif len(tied) > 1:
                axis_issues.append(
                    f"存在与最优均值恰好并列的{label}候选，已按 "
                    "selection_rule 取保守 winner"
                )

        # 收敛性：winner 仍在欠训/发散时，限制基线的可能不是该
        # 轴的超参而是训练配置/数据/实现。
        winner_unconverged_runs: List[str] = []
        if winner is not None:
            winner_unconverged_runs = [
                str(row.get("run"))
                for observed, row in group
                if _axis_value_equal(observed, winner["value"])
                and row.get("curve_status") in (
                    "not_converged",
                    "underfitting",
                    "diverged",
                    "diverging",
                )
            ]
            if winner_unconverged_runs:
                axis_issues.append(
                    f"winner {label}="
                    f"{_axis_value_text(winner['value'])} 的部分候选"
                    "运行没有训练到收敛/稳定（曲线诊断）: "
                    + ", ".join(winner_unconverged_runs)
                    + "。限制基线的可能不是该超参而是训练配置/"
                    "数据/实现：请增加 --steps（或核查数据与实现）"
                    "后重新调参，否则 winner 不能代表「调强过的基线」"
                )

        # 网格边界：winner 或最优均值候选落在候选边界时，最优
        # 可能落在网格之外。
        candidate_grid: List[object] = []
        for value in candidate_values:
            if value is None:
                continue
            if not any(
                _axis_value_equal(value, other)
                for other in candidate_grid
            ):
                candidate_grid.append(value)
        candidate_grid.sort(key=_axis_value_sort_key)
        boundary_values = (
            (candidate_grid[0], candidate_grid[-1])
            if candidate_grid
            else ()
        )
        winner_at_boundary = False
        best_at_boundary = False
        if boundary_values and winner is not None:
            winner_at_boundary = any(
                _axis_value_equal(winner["value"], edge)
                for edge in boundary_values
            )
            best_at_boundary = any(
                _axis_value_equal(best["value"], edge)
                for edge in boundary_values
            )
            if (
                winner_at_boundary or best_at_boundary
            ) and len(candidate_grid) > 1:
                reason = (
                    f"winner {label}="
                    f"{_axis_value_text(winner['value'])}"
                    if winner_at_boundary
                    else f"最优均值候选 {label}="
                    f"{_axis_value_text(best['value'])}"
                )
                axis_issues.append(
                    reason
                    + " 位于候选网格边界（"
                    + ", ".join(
                        _axis_value_text(value)
                        for value in candidate_grid
                    )
                    + "）：最优可能落在网格之外，当前「调过参的"
                    "基线」仍可能被网格边界限制；建议扩展该轴的"
                    "候选后重跑调参"
                )

        axis_reports.append({
            "name": str(axis.get("name", key)),
            "protocol_key": key,
            "label": label,
            "base_value": base_value,
            "values": observed_values,
            "candidates": candidates,
            "winner": (
                {
                    "value": winner["value"],
                    # 学习率轴额外给出 winner.lr：下游的调参来源
                    # 记录与基线合理性检查按 winner.lr 核对。
                    **(
                        {"lr": winner["value"]}
                        if key == "lr"
                        else {}
                    ),
                    "metric": metric,
                    "metric_mean": winner["metric_mean"],
                    "metric_std": winner["metric_std"],
                    "n_runs": winner["n_runs"],
                    "runs": winner["runs"],
                }
                if winner is not None
                else None
            ),
            "tied_values": [
                candidate["value"] for candidate in tied
            ],
            "winner_selection": winner_selection,
            "winner_at_boundary": winner_at_boundary,
            "best_mean_at_boundary": best_at_boundary,
            "winner_unconverged_runs": winner_unconverged_runs,
            "seed_alignment": seed_alignment,
            "training_config": axis_training_config,
            "issues": axis_issues,
        })

    # 推荐配置：各轴 winner 的组合。
    axis_winners: Dict[str, object] = {}
    for axis_report in axis_reports:
        winner = axis_report.get("winner")
        if isinstance(winner, dict):
            axis_winners[str(axis_report["name"])] = winner.get(
                "value"
            )
        else:
            axis_winners[str(axis_report["name"])] = axis_report.get(
                "base_value"
            )

    recommended_config: Dict[str, object] = {}
    for axis in axes:
        name = str(axis.get("name", axis["protocol_key"]))
        recommended_config[name] = axis_winners.get(
            name,
            axis.get("base_value"),
        )

    # 推荐组合是否在网格里被真正评估过：坐标式设计下通常没有。
    recommended_evaluated = False
    for row in rows:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            continue
        matched = True
        for axis in axes:
            key = str(axis["protocol_key"])
            name = str(axis.get("name", key))
            if key not in protocol:
                matched = False
                break
            if not _axis_value_equal(
                protocol.get(key),
                recommended_config.get(name),
            ):
                matched = False
                break
        if matched:
            recommended_evaluated = True
            break

    issues: List[str] = list(global_issues)
    for axis_report in axis_reports:
        for issue in axis_report["issues"]:
            issues.append(str(issue))

    if not recommended_evaluated:
        issues.append(
            "各轴 winner 组合出的推荐配置（"
            + ", ".join(
                f"{name}={_axis_value_text(value)}"
                for name, value in recommended_config.items()
            )
            + "）没有在本轮坐标式网格中被直接评估过（坐标式"
            "设计每次只改一个轴）：它是各轴结论的组合猜测，"
            "必须用该组合再训练一次基线来确认；未确认前不能"
            "把它当作「已验证的强基线」"
        )

    # 训练配置（基线协议的公共部分）：调参轴自身的取值由推荐
    # 配置单独记录。
    lr_axis = next(
        (
            axis_report
            for axis_report in axis_reports
            if axis_report["protocol_key"] == "lr"
        ),
        None,
    )

    training_config: Dict[str, object] = {
        key: None for key in _BASELINE_TUNING_PROTOCOL_KEYS
    }
    if isinstance(lr_axis, dict):
        axis_config = lr_axis.get("training_config")
        if isinstance(axis_config, dict):
            training_config.update(axis_config)

    report: Dict[str, object] = {
        "comparison_family": "baseline_tuning",
        "design": "coordinate",
        "multi_axis": True,
        "mode": mode,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "axes": axis_reports,
        "axis_winners": axis_winners,
        "base_config": dict(base_config),
        "recommended_config": recommended_config,
        "recommended_config_evaluated": recommended_evaluated,
        "grid": [
            float(value)
            for value in (
                lr_axis.get("values") if isinstance(lr_axis, dict) else []
            )
        ],
        "seeds": [int(seed) for seed in seeds],
        "n_candidates": sum(
            len(axis_report["candidates"])
            for axis_report in axis_reports
        ),
        "n_runs": len(rows),
        "training_config": training_config,
        "protocol_keys": list(_BASELINE_TUNING_PROTOCOL_KEYS),
        "candidates": [
            {
                "axis": axis_report["name"],
                **candidate,
            }
            for axis_report in axis_reports
            for candidate in axis_report["candidates"]
        ],
        "winner": (
            lr_axis.get("winner") if isinstance(lr_axis, dict) else None
        ),
        "winner_selection": (
            lr_axis.get("winner_selection")
            if isinstance(lr_axis, dict)
            else "none"
        ),
        "tied_lrs": (
            [
                float(value)
                for value in (lr_axis.get("tied_values") or [])
            ]
            if isinstance(lr_axis, dict)
            else []
        ),
        "winner_at_grid_boundary": (
            bool(lr_axis.get("winner_at_boundary"))
            if isinstance(lr_axis, dict)
            else False
        ),
        "best_mean_at_grid_boundary": (
            bool(lr_axis.get("best_mean_at_boundary"))
            if isinstance(lr_axis, dict)
            else False
        ),
        "winner_unconverged_runs": (
            list(lr_axis.get("winner_unconverged_runs") or [])
            if isinstance(lr_axis, dict)
            else []
        ),
        "seed_plan": _build_seed_plan(
            seeds,
            False,
        ),
        "seed_alignment": (
            lr_axis.get("seed_alignment")
            if isinstance(lr_axis, dict)
            else {}
        ),
        "selection_rule": (
            "逐轴在 val 上先取指标均值最优的候选；与最优均值的"
            "差距小于种子波动（两侧 std 的较大者）的候选视为并列"
            "（无法在统计上区分），从并列集合中取基线值（没有则取"
            "数值更小的候选）作为该轴 winner；各轴 winner 的组合"
            "即推荐配置"
        ),
        "issues": issues,
        "caveats": [
            "调参只看 val：test 集全程尘封，不参与候选排序",
            "坐标式设计：每个候选相对基线只改一个轴，其余轴固定"
            "在基线值，因此各轴结论互不混杂；跨轴的候选之间容量/"
            "正则都不同，不能直接横向比较",
            "推荐配置是各轴 winner 的组合，坐标式网格通常没有直接"
            "评估过它：必须用该组合再训练一次基线来确认",
            "调参后应固定推荐配置训练最终基线，并用 "
            "--baseline-tuning-from 记录来源；该来源会逐项核对"
            "训练协议与各轴取值，协议不一致时拒绝；复杂模型必须"
            "战胜这条调过参的基线才有意义",
        ],
    }

    return report



def print_baseline_tuning_report(
    report: Dict[str, object],
    out_root: Path,
) -> None:
    """打印基线调参网格结果与后续使用方式。"""
    metric = report["metric"]
    direction = (
        "越高越好"
        if report.get("higher_is_better")
        else "越低越好"
    )
    config = report.get("training_config") or {}

    def _fmt(value) -> str:
        return _format_finite(value, ".6f")

    # 多轴（坐标式）调参：逐轴打印候选与 winner，再打印组合出的
    # 推荐配置。
    if report.get("multi_axis"):
        print()
        print(
            "[baseline-tuning] 简单基线多轴坐标式调参（val，"
            "test 集尘封不参与；每个候选相对基线只改一个轴）"
        )
        print(
            f"  模式: {report['mode']}；指标: {metric}（{direction}"
            + (
                f"，来源={report['metric_source']}"
                if report.get("metric_source")
                else ""
            )
            + "）；"
            f"训练配置: steps={config.get('total_steps')}, "
            f"batch_size={config.get('batch_size')}"
        )
        for axis_report in report.get("axes", []):
            label = axis_report.get("label", axis_report.get("name"))
            print(f"  [{label}] 候选：")
            print(
                f"  {'value':>12} {'base':>6} {'n':>3} "
                f"{'mean':>12} {'std':>12}"
            )
            for candidate in axis_report.get("candidates", []):
                print(
                    f"  {_axis_value_text(candidate['value']):>12} "
                    f"{'是' if candidate.get('is_base') else '-':>6} "
                    f"{_nonnegative_int(candidate.get('n_runs')):>3} "
                    f"{_fmt(candidate.get('metric_mean')):>12} "
                    f"{_fmt(candidate.get('metric_std')):>12}"
                )
            axis_winner = axis_report.get("winner")
            if isinstance(axis_winner, dict):
                print(
                    f"  [winner] {label}="
                    f"{_axis_value_text(axis_winner['value'])}, "
                    f"mean={_fmt(axis_winner.get('metric_mean'))} "
                    f"(n={axis_winner.get('n_runs')}, "
                    f"{metric}, {direction}；选择方式="
                    f"{axis_report.get('winner_selection')})"
                )
        recommended = report.get("recommended_config") or {}
        print(
            "  [推荐配置] "
            + ", ".join(
                f"--{_axis_cli_flag(name)} "
                f"{_axis_value_text(value)}"
                for name, value in recommended.items()
            )
            + (
                "（该组合已在本轮网格中直接评估）"
                if report.get("recommended_config_evaluated")
                else "（该组合未在本轮网格中直接评估：必须用它再"
                "训练一次基线来确认）"
            )
        )
        print(
            "  下一步：用上面的推荐配置训练最终基线，并加 "
            "--baseline-tuning-from "
            f"{out_root / _BASELINE_TUNING_FILENAME} 记录调参来源；"
            "复杂模型必须战胜这条调过参的基线"
        )
        for issue in report.get("issues", []):
            print(f"  [warn] {issue}")
        for caveat in report.get("caveats", []):
            print(f"  [note] {caveat}")
        return

    print()
    print(
        "[baseline-tuning] 简单基线学习率网格搜索（val，"
        "test 集尘封不参与）"
    )
    print(
        f"  模式: {report['mode']}；指标: {metric}（{direction}"
        + (
            f"，来源={report['metric_source']}"
            if report.get("metric_source")
            else ""
        )
        + "）；"
        f"训练配置: steps={config.get('total_steps')}, "
        f"batch_size={config.get('batch_size')}"
    )
    print(f"  {'lr':>10} {'n':>3} {'mean':>12} {'std':>12}")

    for candidate in report.get("candidates", []):
        print(
            f"  {_format_finite(candidate.get('lr'), 'g'):>10} "
            f"{_nonnegative_int(candidate.get('n_runs')):>3} "
            f"{_fmt(candidate.get('metric_mean')):>12} "
            f"{_fmt(candidate.get('metric_std')):>12}"
        )

    winner = report.get("winner")
    if isinstance(winner, dict):
        print(
            f"  [winner] lr={_format_finite(winner.get('lr'), 'g')}, "
            f"mean={_fmt(winner.get('metric_mean'))} "
            f"(n={winner.get('n_runs')}, "
            f"{metric}, {direction})"
        )
        selection = report.get("winner_selection")
        if selection == "conservative_tie_break":
            tied = report.get("tied_lrs") or []
            print(
                "  [winner-selection] 最优均值候选与更小 lr 的"
                "差距落在种子波动内，按 selection_rule 取保守"
                "解；并列候选: "
                + ", ".join(
                    _format_finite(lr, "g") for lr in tied
                )
            )
        else:
            print(
                "  [winner-selection] winner 为最优均值候选"
                "（没有与之无法区分的更小 lr）"
            )
        print(
            "  下一步：用 --suite-modes "
            f"{report['mode']} --seeds ... --lr "
            f"{_format_finite(winner.get('lr'), 'g')} 训练最终基线，"
            "并加 --baseline-tuning-from "
            f"{out_root / _BASELINE_TUNING_FILENAME} 记录调参来源；"
            "复杂模型必须战胜这条调过参的基线"
        )
    for issue in report.get("issues", []):
        print(f"  [warn] {issue}")
    for caveat in report.get("caveats", []):
        print(f"  [note] {caveat}")



def write_baseline_tuning_report(
    out_dir: Path,
    report: Dict[str, object],
) -> Path:
    """把调参报告写入 out_dir/baseline_tuning.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _BASELINE_TUNING_FILENAME

    payload = {
        "generated_at": datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        **report,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _json_safe(payload),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[report] 基线调参报告已写入 {path}")
    return path
