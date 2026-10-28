from __future__ import annotations



def _pilot_seed_values(value: object) -> Tuple[object, ...]:
    """把 seed/证据容器归一化为可迭代元组，损坏标量不崩溃。"""
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)



def _pilot_seed_int(value: object) -> Optional[int]:
    """把计划/证据里的 seed 归一化成正整数。

    ``cells[].seed_values`` 按 JSON 对象保存时键一定是字符串；
    直接交给只接受数值的 ``_finite_float`` 会让重叠检测永远为
    0。旧/手工记录也可能把 seed 存成数值字符串，这里一并兼容。
    """
    number = _finite_float(value)
    if number is not None:
        return int(number) if number.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                numeric = float(text)
            except (TypeError, ValueError, OverflowError):
                return None
            if math.isfinite(numeric) and numeric.is_integer():
                return int(numeric)
    return None



def _pilot_seed_overlap(
    pilot_evidence: Dict[str, object],
    planned_seeds: Sequence[object],
) -> Dict[str, object]:
    planned: set = set()
    for seed in _pilot_seed_values(planned_seeds):
        number = _pilot_seed_int(seed)
        if number is not None:
            planned.add(number)
    reused_cells: List[Dict[str, object]] = []
    reused_total = 0
    pilot_evidence = _as_dict(pilot_evidence)
    for cell in _pilot_seed_values(
        pilot_evidence.get("cells")
    ):
        if not isinstance(cell, dict):
            continue
        seed_values = cell.get("seed_values")
        if isinstance(seed_values, dict):
            seed_values = tuple(seed_values)
        pilot_seeds = set()
        for seed in _pilot_seed_values(seed_values):
            number = _pilot_seed_int(seed)
            if number is not None:
                pilot_seeds.add(number)
        reused = sorted(pilot_seeds & planned)
        if reused:
            reused_total += len(reused)
            reused_cells.append({
                "cell": cell.get("label"),
                "reused_seeds": reused,
            })
    return {
        "planned_seed_count": len(planned),
        "reused_seed_count": reused_total,
        "reused_cells": reused_cells,
        "independent_from_confirmatory_runs": reused_total == 0,
        "note": (
            "试点 seed 同时出现在正式计划中时，试点与确证运行"
            "不是独立证据；若只在试点上做工程管道检查可以接受，"
            "但必须在报告中披露，不能把同一批运行既当调参/"
            "设计依据又当独立确认"
        ),
    }



def _pilot_comparability_report(
    rows: Sequence[Dict[str, object]],
    expected: Optional[Dict[str, object]],
    pilot_evidence: Dict[str, object],
) -> Dict[str, object]:
    expected = _as_dict(expected)
    rows = _pilot_row_dicts(rows)
    items: List[Dict[str, object]] = []
    for group, keys, severity in _PILOT_COMPARABILITY_GROUPS:
        for key in keys:
            items.append(
                _pilot_comparability_item(
                    key, group, severity, expected, rows
                )
            )
    blocking_mismatches = [
        item for item in items
        if item["severity"] == "blocking"
        and item["status"] == "mismatch"
    ]
    advisory_mismatches = [
        item for item in items
        if item["severity"] == "advisory"
        and item["status"] == "mismatch"
    ]
    unverified = [
        item for item in items
        if item["severity"] == "blocking"
        and item["status"] in ("not_recorded", "partial")
    ]
    seed_overlap = _pilot_seed_overlap(
        pilot_evidence,
        expected.get("planned_seeds") or [],
    )
    warnings: List[str] = []
    for item in blocking_mismatches:
        warnings.append(
            "试点协议与正式计划不可比："
            f"{item['group']}.{item['key']} "
            f"期望 {item['expected']!r}，"
            "试点值 " + ", ".join(
                f"{name}={value!r}"
                for name, value in list(
                    item["values"].items()
                )[:4]
            )
        )
    for item in advisory_mismatches:
        warnings.append(
            "试点与正式环境/数值条件不同："
            f"{item['group']}.{item['key']} "
            f"期望 {item['expected']!r}，"
            "试点值 " + ", ".join(
                f"{name}={value!r}"
                for name, value in list(
                    item["values"].items()
                )[:4]
            )
        )
    if unverified:
        warnings.append(
            f"{len(unverified)} 个关键协议项在试点运行中缺记录"
            "或只有部分运行可核对；本次校准应标记为 unverified，"
            "正式确证前建议用同一协议重跑试点"
        )
    if seed_overlap["reused_seed_count"]:
        warnings.append(
            "试点运行的 seed 与正式计划重叠 "
            f"{seed_overlap['reused_seed_count']} 个；"
            "试点与确证运行不是完全独立的证据"
        )
    if blocking_mismatches:
        status = "incomparable"
        comparable: Optional[bool] = False
        usable = False
    elif unverified:
        status = "unverified"
        comparable = None
        usable = True
    elif advisory_mismatches:
        status = "comparable_with_cautions"
        comparable = True
        usable = True
    else:
        status = "comparable"
        comparable = True
        usable = True
    return {
        "schema": (
            "smelt-isr/design-pilot-comparability/"
            f"v{_PILOT_COMPARABILITY_SCHEMA_VERSION}"
        ),
        "status": status,
        "comparable": comparable,
        "usable_for_calibration": usable,
        "verified": status in (
            "comparable",
            "comparable_with_cautions",
        ),
        "checked_keys": len(items),
        "blocking_mismatch_count": len(blocking_mismatches),
        "advisory_mismatch_count": len(advisory_mismatches),
        "unverified_key_count": len(unverified),
        "blocking_mismatches": blocking_mismatches,
        "advisory_mismatches": advisory_mismatches,
        "unverified_keys": [
            {
                "group": item["group"],
                "key": item["key"],
                "status": item["status"],
                "missing_runs": item["missing_runs"],
            }
            for item in unverified
        ],
        "items": items,
        "seed_overlap": seed_overlap,
        "warnings": warnings,
        "note": (
            "只有 blocking mismatch 会阻止把试点 sd/脱落率用于"
            "正式设计；unverified 表示旧运行缺少代码/数据/训练/"
            "评估协议记录，不能声称已核对一致。环境/数值条件"
            "差异只作 caution，因为同一代码在不同设备上仍可能"
            "改变种子间波动"
        ),
    }



def build_pilot_power_evidence(
    pilot_runs: Sequence[object],
    metric: str,
    target_step: Optional[float] = None,
    declared_sd: Optional[float] = None,
    declared_attrition: Optional[float] = None,
    planned_modes: Optional[Sequence[object]] = None,
    expected_protocol: Optional[Dict[str, object]] = None,
    calibration_confidence: float = _PILOT_DEFAULT_CONFIDENCE,
) -> Dict[str, object]:
    """从试点/先前运行估计配对差值 sd 与脱落率，并对照声明值。

    先验功效最薄弱的输入是 ``expected_seed_sd_relative`` 与
    ``expected_attrition_rate``：二者若只凭研究者声明，看到结果
    后把 sd 写小、把脱落率写低，就能让任何种子数「通过」设计门。
    这里读取**已经完成的**试点/先前运行目录（含失败目录），按
    相同 seed 配对估计两格差值的样本 sd，并按处理单元统计失败/
    缺指标比例，作为设计计划的一部分冻结。

    约束与口径：

    - 只使用 val 口径指标；``*_at_common_step`` 与
      ``val_loss_curve_mean_to_common_step`` 走与正式分析相同的
      派生逻辑，其它指标直接读运行记录；
    - 只把数值失败或缺失该指标的种子视为不可用；同一处理的
      重复 seed 按一个独立观测折叠并显式告警；
    - 配对 sd 取**所有可配对处理单元对**的最大值（保守上界），
      相对化时用两单元均值中较小者作分母（同样偏保守）；
    - 脱落率取各处理单元的最大值，而不是总体平均：某个臂更
      容易崩掉时，总体平均会把它稀释；
    - 点估计之外同时报告单侧置信上界（配对 sd 用正态/卡方
      上界，脱落率用 Clopper-Pearson 精确上界）；小试点的
      上界会很宽，是否按上界做设计由预注册
      ``power.pilot_calibration`` 决定，不能看到试点后再选；
    - 这不是因果效应估计，也不保证试点协议与正式协议完全
      可比；它只回答「声明的先验噪声/脱落是否至少与已观测
      的运行一致」。
    """
    confidence = _finite_float(calibration_confidence)
    if confidence is None or not 0.5 < confidence < 1.0:
        confidence = _PILOT_DEFAULT_CONFIDENCE
    paths = [str(path) for path in pilot_runs]
    rows = compare_runs(paths)

    annotation_report: Optional[Dict[str, object]] = None
    if metric in _COMMON_STEP_METRICS:
        annotation_kind = "common_step"
        annotation_report = annotate_common_step_metrics(
            rows, target_step=target_step
        )
    elif metric == _CURVE_MEAN_METRIC:
        annotation_kind = "curve_mean"
        annotation_report = annotate_curve_metrics(
            rows, target_step=target_step
        )
    elif metric == _COMPUTE_MATCHED_METRIC:
        annotation_kind = "matched_flops"
        annotation_report = annotate_compute_matched_metrics(rows)
    else:
        annotation_kind = "recorded_metric"

    annotation: Dict[str, object] = {
        "kind": annotation_kind,
        # ``val_loss`` 等记录值没有共同步数/计算量口径：
        # 不能把命令行 --common-step 写进这里冒充「该指标
        # 取自共同步数」。只有派生指标才带 target_step。
        "target_step": (
            None if annotation_kind == "recorded_metric"
            else target_step
        ),
        "target_step_rule": (
            "not_applicable"
            if annotation_kind == "recorded_metric"
            else None
        ),
        "target_step_data_dependent": (
            False if annotation_kind == "recorded_metric"
            else None
        ),
    }
    if isinstance(annotation_report, dict):
        if annotation_kind == "curve_mean":
            annotation["target_step"] = annotation_report.get(
                "end_step", target_step
            )
            annotation["target_step_rule"] = annotation_report.get(
                "end_rule"
            )
            annotation["target_step_data_dependent"] = (
                annotation_report.get("end_data_dependent")
            )
        elif annotation_kind == "matched_flops":
            annotation["target_step"] = annotation_report.get(
                "target_flops"
            )
            annotation["target_step_rule"] = (
                "min_last_curve_flops"
            )
            annotation["target_step_data_dependent"] = True
        else:
            annotation["target_step"] = annotation_report.get(
                "target_step", target_step
            )
            annotation["target_step_rule"] = annotation_report.get(
                "target_step_rule"
            )
            annotation["target_step_data_dependent"] = (
                annotation_report.get(
                    "target_step_data_dependent"
                )
            )
        if annotation_report.get("advisories"):
            annotation["advisories"] = list(
                annotation_report["advisories"]
            )

    warnings: List[str] = []
    source_runs: List[Dict[str, object]] = []
    unassigned: List[str] = []
    unassigned_rows: List[Dict[str, object]] = []
    cells: Dict[tuple, Dict[str, object]] = {}

    for row in rows:
        mode = row.get("mode")
        seed = row.get("seed")
        run_name = str(row.get("run"))
        value = _finite_float(row.get(metric))
        status = row.get("numerical_status")
        source_runs.append({
            "run": run_name,
            "path": row.get("path"),
            "mode": mode,
            "balance_coef": row.get("balance_coef"),
            "seed": seed,
            "numerical_status": status,
            "value": value,
        })
        if (
            mode is None
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            unassigned.append(run_name)
            unassigned_rows.append({
                "run": run_name,
                "value": value,
                "numerical_status": status,
            })
            continue
        key = _pilot_treatment_key(row)
        cell = cells.setdefault(
            key,
            {
                "mode": mode,
                "balance_coef": row.get("balance_coef"),
                "k1": row.get("k1"),
                "k2": row.get("k2"),
                "top_k": row.get("top_k"),
                "dense_fnn_hidden": row.get("dense_fnn_hidden"),
                "n_experts": row.get("n_experts"),
                "expert_hidden": row.get("expert_hidden"),
                "label": None,
                "_seeds": {},
            },
        )
        entry = cell["_seeds"].setdefault(
            int(seed),
            {"values": [], "failed": False, "duplicate": False},
        )
        if entry["values"] or entry["failed"]:
            entry["duplicate"] = True
        if status == "failed" or value is None:
            entry["failed"] = True
        else:
            entry["values"].append(value)
            entry["failed"] = False

    finalized_cells: List[Dict[str, object]] = []
    duplicate_seeds: List[str] = []
    for key in sorted(cells, key=repr):
        cell = cells[key]
        cell["label"] = _pilot_cell_label(cell)
        seed_entries = cell.pop("_seeds")
        values_by_seed: Dict[int, float] = {}
        duplicates = []
        for seed, entry in seed_entries.items():
            if entry["duplicate"]:
                duplicates.append(seed)
            if entry["values"]:
                mean_value = _mean(entry["values"])
                if mean_value is not None:
                    values_by_seed[seed] = mean_value
        n_attempted = len(seed_entries)
        n_usable = len(values_by_seed)
        n_failed = max(0, n_attempted - n_usable)
        if duplicates:
            duplicate_seeds.append(
                f"{cell['label']} seed={sorted(duplicates)}"
            )
        cell["n_attempted"] = n_attempted
        cell["n_usable"] = n_usable
        cell["n_failed"] = n_failed
        cell["attrition_rate"] = (
            n_failed / n_attempted if n_attempted else None
        )
        cell["attrition_upper_confidence_bound"] = (
            _binomial_upper_confidence_bound(
                n_failed, n_attempted, confidence
            )
            if n_attempted
            else None
        )
        cell["seed_values"] = {
            str(seed): value
            for seed, value in sorted(values_by_seed.items())
        }
        cell["seed_sd_abs"] = _sample_std(
            list(values_by_seed.values())
        )
        cell_mean = _mean(list(values_by_seed.values()))
        cell["mean"] = cell_mean
        cell_sd_relative = None
        if (
            cell["seed_sd_abs"] is not None
            and cell_mean is not None
            and cell_mean != 0.0
        ):
            try:
                cell_sd_relative = (
                    cell["seed_sd_abs"] / abs(cell_mean)
                )
            except OverflowError:
                cell_sd_relative = None
            if (
                cell_sd_relative is not None
                and not math.isfinite(cell_sd_relative)
            ):
                cell_sd_relative = None
        cell["seed_sd_relative"] = cell_sd_relative
        finalized_cells.append(cell)

    # mode+balance 相同的处理单元可能只在 k1/k2/top_k 等轴上
    # 不同；标签相同会让 pair 的 first/second 无法区分。只在
    # 发生碰撞时追加结构轴后缀，正常单轴对比保持简短标签。
    label_counts: Dict[str, int] = {}
    for cell in finalized_cells:
        label = str(cell.get("label"))
        label_counts[label] = label_counts.get(label, 0) + 1
    used_labels: set = set()
    for index, cell in enumerate(finalized_cells):
        base_label = str(cell.get("label"))
        candidate = base_label
        if label_counts.get(base_label, 0) > 1:
            suffix = ",".join(
                f"{field}={cell.get(field)!r}"
                for field in (
                    "k1",
                    "k2",
                    "top_k",
                    "dense_fnn_hidden",
                    "n_experts",
                    "expert_hidden",
                )
                if cell.get(field) is not None
            )
            candidate = (
                f"{base_label}[{suffix}]"
                if suffix
                else f"{base_label}#{index + 1}"
            )
        # 极端情况下 repr 仍可能相同（如 1 与 "1" 的稳定键
        # 不同但文本相同）；用序号兜底保证 pair 可区分。
        while candidate in used_labels:
            candidate = f"{candidate}#{index + 1}"
        cell["label"] = candidate
        used_labels.add(candidate)

    pairs: List[Dict[str, object]] = []
    for index, first in enumerate(finalized_cells):
        for second in finalized_cells[index + 1:]:
            first_values = first["seed_values"]
            second_values = second["seed_values"]
            common = sorted(
                set(first_values) & set(second_values), key=str
            )
            differences = [
                float(second_values[seed])
                - float(first_values[seed])
                for seed in common
            ]
            paired_sd = _sample_std(differences)
            first_mean = _mean(
                [float(first_values[seed]) for seed in common]
            )
            second_mean = _mean(
                [float(second_values[seed]) for seed in common]
            )
            denominator = None
            if first_mean is not None and second_mean is not None:
                denominator = min(
                    abs(first_mean), abs(second_mean)
                )
            paired_sd_relative = None
            if (
                paired_sd is not None
                and denominator is not None
                and denominator > 0.0
            ):
                try:
                    paired_sd_relative = paired_sd / denominator
                except OverflowError:
                    paired_sd_relative = None
                if (
                    paired_sd_relative is not None
                    and not math.isfinite(paired_sd_relative)
                ):
                    paired_sd_relative = None
            paired_sd_upper = _sd_upper_confidence_bound(
                paired_sd, len(common), confidence
            )
            paired_sd_relative_upper = None
            if (
                paired_sd_upper is not None
                and denominator is not None
                and denominator > 0.0
            ):
                try:
                    paired_sd_relative_upper = (
                        paired_sd_upper / denominator
                    )
                except OverflowError:
                    paired_sd_relative_upper = None
                if (
                    paired_sd_relative_upper is not None
                    and not math.isfinite(
                        paired_sd_relative_upper
                    )
                ):
                    paired_sd_relative_upper = None
            pairs.append({
                "first": first["label"],
                "second": second["label"],
                "n_pairs": len(common),
                "seeds": [int(seed) for seed in common],
                "paired_sd_abs": paired_sd,
                "paired_sd_relative": paired_sd_relative,
                "paired_sd_abs_upper_confidence_bound": (
                    paired_sd_upper
                ),
                "paired_sd_relative_upper_confidence_bound": (
                    paired_sd_relative_upper
                ),
                "relative_denominator": denominator,
                "evaluable": bool(
                    len(common) >= _DESIGN_PILOT_MIN_PAIRS
                    and paired_sd is not None
                ),
            })

    evaluable_pairs = [
        pair for pair in pairs if pair.get("evaluable")
    ]
    observed_sd_abs = None
    observed_sd_relative = None
    absolute_values = [
        float(pair["paired_sd_abs"])
        for pair in evaluable_pairs
        if _finite_float(pair.get("paired_sd_abs")) is not None
    ]
    relative_values = [
        float(pair["paired_sd_relative"])
        for pair in evaluable_pairs
        if _finite_float(pair.get("paired_sd_relative")) is not None
    ]
    if absolute_values:
        observed_sd_abs = max(absolute_values)
    if relative_values:
        observed_sd_relative = max(relative_values)
    observed_sd_abs_upper = None
    observed_sd_relative_upper = None
    absolute_upper_values = [
        float(pair["paired_sd_abs_upper_confidence_bound"])
        for pair in evaluable_pairs
        if _finite_float(
            pair.get("paired_sd_abs_upper_confidence_bound")
        )
        is not None
    ]
    relative_upper_values = [
        float(
            pair[
                "paired_sd_relative_upper_confidence_bound"
            ]
        )
        for pair in evaluable_pairs
        if _finite_float(
            pair.get(
                "paired_sd_relative_upper_confidence_bound"
            )
        )
        is not None
    ]
    if absolute_upper_values:
        observed_sd_abs_upper = max(absolute_upper_values)
    if relative_upper_values:
        observed_sd_relative_upper = max(
            relative_upper_values
        )

    # 缺少 mode/seed 的运行无法归入处理单元，但如果它本身是
    # 数值失败或缺指标，静默丢掉就会把脱落率低估为 0。把它
    # 当作一个不可配对的保守单元计入脱落率上界，并显式告警。
    unassigned_unusable = [
        item for item in unassigned_rows
        if item.get("numerical_status") == "failed"
        or item.get("value") is None
    ]
    # 置信水平不是「越高越好」：95% 上界在 3~6 个试点配对上
    # 可能宽到无法支持任何设计。把 0.8/0.9/0.95/0.99 的结果
    # 一起落盘，研究者必须在冻结前看到设计对置信水平的敏感度，
    # 而不是只看到默认 95% 的一个数。
    ucb_sensitivity: Dict[str, Dict[str, object]] = {}
    for level in _PILOT_UCB_SENSITIVITY_LEVELS:
        level_sd_values = [
            _sd_upper_confidence_bound(
                _finite_float(pair.get("paired_sd_abs")),
                int(pair.get("n_pairs") or 0),
                level,
            )
            for pair in evaluable_pairs
        ]
        level_relative_values = []
        for pair, upper in zip(
            evaluable_pairs, level_sd_values
        ):
            denominator = _finite_float(
                pair.get("relative_denominator")
            )
            if (
                upper is not None
                and denominator is not None
                and denominator > 0.0
            ):
                level_relative_values.append(
                    upper / denominator
                )
        level_attrition_values = [
            _binomial_upper_confidence_bound(
                int(cell.get("n_failed") or 0),
                int(cell.get("n_attempted") or 0),
                level,
            )
            for cell in finalized_cells
            if cell.get("n_attempted")
        ]
        if unassigned_rows:
            level_attrition_values.append(
                _binomial_upper_confidence_bound(
                    len(unassigned_unusable),
                    len(unassigned_rows),
                    level,
                )
            )
        finite_attrition = [
            float(value)
            for value in level_attrition_values
            if _finite_float(value) is not None
        ]
        ucb_sensitivity[f"{level:g}"] = {
            "paired_sd_relative": (
                max(level_relative_values)
                if level_relative_values
                else None
            ),
            "attrition_rate": (
                max(finite_attrition)
                if finite_attrition
                else None
            ),
        }

    attempted = [
        int(cell["n_attempted"])
        for cell in finalized_cells
        if cell.get("n_attempted")
    ]
    failed = [
        int(cell["n_failed"])
        for cell in finalized_cells
        if cell.get("n_attempted")
    ]
    unassigned_attrition = (
        len(unassigned_unusable) / len(unassigned_rows)
        if unassigned_rows
        else None
    )
    observed_attrition = None
    cell_attrition = [
        float(cell["attrition_rate"])
        for cell in finalized_cells
        if _finite_float(cell.get("attrition_rate")) is not None
    ]
    if cell_attrition:
        observed_attrition = max(cell_attrition)
    if unassigned_attrition is not None:
        observed_attrition = max(
            observed_attrition or 0.0,
            unassigned_attrition,
        )
    observed_attrition_upper = None
    cell_attrition_upper = [
        float(cell["attrition_upper_confidence_bound"])
        for cell in finalized_cells
        if _finite_float(
            cell.get("attrition_upper_confidence_bound")
        )
        is not None
    ]
    if cell_attrition_upper:
        observed_attrition_upper = max(cell_attrition_upper)
    if unassigned_rows:
        unassigned_upper = _binomial_upper_confidence_bound(
            len(unassigned_unusable),
            len(unassigned_rows),
            confidence,
        )
        if unassigned_upper is not None:
            observed_attrition_upper = max(
                observed_attrition_upper or 0.0,
                unassigned_upper,
            )
    total_attempted = sum(attempted) + len(unassigned_rows)
    total_failed = sum(failed) + len(unassigned_unusable)
    overall_attrition = (
        total_failed / total_attempted if total_attempted else None
    )

    if unassigned:
        warnings.append(
            "以下试点运行缺少 mode/整数 seed，无法归入处理单元: "
            + ", ".join(unassigned[:5])
        )
    if unassigned_unusable:
        warnings.append(
            "以下无法归入处理单元的试点运行自身是数值失败/缺"
            "指标，已按保守口径计入脱落率上界: "
            + ", ".join(
                str(item.get("run"))
                for item in unassigned_unusable[:5]
            )
        )
    if duplicate_seeds:
        warnings.append(
            "试点运行存在同一处理内重复 seed，已按一个独立"
            "观测折叠: " + "; ".join(duplicate_seeds[:5])
        )
    insufficient = [
        pair for pair in pairs if not pair.get("evaluable")
    ]
    if insufficient:
        warnings.append(
            f"{len(insufficient)} 个处理单元对的共同 seed 少于 "
            f"{_DESIGN_PILOT_MIN_PAIRS} 个或 sd 不可计算，未参与"
            "试点 sd 上界"
        )
    if len(paths) < 6:
        warnings.append(
            "试点运行少于 6 个，配对 sd 的点估计很不稳定；"
            "这里只能作为下界预警，不能当作精确方差"
        )
    pilot_modes = {
        str(cell.get("mode"))
        for cell in finalized_cells
        if cell.get("mode") is not None
    }
    planned_mode_set = {
        str(mode)
        for mode in (planned_modes or [])
        if mode is not None
    }
    missing_modes = sorted(planned_mode_set - pilot_modes)
    extra_modes = sorted(pilot_modes - planned_mode_set)
    if missing_modes:
        warnings.append(
            "试点运行未覆盖计划中的模式，sd 估计不能代表这些臂: "
            + ", ".join(missing_modes)
        )
    if extra_modes:
        warnings.append(
            "试点运行包含计划外的模式（只作参考，不改变计划）: "
            + ", ".join(extra_modes)
        )
    if observed_attrition is not None and observed_attrition >= 1.0:
        warnings.append(
            "至少一个处理单元的全部试点运行都不可用："
            "该臂在当前口径下没有可估计的脱落率与方差"
        )

    sd_status = _pilot_declared_check(
        declared_sd, observed_sd_relative
    )
    attrition_status = _pilot_declared_check(
        declared_attrition, observed_attrition
    )
    sd_upper_status = _pilot_declared_check(
        declared_sd, observed_sd_relative_upper
    )
    attrition_upper_status = _pilot_declared_check(
        declared_attrition, observed_attrition_upper
    )
    if (
        sd_upper_status == "declared_below_observed"
        and sd_status != "declared_below_observed"
    ):
        warnings.append(
            "试点配对 sd 的点估计不高于预注册值，但 "
            f"{confidence:.0%} 单侧置信上界高于预注册值："
            "小样本点估计可能低估噪声。若设计要按上界准备，"
            "须在预注册 power 中把 pilot_calibration 设为 "
            "upper_confidence_bound，不能看到试点后再选"
        )
    if (
        attrition_upper_status == "declared_below_observed"
        and attrition_status != "declared_below_observed"
    ):
        warnings.append(
            "试点脱落率的点估计不高于预注册值，但 "
            f"{confidence:.0%} 单侧置信上界高于预注册值："
            "少量试点运行不足以排除更高的真实脱落率。若设计"
            "要按上界准备，须在预注册 power 中把 "
            "pilot_calibration 设为 upper_confidence_bound"
        )
    observed_over_declared = None
    if (
        observed_sd_relative is not None
        and declared_sd
        and declared_sd > 0.0
    ):
        try:
            observed_over_declared = (
                observed_sd_relative / declared_sd
            )
        except OverflowError:
            observed_over_declared = None
        if (
            observed_over_declared is not None
            and not math.isfinite(observed_over_declared)
        ):
            observed_over_declared = None
    sd_check = {
        "declared": declared_sd,
        "observed": observed_sd_relative,
        "observed_absolute": observed_sd_abs,
        "observed_upper_confidence_bound": (
            observed_sd_relative_upper
        ),
        "observed_absolute_upper_confidence_bound": (
            observed_sd_abs_upper
        ),
        "upper_confidence_level": confidence,
        "upper_confidence_bound_method": (
            "chi_square_normal_paired_differences"
        ),
        "upper_confidence_bound_status": sd_upper_status,
        "declared_below_upper_confidence_bound": bool(
            sd_upper_status == "declared_below_observed"
        ),
        "ratio_observed_over_declared": observed_over_declared,
        "status": sd_status,
        "min_pairs": _DESIGN_PILOT_MIN_PAIRS,
        "n_evaluable_pairs": len(evaluable_pairs),
        "note": (
            "observed 取各可评估处理单元对配对差值样本 sd 的"
            "最大值；相对化分母取两单元均值中较小者。"
            "observed_upper_confidence_bound 是同一最大点估计"
            "对应的单侧置信上界；小试点时它通常远大于点估计"
        ),
    }
    attrition_check = {
        "declared": declared_attrition,
        "observed": observed_attrition,
        "observed_upper_confidence_bound": (
            observed_attrition_upper
        ),
        "upper_confidence_level": confidence,
        "upper_confidence_bound_method": (
            "clopper_pearson_exact_binomial"
        ),
        "upper_confidence_bound_status": (
            attrition_upper_status
        ),
        "declared_below_upper_confidence_bound": bool(
            attrition_upper_status
            == "declared_below_observed"
        ),
        "observed_overall": overall_attrition,
        "status": attrition_status,
        "unusable": bool(
            observed_attrition is not None
            and observed_attrition >= 1.0
        ),
        "unassigned_runs": len(unassigned_rows),
        "unassigned_unusable": len(unassigned_unusable),
        "note": (
            "observed 取各处理单元脱落率的最大值；失败包括"
            "数值失败与缺失该指标，pilot 目录必须包含失败运行"
            "；无法归入处理单元且自身失败的运行按保守上界计入"
        ),
    }

    evidence: Dict[str, object] = {
        "schema": f"smelt-isr/design-pilot/v{_DESIGN_PILOT_SCHEMA_VERSION}",
        "metric": metric,
        "metric_annotation": annotation,
        "source_run_count": len(paths),
        "parsed_run_count": len(rows),
        "cell_count": len(finalized_cells),
        "pair_count": len(pairs),
        "cells": finalized_cells,
        "pairs": pairs,
        "source_runs": source_runs,
        "observed_paired_sd_abs": observed_sd_abs,
        "observed_paired_sd_relative": observed_sd_relative,
        "observed_paired_sd_abs_upper_confidence_bound": (
            observed_sd_abs_upper
        ),
        "observed_paired_sd_relative_upper_confidence_bound": (
            observed_sd_relative_upper
        ),
        "observed_attrition_rate": observed_attrition,
        "observed_attrition_rate_upper_confidence_bound": (
            observed_attrition_upper
        ),
        "observed_overall_attrition_rate": overall_attrition,
        "uncertainty": {
            "confidence_level": confidence,
            "sd_method": (
                "chi_square_upper_bound_normal_paired_differences"
            ),
            "attrition_method": (
                "clopper_pearson_exact_binomial"
            ),
            "by_confidence_level": ucb_sensitivity,
            "note": (
                "上界只描述「若试点是固定样本、配对差值近似"
                "正态」时真实 sd/脱落率的抽样不确定性；它不"
                "修正试点与正式协议的差异，也不保证新一批"
                "运行的脱落率不超过上界。置信水平越高上界越"
                "宽，必须与预注册策略一起解读"
            ),
        },
        "declared_expected_seed_sd_relative": declared_sd,
        "declared_expected_attrition_rate": declared_attrition,
        "sd_check": sd_check,
        "attrition_check": attrition_check,
        "arm_coverage": {
            "planned_modes": sorted(planned_mode_set),
            "pilot_modes": sorted(pilot_modes),
            "missing_modes": missing_modes,
            "extra_modes": extra_modes,
        },
        "warnings": warnings,
        "applied_to_power": False,
        "note": (
            "试点证据只校准先验 sd 与脱落率，不替代正式运行的"
            "实测方差/失败率；试点协议与正式协议不可比时不能"
            "把观测值直接当作目标设计的真实 sigma"
        ),
    }
    comparability = _pilot_comparability_report(
        rows,
        expected_protocol,
        evidence,
    )
    evidence["comparability"] = comparability
    warnings.extend(
        str(item)
        for item in comparability.get("warnings") or []
    )
    # 配置字段可能来自手工编辑/旧运行的 JSON（含 NaN/Inf），
    # 直接回传会让 pilot_evidence 无法按标准 JSON 写出。这里
    # 统一把非有限浮点降级为 None，保留字段本身。
    return _json_safe(evidence)



def _apply_pilot_evidence_to_power(
    power: Optional[Dict[str, object]],
    evidence: Optional[Dict[str, object]],
) -> Tuple[Optional[Dict[str, object]], bool]:
    """把试点观测到的更大 sd/脱落率上调进设计功效输入。

    若声明值低于试点观测值，设计计划不能继续沿用「按较小 sd
    算出来」的 adequate；这里用观测值替换后强制重算。替换是
    **单向**的：试点观测更小不会把声明值调小，避免用一次小
    样本试点反向放宽预注册的保守假设。
    """
    if not isinstance(power, dict) or not isinstance(evidence, dict):
        return power, False
    comparability = _as_dict(evidence.get("comparability"))
    if comparability.get("usable_for_calibration") is False:
        evidence["applied_to_power"] = False
        evidence["adjusted_power_inputs"] = None
        evidence["not_applied_reason"] = (
            "pilot_protocol_incomparable"
        )
        return power, False
    policy, confidence = _pilot_calibration_setting(power)
    adjusted = dict(power)
    adjusted["pilot_calibration"] = policy
    adjusted["pilot_confidence"] = confidence
    changed = False
    sd_adjusted = False
    attrition_adjusted = False
    sd_point = _finite_float(
        evidence.get("observed_paired_sd_relative")
    )
    sd_upper = _finite_float(
        evidence.get(
            "observed_paired_sd_relative_upper_confidence_bound"
        )
    )
    if policy == "upper_confidence_bound" and sd_upper is not None:
        sd_effective = sd_upper
        sd_basis = "upper_confidence_bound"
    else:
        sd_effective = sd_point
        sd_basis = "point"
        if (
            policy == "upper_confidence_bound"
            and sd_point is not None
        ):
            sd_basis = "point_fallback_no_upper_bound"
    declared_sd = _finite_float(
        power.get("expected_seed_sd_relative")
    )
    if (
        sd_effective is not None
        and sd_effective > 0.0
        and _pilot_declared_check(declared_sd, sd_effective)
        == "declared_below_observed"
    ):
        adjusted["pilot_declared_expected_seed_sd_relative"] = (
            power.get("expected_seed_sd_relative")
        )
        adjusted["expected_seed_sd_relative"] = sd_effective
        adjusted["pilot_adjusted_expected_seed_sd_relative"] = (
            sd_effective
        )
        sd_adjusted = True
        changed = True
    attrition_point = _finite_float(
        evidence.get("observed_attrition_rate")
    )
    attrition_upper = _finite_float(
        evidence.get(
            "observed_attrition_rate_upper_confidence_bound"
        )
    )
    calibration_unusable = None
    if (
        policy == "upper_confidence_bound"
        and attrition_upper is not None
        and attrition_upper >= 1.0
    ):
        calibration_unusable = (
            "attrition_upper_confidence_bound_reaches_one"
        )
        adjusted["pilot_calibration_unusable"] = (
            calibration_unusable
        )
    else:
        adjusted.pop("pilot_calibration_unusable", None)
    if policy == "upper_confidence_bound" and attrition_upper is not None:
        attrition_effective = attrition_upper
        attrition_basis = "upper_confidence_bound"
    else:
        attrition_effective = attrition_point
        attrition_basis = "point"
        if (
            policy == "upper_confidence_bound"
            and attrition_point is not None
        ):
            attrition_basis = (
                "point_fallback_no_upper_bound"
            )
    declared_attrition = _finite_float(
        power.get("expected_attrition_rate")
    )
    if (
        attrition_effective is not None
        and 0.0 <= attrition_effective < 1.0
        and _pilot_declared_check(
            declared_attrition, attrition_effective
        )
        == "declared_below_observed"
    ):
        adjusted["pilot_declared_expected_attrition_rate"] = (
            power.get("expected_attrition_rate")
        )
        adjusted["expected_attrition_rate"] = (
            attrition_effective
        )
        adjusted["pilot_adjusted_expected_attrition_rate"] = (
            attrition_effective
        )
        attrition_adjusted = True
        changed = True
    evidence["calibration"] = {
        "policy": policy,
        "confidence_level": confidence,
        "sd_basis": sd_basis,
        "sd_point_estimate": sd_point,
        "sd_upper_confidence_bound": sd_upper,
        "sd_effective_input": sd_effective,
        "attrition_basis": attrition_basis,
        "attrition_point_estimate": attrition_point,
        "attrition_upper_confidence_bound": attrition_upper,
        "attrition_effective_input": attrition_effective,
        "upper_bound_unavailable": bool(
            policy == "upper_confidence_bound"
            and (sd_upper is None or attrition_upper is None)
        ),
        "unusable": calibration_unusable,
        "note": (
            "point 策略使用试点点估计；upper_confidence_bound "
            "策略使用预注册置信水平的单侧上界。两者都只向"
            "上调，不会因试点更小放宽预注册声明"
        ),
    }
    evidence["applied_to_power"] = changed
    evidence["adjusted_power_inputs"] = (
        {
            "sd_adjusted": sd_adjusted,
            "attrition_adjusted": attrition_adjusted,
            "calibration_policy": policy,
            "calibration_confidence": confidence,
            "sd_basis": sd_basis,
            "attrition_basis": attrition_basis,
            "declared_expected_seed_sd_relative": power.get(
                "expected_seed_sd_relative"
            ),
            "effective_expected_seed_sd_relative": adjusted.get(
                "expected_seed_sd_relative"
            ),
            "declared_expected_attrition_rate": power.get(
                "expected_attrition_rate"
            ),
            "effective_expected_attrition_rate": adjusted.get(
                "expected_attrition_rate"
            ),
        }
        if changed
        else None
    )
    return adjusted, changed



def _pilot_gate_note(evidence: object) -> str:
    """把试点证据里「声明被观测超过」的项写成硬门说明。"""
    if not isinstance(evidence, dict):
        return ""
    comparability = _as_dict(evidence.get("comparability"))
    if comparability.get("usable_for_calibration") is False:
        keys = ", ".join(
            f"{item.get('group')}.{item.get('key')}"
            for item in (
                comparability.get("blocking_mismatches") or []
            )[:4]
            if isinstance(item, dict)
        )
        return (
            "试点运行与正式计划的代码/数据/训练/评估协议不可比"
            + (f"（{keys}）" if keys else "")
            + "；不能把试点 sd/脱落率当作正式设计的校准依据。"
        )
    parts: List[str] = []
    calibration = _as_dict(evidence.get("calibration"))
    sd_check = _as_dict(evidence.get("sd_check"))
    if (
        calibration.get("sd_basis") == "upper_confidence_bound"
        and sd_check.get("upper_confidence_bound_status")
        == "declared_below_observed"
    ):
        parts.append(
            "试点配对差值 sd 的 "
            f"{_format_finite(calibration.get('confidence_level'), '.0%')}"
            " 单侧置信上界≈"
            f"{_format_finite(calibration.get('sd_effective_input'), 'g')}"
            " 高于预注册 "
            f"{_format_finite(sd_check.get('declared'), 'g')}"
        )
    elif sd_check.get("status") == "declared_below_observed":
        parts.append(
            "试点观测的配对差值 sd≈"
            f"{_format_finite(sd_check.get('observed'), 'g')}"
            " 高于预注册 "
            f"{_format_finite(sd_check.get('declared'), 'g')}"
        )
    attrition_check = _as_dict(evidence.get("attrition_check"))
    if (
        calibration.get("attrition_basis") == (
            "upper_confidence_bound"
        )
        and attrition_check.get("upper_confidence_bound_status")
        == "declared_below_observed"
    ):
        parts.append(
            "试点脱落率的 "
            f"{_format_finite(calibration.get('confidence_level'), '.0%')}"
            " 单侧置信上界≈"
            f"{_format_finite(calibration.get('attrition_effective_input'), '.1%')}"
            " 高于预注册 "
            f"{_format_finite(attrition_check.get('declared'), '.1%')}"
        )
    elif attrition_check.get("status") == "declared_below_observed":
        parts.append(
            "试点观测的脱落率≈"
            f"{_format_finite(attrition_check.get('observed'), '.1%')}"
            " 高于预注册 "
            f"{_format_finite(attrition_check.get('declared'), '.1%')}"
        )
    if attrition_check.get("unusable"):
        parts.append("至少一个处理单元的全部试点运行都不可用")
    if calibration.get("unusable"):
        parts.append(
            "试点校准不可用："
            + str(calibration.get("unusable"))
        )
    if not parts:
        return ""
    # ``applied_to_power`` 只表示输入被上调；旧/手工 power
    # 记录可能缺少重算字段，此时必须明确写成重算失败，而不是
    # 沿用「已重算」的说法。
    applied = bool(evidence.get("applied_to_power"))
    recompute_ok = evidence.get("power_recompute_ok")
    recomputed = (
        bool(recompute_ok)
        if recompute_ok is not None
        else applied
    )
    return (
        "；".join(parts)
        + (
            "；设计计划已按试点观测值重算功效"
            if recomputed
            else (
                "；设计计划未能按试点观测值重算功效"
                if applied
                else "；试点证据未改变设计功效输入"
            )
        )
        + "。"
    )



def _training_budget_summary(
    args: argparse.Namespace,
    n_runs: int,
) -> Dict[str, object]:
    """计划阶段可算的训练预算（token 量，不是实际计时）。"""
    steps = int(getattr(args, "steps", 0) or 0)
    batch = int(getattr(args, "batch_size", 0) or 0)
    seq_len = int(getattr(args, "seq_len", 0) or 0)
    tokens_per_run = steps * batch * seq_len
    return {
        "steps": steps,
        "batch_size": batch,
        "seq_len": seq_len,
        "tokens_per_run": tokens_per_run,
        "runs": int(n_runs),
        "total_tokens": tokens_per_run * int(n_runs),
        "note": (
            "解析预算：steps × batch_size × seq_len；不含评估、"
            "重算与优化器开销，也不是墙钟时间估计"
        ),
    }



def _design_plan_id(payload: Dict[str, object]) -> str:
    """设计计划内容的规范化指纹（排除 plan_id 与创建时间）。

    写入与核对必须使用同一套规范化规则；否则一个被改写、但
    plan_id 字段没有同步更新的计划文件会仅凭文件 SHA 通过核对。
    ``plan_id`` / ``created_utc`` 不参与指纹：前者是结果，后者
    只是写入时间，重复生成同一计划时应得到同一个 id。
    """
    core = {
        key: value
        for key, value in payload.items()
        if key not in ("plan_id", "created_utc")
    }
    return _sha256_text(
        json.dumps(
            _json_safe(core),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
