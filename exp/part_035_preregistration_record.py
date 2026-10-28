from __future__ import annotations



def preregistration_data_error(
    args: argparse.Namespace,
    payload: Dict[str, object],
    dataset: object,
) -> Optional[str]:
    """
    核对预注册的数据协议与本次实际加载的数据是否一致。

    只核对 ``dataset`` 上已构造出的真实指纹/比例，不接受
    「文件名相同」「路径相同」这类弱证据；清洗开关、显式
    分隔符与清单指纹也逐项核对，因为它们都会改变实际参与
    训练或评估的字节。
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        return "预注册文件缺少 data 对象"

    actual_sha = getattr(dataset, "data_sha256", None)
    if (
        not isinstance(actual_sha, str)
        or actual_sha.lower() != str(data["sha256"]).lower()
    ):
        return (
            f"预注册 data.sha256={data['sha256']} 与实际语料 "
            f"SHA-256={actual_sha} 不一致：冻结后才能开始确证"
            "运行，换数据必须重新预注册"
        )

    declared_policy = str(data["split_policy"])
    current_policy = str(
        getattr(args, "split_policy", "contiguous")
    )
    if declared_policy != current_policy:
        return (
            f"预注册 data.split_policy={declared_policy} 与当前 "
            f"--split-policy={current_policy} 不一致：切分口径"
            "决定了「前缀」能否被解释为时间外推"
        )

    for key in ("val_fraction", "test_fraction"):
        declared = float(data[key])  # type: ignore[arg-type]
        actual = getattr(dataset, key, None)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                declared,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return (
                f"预注册 data.{key}={declared:g} 与实际切分 "
                f"{key}={actual!r} 不一致"
            )

    declared_group = data.get(
        "group_separator", _PREREGISTRATION_MISSING
    )
    current_group = getattr(args, "group_separator", None)
    # 未声明 group_separator 等价于声明「不使用群组切分」：
    # 否则预注册后再加 --group-separator 会改变切分吸附位置与
    # 实际 train/val/test 字节，绕过数据协议核对。
    if (
        declared_group is _PREREGISTRATION_MISSING
        and current_group is not None
    ):
        return (
            "预注册 data 未声明 group_separator，但当前使用了 "
            f"--group-separator={current_group!r}：群组边界会"
            "改变切分吸附位置与实际训练字节，必须在预注册中固定"
        )
    if declared_group is not _PREREGISTRATION_MISSING:
        try:
            declared_bytes = (
                None
                if declared_group is None
                else parse_group_separator(str(declared_group))
            )
            current_bytes = (
                None
                if current_group is None
                else parse_group_separator(str(current_group))
            )
        except ValueError as error:
            return f"预注册/命令行群组分隔符无效: {error}"
        if declared_bytes != current_bytes:
            return (
                "预注册 data.group_separator 与当前 "
                "--group-separator 不一致：群组边界会改变"
                "切分吸附位置与实际训练字节"
            )

    cleaning = data.get("cleaning")
    if cleaning is not None and not isinstance(cleaning, dict):
        return "data.cleaning 必须是对象"
    # 未声明的清洗开关按默认 False 解释，而不是「不核对」：
    # 只声明一个 false 开关、却启用其它清洗项，会改变实际参与
    # 训练/去重的字节范围；预注册文件不能在看到结果后通过
    # 漏写字段绕过核对。
    declared_cleaning = (
        cleaning if isinstance(cleaning, dict) else {}
    )
    declared_alias_values = {
        bool(declared_cleaning[key])
        for key in _CLEAN_AGAINST_TEST_ALIASES
        if key in declared_cleaning
    }
    if len(declared_alias_values) > 1:
        return (
            "预注册 data.cleaning 的两个同义开关取值相反："
            "clean_train_against_test 与 "
            "allow_eval_content_for_cleaning"
        )
    declared_effective = (
        next(iter(declared_alias_values))
        if declared_alias_values
        else False
    )
    actual_effective = _clean_against_test_effective(args)
    if actual_effective != declared_effective:
        return (
            "预注册 data.cleaning."
            "clean_train_against_test/"
            "allow_eval_content_for_cleaning="
            f"{declared_effective} 与当前 {actual_effective} "
            "不一致：清洗开关会改变实际参与训练/去重的"
            "数据，不能在看到结果后再改"
        )
    for key in _PREREGISTRATION_CLEANING_BOOL_KEYS:
        if key in _CLEAN_AGAINST_TEST_ALIASES:
            continue
        expected = bool(declared_cleaning.get(key, False))
        actual = bool(getattr(args, key, False))
        if actual != expected:
            return (
                f"预注册 data.cleaning.{key}={expected} "
                f"与当前 {actual} 不一致：清洗开关会改变"
                "实际参与训练/去重的数据，不能在看到结果后"
                "再改；未声明的清洗开关按 false 处理"
            )

    numeric_defaults = {
        "near_duplicate_threshold": _NEAR_DUP_DEFAULT_THRESHOLD,
        "near_duplicate_shingle_bytes": (
            _NEAR_DUP_DEFAULT_SHINGLE_BYTES
        ),
    }
    for key, default in numeric_defaults.items():
        expected = declared_cleaning.get(key, default)
        actual = getattr(args, key, default)
        try:
            same = float(expected) == float(actual)
        except (TypeError, ValueError, OverflowError):
            same = False
        if not same:
            return (
                f"预注册 data.cleaning.{key}={expected!r} "
                f"与当前 {actual!r} 不一致：近重复阈值/"
                "shingle 长度会改变删除哪些训练群组，"
                "不能在看到结果后再改；未声明时按默认值"
                f" {default!r} 核对"
            )

    for key, attr in (
        ("entity_manifest_sha256", "entity_manifest"),
        ("time_order_manifest_sha256", "time_order_manifest"),
    ):
        expected = data.get(key)
        if expected is None:
            continue
        path = getattr(args, attr, None)
        if not path:
            return (
                f"预注册声明了 data.{key}，但当前没有 --"
                f"{attr.replace('_', '-')}：无法核对清单内容"
            )
        actual = _preregistration_file_sha256(path)
        if actual is None:
            return f"无法读取 --{attr.replace('_', '-')}={path}"
        if actual != str(expected).lower():
            return (
                f"--{attr.replace('_', '-')}={path} 的 SHA-256 "
                f"{actual} 与预注册 data.{key}={expected} 不一致"
            )

    return None



def preregistration_record(
    payload: Dict[str, object],
    source: object,
    file_sha256: str,
    applied_fields: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """生成写入 config.json 的预注册核对记录。"""
    power_record: Optional[Dict[str, object]] = None
    if isinstance(payload.get("power"), dict):
        declared_power = payload["power"]
        power_record = preregistration_power_plan(
            declared_power,  # type: ignore[arg-type]
            float(payload["min_effect"]),  # type: ignore[arg-type]
            len(payload["seeds"]),  # type: ignore[arg-type]
            len(payload["comparisons"]),  # type: ignore[arg-type]
            equivalence_margin=payload.get(
                "equivalence_margin"
            ),
        )
        # 重算只覆盖功效数值；试点校准策略和置信水平也是
        # 预注册声明，必须随记录落盘，否则后续设计计划会退回
        # 默认的点估计，悄悄丢掉冻结的策略。
        for key in ("pilot_calibration", "pilot_confidence"):
            if key in declared_power:
                power_record[key] = declared_power[key]
    return {
        "source": str(source),
        "file_sha256": file_sha256,
        "study_id": payload.get("study_id"),
        "frozen_commit": payload.get("frozen_commit"),
        "frozen_script_sha256": payload.get(
            "frozen_script_sha256"
        ),
        "primary_metric": payload.get("primary_metric"),
        "common_step": payload.get("common_step"),
        "direction": payload.get("direction"),
        "min_effect": payload.get("min_effect"),
        "baseline_mode": payload.get("baseline_mode"),
        "comparisons": payload.get("comparisons"),
        "seeds": payload.get("seeds"),
        "budget_match": payload.get("budget_match"),
        # 先验功效与运行顺序是训练前设计的一部分：随每个运行
        # 落盘，分析/归档时能核对「这个种子数当时是否被认为
        # 足够」以及「执行顺序是否按预注册的区组随机化」。
        "power": power_record,
        "run_order": payload.get("run_order"),
        "data": payload.get("data"),
        "applied_fields": list(applied_fields),
        "run_mode": getattr(args, "mode", None),
        "run_seed": getattr(args, "seed", None),
        "run_seeds": getattr(args, "seeds", None),
        "data_sha256": getattr(args, "data_sha256", None),
        "data_split_sha256": getattr(
            args, "data_split_sha256", None
        ),
        "data_split_policy": getattr(
            args, "split_policy", "contiguous"
        ),
        "data_verified": bool(
            getattr(args, "data_sha256", None) is not None
        ),
    }



def hypothesis_preregistration_error(
    args: argparse.Namespace,
) -> Optional[str]:
    """
    假设预注册检查：在训练开始前把「预期方向、最小有意义
    效应、主判定指标」写死，避免看完结果再挑指标、解释
    方向。

    规则：
    - ``--hypothesis-direction`` / ``--hypothesis-min-effect``
      必须与 ``--primary-metric`` 一起使用；
    - ``--hypothesis-min-effect`` 必须与方向一起使用，且为
      有限正数（相对对照臂均值的比例）；
    - ``--primary-metric`` 必须是评估管线真正计算的对比指标
      （COMPARISON_METRIC_KEYS），且在 ``--hypothesis`` 文本
      里被点名（假设与判定口径必须对齐）；
    - 统计基线不参与方向/最小效应预注册，但允许用
      ``--primary-metric`` 指定 n-gram 的 val 调参口径；
    - 只读对比 / 结构自检不使用这套预注册。

    返回错误消息（应交给 parser.error 报告）或 None。
    """
    direction = getattr(args, "hypothesis_direction", None)
    min_effect = getattr(args, "hypothesis_min_effect", None)
    primary_metric = getattr(args, "primary_metric", None)
    baseline = getattr(args, "hypothesis_baseline", None)

    if (
        primary_metric is None
        and direction is None
        and min_effect is None
        and baseline is None
    ):
        return None

    if (
        args.compare is not None
        or getattr(args, "final_test_comparison", None) is not None
        or args.sanity_only
    ):
        return (
            "--primary-metric / --hypothesis-direction / "
            "--hypothesis-min-effect / --hypothesis-baseline "
            "只适用于训练入口：--compare / "
            "--final-test-comparison 只读分析与 --sanity-only "
            "结构自检不训练神经模型"
        )

    if args.statistical_baseline:
        if direction is not None or min_effect is not None:
            return (
                "--hypothesis-direction / --hypothesis-min-effect "
                "不适用于 --statistical-baseline：统计基线没有"
                "训练臂，方向与最小效应无法由该入口验证；"
                "可只用 --primary-metric 指定 n-gram 的 val "
                "调参口径"
            )
        if baseline is not None:
            return (
                "--hypothesis-baseline 不适用于 "
                "--statistical-baseline：统计基线本身就是被"
                "比较的对象，没有可预注册的对照臂"
            )
        # 统计基线的调参口径必须落在 n-gram 真正会计算的
        # 指标集合内：路由诊断/等计算量 loss 只存在于神经
        # 模型评估中，放行后会被 resolve_ngram_tuning_metric
        # 静默回退到 val_loss，预注册的主指标与实际调参口径
        # 不一致（形同未预注册）。
        if primary_metric not in _NGRAM_TUNING_METRIC_KEYS:
            return (
                f"--primary-metric={primary_metric} 不是 "
                "n-gram 统计基线可计算的调参指标；可选："
                + ", ".join(_NGRAM_TUNING_METRIC_KEYS)
            )
        return None

    if primary_metric is None:
        if direction is None and min_effect is None:
            # 只预注册对照基线（未声明主指标）是合法的：
            # 它约束的是对比协议而非判定统计量。
            return None
        return (
            "--hypothesis-direction / --hypothesis-min-effect "
            "必须与 --primary-metric 一起使用：方向与最小效应"
            "需要一个明确的判定指标才有意义"
        )

    if primary_metric not in COMPARISON_METRIC_KEYS:
        return (
            f"--primary-metric={primary_metric} 不是评估管线"
            "计算的对比指标；可选："
            + ", ".join(COMPARISON_METRIC_KEYS)
        )

    if min_effect is not None and direction is None:
        return (
            "--hypothesis-min-effect 必须与 "
            "--hypothesis-direction 一起使用：只声明最小效应"
            "而不声明方向无法判定「是否达到」"
        )

    min_effect_number = _finite_float(min_effect)
    if min_effect is not None and (
        min_effect_number is None or min_effect_number <= 0.0
    ):
        return (
            "--hypothesis-min-effect 必须是有限正数"
            "（相对对照臂均值的比例，如 0.02 = 2%）"
        )

    hypothesis_text = args.hypothesis or ""
    mentions = hypothesis_metric_mentions(hypothesis_text)
    aligned = any(
        primary_metric in _hypothesis_supported_metrics(name)
        for name in mentions
    )
    if not aligned:
        return (
            f"--primary-metric={primary_metric} 没有在 "
            "--hypothesis 文本中被点名：假设必须写明用哪个"
            "指标判定成败（如「验证损失至少改善 2%」或"
            "「macro F1 提升」），否则预注册的主指标与假设"
            "无法对齐"
        )

    averaging_mismatch = hypothesis_averaging_qualifier_mismatch(
        primary_metric,
        hypothesis_text,
    )
    if averaging_mismatch is not None:
        return averaging_mismatch

    return None



def hypothesis_registry(
    rows: List[Dict[str, object]],
    metric: Optional[str] = None,
) -> Dict[str, object]:
    """
    每个实验单元登记的科学假设汇总。

    全新训练必须通过 --hypothesis 说明「预期什么改动带来什么
    效果、依据什么指标判定成败」（入口已强制），此处把各运行
    登记的假设按模式汇总，并把缺失假设的运行单独列出——没有
    假设记录的对比与「排列组合超参数」无法区分。统计基线
    （ngram）不受假设纪律约束。

    同时核对假设是否写明了判定指标：没写指标的假设无法与
    评估口径对齐（「效果更好」不可检验），运行被单独点名；
    写明了但当前评估管线不可判定的指标（如路由类指标而运行
    没有 routing_analysis 记录）同样单独点名，避免假性对齐。
    F1 / precision / recall 未注明 macro/micro/weighted 时
    同样点名（三种平均口径可能给出不同结论）；假设写明的
    口径与预注册主指标冲突时也单独点名。

    ``metric`` 为当前对比使用的判定指标（--compare-metric）：
    运行预注册了 ``--primary-metric`` 但与当前指标不一致时，
    「战胜基线」的结论可能来自事后换指标，单独列出
    （metric shopping 守卫）。旧运行没有预注册记录，无法
    核对，只提示不判罚。
    """
    cells: Dict[str, Dict[str, object]] = {}
    order: List[str] = []
    missing_runs: List[str] = []
    skipped_runs: List[str] = []
    hypotheses_without_metric: List[str] = []
    hypotheses_with_unverifiable_metric: List[str] = []
    unverifiable_details: List[str] = []
    ambiguous_averaging_runs: List[str] = []
    ambiguous_averaging_details: List[str] = []
    averaging_mismatch_runs: List[str] = []
    averaging_mismatch_details: List[str] = []
    preregistered_runs: List[str] = []
    preregistration_missing_runs: List[str] = []
    metric_mismatch_runs: List[str] = []
    metric_mismatch_details: List[str] = []

    for row in rows:
        mode = str(row.get("mode"))
        entry = cells.get(mode)
        if entry is None:
            entry = {
                "mode": mode,
                "n_runs": 0,
                "hypotheses": [],
                "hypotheses_metric_mentions": [],
                "hypotheses_without_metric_runs": [],
                "hypotheses_with_unverifiable_metric_runs": [],
                "hypotheses_with_ambiguous_averaging_runs": [],
                "hypotheses_with_averaging_mismatch_runs": [],
                "preregistered_primary_metrics": [],
                "preregistration_missing_runs": [],
                "metric_mismatch_runs": [],
                "research_questions": [],
                "missing_hypothesis_runs": [],
                "hypothesis_check_skipped_runs": [],
            }
            cells[mode] = entry
            order.append(mode)

        entry["n_runs"] += 1

        hypothesis_text = str(row.get("hypothesis") or "").strip()
        rq_text = str(row.get("research_question") or "").strip()

        if rq_text and rq_text not in entry["research_questions"]:
            entry["research_questions"].append(rq_text)

        if hypothesis_text:
            run_name = str(row.get("run"))
            mentions = hypothesis_metric_mentions(hypothesis_text)
            if hypothesis_text not in entry["hypotheses"]:
                entry["hypotheses"].append(hypothesis_text)
            known = {
                item["hypothesis"]: item
                for item in entry["hypotheses_metric_mentions"]
            }
            if hypothesis_text not in known:
                entry["hypotheses_metric_mentions"].append({
                    "hypothesis": hypothesis_text,
                    "metric_mentions": mentions,
                })

            # 多分类平均口径歧义：F1 / precision / recall 不注明
            # macro / micro / weighted 时三种口径可能给出不同
            # 结论，假设无法与评估口径对齐。
            ambiguous = hypothesis_ambiguous_averaging_metrics(
                hypothesis_text,
                mentions,
            )
            if ambiguous:
                entry[
                    "hypotheses_with_ambiguous_averaging_runs"
                ].append(run_name)
                ambiguous_averaging_runs.append(run_name)
                ambiguous_averaging_details.append(
                    f"{run_name}（"
                    + ", ".join(ambiguous)
                    + "）"
                )

            # 预注册的主指标：训练前声明、训练后不可改；
            # 与当前分析指标不一致时可能是事后换指标。
            preregistered = row.get("primary_metric")
            if isinstance(preregistered, str) and preregistered:
                preregistered_runs.append(run_name)
                if (
                    preregistered
                    not in entry["preregistered_primary_metrics"]
                ):
                    entry[
                        "preregistered_primary_metrics"
                    ].append(preregistered)
                if metric is not None and preregistered != metric:
                    entry["metric_mismatch_runs"].append(
                        run_name
                    )
                    metric_mismatch_runs.append(run_name)
                    metric_mismatch_details.append(
                        f"{run_name}（预注册 {preregistered}，"
                        f"当前判定 {metric}）"
                    )
                averaging_mismatch = (
                    hypothesis_averaging_qualifier_mismatch(
                        preregistered,
                        hypothesis_text,
                    )
                )
                if averaging_mismatch is not None:
                    entry[
                        "hypotheses_with_averaging_mismatch_runs"
                    ].append(run_name)
                    averaging_mismatch_runs.append(run_name)
                    averaging_mismatch_details.append(
                        f"{run_name}（{averaging_mismatch}）"
                    )
            elif mode != "ngram":
                entry["preregistration_missing_runs"].append(
                    run_name
                )
                preregistration_missing_runs.append(run_name)

            if not mentions:
                entry["hypotheses_without_metric_runs"].append(
                    run_name
                )
                hypotheses_without_metric.append(run_name)
                continue
            # 提到的指标必须在当前评估管线的可判定集合内：
            # 路由类指标在运行没有 routing_analysis 记录时
            # 无法判定，不能算「假设已写明判定指标」。
            unverifiable = hypothesis_metric_unverifiable(
                mentions,
                routing_available=bool(
                    row.get("routing_available")
                ),
            )
            if unverifiable:
                entry[
                    "hypotheses_with_unverifiable_metric_runs"
                ].append(run_name)
                hypotheses_with_unverifiable_metric.append(
                    run_name
                )
                unverifiable_details.append(
                    f"{run_name}（"
                    + ", ".join(unverifiable)
                    + "）"
                )
            continue

        if mode == "ngram":
            continue

        run_name = str(row.get("run"))
        entry["missing_hypothesis_runs"].append(run_name)
        missing_runs.append(run_name)
        if row.get("hypothesis_check_skipped"):
            entry["hypothesis_check_skipped_runs"].append(
                run_name
            )
            skipped_runs.append(run_name)

    advisories: List[str] = []
    if missing_runs:
        advisories.append(
            "以下运行没有登记科学假设（旧运行或显式跳过检查）："
            + "; ".join(missing_runs)
            + "。缺假设的对比无法与排列组合超参数区分，"
            "结论只能作为探索性结果"
        )
    if hypotheses_without_metric:
        advisories.append(
            "以下运行的假设没有写明判定指标（如验证损失 / "
            "macro F1 / PR-AUC / 召回率），无法与评估口径对齐："
            + "; ".join(hypotheses_without_metric)
            + "。可检验的假设应写出「依据什么指标、预期"
            "改善多少」"
        )
    if hypotheses_with_unverifiable_metric:
        advisories.append(
            "以下运行的假设提到的判定指标当前不可判定"
            "（路由诊断类指标需要该运行有 routing_analysis "
            "记录；dense 基线、--skip-routing-analysis 或旧"
            "运行的假设无法据此核对）："
            + "; ".join(unverifiable_details)
            + "。请改用当前评估管线真正计算的指标，或先补上"
            "对应的诊断"
        )
    if ambiguous_averaging_runs:
        advisories.append(
            "以下运行的假设提到 F1 / precision / recall "
            "但未注明平均口径（macro / micro / weighted "
            "三者结果可能差异很大，同一个实验可能一种口径"
            "显著、另一种不显著）："
            + "; ".join(ambiguous_averaging_details)
            + "。请在假设里写明判定口径，并与 --primary-metric "
            "/ --compare-metric 保持一致"
        )
    if averaging_mismatch_runs:
        advisories.append(
            "以下运行的假设写明的平均口径与预注册主指标不一致"
            "（同一个实验在不同平均口径下可能一个显著、另一个"
            "不显著，假设与判定口径必须一致）："
            + "; ".join(averaging_mismatch_details)
            + "。请把假设改为与 --primary-metric 相同的口径，"
            "或改用假设所写口径的指标"
        )
    if metric_mismatch_runs:
        advisories.append(
            f"以下运行预注册的主指标与当前判定指标 {metric} "
            "不一致（预注册主指标应在看到结果之前声明；"
            "事后改判指标属于指标选择偏差，结论只能作为"
            "探索性结果）："
            + "; ".join(metric_mismatch_details)
        )
    if preregistration_missing_runs:
        advisories.append(
            "以下运行没有预注册主指标（--primary-metric，"
            "旧运行或未声明）：无法排除事后从多个指标里挑"
            "显著的那个（指标选择多重比较）；建议新实验在"
            "训练前用 --primary-metric 声明判定指标"
        )
    if skipped_runs:
        advisories.append(
            "以下运行显式跳过了假设检查"
            "（--skip-hypothesis-check）："
            + "; ".join(skipped_runs)
        )

    return {
        "cells": [cells[mode] for mode in order],
        "metric": metric,
        "missing_hypothesis_runs": missing_runs,
        "hypotheses_without_metric_runs": hypotheses_without_metric,
        "hypotheses_with_unverifiable_metric_runs": (
            hypotheses_with_unverifiable_metric
        ),
        "unverifiable_metric_details": unverifiable_details,
        "hypotheses_with_ambiguous_averaging_runs": (
            ambiguous_averaging_runs
        ),
        "ambiguous_averaging_details": (
            ambiguous_averaging_details
        ),
        "hypotheses_with_averaging_mismatch_runs": (
            averaging_mismatch_runs
        ),
        "averaging_mismatch_details": (
            averaging_mismatch_details
        ),
        "preregistered_runs": preregistered_runs,
        "preregistration_missing_runs": (
            preregistration_missing_runs
        ),
        "metric_mismatch_runs": metric_mismatch_runs,
        "metric_mismatch_details": metric_mismatch_details,
        "hypothesis_check_skipped_runs": skipped_runs,
        "advisories": advisories,
    }



def hypothesis_alignment(
    rows: List[Dict[str, object]],
    aggregates: List[Dict[str, object]],
    contrasts: List[Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    把观测到的对比结果与预注册假设（方向 + 最小效应）核对。

    只有训练前用 --primary-metric / --hypothesis-direction /
    --hypothesis-min-effect 声明过的运行才参与核对；未预注册
    的运行只登记「无法核对」。核对对象是**变体实验单元**相对
    对照单元的差值（delta = 变体均值 - 对照均值）：

    - ``refuted``：观测方向与预注册方向显著相反（Holm 校正
      后显著）——假设被数据推翻，不能靠事后改写解释；
    - ``direction_mismatch_not_significant``：方向相反但不
      显著，结论只能是「不确定」（样本不足）；
    - ``below_declared_min_effect``：方向一致但幅度小于预注册
      的最小有意义效应，不能主张支持假设（哪怕统计显著）；
    - ``min_effect_undefined``：声明了相对最小效应，但对照
      均值为零/近零或缺失，比例阈值没有可解释的尺度，结果
      不可判定（不能静默算作达到最小效应）；
    - ``supported``：方向一致、达到最小效应（若有声明）且
      校正后显著；
    - ``inconclusive_underpowered``：方向一致、达到最小效应
      但不显著，且当前设计（se/df 与家族校正后的 alpha）对
      预注册最小效应的功效低于目标（80%）：不显著是样本量
      不足的预期结果，不能解读为「无效应」，也不能解读为
      「等价」；
    - ``not_significant``／``no_effect_observed``：方向一致
      但不显著（或差值为 0），且功效足够或无法评估，证据
      不足；
    - ``conflicting_preregistration``：同一实验单元的运行给出
      了互相矛盾的方向，核对无法进行。
    """
    declarations: Dict[str, Dict[str, object]] = {}
    for row in rows:
        direction = row.get("hypothesis_direction")
        if direction not in ("decrease", "increase"):
            continue
        declarations[str(row.get("run"))] = {
            "run": str(row.get("run")),
            "primary_metric": row.get("primary_metric"),
            "direction": direction,
            "min_effect": row.get("hypothesis_min_effect"),
        }

    cell_by_label = {
        str(aggregate.get("label")): aggregate
        for aggregate in aggregates
    }

    checks: List[Dict[str, object]] = []
    advisories: List[str] = []
    refuted: List[str] = []
    supported: List[str] = []
    below_min_effect: List[str] = []
    min_effect_undefined: List[str] = []
    inconclusive_underpowered: List[str] = []
    min_effect_unverified: List[str] = []

    for contrast in contrasts:
        if contrast.get("confounded") or contrast.get(
            "delta"
        ) is None:
            continue
        cell = cell_by_label.get(str(contrast.get("variant")))
        if not isinstance(cell, dict):
            continue

        cell_declarations = [
            declarations[run]
            for run in (cell.get("runs") or [])
            if run in declarations
        ]
        metric_declarations = [
            declaration
            for declaration in cell_declarations
            if declaration.get("primary_metric") == metric
        ]
        if not metric_declarations:
            continue

        declared_min_effects = [
            number
            for declaration in metric_declarations
            for number in [
                _finite_float(declaration.get("min_effect"))
            ]
            if number is not None
        ]
        directions = {
            str(declaration["direction"])
            for declaration in metric_declarations
        }
        entry: Dict[str, object] = {
            "baseline": contrast.get("baseline"),
            "variant": contrast.get("variant"),
            "factor": contrast.get("factor"),
            "metric": metric,
            "delta": contrast.get("delta"),
            "significant": bool(contrast.get("significant")),
            "p_value_holm": contrast.get("p_value_holm"),
            "expected_direction": None,
            # 同一实验单元的种子可能分批启动；若各自声明的
            # 最小效应不同，取最严格（最大）阈值，避免用某个
            # 宽松声明把整个单元判成 supported。
            "min_effect": (
                max(declared_min_effects)
                if declared_min_effects
                else None
            ),
            "declared_by": [
                declaration["run"]
                for declaration in metric_declarations
            ],
            "status": None,
        }

        if len(directions) > 1:
            entry["expected_direction"] = sorted(directions)
            entry["status"] = "conflicting_preregistration"
            checks.append(entry)
            advisories.append(
                f"{contrast.get('variant')} 的预注册方向互相"
                "矛盾（" + ", ".join(sorted(directions))
                + "）：同一实验单元的方案应有统一的预期方向，"
                "核对无法进行"
            )
            continue

        direction = next(iter(directions))
        entry["expected_direction"] = direction
        delta = float(contrast["delta"])
        baseline_mean = contrast.get("mean_baseline")
        min_effect = entry["min_effect"]
        threshold_abs = None
        threshold_defined = False
        if (
            isinstance(min_effect, float)
            and math.isfinite(min_effect)
            and isinstance(baseline_mean, (int, float))
            and not isinstance(baseline_mean, bool)
        ):
            baseline_abs = abs(float(baseline_mean))
            if math.isfinite(baseline_abs):
                candidate = min_effect * baseline_abs
                if (
                    math.isfinite(candidate)
                    and candidate > _HYPOTHESIS_MIN_EFFECT_ABS_FLOOR
                ):
                    threshold_abs = candidate
                    threshold_defined = True
        entry["min_effect_abs"] = threshold_abs
        entry["baseline_mean"] = baseline_mean

        # 最小有意义效应不能只靠「点估计超过门槛 + 相对 0
        # 显著」来主张：显著只拒绝「效应为 0」，不拒绝「效应
        # 小于等于门槛」。这里做方向调整后的门槛检验
        # H0: 改善量 <= threshold_abs，并要求其单侧 p 值显著；
        # 有区间时同时要求方向调整后的 95% 置信下界超过门槛。
        directional_delta = (
            -delta if direction == "decrease" else delta
        )
        ci_low = contrast.get("ci95_low")
        ci_high = contrast.get("ci95_high")
        directional_ci_low = None
        if (
            isinstance(ci_low, (int, float))
            and not isinstance(ci_low, bool)
            and isinstance(ci_high, (int, float))
            and not isinstance(ci_high, bool)
        ):
            directional_ci_low = (
                -float(ci_high)
                if direction == "decrease"
                else float(ci_low)
            )
        entry["directional_effect"] = directional_delta
        entry["directional_ci95_low"] = directional_ci_low
        min_effect_p_value = None
        min_effect_verified = False
        if threshold_abs is not None:
            se = contrast.get("se")
            df = contrast.get("df")
            alpha_family = contrast.get(
                "alpha_bonferroni", _SIGNIFICANCE_LEVEL
            )
            if (
                isinstance(se, (int, float))
                and not isinstance(se, bool)
                and float(se) > 0.0
                and isinstance(df, (int, float))
                and not isinstance(df, bool)
            ):
                t_stat = (
                    directional_delta - threshold_abs
                ) / float(se)
                p_value_margin = _t_cdf(
                    t_stat, float(df)
                )
                if p_value_margin is not None:
                    min_effect_p_value = (
                        1.0 - float(p_value_margin)
                    )
                    min_effect_verified = bool(
                        min_effect_p_value
                        <= float(alpha_family)
                    )
            if (
                not min_effect_verified
                and directional_ci_low is not None
            ):
                min_effect_verified = bool(
                    directional_ci_low > threshold_abs
                )
            entry["min_effect_p_value"] = min_effect_p_value
            entry["min_effect_test"] = {
                "h0": "directional_effect <= min_effect_abs",
                "alpha": float(
                    contrast.get(
                        "alpha_bonferroni",
                        _SIGNIFICANCE_LEVEL,
                    )
                ),
                "p_value": min_effect_p_value,
                "ci95_low_directional": directional_ci_low,
                "verified": min_effect_verified,
                "criterion": (
                    "单侧门槛检验显著，或方向调整后的 95% "
                    "置信下界超过门槛；仅点估计超门槛不算"
                ),
            }
        entry["min_effect_verified"] = min_effect_verified

        conflict = direction == "decrease" and delta > 0.0 or (
            direction == "increase" and delta < 0.0
        )
        if delta == 0.0:
            status = "no_effect_observed"
        elif conflict:
            status = (
                "refuted"
                if entry["significant"]
                else "direction_mismatch_not_significant"
            )
        elif isinstance(min_effect, float) and not threshold_defined:
            # 有最小效应声明却没有可解释的对照尺度：既不能判
            # supported，也不能把「无法核对」伪装成不显著。
            status = "min_effect_undefined"
        elif (
            threshold_abs is not None
            and abs(delta) < threshold_abs
        ):
            status = "below_declared_min_effect"
        elif entry["significant"] and threshold_abs is None:
            status = "supported"
        elif entry["significant"] and min_effect_verified:
            status = "supported"
        elif entry["significant"]:
            # 点估计达到门槛且显著，但门槛本身没被统计检验
            # 支持：可能是真效应，也可能是噪声刚好把点估计
            # 推过门槛，不能声称「达到最小有意义改善」。
            status = (
                "min_effect_unverifiable"
                if entry.get("min_effect_test", {}).get(
                    "p_value"
                )
                is None
                and directional_ci_low is None
                else "min_effect_not_verified"
            )
        else:
            status = "not_significant"
            # 功效核对：主指标预注册了最小效应时，「不显著」
            # 只有在设计对最小效应有足够功效时才有信息量。
            # 用家族校正后的 alpha（判定口径与 significant
            # 一致）在预注册最小效应处计算功效；se/df 不可用
            # 时退回同口径 MDE 与阈值比较。功效不足则判为
            # inconclusive_underpowered，而不是 not_significant。
            if (
                threshold_abs is not None
                and threshold_abs > 0.0
            ):
                power_known = False
                se = contrast.get("se")
                df = contrast.get("df")
                alpha_family = contrast.get(
                    "alpha_bonferroni"
                )
                if not isinstance(alpha_family, (int, float)):
                    alpha_family = _SIGNIFICANCE_LEVEL
                if (
                    isinstance(se, (int, float))
                    and not isinstance(se, bool)
                    and isinstance(df, (int, float))
                    and not isinstance(df, bool)
                ):
                    power = _t_power(
                        threshold_abs,
                        float(se),
                        float(df),
                        alpha=float(alpha_family),
                    )
                    if power is not None:
                        power_known = True
                        entry["power_at_min_effect"] = power
                        entry["power_at_min_effect_alpha"] = (
                            float(alpha_family)
                        )
                        if power < _TARGET_POWER:
                            status = (
                                "inconclusive_underpowered"
                            )
                if not power_known:
                    mde = contrast.get("mde_80_bonferroni")
                    if not isinstance(mde, (int, float)):
                        mde = contrast.get("mde_80")
                    if (
                        isinstance(mde, (int, float))
                        and not isinstance(mde, bool)
                        and math.isfinite(float(mde))
                    ):
                        entry["mde_80_vs_min_effect"] = float(
                            mde
                        )
                        if float(mde) > threshold_abs:
                            status = (
                                "inconclusive_underpowered"
                            )
        entry["status"] = status
        checks.append(entry)

        label = (
            f"{contrast.get('baseline')}→"
            f"{contrast.get('variant')}"
        )
        if status == "refuted":
            refuted.append(
                f"{label}（Δ={delta:+.4f}，预期 "
                f"{direction}，Holm p="
                f"{entry['p_value_holm']}）"
            )
        elif status == "supported":
            supported.append(
                f"{label}（Δ={delta:+.4f}，预期 "
                f"{direction}"
                + (
                    f"，最小效应 {min_effect:.2%}"
                    if isinstance(min_effect, float)
                    else ""
                )
                + "）"
            )
        elif status in {
            "min_effect_not_verified",
            "min_effect_unverifiable",
        }:
            test_info = entry.get("min_effect_test") or {}
            min_effect_unverified.append(
                f"{label}（点估计 Δ={delta:+.4f} 达到门槛，但"
                "门槛本身未被统计检验支持："
                f"p={test_info.get('p_value')}，"
                f"方向调整 CI 下界="
                f"{test_info.get('ci95_low_directional')}；"
                "不能声称达到最小有意义改善）"
            )
        elif status == "below_declared_min_effect":
            below_min_effect.append(
                f"{label}（Δ={delta:+.4f}，低于预注册最小效应 "
                f"{min_effect:.2%}×对照均值="
                f"{threshold_abs:.4f}）"
            )
        elif status == "min_effect_undefined":
            baseline_text = (
                f"{float(baseline_mean):.6g}"
                if isinstance(baseline_mean, (int, float))
                and not isinstance(baseline_mean, bool)
                else "缺失"
            )
            min_effect_undefined.append(
                f"{label}（Δ={delta:+.4f}，预注册最小效应 "
                f"{min_effect:.2%}×对照均值，但对照均值="
                f"{baseline_text} 为零/近零或缺失，相对阈值"
                "无法定义）"
            )
        elif status == "inconclusive_underpowered":
            power = entry.get("power_at_min_effect")
            if isinstance(power, (int, float)):
                power_text = f"，对最小效应功效≈{float(power):.2f}"
            else:
                mde = entry.get("mde_80_vs_min_effect")
                power_text = (
                    f"，家族校正 MDE≈{float(mde):.4f} "
                    f"> 最小效应阈值 {threshold_abs:.4f}"
                    if isinstance(mde, (int, float))
                    else ""
                )
            inconclusive_underpowered.append(
                f"{label}（Δ={delta:+.4f}，方向一致但不显著"
                f"{power_text}（目标 {_TARGET_POWER:.2f}））"
            )

    if refuted:
        advisories.append(
            "以下对比的观测效应与预注册方向显著相反（假设"
            "被数据推翻，不能靠事后改写解释；如要提出新"
            "解释，应作为新的预注册假设重新实验）："
            + "; ".join(refuted)
        )
    if below_min_effect:
        advisories.append(
            "以下对比方向一致但幅度未达预注册的最小有意义"
            "效应：即使统计显著也不能主张支持假设（事先声明"
            "的最小效应正是用来区分「显著」与「有意义」）："
            + "; ".join(below_min_effect)
        )
    if min_effect_undefined:
        advisories.append(
            "以下对比声明了相对最小效应，但对照均值为零/近零"
            "或缺失：比例阈值没有可解释的尺度，既不能判为达到"
            "最小效应，也不能判为不显著，结论应为不可判定："
            + "; ".join(min_effect_undefined)
        )
    if inconclusive_underpowered:
        advisories.append(
            "以下对比方向一致但不显著，且当前设计对预注册"
            "最小效应的功效不足：不显著是样本量不足的预期"
            "结果，不能解读为「无效应」，也不能解读为「等价」；"
            "需要增加随机种子或降低种子间方差才能判定："
            + "; ".join(inconclusive_underpowered)
        )
    if min_effect_unverified:
        advisories.append(
            "以下对比的点估计超过预注册最小效应且相对 0 显著，"
            "但「真实效应达到门槛」本身未被检验支持（门槛"
            "单侧检验不显著，或方向调整后的置信下界未超过"
            "门槛）：不能标记为 supported，需要更多种子/更长"
            "训练或重新声明一个与效应量相称的门槛："
            + "; ".join(min_effect_unverified)
        )

    # 预注册主指标的口径族：频率加权的似然（val_loss / val_ppl /
    # bpb）与类平衡似然（val_macro_nll）回答的不是同一个问题
    # （整体似然 vs 稀有字节）。不平衡场景下只用其中一个判定，
    # 结论会片面：声明与判定口径不一致时必须点名。
    frequency_weighted_metrics = {
        "val_loss",
        "val_ppl",
        "val_bits_per_byte",
        _CURVE_MEAN_METRIC,
    }
    balanced_likelihood_metrics = {"val_macro_nll"}
    declared_primary_metrics = {
        str(declaration.get("primary_metric"))
        for declaration in declarations.values()
        if declaration.get("primary_metric")
    }
    for declared_metric in sorted(declared_primary_metrics):
        declared_base = (
            _common_step_metric_base(declared_metric)
            or declared_metric
        )
        if declared_base in frequency_weighted_metrics:
            advisories.append(
                f"预注册主指标 {declared_metric} 按 token 频率加权："
                "类别不平衡时它几乎只反映高频字节，稀有字节学没"
                "学会都看不出来；建议同时预注册并报告类平衡口径 "
                "val_macro_nll（对出现类等权，随机水平 log 词表"
                "大小），两者回答的问题不同，不能只报对自己有利"
                "的一个"
            )
        elif declared_base in balanced_likelihood_metrics:
            advisories.append(
                f"预注册主指标 {declared_metric} 是类平衡似然："
                "它回答「稀有字节学没学会」，与频率加权的 "
                "val_loss 回答的问题不同；结论必须同时报告两者，"
                "不能只报对自己有利的一个"
            )
    mismatched_declared = sorted(
        declared_metric
        for declared_metric in declared_primary_metrics
        if declared_metric != metric
    )
    if mismatched_declared and declarations:
        advisories.append(
            "预注册主指标（"
            + ", ".join(mismatched_declared)
            + "）与当前判定指标（"
            + str(metric)
            + "）不一致：该假设无法被当前对比核对，"
            "两个口径的结论可能方向相反；要么用预注册指标判定"
            "（--compare-metric），要么按新口径重新预注册，"
            "不能事后换口径"
        )
    if not checks:
        advisories.append(
            "没有可核对的预注册对比：运行未声明 "
            "--primary-metric/--hypothesis-direction，或声明"
            "的主指标与当前判定指标不一致；方向与最小效应"
            "核对需要训练前预注册"
        )

    status_counts: Dict[str, int] = {}
    for check in checks:
        status = str(check.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "metric": metric,
        "n_checks": len(checks),
        "checks": checks,
        "status_counts": status_counts,
        "refuted": refuted,
        "supported": supported,
        "below_min_effect": below_min_effect,
        "min_effect_undefined": min_effect_undefined,
        "inconclusive_underpowered": (
            inconclusive_underpowered
        ),
        "advisories": advisories,
    }
