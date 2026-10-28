from __future__ import annotations



def _val_curve_points(
    run_dir: Path,
    flops_per_forward: float,
) -> List[Tuple[float, float]]:
    """
    从 metrics.jsonl 读出 (累计训练 FLOPs, val_loss) 验证曲线点。

    只使用周期验证记录（数值 step + 有限 val_loss）：final 记录
    是「验证集最优 checkpoint」的指标，不是该步的即时指标，把它
    混进曲线会让不同运行在相同计算量点比较时引入选型偏差。
    """
    points: Dict[float, float] = {}

    for record in read_metrics_records(
        run_dir / "metrics.jsonl"
    ):
        step = _finite_float(record.get("step"))
        loss = _finite_float(record.get("val_loss"))
        if step is None or step <= 0.0 or loss is None:
            continue
        # 实际推进过优化器的步数才是有效训练计算量：非有限
        # loss/梯度会跳过 optimizer.step，按 step 计算会虚增
        # 便宜/昂贵臂的累计 FLOPs。旧记录没有该字段时退回
        # step（并保留 note 说明口径）。
        optimizer_steps = _finite_float(
            record.get("optimizer_steps")
        )
        compute_steps = (
            optimizer_steps
            if optimizer_steps is not None
            and optimizer_steps >= 0.0
            else step
        )
        cumulative = (
            compute_steps
            * _TRAIN_FLOPS_MULTIPLIER
            * flops_per_forward
        )
        # 累计 FLOPs 也可能超出 float 范围（例如有限的前向
        # FLOPs 接近上限时再乘训练倍率/步数）。inf 作为曲线
        # 横坐标会让共同计算量变成 inf、比值变成 NaN，最终
        # 产出非法 JSON；这类点按不可用丢弃，而不是当作正常
        # 观测。
        if not math.isfinite(cumulative):
            continue
        # 同一步有多条验证记录（如 NaN 诊断附加记录）时保留
        # 最后一条；评估协议相同，值应一致。
        points[float(cumulative)] = loss

    return sorted(points.items())



def _val_step_curve_points(
    run_dir: Path,
    metric: str = "val_loss",
) -> List[Tuple[float, float]]:
    """
    从 metrics.jsonl 读出 (训练步数, 即时验证指标) 曲线点。

    与 ``_val_curve_points`` 相同，只使用周期验证记录；``final``
    记录是验证集最优 checkpoint 的指标，混入曲线会把选型偏差
    带进「共同步数」对照。同一步有多条记录时保留最后一条。
    ``metric`` 必须是周期验证实际写出的键；缺该键的记录跳过，
    不从其他指标换算，避免把不可得的口径伪装成观测。
    """
    points: Dict[float, float] = {}

    for record in read_metrics_records(
        run_dir / "metrics.jsonl"
    ):
        step = _finite_float(record.get("step"))
        value = _finite_float(record.get(metric))
        if step is None or step <= 0.0 or value is None:
            continue
        points[float(step)] = value

    return sorted(points.items())



def _interpolate_curve(
    points: List[Tuple[float, float]],
    x: float,
) -> Tuple[Optional[float], Optional[str]]:
    """
    在验证曲线上取 x 处的 loss：命中验证点则精确取值，
    否则在相邻两点间线性插值。x 落在曲线范围外时返回
    (None, 原因)，不静默外推。
    """
    if not points:
        return None, "没有周期验证曲线"
    if x < points[0][0] or x > points[-1][0]:
        return None, "共同计算量落在该运行的验证曲线范围之外"
    if x == points[0][0]:
        return points[0][1], "exact_eval_point"
    if x == points[-1][0]:
        return points[-1][1], "exact_eval_point"

    for index in range(1, len(points)):
        x_prev, y_prev = points[index - 1]
        x_next, y_next = points[index]
        if x <= x_next:
            if x == x_next:
                return y_next, "exact_eval_point"
            ratio = (x - x_prev) / (x_next - x_prev)
            direct = y_prev + ratio * (y_next - y_prev)
            if math.isfinite(direct):
                return direct, "linear_between_evals"
            # 两端一正一负且都接近 float 上限时，差值会先溢出；
            # 用两个已按比例缩放的端点相加，真实插值仍在范围内。
            scaled = (
                y_prev * (1.0 - ratio) + y_next * ratio
            )
            return (
                scaled if math.isfinite(scaled) else None,
                "linear_between_evals",
            )

    return None, "共同计算量落在该运行的验证曲线范围之外"



def annotate_compute_matched_metrics(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    为每个运行计算「等训练计算量」的验证 loss，并写回行级
    指标 ``val_loss_at_matched_flops``。

    动机：延迟合并 A/B 在分支创建后按 k1 倍分支执行后续计算，
    每训练步的 FLOPs 高于 smelt。只比较相同训练步数会把「机制
    收益」和「额外计算」混在一起；只报告效率倍率也不能回答
    「给基线同样的计算量，谁更好」。

    协议：
    - 每步训练 FLOPs ≈ _TRAIN_FLOPS_MULTIPLIER × 测量前向
      FLOPs（同一运行内的常数倍不改变曲线形状）；
    - 共同计算量取所有可比运行中「最后一个周期验证点累计
      FLOPs」的最小值——保证每个运行都在自己的验证曲线范围内
      有观测，不做外推；
    - 在共同计算量处对验证曲线做线性插值（命中验证点则精确
      取值），得到 val_loss_at_matched_flops；
    - 这是**分析期**派生指标，不用于训练期选型/早停；它直接
      进入 COMPARISON_METRIC_KEYS，可用于聚合、显著性检验与
      预注册主指标。

    没有效率测量（--skip-efficiency/旧运行）或没有周期验证
    曲线的运行不参与共同计算量，也不会被虚报指标；原因写入返回
    报告。统计基线 ngram 是查表模型，没有训练 FLOPs，同样排除。
    """
    report: Dict[str, object] = {
        "metric": _COMPUTE_MATCHED_METRIC,
        "train_flops_multiplier": _TRAIN_FLOPS_MULTIPLIER,
        "target_flops": None,
        "target_note": None,
        "runs": [],
        "excluded_runs": [],
        "advisories": [],
        "ok": False,
    }

    curves: Dict[int, List[Tuple[float, float]]] = {}
    for index, row in enumerate(rows):
        row.pop(_COMPUTE_MATCHED_METRIC, None)
        row.pop("matched_compute_target_flops", None)
        row.pop("matched_compute_ratio", None)

        # 数值失败运行已被聚合与检验排除，不能让它的（通常更
        # 早结束的）曲线决定共同计算量：否则所有正常运行都会
        # 被截断到故障运行的训练长度，结论不再代表完整对比。
        if row.get("numerical_status") == "failed":
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "数值失败运行已从聚合与检验排除，"
                    "不参与共同计算量"
                ),
            })
            continue

        flops = _finite_float(row.get("flops_per_forward"))
        if flops is None or flops <= 0.0:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "缺少前向 FLOPs（旧运行或 "
                    "--skip-efficiency），无法换算训练计算量"
                ),
            })
            continue

        points = _val_curve_points(
            Path(str(row.get("path"))),
            flops,
        )
        if not points:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "没有可用的周期验证曲线"
                    "（metrics.jsonl 缺数值 step + val_loss）"
                ),
            })
            continue
        curves[index] = points

    if not curves:
        report["advisories"].append(
            "没有任何运行同时具备前向 FLOPs 与周期验证曲线："
            "无法计算等训练计算量的 val_loss；请用未跳过"
            "效率测量的新版运行重跑"
        )
        return report

    target_flops = min(points[-1][0] for points in curves.values())
    report["target_flops"] = target_flops
    report["target_note"] = (
        "共同计算量 = 可比运行最后一个周期验证点累计训练 FLOPs "
        f"的最小值；训练 FLOPs≈{_TRAIN_FLOPS_MULTIPLIER:g}×"
        "前向 FLOPs；累计步数优先用 optimizer_steps（非有限"
        "跳步不计入），旧记录退回训练 step"
    )

    for index, points in curves.items():
        row = rows[index]
        value, method = _interpolate_curve(points, target_flops)
        if value is None:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": method,
            })
            continue

        row[_COMPUTE_MATCHED_METRIC] = value
        row["matched_compute_target_flops"] = target_flops
        row["matched_compute_ratio"] = (
            target_flops / points[-1][0]
            if points[-1][0] > 0.0
            else None
        )
        report["runs"].append({
            "run": row.get("run"),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "value": value,
            "interpolation": method,
            "last_curve_flops": points[-1][0],
            "compute_ratio": row["matched_compute_ratio"],
        })

    if report["runs"]:
        report["ok"] = True
        # 便宜臂与昂贵臂在共同计算量处截断的程度不同：若最贵
        # 运行只能用到自身计算量的一小部分，说明比较点很早，
        # 结论只适用于该计算量点，不能外推到完整训练结束。
        ratios = [
            float(item["compute_ratio"])
            for item in report["runs"]
            if isinstance(item.get("compute_ratio"), (int, float))
            and not isinstance(item["compute_ratio"], bool)
        ]
        if ratios and min(ratios) < 0.5:
            report["advisories"].append(
                "等计算量点只覆盖最贵运行不到一半的训练计算量"
                f"（最小比例 {min(ratios):.1%}）：结论只适用于该"
                "计算量点，不能当作完整训练结束时的对比；如需完整"
                "计算量结论，请给便宜臂增加训练步数或增加昂贵臂"
                "前段评估密度"
            )
    else:
        report["advisories"].append(
            "没有运行能在共同计算量点提供验证 loss：无法做"
            "等计算量对比"
        )

    if report["excluded_runs"]:
        report["advisories"].append(
            "以下运行未参与等计算量对比（不虚报指标）: "
            + "; ".join(
                f"{item['run']}（{item['reason']}）"
                for item in report["excluded_runs"]
            )
        )

    return report



def annotate_common_step_metrics(
    rows: List[Dict[str, object]],
    target_step: Optional[float] = None,
) -> Dict[str, object]:
    """
    为每个运行计算「共同训练步数」处的即时验证指标，并写回
    行级派生指标 ``<metric>_at_common_step``。

    动机：``final`` 的 val 指标是验证集选中的最优 checkpoint
    的指标。同一批 val 数据既参与 checkpoint 选择/早停，又参与
    跨运行比较时，每个方法各自的「最优值」都会吸收一部分种子
    噪声（winner's curse）；噪声大的方法被高估得更多，显著性
    检验因此有偏。周期验证曲线记录的是各评估步当时的即时模型
    指标，不经过 checkpoint 选择，适合做方法对比的选型无关口径。

    协议：
    - 只使用周期验证记录（数值 step + 有限指标值），排除
      ``final``；
    - 共同步数以各运行最后一个周期验证步的最小值为准（以
      val_loss 曲线确定评估步，保证所有指标对齐同一点），
      保证每个运行都在自己的曲线范围内，不做外推；
    - 命中共有评估步则精确取值，否则在相邻评估点间线性插值；
    - 覆盖周期验证实际计算的全部 val 指标：loss / ppl / bpb /
      token accuracy / micro、macro、weighted F1 / precision /
      recall；若运行把 PR-AUC / lift / ROC-AUC 选为选型指标或
      显式传了 ``--periodic-auc``，周期记录里也会有这些排序
      指标，同样生成共同步数派生口径。没有任何运行写出某个
      排序指标时跳过该项，不把「没算」写成「缺失」。
      路由诊断只在最终评估计算，不能伪造共同步数口径；
    - 这是分析期派生指标，不用于训练期选型/早停；可直接作为
      ``--compare-metric`` 或预注册主指标。

    没有周期验证曲线的运行（旧运行或纯统计基线）不参与，也不会
    虚报指标；原因写入报告。
    """
    report: Dict[str, object] = {
        "metric": _COMMON_STEP_METRIC,
        "metrics": list(_COMMON_STEP_METRICS),
        "target_step": None,
        "target_note": None,
        "metric_reports": {},
        "runs": [],
        "excluded_runs": [],
        "advisories": [],
        "selection_bias_note": (
            "这些指标取自周期验证曲线，不经过验证集 checkpoint "
            "选择，因此避开了 best-checkpoint 的 winner's curse；"
            "但若共同步数由观察到的端点决定（默认 min-of-last），"
            "它仍是数据依赖的选择，不能称为无偏。预先固定 "
            "--common-step 后才是 selection-independent（在共同"
            "支持范围内）"
        ),
        "ok": False,
    }

    for row in rows:
        for metric in _COMMON_STEP_METRICS:
            row.pop(metric, None)
        row.pop("common_step_target_step", None)
        row.pop("common_step_ratio", None)

    failed_excluded: List[Dict[str, object]] = []
    loss_curves: Dict[int, List[Tuple[float, float]]] = {}
    for index, row in enumerate(rows):
        # 与 aggregate_runs 一致：数值失败运行不参与对比，不能
        # 用它的短曲线把共同步数拉到故障停止的位置。
        if row.get("numerical_status") == "failed":
            failed_excluded.append({
                "run": row.get("run"),
                "reason": (
                    "数值失败运行已从聚合与检验排除，"
                    "不参与共同步数"
                ),
            })
            continue
        points = _val_step_curve_points(
            Path(str(row.get("path")))
        )
        if not points:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "没有可用的周期验证曲线"
                    "（metrics.jsonl 缺数值 step + val_loss）"
                ),
            })
            continue
        loss_curves[index] = points

    if not loss_curves:
        report["advisories"].append(
            "没有任何运行具备周期验证曲线：无法计算共同步数"
            "处的选型无关验证指标；请用未跳过周期验证的新版运行重跑"
        )
        return report

    if target_step is None:
        target_step = min(
            points[-1][0] for points in loss_curves.values()
        )
        target_step_rule = "min_last_validation_step"
        target_step_data_dependent = True
    else:
        # target_step 可能来自手工/旧记录或直接调用（如超大
        # JSON 整数），float() 会抛 OverflowError；非有限/非正
        # 步数也不能作为可观测的共同点。保留 float() 对数字
        # 字符串/布尔值的旧兼容性，只拒绝不可表示的值。
        try:
            target_value = float(target_step)
        except (TypeError, ValueError, OverflowError):
            target_value = None
        if (
            target_value is None
            or not math.isfinite(target_value)
            or target_value <= 0.0
        ):
            report["advisories"].append(
                "共同步数不是有限正数：无法计算共同步数处的"
                "选型无关验证指标"
            )
            return report
        target_step = target_value
        target_step_rule = "preregistered_fixed_step"
        target_step_data_dependent = False
    report["target_step"] = target_step
    report["target_step_rule"] = target_step_rule
    report["target_step_data_dependent"] = (
        target_step_data_dependent
    )
    report["target_note"] = (
        "共同步数 = "
        + (
            "训练前显式指定的固定步（--common-step）"
            if not target_step_data_dependent
            else "可比运行最后一个周期验证步的最小值"
        )
        + "；取各运行在该步的即时验证指标（不做外推）。"
        + (
            ""
            if not target_step_data_dependent
            else "该步由观察到的端点决定：早停/不等步数时与"
            "结果相关，不是选型无关口径；正式对比应预先固定 "
            "--common-step"
        )
    )

    metric_reports: Dict[str, Dict[str, object]] = {}
    for base_metric in _COMMON_STEP_BASE_METRICS:
        derived_metric = _common_step_metric_name(base_metric)
        if derived_metric is None:
            continue

        curves: Dict[int, List[Tuple[float, float]]] = {}
        metric_excluded: List[Dict[str, object]] = []
        for index, row in enumerate(rows):
            if row.get("numerical_status") == "failed":
                continue
            if base_metric == "val_loss":
                points = loss_curves.get(index)
            else:
                points = _val_step_curve_points(
                    Path(str(row.get("path"))),
                    base_metric,
                )
            if not points:
                metric_excluded.append({
                    "run": row.get("run"),
                    "reason": (
                        "没有可用的周期验证曲线"
                        f"（metrics.jsonl 缺数值 step + {base_metric}）"
                    ),
                })
                continue
            curves[index] = points

        # 排序指标是可选周期指标：本轮没有任何运行计算它们时
        # 直接跳过，不生成空报告，也不把「没算」混进
        # incomplete 告警。只要至少一个运行有记录，缺记录的
        # 运行仍会进入 excluded_runs 并如实披露。
        if (
            base_metric in _COMMON_STEP_OPTIONAL_BASE_METRICS
            and not curves
        ):
            continue

        metric_runs: List[Dict[str, object]] = []
        for index, points in curves.items():
            row = rows[index]
            value, method = _interpolate_curve(
                points,
                target_step,
            )
            if value is None:
                metric_excluded.append({
                    "run": row.get("run"),
                    "reason": method,
                })
                continue

            step_ratio = (
                target_step / points[-1][0]
                if points[-1][0] > 0.0
                else None
            )
            row[derived_metric] = value
            row["common_step_target_step"] = target_step
            row["common_step_ratio"] = step_ratio
            metric_runs.append({
                "run": row.get("run"),
                "mode": row.get("mode"),
                "seed": row.get("seed"),
                "value": value,
                "interpolation": method,
                "last_curve_step": points[-1][0],
                "step_ratio": step_ratio,
                "selected_model": row.get("selected_model"),
            })

        metric_reports[derived_metric] = {
            "base_metric": base_metric,
            "ok": bool(metric_runs),
            "runs": metric_runs,
            "excluded_runs": metric_excluded,
        }
        if base_metric == "val_loss":
            report["runs"] = metric_runs
            report["excluded_runs"] = metric_excluded

    report["metric_reports"] = metric_reports
    # val_loss 的 metric_excluded 会覆盖同名字段；显式并回数值
    # 失败运行，保证「为何未参与」在报告里始终可见。
    report["excluded_runs"] = (
        failed_excluded + list(report["excluded_runs"])
    )
    report["ok"] = bool(report["runs"])

    if report["runs"]:
        ratios = [
            float(item["step_ratio"])
            for item in report["runs"]
            if isinstance(item.get("step_ratio"), (int, float))
            and not isinstance(item["step_ratio"], bool)
        ]
        if ratios and min(ratios) < 0.5:
            report["advisories"].append(
                "共同步数只覆盖最长运行不到一半的训练步数"
                f"（最小比例 {min(ratios):.1%}）：该点代表较早的"
                "训练阶段，不能直接替代完整训练结束时的排序；"
                "如需完整训练结论，请统一各运行的验证步范围"
            )
    else:
        report["advisories"].append(
            "没有运行能在共同步数点提供验证 loss：无法做"
            "选型无关的即时验证对比"
        )

    if report["excluded_runs"]:
        report["advisories"].append(
            "以下运行未参与共同步数对比（不虚报指标）: "
            + "; ".join(
                f"{item['run']}（{item['reason']}）"
                for item in report["excluded_runs"]
            )
        )

    incomplete = [
        derived_metric
        for derived_metric, metric_report in metric_reports.items()
        if metric_report["excluded_runs"]
    ]
    if incomplete:
        report["advisories"].append(
            "以下共同步数指标未能覆盖全部运行（不虚报缺记录）: "
            + ", ".join(incomplete)
        )

    return report



def _curve_trapezoid_mean(
    points: List[Tuple[float, float]],
    start: float,
    end: float,
) -> Optional[Dict[str, object]]:
    """
    在 [start, end] 上对分段线性验证曲线做梯形积分并取均值。

    只对可观测窗口积分：起点/终点都通过相邻验证点线性插值，
    不把窗口外的点外推进来。用「每段权重×段内均值」而不是先
    求总面积再除以窗口长度，避免两个合法但极大的有限 loss
    在求和/相除时溢出；无法得到有限均值时返回 None，由调用方
    把运行标为缺失，而不是写入 ±inf。
    """
    if not points or not (
        math.isfinite(start)
        and math.isfinite(end)
        and end > start
    ):
        return None

    start_value, start_method = _interpolate_curve(points, start)
    end_value, end_method = _interpolate_curve(points, end)
    if start_value is None or end_value is None:
        return None

    breakpoints = [start]
    values = [start_value]
    for x, y in points:
        if start < x < end:
            breakpoints.append(x)
            values.append(y)
    breakpoints.append(end)
    values.append(end_value)

    window = end - start
    # 两端各自有限不代表差值有限：-1e308 到 1e308 的窗口
    # 宽度会溢出为 inf，此时 width/window 全是 0，会静默返回
    # 0.0 这种看似合法的损失。无法表示窗口就按缺失处理。
    if not math.isfinite(window) or window <= 0.0:
        return None
    terms: List[float] = []
    for index in range(1, len(breakpoints)):
        width = breakpoints[index] - breakpoints[index - 1]
        if not math.isfinite(width) or width < 0.0:
            return None
        # v1/2 + v2/2 避免先相加再除二造成的中间溢出。
        average = values[index - 1] / 2.0 + values[index] / 2.0
        fraction = width / window
        term = fraction * average
        if not math.isfinite(term):
            return None
        terms.append(term)

    try:
        mean = math.fsum(terms)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(mean):
        return None
    return {
        "mean": mean,
        "window_steps": window,
        "start_value": start_value,
        "end_value": end_value,
        "start_interpolation": start_method,
        "end_interpolation": end_method,
        "trapezoid_segments": len(terms),
    }



def annotate_curve_metrics(
    rows: List[Dict[str, object]],
    target_step: Optional[float] = None,
) -> Dict[str, object]:
    """
    为每个运行计算共同训练窗口内的「时均验证损失」。

    动机：RQ1 要比较「验证损失和收敛速度」，但
    ``val_loss_at_common_step`` 只是共同步数处的一个瞬时值：
    两条曲线可以在该点相同，前段轨迹却完全不同。这里把从
    第一个共同可观测验证步到共同终点的周期验证曲线做梯形
    积分，再除以窗口长度，得到不经过 checkpoint 选型的
    ``val_loss_curve_mean_to_common_step``（越低越好）。它直接
    回答「在相同训练预算的前半段，谁的整体损失更低」，可作为
    ``--compare-metric`` 或预注册主指标进入聚合与显著性检验。

    协议：
    - 只用周期验证记录中的有限 ``val_loss``，排除 ``final``；
    - 窗口起点 = 所有可比运行第一个验证步的最大值（首段
      0→第一次验证无观测，不外推）；
    - 窗口终点 = 训练前固定的 ``--common-step``；未给定时退回
      各运行最后验证步的最小值，并显式标记为数据依赖；
    - 端点落在两个验证点之间时线性插值，任何一端超出曲线
      范围都判为缺失，不裁剪、不外推；
    - 数值失败运行不参与，也不能用其短曲线缩短共同窗口；
    - 与共同步数单点指标一样，这是分析期派生指标，不用于
      训练期选型/早停。

    局限：时均损失同时受早期下降速度和最终水平影响，不能单独
    解释为「收敛速率」；必须与 ``val_loss_at_common_step`` /
    ``final`` 一起报告。窗口起点是观测端点，改变评估间隔会改变
    该指标，跨运行必须保持相同的验证调度。
    """
    report: Dict[str, object] = {
        "metric": _CURVE_MEAN_METRIC,
        "start_step": None,
        "end_step": None,
        "start_rule": "max_first_validation_step",
        "end_rule": None,
        "start_data_dependent": True,
        "end_data_dependent": None,
        "runs": [],
        "excluded_runs": [],
        "advisories": [],
        "ok": False,
    }

    curves: Dict[int, List[Tuple[float, float]]] = {}
    for index, row in enumerate(rows):
        for field in (
            _CURVE_MEAN_METRIC,
            "curve_mean_start_step",
            "curve_mean_end_step",
            "curve_mean_window_steps",
        ):
            row.pop(field, None)

        if row.get("numerical_status") == "failed":
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "数值失败运行已从聚合与检验排除，"
                    "不参与曲线时均指标"
                ),
            })
            continue
        points = _val_step_curve_points(
            Path(str(row.get("path")))
        )
        if not points:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "没有可用的周期验证曲线"
                    "（metrics.jsonl 缺数值 step + val_loss）"
                ),
            })
            continue
        curves[index] = points

    if not curves:
        report["advisories"].append(
            "没有任何运行具备周期验证曲线：无法计算共同窗口内"
            "的时均验证损失；请用未跳过周期验证的新版运行重跑"
        )
        return report

    start_step = max(points[0][0] for points in curves.values())
    if target_step is None:
        end_step = min(
            points[-1][0] for points in curves.values()
        )
        end_rule = "min_last_validation_step"
        end_data_dependent = True
    else:
        end_value = _finite_float(target_step)
        if end_value is None or end_value <= 0.0:
            report["advisories"].append(
                "共同终点不是正有限数：无法定义曲线时均指标"
            )
            return report
        end_step = end_value
        end_rule = "preregistered_fixed_step"
        end_data_dependent = False

    report["start_step"] = start_step
    report["end_step"] = end_step
    report["end_rule"] = end_rule
    report["end_data_dependent"] = end_data_dependent
    if end_step <= start_step:
        report["advisories"].append(
            "共同窗口为空（终点不晚于各运行第一个共同验证步）："
            "无法计算时均验证损失；请增大 --common-step 或"
            "延长训练/提高验证密度"
        )
        return report

    for index, points in curves.items():
        row = rows[index]
        result = _curve_trapezoid_mean(
            points, start_step, end_step
        )
        if result is None:
            report["excluded_runs"].append({
                "run": row.get("run"),
                "reason": (
                    "验证曲线不覆盖共同窗口 "
                    f"[{start_step:g}, {end_step:g}]，"
                    "不裁剪也不外推"
                ),
            })
            continue

        row[_CURVE_MEAN_METRIC] = result["mean"]
        row["curve_mean_start_step"] = start_step
        row["curve_mean_end_step"] = end_step
        row["curve_mean_window_steps"] = result[
            "window_steps"
        ]
        span = points[-1][0] - points[0][0]
        curve_coverage = (
            result["window_steps"] / span
            if span > 0.0
            else None
        )
        if (
            curve_coverage is not None
            and not math.isfinite(curve_coverage)
        ):
            curve_coverage = None
        report["runs"].append({
            "run": row.get("run"),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "value": result["mean"],
            "start_step": start_step,
            "end_step": end_step,
            "window_steps": result["window_steps"],
            "start_value": result["start_value"],
            "end_value": result["end_value"],
            "start_interpolation": result[
                "start_interpolation"
            ],
            "end_interpolation": result["end_interpolation"],
            "trapezoid_segments": result[
                "trapezoid_segments"
            ],
            "curve_coverage": curve_coverage,
            "selected_model": row.get("selected_model"),
        })

    report["ok"] = bool(report["runs"])
    coverages = [
        float(item["curve_coverage"])
        for item in report["runs"]
        if isinstance(item.get("curve_coverage"), (int, float))
        and not isinstance(item["curve_coverage"], bool)
    ]
    if coverages and min(coverages) < 0.5:
        report["advisories"].append(
            "共同窗口只覆盖某个运行不到一半的验证曲线"
            f"（最小比例 {min(coverages):.1%}）：结论只适用于该"
            "窗口，不能当作完整训练过程/最终收敛水平的替代"
        )
    if not report["runs"]:
        report["advisories"].append(
            "没有运行能在共同窗口内提供验证 loss：无法做曲线"
            "时均对比"
        )
    if report["excluded_runs"]:
        report["advisories"].append(
            "以下运行未参与曲线时均对比（不虚报指标）: "
            + "; ".join(
                f"{item['run']}（{item['reason']}）"
                for item in report["excluded_runs"]
            )
        )
    if end_data_dependent:
        report["advisories"].append(
            "共同终点取各运行最后验证步的最小值，由观察到的"
            "端点决定（早停/不等步数时与结果相关）；正式对比"
            "应在训练前固定 --common-step"
        )
    return report



def selection_bias_report(
    rows: List[Dict[str, object]],
    metric: str,
    common_step: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    审计跨运行判定指标是否来自验证集选中的 checkpoint。

    ``final`` 的 val 指标在训练结束后由 ``best.pt`` 重新评估，
    而 ``best.pt`` 正是按同一验证集指标（默认 val_loss）选出的。
    同一批数据既用于选型又用于判定时，点估计和显著性检验都会
    带乐观偏差，且偏差大小随各运行的种子噪声变化。报告必须
    点名这一风险，并给出可执行的选型无关口径（共同步数指标）。

    只把明确记录为 ``selected_model=best_val`` 的运行计入
    ``biased_runs``；旧运行缺选型记录时列入 unknown，不虚报。
    """
    report: Dict[str, object] = {
        "metric": metric,
        "applicable": metric in _SELECTION_BIASED_METRICS,
        "biased_runs": [],
        "eligible_runs": 0,
        "failed_runs_excluded": [],
        "last_step_runs": [],
        "unknown_selection_runs": [],
        "unbiased_metric": None,
        "unbiased_runs": 0,
        "available_unbiased_metrics": [],
        "advisory": None,
        # 选型机会：每个 best_val 运行参与「取最优」的验证点数。
        # 验证点越多，抽到极端低点的机会越多，best_val 的乐观
        # 偏差也越大；不同运行数量不等时，选型偏差本身在组间
        # 不可比。
        "selection_opportunities": [],
        "missing_selection_opportunity_runs": [],
        "opportunity_imbalance": None,
        "opportunity_advisory": None,
    }

    if not report["applicable"]:
        return report

    # 分母只统计真正参与选型判定的非确定性、非数值失败运行：
    # ngram 等确定性基线不做 checkpoint 选型；数值失败运行的
    # final 指标可能来自 fallback checkpoint，已被排除出聚合与
    # 检验。把两者算进分母都会稀释偏差比例，让「多少运行的
    # 指标来自选型后的验证集」被低估。
    eligible_runs = 0
    for row in rows:
        if _is_deterministic_run(row):
            continue
        if row.get("numerical_status") == "failed":
            report["failed_runs_excluded"].append(
                str(row.get("run"))
            )
            continue
        eligible_runs += 1
        name = str(row.get("run"))
        selected = row.get("selected_model")
        if selected == "best_val":
            report["biased_runs"].append({
                "run": name,
                "selection_metric": row.get("selection_metric"),
                "selected_step": row.get("end_step"),
                "trained_steps": row.get("trained_steps"),
                "n_eval_points": row.get("curve_n_eval_points"),
            })
        elif selected == "last":
            report["last_step_runs"].append(name)
        else:
            report["unknown_selection_runs"].append(name)

    report["eligible_runs"] = eligible_runs

    # best_val 是「在 T 个验证点里取最小」：T 越大，极值噪声
    # 带来的乐观偏差越大。跨运行比较时若各运行的 T 不同，
    # 偏差幅度也不同，必须把 T 一并披露。
    opportunities = []
    missing_opportunities: List[str] = []
    for entry in report["biased_runs"]:
        value = entry.get("n_eval_points")
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            opportunities.append({
                "run": entry["run"],
                "n_eval_points": int(value),
            })
        else:
            missing_opportunities.append(str(entry["run"]))
    report["selection_opportunities"] = opportunities
    report["missing_selection_opportunity_runs"] = (
        missing_opportunities
    )
    if len(opportunities) >= 2:
        counts = [
            entry["n_eval_points"] for entry in opportunities
        ]
        low = min(counts)
        high = max(counts)
        if high > low:
            try:
                # 直接做整数真除法：两个超大但同量级的计数
                # 仍可能给出可表示的倍率，只有倍率本身越过
                # float 上限时才省略。
                ratio = high / low if low > 0 else None
            except (OverflowError, ValueError):
                ratio = None
            if ratio is not None and not math.isfinite(ratio):
                ratio = None
            report["opportunity_imbalance"] = {
                "min_eval_points": low,
                "max_eval_points": high,
                "ratio": ratio,
                "runs": opportunities,
            }
            report["opportunity_advisory"] = (
                "best_val 运行参与选型的验证点数不同（"
                f"{low}~{high}"
                + (
                    f"，最多相差 {ratio:.2f}×"
                    if ratio is not None
                    else ""
                )
                + "）：验证点更多的运行有更多机会抽到低点，"
                "选型乐观偏差也更大，且该偏差在组间不可比；"
                "请用共同步数指标复核方法差异"
            )
    elif report["biased_runs"] and not opportunities:
        report["opportunity_advisory"] = (
            "best_val 运行缺少验证点数量记录"
            "（curve_n_eval_points，旧运行）：无法核对各运行"
            "的选型机会是否相同"
        )
    if missing_opportunities and opportunities:
        partial_note = (
            "选型机会无法完整核对：以下 best_val 运行缺少"
            "验证点数量记录（curve_n_eval_points，旧运行）: "
            + ", ".join(missing_opportunities)
        )
        report["opportunity_advisory"] = (
            (
                str(report["opportunity_advisory"]) + "；"
                if report["opportunity_advisory"]
                else ""
            )
            + partial_note
        )

    metric_reports = (
        _as_dict(common_step).get("metric_reports")
        if common_step is not None
        else None
    )
    metric_reports = _as_dict(metric_reports)
    report["available_unbiased_metrics"] = sorted(
        derived_metric
        for derived_metric, metric_report in metric_reports.items()
        if _as_dict(metric_report).get("ok")
    )
    unbiased_metric = _common_step_metric_name(metric)
    if unbiased_metric is not None:
        unbiased_report = _as_dict(
            metric_reports.get(unbiased_metric)
        )
        if unbiased_report.get("ok"):
            report["unbiased_metric"] = unbiased_metric
            report["unbiased_runs"] = len(
                unbiased_report.get("runs", [])
            )

    if report["biased_runs"]:
        detail = (
            f"{len(report['biased_runs'])}/{eligible_runs} 个运行的"
            f" {metric} 来自验证集选中的 checkpoint（best_val）"
        )
        if report["unbiased_metric"]:
            detail += (
                "；同一验证集既选型又判定，存在乐观偏差。"
                f"选型无关复核请用 --compare-metric "
                f"{report['unbiased_metric']}（覆盖 "
                f"{report['unbiased_runs']} 个运行）"
            )
        else:
            detail += (
                "；同一验证集既选型又判定，存在乐观偏差。"
                "该指标没有曲线派生版本，跨运行结论只能作为"
                "描述性证据，正式结论应在独立评估集/固定步数"
                "口径上复核"
            )
        report["advisory"] = detail

    return report



def numerical_failure_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    跨运行暴露数值失败状态，并说明其不进入聚合/检验。

    一个训练到 NaN/Inf 或触发数值停止的运行，其最终 val 指标
    可能来自 fallback checkpoint，甚至已被跳过；把它与正常
    运行一起求均值/做 t 检验，相当于把故障当成方法效应。
    """
    failed = [
        {
            "run": str(row.get("run")),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "curve_status": row.get("curve_status"),
            "nonfinite_events": row.get("nonfinite_events"),
            "numerical_stop": row.get("numerical_stop"),
            "numerical_polluted": row.get(
                "numerical_polluted"
            ),
            "prior_numerical_stop": row.get(
                "prior_numerical_stop"
            ),
        }
        for row in rows
        if row.get("numerical_status") == "failed"
    ]
    unknown = [
        str(row.get("run"))
        for row in rows
        if row.get("numerical_status") == "unknown"
    ]
    mode_counts: Dict[str, Dict[str, object]] = {}
    for row in rows:
        mode = str(row.get("mode"))
        entry = mode_counts.setdefault(mode, {
            "mode": mode,
            "attempted_runs": 0,
            "failed_runs": 0,
            "failed_run_names": [],
        })
        entry["attempted_runs"] = (
            int(entry["attempted_runs"]) + 1
        )
        if row.get("numerical_status") == "failed":
            entry["failed_runs"] = (
                int(entry["failed_runs"]) + 1
            )
            entry["failed_run_names"].append(
                str(row.get("run"))
            )
    by_mode: List[Dict[str, object]] = []
    for mode in sorted(mode_counts):
        entry = mode_counts[mode]
        attempted = int(entry["attempted_runs"])
        entry["failure_rate"] = (
            float(entry["failed_runs"]) / float(attempted)
            if attempted > 0
            else None
        )
        by_mode.append(entry)
    failure_rates = {
        float(entry["failure_rate"])
        for entry in by_mode
        if entry.get("failure_rate") is not None
    }
    imbalanced = (
        any(rate > 0.0 for rate in failure_rates)
        and len(failure_rates) > 1
    )

    report: Dict[str, object] = {
        "failed": failed,
        "excluded_run_count": len(failed),
        "unknown": unknown,
        "by_mode": by_mode,
        "mode_failure_rates_imbalanced": imbalanced,
        "advisory": None,
    }
    if failed:
        message = (
            f"{len(failed)} 个运行存在 NaN/Inf、数值停止、污染"
            "检查点续跑或 diverged 曲线：已从聚合与显著性检验中"
            "排除；请修复数值问题或从干净检查点重跑后再比较"
        )
        if imbalanced:
            message += (
                "。各 mode 的失败率不同（"
                + "; ".join(
                    f"{entry['mode']}={entry['failed_runs']}/"
                    f"{entry['attempted_runs']}"
                    for entry in by_mode
                )
                + "）：剔除失败运行后比较的是幸存者，"
                "失败率本身可能是方法稳定性差异，不能把幸存"
                "运行的优势归因到方法"
            )
        else:
            message += (
                "。注意：失败运行不是方法效果，但若某臂的失败"
                "率更高，仍属于方法稳定性差异，不能只看幸存"
                "运行的均值"
            )
        report["advisory"] = message
    return report



def failure_rate_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    按实验臂报告数值失败率与成功条件性能（防止幸存者偏差）。

    只比较成功运行会系统性掩盖「某个方法更容易数值失败」的
    稳定性成本。这里把「计划运行」（实际启动的运行目录数）
    作为分母，分别给出失败数、失败率、未知数，并在臂间失败
    率差超过阈值时点名 ``differential_failure``。失败运行仍
    不进入均值/检验（否则故障会被当成效应），但必须与失败率
    一起报告；成功条件下的指标已经由聚合表给出。

    重试规则：本仓库不自动重试。用同一个 seed 重跑成功路径会
    产生伪重复（独立种子数不增加）；失败后应换新 seed 或修复
    数值问题后整批重跑，并在报告中保留失败目录。
    """
    groups: Dict[str, Dict[str, object]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        arm = groups.setdefault(
            mode,
            {
                "mode": mode,
                "planned_runs": 0,
                "observed_runs": 0,
                "failed_runs": 0,
                "unknown_runs": 0,
                "failed_run_names": [],
                "success_runs": 0,
            },
        )
        arm["planned_runs"] = int(arm["planned_runs"]) + 1
        status = row.get("numerical_status")
        if status == "failed":
            arm["failed_runs"] = int(arm["failed_runs"]) + 1
            arm["failed_run_names"].append(str(row.get("run")))
        elif status == "unknown":
            arm["unknown_runs"] = int(arm["unknown_runs"]) + 1
        else:
            arm["success_runs"] = int(arm["success_runs"]) + 1
        arm["observed_runs"] = int(arm["planned_runs"]) - int(
            arm["failed_runs"]
        )

    arms = list(groups.values())
    for arm in arms:
        planned = int(arm["planned_runs"])
        failed = int(arm["failed_runs"])
        arm["failure_rate"] = (
            failed / planned if planned > 0 else None
        )

    rates = [
        (str(arm["mode"]), float(arm["failure_rate"]))
        for arm in arms
        if arm["failure_rate"] is not None
    ]
    differential = None
    if len(rates) >= 2:
        worst = max(rates, key=lambda item: item[1])
        best = min(rates, key=lambda item: item[1])
        if worst[1] - best[1] > 0.10:
            differential = {
                "worst_mode": worst[0],
                "worst_rate": worst[1],
                "best_mode": best[0],
                "best_rate": best[1],
                "gap": worst[1] - best[1],
                "note": (
                    "各臂失败率差异超过 10 个百分点：只比较成功"
                    "运行会掩盖失败率更高的方法的稳定性成本；"
                    "该方法的性能结论必须以「失败率 + 成功条件"
                    "指标」成对报告"
                ),
            }

    return {
        "arms": arms,
        "differential_failure": differential,
        "retry_policy": (
            "不自动重试；同一 seed 重跑成功路径属于伪重复，"
            "不增加独立种子数。失败后换新 seed 或修复数值问题"
            "后整批重跑，并保留失败目录"
        ),
        "note": (
            "planned_runs 为实际启动的运行目录数；失败运行不"
            "进入均值/检验，但失败率必须与成功条件指标一起"
            "报告"
        ),
    }



# 曲线诊断中可跨运行比较的数值字段（来自 metrics.jsonl 的
# curve_analysis 记录）。status 之外还必须带出这些量：只看
# 最终 val_loss 会把「训练更久/选型更晚」误当成方法更好，
# 而回归幅度、泛化 gap 与最优点位置才能说明收益是否来自
# 泛化改善还是过拟合前的短暂低点。
# 坏案例记录中可跨运行汇总的字段（metrics.jsonl 的
# ``step="bad_cases", split="val"``）。错误类别按错误 token
# 加权汇总，混淆对只覆盖各运行落盘的 top-10。
_BAD_CASE_ROW_FIELDS: Tuple[str, ...] = (
    "tokens_evaluated",
    "wrong_tokens",
    "token_accuracy",
    "error_category_counts",
    "error_category_shares",
    "top_confusions",
    "error_rate_by_position",
    "error_rate_by_train_frequency",
    "error_frequency_basis",
    "calibration",
    "label_noise_verification",
)


_CURVE_ROW_FIELDS: Tuple[str, ...] = (
    "best_val_step",
    "best_val_loss",
    "val_regression_from_best",
    "generalization_gap_ratio_last",
    "generalization_gap_trend",
    "recent_train_drop",
    "recent_val_drop",
    "last_lr_over_peak",
    "grad_norm_spike_ratio",
    "gap_widening_warning",
    "finite_diverging",
    "n_eval_points",
    "n_nonfinite_events",
)



def curve_diagnostics_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    汇总各运行的训练曲线数值诊断，供跨运行比较。

    ``status`` 是离散结论，无法回答「过拟合多严重」「最优点
    有多早」；这里保留每个运行的回归幅度、泛化 gap、近期
    训练/验证变化与梯度尖峰倍率。只读取已落盘的
    ``curve_analysis`` 记录，不重新训练、不改变任何选型。
    """
    runs = [
        {
            "run": str(row.get("run")),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "status": row.get("curve_status"),
            **{
                key: _json_safe(row.get(f"curve_{key}"))
                for key in _CURVE_ROW_FIELDS
            },
        }
        for row in rows
        if row.get("curve_available")
    ]
    overfit_runs = [
        entry["run"]
        for entry in runs
        if entry.get("status") == "overfitting"
    ]
    return {
        "runs": runs,
        "n_runs_with_curve": len(runs),
        "overfitting_runs": overfit_runs,
        "note": (
            "best@ 是验证损失最优点；regr 是最终相对最优的"
            "回升幅度；gap 是最终 (val-train)/val。regr 大且"
            "best@ 早说明过拟合严重，不能只看最终 val_loss。"
        ),
    }
