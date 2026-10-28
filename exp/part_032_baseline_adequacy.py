from __future__ import annotations



def baseline_adequacy(
    rows: List[Dict[str, object]],
    baseline_mode: Optional[str],
    metric: str = "val_loss",
) -> Dict[str, object]:
    """
    基线合理性检查（稻草人守卫）。

    复杂模型的涨点只有相对「精心调优过的基线」才有意义：
    没调过参的弱基线只能证明复杂模型比稻草人强。本检查在
    对比阶段核对 ``--compare-baseline`` 指定的基线：

    1. 基线在同口径下必须真的优于无上下文 unigram 参考
       （其值随运行 config.json 的 reference_baselines 存档）；
       否则标记 ``[straw-man]``——战胜这样的基线不能证明
       复杂模型的收益；
    2. 若对比集合里有调优好的字节级 n-gram 统计基线，而它在
       主指标上已不劣于所选神经基线，同样标记 ``[straw-man]``：
       应先把基线训练/调参到强于统计基线，再谈复杂模型；
    3. 神经基线必须有调参证据（--suite-baseline-tuning 产生
       的调参报告 + --baseline-tuning-from 写入 config.json 的
       baseline_tuning 来源记录），否则标记 ``[straw-man]``：
       没调过参的弱基线只能证明复杂模型比稻草人强；调参口径
       与当前对比指标不一致时单独提示（指标未对齐）；
    4. 基线本身种子数不足时提示统计功效不足，避免把基线的
       种子波动误读成改进。

    返回结构化报告，随 comparison_analysis.json 落盘。
    """
    report: Dict[str, object] = {
        "baseline_mode": baseline_mode,
        "metric": metric,
        "n_baseline_runs": 0,
        "n_baseline_runs_failed": 0,
        "baseline_mean": None,
        "checks": [],
        "straw_man": [],
        "advisories": [],
        "tuned": None,
        "tuned_runs": [],
        "untuned": [],
        "tuning_metric_mismatch": [],
        "unconverged_baselines": [],
        "capacity_confounds": [],
        "baseline_capacity": None,
        "ngram_missing": False,
        "ngram_protocol_compatible": None,
        "ngram_protocol_violations": [],
        "ngram_comparison": None,
        "ngram_tuning_metric_mismatch": [],
        "tuning_parity": None,
        "ok": None,
    }

    if not baseline_mode:
        report["advisories"].append(
            "未指定 --compare-baseline：无法定位基线并核对它"
            "是否强于无上下文参考；建议显式指定（如 "
            "--compare-baseline smelt）"
        )
        return report

    matching_baseline_rows = [
        row
        for row in rows
        if str(row.get("mode")) == baseline_mode
    ]
    failed_baseline_rows = [
        row
        for row in matching_baseline_rows
        if row.get("numerical_status") == "failed"
    ]
    baseline_rows = [
        row
        for row in matching_baseline_rows
        if row.get("numerical_status") != "failed"
    ]
    report["n_baseline_runs"] = len(baseline_rows)
    report["n_baseline_runs_failed"] = len(failed_baseline_rows)

    if failed_baseline_rows:
        # 数值失败/发散运行的最终指标可能来自 fallback checkpoint，
        # 不能拿来证明基线「强于稻草人」或「值得战胜」：把它们排除
        # 出本检查，与 aggregate_runs 的均值/检验口径保持一致。
        report["advisories"].append(
            f"{len(failed_baseline_rows)} 个 mode={baseline_mode} "
            "的基线运行存在 NaN/Inf、数值停止或发散，"
            "已从基线合理性核对中排除（"
            + ", ".join(
                str(row.get("run"))
                for row in failed_baseline_rows
            )
            + "）：数值失败运行的 fallback 指标不能证明基线合理"
        )

    if not baseline_rows:
        if failed_baseline_rows:
            report["straw_man"].append(
                f"mode={baseline_mode} 的基线运行全部数值失败/"
                "发散：没有可用的基线结果，无法核对基线合理性"
            )
        else:
            report["straw_man"].append(
                f"未找到 mode={baseline_mode} 的基线运行，"
                "无法核对基线合理性"
            )
        report["ok"] = False
        return report

    baseline_samples = _metric_samples(baseline_rows, metric)
    collapsed_baseline_samples = _collapse_seed_samples(
        baseline_samples
    )
    baseline_values = [
        float(sample["value"])
        for sample in collapsed_baseline_samples
    ]
    duplicate_baseline_seeds = sorted(
        (
            sample.get("seed")
            for sample in collapsed_baseline_samples
            if int(sample.get("n_runs", 1)) > 1
        ),
        key=_factor_sort_rank,
    )
    report["duplicate_baseline_seeds"] = duplicate_baseline_seeds
    baseline_mean = _mean(baseline_values)
    report["baseline_mean"] = baseline_mean

    if baseline_mean is None:
        report["straw_man"].append(
            f"基线运行没有可用的 {metric} 指标，无法核对"
        )
        report["ok"] = False
        return report

    higher_is_better = metric_higher_is_better(metric)
    reference_supported = metric in _TRIVIAL_REFERENCE_FIELDS

    # 1) 基线 vs 无上下文 unigram 参考。
    checked = 0
    missing_reference = 0

    if not reference_supported:
        report["advisories"].append(
            f"指标 {metric} 没有无上下文参考口径，"
            "跳过「基线强于稻草人」的核对"
        )
    else:
        for row in baseline_rows:
            value = _finite_float(row.get(metric))
            if value is None:
                continue

            reference_value = _trivial_reference_value(
                metric,
                row.get("reference_baselines"),
            )
            reference_source = _TRIVIAL_REFERENCE_FIELDS[metric]
            if reference_value is None and metric == (
                "val_pr_auc_macro"
            ):
                # 旧运行的 reference_baselines 没有常数预测器
                # 的 PR-AUC 水平，但该运行自身的
                # val_pr_auc_macro_chance 就是它的随机水平
                # （平均 prevalence），拿它做对照同样能完成
                # 稻草人核对，只是来源不同，需在检查记录里注明。
                fallback_chance = row.get(
                    "val_pr_auc_macro_chance"
                )
                fallback_value = _finite_float(fallback_chance)
                if fallback_value is not None:
                    reference_value = fallback_value
                    reference_source = (
                        "val_pr_auc_macro_chance"
                        "（该运行的 PR-AUC 随机水平）"
                    )
            if reference_value is None:
                missing_reference += 1
                continue

            checked += 1
            better = (
                float(value) > reference_value
                if higher_is_better
                else float(value) < reference_value
            )
            report["checks"].append({
                "run": row.get("run"),
                "value": float(value),
                "trivial_reference": reference_value,
                "trivial_reference_source": reference_source,
                "better_than_trivial": better,
            })
            if not better:
                report["straw_man"].append(
                    f"基线 {row.get('run')} 的 {metric}="
                    f"{float(value):.4f} 没有优于无上下文 "
                    f"unigram 参考 {reference_value:.4f}："
                    "战胜这样的弱基线不能证明复杂模型的收益，"
                    "请先把基线训练/调参到强于参考"
                )

        if missing_reference:
            report["advisories"].append(
                f"{missing_reference} 个基线运行缺少参考基线记录"
                "（旧运行或统计基线），无法核对是否强于"
                "无上下文参考"
            )

    # 2) 基线 vs 调优好的 n-gram 统计基线（若在对比集合中）。
    # 基线本身就是 ngram 时不能拿 ngram 与自己比：baseline_rows
    # 与 ngram_rows 是同一批运行，差值恒为 0，会凭空给出
    # 「基线不劣于 ngram」的稻草人判定；统计基线自身在 val 上
    # 做过网格调参，按第 3 步的豁免口径处理。
    matching_ngram_rows = [
        row
        for row in rows
        if str(row.get("mode")) == "ngram"
    ]
    failed_ngram_rows = [
        row
        for row in matching_ngram_rows
        if row.get("numerical_status") == "failed"
    ]
    ngram_rows = [
        row
        for row in matching_ngram_rows
        if row.get("numerical_status") != "failed"
        and _finite_float(row.get(metric)) is not None
    ]
    if failed_ngram_rows and baseline_mode != "ngram":
        report["advisories"].append(
            f"{len(failed_ngram_rows)} 个 ngram 统计基线运行"
            "存在数值失败/发散，已从「基线不劣于 n-gram」的"
            "对照中排除（"
            + ", ".join(
                str(row.get("run"))
                for row in failed_ngram_rows
            )
            + "）"
        )

    def _ngram_tuning_metric_of(
        row: Dict[str, object],
    ) -> str:
        # 旧版统计基线只在 val_loss 上选 (order, alpha)，
        # 没有记录时按这一历史口径处理，不虚报为已按当前
        # 指标调强。
        value = row.get("ngram_tuning_metric")
        return str(value) if isinstance(value, str) else "val_loss"

    ngram_metric_mismatches = []
    for row in ngram_rows:
        ngram_metric = _ngram_tuning_metric_of(row)
        if ngram_metric != metric:
            ngram_metric_mismatches.append({
                "run": row.get("run"),
                "tuning_metric": ngram_metric,
                "comparison_metric": metric,
            })
    if ngram_metric_mismatches:
        report["ngram_tuning_metric_mismatch"] = (
            ngram_metric_mismatches
        )
        details = "; ".join(
            f"{item['run']}（调参口径={item['tuning_metric']}）"
            for item in ngram_metric_mismatches
        )
        report["advisories"].append(
            "n-gram 基线的 (order, alpha) 不是在当前判定口径"
            f"{metric} 上选出的（{details}）："
            "它在该指标上不能被称为「调优好的简单基线」；"
            "请用 --ngram-tuning-metric "
            f"{metric} --statistical-baseline 重跑"
        )

    baseline_better_than_ngram: Optional[bool] = None
    # ngram 与神经运行必须使用同一份数据、同一组评估窗口，
    # 否则「基线强于 ngram」把数据差异误当成方法差异。
    ngram_cross_family = cross_family_protocol_check(
        ngram_rows + baseline_rows,
        [],
    )
    if ngram_rows and baseline_rows:
        report["ngram_protocol_compatible"] = ngram_cross_family[
            "ok"
        ]
        report["ngram_protocol_violations"] = (
            ngram_cross_family["violations"]
        )
        if ngram_cross_family["violations"]:
            keys = ", ".join(
                str(item["key"])
                for item in ngram_cross_family["violations"]
            )
            message = (
                "ngram 统计基线与神经运行的数据/评估口径不一致"
                f"（{keys}）：两者不能直接比较，"
                "「基线强于/不劣于 ngram」的判断已跳过；"
                "请在同一 --data / 切分比例 / --eval-batches / "
                "数据增强设置下重跑 ngram"
            )
            report["advisories"].append(message)
    if baseline_mode != "ngram" and not ngram_rows:
        report["ngram_missing"] = True
        report["advisories"].append(
            "对比集合未包含 n-gram 统计基线：无法确认神经基线"
            "已强于简单统计基线，简单基线守卫会退化为只看"
            "unigram 参考；请用 --statistical-baseline 在同一"
            "数据/评估口径下训练 ngram，并把该运行加入 --compare"
        )
    if (
        ngram_rows
        and baseline_mode != "ngram"
        and report["ngram_protocol_compatible"] is not False
    ):
        ngram_mean = _mean(
            [float(row[metric]) for row in ngram_rows]
        )
        baseline_better = (
            baseline_mean > float(ngram_mean)
            if higher_is_better
            else baseline_mean < float(ngram_mean)
        )
        baseline_better_than_ngram = baseline_better
        ngram_label = (
            "调优好的 n-gram"
            if not ngram_metric_mismatches
            else "n-gram 统计基线"
        )
        # 均值差必须连同基线自身的种子噪声一起解读：n-gram 是
        # 确定性估计（无抽样方差），而基线均值来自 n 个种子；
        # 单看均值会把种子波动当成「基线强于/弱于统计基线」的
        # 结论。对差值做单样本 t 检验，并给出当前设计能检出的
        # MDE 与所需种子数。
        delta_value = baseline_mean - float(ngram_mean)
        ngram_comparison: Dict[str, object] = {
            "metric": metric,
            "baseline_mode": baseline_mode,
            "baseline_mean": baseline_mean,
            "ngram_mean": float(ngram_mean),
            # 原始差值（基线 - ngram）；方向好坏由
            # baseline_better 与指标方向共同解释。两个有限均值
            # 相减仍可能越过 float 上限，无法表示时按缺失处理。
            "delta": (
                delta_value
                if math.isfinite(delta_value)
                else None
            ),
            "baseline_better": baseline_better,
            "n_baseline_runs": len(baseline_rows),
            "n_baseline_values": len(baseline_values),
            "baseline_std": _sample_std(baseline_values),
            "ngram_deterministic": True,
            "test": None,
            "p_value": None,
            "se": None,
            "df": None,
            "significant": None,
            "mde_80": None,
            "mde_80_relative": None,
            "seeds_to_detect": None,
            "advisories": [],
        }
        differences = [
            value - float(ngram_mean)
            for value in baseline_values
        ]
        test = _paired_t_test(differences)
        ngram_comparison["test"] = test.get("test")
        for key in ("p_value", "se", "df"):
            value = test.get(key)
            if isinstance(value, (int, float)):
                ngram_comparison[key] = float(value)
        p_value = ngram_comparison["p_value"]
        se = ngram_comparison["se"]
        df = ngram_comparison["df"]
        if isinstance(p_value, (int, float)):
            ngram_comparison["significant"] = bool(
                float(p_value) <= _SIGNIFICANCE_LEVEL
            )
        if (
            isinstance(se, (int, float))
            and isinstance(df, (int, float))
            and float(se) > 0.0
            and float(df) > 0.0
        ):
            mde = _minimum_detectable_effect(
                float(se), float(df)
            )
            if mde is not None:
                ngram_comparison["mde_80"] = mde
                if abs(float(baseline_mean)) > 0.0:
                    mde_relative = mde / abs(float(baseline_mean))
                    # 基线均值可以合法地极小：相对 MDE 会越过
                    # float 上限，无法表示时按缺失处理。
                    if math.isfinite(mde_relative):
                        ngram_comparison["mde_80_relative"] = (
                            mde_relative
                        )
        if (
            isinstance(delta_value, float)
            and math.isfinite(delta_value)
            and abs(delta_value) > 0.0
        ):
            baseline_sd = ngram_comparison["baseline_std"]
            if isinstance(baseline_sd, (int, float)):
                seeds_needed = _seeds_to_detect(
                    delta_value,
                    float(baseline_sd),
                )
                if seeds_needed is not None:
                    ngram_comparison["seeds_to_detect"] = (
                        seeds_needed
                    )
        ngram_comparison_advice: List[str] = []
        if len(baseline_values) < _MIN_SEEDS_FOR_STATS:
            ngram_comparison_advice.append(
                f"基线只有 {len(baseline_values)} 个可用种子"
                f"（建议 ≥{_MIN_SEEDS_FOR_STATS}）：无法核对"
                "与 n-gram 的差距是否超出基线自身的种子噪声"
            )
        elif ngram_comparison["significant"] is False:
            mde = ngram_comparison["mde_80"]
            seeds_needed = ngram_comparison["seeds_to_detect"]
            detail = (
                f"80% 功效 MDE={float(mde):.4f}"
                if isinstance(mde, (int, float))
                else "无法估计 MDE"
            ) + (
                f"，约需 {int(seeds_needed)} 个基线种子才能检出"
                "该差距"
                if isinstance(seeds_needed, int)
                else ""
            )
            ngram_comparison_advice.append(
                f"基线 {baseline_mode} 与 n-gram 的 {metric} "
                f"差距在基线种子噪声内（p="
                f"{float(ngram_comparison['p_value']):.3f}）："
                f"{detail}；点估计方向不能当作已确认的结论"
            )
        elif ngram_comparison["significant"] is True:
            ngram_comparison_advice.append(
                f"基线与 n-gram 的 {metric} 差距超出基线种子"
                f"噪声（p="
                f"{float(ngram_comparison['p_value']):.3f}）："
                "该比较在统计上可分辨（n-gram 为确定性估计，"
                "其无抽样方差）"
            )
        ngram_comparison["advisories"] = (
            ngram_comparison_advice
        )
        report["ngram_comparison"] = ngram_comparison
        report["advisories"].extend(ngram_comparison_advice)
        if baseline_better:
            report["advisories"].append(
                f"基线 {baseline_mode}（{baseline_mean:.4f}）"
                f"优于{ngram_label}（{float(ngram_mean):.4f}）"
                "：这是一个值得战胜的基线"
                + (
                    ""
                    if not ngram_metric_mismatches
                    else "（注意：该 n-gram 未在当前口径上调参，"
                    "此比较只说明它不构成更强的简单基线）"
                )
            )
        else:
            report["straw_man"].append(
                f"{ngram_label}在 {metric} 上"
                f"（{float(ngram_mean):.4f}）已不劣于所选基线 "
                f"{baseline_mode}（{baseline_mean:.4f}）：先对基线"
                "充分调参（学习率/模型宽度/训练步数），再比较"
                "复杂模型是否有收益"
            )

    # 3) 基线调参证据：复杂模型必须战胜「精心调优过的基线」，
    #    没调过参的弱基线只能证明复杂模型比稻草人强。神经基线
    #    的调参来源由 --baseline-tuning-from 写入 config.json 的
    #    baseline_tuning 字段；统计基线（ngram）自身就在 val 上
    #    做过 阶数 × 平滑系数 网格调优，不做此检查。
    if baseline_mode == "ngram":
        # 只有 (order, alpha) 确实按当前对比指标选出时，才能
        # 把统计基线当作「在当前口径上调过参」的证据。
        report["tuned"] = not ngram_metric_mismatches
    else:
        tuning_evidence = False
        for row in baseline_rows:
            record = row.get("baseline_tuning")
            if not isinstance(record, dict):
                continue
            run_name = str(row.get("run"))
            axis_error = _baseline_tuning_axis_error(record)
            if axis_error is not None:
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源不完整："
                    + axis_error
                    + "；不把该记录作为基线已调参的证据"
                )
                continue
            winner_lr = record.get("winner_lr")
            if winner_lr is None:
                winner = record.get("winner")
                if isinstance(winner, dict):
                    winner_lr = winner.get("lr")
            if (
                record.get("mode") != baseline_mode
                or winner_lr is None
            ):
                report["advisories"].append(
                    f"基线 {run_name} 的 baseline_tuning 记录"
                    "不完整（缺 mode/winner.lr），无法核对"
                    "调参来源"
                )
                continue

            # 低功效调参（< 3 个独立种子 / --allow-underpowered）
            # 的 winner 可能是种子噪声，不能作为「基线已充分
            # 调参」的证据；协议不一致的调参报告本身不可用；
            # 旧版记录缺 seed_plan 时无法核对功效，同样不作为
            # 证据（与 --baseline-tuning-from 的口径一致）。
            record_seed_plan = record.get("seed_plan")
            if not isinstance(record_seed_plan, dict):
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源缺少 seed_plan"
                    "（旧版记录）：无法核对调参的独立种子数与"
                    "统计功效，不能作为基线已调参的证据"
                )
                continue
            recorded_seeds = _baseline_tuning_seed_count(
                record_seed_plan
            )
            if recorded_seeds is None:
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源 seed_plan 缺少"
                    "有效的 n_seeds：无法核对调参的独立种子数与"
                    "统计功效，不能作为基线已调参的证据"
                )
                continue
            tuning_underpowered = bool(
                record_seed_plan.get("underpowered_override")
            ) or recorded_seeds < _MIN_SEEDS_FOR_STATS
            if tuning_underpowered:
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源在少于 "
                    f"{_MIN_SEEDS_FOR_STATS} 个独立随机种子的"
                    "情况下选出 winner（underpowered）：winner "
                    "可能只是种子噪声，不能作为基线已调参的证据"
                )
                continue
            record_issues = record.get("issues")
            if isinstance(record_issues, list) and any(
                "训练协议不一致" in str(issue)
                for issue in record_issues
            ):
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源存在候选间训练"
                    "协议不一致：调参结果与训练配置混杂，不能"
                    "作为基线已调参的证据"
                )
                continue
            seed_alignment = record.get("seed_alignment")
            if (
                isinstance(seed_alignment, dict)
                and seed_alignment.get("aligned") is False
            ):
                report["advisories"].append(
                    f"基线 {run_name} 的调参来源种子集合不一致/"
                    "存在伪重复：候选 Mean ± Std 的差异可能"
                    "来自种子而不是学习率，不能作为基线已调参"
                    "的证据"
                )
                continue

            record_metric = record.get("metric")
            if (
                isinstance(record_metric, str)
                and record_metric != metric
            ):
                # 「调在 val_loss 上」不能证明在 macro F1 /
                # PR-AUC 上也已调强：tuned 只表示「在当前
                # 判定口径上调过参」，指标不一致时必须重新
                # 调参，而不是只看一条 warning 就当作强基线。
                message = (
                    f"基线 {run_name} 的调参依据是 {record_metric}，"
                    f"当前对比口径是 {metric}：指标未对齐，"
                    "该调参不能证明基线在当前口径上也是强的；"
                    "请用 --baseline-tuning-metric "
                    f"{metric} 重新调参"
                )
                report["tuning_metric_mismatch"].append(message)
                report["advisories"].append(message)
                continue

            tuning_evidence = True
            report["tuned_runs"].append({
                "run": run_name,
                "tuning_metric": record_metric,
                "winner_lr": winner_lr,
                "n_candidates": record.get("n_candidates"),
            })
            if record.get("winner_at_grid_boundary") or record.get(
                "best_mean_at_grid_boundary"
            ):
                report["advisories"].append(
                    f"基线 {run_name} 的调参 winner/最优均值候选"
                    "位于学习率网格边界：最优可能落在网格之外，"
                    "基线可能仍未被调强；建议扩展候选后重跑调参"
                )

        report["tuned"] = tuning_evidence

        if not tuning_evidence:
            if baseline_better_than_ngram:
                report["advisories"].append(
                    "所选基线没有调参来源记录（config.json 无 "
                    "baseline_tuning），但它在主指标上已强于"
                    "调优好的 n-gram：可先视为强于一个简单基线；"
                    "仍建议用 --suite-baseline-tuning 调参并用 "
                    "--baseline-tuning-from 记录来源，"
                    "排除「基线本身还有空间」的质疑"
                )
            else:
                if baseline_better_than_ngram is None:
                    # None = 对比根本没做（ngram 缺失或口径不兼容
                    # 已跳过），不能写成「没有它能战胜的 n-gram」
                    # 这种已核实的否定结论；缺证据只能表述为缺证据。
                    ngram_clause = (
                        "且没有可用于核验的调优好的 n-gram 对照"
                        "（n-gram 缺失或与当前口径不兼容，比较已跳过）"
                    )
                else:
                    ngram_clause = (
                        "对比集合里也没有它能战胜的调优好的 n-gram"
                    )
                message = (
                    f"所选基线 {baseline_mode} 没有任何调参证据"
                    "（config.json 无 baseline_tuning 记录，"
                    f"{ngram_clause}）："
                    "没调过参的弱基线只能证明复杂模型比稻草人强，"
                    "不能证明方法收益。请先运行 "
                    "--suite-baseline-tuning 对基线做 val 调参，"
                    "再用 --baseline-tuning-from 把调参来源"
                    "写入后续基线训练（或先跑 --statistical-baseline "
                    "得到调优好的 n-gram 作为对照）"
                )
                report["untuned"].append(message)
                report["straw_man"].append(message)

    # 3b) 调参对称性：基线做了学习率网格搜索、挑战臂却
    #     沿用默认学习率时，「基线被调强、挑战臂没调」会让
    #     差值无法归因到机制。基线仍可能确实是强基线（3a 的
    #     结论不受影响），所以这里只给混杂提示，不把基线判为
    #     straw-man：挑战臂若在调参不对称下仍胜出，结论是保守
    #     下界；若落败，则不能排除挑战臂只是没调好。
    parity: Dict[str, object] = {
        "baseline_winner_lrs": [],
        "challengers_using_baseline_lr": [],
        "challengers_with_own_tuning": [],
        "challengers_without_tuning_parity": [],
        "advisories": [],
        "ok": None,
    }
    report["tuning_parity"] = parity

    if baseline_mode != "ngram":
        baseline_tuned_lrs: List[float] = []
        for row in baseline_rows:
            record = row.get("baseline_tuning")
            if not isinstance(record, dict):
                continue
            if _baseline_tuning_axis_error(record) is not None:
                continue
            winner_lr = record.get("winner_lr")
            if winner_lr is None:
                winner = record.get("winner")
                if isinstance(winner, dict):
                    winner_lr = winner.get("lr")
            value = _finite_float(winner_lr)
            if value is not None and value > 0.0:
                baseline_tuned_lrs.append(value)

        unique_lrs = sorted({
            round(value, 12) for value in baseline_tuned_lrs
        })
        parity["baseline_winner_lrs"] = unique_lrs

        if len(unique_lrs) > 1:
            message = (
                "基线运行的调参 winner 学习率不一致（"
                + ", ".join(format(lr, "g") for lr in unique_lrs)
                + "）：各基线运行不是同一条调过参的配置，"
                "调参对称性无法核对"
            )
            parity["advisories"].append(message)
            report["advisories"].append(message)
        elif unique_lrs:
            baseline_lr = float(unique_lrs[0])
            for row in rows:
                row_mode = str(row.get("mode"))
                if row_mode in (baseline_mode, "ngram"):
                    continue
                protocol = row.get("protocol")
                if not isinstance(protocol, dict):
                    continue
                challenger_lr = _finite_float(protocol.get("lr"))
                if challenger_lr is None:
                    continue
                run_name = str(row.get("run"))
                if math.isclose(
                    challenger_lr,
                    baseline_lr,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    parity["challengers_using_baseline_lr"].append(
                        run_name
                    )
                    continue

                record = row.get("baseline_tuning")
                own_tuning = False
                if isinstance(record, dict):
                    record_mode = record.get("mode")
                    record_winner_lr = record.get("winner_lr")
                    if record_winner_lr is None:
                        winner = record.get("winner")
                        if isinstance(winner, dict):
                            record_winner_lr = winner.get("lr")
                    record_lr = _finite_float(record_winner_lr)
                    record_metric = record.get("metric")
                    seed_plan = record.get("seed_plan")
                    record_seeds = _baseline_tuning_seed_count(
                        seed_plan
                    )
                    underpowered = bool(
                        isinstance(seed_plan, dict)
                        and seed_plan.get("underpowered_override")
                    ) or (
                        record_seeds is not None
                        and record_seeds < _MIN_SEEDS_FOR_STATS
                    )
                    issues = record.get("issues")
                    inconsistent = bool(
                        isinstance(issues, list)
                        and any(
                            "训练协议不一致" in str(issue)
                            for issue in issues
                        )
                    )
                    seed_alignment = record.get(
                        "seed_alignment"
                    )
                    seeds_misaligned = bool(
                        isinstance(seed_alignment, dict)
                        and seed_alignment.get("aligned") is False
                    )
                    own_tuning = bool(
                        record_mode == row_mode
                        and _baseline_tuning_axis_error(record)
                        is None
                        and isinstance(seed_plan, dict)
                        and record_seeds is not None
                        and record_seeds >= _MIN_SEEDS_FOR_STATS
                        and record_lr is not None
                        and math.isclose(
                            record_lr,
                            challenger_lr,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        and record_metric == metric
                        and not underpowered
                        and not inconsistent
                        and not seeds_misaligned
                    )
                if own_tuning:
                    parity["challengers_with_own_tuning"].append(
                        run_name
                    )
                else:
                    parity["challengers_without_tuning_parity"].append({
                        "run": run_name,
                        "mode": row_mode,
                        "lr": challenger_lr,
                        "baseline_tuned_lr": baseline_lr,
                    })

            if parity["challengers_without_tuning_parity"]:
                details = "; ".join(
                    f"{item['run']}（{item['mode']}, lr="
                    f"{format(item['lr'], 'g')}）"
                    for item in parity[
                        "challengers_without_tuning_parity"
                    ]
                )
                message = (
                    "调参不对称：基线已在 val 上做学习率网格"
                    f"搜索（winner lr={format(baseline_lr, 'g')}），"
                    "以下挑战臂却使用不同学习率且没有自己的调参"
                    f"来源：{details}。若挑战臂落败，不能排除只是"
                    "学习率没调好；若仍胜出，结论是保守下界。"
                    "请为挑战臂复用同一训练协议的学习率网格"
                    "（相同候选集、相同种子）后再比较，或显式"
                    "记录其调参来源"
                )
                parity["advisories"].append(message)
                report["advisories"].append(message)
        parity["ok"] = not parity["advisories"]

    # 4) 基线本身的统计功效（按独立种子数，伪重复不计入）。
    baseline_deterministic = bool(baseline_rows) and all(
        _is_deterministic_run(row) for row in baseline_rows
    )
    report["baseline_deterministic"] = baseline_deterministic
    baseline_seeds = [
        row.get("seed")
        for row in baseline_rows
        if row.get("seed") is not None
    ]
    baseline_independent_seeds = (
        1
        if baseline_deterministic
        else _independent_seed_count(
            len(baseline_rows),
            baseline_seeds,
        )
    )
    if (
        not baseline_deterministic
        and baseline_independent_seeds < _RECOMMENDED_SEEDS
    ):
        report["advisories"].append(
            f"基线只有 {baseline_independent_seeds} 个种子"
            f"（建议 ≥{_RECOMMENDED_SEEDS}）：基线自身的种子"
            "噪声可能主导改进结论"
        )
    elif baseline_deterministic:
        report["advisories"].append(
            "确定性基线（ngram）不产生种子间波动：多个 seed "
            "运行是同一估计的复制，按 1 个观测处理；其评估"
            "不确定性应来自评估窗口/bootstrap 或多次时间切分，"
            "而不是增加种子数"
        )

    # 5) 基线是否训练到收敛：没训练够的基线与「没调过参」同类，
    #    曲线诊断（compare_runs 读入的 curve_analysis.status）
    #    记录 not_converged（仍在下降，没到平台）/ underfitting
    #    （训练损失几乎不降、接近随机基线）/ diverged。
    #    欠训的基线看起来弱并不说明方法更好——不同训练配置下
    #    「战胜基线」没有可比性。
    unconverged_baselines: List[str] = []
    for row in baseline_rows:
        status = row.get("curve_status")
        if status in (
            "not_converged",
            "underfitting",
            "diverged",
            "diverging",
        ):
            unconverged_baselines.append(
                f"{row.get('run')}（curve_status={status}）"
            )
    if unconverged_baselines:
        message = (
            "以下基线运行没有训练到收敛/稳定（曲线诊断）："
            + "; ".join(unconverged_baselines)
            + "。欠训的弱基线不说明方法更强：请先增加训练步数"
            "或按 --suite-baseline-tuning 调强基线，并在两臂"
            "使用同一训练配置后再比较"
        )
        report["advisories"].append(message)
        report["untuned"].append(message)
        report["unconverged_baselines"] = unconverged_baselines
    else:
        report["unconverged_baselines"] = []

    # 6) 容量/计算量对齐：挑战模型（对比集合里主指标最优的
    #    非基线神经运行）参数量/FLOPs 远大于基线时，涨点可能
    #    来自容量而不是机制；必须提示等参数/等计算对照。
    challengers = [
        row
        for row in rows
        if str(row.get("mode")) != baseline_mode
        and str(row.get("mode")) != "ngram"
        and _finite_float(row.get(metric)) is not None
    ]
    capacity_findings: List[str] = []
    if challengers:
        def _finite_cost(value: object) -> Optional[float]:
            return _finite_float(value)

        best_challenger = (
            max(
                challengers,
                key=lambda row: _finite_float(
                    row.get(metric)
                ),
            )
            if higher_is_better
            else min(
                challengers,
                key=lambda row: _finite_float(
                    row.get(metric)
                ),
            )
        )
        for field, label in (
            ("n_params", "参数量"),
            ("flops_per_forward", "前向 FLOPs"),
        ):
            baseline_costs = [
                cost
                for row in baseline_rows
                for cost in [_finite_float(row.get(field))]
                if cost is not None and cost > 0.0
            ]
            baseline_cost = _mean(baseline_costs)
            challenger_cost = _finite_cost(
                best_challenger.get(field)
            )
            if (
                baseline_cost is None
                or challenger_cost is None
                or challenger_cost <= 0.0
            ):
                continue
            ratio = challenger_cost / baseline_cost
            if ratio >= _COST_RATIO_MATERIAL:
                capacity_findings.append(
                    f"{label}是基线的 {ratio:.2f} 倍"
                    f"（挑战模型 {challenger_cost:.3g} vs 基线 "
                    f"{baseline_cost:.3g}，"
                    f"{best_challenger.get('run')}）"
                )
    if capacity_findings:
        message = (
            "挑战模型与所选基线的容量/计算量不对齐（"
            + "; ".join(capacity_findings)
            + "）：涨点可能来自模型容量而不是待验证机制；"
            "请报告等参数对照（dense 用 --dense-fnn-hidden "
            "n_experts×expert_hidden）或等计算对照（如 smelt / "
            "local_base），并在结论中区分容量与机制"
        )
        report["advisories"].append(message)
        report["capacity_confounds"] = capacity_findings
    else:
        report["capacity_confounds"] = []

    # 6b) dense 基线的容量口径：默认 dense_fnn_hidden=None 取
    #     top_k×expert_hidden，是「等激活计算」而不是「等参数」。
    #     挑战模型参数量更大时，即使 FLOPs 对齐，涨点仍可能来自
    #     参数容量。明确给出等参数容量与重跑建议，避免把容量
    #     差异误读成机制收益。
    dense_capacities: List[Tuple[int, int]] = []
    for row in baseline_rows:
        if str(row.get("mode")) != "dense":
            continue
        n_experts = row.get("n_experts")
        expert_hidden = row.get("expert_hidden")
        top_k = row.get("top_k")
        explicit = row.get("dense_fnn_hidden")
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in (n_experts, expert_hidden, top_k)
        ):
            continue
        effective = (
            int(explicit)
            if isinstance(explicit, int)
            and not isinstance(explicit, bool)
            and explicit > 0
            else int(top_k) * int(expert_hidden)
        )
        dense_capacities.append((
            effective,
            int(n_experts) * int(expert_hidden),
        ))
    if dense_capacities and capacity_findings:
        current_capacity = max(
            effective for effective, _ in dense_capacities
        )
        parameter_matched = max(
            matched for _, matched in dense_capacities
        )
        if current_capacity < parameter_matched:
            report["baseline_capacity"] = {
                "effective_dense_fnn_hidden": current_capacity,
                "parameter_matched_dense_fnn_hidden": (
                    parameter_matched
                ),
                "parameter_matched": False,
            }
            report["advisories"].append(
                "dense 基线按等激活计算容量配置"
                f"（dense_fnn_hidden={current_capacity} = "
                "top_k×expert_hidden），不是等参数容量"
                f"（应为 n_experts×expert_hidden="
                f"{parameter_matched}）。挑战模型参数量更大时，"
                "涨点仍可能来自容量；请用 --dense-fnn-hidden "
                f"{parameter_matched} 训练等参数基线，或同时"
                "报告等参数与等计算两条基线"
            )
        else:
            report["baseline_capacity"] = {
                "effective_dense_fnn_hidden": current_capacity,
                "parameter_matched_dense_fnn_hidden": (
                    parameter_matched
                ),
                "parameter_matched": True,
            }

    # 7) n-gram 统计基线对照的稳定性：单次运行的 n-gram 点估计
    #    没有种子间波动可估，拿它当「调优好的强基线」时结论可能
    #    被一次 val 网格搜索的随机性左右；种子数不足时提示。
    ngram_deterministic = bool(ngram_rows) and all(
        _is_deterministic_run(row) for row in ngram_rows
    )
    report["ngram_deterministic"] = ngram_deterministic
    if ngram_rows and ngram_deterministic:
        # ngram 的计数与固定窗口评估都是确定性的：重复运行
        # 不会产生新的独立样本。需要不确定性时应对评估窗口
        # 做 bootstrap，或用不同时间切分重复整个评估，而不是
        # 假装换 seed 就得到独立重复。
        report["advisories"].append(
            "n-gram 对照是确定性基线（换 seed 重跑结果相同）："
            "不能用多种子 Mean ± Std 估计其不确定性；如需"
            "误差线，请对评估窗口做 bootstrap（新版的 "
            "--eval-bootstrap 已为 ngram 输出区间），或在不同的"
            "时间切分上重复评估"
        )
        ngram_tuning_issues: List[str] = []
        for row in ngram_rows:
            tuning = row.get("ngram_tuning")
            if not isinstance(tuning, dict):
                continue
            run_name = str(row.get("run"))
            for advisory in tuning.get("advisories", []) or []:
                ngram_tuning_issues.append(
                    f"{run_name}（{advisory}）"
                )
        if ngram_tuning_issues:
            report["ngram_tuning_diagnostics"] = (
                ngram_tuning_issues
            )
            report["advisories"].append(
                "n-gram 调参诊断发现选择不确定性："
                + "; ".join(ngram_tuning_issues)
                + "。winner 的 val 指标带网格选择偏差，"
                "不能把微小领先当成基线已被充分调强"
            )
    elif ngram_rows:
        ngram_independent_seeds = _independent_seed_count(
            len(ngram_rows),
            [
                row.get("seed")
                for row in ngram_rows
                if row.get("seed") is not None
            ],
        )
        if ngram_independent_seeds < _RECOMMENDED_SEEDS:
            report["advisories"].append(
                f"n-gram 对照只有 {ngram_independent_seeds} 个运行"
                f"（建议 ≥{_RECOMMENDED_SEEDS}）：统计基线的点估计"
                "没有种子间波动可估，调参 winner 的微小优势可能"
                "落在其 val 选取噪声内；建议多种子重跑 n-gram "
                "并报告 Mean ± Std"
            )

    if report["straw_man"]:
        report["ok"] = False
    elif (
        reference_supported
        and checked == len(baseline_samples)
        and missing_reference == 0
    ):
        report["ok"] = True

    return report



# 假设「判定指标」的词表：假设必须写明用什么指标判定成败，
# 否则无法与评估口径对齐（「验证损失改善 2%」可检验，
# 「效果更好」不可检验）。中英文都覆盖，路由诊断类指标
# （Jaccard/熵）同样算有效判定口径。
_HYPOTHESIS_METRIC_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("loss", r"loss|损失|nll|负对数似然"),
    # RQ1 明确包含「收敛速度」；只有写成可判定的曲线口径，
    # 「收敛更快」才不是不可检验的定性说法。
    (
        "convergence",
        r"收敛(?:速度|更快|曲线|轨迹)?|convergence|converge",
    ),
    # 类平衡似然：假设点名它时，判定口径必须与频率加权的
    # val_loss 区分开（不平衡场景下二者结论可能相反）。
    (
        "macro_nll",
        r"macro[\s_-]?nll|class[\s_-]?balanced|类平衡|按类等权",
    ),
    ("ppl", r"ppl|perplexity|困惑度"),
    ("bits_per_byte", r"bits[\s_-]?per[\s_-]?byte|bpb|每字节比特"),
    ("accuracy", r"accuracy|acc\b|准确率|正确率"),
    ("f1", r"f1|f-?score|f分数"),
    (
        "pr_auc_lift",
        r"pr[\s_-]?auc.{0,16}(?:lift|随机)"
        r"|average[\s_-]?precision.{0,16}lift"
        r"|lift.{0,8}(?:over|above).{0,8}chance|高出随机",
    ),
    ("pr_auc", r"pr[\s_-]?auc|average[\s_-]?precision|平?均精度|精确率-召回率"),
    ("roc_auc", r"roc[\s_-]?auc|auroc"),
    ("precision", r"precision|精确率|查准率"),
    ("recall", r"recall|召回率|查全率"),
    ("jaccard", r"jaccard|雅卡尔|重合度|重合率"),
    ("entropy", r"entropy|熵"),
    ("routing_load", r"负载|dead[\s_-]?expert|路由"),
)


# 假设词表 → 当前评估/对比管线真正可判定的指标集合。
# 只有这里列出的名字才算「假设写明了判定指标」：如果某个
# 名字没有映射（或映射到的指标不在 COMPARISON_METRIC_KEYS
# 里），登记时会单独点名「该指标当前没有评估口径」，避免
# 假设自称用某个指标判定、评估管线却从不计算它（假性对齐）。
_HYPOTHESIS_METRIC_SUPPORT: Dict[str, Tuple[str, ...]] = {
    "loss": (
        "val_loss",
        "val_loss_at_matched_flops",
        _CURVE_MEAN_METRIC,
    ),
    "convergence": (_CURVE_MEAN_METRIC,),
    "macro_nll": ("val_macro_nll",),
    "ppl": ("val_ppl",),
    "bits_per_byte": ("val_bits_per_byte",),
    "accuracy": ("val_token_accuracy",),
    "f1": (
        "val_micro_f1",
        "val_macro_f1",
        "val_weighted_f1",
    ),
    "pr_auc_lift": ("val_pr_auc_macro_lift",),
    "pr_auc": ("val_pr_auc_macro",),
    "roc_auc": ("val_roc_auc_macro",),
    "precision": (
        "val_precision_macro",
        "val_precision_weighted",
    ),
    "recall": (
        "val_recall_macro",
        "val_recall_weighted",
    ),
    "jaccard": (
        "routing_jaccard_branch",
        "routing_jaccard_token",
    ),
    "entropy": (
        "routing_r1_load_entropy",
        "routing_r2_load_entropy",
        "routing_r2_router_entropy",
    ),
    "routing_load": (
        "routing_slot_reuse_branch",
        "routing_branch_self_reuse",
        "routing_r2_dead_experts",
    ),
}



def _hypothesis_supported_metrics(name: str) -> Tuple[str, ...]:
    """词表名字对应的可判定指标，含共同步数派生口径。"""
    expanded: List[str] = []
    for metric in _HYPOTHESIS_METRIC_SUPPORT.get(name, ()):
        if metric not in expanded:
            expanded.append(metric)
        common_step = _common_step_metric_name(metric)
        if common_step is not None and common_step not in expanded:
            expanded.append(common_step)
    return tuple(expanded)



# 路由诊断类判定指标依赖 routing_analysis：dense 基线、
# --skip-routing-analysis 或旧运行没有该记录时，假设无法被
# 当前运行的数据判定。
_HYPOTHESIS_ROUTING_METRICS = frozenset(_ROUTING_ROW_FIELDS)


# 多分类下 F1 / precision / recall 必须注明平均口径（macro、
# micro、weighted 三者结果可能差异很大）。假设只写「F1 提升」
# 无法与评估口径对齐：同一个实验在三种口径下可能一个显著、
# 一个不显著。命中下列指标名但通篇没有任何口径限定词时，
# 假设被单独点名。
_HYPOTHESIS_AVERAGING_METRICS = ("f1", "precision", "recall")

_HYPOTHESIS_AVERAGING_QUALIFIER = re.compile(
    r"macro|micro|weighted|宏|微|加权",
    re.IGNORECASE,
)


# 预注册主指标若已经带平均口径（如 val_macro_f1），假设文本
# 写明的口径必须与之兼容：只写「F1」算未注明（由
# hypothesis_ambiguous_averaging_metrics 单独点名），写
# 「macro F1」却预注册 weighted F1 则是明确的口径冲突。
_HYPOTHESIS_AVERAGING_METRIC_QUALIFIERS = {
    "val_micro_f1": "micro",
    "val_macro_f1": "macro",
    "val_weighted_f1": "weighted",
    "val_precision_macro": "macro",
    "val_precision_weighted": "weighted",
    "val_recall_macro": "macro",
    "val_recall_weighted": "weighted",
}

_HYPOTHESIS_AVERAGING_QUALIFIER_PATTERNS = (
    ("macro", re.compile(r"macro|宏(?:平均|均值)?", re.IGNORECASE)),
    ("micro", re.compile(r"micro|微(?:平均|均值)?", re.IGNORECASE)),
    ("weighted", re.compile(r"weighted|加权", re.IGNORECASE)),
)



def hypothesis_metric_averaging_qualifier(
    metric: str,
) -> Optional[str]:
    """返回指标名自带的平均口径（macro/micro/weighted）。"""
    base = _common_step_metric_base(metric) or metric
    return _HYPOTHESIS_AVERAGING_METRIC_QUALIFIERS.get(base)



def hypothesis_averaging_qualifier_mismatch(
    primary_metric: str,
    text: str,
) -> Optional[str]:
    """
    检查假设文本写明的平均口径是否与预注册主指标冲突。

    只处理「文本明确写了口径」的情况：没写任何口径时不在这里
    报错（另行提示口径歧义），避免把「F1 提升」误判成冲突。
    文本同时写了多个口径时，只要包含预注册指标的口径就算兼容
    （例如同时讨论 macro F1 与 weighted recall）。
    """
    expected = hypothesis_metric_averaging_qualifier(
        primary_metric
    )
    if expected is None:
        return None
    found = [
        name
        for name, pattern in _HYPOTHESIS_AVERAGING_QUALIFIER_PATTERNS
        if pattern.search(text)
    ]
    if not found or expected in found:
        return None
    return (
        f"预注册主指标 {primary_metric} 的平均口径是 "
        f"{expected}，但假设文本写的是 {', '.join(found)}："
        "假设与判定口径必须一致"
    )



def hypothesis_metric_mentions(text: str) -> List[str]:
    """
    返回假设文本中出现的判定指标（规范化名字列表）。

    用于核对「假设 ↔ 评估口径」是否对齐：一个可检验的假设
    必须写明依据什么指标判定成败（如「验证损失至少改善 2%」
    或「macro F1 提升」），否则只能说「效果更好」，无法与
    评估结果对照。
    """
    lowered = text.lower()
    mentions: List[str] = []
    for name, pattern in _HYPOTHESIS_METRIC_PATTERNS:
        if re.search(pattern, lowered) and name not in mentions:
            mentions.append(name)
    return mentions



def hypothesis_metric_unverifiable(
    mentions: List[str],
    routing_available: Optional[bool] = None,
) -> List[str]:
    """
    返回假设提到的判定指标中「当前评估管线无法判定」的名字。

    - 词表名字没有映射到 COMPARISON_METRIC_KEYS 的指标时，
      整个名字都不可判定；
    - 映射到的指标全是路由诊断（``routing_*``）且本次运行
      没有 routing_analysis 记录（dense 基线、显式跳过或
      旧运行；``routing_available=False``）时，同样不可判定；
      ``routing_available=None`` 表示未知（如登记阶段尚未
      执行诊断），不做否定判定。
    """
    unverifiable: List[str] = []
    for name in mentions:
        supported = [
            metric
            for metric in _hypothesis_supported_metrics(name)
            if metric in COMPARISON_METRIC_KEYS
        ]
        if not supported:
            unverifiable.append(name)
            continue
        if (
            routing_available is False
            and all(
                metric in _HYPOTHESIS_ROUTING_METRICS
                for metric in supported
            )
        ):
            unverifiable.append(name)
    return unverifiable



def hypothesis_ambiguous_averaging_metrics(
    text: str,
    mentions: Optional[List[str]] = None,
) -> List[str]:
    """
    返回假设中「未注明平均口径」的多分类指标名。

    单标签多分类下 F1 / precision / recall 的 macro、micro、
    weighted 三种平均口径可能给出显著不同的结论，假设只写
    「F1 提升」无法与评估口径对齐；文本里出现口径限定词
    （macro / micro / weighted / 宏平均 / 微平均 / 加权）才
    算写明。未命中任何歧义指标时返回空列表。
    """
    if mentions is None:
        mentions = hypothesis_metric_mentions(text)
    if not any(
        name in _HYPOTHESIS_AVERAGING_METRICS
        for name in mentions
    ):
        return []
    if _HYPOTHESIS_AVERAGING_QUALIFIER.search(text):
        return []
    return [
        name
        for name in _HYPOTHESIS_AVERAGING_METRICS
        if name in mentions
    ]
