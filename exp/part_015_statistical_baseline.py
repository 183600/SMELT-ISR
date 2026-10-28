from __future__ import annotations



def run_statistical_baseline(
    args: argparse.Namespace,
    dataset: ByteDataset,
    out_dir: Path,
    device: torch.device,
    data_audit: Optional[Dict[str, object]],
    data_protocol_check: Optional[Dict[str, object]] = None,
) -> None:
    """
    统计基线入口：计数（train）→ val 网格调优
    （阶数 × 平滑系数）→ 选出最优配置 → 与神经模型同口径
    的 val（及一次性 test）指标，写入与训练运行相同结构的
    config.json / metrics.jsonl，可直接进入 --compare。
    """
    del device  # n-gram 全部在 CPU 上完成。

    # 统计基线当前不消耗任何随机性（计数与固定窗口评估都是
    # 确定性的），但所有入口统一固定全部 RNG 种子：未来任何
    # 采样/初始化改动不会让本入口在无感知的情况下失去
    # 可复现性，config.json 里的种子也始终对应真实生效值。
    set_seed(
        args.seed,
        strict_determinism=args.strict_determinism,
    )

    # 在入口处捕获运行身份：final 记录使用真正的启动时间，
    # 而不是训练/计数结束后的时间。
    ngram_run_metadata = collect_run_metadata(
        torch.device("cpu")
    )

    alpha_grid: List[float] = list(args.ngram_alpha_grid)
    tuning_metric, tuning_metric_source = (
        resolve_ngram_tuning_metric(args)
    )

    excluded_bytes = (
        dataset.train.numel() - dataset.effective_train_bytes
    )
    print(
        "[ngram] 在 train split 上计数 "
        f"({dataset.effective_train_bytes:,} 有效字节"
        + (
            f", 已剔除 {excluded_bytes:,} 字节重复群组"
            if excluded_bytes > 0
            else ""
        )
        + f", max_order={args.ngram_max_order}, "
        f"count_cap={_NGRAM_COUNT_CAP:,})"
    )

    model = NgramBaseline(
        dataset.train,
        max_order=args.ngram_max_order,
        exclude_spans=dataset.train_excluded_spans,
    )

    val_windows = _ngram_eval_batches(
        dataset,
        "val",
        args.batch_size,
        args.eval_batches,
    )

    if not val_windows:
        raise SystemExit(
            "验证集固定窗口为空，无法调优统计基线"
        )

    # ---- 参考基线（uniform / unigram），与训练运行同款 ----
    reference = compute_reference_baselines(
        dataset,
        args.batch_size,
        args.eval_batches,
        torch.device("cpu"),
        split="val",
    )

    print(
        "[ngram] 参考基线: uniform_loss="
        f"{reference['uniform_loss']:.4f}, "
        "unigram_loss="
        f"{reference['unigram_loss']:.4f}"
    )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = open(
        metrics_path,
        "w",
        encoding="utf-8",
    )

    def _write(record: Dict[str, object]) -> None:
        metrics_file.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
            )
            + "\n"
        )
        metrics_file.flush()

    _write({
        "step": "reference_baselines",
        **tracking_header_fields(
            args,
            dataset,
            ngram_run_metadata,
        ),
        **reference,
    })

    # ---- 网格调优：α 每档一遍窗口，同时拿到所有阶数损失 ----
    # 逐阶查表只依赖窗口与训练计数，与 α 无关。
    val_tables = [
        model._per_level_tables(window)
        for window in val_windows
    ]

    candidates: List[Dict[str, object]] = []
    candidate_window_losses: Dict[
        Tuple[int, float], List[float]
    ] = {}
    # 分类调参指标（accuracy / F1 / precision / recall）的逐窗口
    # 同口径取值：供 one-SE 选择按窗口配对，而不是只看聚合后
    # 的微小差距。
    candidate_window_classification: Dict[
        Tuple[int, float], List[float]
    ] = {}
    loss_like_metrics = (
        "val_loss",
        "val_ppl",
        "val_bits_per_byte",
    )
    classification_metric_key = tuning_metric[
        len("val_"):
    ]
    classification_tunable = (
        tuning_metric not in loss_like_metrics
        and classification_metric_key
        in _NGRAM_PER_WINDOW_CLASSIFICATION_METRICS
    )
    higher_is_better = metric_higher_is_better(
        tuning_metric
    )
    print(
        "[ngram] (order, alpha) 的选择指标: "
        f"{tuning_metric}"
        f"（来源={tuning_metric_source}，"
        + (
            "越高越好"
            if higher_is_better
            else "越低越好"
        )
        + "；test 集不参与）"
    )

    for alpha in alpha_grid:
        # log_probs 是 float64（见 _interpolate_log_probs）；累加器
        # 必须同 dtype，否则 += 会把双精度的逐窗口损失静默降为
        # float32，val_loss 与 _ngram_extended_metrics 的 float64
        # 同口径值不一致，还可能改变 (order, alpha) 的选优结果。
        totals = torch.zeros(model.max_order, dtype=torch.float64)
        tokens = 0

        for window, tables in zip(
            val_windows,
            val_tables,
        ):
            log_probs = (
                model._interpolate_log_probs(
                    window,
                    tables,
                    alpha,
                )
            )
            totals += -log_probs.sum(dim=1)
            tokens += log_probs.size(1)
            if tuning_metric in loss_like_metrics:
                # 逐窗口损失用于 one-SE 选择：同一批窗口上
                # 候选与点估计 winner 可以按窗口配对，避免
                # 用聚合后的小差距把噪声当成稳定收益。
                seq_len = int(window.size(1)) - 1
                window_means = (
                    (-log_probs)
                    .reshape(
                        model.max_order,
                        int(window.size(0)),
                        seq_len,
                    )
                    .sum(dim=2)
                    / float(seq_len)
                )
                for order in range(
                    1, model.max_order + 1
                ):
                    candidate_window_losses.setdefault(
                        (order, alpha),
                        [],
                    ).extend(
                        window_means[order - 1].tolist()
                    )

        for order in range(1, model.max_order + 1):
            val_loss = float(
                totals[order - 1] / tokens
            )
            record = {
                "step": "baseline_sweep",
                "order": order,
                "alpha": alpha,
                "val_loss": val_loss,
                "val_ppl": math.exp(
                    min(val_loss, 80.0)
                ),
                "val_bits_per_byte": val_loss
                / math.log(2.0),
                "tokens_evaluated": tokens,
                "tuning_metric": tuning_metric,
            }

            if tuning_metric not in loss_like_metrics:
                # 分类/AUC 指标必须看完整的 argmax/概率分布：
                # 在 val_loss 上选出的 (order, alpha) 不能证明
                # 它在 macro F1 / PR-AUC 上也最优。
                class_window_records: List[
                    Dict[str, object]
                ] = []
                candidate_metrics = _ngram_extended_metrics(
                    model,
                    val_windows,
                    alpha,
                    order,
                    split="val",
                    include_auc=tuning_metric
                    in (
                        "val_pr_auc_macro",
                        "val_pr_auc_macro_lift",
                        "val_roc_auc_macro",
                    ),
                    window_sink=(
                        class_window_records
                        if classification_tunable
                        else None
                    ),
                    window_classification=(
                        classification_tunable
                    ),
                )
                if classification_tunable:
                    values: List[float] = []
                    usable = True
                    for window_record in class_window_records:
                        classification = window_record.get(
                            "classification"
                        )
                        if not isinstance(
                            classification, dict
                        ):
                            usable = False
                            break
                        number = _finite_float(
                            classification.get(
                                classification_metric_key
                            )
                        )
                        if number is None:
                            usable = False
                            break
                        values.append(number)
                    if usable and len(values) >= 3:
                        candidate_window_classification[
                            (order, alpha)
                        ] = values
                for key in (
                    "val_loss",
                    "val_ppl",
                    "val_bits_per_byte",
                ):
                    if key in candidate_metrics:
                        record[key] = candidate_metrics[key]
                number = _finite_float(
                    candidate_metrics.get(tuning_metric)
                )
                if number is not None:
                    record[tuning_metric] = number

            record["tuning_value"] = record.get(tuning_metric)
            _write(record)
            candidates.append(record)

        def _display_value(row: Dict[str, object]) -> str:
            value = _finite_float(row.get(tuning_metric))
            if value is None:
                return "NA"
            return f"{value:.4f}"

        print(
            "[ngram] alpha="
            f"{alpha:g}: "
            + ", ".join(
                f"order {row['order']}="
                + _display_value(row)
                for row in candidates[
                    -model.max_order:
                ]
            )
            + f" ({tuning_metric})"
        )

    # val 上选最优；并列时偏向更小的阶数与更小的 alpha
    # （简单优先，避免在验证集上过拟合出复杂配置）。
    def _candidate_rank(row: Dict[str, object]):
        value = _finite_float(row.get(tuning_metric))
        if value is None:
            return (1, 0.0, row["order"], row["alpha"])
        return (
            0,
            -value if higher_is_better else value,
            row["order"],
            row["alpha"],
        )

    valid_candidates = [
        row for row in candidates if _candidate_rank(row)[0] == 0
    ]
    if not valid_candidates:
        raise SystemExit(
            f"n-gram 调参没有任何候选提供可用的 {tuning_metric}："
            "无法按当前指标选出 (order, alpha)；请检查指标"
            "是否适用于该评估集，或显式改用 val_loss"
        )
    ranked_candidates = sorted(
        valid_candidates,
        key=_candidate_rank,
    )
    point_best = ranked_candidates[0]
    best = point_best
    runner_up = (
        ranked_candidates[1]
        if len(ranked_candidates) > 1
        else None
    )

    # 点估计最优可能只领先很小一截；损失类与分类指标都用
    # 逐窗口配对 one-SE 规则，把统计上不可分辨的候选按更简单
    # 优先收敛，降低网格调参的 winner's curse（AUC 除外，逐
    # 窗口类支持稀疏、与全局口径不一致）。
    selection = _select_ngram_candidate_one_se(
        point_best,
        valid_candidates,
        (
            candidate_window_classification
            if classification_tunable
            else candidate_window_losses
        ),
        tuning_metric,
    )
    if (
        selection is not None
        and selection.get("available")
        and not selection.get("selected_is_point_best")
    ):
        selected_order = int(selection["selected"]["order"])
        selected_alpha = float(selection["selected"]["alpha"])
        selected_candidate = next(
            (
                candidate
                for candidate in valid_candidates
                if int(candidate["order"]) == selected_order
                and abs(
                    float(candidate["alpha"]) - selected_alpha
                )
                <= 1e-15
            ),
            None,
        )
        if selected_candidate is not None:
            best = selected_candidate
            runner_up = point_best

    best_order = int(best["order"])
    best_alpha = float(best["alpha"])
    tuning_diagnostics = _ngram_tuning_diagnostics(
        best,
        runner_up,
        tuning_metric,
        alpha_grid,
        args.ngram_max_order,
        len(candidates),
        len(valid_candidates),
        selection=selection,
    )

    if (
        selection is not None
        and selection.get("available")
        and not selection.get("selected_is_point_best")
    ):
        selected_detail = next(
            (
                detail
                for detail in selection.get("tied", [])
                if int(detail["order"]) == best_order
                and abs(
                    float(detail["alpha"]) - best_alpha
                )
                <= 1e-15
            ),
            None,
        )
        if isinstance(selected_detail, dict):
            print(
                "[ngram] one-SE 选择更简单配置: "
                f"order={best_order}, alpha={best_alpha:g}；"
                "相对点估计最优的逐窗口配对差="
                f"{float(selected_detail['paired_mean_diff']):.6g}"
                "（阈值="
                f"{float(selected_detail['threshold']):.6g}）"
            )

    print(
        "[ngram] val 调优结果: order="
        f"{best_order}, alpha={best_alpha:g}, "
        f"{tuning_metric}="
        f"{float(best[tuning_metric]):.4f}, "
        f"val_loss={best['val_loss']:.4f}"
    )
    if tuning_diagnostics.get("metric_gap") is not None:
        print(
            "[ngram] winner 与 runner-up 的 "
            f"{tuning_metric} 差距="
            f"{float(tuning_diagnostics['metric_gap']):.6g}"
        )
    for advisory in tuning_diagnostics.get(
        "advisories", []
    ):
        print(f"[ngram][warn] {advisory}")

    # ---- 用调优好的配置算同口径完整指标 ----
    # 统计基线是确定性的：评估不确定性只能来自评估窗口抽样，
    # 因此与神经模型同口径给出逐窗口 bootstrap 区间，并落盘
    # 逐窗口记录供跨运行配对比较。
    bootstrap_replicates = int(
        getattr(args, "eval_bootstrap", 0)
    )
    bootstrap_block = int(
        getattr(args, "eval_bootstrap_block", 1)
    )
    val_window_records: List[Dict[str, float]] = []
    final_metrics = _ngram_extended_metrics(
        model,
        val_windows,
        best_alpha,
        best_order,
        split="val",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=_eval_bootstrap_seed(
            dataset.data_sha256,
            "val",
            args.eval_batches,
            args.batch_size,
            dataset.seq_len,
        ),
        bootstrap_block=bootstrap_block,
        window_sink=val_window_records,
    )

    final_record = {
        "step": "final",
        "model": "ngram",
        "ngram_order": best_order,
        "ngram_alpha": best_alpha,
        "ngram_max_order": args.ngram_max_order,
        "ngram_tuning_metric": tuning_metric,
        "ngram_tuning_metric_source": tuning_metric_source,
        "ngram_tuning": tuning_diagnostics,
        # 统计基线同样自带实验追踪字段，便于 metrics.jsonl
        # 脱离 config.json 单独归档。
        "deterministic": True,
        "seed": args.seed,
        "data_sha256": dataset.data_sha256,
        "data_split_sha256": dict(dataset.split_sha256),
        "data_effective_train_sha256": (
            dataset.effective_train_sha256
        ),
        "script_sha256": ngram_run_metadata.get(
            "script_sha256"
        ),
        "git_commit": ngram_run_metadata.get("git_commit"),
        "git_dirty": ngram_run_metadata.get("git_dirty"),
        "run_started_utc": ngram_run_metadata.get(
            "timestamp_utc"
        ),
        "run_completed_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        **final_metrics,
    }

    _write(final_record)
    print(
        json.dumps(
            _json_safe({"final": final_record}),
            ensure_ascii=False,
        )
    )

    if val_window_records:
        overlap_fraction: Optional[float] = None
        starts = dataset._fixed_starts.get(
            "val", {}
        ).get("starts")
        if starts is not None and len(starts) > 1:
            gaps = starts[1:] - starts[:-1]
            overlap_count = int(
                (gaps < dataset.seq_len).sum().item()
            )
            overlap_fraction = overlap_count / (
                len(starts) - 1
            )
        eval_windows_payload: Dict[str, object] = {
            "split": "val",
            "data_sha256": dataset.data_sha256,
            "split_sha256": dataset.split_sha256.get("val"),
            "seq_len": int(dataset.seq_len),
            "batch_size": int(args.batch_size),
            "batches": int(args.eval_batches),
            "n_windows": len(val_window_records),
            "overlap_fraction": overlap_fraction,
            "windows": val_window_records,
        }
        with open(
            out_dir / "eval_windows.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_safe(eval_windows_payload),
                f,
                ensure_ascii=False,
            )

    # ---- 一次性 test 评估（沿用训练运行同一封存协议）----
    # 注意：本入口每次运行都会新建输出目录语义（metrics.jsonl
    # 截断重写），test 只在显式 --evaluate-test 时评估一次，
    # 且 (order, alpha) 完全沿用 val 上调优的结果，绝不在
    # test 上二次调参。
    if args.evaluate_test:
        if not dataset.has_test:
            raise SystemExit(
                "--evaluate-test 需要有效的测试集："
                "请设置正的 --test-fraction 且语料足够长"
            )

        if test_evaluation_sealed(
            out_dir,
            getattr(args, "test_ledger", None),
        ):
            raise SystemExit(
                "[test] 输出目录已有测试集评估记录/封条；"
                "统计基线也遵守唯一一次解封协议，请使用全新的 "
                "--out 目录"
            )

        test_audit = audit_data_splits(
            dataset,
            include_test=True,
        )
        print_data_audit(test_audit)

        leakage_blocked = (
            not args.allow_data_leakage
            and not enforce_data_leakage_gate(
                test_audit,
                "test",
            )
        )
        eval_duplicate_blocked = (
            not getattr(
                args, "allow_eval_duplicates", False
            )
            and not enforce_evaluation_duplicate_gate(
                test_audit,
                "test",
            )
        )
        strict_blocked = (
            args.strict_data_audit
            and not enforce_strict_data_audit(
                test_audit,
                "test",
            )
        )
        label_integrity_blocked = (
            not args.allow_corrupt_test_labels
            and not enforce_test_label_integrity(
                test_audit,
                "test",
            )
        )
        shift_blocked = (
            not enforce_distribution_shift_gate(
                test_audit,
                max_byte_js=args.max_byte_shift,
                max_bigram_js=args.max_bigram_shift,
                max_conditional_js=args.max_conditional_shift,
                context="test",
            )
        )
        if (
            leakage_blocked
            or eval_duplicate_blocked
            or strict_blocked
            or shift_blocked
            or label_integrity_blocked
        ):
            # 统计基线同样不能在泄露/损坏/分布偏移的测试集上
            # 报告指标；不写 test 记录，名额未消耗。
            print(
                "[test] 数据审计在测试集上发现泄露/损坏/"
                "标签噪声或分布偏移问题：拒绝解封测试集"
                "（不写入 test 记录）。请先清洗数据后重跑。"
            )
            test_windows = None
        else:
            test_protocol_check = probe_data_pipeline(
                dataset,
                torch.device("cpu"),
                include_test=True,
            )
            if not test_protocol_check["ok"]:
                print(
                    "[test] 测试集数据管线检查未通过，"
                    "拒绝解封（不写入 test 记录）："
                )
                for failure in test_protocol_check[
                    "failures"
                ]:
                    print(f"  - {failure}")
                test_windows = None
            else:
                # 研究级访问账本与神经路径同一协议：路径级封条
                # 挡不住「新建目录/换 seed 后再次访问同一测试
                # 集」。统计基线也是候选，必须进入同一账本，
                # 否则 --final-test-comparison 会漏报它的访问。
                ledger_path = (
                    Path(args.test_ledger)
                    if getattr(args, "test_ledger", None)
                    else out_dir / "test_access_ledger.jsonl"
                )
                if getattr(args, "test_ledger", None) is None:
                    print(
                        "[test] [warn] 未指定 --test-ledger：访问"
                        "账本只写在本次输出目录，无法阻止其它目录"
                        "重复解封同一测试集"
                    )
                access_key = _test_access_key(
                    dataset.data_sha256,
                    dataset.split_sha256,
                    dataset.test_fraction,
                )
                access_check = check_test_access_ledger(
                    ledger_path,
                    access_key,
                    out_dir,
                )
                if _test_access_repeat_blocked(
                    access_check
                ) and not getattr(
                    args,
                    "allow_repeat_test_access",
                    False,
                ):
                    raise SystemExit(
                        _repeat_test_access_error(access_check)
                    )
                append_test_access_ledger(
                    ledger_path,
                    {
                        **access_key,
                        "out_dir": str(out_dir),
                        "study_id": getattr(
                            args, "study_id", None
                        ),
                        "seed": args.seed,
                        "model": "ngram",
                        "ngram_order": best_order,
                        "ngram_alpha": best_alpha,
                        "script_sha256": ngram_run_metadata.get(
                            "script_sha256"
                        ),
                        "git_commit": ngram_run_metadata.get(
                            "git_commit"
                        ),
                        "hypothesis": getattr(
                            args, "hypothesis", None
                        ),
                        "primary_metric": getattr(
                            args, "primary_metric", None
                        ),
                        "allow_repeat_access": bool(
                            getattr(
                                args,
                                "allow_repeat_test_access",
                                False,
                            )
                        ),
                        "accessed_utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        "note": (
                            "统计基线测试集访问账本；同一 "
                            "data/split 指纹的全部访问都应进入"
                            "最终报告"
                        ),
                    },
                )
                print(
                    "[test] 访问账本已记录: "
                    f"{ledger_path}（历史访问 "
                    f"{access_check['matching_accesses'] + 1} 次）"
                )

                # 与神经训练路径同一封存协议：先落
                # test_started 记录与持久封条，再访问测试
                # 窗口；评估中途崩溃也不能重复解封。
                _write({
                    "step": "test_started",
                    "split": "test",
                    "model": "ngram",
                })
                write_test_evaluation_seal(
                    out_dir,
                    {
                        "dataset_sha256": (
                            dataset.data_sha256
                        ),
                        "test_fraction": (
                            dataset.test_fraction
                        ),
                        "split_sha256": dict(
                            dataset.split_sha256
                        ),
                        "model": "ngram",
                    },
                )
                test_windows = _ngram_eval_batches(
                    dataset,
                    "test",
                    args.batch_size,
                    args.eval_batches,
                )

        if test_windows is not None:
            test_metrics = _ngram_extended_metrics(
                model,
                test_windows,
                best_alpha,
                best_order,
                split="test",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_eval_bootstrap_seed(
                    dataset.data_sha256,
                    "test",
                    args.eval_batches,
                    args.batch_size,
                    dataset.seq_len,
                ),
                bootstrap_block=bootstrap_block,
            )

            _write({
                "step": "test",
                "model": "ngram",
                "ngram_order": best_order,
                "ngram_alpha": best_alpha,
                "ngram_tuning_metric": tuning_metric,
                **test_metrics,
                "data_audit": test_audit,
                "data_protocol_check": test_protocol_check,
            })

            print(
                json.dumps(
                    _json_safe({"test": test_metrics}),
                    ensure_ascii=False,
                )
            )

    metrics_file.close()

    try:
        environment_end = write_run_environment_end_record(
            out_dir,
            ngram_run_metadata,
        )
    except (OSError, ValueError) as error:
        environment_end = None
        print(
            "[env][warn] 无法写入运行结束环境快照: "
            f"{error}"
        )

    # ---- config.json：与训练运行相同结构，可直接 --compare ----
    payload = {
        # 与 write_run_config 使用同一配置 schema；统计基线也是
        # 独立实验单元，归档工具不应遇到两种结构。
        "config_schema": _ENVIRONMENT_CONFIG_SCHEMA,
        # 计数与固定窗口评估都不消耗随机性：显式标记，供
        # --compare 折叠多个 seed 的复制运行（不能当独立重复）。
        "deterministic": True,
        "config": {
            "mode": "ngram",
            "vocab_size": 256,
            "seq_len": dataset.seq_len,
            "ngram_max_order": args.ngram_max_order,
            "ngram_alpha_grid": alpha_grid,
            "ngram_order": best_order,
            "ngram_alpha": best_alpha,
            "ngram_tuning_metric": tuning_metric,
            "ngram_tuning_metric_source": (
                tuning_metric_source
            ),
            "ngram_tuning": tuning_diagnostics,
        },
        # 统计基线没有优化器与训练轮数；comparison_family 让
        # 受控变量检查只在神经网络运行之间进行。
        "comparison_protocol": {
            "comparison_family": "statistical",
            "optimizer": "none",
            "lr": None,
            "lr_schedule": "none",
            "warmup_steps": None,
            "total_steps": None,
            "weight_decay": None,
            "grad_clip": None,
            "batch_size": args.batch_size,
            "data_augmentation": "none",
            "seq_len": dataset.seq_len,
            "val_fraction": dataset.val_fraction,
            "test_fraction": dataset.test_fraction,
            "group_separator": getattr(args, "group_separator", None),
            "deduplicate_train_groups": bool(
                getattr(
                    args,
                    "deduplicate_train_groups",
                    False,
                )
            ),
            "drop_train_shared_chunks": bool(
                getattr(
                    args,
                    "drop_train_shared_chunks",
                    False,
                )
            ),
            "drop_train_near_duplicate_groups": bool(
                getattr(
                    args,
                    "drop_train_near_duplicate_groups",
                    False,
                )
            ),
            "near_duplicate_threshold": (
                _near_duplicate_threshold_or_default(
                    getattr(
                        args,
                        "near_duplicate_threshold",
                        _NEAR_DUP_DEFAULT_THRESHOLD,
                    )
                )
            ),
            "near_duplicate_shingle_bytes": (
                _near_duplicate_shingle_or_default(
                    getattr(
                        args,
                        "near_duplicate_shingle_bytes",
                        _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
                    )
                )
            ),
            "clean_train_against_test": bool(
                getattr(
                    args,
                    "clean_train_against_test",
                    False,
                )
            ),
            "drop_train_corrupt_groups": bool(
                getattr(
                    args,
                    "drop_train_corrupt_groups",
                    False,
                )
            ),
            "drop_train_corrupt_bytes": bool(
                getattr(
                    args,
                    "drop_train_corrupt_bytes",
                    False,
                )
            ),
            "drop_train_degenerate_groups": bool(
                getattr(
                    args,
                    "drop_train_degenerate_groups",
                    False,
                )
            ),
            "drop_train_degenerate_runs": bool(
                getattr(
                    args,
                    "drop_train_degenerate_runs",
                    False,
                )
            ),
            "eval_content_access": (
                getattr(dataset, "eval_content_access", None)
                if dataset is not None
                else None
            ),
            "cleaning_domain_justification": getattr(
                args, "cleaning_domain_justification", None
            ),
            "split_policy": getattr(
                args, "split_policy", "contiguous"
            ),
            "time_order_evidence": getattr(
                args, "time_order_evidence", None
            ),
            "entity_audit": getattr(
                args, "data_entity_audit", None
            ),
            "allow_data_leakage": bool(
                getattr(args, "allow_data_leakage", False)
            ),
            "allow_corrupt_test_labels": bool(
                getattr(args, "allow_corrupt_test_labels", False)
            ),
            "allow_eval_duplicates": bool(
                getattr(args, "allow_eval_duplicates", False)
            ),
            "eval_interval": None,
            "eval_batches": args.eval_batches,
            "eval_bootstrap": args.eval_bootstrap,
            "eval_bootstrap_block": args.eval_bootstrap_block,
            "data_sha256": dataset.data_sha256,
            # 与神经运行相同：逐 split 内容指纹用于核对
            # ngram 与神经运行是否真的看到同一组 train/val/test
            # 字节段；只比整体文件指纹无法发现群组吸附/清洗/
            # 切分实现差异。
            "split_sha256": dict(dataset.split_sha256) or None,
            # 清洗后的有效训练集指纹：ngram 计数必须与神经
            # 窗口覆盖同一套训练字节。
            "effective_train_sha256": (
                dataset.effective_train_sha256
            ),
            "split_isolation": dict(
                dataset.split_isolation
            ),
        },
        "args": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "data_sha256",
                "data_bytes",
                "data_val_fraction",
                "data_test_fraction",
            }
        },
        "n_params": 0,
        "data": {
            "path": args.data,
            "sha256": dataset.data_sha256,
            "bytes": dataset.num_bytes,
            "val_fraction": dataset.val_fraction,
            "test_fraction": dataset.test_fraction,
            "group_separator": getattr(args, "group_separator", None),
            "deduplicate_train_groups": bool(
                getattr(args, "deduplicate_train_groups", False)
            ),
            "drop_train_shared_chunks": bool(
                getattr(args, "drop_train_shared_chunks", False)
            ),
            "drop_train_near_duplicate_groups": bool(
                getattr(
                    args,
                    "drop_train_near_duplicate_groups",
                    False,
                )
            ),
            "near_duplicate_audit": bool(
                getattr(args, "near_duplicate_audit", False)
            ),
            "near_duplicate_threshold": (
                _near_duplicate_threshold_or_default(
                    getattr(
                        args,
                        "near_duplicate_threshold",
                        _NEAR_DUP_DEFAULT_THRESHOLD,
                    )
                )
            ),
            "near_duplicate_shingle_bytes": (
                _near_duplicate_shingle_or_default(
                    getattr(
                        args,
                        "near_duplicate_shingle_bytes",
                        _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
                    )
                )
            ),
            "split_bounds": {
                split: list(bounds)
                for split, bounds in dataset.split_bounds.items()
            },
            "split_isolation": dict(dataset.split_isolation),
            "split_fractions": dataset.split_fractions,
            "split_sha256": dict(dataset.split_sha256),
            "effective_train_bytes": (
                dataset.effective_train_bytes
            ),
            "effective_train_sha256": (
                dataset.effective_train_sha256
            ),
            "train_cleaning": dataset.train_cleaning,
            "group_cuts": (
                dict(dataset.group_cuts)
                if dataset.group_cuts is not None
                else None
            ),
            "groups_per_split": (
                dict(dataset.groups_per_split)
                if dataset.groups_per_split is not None
                else None
            ),
        },
        "data_audit": data_audit,
        "data_protocol_check": data_protocol_check,
        "reference_baselines": reference,
        "run_metadata": ngram_run_metadata,
        "sanity_check": None,
    }
    # 与神经训练入口一致：环境快照单独落一份 environment.json，
    # config.json 顶层只保留带文件哈希的摘要。统计基线也是独立
    # 实验单元，--check-environment 与归档清单必须能核对它产自
    # 哪套数值环境。
    environment_record = write_run_environment_record(
        out_dir,
        ngram_run_metadata,
    )
    if environment_record is not None:
        payload["environment"] = environment_record
    if environment_end is not None:
        payload["environment_end"] = environment_end
    payload["comparison_protocol_sha256"] = _sha256_text(
        json.dumps(
            _json_safe(payload["comparison_protocol"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    # 与训练入口一样原子写 config.json：计数/评估结束后写盘失败
    # 不能留下半截 JSON，否则跟踪信息与归档清单都会失效。
    _atomic_write_json(out_dir / "config.json", payload)



# ============================================================
# Training-curve analysis
# ============================================================

# 曲线诊断最少需要的验证点数：少于该数量时只报告原始曲线，
# 不给出过拟合/欠拟合结论，避免用一两个点下判断。
_CURVE_MIN_EVAL_POINTS = 3


# 验证损失从最优点回升超过该比例、且训练损失仍在下降时
# 判定为过拟合。
_CURVE_OVERFIT_REGRESSION = 0.02


# 判定「训练损失仍在下降」的最小相对降幅。
_CURVE_TRAIN_STILL_DESCENDING = 0.02


# 最近一次验证间隔内训练/验证损失仍下降超过该比例时，
# 判定为尚未收敛（继续训练可能还有收益）。
_CURVE_STILL_DESCENDING = 0.05


# 整体相对下降小于该比例、且最终训练损失仍接近随机基线
# （baseline_loss）时判定为欠拟合/几乎没有学到东西。
_CURVE_NO_LEARN_DROP = 0.05

_CURVE_NO_LEARN_BASELINE = 0.9


# 过拟合的第二条证据：训练损失已不再下降（记忆化 regime 下
# 往往在低位走平），但最终训练损失比验证损失低出该相对比例
# 以上（泛化 gap 大）且验证损失回升达到阈值。只看「训练损失
# 仍在降」会把这类典型过拟合误报成 healthy。
_CURVE_GENERALIZATION_GAP = 0.1


# 泛化 gap 的早期预警：从首个验证点到最后一个验证点，泛化
# gap（相对验证损失的比例）扩大达到该幅度、且最终 gap 已不低
# 时，即使验证损失尚未明显回升（未触发 overfitting），也提示
# 「训练损失继续降但验证收益停滞」的过拟合前兆，建议提前准备
# 早停/加正则化。等待验证损失明显回升后再报警会偏晚。
_CURVE_GAP_WIDENING = 0.05


# 学习率已退火到峰值的该比例以下时，训练/验证双走平更像
# 调度末期的正常收敛平台，而不是学习率不当/数据管道问题
# （默认调度为 cosine 退火到峰值的 10%，留少量余量）。
_CURVE_LR_ANNEALED_FRACTION = 0.15


# 最近一次验证间隔内训练与验证损失同时相对上升达到该比例
# 时，判定为有限发散/训练不稳定（不是过拟合：过拟合要求训练
# 损失仍在下降）。这类曲线继续训练通常只会更糟，应优先降低
# 学习率、收紧梯度裁剪并从稳定检查点恢复。
_CURVE_DIVERGENCE_RISE = 0.10


# 梯度范数尖峰预警：近期窗口最大值达到此前中位数的该倍数时，
# 即使损失尚未变成 NaN，也提示梯度爆炸/学习率过大的前兆。
_CURVE_GRAD_SPIKE_RATIO = 5.0

_CURVE_GRAD_SPIKE_MIN_HISTORY = 20

_CURVE_GRAD_SPIKE_RECENT_FRACTION = 0.2

_CURVE_GRAD_SPIKE_MIN_RECENT = 5


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"



def _sparkline(values, width=48):
    """
    把一列数值渲染成等宽 Unicode 走势条，便于在终端快速扫
    训练曲线走势；数值越小时块越低（损失越小越靠下）。

    width<=0 返回空串；width=1 时只保留最后一个点（最近一次
    观测），避免降采样的端点插值除以 width-1。
    """
    # 记录里的数值可能来自 JSON（含超出 float 范围的整数）或
    # 已污染的曲线；无法得到有限浮点时返回空图，而不是让
    # float()/int(nan) 在展示路径抛异常。
    numeric: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return ""
        if not math.isfinite(number):
            return ""
        numeric.append(number)
    values = numeric

    if not values or width <= 0:
        return ""

    if len(values) > width:
        if width == 1:
            values = [values[-1]]
        else:
            # 采样必须覆盖两个端点：等距取整会让最后一个点落在
            # int((width-1)*stride) < len-1 上被丢掉，走势条看不到
            # 训练末段的收敛/发散，而末段恰恰是诊断最关心的部分。
            last = len(values) - 1
            values = [
                values[
                    min(
                        last,
                        int(round(index * last / (width - 1))),
                    )
                ]
                for index in range(width)
            ]

    # 合法但极大的有限值（如 -1e308 与 +1e308）会让 high-low
    # 溢出成 inf，进而 (value-low)/(high-low) 变成 inf/inf=NaN，
    # int(NaN) 在展示路径抛异常。归一化到 [-1,1] 后分块位置
    # 不变；正常量级不缩放，保持历史输出逐位一致。
    scale = max(abs(value) for value in values)
    if scale > 1.0e150:
        low = min(value / scale for value in values)
        high = max(value / scale for value in values)
        normalized = [value / scale for value in values]
    else:
        low = min(values)
        high = max(values)
        normalized = values

    if high - low <= 0.0:
        return _SPARK_BLOCKS[0] * len(values)

    return "".join(
        _SPARK_BLOCKS[
            min(
                len(_SPARK_BLOCKS) - 1,
                int(
                    (value - low)
                    / (high - low)
                    * (len(_SPARK_BLOCKS) - 1)
                    + 0.5
                ),
            )
        ]
        for value in normalized
    )



def _finite_gap(
    train_loss: Optional[float],
    val_loss: Optional[float],
) -> Optional[float]:
    """训练/验证损失之差；任一端缺失或差值溢出时返回 None。"""
    train_value = _finite_float(train_loss)
    val_value = _finite_float(val_loss)
    if train_value is None or val_value is None:
        return None
    difference = train_value - val_value
    if not math.isfinite(difference):
        return None
    return round(difference, 6)
