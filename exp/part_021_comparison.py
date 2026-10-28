from __future__ import annotations



def bad_case_comparison_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    跨运行汇总坏案例的错误模式。

    单个运行的 top-N 案例可能只是偶然困难样本；真正值得据此
    改模型的证据是「多个种子/模式重复出现」的错误类别与混淆
    对。这里按错误 token 数加权汇总全量归因，并统计每个
    混淆对出现在多少个运行里。

    注意各运行只落盘 top-10 混淆对，合计计数是这些 top 对的
    求和，不是全量混淆矩阵；报告必须保留该覆盖上限，避免把
    「top 对里常见」误读成「全量里常见」。
    """
    available = [
        row for row in rows if row.get("bad_case_available")
    ]
    if not available:
        return {
            "n_runs_with_bad_cases": 0,
            "runs": [],
            "note": "没有运行带坏案例记录。",
        }

    category_names = list(_BAD_CASE_CATEGORIES)
    category_counts = {name: 0 for name in category_names}
    category_share_by_run: Dict[str, List[float]] = {
        name: [] for name in category_names
    }
    total_tokens = 0
    total_wrong = 0
    pair_totals: Dict[Tuple[int, int], int] = {}
    pair_run_counts: Dict[Tuple[int, int], int] = {}
    position_totals: Dict[str, List[int]] = {}
    frequency_totals: Dict[str, List[int]] = {}
    frequency_bases = set()
    runs = []

    def _safe_count(value: object) -> int:
        """把来自旧/损坏记录的计数转成非负整数，坏值按 0。

        bad_cases 记录是 metrics.jsonl 中的普通 JSON；手工编辑、
        旧版本或崩溃留下的半条记录可能把计数字段写成 NaN/
        Infinity/字符串/列表。对比报告必须跳过这些值而不是让
        整份 `--compare` 抛 ValueError/TypeError。
        """
        number = _finite_float(value)
        if number is None or number < 0.0:
            return 0
        return int(number)

    for row in available:
        tokens = _safe_count(
            row.get("bad_case_tokens_evaluated")
        )
        wrong = _safe_count(
            row.get("bad_case_wrong_tokens")
        )
        total_tokens += tokens
        total_wrong += wrong
        counts = _as_dict(
            row.get("bad_case_error_category_counts")
        )
        shares = _as_dict(
            row.get("bad_case_error_category_shares")
        )
        for name in category_names:
            count = _safe_count(counts.get(name))
            category_counts[name] += count
            share = _finite_float(shares.get(name))
            if share is not None:
                category_share_by_run[name].append(share)

        raw_pairs = row.get("bad_case_top_confusions")
        pairs = (
            raw_pairs
            if isinstance(raw_pairs, (list, tuple))
            else []
        )
        seen_pairs = set()
        valid_pairs = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 3:
                continue
            target, predicted, count = pair[:3]
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                for value in (target, predicted, count)
            ):
                continue
            if target < 0 or predicted < 0 or count <= 0:
                continue
            key = (int(target), int(predicted))
            pair_totals[key] = pair_totals.get(key, 0) + int(count)
            seen_pairs.add(key)
            valid_pairs.append([target, predicted, count])
        for key in seen_pairs:
            pair_run_counts[key] = pair_run_counts.get(key, 0) + 1

        for field, sink in (
            ("bad_case_error_rate_by_position", position_totals),
            (
                "bad_case_error_rate_by_train_frequency",
                frequency_totals,
            ),
        ):
            raw_entries = row.get(field)
            entries = (
                raw_entries
                if isinstance(raw_entries, (list, tuple))
                else []
            )
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if name is None:
                    continue
                bucket = sink.setdefault(str(name), [0, 0])
                bucket[0] += _safe_count(entry.get("tokens"))
                bucket[1] += _safe_count(entry.get("wrong"))
        basis = row.get("bad_case_error_frequency_basis")
        if basis is not None:
            frequency_bases.add(str(basis))

        runs.append({
            "run": str(row.get("run")),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "tokens_evaluated": tokens,
            "wrong_tokens": wrong,
            "token_accuracy": _finite_float(
                row.get("bad_case_token_accuracy")
            ),
            "error_category_counts": {
                name: _safe_count(counts.get(name))
                for name in category_names
            },
            # 旧/损坏记录的占比可能是 NaN/Inf/超范围整数；
            # 不能把这些值原样写进报告。
            "error_category_shares": {
                name: _finite_float(shares.get(name))
                for name in category_names
            },
            "top_confusion": (
                valid_pairs[0] if valid_pairs else None
            ),
        })

    category_shares: Dict[str, Optional[float]] = {}
    for name, count in category_counts.items():
        share = None
        if total_wrong > 0:
            try:
                ratio = count / total_wrong
            except OverflowError:
                # 损坏记录可能让单类计数远大于错误总数；真除法
                # 越过 float 上限时按缺失处理，而不是中断报告。
                ratio = None
            if ratio is not None and math.isfinite(ratio):
                share = round(ratio, 6)
        category_shares[name] = share
    n_runs = len(available)
    min_pair_runs = (
        1 if n_runs == 1 else max(2, (n_runs + 1) // 2)
    )
    persistent_pairs = [
        {
            "target": target,
            "predicted": predicted,
            # 同一运行只计一次，避免单个运行的重复计数
            # 把它伪装成跨运行稳定模式。
            "runs_present": pair_run_counts.get(
                (target, predicted), 0
            ),
            "top_count_sum": count,
        }
        for (target, predicted), count in sorted(
            pair_totals.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if pair_run_counts.get((target, predicted), 0)
        >= min_pair_runs
    ][:10]

    def _stratified_summary(
        buckets: Dict[str, List[int]],
    ) -> List[Dict[str, object]]:
        rows_out = []
        for name, (tokens, wrong) in buckets.items():
            error_rate = None
            if tokens > 0:
                try:
                    ratio = wrong / tokens
                except OverflowError:
                    # 损坏记录可能让 wrong 远大于 tokens；无法
                    # 表示的错误率按缺失处理。
                    ratio = None
                if ratio is not None and math.isfinite(ratio):
                    error_rate = round(ratio, 6)
            rows_out.append({
                "name": name,
                "tokens": tokens,
                "wrong": wrong,
                "error_rate": error_rate,
            })
        return rows_out

    category_share_ranges = {}
    unstable_categories = []
    if n_runs >= 2:
        for name, values in category_share_by_run.items():
            if len(values) < 2:
                continue
            spread = max(values) - min(values)
            if math.isfinite(spread):
                category_share_ranges[name] = round(spread, 6)
                if spread >= 0.20:
                    unstable_categories.append(name)
            else:
                # 正负极大占比相减会溢出；没有可表示的极差。
                category_share_ranges[name] = None

    return {
        "n_runs_with_bad_cases": n_runs,
        "runs": runs,
        "tokens_evaluated": total_tokens,
        "wrong_tokens": total_wrong,
        "error_category_counts": category_counts,
        "error_category_shares": category_shares,
        "persistent_confusions": persistent_pairs,
        "persistent_confusion_min_runs": min_pair_runs,
        "error_rate_by_position": _stratified_summary(
            position_totals
        ),
        "error_rate_by_train_frequency": (
            _stratified_summary(frequency_totals)
            if len(frequency_bases) <= 1
            else []
        ),
        "category_share_ranges": category_share_ranges,
        "unstable_categories": unstable_categories,
        "coverage_note": (
            "混淆对只来自各运行落盘的 top-10 对，"
            "合计计数不是全量混淆矩阵；全量错误类别按错误"
            " token 加权汇总。"
        ),
        "note": (
            "跨运行重复出现的错误类别/混淆对才是稳定的改进"
            "线索；类别占比跨运行极差 ≥20pp 时说明失败模式"
            "主要由种子/运行差异驱动，不能当作单一方法缺陷。"
        ),
    }



def compare_runs(paths: List[str]) -> List[Dict[str, object]]:
    """
    汇总多个运行目录的 config 与最终指标。

    读取每个目录的 config.json（结构、参数量、efficiency）与
    metrics.jsonl（final/curve_analysis），输出统一口径的行。

    只汇总 val 口径指标：test 只在单个运行内解封评估一次，
    不进入跨运行对比（否则会诱发「挑 test 最好」的隐式
    测试集调参）。
    """
    rows: List[Dict[str, object]] = []

    for path in paths:
        run_dir = Path(path)
        row: Dict[str, object] = {
            "run": run_dir.name,
            "path": str(run_dir),
        }

        config_payload: Dict[str, object] = {}
        config_path = run_dir / "config.json"
        try:
            config_exists = config_path.exists()
        except OSError:
            config_exists = False
        if config_exists:
            try:
                with open(
                    config_path,
                    encoding="utf-8",
                ) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    config_payload = loaded
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                config_payload = {}

        # 逐窗口评估记录（eval_windows.json）：固定评估窗口在所有
        # 运行之间逐位相同，跨运行比较可以按窗口配对，给出
        # 「在这批固定评估窗口上」的配对证据。缺失（旧运行）时
        # 保持 None，窗口级分析整体跳过而不是静默当成没有差异。
        windows_path = run_dir / "eval_windows.json"
        try:
            windows_exists = windows_path.exists()
        except OSError:
            windows_exists = False
        if windows_exists:
            try:
                with open(
                    windows_path,
                    encoding="utf-8",
                ) as f:
                    loaded_windows = json.load(f)
                if isinstance(loaded_windows, dict):
                    row["eval_windows"] = loaded_windows
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                row["eval_windows"] = None

        # 确定性标记：ngram 统计基线换 seed 不会产生新的独立
        # 重复；聚合与检验必须据此折叠，而不是把复制运行当
        # 独立种子。
        row["deterministic"] = bool(
            config_payload.get("deterministic")
        )

        run_config = config_payload.get("config") or {}
        if isinstance(run_config, dict):
            row["mode"] = run_config.get("mode")
            row["ngram_tuning_metric"] = run_config.get(
                "ngram_tuning_metric"
            )
            row["ngram_tuning"] = run_config.get(
                "ngram_tuning"
            )
            row["top_k"] = run_config.get("top_k")
            row["k1"] = run_config.get("k1")
            row["k2"] = run_config.get("k2")
            row["balance_coef"] = run_config.get("balance_coef")
            row["dense_fnn_hidden"] = run_config.get(
                "dense_fnn_hidden"
            )
            row["n_experts"] = run_config.get("n_experts")
            row["expert_hidden"] = run_config.get(
                "expert_hidden"
            )

        run_args = config_payload.get("args") or {}
        if isinstance(run_args, dict):
            row["seed"] = run_args.get("seed")
            # 选型指标：旧运行没有该字段，历史行为是在
            # val_loss 上选 checkpoint；ngram 不训练，保持
            # None，不参与选型一致性检查。
            if str(row.get("mode")) == "ngram":
                row["selection_metric"] = None
            else:
                row["selection_metric"] = (
                    run_args.get("selection_metric") or "val_loss"
                )

        # 假设纪律：对比分析要能展示每个运行登记的科学假设，
        # 并把缺失假设（旧运行或显式跳过）单独标记出来，避免
        # 把「排列组合超参数」当成假设驱动的实验。
        row["hypothesis"] = config_payload.get("hypothesis")
        row["research_question"] = config_payload.get(
            "research_question"
        )
        row["hypothesis_check_skipped"] = bool(
            config_payload.get("hypothesis_check_skipped")
        )

        # 预注册的假设参数（--primary-metric / 方向 / 最小效应）：
        # 对比阶段据此核对「观测结果是否与训练前的预期一致」
        # 与「分析指标是不是预注册的主指标」，避免事后挑指标、
        # 事后解释方向。旧运行没有该记录时保持 None。
        preregistration = config_payload.get(
            "hypothesis_preregistration"
        )
        if not isinstance(preregistration, dict):
            preregistration = {}
        row["primary_metric"] = preregistration.get(
            "primary_metric"
        )
        row["hypothesis_direction"] = preregistration.get(
            "direction"
        )
        row["hypothesis_min_effect"] = preregistration.get(
            "min_effect"
        )
        # 预注册的等价界（--equivalence-margin 在训练前写入）：
        # TOST 的等价界越宽越容易判等价，看到数据后再放宽就能
        # 「制造」等价结论，因此分析期必须能核对它没有被改动。
        row["hypothesis_equivalence_margin"] = (
            preregistration.get("equivalence_margin")
        )
        # 预注册的对照基线（--hypothesis-baseline）：主指标/
        # 方向/最小效应都声明了，若对照臂可以事后挑，整个
        # 检验仍然可以被 baseline shopping 操纵。旧运行没有
        # 该字段时保持 None。
        row["hypothesis_baseline"] = preregistration.get(
            "baseline"
        )

        # 实验规模守卫（tokens_seen / tokens per param / epochs）：
        # 原型级结果不能外推为规模结论。
        row["scale_guard"] = config_payload.get("scale_guard")

        # 参考基线（uniform / unigram 在 val 上的同口径指标）：
        # 对比分析用它核对「所声明的基线是否真的强于无上下文
        # 预测器」，否则复杂模型战胜的只是稻草人。
        reference = config_payload.get("reference_baselines")
        row["reference_baselines"] = (
            reference if isinstance(reference, dict) else None
        )

        # 基线调参来源（--baseline-tuning-from 写入）：对比分析
        # 用它核对「基线是否真的被充分调过参」，没有该记录且
        # 又没有强于调优好的 n-gram 时，战胜基线说明不了问题。
        tuning = config_payload.get("baseline_tuning")
        row["baseline_tuning"] = (
            tuning if isinstance(tuning, dict) else None
        )

        # RQ3/RQ4 路由诊断（config.json 的 routing_analysis）：
        # 汇总两轮专家重合度与专家负载/路由熵，供跨运行对比。
        # 诊断只基于验证集固定窗口，测试集不参与；缺失时保持
        # None，不虚报指标。
        routing = config_payload.get("routing_analysis")
        row["routing_available"] = isinstance(routing, dict)
        for key in _ROUTING_ROW_FIELDS:
            row[key] = None
        if isinstance(routing, dict):
            rq3 = _as_dict(
                _as_dict(
                    routing.get("rq3_expert_overlap")
                ).get("summary")
            )
            rq4 = _as_dict(
                _as_dict(
                    routing.get("rq4_load_and_entropy")
                ).get("summary")
            )
            divergence = _as_dict(
                _as_dict(
                    routing.get("branch_divergence")
                ).get("summary")
            )
            row["routing_jaccard_branch"] = rq3.get(
                "second_round_jaccard_mean"
            )
            row["routing_slot_reuse_branch"] = rq3.get(
                "second_round_slot_reuse_mean"
            )
            row["routing_jaccard_token"] = rq3.get(
                "token_second_round_jaccard_mean"
            )
            row["routing_branch_self_reuse"] = rq3.get(
                "branch_self_reuse_share"
            )
            round1 = _as_dict(rq4.get("round1"))
            round2 = _as_dict(rq4.get("round2"))
            row["routing_r1_load_entropy"] = round1.get(
                "load_entropy"
            )
            row["routing_r2_load_entropy"] = round2.get(
                "load_entropy"
            )
            row["routing_r2_dead_experts"] = round2.get(
                "dead_experts"
            )
            row["routing_r2_router_entropy"] = round2.get(
                "router_entropy_mean"
            )
            row["routing_branch_cosine"] = divergence.get(
                "branch_cosine_mean"
            )

        # 受控变量协议：对比分析用它检查「除声明因子外，
        # 优化器 / 学习率 / 训练轮数 / 数据增强等设置是否一致」。
        row["protocol"] = experiment_protocol_from_payload(
            config_payload
        )
        # 记录原 config.json 中 comparison_protocol 实际写出的键：
        # 显式 null（如 cosine 调度下的 wsd_stable_steps、未启用
        # 文档 mask 时的 document_separator）是已记录的可比值，
        # 不能和旧运行缺少该键混为一谈。只有回退补齐的键才在
        # 值仍为 None 时标记为「未核对」。
        recorded_protocol = config_payload.get(
            "comparison_protocol"
        )
        row["protocol_recorded_keys"] = (
            sorted(str(key) for key in recorded_protocol)
            if isinstance(recorded_protocol, dict)
            else None
        )

        # 训练前初始化指纹与首批训练数据指纹：同种子下
        # 若两者不一致，结构/指标差异就不能全部归因于
        # 声明因子。旧运行没有这些记录时保持 None。
        initialization = config_payload.get("initialization")
        row["init_fingerprints"] = (
            initialization
            if isinstance(initialization, dict)
            else None
        )
        training_trace = config_payload.get("training_trace")
        row["training_trace"] = (
            training_trace
            if isinstance(training_trace, dict)
            else None
        )

        # 可复现性来源：代码指纹与数值环境（来自各运行
        # config.json 的 run_metadata）。跨代码版本或跨
        # Python/torch/CUDA/设备比较时，方法差异无法与实现/
        # 环境差异分离，对比分析必须能发现。
        metadata = config_payload.get("run_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        provenance_keys = (
            *_REPRODUCIBILITY_HARD_KEYS,
            *_REPRODUCIBILITY_INFO_KEYS,
            "pythonhashseed_at_start",
        )
        row["provenance"] = {
            key: metadata.get(key)
            for key in provenance_keys
        }
        row["provenance_recorded_keys"] = sorted(
            key for key in provenance_keys if key in metadata
        )

        # 执行顺序设计：套件父进程在训练前写入 design_plan.json，
        # 子进程把自己的分配（计划指纹/序号/区组位置）写入
        # config.json。分析期用它诊断时间漂移与「计划被替换」。
        assignment = config_payload.get("design_assignment")
        if not isinstance(assignment, dict):
            assignment = {}
        row["design_assignment"] = assignment or None
        row["design_plan_id"] = assignment.get("plan_id")
        row["design_plan_file_sha256"] = assignment.get(
            "plan_file_sha256"
        )
        row["design_order_scheme"] = assignment.get(
            "order_scheme"
        )
        row["design_order_seed"] = assignment.get("order_seed")
        row["design_order_index"] = assignment.get(
            "order_index"
        )
        row["design_block"] = assignment.get("block")
        row["design_block_position"] = assignment.get(
            "position_in_block"
        )
        original_metadata = config_payload.get(
            "original_run_metadata"
        )
        if not isinstance(original_metadata, dict):
            original_metadata = metadata
        row["run_started_utc"] = original_metadata.get(
            "timestamp_utc"
        )

        row["n_params"] = config_payload.get("n_params")

        efficiency = config_payload.get("efficiency") or {}
        if not isinstance(efficiency, dict):
            efficiency = {}
        row["latency_ms_per_forward"] = efficiency.get(
            "latency_ms_per_forward"
        )
        row["training_ms_per_step"] = efficiency.get(
            "training_ms_per_step"
        )
        row["flops_per_forward"] = efficiency.get(
            "flops_per_forward"
        )
        row["training_peak_memory_mb"] = efficiency.get(
            "training_peak_memory_mb"
        )
        row["actual_training_peak_memory_mb"] = (
            efficiency.get("actual_training_peak_memory_mb")
        )
        # 优先用真实训练循环峰值（含优化器状态）；旧运行没有
        # 该字段时才退回合成前向+反向峰值，最后才用进程 RSS。
        row["peak_memory_mb"] = (
            row["actual_training_peak_memory_mb"]
            if row["actual_training_peak_memory_mb"]
            is not None
            else (
                efficiency.get("training_peak_memory_mb")
                if efficiency.get("training_peak_memory_mb")
                is not None
                else efficiency.get("process_peak_rss_mb")
            )
        )

        records = read_metrics_records(
            run_dir / "metrics.jsonl"
        )
        final_record = None
        curve_record = None
        bad_case_record = None

        for record in records:
            step = record.get("step")
            if step == "final":
                final_record = record
            elif step == "curve_analysis":
                curve_record = record
            elif (
                step == "bad_cases"
                and record.get("split") == "val"
            ):
                # 只取 val：test 坏案例同样受尘封协议保护，
                # 不进入跨运行比较。
                bad_case_record = record

        row["val_loss"] = (
            final_record.get("val_loss")
            if final_record is not None
            else None
        )
        row["val_ppl"] = (
            final_record.get("val_ppl")
            if final_record is not None
            else None
        )
        row["val_bits_per_byte"] = (
            final_record.get("val_bits_per_byte")
            if final_record is not None
            else None
        )
        # 类平衡似然（macro NLL）：频率加权的 val_loss 之外，
        # 不平衡场景下必须有一份按类等权的似然口径。
        row["val_macro_nll"] = (
            final_record.get("val_macro_nll")
            if final_record is not None
            else None
        )
        row["val_macro_nll_chance"] = (
            final_record.get("val_macro_nll_chance")
            if final_record is not None
            else None
        )
        row["val_macro_nll_classes"] = (
            final_record.get("val_macro_nll_classes")
            if final_record is not None
            else None
        )
        row["val_token_accuracy"] = (
            final_record.get("val_token_accuracy")
            if final_record is not None
            else None
        )
        # micro F1 与 token accuracy 口径一致，但在 COMPARISON_
        # METRIC_KEYS 中被 --compare-metric 接受；缺这一行会让
        # val_micro_f1 的跨运行对比永远报「缺指标」。
        row["val_micro_f1"] = (
            final_record.get("val_micro_f1")
            if final_record is not None
            else None
        )
        row["val_macro_f1"] = (
            final_record.get("val_macro_f1")
            if final_record is not None
            else None
        )
        row["val_weighted_f1"] = (
            final_record.get("val_weighted_f1")
            if final_record is not None
            else None
        )
        # precision / recall（macro 与 weighted 口径）：假设以
        # 「召回率/精确率」判定时必须有对应指标可供对比。
        row["val_precision_macro"] = (
            final_record.get("val_precision_macro")
            if final_record is not None
            else None
        )
        row["val_precision_weighted"] = (
            final_record.get("val_precision_weighted")
            if final_record is not None
            else None
        )
        row["val_recall_macro"] = (
            final_record.get("val_recall_macro")
            if final_record is not None
            else None
        )
        row["val_recall_weighted"] = (
            final_record.get("val_recall_weighted")
            if final_record is not None
            else None
        )
        row["val_pr_auc_macro"] = (
            final_record.get("val_pr_auc_macro")
            if final_record is not None
            else None
        )
        # PR-AUC 随机水平（纳入类平均 prevalence）：不平衡下
        # PR-AUC 必须相对它解读，而不是当准确率看。
        row["val_pr_auc_macro_chance"] = (
            final_record.get("val_pr_auc_macro_chance")
            if final_record is not None
            else None
        )
        # PR-AUC lift = 绝对值 - 随机水平，是跨 prevalence
        # 可直接比较的一等口径；旧运行没有该字段时保持 None，
        # 不根据 chance 静默重算（最终指标可能来自旧评估）。
        row["val_pr_auc_macro_lift"] = (
            final_record.get("val_pr_auc_macro_lift")
            if final_record is not None
            else None
        )
        row["val_roc_auc_macro"] = (
            final_record.get("val_roc_auc_macro")
            if final_record is not None
            else None
        )
        # 多数类参考 accuracy（= 评估集中最高频类的 support 占比）：
        # 类别不平衡提示用，判断 accuracy/micro 口径是否被常数
        # 预测主导。
        row["val_majority_accuracy"] = (
            final_record.get("val_majority_accuracy")
            if final_record is not None
            else None
        )
        row["val_classes_present"] = (
            final_record.get("val_classes_present")
            if final_record is not None
            else None
        )
        row["curve_status"] = (
            curve_record.get("status")
            if curve_record is not None
            else None
        )
        # 曲线诊断数值：只看 status 只能知道「是否过拟合」，
        # 看不到过拟合多严重、最优点多早、泛化 gap 多大。
        # 这些字段直接来自 analyze_training_curve 的落盘结果，
        # 让跨运行比较能对齐「同一训练曲线口径」而不只是最终
        # 单点指标；旧记录缺字段时保持 None。
        row["curve_available"] = curve_record is not None
        for curve_key in _CURVE_ROW_FIELDS:
            row[f"curve_{curve_key}"] = (
                curve_record.get(curve_key)
                if curve_record is not None
                else None
            )

        # 坏案例记录（val）：供跨运行汇总稳定错误模式。旧运行
        # 只有部分字段时保持 None，不伪造缺失的归因。
        row["bad_case_available"] = bad_case_record is not None
        for bad_key in _BAD_CASE_ROW_FIELDS:
            row[f"bad_case_{bad_key}"] = (
                bad_case_record.get(bad_key)
                if bad_case_record is not None
                else None
            )

        # 数值完整性：失败的运行不能进入均值与显著性检验。
        # 旧记录没有新字段时用曲线里的 nonfinite 事件与
        # diverged 状态兜底；完全无记录则标 unknown，不臆测。
        def _record_value(name: str):
            if final_record is not None and name in final_record:
                return final_record.get(name)
            return None

        row["numerical_stop"] = _record_value(
            "numerical_stop"
        )
        row["numerical_polluted"] = _record_value(
            "numerical_polluted"
        )
        row["prior_numerical_stop"] = _record_value(
            "prior_numerical_stop"
        )
        final_nonfinite = _record_value("nonfinite_events")
        curve_nonfinite = (
            curve_record.get("n_nonfinite_events")
            if curve_record is not None
            else None
        )
        nonfinite_events = (
            final_nonfinite
            if final_nonfinite is not None
            else curve_nonfinite
        )
        row["nonfinite_events"] = nonfinite_events
        integrity = _record_value("numerical_integrity_ok")
        if integrity is not None:
            integrity = bool(integrity)
        failed = bool(
            row["numerical_stop"]
            or row["numerical_polluted"]
            or row["prior_numerical_stop"]
            or row["curve_status"] in ("diverged", "diverging")
            or (
                isinstance(nonfinite_events, int)
                and not isinstance(nonfinite_events, bool)
                and nonfinite_events > 0
            )
        )
        if integrity is False or failed:
            row["numerical_status"] = "failed"
        elif integrity is True:
            row["numerical_status"] = "ok"
        else:
            row["numerical_status"] = "unknown"
        row["numerical_integrity_ok"] = (
            False
            if row["numerical_status"] == "failed"
            else (
                True
                if row["numerical_status"] == "ok"
                else None
            )
        )
        row["early_stopped"] = (
            final_record.get("early_stopped")
            if final_record is not None
            else None
        )
        # 实际有效训练长度与选型结果：受控变量检查必须核对
        # 「实际训练了多久」而不只是计划步数，且 final/test
        # 指标是否来自验证集选型的模型要能从 final 记录恢复
        # （config 顶层记录可能因崩溃缺失）。
        row["trained_steps"] = (
            final_record.get("trained_steps")
            if final_record is not None
            else None
        )
        row["optimizer_steps"] = (
            final_record.get("optimizer_steps")
            if final_record is not None
            else None
        )
        row["end_step"] = (
            final_record.get("selected_step")
            if final_record is not None
            else None
        )
        row["selected_model"] = (
            final_record.get("selected_model")
            if final_record is not None
            else None
        )

        protocol = row.get("protocol")
        if isinstance(protocol, dict):
            if row["trained_steps"] is not None:
                protocol["trained_steps"] = row["trained_steps"]
            if row["optimizer_steps"] is not None:
                protocol["optimizer_steps"] = row["optimizer_steps"]
            if row["selected_model"] == "best_val":
                protocol["model_selection"] = (
                    "best_val_with_last_fallback"
                )
            elif row["selected_model"] == "last":
                protocol["model_selection"] = "last_step"

        # config/metrics 来自旧版本或手工编辑的文件，可能含
        # NaN/Inf（Python json 默认接受这些字面量）。它们进入
        # 对比报告后会让严格 JSON 落盘失败，也会污染下游聚合；
        # 统一把非有限浮点转成 None，保留字段与有限大整数精度。
        rows.append(_json_safe(row))

    return rows



def print_comparison(rows: List[Dict[str, object]]) -> None:
    """
    以对齐表格打印多个运行的汇总指标。

    只展示 val 口径指标（尘封协议：test 结果只随单个运行的
    metrics.jsonl 报告，绝不并排对比）；除 val loss 外同时
    给出类平衡似然（macroNLL，按出现类等权）、token accuracy
    （micro 口径）与 macro F1：字节分布高度不平衡时 accuracy
    会被高频字节抬高、频率加权的 loss 也几乎只反映高频字节，
    必须类平衡口径与分类指标一起看；完整口径（micro / macro /
    weighted、PR-AUC 等）见各运行 metrics.jsonl。

    """
    # 运行名与模式名不能截断：local_base 与 local_base_k2 在
    # 20/9 字符的固定列宽下会显示成同一前缀，读者无法区分
    # 不同实验臂。列宽按实际内容扩展。
    run_width = max(
        20,
        max(
            (len(str(row.get("run"))) for row in rows),
            default=3,
        ),
    )
    mode_width = max(
        9,
        max(
            (len(str(row.get("mode"))) for row in rows),
            default=4,
        ),
    )
    header = (
        f"{'run':<{run_width}} {'mode':<{mode_width}} "
        f"{'params':>9} "
        f"{'val_loss':>9} {'macroNLL':>9} {'val_bpb':>8} "
        f"{'acc':>6} "
        f"{'macroF1':>8} "
        f"{'lat(ms)':>9} {'train(ms)':>10} {'GFLOPs':>8} "
        f"{'mem(MB)':>9} {'numerical':<10} {'curve':<18}"
    )
    print(header)
    print("-" * len(header))

    def _value(value, spec=".4f", scale=1.0):
        # config/metrics 里的数值可能损坏（含超出 float 范围的
        # 整数或 NaN/Inf）；展示层按缺失值处理，不能让一行坏
        # 记录中断整张对比表。
        return _format_finite(value, spec, scale)

    for row in rows:
        flops = _finite_float(row.get("flops_per_forward"))
        gflops = (
            f"{flops / 1e9:.3f}"
            if flops is not None
            else "-"
        )
        print(
            f"{str(row.get('run')):<{run_width}} "
            f"{str(row.get('mode')):<{mode_width}} "
            f"{str(row.get('n_params')):>9} "
            f"{_value(row.get('val_loss')):>9} "
            f"{_value(row.get('val_macro_nll')):>9} "
            f"{_value(row.get('val_bits_per_byte'), '.4f'):>8} "
            f"{_value(row.get('val_token_accuracy'), '.4f'):>6} "
            f"{_value(row.get('val_macro_f1'), '.4f'):>8} "
            f"{_value(row.get('latency_ms_per_forward'), '.4f'):>9} "
            f"{_value(row.get('training_ms_per_step'), '.1f'):>10} "
            f"{gflops:>8} "
            f"{_value(row.get('peak_memory_mb'), '.1f'):>9} "
            f"{str(row.get('numerical_status')):<10} "
            f"{str(row.get('curve_status')):<18}"
        )

    # RQ3/RQ4 路由诊断：跨越单次运行的结论必须能看到每个运行的
    # 路由指标，而不是只留在各自 config.json 的定性诊断里。
    # 旧运行或 dense 基线没有路由记录时明确注明，不虚报。
    routing_rows = [
        row for row in rows if row.get("routing_available")
    ]
    if routing_rows:
        print()
        print(
            "[routing] RQ3/RQ4 路由诊断（val 固定窗口，"
            "测试集不参与；聚合与显著性检验见 --compare "
            "的对比分析）"
        )
        routing_run_width = max(
            20,
            max(
                (
                    len(str(row.get("run")))
                    for row in routing_rows
                ),
                default=3,
            ),
        )
        routing_header = (
            f"{'run':<{routing_run_width}} {'jac(branch)':>11} "
            f"{'slotReuse':>9} {'selfReuse':>9} "
            f"{'r1LoadH':>8} {'r2LoadH':>8} {'dead':>5} "
            f"{'routerH':>8} {'cosine':>8}"
        )
        print(routing_header)
        print("-" * len(routing_header))

        def _rv(value, spec=".4f") -> str:
            return _format_finite(value, spec)

        for row in routing_rows:
            print(
                f"{str(row.get('run')):<{routing_run_width}} "
                f"{_rv(row.get('routing_jaccard_branch')):>11} "
                f"{_rv(row.get('routing_slot_reuse_branch')):>9} "
                f"{_rv(row.get('routing_branch_self_reuse')):>9} "
                f"{_rv(row.get('routing_r1_load_entropy')):>8} "
                f"{_rv(row.get('routing_r2_load_entropy')):>8} "
                f"{_rv(row.get('routing_r2_dead_experts'), '.0f'):>5} "
                f"{_rv(row.get('routing_r2_router_entropy')):>8} "
                f"{_rv(row.get('routing_branch_cosine')):>8}"
            )
        missing_routing = [
            str(row.get("run"))
            for row in rows
            if not row.get("routing_available")
        ]
        if missing_routing:
            print(
                "  [note] 以下运行没有路由诊断记录（dense 基线"
                "无 MoE 路由，或旧运行/--skip-routing-analysis）："
                + "; ".join(missing_routing)
            )



# ============================================================
# Comparison analysis: controlled variables / seeds / statistics
# ============================================================

# 受控变量协议：对比 A 方法与 B 方法时，除声明为实验因子的
# 模块外，这些设置必须完全一致。每轮训练把这些值写进
# config.json 的 comparison_protocol 字段；--compare 逐项核对。
#
# comparison_family 把「神经网络运行」与「统计基线运行」
# （ngram，无优化器、无训练轮数）分开：受控变量检查只在
# 同一 family 内部进行。
COMPARISON_PROTOCOL_KEYS: Tuple[str, ...] = (
    # mode 是最核心的结构差异：它默认是实验因子，但调用方一旦
    # 用 --compare-factors 收窄因子列表而漏掉 mode，跨模式运行
    # 就会绕过受控变量核对。把它写进协议后，只有显式声明为
    # 因子时才允许不同。
    "mode",
    # 初始化方案决定同种子下共享层是否真的从同一组权重出发；
    # 跨方案比较会把初始化差异混入方法效应，必须受控。
    "init_scheme",
    "optimizer",
    "optimizer_betas",
    "optimizer_eps",
    "lr",
    "lr_schedule",
    # WSD（论文 3.1 节）的稳定期终点与衰减终点决定整条学习率
    # 轨迹；cosine 运行时这两个值为 None。
    "wsd_stable_steps",
    "wsd_final_lr_ratio",
    "warmup_steps",
    "total_steps",
    # 实际有效训练长度：total_steps 只是计划值，早停或
    # non-finite 跳步会让各臂实际执行的步数/优化器更新次数
    # 不同；这些字段在训练结束时由 compare_runs 从 final
    # 记录注入协议并逐项核对（缺失时显式列为未核对）。
    "trained_steps",
    "optimizer_steps",
    "weight_decay",
    "grad_clip",
    "batch_size",
    # 文档级 segment mask 改变注意力可见性；分隔符字节决定
    # 文档边界位置，两者都是模型语义的一部分。
    "document_masks",
    "document_separator",
    "data_augmentation",
    # 早停设置决定「实际训练轮数」：total_steps 相同但早停
    # patience 不同时，某个运行可能提前结束，与「训练同样久」
    # 不是同一口径，必须作为受控变量核对。
    "early_stop_patience",
    "early_stop_min_delta",
    # 数值条件与实现开关：混合精度 dtype 与设备类型会改变
    # 浮点归约/精度（同一方法在 CPU 与 GPU 上不是同一数值
    # 环境），activation checkpointing 与 torch.compile 会
    # 改写反传/内核路径，跨运行对比时应完全一致。
    "amp_dtype",
    "device_type",
    "allow_tf32",
    "grad_checkpoint",
    "compile",
    "seq_len",
    "d_model",
    "n_heads",
    # GQA 的 KV 头数决定注意力容量与 KV cache；同一 mode
    # 下换 KV 头数就是换架构，不能只按 mode 归组后被当成
    # 同一实验单元聚合。
    "n_kv_heads",
    "n_layers",
    "n_experts",
    "expert_hidden",
    "top_k",
    # k1/k2/balance_coef/dense_fnn_hidden 是最容易悄悄改变结果的
    # 结构/损失项：它们默认被声明为实验因子，但只要调用方把
    # --compare-factors 改成更窄的列表，就必须退化为受控变量
    # 逐项核对，否则「延迟合并」的对比会混入分支几何（k1/k2）
    # 或均衡强度（balance_coef）的差异。
    "k1",
    "k2",
    "balance_coef",
    # 注意：balance_width_normalization 依 mode 而变（只有
    # local_base 有折算问题），不能作为跨模式受控变量；
    # 新旧 local_base 的均衡强度差异由 balance_strength_report
    # 在有效宽度/强度层面点名。
    "dense_fnn_hidden",
    "residual_scale",
    # 循环区域残差缩放口径（paper=1/r vs global）直接改变
    # 架构语义；它不随 mode 唯一确定，必须作为受控变量。
    # loop_residual_scale / loop_count 由该口径与 mode 派生，
    # 不重复列入，避免跨模式对比被派生值误报为违规。
    "residual_scaling",
    "val_fraction",
    "test_fraction",
    # 切分口径（contiguous / chronological）决定“前缀”能否
    # 被解释为时间切分；同样的 split_sha256 也可能来自不同
    # 声明，必须逐项核对。
    "split_policy",
    "group_separator",
    "deduplicate_train_groups",
    "drop_train_shared_chunks",
    "drop_train_near_duplicate_groups",
    "near_duplicate_threshold",
    "near_duplicate_shingle_bytes",
    # 训练集清洗是否读取 test 内容：默认 False（test 尘封）；
    # True 会让训练样本集合依赖 test，必须作为协议差异核对。
    "clean_train_against_test",
    "drop_train_corrupt_groups",
    "drop_train_corrupt_bytes",
    "drop_train_degenerate_groups",
    "drop_train_degenerate_runs",
    # 源文件指纹相同不等于实际进入各 split 的字节段相同：
    # 群组吸附、清洗实现或切分代码变化都会改变 train/val/test
    # 的内容。逐 split 指纹把「同一文件 + 同一比例」背后的
    # 实际划分也纳入受控变量，避免两次运行在不同验证/测试
    # 数据上比较却显示协议一致。
    "split_sha256",
    # 清洗后的有效训练集指纹：同一 split/开关下清洗实现变化
    # 会改变实际参与训练的字节，必须能被跨运行核对。
    "effective_train_sha256",
    # 时序隔离记录：绝对边界与顺序让「按时间先后切分、无重叠」
    # 可以从 config 独立复核，而不是只靠比例与内容指纹推断。
    "split_isolation",
    # 泄露豁免会改变「允许哪些数据进入训练」的资格判断；
    # 同一对比里有的运行显式豁免、有的没有，不能算同一协议。
    "allow_data_leakage",
    # 允许 val/test 内容参与训练集清洗、允许实体跨 split 重叠，
    # 都是会改变训练数据边界的显式豁免。两个运行一个允许一个
    # 不允许时，split 指纹即使碰巧相同也不是同一数据口径，
    # 不能把差异归因到方法本身。
    "allow_eval_content_for_cleaning",
    "allow_entity_overlap",
    # 标签噪声豁免会改变 test 是否被封条拒绝；两个运行最终
    # 指标的可信资格不同，同样不能算同一评估口径。
    "allow_corrupt_test_labels",
    "allow_eval_duplicates",
    "eval_interval",
    "eval_batches",
    # bootstrap 重采样次数决定验证/测试置信区间的蒙特卡洛
    # 精度：不同运行用不同的 --eval-bootstrap 会得到不同宽度
    # 的误差线，跨运行比较误差线时口径不再一致。
    "eval_bootstrap",
    # 重采样单位（i.i.d. 窗口 vs 移动块）会改变区间宽度与
    # 自相关处理；同一 bootstrap 次数但 block 不同不是同一
    # 评估口径。
    "eval_bootstrap_block",
    # 周期验证是否计算 PR-AUC / lift / ROC-AUC：显式
    # --periodic-auc 与 AUC 选型都会生成共同步数选型无关 AUC 指标；
    # 一个跑有一个没跑时，敏感性分析的可比口径不同。
    "periodic_auc",
    "data_sha256",
    # final/test 评估所用模型的选型口径：旧运行（无记录）是
    # 最后一步参数，新运行是验证集选中的最优验证点；两者混在
    # 一起对比不是同一口径，必须被检查发现。
    "model_selection",
    # 选型/早停依据的指标：默认 val_loss，但预注册主指标是
    # macro F1 等分类指标时必须能按同一指标选 checkpoint，
    # 否则最终模型不是按假设的判定口径选出来的。
    "selection_metric",
)


# 跨 family 对比（神经模型 vs ngram 统计基线）共用的
# 数据与评估口径：优化器/训练轮数可以不同（ngram 没有这些
# 概念），但训练/验证/测试切分、评估窗口集合与数据增强必须
# 一致，否则「神经基线强于 ngram」的结论会把数据差异误当成
# 方法差异。
_CROSS_FAMILY_PROTOCOL_KEYS: Tuple[str, ...] = (
    "data_sha256",
    # 神经运行与 ngram 必须看到同一组 train/val/test 字节段；
    # 只比整体文件指纹无法发现群组吸附/清洗实现差异。
    "split_sha256",
    # 两者还必须看到同一套清洗后的有效训练字节；否则 ngram
    # 计数与神经窗口覆盖的训练分布不同，基线比较失效。
    "effective_train_sha256",
    "split_isolation",
    "seq_len",
    "val_fraction",
    "test_fraction",
    "split_policy",
    "group_separator",
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
    "allow_data_leakage",
    "allow_eval_content_for_cleaning",
    "allow_entity_overlap",
    "allow_corrupt_test_labels",
    "allow_eval_duplicates",
    "batch_size",
    "eval_batches",
    # 神经模型与 ngram 统计基线都用同一评估窗口 bootstrap
    # 口径报告区间；重采样次数不同会让跨 family 的误差线
    # 无法直接比较。
    "eval_bootstrap",
    "eval_bootstrap_block",
    "data_augmentation",
)


# 这些键不随 mode 唯一确定，却会改变模型/数据口径：同 mode
# 的两条运行只要有一项不同，就已经不是同一个实验单元。展示层
# 用它们把「看似单因子、实际多因子」的违规单独点名。
SINGLE_FACTOR_DESIGN_KEYS: Tuple[str, ...] = (
    "n_kv_heads",
    "residual_scaling",
    "split_policy",
    "allow_eval_content_for_cleaning",
    "allow_entity_overlap",
)


# 对比运行的「可复现性来源」核对项：代码指纹与数值环境。
# 与受控变量协议不同，这些项不是训练超参数，但同样决定结果
# 能否归因：同一份代码、同一个 Python/torch/CUDA/cuDNN 与
# 设备上的两次运行才谈得上「方法差异」。旧运行缺记录时保持
# 未核对，不虚报一致。
_REPRODUCIBILITY_HARD_KEYS: Tuple[str, ...] = (
    "script_sha256",
    # 完整环境身份（关键依赖 + 编译选项 + 设备 + 确定性开关）。
    # 旧运行没有该字段时保持「未核对」，不虚报一致。
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
)


# 信息性字段：允许不同，只提示（同一脚本指纹下的 git commit
# 差异通常来自文档/CI 等与本实验无关的改动）。
_REPRODUCIBILITY_INFO_KEYS: Tuple[str, ...] = (
    "git_commit",
    "git_dirty",
    "git_diff_sha256",
    "cwd",
    "python_executable",
    "cuda_visible_devices",
)


# 允许声明为实验因子（跨运行允许不同）的键：行级字段 +
# 全部协议键。默认因子覆盖本项目的结构差异（模式、k1/k2、
# 分支宽度、balance 系数、种子）；学习率、训练步数、数据增强、
# 数据指纹等不在其中，因此默认就会被当作受控变量检查。
DEFAULT_COMPARISON_FACTORS = (
    "mode",
    "k1",
    "k2",
    "balance_coef",
    "seed",
    "dense_fnn_hidden",
)


# 聚合与显著性检验使用的指标。
#
# 尘封协议：这里只允许 val 口径的指标。测试集在每个运行中
# 只解封评估一次，其结果只能随该运行自身的 metrics.jsonl
# 报告；如果把 test 指标纳入跨运行对比/排序/显著性检验，
# 「在多个候选里挑 test 最好」会让测试集退化成隐式验证集
# （多重比较 + 选择偏差）。因此 --compare 系列工具一律
# 不读取、不聚合、不展示 test 指标。
# F1 的三种平均口径分别列出：micro（单标签多分类下与
# accuracy 恒等）、macro（对出现过的类等权平均，不平衡时
# 最能暴露少数类退化）与 weighted（按 support 加权，介于
# 两者之间）。名字即口径，跨运行对比时必须显式选择。
# precision / recall 同样按 macro 与 weighted 两种口径列出：
# 假设声称「依据召回率判定」时，评估管线必须真的计算它，
# 否则会出现「假设的判定指标不可检验」的假性对齐。
#
# val_loss_at_matched_flops 不是训练时周期评估的指标，而是
# --compare 阶段从各运行 metrics.jsonl 的验证曲线插值出的
# 「等训练计算量」loss（见 annotate_compute_matched_metrics）：
# 延迟合并 B/A 每步 FLOPs 高于 smelt，等步数比较会把机制收益
# 与额外计算混在一起。``*_at_common_step`` 系列同样来自周期
# 验证曲线，但按共同训练步数对齐，用来避开 final 指标的
# checkpoint 选型偏差（见 annotate_common_step_metrics）。它们
# 与对应周期指标同单位、同方向，可直接作为 --compare-metric
# 或预注册主指标做显著性检验；不能只给 loss 提供选型无关口径，
# 否则以 macro F1 / accuracy 判定时仍会落入 winner's curse。
COMPARISON_METRIC_KEYS: Tuple[str, ...] = (
    "val_loss",
    "val_loss_at_matched_flops",
    *_COMMON_STEP_METRICS,
    _CURVE_MEAN_METRIC,
    "val_ppl",
    "val_bits_per_byte",
    # 类平衡似然：val_loss 按 token 频率加权，高频字节几乎
    # 决定全部 loss；不平衡场景下必须同时看按类等权的
    # macro NLL（随机水平 log 词表大小，见 val_macro_nll_chance）。
    "val_macro_nll",
    "val_token_accuracy",
    "val_micro_f1",
    "val_macro_f1",
    "val_weighted_f1",
    "val_precision_macro",
    "val_precision_weighted",
    "val_recall_macro",
    "val_recall_weighted",
    "val_pr_auc_macro",
    # PR-AUC 相对随机水平（平均 prevalence）的 lift。绝对值随
    # prevalence 缩放，跨数据集不可比；lift 随机水平恒为 0，
    # 可直接作为主指标/预注册口径/基线核对口径。
    "val_pr_auc_macro_lift",
    "val_roc_auc_macro",
)


# --compare 行级的 RQ3/RQ4 路由诊断字段（来自各运行 config.json
# 的 routing_analysis；旧运行缺失时保持 None，不虚报指标）。
# 这些字段同样是 val 口径（诊断只在验证集固定窗口上做一次），
# 并接入 COMPARISON_METRIC_KEYS：RQ3/RQ4 的结论必须能跨种子
# 聚合、参与显著性检验，而不是只能靠单次运行的定性诊断。
_ROUTING_ROW_FIELDS: Tuple[str, ...] = (
    "routing_jaccard_branch",
    "routing_slot_reuse_branch",
    "routing_jaccard_token",
    "routing_branch_self_reuse",
    "routing_r1_load_entropy",
    "routing_r2_load_entropy",
    "routing_r2_dead_experts",
    "routing_r2_router_entropy",
    "routing_branch_cosine",
)


# 路由诊断指标作为对比指标：它们描述两轮路由的内部行为，
# 「方向」按本项目 RQ3/RQ4 的目标口径约定——第二轮与第一轮
# 越独立（jaccard / slot_reuse / self_reuse 越低）、分支轨迹
# 越分化（cosine 越低）、死专家越少，越符合假设方向；负载
# 熵与路由熵越高越均匀。方向只影响差值符号与 improvement/
# regression 标签；显著性检验与聚合不受影响，且报告必须
# 同时原样给出差值，不能只引用标签做因果解读。
COMPARISON_METRIC_KEYS = COMPARISON_METRIC_KEYS + _ROUTING_ROW_FIELDS
