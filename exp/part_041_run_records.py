from __future__ import annotations



def write_run_config(
    out_dir: Path,
    cfg: Config,
    args: argparse.Namespace,
    n_params: int,
    run_metadata: Dict[str, object],
    sanity_result: Optional[Dict[str, object]],
    data_audit: Optional[Dict[str, object]] = None,
    reference_baselines: Optional[Dict[str, object]] = None,
    baseline_tuning: Optional[Dict[str, object]] = None,
    data_protocol_check: Optional[Dict[str, object]] = None,
    dataset: Optional[ByteDataset] = None,
    init_fingerprints: Optional[Dict[str, object]] = None,
) -> None:
    """
    写出 config.json：结构配置、完整超参数、参数量、数据指纹、
    数据划分与审计结果、代码版本/运行环境、sanity check 结果、
    数据管线完整性质检结果、本轮实验的科学假设（--hypothesis/--rq）、
    参考基线（uniform / unigram）与基线调参来源
    （--baseline-tuning-from）。
    """
    comparison_protocol = build_comparison_protocol(cfg, args)
    # 协议指纹：final/test 记录只带这一条短标识即可证明自己
    # 对应哪一份完整受控变量集合，避免 metrics.jsonl 单独归档
    # 时无法核对超参数是否在续跑/复制过程中被改动。
    comparison_protocol_sha256 = _sha256_text(
        json.dumps(
            _json_safe(comparison_protocol),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    payload = {
        "config_schema": _ENVIRONMENT_CONFIG_SCHEMA,
        "config": asdict(cfg),
        "args": vars(args),
        # 受控变量协议：--compare 用它核对「除声明因子外，
        # 优化器 / 学习率 / 训练轮数 / 数据增强等设置是否一致」。
        "comparison_protocol": comparison_protocol,
        "comparison_protocol_sha256": (
            comparison_protocol_sha256
        ),
        "n_params": n_params,
        "paper_fidelity": getattr(args, "paper_fidelity", None),
        "cost_profile": getattr(args, "cost_profile", None),
        "budget_match": getattr(
            args, "budget_match_report", None
        ),
        "scale_guard": build_scale_guard(
            cfg,
            args,
            n_params,
            dataset,
        ),
        "initialization": init_fingerprints,
        "training_trace": (
            training_batch_fingerprint(
                dataset,
                int(args.seed),
                int(args.batch_size),
            )
            if dataset is not None
            else None
        ),
        "hypothesis": args.hypothesis,
        "research_question": args.rq,
        # 假设里提到的判定指标（规范化名字）：单次运行的假设
        # 质量也要可追溯，不必等到 --compare 阶段才第一次
        # 发现「假设没写指标无法判定」。
        "hypothesis_metric_mentions": hypothesis_metric_mentions(
            args.hypothesis
        ),
        "hypothesis_check_skipped": bool(
            args.skip_hypothesis_check
        ),
        # 外部预注册文件的核对记录：文件 SHA-256、study_id、
        # 冻结提交、由文件补齐的字段、实际数据指纹与模式/种子。
        # 与 hypothesis_preregistration 分开保存：后者只是本次
        # 运行生效的判定口径，前者证明这些口径在训练前已被一份
        # 独立文件固定并可核对。
        "preregistration": getattr(
            args, "preregistration_record", None
        ),
        # 设计计划分配（套件子进程）：计划指纹、执行序号、区组与
        # 区组内位置。--compare 的顺序漂移诊断只使用与分析直接
        # 相关的最少字段，完整分配随 config.json 归档。
        "design_assignment": getattr(
            args, "design_assignment", None
        ),
        # 训练前预注册的判定口径：主指标、预期方向、最小有意义
        # 效应。对比阶段据此核对「观测方向/幅度是否与训练前的
        # 预期一致」以及「分析指标是否是预注册的主指标」，
        # 避免事后挑指标、事后解释方向。
        "hypothesis_preregistration": {
            "primary_metric": getattr(
                args, "primary_metric", None
            ),
            "direction": getattr(
                args, "hypothesis_direction", None
            ),
            "min_effect": getattr(
                args, "hypothesis_min_effect", None
            ),
            # 预注册对照基线：训练前声明本轮与哪个臂比较，
            # 避免分析期换一个更容易战胜的臂当基线。
            "baseline": getattr(
                args, "hypothesis_baseline", None
            ),
            # 预注册等价界：TOST 的界越宽越容易判等价；训练
            # 前固化取值，分析期才可核对它没有被事后放宽。
            "equivalence_margin": getattr(
                args, "equivalence_margin", None
            ),
        },
        "reference_baselines": reference_baselines,
        "baseline_tuning": baseline_tuning,
        "data": {
            "path": args.data,
            "sha256": getattr(args, "data_sha256", None),
            "bytes": getattr(args, "data_bytes", None),
            "val_fraction": getattr(
                args,
                "data_val_fraction",
                None,
            ),
            "test_fraction": getattr(
                args,
                "data_test_fraction",
                None,
            ),
            "group_separator": getattr(
                args,
                "group_separator",
                None,
            ),
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
            "near_duplicate_audit": bool(
                getattr(
                    args,
                    "near_duplicate_audit",
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
        },
        "data_audit": data_audit,
        "data_protocol_check": data_protocol_check,
        "run_metadata": run_metadata,
        "sanity_check": sanity_result,
    }

    # 数据切分的精确边界与组切分：仅凭 sha256 + 比例无法在
    # 数据文件以外完整还原「哪些字节属于训练/验证/测试」，
    # 边界与去重/组切分记录是数据版本的一部分。
    if dataset is not None:
        payload["data"]["split_bounds"] = {
            split: list(bounds)
            for split, bounds in dataset.split_bounds.items()
        }
        payload["data"]["split_isolation"] = dict(
            dataset.split_isolation
        )
        payload["data"]["split_fractions"] = (
            dataset.split_fractions
        )
        payload["data"]["split_sha256"] = dict(
            dataset.split_sha256
        )
        payload["data"]["effective_train_bytes"] = (
            dataset.effective_train_bytes
        )
        payload["data"]["effective_train_sha256"] = (
            dataset.effective_train_sha256
        )
        payload["data"]["train_cleaning"] = dataset.train_cleaning
        payload["data"]["group_cuts"] = (
            dict(dataset.group_cuts)
            if dataset.group_cuts is not None
            else None
        )
        payload["data"]["groups_per_split"] = (
            dict(dataset.groups_per_split)
            if dataset.groups_per_split is not None
            else None
        )

    # resume 会重写 config.json；保留上次运行结束时补充的
    # efficiency 测量结果与调参来源记录。只有续跑才允许继承：
    # 在既有目录上全新训练（如复用输出目录）不能把上一次运行
    # 的读数错记到自己名下。
    config_path = out_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and getattr(args, "resume", None) is not None
        ):
            if "efficiency" in previous:
                payload.setdefault(
                    "efficiency",
                    previous["efficiency"],
                )
            # resume 重写 config.json 时保留原有的调参来源记录，
            # 避免已声明的「调过参的基线」在续跑后丢失。
            if (
                payload.get("baseline_tuning") is None
                and isinstance(
                    previous.get("baseline_tuning"), dict
                )
            ):
                payload["baseline_tuning"] = previous[
                    "baseline_tuning"
                ]
            # resume 通常不再重复传 --hypothesis/--rq；此时保留原
            # 运行记录的科学假设，避免重写 config.json 时丢失。
            for key in ("hypothesis", "research_question"):
                previous_value = previous.get(key)
                if (
                    not str(payload.get(key) or "").strip()
                    and isinstance(previous_value, str)
                    and previous_value.strip()
                ):
                    payload[key] = previous_value
            if (
                previous.get("hypothesis_check_skipped")
                and not payload["hypothesis_check_skipped"]
            ):
                payload["hypothesis_check_skipped"] = True
            # 预注册同样属于原运行的一部分：resume 未重新声明
            # 时不能把训练前的预注册记录抹掉。
            previous_prereg = previous.get(
                "hypothesis_preregistration"
            )
            if isinstance(previous_prereg, dict):
                current_prereg = payload.get(
                    "hypothesis_preregistration"
                ) or {}
                for key in (
                    "primary_metric",
                    "direction",
                    "min_effect",
                    "baseline",
                ):
                    if (
                        current_prereg.get(key) is None
                        and previous_prereg.get(key) is not None
                    ):
                        current_prereg[key] = previous_prereg[key]
                # equivalence_margin 的 CLI 默认值非 None，不能
                # 套用上面的「当前值为空才保留」规则：否则续跑
                # 会把原运行的非默认预注册界覆盖成默认值。
                previous_margin = _finite_float(
                    previous_prereg.get("equivalence_margin")
                )
                if (
                    previous_margin is not None
                    and previous_margin > 0.0
                    and current_prereg.get("equivalence_margin")
                    != previous_margin
                ):
                    print(
                        "resume preserved equivalence_margin "
                        f"{current_prereg.get('equivalence_margin')} "
                        f"-> {previous_margin} "
                        "(等价界是训练前预注册的一部分)"
                    )
                    current_prereg["equivalence_margin"] = float(
                        previous_margin
                    )
                payload["hypothesis_preregistration"] = (
                    current_prereg
                )

            # 外部预注册记录同样不能在 resume 时被抹掉：它是
            # 「这份运行在训练前对照过哪份独立文件」的唯一证据。
            # 当前 resume 若重新提供了预注册文件，则保留本次
            # 核对后的记录；否则继承原记录。
            if (
                payload.get("preregistration") is None
                and isinstance(
                    previous.get("preregistration"), dict
                )
            ):
                payload["preregistration"] = previous[
                    "preregistration"
                ]

            # 设计计划分配同样属于原运行：手工 resume 一个套件
            # 子运行时没有父进程环境变量，不能把「它当时是哪个
            # 区组/顺序」洗掉；顺序漂移诊断需要这段历史。
            if (
                payload.get("design_assignment") is None
                and isinstance(
                    previous.get("design_assignment"), dict
                )
            ):
                payload["design_assignment"] = previous[
                    "design_assignment"
                ]

            # resume 会重写 run_metadata（新的 timestamp/command/
            # 环境快照）。保留最初运行的元数据快照与续跑次数，
            # 避免「原始运行时间、原始命令、当时的环境」在多次
            # 续跑后丢失；否则一份从检查点续跑而来的结果，其
            # 溯源信息只能指向最后一次续跑。
            original_metadata = previous.get(
                "original_run_metadata"
            )
            if not isinstance(original_metadata, dict):
                original_metadata = previous.get("run_metadata")
            if isinstance(original_metadata, dict):
                payload["original_run_metadata"] = (
                    original_metadata
                )
            payload["resume_count"] = (
                int(previous.get("resume_count", 0)) + 1
            )
            payload["resumed_at_utc"] = run_metadata.get(
                "timestamp_utc"
            )

    # 假设的判定指标以最终生效的假设文本为准（resume 可能从
    # 原运行恢复了假设），不能沿用命令行空文本的解析结果。
    payload["hypothesis_metric_mentions"] = (
        hypothesis_metric_mentions(
            str(payload.get("hypothesis") or "")
        )
    )

    # 环境快照单独落一份 environment.json：它通常比 config.json
    # 小得多，便于跨运行 diff、在重跑前用 --check-environment
    # 核对，也能被 artifact_manifest 逐文件校验。resume 时保留
    # 最初运行的环境，当前环境差异只写入摘要。
    environment_record = write_run_environment_record(
        out_dir,
        run_metadata,
        payload.get("original_run_metadata"),
    )
    if environment_record is not None:
        payload["environment"] = environment_record

    # config.json 是实验追踪的唯一来源，写入中途失败不能留下
    # 半截文件：与 update_run_config 一样走临时文件 + 原子替换。
    previous_mode = 0o644
    if config_path.exists():
        try:
            previous_mode = config_path.stat().st_mode & 0o777
        except OSError:
            previous_mode = 0o644

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(out_dir),
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            # 先记录临时文件路径再写内容：json.dump 中途失败时
            # finally 才能清理掉半截文件。
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            # 先 flush+fsync 再原子替换：机器崩溃/被抢占时，
            # 不能让目录项指向一个尚未写入磁盘的 config.json。
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, config_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    try:
        os.chmod(config_path, previous_mode)
    except OSError:
        pass



def update_run_config(out_dir: Path, key: str, value) -> None:
    """
    训练结束后把补充的分析结果（如 efficiency）并入 config.json。

    读出已有 config.json、合并单个键后整体重写；config.json 的
    主体在训练开始前已经写出，这里只追加事后测量结果。
    """
    path = out_dir / "config.json"
    payload: Dict[str, object] = {}

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            # 不能把损坏的 config.json 当成空对象再用只含一个新
            # 键的内容原子替换：那会静默抹掉整份运行追踪信息。
            # 让调用方显式失败并保留原文件，便于人工恢复。
            raise ValueError(
                f"无法读取 {path}，拒绝用单键配置覆盖已有追踪"
                f"信息: {error}"
            ) from error
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path} 顶层不是 JSON 对象，拒绝用单键配置"
                "覆盖已有追踪信息"
            )
        payload = loaded

    payload[key] = value

    previous_mode = 0o644
    if path.exists():
        try:
            previous_mode = path.stat().st_mode & 0o777
        except OSError:
            previous_mode = 0o644

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(out_dir),
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            # 先记录临时文件路径再写内容：json.dump 中途失败时
            # finally 才能清理掉半截文件。
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    try:
        os.chmod(path, previous_mode)
    except OSError:
        pass



def write_jsonl_record(
    path: Path,
    record: Dict[str, object],
    append: bool = True,
) -> None:
    """
    追加（或重写）一行 JSON 记录到 metrics.jsonl。
    """
    with open(
        path,
        "a" if append else "w",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False,
            )
            + "\n"
        )



def _records_at_or_before_step(
    records: List[Dict[str, object]],
    step: int,
) -> List[Dict[str, object]]:
    """
    只保留 ``step <= step`` 的数值步记录，供 resume 恢复状态。

    metrics.jsonl 每个评估点都会落盘，但检查点只在存档点写入，
    两者节奏不一致（如 --eval-interval 2 --save-interval 4）时，
    崩溃前可能已写入「比 last.pt 更新的步」的评估记录。resume 若
    把这些尚未重新执行的记录也算进早停/选型/曲线状态，就会加入
    幻影验证点并产生重复 step；必须按恢复点过滤。非数值 step
    （reference_baselines / curve_monitor / final 等字符串记录）
    会由调用方按需自行过滤，这里只保留可比较的数值 step。
    """
    threshold = int(step)
    kept: List[Dict[str, object]] = []
    for record in records:
        record_step = record.get("step")
        if (
            isinstance(record_step, int)
            and not isinstance(record_step, bool)
            and record_step <= threshold
        ):
            kept.append(record)
    return kept



# 这些非数值 step 记录是「本次运行已经结束/已经分析完」的结果，
# resume 后旧结果不能继续冒充新训练段的终态；reference_baselines
# 与 curve_monitor 是可复用的历史说明，保留。
_RESUME_TERMINAL_STEP_LABELS = frozenset({
    "final",
    "bad_cases",
    "curve_analysis",
    "efficiency",
    "routing",
    "routing_interventions",
})



def _truncate_metrics_after_step(path: Path, step: int) -> None:
    """
    删除 metrics.jsonl 中属于恢复点之后的记录行。

    删除两类记录：数值 step 大于恢复点的记录，以及上一次运行
    已经写下的终态/分析记录（``final`` / ``bad_cases`` /
    ``curve_analysis`` / ``efficiency`` / ``routing`` 等）。否则
    resume 到更大 --steps 后、在新训练段崩溃时，旧 ``final``
    仍在文件里，后续 --compare 会把上一次的结果当成这次已完成
    的终态。``reference_baselines`` / ``curve_monitor`` 是可复用
    的历史说明，保留。文件不存在或无需裁剪时不改动。
    """
    if not _path_exists(path):
        return
    threshold = int(step)

    def should_drop(record: object) -> bool:
        if not isinstance(record, dict):
            return False
        record_step = record.get("step")
        if isinstance(record_step, int) and not isinstance(
            record_step, bool
        ):
            return record_step > threshold
        # step 是任意 JSON 值时不能直接放进 frozenset 做成员判断：
        # 损坏/手工编辑的记录可能写成 list/dict，会抛
        # TypeError 并让整个 resume 失败。只有字符串才可能匹配
        # 终态标签；其余值沿用「非终态说明，保留」的历史口径。
        if not isinstance(record_step, str):
            return False
        return record_step in _RESUME_TERMINAL_STEP_LABELS

    kept_lines: List[str] = []
    dropped = 0
    changed = False
    with open(path, "rb") as handle:
        raw_lines = handle.readlines()
    lines: List[Optional[str]] = []
    for raw_line in raw_lines:
        try:
            lines.append(raw_line.decode("utf-8"))
        except UnicodeDecodeError:
            # 损坏行不能留在恢复后的文件里，也不能让整个
            # resume 因一条坏记录失败。
            lines.append(None)
    for index, line in enumerate(lines):
        if line is None:
            changed = True
            continue
        last_line = index == len(lines) - 1
        # 文件末尾没有换行的最后一行：若解析失败（崩溃时写了
        # 半条 JSON），必须丢弃；若解析成功，保留但补换行。
        # 否则 resume 以追加模式写下的第一条记录会和它粘成
        # 一行，导致新记录被 read_metrics_records 静默跳过。
        if last_line and not line.endswith("\n"):
            stripped = line.strip()
            if not stripped:
                changed = True
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                changed = True
                continue
            if should_drop(record):
                dropped += 1
            else:
                kept_lines.append(line + "\n")
            changed = True
            continue
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        if should_drop(record):
            dropped += 1
            changed = True
            continue
        kept_lines.append(line)
    if not changed and dropped == 0:
        return
    temp_path = path.with_name(path.name + ".resume_tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)
    os.replace(temp_path, path)



def read_metrics_records(path: Path) -> List[Dict[str, object]]:
    """
    读取 metrics.jsonl 的全部记录；跳过损坏的行。

    训练结束后的曲线诊断、坏案例报告与 --compare 汇总都基于
    这份记录，保证分析与训练写盘使用同一份数据来源。
    """
    # 路径本身可能来自损坏/手工编辑的 config（例如超长字符串），
    # Path.exists()/open() 会抛 OSError（文件名过长、权限、IO）；
    # 分析层应把该运行当作「没有曲线记录」而不是中断整份报告。
    try:
        if not path.exists():
            return []
    except OSError:
        return []

    records: List[Dict[str, object]] = []

    # 逐行按字节解码：崩溃或磁盘损坏可能在任意一行留下非法
    # UTF-8，不能因此让整份指标记录（包括前面完好的行）不可读。
    try:
        with open(path, "rb") as f:
            for raw_line in f:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        # 读取中途失败时保留已经解析出的完整记录。
        return records

    return records



_TEST_SEAL_FILENAME = ".test_evaluated.json"



def _test_seal_path(out_dir: Path) -> Path:
    return Path(out_dir) / _TEST_SEAL_FILENAME



def test_evaluation_sealed(
    out_dir: Path,
    ledger_path: Optional[object] = None,
) -> bool:
    """
    测试集是否已在输出目录解封过。

    除 metrics.jsonl 的 test/test_started 记录外，还检查独立的
    持久封条文件：metrics.jsonl 被截断、重写或误删时，仅靠
    记录检查会让同一目录再次解封测试集，测试集因此退化为
    隐式验证集。封条与记录任一存在都视为已解封。

    ``ledger_path`` 给出 ``--test-ledger`` 全局账本时，也检查
    账本中是否已有本目录的访问记录：账本先于封条写入，进程在
    两者之间崩溃时本地目录里没有任何证据，不能只查本地文件。
    """
    out_dir = Path(out_dir)

    if test_record_exists(out_dir / "metrics.jsonl"):
        return True

    if _test_seal_path(out_dir).exists():
        return True

    # 访问账本先于 test_started/封条写入；若进程恰好在两者之间
    # 崩溃，前两项都看不到解封事实，但本目录的账本已经记录了
    # 一次测试访问。存在即视为已解封，宁可保守拒绝也不能让
    # 同一目录在 resume 后再次消耗测试集。
    if (out_dir / "test_access_ledger.jsonl").exists():
        return True

    if not ledger_path:
        return False
    try:
        ledger_file = Path(os.fspath(ledger_path))
    except (TypeError, ValueError):
        # 路径不可解释时按无法核对处理，保守视为已解封。
        return True
    entries, read_error = _read_test_access_ledger_entries(
        ledger_file
    )
    if read_error is not None:
        # 无法核对全局账本时按已解封处理，避免重复消耗测试集。
        return True
    return any(
        _run_paths_match(entry.get("out_dir"), out_dir)
        for entry in entries
    )



def fresh_out_dir_error(
    out_dir: Path,
    *,
    resume_path: Optional[str],
    allow_overwrite: bool,
) -> Optional[str]:
    """
    全新训练复用非空输出目录的错误消息（None 表示允许）。

    训练会截断 metrics.jsonl 并覆盖 config.json；目录里已有
    上一轮实验的追踪文件时，静默复用会把训练曲线、最终结果与
    原始代码/数据指纹从审计链中抹掉。--resume 与显式豁免
    --allow-overwrite-out 不走该拦截。
    """
    if resume_path is not None or allow_overwrite:
        return None

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None

    try:
        existing_entries = sorted(
            entry.name for entry in out_dir.iterdir()
        )
    except OSError as error:
        # 目录存在但无法枚举时不能当作空目录放行：训练随后
        # 可能覆盖看不见的旧追踪文件。
        return (
            f"--out 目录存在但无法读取（{error}）：无法确认是否为"
            "空目录；请修复权限或换用全新的 --out 目录。"
        )
    if not existing_entries:
        return None

    preview = ", ".join(existing_entries[:5])
    if len(existing_entries) > 5:
        preview += ", ..."
    return (
        f"--out 目录非空（{preview}）：全新训练会截断 "
        "metrics.jsonl 并覆盖 config.json，上一轮实验的训练"
        "曲线、最终结果与代码/数据指纹将不可追溯。请换用全新"
        "的 --out 目录；确需覆盖时显式传 "
        "--allow-overwrite-out。"
    )



def write_test_evaluation_seal(
    out_dir: Path,
    record: Optional[Dict[str, object]] = None,
) -> None:
    """
    在正式评估测试集前写持久封条（原子替换）。

    封条只标记「该输出目录已开始解封测试集」，不包含测试
    指标；即使评估中途崩溃，后续运行也会拒绝重复解封。
    """
    out_dir = Path(out_dir)
    path = _test_seal_path(out_dir)
    payload = {
        "sealed": True,
        "step": "test_started",
        "split": "test",
        **(record or {}),
    }

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(out_dir),
            prefix=".test-seal.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



def test_record_exists(path: Path) -> bool:
    """
    检查 metrics.jsonl 中是否已有 test 评估记录。

    测试集在一次运行中只允许解封评估一次：恢复训练后重复传入
    --evaluate-test 会被拒绝，避免测试集在反复查看中变成隐式
    验证集。
    """
    try:
        path_exists = path.exists()
    except OSError:
        # 路径本身不可判定时不能当作「没有 test 记录」放行
        # 后续操作：尘封协议宁可保守拒绝一次。
        return True
    if not path_exists:
        return False

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                # test 的坏案例记录先于 test 汇总记录写出；中断在
                # 两者之间时测试集实际已被评估过，同样视为已解封。
                # test_started 标记先于评估写出：评估中途崩溃时
                # 测试集实际已开始解封，不能因为没写成 test 记录
                # 就认为名额还在。
                if record.get("step") in ("test_started", "test") or (
                    record.get("step") == "bad_cases"
                    and record.get("split") == "test"
                ):
                    return True
    except (OSError, UnicodeDecodeError):
        # 文件存在但读不出内容（权限/IO 错误、非 UTF-8 字节）时
        # 不能当作「没有 test 记录」放行后续操作：尘封协议宁可
        # 保守拒绝一次，也不能冒重复解封测试集的风险。
        return True

    return False



def _read_test_records(run_dir: Path) -> Dict[str, object]:
    """读取单个运行的 test 汇总记录（最后一次 test 行）。"""
    path = Path(run_dir) / "metrics.jsonl"
    record: Optional[Dict[str, object]] = None
    if not _path_exists(path):
        return {"run": str(run_dir), "available": False}
    try:
        # 逐行按字节解码：崩溃可能只在文件末尾留下半行/非法
        # UTF-8；不能因此丢掉前面已经完整写出的 test 汇总。
        with open(path, "rb") as handle:
            for raw_line in handle:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("step") == "test":
                    record = item
    except OSError:
        return {"run": str(run_dir), "available": False}
    if record is None:
        return {"run": str(run_dir), "available": False}
    return {"run": str(run_dir), "available": True, **record}



def _read_test_access_ledger_entries(
    path: Path,
) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """读取账本并返回 ``(条目, 读取错误)``。

    账本可能由崩溃中的追加写入留下半行/非法 UTF-8；损坏的
    行应跳过，不能让最终对比因一条坏记录整份读不出来。文件
    权限/IO 导致的读取失败与「账本为空」不同：这里返回已解析
    出的完整条目与错误说明，调用方据账本做门控时必须把非 None
    的错误当作「无法核对」并按保守值处理，而不是当作空账本
    放行。
    """
    path = Path(path)
    if not _path_exists(path):
        return [], None
    entries: List[Dict[str, object]] = []
    try:
        with open(path, "rb") as handle:
            for raw_line in handle:
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    entries.append(item)
    except OSError as error:
        # 读取中途失败时保留已经解析出的完整记录；错误随条目
        # 一起返回，供调用方判断账本是否可信。
        return entries, str(error)
    return entries, None



def read_test_access_ledger(path: Path) -> List[Dict[str, object]]:
    """读取测试集访问账本（JSONL，只追加；读失败返回部分条目）。"""
    entries, _error = _read_test_access_ledger_entries(path)
    return entries



def append_test_access_ledger(
    path: Path,
    record: Dict[str, object],
) -> None:
    """原子追加一条测试集访问记录（flush+fsync）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(record), ensure_ascii=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())



def _run_path_key(
    value: object,
    base: object = None,
) -> Optional[str]:
    """把运行目录路径规范化为可比较的键。

    账本记录可能来自相对路径、绝对路径或带 ``./``/符号链接的
    写法；纯字符串比较会把同一目录误判成另一个候选。相对路径
    先按给定基准目录（运行元数据里的 cwd）还原，再解析。
    """
    if value is None:
        return None
    try:
        path = Path(os.fspath(value))
        if not path.is_absolute() and base is not None:
            path = Path(os.fspath(base)) / path
        return os.path.normcase(str(path.resolve(strict=False)))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None



def _run_paths_match(
    left: object,
    right: object,
    left_base: object = None,
) -> bool:
    """比较两个运行目录；无法解析时退回原字符串比较。"""
    left_key = _run_path_key(left, left_base)
    right_key = _run_path_key(right)
    if left_key is not None and right_key is not None:
        return left_key == right_key
    return str(left) == str(right)



def _test_access_key(
    data_sha256: object,
    split_sha256: object,
    test_fraction: object,
) -> Dict[str, object]:
    return {
        "data_sha256": data_sha256,
        "test_split_sha256": (
            (split_sha256 or {}).get("test")
            if isinstance(split_sha256, dict)
            else None
        ),
        "test_fraction": test_fraction,
    }



def _repeat_test_access_error(
    access_check: Dict[str, object],
) -> str:
    """测试集重复解封/账本不可读时的拒绝消息。"""
    if access_check.get("ledger_unreadable"):
        reason = (
            "访问账本不可读（"
            + str(access_check.get("ledger_unreadable"))
            + "），无法核对同一测试集是否已被其它候选解封"
        )
    else:
        other_candidates = (
            access_check.get("other_candidates") or []
        )
        same_candidates = (
            access_check.get("same_candidates") or []
        )
        candidates = ", ".join(
            str(item.get("out_dir"))
            for item in (other_candidates or same_candidates)[:5]
        )
        if other_candidates:
            reason = (
                "访问账本显示同一测试集已被其它候选解封（"
                + candidates
                + "）"
            )
        elif same_candidates:
            reason = (
                "访问账本显示本目录已登记过同一测试集的访问（"
                + candidates
                + "）；即使本地封条或 test 记录缺失，也不能"
                "再次解封"
            )
        else:
            reason = "访问账本显示同一测试集已有历史访问"
    return (
        "[test] 拒绝解封：" + reason
        + "。最终确认应冻结候选后统一解封；"
        "如确需重复访问，显式加 "
        "--allow-repeat-test-access，最终报告必须列出全部访问"
    )



def _test_access_entry_matches_key(
    entry: Dict[str, object],
    key: Dict[str, object],
) -> bool:
    """账本条目是否属于给定测试集；缺失指纹按无法核对保守匹配。"""
    for field, value in key.items():
        if value is None:
            continue
        if field in entry and entry.get(field) != value:
            return False
    return True



def check_test_access_ledger(
    ledger_path: Path,
    key: Dict[str, object],
    out_dir: Path,
) -> Dict[str, object]:
    """检查同一测试集是否已被其它候选/目录解封过。

    路径级封条只能防止同一目录重复解封；新建目录、换种子或
    改代码仍可再次访问同一测试集。账本按
    ``(data_sha256, test_split_sha256, test_fraction)`` 识别
    「同一测试集」，列出全部历史访问，供研究级流程核对候选
    冻结与预注册比较是否完整。
    """
    entries, read_error = _read_test_access_ledger_entries(ledger_path)
    matching = [
        entry
        for entry in entries
        if _test_access_entry_matches_key(entry, key)
    ]
    same_candidates: List[Dict[str, object]] = []
    other_candidates: List[Dict[str, object]] = []
    for entry in matching:
        if _run_paths_match(entry.get("out_dir"), out_dir):
            same_candidates.append(entry)
        else:
            other_candidates.append(entry)
    if read_error is not None:
        # 账本不可读时无法核对该测试集是否已被访问过；尘封
        # 协议宁可保守拒绝一次，也不能把「读不出来」当成
        # 「没有历史访问」而放行重复解封。
        return {
            "ledger": str(ledger_path),
            "matching_accesses": len(matching),
            "same_candidates": same_candidates,
            "other_candidates": other_candidates,
            "repeat_access": True,
            "ledger_unreadable": read_error,
            "note": (
                "访问账本无法读取（" + read_error + "）："
                "无法核对是否已重复解封，按保守值拒绝；"
                "修复账本权限/IO 后重试，或显式使用"
                "--allow-repeat-test-access 并留痕"
            ),
        }
    return {
        "ledger": str(ledger_path),
        "matching_accesses": len(matching),
        "same_candidates": same_candidates,
        "other_candidates": other_candidates,
        "repeat_access": bool(other_candidates),
        "already_accessed": bool(matching),
        "note": (
            "同一测试集被多个目录/候选访问时，测试集已不再"
            "只服务于一次最终确认；最终报告必须列出全部访问"
            "与预注册比较，不能只展示最好的候选"
        ),
    }



def _test_access_repeat_blocked(
    access_check: Dict[str, object],
) -> bool:
    """账本中已有本测试集访问（含同一目录）时是否拒绝解封。"""
    return bool(access_check.get("repeat_access")) or bool(
        access_check.get("matching_accesses")
    )



def print_final_test_comparison(
    run_dirs: List[str],
    ledger_path: Optional[str] = None,
) -> Dict[str, object]:
    """只读汇总冻结候选的 test 记录（研究级最终确认）。

    不做任何评估、不写运行目录；只读取各候选的 test 汇总与
    可选访问账本，打印所有候选（包括失败/无 test 记录的候选），
    避免「只报告跑通的那个」。多重比较与显著性需要后续按
    预注册主指标单独分析，这里先保证**所有**候选可见。
    """
    rows: List[Dict[str, object]] = []
    print("[final-test] 只读汇总冻结候选的测试集记录（不重新评估）")
    for run_dir in run_dirs:
        path = Path(run_dir)
        config: Dict[str, object] = {}
        try:
            config = json.loads(
                (path / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if not isinstance(config, dict):
            config = {}
        record = _read_test_records(path)
        cfg = config.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        config_args = config.get("args")
        if not isinstance(config_args, dict):
            config_args = {}
        row = {
            "run": str(path),
            "mode": cfg.get("mode"),
            "seed": config_args.get("seed"),
            "test_available": bool(record.get("available")),
            "test_loss": record.get("test_loss"),
            "test_ppl": record.get("test_ppl"),
            "test_bits_per_byte": record.get("test_bits_per_byte"),
            "test_token_accuracy": record.get(
                "test_token_accuracy"
            ),
            "selected_model": record.get("model_selection"),
            "selected_step": record.get("selected_step"),
        }
        rows.append(row)
        print(
            f"  {path}: mode={row['mode']} seed={row['seed']} "
            f"test_loss={row['test_loss']} "
            f"bpb={row['test_bits_per_byte']} "
            f"available={row['test_available']}"
        )
    missing = [
        row["run"] for row in rows if not row["test_available"]
    ]
    if missing:
        print(
            "  [warn] 以下候选没有 test 记录（可能未解封或评估失败），"
            "不能静默从最终比较中消失: " + ", ".join(missing)
        )
    ledger_info = None
    if ledger_path:
        entries, ledger_error = _read_test_access_ledger_entries(
            Path(ledger_path)
        )
        counts: Dict[str, int] = {}
        display_names: Dict[str, str] = {}
        for entry in entries:
            raw_name = str(entry.get("out_dir"))
            # 同一目录可能以相对/绝对/带符号链接等不同写法
            # 记入账本；按规范化路径计数，才能发现重复解封。
            path_key = _run_path_key(entry.get("out_dir"))
            if path_key is None:
                path_key = raw_name
            display_names.setdefault(path_key, raw_name)
            counts[path_key] = counts.get(path_key, 0) + 1
        per_candidate = {
            display_names[path_key]: count
            for path_key, count in counts.items()
        }
        ledger_info = {
            "ledger": ledger_path,
            "accesses": len(entries),
            "per_candidate": per_candidate,
            "unreadable": ledger_error,
        }
        repeated = [
            display_names[path_key]
            for path_key, count in counts.items()
            if count > 1
        ]
        if repeated:
            print(
                "  [warn] 以下候选被账本记录为重复解封测试集: "
                + ", ".join(repeated)
            )
        if ledger_error is not None:
            print(
                "  [warn] 访问账本无法读取（"
                + ledger_error
                + "）：无法核对重复解封，报告中的访问"
                "汇总不完整"
            )
    return {
        "candidates": rows,
        "missing_test_records": missing,
        "ledger": ledger_info,
        "protocol_note": (
            "最终确认必须先冻结候选（代码/配置/数据指纹），"
            "一次性解封测试集，并报告全部预注册比较；本命令"
            "只读汇总，不做选择或重复评估"
        ),
    }
