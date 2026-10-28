from __future__ import annotations



def audit_run_directory(
    root: Path,
    *,
    source_path: Optional[object] = None,
    data_path: Optional[object] = None,
    preregistration_path: Optional[object] = None,
    strict: bool = False,
) -> Dict[str, object]:
    """
    对一个已完成的运行目录做只读复现审计。

    检查的是**证据之间的链接**，不是重新训练：
    - config.json 的代码/数据/协议/环境指纹是否齐全且格式合法；
    - environment.json 的 environment_id 是否与自身内容、
      config.json 摘要及 run_metadata 一致；
    - metrics.jsonl 首行与 final/test 记录是否仍指向同一份
      代码、数据、环境、协议与种子（防止日志与配置拼接）；
    - test 记录/封条/访问账本是否互相矛盾（测试集是否可能被
      重复解封或解封后缺少汇总）；
    - sanity 重复运行是否确实逐位一致（同机同种子可复现的
      运行时证据）；
    - artifact_manifest.json 若存在，逐文件完整性是否通过；
    - 可选地核对归档源码、语料与预注册文件的外部 SHA-256。

    ``strict=True`` 时把「可复现性证据不足」的警告也判为失败
    （如未固定 PYTHONHASHSEED、未启用确定性算子、git 工作区
    脏、缺少 artifact_manifest），用于正式归档前的硬门。
    """
    root = Path(root)
    report: Dict[str, object] = {
        "schema": _RUN_AUDIT_SCHEMA,
        "run": str(root),
        "strict": bool(strict),
        "checks": {},
        "errors": [],
        "warnings": [],
        "ok": False,
    }
    errors: List[Dict[str, object]] = report["errors"]  # type: ignore[assignment]
    warnings: List[Dict[str, object]] = report["warnings"]  # type: ignore[assignment]
    checks: Dict[str, object] = report["checks"]  # type: ignore[assignment]

    def add_error(code: str, message: str, **details) -> None:
        errors.append({"code": code, "message": message, **details})

    def add_warning(code: str, message: str, **details) -> None:
        warnings.append({"code": code, "message": message, **details})

    if not root.is_dir():
        add_error(
            "run_dir_missing",
            f"运行目录不存在或不是目录: {root}",
        )
        report["ok"] = False
        return report

    # ---------- config.json ----------
    config_path = root / "config.json"
    config: Dict[str, object] = {}
    try:
        raw = config_path.read_bytes()
        loaded = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        add_error(
            "config_unreadable",
            f"无法读取 config.json: {error}",
        )
        loaded = None
    if not isinstance(loaded, dict):
        if not errors:
            add_error(
                "config_not_object",
                "config.json 顶层必须是 JSON 对象",
            )
        config = {}
    else:
        config = loaded
    checks["config"] = {
        "path": str(config_path),
        "schema": config.get("config_schema"),
        "available": bool(config),
    }
    if config:
        if config.get("config_schema") != _ENVIRONMENT_CONFIG_SCHEMA:
            add_error(
                "config_schema_mismatch",
                "config.json 的 config_schema 不是 "
                f"{_ENVIRONMENT_CONFIG_SCHEMA}",
                value=config.get("config_schema"),
            )

    args = config.get("args")
    if not isinstance(args, dict):
        add_error("config_args_missing", "config.json 缺少 args 对象")
        args = {}
    data = config.get("data")
    if not isinstance(data, dict):
        add_error("config_data_missing", "config.json 缺少 data 对象")
        data = {}
    protocol_sha = config.get("comparison_protocol_sha256")
    if not _audit_digest(protocol_sha):
        add_warning(
            "protocol_sha_missing",
            "config.json 缺少合法的 comparison_protocol_sha256："
            "metrics.jsonl 脱离 config 后无法核对协议是否一致",
        )
        protocol_sha = None

    # 续跑会重写 run_metadata；复现身份必须取最初运行快照。
    metadata = config.get("original_run_metadata")
    if not isinstance(metadata, dict):
        metadata = config.get("run_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        add_error(
            "run_metadata_missing",
            "config.json 缺少 run_metadata/original_run_metadata",
        )
    checks["provenance"] = {
        "resumed": isinstance(
            config.get("original_run_metadata"), dict
        ),
        "resume_count": config.get("resume_count", 0),
        "timestamp_utc": metadata.get("timestamp_utc"),
        "command": metadata.get("command"),
        "cwd": metadata.get("cwd"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "script_sha256": metadata.get("script_sha256"),
        "environment_id": metadata.get("environment_id"),
        "pythonhashseed_at_start": metadata.get(
            "pythonhashseed_at_start"
        ),
        "deterministic_algorithms": metadata.get(
            "deterministic_algorithms"
        ),
    }

    script_sha = metadata.get("script_sha256")
    if not _audit_digest(script_sha):
        add_error(
            "script_sha_missing",
            "run_metadata 缺少合法的 script_sha256："
            "无法证明结果由哪份代码产出",
        )
        script_sha = None
    if not str(metadata.get("command") or "").strip():
        add_warning(
            "command_missing",
            "run_metadata 缺少可重放的命令行",
        )
    if not str(metadata.get("cwd") or "").strip():
        add_warning(
            "cwd_missing",
            "run_metadata 缺少工作目录：相对数据/输出路径无法还原",
        )
    if metadata.get("git_dirty") is True:
        add_warning(
            "git_dirty",
            "运行时代码工作区是脏的：结果不能仅凭 git commit "
            "复现",
            git_diff_sha256=metadata.get("git_diff_sha256"),
        )
        if not _audit_digest(metadata.get("git_diff_sha256")):
            add_warning(
                "git_diff_sha_missing",
                "工作区脏但缺少 git_diff_sha256：无法区分同一 "
                "commit 下的不同未提交状态",
            )
    if not isinstance(metadata.get("rng_seed_manifest"), dict):
        add_warning(
            "rng_manifest_missing",
            "run_metadata 缺少 rng_seed_manifest：无法核对随机源"
            "是否实际播种",
        )
    if metadata.get("pythonhashseed_at_start") is None:
        add_warning(
            "pythonhashseed_unset",
            "进程启动时 PYTHONHASHSEED 未固定：严格逐位复现应"
            "在命令行前缀 PYTHONHASHSEED=<seed>",
        )
    if metadata.get("deterministic_algorithms") is not True:
        add_warning(
            "nondeterministic_algorithms",
            "未启用 PyTorch 确定性算法：同机同种子结果不保证"
            "逐位一致",
        )
    if metadata.get(
        "deterministic_algorithms_warn_only"
    ) is True:
        add_warning(
            "deterministic_warn_only",
            "确定性算法处于 warn_only 模式：遇到无确定性实现的"
            "算子只会告警，正式复现应使用 --strict-determinism",
        )
    if metadata.get("cudnn_deterministic") is not True:
        add_warning(
            "cudnn_nondeterministic",
            "cuDNN 未启用确定性模式",
        )
    if metadata.get("allow_tf32_matmul") is True or metadata.get(
        "allow_tf32_cudnn"
    ) is True:
        add_warning(
            "tf32_enabled",
            "启用了 TF32：矩阵乘尾数被截断，跨设备数值结果"
            "不可比",
        )

    # ---------- 数据指纹 ----------
    data_sha = data.get("sha256")
    if not _audit_digest(data_sha):
        add_error(
            "data_sha_missing",
            "config.data.sha256 缺失或不是 64 位十六进制："
            "无法证明训练语料是哪一份",
        )
        data_sha = None
    split_sha = data.get("split_sha256")
    split_ok = (
        isinstance(split_sha, dict)
        and all(
            _audit_digest(split_sha.get(name))
            for name in ("train", "val", "test")
        )
    )
    if not split_ok:
        add_warning(
            "split_sha_missing",
            "config.data.split_sha256 缺少 train/val/test 逐 split "
            "指纹：整体文件相同但实际切分不同时无法发现",
        )
        split_sha = None

    # ---------- environment.json ----------
    environment_id = metadata.get("environment_id")
    try:
        snapshot, snapshot_source = load_environment_snapshot(root)
    except ValueError as error:
        add_error(
            "environment_unreadable",
            f"无法读取环境快照: {error}",
        )
        snapshot = None
        snapshot_source = None
    if isinstance(snapshot, dict):
        actual_environment_id = snapshot.get("environment_id")
        if not _audit_digest(actual_environment_id):
            add_error(
                "environment_id_missing",
                "environment.json 缺少合法的 environment_id",
            )
            actual_environment_id = None
        else:
            canonical = _canonical_json_sha256(
                _environment_identity_payload(snapshot)
            )
            if actual_environment_id != canonical:
                add_error(
                    "environment_id_inconsistent",
                    "environment.json 的 environment_id 与快照内容"
                    "不一致（文件可能被手工编辑）",
                    recorded=actual_environment_id,
                    recomputed=canonical,
                )
        if (
            isinstance(environment_id, str)
            and actual_environment_id is not None
            and environment_id != actual_environment_id
        ):
            add_error(
                "environment_id_mismatch",
                "config.run_metadata.environment_id 与 "
                "environment.json 不一致",
                config=environment_id,
                snapshot=actual_environment_id,
            )
        checks["environment"] = {
            "source": str(snapshot_source),
            "environment_id": actual_environment_id,
            "schema": snapshot.get("schema"),
        }

        summary = config.get("environment")
        if isinstance(summary, dict):
            summary_id = summary.get("environment_id")
            if (
                isinstance(summary_id, str)
                and actual_environment_id is not None
                and summary_id != actual_environment_id
            ):
                add_error(
                    "environment_summary_mismatch",
                    "config.environment 摘要的 id 与 "
                    "environment.json 不一致",
                    summary=summary_id,
                    snapshot=actual_environment_id,
                )
            file_name = summary.get("file")
            file_sha = summary.get("file_sha256")
            if isinstance(file_name, str) and file_name:
                environment_path = root / file_name
                if not environment_path.is_file():
                    add_error(
                        "environment_file_missing",
                        "config.environment 指向的 "
                        f"{file_name} 不存在",
                    )
                elif _audit_digest(file_sha):
                    try:
                        actual_file_sha = _sha256_file(
                            environment_path
                        )
                    except OSError as error:
                        add_error(
                            "environment_file_unreadable",
                            f"无法读取 {file_name}: {error}",
                        )
                    else:
                        if actual_file_sha != file_sha:
                            add_error(
                                "environment_file_sha_mismatch",
                                f"{file_name} 的 SHA-256 与 "
                                "config.environment 记录不一致",
                                expected=file_sha,
                                actual=actual_file_sha,
                            )
                else:
                    add_warning(
                        "environment_file_sha_missing",
                        "config.environment 缺少 file_sha256："
                        "无法证明 environment.json 未被替换",
                    )
    elif environment_id is not None:
        add_warning(
            "environment_snapshot_missing",
            "config 记录了 environment_id，但没有可读取的"
            " environment.json/内嵌快照",
        )

    # ---------- metrics.jsonl 与跨文件身份 ----------
    metrics_path = root / "metrics.jsonl"
    records: List[Dict[str, object]] = []
    metrics_log: Dict[str, object] = {}
    if not metrics_path.is_file():
        add_error(
            "metrics_missing",
            "缺少 metrics.jsonl：运行曲线与终态记录不可核对",
        )
    else:
        metrics_log = _audit_metrics_log(metrics_path)
        if metrics_log.get("error"):
            add_error(
                "metrics_unreadable",
                "无法读取 metrics.jsonl: "
                f"{metrics_log.get('error')}",
            )
        loaded_records = metrics_log.get("records")
        if isinstance(loaded_records, list):
            records = [
                record
                for record in loaded_records
                if isinstance(record, dict)
            ]
        if not records:
            add_error(
                "metrics_empty",
                "metrics.jsonl 没有可解析的记录",
            )
        malformed_lines = metrics_log.get("malformed_lines") or []
        invalid_utf8_lines = (
            metrics_log.get("invalid_utf8_lines") or []
        )
        blank_lines = metrics_log.get("blank_lines") or []
        if malformed_lines or invalid_utf8_lines:
            add_error(
                "metrics_malformed_lines",
                "metrics.jsonl 含无法解析或非 UTF-8 的行：日志"
                "可能被截断、拼接或手工编辑",
                malformed_lines=malformed_lines,
                invalid_utf8_lines=invalid_utf8_lines,
            )
        if metrics_log.get("trailing_partial"):
            add_warning(
                "metrics_trailing_partial",
                "metrics.jsonl 末尾是没有换行的半行：进程可能在"
                "写记录时崩溃，最后一条记录不可信",
            )
        if blank_lines:
            add_warning(
                "metrics_blank_lines",
                "metrics.jsonl 含空行：日志可能被人工插入内容",
                lines=blank_lines,
            )
        duplicate_steps = metrics_log.get("duplicate_steps") or []
        nonmonotonic_steps = (
            metrics_log.get("nonmonotonic_steps") or []
        )
        if duplicate_steps:
            add_error(
                "metrics_duplicate_steps",
                "metrics.jsonl 含重复的训练步记录：日志可能由两次"
                "运行拼接或写入了重复评估",
                duplicates=duplicate_steps,
            )
        if nonmonotonic_steps:
            add_error(
                "metrics_nonmonotonic_steps",
                "metrics.jsonl 的训练步不是严格递增：日志可能"
                "被拼接、重排或来自恢复点之前",
                transitions=nonmonotonic_steps,
            )

    def check_record_identity(
        record: Dict[str, object],
        label: str,
        *,
        require_fields: bool,
    ) -> None:
        expected = {
            "seed": args.get("seed"),
            "data_sha256": data_sha,
            "data_split_sha256": split_sha,
            "script_sha256": script_sha,
            "environment_id": environment_id,
            "comparison_protocol_sha256": protocol_sha,
        }
        for key, value in expected.items():
            if value is None:
                continue
            if key not in record:
                if require_fields:
                    add_warning(
                        "record_field_missing",
                        f"{label} 缺少 {key}：无法与 config 核对",
                        record=label,
                        field=key,
                    )
                continue
            if not _audit_same(record.get(key), value):
                add_error(
                    "record_identity_mismatch",
                    f"{label} 的 {key} 与 config 不一致"
                    "（日志与配置可能来自不同运行）",
                    record=label,
                    field=key,
                    expected=value,
                    actual=record.get(key),
                )
        # 这些字段只在记录存在时核对：旧格式日志/运行可能没有
        # 它们，缺失本身不新增警告（保持旧运行可审计），但一旦
        # 写入就必须与 config 指向同一次运行。
        optional_expected = {
            "git_commit": metadata.get("git_commit"),
            "git_dirty": metadata.get("git_dirty"),
            "pythonhashseed_at_start": metadata.get(
                "pythonhashseed_at_start"
            ),
            "deterministic_algorithms": metadata.get(
                "deterministic_algorithms"
            ),
            "rng_seed_manifest": metadata.get("rng_seed_manifest"),
            "data_effective_train_sha256": data.get(
                "effective_train_sha256"
            ),
            "run_started_utc": metadata.get("timestamp_utc"),
        }
        for key, value in optional_expected.items():
            if value is None or key not in record:
                continue
            if not _audit_same(record.get(key), value):
                add_error(
                    "record_identity_mismatch",
                    f"{label} 的 {key} 与 config 不一致"
                    "（日志与配置可能来自不同运行）",
                    record=label,
                    field=key,
                    expected=value,
                    actual=record.get(key),
                )

    headers = [
        record
        for record in records
        if record.get("step") == "reference_baselines"
    ]
    if records and not headers:
        add_error(
            "metrics_header_missing",
            "metrics.jsonl 缺少 reference_baselines 首行："
            "无法核对日志属于哪次运行",
        )
    if headers:
        check_record_identity(
            headers[0],
            "reference_baselines",
            require_fields=True,
        )
        if len(headers) > 1:
            add_error(
                "multiple_reference_baselines",
                "metrics.jsonl 含多条 reference_baselines 首行记录："
                "日志可能由多次运行拼接，不能当作一次连续运行",
                count=len(headers),
            )
        if records and records[0] is not headers[0]:
            add_error(
                "metrics_header_not_first",
                "reference_baselines 不是 metrics.jsonl 的首条记录："
                "首行之前存在其他记录，日志可能被拼接或篡改",
            )
        config_reference = config.get("reference_baselines")
        if isinstance(config_reference, dict) and config_reference:
            missing = [
                key
                for key in config_reference
                if key not in headers[0]
            ]
            mismatched = {
                key: {
                    "expected": value,
                    "actual": headers[0].get(key),
                }
                for key, value in config_reference.items()
                if key in headers[0]
                and not _audit_same(headers[0].get(key), value)
            }
            if missing:
                add_warning(
                    "metrics_reference_baseline_missing",
                    "metrics.jsonl 首行缺少 config.reference_baselines "
                    "中的部分基线：无法用日志核对评估窗口",
                    fields=missing,
                )
            if mismatched:
                add_error(
                    "metrics_reference_baseline_mismatch",
                    "metrics.jsonl 首行的参考基线与 config."
                    "reference_baselines 不一致：日志与配置可能"
                    "使用了不同的评估窗口",
                    fields=mismatched,
                )

    final_records = [
        record
        for record in records
        if record.get("step") == "final"
    ]
    test_records = [
        record
        for record in records
        if record.get("step") == "test"
    ]
    test_started_records = [
        record
        for record in records
        if record.get("step") == "test_started"
    ]
    test_bad_case_records = [
        record
        for record in records
        if record.get("step") == "bad_cases"
        and record.get("split") == "test"
    ]
    final_indices = [
        index
        for index, record in enumerate(records)
        if record.get("step") == "final"
    ]
    test_indices = [
        index
        for index, record in enumerate(records)
        if record.get("step") == "test"
    ]
    test_started_indices = [
        index
        for index, record in enumerate(records)
        if record.get("step") == "test_started"
    ]
    if len(final_records) > 1:
        add_error(
            "multiple_final_records",
            "metrics.jsonl 含多条 final 记录：日志可能由多次"
            "运行拼接，终态结果不唯一",
            count=len(final_records),
        )
    if final_indices and test_indices:
        if min(test_indices) < max(final_indices):
            add_error(
                "test_before_final",
                "metrics.jsonl 的 test 记录早于 final 记录：测试集"
                "可能在最终模型选定前就被解封，终态证据链失效",
            )
    if test_indices and test_started_indices:
        first_test = min(test_indices)
        if min(test_started_indices) > first_test:
            add_error(
                "test_started_after_test",
                "test_started 标记出现在 test 汇总之后：解封顺序"
                "不成立，日志可能被重排",
            )
    elif test_indices and not test_started_indices:
        add_warning(
            "test_started_missing",
            "存在 test 汇总但没有 test_started 标记：无法从日志"
            "确认解封前已登记访问",
        )
    if test_indices:
        last_test = max(test_indices)
        trailing = [
            {
                "index": index,
                "step": records[index].get("step"),
            }
            for index in range(last_test + 1, len(records))
        ]
        if trailing:
            add_error(
                "records_after_test",
                "test 汇总之后仍有 metrics 记录：测试集解封后不应"
                "再追加结果或调参记录",
                records=trailing,
            )
    if final_indices:
        last_final = max(final_indices)
        numeric_after_final = [
            {
                "index": index,
                "step": records[index].get("step"),
            }
            for index in range(last_final + 1, len(records))
            if isinstance(records[index].get("step"), int)
            and not isinstance(records[index].get("step"), bool)
        ]
        if numeric_after_final:
            add_error(
                "metrics_numeric_records_after_final",
                "final 记录之后仍有训练步记录：终态汇总早于最后"
                "一次评估，日志可能被截断或拼接",
                records=numeric_after_final,
            )
    for record in final_records:
        check_record_identity(
            record,
            "final",
            require_fields=False,
        )
    for record in test_records:
        check_record_identity(
            record,
            "test",
            require_fields=False,
        )
    if len(test_records) > 1:
        add_error(
            "multiple_test_records",
            "metrics.jsonl 含多条 test 汇总记录：测试集可能被"
            "重复解封，不能当作一次最终确认",
            count=len(test_records),
        )
    if len(test_started_records) > 1:
        add_error(
            "multiple_test_started",
            "metrics.jsonl 含多条 test_started 标记：测试集可能"
            "被重复解封（每次解封都会先写该标记），不能当作"
            "一次最终确认",
            count=len(test_started_records),
        )
    if len(test_bad_case_records) > 1:
        add_error(
            "multiple_test_bad_cases",
            "metrics.jsonl 含多条 test 坏案例记录：测试集可能"
            "被重复评估，不能当作一次最终确认",
            count=len(test_bad_case_records),
        )

    checks["metrics"] = {
        "path": str(metrics_path),
        "lines": metrics_log.get("lines", 0),
        "records": len(records),
        "reference_baselines": len(headers),
        "final": len(final_records),
        "test": len(test_records),
        "malformed_lines": metrics_log.get("malformed_lines", []),
        "invalid_utf8_lines": metrics_log.get(
            "invalid_utf8_lines", []
        ),
        "blank_lines": metrics_log.get("blank_lines", []),
        "trailing_partial": metrics_log.get(
            "trailing_partial", False
        ),
        "duplicate_steps": metrics_log.get(
            "duplicate_steps", []
        ),
        "nonmonotonic_steps": metrics_log.get(
            "nonmonotonic_steps", []
        ),
    }

    # ---------- test 封条与访问账本 ----------
    seal_path = _test_seal_path(root)
    seal_exists = seal_path.is_file()
    test_log_evidence = bool(
        test_records
        or test_started_records
        or test_bad_case_records
    )
    # 访问账本可能写在全局路径（--test-ledger），不一定在运行
    # 目录内；配置里的相对路径按运行时的 cwd 还原，避免把
    # 合法的全局账本误报为缺失。
    local_ledger_path = root / "test_access_ledger.jsonl"
    ledger_candidates = [local_ledger_path]
    configured_ledger = args.get("test_ledger")
    if isinstance(configured_ledger, str) and configured_ledger.strip():
        configured_path = Path(configured_ledger)
        ledger_candidates.insert(0, configured_path)
        cwd = metadata.get("cwd")
        if (
            not configured_path.is_absolute()
            and isinstance(cwd, str)
            and cwd.strip()
        ):
            ledger_candidates.insert(0, Path(cwd) / configured_path)
    existing_ledger_paths: List[Path] = []
    seen_ledger_paths = set()
    for ledger_path in ledger_candidates:
        try:
            exists = ledger_path.is_file()
        except OSError:
            exists = False
        try:
            path_key = str(ledger_path.resolve())
        except OSError:
            path_key = str(ledger_path)
        if exists and path_key not in seen_ledger_paths:
            seen_ledger_paths.add(path_key)
            existing_ledger_paths.append(ledger_path)
    entries: List[Dict[str, object]] = []
    ledger_read_errors: List[str] = []
    for ledger_path in existing_ledger_paths:
        ledger_entries, ledger_error = (
            _read_test_access_ledger_entries(ledger_path)
        )
        entries.extend(ledger_entries)
        if ledger_error is not None:
            ledger_read_errors.append(
                f"{ledger_path}: {ledger_error}"
            )
    metadata_cwd = metadata.get("cwd")
    if not isinstance(metadata_cwd, str) or not metadata_cwd.strip():
        metadata_cwd = None
    audit_access_key = _test_access_key(
        data_sha,
        split_sha,
        data.get("test_fraction"),
    )
    matching = [
        entry
        for entry in entries
        if _test_access_entry_matches_key(
            entry, audit_access_key
        )
        and (
            _run_paths_match(
                entry.get("out_dir"),
                root,
                left_base=metadata_cwd,
            )
            or _run_paths_match(
                entry.get("out_dir"),
                root,
            )
        )
    ]
    # 访问账本先于 test_started/封条写入；若进程在这之间崩溃，
    # 账本条目是唯一解封证据。账本既可能在运行目录，也可能是
    # --test-ledger 指向的全局文件；两者都必须计入，否则会把
    # 中途崩溃的候选误判成「从未解封」。
    local_ledger_evidence = _path_exists(local_ledger_path)
    test_evidence = bool(
        test_log_evidence
        or seal_exists
        or local_ledger_evidence
        or matching
    )
    if test_evidence and not seal_exists:
        add_warning(
            "test_seal_missing",
            "存在测试访问证据（test 记录、开始标记或访问账本）"
            "但没有持久封条：旧运行可能未写封条；不能仅凭"
            " metrics.jsonl 证明测试集只解封一次",
        )
    if test_evidence and not test_records:
        add_warning(
            "test_started_without_summary",
            "测试集已开始解封（封条、test_started/坏案例记录"
            "或访问账本存在）但没有 test 汇总记录：该候选"
            "不能作为最终确认结果",
        )
    if test_evidence:
        if not existing_ledger_paths:
            add_warning(
                "test_ledger_missing",
                "存在测试集访问证据（test/test_started/封条）但"
                "没有访问账本：无法核对这次解封是否已登记",
            )
        else:
            if ledger_read_errors:
                add_warning(
                    "test_ledger_unreadable",
                    "访问账本存在但无法读取：无法核对这次解封"
                    "是否已登记",
                    errors=ledger_read_errors,
                )
            if not matching and not ledger_read_errors:
                add_warning(
                    "test_ledger_entry_missing",
                    "本目录有 test 记录，但访问账本中没有对应条目："
                    "无法核对这次解封是否已登记",
                )
    if len(matching) > 1:
        details = {
            "count": len(matching),
            "out_dirs": [
                str(entry.get("out_dir")) for entry in matching[:5]
            ],
        }
        if args.get("allow_repeat_test_access") is True:
            add_warning(
                "test_ledger_repeated_access",
                "访问账本含多条本目录的测试集访问记录；"
                "已显式允许重复解封，最终报告必须列出全部访问",
                **details,
            )
        else:
            add_error(
                "multiple_test_ledger_entries",
                "访问账本含多条本目录的测试集访问记录：测试集"
                "可能被重复解封，不能当作一次最终确认",
                **details,
            )
    checks["test_seal"] = {
        "seal_exists": seal_exists,
        "test_records": len(test_records),
        "test_started": len(test_started_records),
        "test_bad_cases": len(test_bad_case_records),
        "ledger_exists": bool(existing_ledger_paths),
        "ledger_matches": len(matching),
    }

    # ---------- sanity 重复运行 ----------
    sanity = config.get("sanity_check")
    if not isinstance(sanity, dict):
        add_warning(
            "sanity_check_missing",
            "config.json 缺少 sanity_check：没有训练前过拟合/"
            "梯度探针证据",
        )
    else:
        reproducibility = sanity.get("reproducibility")
        if not isinstance(reproducibility, dict):
            add_warning(
                "sanity_repeat_missing",
                "sanity_check 缺少重复运行记录：同机同种子逐位"
                "一致尚未被验证",
            )
        elif reproducibility.get("identical") is False:
            add_error(
                "sanity_repeat_mismatch",
                "同配置/同种子的 sanity 重复运行结果不一致："
                "当前环境不能逐位复现",
                mismatches=reproducibility.get("mismatches"),
            )
        elif reproducibility.get("identical") is None:
            add_warning(
                "sanity_repeat_not_verified",
                "sanity 重复运行次数不足 2：未验证同机同种子"
                "逐位一致",
                repeats=reproducibility.get("repeats"),
            )

    # ---------- artifact_manifest.json ----------
    manifest_path = root / _ARTIFACT_MANIFEST_NAME
    if manifest_path.is_file():
        manifest_report = verify_artifact_manifest(
            root, strict=False
        )
        untracked = [
            item
            for item in manifest_report.get("untracked", [])
            if item != _RUN_AUDIT_REPORT_FILENAME
        ]
        checks["artifact_manifest"] = {
            "path": str(manifest_path),
            "manifest_sha256": manifest_report.get(
                "manifest_sha256"
            ),
            "integrity_ok": manifest_report.get(
                "integrity_ok"
            ),
            "untracked": untracked,
        }
        if not manifest_report.get("ok"):
            add_error(
                "manifest_integrity",
                "artifact_manifest.json 完整性校验失败：目录内容"
                "与清单不一致",
                missing=manifest_report.get("missing"),
                mismatched=manifest_report.get("mismatched"),
                errors=manifest_report.get("errors"),
            )
        if untracked:
            add_warning(
                "manifest_untracked",
                "存在清单外文件：归档前需重新生成清单，或确认"
                "这些文件不属于归档内容",
                untracked=untracked,
            )
    else:
        add_warning(
            "manifest_missing",
            "缺少 artifact_manifest.json：归档内容尚未逐文件"
            "固定（正式归档前应执行 --write-manifest）",
        )

    # ---------- 外部输入（可选） ----------
    if source_path is not None:
        source = Path(source_path)
        entry: Dict[str, object] = {"path": str(source)}
        if not source.is_file():
            add_error(
                "source_missing",
                f"待核对源码不存在: {source}",
            )
        else:
            try:
                actual = _sha256_file(source)
            except OSError as error:
                add_error(
                    "source_unreadable",
                    f"无法读取源码 {source}: {error}",
                )
                actual = None
            entry["sha256"] = actual
            entry["matches"] = (
                None
                if actual is None or script_sha is None
                else actual == script_sha
            )
            if entry["matches"] is False:
                add_error(
                    "source_sha_mismatch",
                    "归档源码的 SHA-256 与运行记录的 script_sha256 "
                    "不一致",
                    expected=script_sha,
                    actual=actual,
                )
        checks["source"] = entry

    if data_path is not None:
        data_file = Path(data_path)
        entry = {"path": str(data_file)}
        if not data_file.is_file():
            add_error(
                "data_file_missing",
                f"待核对语料不存在: {data_file}",
            )
        else:
            try:
                actual = _sha256_file(data_file)
            except OSError as error:
                add_error(
                    "data_file_unreadable",
                    f"无法读取语料 {data_file}: {error}",
                )
                actual = None
            entry["sha256"] = actual
            entry["matches"] = (
                None
                if actual is None or data_sha is None
                else actual == data_sha
            )
            if entry["matches"] is False:
                add_error(
                    "data_sha_mismatch",
                    "归档语料的 SHA-256 与 config.data.sha256 "
                    "不一致",
                    expected=data_sha,
                    actual=actual,
                )
        checks["data_file"] = entry

    if preregistration_path is not None:
        prereg_file = Path(preregistration_path)
        entry = {"path": str(prereg_file)}
        record = config.get("preregistration")
        if not isinstance(record, dict):
            add_error(
                "preregistration_record_missing",
                "运行目录没有预注册核对记录，无法与外部预注册"
                "文件比对",
            )
        elif not prereg_file.is_file():
            add_error(
                "preregistration_file_missing",
                f"预注册文件不存在: {prereg_file}",
            )
        else:
            try:
                prereg_payload, prereg_sha = load_preregistration(
                    prereg_file
                )
            except ValueError as error:
                add_error(
                    "preregistration_invalid",
                    f"无法读取预注册文件: {error}",
                )
                prereg_payload = None
                prereg_sha = None
            entry["sha256"] = prereg_sha
            entry["matches"] = (
                None
                if prereg_sha is None
                else prereg_sha == record.get("file_sha256")
            )
            if entry["matches"] is False:
                add_error(
                    "preregistration_sha_mismatch",
                    "外部预注册文件的 SHA-256 与运行记录不一致"
                    "（文件在运行后可能被编辑）",
                    expected=record.get("file_sha256"),
                    actual=prereg_sha,
                )
            if isinstance(prereg_payload, dict):
                for key in (
                    "study_id",
                    "frozen_commit",
                    "primary_metric",
                    "common_step",
                    "direction",
                    "min_effect",
                    "baseline_mode",
                    "comparisons",
                    "seeds",
                    "budget_match",
                ):
                    if not _audit_same(
                        prereg_payload.get(key),
                        record.get(key),
                    ):
                        add_error(
                            "preregistration_field_mismatch",
                            f"预注册字段 {key} 与运行记录不一致",
                            field=key,
                            expected=record.get(key),
                            actual=prereg_payload.get(key),
                        )
                prereg_data = prereg_payload.get("data")
                if isinstance(prereg_data, dict) and data_sha is not None:
                    if not _audit_same(
                        prereg_data.get("sha256"), data_sha
                    ):
                        add_error(
                            "preregistration_data_mismatch",
                            "预注册 data.sha256 与运行语料不一致",
                            expected=data_sha,
                            actual=prereg_data.get("sha256"),
                        )
                frozen_script = prereg_payload.get(
                    "frozen_script_sha256"
                )
                if (
                    _audit_digest(frozen_script)
                    and script_sha is not None
                    and frozen_script.lower() != script_sha.lower()
                ):
                    add_error(
                        "frozen_script_mismatch",
                        "预注册 frozen_script_sha256 与运行记录的"
                        " script_sha256 不一致",
                        expected=script_sha,
                        actual=frozen_script,
                    )
        checks["preregistration"] = entry

    report["summary"] = {
        "errors": len(errors),
        "warnings": len(warnings),
        "strict": bool(strict),
    }
    report["ok"] = not errors and not (strict and warnings)
    return report



def print_run_audit(report: Dict[str, object]) -> None:
    """打印运行级复现审计摘要；完整报告由调用方决定是否落盘。"""
    print(f"[audit] run={report.get('run')}")
    checks = report.get("checks")
    if isinstance(checks, dict):
        config = checks.get("config")
        if isinstance(config, dict):
            print(
                "[audit] config schema="
                f"{config.get('schema')}"
            )
        provenance = checks.get("provenance")
        if isinstance(provenance, dict):
            print(
                "[audit] code="
                f"{provenance.get('script_sha256')} "
                f"git={provenance.get('git_commit')} "
                f"dirty={provenance.get('git_dirty')}"
            )
        environment = checks.get("environment")
        if isinstance(environment, dict):
            print(
                "[audit] environment_id="
                f"{environment.get('environment_id')}"
            )
        metrics = checks.get("metrics")
        if isinstance(metrics, dict):
            print(
                "[audit] metrics records="
                f"{metrics.get('records')} "
                f"final={metrics.get('final')} "
                f"test={metrics.get('test')}"
            )
    for item in report.get("errors", []):
        if isinstance(item, dict):
            print(
                f"[audit][error] {item.get('code')}: "
                f"{item.get('message')}"
            )
    for item in report.get("warnings", []):
        if isinstance(item, dict):
            print(
                f"[audit][warn] {item.get('code')}: "
                f"{item.get('message')}"
            )
    if report.get("ok"):
        print(
            "[audit] OK: config / environment / metrics / "
            "归档证据互相一致"
        )
    else:
        print(
            "[audit] FAILED: 复现证据不一致或不足；"
            "修复/补齐后再归档，不能把该运行当作可复现证据"
        )



def audit_mode_error(args) -> Optional[str]:
    """校验 --audit-run 的模式与冲突参数。"""
    audit_run = getattr(args, "audit_run", None)
    report_path = getattr(args, "audit_report", None)
    source_path = getattr(args, "audit_source", None)
    data_path = getattr(args, "audit_data", None)
    prereg_path = getattr(
        args, "audit_preregistration", None
    )
    strict = bool(getattr(args, "audit_strict", False))

    if audit_run is None:
        extras = [
            flag
            for flag, active in (
                ("--audit-report", report_path is not None),
                ("--audit-source", source_path is not None),
                ("--audit-data", data_path is not None),
                (
                    "--audit-preregistration",
                    prereg_path is not None,
                ),
                ("--audit-strict", strict),
            )
            if active
        ]
        if extras:
            return (
                "以下参数必须与 --audit-run 一起使用: "
                + ", ".join(extras)
            )
        return None

    conflicts = [
        flag
        for flag, active in (
            ("--compare", getattr(args, "compare", None) is not None),
            (
                "--compare-report",
                getattr(args, "compare_report", None) is not None,
            ),
            (
                "--final-test-comparison",
                getattr(args, "final_test_comparison", None)
                is not None,
            ),
            (
                "--write-manifest",
                getattr(args, "write_manifest", None) is not None,
            ),
            (
                "--verify-manifest",
                getattr(args, "verify_manifest", None) is not None,
            ),
            (
                "--expect-manifest-sha256",
                getattr(
                    args, "expect_manifest_sha256", None
                )
                is not None,
            ),
            (
                "--verify-manifest-strict",
                bool(
                    getattr(
                        args, "verify_manifest_strict", False
                    )
                ),
            ),
            (
                "--check-environment",
                getattr(args, "check_environment", None)
                is not None,
            ),
            (
                "--check-environment-strict",
                bool(
                    getattr(
                        args, "check_environment_strict", False
                    )
                ),
            ),
            ("--sanity-only", bool(getattr(args, "sanity_only", False))),
            (
                "--statistical-baseline",
                bool(getattr(args, "statistical_baseline", False)),
            ),
            (
                "--suite-baseline-tuning",
                bool(getattr(args, "suite_baseline_tuning", False)),
            ),
            (
                "--preregistration",
                getattr(args, "preregistration", None) is not None,
            ),
            (
                "--plan-design",
                bool(getattr(args, "plan_design", False)),
            ),
            (
                "--design-pilot-runs",
                getattr(args, "design_pilot_runs", None) is not None,
            ),
            (
                "--suite-modes",
                getattr(args, "suite_modes", None) is not None,
            ),
            (
                "--suite-ablation",
                bool(getattr(args, "suite_ablation", False)),
            ),
            ("--data", getattr(args, "data", None) is not None),
            ("--out", getattr(args, "out", None) is not None),
            ("--seeds", getattr(args, "seeds", None) is not None),
            ("--resume", getattr(args, "resume", None) is not None),
            (
                "--evaluate-test",
                bool(getattr(args, "evaluate_test", False)),
            ),
            (
                "--paper-cell",
                getattr(args, "paper_cell", None) is not None,
            ),
        )
        if active
    ]
    if conflicts:
        return (
            "运行级复现审计（--audit-run）是只读入口，不能与"
            "以下训练/分析参数同时使用: "
            + ", ".join(conflicts)
        )
    return None



def training_batch_fingerprint(
    dataset: ByteDataset,
    seed: int,
    batch_size: int,
    batches: int = 2,
) -> Dict[str, object]:
    """
    用独立 CPU 随机流复现训练批采样，记录前若干批的内容指纹。

    该指纹只读数据、不改变训练 RNG；同种子运行若协议完全
    一致，指纹必须逐位相同，否则说明数据顺序/批次组成已经
    成为未声明变量。
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    digest = hashlib.sha256()
    completed = 0
    for batch_index in range(batches):
        inputs, targets = dataset.batch(
            "train",
            batch_size,
            generator,
            torch.device("cpu"),
        )
        digest.update(f"batch={batch_index}".encode("ascii"))
        digest.update(_tensor_content_sha256(inputs).encode("ascii"))
        digest.update(_tensor_content_sha256(targets).encode("ascii"))
        completed += 1
    return {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "batches": completed,
        "sha256": digest.hexdigest(),
    }



def build_scale_guard(
    cfg: Config,
    args: argparse.Namespace,
    n_params: int,
    dataset: Optional[ByteDataset] = None,
) -> Dict[str, object]:
    """记录实验规模与原型级结论守卫。

    默认 32×128×2000 ≈ 8.19M 训练字节位置、约 6.9M 参数，
    只有约 1.2 token/param；这主要支持「原型可行性」，不足以
    外推到大规模 MoE、长上下文或推理任务。报告把规模写成
    tokens_seen / tokens_per_param / epochs，并在达到原型级
    阈值时给出 [scale] 警告，避免把玩具规模的结果说成普遍结论。
    """
    tokens_seen = (
        int(getattr(args, "steps", 0))
        * int(getattr(args, "batch_size", 0))
        * int(cfg.seq_len)
    )
    tokens_per_param = (
        tokens_seen / float(n_params)
        if n_params > 0
        else None
    )
    train_bytes = None
    epochs = None
    if dataset is not None:
        train_bytes = int(dataset.effective_train_bytes)
        if train_bytes > 0:
            epochs = tokens_seen / float(train_bytes)
    prototype = bool(
        (tokens_per_param is not None and tokens_per_param < 20.0)
        or (epochs is not None and epochs < 1.0)
        or cfg.seq_len < 512
        or cfg.n_experts <= 8
    )
    return {
        "tokens_seen": tokens_seen,
        "n_params": int(n_params),
        "tokens_per_param": tokens_per_param,
        "train_bytes": train_bytes,
        "epochs_over_train": epochs,
        "seq_len": int(cfg.seq_len),
        "n_experts": int(cfg.n_experts),
        "top_k": int(cfg.top_k),
        "scale_tier": "prototype" if prototype else "research",
        "note": (
            "tokens_seen=steps×batch_size×seq_len；prototype 级"
            "（<20 token/param、<1 epoch、seq_len<512 或专家数"
            "≤8）的结果只能支持原型可行性，不能外推到大规模 "
            "MoE、长上下文或推理任务。训练收敛应分别按 token、"
            "FLOPs 与墙钟时间报告"
        ),
    }
