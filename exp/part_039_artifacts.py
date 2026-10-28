from __future__ import annotations



def _manifest_run_summary(
    root: Path,
) -> Optional[Dict[str, object]]:
    """
    从 config.json 提取归档时最常用的身份字段。

    这里只做摘要，真正的一致性由 config.json 自身的 SHA-256
    保证；摘要缺失（目录里没有 config.json）不拒绝生成清单，
    因为该命令也可用于归档不含训练配置的分析/图表目录。
    """
    config_path = root / "config.json"
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None

    # 续跑会重写 run_metadata；归档摘要必须指向最初运行的
    # 时间/代码身份（original_run_metadata），而不是最后一次
    # 续跑的时间。与 load_tracking_provenance 保持同一口径。
    metadata = config.get("original_run_metadata")
    if not isinstance(metadata, dict):
        metadata = config.get("run_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    data = config.get("data")
    if not isinstance(data, dict):
        data = {}
    preregistration = config.get("preregistration")
    if not isinstance(preregistration, dict):
        preregistration = {}
    environment = config.get("environment")
    environment = (
        environment if isinstance(environment, dict) else {}
    )
    metadata_environment = metadata.get("environment")
    metadata_environment = (
        metadata_environment
        if isinstance(metadata_environment, dict)
        else {}
    )

    return {
        "study_id": preregistration.get("study_id"),
        "hypothesis": config.get("hypothesis"),
        "research_question": config.get("research_question"),
        "run_started_utc": metadata.get("timestamp_utc"),
        "script_sha256": metadata.get("script_sha256"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        # 环境身份与 environment.json 自身摘要：归档摘要不用
        # 重新解析整份 config 就能核对「结果产自哪套环境」。
        "environment_id": (
            environment.get("environment_id")
            or metadata_environment.get("environment_id")
        ),
        "environment_schema": (
            environment.get("schema")
            or metadata_environment.get("schema")
        ),
        "environment_file_sha256": environment.get(
            "file_sha256"
        ),
        "environment_changed_since_start": environment.get(
            "environment_changed_since_start"
        ),
        "data_sha256": data.get("sha256"),
        "data_split_sha256": data.get("split_sha256"),
        "comparison_protocol_sha256": config.get(
            "comparison_protocol_sha256"
        ),
        "preregistration_file_sha256": preregistration.get(
            "file_sha256"
        ),
        "preregistration_frozen_commit": preregistration.get(
            "frozen_commit"
        ),
    }



def _manifest_current_provenance() -> Dict[str, object]:
    """记录生成清单时的代码与解释器版本，供事后核对。"""
    status = _git_output("status", "--porcelain")
    environment = build_environment_snapshot(
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    environment_id = environment["environment_id"]
    return {
        "script_sha256": _script_sha256(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": (
            bool(status) if status is not None else None
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": (
            getattr(np, "__version__", None)
            if np is not None
            else None
        ),
        "platform": platform.platform(),
        # 清单生成环境的完整身份；在另一台机器上校验归档时
        # 不一致是预期情况，只作为信息字段，不影响完整性判定。
        "environment_id": environment_id,
        "installed_distributions_sha256": (
            environment["packages"]["all_sha256"]
        ),
        "cwd": os.getcwd(),
        # 逐参数 shell quote：含空格的路径（--data "my data.txt"）
        # 不能记录成无法复制重放的 "my data.txt"。
        "command": shlex.join(sys.argv),
    }



def build_artifact_manifest(root: Path) -> Dict[str, object]:
    """构造运行目录的内容清单（不写盘）。"""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"运行目录不存在或不是目录: {root}")
    files = _iter_manifest_entries(root)
    return {
        "schema": _ARTIFACT_MANIFEST_SCHEMA,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        # 只记录目录名而不是绝对路径：归档解压到别处后清单
        # 仍然可用，也不会泄露生成机器的目录结构。
        "root_name": root.name or str(root),
        "provenance": _manifest_current_provenance(),
        "run": _manifest_run_summary(root),
        "files": files,
    }



def write_artifact_manifest(root: Path) -> Path:
    """
    在运行目录写入 ``artifact_manifest.json``（原子写入）。

    清单只描述写入时刻的目录状态；训练/评估仍在追加
    metrics.jsonl 或更新检查点时必须等所有进程结束后重新
    生成，不能把旧清单当作最终归档凭证。
    """
    root = Path(root)
    payload = build_artifact_manifest(root)
    path = root / _ARTIFACT_MANIFEST_NAME

    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(root),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    return path



def verify_artifact_manifest(
    root: Path,
    *,
    strict: bool = False,
) -> Dict[str, object]:
    """
    逐项复核 ``artifact_manifest.json``。

    返回结构化报告而不是只打印：``ok`` 表示完整性通过；
    ``strict=True`` 时清单外的新增文件也算失败（归档必须与
    清单逐字一致）。代码/环境与当前进程是否一致单独作为
    信息字段，不参与完整性判定：在另一台机器上校验归档时
    环境不同是预期情况，但完整性仍应通过。
    """
    root = Path(root)
    path = root / _ARTIFACT_MANIFEST_NAME
    result: Dict[str, object] = {
        "manifest": str(path),
        "manifest_sha256": None,
        "schema": None,
        "strict": bool(strict),
        "checked": 0,
        "missing": [],
        "mismatched": [],
        "untracked": [],
        "errors": [],
        "integrity_ok": False,
        "code_matches_current": None,
        "environment_matches_current": None,
        "ok": False,
    }

    try:
        manifest_bytes = path.read_bytes()
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        result["errors"].append(f"无法读取清单: {error}")
        return result
    result["manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    if not isinstance(payload, dict):
        result["errors"].append("清单顶层必须是 JSON 对象")
        return result

    schema = payload.get("schema")
    result["schema"] = schema
    if schema != _ARTIFACT_MANIFEST_SCHEMA:
        result["errors"].append(
            "清单 schema 不匹配："
            f"{schema!r}（预期 {_ARTIFACT_MANIFEST_SCHEMA}）"
        )

    files = payload.get("files")
    if not isinstance(files, list):
        result["errors"].append("清单缺少 files 数组")
        return result
    if len(files) > _ARTIFACT_MANIFEST_MAX_ENTRIES:
        result["errors"].append(
            "清单条目数超过上限 "
            f"{_ARTIFACT_MANIFEST_MAX_ENTRIES}"
        )
        return result

    root_resolved = root.resolve()
    seen = set()
    missing: List[str] = []
    mismatched: List[Dict[str, object]] = []

    def fail(relative, reason, **extra):
        entry = {"path": relative, "reason": reason}
        entry.update(extra)
        mismatched.append(entry)

    for item in files:
        if not isinstance(item, dict):
            fail("<invalid>", "entry_not_object")
            continue
        relative = _manifest_relative_path(item.get("path"))
        if relative is None:
            fail(str(item.get("path")), "unsafe_path")
            continue
        if relative in seen:
            fail(relative, "duplicate_path")
            continue
        seen.add(relative)

        target = root / relative
        entry_type = item.get("type", "file")
        # 防御目录穿越：条目所在目录解析后必须仍位于 root 内。
        # 对符号链接条目只能检查其父目录，不能再解析链接自身：
        # 清单明确记录符号链接目标而不跟随（见
        # _manifest_file_entry），合法的归档链接可以指向运行
        # 目录之外；跟随解析会把这种合法清单误报为逃逸。
        check_path = (
            target.parent if entry_type == "symlink" else target
        )
        try:
            check_path.resolve(strict=False).relative_to(
                root_resolved
            )
        except (OSError, ValueError):
            fail(relative, "escapes_root")
            continue
        except RuntimeError as error:
            # Path.resolve 遇到符号链接环会抛 RuntimeError（不是
            # OSError）；恶意/损坏清单可用 a/x 这类路径触发它。
            # 校验必须把它报告为路径解析失败，而不是让整个
            # --verify-manifest 以 traceback 退出。
            fail(
                relative,
                "path_resolution_error",
                error=str(error),
            )
            continue
        try:
            exists = target.exists()
            is_symlink = target.is_symlink()
        except OSError as error:
            fail(relative, "stat_error", error=str(error))
            continue
        if not exists and not is_symlink:
            missing.append(relative)
            continue

        if entry_type == "symlink":
            if not target.is_symlink():
                fail(relative, "expected_symlink")
            elif os.readlink(target) != item.get("target"):
                fail(
                    relative,
                    "symlink_target",
                    expected=item.get("target"),
                    actual=os.readlink(target),
                )
        elif entry_type == "file":
            if target.is_symlink() or not target.is_file():
                fail(relative, "expected_regular_file")
                continue
            try:
                actual_size = target.stat().st_size
            except OSError as error:
                fail(relative, "stat_error", error=str(error))
                continue
            if actual_size != item.get("bytes"):
                fail(
                    relative,
                    "size",
                    expected=item.get("bytes"),
                    actual=int(actual_size),
                )
                continue
            try:
                actual_sha256 = _sha256_file(target)
            except OSError as error:
                fail(relative, "read_error", error=str(error))
                continue
            if actual_sha256 != item.get("sha256"):
                fail(
                    relative,
                    "sha256",
                    expected=item.get("sha256"),
                    actual=actual_sha256,
                )
        elif entry_type == "other":
            if (
                target.is_file()
                or target.is_dir()
                or target.is_symlink()
            ):
                fail(relative, "expected_special_file")
        else:
            fail(relative, "unknown_type", value=entry_type)

    try:
        actual_entries = _iter_manifest_entries(
            root,
            with_digest=False,
        )
    except (OSError, ValueError) as error:
        result["errors"].append(
            f"无法遍历运行目录: {error}"
        )
        actual_entries = []
    actual_paths = {
        str(entry.get("path"))
        for entry in actual_entries
        if isinstance(entry.get("path"), str)
    }
    untracked = sorted(actual_paths - seen)

    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        result["code_matches_current"] = (
            provenance.get("script_sha256")
            == _script_sha256()
        )
        current_snapshot = build_environment_snapshot(
            torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        current_environment = {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": (
                getattr(np, "__version__", None)
                if np is not None
                else None
            ),
            "platform": platform.platform(),
            "environment_id": current_snapshot["environment_id"],
            "installed_distributions_sha256": current_snapshot[
                "packages"
            ]["all_sha256"],
        }
        matches = {
            key: provenance.get(key) == value
            for key, value in current_environment.items()
        }
        # 旧清单没有 environment_id 时保持 None（未核对），不能
        # 因为默认值/缺失就把「无法核对」报成「环境一致」。
        if provenance.get("environment_id") is None:
            matches["environment_id"] = None
        result["environment_matches_current"] = matches
        result["environment_id_matches_current"] = matches.get(
            "environment_id"
        )

    result["checked"] = len(seen)
    result["missing"] = missing
    result["mismatched"] = mismatched
    result["untracked"] = untracked
    result["integrity_ok"] = not (
        result["errors"] or missing or mismatched
    )
    result["ok"] = bool(result["integrity_ok"]) and not (
        strict and untracked
    )
    return result



def manifest_mode_error(args) -> Optional[str]:
    """
    校验 --write-manifest / --verify-manifest 的互斥与搭配。

    清单模式是只读的归档工具，不需要 --data/--out/种子，也不能
    与训练、预注册或对比入口混用：混用时用户很容易以为「生成了
    清单」而实际又启动了一次训练，或反过来把训练参数静默忽略。
    """
    write_mode = getattr(args, "write_manifest", None)
    verify_mode = getattr(args, "verify_manifest", None)
    if write_mode is not None and verify_mode is not None:
        return (
            "--write-manifest 与 --verify-manifest 不能同时使用："
            "先写清单，再单独调用校验"
        )
    # strict 只对校验模式有意义；与 --write-manifest 同用时也
    # 必须拒绝，不能因为「不是两个模式都缺」就静默忽略。
    if (
        getattr(args, "verify_manifest_strict", False)
        and verify_mode is None
    ):
        return (
            "--verify-manifest-strict 必须与 "
            "--verify-manifest 一起使用"
        )
    expected_manifest_sha = getattr(
        args, "expect_manifest_sha256", None
    )
    if expected_manifest_sha is not None:
        if verify_mode is None:
            return (
                "--expect-manifest-sha256 必须与 "
                "--verify-manifest 一起使用"
            )
        if not re.fullmatch(
            r"[0-9a-fA-F]{64}", str(expected_manifest_sha)
        ):
            return (
                "--expect-manifest-sha256 必须是 64 位"
                "十六进制 SHA-256"
            )
    if write_mode is None and verify_mode is None:
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
            "清单模式（--write-manifest/--verify-manifest）是"
            "只读归档工具，不能与以下训练/分析参数同时使用: "
            + ", ".join(conflicts)
        )
    return None



# ============================================================
# 运行级复现审计
# ============================================================

# 归档清单回答「目录里的文件有没有被改动」，环境核对回答
# 「当前机器是不是产出结果的那套环境」，但两者都不回答
# 「同一运行目录里的 config / environment / metrics / 预注册
# 记录是否互相一致」。只做逐文件哈希时，一份被整体替换、
# 内部却自洽的 config.json 仍可能通过清单校验；metrics.jsonl
# 也可能来自另一个种子或另一份代码。运行级审计把这些跨文件
# 链接逐项核对，并给出可机读的通过/警告/失败结论。
_RUN_AUDIT_SCHEMA = "smelt-isr/run-audit/v1"

_RUN_AUDIT_REPORT_FILENAME = "reproducibility_audit.json"



def _audit_digest(value: object) -> bool:
    """是否为 64 位十六进制 SHA-256（大小写不敏感）。"""
    return isinstance(value, str) and re.fullmatch(
        r"[0-9a-fA-F]{64}", value
    ) is not None



def _audit_same(left: object, right: object) -> bool:
    """按稳定表示比较 JSON 值，避免 dict 顺序造成假差异。"""
    return _stable_factor_key(left) == _stable_factor_key(right)



def _audit_metrics_log(path: Path) -> Dict[str, object]:
    """
    逐行解析 metrics.jsonl，保留结构完整性证据。

    read_metrics_records 为兼容旧日志会跳过坏行；审计不能静默
    跳过，否则截断/拼接的日志会看起来仍然「可解析」。这里同时
    检查坏行、非法 UTF-8、空行、末尾半行、重复训练步与回退步。
    """
    result: Dict[str, object] = {
        "available": False,
        "records": [],
        "record_lines": [],
        "lines": 0,
        "malformed_lines": [],
        "invalid_utf8_lines": [],
        "blank_lines": [],
        "trailing_partial": False,
        "duplicate_steps": [],
        "nonmonotonic_steps": [],
    }
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        result["error"] = str(error)
        return result
    result["available"] = True

    records: List[Dict[str, object]] = []
    record_lines: List[int] = []
    parsed: List[tuple[int, Dict[str, object]]] = []
    last_line_had_error = False
    for line_number, raw_line in enumerate(
        raw.splitlines(keepends=True), start=1
    ):
        last_line_had_error = False
        if not raw_line.strip():
            result["blank_lines"].append(line_number)  # type: ignore[union-attr]
            continue
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            result["invalid_utf8_lines"].append(line_number)  # type: ignore[union-attr]
            last_line_had_error = True
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            result["malformed_lines"].append(line_number)  # type: ignore[union-attr]
            last_line_had_error = True
            continue
        if not isinstance(loaded, dict):
            result["malformed_lines"].append(line_number)  # type: ignore[union-attr]
            last_line_had_error = True
            continue
        records.append(loaded)
        record_lines.append(line_number)
        parsed.append((line_number, loaded))

    # 只把「末尾无换行且该行无法解析」视为半行。完整的 JSON
    # 对象即使缺少末尾换行也已被完整写入（read_metrics_records
    # 同样接受），不能据此宣称最后一条记录不可信。
    if (
        raw
        and not raw.endswith((b"\n", b"\r"))
        and last_line_had_error
    ):
        result["trailing_partial"] = True
    result["lines"] = len(raw.splitlines())
    result["records"] = records
    result["record_lines"] = record_lines

    seen_steps: Dict[int, int] = {}
    previous_step: Optional[int] = None
    duplicate_steps: List[Dict[str, int]] = []
    nonmonotonic_steps: List[Dict[str, object]] = []
    for line_number, record in parsed:
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            continue
        if step in seen_steps:
            duplicate_steps.append(
                {
                    "step": step,
                    "first_line": seen_steps[step],
                    "duplicate_line": line_number,
                }
            )
        else:
            seen_steps[step] = line_number
        if previous_step is not None and step <= previous_step:
            nonmonotonic_steps.append(
                {
                    "previous_step": previous_step,
                    "step": step,
                    "line": line_number,
                }
            )
        previous_step = step
    result["duplicate_steps"] = duplicate_steps
    result["nonmonotonic_steps"] = nonmonotonic_steps
    return result
