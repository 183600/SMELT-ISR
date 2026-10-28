from __future__ import annotations



def build_environment_snapshot(
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    构造当前运行环境的规范化快照。

    ``environment_id`` 是决定数值结果的关键依赖/编译选项/设备/
    确定性开关的 SHA-256；完整包清单仍在 ``packages.all`` 中，
    供人工审计与重建环境。快照本身不含时间以外的不确定值，
    同一环境重复构造应得到同一 environment_id。
    """
    distributions = _installed_distributions()
    critical = [
        item
        for item in distributions
        if _is_critical_distribution(item["name"])
    ]
    cpu_model = _cpu_model_name()
    cpu_flags = _cpu_flags_digest()
    cpu = {
        "model": cpu_model,
        "count": os.cpu_count(),
        "flags_sha256": cpu_flags["flags_sha256"],
        "flag_count": cpu_flags["flag_count"],
        "flags_source": cpu_flags["source"],
    }
    snapshot: Dict[str, object] = {
        "schema": _ENVIRONMENT_SCHEMA,
        "captured_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "executable": sys.executable,
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "libc": list(platform.libc_ver()),
        },
        "cpu": cpu,
        "torch": _torch_environment(),
        "device": _environment_device_info(device),
        "driver": {
            "nvidia": _nvidia_driver_version(),
        },
        "determinism": _determinism_environment(),
        "packages": {
            "all": distributions,
            "critical": critical,
            "all_sha256": _canonical_json_sha256(distributions),
            "critical_sha256": _canonical_json_sha256(critical),
        },
    }
    snapshot["environment_id"] = _canonical_json_sha256(
        _environment_identity_payload(snapshot)
    )
    return snapshot



def _environment_package_map(
    snapshot: Dict[str, object],
) -> Dict[str, str]:
    packages = snapshot.get("packages")
    if isinstance(packages, dict):
        raw = packages.get("all")
    else:
        raw = packages
    if not isinstance(raw, list):
        return {}
    result: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _canonical_distribution_name(item.get("name"))
        version = item.get("version")
        if name is not None and isinstance(version, str):
            result[name] = version
    return result



def _nested_differences(
    expected: object,
    actual: object,
    prefix: str = "",
) -> List[Dict[str, object]]:
    """递归列出两个简单 JSON 结构的差异（用于环境检查报告）。"""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: List[Dict[str, object]] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(
                _nested_differences(
                    expected.get(key), actual.get(key), path
                )
            )
        return differences
    if expected != actual:
        return [{
            "path": prefix or "<root>",
            "expected": expected,
            "actual": actual,
        }]
    return []



def _environment_identifier(
    snapshot: Dict[str, object],
    identity: Dict[str, object],
) -> Tuple[str, bool]:
    """返回快照声明的 environment_id 及其与内容是否自洽。

    归档文件可能被截断或手工编辑：如果只比较声明的 id，修改
    关键包内容却保留旧 id 时会被误判为「同一环境」。规范化
    快照（带 schema）必须满足 id == SHA256(identity)；缺少 id
    的旧/手工快照按内容重新计算，不虚报一致。没有 schema 的
    最小字典保持旧行为，只用于比较调用方显式给出的 id。
    """
    canonical = _canonical_json_sha256(identity)
    recorded = snapshot.get("environment_id")
    if not isinstance(recorded, str) or not recorded:
        return canonical, True
    if snapshot.get("schema") != _ENVIRONMENT_SCHEMA:
        return recorded, True
    return recorded, recorded == canonical



def compare_environment_snapshots(
    expected: Dict[str, object],
    actual: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    比较归档环境与当前环境，返回可机读差异。

    ``ok`` 要求 environment_id 一致，且归档/当前快照声明的 id
    与各自内容自洽（决定数值结果的关键依赖/编译选项/设备/
    确定性开关）；``strict_ok`` 还要求完整包清单逐项一致。
    CLI 默认用 ``ok``，``--check-environment-strict`` 用
    ``strict_ok``。
    """
    actual = (
        actual
        if isinstance(actual, dict)
        else build_environment_snapshot()
    )
    expected = expected if isinstance(expected, dict) else {}

    expected_identity = _environment_identity_payload(expected)
    actual_identity = _environment_identity_payload(actual)
    expected_id, expected_id_consistent = _environment_identifier(
        expected, expected_identity
    )
    actual_id, actual_id_consistent = _environment_identifier(
        actual, actual_identity
    )

    expected_packages = _environment_package_map(expected)
    actual_packages = _environment_package_map(actual)
    added = [
        {"name": name, "version": actual_packages[name]}
        for name in sorted(set(actual_packages) - set(expected_packages))
    ]
    removed = [
        {"name": name, "version": expected_packages[name]}
        for name in sorted(set(expected_packages) - set(actual_packages))
    ]
    changed = [
        {
            "name": name,
            "expected": expected_packages[name],
            "actual": actual_packages[name],
        }
        for name in sorted(
            set(expected_packages) & set(actual_packages)
        )
        if expected_packages[name] != actual_packages[name]
    ]

    critical_names = set()
    for snapshot in (expected, actual):
        packages = snapshot.get("packages")
        if isinstance(packages, dict):
            raw = packages.get("critical")
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        name = _canonical_distribution_name(
                            item.get("name")
                        )
                        if name is not None:
                            critical_names.add(name)
    critical_differences = [
        item
        for item in added + removed + changed
        if item["name"] in critical_names
        or _is_critical_distribution(item["name"])
    ]

    runtime_differences = _nested_differences(
        expected_identity, actual_identity
    )
    same_id = expected_id == actual_id
    identity_consistent = (
        expected_id_consistent and actual_id_consistent
    )
    all_packages_match = not (added or removed or changed)
    return {
        "expected_environment_id": expected_id,
        "actual_environment_id": actual_id,
        "environment_id_matches": same_id,
        "expected_environment_id_consistent": (
            expected_id_consistent
        ),
        "actual_environment_id_consistent": (
            actual_id_consistent
        ),
        "package_differences": {
            "added": added,
            "removed": removed,
            "changed": changed,
        },
        "critical_package_differences": critical_differences,
        "runtime_differences": runtime_differences,
        "all_packages_match": all_packages_match,
        "ok": same_id and identity_consistent,
        "strict_ok": (
            same_id
            and identity_consistent
            and all_packages_match
        ),
    }



def _environment_snapshot_from_config(
    payload: Dict[str, object],
    base_dir: Path,
) -> Optional[Dict[str, object]]:
    """从 config.json 的环境摘要/运行元数据中取快照。"""
    environment = payload.get("environment")
    if isinstance(environment, dict):
        # 先看 file 摘要：config.json 顶层的 environment 是
        # {schema, environment_id, file, file_sha256} 摘要，不是
        # 完整快照；若先按 schema 判断，会把摘要自己当成快照，
        # 从而跳过文件哈希校验。
        file_name = environment.get("file")
        if isinstance(file_name, str) and file_name:
            candidate = base_dir / file_name
            expected_sha = environment.get("file_sha256")
            try:
                raw = candidate.read_bytes()
            except OSError:
                raw = None
            if raw is not None:
                if (
                    isinstance(expected_sha, str)
                    and hashlib.sha256(raw).hexdigest()
                    != expected_sha
                ):
                    raise ValueError(
                        f"环境快照 {candidate} 的 SHA-256 与 "
                        "config.json 记录不一致（文件被替换或截断）"
                    )
                try:
                    loaded = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    loaded = None
                if (
                    isinstance(loaded, dict)
                    and loaded.get("schema") == _ENVIRONMENT_SCHEMA
                ):
                    # 摘要里的 id 必须指向文件里实际记录的环境；
                    # 否则同一份 config.json 可以带着旧 id、却让
                    # 核对读取另一个（哈希已同步更新的）快照。
                    summary_id = environment.get(
                        "environment_id"
                    )
                    loaded_id = loaded.get("environment_id")
                    if (
                        isinstance(summary_id, str)
                        and summary_id
                        and summary_id != loaded_id
                    ):
                        raise ValueError(
                            f"config.json 记录的环境 id 与 "
                            f"{candidate} 不一致"
                        )
                    return loaded
        elif environment.get("schema") == _ENVIRONMENT_SCHEMA:
            return environment

    for key in ("original_run_metadata", "run_metadata"):
        metadata = payload.get(key)
        if not isinstance(metadata, dict):
            continue
        snapshot = metadata.get("environment")
        if (
            isinstance(snapshot, dict)
            and snapshot.get("schema") == _ENVIRONMENT_SCHEMA
        ):
            return snapshot
    return None



def load_environment_snapshot(
    path: Path,
) -> Tuple[Dict[str, object], Path]:
    """
    读取 environment.json，或从运行目录的 config.json 提取快照。

    返回 ``(snapshot, source_path)``。文件损坏、摘要不一致或找不到
    快照时抛 ValueError，由 CLI 转成明确的错误信息。
    """
    path = Path(path)
    if path.is_dir():
        environment_path = path / _ENVIRONMENT_FILENAME
        if environment_path.is_file():
            path = environment_path
        else:
            config_path = path / "config.json"
            if not config_path.is_file():
                raise ValueError(
                    f"{path} 内既没有 {_ENVIRONMENT_FILENAME}，"
                    "也没有 config.json"
                )
            path = config_path

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"无法读取环境快照 {path}: {error}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"环境快照不是合法 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("环境快照顶层必须是 JSON 对象")

    if payload.get("schema") == _ENVIRONMENT_SCHEMA:
        return payload, path

    snapshot = _environment_snapshot_from_config(
        payload, path.parent
    )
    if snapshot is None:
        raise ValueError(
            f"{path} 不是 {_ENVIRONMENT_SCHEMA} 快照，"
            "且 config.json 中没有可提取的 environment 记录"
        )
    return snapshot, path.parent / _ENVIRONMENT_FILENAME



def _snapshot_device(snapshot: Dict[str, object]) -> torch.device:
    device = snapshot.get("device")
    device_type = "cpu"
    device_index: Optional[int] = None
    if isinstance(device, dict):
        raw_type = device.get("type")
        if raw_type in ("cuda", "mps", "cpu"):
            device_type = str(raw_type)
        # 归档的是 cuda:1 时，核对必须检查 cuda:1 的型号/算力，
        # 不能退化成默认 cuda:0：多卡机器上不同卡可能不是同一
        # 设备。index 只接受非负整数（bool 是 int 子类，需排除）。
        raw_index = device.get("index")
        if (
            isinstance(raw_index, int)
            and not isinstance(raw_index, bool)
            and raw_index >= 0
            and device_type != "cpu"
        ):
            device_index = int(raw_index)
    try:
        if device_index is None:
            return torch.device(device_type)
        return torch.device(device_type, device_index)
    except (RuntimeError, ValueError):
        try:
            return torch.device(device_type)
        except (RuntimeError, ValueError):
            return torch.device("cpu")



def _apply_environment_determinism(
    snapshot: Dict[str, object],
) -> Dict[str, object]:
    """把归档快照里的运行期确定性/线程开关应用到当前进程。

    这些开关由训练入口的 ``set_seed`` / ``configure_cuda_precision``
    / ``configure_cpu_threads`` 设置，不是安装环境自带的属性。
    如果 ``--check-environment`` 在全新进程里直接构造当前快照，
    PyTorch 的默认值（确定性算法关闭、cuDNN benchmark 开启）
    会让同一台机器、同一套依赖也必然 MISMATCH。核对前先按归档
    值恢复这些可控开关，再比较依赖/硬件是否真的变化。

    返回 ``{"applied": [...], "errors": {...}}``；无法应用的项
    保留错误信息，后续比较仍会如实报告差异。
    """
    applied: List[str] = []
    errors: Dict[str, str] = {}
    determinism = snapshot.get("determinism")
    if not isinstance(determinism, dict):
        return {"applied": applied, "errors": errors}

    def _bool(name: str) -> Optional[bool]:
        value = determinism.get(name)
        if isinstance(value, bool):
            return value
        return None

    deterministic = _bool("deterministic_algorithms")
    warn_only = _bool(
        "deterministic_algorithms_warn_only"
    )
    if deterministic is not None:
        try:
            torch.use_deterministic_algorithms(
                deterministic,
                warn_only=bool(warn_only),
            )
            applied.append("deterministic_algorithms")
            if warn_only is not None:
                applied.append(
                    "deterministic_algorithms_warn_only"
                )
        except (RuntimeError, ValueError) as error:
            errors["deterministic_algorithms"] = str(error)

    for name, backend in (
        ("cudnn_deterministic", torch.backends.cudnn),
        ("cudnn_benchmark", torch.backends.cudnn),
    ):
        value = _bool(name)
        if value is None:
            continue
        try:
            setattr(backend, name[len("cudnn_"):], value)
            applied.append(name)
        except (AttributeError, RuntimeError) as error:
            errors[name] = str(error)

    cuda_matmul = torch.backends.cuda.matmul
    for name in (
        "allow_tf32_matmul",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        value = _bool(name)
        if value is None:
            continue
        attr = (
            "allow_tf32"
            if name == "allow_tf32_matmul"
            else name
        )
        if not hasattr(cuda_matmul, attr):
            continue
        try:
            setattr(cuda_matmul, attr, value)
            applied.append(name)
        except (AttributeError, RuntimeError) as error:
            errors[name] = str(error)
    allow_tf32_cudnn = _bool("allow_tf32_cudnn")
    if allow_tf32_cudnn is not None:
        try:
            torch.backends.cudnn.allow_tf32 = allow_tf32_cudnn
            applied.append("allow_tf32_cudnn")
        except (AttributeError, RuntimeError) as error:
            errors["allow_tf32_cudnn"] = str(error)

    threads = determinism.get("torch_num_threads")
    if (
        isinstance(threads, int)
        and not isinstance(threads, bool)
        and threads > 0
    ):
        try:
            torch.set_num_threads(threads)
            applied.append("torch_num_threads")
        except (RuntimeError, ValueError) as error:
            errors["torch_num_threads"] = str(error)
    interop = determinism.get("torch_num_interop_threads")
    if (
        isinstance(interop, int)
        and not isinstance(interop, bool)
        and interop > 0
    ):
        try:
            if torch.get_num_interop_threads() != interop:
                torch.set_num_interop_threads(interop)
            applied.append("torch_num_interop_threads")
        except (RuntimeError, ValueError) as error:
            errors["torch_num_interop_threads"] = str(error)

    env_names = {
        "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG",
        "omp_num_threads": "OMP_NUM_THREADS",
        "mkl_num_threads": "MKL_NUM_THREADS",
    }
    for name, env_name in env_names.items():
        if name not in determinism:
            continue
        value = determinism.get(name)
        if value is None:
            os.environ.pop(env_name, None)
            applied.append(name)
        elif isinstance(value, str):
            os.environ[env_name] = value
            applied.append(name)
    return {"applied": applied, "errors": errors}



def check_environment_snapshot(
    path: Path,
    *,
    strict: bool = False,
) -> Dict[str, object]:
    """
    核对当前环境是否与归档快照一致。

    这是重跑/续跑前的显式门：environment_id 不一致时，即使代码
    与数据相同，浮点归约顺序也可能不同，不能把结果当作逐位复现。
    """
    expected, source = load_environment_snapshot(Path(path))
    # 确定性/线程开关是运行期配置，不是安装环境；先按归档值恢复，
    # 否则全新进程的 PyTorch 默认值会让同一环境必然 MISMATCH。
    applied = _apply_environment_determinism(expected)
    actual = build_environment_snapshot(_snapshot_device(expected))
    report = compare_environment_snapshots(expected, actual)
    report.update({
        "source": str(source),
        "strict": bool(strict),
        "schema": expected.get("schema"),
        "applied_environment_settings": applied["applied"],
        "environment_setting_errors": applied["errors"],
        "ok": (
            report["strict_ok"] if strict else report["ok"]
        ),
    })
    return report



def print_environment_check(report: Dict[str, object]) -> None:
    """打印环境核对摘要；完整差异由调用方决定是否落盘。"""
    print(f"[env] source={report.get('source')}")
    print(
        "[env] expected environment_id="
        f"{report.get('expected_environment_id')}"
    )
    print(
        "[env] actual   environment_id="
        f"{report.get('actual_environment_id')}"
    )
    runtime = report.get("runtime_differences") or []
    if runtime:
        print(f"[env] runtime differences ({len(runtime)}):")
        for item in runtime[:12]:
            print(
                f"[env]   {item['path']}: "
                f"{item['expected']!r} -> {item['actual']!r}"
            )
        if len(runtime) > 12:
            print(f"[env]   ... {len(runtime) - 12} more")
    packages = report.get("package_differences") or {}
    added = packages.get("added") or []
    removed = packages.get("removed") or []
    changed = packages.get("changed") or []
    if added or removed or changed:
        print(
            "[env] package differences: "
            f"+{len(added)} -{len(removed)} ~{len(changed)}"
        )
        for item in changed[:8]:
            print(
                f"[env]   {item['name']}: "
                f"{item['expected']} -> {item['actual']}"
            )
    if report.get("ok"):
        if report.get("strict") and not report.get(
            "all_packages_match", True
        ):
            print(
                "[env] [warn] 关键环境一致，但完整包清单有差异"
                "（严格模式判失败）"
            )
        else:
            print("[env] OK: 当前环境与归档快照一致")
    else:
        print(
            "[env] MISMATCH: 当前环境与归档快照不一致；"
            "同一代码/数据下的数值结果不保证可复现。"
            "请按 environment.json 的 packages 重建环境，"
            "或在报告中明确声明环境差异。"
        )



def environment_mode_error(args) -> Optional[str]:
    """校验 --check-environment 的模式与冲突参数。"""
    check_path = getattr(args, "check_environment", None)
    strict = bool(
        getattr(args, "check_environment_strict", False)
    )
    if strict and check_path is None:
        return (
            "--check-environment-strict 必须与 "
            "--check-environment 一起使用"
        )
    if check_path is None:
        return None

    conflicts = [
        flag
        for flag, active in (
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
                "--compare",
                getattr(args, "compare", None) is not None,
            ),
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
                "--preregistration",
                getattr(args, "preregistration", None) is not None,
            ),
            ("--data", getattr(args, "data", None) is not None),
            ("--out", getattr(args, "out", None) is not None),
            ("--seeds", getattr(args, "seeds", None) is not None),
            (
                "--resume",
                getattr(args, "resume", None) is not None,
            ),
            (
                "--evaluate-test",
                bool(getattr(args, "evaluate_test", False)),
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
            (
                "--suite-baseline-tuning",
                bool(
                    getattr(args, "suite_baseline_tuning", False)
                ),
            ),
            (
                "--statistical-baseline",
                bool(getattr(args, "statistical_baseline", False)),
            ),
            (
                "--sanity-only",
                bool(getattr(args, "sanity_only", False)),
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
            "--check-environment 是只读环境核对入口，不能与"
            "以下训练/分析参数同时使用: " + ", ".join(conflicts)
        )
    return None



def write_run_environment_record(
    out_dir: Path,
    run_metadata: Dict[str, object],
    original_metadata: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """
    把本次（或最初的）环境快照写为 environment.json。

    resume 时优先归档最初运行的环境，而不是最后一次续跑的环境：
    否则同一份运行目录的环境记录会随续跑漂移，无法回答「原始
    结果是在哪套依赖下产生的」。当前环境若已变化，摘要里单独
    记录，不覆盖原始快照。
    """
    current = run_metadata.get("environment")
    current = current if isinstance(current, dict) else None
    archived = None
    if isinstance(original_metadata, dict):
        candidate = original_metadata.get("environment")
        if isinstance(candidate, dict):
            archived = candidate
    if archived is None:
        archived = current
    if archived is None:
        return None

    path = Path(out_dir) / _ENVIRONMENT_FILENAME
    _atomic_write_json(path, archived)
    record: Dict[str, object] = {
        "schema": archived.get("schema", _ENVIRONMENT_SCHEMA),
        "environment_id": archived.get("environment_id"),
        "file": _ENVIRONMENT_FILENAME,
        "file_sha256": _sha256_file(path),
    }
    current_id = (
        current.get("environment_id")
        if isinstance(current, dict)
        else None
    )
    if (
        isinstance(current_id, str)
        and current_id
        and current_id != record["environment_id"]
    ):
        record["resume_environment_id"] = current_id
        record["environment_changed_since_start"] = True
    return record



_ENVIRONMENT_END_SCHEMA = "smelt-isr/environment-end/v1"

_ENVIRONMENT_END_FILENAME = "environment_end.json"



def _metadata_environment_snapshot(
    metadata: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not isinstance(metadata, dict):
        return None
    snapshot = metadata.get("environment")
    return snapshot if isinstance(snapshot, dict) else None



def _snapshot_environment_id(
    snapshot: Optional[Dict[str, object]],
) -> Optional[str]:
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("environment_id")
    return value if isinstance(value, str) and value else None



def _environment_comparison_summary(
    expected: Optional[Dict[str, object]],
    current: Dict[str, object],
) -> Optional[Dict[str, object]]:
    if expected is None:
        return None
    comparison = compare_environment_snapshots(expected, current)
    return {
        "environment_id": _snapshot_environment_id(expected),
        "changed": not bool(comparison.get("ok")),
        "critical_package_differences": (
            comparison.get("critical_package_differences") or []
        ),
        "runtime_differences": (
            comparison.get("runtime_differences") or []
        ),
    }



def write_run_environment_end_record(
    out_dir: Path,
    start_metadata: Optional[Dict[str, object]] = None,
    original_snapshot: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """
    在运行正常结束时再次采集环境，并把结束快照与运行开始/
    最初运行的环境逐项比较。

    `environment.json` 只回答「原始结果在哪套环境下产生」，
    无法发现运行期间升级依赖、切换驱动或改动线程数；本记录
    把结束环境固定为文件，并在 config.json 摘要中保存哈希与
    漂移标志。无法采集时返回 None，由调用方决定是否提示。
    """
    start = _metadata_environment_snapshot(start_metadata)
    original = (
        original_snapshot
        if isinstance(original_snapshot, dict)
        else start
    )
    if start is None and original is None:
        return None
    current_source = start if start is not None else original
    if current_source is None:
        return None

    current = build_environment_snapshot(
        _snapshot_device(current_source)
    )
    path = Path(out_dir) / _ENVIRONMENT_END_FILENAME
    _atomic_write_json(path, current)

    start_summary = _environment_comparison_summary(start, current)
    original_summary = _environment_comparison_summary(
        original, current
    )
    record: Dict[str, object] = {
        "schema": _ENVIRONMENT_END_SCHEMA,
        "captured_utc": current.get("captured_utc"),
        "environment_id": _snapshot_environment_id(current),
        "file": _ENVIRONMENT_END_FILENAME,
        "file_sha256": _sha256_file(path),
        "start": start_summary,
        "original": original_summary,
        "changed_since_start": (
            None
            if start_summary is None
            else bool(start_summary.get("changed"))
        ),
        "changed_since_original": (
            None
            if original_summary is None
            else bool(original_summary.get("changed"))
        ),
        "note": (
            "结束环境快照；changed_* 为 true 表示运行结束时与"
            "启动/最初快照在关键依赖、运行时或确定性开关上不同"
        ),
    }
    return record



def collect_run_metadata(device: torch.device) -> Dict[str, object]:
    """
    收集本轮实验的代码版本与运行环境信息，写入 config.json，
    保证实验可追溯：代码版本 + 环境 + 超参数 + 种子 +
    数据指纹（args.data_sha256）+ 训练曲线（metrics.jsonl）。

    代码版本有三层记录：git commit（仓库可用时）、
    git_diff_sha256（工作区脏时对 git diff HEAD 的指纹，
    用于区分同一 commit 的不同未提交状态）与
    script_sha256（脚本文件本身的指纹，不依赖 git，
    覆盖压缩包运行与脏工作区两种 commit 无法定位代码的场景）。
    """
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")

    git_dirty = bool(status) if status is not None else None

    # 脏工作区的 diff 指纹：两个运行同 commit 但 diff 指纹不同，
    # 说明跑的不是同一份代码，对比结论不可信。
    git_diff_sha256: Optional[str] = None
    if git_dirty:
        diff = _git_output("diff", "HEAD")
        if diff is not None:
            git_diff_sha256 = _sha256_text(diff)

    device_name = None
    device_uuid = None
    cuda_capability = None
    cuda_total_memory_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(device)
            device_name = torch.cuda.get_device_name(device)
            # 物理 GPU UUID：仅记录 cuda:0 之类的逻辑编号无法区分
            # CUDA_VISIBLE_DEVICES 重排后的实际硬件。旧版 torch
            # 没有 uuid 属性时保持 None，不伪造设备身份。
            properties_uuid = getattr(properties, "uuid", None)
            if properties_uuid is not None:
                device_uuid = str(properties_uuid)
            cuda_capability = (
                f"{properties.major}.{properties.minor}"
            )
            cuda_total_memory_mb = round(
                float(properties.total_memory) / (1024 ** 2),
                1,
            )
        except (AssertionError, RuntimeError, ValueError):
            device_name = None

    # 完整环境快照既进入 config.json 的 run_metadata，也单独写成
    # environment.json（见 write_run_environment_record）。这里
    # 只构造一次，避免同一份记录里出现两个可能不一致的版本。
    environment = build_environment_snapshot(device)

    return {
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        # 逐参数 shell quote：含空格的路径（--data "my data.txt"）
        # 不能记录成无法复制重放的 "my data.txt"。
        "command": shlex.join(sys.argv),
        # 相对数据路径 / 相对 --out 只有结合工作目录才能还原；
        # 脚本路径与解释器则用于区分同名入口与不同虚拟环境。
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(
            torch.version, "git_version", None
        ),
        "numpy_version": (
            getattr(np, "__version__", None)
            if np is not None
            else None
        ),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_count": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
        "device_name": device_name,
        "device_index": device.index,
        "device_uuid": device_uuid,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "cuda_capability": cuda_capability,
        "cuda_total_memory_mb": cuda_total_memory_mb,
        "cpu_model": _cpu_model_name(),
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "script_sha256": _script_sha256(),
        # 环境身份：决定数值结果的关键依赖/编译选项/设备/确定性
        # 开关的规范化 SHA-256。同环境同代码的运行应一致；换
        # CUDA/cuBLAS/MKL/Triton/CPU 指令集或线程配置会改变它。
        "environment_id": environment["environment_id"],
        "environment_schema": environment["schema"],
        "python_implementation": environment["python"][
            "implementation"
        ],
        "python_compiler": environment["python"]["compiler"],
        "platform_machine": environment["platform"]["machine"],
        "cpu_flags_sha256": environment["cpu"]["flags_sha256"],
        "installed_distributions_sha256": environment["packages"][
            "all_sha256"
        ],
        "installed_distributions_count": len(
            environment["packages"]["all"]
        ),
        "environment": environment,
        # 进程启动时的实际值：None 说明本进程的 hash 随机化
        # 未被固定（set_seed 的 setdefault 只对之后的子进程
        # 生效），严格逐位复现应在启动命令行显式设置。
        "pythonhashseed_at_start": _PYTHONHASHSEED_AT_START,
        # 实际生效的随机源清单：seed 作用到 Python/NumPy/
        # torch/CUDA 及可选 XPU/MPS 的证据，而不只是参数记录。
        "rng_seed_manifest": (
            dict(_LAST_SEED_MANIFEST)
            if _LAST_SEED_MANIFEST is not None
            else None
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "pytorch_cuda_alloc_conf": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
        # fp16/bf16 的归约精度开关同样改变 GPU 数值结果；
        # 默认值随 torch 版本变化，记录实际生效值供跨运行核对。
        "allow_fp16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
            None,
        ),
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
            None,
        ),
        # CPU 并行归约顺序可能随线程数变化；即使未显式固定，
        # 也要记录当时实际生效的线程环境，供复现/对比追溯。
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }



def _json_safe(value):
    """
    把记录中的非有限浮点值（NaN/Inf）递归转成 None。

    json.dumps 默认会把 NaN/Inf 写成非标准 JSON 字面量
    （NaN / Infinity），其他语言的解析器会拒绝；统一转成 null
    以保留该字段但明确标记为「非有限」，避免污染整份日志。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # JSON 没有集合类型；手工/旧计划或自定义记录可能混入
        # set。按 repr 排序转成列表，保证同一集合跨进程得到
        # 相同顺序，而不是让 json.dumps 抛 TypeError。
        return [
            _json_safe(item)
            for item in sorted(value, key=repr)
        ]
    return value



# ============================================================
# Artifact manifest / archive verification
# ============================================================

# 归档清单：把「运行目录里到底有哪些产物、各自是什么内容」
# 变成可机读、可复核的记录。docs/experiment_protocol.md 要求
# 归档时给出 SHA-256，但此前只能靠人工 sha256sum，容易漏文件、
# 也无法区分「清单外的文件」与「清单内被改动」。该清单由
# --write-manifest 生成，--verify-manifest 逐项复核。
_ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"

_ARTIFACT_MANIFEST_SCHEMA = "smelt-isr/artifact-manifest/v1"

# 防御性上限：清单不应无界膨胀（误把 / 或 home 目录当运行目录
# 传入时尽早失败，而不是递归整块磁盘）。
_ARTIFACT_MANIFEST_MAX_ENTRIES = 200_000

_ARTIFACT_MANIFEST_MAX_DEPTH = 64



def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大检查点整块读入内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()



def _manifest_relative_path(value: object) -> Optional[str]:
    """
    校验并规范化清单内的相对路径。

    只接受规范化的 POSIX 相对路径：拒绝绝对路径、``..``、
    空组件、反斜杠（同一字符串在不同平台可能指向不同文件）
    与非规范写法（``a//b``、``a/./b``）。这些限制保证清单
    不能把校验引到运行目录之外。
    """
    if not isinstance(value, str) or not value:
        return None
    if "\x00" in value or "\\" in value:
        return None
    # Windows 盘符/UNC 前缀在 POSIX 的 PurePosixPath 下不是
    # 绝对路径，但换到 Windows 上 root / "C:/x" 会直接丢弃
    # root，把校验引到运行目录之外。清单是跨平台归档格式，
    # 必须在解析阶段就拒绝这类前缀。
    if value.startswith("//") or (
        len(value) >= 2
        and value[1] == ":"
        and value[0].isalpha()
    ):
        return None
    path = PurePosixPath(value)
    if path.is_absolute():
        return None
    if not path.parts:
        return None
    if any(part in ("", ".", "..") for part in path.parts):
        return None
    normalized = path.as_posix()
    if normalized != value:
        return None
    return normalized



def _manifest_file_entry(path: Path, relative: str) -> Dict[str, object]:
    """
    生成单个目录项的清单记录。

    符号链接只记录链接目标，不跟随链接哈希目标文件：跟随
    会让清单依赖运行目录之外的内容，也可能形成链接环。
    """
    if path.is_symlink():
        return {
            "path": relative,
            "type": "symlink",
            "target": os.readlink(path),
        }
    stat = path.stat()
    return {
        "path": relative,
        "type": "file",
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }



def _iter_manifest_entries(
    root: Path,
    current: Optional[Path] = None,
    depth: int = 0,
    *,
    with_digest: bool = True,
    _budget: Optional[List[int]] = None,
) -> List[Dict[str, object]]:
    """
    按路径排序遍历 root 下的文件与符号链接。

    ``with_digest=False`` 只收集路径与类型，不读取文件内容：
    校验时的「清单外新增文件」扫描不需要把已校验过的大检查点
    再哈希一遍。

    ``_budget`` 是跨递归共享的条目计数，保证上限约束的是整个
    目录树而不只是单个子目录。
    """
    if depth > _ARTIFACT_MANIFEST_MAX_DEPTH:
        raise ValueError(
            "运行目录层级过深（超过 "
            f"{_ARTIFACT_MANIFEST_MAX_DEPTH} 层）："
            "请确认 --write-manifest 指向的是运行目录而不是"
            "整个文件系统"
        )

    if _budget is None:
        _budget = [0]

    def consume() -> None:
        _budget[0] += 1
        if _budget[0] > _ARTIFACT_MANIFEST_MAX_ENTRIES:
            raise ValueError(
                "运行目录文件数超过清单上限 "
                f"{_ARTIFACT_MANIFEST_MAX_ENTRIES}："
                "请缩小 --write-manifest 的范围"
            )

    directory = current if current is not None else root
    with os.scandir(directory) as iterator:
        children = sorted(iterator, key=lambda item: item.name)

    entries: List[Dict[str, object]] = []
    for child in children:
        child_path = Path(child.path)
        relative = child_path.relative_to(root).as_posix()
        # 清单自身不进入清单：否则写入后哈希立刻失效。
        if current is None and child.name == _ARTIFACT_MANIFEST_NAME:
            continue
        # 写入端与校验端必须使用同一套路径规则：Linux 允许
        # 反斜杠/盘符前缀文件名，但这类清单换到 Windows 会
        # 被解析到目录外。宁可让 --write-manifest 明确失败，
        # 也不能生成一份自己都通不过校验的清单。
        if _manifest_relative_path(relative) is None:
            raise ValueError(
                "文件名不能安全写入跨平台归档清单: "
                f"{relative!r}"
            )
        if child.is_symlink():
            consume()
            entries.append(_manifest_file_entry(
                child_path, relative
            ))
        elif child.is_dir(follow_symlinks=False):
            entries.extend(_iter_manifest_entries(
                root,
                child_path,
                depth + 1,
                with_digest=with_digest,
                _budget=_budget,
            ))
        elif child.is_file(follow_symlinks=False):
            consume()
            if with_digest:
                entries.append(_manifest_file_entry(
                    child_path, relative
                ))
            else:
                entries.append({
                    "path": relative,
                    "type": "file",
                })
        else:
            # FIFO/套接字/设备文件不能按内容归档，只登记类型，
            # 让校验能发现它们被替换成普通文件。
            consume()
            entries.append({
                "path": relative,
                "type": "other",
            })
    return entries
