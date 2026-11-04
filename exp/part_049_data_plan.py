from __future__ import annotations


_DATA_PLAN_SCHEMA_VERSION = 1

_DATA_PLAN_FILENAME = "data_plan.json"


def data_plan_cleaning_block(args: object) -> Dict[str, object]:
    """
    把本次实际生效的清洗口径整理成可冻结的对象。

    字段名与预注册 ``data.cleaning`` 完全一致：布尔开关逐项
    列出（未启用也写 false，不能靠漏写绕过核对），近重复阈值
    与 shingle 长度按实际取值写出。两个「允许用评估内容参与
    清洗」的同义开关写成同一个有效值，否则预注册 schema 会
    因同义开关取值相反而拒绝文件。
    """
    cleaning: Dict[str, object] = {}
    effective = _clean_against_test_effective(args)
    for key in _PREREGISTRATION_CLEANING_BOOL_KEYS:
        if key in _CLEAN_AGAINST_TEST_ALIASES:
            cleaning[key] = bool(effective)
            continue
        cleaning[key] = bool(getattr(args, key, False))
    for key in _PREREGISTRATION_CLEANING_NUMERIC_KEYS:
        value = getattr(args, key, None)
        if key == "near_duplicate_threshold" and value is not None:
            cleaning[key] = float(value)
        elif key == "near_duplicate_shingle_bytes" and value is not None:
            cleaning[key] = int(value)
        else:
            cleaning[key] = value
    return cleaning



def data_plan_preregistration_block(
    args: object,
    dataset: object,
) -> Dict[str, object]:
    """
    生成可直接粘进 ``preregistration.json`` 的 ``data`` 块。

    手写这一块最容易出的三类错是：语料指纹抄错、切分比例与
    群组分隔符与命令行不一致、清洗开关漏写（漏写按 false
    核对，会在训练入口被拒绝）。这里直接从真实语料与真实
    命令行口径生成，避免「文件里写的协议」和「实际跑的协议」
    两套。
    """
    block: Dict[str, object] = {
        "sha256": str(getattr(dataset, "data_sha256", "")),
        "split_policy": str(
            getattr(args, "split_policy", "contiguous")
        ),
        "val_fraction": float(getattr(dataset, "val_fraction", 0.0)),
        "test_fraction": float(
            getattr(dataset, "test_fraction", 0.0)
        ),
        "group_separator": getattr(args, "group_separator", None),
        "cleaning": data_plan_cleaning_block(args),
    }
    for key, attribute in (
        ("entity_manifest_sha256", "entity_manifest"),
        ("time_order_manifest_sha256", "time_order_manifest"),
    ):
        path = getattr(args, attribute, None)
        if not path:
            continue
        digest = _sha256_file(Path(path))
        if digest:
            block[key] = digest
    return block



def data_plan_time_order_evidence(
    args: object,
) -> Optional[Dict[str, object]]:
    """
    按当前 ``--split-policy`` 校验时间顺序清单。

    ``chronological`` 切分必须由清单证明文件顺序与时间戳单调
    一致，否则「前缀训练、后缀评估」不能解释为时间外推。数据
    计划阶段就做同样的校验，避免冻结完预注册才发现切分口径
    不成立。
    """
    path = getattr(args, "time_order_manifest", None)
    policy = str(getattr(args, "split_policy", "contiguous"))
    if policy == "chronological" and not path:
        raise ValueError(
            "--split-policy chronological 需要 --time-order-manifest："
            "仅凭文件顺序不能证明时间顺序"
        )
    if not path:
        return None
    evidence = read_time_order_manifest(path)
    if policy == "chronological" and not evidence.get(
        "monotonic_non_decreasing"
    ):
        raise ValueError(
            "--time-order-manifest 的时间戳不是单调不减：语料文件"
            "顺序与时间顺序不一致，不能按时间前缀切分"
        )
    return evidence



def build_data_plan(
    args: object,
    dataset: object,
    *,
    audit: Optional[Dict[str, object]] = None,
    time_order_evidence: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    构造「数据协议准备」产物 ``data_plan.json`` 的内容。

    产物回答的是「这次确证实验到底用哪份数据、按什么口径切
    分与清洗、预注册里应该写什么」，与训练后的
    ``artifact_manifest.json``（归档内容完整性）分工不同：
    前者是训练前的设计证据，后者是训练后的归档证据。
    """
    split_bounds = {
        str(split): [int(begin), int(end)]
        for split, (begin, end) in dict(
            getattr(dataset, "split_bounds", {}) or {}
        ).items()
    }
    split_fingerprints = {
        str(split): str(digest)
        for split, digest in dict(
            getattr(dataset, "split_sha256", {}) or {}
        ).items()
    }
    groups_per_split = getattr(dataset, "groups_per_split", None)
    core: Dict[str, object] = {
        "schema_version": _DATA_PLAN_SCHEMA_VERSION,
        "corpus": {
            "path": str(getattr(args, "data", "") or ""),
            "sha256": str(getattr(dataset, "data_sha256", "")),
            "bytes": int(getattr(dataset, "num_bytes", 0)),
        },
        "split": {
            "policy": str(
                getattr(args, "split_policy", "contiguous")
            ),
            "val_fraction": float(
                getattr(dataset, "val_fraction", 0.0)
            ),
            "test_fraction": float(
                getattr(dataset, "test_fraction", 0.0)
            ),
            "group_separator": getattr(
                args, "group_separator", None
            ),
            "bounds": split_bounds,
            "sha256": split_fingerprints,
            "effective_train_sha256": str(
                getattr(dataset, "effective_train_sha256", "") or ""
            ),
            "groups_per_split": (
                {
                    str(split): int(count)
                    for split, count in dict(groups_per_split).items()
                }
                if isinstance(groups_per_split, dict)
                else None
            ),
            "isolation": dict(
                getattr(dataset, "split_isolation", {}) or {}
            ),
        },
        "cleaning": data_plan_cleaning_block(args),
        "manifests": {
            "entity_manifest_sha256": _sha256_file(
                Path(args.entity_manifest)
            )
            if getattr(args, "entity_manifest", None)
            else None,
            "time_order_manifest_sha256": _sha256_file(
                Path(args.time_order_manifest)
            )
            if getattr(args, "time_order_manifest", None)
            else None,
        },
        "eval_content_access": dict(
            getattr(dataset, "eval_content_access", {}) or {}
        ),
        "preregistration_data_block": (
            data_plan_preregistration_block(args, dataset)
        ),
    }
    if time_order_evidence is not None:
        core["time_order_evidence"] = time_order_evidence
    if audit is not None:
        core["audit"] = audit
    plan_id = data_plan_id(core)
    payload = dict(core)
    payload["plan_id"] = plan_id
    payload["created_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    return payload



def data_plan_id(payload: Dict[str, object]) -> str:
    """
    数据计划内容的规范化指纹（排除 plan_id 与创建时间）。

    与设计计划的 ``plan_id`` 同一套规范化规则：写入与核对必须
    用同一函数，否则一个被改写、但 ``plan_id`` 字段没有同步更
    新的文件会仅凭字段值通过核对。``plan_id`` 是结果、
    ``created_utc`` 只是写入时间，都不参与指纹。
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



def write_data_plan(
    path: object,
    plan: Dict[str, object],
) -> Dict[str, object]:
    """
    原子写 ``data_plan.json``；同一路径已有另一份计划时拒绝覆盖。

    冻结后的数据计划不允许被「重新生成一份更好看的」替换：
    指纹不同的第二份计划必须换路径，否则预注册引用的数据协议
    与归档里的计划会对不上。同一份计划重复生成时保留已冻结的
    文件字节，不改写写入时间。
    """
    target = Path(path)
    if target.exists():
        try:
            existing_raw = target.read_bytes()
            existing = json.loads(existing_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"目标路径已有无法解析的 {target.name}：{error}；"
                "为避免覆盖冻结的数据计划，请先归档/移走该文件"
            ) from error
        if not isinstance(existing, dict):
            raise ValueError(
                f"目标路径已有 {target.name}，但不是 JSON 对象；"
                "为避免覆盖冻结的数据计划，请先归档/移走该文件"
            )
        existing_id = existing.get("plan_id")
        if data_plan_id(existing) != existing_id:
            raise ValueError(
                f"目标路径已有的 {target.name} 内容与其 plan_id "
                "不一致：文件可能被改写，拒绝覆盖"
            )
        if str(existing_id) != str(plan.get("plan_id")):
            raise ValueError(
                "目标路径已有另一份数据计划"
                f"（plan_id={str(existing_id)[:12]}...，"
                f"本次为 {str(plan.get('plan_id'))[:12]}...）："
                "数据协议不能在冻结后被替换，请使用新的路径"
                "（或先把旧计划归档/移走）"
            )
        return {
            "path": str(target),
            "file_sha256": hashlib.sha256(existing_raw).hexdigest(),
            "plan_id": str(existing_id),
            "written": False,
        }
    _atomic_write_json(target, plan)
    return {
        "path": str(target),
        "file_sha256": _sha256_file(target),
        "plan_id": str(plan.get("plan_id")),
        "written": True,
    }



def _data_plan_group_separator_bytes(value: object) -> Optional[bytes]:
    """把群组分隔符字符串解析成字节；None/缺省表示不使用群组切分。"""
    if value is None:
        return None
    return parse_group_separator(str(value))



def data_plan_differences(
    declared: object,
    expected: object,
) -> List[Dict[str, object]]:
    """
    逐项比较预注册的 ``data`` 块与实际数据口径。

    口径与训练入口的 ``preregistration_data_error`` 保持一致：
    缺省的清洗开关按 false 处理、缺省的 ``group_separator`` 按
    「不使用群组切分」处理、两个同义清洗开关按有效值比较。
    返回的差异列表是可执行的补丁说明，而不是一句笼统的
    「不一致」。
    """
    differences: List[Dict[str, object]] = []
    if not isinstance(declared, dict):
        return [
            {
                "field": "data",
                "declared": declared,
                "expected": expected,
                "reason": "预注册文件缺少 data 对象",
            }
        ]
    if not isinstance(expected, dict):
        return [
            {
                "field": "data",
                "declared": declared,
                "expected": expected,
                "reason": "数据计划缺少 preregistration_data_block",
            }
        ]

    declared_sha = str(declared.get("sha256", "")).lower()
    expected_sha = str(expected.get("sha256", "")).lower()
    if declared_sha != expected_sha:
        differences.append(
            {
                "field": "data.sha256",
                "declared": declared.get("sha256"),
                "expected": expected.get("sha256"),
                "reason": "语料指纹不一致：换语料必须重新预注册",
            }
        )

    if str(declared.get("split_policy")) != str(
        expected.get("split_policy")
    ):
        differences.append(
            {
                "field": "data.split_policy",
                "declared": declared.get("split_policy"),
                "expected": expected.get("split_policy"),
                "reason": "切分口径不一致：它决定前缀能否解释为时间外推",
            }
        )

    for key in ("val_fraction", "test_fraction"):
        try:
            declared_value = float(declared.get(key))
        except (TypeError, ValueError, OverflowError):
            declared_value = None
        try:
            expected_value = float(expected.get(key))
        except (TypeError, ValueError, OverflowError):
            expected_value = None
        same = (
            declared_value is not None
            and expected_value is not None
            and math.isclose(
                declared_value,
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if not same:
            differences.append(
                {
                    "field": f"data.{key}",
                    "declared": declared.get(key),
                    "expected": expected.get(key),
                    "reason": "切分比例不一致：会改变实际训练与评估字节",
                }
            )

    declared_group = declared.get("group_separator")
    expected_group = expected.get("group_separator")
    try:
        declared_bytes = _data_plan_group_separator_bytes(declared_group)
        expected_bytes = _data_plan_group_separator_bytes(expected_group)
    except ValueError as error:
        differences.append(
            {
                "field": "data.group_separator",
                "declared": declared_group,
                "expected": expected_group,
                "reason": f"群组分隔符无效: {error}",
            }
        )
    else:
        if declared_bytes != expected_bytes:
            differences.append(
                {
                    "field": "data.group_separator",
                    "declared": declared_group,
                    "expected": expected_group,
                    "reason": "群组分隔符不一致：群组边界会改变切分吸附位置与实际字节",
                }
            )

    declared_cleaning = declared.get("cleaning")
    if declared_cleaning is None:
        declared_cleaning = {}
    if not isinstance(declared_cleaning, dict):
        differences.append(
            {
                "field": "data.cleaning",
                "declared": declared.get("cleaning"),
                "expected": expected.get("cleaning"),
                "reason": "data.cleaning 必须是对象",
            }
        )
        declared_cleaning = {}
    expected_cleaning = expected.get("cleaning")
    if not isinstance(expected_cleaning, dict):
        expected_cleaning = {}

    declared_effective = any(
        bool(declared_cleaning.get(key, False))
        for key in _CLEAN_AGAINST_TEST_ALIASES
    )
    expected_effective = any(
        bool(expected_cleaning.get(key, False))
        for key in _CLEAN_AGAINST_TEST_ALIASES
    )
    if declared_effective != expected_effective:
        differences.append(
            {
                "field": "data.cleaning.clean_train_against_test",
                "declared": declared_effective,
                "expected": expected_effective,
                "reason": "清洗是否读取评估内容不一致：测试集将不再全程尘封",
            }
        )
    for key in _PREREGISTRATION_CLEANING_BOOL_KEYS:
        if key in _CLEAN_AGAINST_TEST_ALIASES:
            continue
        declared_value = bool(declared_cleaning.get(key, False))
        expected_value = bool(expected_cleaning.get(key, False))
        if declared_value != expected_value:
            differences.append(
                {
                    "field": f"data.cleaning.{key}",
                    "declared": declared_value,
                    "expected": expected_value,
                    "reason": "清洗开关不一致：会改变实际参与训练/去重的字节",
                }
            )
    numeric_defaults = {
        "near_duplicate_threshold": _NEAR_DUP_DEFAULT_THRESHOLD,
        "near_duplicate_shingle_bytes": (
            _NEAR_DUP_DEFAULT_SHINGLE_BYTES
        ),
    }
    for key, default in numeric_defaults.items():
        declared_value = declared_cleaning.get(key, default)
        expected_value = expected_cleaning.get(key, default)
        try:
            same = float(declared_value) == float(expected_value)
        except (TypeError, ValueError, OverflowError):
            same = False
        if not same:
            differences.append(
                {
                    "field": f"data.cleaning.{key}",
                    "declared": declared_value,
                    "expected": expected_value,
                    "reason": "近重复阈值/shingle 长度不一致：会改变删除哪些训练群组",
                }
            )

    for key in (
        "entity_manifest_sha256",
        "time_order_manifest_sha256",
    ):
        declared_value = declared.get(key)
        if declared_value is None:
            continue
        expected_value = expected.get(key)
        if str(declared_value).lower() != str(expected_value).lower():
            differences.append(
                {
                    "field": f"data.{key}",
                    "declared": declared_value,
                    "expected": expected_value,
                    "reason": "清单指纹不一致：实体/时间顺序清单会改变切分与隔离审计结果",
                }
            )
    return differences



def check_data_plan(
    preregistration_payload: Dict[str, object],
    plan: Dict[str, object],
    *,
    args: object = None,
    dataset: object = None,
) -> Dict[str, object]:
    """
    把预注册文件的 ``data`` 块与当前语料逐项核对。

    除了结构化差异，还调用训练入口真正使用的
    ``preregistration_data_error``：两者结论必须一致，否则说明
    预检口径与训练口径已经漂移，这种漂移本身必须暴露出来，
    不能只报其中一边。
    """
    declared = preregistration_payload.get("data")
    expected = plan.get("preregistration_data_block")
    differences = data_plan_differences(declared, expected)
    gate_error: Optional[str] = None
    if args is not None and dataset is not None:
        gate_error = preregistration_data_error(
            args, preregistration_payload, dataset
        )
    metadata = preregistration_payload.get("metadata")
    metadata_id: Optional[str] = None
    metadata_status = "not_declared"
    if isinstance(metadata, dict) and metadata.get("data_plan_id"):
        metadata_id = str(metadata.get("data_plan_id"))
        metadata_status = (
            "matched"
            if metadata_id == str(plan.get("plan_id"))
            else "mismatched"
        )
        if metadata_status == "mismatched":
            differences.append(
                {
                    "field": "metadata.data_plan_id",
                    "declared": metadata_id,
                    "expected": str(plan.get("plan_id")),
                    "reason": "预注册声明的数据计划指纹与当前数据计划不一致：冻结的数据协议被替换过",
                }
            )
    notes: List[str] = []
    if gate_error is not None and not differences:
        notes.append(
            "训练入口的数据核对同样拒绝启动，但结构化差异为空："
            "预检与训练口径可能已漂移，请以训练入口的结论为准"
        )
    if gate_error is None and differences:
        notes.append(
            "结构化比较发现差异，但训练入口的核对没有拒绝："
            "两套口径不一致，请以训练入口的结论为准并修正工具"
        )
    return {
        "ok": bool(not differences and gate_error is None),
        "differences": differences,
        "gate_error": gate_error,
        "declared": declared,
        "expected": expected,
        "plan_id": str(plan.get("plan_id")),
        "metadata_data_plan_id": metadata_id,
        "metadata_data_plan_id_status": metadata_status,
        "notes": notes,
    }



def print_data_plan_check(report: Dict[str, object]) -> None:
    """打印数据协议核对结论（逐项差异 + 应写入的 data 块）。"""
    differences = report.get("differences") or []
    if report.get("ok"):
        print(
            "[data-plan] 数据协议核对通过：预注册的 data 块与当前"
            "语料/清洗口径逐项一致"
        )
        metadata_status = report.get("metadata_data_plan_id_status")
        if metadata_status == "matched":
            print(
                "[data-plan] 预注册 metadata.data_plan_id 与当前数据"
                "计划指纹一致"
            )
        return
    print("[data-plan] 数据协议核对未通过：")
    for difference in differences:
        print(
            f"  - {difference.get('field')}：声明 "
            f"{difference.get('declared')!r}，实际 "
            f"{difference.get('expected')!r}"
            f"（{difference.get('reason')}）"
        )
    gate_error = report.get("gate_error")
    if gate_error:
        print(f"[data-plan] 训练入口同样会拒绝启动：{gate_error}")
    for note in report.get("notes") or []:
        print(f"[data-plan] [warn] {note}")
    print(
        "[data-plan] 应写入预注册的 data 块："
    )
    print(
        json.dumps(
            _json_safe(report.get("expected")),
            ensure_ascii=False,
            indent=2,
        )
    )
