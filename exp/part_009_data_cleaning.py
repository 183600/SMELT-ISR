from __future__ import annotations



def _plan_train_near_duplicate_cleaning(
    raw: bytearray,
    separator: bytes,
    train_end: int,
    eval_ranges: Dict[str, Tuple[int, int]],
    already_excluded: List[Tuple[int, int]],
    threshold: float = _NEAR_DUP_DEFAULT_THRESHOLD,
    shingle_bytes: int = _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
    sketch_size: int = _NEAR_DUP_DEFAULT_SKETCH_SIZE,
    apply_cleaning: bool = True,
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    扫描训练集群组的近重复（默认只对照 val，绝不改动 val/test）。

    流程：
    1. 对评估群组与训练群组分别计算字节 k-shingle 的 bottom-k 草图；
    2. 用草图哈希建倒排索引，为每个训练群组找候选；
    3. 对候选计算完整 shingle 集合的精确 Jaccard；
    4. 达到阈值时整组标记/删除（``apply_cleaning``）。

    ``apply_cleaning=False`` 是只读审计：不返回删除区间，但报告
    ``groups_flagged``、每个被标记群组的最佳匹配与 Jaccard。默认
    只对照 val；只有显式把 test 放进 ``eval_ranges`` 才会读取 test
    内容决定训练集删除，调用方负责把该事实写入清洗记录。

    近重复是内容级「相似」而不是逐字节相同：阈值是研究者事先声明
    的相似度门槛，报告同时给出实际 Jaccard 与匹配对象，避免把
    「被标记」当成确定泄露。群组过短（不足一个 shingle）或过大
    （完整集合内存不可控）时显式跳过并计数。
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("near-duplicate threshold 必须是数值") from error
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(
            "near-duplicate threshold 必须落在 (0, 1]"
        )
    if (
        not isinstance(shingle_bytes, int)
        or isinstance(shingle_bytes, bool)
        or not 1 <= shingle_bytes <= _NEAR_DUP_MAX_SHINGLE_BYTES
    ):
        raise ValueError(
            "near-duplicate shingle_bytes 必须是 1.."
            f"{_NEAR_DUP_MAX_SHINGLE_BYTES} 的整数"
        )
    if (
        not isinstance(sketch_size, int)
        or isinstance(sketch_size, bool)
        or not 8 <= sketch_size <= _NEAR_DUP_MAX_SKETCH_SIZE
    ):
        raise ValueError(
            "near-duplicate sketch_size 必须是 8.."
            f"{_NEAR_DUP_MAX_SKETCH_SIZE} 的整数"
        )
    if separator is None or len(separator) == 0:
        raise ValueError(
            "near-duplicate 扫描需要非空群组分隔符"
        )

    excluded_set = set(
        (int(start), int(end))
        for start, end in already_excluded
    )
    eval_order = [
        split
        for split in ("val", "test")
        if split in eval_ranges
    ]
    eval_order.extend(
        sorted(
            split
            for split in eval_ranges
            if split not in ("val", "test")
        )
    )

    eval_groups: List[Tuple[str, int, int]] = []
    eval_index: Dict[int, List[int]] = {}
    eval_too_short = 0
    eval_too_large = 0

    for split in eval_order:
        begin, end = eval_ranges[split]
        if end <= begin:
            continue
        for span_start, span_end in _iter_group_spans(
            raw,
            separator,
            begin,
            end,
        ):
            length = span_end - span_start
            if length > _NEAR_DUP_MAX_GROUP_BYTES:
                eval_too_large += 1
                continue
            sketch = _near_duplicate_bottom_k(
                _near_duplicate_shingle_hashes(
                    raw,
                    span_start,
                    span_end,
                    shingle_bytes,
                ),
                sketch_size,
            )
            if not sketch:
                eval_too_short += 1
                continue

            group_index = len(eval_groups)
            eval_groups.append(
                (split, span_start, span_end)
            )
            for value in sketch:
                eval_index.setdefault(value, []).append(
                    group_index
                )

    # 精确复核缓存：按 (start, end) 复用完整 shingle 集合。
    # NumPy 可用时用 uint64 数组（每个 shingle 8 字节）并调用
    # intersect1d；没有 NumPy 时退回 Python set。缓存按近似
    # 占用限界，避免大语料把内存吃满。
    exact_cache: "OrderedDict[Tuple[int, int], Tuple[object, int]]" = (
        OrderedDict()
    )
    exact_cache_bytes = 0

    def _full_shingle_values(
        span: Tuple[int, int],
    ):
        nonlocal exact_cache_bytes
        cached = exact_cache.get(span)
        if cached is not None:
            exact_cache.move_to_end(span)
            return cached[0]

        count = max(
            0,
            span[1] - span[0] - shingle_bytes + 1,
        )
        if np is not None:
            values = np.unique(
                np.fromiter(
                    _near_duplicate_shingle_hashes(
                        raw,
                        span[0],
                        span[1],
                        shingle_bytes,
                    ),
                    dtype=np.uint64,
                    count=count,
                )
            )
            size = int(values.nbytes)
        else:
            value_set = _near_duplicate_shingle_set(
                raw,
                span[0],
                span[1],
                shingle_bytes,
            )
            values = value_set
            # Python set 中每个 int 对象加哈希槽远大于 8 字节。
            size = len(value_set) * 48 + (
                span[1] - span[0]
            )

        while (
            exact_cache
            and exact_cache_bytes + size
            > _NEAR_DUP_EXACT_CACHE_BYTES
        ):
            _, (_, old_size) = exact_cache.popitem(
                last=False
            )
            exact_cache_bytes -= old_size
        exact_cache[span] = (values, size)
        exact_cache_bytes += size
        return values

    def _jaccard(
        left: Tuple[int, int],
        right: Tuple[int, int],
    ) -> float:
        left_values = _full_shingle_values(left)
        right_values = _full_shingle_values(right)
        if np is not None:
            if left_values.size == 0 or right_values.size == 0:
                return 0.0
            intersection = int(
                np.intersect1d(
                    left_values,
                    right_values,
                    assume_unique=True,
                ).size
            )
            union = (
                int(left_values.size)
                + int(right_values.size)
                - intersection
            )
        else:
            if not left_values or not right_values:
                return 0.0
            intersection = len(
                left_values & right_values
            )
            union = (
                len(left_values)
                + len(right_values)
                - intersection
            )
        if union <= 0:
            return 0.0
        return intersection / union

    split_rank = {"val": 0, "test": 1, "train": 2}
    prior_spans: List[Tuple[int, int]] = []
    prior_index: Dict[int, List[int]] = {}

    stats: Dict[str, object] = {
        "enabled": True,
        "policy": (
            "drop train groups with shingle Jaccard >= threshold "
            "against val/test or earlier train; val/test untouched"
            if apply_cleaning
            else "audit train groups with shingle Jaccard >= threshold "
            "against val/test or earlier train; read-only"
        ),
        "mode": "clean" if apply_cleaning else "audit",
        "threshold": round(threshold, 6),
        "shingle_bytes": shingle_bytes,
        "sketch_size": sketch_size,
        "similarity": "jaccard_of_byte_shingle_sets",
        "shingle_hash": "rolling_polynomial_64bit",
        "exact_verification": True,
        "max_group_bytes": _NEAR_DUP_MAX_GROUP_BYTES,
        "eval_splits": list(eval_order),
        "eval_groups": len(eval_groups),
        "eval_groups_too_short": eval_too_short,
        "eval_groups_too_large": eval_too_large,
        "groups_total": 0,
        "groups_scanned": 0,
        "groups_already_excluded": 0,
        "groups_too_short": 0,
        "groups_too_large": 0,
        "groups_flagged": 0,
        "groups_dropped": 0,
        "flagged_bytes": 0,
        "dropped_bytes": 0,
        "flagged_duplicate_of": {
            "val": 0,
            "test": 0,
            "train": 0,
        },
        "candidate_pairs": 0,
        "examples": [],
    }

    flagged_spans: List[Tuple[int, int]] = []

    for span_start, span_end in _iter_group_spans(
        raw,
        separator,
        0,
        train_end,
    ):
        stats["groups_total"] += 1
        span = (span_start, span_end)

        if span in excluded_set:
            stats["groups_already_excluded"] += 1
            continue

        length = span_end - span_start
        if length > _NEAR_DUP_MAX_GROUP_BYTES:
            stats["groups_too_large"] += 1
            continue

        sketch = _near_duplicate_bottom_k(
            _near_duplicate_shingle_hashes(
                raw,
                span_start,
                span_end,
                shingle_bytes,
            ),
            sketch_size,
        )
        if not sketch:
            stats["groups_too_short"] += 1
            continue

        stats["groups_scanned"] += 1

        candidates = set()
        for value in sketch:
            for index in eval_index.get(value, ()):
                candidates.add(("eval", index))
            for index in prior_index.get(value, ()):
                candidates.add(("train", index))

        stats["candidate_pairs"] += len(candidates)

        best_key = None
        best_jaccard = 0.0
        best_reason: Optional[str] = None
        best_other: Optional[Tuple[int, int]] = None

        # 候选按来源与位置排序，保证同一数据/参数下报告与
        # 示例在不同 PYTHONHASHSEED 下完全一致。
        for origin, index in sorted(candidates):
            if origin == "eval":
                reason, other_start, other_end = eval_groups[
                    index
                ]
            else:
                reason = "train"
                other_start, other_end = prior_spans[index]
            other = (other_start, other_end)
            if other == span:
                continue

            jaccard = _jaccard(span, other)
            if jaccard + 1e-12 < threshold:
                continue

            key = (
                -jaccard,
                split_rank.get(reason, 3),
                other_start,
                other_end,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_jaccard = jaccard
                best_reason = reason
                best_other = other

        if best_reason is not None:
            stats["groups_flagged"] += 1
            stats["flagged_bytes"] += length
            stats["flagged_duplicate_of"][best_reason] = (
                stats["flagged_duplicate_of"].get(best_reason, 0)
                + 1
            )
            flagged_spans.append(span)
            if len(stats["examples"]) < 5:
                stats["examples"].append([
                    span_start,
                    span_end,
                    best_reason,
                    round(best_jaccard, 6),
                    best_other[0],
                    best_other[1],
                ])

            if apply_cleaning:
                stats["groups_dropped"] += 1
                stats["dropped_bytes"] += length
            # 被标记的群组（无论删除还是只读审计）不能再作为
            # 后续群组的「更早 train」匹配对象：只保留第一个
            # 出现的群组，避免重复簇内产生 O(n^2) 候选。
            continue

        # 未标记的群组进入 prior 索引；只保留第一个出现的
        # 群组，避免重复簇内候选爆炸。
        prior_position = len(prior_spans)
        prior_spans.append(span)
        for value in sketch:
            prior_index.setdefault(value, []).append(
                prior_position
            )

    merged = _merge_byte_intervals(
        sorted(
            list(excluded_set)
            + (flagged_spans if apply_cleaning else [])
        )
    )
    projected = _merge_byte_intervals(
        sorted(list(excluded_set) + flagged_spans)
    )
    stats["groups_kept"] = (
        int(stats["groups_scanned"])
        - int(stats["groups_dropped"])
    )
    stats["groups_unflagged"] = (
        int(stats["groups_scanned"])
        - int(stats["groups_flagged"])
    )
    stats["post_clean_train_bytes"] = (
        train_end - sum(end - start for start, end in merged)
    )
    stats["projected_train_bytes_if_flagged_dropped"] = (
        train_end - sum(end - start for start, end in projected)
    )
    if apply_cleaning:
        return flagged_spans, stats
    return [], stats



def _eval_near_duplicate_audit(
    raw: bytearray,
    separator: bytes,
    ranges: Dict[str, Tuple[int, int]],
    threshold: float = _NEAR_DUP_DEFAULT_THRESHOLD,
    shingle_bytes: int = _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
    sketch_size: int = _NEAR_DUP_DEFAULT_SKETCH_SIZE,
) -> Dict[str, object]:
    """
    评估集内部与 val↔test 之间的近重复群组只读审计。

    精确重复群组已由内容指纹门拦截，但同一文档的修订版/转载版
    仍可能以不同字节同时出现在 val 或 test：

    - split 内部近重复群组会把同一内容重复计入评估损失、
      固定窗口 bootstrap 与 checkpoint 选型，使有效样本量
      小于名义窗口/群组数，置信区间偏窄；
    - val↔test 近重复会让 val 选型通过近似内容间接看到 test，
      test 指标不再完全独立。

    本函数只读、只报告，不删除/修改任何字节。相似度仍是字节
    k-shingle 集合的精确 Jaccard，bottom-k 草图只用于生成候选。
    对每个 split 用并查集把达到阈值的近重复群组合并为簇，报告
    ``effective_groups``（簇数，即按「簇内全部冗余」计的独立
    内容单元保守下界）与 ``redundant_groups``（名义群组数 -
    簇数）。

    与训练集近重复审计的分工：训练集审计回答「train 是否把
    val/test 内容带进训练」，本函数回答「评估集自己是否把同一
    内容重复计数」。两者不能互相替代。
    """
    try:
        checked_threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "evaluation near-duplicate threshold 必须是数值"
        ) from error
    if (
        not math.isfinite(checked_threshold)
        or not 0.0 < checked_threshold <= 1.0
    ):
        raise ValueError(
            "evaluation near-duplicate threshold 必须落在 (0, 1]"
        )
    if (
        not isinstance(shingle_bytes, int)
        or isinstance(shingle_bytes, bool)
        or not 1 <= shingle_bytes <= _NEAR_DUP_MAX_SHINGLE_BYTES
    ):
        raise ValueError(
            "evaluation near-duplicate shingle_bytes 必须是 1.."
            f"{_NEAR_DUP_MAX_SHINGLE_BYTES} 的整数"
        )
    if (
        not isinstance(sketch_size, int)
        or isinstance(sketch_size, bool)
        or not 8 <= sketch_size <= _NEAR_DUP_MAX_SKETCH_SIZE
    ):
        raise ValueError(
            "evaluation near-duplicate sketch_size 必须是 8.."
            f"{_NEAR_DUP_MAX_SKETCH_SIZE} 的整数"
        )
    if separator is None or len(separator) == 0:
        raise ValueError(
            "evaluation near-duplicate 扫描需要非空群组分隔符"
        )

    split_order = [
        split
        for split in ("val", "test")
        if (
            split in ranges
            and int(ranges[split][1]) > int(ranges[split][0])
        )
    ]

    groups: List[Tuple[str, int, int]] = []
    split_group_indices: Dict[str, List[int]] = {
        split: [] for split in split_order
    }
    per_split: Dict[str, Dict[str, object]] = {}
    for split in split_order:
        per_split[split] = {
            "groups_total": 0,
            "groups_scanned": 0,
            "groups_too_short": 0,
            "groups_too_large": 0,
            "within_pairs": 0,
            "clusters": 0,
            "effective_groups": 0,
            "redundant_groups": 0,
            "effective_fraction": None,
            "largest_cluster_groups": 0,
        }

    candidate_pairs = 0
    exact_cache: "OrderedDict[Tuple[int, int], Tuple[object, int]]" = (
        OrderedDict()
    )
    exact_cache_bytes = 0

    def _full_shingle_set(
        span: Tuple[int, int],
    ) -> object:
        nonlocal exact_cache_bytes
        cached = exact_cache.get(span)
        if cached is not None:
            exact_cache.move_to_end(span)
            return cached[0]

        values, size = _near_duplicate_exact_values(
            raw,
            span[0],
            span[1],
            shingle_bytes,
        )
        while (
            exact_cache
            and exact_cache_bytes + size
            > _NEAR_DUP_EXACT_CACHE_BYTES
        ):
            _, (_, old_size) = exact_cache.popitem(
                last=False
            )
            exact_cache_bytes -= old_size
        exact_cache[span] = (values, size)
        exact_cache_bytes += size
        return values

    def _exact_jaccard(
        left: Tuple[int, int],
        right: Tuple[int, int],
    ) -> float:
        left_values = _full_shingle_set(left)
        right_values = _full_shingle_set(right)
        if np is not None:
            if (
                left_values.size == 0
                or right_values.size == 0
            ):
                return 0.0
            intersection = int(
                np.intersect1d(
                    left_values,
                    right_values,
                    assume_unique=True,
                ).size
            )
            union = (
                int(left_values.size)
                + int(right_values.size)
                - intersection
            )
        else:
            if not left_values or not right_values:
                return 0.0
            intersection = len(
                left_values & right_values
            )
            union = (
                len(left_values)
                + len(right_values)
                - intersection
            )
        if union <= 0:
            return 0.0
        return intersection / union

    # 只用 bottom-k 草图生成候选；是否近重复由完整 shingle
    # 集合的精确 Jaccard 决定。所有群组按 split 与位置顺序进入
    # 倒排索引，因此每对群组只比较一次，报告顺序确定。
    sketch_index: Dict[int, List[int]] = {}
    within_pairs: List[Tuple[int, int, float]] = []
    cross_pairs: List[Tuple[int, int, float]] = []

    for split in split_order:
        begin, end = (
            int(ranges[split][0]),
            int(ranges[split][1]),
        )
        for span_start, span_end in _iter_group_spans(
            raw,
            separator,
            begin,
            end,
        ):
            per_split[split]["groups_total"] += 1
            length = span_end - span_start
            if length > _NEAR_DUP_MAX_GROUP_BYTES:
                per_split[split]["groups_too_large"] += 1
                continue

            sketch = _near_duplicate_bottom_k(
                _near_duplicate_shingle_hashes(
                    raw,
                    span_start,
                    span_end,
                    shingle_bytes,
                ),
                sketch_size,
            )
            if not sketch:
                per_split[split]["groups_too_short"] += 1
                continue

            group_index = len(groups)
            groups.append((split, span_start, span_end))
            split_group_indices[split].append(group_index)
            per_split[split]["groups_scanned"] += 1

            candidates = set()
            for value in sketch:
                for other_index in sketch_index.get(
                    value, ()
                ):
                    candidates.add(other_index)

            span = (span_start, span_end)
            for other_index in sorted(candidates):
                candidate_pairs += 1
                other_split, other_start, other_end = groups[
                    other_index
                ]
                jaccard = _exact_jaccard(
                    span,
                    (other_start, other_end),
                )
                if jaccard + 1e-12 < checked_threshold:
                    continue
                if other_split == split:
                    within_pairs.append(
                        (group_index, other_index, jaccard)
                    )
                else:
                    cross_pairs.append(
                        (group_index, other_index, jaccard)
                    )

            for value in sketch:
                sketch_index.setdefault(value, []).append(
                    group_index
                )

    # 并查集只在 split 内部合并：同一近重复簇是一个「独立内容
    # 单元」，跨 split 的近似内容不计入 split 内部有效群组数。
    parent = list(range(len(groups)))
    rank = [0] * len(groups)

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for left, right, _jaccard in within_pairs:
        _union(left, right)

    for split in split_order:
        info = per_split[split]
        indices = split_group_indices[split]
        root_sizes: Dict[int, int] = {}
        for index in indices:
            root = _find(index)
            root_sizes[root] = root_sizes.get(root, 0) + 1
        clusters = len(root_sizes)
        scanned = int(info["groups_scanned"])
        info["within_pairs"] = sum(
            1
            for left, right, _jaccard in within_pairs
            if groups[left][0] == split
        )
        info["clusters"] = clusters
        info["effective_groups"] = clusters
        info["redundant_groups"] = max(0, scanned - clusters)
        info["effective_fraction"] = (
            clusters / scanned if scanned > 0 else None
        )
        info["largest_cluster_groups"] = (
            max(root_sizes.values()) if root_sizes else 0
        )
        examples = []
        for left, right, jaccard in within_pairs:
            if groups[left][0] != split:
                continue
            if len(examples) >= 5:
                break
            # 先报告较晚出现的群组，再报告其更早的匹配对象，
            # 与训练集近重复审计的示例口径一致。
            examples.append([
                groups[left][1],
                groups[left][2],
                groups[right][1],
                groups[right][2],
                round(jaccard, 6),
            ])
        info["examples"] = examples

    cross_split: Dict[str, Dict[str, object]] = {}
    for left_index in range(len(split_order)):
        for right_index in range(left_index + 1, len(split_order)):
            left = split_order[left_index]
            right = split_order[right_index]
            key = f"{left}_{right}"
            entries = [
                (i, j, value)
                for i, j, value in cross_pairs
                if {
                    groups[i][0],
                    groups[j][0],
                } == {left, right}
            ]
            cross_split[key] = {
                "pairs": len(entries),
                "examples": [
                    [
                        groups[i][1],
                        groups[i][2],
                        groups[j][1],
                        groups[j][2],
                        round(value, 6),
                    ]
                    for i, j, value in entries[:5]
                ],
            }

    return {
        "schema": "smelt-isr/eval-near-duplicate/v1",
        "mode": "audit",
        "policy": (
            "read-only near-duplicate audit within val/test and "
            "across val/test; no bytes are removed"
        ),
        "threshold": round(checked_threshold, 6),
        "shingle_bytes": shingle_bytes,
        "sketch_size": sketch_size,
        "similarity": "jaccard_of_byte_shingle_sets",
        "shingle_hash": "rolling_polynomial_64bit",
        "exact_verification": True,
        "max_group_bytes": _NEAR_DUP_MAX_GROUP_BYTES,
        "splits": list(split_order),
        "test_content_used": "test" in split_order,
        "candidate_pairs": candidate_pairs,
        "per_split": per_split,
        "cross_split": cross_split,
        "note": (
            "effective_groups 是近重复簇数；按「簇内全部冗余」"
            "计这是独立样本量的保守下界，簇内群组仍可能包含"
            "不同内容"
        ),
    }



def _train_group_health_audit(
    dataset: ByteDataset,
) -> Optional[Dict[str, object]]:
    """
    训练集群组级异常值审计：统计「长度达到阈值且单一字节占比
    极高」的低信息/退化群组。

    只看 train，且跳过已被清洗剔除的群组；val/test 不参与统计。
    这是异常值检查，不会自动删除数据（需要显式
    --drop-train-degenerate-groups 才整组清洗）。
    """
    separator = dataset.group_separator
    if separator is None:
        return None

    excluded = set(dataset.train_excluded_spans)
    stats: Dict[str, object] = {
        "min_bytes": _AUDIT_DEGENERATE_GROUP_MIN_BYTES,
        "top_share_threshold": (
            _AUDIT_DEGENERATE_GROUP_TOP_SHARE
        ),
        "groups_checked": 0,
        "long_groups_checked": 0,
        "degenerate_groups": 0,
        "degenerate_bytes": 0,
        "max_top_byte_share": 0.0,
        "examples": [],
    }

    for span_start, span_end in _iter_group_spans(
        dataset._raw_buffer,
        separator,
        0,
        dataset.split_bounds["train"][1],
    ):
        if (span_start, span_end) in excluded:
            continue

        stats["groups_checked"] += 1
        length = span_end - span_start
        if length < _AUDIT_DEGENERATE_GROUP_MIN_BYTES:
            continue

        stats["long_groups_checked"] += 1
        top_byte, top_share = _group_top_byte_share(
            dataset._raw,
            span_start,
            span_end,
        )
        stats["max_top_byte_share"] = max(
            float(stats["max_top_byte_share"]),
            top_share,
        )

        if top_share < _AUDIT_DEGENERATE_GROUP_TOP_SHARE:
            continue

        stats["degenerate_groups"] += 1
        stats["degenerate_bytes"] += length
        if len(stats["examples"]) < 5:
            stats["examples"].append([
                span_start,
                span_end,
                top_byte,
                round(top_share, 6),
            ])

    return stats



def _group_content_duplicate_audit(
    dataset: ByteDataset,
    split_names: List[str],
) -> Optional[Dict[str, object]]:
    """
    群组级精确内容去重审计。

    窗口签名扫描是有界抽样的，可能漏掉短群组或错位重复；
    这里对每个群组的完整内容做 blake2b-128 指纹，逐 split
    核对完全相同的群组。跨 split 的重复群组是泄露级问题
    （同一文档/用户同时出现在训练与验证/测试两侧），必须
    进入 violations；split 内部的重复群组只作为重复样本/
    标签噪声提醒。

    训练集清洗（deduplicate_train_groups）剔除的群组不参与
    统计，因此清洗后的审计能反映「实际进入训练的内容」。
    """
    separator = dataset.group_separator
    if separator is None:
        return None

    excluded = set(dataset.train_excluded_spans)
    indices: Dict[str, Dict[bytes, List[int]]] = {}
    per_split: Dict[str, Dict[str, int]] = {}

    for split in split_names:
        begin, end = dataset.split_bounds[split]
        if end <= begin:
            continue

        index: Dict[bytes, List[int]] = {}
        groups = 0

        for span_start, span_end in _iter_group_spans(
            dataset._raw_buffer,
            separator,
            begin,
            end,
        ):
            if split == "train" and (
                (span_start, span_end) in excluded
            ):
                continue

            digest = _group_content_digest(
                dataset._raw,
                span_start,
                span_end,
            )
            index.setdefault(digest, []).append(span_start)
            groups += 1

        indices[split] = index
        per_split[split] = {
            "groups": groups,
            "unique_groups": len(index),
            "within_split_duplicates": (
                groups - len(index)
            ),
        }

    cross_split: Dict[str, object] = {}
    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        if left not in indices or right not in indices:
            continue

        left_index = indices[left]
        right_index = indices[right]
        common = left_index.keys() & right_index.keys()

        cross_split[f"{left}_{right}"] = {
            "shared_groups": len(common),
            "examples": [
                [
                    left_index[digest][0],
                    right_index[digest][0],
                ]
                for digest in sorted(common)[:3]
            ],
        }

    return {
        "per_split": per_split,
        "cross_split": cross_split,
    }
