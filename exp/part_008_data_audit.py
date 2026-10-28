from __future__ import annotations



def _content_chunk_duplicate_audit(
    dataset: ByteDataset,
    split_names: List[str],
) -> Dict[str, object]:
    """
    跨 split 内容定义分块重复审计。

    这是窗口签名扫描的独立补充：窗口扫描受采样步长限制，
    错位复制的长片段可能没有任何一个采样窗口完全相同；分块
    指纹由局部内容决定，错位后仍会重新对齐。群组分隔符本身
    是结构字节，分块只在群组内容上进行，避免把重复的分隔符
    误报为内容泄露。
    """
    min_chunk = max(
        _AUDIT_CHUNK_MIN_BYTES,
        dataset.seq_len + 1,
    )
    max_chunk = max(
        _AUDIT_CHUNK_MAX_BYTES,
        min_chunk * 4,
    )

    excluded = set(dataset.train_excluded_spans)
    indices: Dict[str, Dict[bytes, List[Tuple[int, int]]]] = {}
    per_split: Dict[str, Dict[str, object]] = {}

    for split in split_names:
        begin, end = dataset.split_bounds[split]
        if end <= begin:
            continue

        if dataset.group_separator is None:
            intervals = _subtract_byte_intervals(
                begin,
                end,
                (
                    dataset.train_excluded_spans
                    if split == "train"
                    else []
                ),
            )
        else:
            intervals = [
                (span_start, span_end)
                for span_start, span_end in _iter_group_spans(
                    dataset._raw_buffer,
                    dataset.group_separator,
                    begin,
                    end,
                )
                if not (
                    split == "train"
                    and (span_start, span_end) in excluded
                )
            ]

        scan = _content_defined_chunk_scan(
            dataset._raw,
            intervals,
            min_chunk,
            max_chunk,
        )
        indices[split] = scan["chunks"]
        per_split[split] = {
            "content_bytes": sum(
                stop - start for start, stop in intervals
            ),
            "chunks": scan["total_chunks"],
            "unique_chunks": scan["unique_chunks"],
            "within_split_duplicates": scan[
                "within_duplicates"
            ],
            "chunk_bytes": scan["total_bytes"],
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

        shared_chunks = 0
        shared_bytes = 0
        examples = []

        # bytes 哈希随进程随机化；按摘要排序后再取示例，保证
        # 同一归档/审计报告在不同 PYTHONHASHSEED 下逐字一致。
        for digest in sorted(common):
            left_occurrences = left_index[digest]
            right_occurrences = right_index[digest]
            matched = min(
                len(left_occurrences),
                len(right_occurrences),
            )
            length = (
                left_occurrences[0][1]
                - left_occurrences[0][0]
            )
            shared_chunks += matched
            shared_bytes += matched * length
            if len(examples) < 3:
                examples.append([
                    left_occurrences[0][0],
                    right_occurrences[0][0],
                    length,
                ])

        cross_split[f"{left}_{right}"] = {
            "shared_chunks": shared_chunks,
            "shared_bytes": shared_bytes,
            "examples": examples,
        }

    return {
        "min_chunk_bytes": min_chunk,
        "max_chunk_bytes": max_chunk,
        "per_split": per_split,
        "cross_split": cross_split,
    }



def _window_signature_scan(
    raw: memoryview,
    offset: int,
    length: int,
    window_len: int,
    max_windows: int = _AUDIT_MAX_WINDOWS,
    exclude_spans: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """
    对 [offset, offset + length) 内的滑动窗口做有界签名扫描。

    对每个采样窗口取 blake2b-128 摘要，用于发现跨 split 的重复
    内容（数据泄露的常见来源）与 split 内部重复。窗口数超过
    max_windows 时等间距抽样，返回实际采样规模。该扫描只做
    数据审计，不参与特征计算或模型决策。

    exclude_spans 是绝对字节区间（如训练集清洗剔除的群组）；
    与这些区间重叠的窗口不参与签名统计，避免把已经不会进入
    训练的内容继续标记为泄露。
    """
    total = length - window_len + 1

    if total <= 0:
        return {
            "sampled": 0,
            "total": 0,
            "stride": 1,
            "within_duplicates": 0,
            "skipped_excluded": 0,
            "signatures": {},
        }

    stride = max(1, math.ceil(total / max_windows))
    excluded = _merge_byte_intervals(
        list(exclude_spans or [])
    )

    signatures: Dict[bytes, List[int]] = {}
    within_duplicates = 0
    sampled = 0
    skipped_excluded = 0
    excluded_index = 0

    for start in range(0, total, stride):
        absolute_start = offset + start

        while (
            excluded_index < len(excluded)
            and excluded[excluded_index][1] <= absolute_start
        ):
            excluded_index += 1

        if (
            excluded_index < len(excluded)
            and excluded[excluded_index][0]
            < absolute_start + window_len
        ):
            skipped_excluded += 1
            continue

        digest = hashlib.blake2b(
            raw[
                absolute_start:
                absolute_start + window_len
            ],
            digest_size=16,
        ).digest()
        sampled += 1

        bucket = signatures.get(digest)
        if bucket is None:
            signatures[digest] = [start]
        else:
            within_duplicates += 1
            if len(bucket) < 4:
                bucket.append(start)

    return {
        "sampled": sampled,
        "total": total,
        "stride": stride,
        "within_duplicates": within_duplicates,
        "skipped_excluded": skipped_excluded,
        "signatures": signatures,
    }



def _leakage_scan_coverage(
    scans: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    汇总窗口签名扫描的覆盖情况，暴露「零发现」的证据强度。

    「未发现跨 split 重复窗口」只对被采样到的窗口成立：窗口
    总数超过 ``max_windows`` 时等间距抽样（stride>1），未采样
    窗口之间的重复完全不在证据范围内。把 coverage 写进审计，
    读者才能区分「穷尽扫描的零发现」与「抽样扫描的零发现」；
    内容定义分块审计按局部内容指纹穷尽扫描，是错位重复的
    兜底，但同样不能替代窗口级证据的口径披露。
    """
    per_split: Dict[str, Dict[str, object]] = {}
    exhaustive = True

    for split, scan in scans.items():
        total = int(scan["total"])
        sampled = int(scan["sampled"])
        stride = int(scan["stride"])
        per_split[split] = {
            "sampled": sampled,
            "total": total,
            "stride": stride,
            "coverage": (
                round(sampled / total, 6) if total > 0 else None
            ),
        }
        if total > 0 and stride > 1:
            exhaustive = False

    residual_risk: Optional[str] = None
    if not exhaustive:
        worst = max(
            (
                (int(info["stride"]), split)
                for split, info in per_split.items()
                if int(info["total"]) > 0
            ),
            default=(1, ""),
        )
        split_text = ", ".join(
            f"{split}:{info['sampled']}/{info['total']}"
            + (
                f"(stride={info['stride']})"
                if int(info["stride"]) > 1
                else ""
            )
            for split, info in per_split.items()
            if int(info["total"]) > 0
        )
        residual_risk = (
            f"窗口签名扫描未穷尽（最大 stride={worst[0]}；"
            f"{split_text}）：「未发现跨 split 重复窗口」"
            "只覆盖等间距采样的窗口，不等于「不存在重复」；"
            "内容定义分块审计为穷尽扫描，可兜底检出错位重复，"
            "但不能据此主张窗口级零泄露"
        )

    return {
        "exhaustive": exhaustive,
        "per_split": per_split,
        "residual_risk": residual_risk,
    }



def _scan_text_integrity(
    raw: memoryview,
    offset: int,
    length: int,
    chunk: int = 1 << 20,
    final: bool = False,
    warmup: bool = True,
) -> Dict[str, int]:
    """
    扫描一个 split 的文本完整性（数据清洗的标签噪声检查）。

    字节级语言模型的标签就是原始字节，损坏字节等价于标签
    噪声，必须在训练前可见：

    - utf8_replacements：无法按 UTF-8 解码的字节数。这里不用
      ``errors="replace"`` 统计 U+FFFD，因为语料本身可能合法地
      包含字面替换字符 U+FFFD；用 surrogateescape 把每个非法
      字节映射为 U+DC80..U+DCFF 代理码点，既能与
      ``_invalid_utf8_nul_spans`` 的清洗口径逐字节一致，也不会
      把合法文本误判为标签噪声；
    - nul_bytes：NUL（0x00）字节数量，文本语料中出现即疑似
      二进制污染或文件拼接错误。

    分块解码；只做诊断，不修改任何数据。

    切分边界处不制造假阳性：切分点之前最多 3 个字节先喂给
    解码器做状态预热（多字节 UTF-8 字符可以横跨 split 边界，
    文件整体仍然完全合法）；``final`` 只在扫描的是文件真正
    的末尾时为 True，否则末尾不完整的多字节序列留给下一个
    split 续接（其解码计数落在后续扫描里）。

    ``warmup=False`` 供逐群组扫描使用：每个群组按独立文本
    解码，不从前一段（分隔符）借状态。
    """
    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="surrogateescape"
    )

    # 预热：把解码器带到「从 offset 前几个字节开始」的状态。
    # 预热输出（可能包含边界前的字符或无效序列）全部丢弃，
    # 不计入任何统计；但解码器可能缓冲着 offset 之前未完成的
    # 多字节序列。若后续字节把它判为非法，surrogateescape 会
    # 在第一个当前块里吐出属于 offset 之前的代理码点。必须跳过
    # 这些前缀代理码点，否则上一段末尾的坏字节会被记到本段
    #（测试集标签完整性门会因此拒绝一个字节干净的 test）。
    pre_offset_pending = 0
    if warmup and offset > 0:
        decoder.decode(
            bytes(raw[max(0, offset - 3):offset]),
            final=False,
        )
        pre_offset_pending = len(decoder.getstate()[0])

    replacements = 0
    nul_bytes = 0
    position = offset
    end = offset + length

    while position < end:
        stop = min(position + chunk, end)
        block = bytes(raw[position:stop])
        text = decoder.decode(
            block,
            final=final and stop >= end,
        )
        for character in text:
            codepoint = ord(character)
            if 0xDC80 <= codepoint <= 0xDCFF:
                # 前 pre_offset_pending 个代理码点对应 offset
                # 之前的未完成序列，不属于本段。一旦出现普通
                # 字符，说明之前的缓冲已被完整解码，后续代理
                # 码点都属于本段。
                if pre_offset_pending > 0:
                    pre_offset_pending -= 1
                    continue
                replacements += 1
            else:
                pre_offset_pending = 0
                if character == "\x00":
                    nul_bytes += 1
        position = stop

    return {
        "utf8_replacements": replacements,
        "nul_bytes": nul_bytes,
    }



def _invalid_utf8_nul_spans(
    raw: memoryview,
    offset: int,
    length: int,
    final: bool,
    chunk: int = 1 << 20,
) -> List[Tuple[int, int, str]]:
    """
    返回 [offset, offset+length) 内非法 UTF-8 字节与 NUL 字节的
    精确区间。

    使用 ``errors="surrogateescape"`` 的增量解码器：每个非法
    字节映射为一个 U+DC80..U+DCFF 代理码点，再把解码结果逐字符
    重新编码回字节，就能把码点位置映射回原始字节偏移。多字节
    合法字符跨分块时由解码器缓冲，偏移通过 ``pending_start``
    续接；``final=False`` 时末尾不完整的多字节序列留给下一个
    split，不会把合法字符的切分点误判为损坏。
    """
    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="surrogateescape"
    )

    # 预热：offset 可能落在一个合法的多字节字符中间（split 边界
    # 不保证在字符边界上）。先把 offset 前最多 3 个字节喂给
    # 解码器，并记录它缓冲了多少「属于 offset 之前」的字节。
    # 之后解码出的第一个字符若消费了这些前缀字节，只能把其中
    # 落在 [offset, ...) 的实际字节数推进 cursor，不能把整个
    # 字符都算到本段（否则跨边界的合法字符会被误报为损坏）。
    pre_offset_pending = 0
    if offset > 0:
        decoder.decode(
            bytes(raw[max(0, offset - 3):offset]),
            final=False,
        )
        pre_offset_pending = len(decoder.getstate()[0])

    spans: List[Tuple[int, int, str]] = []
    position = offset
    end = offset + length
    cursor = offset
    skip_prefix = pre_offset_pending

    while position < end:
        stop = min(position + chunk, end)
        block = bytes(raw[position:stop])
        text = decoder.decode(
            block,
            final=final and stop >= end,
        )

        for character in text:
            encoded = character.encode(
                "utf-8",
                "surrogateescape",
            )
            size = len(encoded)
            codepoint = ord(character)

            if (
                size == 1
                and 0xDC80 <= codepoint <= 0xDCFF
            ):
                # 前缀遗留的非法字节可能先于本段被吐出；只跳过
                # 属于 offset 之前的那几个代理码点。
                if skip_prefix > 0:
                    skip_prefix -= 1
                    continue
                spans.append(
                    (cursor, cursor + 1, "invalid_utf8")
                )
            elif size > skip_prefix and skip_prefix > 0:
                # 合法的跨边界字符：只把 offset 之后的字节数
                # 推进 cursor，跳过整个字符的统计。
                cursor += size - skip_prefix
                skip_prefix = 0
                continue
            elif character == "\x00":
                spans.append((cursor, cursor + 1, "nul"))

            if skip_prefix > 0:
                # 理论上只有前缀遗留的非法字节会走到这里；安全
                # 起见不把尚未消费完的前缀算进本段。
                skip_prefix -= 1
                continue
            cursor += size

        position = stop

    return spans



def _plan_train_corruption_cleaning(
    raw: bytearray,
    separator: bytes,
    train_end: int,
    already_excluded: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划训练集标签噪声清洗：只从 train 中剔除「整组包含非法
    UTF-8 或 NUL 字节」的群组，绝不改动 val/test。

    字节级语言模型的监督位就是原始字节，无法解码的字节或 NUL
    就是标签噪声。这里整组剔除而不是修补字节：修补会凭空发明
    标签，整组剔除则能保证该群组不会以任何窗口形式进入训练。
    判定口径与 audit_data_splits 的文本完整性扫描一致（逐群组
    独立解码，不把用户配置的分隔符计入损坏）。已经被去重清洗
    剔除的群组不会重复扫描/统计。
    """
    excluded_set = set(already_excluded)
    excluded: List[Tuple[int, int]] = []
    stats: Dict[str, object] = {
        "enabled": True,
        "policy": (
            "drop train groups containing invalid UTF-8 or NUL "
            "bytes; val/test untouched"
        ),
        "groups_total": 0,
        "groups_scanned": 0,
        "groups_already_excluded": 0,
        "groups_dropped": 0,
        "dropped_bytes": 0,
        "groups_with_invalid_utf8": 0,
        "groups_with_nul": 0,
        "invalid_utf8_replacements": 0,
        "nul_bytes": 0,
        "examples": [],
    }

    for span_start, span_end in _iter_group_spans(
        raw,
        separator,
        0,
        train_end,
    ):
        stats["groups_total"] += 1

        if (span_start, span_end) in excluded_set:
            stats["groups_already_excluded"] += 1
            continue

        stats["groups_scanned"] += 1
        integrity = _scan_text_integrity(
            memoryview(raw),
            span_start,
            span_end - span_start,
            final=True,
            warmup=False,
        )

        if (
            integrity["utf8_replacements"] <= 0
            and integrity["nul_bytes"] <= 0
        ):
            continue

        if integrity["utf8_replacements"] > 0:
            stats["groups_with_invalid_utf8"] += 1
        if integrity["nul_bytes"] > 0:
            stats["groups_with_nul"] += 1

        stats["invalid_utf8_replacements"] += integrity[
            "utf8_replacements"
        ]
        stats["nul_bytes"] += integrity["nul_bytes"]
        stats["groups_dropped"] += 1
        stats["dropped_bytes"] += span_end - span_start
        excluded.append((span_start, span_end))
        if len(stats["examples"]) < 5:
            stats["examples"].append([
                span_start,
                span_end,
                integrity["utf8_replacements"],
                integrity["nul_bytes"],
            ])

    stats["post_clean_train_bytes"] = (
        train_end
        - sum(
            end - start
            for start, end in _merge_byte_intervals(
                list(already_excluded) + excluded
            )
        )
    )
    return excluded, stats



def _plan_train_corruption_cleaning_bytes(
    raw: bytearray,
    train_end: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划无群组连续字节流的训练集标签噪声清洗：只从 train 中
    剔除非法 UTF-8 字节与 NUL 字节本身，绝不修补字节，也绝不
    改动 val/test。

    与群组级清洗不同，这里不需要用户提供分隔符：损坏字节的
    位置由解码器直接定位。被剔除的字节不进入训练窗口、unigram/
    n-gram 计数或分布审计；仍保留的合法字节继续参与训练。
    """
    # 按整个文件解码，再把落在 train 内的坏字节归入清洗。只
    # 扫描到 train_end 且 final=False 会漏掉「train 末尾的非法
    # lead byte」：它需要后继 val/test 字节才能判定是否非法。
    # 全局扫描还能保留跨 split 的合法多字节字符，不会误删。
    tagged = [
        (start, end, kind)
        for start, end, kind in _invalid_utf8_nul_spans(
            memoryview(raw),
            0,
            len(raw),
            final=True,
        )
        if start < train_end
    ]

    invalid_bytes = sum(
        1
        for _, _, kind in tagged
        if kind == "invalid_utf8"
    )
    nul_bytes = sum(
        1
        for _, _, kind in tagged
        if kind == "nul"
    )
    excluded = _merge_byte_intervals(
        [(start, end) for start, end, _ in tagged]
    )

    stats: Dict[str, object] = {
        "enabled": True,
        "scope": "bytes",
        "policy": (
            "drop train bytes that are invalid UTF-8 or NUL; "
            "val/test untouched"
        ),
        "invalid_utf8_bytes": invalid_bytes,
        "nul_bytes": nul_bytes,
        "excluded_spans": len(excluded),
        "dropped_bytes": sum(
            end - start for start, end in excluded
        ),
        "examples": [
            [start, end]
            for start, end in excluded[:5]
        ],
    }
    stats["post_clean_train_bytes"] = (
        train_end - int(stats["dropped_bytes"])
    )
    return excluded, stats



def _group_top_byte_share(
    raw: memoryview,
    start: int,
    end: int,
) -> Tuple[int, float]:
    """返回群组内最高频字节及其占比（空群组返回 (0, 0.0)）。"""
    length = end - start
    if length <= 0:
        return 0, 0.0

    # ``torch.frombuffer`` 不能安全承载只读缓冲区：PyTorch 会
    # 发出 UserWarning（``-W error`` 下直接失败），因为它无法
    # 阻止写入。这里用可写的 bytearray 复制切片，代价与原来的
    # ``bytes(...)`` 复制相同。
    content = bytearray(raw[start:end])
    counts = torch.bincount(
        torch.frombuffer(content, dtype=torch.uint8),
        minlength=256,
    )
    top_count = int(counts.max())
    return int(counts.argmax()), top_count / length



def _plan_train_degenerate_cleaning(
    raw: bytearray,
    separator: bytes,
    train_end: int,
    already_excluded: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划训练集异常值清洗：只从 train 中整组剔除「长度达到
    阈值且单一字节占比极高」的低信息/退化群组，绝不改动
    val/test。

    判定只使用群组自身内容（没有跨 split 统计量），阈值固定
    并记录在审计中；已经被去重/标签噪声清洗剔除的群组不重复
    统计。
    """
    excluded_set = set(already_excluded)
    excluded: List[Tuple[int, int]] = []
    stats: Dict[str, object] = {
        "enabled": True,
        "policy": (
            "drop train groups with length >= "
            f"{_AUDIT_DEGENERATE_GROUP_MIN_BYTES} and top-byte "
            f"share >= {_AUDIT_DEGENERATE_GROUP_TOP_SHARE}; "
            "val/test untouched"
        ),
        "min_bytes": _AUDIT_DEGENERATE_GROUP_MIN_BYTES,
        "top_share_threshold": (
            _AUDIT_DEGENERATE_GROUP_TOP_SHARE
        ),
        "groups_total": 0,
        "groups_scanned": 0,
        "groups_already_excluded": 0,
        "groups_dropped": 0,
        "dropped_bytes": 0,
        "max_top_byte_share": 0.0,
        "examples": [],
    }

    for span_start, span_end in _iter_group_spans(
        raw,
        separator,
        0,
        train_end,
    ):
        stats["groups_total"] += 1

        if (span_start, span_end) in excluded_set:
            stats["groups_already_excluded"] += 1
            continue

        stats["groups_scanned"] += 1
        top_byte, top_share = _group_top_byte_share(
            memoryview(raw),
            span_start,
            span_end,
        )
        stats["max_top_byte_share"] = max(
            float(stats["max_top_byte_share"]),
            top_share,
        )

        if (
            span_end - span_start
            < _AUDIT_DEGENERATE_GROUP_MIN_BYTES
            or top_share < _AUDIT_DEGENERATE_GROUP_TOP_SHARE
        ):
            continue

        stats["groups_dropped"] += 1
        stats["dropped_bytes"] += span_end - span_start
        excluded.append((span_start, span_end))
        if len(stats["examples"]) < 5:
            stats["examples"].append([
                span_start,
                span_end,
                top_byte,
                round(top_share, 6),
            ])

    stats["post_clean_train_bytes"] = (
        train_end
        - sum(
            end - start
            for start, end in _merge_byte_intervals(
                list(already_excluded) + excluded
            )
        )
    )
    return excluded, stats



def _degenerate_run_spans(
    raw: memoryview,
    begin: int,
    end: int,
    min_bytes: int,
) -> List[Tuple[int, int, int]]:
    """返回 [begin, end) 内长度 >= min_bytes 的同字节连续段。"""
    spans: List[Tuple[int, int, int]] = []
    position = begin

    while position < end:
        value = raw[position]
        stop = position + 1
        while stop < end and raw[stop] == value:
            stop += 1
        if stop - position >= min_bytes:
            spans.append((position, stop, int(value)))
        position = stop

    return spans



def _plan_train_degenerate_run_cleaning(
    raw: bytearray,
    train_end: int,
    min_bytes: int,
    already_excluded: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划无群组连续字节流的训练集异常值清洗：只从 train 剔除
    长度达到阈值的单一字节连续段（填充、重复字符、错误拼接的
    常见形态），绝不修补字节，也绝不动 val/test。

    与群组版异常值清洗同一防御目标；连续流没有群组边界，
    只能按退化段本身剔除。已经在其他清洗中被剔除的区间不再
    扫描，但同一条原始退化段被排除区间切成两段时，两个剩余
    片段仍分别接受阈值检查：否则单个被排除字节就能让整条退化
    段逃过清洗，清洗审计反而会继续报告残留退化段。
    """
    merged_excluded = _merge_byte_intervals(
        list(already_excluded)
    )
    excluded: List[Tuple[int, int]] = []
    stats: Dict[str, object] = {
        "enabled": True,
        "scope": "bytes",
        "policy": (
            f"drop train byte runs of one value with length "
            f">= {min_bytes}; val/test untouched"
        ),
        "min_bytes": min_bytes,
        "runs_dropped": 0,
        "dropped_bytes": 0,
        "max_run_bytes": 0,
        "examples": [],
    }

    for interval_start, interval_end in _subtract_byte_intervals(
        0,
        train_end,
        merged_excluded,
    ):
        for start, end, value in _degenerate_run_spans(
            memoryview(raw),
            interval_start,
            interval_end,
            min_bytes,
        ):
            stats["runs_dropped"] += 1
            stats["dropped_bytes"] += end - start
            stats["max_run_bytes"] = max(
                int(stats["max_run_bytes"]),
                end - start,
            )
            excluded.append((start, end))
            if len(stats["examples"]) < 5:
                stats["examples"].append([
                    start,
                    end,
                    value,
                ])

    stats["post_clean_train_bytes"] = (
        train_end
        - sum(
            end - start
            for start, end in _merge_byte_intervals(
                list(already_excluded) + excluded
            )
        )
    )
    return excluded, stats



def _train_degenerate_run_audit(
    dataset: ByteDataset,
) -> Optional[Dict[str, object]]:
    """
    无群组连续字节流的训练集异常值审计：统计长度达到阈值的
    同字节连续段。只看 train，并跳过已被清洗剔除的区间；
    val/test 不参与统计。清洗需要显式
    --drop-train-degenerate-runs。
    """
    if dataset.group_separator is not None:
        return None

    begin, end = dataset.split_bounds["train"]
    intervals = _subtract_byte_intervals(
        begin,
        end,
        dataset.train_excluded_spans,
    )
    stats: Dict[str, object] = {
        "min_bytes": _AUDIT_DEGENERATE_RUN_MIN_BYTES,
        "runs": 0,
        "degenerate_bytes": 0,
        "max_run_bytes": 0,
        "examples": [],
    }

    for interval_begin, interval_end in intervals:
        for start, stop, value in _degenerate_run_spans(
            dataset._raw,
            interval_begin,
            interval_end,
            _AUDIT_DEGENERATE_RUN_MIN_BYTES,
        ):
            stats["runs"] += 1
            stats["degenerate_bytes"] += stop - start
            stats["max_run_bytes"] = max(
                int(stats["max_run_bytes"]),
                stop - start,
            )
            if len(stats["examples"]) < 5:
                stats["examples"].append([
                    start,
                    stop,
                    value,
                ])

    return stats



def _plan_train_shared_chunk_cleaning(
    raw: bytearray,
    separator: bytes,
    train_end: int,
    eval_ranges: Dict[str, Tuple[int, int]],
    min_chunk: int,
    max_chunk: int,
    already_excluded: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划训练集群组级「跨 split 近似重复」清洗：只从 train 中
    整组剔除包含与 val/test 共享的内容定义块的群组，绝不改动
    val/test。

    精确群组指纹只能抓内容完全相同的文档；同一段文字被包在
    更长的训练群组里（或整体错位复制）时指纹不同，但共享块
    仍会把评估内容泄露给训练。这里用与
    ``_content_chunk_duplicate_audit`` 完全相同的分块口径：
    只要训练群组内出现一个与 val/test 群组相同、长度至少为
    一个完整窗口的连续块，就整组剔除。整组剔除而不是删单个
    块，避免在连续字节流中制造新的跨群组邻接标签。
    """
    eval_chunks: set = set()
    eval_groups = 0

    for begin, end in eval_ranges.values():
        if end <= begin:
            continue
        for span_start, span_end in _iter_group_spans(
            raw,
            separator,
            begin,
            end,
        ):
            scan = _content_defined_chunk_scan(
                memoryview(raw),
                [(span_start, span_end)],
                min_chunk,
                max_chunk,
            )
            eval_chunks.update(scan["chunks"].keys())
            eval_groups += 1

    excluded_set = set(already_excluded)
    excluded: List[Tuple[int, int]] = []
    stats: Dict[str, object] = {
        "enabled": True,
        "policy": (
            "drop train groups sharing content-defined chunks "
            "with val/test; val/test untouched"
        ),
        "min_chunk_bytes": min_chunk,
        "max_chunk_bytes": max_chunk,
        "eval_groups": eval_groups,
        "eval_chunks": len(eval_chunks),
        "groups_total": 0,
        "groups_scanned": 0,
        "groups_already_excluded": 0,
        "groups_dropped": 0,
        "dropped_bytes": 0,
        "shared_chunks": 0,
        "examples": [],
    }

    for span_start, span_end in _iter_group_spans(
        raw,
        separator,
        0,
        train_end,
    ):
        stats["groups_total"] += 1

        if (span_start, span_end) in excluded_set:
            stats["groups_already_excluded"] += 1
            continue

        stats["groups_scanned"] += 1
        scan = _content_defined_chunk_scan(
            memoryview(raw),
            [(span_start, span_end)],
            min_chunk,
            max_chunk,
        )
        common = scan["chunks"].keys() & eval_chunks
        if not common:
            continue

        stats["groups_dropped"] += 1
        stats["dropped_bytes"] += span_end - span_start
        stats["shared_chunks"] += len(common)
        excluded.append((span_start, span_end))
        if len(stats["examples"]) < 5:
            stats["examples"].append([
                span_start,
                span_end,
                len(common),
            ])

    stats["post_clean_train_bytes"] = (
        train_end
        - sum(
            end - start
            for start, end in _merge_byte_intervals(
                list(already_excluded) + excluded
            )
        )
    )
    return excluded, stats



def _near_duplicate_shingle_hashes(
    raw,
    start: int,
    end: int,
    shingle_bytes: int,
):
    """
    逐个产出 [start, end) 内长度为 shingle_bytes 的字节 shingle
    的 64 位滚动多项式哈希。

    与逐 shingle 调用 blake2b 相比，滚动哈希把「提取全部 shingle」
    的成本降到线性；Jaccard 的精确复核仍在完整 shingle 哈希集合
    上做。64 位碰撞概率很低，但审计记录哈希口径，不把它说成
    逐字节完全相等。
    """
    length = end - start
    if length < shingle_bytes:
        return

    base = _NEAR_DUP_HASH_BASE
    mask = _NEAR_DUP_HASH_MASK
    # base^(shingle_bytes-1) mod 2^64：滚出最左字节时减去它的贡献。
    base_power = pow(base, shingle_bytes - 1, 1 << 64)

    value = 0
    for position in range(start, start + shingle_bytes):
        value = ((value * base) + raw[position]) & mask
    yield value

    for position in range(start + shingle_bytes, end):
        value = (
            (
                value
                - raw[position - shingle_bytes] * base_power
            )
            * base
            + raw[position]
        ) & mask
        yield value



def _near_duplicate_bottom_k(
    hashes,
    sketch_size: int,
) -> Tuple[int, ...]:
    """
    取 shingle 哈希的 bottom-k（最小的 k 个值），返回升序签名。

    bottom-k 是 Jaccard 的 MinHash 家族草图：两个高度重合的集合
    会共享大量最小哈希值，因此可把每个签名值当作倒排键生成候选。
    草图只用于候选生成，是否删除由完整 shingle 集合的精确 Jaccard
    决定。
    """
    heap: List[int] = []
    present: set = set()
    for value in hashes:
        # bottom-k 是 shingle **集合**的最小哈希，不是多重集：
        # 高度重复的内容（同一行重复很多次）会产生大量相同
        # shingle 哈希；若不去重，草图会被同一个值占满，丢掉
        # 真正参与 Jaccard 的其他值，近重复候选因此漏检。
        if value in present:
            continue
        if len(heap) < sketch_size:
            heapq.heappush(heap, -value)
            present.add(value)
        elif value < -heap[0]:
            removed = -heapq.heapreplace(heap, -value)
            present.discard(removed)
            present.add(value)
    return tuple(sorted(-item for item in heap))



def _near_duplicate_shingle_set(
    raw,
    start: int,
    end: int,
    shingle_bytes: int,
) -> set:
    """返回 [start, end) 的完整 shingle 哈希集合（精确复核用）。"""
    return set(
        _near_duplicate_shingle_hashes(
            raw,
            start,
            end,
            shingle_bytes,
        )
    )



def _near_duplicate_exact_values(
    raw,
    start: int,
    end: int,
    shingle_bytes: int,
) -> Tuple[object, int]:
    """
    返回 [start, end) 的完整 shingle 集合与近似内存占用。

    NumPy 可用时用去重后的 uint64 数组（8 字节/值）并配合
    ``intersect1d`` 做精确交集；没有 NumPy 时退回 Python set。
    与训练集近重复审计的精确复核口径一致，只是把缓存管理留给
    调用方。
    """
    count = max(0, end - start - shingle_bytes + 1)
    if np is not None:
        values = np.unique(
            np.fromiter(
                _near_duplicate_shingle_hashes(
                    raw,
                    start,
                    end,
                    shingle_bytes,
                ),
                dtype=np.uint64,
                count=count,
            )
        )
        return values, int(values.nbytes)

    values = _near_duplicate_shingle_set(
        raw,
        start,
        end,
        shingle_bytes,
    )
    # Python set 中每个 int 对象加哈希槽远大于 8 字节。
    return values, len(values) * 48 + max(1, end - start)



def _near_duplicate_threshold_or_default(
    value: object,
) -> float:
    """把存档/协议里的阈值转成有限 (0,1] 浮点，损坏值回退默认。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return _NEAR_DUP_DEFAULT_THRESHOLD
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        return _NEAR_DUP_DEFAULT_THRESHOLD
    return number



def _near_duplicate_shingle_or_default(
    value: object,
) -> int:
    """把存档/协议里的 shingle 长度转成合法整数，损坏值回退默认。"""
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= _NEAR_DUP_MAX_SHINGLE_BYTES
    ):
        return value
    return _NEAR_DUP_DEFAULT_SHINGLE_BYTES
