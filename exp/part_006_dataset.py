from __future__ import annotations



class ByteDataset:
    """
    字节级语言模型数据集。

    数据协议（防止数据泄露）：
    - 先划分、后处理：按原始字节流的时间先后顺序切成
      train / val / test 三段，三个 split 在原语料中前后相邻、
      互不重叠，切分全程不随机打乱（时序数据用过去训练、用未来
      验证和测试）；训练阶段的批级 shuffle 由随机采样窗口起点
      承担，不改变 split 边界；
    - 群组整体不跨界：传入 group_separator 时，切分位置吸附到
      群组边界（同一文档/同一用户的内容不会被拆到两个 split），
      train/val/test 任意两侧之间都不共享半个群组；
    - 训练期间 test 保持尘封：训练循环只访问 train，周期性验证
      只访问 val，test 仅在最终评估（--evaluate-test）时访问一次；
      训练集清洗默认也只对照 val 与更早的 train 群组，绝不用
      test 内容决定剔除哪些训练样本（那会让训练数据依赖 test）；
    - 本管线不做任何依赖全局统计量的预处理（标准化、归一化、
      缺失值填充等）；若后续扩展为连续值特征，必须只在 train 上
      fit，再 transform val/test，禁止在划分前用全量数据统计。

    token id 范围固定为 0..255；验证/测试窗口固定，避免不同实验
    之间采样噪声过大。
    """

    def __init__(
        self,
        filename: str,
        seq_len: int,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        device: Optional[torch.device] = None,
        group_separator: Optional[bytes] = None,
        deduplicate_train_groups: bool = False,
        drop_train_shared_chunks: bool = False,
        drop_train_near_duplicate_groups: bool = False,
        near_duplicate_audit: bool = False,
        near_duplicate_threshold: float = (
            _NEAR_DUP_DEFAULT_THRESHOLD
        ),
        near_duplicate_shingle_bytes: int = (
            _NEAR_DUP_DEFAULT_SHINGLE_BYTES
        ),
        drop_train_corrupt_groups: bool = False,
        drop_train_corrupt_bytes: bool = False,
        drop_train_degenerate_groups: bool = False,
        drop_train_degenerate_runs: bool = False,
        clean_train_against_test: bool = False,
        include_test_in_cleaning: bool = False,
    ):
        # 两个入口名都保留：远程分支的 clean_train_against_test
        # 与本次修正的 include_test_in_cleaning 语义相同。
        include_test_in_cleaning = bool(
            include_test_in_cleaning or clean_train_against_test
        )
        data_path = Path(filename)

        if not 0.0 < val_fraction < 1.0:
            raise ValueError(
                "val_fraction 必须落在 (0, 1)（验证集用于"
                "选型/早停，不能为空）"
            )

        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction 必须落在 [0, 1)")

        if val_fraction + test_fraction >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction 必须小于 1"
                "（训练集不能为空）"
            )

        # 文件在读取期间被截断时按实际读入长度处理。
        buf = bytearray(data_path.stat().st_size)
        with open(data_path, "rb") as handle:
            filled = handle.readinto(buf)

        if filled != len(buf):
            del buf[filled:]

        if len(buf) < seq_len + 2:
            raise ValueError(
                "数据文件太小，至少需要 seq_len + 2 个字节"
            )

        # 语料内容指纹：恢复训练时用来确认数据文件没有变化。
        self.data_sha256 = hashlib.sha256(buf).hexdigest()
        self.num_bytes = len(buf)
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        tokens = torch.frombuffer(buf, dtype=torch.uint8)

        # 数据审计（重复窗口扫描）直接读取原始字节缓冲区；
        # _raw_buffer 保留 bytearray 本身，供按分隔符切段扫描
        #（memoryview 不提供 find）。
        self._raw = memoryview(buf)
        self._raw_buffer = buf

        _ = device

        if group_separator is not None and len(group_separator) == 0:
            raise ValueError("group_separator 不能为空字节串")

        self.group_separator = group_separator
        self.group_cuts: Optional[Dict[str, int]] = None
        self.groups_per_split: Optional[Dict[str, int]] = None

        if deduplicate_train_groups and group_separator is None:
            raise ValueError(
                "deduplicate_train_groups 需要 group_separator："
                "没有显式的群组边界就无法在不改动 val/test 的"
                "前提下做内容级去重"
            )

        if drop_train_shared_chunks and group_separator is None:
            raise ValueError(
                "drop_train_shared_chunks 需要 group_separator："
                "没有显式的群组边界就无法整组剔除与 val/test "
                "共享内容的训练样本，也无法保证 val/test 不变"
            )
        self.drop_train_shared_chunks = bool(
            drop_train_shared_chunks
        )

        if (
            drop_train_near_duplicate_groups
            and group_separator is None
        ):
            raise ValueError(
                "drop_train_near_duplicate_groups 需要 "
                "group_separator：没有显式的群组边界就无法在"
                "不改动 val/test 的前提下整组剔除近重复训练"
                "样本"
            )
        self.drop_train_near_duplicate_groups = bool(
            drop_train_near_duplicate_groups
        )
        if near_duplicate_audit and group_separator is None:
            raise ValueError(
                "near_duplicate_audit 需要 group_separator："
                "近重复按群组定义，没有群组边界就无法区分"
                "文档/记录"
            )
        self.near_duplicate_audit = bool(near_duplicate_audit)
        try:
            checked_threshold = float(
                near_duplicate_threshold
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "near_duplicate_threshold 必须是数值"
            ) from error
        if (
            not math.isfinite(checked_threshold)
            or not 0.0 < checked_threshold <= 1.0
        ):
            raise ValueError(
                "near_duplicate_threshold 必须落在 (0, 1]"
            )
        if (
            not isinstance(
                near_duplicate_shingle_bytes, int
            )
            or isinstance(near_duplicate_shingle_bytes, bool)
            or not 1
            <= near_duplicate_shingle_bytes
            <= _NEAR_DUP_MAX_SHINGLE_BYTES
        ):
            raise ValueError(
                "near_duplicate_shingle_bytes 必须是 1.."
                f"{_NEAR_DUP_MAX_SHINGLE_BYTES} 的整数"
            )
        self.near_duplicate_threshold = checked_threshold
        self.near_duplicate_shingle_bytes = int(
            near_duplicate_shingle_bytes
        )

        if drop_train_corrupt_groups and group_separator is None:
            raise ValueError(
                "drop_train_corrupt_groups 需要 group_separator："
                "没有显式的群组边界就无法在不改动 val/test 的"
                "前提下整组剔除标签噪声"
            )
        self.drop_train_corrupt_groups = bool(
            drop_train_corrupt_groups
        )

        if (
            drop_train_corrupt_bytes
            and group_separator is not None
        ):
            raise ValueError(
                "drop_train_corrupt_bytes 仅用于无群组边界的"
                "连续字节流；有 --group-separator 时请使用 "
                "--drop-train-corrupt-groups 以保持群组完整"
            )
        self.drop_train_corrupt_bytes = bool(
            drop_train_corrupt_bytes
        )

        if (
            drop_train_degenerate_runs
            and group_separator is not None
        ):
            raise ValueError(
                "drop_train_degenerate_runs 仅用于无 "
                "--group-separator 的连续字节流；有群组边界时"
                "请使用 --drop-train-degenerate-groups 以保持"
                "群组完整"
            )
        self.drop_train_degenerate_runs = bool(
            drop_train_degenerate_runs
        )

        if (
            drop_train_degenerate_groups
            and group_separator is None
        ):
            raise ValueError(
                "drop_train_degenerate_groups 需要 "
                "group_separator：没有显式的群组边界就无法在"
                "不改动 val/test 的前提下整组剔除异常值"
            )
        self.drop_train_degenerate_groups = bool(
            drop_train_degenerate_groups
        )

        # test 只用于最终评估。训练集清洗若读取 test 群组/内容，
        # 训练样本集合就会依赖 test，「只解封一次」的协议被绕过；
        # 默认只对照 val 与更早的 train，旧行为必须显式开启并被
        # 记录。未启用任何对照 eval split 的清洗时该开关无效果。
        # include_test_in_cleaning 与 clean_train_against_test 是
        # 同义入口；两者任一为真都必须同时驱动清洗范围、校验
        # 与披露，不能只改 eval_content_access 造成假披露。
        self.clean_train_against_test = bool(
            include_test_in_cleaning
        )
        if self.clean_train_against_test and not (
            deduplicate_train_groups
            or drop_train_shared_chunks
            or drop_train_near_duplicate_groups
        ):
            raise ValueError(
                "clean_train_against_test 需要同时启用 "
                "deduplicate_train_groups、"
                "drop_train_shared_chunks 或 "
                "drop_train_near_duplicate_groups"
            )

        # 训练集清洗（可选的群组级/字节级）：只剔除 train 中
        # 与 val/test 或更早训练群组完全重复的群组、损坏字节或
        # 退化群组。被剔除的字节不会进入任何训练窗口；val/test
        # 的字节与切分位置保持完全不变。
        self.train_excluded_spans: List[Tuple[int, int]] = []
        self.train_cleaning: Optional[Dict[str, object]] = None
        self._train_start_intervals: Optional[
            List[Tuple[int, int]]
        ] = None
        self._train_interval_lows: Optional[torch.Tensor] = None
        self._train_interval_lengths: Optional[torch.Tensor] = None
        self._train_interval_cumulative: Optional[torch.Tensor] = None
        self.train_num_valid_starts: Optional[int] = None

        # 仅做时间顺序上的连续切分，不计算统计量、不打乱：
        # [0, n_train) 训练，[n_train, n_train + n_val) 验证，
        # 其余为测试。
        n_val = int(len(tokens) * val_fraction)
        n_test = int(len(tokens) * test_fraction)
        n_train = len(tokens) - n_val - n_test
        n_val_end = n_train + n_val

        if group_separator is not None:
            # 群组数据的防泄露协议：切分位置吸附到群组边界，
            # 同一群组的全部字节整体落在同一个 split，任何两侧
            # 之间都不交叉。吸附后会验证三段长度，宁可在划分
            # 阶段显式报错（避免无法对齐的评估协议）。
            cut_train, cut_val = _snap_group_split(
                buf,
                group_separator,
                n_train,
                n_val_end,
                min_len=seq_len + 1,
                has_test=test_fraction > 0.0,
            )
            n_train = cut_train
            n_val_end = cut_val
            self.group_cuts = {
                "train": cut_train,
                "val": cut_val,
            }
            self.groups_per_split = dict(
                zip(
                    ("train", "val", "test"),
                    _count_groups(
                        buf,
                        group_separator,
                        [cut_train, cut_val],
                    ),
                )
            )
            # 群组完整性由构造保证：分界点要么是分隔符之后的群组
            # 起始字节，要么是文件末尾（仅在 test_fraction=0 时
            # 作为 val 的右端）。这里再做一次防御性断言，任何
            # 重构错误的划界都会在数据加载阶段立刻暴露，而不是
            # 悄悄泄露。
            for cut in (cut_train, cut_val):
                if 0 < cut < len(buf):
                    if (
                        buf[cut - len(group_separator):cut]
                        != group_separator
                    ):
                        raise AssertionError(
                            "群组切分边界不在分隔符之后："
                            "内部错误，拒绝继续"
                        )

        self.split_bounds = {
            "train": (0, n_train),
            "val": (n_train, n_val_end),
            "test": (n_val_end, len(tokens)),
        }
        self.split_isolation = _split_isolation_record(
            self.split_bounds,
            len(tokens),
        )
        if (
            not self.split_isolation["contiguous"]
            or not self.split_isolation["covers_all_bytes"]
            or int(self.split_isolation["overlap_bytes"]) != 0
        ):
            raise AssertionError(
                "split 时序隔离检查失败：train/val/test 必须"
                "按时间顺序首尾相接、覆盖全部字节且互不重叠"
            )

        # 逐 split 内容指纹：整体数据 sha256 相同时，群组吸附/
        # 清洗/切分实现变化仍可能改变各段实际字节范围。把每段
        # 内容单独指纹化，resume 与跨运行核对才能确认 train/val/
        # test 仍是同一批字节，而不是只看整体文件与比例。
        self.split_sha256 = {
            split: hashlib.sha256(
                self._raw[begin:end]
            ).hexdigest()
            for split, (begin, end) in self.split_bounds.items()
        }

        # 训练集清洗的评估内容访问口径：默认只用 val 决定
        # 删除哪些训练群组。test 内容参与删除决策会让「测试集
        # 全程尘封」不再成立（即使 test 字节本身不被修改）：
        # 数据集构建已经根据 test 内容筛掉了特定训练样本。
        # 只有显式 --allow-eval-content-for-cleaning 才把 test
        # 纳入清洗参照，并把访问事实写入 data_audit。
        self.eval_content_access = {
            "splits": (
                ["val", "test"]
                if include_test_in_cleaning
                else ["val"]
            ),
            "purpose": "train_cleaning_reference",
            "test_content_used": bool(include_test_in_cleaning),
            "note": (
                "默认只用 val 决定训练集删除；test 内容不参与"
                "数据集构建。启用 --allow-eval-content-for-cleaning "
                "后 test 内容会参与删除决策，必须披露且不能"
                "同时声称 test 全程未被读取"
            ),
        }
        dedup_excluded: List[Tuple[int, int]] = []
        dedup_stats: Optional[Dict[str, object]] = None

        # 默认只用 val（验证集用于选型/早停）作为训练集清洗的
        # 对照；test 必须保持尘封，不能参与「剔除哪些训练群组」
        # 的决定。显式 clean_train_against_test 才恢复旧行为，
        # 并把该事实写入 train_cleaning 供审计/对比核对。
        cleaning_eval_ranges: Dict[
            str, Tuple[int, int]
        ] = {"val": (n_train, n_val_end)}
        if self.clean_train_against_test:
            cleaning_eval_ranges["test"] = (
                n_val_end,
                len(tokens),
            )

        if deduplicate_train_groups:
            dedup_excluded, dedup_stats = (
                _plan_train_group_dedup(
                    self._raw_buffer,
                    group_separator,
                    n_train,
                    cleaning_eval_ranges,
                )
            )

        shared_chunk_excluded: List[Tuple[int, int]] = []
        shared_chunk_stats: Optional[Dict[str, object]] = None

        if drop_train_shared_chunks:
            shared_min_chunk = max(
                _AUDIT_CHUNK_MIN_BYTES,
                seq_len + 1,
            )
            shared_chunk_excluded, shared_chunk_stats = (
                _plan_train_shared_chunk_cleaning(
                    self._raw_buffer,
                    group_separator,
                    n_train,
                    cleaning_eval_ranges,
                    shared_min_chunk,
                    max(
                        _AUDIT_CHUNK_MAX_BYTES,
                        shared_min_chunk * 4,
                    ),
                    dedup_excluded,
                )
            )

        near_duplicate_excluded: List[Tuple[int, int]] = []
        near_duplicate_stats: Optional[Dict[str, object]] = None

        if drop_train_near_duplicate_groups:
            (
                near_duplicate_excluded,
                near_duplicate_stats,
            ) = _plan_train_near_duplicate_cleaning(
                self._raw_buffer,
                group_separator,
                n_train,
                cleaning_eval_ranges,
                dedup_excluded + shared_chunk_excluded,
                threshold=self.near_duplicate_threshold,
                shingle_bytes=(
                    self.near_duplicate_shingle_bytes
                ),
                apply_cleaning=True,
            )

        corrupt_excluded: List[Tuple[int, int]] = []
        corrupt_stats: Optional[Dict[str, object]] = None

        if drop_train_corrupt_groups:
            corrupt_excluded, corrupt_stats = (
                _plan_train_corruption_cleaning(
                    self._raw_buffer,
                    group_separator,
                    n_train,
                    (
                        dedup_excluded
                        + shared_chunk_excluded
                        + near_duplicate_excluded
                    ),
                )
            )

        corrupt_byte_excluded: List[Tuple[int, int]] = []
        corrupt_byte_stats: Optional[Dict[str, object]] = None

        if drop_train_corrupt_bytes:
            (
                corrupt_byte_excluded,
                corrupt_byte_stats,
            ) = _plan_train_corruption_cleaning_bytes(
                self._raw_buffer,
                n_train,
            )

        degenerate_run_excluded: List[Tuple[int, int]] = []
        degenerate_run_stats: Optional[Dict[str, object]] = None

        if drop_train_degenerate_runs:
            (
                degenerate_run_excluded,
                degenerate_run_stats,
            ) = _plan_train_degenerate_run_cleaning(
                self._raw_buffer,
                n_train,
                _AUDIT_DEGENERATE_RUN_MIN_BYTES,
                corrupt_byte_excluded,
            )

        degenerate_excluded: List[Tuple[int, int]] = []
        degenerate_stats: Optional[Dict[str, object]] = None

        if drop_train_degenerate_groups:
            degenerate_excluded, degenerate_stats = (
                _plan_train_degenerate_cleaning(
                    self._raw_buffer,
                    group_separator,
                    n_train,
                    (
                        dedup_excluded
                        + shared_chunk_excluded
                        + near_duplicate_excluded
                        + corrupt_excluded
                    ),
                )
            )

        # 不同清洗项可能产生相邻区间（例如损坏字节后面紧跟
        # 退化段）；相邻区间合并是等价的排除集合，因此这里统一
        # 归并，而不是要求各规划器拼出的列表恰好首尾不相接。
        combined_excluded = _merge_byte_intervals(
            sorted(
                dedup_excluded
                + shared_chunk_excluded
                + near_duplicate_excluded
                + corrupt_excluded
                + corrupt_byte_excluded
                + degenerate_run_excluded
                + degenerate_excluded
            )
        )
        merged_excluded = combined_excluded
        if any(
            start < 0 or end > n_train or end <= start
            for start, end in merged_excluded
        ):
            raise AssertionError(
                "训练集清洗区间越出 train split："
                "内部错误，拒绝继续"
            )

        if (
            dedup_stats is not None
            or shared_chunk_stats is not None
            or near_duplicate_stats is not None
            or corrupt_stats is not None
            or degenerate_stats is not None
            or corrupt_byte_stats is not None
            or degenerate_run_stats is not None
        ):
            dropped_bytes = sum(
                end - start
                for start, end in merged_excluded
            )
            group_stats = [
                stats
                for stats in (
                    dedup_stats,
                    shared_chunk_stats,
                    near_duplicate_stats,
                    corrupt_stats,
                    degenerate_stats,
                )
                if stats is not None
            ]

            # 群组级统计的基座沿用第一种清洗的明细（例如
            # 去重的 dropped_duplicate_of、按原因拆分），
            # 再追加本次组合后的总量与字节级清洗。
            combined_stats: Dict[str, object] = (
                dict(group_stats[0]) if group_stats else {}
            )
            combined_stats.update({
                "enabled": True,
                "cleaning_kinds": [],
                "dropped_bytes": dropped_bytes,
                "post_clean_train_bytes": (
                    n_train - dropped_bytes
                ),
                # 清洗对照了哪些 split：test 出现在这里意味着
                # 训练集内容选择读取过测试集，违反默认尘封协议；
                # 只有显式开关才会出现。
                "cleaning_eval_splits": sorted(
                    cleaning_eval_ranges
                ),
                "test_used_for_cleaning": bool(
                    self.clean_train_against_test
                ),
            })

            if group_stats:
                total_groups = int(
                    group_stats[0]["groups_total"]
                )
                groups_dropped = sum(
                    int(stats["groups_dropped"])
                    for stats in group_stats
                )
                combined_stats["groups_total"] = total_groups
                combined_stats["groups_dropped"] = (
                    groups_dropped
                )
                combined_stats["groups_kept"] = (
                    total_groups - groups_dropped
                )

            if shared_chunk_stats is not None:
                combined_stats["shared_chunks"] = (
                    shared_chunk_stats
                )
            if near_duplicate_stats is not None:
                combined_stats["near_duplicates"] = (
                    near_duplicate_stats
                )
            if corrupt_stats is not None:
                combined_stats["corruption"] = corrupt_stats
            if corrupt_byte_stats is not None:
                combined_stats["corruption_bytes"] = (
                    corrupt_byte_stats
                )
            if degenerate_stats is not None:
                combined_stats["degenerate"] = degenerate_stats
            if degenerate_run_stats is not None:
                combined_stats["degenerate_runs"] = (
                    degenerate_run_stats
                )

            cleaning_kinds: List[str] = []
            if dedup_stats is not None:
                cleaning_kinds.append("duplicate")
            if shared_chunk_stats is not None:
                cleaning_kinds.append("shared_chunks")
            if near_duplicate_stats is not None:
                cleaning_kinds.append("near_duplicate")
            if corrupt_stats is not None:
                cleaning_kinds.append("corrupt")
            if degenerate_stats is not None:
                cleaning_kinds.append("degenerate")
            if corrupt_byte_stats is not None:
                cleaning_kinds.append("corrupt_bytes")
            if degenerate_run_stats is not None:
                cleaning_kinds.append("degenerate_runs")
            combined_stats["cleaning_kinds"] = cleaning_kinds

            policy_parts: List[str] = []
            if group_stats:
                policy_parts.append(
                    "drop train groups by kind(s): "
                    + ", ".join(
                        kind
                        for kind in cleaning_kinds
                        if kind != "corrupt_bytes"
                    )
                )
            if corrupt_byte_stats is not None:
                policy_parts.append(
                    "drop train bytes with invalid UTF-8 or NUL"
                )
            if degenerate_run_stats is not None:
                policy_parts.append(
                    "drop train byte runs of one value with "
                    "length >= "
                    f"{_AUDIT_DEGENERATE_RUN_MIN_BYTES}"
                )
            combined_stats["policy"] = (
                "; ".join(policy_parts)
                + ("; " if policy_parts else "")
                + "val/test untouched"
            )
            self.train_excluded_spans = merged_excluded
            self.train_cleaning = combined_stats

        self.train = tokens[:n_train]
        self.val = tokens[n_train:n_val_end]
        self.test = tokens[n_val_end:]
        self.seq_len = seq_len

        # 清洗的删除偏差随 train_cleaning 一起落盘：被删除与保留
        # 内容若分布不同，清洗后的训练分布就是选择性改变的，
        # 后续指标不能与未清洗运行直接比较。只在确有删除时计算，
        # 避免没有清洗的运行承担额外扫描成本。
        if (
            self.train_cleaning is not None
            and self.train_excluded_spans
        ):
            self.train_cleaning["deletion_bias"] = (
                _deletion_bias_record(
                    self._raw,
                    n_train,
                    self.train_excluded_spans,
                )
            )
            if self.train_cleaning["deletion_bias"] is None:
                # 删除区间非空但无法形成有效统计：显式记录，
                # 不能静默省略字段让读者以为没有删除偏差。
                self.train_cleaning["deletion_bias"] = {
                    "schema": "smelt-isr/deletion-bias/v1",
                    "dropped_bytes": sum(
                        end - start
                        for start, end in _merge_byte_intervals(
                            self.train_excluded_spans
                        )
                    ),
                    "kept_bytes": 0,
                    "train_bytes": int(n_train),
                    "byte_js": None,
                    "bigram_js": None,
                    "note": "没有可用于比较的保留内容",
                }

        # 清洗后的有效训练集指纹：resume/对比不能只看整体数据
        # 指纹与清洗开关，必须确认剔除区间及保留内容完全一致。
        self.effective_train_sha256 = (
            _effective_train_fingerprint(
                self._raw,
                n_train,
                self.train_excluded_spans,
            )
        )

        # 窗口暂存槽按调用方的 device 决定是否启用。
        self._stage_slots = None
        self._stage_index = 0

        # 每个 split 至少需要一个完整窗口，否则采样/验证
        # 没有合法起点（fixed_eval_batch 依赖 num_starts >= 1，
        # evaluate 依赖至少一个评估 token 作分母）。
        if len(self.train) < seq_len + 1:
            raise ValueError(
                "训练集长度不足（检查 val/test fraction 或语料大小）"
            )

        if len(self.val) < seq_len + 1:
            raise ValueError(
                "验证集长度不足（检查 val_fraction 或语料大小）"
            )

        if test_fraction > 0.0 and len(self.test) < seq_len + 1:
            raise ValueError(
                "测试集长度不足（检查 test_fraction 或语料大小）"
            )

        if self.train_excluded_spans:
            total_starts = len(self.train) - seq_len
            valid_intervals = _valid_window_start_intervals(
                total_starts,
                seq_len,
                self.train_excluded_spans,
            )
            if not valid_intervals:
                raise ValueError(
                    "训练集清洗后没有剩余任何合法训练窗口："
                    "清洗比例过高（几乎全部训练内容与 val/test "
                    "重复、损坏或退化），请检查语料/清洗开关，"
                    "而不是在清洗后继续训练"
                )

            self._train_start_intervals = valid_intervals
            lengths = [
                high - low + 1
                for low, high in valid_intervals
            ]
            self._train_interval_lows = torch.tensor(
                [low for low, _ in valid_intervals],
                dtype=torch.long,
            )
            self._train_interval_lengths = torch.tensor(
                lengths,
                dtype=torch.long,
            )
            self._train_interval_cumulative = torch.cumsum(
                self._train_interval_lengths,
                dim=0,
            )
            self.train_num_valid_starts = int(
                self._train_interval_cumulative[-1]
            )
            self.train_cleaning["valid_window_starts"] = (
                self.train_num_valid_starts
            )
            self.train_cleaning["total_window_starts"] = (
                total_starts
            )

        # unfold 产生 [num_windows, seq_len + 1] 的滑动窗口
        # 视图（stride 为 1），采样时按起点 index_select。
        # 窗口只在各自 split 内部滑动，不跨越 split 边界。
        self._windows = {
            "train": self.train.unfold(
                0,
                seq_len + 1,
                1,
            ),
            "val": self.val.unfold(
                0,
                seq_len + 1,
                1,
            ),
        }
        if len(self.test) >= seq_len + 1:
            self._windows["test"] = self.test.unfold(
                0,
                seq_len + 1,
                1,
            )

        # 固定评估起点按 split 分开保存。
        self._fixed_starts: Dict[str, Dict[str, object]] = {}

        # 群组（文档）起始位置缓存：评估不确定性的 cluster
        # bootstrap 需要把同一文档的 token 当成一个抽样单位，
        # 而不是把窗口当独立样本。只在支持 --group-separator
        # 的数据上可用；没有群组边界时保持 None，不伪造簇。
        self._split_group_starts: Dict[
            str, Optional[List[int]]
        ] = {}

        # 固定评估批的结果（设备侧）：键为
        # (split, batches, batch_size, batch_index, device)，
        # 值为 (x, y)；累计字节受 _EVAL_BATCH_CACHE_MAX_BYTES
        # 限制。
        self._fixed_batch_cache: Dict[
            tuple, Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._fixed_batch_cache_bytes = 0

    @property
    def has_test(self) -> bool:
        """
        是否存在可用的测试 split（test_fraction > 0 且长度足够）。
        """
        return "test" in self._windows

    @property
    def split_fractions(self) -> Dict[str, float]:
        """
        各 split 的实际字节占比（吸附到群组边界后的真实值）。
        """
        total = max(1, self.num_bytes)
        return {
            split: (end - begin) / total
            for split, (begin, end) in self.split_bounds.items()
        }

    @property
    def effective_train_bytes(self) -> int:
        """
        实际参与训练统计的 train 字节数（扣除被清洗内容）。

        神经网络窗口采样已由 ``train_excluded_spans`` 屏蔽；任何
        在 train 上拟合统计量的基线也必须使用同一口径，否则
        被清洗的重复群组/损坏字节仍会进入 unigram / n-gram 计数。
        """
        excluded = sum(
            end - start
            for start, end in _merge_byte_intervals(
                self.train_excluded_spans
            )
        )
        return len(self.train) - excluded

    def effective_train_byte_counts(self) -> torch.Tensor:
        """
        统计实际参与训练的 train 字节频次（256 个字节类）。

        被 ``--deduplicate-train-groups`` 或
        ``--drop-train-corrupt-bytes`` 剔除的区间不计入；返回
        int64 张量，供 unigram 参考基线与数据审计的分布偏移
        统计使用，保证它们与神经网络看到的训练数据一致。
        """
        try:
            counts = torch.bincount(
                self.train,
                minlength=256,
            )
        except RuntimeError:
            counts = torch.bincount(
                self.train.to(torch.int64),
                minlength=256,
            )

        for start, end in _merge_byte_intervals(
            self.train_excluded_spans
        ):
            try:
                dropped = torch.bincount(
                    self.train[start:end],
                    minlength=256,
                )
            except RuntimeError:
                dropped = torch.bincount(
                    self.train[start:end].to(torch.int64),
                    minlength=256,
                )
            counts = counts - dropped

        if bool((counts < 0).any()):
            raise AssertionError(
                "训练集清洗区间统计为负：内部错误，拒绝继续"
            )

        return counts

    def _gather_windows(
        self,
        windows: torch.Tensor,
        starts: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        按窗口起点从 [num_windows, seq_len + 1] 的窗口视图收集字节，
        并搬运到目标设备（尚未转整数索引）。

        CUDA 下经暂存槽搬运，CPU 下走普通索引路径。
        """
        num_rows = starts.numel()

        if device.type != "cuda" or not torch.cuda.is_available():
            return torch.index_select(
                windows,
                0,
                starts,
            ).to(device)

        if self._stage_slots is None:
            # 每槽: [缓冲区或 None, 拷贝完成 Event 或 None]
            self._stage_slots = [[None, None], [None, None]]

        slot = self._stage_slots[self._stage_index]
        self._stage_index ^= 1

        # 使用槽位前确认上一次拷贝已完成。
        event = slot[1]
        if event is None:
            event = torch.cuda.Event()
            slot[1] = event
        else:
            event.synchronize()

        numel = num_rows * (self.seq_len + 1)
        buffer = slot[0]
        if buffer is None or buffer.numel() < numel:
            with torch.inference_mode(False):
                buffer = torch.empty(
                    numel,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
            slot[0] = buffer

        staged = buffer[:numel].view(
            num_rows,
            self.seq_len + 1,
        )
        torch.index_select(windows, 0, starts, out=staged)

        chunks = staged.to(device, non_blocking=True)

        # 记录该槽位的完成时点。
        with torch.cuda.device(device):
            event.record()

        return chunks

    def _sample_train_starts(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        采样训练窗口起点；启用训练集群组清洗时只在合法起点
        区间内均匀采样，窗口不会触碰任何被剔除的群组内容。

        无清洗时走原路径（整个训练 split 均匀采样），保持
        随机流与历史运行逐位一致。
        """
        total_starts = len(self.train) - self.seq_len

        if self._train_start_intervals is None:
            return torch.randint(
                low=0,
                high=total_starts,
                size=(batch_size,),
                generator=generator,
            )

        assert self._train_interval_lows is not None
        assert self._train_interval_lengths is not None
        assert self._train_interval_cumulative is not None
        assert self.train_num_valid_starts is not None

        # 在每个合法区间内按区间长度均匀采样：先按累计长度
        # 抽全局序号，再映射回区间与区间内偏移。
        global_index = torch.randint(
            low=0,
            high=self.train_num_valid_starts,
            size=(batch_size,),
            generator=generator,
        )
        interval = torch.searchsorted(
            self._train_interval_cumulative,
            global_index,
            right=True,
        )
        offsets = global_index - (
            self._train_interval_cumulative[interval]
            - self._train_interval_lengths[interval]
        )
        return self._train_interval_lows[interval] + offsets

    def batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            windows = self._windows["train"]
            data = self.train
        elif split == "val":
            windows = self._windows["val"]
            data = self.val
        elif split == "test":
            # 训练期间的随机批次绝不能碰测试集，
            # 否则测试集会在无感知的情况下被反复消费。
            raise ValueError(
                "测试集在最终评估前必须尘封，"
                "不能通过 batch() 访问"
            )
        else:
            raise ValueError(f"未知 split: {split}")

        if split == "train":
            starts = self._sample_train_starts(
                batch_size,
                generator,
            )
        else:
            max_start = len(data) - self.seq_len
            starts = torch.randint(
                low=0,
                high=max_start,
                size=(batch_size,),
                generator=generator,
            )

        # 窗口先以 uint8 搬运：输入 embedding 只接受 int32/int64，
        # 在目标设备上转 int32；目标 token 保持 uint8（cross_entropy
        # 直接支持 Byte 目标）。
        chunks = self._gather_windows(
            windows,
            starts,
            device,
        )

        x = chunks[:, :-1].to(torch.int32)
        y = chunks[:, 1:]

        return x, y

    def _eval_windows(
        self,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        取出指定评估 split 的窗口视图与底层 token 张量。

        batch() 只允许 train/val；test 只能经此处的固定窗口评估
        访问。
        """
        if split not in ("val", "test"):
            raise ValueError(f"未知评估 split: {split}")

        if split == "test" and not self.has_test:
            raise ValueError(
                "测试集不可用（test_fraction 为 0 或语料不足）"
            )

        windows = self._windows[split]
        data = self.val if split == "val" else self.test
        return windows, data

    def _ensure_fixed_starts(
        self,
        split: str,
        total_batches: int,
        num_starts: int,
    ) -> torch.Tensor:
        """生成（或复用）固定评估窗口起点。"""
        cache = self._fixed_starts.setdefault(
            split,
            {"total": None, "num_starts": None, "starts": None},
        )
        try:
            requested = int(total_batches)
        except (TypeError, ValueError, OverflowError):
            requested = 0
        try:
            available = int(num_starts)
        except (TypeError, ValueError, OverflowError):
            available = 0
        total_batches = max(0, min(requested, available))

        if (
            cache["total"] != total_batches
            or cache["num_starts"] != num_starts
        ):
            # 均匀覆盖整个评估流。total_batches <= num_starts，
            # linspace 的间距不小于 1，round 后不会重复。
            cache["starts"] = torch.linspace(
                0,
                num_starts - 1,
                steps=total_batches,
                dtype=torch.float64,
            ).round().long()
            cache["total"] = total_batches
            cache["num_starts"] = num_starts
        starts = cache["starts"]
        assert isinstance(starts, torch.Tensor)
        return starts

    def split_group_starts(
        self,
        split: str,
    ) -> Optional[List[int]]:
        """
        评估 split 内各群组（文档）的起始字节位置（相对 split
        起点）；没有 --group-separator 或 split 为空时返回 None。

        群组按 ``_iter_group_spans`` 的口径定义：分隔符之间的
        非空连续片段。分隔符本身不属于任何群组，但在下面的
        ``group_ids_for_starts`` 中并入前一个群组，保证每个
        目标 token 都能归属到一个文档簇；前导分隔符并入第一个
        群组。没有群组边界时不做任何近似（返回 None），避免
        把窗口当作文档簇写入报告。
        """
        if self.group_separator is None:
            return None
        if split not in self.split_bounds:
            raise ValueError(f"未知 split: {split}")
        if split in self._split_group_starts:
            return self._split_group_starts[split]

        begin, end = self.split_bounds[split]
        starts = sorted(
            {
                span_start - begin
                for span_start, _span_end in _iter_group_spans(
                    self._raw_buffer,
                    self.group_separator,
                    begin,
                    end,
                )
            }
        )
        result: Optional[List[int]]
        result = starts if starts else None
        self._split_group_starts[split] = result
        return result

    def group_assignment_fingerprint(self) -> Optional[str]:
        """群组定义指纹；没有群组分隔符时为 None。"""
        if self.group_separator is None:
            return None
        return hashlib.sha256(self.group_separator).hexdigest()

    def group_ids_for_starts(
        self,
        split: str,
        starts: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        固定评估窗口内每个**目标 token**（x 右移一位）所属的
        群组下标，形状 [n_windows, seq_len]，int64 CPU 张量。

        群组下标是 split 内部的 0 基序号，与
        ``split_group_starts`` 的顺序一致。位置 p 归属
        ``max{i: group_start_i <= p}``；因此分隔符与空群组
        并入前一个群组，前导分隔符并入第一个群组。没有群组
        边界时返回 None。
        """
        group_starts = self.split_group_starts(split)
        if group_starts is None:
            return None
        start_tensor = torch.tensor(
            group_starts,
            dtype=torch.long,
        )
        window_starts = torch.as_tensor(
            starts,
            dtype=torch.long,
        ).reshape(-1)
        if window_starts.numel() == 0:
            return torch.empty(
                (0, self.seq_len),
                dtype=torch.long,
            )
        offsets = torch.arange(
            1,
            self.seq_len + 1,
            dtype=torch.long,
        )
        positions = (
            window_starts.unsqueeze(1) + offsets.unsqueeze(0)
        )
        # right=True 让恰好落在群组起点的位置归入新群组；
        # clamp 处理前导分隔符（下标 -1）与空 split。
        ids = (
            torch.searchsorted(
                start_tensor,
                positions.reshape(-1),
                right=True,
            )
            - 1
        ).clamp_(min=0)
        return ids.view(positions.shape)

    def fixed_eval_group_ids(
        self,
        split: str,
        batches: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """全部固定评估窗口的目标 token 群组下标；不可用时 None。"""
        windows, data = self._eval_windows(split)
        _ = windows
        num_starts = len(data) - self.seq_len
        try:
            requested = int(batches * batch_size)
        except (TypeError, ValueError, OverflowError):
            return None
        total_batches = max(0, min(requested, num_starts))
        starts = self._ensure_fixed_starts(
            split,
            total_batches,
            num_starts,
        )
        return self.group_ids_for_starts(split, starts)

    def fixed_eval_batch(
        self,
        split: str,
        batch_index: int,
        batches: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回跨指定评估 split（val/test）均匀分布的固定窗口。

        不用随机数，因此评估估计不随运行进程、CUDA 状态或
        eval_batches 的调用顺序而变化；不同实验看到的窗口集合
        完全相同。请求窗口多于合法起点时仅取一遍无重复起点。

        固定窗口在多次评估间逐位相同，结果张量在设备侧保存；
        总量受 _EVAL_BATCH_CACHE_MAX_BYTES 限制。所有调用方只读
        返回的 x/y。
        """
        windows, data = self._eval_windows(split)

        # 合法起点共 len(data) - seq_len 个。
        num_starts = len(data) - self.seq_len
        total_batches = min(
            batches * batch_size,
            num_starts,
        )

        starts_all = self._ensure_fixed_starts(
            split,
            total_batches,
            num_starts,
        )

        cache_key = (
            split,
            batches,
            batch_size,
            batch_index,
            str(device),
        )
        cached = self._fixed_batch_cache.get(cache_key)
        if cached is not None:
            return cached

        begin = batch_index * batch_size
        # starts 为 CPU 张量，窗口收集与搬运与训练批一致。
        starts = starts_all[begin:begin + batch_size]

        chunks = self._gather_windows(
            windows,
            starts,
            device,
        )

        x = chunks[:, :-1].to(torch.int32)
        y = chunks[:, 1:]

        batch_bytes = (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
        )
        if (
            self._fixed_batch_cache_bytes + batch_bytes
            <= _EVAL_BATCH_CACHE_MAX_BYTES
        ):
            self._fixed_batch_cache[cache_key] = (x, y)
            self._fixed_batch_cache_bytes += batch_bytes

        return x, y

    def clear_eval_batch_cache(self) -> None:
        """
        释放固定评估批的设备侧结果。

        评估协议结束后（最终/测试评估完成）显式清空；评估起点
        （_fixed_starts）只是 CPU 索引，保留。
        """
        self._fixed_batch_cache.clear()
        self._fixed_batch_cache_bytes = 0

    def fixed_val_batch(
        self,
        batch_index: int,
        batches: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fixed_eval_batch("val", ...) 的兼容包装。"""
        return self.fixed_eval_batch(
            "val",
            batch_index,
            batches,
            batch_size,
            device,
        )



# ============================================================
# Pre-training data pipeline probe
# ============================================================

def _expected_window_pairs(
    data: torch.Tensor,
    starts: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按窗口起点从原始 split 字节张量重建 (x, y)。

    x 是窗口去除最后一个字节的输入，y 是窗口去除第一个字节的
    目标（下一 token 监督位）。这是数据协议对 (x, y) 的定义，
    独立于 ByteDataset 的 unfold/index_select 实现，用于核对
    两者的逐位一致性。
    """
    positions = starts.to(torch.long).unsqueeze(1) + torch.arange(
        seq_len + 1,
        dtype=torch.long,
    )
    windows = data[positions]
    return windows[:, :-1].to(torch.int32), windows[:, 1:]
