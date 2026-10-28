from __future__ import annotations



def audit_data_splits(
    dataset: ByteDataset,
    include_test: bool = False,
    max_windows: int = _AUDIT_MAX_WINDOWS,
    cache_scans: bool = False,
) -> Dict[str, object]:
    """
    数据审计：值域校验、文本完整性、分布偏移与重复窗口检测。

    - 校验每个 split 的 token 值域（字节管线为 0..255；整数
      token 中不存在 NaN/Inf，值域校验用于捕获上游 dtype 或
      编码错误）；
    - 扫描文本完整性（无法 UTF-8 解码的字节序列、NUL 字节），
      作为字节级标签噪声与语料损坏的清洗检查；
    - 统计每个 split 的字节直方图与字节二元组直方图，并计算
      train↔val（可选包含 train↔test、val↔test）之间的 JS
      距离：单字节频率发现词表/频率偏移，二元组发现「词表
      相同但上下文顺序改变」的偏移；
    - 对滑动窗口做有界签名扫描，检测跨 split 重复内容与 split
      内部重复；
    - 用内容定义分块指纹补齐窗口抽样漏掉的错位/近似重复：分块
      边界由局部内容决定，长片段整体偏移后仍会产生相同块指纹；
      跨 split 共享块是泄露级问题；
    - 群组划分（传入 group_separator 时）报告每个 split 的完整
      群组数与分隔符，供追踪「群组整体不跨界」的协议。

    ``leakage_scan_coverage`` 记录窗口签名扫描的覆盖（各 split
    采样窗口数/总窗口数与 stride）。stride>1 时窗口级「未发现
    跨 split 重复」只是抽样范围内的负结果，报告会显式写出残余
    风险，避免把抽样零发现读成「数据里没有泄露」。

    返回的 ``warnings`` 是全部提醒；``violations`` 是其中会
    被 ``--strict-data-audit`` 视为致命的问题（越界 token、
    跨 split 重复内容、任何 UTF-8/NUL 损坏）——这些问题直接
    对应数据泄露或标签噪声，严格模式下拒绝训练。
    ``leakage_violations`` 是 ``violations`` 中「跨 split 内容
    重复」这一类纯泄露问题：默认训练入口也会拒绝它们，除非
    显式传入 ``--allow-data-leakage``。损坏/标签噪声仍由
    ``--strict-data-audit`` 控制，避免把「数据脏」和「数据泄露」
    混为同一个开关。

    test 只在 include_test=True（最终评估解封时）才参与统计，
    训练期间默认只审计 train/val。审计只做诊断，不参与任何
    特征计算或模型决策。
    """
    split_names = ["train", "val"]
    if include_test and dataset.has_test:
        split_names.append("test")

    histograms: Dict[str, torch.Tensor] = {}
    scans: Dict[str, Dict[str, object]] = {}
    splits_report: Dict[str, object] = {}
    warnings: List[str] = []
    violations: List[str] = []
    leakage_violations: List[str] = []
    evaluation_violations: List[str] = []

    # 无群组语料：一次性按整个文件解码，得到每个非法/NUL 字节
    # 的绝对位置，再按 split 归属计数。逐 split 预热解码会把
    # 上一段末尾的未完成序列记到下一段（例如 val 末尾的坏字节
    # 让字节干净的 test 被标签完整性门拒绝）；全局定位既保留
    # 跨 split 合法多字节字符，又能把坏字节归到真正包含它的段。
    # 群组模式仍逐群组扫描，以排除用户声明的分隔符字节。
    global_integrity_spans: Optional[
        List[Tuple[int, int, str]]
    ] = None
    if dataset.group_separator is None:
        global_integrity_spans = _invalid_utf8_nul_spans(
            dataset._raw,
            0,
            dataset.num_bytes,
            final=True,
        )

    for split in split_names:
        tokens = getattr(dataset, split)

        # 字节管线固定为 uint8，直方图直接在原始 dtype 上
        # 统计（uint8 天然落在 0..255，越界恒为 0）。其他
        # dtype（若日后扩展管线）回退到 int64 校验路径，
        # 保持诊断语义不变。
        if split == "train" and dataset.train_excluded_spans:
            # 训练集群组清洗后的有效字节：被剔除的重复群组
            # 不会进入神经网络训练，也不应进入分布偏移统计，
            # 否则审计描述的是模型从未见过的数据。
            hist = dataset.effective_train_byte_counts().to(
                torch.float64
            )
            invalid = 0
        elif tokens.dtype == torch.uint8:
            try:
                hist = torch.bincount(
                    tokens,
                    minlength=256,
                )
            except RuntimeError:
                # 旧版本 bincount 仅接受 int64 输入，退回转换。
                hist = torch.bincount(
                    tokens.to(torch.int64),
                    minlength=256,
                )
            hist = hist.to(torch.float64)
            invalid = 0
        else:
            tokens_long = tokens.to(torch.int64)
            # 先统计越界，再对 clamp 后的值做直方图：
            # bincount 不接受负值，且 clamp 保证直方图长度
            # 恰为 256，跨 split 的 JS 距离形状一致。
            invalid = int(
                ((tokens_long < 0) | (tokens_long > 255)).sum()
            )
            hist = torch.bincount(
                tokens_long.clamp(0, 255),
                minlength=256,
            ).to(torch.float64)

        histograms[split] = hist / hist.sum()

        top_share, top_byte = torch.topk(
            histograms[split],
            k=5,
        )

        begin, end = dataset.split_bounds[split]
        # 训练前审计与解封测试集时的审计都会扫描 train/val；
        # 数据在运行期间不可变，按 (起点, 长度, 窗口, 上限)
        # 复用扫描结果。
        scan_cache = getattr(
            dataset, "_audit_scan_cache", None
        )
        scan_key = (
            begin,
            end - begin,
            dataset.seq_len + 1,
            max_windows,
        )
        scan = (
            scan_cache.get(scan_key)
            if scan_cache is not None
            else None
        )
        if scan is None:
            scan = _window_signature_scan(
                dataset._raw,
                begin,
                end - begin,
                dataset.seq_len + 1,
                max_windows,
                exclude_spans=(
                    dataset.train_excluded_spans
                    if split == "train"
                    else None
                ),
            )
            if cache_scans:
                if scan_cache is None:
                    scan_cache = {}
                    dataset._audit_scan_cache = scan_cache
                scan_cache[scan_key] = scan
        scans[split] = scan
        if dataset.group_separator is None:
            # 用全局字节位置归属到 split；train 的清洗区间不再
            # 计入（模型不会看到这些字节），val/test 不做清洗。
            integrity = {
                "utf8_replacements": 0,
                "nul_bytes": 0,
            }
            excluded = _merge_byte_intervals(
                dataset.train_excluded_spans
                if split == "train"
                else []
            )
            excluded_index = 0
            for span_start, _span_end, kind in (
                global_integrity_spans or []
            ):
                if span_start < begin:
                    continue
                if span_start >= end:
                    break
                while (
                    excluded_index < len(excluded)
                    and excluded[excluded_index][1]
                    <= span_start
                ):
                    excluded_index += 1
                if (
                    excluded_index < len(excluded)
                    and excluded[excluded_index][0]
                    <= span_start
                    < excluded[excluded_index][1]
                ):
                    continue
                if kind == "invalid_utf8":
                    integrity["utf8_replacements"] += 1
                elif kind == "nul":
                    integrity["nul_bytes"] += 1
        else:
            # 群组分隔符是用户显式声明的数据结构（可能含 NUL 或
            # 非 UTF-8 字节，如 --group-separator '\x00'），不能
            # 计入损坏统计，否则严格审计会把合法配置当成语料
            # 损坏而拒绝训练。逐群组扫描：每个群组按独立文本
            # 解码，组尾不完整的多字节序列计入该组。
            integrity = {
                "utf8_replacements": 0,
                "nul_bytes": 0,
            }
            train_excluded = (
                set(dataset.train_excluded_spans)
                if split == "train"
                else set()
            )
            for span_start, span_end in _iter_group_spans(
                dataset._raw_buffer,
                dataset.group_separator,
                begin,
                end,
            ):
                if (span_start, span_end) in train_excluded:
                    continue
                part = _scan_text_integrity(
                    dataset._raw,
                    span_start,
                    span_end - span_start,
                    final=True,
                    warmup=False,
                )
                integrity["utf8_replacements"] += part[
                    "utf8_replacements"
                ]
                integrity["nul_bytes"] += part["nul_bytes"]

        nominal_fraction = {
            "train": (
                1.0
                - dataset.val_fraction
                - dataset.test_fraction
            ),
            "val": dataset.val_fraction,
            "test": dataset.test_fraction,
        }[split]
        actual_fraction = dataset.split_fractions[split]

        splits_report[split] = {
            "bytes": end - begin,
            "effective_bytes": (
                dataset.effective_train_bytes
                if split == "train"
                else end - begin
            ),
            "fraction": actual_fraction,
            "nominal_fraction": nominal_fraction,
            "windows": scans[split]["total"],
            "effective_windows": (
                dataset.train_num_valid_starts
                if (
                    split == "train"
                    and dataset.train_excluded_spans
                )
                else scans[split]["total"]
            ),
            "invalid_tokens": invalid,
            "utf8_replacements": integrity["utf8_replacements"],
            "nul_bytes": integrity["nul_bytes"],
            "top_bytes": [
                [int(byte), round(float(share), 6)]
                for share, byte in zip(
                    top_share.tolist(),
                    top_byte.tolist(),
                )
            ],
        }

        if invalid > 0:
            message = f"{split}: 检测到 {invalid} 个越界 token"
            warnings.append(message)
            violations.append(message)

        split_bytes = max(1, end - begin)
        replacement_share = (
            integrity["utf8_replacements"] / split_bytes
        )
        if integrity["utf8_replacements"] > 0:
            message = (
                f"{split}: 检测到 "
                f"{integrity['utf8_replacements']} 处无法按"
                "UTF-8 解码的字节序列"
                f"（占比 {replacement_share:.4f}），"
                "疑似语料损坏（字节级标签噪声）"
            )
            # 严格模式对任何标签噪声都必须拒绝：低频损坏同样会
            # 污染监督位。默认模式只在占比超过告警阈值时打印
            # 警告，但严格判定不依赖该阈值。
            if replacement_share > _AUDIT_UTF8_REPLACE_WARN:
                warnings.append(message)
            violations.append(message)

        nul_share = integrity["nul_bytes"] / split_bytes
        if integrity["nul_bytes"] > 0:
            message = (
                f"{split}: 检测到 {integrity['nul_bytes']} 个"
                f"NUL 字节（占比 {nul_share:.4f}），"
                "疑似二进制/填充污染"
            )
            if nul_share > _AUDIT_NUL_WARN:
                warnings.append(message)
            violations.append(message)

        top_byte_share = float(top_share[0])
        if top_byte_share > _AUDIT_TOP_BYTE_WARN:
            warnings.append(
                f"{split}: 最高频字节占比 "
                f"{top_byte_share:.3f}，疑似填充/退化数据"
            )

        if nominal_fraction > 0.0 and abs(
            actual_fraction - nominal_fraction
        ) > max(
            _AUDIT_SPLIT_FRACTION_MIN_ABS,
            _AUDIT_SPLIT_FRACTION_DRIFT * nominal_fraction,
        ):
            warnings.append(
                f"{split}: 实际占比 {actual_fraction:.4f} "
                f"偏离名义值 {nominal_fraction:.4f}"
                "（群组边界吸附或语料大小导致），"
                "评估集规模/统计功效可能与预期不符"
            )

    # 评估集标签覆盖：训练集从未出现过的字节类在评估 token 中
    # 占比多少。字节级 LM 只能通过上下文降低已见标签的 NLL；
    # 未见标签的最优预测只能退到平滑/先验，属于不可约的外推
    # 误差。把它与分布偏移分开报告，避免把「模型没学会」和
    # 「训练数据根本没有这个标签」混为一谈。
    label_coverage: Dict[str, object] = {}
    train_hist = histograms.get("train")
    if train_hist is not None:
        train_present = train_hist > 0
        train_classes = int(train_present.sum())
        label_coverage["train"] = {
            "observed_byte_classes": train_classes,
        }
        for split in split_names:
            if split == "train":
                continue
            split_hist = histograms[split]
            unseen_mask = (split_hist > 0) & (~train_present)
            unseen_classes = int(unseen_mask.sum())
            unseen_share = float(
                split_hist[unseen_mask].sum()
            )
            info = {
                "observed_byte_classes": int(
                    (split_hist > 0).sum()
                ),
                "train_observed_byte_classes": train_classes,
                "unseen_label_classes": unseen_classes,
                "unseen_label_token_share": round(
                    unseen_share, 8
                ),
                "warn_token_share": (
                    _AUDIT_UNSEEN_LABEL_WARN
                ),
            }
            label_coverage[split] = info
            split_info = splits_report.get(split)
            if isinstance(split_info, dict):
                split_info["unseen_label_classes"] = (
                    unseen_classes
                )
                split_info["unseen_label_token_share"] = (
                    info["unseen_label_token_share"]
                )
            if unseen_share > _AUDIT_UNSEEN_LABEL_WARN:
                warnings.append(
                    f"{split}: 评估集有 {unseen_classes} 个"
                    "字节类在有效训练集中从未出现（占评估 "
                    f"token {unseen_share:.4%}）；这些标签的"
                    "NLL 无法通过训练降低，指标包含不可约的"
                    "外推误差，不能与训练集覆盖相同的运行"
                    "直接比较"
                )

    # 清洗的删除偏差：清洗按内容规则选择性删除，删除后的训练
    # 分布可能与原始分布不同。这里不重复打印所有统计，只在
    # 偏差达到告警阈值时提示；完整数值在
    # data.train_cleaning.deletion_bias。
    cleaning_record = dataset.train_cleaning
    if isinstance(cleaning_record, dict):
        bias = cleaning_record.get("deletion_bias")
        if isinstance(bias, dict):
            drop_fraction = bias.get("drop_fraction")
            if (
                isinstance(drop_fraction, (int, float))
                and drop_fraction > _AUDIT_DELETION_DROP_WARN
            ):
                warnings.append(
                    "train: 清洗删除了 "
                    f"{float(drop_fraction):.1%} 的训练字节"
                    f"（{bias.get('dropped_bytes')}B）；"
                    "删除由重复/损坏/退化规则触发，不是随机"
                    "缺失，清洗后的训练分布已被选择性改变。"
                    "清洗/未清洗运行的指标不能直接比较"
                )
            byte_js = bias.get("byte_js")
            if (
                isinstance(byte_js, (int, float))
                and byte_js > _AUDIT_DELETION_BIAS_WARN
            ):
                warnings.append(
                    "train: 被删除与保留内容的字节分布"
                    f"JS 距离 {float(byte_js):.4f} > "
                    f"{_AUDIT_DELETION_BIAS_WARN:.2f}；"
                    "清洗并非随机删除，训练分布发生了"
                    "选择性变化，见 "
                    "data.train_cleaning.deletion_bias"
                )
            bigram_js = bias.get("bigram_js")
            if (
                isinstance(bigram_js, (int, float))
                and bigram_js
                > _AUDIT_DELETION_BIAS_WARN
            ):
                warnings.append(
                    "train: 被删除与保留内容的二元组分布"
                    f"JS 距离 {float(bigram_js):.4f} > "
                    f"{_AUDIT_DELETION_BIAS_WARN:.2f}；"
                    "删除改变了上下文转移分布，不能把清洗后"
                    "的改善/退化全部归因于方法"
                )

    # 验证集标签噪声即使占比很低也必须显式提示：val 用于
    # checkpoint 选择、早停与超参/基线调参，少量损坏标签会
    # 系统性地改变选型结果，而不像 train 噪声那样只增加方差。
    # 这里只告警不拒绝（合法二进制语料仍可训练），严格模式
    # 或 --drop-train-* 清洗负责处理。
    val_info = splits_report.get("val")
    val_label_noise: Dict[str, object] = {
        "invalid_tokens": 0,
        "utf8_replacements": 0,
        "nul_bytes": 0,
        "has_label_noise": False,
        "used_for_selection": True,
    }
    if isinstance(val_info, dict):
        for key in (
            "invalid_tokens",
            "utf8_replacements",
            "nul_bytes",
        ):
            val_label_noise[key] = int(val_info.get(key, 0) or 0)
        val_label_noise["has_label_noise"] = any(
            int(val_label_noise[key]) > 0
            for key in (
                "invalid_tokens",
                "utf8_replacements",
                "nul_bytes",
            )
        )
        if val_label_noise["has_label_noise"]:
            warnings.append(
                "val: 验证集含标签噪声（越界 token "
                f"{val_label_noise['invalid_tokens']}、"
                "非法 UTF-8 "
                f"{val_label_noise['utf8_replacements']}、"
                f"NUL {val_label_noise['nul_bytes']}）；"
                "val 用于 checkpoint/早停选型，损坏标签会"
                "扭曲模型选择。默认仅告警；请用 "
                "--strict-data-audit 拒绝训练，或先清洗/"
                "重新切分数据"
            )

    # test 在训练期间保持尘封：这里只根据切分边界核对它的
    # 实际占比，不读取任何 test 内容。这样吸附导致的测试集
    # 缩水在训练前就能被发现。
    nominal_by_split = {
        "train": (
            1.0
            - dataset.val_fraction
            - dataset.test_fraction
        ),
        "val": dataset.val_fraction,
        "test": dataset.test_fraction,
    }
    for split, actual_fraction in (
        dataset.split_fractions.items()
    ):
        if split in splits_report:
            continue
        nominal_fraction = nominal_by_split[split]
        if nominal_fraction > 0.0 and abs(
            actual_fraction - nominal_fraction
        ) > max(
            _AUDIT_SPLIT_FRACTION_MIN_ABS,
            _AUDIT_SPLIT_FRACTION_DRIFT * nominal_fraction,
        ):
            warnings.append(
                f"{split}: 实际占比 {actual_fraction:.4f} "
                f"偏离名义值 {nominal_fraction:.4f}"
                "（群组边界吸附或语料大小导致；"
                "test 内容仍保持尘封）"
            )

    bigram_histograms: Dict[str, torch.Tensor] = {}
    bigram_counts: Dict[str, torch.Tensor] = {}

    for split in split_names:
        counts = _byte_bigram_counts(
            getattr(dataset, split),
            (
                dataset.train_excluded_spans
                if split == "train"
                else None
            ),
        )
        bigram_counts[split] = counts
        total = int(counts.sum())
        if total > 0:
            bigram_histograms[split] = (
                counts.to(torch.float64) / total
            )

    distribution_shift: Dict[str, float] = {}
    distribution_shift_bigram: Dict[str, float] = {}
    distribution_shift_conditional: Dict[str, float] = {}
    unseen_transition_share: Dict[str, float] = {}
    seen_context_share: Dict[str, float] = {}

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        key = f"{left}_{right}"

        if left in histograms and right in histograms:
            distance = _js_distance(
                histograms[left],
                histograms[right],
            )
            distribution_shift[key] = round(distance, 6)

            if distance > _AUDIT_SHIFT_WARN:
                warnings.append(
                    f"{key}: 字节分布偏移较大"
                    f"（JS 距离 {distance:.4f}）"
                )

        if (
            left in bigram_histograms
            and right in bigram_histograms
        ):
            bigram_distance = _js_distance(
                bigram_histograms[left],
                bigram_histograms[right],
            )
            distribution_shift_bigram[key] = round(
                bigram_distance,
                6,
            )

            if bigram_distance > _AUDIT_BIGRAM_SHIFT_WARN:
                warnings.append(
                    f"{key}: 字节二元组分布偏移较大"
                    f"（JS 距离 {bigram_distance:.4f}）："
                    "单字节词表可能相同，但上下文/顺序分布"
                    "已发生变化"
                )

        if left in bigram_counts and right in bigram_counts:
            conditional = _conditional_bigram_shift(
                bigram_counts[left],
                bigram_counts[right],
            )
            distribution_shift_conditional[key] = round(
                conditional["conditional_js"],
                6,
            )
            unseen_transition_share[key] = round(
                conditional["unseen_transition_share"],
                6,
            )
            seen_context_share[key] = round(
                conditional["seen_context_share"],
                6,
            )

            if (
                conditional["conditional_js"]
                > _AUDIT_CONDITIONAL_SHIFT_WARN
            ):
                warnings.append(
                    f"{key}: 给定上下文后的标签分布偏移较大"
                    "（条件 JS 距离 "
                    f"{conditional['conditional_js']:.4f}）："
                    "同一上下文在 train 与评估 split 中"
                    "接不同标签，监督信号已漂移"
                )

            if (
                conditional["unseen_transition_share"]
                > _AUDIT_UNSEEN_TRANSITION_WARN
            ):
                warnings.append(
                    f"{key}: 评估集中 "
                    f"{conditional['unseen_transition_share']:.1%} "
                    "的 (上下文, 目标) 转移在 train 从未出现；"
                    "指标包含大量外推，需检查切分/分布偏移"
                )

    temporal_boundary_shift = _temporal_boundary_shift_audit(
        dataset,
        split_names,
        warnings,
    )

    duplicate_windows: Dict[str, object] = {}

    for split, scan in scans.items():
        duplicate_windows[split] = {
            "sampled": scan["sampled"],
            "total": scan["total"],
            "stride": scan["stride"],
            "within_duplicates": scan["within_duplicates"],
        }

        if scan["within_duplicates"] > 0:
            warnings.append(
                f"{split}: 采样窗口中有"
                f"{scan['within_duplicates']}个 split 内部重复"
                + (
                    "；固定窗口 bootstrap 会把重复窗口当作"
                    "独立抽样单位，评估区间可能偏窄"
                    if split in ("val", "test")
                    else ""
                )
            )

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        if left not in scans or right not in scans:
            continue

        left_sigs = scans[left]["signatures"]
        right_sigs = scans[right]["signatures"]
        common = left_sigs.keys() & right_sigs.keys()

        duplicate_windows[f"{left}_{right}"] = {
            "shared_signatures": len(common),
            "examples": [
                [left_sigs[digest][0], right_sigs[digest][0]]
                for digest in sorted(common)[:3]
            ],
        }

        if common:
            message = (
                f"{left}_{right}: 检测到"
                f"{len(common)} 个重复窗口内容"
                "（跨 split 泄露风险）"
            )
            warnings.append(message)
            violations.append(message)
            leakage_violations.append(message)

    # 抽样扫描的零发现不能当成「无泄露」的证明：把覆盖情况与
    # 残余风险写进审计，读者才知道这条负结果的证据边界在哪。
    scan_coverage = _leakage_scan_coverage(scans)
    if scan_coverage["residual_risk"]:
        warnings.append(str(scan_coverage["residual_risk"]))

    groups_report: Optional[Dict[str, object]] = None

    if dataset.group_separator is not None:
        # 群组划分协议：分界点都是群组起始字节，因此每个群组
        # 整体落在一个 split 内；这里把群组规模写进审计，供
        # 事后核对（数据集构造时还做了防御性断言）。
        content_duplicates = _group_content_duplicate_audit(
            dataset,
            split_names,
        )
        content_health = _train_group_health_audit(dataset)

        groups_report = {
            "separator_hex": dataset.group_separator.hex(),
            "separator_length": len(dataset.group_separator),
            "cuts": dataset.group_cuts,
            "groups_per_split": dataset.groups_per_split,
            "groups_total": (
                sum(dataset.groups_per_split.values())
                if dataset.groups_per_split is not None
                else None
            ),
            "content_duplicates": content_duplicates,
            "content_health": content_health,
            "train_cleaning": dataset.train_cleaning,
        }

        if (
            isinstance(dataset.train_cleaning, dict)
            and dataset.train_cleaning.get(
                "test_used_for_cleaning"
            )
        ):
            message = (
                "train: 训练集清洗对照了 test 内容"
                "（显式开启 --clean-train-against-test）："
                "训练样本集合依赖测试集，最终 test 指标不再"
                "来自完全尘封的数据；该事实已写入 train_cleaning"
                "并参与跨运行协议核对"
            )
            warnings.append(message)
            violations.append(message)

        if (
            content_health is not None
            and content_health["degenerate_groups"] > 0
        ):
            warnings.append(
                "train: 检测到 "
                f"{content_health['degenerate_groups']} 个"
                "低信息/退化群组（长度 >= "
                f"{content_health['min_bytes']}B 且单一字节"
                "占比 >= "
                f"{content_health['top_share_threshold']}），"
                "疑似填充/重复字符污染（异常值）；可检查后用 "
                "--drop-train-degenerate-groups 清洗"
            )

        if content_duplicates is not None:
            for split, info in content_duplicates[
                "per_split"
            ].items():
                if info["within_split_duplicates"] > 0:
                    message = (
                        f"{split}: 检测到 "
                        f"{info['within_split_duplicates']} 个"
                        "完全重复的群组（重复样本/标签噪声）"
                    )
                    warnings.append(message)
                    if split in ("val", "test"):
                        evaluation_violations.append(message)

            for pair, info in content_duplicates[
                "cross_split"
            ].items():
                if info["shared_groups"] <= 0:
                    continue
                message = (
                    f"{pair}: 检测到 "
                    f"{info['shared_groups']} 个内容完全"
                    "相同的群组（跨 split 群组泄露风险）"
                )
                warnings.append(message)
                violations.append(message)
                leakage_violations.append(message)

    degenerate_runs = _train_degenerate_run_audit(dataset)

    if (
        degenerate_runs is not None
        and degenerate_runs["runs"] > 0
    ):
        warnings.append(
            "train: 检测到 "
            f"{degenerate_runs['runs']} 个低信息/退化连续段"
            "（长度 >= "
            f"{degenerate_runs['min_bytes']}B 的单一字节段，"
            f"共 {degenerate_runs['degenerate_bytes']} 字节，"
            "疑似填充/重复字符污染/异常值）；可检查后用 "
            "--drop-train-degenerate-runs 只清洗 train"
        )

    content_chunks = _content_chunk_duplicate_audit(
        dataset,
        split_names,
    )

    for pair, info in content_chunks["cross_split"].items():
        if info["shared_chunks"] <= 0:
            continue
        message = (
            f"{pair}: 内容定义分块检测到 "
            f"{info['shared_chunks']} 个相同的"
            f"连续字节块（共享 {info['shared_bytes']} 字节，"
            "错位/近似重复内容的跨 split 泄露风险）"
        )
        warnings.append(message)
        violations.append(message)
        leakage_violations.append(message)

    # 近重复只读审计：精确群组指纹与内容定义分块都要求内容
    # 完全相同/共享完整块；改写、转载、轻微编辑的群组仍可能
    # 把评估内容带进训练。这里按 shingle Jaccard 找近重复，
    # 默认只对照 val；只有解封测试集时才把 test 纳入。
    # 近重复是相似而非逐字节相同，默认只告警、不进入泄露硬门；
    # 清洗需显式 --drop-train-near-duplicate-groups。
    near_duplicates: Optional[Dict[str, object]] = None
    evaluation_near_duplicates: Optional[Dict[str, object]] = None
    if (
        dataset.group_separator is not None
        and getattr(dataset, "near_duplicate_audit", False)
    ):
        near_eval_ranges: Dict[str, Tuple[int, int]] = {
            "val": dataset.split_bounds["val"],
        }
        near_include_test = bool(
            include_test and dataset.has_test
        )
        if near_include_test:
            near_eval_ranges["test"] = (
                dataset.split_bounds["test"]
            )

        _, near_duplicates = (
            _plan_train_near_duplicate_cleaning(
                dataset._raw_buffer,
                dataset.group_separator,
                dataset.split_bounds["train"][1],
                near_eval_ranges,
                dataset.train_excluded_spans,
                threshold=getattr(
                    dataset,
                    "near_duplicate_threshold",
                    _NEAR_DUP_DEFAULT_THRESHOLD,
                ),
                shingle_bytes=getattr(
                    dataset,
                    "near_duplicate_shingle_bytes",
                    _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
                ),
                apply_cleaning=False,
            )
        )
        near_duplicates["include_test"] = near_include_test
        near_duplicates["test_content_used"] = (
            near_include_test
        )

        flagged = int(near_duplicates["groups_flagged"])
        if flagged > 0:
            matched = ", ".join(
                f"{split}:{count}"
                for split, count in (
                    near_duplicates[
                        "flagged_duplicate_of"
                    ] or {}
                ).items()
                if int(count) > 0
            )
            warnings.append(
                "train: 近重复审计发现 "
                f"{flagged} 个训练群组与评估/更早训练群组"
                "的 shingle Jaccard 达到阈值 "
                f"{near_duplicates['threshold']}（{matched}；"
                f"shingle={near_duplicates['shingle_bytes']}B）；"
                "精确指纹/内容定义分块只能发现完全相同或"
                "共享完整块的重复，改写/转载可能仍把评估"
                "内容带入训练。检查 "
                "data_audit.near_duplicates.examples 后可用 "
                "--drop-train-near-duplicate-groups 整组清洗"
            )
        skipped = (
            int(near_duplicates["groups_too_short"])
            + int(near_duplicates["groups_too_large"])
            + int(near_duplicates["eval_groups_too_short"])
            + int(near_duplicates["eval_groups_too_large"])
        )
        if skipped > 0:
            warnings.append(
                "train/eval: 近重复审计跳过 "
                f"{skipped} 个群组（过短或超过 "
                f"{near_duplicates['max_group_bytes']}B）："
                "这些群组没有参与近重复判断，不能把"
                "「未标记」当成「不重复」"
            )

        # 评估集内部的近重复与 train→eval 是两类不同的问题：
        # 前者让同一内容在评估中被重复计数，后者才是训练污染。
        # 使用与训练集审计相同的阈值/shingle 口径；test 只在
        # 显式解封（include_test=True）时读取。
        evaluation_near_duplicates = _eval_near_duplicate_audit(
            dataset._raw_buffer,
            dataset.group_separator,
            near_eval_ranges,
            threshold=getattr(
                dataset,
                "near_duplicate_threshold",
                _NEAR_DUP_DEFAULT_THRESHOLD,
            ),
            shingle_bytes=getattr(
                dataset,
                "near_duplicate_shingle_bytes",
                _NEAR_DUP_DEFAULT_SHINGLE_BYTES,
            ),
        )

        for split, info in (
            evaluation_near_duplicates["per_split"] or {}
        ).items():
            if int(info["within_pairs"]) <= 0:
                continue
            consumers = (
                "固定窗口 bootstrap 与 checkpoint 选型"
                if split == "val"
                else "固定窗口 bootstrap"
            )
            warnings.append(
                f"{split}: 评估集内部检测到 "
                f"{info['within_pairs']} 对近重复群组"
                f"（{info['clusters']} 个独立簇 / 共 "
                f"{info['groups_scanned']} 个可比群组，"
                f"有效群组 {info['effective_groups']}，"
                "最大簇 "
                f"{info['largest_cluster_groups']} 个）；"
                f"{consumers}会把同源内容当作独立样本，"
                "评估区间可能偏窄，也不能把名义群组/窗口数"
                "当作独立样本量"
            )

        for pair, info in (
            evaluation_near_duplicates["cross_split"] or {}
        ).items():
            if int(info["pairs"]) <= 0:
                continue
            warnings.append(
                f"{pair}: 检测到 {info['pairs']} 对跨 split "
                "近重复评估群组（shingle Jaccard >= "
                f"{evaluation_near_duplicates['threshold']}）；"
                "val 选型可能通过近似内容间接看到 test，"
                "test 指标不再完全独立，需披露或重新划分"
            )

    return {
        "max_windows": max_windows,
        "splits": splits_report,
        "train_cleaning": dataset.train_cleaning,
        "split_fractions": {
            split: info["fraction"]
            for split, info in splits_report.items()
        },
        "dataset_split_fractions": dataset.split_fractions,
        "label_coverage": label_coverage,
        "distribution_shift": distribution_shift,
        "distribution_shift_bigram": distribution_shift_bigram,
        "distribution_shift_conditional": (
            distribution_shift_conditional
        ),
        "temporal_boundary_shift": temporal_boundary_shift,
        "unseen_transition_share": unseen_transition_share,
        "seen_context_share": seen_context_share,
        "duplicate_windows": duplicate_windows,
        "leakage_scan_coverage": scan_coverage,
        "content_chunks": content_chunks,
        "near_duplicates": near_duplicates,
        "evaluation_near_duplicates": (
            evaluation_near_duplicates
        ),
        "degenerate_runs": degenerate_runs,
        "val_label_noise": val_label_noise,
        "split_isolation": _split_isolation_record(
            dataset.split_bounds,
            dataset.num_bytes,
        ),
        "groups": groups_report,
        "warnings": warnings,
        "violations": violations,
        "leakage_violations": leakage_violations,
        "evaluation_violations": evaluation_violations,
    }



def enforce_data_leakage_gate(
    report: Dict[str, object],
    context: str = "",
) -> bool:
    """
    跨 split 泄露的默认硬门：只要审计发现内容重复（窗口级、
    群组级或内容定义分块级的跨 split 重复），就拒绝继续，
    而不是仅打印警告。

    与 ``enforce_strict_data_audit`` 的分工：
    - 本函数默认启用，只拦截「泄露」这一类问题；
    - 越界 token、UTF-8/NUL 损坏等标签噪声仍由
      ``--strict-data-audit`` 控制，避免默认行为因少量脏字节
      直接中断；
    - ``--allow-data-leakage`` 是显式豁免，用于确认重复内容
      在业务上不是泄露的场景；豁免会写入运行协议，跨运行
      对比时不会被静默忽略。

    返回 False 表示调用方必须拒绝训练/拒绝解封测试集。
    """
    leakage = report.get("leakage_violations") or []

    if not leakage:
        # 通过也必须交代证据边界：窗口签名扫描可能只是抽样，
        # 「没拦下泄露」不等于「数据里没有泄露」。
        coverage = report.get("leakage_scan_coverage") or {}
        residual = coverage.get("residual_risk")
        if residual:
            print(f"[data] 泄露门通过（未发现跨 split 重复）：{residual}")
        return True

    where = f"（{context}）" if context else ""
    print(f"[data] 跨 split 泄露检查未通过{where}：")
    for violation in leakage:
        print(f"[data]   泄露: {violation}")
    print(
        "[data] 默认拒绝在泄露数据上继续。请先去重/清洗数据；"
        "若已确认这些重复内容不构成泄露，可显式传 "
        "--allow-data-leakage（会被记录进配置）。"
    )

    return False



def enforce_evaluation_duplicate_gate(
    report: Dict[str, object],
    context: str = "",
) -> bool:
    """
    评估集重复样本硬门：val/test 内部完全重复的群组会虚增
    评估样本量，并让固定窗口 bootstrap 把同一条样本重复计入，
    从而低估评估不确定性；验证集重复还会让 checkpoint 选型在
    相同样本上过拟合。

    val/test 不参与清洗（清洗评估集会改变被评估的分布），
    因此默认拒绝继续，要求重新切分/去重或在确认重复是目标
    分布的一部分时显式传 ``--allow-eval-duplicates``。返回
    False 表示调用方必须拒绝训练/拒绝解封测试集。
    """
    duplicates = report.get("evaluation_violations") or []

    if not duplicates:
        return True

    where = f"（{context}）" if context else ""
    print(f"[data] 评估集重复样本检查未通过{where}：")
    for violation in duplicates:
        print(f"[data]   重复: {violation}")
    print(
        "[data] val/test 不参与清洗；重复评估样本会虚增有效样本"
        "并使区间偏窄。请先去重后重新切分，或在确认这些重复"
        "属于目标分布时显式传 --allow-eval-duplicates"
        "（会被记录进配置）。"
    )

    return False



def enforce_strict_data_audit(
    report: Dict[str, object],
    context: str = "",
) -> bool:
    """
    严格数据审计：报告中存在致命问题（数据泄露/标签噪声）时
    打印明细并返回 False，否则返回 True。

    调用方据此拒绝开始训练或拒绝解封测试集：宁可停在这一步，
    也不在已泄露的数据上产出「看起来有效」的结论。
    """
    violations = report.get("violations") or []

    if not violations:
        return True

    where = f"（{context}）" if context else ""
    print(f"[data] 严格数据审计未通过{where}：")
    for violation in violations:
        print(f"[data]   严重: {violation}")

    return False



def test_label_integrity_violations(
    report: Dict[str, object],
) -> List[str]:
    """
    返回测试集标签完整性问题的明细（越界 token、非法 UTF-8、
    NUL）。只看 ``splits["test"]``，train/val 的同类问题仍由
    ``--strict-data-audit`` 控制。

    测试集是最终结论的唯一证据，不能像训练集那样在事后清洗
    （清洗 test 会改变被评估的分布，也等于用测试数据做决策）。
    因此在解封前默认拒绝带标签噪声的测试集，而不是在损坏的
    标签上报告一个看似精确的最终指标。
    """
    test_info = (report.get("splits") or {}).get("test")
    if not isinstance(test_info, dict):
        return []

    violations: List[str] = []
    invalid = int(test_info.get("invalid_tokens", 0) or 0)
    if invalid > 0:
        violations.append(
            f"test: {invalid} 个越界 token（标签超出词表）"
        )
    replacements = int(
        test_info.get("utf8_replacements", 0) or 0
    )
    if replacements > 0:
        violations.append(
            f"test: {replacements} 个无法按 UTF-8 解码的"
            "字节（标签噪声）"
        )
    nul_bytes = int(test_info.get("nul_bytes", 0) or 0)
    if nul_bytes > 0:
        violations.append(
            f"test: {nul_bytes} 个 NUL 字节（二进制/填充污染）"
        )
    return violations



def enforce_test_label_integrity(
    report: Dict[str, object],
    context: str = "test",
) -> bool:
    """
    测试集标签完整性硬门：测试集永远不参与清洗，因此带标签
    噪声的测试集不能给出可信的最终指标。默认拒绝解封；只有
    显式 ``--allow-corrupt-test-labels`` 才放行，且豁免会写入
    运行记录。
    """
    violations = test_label_integrity_violations(report)
    if not violations:
        return True

    where = f"（{context}）" if context else ""
    print(f"[test] 测试集标签完整性检查未通过{where}：")
    for violation in violations:
        print(f"[test]   标签噪声: {violation}")
    print(
        "[test] 测试集不参与清洗，带标签噪声的最终指标不可信；"
        "默认拒绝解封。若这些字节确实是目标分布的一部分，"
        "可显式传 --allow-corrupt-test-labels（会被记录）。"
    )
    return False



def distribution_shift_violations(
    report: Dict[str, object],
    max_byte_js: Optional[float] = None,
    max_bigram_js: Optional[float] = None,
    max_conditional_js: Optional[float] = None,
) -> List[str]:
    """
    按显式阈值检查数据审计中的分布偏移指标。

    JS 距离越大表示两个 split 的分布差异越大；阈值只在该
    flag 被显式设置时生效，默认仍保留「警告但继续」的诊断
    模式。返回逐对 split 的超阈值说明。

    ``max_byte_js`` 同时约束时间边界块（相邻 split 的尾/头
    分布）：只检查全局直方图会让切分边界附近的突变漏过硬门。
    """
    checks = (
        ("distribution_shift", "字节分布 JS", max_byte_js),
        (
            "distribution_shift_bigram",
            "字节二元组分布 JS",
            max_bigram_js,
        ),
        (
            "distribution_shift_conditional",
            "条件标签分布 JS",
            max_conditional_js,
        ),
    )
    violations: List[str] = []

    for field, label, limit in checks:
        if limit is None:
            continue
        limit_value = _finite_float(limit)
        if limit_value is None:
            continue
        values = report.get(field) or {}
        if not isinstance(values, dict):
            continue
        for pair, value in values.items():
            distance = _finite_float(value)
            if distance is None:
                continue
            if distance > limit_value:
                violations.append(
                    f"{pair}: {label} = {distance:.4f} "
                    f"> 阈值 {limit_value:.4f}"
                )

    # 全局分布可能把边界两侧的漂移平均掉；显式设置单字节
    # 阈值时，边界块也必须受同一个硬门约束。
    if max_byte_js is not None:
        limit_value = _finite_float(max_byte_js)
        boundary = report.get("temporal_boundary_shift") or {}
        pairs = (
            boundary.get("pairs")
            if isinstance(boundary, dict)
            else None
        )
        if limit_value is not None and isinstance(pairs, dict):
            for pair, info in pairs.items():
                if not isinstance(info, dict):
                    continue
                distance = _finite_float(info.get("js"))
                if distance is None:
                    continue
                if distance > limit_value:
                    violations.append(
                        f"{pair}: 时间边界字节分布 JS = "
                        f"{distance:.4f} > 阈值 "
                        f"{limit_value:.4f}"
                    )

    return violations



def enforce_distribution_shift_gate(
    report: Dict[str, object],
    max_byte_js: Optional[float] = None,
    max_bigram_js: Optional[float] = None,
    max_conditional_js: Optional[float] = None,
    context: str = "",
) -> bool:
    """
    分布偏移硬门：显式设置阈值后，超阈值的 train/val（以及
    解封时的 test）偏移会拒绝继续，而不是只打印警告。

    分布偏移本身不等于数据泄露：它说明训练与评估数据来自
    不同分布，指标可能包含外推。拒绝继续是为了迫使先检查
    切分/时间顺序或调整评估口径，避免把外推误差当成方法
    收益。未设置任何阈值时返回 True。
    """
    violations = distribution_shift_violations(
        report,
        max_byte_js=max_byte_js,
        max_bigram_js=max_bigram_js,
        max_conditional_js=max_conditional_js,
    )
    if not violations:
        return True

    where = f"（{context}）" if context else ""
    print(f"[data] 分布偏移检查未通过{where}：")
    for violation in violations:
        print(f"[data]   偏移: {violation}")
    print(
        "[data] 默认拒绝在分布偏移超阈值的数据上继续；"
        "请检查切分/时间顺序与评估口径，或显式放宽/去掉 "
        "--max-byte-shift / --max-bigram-shift / "
        "--max-conditional-shift。"
    )
    return False
