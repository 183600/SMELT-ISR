from __future__ import annotations



def probe_data_pipeline(
    dataset: ByteDataset,
    device: torch.device,
    *,
    batch_size: int = 4,
    batches: int = 2,
    seed: int = 20240607,
    include_test: bool = False,
) -> Dict[str, object]:
    """
    训练前的数据管线完整性质检（中间结果检查）。

    过拟合小批量只能证明「模型在本数据管线的输出上可以学到
    东西」：如果 x/y 错位、标签映射错误或切分越界，模型可以
    照样把固定小批次背下来，sanity check 无法发现。这里用
    独立重建的方式逐位核对数据协议：

    - 三段切分边界连续、长度与字节总数一致，train/val/test
      互不重叠（窗口不跨越 split 边界）；
    - 训练批的采样起点可复现，且 ``batch()`` 返回的 (x, y)
      与「从原始 train 字节按起点直接切片」逐位一致（覆盖
      群组去重后的合法起点区间路径）；
    - 验证固定窗口的起点可复现，且 ``fixed_eval_batch()``
      的 (x, y) 同样逐位一致；
    - 输入 token 落在 [0, vocab_size) 内、dtype 与形状符合
      约定（x 为 int32、y 为 uint8，监督位是后移一位的目标）。

    返回结构化报告；``failures`` 非空表示数据协议存在实质
    错误（错位、越界、不可复现），在训练前必须处理。
    """
    seq_len = dataset.seq_len
    failures: List[str] = []

    report: Dict[str, object] = {
        "ok": True,
        "failures": failures,
        "seq_len": seq_len,
        "train_checked": False,
        "val_fixed_checked": False,
        "test_fixed_checked": False,
        "split_bounds": {
            split: list(bounds)
            for split, bounds in dataset.split_bounds.items()
        },
    }

    bounds = dataset.split_bounds
    train_bounds = bounds["train"]
    val_bounds = bounds["val"]
    test_bounds = bounds["test"]

    if not (
        train_bounds[0] == 0
        and train_bounds[1] == val_bounds[0]
        and val_bounds[1] == test_bounds[0]
        and test_bounds[1] == dataset.num_bytes
    ):
        failures.append(
            "split 边界不连续（train/val/test 必须首尾相接且"
            "覆盖全部字节）"
        )

    split_lengths = (
        train_bounds[1] - train_bounds[0]
        + (val_bounds[1] - val_bounds[0])
        + (test_bounds[1] - test_bounds[0])
    )
    if split_lengths != dataset.num_bytes:
        failures.append(
            "split 长度之和与字节总数不一致"
        )

    if len(dataset.train) != train_bounds[1] - train_bounds[0]:
        failures.append("训练集长度与切分边界不一致")
    if len(dataset.val) != val_bounds[1] - val_bounds[0]:
        failures.append("验证集长度与切分边界不一致")
    if dataset.has_test and (
        len(dataset.test) != test_bounds[1] - test_bounds[0]
    ):
        failures.append("测试集长度与切分边界不一致")

    max_train_start = len(dataset.train) - seq_len
    if max_train_start < 1:
        failures.append("训练集没有合法窗口起点")

    # ---- 训练批：采样起点 + x/y 逐位对齐 ----
    try:
        probe_generator = torch.Generator().manual_seed(seed)
        starts = dataset._sample_train_starts(
            batch_size,
            probe_generator,
        )

        if int(starts.numel()) != batch_size:
            failures.append("训练批采样起点数量与 batch_size 不一致")

        if int(starts.min()) < 0 or int(starts.max()) > max_train_start - 1:
            failures.append(
                "训练批采样起点越界（窗口会跨越训练 split 边界）"
            )

        expected_x, expected_y = _expected_window_pairs(
            dataset.train,
            starts,
            seq_len,
        )

        batch_generator = torch.Generator().manual_seed(seed)
        x, y = dataset.batch(
            "train",
            batch_size,
            batch_generator,
            device,
        )

        report["train_checked"] = True

        if x.cpu().dtype != torch.int32:
            failures.append(
                f"训练批 x dtype 为 {x.dtype}，期望 int32"
            )
        if y.cpu().dtype != torch.uint8:
            failures.append(
                f"训练批 y dtype 为 {y.dtype}，期望 uint8"
            )

        if tuple(x.shape) != tuple(expected_x.shape):
            failures.append(
                f"训练批 x 形状 {tuple(x.shape)} 与期望 "
                f"{tuple(expected_x.shape)} 不一致"
            )
        elif not torch.equal(x.cpu(), expected_x):
            failures.append(
                "训练批 x 与原始字节切片不一致（输入窗口错位）"
            )

        if tuple(y.shape) != tuple(expected_y.shape):
            failures.append(
                f"训练批 y 形状 {tuple(y.shape)} 与期望 "
                f"{tuple(expected_y.shape)} 不一致"
            )
        elif not torch.equal(y.cpu(), expected_y):
            failures.append(
                "训练批 y 与原始字节切片不一致（下一 token "
                "标签错位）"
            )

        # 同一 seed 的第二次采样必须逐位一致：训练批采样
        # 不允许依赖生成器之外的任何隐式状态。
        repeat_x, repeat_y = dataset.batch(
            "train",
            batch_size,
            torch.Generator().manual_seed(seed),
            device,
        )
        if not (
            torch.equal(x.cpu(), repeat_x.cpu())
            and torch.equal(y.cpu(), repeat_y.cpu())
        ):
            failures.append(
                "训练批采样不可复现（同一 seed 两次采样结果"
                "不同）"
            )

        if x.numel() > 0 and (
            int(x.min()) < 0 or int(x.max()) >= 256
        ):
            failures.append(
                "训练批输入 token 超出 [0, 256) 字节范围"
            )

        if y.numel() > 0 and (
            int(y.min()) < 0 or int(y.max()) >= 256
        ):
            failures.append(
                "训练批标签 token 超出 [0, 256) 字节范围"
            )
    except (AssertionError, RuntimeError, ValueError) as error:
        failures.append(f"训练批检查抛出异常: {error}")

    # ---- 验证固定窗口：起点可复现 + x/y 逐位对齐 ----
    num_starts = len(dataset.val) - seq_len
    try:
        total_batches = min(batches * batch_size, num_starts)
        fixed_starts_all = torch.linspace(
            0,
            num_starts - 1,
            steps=total_batches,
            dtype=torch.float64,
        ).round().long()
        # batch_index=0 只返回前 batch_size 个固定起点。
        fixed_starts = fixed_starts_all[:batch_size]

        expected_x, expected_y = _expected_window_pairs(
            dataset.val,
            fixed_starts,
            seq_len,
        )

        dataset.clear_eval_batch_cache()
        x, y = dataset.fixed_eval_batch(
            "val",
            0,
            batches,
            batch_size,
            device,
        )

        report["val_fixed_checked"] = True

        if x.cpu().dtype != torch.int32:
            failures.append(
                f"验证固定窗口 x dtype 为 {x.dtype}，期望 int32"
            )
        if y.cpu().dtype != torch.uint8:
            failures.append(
                f"验证固定窗口 y dtype 为 {y.dtype}，期望 uint8"
            )

        if tuple(x.shape) != tuple(expected_x.shape):
            failures.append(
                f"验证固定窗口 x 形状 {tuple(x.shape)} 与期望 "
                f"{tuple(expected_x.shape)} 不一致"
            )
        elif not torch.equal(x.cpu(), expected_x):
            failures.append(
                "验证固定窗口 x 与原始字节切片不一致（固定评估"
                "窗口不可复现或输入错位）"
            )
        if tuple(y.shape) != tuple(expected_y.shape):
            failures.append(
                f"验证固定窗口 y 形状 {tuple(y.shape)} 与期望 "
                f"{tuple(expected_y.shape)} 不一致"
            )
        elif not torch.equal(y.cpu(), expected_y):
            failures.append(
                "验证固定窗口 y 与原始字节切片不一致（评估标签"
                "错位）"
            )

        if x.numel() > 0 and (
            int(x.min()) < 0 or int(x.max()) >= 256
        ):
            failures.append(
                "验证固定窗口输入 token 超出 [0, 256) 字节范围"
            )
        if y.numel() > 0 and (
            int(y.min()) < 0 or int(y.max()) >= 256
        ):
            failures.append(
                "验证固定窗口标签 token 超出 [0, 256) 字节范围"
            )

        # 清空结果后重建必须逐位一致：固定评估窗口跨进程/
        # 跨调用不依赖任何随机状态。
        dataset.clear_eval_batch_cache()
        repeat_x, repeat_y = dataset.fixed_eval_batch(
            "val",
            0,
            batches,
            batch_size,
            device,
        )
        if not (
            torch.equal(x.cpu(), repeat_x.cpu())
            and torch.equal(y.cpu(), repeat_y.cpu())
        ):
            failures.append(
                "验证固定窗口重复构建不可复现（结果依赖调用"
                "顺序或随机状态）"
            )
    except (AssertionError, RuntimeError, ValueError) as error:
        failures.append(f"验证固定窗口检查抛出异常: {error}")

    # ---- 测试固定窗口：只在最终解封时核对 ----
    if include_test and dataset.has_test:
        num_starts = len(dataset.test) - seq_len
        try:
            total_batches = min(
                batches * batch_size,
                num_starts,
            )
            fixed_starts_all = torch.linspace(
                0,
                num_starts - 1,
                steps=total_batches,
                dtype=torch.float64,
            ).round().long()
            fixed_starts = fixed_starts_all[:batch_size]

            expected_x, expected_y = _expected_window_pairs(
                dataset.test,
                fixed_starts,
                seq_len,
            )

            dataset.clear_eval_batch_cache()
            x, y = dataset.fixed_eval_batch(
                "test",
                0,
                batches,
                batch_size,
                device,
            )

            report["test_fixed_checked"] = True

            if x.cpu().dtype != torch.int32:
                failures.append(
                    f"测试固定窗口 x dtype 为 {x.dtype}，"
                    "期望 int32"
                )
            if y.cpu().dtype != torch.uint8:
                failures.append(
                    f"测试固定窗口 y dtype 为 {y.dtype}，"
                    "期望 uint8"
                )

            if tuple(x.shape) != tuple(expected_x.shape):
                failures.append(
                    "测试固定窗口 x 形状与期望不一致"
                )
            elif not torch.equal(x.cpu(), expected_x):
                failures.append(
                    "测试固定窗口 x 与原始测试字节不一致"
                    "（测试输入错位或窗口越界）"
                )

            if tuple(y.shape) != tuple(expected_y.shape):
                failures.append(
                    "测试固定窗口 y 形状与期望不一致"
                )
            elif not torch.equal(y.cpu(), expected_y):
                failures.append(
                    "测试固定窗口 y 与原始测试字节不一致"
                    "（测试标签错位）"
                )

            if x.numel() > 0 and (
                int(x.min()) < 0 or int(x.max()) >= 256
            ):
                failures.append(
                    "测试固定窗口输入 token 超出 [0, 256) "
                    "字节范围"
                )
            if y.numel() > 0 and (
                int(y.min()) < 0 or int(y.max()) >= 256
            ):
                failures.append(
                    "测试固定窗口标签 token 超出 [0, 256) "
                    "字节范围"
                )

            dataset.clear_eval_batch_cache()
            repeat_x, repeat_y = dataset.fixed_eval_batch(
                "test",
                0,
                batches,
                batch_size,
                device,
            )
            if not (
                torch.equal(x.cpu(), repeat_x.cpu())
                and torch.equal(y.cpu(), repeat_y.cpu())
            ):
                failures.append(
                    "测试固定窗口重复构建不可复现"
                )
        except (AssertionError, RuntimeError, ValueError) as error:
            failures.append(
                f"测试固定窗口检查抛出异常: {error}"
            )

    # ---- 群组边界：独立核对「任何群组不跨 split」----
    if dataset.group_separator is not None:
        report["group_boundaries_checked"] = True
        separator = dataset.group_separator

        for split, (begin, end) in bounds.items():
            # 空 split（如 test_fraction=0 时的 test）没有群组，
            # 其起点可能恰为文件末尾，不应要求末尾再有一个
            # 分组分隔符。
            if begin <= 0 or end <= begin:
                continue
            prefix = bytes(
                dataset._raw[
                    begin - len(separator):begin
                ]
            )
            if prefix != separator:
                failures.append(
                    f"{split} split 起点不在群组分隔符之后"
                    "（群组可能被拆到两侧）"
                )

        if dataset.groups_per_split is not None:
            recounted = dict(
                zip(
                    ("train", "val", "test"),
                    _count_groups(
                        dataset._raw_buffer,
                        separator,
                        [val_bounds[0], val_bounds[1]],
                    ),
                )
            )
            if recounted != dataset.groups_per_split:
                failures.append(
                    "群组计数与构造记录不一致（群组划分"
                    "可能被破坏）"
                )

    # ---- 训练集群组清洗：窗口不得触碰被剔除群组 ----
    if dataset.train_excluded_spans:
        report["train_cleaning_checked"] = True
        merged = _merge_byte_intervals(
            dataset.train_excluded_spans
        )

        if merged != list(dataset.train_excluded_spans):
            failures.append(
                "训练集清洗区间无序或重叠"
            )

        if any(
            start < 0 or end > len(dataset.train)
            for start, end in merged
        ):
            failures.append(
                "训练集清洗区间越出 train split"
            )

        cleaning = dataset.train_cleaning or {}
        dropped_bytes = sum(
            end - start for start, end in merged
        )
        if int(cleaning.get("dropped_bytes", -1)) != dropped_bytes:
            failures.append(
                "训练集清洗统计的 dropped_bytes 与"
                "实际排除区间不一致"
            )
        if int(
            cleaning.get("post_clean_train_bytes", -1)
        ) != len(dataset.train) - dropped_bytes:
            failures.append(
                "训练集清洗统计的剩余字节数不一致"
            )

        probe_starts = dataset._sample_train_starts(
            32,
            torch.Generator().manual_seed(seed + 1),
        )
        for start in probe_starts.tolist():
            window_end = start + seq_len
            if any(
                low <= window_end and high > start
                for low, high in merged
            ):
                failures.append(
                    "训练批采样窗口触碰被清洗内容"
                    "（清洗内容仍会进入训练）"
                )
                break

        if _valid_window_start_intervals(
            len(dataset.train) - seq_len,
            seq_len,
            merged,
        ) is None:
            failures.append(
                "训练集存在清洗区间但没有生成合法"
                "起点区间"
            )

    report["ok"] = not failures
    return report




# 重复窗口扫描的采样上限：窗口数超过该值时按等间距抽样。
_AUDIT_MAX_WINDOWS = 100000


# train/val(/test) 字节分布 JS 距离超过该值时报分布偏移警告。
_AUDIT_SHIFT_WARN = 0.1


# 字节二元组分布偏移的告警阈值：只看单字节频率会漏掉
# 「词表相同但上下文/顺序变了」的偏移，例如训练集是
# abab...、评估集是 baba...；二元组 JS 距离对这类偏移敏感。
_AUDIT_BIGRAM_SHIFT_WARN = 0.1


# 条件分布 P(next_byte | prev_byte) 的 JS 距离阈值：联合二元组
# 分布相同也可能存在「同一上下文接不同标签」的监督漂移。
_AUDIT_CONDITIONAL_SHIFT_WARN = 0.1


# eval 中未见过的 (上下文, 目标) 转移比例阈值：比例过高说明
# 训练集没有覆盖评估集的监督模式，泛化指标会包含大量外推。
_AUDIT_UNSEEN_TRANSITION_WARN = 0.5


# 时间边界分布漂移：全局直方图会把「train 尾部已经漂移、val
# 头部继续漂移」相互抵消。分别取 train 尾部与 val 头部（以及
# val 尾部与 test 头部）各一个块，比较字节分布的 JS 距离。
# 块大小取相邻 split 较小者的 10%，限制在 [4KiB, 1MiB]；
# split 太小时不做边界审计，避免把抽样噪声当漂移。
_AUDIT_BOUNDARY_BLOCK_FRACTION = 0.1

_AUDIT_BOUNDARY_MIN_BYTES = 4096

_AUDIT_BOUNDARY_MAX_BYTES = 1024 * 1024

_AUDIT_BOUNDARY_SHIFT_WARN = 0.1


# 单个字节占比超过该阈值时疑似填充/退化数据（异常值检查）。
_AUDIT_TOP_BYTE_WARN = 0.5


# UTF-8 解码失败（替换字符）占比超过该阈值时报语料损坏警告：
# 字节级语言模型直接以字节为标签，无法解码的字节就是
# 「标签噪声」的语料来源，必须先发现再决定是否清洗。
_AUDIT_UTF8_REPLACE_WARN = 1e-3


# NUL 字节占比超过该阈值时报二进制/填充污染警告
# （文本语料中 NUL 基本只来自错误编码或文件拼接损坏）。
_AUDIT_NUL_WARN = 1e-4


# 群组级异常值检查：长度至少为该值、且单一字节占比达到阈值的
# 群组视为低信息/退化群组（填充、重复字符、错误拼接的常见形态）。
# 只在 train 上统计；--drop-train-degenerate-groups 可整组清洗。
_AUDIT_DEGENERATE_GROUP_MIN_BYTES = 64

_AUDIT_DEGENERATE_GROUP_TOP_SHARE = 0.99


# 无群组连续字节流的异常值：长度至少为该值的同字节连续段
# 视为低信息/退化段（填充、错误拼接），可用
# --drop-train-degenerate-runs 只从 train 剔除。
_AUDIT_DEGENERATE_RUN_MIN_BYTES = 64


# 实际 split 比例相对名义值漂移超过该幅度（且绝对差
# 至少 _AUDIT_SPLIT_FRACTION_MIN_ABS）时告警：群组边界
# 吸附可能把验证/测试集缩小到名义比例的一小部分。
_AUDIT_SPLIT_FRACTION_DRIFT = 0.2

_AUDIT_SPLIT_FRACTION_MIN_ABS = 0.01


# 内容定义分块审计的最小/最大块长与边界概率掩码。分块边界
# 由块内局部内容的滚动哈希决定，而不是全局固定偏移，因此
# 同一段内容即使整体错位复制也会在后续块上重新对齐。最小
# 块长同时限制为至少 seq_len + 1：共享块足以覆盖一个完整
# 训练/评估窗口，才能作为跨 split 泄露证据。
_AUDIT_CHUNK_MIN_BYTES = 32

_AUDIT_CHUNK_MAX_BYTES = 512

_AUDIT_CHUNK_BOUNDARY_MASK = 0x3F

_AUDIT_CHUNK_GEAR_TABLE = tuple(
    int.from_bytes(
        hashlib.blake2b(
            bytes((value,)),
            digest_size=8,
        ).digest(),
        "little",
    )
    for value in range(256)
)

_AUDIT_CHUNK_UINT64_MASK = (1 << 64) - 1



# 训练集清洗的删除偏差审计：被删除内容与保留内容的字节/二元组
# 分布差异过大时，说明清洗不是随机删除，训练分布已被选择性改变。
# 这不是「清洗有 bug」，而是必须在报告里披露的干预效应：清洗后的
# 损失不能直接与未清洗运行比较，也不能把删除比例高当成数据更多。
_AUDIT_DELETION_BIAS_WARN = 0.10

_AUDIT_DELETION_DROP_WARN = 0.20


# 评估集标签覆盖：训练集从未出现过的字节类在评估 token 中占比
# 超过该阈值时告警。字节级 LM 无法通过上下文学习这些标签，
# 它们构成不可约的外推误差，不能全部归因于模型能力。
_AUDIT_UNSEEN_LABEL_WARN = 1e-4



def _byte_bigram_counts(
    tokens: torch.Tensor,
    excluded_spans: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    统计 split 内相邻字节对的频次（256×256 展平）。

    排除区间两侧的字节不能组成跨越清洗内容的虚假二元组：
    只在每个连续有效内容区间内部计数，与训练窗口/统计基线
    的有效训练口径一致。
    """
    vocab_size = 256
    counts = torch.zeros(
        vocab_size * vocab_size,
        dtype=torch.int64,
    )

    intervals = _subtract_byte_intervals(
        0,
        int(tokens.numel()),
        list(excluded_spans or []),
    )

    for begin, end in intervals:
        if end - begin < 2:
            continue
        left = tokens[begin:end - 1].to(torch.int64)
        right = tokens[begin + 1:end].to(torch.int64)
        pairs = left * vocab_size + right
        counts += torch.bincount(
            pairs,
            minlength=vocab_size * vocab_size,
        )

    return counts



def _effective_edge_histogram(
    dataset: "ByteDataset",
    split: str,
    max_bytes: int,
    from_start: bool,
) -> Tuple[Optional[torch.Tensor], int]:
    """
    取 split 有效内容的头部或尾部连续字节，返回归一化字节分布
    与实际字节数。train 的清洗区间不参与边界漂移统计，避免把
    模型从未见过的字节当成训练尾部。
    """
    tokens = getattr(dataset, split)
    total = int(tokens.numel())
    if total <= 0 or max_bytes <= 0:
        return None, 0

    intervals = _subtract_byte_intervals(
        0,
        total,
        (
            dataset.train_excluded_spans
            if split == "train"
            else []
        ),
    )
    if not intervals:
        return None, 0

    ordered = intervals if from_start else list(reversed(intervals))
    remaining = max_bytes
    counts = torch.zeros(256, dtype=torch.float64)
    used = 0

    for begin, end in ordered:
        if remaining <= 0:
            break

        take = min(end - begin, remaining)
        if take <= 0:
            continue

        if from_start:
            chunk = tokens[begin:begin + take]
        else:
            chunk = tokens[end - take:end]

        counts += torch.bincount(
            chunk.to(torch.int64),
            minlength=256,
        ).to(torch.float64)
        used += take
        remaining -= take

    if used <= 0:
        return None, 0

    return counts / counts.sum(), used



def _temporal_boundary_shift_audit(
    dataset: "ByteDataset",
    split_names: List[str],
    warnings: List[str],
) -> Dict[str, object]:
    """
    时间边界分布漂移审计：全局直方图可能让「train 尾部已经
    漂移」和「val 头部继续漂移」相互抵消。分别取相邻 split
    的尾/头块做字节 JS 距离，专门检查切分边界附近是否存在
    全局统计看不到的分布突变。

    test 只在 include_test=True（解封评估）时参与，训练期间
    不会读取 test 内容。数据太少或清洗后有效内容不足时跳过，
    避免把小块抽样噪声当成漂移。
    """
    pairs: Dict[str, object] = {}

    for left, right in (
        ("train", "val"),
        ("val", "test"),
    ):
        if left not in split_names or right not in split_names:
            continue

        left_total = (
            int(dataset.effective_train_bytes)
            if left == "train"
            else int(
                dataset.split_bounds[left][1]
                - dataset.split_bounds[left][0]
            )
        )
        right_total = int(
            dataset.split_bounds[right][1]
            - dataset.split_bounds[right][0]
        )
        smaller = min(left_total, right_total)

        if smaller < 2 * _AUDIT_BOUNDARY_MIN_BYTES:
            continue

        block_bytes = int(
            smaller * _AUDIT_BOUNDARY_BLOCK_FRACTION
        )
        block_bytes = max(
            _AUDIT_BOUNDARY_MIN_BYTES,
            min(block_bytes, _AUDIT_BOUNDARY_MAX_BYTES),
        )

        left_hist, left_used = _effective_edge_histogram(
            dataset,
            left,
            block_bytes,
            from_start=False,
        )
        right_hist, right_used = _effective_edge_histogram(
            dataset,
            right,
            block_bytes,
            from_start=True,
        )
        if (
            left_hist is None
            or right_hist is None
            or left_used <= 0
            or right_used <= 0
        ):
            continue

        distance = _js_distance(left_hist, right_hist)
        key = f"{left}_tail_{right}_head"
        pairs[key] = {
            "js": round(distance, 6),
            "left_bytes": left_used,
            "right_bytes": right_used,
        }

        if distance > _AUDIT_BOUNDARY_SHIFT_WARN:
            warnings.append(
                f"{key}: 时间边界字节分布偏移较大"
                f"（JS 距离 {distance:.4f}，各取 "
                f"{left_used}/{right_used} 字节）；"
                "全局直方图可能掩盖切分边界附近的分布突变，"
                "需检查时间漂移或切分点"
            )

    return {
        "block_fraction": _AUDIT_BOUNDARY_BLOCK_FRACTION,
        "min_bytes": _AUDIT_BOUNDARY_MIN_BYTES,
        "max_bytes": _AUDIT_BOUNDARY_MAX_BYTES,
        "warn_js": _AUDIT_BOUNDARY_SHIFT_WARN,
        "pairs": pairs,
    }



def _js_distance(
    p: torch.Tensor,
    q: torch.Tensor,
) -> float:
    """
    两个离散分布之间的 Jensen-Shannon 距离
    （JS 散度开方，log 以 2 为底，取值 [0, 1]；0 表示分布相同）。
    """
    m = 0.5 * (p + q)

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        # log2 才能让 JS 散度上界为 1 bit、距离落在 [0, 1]
        #（自然对数下上界是 ln 2，距离最大 sqrt(ln 2)≈0.83，
        # 与审计阈值按 [0, 1] 标定的口径不一致）。
        return (a[mask] * (a[mask] / b[mask]).log2()).sum()

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return math.sqrt(max(0.0, float(js)))



def _conditional_bigram_shift(
    train_counts: torch.Tensor,
    eval_counts: torch.Tensor,
    vocab_size: int = 256,
) -> Dict[str, float]:
    """
    给定上下文后的标签分布偏移（P(next_byte | prev_byte)）。

    单字节/联合二元组分布可能掩盖条件分布变化：同样的二元组
    词表，若「同一上下文后面接什么」发生变化，监督信号已经
    漂移。这里逐上下文比较 train 与 eval 的条件分布：

    - ``conditional_js``：按 eval 中该上下文的出现次数加权的
      条件 JS 距离（只统计 train 中出现过的上下文；对每行做
      Laplace 平滑，避免零概率使距离失真）；
    - ``seen_context_share``：eval 二元组中，其上下文在 train
      至少出现过一次的比例；
    - ``unseen_transition_share``：eval 二元组中，该
      (上下文, 目标) 转移在 train 从未出现的比例。

    统计量只用于审计，不进入模型训练或选型。train 计数已经
    扣除清洗区间，与神经网络/统计基线看到的训练内容一致。
    """
    train = train_counts.view(
        vocab_size,
        vocab_size,
    ).to(torch.float64)
    evaluation = eval_counts.view(
        vocab_size,
        vocab_size,
    ).to(torch.float64)

    train_ctx = train.sum(dim=1)
    eval_ctx = evaluation.sum(dim=1)
    total_eval_pairs = float(eval_ctx.sum())

    if total_eval_pairs <= 0.0:
        return {
            "conditional_js": 0.0,
            "seen_context_share": 0.0,
            "unseen_transition_share": 0.0,
            "eval_pairs": 0.0,
        }

    supported = (eval_ctx > 0) & (train_ctx > 0)
    seen_pairs = float(eval_ctx[supported].sum())
    seen_context_share = seen_pairs / total_eval_pairs

    unseen_pairs = float(
        (evaluation * (train == 0)).sum()
    )
    unseen_transition_share = unseen_pairs / total_eval_pairs

    if seen_pairs <= 0.0:
        conditional_js = 1.0
    else:
        train_rows = train[supported] + 1.0
        eval_rows = evaluation[supported] + 1.0
        train_probs = train_rows / train_rows.sum(
            dim=1,
            keepdim=True,
        )
        eval_probs = eval_rows / eval_rows.sum(
            dim=1,
            keepdim=True,
        )
        mean_probs = 0.5 * (train_probs + eval_probs)

        def _kl_rows(
            left: torch.Tensor,
            right: torch.Tensor,
        ) -> torch.Tensor:
            return (
                left * (left / right).log2()
            ).sum(dim=1)

        js_rows = 0.5 * _kl_rows(
            train_probs,
            mean_probs,
        ) + 0.5 * _kl_rows(
            eval_probs,
            mean_probs,
        )
        distances = js_rows.clamp_min(0.0).sqrt()
        weights = eval_ctx[supported]
        conditional_js = float(
            (distances * weights).sum() / weights.sum()
        )

    return {
        "conditional_js": conditional_js,
        "seen_context_share": seen_context_share,
        "unseen_transition_share": unseen_transition_share,
        "eval_pairs": total_eval_pairs,
    }



def _subtract_byte_intervals(
    begin: int,
    end: int,
    excluded_spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    返回 [begin, end) 内扣除排除区间后的连续内容区间。
    """
    result: List[Tuple[int, int]] = []
    cursor = begin

    for excluded_start, excluded_end in _merge_byte_intervals(
        excluded_spans
    ):
        if excluded_end <= begin or excluded_start >= end:
            continue
        if cursor < excluded_start:
            result.append(
                (cursor, min(excluded_start, end))
            )
        cursor = max(cursor, excluded_end)
        if cursor >= end:
            break

    if cursor < end:
        result.append((cursor, end))

    return [
        (start, stop)
        for start, stop in result
        if stop > start
    ]



def _interval_byte_counts(
    raw: memoryview,
    intervals: List[Tuple[int, int]],
    chunk_bytes: int = 1 << 20,
) -> torch.Tensor:
    """
    统计若干连续区间内的字节频次（256 类，int64）。

    从原始缓冲区按块读取，避免把整个 split 复制成 Python bytes。
    与 ``_byte_bigram_counts`` 同一口径：只统计区间内部，不制造
    跨越被删除内容的虚假计数。
    """
    counts = torch.zeros(256, dtype=torch.int64)
    # 分块大小必须至少推进一个字节；0/负值会让 while 原地
    # 踏步而不是报错或返回。调用方可能传实验性小块做核对。
    step_bytes = max(1, int(chunk_bytes))

    for span_start, span_end in intervals:
        position = span_start
        while position < span_end:
            stop = min(position + step_bytes, span_end)
            block_view = raw[position:stop]
            # 只读 memoryview（如 memoryview(bytes)）不能安全传给
            # torch.frombuffer：PyTorch 会警告它无法阻止写入。
            block = torch.frombuffer(
                bytearray(block_view)
                if block_view.readonly
                else block_view,
                dtype=torch.uint8,
            )
            if block.numel() > 0:
                counts += torch.bincount(
                    block,
                    minlength=256,
                )
            position = stop

    return counts



def _interval_bigram_counts(
    raw: memoryview,
    intervals: List[Tuple[int, int]],
    chunk_bytes: int = 1 << 20,
) -> torch.Tensor:
    """
    统计若干连续区间内的相邻字节对频次（256*256 展平）。

    分块时保留一个字节的重叠，确保块边界上的
    (stop-1, stop) 转移不会被漏计；区间之间不连接，避免把
    「删除内容之后紧接保留内容」误算成一个训练二元组。
    """
    vocab_size = 256
    counts = torch.zeros(
        vocab_size * vocab_size,
        dtype=torch.int64,
    )
    # 二元组分块至少取 2 字节：chunk_bytes=1 时 stop-1 会把
    # position 推回原处，造成死循环；0/负值同理。
    step_bytes = max(2, int(chunk_bytes))

    for span_start, span_end in intervals:
        if span_end - span_start < 2:
            continue

        position = span_start
        while position < span_end - 1:
            stop = min(position + step_bytes, span_end)
            block_view = raw[position:stop]
            block = torch.frombuffer(
                bytearray(block_view)
                if block_view.readonly
                else block_view,
                dtype=torch.uint8,
            )
            left = block[:-1].to(torch.int64)
            right = block[1:].to(torch.int64)
            counts += torch.bincount(
                left * vocab_size + right,
                minlength=vocab_size * vocab_size,
            )
            if stop >= span_end:
                break
            # 下一块从上一块最后一个字节开始，保留边界转移。
            position = stop - 1

    return counts



def _interval_length_summary(
    intervals: List[Tuple[int, int]],
) -> Dict[str, object]:
    """返回区间长度（span 数、总字节、均值与分位数）。"""
    lengths = sorted(
        end - start
        for start, end in intervals
        if end > start
    )
    if not lengths:
        return {
            "spans": 0,
            "bytes": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    def _quantile(fraction: float) -> int:
        index = int(round(fraction * (len(lengths) - 1)))
        return lengths[max(0, min(index, len(lengths) - 1))]

    return {
        "spans": len(lengths),
        "bytes": sum(lengths),
        "mean": round(sum(lengths) / len(lengths), 3),
        "median": _quantile(0.5),
        "p90": _quantile(0.9),
        "max": lengths[-1],
    }



def _top_byte_summary(
    counts: torch.Tensor,
    top_k: int = 5,
) -> List[List[object]]:
    """把字节频次转成 [[byte, share], ...]（share 保留 6 位）。"""
    total = int(counts.sum())
    if total <= 0:
        return []
    k = min(top_k, int(counts.numel()))
    values, indices = torch.topk(counts, k=k)
    return [
        [int(byte), round(float(value) / total, 6)]
        for value, byte in zip(values.tolist(), indices.tolist())
    ]



def _deletion_bias_record(
    raw: memoryview,
    train_end: int,
    excluded_spans: List[Tuple[int, int]],
) -> Optional[Dict[str, object]]:
    """
    量化训练集清洗的「删除偏差」：被删除的 train 内容与保留内容
    在字节分布、二元组分布与连续段长度上是否不同。

    清洗不是随机缺失：去重会优先删除与 val/test 重复或更早出现
    的群组，损坏字节清洗会优先删除含非法 UTF-8/NUL 的片段。
    因此删除后的训练分布是**选择性**的，清洗前后的损失不可直接
    比较，也不能把「删掉重复后指标变差/变好」全部归因于模型。
    本记录只做诊断，不改变任何删除决策；返回 None 表示没有删除。
    """
    merged = _merge_byte_intervals(excluded_spans)
    if not merged or train_end <= 0:
        return None

    kept_intervals = _subtract_byte_intervals(
        0,
        train_end,
        merged,
    )
    dropped_bytes = sum(end - start for start, end in merged)
    kept_bytes = sum(
        end - start for start, end in kept_intervals
    )

    dropped_counts = _interval_byte_counts(raw, merged)
    kept_counts = _interval_byte_counts(raw, kept_intervals)
    dropped_total = int(dropped_counts.sum())
    kept_total = int(kept_counts.sum())

    byte_js: Optional[float] = None
    if dropped_total > 0 and kept_total > 0:
        byte_js = round(
            _js_distance(
                dropped_counts.to(torch.float64) / dropped_total,
                kept_counts.to(torch.float64) / kept_total,
            ),
            6,
        )

    dropped_bigram = _interval_bigram_counts(raw, merged)
    kept_bigram = _interval_bigram_counts(
        raw,
        kept_intervals,
    )
    bigram_js: Optional[float] = None
    dropped_bigram_total = int(dropped_bigram.sum())
    kept_bigram_total = int(kept_bigram.sum())
    if dropped_bigram_total > 0 and kept_bigram_total > 0:
        bigram_js = round(
            _js_distance(
                dropped_bigram.to(torch.float64)
                / dropped_bigram_total,
                kept_bigram.to(torch.float64)
                / kept_bigram_total,
            ),
            6,
        )

    train_bytes = max(1, int(train_end))
    return {
        "schema": "smelt-isr/deletion-bias/v1",
        "dropped_bytes": dropped_bytes,
        "kept_bytes": kept_bytes,
        "train_bytes": int(train_end),
        "drop_fraction": round(dropped_bytes / train_bytes, 6),
        "kept_fraction": round(kept_bytes / train_bytes, 6),
        "excluded_span_lengths": _interval_length_summary(
            merged
        ),
        "kept_span_lengths": _interval_length_summary(
            kept_intervals
        ),
        "dropped_top_bytes": _top_byte_summary(dropped_counts),
        "kept_top_bytes": _top_byte_summary(kept_counts),
        "byte_js": byte_js,
        "bigram_js": bigram_js,
        "warn_byte_js": _AUDIT_DELETION_BIAS_WARN,
        "warn_drop_fraction": _AUDIT_DELETION_DROP_WARN,
        "interpretation": (
            "删除由内容/群组规则触发，不是随机缺失；byte_js / "
            "bigram_js 越大、drop_fraction 越高，清洗后的训练"
            "分布与原始分布差异越大。比较清洗/未清洗运行的"
            "指标前必须先核对本记录，不能把选择性删除当成同一"
            "数据分布上的方法差异"
        ),
    }



def _content_defined_chunk_scan(
    raw: memoryview,
    intervals: List[Tuple[int, int]],
    min_chunk: int,
    max_chunk: int,
) -> Dict[str, object]:
    """
    对若干内容区间做 content-defined chunking 并返回块指纹。

    每个块的有效内容指纹与它出现的绝对位置无关；共享同一
    指纹即表示两个 split 含有相同的连续字节片段。分块只读取
    原始字节，不参与任何特征计算。
    """
    chunks: Dict[bytes, List[Tuple[int, int]]] = {}
    total_chunks = 0
    total_bytes = 0
    shortest: Optional[int] = None
    longest = 0

    if min_chunk <= 0 or max_chunk < min_chunk:
        raise ValueError(
            "内容定义分块要求 0 < min_chunk <= max_chunk"
        )

    for begin, end in intervals:
        chunk_start = begin
        rolling = 0

        for position in range(begin, end):
            rolling = (
                (rolling << 1)
                + _AUDIT_CHUNK_GEAR_TABLE[raw[position]]
            ) & _AUDIT_CHUNK_UINT64_MASK
            length = position - chunk_start + 1

            if length < min_chunk:
                continue

            if (
                length >= max_chunk
                or (
                    rolling
                    & _AUDIT_CHUNK_BOUNDARY_MASK
                ) == 0
            ):
                digest = hashlib.blake2b(
                    raw[chunk_start:position + 1],
                    digest_size=16,
                ).digest()
                chunks.setdefault(digest, []).append(
                    (chunk_start, position + 1)
                )
                total_chunks += 1
                total_bytes += length
                shortest = (
                    length
                    if shortest is None
                    else min(shortest, length)
                )
                longest = max(longest, length)
                chunk_start = position + 1
                rolling = 0

        tail_length = end - chunk_start
        if tail_length >= min_chunk:
            digest = hashlib.blake2b(
                raw[chunk_start:end],
                digest_size=16,
            ).digest()
            chunks.setdefault(digest, []).append(
                (chunk_start, end)
            )
            total_chunks += 1
            total_bytes += tail_length
            shortest = (
                tail_length
                if shortest is None
                else min(shortest, tail_length)
            )
            longest = max(longest, tail_length)

    return {
        "chunks": chunks,
        "total_chunks": total_chunks,
        "total_bytes": total_bytes,
        "unique_chunks": len(chunks),
        "within_duplicates": total_chunks - len(chunks),
        "min_chunk_bytes": shortest,
        "max_chunk_bytes": longest,
    }
