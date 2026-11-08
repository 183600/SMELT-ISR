from __future__ import annotations



def _bad_case_report_from_buffers(
    loss_buffer: torch.Tensor,
    target_buffer: torch.Tensor,
    predicted_buffer: torch.Tensor,
    confidence_buffer: torch.Tensor,
    margin_buffer: torch.Tensor,
    written: int,
    top_candidates: List[
        Tuple[float, int, torch.Tensor, int, int, int, float, float]
    ],
    split: str,
    top_n: int,
    context_before: int,
    context_after: int,
    vocab_size: int,
    dataset: Optional["ByteDataset"] = None,
) -> Dict[str, object]:
    """由紧凑逐 token 缓冲生成坏案例报告（供流式收集器复用）。"""
    loss_all = loss_buffer[:written]
    target_all = target_buffer[:written].to(torch.int64)
    predicted_all = predicted_buffer[:written].to(torch.int64)
    confidence_all = confidence_buffer[:written].float()
    margin_all = margin_buffer[:written].float()

    num_tokens = loss_all.numel()

    wrong_mask = target_all != predicted_all
    token_accuracy = float(
        (1.0 - wrong_mask.float().mean()).item()
    )

    # 逐目标字节在评估 split 中的占比，用于标注噪声提示。
    target_counts = torch.bincount(
        target_all,
        minlength=vocab_size,
    ).to(torch.float64)
    target_share = target_counts / max(1.0, float(num_tokens))

    # 训练集目标分布：评估 split 中稀少的目标若在训练中很常见，
    # 更可能是评估抽样波动/上下文特异，而不是标注错误。
    # 与 unigram 基线同口径：被清洗群组不进入监督，也不进入该
    # 分布（见 effective_train_byte_counts 的约定），否则
    # train_rare 会把模型从未见过的字节当成「训练中常见」。
    train_target_counts: Optional[torch.Tensor] = None
    train_target_share: Optional[torch.Tensor] = None
    if dataset is not None:
        train_tokens = int(dataset.effective_train_bytes)
        if train_tokens > 0:
            train_target_counts = (
                dataset.effective_train_byte_counts().to(
                    torch.float64
                )
            )
            train_target_share = (
                train_target_counts / float(train_tokens)
            )

    # 错误归因的两个关键分层：位置与目标频率。它们把
    # 「模型错在哪些 token」从个例提升为可行动的分布结论：
    # 开头位置错误率高指向前缀/位置建模，稀有目标错误占比高
    # 指向数据覆盖或重采样，高频目标错误率高才更像模型盲区。
    total_wrong = int(wrong_mask.sum().item())
    error_rate_by_position: List[Dict[str, object]] = []
    if dataset is not None and int(dataset.seq_len) > 1:
        positions = (
            torch.arange(num_tokens)
            % int(dataset.seq_len)
        )
        error_rate_by_position = _stratified_error_rows(
            positions,
            wrong_mask,
            _BAD_CASE_POSITION_BUCKETS,
            total_wrong,
        )

    if train_target_share is not None:
        frequency_values = train_target_share[target_all]
        frequency_basis = "train_target_share"
    else:
        frequency_values = target_share[target_all]
        frequency_basis = "eval_target_share"
    error_rate_by_frequency = _stratified_error_rows(
        frequency_values,
        wrong_mask,
        _BAD_CASE_FREQUENCY_BUCKETS,
        total_wrong,
    )

    # 标签噪声核验：只对 top-N 候选做训练集上下文查找，避免为
    # 全量错误 token 建索引；这些候选正是报告里被人工复核的
    # 样本，核验结果直接决定 label_noise_suspect 归类。
    context_evidence: Dict[bytes, Dict[str, object]] = {}
    if dataset is not None and top_candidates:
        candidate_contexts = []
        for entry in top_candidates:
            window_row = entry[2]
            position = int(entry[3])
            target_position = position + 1
            begin = max(
                0,
                target_position - _BAD_CASE_LABEL_CONTEXT,
            )
            candidate_contexts.append(
                bytes(
                    window_row[begin:target_position].tolist()
                )
            )
        context_evidence = _train_next_byte_evidence(
            dataset,
            candidate_contexts,
        )

    # 高频混淆对：target * vocab + predicted。
    confusion_index = (
        target_all * vocab_size + predicted_all
    )
    confusion_counts = torch.bincount(
        confusion_index[wrong_mask],
        minlength=vocab_size * vocab_size,
    )
    confusion_count_by_token = confusion_counts[
        confusion_index
    ]

    top_confusion_values, top_confusion_indices = torch.topk(
        confusion_counts,
        k=min(10, int((confusion_counts > 0).sum().item()) or 1),
    )

    top_confusions = []
    for count, index in zip(
        top_confusion_values.tolist(),
        top_confusion_indices.tolist(),
    ):
        if count <= 0:
            break
        top_confusions.append([
            int(index) // vocab_size,
            int(index) % vocab_size,
            int(count),
        ])

    # 逐目标字节错误率（只保留出现过的目标字节，按错误率排序）。
    error_counts = torch.bincount(
        target_all[wrong_mask],
        minlength=vocab_size,
    )
    error_rate = torch.where(
        target_counts > 0,
        error_counts.to(torch.float64)
        / target_counts.clamp(min=1),
        torch.zeros(vocab_size, dtype=torch.float64),
    )
    seen = target_counts > 0
    seen_indices = torch.nonzero(seen).reshape(-1)
    seen_rates = error_rate[seen_indices]
    order = torch.argsort(seen_rates, descending=True)
    error_rate_by_target = []
    for index in order[:10].tolist():
        byte = int(seen_indices[index])
        error_rate_by_target.append([
            byte,
            round(float(seen_rates[index]), 6),
            int(target_counts[byte]),
        ])

    # 污染最严重的 top_n token：候选在流式循环中以「每批
    # top-N + 跨批截断」精确维护（全局 top-N 必在各批
    # top-N 内），这里只做最终排序与上下文渲染。
    top_candidates.sort(
        key=lambda entry: (-entry[0], entry[1])
    )

    top_cases = []
    for rank, entry in enumerate(top_candidates):
        (
            entry_loss,
            _entry_order,
            window_row,
            position,
            target,
            predicted,
            confidence,
            margin,
        ) = entry

        pair_count = int(
            confusion_counts[
                target * vocab_size + predicted
            ]
        )

        # 窗口 = [x[0]] + y，目标 token 是窗口的第
        # position+1 个元素（候选构建时已知）。
        target_position = position + 1

        context_begin = max(
            0,
            target_position - _BAD_CASE_LABEL_CONTEXT,
        )
        context_key = bytes(
            window_row[
                context_begin:target_position
            ].tolist()
        )
        evidence = context_evidence.get(context_key)
        label_noise_verified, train_context_target_share = (
            _label_noise_context_verdict(evidence, target)
        )

        train_rare = bool(
            train_target_share is None
            or float(train_target_share[target])
            < _BAD_CASE_RARE_TARGET_SHARE
        )

        if margin < _BAD_CASE_AMBIGUOUS_MARGIN:
            category = "ambiguous"
        elif (
            pair_count >= _BAD_CASE_SYSTEMATIC_COUNT
            and confidence >= 0.5
        ):
            category = "systematic_confusion"
        elif (
            confidence >= _BAD_CASE_CONFIDENT
            and train_rare
            and float(target_share[target])
            < _BAD_CASE_RARE_TARGET_SHARE
            and label_noise_verified
        ):
            # 「稀有目标 + 高置信度」本身不是标注错误：还必须
            # 在训练集中找到同一上下文，且训练监督的主流后续
            # 字节与评估标签不同。
            category = "label_noise_suspect"
        else:
            category = "uncertain"

        begin = max(0, target_position - context_before)
        end = min(
            window_row.numel(),
            target_position + 1 + context_after,
        )
        fragment = window_row[begin:end].tolist()
        marker = target_position - begin

        context = (
            _render_bytes(fragment[:marker])
            + "["
            + _render_byte(target)
            + "]"
            + _render_bytes(fragment[marker + 1:])
        )

        top_cases.append({
            "rank": rank + 1,
            "position": target_position,
            "target": target,
            "target_text": _render_byte(target),
            "predicted": predicted,
            "predicted_text": _render_byte(predicted),
            "loss": round(float(entry_loss), 6),
            "confidence": round(confidence, 6),
            "margin": round(margin, 6),
            "category": category,
            "confusion_count": pair_count,
            # 标签噪声核验证据：训练集同一上下文的支持数、
            # 主流后续字节及其占比、评估目标在该上下文中的
            # 占比。support=0 表示无法核验，不等于标注正确。
            "label_noise_verified": label_noise_verified,
            "train_context_support": (
                int(evidence["support"])
                if evidence is not None
                else 0
            ),
            "train_context_majority_byte": (
                int(evidence["majority_byte"])
                if evidence is not None
                and evidence.get("majority_byte") is not None
                else None
            ),
            "train_context_majority_share": (
                float(evidence["majority_share"])
                if evidence is not None
                and evidence.get("majority_share") is not None
                else None
            ),
            "train_context_target_share": (
                round(train_context_target_share, 6)
                if train_context_target_share is not None
                else None
            ),
            "train_target_share": (
                round(float(train_target_share[target]), 6)
                if train_target_share is not None
                else None
            ),
            "context": context,
            "window_text": _render_bytes(
                window_row.tolist()
            ),
        })

    category_counts: Dict[str, int] = {}
    for case in top_cases:
        key = str(case["category"])
        category_counts[key] = (
            category_counts.get(key, 0) + 1
        )

    # top-N 案例的来源窗口是否重复：同一窗口的多个位置都被
    # 选中时，案例数会被高估为独立证据；保留重复计数并在
    # 报告层提示，避免把「一个坏窗口」读成「N 种失败模式」。
    window_counts: Dict[str, int] = {}
    for case in top_cases:
        window_key = str(case["window_text"])
        window_counts[window_key] = (
            window_counts.get(window_key, 0) + 1
        )
    for case in top_cases:
        case["window_repeat_count"] = window_counts[
            str(case["window_text"])
        ]
    top_cases_window_concentration = (
        max(window_counts.values()) / len(top_cases)
        if top_cases
        else None
    )
    top_cases_concentration_warning = bool(
        top_cases
        and len(top_cases) >= _BAD_CASE_WINDOW_MIN_CASES
        and top_cases_window_concentration is not None
        and top_cases_window_concentration
        >= _BAD_CASE_WINDOW_CONCENTRATION
    )

    # 全量错误 token 的类别归因（不只是 top_n 个最差样本）：
    # 错误原因分布是提出新改进思路的最重要证据，必须按整个
    # 评估 split 统计，避免「最高损失的几个样本正好都是同一
    # 类」把结论带偏。分类规则与逐案例完全一致，优先级为
    # ambiguous > systematic_confusion > label_noise_suspect
    # > uncertain。
    train_rare_all = (
        torch.ones_like(target_all, dtype=torch.bool)
        if train_target_share is None
        else (
            train_target_share[target_all]
            < _BAD_CASE_RARE_TARGET_SHARE
        )
    )
    wrong_category_code = torch.full_like(target_all, 3)
    wrong_category_code = torch.where(
        (
            (confidence_all >= _BAD_CASE_CONFIDENT)
            & (
                target_share[target_all]
                < _BAD_CASE_RARE_TARGET_SHARE
            )
            # 全量归因无法逐 token 做上下文查找（成本过高），
            # 但至少要求目标在训练集同样稀少；否则「评估稀有」
            # 只是抽样波动，不能归因于标注。
            & train_rare_all
        ),
        torch.full_like(wrong_category_code, 2),
        wrong_category_code,
    )
    wrong_category_code = torch.where(
        (
            (confusion_count_by_token >= _BAD_CASE_SYSTEMATIC_COUNT)
            & (confidence_all >= 0.5)
        ),
        torch.full_like(wrong_category_code, 1),
        wrong_category_code,
    )
    wrong_category_code = torch.where(
        margin_all < _BAD_CASE_AMBIGUOUS_MARGIN,
        torch.full_like(wrong_category_code, 0),
        wrong_category_code,
    )
    wrong_category_code = wrong_category_code[wrong_mask]

    category_names = _BAD_CASE_CATEGORIES
    wrong_tokens = int(wrong_mask.sum())
    error_category_counts = {
        name: int((wrong_category_code == code).sum())
        for code, name in enumerate(category_names)
    }
    error_category_shares = {
        name: (
            round(count / wrong_tokens, 6)
            if wrong_tokens > 0
            else None
        )
        for name, count in error_category_counts.items()
    }

    calibration = _calibration_report(
        confidence_all,
        ~wrong_mask,
    )
    high_confidence_errors = int(
        (
            wrong_mask
            & (
                confidence_all
                >= _BAD_CASE_OVERCONFIDENT_THRESHOLD
            )
        ).sum().item()
    )
    if wrong_tokens > 0:
        calibration["high_confidence_error_count"] = (
            high_confidence_errors
        )
        calibration["high_confidence_error_share"] = round(
            high_confidence_errors / wrong_tokens,
            6,
        )

    verified_label_noise = sum(
        1
        for case in top_cases
        if case.get("label_noise_verified")
    )
    unverifiable_label_noise = sum(
        1
        for case in top_cases
        if int(case.get("train_context_support") or 0) == 0
    )
    label_noise_verification = {
        "context_length": _BAD_CASE_LABEL_CONTEXT,
        "min_train_support": _BAD_CASE_LABEL_MIN_SUPPORT,
        "max_target_share_in_context": (
            _BAD_CASE_LABEL_TARGET_MAX_SHARE
        ),
        "min_majority_share_in_context": (
            _BAD_CASE_LABEL_MAJORITY_MIN_SHARE
        ),
        "candidates": len(top_cases),
        "verified": verified_label_noise,
        "unverified": len(top_cases)
        - verified_label_noise,
        "unverifiable_no_train_context": (
            unverifiable_label_noise
        ),
        "note": (
            "只有训练集中同一上下文有足够支持、评估目标占比低、"
            "且主流后续字节有实质优势（≥50%）时，top-N 案例才"
            "标记为 label_noise_suspect；多数类优势不足说明"
            "上下文本身歧义。support=0 表示无法核验，不能据此"
            "断定标注正确。全量错误归因只做训练/评估目标稀有度"
            "交叉核对。"
        ),
    }

    top1 = top_cases[0] if top_cases else None
    dominant_category = (
        max(
            error_category_counts,
            key=lambda name: error_category_counts[name],
        )
        if wrong_tokens > 0
        else None
    )
    summary = (
        f"评估 {num_tokens} 个 token，token 准确率 "
        f"{token_accuracy:.4f}（{wrong_tokens} 个错误）；"
        + (
            "全量错误归因以 "
            f"{dominant_category}"
            f"（{error_category_shares[dominant_category] * 100:.1f}%）"
            "为主；最差样本 "
            if dominant_category is not None
            else "无错误 token；最差样本 "
        )
        + (
            "target="
            f"'{_render_byte(int(top1['target']))}', "
            "predicted="
            f"'{_render_byte(int(top1['predicted']))}', "
            f"loss={float(top1['loss']):.4f}, "
            f"category={top1['category']}。"
            if top1 is not None
            else "无。"
        )
    )

    report: Dict[str, object] = {
        "split": split,
        "tokens_evaluated": int(num_tokens),
        "token_accuracy": round(token_accuracy, 6),
        "wrong_tokens": wrong_tokens,
        "top_cases": top_cases,
        "top_cases_unique_windows": len(window_counts),
        "top_cases_window_concentration": (
            round(top_cases_window_concentration, 6)
            if top_cases_window_concentration is not None
            else None
        ),
        "top_cases_concentration_warning": (
            top_cases_concentration_warning
        ),
        "top_confusions": top_confusions,
        "error_rate_by_target": error_rate_by_target,
        "error_rate_by_position": error_rate_by_position,
        "error_rate_by_train_frequency": (
            error_rate_by_frequency
        ),
        "error_frequency_basis": frequency_basis,
        "category_counts": category_counts,
        "error_category_counts": error_category_counts,
        "error_category_shares": error_category_shares,
        "calibration": calibration,
        "label_noise_verification": (
            label_noise_verification
        ),
        "summary": summary,
    }

    return report



class _BadCaseCollector:
    """
    在评估前向中流式收集坏案例所需的全量紧凑统计。

    ``evaluate`` 与 ``analyze_bad_cases`` 使用完全相同的固定
    窗口与预测口径；最终评估时把收集器传入 ``evaluate``，即可
    得到同样的前向统计。收集器只保留逐 token 标量与 top-N 窗口。
    """

    def __init__(
        self,
        dataset: ByteDataset,
        split: str,
        batch_size: int,
        batches: int,
        top_n: int,
        context_before: int = 16,
        context_after: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.split = split
        self.top_n = top_n
        self.context_before = context_before
        self.context_after = context_after
        self.device = device
        self.sequence_length = dataset.seq_len

        _, eval_data = dataset._eval_windows(split)
        num_starts = len(eval_data) - self.sequence_length
        capacity = (
            min(batches * batch_size, num_starts)
            * self.sequence_length
        )

        self.async_copy = (
            device is not None and device.type == "cuda"
        )
        pin = self.async_copy
        self.loss_buffer = torch.empty(
            capacity, dtype=torch.float32, pin_memory=pin
        )
        self.target_buffer = torch.empty(
            capacity, dtype=torch.uint8, pin_memory=pin
        )
        self.predicted_buffer = torch.empty(
            capacity, dtype=torch.int16, pin_memory=pin
        )
        self.confidence_buffer = torch.empty(
            capacity, dtype=torch.float16, pin_memory=pin
        )
        self.margin_buffer = torch.empty(
            capacity, dtype=torch.float16, pin_memory=pin
        )

        self.top_candidates: List[
            Tuple[
                float,
                int,
                torch.Tensor,
                int,
                int,
                int,
                float,
                float,
            ]
        ] = []
        self.pending_candidates: List[
            Dict[str, torch.Tensor]
        ] = []
        self._candidates_materialized = False
        self.written = 0

    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """处理一个评估批（与 analyze_bad_cases 的单批逻辑一致）。"""
        logits = logits.float()

        log_normalizer = torch.logsumexp(logits, dim=-1)
        target_logits = logits.gather(
            -1,
            y.unsqueeze(-1).to(torch.int64),
        ).squeeze(-1)
        loss_tokens = log_normalizer - target_logits

        top_values, predicted = logits.max(dim=-1)
        confidence = (top_values - log_normalizer).exp()
        target_probability = (
            target_logits - log_normalizer
        ).exp()

        window = torch.cat(
            [x[:, :1].to(torch.uint8), y],
            dim=1,
        )

        batch_loss = loss_tokens.reshape(-1)
        batch_target = y.reshape(-1)
        batch_predicted = predicted.reshape(-1)
        batch_confidence = confidence.reshape(-1)
        batch_margin = (
            confidence - target_probability
        ).reshape(-1)

        batch_rows = batch_loss.numel()
        batch_end = self.written + batch_rows
        self.loss_buffer[
            self.written:batch_end
        ].copy_(
            batch_loss, non_blocking=self.async_copy
        )
        self.target_buffer[
            self.written:batch_end
        ].copy_(
            batch_target.to(torch.uint8),
            non_blocking=self.async_copy,
        )
        self.predicted_buffer[
            self.written:batch_end
        ].copy_(
            batch_predicted.to(torch.int16),
            non_blocking=self.async_copy,
        )
        self.confidence_buffer[
            self.written:batch_end
        ].copy_(
            batch_confidence.to(torch.float16),
            non_blocking=self.async_copy,
        )
        self.margin_buffer[
            self.written:batch_end
        ].copy_(
            batch_margin.to(torch.float16),
            non_blocking=self.async_copy,
        )
        self.written = batch_end

        if self.top_n > 0 and batch_loss.numel() > 0:
            batch_k = min(self.top_n, batch_loss.numel())
            batch_top_losses, batch_top_indices = (
                torch.topk(batch_loss, k=batch_k)
            )
            batch_top_indices = (
                batch_top_indices.reshape(-1)
            )
            rows = torch.div(
                batch_top_indices,
                self.sequence_length,
                rounding_mode="floor",
            )
            # 候选数据先在设备上暂存，报告阶段一次取回：
            # 每批 top-N 已经保证全局 top-N 必在并集中，
            # 最终按 (loss, 插入序号) 排序截断即可。
            self.pending_candidates.append({
                "losses": batch_top_losses,
                "indices": batch_top_indices,
                "windows": window.index_select(0, rows),
                "targets": batch_target.index_select(
                    0, batch_top_indices
                ),
                "predicted": batch_predicted.index_select(
                    0, batch_top_indices
                ),
                "confidence": batch_confidence.index_select(
                    0, batch_top_indices
                ),
                "margin": batch_margin.index_select(
                    0, batch_top_indices
                ),
            })

    def _materialize_candidates(self) -> None:
        """一次性取回流式暂存的 top-N 候选并重建全局列表。"""
        if self._candidates_materialized:
            return
        self._candidates_materialized = True

        if not self.pending_candidates:
            return

        losses = torch.cat(
            [item["losses"] for item in self.pending_candidates]
        ).cpu().tolist()
        indices = torch.cat(
            [item["indices"] for item in self.pending_candidates]
        ).cpu().tolist()
        windows = torch.cat(
            [item["windows"] for item in self.pending_candidates]
        ).cpu()
        targets = torch.cat(
            [item["targets"] for item in self.pending_candidates]
        ).cpu().tolist()
        predicted = torch.cat(
            [item["predicted"] for item in self.pending_candidates]
        ).cpu().tolist()
        confidence = torch.cat(
            [item["confidence"] for item in self.pending_candidates]
        ).cpu().tolist()
        margin = torch.cat(
            [item["margin"] for item in self.pending_candidates]
        ).cpu().tolist()

        offset = 0
        counter = 0
        for item in self.pending_candidates:
            count = int(item["losses"].numel())
            for slot in range(offset, offset + count):
                self.top_candidates.append((
                    float(losses[slot]),
                    counter,
                    windows[slot],
                    int(indices[slot])
                    % self.sequence_length,
                    int(targets[slot]),
                    int(predicted[slot]),
                    float(confidence[slot]),
                    float(margin[slot]),
                ))
                counter += 1
            offset += count

        self.pending_candidates.clear()
        self.top_candidates.sort(
            key=lambda entry: (-entry[0], entry[1])
        )
        del self.top_candidates[self.top_n:]

    def report(self, vocab_size: int) -> Dict[str, object]:
        """生成与独立分析路径逐位一致的坏案例报告。"""
        if self.async_copy and self.written > 0:
            torch.cuda.synchronize(self.device)

        self._materialize_candidates()

        if self.written == 0:
            return _empty_bad_case_report(self.split)

        return _bad_case_report_from_buffers(
            self.loss_buffer,
            self.target_buffer,
            self.predicted_buffer,
            self.confidence_buffer,
            self.margin_buffer,
            self.written,
            self.top_candidates,
            self.split,
            self.top_n,
            self.context_before,
            self.context_after,
            vocab_size,
            dataset=self.dataset,
        )



@torch.inference_mode()
def analyze_bad_cases(
    model: LanguageModel,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    split: str = "val",
    top_n: int = 10,
    context_before: int = 16,
    context_after: int = 8,
    collector: Optional["_BadCaseCollector"] = None,
) -> Dict[str, object]:
    """
    坏案例分析：把预测错误的样本打印出来并归类错误原因。

    使用与周期验证完全相同的固定窗口（protocol 口径一致），
    对每个目标 token 计算：

    - loss / 置信度（top1 概率）/ margin（top1 概率 − 目标概率）；
    - 污染最严重的 top_n 个 token，附上下文文本、目标与预测
      字节、错误类别；
    - 高频混淆对（target→predicted）与逐目标字节错误率；
    - 错误类别统计：ambiguous（样本极其模糊）、
      systematic_confusion（同类混淆反复出现，模型盲区）、
      label_noise_suspect（目标在训练/评估中都稀少、模型高
      置信度答错，且训练集同一上下文的主流后续字节与评估
      标签矛盾）、uncertain（其余）；
    - 置信度校准：ECE、平均置信度、实际准确率、过度自信
      幅度与可靠性分箱，回答「模型说 90% 时是否真的对 90%」；
    - 标签噪声核验：只对 top_n 候选查找训练集同上下文支持，
      报告 support / 主流后续字节 / 目标占比，无法核验时明确
      标记而不是默认相信。

    统计口径与实现：在固定窗口上流式处理全部目标 token，
    token 准确率、混淆对、逐目标错误率与错误类别归因都基于
    全量评估 token（与 evaluate 同一窗口集合、同一预测口径），
    不对评估集做随机下采样；top_n 最差样本通过「每批 top-N +
    跨批截断」精确维护（全局 top-N 必然出现在各自批次的
    top-N 内），窗口只为入选样本保留。相同数据与模型下报告
    逐位可复现。

    仅访问 val（或显式解封后的 test），绝不接触训练期间尘封的
    测试集；输出既打印，也写回 metrics.jsonl 的
    ``{"step": "bad_cases", ...}`` 记录，完整报告另存
    bad_cases.json，作为提出新改进思路的证据来源。

    ``collector`` 非 None 时直接使用最终 ``evaluate`` 已完成的
    同窗口前向统计。
    """
    if collector is not None:
        return collector.report(
            _unwrap_model(model).cfg.vocab_size
        )

    was_training = model.training
    model.eval()

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    vocab_size = _unwrap_model(model).cfg.vocab_size

    # fixed_eval_batch 的每个窗口行含 seq_len+1 个 token，
    # 输入 x 与目标 y 各 seq_len 个位置。
    sequence_length = dataset.seq_len

    # 全量统计缓冲：每个 token 保留 loss fp32、target uint8、
    # predicted int16、confidence/margin fp16；统计口径与 evaluate
    # 的固定窗口评估完全一致。缓冲按固定窗口的实际位置数分配、
    # 逐批写入。
    _, eval_data = dataset._eval_windows(split)
    num_starts = len(eval_data) - sequence_length
    capacity = min(batches * batch_size, num_starts) * sequence_length
    # CUDA 路径在全部批次结束后统一同步。CPU 路径保持同步语义。
    async_copy = device.type == "cuda"
    loss_buffer = torch.empty(
        capacity, dtype=torch.float32, pin_memory=async_copy
    )
    target_buffer = torch.empty(
        capacity, dtype=torch.uint8, pin_memory=async_copy
    )
    predicted_buffer = torch.empty(
        capacity, dtype=torch.int16, pin_memory=async_copy
    )
    confidence_buffer = torch.empty(
        capacity, dtype=torch.float16, pin_memory=async_copy
    )
    margin_buffer = torch.empty(
        capacity, dtype=torch.float16, pin_memory=async_copy
    )
    written = 0

    # 全局 top-N 最差 token：每批先取本批 top-N 候选再跨批截断
    #（全局 top-N 必然在各自批次的 top-N 内），窗口只为入选
    # 样本保留。
    # 插入序号保证并列损失下的排序确定（报告可复现）。
    # 候选项：(loss, 插入序号, 窗口行, 目标位置, target,
    # predicted, confidence, margin)。
    top_candidates: List[
        Tuple[float, int, torch.Tensor, int, int, int, float, float]
    ] = []
    top_insert_counter = 0

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            for batch_index in range(batches):
                x, y = dataset.fixed_eval_batch(
                    split,
                    batch_index,
                    batches,
                    batch_size,
                    device,
                )

                if x.size(0) == 0:
                    break

                result = model(x)

                # 坏案例统计只在 fp32 下做，避免半精度下
                # softmax/gather 的数值误差影响错误归因。
                logits = result["logits"].float()
                del result

                log_normalizer = torch.logsumexp(logits, dim=-1)
                target_logits = logits.gather(
                    -1,
                    y.unsqueeze(-1).to(torch.int64),
                ).squeeze(-1)
                loss_tokens = log_normalizer - target_logits

                # 预测与置信度只取 top1，与 evaluate 的
                # argmax 口径一致。
                top_values, predicted = logits.max(dim=-1)
                confidence = (
                    top_values - log_normalizer
                ).exp()
                target_probability = (
                    target_logits - log_normalizer
                ).exp()

                # 还原完整窗口：x = tokens[:-1]，y = tokens[1:]。
                window = torch.cat(
                    [x[:, :1].to(torch.uint8), y],
                    dim=1,
                )

                batch_loss = loss_tokens.reshape(-1)
                batch_target = y.reshape(-1)
                batch_predicted = predicted.reshape(-1)
                batch_confidence = confidence.reshape(-1)
                batch_margin = (
                    confidence - target_probability
                ).reshape(-1)

                batch_rows = batch_loss.numel()
                batch_end = written + batch_rows
                loss_buffer[written:batch_end].copy_(
                    batch_loss, non_blocking=async_copy
                )
                target_buffer[written:batch_end].copy_(
                    batch_target.to(torch.uint8),
                    non_blocking=async_copy,
                )
                predicted_buffer[written:batch_end].copy_(
                    batch_predicted.to(torch.int16),
                    non_blocking=async_copy,
                )
                confidence_buffer[written:batch_end].copy_(
                    batch_confidence.to(torch.float16),
                    non_blocking=async_copy,
                )
                margin_buffer[written:batch_end].copy_(
                    batch_margin.to(torch.float16),
                    non_blocking=async_copy,
                )
                written = batch_end

                # 本批的 top-N 候选（全局 top-N 必在其中）。
                if top_n > 0 and batch_loss.numel() > 0:
                    batch_k = min(top_n, batch_loss.numel())
                    batch_top_losses, batch_top_indices = (
                        torch.topk(batch_loss, k=batch_k)
                    )
                    # 取回本批候选损失，随后按索引取回候选标量
                    # 与窗口行。
                    batch_top_losses = (
                        batch_top_losses.cpu().tolist()
                    )
                    batch_top_indices = (
                        batch_top_indices.reshape(-1)
                    )
                    rows = torch.div(
                        batch_top_indices,
                        sequence_length,
                        rounding_mode="floor",
                    )
                    candidate_target = (
                        batch_target.index_select(
                            0, batch_top_indices
                        ).cpu().tolist()
                    )
                    candidate_predicted = (
                        batch_predicted.index_select(
                            0, batch_top_indices
                        ).cpu().tolist()
                    )
                    candidate_confidence = (
                        batch_confidence.index_select(
                            0, batch_top_indices
                        ).cpu().tolist()
                    )
                    candidate_margin = (
                        batch_margin.index_select(
                            0, batch_top_indices
                        ).cpu().tolist()
                    )
                    window_cpu = window.index_select(0, rows).cpu()
                    for candidate_slot, (
                        local_loss,
                        local_index,
                    ) in enumerate(zip(
                        batch_top_losses,
                        batch_top_indices.tolist(),
                    )):
                        position = local_index % sequence_length
                        top_candidates.append((
                            float(local_loss),
                            top_insert_counter,
                            window_cpu[candidate_slot],
                            position,
                            int(candidate_target[candidate_slot]),
                            int(candidate_predicted[candidate_slot]),
                            float(
                                candidate_confidence[candidate_slot]
                            ),
                            float(candidate_margin[candidate_slot]),
                        ))
                        top_insert_counter += 1

                    if len(top_candidates) > top_n:
                        top_candidates.sort(
                            key=lambda entry: (-entry[0], entry[1])
                        )
                        del top_candidates[top_n:]

        # 读取 CPU 缓冲前统一等待。
        if async_copy and written > 0:
            torch.cuda.synchronize(device)

        if written == 0:
            return _empty_bad_case_report(split)

        return _bad_case_report_from_buffers(
            loss_buffer,
            target_buffer,
            predicted_buffer,
            confidence_buffer,
            margin_buffer,
            written,
            top_candidates,
            split,
            top_n,
            context_before,
            context_after,
            vocab_size,
            dataset=dataset,
        )
    finally:
        model.train(was_training)



def print_bad_cases(report: Dict[str, object]) -> None:
    """打印坏案例报告：最差样本、混淆对与错误分类统计。"""
    split = report.get("split")
    print(
        f"[bad-case] split={split} tokens="
        f"{report.get('tokens_evaluated')} "
        f"token_acc={report.get('token_accuracy')}"
    )

    for case in report.get("top_cases", []):
        print(
            f"[bad-case]   #{case['rank']} "
            f"loss={_format_finite(case.get('loss'), '.4f')} "
            f"conf={_format_finite(case.get('confidence'), '.3f')} "
            f"margin={_format_finite(case.get('margin'), '.3f')} "
            f"target='{case['target_text']}' "
            f"pred='{case['predicted_text']}' "
            f"category={case['category']} | "
            f"{case['context']}"
        )

    if report.get("top_cases_concentration_warning"):
        print(
            "[bad-case] top-N 案例集中在少数窗口："
            f"唯一窗口 {report.get('top_cases_unique_windows')} 个，"
            "单一窗口最高占比 "
            f"{_format_finite(report.get('top_cases_window_concentration'), '.1f', 100.0)}%"
            "；这些案例不是彼此独立的失败模式"
        )

    if report.get("top_confusions"):
        pairs = " ".join(
            f"'{_render_byte(target)}'→"
            f"'{_render_byte(predicted)}'({count})"
            for target, predicted, count in report[
                "top_confusions"
            ][:5]
        )
        print(f"[bad-case] 高频混淆对: {pairs}")

    def _stratified_text(rows) -> str:
        return " ".join(
            f"{row['name']}="
            f"{_format_finite(row.get('error_rate'), '.1f', 100.0)}%"
            f"({row.get('wrong')}/{row.get('tokens')})"
            for row in rows
        )

    position_rows = report.get("error_rate_by_position")
    if position_rows:
        print(
            "[bad-case] 按序列位置错误率: "
            + _stratified_text(position_rows)
        )

    frequency_rows = report.get(
        "error_rate_by_train_frequency"
    )
    if frequency_rows:
        basis = report.get("error_frequency_basis")
        basis_text = (
            "训练集目标占比"
            if basis == "train_target_share"
            else "评估集目标占比"
        )
        print(
            f"[bad-case] 按目标频率错误率（{basis_text}）: "
            + _stratified_text(frequency_rows)
        )

    if report.get("category_counts"):
        print(
            "[bad-case] top-N 错误类别统计: "
            + json.dumps(
                report["category_counts"],
                ensure_ascii=False,
            )
        )

    calibration = report.get("calibration")
    if isinstance(calibration, dict) and calibration.get(
        "ece"
    ) is not None:
        print(
            "[bad-case] 置信度校准: "
            f"ECE={_format_finite(calibration.get('ece'))} "
            "平均置信度="
            f"{_format_finite(calibration.get('mean_confidence'))} "
            "实际准确率="
            f"{_format_finite(calibration.get('accuracy'))} "
            "过度自信="
            f"{_format_finite(calibration.get('overconfidence'), '+.4f')} "
            "（高置信度错误 "
            f"{calibration.get('high_confidence_error_count', 0)}"
            " 个）"
        )

    verification = report.get(
        "label_noise_verification"
    )
    if isinstance(verification, dict):
        print(
            "[bad-case] 标签噪声核验: "
            f"top-N 候选 {verification['candidates']} 个，"
            f"训练上下文支持后确认 "
            f"{verification['verified']} 个，"
            f"未确认 {verification['unverified']} 个"
            "（其中 "
            f"{verification['unverifiable_no_train_context']}"
            " 个无训练上下文可核验）"
        )

    if report.get("error_category_counts"):
        shares = report.get("error_category_shares") or {}
        detail = ", ".join(
            f"{name}={count}"
            + (
                f"({_format_finite(shares.get(name), '.1f', 100.0)}%)"
                if shares.get(name) is not None
                else ""
            )
            for name, count in report[
                "error_category_counts"
            ].items()
        )
        print(
            "[bad-case] 全量错误归因"
            f"（共 {report.get('wrong_tokens')} 个错误 token）: "
            f"{detail}"
        )

    print(f"[bad-case] 小结: {report.get('summary')}")



# ============================================================
# Routing diagnostics (RQ3 / RQ4)
# ============================================================
#
# 研究问题 RQ3/RQ4 需要的不只是最终指标，而是训练后模型内部的
# 路由行为证据：第二轮是否选中与第一轮不同的专家（专家重合度 /
# Jaccard 相似度），以及专家负载分布与路由熵（专家专门化）。
# 本节的诊断在验证集的同一批固定窗口上、在验证集选定模型上做
# 一次无梯度前向，按 (Block, 轮次) 记录每次 route() 的 top-k
# 专家下标、门控权重与完整路由分布，汇总为结构化报告。
#
# 口径（随报告落盘，避免事后各自解释）：
# - 轮次 = 同一 Block 内 route() 的调用序号：第 1 次为第一轮，
#   第 2 次为第二轮。smelt 的两次中间循环、A/local_base 的
#   两阶段、B 的分支创建 + 第二轮完整中间区域都满足该顺序。
# - 专家重合度在「同一 token、同一 Block」的两轮路由之间比较；
#   分支结构（A/B）另外报告逐分支与 token 级（跨分支并集）口径。
# - 只使用验证集固定窗口；测试集保持尘封，不参与路由诊断。
# - 路由诊断只描述模型的内部行为，不参与模型选择、早停或任何
#   指标优化，避免把诊断本身变成调参信号。


def _routing_set_bool(
    indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """把 [N, k] 的专家下标转成 [N, n_experts] 的集合指示位。"""
    # 不能用原地 clamp_：.to(int64) 对已经是 int64 的输入返回
    # 同一张量，会改写调用方数据；对 expand 出来的 stride-0
    # 视图则直接抛「multiple elements refer to single memory」。
    indices = indices.to(torch.int64).clamp(
        min=0,
        max=n_experts - 1,
    )
    out = torch.zeros(
        indices.size(0),
        n_experts,
        dtype=torch.bool,
    )
    out.scatter_(1, indices, True)
    return out



def _routing_set_metrics_bool(
    a_bool: torch.Tensor,
    b_bool: torch.Tensor,
) -> Optional[Dict[str, float]]:
    """
    两组 [N, E] 专家集合指示位的逐 token 重合指标。

    - jaccard：|A∩B| / |A∪B|；
    - exact_match：两轮集合完全相同的 token 占比；
    - first_recall：|A∩B| / |A|，第一轮专家在第二轮复现的比例；
    - second_unique_reuse：|A∩B| / |B|，第二轮独特专家来自
      第一轮的比例。

    Jaccard 不能直接跨 $k_1/k_2$、专家数与负载集中度比较：
    均匀随机选择时两个 k 元集合的期望交集是 k1*k2/E。因此
    同时给出两个机会水平（chance level）：

    - ``jaccard_chance_uniform``：每个 token 按 |A|、|B| 与
      E 计算期望交集的均匀随机基线；
    - ``jaccard_chance_marginal``：第二轮的专家边际负载保持
      不变（P(e∈B)=第二轮负载份额），只随机化组合；这对应
      「保持负载分布、打乱路由」的置换基线。

    ``*_lift`` 为观测 Jaccard 相对机会水平的倍数，``*_adjusted``
    为 (obs − chance)/(1 − chance)。低重合本身不证明路由更
    有效，必须结合干预实验（见 analyze_routing_interventions）。
    """
    if a_bool.numel() == 0 or b_bool.numel() == 0:
        return None
    if a_bool.size(0) != b_bool.size(0):
        return None

    intersection = (a_bool & b_bool).sum(dim=-1).float()
    union = (a_bool | b_bool).sum(dim=-1).float()
    r1_size = a_bool.sum(dim=-1).float()
    r2_size = b_bool.sum(dim=-1).float()
    n_experts = int(a_bool.size(-1))

    jaccard = intersection / union.clamp_min(1e-12)
    jaccard_mean = float(jaccard.mean())

    # 均匀随机基线：E|A∩B| = |A|·|B|/E。
    expected_uniform = (
        r1_size * r2_size / max(1, n_experts)
    )
    union_uniform = (
        r1_size + r2_size - expected_uniform
    ).clamp_min(1e-12)
    chance_uniform = float(
        (expected_uniform / union_uniform).mean()
    )

    # 边际负载保持基线：P(e∈B) 取第二轮的经验负载份额，
    # 对每个 token 的 A 集合求和得到期望交集。该口径保留
    # 「某些专家本来就更常被选中」带来的重合，避免把负载
    # 集中误读成路由机制。
    b_load = b_bool.float().mean(dim=0)
    expected_marginal = (
        a_bool.float() * b_load.unsqueeze(0)
    ).sum(dim=-1)
    union_marginal = (
        r1_size + r2_size - expected_marginal
    ).clamp_min(1e-12)
    chance_marginal = float(
        (expected_marginal / union_marginal).mean()
    )

    def _adjusted(chance: float) -> Optional[float]:
        if chance >= 1.0:
            return None
        return (jaccard_mean - chance) / (1.0 - chance)

    def _lift(chance: float) -> Optional[float]:
        if chance <= 0.0:
            return None
        return jaccard_mean / chance

    return {
        "units": int(a_bool.size(0)),
        "second_round_jaccard_mean": jaccard_mean,
        "second_round_exact_match_share": float(
            (intersection == union).float().mean()
        ),
        "first_round_recall_mean": float(
            (intersection / r1_size.clamp_min(1e-12)).mean()
        ),
        "second_round_unique_reuse_mean": float(
            (intersection / r2_size.clamp_min(1e-12)).mean()
        ),
        "expected_intersection_uniform": float(
            expected_uniform.mean()
        ),
        "jaccard_chance_uniform": chance_uniform,
        "jaccard_chance_marginal": chance_marginal,
        "jaccard_lift_over_uniform": _lift(chance_uniform),
        "jaccard_lift_over_marginal": _lift(chance_marginal),
        "jaccard_adjusted_uniform": _adjusted(chance_uniform),
        "jaccard_adjusted_marginal": _adjusted(chance_marginal),
        "chance_note": (
            "uniform = 均匀随机集合（E|A∩B|=k1*k2/E）；"
            "marginal = 保持第二轮专家负载份额、随机化组合。"
            "低 Jaccard 必须先减去机会水平才可与其它 k1/k2 "
            "或负载分布比较"
        ),
    }



def _routing_set_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    n_experts: int,
) -> Optional[Dict[str, float]]:
    """[N, ka] / [N, kb] 专家下标对的集合重合指标（含 slot 口径）。"""
    if a.numel() == 0 or b.numel() == 0:
        return None
    if a.size(0) != b.size(0):
        return None

    a_bool = _routing_set_bool(a, n_experts)
    b_bool = _routing_set_bool(b, n_experts)
    metrics = _routing_set_metrics_bool(a_bool, b_bool)
    if metrics is None:
        return None

    # slot 口径：第二轮每个 assignment 的专家是否落在第一轮集合中
    # （重复专家按 slot 各计一次，反映实际计算里的复用比例）。
    b_slots = b.to(torch.int64).clamp(
        min=0, max=n_experts - 1
    )
    metrics["second_round_slot_reuse_mean"] = float(
        a_bool.float().gather(1, b_slots).mean()
    )
    return metrics
