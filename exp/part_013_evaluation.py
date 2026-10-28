from __future__ import annotations



# ============================================================
# Evaluation
# ============================================================

# 评估张量不会回流到训练图，此处使用 inference_mode 是安全的。
@torch.inference_mode()
def evaluate(
    model: LanguageModel,
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    split: str = "val",
    compute_auc: bool = False,
    routing_probe: Optional["RoutingProbe"] = None,
    bad_case_collector: Optional["_BadCaseCollector"] = None,
    bootstrap_replicates: int = 0,
    window_sink: Optional[List[Dict[str, float]]] = None,
    bootstrap_block: int = 1,
) -> Dict[str, float]:
    """
    在指定评估 split 上计算加权平均语言模型损失与
    类别不平衡感知的分类指标。

    split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    test 只在最终评估时调用一次；返回指标键带 split 前缀
    （val_loss/test_loss 等）。

    每次评估都基于同一批固定窗口累计混淆矩阵，并报告：

    - ``{split}_loss`` / ``{split}_ppl`` /
      ``{split}_bits_per_byte``：主指标（与 LM 研究目标
      对齐的似然口径）；
    - ``{split}_token_accuracy``（= micro 口径）与
      ``{split}_micro_f1``：单标签多分类下两者恒等；
    - ``{split}_macro_f1`` / ``{split}_weighted_f1``：
      必须注明平均口径，macro 只对评估集中出现过的类
      求平均；
    - ``{split}_precision_macro`` /
      ``{split}_precision_weighted`` /
      ``{split}_recall_macro`` /
      ``{split}_recall_weighted``：precision / recall 的
      macro（对出现过的类等权）与 weighted（按 support
      加权）口径；micro precision/recall 与 accuracy 恒等
      故不单列；
    - ``{split}_majority_accuracy``：常数预测器（永远
      猜最高频字节）的 accuracy，类别不平衡下解读
      accuracy 的参考下界。

    ``bootstrap_replicates > 0`` 时额外对固定评估窗口做
    有放回重采样，给出 loss / ppl / macro NLL / token accuracy /
    macro 与 weighted F1 / macro 与 weighted precision / recall
    的 95% 置信区间（``{split}_*_ci95_low/high``）。测试集
    只解封一次，点估计本身无法区分「方法差异」与「评估窗口
    抽样噪声」；置信区间是评估集层面的不确定性，与跨随机
    种子的训练方差互补，不参与训练或模型选择。假设以
    macro F1 / 召回率等分类指标判定时，区间与点估计同口径，
    不会只给 loss 不确定性而让分类结论裸奔。

    compute_auc=True 时（最终/测试评估）额外计算
    one-vs-rest 的 ``{split}_pr_auc_macro`` 与
    ``{split}_roc_auc_macro``，只纳入 support 达到
    _AUC_MIN_SUPPORT 的类；极度不平衡时优先看 PR-AUC，
    并与 ``{split}_pr_auc_macro_chance``（随机水平 =
    平均 prevalence）对照，而不是把 PR-AUC 绝对值当
    「准确率」解读。

    routing_probe 非 None 时，本次评估前向同时充当路由
    诊断的采样前向（探针在评估期间挂载、退出时卸载）：
    调用方在返回后用同一探针 build_report()；诊断只在
    val 上使用。

    window_sink 非 None 时，逐窗口记录（损失和、目标 token
    数、命中数）并追加到该列表：固定评估窗口在所有运行之间
    逐位相同，因此两个运行可以按窗口配对比较——这给出
    「在这批固定评估窗口上」的配对证据，与种子级检验
    （训练随机性）互补。
    """
    was_training = model.training
    model.eval()

    probe_attached = False
    if routing_probe is not None:
        routing_probe.attach(_unwrap_model(model))
        probe_attached = True

    use_amp = (
        amp_dtype is not None
        and device.type == "cuda"
    )

    # split 仅允许 val/test：val 用于训练期间的周期验证与选型，
    # test 只在最终评估时调用一次。

    vocab_size = _unwrap_model(model).cfg.vocab_size

    # 按目标 token 数加权累加。
    total = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    evaluated_tokens = 0

    # 混淆矩阵 [目标类, 预测类]，在评估设备上以 int64 累积，
    # 收尾时转 fp64。
    confusion = torch.zeros(
        (vocab_size, vocab_size),
        device=device,
        dtype=torch.int64,
    )

    # AUC 需要全部位置的预测概率；只在最终/测试评估时收集，
    # 并按 _AUC_MAX_TOKENS 限制。收集缓冲按固定窗口的实际
    # 位置数分配，逐批写入。
    auc_probs: Optional[torch.Tensor] = None
    auc_targets: Optional[torch.Tensor] = None
    auc_tokens = 0

    # 评估窗口级 bootstrap 的逐窗口充分统计量：窗口损失和、
    # token 数，以及每类 TP / 预测数 / support。只在最终与
    # 测试评估时收集，周期验证保持轻量。
    bootstrap_enabled = int(bootstrap_replicates) > 1
    window_loss_sums: List[torch.Tensor] = []
    window_tokens: List[torch.Tensor] = []
    window_true_positive: List[torch.Tensor] = []
    window_predicted: List[torch.Tensor] = []
    window_support: List[torch.Tensor] = []
    window_class_loss_sums: List[torch.Tensor] = []

    # 类平衡似然（macro NLL）的逐类充分统计量：按目标类累计
    # 损失和与 token 数。它与频率加权的 total 累加共用同一次
    # 前向的 logits，不额外跑模型。
    class_loss_sums = torch.zeros(
        vocab_size,
        device=device,
        dtype=torch.float64,
    )
    class_counts = torch.zeros(
        vocab_size,
        device=device,
        dtype=torch.float64,
    )

    # CUDA 路径在全部批次结束后统一同步；CPU 路径保持同步语义。
    auc_async = compute_auc and device.type == "cuda"

    auc_total_positions = 0
    auc_sample_positions: Optional[torch.Tensor] = None
    if compute_auc:
        _, eval_data = dataset._eval_windows(split)
        num_starts = len(eval_data) - dataset.seq_len
        auc_total_positions = (
            min(batches * batch_size, num_starts)
            * dataset.seq_len
        )
        auc_capacity = min(
            _AUC_MAX_TOKENS,
            auc_total_positions,
        )
        auc_sample_positions = _auc_uniform_sample_positions(
            auc_total_positions,
            auc_capacity,
        )
        auc_probs = torch.empty(
            auc_capacity,
            vocab_size,
            dtype=torch.float32,
            pin_memory=auc_async,
        )
        auc_targets = torch.empty(
            auc_capacity,
            dtype=torch.int64,
            pin_memory=auc_async,
        )
        auc_processed_positions = 0

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

                # 请求窗口超过验证集合法起点时，后继批次为空。
                if x.size(0) == 0:
                    break

                result = model(x, y)

                # argmax 直接取模型输出 dtype（bf16/fp16 -> fp32
                # 是精确升位，不改变数值与并列顺序）；AUC 的
                # softmax 仍在 fp32 下统计，避免半精度数值噪声
                # 影响逐类指标。
                raw_logits = result.pop("logits")

                # 坏案例收集复用同一次前向（仅最终评估传入）；
                # 不影响下面任何指标的计算与数值。
                if bad_case_collector is not None:
                    bad_case_collector.update(x, y, raw_logits)

                predicted = raw_logits.argmax(dim=-1)
                flat_target = y.reshape(-1).to(
                    torch.int64
                )
                flat_predicted = predicted.reshape(-1)

                # add(alpha=) 单次完成 target * vocab_size +
                # predicted（逐位等值）；flat_target 保持原值供
                # AUC 收集使用。
                confusion += torch.bincount(
                    torch.add(
                        flat_predicted,
                        flat_target,
                        alpha=vocab_size,
                    ),
                    minlength=vocab_size ** 2,
                ).view(
                    vocab_size,
                    vocab_size,
                )

                # 逐 token 损失：类平衡 NLL（macro_nll）与评估
                # 窗口 bootstrap 的逐窗口统计量共用它，不额外跑
                # 模型。logits 就是本批前向的输出，upcast 到
                # fp32 与 loss 计算口径一致。
                token_logits = raw_logits.reshape(
                    -1,
                    vocab_size,
                ).float()
                token_losses = F.cross_entropy(
                    token_logits,
                    flat_target,
                    reduction="none",
                )

                # 类平衡似然的逐类充分统计量：按目标类累计
                # 损失和与 token 数（fp64 累加，避免高频类的
                # 求和误差污染每类 NLL）。
                class_loss_sums.index_add_(
                    0,
                    flat_target,
                    token_losses.to(torch.float64),
                )
                class_counts.index_add_(
                    0,
                    flat_target,
                    torch.ones_like(
                        token_losses,
                        dtype=torch.float64,
                    ),
                )

                if window_sink is not None:
                    # 逐窗口统计：与整批口径同一份 logits 与
                    # 预测结果（不额外跑模型），供跨运行按窗口
                    # 配对比较使用。
                    window_losses = token_losses.view(
                        y.shape
                    ).sum(dim=1)
                    window_correct = (
                        predicted.to(torch.int64)
                        == flat_target.view(y.shape)
                    ).sum(dim=1)
                    loss_list = window_losses.detach().tolist()
                    correct_list = window_correct.detach().tolist()
                    window_token_count = int(y.size(1))
                    for row_index in range(y.size(0)):
                        window_sink.append({
                            "loss_sum": float(loss_list[row_index]),
                            "tokens": window_token_count,
                            "correct": int(correct_list[row_index]),
                        })

                if bootstrap_enabled:
                    # 逐窗口 loss 与混淆充分统计量：用与点估计
                    # 同一前向的 logits 计算，不额外跑模型。
                    window_loss_sums.append(
                        token_losses.view(y.shape)
                        .sum(dim=1)
                        .detach()
                        .to(torch.float64)
                        .cpu()
                    )
                    window_tokens.append(
                        torch.full(
                            (y.size(0),),
                            float(y.size(1)),
                            dtype=torch.float64,
                        )
                    )
                    for row in range(y.size(0)):
                        target_row = y[row].to(torch.int64)
                        predicted_row = predicted[row].to(
                            torch.int64
                        )
                        correct = target_row == predicted_row
                        window_true_positive.append(
                            torch.bincount(
                                target_row[correct],
                                minlength=vocab_size,
                            ).cpu()
                        )
                        window_predicted.append(
                            torch.bincount(
                                predicted_row,
                                minlength=vocab_size,
                            ).cpu()
                        )
                        row_support = torch.bincount(
                            target_row,
                            minlength=vocab_size,
                        )
                        window_support.append(row_support.cpu())
                        # 逐类损失和：供 macro NLL 的窗口级
                        # bootstrap 使用，与点估计共用同一批
                        # token 损失，不额外跑前向。
                        row_class_loss = torch.zeros(
                            vocab_size,
                            dtype=torch.float64,
                            device=device,
                        ).index_add_(
                            0,
                            target_row,
                            token_losses.view(y.shape)[
                                row
                            ].to(torch.float64),
                        )
                        window_class_loss_sums.append(
                            row_class_loss.cpu()
                        )

                if compute_auc:
                    flat_logits = raw_logits.reshape(
                        -1,
                        vocab_size,
                    )
                    if auc_sample_positions is None:
                        selected_logits = flat_logits
                        selected_targets = flat_target
                    else:
                        # 只对全局等距抽中的位置做 softmax：
                        # 两侧都按同一全局位置切分，避免把
                        # 后段评估窗口系统性排除。
                        start_position = (
                            auc_processed_positions
                        )
                        end_position = (
                            start_position
                            + flat_logits.size(0)
                        )
                        lower = int(
                            torch.searchsorted(
                                auc_sample_positions,
                                start_position,
                            )
                        )
                        upper = int(
                            torch.searchsorted(
                                auc_sample_positions,
                                end_position,
                            )
                        )
                        local_positions = (
                            auc_sample_positions[lower:upper]
                            - start_position
                        ).to(flat_logits.device)
                        selected_logits = flat_logits[
                            local_positions
                        ]
                        selected_targets = flat_target[
                            local_positions
                        ]
                    num_new = int(selected_logits.size(0))
                    # 按行分块：逐行 softmax 与整批等价。
                    for start in range(
                        0,
                        num_new,
                        _AUC_SOFTMAX_CHUNK_ROWS,
                    ):
                        stop = min(
                            start + _AUC_SOFTMAX_CHUNK_ROWS,
                            num_new,
                        )
                        probs = F.softmax(
                            selected_logits[
                                start:stop
                            ].float(),
                            dim=-1,
                        )
                        # 拷入 CPU 收集缓冲，循环结束统一同步。
                        auc_probs[
                            auc_tokens + start:
                            auc_tokens + stop
                        ].copy_(
                            probs,
                            non_blocking=auc_async,
                        )
                    auc_targets[
                        auc_tokens:auc_tokens + num_new
                    ].copy_(
                        selected_targets,
                        non_blocking=auc_async,
                    )
                    auc_tokens += num_new
                    auc_processed_positions += int(
                        flat_logits.size(0)
                    )

                # cross_entropy 返回批均值，乘回目标 token 数。
                # add_(alpha=) 完成「乘 token 数 + 累加」。
                total.add_(
                    result["lm_loss"].float(),
                    alpha=float(y.numel()),
                )
                evaluated_tokens += y.numel()
                del result, raw_logits, x, y
    finally:
        # 无论评估是否中途出错，都卸载探针并恢复原来的训练模式。
        if probe_attached:
            routing_probe.detach()
        model.train(was_training)

    # ByteDataset 保证至少有一个合法评估起点。
    loss = (total / evaluated_tokens).item()

    metrics = {
        f"{split}_loss": loss,
        f"{split}_ppl": math.exp(min(loss, 80.0)),
        f"{split}_bits_per_byte": loss / math.log(2.0),
    }

    class_metrics = confusion_matrix_metrics(confusion.double())

    for key, value in class_metrics.items():
        metrics[f"{split}_{key}"] = value

    # 类平衡似然口径：与频率加权的 {split}_loss 同单位（自然
    # 对数似然），但对评估集中出现的类等权，稀有字节不再被
    # 高频字节淹没。类别不平衡场景下它才是「有没有真的学会
    # 稀有字节」的似然口径。
    balanced_metrics = class_balanced_loss_metrics(
        class_loss_sums,
        class_counts,
        vocab_size,
    )
    for key, value in balanced_metrics.items():
        metrics[f"{split}_{key}"] = value

    if bootstrap_enabled and window_loss_sums:
        bootstrap_metrics = _bootstrap_eval_intervals(
            torch.cat(window_loss_sums),
            torch.cat(window_tokens),
            torch.stack(window_true_positive),
            torch.stack(window_predicted),
            torch.stack(window_support),
            int(bootstrap_replicates),
            _eval_bootstrap_seed(
                dataset.data_sha256,
                split,
                batches,
                batch_size,
                dataset.seq_len,
            ),
            window_class_loss_sums=torch.stack(
                window_class_loss_sums
            ),
            block_size=int(bootstrap_block),
        )
        for key, value in bootstrap_metrics.items():
            metrics[f"{split}_{key}"] = value

        # 评估窗口可能因 split 较短而重叠：重叠窗口不是独立
        # 抽样单位，bootstrap 区间会偏窄。把重叠比例随指标
        # 落盘，报告可据此判断区间是否被高估。
        starts = (
            dataset._fixed_starts.get(split, {}).get("starts")
        )
        if starts is not None and len(starts) > 1:
            gaps = starts[1:] - starts[:-1]
            overlap_count = int(
                (gaps < dataset.seq_len).sum().item()
            )
            metrics[f"{split}_eval_windows"] = int(len(starts))
            metrics[f"{split}_eval_window_overlap_fraction"] = (
                overlap_count / (len(starts) - 1)
            )

    if compute_auc:
        # 读取 CPU 收集缓冲前统一等待。
        if auc_async and auc_tokens > 0:
            torch.cuda.synchronize(device)

        metrics[f"{split}_auc_tokens_available"] = (
            auc_total_positions
        )
        metrics[f"{split}_auc_coverage_fraction"] = (
            auc_tokens / auc_total_positions
            if auc_total_positions > 0
            else 0.0
        )
        metrics[f"{split}_auc_sampling"] = (
            "uniform_stride"
            if auc_sample_positions is not None
            else "full"
        )

        if auc_tokens > 0:
            auc_metrics = macro_auc_metrics(
                auc_probs[:auc_tokens],
                auc_targets[:auc_tokens],
                vocab_size,
            )
            for key, value in auc_metrics.items():
                metrics[f"{split}_{key}"] = value
        else:
            # 空评估（理论上不可达）：显式记录而不是
            # 静默缺键。
            metrics[f"{split}_pr_auc_macro"] = None
            metrics[f"{split}_pr_auc_macro_chance"] = None
            metrics[f"{split}_pr_auc_macro_lift"] = None
            metrics[f"{split}_roc_auc_macro"] = None
            metrics[f"{split}_auc_n_classes"] = 0
            metrics[f"{split}_pr_auc_n_classes"] = 0
            metrics[f"{split}_roc_auc_n_classes"] = 0
            metrics[f"{split}_auc_min_support"] = (
                _AUC_MIN_SUPPORT
            )
            metrics[f"{split}_auc_tokens_evaluated"] = 0

    return metrics



# ============================================================
# Trivial reference baselines (uniform / unigram)
# ============================================================

@torch.inference_mode()
def compute_reference_baselines(
    dataset: ByteDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    split: str = "val",
) -> Dict[str, object]:
    """
    在与模型评估完全相同的固定窗口上计算两个「稻草人之下」
    的参考基线，作为解读指标的坐标系：

    - uniform：每个字节等概率 1/256，NLL = ln(256)，是
      任何模型必须显著低于的绝对下限；
    - unigram：只用 train split 的字节频率做无上下文预测
      （Laplace 平滑，统计量只在 train 上 fit），对应
      「类别不平衡下常数/多数类预测器」的视角——它的
      token accuracy 已经有相当份额，单看 accuracy 无法
      判断模型是否真的学到了上下文信息。

    返回 uniform / unigram 在指定 split 上的 loss、
    accuracy、macro/weighted F1、macro/weighted
    precision/recall 与多数类参考 accuracy，以及 AUC 类
    指标的常数预测器水平（PR-AUC 随机水平 = 纳入类平均
    prevalence，ROC-AUC = 0.5），随 config.json /
    metrics.jsonl 存档。训练曲线诊断把「是否学到东西」的
    欠拟合参照从 uniform 换成更强的 unigram，避免只战胜
    稻草人。
    """
    # 字节级管线的词表固定为 256。
    vocab_size = 256

    # unigram 统计量只在 train 上 fit，且与神经网络窗口采样
    # 使用同一「有效训练字节」口径：被群组去重或字节级标签噪声
    # 清洗掉的内容不能进入基线计数，否则基线会通过重复/损坏
    # 内容看到不同的训练分布，比较不再公平。
    train_counts = dataset.effective_train_byte_counts().double()
    train_total = float(train_counts.sum())
    raw_train_total = len(dataset.train)

    unigram_probs = (
        train_counts + 1.0
    ) / (train_total + vocab_size)
    uniform_probs = torch.full(
        (vocab_size,),
        1.0 / vocab_size,
        dtype=torch.float64,
    )

    unigram_prediction = int(train_counts.argmax())

    # log 概率预先算好（词表仅 256 项），循环内只做查表求和：
    # 与逐批对整批 token 求 log 的取值逐位一致。
    uniform_log_probs = uniform_probs.log()
    unigram_log_probs = unigram_probs.log()

    uniform_total = 0.0
    unigram_total = 0.0
    evaluated_tokens = 0

    confusion = torch.zeros(
        (vocab_size, vocab_size),
        dtype=torch.float64,
    )

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

        flat_target = y.reshape(-1).to(torch.int64)

        uniform_total -= float(
            uniform_log_probs[flat_target].sum()
        )
        unigram_total -= float(
            unigram_log_probs[flat_target].sum()
        )
        evaluated_tokens += flat_target.numel()

        # unigram 预测器对所有位置输出同一个字节。
        predicted = torch.full_like(
            flat_target,
            unigram_prediction,
        )
        confusion += torch.bincount(
            flat_target * vocab_size + predicted,
            minlength=vocab_size ** 2,
        ).view(vocab_size, vocab_size).double()

    loss_uniform = uniform_total / evaluated_tokens
    loss_unigram = unigram_total / evaluated_tokens

    class_metrics = confusion_matrix_metrics(confusion)

    # PR-AUC / ROC-AUC 的常数预测器参考：unigram 对所有位置输出
    # 同一个（逐类不同的）分布，one-vs-rest 排序分数逐类恒定，
    # 因此
    # - PR-AUC 恰好等于该类在评估集上的 prevalence（平均精度
    #   在全部并列时 = 正类比例），macro 随机水平 = 纳入类
    #   （support >= _AUC_MIN_SUPPORT）prevalence 的均值；
    # - ROC-AUC 恒为 0.5（并列分数无排序信息）。
    # 只对有序指标的类（0 < support < 评估 token 数）定义
    # ROC 参考，与 macro_auc_metrics 的纳入口径一致。这样在
    # 按项目建议切换到 PR-AUC 对比时，稻草人核对不会静默失效。
    eval_support = confusion.sum(dim=1)
    eval_tokens = float(eval_support.sum())
    auc_eligible = eval_support >= _AUC_MIN_SUPPORT
    pr_auc_chance = None
    roc_auc_reference = None
    if eval_tokens > 0.0:
        if bool(auc_eligible.any()):
            pr_auc_chance = float(
                (eval_support[auc_eligible] / eval_tokens).mean()
            )
        roc_eligible = auc_eligible & (eval_support > 0.0) & (
            eval_support < eval_tokens
        )
        if bool(roc_eligible.any()):
            roc_auc_reference = 0.5

    # 类平衡（macro）NLL 参考：与 evaluate() 的 val_macro_nll
    # 同口径（对评估集中出现的类等权平均每类 NLL）。
    # - uniform：每类 NLL 恒为 log 256，类平衡平均后不变；
    # - unigram：预测器与上下文无关，每类 NLL 恒为 -log p(c)，
    #   类平衡平均即出现类上的平均 -log p(c)。
    # 这两个数是「类平衡似然」的稻草人下界：基线合理性检查在
    # val_macro_nll 上用它核对基线是否真的强于无上下文预测器。
    present_classes = eval_support > 0.0
    uniform_macro_nll = float(-uniform_log_probs.mean())
    if bool(present_classes.any()):
        unigram_macro_nll = float(
            (-unigram_log_probs[present_classes]).mean()
        )
    else:
        unigram_macro_nll = None

    return {
        "split": split,
        "tokens_evaluated": evaluated_tokens,
        "uniform_loss": loss_uniform,
        "unigram_loss": loss_unigram,
        "unigram_train_bytes": int(train_total),
        "unigram_train_bytes_raw": int(raw_train_total),
        "unigram_train_excluded_bytes": int(
            raw_train_total - train_total
        ),
        "majority_byte": unigram_prediction,
        "unigram_token_accuracy": class_metrics[
            "token_accuracy"
        ],
        "unigram_majority_accuracy": class_metrics[
            "majority_accuracy"
        ],
        "unigram_macro_f1": class_metrics["macro_f1"],
        "unigram_weighted_f1": class_metrics[
            "weighted_f1"
        ],
        # precision / recall：与 evaluate 同口径（macro 对
        # support > 0 的类等权、weighted 按 support 加权），
        # 供假设以「召回率/精确率」判定时核对基线是否强于
        # 无上下文预测器。
        "unigram_precision_macro": class_metrics[
            "precision_macro"
        ],
        "unigram_precision_weighted": class_metrics[
            "precision_weighted"
        ],
        "unigram_recall_macro": class_metrics[
            "recall_macro"
        ],
        "unigram_recall_weighted": class_metrics[
            "recall_weighted"
        ],
        # 序数指标（AUC）的常数预测器参考（见上方注释）。
        "unigram_pr_auc_macro_chance": pr_auc_chance,
        # 常数预测器的 PR-AUC lift = prevalence - prevalence = 0；
        # 显式落盘，基线守卫在 lift 口径上有同源参考。
        "unigram_pr_auc_macro_lift": 0.0,
        "unigram_roc_auc_macro": roc_auc_reference,
        # 类平衡似然参考（macro NLL）：与 val_macro_nll 同口径。
        "uniform_macro_nll": uniform_macro_nll,
        "unigram_macro_nll": unigram_macro_nll,
        "macro_nll_classes_present": float(
            int(present_classes.sum())
        ),
    }



# ============================================================
# N-gram statistical baseline (最简基线，先于任何神经模型)
# ============================================================
#
# 方法论要求：复杂模型必须显著战胜「精心调优过的简单基线」
# 才有说服力。本节实现字节级 n-gram 统计基线：
#
# - 计数只在 train split 上做（无泄露），评估用与神经模型
#   完全相同的固定窗口协议；
# - 插值平滑 P_o(c|ctx) =
#       (count(ctx,c) + α·P_{o-1}(c|ctx')) / (count(ctx) + α)，
#   一路回退到 uniform；阶数 o 与平滑系数 α 在 val 上
#   网格调优（简单基线同样要充分调优，而不是随便跑一档）；
# - test 只在显式 --evaluate-test 时用 val 上选出的
#   (o, α) 评估一次，绝不在 test 上调参。

# 上下文 + 目标按字节精确打包进 int64，7 个字节（56 bit）
# 是不越界的上限；更大的阶数需要哈希，超出即报错。
_NGRAM_MAX_ORDER = 7


# 每阶计数使用的最大训练窗口数；超过时按等距步长抽样
#（抽样是确定性的）。
_NGRAM_COUNT_CAP = 4_000_000



class NgramBaseline:
    """
    字节级插值 n-gram 基线（含 unigram）。

    每个阶 o 存三张有序表（torch.unique 保证升序）：
    - pair_keys / pair_counts：(ctx, c) 对的键
      （ctx*256 + c）与计数；
    - ctx_uniq / ctx_counts：ctx 键与出现次数。

    键按升序排列使得同一 ctx 的所有 (ctx, c) 对在
    pair_keys 中连续，查询用 searchsorted 完成。
    """

    def __init__(
        self,
        train_bytes: torch.Tensor,
        max_order: int,
        count_cap: int = _NGRAM_COUNT_CAP,
        exclude_spans: Optional[List[Tuple[int, int]]] = None,
    ):
        if not 1 <= max_order <= _NGRAM_MAX_ORDER:
            raise ValueError(
                "n-gram 阶数必须落在 "
                f"[1, {_NGRAM_MAX_ORDER}]（字节精确打包"
                "的 int64 上限）"
            )

        self.max_order = max_order
        self.vocab_size = 256
        self.exclude_spans = list(exclude_spans or [])

        data = train_bytes
        n_train = data.numel()

        self.levels: Dict[int, Dict[str, torch.Tensor]] = {}

        def _empty_level() -> Dict[str, torch.Tensor]:
            return {
                "pair_keys": torch.empty(
                    0,
                    dtype=torch.int64,
                ),
                "pair_counts": torch.empty(
                    0,
                    dtype=torch.int64,
                ),
                "ctx_uniq": torch.empty(
                    0,
                    dtype=torch.int64,
                ),
                "ctx_counts": torch.empty(
                    0,
                    dtype=torch.int64,
                ),
            }

        for order in range(1, max_order + 1):
            if n_train < order:
                # 语料比阶数还短：该阶没有可计数的窗口，
                # 查询时自然回退到低阶。
                self.levels[order] = _empty_level()
                continue

            num_windows = n_train - order + 1
            valid_intervals = _valid_window_start_intervals(
                num_windows,
                order - 1,
                self.exclude_spans,
            )

            if valid_intervals is None:
                stride = max(
                    1,
                    math.ceil(num_windows / count_cap),
                )

                # data.unfold(0, order, 1) 是
                # [num_windows, order] 的滑动窗口视图；
                # 按步长取行仍是视图，打包时才产生 int64
                # 张量。
                windows = data.unfold(0, order, 1)[::stride]
            elif not valid_intervals:
                self.levels[order] = _empty_level()
                continue
            else:
                # 与神经网络训练窗口同一口径：完全落在被
                # 清洗群组之外的窗口才参与计数，避免统计
                # 基线通过重复群组看到 val/test。
                lengths = torch.tensor(
                    [
                        high - low + 1
                        for low, high in valid_intervals
                    ],
                    dtype=torch.int64,
                )
                total_valid = int(lengths.sum())
                stride = max(
                    1,
                    math.ceil(total_valid / count_cap),
                )
                global_index = torch.arange(
                    0,
                    total_valid,
                    stride,
                    dtype=torch.int64,
                )
                cumulative = torch.cumsum(lengths, dim=0)
                interval = torch.searchsorted(
                    cumulative,
                    global_index,
                    right=True,
                )
                offsets = global_index - (
                    cumulative[interval] - lengths[interval]
                )
                lows = torch.tensor(
                    [low for low, _ in valid_intervals],
                    dtype=torch.int64,
                )
                starts = lows[interval] + offsets
                windows = data.unfold(0, order, 1)[starts]

            keys = windows[:, 0].to(torch.int64)
            for column in range(1, order):
                keys = keys * 256 + windows[
                    :, column
                ].to(torch.int64)

            pair_keys, pair_counts = torch.unique(
                keys,
                return_counts=True,
            )

            ctx_keys = pair_keys // 256

            ctx_uniq, inverse = torch.unique(
                ctx_keys,
                return_inverse=True,
            )
            ctx_counts = torch.zeros(
                ctx_uniq.numel(),
                dtype=torch.int64,
            ).scatter_add_(
                0,
                inverse,
                pair_counts,
            )

            self.levels[order] = {
                "pair_keys": pair_keys,
                "pair_counts": pair_counts,
                "ctx_uniq": ctx_uniq,
                "ctx_counts": ctx_counts,
            }

        # unigram 总计数（level 1 的 ctx 计数），用于
        # 参考/归一。
        first = self.levels[1]
        if first["ctx_uniq"].numel():
            self.train_total = int(
                first["ctx_counts"][0]
            )
        else:
            self.train_total = 0

    def _pack_context(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        [n, ctx_len] 字节矩阵打包成 [n] 的 int64 键。
        """
        keys = context[:, 0].to(torch.int64)
        for column in range(1, context.size(1)):
            keys = keys * 256 + context[
                :, column
            ].to(torch.int64)
        return keys

    def _valid_contexts(
        self,
        windows: torch.Tensor,
        order: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """
        返回给定阶数下每个有效位置的 (ctx 键, 扁平位置)。

        目标 y[t] = w[t+1]；长度 order-1 的上下文需要
        t+1-(order-1) >= 0，即 t >= order-2。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        if order == 1:
            return (
                torch.zeros(
                    n_positions,
                    dtype=torch.int64,
                ),
                # 位置索引只需覆盖 batch*seq_len（远小于 2^31），
                # 用 int32 存储。
                torch.arange(
                    n_positions,
                    dtype=torch.int32,
                ),
                n_positions,
            )

        context_len = order - 1

        if context_len > seq_len:
            # 阶数高于窗口可提供的最大上下文长度：没有任何
            # 有效位置，调用方将整阶回退到低阶模型；此处直接
            # 返回空结果。
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty, 0

        t_min = context_len - 1
        n_valid = seq_len - t_min

        # windows.unfold(1, ctx_len, 1) 的第 j 列是
        # w[:, j : j+ctx_len]；目标 t 对应 j = t+1-ctx_len，
        # j 取值范围 [0, seq_len - ctx_len] = [0, n_valid-1]。
        contexts = windows.unfold(
            1,
            context_len,
            1,
        )[:, :n_valid].reshape(-1, context_len)

        batch_index = torch.arange(
            batch,
            dtype=torch.int32,
        ).repeat_interleave(n_valid)
        offsets = (
            torch.arange(n_valid, dtype=torch.int32)
            + t_min
        ).repeat(batch)

        positions = batch_index * seq_len + offsets

        return (
            self._pack_context(contexts),
            positions,
            n_valid,
        )

    @staticmethod
    def _lookup(
        sorted_keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        在有序键值表中查询，未命中返回 0。
        """
        if sorted_keys.numel() == 0:
            return torch.zeros_like(queries)

        locations = torch.searchsorted(
            sorted_keys,
            queries,
        )
        locations = locations.clamp(
            max=sorted_keys.numel() - 1,
        )
        found = sorted_keys[locations] == queries
        return torch.where(
            found,
            values[locations],
            torch.zeros_like(values[locations]),
        )

    def _per_level_tables(
        self,
        windows: torch.Tensor,
    ) -> List[Optional[Dict[str, torch.Tensor]]]:
        """
        逐阶的 alpha 无关查表结果（ctx 计数与 (ctx, c) 对计数）。

        查表只依赖窗口与训练计数，与平滑系数 alpha 无关；
        positions 为空（窗口比阶还短）的阶记为 None，由插值
        阶段整阶保持低阶回退值。
        """
        targets = windows[
            :, 1:
        ].reshape(-1).to(torch.int64)

        tables: List[
            Optional[Dict[str, torch.Tensor]]
        ] = []

        for order in range(1, self.max_order + 1):
            level = self.levels[order]

            ctx_keys, positions, _ = (
                self._valid_contexts(windows, order)
            )

            if positions.numel() == 0:
                tables.append(None)
                continue

            n_ctx = self._lookup(
                level["ctx_uniq"],
                level["ctx_counts"],
                ctx_keys,
            ).double()

            pair_keys = ctx_keys * 256 + targets[
                positions
            ]
            count_pair = self._lookup(
                level["pair_keys"],
                level["pair_counts"],
                pair_keys,
            ).double()

            tables.append({
                "positions": positions,
                "n_ctx": n_ctx,
                "count_pair": count_pair,
            })

        return tables

    def _interpolate_log_probs(
        self,
        windows: torch.Tensor,
        tables: List[
            Optional[Dict[str, torch.Tensor]]
        ],
        alpha: float,
    ) -> torch.Tensor:
        """
        用 _per_level_tables 的查表结果按指定 alpha 计算逐阶
        目标 log 概率，语义与逐次查表版本完全一致。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        log_probs = torch.zeros(
            (self.max_order, n_positions),
            dtype=torch.float64,
        )

        # level 0 的目标概率：uniform。
        probability = torch.full(
            (n_positions,),
            1.0 / self.vocab_size,
            dtype=torch.float64,
        )

        for order in range(1, self.max_order + 1):
            table = tables[order - 1]

            row = log_probs[order - 1]

            if table is None:
                # 窗口比阶还短：整阶保持低阶回退值。
                if order > 1:
                    row.copy_(log_probs[order - 2])
                continue

            positions = table["positions"]

            if order > 1:
                # 上下文不足的位置保持低阶值：
                # 先整行复制，再覆写本阶的有效位置，
                # 与「按掩码只复制无效位置」结果一致。
                row.copy_(log_probs[order - 2])

            interpolated = (
                table["count_pair"]
                + alpha * probability[positions]
            ) / (table["n_ctx"] + alpha)

            row[positions] = interpolated.log()

            probability[positions] = interpolated

        return log_probs

    def target_log_probs_per_level(
        self,
        windows: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        对一批窗口 [batch, seq_len + 1] 计算每个阶数
        （1..max_order）下目标字节的 log 概率。

        返回 [max_order, batch*seq_len]：第 o-1 行是
        「最高可用阶不超过 o」的回退模型在每个位置目标
        字节的 log 概率——上下文不足或从未见过的位置
        自动回退到低阶，供网格调优一次性拿到所有候选
        阶数的损失。
        """
        return self._interpolate_log_probs(
            windows,
            self._per_level_tables(windows),
            alpha,
        )

    def position_distributions(
        self,
        windows: torch.Tensor,
        alpha: float,
        order: int,
    ) -> torch.Tensor:
        """
        返回「最高可用阶不超过 order」的回退模型在每个
        位置的完整 [batch*seq_len, vocab] 概率分布，
        用于最终评估的 argmax / F1 / AUC 指标。
        """
        batch, total_len = windows.shape
        seq_len = total_len - 1
        n_positions = batch * seq_len

        probabilities = torch.full(
            (n_positions, self.vocab_size),
            1.0 / self.vocab_size,
            dtype=torch.float64,
        )

        for current in range(1, order + 1):
            level = self.levels[current]

            ctx_keys, positions, _ = (
                self._valid_contexts(
                    windows,
                    current,
                )
            )

            if positions.numel() == 0:
                continue

            n_ctx = self._lookup(
                level["ctx_uniq"],
                level["ctx_counts"],
                ctx_keys,
            ).double()

            if n_ctx.numel() and (
                level["pair_keys"].numel() == 0
            ):
                # 该阶没有任何计数：分布保持低阶回退。
                continue

            denominator = n_ctx + alpha

            updated = probabilities[
                positions
            ] * (alpha / denominator.unsqueeze(1))

            # 同一 ctx 的 (ctx, c) 对在升序 pair_keys 中
            # 连续：散射该 ctx 的全部非零计数行。
            starts = torch.searchsorted(
                level["pair_keys"],
                ctx_keys * 256,
            )
            ends = torch.searchsorted(
                level["pair_keys"],
                (ctx_keys + 1) * 256,
            )
            lengths = (ends - starts).clamp(min=0)

            total_nnz = int(lengths.sum())

            if total_nnz > 0:
                row_of = torch.repeat_interleave(
                    torch.arange(
                        positions.numel(),
                        dtype=torch.int64,
                    ),
                    lengths,
                )
                offsets = torch.cumsum(lengths, 0) - (
                    lengths
                )
                intra = torch.arange(total_nnz) - (
                    offsets[row_of]
                )
                flat_index = starts[row_of] + intra

                classes = level["pair_keys"][
                    flat_index
                ] % 256
                counts = level["pair_counts"][
                    flat_index
                ].double()

                updated.index_put_(
                    (row_of, classes),
                    counts
                    / denominator[row_of],
                    accumulate=True,
                )

            probabilities[positions] = updated

        return probabilities



def _ngram_eval_batches(
    dataset: ByteDataset,
    split: str,
    batch_size: int,
    batches: int,
) -> List[torch.Tensor]:
    """
    收集指定评估 split 的全部固定窗口（CPU uint8，
    [batch, seq_len + 1]）。
    """
    cpu = torch.device("cpu")
    windows: List[torch.Tensor] = []

    for batch_index in range(batches):
        x, y = dataset.fixed_eval_batch(
            split,
            batch_index,
            batches,
            batch_size,
            cpu,
        )

        if x.size(0) == 0:
            break

        windows.append(
            torch.cat(
                [
                    x[:, :1].to(torch.uint8),
                    y,
                ],
                dim=1,
            )
        )

    return windows
