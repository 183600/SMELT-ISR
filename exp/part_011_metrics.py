from __future__ import annotations



def print_data_audit(report: Dict[str, object]) -> None:
    """打印数据审计摘要与全部警告。"""
    splits = report.get("splits", {})
    parts = [
        (
            f"{name}={info['bytes']}B"
            + (
                f"(有效 {info['effective_bytes']}B)"
                if info.get("effective_bytes")
                != info.get("bytes")
                else ""
            )
        )
        for name, info in splits.items()
    ]
    print("[data] 数据审计: " + ", ".join(parts))

    isolation = report.get("split_isolation")
    if isinstance(isolation, dict):
        print(
            "[data]   时序隔离: "
            + " < ".join(isolation.get("split_order") or [])
            + "，bounds="
            + ", ".join(
                f"{name}={bounds[0]}..{bounds[1]}"
                for name, bounds in (
                    isolation.get("bounds") or {}
                ).items()
            )
            + (
                "，连续且无重叠"
                if isolation.get("contiguous")
                and int(isolation.get("overlap_bytes", 0)) == 0
                else "，边界异常"
            )
        )

    fractions = report.get("dataset_split_fractions") or {}
    if fractions:
        print(
            "[data]   实际 split 占比: "
            + ", ".join(
                f"{name}:{_format_finite(value)}"
                for name, value in fractions.items()
            )
        )

    groups = report.get("groups")
    if isinstance(groups, dict):
        per_split = groups.get("groups_per_split") or {}
        print(
            "[data]   群组划分: 分隔符长度 "
            f"{groups.get('separator_length')}B, "
            "groups="
            + ", ".join(
                f"{name}:{count}"
                for name, count in per_split.items()
            )
        )

    cleaning = report.get("train_cleaning")
    if isinstance(cleaning, dict):
        kinds = ", ".join(
            str(kind)
            for kind in (cleaning.get("cleaning_kinds") or [])
        )
        detail_parts: List[str] = []
        if cleaning.get("groups_total") is not None:
            detail_parts.append(
                f"drop={cleaning.get('groups_dropped')}"
                f"/{cleaning.get('groups_total')} 群组"
            )
        if cleaning.get("dropped_bytes") is not None:
            detail_parts.append(
                f"{cleaning.get('dropped_bytes')}B"
            )
        if cleaning.get("valid_window_starts") is not None:
            detail_parts.append(
                "valid_starts="
                f"{cleaning.get('valid_window_starts')}"
                f"/{cleaning.get('total_window_starts')}"
            )
        print(
            "[data]   训练集清洗"
            + (f"（{kinds}）" if kinds else "")
            + ": "
            + ", ".join(detail_parts)
        )

        bias = cleaning.get("deletion_bias")
        if isinstance(bias, dict):
            bias_parts = [
                f"drop_fraction={bias.get('drop_fraction')}",
                f"byte_js={bias.get('byte_js')}",
                f"bigram_js={bias.get('bigram_js')}",
            ]
            print(
                "[data]   清洗删除偏差: "
                + ", ".join(bias_parts)
                + (
                    "（删除非随机，清洗前后分布不可直接比较）"
                    if (
                        _finite_float(bias.get("drop_fraction"))
                        is not None
                        and _finite_float(bias.get("drop_fraction"))
                        > _AUDIT_DELETION_DROP_WARN
                    )
                    or (
                        _finite_float(bias.get("byte_js"))
                        is not None
                        and _finite_float(bias.get("byte_js"))
                        > _AUDIT_DELETION_BIAS_WARN
                    )
                    else ""
                )
            )

    label_coverage = report.get("label_coverage") or {}
    train_coverage = label_coverage.get("train")
    if isinstance(train_coverage, dict):
        print(
            "[data]   训练集标签覆盖: "
            f"{train_coverage.get('observed_byte_classes')}/256 "
            "个字节类"
        )
    for split, info in label_coverage.items():
        if split == "train" or not isinstance(info, dict):
            continue
        print(
            f"[data]   {split} 标签覆盖: 未见 "
            f"{info.get('unseen_label_classes')} 类，占评估 "
            "token "
            f"{_format_finite(info.get('unseen_label_token_share'), '.4%')}"
        )

    if isinstance(groups, dict):
        health = groups.get("content_health")
        if isinstance(health, dict):
            print(
                "[data]   训练群组异常值: "
                f"degenerate={health.get('degenerate_groups')}"
                f"/{health.get('long_groups_checked')} "
                "长群组（单一字节占比阈值 "
                f"{health.get('top_share_threshold')}），"
                "最大占比 "
                f"{_format_finite(health.get('max_top_byte_share'))}"
            )

        duplicates = groups.get("content_duplicates") or {}
        for pair, info in (
            duplicates.get("cross_split") or {}
        ).items():
            print(
                f"[data]   群组内容重复 {pair} = "
                f"{info.get('shared_groups')}"
            )

    for key, distance in report.get(
        "distribution_shift",
        {},
    ).items():
        print(
            f"[data]   {key} 字节分布 JS 距离 = "
            f"{_format_finite(distance)}"
        )

    for key, distance in report.get(
        "distribution_shift_bigram",
        {},
    ).items():
        print(
            f"[data]   {key} 字节二元组分布 JS 距离 = "
            f"{_format_finite(distance)}"
        )

    for key, distance in report.get(
        "distribution_shift_conditional",
        {},
    ).items():
        unseen_share = (
            report.get("unseen_transition_share", {}).get(key, 0.0)
        )
        print(
            f"[data]   {key} 条件标签分布 JS 距离 = "
            f"{_format_finite(distance)} "
            "(未见转移 "
            f"{_format_finite(unseen_share, '.1%')})"
        )

    for key, info in report.get(
        "duplicate_windows",
        {},
    ).items():
        if "shared_signatures" in info:
            print(
                f"[data]   {key} 共享窗口签名数 = "
                f"{info['shared_signatures']}"
            )

    coverage = report.get("leakage_scan_coverage") or {}
    per_split_coverage = coverage.get("per_split") or {}
    if per_split_coverage:
        print(
            "[data]   泄露窗口扫描覆盖: "
            + ", ".join(
                f"{name}:{info['sampled']}/{info['total']}"
                + (
                    f"(stride={info['stride']})"
                    if int(info["stride"]) > 1
                    else ""
                )
                for name, info in per_split_coverage.items()
                if int(info["total"]) > 0
            )
            + (
                "（穷尽扫描）"
                if coverage.get("exhaustive")
                else "（抽样扫描，零发现 ≠ 无泄露）"
            )
        )

    chunks = report.get("content_chunks") or {}
    for key, info in (
        chunks.get("cross_split") or {}
    ).items():
        if info.get("shared_chunks", 0) > 0:
            print(
                f"[data]   {key} 共享内容定义块 = "
                f"{info['shared_chunks']} "
                f"({info['shared_bytes']}B)"
            )

    near = report.get("near_duplicates")
    if isinstance(near, dict):
        print(
            "[data]   近重复审计: "
            f"flagged={near.get('groups_flagged')}"
            f"/{near.get('groups_scanned')} 群组, "
            f"Jaccard>={near.get('threshold')}, "
            f"shingle={near.get('shingle_bytes')}B, "
            f"candidates={near.get('candidate_pairs')}"
            + (
                "（含 test 内容）"
                if near.get("test_content_used")
                else "（仅 val）"
            )
        )
        for example in (near.get("examples") or [])[:3]:
            if not isinstance(example, (list, tuple)) or len(
                example
            ) < 6:
                continue
            print(
                "[data]     近重复示例: "
                f"[{example[0]},{example[1]}) ≈ "
                f"[{example[4]},{example[5]}) "
                f"reason={example[2]} J={example[3]}"
            )

    eval_near = report.get("evaluation_near_duplicates")
    if isinstance(eval_near, dict):
        print(
            "[data]   评估集近重复审计: "
            f"Jaccard>={eval_near.get('threshold')}, "
            f"shingle={eval_near.get('shingle_bytes')}B, "
            f"candidates={eval_near.get('candidate_pairs')}"
            + (
                "（含 test 内容）"
                if eval_near.get("test_content_used")
                else "（仅 val）"
            )
        )
        for split, info in (
            eval_near.get("per_split") or {}
        ).items():
            print(
                f"[data]     {split}: 可比群组 "
                f"{info.get('groups_scanned')}, "
                f"近重复对 {info.get('within_pairs')}, "
                f"有效群组 {info.get('effective_groups')}"
                f"/{info.get('groups_scanned')}, "
                "最大簇 "
                f"{info.get('largest_cluster_groups')}"
            )
        for pair, info in (
            eval_near.get("cross_split") or {}
        ).items():
            if int(info.get("pairs", 0)) <= 0:
                continue
            print(
                f"[data]     {pair} 跨 split 近重复对 = "
                f"{info.get('pairs')}"
            )

    violation_set = set(report.get("violations", []))

    for warning in report.get("warnings", []):
        # 致命问题在下面以「严重」单独打印，避免重复两遍。
        if warning not in violation_set:
            print(f"[data] 警告: {warning}")

    for violation in report.get("violations", []):
        print(f"[data] 严重: {violation}")



# ============================================================
# Class-imbalance-aware metrics
# ============================================================
#
# 指标口径说明（多分类、类别不平衡场景）：
# - 下一 token 预测是 256 类单标签多分类问题，字节分布天然
#   高度不平衡（空格/换行/常见字母 vs 罕见控制字节）；
# - token accuracy 单独看会被高频字节抬高，必须与
#   majority_accuracy（常数预测器参考）对照解读，并用
#   macro / weighted F1 补充逐类视角；
# - 单标签多分类下 micro-F1 数值恒等于 token accuracy，
#   两者都保留是为了让键名明确交代「micro」口径；
# - macro F1 只对评估 split 中出现过的类（support > 0）
#   求平均，避免给从未出现的字节类记 0 分拉低均值；
# - one-vs-rest 的 macro PR-AUC / ROC-AUC 只统计 support
#   不小于 _AUC_MIN_SUPPORT 的类。极度不平衡时优先看
#   PR-AUC：ROC-AUC 会因负样本基数巨大、FPR 增长缓慢而
#   给出过于乐观的分数。

# macro AUC 只纳入 support 达到该阈值的字节类。
_AUC_MIN_SUPPORT = 20


# AUC 计算最多保留的评估位置数。超出时不再截取评估序列
# 前缀（会让后段窗口系统性缺席），而是在全局评估位置上等距
# 抽样，并记录覆盖率与抽样方式。
_AUC_MAX_TOKENS = 100_000


# AUC 收集每次处理的最大行数；分块 softmax 与整批结果逐位一致。
_AUC_SOFTMAX_CHUNK_ROWS = 8192


# AUC 分块排序的块大小上限（纳入类数 × 评估位置数）。
_AUC_SORT_CHUNK_ELEMENTS = 1_600_000



def _auc_uniform_sample_positions(
    total_positions: int,
    capacity: int,
) -> Optional[torch.Tensor]:
    """
    在全部评估位置上等距抽取 capacity 个位置。

    ``total_positions <= capacity`` 时返回 None（无需抽样，直接
    全量收集）。否则用全局位置步长 floor(i·N/K) 抽样，保证前后
    段窗口都有代表，而不是只保留序列前缀。
    """
    if capacity <= 0 or total_positions <= capacity:
        return None
    if capacity == 1:
        return torch.tensor(
            [total_positions // 2],
            dtype=torch.int64,
        )
    step = float(total_positions - 1) / float(capacity - 1)
    positions = torch.round(
        torch.arange(capacity, dtype=torch.float64) * step
    ).to(torch.int64)
    positions.clamp_(0, total_positions - 1)
    return torch.unique(positions)



def confusion_matrix_metrics(
    confusion: torch.Tensor,
) -> Dict[str, float]:
    """
    从 [n_classes, n_classes] 混淆矩阵（行 = 目标类，
    列 = 预测类）计算 token accuracy 与 micro / macro /
    weighted F1 以及 macro / weighted precision、recall，
    还有多数类参考 accuracy。

    返回键全部不带 split 前缀，由调用方加上
    （val_ / test_），并保证口径在文档中显式说明：
    - ``token_accuracy`` / ``micro_f1``：micro 口径
      （单标签多分类下两者恒等；micro precision =
      micro recall = accuracy，故不单列）；
    - ``macro_f1`` / ``precision_macro`` / ``recall_macro``：
      对 support > 0 的类求算术平均（每类等权，少数类
      退化不会被高频类掩盖）；
    - ``weighted_f1`` / ``precision_weighted`` /
      ``recall_weighted``：按各类 support 加权平均；
    - ``majority_accuracy``：把所有位置都预测成评估集
      中最频繁字节时的 accuracy，是类别不平衡下
      解读 accuracy 的下界参考。

    precision / recall 让假设里「按研究目标选指标」（如
    「召回率不低于 90%」）真正可判定，而不是登记一个
    评估管线从不计算的指标。
    """
    confusion = confusion.double()
    total = confusion.sum()

    if float(total) <= 0.0:
        return {
            "token_accuracy": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "precision_macro": float("nan"),
            "precision_weighted": float("nan"),
            "recall_macro": float("nan"),
            "recall_weighted": float("nan"),
            "majority_accuracy": float("nan"),
            "classes_present": 0.0,
        }

    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positive = confusion.diagonal()

    precision = true_positive / predicted.clamp(min=1.0)
    recall = true_positive / support.clamp(min=1.0)

    denominator = precision + recall
    f1 = torch.zeros_like(true_positive)
    nonzero = denominator > 0.0
    f1[nonzero] = (
        2.0
        * precision[nonzero]
        * recall[nonzero]
        / denominator[nonzero]
    )

    present = support > 0.0

    return {
        "token_accuracy": float(
            true_positive.sum() / total
        ),
        "micro_f1": float(true_positive.sum() / total),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float(
            (f1 * support).sum() / total
        ),
        "precision_macro": float(precision[present].mean()),
        "precision_weighted": float(
            (precision * support).sum() / total
        ),
        "recall_macro": float(recall[present].mean()),
        "recall_weighted": float(
            (recall * support).sum() / total
        ),
        "majority_accuracy": float(
            support.max() / total
        ),
        "classes_present": float(present.sum()),
    }



def class_balanced_loss_metrics(
    class_loss_sums: torch.Tensor,
    class_counts: torch.Tensor,
    vocab_size: int,
) -> Dict[str, float]:
    """
    类平衡（macro）似然口径：对评估集中出现的每个目标类等权
    平均每类 NLL。

    ``val_loss`` 是按 token 频率加权的平均：字节级语料里
    少数高频字节（空格、换行、常见标点）占绝大多数 token，
    稀有字节即使完全没学会，val_loss 也几乎不动——用频率
    加权的似然判定「学会了上下文」会系统性偏乐观。macro NLL
    先对每个类算它自己的平均 NLL，再对类等权平均，是似然
    口径下与 macro F1 对应的类平衡指标（不平衡场景下与
    macro F1 / PR-AUC 一起看才有意义）。

    随机水平参照 ``macro_nll_chance`` = log(vocab_size)：
    均匀预测器对每个类的 NLL 都恰为 log V，类平衡平均后不变，
    因此它是任何模型必须显著低于的绝对下限。

    ``macro_nll_min_support`` 披露被纳入的最小类 support：
    只有 1~2 个样本的稀有类会让 macro NLL 的方差被放大，
    解读时必须看到这一项（与 macro F1 同样的注意事项）。
    """
    counts = class_counts.to(torch.float64)
    sums = class_loss_sums.to(torch.float64)
    present = counts > 0.0

    chance = math.log(float(vocab_size))

    if not bool(present.any()):
        return {
            "macro_nll": float("nan"),
            "macro_nll_classes": 0.0,
            "macro_nll_chance": chance,
            "macro_nll_min_support": 0.0,
        }

    per_class = sums[present] / counts[present]
    return {
        "macro_nll": float(per_class.mean()),
        "macro_nll_classes": float(int(present.sum())),
        "macro_nll_chance": chance,
        "macro_nll_min_support": float(counts[present].min()),
    }



# 评估集不确定性：对固定评估窗口做有放回重采样。窗口是
# 评估协议的最小独立单位（同一窗口内 token 高度相关），
# 逐 token 重采样会低估不确定性。
_EVAL_BOOTSTRAP_CI_LEVEL = 0.95


# 评估窗口 lag-1 自相关达到该阈值时，把 bootstrap 区间按
# AR(1) 方差膨胀因子放宽（见 _bootstrap_eval_intervals）。
# 0.3 是「明显正自相关」的经验门槛：低于它时样本 ρ̂1 本身
# 的抽样噪声与调整量同量级，调整反而引入随机扰动；只放宽
# 不收窄，负相关不用于把区间做窄。
_EVAL_WINDOW_AUTOCORRELATION = 0.3


# 评估窗口 bootstrap 的重采样次数。该常量必须在使用它的
# 函数默认参数之前定义（模块导入时即求值），因此放在评估
# 不确定性工具区的开头。
_EVAL_WINDOW_BOOTSTRAP_REPLICATES = 2000


# 防御损坏/伪造报告里的超大重采样次数：再大既没有实际
# 统计收益，也无法在有限内存内完成。正常配置远低于该上限。
_MAX_BOOTSTRAP_REPLICATES = 10_000_000



def _eval_bootstrap_seed(
    data_sha256: str,
    split: str,
    batches: int,
    batch_size: int,
    seq_len: int,
) -> int:
    """
    由数据指纹与评估协议生成稳定的 bootstrap 随机种子。

    不使用全局 RNG：最终/测试评估的置信区间必须在同一
    运行内可复现，且不能扰动训练或数据采样轨迹。
    """
    material = (
        f"{data_sha256}|{split}|{batches}|{batch_size}|"
        f"{seq_len}|bootstrap-v1"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(),
        "big",
    )



def _percentile_interval(
    values: torch.Tensor,
    ci_level: float,
) -> Tuple[float, float]:
    """返回 [lower, upper] 分位区间（线性插值）。"""
    lower = (1.0 - ci_level) / 2.0
    upper = 1.0 - lower
    quantiles = torch.quantile(
        values.to(torch.float64).reshape(-1),
        torch.tensor(
            [lower, upper],
            dtype=torch.float64,
        ),
    )
    return float(quantiles[0]), float(quantiles[1])



def _lag1_autocorrelation(
    values: torch.Tensor,
) -> Optional[float]:
    """
    序列的 lag-1 自相关 ρ̂1；样本不足或方差为 0 时返回 None。

    bootstrap 把窗口当 i.i.d. 抽样单位，但固定窗口沿评估序列
    排列，窗口指标常随数据顺序渐变（同一类块、主题相邻等）。
    正自相关会让区间偏窄；报告 ρ̂1 与 AR(1) 有效窗口数
    n·(1-ρ)/(1+ρ) 用于方差膨胀。ρ̂1 只依赖顺序与数值，与
    bootstrap 随机数无关。
    """
    numeric = values.to(torch.float64).reshape(-1)
    n = int(numeric.numel())
    if n < 4 or not bool(torch.isfinite(numeric).all()):
        return None
    # 合法的大数值（如 1e308）会让 ``head.mean()`` 与平方和溢出
    # 成 inf；此时 ``inf / inf`` 得到 NaN，后续 clamp 会把它变成
    # 伪造的 rho=1（常数序列也会被误判为强正自相关）。自相关对
    # 公共尺度不变，按最大绝对值归一化后再计算；正常量级保持
    # 原路径，避免改变历史结果的末位舍入。
    scale = float(numeric.abs().max().item())
    if scale > 1.0e150 or 0.0 < scale < 1.0e-150:
        numeric = numeric / scale
    head = numeric[:-1]
    tail = numeric[1:]
    head = head - head.mean()
    tail = tail - tail.mean()
    # 正常量级沿用「平方和相乘后开方」以保持历史末位；两个
    # 平方和各自可能合法但很小（如 1e-200），先相乘会在 1e-400
    # 量级下溢成 0，此时改用分别开方后相乘（数学等价）。
    head_square = float((head * head).sum())
    tail_square = float((tail * tail).sum())
    product = head_square * tail_square
    if product > 0.0 and math.isfinite(product):
        denominator = math.sqrt(product)
    else:
        denominator = math.sqrt(head_square) * math.sqrt(tail_square)
    if float(denominator) <= 0.0:
        return None
    raw_rho = float((head * tail).sum() / denominator)
    return max(-1.0, min(1.0, raw_rho))



def _ar1_effective_windows(
    n_windows: int,
    rho1: Optional[float],
) -> Tuple[float, float]:
    """
    由 ρ̂1 给出 AR(1) 有效窗口数与区间方差膨胀因子。

    负相关不用于缩窄区间（只放宽不收窄）：ρ→-1 时
    n·(1-ρ)/(1+ρ) 发散甚至除零（完全交替序列 ρ̂1 恰为 -1），
    此时直接取 n。ρ̂1 达到 _EVAL_WINDOW_AUTOCORRELATION 才
    启用膨胀，低于阈值时样本 ρ̂ 的噪声与调整量同量级。
    """
    if rho1 is None:
        return float(n_windows), 1.0
    if rho1 <= 0.0:
        effective = float(n_windows)
    elif rho1 >= 0.999:
        effective = 1.0
    else:
        effective = max(
            1.0,
            min(
                float(n_windows),
                float(n_windows)
                * (1.0 - rho1)
                / (1.0 + rho1),
            ),
        )
    inflation = 1.0
    if rho1 >= _EVAL_WINDOW_AUTOCORRELATION:
        inflation = math.sqrt(float(n_windows) / effective)
    return effective, inflation



def _ar1_adjusted_mean_test(
    values: List[float],
) -> Dict[str, object]:
    """
    对逐窗口配对差值的均值做 AR(1) 有效窗口数校正检验。

    普通配对 t 检验把沿评估序列排列的窗口当独立样本；固定
    窗口可能重叠或同属相邻主题块，正自相关会让标准误偏小。
    这里用 ρ̂1 推导有效窗口数 n_eff = n(1-ρ)/(1+ρ)，把标准误
    放大 sqrt(n/n_eff)，自由度取 n_eff-1。只放宽不收窄；ρ̂1
    低于阈值时不调整。n_eff<3 时不给 p 值（有效证据不足），
    避免用 1 个有效窗口伪造显著性。
    """
    n = len(values)
    if n < _MIN_SEEDS_FOR_STATS:
        return {
            "applied": False,
            "lag1_autocorrelation": None,
            "effective_windows": float(n),
            "ci_inflation": 1.0,
            "p_value": None,
            "df": None,
            "note": "insufficient_windows",
        }
    numeric = _finite_values(values)
    if numeric is None:
        return {
            "applied": False,
            "lag1_autocorrelation": None,
            "effective_windows": float(n),
            "ci_inflation": 1.0,
            "p_value": None,
            "df": None,
            "note": "non_finite_values",
        }
    values = numeric

    tensor = torch.tensor(values, dtype=torch.float64)
    rho1 = _lag1_autocorrelation(tensor)
    effective, inflation = _ar1_effective_windows(n, rho1)
    result: Dict[str, object] = {
        "applied": inflation > 1.0,
        "lag1_autocorrelation": rho1,
        "effective_windows": effective,
        "ci_inflation": inflation,
        "p_value": None,
        "df": None,
        "note": None,
    }
    if inflation <= 1.0:
        result["note"] = "autocorrelation_below_threshold"
        return result

    mean = _mean(values)
    std = _sample_std(values)
    if mean is None or std is None:
        result["note"] = "invalid_values"
        return result
    se = std / math.sqrt(n) * inflation
    if not math.isfinite(se):
        result["note"] = "invalid_values"
        return result
    if se == 0.0:
        if mean == 0.0:
            result["p_value"] = 1.0
            result["df"] = None
            result["note"] = "zero_variance"
        else:
            result["p_value"] = 0.0
            result["df"] = None
            result["note"] = "noise_free_separation"
        return result

    if effective < _MIN_SEEDS_FOR_STATS:
        result["note"] = "effective_windows_below_three"
        return result

    df = effective - 1.0
    t_value = mean / se
    result["t"] = t_value
    result["se"] = se
    result["df"] = df
    result["p_value"] = _two_sided_t_p_value(
        abs(t_value),
        df,
    )
    return result



# ============================================================
# 文档/群组 cluster bootstrap
# ============================================================

# 评估窗口不是天然的独立抽样单位：同一文档会被多个重叠窗口
# 覆盖，同一来源/主题的相邻文档也高度相关。窗口级 bootstrap
# 把同一文档的重复证据当成独立样本，会系统性低估评估不确定性。
# 有 --group-separator 时，把每个目标 token 归入其所属群组，
# 以群组为 resampling unit；没有群组边界时不伪造簇。
_GROUP_STATS_SCHEMA = "smelt-isr/eval-group-stats/v1"

_GROUP_ASSIGNMENT = (
    "split_group_starts_gaps_to_preceding_v1"
)



def _parse_group_stats(
    payload: object,
) -> Optional[Dict[str, object]]:
    """
    解析并校验 eval_windows.json 里的 ``group_stats``。

    损坏/缺字段/重复 id/非有限值一律返回 None：不能把无法
    核对的群组记录当成可用证据。
    """
    if not isinstance(payload, dict):
        return None
    stats = payload.get("group_stats")
    if not isinstance(stats, dict):
        return None
    if stats.get("schema") != _GROUP_STATS_SCHEMA:
        return None
    if stats.get("assignment") != _GROUP_ASSIGNMENT:
        return None
    records = stats.get("groups")
    if not isinstance(records, list) or not records:
        return None
    mapping: Dict[int, Tuple[float, float, float]] = {}
    for record in records:
        if not isinstance(record, (list, tuple)):
            return None
        if len(record) < 4:
            return None
        group_id = record[0]
        if (
            not isinstance(group_id, int)
            or isinstance(group_id, bool)
            or group_id < 0
        ):
            return None
        tokens = _finite_float(record[1])
        loss_sum = _finite_float(record[2])
        correct = _finite_float(record[3])
        if (
            tokens is None
            or loss_sum is None
            or correct is None
            or tokens <= 0.0
            or loss_sum < 0.0
            or correct < 0.0
            or correct > tokens + 1e-9
        ):
            return None
        if group_id in mapping:
            return None
        mapping[group_id] = (tokens, loss_sum, correct)
    separator = stats.get("group_separator_sha256")
    if separator is not None and not isinstance(separator, str):
        return None
    unit = stats.get("unit")
    if unit is not None and not isinstance(unit, str):
        return None
    return {
        "mapping": mapping,
        "group_separator_sha256": separator,
        "unit": unit,
    }



def _group_stats_bootstrap_ci(
    point_values: torch.Tensor,
    weights: torch.Tensor,
    replicates: int,
    generator: torch.Generator,
    ci_level: float = _EVAL_BOOTSTRAP_CI_LEVEL,
) -> Optional[Tuple[float, float]]:
    """按群组有放回重采样、再加权平均的分位区间。"""
    try:
        replicate_count = int(replicates)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        isinstance(ci_level, bool)
        or not isinstance(ci_level, (int, float))
    ):
        return None
    try:
        level = float(ci_level)
    except OverflowError:
        return None
    if not 0.0 < level < 1.0:
        return None
    n_groups = int(point_values.numel())
    if (
        n_groups < 2
        or replicate_count <= 1
        or replicate_count > _MAX_BOOTSTRAP_REPLICATES
    ):
        return None
    if (
        point_values.dim() != 1
        or weights.dim() != 1
        or int(weights.numel()) != n_groups
        or not bool(torch.isfinite(point_values).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0.0).any())
    ):
        return None
    value_scale = float(point_values.abs().max().item())
    if value_scale > 1.0e150 or 0.0 < value_scale < 1.0e-150:
        point_values = point_values / value_scale
    else:
        value_scale = 1.0
    max_weight = float(weights.abs().max().item())
    if max_weight <= 0.0:
        return None
    if max_weight > 1.0e150 or max_weight < 1.0e-150:
        weights = weights / max_weight
    indices = torch.randint(
        low=0,
        high=n_groups,
        size=(replicate_count, n_groups),
        generator=generator,
    )
    sampled_weights = weights[indices].sum(dim=1)
    valid = sampled_weights > 0.0
    if not bool(valid.any()):
        return None
    sampled = (
        (point_values[indices] * weights[indices]).sum(dim=1)
        / sampled_weights
    )
    sampled = sampled[valid]
    if sampled.numel() == 0:
        return None
    low, high = _percentile_interval(sampled, level)
    low *= value_scale
    high *= value_scale
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return low, high



def _group_cluster_eval_intervals(
    group_stats: Dict[str, object],
    replicates: int,
    seed: int,
    ci_level: float = _EVAL_BOOTSTRAP_CI_LEVEL,
) -> Dict[str, object]:
    """
    以文档/群组为单位的评估集 cluster bootstrap 区间。

    逐窗口 bootstrap 把同一文档上的重叠窗口当独立样本；本函数
    用 ``group_stats`` 的逐群组 loss 和/正确数直接重采样群组，
    给出 token 加权的 loss / ppl / token accuracy 区间。它不
    替代窗口级区间，而是回答「换一批同分布文档」时点估计有
    多不确定；没有群组记录时返回不可用。
    """
    parsed = _parse_group_stats({
        "group_stats": group_stats,
    })
    if parsed is None:
        return {
            "available": False,
            "reason": "missing_or_invalid_group_stats",
        }
    if isinstance(ci_level, bool) or not isinstance(
        ci_level, (int, float)
    ):
        return {
            "available": False,
            "reason": "invalid_ci_level",
            "ci_level": _finite_float(ci_level),
        }
    try:
        level = float(ci_level)
    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "reason": "invalid_ci_level",
            "ci_level": _finite_float(ci_level),
        }
    if not 0.0 < level < 1.0:
        return {
            "available": False,
            "reason": "invalid_ci_level",
            "ci_level": _finite_float(ci_level),
        }
    try:
        replicate_count = int(replicates)
    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "reason": "invalid_bootstrap_replicates",
            "replicates": _finite_float(replicates),
        }
    if replicate_count <= 1:
        return {
            "available": False,
            "reason": "insufficient_bootstrap_replicates",
            "replicates": replicate_count,
        }
    if replicate_count > _MAX_BOOTSTRAP_REPLICATES:
        return {
            "available": False,
            "reason": "invalid_bootstrap_replicates",
            "replicates": None,
        }
    if parsed["unit"] not in (None, "document_group"):
        return {
            "available": False,
            "reason": "unsupported_group_unit",
            "unit": parsed["unit"],
        }
    mapping = parsed["mapping"]
    assert isinstance(mapping, dict)
    group_ids = sorted(mapping)
    if len(group_ids) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_document_groups",
            "n_groups": len(group_ids),
        }
    tokens = torch.tensor(
        [mapping[group_id][0] for group_id in group_ids],
        dtype=torch.float64,
    )
    loss_sums = torch.tensor(
        [mapping[group_id][1] for group_id in group_ids],
        dtype=torch.float64,
    )
    correct = torch.tensor(
        [mapping[group_id][2] for group_id in group_ids],
        dtype=torch.float64,
    )
    loss_point = loss_sums / tokens
    scale = float(loss_point.abs().max().item())
    if scale > 1.0e150:
        loss_scaled = loss_point / scale
    elif 0.0 < scale < 1.0e-150:
        # 极小 loss 的加权和可能下溢；归一化后还原。
        loss_scaled = loss_point / scale
    else:
        scale = 1.0
        loss_scaled = loss_point
    accuracy_point = correct / tokens
    try:
        seed_value = int(seed) % (2 ** 64)
    except (TypeError, ValueError, OverflowError):
        return {
            "available": False,
            "reason": "invalid_bootstrap_seed",
        }
    generator = torch.Generator().manual_seed(seed_value)
    loss_interval = _group_stats_bootstrap_ci(
        loss_scaled,
        tokens,
        replicate_count,
        generator,
        level,
    )
    accuracy_interval = _group_stats_bootstrap_ci(
        accuracy_point,
        tokens,
        replicate_count,
        generator,
        level,
    )
    if loss_interval is None:
        return {
            "available": False,
            "reason": "bootstrap_unavailable",
            "n_groups": len(group_ids),
        }
    low = loss_interval[0] * scale
    high = loss_interval[1] * scale
    if not (math.isfinite(low) and math.isfinite(high)):
        return {
            "available": False,
            "reason": "unrepresentable_bounds",
            "n_groups": len(group_ids),
        }
    token_total = float(tokens.sum().item())
    result: Dict[str, object] = {
        "available": True,
        "n_groups": len(group_ids),
        "tokens": (
            token_total if math.isfinite(token_total) else None
        ),
        "bootstrap_replicates": replicate_count,
        "bootstrap_ci_level": level,
        "bootstrap_unit": "document_group",
        "loss_ci95_low": max(0.0, low),
        "loss_ci95_high": max(0.0, high),
        "ppl_ci95_low": math.exp(min(80.0, max(0.0, low))),
        "ppl_ci95_high": math.exp(min(80.0, max(0.0, high))),
        "method": "group_cluster_percentile_bootstrap",
        "note": (
            "以文档/群组为重采样单位的 token 加权区间；"
            "群组来自 --group-separator。它反映换一批同分布"
            "文档的评估不确定性，不替代训练种子级检验"
        ),
    }
    if accuracy_interval is not None:
        result["token_accuracy_ci95_low"] = min(
            1.0, max(0.0, accuracy_interval[0])
        )
        result["token_accuracy_ci95_high"] = min(
            1.0, max(0.0, accuracy_interval[1])
        )
    return result
