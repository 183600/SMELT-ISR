from __future__ import annotations



def imbalance_metric_advisory(
    rows: List[Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    类别不平衡下的指标解读提示。

    类别不平衡时 accuracy / micro F1 会被高频类抬高；极度
    不平衡（多数类 ≥99.9%，约 1:1000 以上）时 ROC-AUC 的
    FPR 分母由海量负样本决定，随机水平仍是 0.5，而随机
    PR-AUC 只有约正的 prevalence（1:1000 时 ≈0.001）：此时
    排序判定只看 PR-AUC，且必须与 chance 水平对照，不能把
    PR-AUC 的绝对值当「准确率」读。检查基于每个运行
    metrics.jsonl 记录的 val_majority_accuracy（最高频类的
    support 占比）与 val_pr_auc_macro_chance（PR-AUC 随机
    水平 = 纳入类平均 prevalence，有 AUC 记录时可用）。
    """
    report: Dict[str, object] = {
        "metric": metric,
        "max_majority_accuracy": None,
        "most_imbalanced_run": None,
        "pr_auc_chance_mean": None,
        "pr_auc_over_chance": None,
        "advisories": [],
    }

    # JSON 允许 10**400 这类超出 float 范围的整数：按不可用值
    # 跳过，不能因为单个损坏记录让整份不平衡提示崩溃。
    entries = [
        (str(row.get("run")), value)
        for row in rows
        for value in [_finite_float(
            row.get("val_majority_accuracy")
        )]
        if value is not None
    ]

    # PR-AUC 随机水平：只统计记录了该字段的运行。
    chances = [
        value
        for row in rows
        for value in [_finite_float(
            row.get("val_pr_auc_macro_chance")
        )]
        if value is not None
    ]
    chance_mean = _mean(chances)
    report["pr_auc_chance_mean"] = chance_mean

    if not entries:
        return report

    most_imbalanced_run, max_majority = max(
        entries,
        key=lambda item: item[1],
    )
    report["max_majority_accuracy"] = max_majority
    report["most_imbalanced_run"] = most_imbalanced_run

    if max_majority >= _IMBALANCE_EXTREME_MAJORITY:
        chance_text = (
            f"，macro PR-AUC 的随机水平≈平均 prevalence"
            f"≈{chance_mean:.4f}"
            if chance_mean is not None
            else ""
        )
        report["advisories"].append(
            f"最高频类占比达 {max_majority:.2%}"
            f"（{most_imbalanced_run}，val_majority_accuracy，"
            "约 1:1000 以上的极度不平衡）：accuracy 与 micro "
            "口径已被常数预测主导。字节语言模型的主指标仍应是 "
            "NLL / bits-per-byte（预注册口径），并辅以类平衡 "
            "val_macro_nll；PR-AUC 只作为诊断，必须报告相对"
            "随机水平（≈平均 prevalence）的 lift，而不是把"
            "PR-AUC 绝对值当归一化准确率"
            f"{chance_text}；PR-AUC 必须显著高于该随机水平"
        )
    elif max_majority >= _IMBALANCE_DOMINANT_MAJORITY:
        report["advisories"].append(
            f"最高频类占比达 {max_majority:.1%}"
            f"（{most_imbalanced_run}，val_majority_accuracy）："
            "accuracy 与 micro 口径已被常数预测主导；主指标应"
            "保持 NLL / bits-per-byte，辅以 macro NLL 与 macro "
            "F1 诊断。ROC-AUC 对 prevalence 不敏感，但高 AUC "
            "不保证低 prevalence 下有可用的 precision/recall；"
            "PR-AUC 需按 lift 解读"
        )
    elif max_majority >= _IMBALANCE_WEAK_MAJORITY:
        report["advisories"].append(
            f"最高频类占比达 {max_majority:.1%}"
            f"（{most_imbalanced_run}）：accuracy/micro 口径"
            "参考价值有限，应同时报告 macro F1"
        )

    base_metric = _common_step_metric_base(metric) or metric
    if (
        base_metric in ("val_token_accuracy", "val_micro_f1")
        and max_majority >= _IMBALANCE_WEAK_MAJORITY
    ):
        report["advisories"].append(
            f"当前主指标 {metric} 受多数类主导：若研究问题是"
            "语言建模似然，应保留 NLL / bits-per-byte 为主指标，"
            "把 accuracy / macro F1 / PR-AUC 作为辅助诊断；"
            "只有任务本身是分类决策时才应预注册分类指标"
        )

    # 似然口径同样是频率加权的：val_loss / val_ppl / bpb 按
    # token 数加权，高频字节几乎决定全部数值，稀有字节即使
    # 完全没学会也几乎不影响它。不平衡场景下判定必须同时看
    # 类平衡口径 val_macro_nll（对出现类等权，随机水平
    # log 词表大小）。
    frequency_weighted_metrics = (
        "val_loss",
        "val_ppl",
        "val_bits_per_byte",
        _CURVE_MEAN_METRIC,
    )
    if (
        base_metric in frequency_weighted_metrics
        and max_majority >= _IMBALANCE_WEAK_MAJORITY
    ):
        report["advisories"].append(
            f"当前主指标 {metric} 按 token 频率加权："
            f"最高频类占 {max_majority:.1%}，它几乎决定全部"
            "似然值，稀有字节学没学会都看不出来；不平衡场景"
            "请同时看类平衡口径 val_macro_nll"
            "（--compare-metric val_macro_nll，随机水平 "
            "log 词表大小）"
        )
    if base_metric == "val_macro_nll":
        report["advisories"].append(
            "当前主指标是类平衡似然（val_macro_nll，对评估集"
            "中出现的类等权）：它与频率加权的 val_loss 回答的"
            "问题不同（稀有字节 vs 整体似然），结论不一致时"
            "必须同时报告两者，不要只挑对自己有利的一个"
        )

    # PR-AUC 自身必须相对随机水平解读：正类越稀疏，随机
    # PR-AUC 越低，绝对值看起来「差」不代表模型无用。
    pr_values = [
        (str(row.get("run")), value)
        for row in rows
        for value in [_finite_float(row.get(metric))]
        if value is not None
    ]
    if metric == "val_pr_auc_macro" and pr_values:
        best_run, best_value = min(pr_values, key=lambda item: item[1])
        if chance_mean is not None:
            lift = best_value - chance_mean
            report["pr_auc_over_chance"] = (
                lift if math.isfinite(lift) else None
            )
            report["pr_auc_worst_run"] = best_run
            report["advisories"].append(
                f"PR-AUC 的随机水平不是 0 而是≈平均 prevalence"
                f"（{chance_mean:.4f}）：当前最差运行 {best_run} "
                f"的 {metric}={best_value:.4f}，高出随机水平 "
                + (
                    f"{lift:+.4f}；"
                    if math.isfinite(lift)
                    else "（差值超出浮点范围，无法表示）；"
                )
                + "对比 PR-AUC 时应同时报告该差值（lift），"
                "而不是只看绝对值"
            )
        else:
            report["advisories"].append(
                "运行记录缺少 val_pr_auc_macro_chance：无法"
                "给出 PR-AUC 的随机水平参照；请用新版评估"
                "（不断更新代码）重跑，或手工核对该水平"
            )
    if (
        metric == "val_roc_auc_macro"
        and max_majority >= _IMBALANCE_WEAK_MAJORITY
    ):
        extreme = max_majority >= _IMBALANCE_EXTREME_MAJORITY
        report["advisories"].append(
            "类别不平衡下 ROC-AUC 本身不因负样本多而升高"
            "（TPR/FPR 均以真实类别为条件），但高 ROC-AUC 仍"
            "可能对应低 precision/recall"
            + (
                "，随机水平 0.5 与 PR-AUC 随机水平相差"
                f"约 {max_majority:.0%} 量级的比例"
                if extreme
                else ""
            )
            + "；排序诊断请同时报告 PR-AUC 相对随机水平的 "
            "lift（--compare-metric val_pr_auc_macro_lift）"
        )

    return report



def same_seed_randomness_check(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    同种子跨运行核对初始化与首批训练数据是否逐位一致。

    统计比较假设：同一种子下，除声明因子外其余随机性来源
    相同。若共享初始化、结构分支初始化或首批训练数据指纹
    不一致，则同种子配对不再成立，显著差异不能归因到声明
    因子。旧运行缺指纹时只报告「无法核对」，不伪造一致。
    """
    groups: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("deterministic"):
            continue
        mode = str(row.get("mode") or "")
        if mode == "ngram":
            continue
        seed = row.get("seed")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            continue
        groups.setdefault(seed, []).append(row)

    category_fields = (
        ("shared", "共享参数"),
        ("moe", "MoE 独有参数"),
        ("ffn", "稠密 FFN 独有参数"),
    )
    confounds: List[Dict[str, object]] = []
    unverified: List[str] = []
    seed_groups: List[Dict[str, object]] = []
    checked_groups = 0

    for seed in sorted(groups):
        entries = groups[seed]
        if len(entries) < 2:
            continue
        checked_groups += 1
        group_confounds: List[Dict[str, object]] = []
        group_unverified: List[str] = []

        initializations = []
        missing_init_runs = []
        for row in entries:
            init = row.get("init_fingerprints")
            if isinstance(init, dict):
                initializations.append((row, init))
            else:
                missing_init_runs.append(str(row.get("run")))
        if len(initializations) < 2:
            group_unverified.append(
                "种子 " + str(seed) + " 的运行缺少初始化指纹"
                + (
                    "（"
                    + ", ".join(missing_init_runs)
                    + "）"
                    if missing_init_runs
                    else ""
                )
            )
        else:
            if missing_init_runs:
                # 同一个种子组里只要有一半有指纹还不足以支撑
                # 「同起点」：缺少指纹的运行无法核对，必须显式
                # 列为未验证，不能因为其余运行一致就静默放过。
                group_unverified.append(
                    "种子 " + str(seed) + " 的以下运行缺少"
                    "初始化指纹，无法核对: "
                    + ", ".join(missing_init_runs)
                )
            schemes = sorted({
                str(init.get("scheme"))
                for _, init in initializations
                if init.get("scheme") is not None
            })
            if len(schemes) > 1:
                group_confounds.append({
                    "type": "initialization_scheme",
                    "seed": seed,
                    "values": {
                        str(row.get("run")): str(
                            init.get("scheme")
                        )
                        for row, init in initializations
                    },
                    "modes": sorted({
                        str(row.get("mode"))
                        for row, _ in initializations
                    }),
                })
            for category, label in category_fields:
                key = f"{category}_sha256"
                recorded = [
                    (
                        str(row.get("run")),
                        str(row.get("mode")),
                        str(init.get(key)),
                    )
                    for row, init in initializations
                    if init.get(key) is not None
                ]
                if len(recorded) < 2:
                    continue
                distinct = {digest for _, _, digest in recorded}
                if len(distinct) <= 1:
                    continue
                group_confounds.append({
                    "type": "initialization",
                    "category": category,
                    "label": label,
                    "seed": seed,
                    "values": {
                        run: digest
                        for run, _, digest in recorded
                    },
                    "modes": sorted({
                        mode for _, mode, _ in recorded
                    }),
                })

        traces = []
        missing_trace_runs = []
        for row in entries:
            trace = row.get("training_trace")
            if isinstance(trace, dict) and trace.get(
                "sha256"
            ) is not None:
                traces.append((row, trace))
            else:
                missing_trace_runs.append(str(row.get("run")))
        if len(traces) < 2:
            group_unverified.append(
                "种子 " + str(seed) + " 的运行缺少首批训练数据指纹"
                + (
                    "（"
                    + ", ".join(missing_trace_runs)
                    + "）"
                    if missing_trace_runs
                    else ""
                )
            )
        else:
            if missing_trace_runs:
                group_unverified.append(
                    "种子 " + str(seed) + " 的以下运行缺少"
                    "首批训练数据指纹，无法核对: "
                    + ", ".join(missing_trace_runs)
                )
            distinct = {
                str(trace.get("sha256"))
                for _, trace in traces
            }
            if len(distinct) > 1:
                group_confounds.append({
                    "type": "training_batch",
                    "seed": seed,
                    "values": {
                        str(row.get("run")): str(
                            trace.get("sha256")
                        )
                        for row, trace in traces
                    },
                    "modes": sorted({
                        str(row.get("mode"))
                        for row, _ in traces
                    }),
                })

        confounds.extend(group_confounds)
        unverified.extend(group_unverified)
        seed_groups.append({
            "seed": seed,
            "runs": [
                str(row.get("run")) for row in entries
            ],
            "confounds": len(group_confounds),
            "unverified": group_unverified,
        })

    return {
        "ok": not confounds,
        "groups_checked": checked_groups,
        "seed_groups": seed_groups,
        "confounds": confounds,
        "unverified": unverified,
    }



def annotate_factorial_confounds(
    factorial: Dict[str, object],
    controlled: Dict[str, object],
    coverage: Dict[str, object],
) -> None:
    """
    把受控变量违规、结构混杂与计算量不匹配挂到因子效应上。

    统计显著只说明数据里存在差异；若某个臂换了优化器/学习率/
    训练步数，或消融单元之间 k1/top_k 不同，或 B 的 FLOPs 远高
    于 local_base，这个差异就不能归因到待验证的组件。此时把
    ``claim`` 从「支持/有害/交互显著」降级为 ``confounded`` 或
    ``compute_confounded``，保留原始统计量供复核。若部分协议项
    在所有运行中都未记录，则降级为 ``unverified_controls``：
    「没记录」不等于「一致」，不能据此归因。
    """
    factorial["controlled_variables_ok"] = bool(
        controlled.get("ok")
    )
    positive_claims = {
        "supports_component",
        "harms_component",
        "interaction_detected",
    }

    if not controlled.get("ok"):
        for effect in factorial.get("effects", []):
            effect["controlled_variables_ok"] = False
            if effect.get("claim") in positive_claims:
                effect["claim"] = "confounded"
        factorial.setdefault("advisories", []).append(
            "受控变量检查存在违规：主效应/交互的差值混入了"
            "优化器/学习率/训练步数等非实验因子，不能据此"
            "声称组件有效"
        )

    unchecked_controls = [
        str(key)
        for key in (controlled.get("unchecked_keys") or [])
    ]
    if unchecked_controls:
        factorial["unchecked_control_keys"] = unchecked_controls
        for effect in factorial.get("effects", []):
            effect["uncontrolled_keys"] = list(
                unchecked_controls
            )
            if effect.get("claim") in positive_claims:
                effect["claim"] = "unverified_controls"
        factorial.setdefault("advisories", []).append(
            "以下控制项在所有运行中都没有记录，无法确认一致："
            + ", ".join(unchecked_controls)
            + "；把「未记录」当作「已控制」会把未知差异算进"
            "组件效应，显著结果不能归因"
        )

    if coverage.get("structural_confounds"):
        for effect in factorial.get("effects", []):
            effect["structural_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "unverified_controls"
            }:
                effect["claim"] = "confounded"
        factorial.setdefault("advisories", []).append(
            "消融单元之间存在结构设置混杂：主效应/交互不能"
            "归因到待验证设计"
        )

    balance_confounds = coverage.get("balance_confounds") or []
    if balance_confounds:
        for effect in factorial.get("effects", []):
            effect["balance_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "confounded",
                "unverified_controls",
            }:
                effect["claim"] = "balance_confounded"
        factorial.setdefault("advisories", []).append(
            "均衡损失强度（balance_coef × 有效路由宽度）在"
            "消融单元间不一致：主效应/交互混入均衡压力差异，"
            "不能归因到延迟合并或均衡损失本身"
        )

    attrition_confounds = coverage.get(
        "attrition_confounds"
    ) or []
    if attrition_confounds:
        affected_modes = set()
        for confound in attrition_confounds:
            affected_modes.update(
                str(mode)
                for mode in (confound.get("affected_modes") or [])
            )
        for effect in factorial.get("effects", []):
            involved = {
                "local_base",
                str(effect.get("delayed_mode")),
            }
            if not involved & affected_modes:
                continue
            effect["attrition_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "unverified_controls",
                "confounded",
                "compute_confounded",
            }:
                effect["claim"] = "attrition_confounded"
        factorial.setdefault("advisories", []).append(
            "数值失败在消融单元间不平衡：幸存运行的条件均值"
            "不能直接归因到组件，须同时报告失败率或重跑"
        )

    randomness_confounds = coverage.get(
        "randomness_confounds"
    ) or []
    if randomness_confounds:
        affected_modes = set()
        for confound in randomness_confounds:
            affected_modes.update(
                str(mode)
                for mode in (confound.get("modes") or [])
            )
        for effect in factorial.get("effects", []):
            involved = {
                "local_base",
                str(effect.get("delayed_mode")),
            }
            if not involved & affected_modes:
                continue
            effect["randomness_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "unverified_controls",
                "confounded",
                "compute_confounded",
            }:
                effect["claim"] = "randomness_confounded"
        factorial.setdefault("advisories", []).append(
            "同种子运行的初始化或首批训练数据不同：同种子配对"
            "假设不成立，主效应/交互不能归因到声明因子"
        )

    compute_confounds = {
        entry.get("delayed_mode")
        for entry in coverage.get("compute_parity", [])
        if not entry.get("matched")
    }
    compute_confounds.update(
        entry.get("delayed_mode")
        for entry in coverage.get("width_confounds", [])
    )
    compute_confounds.update(
        entry.get("delayed_mode")
        for entry in coverage.get("capacity_parity", [])
        if not entry.get("matched")
    )
    compute_confounds.discard(None)
    if compute_confounds:
        for effect in factorial.get("effects", []):
            if effect.get("delayed_mode") not in compute_confounds:
                continue
            effect["compute_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "confounded",
                "unverified_controls",
            }:
                effect["claim"] = "compute_confounded"
        factorial.setdefault("advisories", []).append(
            "部分方案与 local_base 的计算量不匹配：这些方案的"
            "主效应混入计算量，须在等训练计算量指标上复核"
        )

    capacity_confounds = {
        entry.get("delayed_mode")
        for entry in coverage.get("capacity_parity", [])
        if not entry.get("matched")
    }
    capacity_confounds.discard(None)
    if capacity_confounds:
        for effect in factorial.get("effects", []):
            if effect.get("delayed_mode") not in capacity_confounds:
                continue
            effect["capacity_confounded"] = True
            if effect.get("claim") in positive_claims | {
                "confounded",
                "unverified_controls",
                "compute_confounded",
            }:
                effect["claim"] = "capacity_confounded"
        factorial.setdefault("advisories", []).append(
            "部分方案与 local_base 的参数量不匹配：即使 FLOPs "
            "对齐，多出的参数容量也可能带来收益，主效应不能"
            "归因到延迟合并本身"
        )



def _balance_width_profile(
    protocol: Dict[str, object],
) -> Optional[Dict[str, object]]:
    """
    估计一条运行的有效均衡宽度（每次路由调用的平均 top-k）。

    辅助损失在每次 ``SparseMoE.route`` 调用上约为 k（均匀路由
    下 E·sum(f_i P_i)=k），一个 Block 的 aux 是各次调用的均值：
    - smelt：每层一次路由，宽度 top_k；
    - A/B：第一轮 k1、第二轮每分支 k2，均值 (k1+k2)/2；
    - local_base：第一轮 k1、第二轮为对齐计算量的
      min(n_experts, k1*k2)；新实现把第二轮 aux 折算回 k2，
      旧实现（normalization="none"）未折算。
    该值是用于发现混杂的一阶近似，不替代实测路由统计。
    """
    if not isinstance(protocol, dict):
        return None

    mode = str(protocol.get("mode") or "")
    if mode not in {"smelt", "A", "B", "local_base"}:
        return None

    try:
        balance_coef = float(protocol.get("balance_coef"))
    except (TypeError, ValueError, OverflowError):
        return None
    if balance_coef < 0.0:
        return None

    def _as_int(key: str) -> Optional[int]:
        value = protocol.get(key)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed <= 0:
            return None
        # 协议里的计数可能被手工/旧记录写成 10**400；int() 本身
        # 可以接受，但后续转成 float 计算强度会溢出。这里提前
        # 拒绝不可表示的计数，让该运行按「缺少可核对协议」跳过。
        try:
            numeric = float(parsed)
        except (OverflowError, ValueError):
            return None
        return parsed if math.isfinite(numeric) else None

    if mode == "smelt":
        top_k = _as_int("top_k")
        if top_k is None:
            return None
        return {
            "mode": mode,
            "balance_coef": balance_coef,
            "effective_width": float(top_k),
            "normalization": "not_applicable",
        }

    k1 = _as_int("k1")
    k2 = _as_int("k2")
    if k1 is None or k2 is None:
        return None

    if mode in {"A", "B"}:
        # 各自折半后再相加：k1+k2 可以先越过 float 上限，而
        # 真实平均宽度仍可能可表示。
        width = 0.5 * float(k1) + 0.5 * float(k2)
        normalization = "not_applicable"
    else:
        normalization = str(
            protocol.get("balance_width_normalization")
            or "none"
        )
        if normalization == _BALANCE_WIDTH_NORMALIZATION:
            second_width = k2
        else:
            n_experts = _as_int("n_experts")
            if n_experts is None:
                return None
            second_width = min(n_experts, k1 * k2)
        width = 0.5 * float(k1) + 0.5 * float(second_width)
    if not math.isfinite(width):
        return None

    return {
        "mode": mode,
        "balance_coef": balance_coef,
        "effective_width": width,
        "normalization": normalization,
    }



def balance_strength_report(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    核对跨运行的均衡损失强度 ``balance_coef × 有效路由宽度``。

    路由宽度不同本身不是问题（k1/k2 可以作为实验因子），但
    若宽度变化而均衡强度没有相应折算，方法效应就会与均衡
    压力混杂。系数为 0 的消融臂不施加均衡压力，不参与该
    强度比较（其全因子作用仍由消融覆盖检查负责）。
    """
    report: Dict[str, object] = {
        "applicable": False,
        # 是否真的完成了强度比较：两个早退分支（可比运行不足、
        # 施加均衡压力的有效臂不足）返回时 widths/strengths 根本
        # 没算，打印端不能把这种情况报成「一致」。
        "compared": False,
        "width_varied": False,
        "strength_varied": False,
        "confounded": False,
        "profiles": [],
        "advisories": [],
        "note": (
            "均衡强度 = balance_coef × 平均每次路由的 top-k；"
            "跨路由宽度比较时必须保持该乘积一致"
        ),
    }

    for row in rows:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            continue
        profile = _balance_width_profile(protocol)
        if profile is None:
            continue
        strength = (
            float(profile["balance_coef"])
            * float(profile["effective_width"])
        )
        if not math.isfinite(strength):
            # 合法但极大的 coef/width 相乘会越过 float 上限；
            # 无法参与强度比较，记录后跳过，不能写 inf。
            report.setdefault("invalid_profiles", []).append({
                "run": str(row.get("run")),
                "mode": profile["mode"],
                "reason": "balance_strength_not_finite",
            })
            continue
        report["profiles"].append({
            "run": str(row.get("run")),
            "mode": profile["mode"],
            "balance_coef": profile["balance_coef"],
            "effective_width": profile["effective_width"],
            "balance_strength": strength,
            "normalization": profile["normalization"],
        })

    profiles = report["profiles"]
    if len(profiles) < 2:
        return report

    # 均衡强度比较只看施加了均衡压力的臂：balance_coef=0 的
    # 对照臂强度恒为 0，把它算进极差会让「已按宽度折算 coef」
    # 的正确做法仍被误报为混杂（其全因子作用由消融覆盖检查
    # 负责）。少于两个有效臂时无法比较强度，直接返回。
    active_profiles = [
        profile
        for profile in profiles
        if float(profile["balance_coef"]) > 0.0
    ]
    if len(active_profiles) < 2:
        return report

    widths = [
        float(profile["effective_width"])
        for profile in active_profiles
    ]
    strengths = [
        float(profile["balance_strength"])
        for profile in active_profiles
    ]
    width_span = max(widths) - min(widths)
    strength_span = max(strengths) - min(strengths)
    # 到这里才真正做了跨臂比较：至少两个 coef>0 的臂参与了
    # 极差计算，报告允许据此给出「一致/不一致」的结论。
    report["compared"] = True
    strength_scale = max(
        1.0,
        max(abs(value) for value in strengths),
    )
    width_varied = width_span > 1e-9 * max(
        1.0,
        max(abs(value) for value in widths),
    )
    strength_varied = strength_span > 1e-9 * strength_scale

    report["width_varied"] = width_varied
    report["strength_varied"] = strength_varied
    report["applicable"] = width_varied

    if not width_varied or not strength_varied:
        return report

    report["confounded"] = True
    lowest = min(
        active_profiles,
        key=lambda item: float(item["balance_strength"]),
    )
    highest = max(
        active_profiles,
        key=lambda item: float(item["balance_strength"]),
    )
    ratio = (
        float(highest["balance_strength"])
        / float(lowest["balance_strength"])
        if float(lowest["balance_strength"]) > 0.0
        else None
    )
    if ratio is not None and math.isfinite(ratio):
        report["strength_ratio"] = ratio
    report["advisories"].append(
        "均衡强度随路由宽度变化但未被抵消："
        f"{lowest['run']}（mode={lowest['mode']}, "
        f"coef={float(lowest['balance_coef']):g}, "
        f"width={float(lowest['effective_width']):g}, "
        f"strength={float(lowest['balance_strength']):g}）"
        " vs "
        f"{highest['run']}（mode={highest['mode']}, "
        f"coef={float(highest['balance_coef']):g}, "
        f"width={float(highest['effective_width']):g}, "
        f"strength={float(highest['balance_strength']):g}）"
        + (
            f"，最高/最低={ratio:.2f}×"
            if ratio is not None
            else ""
        )
        + "。请按宽度折算 balance_coef（保持 "
        "balance_coef × width 不变）后重跑，否则路由宽度会"
        "与均衡强度混杂"
    )
    return report



def factorial_balance_strength(
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    eligible = []
    for row in rows:
        protocol = row.get("protocol")
        if not isinstance(protocol, dict):
            continue
        if str(protocol.get("mode") or "") not in {
            "local_base",
            *_DELAYED_MERGE_MODES,
        }:
            continue
        eligible.append(row)
    return balance_strength_report(eligible)



def _format_factorial_effect(effect: Dict[str, object]) -> str:
    """把一条因子效应格式化为控制台行（主分析/等计算量分析共用）。"""
    p_holm = effect.get("p_value_holm")
    if p_holm is not None:
        p_text = f"Holm p={_format_finite(p_holm)}"
    elif effect.get("p_value") is not None:
        p_text = f"p={_format_finite(effect.get('p_value'))}"
    else:
        p_text = "p=NA"
    ci_text = ""
    if (
        effect.get("ci95_low") is not None
        and effect.get("ci95_high") is not None
    ):
        ci_text = (
            " 95%CI=["
            f"{_format_finite(effect.get('ci95_low'), '+.4f')},"
            f"{_format_finite(effect.get('ci95_high'), '+.4f')}]"
        )
    adjusted_low = effect.get("ci_family_adjusted_low")
    adjusted_high = effect.get("ci_family_adjusted_high")
    adjusted_family = effect.get("ci_family_adjusted_family_size")
    if (
        adjusted_low is not None
        and adjusted_high is not None
        and isinstance(adjusted_family, int)
        and adjusted_family > 1
    ):
        ci_text += (
            "；同时95%CI=["
            f"{_format_finite(adjusted_low, '+.4f')},"
            f"{_format_finite(adjusted_high, '+.4f')}]"
            f"(Bonferroni α/{adjusted_family}，比Holm保守)"
        )
    qualifiers = []
    if effect.get("test") == "no_common_seeds":
        qualifiers.append("无共同种子，未做配对检验")
    elif not effect.get("aligned_seeds"):
        qualifiers.append("种子集合未完全对齐，仅共同子集")
    if effect.get("n_pairs", 0) < _MIN_SEEDS_FOR_STATS:
        qualifiers.append(
            f"配对种子 {effect.get('n_pairs', 0)} < "
            f"{_MIN_SEEDS_FOR_STATS}"
        )
    if (
        effect.get("randomization_method") == "exact"
        and effect.get("randomization_can_reach_significance")
        is False
    ):
        minimum = effect.get("randomization_min_attainable_p")
        required = effect.get("randomization_required_seeds")
        qualifiers.append(
            "精确符号翻转最小可达 p="
            + (
                _format_finite(minimum, ".3f")
                if isinstance(minimum, (int, float))
                else "NA"
            )
            + " > alpha（n="
            + str(effect.get("n_pairs", 0))
            + "）；"
            + (
                f"至少需 {required} 个独立种子"
                if required is not None
                else "需增加独立种子"
            )
            + "，不显著不能解读为无效应"
        )
    if effect.get("controlled_variables_ok") is False:
        qualifiers.append("受控变量检查未通过")
    if effect.get("uncontrolled_keys"):
        qualifiers.append(
            "未记录协议项: "
            + ", ".join(
                str(key)
                for key in effect["uncontrolled_keys"][:6]
            )
            + (
                " 等"
                if len(effect["uncontrolled_keys"]) > 6
                else ""
            )
        )
    if effect.get("structural_confounded"):
        qualifiers.append("消融结构设置混杂")
    if effect.get("compute_confounded"):
        qualifiers.append("计算量未对齐")
    if effect.get("capacity_confounded"):
        qualifiers.append("模型参数量未对齐")
    if effect.get("randomness_confounded"):
        qualifiers.append(
            "同种子初始化/训练数据不一致"
        )
    if effect.get("duplicate_seeds"):
        qualifiers.append("存在重复随机种子（已折叠）")
    if not effect.get("complete_seed_records", True):
        qualifiers.append(
            "存在未记录种子的运行（配对仅覆盖有种子子集）"
        )
    tost_holm = effect.get("tost_p_value_holm")
    if tost_holm is not None:
        margin_abs = effect.get("equivalence_margin_abs")
        margin_text = (
            f"±{_format_finite(margin_abs)}"
            if margin_abs is not None
            else "界=NA"
        )
        if effect.get("equivalent"):
            qualifiers.append(
                "等价 TOST Holm p="
                f"{_format_finite(tost_holm)}（{margin_text}）"
            )
        else:
            qualifiers.append(
                "未达等价 TOST Holm p="
                f"{_format_finite(tost_holm)}（{margin_text}）"
            )
    if effect.get("significant"):
        claim = {
            "supports_component": "支持该组件有效",
            "harms_component": "该组件显著变差",
            "interaction_detected": "交互显著",
            "confounded": "显著但受控变量/结构混杂，不能归因",
            "unverified_controls": (
                "显著但关键协议项未记录，不能归因"
            ),
            "compute_confounded": (
                "显著但计算量不匹配，需等计算量复核"
            ),
            "capacity_confounded": (
                "显著但参数量/容量不匹配，不能归因"
            ),
            "randomness_confounded": (
                "显著但同种子随机性未对齐，不能归因"
            ),
            "attrition_confounded": (
                "显著但数值失败率在单元间不平衡（幸存者偏差），"
                "不能归因"
            ),
        }.get(str(effect.get("claim")), "显著")
    else:
        claim = "不显著"
        if effect.get("mde_80") is not None:
            qualifiers.append(
                "80% 功效 MDE="
                f"{_format_finite(effect.get('mde_80'))}"
            )
    if effect.get("test") in (
        "paired_t",
        "welch_t",
    ) and effect.get("p_value") is not None:
        test_text = f"配对 t(n={effect.get('n_pairs', 0)})"
    else:
        test_text = "未检验"
    return (
        f"  [{effect['kind']}] {effect['design']} "
        f"{effect['description']}  "
        f"Δ={_format_finite(effect.get('mean_effect'), '+.4f')}  "
        f"{test_text} {p_text}{ci_text}  {claim}"
        + (
            "（" + "；".join(qualifiers) + "）"
            if qualifiers
            else ""
        )
    )



def equivalence_margin_preregistration(
    rows: List[Dict[str, object]],
    equivalence_margin: Optional[float],
) -> Dict[str, object]:
    """
    核对分析期使用的 TOST 等价界是否等于训练前预注册的界。

    等价界越宽，越容易把「不确定」判成「等价」；如果允许看到
    数据后再调宽 `--equivalence-margin`，等价性结论就可以被
    事后制造。训练入口已把等价界写入
    ``hypothesis_preregistration.equivalence_margin``，这里逐
    运行核对：

    - ``consistent``：当前界与全部新运行预注册值一致；
    - ``widened_post_hoc``：当前界宽于预注册界，等价结论只能
      作为探索性/非确证性结果；
    - ``narrowed_post_hoc``：当前界更窄（更严格），不构成
      「制造等价」的风险，但仍不是训练前声明的口径；
    - ``inconsistent_declarations``：不同运行预注册了不同界，
      无法裁决；
    - ``unverifiable``：旧运行没有记录，或缺当前界。
    """
    current = _finite_float(equivalence_margin)
    if current is not None and current <= 0.0:
        current = None

    declared: Dict[str, float] = {}
    missing: List[str] = []
    invalid: List[str] = []
    for row in rows:
        if _is_deterministic_run(row):
            continue
        run = str(row.get("run"))
        raw_value = row.get("hypothesis_equivalence_margin")
        if raw_value is None:
            missing.append(run)
            continue
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float)
        ):
            invalid.append(run)
            continue
        # 超范围整数/非有限值按无效预注册处理，不能因为
        # 单个损坏运行让整份等价性核对崩溃。
        numeric = _finite_float(raw_value)
        if numeric is None or numeric <= 0.0:
            invalid.append(run)
            continue
        declared[run] = numeric

    values = sorted(set(declared.values()))
    report: Dict[str, object] = {
        "current_margin": current,
        "declared_margins": declared,
        "declared_values": values,
        "missing_runs": missing,
        "invalid_runs": invalid,
        "status": None,
        "ok": None,
        "widened_post_hoc": False,
        "advisories": [],
    }

    if not declared:
        report["status"] = "unverifiable"
        report["advisories"].append(
            "没有任何运行记录训练前预注册的等价界"
            "（旧版运行）：无法核对当前 --equivalence-margin "
            "是否被事后调整；等价结论只能作为探索性证据"
        )
        return report

    if len(values) > 1:
        report["status"] = "inconsistent_declarations"
        report["ok"] = False
        report["advisories"].append(
            "不同运行预注册的等价界不一致（"
            + ", ".join(format(value, "g") for value in values)
            + "）：无法确定统一的事先声明口径，等价结论只能"
            "作为探索性证据；请统一 --equivalence-margin 后重跑"
        )
        return report

    declared_value = values[0]
    if current is None:
        report["status"] = "unverifiable"
        report["advisories"].append(
            "当前分析没有可用的等价界，无法与训练前预注册的 "
            f"{format(declared_value, 'g')} 核对"
        )
        return report

    if math.isclose(
        current,
        declared_value,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        report["status"] = "consistent"
        if missing or invalid:
            report["ok"] = None
            detail: List[str] = []
            if missing:
                detail.append(
                    "以下运行没有预注册等价界（旧版运行）: "
                    + ", ".join(missing)
                )
            if invalid:
                detail.append(
                    "以下运行的预注册等价界无效: "
                    + ", ".join(invalid)
                )
            report["advisories"].append(
                "当前等价界与已记录运行的预注册值一致，但"
                + "；".join(detail)
                + "：无法核对全部运行，等价结论只能视为对"
                "已记录运行有效"
            )
        else:
            report["ok"] = True
        return report

    report["ok"] = False
    if current > declared_value:
        report["status"] = "widened_post_hoc"
        report["widened_post_hoc"] = True
        report["advisories"].append(
            "当前等价界 "
            f"±{current:.4%} 宽于训练前预注册的 "
            f"±{declared_value:.4%}：事后放宽等价界会提高"
            "「等价」的判定率，本次等价性结论不能作为确证性"
            "证据；应使用预注册界，或把新界作为新的预注册"
            "假设重新实验"
        )
    else:
        report["status"] = "narrowed_post_hoc"
        report["advisories"].append(
            "当前等价界 "
            f"±{current:.4%} 窄于训练前预注册的 "
            f"±{declared_value:.4%}：更严格的界不会「制造」"
            "等价，但口径已不是训练前声明的那个；报告需"
            "注明两者差异"
        )
    return report



def resolve_preregistered_baseline(
    rows: List[Dict[str, object]],
    baseline_mode: Optional[str],
) -> Dict[str, object]:
    """
    核对「分析期选用的对照基线」与「训练前预注册的对照基线」。

    预注册了主指标/方向/最小效应、却没有预注册对照臂时，事后
    更换对照基线仍属于 baseline shopping：挑一个最容易战胜的
    臂当基线，任何显著性/等价性结论都失去意义。规则：

    - 各运行在 ``--hypothesis-baseline`` 里写入同一模式时，
      该模式是唯一合法的分析基线；``--compare-baseline`` 未
      显式指定或指定了别的模式都会在这里核对并提示；
    - 不同运行预注册了不同基线时不一致，无法自动裁决，保持
      调用方传入的基线并显式告警；
    - 旧运行没有该字段时保持 ``ok=None``，不虚报通过。
    """
    preregistered: Dict[str, str] = {}
    missing: List[str] = []
    for row in rows:
        if _is_deterministic_run(row):
            continue
        run = str(row.get("run"))
        declared = row.get("hypothesis_baseline")
        if isinstance(declared, str) and declared:
            preregistered[run] = declared
        else:
            missing.append(run)
    declared_modes = sorted(set(preregistered.values()))
    run_modes = {
        str(row.get("mode"))
        for row in rows
        if row.get("mode") is not None
    }
    report: Dict[str, object] = {
        "baseline_mode": baseline_mode,
        "preregistered_runs": preregistered,
        "declared_modes": declared_modes,
        "runs_without_preregistration": missing,
        "consistent": None,
        "auto_selected": False,
        "mismatch": False,
        "baseline_in_runs": None,
        "ok": None,
        "advisories": [],
    }

    if not declared_modes:
        return report

    if len(declared_modes) > 1:
        report["consistent"] = False
        report["ok"] = False
        report["advisories"].append(
            "各运行预注册的对照基线不一致（"
            + ", ".join(declared_modes)
            + "）；无法自动裁决，分析继续使用 "
            + (
                f"--compare-baseline={baseline_mode}，"
                if baseline_mode is not None
                else "当前对比集合的分组，"
            )
            + "但请先统一训练前声明并重跑，避免事后挑基线"
        )
        return report

    declared = declared_modes[0]
    report["baseline_in_runs"] = (
        declared in run_modes if run_modes else None
    )
    if baseline_mode is None:
        report["baseline_mode"] = declared
        report["auto_selected"] = True
        report["consistent"] = True
        report["ok"] = True
        report["advisories"].append(
            f"未显式传 --compare-baseline，采用训练前预注册的"
            f"对照基线 {declared}（--hypothesis-baseline）"
        )
    elif baseline_mode == declared:
        report["consistent"] = True
        report["ok"] = True
    else:
        report["consistent"] = False
        report["mismatch"] = True
        report["ok"] = False
        report["advisories"].append(
            f"分析使用的对照基线 {baseline_mode} 与训练前预注册的"
            f" {declared} 不一致；事后更换对照基线属于 baseline "
            "shopping，显著性与等价性结论需在预注册基线上复核"
        )
    if report["baseline_in_runs"] is False:
        report["advisories"].append(
            f"预注册的对照基线 {declared} 不在本次对比集合中："
            "本次分析无法给出相对预注册基线的直接对比，请把该"
            "基线的运行一并加入 --compare"
        )
        if report["ok"] is True:
            report["ok"] = None
    if missing:
        report["advisories"].append(
            "以下运行没有预注册对照基线（旧版运行）: "
            + ", ".join(missing)
            + "：无法核对这些运行是否与其余运行声明了同一"
            "分析基线"
        )
        if report["ok"] is True:
            report["ok"] = None
    return report

def print_comparison_analysis(rows: List[Dict[str, object]], factors: Optional[List[str]]=None, baseline_mode: Optional[str]=None, metric: str='val_loss', ablation_modes: Optional[List[str]]=None, equivalence_margin: Optional[float]=_DEFAULT_EQUIVALENCE_MARGIN, allow_partial_ablation: bool=False, common_step: Optional[float]=None) -> Dict[str, object]:
    from exp._runner import run_isolated
    from exp import _manifest
    _root = os.path.dirname(os.path.abspath(__file__))
    _paths = [os.path.join(_root, 'exp', 'analysisflow', _name) for _name in _manifest.PA_PARTS]
    return run_isolated(_paths, globals(), _manifest.PA_OWNED, _manifest.PA_ENTRY, {'rows': rows, 'factors': factors, 'baseline_mode': baseline_mode, 'metric': metric, 'ablation_modes': ablation_modes, 'equivalence_margin': equivalence_margin, 'allow_partial_ablation': allow_partial_ablation, 'common_step': common_step})



def write_comparison_report_file(
    path: Path,
    analysis: Dict[str, object],
) -> Path:
    """
    把对比分析写到显式指定的 JSON 文件（--compare-report）。

    与 ``write_comparison_report`` 共用同一 payload 结构；
    只读的 ``--compare`` 默认不写文件，显式给出路径才落盘。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        **analysis,
    }
    # 与 config/manifest 一样先写临时文件再原子替换：序列化
    # 中途失败不能留下半截 JSON，也不能跟随已有符号链接去
    # 覆盖链接目标。
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print(f"[report] 对比分析已写入 {path}")
    return path



def write_comparison_report(
    out_dir: Path,
    analysis: Dict[str, object],
) -> Path:
    """
    把对比分析写入 out_dir/comparison_analysis.json。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_comparison_report_file(
        out_dir / "comparison_analysis.json",
        analysis,
    )



# ============================================================
# Sanity overfit check
# ============================================================

def _expert_parameter_ids(model: nn.Module) -> set:
    """
    收集 MoE 专家参数的 id 集合。

    专家参数是本模型里唯一可能因「本批次没有 token 路由到该
    专家」而合法缺失/零梯度的参数；其余参数（embedding /
    attention / router / norm / lm_head）在任何输入下都必须
    收到梯度，缺失即说明计算图断链。
    """
    expert_ids = set()

    for module in model.modules():
        if isinstance(module, SparseMoE):
            for expert in module.experts:
                for parameter in expert.parameters():
                    expert_ids.add(id(parameter))

    return expert_ids
