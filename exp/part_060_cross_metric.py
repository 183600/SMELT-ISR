from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import math

from .part_021_comparison import compare_runs, COMPARISON_METRIC_KEYS
from .part_027_aggregation import aggregate_runs
from .part_029_ablation import (
    factorial_effects,
    mechanism_control_report,
    _DELAYED_MERGE_MECHANISM_CONTROLS,
)
from .part_031_ablation_coverage import ablation_coverage
from .part_030_factorial import _DELAYED_MERGE_MODES
from .part_023_factorial_helpers import _finite_float, _stable_factor_key
from .part_027_aggregation import metric_higher_is_better
from .part_028_contrasts import factor_contrasts


# Metric categories for cross-metric validation
METRIC_CATEGORIES: Dict[str, List[str]] = {
    "loss_based": [
        "val_loss",
        "val_loss_at_matched_flops",
        "val_loss_at_common_step",
        "val_ppl",
        "val_bits_per_byte",
        "val_macro_nll",
        "_curve_mean_metric",
    ],
    "accuracy_based": [
        "val_token_accuracy",
        "val_micro_f1",
        "val_macro_f1",
        "val_weighted_f1",
    ],
    "ranking_based": [
        "val_pr_auc_macro",
        "val_pr_auc_macro_lift",
        "val_roc_auc_macro",
    ],
    "precision_recall": [
        "val_precision_macro",
        "val_precision_weighted",
        "val_recall_macro",
        "val_recall_weighted",
    ],
    "routing": [
        "routing_jaccard_branch",
        "routing_slot_reuse_branch",
        "routing_jaccard_token",
        "routing_branch_self_reuse",
        "routing_r1_load_entropy",
        "routing_r2_load_entropy",
        "routing_r2_dead_experts",
        "routing_r2_router_entropy",
        "routing_branch_cosine",
    ],
}

# Metric direction: higher_is_better for each metric
METRIC_DIRECTION: Dict[str, bool] = {
    "val_loss": False,
    "val_loss_at_matched_flops": False,
    "val_loss_at_common_step": False,
    "val_ppl": False,
    "val_bits_per_byte": False,
    "val_macro_nll": False,
    "_curve_mean_metric": False,
    "val_token_accuracy": True,
    "val_micro_f1": True,
    "val_macro_f1": True,
    "val_weighted_f1": True,
    "val_pr_auc_macro": True,
    "val_pr_auc_macro_lift": True,
    "val_roc_auc_macro": True,
    "val_precision_macro": True,
    "val_precision_weighted": True,
    "val_recall_macro": True,
    "val_recall_weighted": True,
    "routing_jaccard_branch": False,  # Lower = more independent
    "routing_slot_reuse_branch": False,
    "routing_jaccard_token": False,
    "routing_branch_self_reuse": False,
    "routing_r1_load_entropy": True,
    "routing_r2_load_entropy": True,
    "routing_r2_dead_experts": False,
    "routing_r2_router_entropy": True,
    "routing_branch_cosine": False,
}


def run_cross_metric_validation(
    args: Any,
    base_config: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    ablation_modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Run cross-metric validation: evaluate the same ablation on multiple metrics.

    This checks whether component conclusions are consistent across
    different evaluation metrics, reducing the risk of metric-specific artifacts.
    """
    from pathlib import Path
    import subprocess
    import sys

    metrics = metrics or [
        "val_loss",
        "val_macro_nll",
        "val_token_accuracy",
        "val_macro_f1",
        "val_pr_auc_macro_lift",
    ]

    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)
    arm_modes = ["local_base", *delayed_modes]

    if base_config.get("mechanism_controls"):
        for mode in delayed_modes:
            control = _DELAYED_MERGE_MECHANISM_CONTROLS.get(mode)
            if control and control not in arm_modes:
                arm_modes.append(control)

    balance_coef = base_config.get("balance_coef", 0.01)

    # Run experiments once (shared across metrics)
    all_run_paths = []
    for mode in arm_modes:
        for coefficient in (0.0, balance_coef):
            for seed in range(n_seeds):
                run_label = f"crossmetric_{mode}_bal{format(coefficient, 'g')}_seed{seed}"
                run_dir = Path(base_config["out"]) / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, "experiment.py",
                    "--data", str(base_config["data"]),
                    "--mode", mode,
                    "--balance-coef", str(coefficient),
                    "--seed", str(seed),
                    "--out", str(run_dir),
                    "--steps", str(base_config.get("steps", 1000)),
                ]

                for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                            "seq_len", "d_model", "n_heads", "n_layers", "n_experts",
                            "expert_hidden", "top_k", "k1", "k2", "residual_scale", "residual_scaling",
                            "n_kv_heads", "init_scheme"]:
                    if key in base_config:
                        cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

                print(f"[cross-metric] {run_label}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    all_run_paths.append(str(run_dir))

    # Analyze on each metric
    rows = compare_runs(all_run_paths)
    analysis = analyze_cross_metric_consistency(
        rows=rows,
        metrics=metrics,
        ablation_modes=ablation_modes,
    )

    return analysis


def analyze_cross_metric_consistency(
    rows: List[Dict[str, object]],
    metrics: List[str],
    ablation_modes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Analyze component verdicts across multiple metrics.

    Checks:
    1. Direction consistency: do all metrics agree on effect direction?
    2. Significance consistency: do metrics agree on significance?
    3. Magnitude correlation: are effect sizes correlated across metrics?
    2. Verdict agreement: do component verdicts match?
    """
    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)

    report: Dict[str, object] = {
        "metrics": metrics,
        "ablation_modes": delayed_modes,
        "per_metric": {},
        "cross_metric": {},
        "consistency": {},
        "decision_theoretic": {},
        "advisories": [],
    }

    # Per-metric analysis
    for metric in metrics:
        if metric not in COMPARISON_METRIC_KEYS:
            report["advisories"].append(f"Metric {metric} not in COMPARISON_METRIC_KEYS, skipping")
            continue

        coverage = ablation_coverage(rows, ["mode", "balance_coef"], ablation_modes)
        aggregates = aggregate_runs(rows, ["mode", "balance_coef"])
        factorial = factorial_effects(aggregates, ["mode", "balance_coef"], metric)
        mechanism = mechanism_control_report(
            aggregates, ["mode", "balance_coef"], metric
        )

        verdicts = None
        if factorial.get("applicable"):
            from .part_029_ablation import ablation_component_verdicts
            verdicts = ablation_component_verdicts(factorial, coverage, mechanism=mechanism)

        report["per_metric"][metric] = {
            "coverage": coverage,
            "factorial": factorial,
            "mechanism": mechanism,
            "verdicts": verdicts,
            "direction": "higher_better" if metric_higher_is_better(metric) else "lower_better",
        }

    # Cross-metric consistency
    components = ["delayed_merge_A", "delayed_merge_B", "balance_loss"]
    report["cross_metric"] = analyze_metric_agreement(
        report["per_metric"],
        metrics,
        components,
    )

    # Decision-theoretic evaluation
    report["decision_theoretic"] = decision_theoretic_evaluation(
        report["per_metric"],
        metrics,
        components,
    )

    return report


def _primary_evidence_entry(
    summary: object,
) -> Optional[Dict[str, object]]:
    """
    从组件的 primary 汇总里取主效应证据条目。

    ``ablation_component_verdicts`` 的 ``primary`` 是
    ``_component_evidence_summary`` 的输出（status/reason/
    evidence），效应量与显著性在 ``evidence`` 列表里，不在
    顶层；跨指标对比必须从那里取，否则永远拿不到 mean_effect。
    """
    if not isinstance(summary, dict):
        return None
    evidence = summary.get("evidence")
    if not isinstance(evidence, list):
        return None
    for entry in evidence:
        if (
            isinstance(entry, dict)
            and entry.get("mean_effect") is not None
        ):
            return entry
    return None


def analyze_metric_agreement(
    per_metric: Dict[str, Dict[str, object]],
    metrics: List[str],
    components: List[str],
) -> Dict[str, object]:
    """
    Analyze agreement across metrics for each component.
    """
    agreement: Dict[str, object] = {
        "components": {},
        "overall": {},
        "metric_correlations": {},
    }

    for comp in components:
        # Collect verdicts and effects for each metric
        metric_verdicts = {}
        metric_effects = {}
        metric_significant = {}

        for metric in metrics:
            data = per_metric.get(metric, {})
            # 指标不在 COMPARISON_METRIC_KEYS 等情况下 verdicts
            # 会以 None 存在（键存在但值为 None），这里必须同时
            # 处理「键缺失」与「键存在但为 None」，否则 .get 会
            # 在 None 上抛 AttributeError。
            verdicts = data.get("verdicts") or {}
            for c in verdicts.get("components", []):
                if c.get("component") == comp:
                    metric_verdicts[metric] = c.get("status")
                    entry = _primary_evidence_entry(
                        c.get("primary")
                    )
                    metric_effects[metric] = (
                        entry.get("mean_effect")
                        if entry is not None
                        else None
                    )
                    metric_significant[metric] = bool(
                        entry.get("significant", False)
                        if entry is not None
                        else False
                    )
                    break

        if len(metric_verdicts) < 2:
            agreement["components"][comp] = {"applicable": False, "reason": "Insufficient metrics"}
            continue

        # Direction consistency
        effects = {m: e for m, e in metric_effects.items() if e is not None}
        if effects:
            directions = {}
            for m, e in effects.items():
                higher_better = metric_higher_is_better(m)
                directions[m] = "improvement" if (e > 0) == higher_better else "regression"

            direction_consistent = len(set(directions.values())) == 1
            majority_direction = max(set(directions.values()), key=list(directions.values()).count)
        else:
            direction_consistent = False
            majority_direction = "unknown"

        # Significance consistency
        sig_count = sum(1 for v in metric_significant.values() if v)
        sig_consistent = sig_count == 0 or sig_count == len(metric_significant)

        # Verdict agreement
        verdicts = list(metric_verdicts.values())
        verdict_consistent = len(set(verdicts)) == 1
        majority_verdict = max(set(verdicts), key=verdicts.count)

        agreement["components"][comp] = {
            "metric_verdicts": metric_verdicts,
            "metric_effects": metric_effects,
            "metric_directions": directions if effects else {},
            "direction_consistent": direction_consistent,
            "majority_direction": majority_direction,
            "significance_consistent": sig_consistent,
            "sig_count": f"{sig_count}/{len(metric_significant)}",
            "verdict_consistent": verdict_consistent,
            "majority_verdict": majority_verdict,
            "verdicts": verdicts,
        }

    # Metric correlations (effect size correlations)
    for i, m1 in enumerate(metrics):
        for m2 in metrics[i+1:]:
            # 某组件证据不足时 agreement["components"][c] 只有
            # {"applicable": False, "reason": ...}，没有
            # metric_effects 键：统一用 .get 兜底，不能让相关
            # 性计算在部分组件缺失时 KeyError。
            e1 = {c: agreement["components"].get(c, {}).get("metric_effects", {}).get(m1)
                  for c in components}
            e2 = {c: agreement["components"].get(c, {}).get("metric_effects", {}).get(m2)
                  for c in components}

            pairs = [(e1[c], e2[c]) for c in components
                     if e1.get(c) is not None and e2.get(c) is not None]

            if len(pairs) >= 3:
                x_vals = [p[0] for p in pairs]
                y_vals = [p[1] for p in pairs]
                n = len(pairs)
                x_mean = sum(x_vals) / n
                y_mean = sum(y_vals) / n
                x_var = sum((x - x_mean) ** 2 for x in x_vals)
                y_var = sum((y - y_mean) ** 2 for y in y_vals)
                if x_var > 0 and y_var > 0:
                    xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
                    corr = xy_cov / math.sqrt(x_var * y_var)
                    agreement["metric_correlations"][f"{m1}_vs_{m2}"] = corr

    # Overall agreement
    all_dir_consistent = all(
        c.get("direction_consistent", False)
        for c in agreement["components"].values()
    )
    all_sig_consistent = all(
        c.get("significance_consistent", False)
        for c in agreement["components"].values()
    )
    all_verdict_consistent = all(
        c.get("verdict_consistent", False)
        for c in agreement["components"].values()
    )

    agreement["overall"] = {
        "direction_consistent": all_dir_consistent,
        "significance_consistent": all_sig_consistent,
        "verdict_consistent": all_verdict_consistent,
        "verdict": "fully_consistent" if all_dir_consistent and all_sig_consistent and all_verdict_consistent else "inconsistent",
    }

    return agreement


def decision_theoretic_evaluation(
    per_metric: Dict[str, Dict[str, object]],
    metrics: List[str],
    components: List[str],
) -> Dict[str, object]:
    """
    Decision-theoretic framework for component evaluation.

    Uses a utility-based approach:
    - Define utility of each component being truly effective/ineffective
    - Compute posterior probability of effectiveness given multi-metric evidence
    - Make decision that maximizes expected utility
    """
    # Utility matrix (customizable)
    utility = {
        "true_positive": 10,    # Correctly identify effective component
        "true_negative": 5,     # Correctly identify ineffective component
        "false_positive": -20,  # Claim effective when not (wasted effort)
        "false_negative": -10,  # Miss effective component (lost opportunity)
    }

    evaluation: Dict[str, object] = {
        "utility_matrix": utility,
        "components": {},
        "optimal_decisions": {},
        "expected_utilities": {},
    }

    for comp in components:
        # Aggregate evidence across metrics
        evidence = []
        for metric in metrics:
            data = per_metric.get(metric, {})
            # 与 analyze_metric_agreement 相同：verdicts 可能是
            # 键存在但值为 None，必须兜底成空 dict。
            verdicts = data.get("verdicts") or {}
            for c in verdicts.get("components", []):
                if c.get("component") == comp:
                    entry = _primary_evidence_entry(
                        c.get("primary")
                    )
                    mean_effect = (
                        entry.get("mean_effect")
                        if entry is not None
                        else None
                    )
                    # 证据条目里的显著性 p 值是 Holm 校正后的
                    # p_value_holm（不存在裸 p_value 键）。
                    p_val = (
                        entry.get("p_value_holm")
                        if entry is not None
                        else None
                    )
                    significant = bool(
                        entry.get("significant", False)
                        if entry is not None
                        else False
                    )
                    if mean_effect is not None:
                        # 效应符号是原始指标单位：lower-is-better
                        # 指标（val_loss 等）的正效应是变差，不
                        # 能当成「支持有效」。方向判定必须按指标
                        # 的 higher_is_better 口径折算，与
                        # ablation_component_verdicts 的 claim
                        # 一致。
                        improves = (
                            (mean_effect > 0)
                            == metric_higher_is_better(metric)
                        )
                        evidence.append({
                            "metric": metric,
                            "effect": mean_effect,
                            "p": p_val,
                            "significant": significant,
                            "improves": improves,
                        })
                    break

        if not evidence:
            continue

        # Simple Bayesian updating (conjugate prior)
        # Prior: P(effective) = 0.5
        prior_effective = 0.5
        prior_ineffective = 0.5

        # Likelihood: P(evidence | effective) and P(evidence | ineffective)
        # Simplified: each significant result with correct direction supports effective
        # each non-significant or wrong direction supports ineffective
        likelihood_effective = 1.0
        likelihood_ineffective = 1.0

        for e in evidence:
            if e["significant"] and e["improves"]:
                # Supports effectiveness
                likelihood_effective *= 0.8  # Power-like
                likelihood_ineffective *= 0.05  # Type I error
            elif e["significant"] and not e["improves"]:
                # Supports harm
                likelihood_effective *= 0.01
                likelihood_ineffective *= 0.8
            else:
                # Non-significant
                likelihood_effective *= 0.3  # 1 - power
                likelihood_ineffective *= 0.7

        # Posterior
        post_effective = prior_effective * likelihood_effective
        post_ineffective = prior_ineffective * likelihood_ineffective
        total = post_effective + post_ineffective
        post_effective /= total
        post_ineffective /= total

        # Expected utility for each decision
        eu_declare_effective = post_effective * utility["true_positive"] + post_ineffective * utility["false_positive"]
        eu_declare_ineffective = post_effective * utility["false_negative"] + post_ineffective * utility["true_negative"]
        eu_inconclusive = 0  # No action utility

        optimal_decision = max(
            ("declare_effective", eu_declare_effective),
            ("declare_ineffective", eu_declare_ineffective),
            ("inconclusive", eu_inconclusive),
            key=lambda x: x[1]
        )

        evaluation["components"][comp] = {
            "evidence": evidence,
            "posterior_effective": post_effective,
            "posterior_ineffective": post_ineffective,
            "expected_utility_declare_effective": eu_declare_effective,
            "expected_utility_declare_ineffective": eu_declare_ineffective,
            "expected_utility_inconclusive": eu_inconclusive,
            "optimal_decision": optimal_decision[0],
            "optimal_utility": optimal_decision[1],
        }

        evaluation["optimal_decisions"][comp] = optimal_decision[0]
        evaluation["expected_utilities"][comp] = optimal_decision[1]

    return evaluation


def run_metric_prioritization(
    args: Any,
    base_config: Dict[str, Any],
    candidate_metrics: Optional[List[str]] = None,
    ablation_modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Metric prioritization: evaluate which metrics are most informative
    for distinguishing between components.

    Uses criteria:
    - Discriminability: how well metric separates components
    - Stability: low variance across seeds
    - Sensitivity: detects effects other metrics miss
    - Orthogonality: provides independent information
    """
    candidate_metrics = candidate_metrics or [
        "val_loss",
        "val_macro_nll",
        "val_token_accuracy",
        "val_macro_f1",
        "val_pr_auc_macro_lift",
        "val_roc_auc_macro",
    ]

    # Run cross-metric analysis
    rows = compare_runs([])  # Would need actual runs
    # Placeholder for design
    return {
        "applicable": False,
        "reason": "Metric prioritization requires running experiments on candidate metrics first. "
                  "Design: (1) run ablation on all candidate metrics, "
                  "(2) compute discriminability/stability/sensitivity scores, "
                  "(3) rank metrics, (4) select top-k for confirmatory studies.",
        "design": [
            "1. Define candidate metric set",
            "2. Run pilot ablation (3-5 seeds) on all metrics",
            "3. For each metric compute:",
            "   - Discriminability: effect size variance across components",
            "   - Stability: inverse of CV across seeds",
            "   - Sensitivity: fraction of components detected",
            "   - Orthogonality: 1 - max correlation with other metrics",
            "4. Composite score = weighted combination",
            "5. Select top-k for confirmatory study",
        ],
    }


def print_cross_metric_report(report: Dict[str, object]) -> None:
    """Print formatted cross-metric consistency report."""
    print("\n" + "=" * 80)
    print("CROSS-METRIC CONSISTENCY REPORT")
    print("=" * 80)

    print(f"\nMetrics: {', '.join(report.get('metrics', []))}")
    print(f"Ablation modes: {', '.join(report.get('ablation_modes', []))}")

    # Per-metric verdicts
    print("\n--- Per-Metric Component Verdicts ---")
    for metric, data in report.get("per_metric", {}).items():
        print(f"\n  {metric} ({data.get('direction')}):")
        verdicts = data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            print(f"    {comp.get('label')}: {comp.get('status')}")

    # Agreement
    print("\n--- Cross-Metric Agreement ---")
    for comp, agree in report.get("cross_metric", {}).get("components", {}).items():
        print(f"\n  {comp}:")
        print(f"    Verdicts: {agree.get('metric_verdicts')}")
        print(f"    Directions: {agree.get('metric_directions')}")
        print(f"    Direction consistent: {agree.get('direction_consistent')}")
        print(f"    Majority direction: {agree.get('majority_direction')}")
        print(f"    Significance consistent: {agree.get('significance_consistent')}")
        print(f"    Verdict consistent: {agree.get('verdict_consistent')}")
        print(f"    Majority verdict: {agree.get('majority_verdict')}")

    # Metric correlations
    print("\n--- Metric Correlations (effect sizes) ---")
    for pair, corr in report.get("cross_metric", {}).get("metric_correlations", {}).items():
        print(f"  {pair}: r = {corr:.3f}")

    # Decision-theoretic
    print("\n--- Decision-Theoretic Evaluation ---")
    for comp, eval_data in report.get("decision_theoretic", {}).get("components", {}).items():
        print(f"\n  {comp}:")
        print(f"    Posterior P(effective): {eval_data.get('posterior_effective', 0):.3f}")
        print(f"    Optimal decision: {eval_data.get('optimal_decision')}")
        print(f"    EU(declare effective): {eval_data.get('expected_utility_declare_effective', 0):.2f}")
        print(f"    EU(declare ineffective): {eval_data.get('expected_utility_declare_ineffective', 0):.2f}")
        print(f"    EU(inconclusive): {eval_data.get('expected_utility_inconclusive', 0):.2f}")

    # Overall
    overall = report.get("cross_metric", {}).get("overall", {})
    print(f"\n--- Overall Consistency: {overall.get('verdict')} ---")
    print(f"  Direction consistent: {overall.get('direction_consistent')}")
    print(f"  Significance consistent: {overall.get('significance_consistent')}")
    print(f"  Verdict consistent: {overall.get('verdict_consistent')}")

    # Advisories
    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")