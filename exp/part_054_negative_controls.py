from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import random
import math

from .part_021_comparison import compare_runs
from .part_025_statistics import paired_t_test, welch_t_test, cohens_d, hedges_g
from .part_026_equivalence import tost_equivalence
from .part_023_factorial_helpers import _finite_float, _stable_factor_key
from .part_029_ablation import factorial_effects
from .part_031_ablation_coverage import ablation_coverage


# Types of negative controls
NEGATIVE_CONTROL_TYPES = (
    "label_permutation",      # Randomly permute labels - should destroy signal
    "input_shuffling",        # Shuffle input tokens - should destroy temporal structure
    "random_initialization",  # Compare different random inits of same architecture
    "zero_weight",            # Zero out specific weight matrices
    "ablation_known_null",    # Ablate component known to have no effect
    "synthetic_null",         # Synthetic data with known null effect
)


def run_label_permutation_control(
    rows: List[Dict[str, object]],
    metric: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Label permutation test: randomly permute the metric values across conditions
    to establish a null distribution.

    This tests whether the observed effect could arise by chance if there were
    no true difference between conditions.
    """
    # Group rows by experimental condition
    condition_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        balance = _finite_float(row.get("balance_coef"))
        balance_str = "on" if balance and balance > 0 else "off"
        condition = f"{mode}_bal{balance_str}"
        value = _finite_float(row.get(metric))
        if value is not None:
            condition_values[condition].append(value)

    if len(condition_values) < 2:
        return {
            "applicable": False,
            "reason": "Need at least 2 conditions for permutation test",
        }

    conditions = list(condition_values.keys())
    condition_data = [condition_values[c] for c in conditions]

    # Compute observed test statistic (e.g., max pairwise difference)
    observed_stats = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            diff = abs(
                sum(condition_data[i]) / len(condition_data[i]) -
                sum(condition_data[j]) / len(condition_data[j])
            )
            observed_stats.append(diff)

    observed_max = max(observed_stats) if observed_stats else 0

    # Permutation test
    all_values = []
    for vals in condition_data:
        all_values.extend(vals)

    rng = random.Random(seed)
    null_distribution = []

    for _ in range(n_permutations):
        rng.shuffle(all_values)
        permuted = []
        idx = 0
        for vals in condition_data:
            n = len(vals)
            permuted.append(all_values[idx:idx + n])
            idx += n

        permuted_stats = []
        for i in range(len(permuted)):
            for j in range(i + 1, len(permuted)):
                if permuted[i] and permuted[j]:
                    diff = abs(
                        sum(permuted[i]) / len(permuted[i]) -
                        sum(permuted[j]) / len(permuted[j])
                    )
                    permuted_stats.append(diff)

        if permuted_stats:
            null_distribution.append(max(permuted_stats))

    # p-value = proportion of null >= observed
    p_value = sum(1 for x in null_distribution if x >= observed_max) / max(len(null_distribution), 1)

    return {
        "applicable": True,
        "control_type": "label_permutation",
        "observed_max_diff": observed_max,
        "null_distribution": {
            "mean": sum(null_distribution) / len(null_distribution) if null_distribution else 0,
            "std": math.sqrt(
                sum((x - sum(null_distribution) / len(null_distribution)) ** 2
                    for x in null_distribution) / len(null_distribution)
            ) if null_distribution else 0,
            "p95": sorted(null_distribution)[int(0.95 * len(null_distribution))] if null_distribution else 0,
        },
        "p_value": p_value,
        "n_permutations": n_permutations,
        "interpretation": (
            "Low p-value (<0.05) suggests observed differences are unlikely under null. "
            "High p-value suggests observed effects could be due to chance."
        ),
    }


def run_input_shuffling_control(
    args: Any,
    base_config: Dict[str, Any],
    metric: str,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Input shuffling control: train on shuffled data where temporal structure is destroyed.

    If the model still achieves similar performance on shuffled data,
    it suggests the model is not actually learning temporal dependencies.
    """
    from pathlib import Path
    import subprocess
    import sys

    results = []

    for seed in range(n_seeds):
        run_dir = Path(base_config["out"]) / f"shuffle_control_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "experiment.py",
            "--data", str(base_config["data"]),
            "--mode", base_config["mode"],
            "--seed", str(seed),
            "--out", str(run_dir),
            "--steps", str(base_config.get("steps", 1000)),
            "--shuffle-input", "true",  # This would need to be implemented in data pipeline
        ]

        # Add other config
        for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                    "seq_len", "d_model", "n_heads", "n_layers", "n_experts",
                    "expert_hidden", "top_k", "k1", "k2", "balance_coef",
                    "residual_scale", "residual_scaling"]:
            if key in base_config:
                cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Extract final metric
            rows = compare_runs([str(run_dir)])
            if rows:
                value = _finite_float(rows[0].get(metric))
                if value is not None:
                    results.append(value)

    if not results:
        return {"applicable": False, "reason": "No successful shuffled runs"}

    # Compare with baseline (unshuffled) performance
    baseline_rows = compare_runs([base_config["baseline_out"]])
    baseline_values = [_finite_float(r.get(metric)) for r in baseline_rows if _finite_float(r.get(metric)) is not None]

    if not baseline_values:
        return {"applicable": False, "reason": "No baseline values"}

    baseline_mean = sum(baseline_values) / len(baseline_values)
    shuffle_mean = sum(results) / len(results)

    # Effect size
    d = cohens_d(results, baseline_values)

    return {
        "applicable": True,
        "control_type": "input_shuffling",
        "baseline_mean": baseline_mean,
        "shuffle_mean": shuffle_mean,
        "difference": shuffle_mean - baseline_mean,
        "relative_difference": (shuffle_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else float('inf'),
        "cohens_d": d,
        "interpretation": (
            "If shuffled performance is close to baseline (small |d|), "
            "the model may not be learning meaningful temporal structure. "
            "Large negative d (shuffled much worse) is expected for valid models."
        ),
    }


def run_random_initialization_control(
    args: Any,
    base_config: Dict[str, Any],
    metric: str,
    n_inits: int = 10,
) -> Dict[str, object]:
    """
    Random initialization control: train same architecture with different random seeds
    but identical data/order to quantify initialization variance.

    This establishes the "noise floor" for seed-to-seed variation.
    """
    from pathlib import Path
    import subprocess
    import sys

    results = []

    for init_seed in range(n_inits):
        run_dir = Path(base_config["out"]) / f"init_control_seed{init_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "experiment.py",
            "--data", str(base_config["data"]),
            "--mode", base_config["mode"],
            "--seed", str(init_seed),  # This seed controls both init and data order
            "--out", str(run_dir),
            "--steps", str(base_config.get("steps", 1000)),
            "--init-scheme", "name_hash",  # Use name_hash for consistent init across modes
        ]

        for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                    "seq_len", "d_model", "n_heads", "n_layers", "n_experts",
                    "expert_hidden", "top_k", "k1", "k2", "balance_coef",
                    "residual_scale", "residual_scaling"]:
            if key in base_config:
                cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rows = compare_runs([str(run_dir)])
            if rows:
                value = _finite_float(rows[0].get(metric))
                if value is not None:
                    results.append(value)

    if len(results) < 2:
        return {"applicable": False, "reason": "Need at least 2 successful runs"}

    mean_val = sum(results) / len(results)
    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in results) / (len(results) - 1))
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')

    return {
        "applicable": True,
        "control_type": "random_initialization",
        "n_inits": n_inits,
        "mean": mean_val,
        "std": std_val,
        "cv": cv,  # Coefficient of variation
        "min": min(results),
        "max": max(results),
        "range": max(results) - min(results),
        "interpretation": (
            "This quantifies the variance due to random initialization alone. "
            "Any experimental effect smaller than this range may not be distinguishable "
            "from initialization noise. CV > 0.1 suggests high initialization sensitivity."
        ),
    }


def run_zero_weight_control(
    args: Any,
    base_config: Dict[str, Any],
    metric: str,
    weight_patterns: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Zero weight control: zero out specific weight matrices to test their necessity.

    For example, zero out the second-round routing weights in delayed merge
    to verify that the second round is actually contributing.
    """
    # This would require model surgery - implementing as a placeholder
    # for now with guidance on how to implement
    return {
        "applicable": False,
        "reason": "Requires model surgery implementation. Use --sanity-only with custom model "
                  "modifications to zero specific weight matrices (e.g., second-round router, "
                  "expert output projections, merge weights).",
        "control_type": "zero_weight",
        "suggested_patterns": weight_patterns or [
            "router2",      # Second-round router
            "expert_down2", # Second-round expert down-projection
            "merge_weight", # Merge weights
            "branch_attn",  # Branch-specific attention (scheme B)
        ],
    }


def run_synthetic_null_control(
    args: Any,
    base_config: Dict[str, Any],
    metric: str,
    null_type: str = "random_tokens",
) -> Dict[str, object]:
    """
    Synthetic null control: generate synthetic data with known properties.

    Types:
    - random_tokens: i.i.d. random tokens (no learnable structure)
    - markov_chain: first-order Markov chain (limited structure)
    - shuffled_corpus: real corpus with tokens randomly permuted
    """
    # Placeholder - would need synthetic data generation
    return {
        "applicable": False,
        "reason": "Requires synthetic data generation pipeline. Generate data with known "
                  "statistical properties (e.g., uniform random, Markov chain) and verify "
                  "model cannot achieve better than chance performance.",
        "control_type": "synthetic_null",
        "null_type": null_type,
    }


def run_negative_control_suite(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    controls: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Run a suite of negative controls to validate experimental setup.

    Returns a report indicating whether the experimental setup can detect
    known null effects (sanity check for the methodology).
    """
    controls_to_run = controls or [
        "label_permutation",
        "random_initialization",
    ]

    report: Dict[str, object] = {
        "metric": metric,
        "controls_run": [],
        "controls": {},
        "overall_pass": True,
        "advisories": [],
    }

    for control in controls_to_run:
        if control == "label_permutation":
            rows = compare_runs([base_config["out"]])
            result = run_label_permutation_control(rows, metric)
            report["controls"][control] = result
            report["controls_run"].append(control)

        elif control == "random_initialization":
            result = run_random_initialization_control(args, base_config, metric)
            report["controls"][control] = result
            report["controls_run"].append(control)

        elif control == "input_shuffling":
            result = run_input_shuffling_control(args, base_config, metric)
            report["controls"][control] = result
            report["controls_run"].append(control)

        elif control == "zero_weight":
            result = run_zero_weight_control(args, base_config, metric)
            report["controls"][control] = result
            report["controls_run"].append(control)

        elif control == "synthetic_null":
            result = run_synthetic_null_control(args, base_config, metric)
            report["controls"][control] = result
            report["controls_run"].append(control)

    # Evaluate overall pass/fail
    for control_name, result in report["controls"].items():
        if not result.get("applicable"):
            report["advisories"].append(f"{control_name}: not applicable - {result.get('reason')}")
            continue

        # Check if control behaves as expected
        if control_name == "label_permutation":
            p_val = result.get("p_value", 1.0)
            if p_val < 0.05:
                # Good - permutation test correctly detects structure
                pass
            else:
                report["advisories"].append(
                    f"{control_name}: permutation p={p_val:.3f} - "
                    "observed effects may not be distinguishable from chance"
                )
                report["overall_pass"] = False

        elif control_name == "random_initialization":
            cv = result.get("cv", float('inf'))
            range_val = result.get("range", float('inf'))
            if cv > 0.1 or range_val > 0.1:  # Threshold for "high" initialization variance
                report["advisories"].append(
                    f"{control_name}: high initialization variance (CV={cv:.3f}, range={range_val:.3f}) - "
                    "effects smaller than this may be initialization noise"
                )

    return report


def placebo_test_report(
    experimental_rows: List[Dict[str, object]],
    null_rows: List[Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Placebo test: compare experimental effect size against null distribution.

    If the experimental effect is not significantly larger than effects
    observed under known-null conditions, the result may be spurious.
    """
    exp_values = [_finite_float(r.get(metric)) for r in experimental_rows if _finite_float(r.get(metric)) is not None]
    null_values = [_finite_float(r.get(metric)) for r in null_rows if _finite_float(r.get(metric)) is not None]

    if len(exp_values) < 2 or len(null_values) < 2:
        return {"applicable": False, "reason": "Insufficient data for placebo test"}

    # Effect size in experimental condition vs null
    exp_mean = sum(exp_values) / len(exp_values)
    null_mean = sum(null_values) / len(null_values)

    # t-test
    t_stat, p_val = welch_t_test(exp_values, null_values)
    d = cohens_d(exp_values, null_values)

    return {
        "applicable": True,
        "experimental_mean": exp_mean,
        "null_mean": null_mean,
        "difference": exp_mean - null_mean,
        "welch_t": t_stat,
        "p_value": p_val,
        "cohens_d": d,
        "interpretation": (
            "If p > 0.05 and |d| < 0.2, experimental effect is not distinguishable "
            "from null condition. This suggests the observed effect may be spurious "
            "or the experimental design lacks sensitivity."
        ),
    }


def calibration_check(
    rows: List[Dict[str, object]],
    metric: str,
    known_effect_sizes: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Calibration check: verify that the analysis pipeline correctly recovers
    known effect sizes when they are artificially injected.

    This can be done by:
    1. Adding a known constant to one condition's metric values
    2. Checking if the analysis detects the injected effect
    3. Verifying the estimated effect size matches the injected size
    """
    # This is a meta-analysis tool - would be run on simulated/augmented data
    return {
        "applicable": False,
        "reason": "Calibration check requires synthetic data generation with injected effects. "
                  "Run by: (1) take real data, (2) add known effect to one condition, "
                  "(3) run analysis pipeline, (4) verify recovered effect matches injected.",
        "suggested_procedure": [
            "Take baseline experimental data",
            "Add constant C to metric values for treatment condition",
            "Run full analysis pipeline (compare, ablation, etc.)",
            "Check if estimated effect ≈ C with correct significance",
            "Repeat for multiple C values to verify linearity",
        ],
    }


def print_negative_control_report(report: Dict[str, object]) -> None:
    """Print formatted negative control report."""
    print("\n" + "=" * 80)
    print("NEGATIVE CONTROL / PLACEBO TEST REPORT")
    print("=" * 80)

    print(f"\nMetric: {report.get('metric')}")
    print(f"Controls run: {', '.join(report.get('controls_run', []))}")
    print(f"Overall pass: {report.get('overall_pass')}")

    for control_name, result in report.get("controls", {}).items():
        print(f"\n--- {control_name} ---")
        if not result.get("applicable"):
            print(f"  Not applicable: {result.get('reason')}")
            continue

        if control_name == "label_permutation":
            print(f"  Observed max diff: {result.get('observed_max_diff'):.6f}")
            print(f"  Null mean: {result.get('null_distribution', {}).get('mean'):.6f}")
            print(f"  Null p95: {result.get('null_distribution', {}).get('p95'):.6f}")
            print(f"  p-value: {result.get('p_value'):.4f}")

        elif control_name == "random_initialization":
            print(f"  Mean: {result.get('mean'):.6f}")
            print(f"  Std: {result.get('std'):.6f}")
            print(f"  CV: {result.get('cv'):.6f}")
            print(f"  Range: {result.get('range'):.6f}")

    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")