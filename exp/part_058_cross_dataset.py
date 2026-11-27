from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math

from .part_021_comparison import compare_runs
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
from .part_025_statistics import paired_t_test, welch_t_test, cohens_d


def run_cross_dataset_validation(
    args: Any,
    base_config: Dict[str, Any],
    datasets: List[Dict[str, Any]],
    metric: str = "val_loss",
    ablation_modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Run cross-dataset validation: train and evaluate on multiple datasets.

    This tests whether experimental conclusions generalize across
    different data distributions.
    """
    from pathlib import Path
    import subprocess
    import sys

    all_run_paths = []
    dataset_names = [d.get("name", f"dataset_{i}") for i, d in enumerate(datasets)]

    for dataset in datasets:
        dataset_name = dataset.get("name", "unknown")
        dataset_path = dataset.get("path")
        if not dataset_path:
            continue

        # Run full ablation on this dataset
        run_paths = run_ablation_on_dataset(
            args=args,
            base_config=base_config,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            ablation_modes=ablation_modes,
            n_seeds=n_seeds,
        )
        all_run_paths.extend(run_paths)

    # Analyze cross-dataset results
    rows = compare_runs(all_run_paths)
    analysis = analyze_cross_dataset_results(rows, dataset_names, metric, ablation_modes)

    return analysis


def run_ablation_on_dataset(
    args: Any,
    base_config: Dict[str, Any],
    dataset_name: str,
    dataset_path: str,
    ablation_modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> List[str]:
    """Run the standard ablation suite on a single dataset."""
    from pathlib import Path
    import subprocess
    import sys

    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)
    arm_modes = ["local_base", *delayed_modes]

    # Add mechanism controls if configured
    if base_config.get("mechanism_controls"):
        for mode in delayed_modes:
            control = _DELAYED_MERGE_MECHANISM_CONTROLS.get(mode)
            if control and control not in arm_modes:
                arm_modes.append(control)

    balance_coef = base_config.get("balance_coef", 0.01)
    run_paths = []

    for mode in arm_modes:
        for coefficient in (0.0, balance_coef):
            for seed in range(n_seeds):
                run_label = f"crossds_{dataset_name}_{mode}_bal{format(coefficient, 'g')}_seed{seed}"
                run_dir = Path(base_config["out"]) / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, "experiment.py",
                    "--data", dataset_path,
                    "--mode", mode,
                    "--balance-coef", str(coefficient),
                    "--seed", str(seed),
                    "--out", str(run_dir),
                    "--steps", str(base_config.get("steps", 1000)),
                ]

                # Add other config
                for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                            "seq_len", "d_model", "n_heads", "n_layers", "n_experts",
                            "expert_hidden", "top_k", "k1", "k2", "residual_scale", "residual_scaling",
                            "n_kv_heads", "init_scheme"]:
                    if key in base_config:
                        cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

                print(f"[cross-dataset] {dataset_name}: {run_label}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    run_paths.append(str(run_dir))
                else:
                    print(f"[cross-dataset] FAILED: {run_label}")

    return run_paths


def analyze_cross_dataset_results(
    rows: List[Dict[str, object]],
    dataset_names: List[str],
    metric: str,
    ablation_modes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Analyze results across datasets.

    Performs:
    1. Per-dataset ablation analysis
    2. Cross-dataset component consistency (meta-analysis)
    3. Dataset heterogeneity quantification
    4. Generalization gap analysis
    """
    from .part_027_aggregation import aggregate_runs

    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)
    factors = ["dataset", "mode", "balance_coef"]

    report: Dict[str, object] = {
        "metric": metric,
        "datasets": dataset_names,
        "ablation_modes": delayed_modes,
        "per_dataset": {},
        "cross_dataset": {},
        "meta_analysis": {},
        "heterogeneity": {},
        "generalization": {},
        "advisories": [],
    }

    # Per-dataset analysis
    for dataset in dataset_names:
        ds_rows = [r for r in rows if r.get("dataset") == dataset]
        if not ds_rows:
            report["advisories"].append(f"No runs found for dataset {dataset}")
            continue

        # Standard ablation analysis
        coverage = ablation_coverage(ds_rows, ["mode", "balance_coef"], ablation_modes)
        aggregates = aggregate_runs(ds_rows, ["mode", "balance_coef"])
        factorial = factorial_effects(aggregates, ["mode", "balance_coef"], metric)
        mechanism = mechanism_control_report(
            aggregates, ["mode", "balance_coef"], metric
        )

        verdicts = None
        if factorial.get("applicable"):
            from .part_029_ablation import ablation_component_verdicts
            verdicts = ablation_component_verdicts(factorial, coverage, mechanism=mechanism)

        report["per_dataset"][dataset] = {
            "coverage": coverage,
            "factorial": factorial,
            "mechanism": mechanism,
            "verdicts": verdicts,
            "n_runs": len(ds_rows),
        }

    # Cross-dataset meta-analysis
    report["meta_analysis"] = run_meta_analysis(report["per_dataset"], metric)

    # Heterogeneity analysis
    report["heterogeneity"] = analyze_heterogeneity(report["per_dataset"], metric)

    # Generalization analysis
    report["generalization"] = analyze_generalization(report["per_dataset"], metric)

    return report


def run_meta_analysis(
    per_dataset: Dict[str, Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Meta-analysis of component effects across datasets.

    Uses fixed-effect and random-effects models to combine
    effect sizes across datasets.
    """
    components = ["delayed_merge_A", "delayed_merge_B", "balance_loss"]
    meta: Dict[str, object] = {
        "components": {},
        "overall": {},
    }

    for comp in components:
        # Collect effect sizes and standard errors from each dataset
        effects = []
        for dataset, data in per_dataset.items():
            verdicts = data.get("verdicts", {})
            for c in verdicts.get("components", []):
                if c.get("component") == comp:
                    primary = c.get("primary", {})
                    mean_effect = primary.get("mean_effect")
                    se = primary.get("se")
                    n = primary.get("n_pairs", 0)
                    if mean_effect is not None and se is not None and se > 0:
                        effects.append({
                            "dataset": dataset,
                            "effect": mean_effect,
                            "se": se,
                            "n": n,
                            "weight": 1.0 / (se ** 2),  # Inverse variance weight
                        })
                    break

        if len(effects) < 2:
            meta["components"][comp] = {"applicable": False, "reason": "Insufficient datasets"}
            continue

        # Fixed-effect meta-analysis
        total_weight = sum(e["weight"] for e in effects)
        fixed_effect = sum(e["effect"] * e["weight"] for e in effects) / total_weight
        fixed_se = math.sqrt(1.0 / total_weight)
        fixed_z = fixed_effect / fixed_se if fixed_se > 0 else 0

        # Heterogeneity (Q statistic)
        Q = sum(e["weight"] * (e["effect"] - fixed_effect) ** 2 for e in effects)
        df = len(effects) - 1
        # Chi-square survival approximation (Wilson-Hilferty normal approx)
        if df > 0 and Q > 0:
            z_wh = ((Q / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
            p_het = 0.5 * math.erfc(z_wh / math.sqrt(2))
            p_het = max(0.0, min(1.0, p_het))
        else:
            p_het = 1.0

        # Random-effects (DerSimonian-Laird)
        tau2 = max(0, (Q - df) / (total_weight - sum(e["weight"]**2 for e in effects) / total_weight)) if total_weight > 0 else 0
        re_weights = [1.0 / (e["se"]**2 + tau2) for e in effects]
        total_re_weight = sum(re_weights)
        random_effect = sum(e["effect"] * w for e, w in zip(effects, re_weights)) / total_re_weight if total_re_weight > 0 else 0
        random_se = math.sqrt(1.0 / total_re_weight) if total_re_weight > 0 else 0

        # I² statistic
        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

        meta["components"][comp] = {
            "n_datasets": len(effects),
            "fixed_effect": fixed_effect,
            "fixed_se": fixed_se,
            "fixed_z": fixed_z,
            "random_effect": random_effect,
            "random_se": random_se,
            "tau2": tau2,
            "I2": I2,
            "Q": Q,
            "df": df,
            "p_heterogeneity": p_het,
            "dataset_effects": {e["dataset"]: e["effect"] for e in effects},
        }

        # Overall verdict
        if abs(fixed_effect) > 1.96 * fixed_se:
            meta["components"][comp]["verdict"] = "significant_fixed"
        elif abs(random_effect) > 1.96 * random_se:
            meta["components"][comp]["verdict"] = "significant_random"
        else:
            meta["components"][comp]["verdict"] = "not_significant"

    # Overall meta-analysis
    sig_fixed = sum(1 for c in meta["components"].values() if c.get("verdict") == "significant_fixed")
    sig_random = sum(1 for c in meta["components"].values() if c.get("verdict") == "significant_random")

    meta["overall"] = {
        "n_components": len(components),
        "significant_fixed": sig_fixed,
        "significant_random": sig_random,
        "verdict": (
            "robust" if sig_fixed == len(components) else
            "partially_robust" if sig_random > 0 else
            "not_robust"
        ),
    }

    return meta


def analyze_heterogeneity(
    per_dataset: Dict[str, Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Analyze heterogeneity of results across datasets.

    Quantifies how much results vary across datasets beyond sampling error.
    """
    components = ["delayed_merge_A", "delayed_merge_B", "balance_loss"]
    heterogeneity: Dict[str, object] = {
        "components": {},
        "overall": {},
    }

    for comp in components:
        effects = []
        for dataset, data in per_dataset.items():
            verdicts = data.get("verdicts", {})
            for c in verdicts.get("components", []):
                if c.get("component") == comp:
                    primary = c.get("primary", {})
                    mean_effect = primary.get("mean_effect")
                    if mean_effect is not None:
                        effects.append((dataset, mean_effect))
                    break

        if len(effects) < 2:
            continue

        dataset_names = [e[0] for e in effects]
        effect_vals = [e[1] for e in effects]

        mean_effect = sum(effect_vals) / len(effect_vals)
        var_effect = sum((e - mean_effect) ** 2 for e in effect_vals) / len(effect_vals)
        sd_effect = math.sqrt(var_effect)

        # Coefficient of variation
        cv = sd_effect / abs(mean_effect) if mean_effect != 0 else float('inf')

        # Range
        effect_range = max(effect_vals) - min(effect_vals)

        heterogeneity["components"][comp] = {
            "datasets": dataset_names,
            "effects": dict(effects),
            "mean_effect": mean_effect,
            "sd_effect": sd_effect,
            "cv": cv,
            "range": effect_range,
            "heterogeneous": cv > 0.5 or effect_range > abs(mean_effect),
        }

    # Overall heterogeneity
    het_components = [c for c, h in heterogeneity["components"].items() if h.get("heterogeneous")]
    heterogeneity["overall"] = {
        "n_heterogeneous": len(het_components),
        "heterogeneous_components": het_components,
        "verdict": "homogeneous" if len(het_components) == 0 else "heterogeneous",
    }

    return heterogeneity


def analyze_generalization(
    per_dataset: Dict[str, Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Analyze generalization: do conclusions from one dataset hold on others?

    Uses leave-one-dataset-out cross-validation style analysis.
    """
    datasets = list(per_dataset.keys())
    if len(datasets) < 2:
        return {"applicable": False, "reason": "Need at least 2 datasets"}

    components = ["delayed_merge_A", "delayed_merge_B", "balance_loss"]
    generalization: Dict[str, object] = {
        "leave_one_out": {},
        "components": {},
        "overall": {},
    }

    for comp in components:
        # Get verdict for each dataset
        dataset_verdicts = {}
        for dataset in datasets:
            data = per_dataset[dataset]
            verdicts = data.get("verdicts", {})
            for c in verdicts.get("components", []):
                if c.get("component") == comp:
                    dataset_verdicts[dataset] = c.get("status")
                    break

        if len(dataset_verdicts) < 2:
            continue

        # Leave-one-out: if we exclude each dataset, does the conclusion change?
        loo_results = {}
        for holdout in datasets:
            train_verdicts = {d: v for d, v in dataset_verdicts.items() if d != holdout}
            if not train_verdicts:
                continue

            # Majority verdict on training datasets
            statuses = list(train_verdicts.values())
            majority = max(set(statuses), key=statuses.count)
            holdout_status = dataset_verdicts.get(holdout)

            loo_results[holdout] = {
                "train_majority": majority,
                "holdout_status": holdout_status,
                "consistent": holdout_status == majority,
            }

        # Overall consistency
        consistent_count = sum(1 for r in loo_results.values() if r.get("consistent"))

        generalization["components"][comp] = {
            "dataset_verdicts": dataset_verdicts,
            "leave_one_out": loo_results,
            "consistency_rate": consistent_count / len(loo_results) if loo_results else 0,
            "verdict": "consistent" if consistent_count == len(loo_results) else "inconsistent",
        }

    # Overall
    comp_verdicts = [g.get("verdict") for g in generalization["components"].values()]
    generalization["overall"] = {
        "verdict": "consistent" if all(v == "consistent" for v in comp_verdicts) else "inconsistent",
        "components_consistent": sum(1 for v in comp_verdicts if v == "consistent"),
        "components_total": len(comp_verdicts),
    }

    return generalization


def print_cross_dataset_report(report: Dict[str, object]) -> None:
    """Print formatted cross-dataset validation report."""
    print("\n" + "=" * 80)
    print("CROSS-DATASET VALIDATION REPORT")
    print("=" * 80)

    print(f"\nMetric: {report.get('metric')}")
    print(f"Datasets: {', '.join(report.get('datasets', []))}")
    print(f"Ablation modes: {', '.join(report.get('ablation_modes', []))}")

    # Per-dataset verdicts
    print("\n--- Per-Dataset Component Verdicts ---")
    for dataset, data in report.get("per_dataset", {}).items():
        print(f"\n  {dataset}:")
        verdicts = data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            print(f"    {comp.get('label')}: {comp.get('status')}")

    # Meta-analysis
    print("\n--- Meta-Analysis (Fixed & Random Effects) ---")
    for comp, meta in report.get("meta_analysis", {}).get("components", {}).items():
        if meta.get("applicable", True):
            print(f"\n  {comp}:")
            print(f"    Fixed effect: {meta.get('fixed_effect', 0):.6f} ± {meta.get('fixed_se', 0):.6f}")
            print(f"    Random effect: {meta.get('random_effect', 0):.6f} ± {meta.get('random_se', 0):.6f}")
            print(f"    I²: {meta.get('I2', 0):.1f}%, τ²: {meta.get('tau2', 0):.6f}")
            print(f"    Verdict: {meta.get('verdict')}")

    # Heterogeneity
    print("\n--- Heterogeneity ---")
    for comp, het in report.get("heterogeneity", {}).get("components", {}).items():
        print(f"  {comp}: CV={het.get('cv', 0):.2f}, range={het.get('range', 0):.6f}, "
              f"heterogeneous={het.get('heterogeneous')}")

    # Generalization
    print("\n--- Generalization (Leave-One-Out) ---")
    for comp, gen in report.get("generalization", {}).get("components", {}).items():
        print(f"  {comp}: consistency={gen.get('consistency_rate', 0):.1%}, "
              f"verdict={gen.get('verdict')}")
        for holdout, res in gen.get("leave_one_out", {}).items():
            print(f"    Exclude {holdout}: train majority={res.get('train_majority')}, "
                  f"holdout={res.get('holdout_status')}, consistent={res.get('consistent')}")

    # Overall
    meta_overall = report.get("meta_analysis", {}).get("overall", {})
    print(f"\n--- Overall Meta-Analysis Verdict: {meta_overall.get('verdict')} ---")
    print(f"  Significant (fixed): {meta_overall.get('significant_fixed')}/{meta_overall.get('n_components')}")
    print(f"  Significant (random): {meta_overall.get('significant_random')}/{meta_overall.get('n_components')}")

    het_overall = report.get("heterogeneity", {}).get("overall", {})
    print(f"\n--- Overall Heterogeneity Verdict: {het_overall.get('verdict')} ---")
    print(f"  Heterogeneous components: {', '.join(het_overall.get('heterogeneous_components', [])) or 'none'}")

    gen_overall = report.get("generalization", {}).get("overall", {})
    print(f"\n--- Overall Generalization Verdict: {gen_overall.get('verdict')} ---")
    print(f"  Consistent components: {gen_overall.get('components_consistent')}/{gen_overall.get('components_total')}")

    # Advisories
    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")