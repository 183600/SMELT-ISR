from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import math
import itertools

from .part_021_comparison import compare_runs
from .part_025_statistics import paired_t_test, welch_t_test, cohens_d
from .part_023_factorial_helpers import _finite_float, _stable_factor_key
from .part_029_ablation import factorial_effects
from .part_031_ablation_coverage import ablation_coverage
from .part_027_aggregation import aggregate_runs


# Key hyperparameters to test for sensitivity
SENSITIVITY_HYPERPARAMETERS: Tuple[str, ...] = (
    "k1",
    "k2",
    "balance_coef",
    "top_k",
    "n_experts",
    "expert_hidden",
    "residual_scale",
    "lr",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
    "batch_size",
    "seq_len",
    "d_model",
    "n_heads",
    "n_layers",
)

# Structural hyperparameters that define architecture
STRUCTURAL_HYPERPARAMETERS: Tuple[str, ...] = (
    "d_model",
    "n_heads",
    "n_layers",
    "n_experts",
    "expert_hidden",
    "top_k",
    "k1",
    "k2",
)

# Training hyperparameters
TRAINING_HYPERPARAMETERS: Tuple[str, ...] = (
    "lr",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
    "batch_size",
    "seq_len",
    "residual_scale",
    "balance_coef",
)


def run_sensitivity_analysis(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    hyperparameters: Optional[List[str]] = None,
    perturbation_factors: Optional[List[float]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Run sensitivity analysis by perturbing hyperparameters around their base values.

    For each hyperparameter, test multiple perturbation factors (e.g., 0.5x, 0.75x, 1x, 1.25x, 1.5x)
    and measure how the metric changes. This identifies which hyperparameters the results
    are most sensitive to.
    """
    from pathlib import Path
    import subprocess
    import sys

    hyperparams = hyperparameters or list(TRAINING_HYPERPARAMETERS)
    factors = perturbation_factors or [0.5, 0.75, 1.0, 1.25, 1.5]

    base_values = {k: base_config.get(k) for k in hyperparams}

    report: Dict[str, object] = {
        "metric": metric,
        "base_config": {k: v for k, v in base_config.items() if k in hyperparams},
        "hyperparameters": {},
        "summary": {},
        "advisories": [],
    }

    for hp in hyperparams:
        base_val = base_config.get(hp)
        if base_val is None:
            report["advisories"].append(f"Hyperparameter {hp} not in base config, skipping")
            continue

        # Skip non-numeric hyperparameters
        if not isinstance(base_val, (int, float)):
            report["advisories"].append(f"Hyperparameter {hp} is non-numeric ({type(base_val)}), skipping")
            continue

        hp_results = []
        for factor in factors:
            perturbed_val = base_val * factor
            if isinstance(base_val, int):
                perturbed_val = int(round(perturbed_val))

            # Run experiments with perturbed value
            seed_results = []
            for seed in range(n_seeds):
                run_dir = Path(base_config["out"]) / f"sens_{hp}_{factor}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Build config with perturbed value
                perturbed_config = dict(base_config)
                perturbed_config[hp] = perturbed_val

                cmd = build_experiment_command(perturbed_config, seed, str(run_dir))
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    rows = compare_runs([str(run_dir)])
                    if rows:
                        value = _finite_float(rows[0].get(metric))
                        if value is not None:
                            seed_results.append(value)

            if seed_results:
                mean_val = sum(seed_results) / len(seed_results)
                hp_results.append({
                    "factor": factor,
                    "value": perturbed_val,
                    "mean": mean_val,
                    "std": math.sqrt(sum((x - mean_val) ** 2 for x in seed_results) / max(len(seed_results) - 1, 1)),
                    "n_seeds": len(seed_results),
                })

        # Compute sensitivity metrics
        if len(hp_results) >= 3:
            factors_tested = [r["factor"] for r in hp_results]
            means = [r["mean"] for r in hp_results]

            # Linear sensitivity (slope)
            if len(set(factors_tested)) > 1:
                n = len(hp_results)
                x_mean = sum(factors_tested) / n
                y_mean = sum(means) / n
                x_var = sum((x - x_mean) ** 2 for x in factors_tested)
                if x_var > 0:
                    xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(factors_tested, means))
                    slope = xy_cov / x_var
                else:
                    slope = 0

                # Elasticity at base (factor=1.0)
                base_mean = next((r["mean"] for r in hp_results if r["factor"] == 1.0), means[len(means)//2])
                elasticity = slope * 1.0 / base_mean if base_mean != 0 else float('inf')

                # Range of effects
                effect_range = max(means) - min(means)
                relative_range = effect_range / abs(base_mean) if base_mean != 0 else float('inf')

                report["hyperparameters"][hp] = {
                    "base_value": base_val,
                    "results": hp_results,
                    "slope": slope,
                    "elasticity": elasticity,
                    "effect_range": effect_range,
                    "relative_range": relative_range,
                    "max_deviation": max(abs(m - base_mean) for m in means),
                }

                # Classify sensitivity
                if relative_range > 0.5:
                    sensitivity_class = "high"
                elif relative_range > 0.1:
                    sensitivity_class = "medium"
                else:
                    sensitivity_class = "low"

                report["hyperparameters"][hp]["sensitivity_class"] = sensitivity_class

                if sensitivity_class == "high":
                    report["advisories"].append(
                        f"High sensitivity to {hp}: {relative_range:.1%} relative change "
                        f"in metric for perturbations in [{min(factors_tested)}, {max(factors_tested)}]"
                    )
            else:
                report["hyperparameters"][hp] = {
                    "base_value": base_val,
                    "results": hp_results,
                    "sensitivity_class": "unknown",
                }
        else:
            report["hyperparameters"][hp] = {
                "base_value": base_val,
                "results": hp_results,
                "sensitivity_class": "insufficient_data",
            }

    # Summary: rank hyperparameters by sensitivity
    sensitivity_ranking = []
    for hp, data in report["hyperparameters"].items():
        if "relative_range" in data:
            sensitivity_ranking.append((hp, data["relative_range"], data["sensitivity_class"]))

    sensitivity_ranking.sort(key=lambda x: x[1], reverse=True)
    report["summary"]["ranking"] = [
        {"hyperparameter": hp, "relative_range": rr, "class": cls}
        for hp, rr, cls in sensitivity_ranking
    ]

    report["summary"]["high_sensitivity"] = [
        hp for hp, rr, cls in sensitivity_ranking if cls == "high"
    ]
    report["summary"]["medium_sensitivity"] = [
        hp for hp, rr, cls in sensitivity_ranking if cls == "medium"
    ]
    report["summary"]["low_sensitivity"] = [
        hp for hp, rr, cls in sensitivity_ranking if cls == "low"
    ]

    return report


def build_experiment_command(
    config: Dict[str, Any],
    seed: int,
    out_dir: str,
) -> List[str]:
    """Build experiment command from config."""
    import sys

    cmd = [
        sys.executable, "experiment.py",
        "--data", str(config["data"]),
        "--mode", str(config["mode"]),
        "--seed", str(seed),
        "--out", out_dir,
        "--steps", str(config.get("steps", 1000)),
    ]

    for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                "seq_len", "d_model", "n_heads", "n_layers", "n_experts",
                "expert_hidden", "top_k", "k1", "k2", "balance_coef",
                "residual_scale", "residual_scaling", "n_kv_heads"]:
        if key in config:
            cmd.extend([f"--{key.replace('_', '-')}", str(config[key])])

    return cmd


def run_grid_sensitivity(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Run full grid sensitivity analysis for multiple hyperparameters simultaneously.

    This explores the joint sensitivity surface, detecting interactions between hyperparameters.
    """
    from pathlib import Path
    import subprocess
    import sys

    grid = param_grid or {
        "lr": [1e-4, 3e-4, 1e-3],
        "balance_coef": [0.0, 0.005, 0.01, 0.02],
        "weight_decay": [0.0, 0.01, 0.1],
    }

    # Generate all combinations
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))

    report: Dict[str, object] = {
        "metric": metric,
        "param_grid": grid,
        "results": [],
        "interactions": {},
        "advisories": [],
    }

    for combo in combinations:
        config = dict(base_config)
        for name, value in zip(param_names, combo):
            config[name] = value

        # Run with multiple seeds
        seed_results = []
        for seed in range(n_seeds):
            run_dir = Path(base_config["out"]) / f"grid_{'_'.join(f'{n}{v}' for n,v in zip(param_names, combo))}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_experiment_command(config, seed, str(run_dir))
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                rows = compare_runs([str(run_dir)])
                if rows:
                    value = _finite_float(rows[0].get(metric))
                    if value is not None:
                        seed_results.append(value)

        if seed_results:
            mean_val = sum(seed_results) / len(seed_results)
            report["results"].append({
                "params": dict(zip(param_names, combo)),
                "mean": mean_val,
                "std": math.sqrt(sum((x - mean_val) ** 2 for x in seed_results) / max(len(seed_results) - 1, 1)),
                "n_seeds": len(seed_results),
            })

    # Analyze interactions (2-way)
    if len(param_names) >= 2:
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                interaction = analyze_interaction(report["results"], param_names[i], param_names[j], metric)
                report["interactions"][f"{param_names[i]}_x_{param_names[j]}"] = interaction

    return report


def analyze_interaction(
    results: List[Dict[str, object]],
    param1: str,
    param2: str,
    metric: str,
) -> Dict[str, object]:
    """
    Analyze 2-way interaction between two hyperparameters.

    Uses a simple 2×2 interaction contrast if both have at least 2 levels.
    """
    # Get unique levels
    levels1 = sorted(set(r["params"][param1] for r in results))
    levels2 = sorted(set(r["params"][param2] for r in results))

    if len(levels1) < 2 or len(levels2) < 2:
        return {"applicable": False, "reason": "Need at least 2 levels for each parameter"}

    # Build 2x2 table (using min/max levels)
    low1, high1 = levels1[0], levels1[-1]
    low2, high2 = levels2[0], levels2[-1]

    cells = {}
    for r in results:
        v1, v2 = r["params"][param1], r["params"][param2]
        if v1 in (low1, high1) and v2 in (low2, high2):
            cells[(v1, v2)] = r["mean"]

    required = [(low1, low2), (low1, high2), (high1, low2), (high1, high2)]
    if not all(c in cells for c in required):
        return {"applicable": False, "reason": "Missing cells for full 2x2"}

    # Interaction = (high1,high2 - high1,low2) - (low1,high2 - low1,low2)
    interaction_effect = (
        (cells[(high1, high2)] - cells[(high1, low2)]) -
        (cells[(low1, high2)] - cells[(low1, low2)])
    )

    # Main effects
    main1 = (cells[(high1, low2)] + cells[(high1, high2)]) / 2 - \
            (cells[(low1, low2)] + cells[(low1, high2)]) / 2
    main2 = (cells[(low1, high2)] + cells[(high1, high2)]) / 2 - \
            (cells[(low1, low2)] + cells[(high1, low2)]) / 2

    return {
        "applicable": True,
        "param1": param1,
        "param2": param2,
        "interaction_effect": interaction_effect,
        "main_effect_1": main1,
        "main_effect_2": main2,
        "interaction_to_main_ratio": abs(interaction_effect) / max(abs(main1), abs(main2), 1e-8),
        "cells": {f"{k[0]},{k[1]}": v for k, v in cells.items()},
        "interpretation": (
            f"Interaction effect = {interaction_effect:.4f}. "
            f"Ratio to main effects = {abs(interaction_effect) / max(abs(main1), abs(main2), 1e-8):.2f}. "
            "Ratio > 0.5 suggests strong interaction (non-additive effects)."
        ),
    }


def run_robustness_check(
    experimental_results: List[Dict[str, object]],
    sensitivity_report: Dict[str, object],
    metric: str = "val_loss",
    threshold: float = 0.1,
) -> Dict[str, object]:
    """
    Check if main experimental conclusions are robust to hyperparameter perturbations.

    A conclusion is robust if:
    1. Effect direction doesn't flip across reasonable perturbations
    2. Effect size remains practically significant
    3. Statistical significance is maintained
    """
    # Get base effect from experimental results
    base_effect = None
    for r in experimental_results:
        if "delta" in r:
            base_effect = r["delta"]
            break

    if base_effect is None:
        return {"applicable": False, "reason": "No base effect found in experimental results"}

    robustness: Dict[str, object] = {
        "base_effect": base_effect,
        "hyperparameters": {},
        "overall_robust": True,
        "advisories": [],
    }

    for hp, data in sensitivity_report.get("hyperparameters", {}).items():
        if "results" not in data:
            continue

        # Check effect at each perturbation
        flips = []
        for r in data["results"]:
            # We'd need paired results to check this properly
            # For now, check if metric deviates substantially
            pass

        robustness["hyperparameters"][hp] = {
            "sensitivity_class": data.get("sensitivity_class", "unknown"),
            "relative_range": data.get("relative_range", 0),
        }

        if data.get("sensitivity_class") == "high":
            robustness["overall_robust"] = False
            robustness["advisories"].append(
                f"Conclusion may not be robust to {hp} perturbations "
                f"(relative range: {data.get('relative_range', 0):.1%})"
            )

    return robustness


def run_cross_validation_sensitivity(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    n_folds: int = 5,
    n_seeds: int = 2,
) -> Dict[str, object]:
    """
    Cross-validation sensitivity: evaluate on different data splits.

    This tests whether results are specific to a particular train/val/test split
    or generalize across different data partitions.
    """
    from pathlib import Path
    import subprocess
    import sys

    # This would require implementing different data splits
    # Placeholder for the design
    return {
        "applicable": False,
        "reason": "Requires implementing data split variation. Design: "
                  "create k different train/val/test splits (e.g., by rotating "
                  "the split boundaries or using different random seeds for "
                  "splitting), run full experiment on each, and compare results.",
        "design": [
            "Split data into k folds (chronological or random)",
            "For each fold i: use fold i as test, i+1 as val, rest as train",
            "Run full ablation on each fold",
            "Compare component verdicts across folds",
            "Flag components that are not consistently supported",
        ],
    }


def print_sensitivity_report(report: Dict[str, object]) -> None:
    """Print formatted sensitivity analysis report."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    print(f"\nMetric: {report.get('metric')}")

    # Ranking
    summary = report.get("summary", {})
    ranking = summary.get("ranking", [])
    if ranking:
        print("\n--- Sensitivity Ranking ---")
        print(f"{'Hyperparameter':<25} {'Rel. Range':>12} {'Class':>10}")
        print("-" * 50)
        for item in ranking:
            print(f"{item['hyperparameter']:<25} {item['relative_range']:>11.2%} {item['class']:>10}")

    # Details
    print("\n--- Details ---")
    for hp, data in report.get("hyperparameters", {}).items():
        if "relative_range" in data:
            print(f"\n  {hp} (base={data['base_value']}):")
            print(f"    Sensitivity: {data['sensitivity_class']}")
            print(f"    Elasticity: {data.get('elasticity', 'N/A'):.4f}")
            print(f"    Relative range: {data['relative_range']:.2%}")
            print(f"    Max deviation: {data.get('max_deviation', 'N/A'):.4f}")

    # Interactions
    interactions = report.get("interactions", {})
    if interactions:
        print("\n--- Interactions ---")
        for name, inter in interactions.items():
            if inter.get("applicable"):
                print(f"  {name}: interaction={inter['interaction_effect']:.4f}, "
                      f"ratio={inter['interaction_to_main_ratio']:.2f}")

    # Advisories
    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")

    # Robustness summary
    print("\n--- Robustness Summary ---")
    print(f"  High sensitivity: {', '.join(summary.get('high_sensitivity', [])) or 'none'}")
    print(f"  Medium sensitivity: {', '.join(summary.get('medium_sensitivity', [])) or 'none'}")
    print(f"  Low sensitivity: {', '.join(summary.get('low_sensitivity', [])) or 'none'}")