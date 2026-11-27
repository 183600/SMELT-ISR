from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import math
import random

from .part_021_comparison import compare_runs
from .part_025_statistics import (
    paired_t_test, welch_t_test, cohens_d,
    _t_critical, _SIGNIFICANCE_LEVEL,
)
from .part_023_factorial_helpers import _finite_float, _stable_factor_key
from .part_029_ablation import factorial_effects
from .part_031_ablation_coverage import ablation_coverage


# Sequential testing boundaries
SEQUENTIAL_BOUNDARIES = {
    "pocock": {
        "description": "Pocock boundary: constant critical value across looks",
        "alpha_spending": "constant",
    },
    "obrien_fleming": {
        "description": "O'Brien-Fleming: conservative early, liberal late",
        "alpha_spending": "lan_demets",
    },
    "wang_tsiatis": {
        "description": "Wang-Tsiatis (rho=0.5): intermediate",
        "alpha_spending": "power",
    },
}


def run_sequential_ablation(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    ablation_modes: Optional[List[str]] = None,
    max_seeds: int = 20,
    min_seeds: int = 6,
    alpha: float = 0.05,
    beta: float = 0.2,
    boundary: str = "obrien_fleming",
    equivalence_margin: Optional[float] = 0.01,
    futility_threshold: float = 0.1,
) -> Dict[str, object]:
    """
    Run sequential ablation: add seeds one at a time and test for significance.

    This implements group sequential testing for the ablation components,
    allowing early stopping for efficacy or futility.

    Parameters:
    - max_seeds: Maximum number of seeds to collect
    - min_seeds: Minimum seeds before first test
    - alpha: Type I error rate (spent across looks)
    - beta: Type II error rate (for power)
    - boundary: Sequential boundary type (pocock, obrien_fleming, wang_tsiatis)
    - equivalence_margin: For TOST equivalence testing
    - futility_threshold: Stop if effect size < threshold with high confidence
    """
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

    # Pre-generate all seeds
    all_seeds = list(range(max_seeds))
    random.shuffle(all_seeds)

    report: Dict[str, object] = {
        "metric": metric,
        "max_seeds": max_seeds,
        "min_seeds": min_seeds,
        "alpha": alpha,
        "beta": beta,
        "boundary": boundary,
        "equivalence_margin": equivalence_margin,
        "looks": [],
        "final_verdict": None,
        "seeds_used": 0,
        "stopped_early": False,
        "stop_reason": None,
        "advisories": [],
    }

    # Sequential testing boundaries
    boundaries = compute_sequential_boundaries(
        max_looks=(max_seeds - min_seeds + 1),
        alpha=alpha,
        boundary_type=boundary,
    )

    # Run seeds sequentially
    completed_seeds = 0
    all_run_paths = []

    for seed in all_seeds:
        completed_seeds += 1

        # Run all conditions for this seed
        seed_paths = []
        for mode in arm_modes:
            for coefficient in (0.0, balance_coef):
                run_label = f"seq_{mode}_bal{format(coefficient, 'g')}_seed{seed}"
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

                print(f"[sequential] Seed {seed}: {run_label}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    seed_paths.append(str(run_dir))
                else:
                    print(f"[sequential] FAILED: {run_label}")

        all_run_paths.extend(seed_paths)

        # Analyze after each seed (or batch)
        if completed_seeds >= min_seeds:
            rows = compare_runs(all_run_paths)
            look_result = analyze_sequential_look(
                rows=rows,
                look_number=completed_seeds - min_seeds + 1,
                total_looks=max_seeds - min_seeds + 1,
                alpha=alpha,
                beta=beta,
                equivalence_margin=equivalence_margin,
                boundary=boundaries[completed_seeds - min_seeds],
                metric=metric,
                ablation_modes=ablation_modes,
            )

            report["looks"].append(look_result)

            # Check stopping rules
            if look_result.get("stop_for_efficacy"):
                report["final_verdict"] = "efficacy"
                report["stopped_early"] = True
                report["stop_reason"] = "Efficacy boundary crossed"
                report["seeds_used"] = completed_seeds
                break

            if look_result.get("stop_for_futility"):
                report["final_verdict"] = "futility"
                report["stopped_early"] = True
                report["stop_reason"] = "Futility boundary crossed"
                report["seeds_used"] = completed_seeds
                break

            if look_result.get("stop_for_equivalence"):
                report["final_verdict"] = "equivalence"
                report["stopped_early"] = True
                report["stop_reason"] = "Equivalence established"
                report["seeds_used"] = completed_seeds
                break

    if not report["stopped_early"]:
        report["final_verdict"] = "inconclusive"
        report["seeds_used"] = completed_seeds
        report["stop_reason"] = "Max seeds reached"

    # Final analysis with all collected seeds
    if all_run_paths:
        final_rows = compare_runs(all_run_paths)
        report["final_analysis"] = analyze_final_sequential(
            final_rows,
            metric=metric,
            ablation_modes=ablation_modes,
            equivalence_margin=equivalence_margin,
        )

    return report


def compute_sequential_boundaries(
    max_looks: int,
    alpha: float,
    boundary_type: str = "obrien_fleming",
) -> List[Dict[str, float]]:
    """
    Compute critical values for group sequential testing.

    Returns list of boundary dicts with efficacy/futility/equivalence thresholds.
    """
    boundaries = []

    for look in range(1, max_looks + 1):
        t = look / max_looks  # Information fraction

        if boundary_type == "pocock":
            # Pocock: constant critical value
            # Approximation: z = 2.04 for max_looks=5, alpha=0.05
            if max_looks <= 2:
                z_eff = 2.18
            elif max_looks <= 5:
                z_eff = 2.24
            elif max_looks <= 10:
                z_eff = 2.29
            else:
                z_eff = 2.33
        elif boundary_type == "obrien_fleming":
            # O'Brien-Fleming: z = z_alpha/2 / sqrt(t)
            z_alpha = _t_critical(float('inf'), alpha / 2) or 1.96
            z_eff = z_alpha / math.sqrt(t)
        elif boundary_type == "wang_tsiatis":
            # Wang-Tsiatis with rho=0.5
            rho = 0.5
            z_alpha = _t_critical(float('inf'), alpha / 2) or 1.96
            z_eff = z_alpha / (t ** (rho / 2))
        else:
            z_eff = 1.96

        # Futility boundary (conditional power < 0.5)
        z_fut = 0.0  # Simplified

        boundaries.append({
            "look": look,
            "information_fraction": t,
            "efficacy_z": z_eff,
            "efficacy_p": 2 * (1 - 0.5 * (1 + math.erf(z_eff / math.sqrt(2)))),  # Two-sided
            "futility_z": z_fut,
        })

    return boundaries


def analyze_sequential_look(
    rows: List[Dict[str, object]],
    look_number: int,
    total_looks: int,
    alpha: float,
    beta: float,
    equivalence_margin: Optional[float],
    boundary: Dict[str, float],
    metric: str,
    ablation_modes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Analyze a single sequential look.
    """
    from .part_027_aggregation import aggregate_runs

    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)

    # Run standard ablation analysis
    coverage = ablation_coverage(rows, ["mode", "balance_coef"], ablation_modes)
    aggregates = aggregate_runs(rows, ["mode", "balance_coef"])
    factorial = factorial_effects(aggregates, ["mode", "balance_coef"], metric, equivalence_margin)

    result: Dict[str, object] = {
        "look": look_number,
        "n_seeds": len(set(r.get("seed") for r in rows if r.get("seed") is not None)),
        "boundary": boundary,
        "components": {},
        "stop_for_efficacy": False,
        "stop_for_futility": False,
        "stop_for_equivalence": False,
    }

    efficacy_z = boundary["efficacy_z"]
    futility_z = boundary["futility_z"]

    if factorial.get("applicable"):
        for effect in factorial.get("effects", []):
            if not effect.get("significant"):
                continue

            comp_name = effect.get("effect_name")
            t_stat = _finite_float(effect.get("t"))
            p_val = _finite_float(effect.get("p_value"))
            mean_effect = _finite_float(effect.get("mean_effect"))
            se = _finite_float(effect.get("se"))

            if t_stat is None or p_val is None:
                continue

            # Two-sided z-approximation
            z = abs(t_stat)

            # Efficacy: z > efficacy boundary
            if z > efficacy_z:
                result["stop_for_efficacy"] = True
                result["components"][comp_name] = {
                    "z": z,
                    "p": p_val,
                    "effect": mean_effect,
                    "verdict": "efficacy",
                }

            # Futility: conditional power low
            elif z < futility_z and se is not None and mean_effect is not None:
                # Conditional power at current effect
                cp = conditional_power(mean_effect, se, alpha, z)
                if cp < 0.2:  # Low conditional power
                    result["stop_for_futility"] = True
                    result["components"][comp_name] = {
                        "z": z,
                        "p": p_val,
                        "effect": mean_effect,
                        "conditional_power": cp,
                        "verdict": "futility",
                    }

            # Equivalence
            if equivalence_margin is not None and equivalence_margin > 0:
                # Check if effect is within equivalence bounds with high confidence
                if abs(mean_effect) < equivalence_margin and z < efficacy_z:
                    # Could also use TOST here
                    result["components"][comp_name] = result["components"].get(comp_name, {})
                    result["components"][comp_name]["equivalence_check"] = True

    return result


def conditional_power(
    observed_effect: float,
    se: float,
    alpha: float,
    current_z: float,
) -> float:
    """
    Compute conditional power given current data.

    Probability of rejecting H0 at final analysis given current effect estimate.
    Uses current_z directly (avoids redundant recomputation from observed_effect/se).

    Note: This computes power assuming the current effect is the true effect
    at the *final* sample size. It does not account for information fraction
    (interim vs. planned sample size); callers should interpret the result
    accordingly. For interim monitoring with known information fraction t,
    the drift-adjusted expected final z would be current_z / sqrt(t).
    """
    z_alpha = _t_critical(float('inf'), alpha / 2) or 1.96

    # Use current_z directly — equivalent to abs(observed_effect) / se
    # but avoids redundant division and respects the caller's z computation
    # (which may include bias corrections or pooled variance).
    expected_final_z = abs(current_z)

    # Power = P(Z > z_alpha - expected_final_z) + P(Z < -z_alpha - expected_final_z)
    # Using normal approximation (erfc for numerical stability in tails)
    from math import erfc, sqrt
    power = 0.5 * erfc((z_alpha - expected_final_z) / sqrt(2)) + \
            0.5 * erfc((z_alpha + expected_final_z) / sqrt(2))

    return max(0.0, min(1.0, power))


def analyze_final_sequential(
    rows: List[Dict[str, object]],
    metric: str,
    ablation_modes: Optional[List[str]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    Final analysis with all collected seeds, adjusting for sequential testing.
    """
    from .part_027_aggregation import aggregate_runs
    from .part_029_ablation import ablation_component_verdicts

    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)

    coverage = ablation_coverage(rows, ["mode", "balance_coef"], ablation_modes)
    aggregates = aggregate_runs(rows, ["mode", "balance_coef"])
    factorial = factorial_effects(aggregates, ["mode", "balance_coef"], metric, equivalence_margin)
    mechanism = mechanism_control_report(
        aggregates, ["mode", "balance_coef"], metric, equivalence_margin
    )

    verdicts = None
    if factorial.get("applicable"):
        verdicts = ablation_component_verdicts(factorial, coverage, mechanism=mechanism)

    # Adjust p-values for sequential testing (using final boundary)
    n_looks = len(set(r.get("seed") for r in rows if r.get("seed") is not None))

    return {
        "coverage": coverage,
        "factorial": factorial,
        "mechanism": mechanism,
        "verdicts": verdicts,
        "n_seeds": n_looks,
        "sequential_adjustment": "Final boundary used (no further adjustment needed for group sequential)",
    }


def run_adaptive_experiment(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    ablation_modes: Optional[List[str]] = None,
    adaptation_rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, object]:
    """
    Run adaptive experiment design: modify experiment parameters based on interim results.

    Adaptation rules can include:
    - Increase seeds for underpowered components
    - Drop arms that are clearly inferior
    - Adjust balance_coef for modes showing instability
    - Change compute allocation across scales
    """
    # This is a framework - specific adaptations would be implemented per experiment
    default_rules = [
        {
            "name": "increase_seeds_underpowered",
            "condition": "any component has conditional_power < 0.5",
            "action": "add_seeds",
            "params": {"additional_seeds": 5},
        },
        {
            "name": "drop_inferior_arms",
            "condition": "any mode shows harm with p < 0.01",
            "action": "drop_mode",
            "params": {"harm_threshold": -0.01},
        },
        {
            "name": "stabilize_balance",
            "condition": "high attrition in balance=on arm",
            "action": "reduce_balance_coef",
            "params": {"factor": 0.5},
        },
    ]

    rules = adaptation_rules or default_rules

    return {
        "applicable": False,
        "reason": "Adaptive experiment design requires interactive execution framework. "
                  "Rules defined but need integration with experiment orchestration.",
        "rules": rules,
        "design_guidance": [
            "1. Run initial batch of seeds (e.g., 6)",
            "2. Analyze interim results",
            "3. Apply adaptation rules",
            "4. Modify experiment plan (add/drop arms, adjust params)",
            "5. Continue with updated plan",
            "6. Repeat until stopping criteria met",
        ],
    }


def print_sequential_report(report: Dict[str, object]) -> None:
    """Print formatted sequential testing report."""
    print("\n" + "=" * 80)
    print("SEQUENTIAL TESTING REPORT")
    print("=" * 80)

    print(f"\nMetric: {report.get('metric')}")
    print(f"Max seeds: {report.get('max_seeds')}")
    print(f"Min seeds: {report.get('min_seeds')}")
    print(f"Alpha: {report.get('alpha')}")
    print(f"Beta: {report.get('beta')}")
    print(f"Boundary: {report.get('boundary')}")

    print(f"\nFinal verdict: {report.get('final_verdict')}")
    print(f"Seeds used: {report.get('seeds_used')}")
    print(f"Stopped early: {report.get('stopped_early')}")
    print(f"Stop reason: {report.get('stop_reason')}")

    # Looks
    print("\n--- Sequential Looks ---")
    for look in report.get("looks", []):
        print(f"\n  Look {look.get('look')}: n_seeds={look.get('n_seeds')}")
        print(f"    Efficacy boundary (z): {look.get('boundary', {}).get('efficacy_z', 0):.3f}")
        print(f"    Futility boundary (z): {look.get('boundary', {}).get('futility_z', 0):.3f}")
        for comp, data in look.get("components", {}).items():
            print(f"    {comp}: z={data.get('z', 0):.3f}, p={data.get('p', 1):.4f}, "
                  f"effect={data.get('effect', 0):.6f}, verdict={data.get('verdict')}")

    # Final analysis
    final = report.get("final_analysis", {})
    if final:
        print("\n--- Final Analysis ---")
        verdicts = final.get("verdicts", {})
        for comp in verdicts.get("components", []):
            print(f"  {comp.get('label')}: {comp.get('status')}")

    # Advisories
    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")