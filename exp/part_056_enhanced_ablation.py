from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import itertools

from .part_021_comparison import compare_runs
from .part_027_aggregation import aggregate_runs
from .part_029_ablation import factorial_effects, mechanism_control_report
from .part_031_ablation_coverage import ablation_coverage
from .part_030_factorial import (
    _DELAYED_MERGE_MODES,
    _DELAYED_MERGE_MECHANISM_CONTROLS,
    _ABLATION_STRUCTURAL_KEYS,
)
from .part_023_factorial_helpers import _stable_factor_key, _finite_float, _row_factor_value
from .part_028_contrasts import factor_contrasts
from .part_025_statistics import metric_higher_is_better


# Extended ablation factors beyond the basic (mode x balance_coef)
EXTENDED_ABLATION_FACTORS: Tuple[str, ...] = (
    "residual_scaling",
    "n_experts",
    "top_k",
    "residual_scale",
    "init_scheme",
)

# Factor configurations for extended ablation
FACTOR_LEVELS: Dict[str, List[Any]] = {
    "residual_scaling": ["paper", "global"],
    "init_scheme": ["module_order", "name_hash"],
    # n_experts, top_k, residual_scale are numeric - levels defined per experiment
}

# Default levels for numeric factors (relative to base)
NUMERIC_FACTOR_LEVELS: Dict[str, List[float]] = {
    "n_experts": [0.5, 1.0, 2.0],  # Multipliers
    "top_k": [0.5, 1.0, 2.0],
    "residual_scale": [0.5, 1.0, 2.0],
}


def build_extended_ablation_plan(
    seeds: List[int],
    base_config: Dict[str, Any],
    ablation_modes: Optional[List[str]] = None,
    extended_factors: Optional[List[str]] = None,
    factor_levels: Optional[Dict[str, List[Any]]] = None,
    balance_coef: float = 0.01,
    mechanism_controls: bool = False,
) -> List[Dict[str, object]]:
    """
    Build an extended ablation plan with additional factors.

    The base design is (1+N) x 2 (mode x balance_coef).
    Extended factors multiply this design.

    For categorical factors (residual_scaling, init_scheme): use specified levels
    For numeric factors (n_experts, top_k, residual_scale): use multipliers relative to base
    """
    delayed_modes = (
        list(dict.fromkeys(ablation_modes))
        if ablation_modes
        else list(_DELAYED_MERGE_MODES)
    )
    arm_modes = ["local_base", *delayed_modes]

    if mechanism_controls:
        for mode in delayed_modes:
            control = _DELAYED_MERGE_MECHANISM_CONTROLS.get(mode)
            if control and control not in arm_modes:
                arm_modes.append(control)

    # Base factorial: mode x balance_coef
    base_conditions = []
    for mode in arm_modes:
        for coefficient in (0.0, balance_coef):
            base_conditions.append({"mode": mode, "balance_coef": coefficient})

    # Extended factors
    ext_factors = extended_factors or []
    factor_values: Dict[str, List[Any]] = {}

    for factor in ext_factors:
        if factor in FACTOR_LEVELS:
            factor_values[factor] = factor_levels.get(factor, FACTOR_LEVELS[factor]) if factor_levels else FACTOR_LEVELS[factor]
        elif factor in NUMERIC_FACTOR_LEVELS:
            base_val = base_config.get(factor)
            if base_val is not None and isinstance(base_val, (int, float)):
                multipliers = NUMERIC_FACTOR_LEVELS[factor]
                factor_values[factor] = [base_val * m for m in multipliers]
            else:
                factor_values[factor] = []
        else:
            factor_values[factor] = []

    # Filter out factors with no levels
    factor_values = {k: v for k, v in factor_values.items() if v}

    # Generate full factorial
    plan: List[Dict[str, object]] = []
    factor_names = list(factor_values.keys())

    for base_cond in base_conditions:
        if not factor_names:
            # No extended factors - just base design
            for seed in seeds:
                plan.append({
                    **base_cond,
                    "seed": seed,
                })
        else:
            # Full factorial with extended factors
            factor_combinations = itertools.product(*factor_values.values())
            for combo in factor_combinations:
                ext_cond = dict(zip(factor_names, combo))
                for seed in seeds:
                    plan.append({
                        **base_cond,
                        **ext_cond,
                        "seed": seed,
                    })

    return plan


def run_extended_ablation(
    args: Any,
    out_root: Any,
    base_config: Dict[str, Any],
    extended_factors: Optional[List[str]] = None,
    factor_levels: Optional[Dict[str, List[Any]]] = None,
    equivalence_margin: Optional[float] = 0.01,
) -> Dict[str, object]:
    """
    Run extended ablation and analyze results.

    Performs factorial analysis including extended factors.
    """
    from pathlib import Path
    import subprocess
    import sys

    seeds = list(args.seeds)

    plan = build_extended_ablation_plan(
        seeds=seeds,
        base_config=base_config,
        ablation_modes=base_config.get("ablation_modes"),
        extended_factors=extended_factors,
        factor_levels=factor_levels,
        balance_coef=base_config.get("balance_coef", 0.01),
        mechanism_controls=base_config.get("mechanism_controls", False),
    )

    # Execute each run
    run_paths = []
    for run_spec in plan:
        run_label = build_run_label(run_spec)
        run_dir = out_root / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_experiment_command(run_spec, base_config, str(run_dir))

        print(f"[extended-ablation] Running: {run_label}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[extended-ablation] FAILED: {run_label}")
            print(result.stderr[:500])
        else:
            print(f"[extended-ablation] OK: {run_label}")

        run_paths.append(str(run_dir))

    # Analyze results
    rows = compare_runs(run_paths)
    analysis = analyze_extended_ablation(
        rows,
        extended_factors=extended_factors,
        equivalence_margin=equivalence_margin,
    )

    return analysis


def build_run_label(run_spec: Dict[str, object]) -> str:
    """Build a unique label for a run specification."""
    parts = []
    for key in ["mode", "balance_coef", "residual_scaling", "n_experts", "top_k", "residual_scale", "init_scheme", "seed"]:
        if key in run_spec:
            val = run_spec[key]
            if key == "balance_coef":
                parts.append(f"bal{format(val, 'g')}")
            else:
                parts.append(f"{key}{val}")
    return "_".join(parts)


def build_experiment_command(
    run_spec: Dict[str, object],
    base_config: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    """Build experiment command from run specification."""
    import sys

    cmd = [
        sys.executable, "experiment.py",
        "--data", str(base_config["data"]),
        "--out", out_dir,
        "--steps", str(base_config.get("steps", 1000)),
    ]

    # Add all parameters from run_spec
    for key, value in run_spec.items():
        if key != "seed":
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # Add seed
    cmd.extend(["--seed", str(run_spec.get("seed", 42))])

    # Add any base config not overridden
    for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                "seq_len", "d_model", "n_heads", "n_layers", "n_kv_heads",
                "expert_hidden", "k1", "k2"]:
        if key not in run_spec and key in base_config:
            cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

    return cmd


def analyze_extended_ablation(
    rows: List[Dict[str, object]],
    extended_factors: Optional[List[str]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    Analyze extended ablation results.

    Performs:
    1. Main factorial effects (mode x balance_coef) within each extended factor combination
    2. Extended factor main effects and interactions
    3. Component verdicts per extended factor combination
    """
    ext_factors = extended_factors or []
    all_factors = ["mode", "balance_coef"] + ext_factors

    report: Dict[str, object] = {
        "extended_factors": ext_factors,
        "extended_factor_combinations": [],
        "main_ablation": {},
        "extended_effects": {},
        "component_verdicts": {},
        "advisories": [],
    }

    # Group by extended factor combination
    ext_combinations: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        ext_key = tuple(_stable_factor_key(row.get(f)) for f in ext_factors)
        ext_combinations[ext_key].append(row)

    report["extended_factor_combinations"] = [
        dict(zip(ext_factors, key)) for key in ext_combinations.keys()
    ]

    # Analyze each extended factor combination separately
    for ext_key, ext_rows in ext_combinations.items():
        ext_label = "_".join(f"{k}={v}" for k, v in zip(ext_factors, ext_key))

        # Run standard ablation analysis on this subset
        factors = ["mode", "balance_coef"]
        coverage = ablation_coverage(ext_rows, factors, base_config.get("ablation_modes"), allow_partial=False)

        group_keys = [k for k in factors if k in set(factors)]
        aggregates = aggregate_runs(ext_rows, group_keys)

        factorial = factorial_effects(aggregates, group_keys, base_config.get("compare_metric", "val_loss"), equivalence_margin)

        mechanism = mechanism_control_report(
            aggregates,
            group_keys,
            base_config.get("compare_metric", "val_loss"),
            equivalence_margin=equivalence_margin,
        )

        verdicts = None
        if factorial.get("applicable"):
            from .part_029_ablation import ablation_component_verdicts
            verdicts = ablation_component_verdicts(factorial, coverage, mechanism=mechanism)

        report["main_ablation"][ext_label] = {
            "coverage": coverage,
            "factorial": factorial,
            "mechanism": mechanism,
            "verdicts": verdicts,
            "n_runs": len(ext_rows),
        }

    # Analyze extended factor effects (marginalized over mode x balance)
    if ext_factors:
        report["extended_effects"] = analyze_extended_factor_effects(
            rows,
            ext_factors,
            base_config.get("compare_metric", "val_loss"),
            equivalence_margin,
        )

    # Cross-extended-factor component stability
    report["component_verdicts"] = analyze_cross_extended_verdicts(
        report["main_ablation"],
        ext_factors,
    )

    return report


def analyze_extended_factor_effects(
    rows: List[Dict[str, object]],
    extended_factors: List[str],
    metric: str,
    equivalence_margin: Optional[float],
) -> Dict[str, object]:
    """
    Analyze main effects and interactions of extended factors.

    Uses a factorial ANOVA-like approach on the aggregated cell means.
    """
    from .part_027_aggregation import aggregate_runs

    all_factors = ["mode", "balance_coef"] + extended_factors
    aggregates = aggregate_runs(rows, all_factors)

    # For each extended factor, compute marginal effect
    effects: Dict[str, object] = {}

    for ext_factor in extended_factors:
        # Get other factors
        other_factors = [f for f in all_factors if f != ext_factor]

        # Compute marginal means for each level of ext_factor
        level_means = defaultdict(list)
        for agg in aggregates:
            level = _stable_factor_key(agg["keys"].get(ext_factor))
            mean_val = _finite_float(agg.get(f"{metric}_mean"))
            if mean_val is not None:
                level_means[level].append(mean_val)

        marginal_means = {
            level: sum(vals) / len(vals)
            for level, vals in level_means.items()
        }

        if len(marginal_means) < 2:
            effects[ext_factor] = {"applicable": False, "reason": "Insufficient levels"}
            continue

        # Main effect: range of marginal means
        mean_vals = list(marginal_means.values())
        main_effect_range = max(mean_vals) - min(mean_vals)

        # Interaction with mode (if mode varies)
        mode_interactions = {}
        for mode in _DELAYED_MERGE_MODES:
            mode_means = defaultdict(list)
            for agg in aggregates:
                if agg["keys"].get("mode") == mode:
                    level = _stable_factor_key(agg["keys"].get(ext_factor))
                    mean_val = _finite_float(agg.get(f"{metric}_mean"))
                    if mean_val is not None:
                        mode_means[level].append(mean_val)

            mode_marginal = {l: sum(v)/len(v) for l, v in mode_means.items()}
            if len(mode_marginal) >= 2:
                mode_range = max(mode_marginal.values()) - min(mode_marginal.values())
                mode_interactions[mode] = {
                    "marginal_means": mode_marginal,
                    "range": mode_range,
                }

        effects[ext_factor] = {
            "applicable": True,
            "marginal_means": marginal_means,
            "main_effect_range": main_effect_range,
            "mode_interactions": mode_interactions,
            "levels_tested": list(marginal_means.keys()),
        }

    return effects


def analyze_cross_extended_verdicts(
    main_ablation: Dict[str, Dict[str, object]],
    extended_factors: List[str],
) -> Dict[str, object]:
    """
    Analyze component verdict stability across extended factor combinations.

    For each component (delayed_merge_A, delayed_merge_B, balance_loss):
    - Check if verdict is consistent across extended factor levels
    - Identify conditions where component fails/succeeds
    """
    components = ["delayed_merge_A", "delayed_merge_B", "balance_loss"]
    # Filter to components that appear
    present_components = set()
    for data in main_ablation.values():
        verdicts = data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            present_components.add(comp.get("component"))

    verdicts_by_component: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for ext_label, data in main_ablation.items():
        verdicts = data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            comp_name = comp.get("component")
            verdicts_by_component[comp_name].append({
                "extended_condition": ext_label,
                "status": comp.get("status"),
                "mean_effect": comp.get("primary", {}).get("mean_effect"),
            })

    cross_verdicts: Dict[str, object] = {}
    for comp in present_components:
        vlist = verdicts_by_component.get(comp, [])
        if not vlist:
            continue

        statuses = [v["status"] for v in vlist]
        effects = [v["mean_effect"] for v in vlist if v["mean_effect"] is not None]

        all_supported = all(s == "supported" for s in statuses)
        any_supported = any(s == "supported" for s in statuses)
        consistent_direction = len(set(1 if e > 0 else -1 if e < 0 else 0 for e in effects)) <= 1 if effects else True

        cross_verdicts[comp] = {
            "n_conditions": len(vlist),
            "statuses": {v["extended_condition"]: v["status"] for v in vlist},
            "effects": {v["extended_condition"]: v["mean_effect"] for v in vlist if v["mean_effect"] is not None},
            "all_supported": all_supported,
            "any_supported": any_supported,
            "consistent_direction": consistent_direction,
            "verdict": (
                "stable" if all_supported and consistent_direction else
                "conditionally_supported" if any_supported and consistent_direction else
                "unstable"
            ),
        }

    return {
        "components": cross_verdicts,
        "overall": "stable" if all(v["verdict"] == "stable" for v in cross_verdicts.values()) else "conditionally_stable",
    }


def print_extended_ablation_report(analysis: Dict[str, object]) -> None:
    """Print formatted extended ablation report."""
    print("\n" + "=" * 80)
    print("EXTENDED ABLATION REPORT")
    print("=" * 80)

    print(f"\nExtended factors: {', '.join(analysis.get('extended_factors', []))}")
    print(f"Extended factor combinations: {len(analysis.get('extended_factor_combinations', []))}")

    # Main ablation per combination
    print("\n--- Main Ablation per Extended Factor Combination ---")
    for ext_label, data in analysis.get("main_ablation", {}).items():
        print(f"\n  {ext_label}:")
        verdicts = data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            print(f"    {comp.get('label')}: {comp.get('status')}")

    # Extended factor effects
    print("\n--- Extended Factor Effects ---")
    for factor, effect in analysis.get("extended_effects", {}).items():
        if effect.get("applicable"):
            print(f"\n  {factor}:")
            print(f"    Marginal means: {effect.get('marginal_means')}")
            print(f"    Main effect range: {effect.get('main_effect_range'):.6f}")
            for mode, inter in effect.get("mode_interactions", {}).items():
                print(f"    {mode} interaction range: {inter.get('range'):.6f}")

    # Cross-verdicts
    print("\n--- Component Verdict Stability ---")
    cross = analysis.get("component_verdicts", {})
    for comp, verdict in cross.get("components", {}).items():
        print(f"\n  {comp}: {verdict.get('verdict')}")
        print(f"    Statuses: {verdict.get('statuses')}")
        print(f"    Consistent direction: {verdict.get('consistent_direction')}")

    # Advisories
    if analysis.get("advisories"):
        print("\n--- Advisories ---")
        for adv in analysis["advisories"]:
            print(f"  [adv] {adv}")