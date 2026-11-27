from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .part_029_ablation import (
    factorial_effects,
    mechanism_control_report,
    _DELAYED_MERGE_MECHANISM_CONTROLS,
)
from .part_031_ablation_coverage import ablation_coverage
from .part_028_contrasts import factor_contrasts
from .part_023_factorial_helpers import (
    _stable_factor_key,
    _finite_float,
    _row_factor_value,
    _mean,
)
from .part_028_contrasts import _apply_multiple_comparison_correction
from .part_021_comparison import compare_runs
from .part_030_factorial import (
    _DELAYED_MERGE_MODES,
    _ABLATION_STRUCTURAL_KEYS,
)
from .part_027_aggregation import metric_higher_is_better


# Default scales for multi-scale ablation (corresponding to paper scales)
DEFAULT_ABLATION_SCALES: Tuple[Dict[str, object], ...] = (
    {"name": "100M", "d_model": 672, "n_layers": 10, "n_experts": 64, "top_k": 8},
    {"name": "200M", "d_model": 1280, "n_layers": 12, "n_experts": 64, "top_k": 8},
    {"name": "600M", "d_model": 1792, "n_layers": 20, "n_experts": 64, "top_k": 8},
    {"name": "1.6B", "d_model": 2304, "n_layers": 30, "n_experts": 64, "top_k": 8},
)

# Scale factor keys that define a scale configuration
SCALE_FACTOR_KEYS: Tuple[str, ...] = (
    "d_model",
    "n_layers",
    "n_experts",
    "top_k",
)


def build_multi_scale_ablation_plan(
    seeds: List[int],
    scales: List[Dict[str, object]],
    ablation_modes: Optional[List[str]] = None,
    balance_coef: float = 0.01,
    mechanism_controls: bool = False,
) -> List[Dict[str, object]]:
    """
    Build a multi-scale ablation plan.

    For each scale, runs the full (1+N)×2 factorial ablation:
    - modes: local_base + delayed merge schemes (A, B by default)
    - balance_coef: 0 (off) and current value (on)
    - seeds: all provided seeds

    Each run gets a unique label like: scale_200M_mode_A_bal0.01_seed42
    """
    plan: List[Dict[str, object]] = []

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

    for scale in scales:
        scale_name = scale.get("name", "unknown")
        for mode in arm_modes:
            for coefficient in (0.0, balance_coef):
                label_coefficient = format(coefficient, "g")
                for seed in seeds:
                    plan.append({
                        "label": f"scale_{scale_name}_mode_{mode}_bal{label_coefficient}_seed{seed}",
                        "mode": mode,
                        "balance_coef": coefficient,
                        "seed": seed,
                        "scale": scale_name,
                        "scale_config": scale,
                    })

    return plan


def run_multi_scale_ablation(
    args: Any,
    out_root: Any,
    scales: List[Dict[str, object]],
    ablation_modes: Optional[List[str]] = None,
    balance_coef: float = 0.01,
    mechanism_controls: bool = False,
    equivalence_margin: Optional[float] = 0.01,
) -> Dict[str, object]:
    """
    Execute multi-scale ablation and analyze results.

    Runs the full factorial ablation at each scale, then performs:
    1. Per-scale factorial analysis
    2. Cross-scale component stability analysis (scale × component interaction)
    3. Scaling law analysis for each component
    """
    from pathlib import Path
    import subprocess
    import sys

    seeds = list(args.seeds)
    plan = build_multi_scale_ablation_plan(
        seeds=seeds,
        scales=scales,
        ablation_modes=ablation_modes,
        balance_coef=balance_coef,
        mechanism_controls=mechanism_controls,
    )

    # Execute each run
    run_paths = []
    base_argv = [a for a in sys.argv[1:] if not a.startswith("--")]

    for run_spec in plan:
        run_label = run_spec["label"]
        run_dir = out_root / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build command for this run
        cmd = [
            sys.executable, "experiment.py",
            "--data", str(args.data),
            "--mode", run_spec["mode"],
            "--balance-coef", str(run_spec["balance_coef"]),
            "--seed", str(run_spec["seed"]),
            "--out", str(run_dir),
            "--steps", str(args.steps),
        ]

        # Add scale configuration
        scale_config = run_spec["scale_config"]
        for key, value in scale_config.items():
            if key != "name":
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Add other args that should be consistent
        for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                    "seq_len", "n_heads", "n_kv_heads", "expert_hidden",
                    "k1", "k2", "residual_scale", "residual_scaling",
                    "document_masks", "group_separator", "eval_interval",
                    "eval_batches", "early_stop_patience", "early_stop_min_delta",
                    "amp_dtype", "device", "allow_tf32", "grad_checkpoint", "compile"]:
            val = getattr(args, key, None)
            if val is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(val)])

        print(f"[multi-scale] Running: {run_label}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[multi-scale] FAILED: {run_label}")
            print(result.stderr)
        else:
            print(f"[multi-scale] OK: {run_label}")

        run_paths.append(str(run_dir))

    # Analyze results
    rows = compare_runs(run_paths)
    analysis = analyze_multi_scale_ablation(
        rows,
        scales=scales,
        ablation_modes=ablation_modes,
        equivalence_margin=equivalence_margin,
    )

    return analysis


def analyze_multi_scale_ablation(
    rows: List[Dict[str, object]],
    scales: List[Dict[str, object]],
    ablation_modes: Optional[List[str]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    Analyze multi-scale ablation results.

    Performs:
    1. Per-scale factorial analysis
    2. Cross-scale component stability (does component effect depend on scale?)
    3. Scaling law fitting for each component
    """
    from .part_021_comparison import DEFAULT_COMPARISON_FACTORS
    from .part_029_ablation import ablation_component_verdicts

    factors = list(DEFAULT_COMPARISON_FACTORS) + ["scale"]
    scale_names = [s.get("name", "unknown") for s in scales]

    report: Dict[str, object] = {
        "scales": scale_names,
        "ablation_modes": ablation_modes or list(_DELAYED_MERGE_MODES),
        "per_scale": {},
        "cross_scale": {},
        "scaling_laws": {},
        "component_stability": {},
        "advisories": [],
    }

    # Per-scale analysis
    for scale_name in scale_names:
        scale_rows = [r for r in rows if r.get("scale") == scale_name]
        if not scale_rows:
            report["advisories"].append(f"No runs found for scale {scale_name}")
            continue

        # Run standard ablation analysis for this scale
        coverage = ablation_coverage(
            scale_rows,
            factors=[f for f in factors if f != "scale"],
            ablation_modes=ablation_modes,
            allow_partial=False,
        )

        # Aggregate runs by experimental cell
        from .part_027_aggregation import aggregate_runs
        group_keys = [k for k in factors if k not in ("seed", "scale")]
        aggregates = aggregate_runs(scale_rows, group_keys)

        # Factorial effects
        factorial = factorial_effects(aggregates, group_keys, args.compare_metric, equivalence_margin)

        # Mechanism controls
        mechanism = mechanism_control_report(
            aggregates,
            group_keys,
            args.compare_metric,
            equivalence_margin=equivalence_margin,
        )

        # Component verdicts
        verdicts = ablation_component_verdicts(factorial, coverage, mechanism=mechanism)

        report["per_scale"][scale_name] = {
            "coverage": coverage,
            "factorial": factorial,
            "mechanism": mechanism,
            "verdicts": verdicts,
            "n_runs": len(scale_rows),
        }

    # Cross-scale analysis: check if component effects are stable across scales
    report["cross_scale"] = analyze_cross_scale_stability(
        report["per_scale"],
        scales,
        ablation_modes,
        equivalence_margin,
    )

    # Scaling law analysis
    report["scaling_laws"] = analyze_scaling_laws(
        report["per_scale"],
        scales,
    )

    # Overall component stability verdict
    report["component_stability"] = compute_component_stability(
        report["per_scale"],
        scales,
        ablation_modes,
    )

    return report


def analyze_cross_scale_stability(
    per_scale: Dict[str, Dict[str, object]],
    scales: List[Dict[str, object]],
    ablation_modes: Optional[List[str]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, object]:
    """
    Analyze whether component effects are consistent across scales.

    For each component (delayed_merge_A, delayed_merge_B, balance_loss):
    - Check if effect direction is consistent across scales
    - Check if effect magnitude scales predictably
    - Test for scale × component interaction
    """
    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)
    components = [f"delayed_merge_{mode}" for mode in delayed_modes] + ["balance_loss"]

    cross_scale: Dict[str, object] = {
        "components": {},
        "scale_interactions": {},
        "advisories": [],
    }

    for component in components:
        # Collect effect estimates across scales
        scale_effects = {}
        for scale_name, scale_data in per_scale.items():
            verdicts = scale_data.get("verdicts", {})
            if not verdicts.get("applicable"):
                continue
            for comp in verdicts.get("components", []):
                if comp.get("component") == component:
                    primary = comp.get("primary", {})
                    scale_effects[scale_name] = {
                        "status": comp.get("status"),
                        "mean_effect": primary.get("mean_effect"),
                        "significant": primary.get("significant", False),
                        "direction": primary.get("direction") if "direction" in primary else None,
                    }
                    break

        if not scale_effects:
            continue

        # Check direction consistency
        directions = [e.get("direction") for e in scale_effects.values() if e.get("direction")]
        statuses = [e.get("status") for e in scale_effects.values()]

        direction_consistent = len(set(directions)) <= 1 if directions else False
        all_supported = all(s == "supported" for s in statuses) if statuses else False

        cross_scale["components"][component] = {
            "scale_effects": scale_effects,
            "direction_consistent": direction_consistent,
            "all_supported": all_supported,
            "n_scales_tested": len(scale_effects),
        }

        if not direction_consistent and len(directions) > 1:
            cross_scale["advisories"].append(
                f"{component}: effect direction differs across scales "
                f"({', '.join(f'{k}={v}' for k, v in scale_effects.items())})"
            )

        # Fit scaling law if we have numeric effects
        numeric_effects = {
            k: v["mean_effect"] for k, v in scale_effects.items()
            if v.get("mean_effect") is not None
        }
        if len(numeric_effects) >= 3:
            cross_scale["scale_interactions"][component] = fit_scaling_law(
                numeric_effects, scales
            )

    return cross_scale


def fit_scaling_law(
    scale_effects: Dict[str, float],
    scales: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    Fit a power-law scaling relationship: effect ~ scale^alpha.

    Uses model size (d_model * n_layers) as the scale proxy.
    """
    import math

    # Map scale names to compute proxy (parameter count approximation)
    scale_to_compute = {}
    for s in scales:
        name = s.get("name")
        d_model = s.get("d_model", 0)
        n_layers = s.get("n_layers", 0)
        n_experts = s.get("n_experts", 0)
        # Rough parameter count proxy
        compute = d_model * d_model * n_layers  # Attention params
        compute += d_model * n_experts * 3 * 256  # MoE params (rough)
        scale_to_compute[name] = compute

    # Prepare data points
    points = []
    for name, effect in scale_effects.items():
        if name in scale_to_compute and effect > 0:
            points.append((math.log(scale_to_compute[name]), math.log(effect)))

    if len(points) < 3:
        return {"available": False, "reason": "Insufficient positive effects for scaling law"}

    # Linear regression in log-log space
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    n = len(points)

    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    if x_var == 0:
        return {"available": False, "reason": "No variance in scale"}

    xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    alpha = xy_cov / x_var  # Scaling exponent
    intercept = y_mean - alpha * x_mean

    # R-squared
    y_pred = [intercept + alpha * x for x in x_vals]
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "available": True,
        "scaling_exponent": alpha,
        "intercept": intercept,
        "r_squared": r2,
        "n_points": n,
        "scale_proxy": {k: math.log(v) for k, v in scale_to_compute.items() if k in scale_effects},
        "effects_log": {k: math.log(v) for k, v in scale_effects.items() if v > 0},
    }


def analyze_scaling_laws(
    per_scale: Dict[str, Dict[str, object]],
    scales: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    Analyze scaling laws for overall performance metrics.

    Fits loss ~ compute^alpha for each mode across scales.
    """
    import math

    scale_to_compute = {}
    for s in scales:
        name = s.get("name")
        d_model = s.get("d_model", 0)
        n_layers = s.get("n_layers", 0)
        n_experts = s.get("n_experts", 0)
        compute = d_model * d_model * n_layers
        compute += d_model * n_experts * 3 * 256
        scale_to_compute[name] = compute

    scaling_laws: Dict[str, object] = {}

    # Get all modes present
    all_modes = set()
    for scale_data in per_scale.values():
        coverage = scale_data.get("coverage", {})
        for mode in coverage.get("expected_delayed_modes", []):
            all_modes.add(mode)
        all_modes.add("local_base")

    for mode in all_modes:
        # Collect loss at each scale
        points = []
        for scale_name, scale_data in per_scale.items():
            aggregates = scale_data.get("factorial", {}).get("effects", [])
            # Find the cell for this mode at balance=off
            for agg in scale_data.get("coverage", {}).get("present_cells", []):
                if agg.startswith(f"{mode}/b=off"):
                    # Need to get actual loss from rows
                    break

        if len(points) >= 3:
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            n = len(points)
            x_mean = sum(x_vals) / n
            y_mean = sum(y_vals) / n
            x_var = sum((x - x_mean) ** 2 for x in x_vals)
            if x_var > 0:
                xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
                alpha = xy_cov / x_var
                scaling_laws[mode] = {
                    "scaling_exponent": alpha,
                    "r_squared": 1 - (sum((y - (y_mean + alpha * (x - x_mean))) ** 2 for x, y in zip(x_vals, y_vals)) / sum((y - y_mean) ** 2 for y in y_vals)) if sum((y - y_mean) ** 2 for y in y_vals) > 0 else 0,
                    "n_scales": n,
                }

    return scaling_laws


def compute_component_stability(
    per_scale: Dict[str, Dict[str, object]],
    scales: List[Dict[str, object]],
    ablation_modes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute overall component stability verdict across scales.

    A component is "stable" if:
    - It is supported at all scales (or at largest scales)
    - Effect direction is consistent
    - No scale × component interaction detected
    """
    delayed_modes = ablation_modes or list(_DELAYED_MERGE_MODES)
    components = [f"delayed_merge_{mode}" for mode in delayed_modes] + ["balance_loss"]

    stability: Dict[str, object] = {
        "components": {},
        "overall": "unknown",
        "blockers": [],
    }

    for component in components:
        statuses = []
        directions = []
        for scale_name, scale_data in per_scale.items():
            verdicts = scale_data.get("verdicts", {})
            for comp in verdicts.get("components", []):
                if comp.get("component") == component:
                    statuses.append(comp.get("status"))
                    primary = comp.get("primary", {})
                    # Determine direction from mean_effect
                    effect = primary.get("mean_effect")
                    if effect is not None:
                        higher_better = metric_higher_is_better("val_loss")  # Default
                        directions.append("improvement" if (effect > 0) == higher_better else "regression")
                    break

        if not statuses:
            stability["components"][component] = {"status": "missing", "reason": "No data across scales"}
            continue

        # Check if all supported
        all_supported = all(s == "supported" for s in statuses)
        direction_consistent = len(set(directions)) <= 1 if directions else False

        if all_supported and direction_consistent:
            comp_status = "stable"
        elif all_supported and not direction_consistent:
            comp_status = "direction_inconsistent"
            stability["blockers"].append(f"{component}: direction flips across scales")
        elif any(s == "supported" for s in statuses):
            comp_status = "partially_stable"
        else:
            comp_status = "unstable"

        stability["components"][component] = {
            "status": comp_status,
            "scale_statuses": dict(zip([s.get("name") for s in scales], statuses)),
            "directions": directions,
        }

    if all(c.get("status") == "stable" for c in stability["components"].values()):
        stability["overall"] = "all_stable"
    elif any(c.get("status") == "direction_inconsistent" for c in stability["components"].values()):
        stability["overall"] = "blocked"
    elif all(c.get("status") in ("stable", "partially_stable") for c in stability["components"].values()):
        stability["overall"] = "partially_stable"
    else:
        stability["overall"] = "unstable"

    return stability


def print_multi_scale_ablation_report(analysis: Dict[str, object]) -> None:
    """Print a formatted multi-scale ablation report."""
    print("\n" + "=" * 80)
    print("MULTI-SCALE ABLATION REPORT")
    print("=" * 80)

    print(f"\nScales tested: {', '.join(analysis.get('scales', []))}")
    print(f"Ablation modes: {', '.join(analysis.get('ablation_modes', []))}")

    # Per-scale summary
    print("\n--- Per-Scale Component Verdicts ---")
    for scale_name, scale_data in analysis.get("per_scale", {}).items():
        print(f"\n  Scale: {scale_name}")
        verdicts = scale_data.get("verdicts", {})
        for comp in verdicts.get("components", []):
            status = comp.get("status")
            label = comp.get("label")
            print(f"    {label}: {status}")

    # Cross-scale stability
    print("\n--- Cross-Scale Stability ---")
    cross = analysis.get("cross_scale", {})
    for comp_name, comp_data in cross.get("components", {}).items():
        consistent = comp_data.get("direction_consistent")
        all_supp = comp_data.get("all_supported")
        print(f"  {comp_name}: direction_consistent={consistent}, all_supported={all_supp}")

    # Component stability
    print("\n--- Component Stability Verdict ---")
    stability = analysis.get("component_stability", {})
    print(f"  Overall: {stability.get('overall')}")
    for comp_name, comp_data in stability.get("components", {}).items():
        print(f"  {comp_name}: {comp_data.get('status')}")

    # Scaling laws
    print("\n--- Scaling Laws ---")
    for comp_name, law in analysis.get("scaling_laws", {}).items():
        if law.get("available"):
            print(f"  {comp_name}: exponent={law.get('scaling_exponent'):.3f}, R²={law.get('r_squared'):.3f}")

    # Advisories
    all_advisories = []
    for scale_data in analysis.get("per_scale", {}).values():
        all_advisories.extend(scale_data.get("verdicts", {}).get("advisories", []))
    all_advisories.extend(analysis.get("cross_scale", {}).get("advisories", []))
    all_advisories.extend(stability.get("blockers", []))

    if all_advisories:
        print("\n--- Advisories ---")
        for adv in all_advisories:
            print(f"  [adv] {adv}")


# Integration with existing sweep infrastructure
def add_multi_scale_ablation_args(parser: Any) -> None:
    """Add multi-scale ablation arguments to the parser."""
    parser.add_argument(
        "--suite-ablation-scales",
        action="store_true",
        help="Run multi-scale ablation: full factorial at each scale (100M, 200M, 600M, 1.6B)."
    )
    parser.add_argument(
        "--ablation-scales",
        nargs="+",
        type=str,
        default=None,
        metavar="SCALE",
        help="Scales to include in multi-scale ablation (default: all paper scales)."
    )
    parser.add_argument(
        "--ablation-scale-config",
        type=str,
        default=None,
        metavar="JSON",
        help="JSON file with custom scale configurations."
    )