from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math
import itertools

from .part_021_comparison import compare_runs
from .part_027_aggregation import aggregate_runs
from .part_000_config import mode_cost_profile, Config, PAPER_SCALES, paper_preset, _DELAYED_MERGE_MODES
from .part_023_factorial_helpers import _finite_float, _stable_factor_key
from .part_025_statistics import metric_higher_is_better


# Scaling law experiment configurations
SCALING_LAW_DESIGNS: Dict[str, Dict[str, Any]] = {
    "compute_matched": {
        "description": "Match compute (FLOPs) across modes at each scale",
        "vary": ["k1", "k2", "top_k"],
        "match_metric": "flops_per_token",
    },
    "parameter_matched": {
        "description": "Match parameter count across modes",
        "vary": ["n_experts", "expert_hidden"],
        "match_metric": "n_params",
    },
    "width_matched": {
        "description": "Match expert width (k * expert_hidden)",
        "vary": ["top_k", "expert_hidden", "k1", "k2"],
        "match_metric": "expert_assignments_per_token",
    },
}


def run_compute_matched_scaling_experiment(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    scales: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    Run compute-matched scaling experiments.

    For each scale and mode, find configurations that match the compute
    of a reference (e.g., smelt at that scale), then compare performance
    at matched compute.
    """
    from pathlib import Path
    import subprocess
    import sys

    scales = scales or list(PAPER_SCALES)
    modes = modes or ["smelt", "A", "B", "local_base", "local_base_k2", "dense"]

    # Get reference configurations for each scale (smelt)
    reference_configs = {}
    for scale in scales:
        try:
            preset = paper_preset(scale, 95, "smelt")
            reference_configs[scale] = preset["config"]
        except Exception:
            pass

    if not reference_configs:
        return {"applicable": False, "reason": "Could not load reference configurations"}

    # For each mode at each scale, find compute-matched config
    matched_configs = {}
    for scale in scales:
        if scale not in reference_configs:
            continue
        ref_config = reference_configs[scale]
        ref_profile = mode_cost_profile(Config(**ref_config))
        ref_flops = ref_profile["analytic_flops_per_token"]

        matched_configs[scale] = {"reference": ref_config, "reference_flops": ref_flops, "modes": {}}

        for mode in modes:
            if mode == "smelt":
                matched_configs[scale]["modes"][mode] = ref_config
                continue

            # Search for matched config
            base_mode_config = dict(ref_config)
            base_mode_config["mode"] = mode

            # For delayed merge modes, search k1, k2 space
            best_config = find_compute_matched_config(
                base_mode_config, ref_flops, mode
            )
            matched_configs[scale]["modes"][mode] = best_config

    # Run experiments
    report = run_scaling_experiment(
        args=args,
        base_config=base_config,
        metric=metric,
        experiment_configs=matched_configs,
        n_seeds=n_seeds,
        experiment_type="compute_matched",
    )

    return report


def find_compute_matched_config(
    base_config: Dict[str, Any],
    target_flops: int,
    mode: str,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Find a configuration for the given mode that matches target FLOPs.

    Searches over k1, k2, top_k, n_experts, expert_hidden space.
    """
    best_config = dict(base_config)
    best_diff = float('inf')

    # Search space depends on mode
    if mode in ("A", "local_base", "local_base_k2"):
        k1_range = range(1, base_config.get("n_experts", 8) + 1)
        k2_range = range(1, base_config.get("n_experts", 8) + 1)
    elif mode in ("B", "B_early"):
        k1_range = range(1, base_config.get("n_experts", 8) + 1)
        k2_range = range(1, base_config.get("n_experts", 8) + 1)
    else:
        k1_range = [base_config.get("k1", 2)]
        k2_range = [base_config.get("k2", 2)]

    # Limit search for efficiency
    k1_range = list(k1_range)[:8]
    k2_range = list(k2_range)[:8]

    for k1 in k1_range:
        for k2 in k2_range:
            test_config = dict(base_config)
            test_config["k1"] = k1
            test_config["k2"] = k2

            try:
                cfg = Config(**test_config)
                profile = mode_cost_profile(cfg)
                flops = profile["analytic_flops_per_token"]
                diff = abs(flops - target_flops) / target_flops

                if diff < best_diff:
                    best_diff = diff
                    best_config = test_config
                    if diff < tolerance:
                        return best_config
            except Exception:
                continue

    return best_config


def run_scaling_experiment(
    args: Any,
    base_config: Dict[str, Any],
    metric: str,
    experiment_configs: Dict[str, Any],
    n_seeds: int,
    experiment_type: str,
) -> Dict[str, object]:
    """
    Run a scaling law experiment with given configurations.
    """
    from pathlib import Path
    import subprocess
    import sys

    all_runs = []

    for scale, scale_data in experiment_configs.items():
        for mode, config in scale_data.get("modes", {}).items():
            if mode == "reference":
                continue

            for seed in range(n_seeds):
                run_label = f"scaling_{experiment_type}_{scale}_{mode}_seed{seed}"
                run_dir = Path(base_config["out"]) / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, "experiment.py",
                    "--data", str(base_config["data"]),
                    "--out", str(run_dir),
                    "--steps", str(base_config.get("steps", 1000)),
                    "--seed", str(seed),
                ]

                for key, value in config.items():
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

                # Add other base config
                for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                            "seq_len", "d_model", "n_heads", "n_layers", "n_kv_heads",
                            "expert_hidden", "balance_coef", "residual_scale", "residual_scaling"]:
                    if key not in config and key in base_config:
                        cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

                print(f"[scaling] Running: {run_label}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    all_runs.append(str(run_dir))
                else:
                    print(f"[scaling] FAILED: {run_label}")

    # Analyze
    rows = compare_runs(all_runs)
    analysis = analyze_scaling_results(rows, experiment_configs, metric, experiment_type)

    return analysis


def analyze_scaling_results(
    rows: List[Dict[str, object]],
    experiment_configs: Dict[str, Any],
    metric: str,
    experiment_type: str,
) -> Dict[str, object]:
    """
    Analyze scaling law experiment results.
    """
    from .part_027_aggregation import aggregate_runs

    report: Dict[str, object] = {
        "experiment_type": experiment_type,
        "metric": metric,
        "scales": list(experiment_configs.keys()),
        "per_scale": {},
        "scaling_laws": {},
        "compute_matched_comparison": {},
        "advisories": [],
    }

    # Group by scale and mode
    for scale in experiment_configs:
        scale_rows = [r for r in rows if r.get("scale") == scale]
        if not scale_rows:
            continue

        # Get reference FLOPs
        ref_flops = experiment_configs[scale].get("reference_flops")

        # Aggregate by mode
        aggregates = aggregate_runs(scale_rows, ["mode"])

        # Compute matched metric analysis
        if ref_flops:
            for agg in aggregates:
                mode = agg["keys"].get("mode")
                if mode and mode in experiment_configs[scale]["modes"]:
                    config = experiment_configs[scale]["modes"][mode]
                    try:
                        cfg = Config(**config)
                        profile = mode_cost_profile(cfg)
                        flops = profile["analytic_flops_per_token"]
                        ratio = flops / ref_flops if ref_flops > 0 else 0
                        agg["compute_ratio"] = ratio
                        agg["matched"] = abs(ratio - 1.0) < 0.1
                    except Exception:
                        pass

        report["per_scale"][scale] = {
            "aggregates": aggregates,
            "reference_flops": ref_flops,
        }

    # Fit scaling laws
    report["scaling_laws"] = fit_scaling_laws(report["per_scale"], metric)

    # Compute-matched comparison
    report["compute_matched_comparison"] = compute_matched_comparison(
        report["per_scale"], metric
    )

    return report


def fit_scaling_laws(
    per_scale: Dict[str, Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Fit scaling laws: metric ~ compute^alpha for each mode.
    """
    scaling_laws: Dict[str, object] = {}

    # Get all modes
    all_modes = set()
    for scale_data in per_scale.values():
        for agg in scale_data.get("aggregates", []):
            mode = agg["keys"].get("mode")
            if mode:
                all_modes.add(mode)

    for mode in all_modes:
        points = []
        for scale, scale_data in per_scale.items():
            for agg in scale_data.get("aggregates", []):
                if agg["keys"].get("mode") == mode:
                    mean_val = _finite_float(agg.get(f"{metric}_mean"))
                    flops = _finite_float(agg.get("flops_per_forward"))
                    if mean_val is not None and flops is not None and flops > 0:
                        points.append((math.log(flops), math.log(mean_val) if mean_val > 0 else 0, scale))

        if len(points) >= 3:
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            # Linear regression in log-log space
            n = len(points)
            x_mean = sum(x_vals) / n
            y_mean = sum(y_vals) / n
            x_var = sum((x - x_mean) ** 2 for x in x_vals)

            if x_var > 0:
                xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
                alpha = xy_cov / x_var
                intercept = y_mean - alpha * x_mean

                # R-squared
                y_pred = [intercept + alpha * x for x in x_vals]
                ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
                ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred))
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                scaling_laws[mode] = {
                    "scaling_exponent": alpha,
                    "intercept": intercept,
                    "r_squared": r2,
                    "n_points": n,
                    "points": {p[2]: {"flops": math.exp(p[0]), "metric": math.exp(p[1])} for p in points},
                }

    return scaling_laws


def compute_matched_comparison(
    per_scale: Dict[str, Dict[str, object]],
    metric: str,
) -> Dict[str, object]:
    """
    Compare modes at matched compute within each scale.
    """
    comparison: Dict[str, object] = {
        "per_scale": {},
        "modes_compared": [],
    }

    for scale, scale_data in per_scale.items():
        aggregates = scale_data.get("aggregates", [])

        # Find matched modes (compute_ratio ≈ 1)
        matched = [a for a in aggregates if a.get("matched")]
        reference = next((a for a in aggregates if a["keys"].get("mode") == "smelt"), None)

        if reference:
            ref_mean = _finite_float(reference.get(f"{metric}_mean"))
            comparison["per_scale"][scale] = {
                "reference_mode": "smelt",
                "reference_metric": ref_mean,
                "matched_modes": [],
            }

            for agg in matched:
                mode = agg["keys"].get("mode")
                if mode and mode != "smelt":
                    mean_val = _finite_float(agg.get(f"{metric}_mean"))
                    if mean_val is not None and ref_mean is not None:
                        diff = mean_val - ref_mean
                        rel_diff = diff / abs(ref_mean) if ref_mean != 0 else 0
                        comparison["per_scale"][scale]["matched_modes"].append({
                            "mode": mode,
                            "metric": mean_val,
                            "difference": diff,
                            "relative_difference": rel_diff,
                            "compute_ratio": agg.get("compute_ratio"),
                        })
                        if mode not in comparison["modes_compared"]:
                            comparison["modes_compared"].append(mode)

    return comparison


def run_isoflops_profiling(
    args: Any,
    base_config: Dict[str, Any],
    metric: str = "val_loss",
    target_flops_list: Optional[List[int]] = None,
    modes: Optional[List[str]] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """
    IsoFLOPs profiling: train each mode at multiple FLOPs budgets
    and interpolate performance at common FLOPs.

    This implements the "IsoFLOPs" methodology from Kaplan et al.
    """
    target_flops = target_flops_list or [1e8, 3e8, 1e9, 3e9]
    modes = modes or ["smelt", "A", "B", "local_base", "dense"]

    from pathlib import Path
    import subprocess
    import sys

    all_runs = []

    for target in target_flops:
        for mode in modes:
            for seed in range(n_seeds):
                # Find config matching target FLOPs
                base_mode_config = dict(base_config)
                base_mode_config["mode"] = mode

                matched_config = find_compute_matched_config(
                    base_mode_config, target, mode, tolerance=0.2
                )

                run_label = f"isoflops_{mode}_flops{target:.0e}_seed{seed}"
                run_dir = Path(base_config["out"]) / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, "experiment.py",
                    "--data", str(base_config["data"]),
                    "--out", str(run_dir),
                    "--steps", str(base_config.get("steps", 1000)),
                    "--seed", str(seed),
                ]

                for key, value in matched_config.items():
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

                for key in ["batch_size", "lr", "warmup_steps", "weight_decay", "grad_clip",
                            "seq_len", "d_model", "n_heads", "n_layers", "n_kv_heads",
                            "expert_hidden", "balance_coef", "residual_scale", "residual_scaling"]:
                    if key not in matched_config and key in base_config:
                        cmd.extend([f"--{key.replace('_', '-')}", str(base_config[key])])

                print(f"[isoflops] Running: {run_label}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    all_runs.append(str(run_dir))

    # Analyze: interpolate performance at each target FLOPs
    rows = compare_runs(all_runs)
    return analyze_isoflops_results(rows, target_flops, metric)


def analyze_isoflops_results(
    rows: List[Dict[str, object]],
    target_flops_list: List[int],
    metric: str,
) -> Dict[str, object]:
    """
    Analyze IsoFLOPs results: interpolate performance curves.
    """
    from .part_027_aggregation import aggregate_runs

    report: Dict[str, object] = {
        "target_flops": target_flops_list,
        "metric": metric,
        "curves": {},
        "interpolated": {},
        "advisories": [],
    }

    aggregates = aggregate_runs(rows, ["mode", "target_flops"])

    # Group by mode
    for agg in aggregates:
        mode = agg["keys"].get("mode")
        target = agg["keys"].get("target_flops")
        if mode and target:
            if mode not in report["curves"]:
                report["curves"][mode] = {}
            mean_val = _finite_float(agg.get(f"{metric}_mean"))
            if mean_val is not None:
                report["curves"][mode][target] = mean_val

    # Interpolate at common FLOPs points
    all_flops = set()
    for curve in report["curves"].values():
        all_flops.update(curve.keys())

    common_flops = sorted(all_flops)

    for mode, curve in report["curves"].items():
        if len(curve) >= 2:
            # Sort by FLOPs
            sorted_points = sorted(curve.items())
            x_vals = [math.log(x) for x, _ in sorted_points]
            y_vals = [y for _, y in sorted_points]

            # Interpolate at common points
            interpolated = {}
            for target in common_flops:
                log_target = math.log(target)
                # Find bracketing points
                for i in range(len(x_vals) - 1):
                    if x_vals[i] <= log_target <= x_vals[i + 1]:
                        # Linear interpolation in log-log space
                        t = (log_target - x_vals[i]) / (x_vals[i + 1] - x_vals[i])
                        interp_y = y_vals[i] + t * (y_vals[i + 1] - y_vals[i])
                        interpolated[target] = interp_y
                        break
                else:
                    # Extrapolate (flag as extrapolated)
                    if log_target < x_vals[0]:
                        extrap_y = y_vals[0]  # Constant extrapolation
                    else:
                        extrap_y = y_vals[-1]
                    interpolated[target] = extrap_y

            report["interpolated"][mode] = interpolated

    return report


def print_scaling_law_report(report: Dict[str, object]) -> None:
    """Print formatted scaling law report."""
    print("\n" + "=" * 80)
    print("SCALING LAW REPORT")
    print("=" * 80)

    print(f"\nExperiment type: {report.get('experiment_type')}")
    print(f"Metric: {report.get('metric')}")

    # Scaling laws
    print("\n--- Scaling Laws (metric ~ compute^alpha) ---")
    for mode, law in report.get("scaling_laws", {}).items():
        if "scaling_exponent" in law:
            print(f"  {mode}: alpha={law['scaling_exponent']:.4f}, R²={law['r_squared']:.4f}")

    # Compute-matched comparison
    print("\n--- Compute-Matched Comparison (vs smelt) ---")
    for scale, data in report.get("compute_matched_comparison", {}).get("per_scale", {}).items():
        print(f"\n  Scale: {scale}")
        ref = data.get("reference_metric")
        if ref is not None:
            print(f"    smelt (reference): {ref:.6f}")
        for m in data.get("matched_modes", []):
            print(f"    {m['mode']}: {m['metric']:.6f} (diff={m['difference']:+.6f}, "
                  f"rel={m['relative_difference']:.2%}, compute_ratio={m['compute_ratio']:.2f})")

    # Advisories
    if report.get("advisories"):
        print("\n--- Advisories ---")
        for adv in report["advisories"]:
            print(f"  [adv] {adv}")