from __future__ import annotations

from .helpers import (
    _finite_float,
    _finite_values,
    _mean,
    _median,
    _sample_std,
    _stable_factor_key,
    _factor_sort_rank,
    _row_factor_value,
    _format_factor_value,
    _randomization_seed,
    _MIN_SEEDS_FOR_STATS,
    _RANDOMIZATION_EXACT_MAX_N,
    _RANDOMIZATION_MONTE_CARLO_REPLICATES,
    _SIGNIFICANCE_LEVEL,
    _TARGET_POWER,
    _FACTOR_VALUE_ORDER,
)


def ablation_coverage(
    rows: List[Dict[str, object]],
    factors: List[str],
    modes: List[str],
    *,
    allow_partial: bool = False,
) -> Dict[str, object]:
    """计算覆盖度：有多少运行为每个因子组合提供了有效数据。

    返回每个因子组合的运行计数与可用性标记。
    """
    from .part_019_diagnostics import _is_deterministic_run

    coverage: Dict[str, object] = {}
    factor_values: Dict[str, set] = {f: set() for f in factors}

    for row in rows:
        key_parts = []
        for f in factors:
            val = row.get(f)
            if val is not None:
                factor_values[f].add(val)
                key_parts.append(f"{f}={val}")
            else:
                key_parts.append(f"{f}=<missing>")
        key = " × ".join(key_parts)

        if allow_partial or _is_deterministic_run(row):
            coverage[key] = coverage.get(key, 0) + 1

    total = len(rows)
    for key in coverage:
        coverage[key] = {"n": coverage[key], "total": total, "pct": coverage[key] / total if total else 0.0}

    return coverage