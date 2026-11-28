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


def _is_deterministic_run(row) -> bool:
    """检查一个运行行是否为确定性运行（即具有有效数据且未因失败而排除）。"""

    return row.get("numerical_status") != "failed" and row.get("run") is not None