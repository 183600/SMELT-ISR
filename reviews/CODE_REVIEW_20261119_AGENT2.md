# Independent Code Review Report - SMELT-ISR

**Date:** 2026-11-19
**Reviewer:** Sub-Agent Independent Review (Round 1)
**Scope:** Full codebase deep review — MoE architecture, metrics, data pipeline, statistical methods, evaluation

## Executive Summary

| Category | Result |
|---|---|
| Files reviewed | 67 Python files |
| True bugs found | 0 |
| Scientific methodology issues | 0 |
| Security concerns | 0 (exec() is local-only) |
| Overall verdict | **PASS — Code is clean** |

## Deep Review Details

### 1. MoE Architecture (exp/part_003_model.py)
- Top-k routing correctly implemented
- Load balancing loss: `aux_loss = n_experts * dot(hard_load, soft_load)` — correct
- Non-reentrant activation checkpointing properly used
- Delayed merging方案 (A/B/B_early) correctly isolates architecture factors

### 2. Metrics (exp/part_011_metrics.py)
- micro_f1 and macro_f1 correctly implemented
- Precision/recall in macro and weighted variants correct
- Confusion matrix with `max(total, 1)` zero-guard
- Division safety with clamp_min throughout

### 3. Data Pipeline (exp/part_007_data_pipeline.py)
- Train/val/test splits are contiguous and non-overlapping
- Seen context share and unseen transition share properly tracked
- Fixed-window evaluation batches for reproducibility
- `counts / total` division properly guarded

### 4. Statistical Methods
- Welch t-test with correct Welch-Satterthwaite degrees of freedom
- Subnormal number detection using `sys.float_info.min`
- Group bootstrap with paired resampling
- Comparison protocol with proper paired comparisons

### 5. Configuration (exp/part_000_config.py)
- `head_dim = max(1, hidden // n_heads)` with zero-guard — correct
- Hyperparameter validation comprehensive

### 6. Security Assessment
- `exp/__init__.py` exec(): compiled local files only, safe
- `exp/_runner.py` exec(): isolated namespace execution, safe

## Conclusion

Independent deep review confirms the SMELT-ISR codebase is clean. The MoE routing mechanism is correctly
implemented, statistical methods (Welch t-test, bootstrap) are rigorous, and the data pipeline prevents
leakage. No bugs or scientific methodology issues found.
