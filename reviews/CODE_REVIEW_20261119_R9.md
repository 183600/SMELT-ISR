# Code Review Round 9 — SMELT-ISR

**Date:** 2026-11-19  
**Reviewer:** Independent deep code review agent  
**Scope:** Full repository (67 Python files)

## Summary

**Result: No bugs found.**

This is the 9th independent review round. All static analysis flags and core scientific code were manually verified.

## Static Analysis Review

113 issues flagged across 67 files (highest count due to the multi-part experiment structure). Manual verification results:

### 1. exec_call (2 flags) — INTENTIONAL DESIGN
- `exp/__init__.py:25`: Composes experiment parts from local files with proper `# nosec B102`
- `exp/_runner.py:77`: Executes experiment parts in isolated namespace — proper sandboxing
- Both have compilation error handling and security annotations

### 2. unprotected_division (many flags) — ALL FALSE POSITIVES
- `part_000_config.py:1002,1036`: `max(1, int(hidden) // n_heads)` guards against zero head_dim
- `part_007_data_pipeline.py:728`: `total <= 0` check before division
- `part_011_metrics.py:527`: `present.any()` check ensures non-zero counts
- All divisions in metrics/analysis files have proper guards

### 3. no_grad_context_missing (1 flag) — DOCUMENTATION
- `part_003_model.py:1184`: Comment explaining autocast state restoration during backward
  recompute — not a code issue but documentation of design decision

## Deep Manual Review

### Model Architecture (part_003_model.py)
- **_BlockCheckpoint**: Custom autograd Function that properly saves/restores autocast state
  - Forward records CUDA/CPU autocast snapshots
  - Backward re-enters saved autocast context for recomputation
  - Ensures gradient equivalence with/without checkpointing

### Configuration (part_000_config.py)
- `head_dim = max(1, int(hidden) // n_heads)` — proper guard against degenerate configurations
- Architecture parameters validated at config level

### Data Pipeline (part_007_data_pipeline.py)
- Temporal boundary shift audit for distribution drift detection
- Proper normalization: `counts / total` only after `total > 0` check
- Seen/unseen context sharing metrics properly computed

### Metrics (part_011_metrics.py)
- Per-class NLL with proper masking of empty classes
- `present.any()` guard before computing means
- Bootstrap CI with window-level resampling (correct granularity)

### Statistical Baseline (part_015_statistical_baseline.py)
- Proper null model estimation
- Appropriate confidence interval construction

### Evaluation (part_013_evaluation.py)
- Proper train/eval mode management
- No data leakage between splits

### Data Leakage Prevention
- `--allow-data-leakage` flag for explicit opt-in
- Default behavior rejects cross-split duplicates
- Audit checks for train/val/test window overlap

### Curve Analysis (part_016_curve_analysis.py, part_020_curve_metrics.py)
- Proper curve fitting with numerical stability guards
- Metric computation with appropriate averaging

## Conclusion

Codebase demonstrates:
- Careful attention to autocast/checkpoint interaction
- Proper data integrity safeguards
- Sound statistical methodology
- Well-structured multi-part experiment architecture

No code changes required.
