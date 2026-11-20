# Code Review Report — SMELT-ISR
**Date:** 2026-11-19  
**Reviewer:** Automated multi-agent code review  
**Scope:** Full codebase (67 Python files, ~35K LOC)

## Summary

The SMELT-ISR repository implements a comprehensive experimental framework for sequence modeling with mixture-of-experts (MoE) routing, statistical analysis, and rigorous evaluation protocols. Code quality is **excellent** — the modular pipeline architecture (`part_000` through `part_050+`) provides clear separation of concerns with thorough defensive programming.

## Issues Found

### No Bugs Found

After thorough review of all core modules (model, dataset, evaluation, routing, statistics, meta-analysis, contrasts), no functional bugs were identified.

### Static Analysis Review

| Pattern | Count | Verdict |
|---------|-------|---------|
| `eval()` calls | 6 | All in proper training/eval toggle |
| Unprotected division | 25+ | All verified safe with guards |
| `loss.backward()` | 6 | All in proper training context (with/without GradScaler) |

### Verified Division Safety

| Location | Expression | Guard |
|----------|-----------|-------|
| `part_018_routing.py:323` | `sum(usage_share) / n_experts` | `if n_experts > 0` |
| `part_018_routing.py:1136` | `total_loss / total_tokens` | `if total_tokens <= 0: return nan` |
| `part_024_window_comparison.py:125` | `/ n_windows` | `n_windows = len(loss_differences)`, unreachable if empty (mean returns None first) |
| `part_025_statistics.py:346` | `var_a / n_a` | `n_a >= _MIN_SEEDS_FOR_STATS (3)` enforced earlier |
| `part_028_contrasts.py:75` | `relative_a² / n_a` | `n_a >= 2 and n_b >= 2` in guard |
| `part_050_meta.py:259` | `/ total_weight` | `_META_MIN_STUDIES = 2` enforced; all variances validated `> 0` |
| `part_011_metrics.py:527` | `sums[present] / counts[present]` | `present` mask ensures non-zero counts |
| `part_007_data_pipeline.py:728` | `counts / total` | `if total <= 0: return None` |
| `part_007_data_pipeline.py:896` | `seen_pairs / total_eval_pairs` | `if total_eval_pairs <= 0.0: return` |

### Scientific Methodology Assessment

**Strengths:**
- Welch t-test with recursive normalization for extreme scales (1e-150 to 1e150)
- `_finite_values()` is all-or-nothing, preventing silent truncation bias
- Meta-analysis with `_META_MIN_STUDIES = 2` guard and variance positivity validation
- DerSimonian-Laird random effects with prediction interval
- Cochran's Q, I², τ² heterogeneity diagnostics
- Window-level paired analysis prevents pseudo-replication
- Two-way variance decomposition (seed × window)
- N-gram statistical baseline with grid-tuned smoothing
- Data audit pipeline (temporal boundary shift, cleaning, leakage detection)
- Gradient flow diagnostics and environment probe
- Gradient checkpointing with proper autocast handling

**Notable design patterns:**
- `_welch_t_test` handles three scale regimes (normal, tiny <1e-150, huge >1e150) via recursive rescaling
- All statistical tests return structured dicts with explicit `"test"` type and `"invalid_values"` abstention
- Evaluation properly separates training and held-out data with temporal isolation checks

**No methodological deficiencies found.**

## Recommendation

No code changes needed. The pipeline architecture is well-organized and statistically sound.
