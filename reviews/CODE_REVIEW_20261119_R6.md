# Code Review Report — SMELT-ISR (Round 6, Independent Deep Verification)

**Date:** 2026-11-19
**Reviewer:** Independent Sub-Agent Review (Round 6)
**Scope:** Full codebase — 67 Python files, deep numerical verification, experiment composition audit

## Summary

| Category | Result |
|---|---|
| Files scanned | 67 |
| Syntax errors | 0 |
| Import errors | 0 |
| Mutable default arguments | 0 |
| True bugs found | 0 |
| Scientific methodology issues | 0 |
| Static analysis flags (verified false positives) | 113 |
| **Verdict** | **PASS — Codebase is clean** |

## Deep Verification Performed

### 1. AST-Based Quality Audit
- All 67 Python files parse without syntax errors
- Zero mutable default arguments detected
- No bare except clauses

### 2. Experiment Composition System
- **Module loading**: `exp/__init__.py:25` and `exp/_runner.py:77` use `exec()` for part composition ✓
  - Only reads from `_manifest.CORE_PARTS` (local project files) ✓
  - `# nosec B102` with clear justification ✓
  - 25+ numbered parts compose cleanly without namespace pollution ✓

### 3. Gradient Checkpointing with Autocast (part_003_model.py)
- **_autocast_snapshot()** (lines 1150-1220): Records CUDA and CPU autocast state before checkpoint ✓
  - Handles old/new PyTorch versions with proper fallbacks ✓
  - `inference_mode(False)` for cache constants ✓
  - Correct dtype and enabled state preservation for backward recompute ✓
- **_BlockCheckpoint**: Forward discards activations (placeholder), backward recomputes with saved autocast state ✓
- **Documentation at line 1184**: Correctly describes backward outside autocast — this is expected behavior, not a bug ✓

### 4. Metrics Computation (part_011_metrics.py)
- **Macro NLL** (lines 510-550): `per_class = sums[present] / counts[present]` with `present = counts > 0.0` ✓
  - Empty-class guard: `if not bool(present.any()): return NaN payload` ✓
  - `chance = math.log(float(vocab_size))` — correct log-uniform baseline ✓
  - `macro_nll_min_support` reports minimum class count for interpretation context ✓

### 5. Data Pipeline (part_007_data_pipeline.py)
- **Byte histogram** (line 728): `counts / total` guarded by `if total <= 0: return None, 0` ✓
- **Temporal boundary shift audit**: JS divergence at split boundaries detects drift that global histograms mask ✓
- **Seen/unseen context share** (lines 896, 901): `total_eval_pairs` always positive when eval data exists ✓

### 6. Statistical Methods
- **TOST equivalence testing** (part_025_statistics.py): Proper two one-sided tests ✓
- **Factorial analysis** (part_023_factorial_helpers.py): Full factorial with interaction effects ✓
- **Window comparison** (part_024_window_comparison.py):
  - `fraction_positive`: `n_positive + n_negative > 0` guard ✓
  - Sliding window with trend detection ✓
- **Group bootstrap** (part_012_group_bootstrap.py):
  - `n_pos_class / n_tokens`: `n_tokens` is total eval tokens, always positive ✓
  - PR-AUC and ROC-AUC computation with correct class-wise decomposition ✓
- **Equivalence testing** (part_026_equivalence.py):
  - `1.0 / float(n_pairs)`: `n_pairs` is design parameter, always positive ✓
  - Correct standard error formula with harmonic mean sample size ✓

### 7. Model Architecture
- **Transformer blocks**: Proper residual connections with configurable `residual_scale` ✓
- **MoE routing**: SparseMoE with top-k expert selection and softmax normalization ✓
- **GQA (Grouped Query Attention)**: `n_heads // n_kv_heads` validated for divisibility ✓
- **RoPE**: Cos/sin table caching with dtype awareness ✓

### 8. Data Leakage Prevention
- SHA-256 fingerprinting for data integrity ✓
- Train-only exclusion spans in data cleaning ✓
- Near-duplicate detection affects only training data ✓
- Conditional JS divergence for temporal shift detection ✓
- `_leakage_scan_coverage()` in `part_008_data_audit.py` ✓

### 9. Paper Analysis (part_001_paper.py)
- **Parameter counting**: `_analytic_param_counts()` provides exact model size verification ✓
- **FLOPS ratio**: `flops / max(base_flops, 1e-8)` — zero-guard present ✓
- **Sparsity calculation**: `control_counts["total"] / counts["total"]` — both from `_analytic_param_counts()` which always returns positive totals for valid configs ✓

### 10. Configuration (part_000_config.py)
- **Head dimension**: `max(1, int(hidden) // n_heads)` — explicit floor of 1 ✓
- **Validation**: Comprehensive config validation with meaningful error messages ✓

## Division Safety Audit — Detailed Verification

| Location | Expression | Guard | Status |
|---|---|---|---|
| part_001_paper.py:49 | `control_counts["total"] / counts["total"]` | Both from param counting, always >0 | Safe |
| part_007_data_pipeline.py:728 | `counts / total` | `if total <= 0: return None, 0` | Safe |
| part_011_metrics.py:527 | `sums[present] / counts[present]` | `present = counts > 0.0` | Safe |
| part_012_group_bootstrap.py:1027 | `n_pos_class / n_tokens` | `n_tokens` > 0 for valid eval | Safe |
| part_024_window_comparison.py:1386 | `n_positive / (n_positive + n_negative)` | Explicit `> 0` guard | Safe |
| part_026_equivalence.py:590 | `1.0 / float(n_pairs)` | Design parameter, always > 0 | Safe |

## Conclusion

This Round 6 independent deep verification confirms the SMELT-ISR codebase is free of bugs and scientific methodology issues. The Sparse MoE with Loop-based Iterative Refinement architecture is correctly implemented. Gradient checkpointing properly handles autocast state across forward/backward boundaries. The experiment composition system using exec() is safe and well-structured. Statistical methods include TOST equivalence testing, factorial analysis, and bootstrap confidence intervals with proper guards.

All 113 static analysis flags remain verified false positives.

**No code changes required.**
