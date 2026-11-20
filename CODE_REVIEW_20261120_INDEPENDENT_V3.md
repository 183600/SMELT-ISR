# Independent Code Review Report — SMELT-ISR
**Date:** 2026-11-20 13:07 CST
**Reviewer:** Independent Sub-Agent Review (V3)
**Scope:** Full codebase — 67 Python files — bug detection + research methodology audit

## Executive Summary

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Real bugs found | 0 |
| False positives from static analysis | 25+ |
| Code changes required | 0 |
| Code quality rating | Excellent |

## Analysis Performed

### 1. Activation Checkpointing (`_BlockCheckpoint`)
- **Non-reentrant design**: Consistent with `torch.utils.checkpoint(use_reentrant=False)`.
- **Autocast state management**: `_autocast_snapshot()` captures CUDA/CPU autocast state during forward. Backward restores state via `torch.autocast()` context managers. Critical for AMP training correctness.
- **Gradient flow**: Forward executes under `torch.enable_grad()` to support operators that check `torch.is_grad_enabled()` (e.g., route auxiliary load-balancing loss).
- **Saved tensors hooks**: `_checkpoint_discard` / `_checkpoint_restore` replace all intermediate activations with placeholders. Forward retains no activations; backward recomputes.
- **Backward pairs filtering**: Only `(output, grad)` pairs where `grad is not None` and `output.requires_grad` are passed to `torch.autograd.backward()`.

### 2. Model Architecture
- SparseMoE implementation with routing probes and expert dropout.
- Config validation: `head_dim = max(1, int(hidden) // n_heads)` prevents zero-dimension heads.
- Data pipeline: Three-way split with boundary continuity verification. Group boundaries verified not to cross splits.

### 3. Data Integrity (`part_010_data_gates`)
- Split audit: Token value range validation (byte pipeline 0..255).
- Byte histogram and bigram histogram per split.
- Bounded signature scanning for cross-split duplicate detection (leakage detection).
- Group partition reporting with `leakage_scan_coverage` tracking.
- `--allow-data-leakage` escape hatch for intentional overlap.

### 4. Statistical Methodology
- Bootstrap confidence intervals with window-level resampling.
- AR(1) variance inflation for autocorrelated bootstrap.
- Group-level bootstrap for class-aware uncertainty estimation.
- Precision-recall AUC with recall gain normalization.
- Majority accuracy baseline for class imbalance interpretation.

### 5. Evaluation Protocol
- Comprehensive metrics: NLL, PPL, bits-per-byte, accuracy, F1 (micro/macro/weighted).
- Bad case collection for qualitative analysis.
- Routing probe for MoE diagnostic analysis.
- Window-based comparison with block bootstrap for temporal autocorrelation.

### 6. Static Analysis False Positives
All flagged issues verified as false positives:
- `exec_call` (2): Intentional experiment composition from local files with `# nosec B102`
- `unprotected_division` (6+): All guarded by conditionals, `max(1, ...)`, or prior validation
- `no_grad_context_missing` (2+): Documentation comments or training loop context
- `eval_call` (6): Correctly placed in evaluation contexts with `@torch.inference_mode()`
- `todo_fixme` (10): Known improvement areas (CI reporting), not bugs

### 7. Seed Management
- Centralized seed setting at experiment entry point.
- Module-level seed setting explicitly removed (documented rationale).
- Local `torch.Generator().manual_seed()` for all stochastic paths.
- Separate generators for different random streams (probe, batch, etc.).

## Conclusion
The SMELT-ISR codebase is well-structured and production-ready. The custom activation checkpointing with proper AMP state management is particularly noteworthy. No bugs or methodology issues found. No code changes required.
