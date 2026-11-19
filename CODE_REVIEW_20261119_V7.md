# Independent Code Review Report - SMELT-ISR (Round 7)

**Date:** 2026-11-19
**Reviewer:** Independent Agent Review (Round 7 - Fresh Analysis)
**Scope:** Full codebase (67 Python files) - bug detection + scientific methodology audit

## Summary

| Category | Result |
|---|---|
| Files scanned | 67 |
| Static analysis flags | 113 |
| True bugs found | 0 |
| Scientific methodology issues | 0 |
| **Verdict** | **PASS - Codebase is clean** |

## Independent Verification

### 1. SwiGLU Custom Autograd (part_002_layers.py:260-360)

`_SwiGLUDown` forward/backward verified:
- Forward: `output = linear(silu(gate) * up, down_weight)`
- Backward correctly computes:
  - `grad_up = grad_hidden * silu(gate)`
  - `grad_gate = grad_hidden * up * silu'(gate)` via `aten.silu_backward`
  - `grad_down_weight = hidden^T @ grad_out`
- Hidden state recomputed in backward (memory-efficient checkpointing)
- dtype alignment for autocast properly handled

### 2. MoE Expert Routing (part_002_layers.py:440-570)

- `_MergeExpertOutputs`: `index_copy_` forward, `index_select` backward (correct inverse)
- `_GatherPaddedExpertOutputs`: `index_select` forward, `index_copy_` backward (correct inverse)
- No duplicate index issues since each assignment maps to unique padded buffer row

### 3. Checkpoint/Recompute (part_003_model.py:1184-1360)

- `_autocast_snapshot()` correctly records CUDA/CPU autocast state
- Recomputation restores autocast state for dtype-consistent backward
- `torch.enable_grad()` used for recomputation graph
- Gradient routing: only outputs with `requires_grad=True` participate

### 4. Data Pipeline Integrity (part_007_data_pipeline.py)

Division safety verified:
- Line 728: `counts / total` after guard `if total <= 0: return None, 0`
- Line 896/901: `seen_pairs / total_eval_pairs` after guard `if total_eval_pairs <= 0.0`
- Temporal boundary shift audit correctly implemented

### 5. Statistical Methods (part_011_metrics.py, part_012_group_bootstrap.py)

- `macro_nll`: `per_class = sums[present] / counts[present]` where `present = counts > 0.0`
- TOST equivalence testing correctly implemented
- Factorial analysis with interaction effect decomposition
- Window comparison with trend detection

### 6. MoE Routing Analysis (part_018_routing.py)

- Causal interventions (shuffle, drop) with paired bootstrap CIs
- Blake2b-derived seeds for reproducible intervention bootstrap
- Correctly reports "CI crossing 0 → cannot claim significant difference"

### 7. Static Analysis Flag Resolution (113 flags)

| Pattern | Count | Verdict |
|---|---|---|
| `exec_call` | 2 | Local module composition, nosec B102 |
| `unprotected_division` | 85+ | All guarded (clamp_min, ternary, conditional) |
| `no_grad_context_missing` | 1 | Documentation text, not executable |
| `eval_call` | ~25 | Standard PyTorch lifecycle |

**No code changes required.**
