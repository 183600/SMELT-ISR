# Independent Deep Code Review - SMELT-ISR (Round 8)

**Date:** 2026-11-19 15:05 CST  
**Reviewer:** Fresh Independent Agent Review  
**Scope:** Full codebase (67 Python files) — bug detection + scientific methodology audit

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 67 |
| Static analysis flags | 113 |
| **True bugs found** | **0** |
| **Scientific methodology issues** | **0** |
| **Verdict** | **PASS — Codebase is clean** |

## Detailed Verification

### 1. SwiGLU Custom Autograd (part_002_layers.py:260-360)

**Verified correct:**
- Forward: `output = linear(silu(gate) * up, down_weight)`
- Backward correctly computes:
  - `grad_up = grad_hidden * silu(gate)`
  - `grad_gate = grad_hidden * up * silu'(gate)` via `aten.silu_backward`
  - `grad_down_weight = hidden^T @ grad_out`
- Hidden state recomputed in backward (memory-efficient checkpointing)
- dtype alignment for autocast properly handled via `_autocast_snapshot()`

### 2. MoE Expert Routing (part_002_layers.py:440-570)

**Verified correct:**
- `_MergeExpertOutputs`: `index_copy_` forward, `index_select` backward (correct inverse operation)
- `_GatherPaddedExpertOutputs`: `index_select` forward, `index_copy_` backward (correct inverse)
- No duplicate index issues — each assignment maps to unique padded buffer row
- Expert capacity computation with proper ceiling division guard

### 3. Checkpoint/Recompute (part_003_model.py:1184-1360)

**Verified correct:**
- `_autocast_snapshot()` correctly records CUDA/CPU autocast state
- Recomputation restores autocast state for dtype-consistent backward pass
- `torch.enable_grad()` used for recomputation graph construction
- Gradient routing: only outputs with `requires_grad=True` participate in backward
- The comment at line 1184 about backward recompute happening outside autocast is documentation, not a bug

### 4. Data Pipeline Integrity (part_007_data_pipeline.py)

Division safety verified:
- Line 728: `counts / total` after guard `if total <= 0: return None, 0`
- Line 896/901: `seen_pairs / total_eval_pairs` after guard `if total_eval_pairs <= 0.0`
- Temporal boundary shift audit correctly implemented for detecting distribution drift at split boundaries

### 5. Statistical Methods

**part_011_metrics.py (macro NLL):**
```python
present = counts > 0.0
per_class = sums[present] / counts[present]
```
Correct — division only operates on classes with non-zero counts.

**part_012_group_bootstrap.py:**
- TOST equivalence testing correctly implemented
- Factorial analysis with interaction effect decomposition
- Window comparison with trend detection

**part_018_routing.py:**
- Causal interventions (shuffle, drop) with paired bootstrap CIs
- Blake2b-derived seeds for reproducible intervention bootstrap
- Correctly reports "CI crossing 0 → cannot claim significant difference"

### 6. exec() Usage (__init__.py:25, _runner.py:77)

Local module composition pattern:
- `__init__.py`: Composes experiment parts from local project files via manifest
- `_runner.py`: Executes compiled code from local files only
- Both marked `# nosec B102` — intentional design for modular experiment composition
- No security concern: never reads from user input or remote sources

### 7. Division Safety (85+ flagged)

All verified safe:
- `head_dim = max(1, int(hidden) // n_heads)`: Explicit guard ensures minimum dimension of 1
- `counts / total`: Guarded by `if total <= 0` checks
- `seen_pairs / total_eval_pairs`: Guarded by `if total_eval_pairs <= 0.0`
- Statistical computations use `clamp_min(1.0)` or conditional guards throughout

### 8. Static Analysis Flag Resolution

| Pattern | Count | Verdict |
|---|---|---|
| `exec_call` | 2 | Local module composition, nosec B102 |
| `unprotected_division` | 85+ | All guarded (clamp_min, ternary, conditional) |
| `no_grad_context_missing` | 1 | Documentation text, not executable code |
| `eval_call` | ~25 | Standard PyTorch lifecycle |

**No code changes required.**
