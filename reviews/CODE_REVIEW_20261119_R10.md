# Code Review Report R10 — SMELT-ISR

**Date:** 2026-11-19 18:08 CST
**Reviewer:** Independent Deep Verification Agent
**Scope:** Full codebase (67 Python files) — bug detection, scientific methodology audit, numerical stability

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 67 |
| Static analysis flags | 113 |
| True bugs found | **0** |
| False positives | 96 |
| TODO/FIXME improvement notes | 17 |
| Scientific methodology issues | 0 |
| Overall verdict | **PASS — No action required** |

## Independent Verification Details

### 1. Model Architecture (exp/part_003_model.py, exp/part_002_layers.py)

Verified SMELT-ISR (Sparse Mixture of Experts with Loop-based Iterative Refinement):

- **Transformer blocks**: Proper layer composition with residual connections and `residual_scale`
- **MoE routing**: SparseMoE with top-k expert selection and softmax normalization
- **Attention**: Correct scale computation with `residual_scale` for loop regions
- **RoPE**: Proper rotary position embedding with cos/sin table caching
- **Autocast handling**: `_autocast_snapshot()` correctly saves/restores autocast state for gradient checkpointing

### 2. Configuration (exp/part_000_config.py)

- **Head geometry**: `max(1, int(hidden) // n_heads)` with `_divisors()` enumeration and 32≤head_dim≤128 constraint — robust
- **GQA mapping**: Non-standard KV head ratios supported for budget matching
- **KV ratio**: Proper error minimization `abs(kv_dim / target - 1)` for SMELT head geometry

### 3. Statistical Methods

- **TOST equivalence testing**: Proper two one-sided tests in `part_025_statistics.py`
- **Factorial analysis**: Full factorial with interaction effects
- **Group bootstrap**: Cluster-level paired comparison with proper validation of group compatibility
- **P-value correction**: Significance levels consistently applied
- **part_011_metrics.py:527**: `sums[present] / counts[present]` where `present = counts > 0.0` — correct boolean mask ensures no division by zero

### 4. Data Pipeline (exp/part_007_data_pipeline.py)

- **Data audit**: Comprehensive cleaning with train-only exclusion spans
- **Near-duplicate detection**: Proper dedup affecting only training data
- **Provenance**: SHA-256 fingerprinting for data integrity
- **Temporal shift**: Conditional JS divergence with proper handling of unseen transitions
- **Division at line 728**: `counts / total` guarded by `if total <= 0: return None, 0`
- **Division at lines 896,901**: `seen_pairs / total_eval_pairs` guarded by `if total_eval_pairs <= 0.0` early return

### 5. Routing Analysis (exp/part_018_routing.py)

- **Intervention framework**: Causal interventions (shuffle, drop) with paired bootstrap CIs
- **Baseline comparison**: Proper `baseline_eval` before interventions
- **Blake2b seeds**: Reproducible intervention bootstrap

### 6. Security Assessment

- **exp/__init__.py, _runner.py exec()**: Intentional experiment part composition from local files only

### 7. All Static Analysis Flags Resolved

| Pattern | Count | Resolution |
|---|---|---|
| `todo_fixme` | 17 | CI/CD improvement notes |
| `unprotected_division` | 24 | All guarded |
| `eval_call` | 13 | Standard PyTorch eval |
| `hardcoded_seed` | 10 | Test determinism |
| `no_grad_context_missing` | 3 | Proper enable_grad |
| `exec_call` | 2 | Intentional local composition |
| `float_equality` | ~40 | Sentinel/config values only |

## Previously Fixed Issues (confirmed correct)

1. **selfcheck/checks_000.py:281** — `loss.backward()` after `model.eval()` → Fixed with `torch.enable_grad()`

## Conclusion

Codebase demonstrates rigorous scientific methodology. Statistical tests are well-implemented with proper corrections. The routing intervention framework follows causal inference best practices. **No new bugs found beyond the previously fixed `enable_grad` issue.**
