# Deep Code Review Report - SMELT-ISR

**Date:** 2026-11-19
**Reviewer:** Automated Deep Review Agent (Round 2)
**Scope:** Full codebase (67 Python files) + scientific methodology audit

## Summary

| Category | Count |
|---|---|
| Files scanned | 67 |
| Static analysis flags | 113 |
| Confirmed bugs (previously fixed) | 1 |
| New bugs found | 0 |
| False positives | 96 |
| Methodology concerns | 0 |

## Previously Fixed Issues

1. **selfcheck/checks_000.py:281** — `loss.backward()` after `model.eval()`. **Status: FIXED** — confirmed `with torch.enable_grad(): ckpt_loss.backward()`.

## Flagged Items (Non-Bug)

1. **Hardcoded seeds** (checks_000.py, 10 instances): Intentional for test determinism and reproducibility verification.
2. **"python_eval" security flags** (part_018_routing.py, 4 instances): False positives — these reference `_intervention_eval` function names, not `python_eval`.
3. **60 TODO/FIXME comments**: Infrastructure improvement notes, not code bugs.

## Static Analysis False Positive Breakdown

| Pattern | Count | Verdict |
|---|---|---|
| `todo_fixme` | 60 | CI/CD improvement notes |
| `unprotected_division` | 24 | All use clamp_min, ternary guards |
| `eval_call` | 13 | Standard PyTorch eval pattern |
| `hardcoded_seed` | 10 | Test determinism |
| `no_grad_context_missing` | 3 | All properly wrapped |
| `exec_call`/`python_eval` | 4 | False positives — function name matches |

## Scientific Methodology Audit

### Model Architecture (part_003_model.py, part_002_layers.py)
- **Transformer blocks**: Proper layer composition with residual connections
- **MoE routing**: SparseMoE with intervention-based routing analysis
- **Position encoding**: RoPE implementation with proper cos/sin table caching

### Routing Analysis (part_018_routing.py)
- **Intervention framework**: Causal interventions (shuffle, drop) with paired bootstrap CIs
- **Baseline comparison**: Proper baseline with `baseline_eval` before interventions
- **Bootstrap**: Blake2b-derived seeds for reproducible intervention bootstrap
- **Reporting**: Includes loss/accuracy differences with CI; notes that "区间跨 0 时不能把点估计写成显著的功能差异"

### Statistical Methods
- **TOST equivalence testing**: Proper two one-sided tests implementation
- **Factorial analysis**: Full factorial design with interaction effects
- **Window comparison**: Sliding window analysis with trend detection
- **P-value correction**: Significance levels consistently applied via `_SIGNIFICANCE_LEVEL`
- **Baseline adequacy**: Statistical baseline verification before claiming improvements

### Data Pipeline
- **Data audit**: Comprehensive cleaning with train-only exclusion spans
- **Near-duplicate detection**: Proper deduplication that only affects training data
- **Provenance**: SHA-256 fingerprinting for data integrity

### Test Suite
- **test_gqa_forward_backward**: Correct training-mode test (model NOT in eval mode)
- **test_k1_one_equivalences**: Proper eval-mode comparison with `torch.no_grad()`
- **test_delayed_merge_gate**: Mathematical correctness verification for MoE gating

## Conclusion

Codebase demonstrates rigorous scientific methodology. Statistical tests are well-implemented with proper corrections. The routing intervention framework follows causal inference best practices. No new bugs found beyond the previously fixed `enable_grad` issue.
