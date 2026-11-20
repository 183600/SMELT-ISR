# Independent Code Audit — 2026-11-21 (v2)

## Scope
Full review of SMELT-ISR repository: static analysis triage, statistical methodology validation, numerical stability audit, and scientific methodology review.

## Static Analysis Triage
No high-priority issues flagged. 157 low-priority issues reviewed — no actionable bugs.

## Numerical Stability
- ✅ All division operations protected via `clamp_min()` or conditional checks
- ✅ Token-level loss computation uses proper `clamp_min(1e-10)` guards
- ✅ Metric computation uses `predicted.clamp(min=1.0)` and `support.clamp(min=1.0)` for precision/recall
- ✅ Weighted statistics use `sampled_weights.clamp_min(1e-10)` and `weights.sum().clamp_min(1)`
- ✅ Bootstrap uses deterministic seed from data fingerprint (blake2b hash)

## Statistical Methods Validation
- ✅ `evaluate()`: Comprehensive evaluation with weighted loss, ppl, bits-per-byte, macro/weighted F1, precision/recall
- ✅ Bootstrap CI uses group-level resampling with percentile intervals
- ✅ Equivalence testing (TOST) properly implemented with pre-registered margins
- ✅ Bonferroni correction applied in equivalence analysis (`alpha_bonferroni = alpha/m`)
- ✅ Holm-Bonferroni correction implemented for multiple hypothesis testing
- ✅ All eval functions properly guarded with `@torch.inference_mode()` and `@torch.no_grad()`

## Scientific Methodology
- ✅ Test split sealed — only called once for final evaluation
- ✅ Proper separation of val (model selection) and test (final assessment)
- ✅ Bootstrap seeds derived from data fingerprint for reproducibility
- ✅ Group-level bootstrap accounts for within-group correlation
- ✅ Equivalence margins pre-registered before training
- ✅ import torch properly present in all files that use it (prior fix confirmed)

## Code Quality
- ✅ No TODO/FIXME markers in critical paths (only documentation references to escape sequences)
- ✅ Exception handling is specific and well-documented
- ✅ No hardcoded paths or environment-specific configurations

## Conclusion
No new bugs found. All prior fixes verified correct. Code is well-structured with thorough statistical methodology.
