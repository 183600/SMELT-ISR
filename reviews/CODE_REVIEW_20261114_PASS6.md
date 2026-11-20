# Code Review Report - SMELT-ISR
## Pass 6: Independent Deep Source Code Audit

**Date:** 2026-11-14 05:00 UTC
**Reviewer:** QwenGo Independent Review System (Pass 6)
**Scope:** Full independent source code audit — focus on statistical methods,
model architecture, experiment pipeline, and data integrity.

---

## Summary

### Verdict: CODE QUALITY CONFIRMED — No new issues found

This pass performed direct source-code reading of all core modules, with
particular focus on:
- `exp/part_025_statistics.py` (2026 lines) — Welch t-test, TOST equivalence,
  non-central t power analysis, Bayesian contrasts, minimum detectable effect
- `exp/part_003_model.py` (1378 lines) — model architecture, residual
  connections, expert routing, gradient checkpointing
- `exp/part_007_data_pipeline.py` — data loading, seeding, reproducibility
- `selfcheck/` — artifact manifest, regression tests

---

## Detailed Checks Performed

### 1. Welch t-test (part_025_statistics.py:270-420)
**Status:** ✅ CORRECT

- **Scale invariance:** Extreme magnitudes handled via recursive rescaling:
  - `max_abs < 1e-150`: scale up to prevent underflow to zero variance.
  - `max_abs > 1e150`: scale down to prevent overflow in `**2`.
  - Effect and SE rescaled back with finiteness checks.
- **Welch-Satterthwaite df:** Dual computation path:
  - Normal path: `se⁴ / (term_a²/(n_a-1) + term_b²/(n_b-1))` with
    finiteness and subnormal-range guards.
  - Scaled path: normalizes by `max(term_a, term_b)` to prevent both
    overflow and underflow in the squared terms.
- **Zero-variance handling:** Correctly distinguishes `delta == 0`
  (p=1.0, "zero_variance") from `delta ≠ 0` (p=0.0, "noise-free_separation").

### 2. Non-central t Power Analysis (part_025_statistics.py:649-1145)
**Status:** ✅ CORRECT

- **`_t_power` function:** Uses numerical integration of the non-central
  t-distribution via Gauss-Legendre quadrature with adaptive node counts.
- **Large-shift regime:** When the non-centrality parameter is large,
  switches to a normal-mixture approximation to prevent quadrature
  instability.
- **Small-df large-scale regime:** Special handling for extreme parameter
  combinations where the standard quadrature nodes miss the probability mass.
- **Minimum detectable effect:** Binary search on the power curve with
  convergence checks and fallback to analytical approximations.

### 3. Bayesian Contrasts (part_025_statistics.py:1863-1972)
**Status:** ✅ CORRECT

- **Jeffreys-Zellner-Siow prior:** Uses the JZS Bayes factor with
  Cauchy-distributed effect sizes under H1.
- **Log Bayes factor:** Computed via numerical integration of the
  log-marginal-likelihood ratio with Gauss-Legendre quadrature.
- **Posterior summaries:** Mixture of point mass at zero (under H0) and
  continuous posterior (under H1), weighted by the Bayes factor.
- **Trapezoidal integration:** `_trapz_integral` provides a fallback for
  cases where Gauss-Legendre nodes are insufficient.

### 4. Model Architecture (part_003_model.py)
**Status:** ✅ CORRECT

- **Residual connections:** Properly handle dtype mismatches in autocast
  scenarios by casting residuals to match the stream dtype.
- **Expert routing:** Index clamping ensures routing indices stay within
  valid expert range.
- **Gradient checkpointing:** Correctly implemented with a custom autograd
  function that saves inputs (not outputs) for recomputation.

### 5. Data Pipeline (part_007_data_pipeline.py)
**Status:** ✅ NO LEAKAGE

- **Seeding:** Dedicated `torch.Generator` instances for probe and batch
  streams, both seeded from a single `seed` parameter.
- **Reproducibility check:** Same-seed second sampling verified to be
  bit-identical, catching RNG state corruption.
- **No fit-on-test:** All preprocessing (normalization, tokenization)
  is fit on training data only.

### 6. Self-check System (selfcheck/)
**Status:** ✅ ROBUST

- Artifact manifest tracks all experiment outputs with SHA-256 digests.
- Regression tests verify statistical test correctness with known-value
  assertions (e.g., t-test p-values at extreme parameter values).
- Edge case tests verify numerical precision at boundary conditions
  (e.g., df~1.5e8, confidence near 1.0).

## Conclusion

The SMELT-ISR codebase maintains exceptional quality in its statistical
methods implementation. The Welch t-test handles all edge cases correctly,
the power analysis uses adaptive numerical methods, and the Bayesian
contrasts use proper prior specification. No bugs or methodology errors
were found. No code changes were necessary.
