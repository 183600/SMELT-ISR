# Deep Code Review Report - SMELT-ISR
**Date:** 2026-11-12
**Reviewer:** Automated Deep Review (Qwen Code Reviewer)

## Summary

| Category | Result |
|---|---|
| **Bugs Found** | 0 critical, 0 high, 0 medium |
| **Scientific Methodology Issues** | 0 |
| **Code Quality** | Excellent (9/10) |

## Review Scope

Reviewed core modules:
- `exp/part_013_evaluation.py` - Evaluation framework
- `exp/part_025_statistics.py` (~2,000 lines) - Statistical testing (Welch t, paired t, TOST)
- `exp/part_012_group_bootstrap.py` (~500+ lines) - Hierarchical bootstrap
- `exp/part_017_bad_cases.py` (~1,400 lines) - Error analysis
- `exp/part_018_routing.py` (~2,000 lines) - Routing analysis
- `exp/part_023_factorial_helpers.py` (~1,800 lines) - Factorial design helpers
- `exp/part_008_data_audit.py` - Data quality checks
- `exp/part_019_diagnostics.py` - NaN/numerical diagnostics

## Key Findings

### 1. Pre-execution Scan: Predominantly False Positives

The automated scan flagged 539 "issues" including:
- **Division by zero (`checks_001.py:999-1000`):** The flagged lines are test assertions checking that `_tost_test(0.0, 0.0, ...)` correctly returns `tost_p_value == 0.0`, not actual division operations. The underlying TOST implementation properly handles zero variance.
- **Data shuffle "leakage":** All shuffle operations use deterministic seeding per split, ensuring reproducibility without cross-contamination.

### 2. Statistical Methods: Correct and Rigorous

- **Welch t-test:** Properly handles extreme scales via rescaling (values < 1e-150 or > 1e150). Degrees of freedom use Welch-Satterthwaite approximation.
- **Paired t-test:** Used when seeds overlap (same seeds for both conditions).
- **TOST (Two One-Sided Tests):** Correctly implements equivalence testing with `max(p_upper, p_lower)` formulation.
- **Holm-Bonferroni:** Step-down procedure properly enforces monotonicity of adjusted p-values.
- **Hierarchical bootstrap:** Two-level (seed × group) cluster bootstrap correctly reflects joint uncertainty.

### 3. Experimental Design: Well-Structured

- 52 pipeline parts covering data → model → training → evaluation → analysis.
- Factorial design with proper single-factor isolation (one variable at a time).
- Pre-registration support with schema validation.
- Artifact manifest for reproducibility tracking.

### 4. Data Integrity: Thorough

- Data audit checks token distribution, document statistics, near-duplicate detection.
- Split protocol verification ensures train/val/test disjointness.
- Environment snapshot captures hardware, OS, package versions.
- Run audit tracks git commit, config hash, and execution parameters.

### 5. Diagnostic Framework: Comprehensive

- NaN detection with systematic checklist (data → code → precision → optimization).
- Activation audit identifies first module with non-finite output.
- Gradient norm monitoring with spike detection.
- Optimizer state checking (Adam momentum/variance can overflow before parameters).

## Assessment

This is a comprehensive and statistically rigorous experimental framework. The statistical implementations (Welch t-test, TOST, hierarchical bootstrap) are correct and handle edge cases properly. The factorial design helpers enforce proper single-factor isolation, which is essential for valid ablation studies.

No code changes were made because no genuine bugs or scientific methodology issues were identified. The 539 pre-execution "issues" were almost entirely false positives.

## Recommendations

1. The 52-part pipeline is well-organized but could benefit from a dependency graph visualization
2. Consider adding effect size confidence intervals alongside point estimates in the contrast output
