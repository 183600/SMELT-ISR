# Code Review Report — 2026-11-26 (Independent Review)

**Reviewer**: QwenGo! Code Review Agent  
**Repository**: SMELT-ISR  
**Scope**: Full code review of experiment framework (53 parts), self-checks, tests, and documentation

## Summary

**Verdict: No new bugs or scientific methodology issues found.**

This independent review follows Round 6 and confirms all recent fixes are correctly applied across the large experiment framework.

## Areas Reviewed

### 1. Experiment Framework (`exp/part_*.py`)
- **Data pipeline** (`part_006`–`part_010`): Byte-level dataset with proper train/val/test splitting, deduplication, and contamination detection. The `_eval_windows` method uses fixed (non-random) windows for reproducible evaluation.
- **Statistical methods** (`part_012`, `part_025`–`part_028`): Group-level bootstrap CI, equivalence testing, factorial ANOVA, and ablation coverage analysis all follow standard statistical methodology.
- **Curve analysis** (`part_016`): Training curve diagnostics (overfitting, underfitting, plateau, diverging) with gradient norm spike detection are well-designed for automated monitoring.

### 2. Data Integrity
- **Deletion bias recording**: Correctly documents that data cleaning is not random (deduplication preferentially removes earlier/repeated groups), making pre/post cleaning comparisons unreliable without this audit.
- **Conditional bigram shift**: Properly detects distribution drift in `P(next_byte | prev_byte)` between train and eval splits.
- **n-gram contamination check**: Correctly prevents test data from leaking into training through shared n-gram sequences.

### 3. Model & Training
- **MoE implementation**: `_StackedExpertWeights` autograd function correctly handles per-expert gradient slicing, matching `torch.stack` backward semantics.
- **Segment attention mask**: Document boundary masking is shared across all blocks within a batch, ensuring consistent treatment.

### 4. Self-checks (`selfcheck/`)
- **Schema validation**: Checks verify that all experiment parts produce outputs matching their declared schemas.
- **Protocol invariants**: Tests verify that data pipeline invariants (split boundaries, window overlap, deduplication) hold across all configurations.

### 5. Recent Fixes Verified
- `.clamp_min(1e-10)` artifact removal from docstrings, comments, schema strings, and error messages: ✅ Correctly applied across 24+ files
- Curve interpolation denominator fix (`d88bbd7`): ✅ Correct — adjacent point spacing < 1 was causing undersized interpolation results
- Module-level seed setting removal: ✅ Correct — prevents silent RNG state reset on import

## Conclusion

The SMELT-ISR experiment framework is a well-structured, scientifically rigorous system with comprehensive data integrity checks, proper statistical methodology, and reproducible evaluation protocols. No new issues identified.
