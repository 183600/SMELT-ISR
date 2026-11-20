# Code Review Report — SMELT-ISR
**Date:** 2026-11-20  
**Reviewer:** Automated Deep Review Agent  

## Summary

Deep code review of the SMELT-ISR repository (95 Python files). Focused on syntax correctness, model architecture, routing logic, and statistical methodology.

## Bugs Found

**No syntax errors or critical bugs found.** All Python files compile successfully.

## Issues Reviewed (No Fix Needed)

### Division Guards
- `count / total_wrong` in `exp/part_021_comparison.py:165`: guarded by `if total_wrong > 0` check at line 163. Also wrapped in try/except OverflowError for extreme values.
- `math.ceil(num_windows / count_cap)` in `exp/part_013_evaluation.py:1134`: `count_cap` validated as positive integer at lines 1038-1042 (`count_cap <= 0` raises ValueError).
- `math.ceil(total_valid / count_cap)` in `exp/part_013_evaluation.py:1166`: same validation as above.
- All other divisions in metrics code use `clamp_min`, `max(1, ...)`, or conditional guards.
- **Verdict:** All divisions properly guarded.

### `exec()` Usage
- `exp/__init__.py:25`: `exec(compile(_part_source, str(_part_path), "exec"), globals())` with `# nosec B102`.
- `exp/_runner.py:77`: `exec(code, namespace)` with `# nosec B102`.
- Both are controlled code execution for experiment orchestration with safety annotations.
- **Verdict:** Acceptable with existing safety annotations.

### `backward()` in Test Functions
- 5 instances in `tests/_core_CoreModelTests.py` and `tests/test_paper.py` — all are intentional gradient verification tests.
- Test names like `test_all_modes_forward_backward`, `test_top1_router_receives_lm_gradient` confirm gradient testing.
- **Verdict:** Correct usage — not bugs.

### Hardcoded Seeds
- `seed(0)` in `exp/part_019_diagnostics.py` efficiency measurement — correct for reproducible timing.
- `seed(0)` in self-check scripts — correct for deterministic validation.
- **Verdict:** Standard practice.

### Research Methodology
- P-value computations use appropriate tests for individual hypotheses.
- Bootstrap confidence intervals are used where appropriate.
- Factorial experiment design is sound.
- **Verdict:** Methodology is correct.

## Conclusion

The repository is well-structured with no syntax errors or critical bugs. The experiment orchestration uses `exec()` appropriately with safety annotations. All numerical operations are properly guarded. Research methodology is sound.
