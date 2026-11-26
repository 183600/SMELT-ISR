# Code Review Round 7 — 2026-11-26

## Scope
- **Files scanned:** 97 Python files (excluding .git, __pycache__, .github, CI scripts)
- **Review method:** Automated static analysis (AST-based + regex pattern matching) followed by manual verification of all findings
- **Categories checked:** numerical stability, data leakage, statistical methodology, resource management, gradient correctness, shape safety

## Findings Summary

| Severity | Count |
|----------|-------|
| Error    | 0     |
| Warning  | 12 (11 false positives, 1 real bug) |
| Info     | 166 (all false positives or style) |

## Detailed Analysis

### Real Bug Found and Fixed

**exp/part_048_training_sweep.py:296** — File handle leak: `open(path, "rb").read()` creates a file object that relies on CPython reference counting for cleanup. Changed to `path.read_bytes()` which uses pathlib's context-managed I/O.

**Fix applied:**
```python
# Before:
actual_sha = hashlib.sha256(
    open(path, "rb").read()
).hexdigest()

# After:
actual_sha = hashlib.sha256(
    path.read_bytes()
).hexdigest()
```

### Warning-level findings (11 false positives verified)

1. **exp/part_003_model.py:146** — "optimizer.step() without zero_grad()". Actual code: docstring describing the weight cache invalidation design ("optimizer.step() 后经 invalidate_weight_cache 递增失效代数"). **False positive** (matched docstring).

2. **exp/part_004_model_data.py:195** — Same pattern: docstring describing the MoE weight cache invalidation protocol. **False positive**.

3. **exp/part_013_evaluation.py:509** — "F.softmax() without dim". Actual code: `F.softmax(selected_logits[...].float(), dim=-1)` — `dim=-1` IS present, split across lines. **False positive** (multi-line formatting).

4. **exp/part_019_diagnostics.py:733** — `_safe_cv`: `ratio = std / mean`. Protected by `if mean <= 0.0 or not math.isfinite(mean): return None`. **Safe**.

5. **exp/part_037_environment_probe.py:521** — Docstring/comment mentioning optimizer.step(). **False positive**.

6. **exp/mainflow/part_001.py:122** — Docstring/comment mentioning optimizer.step(). **False positive**.

7-11. **selfcheck/checks_000.py:1268-1317** and **tests/test_paper.py:455** — All are test code that calls `optimizer.step()` to verify learning rate scheduler state transitions, not actual training loops. Gradient accumulation is irrelevant here. **Intentional test pattern**.

### Info-level bulk analysis
- **106 methodology findings:** All matched in docstrings/comments or in proper statistical test wrappers.
- **48 "bug" findings:** `.view()` without `-1` (34), file open patterns (3), optimizer.step in tests (10), F.softmax (1) — all verified safe.

## Conclusion

**1 real bug found and fixed:** file handle leak in `exp/part_048_training_sweep.py`. All other findings are false positives from static analysis matching comments, docstrings, and test patterns.

## Changes Made
- **exp/part_048_training_sweep.py**: Changed `open(path, "rb").read()` to `path.read_bytes()` to prevent file handle leak.
