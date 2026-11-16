# Potential Bugs Found in SMELT-ISR Codebase

## Summary

A thorough analysis of the SMELT-ISR codebase at `/home/runner/work/SMELT-ISR/SMELT-ISR` identified the following potential bugs and code quality issues.

## 1. `exp/__init__.py` lines 23-25: `exec()` Usage

**File**: `exp/__init__.py`, lines 23-25

```python
# nosec B102: exec() is used intentionally to compose experiment parts
# from local project files only, never from user input.
exec(compile(_part_source, str(_part_path), "exec"), globals())  # nosec B102
```

**Issue**: Uses `exec()` to dynamically load experiment parts from local files. While the comment indicates this is intentional and restricted to local files, this pattern:
- Masks import errors since execution happens at module load time
- Creates a security surface if local files are ever compromised
- Makes the code harder to debug and analyze

**Status**: Intentional pattern for the refactored experiment structure, but a code smell

## 2. `exp/_runner.py` lines 75-77: `exec()` Usage

**File**: `exp/_runner.py`, lines 75-77

```python
# nosec B102: exec() is used intentionally to execute experiment
# parts from local project files only, never from user input.
exec(code, namespace)  # nosec B102
```

**Issue**: Same `exec()` pattern as above, used to execute experiment parts in an isolated namespace.

**Status**: Intentional pattern

## 3. `selfcheck/checks_003.py`: Assertion Issues

**File**: `selfcheck/checks_003.py`, lines 6-39

**Issue**: Assertions depend on `check_run_provenance()` function behavior. The assertions check provenance consistency across runs (script SHA256, git commit, hashseed, etc.), but if the function has edge cases (e.g., empty provenance dicts, missing keys), the assertions could:
- Fail spuriously in edge cases
- Mask real provenance tracking bugs
- Cause false-positive failure reports

**Notable assertions**:
- `assert same_provenance['ok'], same_provenance['mismatches']` - line 6
- `assert 'script_sha256' in same_provenance['checked_keys']` - line 7
- `assert not legacy_provenance['commits_differ']` - line 9
- `assert recorded_none['commits_differ']` - line 13

**Status**: Real issue - assertions should be more robust with better error handling

## 4. `exp/part_018_routing.py` line 189: Potential Division by Zero

**File**: `exp/part_018_routing.py`, line 189

```python
batch = rows // cfg.k1
```

**Issue**: Integer division by `cfg.k1`. If `cfg.k1 == 0`, this causes `ZeroDivisionError`.

**Likelihood**: Low - `cfg.k1` is typically >= 1 in valid configurations, but the risk exists if someone configures `k1=0`.

**Risk**: Crash when running with `k1=0` configuration.

**Suggestion**: Add guard: `batch = rows // max(cfg.k1, 1)`

## 5. `exp/part_018_routing.py` line 497: Potential Division by Zero

**File**: `exp/part_018_routing.py`, line 497

```python
batch = rows // (k1 * time_steps)
```

**Issue**: Integer division by `k1 * time_steps`. If either `k1 == 0` or `time_steps == 0`, this causes `ZeroDivisionError`.

**Likelihood**: Low but possible in edge cases.

**Risk**: Crash.

**Suggestion**: Add guard: `batch = rows // max(k1 * time_steps, 1)`

## 6. `exp/part_015_statistical_baseline.py` line 1226: Documented NaN/Inf Issue

**File**: `exp/part_015_statistical_baseline.py`, line 1226

```python
# 溢出成 inf，进而 (value-low)/(high-low) 变成 inf/inf=NaN，
# int(NaN) 在展示路径抛异常。归一化到 [-1,1] 后分块位置
# 不变；正常量级不缩放，保持历史输出逐位一致。
```

**Issue**: Documented known issue where normalization can overflow to `inf/inf=NaN`.

**Mitigation**: Already handled with scale check at lines 1230-1237:
```python
scale = max(abs(value) for value in values)
if scale > 1.0e150:
    low = min(value / scale for value in values)
    high = max(value / scale for value in values)
    normalized = [value / scale for value in values]
else:
    low = min(values)
    high = max(values)
    normalized = values
```

**Status**: Documented issue, properly mitigated

## 7. `exp/part_025_statistics.py` lines 383-384: Division Guard with Statistical Impact

**File**: `exp/part_025_statistics.py`, lines 383-384

```python
old_denominator = (
    term_a * term_a / max(n_a - 1, 1)
    + term_b * term_b / max(n_b - 1, 1)
)
```

**Issue**: Uses `max(n_a - 1, 1)` and `max(n_b - 1, 1)` to guard against zero division in Welch-Satterthwaite degrees of freedom calculation.

**Impact**: When `n_a = 1` or `n_b = 1`, the denominator uses 1 instead of 0, which changes the statistical result. However, the code already handles the zero_variance case at lines 353-372, returning appropriate results.

**Status**: Real issue but properly mitigated with existing guards and zero-variance handling

## 8. `exp/part_003_model.py` line 559: Division with Guard

**File**: `exp/part_003_model.py`, line 559

```python
static_cap = -(-total // max(n_experts, 1))
```

**Issue**: Integer division by `n_experts` with guard `max(n_experts, 1)`.

**Status**: Properly guarded, no issue

## 9. `exp/part_019_diagnostics.py` line 137: Division with Guard

**File**: `exp/part_019_diagnostics.py`, line 137

```python
usage_share = usage.float() / usage.sum().clamp_min(1)
```

**Issue**: Division by `usage.sum()` with `clamp_min(1)` guard.

**Status**: Properly guarded, no issue

## 10. `exp/part_042_checkpoint_tracking.py` line 1278: Division Guard

**File**: `exp/part_042_checkpoint_tracking.py`, line 1278

```python
start = rng.randrange(len(base)) if base else 0
rotation = (start + block_index) % max(len(base), 1)
```

**Issue**: Uses `max(len(base), 1)` to guard against ZeroDivisionError in modular arithmetic.

**Status**: Properly guarded, no issue

## Conclusion

The codebase shows good defensive programming practices with many guards against division by zero, NaN, and Inf issues. The most significant findings are:

1. **`exec()` usage patterns** in `exp/__init__.py` and `exp/_runner.py` - intentional but code-smelly
2. **Assertions in `selfcheck/checks_003.py`** that depend on provenance function behavior and could fail spuriously
3. **Two potential ZeroDivisionError paths** in `exp/part_018_routing.py` if config values are 0

Most numerical issues are already properly guarded with `clamp_min()`, `max(..., 1)`, or explicit zero checks.