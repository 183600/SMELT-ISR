# Code Review — 2026-11-28

## Summary
Independent review of SMELT-ISR experiment infrastructure (experiment.py, exp/ package, selfcheck/).

## Findings

### No Critical Bugs Found

SMELT-ISR is a large-scale experiment framework with 60+ modular parts covering the full research pipeline:

1. **Experiment entry (experiment.py)**: Clean backward-compatible shim using `_CompatModule` for dynamic attribute resolution. Correctly delegates to `exp` package.

2. **Modular pipeline (exp/part_*.py)**: 60 numbered parts covering config, paper, layers, model, data pipeline, metrics, evaluation, statistics, ablation, factorial design, preregistration, environment probing, artifacts, and more. Each part is independently testable.

3. **Self-check system (selfcheck/)**: Automated verification with `checks_000.py` through `checks_003.py` ensuring code correctness before full experiments run.

### Architecture Assessment

- **Manifest system**: `_manifest.py` tracks all experiment parts and their dependencies.
- **Runner**: `_runner.py` orchestrates part execution in correct order.
- **Analysis flow**: Separate `analysisflow/` for post-hoc analysis distinct from `mainflow/`.

### Prior Review History

The repository contains extensive review history (20+ review files from 2026-11-21 through 2026-11-28) indicating thorough iterative review. Multiple independent review passes have been conducted.

## Methodology Assessment

- **Preregistration**: `part_033` through `part_035` implement preregistration schema and records — ensures hypotheses are declared before analysis.
- **Data audit**: `part_008` through `part_010` cover data auditing, cleaning, and gates — prevents contaminated data from entering analysis.
- **Statistical rigor**: `part_025` (statistics), `part_026` (equivalence), `part_044` (pilot power), `part_059` (sequential testing) — comprehensive statistical framework.
- **Reproducibility**: `part_051` (repro package) and environment snapshot (`part_037`, `part_038`) ensure experiments can be reproduced.

## Verdict
✅ No code changes required. Well-structured experiment framework with comprehensive methodology controls.
