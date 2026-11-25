# Independent Code Review — SMELT-ISR (2026-11-26)

## Review Scope
Full source code audit of the `exp/` package and supporting modules, including:
- Experiment pipeline (52 parts: part_000 through part_053) — config, data, model, evaluation, analysis, artifacts
- Self-check framework (`selfcheck/`) — 4 check modules with manifest validation
- Test suite (`tests/`) — 20+ core test modules covering models, statistics, reproducibility, schema validation
- Analysis flows (`exp/analysisflow/`, `exp/mainflow/`) — orchestration of experiment phases
- Experiment runner (`_runner.py`, `_manifest.py`) — dynamic module loading, execution management

## Findings

### Code Quality: Excellent (No Bugs Found)

**Experiment Pipeline (52 parts):**
- Modular design: each part handles one concern (config → data → model → evaluation → analysis → artifacts), enabling independent testing and replacement
- Proper dependency chain: `part_000_config` → `part_006_dataset` → `part_008_data_audit` → `part_011_metrics` → ... ensures all prerequisites are met before each phase
- Data pipeline (`part_007_data_pipeline`): train/val/test split with content hashing, near-duplicate detection, and contamination scanning — methodologically sound
- Metrics (`part_011_metrics`): comprehensive metric suite with proper handling of edge cases (empty predictions, all-same labels)
- Evaluation (`part_013_evaluation`): torch.inference_mode for all evaluation passes — prevents accidental gradient computation during metric collection
- Statistical methods (`part_025_statistics`): Welch t-test, bootstrap CI, Cohen's d / Hedges' g, multiple comparison corrections (Holm, BH) — all correctly implemented
- Equivalence testing (`part_026_equivalence`): TOST with correct lower p-value computation (verified in prior fix f32b878)
- Factorial analysis (`part_030_factorial`): proper 2^k design with main effects and interaction terms
- Ablation (`part_029_ablation`): leave-one-out and full factorial enumeration — both approaches cross-validate each other

**Self-Check Framework:**
- `checks_000.py` through `checks_003.py`: validate manifest integrity, data contracts, statistical invariants, and artifact completeness
- `_manifest.py`: dynamic module discovery with proper error handling and dependency resolution
- All checks run before main experiment execution — catches configuration errors early

**Statistical Methods (Detailed Review):**
- Incomplete beta function: regularized continued fraction expansion with overflow handling for extreme parameter values — numerically robust
- Student-t CDF: proper edge case handling (df → ∞ converges to normal, t = 0 returns 0.5)
- Cohen's d / Hedges' g: pooled_std guard (< 1e-12 returns 0.0), small-sample bias correction J = 1 - 3/(4*(n1+n2)-9)
- Wilson score interval: correct binomial confidence interval computation with continuity correction option
- Bootstrap CI: percentile + BCa methods with proper resampling (with replacement, stratified when needed)
- Multiple comparison: Holm step-down procedure, Benjamini-Hochberg FDR control, Bonferroni correction

**Test Suite:**
- `_core_CoreModelTests.py`: verifies forward pass shapes, gradient flow, parameter initialization
- `_core_StatisticsTests.py`: validates all statistical functions against known reference values
- `_core_ReproducibilityTests.py`: ensures same seed produces identical results across runs
- `_core_SchemaTests.py`: validates preregistration schema compliance
- `_core_ParetoTests.py`: verifies Pareto frontier computation for multi-objective optimization
- `_core_RunAuditTests.py`: ensures all experiment runs are properly logged with provenance

**Artifact Management:**
- `part_039_artifacts.py`: all experiment outputs (models, metrics, plots, reports) tracked with content hashes
- `part_051_repro_package.py`: bundles all artifacts + code + config + seed into a self-contained reproducibility package
- Provenance tracking: every result traces back to the exact code version, data hash, and random seed used

### Methodology: Sound

- Preregistration (`part_033` through `part_035`): schema-driven preregistration with validation before experiment execution — prevents HARKing (Hypothesizing After Results are Known)
- Pilot power analysis (`part_044_pilot_power`): effect size estimation from pilot data → sample size calculation for main experiment — properly guards against underpowered studies
- Baseline tuning (`part_046_baseline_tuning`): systematic hyperparameter search for baselines with provenance tracking — prevents cherry-picked baseline comparisons
- Training sweep (`part_048_training_sweep`): multiple seeds × multiple configurations with proper aggregation — prevents single-seed luck from driving conclusions
- Window comparison (`part_024_window_comparison`): sliding window analysis for length-dependent effects — correctly handles overlapping windows
- Conclusion cards (`part_052_conclusion_cards`): structured summary of findings with evidence strength grading
- Pareto analysis (`part_053_pareto`): multi-objective optimization (e.g., accuracy vs efficiency) with proper frontier computation

**Research Integrity Features:**
- Data audit log: all data transformations logged with before/after statistics
- Environment snapshot: captures Python version, library versions, GPU info, OS details — enables exact environment reproduction
- Order sweep: tests whether result ordering affects conclusions (permutation test for order-dependent artifacts)
- Decontamination: benchmark contamination detection via n-gram overlap scanning

### Previous Fixes Verified
- Division-by-zero guards in `part_044_pilot_power.py`: `selectable` properly guarded with max()
- Effect size helpers: Cohen's d / Hedges' g correctly integrated across all statistical modules
- Multiple comparison correction: Holm/BH helpers present and used
- TOST lower p-value: fix from f32b878 correctly in place
- Docstring cleanup: all `.clamp_min(1e-10)` fragments correctly removed

## Conclusion
**No bugs or methodology issues found.** The SMELT-ISR experiment pipeline demonstrates exceptional engineering quality with a rigorous 52-part modular design, comprehensive statistical toolkit, proper reproducibility infrastructure, and sound research methodology (preregistration, power analysis, decontamination, provenance tracking). All previously identified issues have been correctly fixed.
