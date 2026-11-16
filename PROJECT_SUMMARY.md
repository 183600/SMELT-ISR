# SMELT-ISR Project Summary

## 1. Project Overview

**Type**: PyTorch-based research project for Sparse MoE Transformer experiments, focusing on "Delayed-Merge" routing mechanisms.

**Language**: Python 3.10+ (pyproject.toml specifies `requires-python = ">=3.10"`)

**Framework**: PyTorch 2.x, with NumPy for numerical operations

**Purpose**: Implementing and experimenting with delayed-merge routing mechanisms for MoE (Mixture of Experts) transformers, evaluating variants A, B, local_base, smelt, base, and dense baselines. Includes extensive statistical analysis, factorial design, and reproducibility features.

**Key Features**:
- Multiple model modes: `base`, `smelt`, `local_base`, `local_base_k2`, `A`, `B`, `B_early`, `dense`
- Delayed-merge routing mechanisms (local vs global branch retention)
- Factorial experimental design with full ablation coverage
- Statistical rigor: TOST equivalence testing, Holm correction, power analysis
- Data pipeline with rigorous audit (leakage detection, duplicate removal, corrupt byte cleaning)
- Paper-reproduction fidelity reporting (architecture_faithful, protocol_faithful, numerical_reproduction)
- Test-ledger system for controlled test-set access

## 2. Directory Structure

```
SMELT-ISR/
├── experiment.py              # Entry point (backward-compatible module shim)
├── pyproject.toml             # Project config: deps, ruff, mypy settings
├── package.json               # JS dependencies (commitlint, eslint, prettier) - secondary
├── ANALYSIS_IMPROVEMENTS.md
├── CODE_REVIEW_*.md           # 15 code review reports from Nov 2026
├── CODE_REVIEW_REPORT.md      # Consolidated code review report
├── LICENSE
├── README.md                  # Comprehensive documentation (565+ lines)
│   └── README_en.md           # English version
├── requirements-ci.txt        # CI pinning: numpy==2.5.4, torch==2.14.1
├── exp/                     # Main experiment code (modular parts)
│   ├── __init__.py            # Compose parts via exec(); 53 parts loaded
│   ├── _manifest.py           # Lists CORE_PARTS and MAIN_PARTS
│   ├── _runner.py             # Isolated namespace execution helper
│   ├── part_000_config.py     # Config dataclass with all experiment settings
│   ├── part_001_paper.py      # Paper preset/fidelity logic
│   ├── part_002_layers.py     # Transformer block / attention definitions
│   ├── part_003_model.py      # LanguageModel and SparseMoE definitions
│   ├── part_004_model_data.py # Model/data configuration helpers
│   ├── part_005_data_manifest.py # Artifact manifest generation
│   ├── part_006_dataset.py    # ByteDataset with splits, groups, cleaning
│   ├── part_007_data_pipeline.py # Data processing pipeline
│   ├── part_008_data_audit.py # Leakage/duplicate audit logic
│   ├── part_009_data_cleaning.py # Corrupt/degenerate byte cleaning
│   ├── part_010_data_gates.py # Audit gate enforcement
│   ├── part_011_metrics.py    # Metrics computation (loss, ppl, auc, etc.)
│   ├── part_012_group_bootstrap # Bootstrap/resampling for CIs
│   ├── part_013_evaluation.py # Evaluation loop (train/val/test)
│   ├── part_014_ngram.py      # N-gram statistical baseline
│   ├── part_015_statistical_baseline # Statistical baseline analysis
│   ├── part_016_curve_analysis # Training curve analysis & monitoring
│   ├── part_017_bad_cases.py  # Bad case / error analysis
│   ├── part_018_routing.py    # Routing probes, RQ3/RQ4 diagnostics
│   ├── part_019_diagnostics.py # Non-finite diagnosis (NaN/Inf)
│   ├── part_020_curve_metrics # Curve metrics & diagnostics
│   ├── part_021_comparison.py # Comparison infrastructure
│   ├── part_022_comparison_protocol # Controlled-variable comparison
│   ├── part_023_factorial_helpers # Factorial design effects
│   ├── part_024_window_comparison # Windowed comparison logic
│   ├── part_025_statistics.py # Statistical tests (t, Welch, bootstrap)
│   ├── part_026_equivalence.py # TOST equivalence testing
│   ├── part_027_aggregation.py # Run aggregation helpers
│   ├── part_028_contrasts     # Contrast analysis
│   ├── part_029_ablation.py   # Ablation study framework
│   ├── part_030_factorial.py  # Full factorial effect decomposition
│   ├── part_031_ablation_coverage # Coverage checks for ablation
│   ├── part_032_baseline_adequacy # Baseline adequacy checking
│   ├── part_033_preregistration_core # Preregistration core logic
│   ├── part_034_preregistration_schema # Schema validation
│   ├── part_035_preregistration_record # Record management
│   ├── part_036_checks.py     # Numerical integrity checks
│   ├── part_037_environment_probe # Environment snapshot logic
│   ├── part_038_environment_snapshot # Environment snapshot storage
│   ├── part_039_artifacts.py  # Artifact management
│   ├── part_040_run_audit.py  # Run audit logic
│   ├── part_041_run_records.py # Run record keeping
│   ├── part_042_checkpoint_tracking # Checkpoint management
│   ├── part_043_order_sweep.py # Sweep order design (grouped/blocked/randomized)
│   ├── part_044_pilot_power.py # Pilot run power analysis
│   ├── part_045_design_plan.py # Design plan generation
│   ├── part_046_baseline_tuning.py # LR grid search for baselines
│   ├── part_047_baseline_tuning_provenance # Provenance tracking
│   ├── part_048_training_sweep.py # Training sweep orchestration
│   ├── part_049_data_plan.py  # Data plan management
│   ├── part_050_meta.py       # Metadata tracking
│   ├── part_051_repro_package.py # Reproducibility packaging
│   └── part_052_conclusion_cards # Conclusion card generation
│
├── selfcheck/                 # Self-validation test suite (7 files)
│   ├── __init__.py
│   ├── _manifest.py
│   ├── checks_000.py           # Structure/gradient/causality tests
│   ├── checks_001.py           # Sanity/repeatability/gradient flow
│   ├── checks_002.py           # Comparison/ablation statistical tests
│   └── checks_003.py           # Run provenance & environment checks
│
├── tests/                     # Aggregated unittest test modules (30 files)
│   ├── __pycache__
│   ├── test_core.py            # Aggregates all _core_* test classes
│   ├── test_paper.py           # Paper fidelity reproduction tests
│   ├── test_protocol_invariants.py # Protocol invariant regression tests
│   ├── _core_ArtifactManifestTests.py
│   ├── _core_CoreModelTests.py
│   ├── _core_CostModelTests.py
│   ├── _core_CurveMeanMetricTests.py
│   ├── _core_DataPlanTests.py
│   ├── _core_DesignPreflightTests.py
│   ├── _core_DesignPreparationTests_00.py thru _03.py
│   ├── _core_EdgeCaseRegressionTests_00.py thru _01.py
│   ├── _core_EnvironmentSnapshotTests.py
│   ├── _core_GroupClusterRegressionTests.py
│   ├── _core_NearDuplicateDataTests.py
│   ├── _core_NumericalScaleTests.py
│   ├── _core_PreregistrationTests.py
│   ├── _core_ProcessTests.py
│   ├── _core_RoutingTests.py
│   ├── _core_ReproducibilityTests.py
│   ├── _core_RunAuditTests.py
│   ├── _core_SchemaTests.py
│   ├── _core_StatisticsTests.py
│   └── _core_support.py        # Shared test support (_load_experiment, _tiny_config)
│
├── exp/mainflow/              # Main training flow (2 parts)
│   ├── part_000.py            # Argument parser (--data, --mode, --seeds, etc.)
│   └── part_001.py            # Training loop & evaluation
│
├── exp/analysisflow/           # Analysis flow (1 part)
│   └── part_000.py
│
├── docs/                      # Documentation directory
├── examples/                  # Example scripts
├── exp/                     # Top-level exp/ (same as exp/ above)
├── .github/                   # CI workflows
│   └── workflows/
│       ├── tests.yml           # pytest run on main push
│       ├── continuous-loop.yml
│       ├── opencode-issue.yml
│       └── sync-to-gitee.yml
├── .pre-commit-config.yaml    # Pre-commit hooks config
├── .gitignore
├── .git/
├── alg 力申请.md
└── 算力申请.md
```

## 3. How to Build/Test/Lint

### Installation
```bash
pip install -e .
# Or with CPU-only torch:
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

### Running Tests

**Pytest (CI)**:
```bash
python -m pytest tests/ -x -q
```

**Unittest discovery** (from README):
```bash
python -m unittest discover -s tests -v
```

**Built-in self-test**:
```bash
python experiment.py --self-test
```

**Paper fidelity tests**:
```bash
python -m unittest discover -s tests -v  # test_paper.py runs
```

### Linting

**Ruff** (configured in pyproject.toml):
```bash
ruff check .        # Check style/lint issues
ruff format .       # Format code
```

MyPy type checking:
```bash
python -m mypy .    # Type check (config: python 3.10, warn_return_any=True)
```

### Common Commands

**Sanity check only** (no training):
```bash
python experiment.py --data corpus.txt --mode B --sanity-only --out runs/sanity_B
```

**Self-test**:
```bash
python experiment.py --self-test
```

**Compare runs** (no training, just analysis):
```bash
python experiment.py --compare runs/run1 runs/run2 --compare-csv analysis/contrast.csv
```

**Paper cell preset**:
```bash
python experiment.py --data corpus.txt --mode B --paper-cell 200M:95 --paper-arch smelt --paper-protocol
```

## 4. Potential Bugs, Issues, and Areas Needing Fixing

### Confirmed Issues (from Code Review Report)
- **No severe bugs found** - The consolidated CODE_REVIEW_REPORT.md (2026-11-09) states: "代码质量**优秀**。项目实现了统计检验和实验分析框架，代码严谨，统计方法实现正确。未发现严重bug或方法学错误。"
- **除零错误** (false positives) at selfcheck/checks_001.py:999-1000 and exp/part_025_statistics.py:913 - marked as false positives, no risk
- **`eval()`调用** (45处) - all PyTorch `.eval()` method calls or function names, no risk
- **`log()`无epsilon防护** - mostly constants with proper guards, TOST has explicit handling
- **random无种子** - selfcheck code uses randomness for testing; production code seeds managed uniformly

### Design/Architecture Notes

1. **`exec()` usage** (exp/__init__.py:25, exp/_runner.py:67) - Intentional modular execution pattern. Executes local project files only, not user input. Marked as "非安全风险" in code review.

2. **`package.json` present but secondary** - Contains JS dependencies (commitlint, eslint, prettier) but the project is Python-only. These appear to be residual or for a secondary concern.

3. **Division by zero warnings** in selfcheck - marked as false positives (division by constant 0.05, or `intervals` guaranteed >= 200 by `max(200, ...)`)

4. **`log()` without epsilon** - 45 occurrences across codebase; most are `math.log(2.0)` constants; some use `max(value, eps)` protection; TOST has explicit se ≤ 0 handling

5. **Selfcheck randomness** - Several selfcheck functions have comments like "No random seed set in this function - results may not be reproducible". This is intentional - the selfcheck tests specific behaviors, and RNG state is managed at the experiment level.

### Known Areas of Active Scrutiny

Based on the 15 CODE_REVIEW_*.md files and ANALYSIS_IMPROVEMENTS.md, the project is under active code review during November 2026. The review files document:

- **Statistical method correctness**: TOST, Holm correction, power analysis - all confirmed correct
- **Numerical stability**: log-sum-exp, t-distribution CDF, boundary handling - well implemented
- **Experimental design**: Factorial design, ablation framework, sequence effect analysis - all implemented correctly
- **Suggestions for improvement**:
  - Add mathematical derivation documentation for statistical tests
  - Add more end-to-end usage examples

### Code Quality Assessment

The project demonstrates **excellent code quality** for a research codebase:

- Rigorous statistical implementation (TOST, bootstrap CIs, effect sizes)
- Comprehensive data audit pipeline with leakage detection
- Careful experimental design with factorial coverage checks
- Strong reproducibility mechanisms (source fingerprinting, environment snapshots, test ledger)
- Well-organized modular structure despite heavy `exec()` composition
- Extensive self-test coverage (7 selfcheck files with 50+ test functions)

**No blocking bugs found** that would prevent the project from functioning. The code review explicitly states "无需修复" (no need for repair).

### Reported Warnings/Edge Cases (Not Bugs)

1. **Gradient flow checks** - Some selfcheck paths note "Consider adding device= parameter to torch.tensor() to avoid device mismatch" - cosmetic, not functional

2. **AUC sampling mode** - Comments note "Consider adding device= parameter" - same class of minor issue

3. **`_SC_BASE` global mutations** in selfcheck - documented as test-only behavior

4. **`module_order` vs `name_hash` initialization differences** - intentional design decision documented in comments

5. **`dense_fnn_hidden=-1` validation** - correctly rejected; noted as potential confusion point

6. **`--max-byte-shift` / `--max-bigram-shift` / `--max-conditional-shift`** - default None means warning-only mode; hard door requires explicit setup

7. **`--budget-match` tolerance** - default 10% is a relaxation from paper's <4%, noted in comments as "default放宽到 10%% 但仍会报告实测比例"

8. **`--compare-baseline` straw-man guard** - correctly prevents comparing against unadjusted weak baselines

9. **`--suite-ablation` coverage** - properly checks that all (1+N)×2 factorial combinations are covered

10. **Seed overlap detection** - correctly identifies pseudo-replication and prevents inflated significance claims

## 5. Key Dependencies

**Direct dependencies** (pyproject.toml):
- `numpy>=1.24`
- `torch>=2.2`

**CI-pinned** (requirements-ci.txt):
- `numpy==2.5.4`
- `torch==2.14.1`

**Dev/tools** (pyproject.toml ruff config):
- Full ruff rule set (E, W, F, I, B, C4, UP, PGH, RUF, PLR, RET, SIM, TCH, PTH, TRY, DTZ, G, LOG, PERF, NPY, PLC, PLE, PLW)
- Ignored: E501 (line too long, handled by formatter), B008 (function calls in defaults), UP037, RET504
- Unfixable: B008, UP037, RET504

**Test dependencies** (implicit from tests/):
- `pytest` (used in CI)
- `unittest` (standard library, used for test discovery)
- `torch`, `numpy` (for test execution)