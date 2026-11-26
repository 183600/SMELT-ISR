# Code Review — 2026-11-27 (Independent Review)

**Reviewer**: Independent Code Audit  
**Scope**: Experiment pipeline (exp/), evaluation (part_013_evaluation.py), statistics (part_025_statistics.py), data audit (part_008_data_audit.py), metrics (part_011_metrics.py)

## Summary

**No bugs or methodology issues found.**

SMELT-ISR implements a comprehensive experimental pipeline for byte-level language modeling evaluation. The 53-part modular design covers data processing through statistical analysis with thorough audit trails.

## Detailed Findings

### 1. Evaluation (`exp/part_013_evaluation.py`)

- **Inference mode safety**: The `evaluate()` function correctly uses `@torch.inference_mode()` decorator with explicit `model.eval()` and `torch.no_grad()` wrapping.
- **Split restriction**: Only `val` and `test` splits are accepted, with `test` reserved for final evaluation.
- **Confusion matrix**: Accumulated in int64 on-device to avoid overflow with large vocabularies (up to 256 byte classes).
- **Bootstrap**: Window-level bootstrap with configurable replicates for confidence intervals on loss, PPL, accuracy, and F1 metrics. The `bootstrap_block` parameter controls the resampling granularity.
- **Group-level statistics**: When `--group-separator` defines document boundaries, group-level accumulators enable cluster bootstrap (documents as resampling units), correctly handling non-independent overlapping windows.
- **AUC computation**: One-vs-rest PR-AUC and ROC-AUC with minimum support threshold (`_AUC_MIN_SUPPORT`) and chance-level baseline reporting.
- **Class-balanced NLL**: Per-class loss sums enable macro NLL computation alongside frequency-weighted metrics.

### 2. Statistics (`exp/part_025_statistics.py`)

- **Fragility analysis**: Comprehensive fragility index computation including `min_seed_deletions_to_lose_significance`, sign test p-values, and leave-one-out maximum p-values.
- **Equivalence testing**: TOST (Two One-Sided Tests) with margin computation and sample size planning.
- **Scale replication**: Meta-analysis pooling with fixed and random effects models, Cochran's Q, I², tau², and prediction intervals.
- **Deduplication**: Contrasts are correctly deduplicated by unordered (baseline, variant) pair + factor to prevent double-counting when the same comparison appears in both single-factor and direct comparison outputs.

### 3. Data Audit (`exp/part_008_data_audit.py`)

- **Split isolation**: Temporal ordering verification with contiguous bounds checking and overlap detection.
- **Distribution shift**: Byte-level and bigram JS divergence between train/val/test splits.
- **Deletion bias**: Cleaning operations are audited for non-random deletion patterns (byte JS divergence and bigram JS divergence of dropped content).
- **Label coverage**: Tracks unseen byte classes in evaluation splits and their token share.
- **Content health**: Detects degenerate groups (single byte dominating) and cross-split content duplicates.

### 4. Metrics (`exp/part_011_metrics.py`)

- **Averaging clarity**: Explicit distinction between macro_f1, weighted_f1, and micro_f1 (= accuracy), with naming in output keys.
- **Imbalance reporting**: `imbalance_report` flags the regime where ROC-AUC becomes misleadingly optimistic (>1000:1 negatives-per-positive), directing readers to PR-AUC instead.
- **Wilson intervals**: Applied consistently to binomial proportions (accuracy, pass rates).

### 5. Experiment Pipeline Architecture

- **53-part modular design**: Each part has a single responsibility, with clear dependencies and artifact passing.
- **Provenance tracking**: SHA-256 fingerprints of data, configs, code version, and RNG state.
- **Self-check system**: `selfcheck/` package with multi-layer verification of implementation invariants.
- **Pre-registration support**: `part_033`-`part_035` implement pre-registration schema and record keeping.

### 6. Methodology Assessment

- **Paired evaluation**: Fixed evaluation windows shared across all runs enable paired statistical comparisons at the window level, complementing seed-level tests.
- **Cluster bootstrap**: Document-level resampling when group boundaries are defined, addressing the non-independence of overlapping windows.
- **Multiple comparison correction**: Holm-Bonferroni for confirmatory tests, Benjamini-Hochberg for exploratory screens.
- **Trend testing**: `paired_trend_test` fits per-seed slopes to detect dose-response relationships without pseudo-replication.

## Conclusion

No actionable issues found. The SMELT-ISR pipeline demonstrates exceptional methodological rigor with comprehensive audit trails, statistical testing, and reproducibility measures.
