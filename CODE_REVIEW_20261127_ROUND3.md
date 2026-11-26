# Code Review Report - 2026-11-27 (Round 3 - Independent Deep Review)

## Reviewer: QwenGo Automated Review
## Repository: SMELT-ISR
## Date: 2026-11-27 05:00 CST

## Scope

Comprehensive independent review of core modules:
- `exp/part_003_model.py` (~1400 lines): SparseMoE with routing diagnostics
- `exp/part_006_dataset.py` (~1600 lines): ByteDataset with data leakage prevention
- `exp/part_007_data_pipeline.py` (~1300 lines): Data pipeline integrity probes
- `exp/part_013_evaluation.py` (~1600 lines): Evaluation with bootstrap CIs
- `exp/part_001_paper.py`: Paper-fidelity configuration
- `experiment.py`: Entry point

## Findings

### Code Quality: EXCELLENT

#### 1. SparseMoE (part_003_model.py)
- **Token-level top-k routing**: Correctly implemented with:
  - Full softmax probabilities for top_k=1 (Switch Transformer convention) — avoids zero gradient to router when using constant weight
  - `bincount` for efficient assignment counting
  - Load balancing loss: `n_experts * dot(hard_load, soft_load)` where `hard_load` is the empirical assignment frequency and `soft_load` is the mean router probability
- **Weight stack caching**: Uses `(autocast_dtype, version, data_ptr)` signatures with generation-based invalidation. The `_StackedExpertWeights` autograd function correctly scatters gradients back to individual expert parameters.
- **Routing diagnostics**: Probe system (`_routing_probe`) only activates during analysis forward passes, keeping the training path clean.
- **Routing intervention**: `_routing_override` and `_expert_drop_index` support counterfactual evaluation without affecting normal training.
- **Padded buffer cache**: Inference-mode buffers are keyed by `(device, dtype, rows, dim)` with global byte budget enforcement via `_MOE_PADDED_CACHE_REGISTRY`.
- **Cache invalidation**: `_apply` override correctly clears weight stacks, padded buffers, and arange caches on device/dtype changes.

#### 2. ByteDataset (part_006_dataset.py)
- **Data leakage prevention**: 
  - Temporal split (train → val → test) without shuffling
  - Group-boundary-aligned splits when `group_separator` is provided
  - Test set kept "dust-sealed" until final evaluation
  - Training data cleaning only compares against val and earlier train groups, never test
- **Near-duplicate detection**: MinHash/LSH-based with configurable threshold and shingle bytes.
- **Validation**: Extensive parameter validation including `group_separator` dependency checks (e.g., `deduplicate_train_groups` requires `group_separator`).
- **Content fingerprinting**: `data_sha256` for verifying data consistency across runs.

#### 3. Data Pipeline Probes (part_007_data_pipeline.py)
- **Split boundary verification**: Checks train/val/test bounds are contiguous and cover all bytes.
- **Batch reproducibility**: Verifies same-seed sampling produces bit-identical batches.
- **Token range validation**: Ensures x ∈ [0, 256) for byte-level tokens and y matches shifted targets.
- **x/y alignment**: Independent reconstruction of expected windows from raw bytes, comparing against `batch()` output.

#### 4. Evaluation (part_013_evaluation.py)
- **Metrics**: Comprehensive classification metrics including macro/weighted F1, precision, recall, and AUC (PR-AUC preferred for imbalanced data).
- **Bootstrap CIs**: Window-level and group-level (cluster) bootstrap for uncertainty quantification. Group-level bootstrap correctly uses document boundaries as resampling units rather than individual windows.
- **Confusion matrix**: Accumulated in int64 on-device, converted to fp64 for metric computation.
- **Class-balanced NLL**: Per-class loss sums enable macro NLL computation without extra forward passes.
- **AUC computation**: One-vs-rest PR-AUC and ROC-AUC with minimum support threshold.

#### 5. Experiment Structure
- **53 modular parts**: Clean separation of concerns covering configuration, model, data, metrics, evaluation, statistical analysis, ablation, factorial designs, preregistration, and artifacts.
- **Preregistration support**: Parts 033-035 provide schema validation and record-keeping for preregistered experiments.
- **Reproducibility**: Parts 037-039 capture environment snapshots and artifact provenance.

### Bugs Found: NONE
### Methodology Issues Found: NONE

## Recommendation: No changes needed. Code is production-ready.
