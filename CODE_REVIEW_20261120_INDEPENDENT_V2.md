# Independent Code Review Report — SMELT-ISR
**Date:** 2026-11-20 12:10 UTC  
**Reviewer:** Independent Code Review Agent (Qwen)  
**Scope:** Full codebase bug detection + research methodology audit  
**Python files reviewed:** 67  

## Executive Summary

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Critical bugs found | 0 |
| Research methodology issues | 0 |
| Defensive improvements applied | 0 |
| Code quality rating | Excellent |

## Detailed Analysis

### 1. Division Safety (25 flagged → 0 real bugs)
All flagged divisions verified:
- `part_000_config.py:1002,1036` — `max(1, hidden // n_heads)` guard ensures head_dim >= 1
- `part_001_paper.py:49` — `counts["total"]` always positive for valid model configs
- `part_011_metrics.py:527` — `sums[present] / counts[present]` where `present = counts > 0.0`
- `part_012_group_bootstrap.py:985` — `n_pos.gather()` in boundary computation with positive counts
- `part_013_evaluation.py:1134,1166` — `math.ceil(num_windows / count_cap)` with positive count_cap
- `part_018_routing.py:323,331` — `n_experts > 0` guard before division
- `part_018_routing.py:1136,1137` — `total_tokens <= 0` early return before division
- `part_021_comparison.py:165` — `total_wrong > 0` guard before division
- `part_024_window_comparison.py:125,129` — `n_windows = len(loss_differences)` with prior empty check via `_mean()` returning None
- `part_024_window_comparison.py:706-711` — `total_mean_var_norm > 0.0` guard

### 2. Gradient Flow (4 flagged → 0 real bugs)
- `part_019_diagnostics.py:909,922` — `output["loss"].backward()` in training timing measurement, within `autocast` context (correct for capturing AMP training behavior)
- `part_003_model.py:1184` — Documentation comment about `_BlockCheckpoint` backward behavior, not actual call site

### 3. exec() Usage (2 flagged → acceptable)
- `exp/__init__.py:25` — Intentional experiment part composition from local files with `compile()` validation
- `exp/_runner.py:77` — Experiment runner executing local project files only

### 4. Eval/Train Mode (7 flagged → 0 real bugs)
All `model.eval()` calls properly paired with `torch.no_grad()` or `torch.inference_mode()`.
Training diagnostics correctly use `model.train()` before backward passes.

### 5. Research Methodology — PASS
- Three-way split (train/val/test) with boundary continuity verification
- Group boundaries verified to not cross splits
- Temporal boundary shift audit for distribution drift detection
- Conditional JS divergence for seen/unseen context analysis
- Bootstrap CI with window-level resampling and AR(1) variance inflation
- Group-level bootstrap for class-aware uncertainty estimation
- Comprehensive metrics: NLL, PPL, bits-per-byte, accuracy, F1 (micro/macro/weighted)
- Majority accuracy baseline for class imbalance interpretation
- Custom activation checkpointing with proper AMP state management
- Routing probe for MoE diagnostic analysis

### 6. Code Architecture
- Clean modular experiment design with numbered parts
- SparseMoE implementation with routing probes and expert dropout
- `_BlockCheckpoint` correctly saves/restores autocast state for gradient recomputation

## Conclusion
The SMELT-ISR codebase is well-structured and production-ready. No bugs or research methodology issues found. The code demonstrates rigorous attention to numerical stability, proper PyTorch idioms, and sound experimental methodology.
