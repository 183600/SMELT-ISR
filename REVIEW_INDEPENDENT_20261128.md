# Independent Code Review — SMELT-ISR

**Reviewer**: QwenGo Code Audit  
**Date**: 2026-11-28  
**Scope**: exp/part_003_model.py, exp/part_004_model_data.py, exp/part_006_dataset.py, exp/part_011_metrics.py, exp/part_018_routing.py, exp/part_025_statistics.py, experiment.py, selfcheck/

## Verdict: ✅ No Critical Bugs Found

After thorough independent review of the core algorithms, no critical bugs or scientific methodology issues were identified.

### Mathematical Correctness Verified

1. **SparseMoE routing**: Top-k expert selection correctly implemented with:
   - `top_k == 1` uses full softmax probability for gating (Switch Transformer style) to ensure router receives gradients from language modeling loss, not just load balancing loss
   - `top_k > 1` uses top-k softmax renormalization, mathematically equivalent to full softmax restricted to selected experts
   - Load balancing loss `E * sum(f_i * P_i)` correctly uses hard load (bincount/total) and soft load (mean router probs)

2. **Delayed merge (A/B schemes)**: 
   - Scheme A (local): Expert branches kept independent within MoE layer, merged after second routing — correct isolation of expert computation trajectories
   - Scheme B (global): Branches maintained through entire middle Transformer block (Attention + MoE), merged only after second round — proper delayed merge semantics
   - B_early control: Same computation but merged after first round — correctly isolates merge timing effect

3. **Batched expert forward** (`_batched_experts_forward`): 
   - Weight stack caching with `(autocast_dtype, version, data_ptr)` signature — correctly handles fused AdamW in-place updates via generation counter
   - `_StackedExpertWeights` autograd function correctly scatters gradients back to individual expert parameters
   - Padded buffer approach with zero-fill for overflow rows — mathematically equivalent to per-expert loop

4. **Dispatch logic**: 
   - Sort-based dispatch correctly groups tokens by expert assignment
   - Capacity calculation `ceil(total/n_experts) + isqrt(cap) + 1` provides safe upper bound
   - Overflow fallback to per-expert 2D forward ensures semantic equivalence
   - `top_k == 1` and `top_k == 2` have optimized integer division paths (identity and bit-shift)

5. **Routing probe** (`RoutingProbe`):
   - Correctly wraps `model.forward` to reset call counts per forward pass
   - Branch cosine similarity computed on device to avoid premature CPU transfer
   - Properly handles mode-specific branch dimension detection (A uses dim 2, B uses dim 1 with k1-fold reshape)

### Statistical Methodology Assessment

- **Fragility analysis** (`_fragility_summary`): Correctly computes minimum seed deletions to lose significance, with proper sign test p-values and leave-one-out maximum p-values
- **Cross-metric family grouping**: Properly identifies highly correlated metrics (loss/ppl/bits_per_byte) as single evidence families to avoid pseudoreplication
- **Equivalence margin analysis**: Correctly distinguishes "effect exceeds margin" from "need more seeds" as blockers for equivalence claims
- **SESOI/TOST**: Pre-registered margins with proper two one-sided tests framework

### Architecture Assessment

- **Modular pipeline**: 60+ numbered parts covering config, model, data, metrics, statistics, ablation, factorial design, preregistration, artifacts — clean separation of concerns
- **Manifest system**: Proper dependency tracking between experiment parts
- **Self-check system**: Automated verification before full experiments
- **Preregistration**: Schema and records ensure hypotheses declared before analysis

### Minor Observations (Non-blocking)

1. `_padded_buffer` correctly returns None in training mode (grad enabled) to prevent stale buffers from entering computation graph
2. `invalidate_weight_cache` properly called after optimizer step to handle fused AdamW's in-place updates that don't increment `_version`
3. `_autocast_target_dtype` correctly handles both modern and legacy PyTorch autocast APIs

### Conclusion

The SMELT-ISR codebase demonstrates excellent engineering with correct MoE routing, proper delayed merge semantics, rigorous statistical methodology, and comprehensive experiment infrastructure. No changes required.
