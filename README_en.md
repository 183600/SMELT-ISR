# SMELT-ISR

> **SMELT with Independent Second Round Routing (Delayed-Merge)**

**English** | [中文说明](README.md)

## 📖 Introduction

SMELT-ISR is a routing-improvement experiment built on the [SMELT](https://arxiv.org/html/2609.01343v1) (Sparse MoE Transformer, middle layers loop twice) architecture.

In the original SMELT, expert outputs of each MoE round are merged immediately while the middle layers loop. This project proposes a **delayed-merge** mechanism:

1. **No merge after round 1**: outputs of the Top-$k_1$ experts are kept as independent branches, neither summed nor concatenated.
2. **Independent round-2 routing**: for each round-1 branch state, round-2 Top-$k_2$ expert routing is performed independently.
3. **Final aggregation**: results are merged across round-1 branches only after the full second round completes.

This structure explores whether delayed merging improves the expressiveness of expert combinations and whether different branches can form fully independent computation trajectories.

## 🚀 Key features

- **Four model modes**:
  - `smelt`: the original SMELT architecture (middle layers loop twice, merged immediately each round).
  - `A`: local delayed merge (expert branches stay independent only within the same MoE layer, merged after phase two).
  - `B`: global delayed merge (after round 1 creates independent branches, the entire middle Transformer block — attention and MoE — stays branch-independent, merging only after round 2).
  - `local_base`: an immediate-merge control for scheme A, isolating the benefit of delayed merging itself.
- **Rigorous self-test module**: verifies gradient flow, autoregressive causality, information isolation between branches, and mathematical equivalence at boundary conditions (e.g., scheme B equals SMELT when $k=1$).
- **Complete training foundation**: resume from checkpoints, gradient clipping, load-balancing loss, LR scheduling, and detailed experiment metrics.
- **Controlled comparisons and statistical reliability**: every run records a comparison protocol (optimizer, LR schedule, training steps, data augmentation, data fingerprint, ...); `--compare` verifies that everything except the declared experimental factors is identical; `--seeds` runs multiple random seeds and reports Mean ± Std with significance tests; `--suite-ablation` runs the 2×2 full-factorial ablation and checks coverage automatically.

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.x

```bash
pip install "torch>=2.2"
```

## 📦 Usage

### 1. Run structural self-tests
Before experimenting, run the built-in self-tests to verify structural correctness, branch independence and equivalence:
```bash
python experiment.py --self-test
```

### 2. Prepare data
The code defaults to a byte-level tokenizer for quick validation. Prepare a UTF-8 plain-text file (e.g., `corpus.txt`).

The pipeline is strictly leakage-safe: **split first, then process**. The corpus is cut chronologically into `train` (first 90%) / `val` (next 5%) / `test` (last 5%): the three splits are contiguous, non-overlapping, and never randomly shuffled (time-series data must train on the past and validate/test on the future). Batch-level shuffling during training is provided by random window starts and never crosses split boundaries. The byte pipeline performs no standardization/normalization/imputation; if you extend it to continuous features, every statistic (mean, variance, min/max, ...) must be fitted on `train` only and applied to val/test via `transform`.

`val` is used for periodic validation, hyperparameter selection, early stopping and model selection; `test` stays sealed for the whole training run and is evaluated exactly once after training, only when `--evaluate-test` is passed. Re-evaluation within the same run is refused. Fractions are configurable with `--val-fraction` / `--test-fraction`. Before training, a data audit (token range, text-integrity/label-noise checks for bytes that cannot be UTF-8 decoded and for NUL bytes, duplicate windows, train↔val distribution shift) runs automatically and is stored in the `data_audit` field of `config.json`; unsealing the test set appends a test-inclusive audit.

**Grouped data must stay on one side**: if the corpus is a concatenation of groups (the same document / user / object producing multiple records), pass `--group-separator` (e.g., `--group-separator '<|doc|>'` or `--group-separator '\n\n'`; `\n`, `\xHH`, `\uXXXX` escapes are supported). Split positions are snapped to group boundaries, so no group is ever cut across two splits and train/val/test never cross. The audit records the number of whole groups per split. When snapping cannot satisfy the minimum length of every split (e.g., a single oversized group), the script raises an explicit error instead of silently splitting a group.

**Strict data audit**: with `--strict-data-audit`, leakage-class findings (cross-split duplicate windows, out-of-range tokens, UTF-8/NUL corruption above threshold) abort training; when unsealing the test set, the same problems in test refuse the unsealing (no `test` record is written). By default these are warnings only, so you can inspect the data before deciding how to clean it.

Whenever a new best validation loss is reached, the weights are written to `best.pt`. The `final` record and the single test evaluation both use this validation-selected model (the selection policy is recorded in `model_selection` inside `config.json`); if `best.pt` is missing (e.g., resuming from an older checkpoint), the script says so explicitly and falls back to the last-step parameters.

```bash
# Run the single final test evaluation after training
python experiment.py --data corpus.txt --mode B --out runs/B_seed42 --evaluate-test
```
*(Note: replace with your own tokenizer and data pipeline for real experiments.)*

### 3. Start training
Select the mode via `--mode`.

**Original SMELT baseline**:
```bash
python experiment.py \
  --data corpus.txt \
  --mode smelt \
  --top-k 2 \
  --steps 2000 \
  --seed 42 \
  --out runs/smelt_seed42
```

**Global delayed merge (scheme B)**:
```bash
python experiment.py \
  --data corpus.txt \
  --mode B \
  --top-k 2 \
  --k1 2 \
  --k2 2 \
  --steps 2000 \
  --seed 42 \
  --out runs/B_seed42
```

**Local delayed merge (scheme A) and its control**:
```bash
python experiment.py --data corpus.txt --mode local_base --k1 2 --k2 2 --out runs/local_base_seed42
python experiment.py --data corpus.txt --mode A --k1 2 --k2 2 --out runs/A_seed42
```

### 4. Pre-training overfit sanity check
Before every fresh training run (i.e., without `--resume`), the script first
draws a small batch (`--sanity-batch-size`, 4 sequences by default) and trains
the same model architecture on it with all regularization disabled — the
auxiliary load-balancing loss, weight decay, gradient clipping and warmup are
turned off. Training must drive the loss quickly toward 0 or the next-token
accuracy toward 100% (thresholds: `--sanity-loss-threshold`,
`--sanity-acc-threshold`). On failure the script reports the problem and exits
with a non-zero status instead of blindly starting full training, since failing
to overfit even a tiny batch usually indicates broken gradients, a wrong loss,
or an optimizer/LR misconfiguration.

Note this is a **necessary but not sufficient** check: passing it does not prove
the code is fully correct — bugs in data augmentation, feature alignment or
label mapping can still slip through, so intermediate-result inspection is
still recommended.

```bash
# Run the sanity check only, without starting training
python experiment.py --data corpus.txt --mode B --sanity-only --out runs/sanity_B

# Explicitly skip the check (once the setup is well validated)
python experiment.py --data corpus.txt --mode B --skip-sanity-check --out runs/B_seed42
```

Results are stored in `metrics.jsonl` (`{"step": "sanity", ...}`) and in the
`sanity_check` field of `config.json`. The check fixes its own seed and restores
the RNG state afterwards, so running it does not change the main training
trajectory.

### 5. Resume training
Model config is read from the checkpoint automatically; just keep data and total steps consistent:
```bash
python experiment.py \
  --data corpus.txt \
  --resume runs/B_seed42/last.pt \
  --steps 2000 \
  --out runs/B_seed42
```

## 🧪 Experimental design and research questions

- **RQ1**: Does delayed merge beat the original SMELT? (validation loss & convergence speed)
- **RQ2**: How does the split between round-1 branches ($k_1$) and round-2 experts ($k_2$) affect results? (e.g., combinations under $k_1k_2 \approx 2k$)
- **RQ3**: Does round 2 choose different experts than round 1? (expert overlap & Jaccard similarity)
- **RQ4**: Does delayed merging improve expert specialization? (expert load distribution & routing entropy)

### Controlled-comparison conventions and reproducibility

- **Scheme B routing and merge weights**: scheme B uses $k_1$ throughout the first-round intermediate region and $k_2$ only in the full second-round region; merging uses fixed weights from round-1 routing with Top-$k_1$ normalization, never recomputed from final branch states. Analyses/ablations should state this explicitly.
- **Compute accounting**: after round-1 branch creation, scheme B executes downstream attention and MoE at $k_1$× branch count; parameter-matched is not FLOPs-matched. When answering RQ1/RQ2, report both step-aligned and token/FLOPs-aligned comparisons to avoid misattributing extra compute to the routing mechanism.
- **Residual-scaling convention**: `residual_scale` applies globally to every residual update in all regions (inside and outside the looped span). The SMELT paper scales residual updates within the looped span by $1/r$ (1/2 for two visits); this base does not implement region-specific $1/r$ scaling. Keep `residual_scale` equal across modes; a global 0.5 is not equivalent to the paper's convention.
- **Load-balancing scale**: $f_i$ in the auxiliary loss is normalized by token count, not assignment count; under uniform routing $E \sum_i f_i P_i \approx k$, so balancing strength grows linearly with routing width. When comparing across $k_1/k_2/\text{top-}k$, scale the coefficient accordingly (keep `balance_coef × k` constant); otherwise routing width is confounded with balancing strength.
- **Validation/test protocol**: each evaluation uses fixed windows evenly spaced across the corresponding evaluation split, independent of any RNG, bit-aligned across models and resumed runs. Every new best validation loss is snapshotted to `best.pt`, and both the `final` record and the single test evaluation run on this validation-selected model (the `model_selection` field records the policy; when `best.pt` is unavailable the script reports it and falls back to the last-step parameters). The test split stays sealed for the entire training run and is evaluated exactly once when `--evaluate-test` is passed (`{"step": "test", ...}`); repeating the flag is refused, so the test set cannot silently become a validation set. All results are appended to `metrics.jsonl`.
- **Determinism**: the training entry point fixes Python / NumPy / PyTorch (CPU+CUDA) / cuDNN seeds; data sampling is driven by an explicit `torch.Generator` whose seed and state are checkpointed; `CUBLAS_WORKSPACE_CONFIG` is set and PyTorch is asked to prefer deterministic kernels (a warning is emitted if a CUDA kernel lacks a deterministic implementation). Small cross-hardware numeric differences from reduction order may remain.
- **Experiment tracking**: each run automatically records the code revision (git commit / dirty flag), environment (Python / torch / CUDA / device), full hyperparameters and seeds, data fingerprint (SHA-256, byte count), training curves and final results in `config.json` and `metrics.jsonl` — no manual bookkeeping required.
- **Controlled-variable checking**: `config.json` stores a `comparison_protocol` field (optimizer AdamW with betas/eps, LR and schedule, warmup, total steps, weight decay, gradient clipping, batch size, data augmentation, data fingerprint, model architecture, evaluation protocol and model-selection policy — new runs select on validation, legacy runs without a record are marked as last-step). `--compare` requires all of these to match exactly unless declared as a factor via `--compare-factors` (default `mode,k1,k2,balance_coef,seed,dense_fnn_hidden`); a run that silently changed optimizer, LR, step count, or the `final` measurement basis is reported as a violation, so tuning gains cannot be misattributed to a method.
- **Statistical reliability**: `--seeds 42 43 44` trains the same configuration for multiple seeds and reports Mean ± Std (sample std) per experimental cell. Shared seed sets use a paired t-test, otherwise Welch's t-test; p-values and the |Δ|/pooled-std noise ratio are shown. Cells with fewer than 3 seeds are flagged as underpowered, and an apparent +0.2% gain with 0.5% seed variance is not treated as real.
- **Ablation completeness**: `--suite-ablation` runs the 2×2 full factorial of delayed merge × balance loss (`local_base`/`A` × `balance_coef`=0/current), covering all four cells needed to show that each design choice helps on its own; the coverage check lists any missing cell. `--suite-modes` runs multiple modes under one controlled setting (e.g. `smelt local_base A B`). Suite runs write the full analysis to `<out>/comparison_analysis.json`.
- **Results analysis (curves / NaN / bad cases / efficiency)**:
  - **Live curve monitoring**: instead of inspecting only the final number, every periodic validation reuses `analyze_training_curve`; status transitions (e.g. healthy→overfitting, divergence) immediately print recommendations and append a `{"step": "curve_monitor", ...}` record. Overfitting calls for regularization / more data / `--early-stop-patience`; both losses flat points to learning rate, optimizer, or a broken data pipeline.
  - **Systematic NaN/Inf diagnosis**: evidence is collected along the path data NaN/Inf → log(0)/division-by-zero/softmax overflow → fp16 mixed-precision overflow → gradient explosion / learning rate too large; the `nan_diagnosis` record compares the failing-step gradient norm with the recent median to separate gradient explosion from forward/loss-side root causes.
  - **Bad-case analysis**: the worst-loss samples are printed with context and categorized (ambiguous / systematic_confusion / label_noise_suspect / uncertain, i.e. blurry samples, model blind spots, suspect labels); category attribution is computed over **all wrong tokens** (`error_category_counts` / `error_category_shares`), not just the top-N, as the main source of new improvement ideas.
  - **Efficiency vs accuracy trade-off**: `--compare` summarizes parameters, FLOPs, inference latency and memory per cell and annotates cost ratios in each single-factor contrast; a significant gain costing ≥1.5×, or a regression that is also more expensive, is explicitly flagged (evaluate cost-effectiveness / strictly dominated), so "+0.5% at 10× compute" is not mistaken for real progress.

### Multi-seed and ablation suite examples

```bash
# RQ1: compare smelt / local_base / A / B under identical settings, 3 seeds each
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 --suite-modes smelt local_base A B \
  --steps 2000 --hypothesis "delayed merge improves val loss" --rq RQ1

# 2x2 full-factorial ablation: local_base/A x balance_coef=0/0.01
python experiment.py --data corpus.txt --out runs/ablation \
  --seeds 42 43 44 --suite-ablation \
  --steps 2000 --hypothesis "each design choice helps" --rq RQ1

# Read-only summary of finished runs: controlled-variable check + Mean ± Std + tests
python experiment.py --compare runs/rq1/smelt_seed42 runs/rq1/A_seed42 \
  runs/ablation/A_bal0_seed42 --compare-baseline smelt
```

## 📂 Output layout

Each run produces:
```text
runs/<exp_name>/
├── config.json      # full config, parameter count, seeds, data fingerprint, data audit, model-selection policy, code/env info, sanity-check result
├── metrics.jsonl    # per-line train/val metrics and final eval (loss, ppl, lr, grad_norm, ...); includes the single {"step": "test", ...} record when --evaluate-test is used
├── best.pt          # validation-selected weights used by final/test evaluation (atomic write)
└── last.pt          # weights, optimizer, LR scheduler and RNG states for resume (atomic write)
```

## 📝 Notes and disclaimer

- This project is a **research experiment foundation**; sparse dispatch is implemented with expert-level Python loops by default — intended for research prototypes.
- Scheme B does not yet implement incremental-decode KV cache; inference needs a redesigned caching strategy.
