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
- **Controlled comparisons and statistical reliability**: every run records a comparison protocol (optimizer, LR schedule, training steps, data augmentation, data fingerprint, early-stop/AMP/device numerics, ...); `--compare` verifies that everything except the declared experimental factors is identical and detects duplicate-seed pseudo-replication; `--seeds` runs multiple random seeds and reports Mean ± Std with significance tests; `--suite-ablation` runs the (1+N)×2 full-factorial ablation (covering schemes A and B by default) and checks coverage automatically.

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.x

```bash
pip install "torch>=2.2"
```

## 📦 Usage

### 1. Run structural self-tests
Before experimenting, run the built-in self-tests to verify structural correctness, branch independence and equivalence. The self-test also includes a bitwise same-seed double-run reproducibility check and negative-case validation of the gradient probe:
```bash
python experiment.py --self-test
```

### 2. Prepare data
The code defaults to a byte-level tokenizer for quick validation. Prepare a UTF-8 plain-text file (e.g., `corpus.txt`).

The pipeline is strictly leakage-safe: **split first, then process**. The corpus is cut chronologically into `train` (first 90%) / `val` (next 5%) / `test` (last 5%): the three splits are contiguous, non-overlapping, and never randomly shuffled (time-series data must train on the past and validate/test on the future). Batch-level shuffling during training is provided by random window starts and never crosses split boundaries. The byte pipeline performs no standardization/normalization/imputation; if you extend it to continuous features, every statistic (mean, variance, min/max, ...) must be fitted on `train` only and applied to val/test via `transform`.

`val` is used for periodic validation, hyperparameter selection, early stopping and model selection; `test` stays sealed for the whole training run and is evaluated exactly once after training, only when `--evaluate-test` is passed. Re-evaluation within the same run is refused. Fractions are configurable with `--val-fraction` / `--test-fraction`. Before training, a data audit (token range, text-integrity/label-noise checks for bytes that cannot be UTF-8 decoded and for NUL bytes, duplicate windows, train↔val distribution shift) runs automatically and is stored in the `data_audit` field of `config.json`; unsealing the test set appends a test-inclusive audit.

**Grouped data must stay on one side**: if the corpus is a concatenation of groups (the same document / user / object producing multiple records), pass `--group-separator` (e.g., `--group-separator '<|doc|>'` or `--group-separator '\n\n'`; `\n`, `\xHH`, `\uXXXX` escapes are supported). Split positions are snapped to group boundaries, so no group is ever cut across two splits and train/val/test never cross. The audit records the number of whole groups per split. When snapping cannot satisfy the minimum length of every split (e.g., a single oversized group), the script raises an explicit error instead of silently splitting a group.

**Training-set deduplication**: `--deduplicate-train-groups` (requires a
group separator) removes whole training groups whose content is identical
to a val/test group or to an earlier training group; val/test bytes and
split positions are left untouched. Removed content enters neither neural
training windows nor the unigram / n-gram statistical baselines or the
distribution audit. `data_audit` records `effective_bytes` /
`effective_windows`, and `config.data` records the actual split fractions
and cleaning statistics. The training-time audit checks the test fraction
from split bounds only (without reading test content); before unsealing,
the fixed test windows are independently verified for bit-exact input/target
alignment.

**Leakage is a hard gate by default**: if the audit finds cross-split duplicate windows or duplicate groups, training refuses to start; when unsealing the test set, the same findings in test refuse the unsealing (no `test` record is written). Only an explicit `--allow-data-leakage` overrides the gate, and the override is recorded in `config.json` and the comparison protocol, so a leaky run cannot silently pass as leak-free evidence. The window signature scan is strided (equally spaced sampling) once the window count exceeds the cap; `data_audit.leakage_scan_coverage` records the sampled/total window counts, stride and coverage per split, and both the audit and the gate spell out the residual risk when stride>1: "no cross-split duplicate windows found" inside the sampled subset is not "none exist". The content-defined chunk fingerprint scan is exhaustive and catches misaligned duplicates that the stride missed. Label noise (out-of-range tokens, UTF-8/NUL corruption above threshold) is controlled by `--strict-data-audit`: with it, such findings also abort training/unsealing; by default they are warnings so you can inspect the data before deciding how to clean it.

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

After the first backward pass a gradient probe runs first: the loss must be
attached to the autograd graph, every gradient must be finite, at least one
parameter must receive a non-zero gradient, and no non-expert parameter may be
missing a gradient (expert parameters may legitimately be missing/zero when no
token is routed to them in the batch). A probe failure stops the check
immediately with troubleshooting hints instead of wasting the remaining steps;
the probe result is recorded as `sanity_check.grad_probe` in `config.json` and
in `metrics.jsonl`.

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

The routing diagnostics behind RQ3/RQ4 (round-2 Jaccard / slot reuse / branch self-reuse, round-1/2 load and router entropy, dead experts, branch cosine) are stored per run in `config.json` under `routing_analysis`; `--compare` prints them per run and aggregates them across seeds (Mean ± Std), and they are part of the comparable metric set (`--compare-metric routing_jaccard_branch`, etc.). Routing conclusions therefore carry across-seed statistics and significance tests instead of relying on a single run's qualitative description.

### Controlled-comparison conventions and reproducibility

- **Scheme B routing and merge weights**: scheme B uses $k_1$ throughout the first-round intermediate region and $k_2$ only in the full second-round region; merging uses fixed weights from round-1 routing with Top-$k_1$ normalization, never recomputed from final branch states. Analyses/ablations should state this explicitly.
- **Compute accounting**: after round-1 branch creation, scheme B executes downstream attention and MoE at $k_1$× branch count; parameter-matched is not FLOPs-matched. When answering RQ1/RQ2, report both step-aligned and token/FLOPs-aligned comparisons to avoid misattributing extra compute to the routing mechanism.
- **Residual-scaling convention**: `residual_scale` applies globally to every residual update in all regions (inside and outside the looped span). The SMELT paper scales residual updates within the looped span by $1/r$ (1/2 for two visits); this base does not implement region-specific $1/r$ scaling. Keep `residual_scale` equal across modes; a global 0.5 is not equivalent to the paper's convention.
- **Load-balancing scale**: $f_i$ in the auxiliary loss is normalized by token count, not assignment count; under uniform routing $E \sum_i f_i P_i \approx k$, so balancing strength grows linearly with routing width. When comparing across $k_1/k_2/\text{top-}k$, scale the coefficient accordingly (keep `balance_coef × k` constant); otherwise routing width is confounded with balancing strength. `local_base` uses width $\min(n_\text{experts}, k_1 k_2)$ in its second round to match compute; the implementation scales that round's aux back to the per-branch A/B width $k_2$ by $k_2/\min(n_\text{experts}, k_1 k_2)$, and records `balance_width_normalization` in `comparison_protocol` to distinguish new from legacy runs. `--compare` also checks `balance_coef × effective routing width` and emits a `[balance-scale]` warning when widths differ without an offsetting strength change.
- **Validation/test protocol**: each evaluation uses fixed windows evenly spaced across the corresponding evaluation split, independent of any RNG, bit-aligned across models and resumed runs. Every new best validation loss is snapshotted to `best.pt`, and both the `final` record and the single test evaluation run on this validation-selected model (the `model_selection` field records the policy; when `best.pt` is unavailable the script reports it and falls back to the last-step parameters). The test split stays sealed for the entire training run and is evaluated exactly once when `--evaluate-test` is passed (`{"step": "test", ...}`); repeating the flag is refused, so the test set cannot silently become a validation set. The `final` and `test` records additionally report bootstrap 95% confidence intervals over the fixed evaluation windows (loss / ppl / token accuracy / macro F1; `--eval-bootstrap N` sets the number of resamples, default 500, 0 disables it), separating evaluation-window sampling noise from training-seed variance. Because using the same validation split for both checkpoint selection and evaluation induces winner's curse, `--compare` derives `<metric>_at_common_step` unbiased versions of every periodic validation metric (loss / ppl / bpb / accuracy / micro, macro, weighted F1 / precision / recall), usable directly as `--compare-metric`. All results are appended to `metrics.jsonl`.
- **Determinism**: the training entry point fixes Python / NumPy / PyTorch (CPU+CUDA) / cuDNN seeds; data sampling is driven by an explicit `torch.Generator` whose seed and state are checkpointed; the Python / torch / CUDA / NumPy RNG states are all stored in `last.pt` and restored bit-for-bit on `--resume`; `CUBLAS_WORKSPACE_CONFIG` is set and PyTorch is asked to prefer deterministic kernels (a warning is emitted if a CUDA kernel lacks a deterministic implementation). `PYTHONHASHSEED` can only be fixed before interpreter start, so a one-time hint is printed when it was not set on the command line; prefix `PYTHONHASHSEED=<seed>` for strict bitwise reproducibility. Small cross-hardware numeric differences from reduction order may remain; on the same machine, same-seed runs should be bitwise identical (`--self-test` double-checks this).
- **Experiment tracking**: each run automatically records the code revision (git commit / dirty flag), environment (Python / torch / CUDA / device), full hyperparameters and seeds, data fingerprint (SHA-256, byte count), training curves and final results in `config.json` and `metrics.jsonl` — no manual bookkeeping required.
- **Controlled-variable checking**: `config.json` stores a `comparison_protocol` field (optimizer AdamW with betas/eps, LR and schedule, warmup, total steps, weight decay, gradient clipping, batch size, data augmentation, data fingerprint, model architecture, evaluation protocol and model-selection policy — new runs select on validation, legacy runs without a record are marked as last-step) as well as **effective training length and numerics** (early-stop tolerances `early_stop_patience`/`early_stop_min_delta`, AMP dtype, device type, gradient checkpointing, `torch.compile`). `--compare` requires all of these to match exactly unless declared as a factor via `--compare-factors` (default `mode,k1,k2,balance_coef,seed,dense_fnn_hidden`); a run that silently changed optimizer, LR, step count, AMP/device, or the `final` measurement basis is reported as a violation, so tuning gains cannot be misattributed to a method.
- **Statistical reliability**: `--seeds 42 43 44` trains the same configuration for multiple seeds and reports Mean ± Std (sample std) per experimental cell, together with a Student-t 95% confidence interval for the cell mean itself (`<metric>_ci95_low/high` and the `ci95` block; std only describes seed-to-seed spread, and no interval is emitted with fewer than two independent seeds). The `n(seed)` column of the aggregate table prints `4(3)` whenever the independent seed count is below the run count, exposing pseudo-replication or deterministic folding instead of inflating the apparent sample size. Shared seed sets use a paired t-test, otherwise Welch's t-test; p-values and the |Δ|/pooled-std noise ratio are shown. Cells with fewer than 3 seeds are flagged as underpowered, and an apparent +0.2% gain with 0.5% seed variance is not treated as real. **Non-significance is not equivalence**: every tested contrast also reports the minimum detectable effect at 80% power (MDE, absolute and relative to the baseline mean), the achieved power at the observed effect (`power_at_effect`), and the approximate number of seeds needed to detect that effect given the observed seed-to-seed variance; equivalence is tested with TOST against a pre-declared margin (`--equivalence-margin`, relative to the baseline mean, default 1%). TOST p-values share the significance family correction (Holm, FWER): "no material difference" may only be claimed when the corrected test passes; when a contrast is neither significant nor equivalent the conclusion is "inconclusive" (the display prints the MDE and the required seed count, separating "effect too small" from "too few seeds"). Every contrast that fails TOST additionally reports the smallest symmetric margin that would support equivalence at the current precision (`equivalence_margin_needed` = |Δ̂| + t·se) together with the reason it failed: `effect_exceeds_margin` when the point estimate is already outside the declared margin (more seeds would only make non-equivalence more certain), and `insufficient_precision` when the point estimate is inside but the noise half-width pushes the interval out (then the approximate `equivalence_seeds_to_achieve` is reported as well); the report flags these with an explicit `[warn]` — a failed TOST must not be read as "the two differ", and the fix is more seeds rather than a wider margin. **Power follows the family correction**: MDE, power at the observed effect, and required seed counts are computed at the uncorrected α=0.05 while decisions use Holm-corrected p-values, so the report also computes the corresponding conservative quantities at the Holm first-step threshold α/m (`significance_family_size`, `alpha_bonferroni`, `mde_80_bonferroni`, `seeds_to_detect_bonferroni`, `equivalence_margin_needed_bonferroni`), prints a `[power]` line naming the family size and α/m, and seed planning or margin choice should follow the conservative figures rather than the uncorrected ones. The margin must be declared in advance — widening it after seeing the data can manufacture equivalence. The same TOST protocol covers full-factorial main effects, interactions, and simple effects: the margin is relative to the control-cell mean(s) being contrasted, and TOST p-values are Holm/BH-corrected within the effect family; only corrected equivalence supports a claim that a component has no material effect. **Pseudo-replication detection**: the same seed appearing multiple times inside one cell (e.g. copied run directories) is not independent replication — duplicates are named, the paired test is disabled and Welch is used instead. Key suites (`--suite-modes`/`--suite-ablation`/`--suite-baseline-tuning`) refuse to run with fewer than 3 seeds; smoke/debug runs must pass `--allow-underpowered` explicitly, and the waiver is recorded with the seed plan in the report. **Differential-missingness detection**: when a run lacks the comparison metric (missing curve, metric not computed), the contrast is restricted to complete cases; if the two arms lose different numbers of runs the report names them explicitly (`n_missing_baseline`/`n_missing_variant`/`differential_missingness`), because the apparent difference may come from which runs survived rather than from the method. **Distribution-free paired test**: paired contrasts also run a sign-flip randomization test (exact enumeration for n≤12, fixed-seed Monte Carlo for n>12) that does not assume normal differences; if the t-test and the randomization test disagree after the same Holm correction, the report says so explicitly, and `significant_robust` requires both to be significant. The minimum attainable exact p-value (2/2^n) is recorded, and an n too small to reach 0.05 is flagged so non-significance is not read as no effect. **Leave-one-out robustness**: the effect is recomputed with each seed omitted in turn; a sign flip or collapse flags a conclusion potentially driven by a single seed.
- **Baseline adequacy and hypothesis registry**: besides deltas, `--compare-baseline MODE` checks that the baseline is worth beating: on the primary metric it must beat the context-free unigram reference (each run's `reference_baselines`; PR-AUC's constant-predictor reference is the mean prevalence over eligible classes and ROC-AUC's is 0.5, so the straw-man guard keeps working after switching the comparison metric to PR-AUC), must not be worse than the tuned byte-level n-gram statistical baseline, should have ≥3 seeds, and must carry tuning evidence (a `baseline_tuning` provenance record in `config.json`; the statistical baseline counts as tuned because it is grid-searched on val); otherwise it is flagged `[straw-man]` (beating an untuned weak baseline proves nothing about a complex model — tune the baseline with `--suite-baseline-tuning` first). A tuning metric that does not match the comparison metric no longer prints an unconditional OK; instead it explicitly warns that the baseline is not proven to be tuned on the current metric. The analysis also lists the `--hypothesis`/`--rq` recorded per mode (runs without one, whose hypothesis names no decision metric such as val loss / macro F1 / PR-AUC / recall, or whose named metric is not verifiable by the current pipeline — e.g. routing metrics without a `routing_analysis` record — are named; the statistical baseline is exempt; a single training run also warns at start-up and stores `hypothesis_metric_mentions` in `config.json`) and warns on class imbalance via `val_majority_accuracy`: once the majority class dominates, accuracy/micro is misleading; at ≥99.9% (≈1:1000) imbalance is tiered separately — PR-AUC must be read as lift over its chance level (≈ mean prevalence, recorded as `val_pr_auc_macro_chance`), while ROC-AUC keeps a chance level of 0.5 and looks overly optimistic. All checks are persisted to `comparison_analysis.json`. If the ngram runs and neural runs differ in data fingerprint, split fractions, evaluation-window count, batch size or data augmentation, a cross-family `[violation]` is reported and the "baseline beats ngram" judgment is skipped.
- **Baseline tuning protocol**: `--suite-baseline-tuning` (with `--seeds`) runs a pre-registered learning-rate grid search for the simple baseline (`--baseline-tuning-lrs`, default `1e-4 3e-4 1e-3`; `--baseline-tuning-mode` selects the mode). Candidates may differ only in LR (steps/batch/data/architecture/schedule are inherited from the same command line). The winner is picked on val by Mean ± Std: the best-mean candidate is identified first, and every candidate whose gap to it is smaller than the seed noise (the larger of the two stds) is treated as indistinguishable; the conservative winner is then the smallest LR in that tied set, and the tie set and selection mode (`winner_selection`/`tied_lrs`) are recorded verbatim. If the winner sits on the grid boundary (smallest/largest candidate), the report warns that the optimum may lie outside the grid and recommends extending `--baseline-tuning-lrs` and re-running. The report is written to `<out>/baseline_tuning.json`. A later baseline training run passes `--baseline-tuning-from <report>`: mode / winner LR / steps / data fingerprint and the **full training protocol** (warmup, schedule, weight decay, gradient clipping, data splits, eval windows, …) are verified item by item (old reports lacking protocol fields are rejected instead of passing silently), and the provenance is stored in `config.json` as `baseline_tuning` — the only evidence that the baseline was actually tuned. The test set stays sealed throughout.
- **Ablation completeness**: `--suite-ablation` runs the full factorial of delayed merge × balance loss (`local_base` and each delayed-merge scheme × `balance_coef`=0/current); by default it covers **all** schemes proposed by the project (A and B, (1+N)×2 = 3×2 = six cells), because every innovation must be shown to help on its own — running A alone cannot support claims about B. The coverage check lists missing cells, names structural confounds (inconsistent k1/k2/top_k/residual_scale or balance strength across cells) and pseudo-replication; declare the schemes a round must cover with `--ablation-modes A B`. `--suite-modes` runs multiple modes under one controlled setting (e.g. `smelt local_base A B`). Suite runs write the full analysis to `<out>/comparison_analysis.json`.
- **Results analysis (curves / NaN / bad cases / efficiency)**:
  - **Live curve monitoring**: instead of inspecting only the final number, every periodic validation reuses `analyze_training_curve`; status transitions (e.g. healthy→overfitting, divergence) immediately print recommendations and append a `{"step": "curve_monitor", ...}` record. Overfitting calls for regularization / more data / `--early-stop-patience`; both losses flat points to learning rate, optimizer, or a broken data pipeline. When validation loss has not yet regressed but the generalization gap keeps widening, an early-warning (`gap_widening_warning`) recommends early stopping before the regression shows up.
  - **Systematic NaN/Inf diagnosis**: evidence is collected along the path data NaN/Inf → log(0)/division-by-zero/softmax overflow → fp16 mixed-precision overflow → gradient explosion / learning rate too large; the `nan_diagnosis` record compares the failing-step gradient norm with the recent median to separate gradient explosion from forward/loss-side root causes.
  - **Bad-case analysis**: statistics are streamed over **all** evaluated tokens on the same fixed windows as periodic validation (about 10 bytes of compact scalars per token, no random subsampling, token accuracy identical to `evaluate`); top-N worst samples are maintained exactly via per-batch top-N plus cross-batch truncation and categorized (ambiguous / systematic_confusion / label_noise_suspect / uncertain, i.e. blurry samples, model blind spots, suspect labels). Category attribution over all wrong tokens (`error_category_counts` / `error_category_shares`), not just the top-N, is the main source of new improvement ideas.
  - **Efficiency vs accuracy trade-off**: `--compare` summarizes parameters, FLOPs, inference latency, training step time (forward+backward) and memory per cell and annotates cost ratios in each single-factor contrast; a significant gain costing ≥1.5×, or a regression that is also more expensive, is explicitly flagged (evaluate cost-effectiveness / strictly dominated), so "+0.5% at 10× compute" is not mistaken for real progress. Every efficiency field also carries a Student-t 95% confidence interval for its mean (`ci95_low`/`ci95_high`, omitted below n=2), and the `[efficiency-ci]` line reports the intervals for inference latency and training step time: an interval as wide as the mean itself means the efficiency difference may be pure measurement noise.

### Multi-seed and ablation suite examples

```bash
# RQ1: compare smelt / local_base / A / B under identical settings, 3 seeds each
python experiment.py --data corpus.txt --out runs/rq1 \
  --seeds 42 43 44 --suite-modes smelt local_base A B \
  --steps 2000 --hypothesis "delayed merge improves val loss" --rq RQ1

# 3x2 full-factorial ablation: local_base/A/B x balance_coef=0/0.01
# (all proposed schemes by default; declare a subset with --ablation-modes)
python experiment.py --data corpus.txt --out runs/ablation \
  --seeds 42 43 44 --suite-ablation --ablation-modes A B \
  --steps 2000 --hypothesis "each design choice helps" --rq RQ1

# Tune the simple baseline first: LR grid (selected on val), report in runs/tuning;
# --baseline-tuning-mode must be the mode you will later declare as the baseline
python experiment.py --data corpus.txt --out runs/tuning \
  --seeds 42 43 44 --suite-baseline-tuning --baseline-tuning-mode dense \
  --steps 2000 --hypothesis "tune the baseline so it is worth beating" --rq RQ1

# Train the final baseline at the winner LR and record the provenance;
# replace --lr with the winner lr from runs/tuning/baseline_tuning.json
python experiment.py --data corpus.txt --out runs/baseline_dense \
  --mode dense --lr 3e-4 \
  --baseline-tuning-from runs/tuning/baseline_tuning.json \
  --steps 2000 --hypothesis "tuned dense baseline performance" --rq RQ1

# Read-only summary of finished runs: controlled-variable check + Mean ± Std + tests.
# --compare-baseline must point at a mode that was actually tuned (here dense,
# with runs/baseline_dense carrying the baseline_tuning provenance); to declare
# smelt as the baseline, tune smelt with --suite-baseline-tuning and record its
# provenance the same way, or include an ngram run from --statistical-baseline
# in the comparison set.
python experiment.py --compare runs/rq1/smelt_seed42 runs/rq1/A_seed42 \
  runs/ablation/A_bal0_seed42 runs/baseline_dense --compare-baseline dense
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
