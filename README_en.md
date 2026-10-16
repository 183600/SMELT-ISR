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
The code defaults to a byte-level tokenizer for quick validation. Prepare a UTF-8 plain-text file (e.g., `corpus.txt`); the last 5% is automatically used as the validation set.
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

### 4. Resume training
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
- **Validation protocol**: each evaluation uses fixed windows evenly spaced across the validation set, independent of any RNG, bit-aligned across models and resumed runs; final results are appended to `metrics.jsonl`.
- **Determinism**: the training entry point asks PyTorch to prefer deterministic kernels; a warning is emitted if a CUDA kernel lacks a deterministic implementation. Small cross-device numeric differences from reduction order may remain.

## 📂 Output layout

Each run produces:
```text
runs/<exp_name>/
├── config.json      # full config and parameter count
├── metrics.jsonl    # per-line train/val metrics and final eval (loss, ppl, lr, grad_norm, ...)
└── last.pt          # weights, optimizer, LR scheduler and RNG states (atomic write)
```

## 📝 Notes and disclaimer

- This project is a **research experiment foundation**; sparse dispatch is implemented with expert-level Python loops by default — intended for research prototypes.
- Scheme B does not yet implement incremental-decode KV cache; inference needs a redesigned caching strategy.
