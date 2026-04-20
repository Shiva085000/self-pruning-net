# Self-Pruning Neural Network — Tredence AI Engineering Case Study

A from-scratch implementation of a neural network that learns to prune itself
during training, using learnable per-weight sigmoid gates regularised toward
zero. Built for the Tredence Studio AI Agents Engineering internship case
study, November 2025.

**TL;DR** — every weight in the network has a companion learnable scalar, a
"gate". In the forward pass, each weight is multiplied by its gate (after a
stretched-sigmoid transform that keeps values in `[0, 1]` and produces genuine
hard zeros). The training loss adds an L1 penalty on the gates, so the
optimiser is rewarded for turning gates off. The result: most of the network
prunes itself away, and the surviving weights do the work.

## Results preview (sklearn Digits sanity check — real data, <1 min on CPU)

| Lambda | Test Accuracy | Sparsity (gate < 1e-2) | Exact zeros |
|---|---|---|---|
| 5e-4  | **93.1%** | 81.5% | 80.8% |
| 2e-3  | 91.1%     | 94.4% | 94.3% |
| 5e-3  | 68.6%     | 98.6% | 98.5% |

> CIFAR-10 numbers are filled in by `python -m src.main` — see `report.md`
> for the full analysis and discussion.

## What's in the repo

```
self_pruning_net/
├── src/
│   ├── prunable_layer.py    # PrunableLinear — gated linear layer (from scratch)
│   ├── model.py             # PruningNet — MLP stacking PrunableLinear layers
│   ├── training.py          # training loop, temperature anneal, evaluation
│   ├── main.py              # full CIFAR-10 experiment: λ sweep + ablation + plots
│   └── sanity_check.py      # fast real-data sanity run on sklearn Digits
├── outputs/                 # generated plots, results.json, model checkpoints
├── report.md                # analysis, tables, plots for the reviewer
├── requirements.txt
├── Dockerfile               # one-command reproducible environment
└── README.md                # this file
```

## Quick start

### Option A — native Python (fastest)
```bash
pip install -r requirements.txt

# 1-minute sanity check on sklearn Digits (no downloads needed)
python -m src.sanity_check

# Full CIFAR-10 experiment (~15 min on RTX 5050, ~2 hr on CPU)
python -m src.main

# Faster smoke test — 2 epochs, one lambda
python -m src.main --quick
```

### Option B — Docker (one command)
```bash
docker build -t self-pruning-net .
docker run --rm -v $(pwd)/outputs:/app/outputs self-pruning-net
```

## The three key design choices (why this is not a tutorial copy)

### 1. From-scratch `PrunableLinear` — not `torch.nn.utils.prune`
The built-in PyTorch pruning utility is a post-training mask. The JD asks
for pruning *during* training, so we subclass `nn.Module` and build the gating
mechanism into the forward pass. `weight`, `bias`, and `gate_scores` are all
registered as `nn.Parameter`, so Adam updates them together. Gradients flow
through both `weight` and `gate_scores` on every backward pass — which is what
makes this "end-to-end" pruning rather than a two-phase procedure.

### 2. Stretched-and-clamped sigmoid for genuine hard zeros
A plain `sigmoid` only asymptotes toward zero — gates get arbitrarily small
but never cross the `<1e-2` threshold the JD uses for the sparsity metric.
Following Louizos et al. 2018 ("Learning Sparse Networks through L₀
Regularization"), we compute:

```
gate = clamp(sigmoid(score / T) · (β − γ) + γ,  0,  1)
```

with `(γ, β) = (−0.3, 1.3)`. The stretch pushes part of the sigmoid output
below zero, where it's clamped to exactly zero. This makes the sparsity metric
move in practice, not just in the limit. With `(γ, β) = (0, 1)` the layer
reduces to the plain-sigmoid variant the JD sketches.

### 3. Cosine temperature annealing (1.0 → 0.3)
The sigmoid temperature is a non-trainable buffer that we decay across epochs.
Early on, gates sit near 0.5 and both CE and L1 have strong gradients, so the
network can freely decide which gates matter. Late in training the sigmoid
sharpens, committing each gate more decisively to 0 or to a stable active
value. This is the "annealing" trick from gradient-based discrete pruning
literature, and the ablation in `report.md` shows it's worth a few percentage
points of sparsity at equal accuracy.

## Reviewer reading order

1. `src/prunable_layer.py` — the mechanism, heavily commented.
2. `src/training.py` — the training loop with L1 sparsity term and annealing.
3. `report.md` — analysis, tables, and the sparsity-vs-accuracy discussion.
4. `src/main.py` — the driver tying it all together.

## Known caveats

- The sanity-check results use `sklearn.datasets.load_digits` (8×8 digits),
  not CIFAR-10. The CIFAR-10 experiment in `src/main.py` is the submission
  experiment; the Digits one exists so a reviewer can verify the mechanism
  works end-to-end in ~1 minute without downloading anything.
- The "surviving" gates cluster near ~0.1 rather than saturating at 1. The
  report discusses why this is an expected property of L1-on-sigmoid rather
  than a bug, and sketches how `L₀` or hard-concrete gates would tighten it.

## Author

Shiva (Gokul Shiva) — B.Tech AI, SRM Institute of Science and Technology, Chennai
GitHub: [Shiva085000](https://github.com/Shiva085000)
