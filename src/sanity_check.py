"""Quick sanity check on sklearn Digits — no downloads, ~1 min on CPU.

The real experiment is python -m src.main (CIFAR-10). This exists to let you
verify the mechanism works end-to-end without waiting for a dataset download.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .model import PruningNet
from .prunable_layer import PrunableLinear
from . import training as tr
from .training import TrainConfig, train


OUT = Path("outputs") / "sanity_digits"
OUT.mkdir(parents=True, exist_ok=True)


class DigitsNet(PruningNet):
    """Small MLP for 8x8 sklearn Digits (64-dim input)."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([
            PrunableLinear(64, 128),
            PrunableLinear(128, 64),
            PrunableLinear(64, 10),
        ])
        self.dropout = nn.Identity()


def _digits_loaders(cfg: TrainConfig):
    d = load_digits()
    X = d.data.astype("float32") / 16.0
    y = d.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    to_ds = lambda a, b: TensorDataset(torch.tensor(a), torch.tensor(b, dtype=torch.long))
    return (
        DataLoader(to_ds(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(to_ds(X_te, y_te), batch_size=cfg.batch_size),
    )


def _plot_histogram(gates: torch.Tensor, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gates.numpy(), bins=60, color="#2a6df4", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Number of weights")
    ax.set_title(title)
    ax.axvline(1e-2, color="crimson", linestyle="--", linewidth=1, label="pruning threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    tr.get_loaders = _digits_loaders

    results = []
    for lam in [5e-4, 2e-3, 5e-3]:
        cfg = TrainConfig(epochs=40, lam=lam, device="cpu", batch_size=64, final_temp=0.3)
        model = DigitsNet()
        model, hist = train(cfg, model=model, verbose=False)

        gf = hist.gate_snapshots[-1]
        exact_zero = (gf == 0).float().mean().item() * 100

        results.append({
            "lam": lam,
            "test_acc": hist.test_acc[-1] * 100,
            "sparsity_1e_2": hist.sparsity[-1] * 100,
            "exact_zero": exact_zero,
        })

        _plot_histogram(
            gf,
            f"Digits · λ={lam} · acc {hist.test_acc[-1]*100:.1f}% · exact zeros {exact_zero:.1f}%",
            OUT / f"digits_gates_lam_{lam}.png",
        )
        print(f"lam={lam}: acc={hist.test_acc[-1]*100:.1f}%  "
              f"sparsity@1e-2={hist.sparsity[-1]*100:.1f}%  "
              f"exact_zeros={exact_zero:.1f}%")

    print("\n| Lambda | Test Acc | Sparsity@1e-2 | Exact zeros |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['lam']} | {r['test_acc']:.2f}% | "
              f"{r['sparsity_1e_2']:.2f}% | {r['exact_zero']:.2f}% |")


if __name__ == "__main__":
    main()
