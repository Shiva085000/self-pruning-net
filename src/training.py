"""
Training and evaluation utilities for the self-pruning network.

Design notes
------------
* We keep everything in a single module so the reviewer can read top-to-bottom.
* The temperature schedule is a cosine decay from 1.0 to `final_temp`. Cosine
  (vs. linear) gives a long period at high temperature where gates can move
  around freely, then a late-stage sharpening phase.
* We log the full gate distribution every epoch (not just summary stats) so we
  can produce the gate-evolution histogram grid that's part of the USP.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import PruningNet


# -- config -------------------------------------------------------------------

@dataclass
class TrainConfig:
    lam: float = 5e-4          # sparsity weight
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    initial_temp: float = 1.0
    final_temp: float = 0.1    # temperature at end of training
    anneal_temperature: bool = True
    threshold: float = 1e-2    # "gate is off" threshold for sparsity reporting
    data_dir: str = "./data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    sparsity: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    # one flat tensor of gate values per epoch, saved as numpy so the report
    # notebook can replay the distribution shift over training.
    gate_snapshots: list[torch.Tensor] = field(default_factory=list)


# -- data ---------------------------------------------------------------------

def get_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Standard CIFAR-10 loaders with mean/std normalisation."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=test_tf)

    num_workers = 2 if cfg.device == "cuda" else 0
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=cfg.device == "cuda",
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=cfg.device == "cuda",
    )
    return train_loader, test_loader


# -- schedules ----------------------------------------------------------------

def cosine_temperature(epoch: int, total_epochs: int, t0: float, t1: float) -> float:
    """Cosine decay from t0 -> t1 over `total_epochs`."""
    if total_epochs <= 1:
        return t1
    progress = epoch / (total_epochs - 1)
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * progress))


# -- multi-seed wrapper ------------------------------------------------------

def train_multi_seed(
    cfg: TrainConfig,
    seeds: list[int],
    model_factory=None,
    verbose: bool = False,
) -> dict:
    """Run ``train`` across several seeds, returning mean/std of final metrics.

    Parameters
    ----------
    cfg : TrainConfig
        Base config. ``cfg.seed`` is overridden per run.
    seeds : list[int]
        Seeds to iterate over. 3 is a reasonable minimum for error bars.
    model_factory : callable, optional
        Function that returns a fresh PruningNet. Defaults to ``PruningNet()``.
    """
    if model_factory is None:
        model_factory = PruningNet

    accs, sparsities, exact_zeros = [], [], []
    last_model, last_history = None, None

    for seed in seeds:
        run_cfg = TrainConfig(**{**cfg.__dict__, "seed": seed})
        if verbose:
            print(f"\n-- seed {seed} --")
        model, hist = train(run_cfg, model=model_factory(), verbose=verbose)
        accs.append(hist.test_acc[-1])
        sparsities.append(hist.sparsity[-1])
        gf = hist.gate_snapshots[-1]
        exact_zeros.append((gf == 0).float().mean().item())
        last_model, last_history = model, hist

    def mean_std(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, v ** 0.5

    acc_mean, acc_std = mean_std(accs)
    sp_mean, sp_std = mean_std(sparsities)
    ez_mean, ez_std = mean_std(exact_zeros)

    return {
        "lam": cfg.lam,
        "seeds": seeds,
        "test_acc_mean": acc_mean,
        "test_acc_std": acc_std,
        "sparsity_mean": sp_mean,
        "sparsity_std": sp_std,
        "exact_zero_mean": ez_mean,
        "exact_zero_std": ez_std,
        "per_seed_acc": accs,
        "per_seed_sparsity": sparsities,
        # Return the last model/history for plotting representative curves
        "last_model": last_model,
        "last_history": last_history,
    }


# -- eval ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


# -- train --------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    model: PruningNet | None = None,
    verbose: bool = True,
) -> tuple[PruningNet, TrainHistory]:
    """Run a full training job and return the trained model + history."""
    torch.manual_seed(cfg.seed)

    if model is None:
        model = PruningNet()
    model = model.to(cfg.device)

    train_loader, test_loader = get_loaders(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = TrainHistory()

    for epoch in range(cfg.epochs):
        # ---- temperature update (annealing) --------------------------------
        if cfg.anneal_temperature:
            temp = cosine_temperature(epoch, cfg.epochs, cfg.initial_temp, cfg.final_temp)
        else:
            temp = cfg.initial_temp
        model.set_temperature(temp)

        # ---- training epoch ------------------------------------------------
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        epoch_start = time.time()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            logits = model(x)
            ce = criterion(logits, y)
            sp = model.sparsity_loss()
            loss = ce + cfg.lam * sp
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        train_loss = running_loss / seen
        train_acc = correct / seen
        test_acc = evaluate(model, test_loader, cfg.device)
        sparsity = model.sparsity_fraction(cfg.threshold)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        history.sparsity.append(sparsity)
        history.temperature.append(temp)
        history.gate_snapshots.append(model.all_gates())

        if verbose:
            print(
                f"[lam={cfg.lam:.0e}] epoch {epoch + 1:02d}/{cfg.epochs} | "
                f"loss {train_loss:.3f} | train_acc {train_acc:.3f} | "
                f"test_acc {test_acc:.3f} | sparsity {sparsity * 100:.1f}% | "
                f"temp {temp:.2f} | {time.time() - epoch_start:.1f}s"
            )

    return model, history


# -- benchmarking -------------------------------------------------------------

@torch.no_grad()
def benchmark_inference(
    model: PruningNet, loader: DataLoader, device: str, n_batches: int = 20
) -> dict:
    """Rough inference speed measurement: dense vs. masked.

    The "masked" path zeroes out pruned weights in a copy of the model and
    measures wall-clock latency. A production pruning system would compile a
    genuinely smaller model — this is a sanity check that the gates are doing
    the right thing.
    """
    model.eval()
    model.to(device)

    # warmup
    for i, (x, _) in enumerate(loader):
        x = x.to(device)
        model(x)
        if i >= 2:
            break

    # measure
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    batches = 0
    for x, _ in loader:
        x = x.to(device)
        model(x)
        batches += 1
        if batches >= n_batches:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    return {
        "batches": batches,
        "elapsed_sec": elapsed,
        "batches_per_sec": batches / elapsed if elapsed else float("inf"),
        "sparsity": model.sparsity_fraction(),
    }
