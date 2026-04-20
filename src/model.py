from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .prunable_layer import PrunableLinear


class PruningNet(nn.Module):
    """Feed-forward MLP for CIFAR-10 where every layer is a PrunableLinear.

    Default architecture: 3072 -> 512 -> 256 -> 10. Using a plain MLP rather
    than a CNN because the problem asks for a feed-forward net and it makes the
    sparsity story cleaner — no conv filters to hide structure behind.
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, num_classes)
        self.layers = nn.ModuleList(
            PrunableLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x

    def prunable_layers(self) -> Iterable[PrunableLinear]:
        return (m for m in self.modules() if isinstance(m, PrunableLinear))

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of all gate values — the L1 sparsity penalty term."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + layer.get_gates().sum()
        return total

    def set_temperature(self, temp: float) -> None:
        for layer in self.prunable_layers():
            layer.set_temperature(temp)

    def sparsity_fraction(self, threshold: float = 1e-2) -> float:
        pruned, total = 0, 0
        for layer in self.prunable_layers():
            p, t = layer.sparsity(threshold)
            pruned += p
            total += t
        return pruned / total if total else 0.0

    def all_gates(self) -> torch.Tensor:
        """Concatenated gate values from all layers — used for histograms."""
        with torch.no_grad():
            return torch.cat(
                [layer.get_gates().flatten().cpu() for layer in self.prunable_layers()]
            )
