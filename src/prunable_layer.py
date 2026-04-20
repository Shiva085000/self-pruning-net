from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """Linear layer where each weight is multiplied by a learnable gate in [0, 1].

    gate_scores has the same shape as weight and is registered as nn.Parameter,
    so the optimizer updates gates and weights together. The stretched-sigmoid
    transform (Louizos et al. 2018) produces genuine hard zeros rather than
    just asymptotically small values.

    Args:
        gate_init: starting value for gate_scores. 0.0 means gates start at 0.5,
                   which gives the L1 penalty room to push them toward zero.
        stretch: (gamma, beta) for the stretched sigmoid. (-0.3, 1.3) is the
                 Louizos et al. value. Pass (0.0, 1.0) to get plain sigmoid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 0.0,
        stretch: tuple[float, float] = (-0.3, 1.3),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.gate_scores = nn.Parameter(torch.full_like(self.weight, gate_init))

        # buffers so they move with .to(device) but aren't trained by the optimizer
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("gamma", torch.tensor(float(stretch[0])))
        self.register_buffer("beta", torch.tensor(float(stretch[1])))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self) -> torch.Tensor:
        """Stretched-and-clamped sigmoid — produces exact zeros, not just small values.

        gate = clamp(sigmoid(score / T) * (beta - gamma) + gamma,  0,  1)

        Any score where the raw output falls below 0 gets clamped to exactly 0.
        Straight-through gradient from clamp keeps training signal flowing.
        """
        raw = torch.sigmoid(self.gate_scores / self.temperature)
        return torch.clamp(raw * (self.beta - self.gamma) + self.gamma, 0.0, 1.0)

    def set_temperature(self, temp: float) -> None:
        self.temperature.fill_(float(temp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # element-wise gate * weight; autograd routes gradients to both
        return F.linear(x, self.weight * self.get_gates(), self.bias)

    def sparsity(self, threshold: float = 1e-2) -> tuple[int, int]:
        """Returns (num_pruned, total_weights) at the given gate threshold."""
        with torch.no_grad():
            gates = self.get_gates()
            return (gates < threshold).sum().item(), gates.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
