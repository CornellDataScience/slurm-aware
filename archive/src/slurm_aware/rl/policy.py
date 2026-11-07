"""Gaussian policy over log-scale priority weights, and the action-to-weights mapping.

The network outputs a mean per weight; a learned, state-independent log-std sets
exploration. An action `z` lives in log10 space:

- integer weights (age, fairshare, job_size, qos): `round(10 ** min(z, 9))`, or 0 if z <= 0
- signed TRES weights: `sign(z) * 10 ** min(|z|, 9)`, or 0 if z == 0

Clipping happens only when converting to weights; log-probabilities use the raw `z`.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal

from slurm_aware.simulator import PriorityWeights

WEIGHT_NAMES = ("age", "fairshare", "job_size", "qos", "tres_cpu", "tres_memory", "tres_gpu")
MAX_LOG10_WEIGHT = 9.0
LOG_STD_BOUNDS = (-3.0, 1.0)


def is_tres(name: str) -> bool:
    return name.startswith("tres_")


def weights_to_action(weights: PriorityWeights) -> list[float]:
    """Inverse of `action_to_weights` for in-scope weights (zero maps to z = 0)."""
    action = []
    for name in WEIGHT_NAMES:
        value = float(getattr(weights, name))
        magnitude = math.log10(abs(value)) if abs(value) >= 1 else 0.0
        action.append(math.copysign(magnitude, value) if is_tres(name) else magnitude)
    return action


def action_to_weights(action: Sequence[float]) -> PriorityWeights:
    """Out-of-scope weights (association, partition) stay zero: they are constant here."""
    if len(action) != len(WEIGHT_NAMES):
        raise ValueError(f"Expected {len(WEIGHT_NAMES)} action values, got {len(action)}.")
    values = {}
    for name, z in zip(WEIGHT_NAMES, map(float, action)):
        if not math.isfinite(z):
            raise ValueError(f"Non-finite action for {name}.")
        if is_tres(name):
            values[name] = math.copysign(10 ** min(abs(z), MAX_LOG10_WEIGHT), z) if z else 0.0
        else:
            # z <= 0 switches the factor off, so the policy can express a zero weight.
            values[name] = round(10 ** min(z, MAX_LOG10_WEIGHT)) if z > 0 else 0
    return PriorityWeights(association=0, partition=0, **values)


def weights_dict(weights: PriorityWeights) -> dict[str, float]:
    return asdict(weights)


class GaussianPolicy(nn.Module):
    """MLP from normalized trace features to a diagonal Gaussian over weight actions."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        initial_mean: Sequence[float] | None = None,
        initial_std: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.body = nn.Sequential(*layers)
        self.mean_head = nn.Linear(width, len(WEIGHT_NAMES))
        # Start every context at the same mean: tiny output weights, bias = initial mean.
        nn.init.normal_(self.mean_head.weight, std=0.01)
        initial = torch.tensor(
            initial_mean if initial_mean is not None else weights_to_action(PriorityWeights()),
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.mean_head.bias.copy_(initial)
        self.log_std = nn.Parameter(torch.full((len(WEIGHT_NAMES),), math.log(initial_std)))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(self.body(features))
        log_std = self.log_std.clamp(*LOG_STD_BOUNDS).expand_as(mean)
        return mean, log_std

    def distribution(self, features: torch.Tensor) -> Normal:
        mean, log_std = self(features)
        return Normal(mean, log_std.exp())
