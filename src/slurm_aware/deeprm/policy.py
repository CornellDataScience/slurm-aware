"""Policy network: flattened state images -> softmax over {void, slot 1..M}.

The default matches the original DeepRM network (one hidden layer of 20 ReLU
units); depth, width and activation are configurable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from slurm_aware.deeprm.env import DeepRMEnv

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (20,),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}.")
        layers: list[nn.Module] = []
        width = observation_size
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), ACTIVATIONS[activation]()]
            width = size
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Unnormalized log-probabilities (logits), shape (batch, n_actions)."""
        return self.net(observations)

    def probabilities(self, observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self(torch.as_tensor(observations, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy().astype(np.float64)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row (inverse-CDF on a single uniform per row)."""
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((probs.shape[0], 1)) * cdf[:, -1:]
    return np.minimum((u > cdf).sum(axis=-1), probs.shape[1] - 1)


class NetworkPolicy:
    """Env-facing wrapper: greedy (argmax) for evaluation, or sampling with a seeded rng."""

    def __init__(
        self, network: PolicyNetwork, *, greedy: bool = True, seed: int | None = None
    ) -> None:
        self.network, self.greedy = network, greedy
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: DeepRMEnv) -> int:
        probs = self.network.probabilities(env.observation()[None, :])
        if self.greedy:
            return int(probs[0].argmax())
        return int(sample_actions(probs, self.rng)[0])
