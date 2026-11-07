"""Policy networks: an observation -> softmax over {void, slot 1..M}.

`PolicyNetwork` matches the original DeepRM network (one hidden layer of 20 ReLU
units over the flattened images); depth, width and activation are configurable.

`AttentionPolicy` reads the "features" observation instead:

1. one shared job encoder turns each visible job's (demand, duration) into a vector,
   so a job is scored the same way whichever slot it occupies;
2. a cluster encoder summarizes free capacity over the next T steps plus the backlog;
3. attention layers let each job (and the cluster summary) look at the other jobs;
4. each job is scored from its vector and the cluster summary; void is scored from
   the cluster summary alone. Empty slots share one learned score, so the action
   space and its semantics are unchanged from the image policy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from slurm_aware.deeprm.env import DeepRMEnv, FeatureLayout

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class PolicyModel(nn.Module):
    """Any network mapping a batch of flat observations to action logits."""

    def probabilities(self, observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self(torch.as_tensor(observations, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy().astype(np.float64)


class PolicyNetwork(PolicyModel):
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


def _mlp(width_in: int, width: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width_in, width), nn.ReLU(), nn.Linear(width, width_out))


class AttentionPolicy(PolicyModel):
    def __init__(
        self,
        layout: FeatureLayout,
        embed_dim: int = 32,
        heads: int = 4,
        layers: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads.")
        self.layout = layout
        job_inputs = layout.job_features - 1  # everything but the present flag
        self.job_encoder = _mlp(job_inputs, embed_dim, embed_dim)
        self.cluster_encoder = _mlp(layout.cluster_size + 1, embed_dim, embed_dim)
        # layers=0 keeps the shared encoders but drops attention (an ablation).
        self.attention = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    embed_dim, heads, 2 * embed_dim, dropout=0.0, batch_first=True
                ),
                layers,
                enable_nested_tensor=False,
            )
            if layers
            else None
        )
        self.job_score = _mlp(2 * embed_dim, embed_dim, 1)
        self.void_score = _mlp(embed_dim, embed_dim, 1)
        self.empty_slot_logit = nn.Parameter(torch.zeros(1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Logits of shape (batch, 1 + num_slots): void first, then each slot."""
        layout = self.layout
        batch = observations.shape[0]
        jobs = observations[:, : layout.jobs_size].reshape(
            batch, layout.num_slots, layout.job_features
        )
        present = jobs[..., 0] > 0.5
        job_tokens = self.job_encoder(jobs[..., 1:])
        cluster = self.cluster_encoder(observations[:, layout.jobs_size :])

        if self.attention is not None:
            tokens = torch.cat([cluster[:, None], job_tokens], dim=1)
            # The cluster token is never padded, so every query has something to attend to.
            ignore = torch.cat([present.new_zeros(batch, 1), ~present], dim=1)
            tokens = self.attention(tokens, src_key_padding_mask=ignore)
            cluster, job_tokens = tokens[:, 0], tokens[:, 1:]
        job_tokens = job_tokens.masked_fill(~present[..., None], 0.0)

        pairs = torch.cat([job_tokens, cluster[:, None].expand_as(job_tokens)], dim=-1)
        job_logits = self.job_score(pairs).squeeze(-1)
        job_logits = torch.where(present, job_logits, self.empty_slot_logit.expand_as(job_logits))
        return torch.cat([self.void_score(cluster), job_logits], dim=1)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row (inverse-CDF on a single uniform per row)."""
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((probs.shape[0], 1)) * cdf[:, -1:]
    return np.minimum((u > cdf).sum(axis=-1), probs.shape[1] - 1)


class NetworkPolicy:
    """Env-facing wrapper: greedy (argmax) for evaluation, or sampling with a seeded rng."""

    def __init__(
        self, network: PolicyModel, *, greedy: bool = True, seed: int | None = None
    ) -> None:
        self.network, self.greedy = network, greedy
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: DeepRMEnv) -> int:
        probs = self.network.probabilities(env.observation()[None, :])
        if self.greedy:
            return int(probs[0].argmax())
        return int(sample_actions(probs, self.rng)[0])
