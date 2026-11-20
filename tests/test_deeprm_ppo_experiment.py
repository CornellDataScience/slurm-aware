"""Unit tests for experiments/ppo/train.py.

Run from the repo root: uv run pytest tests/test_deeprm_ppo_experiment.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from experiments.ppo.train import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    collect_episode,
    gae_targets,
    ppo_update,
)


class TinyEnvironment:
    """Minimal Gym-style fixture that avoids importing a separate simulator."""

    observation_size = 4
    n_actions = 3

    def reset(self, jobs):
        self.time = 0
        self.done = False
        self.truncated = False
        return np.array([1, 0, 0, 0], dtype=np.float32), {}

    def step(self, action):
        self.time += 1
        self.done = self.time == 3
        return np.array([0, self.time, action, 0], dtype=np.float32), -1.0, self.done, False, {}

    def slowdowns(self, include_unfinished=False):
        return [2.0, 1.0]


def test_gae_without_bootstrap_matches_episode_returns_for_lambda_one():
    adv, returns = gae_targets([-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], 0.0, 1.0, 1.0)
    np.testing.assert_allclose(adv, [-6, -5, -3])
    np.testing.assert_allclose(returns, [-6, -5, -3])


def test_gae_bootstraps_on_truncation():
    _, returns = gae_targets([1.0], [0.0], 4.0, 0.9, 1.0)
    np.testing.assert_allclose(returns, [4.6])


def test_model_and_ppo_update():
    torch.manual_seed(0)
    model = ActorCritic(4, 3, 16)
    env = TinyEnvironment()
    ep = collect_episode(env, [], model, PPOConfig())
    assert len(ep.actions) == 3
    assert ep.raw_return == -3.0
    assert ep.mean_slowdown == 1.5
    assert not ep.truncated
    original = [p.detach().clone() for p in model.parameters()]
    result = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), [ep], PPOConfig())
    assert result["decisions"] == 3
    assert any(not torch.equal(old, new) for old, new in zip(original, model.parameters()))


def test_choose_action_within_bounds():
    model = ActorCritic(4, 3)
    action, log_prob, value = model.choose(np.zeros(4, dtype=np.float32))
    assert 0 <= action < 3
    assert np.isfinite(log_prob) and np.isfinite(value)
