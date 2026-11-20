"""PPO experiment on the existing event-driven DeepRMEnv.

Drop into ``experiments/`` in CornellDataScience/slurm-aware and run from the
repository root after ``uv sync --extra deeprm``.

This is deliberately an experiment, not a replacement for REINFORCE. The
included seeded synthetic jobset generator is a smoke-test workload; for a
publishable comparison, use the same generator/splits as the repo's existing
REINFORCE training and evaluation pipelines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical



def available_action_mask(env):
    mask = np.ones(env.n_actions, dtype=bool)
    if hasattr(env, "fitting_slots"):
        mask[:] = False
        mask[0] = True
        mask[env.fitting_slots()] = True
    return mask

def mask_logits(logits, mask):
    if logits.shape != mask.shape or not torch.all(mask.any(dim=-1)):
        raise ValueError("Invalid action mask")
    return logits.masked_fill(
        ~mask, torch.finfo(logits.dtype).min
    )

class ActorCritic(nn.Module):
    """Shared MLP features with separate policy logits and state value heads."""

    def __init__(self, observation_size: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def choose(self, observation: np.ndarray, deterministic: bool = False, action_mask=None) -> tuple[int, float, float]:
        obs = torch.as_tensor(np.asarray(observation), dtype=torch.float32).unsqueeze(0)
        logits, value = self(obs)
        if action_mask is not None:
            mask = torch.as_tensor(
                action_mask, dtype=torch.bool
            ).unsqueeze(0)
            logits = mask_logits(logits, mask)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


@dataclass
class PPOConfig:
    gamma: float = 1.0  # Matches total slowdown objective across completed episodes.
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 128
    max_steps_per_episode: int = 2500


@dataclass
class Episode:
    observations: list[np.ndarray]
    action_masks: list[np.ndarray]
    actions: list[int]
    old_log_probs: list[float]
    rewards: list[float]
    values: list[float]
    advantages: np.ndarray
    returns: np.ndarray
    raw_return: float
    mean_slowdown: float
    truncated: bool


def gae_targets(
    rewards: list[float],
    values: list[float],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE on one episode, bootstrapping if it is truncated."""
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def collect_episode(env, jobs, model: ActorCritic, cfg: PPOConfig) -> Episode:
    """Collect one on-policy trajectory using the existing env.reset/step API."""
    obs, _ = env.reset(jobs)
    observations, action_masks, actions, log_probs, values, rewards = [], [], [], [], [], []
    terminated, truncated = bool(env.done), False
    for _ in range(cfg.max_steps_per_episode):
        if terminated or truncated:
            break
        action_mask = available_action_mask(env)
        action, log_prob, value = model.choose(
            obs, action_mask=action_mask
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        action_masks.append(action_mask.copy())
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(float(reward))
        obs = next_obs
    if not terminated and not truncated:
        truncated = True  # Local safety cap; bootstrap from the final observation.
    bootstrap = 0.0 if terminated else model.choose(
        obs, deterministic=True,
        action_mask=available_action_mask(env)
    )[2]
    advantages, targets = gae_targets(rewards, values, bootstrap, cfg.gamma, cfg.gae_lambda)
    slowdowns = env.slowdowns(include_unfinished=True)
    return Episode(
        observations, action_masks, actions, log_probs, rewards, values, advantages, targets,
        float(sum(rewards)), float(np.mean(slowdowns)) if slowdowns else 0.0, truncated,
    )


def ppo_update(model: ActorCritic, optimizer, episodes: list[Episode], cfg: PPOConfig) -> dict:
    """Update on trajectories generated by the *previous* actor parameters."""
    episodes = [episode for episode in episodes if episode.actions]
    if not episodes:
        raise ValueError("No decisions were collected")
    observations = torch.as_tensor(np.stack([x for ep in episodes for x in ep.observations]))
    action_masks = torch.as_tensor(
        np.stack([m for ep in episodes for m in ep.action_masks]),
        dtype=torch.bool
    )
    actions = torch.as_tensor(
        [x for ep in episodes for x in ep.actions],
        dtype=torch.int64
    )
    old_log_probs = torch.as_tensor([x for ep in episodes for x in ep.old_log_probs])
    advantages = torch.as_tensor(np.concatenate([ep.advantages for ep in episodes]))
    value_targets = torch.as_tensor(np.concatenate([ep.returns for ep in episodes]))
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    stats = []
    for _ in range(cfg.update_epochs):
        order = torch.randperm(len(actions))
        for inds in order.split(cfg.minibatch_size):
            logits, values = model(observations[inds])
            logits = mask_logits(logits, action_masks[inds])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[inds])
            ratios = (new_log_probs - old_log_probs[inds]).exp()
            gain = ratios * advantages[inds]
            clipped_gain = ratios.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages[inds]
            policy_loss = -torch.minimum(gain, clipped_gain).mean()
            value_loss = (values - value_targets[inds]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            stats.append((float(policy_loss.item()), float(value_loss.item()), float(entropy.item())))
    means = np.mean(stats, axis=0)
    return {"policy_loss": float(means[0]), "value_loss": float(means[1]), "entropy": float(means[2]), "decisions": int(len(actions))}


def make_jobset(seed: int, spec, *, horizon: int, lookahead: int, rate: float):
    """Small reproducible synthetic workload for an initial PPO smoke test.

    IMPORTANT: Not the repo's JobTraceGenerator. Use the repo generator for the
    actual head-to-head experiment against REINFORCE.
    """
    from slurm_aware.deeprm.jobs import Job

    rng = np.random.default_rng(seed)
    jobs = []
    for time in range(horizon):
        for _ in range(int(rng.poisson(rate))):
            duration = int(np.clip(round(rng.lognormal(mean=1.1, sigma=0.7)), 1, lookahead))
            demands = tuple(int(rng.integers(0, max(2, c // 2 + 1))) for c in spec.capacities)
            if not any(demands):
                demands = (1,) + demands[1:]
            jobs.append(Job(job_id=len(jobs), arrival=time, duration=duration, demand=demands))
    if not jobs:
        jobs.append(Job(job_id=0, arrival=0, duration=1, demand=(1,) + (0,) * (spec.n_resources - 1)))
    return jobs


def evaluate(model: ActorCritic, env, jobsets: list) -> dict:
    """Deterministic policy on held-out jobsets (same ones for all algorithms)."""
    model.eval()
    results = []
    for jobs in jobsets:
        obs, _ = env.reset(jobs)
        reward_total, steps = 0.0, 0
        while not env.done and steps < 5000:
            action = model.choose(
                obs, deterministic=True,
                action_mask=available_action_mask(env)
            )[0]
            obs, reward, _, _, _ = env.step(action)
            reward_total += float(reward)
            steps += 1
        results.append((float(np.mean(env.slowdowns(include_unfinished=True))), reward_total, int(env.truncated)))
    model.train()
    return {
        "validation_slowdown": float(np.mean([x[0] for x in results])),
        "validation_return": float(np.mean([x[1] for x in results])),
        "validation_truncated": int(sum(x[2] for x in results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental PPO training for the existing DeepRM simulator")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--arrival-horizon", type=int, default=50)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/deeprm-ppo"))
    args = parser.parse_args()
    from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig
    from slurm_aware.deeprm.jobs import ClusterSpec

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    spec, env_cfg, cfg = ClusterSpec(), EnvConfig(), PPOConfig()
    env = DeepRMEnv(spec, env_cfg)
    model = ActorCritic(env.observation_size, env.n_actions, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    validation_jobsets = [
        make_jobset(100000 + args.seed + i, spec, horizon=args.arrival_horizon,
                    lookahead=env_cfg.lookahead, rate=args.arrival_rate)
        for i in range(args.validation_episodes)
    ]
    rows = []
    best = float("inf")
    for update in range(args.updates):
        episodes = [
            collect_episode(
                env,
                make_jobset(args.seed * 100000 + update * args.episodes_per_update + i,
                            spec, horizon=args.arrival_horizon,
                            lookahead=env_cfg.lookahead, rate=args.arrival_rate),
                model,
                cfg,
            )
            for i in range(args.episodes_per_update)
        ]
        train_stats = ppo_update(model, optimizer, episodes, cfg)
        eval_stats = evaluate(model, env, validation_jobsets)
        row = {
            "update": update + 1,
            "train_slowdown": float(np.mean([ep.mean_slowdown for ep in episodes])),
            "train_return": float(np.mean([ep.raw_return for ep in episodes])),
            "train_truncated": int(sum(ep.truncated for ep in episodes)),
            **train_stats,
            **eval_stats,
        }
        rows.append(row)
        print(f"{update + 1:04d} train slowdown={row['train_slowdown']:.3f} "
              f"validation slowdown={row['validation_slowdown']:.3f} "
              f"truncated={row['validation_truncated']}")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "observation_size": env.observation_size,
            "n_actions": env.n_actions,
            "hidden": args.hidden,
            "seed": args.seed,
            "ppo_config": asdict(cfg),
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if row["validation_slowdown"] < best:
            best = row["validation_slowdown"]
            torch.save(checkpoint, args.output / "best.pt")
        with (args.output / "train_log.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
