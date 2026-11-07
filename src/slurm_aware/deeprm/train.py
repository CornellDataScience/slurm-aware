"""REINFORCE with a time-dependent baseline, as in DeepRM (Mao et al., 2016).

Each iteration, for every training jobset:

1. run `episodes_per_jobset` (N) Monte Carlo episodes with the current policy,
   recording (state, action, reward) at every decision;
2. compute returns v_t = sum_{s >= t} gamma^(s - t) r_s per episode;
3. baseline b_t = mean over the N episodes of v_t at the same decision index t
   (an episode that has already ended contributes a return of 0 there);
4. accumulate sum_t grad log pi(a_t | s_t) * (v_t - b_t).

Gradients from all jobsets are combined into one RMSprop step per iteration. The
sum is divided by the number of decisions in the batch (a mean, not a sum); RMSprop
divides by the running gradient RMS, so this only changes the effective epsilon.

Defaults are the paper's. They were tuned for its cleaner setup (bimodal
durations, Bernoulli arrivals, 2 resources); heavy-tailed durations, bursty
arrivals and 3 resources give much noisier returns and may need a lower learning
rate, more episodes per jobset or more iterations.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from slurm_aware.deeprm.baselines import FIFO, SJF
from slurm_aware.deeprm.env import (
    DeepRMEnv,
    EnvConfig,
    run_episode,
    summarize_scheduling_metrics,
)
from slurm_aware.deeprm.generator import GeneratorConfig, JobTraceGenerator, jobset_seed
from slurm_aware.deeprm.jobs import Job
from slurm_aware.deeprm.policy import NetworkPolicy, PolicyNetwork, sample_actions

CHECKPOINT_VERSION = 1
_ROLLOUT_STREAM = 1000  # SeedSequence spawn-key tag for action sampling, apart from jobsets
_GRAD_CHUNK = 4096  # decisions per forward/backward pass, to bound memory


@dataclass
class TrainConfig:
    output: str = "artifacts/deeprm"
    seed: int = 0
    # Data: GeneratorConfig.to_dict(), or None for built-in defaults. Training jobsets
    # cycle through `train_loads`; one load reproduces the paper's per-load training.
    generator: dict | None = None
    train_loads: tuple[float, ...] = (0.7,)
    num_jobsets: int = 100
    val_jobsets: int = 20
    # Environment
    num_slots: int = 10
    backlog_size: int = 60
    lookahead: int = 20
    max_time: int = 500
    # Network
    hidden_sizes: tuple[int, ...] = (20,)
    activation: str = "relu"
    # REINFORCE
    episodes_per_jobset: int = 20
    iterations: int = 1000
    lr: float = 1e-3
    rmsprop_alpha: float = 0.9
    rmsprop_eps: float = 1e-9
    gamma: float = 1.0
    # Bookkeeping
    eval_every: int = 10
    checkpoint_every: int = 10
    workers: int = max(0, min(8, (os.cpu_count() or 2) - 1))


def generator_config(cfg: TrainConfig) -> GeneratorConfig:
    gen = GeneratorConfig.from_dict(cfg.generator) if cfg.generator else GeneratorConfig()
    if gen.max_duration > cfg.lookahead:
        raise ValueError("Generator max_duration must not exceed the env lookahead T.")
    return gen


def env_config(cfg: TrainConfig) -> EnvConfig:
    return EnvConfig(cfg.num_slots, cfg.backlog_size, cfg.lookahead, cfg.max_time)


def make_env(cfg: TrainConfig, **kwargs) -> DeepRMEnv:
    return DeepRMEnv(generator_config(cfg).cluster, env_config(cfg), **kwargs)


def build_jobsets(cfg: TrainConfig, split: str, count: int) -> list[list[Job]]:
    base = JobTraceGenerator(generator_config(cfg))
    gens = {load: base.with_load(load) for load in cfg.train_loads}
    return [
        gens[cfg.train_loads[i % len(cfg.train_loads)]].jobset(jobset_seed(cfg.seed, split, i))
        for i in range(count)
    ]


def build_network(cfg: TrainConfig, env: DeepRMEnv) -> PolicyNetwork:
    return PolicyNetwork(env.observation_size, env.n_actions, cfg.hidden_sizes, cfg.activation)


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def time_baseline(returns: Sequence[np.ndarray]) -> np.ndarray:
    """Mean return at each decision index across episodes, padding ended episodes with 0."""
    length = max((len(r) for r in returns), default=0)
    padded = np.zeros((len(returns), length))
    for i, r in enumerate(returns):
        padded[i, : len(r)] = r
    return padded.mean(axis=0)


def collect_episodes(
    network: PolicyNetwork, envs: Sequence[DeepRMEnv], jobs: Sequence[Job], rng
) -> list[dict]:
    """Run len(envs) episodes of one jobset in lockstep, batching the policy forward pass."""
    current = [env.reset(jobs)[0] for env in envs]
    episodes = [{"obs": [], "actions": [], "rewards": []} for _ in envs]
    active = [i for i, env in enumerate(envs) if not env.done]
    while active:
        actions = sample_actions(network.probabilities(np.stack([current[i] for i in active])), rng)
        for i, action in zip(active, actions.tolist()):
            episodes[i]["obs"].append(current[i])
            episodes[i]["actions"].append(action)
            current[i], reward, *_ = envs[i].step(action)
            episodes[i]["rewards"].append(reward)
        active = [i for i in active if not envs[i].done]
    for episode, env in zip(episodes, envs):
        slowdowns = env.slowdowns(include_unfinished=True)
        episode["mean_slowdown"] = float(np.mean(slowdowns)) if slowdowns else 0.0
        episode["truncated"] = env.truncated
        episode.update(env.scheduling_metrics())
    return episodes


def accumulate_gradient(
    network: PolicyNetwork, episodes: Sequence[dict], gamma: float
) -> tuple[int, dict]:
    """Add sum_t grad log pi * (v_t - b_t) for one jobset's episodes into `.grad`.

    Returns the number of decisions and per-jobset statistics. The loss is the
    negated objective, so `backward()` yields the gradient for descent.
    """
    returns = [discounted_returns(e["rewards"], gamma) for e in episodes]
    baseline = time_baseline(returns)
    advantages = np.concatenate([r - baseline[: len(r)] for r in returns] or [np.zeros(0)])
    actions = np.concatenate([np.asarray(e["actions"], dtype=np.int64) for e in episodes])
    observations = [obs for e in episodes for obs in e["obs"]]
    entropy = 0.0
    for begin in range(0, len(observations), _GRAD_CHUNK):
        end = begin + _GRAD_CHUNK
        logits = network(torch.from_numpy(np.stack(observations[begin:end])))
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, torch.from_numpy(actions[begin:end])[:, None])[:, 0]
        adv = torch.from_numpy(advantages[begin:end]).float()
        (-(chosen * adv).sum()).backward()
        entropy -= float((log_probs.exp() * log_probs).sum().detach())
    totals = [float(r[0]) if len(r) else 0.0 for r in returns]
    return len(observations), {
        "mean_return": float(np.mean(totals)),
        "max_return": float(np.max(totals)),
        "mean_slowdown": float(np.mean([e["mean_slowdown"] for e in episodes])),
        "mean_decisions": len(observations) / len(episodes),
        "entropy_sum": entropy,
        "truncated": sum(e["truncated"] for e in episodes),
        "wait_times": [wait for e in episodes for wait in e.get("wait_times", [])],
        "resource_usage": np.sum(
            [e["resource_usage"] for e in episodes if "resource_usage" in e], axis=0
        ).tolist(),
        "timesteps": sum(e.get("timesteps", 0) for e in episodes),
    }


# Per-process state for worker processes (macOS uses spawn: build once per worker).
_WORKER: dict = {}


def _init_worker(cfg_dict: dict, jobsets: list[list[dict]], single_thread: bool) -> None:
    if single_thread:  # Workers already run in parallel; avoid oversubscribing cores.
        torch.set_num_threads(1)
    cfg = TrainConfig(**cfg_dict)
    env = make_env(cfg)
    _WORKER.update(
        cfg=cfg,
        network=build_network(cfg, env),
        envs=[env] + [make_env(cfg) for _ in range(cfg.episodes_per_jobset - 1)],
        jobsets=[[Job.from_dict(j) for j in js] for js in jobsets],
    )


def _gradient_task(
    state: Mapping[str, torch.Tensor], iteration: int, indices: Sequence[int]
) -> tuple[list[np.ndarray], int, list[dict]]:
    cfg, network = _WORKER["cfg"], _WORKER["network"]
    network.load_state_dict(state)
    for p in network.parameters():
        p.grad = torch.zeros_like(p)
    total, stats = 0, []
    for index in indices:
        seq = np.random.SeedSequence(cfg.seed, spawn_key=(_ROLLOUT_STREAM, iteration, index))
        episodes = collect_episodes(
            network, _WORKER["envs"], _WORKER["jobsets"][index], np.random.default_rng(seq)
        )
        n, s = accumulate_gradient(network, episodes, cfg.gamma)
        total += n
        stats.append({"jobset": index, **s})
    return [p.grad.numpy().copy() for p in network.parameters()], total, stats


class RolloutRunner:
    """Computes the summed policy gradient over all training jobsets, optionally in parallel."""

    def __init__(self, cfg: TrainConfig, jobsets: Sequence[Sequence[Job]]) -> None:
        payload = (asdict(cfg), [[j.to_dict() for j in js] for js in jobsets])
        self.n_jobsets = len(jobsets)
        self.workers = min(cfg.workers, self.n_jobsets)
        self._pool = None
        if self.workers:
            self._pool = ProcessPoolExecutor(
                self.workers, initializer=_init_worker, initargs=(*payload, True)
            )
        else:
            _init_worker(*payload, False)

    def gradient(
        self, state: Mapping[str, torch.Tensor], iteration: int
    ) -> tuple[list[np.ndarray], int, list[dict]]:
        indices = list(range(self.n_jobsets))
        if self._pool is None:
            return _gradient_task(state, iteration, indices)
        chunks = [indices[w :: self.workers] for w in range(self.workers)]
        k = len(chunks)
        results = list(self._pool.map(_gradient_task, [state] * k, [iteration] * k, chunks))
        grads = [sum(parts) for parts in zip(*(r[0] for r in results))]
        stats = sorted((s for r in results for s in r[2]), key=lambda s: s["jobset"])
        return grads, sum(r[1] for r in results), stats

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        _WORKER.clear()

    def __enter__(self) -> RolloutRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def greedy_slowdown(network: PolicyNetwork, env: DeepRMEnv, jobsets) -> float:
    """Mean job slowdown of the argmax policy, pooled over jobsets."""
    policy = NetworkPolicy(network, greedy=True)
    slowdowns = [s for js in jobsets for s in run_episode(env, policy, js)["slowdowns"]]
    return float(np.mean(slowdowns)) if slowdowns else 0.0


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # Atomic: an interrupted save never corrupts the last checkpoint.


def load_checkpoint(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a version-{CHECKPOINT_VERSION} DeepRM checkpoint.")
    return state


def config_from_checkpoint(state: Mapping) -> TrainConfig:
    raw = dict(state["train_config"])
    for key in ("train_loads", "hidden_sizes"):
        raw[key] = tuple(raw[key])
    return TrainConfig(**raw)


def network_from_checkpoint(state: Mapping) -> tuple[PolicyNetwork, TrainConfig]:
    cfg = config_from_checkpoint(state)
    network = build_network(cfg, make_env(cfg))
    network.load_state_dict(state["policy"])
    network.eval()
    return network, cfg


LOG_FIELDS = (
    "iteration",
    "mean_return",
    "max_return",
    "mean_slowdown",
    "mean_entropy",
    "mean_decisions",
    "truncated_episodes",
    "val_slowdown",
    "iteration_seconds",
    "elapsed_seconds",
    "p95_wait",
    "p99_wait",
    "resource_utilization",
    "fifo_val_slowdown",
    "fifo_val_p95_wait",
    "fifo_val_p99_wait",
    "fifo_val_resource_utilization",
)


def train(
    cfg: TrainConfig,
    resume: str | Path | None = None,
    log=functools.partial(print, flush=True),
) -> dict:
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(resume) if resume else None
    if state:
        # Everything but output location, parallelism and total iterations is restored.
        overrides = {"output": cfg.output, "workers": cfg.workers, "iterations": cfg.iterations}
        cfg = replace(config_from_checkpoint(state), **overrides)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg)
    network = build_network(cfg, env)
    optimizer = torch.optim.RMSprop(
        network.parameters(), lr=cfg.lr, alpha=cfg.rmsprop_alpha, eps=cfg.rmsprop_eps
    )
    start, best, elapsed = 0, math.inf, 0.0
    if state:
        network.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, elapsed = state["iteration"] + 1, state["best_val_slowdown"], state["elapsed"]
        log(f"Resumed from {resume} at iteration {start}.")

    train_sets = build_jobsets(cfg, "train", cfg.num_jobsets)
    val_sets = build_jobsets(cfg, "val", cfg.val_jobsets)
    if not train_sets or not val_sets:
        raise ValueError("Need at least one training and one validation jobset.")
    references = {}
    for policy in (SJF(), FIFO()):
        episodes = [run_episode(env, policy, js) for js in val_sets]
        slowdowns = [s for episode in episodes for s in episode["slowdowns"]]
        reference = {
            "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else 0.0,
            **summarize_scheduling_metrics(episodes, env.spec),
            "truncated_episodes": sum(e["truncated"] for e in episodes),
        }
        references[policy.name] = reference
        utilization = " ".join(
            f"{r}={u:.1%}" for r, u in reference["resource_utilization"].items()
        )
        log(
            f"Validation reference: {policy.name.upper()} "
            f"mean slowdown {reference['mean_slowdown']:.3f} "
            f"| wait p95 {reference['p95_wait']:.2f} p99 {reference['p99_wait']:.2f} "
            f"| util {utilization} | truncated {reference['truncated_episodes']}"
        )
    sjf_val = references["sjf"]["mean_slowdown"]
    fifo_val = references["fifo"]

    log_path = output / "train_log.csv"
    new_log = not (state and log_path.exists())
    if not new_log:
        # Upgrade older logs before appending rows with the new metric columns.
        with log_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            old_fields = reader.fieldnames
            old_rows = list(reader) if old_fields != list(LOG_FIELDS) else None
        if old_rows is not None:
            temporary = log_path.with_suffix(".tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
                writer.writeheader()
                writer.writerows(old_rows)
            temporary.replace(log_path)
    last_iteration = start - 1
    with RolloutRunner(cfg, train_sets) as runner, log_path.open(
        "w" if new_log else "a", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        if new_log:
            writer.writeheader()
        for iteration in range(start, cfg.iterations):
            began = time.monotonic()
            grads, n_decisions, stats = runner.gradient(network.state_dict(), iteration)
            optimizer.zero_grad()
            for p, g in zip(network.parameters(), grads):
                p.grad = torch.from_numpy(g / max(1, n_decisions))
            optimizer.step()
            seconds = time.monotonic() - began
            elapsed += seconds
            last_iteration = iteration
            metrics = summarize_scheduling_metrics(stats, env.spec)

            row = {
                "iteration": iteration,
                "mean_return": float(np.mean([s["mean_return"] for s in stats])),
                "max_return": float(np.max([s["max_return"] for s in stats])),
                "mean_slowdown": float(np.mean([s["mean_slowdown"] for s in stats])),
                "mean_entropy": sum(s["entropy_sum"] for s in stats) / max(1, n_decisions),
                "mean_decisions": float(np.mean([s["mean_decisions"] for s in stats])),
                "truncated_episodes": sum(s["truncated"] for s in stats),
                "val_slowdown": "",
                "iteration_seconds": seconds,
                "elapsed_seconds": elapsed,
                "p95_wait": metrics["p95_wait"],
                "p99_wait": metrics["p99_wait"],
                "resource_utilization": json.dumps(metrics["resource_utilization"]),
                "fifo_val_slowdown": fifo_val["mean_slowdown"],
                "fifo_val_p95_wait": fifo_val["p95_wait"],
                "fifo_val_p99_wait": fifo_val["p99_wait"],
                "fifo_val_resource_utilization": json.dumps(fifo_val["resource_utilization"]),
            }
            last = iteration == cfg.iterations - 1
            is_best = False
            if (iteration + 1) % cfg.eval_every == 0 or last:
                row["val_slowdown"] = greedy_slowdown(network, env, val_sets)
                is_best = row["val_slowdown"] < best
                best = min(best, row["val_slowdown"])
            writer.writerow(row)
            stream.flush()
            log(
                f"iter {iteration:4d} | return {row['mean_return']:9.2f} "
                f"slowdown {row['mean_slowdown']:6.3f} entropy {row['mean_entropy']:.3f} "
                f"| wait p95 {row['p95_wait']:.2f} p99 {row['p99_wait']:.2f} "
                "| util "
                + " ".join(f"{r}={u:.1%}" for r, u in metrics["resource_utilization"].items())
                + f" | {seconds:5.2f}s"
                + (
                    f" | val slowdown {row['val_slowdown']:.3f} "
                    f"(SJF {sjf_val:.3f}, FIFO {fifo_val['mean_slowdown']:.3f})"
                    + (" new best" if is_best else "")
                    if row["val_slowdown"] != ""
                    else ""
                )
            )
            if (iteration + 1) % cfg.checkpoint_every == 0 or last or is_best:
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "train_config": asdict(cfg),
                    "generator": generator_config(cfg).to_dict(),
                    "policy": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "best_val_slowdown": best,
                    "sjf_val_slowdown": sjf_val,
                    "fifo_val_metrics": fifo_val,
                    "elapsed": elapsed,
                }
                if (iteration + 1) % cfg.checkpoint_every == 0 or last:
                    _save(output / "latest.pt", checkpoint)
                if is_best:
                    _save(output / "best.pt", checkpoint)
    return {
        "best_val_slowdown": best,
        "sjf_val_slowdown": sjf_val,
        "fifo_val_metrics": fifo_val,
        "iterations": last_iteration + 1,
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    d = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=d.output)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--generator-config", help="GeneratorConfig JSON (default: built-in)")
    parser.add_argument("--regime", choices=("steady", "bursty", "diurnal"),
                        help="Override the generator's arrival regime")
    parser.add_argument("--train-loads", type=float, nargs="+", default=list(d.train_loads))
    parser.add_argument("--num-jobsets", type=int, default=d.num_jobsets)
    parser.add_argument("--val-jobsets", type=int, default=d.val_jobsets)
    parser.add_argument("--num-slots", type=int, default=d.num_slots)
    parser.add_argument("--backlog-size", type=int, default=d.backlog_size)
    parser.add_argument("--lookahead", type=int, default=d.lookahead)
    parser.add_argument("--max-time", type=int, default=d.max_time)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(d.hidden_sizes))
    parser.add_argument("--activation", default=d.activation)
    parser.add_argument("--episodes-per-jobset", type=int, default=d.episodes_per_jobset)
    parser.add_argument("--iterations", type=int, default=d.iterations)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--rmsprop-alpha", type=float, default=d.rmsprop_alpha)
    parser.add_argument("--rmsprop-eps", type=float, default=d.rmsprop_eps)
    parser.add_argument("--gamma", type=float, default=d.gamma)
    parser.add_argument("--eval-every", type=int, default=d.eval_every)
    parser.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    parser.add_argument("--workers", type=int, default=d.workers, help="0 = no subprocesses")
    parser.add_argument("--resume", help="Checkpoint to continue from (e.g. .../latest.pt)")
    args = parser.parse_args(argv)
    try:
        generator = (
            json.loads(Path(args.generator_config).read_text()) if args.generator_config else None
        )
        if args.regime:
            gen = GeneratorConfig.from_dict(generator) if generator else GeneratorConfig()
            generator = replace(gen, arrival=replace(gen.arrival, regime=args.regime)).to_dict()
        cfg = TrainConfig(
            output=args.output,
            seed=args.seed,
            generator=generator,
            train_loads=tuple(args.train_loads),
            num_jobsets=args.num_jobsets,
            val_jobsets=args.val_jobsets,
            num_slots=args.num_slots,
            backlog_size=args.backlog_size,
            lookahead=args.lookahead,
            max_time=args.max_time,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
            episodes_per_jobset=args.episodes_per_jobset,
            iterations=args.iterations,
            lr=args.lr,
            rmsprop_alpha=args.rmsprop_alpha,
            rmsprop_eps=args.rmsprop_eps,
            gamma=args.gamma,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            workers=args.workers,
        )
        result = train(cfg, resume=args.resume)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
