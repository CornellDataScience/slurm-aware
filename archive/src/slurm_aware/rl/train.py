"""REINFORCE training loop for the priority-weight contextual bandit.

Each training example is one trace (job set): observe its features, sample one
weight vector, replay the trace once, receive one reward. There are no state
transitions, returns, discounting, or critic.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from slurm_aware.rl.data import load_traces, split_trace_ids
from slurm_aware.rl.features import FEATURE_NAMES, FeatureNormalizer, feature_matrix
from slurm_aware.rl.policy import WEIGHT_NAMES, GaussianPolicy, action_to_weights
from slurm_aware.rl.reward import (
    DEFAULT_OBJECTIVE,
    EMABaseline,
    SimulatorRunner,
    compute_reward,
    reference_objectives,
)
from slurm_aware.simulator import PriorityWeights

CHECKPOINT_VERSION = 1


@dataclass
class TrainConfig:
    workload: str = "data/generated_traces/trace_job_1.csv"
    config: str | None = "examples/cds_cluster.json"
    output: str = "artifacts/rl"
    objective: str = DEFAULT_OBJECTIVE
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    hidden_sizes: tuple[int, ...] = (64, 64)
    initial_std: float = 0.5
    baseline_decay: float = 0.9
    seed: int = 0
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    workers: int = max(1, min(8, (os.cpu_count() or 2) - 1))
    log_every: int = 5


@dataclass
class StepStats:
    mean_reward: float
    baseline: float
    mean_advantage: float
    loss: float
    mean_objective: float
    mean_std: float


def reinforce_step(
    policy: GaussianPolicy,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    trace_ids: Sequence[str],
    runner: SimulatorRunner,
    references: Mapping[str, float],
    baseline: EMABaseline,
    objective: str = DEFAULT_OBJECTIVE,
) -> StepStats:
    """One policy update from a batch of independent (context, action, reward) triples."""
    dist = policy.distribution(features)
    actions = dist.sample()  # No gradient flows through sampling or the simulator.
    weights = [action_to_weights(row.tolist()) for row in actions]
    metrics = runner.run(list(zip(trace_ids, weights)))
    values = [float(m[objective]) for m in metrics]
    rewards = [compute_reward(v, references[t]) for v, t in zip(values, trace_ids)]
    baseline_before = baseline.value if baseline.value is not None else sum(rewards) / len(rewards)
    advantages = torch.tensor(baseline.advantages(rewards), dtype=torch.float32)

    # Score-function gradient: raise log-probability of actions with positive advantage.
    log_prob = dist.log_prob(actions).sum(dim=-1)
    loss = -(advantages * log_prob).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return StepStats(
        mean_reward=sum(rewards) / len(rewards),
        baseline=baseline_before,
        mean_advantage=float(advantages.mean()),
        loss=float(loss.detach()),
        mean_objective=sum(values) / len(values),
        mean_std=float(dist.stddev.detach().mean()),
    )


def greedy_weights(policy: GaussianPolicy, features: torch.Tensor) -> list[PriorityWeights]:
    """Deterministic deployment action: the Gaussian mean for each context."""
    with torch.no_grad():
        mean, _ = policy(features)
    return [action_to_weights(row.tolist()) for row in mean]


def validate(
    policy: GaussianPolicy,
    features: torch.Tensor,
    trace_ids: Sequence[str],
    runner: SimulatorRunner,
    references: Mapping[str, float],
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, float]:
    metrics = runner.run(list(zip(trace_ids, greedy_weights(policy, features))))
    values = [float(m[objective]) for m in metrics]
    ratios = [v / references[t] if references[t] else math.nan for v, t in zip(values, trace_ids)]
    return {
        "val_mean_objective": sum(values) / len(values),
        "val_mean_ratio": sum(ratios) / len(ratios),
        "val_win_rate": sum(r < 1 for r in ratios) / len(ratios),
    }


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # Atomic: an interrupted save never corrupts the last checkpoint.


def load_checkpoint(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a version-{CHECKPOINT_VERSION} RL checkpoint.")
    return state


def build_policy(state: Mapping) -> GaussianPolicy:
    cfg = state["train_config"]
    policy = GaussianPolicy(
        len(FEATURE_NAMES), hidden_sizes=cfg["hidden_sizes"], initial_std=cfg["initial_std"]
    )
    policy.load_state_dict(state["policy"])
    return policy


def train(
    cfg: TrainConfig,
    resume: str | Path | None = None,
    log=functools.partial(print, flush=True),
) -> dict:
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(resume) if resume else None
    if state:
        # Everything but output location, parallelism, and total epochs comes from the checkpoint.
        overrides = {"output": cfg.output, "workers": cfg.workers, "epochs": cfg.epochs}
        cfg = TrainConfig(**{**state["train_config"], **overrides})
    else:
        (output / "val_log.jsonl").unlink(missing_ok=True)
    torch.manual_seed(cfg.seed)

    traces = load_traces(cfg.workload)
    splits = (
        state["splits"]
        if state
        else split_trace_ids(list(traces), seed=cfg.seed, fractions=cfg.split_fractions)
    )
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training and validation splits must both be nonempty.")
    with SimulatorRunner(cfg.workload, cfg.config, workers=cfg.workers) as runner:
        sim_config = runner.config
        raw_train = feature_matrix(traces, splits["train"], sim_config)
        normalizer = (
            FeatureNormalizer.from_state_dict(state["normalizer"])
            if state
            else FeatureNormalizer.fit(raw_train)
        )
        train_x = torch.from_numpy(normalizer(raw_train))
        val_x = torch.from_numpy(normalizer(feature_matrix(traces, splits["val"], sim_config)))
        index = {trace_id: i for i, trace_id in enumerate(splits["train"])}

        if state:
            references = state["references"]
        else:
            log("Simulating reference (default) weights on train/val traces...")
            references = reference_objectives(
                runner, splits["train"] + splits["val"], PriorityWeights(), cfg.objective
            )

        policy = GaussianPolicy(
            len(FEATURE_NAMES), hidden_sizes=cfg.hidden_sizes, initial_std=cfg.initial_std
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        baseline = EMABaseline(cfg.baseline_decay)
        start_epoch, step, best = 0, 0, math.inf
        if state:
            policy.load_state_dict(state["policy"])
            optimizer.load_state_dict(state["optimizer"])
            baseline.value = state["baseline"]
            start_epoch, step, best = state["epoch"] + 1, state["step"], state["best_val_ratio"]
            torch.set_rng_state(state["torch_rng"])
            log(f"Resumed from {resume} at epoch {start_epoch}, step {step}.")

        log_path = output / "train_log.csv"
        fields = ["epoch", "step", *StepStats.__dataclass_fields__, "elapsed_seconds"]
        new_log = not (state and log_path.exists())
        with log_path.open("w" if new_log else "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if new_log:
                writer.writeheader()
            started = time.monotonic()
            for epoch in range(start_epoch, cfg.epochs):
                order = list(splits["train"])
                random.Random(cfg.seed * 100_003 + epoch).shuffle(order)
                for begin in range(0, len(order), cfg.batch_size):
                    batch = order[begin : begin + cfg.batch_size]
                    x = train_x[[index[trace_id] for trace_id in batch]]
                    stats = reinforce_step(
                        policy, optimizer, x, batch, runner, references, baseline, cfg.objective
                    )
                    step += 1
                    row = {"epoch": epoch, "step": step, **asdict(stats)}
                    writer.writerow({**row, "elapsed_seconds": time.monotonic() - started})
                    if step % cfg.log_every == 0:
                        log(
                            f"epoch {epoch:3d} step {step:5d} | reward {stats.mean_reward:+.4f} "
                            f"baseline {stats.baseline:+.4f} adv {stats.mean_advantage:+.4f} "
                            f"loss {stats.loss:+.4f} | {cfg.objective} {stats.mean_objective:7.2f} "
                            f"std {stats.mean_std:.3f}"
                        )
                stream.flush()

                val = validate(policy, val_x, splits["val"], runner, references, cfg.objective)
                is_best = val["val_mean_ratio"] < best
                best = min(best, val["val_mean_ratio"])
                log(
                    f"== epoch {epoch:3d} validation: {cfg.objective} "
                    f"{val['val_mean_objective']:.2f}"
                    f" | ratio vs default {val['val_mean_ratio']:.4f}"
                    f" | beats default on {val['val_win_rate']:.0%} of traces"
                    + (" | new best" if is_best else "")
                )
                with (output / "val_log.jsonl").open("a") as val_stream:
                    val_stream.write(json.dumps({"epoch": epoch, "step": step, **val}) + "\n")
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "train_config": asdict(cfg),
                    "feature_names": list(FEATURE_NAMES),
                    "weight_names": list(WEIGHT_NAMES),
                    "splits": splits,
                    "normalizer": normalizer.state_dict(),
                    "references": references,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "baseline": baseline.value,
                    "epoch": epoch,
                    "step": step,
                    "best_val_ratio": best,
                    "torch_rng": torch.get_rng_state(),
                }
                _save(output / "latest.pt", checkpoint)
                if is_best:
                    _save(output / "best.pt", checkpoint)
    return {"best_val_ratio": best, "steps": step, "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", default=defaults.workload, help="Trace CSV")
    parser.add_argument("--config", default=defaults.config, help="Simulator config JSON")
    parser.add_argument("--output", default=defaults.output, help="Checkpoint/log directory")
    parser.add_argument("--objective", default=defaults.objective, help="Metric to minimize")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(defaults.hidden_sizes))
    parser.add_argument("--initial-std", type=float, default=defaults.initial_std)
    parser.add_argument("--baseline-decay", type=float, default=defaults.baseline_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=list(defaults.split_fractions),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--workers", type=int, default=defaults.workers, help="0 = no subprocesses")
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--resume", help="Checkpoint to continue from (e.g. artifacts/rl/latest.pt)"
    )
    args = parser.parse_args(argv)
    cfg = TrainConfig(
        workload=args.workload,
        config=args.config,
        output=args.output,
        objective=args.objective,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_sizes=tuple(args.hidden_sizes),
        initial_std=args.initial_std,
        baseline_decay=args.baseline_decay,
        seed=args.seed,
        split_fractions=tuple(args.split),
        workers=args.workers,
        log_every=args.log_every,
    )
    try:
        result = train(cfg, resume=args.resume)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
