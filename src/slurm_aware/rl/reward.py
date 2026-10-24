"""Simulator interface and reward: the non-differentiable part of each training step.

`SimulatorRunner` replays (trace, weights) pairs, optionally in worker processes
that each load the workload once. The reward for a trace is the negative ratio of
the policy's objective to a fixed reference policy's objective on the same trace,
so a reward of -1.0 means "as good as the reference" and -0.9 means 10% better.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from slurm_aware.sim_cli import load_config
from slurm_aware.simulator import PriorityWeights, SimulationConfig, simulate
from slurm_aware.workloads import read_traces

DEFAULT_OBJECTIVE = "mean_wait_time_seconds"
Metrics = dict[str, float | int | None]

# Per-process state for worker processes (macOS uses spawn, so load once per worker).
_WORKER: dict = {}


def _init_worker(workload: str, config: str | None) -> None:
    _WORKER["traces"] = read_traces(workload)
    _WORKER["config"] = load_config(Path(config) if config else None)


def _simulate_one(trace_id: str, weights: Mapping[str, float]) -> Metrics:
    return simulate(
        _WORKER["traces"][trace_id], PriorityWeights(**weights), _WORKER["config"]
    ).metrics


class SimulatorRunner:
    """Replay traces with given weights; `workers=0` runs in this process (tests, debugging)."""

    def __init__(
        self, workload: str | Path, config: str | Path | None, *, workers: int = 0
    ) -> None:
        self.workload, self.config_path = str(workload), str(config) if config else None
        self.workers = workers
        self._pool = None
        if workers:
            self._pool = ProcessPoolExecutor(
                workers, initializer=_init_worker, initargs=(self.workload, self.config_path)
            )
        else:
            _init_worker(self.workload, self.config_path)

    @property
    def config(self) -> SimulationConfig:
        return load_config(Path(self.config_path) if self.config_path else None)

    def run(self, requests: Sequence[tuple[str, PriorityWeights]]) -> list[Metrics]:
        payload = [(trace_id, asdict(weights)) for trace_id, weights in requests]
        if self._pool is None:
            return [_simulate_one(*item) for item in payload]
        ids, weights = zip(*payload) if payload else ((), ())
        return list(self._pool.map(_simulate_one, ids, weights, chunksize=4))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self) -> SimulatorRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def reference_objectives(
    runner: SimulatorRunner,
    trace_ids: Iterable[str],
    weights: PriorityWeights,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, float]:
    """Objective of a fixed reference weight vector on each trace (computed once, cached)."""
    trace_ids = list(trace_ids)
    metrics = runner.run([(trace_id, weights) for trace_id in trace_ids])
    return {trace_id: float(m[objective]) for trace_id, m in zip(trace_ids, metrics)}


def compute_reward(policy_value: float, reference_value: float) -> float:
    """Negative relative objective (lower objective is better, so higher reward is better)."""
    if reference_value <= 0:
        # A reference with zero waiting cannot be beaten; reward only matching it.
        return -1.0 - policy_value
    return -policy_value / reference_value


class EMABaseline:
    """Exponential moving average of batch-mean rewards; the first batch initializes it."""

    def __init__(self, decay: float = 0.9, value: float | None = None) -> None:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1).")
        self.decay, self.value = decay, value

    def advantages(self, rewards: Sequence[float]) -> list[float]:
        """Advantages against the baseline *before* this batch, then update the baseline."""
        batch_mean = sum(rewards) / len(rewards)
        baseline = batch_mean if self.value is None else self.value
        self.value = (
            batch_mean
            if self.value is None
            else self.decay * self.value + (1 - self.decay) * batch_mean
        )
        return [reward - baseline for reward in rewards]
