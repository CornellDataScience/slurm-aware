"""Fixed-size summary statistics of a job set: the bandit's context.

Features use only information Slurm has at submission time, plus actual runtimes
for the requested-versus-actual statistics (the trace is replayed in hindsight).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from slurm_aware.simulator import Job, SimulationConfig

ARRIVAL_WINDOW_SECONDS = 60.0

FEATURE_NAMES = (
    "job_count",
    "log_runtime_mean",
    "log_runtime_std",
    "overrequest_mean",
    "overrequest_cv",
    "nodes_mean",
    "nodes_cv",
    "cpus_per_node_mean",
    "cpus_per_node_cv",
    "offered_load",
    "gpu_job_fraction",
    "memory_fraction_mean",
    "aarch64_fraction",
    "top_account_job_share",
    "account_cpu_seconds_gini",
    "top_qos_fraction",
    "interarrival_cv",
    "peak_to_mean_arrival_rate",
)


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    return float(values.std() / mean) if mean > 0 else 0.0


def _gini(values: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    total = ordered.sum()
    if len(ordered) < 2 or total <= 0:
        return 0.0
    ranks = np.arange(1, len(ordered) + 1)
    return float((2 * ranks - len(ordered) - 1) @ ordered / (len(ordered) * total))


def extract_features(jobs: Sequence[Job], config: SimulationConfig) -> np.ndarray:
    """Return a float32 vector ordered like FEATURE_NAMES."""
    if not jobs:
        raise ValueError("Cannot extract features from an empty job set.")
    cluster = config.cluster
    runtime = np.array([job.runtime for job in jobs])
    limit = np.array([job.time_limit for job in jobs])
    nodes = np.array([job.nodes for job in jobs], dtype=float)
    cpus = np.array([job.cpus_per_node for job in jobs], dtype=float)
    cpu_seconds = nodes * cpus * runtime
    submits = np.sort([job.submit_time for job in jobs])
    span = float(submits[-1] - submits[0])

    by_account: dict[str, float] = dict.fromkeys(config.accounts, 0.0)
    for job, work in zip(jobs, cpu_seconds):
        by_account[job.account] += float(work)
    top_qos = max(config.qos_priorities.values())

    interarrival = np.diff(submits)
    if span > 0:
        bins = max(1, math.ceil(span / ARRIVAL_WINDOW_SECONDS))
        counts = np.histogram(
            submits, bins=bins, range=(submits[0], submits[0] + bins * ARRIVAL_WINDOW_SECONDS)
        )[0]
        peak_to_mean = float(counts.max() / counts.mean())
    else:
        peak_to_mean = 1.0
    # Work offered per second of arrivals, relative to schedulable CPU capacity.
    offered_load = float(cpu_seconds.sum() / (cluster.available_cpus * span)) if span else 0.0

    values = [
        len(jobs),
        float(np.log(runtime).mean()),
        float(np.log(runtime).std()),
        float((limit / runtime).mean()),
        _cv(limit / runtime),
        float(nodes.mean()),
        _cv(nodes),
        float(cpus.mean()),
        _cv(cpus),
        offered_load,
        float(np.mean([job.gpus_per_node > 0 for job in jobs])),
        float(
            np.mean([job.nodes * job.memory_mb_per_node for job in jobs]) / cluster.total_memory_mb
        ),
        float(np.mean([job.architecture == "aarch64" for job in jobs])),
        Counter(job.account for job in jobs).most_common(1)[0][1] / len(jobs),
        _gini(list(by_account.values())),
        float(np.mean([config.qos_priorities[job.qos] == top_qos for job in jobs])),
        _cv(interarrival) if len(interarrival) else 0.0,
        peak_to_mean,
    ]
    return np.asarray(values, dtype=np.float32)


class FeatureNormalizer:
    """Standardize with training-split statistics; constant features map to zero."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, features: np.ndarray) -> FeatureNormalizer:
        std = features.std(axis=0)
        return cls(features.mean(axis=0), np.where(std > 1e-8, std, 1.0))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Sequence[float]]) -> FeatureNormalizer:
        return cls(state["mean"], state["std"])


def feature_matrix(
    traces: Mapping[str, Sequence[Job]], trace_ids: Sequence[str], config: SimulationConfig
) -> np.ndarray:
    return np.stack([extract_features(traces[trace_id], config) for trace_id in trace_ids])
