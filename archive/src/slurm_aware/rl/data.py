"""Trace loading and reproducible train/validation/test splits by whole trace."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from slurm_aware.simulator import Job
from slurm_aware.workloads import read_traces

SPLITS = ("train", "val", "test")


def load_traces(path: str | Path) -> dict[str, list[Job]]:
    """The workload CSV schema already matches `read_traces`; no adapter is needed."""
    return read_traces(path)


def split_trace_ids(
    trace_ids: Sequence[str],
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Shuffle sorted IDs with a fixed seed so splits never depend on CSV order."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must be three nonnegative numbers summing to 1.")
    ids = sorted(trace_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Trace IDs must be unique.")
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * fractions[0])
    n_val = round(len(ids) * fractions[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
