"""Heuristic schedulers from the DeepRM paper, as env policies (`policy(env) -> action`).

Each sees exactly what the agent sees: the M visible job slots, not the backlog.
Among slot jobs that fit now (for their whole duration), it returns the best
scoring slot, or void when none fits; time stays frozen while jobs keep fitting,
so repeated calls fill the cluster greedily within a timestep, as in the paper.

Packer's alignment uses demand and current free capacity as fractions of each
resource's capacity, so resources with more units do not dominate the score.
"""

from __future__ import annotations

import numpy as np

from slurm_aware.deeprm.env import VOID, DeepRMEnv
from slurm_aware.deeprm.jobs import Job


def packer_score(job: Job, free_now: np.ndarray, capacities: np.ndarray) -> float:
    """Alignment of demand with free capacity, normalized to [0, 1]."""
    return float((np.array(job.demand) / capacities) @ (free_now / capacities)) / len(capacities)


def sjf_score(job: Job) -> float:
    """Higher for shorter jobs, in (0, 1]."""
    return 1.0 / job.duration


class _ScoredHeuristic:
    name = "heuristic"

    def score(self, job: Job, free_now: np.ndarray, capacities: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, env: DeepRMEnv) -> int:
        fitting = env.fitting_slots()
        if not fitting:
            return VOID
        free_now = env.backend.available()[0].astype(float)
        capacities = np.array(env.spec.capacities, dtype=float)
        scores = [self.score(env.slots[a - 1], free_now, capacities) for a in fitting]
        return fitting[int(np.argmax(scores))]  # Ties go to the lowest slot.


class SJF(_ScoredHeuristic):
    """Shortest Job First."""

    name = "sjf"

    def score(self, job, free_now, capacities):
        return sjf_score(job)


class Packer(_ScoredHeuristic):
    """Best alignment between job demand and currently available resources."""

    name = "packer"

    def score(self, job, free_now, capacities):
        return packer_score(job, free_now, capacities)


class TetrisStar(_ScoredHeuristic):
    """(1 - w) * packer score + w * SJF score; w = 0 is Packer, w = 1 is SJF."""

    name = "tetris"

    def __init__(self, sjf_weight: float = 0.5) -> None:
        if not 0 <= sjf_weight <= 1:
            raise ValueError("sjf_weight must be in [0, 1].")
        self.sjf_weight = sjf_weight

    def score(self, job, free_now, capacities):
        w = self.sjf_weight
        return (1 - w) * packer_score(job, free_now, capacities) + w * sjf_score(job)


class RandomFit:
    """Uniformly random fitting slot (void when none fits); a sanity floor."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: DeepRMEnv) -> int:
        fitting = env.fitting_slots()
        return int(self.rng.choice(fitting)) if fitting else VOID


def heuristics(seed: int = 0, tetris_weight: float = 0.5) -> dict:
    return {
        "sjf": SJF(),
        "packer": Packer(),
        "tetris": TetrisStar(tetris_weight),
        "random": RandomFit(seed),
    }
