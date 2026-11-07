"""Cluster backends: who tracks allocations, availability and completions for the env.

The environment only talks to `ClusterBackend`. `InProcessBackend` is the pooled,
timestep-grid model from the original paper and is what training uses. A real
backend (e.g. the systems team's allocator or Slurm) implements the same five
methods; it needs no image code, since `canvas` has a default built from
`available`.

Allocation is fixed and non-preemptive: a job placed at time `now` holds its
demand over `[now, now + duration)`. The backend sees a lookahead window of
`horizon` timesteps, so durations longer than `horizon` cannot be placed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from slurm_aware.deeprm.jobs import ClusterSpec, Job


class ClusterBackend(ABC):
    def __init__(self, spec: ClusterSpec, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        self.spec, self.horizon = spec, horizon

    @abstractmethod
    def reset(self) -> None:
        """Empty cluster at time 0."""

    @abstractmethod
    def available(self) -> np.ndarray:
        """Free units, shape (horizon, n_resources); row 0 is the current timestep."""

    @abstractmethod
    def allocate(self, job: Job, now: int) -> bool:
        """Start `job` at `now` if its demand fits for its whole duration; False if not."""

    @abstractmethod
    def advance(self) -> list[tuple[Job, int, int]]:
        """Move one timestep forward; return (job, start, finish) for jobs that finished."""

    @abstractmethod
    def running(self) -> list[Job]:
        """Jobs currently holding resources."""

    def fits(self, job: Job) -> bool:
        if job.duration > self.horizon:
            return False
        return bool(np.all(self.available()[: job.duration] >= np.array(job.demand)))

    def canvas(self) -> list[np.ndarray]:
        """Per-resource image, shape (capacity, horizon); used units are 1, from the top."""
        used = np.array(self.spec.capacities) - self.available()
        images = []
        for r, capacity in enumerate(self.spec.capacities):
            rows = np.arange(capacity)[:, None]
            images.append((rows < used[:, r][None, :]).astype(np.float32))
        return images


class InProcessBackend(ClusterBackend):
    """Pooled cluster as in the original code: each job gets distinct unit rows per timestep.

    Canvas cells hold a per-job shade in (0, 1] so the policy can tell neighbouring
    jobs apart, as in the original's colored cluster image.
    """

    def __init__(self, spec: ClusterSpec, horizon: int, n_shades: int = 40) -> None:
        super().__init__(spec, horizon)
        self.shades = np.linspace(1.0, 1.0 / n_shades, n_shades, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.now = 0
        self._free = np.tile(np.array(self.spec.capacities), (self.horizon, 1))
        # Job index + 1 per (unit, time) cell; 0 = free.
        self._owner = [np.zeros((c, self.horizon), dtype=np.int64) for c in self.spec.capacities]
        self._running: dict[int, tuple[Job, int]] = {}
        self._shade_of: dict[int, float] = {}
        self._next_shade = 0

    def available(self) -> np.ndarray:
        return self._free.copy()

    def allocate(self, job: Job, now: int) -> bool:
        if now != self.now:
            raise ValueError(f"Backend is at time {self.now}, not {now}.")
        if job.job_id in self._running or not self.fits(job):
            return False
        span = slice(0, job.duration)
        self._free[span] -= np.array(job.demand)
        for r, units in enumerate(job.demand):
            if not units:
                continue
            owner = self._owner[r]
            for t in range(job.duration):
                rows = np.flatnonzero(owner[:, t] == 0)[:units]
                owner[rows, t] = job.job_id + 1
        self._running[job.job_id] = (job, now)
        self._shade_of[job.job_id] = float(self.shades[self._next_shade % len(self.shades)])
        self._next_shade += 1
        return True

    def advance(self) -> list[tuple[Job, int, int]]:
        self.now += 1
        self._free = np.vstack([self._free[1:], np.array(self.spec.capacities)[None, :]])
        for r, owner in enumerate(self._owner):
            self._owner[r] = np.hstack([owner[:, 1:], np.zeros((owner.shape[0], 1), np.int64)])
        finished = []
        for job_id, (job, start) in list(self._running.items()):
            if start + job.duration <= self.now:
                finished.append((job, start, start + job.duration))
                del self._running[job_id]
                del self._shade_of[job_id]
        return finished

    def running(self) -> list[Job]:
        return [job for job, _ in self._running.values()]

    def canvas(self) -> list[np.ndarray]:
        size = max(self._shade_of, default=-1) + 2
        lookup = np.zeros(max(size, max(int(o.max()) for o in self._owner) + 1), np.float32)
        for job_id, shade in self._shade_of.items():
            lookup[job_id + 1] = shade
        return [lookup[owner] for owner in self._owner]
