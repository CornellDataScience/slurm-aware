"""DeepRM scheduling environment with Gym-style `reset()` / `step()` (no gym dependency).

State: one image per resource for the cluster (rows = units, columns = the next T
timesteps, cells shaded by the occupying job), one image per resource for each of
the first M waiting jobs (its demand over its duration), and a backlog count
encoded as `backlog_size` cells with the first `len(backlog)` set to 1. All of it is
flattened into one float32 vector.

Actions: 0 = void (NOP); 1..M = schedule the job in that slot. A valid schedule
returns reward 0 and freezes time, so several jobs can start in one timestep. Void,
an empty slot or a job that does not fit all advance time by one step.

Reward on a time-advancing step: -sum(1 / duration) over every job in the system
(running, in a slot, or backlogged) during the timestep being left. A job then
contributes exactly (finish - arrival) / duration over its life, so with gamma = 1
the episode return is minus the total slowdown of admitted jobs. Jobs arriving to
a full backlog are dropped (as in the original) and contribute nothing.

The episode terminates once all arrivals have happened and the system is empty,
and is truncated at `max_time` timesteps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from slurm_aware.deeprm.backend import ClusterBackend, InProcessBackend
from slurm_aware.deeprm.jobs import ClusterSpec, Job, check_jobset

VOID = 0
Policy = Callable[["DeepRMEnv"], int]  # any scheduler: observe the env, return an action


@dataclass(frozen=True)
class EnvConfig:
    num_slots: int = 10  # M
    backlog_size: int = 60
    lookahead: int = 20  # T: image width, and the longest placeable job
    max_time: int = 500  # truncation cap in timesteps (arrivals end at the generator horizon)

    def __post_init__(self) -> None:
        if min(self.num_slots, self.lookahead, self.max_time) < 1 or self.backlog_size < 0:
            raise ValueError("num_slots, lookahead and max_time must be >= 1; backlog >= 0.")


@dataclass
class JobRecord:
    job: Job
    start: int | None = None
    finish: int | None = None

    @property
    def slowdown(self) -> float | None:
        if self.finish is None:
            return None
        return (self.finish - self.job.arrival) / self.job.duration


class DeepRMEnv:
    def __init__(
        self,
        spec: ClusterSpec = ClusterSpec(),
        config: EnvConfig = EnvConfig(),
        backend: ClusterBackend | None = None,
        *,
        track_decisions: bool = False,
    ) -> None:
        self.spec, self.config = spec, config
        self.backend = backend or InProcessBackend(spec, config.lookahead)
        if self.backend.spec != spec or self.backend.horizon != config.lookahead:
            raise ValueError("Backend spec/horizon must match the environment's.")
        self.track_decisions = track_decisions
        self.n_actions = config.num_slots + 1
        self._image_size = sum(spec.capacities) * config.lookahead
        self.observation_size = (1 + config.num_slots) * self._image_size + config.backlog_size
        self.jobs: list[Job] = []
        self.done = True
        self.truncated = False

    # -- Gym-style API -------------------------------------------------------------

    def reset(self, jobs: Sequence[Job] | None = None) -> tuple[np.ndarray, dict]:
        """Start an episode on `jobs` (or replay the previous jobset)."""
        if jobs is not None:
            check_jobset(jobs, self.spec)
            if any(job.duration > self.config.lookahead for job in jobs):
                raise ValueError("Job durations must not exceed the lookahead T.")
            self.jobs = sorted(jobs, key=lambda job: (job.arrival, job.job_id))
        self.backend.reset()
        self.time = 0
        self.slots: list[Job | None] = [None] * self.config.num_slots
        self.backlog: deque[Job] = deque()
        self.records = {job.job_id: JobRecord(job) for job in self.jobs}
        self.dropped: list[Job] = []
        self.decisions: list[dict] = []
        self._next_arrival = 0
        self._enqueue_arrivals()
        self.done = self._finished()
        self.truncated = False
        return self.observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be in [0, {self.n_actions}).")
        job = self.slots[action - 1] if action != VOID else None
        if job is not None and self.backend.allocate(job, self.time):
            self.records[job.job_id].start = self.time
            self.slots[action - 1] = self.backlog.popleft() if self.backlog else None
            if self.track_decisions:
                self._log_decision(action, scheduled=job)
            return self.observation(), 0.0, False, False, self._info(scheduled=job)

        if self.track_decisions:
            self._log_decision(action, scheduled=None)
        reward = -sum(1.0 / j.duration for j in self.jobs_in_system())
        for finished, _start, finish in self.backend.advance():
            self.records[finished.job_id].finish = finish
        self.time += 1
        self._enqueue_arrivals()
        terminated = self._finished()
        truncated = not terminated and self.time >= self.config.max_time
        self.done, self.truncated = terminated or truncated, truncated
        return self.observation(), reward, terminated, truncated, self._info()

    # -- State -----------------------------------------------------------------------

    def observation(self) -> np.ndarray:
        cfg, T = self.config, self.config.lookahead
        obs = np.zeros(self.observation_size, dtype=np.float32)
        offset = 0
        for image in self.backend.canvas():
            obs[offset : offset + image.size] = image.ravel()
            offset += image.size
        for job in self.slots:
            for units, capacity in zip(job.demand if job else (), self.spec.capacities):
                image = obs[offset : offset + capacity * T].reshape(capacity, T)
                image[:units, : job.duration] = 1.0
                offset += capacity * T
            if job is None:
                offset += self._image_size
        obs[offset : offset + min(len(self.backlog), cfg.backlog_size)] = 1.0
        return obs

    def image_shapes(self) -> list[tuple[str, int, int]]:
        """(name, rows, cols) of each image in observation order, for reshaping/plots."""
        T = self.config.lookahead
        dims = list(zip(self.spec.resources, self.spec.capacities))
        prefixes = ["cluster"] + [f"slot{m + 1}" for m in range(self.config.num_slots)]
        return [(f"{prefix}/{r}", c, T) for prefix in prefixes for r, c in dims]

    def jobs_in_system(self) -> list[Job]:
        waiting = [job for job in self.slots if job is not None]
        return self.backend.running() + waiting + list(self.backlog)

    def fitting_slots(self) -> list[int]:
        """Actions (1..M) whose job could start now."""
        return [
            m + 1
            for m, job in enumerate(self.slots)
            if job is not None and self.backend.fits(job)
        ]

    def slowdowns(self, include_unfinished: bool = False) -> list[float]:
        """Admitted jobs' slowdowns; unfinished ones optionally as (now - arrival) / duration,
        a lower bound, so truncated episodes are not flattered by leaving them out."""
        values = []
        for record in self.records.values():
            if record.slowdown is not None:
                values.append(record.slowdown)
            elif include_unfinished and record.job.arrival <= self.time:
                values.append((self.time - record.job.arrival) / record.job.duration)
        return values

    # -- Internals -------------------------------------------------------------------

    def _enqueue_arrivals(self) -> None:
        while (
            self._next_arrival < len(self.jobs)
            and self.jobs[self._next_arrival].arrival <= self.time
        ):
            job = self.jobs[self._next_arrival]
            self._next_arrival += 1
            if None in self.slots:
                self.slots[self.slots.index(None)] = job
            elif len(self.backlog) < self.config.backlog_size:
                self.backlog.append(job)
            else:
                self.dropped.append(job)
                del self.records[job.job_id]

    def _finished(self) -> bool:
        return self._next_arrival == len(self.jobs) and not self.jobs_in_system()

    def _log_decision(self, action: int, scheduled: Job | None) -> None:
        fitting = self.fitting_slots() if scheduled is None else []
        self.decisions.append(
            {
                "time": self.time,
                "action": action,
                "scheduled": scheduled.job_id if scheduled else None,
                # Time advanced although a visible job could have started.
                "withheld": [self.slots[a - 1].job_id for a in fitting],
                "waiting": [job.job_id for job in self.slots if job is not None]
                + [job.job_id for job in self.backlog],
            }
        )

    def _info(self, scheduled: Job | None = None) -> dict:
        return {
            "time": self.time,
            "scheduled": scheduled.job_id if scheduled else None,
            "backlog": len(self.backlog),
            "dropped": len(self.dropped),
        }


def run_episode(env: DeepRMEnv, policy: Policy, jobs: Sequence[Job] | None = None) -> dict:
    """Play one episode; return its return, slowdowns and per-job records."""
    env.reset(jobs)
    total, steps = 0.0, 0
    truncated = False
    while not env.done:
        _, reward, _, truncated, _ = env.step(policy(env))
        total += reward
        steps += 1
    slowdowns = env.slowdowns(include_unfinished=True)
    return {
        "return": total,
        "decisions": steps,
        "timesteps": env.time,
        "truncated": truncated,
        "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else 0.0,
        "slowdowns": slowdowns,
        "unfinished": sum(r.finish is None for r in env.records.values()),
        "dropped": len(env.dropped),
    }
