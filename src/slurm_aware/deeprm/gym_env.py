"""Gymnasium interface for the event-driven DeepRM simulator."""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from slurm_aware.deeprm.backend import ClusterBackend
from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig
from slurm_aware.deeprm.generator import GeneratorConfig, JobTraceGenerator
from slurm_aware.deeprm.jobs import ClusterSpec, Job


class DeepRMGymEnv(gym.Env[np.ndarray, int]):
    """Generate a workload per reset, or replay fixed `jobs` when supplied.

    `reset(options={"jobs": jobs})` overrides the workload for one episode.
    Scheduling, observations and rewards come from `simulator` unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: ClusterSpec = ClusterSpec(),
        config: EnvConfig = EnvConfig(),
        backend: ClusterBackend | None = None,
        *,
        jobs: Sequence[Job] | None = None,
        generator: JobTraceGenerator | None = None,
        track_decisions: bool = False,
    ) -> None:
        super().__init__()
        if jobs is not None and generator is not None:
            raise ValueError("Supply fixed jobs or a generator, not both.")
        self.jobs = tuple(jobs) if jobs is not None else None
        self.generator = generator
        if self.jobs is None and self.generator is None:
            self.generator = JobTraceGenerator(
                GeneratorConfig(cluster=spec, max_duration=config.lookahead)
            )
        if self.generator is not None:
            if self.generator.config.cluster != spec:
                raise ValueError("Generator cluster must match the environment's.")
            if self.generator.config.max_duration > config.lookahead:
                raise ValueError("Generator max_duration must not exceed lookahead.")
        # Gymnasium reserves `spec` for registration metadata.
        self.simulator = DeepRMEnv(spec, config, backend, track_decisions=track_decisions)
        self.action_space = spaces.Discrete(self.simulator.n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.simulator.observation_size,), dtype=np.float32
        )
        self._episode_done = True

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        jobs = self.jobs
        if options is not None and "jobs" in options:
            jobs = options["jobs"]
        elif self.generator is not None:
            jobs = self.generator.jobset(int(self.np_random.integers(0, 2**32)))
        observation, self._info = self.simulator.reset(jobs)
        del self._info["scheduled"]  # no allocation on reset
        self._episode_done = False
        return observation, self._info.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must belong to {self.action_space}.")
        if self.simulator.done:
            # An empty workload still needs a terminal transition for RL libraries.
            self._episode_done = True
            return self.simulator.observation(), 0.0, True, False, self._info.copy()
        observation, reward, terminated, truncated, info = self.simulator.step(int(action))
        if info["scheduled"] is None:
            del info["scheduled"]  # vector envs mask missing keys, but cannot mix int and None
        self._episode_done = terminated or truncated
        return observation, reward, terminated, truncated, info


gym.register(
    id="SlurmAware-DeepRM-v0",
    entry_point="slurm_aware.deeprm.gym_env:DeepRMGymEnv",
)
