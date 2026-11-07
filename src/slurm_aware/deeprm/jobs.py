"""Job and cluster descriptions shared by the generator, backend, environment and baselines.

Time is in whole timesteps and resources are in whole units of one pooled cluster
(no machine boundaries), as in the original DeepRM model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ClusterSpec:
    """Named resource dimensions and their pooled capacity in units.

    Capacities may differ per dimension, so the same code models CPU-rich,
    GPU-scarce or memory-bound clusters; nothing assumes three resources.
    """

    resources: tuple[str, ...] = ("cpu", "gpu", "mem")
    capacities: tuple[int, ...] = (10, 4, 16)

    def __post_init__(self) -> None:
        if not self.resources or len(self.resources) != len(self.capacities):
            raise ValueError("resources and capacities must be nonempty and the same length.")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("Resource names must be unique.")
        if any(not isinstance(c, int) or c < 1 for c in self.capacities):
            raise ValueError("Capacities must be positive integers.")

    @property
    def n_resources(self) -> int:
        return len(self.resources)

    def to_dict(self) -> dict:
        return {"resources": list(self.resources), "capacities": list(self.capacities)}

    @classmethod
    def from_dict(cls, data: Mapping) -> ClusterSpec:
        return cls(tuple(data["resources"]), tuple(int(c) for c in data["capacities"]))


@dataclass(frozen=True)
class Job:
    job_id: int
    arrival: int
    duration: int
    demand: tuple[int, ...]  # units per resource, ordered like ClusterSpec.resources
    job_type: str = ""

    def __post_init__(self) -> None:
        if self.arrival < 0 or self.duration < 1:
            raise ValueError(f"Job {self.job_id}: arrival must be >= 0 and duration >= 1.")
        if any(d < 0 for d in self.demand) or not any(self.demand):
            raise ValueError(f"Job {self.job_id}: demand must be nonnegative and not all zero.")

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "arrival": self.arrival,
            "duration": self.duration,
            "demand": list(self.demand),
            "job_type": self.job_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Job:
        return cls(
            int(data["job_id"]),
            int(data["arrival"]),
            int(data["duration"]),
            tuple(int(d) for d in data["demand"]),
            data.get("job_type", ""),
        )


def check_jobset(jobs: Sequence[Job], spec: ClusterSpec) -> None:
    """Every job must fit an empty cluster, or no policy could ever schedule it."""
    for job in jobs:
        if len(job.demand) != spec.n_resources:
            raise ValueError(f"Job {job.job_id} has {len(job.demand)} demands, expected "
                             f"{spec.n_resources}.")
        if any(d > c for d, c in zip(job.demand, spec.capacities)):
            raise ValueError(f"Job {job.job_id} demand {job.demand} exceeds {spec.capacities}.")
