"""Reproducible synthetic jobsets: typed resource bundles, heavy-tailed durations, arrival regimes.

Changes from the original DeepRM generator:

- Durations are log-normal per job type (no bimodal short/long split), rounded and
  clipped to `[1, max_duration]`; `max_duration` should equal the env's image
  lookahead T, since a job-slot image cannot show a longer job.
- Arrivals are a Poisson count per timestep whose rate follows a regime
  (`steady`, `bursty` square-wave windows, or `diurnal` sinusoid), not Bernoulli.
- Demands come from job types (e.g. GPU training) as fractions of each resource's
  capacity, with a shared "size" draw so a job's dimensions are correlated.
- `load` is bottleneck-based: the arrival rate is set so that the most loaded
  resource receives `load` x its capacity of work per timestep on average.

Every jobset is a pure function of (config, seed); splits derive their seeds from
one base seed via `numpy.random.SeedSequence`, so any trace can be regenerated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from slurm_aware.deeprm.jobs import ClusterSpec, Job, check_jobset

GENERATOR_VERSION = 1
REGIMES = ("steady", "bursty", "diurnal")
SPLITS = ("train", "val", "test")
# Monte Carlo draws used to estimate expected work per job when converting load to a rate.
_WORK_ESTIMATE_SAMPLES = 50_000


@dataclass(frozen=True)
class JobType:
    """A family of jobs: mixture weight, demand ranges and a duration distribution.

    `demand` maps resource name to a (low, high) fraction of that resource's
    capacity; omitted resources are not requested. Durations are
    log-normal with the given median (timesteps) and log-space sigma.
    """

    name: str
    weight: float
    demand: Mapping[str, tuple[float, float]] = field(hash=False)
    duration_median: float
    duration_sigma: float

    def __post_init__(self) -> None:
        if self.weight <= 0 or self.duration_median <= 0 or self.duration_sigma < 0:
            raise ValueError(f"{self.name}: weight and median must be positive, sigma >= 0.")
        for resource, (low, high) in self.demand.items():
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{self.name}.{resource}: need 0 <= low <= high <= 1.")
        if not any(low > 0 for low, _ in self.demand.values()):
            raise ValueError(f"{self.name}: at least one resource needs a positive lower bound.")


# Illustrative defaults, not fitted to any real cluster. Shorter-running types
# request less, so small jobs are also short and the tail comes from GPU training.
DEFAULT_JOB_TYPES = (
    JobType("small_interactive", 0.30, {"cpu": (0.05, 0.15), "mem": (0.05, 0.10)}, 1.5, 0.6),
    JobType("cpu_batch", 0.30, {"cpu": (0.25, 0.50), "mem": (0.05, 0.15)}, 4.0, 0.8),
    JobType("mem_analytics", 0.20, {"cpu": (0.10, 0.20), "mem": (0.25, 0.50)}, 5.0, 0.8),
    JobType(
        "gpu_train",
        0.20,
        {"cpu": (0.10, 0.20), "gpu": (0.25, 0.50), "mem": (0.20, 0.40)},
        8.0,
        0.9,
    ),
)


@dataclass(frozen=True)
class ArrivalConfig:
    """Time-varying arrival-rate multiplier; every regime averages to 1 over whole periods.

    - steady: constant rate
    - bursty: the first `duty` fraction of each `period` runs at `peak_ratio` times
      the rate of the rest (square wave)
    - diurnal: `1 + amplitude * sin(2 pi t / period)`
    """

    regime: str = "steady"
    period: int = 20
    duty: float = 0.25
    peak_ratio: float = 8.0
    amplitude: float = 0.8
    phase: int = 0

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}.")
        if self.period < 1 or not 0 < self.duty < 1 or self.peak_ratio < 1:
            raise ValueError("Need period >= 1, 0 < duty < 1 and peak_ratio >= 1.")
        if not 0 <= self.amplitude <= 1:
            raise ValueError("amplitude must be in [0, 1].")

    def multipliers(self, horizon: int) -> np.ndarray:
        t = np.arange(horizon) + self.phase
        if self.regime == "steady":
            return np.ones(horizon)
        if self.regime == "bursty":
            low = 1 / (self.duty * self.peak_ratio + 1 - self.duty)
            on = (t % self.period) < self.duty * self.period
            return np.where(on, self.peak_ratio * low, low)
        return 1 + self.amplitude * np.sin(2 * math.pi * t / self.period)


@dataclass(frozen=True)
class GeneratorConfig:
    cluster: ClusterSpec = ClusterSpec()
    job_types: tuple[JobType, ...] = DEFAULT_JOB_TYPES
    arrival: ArrivalConfig = ArrivalConfig()
    load: float = 0.7
    horizon: int = 50  # timesteps during which jobs arrive (original episode length)
    max_duration: int = 20  # clip; keep equal to the env's lookahead T
    size_correlation: float = 0.8  # 0 = independent dimensions, 1 = one shared size draw

    def __post_init__(self) -> None:
        if self.load <= 0 or self.horizon < 1 or self.max_duration < 1:
            raise ValueError("Need load > 0, horizon >= 1 and max_duration >= 1.")
        if not 0 <= self.size_correlation <= 1:
            raise ValueError("size_correlation must be in [0, 1].")
        if not self.job_types:
            raise ValueError("At least one job type is required.")
        unknown = {r for jt in self.job_types for r in jt.demand} - set(self.cluster.resources)
        if unknown:
            raise ValueError(f"Job types request unknown resources: {sorted(unknown)}")

    def to_dict(self) -> dict:
        return {
            "version": GENERATOR_VERSION,
            "cluster": self.cluster.to_dict(),
            "job_types": [
                {**asdict(jt), "demand": {k: list(v) for k, v in jt.demand.items()}}
                for jt in self.job_types
            ],
            "arrival": asdict(self.arrival),
            "load": self.load,
            "horizon": self.horizon,
            "max_duration": self.max_duration,
            "size_correlation": self.size_correlation,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> GeneratorConfig:
        if data.get("version") != GENERATOR_VERSION:
            raise ValueError(f"Expected generator config version {GENERATOR_VERSION}.")
        return cls(
            cluster=ClusterSpec.from_dict(data["cluster"]),
            job_types=tuple(
                JobType(**{**jt, "demand": {k: tuple(v) for k, v in jt["demand"].items()}})
                for jt in data["job_types"]
            ),
            arrival=ArrivalConfig(**data["arrival"]),
            load=data["load"],
            horizon=data["horizon"],
            max_duration=data["max_duration"],
            size_correlation=data["size_correlation"],
        )


def jobset_seed(seed: int, split: str, index: int) -> np.random.SeedSequence:
    """Independent stream per (base seed, split, jobset index); splits never overlap."""
    return np.random.SeedSequence(seed, spawn_key=(SPLITS.index(split), index))


class JobTraceGenerator:
    def __init__(self, config: GeneratorConfig = GeneratorConfig()) -> None:
        self.config = config
        weights = np.array([jt.weight for jt in config.job_types])
        self._type_p = weights / weights.sum()
        caps = np.array(config.cluster.capacities, dtype=float)
        self._caps = caps
        bounds = np.zeros((len(config.job_types), config.cluster.n_resources, 2))
        for i, jt in enumerate(config.job_types):
            for resource, (low, high) in jt.demand.items():
                bounds[i, config.cluster.resources.index(resource)] = (low, high)
        self._bounds = bounds

    def with_load(self, load: float) -> JobTraceGenerator:
        other = JobTraceGenerator(replace(self.config, load=load))
        if "expected_work" in self.__dict__:  # Load does not change per-job work.
            other.__dict__["expected_work"] = self.expected_work
        return other

    def sample_jobs(self, rng: np.random.Generator, arrivals: Sequence[int]) -> list[Job]:
        """Draw one job per arrival time; ids are positions in `arrivals`."""
        cfg = self.config
        n, n_res = len(arrivals), cfg.cluster.n_resources
        kinds = rng.choice(len(cfg.job_types), size=n, p=self._type_p)
        c = cfg.size_correlation
        size = c * rng.random((n, 1)) + (1 - c) * rng.random((n, n_res))
        low, high = self._bounds[kinds, :, 0], self._bounds[kinds, :, 1]
        fraction = low + size * (high - low)
        demand = np.where(fraction > 0, np.ceil(fraction * self._caps - 1e-9), 0)
        demand = np.minimum(demand, self._caps).astype(int)
        medians = np.array([jt.duration_median for jt in cfg.job_types])[kinds]
        sigmas = np.array([jt.duration_sigma for jt in cfg.job_types])[kinds]
        raw = rng.lognormal(np.log(medians), sigmas)
        durations = np.clip(np.rint(raw), 1, cfg.max_duration).astype(int)
        return [
            Job(i, int(arrivals[i]), int(durations[i]), tuple(demand[i].tolist()),
                cfg.job_types[kinds[i]].name)
            for i in range(n)
        ]

    @cached_property
    def expected_work(self) -> np.ndarray:
        """E[demand_r x duration] / capacity_r per job, estimated with a fixed private seed."""
        jobs = self.sample_jobs(np.random.default_rng(0), [0] * _WORK_ESTIMATE_SAMPLES)
        work = np.array([np.array(job.demand) * job.duration for job in jobs])
        return work.mean(axis=0) / self._caps

    @property
    def arrival_rate(self) -> float:
        """Mean jobs per timestep so the bottleneck resource sees `load` on average."""
        return self.config.load / float(self.expected_work.max())

    def jobset(self, seed: int | np.random.SeedSequence) -> list[Job]:
        rng = np.random.default_rng(seed)
        rates = self.arrival_rate * self.config.arrival.multipliers(self.config.horizon)
        arrivals = np.repeat(np.arange(self.config.horizon), rng.poisson(rates))
        jobs = self.sample_jobs(rng, arrivals.tolist())
        check_jobset(jobs, self.config.cluster)
        return jobs

    def split(self, split: str, count: int, seed: int) -> list[list[Job]]:
        return [self.jobset(jobset_seed(seed, split, i)) for i in range(count)]

    def splits(self, sizes: Mapping[str, int], seed: int) -> dict[str, list[list[Job]]]:
        return {name: self.split(name, n, seed) for name, n in sizes.items()}


def offered_load(jobs: Sequence[Job], spec: ClusterSpec, horizon: int) -> list[float]:
    """Realized work per resource per arrival timestep, as a fraction of capacity."""
    work = np.zeros(spec.n_resources)
    for job in jobs:
        work += np.array(job.demand) * job.duration
    return (work / (np.array(spec.capacities) * horizon)).tolist()


def save_jobsets(path: str | Path, jobsets: Mapping[str, Sequence[Sequence[Job]]], meta: Mapping):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "jobsets": {k: [[j.to_dict() for j in js] for js in v] for k, v in jobsets.items()},
    }
    path.write_text(json.dumps(payload) + "\n")


def load_jobsets(path: str | Path) -> tuple[dict[str, list[list[Job]]], dict]:
    payload = json.loads(Path(path).read_text())
    raw = payload.pop("jobsets")
    return {k: [[Job.from_dict(j) for j in js] for js in v] for k, v in raw.items()}, payload


def describe(jobsets: Sequence[Sequence[Job]], spec: ClusterSpec, horizon: int) -> dict:
    jobs = [job for js in jobsets for job in js]
    durations = np.array([job.duration for job in jobs]) if jobs else np.zeros(1)
    loads = np.array([offered_load(js, spec, horizon) for js in jobsets])
    types: dict[str, int] = {}
    for job in jobs:
        types[job.job_type] = types.get(job.job_type, 0) + 1
    return {
        "jobsets": len(jobsets),
        "mean_jobs_per_set": len(jobs) / max(1, len(jobsets)),
        "duration_quantiles": {
            q: float(np.quantile(durations, q / 100)) for q in (10, 50, 90, 99)
        },
        "mean_offered_load": dict(zip(spec.resources, loads.mean(axis=0).tolist())),
        "type_fractions": {k: v / max(1, len(jobs)) for k, v in sorted(types.items())},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate seeded DeepRM jobset splits as JSON.")
    parser.add_argument("--output", default="data/generated/deeprm/jobsets.json")
    parser.add_argument("--config", help="GeneratorConfig JSON (default: built-in defaults)")
    parser.add_argument("--load", type=float, help="Override bottleneck load (e.g. 0.7)")
    parser.add_argument("--regime", choices=REGIMES, help="Override the arrival regime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        config = (
            GeneratorConfig.from_dict(json.loads(Path(args.config).read_text()))
            if args.config
            else GeneratorConfig()
        )
        if args.load is not None:
            config = replace(config, load=args.load)
        if args.regime:
            config = replace(config, arrival=replace(config.arrival, regime=args.regime))
        generator = JobTraceGenerator(config)
        sizes = {"train": args.train, "val": args.val, "test": args.test}
        jobsets = generator.splits(sizes, args.seed)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    meta = {"generator": config.to_dict(), "seed": args.seed, "sizes": sizes}
    save_jobsets(args.output, jobsets, meta)
    summary = {
        "arrival_rate": generator.arrival_rate,
        **{k: describe(v, config.cluster, config.horizon) for k, v in jobsets.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
