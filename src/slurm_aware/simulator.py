"""Deterministic scheduling experiments, not an emulator of slurmctld.

Time is simulated in seconds. Named, heterogeneous nodes are supported; the
generic fallback remains four homogeneous, two-CPU nodes.
See docs/simulator.md for the scheduling and priority approximations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import cached_property
from statistics import fmean
from typing import Mapping, Sequence

SIMULATOR_VERSION = 2


def _number(name: str, value: float, minimum: float = 0, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (positive and value == 0)
    ):
        qualifier = "positive" if positive else f">= {minimum}"
        raise ValueError(f"{name} must be finite and {qualifier}.")


def _integer(name: str, value: int, minimum: int = 0) -> None:
    _number(name, value, minimum)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number.")


def _name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


@dataclass(frozen=True)
class Node:
    name: str
    cpus: int | None
    memory_mb: int | None
    gpus: int | None = 0
    architecture: str | None = None
    gpu_type: str | None = None
    reserved_cpus: int = 0
    reserved_memory_mb: int = 0
    reserved_gpus: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        _name("node name", self.name)
        if not isinstance(self.enabled, bool):
            raise ValueError(f"{self.name}.enabled must be true or false.")
        for name in (
            "cpus",
            "memory_mb",
            "gpus",
            "reserved_cpus",
            "reserved_memory_mb",
            "reserved_gpus",
        ):
            if getattr(self, name) is None and (
                name == "gpus" or (not self.enabled and name in {"cpus", "memory_mb"})
            ):
                continue
            _integer(f"{self.name}.{name}", getattr(self, name), 1 if name == "cpus" else 0)
        for name in ("architecture", "gpu_type"):
            if getattr(self, name) is not None:
                _name(f"{self.name}.{name}", getattr(self, name))
        if self.gpu_type and not self.gpus:
            raise ValueError(f"{self.name}: gpu_type requires at least one GPU.")
        if any(
            reserved > (total or 0)
            for reserved, total in zip(
                (self.reserved_cpus, self.reserved_memory_mb, self.reserved_gpus),
                (self.cpus, self.memory_mb, self.gpus),
            )
        ):
            raise ValueError(f"{self.name}: background reservations exceed node capacity.")

    @property
    def available_resources(self) -> tuple[int, int, int]:
        if not self.enabled:
            return (0, 0, 0)
        return (
            self.cpus - self.reserved_cpus,
            self.memory_mb - self.reserved_memory_mb,
            (self.gpus or 0) - self.reserved_gpus,
        )


@dataclass(frozen=True)
class Cluster:
    nodes: int = 4
    cpus_per_node: int = 2
    memory_mb_per_node: int = 1024
    gpus_per_node: int = 0
    node_specs: tuple[Node, ...] = ()
    head_node: str | None = None

    def __post_init__(self) -> None:
        for name in ("nodes", "cpus_per_node", "memory_mb_per_node", "gpus_per_node"):
            _integer(name, getattr(self, name), 1 if name in {"nodes", "cpus_per_node"} else 0)
        if not isinstance(self.node_specs, tuple) or any(
            not isinstance(node, Node) for node in self.node_specs
        ):
            raise ValueError("node_specs must be a tuple of Node objects.")
        if self.node_specs:
            object.__setattr__(self, "nodes", len(self.node_specs))
        names = [node.name for node in self.inventory]
        if len(set(names)) != len(names):
            raise ValueError("Compute node names must be unique.")
        if self.head_node is not None:
            _name("head_node", self.head_node)
            if self.head_node in names:
                raise ValueError("The head/login node cannot also be a compute node.")
        if not self.available_cpus:
            raise ValueError("Background reservations leave no CPUs for submitted jobs.")

    @cached_property
    def inventory(self) -> tuple[Node, ...]:
        return self.node_specs or tuple(
            Node(f"node-{i + 1}", self.cpus_per_node, self.memory_mb_per_node, self.gpus_per_node)
            for i in range(self.nodes)
        )

    @property
    def total_cpus(self) -> int:
        return sum(node.cpus for node in self.inventory if node.enabled)

    @property
    def total_memory_mb(self) -> int:
        return sum(node.memory_mb for node in self.inventory if node.enabled)

    @property
    def total_gpus(self) -> int:
        return sum(node.gpus or 0 for node in self.inventory if node.enabled)

    @property
    def enabled_nodes(self) -> int:
        return sum(node.enabled for node in self.inventory)

    @property
    def available_cpus(self) -> int:
        return sum(node.available_resources[0] for node in self.inventory)

    @property
    def available_gpus(self) -> int:
        return sum(node.available_resources[2] for node in self.inventory)


@dataclass(frozen=True)
class Account:
    share: float = 1.0
    initial_usage_cpu_seconds: float = 0.0
    priority: float = 0.0

    def __post_init__(self) -> None:
        _number("account share", self.share, positive=True)
        _number("initial usage", self.initial_usage_cpu_seconds)
        _number("association priority", self.priority)


@dataclass(frozen=True)
class SimulationConfig:
    cluster: Cluster = field(default_factory=Cluster)
    scheduler: str = "easy_backfill"
    max_age_seconds: float = 604800.0
    favor_small: bool = False
    schedule_interval_seconds: float = 30.0
    fairshare_half_life_seconds: float | None = 604800.0
    accounts: dict[str, Account] = field(default_factory=lambda: {"default": Account()})
    qos_priorities: dict[str, float] = field(default_factory=lambda: {"normal": 1.0})
    profile_name: str = "Generic synthetic cluster"
    profile_notes: tuple[str, ...] = ()
    partition_name: str = "compute"

    def __post_init__(self) -> None:
        if not isinstance(self.cluster, Cluster):
            raise ValueError("cluster must be a Cluster.")
        _name("profile_name", self.profile_name)
        _name("partition_name", self.partition_name)
        if not isinstance(self.profile_notes, tuple):
            raise ValueError("profile_notes must be a tuple of strings.")
        for note in self.profile_notes:
            _name("profile note", note)
        if self.scheduler not in {"strict", "easy_backfill"}:
            raise ValueError("scheduler must be strict or easy_backfill.")
        if not isinstance(self.favor_small, bool):
            raise ValueError("favor_small must be true or false.")
        _number("max_age_seconds", self.max_age_seconds, positive=True)
        _number("schedule_interval_seconds", self.schedule_interval_seconds, positive=True)
        if self.fairshare_half_life_seconds is not None:
            _number("fairshare_half_life_seconds", self.fairshare_half_life_seconds, positive=True)
        if not self.accounts or not self.qos_priorities:
            raise ValueError("Configure at least one account and QoS.")
        for name, account in self.accounts.items():
            _name("account name", name)
            if not isinstance(account, Account):
                raise ValueError("Each account must be an Account.")
        for name, priority in self.qos_priorities.items():
            _name("QoS name", name)
            _number("QoS priority", priority)


@dataclass(frozen=True)
class PriorityWeights:
    age: int = 1000
    fairshare: int = 0
    job_size: int = 1000
    association: int = 0
    partition: int = 0
    qos: int = 0
    tres_cpu: float = 0
    tres_memory: float = 0
    tres_gpu: float = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name.startswith("tres_"):
                _number(name, value, -(2**32 - 1))
            else:
                _integer(name, value)
            if value > 2**32 - 1:
                raise ValueError(f"{name} exceeds the supported weight range.")


@dataclass(frozen=True)
class Job:
    job_id: str
    submit_time: float
    runtime: float
    time_limit: float
    nodes: int = 1
    cpus_per_node: int = 1
    memory_mb_per_node: int = 0
    gpus_per_node: int = 0
    account: str = "default"
    qos: str = "normal"
    eligible_time: float | None = None
    nice: int = 0
    site_factor: int = 0
    architecture: str | None = None
    gpu_type: str | None = None
    allowed_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("job_id", "account", "qos"):
            _name(name, getattr(self, name))
        _number("submit_time", self.submit_time)
        _number("runtime", self.runtime, positive=True)
        _number("time_limit", self.time_limit, positive=True)
        if self.time_limit < self.runtime:
            raise ValueError("runtime exceeds time_limit; timeouts are not modeled.")
        if self.eligible_time is not None:
            _number("eligible_time", self.eligible_time, self.submit_time)
        for name in ("nodes", "cpus_per_node", "memory_mb_per_node", "gpus_per_node"):
            _integer(name, getattr(self, name), 1 if name in {"nodes", "cpus_per_node"} else 0)
        for name in ("nice", "site_factor"):
            _integer(name, getattr(self, name), -(2**31 - 3))
            if getattr(self, name) > 2**31 - 3:
                raise ValueError(f"{name} exceeds the supported adjustment range.")
        for name in ("architecture", "gpu_type"):
            if getattr(self, name) is not None:
                _name(name, getattr(self, name))
        if self.gpu_type is not None and not self.gpus_per_node:
            raise ValueError("gpu_type requires gpus_per_node > 0.")
        if not isinstance(self.allowed_nodes, tuple):
            raise ValueError("allowed_nodes must be a tuple of node names.")
        for name in self.allowed_nodes:
            _name("allowed node", name)
        if len(set(self.allowed_nodes)) != len(self.allowed_nodes):
            raise ValueError("allowed_nodes contains duplicate names.")

    @property
    def eligible_at(self) -> float:
        return self.submit_time if self.eligible_time is None else self.eligible_time

    @property
    def total_cpus(self) -> int:
        return self.nodes * self.cpus_per_node


@dataclass(frozen=True)
class JobResult:
    job: Job
    start_time: float
    finish_time: float
    allocated_nodes: tuple[int, ...]
    priority_at_start: int
    factors_at_start: dict[str, float]

    @property
    def wait_time(self) -> float:
        return self.start_time - self.job.submit_time

    @property
    def response_time(self) -> float:
        return self.finish_time - self.job.submit_time


@dataclass(frozen=True)
class SimulationResult:
    jobs: tuple[JobResult, ...]
    metrics: dict[str, float | int | None]


def priority_factors(
    job: Job, now: float, usage: Mapping[str, float], config: SimulationConfig
) -> dict[str, float]:
    """Use only submission metadata, current time, and accumulated past usage."""
    cluster = config.cluster
    shares = math.fsum(account.share for account in config.accounts.values())
    total_usage = math.fsum(usage.values())
    normalized_usage = usage.get(job.account, 0.0) / total_usage if total_usage else 0.0
    share = config.accounts[job.account].share / shares
    size = (job.nodes / cluster.enabled_nodes + job.total_cpus / cluster.total_cpus) / 2
    max_assoc = max(account.priority for account in config.accounts.values())
    max_qos = max(config.qos_priorities.values())
    return {
        "age": min(1.0, max(0.0, now - job.eligible_at) / config.max_age_seconds),
        "fairshare": 2 ** (-normalized_usage / share),
        "job_size": 1 - size if config.favor_small else size,
        "association": config.accounts[job.account].priority / max_assoc if max_assoc else 0.0,
        "partition": 1.0,  # One shared partition: constant across all jobs.
        "qos": config.qos_priorities[job.qos] / max_qos if max_qos else 0.0,
        "tres_cpu": job.total_cpus / cluster.total_cpus,
        "tres_memory": (
            job.nodes * job.memory_mb_per_node / cluster.total_memory_mb
            if cluster.total_memory_mb
            else 0.0
        ),
        "tres_gpu": (
            job.nodes * job.gpus_per_node / cluster.total_gpus if cluster.total_gpus else 0.0
        ),
    }


def _priority(job: Job, factors: Mapping[str, float], weights: PriorityWeights) -> int:
    score = math.fsum(getattr(weights, name) * value for name, value in factors.items())
    score += job.site_factor - job.nice
    return max(1, min(2**32 - 1, int(score)))


def _resources(job: Job) -> tuple[int, int, int]:
    return job.cpus_per_node, job.memory_mb_per_node, job.gpus_per_node


def _free_resources(cluster: Cluster, running: Sequence[JobResult]) -> list[list[int]]:
    free = [list(node.available_resources) for node in cluster.inventory]
    for run in running:
        _consume(free, run.job, run.allocated_nodes)
    return free


def _consume(free: list[list[int]], job: Job, allocation: tuple[int, ...]) -> None:
    for node in allocation:
        for resource, amount in enumerate(_resources(job)):
            free[node][resource] -= amount


def compatible_node(job: Job, node: Node) -> bool:
    """Architecture/model/name constraints; None means an unrestricted job."""
    return (
        node.enabled
        and (not job.allowed_nodes or node.name in job.allowed_nodes)
        and (job.architecture is None or job.architecture == node.architecture)
        and (job.gpu_type is None or job.gpu_type == node.gpu_type)
    )


def _place(
    job: Job, free: list[list[int]], cluster: Cluster, excluded: tuple[int, ...] = ()
) -> tuple[int, ...] | None:
    available = [
        index
        for index, capacity in enumerate(free)
        if index not in excluded
        and compatible_node(job, cluster.inventory[index])
        and all(amount <= remaining for amount, remaining in zip(_resources(job), capacity))
    ]
    if len(available) >= job.nodes:
        return tuple(available[: int(job.nodes)])
    return None


def _reservation(
    job: Job, running: Sequence[JobResult], cluster: Cluster
) -> tuple[float, tuple[int, ...]]:
    """Reserve the first blocked job using requested limits, never true runtimes."""
    for when in sorted({run.start_time + run.job.time_limit for run in running}):
        remaining = [run for run in running if run.start_time + run.job.time_limit > when]
        allocation = _place(job, _free_resources(cluster, remaining), cluster)
        if allocation is not None:
            return when, allocation
    raise RuntimeError("A validated job could not be reserved on an empty cluster.")


def _advance_usage(
    usage: dict[str, float], running: Sequence[JobResult], elapsed: float, half_life: float | None
) -> None:
    # Exact continuous exponential decay/integration: independent of event frequency.
    if half_life is None:
        decay, effective_elapsed = 1.0, elapsed
    else:
        rate = math.log(2) / half_life
        decay = math.exp(-rate * elapsed)
        effective_elapsed = -math.expm1(-rate * elapsed) / rate
    for account in usage:
        usage[account] *= decay
    for run in running:
        usage[run.job.account] += run.job.total_cpus * effective_elapsed


def _validate_workload(jobs: Sequence[Job], config: SimulationConfig) -> None:
    if not jobs:
        raise ValueError("A workload must contain at least one job.")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("job_id must be unique within a trace.")
    cluster = config.cluster
    names = {node.name for node in cluster.inventory}
    for job in jobs:
        unknown_nodes = set(job.allowed_nodes) - names
        if unknown_nodes:
            raise ValueError(
                f"Job {job.job_id} requests unknown compute nodes: {sorted(unknown_nodes)}"
            )
        if _place(job, _free_resources(cluster, []), cluster) is None:
            raise ValueError(
                f"Job {job.job_id} cannot fit on compatible nodes after background reservations."
            )
        if job.account not in config.accounts:
            raise ValueError(f"Unknown account {job.account!r} in job {job.job_id}.")
        if job.qos not in config.qos_priorities:
            raise ValueError(f"Unknown QoS {job.qos!r} in job {job.job_id}.")


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _job_metrics(runs: Sequence[JobResult]) -> dict[str, float | int]:
    waits = [run.wait_time for run in runs]
    responses = [run.response_time for run in runs]
    slowdowns = [run.response_time / run.job.runtime for run in runs]
    return {
        "completed_jobs": len(runs),
        "mean_response_time_seconds": fmean(responses),
        "mean_wait_time_seconds": fmean(waits),
        "p95_response_time_seconds": _percentile(responses, 0.95),
        "p95_wait_time_seconds": _percentile(waits, 0.95),
        "max_wait_time_seconds": max(waits),
        "mean_slowdown": fmean(slowdowns),
        "p95_slowdown": _percentile(slowdowns, 0.95),
    }


def simulate(
    jobs: Sequence[Job],
    weights: PriorityWeights | None = None,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Replay a complete trace with fixed weights; every call starts fresh.

    Events are eligibility, actual completion, and periodic scheduling ticks.
    At equal timestamps all completions and arrivals precede dispatch. Ties in
    integer priority use submission time, then lexicographic job ID.
    """
    config = config or SimulationConfig()
    weights = weights or PriorityWeights()
    _validate_workload(jobs, config)
    future = sorted(jobs, key=lambda job: (job.eligible_at, job.submit_time, job.job_id))
    cursor = 0
    pending: list[Job] = []
    running: list[JobResult] = []
    results: list[JobResult] = []
    usage = {name: account.initial_usage_cpu_seconds for name, account in config.accounts.items()}
    now = 0.0

    while cursor < len(future) or pending or running:
        running = [run for run in running if run.finish_time > now]
        while cursor < len(future) and future[cursor].eligible_at <= now:
            pending.append(future[cursor])
            cursor += 1

        scored = []
        for job in pending:
            factors = priority_factors(job, now, usage, config)
            scored.append((_priority(job, factors, weights), job, factors))
        scored.sort(key=lambda item: (-item[0], item[1].submit_time, item[1].job_id))
        free = _free_resources(config.cluster, running)
        reservation: tuple[float, tuple[int, ...]] | None = None
        started = set()
        for priority, job, factors in scored:
            excluded = ()
            if reservation is not None and now + job.time_limit > reservation[0]:
                excluded = reservation[1]
            allocation = _place(job, free, config.cluster, excluded)
            if allocation is None:
                if config.scheduler == "strict":
                    break
                if reservation is None:
                    reservation = _reservation(job, running, config.cluster)
                continue
            finish = now + job.runtime
            if not math.isfinite(now + job.time_limit) or finish <= now:
                raise ValueError("Job times exceed supported floating-point precision/range.")
            run = JobResult(job, now, finish, allocation, priority, factors)
            running.append(run)
            results.append(run)
            started.add(job.job_id)
            _consume(free, job, allocation)
        pending = [job for job in pending if job.job_id not in started]

        events = [run.finish_time for run in running]
        if cursor < len(future):
            events.append(future[cursor].eligible_at)
        if pending:
            if not running:
                raise RuntimeError("Pending jobs made no progress on an empty cluster.")
            interval = config.schedule_interval_seconds
            next_tick = (math.floor(now / interval) + 1) * interval
            if next_tick <= now:
                raise ValueError("Scheduling interval is too small for the timestamp scale.")
            events.append(next_tick)
        if not events:
            break
        next_time = min(events)
        _advance_usage(usage, running, next_time - now, config.fairshare_half_life_seconds)
        now = next_time

    metrics = _job_metrics(results)
    span = max(run.finish_time for run in results) - min(job.submit_time for job in jobs)
    metrics["makespan_seconds"] = span
    metrics["throughput_jobs_per_second"] = len(results) / span
    metrics["cpu_utilization"] = math.fsum(
        run.job.total_cpus * run.job.runtime for run in results
    ) / (config.cluster.available_cpus * span)
    gpu_work = math.fsum(run.job.nodes * run.job.gpus_per_node * run.job.runtime for run in results)
    metrics["gpu_utilization"] = (
        gpu_work / (config.cluster.available_gpus * span) if config.cluster.available_gpus else None
    )
    return SimulationResult(tuple(results), metrics)


def evaluate(
    weights: PriorityWeights, jobs: Sequence[Job], config: SimulationConfig | None = None
) -> float:
    """The objective for a future optimizer: mean submission-to-completion time."""
    return float(simulate(jobs, weights, config).metrics["mean_response_time_seconds"])


def aggregate_metrics(results: Mapping[str, SimulationResult]) -> dict[str, float | int | None]:
    """Pool jobs across independent traces, weighting every job equally."""
    if not results:
        raise ValueError("At least one trace is required.")
    metrics = _job_metrics([run for result in results.values() for run in result.jobs])
    span = math.fsum(result.metrics["makespan_seconds"] for result in results.values())
    metrics["traces"] = len(results)
    metrics["total_trace_span_seconds"] = span
    metrics["throughput_jobs_per_second"] = metrics["completed_jobs"] / span
    metrics["cpu_utilization"] = (
        math.fsum(
            result.metrics["cpu_utilization"] * result.metrics["makespan_seconds"]
            for result in results.values()
        )
        / span
    )
    gpu_results = [
        result for result in results.values() if result.metrics["gpu_utilization"] is not None
    ]
    gpu_span = math.fsum(result.metrics["makespan_seconds"] for result in gpu_results)
    metrics["gpu_utilization"] = (
        math.fsum(
            result.metrics["gpu_utilization"] * result.metrics["makespan_seconds"]
            for result in gpu_results
        )
        / gpu_span
        if gpu_span
        else None
    )
    return metrics
