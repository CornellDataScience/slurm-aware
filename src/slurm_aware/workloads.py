"""CSV workload interchange and reproducible synthetic scheduling traces."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping, Sequence

from slurm_aware.simulator import Job, SimulationConfig, _integer, _name, _number, compatible_node

WORKLOAD_GENERATOR_VERSION = 2

INTEGER_COLUMNS = {
    "nodes",
    "cpus_per_node",
    "memory_mb_per_node",
    "gpus_per_node",
    "nice",
    "site_factor",
}
TIME_COLUMNS = {"submit_time", "runtime", "time_limit", "eligible_time"}
REQUIRED_COLUMNS = {"job_id", "submit_time", "runtime", "time_limit"}


def read_traces(path: str | Path) -> dict[str, list[Job]]:
    """Read seconds-based traces, rejecting unknown fields and invalid rows."""
    traces: dict[str, list[Job]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)):
            raise ValueError("The workload has duplicate column names.")
        missing = REQUIRED_COLUMNS - set(names)
        unknown = set(names) - {item.name for item in fields(Job)} - {"trace_id"}
        if missing or unknown:
            raise ValueError(
                f"Workload columns: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for row_number, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ValueError("More values than column names.")
                if any(not row.get(name, "") for name in REQUIRED_COLUMNS):
                    raise ValueError("Required job fields cannot be blank.")
                trace_id = row.pop("trace_id", "trace-0")
                _name("trace_id", trace_id)
                values = {key: value for key, value in row.items() if value not in (None, "")}
                for name in values.keys() & INTEGER_COLUMNS:
                    values[name] = int(values[name])
                for name in values.keys() & TIME_COLUMNS:
                    values[name] = float(values[name])
                if "allowed_nodes" in values:
                    values["allowed_nodes"] = tuple(values["allowed_nodes"].split(";"))
                traces.setdefault(trace_id, []).append(Job(**values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Workload row {row_number}: {exc}") from exc
    if not traces:
        raise ValueError("The workload contains no jobs.")
    return traces


def write_traces(path: str | Path, traces: Mapping[str, Sequence[Job]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trace_id", *[f.name for f in fields(Job)]])
        writer.writeheader()
        for trace_id, jobs in traces.items():
            for job in jobs:
                writer.writerow(
                    {
                        "trace_id": trace_id,
                        **asdict(job),
                        "allowed_nodes": ";".join(job.allowed_nodes),
                    }
                )


def generate_traces(
    config: SimulationConfig,
    *,
    traces: int = 5,
    jobs_per_trace: int = 200,
    seed: int = 42,
    offered_load: float = 0.85,
    arrival_pattern: str = "bursty",
) -> dict[str, list[Job]]:
    """Toy independent resource/duration mixtures; no runtime predictor is used.

    Offered load controls expected CPU work per arrival interval. It is not a
    promised achieved utilization, particularly for multi-node placement.
    """
    for name, value in (("traces", traces), ("jobs_per_trace", jobs_per_trace)):
        _integer(name, value, 1)
    _integer("seed", seed)
    _number("offered_load", offered_load, positive=True)
    if arrival_pattern not in {"steady", "bursty"}:
        raise ValueError("arrival_pattern must be steady or bursty.")
    rng = random.Random(seed)
    cluster = config.cluster
    available_nodes = [node for node in cluster.inventory if node.available_resources[0] > 0]
    result = {}
    for trace in range(traces):
        specs = []
        for _ in range(jobs_per_trace):
            runtime = round(max(0.1, rng.lognormvariate(math.log(15), 1.0)), 3)
            anchor = rng.choice(available_nodes)
            cpus, memory, gpus = anchor.available_resources
            gpu_request = rng.choice([0, gpus])
            spec = {
                "runtime": runtime,
                "time_limit": float(math.ceil(runtime * rng.uniform(1.1, 3.0))),
                "cpus_per_node": rng.randint(1, cpus),
                "memory_mb_per_node": rng.choice([0, memory // 2]),
                "gpus_per_node": gpu_request,
                "architecture": anchor.architecture,
                "gpu_type": anchor.gpu_type if gpu_request else None,
                "account": rng.choice(sorted(config.accounts)),
                "qos": rng.choice(sorted(config.qos_priorities)),
            }
            probe = Job(job_id="candidate", submit_time=0, **spec)
            compatible = [
                node
                for node in available_nodes
                if compatible_node(probe, node)
                and all(
                    request <= capacity
                    for request, capacity in zip(
                        (probe.cpus_per_node, probe.memory_mb_per_node, probe.gpus_per_node),
                        node.available_resources,
                    )
                )
            ]
            spec["nodes"] = rng.randint(1, len(compatible))
            specs.append(spec)
        mean_cpu_work = (
            sum(spec["runtime"] * spec["nodes"] * spec["cpus_per_node"] for spec in specs)
            / jobs_per_trace
        )
        mean_interval = mean_cpu_work / (cluster.available_cpus * offered_load)
        now = 0.0
        jobs = []
        for index, spec in enumerate(specs):
            if index:
                multiplier = 1.0
                if arrival_pattern == "bursty":
                    multiplier = 0.2 if (index // 20) % 2 == 0 else 1.8
                now += rng.expovariate(1 / mean_interval) * multiplier
            jobs.append(Job(job_id=f"job-{index:05}", submit_time=round(now, 6), **spec))
        result[f"trace-{trace:03}"] = jobs
    return result
