"""Compare full results and paired timings against a saved simulator.py.

Run from the repository root with an installed slurm-aware package:
    python scripts/benchmark_simulator.py --reference /path/to/old/simulator.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from slurm_aware import sim_cli, simulator
from slurm_aware.simulator import Account, Cluster, Job, Node, PriorityWeights, SimulationConfig
from slurm_aware.workloads import generate_traces, read_traces


def load_reference(path: Path):
    name = "slurm_aware_reference_simulator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical(result) -> str:
    # Compare every job, dispatch factor, allocation, timestamp, and metric.
    # JSON also distinguishes signed zero; no floating-point tolerance is used.
    return json.dumps(asdict(result), sort_keys=True, allow_nan=False, separators=(",", ":"))


def random_case(seed: int):
    rng = random.Random(seed)
    nodes = []
    for index in range(rng.randint(1, 8)):
        cpus, memory, gpus = rng.randint(1, 8), rng.randint(32, 256), rng.randint(0, 2)
        nodes.append(
            Node(
                f"worker-{index}",
                cpus,
                memory,
                gpus,
                rng.choice(["x86_64", "aarch64"]),
                rng.choice(["A", "B"]) if gpus else None,
                reserved_cpus=rng.randrange(cpus),
                reserved_memory_mb=rng.randint(0, memory // 4),
                reserved_gpus=rng.randint(0, gpus),
            )
        )
    if seed % 3 == 0:
        nodes.append(Node("unknown", None, None, None, enabled=False))
    config = SimulationConfig(
        cluster=Cluster(node_specs=tuple(nodes), head_node="login"),
        scheduler="strict" if seed % 2 else "easy_backfill",
        max_age_seconds=rng.choice([1, 10, 604800]),
        favor_small=rng.choice([False, True]),
        schedule_interval_seconds=rng.choice([0.1, 0.2, 0.3, 0.7, 1, 7, 30]),
        fairshare_half_life_seconds=rng.choice([None, 1, 30, 604800]),
        accounts={
            name: Account(
                share=rng.choice([0.5, 1, 3]),
                initial_usage_cpu_seconds=rng.choice([0, 1, 100]),
                priority=rng.choice([0, 1, 3]),
            )
            for name in ["alice", "bob", "carol"]
        },
        qos_priorities={"normal": 0, "urgent": 2},
    )
    jobs = generate_traces(config, traces=1, jobs_per_trace=rng.randint(8, 32), seed=seed)[
        "trace-000"
    ]
    adjusted = []
    for job in jobs:
        runtime = rng.choice([0.125, 0.25, 0.5, 1, 2, 4])
        submit = rng.choice([0, 0, 0.7, 1, 2, 3, 6, 10])
        compatible = [
            node.name
            for node in nodes
            if simulator.compatible_node(job, node)
            and all(
                requested <= available
                for requested, available in zip(
                    (job.cpus_per_node, job.memory_mb_per_node, job.gpus_per_node),
                    node.available_resources,
                )
            )
        ]
        adjusted.append(
            replace(
                job,
                submit_time=submit,
                eligible_time=submit + rng.choice([0, 0, 0.3, 2]),
                runtime=runtime,
                time_limit=runtime + rng.choice([0, 0.25, 1, 5]),
                nice=rng.randint(-2, 2),
                site_factor=rng.choice([-7, 0, 7]),
                allowed_nodes=tuple(rng.sample(compatible, job.nodes)) if seed % 3 else (),
            )
        )
    rng.shuffle(adjusted)
    weights = PriorityWeights(
        **{
            name: rng.choice([-1000.5, -1, 0, 0.25, 1000.5])
            if name.startswith("tres_")
            else rng.choice([0, 1, 50, 1000, 100000, 2**32 - 1])
            for name in asdict(PriorityWeights())
        }
    )
    return adjusted, weights, config


def check_equal(reference, jobs, weights, config, label: str) -> None:
    expected = canonical(reference.simulate(jobs, weights, config))
    actual = canonical(simulator.simulate(jobs, weights, config))
    if actual != expected:
        raise AssertionError(f"Full simulation result differs: {label}")


def check_cli(reference) -> int:
    checked = 0
    with tempfile.TemporaryDirectory(prefix="slurm-cli-equivalence-") as temporary:
        for profile, workload in [("cds_cluster", "cds_jobs"), ("scheduler", "scheduling_jobs")]:
            for command in ["evaluate", "compare"]:
                arguments = [
                    command,
                    "--config",
                    f"examples/{profile}.json",
                    "--workload",
                    f"examples/{workload}.csv",
                    "--weights",
                    "current=examples/priority_weights.json"
                    if command == "compare"
                    else "examples/priority_weights.json",
                ]
                if command == "compare":
                    arguments += ["--weights", "age_only=examples/age_only_weights.json"]
                directories = []
                for name, implementation in [("reference", reference), ("optimized", simulator)]:
                    output = Path(temporary) / profile / command / name
                    directories.append(output)
                    with patch.object(sim_cli, "simulate", implementation.simulate):
                        with redirect_stdout(io.StringIO()):
                            sim_cli.main([*arguments, "--output", str(output)])
                snapshots = [
                    {
                        path.relative_to(directory).as_posix(): path.read_bytes()
                        for path in directory.rglob("*")
                        if path.is_file()
                    }
                    for directory in directories
                ]
                if snapshots[0] != snapshots[1]:
                    raise AssertionError(f"CLI files differ: {profile}/{command}")
                checked += len(snapshots[0])
    return checked


def benchmark_cases():
    cds = sim_cli.load_config(Path("examples/cds_cluster.json"))
    weights = sim_cli.load_weights(Path("examples/priority_weights.json"))
    yield "CDS example (8 jobs)", read_traces("examples/cds_jobs.csv")["cds-example"], weights, cds
    for size in [200, 1000, 3000]:
        jobs = generate_traces(cds, traces=1, jobs_per_trace=size, seed=42)["trace-000"]
        yield f"CDS backfill ({size} jobs)", jobs, weights, cds
    strict = replace(cds, scheduler="strict")
    jobs = generate_traces(strict, traces=1, jobs_per_trace=1000, seed=42)["trace-000"]
    yield "CDS strict (1000 jobs)", jobs, weights, strict
    larger = replace(cds, cluster=Cluster(nodes=32, cpus_per_node=8, gpus_per_node=1))
    jobs = generate_traces(larger, traces=1, jobs_per_trace=1000, seed=42)["trace-000"]
    yield "32 nodes, backfill (1000 jobs)", jobs, weights, larger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulator-performance.json"))
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("--cases and --repeats must be positive")
    reference = load_reference(args.reference)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(simulator.__file__).read_bytes()).hexdigest(),
        "comparison": "Exact canonical JSON of all result fields; no numeric tolerance",
        "randomized_cases": args.cases,
        "randomized_seeds": [10000, 10000 + args.cases - 1],
        "repeats": args.repeats,
        "timing_method": "Warm both implementations; alternate run order; exclude generation/I/O",
        "benchmarks": [],
    }
    check_equal(
        reference,
        [Job("later", 2, 1, 1, account="alice")],
        PriorityWeights(fairshare=1000),
        SimulationConfig(
            accounts={
                "alice": Account(initial_usage_cpu_seconds=1e308),
                "bob": Account(initial_usage_cpu_seconds=1e308),
            },
            fairshare_half_life_seconds=1,
        ),
        "large initial usage decays before first eligibility",
    )
    report["delayed_history_equivalence"] = True
    for seed in range(10000, 10000 + args.cases):
        check_equal(reference, *random_case(seed), f"random seed {seed}")
        if (seed - 10000 + 1) % 100 == 0:
            print(f"Exact equivalence: {seed - 10000 + 1} randomized cases passed", flush=True)
    report["identical_cli_files"] = check_cli(reference)
    print(f"CLI equivalence: {report['identical_cli_files']} files byte-identical", flush=True)
    for name, jobs, weights, config in benchmark_cases():
        check_equal(reference, jobs, weights, config, name)
        samples = {"reference": [], "optimized": []}
        for index in range(args.repeats):
            implementations = [("reference", reference), ("optimized", simulator)]
            if index % 2:
                implementations.reverse()
            for label, implementation in implementations:
                start = time.perf_counter()
                implementation.simulate(jobs, weights, config)
                samples[label].append(time.perf_counter() - start)
        before = statistics.median(samples["reference"])
        after = statistics.median(samples["optimized"])
        report["benchmarks"].append(
            {
                "case": name,
                "jobs": len(jobs),
                "reference_median_seconds": before,
                "optimized_median_seconds": after,
                "speedup": before / after,
                "samples_seconds": samples,
            }
        )
        print(f"{name}: {before:.6f}s -> {after:.6f}s ({before / after:.2f}x)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
