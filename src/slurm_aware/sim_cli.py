"""Generate workloads and evaluate fixed priority vectors without running Docker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from slurm_aware.simulator import (
    SIMULATOR_VERSION,
    Account,
    Cluster,
    Node,
    PriorityWeights,
    SimulationConfig,
    SimulationResult,
    aggregate_metrics,
    simulate,
)
from slurm_aware.workloads import (
    WORKLOAD_GENERATOR_VERSION,
    generate_traces,
    read_traces,
    write_traces,
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def load_config(path: Path | None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()
    values = _object(path)
    try:
        if "cluster" in values:
            cluster = values["cluster"]
            if not isinstance(cluster, dict):
                raise ValueError("cluster must be an object.")
            if "node_specs" in cluster:
                if not isinstance(cluster["node_specs"], list) or not cluster["node_specs"]:
                    raise ValueError("node_specs must be a nonempty list of node definitions.")
                if set(cluster) & {"nodes", "cpus_per_node", "memory_mb_per_node", "gpus_per_node"}:
                    raise ValueError("Use node_specs or homogeneous capacities, not both.")
                cluster["node_specs"] = tuple(Node(**node) for node in cluster["node_specs"])
            values["cluster"] = Cluster(**cluster)
        if "profile_notes" in values:
            if not isinstance(values["profile_notes"], list):
                raise ValueError("profile_notes must be a list of strings.")
            values["profile_notes"] = tuple(values["profile_notes"])
        if "accounts" in values:
            if not isinstance(values["accounts"], dict):
                raise ValueError("accounts must be an object keyed by account name.")
            values["accounts"] = {
                name: Account(**account) for name, account in values["accounts"].items()
            }
        if "qos_priorities" in values and not isinstance(values["qos_priorities"], dict):
            raise ValueError("qos_priorities must be an object keyed by QoS name.")
        return SimulationConfig(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid simulator configuration: {exc}") from exc


def load_weights(path: Path | None) -> PriorityWeights:
    if path is None:
        return PriorityWeights()
    try:
        return PriorityWeights(**_object(path))
    except TypeError as exc:
        raise ValueError(f"Invalid priority weights: {exc}") from exc


def config_dict(config: SimulationConfig) -> dict:
    """Serialize effective capacities without contradictory homogeneous defaults."""
    result = asdict(config)
    if config.cluster.node_specs:
        result["cluster"] = {
            "node_specs": result["cluster"]["node_specs"],
            "head_node": config.cluster.head_node,
        }
    else:
        result["cluster"].pop("node_specs")
    return result


def _write_results(
    output: Path,
    results: dict[str, SimulationResult],
    config: SimulationConfig,
    weights: PriorityWeights,
    workload: Path,
) -> dict:
    report = {
        "simulator_version": SIMULATOR_VERSION,
        "source": "simulation; no Slurm jobs were executed",
        "objective": "job-weighted mean submission-to-completion time in seconds",
        "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "config": config_dict(config),
        "weights": asdict(weights),
        "aggregate": aggregate_metrics(results),
        "traces": {trace: result.metrics for trace, result in results.items()},
        "limitations": [
            "One shared partition; no preemption, failures, or timeouts.",
            "Runtime is fixed across compatible nodes; no CPU/GPU speed or NFS contention model.",
            "Unknown GPUs and disabled nodes contribute no schedulable capacity.",
            "EASY backfill protects only the first blocked job, unlike Slurm's full backfill.",
            "Flat classic fair share with continuous usage decay, not Slurm Fair Tree.",
            "Event/tick dispatch has no Slurm daemon overhead; validate against Slurm before use.",
        ],
    }
    rows = []
    for trace, result in results.items():
        for run in result.jobs:
            rows.append(
                {
                    "trace_id": trace,
                    **asdict(run.job),
                    "allowed_nodes": ";".join(run.job.allowed_nodes),
                    "start_time": run.start_time,
                    "finish_time": run.finish_time,
                    "wait_time": run.wait_time,
                    "response_time": run.response_time,
                    "allocated_nodes": ";".join(
                        config.cluster.inventory[node].name for node in run.allocated_nodes
                    ),
                    "priority_at_start": run.priority_at_start,
                    **{f"factor_{name}": value for name, value in run.factors_at_start.items()},
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "schedule.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed Slurm-style priority weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate independent synthetic workload traces."
    )
    generate.add_argument("--output", type=Path, default=Path("data/generated/scheduling.csv"))
    generate.add_argument("--config", type=Path)
    generate.add_argument("--traces", type=int, default=5)
    generate.add_argument("--jobs", type=int, default=200, help="Jobs per trace.")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--load", type=float, default=0.85, help="Target offered CPU load.")
    generate.add_argument("--arrival-pattern", choices=["steady", "bursty"], default="bursty")
    evaluate = commands.add_parser(
        "evaluate", help="Replay traces and write metrics and schedules."
    )
    evaluate.add_argument("--workload", type=Path, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--weights", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/simulator"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        print(f"Profile: {config.profile_name}")
        for note in config.profile_notes:
            print(f"  {note}")
        if args.command == "generate":
            metadata_path = args.output.with_suffix(".metadata.json")
            if args.output.resolve() == metadata_path.resolve():
                raise ValueError("Choose a workload output distinct from its .metadata.json file.")
            if args.config and args.config.resolve() in {
                args.output.resolve(),
                metadata_path.resolve(),
            }:
                raise ValueError("Output paths must not overwrite the configuration.")
            traces = generate_traces(
                config,
                traces=args.traces,
                jobs_per_trace=args.jobs,
                seed=args.seed,
                offered_load=args.load,
                arrival_pattern=args.arrival_pattern,
            )
            write_traces(args.output, traces)
            metadata = {
                "source": "synthetic scheduling workload; no measured cluster runtimes",
                "generator_version": WORKLOAD_GENERATOR_VERSION,
                "seed": args.seed,
                "traces": args.traces,
                "jobs_per_trace": args.jobs,
                "offered_load": args.load,
                "arrival_pattern": args.arrival_pattern,
                "config": config_dict(config),
                "workload_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Generated {args.traces} traces / {args.traces * args.jobs} jobs: {args.output}")
            return

        output_paths = {(args.output / name).resolve() for name in ("report.json", "schedule.csv")}
        for source in (args.workload, args.config, args.weights):
            if source and source.resolve() in output_paths:
                raise ValueError("Output paths must not overwrite workload/configuration/weights.")
        weights = load_weights(args.weights)
        traces = read_traces(args.workload)
        results = {name: simulate(jobs, weights, config) for name, jobs in traces.items()}
        report = _write_results(args.output, results, config, weights, args.workload)
        metrics = report["aggregate"]
        print(f"Simulated {metrics['completed_jobs']} jobs across {metrics['traces']} traces")
        cluster = config.cluster
        print(
            f"Cluster: {cluster.nodes} compute nodes ({cluster.enabled_nodes} enabled), "
            f"{cluster.available_cpus} available CPU slots, {cluster.available_gpus} available GPUs"
        )
        print(f"Mean response time: {metrics['mean_response_time_seconds']:.3f} seconds")
        print(f"Mean wait time:     {metrics['mean_wait_time_seconds']:.3f} seconds")
        print(f"P95 wait time:      {metrics['p95_wait_time_seconds']:.3f} seconds")
        print(f"CPU utilization:   {metrics['cpu_utilization']:.1%}")
        if metrics["gpu_utilization"] is not None:
            print(f"GPU utilization:   {metrics['gpu_utilization']:.1%}")
        print(f"Report and per-job schedule: {args.output}")
        print("Simulation results; validate against Slurm before deploying weights.")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
