import json
from dataclasses import replace
from pathlib import Path

import pytest

from slurm_aware.sim_cli import config_dict, load_config, main
from slurm_aware.simulator import (
    Cluster,
    Job,
    Node,
    PriorityWeights,
    SimulationConfig,
    priority_factors,
    simulate,
)
from slurm_aware.workloads import generate_traces, read_traces, write_traces

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIFO = PriorityWeights(age=0, job_size=0)


def test_cds_inventory_preserves_known_and_unknown_capacity():
    config = load_config(EXAMPLES / "cds_cluster.json")
    cluster = config.cluster
    assert [node.name for node in cluster.inventory] == [f"compute{i}" for i in range(1, 5)]
    assert cluster.head_node == "orca"
    assert cluster.nodes == 4
    assert cluster.enabled_nodes == 3
    assert cluster.total_cpus == 56
    assert cluster.total_memory_mb == 55536
    assert cluster.total_gpus == 2
    assert cluster.inventory[1].gpus is None
    assert cluster.inventory[3].cpus is None
    assert cluster.inventory[3].available_resources == (0, 0, 0)
    assert config.partition_name == "compute"


def test_typed_gpu_jobs_wait_for_compute1_and_arm_jobs_use_compute3():
    config = load_config(EXAMPLES / "cds_cluster.json")
    jobs = [
        Job("gpu-first", 0, 10, 10, account="alice", gpus_per_node=2, gpu_type="rtx2080ti"),
        Job("gpu-next", 1, 1, 1, account="bob", gpus_per_node=1, gpu_type="rtx2080ti"),
        Job("arm", 1, 1, 1, account="alice", architecture="aarch64", cpus_per_node=4),
    ]
    runs = {run.job.job_id: run for run in simulate(jobs, FIFO, config).jobs}
    assert runs["gpu-first"].allocated_nodes == (0,)
    assert runs["gpu-next"].allocated_nodes == (0,)
    assert runs["gpu-next"].start_time == 10
    assert runs["arm"].allocated_nodes == (2,)
    assert runs["arm"].start_time == 1


def test_gpu_model_and_architecture_constraints_are_used_in_backfill_reservations():
    config = SimulationConfig(
        cluster=Cluster(
            node_specs=(
                Node("gpu-a", 2, 100, 1, "x86_64", "A"),
                Node("gpu-b", 2, 100, 1, "aarch64", "B"),
            )
        )
    )
    jobs = [
        Job("first", 0, 10, 10, cpus_per_node=2, gpus_per_node=1, gpu_type="A"),
        Job("head", 1, 2, 2, gpus_per_node=1, gpu_type="A"),
        Job("other", 2, 20, 20, architecture="aarch64", gpus_per_node=1, gpu_type="B"),
    ]
    runs = {run.job.job_id: run for run in simulate(jobs, FIFO, config).jobs}
    assert runs["head"].start_time == 10
    assert runs["other"].start_time == 2
    assert runs["other"].allocated_nodes == (1,)


def test_cpus_and_memory_are_checked_against_each_actual_node():
    config = SimulationConfig(
        cluster=Cluster(
            node_specs=(
                Node("small", 2, 100),
                Node("large", 8, 1000),
            )
        )
    )
    result = simulate(
        [Job("large-job", 0, 1, 1, cpus_per_node=6, memory_mb_per_node=500)], FIFO, config
    )
    assert result.jobs[0].allocated_nodes == (1,)
    with pytest.raises(ValueError, match="cannot fit"):
        simulate([Job("two-large-nodes", 0, 1, 1, nodes=2, cpus_per_node=6)], FIFO, config)


def test_priority_normalizes_against_modeled_inventory_totals():
    config = load_config(EXAMPLES / "cds_cluster.json")
    job = Job(
        "request",
        0,
        1,
        1,
        account="alice",
        cpus_per_node=28,
        memory_mb_per_node=1000,
        gpus_per_node=1,
    )
    factors = priority_factors(job, 0, {}, config)
    assert factors["tres_cpu"] == 0.5
    assert factors["tres_memory"] == pytest.approx(1000 / 55536)
    assert factors["tres_gpu"] == 0.5
    assert factors["job_size"] == pytest.approx((1 / 3 + 0.5) / 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"allowed_nodes": ("orca",)},
        {"allowed_nodes": ("compute4",)},
        {"allowed_nodes": ("compute2",), "gpus_per_node": 1},
        {"architecture": "aarch64", "cpus_per_node": 5},
        {"gpu_type": "not-a-2080ti", "gpus_per_node": 1},
        {"nodes": 2, "gpus_per_node": 1},
    ],
)
def test_unavailable_nodes_and_incompatible_requests_fail_fast(changes):
    config = load_config(EXAMPLES / "cds_cluster.json")
    with pytest.raises(ValueError):
        simulate([Job("bad", 0, 1, 1, account="alice", **changes)], FIFO, config)


def test_explicit_node_affinity_is_respected_and_roundtrips_csv(tmp_path):
    config = load_config(EXAMPLES / "cds_cluster.json")
    jobs = [Job("pinned", 0, 1, 1, account="alice", allowed_nodes=("compute2",))]
    path = tmp_path / "workload.csv"
    write_traces(path, {"trace": jobs})
    assert read_traces(path)["trace"] == jobs
    assert simulate(jobs, FIFO, config).jobs[0].allocated_nodes == (1,)


def test_persistent_service_reservations_reduce_available_capacity_not_objective_jobs():
    config = SimulationConfig(
        cluster=Cluster(
            node_specs=(
                Node(
                    "compute1", 8, 100, 2, reserved_cpus=4, reserved_memory_mb=50, reserved_gpus=1
                ),
            )
        )
    )
    jobs = [
        Job(str(i), 0, 10, 10, cpus_per_node=4, memory_mb_per_node=50, gpus_per_node=1)
        for i in range(2)
    ]
    result = simulate(jobs, FIFO, config)
    assert [run.start_time for run in result.jobs] == [0, 10]
    assert result.metrics["completed_jobs"] == 2
    assert result.metrics["mean_response_time_seconds"] == 15
    assert result.metrics["cpu_utilization"] == result.metrics["gpu_utilization"] == 1
    with pytest.raises(ValueError, match="cannot fit"):
        simulate([replace(jobs[0], gpus_per_node=2)], FIFO, config)


def test_unknown_node_can_be_enabled_after_inventory_is_supplied():
    config = load_config(EXAMPLES / "cds_cluster.json")
    nodes = config.cluster.node_specs
    cluster = replace(config.cluster, node_specs=(*nodes[:3], Node("compute4", 6, 8000)))
    updated = replace(config, cluster=cluster)
    job = Job("fourth-node", 0, 1, 1, account="alice", allowed_nodes=("compute4",))
    assert simulate([job], FIFO, updated).jobs[0].allocated_nodes == (3,)
    assert cluster.enabled_nodes == 4


@pytest.mark.parametrize(
    "make_node",
    [
        lambda: Node("x", None, 100),
        lambda: Node("x", 1, None),
        lambda: Node("x", 1, 100, reserved_cpus=2),
        lambda: Node("x", 1, 100, None, reserved_gpus=1),
        lambda: Node("x", 1, 100, gpu_type="missing-gpu"),
    ],
)
def test_invalid_inventory_is_rejected(make_node):
    with pytest.raises(ValueError):
        make_node()


def test_head_node_and_duplicate_worker_names_are_rejected():
    with pytest.raises(ValueError, match="head/login"):
        Cluster(node_specs=(Node("orca", 4, 100),), head_node="orca")
    with pytest.raises(ValueError, match="unique"):
        Cluster(node_specs=(Node("same", 1, 100), Node("same", 2, 200)))


def test_cds_generator_produces_feasible_architecture_and_gpu_requests(tmp_path):
    config = load_config(EXAMPLES / "cds_cluster.json")
    traces = generate_traces(config, traces=2, jobs_per_trace=100, seed=7)
    path = tmp_path / "generated.csv"
    write_traces(path, traces)
    assert read_traces(path) == traces
    assert traces == generate_traces(config, traces=2, jobs_per_trace=100, seed=7)
    for jobs in traces.values():
        result = simulate(jobs, FIFO, config)
        assert len(result.jobs) == 100
        for run in result.jobs:
            for index in run.allocated_nodes:
                node = config.cluster.inventory[index]
                assert node.name != "orca" and node.name != "compute4"
                assert node.architecture == run.job.architecture
                assert run.job.gpus_per_node <= (node.gpus or 0)
                if run.job.gpus_per_node:
                    assert node.name == "compute1"
        for time in {r.start_time for r in result.jobs}:
            active = [r for r in result.jobs if r.start_time <= time < r.finish_time]
            for index, node in enumerate(config.cluster.inventory):
                allocated = [r.job for r in active if index in r.allocated_nodes]
                used = (
                    sum(j.cpus_per_node for j in allocated),
                    sum(j.memory_mb_per_node for j in allocated),
                    sum(j.gpus_per_node for j in allocated),
                )
                assert all(a <= b for a, b in zip(used, node.available_resources))


def test_report_uses_real_node_names_and_reloadable_configuration(tmp_path):
    main(
        [
            "evaluate",
            "--workload",
            str(EXAMPLES / "cds_jobs.csv"),
            "--config",
            str(EXAMPLES / "cds_cluster.json"),
            "--weights",
            str(EXAMPLES / "priority_weights.json"),
            "--output",
            str(tmp_path),
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["aggregate"]["completed_jobs"] == 8
    assert report["aggregate"]["gpu_utilization"] > 0
    assert "compute1" in (tmp_path / "schedule.csv").read_text()
    assert "node-1" not in (tmp_path / "schedule.csv").read_text()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(report["config"]))
    assert config_dict(load_config(config_path)) == config_dict(
        load_config(EXAMPLES / "cds_cluster.json")
    )
