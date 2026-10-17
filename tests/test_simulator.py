import json
from dataclasses import asdict, replace

import pytest

from slurm_aware.sim_cli import load_config, load_weights, main
from slurm_aware.simulator import (
    Account,
    Cluster,
    Job,
    PriorityWeights,
    SimulationConfig,
    aggregate_metrics,
    evaluate,
    priority_factors,
    simulate,
)
from slurm_aware.workloads import generate_traces, read_traces, write_traces


def weights(**values):
    return PriorityWeights(**{**dict.fromkeys(asdict(PriorityWeights()), 0), **values})


def job(name, submit=0, runtime=1, limit=None, **values):
    return Job(name, submit, runtime, runtime if limit is None else limit, **values)


def starts(result):
    return {run.job.job_id: run.start_time for run in result.jobs}


def test_hand_calculated_response_time_and_priority_change():
    config = SimulationConfig(
        cluster=Cluster(nodes=1, cpus_per_node=1), qos_priorities={"normal": 0, "urgent": 1}
    )
    jobs = [job("a", runtime=10), job("b", runtime=1, qos="urgent")]
    fifo = simulate(jobs, weights(), config)
    assert starts(fifo) == {"a": 0, "b": 10}
    assert fifo.metrics["mean_response_time_seconds"] == 10.5
    assert fifo.metrics["mean_wait_time_seconds"] == 5
    assert evaluate(weights(qos=1000), jobs, config) == 6
    assert fifo.metrics["cpu_utilization"] == 1


def test_default_cluster_has_four_nodes_and_eight_cpu_slots():
    jobs = [job(str(i), runtime=10, cpus_per_node=2) for i in range(8)]
    result = simulate(jobs, weights())
    assert [run.start_time for run in result.jobs] == [0] * 4 + [10] * 4
    assert {run.allocated_nodes for run in result.jobs[:4]} == {(0,), (1,), (2,), (3,)}
    assert result.metrics["mean_response_time_seconds"] == 15
    assert result.metrics["cpu_utilization"] == 1


def test_free_cpus_on_different_nodes_cannot_be_combined():
    jobs = [job("a", runtime=10, nodes=4), job("b", runtime=1, cpus_per_node=2)]
    result = simulate(jobs, weights())
    assert result.jobs[0].allocated_nodes == (0, 1, 2, 3)
    assert starts(result)["b"] == 10  # Four CPUs free, but only one per node.


@pytest.mark.parametrize("resource,amount", [("memory_mb_per_node", 3), ("gpus_per_node", 1)])
def test_memory_and_gpu_capacity_are_enforced(resource, amount):
    config = SimulationConfig(
        cluster=Cluster(nodes=1, cpus_per_node=4, memory_mb_per_node=4, gpus_per_node=1)
    )
    jobs = [job("a", runtime=10, **{resource: amount}), job("b", **{resource: amount})]
    assert starts(simulate(jobs, weights(), config))["b"] == 10


def test_backfill_fills_a_gap_without_delaying_the_head_job():
    config = SimulationConfig(cluster=Cluster(nodes=2), schedule_interval_seconds=1)
    jobs = [
        job("running", runtime=10, cpus_per_node=2),
        job("head", submit=1, runtime=4, nodes=2, cpus_per_node=2),
        job("small", submit=2, runtime=3, cpus_per_node=2),
    ]
    backfill = simulate(jobs, weights(), config)
    strict = simulate(jobs, weights(), replace(config, scheduler="strict"))
    assert starts(backfill) == {"running": 0, "small": 2, "head": 10}
    assert starts(strict) == {"running": 0, "head": 10, "small": 14}


def test_backfill_uses_requested_limit_instead_of_knowing_true_duration():
    config = SimulationConfig(cluster=Cluster(nodes=2), schedule_interval_seconds=1)
    jobs = [
        job("running", runtime=10, cpus_per_node=2),
        job("head", submit=1, runtime=4, nodes=2, cpus_per_node=2),
        job("small", submit=2, runtime=1, limit=20, cpus_per_node=2),
    ]
    assert starts(simulate(jobs, weights(), config))["small"] == 14
    jobs[-1] = replace(jobs[-1], time_limit=1)
    assert starts(simulate(jobs, weights(), config))["small"] == 2


def test_backfill_can_use_other_nodes_beyond_the_reserved_start():
    config = SimulationConfig(cluster=Cluster(nodes=2), schedule_interval_seconds=1)
    jobs = [
        job("a", runtime=10, cpus_per_node=2),
        job("b", runtime=30),
        job("head", submit=1, runtime=4, cpus_per_node=2),
        job("small", submit=2, runtime=20),
    ]
    result = simulate(jobs, weights(), config)
    assert starts(result)["small"] == 2
    assert starts(result)["head"] == 10
    assert next(run for run in result.jobs if run.job.job_id == "small").allocated_nodes == (1,)


def test_age_and_size_weights_change_scheduling_decisions():
    config = SimulationConfig(
        cluster=Cluster(nodes=1), favor_small=True, max_age_seconds=10, scheduler="strict"
    )
    jobs = [
        job("blocker", runtime=10, cpus_per_node=2),
        job("old", submit=1, runtime=10, cpus_per_node=2),
        job("young", submit=9, runtime=1),
    ]
    aged = simulate(jobs, weights(age=1000), config)
    sized = simulate(jobs, weights(job_size=1000), config)
    assert starts(aged)["old"] == 10
    assert starts(sized)["young"] == 10
    assert aged.metrics["mean_response_time_seconds"] > sized.metrics["mean_response_time_seconds"]


def test_eligible_age_differs_from_submission_wait():
    config = SimulationConfig(max_age_seconds=10)
    delayed = job("held-until-five", submit=1, eligible_time=5, runtime=2)
    assert priority_factors(delayed, 5, {"default": 0}, config)["age"] == 0
    assert priority_factors(delayed, 10, {"default": 0}, config)["age"] == 0.5
    result = simulate([delayed], config=config)
    assert result.jobs[0].start_time == 5
    assert result.jobs[0].wait_time == 4
    assert result.jobs[0].response_time == 6


def test_completions_and_arrivals_at_same_time_precede_dispatch():
    config = SimulationConfig(
        cluster=Cluster(nodes=1, cpus_per_node=1), qos_priorities={"normal": 0, "urgent": 1}
    )
    jobs = [
        job("running", runtime=5),
        job("old", submit=1),
        job("new", submit=5, qos="urgent"),
    ]
    assert starts(simulate(jobs, weights(qos=1000), config)) == {"running": 0, "new": 5, "old": 6}


def test_fairshare_changes_after_consuming_resources_and_resets_between_replays():
    config = SimulationConfig(
        cluster=Cluster(nodes=1, cpus_per_node=1),
        accounts={"alice": Account(), "bob": Account()},
        fairshare_half_life_seconds=10,
        schedule_interval_seconds=1,
    )
    jobs = [
        job("initial", runtime=10, account="alice"),
        job("a", submit=1, account="alice"),
        job("b", submit=1, account="bob"),
    ]
    first = simulate(jobs, weights(fairshare=1000), config)
    assert starts(first) == {"initial": 0, "b": 10, "a": 11}
    assert first == simulate(jobs, weights(fairshare=1000), config)
    assert first == simulate(
        jobs, weights(fairshare=1000), replace(config, schedule_interval_seconds=7)
    )
    assert config.accounts["alice"].initial_usage_cpu_seconds == 0


def test_initial_account_history_influences_priority():
    config = SimulationConfig(
        cluster=Cluster(nodes=1, cpus_per_node=1),
        accounts={"alice": Account(initial_usage_cpu_seconds=100), "bob": Account()},
    )
    jobs = [job("a", account="alice"), job("b", account="bob")]
    assert starts(simulate(jobs, weights(fairshare=1000), config)) == {"b": 0, "a": 1}


def test_other_factors_are_normalized_and_runtime_is_not_a_priority_feature():
    config = SimulationConfig(
        cluster=Cluster(nodes=4, cpus_per_node=2, memory_mb_per_node=100, gpus_per_node=1),
        accounts={"default": Account(priority=2), "other": Account(priority=4)},
        qos_priorities={"normal": 2, "urgent": 4},
    )
    candidate = job("test", runtime=1, limit=100, nodes=2, memory_mb_per_node=50, gpus_per_node=1)
    factors = priority_factors(candidate, 0, {"default": 0, "other": 0}, config)
    assert factors["job_size"] == 0.375
    assert factors["association"] == factors["qos"] == 0.5
    assert factors["tres_cpu"] == factors["tres_memory"] == 0.25
    assert factors["tres_gpu"] == 0.5
    assert factors == priority_factors(replace(candidate, runtime=99), 0, {}, config)


def test_row_order_does_not_change_tied_priority_decisions():
    jobs = [job(str(i), runtime=10, cpus_per_node=2) for i in range(10)]
    assert simulate(jobs, weights()) == simulate(list(reversed(jobs)), weights())


def test_aggregate_objective_weights_jobs_not_traces_equally():
    config = SimulationConfig(cluster=Cluster(nodes=1, cpus_per_node=1))
    results = {
        "short": simulate([job("a")], weights(), config),
        "long": simulate([job(str(i), runtime=3) for i in range(3)], weights(), config),
    }
    metrics = aggregate_metrics(results)
    assert metrics["completed_jobs"] == 4
    assert metrics["mean_response_time_seconds"] == (1 + 3 + 6 + 9) / 4
    assert metrics["cpu_utilization"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": 5},
        {"cpus_per_node": 3},
        {"memory_mb_per_node": 1025},
        {"gpus_per_node": 1},
        {"account": "unknown"},
        {"qos": "unknown"},
    ],
)
def test_infeasible_or_unknown_job_requests_fail_before_simulation(changes):
    with pytest.raises(ValueError):
        simulate([job("invalid", **changes)])


@pytest.mark.parametrize(
    "changes",
    [
        {"runtime": 0},
        {"runtime": float("nan")},
        {"time_limit": 0.5},
        {"submit_time": -1},
        {"nodes": 1.5},
        {"nodes": 1.0},
        {"nodes": True},
        {"eligible_time": -1},
        {"runtime": float("inf")},
    ],
)
def test_invalid_job_fields_are_rejected(changes):
    with pytest.raises(ValueError):
        Job(**{**asdict(job("invalid")), **changes})


def test_empty_and_duplicate_jobs_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        simulate([])
    with pytest.raises(ValueError, match="unique"):
        simulate([job("same"), job("same")])
    with pytest.raises(ValueError):
        PriorityWeights(age=-1)


def test_generated_workloads_are_reproducible_and_resource_safe(tmp_path):
    config = SimulationConfig(
        cluster=Cluster(gpus_per_node=1), accounts={"alice": Account(), "bob": Account()}
    )
    traces = generate_traces(config, traces=2, jobs_per_trace=60, seed=7)
    assert traces == generate_traces(config, traces=2, jobs_per_trace=60, seed=7)
    assert traces != generate_traces(config, traces=2, jobs_per_trace=60, seed=8)
    path = tmp_path / "traces.csv"
    write_traces(path, traces)
    assert read_traces(path) == traces
    for scheduler in ("strict", "easy_backfill"):
        for jobs in traces.values():
            result = simulate(
                jobs,
                weights(age=1000, job_size=1000, fairshare=1000),
                replace(config, scheduler=scheduler),
            )
            assert len(result.jobs) == len(jobs)
            assert 0 < result.metrics["cpu_utilization"] <= 1
            for run in result.jobs:
                assert run.start_time >= run.job.eligible_at
                assert run.finish_time - run.start_time == pytest.approx(run.job.runtime)
            for time in {r.start_time for r in result.jobs} | {r.finish_time for r in result.jobs}:
                active = [r for r in result.jobs if r.start_time <= time < r.finish_time]
                for node in range(config.cluster.nodes):
                    on_node = [r.job for r in active if node in r.allocated_nodes]
                    assert sum(j.cpus_per_node for j in on_node) <= config.cluster.cpus_per_node
                    assert sum(j.memory_mb_per_node for j in on_node) <= 1024
                    assert sum(j.gpus_per_node for j in on_node) <= 1


def test_cli_generates_and_evaluates_auditable_outputs(tmp_path, capsys):
    workload = tmp_path / "workload.csv"
    main(["generate", "--traces", "2", "--jobs", "20", "--output", str(workload)])
    output = tmp_path / "results"
    main(["evaluate", "--workload", str(workload), "--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["config"]["cluster"]["nodes"] == 4
    assert report["aggregate"]["completed_jobs"] == 40
    assert report["aggregate"]["traces"] == 2
    assert len(report["workload_sha256"]) == 64
    assert (output / "schedule.csv").read_text().count("\n") == 41
    assert "Mean response time" in capsys.readouterr().out


def test_cli_rejects_unknown_weight_names_and_preserves_input(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text('{"unknown_factor": 100}')
    with pytest.raises(ValueError, match="Invalid priority"):
        load_weights(path)
    workload = tmp_path / "schedule.csv"
    write_traces(workload, {"trace": [job("a")]})
    original = workload.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--workload", str(workload), "--output", str(tmp_path)])
    assert exc.value.code == 2
    assert workload.read_bytes() == original


def test_examples_can_be_loaded_and_evaluated():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples"
    config = load_config(root / "scheduler.json")
    candidate = load_weights(root / "priority_weights.json")
    result = simulate(read_traces(root / "scheduling_jobs.csv")["example"], candidate, config)
    assert result.metrics["completed_jobs"] == 6
    assert config.cluster.nodes == 4
