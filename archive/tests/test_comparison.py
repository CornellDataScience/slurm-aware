import csv
import json
from dataclasses import asdict

import pytest

from slurm_aware.sim_cli import main
from slurm_aware.simulator import (
    Cluster,
    Job,
    PriorityWeights,
    SimulationConfig,
    aggregate_metrics,
    simulate,
)
from slurm_aware.workloads import write_traces


def test_slowdown_and_throughput_use_jobs_and_elapsed_span():
    config = SimulationConfig(cluster=Cluster(nodes=1, cpus_per_node=1))
    result = simulate([Job("a", 5, 10, 10), Job("b", 5, 2, 2)], config=config)
    assert result.metrics["mean_slowdown"] == 3.5  # (10/10 + 12/2) / 2
    assert result.metrics["p95_slowdown"] == pytest.approx(5.75)
    assert result.metrics["throughput_jobs_per_second"] == pytest.approx(2 / 12)
    second = simulate([Job("c", 100, 3, 3)], config=config)
    pooled = aggregate_metrics({"first": result, "second": second})
    assert pooled["mean_slowdown"] == pytest.approx(8 / 3)
    assert pooled["throughput_jobs_per_second"] == pytest.approx(3 / 15)


def comparison_inputs(tmp_path):
    workload = tmp_path / "jobs.csv"
    write_traces(
        workload,
        {
            "trace": [
                Job("a", 0, 10, 10),
                Job("b", 0, 1, 1, qos="urgent"),
            ]
        },
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "cluster": {"nodes": 1, "cpus_per_node": 1},
                "qos_priorities": {"normal": 0, "urgent": 1},
            }
        )
    )
    baseline = tmp_path / "baseline.json"
    zero = dict.fromkeys(asdict(PriorityWeights()), 0)
    baseline.write_text(json.dumps(zero))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({**zero, "qos": 1000}))
    return ["compare", "--workload", str(workload), "--config", str(config)], baseline, candidate


def test_comparison_ranking_reports_and_independent_replays(tmp_path, capsys):
    args, baseline, candidate = comparison_inputs(tmp_path)
    output = tmp_path / "results"
    main(
        [
            *args,
            "--weights",
            f"baseline={baseline}",
            "--weights",
            f"qos={candidate}",
            "--output",
            str(output),
        ]
    )
    report = json.loads((output / "comparison.json").read_text())
    assert report["baseline"] == "baseline"
    assert [r["policy"] for r in report["results"]] == ["qos", "baseline"]
    assert report["results"][0]["mean_response_time_seconds"] == 6
    assert report["results"][1]["mean_response_time_seconds"] == 10.5
    assert report["results"][0]["response_improvement_percent"] == pytest.approx(100 * 4.5 / 10.5)
    assert "N/A" in (output / "comparison.md").read_text()
    assert "Jobs/s" in capsys.readouterr().out
    with (output / "comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    for name in ("qos", "baseline"):
        detail = json.loads((output / name / "report.json").read_text())
        assert detail["workload_sha256"] == report["workload_sha256"]
        assert detail["traces"] == report["policies"][name]["traces"]
        with (output / name / "schedule.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        assert all(float(row["slowdown"]) >= 1 for row in rows)
    reverse = tmp_path / "reverse"
    main(
        [
            *args,
            "--weights",
            f"qos={candidate}",
            "--weights",
            f"baseline={baseline}",
            "--output",
            str(reverse),
        ]
    )
    for name in ("qos", "baseline"):
        assert (reverse / name / "report.json").read_bytes() == (
            output / name / "report.json"
        ).read_bytes()


@pytest.mark.parametrize("names", [("same", "same"), ("Same", "same"), ("../escape", "ok")])
def test_comparison_rejects_unsafe_or_duplicate_names(tmp_path, names):
    args, baseline, candidate = comparison_inputs(tmp_path)
    output = tmp_path / "results"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                *args,
                "--weights",
                f"{names[0]}={baseline}",
                "--weights",
                f"{names[1]}={candidate}",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    assert not output.exists()


def test_comparison_requires_two_candidates_and_preserves_inputs(tmp_path):
    args, baseline, candidate = comparison_inputs(tmp_path)
    with pytest.raises(SystemExit):
        main([*args, "--weights", f"only={baseline}"])
    output = tmp_path / "results"
    output.mkdir()
    protected = output / "comparison.json"
    protected.write_bytes(candidate.read_bytes())
    original = protected.read_bytes()
    with pytest.raises(SystemExit):
        main(
            [
                *args,
                "--weights",
                f"baseline={baseline}",
                "--weights",
                f"qos={protected}",
                "--output",
                str(output),
            ]
        )
    assert protected.read_bytes() == original
