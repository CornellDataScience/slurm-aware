import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from slurm_aware.deeprm.evaluate import (  # noqa: E402
    convergence,
    job_footprint,
    main,
    size_analysis,
)
from slurm_aware.deeprm.generator import GeneratorConfig  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec  # noqa: E402
from slurm_aware.deeprm.train import TrainConfig, train  # noqa: E402


def test_footprint_uses_dominant_share():
    assert job_footprint(4, (2, 0, 5), (10, 4, 10)) == pytest.approx(2.0)


def test_convergence_finds_where_curve_settles():
    curve = [10.0] * 10 + [5.0] * 30
    rows = [
        {"iteration": str(i), "iteration_seconds": "2.0", "mean_slowdown": str(v),
         "val_slowdown": "4.5" if i == 30 else ""}
        for i, v in enumerate(curve)
    ]
    out = convergence(rows, window=5, tolerance=0.05)
    assert out["iterations_to_convergence"] == 15  # first full window of 5.0s ends at index 14
    assert out["mean_iteration_seconds"] == 2.0 and out["total_training_seconds"] == 80.0
    assert out["best_val_iteration"] == 30


def test_size_analysis_buckets_and_withheld_ratio():
    jobs = [{"footprint": f, "slowdown": s} for f, s in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                                                         (6, 6)]]
    results = {
        "a": {"_jobs": jobs, "withhold_rate": 0.25, "mean_withheld_footprint": 7.0},
        "b": {"_jobs": jobs, "withhold_rate": 0.0, "mean_withheld_footprint": None},
    }
    out = size_analysis(results)
    a = out["policies"]["a"]
    assert a["slowdown_by_size"] == {"small": 1.5, "medium": 3.5, "large": 5.5}
    assert a["withheld_footprint_ratio"] == pytest.approx(2.0)
    assert out["policies"]["b"]["withheld_footprint_ratio"] is None


def test_end_to_end_report(tmp_path, capsys):
    gen = GeneratorConfig(cluster=ClusterSpec(("cpu", "gpu", "mem"), (5, 2, 5)), horizon=10,
                          max_duration=6)
    cfg = TrainConfig(output=str(tmp_path / "run"), generator=gen.to_dict(), num_jobsets=2,
                      val_jobsets=1, num_slots=3, backlog_size=6, lookahead=6, max_time=60,
                      episodes_per_jobset=2, iterations=2, eval_every=1, workers=0)
    train(cfg, log=lambda *_: None)
    out = tmp_path / "eval"
    args = ["--checkpoint", str(tmp_path / "run" / "best.pt"), "--output", str(out),
            "--loads", "0.5", "1.5", "--jobsets", "2", "--regime", "bursty"]
    assert main(args) == 0
    report = json.loads((out / "evaluation.json").read_text())
    assert report["regime"] == "bursty" and report["analysis_load"] == 0.5
    assert set(report["by_load"]["1.5"]) == {"deeprm", "sjf", "packer", "tetris", "random"}
    assert report["training"]["iterations"] == 2
    assert {p.name for p in out.glob("*.png")} == {
        "slowdown_vs_load.png", "slowdown_by_size.png", "training_curve.png"}
    assert "Withhold rate" in (out / "evaluation.md").read_text()
