import json

import pytest

np = pytest.importorskip("numpy")

from slurm_aware.deeprm.generator import (  # noqa: E402
    ArrivalConfig,
    GeneratorConfig,
    JobTraceGenerator,
    JobType,
    describe,
    jobset_seed,
    load_jobsets,
    main,
    offered_load,
    save_jobsets,
)
from slurm_aware.deeprm.jobs import ClusterSpec  # noqa: E402


def test_same_seed_same_jobset_and_splits_are_disjoint():
    gen = JobTraceGenerator()
    assert gen.jobset(jobset_seed(7, "train", 3)) == gen.jobset(jobset_seed(7, "train", 3))
    splits = gen.splits({"train": 5, "val": 5, "test": 5}, seed=7)
    keys = {
        name: {tuple((j.arrival, j.duration, j.demand) for j in js) for js in sets}
        for name, sets in splits.items()
    }
    assert not keys["train"] & keys["val"] and not keys["train"] & keys["test"]


def test_durations_clipped_and_types_shift_duration():
    gen = JobTraceGenerator(GeneratorConfig(max_duration=20, load=1.5))
    jobs = [j for js in gen.split("train", 60, 0) for j in js]
    durations = np.array([j.duration for j in jobs])
    assert durations.min() >= 1 and durations.max() <= 20
    # Continuous heavy tail: many distinct values, no gap between short and long.
    assert len(set(durations.tolist())) >= 15
    by_type = {t: np.median([j.duration for j in jobs if j.job_type == t]) for t in
               ("small_interactive", "gpu_train")}
    assert by_type["gpu_train"] > 3 * by_type["small_interactive"]


def test_demands_follow_job_type_profiles():
    gen = JobTraceGenerator()
    jobs = [j for js in gen.split("train", 30, 1) for j in js]
    gpu = gen.config.cluster.resources.index("gpu")
    assert all((j.demand[gpu] > 0) == (j.job_type == "gpu_train") for j in jobs)
    assert all(d <= c for j in jobs for d, c in zip(j.demand, gen.config.cluster.capacities))


@pytest.mark.parametrize("load", [0.1, 0.7, 1.9])
def test_bottleneck_load_is_hit_on_average(load):
    gen = JobTraceGenerator().with_load(load)
    sets = gen.split("train", 300, 2)
    realized = np.mean([offered_load(js, gen.config.cluster, gen.config.horizon) for js in sets],
                       axis=0)
    assert realized.max() == pytest.approx(load, rel=0.08)


def test_bursty_multipliers_average_one_and_concentrate_arrivals():
    arrival = ArrivalConfig("bursty", period=10, duty=0.2, peak_ratio=10)
    m = arrival.multipliers(50)
    assert m.mean() == pytest.approx(1.0)
    assert m[0] / m[5] == pytest.approx(10)
    gen = JobTraceGenerator(GeneratorConfig(arrival=arrival, load=1.0))
    arrivals = np.array([j.arrival for js in gen.split("train", 100, 3) for j in js])
    assert np.mean(arrivals % 10 < 2) > 0.6  # 20% of the time, ~71% of the jobs
    diurnal = ArrivalConfig("diurnal", period=25, amplitude=0.5).multipliers(50)
    assert diurnal.mean() == pytest.approx(1.0) and diurnal.max() == pytest.approx(1.5, abs=0.01)


def test_custom_cluster_and_types_generalize():
    spec = ClusterSpec(("cpu", "mem", "ssd", "gpu"), (32, 64, 8, 8))
    types = (JobType("io", 1.0, {"cpu": (0.1, 0.2), "ssd": (0.5, 1.0)}, 3.0, 0.5),)
    gen = JobTraceGenerator(GeneratorConfig(cluster=spec, job_types=types, load=0.5))
    jobs = gen.jobset(0)
    assert jobs and all(len(j.demand) == 4 and j.demand[2] >= 4 for j in jobs)
    assert describe([jobs], spec, 50)["mean_offered_load"]["ssd"] == pytest.approx(0.5, rel=0.5)


def test_config_and_jobsets_roundtrip(tmp_path):
    cfg = GeneratorConfig(arrival=ArrivalConfig("bursty"), load=1.3)
    assert GeneratorConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    sets = JobTraceGenerator(cfg).splits({"train": 2, "test": 1}, seed=4)
    save_jobsets(tmp_path / "j.json", sets, {"generator": cfg.to_dict()})
    loaded, meta = load_jobsets(tmp_path / "j.json")
    assert loaded == sets and GeneratorConfig.from_dict(meta["generator"]) == cfg


def test_cli_writes_reproducible_file(tmp_path, capsys):
    args = ["--train", "2", "--val", "1", "--test", "1", "--regime", "bursty", "--seed", "5"]
    assert main([*args, "--output", str(tmp_path / "a.json")]) == 0
    assert main([*args, "--output", str(tmp_path / "b.json")]) == 0
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
    assert "arrival_rate" in capsys.readouterr().out


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        ArrivalConfig("poisson")
    with pytest.raises(ValueError):
        GeneratorConfig(job_types=(JobType("x", 1, {"tpu": (0.1, 0.2)}, 2, 0.5),))
    with pytest.raises(ValueError):
        JobType("x", 1, {"cpu": (0.0, 0.2)}, 2, 0.5)
