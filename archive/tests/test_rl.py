import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from slurm_aware.rl import evaluate as rl_evaluate  # noqa: E402
from slurm_aware.rl.data import split_trace_ids  # noqa: E402
from slurm_aware.rl.features import (  # noqa: E402
    FEATURE_NAMES,
    FeatureNormalizer,
    _gini,
    extract_features,
)
from slurm_aware.rl.policy import (  # noqa: E402
    WEIGHT_NAMES,
    GaussianPolicy,
    action_to_weights,
    weights_to_action,
)
from slurm_aware.rl.reward import (  # noqa: E402
    EMABaseline,
    SimulatorRunner,
    compute_reward,
    reference_objectives,
)
from slurm_aware.rl.train import TrainConfig, load_checkpoint, reinforce_step, train  # noqa: E402
from slurm_aware.sim_cli import load_config  # noqa: E402
from slurm_aware.simulator import Job, PriorityWeights, simulate  # noqa: E402
from slurm_aware.workloads import generate_traces, write_traces  # noqa: E402

CDS_CONFIG = Path("examples/cds_cluster.json")


@pytest.fixture(scope="module")
def workload(tmp_path_factory):
    path = tmp_path_factory.mktemp("rl") / "traces.csv"
    write_traces(path, generate_traces(load_config(CDS_CONFIG), traces=10, jobs_per_trace=30))
    return path


def feature(jobs, name, config):
    return extract_features(jobs, config)[FEATURE_NAMES.index(name)]


# data ---------------------------------------------------------------------------------


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"trace-{i:03}" for i in range(1000)]
    first = split_trace_ids(ids, seed=0)
    assert first == split_trace_ids(list(reversed(ids)), seed=0)
    assert [len(first[s]) for s in ("train", "val", "test")] == [800, 100, 100]
    assert sorted(sum(first.values(), [])) == ids
    assert first != split_trace_ids(ids, seed=1)


# features -----------------------------------------------------------------------------


def test_features_are_fixed_size_finite_and_hand_checkable():
    config = load_config(CDS_CONFIG)
    jobs = [
        Job("a", 0, 10, 20, cpus_per_node=4, account="alice", qos="urgent", gpus_per_node=2),
        Job("b", 30, 10, 10, nodes=2, cpus_per_node=2, account="alice", architecture="aarch64"),
        Job("c", 60, 10, 30, cpus_per_node=6, account="bob"),
    ]
    values = extract_features(jobs, config)
    assert values.shape == (len(FEATURE_NAMES),) and np.isfinite(values).all()
    assert feature(jobs, "job_count", config) == 3
    assert feature(jobs, "overrequest_mean", config) == pytest.approx(2.0)
    assert feature(jobs, "top_account_job_share", config) == pytest.approx(2 / 3)
    assert feature(jobs, "top_qos_fraction", config) == pytest.approx(1 / 3)
    assert feature(jobs, "aarch64_fraction", config) == pytest.approx(1 / 3)
    assert feature(jobs, "interarrival_cv", config) == pytest.approx(0.0)
    # 140 CPU-seconds offered over 60 s of arrivals on 56 CPUs.
    assert feature(jobs, "offered_load", config) == pytest.approx(140 / (56 * 60))


def test_single_job_and_gini_edge_cases():
    config = load_config(CDS_CONFIG)
    assert np.isfinite(extract_features([Job("a", 5, 1, 1, account="bob")], config)).all()
    assert _gini([1, 1, 1]) == pytest.approx(0)
    assert _gini([0, 0, 1]) == pytest.approx(2 / 3)


def test_normalizer_round_trips_and_zeroes_constant_columns():
    x = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    normalizer = FeatureNormalizer.fit(x)
    assert normalizer(x)[:, 1].tolist() == [0, 0]
    assert normalizer(x)[:, 0].tolist() == pytest.approx([-1, 1])
    restored = FeatureNormalizer.from_state_dict(normalizer.state_dict())
    assert (restored(x) == normalizer(x)).all()


# policy -------------------------------------------------------------------------------


def test_action_mapping_inverts_defaults_and_always_yields_valid_weights():
    default = PriorityWeights()
    assert action_to_weights(weights_to_action(default)) == default
    weights = action_to_weights([-5, 2.5, 20, 3, -2, 0, 4])
    assert weights.age == 0 and weights.fairshare == round(10**2.5)
    assert weights.job_size == 10**9 and weights.qos == 1000
    assert weights.tres_cpu == pytest.approx(-100) and weights.tres_memory == 0
    assert weights.association == weights.partition == 0
    with pytest.raises(ValueError):
        action_to_weights([0] * (len(WEIGHT_NAMES) - 1))


def test_policy_starts_at_initial_mean_with_requested_std():
    policy = GaussianPolicy(len(FEATURE_NAMES), initial_std=0.5)
    mean, log_std = policy(torch.randn(5, len(FEATURE_NAMES)))
    assert mean.shape == log_std.shape == (5, len(WEIGHT_NAMES))
    expected = torch.tensor(weights_to_action(PriorityWeights())).expand(5, -1)
    assert torch.allclose(mean, expected, atol=0.1)
    assert torch.allclose(log_std.exp(), torch.full_like(log_std, 0.5))


# reward -------------------------------------------------------------------------------


def test_reward_and_baseline():
    assert compute_reward(90, 100) == pytest.approx(-0.9)
    assert compute_reward(0, 0) == -1.0
    baseline = EMABaseline(decay=0.5)
    assert baseline.advantages([-1.0, -0.5]) == pytest.approx([-0.25, 0.25])
    assert baseline.value == pytest.approx(-0.75)
    assert baseline.advantages([-0.25]) == pytest.approx([0.5])
    assert baseline.value == pytest.approx(-0.5)


def test_runner_matches_direct_simulation_in_and_out_of_process(workload):
    from slurm_aware.workloads import read_traces

    jobs = read_traces(workload)["trace-000"]
    expected = simulate(jobs, PriorityWeights(), load_config(CDS_CONFIG)).metrics
    for workers in (0, 2):
        with SimulatorRunner(workload, CDS_CONFIG, workers=workers) as runner:
            assert runner.run([("trace-000", PriorityWeights())]) == [expected]
            refs = reference_objectives(runner, ["trace-000"], PriorityWeights())
            assert refs == {"trace-000": expected["mean_wait_time_seconds"]}


# training -----------------------------------------------------------------------------


class FakeRunner:
    """Reward depends only on the age action: higher age weight means less waiting."""

    def run(self, requests):
        return [
            {"mean_wait_time_seconds": 100 - 10 * math.log10(max(w.age, 1))} for _, w in requests
        ]


def test_reinforce_moves_the_mean_toward_higher_reward():
    torch.manual_seed(0)
    policy = GaussianPolicy(3, hidden_sizes=(8,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    baseline, ids = EMABaseline(0.9), [f"t{i}" for i in range(64)]
    refs = dict.fromkeys(ids, 100.0)
    x = torch.zeros(64, 3)
    with torch.no_grad():
        before = float(policy(x)[0][0, 0])
    for _ in range(40):
        stats = reinforce_step(policy, optimizer, x, ids, FakeRunner(), refs, baseline)
        assert math.isfinite(stats.loss)
    with torch.no_grad():
        assert float(policy(x)[0][0, 0]) > before + 0.5


def test_train_checkpoint_resume_and_evaluate(workload, tmp_path, capsys):
    output = tmp_path / "run"
    cfg = TrainConfig(
        workload=str(workload),
        config=str(CDS_CONFIG),
        output=str(output),
        epochs=1,
        batch_size=4,
        workers=0,
        split_fractions=(0.6, 0.2, 0.2),
    )
    train(cfg, log=lambda *_: None)
    first = load_checkpoint(output / "latest.pt")
    assert first["epoch"] == 0 and first["step"] == 2
    assert (output / "best.pt").exists()

    cfg.epochs = 2
    result = train(cfg, resume=output / "latest.pt", log=lambda *_: None)
    resumed = load_checkpoint(output / "latest.pt")
    assert resumed["epoch"] == 1 and resumed["step"] == result["steps"] == 4
    assert resumed["splits"] == first["splits"]
    assert len((output / "train_log.csv").read_text().splitlines()) == 5
    assert len((output / "val_log.jsonl").read_text().splitlines()) == 2

    code = rl_evaluate.main(
        [
            "--checkpoint",
            str(output / "best.pt"),
            "--output",
            str(tmp_path / "eval"),
            "--workers",
            "0",
        ]
    )
    assert code == 0
    report = json.loads((tmp_path / "eval" / "evaluation.json").read_text())
    assert report["traces"] == 2 and set(report["summary"]) == {
        "policy",
        "slurm_default_zero",
        "simulator_default",
    }
    assert "Policy vs" in capsys.readouterr().out
