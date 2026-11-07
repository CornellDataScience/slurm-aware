import csv
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig  # noqa: E402
from slurm_aware.deeprm.generator import GeneratorConfig  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec  # noqa: E402
from slurm_aware.deeprm.policy import NetworkPolicy, PolicyNetwork, sample_actions  # noqa: E402
from slurm_aware.deeprm.train import (  # noqa: E402
    RolloutRunner,
    TrainConfig,
    accumulate_gradient,
    build_jobsets,
    discounted_returns,
    fold_jobset_indices,
    load_checkpoint,
    main,
    network_from_checkpoint,
    time_baseline,
    train,
)

SMALL_GEN = GeneratorConfig(cluster=ClusterSpec(("cpu", "gpu", "mem"), (5, 2, 5)),
                            horizon=10, max_duration=6)


def small_cfg(tmp_path, **overrides):
    values = dict(
        output=str(tmp_path / "run"),
        generator=SMALL_GEN.to_dict(),
        num_jobsets=4,
        val_jobsets=2,
        num_slots=3,
        backlog_size=6,
        lookahead=6,
        max_time=80,
        episodes_per_jobset=4,
        iterations=3,
        eval_every=2,
        checkpoint_every=2,
        workers=0,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_returns_and_padded_baseline():
    assert discounted_returns([-1, 0, -2], 1.0).tolist() == [-3, -2, -2]
    assert discounted_returns([-1, -1], 0.5).tolist() == [-1.5, -1]
    baseline = time_baseline([np.array([-4.0, -2.0, -1.0]), np.array([-2.0])])
    assert baseline.tolist() == [-3.0, -1.0, -0.5]  # ended episode counts as 0


def test_policy_shapes_and_sampling():
    net = PolicyNetwork(12, 4)
    assert [m.out_features for m in net.net if hasattr(m, "out_features")] == [20, 4]
    probs = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
    draws = sample_actions(np.repeat(probs, 500, axis=0), np.random.default_rng(0))
    assert set(draws[:500]) == {1} and set(draws[500:]) == {0, 2}
    with pytest.raises(ValueError):
        PolicyNetwork(4, 2, activation="gelu")


def test_gradient_step_favours_above_baseline_action():
    torch.manual_seed(0)
    net = PolicyNetwork(3, 2, hidden_sizes=())
    obs = np.ones(3, dtype=np.float32)
    extra = {"mean_slowdown": 1.0, "truncated": False}
    episodes = [
        {"obs": [obs], "actions": [1], "rewards": [-1.0], **extra},  # better than average
        {"obs": [obs], "actions": [0], "rewards": [-5.0], **extra},
    ]
    before = net.probabilities(obs[None])[0, 1]
    n, stats = accumulate_gradient(net, episodes, 1.0)
    assert n == 2 and stats["mean_return"] == -3.0
    # Analytic gradient of -sum (v - b) log pi for a linear softmax.
    p = torch.softmax(net(torch.from_numpy(obs[None])), -1)[0].detach().numpy()
    expected_bias_grad = -(2 * (np.eye(2)[1] - p) + -2 * (np.eye(2)[0] - p))
    assert net.net[0].bias.grad.numpy() == pytest.approx(expected_bias_grad, abs=1e-5)
    torch.optim.SGD(net.parameters(), lr=0.1).step()
    assert net.probabilities(obs[None])[0, 1] > before


def test_parallel_and_serial_gradients_match(tmp_path):
    cfg = small_cfg(tmp_path)
    jobsets = build_jobsets(cfg, "train", cfg.num_jobsets)
    env = DeepRMEnv(SMALL_GEN.cluster, EnvConfig(3, 6, 6, 80))
    state = PolicyNetwork(env.observation_size, env.n_actions).state_dict()
    with RolloutRunner(cfg, jobsets) as serial:
        g0, n0, s0 = serial.gradient(state, iteration=5)
    with RolloutRunner(TrainConfig(**{**cfg.__dict__, "workers": 2}), jobsets) as parallel:
        g1, n1, s1 = parallel.gradient(state, iteration=5)
    assert n0 == n1 and s0 == s1
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(g0, g1))


def test_train_logs_checkpoints_and_resumes(tmp_path):
    cfg = small_cfg(tmp_path)
    result = train(cfg, log=lambda *_: None)
    assert result["iterations"] == 3
    out = tmp_path / "run"
    state = load_checkpoint(out / "latest.pt")
    assert state["iteration"] == 2 and (out / "best.pt").exists()
    train(small_cfg(tmp_path, iterations=5), resume=out / "latest.pt", log=lambda *_: None)
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2, 3, 4]
    assert all(float(r["mean_slowdown"]) >= 1 for r in rows)
    network, restored = network_from_checkpoint(load_checkpoint(out / "latest.pt"))
    assert restored.num_slots == 3 and restored.hidden_sizes == (20,)
    env = DeepRMEnv(SMALL_GEN.cluster, EnvConfig(3, 6, 6, 80))
    env.reset(build_jobsets(restored, "val", 1)[0])
    assert 0 <= NetworkPolicy(network)(env) < env.n_actions


def test_training_improves_on_small_problem(tmp_path):
    """REINFORCE should beat its random initialization on a small fixed problem."""
    cfg = small_cfg(tmp_path, iterations=40, lr=0.01, episodes_per_jobset=8, num_jobsets=6,
                    eval_every=100, train_loads=(1.0,))
    train(cfg, log=lambda *_: None)
    rows = list(csv.DictReader((tmp_path / "run" / "train_log.csv").open()))
    first = np.mean([float(r["mean_slowdown"]) for r in rows[:5]])
    last = np.mean([float(r["mean_slowdown"]) for r in rows[-5:]])
    assert last < first


def test_cli_rejects_generator_longer_than_lookahead(tmp_path, capsys):
    assert main(["--output", str(tmp_path / "x"), "--lookahead", "10", "--iterations", "1",
                 "--workers", "0"]) == 2
    assert "max_duration" in capsys.readouterr().err


def test_early_stopping_counts_validation_checks_and_preserves_best(tmp_path, monkeypatch):
    scores = iter([10.0, 10.0, 8.0, 8.0, 8.0])
    monkeypatch.setattr("slurm_aware.deeprm.train.greedy_slowdown", lambda *_: next(scores))
    cfg = small_cfg(tmp_path, iterations=30, early_stopping_patience=2, checkpoint_every=20)
    result = train(cfg, log=lambda *_: None)
    assert result["iterations"] == 10 and result["stopped_early"]
    out = tmp_path / "run"
    best = load_checkpoint(out / "best.pt")
    latest = load_checkpoint(out / "latest.pt")
    assert best["iteration"] == 5 and best["best_val_slowdown"] == 8.0
    assert latest["iteration"] == 9 and latest["early_stopping_stale_evals"] == 2
    # Resuming a stopped run must not reset patience or perform another update.
    resumed = train(replace(cfg, iterations=40), resume=out / "latest.pt", log=lambda *_: None)
    assert resumed == result
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    assert len(rows) == 10
    assert [int(r["iteration"]) for r in rows if r["val_slowdown"]] == [1, 3, 5, 7, 9]


def test_early_stopping_min_delta_accumulates_and_saves_absolute_best(tmp_path, monkeypatch):
    scores = iter([10.0, 9.8, 9.4, 9.2, 9.0])
    monkeypatch.setattr("slurm_aware.deeprm.train.greedy_slowdown", lambda *_: next(scores))
    cfg = small_cfg(tmp_path, iterations=20, eval_every=1, early_stopping_patience=2,
                    early_stopping_min_delta=0.5)
    result = train(cfg, log=lambda *_: None)
    assert result["iterations"] == 5 and result["stopped_early"]
    latest = load_checkpoint(tmp_path / "run" / "latest.pt")
    best = load_checkpoint(tmp_path / "run" / "best.pt")
    assert latest["early_stopping_best"] == 9.4
    assert best["best_val_slowdown"] == 9.0 and best["iteration"] == 4


def test_early_stopping_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr("slurm_aware.deeprm.train.greedy_slowdown", lambda *_: 10.0)
    cfg = small_cfg(tmp_path, iterations=4, early_stopping_patience=2)
    train(cfg, log=lambda *_: None)
    latest = tmp_path / "run" / "latest.pt"
    assert load_checkpoint(latest)["early_stopping_stale_evals"] == 1
    resumed = train(replace(cfg, iterations=20), resume=latest, log=lambda *_: None)
    full = train(replace(cfg, iterations=20, output=str(tmp_path / "full")), log=lambda *_: None)
    assert resumed["iterations"] == full["iterations"] == 6
    assert resumed["stopped_early"] and full["stopped_early"]
    resumed_state = load_checkpoint(latest)
    full_state = load_checkpoint(tmp_path / "full" / "latest.pt")
    assert all(torch.equal(value, full_state["policy"][key])
               for key, value in resumed_state["policy"].items())


def test_early_stopping_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("slurm_aware.deeprm.train.greedy_slowdown", lambda *_: 10.0)
    cfg = small_cfg(tmp_path, iterations=4, eval_every=1, early_stopping_patience=0)
    result = train(cfg, log=lambda *_: None)
    assert result["iterations"] == 4 and not result["stopped_early"]


def test_fold_splits_are_reproducible_disjoint_and_cover_all_jobsets(tmp_path):
    cfg = small_cfg(tmp_path, folds=5, num_jobsets=11)
    splits = fold_jobset_indices(cfg)
    assert splits == fold_jobset_indices(cfg)
    assert splits != fold_jobset_indices(replace(cfg, seed=1))
    assert sorted(i for _, val in splits for i in val) == list(range(cfg.num_jobsets))
    assert [len(val) for _, val in splits] == [3, 2, 2, 2, 2]
    for train_indices, val_indices in splits:
        assert not set(train_indices) & set(val_indices)
        assert sorted(train_indices + val_indices) == list(range(cfg.num_jobsets))


def test_five_fold_cli_trains_fresh_models_on_correct_jobsets(tmp_path, monkeypatch):
    import slurm_aware.deeprm.train as training

    initial_weights, train_sets, val_sets = [], [], []
    original_runner_init = RolloutRunner.__init__
    original_build_network = training.build_network

    def record_runner(self, cfg, jobsets):
        train_sets.append(jobsets)
        original_runner_init(self, cfg, jobsets)

    def record_network(cfg, env):
        network = original_build_network(cfg, env)
        initial_weights.append({key: value.clone() for key, value in network.state_dict().items()})
        return network

    def record_validation(_network, _env, jobsets):
        val_sets.append(jobsets)
        return float(len(val_sets))

    monkeypatch.setattr(RolloutRunner, "__init__", record_runner)
    monkeypatch.setattr(training, "build_network", record_network)
    monkeypatch.setattr(training, "greedy_slowdown", record_validation)
    gen_path = tmp_path / "generator.json"
    gen_path.write_text(json.dumps(SMALL_GEN.to_dict()))
    out = tmp_path / "cv"
    assert main(["--output", str(out), "--generator-config", str(gen_path), "--folds", "5",
                 "--num-jobsets", "6", "--val-jobsets", "0", "--num-slots", "3",
                 "--backlog-size", "6", "--lookahead", "6", "--max-time", "80",
                 "--iterations", "1", "--episodes-per-jobset", "2", "--workers", "0"]) == 0
    summary = json.loads((out / "cross_validation.json").read_text())
    assert len(summary["folds"]) == 5
    assert summary["mean_best_val_slowdown"] == 3.0
    assert summary["std_best_val_slowdown"] == pytest.approx(np.std([1, 2, 3, 4, 5]))
    pool = build_jobsets(small_cfg(tmp_path, num_jobsets=6), "train", 6)
    for i, fold in enumerate(summary["folds"]):
        assert train_sets[i] == [pool[j] for j in fold["train_indices"]]
        assert val_sets[i] == [pool[j] for j in fold["val_indices"]]
        checkpoint = load_checkpoint(out / f"fold_{i + 1}" / "latest.pt")
        assert checkpoint["train_config"]["folds"] == 5
        assert checkpoint["train_config"]["fold_index"] == i
        assert checkpoint["iteration"] == 0
        assert (out / f"fold_{i + 1}" / "best.pt").exists()
        assert all(torch.equal(value, initial_weights[0][key])
                   for key, value in initial_weights[2 * i].items())
    # A fold checkpoint alone reconstructs the same held-out data on resume.
    resumed = train(small_cfg(tmp_path, iterations=2, output=str(out / "fold_1")),
                    resume=out / "fold_1" / "latest.pt", log=lambda *_: None)
    assert resumed["iterations"] == 2
    assert train_sets[-1] == train_sets[0] and val_sets[-1] == val_sets[0]


@pytest.mark.parametrize("args, message", [
    (["--folds", "5", "--num-jobsets", "4"], "folds"),
    (["--folds", "0"], "folds"),
    (["--early-stopping-patience", "-1"], "early_stopping_patience"),
    (["--early-stopping-min-delta", "-0.1"], "early_stopping_min_delta"),
    (["--early-stopping-min-delta", "nan"], "early_stopping_min_delta"),
    (["--eval-every", "0"], "eval_every"),
])
def test_cli_rejects_invalid_stopping_and_fold_options(tmp_path, capsys, args, message):
    out = tmp_path / "invalid"
    assert main(["--output", str(out), *args]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()
