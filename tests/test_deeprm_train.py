import csv

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
