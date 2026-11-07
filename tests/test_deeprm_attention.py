import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig, feature_layout  # noqa: E402
from slurm_aware.deeprm.generator import GeneratorConfig  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402
from slurm_aware.deeprm.policy import AttentionPolicy, NetworkPolicy  # noqa: E402
from slurm_aware.deeprm.train import (  # noqa: E402
    TrainConfig,
    accumulate_gradient,
    build_network,
    load_checkpoint,
    make_env,
    network_from_checkpoint,
    observation_mode,
    train,
)

SPEC = ClusterSpec(("cpu", "gpu", "mem"), (4, 2, 8))
FEATURES = EnvConfig(num_slots=3, backlog_size=2, lookahead=5, max_time=100,
                     observation="features")


def split(obs, layout):
    jobs = obs[: layout.jobs_size].reshape(layout.num_slots, layout.job_features)
    cluster = obs[layout.jobs_size : layout.jobs_size + layout.cluster_size]
    return jobs, cluster.reshape(layout.lookahead, layout.n_resources), obs[-1]


def test_feature_observation_layout_and_values():
    env = DeepRMEnv(SPEC, FEATURES)
    jobs = [Job(0, 0, 5, (2, 0, 0)), Job(1, 0, 2, (1, 1, 4)), Job(2, 0, 1, (1, 0, 0)),
            Job(3, 0, 1, (1, 0, 0)), Job(4, 0, 1, (1, 0, 0)), Job(5, 0, 1, (1, 0, 0))]
    obs, _ = env.reset(jobs)
    layout = env.layout
    assert obs.shape == (env.observation_size,) == (layout.size,)
    assert layout.job_features == 5 and layout.size == 3 * 5 + 5 * 3 + 1
    slot_jobs, cluster, backlog = split(obs, layout)
    assert slot_jobs[1].tolist() == pytest.approx([1, 0.25, 0.5, 0.5, 0.4])
    assert cluster.tolist() == [[1, 1, 1]] * 5
    assert backlog == 1.0  # 3 jobs in a backlog of 2 observation cells saturates
    env.step(1)  # start job 0 (2 CPUs for 5 steps)
    slot_jobs, cluster, backlog = split(env.observation(), layout)
    assert cluster[:, 0].tolist() == [0.5] * 5
    assert backlog == 1.0 and slot_jobs[0][0] == 1  # slot refilled from the backlog
    assert obs.min() >= 0 and obs.max() <= 1


def test_feature_observation_marks_empty_slots():
    env = DeepRMEnv(SPEC, FEATURES)
    obs, _ = env.reset([Job(0, 0, 2, (1, 0, 0))])
    slot_jobs, _, backlog = split(obs, env.layout)
    assert slot_jobs[0][0] == 1 and not slot_jobs[1:].any() and backlog == 0


def make_policy(layers=1, seed=0):
    torch.manual_seed(seed)
    return AttentionPolicy(feature_layout(SPEC, FEATURES), embed_dim=16, heads=2, layers=layers)


def random_obs(present):
    layout = feature_layout(SPEC, FEATURES)
    rng = np.random.default_rng(1)
    jobs = rng.random((layout.num_slots, layout.job_features)).astype(np.float32)
    jobs[:, 0] = present
    jobs[np.array(present) == 0] = 0
    rest = rng.random(layout.cluster_size + 1).astype(np.float32)
    return np.concatenate([jobs.ravel(), rest])


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_slot_order_does_not_change_a_jobs_score(layers):
    policy = make_policy(layers)
    layout = policy.layout
    obs = random_obs([1, 1, 1])
    jobs = obs[: layout.jobs_size].reshape(layout.num_slots, -1)
    permuted = np.concatenate([jobs[[2, 0, 1]].ravel(), obs[layout.jobs_size :]])
    with torch.no_grad():
        a = policy(torch.from_numpy(obs[None]))[0]
        b = policy(torch.from_numpy(permuted[None]))[0]
    assert b[0] == pytest.approx(a[0], abs=1e-5)  # void
    assert b[1:].tolist() == pytest.approx(a[1:][[2, 0, 1]].tolist(), abs=1e-5)


def test_empty_slots_share_learned_logit_and_are_ignored_by_attention():
    policy = make_policy()
    obs = random_obs([1, 0, 0])
    with torch.no_grad():
        logits = policy(torch.from_numpy(obs[None]))[0]
        policy.empty_slot_logit.fill_(3.0)
        changed = policy(torch.from_numpy(obs[None]))[0]
    assert logits[2] == logits[3] == 0.0
    assert changed[2] == changed[3] == 3.0
    assert changed[:2].tolist() == pytest.approx(logits[:2].tolist())
    # Garbage in an empty slot must not leak into the visible job's score.
    noisy = obs.copy()
    noisy[policy.layout.job_features + 1 : 2 * policy.layout.job_features] = 0.9
    with torch.no_grad():
        assert policy(torch.from_numpy(noisy[None]))[0][:2].tolist() == pytest.approx(
            changed[:2].tolist(), abs=1e-5)


def test_all_empty_slots_give_finite_logits_and_gradients():
    policy = make_policy()
    obs = torch.from_numpy(random_obs([0, 0, 0])[None])
    logits = policy(obs)
    assert torch.isfinite(logits).all()
    torch.log_softmax(logits, -1)[0, 0].backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_attention_policy_learns_from_reinforce_gradient():
    policy = make_policy()
    obs = random_obs([1, 1, 0])
    extra = {"mean_slowdown": 1.0, "truncated": False}
    episodes = [
        {"obs": [obs], "actions": [2], "rewards": [-1.0], **extra},
        {"obs": [obs], "actions": [1], "rewards": [-5.0], **extra},
    ]
    before = policy.probabilities(obs[None])[0]
    accumulate_gradient(policy, episodes, 1.0)
    torch.optim.SGD(policy.parameters(), lr=1e-3).step()
    after = policy.probabilities(obs[None])[0]
    # The above-baseline action gains on the below-baseline one; mass may also move to
    # the other actions, so compare the ratio rather than one probability.
    assert np.log(after[2] / after[1]) > np.log(before[2] / before[1])


def small_cfg(tmp_path, **overrides):
    gen = GeneratorConfig(cluster=ClusterSpec(("cpu", "gpu", "mem"), (5, 2, 5)), horizon=10,
                          max_duration=6)
    values = dict(output=str(tmp_path / "run"), generator=gen.to_dict(), num_jobsets=2,
                  val_jobsets=1, num_slots=3, backlog_size=6, lookahead=6, max_time=60,
                  episodes_per_jobset=2, iterations=2, eval_every=1, workers=0,
                  policy="attention", embed_dim=16, attention_heads=2)
    values.update(overrides)
    return TrainConfig(**values)


def test_config_selects_observation_and_network(tmp_path):
    assert observation_mode(TrainConfig()) == "image"
    assert observation_mode(TrainConfig(policy="attention")) == "features"
    assert observation_mode(TrainConfig(observation="features")) == "features"
    with pytest.raises(ValueError):
        observation_mode(TrainConfig(policy="attention", observation="image"))
    with pytest.raises(ValueError):
        observation_mode(TrainConfig(policy="transformer"))
    cfg = small_cfg(tmp_path, policy="mlp", observation="features")
    env = make_env(cfg)
    assert env.observation_size == env.layout.size
    assert build_network(cfg, env)(torch.zeros(1, env.observation_size)).shape == (1, 4)


def test_attention_training_checkpoint_roundtrip(tmp_path):
    train(small_cfg(tmp_path), log=lambda *_: None)
    network, cfg = network_from_checkpoint(load_checkpoint(tmp_path / "run" / "latest.pt"))
    assert isinstance(network, AttentionPolicy) and cfg.policy == "attention"
    env = make_env(cfg)
    env.reset([Job(0, 0, 2, (1, 0, 1))])
    assert 0 <= NetworkPolicy(network)(env) < env.n_actions


def test_gym_wrapper_accepts_feature_observations():
    pytest.importorskip("gymnasium")
    from slurm_aware.deeprm.generator import JobTraceGenerator
    from slurm_aware.deeprm.gym_env import DeepRMGymEnv

    spec = ClusterSpec()
    gen = JobTraceGenerator(GeneratorConfig(cluster=spec))
    env = DeepRMGymEnv(spec, EnvConfig(observation="features"), generator=gen)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, *_ = env.step(0)
        assert env.observation_space.contains(obs)
