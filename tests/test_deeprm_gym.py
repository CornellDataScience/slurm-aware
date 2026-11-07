import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from slurm_aware.deeprm.env import VOID, DeepRMEnv, EnvConfig  # noqa: E402
from slurm_aware.deeprm.generator import GeneratorConfig, JobTraceGenerator  # noqa: E402
from slurm_aware.deeprm.gym_env import DeepRMGymEnv  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402

SPEC = ClusterSpec(("cpu",), (2,))
CONFIG = EnvConfig(num_slots=2, backlog_size=2, lookahead=5, max_time=20)
JOBS = [Job(0, 0, 3, (2,)), Job(1, 1, 2, (1,)), Job(2, 3, 1, (1,))]


def test_registered_environment_passes_gymnasium_checker():
    env = gym.make("slurm_aware.deeprm.gym_env:SlurmAware-DeepRM-v0")
    check_env(env.unwrapped)
    obs, _ = env.reset(seed=42)
    assert env.action_space == gym.spaces.Discrete(11)
    assert env.observation_space.contains(obs) and obs.dtype == np.float32
    assert env.unwrapped.spec.id == "SlurmAware-DeepRM-v0"
    env.close()


def test_adapter_matches_event_simulator_through_complete_episode():
    env = DeepRMGymEnv(SPEC, CONFIG, jobs=JOBS)
    core = DeepRMEnv(SPEC, CONFIG)
    obs, info = env.reset(seed=0)
    expected_obs, expected_info = core.reset(JOBS)
    np.testing.assert_array_equal(obs, expected_obs)
    assert info == {key: value for key, value in expected_info.items() if value is not None}
    total = 0.0
    for action in [1, VOID, 1, 1, 2, VOID, VOID]:
        obs, reward, terminated, truncated, info = env.step(np.int64(action))
        expected = core.step(action)
        np.testing.assert_array_equal(obs, expected[0])
        assert (reward, terminated, truncated) == expected[1:4]
        assert info == {key: value for key, value in expected[4].items() if value is not None}
        assert env.observation_space.contains(obs)
        total += reward
    assert terminated and not truncated and info["time"] == 5
    assert total == pytest.approx(-4)
    assert env.simulator.slowdowns() == [1, 2, 1]
    with pytest.raises(gym.error.ResetNeeded):
        env.step(VOID)


def test_seed_reproduces_workloads_and_unseeded_reset_continues_stream():
    env = DeepRMGymEnv()
    first_obs, first_info = env.reset(seed=123)
    first_jobs = list(env.simulator.jobs)
    env.step(VOID)
    env.reset()
    second_jobs = list(env.simulator.jobs)
    assert first_jobs != second_jobs
    obs, info = env.reset(seed=123)
    np.testing.assert_array_equal(obs, first_obs)
    assert info == first_info and env.simulator.jobs == first_jobs
    env.reset()
    assert env.simulator.jobs == second_jobs


def test_fixed_jobs_and_one_episode_options_override():
    env = DeepRMGymEnv(SPEC, CONFIG, jobs=JOBS)
    override = [Job(10, 2, 1, (1,))]
    env.reset(seed=7, options={"jobs": override})
    assert env.simulator.jobs == override
    env.reset()
    assert env.simulator.jobs == JOBS
    env.step(1)
    env.reset(seed=99)
    assert env.simulator.slots == [JOBS[0], None]
    assert env.simulator.backend.running() == []


def test_explicit_jobs_do_not_advance_generator_random_stream():
    env = DeepRMGymEnv()
    env.reset(seed=42)
    expected = list(env.simulator.jobs)
    env.reset(seed=42, options={"jobs": []})
    assert env.simulator.jobs == []
    env.reset()
    assert env.simulator.jobs == expected


@pytest.mark.parametrize("duration,terminated", [(3, True), (5, False)])
def test_completion_and_simulated_time_limit(duration, terminated):
    env = DeepRMGymEnv(
        SPEC, EnvConfig(2, 2, 5, 3), jobs=[Job(0, 0, duration, (1,))]
    )
    env.reset()
    env.step(1)
    _, reward, ended, truncated, info = env.step(VOID)
    assert ended == terminated and truncated == (not terminated)
    assert reward == pytest.approx(-3 / duration) and info["time"] == 3
    with pytest.raises(gym.error.ResetNeeded):
        env.step(VOID)


def test_empty_workload_has_one_terminal_transition():
    env = DeepRMGymEnv(SPEC, CONFIG, jobs=[])
    with pytest.raises(gym.error.ResetNeeded):
        env.step(VOID)
    obs, info = env.reset()
    assert env.observation_space.contains(obs) and not obs.any()
    _, reward, terminated, truncated, step_info = env.step(VOID)
    assert reward == 0 and terminated and not truncated and step_info == info
    with pytest.raises(gym.error.ResetNeeded):
        env.step(VOID)
    env.reset()
    assert env.step(VOID)[2]


@pytest.mark.parametrize("action", [-1, 3, 1.5, np.array([1]), "1"])
def test_out_of_space_actions_are_rejected_without_changing_state(action):
    env = DeepRMGymEnv(SPEC, CONFIG, jobs=JOBS)
    obs, _ = env.reset()
    with pytest.raises(ValueError, match="Action"):
        env.step(action)
    np.testing.assert_array_equal(env.simulator.observation(), obs)
    assert env.simulator.time == 0


def test_generator_configuration_is_validated():
    generator = JobTraceGenerator()
    with pytest.raises(ValueError, match="not both"):
        DeepRMGymEnv(jobs=[], generator=generator)
    with pytest.raises(ValueError, match="cluster"):
        DeepRMGymEnv(SPEC, generator=generator)
    with pytest.raises(ValueError, match="max_duration"):
        DeepRMGymEnv(config=CONFIG, generator=generator)
    generator = JobTraceGenerator(GeneratorConfig(horizon=3, max_duration=5))
    env = DeepRMGymEnv(config=CONFIG, generator=generator)
    env.reset(seed=0)
    assert all(job.arrival < 3 and job.duration <= 5 for job in env.simulator.jobs)


def test_vector_environment_autoresets_completed_and_empty_workloads():
    env = gym.vector.SyncVectorEnv([
        lambda: DeepRMGymEnv(SPEC, CONFIG, jobs=[Job(0, 0, 1, (1,))]),
        lambda: DeepRMGymEnv(SPEC, CONFIG, jobs=[]),
    ])
    obs, _ = env.reset(seed=5)
    assert env.observation_space.contains(obs)
    _, _, terminated, _, _ = env.step(np.array([1, VOID]))
    assert terminated.tolist() == [False, True]
    _, _, terminated, _, _ = env.step(np.array([VOID, VOID]))
    assert terminated.tolist() == [True, False]
    obs, _, terminated, _, _ = env.step(np.array([VOID, VOID]))
    assert terminated.tolist() == [False, True]
    assert env.observation_space.contains(obs)
    env.close()
