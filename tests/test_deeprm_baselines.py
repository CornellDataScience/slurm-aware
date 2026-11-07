import pytest

np = pytest.importorskip("numpy")

from slurm_aware.deeprm.baselines import (  # noqa: E402
    SJF,
    Packer,
    RandomFit,
    TetrisStar,
    heuristics,
    packer_score,
)
from slurm_aware.deeprm.env import VOID, DeepRMEnv, EnvConfig, run_episode  # noqa: E402
from slurm_aware.deeprm.generator import JobTraceGenerator  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402

SPEC = ClusterSpec(("cpu", "gpu", "mem"), (10, 2, 10))
CFG = EnvConfig(num_slots=3, backlog_size=5, lookahead=6, max_time=100)


def env_with(jobs, running=()):
    """Env at time 0 with `running` already started and `jobs` waiting in slots.

    Running jobs need the lowest ids: same-time arrivals fill slots in id order."""
    env = DeepRMEnv(SPEC, CFG)
    env.reset(list(running) + list(jobs))
    for job in running:
        assert env.step(env.slots.index(job) + 1)[1] == 0
    return env


def test_sjf_picks_shortest_fitting_job():
    jobs = [Job(1, 0, 4, (1, 0, 0)), Job(2, 0, 1, (10, 0, 0)), Job(3, 0, 2, (1, 0, 0))]
    assert SJF()(env_with(jobs)) == 2
    blocker = Job(0, 0, 6, (1, 0, 0))
    env = env_with(jobs, running=[blocker])
    assert env.slots.index(jobs[1]) + 1 not in env.fitting_slots()  # needs all 10 CPUs
    assert SJF()(env) == env.slots.index(jobs[2]) + 1


def test_packer_prefers_job_aligned_with_free_resources():
    a, b = Job(1, 0, 3, (4, 0, 1)), Job(2, 0, 3, (1, 0, 3))
    assert Packer()(env_with([a, b])) == 1  # empty: 0.4 + 0.1 > 0.1 + 0.3
    env = env_with([a, b], running=[Job(0, 0, 6, (5, 0, 0))])
    assert Packer()(env) == env.slots.index(b) + 1  # half the CPUs busy: 0.30 < 0.35
    caps = np.array(SPEC.capacities, float)
    assert packer_score(a, caps, caps) == pytest.approx(0.5 / 3)


def test_tetris_interpolates_between_packer_and_sjf():
    long_aligned, short_small = Job(0, 0, 6, (8, 2, 8)), Job(1, 0, 1, (1, 0, 0))
    env = env_with([long_aligned, short_small])
    assert TetrisStar(0.0)(env) == Packer()(env) == 1
    assert TetrisStar(1.0)(env) == SJF()(env) == 2
    with pytest.raises(ValueError):
        TetrisStar(1.5)


def test_heuristics_void_when_nothing_fits():
    env = env_with([Job(1, 0, 2, (6, 0, 0))], running=[Job(0, 0, 6, (5, 0, 0))])
    assert all(policy(env) == VOID for policy in heuristics().values())


def test_heuristics_finish_generated_jobsets_and_beat_random():
    spec = ClusterSpec()
    gen = JobTraceGenerator().with_load(1.0)
    env = DeepRMEnv(spec, EnvConfig())
    results = {name: [] for name in heuristics()}
    for js in gen.split("test", 15, 0):
        for name, policy in heuristics(seed=1).items():
            out = run_episode(env, policy, js)
            assert not out["truncated"] and out["unfinished"] == 0
            results[name] += out["slowdowns"]
    means = {name: np.mean(v) for name, v in results.items()}
    assert means["sjf"] < means["random"] and means["tetris"] < means["random"]
    assert RandomFit(0).name == "random"
