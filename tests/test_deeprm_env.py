import pytest

np = pytest.importorskip("numpy")

from slurm_aware.deeprm.backend import ClusterBackend, InProcessBackend  # noqa: E402
from slurm_aware.deeprm.env import VOID, DeepRMEnv, EnvConfig  # noqa: E402
from slurm_aware.deeprm.generator import JobTraceGenerator  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402

SPEC = ClusterSpec(("cpu", "gpu", "mem"), (4, 2, 4))
SMALL = EnvConfig(num_slots=2, backlog_size=2, lookahead=5, max_time=100)


def test_backend_fixed_allocation_and_completion():
    backend = InProcessBackend(SPEC, horizon=5)
    job = Job(0, 0, 3, (3, 1, 0))
    assert backend.allocate(job, 0)
    assert backend.available()[:3].tolist() == [[1, 1, 4]] * 3
    assert not backend.fits(Job(1, 0, 1, (2, 0, 0)))
    assert backend.fits(Job(1, 0, 2, (1, 1, 4)))
    assert backend.advance() == [] and backend.advance() == []
    assert backend.advance() == [(job, 0, 3)] and backend.running() == []
    assert backend.available().tolist() == [[4, 2, 4]] * 5


def test_canvas_marks_each_jobs_units():
    backend = InProcessBackend(SPEC, horizon=5)
    backend.allocate(Job(0, 0, 2, (1, 0, 2)), 0)
    backend.allocate(Job(1, 0, 4, (2, 0, 1)), 0)
    cpu, gpu, mem = backend.canvas()
    assert cpu.shape == (4, 5) and not gpu.any()
    assert (cpu[:, :2] > 0).sum(axis=0).tolist() == [3, 3]
    assert (cpu[:, 2:4] > 0).sum(axis=0).tolist() == [2, 2] and not cpu[:, 4].any()
    assert len(set(cpu[cpu > 0].tolist())) == 2  # two distinct job shades
    # The default (backend-agnostic) canvas agrees on occupancy.
    default = ClusterBackend.canvas(backend)
    assert all(((a > 0) .sum(0) == (b > 0).sum(0)).all() for a, b in zip(default, (cpu, gpu, mem)))


def test_time_freezes_on_valid_actions_and_advances_otherwise():
    jobs = [Job(0, 0, 2, (2, 0, 0)), Job(1, 0, 1, (2, 0, 0)), Job(2, 0, 1, (1, 0, 0))]
    env = DeepRMEnv(SPEC, SMALL)
    obs, info = env.reset(jobs)
    assert obs.shape == (env.observation_size,) and info["backlog"] == 1
    _, r, *_ = env.step(1)
    assert r == 0 and env.time == 0 and env.slots[0] == jobs[2]  # refilled from backlog
    _, r, *_ = env.step(2)
    assert r == 0 and env.time == 0
    _, r, *_ = env.step(1)  # job 2 does not fit (cluster full): invalid -> time advances
    assert env.time == 1 and r == pytest.approx(-(1 / 2 + 1 + 1))
    _, r, *_ = env.step(VOID)
    assert env.time == 2


def test_observation_layout():
    env = DeepRMEnv(SPEC, SMALL)
    env.reset([Job(0, 0, 3, (2, 1, 0)), Job(1, 0, 1, (1, 0, 1)), Job(2, 0, 1, (1, 0, 0))])
    obs = env.observation()
    shapes = env.image_shapes()
    assert sum(r * c for _, r, c in shapes) + SMALL.backlog_size == obs.size
    images, offset = {}, 0
    for name, rows, cols in shapes:
        images[name] = obs[offset : offset + rows * cols].reshape(rows, cols)
        offset += rows * cols
    assert images["slot1/cpu"][:2, :3].all() and images["slot1/cpu"].sum() == 6
    assert images["slot1/gpu"].sum() == 3 and not images["slot1/mem"].any()
    assert not images["cluster/cpu"].any()
    assert obs[offset:].tolist() == [1.0, 0.0]


def test_backlog_overflow_drops_jobs():
    jobs = [Job(i, 0, 1, (1, 0, 0)) for i in range(6)]
    env = DeepRMEnv(SPEC, SMALL)
    _, info = env.reset(jobs)
    assert info["dropped"] == 2 and len(env.records) == 4


@pytest.mark.parametrize("seed", range(5))
def test_return_equals_negative_total_slowdown(seed):
    spec = ClusterSpec()
    jobs = JobTraceGenerator().with_load(1.2).jobset(seed)
    env = DeepRMEnv(spec, EnvConfig(backlog_size=200))
    rng = np.random.default_rng(seed)
    env.reset(jobs)
    total = 0.0
    while not env.done:
        _, r, terminated, truncated, _ = env.step(int(rng.integers(env.n_actions)))
        total += r
    assert terminated and not truncated
    assert len(env.slowdowns()) == len(jobs)
    assert total == pytest.approx(-sum(env.slowdowns()))
    assert min(env.slowdowns()) >= 1


def test_truncation_and_reset_replays_jobset():
    jobs = [Job(0, 0, 5, (4, 0, 0)), Job(1, 0, 5, (4, 0, 0))]
    env = DeepRMEnv(SPEC, EnvConfig(num_slots=2, backlog_size=2, lookahead=5, max_time=3))
    env.reset(jobs)
    env.step(1)
    while not env.done:
        *_, terminated, truncated, _ = env.step(VOID)
    assert truncated and not terminated and env.time == 3
    env.reset()
    assert env.time == 0 and env.slots == jobs


def test_decision_log_records_withheld_jobs():
    env = DeepRMEnv(SPEC, SMALL, track_decisions=True)
    env.reset([Job(0, 0, 1, (1, 0, 0)), Job(1, 1, 1, (1, 0, 0))])
    env.step(VOID)
    assert env.decisions[0]["withheld"] == [0] and env.decisions[0]["scheduled"] is None


def test_rejects_jobs_longer_than_lookahead_and_mismatched_backend():
    with pytest.raises(ValueError):
        DeepRMEnv(SPEC, SMALL).reset([Job(0, 0, 6, (1, 0, 0))])
    with pytest.raises(ValueError):
        DeepRMEnv(SPEC, SMALL, backend=InProcessBackend(SPEC, horizon=7))
