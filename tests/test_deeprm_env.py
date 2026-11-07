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
    backend.advance_to(2)
    assert backend.running() == [job]
    backend.advance_to(3)
    assert backend.running() == []
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


@pytest.mark.parametrize("backlog_size", [0, 2])
def test_backlog_preserves_all_jobs(backlog_size):
    jobs = [Job(i, 0, 1, (1, 0, 0)) for i in range(6)]
    env = DeepRMEnv(SPEC, EnvConfig(2, backlog_size, 5, 100))
    obs, info = env.reset(jobs)
    assert info["dropped"] == 0 and len(env.records) == 6
    assert info["backlog"] == 4
    assert obs[env.observation_size - backlog_size:].tolist() == [1.0] * backlog_size
    while not env.done:
        fitting = env.fitting_slots()
        env.step(fitting[0] if fitting else VOID)
    assert not env.truncated and len(env.slowdowns()) == 6


@pytest.mark.parametrize("seed", range(5))
def test_return_equals_negative_total_slowdown(seed):
    spec = ClusterSpec()
    jobs = JobTraceGenerator().with_load(1.2).jobset(seed)
    env = DeepRMEnv(spec, EnvConfig(backlog_size=200))
    rng = np.random.default_rng(seed)
    env.reset(jobs)
    total = 0.0
    while not env.done:
        fitting = env.fitting_slots()
        _, r, terminated, truncated, _ = env.step(int(rng.choice(fitting)) if fitting else VOID)
        total += r
        free = env.backend.available()
        assert np.all(free >= 0) and np.all(free <= np.array(spec.capacities))
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


def test_hand_calculated_event_schedule_and_rewards():
    env = DeepRMEnv(ClusterSpec(("cpu",), (2,)), SMALL)
    jobs = [Job(0, 0, 3, (2,)), Job(1, 1, 2, (1,)), Job(2, 3, 1, (1,))]
    env.reset(jobs)
    assert env.step(1)[1] == 0 and env.time == 0
    rewards = [env.step(VOID)[1]]  # arrival B at 1
    assert env.time == 1 and not env.fitting_slots()
    rewards.append(env.step(1)[1])  # B cannot fit; completion A and arrival C at 3
    assert env.time == 3 and env.backend.now == 3
    assert env.records[0].finish == 3
    assert env.slots == jobs[1:] and env.fitting_slots() == [1, 2]
    assert env.step(1)[1] == 0 and env.step(2)[1] == 0 and env.time == 3
    rewards.append(env.step(VOID)[1])
    assert env.time == 4 and env.records[2].finish == 4
    _, reward, terminated, truncated, _ = env.step(VOID)
    rewards.append(reward)
    assert env.time == 5 and terminated and not truncated
    assert [(r.start, r.finish) for r in env.records.values()] == [(0, 3), (3, 5), (3, 4)]
    assert rewards == pytest.approx([-1 / 3, -2 * (1 / 3 + 1 / 2), -1.5, -0.5])
    assert sum(rewards) == pytest.approx(-4)
    assert env.slowdowns() == [1, 2, 1]


@pytest.mark.parametrize("action", [VOID, 2])
def test_idle_jump_beyond_lookahead_and_empty_slot_is_nop(action):
    env = DeepRMEnv(SPEC, SMALL)
    job = Job(0, 40, 2, (2, 1, 3))
    env.reset([job])
    _, reward, terminated, truncated, _ = env.step(action)
    assert env.time == env.backend.now == 40 and reward == 0
    assert not terminated and not truncated and env.slots[0] == job
    env.step(1)
    _, reward, terminated, truncated, _ = env.step(VOID)
    assert env.time == 42 and reward == -1 and terminated and not truncated
    assert all(not image.any() for image in env.backend.canvas())
    env.reset()
    assert env.time == env.backend.now == 0 and env.slots == [None, None]
    assert env.records[0].start is None and env.records[0].finish is None
    env.step(VOID)
    assert env.time == 40


def test_all_simultaneous_events_are_processed_before_next_action():
    env = DeepRMEnv(SPEC, SMALL)
    jobs = [Job(i, arrival, 3, (2, 1, 2)) for i, arrival in enumerate([0, 0, 3, 3])]
    env.reset(list(reversed(jobs)))
    env.step(1)
    env.step(2)
    _, reward, *_ = env.step(VOID)
    assert env.time == 3 and reward == -2
    assert [env.records[i].finish for i in (0, 1)] == [3, 3]
    assert env.backend.running() == [] and env.slots == jobs[2:]
    assert env.fitting_slots() == [1, 2]


def test_backend_shifts_remaining_occupancy_by_elapsed_time():
    backend = InProcessBackend(SPEC, horizon=5)
    short, long = Job(0, 0, 2, (2, 1, 1)), Job(1, 0, 5, (1, 1, 3))
    backend.allocate(short, 0)
    backend.allocate(long, 0)
    backend.advance_to(2)
    assert backend.running() == [long]
    assert backend.available().tolist() == [[3, 1, 1]] * 3 + [[4, 2, 4]] * 2
    for image, demand in zip(backend.canvas(), long.demand):
        assert (image > 0).sum(axis=0).tolist() == [demand] * 3 + [0, 0]
    backend.advance_to(20)
    assert backend.running() == []
    assert backend.available().tolist() == [[4, 2, 4]] * 5
    assert all(not image.any() for image in backend.canvas())
    with pytest.raises(ValueError, match="backward"):
        backend.advance_to(19)


@pytest.mark.parametrize("arrival,duration", [(0, 5), (10, 1), (0, 3)])
def test_time_limit_caps_jumps_and_completion_at_limit_terminates(arrival, duration):
    env = DeepRMEnv(SPEC, EnvConfig(2, 2, 5, 3))
    env.reset([Job(0, arrival, duration, (1, 0, 0))])
    if arrival == 0:
        env.step(1)
    _, reward, terminated, truncated, _ = env.step(VOID)
    assert env.time == env.backend.now == 3
    assert reward == pytest.approx(-3 / duration if arrival == 0 else 0)
    assert terminated == (arrival == 0 and duration == 3)
    assert truncated == (not terminated)
    assert env.records[0].finish == (3 if terminated else None)


def test_nop_without_future_events_truncates_waiting_jobs():
    env = DeepRMEnv(SPEC, SMALL)
    env.reset([Job(0, 0, 2, (1, 0, 0))])
    _, reward, terminated, truncated, _ = env.step(VOID)
    assert env.time == SMALL.max_time and reward == -SMALL.max_time / 2
    assert not terminated and truncated and env.records[0].start is None
    with pytest.raises(RuntimeError, match="Episode is over"):
        env.step(1)


def test_empty_workload_is_already_complete():
    env = DeepRMEnv(SPEC, SMALL)
    obs, info = env.reset([])
    assert env.done and not env.truncated and info["time"] == 0 and not obs.any()
    with pytest.raises(RuntimeError, match="Episode is over"):
        env.step(VOID)
