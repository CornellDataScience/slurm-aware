import json
import re
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from slurm_aware.deeprm.baselines import SJF  # noqa: E402
from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig  # noqa: E402
from slurm_aware.deeprm.generator import GeneratorConfig  # noqa: E402
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402
from slurm_aware.deeprm.policy import NetworkPolicy, PolicyNetwork  # noqa: E402
from slurm_aware.deeprm.runtime import RuntimePredictor  # noqa: E402
from slurm_aware.deeprm.train import TrainConfig, train  # noqa: E402
from slurm_aware.deeprm.visualize import main, record_episode, render_html  # noqa: E402

SPEC = ClusterSpec(("cpu", "gpu"), (4, 2))
JOBS = [Job(0, 0, 3, (2, 1)), Job(1, 0, 2, (3, 0)), Job(2, 1, 1, (1, 1))]


def make_env():
    return DeepRMEnv(SPEC, EnvConfig(num_slots=2, backlog_size=4, lookahead=5, max_time=40),
                     track_decisions=True)


def test_trace_matches_environment_decisions():
    env = make_env()
    network = PolicyNetwork(env.observation_size, env.n_actions)
    trace = record_episode(env, NetworkPolicy(network), JOBS, network)
    steps = trace["steps"]
    assert [s["action"] for s in steps] == [d["action"] for d in env.decisions]
    assert [s["started"] for s in steps] == [d["scheduled"] for d in env.decisions]
    for s in steps:
        assert sum(s["probs"]) == pytest.approx(1.0, abs=1e-3)
        assert s["probs"][s["action"]] == max(s["probs"])  # greedy policy
        assert s["withheld"] == (s["started"] is None and bool(s["fitting"]))
    started = {s["started"] for s in steps if s["started"] is not None}
    assert started == {j["id"] for j in trace["jobs"] if j["start"] is not None}
    assert trace["summary"]["decisions"] == len(steps)
    assert trace["summary"]["withheld"] == sum(s["withheld"] for s in steps)


def test_cluster_cells_belong_to_running_jobs():
    env = make_env()
    trace = record_episode(env, SJF(), JOBS)
    by_id = {job.job_id: job for job in JOBS}
    record = {j["id"]: j for j in trace["jobs"]}
    for s in trace["steps"]:
        assert s["probs"] is None
        for r, cells in enumerate(s["cluster"]):
            ids, counts = np.unique([c for c in cells if c >= 0], return_counts=True)
            for job_id, n in zip(ids.tolist(), counts.tolist()):
                job = by_id[job_id]
                remaining = record[job_id]["start"] + job.duration - s["time"]
                assert n == job.demand[r] * remaining


def test_html_embeds_trace_safely():
    env = make_env()
    trace = record_episode(env, SJF(), JOBS)
    trace["meta"] = {"policy": "sjf", "checkpoint": "</script><b>x"}
    page = render_html(trace)
    assert "__TRACE_JSON__" not in page
    blob = re.search(r'<script type="application/json" id="trace">(.*?)</script>', page, re.S)
    assert json.loads(blob.group(1))["meta"]["checkpoint"] == "</script><b>x"


def test_cli_writes_page(tmp_path, capsys):
    gen = GeneratorConfig(cluster=ClusterSpec(("cpu", "gpu", "mem"), (5, 2, 5)), horizon=10,
                          max_duration=6)
    cfg = TrainConfig(output=str(tmp_path / "run"), generator=gen.to_dict(), num_jobsets=2,
                      val_jobsets=1, num_slots=3, backlog_size=6, lookahead=6, max_time=60,
                      episodes_per_jobset=2, iterations=2, eval_every=1, workers=0)
    train(cfg, log=lambda *_: None)
    out = tmp_path / "viz" / "page.html"
    args = ["--checkpoint", str(tmp_path / "run" / "best.pt"), "--policy", "tetris",
            "--load", "1.2", "--jobset", "1", "--output", str(out)]
    assert main(args) == 0
    page = out.read_text()
    blob = re.search(r'<script type="application/json" id="trace">(.*?)</script>', page, re.S)
    trace = json.loads(blob.group(1))
    assert trace["meta"]["policy"] == "tetris" and trace["meta"]["load"] == 1.2
    assert trace["steps"] and all(len(s["probs"]) == 4 for s in trace["steps"])
    assert "tetris" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["requested", "predicted"])
def test_estimated_trace_matches_policy_observation_without_future_runtime(mode):
    spec = ClusterSpec(("cpu",), (2,))
    jobs = [Job(0, 0, 4, (2,), requested_duration=2),
            Job(1, 3, 1, (2,), requested_duration=9)]
    traces = []
    for duration in (4, 6):
        predictor = RuntimePredictor({"version": 1, "resources": ["cpu"], "job_types": [],
                                      "model": {"kind": "requested"}})
        env = DeepRMEnv(spec, EnvConfig(2, 2, 6, 30, mode),
                        predictor=predictor if mode == "predicted" else None)
        observations = []

        def policy(env):
            observations.append(env.observation())
            return SJF()(env)

        trace = record_episode(env, policy, [replace(jobs[0], duration=duration), jobs[1]])
        assert trace["runtime_mode"] == mode
        for step, observation in zip(trace["steps"], observations):
            cluster = np.asarray(step["cluster"][0])
            np.testing.assert_array_equal(cluster >= 0, observation[:12] > 0)
            for index, job in enumerate(step["slots"]):
                expected = np.zeros((2, 6), dtype=bool)
                if job:
                    expected[:job["demand"][0], :min(6, int(job["duration"]))] = True
                offset = 12 * (index + 1)
                np.testing.assert_array_equal(expected.ravel(), observation[offset:offset + 12] > 0)
        traces.append(trace)
    # The first completion has not happened in either trace at these decision points.
    for a, b in zip(traces[0]["steps"][:3], traces[1]["steps"][:3]):
        assert a["time"] == b["time"] and a["slots"] == b["slots"] and a["cluster"] == b["cluster"]
