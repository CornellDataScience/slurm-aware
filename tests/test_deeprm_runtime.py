import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")

from slurm_aware.deeprm.baselines import SJF, TetrisStar  # noqa: E402
from slurm_aware.deeprm.env import DeepRMEnv, EnvConfig, run_episode  # noqa: E402
from slurm_aware.deeprm.generator import (  # noqa: E402
    PREDICTOR_SPLITS,
    SPLITS,
    GeneratorConfig,
    JobTraceGenerator,
    jobset_seed,
)
from slurm_aware.deeprm.jobs import ClusterSpec, Job  # noqa: E402
from slurm_aware.deeprm.runtime import (  # noqa: E402
    RuntimePredictor,
    _feature_matrix,
    fit_predictors,
    prediction_metrics,
    predictor_digest,
    submission_features,
    train_predictor,
)

SPEC = ClusterSpec(("cpu",), (2,))


def requested_model(resources=("cpu",)):
    return {"version": 1, "resources": list(resources), "job_types": [],
            "model": {"kind": "requested"}}


def make_env(mode="requested", **kwargs):
    return DeepRMEnv(SPEC, EnvConfig(2, 2, 6, 30, mode),
                     predictor=RuntimePredictor(requested_model()) if mode == "predicted" else None,
                     **kwargs)


def test_job_and_generator_compatibility_and_estimate_noise():
    old_job = {"job_id": 0, "arrival": 0, "duration": 3, "demand": [1]}
    assert Job.from_dict(old_job).requested_duration is None
    job = replace(Job.from_dict(old_job), requested_duration=50)
    assert Job.from_dict(json.loads(json.dumps(job.to_dict()))) == job
    cfg = GeneratorConfig()
    old_config = cfg.to_dict()
    del old_config["estimate_multiplier"], old_config["estimate_sigma"]
    assert GeneratorConfig.from_dict(old_config) == cfg
    normal = JobTraceGenerator(cfg).jobset(7)
    precise = JobTraceGenerator(replace(cfg, estimate_multiplier=1, estimate_sigma=0)).jobset(7)
    assert [replace(j, requested_duration=None) for j in normal] == [
        replace(j, requested_duration=None) for j in precise]
    assert all(j.requested_duration == j.duration for j in precise)
    jobs = [j for js in JobTraceGenerator().split("train", 20, 7) for j in js]
    assert any(j.requested_duration < j.duration for j in jobs)
    assert any(j.requested_duration > cfg.max_duration for j in jobs)
    streams = [tuple(jobset_seed(7, split, 0).generate_state(4))
               for split in SPLITS + PREDICTOR_SPLITS]
    assert len(set(streams)) == 6


@pytest.mark.parametrize("mode", ["requested", "predicted"])
def test_hidden_duration_does_not_leak_through_waiting_or_running_images(mode):
    original, changed = make_env(mode), make_env(mode)
    jobs = [Job(0, 0, 4, (2,), requested_duration=2),
            Job(1, 3, 1, (1,), requested_duration=3)]
    altered = [replace(jobs[0], duration=6), jobs[1]]
    np.testing.assert_array_equal(original.reset(jobs)[0], changed.reset(altered)[0])
    np.testing.assert_array_equal(original.step(1)[0], changed.step(1)[0])
    # The arrival at t=3 is observable in both worlds; the first job has overrun.
    np.testing.assert_array_equal(original.step(0)[0], changed.step(0)[0])
    assert original.time == changed.time == 3
    assert not np.array_equal(original.backend.canvas()[0], changed.backend.canvas()[0])
    for env in (original, changed):
        image = env.observation()[:12].reshape(2, 6)
        assert (image[:, 0] > 0).all() and not image[:, 1:].any()
        assert env.fitting_slots() == []
        assert env.backend.available()[0].tolist() == [0]
    # Real completion releases capacity, even though the estimate already expired.
    original.step(0)
    assert original.time == 4 and original.fitting_slots() == [1]
    changed.step(0)
    assert changed.time == 6 and changed.fitting_slots() == [1]


@pytest.mark.parametrize("mode", ["requested", "predicted"])
def test_overestimates_clip_only_the_image_and_release_on_actual_completion(mode):
    env = make_env(mode)
    jobs = [Job(0, 0, 2, (2,), requested_duration=100),
            Job(1, 0, 1, (2,), requested_duration=1)]
    env.reset(jobs)
    assert env.estimated_duration(jobs[0]) == 100 and 1 in env.fitting_slots()
    env.step(1)
    assert (env.observation()[:12] > 0).all()
    assert env.step(0)[1] == pytest.approx(-3)
    assert env.time == 2 and not env.observation()[:12].any()
    assert env.records[0].finish == 2 and env.fitting_slots() == [2]
    env.step(2)
    assert env.step(0)[2]  # done at the true finish, t=3
    assert env.slowdowns() == [1, 3]


def test_oracle_ignores_submitted_estimates_and_keeps_original_canvas():
    jobs = [Job(0, 0, 3, (1,)), Job(1, 1, 2, (2,))]
    original, changed = make_env("oracle"), make_env("oracle")
    a = run_episode(original, SJF(), jobs)
    b = run_episode(changed, SJF(), [replace(j, requested_duration=100) for j in jobs])
    assert a == b
    original.reset(jobs)
    original.step(1)
    np.testing.assert_array_equal(original.observation()[:12], original.backend.canvas()[0].ravel())


@pytest.mark.parametrize("mode", ["requested", "predicted"])
def test_duration_heuristics_share_the_estimates_and_reward_uses_actual_time(mode):
    env = make_env(mode)
    jobs = [Job(0, 0, 1, (2,), requested_duration=5),
            Job(1, 0, 5, (2,), requested_duration=1)]
    env.reset(jobs)
    assert SJF()(env) == TetrisStar(1.0)(env) == 2
    result = run_episode(env, SJF(), jobs)
    assert result["return"] == pytest.approx(-sum(result["slowdowns"]))
    assert result["slowdowns"] == [6, 1]


def test_explicit_modes_and_missing_estimates_fail_without_oracle_fallback():
    with pytest.raises(ValueError, match="requested_duration"):
        make_env().reset([Job(0, 0, 1, (1,))])
    with pytest.raises(ValueError, match="runtime predictor"):
        DeepRMEnv(SPEC, EnvConfig(runtime_mode="predicted"))
    with pytest.raises(ValueError, match="runtime_mode"):
        EnvConfig(runtime_mode="unknown")
    with pytest.raises(ValueError, match="resources"):
        DeepRMEnv(SPEC, EnvConfig(runtime_mode="predicted"),
                  predictor=RuntimePredictor(requested_model(("gpu",))))
    with pytest.raises(ValueError, match="requested_duration"):
        Job(0, 0, 1, (1,), requested_duration=0)


def test_predictions_are_cached_by_submission_features_only(monkeypatch):
    predictor = RuntimePredictor(requested_model())
    calls = []
    predict_rows = predictor._predict_rows

    def counted(rows):
        calls.append(len(rows))
        return predict_rows(rows)

    monkeypatch.setattr(predictor, "_predict_rows", counted)
    job = Job(0, 0, 1, (1,), requested_duration=3)
    assert predictor.predict([job, job]).tolist() == [3, 3]
    assert predictor.predict([replace(job, duration=6, job_id=100, arrival=5)]).tolist() == [3]
    assert calls == [1]


def test_exported_models_match_sklearn_and_ignore_labels():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    gen = JobTraceGenerator()
    jobs = [j for js in gen.split("runtime_train", 5, 4) for j in js]
    heldout = [j for js in gen.split("runtime_test", 3, 4) for j in js]
    heldout.append(replace(heldout[0], job_type="unseen"))
    artifacts = fit_predictors(jobs, gen.config.cluster, seed=4)
    kinds = artifacts["ridge"]["job_types"]
    x = _feature_matrix([submission_features(j) for j in jobs], kinds)
    y = np.log1p([j.duration for j in jobs])
    test_x = _feature_matrix([submission_features(j) for j in heldout], kinds)
    estimators = {
        "ridge": Ridge(alpha=1.0),
        "forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                        random_state=4, n_jobs=1),
    }
    for name, estimator in estimators.items():
        expected = np.maximum(1, np.rint(np.expm1(estimator.fit(x, y).predict(test_x))))
        predictor = RuntimePredictor(json.loads(json.dumps(artifacts[name])))
        np.testing.assert_array_equal(predictor.predict(heldout), expected)
        # A fresh predictor prevents its cache from hiding accidental label access.
        changed = [replace(j, duration=j.duration * 100, arrival=999, job_id=i + 1000)
                   for i, j in enumerate(heldout)]
        np.testing.assert_array_equal(RuntimePredictor(artifacts[name]).predict(changed), expected)


def test_prediction_metrics_include_underestimate_severity():
    metrics = prediction_metrics([1, 4, 10], [2, 2, 2])
    assert metrics["mae"] == pytest.approx(11 / 3)
    assert metrics["within_2x_fraction"] == pytest.approx(2 / 3)
    assert metrics["underprediction_fraction"] == pytest.approx(2 / 3)
    assert metrics["mean_underprediction_timesteps"] == 5


def test_test_labels_do_not_influence_training_or_model_selection(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    gen = JobTraceGenerator()
    sizes = dict(train_jobsets=5, val_jobsets=3, test_jobsets=3)
    first = train_predictor(gen, tmp_path / "first", seed=7, **sizes)
    original_splits = gen.splits

    def changed_test(sizes, seed):
        splits = original_splits(sizes, seed)
        splits["runtime_test"] = [[replace(j, duration=j.duration * 100) for j in js]
                                   for js in splits["runtime_test"]]
        return splits

    monkeypatch.setattr(gen, "splits", changed_test)
    second = train_predictor(gen, tmp_path / "second", seed=7, **sizes)
    assert first["selected_model"] == second["selected_model"]
    assert first["validation"] == second["validation"] and first["test"] != second["test"]
    a = json.loads((tmp_path / "first/model.json").read_text())
    b = json.loads((tmp_path / "second/model.json").read_text())
    assert a["model"] == b["model"]
    assert (tmp_path / "first/prediction_error.png").stat().st_size > 0
    assert (tmp_path / "first/metrics.csv").stat().st_size > 0


def test_predicted_gym_checker_and_checkpoint_resume_workers_and_evaluation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("sklearn")
    from gymnasium.utils.env_checker import check_env

    from slurm_aware.deeprm.evaluate import main as evaluate
    from slurm_aware.deeprm.gym_env import DeepRMGymEnv
    from slurm_aware.deeprm.train import (
        RolloutRunner,
        TrainConfig,
        build_jobsets,
        build_network,
        load_checkpoint,
        make_env,
        network_from_checkpoint,
        train,
    )
    from slurm_aware.deeprm.train import (
        main as train_cli,
    )

    gen = GeneratorConfig(horizon=8, max_duration=6)
    history = [j for js in JobTraceGenerator(gen).split("runtime_train", 3, 0) for j in js]
    artifact = fit_predictors(history, gen.cluster)["forest"]
    model_path, config_path = tmp_path / "model.json", tmp_path / "generator.json"
    model_path.write_text(json.dumps(artifact))
    config_path.write_text(json.dumps(gen.to_dict()))
    gym_env = DeepRMGymEnv(gen.cluster, EnvConfig(3, 4, 6, 80, "predicted"),
                          generator=JobTraceGenerator(gen), predictor=RuntimePredictor(artifact))
    check_env(gym_env, skip_render_check=True)

    output = tmp_path / "run"
    assert train_cli(["--output", str(output), "--generator-config", str(config_path),
                      "--runtime-mode", "predicted", "--runtime-model", str(model_path),
                      "--lookahead", "6", "--num-slots", "3", "--backlog-size", "4",
                      "--max-time", "80", "--num-jobsets", "2", "--val-jobsets", "2",
                      "--iterations", "1", "--episodes-per-jobset", "2", "--workers", "0"]) == 0
    model_path.unlink()  # Checkpoints must be self-contained.
    state = load_checkpoint(output / "latest.pt")
    _, cfg = network_from_checkpoint(state)
    assert cfg.runtime_mode == "predicted" and cfg.runtime_predictor == artifact
    assert state["runtime_predictor_sha256"] == predictor_digest(artifact)
    env = make_env(cfg)
    jobsets = build_jobsets(cfg, "train", 2)
    env.reset(jobsets[0])
    expected = RuntimePredictor(artifact).predict(env.jobs)
    while not env.done:
        env.step(SJF()(env))
    assert [env.estimated_duration(j) for j in env.jobs] == expected.tolist()
    net_state = build_network(cfg, env).state_dict()
    with RolloutRunner(cfg, jobsets) as serial:
        g0, n0, _ = serial.gradient(net_state, 2)
    with RolloutRunner(replace(cfg, workers=2), jobsets) as parallel:
        g1, n1, _ = parallel.gradient(net_state, 2)
    assert n0 == n1
    for a, b in zip(g0, g1):
        np.testing.assert_allclose(a, b, atol=1e-5)

    train(TrainConfig(output=str(output), workers=0, iterations=2),
          resume=output / "latest.pt", log=lambda *_: None)
    resumed = load_checkpoint(output / "latest.pt")
    assert resumed["train_config"]["runtime_predictor"] == artifact
    assert resumed["iteration"] == 1
    assert evaluate(["--checkpoint", str(output / "best.pt"), "--output", str(tmp_path / "eval"),
                     "--loads", "0.5", "1.5", "--jobsets", "2", "--no-plots"]) == 0
    report = json.loads((tmp_path / "eval/evaluation.json").read_text())
    assert report["runtime_mode"] == "predicted"
    assert report["runtime_predictor_sha256"] == predictor_digest(artifact)
    results = report["by_load"]["0.5"]
    assert results["sjf"]["runtime_prediction"] == results["tetris"]["runtime_prediction"]
    assert "Runtime prediction accuracy" in (tmp_path / "eval/evaluation.md").read_text()

    # Checkpoints written before this extension remain oracle-mode checkpoints.
    legacy = dict(state)
    legacy["train_config"] = dict(state["train_config"])
    del legacy["train_config"]["runtime_mode"], legacy["train_config"]["runtime_predictor"]
    assert network_from_checkpoint(legacy)[1].runtime_mode == "oracle"
