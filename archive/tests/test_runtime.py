import joblib
import numpy as np
import pandas as pd
import pytest

from slurm_aware.modeling import predict_runtime, split_workloads, train_and_evaluate
from slurm_aware.synthetic import FEATURES, GROUP, TARGET, generate_jobs, validate_jobs


def test_generator_is_reproducible_and_respects_cluster_capacity():
    jobs = generate_jobs(300, 30, seed=7)
    pd.testing.assert_frame_equal(jobs, generate_jobs(300, 30, seed=7))
    assert not jobs.equals(generate_jobs(300, 30, seed=8))
    assert jobs.job_id.is_unique
    assert jobs[GROUP].nunique() == 30
    assert (jobs[TARGET] > 0).all()
    assert jobs.nodes.between(1, 2).all()
    assert (jobs.tasks_per_node * jobs.cpus_per_task <= 2).all()
    assert set(jobs.workload_type) == {"cpu_bound", "io_bound", "mixed"}
    validate_jobs(jobs)


@pytest.mark.parametrize("jobs,workloads", [(10, 20), (100, 10)])
def test_generator_rejects_insufficient_workloads(jobs, workloads):
    with pytest.raises(ValueError, match="at least"):
        generate_jobs(jobs, workloads)


@pytest.mark.parametrize(
    "column,value",
    [(TARGET, 0), ("input_size_mb", np.nan), ("iterations", np.inf), ("nodes", 1.5)],
)
def test_validation_rejects_invalid_measurements(column, value):
    jobs = generate_jobs(100, 20)
    jobs[column] = jobs[column].astype(float)
    jobs.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_jobs(jobs)


def test_split_keeps_related_jobs_together_and_covers_every_row():
    jobs = generate_jobs(300, 30)
    splits = split_workloads(jobs)
    group_sets = [set(part[GROUP]) for part in splits.values()]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert set.union(*group_sets) == set(jobs[GROUP])
    assert sorted(index for part in splits.values() for index in part.index) == list(jobs.index)


def test_test_labels_never_influence_training_or_selection(tmp_path):
    jobs = generate_jobs(300, 30)
    data_path = tmp_path / "jobs.csv"
    jobs.to_csv(data_path, index=False)
    first = train_and_evaluate(data_path, tmp_path / "original")

    # Changing only unseen targets must affect scores, never model selection or predictions.
    test_indices = split_workloads(jobs)["test"].index
    jobs.loc[test_indices, TARGET] *= 100
    jobs.to_csv(data_path, index=False)
    second = train_and_evaluate(data_path, tmp_path / "changed")
    assert first["selected_model"] == second["selected_model"]
    original = pd.read_csv(tmp_path / "original/test_predictions.csv")
    changed = pd.read_csv(tmp_path / "changed/test_predictions.csv")
    prediction_columns = [name for name in original if name.endswith("_prediction_seconds")]
    np.testing.assert_allclose(original[prediction_columns], changed[prediction_columns])
    assert first["metrics"] != second["metrics"]

    bundle = joblib.load(tmp_path / "original/model.joblib")
    actual = predict_runtime(bundle["model"], original[FEATURES])
    np.testing.assert_allclose(actual, original[f"{first['selected_model']}_prediction_seconds"])
    # IDs, a supplied target, and arbitrary future information are explicitly ignored.
    original[TARGET] = 1e12
    original[GROUP] = "changed-id"
    original["future_end_time"] = 1e12
    np.testing.assert_allclose(actual, predict_runtime(bundle["model"], original))
    assert np.isfinite(actual).all() and (actual > 0).all()
    assert (tmp_path / "original/evaluation.png").stat().st_size > 0
