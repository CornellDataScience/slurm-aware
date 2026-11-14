"""Fit submission-time runtime predictors on independent synthetic history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from slurm_aware.deeprm.generator import (
    PREDICTOR_SPLITS,
    SPLITS,
    GeneratorConfig,
    JobTraceGenerator,
)
from slurm_aware.deeprm.jobs import ClusterSpec, Job

RUNTIME_MODES = ("oracle", "requested", "predicted")
MODEL_VERSION = 1


def predictor_digest(artifact: dict | None) -> str | None:
    if artifact is None:
        return None
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()


def submission_features(job: Job) -> tuple:
    """Only fields known at submission; never inspect duration, ID or arrival."""
    if job.requested_duration is None:
        raise ValueError("Runtime estimation requires requested_duration on every job.")
    return (*job.demand, job.requested_duration, job.job_type)


def _feature_matrix(rows: Sequence[tuple], job_types: Sequence[str]) -> np.ndarray:
    numeric = np.log1p(np.asarray([row[:-1] for row in rows], dtype=float))
    categorical = np.array([[row[-1] == kind for kind in job_types] for row in rows], float)
    return np.concatenate([numeric, categorical], axis=1)


def _rounded(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Runtime predictions must be finite.")
    return np.maximum(1, np.rint(values))


class RuntimePredictor:
    """Frozen JSON model, with a cache keyed only by submission-time features.

    Coefficients and tree arrays keep checkpoints portable and let inference run
    with NumPy alone. scikit-learn is needed only for fitting.
    """

    def __init__(self, artifact: dict) -> None:
        if artifact.get("version") != MODEL_VERSION:
            raise ValueError(f"Expected runtime model version {MODEL_VERSION}.")
        self.artifact = artifact
        self.resources = tuple(artifact["resources"])
        self.job_types = tuple(artifact["job_types"])
        self.model = artifact["model"]
        if self.model["kind"] not in ("requested", "median", "type_median", "ridge", "forest"):
            raise ValueError("Unknown runtime model kind.")
        self._trees = [
            {k: np.asarray(v) for k, v in tree.items()}
            for tree in self.model.get("trees", [])
        ]
        self._cache: dict[tuple, float] = {}

    def predict(self, jobs: Sequence[Job]) -> np.ndarray:
        rows = [submission_features(job) for job in jobs]
        if any(len(row) != len(self.resources) + 2 for row in rows):
            raise ValueError("Runtime model resource dimensions do not match jobs.")
        missing = list(dict.fromkeys(row for row in rows if row not in self._cache))
        if missing:
            values = self._predict_rows(missing)
            self._cache.update(zip(missing, _rounded(values).tolist()))
        return np.asarray([self._cache[row] for row in rows], dtype=float)

    def _predict_rows(self, rows: Sequence[tuple]) -> np.ndarray:
        model, kind = self.model, self.model["kind"]
        if kind == "requested":
            return np.asarray([row[-2] for row in rows], float)
        if kind == "median":
            return np.full(len(rows), model["median"])
        if kind == "type_median":
            return np.asarray([model["medians"].get(row[-1], model["median"]) for row in rows])
        features = _feature_matrix(rows, self.job_types)
        if kind == "ridge":
            log_duration = features @ np.asarray(model["coef"]) + model["intercept"]
        else:
            features = features.astype(np.float32)  # sklearn trees use float32 at inference
            log_duration = np.zeros(len(rows))
            for tree in self._trees:
                nodes = np.zeros(len(rows), dtype=int)
                active = np.flatnonzero(tree["left"][nodes] != -1)
                while len(active):
                    current = nodes[active]
                    left = features[active, tree["feature"][current]] <= tree["threshold"][current]
                    nodes[active] = np.where(left, tree["left"][current], tree["right"][current])
                    active = active[tree["left"][nodes[active]] != -1]
                log_duration += tree["value"][nodes]
            log_duration /= len(self._trees)
        return np.expm1(log_duration)


def prediction_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if not len(actual):
        return {"jobs": 0}
    error = actual - predicted
    under = error[error > 0]
    return {
        "jobs": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "within_2x_fraction": float(np.mean(
            np.maximum(actual / predicted, predicted / actual) <= 2)),
        "underprediction_fraction": float(np.mean(error > 0)),
        "mean_underprediction_timesteps": float(under.mean()) if len(under) else 0.0,
    }


def fit_predictors(jobs: Sequence[Job], spec: ClusterSpec, seed: int = 0) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    if not jobs:
        raise ValueError("Need nonempty predictor training history.")
    kinds = sorted({job.job_type for job in jobs})
    features = _feature_matrix([submission_features(job) for job in jobs], kinds)
    target = np.asarray([job.duration for job in jobs], float)
    median = float(np.median(target))
    models = {
        "requested": {"kind": "requested"},
        "median": {"kind": "median", "median": median},
        "type_median": {
            "kind": "type_median", "median": median,
            "medians": {kind: float(np.median([j.duration for j in jobs if j.job_type == kind]))
                        for kind in kinds},
        },
    }
    ridge = Ridge(alpha=1.0).fit(features, np.log1p(target))
    models["ridge"] = {"kind": "ridge", "coef": ridge.coef_.tolist(),
                       "intercept": float(ridge.intercept_)}
    forest = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1,
    ).fit(features, np.log1p(target))
    models["forest"] = {
        "kind": "forest",
        "trees": [
            {"left": t.children_left.tolist(), "right": t.children_right.tolist(),
             "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
             "value": t.value[:, 0, 0].tolist()}
            for t in (estimator.tree_ for estimator in forest.estimators_)
        ],
    }
    return {
        name: {"version": MODEL_VERSION, "resources": list(spec.resources),
               "job_types": kinds, "model": model}
        for name, model in models.items()
    }


def _plot_predictions(actual, predictions: dict, selected: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    predicted = predictions[selected]
    upper = max(max(actual), max(predicted))
    axes[0].scatter(actual, predicted, s=10, alpha=0.3)
    axes[0].plot([1, upper], [1, upper], "--", color="gray")
    axes[0].set(xlabel="Actual runtime (timesteps)", ylabel="Predicted runtime (timesteps)",
                title=f"Held-out synthetic jobs: {selected}")
    names = list(predictions)
    axes[1].barh(names, [prediction_metrics(actual, predictions[n])["mae"] for n in names])
    axes[1].set(xlabel="Test MAE (timesteps)", title="Prediction baselines")
    fig.savefig(output / "prediction_error.png", dpi=150)
    plt.close(fig)


def train_predictor(
    generator: JobTraceGenerator, output: Path, *, seed: int = 0,
    train_jobsets: int = 100, val_jobsets: int = 20, test_jobsets: int = 20,
) -> dict:
    import sklearn

    sizes = dict(zip(PREDICTOR_SPLITS, (train_jobsets, val_jobsets, test_jobsets)))
    if min(sizes.values()) < 1:
        raise ValueError("Need at least one jobset in every predictor split.")
    splits = generator.splits(sizes, seed)
    jobs = {name: [job for js in sets for job in js] for name, sets in splits.items()}
    if not all(jobs.values()):
        raise ValueError("Predictor splits must contain jobs; increase the jobset counts or load.")
    artifacts = fit_predictors(jobs["runtime_train"], generator.config.cluster, seed)
    models = {name: RuntimePredictor(artifact) for name, artifact in artifacts.items()}
    validation = {
        name: prediction_metrics([j.duration for j in jobs["runtime_val"]],
                                 model.predict(jobs["runtime_val"]))
        for name, model in models.items()
    }
    selected = min(validation, key=lambda name: validation[name]["mae"])
    # Selection is frozen before the test labels are inspected. No refit on validation.
    actual = [j.duration for j in jobs["runtime_test"]]
    predictions = {name: model.predict(jobs["runtime_test"]) for name, model in models.items()}
    report = {
        "selected_model": selected,
        "selection_rule": "Lowest validation MAE after rounding to positive timesteps.",
        "target": "Execution time from start to finish, excluding queue waiting.",
        "features": [*generator.config.cluster.resources, "requested_duration", "job_type"],
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "generator": generator.config.to_dict(),
        "splits": {
            name: {"stream": (SPLITS + PREDICTOR_SPLITS).index(name),
                   "jobset_indices": list(range(count)), "jobs": len(jobs[name])}
            for name, count in sizes.items()
        },
        "validation": validation,
        "test": {name: prediction_metrics(actual, values) for name, values in predictions.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = {**artifacts[selected], "metadata": report}
    (output / "model.json").write_text(json.dumps(artifact, allow_nan=False) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "model", *validation[selected]])
        writer.writeheader()
        for split in ("validation", "test"):
            writer.writerows({"split": split, "model": name, **metrics}
                             for name, metrics in report[split].items())
    identities = [(i, job.job_id) for i, js in enumerate(splits["runtime_test"]) for job in js]
    with (output / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["jobset", "job_id", "actual_duration", *models])
        writer.writerows([*identity, duration, *values] for identity, duration, *values in
                         zip(identities, actual, *predictions.values()))
    _plot_predictions(actual, predictions, selected, output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime"))
    parser.add_argument("--generator-config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-jobsets", type=int, default=100)
    parser.add_argument("--val-jobsets", type=int, default=20)
    parser.add_argument("--test-jobsets", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        config = (GeneratorConfig.from_dict(json.loads(args.generator_config.read_text()))
                  if args.generator_config else GeneratorConfig())
        report = train_predictor(
            JobTraceGenerator(config), args.output, seed=args.seed,
            train_jobsets=args.train_jobsets, val_jobsets=args.val_jobsets,
            test_jobsets=args.test_jobsets,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
