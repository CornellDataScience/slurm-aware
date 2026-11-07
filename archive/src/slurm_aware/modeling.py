"""Train on submission-time features and evaluate unseen workload templates."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from slurm_aware.synthetic import FEATURES, GROUP, NUMERIC_FEATURES, TARGET, validate_jobs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def split_workloads(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Approximately 70/15/15 train/validation/test, with disjoint template IDs."""
    validate_jobs(frame)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_indices, heldout_indices = next(splitter.split(frame, groups=frame[GROUP]))
    heldout = frame.iloc[heldout_indices]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_indices, test_indices = next(splitter.split(heldout, groups=heldout[GROUP]))
    return {
        "train": frame.iloc[train_indices].copy(),
        "validation": heldout.iloc[val_indices].copy(),
        "test": heldout.iloc[test_indices].copy(),
    }


def make_models(seed: int = 42) -> dict:
    numeric = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            (
                "type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["workload_type"],
            ),
        ],
        remainder="drop",
    )
    models = {"median": DummyRegressor(strategy="median")}
    for name, regressor in {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    }.items():
        models[name] = TransformedTargetRegressor(
            regressor=Pipeline([("features", clone(preprocess)), ("regressor", regressor)]),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return models


def predict_runtime(model, frame: pd.DataFrame) -> np.ndarray:
    validate_jobs(frame, training=False)
    predictions = np.asarray(model.predict(frame[FEATURES]), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced non-finite predictions.")
    return np.maximum(predictions, 0.001)


def score_predictions(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "mae_seconds": float(mean_absolute_error(actual, predicted)),
        "rmse_seconds": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error_seconds": float(np.median(np.abs(actual - predicted))),
        "within_2x_fraction": float(
            np.mean(np.maximum(actual / predicted, predicted / actual) <= 2)
        ),
        "underprediction_fraction": float(np.mean(predicted < actual)),
    }


def plot_results(predictions: pd.DataFrame, metrics: pd.DataFrame, best: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    actual = predictions[TARGET]
    predicted = predictions[f"{best}_prediction_seconds"]
    axes[0].scatter(actual, predicted, s=14, alpha=0.5, color="#2563eb", edgecolors="none")
    limits = [min(actual.min(), predicted.min()) * 0.8, max(actual.max(), predicted.max()) * 1.2]
    axes[0].plot(limits, limits, "--", color="#64748b", linewidth=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        xlabel="Synthetic actual runtime (seconds)",
        ylabel="Predicted runtime (seconds)",
        title=f"Validation-selected model: {best}",
    )
    test_metrics = metrics[metrics["split"] == "test"].sort_values("mae_seconds")
    axes[1].barh(test_metrics["model"], test_metrics["mae_seconds"], color="#0d9488")
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Mean absolute error (seconds; lower is better)", title="Held-out test workloads"
    )
    fig.suptitle("Synthetic runtime experiment — not a real-cluster benchmark", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_and_evaluate(data_path: Path, output_dir: Path, seed: int = 42) -> dict:
    frame = pd.read_csv(data_path)
    splits = split_workloads(frame, seed)
    models = make_models(seed)
    records = []
    for name, model in models.items():
        model.fit(splits["train"][FEATURES], splits["train"][TARGET])
        metrics = score_predictions(
            splits["validation"][TARGET], predict_runtime(model, splits["validation"])
        )
        records.append({"model": name, "split": "validation", **metrics})
    # Freeze the choice BEFORE evaluating any model on the test partition.
    best = min(records, key=lambda row: row["mae_seconds"])["model"]
    fit_frame = pd.concat([splits["train"], splits["validation"]])
    predictions = splits["test"].copy()
    for name, model in models.items():
        model.fit(fit_frame[FEATURES], fit_frame[TARGET])
        predicted = predict_runtime(model, predictions)
        predictions[f"{name}_prediction_seconds"] = predicted
        records.append(
            {"model": name, "split": "test", **score_predictions(predictions[TARGET], predicted)}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(records)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    assignment = pd.concat(
        [part[[GROUP]].drop_duplicates().assign(split=name) for name, part in splits.items()]
    )
    assignment.to_csv(output_dir / "split_assignments.csv", index=False)
    report = {
        "target": TARGET,
        "target_definition": "Execution time from start to finish; excludes queue waiting.",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "split": {
            name: {"jobs": len(part), "workloads": part[GROUP].nunique()}
            for name, part in splits.items()
        },
        "selected_model": best,
        "selection_rule": "Lowest validation MAE; refit on train+validation before testing.",
        "metrics": records,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump({"model": models[best], "metadata": report}, output_dir / "model.joblib")
    plot_results(predictions, metrics_frame, best, output_dir / "evaluation.png")
    return report
