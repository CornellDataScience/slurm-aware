"""Command line entry points for generating, training, and predicting."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from slurm_aware.modeling import predict_runtime, train_and_evaluate
from slurm_aware.synthetic import GENERATOR_VERSION, generate_jobs

DEFAULT_DATA = Path("data/generated/jobs.csv")
DEFAULT_OUTPUT = Path("artifacts/runtime")


def write_dataset(path: Path, jobs: int, workloads: int, seed: int) -> None:
    frame = generate_jobs(jobs, workloads, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    metadata = {
        "source": "synthetic; no Slurm jobs were executed",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "jobs": jobs,
        "workloads": workloads,
        "target": "runtime_seconds",
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Generated {len(frame):,} synthetic jobs across {workloads} workload templates: {path}")


def print_report(report: dict, output: Path) -> None:
    rows = pd.DataFrame(report["metrics"])
    print("\nTest metrics (model selection used validation data):")
    print(
        rows.loc[
            rows["split"] == "test", ["model", "mae_seconds", "rmse_seconds", "within_2x_fraction"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nSelected model: {report['selected_model']}")
    print(f"Report, predictions, plot, and saved model: {output}")
    print("These are synthetic-data results, not evidence of real Slurm performance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict execution time for synthetic batch jobs.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "demo"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--jobs", type=int, default=3000)
        subparser.add_argument("--workloads", type=int, default=150)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--data", type=Path, default=DEFAULT_DATA)
        if command == "demo":
            subparser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train = subcommands.add_parser("train")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--seed", type=int, default=42)
    predict = subcommands.add_parser(
        "predict", help="Predict from a CSV; runtime labels are optional."
    )
    predict.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("generate", "demo"):
            write_dataset(args.data, args.jobs, args.workloads, args.seed)
        if args.command in ("train", "demo"):
            print_report(train_and_evaluate(args.data, args.output, args.seed), args.output)
        elif args.command == "predict":
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Use a different output path to preserve the input CSV.")
            bundle = joblib.load(args.model)
            frame = pd.read_csv(args.input)
            frame["predicted_runtime_seconds"] = predict_runtime(bundle["model"], frame)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"Wrote {len(frame)} runtime predictions: {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
