"""Compare equally trained runtime modes on paired test traces across seeds and regimes."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from slurm_aware.deeprm.baselines import heuristics
from slurm_aware.deeprm.evaluate import evaluate_load
from slurm_aware.deeprm.generator import ArrivalConfig, GeneratorConfig, JobTraceGenerator
from slurm_aware.deeprm.policy import NetworkPolicy
from slurm_aware.deeprm.runtime import RUNTIME_MODES, train_predictor
from slurm_aware.deeprm.train import TrainConfig, load_checkpoint, network_from_checkpoint, train

LOADS = (0.5, 1.0, 1.5)
REGIMES = ("steady", "bursty")


def write_comparison(records, output):
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# Synthetic runtime prediction comparison", "",
             "Each mode has its own RL policy, trained with the same budget and seeds. "
             "All modes replay the same test traces. Training uses steady arrivals; "
             "bursty arrivals test generalization.", "",
             "Values are means over seeds. Truncated/unfinished counts are totals over seeds. "
             "Truncated slowdown and waiting time are lower bounds.", "",
             "| Regime | Load | Mode | Policy | Slowdown | p95 wait | Truncated | Unfinished |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    groups = {}
    for row in records:
        key = (row["regime"], row["load"], row["mode"], row["policy"])
        groups.setdefault(key, []).append(row)
    for (regime, load, mode, policy), rows in groups.items():
        if policy not in ("deeprm", "sjf", "tetris"):
            continue
        lines.append(f"| {regime} | {load:.0%} | {mode} | {policy} | "
                     f"{np.mean([r['mean_slowdown'] for r in rows]):.3f} | "
                     f"{np.mean([r['p95_wait'] for r in rows]):.2f} | "
                     f"{sum(r['truncated_episodes'] for r in rows)} | "
                     f"{sum(r['unfinished_jobs'] for r in rows)} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, regime in enumerate(REGIMES):
        for k, policy in enumerate(("deeprm", "sjf", "tetris")):
            ax = axes[i, k]
            for mode in RUNTIME_MODES:
                values = [[r["mean_slowdown"] for r in groups[regime, load, mode, policy]]
                          for load in LOADS]
                means, stds = np.mean(values, axis=1), np.std(values, axis=1)
                ax.plot(LOADS, means, marker="o", label=mode)
                ax.fill_between(LOADS, np.maximum(0, means - stds), means + stds, alpha=0.12)
            ax.set(title=f"{policy} / {regime}", xlabel="Bottleneck load", ylabel="Mean slowdown")
            ax.legend(fontsize=8)
    title = "Synthetic test traces; mean ± one standard deviation across training seeds"
    if any(row["truncated_episodes"] for row in records):
        title += "\nIncomplete episodes: unfinished-job slowdowns are lower bounds (see report)."
    fig.suptitle(title, fontsize=12)
    fig.savefig(output / "scheduling_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime-comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--train-jobsets", type=int, default=20)
    parser.add_argument("--test-jobsets", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if min(args.iterations, args.train_jobsets, args.test_jobsets) < 1:
        parser.error("Iteration and jobset counts must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    gen = GeneratorConfig()
    template = TrainConfig(
        generator=gen.to_dict(), train_loads=LOADS, num_jobsets=args.train_jobsets,
        val_jobsets=5, num_slots=5, max_time=1000, episodes_per_jobset=4,
        iterations=args.iterations, eval_every=5, workers=args.workers,
    )
    (args.output / "experiment.json").write_text(json.dumps({
        "arguments": {**vars(args), "output": str(args.output)}, "train_config": asdict(template),
        "test_regimes": REGIMES, "test_loads": LOADS,
        "predictor_jobsets": {"train": 100, "val": 20, "test": 20},
    }, indent=2) + "\n")
    records = []
    for seed in args.seeds:
        directory = args.output / f"seed_{seed}"
        prediction = train_predictor(JobTraceGenerator(gen), directory / "predictor", seed=seed)
        artifact = json.loads((directory / "predictor/model.json").read_text())
        print(f"seed {seed}: predictor {prediction['selected_model']}", flush=True)
        for mode in RUNTIME_MODES:
            run_dir = directory / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = replace(template, output=str(run_dir), seed=seed, runtime_mode=mode,
                          runtime_predictor=artifact if mode == "predicted" else None)
            with (run_dir / "train.txt").open("w") as log:
                train(cfg, log=lambda message: print(message, file=log, flush=True))
            network, cfg = network_from_checkpoint(load_checkpoint(run_dir / "best.pt"))
            for regime in REGIMES:
                base = JobTraceGenerator(replace(gen, arrival=ArrivalConfig(regime)))
                for load in LOADS:
                    policies = {"deeprm": NetworkPolicy(network), **heuristics(seed)}
                    results = evaluate_load(policies, cfg, base.with_load(load),
                                            args.test_jobsets, seed)
                    for policy, metrics in results.items():
                        records.append({
                            "seed": seed, "regime": regime, "load": load, "mode": mode,
                            "policy": policy,
                            **{key: metrics[key] for key in
                               ("mean_slowdown", "p95_wait", "p99_wait", "unfinished_jobs",
                                "truncated_episodes", "jobs")},
                            **{f"utilization_{r}": u
                               for r, u in metrics["resource_utilization"].items()},
                            "prediction_mae": metrics["runtime_prediction"].get("mae"),
                        })
            print(f"seed {seed}: {mode} training and evaluation complete", flush=True)
    write_comparison(records, args.output)
    print(f"Results: {args.output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
