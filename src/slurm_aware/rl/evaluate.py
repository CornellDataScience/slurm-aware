"""Compare a trained policy against static default weights on held-out traces.

The policy uses its Gaussian mean (no sampling) for each trace. Baselines are
simulated directly with fixed weights; no search is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from slurm_aware.rl.data import SPLITS, load_traces
from slurm_aware.rl.features import FeatureNormalizer, feature_matrix
from slurm_aware.rl.policy import WEIGHT_NAMES
from slurm_aware.rl.reward import Metrics, SimulatorRunner
from slurm_aware.rl.train import build_policy, greedy_weights, load_checkpoint
from slurm_aware.simulator import PriorityWeights

BASELINES = {
    # Slurm's shipped PriorityWeight* defaults are all zero: every job ties, so FIFO.
    "slurm_default_zero": PriorityWeights(age=0, job_size=0),
    # This simulator's PriorityWeights() default (age=1000, job_size=1000).
    "simulator_default": PriorityWeights(),
}


def summarize(per_trace: Sequence[Metrics]) -> dict[str, float | None]:
    """Pool per-trace metrics: job-weighted means, span-weighted utilization."""
    jobs = [m["completed_jobs"] for m in per_trace]
    spans = [m["makespan_seconds"] for m in per_trace]

    def job_mean(key: str) -> float:
        return math.fsum(m[key] * n for m, n in zip(per_trace, jobs)) / sum(jobs)

    gpu = [
        (m["gpu_utilization"], s)
        for m, s in zip(per_trace, spans)
        if m["gpu_utilization"] is not None
    ]
    return {
        "mean_wait_time_seconds": job_mean("mean_wait_time_seconds"),
        "mean_p95_wait_time_seconds": statistics.fmean(
            m["p95_wait_time_seconds"] for m in per_trace
        ),
        "mean_response_time_seconds": job_mean("mean_response_time_seconds"),
        "mean_slowdown": job_mean("mean_slowdown"),
        "cpu_utilization": math.fsum(m["cpu_utilization"] * s for m, s in zip(per_trace, spans))
        / math.fsum(spans),
        "gpu_utilization": math.fsum(u * s for u, s in gpu) / math.fsum(s for _, s in gpu)
        if gpu
        else None,
    }


def head_to_head(
    policy: Sequence[Metrics], baseline: Sequence[Metrics], objective: str
) -> dict[str, float]:
    """Per-trace comparison: how often and by how much the policy beats a baseline."""
    pairs = [(p[objective], b[objective]) for p, b in zip(policy, baseline)]
    ratios = [p / b for p, b in pairs if b > 0]
    return {
        "win_rate": sum(p < b for p, b in pairs) / len(pairs),
        "tie_rate": sum(p == b for p, b in pairs) / len(pairs),
        "mean_ratio": statistics.fmean(ratios) if ratios else math.nan,
        "median_ratio": statistics.median(ratios) if ratios else math.nan,
    }


def weight_summary(weights: Sequence[PriorityWeights]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": min(getattr(w, name) for w in weights),
            "median": statistics.median(getattr(w, name) for w in weights),
            "max": max(getattr(w, name) for w in weights),
        }
        for name in WEIGHT_NAMES
    }


def evaluate_policy(
    checkpoint: Mapping,
    runner: SimulatorRunner,
    traces: Mapping,
    trace_ids: Sequence[str],
    objective: str,
) -> dict:
    policy = build_policy(checkpoint)
    policy.eval()
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    features = torch.from_numpy(normalizer(feature_matrix(traces, trace_ids, runner.config)))
    learned = greedy_weights(policy, features)
    runs = {"policy": runner.run(list(zip(trace_ids, learned)))}
    for name, weights in BASELINES.items():
        runs[name] = runner.run([(trace_id, weights) for trace_id in trace_ids])
    return {
        "objective": objective,
        "traces": len(trace_ids),
        "summary": {name: summarize(metrics) for name, metrics in runs.items()},
        "policy_vs": {
            name: head_to_head(runs["policy"], runs[name], objective) for name in BASELINES
        },
        "learned_weights": weight_summary(learned),
        "baseline_weights": {name: asdict(w) for name, w in BASELINES.items()},
        "per_trace": {
            trace_id: {
                "policy_weights": asdict(w),
                **{name: runs[name][i][objective] for name in runs},
            }
            for i, (trace_id, w) in enumerate(zip(trace_ids, learned))
        },
    }


def render(report: Mapping) -> str:
    rows = [
        ("Mean wait (s)", "mean_wait_time_seconds", "{:.2f}"),
        ("Mean of per-trace P95 wait (s)", "mean_p95_wait_time_seconds", "{:.2f}"),
        ("Mean response (s)", "mean_response_time_seconds", "{:.2f}"),
        ("Mean slowdown", "mean_slowdown", "{:.3f}"),
        ("CPU utilization", "cpu_utilization", "{:.2%}"),
        ("GPU utilization", "gpu_utilization", "{:.2%}"),
    ]
    names = list(report["summary"])
    lines = [
        f"# Policy evaluation on {report['traces']} traces (objective: {report['objective']})",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for label, key, fmt in rows:
        cells = [report["summary"][n][key] for n in names]
        lines.append(
            f"| {label} | "
            + " | ".join("n/a" if v is None else fmt.format(v) for v in cells)
            + " |"
        )
    lines += [
        "",
        "| Policy vs | Wins | Ties | Mean ratio | Median ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, h in report["policy_vs"].items():
        lines.append(
            f"| {name} | {h['win_rate']:.0%} | {h['tie_rate']:.0%} | "
            f"{h['mean_ratio']:.4f} | {h['median_ratio']:.4f} |"
        )
    lines += ["", "| Learned weight | Min | Median | Max |", "| --- | ---: | ---: | ---: |"]
    for name, s in report["learned_weights"].items():
        lines.append(f"| {name} | {s['min']:.4g} | {s['median']:.4g} | {s['max']:.4g} |")
    lines += [
        "",
        "Ratio = policy objective / baseline objective on the same trace; below 1 means the "
        "policy is better. Results describe these simulated traces only.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="artifacts/rl/best.pt")
    parser.add_argument("--workload", help="Trace CSV (default: the training workload)")
    parser.add_argument("--config", help="Simulator config JSON (default: the training config)")
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="test",
        help="Checkpoint split to evaluate; use 'all' for a separate held-out CSV",
    )
    parser.add_argument("--output", default="artifacts/rl/eval", help="Report directory")
    parser.add_argument("--workers", type=int, default=4, help="0 = no subprocesses")
    args = parser.parse_args(argv)
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = checkpoint["train_config"]
        workload = args.workload or cfg["workload"]
        config = args.config or cfg["config"]
        traces = load_traces(workload)
        trace_ids = sorted(traces) if args.split == "all" else checkpoint["splits"][args.split]
        missing = set(trace_ids) - set(traces)
        if not trace_ids or missing:
            raise ValueError(
                f"Split {args.split!r} does not match {workload}: missing {sorted(missing)[:3]}"
            )
        with SimulatorRunner(workload, config, workers=args.workers) as runner:
            report = evaluate_policy(checkpoint, runner, traces, trace_ids, cfg["objective"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report.update(checkpoint=args.checkpoint, workload=workload, config=config, split=args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    (output / "evaluation.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
