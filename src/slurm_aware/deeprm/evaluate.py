"""Evaluate DeepRM against FIFO, SJF, Packer, Tetris* and random on held-out jobsets.

Reports, for test jobsets regenerated from the checkpoint's generator config:

- mean job slowdown vs. cluster (bottleneck) load, per policy;
- "withholding": how often a policy lets time advance while a visible job fits,
  how large the withheld jobs are relative to all jobs, and slowdown by job-size
  bucket (does it delay large jobs so small ones can run?);
- training cost from the run's `train_log.csv`: seconds per iteration and
  iterations to convergence.

Unfinished jobs at truncation count with their slowdown so far (a lower bound).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from slurm_aware.deeprm.baselines import heuristics
from slurm_aware.deeprm.env import Policy, run_episode, summarize_scheduling_metrics
from slurm_aware.deeprm.generator import REGIMES, GeneratorConfig, JobTraceGenerator
from slurm_aware.deeprm.policy import NetworkPolicy
from slurm_aware.deeprm.train import (
    generator_config,
    load_checkpoint,
    make_env,
    network_from_checkpoint,
)

DEFAULT_LOADS = tuple(round(0.1 + 0.2 * i, 1) for i in range(10))  # 10% .. 190%
SIZE_BUCKETS = ("small", "medium", "large")
# Fixed policy colors across comparison plots.
SERIES_COLORS = {
    "deeprm": "#2a78d6",
    "sjf": "#eb6834",
    "fifo": "#8856a7",
    "packer": "#1baf7a",
    "tetris": "#eda100",
    "random": "#e87ba4",
}
SERIES_MARKERS = {"deeprm": "o", "sjf": "s", "fifo": "P", "packer": "^", "tetris": "D",
                  "random": "v"}
LABELS = {"deeprm": "DeepRM", "sjf": "SJF", "packer": "Packer", "tetris": "Tetris*",
          "random": "Random", "fifo": "FIFO"}


def job_footprint(duration: int, demand: Sequence[int], capacities: Sequence[int]) -> float:
    """Duration x dominant resource share: how much of the bottleneck a job occupies."""
    return duration * max(d / c for d, c in zip(demand, capacities))


def evaluate_load(
    policies: Mapping[str, Policy], cfg, generator: JobTraceGenerator, n_jobsets: int, seed: int
) -> dict[str, dict]:
    """Run each policy on the same test jobsets; return per-policy episode and job records."""
    env = make_env(cfg, track_decisions=True)
    caps = env.spec.capacities
    jobsets = generator.split("test", n_jobsets, seed)
    out = {}
    for name, policy in policies.items():
        jobs, episodes, advances, withheld_any, withheld_jobs = [], [], 0, 0, []
        for js in jobsets:
            result = run_episode(env, policy, js)
            episodes.append({k: v for k, v in result.items() if k != "slowdowns"})
            for record in env.records.values():
                job = record.job
                if job.arrival > env.time:
                    continue
                finish = record.finish if record.finish is not None else env.time
                jobs.append({
                    "duration": job.duration,
                    "type": job.job_type,
                    "footprint": job_footprint(job.duration, job.demand, caps),
                    "slowdown": (finish - job.arrival) / job.duration,
                    "wait": (record.start if record.start is not None else env.time)
                    - job.arrival,
                    "finished": record.finish is not None,
                })
            by_id = {job.job_id: job for job in js}
            for decision in env.decisions:
                if decision["scheduled"] is None:
                    advances += 1
                    withheld_any += bool(decision["withheld"])
                    withheld_jobs += [
                        job_footprint(by_id[i].duration, by_id[i].demand, caps)
                        for i in decision["withheld"]
                    ]
        slowdowns = [j["slowdown"] for j in jobs]
        out[name] = {
            **summarize_scheduling_metrics(episodes, env.spec),
            "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else 0.0,
            "mean_wait": float(np.mean([j["wait"] for j in jobs])) if jobs else 0.0,
            "jobs": len(jobs),
            "unfinished_jobs": sum(not j["finished"] for j in jobs),
            "truncated_episodes": sum(e["truncated"] for e in episodes),
            "dropped_jobs": sum(e["dropped"] for e in episodes),
            "withhold_rate": withheld_any / advances if advances else 0.0,
            "mean_withheld_footprint": float(np.mean(withheld_jobs)) if withheld_jobs else None,
            "_jobs": jobs,
        }
    return out


def size_analysis(results: Mapping[str, dict]) -> dict:
    """Slowdown per footprint tercile; the same jobs appear for every policy."""
    pooled = np.array([j["footprint"] for r in results.values() for j in r["_jobs"]])
    if not len(pooled):
        return {"edges": [], "policies": {}}
    edges = np.quantile(pooled, [1 / 3, 2 / 3]).tolist()
    analysis = {}
    for name, r in results.items():
        buckets = {b: [] for b in SIZE_BUCKETS}
        for j in r["_jobs"]:
            buckets[SIZE_BUCKETS[int(np.searchsorted(edges, j["footprint"], side="right"))]].append(
                j["slowdown"]
            )
        all_footprints = [j["footprint"] for j in r["_jobs"]]
        analysis[name] = {
            "slowdown_by_size": {b: float(np.mean(v)) if v else None for b, v in buckets.items()},
            "jobs_by_size": {b: len(v) for b, v in buckets.items()},
            "withhold_rate": r["withhold_rate"],
            # > 1 means the jobs a policy holds back are larger than the typical job.
            "withheld_footprint_ratio": r["mean_withheld_footprint"] / float(
                np.mean(all_footprints)
            )
            if r["mean_withheld_footprint"] is not None and all_footprints
            else None,
        }
    return {"edges": edges, "policies": analysis}


def convergence(
    rows: Sequence[Mapping[str, str]], window: int = 20, tolerance: float = 0.05
) -> dict:
    """Training cost and convergence from train_log.csv.

    Converged = first iteration after which the `window`-iteration moving average of
    training mean slowdown stays within `tolerance` (relative) of its final value.
    """
    if not rows:
        return {}
    seconds = np.array([float(r["iteration_seconds"]) for r in rows])
    slowdown = np.array([float(r["mean_slowdown"]) for r in rows])
    w = max(1, min(window, len(slowdown)))
    smooth = np.convolve(slowdown, np.ones(w) / w, mode="valid")
    final = smooth[-1]
    outside = np.flatnonzero(np.abs(smooth - final) > tolerance * final)
    first_inside = int(outside[-1] + 1) if len(outside) else 0
    val = [(int(r["iteration"]), float(r["val_slowdown"])) for r in rows if r["val_slowdown"]]
    best = min(val, key=lambda v: v[1]) if val else None
    return {
        "iterations": len(rows),
        "mean_iteration_seconds": float(seconds.mean()),
        "median_iteration_seconds": float(np.median(seconds)),
        "total_training_seconds": float(seconds.sum()),
        # Index into the smoothed curve -> the iteration at the end of that window.
        "iterations_to_convergence": int(rows[first_inside + w - 1]["iteration"]) + 1,
        "convergence_window": w,
        "convergence_tolerance": tolerance,
        "final_smoothed_train_slowdown": float(final),
        "best_val_iteration": best[0] if best else None,
        "best_val_slowdown": best[1] if best else None,
    }


def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(True, axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")


def _log_y(ax) -> None:
    import matplotlib.ticker as ticker

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def plot_all(report: Mapping, train_rows: Sequence[Mapping[str, str]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    loads = report["loads"]
    names = list(report["by_load"][str(loads[0])])

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    values = {n: [report["by_load"][str(x)][n]["mean_slowdown"] for x in loads] for n in names}
    for n in names:
        ax.plot(loads, values[n], color=SERIES_COLORS.get(n), marker=SERIES_MARKERS.get(n),
                markersize=6, linewidth=2, label=LABELS.get(n, n))
    flat = [v for vs in values.values() for v in vs if v > 0]
    if flat and max(flat) / min(flat) > 10:
        _log_y(ax)
    _setup_axes(ax, "Mean job slowdown vs. cluster load", "Bottleneck load", "Mean slowdown")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    # Outside the plot: five close series leave no room for in-plot labels.
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(output / "slowdown_vs_load.png", dpi=150)
    plt.close(fig)
    written.append("slowdown_vs_load.png")

    sizes = report["size_analysis"]["policies"]
    if sizes:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        width = 0.8 / len(names)
        for i, n in enumerate(names):
            heights = [sizes[n]["slowdown_by_size"][b] or 0 for b in SIZE_BUCKETS]
            x = np.arange(len(SIZE_BUCKETS)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, heights, width * 0.92, color=SERIES_COLORS.get(n), label=LABELS.get(n, n),
                   edgecolor="#fcfcfb", linewidth=1)
        ax.set_xticks(range(len(SIZE_BUCKETS)),
                      [f"{b} jobs" for b in SIZE_BUCKETS])
        _setup_axes(ax, f"Slowdown by job size at load {report['analysis_load']:.0%}",
                    "Job footprint tercile (duration x dominant share)", "Mean slowdown")
        ax.legend(frameon=False, fontsize=9, ncols=len(names))
        fig.tight_layout()
        fig.savefig(output / "slowdown_by_size.png", dpi=150)
        plt.close(fig)
        written.append("slowdown_by_size.png")

    if train_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        it = [int(r["iteration"]) for r in train_rows]
        ax.plot(it, [float(r["mean_slowdown"]) for r in train_rows], color="#2a78d6",
                linewidth=1, alpha=0.45, label="Train (sampled policy)")
        val = [(int(r["iteration"]), float(r["val_slowdown"])) for r in train_rows
               if r["val_slowdown"]]
        if val:
            ax.plot(*zip(*val), color="#2a78d6", marker="o", markersize=5, linewidth=2,
                    label="Validation (greedy)")
        sjf = report.get("training", {}).get("sjf_val_slowdown")
        if sjf:
            ax.axhline(sjf, color="#eb6834", linewidth=1.5, linestyle="--",
                       label="SJF (validation)")
        conv = report.get("training", {}).get("iterations_to_convergence")
        if conv:
            ax.axvline(conv - 1, color="#c3c2b7", linewidth=1)
            ax.annotate("converged", (conv - 1, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=9,
                        color="#52514e")
        _log_y(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _setup_axes(ax, "Training curve", "Iteration", "Mean slowdown")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(output / "training_curve.png", dpi=150)
        plt.close(fig)
        written.append("training_curve.png")
    return written


def render(report: Mapping) -> str:
    loads = report["loads"]
    names = list(report["by_load"][str(loads[0])])
    header = "| Load | " + " | ".join(LABELS.get(n, n) for n in names) + " |"
    rule = "| ---: | " + " | ".join("---:" for _ in names) + " |"
    lines = [f"# DeepRM evaluation ({report['jobsets_per_load']} test jobsets per load, "
             f"regime: {report['regime']})", "", "## Mean job slowdown", "", header, rule]
    for x in loads:
        row = report["by_load"][str(x)]
        lines.append(f"| {x:.0%} | " + " | ".join(f"{row[n]['mean_slowdown']:.3f}"
                                                   for n in names) + " |")
    for metric, title in (("p95_wait", "p95 wait time (timesteps)"),
                          ("p99_wait", "p99 wait time (timesteps)")):
        lines += ["", f"## {title}", "", header, rule]
        for x in loads:
            row = report["by_load"][str(x)]
            lines.append(f"| {x:.0%} | " + " | ".join(f"{row[n][metric]:.2f}"
                                                       for n in names) + " |")
    resources = report["by_load"][str(loads[0])][names[0]]["resource_utilization"]
    for resource in resources:
        lines += ["", f"## {resource} utilization", "", header, rule]
        for x in loads:
            row = report["by_load"][str(x)]
            lines.append(f"| {x:.0%} | " + " | ".join(
                f"{row[n]['resource_utilization'][resource]:.1%}" for n in names
            ) + " |")
    lines += ["", "Wait percentiles pool all arrived jobs; unstarted jobs contribute their "
              "wait so far (a lower bound). Utilization is resource-unit time divided by "
              "capacity times elapsed episode time, including initial idle time and draining."]
    lines += ["", "## Withhold rate (time advanced while a visible job fit)", "", header, rule]
    for x in loads:
        row = report["by_load"][str(x)]
        lines.append(f"| {x:.0%} | " + " | ".join(f"{row[n]['withhold_rate']:.1%}"
                                                   for n in names) + " |")
    issues = [
        f"{LABELS.get(n, n)} @ {x:.0%}: {r['truncated_episodes']} truncated episodes, "
        f"{r['unfinished_jobs']} unfinished, {r['dropped_jobs']} dropped jobs"
        for x in loads
        for n, r in report["by_load"][str(x)].items()
        if r["truncated_episodes"] or r["dropped_jobs"]
    ]
    if issues:
        lines += ["", "Incomplete runs (slowdown of unfinished jobs is a lower bound):", ""]
        lines += [f"- {i}" for i in issues]

    sa = report["size_analysis"]
    if sa["policies"]:
        lines += [
            "",
            f"## Job size analysis at load {report['analysis_load']:.0%}",
            "",
            "Size = duration x dominant resource share; terciles over all test jobs "
            f"(edges {sa['edges'][0]:.2f}, {sa['edges'][1]:.2f}). Withheld-size ratio > 1 means "
            "the jobs a policy holds back while they fit are larger than average.",
            "",
            "| Policy | Small | Medium | Large | Withhold rate | Withheld-size ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for n, a in sa["policies"].items():
            s = a["slowdown_by_size"]
            ratio = a["withheld_footprint_ratio"]
            lines.append(
                f"| {LABELS.get(n, n)} | "
                + " | ".join("n/a" if s[b] is None else f"{s[b]:.3f}" for b in SIZE_BUCKETS)
                + f" | {a['withhold_rate']:.1%} | "
                + ("n/a" if ratio is None else f"{ratio:.2f}")
                + " |"
            )
    t = report.get("training")
    if t:
        lines += [
            "",
            "## Training cost",
            "",
            f"- Iterations run: {t['iterations']}",
            f"- Seconds per iteration: mean {t['mean_iteration_seconds']:.2f}, "
            f"median {t['median_iteration_seconds']:.2f} "
            f"(total {t['total_training_seconds'] / 60:.1f} min)",
            f"- Iterations to convergence: {t['iterations_to_convergence']} "
            f"({t['convergence_window']}-iteration moving average of training slowdown within "
            f"{t['convergence_tolerance']:.0%} of its final value)",
        ]
        if t.get("best_val_iteration") is not None:
            lines.append(f"- Best validation slowdown {t['best_val_slowdown']:.3f} at iteration "
                         f"{t['best_val_iteration']} (SJF on validation: "
                         f"{t.get('sjf_val_slowdown', float('nan')):.3f})")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="artifacts/deeprm/best.pt")
    parser.add_argument("--train-log", help="train_log.csv (default: next to the checkpoint)")
    parser.add_argument("--output", default="artifacts/deeprm/eval")
    parser.add_argument("--loads", type=float, nargs="+", default=list(DEFAULT_LOADS))
    parser.add_argument("--jobsets", type=int, default=20, help="Test jobsets per load")
    parser.add_argument("--seed", type=int, help="Test seed (default: the training seed)")
    parser.add_argument("--regime", choices=REGIMES, help="Override the arrival regime")
    parser.add_argument("--analysis-load", type=float,
                        help="Load for the size analysis (default: sweep load nearest training)")
    parser.add_argument("--tetris-weight", type=float, default=0.5)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = load_checkpoint(args.checkpoint)
        network, cfg = network_from_checkpoint(state)
        gen_cfg: GeneratorConfig = generator_config(cfg)
        if args.regime:
            gen_cfg = replace(gen_cfg, arrival=replace(gen_cfg.arrival, regime=args.regime))
        seed = cfg.seed if args.seed is None else args.seed
        base = JobTraceGenerator(gen_cfg)
        loads = [round(x, 4) for x in args.loads]
        target = args.analysis_load or float(np.mean(cfg.train_loads))
        analysis_load = min(loads, key=lambda x: abs(x - target))

        report: dict = {
            "checkpoint": args.checkpoint,
            "regime": gen_cfg.arrival.regime,
            "loads": loads,
            "jobsets_per_load": args.jobsets,
            "seed": seed,
            "greedy": not args.sample,
            "analysis_load": analysis_load,
            "by_load": {},
        }
        for load in loads:
            policies = {"deeprm": NetworkPolicy(network, greedy=not args.sample, seed=seed),
                        **heuristics(seed, args.tetris_weight)}
            results = evaluate_load(policies, cfg, base.with_load(load), args.jobsets, seed)
            if load == analysis_load:
                report["size_analysis"] = size_analysis(results)
            report["by_load"][str(load)] = {
                n: {k: v for k, v in r.items() if not k.startswith("_")}
                for n, r in results.items()
            }
            print(f"load {load:.0%}: " + ", ".join(
                f"{LABELS[n]} {r['mean_slowdown']:.3f}" for n, r in results.items()), flush=True)

        log_path = Path(args.train_log) if args.train_log else (
            Path(args.checkpoint).parent / "train_log.csv")
        train_rows = list(csv.DictReader(log_path.open())) if log_path.exists() else []
        if train_rows:
            report["training"] = {
                **convergence(train_rows, args.convergence_window, args.convergence_tolerance),
                "sjf_val_slowdown": state.get("sjf_val_slowdown"),
            }
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    (output / "evaluation.md").write_text(text)
    if not args.no_plots:
        plot_all(report, train_rows, output)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
