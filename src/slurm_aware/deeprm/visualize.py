"""Record one episode of a scheduling policy and write a self-contained HTML viewer.

For every decision the trace keeps the state the policy saw (cluster occupancy by
job, the M visible slots and the backlog length), the checkpoint's action
probabilities, the action taken and its effect: a job started, or time advanced.
The page steps through decisions next to a timeline of every job. It uses no
external scripts, so it opens offline in any browser.

Heuristic policies have no distribution of their own; for them the page shows what
the checkpoint would have done in the same state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

from slurm_aware.deeprm.baselines import heuristics
from slurm_aware.deeprm.env import DeepRMEnv, Policy, summarize_scheduling_metrics
from slurm_aware.deeprm.generator import REGIMES, JobTraceGenerator
from slurm_aware.deeprm.jobs import Job
from slurm_aware.deeprm.policy import NetworkPolicy, PolicyNetwork
from slurm_aware.deeprm.train import (
    generator_config,
    load_checkpoint,
    make_env,
    network_from_checkpoint,
)

POLICIES = ("deeprm", "sjf", "fifo", "packer", "tetris", "random")
TEMPLATE = "policy_viz.html"
PLACEHOLDER = "__TRACE_JSON__"


def cluster_owners(env: DeepRMEnv) -> list[list[int]]:
    """Per resource, the job id in each (unit, timestep) cell, row-major; -1 is free.

    The in-process backend tracks which job owns each cell. Other backends only
    expose a used/free canvas, so their occupied cells are -2 (job unknown).
    """
    owners = getattr(env.backend, "_owner", None)
    if owners is not None:
        return [(owner - 1).ravel().tolist() for owner in owners]
    return [np.where(image > 0, -2, -1).ravel().tolist() for image in env.backend.canvas()]


def _slot(job: Job | None, now: int) -> dict | None:
    if job is None:
        return None
    return {"id": job.job_id, "duration": job.duration, "demand": list(job.demand),
            "waited": now - job.arrival}


def record_episode(
    env: DeepRMEnv,
    policy: Policy,
    jobs: Sequence[Job],
    network: PolicyNetwork | None = None,
) -> dict:
    """Play one episode, keeping a snapshot of the state before every decision."""
    env.reset(jobs)
    steps = []
    while not env.done:
        fitting = env.fitting_slots()
        step = {
            "time": env.time,
            "slots": [_slot(job, env.time) for job in env.slots],
            "backlog": len(env.backlog),
            "fitting": fitting,
            "cluster": cluster_owners(env),
            "probs": None,
        }
        if network is not None:
            probs = network.probabilities(env.observation()[None, :])[0]
            step["probs"] = [round(float(p), 4) for p in probs]
        action = int(policy(env))
        _, reward, _, _, info = env.step(action)
        started = info["scheduled"]
        step.update({
            "action": action,
            "started": started,
            "advanced_to": None if started is not None else env.time,
            "reward": round(float(reward), 4),
            # Time moved on although a visible job could have started.
            "withheld": started is None and bool(fitting),
        })
        steps.append(step)

    metrics = env.scheduling_metrics()
    slowdowns = env.slowdowns(include_unfinished=True)
    summary = summarize_scheduling_metrics([metrics], env.spec)
    job_rows = []
    for record in env.records.values():
        job = record.job
        job_rows.append({
            "id": job.job_id, "arrival": job.arrival, "duration": job.duration,
            "demand": list(job.demand), "type": job.job_type,
            "start": record.start, "finish": record.finish,
        })
    return {
        "resources": list(env.spec.resources),
        "capacities": list(env.spec.capacities),
        "lookahead": env.config.lookahead,
        "num_slots": env.config.num_slots,
        "backlog_size": env.config.backlog_size,
        "jobs": job_rows,
        "steps": steps,
        "summary": {
            "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else 0.0,
            "p95_wait": summary["p95_wait"],
            "p99_wait": summary["p99_wait"],
            "utilization": summary["resource_utilization"],
            "decisions": len(steps),
            "withheld": sum(s["withheld"] for s in steps),
            "timesteps": env.time,
            "truncated": env.truncated,
            "unfinished": sum(r.finish is None for r in env.records.values()),
        },
    }


def render_html(trace: dict) -> str:
    template = resources.files("slurm_aware.deeprm").joinpath(TEMPLATE).read_text()
    # "</" inside the JSON would otherwise close the <script> element early.
    data = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="artifacts/deeprm/best.pt")
    parser.add_argument("--policy", choices=POLICIES, default="deeprm")
    parser.add_argument("--load", type=float, help="Cluster load (default: mean training load)")
    parser.add_argument("--jobset", type=int, default=0,
                        help="Index into the test split, matching slurm-deeprm-evaluate")
    parser.add_argument("--seed", type=int, help="Test seed (default: the training seed)")
    parser.add_argument("--regime", choices=REGIMES, help="Override the arrival regime")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--tetris-weight", type=float, default=0.5)
    parser.add_argument("--output", help="HTML path (default: artifacts/deeprm/viz/...)")
    args = parser.parse_args(argv)
    if args.jobset < 0:
        parser.error("--jobset must be >= 0")
    try:
        state = load_checkpoint(args.checkpoint)
        network, cfg = network_from_checkpoint(state)
        gen_cfg = generator_config(cfg)
        if args.regime:
            gen_cfg = replace(gen_cfg, arrival=replace(gen_cfg.arrival, regime=args.regime))
        seed = cfg.seed if args.seed is None else args.seed
        load = args.load if args.load is not None else float(np.mean(cfg.train_loads))
        generator = JobTraceGenerator(gen_cfg).with_load(load)
        jobs = generator.split("test", args.jobset + 1, seed)[args.jobset]
        if args.policy == "deeprm":
            policy = NetworkPolicy(network, greedy=not args.sample, seed=seed)
        else:
            policy = heuristics(seed, args.tetris_weight)[args.policy]
        trace = record_episode(make_env(cfg), policy, jobs, network)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    trace["meta"] = {
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "load": load,
        "jobset": args.jobset,
        "seed": seed,
        "regime": gen_cfg.arrival.regime,
        "greedy": not args.sample,
    }
    output = Path(args.output or
                  f"artifacts/deeprm/viz/{args.policy}_load{load:g}_jobset{args.jobset}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(trace))
    s = trace["summary"]
    print(f"{args.policy}: {s['decisions']} decisions, mean slowdown {s['mean_slowdown']:.3f}, "
          f"{s['withheld']} held back -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
