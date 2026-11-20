"""Compare PPO against REINFORCE on identical test workloads."""

from pathlib import Path
import csv
import matplotlib.pyplot as plt
import torch

from slurm_aware.deeprm import evaluate as ev
from experiments.ppo.train import ActorCritic, available_action_mask

SEED = 2026
LOADS = [0.5, 0.9]
N_JOBSETS = 8

OUT = Path("artifacts/ppo-vs-reinforce")
PPO_PATH = "artifacts/deeprm-ppo-masked/best.pt"
REINFORCE_PATH = "artifacts/deeprm/reinforce-20/best.pt"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    state = ev.load_checkpoint(REINFORCE_PATH)
    reinforce_net, cfg = ev.network_from_checkpoint(state)
    generator = ev.JobTraceGenerator(ev.generator_config(cfg))

    saved = torch.load(PPO_PATH, map_location="cpu", weights_only=True)
    ppo = ActorCritic(
        saved["observation_size"],
        saved["n_actions"],
        saved["hidden"],
    )
    ppo.load_state_dict(saved["model_state_dict"])
    ppo.eval()

    env = ev.make_env(cfg)
    if (env.observation_size != saved["observation_size"]
            or env.n_actions != saved["n_actions"]):
        raise ValueError("Checkpoint and environment sizes differ")

    def ppo_policy(environment):
        obs = environment.observation()
        action, _, _ = ppo.choose(
            obs,
            deterministic=True,
            action_mask=available_action_mask(environment),
        )
        return action

    rows = []

    for load in LOADS:
        rules = ev.heuristics(SEED, 0.5)

        policies = {
            "Masked PPO": ppo_policy,
            "REINFORCE": ev.NetworkPolicy(
                reinforce_net, greedy=True, seed=SEED
            ),
            "SJF": rules["sjf"],
            "FIFO": rules["fifo"],
        }

        results = ev.evaluate_load(
            policies, cfg, generator.with_load(load),
            N_JOBSETS, SEED
        )

        for name, result in results.items():
            row = {
                "load": load,
                "scheduler": name,
                "mean_slowdown": float(result["mean_slowdown"]),
                "truncated_episodes": int(result["truncated_episodes"]),
                "cpu_utilization_pct": 100 * float(
                    result["resource_utilization"]["cpu"]
                ),
            }
            rows.append(row)

            print(
                f"{load:.0%} load | {name}: "
                f"slowdown={row['mean_slowdown']:.3f}, "
                f"truncated={row['truncated_episodes']}/{N_JOBSETS}, "
                f"CPU utilization={row['cpu_utilization_pct']:.1f}%",
                flush=True,
            )

    with (OUT / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    graphs = [
        ("mean_slowdown", "Mean slowdown", "slowdown_vs_load.png"),
        ("cpu_utilization_pct", "CPU utilization (%)",
         "cpu_utilization_vs_load.png"),
        ("truncated_episodes", "Truncated episodes",
         "truncations_vs_load.png"),
    ]

    for metric, ylabel, filename in graphs:
        fig, ax = plt.subplots(figsize=(8, 5))

        for name in ("Masked PPO", "REINFORCE", "SJF", "FIFO"):
            data = sorted(
                (r for r in rows if r["scheduler"] == name),
                key=lambda r: r["load"],
            )
            ax.plot(
                [r["load"] * 100 for r in data],
                [r[metric] for r in data],
                marker="o", linewidth=2, label=name,
            )

        ax.set_xlabel("Offered load (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by offered load")
        ax.set_xticks([50, 90])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=200)
        plt.close(fig)

    print("SUCCESS: Comparison complete! Graphs saved in", OUT)


if __name__ == "__main__":
    main()
