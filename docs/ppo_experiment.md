# PPO vs. REINFORCE — initial DeepRM experiment

This adds an **experimental** PPO actor–critic trainer without modifying the
existing event-driven `DeepRMEnv` or REINFORCE implementation.

## Setup

From the existing `CornellDataScience/slurm-aware` repository root:

```bash
uv sync --extra deeprm
uv run python experiments/ppo/train.py --updates 20 --output artifacts/deeprm-ppo
uv run pytest tests/test_deeprm_ppo_experiment.py -q
```

This proof of concept uses a tiny, deterministic Poisson/log-normal synthetic
job generator inside `train.py`, **not** the existing repo's `JobTraceGenerator`.
Do **not** compare its metrics directly to existing REINFORCE numbers trained or
evaluated on other traces.

## Method

- Observations and actions: unchanged; action 0 = NOP, 1..M = schedule slot.
- Reward: unchanged event-driven slowdown penalty.
- Algorithm: clipped PPO policy objective, GAE advantages, value network,
  entropy regularization, Adam, minibatch updates.
- `gamma=1`: a full episode's undiscounted reward corresponds to negative total
  slowdown under the existing environment. `gae_lambda=0.95` reduces variance.
- No action masking: REINFORCE currently permits invalid actions and NOP; do
  not change the action set while comparing algorithms.
- Seeded validation sets do not overlap the generated training seeds.
- Output: `train_log.csv`, `latest.pt` and `best.pt` (standalone PPO format;
  **not interchangeable** with existing REINFORCE checkpoint format).

## What to do before reporting a head-to-head result

1. Replace `make_jobset()` with the **same** `JobTraceGenerator` and train/val/test
   seeds, loads and arrival regimes used by the team's REINFORCE implementation.
2. Evaluate PPO and a freshly retrained event-driven REINFORCE checkpoint on
   identical held-out traces with the **same** environment config and reward.
3. Report mean slowdown, truncated episodes, training cost, and withholding rate.
   Use multiple seeds and report variability, not one cherry-picked run.
4. Tune both algorithms with a comparable budget; record total environment
   decisions, not only the number of optimizer updates.
5. Consider separately testing action masking, since that is a second change
   to the problem and would confound a pure PPO-vs-REINFORCE comparison.

## Suggested PR title

`Experiment: PPO actor-critic baseline for event-driven DeepRM`

## Suggested PR description

> Adds an experimental clipped-PPO trainer that consumes the existing `DeepRMEnv`
> observations and rewards. Includes GAE, value estimation, seeded synthetic
> smoke-test workloads, validation logging and unit tests. This is **not yet a
> reproducible apples-to-apples benchmark** against the existing REINFORCE
> training pipeline; the next step is to share `JobTraceGenerator` and held-out
> evaluation traces between both implementations.
