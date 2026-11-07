# Slurm-Aware

The active project trains and evaluates a DeepRM-style job picker in simulation.
It generates synthetic workloads, lets a policy select a waiting job or wait,
and compares scheduling outcomes using job slowdown.

The earlier priority-weight experiments, Slurm Docker playground, runtime
prediction experiment, and execution-system plan are preserved in
[archive/](archive/README.md). The Rust allocator prototype stays at the
repository root in [allocator/](allocator/README.md).

## Active code

| Location | Purpose |
| --- | --- |
| `src/slurm_aware/deeprm/` | Workload generation, simulation, policies, training, and evaluation |
| `tests/test_deeprm_*.py` | Tests for the active pipeline |
| [allocator/](allocator/README.md) | Rust allocator/compute-node connectivity prototype |
| [docs/deeprm.md](docs/deeprm.md) | Configuration, commands, and implementation details |
| [Meeting 4 slides](<Slurm-Aware Meeting 4_ .pdf>) | Current project direction |
| [DeepRM paper](deeprm_hotnets16.pdf) | Research reference |

Meeting 4 puts simulation and RL evaluation first, extensions such as preemption
and runtime prediction next, and real-cluster integration in a tentative later
phase. The current implementation advances between arrival and completion
events, starts jobs immediately, and exposes a Gymnasium interface for RL
libraries. Future placement and preemption still need implementation.

## Setup and tests

Use Python 3.11 or newer and uv:

```sh
uv sync --extra deeprm
uv run --extra deeprm pytest
```

## Small pipeline check

```sh
uv run --extra deeprm slurm-deeprm-generate --output data/generated/deeprm/jobsets.json
uv run --extra deeprm slurm-deeprm-train --num-jobsets 4 --val-jobsets 2 --episodes-per-jobset 2 --iterations 3 --workers 0 --output artifacts/deeprm/smoke
uv run --extra deeprm slurm-deeprm-evaluate --checkpoint artifacts/deeprm/smoke/best.pt --loads 0.5 0.9 --jobsets 2 --output artifacts/deeprm/smoke/eval
```

The generator command exports an inspectable workload. Training generates its
own workloads from its configuration and seeds. This short run checks the
pipeline; it is not evidence of improved scheduling. See the
[DeepRM guide](docs/deeprm.md) for full experiments and evaluation caveats.

New generated workloads and experiment outputs go under `data/generated/` and
`artifacts/`. Those paths are ignored by Git. Historical outputs are under
`archive/data/` and `archive/artifacts/`.

## DeepRM and Gymnasium

`src/slurm_aware/deeprm/` provides an event-driven simulator where an agent
chooses a visible job to start or NOP to advance to the next event. It models
pooled CPU, GPU and memory capacity and rewards lower job slowdown. It includes
REINFORCE training and SJF, Packer and Tetris* baselines.

Install the optional dependencies and open a Python session:

```sh
uv sync --locked --extra deeprm
uv run --extra deeprm python
```

```python
import gymnasium as gym

env = gym.make("slurm_aware.deeprm.gym_env:SlurmAware-DeepRM-v0")
observation, info = env.reset(seed=42)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

The Gymnasium adapter exposes job-slot/NOP actions and a flat float32
observation of the cluster, visible jobs and backlog. Seeded workload generation,
fixed-trace replay and vector environments are supported. Stable-Baselines3 can
use the adapter directly; install that library separately.

See [the DeepRM guide](docs/deeprm.md) for the event and reward rules, training
and evaluation commands, and [Gymnasium usage](docs/deeprm.md#gymnasium-interface)
for reset options, custom backends and a PPO example. Future reservations and
preemption are not implemented.

## Archived work

[archive/README.md](archive/README.md) catalogs the retired and deferred work.
It preserves the former directory layout, including local simulator changes,
old tests, datasets, reports, and PDF/LaTeX plans. The root package exposes only
the three DeepRM commands; default test discovery and linting exclude the archive.
