# DeepRM: learning to schedule jobs directly

`src/slurm_aware/deeprm/` replicates and extends the scheduler from Mao et al.,
"Resource Management with Deep Reinforcement Learning" (HotNets 2016). A policy
network looks at an image of the cluster and of the waiting jobs and picks which
job to start next. It is trained with REINFORCE to minimize average job slowdown
(completion time ÷ ideal duration) and compared with SJF, Packer and Tetris*.

It is separate from `src/slurm_aware/rl/` (see `docs/rl.md`). That package picks
one set of Slurm priority weights per trace. This one makes every scheduling
decision itself, on a pooled cluster model with whole timesteps and resource units.

## Modules

| Module | Responsibility |
| --- | --- |
| `jobs.py` | `Job` and `ClusterSpec` (named resources and their capacities) |
| `generator.py` | `JobTraceGenerator`: seeded jobsets, job types, arrival regimes, load, splits |
| `backend.py` | `ClusterBackend` interface and the in-process `InProcessBackend` |
| `env.py` | `DeepRMEnv`: Gym-style `reset()` / `step()`, images, reward, `run_episode` |
| `policy.py` | `PolicyNetwork` (MLP → softmax over actions), `NetworkPolicy` wrapper |
| `train.py` | REINFORCE with the per-decision baseline, workers, logging, checkpoints |
| `baselines.py` | `SJF`, `Packer`, `TetrisStar`, `RandomFit` as env policies |
| `evaluate.py` | Slowdown vs. load, withholding/job-size analysis, training cost, plots |

Each module has its own test file (`tests/test_deeprm_*.py`).

## What changed from the paper

| Aspect | Paper | Here |
| --- | --- | --- |
| Durations | 80% short / 20% long, bimodal | log-normal per job type, rounded, clipped to `[1, T]` |
| Arrivals | Bernoulli per timestep | Poisson count per timestep; `steady`, `bursty` or `diurnal` rate |
| Resources | 2 generic, 10 units each | any named set; default CPU 10, GPU 4, memory 16 units |
| Demands | random dominant resource | job types with correlated per-resource ranges |
| Load | sweep ~10%–190% | same, defined on the bottleneck resource |
| Cluster state | built into the env | behind a swappable `ClusterBackend` |

Everything else follows the paper: M visible job slots, a backlog count, the
`{void, 1..M}` actions, time frozen while valid jobs are scheduled, reward
`−Σ 1/duration` per timestep, and REINFORCE with a time-dependent baseline.

## Defaults chosen during implementation (please review)

These were not in the original spec. All are configurable.

- **Job types** (`DEFAULT_JOB_TYPES`). Demand ranges are fractions of capacity.

  | Type | Weight | CPU | GPU | Memory | Median duration | σ |
  | --- | ---: | --- | --- | --- | ---: | ---: |
  | small_interactive | 0.30 | 0.05–0.15 | – | 0.05–0.10 | 1.5 | 0.6 |
  | cpu_batch | 0.30 | 0.25–0.50 | – | 0.05–0.15 | 4 | 0.8 |
  | mem_analytics | 0.20 | 0.10–0.20 | – | 0.25–0.50 | 5 | 0.8 |
  | gpu_train | 0.20 | 0.10–0.20 | 0.25–0.50 | 0.20–0.40 | 8 | 0.9 |

  `size_correlation = 0.8` blends one shared uniform "size" draw with an
  independent draw per resource, so a job that is large on one resource tends
  to be large on the others.
- **Clipping at T = 20** puts about 5% of jobs exactly at 20 timesteps (measured
  over 400 jobsets). Most of them are `gpu_train`, roughly 1 in 6 of which hits the
  cap. That is the cost of a fixed-width job-slot image. Lowering `gpu_train`'s
  median or σ, or raising T, reduces it.
- **Bursty regime**: square wave with `period = 20`, `duty = 0.25` and
  `peak_ratio = 8`. Arrivals run 8× faster in the first 5 of every 20 timesteps,
  and the long-run mean rate is unchanged. **Diurnal**: `1 + 0.8 sin(2πt/20)`.
- **Cluster** default `(cpu, gpu, mem) = (10, 4, 16)` units.
- **Episodes**: jobs arrive during the first 50 timesteps. The episode runs until
  every job finishes, or truncates at `max_time = 500`.
- **Gradient scale**: the summed REINFORCE gradient is divided by the number of
  decisions in the iteration. RMSprop (α = 0.9, ε = 1e-9, as in the original
  code) is close to scale-invariant, so this only shifts its ε.
- **Packer** scores alignment on capacity-normalized demand and free resources.
  Without that, memory (16 units) would dominate GPU (4 units). **Tetris\***
  scores `0.5 × packer + 0.5 × (1 / duration)`.
- **Convergence** (evaluation): the first iteration after which the 20-iteration
  moving average of training slowdown stays within 5% of its final value.

## Running it

```sh
uv sync --extra deeprm
uv run pytest tests/test_deeprm_*.py

# Optional: inspect generated jobsets (training regenerates its own from seeds)
uv run slurm-deeprm-generate --load 1.1 --regime bursty --output data/generated/deeprm/bursty.json

# Train (paper defaults: 100 jobsets x 20 episodes, 1000 iterations, lr 1e-3)
uv run slurm-deeprm-train --output artifacts/deeprm
uv run slurm-deeprm-train --output artifacts/deeprm --resume artifacts/deeprm/latest.pt --iterations 1500

# Evaluate on held-out jobsets at loads 10%..190%
uv run slurm-deeprm-evaluate --checkpoint artifacts/deeprm/best.pt --output artifacts/deeprm/eval
```

Useful training options:
- `--regime bursty`
- `--train-loads 0.5 0.9 1.3`: cycle jobsets through several loads
- `--hidden-sizes 64 32`
- `--episodes-per-jobset`
- `--workers N`: parallel rollout processes. Results don't depend on N, because
  each (iteration, jobset) has its own sampling seed.

The defaults come from the paper's cleaner synthetic setup. Heavy-tailed,
bursty traces may need a lower learning rate, more episodes per jobset or more
iterations.

**Outputs** in the training directory:
- `train_log.csv`, one row per iteration: mean and max return, mean slowdown,
  entropy, decisions per episode, truncated episodes, validation slowdown every
  `eval_every` iterations, and seconds.
- `latest.pt` and `best.pt` (best greedy validation slowdown). A checkpoint holds
  the full config, so evaluation regenerates the same generator and environment.

**Evaluation** writes:
- `evaluation.md` and `evaluation.json`
- `slowdown_vs_load.png`, `slowdown_by_size.png` and `training_curve.png`

The withholding analysis reports three things:
- *withhold rate*: the share of time-advancing decisions taken while some visible
  job would have fit;
- *withheld-size ratio*: the mean footprint of those jobs ÷ the mean footprint of
  all jobs. Footprint = duration × dominant resource share, and above 1 means the
  policy holds back larger jobs;
- mean slowdown by footprint tercile.

Heuristics never withhold, by construction.

## Plugging in a real cluster backend

`DeepRMEnv` only calls `ClusterBackend`:

| Method | Contract |
| --- | --- |
| `reset()` | empty cluster at time 0 |
| `available()` | free units, shape `(T, n_resources)`; row 0 is now |
| `allocate(job, now)` | start the job now if it fits for its whole duration, else return `False` |
| `advance()` | move one timestep; return `(job, start, finish)` for jobs that finished |
| `running()` | jobs holding resources |

`fits()` and `canvas()` have default implementations built on `available()`,
so a backend wrapping the allocator or Slurm does not need any image code. Pass
it as `DeepRMEnv(spec, config, backend=MyBackend(spec, T))`. Its `spec` and
horizon must match the environment's.

## Caveats

- One pooled cluster with no machine boundaries, as in the paper. The
  node-aware simulator in `docs/simulator.md` would be a separate test of how
  well results carry over.
- Jobs that arrive to a full backlog (60) are dropped, as in the original. They
  contribute no reward and are reported as `dropped`.
- Truncated episodes count unfinished jobs at their slowdown so far, which is a
  lower bound.
- The paper's "Tetris\*" knob is not fully specified. The weighting above is our
  reading.
