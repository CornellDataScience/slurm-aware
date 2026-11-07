# DeepRM: learning to schedule jobs directly

`src/slurm_aware/deeprm/` implements a scheduler inspired by Mao et al.,
"Resource Management with Deep Reinforcement Learning" (HotNets 2016). A policy
network looks at an image of the cluster and of the waiting jobs and picks which
job to start next. It is trained with REINFORCE to minimize average job slowdown
(completion time ÷ ideal duration) and compared with SJF, Packer and Tetris*.

The earlier weight-learning package now lives in `archive/src/slurm_aware/rl/`
(see the [archived RL guide](../archive/docs/rl.md)). It picks one set of Slurm
priority weights per trace. The active DeepRM package makes every scheduling
decision itself, on a pooled cluster model with whole timesteps and resource units.
See the [archive index](../archive/README.md) for the retired and deferred work.

## Modules

| Module | Responsibility |
| --- | --- |
| `jobs.py` | `Job` and `ClusterSpec` (named resources and their capacities) |
| `generator.py` | `JobTraceGenerator`: seeded jobsets, job types, arrival regimes, load, splits |
| `backend.py` | `ClusterBackend` interface and the in-process `InProcessBackend` |
| `env.py` | `DeepRMEnv`: Gym-style `reset()` / `step()`, images, reward, `run_episode` |
| `gym_env.py` | `DeepRMGymEnv`: Gymnasium spaces, seeded workloads, registration |
| `policy.py` | `PolicyNetwork` (paper's MLP on images), `AttentionPolicy` (on per-job numbers), `NetworkPolicy` wrapper |
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
| Clock | one timestep per void/invalid action | jump to the next arrival or completion |
| Backlog | bounded, overflow dropped | unbounded; observation count saturates |

The environment retains M visible job slots, a backlog count, the `{void, 1..M}`
actions, time frozen while valid jobs are scheduled, and the slowdown objective.
Training uses REINFORCE with a per-decision baseline.
The observation omits the original implementation's time-since-last-arrival
feature. It is not an exact reproduction of the original DeepRM environment.

## Event-driven simulator

`reset(jobs)` inserts arrivals into a time-ordered heap, clears the cluster, and
processes arrivals at time zero. Jobs with the same arrival time enter the queue
in job-ID order. A successful scheduling action starts the selected job now,
refills its visible slot from the FIFO backlog, and inserts its completion event
at `now + duration`. It returns zero reward without advancing the clock.

NOP (action 0), an empty slot, or a job that cannot fit advances directly to the
next event, capped at `max_time`. The backend shifts its occupancy window by the
elapsed time and releases completed allocations. The environment processes all
events at that timestamp, completions before arrivals, before returning an
observation. There are no intermediate scheduling decisions during the jump.

The reward is `−Δt × Σ(1 / duration)` over jobs running or waiting **before** the
jump. New arrivals are charged only from their arrival time onward. With
`gamma = 1`, the return of a completed episode is exactly negative total slowdown.
Training's optional discount `gamma < 1` remains per decision, not per time unit.

The episode terminates when all jobs finish. If the next event is beyond
`max_time`, it advances only to that limit and truncates. Completions exactly at
the limit are processed first, so an otherwise finished episode terminates
normally. If a policy chooses NOP with jobs waiting and no future events, it
advances to `max_time`, charges the remaining waiting time, and truncates.

`backlog_size` controls only the number of backlog observation cells. The count
saturates at that size, but every waiting job is retained. `info["backlog"]` gives
the actual length; the existing `dropped` report field remains zero.

For example, on a two-CPU cluster, start A at time zero, wait for B at time 1,
then wait for A's completion and C's arrival at time 3. Start B and C together:

| Job | Arrival | CPU | Duration | Start | Finish | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 2 | 3 | 0 | 3 | 1 |
| B | 1 | 1 | 2 | 3 | 5 | 2 |
| C | 3 | 1 | 1 | 3 | 4 | 1 |

The clock visits `0 → 1 → 3 → 4 → 5`, and total reward is `−4`.
`test_hand_calculated_event_schedule_and_rewards` verifies this schedule.

## Attention policy (optional, not in the paper)

`--policy attention` replaces the paper's MLP over flattened images with a network
that reads the same information as numbers. Only the input and the network
change; the generator, environment rules, reward, actions and REINFORCE are
unchanged, so both policies train and test on identical jobsets.

**Input** (`EnvConfig(observation="features")`, all values in [0, 1]):

| Part | Shape | Contents |
| --- | --- | --- |
| Jobs | M × (R + 2) | per slot: present flag, demand ÷ capacity per resource, duration ÷ T |
| Cluster | T × R | free units ÷ capacity for each of the next T timesteps |
| Backlog | 1 | waiting jobs beyond the slots ÷ `backlog_size`, saturating at 1 |

This drops only the shading that tells running jobs apart in the cluster image.

**Network** (`AttentionPolicy`):

1. One shared job encoder (2-layer MLP) turns each job's demand and duration
   into a vector, so a job gets the same score in any slot.
2. A cluster encoder (2-layer MLP) summarizes the free-capacity grid and backlog.
3. Transformer encoder layers run over the cluster summary plus the visible jobs,
   so each job's vector can depend on the other waiting jobs. Empty slots are
   masked out of attention.
4. Each job is scored by an MLP on (its vector, the cluster summary); void is
   scored from the cluster summary. Empty slots share one learned score, so
   choosing one still behaves exactly as in the image policy.

Defaults, chosen for this branch and not tuned: `embed_dim = 32`,
`attention_heads = 4`, `attention_layers = 1`, feed-forward width `2 × embed_dim`,
no dropout. `--attention-layers 0` keeps the shared encoders but removes
attention, and `--policy mlp --observation features` runs the paper's MLP on the
numbers. Together they separate the effect of the input, the shared encoder and
attention. Like the image observation, it omits time since the last arrival and
how long each job has waited.

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
uv sync --locked --extra deeprm
uv run --extra deeprm pytest tests/test_deeprm_*.py

# Optional: inspect generated jobsets (training regenerates its own from seeds)
uv run --extra deeprm slurm-deeprm-generate --load 1.1 --regime bursty --output data/generated/deeprm/bursty.json

# Train (paper defaults: 100 jobsets x 20 episodes, 1000 iterations, lr 1e-3)
uv run --extra deeprm slurm-deeprm-train --output artifacts/deeprm
uv run --extra deeprm slurm-deeprm-train --output artifacts/deeprm --resume artifacts/deeprm/latest.pt --iterations 1500

# Evaluate on held-out jobsets at loads 10%..190%
uv run --extra deeprm slurm-deeprm-evaluate --checkpoint artifacts/deeprm/best.pt --output artifacts/deeprm/eval
```

Useful training options:
- `--regime bursty`
- `--train-loads 0.5 0.9 1.3`: cycle jobsets through several loads
- `--hidden-sizes 64 32`
- `--policy attention` (with `--embed-dim`, `--attention-heads`,
  `--attention-layers`), or `--observation features` for the MLP on numbers
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

## Gymnasium interface

`DeepRMGymEnv` adapts the event simulator to the
[Gymnasium API](https://gymnasium.farama.org/api/env/). Open a Python session
with the `deeprm` dependencies:

```sh
uv run --extra deeprm python
```

Then use the registered environment:

```python
import gymnasium as gym

env = gym.make("slurm_aware.deeprm.gym_env:SlurmAware-DeepRM-v0")
observation, info = env.reset(seed=42)
env.action_space.seed(42)  # action sampling has its own RNG
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

The action space is `Discrete(M + 1)` (0 = NOP, 1..M = job slots). The observation
space is a flat float32 `Box(0, 1)` containing the existing cluster images, job
images and backlog cells. Empty or non-fitting slots retain their NOP behavior.
Scheduling, event timing and the slowdown reward are unchanged. `max_time`
limits simulated time, not the number of agent decisions. Rendering is not
implemented. `info["scheduled"]` contains the job ID only when a job starts;
otherwise that key is absent so vector environments can mask it. The `time`,
`backlog` and `dropped` info fields retain their existing meanings.

By default, each reset generates a new workload. Repeating `reset(seed=42)`
reproduces a workload; `reset()` continues the seeded random stream. For a
custom workload distribution, pass `generator=JobTraceGenerator(...)` with a
matching cluster spec and `max_duration <= config.lookahead`. The default
generator uses the CPU/GPU/memory job types above; custom resource names require
matching job types in `GeneratorConfig`.

To replay a fixed trace on every reset, construct
`DeepRMGymEnv(spec, config, jobs=jobs)`. To override just one episode, use
`reset(options={"jobs": jobs})`; the next ordinary reset returns to the configured
workload source. An empty trace returns zero reward and `terminated=True` on its
first step. Call `reset()` after either termination or truncation.

The underlying simulator is available as `env.unwrapped.simulator`, including
its cluster spec, records and slowdown metrics. Gymnasium's `env.spec` remains
registration metadata. The existing `DeepRMEnv.reset(jobs)`, heuristics and
REINFORCE training entry points continue to work as before.

The adapter also accepts `backend=` and `track_decisions=`. It can be passed
directly to [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
That library is optional and is not included in the `deeprm` extra. To run the
example below, open a Python session with it added for this invocation:

```sh
uv run --extra deeprm --with stable-baselines3 python
```

```python
from stable_baselines3 import PPO
from slurm_aware.deeprm.gym_env import DeepRMGymEnv

env = DeepRMGymEnv()
model = PPO("MlpPolicy", env, gamma=1.0, seed=42)
model.learn(total_timesteps=10_000)
env.close()
```

As with the existing trainer, discounts below 1 apply per decision. `gamma=1`
preserves the undiscounted slowdown objective.

Run the interface checks with:

```sh
uv run --extra deeprm pytest tests/test_deeprm_gym.py -q
```

They cover Gymnasium's environment checker, seeded reset sequences, fixed and
empty workloads, action validation, event/reward equivalence with `DeepRMEnv`,
termination versus truncation, and vector environment resets.

## Plugging in a real cluster backend

`DeepRMEnv` only calls `ClusterBackend`:

| Method | Contract |
| --- | --- |
| `reset()` | empty cluster at time 0 |
| `available()` | free units, shape `(T, n_resources)`; row 0 is now |
| `allocate(job, now)` | start the job now if it fits for its whole duration, else return `False` |
| `advance_to(now)` | move to the target time, release jobs finished by then, and shift the resource window; return `None` |
| `running()` | jobs holding resources |

`fits()` and `canvas()` have default implementations built on `available()`,
so a backend wrapping the allocator or Slurm does not need any image code. Pass
it as `DeepRMEnv(spec, config, backend=MyBackend(spec, T))`. Its `spec` and
horizon must match the environment's. The environment owns arrival/completion
events and finish records; the backend owns resource availability and running
allocations. Custom backends must implement `advance_to(now)` instead of the old
single-step `advance()` method.

## Caveats

- One pooled cluster with no machine boundaries, as in the paper. The
  [archived node-aware simulator](../archive/docs/simulator.md) remains a
  reference for a separate test of how well results carry over.
- Backlog observation counts saturate (60 by default), even though jobs are never
  dropped. The policy cannot distinguish larger backlogs from that encoding alone.
- Job durations remain known, positive whole time units bounded by the lookahead.
  Jobs start immediately; future reservations and preemption are not implemented.
- Checkpoints trained with timestep advancement remain shape-compatible, but the
  decision timing and backlog semantics have changed. Retrain for comparisons
  under the event-driven environment.
- Truncated episodes count unfinished jobs at their slowdown so far, which is a
  lower bound.
- The paper's "Tetris\*" knob is not fully specified. The weighting above is our
  reading.
