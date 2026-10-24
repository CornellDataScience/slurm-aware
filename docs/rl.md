# Learning priority weights with a policy network

This guide explains the training code in `src/slurm_aware/rl/`, assuming no
background in reinforcement learning (RL). It has three parts: the ideas, how
the code is organized, and how to run training and evaluation.

## The problem in one paragraph

Slurm decides which waiting job runs next by computing a **priority** for each
job: a weighted sum of factors such as how long the job has waited (age),
how much its account has used recently (fair share), how big it is (job size),
its QoS, and its resource requests (TRES). The **weights** are site
configuration, and choosing them is guesswork. Our simulator
(`docs/simulator.md`) can replay a set of jobs, the "job set" or "trace", under
any weight vector and report how long jobs waited. We want a model that looks at
a job set and proposes weights that make jobs wait less than Slurm's defaults.

The simulator is a black box to the model. It is a discrete event program, not a
smooth mathematical function, so we cannot ask it "which direction should the
weights move to reduce waiting?" the way ordinary neural network training does.
The method below learns by trial and error instead.

## Key ideas

### What a policy network is and what it does here

A **policy** is a rule for choosing an action given what you observe. Here:

- **What it observes (the context):** 18 summary statistics of a job set. Each
  statistic is a single number: how many jobs there are, how long they run, how
  much users over-request their time limit, how large the resource requests are,
  how much of the cluster's capacity the jobs would use, how evenly submissions
  are split between accounts, and how bursty arrivals are. The full list is in
  `features.py`. The numbers are standardized: each one is shifted and scaled
  using training-set statistics so they share a common scale.
- **The action:** a priority weight vector with 7 entries: `age`, `fairshare`,
  `job_size`, `qos`, `tres_cpu`, `tres_memory` and `tres_gpu`. `partition` and
  `association` are left out because with the current CDS config they are
  identical for every job, so no weight on them can change the order.

The **policy network** is a small neural network (a multilayer perceptron: two
hidden layers of 64 units) that turns the 18 numbers into a *recommendation*
for the 7 weights. Training adjusts the network so its recommendations produce
less waiting.

Weights are learned on a **log scale**. The network outputs a number `z` for
each weight, and the actual weight is about `10^z`: `z = 3` means 1,000 and
`z = 6` means 1,000,000. Useful Slurm weights span many orders of magnitude
because each factor lies between 0 and 1 and the priority is rounded down to an
integer. On a log scale, "make age 10× more important" is a step of size 1
whatever the starting value. The exact conversion (`policy.py`) is:

- integer weights: `round(10^z)`, capped at 10^9; any `z ≤ 0` means weight 0 (factor off)
- TRES weights, which may be negative: `sign(z) · 10^|z|`, capped at ±10^9

### What "sampling an action" means and why randomness is involved

The network does not output one weight vector. For each weight it outputs a
**mean** (its best guess for `z`) and a **standard deviation** (how far to
wander from that guess). Together these describe a bell curve (a Gaussian
distribution) for each weight. **Sampling an action** means drawing a random
weight vector from those bell curves, the way you would draw a random number
from a distribution.

Why not always use the best guess? Because the simulator tells us nothing about
weights we didn't try. If the model only ever tried its current guess, it would
get back one number ("this guess produced 110 s of waiting") and no hint of
whether more age weight or less QoS weight would have been better. Random
variation around the guess is how it **explores**. Some samples happen to do
better than usual and some worse. Training shifts the mean toward the ones that
did better. That comparison of nearby tries replaces the gradient that the
simulator cannot give us.

The standard deviation is learned too. It starts at 0.5 (about a factor of 3 in
weight either side of the mean). As training finds which directions help, the
standard deviation for weights that matter can shrink, so the model explores
less and exploits more. At **evaluation time there is no randomness**: we use
the mean.

### What a reward is

The **reward** is the score a sampled action earns; higher is better. We
optimize mean queue wait time. Wait varies a lot from trace to trace (roughly
40–200 s here), mostly because some traces are just busier. To focus the score
on the weights rather than the trace, the reward compares against a fixed
reference: the simulator's default weights (`age=1000`, `job_size=1000`) run
once on the same trace:

```
reward = −(policy's mean wait on this trace) / (default weights' mean wait on this trace)
```

So −1.00 means "as good as the default", −0.95 means 5% less waiting, and
−1.05 means 5% more. The reference waits are computed once at the start of
training and stored in the checkpoint.

### What the baseline and advantage are, and why advantage is used

The **baseline** is a running average of recent rewards: what the model
"usually" scores right now. Concretely it is an exponential moving average.
After each batch, `baseline ← 0.9 · baseline + 0.1 · (batch's average reward)`.

The **advantage** of a sample is `reward − baseline`: how much better or worse
it did than usual.

Why not use the raw reward? Every reward here is negative (around −1). If we
used rewards directly, the update for every sample would say "make this action
*less* likely", and the good samples would differ from the bad ones only by
degree. The learning signal would be noisy, and the model would mostly learn
from an unhelpful overall pressure. Subtracting the baseline centres the
signal. Better-than-usual samples get a positive advantage (do more of this).
Worse-than-usual samples get a negative one (do less of this). This does not
change what the model converges toward on average. It just makes each update
far less noisy, which is the main reason simple trial-and-error methods train
at all.

### Why this is a contextual bandit, not general RL

General RL deals with **sequences**: an action changes the world's *state*,
which affects the next decision, and rewards may arrive long after the action
that earned them (think of a chess move). That requires extra machinery: state
transitions, discounting of future rewards, and often a second "critic" network
to estimate long-term value.

Our setting has none of that. For each job set the model makes **one decision**
(a weight vector for the whole replay) and gets **one reward** when the replay
finishes. One trace's choice does not affect any other trace. Each trace is
independent, like separate rounds of a game. This one-shot, "observe context →
act → get reward" setting is called a **contextual bandit**. The name comes from
choosing which slot-machine ("one-armed bandit") arm to pull, where the best arm
depends on the situation.

Practically, this means:

- no discount factor, value function or critic; the code has none
- each training example is just (features, sampled weights, reward), and
  examples can be shuffled and batched freely
- the simulator can replay all of a batch's traces in parallel, since they don't
  interact

Inside one replay the simulator does make many scheduling decisions over time,
but the *learner* only chooses the weights once, up front. Those internal
decisions belong to the simulator, not the policy.

### One training step, start to finish

`reinforce_step` in `train.py` does the following for a batch of 32 traces.
Here is what happens to one of them:

1. **Job set in.** Take trace `trace-417` (200 jobs). Its 18 standardized
   features were computed once at startup.
2. **Policy output.** The network maps the features to a mean `z` for each of
   the 7 weights, e.g. `age 3.0, fairshare 0.0, job_size 3.0, …`. The standard
   deviation for each is, e.g., 0.5.
3. **Sample.** Draw one random `z` per weight from those bell curves, e.g.
   `age 3.4, fairshare −0.2, job_size 2.7, …`. Convert to real weights:
   `age = 2512, fairshare = 0, job_size = 501, …`.
4. **Simulator run.** Replay `trace-417` with those weights (in a worker
   process). Out comes the mean wait, say 98.0 s.
5. **Reward.** The default weights waited 104.0 s on this trace, so
   `reward = −98.0 / 104.0 = −0.942`.
6. **Advantage.** The baseline is −0.990, so `advantage = −0.942 − (−0.990) =
   +0.048`. This sample did better than usual.
7. **Policy update.** Compute the **log-probability** of the sampled `z` under
   the network's bell curves: how likely the network was to pick exactly that
   action. The loss for the batch is `−mean(advantage × log-probability)`.
   Backpropagation and the Adam optimizer (learning rate 3e-4) nudge the network
   so that this sample's `z` becomes *more* likely next time, because its
   advantage was positive. Mostly this moves the mean toward `age 3.4,
   job_size 2.7`. Samples with negative advantage are pushed the other way.
   Gradients flow only through the log-probability, which is ordinary math the
   network computed. They never go through the simulator, which is why the
   simulator doesn't need to be differentiable. This trick is called
   **REINFORCE** (or the "score-function" or "likelihood-ratio" gradient).
8. **Baseline update.** The baseline moves 10% toward this batch's average
   reward.

One **epoch** is 25 such steps, covering all 800 training traces once in a new
random order. After each epoch the model is scored on 100 validation traces
using its mean (no sampling), and checkpoints are saved.

## How the code is organized

| Module | Responsibility | Can be tested without |
| --- | --- | --- |
| `rl/data.py` | Load the trace CSV (the existing `read_traces`; the schema already matches) and make a seeded 80/10/10 split by whole trace | torch |
| `rl/features.py` | `extract_features(jobs, config)` → 18-number vector; `FeatureNormalizer` | torch, simulator runs |
| `rl/policy.py` | `GaussianPolicy` network; `action_to_weights` (`z` → `PriorityWeights`) | the simulator |
| `rl/reward.py` | `SimulatorRunner` (parallel replays), reference objectives, `compute_reward`, `EMABaseline` | torch |
| `rl/train.py` | `reinforce_step`, validation, checkpointing, training CLI | — |
| `rl/evaluate.py` | Held-out comparison against static weights, evaluation CLI | — |

Tests are in `tests/test_rl.py`. One of them checks that `reinforce_step`
learns the correct direction against a fake simulator, without running the real
one.

## Running it

### Setup

The simulator itself has no dependencies. The RL code needs PyTorch and NumPy,
which are an optional extra:

```sh
uv sync --extra rl
uv run pytest tests/test_rl.py      # quick sanity check, a few seconds
```

### (a) Train from scratch

With all defaults (trace CSV `data/generated_traces/trace_job_1.csv`, config
`examples/cds_cluster.json`, output `artifacts/rl/`):

```sh
uv run slurm-rl-train
```

The same run with every setting spelled out:

```sh
uv run slurm-rl-train \
  --workload data/generated_traces/trace_job_1.csv \
  --config examples/cds_cluster.json \
  --output artifacts/rl \
  --objective mean_wait_time_seconds \
  --epochs 50 --batch-size 32 --lr 3e-4 \
  --hidden-sizes 64 64 --initial-std 0.5 --baseline-decay 0.9 \
  --seed 0 --split 0.8 0.1 0.1 --workers 8
```

| Argument | Meaning |
| --- | --- |
| `--workload` | Trace CSV in the simulator schema |
| `--config` | Simulator cluster config; **must be the config the traces were generated for** |
| `--output` | Directory for checkpoints and logs; a fresh run overwrites its logs |
| `--objective` | Any per-trace metric key from `simulate()` to minimize, e.g. `p95_wait_time_seconds` or `mean_slowdown` |
| `--epochs` | Passes over the training traces |
| `--batch-size` | Traces (independent bandit rounds) per policy update |
| `--lr` | Adam learning rate |
| `--initial-std` | Starting exploration width, in log10 units |
| `--baseline-decay` | How slowly the baseline moves (closer to 1 = slower) |
| `--seed` | Controls the train/val/test split, network initialization, sampling and shuffling |
| `--split` | Train/validation/test fractions, by whole trace |
| `--workers` | Parallel simulator processes (default: CPUs − 1, at most 8; `0` = none) |
| `--resume` | Continue from a checkpoint (see below) |

The 1000-trace dataset takes about 6 s per epoch with 8 workers, so about 5
minutes for 50 epochs.

**Output files** in `artifacts/rl/`:

- `train_log.csv`: one row per step (reward, baseline, advantage, loss, mean wait, mean std)
- `val_log.jsonl`: one line per epoch (validation mean wait, ratio vs. default, win rate)
- `latest.pt`: checkpoint after the most recent epoch
- `best.pt`: checkpoint with the best validation ratio so far

**What the logs look like**, from a real run of the defaults:

```text
Simulating reference (default) weights on train/val traces...
epoch   0 step     5 | reward -1.0034 baseline -0.9968 adv -0.0065 loss -0.0157 | mean_wait_time_seconds  113.42 std 0.500
epoch   0 step    10 | reward -0.9944 baseline -0.9981 adv +0.0037 loss +0.0347 | mean_wait_time_seconds  107.55 std 0.500
epoch   0 step    15 | reward -1.0233 baseline -0.9984 adv -0.0249 loss -0.1361 | mean_wait_time_seconds   97.80 std 0.500
== epoch   0 validation: mean_wait_time_seconds 103.39 | ratio vs default 0.9962 | beats default on 57% of traces | new best
...
== epoch  31 validation: mean_wait_time_seconds 101.71 | ratio vs default 0.9855 | beats default on 53% of traces | new best
...
epoch  45 step  1145 | reward -0.9706 baseline -0.9932 adv +0.0226 loss +0.0496 | mean_wait_time_seconds  108.23 std 0.494
epoch  45 step  1150 | reward -0.9894 baseline -0.9881 adv -0.0014 loss -0.0299 | mean_wait_time_seconds   99.63 std 0.494
== epoch  45 validation: mean_wait_time_seconds 101.55 | ratio vs default 0.9833 | beats default on 58% of traces | new best
...
== epoch  49 validation: mean_wait_time_seconds 102.69 | ratio vs default 0.9957 | beats default on 49% of traces
{"best_val_ratio": 0.9832598558457645, "steps": 1250, "output": "artifacts/rl"}
```

How to read them:

- `reward`: the batch's average reward (−1.00 = same as default). It is very
  noisy from step to step, because each batch is different traces with randomly
  sampled weights. With the defaults, the upward drift is small (about −1.00 to
  −0.99) and only visible in the baseline.
- `baseline`: a smoothed version of `reward`; watch this for the trend.
- `adv`: the average advantage. It hovers near 0 by construction; that is normal.
- `loss`: **not** a quality measure in REINFORCE. Its size and sign jump around
  and it does not trend toward zero. Ignore it except for spotting `nan`.
- `mean_wait_time_seconds`: the batch's average wait with sampled weights.
  It's noisy because the traces differ.
- `std`: exploration width. A slow decline means the policy is growing more
  confident. With `--lr 3e-4` it barely moves in 50 epochs (0.500 → 0.494). A collapse to about 0.05 (the lower clamp) early on means it
  stopped exploring, so try a lower `--lr`.
- `== validation` lines are the reliable signal: mean wait with the policy's
  best-guess weights on the same 100 held-aside traces each epoch, the average
  ratio to default (below 1 is better), and the fraction of traces where it
  beats default. `best.pt` is updated when the ratio improves.

**Resume** after an interruption, or to train longer. The split, feature
scaling, reference waits, optimizer state and baseline all come from the
checkpoint. `--epochs` is the new *total*:

```sh
uv run slurm-rl-train --output artifacts/rl --resume artifacts/rl/latest.pt --epochs 80
```

Checkpoints are written once per epoch, so at most one epoch of work is lost.

### (b) Evaluate on held-out traces

```sh
uv run slurm-rl-evaluate --checkpoint artifacts/rl/best.pt --split test --output artifacts/rl/eval
```

This uses the 100 test traces that training never saw: neither the training nor
the validation split. To evaluate on a different CSV entirely (e.g. traces
generated with another seed), pass `--workload new.csv --split all`, and
`--config` if needed.

The policy is compared with two **static weight vectors simulated directly, with
no search**:

- `slurm_default_zero`: Slurm's shipped defaults. Every `PriorityWeight*` is 0,
  so all jobs tie and the order is first-come first-served (with backfill).
- `simulator_default`: this simulator's `PriorityWeights()` (`age=1000,
  job_size=1000`), the training reference.

Real output from the run above:

```text
| Metric | policy | slurm_default_zero | simulator_default |
| --- | ---: | ---: | ---: |
| Mean wait (s) | 107.68 | 115.40 | 110.60 |
| Mean of per-trace P95 wait (s) | 470.67 | 404.31 | 457.89 |
| Mean response (s) | 132.09 | 139.81 | 135.01 |
| Mean slowdown | 10.664 | 12.708 | 11.066 |
| CPU utilization | 63.60% | 64.43% | 63.87% |
| GPU utilization | 52.11% | 52.79% | 52.33% |

| Policy vs | Wins | Ties | Mean ratio | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| slurm_default_zero | 79% | 0% | 0.9332 | 0.9123 |
| simulator_default | 51% | 3% | 0.9857 | 0.9992 |

| Learned weight | Min | Median | Max |
| --- | ---: | ---: | ---: |
| age | 379 | 1155 | 3181 |
| fairshare | 0 | 0 | 1 |
| job_size | 67 | 516.5 | 1.034e+04 |
| qos | 0 | 3 | 34 |
| tres_cpu | -1.519 | 1.018 | 1.478 |
| tres_memory | -3.529 | 1.494 | 4.263 |
| tres_gpu | -6.104 | 2.386 | 12.26 |

Ratio = policy objective / baseline objective on the same trace; below 1 means the policy is better. Results describe these simulated traces only.
```

How to read it:

- **First table:** pooled results per policy. Mean wait/response/slowdown
  weight every job equally. "Mean of per-trace P95 wait" averages each trace's
  95th-percentile wait, a tail measure. Utilization is allocated CPU/GPU-time ÷
  available time, weighted by trace length. Lower wait is better. Utilization
  rarely changes much, since the same work is done either way; a shorter
  schedule raises it slightly.
- **"Policy vs" table:** per-trace comparisons. *Wins* is the share of test
  traces where the policy's mean wait is strictly lower. *Mean/median ratio* is
  policy wait ÷ baseline wait on the same trace (0.97 = 3% less waiting).
  Ratios are more trustworthy than the pooled means, because they compare each
  trace with itself.
- **Learned weight table:** the range of weights the policy chose across test
  traces. A narrow min–max range would mean the policy learned roughly one
  weight vector for everything rather than adapting per trace (see caveats).
- `evaluation.json` has everything, including per-trace weights and waits.

### What the default run achieved

These are the numbers from the logs and table above, and they are modest:

- Validation improved from a ratio of about 1.00 to a best of 0.983 (epoch
  45), then fell back to 0.996 by epoch 49. That's why evaluation uses
  `best.pt`, not `latest.pt`.
- On the 100 test traces, the policy cut mean wait **6.7% versus Slurm's
  all-zero default** (79% of traces won), but only **1.4% versus the simulator
  default**. The median ratio there is 0.999 and it wins 51% of traces, which is
  close to a tie.
- It **raised tail waiting**: the mean per-trace P95 wait is 471 s, against 458 s
  (simulator default) and 404 s (FIFO). Minimizing *mean* wait favours letting
  small jobs past, which some large jobs pay for. If tails matter, train with
  `--objective p95_wait_time_seconds` or add a combined objective.
- The learned weights do differ between traces (e.g. job_size from about 70 to
  10,000), so the features carry some signal.
- Learning is slow: the exploration width hardly moved. Larger `--lr`, more
  `--epochs`, or larger batches are the obvious levers. These are untested; the
  defaults above were chosen up front, not tuned.

## Caveats

- **Small headroom.** On these traces, very different static weights changed
  mean wait by only about ±5%. Expect improvements of a few percent, not 30%.
- **Similar contexts.** All 1000 traces come from one generator setting (same
  load, bursty arrivals, cluster), so their features vary little. The policy
  may learn nearly the same weights for every trace. Per-trace adaptation needs
  more varied training traces (e.g. mixed `--load` and `--arrival-pattern`).
- **Simulator, not Slurm.** Results inherit every approximation in
  `docs/simulator.md`: EASY-only backfill, flat fair share, one partition. Learned
  weights need validating against real Slurm before use.
- **Age is nearly inert here.** With a 7-day `max_age_seconds` and traces lasting
  minutes, the age factor stays tiny, so only enormous age weights register.
  The log scale lets the policy reach them if useful.
- **The `features` include actual runtimes** (requested-vs-actual statistics).
  That's fine for replaying historical traces, but a live deployment would not
  know actual runtimes in advance.
