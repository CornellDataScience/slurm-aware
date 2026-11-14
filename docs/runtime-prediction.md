# Runtime prediction

The scheduler can use submission-time runtime estimates while the simulator
executes jobs for their actual durations. The target is execution time from start
to finish, excluding queue waiting. All time values are simulator timesteps.

## Data and prediction

`Job.duration` remains the actual runtime. `requested_duration` is a positive
integer submission estimate, not an enforced execution limit. Old job JSON files
without this field remain valid in oracle mode; estimate modes require it.

The synthetic generator creates estimates with
`max(1, round(duration * LogNormal(log(estimate_multiplier), estimate_sigma)))`.
The defaults are multiplier 2 and sigma 0.75. These are simulation assumptions,
with both overestimation and underestimation, and can be changed in generator
configuration JSON. Estimate noise is drawn after all physical workload fields,
preserving the original arrival/resource/actual-runtime sequences for each seed.

The predictor uses resource requests in cluster order, the submitted estimate,
and job type (assumed to be declared at submission). IDs, arrival times, queue
waits, actual runtimes, and completion times are excluded from its features.
Resource requests and the submitted estimate use `log1p`; job types are one-hot
encoded. Unseen types use zero one-hot entries; the type-median baseline falls
back to the global training median.

Candidates are the submitted estimate, global median, per-type median, ridge
regression, and a random forest. Regression targets use `log1p(duration)`.
Predictions are transformed back, rounded to the nearest timestep, and floored
at one. Candidate selection minimizes validation MAE; there is no test-label
selection or refit on validation data.

Predictor history uses independent `runtime_train`, `runtime_val`, and
`runtime_test` seed streams (3, 4, 5). RL uses `train`, `val`, and `test` streams
(0, 1, 2). Entire jobsets stay within a split. The predictor is fixed before RL
training. Model artifacts record the generator, base seed, split indices, feature
schema, library version, and validation/test scores.

Within a type, the current generator samples runtime independently of resource
size. The per-type median is therefore an essential baseline. These experiments
measure performance on synthetic workloads and the stated submission-noise model.

## Commands

```sh
uv sync --locked --extra deeprm --extra runtime

# Fit on historical jobs, select on validation, and report on held-out history.
uv run --extra deeprm --extra runtime slurm-runtime-train --output artifacts/runtime --seed 0

# Train a policy using the frozen predictor.
uv run --extra deeprm slurm-deeprm-train --runtime-mode predicted --runtime-model artifacts/runtime/model.json --output artifacts/deeprm/predicted

# Same model and estimate mode are restored from the policy checkpoint.
uv run --extra deeprm slurm-deeprm-evaluate --checkpoint artifacts/deeprm/predicted/best.pt --output artifacts/deeprm/predicted/eval

# Full paired comparison: three seeds, three modes, three loads, two test regimes.
uv run --extra deeprm --extra runtime python scripts/runtime_prediction_experiment.py

uv run --extra deeprm --extra runtime pytest
```

Use `--runtime-mode requested` without a model for submitted estimates.
`--runtime-mode oracle` is the default, including for older policy checkpoints.
`--generator-config` on the predictor and policy trainers accepts the same
generator JSON format. A smaller integration experiment is:

```sh
uv run --extra deeprm --extra runtime python scripts/runtime_prediction_experiment.py --output artifacts/runtime-comparison-smoke --iterations 20 --train-jobsets 12 --test-jobsets 5
```

Predictor outputs include `model.json`, `report.json`, `metrics.csv`,
`test_predictions.csv`, and `prediction_error.png`. Metrics include MAE, the
fraction within a factor of two, underestimation frequency, and mean shortfall
among underestimated jobs.

Policy checkpoints embed the complete predictor as JSON-compatible coefficients
or tree arrays and record its SHA256. Worker processes, resume, and evaluation
restore it directly; the original `model.json` is not needed afterward.
Inference needs NumPy; the `runtime` extra supplies scikit-learn for fitting.

The comparison script saves the experiment configuration, each seed's predictor,
each mode's policy checkpoints and logs, detailed comparison CSV/JSON, a Markdown
summary, and scheduling plots. It trains a separate policy per mode with equal
budgets, using steady arrivals and loads 0.5, 1.0, and 1.5. Every mode is evaluated
on the same test traces at those loads under steady and bursty arrivals. Plots
show mean and standard deviation over seeds; report truncation counts alongside
slowdown and waiting-time results, which are lower bounds for unfinished jobs.

## Environment behavior

The environment computes estimates at arrival and caches them for the episode.
The frozen predictor also caches repeated feature combinations across training
episodes. Waiting-job images use estimated duration. Running-job images use
`max(1, estimate - elapsed_execution_time)`, clipped to the image horizon. The
one-step floor keeps an overrun visibly occupied at the current time.

Actual completion events release resources, including early finishes. Estimated
completion times do not create events or release resources. Immediate scheduling
eligibility uses current free capacity. Rewards and outcome metrics retain actual
runtimes; rewards are training feedback and are not policy observation features.
True future occupancy from `backend.canvas()` is used only in oracle mode.
Estimate modes render running jobs in start order with stable job shades.
The policy visualizer uses those same estimates for the observation panel and
labels the runtime mode; its retrospective job timeline retains actual runtimes.

The simulator and backend still contain actual job records for execution and
evaluation. Custom policies must use `estimated_duration(job)`, `fits_now(job)`,
and `observation()` for scheduling instead of reading hidden simulator state.

```python
import json
from pathlib import Path

from slurm_aware.deeprm.env import EnvConfig
from slurm_aware.deeprm.gym_env import DeepRMGymEnv
from slurm_aware.deeprm.runtime import RuntimePredictor

predictor = RuntimePredictor(json.loads(Path("artifacts/runtime/model.json").read_text()))
env = DeepRMGymEnv(config=EnvConfig(runtime_mode="predicted"), predictor=predictor)
observation, info = env.reset(seed=42)
```

Tests cover hidden-duration leakage at identical observable states, resource
ownership during overruns, early completion, horizon clipping, model export
equivalence with scikit-learn, held-out-label isolation, Gymnasium checks,
worker reproducibility, checkpoint resume, and oracle compatibility.

## Initial integration experiment (2026-11-14)

The smaller command above completed with seeds 0, 1, and 2, 20 training
iterations per mode, and five test jobsets per load/regime. Ridge regression was
selected on validation in all three seeds. Held-out predictor MAE was:

| Seed | Submitted estimate | Type median | Selected predictor |
| ---: | ---: | ---: | ---: |
| 0 | 8.465 | 2.761 | 1.962 |
| 1 | 9.388 | 2.708 | 1.926 |
| 2 | 9.960 | 2.901 | 2.156 |

The three runtime modes compare as follows. Prediction MAE is averaged over the
three held-out predictor test splits. Scheduling slowdown is averaged equally
over all 18 seed/load/regime combinations; lower is better.

| Mode | Runtime supplied to scheduler | Runtime MAE | SJF slowdown | Tetris* slowdown | DeepRM truncated |
| --- | --- | ---: | ---: | ---: | ---: |
| Oracle | Actual execution time | 0 | 3.035 | 3.219 | 90/90 |
| Requested | Simulated submitted estimate | 9.271 | 3.581 | 3.692 | 90/90 |
| Predicted | Frozen ridge prediction | 2.014 | 3.578 | 3.590 | 90/90 |

Learned predictions reduced runtime MAE by 78.3% relative to submitted estimates.
SJF's average slowdown was nearly unchanged; Tetris*'s was 2.8% lower. Oracle
timing produced the lowest average slowdown for both heuristics in this run.
The prediction-error reduction therefore translated into a much smaller change
in scheduling outcomes on these synthetic traces.

All 270 greedy DeepRM evaluation episodes truncated across the three modes;
the heuristic evaluations completed. The short run verifies integration and
prediction accuracy but does not establish an RL scheduling improvement.
The reported DeepRM slowdowns were 49.779 (oracle), 45.419 (requested), and
52.851 (predicted), with unfinished jobs counted only through the 1,000-timestep
limit. These are lower bounds and do not establish a ranking of the modes for
completed RL schedules.
Detailed local outputs are in `artifacts/runtime-comparison-smoke/`, which is
Git-ignored along with other experiment artifacts.
