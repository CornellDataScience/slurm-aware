# Evaluating priority weights

The current goal is a reliable objective evaluator. This implementation evaluates
a supplied weight vector; it does not search for weights, train a runtime model,
or modify the running Docker cluster.

For job `j`, response time is `finish_time[j] - submit_time[j]`, including both
waiting and execution. The objective is the arithmetic mean across all jobs.
Every submitted job completes before scoring, including those still queued
after the final arrival. For multiple traces, every job has equal weight; traces
with more jobs contribute more observations.

## Python interface

```python
from pathlib import Path

from slurm_aware.sim_cli import load_config, load_weights
from slurm_aware.simulator import evaluate, simulate, aggregate_metrics
from slurm_aware.workloads import read_traces

config = load_config(Path("examples/scheduler.json"))
weights = load_weights(Path("examples/priority_weights.json"))
traces = read_traces("examples/scheduling_jobs.csv")

# Scalar objective for one trace; lower is better.
loss = evaluate(weights, traces["example"], config)
print(loss)  # 18.666666666666668 simulated seconds

# Detailed schedules plus a pooled objective for any number of traces.
results = {name: simulate(jobs, weights, config) for name, jobs in traces.items()}
metrics = aggregate_metrics(results)
print(metrics["mean_response_time_seconds"])
```

`Node`, `Cluster`, `Job`, `Account`, `PriorityWeights`, and `SimulationConfig` are also
available directly from `slurm_aware.simulator` for constructing experiments in
Python. The core simulator uses only the Python standard library.

## Configuration and priority factors

Without a configuration, the generic default is **4 homogeneous nodes, 2 CPUs
and 1024 MiB per node, and no GPUs**. `examples/scheduler.json` uses that toy
topology. For the actual CDS topology, use `examples/cds_cluster.json` and
`examples/cds_jobs.csv`; see [the sourced cluster profile](cds-cluster.md).
It lists four workers, with three enabled because compute4's capacity is unknown.

Heterogeneous profiles use `cluster.node_specs`, a list of named nodes with
`cpus`, `memory_mb`, `gpus`, optional `architecture`/`gpu_type`, and optional
`reserved_cpus`/`reserved_memory_mb`/`reserved_gpus` for background services.
`cluster.head_node` is metadata and cannot also be a worker. Do not combine
`node_specs` with the homogeneous `nodes`/`*_per_node` cluster settings.
An enabled node needs known CPU and memory capacities; unknown GPU counts may
be `null` and contribute no modeled GPUs. A node with unknown CPU or RAM must
be disabled with `enabled: false`. Disabled nodes contribute no capacity.

Requests remain uniform across the job's exact number of nodes. Free resources
are tracked separately on each compatible node: four free CPU slots spread
across four nodes cannot satisfy a request for two CPU slots on one node.
Memory fields retain the `mb` spelling but use MiB consistently. GPU memory
and differences in CPU/GPU processing speed are not modeled.

The priority is the sum of the weighted factors, plus `site_factor`, minus
`nice`. It is truncated to an integer and clipped to `[1, 4294967295]`. The
following JSON keys are accepted in a weights file:

| Key | Factor in this simulator |
| --- | --- |
| `age` | Eligible waiting time divided by `max_age_seconds`, capped at 1 |
| `job_size` | Mean of requested node fraction and requested CPU fraction; `1 - factor` when `favor_small` is true |
| `fairshare` | `2 ** (-normalized_usage / normalized_share)` for flat accounts |
| `association` | Account priority divided by the largest configured account priority |
| `partition` | Constant 1: there is only one shared partition |
| `qos` | QoS priority divided by the largest configured QoS priority |
| `tres_cpu` | Requested total CPUs divided by cluster total CPUs |
| `tres_memory` | Requested total memory divided by cluster total memory |
| `tres_gpu` | Requested total GPUs divided by cluster total GPUs |

Association/QoS factors are zero when all their configured priorities are zero.
A resource factor is zero when the cluster has none of that resource; requests
for unavailable resources are rejected. Non-TRES weights are nonnegative
integers; TRES weights can be signed finite numbers within the supported range.
Unknown weight/configuration fields are rejected instead of silently ignored.
Node fractions and resource totals include enabled nodes only. Static background
reservations reduce placement capacity, but resource-factor normalization uses
the enabled hardware totals before subtracting those reservations.

There is no fixed-total or two-weight restriction. The example weights activate
age, fair share, job size, and QoS. Identical association, partition, or QoS
factors cannot distinguish jobs. CPU TRES and job size can be strongly correlated.
Weight magnitude matters for integer rounding and clipping; extremely small
weights can produce ties, and extremely large sums can saturate.

Account usage is charged for allocated CPU slots while a job runs, including
elapsed portions of unfinished jobs. It decays continuously with the configured
`fairshare_half_life_seconds`; use `null` for no decay. `initial_usage_cpu_seconds`
allows a repeatable starting history. Shares are normalized across all configured
accounts, including accounts with no jobs in the current trace. Each replay copies
the initial usage and begins with no workload jobs running; configured static
background reservations remain in place. Reservations do not accrue usage.

Library defaults use a seven-day age cap and usage half-life, a 30-second
scheduling interval, and favor large resource requests. The generic example config uses
a 120-second age cap, 600-second half-life, five-second interval, and favors small
requests to make the factors vary in short experiments. The CDS profile retains
the library timing defaults as experiment settings; these have not been verified
on the real cluster. Keep these settings fixed when comparing weight vectors.

The formulas follow the documented
[multifactor components](https://slurm.schedmd.com/priority_multifactor.html), the
[Slurm 21.08 job-size implementation](https://github.com/SchedMD/slurm/blob/slurm-21-08-5-1/src/plugins/priority/multifactor/priority_multifactor.c),
and the [simplified classic fair-share formula](https://slurm.schedmd.com/classic_fair_share.html).
The flat account model does **not** implement account hierarchies or Slurm's
default Fair Tree algorithm. It must not be presented as equivalent to that policy.

## Scheduling behavior

The clock jumps between eligibility events, actual completions, and periodic
scheduling ticks. Nothing sleeps in wall-clock time. At a shared timestamp,
completions release resources and all newly eligible jobs enter the queue before
dispatch. Factors are recalculated on each scheduling pass. Higher integer
priority wins; ties use earlier submission, then lexicographic job ID. Use
zero-padded IDs if numeric ordering matters. Node placement uses increasing node
index in the inventory and reserves the same nodes for the entire execution.
Architecture, GPU model, node affinity, and per-node capacity are checked during
both immediate placement and future backfill reservations.

- `strict`: dispatch in priority order until the first job that cannot fit.
- `easy_backfill` (default): reserve an expected start and nodes for the first
  blocked job, based on running jobs' **requested time limits**. Lower-priority
  jobs may start if their requested limit ends by that reservation, or if they
  fit entirely outside the reserved nodes. Reserved nodes are protected as
  whole nodes. The reservation is recomputed on each scheduling pass.

Actual runtimes determine completion events only. The priority function and
backfill estimates never use them to anticipate completions. A job finishing
early releases resources immediately, allowing the reservation to move forward.

This EASY approximation protects the first blocked job only. Slurm's full
backfill scheduler considers additional higher-priority jobs and reservations;
it also has daemon timings and scheduling limits absent here. See
[Slurm scheduling behavior](https://slurm.schedmd.com/sched_config.html).
Arrival/dispatch batching and continuous fair-share decay also differ from Slurm.

## Workload CSV

All time values are seconds relative to the beginning of a trace. Counts must
be integers. Jobs in different `trace_id` groups are independent experiments.

| Field | Required/default | Meaning |
| --- | --- | --- |
| `trace_id` | Optional, `trace-0` | Independent replay identifier |
| `job_id` | Required | Unique string within its trace |
| `submit_time` | Required | Nonnegative submission timestamp |
| `runtime` | Required | Positive actual execution duration, visible to the event engine |
| `time_limit` | Required | Positive requested duration, visible to backfill |
| `nodes` | 1 | Exact number of requested nodes |
| `cpus_per_node` | 1 | CPU slots required on each requested node |
| `memory_mb_per_node` | 0 | Reserved memory on each requested node |
| `gpus_per_node` | 0 | Reserved GPUs on each requested node |
| `architecture` | Unrestricted | Exact required node architecture, e.g. `x86_64` or `aarch64` |
| `gpu_type` | Unrestricted | Exact required GPU model; requires a positive GPU request |
| `allowed_nodes` | Unrestricted | Semicolon-separated permitted worker names |
| `account` | `default` | Key in the configured account map |
| `qos` | `normal` | Key in the configured QoS priority map |
| `eligible_time` | Submission time | Earliest execution/age-accrual timestamp |
| `nice`, `site_factor` | 0 | Signed integer priority adjustments |

Zero memory/GPU requests reserve none of those resources. Eligibility must not
precede submission. Runtime must not exceed the requested limit: failures,
timeouts, cancellations, and preemption are currently unsupported. Impossible
requests, duplicate IDs, missing data, non-finite numbers, unknown fields, and
unconfigured accounts/QoS are rejected. Dependencies are not inferred from
`eligible_time`; dependency-driven eligibility needs a future DAG implementation.
Only one partition is supported. An unrestricted architecture means a job is
compatible with any worker; actual binaries/images need appropriate constraints.

The older runtime-prediction CSV uses a different schema. It must be converted
and given submission times/time limits before it can be used here.

## Synthetic traces and reports

`slurm-sim generate` creates independent traces with lognormal execution
durations, varying resource requests, accounts/QoS, and padded time limits.
Resource size and duration are sampled independently. These are invented
workloads, not measurements from a real cluster or predictions from a model.
The generator chooses an enabled node, samples requests within its unreserved
capacity, and chooses a node count that fits the compatible inventory. Jobs
carry the sampled architecture and, when applicable, GPU model. This ensures
feasible requests but does not reproduce the real cluster's submission mix.

`--load` controls the expected ratio of offered CPU work to capacity. Achieved
utilization also depends on placement and queue ordering. `--arrival-pattern`
accepts `steady` (exponential interarrival times) and `bursty` (alternating
20-job blocks with shorter/longer intervals). Separate seeds generate separate
traces. The generator writes a `.metadata.json` sidecar with seed, configuration,
and checksum. Changing only the seed does not test a different workload family.

`report.json` records the effective configuration/weights, workload SHA-256,
simulator version, profile notes, limitations, and metrics. `schedule.csv`
contains every job, including allocated node names and the factors used at its
actual dispatch time. The simulator and workload generator are now version 2;
the new generator produces different traces from version 1 for a given seed.
The mean response is the optimization objective; mean/P95/max wait and P95
response help identify poor outcomes hidden by an average. CPU utilization is
allocated CPU-seconds divided by available CPU-seconds from first submission
to final completion. GPU utilization is computed analogously, or is `null` if
no GPUs remain available. Both exclude disabled capacity and static background
reservations. Multiple-trace utilization weights those spans by duration and
assumes the same cluster configuration for all traces.

Keep the complete workload, config, and initial usage identical when comparing
weights. Hold out whole traces for validation/testing, and use later time periods
for real historical traces. Do not reuse observed historical waiting times as
the outcome of a different policy; replay recomputes those waits.

## Verification and next boundary

### Comparing weight sets

```sh
uv run slurm-sim compare --workload examples/cds_jobs.csv --config examples/cds_cluster.json --weights current=examples/priority_weights.json --weights age_only=examples/age_only_weights.json --output artifacts/comparison
```

Repeat `--weights NAME=PATH` for at least two candidates. Names must be unique
(case-insensitive) and contain only letters, digits, hyphens, or underscores,
starting with a letter or digit. Omitted weight fields use the same defaults as
`evaluate`; the supplied age-only example explicitly zeros all other weights.
The first candidate is the baseline. The shared configuration, scheduler mode,
workloads, and initial usage remain identical for every candidate. Each replay
starts fresh. Age-only still uses the configured backfill policy, eligibility,
and job nice/site adjustments; it is not a strict FIFO baseline.

Results are ranked by pooled, job-weighted mean response time. Equal scores
retain candidate input order. Positive improvement means a percentage reduction
in response time versus the first candidate. This ranking describes the supplied
traces only and does not establish statistical significance or generalization.

`comparison.md` is a readable summary, `comparison.csv` contains aggregate metrics,
and `comparison.json` includes effective weights, configuration, workload checksum,
limitations, and per-trace metrics. Each named subdirectory contains `report.json`
and `schedule.csv`, as with `evaluate`. Existing outputs at these paths are replaced;
inputs cannot be overwritten. Unrelated files or older candidate directories are
left alone; the current comparison files identify the candidates in this run.

Both evaluation and comparison now report mean/P95 slowdown, calculated per job
as `(finish - submit) / runtime`. This is ordinary, unbounded slowdown, so very
short jobs can have large values. Throughput is completed jobs divided by the
sum of trace spans, each from first submission to final completion, including
idle gaps. It is not an average of per-trace throughput rates. CPU/GPU utilization
measures allocated resource-time, not measured CPU/GPU activity. No memory
utilization metric is currently reported. Per-job schedules also include slowdown.

### Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover hand-computed objectives, four-node capacity, node fragmentation,
memory/GPU constraints, age/size/QoS/fair-share decisions, safe backfill and
requested-versus-actual duration, deterministic resets, simultaneous events,
pooled metrics, malformed inputs, generated trace invariants, and CLI artifacts.
Cluster-profile tests also cover heterogeneous capacity, ARM/x86 compatibility,
GPU models, node affinity, background reservations, unknown inventory, and
configuration round-tripping.

This establishes the behavior of this simulator. It has **not yet been validated
against Slurm for multifactor scheduling**. The existing Docker setup uses
`priority/basic` and no accounting; the public CDS cluster configuration includes
SlurmDBD. Controlled replay comparisons need the actual multifactor configuration,
account policy, resource allocations, and representative workloads. On this
heterogeneous cluster, a job's runtime is currently fixed across compatible
nodes, and network/shared-storage contention is omitted. Completing the missing
inventory and checking these approximations precede deploying optimized weights.
The weight optimizer itself is still outside this evaluator implementation.
