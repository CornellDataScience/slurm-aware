# CDS compute cluster profile

Use `examples/cds_cluster.json` to evaluate weights against the documented CDS
hardware. The profile lists **compute1–compute4**, with **orca** as the head/login
node. Orca contributes no compute capacity. Three workers have enough published
information to enable scheduling; compute4 remains listed but disabled until
its inventory is known.

## Hardware and sources

The local **FA25 Compute Cluster Presentation.pdf** establishes the topology
(slides 8–10 and 29), mixed ARM/x86 architectures (slide 12), and two RTX 2080 Ti
GPUs on compute1 used for LLM serving (slides 15 and 17). Slide 9 depicts GPUs
on compute4 but does not establish its hardware specifications.

Public sources were checked on 2026-10-17 and pinned to
[CornellDataScience/computecluster commit 5d281958](https://github.com/CornellDataScience/computecluster/tree/5d281958afb2c3fb964c24cc5903711c05bdf9a3)
(2026-05-28):

| Worker | Architecture | Logical CPU slots | Modeled RAM (MiB) | GPUs | Enabled |
| --- | --- | ---: | ---: | --- | --- |
| compute1 | x86_64 | 32 | 31,921 | 2 × RTX 2080 Ti | Yes |
| compute2 | x86_64 | 20 | 15,678 | Unknown; zero modeled | Yes |
| compute3 | aarch64 | 4 | 7,937 | None | Yes |
| compute4 | Unknown | Unknown | Unknown | Unspecified in inventory | No |

CPU counts, architectures, and compute1/compute3 GPU information come from
[Architecture.md](https://github.com/CornellDataScience/computecluster/blob/5d281958afb2c3fb964c24cc5903711c05bdf9a3/Architecture.md).
These counts represent logical CPUs, not a claim of equal performance per CPU
slot. The inventory leaves compute2's GPU section as TODO and does not provide
a compute4 section.

Memory comes from committed health dashboard snapshots:
[compute1.json](https://github.com/CornellDataScience/computecluster/blob/5d281958afb2c3fb964c24cc5903711c05bdf9a3/services/health-dashboard/compute1.json),
[compute2.json](https://github.com/CornellDataScience/computecluster/blob/5d281958afb2c3fb964c24cc5903711c05bdf9a3/services/health-dashboard/compute2.json),
and [compute3.json](https://github.com/CornellDataScience/computecluster/blob/5d281958afb2c3fb964c24cc5903711c05bdf9a3/services/health-dashboard/compute3.json).
Their timestamps are from 2025-12-09. The respective `total_kb` values are
32,687,844, 16,055,236, and 8,128,008. These Linux MemTotal values are in KiB;
integer division by 1024 supplies the profile's MiB capacities. The existing
`memory_mb` field names are retained, and all workload/config memory values
must use the same units.

These are historical OS memory totals, not current free memory or verified
Slurm `RealMemory` allocations. GPU VRAM is not modeled. There is no published
compute4 dashboard snapshot in this revision. The public dashboard endpoint
returned HTTP 403 when checked, so this profile does not claim live inventory.

The enabled subset provides **56 logical CPU slots, 55,536 MiB RAM, and two
known GPUs**. Setting an unknown GPU count to `null` preserves the uncertainty;
it contributes zero modeled GPU capacity. A disabled node contributes no
capacity or priority-normalization totals. Results therefore describe this
documented subset, not the complete four-worker cluster.

## Cluster policy versus experiment settings

The published [slurm.conf](https://github.com/CornellDataScience/computecluster/blob/5d281958afb2c3fb964c24cc5903711c05bdf9a3/slurm/slurm.conf)
registers compute1–compute4, names the shared partition `compute`, uses
`sched/backfill` and `select/cons_tres`, and configures SlurmDBD accounting on
orca. It does not supply per-node CPU, memory, or GPU capacities. Its priority
type and weight examples are commented out, so it does not establish deployed
multifactor weights or the real cluster's current policy.

The simulator uses its existing EASY backfill approximation with this topology.
`alice`/`bob`, their shares, `normal`/`urgent` QoS, the age cap, scheduling
interval, usage half-life, and `examples/priority_weights.json` are experiment
settings, not recovered production settings. The goal is still to compare
fixed priority weights using mean submission-to-completion time.

The Docker playground in this repository is separate: it still has two
compute containers, `priority/basic`, and no accounting. Loading this profile
does not start containers, submit jobs, or change the CDS cluster.

## Placement and persistent services

Jobs can specify `architecture`, `gpu_type`, and `allowed_nodes` in addition
to CPU, memory, and GPU requests. These constrain every allocated node. Use
`x86_64`, `aarch64`, and `rtx2080ti` for this profile; matching is exact. They
are simulator fields, not a claim that the deployed Slurm configuration
advertises equivalent Features or GRES names. An empty architecture constraint
means the workload is compatible with either architecture.

The workload generator selects requests that fit compatible enabled nodes.
For example, a two-GPU request can use compute1, while a four-CPU ARM request
uses compute3. It cannot invent a second GPU worker for a multi-node GPU job.
In CSV, `allowed_nodes` is a semicolon-separated list, such as
`compute1;compute2`; it defines permitted nodes rather than requiring all of them.

The slides describe compute1 as the NFS server with shared storage mounted
across machines. Shared files do not by themselves require every job to run on
compute1. Use `allowed_nodes` only when a workload actually needs a particular
machine. Disk, network, and shared-filesystem contention are not simulated.

For a controlled experiment with a continuously running service, add
`reserved_cpus`, `reserved_memory_mb`, and `reserved_gpus` to its node entry.
They subtract constant capacity for the entire replay. For example, setting
`reserved_gpus` to `2` on compute1 leaves no modeled GPUs for submitted jobs;
GPU workload requests will then fail feasibility validation. CPU and RAM
reservations should reflect that service's actual allocation, not guesses.
Defaults are zero because the slides do not establish current allocations.
The two LLM-serving slides are not evidence of two simultaneous services.

Background reservations are not scored as jobs and do not charge account
fair-share usage. Utilization is calculated against the capacity left for the
replayed workload. Priority resource fractions still use enabled hardware
totals before background reservations, as for occupied cluster resources.

To enable compute4, fill in its actual `cpus`, `memory_mb`, architecture, and
known GPU information, then set `enabled` to `true`. Do not infer a particular
Alienware/Dell configuration solely from the chassis pictured in the slides.
Update the profile notes when replacing historical values with verified ones.

## Run an experiment

```sh
uv run slurm-sim evaluate --workload examples/cds_jobs.csv --config examples/cds_cluster.json --weights examples/priority_weights.json --output artifacts/cds-simulator-example
uv run slurm-sim generate --config examples/cds_cluster.json --traces 5 --jobs 200 --seed 42 --output data/generated/cds-scheduling.csv
uv run slurm-sim evaluate --workload data/generated/cds-scheduling.csv --config examples/cds_cluster.json --weights examples/priority_weights.json --output artifacts/cds-simulator
```

The eight-job example mixes CPU, ARM, and GPU jobs; its mean response time is
**21.875 simulated seconds** with the supplied settings. The finite `llm-batch`
job is illustrative, not a measurement or representation of an indefinitely
running service. Reports include the inventory and uncertainty notes, and
schedule CSVs use real node names.

For weight comparisons, keep the profile and workload fixed. These traces are
synthetic: their arrival mix, resource requests, and durations are not fitted
to CDS submissions. Before trusting an optimum, replay representative
historical jobs and evaluate on held-out traces. In particular, each job's
runtime is currently constant across compatible nodes; the simulator does not
estimate speed differences between a Threadripper, an Intel CPU, and a Pi.
Use measured durations with appropriate placement constraints for controlled
validation. Modeling runtime changes across eligible machines is future work.

See [the simulator guide](simulator.md) for the objective, priority formulas,
full workload schema, and remaining differences from Slurm.
