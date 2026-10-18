# Slurm-Aware

A small, two-node Slurm playground running in Docker Desktop. Use it to
practice submitting jobs or develop software that calls Slurm.

Docker packages a Linux environment into an **image** (built from `Dockerfile`).
A **container** is a running instance of that image. Docker Compose uses
`compose.yaml` to build and start it with the right settings.

The cluster runs as three containers. `controller` runs `slurmctld`, which
schedules jobs. `slurm1` and `slurm2` are compute nodes that each run `slurmd`,
which runs jobs. MUNGE provides authentication using a key shared by all three
containers (`docker/munge.key`). Your usual Slurm client commands are installed
in every container. The examples below run them on `controller`.

## Start it

Open Docker Desktop and make sure it is using Linux containers. Run these
commands in PowerShell from this repository. The first command creates the
shared MUNGE key and is only needed once. The key is ignored by git, so each
person makes their own:

```powershell
docker run --rm -v "${PWD}/docker:/out" ubuntu:22.04 dd if=/dev/urandom of=/out/munge.key bs=1024 count=1
docker compose up --build -d --wait
docker compose exec --user student controller sinfo
```

The first build downloads Ubuntu and installs Slurm, so allow a few minutes.
`-d` keeps the containers running in the background; `--wait` waits until Slurm
is ready. You should see a `debug` partition with two `idle` nodes, `slurm1` and `slurm2`.

## Submit your first job

```powershell
docker compose exec --user student controller sbatch hello.sbatch
docker compose exec --user student controller squeue
```

The example prints a message through `srun`, then sleeps for 15 seconds so you
can see it in the queue. To use both nodes, add `#SBATCH --nodes=2` to a
script, or try `srun -N2 hostname`. Once it finishes, `squeue` will be empty. Read its output
on Windows:

```powershell
Get-Content jobs/hello-*.out
```

The `jobs` directory is shared with `/workspace` in every container. Put your own
scripts and input files there; job output appears there too. Paths used inside
jobs must be Linux paths, such as `/workspace/input.txt`.

For an interactive Linux shell with Slurm commands available:

```powershell
docker compose exec --user student controller bash
```

Inside that shell, try `sinfo`, `squeue`, or `srun hostname`. Type `exit` to leave.
Cancel a job using `scancel JOB_ID` inside the shell, or prefix it with
`docker compose exec --user student slurm` from PowerShell.

## Stop, restart, and troubleshoot

```powershell
docker compose stop
docker compose start --wait
docker compose logs --tail 100 controller
```

`stop` preserves container state. To remove the containers and start fresh:

```powershell
docker compose down
docker compose up -d --wait
```

Files in `jobs` and the MUNGE key survive removal, but Slurm's internal state and job IDs reset.
After changing the Dockerfile or files under `docker`, rebuild using
`docker compose up --build -d --wait`.

If Docker cannot connect to its engine, check that Docker Desktop is running
and using Linux containers. If a new script produces an error about DOS line
breaks, save it with LF line endings in your editor. The startup script also
converts `.sbatch` files already in `jobs` to LF.

## Priority-weight simulator (current focus)

The simulator evaluates **fixed priority weights** by replaying workload traces
and calculating mean submission-to-completion time. The **CDS cluster profile**
uses named workers with their documented CPU, RAM, GPU, and architecture
constraints. It lists compute1–compute4 and excludes the orca head node from
compute capacity. **Three workers are enabled; compute4's hardware inventory
was not published in the sources inspected.** The enabled subset has 56 logical
CPU slots and two known RTX 2080 Ti GPUs.

See [the cluster profile and sources](docs/cds-cluster.md) for the hardware,
unknowns, and optional resource reservations for persistent services. The
simulator runs locally without Docker or a runtime prediction model. The Docker
playground above still has two compute containers; its configuration is separate.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run slurm-sim evaluate --workload examples/cds_jobs.csv --config examples/cds_cluster.json --weights examples/priority_weights.json --output artifacts/cds-simulator-example
```

The eight-job example has a mean response time of **21.875 simulated seconds**.
Results go to `artifacts/cds-simulator-example/report.json` and `schedule.csv`. The report
records the complete configuration, weight vector, input checksum, aggregate
metrics, and per-trace metrics. The schedule lists each job's allocation,
start/end times, wait/response times, and priority factors at dispatch.

Generate a larger repeatable experiment with arrivals and queue contention:

```sh
uv run slurm-sim generate --config examples/cds_cluster.json --traces 5 --jobs 200 --seed 42 --output data/generated/cds-scheduling.csv
uv run slurm-sim evaluate --workload data/generated/cds-scheduling.csv --config examples/cds_cluster.json --weights examples/priority_weights.json --output artifacts/cds-simulator
```

Change any supported weight in `examples/priority_weights.json`, then replay the
same workload into another output directory to compare scores. These example
weights are illustrative, not optimized. Every replay resets the cluster and
account usage. Generated files are ignored by Git; commands replace existing
outputs at their selected destinations.

Compare multiple weight sets in one run (the first is the baseline):

```sh
uv run slurm-sim compare --workload examples/cds_jobs.csv --config examples/cds_cluster.json --weights current=examples/priority_weights.json --weights age_only=examples/age_only_weights.json --output artifacts/comparison
```

The command prints a table ranked by average response time and writes
`comparison.md`, `comparison.csv`, and `comparison.json`. It includes queue wait,
slowdown, CPU/GPU allocation utilization, throughput, and response improvement
relative to the baseline. Each candidate also gets its own report and job schedule.
Use the generated workload above for larger comparisons. This tool evaluates
supplied weights; it does not train an ML model or optimize weights.

Omitting `--config` still selects a generic four-node cluster with two CPU slots
per node. `examples/scheduler.json` and `examples/scheduling_jobs.csv` retain the
earlier toy experiment, whose mean response time is 18.667 simulated seconds.

This evaluator implements strict priority and an **EASY backfill
approximation**, plus a **flat classic fair-share approximation**. It does not
reproduce Slurm's full backfill or default Fair Tree algorithm. Simulator results
need comparison with real Slurm before being used to configure a cluster.

See [the simulator guide](docs/simulator.md) for the Python objective function,
workload schema, supported factors, exact assumptions, and tests.

## Scope

This uses Ubuntu 22.04's packaged Slurm, two virtual two-CPU compute nodes,
and a single partition. CPU counts are scheduling slots, not dedicated physical cores.
It does not configure GPUs, accounting (`sacct`), or cgroup
enforcement of job CPU/memory limits. Run trusted practice jobs here; this is
a local learning environment rather than a production cluster. No host ports
or privileged-container access are required.

Slurm documentation: [user guide](https://slurm.schedmd.com/quickstart.html)
and [administrator guide](https://slurm.schedmd.com/quickstart_admin.html).
