# Slurm-Aware

A small, single-node Slurm playground running in Docker Desktop. Use it to
practice submitting jobs or develop software that calls Slurm.

Docker packages a Linux environment into an **image** (built from `Dockerfile`).
A **container** is a running instance of that image. Docker Compose uses
`compose.yaml` to build and start it with the right settings.

Inside this container, `slurmctld` schedules jobs, `slurmd` runs them, and MUNGE
provides authentication. Your usual Slurm client commands are installed too.
The controller and compute node share one container named `slurm`.

## Start it

Open Docker Desktop and make sure it is using Linux containers. Run these
commands in PowerShell from this repository:

```powershell
docker compose up --build -d --wait
docker compose exec --user student slurm sinfo
```

The first build downloads Ubuntu and installs Slurm, so allow a few minutes.
`-d` keeps the container running in the background; `--wait` waits until Slurm
is ready. You should see a `debug` partition with one `idle` node.

## Submit your first job

```powershell
docker compose exec --user student slurm sbatch hello.sbatch
docker compose exec --user student slurm squeue
```

The example prints a message through `srun`, then sleeps for 15 seconds so you
can see it in the queue. Once it finishes, `squeue` will be empty. Read its output
on Windows:

```powershell
Get-Content jobs/hello-*.out
```

The `jobs` directory is shared with `/workspace` in the container. Put your own
scripts and input files there; job output appears there too. Paths used inside
jobs must be Linux paths, such as `/workspace/input.txt`.

For an interactive Linux shell with Slurm commands available:

```powershell
docker compose exec --user student slurm bash
```

Inside that shell, try `sinfo`, `squeue`, or `srun hostname`. Type `exit` to leave.
Cancel a job using `scancel JOB_ID` inside the shell, or prefix it with
`docker compose exec --user student slurm` from PowerShell.

## Stop, restart, and troubleshoot

```powershell
docker compose stop
docker compose start --wait
docker compose logs --tail 100 slurm
```

`stop` preserves container state. To remove the container and start fresh:

```powershell
docker compose down
docker compose up -d --wait
```

Files in `jobs` survive removal, but Slurm's internal state and job IDs reset.
After changing the Dockerfile or files under `docker`, rebuild using
`docker compose up --build -d --wait`.

If Docker cannot connect to its engine, check that Docker Desktop is running
and using Linux containers. If a new script produces an error about DOS line
breaks, save it with LF line endings in your editor. The startup script also
converts `.sbatch` files already in `jobs` to LF.

## Scope

This uses Ubuntu 22.04's packaged Slurm, a virtual two-CPU node, and a single
partition. CPU counts are scheduling slots, not dedicated physical cores.
It does not configure GPUs, accounting (`sacct`), multiple nodes, or cgroup
enforcement of job CPU/memory limits. Run trusted practice jobs here; this is
a local learning environment rather than a production cluster. No host ports
or privileged-container access are required.

Slurm documentation: [user guide](https://slurm.schedmd.com/quickstart.html)
and [administrator guide](https://slurm.schedmd.com/quickstart_admin.html).
