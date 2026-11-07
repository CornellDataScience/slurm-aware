# Archived and deferred work

This folder contains work that is outside the current Meeting 4 implementation
path. The active simulation and learning pipeline lives in
[`../src/slurm_aware/deeprm/`](../src/slurm_aware/deeprm/).

The former directory layout is preserved here so related source, tests, examples,
documentation, and results stay together. The moved files include the simulator
fixes and runtime-prediction experiments present in the working tree. They were moved
from the working tree, not replaced with older Git versions.

| Location | Contents | Status |
| --- | --- | --- |
| `src/slurm_aware/simulator.py`, `sim_cli.py`, `workloads.py` | Priority-weight simulator, CLI, and workload generator | Retired from the active pipeline; reference for event processing and node constraints |
| `src/slurm_aware/rl/` | Model that chooses one weight vector per workload | Retired approach |
| `src/slurm_aware/modeling.py`, `synthetic.py`, `cli.py` | Synthetic runtime-prediction experiment | Deferred; separate from the active scheduler |
| `tests/`, `examples/`, `scripts/` | Historical tests, fixtures, and simulator benchmark | Retained with the corresponding source |
| `docs/` | Simulator, weight-learning, and CDS cluster guides | Historical documentation |
| `data/`, `artifacts/` | Old workload traces, measurements, reports, and runtime model | Historical experiment inputs and outputs |
| `Dockerfile`, `compose.yaml`, `docker/`, `jobs/` | Slurm Docker playground and example job | Deferred comparison environment |
| `output/pdf/` | Execution-system work plan in PDF and LaTeX | Superseded implementation order; reference for later execution work |
| `FA25 Compute Cluster Presentation.pdf`, `Slurm-Aware First Meeting - Google Slides.pdf` | Earlier project material | Historical references |
| [README.legacy.md](README.legacy.md) | Previous root README | Historical commands and explanations |

## Using the archive

The root project does not import these modules or install their commands. Its
default test discovery and linting also exclude this folder.

The archived Python project has its own `pyproject.toml` and `uv.lock`. Run its
historical commands from this directory, using its own virtual environment.
Do not install both projects into the same environment: both use the
`slurm_aware` package name. The runtime-prediction experiment also
uses pandas, scikit-learn, joblib, and matplotlib, which are not declared by the
historical priority-weight project.

The Slurm Compose paths are relative to this directory. The Rust prototype
remains outside the archive in [`../allocator/`](../allocator/README.md).

Archive-local ignore rules preserve the status of generated data, experiment
outputs, job logs, and the local MUNGE key. Ignored local files are retained on
disk and are not added to version control by this reorganization.
