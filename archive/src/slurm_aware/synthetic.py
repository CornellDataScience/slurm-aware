"""Generate hypothetical job runtimes, not measurements from a Slurm cluster."""

import numpy as np
import pandas as pd

FEATURES = [
    "workload_type",
    "input_size_mb",
    "iterations",
    "nodes",
    "tasks_per_node",
    "cpus_per_task",
]
NUMERIC_FEATURES = FEATURES[1:]
TARGET = "runtime_seconds"
GROUP = "workload_id"
WORKLOAD_TYPES = ["cpu_bound", "io_bound", "mixed"]
GENERATOR_VERSION = 1


def generate_jobs(n_jobs: int = 3000, n_workloads: int = 150, seed: int = 42) -> pd.DataFrame:
    """Create repeated resource configurations of distinct workload templates.

    Workload templates share a type, input size, iteration count, and unobserved
    implementation cost. A group split keeps their variants out of other splits.
    Each node has two CPU slots, matching the repository's two-node playground.
    """
    if n_workloads < 20 or n_jobs < n_workloads:
        raise ValueError("Use at least 20 workloads and at least as many jobs as workloads.")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    template_type = np.arange(n_workloads) % len(WORKLOAD_TYPES)
    rng.shuffle(template_type)
    template_size = np.exp(rng.uniform(np.log(16), np.log(4096), n_workloads))
    template_iterations = rng.integers(1, 21, n_workloads)
    # These hidden effects and the generated duration are NEVER input features.
    implementation_cost = rng.lognormal(0, 0.15, n_workloads)

    template = np.arange(n_jobs) % n_workloads
    rng.shuffle(template)
    kind = template_type[template]
    size = template_size[template]
    iterations = template_iterations[template]
    nodes = rng.choice([1, 2], n_jobs, p=[0.7, 0.3])
    cpus_per_task = rng.choice([1, 2], n_jobs)
    tasks_per_node = np.where(cpus_per_task == 2, 1, rng.choice([1, 2], n_jobs))
    total_cpus = nodes * tasks_per_node * cpus_per_task

    # Toy cost model: serial work + parallel work + I/O + multi-node overhead.
    # Coefficients are chosen for variety, not fitted to any real hardware.
    cpu_cost = np.array([5.0, 0.3, 2.0])[kind] * (size / 64) ** 1.15 * iterations
    parallel_fraction = np.array([0.90, 0.20, 0.65])[kind]
    cpu_seconds = cpu_cost * (1 - parallel_fraction + parallel_fraction / total_cpus)
    io_seconds = np.array([0.1, 3.0, 1.0])[kind] * (size / 64) * np.sqrt(iterations)
    overhead = 2 + (nodes - 1) * (1 + 0.25 * np.log1p(size))
    runtime = (overhead + cpu_seconds + io_seconds) * implementation_cost[template]
    runtime *= rng.lognormal(0, 0.10, n_jobs)  # Unobserved run-to-run variation.

    return pd.DataFrame(
        {
            "job_id": [f"synthetic-{i:06d}" for i in range(n_jobs)],
            GROUP: [f"workload-{i:04d}" for i in template],
            "workload_type": np.asarray(WORKLOAD_TYPES)[kind],
            "input_size_mb": np.round(size, 3),
            "iterations": iterations,
            "nodes": nodes,
            "tasks_per_node": tasks_per_node,
            "cpus_per_task": cpus_per_task,
            TARGET: np.round(runtime, 3),
        }
    )


def validate_jobs(frame: pd.DataFrame, *, training: bool = True) -> None:
    """Reject unusable input early; IDs and labels are never model features."""
    required = FEATURES + ([GROUP, TARGET] if training else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    numeric = NUMERIC_FEATURES + ([TARGET] if training else [])
    try:
        values = frame[numeric].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Resource fields and runtimes must be numeric.") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Resource fields and runtimes must be finite and positive.")
    for column in ["iterations", "nodes", "tasks_per_node", "cpus_per_task"]:
        counts = frame[column].to_numpy(dtype=float)
        if not np.equal(counts, np.floor(counts)).all():
            raise ValueError(f"{column} must contain whole numbers.")
    for column in ["workload_type"] + ([GROUP] if training else []):
        if (
            not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"{column} must contain nonempty strings.")
    if training and frame[GROUP].nunique() < 20:
        raise ValueError("Training requires at least 20 distinct workload_id groups.")
