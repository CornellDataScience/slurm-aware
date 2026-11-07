#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=hello-%j.out

set -euo pipefail
echo "Job $SLURM_JOB_ID running on $(hostname)"
srun python3 -c 'import os; print("Hello from Slurm! Job:", os.environ["SLURM_JOB_ID"])'
sleep 15
echo "Finished"
