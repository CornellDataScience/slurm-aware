#!/bin/bash
set -euo pipefail

# Every node must shar  e the same MUNGE key, so it is provided as a Compose secret.
install -o munge -g munge -m 400 /run/secrets/munge.key /etc/munge/munge.key

# Windows-mounted files may have CRLF endings; submit Linux-formatted scripts.
find /workspace -maxdepth 1 -type f -name '*.sbatch' -exec sed -i 's/\r$//' {} +

runuser -u munge -- /usr/sbin/munged

case "${SLURM_ROLE:-}" in
    controller) exec slurmctld -D ;;
    compute)    exec slurmd -D ;;
    *)          exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/slurm.conf ;;
esac
