#!/bin/bash
set -euo pipefail

# Create the authentication key at runtime, rather than storing it in the image.
if [ ! -s /etc/munge/munge.key ]; then
    dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
fi
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

# Windows-mounted files may have CRLF endings; submit Linux-formatted scripts.
find /workspace -maxdepth 1 -type f -name '*.sbatch' -exec sed -i 's/\r$//' {} +

runuser -u munge -- /usr/sbin/munged
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/slurm.conf
