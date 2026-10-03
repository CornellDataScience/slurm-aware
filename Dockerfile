FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       slurm-wlm munge supervisor python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash student \
    && mkdir -p /var/spool/slurmctld /var/spool/slurmd /run/munge /var/log/munge /workspace \
    && chown slurm:slurm /var/spool/slurmctld \
    && chown munge:munge /run/munge /var/log/munge

COPY docker/slurm.conf /etc/slurm/slurm.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/slurm.conf
COPY docker/entrypoint.sh /usr/local/bin/slurm-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/slurm-entrypoint \
    && chmod +x /usr/local/bin/slurm-entrypoint \
    && rm -f /etc/munge/munge.key

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/slurm-entrypoint"]
