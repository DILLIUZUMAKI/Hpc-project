FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libgomp1 openmpi-bin libopenmpi-dev python3 valgrind linux-tools-generic \
    && rm -rf /var/lib/apt/lists/*
COPY hpc_optimizer /opt/hpc_optimizer
COPY sandbox/worker.py /opt/worker.py
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
USER 65534:65534
CMD ["python3", "/opt/worker.py"]
