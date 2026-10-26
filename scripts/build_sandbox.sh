#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v docker >/dev/null || { echo "Install Docker Desktop and enable Ubuntu WSL integration." >&2; exit 1; }
docker info >/dev/null
docker build -f sandbox/Dockerfile -t hpc-optimizer-sandbox:1.0 .
echo "Sandbox image built. Run bash scripts/run.sh"
