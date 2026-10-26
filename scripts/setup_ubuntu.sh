#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script inside Ubuntu/WSL2 or Linux." >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
if [[ ! -f .env ]]; then cp .env.example .env; chmod 600 .env; fi
echo "Python environment ready. Start Docker Desktop, then run bash scripts/build_sandbox.sh"
