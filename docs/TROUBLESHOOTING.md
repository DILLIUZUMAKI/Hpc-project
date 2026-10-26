# Troubleshooting

| Symptom | Action |
|---|---|
| Closed the WSL installer | Reopen administrator PowerShell, run `wsl --list --verbose`. If Ubuntu is missing, rerun `wsl --install -d Ubuntu`; restart when prompted. |
| Virtualization / WSL VM error | Check Task Manager > Performance > CPU > Virtualization. Follow Microsoft's WSL installation troubleshooting guide for the exact error; firmware settings vary. |
| `docker: command not found` in Ubuntu | Start Docker Desktop; enable Settings > Resources > WSL Integration for this Ubuntu distribution. Reopen Ubuntu. |
| Cannot connect to Docker daemon | Keep Docker Desktop running. Use `docker info` in Ubuntu; resolve its error before running experiments. |
| Sandbox image missing | From the project folder run `bash scripts/build_sandbox.sh`. |
| First Docker build fails to download packages | Check network/proxy access and disk space, then rerun the build. A failed build is never replaced by a fake runner. |
| Python packages missing | Run `bash scripts/setup_ubuntu.sh`; launch via `bash scripts/run.sh` to use the project virtual environment. |
| Port 8501 is occupied | Stop the previous app or run `.venv/bin/python -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1`. |
| Worker count exceeds CPU budget | Choose fewer workers. Check both WSL and Docker CPU allocations. Logical CPUs do not guarantee linear speedup. |
| Memory setting rejected / exit 137 | Lower input size, reduce MPI workers or increase Docker memory within your laptop's resources. Per-rank arrays increase memory use. |
| Runtime timeout | Start with a smaller size or increase per-run timeout within the UI's supported bounds. BFS and profiling runs can be costly. |
| API 401 / 403 | Check the provider-specific API key and account permissions. A chat subscription is not assumed to supply API credentials. |
| API 400 / 404 | Check the base URL and model ID; use a model supporting the Chat Completions text endpoint and JSON instructions. The endpoint must end at `/v1` or the provider's equivalent base, not `/chat/completions`. |
| Model response is not valid JSON | Choose a suitable coding model. The app does not execute partial or malformed responses. No automatic API retries are hidden. |
| `perf` unavailable / permission denied | Use basic profiling or Cachegrind. Hardware counters depend on kernel, hardware and sandbox permissions. Do not remove sandbox restrictions. |
| MPI Cachegrind unavailable | MPI rank aggregation is not included. Use basic timings and the stored MPI diagnostic output. |
| Parallel code slower than serial | This is a legitimate measurement. Examine workload size, startup overhead and variability; retain the result. |
| Incorrect candidate | Inspect compilation/stdout/stderr and the failed case. It is excluded from timing comparisons; feedback can request a repair. |
| Empty charts | No valid measurements exist yet. Run a reference-only experiment first and inspect diagnostics. |
| Run stays marked running after closing the app | The saved checkpoint is incomplete. Start a fresh run; automatic resumption is not implemented. |

Official installation references: https://learn.microsoft.com/en-us/windows/wsl/install and https://docs.docker.com/desktop/features/wsl/.
