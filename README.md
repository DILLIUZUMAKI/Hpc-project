# ParallelLab — Profiling-Guided Iterative Optimization

A local Python/Streamlit research application for **Profiling-Guided Iterative Optimization of LLM-Generated OpenMP and MPI Programs**. It compiles and validates C, measures repeated program runtimes, gathers available profiling evidence, requests LLM revisions and retains valid measured improvements. It includes five deterministic workloads and a restricted Docker execution worker.

**Start with [QUICKSTART.md](QUICKSTART.md).** Run from Ubuntu/WSL2 or Linux; Docker is mandatory for C execution. The dashboard opens at http://localhost:8501.

```bash
bash scripts/setup_ubuntu.sh
bash scripts/build_sandbox.sh
bash scripts/run.sh
```

## Included

- Matrix multiplication, Jacobi 2D stencil, Monte Carlo Pi, 256-bin histogram and sparse-graph BFS.
- Serial, curated OpenMP and curated single-node MPI C implementations (15 source files).
- Independent small-case Python oracles, large-input serial fingerprints, compilation/runtime/timeout checks, bounded output and per-count validation.
- Repeated measurements, warmups, all raw trials, median/variability, consistent serial-baseline speedup and parallel efficiency.
- Optional perf and Cachegrind with explicit unavailable states; evidence-based feedback without invented counters.
- OpenAI-compatible API client, one-shot baseline, up to three feedback revisions, correctness-first selection and preserved histories.
- Ten dashboard views, plots, source/prompt inspection and CSV/JSON/Markdown exports.
- Setup/build/run scripts, command-line mode, unit/UI/conditional Docker integration tests and methodology/security guides.

## Repository guide

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `cli.py` | Headless experiment runner and environment check |
| `hpc_optimizer/` | Config, compiler, sandbox, oracles, metrics, profiling, API, prompts, engine, storage and charts |
| `benchmarks/` | Five workloads, each with serial/OpenMP/MPI source, plus common.h |
| `sandbox/` | Restricted Docker image and trusted evaluation worker |
| `scripts/` | Ubuntu setup, image build, app launch and test scripts |
| `tests/` | Unit, UI and optional real-Docker tests |
| `docs/` | Program contract, methodology, security, troubleshooting and delivery validation |
| `experiments/` | Local experiment histories, initially empty |

## Honest scope

AI source files and optimized versions are produced by real API calls during your experiments; no placeholder model outputs or fabricated timings are bundled. Reference-only mode works without an API key. API credentials must be supplied locally. Generated candidates are not automatically trusted merely because they compile.

The bundled references are assistant-authored, **not independently human-written**. Obtain a genuine human-authored/reviewed baseline for a study using that label. All MPI examples are single-node. General C programs need the documented input/output adapter. No compatible race detector is preconfigured, and numerical testing is not a proof of correctness. Advanced profiler availability is environment-dependent. See [delivery validation](docs/VALIDATION.md) for what was actually executed, what was mocked and what remains to test on your laptop.

## Tests

```bash
bash scripts/test.sh -m "not docker"
bash scripts/test.sh -m docker
```

Docker tests skip when the daemon/image is unavailable and are never reported as passing. Build checks compile known bundled references. Additional Linux x86_64 smoke checks run statically linked, known reference programs under a deny-by-default seccomp policy and resource limits; this test-only executor is never used by the application or exposed to uploads/LLM outputs. Runtime application code requires Docker.

## Technical references

- GCC OpenMP: https://gcc.gnu.org/onlinedocs/libgomp/
- Open MPI launch guide: https://docs.open-mpi.org/en/main/launching-apps/quickstart.html
- Docker runtime controls: https://docs.docker.com/engine/containers/run/
- Valgrind Cachegrind: https://valgrind.org/docs/manual/cg-manual.html
- Python subprocess: https://docs.python.org/3/library/subprocess.html
- Streamlit testing: https://docs.streamlit.io/develop/api-reference/app-testing
- OpenAI-compatible request reference: https://developers.openai.com/api/reference/resources/chat

Direct package versions are pinned in the requirements files. `requirements-tested.txt` records the complete Python 3.12 dependency set used during validation. Compiler and container image IDs are recorded per experiment. Pin your final verified Docker image digest as well for a published study.
