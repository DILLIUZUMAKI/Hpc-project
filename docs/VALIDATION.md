# Delivery validation

Validated on 25 October 2026 in a Linux x86_64 build environment using Python 3.12.14. These are software-validation results, not research performance findings and not measurements of your laptop.

## Automated checks

The complete suite passed **69 tests**, with **12 Docker integration tests skipped** because no Docker CLI/daemon was available in this environment. Tests were run with pytest 9.1.1. The distribution includes all test source; repeat the Docker suite on your laptop after building the image.

| Area | What was actually checked |
|---|---|
| C compilation | All 15 bundled serial/OpenMP/MPI reference sources compiled and linked. GCC was used for serial/OpenMP; MPICH 5.0.2 headers/libraries were used for MPI compile checks. The production image uses OpenMPI, whose runtime was not exercised here. |
| Numerical execution | All five serial and OpenMP workloads matched independent Python oracles. The static execution smoke checks covered 60 complete-vector cases, including 1/2/4 OpenMP threads and wrapped 32-bit seeds. |
| Evaluation worker | Actual serial/OpenMP compilation, small-case validation, benchmark-size fingerprints, warmups, repeated measurements and wrong-output rejection were exercised for all five workloads. |
| Test execution isolation | Static known reference executables ran under a deny-by-default seccomp policy with memory/CPU/output/wall-time limits. Dedicated probes confirmed file opening, network socket creation and process forking were blocked; an infinite loop timed out. This test-only executor is not a substitute for production Docker validation. |
| Core logic | Invalid configuration, consent requirement, malformed/non-finite/inconsistent output, timing statistics, consistent speedup baselines, threshold selection, slower/invalid candidate rejection, output limits, timeouts, storage IDs and checkpoint/export logic. |
| AI integration | Synthetic API fixtures verified request format, JSON parsing, failure handling, API-key redaction and saved failed prompts. Mocked experiments verified baseline/feedback separation and selection across iterations. No live API request was made. |
| Dashboard | Streamlit AppTest exercised all ten pages with empty and populated fixture history, the Run button, new-run selection and missing-Docker messages. Fixtures are explicitly synthetic and are not included in your experiment directory. |
| Web server | The local Streamlit server started successfully and returned HTTP 200 with `ok` from its health endpoint. |

The test-only execution timer includes the seccomp helper's startup, unlike the production Docker worker timer. These smoke-test timings are not bundled as performance data. The delivered `experiments/` directory is empty. No AI-generated versions or claimed improvements are prepopulated.

## Not verified here

- Building/running the production Docker image and its 12 integration tests.
- MPI program execution or multi-rank numerical/scaling behavior; compilation alone does not establish correctness.
- Live LLM generation/optimization with a real provider account, key and model.
- Real perf hardware counters, Cachegrind execution or a compatible race detector.
- Browser pixel/layout inspection; browser download was unavailable. Dashboard behavior was checked with AppTest.
- Windows/WSL setup on the user's specific laptop, multi-node MPI or statistically significant research gains.

## Reproduce

On Ubuntu with the Python environment and Docker image ready:

```bash
bash scripts/test.sh -m "not docker"
bash scripts/test.sh -m docker
.venv/bin/python cli.py --check
.venv/bin/python cli.py --config sample_config.json
```

Optional static seccomp checks need x86_64 Linux, GCC/static runtime libraries and libseccomp. They skip if that toolchain is absent. The application never selects this test helper as a fallback for generated or uploaded code. To reproduce the exact Python 3.12 dependency set, install `requirements-tested.txt` into a separate Python 3.12 virtual environment. Results with other interpreter/library versions should be recorded separately.
