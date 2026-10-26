# Experimental methodology

Research question: does profiling-guided feedback improve LLM-generated parallel C performance while preserving measured functional correctness?

## Comparisons

For each selected workload/input/paradigm, the app evaluates a serial C baseline, a curated parallel reference, a one-shot API-generated candidate and up to three feedback revisions. The one-shot prompt has serial source, the common header, input/worker configuration and contract but no profiler results or reference parallel code. A revision receives the current candidate and its real measurements. If the baseline fails, repair feedback includes the actual diagnostics. API requests, generated source, explanations and decision reasons are persisted.

The reference sources in this package are **assistant-authored curated implementations**. They must not be described as independently human-written. For a human-vs-AI study, replace or independently review these reference files and document the reviewer, authorship and provenance. Human reference authorship cannot honestly be manufactured by this software.

## Correctness and timing

Three deterministic small cases use independent Python oracles and compare complete output vectors at every requested worker count, with relative and absolute tolerance 1e-8. Cases vary input sizes, seeds and iteration counts. Full vectors also have their own checksums recomputed. The benchmark-size output is checked against three serial fingerprints on every warmup and measured run. Checksums are not a proof of full large-array equivalence. Passing test inputs is not a proof of correctness for all inputs or freedom from races. No race detector is claimed unless independently configured and validated.

Compilation failure, nonzero exit, timeout, output overflow, invalid JSON, NaN/Infinity or incorrect outputs prevent performance comparison for that candidate. A timing-stage failure discards all its partial timing rows. Trial stdout/stderr, exit codes and diagnostics remain saved.

Timing uses Python `perf_counter` around process launch/completion **inside an already running sandbox**. It includes executable or MPI launcher startup, input initialization, allocation, computation and checksum output. Compilation, Docker startup and profiling are excluded. Self-reported kernel/communication times are retained as diagnostics and never used to select candidates. Short runs may be dominated by startup; inspect several increasing input sizes. Strong scaling fixes the input size and varies workers.

Each configuration has configurable warmups followed by at least three measured trials; five is the default. Worker-count trial order is deterministically shuffled within a candidate. Candidate order is sequential, so thermal drift/order effects remain a limitation. Use consistent power mode, a plugged-in laptop, quiet background load and repeated independent experiments. The app records CPU model when available, logical CPU count/budget, visible memory, platform, image ID, kernel, compiler version, flags and resource limits. Add physical-core topology and OS power settings to study notes. Worker counts are requested thread/process counts; source review is still needed to verify that an AI implementation actually uses the intended parallel constructs. All comparisons within a run share the same selected resource budget.

Median is the primary statistic. Every timing, sample standard deviation, median absolute deviation, minimum and maximum is stored. Error bars are sample standard deviations, not confidence intervals. Speedup is `serial_median / parallel_median`; efficiency divides this speedup by the selected thread/process count. Serial time always means the serial C implementation at the same input, not the one-thread parallel implementation. A speedup below one or negative optimization finding is retained honestly.

## Selection and validity

Among valid AI candidates, selection minimizes the geometric mean of median times over exactly the same worker counts. A configurable relative improvement threshold (default 2%) is required to replace the incumbent. This avoids automatic acceptance of every small fluctuation but does not establish statistical significance. Rebenchmark selected code on held-out runs/seeds and report intervals before claiming research improvements. If the initial AI program is invalid and a revision becomes valid, that is correctness recovery, not a measured speedup relative to a valid baseline.

The reference is a comparator, not a replacement labeled as AI. Experimental failure and failed API requests remain visible in history. An interrupted process may leave a checkpoint marked running; it is incomplete and should be rerun.

## Profiling scope

Basic evidence comprises wall times, scaling and variability. `perf stat` attempts cycles, instructions and cache events without granting extra privileges. It often cannot access hardware counters in Docker/WSL; this is recorded as unavailable. Do not disable Docker seccomp or run privileged to make counters work. Cachegrind uses explicit cache simulation, clearly labeled as simulated rather than measured hardware events. Profiler runs are separate and bounded; MPI Cachegrind aggregation is not implemented. MPI reference sources report communication elapsed time, including waits in selected MPI calls, as self-reported diagnostics. These are not pure network latency and do not identify a slow thread/rank.

Low efficiency is evidence of limited scaling, not proof of false sharing, load imbalance or synchronization overhead. Unmeasured counters remain missing. The app never populates charts with fictional data. All MPI experiments in this package are single-node; cluster network scaling requires additional physical nodes and a separate experiment design.
