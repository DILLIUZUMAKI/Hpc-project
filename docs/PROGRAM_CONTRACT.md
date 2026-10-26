# C program contract

Each source is a single C11 translation unit. It may include the supplied `common.h`. It must accept:

```text
program N STEPS SEED --check
program N STEPS SEED --bench
```

`N` is the matrix/grid dimension, sample/item count or graph vertex count. `STEPS` controls Jacobi iterations and is ignored by other kernels. `SEED` is an unsigned 32-bit integer. `common.h` contains deterministic input helpers, validated argument parsing, allocation checks and JSON output; preserve their meaning. Do not require input files or stdin. Only MPI rank 0 prints output.

The only stdout content must be one JSON object with `length`, `sum`, `sum_squares` and `weighted_sum`. In `--check` mode also include `values`, the complete numeric output vector. Weighted sum is `sum(values[i] * (i % 97 + 1))`. Every value must be finite. In `--bench` mode omit the full vector to avoid timing large text serialization. Optional `kernel_seconds` and `communication_seconds` are self-reported diagnostics, not the selection timer. Use `emit()` for a matching implementation.

## Exact workload definitions

- **Matrix multiplication:** row-major square double matrices. `A[i]=initial(2*i,seed)` and `B[i]=initial(2*i+1,seed)`. Output row-major `A*B`.
- **Jacobi:** zero boundaries; interior `initial(row*N+column,seed)`. Each step replaces an interior cell with one quarter of its four direct neighbors from the previous step. Output the complete row-major grid after STEPS iterations.
- **Monte Carlo Pi:** sample i uses `x=mix32(seed+2*i)/4294967296.0`, `y=mix32(seed+2*i+1)/4294967296.0`. A hit satisfies `x*x+y*y<=1`. Output `[hit_count, 4.0*hit_count/N]`. Index-based RNG makes results independent of scheduling and worker count.
- **Histogram:** input i maps to bin `mix32(seed+i)%256`. Output 256 bin counts converted to doubles.
- **BFS:** directed sparse graph with four outgoing edges per vertex, specified by `neighbor(u,k,N,seed)`, k in 0..3. Start at vertex 0; output every shortest-path distance, using -1 for unreachable vertices. The first edge forms a cycle, so all vertices in these built-in graphs are reachable. Duplicate edges are allowed.

All RNG arithmetic wraps modulo 2^32. `initial(i,seed)` is `(mix32(seed+i)%100)/100.0`. Exact `mix32` operations are supplied in `common.h` and independently implemented in Python.

OpenMP programs must honor OMP_NUM_THREADS. MPI programs must initialize/finalize MPI and support ranks with no local work, including when the process count exceeds a small validation input's dimension.

Source uploads replace the serial implementation of a selected built-in workload. They must match these semantics and pass the independent oracle. Arbitrary C input/output formats are not automatically inferred; adapt them and add an oracle/catalog entry before treating them as a new workload. No generic semantic-equivalence checker is claimed.
