"""Trusted controller inside the restricted container. Expects its job on stdin."""
import ctypes
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

from hpc_optimizer.compiler import compile_source
from hpc_optimizer.metrics import summarize
from hpc_optimizer.process import run_process
from hpc_optimizer.profiling import collect
from hpc_optimizer.validation import compare, parse_output


def run_job(job, *, work_dir="/work", input_dir="/input", executor=None, compiler=None):
    # Protect the controller's oracle values from same-UID /proc memory inspection.
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0
    executor = executor or run_process
    compiler = compiler or compile_source
    work = Path(work_dir)
    source = Path(input_dir)/"program.c"
    binary = work/"program"
    paradigm = job["paradigm"]
    result = {"status": "compilation_failed", "compilation": compiler(source, binary, paradigm, input_dir),
              "correctness": [], "measurements": [], "profile": {"status": "not_run"},
              "race_detection": {"status": "unavailable", "reason": "No validated OpenMP/MPI race detector configured. Passing numeric tests does not prove race freedom."},
              "environment": {"kernel": platform.release(), "architecture": platform.machine(),
                              "available_cpus": len(os.sched_getaffinity(0)),
                              "timing_scope": "Python perf_counter around the program/launcher inside an already running container; includes startup, allocation and output; excludes compilation and container startup"}}
    if result["compilation"]["returncode"] != 0 or result["compilation"].get("failure"):
        return result

    def command(p, case, full=False):
        args = [str(binary), str(case["size"]), str(case["steps"]), str(case["seed"]), "--check" if full else "--bench"]
        if paradigm == "mpi":
            args = ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", str(p), *args]
        return args

    def environment(p):
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp", "LC_ALL": "C",
                "OMP_NUM_THREADS": str(p), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE",
                "OMP_THREAD_LIMIT": str(max(job["counts"]))}

    def execute(p, case, full=False):
        run = executor(command(p, case, full), env=environment(p), cwd=str(work), timeout=job["timeout"])
        output, error = None, None
        if run["returncode"] == 0 and not run["failure"]:
            try:
                output = parse_output(run["stdout"])
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
        else:
            error = run["failure"] or f"Exit status {run['returncode']}"
        return run, output, error

    result["status"] = "incorrect"
    for p in job["counts"]:
        for case in job["checks"]:
            run, output, error = execute(p, case, True)
            ok, detail = compare(output, case["expected"]) if output is not None else (False, error)
            result["correctness"].append({"workers": p, "size": case["size"], "seed": case["seed"], "passed": ok,
                                           "detail": detail, "run": run})
            if not ok:
                return result

    expected = job.get("expected_benchmark")
    benchmark = job["benchmark"]
    # Correctness at the timed problem size, in addition to complete small-case tests.
    for p in job["counts"]:
        run, output, error = execute(p, benchmark)
        if expected is None and output is not None:
            expected = {k: output[k] for k in ("length", "sum", "sum_squares", "weighted_sum")}
        ok, detail = compare(output, expected) if output is not None else (False, error)
        result["correctness"].append({"workers": p, "size": benchmark["size"], "seed": benchmark["seed"],
                                       "passed": ok, "detail": detail, "scope": "large-input fingerprints", "run": run})
        if not ok:
            return result
    result["benchmark_output"] = expected
    samples = {p: [] for p in job["counts"]}
    result["warmup_runs"] = []
    result["timed_runs"] = []
    for p in job["counts"]:
        for _ in range(job["warmups"]):
            run, output, error = execute(p, benchmark)
            ok = output is not None and compare(output, expected)[0]
            result["warmup_runs"].append({"workers": p, "run": run, "passed": ok})
            if not ok:
                result["status"] = "runtime_failed"
                return result
    order = job["counts"] * job["repeats"]
    random.Random(benchmark["seed"]).shuffle(order)
    for p in order:
        run, output, error = execute(p, benchmark)
        ok = output is not None and compare(output, expected)[0]
        diagnostics = {k:output[k] for k in ("kernel_seconds", "communication_seconds") if output is not None and k in output}
        result["timed_runs"].append({"workers": p, "run": run, "passed": ok, "program_reported": diagnostics})
        if not ok:
            result["status"] = "runtime_failed"
            result["error"] = error or "Output changed during timing."
            return result
        samples[p].append(run["seconds"])
    result["measurements"] = [{"workers": p, **summarize(samples[p])} for p in job["counts"]]
    for row in result["measurements"]:
        for name in ("kernel_seconds", "communication_seconds"):
            values = [trial["program_reported"][name] for trial in result["timed_runs"]
                      if trial["workers"] == row["workers"] and name in trial["program_reported"]]
            row["reported_median_"+name] = statistics.median(values) if len(values) == job["repeats"] else None
    result["status"] = "valid"
    p = max(job["counts"])
    result["profile"] = collect(job["profiler"], command(p, benchmark), environment(p), str(work), job["timeout"]*2, paradigm)
    result["profile"]["workers"] = p
    result["profile"]["input_size"] = benchmark["size"]
    return result


if __name__ == "__main__":
    try:
        job = json.loads(sys.stdin.buffer.read(2_000_001))
        print(json.dumps(run_job(job), allow_nan=False))
    except Exception as exc:
        print(json.dumps({"status": "worker_error", "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
