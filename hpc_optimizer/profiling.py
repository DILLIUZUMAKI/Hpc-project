"""Optional evidence collection, always separate from timed benchmark runs."""
import re
import shutil
from pathlib import Path
from .process import run_process


def parse_perf(text):
    counters = {}
    for line in text.splitlines():
        fields = line.split(";")
        if len(fields) < 3:
            continue
        try:
            value = float(fields[0].strip())
        except ValueError:
            continue
        if fields[2].strip():
            counters[fields[2].strip()] = value
    return counters


def parse_cachegrind(text):
    events, values = None, None
    for line in text.splitlines():
        if line.startswith("events:"):
            events = line.split()[1:]
        elif line.startswith("summary:"):
            values = [int(x) for x in line.split()[1:]]
    return dict(zip(events, values)) if events and values and len(events) == len(values) else {}


def collect(kind, command, env, work_dir, timeout, paradigm):
    if kind == "basic":
        return {"status": "available", "kind": "timing", "counters": {},
                "note": "External process timings and variability only; no hardware-counter evidence."}
    if kind == "perf":
        if not shutil.which("perf"):
            return {"status": "unavailable", "kind": kind, "reason": "perf is not installed in the sandbox image."}
        run = run_process(["perf", "stat", "-x", ";", "-e", "cycles,instructions,cache-references,cache-misses", "--", *command],
                          timeout=timeout, env=env, cwd=work_dir)
        counters = parse_perf(run["stderr"])
        status = "available" if counters and run["returncode"] == 0 and not run["failure"] else "unavailable"
        return {"kind": kind, "status": status, "counters": counters, "run": run,
                "reason": None if status == "available" else "Hardware counters are absent or restricted by kernel/container permissions."}
    if paradigm == "mpi":
        return {"kind": kind, "status": "unavailable", "reason": "Cachegrind aggregation for MPI ranks is not implemented; timing instrumentation remains available."}
    if not shutil.which("valgrind"):
        return {"kind": kind, "status": "unavailable", "reason": "Valgrind is not installed in the sandbox image."}
    target = Path(work_dir)/"cachegrind.out"
    run = run_process(["valgrind", "--tool=cachegrind", "--cache-sim=yes", f"--cachegrind-out-file={target}", *command],
                      timeout=timeout, env=env, cwd=work_dir)
    counters = parse_cachegrind(target.read_text(errors="replace")[:2_000_000]) if target.is_file() else {}
    return {"kind": kind, "status": "available" if counters and run["returncode"] == 0 and not run["failure"] else "unavailable",
            "counters": counters, "run": run, "note": "Simulated cache events, not hardware measurements. Profile runtime is excluded from speedup calculations."}


def analyze(candidate):
    observations, suggestions = [], []
    for row in candidate.get("measurements", []):
        p = row["workers"]
        if row.get("efficiency", 1) < 0.5 and p > 1:
            observations.append(f"At {p} workers, measured parallel efficiency is {row['efficiency']:.3f}.")
            suggestions.append("Investigate task size, serial work and synchronization; timing alone does not identify which cause dominates.")
        if row["stdev_seconds"] / row["median_seconds"] > 0.1:
            observations.append(f"At {p} workers, sample standard deviation exceeds 10% of median runtime.")
            suggestions.append("Repeat in a quiet environment and consider a larger workload before interpreting small changes.")
    profile = candidate.get("profile", {})
    counters = profile.get("counters", {}) if profile.get("status") == "available" else {}
    if counters.get("instructions", 0) and counters.get("cycles", 0):
        observations.append(f"Measured instructions/cycle: {counters['instructions']/counters['cycles']:.3f}.")
    if counters.get("cache-references", 0):
        observations.append(f"Measured cache-miss/reference ratio: {counters.get('cache-misses', 0)/counters['cache-references']:.3%}.")
        suggestions.append("Evaluate locality-oriented changes and remeasure; cache misses alone do not prove false sharing.")
    if profile.get("kind") == "cachegrind" and counters:
        observations.append("Cachegrind simulated cache events were collected; these are not measured hardware misses.")
        suggestions.append("Use the simulated event counts to propose locality changes, then test actual runtime separately.")
    return {"observed_bottlenecks": list(dict.fromkeys(observations)),
            "optimization_suggestions": list(dict.fromkeys(suggestions)),
            "unmeasured": ["per-thread load imbalance", "false sharing", "CPU stall breakdown", "proven absence of races"],
            "profile_status": profile.get("status", "not_run")}
