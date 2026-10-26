import math
import statistics


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timings must be positive finite measurements.")
    median = statistics.median(samples)
    return {"samples_seconds": samples, "median_seconds": median,
            "min_seconds": min(samples), "max_seconds": max(samples),
            "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "mad_seconds": statistics.median(abs(x-median) for x in samples)}


def add_speedups(rows, serial_seconds):
    if serial_seconds <= 0:
        raise ValueError("Serial time must be positive.")
    for row in rows:
        row["speedup"] = serial_seconds / row["median_seconds"]
        row["efficiency"] = row["speedup"] / row["workers"]
        row["baseline"] = "serial C process wall time, same input"
    return rows


def score(candidate, counts):
    if candidate.get("status") != "valid":
        return math.inf
    rows = {r["workers"]: r for r in candidate.get("measurements", [])}
    if set(rows) != set(counts):
        return math.inf
    return math.exp(statistics.mean(math.log(rows[p]["median_seconds"]) for p in counts))


def choose(incumbent, candidate, counts, threshold=0.02):
    new = score(candidate, counts)
    old = score(incumbent, counts)
    if not math.isfinite(new):
        return False, "Rejected: correctness or complete timing data is missing."
    if not math.isfinite(old):
        return True, "Accepted: first valid measured AI candidate."
    improvement = 1 - new/old
    if improvement > threshold:
        return True, f"Accepted: geometric-mean runtime improved by {improvement:.2%} (threshold {threshold:.1%})."
    return False, f"Retained previous candidate: measured change {improvement:.2%}, threshold {threshold:.1%}."
