"""Independent Python oracles, strict output parsing and numeric comparisons."""
import json
import math
from collections import deque


def mix32(x):
    x &= 0xffffffff
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xffffffff
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xffffffff
    return x ^ (x >> 16)


def initial(i, seed):
    return (mix32(seed + i) % 100) / 100.0


def neighbors(u, n, seed):
    return [(u + 1) % n] + [mix32(seed + 3*u + k) % n for k in (1, 2, 3)]


def oracle(benchmark, n, steps, seed):
    if benchmark == "matmul":
        a = [initial(2*i, seed) for i in range(n*n)]
        b = [initial(2*i+1, seed) for i in range(n*n)]
        return [sum(a[i*n+k] * b[k*n+j] for k in range(n)) for i in range(n) for j in range(n)]
    if benchmark == "jacobi":
        a = [initial(i*n+j, seed) if 0 < i < n-1 and 0 < j < n-1 else 0.0 for i in range(n) for j in range(n)]
        for _ in range(steps):
            b = [0.0] * (n*n)
            for i in range(1, n-1):
                for j in range(1, n-1):
                    q = i*n+j
                    b[q] = 0.25*(a[q-n]+a[q+n]+a[q-1]+a[q+1])
            a = b
        return a
    if benchmark == "monte_carlo":
        hits = 0
        for i in range(n):
            x = mix32(seed+2*i)/4294967296.0
            y = mix32(seed+2*i+1)/4294967296.0
            hits += x*x + y*y <= 1.0
        return [float(hits), 4.0*hits/n]
    if benchmark == "histogram":
        counts = [0.0] * 256
        for i in range(n):
            counts[mix32(seed+i) % 256] += 1
        return counts
    if benchmark == "bfs":
        dist = [-1.0] * n
        dist[0] = 0.0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in neighbors(u, n, seed):
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist
    raise ValueError("Unknown benchmark")


def fingerprint(values, include_values=False):
    result = {"length": len(values), "sum": sum(values), "sum_squares": sum(x*x for x in values),
              "weighted_sum": sum(x * (i % 97 + 1) for i, x in enumerate(values))}
    if include_values:
        result["values"] = values
    return result


def correctness_cases(benchmark, run_seed):
    sizes = {"matmul": [1, 7, 13], "jacobi": [3, 7, 11], "monte_carlo": [1, 37, 101],
             "histogram": [1, 257, 1021], "bfs": [1, 17, 31]}[benchmark]
    cases = []
    for i, n in enumerate(sizes):
        seed = (run_seed + 7919*(i+1)) & 0xffffffff
        steps = i+2
        cases.append({"size": n, "steps": steps, "seed": seed,
                      "expected": fingerprint(oracle(benchmark, n, steps, seed), True)})
    return cases


def parse_output(text):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    obj = json.loads(text.strip(), parse_constant=reject)
    if not isinstance(obj, dict) or type(obj.get("length")) is not int or obj["length"] < 1:
        raise ValueError("Expected one JSON object with positive integer length.")
    for key in ("sum", "sum_squares", "weighted_sum"):
        if type(obj.get(key)) not in (int, float) or not math.isfinite(obj[key]):
            raise ValueError(f"Missing or non-finite {key}.")
    if "values" in obj:
        values = obj["values"]
        if not isinstance(values, list) or len(values) != obj["length"]:
            raise ValueError("Output vector length mismatch.")
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
            raise ValueError("Invalid output vector.")
        check = fingerprint(values)
        for key in ("sum", "sum_squares", "weighted_sum"):
            if not math.isclose(obj[key], check[key], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("Reported checksum does not match full output.")
    for key in ("kernel_seconds", "communication_seconds"):
        if key in obj and (type(obj[key]) not in (int, float) or not math.isfinite(obj[key]) or obj[key] < 0):
            raise ValueError(f"Invalid optional {key}.")
    return obj


def compare(actual, expected):
    if actual.get("length") != expected.get("length"):
        return False, "Result length differs."
    for key in ("sum", "sum_squares", "weighted_sum"):
        if not math.isclose(actual[key], expected[key], rel_tol=1e-8, abs_tol=1e-8):
            return False, f"{key} differs."
    if "values" in expected:
        if "values" not in actual or len(actual["values"]) != len(expected["values"]):
            return False, "Full correctness vector is missing."
        for i, (a, b) in enumerate(zip(actual["values"], expected["values"])):
            if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8):
                return False, f"Output element {i} differs."
    return True, "Passed numerical comparison."
