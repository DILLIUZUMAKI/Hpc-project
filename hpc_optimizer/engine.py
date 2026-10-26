import hashlib
import platform
import time

from .config import ROOT, hardware_info
from .metrics import add_speedups, choose
from .prompts import generation_prompt, optimization_prompt
from .profiling import analyze
from .validation import correctness_cases
from .sandbox import validate_source


def load_source(benchmark, variant):
    from .config import BENCHMARKS
    if benchmark not in BENCHMARKS or variant not in {"serial", "reference_openmp", "reference_mpi"}:
        raise ValueError("Unknown built-in source.")
    return (ROOT/"benchmarks"/benchmark/(variant+".c")).read_text()


class Engine:
    def __init__(self, store, sandbox, llm=None, progress=None):
        self.store, self.sandbox, self.llm = store, sandbox, llm
        self.progress = progress or (lambda text: None)

    def run(self, config, serial_source=None):
        config.validate()
        info = self.sandbox.preflight()
        config.validate(info["cpu_budget"])
        if config.use_ai and self.llm is None:
            raise ValueError("AI mode requires a configured API client.")
        serial_source = validate_source(serial_source or load_source(config.benchmark, "serial"))
        experiment = self.store.create(config)
        experiment["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "hardware": hardware_info(), **info}

        def log(message):
            experiment["events"].append({"at": time.time(), "message": message})
            self.store.save(experiment)
            self.progress(message)

        def evaluate(ident, origin, source, expected=None, prompt=None, explanation=""):
            candidate = {"id": ident, "origin": origin, "source": source,
                         "sha256": hashlib.sha256(source.encode()).hexdigest(), "prompt": prompt,
                         "explanation": explanation, "status": "running", "measurements": []}
            experiment["candidates"].append(candidate)
            log(f"Compiling and validating {ident}.")
            counts = [1] if ident == "serial" else config.counts
            job = {"paradigm": "serial" if ident == "serial" else config.paradigm,
                   "checks": correctness_cases(config.benchmark, config.seed), "expected_benchmark": expected,
                   "benchmark": {"size": config.size, "steps": config.steps, "seed": config.seed},
                   "counts": counts, "warmups": config.warmups, "repeats": config.repeats,
                   "timeout": config.timeout, "profiler": "basic" if ident == "serial" else config.profiler}
            candidate.update(self.sandbox.evaluate(source, job, config))
            if candidate["status"] == "valid":
                baseline = candidate["measurements"][0]["median_seconds"] if ident == "serial" else experiment["serial_seconds"]
                add_speedups(candidate["measurements"], baseline)
                candidate["analysis"] = analyze(candidate)
            log(f"{ident}: {candidate['status']}.")
            return candidate

        def generate(prompt, label):
            attempt = {"label": label, "prompt": prompt, "status": "pending"}
            experiment.setdefault("generation_attempts", []).append(attempt)
            self.store.save(experiment)
            try:
                response = self.llm.generate(prompt)
                attempt["status"] = "completed"
                attempt["metadata"] = {k:v for k,v in response.items() if k not in {"code", "explanation"}}
                return response
            except Exception:
                attempt["status"] = "failed"
                raise

        try:
            baseline = evaluate("serial", "curated_serial", serial_source)
            if baseline["status"] != "valid":
                experiment["status"] = "failed"
                log("Serial baseline failed; experiment stopped before comparisons.")
                return experiment
            experiment["serial_seconds"] = baseline["measurements"][0]["median_seconds"]
            expected = baseline["benchmark_output"]
            evaluate("reference", "curated_reference", load_source(config.benchmark, "reference_"+config.paradigm), expected)
            if config.use_ai:
                prompt = generation_prompt(serial_source, config)
                log("Sending serial C and its contract to the configured LLM endpoint for one-shot generation.")
                response = generate(prompt, "ai_baseline")
                current = evaluate("ai_baseline", "llm_one_shot", response["code"], expected, prompt, response["explanation"])
                current["generation"] = {k:v for k,v in response.items() if k not in {"code", "explanation"}}
                best = current if current["status"] == "valid" else {"status": "invalid"}
                if best.get("id"):
                    experiment["best_ai"] = best["id"]
                for i in range(1, config.iterations+1):
                    prompt = optimization_prompt(current, config)
                    log(f"Sending measured evidence for optimization iteration {i}.")
                    response = generate(prompt, f"ai_iteration_{i}")
                    candidate = evaluate(f"ai_iteration_{i}", "llm_feedback", response["code"], expected, prompt, response["explanation"])
                    candidate["generation"] = {k:v for k,v in response.items() if k not in {"code", "explanation"}}
                    accepted, reason = choose(best, candidate, config.counts, config.min_improvement)
                    candidate["accepted"], candidate["decision"] = accepted, reason
                    if accepted:
                        best = candidate
                        experiment["best_ai"] = candidate["id"]
                    current = best if best.get("id") else candidate
                    log(reason)
            experiment["status"] = "completed"
            log("Experiment completed; measurements and source versions saved.")
            return experiment
        except Exception as exc:
            experiment["status"] = "failed"
            # API client deliberately raises sanitized exceptions, never raw provider bodies.
            experiment["error"] = f"{type(exc).__name__}: {exc}"
            log(experiment["error"])
            return experiment
        finally:
            self.store.save(experiment)
