"""Docker is required. There is deliberately no direct-host execution fallback."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .config import ROOT, cpu_limit
from .process import run_process


class SandboxUnavailable(RuntimeError):
    pass


def validate_source(source):
    if not isinstance(source, str) or not source.strip() or "\x00" in source:
        raise ValueError("Provide UTF-8 C source without NUL bytes.")
    if len(source.encode("utf-8")) > 256_000:
        raise ValueError("Source exceeds the 256 KB limit.")
    return source


def container_args(image, source_dir, name, cpus, memory_mb):
    return ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "128",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", "--cpus", str(cpus),
            "--ulimit", "cpu=120:120", "--ulimit", "fsize=16777216:16777216", "--ulimit", "nofile=256:256",
            "--ulimit", "core=0:0", "--user", "65534:65534", "--shm-size", "64m",
            "--tmpfs", "/work:rw,exec,nosuid,nodev,size=512m,mode=1777",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
            "--mount", f"type=bind,src={source_dir},dst=/input,readonly",
            image, "python3", "/opt/worker.py"]


class DockerSandbox:
    def __init__(self, image="hpc-optimizer-sandbox:1.0"):
        self.image = image
        self.info = None

    def preflight(self):
        if os.name != "posix":
            raise SandboxUnavailable("Run this project inside Ubuntu/WSL2 or Linux; native Windows execution is unsupported.")
        if not shutil.which("docker"):
            raise SandboxUnavailable("Docker CLI is missing. Install Docker Desktop and enable Ubuntu WSL integration.")
        result = run_process(["docker", "info", "--format", "{{json .}}"], timeout=15)
        if result["returncode"] != 0 or result["failure"]:
            raise SandboxUnavailable("Docker is not running or permission is denied. Open Docker Desktop and check WSL integration.")
        try:
            info = json.loads(result["stdout"])
        except ValueError as exc:
            raise SandboxUnavailable("Could not read Docker resource information.") from exc
        inspect = run_process(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"], timeout=10)
        if inspect["returncode"] != 0:
            raise SandboxUnavailable("Sandbox image missing. Run: bash scripts/build_sandbox.sh")
        self.info = {"docker_version": info.get("ServerVersion"), "docker_cpus": info.get("NCPU", 1),
                     "docker_memory_bytes": info.get("MemTotal", 0), "image_id": inspect["stdout"].strip(),
                     "cpu_budget": min(cpu_limit(), int(info.get("NCPU", 1))), "backend": "docker",
                     "network": "none", "scope": "single machine"}
        return self.info

    def evaluate(self, source, job, config):
        validate_source(source)
        info = self.info or self.preflight()
        config.validate(info["cpu_budget"])
        if info["docker_memory_bytes"] and config.memory_mb*1024*1024 > info["docker_memory_bytes"]*0.8:
            raise ValueError("Sandbox memory exceeds 80% of Docker's available memory. Lower the memory limit or increase Docker memory.")
        name = "hpc-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="hpc-input-") as tmp:
            directory = Path(tmp)
            directory.chmod(0o755)
            (directory/"program.c").write_text(source, encoding="utf-8")
            (directory/"common.h").write_text((ROOT/"benchmarks/common.h").read_text(), encoding="utf-8")
            for file in directory.iterdir():
                file.chmod(0o644)
            args = container_args(self.image, directory, name, max(config.counts), config.memory_mb)
            max_runs = len(job["counts"])*(len(job["checks"])+1+job["warmups"]+job["repeats"])
            timeout = 90 + (max_runs+2)*job["timeout"]
            try:
                run = run_process(args, timeout=timeout, input_bytes=json.dumps(job).encode(), limit=8_000_000)
            finally:
                # A timed-out docker client does not reliably terminate the container.
                run_process(["docker", "rm", "-f", name], timeout=15, limit=100_000)
            if run["failure"]:
                return {"status": "sandbox_failed", "error": run["failure"], "measurements": [], "correctness": []}
            try:
                result = json.loads(run["stdout"])
            except ValueError:
                return {"status": "sandbox_failed", "error": run["stderr"][-8000:] or "No valid worker result.", "measurements": [], "correctness": []}
            if run["returncode"] != 0 and result.get("status") == "valid":
                result = {"status": "sandbox_failed", "error": "Container exited unsuccessfully.", "measurements": [], "correctness": []}
            result["sandbox"] = info
            return result
