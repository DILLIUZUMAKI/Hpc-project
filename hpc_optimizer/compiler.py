from pathlib import Path
import shutil
from .process import run_process


FLAGS = ["-std=c11", "-O2", "-g", "-Wall", "-Wextra", "-Wno-unused-function"]


def compile_source(source, destination, paradigm, include_dir):
    compiler = "mpicc" if paradigm == "mpi" else "gcc"
    if not shutil.which(compiler):
        return {"returncode": None, "failure": "missing_compiler", "stdout": "", "stderr": f"{compiler} unavailable"}
    args = [compiler, *FLAGS]
    if paradigm == "openmp":
        args.append("-fopenmp")
    args += [str(source), "-I", str(include_dir), "-o", str(destination), "-lm"]
    result = run_process(args, timeout=60, cwd=str(Path(destination).parent))
    result["compiler_version"] = run_process([compiler, "--version"], timeout=5)["stdout"].splitlines()[:1]
    return result
