"""Real reference evaluation through the worker using a test-only seccomp executor.

Runtime measurements include test-wrapper startup and are never bundled as
research results or equated to measurements from the production Docker image.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from sandbox.worker import run_job
from hpc_optimizer.config import ROOT
from hpc_optimizer.process import run_process
from hpc_optimizer.validation import correctness_cases
from test_static_smoke import require_platform, RUNNER

pytestmark=pytest.mark.seccomp


def static_compile(source,destination,paradigm,include_dir):
    args=["gcc","-static","-std=c11","-O2",str(source),"-I",str(include_dir),"-o",str(destination),"-lm"]
    if paradigm=="openmp": args.append("-fopenmp")
    return run_process(args,timeout=30)


def isolated_execute(args,**kwargs):
    return run_process([sys.executable,str(RUNNER),*args],**kwargs)


def evaluate_reference(tmp_path,benchmark,variant,expected=None):
    source_dir=tmp_path/variant; source_dir.mkdir()
    work=source_dir/"work"; work.mkdir()
    shutil.copy(ROOT/"benchmarks"/benchmark/(variant+".c"),source_dir/"program.c")
    shutil.copy(ROOT/"benchmarks/common.h",source_dir/"common.h")
    serial=variant=="serial"
    job={"paradigm":"serial" if serial else "openmp", "counts":[1] if serial else [1,2],
         "checks":correctness_cases(benchmark,42),"expected_benchmark":expected,
         "benchmark":{"size":32 if benchmark in {"matmul","jacobi"} else 500,"steps":4,"seed":42},
         "timeout":5,"repeats":3,"warmups":1,"profiler":"basic"}
    result=run_job(job,work_dir=work,input_dir=source_dir,executor=isolated_execute,compiler=static_compile)
    result["environment"]["timing_scope"]="TEST ONLY: static executable plus seccomp helper launch"
    return result


@pytest.mark.parametrize("benchmark",["matmul","jacobi","monte_carlo","histogram","bfs"])
def test_worker_runs_full_real_reference_evaluation(tmp_path,benchmark):
    require_platform()
    baseline=evaluate_reference(tmp_path,benchmark,"serial")
    assert baseline["status"]=="valid",baseline
    parallel=evaluate_reference(tmp_path,benchmark,"reference_openmp",baseline["benchmark_output"])
    assert parallel["status"]=="valid",parallel
    assert len(parallel["measurements"])==2 and len(parallel["timed_runs"])==6
    assert all(row["median_seconds"]>0 and len(row["samples_seconds"])==3 for row in parallel["measurements"])
    assert all(check["passed"] for check in parallel["correctness"])


def test_worker_rejects_wrong_result_before_timing(tmp_path):
    require_platform()
    result=evaluate_reference(tmp_path,"matmul","reference_openmp",{"length":1024,"sum":0,"sum_squares":0,"weighted_sum":0})
    assert result["status"]=="incorrect" and result["measurements"]==[]
    assert "timed_runs" not in result
