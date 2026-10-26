"""Build known bundled reference sources; do not execute C on the host."""
import shutil
import subprocess
import pytest
from hpc_optimizer.config import ROOT, BENCHMARKS


@pytest.mark.parametrize("benchmark",list(BENCHMARKS))
@pytest.mark.parametrize("mode",["serial","openmp","mpi"])
def test_bundled_source_compiles(benchmark,mode,tmp_path):
    compiler=shutil.which("mpicc" if mode=="mpi" else "gcc")
    if not compiler: pytest.skip("Compiler not installed: "+mode)
    filename="serial.c" if mode=="serial" else "reference_"+mode+".c"
    args=[compiler,"-std=c11","-O2","-Wall","-Wextra","-Wno-unused-function"]
    if mode=="openmp": args.append("-fopenmp")
    args += [str(ROOT/"benchmarks"/benchmark/filename),"-I",str(ROOT/"benchmarks"),"-o",str(tmp_path/"program"),"-lm"]
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
