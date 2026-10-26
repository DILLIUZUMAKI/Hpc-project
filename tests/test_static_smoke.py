"""Extra actual C execution checks when Docker is absent.

Only bundled, reviewed references are compiled with static linking. Executables
run under a deny-by-default seccomp policy, resource limits and a wall timeout.
This test path is not exposed to uploads/LLM candidates or the application.
"""
import ctypes.util
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from hpc_optimizer.config import ROOT, BENCHMARKS
from hpc_optimizer.process import run_process
from hpc_optimizer.validation import correctness_cases, parse_output, compare

pytestmark=pytest.mark.seccomp
RUNNER=ROOT/"tests/support/static_seccomp_exec.py"


def require_platform():
    if platform.system()!="Linux" or platform.machine()!="x86_64" or not ctypes.util.find_library("seccomp") or not shutil.which("gcc"):
        pytest.skip("Static seccomp smoke tests require x86_64 Linux, GCC and libseccomp.")


def compile_static(source,binary,openmp=False):
    args=["gcc","-static","-std=c11","-O2",str(source),"-I",str(ROOT/"benchmarks"),"-o",str(binary),"-lm"]
    if openmp: args.append("-fopenmp")
    run=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if run.returncode:
        pytest.skip("Static toolchain is unavailable: "+run.stderr[:300])


@pytest.mark.parametrize("benchmark",list(BENCHMARKS))
@pytest.mark.parametrize("mode",["serial","openmp"])
def test_actual_reference_matches_independent_oracle(benchmark,mode,tmp_path):
    require_platform()
    source=ROOT/"benchmarks"/benchmark/("serial.c" if mode=="serial" else "reference_openmp.c")
    binary=tmp_path/"program"
    compile_static(source,binary,mode=="openmp")
    for p in ([1] if mode=="serial" else [1,2,4]):
        for case in correctness_cases(benchmark,4294967295):
            args=[sys.executable,str(RUNNER),str(binary),str(case["size"]),str(case["steps"]),str(case["seed"]),"--check"]
            result=run_process(args,timeout=10,env={"PATH":os.environ.get("PATH",""),"OMP_NUM_THREADS":str(p)})
            assert result["returncode"]==0 and not result["failure"],result
            actual=parse_output(result["stdout"])
            assert compare(actual,case["expected"])[0],(benchmark,mode,p,case,actual)


def test_seccomp_denies_file_network_process_creation_and_times_out(tmp_path):
    require_platform()
    source=tmp_path/"probe.c"
    source.write_text('''#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <errno.h>
int main(void) {
 int a=open("/etc/passwd",O_RDONLY); if(a!=-1 || errno!=EPERM) return 1;
 int b=socket(AF_INET,SOCK_STREAM,0); if(b!=-1 || errno!=EPERM) return 2;
 int c=fork(); if(c!=-1 || errno!=EPERM) return 3;
 return 0;
}''')
    binary=tmp_path/"probe"
    compile_static(source,binary)
    result=run_process([sys.executable,str(RUNNER),str(binary)],timeout=10)
    assert result["returncode"]==0,result
    source.write_text('int main(void){for(;;){}}')
    compile_static(source,binary)
    result=run_process([sys.executable,str(RUNNER),str(binary)],timeout=.3)
    assert result["failure"]=="timeout" and result["seconds"]<3
