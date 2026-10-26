import pytest
from hpc_optimizer.config import ExperimentConfig
from hpc_optimizer.engine import Engine
from hpc_optimizer.sandbox import DockerSandbox, SandboxUnavailable
from hpc_optimizer.storage import Store
from hpc_optimizer.validation import correctness_cases

pytestmark=pytest.mark.docker


@pytest.fixture
def sandbox():
    value=DockerSandbox()
    try: value.preflight()
    except SandboxUnavailable as exc: pytest.skip(str(exc))
    return value


@pytest.mark.parametrize("benchmark,size",[("matmul",32),("jacobi",32),("monte_carlo",10000),("histogram",10000),("bfs",100)])
@pytest.mark.parametrize("paradigm",["openmp","mpi"])
def test_real_workload(benchmark,size,paradigm,sandbox,tmp_path):
    counts=[1,2] if sandbox.info["cpu_budget"]>=2 else [1]
    config=ExperimentConfig(benchmark=benchmark,size=size,paradigm=paradigm,counts=counts,repeats=3,steps=4)
    result=Engine(Store(tmp_path),sandbox).run(config)
    assert result["status"]=="completed",result.get("error")
    assert all(c["status"]=="valid" for c in result["candidates"]),result


@pytest.mark.parametrize("source,status",[("this is not C","compilation_failed"),("int main(void){for(;;){}}","incorrect")])
def test_real_failure_paths(source,status,sandbox):
    config=ExperimentConfig(size=8,counts=[1],timeout=1,repeats=3)
    job={"paradigm":"openmp","counts":[1],"checks":correctness_cases("matmul",42),
         "benchmark":{"size":8,"steps":2,"seed":42},"repeats":3,"warmups":0,"timeout":1,"profiler":"basic"}
    result=sandbox.evaluate(source,job,config)
    assert result["status"]==status and not result.get("measurements")
