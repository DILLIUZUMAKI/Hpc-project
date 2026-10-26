import json
import math
import sys
from unittest.mock import patch
import pytest

from hpc_optimizer.config import ExperimentConfig
from hpc_optimizer.validation import oracle, fingerprint, parse_output, compare, correctness_cases
from hpc_optimizer.metrics import summarize, add_speedups, choose
from hpc_optimizer.process import run_process
from hpc_optimizer.sandbox import DockerSandbox, SandboxUnavailable, container_args, validate_source
from hpc_optimizer.profiling import parse_perf, parse_cachegrind, analyze
from hpc_optimizer.llm import LLMConfig, LLMClient, LLMError, parse_generation
from hpc_optimizer.storage import Store


@pytest.mark.parametrize("benchmark", ["matmul","jacobi","monte_carlo","histogram","bfs"])
def test_oracle_full_output_contract(benchmark):
    for case in correctness_cases(benchmark, 4294967295):
        output = parse_output(json.dumps(case["expected"]))
        assert compare(output,case["expected"])[0]
    n = 17
    values = oracle(benchmark,n,3,42)
    if benchmark == "histogram":
        assert sum(values)==n
    if benchmark == "monte_carlo":
        assert 0<=values[0]<=n and values[1]==4*values[0]/n
    if benchmark == "bfs":
        assert values[0]==0 and all(0<=x<n for x in values)
    if benchmark == "jacobi":
        assert all(values[i]==0 for i in range(n))


@pytest.mark.parametrize("text", ['{}', '[]', '{"length":true}', '{"length":1,"sum":NaN}',
    '{"length":1,"sum":0,"sum_squares":0,"weighted_sum":0,"values":[true]}',
    '{"length":1,"sum":2,"sum_squares":4,"weighted_sum":2,"values":[1]}'])
def test_reject_malformed_or_inconsistent_output(text):
    with pytest.raises((ValueError, TypeError)):
        parse_output(text)


def test_numeric_mismatch_and_missing_vector_rejected():
    expected=fingerprint([1.,2.],True)
    assert not compare(fingerprint([1.,3.],True),expected)[0]
    assert not compare(fingerprint([1.,2.]),expected)[0]
    actual=fingerprint([1.+1e-10,2.],True)
    assert compare(actual,expected)[0]


def test_metrics_and_selection_use_real_baseline_and_all_counts():
    row={"workers":2,**summarize([2,3,4])}
    add_speedups([row],6)
    assert row["median_seconds"]==3 and row["speedup"]==2 and row["efficiency"]==1
    def candidate(t): return {"status":"valid","measurements":[{"workers":1,"median_seconds":t},{"workers":2,"median_seconds":t/2}]}
    assert choose(candidate(10),candidate(8),[1,2])[0]
    assert not choose(candidate(10),candidate(11),[1,2])[0]
    assert not choose(candidate(10),candidate(9.9),[1,2])[0]
    invalid=candidate(1); invalid["status"]="incorrect"
    assert not choose(candidate(10),invalid,[1,2])[0]
    incomplete=candidate(1); incomplete["measurements"].pop()
    assert not choose(candidate(10),incomplete,[1,2])[0]
    with pytest.raises(ValueError): summarize([0,math.inf])


@pytest.mark.parametrize("kwargs", [{"iterations":4},{"counts":[8]},{"counts":[]},{"counts":[True]},
    {"repeats":1},{"use_ai":True},{"seed":-1},{"profiler":"fake"},{"size":0}])
def test_configuration_rejects_invalid_or_unapproved_runs(kwargs):
    with pytest.raises(ValueError): ExperimentConfig(**kwargs).validate(2)


def test_sandbox_fails_closed_and_has_no_secret_mounts():
    with patch("hpc_optimizer.sandbox.shutil.which",return_value=None):
        with pytest.raises(SandboxUnavailable): DockerSandbox().preflight()
    args=container_args("image:1","/tmp/hpc-test","hpc-test",2,1024)
    assert args[args.index("--network")+1]=="none"
    assert "--read-only" in args and "ALL" in args and "65534:65534" in args
    assert "readonly" in args[args.index("--mount")+1]
    assert "/var/run/docker.sock" not in " ".join(args)
    for source in ["", "bad\x00source", "x"*256001]:
        with pytest.raises(ValueError): validate_source(source)


def test_timeout_and_output_limit_are_enforced():
    timeout=run_process([sys.executable,"-c","import time; time.sleep(10)"],timeout=.15)
    assert timeout["failure"]=="timeout" and timeout["seconds"]<3
    flood=run_process([sys.executable,"-c","import os; os.write(1,b'x'*500000)"],limit=1000)
    assert flood["failure"]=="output_limit" and len(flood["stdout"])<=1000


def test_profiler_parsers_do_not_invent_missing_counters():
    assert parse_perf("<not supported>;;cycles\n1000;;instructions\n")=={"instructions":1000.0}
    assert parse_cachegrind("events: Ir Dr D1mr\nsummary: 100 30 4\n")=={"Ir":100,"Dr":30,"D1mr":4}
    assert parse_cachegrind("failed")=={}
    analysis=analyze({"measurements":[],"profile":{"status":"unavailable"}})
    assert analysis["observed_bottlenecks"]==[] and "false sharing" in analysis["unmeasured"]


def test_llm_contract_and_secrets():
    content=json.dumps({"code":"int main(void){return 0;}","explanation":"Example test fixture"})
    assert parse_generation("```json\n"+content+"\n```")["code"].startswith("int")
    for text in ["some C", '{"code":"x"}', '{"code":null,"explanation":"x"}']:
        with pytest.raises(LLMError): parse_generation(text)
    config=LLMConfig("https://example.invalid/v1","test-model","DO_NOT_LOG_THIS")
    assert "DO_NOT_LOG_THIS" not in repr(config)
    with pytest.raises(LLMError): LLMConfig("http://remote.invalid/v1","m","k").validate()
    with pytest.raises(LLMError): LLMConfig("https://user:secret@example.invalid/v1","m","k").validate()
    with pytest.raises(LLMError): LLMConfig("https://example.invalid/v1","","k").validate()


def test_storage_blocks_path_traversal(tmp_path):
    store=Store(tmp_path)
    with pytest.raises(ValueError): store.load("../../.env")
    run=store.create(ExperimentConfig(counts=[1]))
    assert store.load(run["id"])["id"]==run["id"]
    assert len(store.list())==1
