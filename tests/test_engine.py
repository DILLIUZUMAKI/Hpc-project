"""Synthetic fixtures test orchestration only; they are not experimental results."""
import json
from hpc_optimizer.config import ExperimentConfig
from hpc_optimizer.engine import Engine
from hpc_optimizer.metrics import summarize
from hpc_optimizer.storage import Store


class FakeSandbox:
    def __init__(self, times, invalid_index=None):
        self.times=iter(times); self.calls=0; self.invalid_index=invalid_index
    def preflight(self): return {"cpu_budget":2,"backend":"test_fixture"}
    def evaluate(self, source, job, config):
        self.calls+=1
        t=next(self.times)
        if self.calls==self.invalid_index:
            return {"status":"incorrect","measurements":[],"correctness":[{"passed":False}],"compilation":{"returncode":0}}
        return {"status":"valid","compilation":{"returncode":0,"failure":None},"correctness":[{"passed":True}],
                "benchmark_output":{"length":1,"sum":1,"sum_squares":1,"weighted_sum":1},
                "measurements":[{"workers":p,**summarize([t/p]*3)} for p in job["counts"]],
                "profile":{"status":"available","kind":"timing","counters":{}}}


class FakeLLM:
    def __init__(self): self.prompts=[]
    def generate(self,prompt):
        self.prompts.append(prompt)
        return {"code":"int main(void){return 0;}","explanation":"Synthetic test fixture, not real model output."}


def test_loop_keeps_best_and_saves_all_versions(tmp_path):
    llm=FakeLLM(); store=Store(tmp_path)
    config=ExperimentConfig(counts=[1,2],use_ai=True,allow_source_transfer=True,iterations=2)
    result=Engine(store,FakeSandbox([10,6,8,5,9]),llm).run(config)
    assert result["status"]=="completed" and result["best_ai"]=="ai_iteration_1"
    assert result["candidates"][-1]["accepted"] is False
    assert len(llm.prompts)==3
    assert "EVIDENCE:" not in llm.prompts[0]["user"]
    assert "EVIDENCE:" in llm.prompts[1]["user"]
    saved=store.load(result["id"])
    assert len(saved["candidates"])==5
    assert (store.directory(result["id"])/"ai_iteration_2.c").exists()
    assert "median_seconds" in (store.directory(result["id"])/"results.csv").read_text()


def test_bad_baseline_stops_and_invalid_revision_cannot_win(tmp_path):
    config=ExperimentConfig(counts=[1],use_ai=True,allow_source_transfer=True,iterations=1)
    llm=FakeLLM()
    result=Engine(Store(tmp_path/"bad"),FakeSandbox([10],invalid_index=1),llm).run(config)
    assert result["status"]=="failed" and len(result["candidates"])==1 and not llm.prompts
    result=Engine(Store(tmp_path/"revision"),FakeSandbox([10,8,7,1],invalid_index=4),llm).run(config)
    assert result["best_ai"]=="ai_baseline" and result["candidates"][-1]["status"]=="incorrect"


def test_no_ai_calls_in_reference_mode(tmp_path):
    llm=FakeLLM()
    result=Engine(Store(tmp_path),FakeSandbox([10,6]),llm).run(ExperimentConfig(counts=[1]))
    assert result["status"]=="completed" and not llm.prompts


def test_api_failure_preserves_prompt_and_completed_measurements(tmp_path):
    from hpc_optimizer.llm import LLMError
    class Broken:
        def generate(self,prompt): raise LLMError("API returned HTTP 401.")
    store=Store(tmp_path)
    config=ExperimentConfig(counts=[1],use_ai=True,allow_source_transfer=True)
    result=Engine(store,FakeSandbox([10,6]),Broken()).run(config)
    assert result["status"]=="failed" and len(result["candidates"])==2
    assert result["generation_attempts"][0]["status"]=="failed"
    assert "SERIAL C SOURCE" in store.load(result["id"])["generation_attempts"][0]["prompt"]["user"]
