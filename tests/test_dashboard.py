import pytest
from streamlit.testing.v1 import AppTest
from hpc_optimizer.config import ROOT, ExperimentConfig
from hpc_optimizer.engine import Engine
from hpc_optimizer.storage import Store
from test_engine import FakeSandbox, FakeLLM
from hpc_optimizer.sandbox import SandboxUnavailable

PAGES=["Overview","Code & benchmarks","AI generation","Compilation & correctness","Benchmarking","Profiling",
       "Iterative optimization","Results comparison","Experiment history","Documentation"]


@pytest.mark.parametrize("populated",[False,True])
def test_every_page_renders_without_exception(tmp_path,monkeypatch,populated):
    monkeypatch.setenv("HPC_RESULTS_DIR",str(tmp_path))
    if populated:
        Engine(Store(tmp_path),FakeSandbox([10,6,8,5]),FakeLLM()).run(
            ExperimentConfig(counts=[1],use_ai=True,allow_source_transfer=True,iterations=1))
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=20).run()
    assert not app.exception
    for page in PAGES:
        app.sidebar.radio(key="nav").set_value(page).run()
        assert not app.exception,(page,app.exception)


def test_run_button_saves_and_selects_new_experiment(tmp_path,monkeypatch):
    monkeypatch.setenv("HPC_RESULTS_DIR",str(tmp_path))
    monkeypatch.setattr("hpc_optimizer.sandbox.DockerSandbox",lambda *args,**kwargs: FakeSandbox([10,6]))
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=20).run()
    app.sidebar.radio(key="nav").set_value("Code & benchmarks").run()
    app.multiselect[0].set_value([1]).run()
    next(b for b in app.button if b.label=="Run experiment").click().run()
    assert not app.exception
    runs=Store(tmp_path).list()
    assert len(runs)==1 and runs[0]["status"]=="completed"
    assert app.session_state["selected_run"]==runs[0]["id"]


def test_missing_docker_is_an_error_not_success(tmp_path,monkeypatch):
    monkeypatch.setenv("HPC_RESULTS_DIR",str(tmp_path))
    class Missing:
        def preflight(self): raise SandboxUnavailable("Test: Docker is unavailable")
    monkeypatch.setattr("hpc_optimizer.sandbox.DockerSandbox",Missing)
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=20).run()
    next(b for b in app.button if b.label=="Check Docker environment").click().run()
    assert not app.exception and app.error and not app.success
