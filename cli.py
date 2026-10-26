import argparse
import json
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from hpc_optimizer.config import ROOT, ExperimentConfig
from hpc_optimizer.engine import Engine
from hpc_optimizer.llm import LLMClient, LLMConfig
from hpc_optimizer.sandbox import DockerSandbox
from hpc_optimizer.storage import Store


def main():
    load_dotenv(ROOT/".env", override=False)
    parser = argparse.ArgumentParser(description="Run real, container-isolated HPC experiments.")
    parser.add_argument("--config", default=str(ROOT/"sample_config.json"))
    parser.add_argument("--source", help="Serial C source implementing the selected workload contract")
    parser.add_argument("--check", action="store_true", help="Check Docker and sandbox image only")
    args = parser.parse_args()
    try:
        config = ExperimentConfig(**json.loads(Path(args.config).read_text())).validate()
        sandbox = DockerSandbox(config.image)
        if args.check:
            print(json.dumps(sandbox.preflight(), indent=2)); return 0
        llm = LLMClient(LLMConfig.from_env()) if config.use_ai else None
        result = Engine(Store(os.getenv("HPC_RESULTS_DIR", str(ROOT/"experiments"))), sandbox, llm, print).run(
            config, Path(args.source).read_text() if args.source else None)
        print("Experiment:",result["id"],"Status:",result["status"])
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        print(f"Cannot run: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
