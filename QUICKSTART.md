# Start ParallelLab on your Windows laptop

Run Windows commands in PowerShell and Linux commands inside Ubuntu. The project is a local application; you do not need a hosting service.

## 1. Finish Ubuntu / WSL2 installation

Open PowerShell as administrator:

```powershell
wsl --list --verbose
```

If Ubuntu is missing:

```powershell
wsl --install -d Ubuntu
```

Restart if requested, open Ubuntu and create your Linux username/password. If Ubuntu already exists with VERSION 1, convert the listed distribution using `wsl --set-version Ubuntu 2` (use its exact listed name).

## 2. Install Docker Desktop for Windows

Download from https://docs.docker.com/desktop/setup/install/windows-install/ and start it. Use Linux containers with the WSL2 engine. Under Settings > Resources > WSL Integration, enable your Ubuntu distribution and apply the change. Keep Docker Desktop running during experiments. Do not also install a second Docker daemon inside the same WSL distribution.

## 3. Extract the project

Download and extract `hpc-optimizer.zip`. In Ubuntu, copy the extracted `hpc-optimizer` directory into your Linux home folder. You can use File Explorer's Linux > Ubuntu > home > your Linux username location. Run the project from the Linux filesystem for consistent filesystem performance.

```bash
cd ~/hpc-optimizer
bash scripts/setup_ubuntu.sh
bash scripts/build_sandbox.sh
bash scripts/run.sh
```

The setup script installs Python dependencies. The Docker build installs GCC, OpenMPI and optional profilers inside the execution image. The first build downloads packages and needs internet access. If the folder is somewhere else, change the `cd` path accordingly.

Open **http://localhost:8501** in your Windows browser. If the browser opens automatically, use that window.

## 4. Run the first experiment without an API key

In **Code & benchmarks** choose Dense matrix multiplication, OpenMP, size 256, five measured repetitions and worker counts supported by your laptop (start with 1 and 2). Keep AI mode off and profiler set to basic. Click **Run experiment**. Open Compilation & correctness, then Results comparison. The serial and curated reference rows will contain actual measurements. No AI result is simulated.

If one-shot startup overhead dominates, increase size to 512 and rerun under the same conditions. Keep CPU/memory within the displayed budget. A slower parallel result is a valid finding.

## 5. Enable the AI loop

Edit `.env` locally:

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_actual_key_here
LLM_MODEL=your_provider_model_id
```

Use the provider's API key and currently enabled model ID. Alternatively enter them in the app's session-only fields. Do not include keys in source code or exported files. An external API may charge for usage. No model training or local GPU is required with an external endpoint.

Enable AI mode, set 1–3 optimization iterations, review the source-transfer disclosure and check its box. Run the experiment. The model receives your serial source and contract, then the measured evidence and current source for each revision. A local OpenAI-compatible server can also be used through a localhost base URL.

## 6. MPI, tests and command-line usage

Select MPI after the OpenMP run passes. This is single-machine MPI, not a multi-node cluster.

```bash
.venv/bin/python cli.py --check
.venv/bin/python cli.py --config sample_config.json
bash scripts/test.sh -m "not docker"
bash scripts/test.sh -m docker
```

The Docker integration tests require the sandbox image. Results and source versions are stored under `experiments/<run-id>/`; CSV, JSON and Markdown exports are available in the dashboard. Read `docs/VALIDATION.md` for the checks actually performed during delivery.
