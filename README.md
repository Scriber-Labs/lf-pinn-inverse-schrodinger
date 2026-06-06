# inverse-pinn-schrodinger

Inverse Schrödinger solver using a low-fidelity physics-informed neural network, POD diagnostics, and saved experiment artifacts.

The main command-line workflow mirrors the demo notebook: it trains the inverse Schrödinger PINN on a synthetic harmonic oscillator problem, logs metrics, saves run artifacts, and generates notebook-matching figures.

## Research ARtifacts

```text
artifacts/demo_visuals/     # generated figures
artifacts/figures.md        # figure-by-figure analysis
notebooks/demo.ipynb        # reproducible workflow
```


## Setup

From the project root, create and activate a Python enviornment.

### macOS

```bash
python3 -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Alternatively, install from `pyproject.toml` in editable mode:
```bash
pip install -e .
```

## CLI usage

For the full CLI walkthrough, see: 
```text
docs/cli_user_guide.md
```

### Train a model and generate notebook-matching figures

Run the default training workflow from the project root:

```bash
python -m cli.cli_train
```
This trains the inverse Schrödinger PINN, logs training metrics, and writes a new run directory under:
```text
text artifacts/runs/<run_id>/
```

The latest run ID is also written to:
```text
data/run_id.txt
```
By default, figures are saved into the same run directory under
```text
artifacts/runs/<run_id>/figures/
```
### Recreate the demo visual outputs
To write the notebook-style figures directly into `artifacts/demo_visuals/`, run:
```bash
python -m cli.cli_train --figures-dir artifacts/demo_visuals
```

The CLI generates the following notebook-matching PNG files:
```text
training_curves.png 
learned_potential.png 
learned_wavefunctions.png 
learned_energies.png density.png 
pod_singular_values.png 
pod_modes.png overlap_heatmap.png 
cross_overlap_heatmap.png 
pod_eigen_alignment.png 
pod_temporal_modes.png 
pod_temporal_overlap.png
```
### Display figures interactively
To display figures after saving them:
```bash
python -m cli.cli_train --show-figures
```
For batch or headless runs, you can close figures after saving:
```bash
python -m cli.cli_train --close-figures
```

### Notebook-matching default configuration
The CLI defaults are aligned with the demo workflow:

| Setting | Default |
| :------ | ------: |
| `--n_modes` | `3` |
| `--hidden` | `64` |
| `--epochs` | `6000` |
| `--lr` | `0.005` |
| `--n_points` | `256` |
| `--seed` | `27` |
| `--log_every` | `800` |
| `--lambda-data` | `1.5` |
| `--lambda-physics` | `0.5` |
| `--lambda-smooth` | `0.05` |
| `--lambda-ordered` | `1.0` |


### Common training options

You can override the main training hyperparameters from the command line:

#### macOS
```bash
python -m cli.cli_train
--n_modes 3
--hidden 64
--epochs 6000
--lr 0.005
--n_points 256
--seed 27
--device cpu
--log_every 800
```

#### Windows Powershell
```powershell
python -m cli.cli_train 
--n_modes 3 
--hidden 64 
--epochs 6000 
--lr 0.005 
--n_points 256 
--seed 27 
--device cpu 
--log_every 800
```

#### Use CUDA when available:
```bash
python -m cli.cli_train --device cuda
```

Or select a specific cuda device:
```bash
python -m cli.cli_train --device cuda:0
```

### Loss-weight configuration

The CLI exposes the main loss weights:
```bash
python -m cli.cli_train
--lambda-data 1.5
--lambda-physics 0.5
--lambda-smooth 0.05
--lambda-ordered 1.0
```

Available loss-weight flags:

| **Flag** | **Description** |
| :------- | :-------------- |
| `--lambda-data` | Weight for density and energy data-fit loss |
| `--lambda-physics` | Weight for the time-independent Schrödinger equation residual |
| `--lambda-smooth` | Weight for potential smoothness regularization |
| `--lambda-ordered` | Weight for the energy-ordering constraint |

### SQLite logging

By default, training loogs to:
```text
data/training_runs.db
```

To use a custom database path:
```bash
python -m cli.cli_train --log-db data/my_training_runs.db
```

A completed run includes files such as:
```text
config.json 
history.json 
diagnostics.npz 
ground_truth.pt 
model.pt 
run_id.txt
```

The latest run ID is also written to:
```text
data/run_id.txt
```

### Common training options

You can override the main training hyperparameters from the command line:
```bash
python3 -m cli.cli_train
--n_modes 3
--hidden 64
--epochs 6000
--lr 0.005
--n_points 256
--seed 27
--device cpu
--log_every 800
```

Use CUDA when available:
```bash 
python3 -m cli.cli_train --device cuda
```

Or select a specific CUDA device:
```bash 
python3 -m cli.cli_train --device cuda:0
```

### Loss-weight configuration
The CLI exposes the main loss weights:
```bash
python3 -m cli.cli_train
--lambda-data 1.5
--lambda-physics 0.5
--lambda-smooth 0.25
--lambda-ordered 1.0
```

Available loss-weight flags:

| **Flag** | **Description** |
| :------- | :-------------- |
| `--lambda-data` | Weight for density and energy data-fit loss |
| `--lambda-physics` | Weight for time-independent Schrödinger equation residual |
| `--lambda-smooth` | Weight for potential smoothness regularization |
| `--lambda-ordered` | Weight for energy-ordering constraint |

### SQLite logging

By default, training loogs to:
```text
data/training_runs.db
```

To use a custom database path:
```bash
python3 -m cli.cli_train --log-db data/my_training_runs.db
```

### Quick smoke-style training check
To verify that the CLI works without running the full 6000-epoch workflow:
```bash
python -m cli.cli_train --epochs 10 --log_every 1 --device cpu
```

```bash
python -m cli.extract_metrics
--artifacts-dir artifacts/runs/<run_id>
--output-dir artifacts/runs/<run_id>
```

This writes:
```text
pod_metrics.csv training_analysis.csv
```

If `--output-dir` is ommitted, outputs are written to the artifact directory.

### Run the metrics smoke test

To validate the metrics extraction wrokflow with dummy data:
``bash
python -m cli.extract_metrics --smoke-test
``

If no `--artifacts-dir` is provided, the metrics CLI defaults to smoke-test mode.


## 📖 Demo Notebookk, Writeups, and Figure Analysis

The primary writeup for this project is in the Scriber Labs research notebook website:

🔗 https://scriber-labs.github.io/research-notebook/project_batch_1/project_2/

### Figure Analysis

This repository treats figure interpretation as a first-class research artifact.

The file `artifacts/figures.md` containes detailesd writeups for each generated figure, including:
- What is being shown.
- Physical interpretation.
- Failure modes and limitations.
- Connections between diagnostics.
- Interpretability insights.

Rather than viewing figures as standalone plots, the goal is to use them as diagnostic tools for understanding:
- Spectral recovery.
- Operator consistency.
- Eigenfunction structure.
- Identifiability of the learned potential.
- Emergent orthogonality and basis conditioning.
- POD-derived geometric structure.

### Writeup Philosophy

This project emphasizes interpretability over benchmark performance.

The write-ups going with each figure attempt to answer questions such as:
- What aspects of the Hamiltonian were successfully recovered?
- How do the learned eigenstates relate to the underlying operator?
- What failure modes emerge under low-fidelity constraints?
- What additional information is revealed by POD that are not directly enforced during training?

A successful experiment is therefore not defined solely by a low loss value, but by whether the resulting diagnostics support a coherent physical interpretation.

### Related Research Notes

Additional project notes and exploratory write-ups can be found in:
```text
artifacts/figures.md 
artifacts/interpretability.md 
artifacts/project_1_followup.md
```

These documents contain ongoing analysis, interpretability discussions, and connections between the inverse Schrödinger problem, operator learning, spectral geometry, and low-fidelity PINN behavior.

## 🦴 Repository Skeleton

```aiignore
lf-pin-inverse-schrodinger/
  ├── artifacts/ 
  │   ├── demo_visuals/ 
  │   ├── figures.md 
  │   └── runs/ 
  ├── assets/ 
  ├── cli/ 
  │   ├── init.py 
  │   ├── cli_train.py 
  │   └── extract_metrics.py 
  ├── data/ 
  ├── docs/ 
  │   └── cli_user_guide.md 
  ├── notebooks/ 
  │   ├── demo.ipynb 
  │   └── load_run_analysis.ipynb 
  ├── src/ 
  │   ├── __init__.py 
  │   ├── db_logger.py 
  │   ├── inverse.py 
  │   ├── model.py 
  │   ├── physics.py 
  │   ├── pod.py 
  │   ├── train.py 
  │   ├── utils.py 
  │   └── visualizations.py 
  ├── pyproject.toml 
  ├── README.md 
  └── requirements.txt
```
