# inverse-pinn-schrodinger

Inverse Schrödinger solver using a low-fidelity physics-informed neural network, POD diagnostics, and saved experiment artifacts.

## Setup

From the project root, create and activate a Python environment, then install dependencies:
```bash
python3 -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt
```
Alternatively, install from `pyproject.toml` in editable mode:
```bash
pip install -e .
```

## CLI usage

### Train a model

Run the default training workflow from the project root:
```bash
python3 -m cli.cli_train
```
This trains the inverse Schrödinger PINN, logs training metrics, and writes a new run directory under:
```text
text artifacts/runs/<run_id>/
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
```
bash python3 -m cli.cli_train --device cuda:0
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

## Extract metrics from a saved run

After training, generate CSV summaries from a saved run directory:
```bash
python3 -m cli.extract_metrics
--artifacts-dir artifacts/runs/<run_id>
--output-dir artifacts/runs/<run_id>
```

This writes:
```text
pod_metrics.csv training_analysis.csv
```

If `--output-dir` is omitted, outputs are written to the artifact directory.

### Run the metrics smoke test

To validate the metrics extraction workflow with dummy data:
```bash
python3 -m cli.extract_metrics --smoke-test
```

If no `--artifacts-dir` is provided, the metrics CLI defaults to smoke-test mode.

## 🦴 Repository Skeleton

```aiignore
lf-pin-inverse-schrodinger/
 ├── artifacts/ 
 │ ├── figures.md
 │ ├── demo_visuals/ 
 │ └── ...
 ├── assets/
 ├── cli/ 
 ├── data/ 
 ├── docs/
 ├── notebooks/ 
 ├── src/ 
 │ ├── init.py 
 │ ├── db_logger.py 
 │ ├── inverse.py 
 │ ├── model.py 
 │ ├── physics.py 
 │ ├── pod.py 
 │ ├── train.py 
 │ ├── utils.py 
 │ └── visualizations.py 
 └── ...
```
