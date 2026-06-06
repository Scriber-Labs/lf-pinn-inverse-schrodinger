# inverse-pinn-schrodinger

Inverse Schrödinger solver using a low-fidelity physics-informed neural network (PINN), Proper Orthogonal Decomposition (POD) diagnostics, and reproducible experiment artifacts.

This project investigates how much spectral and operator structure caan be recovered from limited quantum observations using a lightweight inverse PINN workflow.
Training runs produce saved artifacts, diagnostic metrics, and figure analyses intended to support interpretability rather than benchmark optimization.

## 📖 Research Notebook
The primary research writeup for this project is hosted in the Scriber Labs Reserach Notebook:

🔗 https://scriber-labs.github.io/research-notebook/project_batch_1/project_2/

The Reseearch Notebook is maintained in a separate repository and serves as the narerative companion to this codebase. It contains:
- Design and architecture.
- Experimental notes.
- Description of loss terms.
- Figure analysis and interpretation.
- Discussion of model behavior and failure modes.
- Connections to related Scriber Labs projects.

This repository contains the implementation and generated artifacts; the Research Notebook contains the accompanying scientific writeup.

---

## 🔬 Research Artifacts
The following artifacts accompany the implementation:
```text
artifacts/demo_visuals/ # generated figures 
artifacts/figures.md # figure-by-figure analysis 
notebooks/demo.ipynb # local reproducible workflow
```

Generated figures in `notebooks/demo.ipynb` are treated as research artifacts rather than standalone visualization. The accompanying analyses document:
- What each diagnostic measures.
- Physical interpretation.
- Failure modes and limitations.
- Relationships between diagnostics.
- Interpretability and insights.

Key topics explored throughout the diagnostics include:
- Spectral recovery.
- Operator consistency.
- Eigenfunction structure.
- Potential identifiability
- Orthogonality and basis conditioning.
- POD-derived geometric structure.

---

## 🚀 Quick Start

### Create an environment

#### macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Alternatively:
```bash
pip install -e .
```

## Run Training
```bash
python -m cli.cli_train
```

Artifacts are written to:
```text
artifacts/runs/<run_id>/
```

The latest run identifier is also saved to:
```text
data/run_id.txt
```

## Generate Notebook-Matching Figures
```bash
python -m cli.cli_train --figures-dir artifacts/demo_visuals
```

## Extract Metrics
```bash
python -m cli.extract_metrics \ 
    --artifacts-dir artifacts/runs/<run_id>
```

## Full Usage Guide
Detailed CLI documentation is availible in:
```text
cli/USAGE_GUIDE.md
```

---

## 🦴 Repository Skeleton

```aiignore
inverse-pinn-schrodinger/
 ├── artifacts/ 
 │ ├── demo_visuals/ 
 │ ├── figures.md 
 │ └── runs/ 
 ├── assets/ 
 ├── cli/ 
 │ ├── cli_train.py 
 │ ├── extract_metrics.py 
 │ └── USAGE_GUIDE.md 
 ├── data/ 
 ├── docs/ 
 ├── notebooks/ 
 │ ├── demo.ipynb 
 │ └── load_run_analysis.ipynb 
 ├── src/ 
 ├── pyproject.toml 
 ├── README.md 
 └── requirements.txt
```
