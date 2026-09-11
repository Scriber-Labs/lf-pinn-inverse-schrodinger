# Developer Guidelines: `lf-pinn-inverse-schrodinger`

This guide provides project-specific configuration, testing workflows, and architectural standards for advanced developers working on the `lf-pinn-inverse-schrodinger` repository.

---

## 1. Build and Environment Setup

### Prerequisites
- **Python**: `>= 3.9` (Python 3.10–3.12 recommended)
- **PyTorch**: `>= 2.0` with CPU, CUDA, or Apple Silicon (MPS) support.

### Environment Installation
Create a virtual environment and install the dependencies:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

Alternatively, to install development extras directly via `pyproject.toml`:
```bash
pip install -e ".[dev]"
```

### Path Resolution
The core package lives under the `src/` directory without a top-level package namespace prefix (e.g., modules import each other via `from utils import ...` or `from physics import ...`).
- When running CLI tools from the repository root: `python -m cli.cli_train` or `python -m cli.extract_metrics`.
- When running scripts or test runners without an editable install: ensure `PYTHONPATH=src` is set in your environment (`export PYTHONPATH=src:$PYTHONPATH`).

---

## 2. Testing and Validation

### Built-in Self-Contained Smoke Tests
Each module in `src/` and CLI tool includes a standalone self-contained smoke test executed via `__main__`:

```bash
# Run individual module smoke tests
python3 src/physics.py
python3 src/model.py
python3 src/pod.py
python3 src/inverse.py
python3 src/train.py
python3 src/utils.py

# Run CLI metrics smoke test
python3 -m cli.extract_metrics --smoke-test
```

### Automated Unit Testing with Pytest
Pytest is used as the test framework (configured in `pyproject.toml`).

#### Running Tests
```bash
# Run pytest across the repository
PYTHONPATH=src pytest -v
```

#### Guidelines for Adding New Tests
When creating unit tests:
1. **Precision & Dtype**: Physics equations and model networks default to `torch.float64` (`Double`). Ensure input tensors and model initializations specify `dtype=torch.float64` to prevent `RuntimeError: mat1 and mat2 must have the same dtype`.
2. **Grid Layout**: Grid coordinates $x$ are 1D column vectors of shape `(N, 1)`.
3. **Module Imports**: Tests should import directly from the module names (e.g., `from physics import tise_residual`, `from model import InverseSchrodingerModel`) when `src` is in `PYTHONPATH`.
4. **Reproducibility**: Use `set_global_seed(seed)` from `utils` if testing stochastic training or initializations.

#### Working Test Example (`tests/test_core.py`)
```python
import torch
import pytest
from physics import tise_residual, potential_smoothness_loss, energy_ordering_loss
from model import InverseSchrodingerModel
from pod import pod_decomposition, physical_pod_decomposition

def test_tise_residual_harmonic_ground_state():
    dx = 0.05
    x = torch.linspace(-5.0, 5.0, 201, dtype=torch.float64).unsqueeze(-1)
    # Ground state harmonic oscillator: psi_0(x) = (1/pi)^(1/4) * exp(-x^2 / 2), E_0 = 0.5, V(x) = 0.5 * x^2
    psi = torch.exp(-0.5 * x**2)
    V = 0.5 * x**2
    energy = torch.tensor(0.5, dtype=torch.float64)
    
    res = tise_residual(psi, V, energy, dx=dx)
    assert res.shape == (201, 1)
    mse = torch.mean(res**2).item()
    assert mse < 1e-2

def test_inverse_schrodinger_model_forward():
    model = InverseSchrodingerModel(n_states=3, hidden_dims=[32, 32], dx=0.1, dtype=torch.float64)
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64).unsqueeze(-1)
    
    V_pred = model.V_theta(x)
    assert V_pred.shape == (101, 1)
    
    psis = model.psi_theta(x, dx=0.1)
    assert len(psis) == 3
    assert psis[0].shape == (101,)
    
    energies = model.E_theta()
    assert energies.shape == (3,)

def test_pod_decomposition_orthonormality():
    psi_matrix = torch.randn(100, 3, dtype=torch.float64)
    U, S, Vh = pod_decomposition(psi_matrix)
    assert U.shape == (100, 3)
    assert S.shape == (3,)
    # Verify Euclidean orthonormality U.T @ U == I
    eye_approx = U.T @ U
    assert torch.allclose(eye_approx, torch.eye(3, dtype=torch.float64), atol=1e-5)
```

---

## 3. Architecture & Code Style Conventions

### Code Structure
- `src/model.py`: `MLP` and `InverseSchrodingerModel` architecture. Parameterizes $V_\theta(x)$, $\psi_n^\theta(x)$, and $E_n^\theta$.
- `src/physics.py`: TISE residual calculation, smoothness loss $\int (V'')^2 dx$, and energy ordering penalty $\mathrm{ReLU}(E_n - E_{n+1} + \Delta E)$.
- `src/pod.py`: Proper Orthogonal Decomposition (Euclidean and physical measure-weighted SVD), spatial and temporal mode alignment, and modal overlap matrices.
- `src/train.py` & `cli/cli_train.py`: Unified loss function, training loops, SQLite run logging, and figure generation.
- `src/visualizations.py`: Diagnostic plot generation (potential, wavefunctions, energy ladder, POD singular values/modes, cross-overlap matrices).
- `src/db_logger.py`: Thread-safe SQLite logging into `data/training_runs.db`.
- `src/utils.py`: Central finite differences, numerical integration, global seeding, and normalization helpers.

### Code Style Standards
1. **Typing**: Use standard Python 3.9+ type annotations (`from __future__ import annotations`, `List`, `Tuple`, `Optional`, `Literal`, `torch.Tensor`).
2. **Precision & Dtype**: Default to `torch.float64` for neural nets and physical calculations to ensure numerical stability in higher-order derivatives (`second_derivative`).
3. **Module Layout**: Modules use numbered section headers with emojis (e.g., `# 1️⃣ Public API`, `# 2️⃣ Smoke‑test entry point`) and maintain an explicit `__all__` list.
4. **Data Shapes**:
   - 1D spatial coordinate: `(N, 1)`
   - Multi-state wavefunctions: `List[torch.Tensor]` each of shape `(N,)` or stacked snapshot matrix `(N_x, N_modes)`
   - Eigenvalues: 1D tensor `(n_states,)`
5. **Logging and Runs**: Run artifacts are saved under `artifacts/runs/<run_id>/` with metadata in `data/training_runs.db` and the current run ID in `data/run_id.txt`.
