# inverse-piml-schrodinger

## Planned Repo Structure

```
inverse-piml-schrodinger/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── model.py          # neural network ansatz for V(x)
│   ├── physics.py        # TISE residual, scale-aware smoothness, + BCs(❓)
│   ├── inverse.py        # inverse-specific losses
│   ├── pod.py            # POD decomposition + reconstruction
│   ├── train.py          # training loop and CLI
│   └── utils.py          # helper functions (e.g., seeding and grids)
├── notebooks/
│   └── demo.ipynb        # visual + narrative
├── references/
│   ├── README.md         # a blank README.md idk what for though. chat gpt told me to include to when i decided to make the references folder
│   └── references.bib    # sources (❓🙋🏻‍♀️ not necessarily used explicity by the github repo code, but that's mainly because i have no idea how to do that. if there is a standard way of implementing bib files easily like in overleaf im open to something like that... just no relearning entire languages and new software related bullshit (pardon my French)
├── assets/
│   ├── images/
│   │   └── interpretability_axis.png    # ⚠️ check spelling
└── artifacts/
    ├── notes.md                         # conceptual notes and reflection
    ├── figures.md                       # structure-based analysis of demo visuals
    └── demo_visuals/                          
```

---
This project extends the `lf-pinn-harmonic-oscillator` framework to an inverse quantum problem.
While project 1 investigated the robustness  of physics-informed neural networks (PINNs) under low fidelity discretization for a *known Hamiltonian*, this project is concerned with the information about an *unknown potential* that can be recovered from partial, noisy observations of quantum states.

Using the time-independent Schrodinger equation (TISE) as a hard physics constraint, we treat the potential $V(x)$ as a learnable function while wavefunctions act as auxiliary fields constrained by the PDE. The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability. 
> Specifically, the aim of this repository is to understand which operator features are robustly recoverable under strong physics priors and limited data.

---

## Loss Function
📝 For this repo, we will assume:
- atomic units and normalized parameters (thus, $\hbar=1$ and $m=1$ electron rest mass)
- $\psi_n$ normalization is handled either implicitly or by the PDE
- probability density observations are on an absolute scale

🔮 Later, can include:
- noisy or unnormalized densities
- partial observation windows
- unknown normalization constants
- orthogonality constraints between $\psi_n(x)$ 
- add energy ordering regularization

### TISE Residual
$$\mathcal{L}_\text{TISE}=\Bigg<\bigg(-\frac{\hbar^2}{2m}\psi_n''(x)+V(x)\psi_n(x)-E_n\psi_n(x)\bigg)^2\Bigg>$$

### Scale-Aware Smoothness
$$\mathcal{L}_\text{smooth}=\Bigg<\frac{|V''(x)|^2}{\epsilon + |V(x)|^2}\Bigg>$$

### Data Mismatch (Noisy Observations)
$$\mathcal{L}_{data} = \sum _n {\Big|\Big| E_n - E_n^\text{obs} \Big|\Big|^2 + \Big|\Big| \Big( |\psi_n |^2 - \rho_n^\text{obs} \Big)}\Big|\Big|^2 $$

### Total Loss
$$\mathcal{L}_\text{total}=\lambda_\text{data}\mathcal{L}_\text{data}+\lambda_\text{phys}\mathcal{L}_\text{TISE}+\lambda_\text{smooth}\mathcal{L}_\text{smooth}$$

---
3) src/utils.py  
```python
from __future__ import annotations
import random
from typing import Optional
import numpy as np
import torch

def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)

def make_grid(
    x_min: float,
    x_max: float,
    N_x: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a 1D column vector grid tensor of shape (N_x,1).
    """
    xs = torch.linspace(x_min, x_max, N_x, dtype=torch.double, device=device)
    return xs.unsqueeze(1)

def _smoke_test() -> None:
    """Quick test for make_grid & seeding."""
    set_global_seed(0, deterministic=True)
    grid = make_grid(-1.0, 1.0, 5)
    print("✔️  Grid:", grid.squeeze().tolist())

if __name__ == "__main__":
    _smoke_test()
```

4) src/train.py  
```python
from __future__ import annotations
import argparse
import os
from typing import Tuple
import numpy as np
import torch
from torch import nn, optim, Tensor
from model import PotentialNet
from physics import se_residual_loss, smoothness_loss
from utils import set_global_seed, make_grid

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train inverse‐PIML Schrödinger potential recovery"
    )
    parser.add_argument("--seed",        type=int,   default=1234)
    parser.add_argument("--x_min",       type=float, default=-1.0)
    parser.add_argument("--x_max",       type=float, default=1.0)
    parser.add_argument("--N_x",         type=int,   default=200)
    parser.add_argument("--N_modes",     type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=2000)
    parser.add_argument("--lambda_phys", type=float, default=1.0)
    parser.add_argument("--lambda_e",    type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=1e-3)
    parser.add_argument("--data_path",   type=str,   default="assets/data_schroedinger.npz")
    return parser.parse_args()

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load x (N_x,), psi (N_modes,N_x), E (N_modes,) from .npz.
    """
    data = np.load(path)
    return data["x"], data["psi"], data["E"]

def train(
    x: Tensor,
    psi_np: np.ndarray,
    E_np: np.ndarray,
    args: argparse.Namespace
) -> None:
    """
    Training loop: recover V(x) and E_pred.
    """
    # to torch tensors
    psi_obs: Tensor = torch.from_numpy(psi_np).double()         # (N_modes, N_x)
    E_obs:   Tensor = torch.from_numpy(E_np).double()           # (N_modes,)

    net = PotentialNet().double()
    E_pred = nn.Parameter(E_obs.clone())
    optimizer = optim.Adam(list(net.parameters()) + [E_pred], lr=args.lr)

    # spatial grid
    x = x.requires_grad_(True)  # (N_x,1)
    N_modes = psi_np.shape[0]

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        V = net(x)  # (N_x,1)

        # physics loss per mode using list comprehension
        L_phys = sum(
            se_residual_loss(
                psi_obs[n].unsqueeze(1), x, V, E_pred[n]
            ) for n in range(N_modes)
        ) / N_modes

        # data loss on eigenvalues
        L_e = nn.functional.mse_loss(E_pred, E_obs)

        # smoothness regularizer
        L_s = smoothness_loss(V, x)

        loss = args.lambda_phys * L_phys + args.lambda_e * L_e + args.lambda_smooth * L_s
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"L_phys={L_phys.item():.3e}  "
                f"L_e={L_e.item():.3e}  "
                f"L_smooth={L_s.item():.3e}"
            )

    # save checkpoint
    os.makedirs("artifacts/demo_visuals", exist_ok=True)
    torch.save(
        {"net_state": net.state_dict(), "E_pred": E_pred.detach().cpu()},
        "artifacts/demo_visuals/checkpoint.pt"
    )
    print("✔️  Training complete; checkpoint saved.")

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    x_np, psi_np, E_np = load_data(args.data_path)
    x = make_grid(args.x_min, args.x_max, args.N_x)
    train(x, psi_np, E_np, args)

if __name__ == "__main__":
    main()
```

Key Python habits demonstrated here:

1. Every script with executable behavior uses  
   ```python
   if __name__ == "__main__":
       main()
   ```
2. A `main() -> None:` function bundles orchestration.
3. Smaller helper functions (`parse_args`, `load_data`, `train`, `se_residual_loss`, etc.) keep logic modular.
4. Type annotations on all functions and variables for clarity.
5. List comprehensions in `train()` compute per-mode physics losses cleanly in one line.

You can now run:

  • `python -m src.train`  
  • Inspect the saved checkpoint under `artifacts/demo_visuals/`  
  • Extend with geometric/Clifford algebra notes in `artifacts/notes.md` as you explore covectors, bivectors, etc.
