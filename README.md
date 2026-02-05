# inverse-piml-schrodinger

## Planned Repo Structure

```
inverse-piml-schrodinger/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── model.py          # neural network ansatz for V(x)
│   ├── physics.py        # TISE residual + BCs
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

Using the time-independent Schrodinger equation (TISE) as a hard physics constraint, we treat tahe potential $V(x)$ as a learnable function while wavefunctions act as auxiliary fields constrained by the PDE. The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability. 
> Specifcially, the aim of this repository is to understand which operator features are robustly recoverable under strong physics priors and limited data.

---
Below is the complete code for your baseline inverse-Schrödinger PINN repo, reorganized under the structure we discussed and illustrating the five Python “good habits.”

1) src/model.py  
```python
from __future__ import annotations
from typing import Any
import torch
from torch import nn, Tensor

class PotentialNet(nn.Module):
    """2‐hidden‐layer tanh MLP: x → V(x)"""
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (N_x,1)
        Returns:
            V: Tensor of shape (N_x,1)
        """
        return self.net(x)

def _smoke_test() -> None:
    """Quick sanity check for PotentialNet."""
    model = PotentialNet()
    x = torch.linspace(-1, 1, 5).unsqueeze(1)
    V = model(x)
    print("✔️  PotentialNet forward OK; output shape:", V.shape)

if __name__ == "__main__":
    _smoke_test()
```

2) src/physics.py  
```python
from __future__ import annotations
import math
from typing import Any
import torch
from torch import Tensor, autograd

def se_residual_loss(
    psi: Tensor,
    x: Tensor,
    V: Tensor,
    E: Tensor,
    hbar: float = 1.0,
    m: float = 1.0,
) -> Tensor:
    """
    MSE of Schrödinger residual:
      -(ℏ²/2m) ψ'' + V ψ - E ψ = 0
    """
    d1: Tensor = autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    d2: Tensor = autograd.grad(d1, x, torch.ones_like(d1), create_graph=True)[0]
    resid: Tensor = - (hbar**2 / (2*m)) * d2 + V * psi - E * psi
    return (resid.pow(2)).mean()

def smoothness_loss(
    V: Tensor,
    x: Tensor,
) -> Tensor:
    """
    Penalize high curvature of V(x): mean |V''(x)|².
    """
    d1: Tensor = autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
    d2: Tensor = autograd.grad(d1, x, torch.ones_like(d1), create_graph=True)[0]
    return d2.pow(2).mean()

def _smoke_test() -> None:
    """Quick check of loss functions on dummy data."""
    import torch
    x = torch.linspace(-1,1,10, requires_grad=True).unsqueeze(1)
    psi = torch.sin(math.pi * x)
    V = torch.zeros_like(x)
    E = torch.tensor([math.pi**2/2], dtype=torch.double)
    Lr = se_residual_loss(psi, x, V, E)
    Ls = smoothness_loss(V, x)
    print(f"✔️  SE loss={Lr.item():.3e}, smoothness loss={Ls.item():.3e}")

if __name__ == "__main__":
    _smoke_test()
```

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
