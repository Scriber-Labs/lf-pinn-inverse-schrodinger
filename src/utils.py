from __future__ import annotations
import random
from typing import Optional

import numpy as np
import torch

# ------------------------------------------------------------------
# Helper: compute d/dt of the network output using autograd
# ------------------------------------------------------------------
def model_derivative(
    net: torch.nn.Module,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Returns ∂ₜ net(t)  (the velocity) evaluated at the times `t`.

    Parameters
    ----------
    net : torch.nn.Module
        Trained PINN that maps scalar time → position.
    t : torch.Tensor
        Shape ``(N, 1)`` – time points (requires_grad=True).

    Returns
    -------
    torch.Tensor
        Shape ``(N, 1)`` – time derivative of the network output.
    """
    t = t.clone().detach().requires_grad_(True)          # ensure grad tracking
    x = net(t)                                          # forward pass → position
    # torch.autograd.grad returns a tuple; we take the first element
    dx_dt, = torch.autograd.grad(
        outputs=x,
        inputs=t,
        grad_outputs=torch.ones_like(x),
        create_graph=False,      # we only need the first derivative
        retain_graph=False,
    )
    return dx_dt

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Integer seed.
    deterministic : bool, default = False
        If True, enables deterministic algorithms (slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.maniual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

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

# ------------------------------------------------------------------------------
# 2️⃣ Smoke Test and Entry point
# ------------------------------------------------------------------------------

def _smoke_test() -> None:
    """Quick test for make_grid & seeding."""
    set_global_seed(0, deterministic=True)
    grid = make_grid(-1.0, 1.0, 5)
    print("✔️  Grid:", grid.squeeze().tolist())

if __name__ == "__main__":
    _smoke_test()