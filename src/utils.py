# src/utils.py

from __future__ import annotations
import random
from typing import Optional

import numpy as np
import torch

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
    n_points: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a 1D spatial column vector grid tensor of shape (n_points,1).

    Parameters
    ----------
    x_min : float
        Minimum x value.
    x_max : float
        Maximum x value.
    n_points : int
        Number of grid points.
    device : torch.device or str, optional
        Device on which the parameters will be allocated. If omitted, they will be kept on the current default device.

    Returns
    -------
    x: shape(N, 1)
        1D spatial vector grid tensor.
    """
    x: torch.Tensor = torch.linspace(x_min, x_max, n_points, device=device)
    return x.unsqueeze(1)

def second_derivative(
    y: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Second-order central finite difference approximation for y''.

    Parameters
    ----------
    y : shape(N, 1)
    dx :
        grid spacing between points.

    Returns
    -------
    y_xx : shape(N, 1)
    """
    y_xx: torch.Tensor = (
        y[:-2] - 2.0 * y[1:-1] + y[2:]
    ) / dx**2

    # Pad to preserve shape
    return torch.cat(
        [y_xx[:1], y_xx, y_xx[-1:]],
        dim=0
    )

def l2_inner_product(
    f: torch.Tensor,
    g: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """Compute < f | g > via trapezoidal rule."""
    return torch.sum(f * g) * dx


# ------------------------------------------------------------------------------
# 2️⃣ Smoke Test and Entry point
# ------------------------------------------------------------------------------

def _smoke_test() -> None:
    """Quick test for make_grid & seeding."""
    set_global_seed(0, deterministic=True)
    device = torch.device("cpu")
    x = make_grid(-1.0, 1.0, 100, device)
    dx = float(x[1] - x[0])
    y = x**2
    y_xx = second_derivative(y, dx)

    print("✔️ utils.py sanity check passed")

if __name__ == "__main__":
    _smoke_test()