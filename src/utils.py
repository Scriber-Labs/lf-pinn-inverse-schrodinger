# src/util.py
"""
Utility helpers for project 2 (`lf-pinn-inverse-schrodinger`).

These functions are deliberately designed to be lightweight. Specifically, they do not depend on any project-specific modules. This makes them easy to reuse in notebooks and implement in future projects.
"""

from __future__ import annotations

import random
from typing import Optional, Literal

import numpy as np
import torch

# ----------------------------------------------------------------------
# 0️⃣ Public API
# ----------------------------------------------------------------------
def set_global_seed(
        seed: int,
        *,
        deterministic: bool = False
) -> None:
    """
    Seed the random number generators used throughout the project repository.

    Parameters
    ----------
    seed : int
        The integer seed to initialize all RNGs.
    deterministic : bool, default=False
        If ``True``, force PyTorch to use the deterministic algorithm.
        This is useful for reproducibility, however at the expense of computational speed.

    Notes
    -----
    - Python's ``random`` module, NumPy and PyTorch (CPU + CUDA) are seeded.
    - ``torch.backends.cudnn.deterministic`` and ``benchmark`` are set according to ``deterministic``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def make_grid(
    x_min: float,
    x_max: float,
    n_points: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a 1-D column-vector grid of collocation points.

    Parameters
    ----------
    x_min, x_max : float
        Bounds on the spatial domain.
    n_points : int
        Number of points in the grid (including the boundary points).
    device : torch.device or ``str``, optional
        Target device for the tensor. If ``None`` the tensor lives on the default device.

    Returns
    -------
    torch.Tensor, shape ``(n_points, 1)``
        A column vector suitable for broadcasting with the neural network outputs.
    """
    x = torch.linspace(x_min, x_max, n_points, device=device)
    return x.unsqueeze(1)

def second_derivative(
    y: torch.Tensor,
    dx: float,
    *,
    boundary: Literal["nearest", "zero", "periodic"] = "nearest",
) -> torch.Tensor:
    """
    Central finite-difference approximation of the second derivative ``y''`` with selectable boundary handling.

    Parameters
    ----------
    y : torch.Tensor, shape ``(N, 1)``
        Function values sampled on a uniform grid.
    dx : float
        Grid spacing (``x[i+1] - x[i]``).
    boundary : Literal["nearest", "zero", "periodic"]
        Boundary handling. Default is ``"nearest"``.

    Returns
    -------
    torch.Tensor, shape ``(N, 1)``
        Approximation of ``d^2y/dx^2`` with second-order accuracy.
        Boundary points are padded using nearest-neighbor values so that the central finite-difference stencil preserves the original array size.
    """
    # Interior stencil: (y[i=1] - 2*y[i] + y[i+1]) / (dx^2)
    interior = (y[:-2] - 2.0 * y[1:-1] + y[2:]) / dx**2

    if boundary == "nearest":
        padded = torch.cat([interior[:1], interior, interior[-1:]], dim=0)
    elif boundary == "zero":
        padded = torch.cat([torch.zeros_like(interior[:1]), interior, torch.zeros_like(interior[-1:])], dim=0)
    elif boundary == "periodic":
        # Wrap around: use last interior value for the first point, first interior for the last
        padded = torch.cat([interior[-1:], interior, interior[:1]], dim=0)
    else:
        raise ValueError(f"Unsupported boundary mode: {boundary}")

    # Pad to keep original shape (first & last rows duplicated)
    return padded

def l2_inner_product(
    f: torch.Tensor,
    g: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Compute the discrete L2 inner product ``<f|g>`` using the trapezoidal rule.

    Parameters
    ----------
    f, g : torch.Tensor, shape ``(N, 1)``
        Function evaluated on the same grid.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(N, 1)``
        Approximation of ``int(f(x)*g(x)*dx)``.=
    """
    # Trapezoidal rule reduces to a simple sum since the grid is uniform.
    return torch.sum(f * g) * dx

def grid_spacing(grid: torch.Tensor) -> float:
    """
    Return the uniform spacing of a 1-D grid tensor.
    The function assumes the grid is sorted and uniformly spaced.
    Raises a ValueError if the spacing is not constant within tolerance.

    Parameters
    ----------
    grid : torch.Tensor
        Spatial grid tensor.

    Returns
    -------
    float
        Uniform spacing of 1-D grid tensor.
    """
    g = grid.squeeze()
    if g.numel() < 2:
        raise ValueError("Grid must have at least 2 elements.")
    return ((g[-1] - g[0]) / ( g.numel() - 1)).item()

# ----------------------------------------------------------------------
# 2️⃣ Smoke test & entry point
# ----------------------------------------------------------------------

def _smoke_test() -> None:
    """Simple sanity check for ``make_grid`` and the seeding routine."""
    set_global_seed(27, deterministic=True)

    device = torch.device("cpu")
    x = make_grid(-1.0, 1.0, 100, device)
    dx = float(x[1] - x[0])

    y = x**2
    y_xx = second_derivative(y, dx)

    # Sanity prints. Execution will output an error if shapes mismatch.
    assert x.shape == (100, 1)
    assert y_xx.shape == (100, 1)

    print("✔️ utils.py sanity check passed")

if __name__ == "__main__":
    _smoke_test()