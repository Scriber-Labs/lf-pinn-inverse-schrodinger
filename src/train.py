# src/train.py
"""
Training loop for the inverse Schrödinger problem.

The loss is a weighted sum of three physically motivated terms:
- **Physics loss**: enforces the TISE on the learned wavefunctions.
- **Smoothness loss**: regularizes the potential to avoid spurious wiggles.
- **Data-fit loss**: matches learned quantities to observed densities/energies.

All three components live in separate modules (`physics`, `inverse`, `utils`). This file wires them together, handles the optimizer, and provides a minimal smoke-test that runs a single optimization step.
"""

from __future__ import annotations

from typing import List

import torch
from model import InverseSchrodingerModel
from physics import tise_loss, potential_smoothness_loss
from inverse import data_mismatch_loss
from utils import make_grid, set_global_seed

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ----------------------------------------------------------------------

def train_step(
    model: InverseSchrodingerModel,
    x: torch.Tensor,
    dx: float,
    rho_obs: List[torch.Tensor],
    E_obs: torch.Tensor,
    lambdas: dict[str, float],
) -> torch.Tensor:
    """
    Execute a single gradient descent step.

    Parameters
    ----------
    model :
        Instance of :class:`~model.InverseSchrodingerModel`.
    x: torch.Tensor, shape ``(N, 1)``
        Collocation points on which the PDE is enforced.
    dx : float
        Uniform
    rho_obs
    E_obs
    lambdas

    Returns
    -------

    """