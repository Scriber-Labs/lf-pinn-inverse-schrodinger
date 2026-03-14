# src/train.py
"""
Training loop for the inverse Schrödinger problem.

The loss is a weighted sum of three physically motivated terms:
- **Physics loss**: enforces the TISE on the learned wavefunctions.
- **Wavefunction normalization loss**: enforces normalization of the learned wavefunctions.
- **Smoothness loss**: regularizes the potential to avoid spurious wiggles.
- **Data-fit loss**: matches learned quantities to observed densities/energies.

All three components live in separate modules (`physics`, `inverse`, `utils`). This file wires them together, handles the optimizer, and provides a minimal smoke-test that runs a single optimization step.
"""

from __future__ import annotations

from typing import List

import torch
from model import InverseSchrodingerModel
from physics import tise_loss, potential_smoothness_loss, wavefunction_normalization_loss
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
    x: torch.Tensor, shape ``(n_points, 1)``
        Collocation points on which the PDE is enforced.
    dx : float
        Uniform grid spacing.
    rho_obs : List[torch.Tensor], each shape ``(n_states, n_points)``
        List of observed probability density tensors, one per eigenstate.
    E_obs : torch.Tensor, shape ``(n_states,)``
        Tensor of observed energies.
    lambdas : dict[str, float]
        Dictionary mapping loss identifies to scalar weights, e.g.
        ``{'data': 1.0, 'physics': 1.0, 'smooth': 1e-2, 'norm': 10.0}``.

    Returns
    -------
    torch.Tensor
        The total weighted loss (scalar) -> ready for ``backward()``.
    """
    # ----- Forward pass -------------------------------------------------
    V_theta = model.V_theta(x)      # potential V(theta, x)
    psi_list = model.psi_theta(x)   # list[psi_n(theta, x)]

    # ----- Individual loss terms ----------------------------------------
    L_physics = tise_loss(psi_list, V_theta, model.E_theta(), dx)
    L_norm = wavefunction_normalization_loss(psi_list,dx)
    L_smooth = potential_smoothness_loss(V_theta, dx)
    L_data = data_mismatch_loss(psi_list, model.E_theta(), rho_obs, E_obs)

    # ----- Weighted sum --------------------------------------------------
    total = (
        lambdas["data"] * L_data
        + lambdas["physics"] * L_physics
        + lambdas["smooth"] * L_smooth
        + lambdas["norm"] * L_norm
    )

    return total

# ----------------------------------------------------------------------
# 2️⃣ Smoke‑test entry point
# ----------------------------------------------------------------------
def _run_train_smoke_test() -> None:
    """
    Runs a *single* optimization step on synthetic data.
    Useful for CI pipelines or quick sanity checks inside a notebook.
    """
    set_global_seed(27)

    device = torch.device("cpu")
    model = InverseSchrodingerModel(
        n_states=3,
        hidden_dims=[64, 64],
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Spatial grid
    x = make_grid(-5.0, 5.0, 128, device)
    dx = float(x[1] - x[0])

    # Synthetic observations
    rho_obs = [torch.exp(-x**2).squeeze() for _ in range(3)]
    E_obs = torch.tensor([0.5, 1.5, 2.5], device=device)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "norm": 1e-2}

    optimizer.zero_grad()
    loss = train_step(model, x, dx, rho_obs, E_obs, lambdas)
    loss.backward()
    optimizer.step()

    print("✔️ train.py smoke test - loss after one step:", float(loss.detach()))

def main() -> None:
    """Entry point for ``python -m src.train`` -> runs the smoke test."""
    _run_train_smoke_test()

if __name__ == "__main__":
    main()