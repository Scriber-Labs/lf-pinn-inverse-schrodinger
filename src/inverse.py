# src/inverse.py
"""
Data-mismatch loss utilities.

These losses compare the *learned* quantum objects (wavefunctions and energy eigenvalues) against *observed* quantities (probability densities and measured energies). The functions expect tensors that have already been produced by the model -> they do **not** call the model themselves. This keeps the dependency graph clean.

- ``psi_theta_n``: learned wavefunctions/eigenmodes
- ``E_theta_n``: learned energies

Author: Eigenscribe
Development note: LLM assistance was used during construction; implementation has been reviewed and adapted for this project.
Review status: Reviewed and maintained by Eigenscribe.
Date: 02-2026
"""

from __future__ import annotations

from typing import List

import torch

__all__: list[str] = [
    "data_mismatch_loss",
]

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ----------------------------------------------------------------------

def data_mismatch_loss(
    multi_psi_theta: List[torch.Tensor],
    energies_theta: torch.Tensor,
    rho_obs: List[torch.Tensor],
    E_obs: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised loss term for data mismatching.

    Parameters
    ----------
    multi_psi_theta : List[torch.Tensor], each shape ``(M, 1)``
        List of learned wavefunctions/eigenmodes ``[psi0, psi1, ... ]``.
    energies_theta : torch.Tensor, shape ``(n_states,)``
        Tensor of learned energies ``[E0, E1, ... ]```.
    rho_obs : List[torch.Tensor], each shape ``(M, 1)``
        List of observed probability-density tensors (same shape as corresponding ``psi`` entries). Typically obtained from experiment or high-fidelity simulation.
    E_obs : torch.Tensor, shape ``(n_states,)``
        Tensor of observed energies.

    Returns
    -------
    torch.Tensor (scalar)
        Mean squared mismatch across all states.
    """
    # Energy term: simple MSE across the energy vector.
    energy_loss = torch.mean((energies_theta - E_obs) ** 2)

    # Density term: compare |psi|^2 to the observed density for each state.
    density_losses = [
        torch.mean((psi.squeeze() ** 2 - rho.squeeze()) ** 2)
        for psi, rho in zip(multi_psi_theta, rho_obs)
    ]

    density_loss = torch.mean(torch.stack(density_losses))
    return energy_loss + density_loss

# ----------------------------------------------------------------------
# 2️⃣ Smoke‑test entry point
# ----------------------------------------------------------------------

def _run_inverse_smoke_test() -> None:
    """
    Minimal sanity check that the loss accepts realistic shapes (same style as used in `src/physics.py`).
    """
    from utils import make_grid, set_global_seed

    set_global_seed(27)

    # Dummy spatial grid
    x = make_grid(-1.0, 1.0, 50)
    # Fake wavefunctions (simple sinusoids)
    psi0 = torch.sin(x)
    psi1 = torch.cos(2 * x)
    # Observed densities (square of the true functions plus noise)
    rho0 = psi0**2 + 0.01 * torch.randn_like(psi0)
    rho1 = psi1**2 + 0.01 * torch.randn_like(psi1)

    # Energies (learned vs. observed)
    energies_theta = torch.tensor([0.45, 1.55])
    E_obs = torch.tensor([0.5, 1.5])

    loss = data_mismatch_loss([psi0, psi1], energies_theta, [rho0, rho1], E_obs)

    print("✔️ inverse.pyl smoke test:")
    print(f"  loss value = {loss.item():.6f}")

def main() -> None:
    """Entry-point for ``python src/inverse.py`` -> runs the minimal smoke test."""
    _run_inverse_smoke_test()

if __name__ == "__main__":
    main()