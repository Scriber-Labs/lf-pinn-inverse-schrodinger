# src/inverse.py

from __future__ import annotations

from typing import List

import torch

__all__: list[str] = [
    "data_mismatch_loss",
]

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def data_mismatch_loss(
    multi_psi_model: List[torch.Tensor],
    energies: torch.Tensor,
    rho_obs: List[torch.Tensor],
    E_obs: torch.Tensor,
) -> torch.Tensor:
    """
    Probability density mismatch loss:

        L_data = sum(  ||E_n - E_n_obs||^2 + || |psi_n_obs|^2 - rho_n_obs ||^2  )

    Parameters
    ----------
    multi_psi_model :
        List of eigenstates inferred from learned energies (❓).
    energies :
        (Like in `src/physics.py`, I'm officially a little lost here ❓)
    rho_obs :
        Observed probability density values.
    E_obs : torch.Tensor
        Observed energy for ❓(help)❓.

    Returns
    -------
    torch.Tensor
        Density mismatch loss.
    """
    energy_loss = torch.mean((energies - E_obs)**2)

    density_losses = [
        torch.mean((psi**2 - rho_obs[i])**2)
        for i, psi in enumerate(multi_psi_model)
    ]

    density_loss = torch.mean(torch.stack(density_losses))
    return energy_loss + density_loss

# ------------------------------------------------------------------------------
# 2️⃣ Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """❓‼️Need help with this part. would like to use the _run_smoke_test() method.‼️❓"""
    print("✔️ inverse.py loaded")

if __name__ == "__main__":
    main()
