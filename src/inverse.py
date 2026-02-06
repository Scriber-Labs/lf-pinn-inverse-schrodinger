from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from physics import (
    tise_residual_loss,
    potential_smoothness_loss,
    density_mismatch_loss,
)

__all__: list[str] = [
    "inverse_schrodinger_loss",
]

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def inverse_schrodinger_loss(
    *,
    psi_model: Callable[[Tensor], Tensor],
    V_model: Callable[[Tensor], Tensor],
    x_phys: Tensor,
    energy: float | Tensor,

    # --- data supervision ---
    energy_obs: float | Tensor | None = None,
    x_density: Tensor | None = None,
    rho_obs: Tensor | None = None,

    # --- weights ---
    lambda_phys: float = 1.0,
    lambda_energy: float = 1.0,
    lambda_density: float = 1.0,
    lambda_smooth: float = 1e-4,
) -> Tensor:
    """
    Composite loss for the inverse 1D time-independent Schrödinger problem.

    Combines:
      - physics-informed TISE residual
      - optional data mismatch losses on spectral energies and probability densities
      - smoothness regularization on V(x)

    Parameters
    ----------
    psi_model :
        Callable mapping ``x -> psi(x)``.
    V_model :
        Callable mapping ``x -> V(x)``.
    x_phys :
        Collocation points for enforcing the TISE.
    energy :
        Energy eigenvalue E.

    energy_obs :
         Observed energy eigenvalue (from spectral data).
    x_density :
        Coordinates where probability density is observed.
    rho_obs  :
        Observed probability density |psi(x)|^2.

    lambda_phys :
        Weight for the TISE residual loss.
    lambda_energy :
        Weight for the energy mismatch loss (from spectral data).
    lambda_density :
        Weight for the |psi|^2 mismatch loss (from density data).
    lambda_smooth :
        Weight for the potential smoothness regularizer.

    Returns
    -------
    Tensor
        Scalar total loss.
    """
    total_loss: Tensor = torch.tensor(0.0, device=x_phys.device)

    # ------------------------------------------------------------------
    # Physics loss
    # ------------------------------------------------------------------
    L_phys: Tensor = tise_residual_loss(
        psi_model,
        V_model,
        x_phys,
        energy=energy,
        reduction="mean",
    )
    total_loss = total_loss + lambda_phys * L_phys

    # ------------------------------------------------------------------
    # Spectral data: energy mismatch
    # ------------------------------------------------------------------
    if energy_obs is not None:
        L_energy = (energy - energy_obs).pow(2)
        total_loss = total_loss + lambda_energy * L_energy

    # ------------------------------------------------------------------
    # Density data: |psi|^2 mismatch
    # ------------------------------------------------------------------
    if x_density is not None and rho_obs is not None:
        L_density = density_mismatch_loss(
            psi_model,
            x_density,
            rho_obs,
            reduction="mean",
        )
        total_loss = total_loss + lambda_density * L_density
    # ------------------------------------------------------------------
    # Smoothness regularization
    # ------------------------------------------------------------------
    L_smooth: Tensor = potential_smoothness_loss(
        V_model,
        x_phys,
        reduction="mean",
    )
    total_loss = total_loss + lambda_smooth * L_smooth

    return total_loss