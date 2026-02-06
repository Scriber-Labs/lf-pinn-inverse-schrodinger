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
    lambda_data: float = 1.0,
    lambda_density: float = 1.0,
    lambda_smooth: float = 1e-4,
) -> Tensor:
    """
    Composite loss for the inverse 1D time-independent Schrödinger problem.

    Combines:
      - physics-informed TISE residual
      - optional data mismatch loss on psi(x)
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
         Observed energy.
    x_density :
        Coordinates where density observed.
    rho_obs  :
        Observed density.

    lambda_phys :
        Weight for the TISE residual loss.
    lambda_data :
        Weight for the data mismatch loss.
    lambda_density :
        Weight for the density mismatch loss.
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
    # Data loss (optional)
    # ------------------------------------------------------------------
    if x_data is not None and psi_data is not None:
        psi_pred: Tensor = psi_model(x_data)
        L_data: Tensor = (psi_pred - psi_data).pow(2).mean()
        total_loss = total_loss  + lambda_data * L_data

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