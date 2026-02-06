from __future__ import annotations

from typing import Iterable, Callable

import torch
from torch import Tensor

from physics import (
    tise_residual_loss,
    potential_smoothness_loss,
    density_mismatch_loss,
)

__all__: list[str] = [
    "inverse_schrodinger_loss",
    "inverse_multi_state_loss",
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

    # --- optional observed data ---
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
        Observed probability density |psi(x_density)|^2.

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
        Scalar total loss for a single eigenmode.
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

def inverse_multi_state_loss(
    *,
    psi_models: Iterable[Callable[[Tensor], Tensor]],
    energies: Iterable[float | Tensor],
    V_model: Callable[[Tensor], Tensor],
    x_phys: Tensor,

    # --- optional observed data ---
    energies_obs: Iterable[float | Tensor] | None = None,
    x_densities: Iterable[Tensor] | None = None,
    rho_obs_list: Iterable[Tensor] | None = None,

    # --- weights ---
    lambda_phys: float = 1.0,
    lambda_energy: float = 1.0,
    lambda_density: float = 1.0,
    lambda_smooth: float = 1e-4,
) -> Tensor:
    """
    Multi-state inverse Schrödinger loss.

    Sum inverse losses over multiple eigenstates psi_n(x), all coupled through a shared potential V(x)

    This corresponds to an inverse spectral problem.

    Parameters
    ----------
    📝 Same parameters as inverse_schrodinger_loss except for the following:
        psi_models : Iterable[Callable[[Tensor], Tensor]]
            Callable mapping ``x -> psi_n(x)``.
        energies_obs :
            Observed energy eigenvalues from spectral data (one for each eigenmode).
        x_densities :
            Coordinates where probability density is observed (one for each eigenmode).
        rho_obs_list :
            Observed probability densities |psi(x_densities)|^2 (one for each eigenmode).

    Returns
    -------
    Tensor
        Scalar total loss for multiple eigenmodes.
    """
    total_loss: Tensor = torch.tensor(0.0, device=x_phys.device)

    psi_models = list(psi_models)
    energies = list(energies)
    energies_obs = list(energies_obs) if energies_obs is not None else None
    x_densities = list(x_densities) if x_densities is not None else None
    rho_obs_list = list(rho_obs_list) if rho_obs_list is not None else None

    for n, (psi_n, energy) in enumerate(zip(psi_models, energies)):
        total_loss = total_loss + inverse_schrodinger_loss(
            psi_models=psi_n,
            V_model=V_model,
            x_phys=x_phys,
            energy=energy,
            energy_obs=None if energies_obs is None else energies_obs[n],
            x_density=None if x_densities is None else x_densities[n],
            rho_obs=None if rho_obs_list is None else rho_obs_list[n],
            lambda_phys=lambda_phys,
            lambda_energy=lambda_energy,
            lambda_density=lambda_density,
            lambda_smooth=lambda_smooth,
        )

    return total_loss