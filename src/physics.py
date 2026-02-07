# src/physics.py

from __future__ import annotations

from typing import List

import torch

from utils import set_global_seed, second_derivative

__all__: list[str] = [
    "tise_residual",
    "tise_loss",
    "potential_smoothness_loss",
]

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def tise_residual(
    psi_model: torch.Tensor,
    V_model: torch.Tensor,
    E_model: torch.Tensor,
    dx: float,
    *,
    hbar: float | torch.Tensor = 1.0,
    mass: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Physics-informed residual loss for one eigenstate.

    Enforces:
        -(hbar^2/(2m))*psi''(x) + V(x)*psi(x) = E*psi(x)

    Parameters
    ----------
    psi_model :
        Eigenstate inferred from the model.
    V_model :
        Potential energy function inferred from the model.
    dx :
        spatial grid step size
    E_model :
        Learned energy eigenvalue corresponding to ``psi_model(x)``.
    hbar :
        Reduced Planck constant (atomic units by default).
    mass :
        Particle mass m.

    Returns
    -------
    Tensor
        TISE residual for one eigenstate.
    """
    psi_xx = second_derivative(psi_model, dx)
    return -0.5 * (hbar**2 / mass)  * psi_xx + V_model * E_model * psi_model

def tise_loss(
    multi_psi_model: List[torch.Tensor],
    V_model: torch.Tensor,
    energies: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Physics-informed loss for all eigenstates.

    Parameters
    ----------
    multi_psi_model :
        List of eigenstates.
    V_model :
    energies :
        List of energy eigenvalues (learned? 📝 I'm officially a little lost here ❓)
    dx :
        Spatial grid step size.

    Returns
    -------
    torch.Tensor
        Physics loss term from all eigenstates.
    """
    residuals = [
        tise_residual(psi_model, V_model, energies[i], dx)
        for i, psi_model in enumerate(multi_psi_model)
    ]
    return torch.mean(
        torch.stack([torch.mean(r**2) for r in residuals])
    )

def potential_smoothness_loss(
    V_model: torch.Tensor,
    dx: float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scale-aware smoothness regularization for the inferred potential V(x).

    Penalizes relative curvature with respect to the local magnitude of V(x).

        |V''(x)|^2 / (|V(x)|^2 + eps)

    Parameters
    ----------
    V_model :
        Callable mapping ``x -> V(x)``. (❓ is this correct? is it a stronger description than the other ones for V_model?)
    dx :
        Spatial grid step size.
    eps :
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        Smoothness penalty.
    """
    V_xx = second_derivative(V_model, dx)
    return torch.mean((V_xx**2) / (eps + V_model**2))


# ------------------------------------------------------------------------------
# 2️⃣ Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """❓‼️Need help with this part. would like to use the _run_smoke_test() method.‼️❓"""
    print("physics.py loaded")

if __name__ == "__main__":
    main()