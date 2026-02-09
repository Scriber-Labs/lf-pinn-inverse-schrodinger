# src/physics.py
"""
Physics-informed loss utilities for the inverse TISE problem.

All functions operate on *already-evaluated* tensors (i.e., the outputs of the neural networks defined by ``model.InverseSchrodingerModel``.

Learned functions:
- A potential function: ``V_theta(x)``
- A wavefunction for state *n*: ``psi_theta_n(x)``
- Energy eigenvalues for state *n*: ``E_theta_n``

These helpers are deliberately coded to be thin wrappers (❓) around the generic utilities in ``utils.py`` (e.g., ``second_derivative``) so that the physics stays explict and can be easily audited.
"""

from __future__ import annotations

from typing import List

import torch

from utils import second_derivative

__all__: list[str] = [
    "tise_residual",
    "tise_loss",
    "potential_smoothness_loss",
]

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ----------------------------------------------------------------------

def tise_residual(
    psi_theta: torch.Tensor,
    V_theta: torch.Tensor,
    E_theta: torch.Tensor,
    dx: float,
    *,
    hbar: float | torch.Tensor = 1.0,   # atomic units
    mass: float | torch.Tensor = 1.0,   # non-dimensionalization
) -> torch.Tensor:
    """
    Residual of the time-independent Schrödinger equation for a *single* eigenstate.

    Parameters
    ----------
    psi_theta : torch.Tensor, shape ``(N, 1)``
        Tensor ``psi(x)`` for one eigenstate.
    V_theta : torch.Tensor, shape ``(N, 1)``
        Tensor ``V(x)`` for a potential function.
    E_theta : torch.Tensor
        Scalar energy eigenvalue for this eigenstate. Either a zero dimensional tensor or a Python float.
    dx : float
        Uniform grid spacing.
    hbar, mass: float, default: 1.0
        Physical constants (default to atomic units) and mass (default to nondimensional value).

    Returns
    -------
    torch.Tensor, shape ``(N, 1)``
        The pointwise residual ``R(x)``. Note that a perfect solution gives ``R = 0`` everywhere.
    """
    # Second derivative of psi using the central-difference helper from utils.
    psi_xx = second_derivative(psi_theta, dx)

    # Assemble the residual term-by-term.
    kinetic = -0.5 * (hbar**2 / mass) * psi_xx
    potential =V_theta * psi_theta
    rhs = E_theta * psi_theta

    return kinetic + potential - rhs

def tise_loss(
    multi_psi_theta: List[torch.Tensor],
    V_theta: torch.Tensor,
    energies_theta: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Aggregate physics loss over *all* learned eigenstates.

    Parameters
    ----------
    multi_psi_theta : List[torch.Tensor]
        List of the wavefunction tnesors ``[psi_theta_0, psi_theta_1, ... ]``.
    V_theta : torch.Tensor
        Potential function tensor (shared across learned eigenstates).
    energies_theta : torch.Tensor
        Tensor of learned energy eigenvalues ``[E_theta_0, E_theta_1, ... ]``.
    dx : float
        Grid spacing.

    Returns
    -------
    torch.Tensor (scalar)
        Mean-squared TISE residual across all eigenstates.
    """
    # Compute a residual for each eigenstate.
    residuals = [
        tise_residual(psi, V_theta, energies_theta[i], dx) for i, psi in enumerate(multi_psi_theta)
    ]

    # Square, average per-state, then average across all states.
    per_state_mse = [torch.mean(r**2) for r in residuals]

    return torch.mean(torch.stack(per_state_mse))

def potential_smoothness_loss(
    V_theta: torch.Tensor,
    dx: float,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scale-aware smoothness regularization for the learned potential.

    Large curvature relative to the local magnitude of the potential is penalized.
    Parameters
    ----------
    V_theta
    dx
    eps

    Returns
    -------

    """