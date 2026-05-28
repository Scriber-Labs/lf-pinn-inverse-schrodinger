# src/physics.py
"""
Physics-informed loss utilities for the inverse TISE problem.

All functions operate on *already-evaluated* tensors (i.e., the outputs of the neural networks defined by ``model.InverseSchrodingerModel``.

Learned functions:
- A potential function: ``V_theta(x)``
- A wavefunction for state *n*: ``psi_theta_n(x)``
- Energy eigenvalues for state *n*: ``E_theta_n``

These helpers are deliberately coded to be thin wrappers around the generic utilities in ``utils.py`` (e.g., ``second_derivative``) so that the physics stays explict and can be easily audited.

Author: Eigenscribe
Development note: LLM assistance was used during construction; implmentation has been reviewed and adapted for this project.
Review status: Reviewed and maintained by Eigenscribe.
Date: 02-2026
"""

from __future__ import annotations

from typing import List

import torch

from utils import second_derivative

__all__: list[str] = [
    "tise_residual",
    "tise_loss",
    "potential_smoothness_loss",
    "energy_ordering_loss",
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
    potential = V_theta * psi_theta
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
        List of the wavefunction tensors ``[psi_theta_0, psi_theta_1, ... ]``.
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
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale-aware smoothness regularization for the learned potential.

    Large curvature relative to the local magnitude of the potential is penalized.
    Parameters
    ----------
    V_theta : torch.Tensor, shape ``(N, 1)``
        Potential function tensor ``V(x)``.
    dx : float
        Grid spacing.
    eps : float, default: 1e-6
        Small constant to avoid dividing by zero.

    Returns
    -------
    torch.Tensor (scalar)
        Smoothness penalty.
    """
    V_xx = second_derivative(V_theta, dx)
    return torch.mean((V_xx**2) / (eps + V_theta**2))

def energy_ordering_loss(energies: torch.Tensor) -> torch.Tensor:
    """
    Enforces the physical requirement that energy eigenvalues are strictly ordered: E_0 < E_1 < ...

    This prevents the "state swapping" issue where networks lose their identity during training because the optimizer identifies gradients flowing through different indices that depend on the current iteration energy ordering.

    Parameters
    ----------
    energies : torch.Tensor, shape ``(n_states,)``
        Tensor of learned energy eigenvalues.

    Returns
    -------
    torch.Tensor (scalar)
        Penalty loss. Zero if strictly ordered, positive if otherwise.
    """
    if len(energies) < 2:
        return torch.tensor(0.0, device=energies.device, dtype=energies.dtype)

    # Calculate differences between adjacent sorted energies
    # We want E[i] < E[i+1] => E[i] - E[i+1] < 0
    # We penalize positive violations using ReLU: max(0, E[i] - E[i+1])
    diffs = energies[:-1] - energies[1:]

    # Apply ReLU to penalize only violations (where diff > 0)
    violations = torch.relu(diffs)

    # Return mean squared violation
    return torch.mean(violations ** 2)

# ----------------------------------------------------------------------
# 2️⃣ Smoke‑test entry point
# ----------------------------------------------------------------------

def _run_physics_smoke_test() -> None:
    """
    Minimal sanity check that the loss functions accept realistic shapes.

    Builds a dummy spatial grid, feeds random tensors through the loss utilities, and prints the resulting scalars.  Since this is purely a shape-check, no gradients are taken into account.
    """
    from utils import make_grid, set_global_seed

    set_global_seed(27)

    # Dummy grid (100 points from -1 to 1)
    x = make_grid(-1.0, 1.0, 100)
    dx = float(x[1] - x[0])

    # Random but well-shaped tensors
    V = torch.sin(x)                    # potential
    psi0 = torch.cos(x)                 # ground-state guess
    psi1 = torch.sin(2 * x)             # first excited state guess
    energies = torch.tensor([0.5, 1.5]) # arbitrary energies

    # Run each loss
    res0 = tise_residual(psi0, V, energies[0], dx)
    loss_tise = tise_loss([psi0, psi1], V, energies, dx)
    loss_smooth = potential_smoothness_loss(V, dx)

    # Test energy ordering loss
    # Case 1: Correctly ordered
    E_ordered = torch.tensor([0.5, 1.5, 2.5])
    loss_ordered = energy_ordering_loss(E_ordered)

    # Case 2: Violated ordering
    E_violated = torch.tensor([2.5, 0.5, 1.5])
    loss_violated = energy_ordering_loss(E_violated)

    print("✔️ physics.py smoke test:")
    print(f"  residual shape     : {res0.shape}")
    print(f"  tise loss          : {loss_tise.item():.6f}")
    print(f"  smoothness loss    : {loss_smooth.item():.6f}")
    print(f"  loss (ordered)     : {loss_ordered.item():.6f}")
    print(f"  loss (violated)    : {loss_violated.item():.6f}")

    # Verification
    assert loss_ordered.item() < 1e-6, "Ordered energies should have near-zero loss."
    assert loss_violated.item() > 0.0, "Violated ordering should have positive loss"
    print("   ✅ Energy ordering logic passed.")

def main() -> None:
    """
    Entry-point for ``python src/physics.py`` -> runs the minimal smoke-test.
    """
    # Use UTF-8 for output to support emojis on Windows
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    _run_physics_smoke_test()

if __name__ == "__main__":
    main()
