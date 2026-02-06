from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, autograd

from utils import set_global_seed

__all__: list[str] = [
    "tise_residual_loss",
    "potential_smoothness_loss",
]


# ------------------------------------------------------------------------------
# 0️⃣ Typing helpers
# ------------------------------------------------------------------------------
class SupportsForward(Protocol):
    """
    Protocol for any callable mapping, coordinates -> variables.

    Compatible with:
    - torch.nn.Module
    - lambda functions
    - functional wrappers
    """

    def __call__(self, x: Tensor) -> Tensor:
        ...


# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def tise_residual_loss(
    psi_model: SupportsForward,
    V_model: SupportsForward,
    x: Tensor,
    *,
    energy: float | Tensor,
    hbar: float | Tensor = 1.0,
    mass: float | Tensor = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Physics-informed residual loss for the 1D TISE.

    Enforces:
        -(hbar^2/(2m))*psi''(x) + V(x)*psi(x) = E*psi(x)

    Parameters
    ----------
    psi_model :
        Callable mapping ``x -> psi(x)``.
    V_model :
        Callable mapping ``x -> V(x)``.
    x :
        Input tensor of shape ``(N_x, 1)``
    energy :
        Energy eigenvalue E.
    hbar :
        Reduced Planck constant (atomic units by default).
    mass :
        Particle mass m.
    reduction :
        "mean", "sum", or "none".

    Returns
    -------
    Tensor
        Scalar loss or pointwise residuals.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"`x` must be a torch.Tensor, got {type(x)}.")

    x = x.detach().requires_grad_(True)

    psi: Tensor = psi_model(x)
    V: Tensor = V_model(x)

    dpsi_dx: Tensor = _grad(psi, x)
    d2psi_dx2: Tensor = _grad(dpsi_dx, x)

    residual: Tensor = (
        -(hbar ** 2) / (2.0 * mass) * d2psi_dx2 + V * psi - energy * psi
    )

    pointwise_loss: Tensor = residual.pow(2)

    if reduction == "mean":
        return pointwise_loss.mean()
    elif reduction == "sum":
        return pointwise_loss.sum()
    elif reduction == "none":
        return pointwise_loss
    else:
        raise TypeError(f"`reduction` must be 'mean' or 'sum' or 'none', got {reduction}.")

def potential_smoothness_loss(
    V_model: SupportsForward,
    x: Tensor,
    *,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    """
    Scale-aware smoothness regularization for the inferred potential V(x).

    Penalizes relative curvature with respect to the local magnitude of V(x).

        |V''(x)|^2 / (|V(x)|^2 + eps)

    Parameters
    ----------
    V_model :
        Callable mapping ``x -> V(x)``.
    x :
        Tensor of shape ``(N_x, 1)``.
    eps :
        Small constant for numerical stability.
    reduction :
        "mean", "sum", or "none".

    Returns
    -------
    Tensor
        Smoothness penalty.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"`x` must be a torch.Tensor, got {type(x)}.")

    x = x.detach().requires_grad_(True)

    V: Tensor = V_model(x)
    dV_dx: Tensor = _grad(V, x)
    d2V_dx2: Tensor = _grad(dV_dx, x)

    pointwise_loss: Tensor = d2V_dx2.pow(2) / (V.pow(2) + eps)

    if reduction == "mean":
        return pointwise_loss.mean()
    elif reduction == "sum":
        return pointwise_loss.sum()
    elif reduction == "none":
        return pointwise_loss
    else:
        raise TypeError(f"`reduction` must be 'mean', 'sum', or 'none', got {reduction}.")

def density_mismatch_loss(
    psi_model: SupportsForward,
    x: Tensor,
    rho_obs: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """
    Probability density mismatch loss.

    Penalize deviation between predicted |psi(x)^2|^2 and observed probability density rho_obs(x).

        || |psi(x)|^2 - rho_obs(x) ||^2

    Parameters
    ----------
    psi_model :
        Callable mapping ``x -> psi(x)``.
    x :
        Coordinates of density observations.
    rho_obs :
        Observed probability density values.
    reduction :
        "mean", "sum", or "none".

    Returns
    -------
    Tensor
        Density mismatch loss.
    """
    psi: Tensor = psi_model(x)
    rho_pred: Tensor = psi.pow(2)

    pointwise_loss: Tensor = (rho_pred - rho_obs).pow(2)

    if reduction == "mean":
        return pointwise_loss.mean()
    elif reduction == "sum":
        return pointwise_loss.sum()
    elif reduction == "none":
        return pointwise_loss
    else:
        raise TypeError(f"`reduction` must be 'mean', 'sum', or `none`, got {reduction}.")
# ------------------------------------------------------------------------------
# 2️⃣ Private Helpers
# ------------------------------------------------------------------------------

def _grad(y: Tensor, x: Tensor) -> Tensor:
    """
    Compute dy/dx using torch.autograd while preserving the graph.
    """
    (dy_dx,) = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True,
    )
    if dy_dx is None:
        return torch.zeros_like(x)

    return dy_dx

# ------------------------------------------------------------------------------
# 3️⃣ Smoke test helpers
# ------------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Sanity check for Schrödinger residual and smoothness losses."""
    set_global_seed(42)

    psi_model = lambda x: torch.sin(torch.pi * x)     # smooth test function (PIB eigenmode)
    # psi_model = lambda x: torch.exp(-0.5 * x**2)    # HO ground-state shape (unnormalized)
    V_model = lambda x: 0.1 * x**2                    # pure quadratic (harmonic) potential

    x: Tensor = torch.linspace(0.0, 1.0, 50).unsqueeze(1)
    energy: float = (torch.pi ** 2) / 2.0

    L_res: Tensor = tise_residual_loss(
        psi_model,
        V_model,
        x,
        energy=energy,
    )

    L_smooth: Tensor = potential_smoothness_loss(
        V_model,
        x,
    )

    print("✔️ Physics losses computed successfully.")
    print(f"TISE residual loss    : {L_res.item():.3e}")
    print(f"Potential smoothness  : {L_smooth.item():.3e}")

# ------------------------------------------------------------------------------
# 4️⃣ Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """Run local tests when executed as a script."""
    run_smoke_test()


if __name__ == "__main__":
    main()