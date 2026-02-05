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
    reduction: str = "mean",
) -> Tensor:
    """
    Smoothness regularization for the inferred potential V(x).

    Penalizes large curvature via:
        mean |V''(x)|^2

    Parameters
    ----------
    V_model :
        Callable mapping ``x -> V(x)``.
    x :
        Tensor of shape ``(N_x, 1)``.
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

    pointwise_loss: Tensor = d2V_dx2.pow(2)

    if reduction == "mean":
        return pointwise_loss.mean()
    elif reduction == "sum":
        return pointwise_loss.sum()
    elif reduction == "none":
        return pointwise_loss
    else:
        raise TypeError(f"`reduction` must be 'mean' or 'sum' or 'none', got {reduction}.")

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
    )
    return dy_dx

# ------------------------------------------------------------------------------
# 3️⃣ Smoke test helpers
# ------------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Sanity check for Schrödinger residual and smoothness losses."""
    set_global_seed(42)     # ‼️switched from `torch.manual_seed(0)` ; idk if this was the correct thing to do

    psi_model = lambda x: torch.sin(torch.pi * x)
    V_model = lambda x: torch.zeros_like(x)

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