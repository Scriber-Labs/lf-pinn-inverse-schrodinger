# src/train.py

from __future__ import annotations

import torch
from model import InverseSchrodingerModel
# 🔮🔮🔮🔮 CONTINUE HERE! 🔮🔮🔮🔮🔮🔮🔮🔮🔮🔮

def train_inverse(
    *,
    psi_models: list[nn.Module],
    V_model: nn.Module,
    x_phys: Tensor,
    energies: list[Tensor],
    energies_obs: list[Tensor] | None = None,
    n_epochs: int = 2_000,
    lr: float = 1e-3,
) -> None:
    """
    Minimal training loop for the inverse Schrödinger problem.
    Parameters
    ----------
    psi_models : list[nn.Module]
        Callable mapping ``x -> psi_n(x)``.
    V_model : nn.Module
        Callable mapping ``x -> V(x)``.
    x_phys : Tensor
        Collocation points for enforcing the TISE.
    energies : list[Tensor]
        Energy eigenvalues E_n.
    energies_obs : list[Tensor] | None
        Observed energy eigenvalues (from spectral data).
    n_epochs : int, default = 2_000
        Number of training epochs.
    lr : float, default = 1e-3
        Learning rate for the Adam optimizer.
    """
    set_global_seed(42)

    params = list(V_model.parameters())
    for psi in psi_models:
        params.extend(psi.parameters())

    optimizer = optim.Adam(params, lr=lr)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        loss: Tensor = inverse_multi_state_loss(
            psi_models=psi_models,
            energies=energies,
            V_model=V_model,
            x_phys=x_phys,
            energies_obs=energies_obs,
            lambda_phys=1.0,
            lambda_energy=1.0,
            lambda_density=1.0,
            lambda_smooth=1e-4,
        )

        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------
        # Physics-aware logging
        # ------------------------------------------------------------
        if epoch % 100 == 0 or epoch == 1:
            with torch.no_grad():
                V_vals = V_model(x_phys).squeeze()
                dx = (x_phys[1] - x_phys[0]).item()
                d2V = torch.gradient(torch.gradient(V_vals, spacing=(dx,))[0], spacing=(dx,))[0]
                curvature = torch.mean(torch.abs(d2V))

            print(
                f"[epoch {epoch:04d}] "
                f"total loss = {loss.item():.3e} | "
                f"(|V''(x)|) is approximately {curvature:.3e}"
            )
