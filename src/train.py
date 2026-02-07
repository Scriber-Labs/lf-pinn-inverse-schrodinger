# src/train.py

from __future__ import annotations

import torch
from model import InverseSchrodingerModel
from physics import tise_loss, potential_smoothness_loss
from inverse import data_mismatch_loss
from utils import make_grid, set_global_seed

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------

def train_step(
    model: InverseSchrodingerModel,
    x: torch.Tensor,
    dx: float,
    rho_obs: torch.Tensor,
    E_obs: torch.Tensor,
    lambdas: list[float],   # ❓is this type annotation correct?❓
) -> torch.Tensor:
    """
    Training the Inverse Schrodinger MLP model.

    Parameters
    ----------
    model :
        Inverse Schrodinger MLP model.
    x :
        Collocation points for enforcing the TISE.
    dx :
        Spatial grid step size.
    rho_obs :
        Observed probability density.
    E_obs :
        Observed energy eigenvalues. (❓Is this correct?)
    lambdas :
        Loss coefficients.

    Returns
    -------
    torch.Tensor :
        Total loss.
    """

    V_model = model.potential(x)    # is V_model the correct naming of this parameter or should it be regular V? I just want to make sure this is inferred from learned energy eigenvalues (❓again, I don't know if i'm getting the 'story' entirely correct. make sure all comments like these are reconciled!❓)
    multi_psi_model =  model.psi(x)

    L_physics = tise_loss(multi_psi_model, V_model, model.energies, dx)
    L_smooth = potential_smoothness_loss(V_model, dx)
    L_data = data_mismatch_loss(multi_psi_model, model.energies, rho_obs, E_obs)

    return (
        lambdas["data"] * L_data
        + lambdas["physics"] * L_physics
        + lambdas["smooth"] * L_smooth
    )

# ------------------------------------------------------------------------------
# 2️⃣ Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """❓‼️Need help with this part. would like to use the _run_smoke_test() method.‼️❓"""
    set_global_seed(42)
    device = torch.device("cpu")

    model = InverseSchrodingerModel(3, [64, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = make_grid(-5.0, 5.0, 256, device)
    dx = float(x[1] - x[0])

    # Placeholder/synthetic/fake data
    rho_obs = [torch.exp(-x**2) for _ in range(3)]
    E_obs = torch.tensor([0.5, 1.5, 2.5])

    lambdas = dict(data=1.0, physics=1.0, smooth=1e-2)

    for epoch in range(5000):
        optimizer.zero_grad()
        loss = train_step(model, x, dx, rho_obs, E_obs, lambdas)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(epoch, float(loss))

if __name__ == "__main__":
    main()