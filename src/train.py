# src/train.py
"""
Training loop for the inverse Schrödinger problem.

The loss is a weighted sum of three physically motivated terms:
- **Physics loss**: enforces the TISE on the learned wavefunctions.
- **Wavefunction orthogonality loss**: enforces orthogonality of the learned wavefunctions.
- **Smoothness loss**: regularizes the potential to avoid spurious wiggles.
- **Data-fit loss**: matches learned quantities to observed densities/energies.

All three components live in separate modules (`physics`, `inverse`, `utils`). This file wires them together, handles the optimizer, and provides a minimal smoke-test that runs a single optimization step.

Author: Eigenscribe
Date: 02-2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from db_logger import RunLogger
from model import InverseSchrodingerModel
from physics import tise_loss, potential_smoothness_loss, energy_ordering_loss
from inverse import data_mismatch_loss
from utils import make_grid, set_global_seed, normalize_wavefunctions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs"

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ----------------------------------------------------------------------

def train_step(
    model: InverseSchrodingerModel,
    x: torch.Tensor,
    dx: float,
    rho_obs: List[torch.Tensor],
    E_obs: torch.Tensor,
    lambdas: dict[str, float],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Execute a single gradient descent step.

    Parameters
    ----------
    model :
        Instance of :class:`~model.InverseSchrodingerModel`.
    x: torch.Tensor, shape ``(n_points, 1)``
        Collocation points on which the PDE is enforced.
    dx : float
        Uniform grid spacing.
    rho_obs : List[torch.Tensor], each shape ``(n_states, n_points)``
        List of observed probability density tensors, one per eigenstate.
    E_obs : torch.Tensor, shape ``(n_states,)``
        Tensor of observed energies.
    lambdas : dict[str, float]
        Dictionary mapping loss identifies to scalar weights, e.g.
        ``{'data': 1.0, 'physics': 1.0, 'smooth': 1e-2, 'ordered': 1.0}``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        (total_loss, physics_loss, data_loss, smooth_loss, ordered_loss)
        -> ready for ``backward()``.
    """

    # ----- Forward pass -------------------------------------------------
    V_theta = model.V_theta(x)  # potential V(theta, x)
    psi_list = model.psi_theta(x, dx)  # list[psi_n(theta, x)]
    E_theta = model.E_theta()

    psi_list = normalize_wavefunctions(psi_list, dx)

    idx = torch.argsort(E_theta)
    E_theta = E_theta[idx]
    psi_list = [psi_list[i] for i in idx]

    # ------------------- Physics‑informed loss -------------------
    loss_physics = tise_loss(
        psi_list,  # list[psi_n(theta, x)]
        V_theta,  # V(theta, x)
        E_theta,  # E(theta)
        dx,
    )

    # ------------------- Smoothness regularizer -------------------
    loss_smooth = potential_smoothness_loss(V_theta, dx)

    # ------------------- Data‑fit loss ----------------------------
    loss_data = data_mismatch_loss(
        psi_list,  # learned psi_n(theta, x)
        E_theta,         # learned energies
        rho_obs,   # observed probability densities
        E_obs,     # observed energies
    )

    # ------------------- Orthogonalization penalty -------------------
    #loss_ortho = compute_orthogonality_loss(
    #    psi_list,
    #    dx,
    #)

    # ------------------- Energy ordering loss ----------------------------
    loss_ordered = energy_ordering_loss(E_theta)

    # ------------------- Weighted sum ----------------------------
    total_loss = (
            lambdas["data"] * loss_data
            + lambdas["physics"] * loss_physics
            + lambdas["smooth"] * loss_smooth
            + lambdas["ordered"] * loss_ordered
    )

    return total_loss, loss_physics, loss_data, loss_smooth, loss_ordered

def train_model(
    *,
    epochs: int | int = 6000,
    lr: float | float = 5e-3,
    hidden_dims: list[int] | list[int] = [64, 64],
    n_states: int | int = 3,
    seed: int | int = 27,
    log_db: str,
    artifacts_path: str | None = None,
) -> str:
    """
    Train the model, log metrics to SQLite, and save run artifacts.

    Returns
    -------
    str
        The generated run_id.
    """
    set_global_seed(seed)
    device = torch.device("cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    db_path = Path(log_db)
    if not db_path.is_absolute():
        db_path = DATA_DIR / db_path

    model = InverseSchrodingerModel(
        n_states=n_states,
        hidden_dims=hidden_dims,
        device=device,
        dx=0.01,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = make_grid(-5.0, 5.0, 128, device)
    dx = float(x[1] - x[0])
    rho_obs = [torch.exp(-x**2).squeeze() for _ in range(n_states)]
    E_obs = torch.arange(0.5, n_states + 0.5, 1.0, device=device)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "ordered": 1e-2}

    hyperparams = {
        "lr": lr,
        "epochs": epochs,
        "hidden_dims": hidden_dims,
        "n_states": n_states,
        "dx": dx,
        "lambdas": lambdas,
    }

    with RunLogger(db_path) as logger:
        run_id = logger.start_run(hyperparams=hyperparams, seed=seed)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            total_loss, loss_physics, loss_data, loss_smooth, loss_ordered = train_step(
                model, x, dx, rho_obs, E_obs, lambdas
            )
            total_loss.backward()
            optimizer.step()

            logger.log_metric(
                run_id=run_id,
                epoch=epoch,
                losses_dict={
                    "total_loss": total_loss.detach(),
                    "physics_loss": loss_physics.detach(),
                    "data_loss": loss_data.detach(),
                    "smooth_loss": loss_smooth.detach(),
                    "ordered_loss": loss_ordered.detach(),
                },
            )

        run_artifacts_dir = Path(artifacts_path) if artifacts_path is not None else ARTIFACTS_DIR / run_id
        if not run_artifacts_dir.is_absolute():
            run_artifacts_dir = (PROJECT_ROOT / run_artifacts_dir).resolve()
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), run_artifacts_dir / "model.pt")
        (run_artifacts_dir / "config.json").write_text(
            json.dumps(hyperparams, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_artifacts_dir / "run_id.txt").write_text(run_id, encoding="utf-8")

        logger.update_artifacts_path(run_id, run_artifacts_dir)

    run_id_file = DATA_DIR / "run_id.txt"
    run_id_file.write_text(run_id, encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Saved run ID to: {run_id_file.resolve()}")
    print(f"SQLite DB: {db_path.resolve()}")
    print(f"Artifacts: {run_artifacts_dir}")

    return run_id
# ----------------------------------------------------------------------
# 2️⃣ Smoke‑test entry point
# ----------------------------------------------------------------------
def _run_train_smoke_test() -> None:
    """
    Runs a *single* optimization step on synthetic data.
    Useful for CI pipelines or quick sanity checks inside a notebook.
    """
    set_global_seed(27)

    device = torch.device("cpu")
    model = InverseSchrodingerModel(
        n_states=3,
        hidden_dims=[64, 64],
        device=device,
        dx=0.01,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Spatial grid
    x = make_grid(-5.0, 5.0, 128, device)
    dx = float(x[1] - x[0])

    # Synthetic observations
    rho_obs = [torch.exp(-x**2).squeeze() for _ in range(3)]
    E_obs = torch.tensor([0.5, 1.5, 2.5], device=device)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "ordered": 1e-2}

    optimizer.zero_grad()
    total_loss, loss_physics, loss_data, loss_smooth, loss_ordered = train_step(model, x, dx, rho_obs, E_obs, lambdas)
    total_loss.backward()
    optimizer.step()

    print("✔️ train.py smoke test - loss after one step:", float(total_loss.detach()))

def main() -> None:
    """Entry point for ``python -m src.train`` -> runs the smoke test."""
    _run_train_smoke_test()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--log-db", type=str, default="training_runs.db")
    parser.add_argument("--artifacts-path", type=str, default=None)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        n_states=args.n_states,
        seed=args.seed,
        log_db=args.log_db,
        artifacts_path=args.artifacts_path,
    )

if __name__ == "__main__":
    main()