# cli_train.py
"""
Command-line interface for the inverse Schrödinger training routine (src/train.py).

Usage examples
--------------
$ python -m cli_train --hidden 128 --epochs 5000 --lr 5e-4 --device cuda
$ cli-trian --hidden 64 --epochs 3000       # if installed as a console script
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import torch

torch.set_default_dtype(torch.float64)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "runs"

# ----------------------------------------------------------------------
# 1️⃣ Import the core training utilities from src/train.py
# ----------------------------------------------------------------------
from db_logger import RunLogger
from train import (
    train_step,
    set_global_seed,
    make_grid,
    InverseSchrodingerModel,
)

# ----------------------------------------------------------------------
# 2️⃣ Argument parser for CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cli_train",
        description="Train a low-fidelity PINN for the inverse Schrödinger problem.",
    )
    parser.add_argument("--n_modes", type=int, default=3,
                        help="Number of eigenmodes and associated energy eigenvalues to learn")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Number of neurons per hidden layer")
    parser.add_argument("--epochs", type=int, default=6_000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument("--n_points", type=int, default=256,
                        help="Number of spatial collocation points (i.e., grid resolution)")
    parser.add_argument("--seed", type=int, default=27,
                        help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu | cuda | cuda:0 | ...",
    )
    parser.add_argument("--log_every", type=int, default=800,
                        help="Print interval every x epochs")
    parser.add_argument(
        "--log-db",
        type=str,
        default=str(DATA_DIR / "training_runs.db"),
        help="Path to the SQLite training database",
    )

    return parser.parse_args(argv)

# ----------------------------------------------------------------------
# 3️⃣ Helper to build the model & synthetic data
# ----------------------------------------------------------------------
def _build_problem(args: argparse.Namespace):
    """
    Build the model and synthetic data for the quantum harmonic oscillator (similar to a smoke test).

    Parameters
    ----------
    args: all arguments entered by the user

    Returns
    -------
        model, optimizer, x (grid), dx, rho_obs, E_obs, lambdas
    """
    device: Final = torch.device(args.device)

    # Spatial grid
    x = make_grid(-5.0, 5.0, args.n_points, device=device)
    dx = float(x[1] - x[0])

    # Analytic eigenfunctions (Hermite-Gaussians) for the chosen analytic potential (harmonic oscillator)
    def hermite_gauss(n: int, x_vals: torch.Tensor) -> torch.Tensor:
        """Return the nth normalized harmonic oscillator eigenfunction."""
        from math import factorial, sqrt, pi
        import scipy.special as sp
        norm = 1.0 / sqrt(2.0 ** n * factorial(n)) * (pi ** -0.25)
        # sp.hermite returns a callable polynomial; evaluate on x_vals
        return norm * sp.hermite(n)(x_vals) * torch.exp(-0.5 * x_vals ** 2)

    # Synthetic observations (replace these with real data as you see fit)
    psi_true = [hermite_gauss(n, x.squeeze()) for n in range(args.n_modes)]
    E_true = torch.arange(args.n_modes,
                          dtype=torch.float64) + 0.5  # exact energies for the quantum harmonic oscillator (hbar = omega = 1)

    rho_obs = [psi ** 2 + 0.02 * torch.randn_like(psi) for psi in psi_true]
    E_obs = E_true + 0.05 * torch.randn_like(E_true)

    # Loss weights (feel free to add these to CLI arguments)
    lambdas = {
        "data": 1.5,
        "physics": 0.5,
        "smooth": 0.25,
        "ordered": 1.0,
    }

    # Model
    model = InverseSchrodingerModel(
        n_states=args.n_modes,
        hidden_dims=[args.hidden, args.hidden],
        device=device,
        dx=dx,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, optimizer, x, dx, rho_obs, E_obs, lambdas

# ------------------------------------------------------------------------------
# 3️⃣ Entry-point
# ------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:    # noqa: D401
    """
    Entry point for ``python -m cli_train``.
    """
    args = parse_args(argv)

    # Seed everything for reproducibility
    set_global_seed(args.seed, deterministic=True)

    device: Final = torch.device(args.device)
    print(f"🖥️  Using device: {device}")

    # Build model, optimizer, data, etc.
    model, optimizer, x, dx, rho_obs, E_obs, lambdas = _build_problem(args)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.log_db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    hyperparams = {
        "n_modes": args.n_modes,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_points": args.n_points,
        "seed": args.seed,
        "device": args.device,
        "log_every": args.log_every,
        "lambdas": lambdas,
        "dx": dx,
    }

    print(f"dx: {dx}")
    print(f"x dtype: {x.dtype}")
    print(f"E_obs dtype: {E_obs.dtype}")
    print(f"rho_obs[0] dtype: {rho_obs[0].dtype}")
    print(f"model param dtype: {next(model.parameters()).dtype}")

    with RunLogger(db_path) as logger:
        run_id = logger.start_run(hyperparams=hyperparams, seed=args.seed)
        run_artifacts_dir = ARTIFACTS_DIR / run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Training loop (simple version – prints every `log_every` steps)
        # ------------------------------------------------------------------
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad()
            total_loss, loss_physics, loss_data, loss_smooth, loss_ordered = train_step(model, x, dx, rho_obs, E_obs, lambdas)
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

            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"[{epoch:>5}/{args.epochs}] loss = {total_loss.item():.6e}")

        torch.save(model.state_dict(), run_artifacts_dir / "model.pt")
        (run_artifacts_dir / "config.json").write_text(
            json.dumps(hyperparams, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_artifacts_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
        logger.update_artifacts_path(run_id, run_artifacts_dir)

    (DATA_DIR / "run_id.txt").write_text(run_id, encoding="utf-8")

    print("✅ Training finished. Final loss:", f"{total_loss.item():.6e}")
    print(f"Run ID: {run_id}")
    print(f"Artifacts saved to: {run_artifacts_dir.resolve()}")

# ------------------------------------------------------------------------------
# 4️⃣ Script execution guard
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Pass through the real command line (skip the script name)
    main(sys.argv[1:])