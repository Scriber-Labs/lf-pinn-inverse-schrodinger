# cli_train.py
"""
Command-line interface for the inverse Schrödinger training routine (src/train.py).

Usage examples
--------------
$ python -m cli_train --hidden 128 --epochs 5000 --lr 5e4 --device cuda
$ cli-trian --hidden 64 --epochs 3000       # if installed as a console script
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import torch

# ----------------------------------------------------------------------
# 1️⃣ Import the core training utilities from src/train.py
# ----------------------------------------------------------------------
from src.train import (
    train_step,
    set_global_seed,            # re-exported from src/utils.py
    make_grid,                  # re-exported in src/utils.py
    InverseSchrodingerModel,    # re-exported in src/model.py (the model class)
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
    parser.add_argument("--epochs", type=int, default=3000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--n_points", type=int, default=200,
                        help="Number of spatial collocation points (i.e., grid resolution)")
    parser.add_argument("--seed", type=int, default=27,
                        help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu | cuda | cuda:0 | ...",
    )
    parser.add_argument("--log-every", type=int, default=500,
                        help="Print interval every x epochs")
    return parser.parse_args(argv)

# ----------------------------------------------------------------------
# 3️⃣ Helper to build the model & synthetic data
# ----------------------------------------------------------------------
def _build_problem(args: argparse.Namespace):
    """
    Build the model and synthetic data (similar to a smoke test).

    Parameters
    ----------
    args: all arguments entered by the user

    Returns
    -------
        model, optimizer, x (grid), dx, rho_obs, E_obs, lambdas
    """
    device: Final = torch.device(args.device)

    # Model
    model = InverseSchrodingerModel(
        n_states=args.n_modes,
        hidden_dims=[args.hidden, args.hidden],
        device=device,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Spatial grid
    x = make_grid(-5.0, 5.0, args.n_points, device=device)
    dx = float(x[1] - x[0])

    # Synthetic observations (replace these with real data as you see fit)
    rho_obs = [torch.exp(-x**2).squeeze() for _ in range(args.n_modes)]
    E_obs = torch.arange(start=0.5, end=0.5+0.5*args.n_modes, step=0.5)

    # Loss weights (feel free to add these to CLI arguments)
    lambdas = {
        "data": 1.0,
        "physics": 4.0,
        "smooth": 5e-4,
        "norm": 1.0,
    }

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
    print(f"🖥️ Using device: {device}")

    # Build model, optimizer, data, etc.
    model, optimizer, x, dx, rho_obs, E_obs, lambdas = _build_problem(args)

    # ------------------------------------------------------------------
    # Training loop (simple version – prints every `log_every` steps)
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss = train_step(model, x, dx, rho_obs, E_obs, lambdas)
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[{epoch:>5}/{args.epochs}] loss = {loss.item():.6e}")

    print("✅ Training finished. Final loss:", f"{loss.item():.6e}")

# ------------------------------------------------------------------------------
# 4️⃣ Script execution guard
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Pass through the real command line (skip the script name)
    main(sys.argv[1:])