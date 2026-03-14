# src/orthogonality_loss.py
"""
Orthogonality loss utilities for quantum wavefunctions.

These functions compute the loss term that penalizes non-orthogonal states.
The goal is to minimize the overlap between distinct wavefunctions in the basis set.
Functions expect pre-computed tensors and do not call the model itself.

- ``psi_list``: List of learned wavefunction tensors
- ``dx``: Discretization step size for integration
"""

from __future__ import annotations

from typing import List

import torch

__all__: list[str] = [
    "compute_orthogonality_loss",
]

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ----------------------------------------------------------------------

def compute_orthogonality_loss(
    psi_list: List[torch.Tensor],
    dx: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute the orthogonality loss for a list of wavefunctions.

    This calculates the sum of squared overlaps between all unique pairs of wavefunctions. A loss of 0.0 corresponds with perfect orthogonality.

    Parameters
    ----------
    psi_list : List[torch.Tensor]
        List of learned wavefunction tensors, each with shape (n_points, 1) or (n_points,).
    dx : float
        Discretization step size for the numerical integration.
    dtype : torch.dtype, optional (default: torch.float32)
        Data type.
    Raises
    -------
    ValueError
        If the list contains fewer than 2 wavefunctions.
    """
    if len(psi_list) < 2:
        raise ValueError("✖️ Orthogonality loss requires at least 2 wavefunctions.")

    loss: torch.Tensor = torch.tensor(0.0, dtype=dtype)
    n_states: int = len(psi_list)

    # Iterate over unique pairs (i, j) where j > i
    for i in range(n_states):
        for j in range(i + 1, n_states):
            # Calculate overlap integral
            overlap: torch.Tensor = torch.sum(psi_list[i] * psi_list[j]) * dx

            # Accumulate squared overlap
            loss += overlap ** 2

    return loss

# ----------------------------------------------------------------------
# 2️⃣ Smoke-test entry point
# ----------------------------------------------------------------------
def _run_orthogonality_smoke_test() -> None:
    """
    Minimal sanity check that the loss term behaves as expected.

    Tests:
    1. Perfectly orthogonal states (sin/cos) => expected results are near-zero loss.
    2. Identical states => expected results are a high loss.
    """
    # Set seed for reproducibility
    torch.manual_seed(27)

    # Define spatial grid
    x: torch.Tensor = torch.linspace(-1.0, 1.0, 100)
    dx: float = x[1] - x[0]

    # Case 1: Orthogonal states (sin and cos on symmetric interval)
    psi_ortho_1: torch.Tensor = torch.sin(x)
    psi_ortho_2: torch.Tensor = torch.cos(2 * x)

    # Case 2: Identical states
    psi_same_1: torch.Tensor = torch.sin(x)
    psi_same_2: torch.Tensor = torch.sin(x)

    # Test 1: Orthogonal pair
    loss_ortho: torch.Tensor = compute_orthogonality_loss([psi_ortho_1, psi_ortho_2], dx)

    # Test 2: Identical pair
    loss_same: torch.Tensor = compute_orthogonality_loss([psi_same_1, psi_same_2], dx)

    print("✔️ `src/orthogonality_loss.py` smoke test:`")
    print(f"   Loss (Orthogonal pair): {loss_ortho.item():.6f}")
    print(f"   Loss (Identical pair): {loss_same.item():.6f}")

    # Validation check
    if loss_ortho.item() < 0.1:
        print("    ✅ Orthogonality check passed.")
    else:
        print("    ⚠️ Warning: Orthogonality check failed (loss too high).")

def main() -> None:
    """
    Entry point for ``python src/orthogonality_loss.py`` -> runs the minimal smoke test.
    """
    _run_orthogonality_smoke_test()

if __name__ == "__main__":
    main()
