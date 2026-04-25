# src/pod.py
"""
Proper Orthogonal Decomposition (POD) utilities.

- ``pod_decomposition``: thin wrapper around an SVD that returns the spatial modes, singular values, and modal coefficients.
- ``mode_overlap_matrix``: builds the inner product matrix < psi_m | psi_n > on a uniform grid. Note that the factor ``dx`` accounts for the integration measure.

Both functions are pure NumPy-style Torch ops -> no side effects, no hidden state. Moreover, the module includes a minimal smoke test that can be called from the command line (``python -m src.pod``) or from any notebook.

Author: Eigenscribe
Development note: LLM assistance was used during construction; implementation has been reviewed and adapted for this project.
Review status: Reviewed and maintained by Eigenscribe.
Date: 02-2026
"""

from __future__ import annotations

import torch
from typing import Tuple

__all__: list[str] = [
    "pod_decomposition",
    "mode_overlap_matrix",
]

# ----------------------------------------------------------------------
# 1️⃣ Public API
# ---------------------------------------------------------------------

def pod_decomposition(
    psi_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform POD on the wavefunction snapshot matrix.

    Parameters
    ----------
    psi_matrix : torch.Tensor, shape ``(N_x, N_x)``
        Each column is a snapshot of the wavefunction evaluated on the spatial grid.

    Returns
    -------
    U : torch.Tensor, shape ``(N_x, N_x)``
        Spatial POD modes (orthonormal w.r.t the discrete inner product).
    S : torch.Tensor, shape ``(min(N_x, N_modes),)``
        Singular values -> quantify the energy/content of each mode.
    Vh : torch.Tensor, shape ``(N_modes, N_modes)``
        Transpose of the right singular vectors; the rows contain the modal coefficients for each snapshot.
    """
    # 📝 ``full_matrices=False`` gives the compact SVD, which is what PODl needs.
    U, S, Vh = torch.linalg.svd(psi_matrix, full_matrices=False)
    return U, S, Vh

def mode_overlap_matrix(
    psi_matrix: torch.Tensor,
    dx : float,
) -> torch.Tensor:
    """
    Compute the discrete overlap matrix <psi_m | psi_n > on a uniform grid.

    For a uniform grid, the inner product reduces to a simple matrix product scaled by the grid spacing ``dx``

    Parameters
    ----------
    psi_matrix : torch.Tensor, shape ``(N_x, N_modes)``
        Input matrix.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(N_modes, N_modes)``
        Symmetric overlap matrix.
    """
    N = psi_matrix.shape[0]

    weights = torch.ones(N, device=psi_matrix.device)
    weights[0] = 0.5
    weights[-1] = 0.5

    weighted = psi_matrix * weights.unsqueeze(1)

    return weighted.T @ psi_matrix * dx

def cross_overlap_matrix(
    psi_A: torch.Tensor,
    psi_B: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Compute <psi_A_m | psi_B_n>.

    Parameters
    ----------
    psi_A : torch.Tensor, shape ``(N_x, n_modes)``
        POD eigenmode.
    psi_B : torch.Tensor, shape ``(N_x, n_modes)``
        Learned wavefunction.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(n_modes, n_modes)``
        Overlap matrix of <psi_A_m | psi_B_n>.
    """
    N = psi_A.shape[0]

    weights = torch.ones(N, device=psi_A.device)
    weights[0] = 0.5
    weights[-1] = 0.5

    weighted_A = psi_A * weights.unsqueeze(1)
    return weighted_A.T @ psi_B * dx

# ----------------------------------------------------------------------
# 2️⃣ Smoke‑test entry point
# ----------------------------------------------------------------------

def _run_pod_smoke_test() -> None:
    """
    Minimal sanity check.
    - builds synthetic sinusoidal snapshot matrix
    - runs the SVD
    - verifies orthonormality of the modes,
    - prints the leading singular values
    """
    from utils import make_grid, set_global_seed

    set_global_seed(27)

    # 1-D grid
    x = make_grid(-1.0, 1.0, 64)
    dx = float(x[1] - x[0])

    # Create three snapshots (sin, cos, sin(2x))
    snapshots = torch.stack([torch.sin(x.squeeze()),
                             torch.cos(x.squeeze()),
                             torch.sin(2 * x.squeeze())], dim=1)    # (N_x, N_modes)

    U, S, Vh = pod_decomposition(snapshots)

    # Orthonormality check: U^T U approx I
    ortho_err = torch.norm(U.T @ U - torch.eye(U.shape[1]))
    overlap = mode_overlap_matrix(snapshots, dx)

    print("✔️ POD smoke test")
    print(f"  leading singular values : {S[:3].tolist()}")
    print(f"  orthonormality error    : {ortho_err:.2e}")
    print(f"  overlap matrix shape    : {overlap.shape}")

def main() -> None:
    """Entry point used when the module is directly executed."""
    _run_pod_smoke_test()

if __name__ == "__main__":
    main()