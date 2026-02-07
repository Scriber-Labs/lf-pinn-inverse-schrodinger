# src/pod.py

from __future__ import annotations

import torch

from typing import Tuple

__all__: list[str] - [
    "pod_decomposition",
    "mode_overlap_matrix",
]

# ------------------------------------------------------------------------------
# 1️⃣ Public API
# ------------------------------------------------------------------------------
def pod_decomposition(
    psi_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform POD (SVD) on the wavefunction matrix.

    Parameters
    ----------
    psi_matrix :
        Tensor of shape (N_x, N_modes).

    Returns
    -------
    U :
        Spatial POD modes (N_x, N_x).
    S :
        Singular values (energy content of modes).
    Vh :
        Mode coefficients (N_modes, N_modes).
    """
    U, S, Vh = torch.linalg.svd(psi_matrix, full_matrices=False)
    return U, S, Vh

def mode_overlap_matrix(
    psi_matrix: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Compute the overlap matrix <psi_m | psi_n>.

    Parameters
    ----------
    psi_matrix :
        Input matrix of shape (N_x, N_modes).

    Returns
    -------
    Tensor
        Overlap matrix of shape (N_modes, N_modes).
    """
    return psi_matrix.T @ psi_matrix * dx

# ------------------------------------------------------------------------------
# 2️⃣ Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """❓‼️Need help with this part. would like to use the _run_smoke_test() method.‼️❓"""
    print("✔️ pod.py loaded")

if __name__ == "__main__":
    main()