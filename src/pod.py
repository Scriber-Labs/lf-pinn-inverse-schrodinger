from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

__all__: list[str] - [
    "compute_psi_matrix",
    "pod_decomposition",
    "mode_overlap_matrix",
]

# ------------------------------------------------------------------------------
# 0️⃣ Helpers: collect wavefunctions
# ------------------------------------------------------------------------------
def compute_psi_matrix(
    psi_models: Iterable,
    x: Tensor,
) -> Tensor:
    """
    Evaluate multiple wavefunction models on a common spatial grid.

    Parameters
    ----------
    psi_models :
        Iterable of callables mapping ``x -> psi_n(x)``.
    x :
        Tensor of shape (N_x, 1).

    Returns
    -------
    Tensor
        Matrix of shape (N_x, N_modes), where each column is psi_n(x).
    """
    psi_vals = [psi(x).squeeze() for psi in psi_models]
    return torch.stack(psi_vals, dim=1)


# ------------------------------------------------------------------------------
# 2️⃣ Proper Orthogonal Decomposition (POD)
# ------------------------------------------------------------------------------

def pod_decomposition(
    psi_matrix: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
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
    V :
        Mode coefficients (N_modes, N_modes).
    """
    U, S, V = torch.linalg.svd(psi_matrix, full_matrices=False)
    return U, S, V

# ------------------------------------------------------------------------------
# 3️⃣ Mode overlap diagnostics
# ------------------------------------------------------------------------------

def mode_overlap_matrix(
    psi_matrix: Tensor,
) -> Tensor:
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
    return psi_matrix.T @ psi_matrix