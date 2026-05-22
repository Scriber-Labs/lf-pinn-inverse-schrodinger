# src/pod.py
"""
Proper Orthogonal Decomposition (POD) utilities.

- ``pod_decomposition``: thin wrapper around an SVD that returns the spatial modes, singular values, and modal coefficients.
- ``physical_pod_decomposition``: POD workflow using spatial-measure weighting, optional sign alignment, and physical POD-mode scaling.
- ``mode_overlap_matrix``: builds the inner product matrix < psi_m | psi_n > on a uniform grid. Note that the factor ``dx`` accounts for the integration measure.

Both functions are pure NumPy-style Torch ops -> no side effects, no hidden state.
Moreover, the module includes a minimal smoke test that can be called from the command line (``python -m src.pod``) or from any notebook.

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
    "physical_pod_decomposition",
    "weight_snapshot_matrix",
    "scale_pod_modes_to_physical",
    "align_modes_by_reference",
    "mode_overlap_matrix",
    "cross_overlap_matrix",
]

# ----------------------------------------------------------------------
# 0️⃣ Public API
# ----------------------------------------------------------------------

def pod_decomposition(
    psi_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a standard Euclidean POD/SVD on the wavefunction snapshot matrix.

    This routine intentionally does not apply spatial-measure weighting or
    physical POD-mode scaling. It returns Euclidean-normalized SVD modes.

    For physical POD diagnostics, prefer ``physical_pod_decomposition``.

    Parameters
    ----------
    psi_matrix : torch.Tensor, shape ``(N_x, N_modes)``
        Each column is a snapshot of the wavefunction evaluated on the spatial grid.

    Returns
    -------
    U : torch.Tensor, shape ``(N_x, N_modes)``
        Unitary spatial modes, orthonormal with respect to the Euclidean inner product.
    S : torch.Tensor, shape ``(N_modes,)``
        Singular values.
    Vh : torch.Tensor, shape ``(N_modes, N_modes)``
        Unitary temporal modes (❓), orthonormal with respect to the Euclidean inner product.
    """
    U, S, Vh = torch.linalg.svd(psi_matrix, full_matrices=False)
    return U, S, Vh

def weight_snapshot_matrix(
    psi_matrix: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Apply spatial-measure weighting before POD.
    
    For a uniform grid, physical inner products contain the measure ``dx``.
    Weighting the snapshot matrix by ``sqrt(dx)`` enables the Euclidean SVD to compute with the correct spatial measure.
    
    Parameters
    ----------
    psi_matrix : torch.Tensor, shape ``(N_x, N_modes)``
        Learned wavefunction snapshot matrix.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(N_x, N_modes)``
        Weighted snapshot matrix ``sqrt(dx) * psi_matrix``.
    """
    if dx <= 0:
        raise ValueError(f"❌ dx must be positive, got {dx}.")

    return psi_matrix * torch.sqrt(
        torch.as_tensor(dx, dtype=psi_matrix.dtype, device=psi_matrix.device)
    )

def scale_pod_modes_to_physical(
    pod_modes_euclidean: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Convert Euclidean-normalized POD modes to physically normalized modes.

    The SVD returns modes satisfying ``U.T @ U = I``. Physical normalization instead requires

    ``sum_i |u_k(x_i)|^2 = 1``.

    For a uniform grid, this is obtained by scaling

    ``u_k^phys = u_k / sqrt(dx)``.

    Parameters
    ----------
    pod_modes_euclidean : torch.Tensor, shape ``(N_x, N_modes)``
        Euclidean-normalized POD modes returned SVD.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(N_x, N_modes)``
        Physically normalized POD modes.
    """
    if dx <= 0:
        raise ValueError(f"❌ dx must be positive, got {dx}.")

    return pod_modes_euclidean / torch.sqrt(
        torch.as_tensor(dx, dtype=pod_modes_euclidean.dtype, device=pod_modes_euclidean.device)
    )

def align_modes_by_reference(
    modes: torch.Tensor,
    reference: torch.Tensor,
    dx: float | None = None,
) -> torch.Tensor:
    """"
    Align the sign of each mode against the strongest-overlap reference column.

    Singular vectors are only defined up to sign: if ``u_k`` is valid, then
    ``-u_k`` is equally valid. This function ensures plots and heatmaps are stable
    by flipping each mode so its strongest reference overlap is positive.

    Parameters
    ----------
    modes : torch.Tensor, shape ``(N_x, N_modes)``
        Modes whose column signs should be aligned.
    reference : torch.Tensor, shape ``(N_x, N_ref)``
        Reference wavefunctions or modes.
    dx : float | None, optional
        If provided, overlaps use the physical inner product with trapezoidal
        weights and spacing ``dx``. If ``None``, overlaps use a Euclidean dot product.

    Returns
    -------
    torch.Tensor, shape ``(N_x, N_modes)``
        Sign-aligned copy of ``modes``.
    """
    if modes.ndim != 2 or reference.ndim != 2:
        raise ValueError ("❌ modes and reference must both be 2-D matrices.")

    if modes.shape[0] != reference.shape[0]:
        raise ValueError(
            f"❌ Grid mismatch: modes has {modes.shape[0]} rows, "
            f"reference has {reference.shape[0]} rows."
        )

    aligned = modes.clone()

    if dx is None:
        overlaps = aligned.T @ reference
    else:
        overlaps = cross_overlap_matrix(aligned, reference, dx)

    strongest_ref_idx = torch.argmax(torch.abs(overlaps), dim=1)

    for mode_idx, ref_idx in enumerate(strongest_ref_idx):
        if overlaps[mode_idx, ref_idx] < 0:
            aligned[:, mode_idx] = -aligned[:, mode_idx]

    return aligned

def physical_pod_decomposition(
    psi_matrix: torch.Tensor,
    dx: float,
    *,
    reference_modes: torch.Tensor | None = None,
    align_signs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform the recommended physical POD diagnostic workflow.

    This function is based on the project 2 architecture diagram:

    1. Form the learned snapshot matrix ``Psi_theta``.
    2. Apply spatial-measure weighting: ``Psi_w = sqrt(dx) * Psi_theta``.
    3. Compute the Euclidean SVD: ``Psi_w = U Sigma V^T``.
    4. Optionally sign-align the Euclidean POD modes after SVD.
    5. Scale POD modes to physical normalization:
        ``u_k^phys = u_k / sqrt(dx)``.

    Phase/sign alignment is intentionally performed after the SVD and before physical POD scaling.

    Parameters
    ----------
    psi_matrix : matrix torch.Tensor, shape ``(N_x, N_modes)``
        Learned wavefunction snapshot matrix.
    dx : float
        Uniform grid spacing.
    reference_modes : torch.Tensor | None, optional
        Reference matrix used for sign alignment. If ``None`` and
        ``align_signs=True``, the learned snapshot matrix itself is used.
    align_signs : bool, default=True
        Whether to apply POD-mode sign alignment after SVD.

    Returns
    -------
    pod_modes_physical : torch.Tensor, shape ``(N_x, N_modes)``
        Physically normalized POD spatial modes.
    S : torch.Tensor, shape ``(N_modes),``
        Singular values form the weighted snapshot matrix.
    Vh : torch.Tensor, shape ``(N_modes, N_modes)``
        Transpose of the right singular-vector matrix.
    pod_modes_euclidean : torch.Tensor, shape ``(N_x, N_modes)``
        Euclidean-normalized POD modes after optional sign alignment.
    """
    psi_weighted = weight_snapshot_matrix(psi_matrix, dx)
    pod_modes_euclidean, S, Vh = pod_decomposition(psi_weighted)

    if align_signs:
        reference = psi_matrix if reference_modes is None else reference_modes
        pod_modes_euclidean = align_modes_by_reference(
            pod_modes_euclidean,
            reference,
            dx=None,
        )

    pod_modes_physical = scale_pod_modes_to_physical(pod_modes_euclidean, dx)

    return pod_modes_physical, S, Vh, pod_modes_euclidean

def mode_overlap_matrix(
    psi_matrix: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Compute the discrete overlap matrix <psi_m | psi_n> on a uniform grid.

    For a uniform grid, the inner product reduces to a simple matrix product scaled by the grid spacing ``dx``.

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
    - runs physical POD
    - verifies Euclidean and phsyical normalization
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
                             torch.sin(2 * x.squeeze())], dim=1)

    pod_modes_physical, S, Vh, pod_modes_euclidean = physical_pod_decomposition(
        snapshots,
        dx,
        reference_modes=snapshots,
        align_signs=True,
    )

    # Orthonormality check: U^T U approx I for Euclidean modes
    eye = torch.eye(pod_modes_euclidean.shape[1], dtype=pod_modes_euclidean.dtype)
    euclidean_ortho_err = torch.norm(pod_modes_euclidean.T @ pod_modes_euclidean - eye)


    # Physical overlap check: <u_i^phys | u_j^phys>_dx approx I
    physical_overlap = mode_overlap_matrix(pod_modes_physical, dx)
    physical_ortho_err = torch.norm(physical_overlap - eye)

    overlap = mode_overlap_matrix(snapshots, dx)

    print("✔️ POD smoke test")
    print(f"  leading singular values       : {S[:3].tolist()}")
    print(f"  Euclidean orthonormality error: {euclidean_ortho_err:.2e}")
    print(f"  physical orthonormality error : {physical_ortho_err:.2e}")
    print(f"  overlap matrix shape          : {overlap.shape}")


def main() -> None:
    """Entry point used when the module is directly executed."""
    _run_pod_smoke_test()

if __name__ == "__main__":
    main()