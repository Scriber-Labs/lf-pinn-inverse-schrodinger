# src/model.py
"""
Neural network architecture for the inverse time-independent Schrödinger problem.

The model learns three families of quantities:
    1. **Potential**                        `V_theta(x)`
    2. **Eigenmodes** (wavefunctions)       `psi_theta_n(x)`
    3. **Energy eigenvalues**               `E_theta_n`

All three quantities are parameterized by a shared latent vector ``theta`` (the network weights).
The code purposefully employs a clean, "over-engineered" house style so that each component can be independently tested and swapped out later (e.g., different activation functions, different PDE constraints, etc.).

Author: Eigenscribe
Development note: LLM assistance was used during construction; implementation has been reviewed and adapted for this project.
Review status: Reviewed and maintained by Eigenscribe.
Date: 02-2026
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils import set_global_seed, normalize_wavefunctions, l2_inner_product

# ----------------------------------------------------------------------
# 1️⃣ Neural network model
# ----------------------------------------------------------------------

class MLP(nn.Module):
    """
    Minimal multilayer perceptron with two hidden layers and ``tanh`` activations.

    This network is used for **all** learned scalar fields:
    - the potential ``V_theta(x)``
    - each eigenmode ``psi_theta_n(x)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (for project 2, this is always ``1`` -> corresponds with the spatial coordinate ``x``).
    output_dim : int
        Dimensionality of the output (for project 2, this is also always ``1`` -> corresponds with the scalar field value).
    hidden_dims : List[int]
        Width of each hidden layer. Typical values are ``[64, 64]`` or ``[128, 128]``.
    device : torch.device or ``str``, optional
        Target device for the parameters. If ``None`` the model inherits the default device of the surrounding context.
    dtype : torch.dtype, optional
        Desired floating-point precision (``torch.float64`` or ``torch.float64``).

    Notes
    -----
    - The activation is applied **after** each linear layer except for the final one.
    - ``tanh`` is chosen because it is bounded and differentiable. This implements nicely with the physics-informed loss term.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Build the sequential stack: Linear -> Tanh -> Linear -> ... -> Linear
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], device=device, dtype=dtype))
            # Insert activation after every hidden layer (not after the output)
            if i < len(dims) -2:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -> forwards ``x`` through the stacked MLP."""
        return self.net(x)

class NormalizedWavefunctionNet(nn.Module):
    """
    Wraps an MLP to enforce L2 normalization by construction.
    Eliminates the need for a separate normalization loss term.
    """
    def __init__(self, base_net: nn.Module, dx: float) -> None:
        super().__init__()
        self.base_net = base_net
        self.dx = dx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw prediction
        psi_raw: torch.Tensor = self.base_net(x)

        # Enforce normalization via utility to ensure consistency
        # We use a single element list for normalize_wavefunctions
        return normalize_wavefunctions([psi_raw.squeeze()], self.dx)[0].unsqueeze(1)


class InverseSchrodingerModel(nn.Module):
    """
    Joint model that bundles together:
    - ``V_theta``           a learned potential,
    - ``psi_theta_n``       the *n*-th learned eigenmode,
    - ``E_theta_n``         the *n*-th learned energy eigenvalue,

    All three families share the **same hidden-layer architecture** (``hidden_dimes``). However, they are separately initiated so they can learn independent parameters.

    Parameters
    ----------
    n_states : int
        Number of eigenstates to learn.
    hidden_dims : List[int]
        Hidden layer sizes for the underlying MLP.
    device : torch.device or ``str``, optional
        Device on which to place all sub-modules. If ``None`` the model inherits the default device of the surrounding context.
    dtype : torch.dtype, optional
        Precision for the parameters (defaults to ``torch.float64``).

    Attributes
    ----------
    potential_net : MLP
        Represents ``V_theta(x)``.
    psi_nets : nn.ModuleList(MLP)
        One MLP per eigenstate, representing ``psi_theta_n(x)``.
    energies : nn.Parameter
        Tensor of shape ``(n_states,)`` holding ``E_theta_n`.
    """

    def __init__(
        self,
        n_states: int,
        hidden_dims: List[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        dx: float,
    ) -> None:
        super().__init__()

        self.n_states = n_states

        # Potential network -> single scalar field
        self.potential_net = MLP(
            input_dim=1,
            output_dim=1,
            hidden_dims=hidden_dims,
            device=device,
            dtype=dtype,
        )

        # One wavefunction network per eigenstate
        self.psi_nets = nn.ModuleList(
            [
                NormalizedWavefunctionNet(
                    MLP(
                        input_dim=1,
                        output_dim=1,
                        hidden_dims=hidden_dims,
                        device=device,
                        dtype=dtype,
                    ),
                    dx=dx,
                )
                for _ in range(n_states)
            ]
        )

        # Energy parameters -> learnable scalars (no need for bias term)
        # We initialize with a fixed seed if we want cross-platform identity, 
        # but since set_global_seed is called before model init, we are covered.
        self.energies = nn.Parameter(torch.linspace(0.5, n_states - 0.5, n_states, dtype=dtype or torch.float64))

    # ------------------------------------------------------------------
    # 2️⃣ Helper methods – expose the learned fields with the desired names
    # ------------------------------------------------------------------

    def V_theta(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned potential ``V_theta(x)``."""
        return self.potential_net(x)

    def psi_theta(self, x: torch.Tensor, dx: float | None = None) -> List[torch.Tensor]:
        """
        Return a list of orthonormal wavefunctions via Gram-Schmidt.
        """
        # 1. Get raw normalized predictions from sub-nets
        psi_list = [net(x).squeeze() for net in self.psi_nets]
        
        # 2. Orthonormalize via Gram-Schmidt
        ortho_list = []
        dx_val = dx if dx is not None else self.psi_nets[0].dx

        for psi in psi_list:
            for prev in ortho_list:
                overlap = l2_inner_product(psi, prev, dx_val)
                psi = psi - overlap * prev

            # Re-normalize after subtraction
            psi = normalize_wavefunctions([psi], dx_val)[0]
            ortho_list.append(psi)

        return ortho_list

    def E_theta(self) -> torch.Tensor:
        """
        Return the learned energy vector ``E_theta`` (shape ``(n_states,)``).
        """
        return self.energies

    # ------------------------------------------------------------------
    # 3️⃣ Convenience wrappers
    # ------------------------------------------------------------------
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy alais for :meth:`V_theta`."""
        return self.V_theta(x)

    def psi(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Legacy alias for :meth:`psi_theta`."""
        return self.psi_theta(x)

# ----------------------------------------------------------------------
# 4️⃣ Smoke‑test helpers
# ----------------------------------------------------------------------

def _run_smoke_test() -> None:
    """Basic sanity check -> forward pass through all components."""
    set_global_seed(27)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dx: float = 0.01

    # Instantiate a model with three eigenstates and a modest hidden size.
    model = InverseSchrodingerModel(n_states=3, hidden_dims=[64, 64], device=device,dx=dx)

    # Sample a spatial grid
    x = torch.linspace(-1.0, 1.0, 100, device=device).unsqueeze(1)

    # Forward pass
    V = model.V_theta(x)
    psi_list = model.psi_theta(x)
    E = model.E_theta()

    # Quick sanity prints
    print("✔️ InverseSchrodingerModel forward OK")
    print(f"Input shape             : {x.shape}")
    print(f"Potential shape         : {V.shape}")
    print(f"Number of eigenmodes    : {len(psi_list)} (each {psi_list[0].shape})")
    print(f"Energies shape          : {E.shape}")

# ----------------------------------------------------------------------
# 5️⃣ Entry point
# ----------------------------------------------------------------------
def main() -> None:
    """Run the smoke test when the module is executed as a script."""
    # Use UTF-8 for output to support emojis on Windows
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    _run_smoke_test()

if __name__ == "__main__":
    main()