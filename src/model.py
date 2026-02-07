# src/model.py
"""
Neural network architecture for the inverse time-independent Schrödinger problem.

The model learns three families of quantities:
    1. **Potential**                        `V_theta(x)`
    2. **Eigenmodes** (wavefunctions)       `psi_theta_n(x)`
    3. **Energy eigenvalues**               `E_theta_n`

All three quantities are parameterized by a shared latent vector ``theta`` (the network weights).
The code purposefully employs a clean, "over-engineered" house style so that each component can be independently tested and swapped out later (e.g., different activation functions, different PDE constraints, etc.).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils import set_global_seed

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
        Desired floating-point precision (``torch.float32`` or ``torch.float64``).

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
        Precision for the parameters (defaults to ``torch.float32``).

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
    ): -> None:
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
                MLP(
                    input_dim=1,
                    output_dim=1,
                    hidden_dims=hidden_dims,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(n_states)
            ]
        )

        # Energy parameters -> learnable scalars (no need for bias term)
        self.energies = nn.Parameter(torch.randn(n_states, dtype=dtype or torch.float32))

    