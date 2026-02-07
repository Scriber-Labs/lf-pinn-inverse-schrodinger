# src/model.py

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from utils import set_global_seed

# ------------------------------------------------------------------------------
# 1️⃣ Neural network model
# ------------------------------------------------------------------------------

class MLP(nn.Module):
    """
    A minimal 2-hidden-layer tanh MLP: x -> V(x)

    This network represents the unknown potential V(x) in the inverse time-independent Schrödinger equation.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    hidden_dims : List[int], default = ❓
        Number of hidden units per hidden layer.
    device : torch.device or str, optional, default = None
        Device on which the parameters will be allocated. If omitted, they will be kept on the current default device.
    dtype : torch.dtype, optional, default = None
        Floating-point precision of the parameters.
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

        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class InverseSchrodingerModel(nn.Module):
    """
    Joint model for:
    - Inferred (❓) Potential V_theta(x)
    - Learned (❓) Wavefunctions psi_n(x)
    - Learned (❓) Energies E_n
    """

    def __init__(
        self,
        n_states: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()

        self.n_states = n_states

        self.potential_net = MLP(1, 1, hidden_dims)

        self.psi_nets = nn.ModuleList(
            [MLP(1, 1, hidden_dims) for _ in range(n_states)]
        )

        self.energies = nn.Parameter(
            torch.randn(n_states)
        )

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.potential_net(x)

    def psi(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [net(x) for net in self.psi_nets]


# ------------------------------------------------------------------------------
# 2️⃣ Smoke test helpers
# ------------------------------------------------------------------------------

def _run_smoke_test() -> None:
    """Sanity check or MLP forward pass."""
    set_global_seed(42)

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model: InverseSchrodingerModel = InverseSchrodingerModel(n_states=3, hidden_dims=[64, 64])

    x: torch.Tensor = torch.linspace(-1.0, 1.0, 100, device=device).unsqueeze(1)
    V: torch.Tensor = model.potential(x)
    psi: List[torch.Tensor] = model.psi(x)

    print("✔️ MLP forward OK")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {V.shape}")
    print(f"Wavefunction shape: {len(psi)}")

# ------------------------------------------------------------------------------
# 5️⃣ Entry point
# ------------------------------------------------------------------------------
def main() -> None:
    """Run local tests when executed as a script."""
    _run_smoke_test()

if __name__ == "__main__":
    main()