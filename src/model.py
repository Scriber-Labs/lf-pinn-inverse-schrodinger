from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from utils import set_global_seed


# ------------------------------------------------------------------------------
# 1️⃣ Neural network model
# ------------------------------------------------------------------------------

class PotentialNet(nn.Module):
    """
    A minimal 2-hidden-layer tanh MLP: x -> V(x)

    This network represents the unknown potential V(x) in the inverse time-independent Schrödinger equation.

    Parameters
    ----------
    hidden : int, default = 64
        Number of hidden units per hidden layer.
    device : torch.device or str, optional
        Device on which the parameters will be allocated. If omitted, they will be kept on the current default device.
    dtype : torch.dtype, optional
        Floating-point precision of the parameters.
    """

    INPUT_DIM: Final[int] = 1
    OUTPUT_DIM: Final[int] = 1

    def __init__(
            self,
            hidden: int = 64,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Build the network
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, self.OUTPUT_DIM, device=device, dtype=dtype),
        )

        self._initialize_weights()

    # ------------------------------------------------------------------------------
    # 2️⃣ Public API
    # ------------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x: Tensor of shape (N_x, 1)

        Returns
        -------
        Tensor of shape (N_x, 1) representing the potential V(x)
        """
        return self.net(x)

    # ------------------------------------------------------------------------------
    # 3️⃣ Private helpers
    # ------------------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        """
        Initialize linear layers with Xavier initialization (appropriate for tanh activations).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# ------------------------------------------------------------------------------
# 4️⃣ Smoke test helpers
# ------------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Sanity check or PotentialNet forward pass."""
    set_global_seed(42)

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model: PotentialNet = PotentialNet(device=device).to(device)

    x: Tensor = torch.linspace(-1.0, 1.0, 5, device=device).unsqueeze(1)
    V: Tensor = model(x)

    print("✔️ PotentialNet forward OK")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {V.shape}")

# ------------------------------------------------------------------------------
# 5️⃣ Entry point
# ------------------------------------------------------------------------------
def main() -> None:
    """Run local tests when executed as a script."""
    run_smoke_test()

if __name__ == "__main__":
    main()