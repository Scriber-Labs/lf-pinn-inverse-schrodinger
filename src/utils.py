from __future__ import annotations
import random
import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Integer seed.
    deterministic : bool, default = False
        If True, enables deterministic algorithms (slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.maniual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ------------------------------------------------------------------
# Helper: compute d/dt of the network output using autograd
# ------------------------------------------------------------------
def model_derivative(
    net: torch.nn.Module,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Returns ∂ₜ net(t)  (the velocity) evaluated at the times `t`.

    Parameters
    ----------
    net : torch.nn.Module
        Trained PINN that maps scalar time → position.
    t : torch.Tensor
        Shape ``(N, 1)`` – time points (requires_grad=True).

    Returns
    -------
    torch.Tensor
        Shape ``(N, 1)`` – time derivative of the network output.
    """
    t = t.clone().detach().requires_grad_(True)          # ensure grad tracking
    x = net(t)                                          # forward pass → position
    # torch.autograd.grad returns a tuple; we take the first element
    dx_dt, = torch.autograd.grad(
        outputs=x,
        inputs=t,
        grad_outputs=torch.ones_like(x),
        create_graph=False,      # we only need the first derivative
        retain_graph=False,
    )
    return dx_dt