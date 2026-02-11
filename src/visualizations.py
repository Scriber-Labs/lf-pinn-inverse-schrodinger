# src/visualization.py
"""
🖼️ Visualization utilities for the inverse Schrödinger demo.

All functions accept plain NumPy / PyTorch objects are return the matplotlib Figure they create -> they are easy to unit test and reuse from notebooks or scripts.

✨ Features
    - type-annotated
    - emoji section dividers for readability
    - list-comprehensions wherever relevant
    - a minimal smoke-test
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# ----------------------------------------------------------------------
# 🌍 Global style helper
# ----------------------------------------------------------------------
def _apply_style() -> None:
    """Set the global rcParams used throughout the module."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 120,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
        }
    )

# ----------------------------------------------------------------------
# ✨ Helper: gradient bar plotting
# ----------------------------------------------------------------------
def _plot_gradient_bar(ax, x, height, width, cmap, label=None):
    """
    Draw a bar with a smooth gradient fill (lighter at bottom, darker at top).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the bar on.
    x : float
        The x-position of the bar.
    height : float
        The height (value) of the bar.
    width : float
        The width of the bar.
    cmap : matplotlib.colors.Colormap
        The colormap to use for the gradient.
    label : str, optional
        Legend label (only used once per bar group).
    """
    n_segments = 50  # Number of thin bars to simulate gradient
    segment_height = height / n_segments
    for i in range(n_segments):
        # Reverse the color index so lighter colors are at bottom (i=0) and darker at top
        color = cmap(1 - i / (n_segments - 1)) if n_segments > 1 else cmap(1)
        ax.bar(x, segment_height, width=width, bottom=i*segment_height, 
               color=color, edgecolor='none')
    # Add label to legend using a proxy artist (a visible patch)
    if label:
        from matplotlib.patches import Patch
        ax.patches[-1].set_label(label)

# ----------------------------------------------------------------------
# ✨ Helper: horizontal lambdas‑row (figure‑level)
# ----------------------------------------------------------------------
def _add_lambda_row(
        fig: plt.Figure,
        lambdas: Dict[str, float],
        *,
        ax: plt.Axes | None = None,
) -> None:
    """
    Render the loss-weight dictionary as a single horizontal row.
    The row is placed **just below the title** (if it exists):

    - If the figure has ``suptitle`` -> below that.
    - Else if an ``ax`` is supplied (or can be inferred) -> below the Axes title.
    - Otherwise fall back to safe default near the top of the canvas.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure on which the annotation the loss weights will be rendered.
    lambdas : dict[str, float]
        Mapping of loss-weight names -> numeric values.
    ax : matplotlib.axes.Axes, optional
        The axes whose title should be used as a reference point.
        If omitted, the function will try to locate the first Axes in ``fig.axes`.
    """
    # Build the formatted string of loss weights (four spaces between entries)
    lambda_str = "    ".join(
        rf"$\lambda_{{{k}}} = {v:g}$" for k, v in lambdas.items()
    )

    # Determine where the suptitle lives (if it exists)
    # ``fig._suptitle`` is the Text object created by ``fig.suptitle``.
    # It may be ``None`` if the user never called a ``suptitle``.
    suptitle = getattr(fig, "_suptitle", None)

    if suptitle is not None:
        # Get the title's *figure* coordinates (x, y) - y is near 0.98.
        _, title_y = suptitle.get_position()
        # Pull the text down by a modest amount (approximately 5% of the figure height).
        # The factor 0.05 works well for the default 9x5 inch canvas.
        lambda_y = title_y - 0.05
    else:
        # No suptitle -> fall back to an Axes title (most of the plots in this module use ax.set_title)
        # If the caller supplied an Axes, use it; otherwise grab the first one.
        if ax is None:
            if fig.axes:
                ax = fig.axes[0]        # first Axes in the figure
            else:
                # No Axes at all -> use a generic safe default
                lambda_y = 0.94
                fig.text(
                    0.5,
                    lambda_y,
                    lambda_str,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85, pad=3.0),
                    transform=fig.transFigure,
                )
            return

        # Convert the Axes bounding box to figure coordinates
        # ``ax.get_position()`` returns a Bbox in *figure* coordinates already.
        bbox = ax.get_position()
        # ``box.y1`` is the top edge of the Axes (0-1 in figure space)
        # Pull the text down by a small fraction for the figure height.
        lambda_y = bbox.y1 - 0.03   # 3% of figure height works well for 9x5

    # Draw the annotation
    fig.text(
        0.5,
        lambda_y,
        lambda_str,
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=3.0),
        transform=fig.transFigure,
    )


# ----------------------------------------------------------------------
# 📊 1️⃣ Training Curves
# ----------------------------------------------------------------------
def plot_loss_history(
        epochs: Sequence[int],
        total: Sequence[float],
        physics: Sequence[float],
        norm: Sequence[float],
        smooth: Sequence[float],
        data: Sequence[float],
        lambdas: Dict[str, float],
        out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Render a log-scale line plot of all loss components.

    Parameters
    ----------
    epochs : Sequence[int]
        Epoch numbers (usually ``range(1, N+1)``).
    total, physics, data, smooth, norm : Sequence[float]
        Per-epoch scalar losses.
    lambdas : dict[str, float]
        Mapping ``{'data':..., `physics`:..., `smooth`:..., `norm`:...}``.
    out_path : Path or None (optional)
        If provided, the figure is saved to this path location (PNG).

    Returns
    -------
    matplotlib.figure.Figure
        A 2x2 figure containing the log-scale training curves for each individual loss term.
    """
    _apply_style()

    # 🎨 color / label mapping (list comprehension keeps it tidy)
    comps: List[Tuple[str, str, Sequence[float]]] = [
        ("Total", "#8000FF", total),
        ("Physics", "#007FFF", physics),
        ("Norm", "#0FFFFF", norm),
        ("Smoothness", "#39FF14", smooth),
        ("Data-fit", "#E52B50", data),
    ]

    fig, ax = plt.subplots()
    for label, color, series in comps:
        ax.plot(epochs, series, label=label, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss components")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()

    # ⚖️ Horizontal lambdas row
    _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 🌠 2️⃣ Potential plot (true vs. learned)
# ----------------------------------------------------------------------
def plot_potential(
    x: torch.Tensor,
    V_true: torch.Tensor,
    V_learned: torch.Tensor,
    lambdas: Dict[str, float],
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Plot the analytic potential and the network's prediction.

    All tensors are expected to be a 1-D (shape ``(N,)``) and on CPU
    """
    _apply_style()

    # Ensure everything is on the CPU and NumPy for Matplotlib
    x_np = x.squeeze().cpu().numpy()
    Vt_np = V_true.squeeze().cpu().numpy()
    Vl_np = V_learned.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(x_np, Vt_np, label=r"True $V(x)$", color="#E52B50", linewidth=4)
    ax.plot(
        x_np,
        Vl_np,
        label=r"$V_\theta(x)$",
        color="#39FF14",
        linewidth=4,
        ls="--",
    )
    # domain shapes
    for edge in (x_np.min(), x_np.max()):
        ax.axvline(edge, color="#A9A9A9", lw=3, ls=":", alpha=0.6)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$V$")
    ax.set_title("Learned vs. Ground Truth Potential")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="center")

    _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 🔱 3️⃣ Wave‑function comparison
# ----------------------------------------------------------------------
def plot_wavefunctions(
    x: torch.Tensor,
    psi_true: Sequence[torch.Tensor],
    psi_learned: Sequence[torch.Tensor],
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Side-by-side plot of each eigenmode (learned vs. ground truth)

    Parameters
    ----------
    x : torch.Tensor
        1-D tensor of spatial coordinates.
    psi_true, psi_learned : Sequence[torch.Tensor]
        Iterables of 1-D tensors, length = number of modes.
    out_path : pathlib.Path | None
        Optional output path.
    Returns
    -------
    matplotlib.figure.Figure
        A 1x3 figure whose subplots compare the learned vs. ground truth wavefunctions for the first three eigenmodes.
    """
    _apply_style()

    n_modes = len(psi_true)
    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(15, 4),
        sharey=True,
        constrained_layout=True,
    )
    # If there is only one mode, ``axes`` is not a list -> wrap it.
    if n_modes == 1:
        axes = [axes]

    x_np = x.squeeze().cpu().numpy()
    true_col, learn_col = "#E52B50", "#39FF14"

    for idx, (ax, pt, pl) in enumerate(
        zip(axes, psi_true, psi_learned)
    ):
        ax.plot(
            x_np,
            pt.squeeze().cpu().numpy(),
            label=rf"True $\psi_{idx}(x)$",
            color=true_col,
            linewidth=4,
        )
        ax.plot(
            x_np,
            pl.squeeze().cpu().numpy(),
            label=rf"Learned $\psi_{idx}^\theta(x)$",
            color=learn_col,
            ls="--",
            linewidth=4,
        )
        ax.set_xlabel(r"$x$")
        ax.set_title(rf"Mode $n={idx}$")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel(r"$\psi(x)$")
    fig.suptitle(
        f"Learned vs. Ground Truth Wavefunctions ( {n_modes} modes)",
        fontsize=16,
    )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ──────────────────────────────────────────────────────────────
# 📊 4️⃣ Energy‑spectrum histogram (ground‑truth vs. learned)
# ──────────────────────────────────────────────────────────────
from typing import Mapping

def plot_energy_spectrum(
    E_true: torch.Tensor,
    E_learned: torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Bar-chart comparison of the first ``n_states`` energy levels.

    Parameters
    ----------
    E_true : torch.Tensor
        Ground-truth energies, shape ``(n_states,)``.
    E_learned : torch.Tensor
        Learned energies from ``model.E_theta()``, same shape as ``E_true``.
    out_path : pathlib.Path | None, optional
        Optional output path (saved as a PNG).

    Returns
    -------
    matplotlib.figure.Figure
        A bar chart comparing the first ``n_states`` energy levels (learned vs. ground truth).
    """
    _apply_style()

    # Softer color pairs for better balance: warm reds to cool purples, greens to warm yellows
    true_col_1, true_col_2 = "#D9534F", "#9B59B6"
    learn_col_1, learn_col_2 = "#5CB85C", "#F0AD4E"

    # Ensure we are working with CPU NumPy arrays -> no gradient tracking
    E_true_np = E_true.detach().cpu().numpy()
    E_learn_np = E_learned.detach().cpu().numpy()

    # Indices for the spatial grid (assumes tensors are already ordered)
    indices = np.arange(len(E_true_np))

    fig, ax = plt.subplots()

    # Configure grid settings
    ax.grid(
        visible=True,
        which='major',
        axis='y',
        color='grey',
        linestyle=':',
        linewidth=1.0,
        alpha=0.6,
        zorder=0  # Place grid behind bars
    )

    # Create gradient colormaps with balanced colors
    true_cmap = mcolors.LinearSegmentedColormap.from_list("true_grad", [true_col_1, true_col_2])
    learn_cmap = mcolors.LinearSegmentedColormap.from_list("learn_grad", [learn_col_1, learn_col_2])
    
    # Plot true energy bars with gradient
    for i, idx in enumerate(indices - 0.15):
        _plot_gradient_bar(ax, idx, E_true_np[i], 0.3, true_cmap, 
                          label="True" if i == 0 else None)
    
    # Plot learned energy bars with gradient
    for i, idx in enumerate(indices + 0.15):
        _plot_gradient_bar(ax, idx, E_learn_np[i], 0.3, learn_cmap, 
                          label="Learned" if i == 0 else None)

    ax.set_xticks(indices)
    ax.set_xticklabels([rf"$n={i}$" for i in indices])
    ax.set_ylabel(rf"Energy ($\hbar \omega_n$ units)")
    ax.set_title("Exact vs. Learned Energy Eigenvalues")
    
    # Create custom legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=true_col_1, edgecolor='black', label='True'),
        Patch(facecolor=learn_col_1, edgecolor='black', label='Learned')
    ]
    ax.legend(handles=legend_patches)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 🧪 Smoke test – runs when the module is executed directly
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    """Generate dummy data and produce all figures."""
    torch.manual_seed(27)

    # Dummy grid
    N = 128
    x = torch.linspace(-5.0, 5.0, N)

    # Fake potentials
    V_true = 0.5 * x**2
    V_learned = V_true + 0.2 * torch.randn_like(V_true)

    # Fake eigenfunctions (sinusoidal basis)
    psi_true = [torch.sin((i + 1) * x) for i in range(3)]
    psi_learned = [
        pt + 0.1 * torch.randn_like(pt) for pt in psi_true
    ]

    # Dummy loss histories (exponential and decay noise)
    epochs = list(range(1, 101))
    total = np.exp(-0.03 * np.arange(100)) + 0.02 * np.random.rand(100)
    physics = np.exp(-0.025 * np.arange(100)) + 0.015 * np.random.rand(100)
    norm = np.exp(-0.04 * np.arange(100)) + 0.008 * np.random.rand(100)
    smooth = np.exp(-0.02 * np.arange(100)) + 0.005 * np.random.rand(100)
    data = np.exp(-0.035 * np.arange(100)) + 0.01 * np.random.rand(100)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "norm": 10}

    # Produce figures in a temporary folder
    out_dir = pathlib.Path("./_smoke_outputs")
    out_dir.mkdir(exist_ok=True)

    plot_loss_history(
        epochs,
        total,
        physics,
        norm,
        smooth,
        data,
        lambdas,
        out_path=out_dir / "loss_history.png",
    )
    plot_potential(
        x,
        V_true,
        V_learned,
        lambdas,
        out_path=out_dir / "potential.png",
    )
    plot_wavefunctions(
        x,
        psi_true,
        psi_learned,
        out_path=out_dir / "wavefunctions.png",
    )

    # Energy-spectrum bar plot
    # Dummy ground truth energies (linear ladder)
    E_true = torch.tensor([0.5, 1.5, 2.5])          # hbar*omega_n units
    # Fake learned energies -> perturb the true values slightly
    E_learned = E_true + 0.1 * torch.randn_like(E_true)

    energy_path = out_dir / "energy.png"
    plot_energy_spectrum(
        E_true=E_true,
        E_learned=E_learned,
        out_path=energy_path,
    )

    print(
        f"\n✅ Smoke test complete. All {len(list(out_dir.iterdir()))} figures written to {out_dir.resolve()}\n"
    )

def main() -> None:
    """Entry point for ``python -m src.visualizations`` -> runs the smoke test."""
    _smoke_test()

if __name__ == "__main__":
    main()
