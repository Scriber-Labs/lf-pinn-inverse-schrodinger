# src/visualizations.py
"""
🖼️ Visualization utilities for the inverse Schrödinger demo.

All functions accept plain NumPy / PyTorch objects are return the matplotlib Figure they create -> they are easy to unit test and reuse from notebooks or scripts.

✨ Features
    - type-annotated
    - emoji section dividers for readability
    - list-comprehensions wherever relevant
    - a minimal smoke-test

Author: Eigenscribe
Development note: LLM assistance was used during construction; implementation has been reviewed and adapted for this project.
Review status: Reviewed and maintained by Eigenscribe.
Date: 02-2026
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import numpy as np
import torch

from pod import pod_decomposition, physical_pod_decomposition, cross_overlap_matrix

# ----------------------------------------------------------------------
# 🌍 Global style helper
# ----------------------------------------------------------------------
def _apply_style() -> None:
    """Set global plotting style (Seaborn + custom rcParams)."""

    palette = ["#FE28A2", "#845CCC", "#4361EE", "#2CA9E6", "#309592"]

    sns.set_theme(
        style="darkgrid",
        context="notebook",
        palette=palette,
        font_scale=0.8,
        rc={
            "axes.facecolor": "#0d1117",
            "figure.facecolor": "#0d1117",
            "savefig.facecolor": "#0d1117",
            "grid.color": "#444444",
            "text.color": "#E6E6E6",
            "axes.labelcolor": "#E6E6E6",
            "xtick.color": "#DDDDDD",
            "ytick.color": "#DDDDDD",
            "axes.edgecolor": "#DDDDDD",
        }
    )
    sns.set_palette(sns.color_palette(palette, desat=1.0))

    #plt.rcParams.update(
    #    {
    #        "figure.figsize": (9, 5),
    #        "figure.dpi": 120,
    #        "axes.labelsize": 13,
    #        "axes.titlesize": 14,
    #        "legend.fontsize": 11,
    #        "lines.linewidth": 2,
    #    }
    #)

    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 120,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
        }
    )
_apply_style()

PROJECT_COLORS = {
    "purple": "#845CCC",
    "pink": "#FE28A2",
    "green": "#39FF14",
    "cyan": "#0FFFFE",
    "blue": "#007FFF",
}

spatial_overlap_cmap = mcolors.LinearSegmentedColormap.from_list("spatial_overlap", [
    (0.00, "#00F0FF"),  # Neon Cyan
    (0.25, "#007BFF"),  # Royal Blue
    (0.50, "#5E17EB"),  # Electric Purple
    (0.75, "#F72585"),  # Neon Pink
    (1.00, "#FFBD00"),  # Bright Goldenrod (for max overlap pop)
])

BLUE_TO_PINK = mcolors.LinearSegmentedColormap.from_list("blue_to_pink_fancy", [
    (0.00, "#00BFFF"),  # Deep Sky Blue
    (0.35, "#7000FF"),  # Vivid Violet
    (0.65, "#FF007F"),  # Bright Rose
    (1.00, "#FFEE00"),  # Vibrant Yellow
])

temporal_cmap = mcolors.LinearSegmentedColormap.from_list("temporal_green_polished", [
    (0.00, "#003200"),  # Deep Jungle Green
    (0.30, "#00A000"),  # Vivid Green
    (0.60, "#39FF14"),  # Neon Green (Alien)
    (0.85, "#CCFF33"),  # Electric Lime
    (1.00, "#0FFFFE"),  # Electric Cyan
])

temporal_overlap_cmap = mcolors.LinearSegmentedColormap.from_list("temporal_overlap_polished", [
    (0.00, "#3D34EB"),  # Intense Blue
    (0.25, "#14B5FF"),  # Azure
    (0.50, "#00FFC2"),  # Bright Aquamarine
    (0.75, "#70E000"),  # Lime
    (1.00, "#FF5400"),  # International Orange (Contrast pop)
])



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
#🩵 1️⃣ Training Curves
# ----------------------------------------------------------------------
def plot_loss_history(
    epochs: Sequence[int],
    total: Sequence[float],
    physics: Sequence[float],
    data: Sequence[float],
    smooth: Sequence[float],
    ordered: Sequence[float],
    lambdas: Dict[str, float],
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Render a log-scale line plot of all loss components.

    Parameters
    ----------
    epochs : Sequence[int]
        Epoch numbers (usually ``range(1, N+1)``).
    total, physics, data, smooth, ordered : Sequence[float]
        Per-epoch scalar losses.
    lambdas : dict[str, float]
        Mapping ``{'data':..., `physics`:..., `smooth`:..., `ordered`:...}``.
    out_path : Path or None (optional)
        If provided, the figure is saved to this path location (PNG).

    Returns
    -------
    matplotlib.figure.Figure
        A 2x2 figure containing the log-scale training curves for each individual loss term.
    """

    # 🎨 color / label mapping (list comprehension keeps it tidy)
    comps: List[Tuple[str, str, Sequence[float]]] = [
        ("Total", PROJECT_COLORS["purple"], total),
        ("Physics", PROJECT_COLORS["blue"], physics),
        ("Ordered", PROJECT_COLORS["cyan"], ordered),
        ("Smoothness", PROJECT_COLORS["green"], smooth),
        ("Data-fit", PROJECT_COLORS["pink"], data),
    ]

    fig, ax = plt.subplots(facecolor="#0d1117")
    for label, color, series in comps:
        sns.lineplot(
            x=epochs,
            y=series,
            ax=ax,
            label=label,
            color=color,
        )

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

    # Ensure everything is on the CPU and NumPy for Matplotlib
    x_np = x.squeeze().cpu().numpy()
    Vt_np = V_true.squeeze().cpu().numpy()
    Vl_np = V_learned.squeeze().cpu().numpy()

    fig, ax = plt.subplots(facecolor="#0d1117")

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

# ----------------------------------------------------------------------
# 🔷 4️⃣ Energy‑spectrum histogram (ground‑truth vs. learned)
# ----------------------------------------------------------------------
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
    true_col_1, true_col_2 =  "#FF0090", "#E52B50"
    learn_col_1, learn_col_2 = "#03C03C", "#39FF14"

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

    # ------------------------------------------------------------------
    # 🗃 Save if requested
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 🫟 5️⃣ Probability‑density comparison (|ps_theta_n|^2 vs. rho_obs_n)
# ----------------------------------------------------------------------
def plot_density_vs_observed(
        x: torch.Tensor,
        psi_learned: Sequence[torch.Tensor],
        rho_obs: Sequence[torch.Tensor],
        *,
        lambdas: Dict[str, float] | None = None,
        out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Plot the learned probability densities |psi_theta_n(x)|^2 alongside the observed densities rho_obs_n(x) for each mode n.

    Parameters
    ----------
    x : torch.Tensor
        1-D spatial grid, shape ``(n_modes,)``.
    psi_learned : Sequence[torch.Tensor]
        Learned wavefunctions psi_theta_n(x). Each tensor must be 1-D of length ``n_modes``.
    rho_obs : Sequence[torch.Tensor]
        Corresponding observed probability densities rho_obs_n(x). Same shape as ``psi_learned``.
    lambdas : Dict[str, float] | None, optional
        If supplied, the loss-weight row will be added below the title.
    out_path : pathlib.Path | None, optional
        Destination path (saved as a PNG). If ``None``, the figure is only returned.

    Returns
    -------
    matplotlib.figure.Figure
        Probability density comparison (|psi_theta_n(x)|^2 vs. rho_obs_n(x)).
    """
    _apply_style()

    # ------------------------------------------------------------------
    # 🤯 Sanity checks (will raise early if shapes mismatch)
    # ------------------------------------------------------------------
    n_modes = len(psi_learned)
    assert n_modes == len(rho_obs), "❌ Mismatched number of modes."
    # Convert the grid once -> everything else will be Numpy for Matplotlib
    x_np = x.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    # 🖼️ Create a subplot for each mode (1 × n_modes)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(max(5 * n_modes, 12), 4),    # wider for more modes
        sharey=True,
        constrained_layout=True,
    )
    if n_modes == 1:
        axes = [axes]    # make the iterator uniform

    psi_theta_col, rho_obs_col = "#38FE84", "#F78F55"

    for idx, (ax, psi, rho) in enumerate(
            zip(axes, psi_learned, rho_obs),
    ):
        # |psi|^2 -> detach, move to CPU, and square element-wise
        prob_density = (psi.squeeze().detach().cpu() ** 2).numpy()
        obs_density  = rho.squeeze().detach().cpu().numpy()

        ax.plot(
            x_np,
            obs_density,
            label=r"Observed $\rho_n^{\text{obs}}(x)$",
            color=rho_obs_col,
            linewidth=4,
        )
        ax.plot(
            x_np,
            prob_density,
            label=r"$|\hat{\psi}_n^{\theta}(x)|^2$",
            color=psi_theta_col,
            linewidth=4,
            ls="--",
        )
        ax.set_xlabel(r"$x$")
        ax.set_title(rf"Mode $n={idx}$")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel(r"Probability density")

    fig.suptitle(
        "Learned vs. Observed Probability Densities",
        fontsize=16,
    )

    # ------------------------------------------------------------------
    # ⚖️Optional row of loss weights
    # ------------------------------------------------------------------
    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=axes[0])   # any axis works for reference

    # ------------------------------------------------------------------
    # 🗃️ Save if requested
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig



# ----------------------------------------------------------------------
# 📊6️⃣a) POD singular values (log plot, all values)
# ----------------------------------------------------------------------
def plot_pod_singular_values(
        singular_values: torch.Tensor,
        *,
        title: str = "POD singular values",
        ylabel: str = "Singular value (log scale)",
        out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Plot the singular values obtained from a POD decomposition on a logarithmic y-axis.

    Parameters
    ----------
    singular_values : torch.Tensor
        1-D tensor of singular values (sigma_k) -> typically the output of ``pod_decomposition`` (the `S` component).
    title : str, optional
        Figure title. Defaults to a generic POD-SV caption.
    ylabel : str, optional
        Y-axis label. Defaults to "Singular value (log scale)".
    cmap : str, optional
        Color map. Defaults to "cool". Unused but kep for backward compatibility with the previous signature.
    out_path : pathlib.Path | None, optional
        Destination path (saved as a PNG). If ``None``, the figure is only returned.

    Returns
    -------
    matplotlib.figure.Figure
        The log plot of singular values.
    """
    _apply_style()

    # ------------------------------------------------------------------
    # 1️⃣ Convert to NumPy (detach, CPU) -> no gradients needed
    # ------------------------------------------------------------------
    sv = singular_values.detach().cpu().numpy()

    # -------------------------------------------------
    # 2️⃣ Structured console printout
    # -------------------------------------------------
    print("\nSingular Values:")
    print("-" * 30)
    print(f"{'Mode (k)':>10} | {'Sigma_k':>15}")
    print("-" * 30)

    for i, val in enumerate(sv, start=1):
        print(f"{i:10d} | {val:15.6e}")

    print("-" * 30)

    # ------------------------------------------------------------------
    # 3️⃣ Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    indices = np.arange(1, len(sv) + 1)

    # 1. Draw the connecting line
    ax.semilogy(
        indices,
        sv,
        color="white",
        linewidth=2,
        alpha=0.4,
        zorder=1
    )

    # 2. Draw the scatter points with a colormap
    sctr = ax.scatter(
        indices,
        sv,
        c=sv,
        cmap="cool",
        edgecolor="white",
        linewidth=0.5,
        s=60,
        zorder=2,
        label="Singular values",
    )

    # Set y-scale to log explicitly for the scatter points
    ax.set_yscale("log")

    plt.colorbar(sctr, ax=ax, label="Magnitude")

    # Annotate each point with its numerical value
    for i, val in enumerate(sv, start=1):
        # Default alignment and offset
        ha = "center"
        va = "bottom"
        xytext = (0, 7)
        
        # Adjust horizontal alignment for the first and last points to avoid axes overlap
        if i == 1:
            ha = "left"
            va = "top"
            xytext = (5, -7)
        elif i == len(sv):
            ha = "right"
            xytext = (-5, 7)

        ax.annotate(
            f"{val:.4e}",
            (i, val),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=10,
            alpha=0.9,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
        )

    ax.set_xlabel(r"Mode index $k$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, pad=12)
    ax.grid(True, which="both", alpha=0.2)

    # ------------------------------------------------------------------
    # 4️⃣ Optional save
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊6️⃣ b) First three spatial POD modes (should resemble the true eigenmodes)
# ----------------------------------------------------------------------
def plot_pod_first_three_spatial_modes(
        x: torch.Tensor,
        spatial_modes: torch.Tensor,
        *,
        ground_truth: Sequence[torch.Tensor] | None = None,
        psi_learned: Sequence[torch.Tensor] | None = None,
        lambdas: Dict[str, float] | None = None,
        out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Plot the first three columns of the POD spatial-mode matrix ``U``.
    If a list of ground-truth wavefunctions is supplied, each POD mode is overlaid with the corresponding ground truth wavefunction for visual comaprison.

    Parameters
    ----------
    x : torch.Tensor
        1-D grid on which the modes are evaluated (shape ``(N, 1)`` or ``(N,)``).
    spatial_modes : torch.Tensor
        POD spatial modes matrix `U` of shape ``(N, n_modes)``. The function will plot the first three columns (or fewer if ``n_modes < 3``).
    ground_truth : Sequence[torch.Tensor], optional
        Ground-truth wavefunctions `psi_0, psi_1, ...`, each of shape ``(N,)``. If provided, the i-th ground truth wavefunction is plotted with POD mode i.
    psi_learned : Sequence[torch.Tensor], optional
        Learned wavefunctions `psi_theta_0, psi_theta_1, ...`, each of shape ``(N,)``. If provided, the i-th learned wavefunction is plotted with POD mode i.
    lambdas : Dict[str, float] | None, optional
        Optional string with loss weights to be displayed on the figure.
    out_path : pathlib.Path | None, optional
        Destination path (saved as a PNG). If ``None``, the figure is only returned.

    Returns
    -------
    matplotlib.figure.Figure
        A plot of the first `n_modes` spatial POD modes.
    """
    _apply_style()

    # Ensure we work on CPU and detach from the autograd graph
    x_np = x.squeeze().detach().cpu().numpy()
    modes_np = spatial_modes.detach().cpu().numpy()     # shape (N, n_modes)

    # Ground-truth handling
    if ground_truth is None:
        gt_np = []
    else:
        gt_np = [
            ground_truth.squeeze().detach().cpu().numpy()
            for ground_truth in ground_truth[: modes_np.shape[1]]
        ]

    # Learned wavefunction handling
    if psi_learned is None:
        psi_learned_np = []
    else:
        psi_learned_np = [
            psi_learned.squeeze().detach().cpu().numpy()
            for psi_learned in psi_learned[: modes_np.shape[1]]
        ]

    n_plot = min(3, modes_np.shape[1])

    # ------------------------------------------------------------------
    # 1️⃣ Create subplots (1 x n_plot)
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(
        1,
        n_plot,
        figsize=(5 * n_plot, 4),
        constrained_layout=True,
    )
    # If there is only one subplot, `axs` is not iterable -> wrap it.
    if n_plot == 1:
        axs = [axs]

    pod_mode_color = "#009DFF"
    true_color = "#E52B50"
    learned_color = "#39FF14"

    for k in range(n_plot):
        ax = axs[k]

        # ---- Ground-truth (if provided) ----
        if k < len(gt_np):
            ax.plot(x_np,
                    gt_np[k],
                    label=rf"True $\psi_{k}$",
                    color=true_color,
                    linewidth=4.5,
                    )

        # ---- Learned wavefunctions (if provided) ----
        if k < len(psi_learned_np):
            ax.plot(x_np,
                    psi_learned_np[k],
                    label=rf"$\hat{{\psi}}_{k}^\theta$",
                    color=learned_color,
                    ls="--",
                    linewidth=4.5,
                    )

        # ---- POD mode ----
        ax.plot(
            x_np,
            modes_np[:, k],
            label=f"POD mode {k}",
            color=pod_mode_color,
            ls=":",
            linewidth=4.75,
        )

        ax.set_xlabel(r"$x$")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"POD spatial model {k}")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.2)

    # ------------------------------------------------------------------
    # 2️⃣ Optional loss weights row
    # ------------------------------------------------------------------
    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=axs[0])

    # ------------------------------------------------------------------
    # 3️⃣ Optional save
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 🗺️7️⃣a) -  Overlap matrix heatmap (POD diagnostic)
# ----------------------------------------------------------------------
def plot_overlap_heatmap(
    psi_theta: Sequence[torch.Tensor],
    dx: float,
    *,
    lambdas: Dict[str, float] | None = None,
    cmap: mcolors.Colormap | str = spatial_overlap_cmap,
    fmt: str = ".2f",
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Render a heat map of the overlap matrix <psi_theta_m | psi_theta_n>.

    Parameters
    ----------
    psi_theta : Sequence[torch.Tensor]
        Learned wavefunctions, each 1-D with the same length. The function will stack them into a (n_modes, N) matrix.
    dx : float
        Grid spacing.
    lambdas : Dict[str, float] | None, optional
        Optional string with loss weights to be displayed on the figure.
    cmap, fmt : str
        Color map, print settings for inputs values, and output path.
    out_path : pathlib.Path | None
        Destination path (saved as a PNG). If ``None``, the figure is only returned.

    Returns
    -------
    matplotlib.figure.Figure
        Overlap matrix heatmap -> POD diagnostic.
    """

    n_modes = len(psi_theta)
    # ------------------------------------------------------------------
    # 1️⃣ Stack and normalize the wavefunctions
    # ------------------------------------------------------------------
    psi_theta_mat = torch.stack([p.squeeze().detach().cpu() for p in psi_theta]).T  # (N, n_modes)

    # We use mode_overlap_matrix from pod.py to be consistent with how other overlaps are calculated.
    from pod import mode_overlap_matrix
    overlap_tensor = mode_overlap_matrix(psi_theta_mat, dx)

    # To ensure diagonal values are exactly 1, we normalize the overlap matrix.
    # This accounts for both the physical normalization of the wavefunctions
    # and the specific numerical integration scheme (trapezoidal rule).
    norms = torch.sqrt(torch.diag(overlap_tensor))
    overlap_tensor = overlap_tensor / torch.outer(norms, norms)

    overlap = overlap_tensor.numpy()

    # ------------------------------------------------------------------
    # 2️⃣ Plot the heat map
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(5, n_modes * 1.2), 5), facecolor="#0d1117")

    sns.heatmap(
        overlap,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=fmt,
        cbar_kws={"label": "Overlap matrix"},
        linewidths=0,
    )

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"$n={i}$" for i in range(n_modes)], rotation=45, ha="right")
    ax.set_yticklabels([rf"$n={i}$" for i in range(n_modes)])


    ax.set_title(r"Overlap Matrix $\langle \hat{\psi}_m^\theta | \hat{\psi}_n^\theta \rangle$")
    #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Overlap matrix")

    # ------------------------------------------------------------------
    # 3️⃣ Loss weights row
    # ------------------------------------------------------------------
    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    # ------------------------------------------------------------------
    # 4️⃣ Save if requested
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊7️⃣b) Cross-overlap matrix heatmap
# ----------------------------------------------------------------------
def plot_cross_overlap_heatmap(
    pod_modes_physical: Sequence[torch.Tensor] | torch.Tensor,
    psi_matrix: Sequence[torch.Tensor] | torch.Tensor,
    dx: float,
    *,
    cmap: mcolors.Colormap | str = BLUE_TO_PINK,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Create a heatmap of the *cross* overlap matrix <u_k | psi_n^theta> where ``u_k`` are the physical POD modes and ``psi_n^theta`` are learned wavefunctions.

    This function uses the ``cross_overlap_matrix`` routine defined `src.pod.py`.

    Parameters
    ----------
    pod_modes_physical : Sequence[torch.Tensor]
        Physical POD modes (each 1-D, same length). If a single tensor is passed it is interpreted as a stacked matrix of shape ``(n_modes, N)``.
    psi_matrix : Sequence[torch.Tensor]
        Learned wavefunctions Psi^theta = [psi_1^theta psi_2^theta ...]
    dx : float
        Spatial grid spacing.
    cmap, fmt : str, optional
        Colormap and numeric formatting for the cell.
    lambdas : Dict[str, float], optional
        Optional loss-weight dictionary.
    out_path : pathlib.Path | None, optional
        Optional output path to save the image (PNG).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Cross overlap matrix <u_k | psi_n^theta> heatmap.
    """
    cross_overlap = cross_overlap_matrix(
        pod_modes_physical,
        psi_matrix,
        dx=dx,
    )       # Expected shape: (n_modes, n_modes)

    n_modes = cross_overlap.shape[0]

    fig, ax = plt.subplots(figsize=(max(5, n_modes * 1.2), 5), facecolor="#0d1117")


    im = ax.imshow(
        cross_overlap,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # Ensure no grid lines for the heatmap
    ax.grid(False)

    # Axis ticks
    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"$n={i}$" for i in range(n_modes)],
                       rotation=45, ha="right")
    ax.set_yticklabels([rf"$n={i}$" for i in range(n_modes)])

    # ------------------------------------------------------------------
    # #️⃣ Annotate every cell with its numeric value
    # ------------------------------------------------------------------
    for i in range(n_modes):
        for j in range(n_modes):
            txt = f"{cross_overlap[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if abs(cross_overlap[i, j]) > 0.5 else "black",
                fontsize=9,
            )

    # ------------------------------------------------------------------
    # 🔖 Titles, color‑bar and optional λ‑row
    # ------------------------------------------------------------------
    ax.set_title(r"Cross Overlap Matrix $\langle u_k | \hat{\psi}_n^\theta \rangle$",
                 fontsize=16,
                 )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Overlap matrix",
    )

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    # ------------------------------------------------------------------
    # 3️⃣ Optional save
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊7️⃣c) POD Temporal Modes (Composition Matrix)
# ----------------------------------------------------------------------
def plot_pod_temporal_modes(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = temporal_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Create a heatmap of the POD temporal modes matrix V (where V is the right-singular vector matrix).
    In this context, V acts as a 'Modal Composition Matrix' showing how each POD mode is distributed across learned states.

    Parameters
    ----------
    Vh : torch.Tensor | np.ndarray
        The H-transpose of the right singular matrix V (from SVD: U S Vh).
        Expected shape: (n_modes, n_modes).
    cmap, fmt : str, optional
        Colormap and numeric formatting.
    lambdas : Dict[str, float], optional
        Optional loss-weight dictionary.
    out_path : pathlib.Path | None, optional
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if isinstance(Vh, torch.Tensor):
        Vh_np = Vh.detach().cpu().numpy()
    else:
        Vh_np = Vh

    # V is the matrix whose columns are temporal modes. Since we have Vh,
    # V = Vh.conj().T. For real-valued SVD, V = Vh.T.
    V = Vh_np.T
    n_states, n_pod_modes = V.shape

    fig, ax = plt.subplots(figsize=(max(5, n_pod_modes * 1.2), 5), facecolor="#0d1117")

    # We use symmetric limits because V is often orthonormal (entries between -1 and 1)
    im = ax.imshow(
        V,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # Ensure no grid lines for the heatmap
    ax.grid(False)

    # Axis ticks
    ax.set_xticks(np.arange(n_pod_modes))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([rf"Mode $k={i}$" for i in range(n_pod_modes)],
                       rotation=45, ha="right")
    ax.set_yticklabels([rf"State $n={i}$" for i in range(n_states)])

    # Annotate cells
    for i in range(n_states):
        for j in range(n_pod_modes):
            val = V[i, j]
            # Determine text color based on background lightness
            # The RdBu colormap is dark at ends (-1, 1) and light in middle (0)
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j, i, f"{val:{fmt}}",
                ha="center", va="center",
                color=text_color,
                fontsize=9
            )

    ax.set_title(r"Temporal Modes (Modal Composition $V_{nk}$)", fontsize=16)
    ax.set_ylabel("Learned States ($n$)")
    ax.set_xlabel("POD Modes ($k$)")

    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label="Coefficient Value"
    )

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊7️⃣d) POD Temporal Overlap Heatmap
# ----------------------------------------------------------------------
def plot_pod_temporal_overlap_heatmap(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = temporal_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Render a heatmap of the overlap matrix between temporal modes (columns of V).
    Since V is unitary (V^H V = I), this should be an identity matrix.

    Parameters
    ----------
    Vh : torch.Tensor | np.ndarray
        The H-transpose of the right singular matrix V.
    cmap, fmt : str, optional
    lambdas : Dict[str, float], optional
    out_path : pathlib.Path | None, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if isinstance(Vh, torch.Tensor):
        Vh_np = Vh.detach().cpu().numpy()
    else:
        Vh_np = Vh

    # V columns are temporal modes. Overlap matrix is V^H @ V.
    V = Vh_np.conj().T
    overlap = V.conj().T @ V
    overlap = np.real(overlap)

    n_modes = overlap.shape[0]

    fig, ax = plt.subplots(figsize=(max(5, n_modes * 1.2), 5), facecolor="#0d1117")

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # Ensure no grid lines for the heatmap
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"Mode $k={i}$" for i in range(n_modes)], rotation=45, ha="right")
    ax.set_yticklabels([rf"Mode $k={i}$" for i in range(n_modes)])

    for i in range(n_modes):
        for j in range(n_modes):
            val = overlap[i, j]
            ax.text(
                j, i, f"{val:{fmt}}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=9
            )

    ax.set_title(r"Temporal Mode Overlap $\langle v_m | v_n \rangle$", fontsize=16)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Overlap")

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊7️⃣e) POD Temporal Cross-Overlap Heatmap
# ----------------------------------------------------------------------
def plot_pod_temporal_cross_overlap_heatmap(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = temporal_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Render a heatmap of the cross-overlap between learned states (standard basis) 
    and POD temporal modes (columns of V). This is exactly the matrix V itself.

    Parameters
    ----------
    Vh : torch.Tensor | np.ndarray
    cmap, fmt : str, optional
    lambdas : Dict[str, float], optional
    out_path : pathlib.Path | None, optional
    """
    if isinstance(Vh, torch.Tensor):
        Vh_np = Vh.detach().cpu().numpy()
    else:
        Vh_np = Vh

    V = Vh_np.T
    n_states, n_modes = V.shape

    fig, ax = plt.subplots(figsize=(max(5, n_modes * 1.2), 5), facecolor="#0d1117")

    im = ax.imshow(
        np.abs(V),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # Ensure no grid lines for the heatmap
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([rf"Mode $k={i}$" for i in range(n_modes)], rotation=45, ha="right")
    ax.set_yticklabels([rf"State $n={i}$" for i in range(n_states)])

    for i in range(n_states):
        for j in range(n_modes):
            val = V[i, j]
            ax.text(
                j, i, f"{val:{fmt}}",
                ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=9
            )

    ax.set_title(r"Temporal Cross-Overlap $|V_{nk}| = |\langle \mathbf{e}_n | v_k \rangle|$", fontsize=16)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Absolute overlap")

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 📊7️⃣f) POD–Eigenbasis Alignment Heatmap
# ----------------------------------------------------------------------
def plot_pod_eigen_alignment(
    pod_modes_physical: Sequence[torch.Tensor] | torch.Tensor,
    psi_true_matrix: Sequence[torch.Tensor] | torch.Tensor,
    dx: float,
    *,
    cmap: mcolors.Colormap | str = spatial_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Create a heat map of the overlap matrix <u_k | psi_n> where ``u_k`` are the physical POD modes and ``psi_n`` are the ground truth wavefunctions.

    This function uses the `cross_overlap_matrix`` routine defined in `src.pod.py`.

    Parameters
    ----------
    pod_modes_physical : Sequence[torch.Tensor]
        Physical POD modes (each 1-D, same length). If a single tensor is passed, it is interpreted as a stacked matrix of shape ``(n_modes, N)``.
    psi_true_matrix : Sequence[torch.Tensor]
        Ground truth wavefunctions Psi = [psi_1 psi_2 ...]
    dx : float | None, optional
    cmap, fmt : str, optional
        Colormap and numeric formatting for the cell
    lambdas : Dict[str, float], optional
        Optional loss-weight dictionary.
    out_path : pathlib.Path | None, optional
        Optional output path to save the image (PNG).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Overlap matrix <u_k | psi_n> heatmap.
    """
    overlap = cross_overlap_matrix(
        pod_modes_physical,
        psi_true_matrix,
        dx=dx,
    )           # Expected shape: (n_modes, n_modes)

    n_modes = overlap.shape[0]

    fig, ax = plt.subplots(figsize=(max(5, n_modes * 1.2), 5), facecolor="#0d1117")

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest'
    )
    # Ensure no grid lines for the heatmap
    ax.grid(False)

    # Axis ticks
    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"$n={i}$" for i in range(n_modes)],
                       rotation=45, ha="right")
    ax.set_yticklabels([rf"$n={i}$" for i in range(n_modes)])

    # ------------------------------------------------------------------
    # #️⃣ Annotate every cell with its numeric value
    # ------------------------------------------------------------------
    for i in range(n_modes):
        for j in range(n_modes):
            txt = f"{overlap[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if abs(overlap[i, j]) > 0.5 else "black",
                fontsize=9,
            )
    # ------------------------------------------------------------------
    # 🔖 Titles, color‑bar and optional λ‑row
    # ------------------------------------------------------------------
    ax.set_title(r"Overlap Matrix $\langle u_k | \hat{\psi}_n \rangle$",
                 fontsize=16,
                 )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Overlap matrix",
    )

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    # ------------------------------------------------------------------
    # 3️⃣ Optional save
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------
# 8️⃣🫟 Hilbert Space Phase Portrait
# ----------------------------------------------------------------------
def plot_hilbert_phase_portrait(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    true_wavefunctions: np.ndarray | torch.Tensor,
    x: np.ndarray | torch.Tensor,
    *,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Embed learned wavefunctions in eigenstate coefficient space (Hilbert portrait).

    Projects high-dimensional wavefunctions onto the subspace spanned by the first few true eigenstates, visualizing the distribution of learned states in a 3D coefficient space.

    Parameters
    ----------
    learned_wavefunctions : np.ndarray | torch.Tensor
        Array of shape (n_grid_points, n_samples) containing learned wavefunction approximations.
    true_wavefunctions : np.ndarray | torch.Tensor
        Array of shape (n_grid_points, n_eigenstates) containing exact eigenstates.
    x : np.ndarray | torch.Tensor
        Array of shape (n_grid_points,) containing the spatial grid points for integration.
    out_path : pathlib.Path | None, optional
        Destination path (saved as a PNG). If ``None``, the figure is only returned.

    Returns
    -------
    matplotlib.figure.Figure
        3D scatter plot of wavefunction coefficients.

    Notes
    -----
    The coefficients are computed as the overlap integral:
        c_j = int(psi_learned(x) * psi_true_j(x) dx)

    Using the trapezoidal rule for integration
    """
    _apply_style()

    # ------------------------------------------------------------------
    # 1️⃣ Convert to NumPy and validate shapes
    # ------------------------------------------------------------------
    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()
    if isinstance(true_wavefunctions, torch.Tensor):
        true_wavefunctions = true_wavefunctions.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Ensure x is 1D
    x = np.atleast_1d(x).squeeze()

    if learned_wavefunctions.ndim != 2:
        raise ValueError(
            f"❌ learned_wavefunctions must be 2D, got {learned_wavefunctions.ndim}D"
        )
    if true_wavefunctions.ndim != 2:
        raise ValueError(
            f"❌ true_wavefunctions must be 2D, got {true_wavefunctions.ndim}D"
        )

    n_grid = learned_wavefunctions.shape[0]
    if true_wavefunctions.shape[0] != n_grid:
        raise ValueError(
            f"❌ Grid mismatch: learned has {n_grid} points, "
            f"true has {true_wavefunctions.shape[0]} points"
        )
    if x.shape[0] != n_grid:
        raise ValueError(
            f"❌ Grid mismatch: x has {x.shape[0]} points, "
            f"wavefunctions have {n_grid} points"
        )

    # ------------------------------------------------------------------
    # 2️⃣ Compute coefficients (Overlap Integrals)
    # ------------------------------------------------------------------
    # Limit to first 3 eigenstates for 3D visualization
    n_states = min(3, true_wavefunctions.shape[1])
    n_modes = min(3, learned_wavefunctions.shape[1])    # ⚠️ added minimum number myself

    coeffs =  np.zeros((n_modes, n_states), dtype=np.float64)

    for i in range(n_modes):
        psi_learned = learned_wavefunctions[:, i]
        for j in range(n_states):
            true_psi = true_wavefunctions[:, j]

            integrand = psi_learned * true_psi
            coeffs[i, j] = np.trapezoid(integrand, x)

    # ------------------------------------------------------------------
    # 3️⃣ Plot 3D Stem Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate data ranges for setting limits
    x_vals = coeffs[:, 0]
    y_vals = coeffs[:, 1]
    z_vals = coeffs[:, 2]

    # Add padding to limits to prevent clipping
    margin_x = (x_vals.max() - x_vals.min()) * 0.2 if x_vals.max() != x_vals.min() else 0.5
    margin_y = (y_vals.max() - y_vals.min()) * 0.2 if y_vals.max() != y_vals.min() else 0.5
    margin_z = (z_vals.max() - z_vals.min()) * 0.2 if z_vals.max() != z_vals.min() else 0.5

    ax.set_xlim(x_vals.min() - margin_x, x_vals.max() + margin_x)
    ax.set_ylim(y_vals.min() - margin_y, y_vals.max() + margin_y)
    ax.set_zlim(0, max(z_vals.max() + margin_z, 0.1))   # Ensure z starts at zero and has some room

    # Draw stems from z=0 to each point
    for i in range(n_modes):
        x0, y0, z0 = coeffs[i, 0], coeffs[i, 1], coeffs[i, 2]

        # Draw stems from z=0.0 to z=z0
        z_start = 0.0
        z_end = max(z0, 0.001)

        ax.plot(
            [coeffs[i, 0], coeffs[i, 0]],   # x: constant
            [coeffs[i, 1], coeffs[i, 1]],   # y: constant
            [0, coeffs[i, 2]],
            color=PROJECT_COLORS["blue"],
            linewidth=2.5,
            alpha=0.8,
            zorder=2,
        )

        ax.scatter(
            coeffs[i, 0],
            coeffs[i, 1],
            coeffs[i, 2],
            c=PROJECT_COLORS["blue"],
            alpha=0.7,
            s=100,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )

        # Add text annotations
        text_z = z0 + margin_z * 0.5 if z0 > 0 else 0.05
        ax.text(
            x0, y0, text_z,
            f"({x0:.2f}, {y0:.2f}, {z0:.2f})",
            fontsize=9,
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
            zorder=6,
        )

    ax.set_xlabel(r"$\langle \psi_i^\theta | \psi_0 \rangle$", fontsize=11)
    ax.set_ylabel(r"$\langle \psi_i^\theta | \psi_1 \rangle$", fontsize=11)
    ax.set_zlabel(r"$\langle \psi_i^\theta | \psi_2 \rangle$", fontsize=11)

    ax.set_title("Hilbert Space Phase Portrait", fontsize=12, pad=10)

    ax.view_init(elev=25, azim=55)

    # ------------------------------------------------------------------
    # 4️⃣ Optional save
    # ------------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig


# ----------------------------------------------------------------------
# 📊9️⃣ Spectral Energy Cascade
# ----------------------------------------------------------------------
def plot_spectral_energy_cascade(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    energies: np.ndarray | torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Compare POD singular values with Hamiltonian energy spectrum.

    Parameters
    ----------
    learned_wavefunctions : np.ndarray | torch.Tensor
        Learned wavefunctions matrix (n_grid, n_modes).
    energies : np.ndarray | torch.Tensor
        Learned or true energy eigenvalues.
    out_path : pathlib.Path | None, optional
    """
    _apply_style()

    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()
    if isinstance(energies, torch.Tensor):
        energies = energies.detach().cpu().numpy()

    spatial_modes, singular_values, _ = pod_decomposition(learned_wavefunctions)

    n = min(len(singular_values), len(energies))

    fig, ax1 = plt.subplots(figsize=(7, 5), facecolor="#0d1117")

    # POD spectrum
    sns.lineplot(
        x=np.arange(1, n + 1),
        y=singular_values[:n],
        marker="o",
        ax=ax1,
        label="POD singular values",
        color=PROJECT_COLORS["pink"],
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Mode index")
    ax1.set_ylabel("Singular value", color=PROJECT_COLORS["pink"])
    ax1.tick_params(axis='y', labelcolor=PROJECT_COLORS["pink"])

    # second axis for energies
    ax2 = ax1.twinx()

    sns.lineplot(
        x=np.arange(1, n + 1),
        y=energies[:n],
        marker="s",
        ax=ax2,
        linestyle="--",
        label="Energy eigenvalues",
        color=PROJECT_COLORS["cyan"],
    )

    ax2.set_ylabel("Energy", color=PROJECT_COLORS["cyan"])
    ax2.tick_params(axis='y', labelcolor=PROJECT_COLORS["cyan"])
    ax2.grid(False)

    ax1.set_title("Spectral Energy Cascade", fontsize=14, pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax2.get_legend().remove()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig


# ----------------------------------------------------------------------
# 📊🔟 POD Partition Function Spectrum
# ----------------------------------------------------------------------
def plot_partition_function_spectrum(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Interpret POD spectrum as a thermodynamic ensemble.

    Parameters
    ----------
    learned_wavefunctions : np.ndarray | torch.Tensor
        Learned wavefunctions matrix (n_grid, n_modes).
    out_path : pathlib.Path | None, optional
    """
    _apply_style()

    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()

    _, singular_values, _ = pod_decomposition(learned_wavefunctions)

    if isinstance(singular_values, torch.Tensor):
        singular_values = singular_values.detach().cpu().numpy()

    # probability weights
    p = singular_values**2
    p = p / np.sum(p)

    # effective energies
    E_eff = -np.log(p + 1e-12)

    # entropy
    entropy = -np.sum(p * np.log(p + 1e-12))

    modes = np.arange(1, len(p) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 5), facecolor="#0d1117")

    sns.lineplot(
        x=modes,
        y=p,
        marker="o",
        ax=ax1,
        label="Boltzmann weights",
        color=PROJECT_COLORS["purple"],
    )

    ax1.set_ylabel("Mode probability", color=PROJECT_COLORS["purple"])
    ax1.tick_params(axis='y', labelcolor=PROJECT_COLORS["purple"])
    ax1.set_xlabel("Mode index")

    ax2 = ax1.twinx()

    sns.lineplot(
        x=modes,
        y=E_eff,
        marker="s",
        linestyle="--",
        ax=ax2,
        label="Effective energy",
        color=PROJECT_COLORS["green"],
    )

    ax2.set_ylabel("Effective energy", color=PROJECT_COLORS["green"])
    ax2.tick_params(axis='y', labelcolor=PROJECT_COLORS["green"])
    ax2.grid(False)

    ax1.set_title(f"POD Partition Function Spectrum (Entropy={entropy:.3f})", fontsize=14, pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax2.get_legend().remove()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

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
    ortho = np.exp(-0.04 * np.arange(100)) + 0.008 * np.random.rand(100)
    smooth = np.exp(-0.02 * np.arange(100)) + 0.005 * np.random.rand(100)
    data = np.exp(-0.035 * np.arange(100)) + 0.01 * np.random.rand(100)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "ortho": 10}

    # Produce figures in a temporary folder
    out_dir = pathlib.Path("./_smoke_outputs")
    out_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------
    # 1️⃣ Plot the training curves for each term and the total loss
    # --------------------------------------------------------------
    plot_loss_history(
        epochs,
        total,
        physics,
        ortho,
        smooth,
        data,
        lambdas,
        out_path=out_dir / "loss_history.png",
    )

    # --------------------------------------------------------------
    # 2️⃣ Plot the learned potential and the ground truth potential
    # --------------------------------------------------------------
    plot_potential(
        x,
        V_true,
        V_learned,
        lambdas,
        out_path=out_dir / "potential.png",
    )

    # --------------------------------------------------------------
    # 3️⃣ Plot wavefunctions for first `n_modes` eigenmodes
    # --------------------------------------------------------------
    plot_wavefunctions(
        x,
        psi_true,
        psi_learned,
        out_path=out_dir / "wavefunctions.png",
    )

    # --------------------------------------------------------------
    # 4️⃣ Energy-spectrum bar plot
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # 5️⃣ Probability‑density comparison (dummy data)
    # --------------------------------------------------------------
    # Fake observed densities -> noisy version of |psi|^2
    rho_obs = [
        (pt.squeeze() ** 2 + 0.02 * torch.rand_like(pt.squeeze())) for pt in psi_true
    ]

    density_path = out_dir / "density_vs_observed.png"
    plot_density_vs_observed(
        x=x,
        psi_learned=psi_learned,    # from the earlier dummy wavefunctions
        rho_obs=rho_obs,
        lambdas=lambdas,            # optional -> omit if you don't want loss terms to render on probability density plot
        out_path=density_path,
    )

    # --------------------------------------------------------------
    # 6️⃣ a) POD singular-value spectrum (dummy data)
    # --------------------------------------------------------------
    # Use the same psi_theta matrix you already built for POD demo
    psi_matrix = torch.stack(psi_learned, dim=1)    # shape (N, n_modes)
    dx = float(x[1] - x[0])

    pod_modes_physical, S, Vh_dummy, pod_modes_euclidean = physical_pod_decomposition(
        psi_matrix,
        dx,
        reference_modes=psi_matrix,
        align_signs=True,
    )

    sv_path = out_dir / "pod_singular_values.png"
    plot_pod_singular_values(
        singular_values=S,
        out_path=sv_path,
    )

    # --------------------------------------------------------------
    # 6️⃣b) First three POD spatial modes
    # -------------------------------------------------------------
    pod_modes_path = out_dir / "pod_modes.png"
    plot_pod_first_three_spatial_modes(
        x=x,
        spatial_modes=pod_modes_physical,
        ground_truth=psi_true,
        lambdas=lambdas,
        out_path=pod_modes_path,
    )

    # --------------------------------------------------------------
    # 7️⃣a) Overlap‑matrix heatmap (dummy POD diagnostic)
    # --------------------------------------------------------------
    overlap_path = out_dir / "overlap_heatmap.png"
    plot_overlap_heatmap(
        psi_theta=psi_learned,  # use the same learned wavefunction from the dummy data
        dx=dx,
        lambdas=lambdas,  # optional - show loss weights
        out_path=overlap_path,
    )

    # --------------------------------------------------------------
    # 7️⃣b) Temporal modes heatmap (Composition Matrix V)
    # --------------------------------------------------------------
    temporal_path = out_dir / "pod_temporal_modes.png"
    plot_pod_temporal_modes(
        Vh=Vh_dummy,  # we have Vh from the physical_pod_decomposition call above
        lambdas=lambdas,
        out_path=temporal_path,
    )

    temporal_overlap_path = out_dir / "pod_temporal_overlap.png"
    plot_pod_temporal_overlap_heatmap(
        Vh=Vh_dummy,
        lambdas=lambdas,
        out_path=temporal_overlap_path,
    )

    temporal_cross_path = out_dir / "pod_temporal_cross_overlap.png"
    plot_pod_temporal_cross_overlap_heatmap(
        Vh=Vh_dummy,
        lambdas=lambdas,
        out_path=temporal_cross_path,
    )

    # --------------------------------------------------------------
    # 7️⃣c) POD–eigenbasis alignment heatmap
    # --------------------------------------------------------------
    alignment_path = out_dir / "pod_eigen_alignment.png"
    plot_pod_eigen_alignment(
        pod_modes_physical,
        torch.stack(psi_true, dim=0).T,
        dx=dx,
        lambdas=lambdas,
        out_path=alignment_path
    )

    # --------------------------------------------------------------
    # 8️⃣ Hilbert Space Phase Portrait
    # --------------------------------------------------------------
    # Stack the list of wavefunctions into a 2D tensor (n_modes, n_grid)
    psi_learned_stacked = torch.stack(psi_learned, dim=1)   # Shape: (N, n_modes)
    psi_true_stacked = torch.stack(psi_true, dim=1)         # Shape: (N, n_eigenmodes)

    hilbert_path = out_dir / "hilbert_portrait.png"
    plot_hilbert_phase_portrait(
        learned_wavefunctions=psi_learned_stacked,
        true_wavefunctions=psi_true_stacked,
        x=x,
        out_path=hilbert_path,
    )

    # --------------------------------------------------------------
    # 9️⃣ Spectral Energy Cascade
    # --------------------------------------------------------------
    cascade_path = out_dir / "spectral_cascade.png"
    plot_spectral_energy_cascade(
        learned_wavefunctions=psi_matrix,
        energies=E_true,
        out_path=cascade_path,
    )

    # --------------------------------------------------------------
    # 🔟 POD Partition Function Spectrum
    # --------------------------------------------------------------
    partition_path = out_dir / "partition_spectrum.png"
    plot_partition_function_spectrum(
        learned_wavefunctions=psi_matrix,
        out_path=partition_path,
    )

    try:
        print(
            f"\n✅ Smoke test complete. All {len(list(out_dir.iterdir()))} figures written to {out_dir.resolve()}\n"
        )
    except UnicodeEncodeError:
        print(
            f"\n[OK] Smoke test complete. All {len(list(out_dir.iterdir()))} figures written to {out_dir.resolve()}\n"
        )

def main() -> None:
    """Entry point for ``python -m src.visualizations`` -> runs the smoke test."""
    _smoke_test()

if __name__ == "__main__":
    main()
