# src/visualizations.py
"""
🖼️ Visualization utilities for the inverse Schrödinger demo.

All functions accept plain NumPy / PyTorch objects and return the matplotlib Figure
they create -> easy to unit test and reuse from notebooks, CLI, or scripts.

Consistent styling adhering to the design tokens and palettes of
`there-and-back-again` and `research-notebook-1`.

✨ Features:
    - Unified dark slate theme with glassmorphic accents
    - High-contrast, perceptually smooth colormaps for overlap diagnostics
    - Standardized typography and font sizing across all figures
    - Constant semantic variable colors (True vs Learned vs Observed vs POD)
    - Full type annotations and self-contained smoke test

Author: Eigenscribe
Review status: Reviewed and maintained.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Final, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import torch

try:
    from .pod import (
        cross_overlap_matrix,
        mode_overlap_matrix,
        physical_pod_decomposition,
        pod_decomposition,
    )
except (ImportError, ValueError):
    from pod import (
        cross_overlap_matrix,
        mode_overlap_matrix,
        physical_pod_decomposition,
        pod_decomposition,
    )


# ======================================================================
# 🎨 Design Tokens & Color Palette (from research-notebook-1 & there-and-back-again)
# ======================================================================

# Core Design Tokens
THEME_BG = "#0d1117"        # Dark slate background
CARD_BG = "#161b22"         # Glassmorphic card / panel background
BORDER_COLOR = "#30363d"    # Structural borders and spines
GRID_COLOR = "#21262d"      # Subtle grid lines
TEXT_PRIMARY = "#e6edf3"    # High-contrast primary text
TEXT_MUTED = "#8b949e"      # Muted labels, secondary notes, and ticks

PROJECT_COLORS = {
    # Core brand palette
    "cyan_light": "#00FFEE",
    "cyan_blue": "#00E8FF",
    "blue_light": "#14B5FF",
    "blue_mid": "#0A95EB",
    "blue_deep": "#0070EB",
    "indigo": "#5280FF",
    "indigo_blue": "#5280FF",
    "purple_deep": "#A855F7",
    "purple_light": "#7952F5",
    "purple_dark": "#7C5CFF",
    "purple_lavender": "#D8B4FE",
    "pink_vibrant": "#FF66B3",
    "pink_pink": "#F72585",
    "pink_dark": "#AD1457",
    "pink_light": "#FFB4F6",
    "pink_alt": "#FF40A1",
    "orange_warm": "#FF9D57",
    "orange_soft": "#F78166",
    "green_neon": "#31FF48",
    "green_lime": "#70E000",
    "green_light": "#57FFBC",
    "green_emerald": "#059669",
    "green_dark": "#035100",
    "green_jade": "#00FF7F",
    "cyan_green": "#00F5D4",
    "yellow_orange": "#FFD166",
}

# ======================================================================
# 🎯 Quantity-Specific Comparison Palette (Learned vs. Ground Truth)
# ======================================================================
# Visual semantic rule:
# - Ground Truth / Observed = darker, saturated color (anchored base truth)
# - Learned (PINN) = lighter, luminous tint (high contrast against GT & dark bg)
#
# Quantity mapping:
#   - Potential            -> Purple (Dark Violet vs. Light Lavender)
#   - Wavefunctions        -> Magenta vs. Green (Vibrant Neon Magenta vs. Luminous Green)
#   - Energy Eigenvalues   -> Blue   (Deep Royal Blue vs. Vivid Sky/Cyan Blue)
#   - Probability Densities-> Green vs. Cyan-Green (Vivid Emerald Green vs. Luminous Cyan-Green)

COMPARISON_PALETTE: Final[Dict[str, Dict[str, str]]] = {
    "potential": {
        "true": "#7952F5",       # Vibrant Dark Purple / Violet (Ground Truth)
        "learned": "#D8B4FE",    # Light Lavender / Lilac (Learned PINN)
    },
    "wavefunctions": {
        "true": "#F72585",       # Vibrant Eigenscribe Neon Magenta/Pink (Ground Truth)
        "learned": "#31FF48",    # Luminous Vibrant Green (Learned PINN)
    },
    "energy": {
        "true": "#0070EB",       # Deep Royal Blue (Ground Truth)
        "learned": "#38BDF8",    # Light Vivid Sky / Cyan Blue (Learned PINN)
    },
    "density": {
        "true": "#059669",       # Vivid Emerald Green (Observed / Ground Truth)
        "observed": "#059669",   # Alias for observed data
        "learned": "#00F5D4",    # Luminous Cyan-Green (Learned PINN)
    },
}

# Quantity-specific semantic color constants for direct access:
COLOR_POTENTIAL_TRUE: Final[str] = COMPARISON_PALETTE["potential"]["true"]
COLOR_POTENTIAL_LEARNED: Final[str] = COMPARISON_PALETTE["potential"]["learned"]

COLOR_WAVEFUNCTION_TRUE: Final[str] = COMPARISON_PALETTE["wavefunctions"]["true"]
COLOR_WAVEFUNCTION_LEARNED: Final[str] = COMPARISON_PALETTE["wavefunctions"]["learned"]

COLOR_ENERGY_TRUE: Final[str] = COMPARISON_PALETTE["energy"]["true"]
COLOR_ENERGY_LEARNED: Final[str] = COMPARISON_PALETTE["energy"]["learned"]

COLOR_DENSITY_TRUE: Final[str] = COMPARISON_PALETTE["density"]["true"]
COLOR_DENSITY_OBSERVED: Final[str] = COMPARISON_PALETTE["density"]["observed"]
COLOR_DENSITY_LEARNED: Final[str] = COMPARISON_PALETTE["density"]["learned"]

# General semantic fallbacks & POD design tokens:
COLOR_TRUE: Final[str] = "#FF5376"        # Vibrant Rose/Coral (Default ground truth fallback)
COLOR_LEARNED: Final[str] = "#00E8FF"     # Electric Cyan (Default learned PINN fallback)
COLOR_OBSERVED: Final[str] = "#FF9D57"    # Warm Orange/Amber (Default observed data fallback)
COLOR_POD_MODE: Final[str] = "#00E8FF"    # Electric Cyan for POD spatial modes
COLOR_POD_ALT: Final[str] = "#A855F7"     # Deep Purple for secondary POD / partition weights

# Training Loss Components Palette:
LOSS_COLORS: Final[Dict[str, str]] = {
    "Total": PROJECT_COLORS["purple_deep"],     # #A855F7
    "Physics": PROJECT_COLORS["blue_light"],    # #14B5FF
    "Data-fit": PROJECT_COLORS["pink_vibrant"],  # #FF66B3
    "Smoothness": PROJECT_COLORS["green_jade"], # #00FF7F
    "Ordered": PROJECT_COLORS["orange_warm"],   # #FF9D57
}

# Backwards compatibility / convenient alias for Figure 2 potential styling
FIGURE_2: Final[Dict[str, str]] = {
    "Ground_Truth_Potential": COLOR_POTENTIAL_TRUE,
    "Learned_Potential": COLOR_POTENTIAL_LEARNED,
}


def get_comparison_colors(quantity: str) -> Tuple[str, str]:
    """Return (true_color, learned_color) for a given physical quantity.

    Supported quantities: 'potential', 'wavefunctions' (or 'psi'),
    'energy' (or 'eigenvalues'), 'density' (or 'probability_density').
    """
    key = quantity.lower().strip()
    if key in ("psi", "wavefunction", "wavefunctions", "eigenfunctions"):
        key = "wavefunctions"
    elif key in ("energy", "energies", "eigenvalues", "energy_eigenvalues"):
        key = "energy"
    elif key in ("density", "densities", "prob_density", "probability_density", "probability_densities"):
        key = "density"
    elif key in ("potential", "v"):
        key = "potential"

    palette = COMPARISON_PALETTE.get(key, {"true": COLOR_TRUE, "learned": COLOR_LEARNED})
    return palette["true"], palette["learned"]

# ======================================================================
# 🌈 Perceptually Smooth, Intuitive Colormaps
# ======================================================================

# 1. Sequential Colormap for [0, 1] Overlaps (0 = Orthogonal/Quiet -> 1 = Unit Overlap/Bright Glow)
# Off-diagonal zeros stay dark; diagonals pop in glowing cyan/rose.
spatial_overlap_cmap = mcolors.LinearSegmentedColormap.from_list(
    "spatial_overlap_smooth",
    [
        (0.00, "#0d1117"),  # Dark background (0 overlap = quiet)
        (0.20, "#1c1445"),  # Deep navy-violet
        (0.45, "#4361EE"),  # Royal Indigo
        (0.70, "#7952F5"),  # Electric Purple
        (0.88, "#FF66B3"),  # Vibrant Rose Pink
        (1.00, "#00FFEE"),  # Glowing Electric Cyan (1.0 peak)
    ],
)

# 2. Symmetric Diverging Colormap for [-1, 1] Cross-Overlaps & Modal Matrices
# -1.0 = Vibrant Pink/Rose, 0.0 = Dark Slate Neutral, +1.0 = Electric Cyan
cross_overlap_cmap = mcolors.LinearSegmentedColormap.from_list(
    "cross_overlap_diverging",
    [
        (0.00, "#F72585"),  # -1.0 : Neon Rose Pink
        (0.25, "#7952F5"),  # -0.5 : Electric Purple
        (0.50, "#161b22"),  #  0.0 : Neutral Dark Slate
        (0.75, "#0A95EB"),  # +0.5 : Vivid Sky Blue
        (1.00, "#00FFEE"),  # +1.0 : Bright Electric Cyan
    ],
)

BLUE_TO_PINK = cross_overlap_cmap

# 3. Temporal Modal Composition Colormap
temporal_cmap = cross_overlap_cmap

# 4. Temporal Unitary Overlap Colormap [0, 1]
temporal_overlap_cmap = spatial_overlap_cmap


# ======================================================================
# 🌍 Global Style & Typography Helper
# ======================================================================
def _apply_style() -> None:
    """Set global plotting style with unified fonts, sizing, and colors."""
    font_family = ["Aclonica", "DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"]

    sns.set_theme(
        style="darkgrid",
        context="notebook",
        font_scale=0.9,
        rc={
            "axes.facecolor": THEME_BG,
            "figure.facecolor": THEME_BG,
            "savefig.facecolor": THEME_BG,
            "grid.color": GRID_COLOR,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "axes.edgecolor": BORDER_COLOR,
            "font.family": "sans-serif",
            "font.sans-serif": font_family,
        },
    )

    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": font_family,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "legend.frameon": True,
            "legend.facecolor": CARD_BG,
            "legend.edgecolor": BORDER_COLOR,
            "legend.framealpha": 0.85,
            "lines.linewidth": 2.5,
        }
    )


_apply_style()


# ======================================================================
# ✨ Helpers: Gradient Bars & Loss-Weight Badge
# ======================================================================
def _plot_gradient_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float,
    cmap: mcolors.Colormap,
    label: str | None = None,
) -> None:
    """Draw a bar with a smooth gradient fill (lighter at bottom, darker at top)."""
    n_segments = 40
    segment_height = height / n_segments
    for i in range(n_segments):
        color = cmap(1.0 - i / (n_segments - 1)) if n_segments > 1 else cmap(1.0)
        ax.bar(
            x,
            segment_height,
            width=width,
            bottom=i * segment_height,
            color=color,
            edgecolor="none",
        )
    if label:
        ax.patches[-1].set_label(label)


def _add_lambda_row(
    fig: plt.Figure,
    lambdas: Dict[str, float],
    *,
    ax: plt.Axes | None = None,
) -> None:
    """
    Render loss-weight dictionary as a sleek glassmorphic pill badge.
    Placed consistently below the figure suptitle or axes title.
    """
    lambda_str = "    ".join(rf"$\lambda_{{{k}}} = {v:g}$" for k, v in lambdas.items())

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        _, title_y = suptitle.get_position()
        lambda_y = title_y - 0.05
    else:
        if ax is None:
            ax = fig.axes[0] if fig.axes else None

        if ax is not None:
            bbox = ax.get_position()
            lambda_y = bbox.y1 - 0.03
        else:
            lambda_y = 0.94

    fig.text(
        0.5,
        lambda_y,
        lambda_str,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_PRIMARY,
        bbox=dict(
            boxstyle="round,pad=0.5,rounding_size=0.3",
            facecolor=CARD_BG,
            edgecolor=BORDER_COLOR,
            alpha=0.90,
            linewidth=1.0,
        ),
        transform=fig.transFigure,
    )


# ======================================================================
# 🩵 1️⃣ Training Curves
# ======================================================================
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
    """Render a log-scale line plot of all loss components."""
    _apply_style()

    comps: List[Tuple[str, str, Sequence[float]]] = [
        ("Total", LOSS_COLORS["Total"], total),
        ("Physics", LOSS_COLORS["Physics"], physics),
        ("Data-fit", LOSS_COLORS["Data-fit"], data),
        ("Smoothness", LOSS_COLORS["Smoothness"], smooth),
        ("Ordered", LOSS_COLORS["Ordered"], ordered),
    ]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=THEME_BG)
    for label, color, series in comps:
        sns.lineplot(
            x=epochs,
            y=series,
            ax=ax,
            label=label,
            color=color,
            linewidth=2.5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training Loss Components", fontsize=13, pad=12)
    ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.85)

    _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🌠 2️⃣ Potential plot (true vs. learned)
# ======================================================================
def plot_potential(
    x: torch.Tensor,
    V_true: torch.Tensor,
    V_learned: torch.Tensor,
    lambdas: Dict[str, float],
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot the analytic ground truth potential and the network's prediction."""
    _apply_style()

    x_np = x.squeeze().detach().cpu().numpy()
    Vt_np = V_true.squeeze().detach().cpu().numpy()
    Vl_np = V_learned.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=THEME_BG)

    ax.plot(
        x_np,
        Vt_np,
        label=r"True $V(x)$",
        color=COLOR_POTENTIAL_TRUE,
        linewidth=2.8,
        linestyle="-",
    )
    ax.plot(
        x_np,
        Vl_np,
        label=r"Learned $V_\theta(x)$",
        color=COLOR_POTENTIAL_LEARNED,
        linewidth=2.8,
        linestyle="--",
    )

    # Domain boundary guides
    for edge in (x_np.min(), x_np.max()):
        ax.axvline(edge, color=TEXT_MUTED, lw=1.5, ls=":", alpha=0.5)

    ax.set_xlabel(r"Position $x$", fontsize=11)
    ax.set_ylabel(r"Potential $V(x)$", fontsize=11)
    ax.set_title("Learned vs. Ground Truth Potential", fontsize=13, pad=12)
    ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", framealpha=0.85)

    _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🔱 3️⃣ Wavefunction comparison
# ======================================================================
def plot_wavefunctions(
    x: torch.Tensor,
    psi_true: Sequence[torch.Tensor],
    psi_learned: Sequence[torch.Tensor],
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Side-by-side plot of each eigenmode (learned vs. ground truth)."""
    _apply_style()

    n_modes = len(psi_true)
    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(max(5 * n_modes, 12), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if n_modes == 1:
        axes = [axes]

    x_np = x.squeeze().detach().cpu().numpy()

    for idx, (ax, pt, pl) in enumerate(zip(axes, psi_true, psi_learned)):
        ax.plot(
            x_np,
            pt.squeeze().detach().cpu().numpy(),
            label=rf"True $\psi_{idx}(x)$",
            color=COLOR_WAVEFUNCTION_TRUE,
            linewidth=2.8,
            linestyle="-",
        )
        ax.plot(
            x_np,
            pl.squeeze().detach().cpu().numpy(),
            label=rf"Learned $\psi_{idx}^\theta(x)$",
            color=COLOR_WAVEFUNCTION_LEARNED,
            linewidth=2.8,
            linestyle="--",
        )
        ax.set_xlabel(r"Position $x$", fontsize=11)
        ax.set_title(rf"Mode $n={idx}$", fontsize=12)
        ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    axes[0].set_ylabel(r"Amplitude $\psi(x)$", fontsize=11)
    fig.suptitle(
        f"Learned vs. Ground Truth Wavefunctions ({n_modes} modes)",
        fontsize=15,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🔷 4️⃣ Energy-spectrum comparison
# ======================================================================
def plot_energy_spectrum(
    E_true: torch.Tensor,
    E_learned: torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Bar-chart comparison of the first energy levels (ground truth vs learned)."""
    _apply_style()

    E_true_np = E_true.detach().cpu().numpy()
    E_learn_np = E_learned.detach().cpu().numpy()
    indices = np.arange(len(E_true_np))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THEME_BG)

    ax.grid(
        visible=True,
        which="major",
        axis="y",
        color=GRID_COLOR,
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
        zorder=0,
    )

    bar_width = 0.32
    ax.bar(
        indices - bar_width / 2,
        E_true_np,
        width=bar_width,
        color=COLOR_ENERGY_TRUE,
        edgecolor=BORDER_COLOR,
        linewidth=1.0,
        label="True $E_n$",
        zorder=3,
        alpha=0.90,
    )
    ax.bar(
        indices + bar_width / 2,
        E_learn_np,
        width=bar_width,
        color=COLOR_ENERGY_LEARNED,
        edgecolor=BORDER_COLOR,
        linewidth=1.0,
        label=r"Learned $E_n^\theta$",
        zorder=3,
        alpha=0.90,
    )

    # Numerical value annotations on top of bars
    for i, (yt, yl) in enumerate(zip(E_true_np, E_learn_np)):
        ax.text(
            i - bar_width / 2,
            yt + 0.03 * max(E_true_np.max(), 1.0),
            f"{yt:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_PRIMARY,
        )
        ax.text(
            i + bar_width / 2,
            yl + 0.03 * max(E_learn_np.max(), 1.0),
            f"{yl:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_PRIMARY,
        )

    ax.set_xticks(indices)
    ax.set_xticklabels([rf"$n={i}$" for i in indices], fontsize=10)
    ax.set_ylabel(r"Energy ($\hbar\omega$ units)", fontsize=11)
    ax.set_title("Exact vs. Learned Energy Eigenvalues", fontsize=13, pad=12)
    ax.legend(loc="upper left", framealpha=0.85)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🫟 5️⃣ Probability-density comparison (|psi|^2 vs observed)
# ======================================================================
def plot_density_vs_observed(
    x: torch.Tensor,
    psi_learned: Sequence[torch.Tensor],
    rho_obs: Sequence[torch.Tensor],
    *,
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot learned probability densities |psi_n|^2 alongside observed densities rho_n."""
    _apply_style()

    n_modes = len(psi_learned)
    assert n_modes == len(rho_obs), "❌ Mismatched number of modes."
    x_np = x.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(max(5 * n_modes, 12), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if n_modes == 1:
        axes = [axes]

    for idx, (ax, psi, rho) in enumerate(zip(axes, psi_learned, rho_obs)):
        prob_density = (psi.squeeze().detach().cpu() ** 2).numpy()
        obs_density = rho.squeeze().detach().cpu().numpy()

        ax.plot(
            x_np,
            obs_density,
            label=r"Observed $\rho_n^{\text{obs}}(x)$",
            color=COLOR_DENSITY_OBSERVED,
            linewidth=2.8,
            linestyle="-",
        )
        ax.plot(
            x_np,
            prob_density,
            label=r"Learned $|\hat{\psi}_n^{\theta}(x)|^2$",
            color=COLOR_DENSITY_LEARNED,
            linewidth=2.8,
            linestyle="--",
        )
        ax.set_xlabel(r"Position $x$", fontsize=11)
        ax.set_title(rf"Mode $n={idx}$", fontsize=12)
        ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    axes[0].set_ylabel(r"Probability Density $\rho(x)$", fontsize=11)
    fig.suptitle(
        "Learned vs. Observed Probability Densities",
        fontsize=15,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=axes[0])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 6️⃣ POD singular values (log plot)
# ======================================================================
def plot_pod_singular_values(
    singular_values: torch.Tensor,
    *,
    title: str = "POD Singular Value Spectrum",
    ylabel: str = r"Singular value $\sigma_k$ (log scale)",
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot singular values from POD decomposition on a logarithmic scale."""
    _apply_style()

    sv = singular_values.detach().cpu().numpy()

    # Structured console printout
    print("\nSingular Values:")
    print("-" * 30)
    print(f"{'Mode (k)':>10} | {'Sigma_k':>15}")
    print("-" * 30)
    for i, val in enumerate(sv, start=1):
        print(f"{i:10d} | {val:15.6e}")
    print("-" * 30)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THEME_BG)
    indices = np.arange(1, len(sv) + 1)

    # Connecting line
    ax.semilogy(
        indices,
        sv,
        color=COLOR_POD_MODE,
        linewidth=2.2,
        alpha=0.7,
        zorder=1,
    )

    # Scatter points with glowing markers
    sctr = ax.scatter(
        indices,
        sv,
        c=indices,
        cmap=spatial_overlap_cmap,
        edgecolor=TEXT_PRIMARY,
        linewidth=1.2,
        s=80,
        zorder=3,
        label=r"$\sigma_k$",
    )

    ax.set_yscale("log")

    # Annotate points with formatted values
    for i, val in enumerate(sv, start=1):
        ha = "left" if i == 1 else ("right" if i == len(sv) else "center")
        xytext = (6, -6) if i == 1 else ((-6, 8) if i == len(sv) else (0, 8))

        ax.annotate(
            f"{val:.3e}",
            (i, val),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=CARD_BG,
                edgecolor=BORDER_COLOR,
                alpha=0.85,
            ),
        )

    ax.set_xlabel(r"Mode Index $k$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(indices)
    ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🗺️ 7️⃣ Overlap matrix heatmap (POD diagnostic)
# ======================================================================
def plot_overlap_heatmap(
    psi_theta: Sequence[torch.Tensor],
    dx: float,
    *,
    lambdas: Dict[str, float] | None = None,
    cmap: mcolors.Colormap | str = spatial_overlap_cmap,
    fmt: str = ".2f",
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Render a heatmap of the overlap matrix <psi_m^theta | psi_n^theta>."""
    _apply_style()

    n_modes = len(psi_theta)
    psi_theta_mat = torch.stack([p.squeeze().detach().cpu() for p in psi_theta]).T

    overlap_tensor = mode_overlap_matrix(psi_theta_mat, dx)
    norms = torch.sqrt(torch.diag(overlap_tensor))
    overlap_tensor = overlap_tensor / torch.outer(norms, norms)
    overlap = overlap_tensor.numpy()

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_modes * 1.3), max(4.8, n_modes * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"$n={i}$" for i in range(n_modes)], fontsize=10)
    ax.set_yticklabels([rf"$m={i}$" for i in range(n_modes)], fontsize=10)

    # Clean numeric annotations
    for i in range(n_modes):
        for j in range(n_modes):
            val = overlap[i, j]
            text_color = THEME_BG if val > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        r"Spatial Overlap Matrix $\langle \hat{\psi}_m^\theta \mid \hat{\psi}_n^\theta \rangle$",
        fontsize=13,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overlap Value", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 8️⃣ First three spatial POD modes
# ======================================================================
def plot_pod_first_three_spatial_modes(
    x: torch.Tensor,
    spatial_modes: torch.Tensor,
    *,
    ground_truth: Sequence[torch.Tensor] | None = None,
    psi_learned: Sequence[torch.Tensor] | None = None,
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Plot first spatial POD modes overlaid with ground truth and learned modes."""
    _apply_style()

    x_np = x.squeeze().detach().cpu().numpy()
    modes_np = spatial_modes.detach().cpu().numpy()

    gt_np = (
        [gt.squeeze().detach().cpu().numpy() for gt in ground_truth[: modes_np.shape[1]]]
        if ground_truth is not None
        else []
    )
    psi_learned_np = (
        [pl.squeeze().detach().cpu().numpy() for pl in psi_learned[: modes_np.shape[1]]]
        if psi_learned is not None
        else []
    )

    n_plot = min(3, modes_np.shape[1])
    fig, axs = plt.subplots(
        1,
        n_plot,
        figsize=(max(5 * n_plot, 12), 4.2),
        constrained_layout=True,
    )
    if n_plot == 1:
        axs = [axs]

    for k in range(n_plot):
        ax = axs[k]

        if k < len(gt_np):
            ax.plot(
                x_np,
                gt_np[k],
                label=rf"True $\psi_{k}$",
                color=COLOR_WAVEFUNCTION_TRUE,
                linewidth=2.8,
                linestyle="-",
            )

        if k < len(psi_learned_np):
            ax.plot(
                x_np,
                psi_learned_np[k],
                label=rf"Learned $\hat{{\psi}}_{k}^\theta$",
                color=COLOR_WAVEFUNCTION_LEARNED,
                linewidth=2.8,
                linestyle="--",
            )

        ax.plot(
            x_np,
            modes_np[:, k],
            label=f"POD mode $u_{k}$",
            color=COLOR_POD_MODE,
            linewidth=2.8,
            linestyle="-.",
        )

        ax.set_xlabel(r"Position $x$", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title(f"POD Spatial Mode $k={k}$", fontsize=12)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        ax.grid(True, which="both", color=GRID_COLOR, linestyle=":", alpha=0.6)

    fig.suptitle(
        "POD Spatial Modes vs. Physical Eigenfunctions",
        fontsize=15,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=axs[0])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 9️⃣ Cross-overlap matrix heatmap (POD vs Learned)
# ======================================================================
def plot_cross_overlap_heatmap(
    pod_modes_physical: Sequence[torch.Tensor] | torch.Tensor,
    psi_matrix: Sequence[torch.Tensor] | torch.Tensor,
    dx: float,
    *,
    cmap: mcolors.Colormap | str = cross_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Create a heatmap of cross overlap matrix <u_k | psi_n^theta>."""
    _apply_style()

    cross_overlap = cross_overlap_matrix(pod_modes_physical, psi_matrix, dx=dx)
    n_modes = cross_overlap.shape[0]

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_modes * 1.3), max(4.8, n_modes * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        cross_overlap,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"State $n={i}$" for i in range(n_modes)], rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels([rf"POD $u_{i}$" for i in range(n_modes)], fontsize=9.5)

    for i in range(n_modes):
        for j in range(n_modes):
            val = cross_overlap[i, j]
            text_color = THEME_BG if abs(val) > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        r"Cross-Overlap Matrix $\langle u_k \mid \hat{\psi}_n^\theta \rangle$",
        fontsize=13,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Projection Value", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 🔟 POD–Eigenbasis Alignment Heatmap (POD vs True)
# ======================================================================
def plot_pod_eigen_alignment(
    pod_modes_physical: Sequence[torch.Tensor] | torch.Tensor,
    psi_true_matrix: Sequence[torch.Tensor] | torch.Tensor,
    dx: float,
    *,
    cmap: mcolors.Colormap | str = cross_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Create a heatmap of overlap matrix <u_k | psi_n> with ground truth."""
    _apply_style()

    overlap = cross_overlap_matrix(pod_modes_physical, psi_true_matrix, dx=dx)
    n_modes = overlap.shape[0]

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_modes * 1.3), max(4.8, n_modes * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"True $\psi_{i}$" for i in range(n_modes)], rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels([rf"POD $u_{i}$" for i in range(n_modes)], fontsize=9.5)

    for i in range(n_modes):
        for j in range(n_modes):
            val = overlap[i, j]
            text_color = THEME_BG if abs(val) > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        r"POD Eigen-Alignment $\langle u_k \mid \psi_n \rangle$",
        fontsize=13,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Alignment Value", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣1️⃣ POD Temporal Modes (Modal Composition Matrix V)
# ======================================================================
def plot_pod_temporal_modes(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = cross_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Create a heatmap of modal composition matrix V (right-singular vectors)."""
    _apply_style()

    Vh_np = Vh.detach().cpu().numpy() if isinstance(Vh, torch.Tensor) else Vh
    V = Vh_np.T
    n_states, n_pod_modes = V.shape

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_pod_modes * 1.3), max(4.8, n_states * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        V,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_pod_modes))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([rf"POD $k={i}$" for i in range(n_pod_modes)], rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels([rf"State $n={i}$" for i in range(n_states)], fontsize=9.5)

    for i in range(n_states):
        for j in range(n_pod_modes):
            val = V[i, j]
            text_color = THEME_BG if abs(val) > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(r"Temporal Modal Composition $V_{nk}$", fontsize=13, pad=12)
    ax.set_ylabel("Learned States ($n$)", fontsize=11)
    ax.set_xlabel("POD Modes ($k$)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient Value", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣2️⃣ POD Temporal Overlap Heatmap
# ======================================================================
def plot_pod_temporal_overlap_heatmap(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = spatial_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Render a heatmap of temporal mode overlap matrix <v_m | v_n> (should equal I)."""
    _apply_style()

    Vh_np = Vh.detach().cpu().numpy() if isinstance(Vh, torch.Tensor) else Vh
    V = Vh_np.conj().T
    overlap = np.real(V.conj().T @ V)
    n_modes = overlap.shape[0]

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_modes * 1.3), max(4.8, n_modes * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_modes))
    ax.set_xticklabels([rf"$v_{i}$" for i in range(n_modes)], fontsize=10)
    ax.set_yticklabels([rf"$v_{i}$" for i in range(n_modes)], fontsize=10)

    for i in range(n_modes):
        for j in range(n_modes):
            val = overlap[i, j]
            text_color = THEME_BG if val > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(r"Temporal Mode Overlap $\langle v_m \mid v_n \rangle$", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overlap Value", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣3️⃣ POD Temporal Cross-Overlap Heatmap
# ======================================================================
def plot_pod_temporal_cross_overlap_heatmap(
    Vh: torch.Tensor | np.ndarray,
    *,
    cmap: mcolors.Colormap | str = spatial_overlap_cmap,
    fmt: str = ".2f",
    lambdas: Dict[str, float] | None = None,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Render heatmap of absolute temporal cross-overlap |V_{nk}|."""
    _apply_style()

    Vh_np = Vh.detach().cpu().numpy() if isinstance(Vh, torch.Tensor) else Vh
    V = Vh_np.T
    n_states, n_modes = V.shape

    fig, ax = plt.subplots(
        figsize=(max(5.2, n_modes * 1.3), max(4.8, n_states * 1.2)),
        facecolor=THEME_BG,
    )

    im = ax.imshow(
        np.abs(V),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.grid(False)

    ax.set_xticks(np.arange(n_modes))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([rf"POD $k={i}$" for i in range(n_modes)], rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels([rf"State $n={i}$" for i in range(n_states)], fontsize=9.5)

    for i in range(n_states):
        for j in range(n_modes):
            val = np.abs(V[i, j])
            text_color = THEME_BG if val > 0.65 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        r"Temporal Cross-Overlap $|V_{nk}| = |\langle \mathbf{e}_n \mid v_k \rangle|$",
        fontsize=13,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Overlap", fontsize=10, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_MUTED)

    if lambdas is not None:
        _add_lambda_row(fig, lambdas, ax=ax)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣4️⃣ Hilbert Space Phase Portrait
# ======================================================================
def plot_hilbert_phase_portrait(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    true_wavefunctions: np.ndarray | torch.Tensor,
    x: np.ndarray | torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Project learned wavefunctions onto true eigenstate subspace in 3D."""
    _apply_style()

    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()
    if isinstance(true_wavefunctions, torch.Tensor):
        true_wavefunctions = true_wavefunctions.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.atleast_1d(x).squeeze()
    n_states = min(3, true_wavefunctions.shape[1])
    n_modes = min(3, learned_wavefunctions.shape[1])

    coeffs = np.zeros((n_modes, n_states), dtype=np.float64)
    for i in range(n_modes):
        psi_l = learned_wavefunctions[:, i]
        for j in range(n_states):
            true_psi = true_wavefunctions[:, j]
            coeffs[i, j] = np.trapezoid(psi_l * true_psi, x)

    fig = plt.figure(figsize=(7.5, 7.0), facecolor=THEME_BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=THEME_BG)

    # Styling 3D panes
    ax.xaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))
    ax.yaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))
    ax.zaxis.set_pane_color((0.05, 0.07, 0.09, 1.0))

    for i in range(n_modes):
        x0, y0, z0 = coeffs[i, 0], coeffs[i, 1], coeffs[i, 2]

        ax.plot(
            [x0, x0],
            [y0, y0],
            [0, z0],
            color=COLOR_POD_MODE,
            linewidth=2.5,
            alpha=0.7,
            zorder=2,
        )

        ax.scatter(
            x0,
            y0,
            z0,
            c=COLOR_LEARNED,
            edgecolor=TEXT_PRIMARY,
            linewidth=1.2,
            s=90,
            alpha=0.9,
            zorder=5,
        )

        ax.text(
            x0,
            y0,
            z0 + 0.04,
            f"State {i}\n({x0:.2f}, {y0:.2f}, {z0:.2f})",
            fontsize=8.5,
            color=TEXT_PRIMARY,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=CARD_BG,
                edgecolor=BORDER_COLOR,
                alpha=0.85,
            ),
            zorder=6,
        )

    ax.set_xlabel(r"$\langle \psi_i^\theta \mid \psi_0 \rangle$", fontsize=10, labelpad=8)
    ax.set_ylabel(r"$\langle \psi_i^\theta \mid \psi_1 \rangle$", fontsize=10, labelpad=8)
    ax.set_zlabel(r"$\langle \psi_i^\theta \mid \psi_2 \rangle$", fontsize=10, labelpad=8)
    ax.set_title("Hilbert Space Phase Portrait", fontsize=13, pad=12)

    ax.view_init(elev=25, azim=55)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣5️⃣ Spectral Energy Cascade
# ======================================================================
def plot_spectral_energy_cascade(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    energies: np.ndarray | torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Compare POD singular values with Hamiltonian energy spectrum."""
    _apply_style()

    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()
    if isinstance(energies, torch.Tensor):
        energies = energies.detach().cpu().numpy()

    spatial_modes, singular_values, _ = pod_decomposition(learned_wavefunctions)
    n = min(len(singular_values), len(energies))
    modes = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), facecolor=THEME_BG)

    sns.lineplot(
        x=modes,
        y=singular_values[:n],
        marker="o",
        markersize=7,
        ax=ax1,
        label=r"POD singular values $\sigma_k$",
        color=PROJECT_COLORS["pink_vibrant"],
        linewidth=2.5,
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Mode Index $k$", fontsize=11)
    ax1.set_ylabel(r"Singular Value $\sigma_k$", color=PROJECT_COLORS["pink_vibrant"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=PROJECT_COLORS["pink_vibrant"])
    ax1.set_xticks(modes)

    ax2 = ax1.twinx()
    sns.lineplot(
        x=modes,
        y=energies[:n],
        marker="s",
        markersize=7,
        ax=ax2,
        linestyle="--",
        label=r"Energy $E_k$",
        color=COLOR_ENERGY_LEARNED,
        linewidth=2.5,
    )

    ax2.set_ylabel("Energy (eigenvalues)", color=COLOR_ENERGY_LEARNED, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=COLOR_ENERGY_LEARNED)
    ax2.grid(False)

    ax1.set_title("Spectral Energy Cascade", fontsize=13, pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)
    ax2.get_legend().remove()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 📊 1️⃣6️⃣ POD Partition Function Spectrum
# ======================================================================
def plot_partition_function_spectrum(
    learned_wavefunctions: np.ndarray | torch.Tensor,
    *,
    out_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Interpret POD spectrum as a thermodynamic ensemble."""
    _apply_style()

    if isinstance(learned_wavefunctions, torch.Tensor):
        learned_wavefunctions = learned_wavefunctions.detach().cpu().numpy()

    _, singular_values, _ = pod_decomposition(learned_wavefunctions)
    if isinstance(singular_values, torch.Tensor):
        singular_values = singular_values.detach().cpu().numpy()

    p = singular_values**2
    p = p / np.sum(p)
    E_eff = -np.log(p + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    modes = np.arange(1, len(p) + 1)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), facecolor=THEME_BG)

    sns.lineplot(
        x=modes,
        y=p,
        marker="o",
        markersize=7,
        ax=ax1,
        label="Boltzmann weights $p_k$",
        color=PROJECT_COLORS["purple_deep"],
        linewidth=2.5,
    )

    ax1.set_ylabel("Mode Probability $p_k$", color=PROJECT_COLORS["purple_deep"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=PROJECT_COLORS["purple_deep"])
    ax1.set_xlabel("Mode Index $k$", fontsize=11)
    ax1.set_xticks(modes)

    ax2 = ax1.twinx()
    sns.lineplot(
        x=modes,
        y=E_eff,
        marker="s",
        markersize=7,
        linestyle="--",
        ax=ax2,
        label=r"Effective energy $E_{\text{eff}}$",
        color=PROJECT_COLORS["green_jade"],
        linewidth=2.5,
    )

    ax2.set_ylabel(r"Effective Energy $E_{\text{eff}}$", color=PROJECT_COLORS["green_jade"], fontsize=11)
    ax2.tick_params(axis="y", labelcolor=PROJECT_COLORS["green_jade"])
    ax2.grid(False)

    ax1.set_title(
        f"POD Partition Function Spectrum (Entropy = {entropy:.3f})",
        fontsize=13,
        pad=12,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.85)
    ax2.get_legend().remove()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=THEME_BG)

    return fig


# ======================================================================
# 🧪 Smoke Test & Main Entry Point
# ======================================================================
def _smoke_test() -> None:
    """Generate dummy data and produce all figures."""
    torch.manual_seed(27)

    N = 128
    x = torch.linspace(-5.0, 5.0, N)

    V_true = 0.5 * x**2
    V_learned = V_true + 0.2 * torch.randn_like(V_true)

    psi_true = [torch.sin((i + 1) * x) for i in range(3)]
    psi_learned = [pt + 0.1 * torch.randn_like(pt) for pt in psi_true]

    epochs = list(range(1, 101))
    total = np.exp(-0.03 * np.arange(100)) + 0.02 * np.random.rand(100)
    physics = np.exp(-0.025 * np.arange(100)) + 0.015 * np.random.rand(100)
    ordered = np.exp(-0.04 * np.arange(100)) + 0.008 * np.random.rand(100)
    smooth = np.exp(-0.02 * np.arange(100)) + 0.005 * np.random.rand(100)
    data = np.exp(-0.035 * np.arange(100)) + 0.01 * np.random.rand(100)

    lambdas = {"data": 1.0, "physics": 1.0, "smooth": 1e-2, "ordered": 1.0}

    out_dir = pathlib.Path("./_smoke_outputs")
    out_dir.mkdir(exist_ok=True)

    plot_loss_history(epochs, total, physics, data, smooth, ordered, lambdas, out_path=out_dir / "loss_history.png")
    plot_potential(x, V_true, V_learned, lambdas, out_path=out_dir / "potential.png")
    plot_wavefunctions(x, psi_true, psi_learned, out_path=out_dir / "wavefunctions.png")

    E_true = torch.tensor([0.5, 1.5, 2.5])
    E_learned = E_true + 0.1 * torch.randn_like(E_true)
    plot_energy_spectrum(E_true=E_true, E_learned=E_learned, out_path=out_dir / "energy.png")

    rho_obs = [(pt.squeeze() ** 2 + 0.02 * torch.rand_like(pt.squeeze())) for pt in psi_true]
    plot_density_vs_observed(x=x, psi_learned=psi_learned, rho_obs=rho_obs, lambdas=lambdas, out_path=out_dir / "density_vs_observed.png")

    psi_matrix = torch.stack(psi_learned, dim=1)
    dx = float(x[1] - x[0])

    pod_modes_physical, S, Vh_dummy, pod_modes_euclidean = physical_pod_decomposition(
        psi_matrix,
        dx,
        reference_modes=psi_matrix,
        align_signs=True,
    )

    plot_pod_singular_values(singular_values=S, out_path=out_dir / "pod_singular_values.png")
    plot_pod_first_three_spatial_modes(x=x, spatial_modes=pod_modes_physical, ground_truth=psi_true, lambdas=lambdas, out_path=out_dir / "pod_modes.png")
    plot_overlap_heatmap(psi_theta=psi_learned, dx=dx, lambdas=lambdas, out_path=out_dir / "overlap_heatmap.png")
    plot_cross_overlap_heatmap(pod_modes_physical=pod_modes_physical, psi_matrix=psi_matrix, dx=dx, lambdas=lambdas, out_path=out_dir / "cross_overlap_heatmap.png")
    plot_pod_temporal_modes(Vh=Vh_dummy, lambdas=lambdas, out_path=out_dir / "pod_temporal_modes.png")
    plot_pod_temporal_overlap_heatmap(Vh=Vh_dummy, lambdas=lambdas, out_path=out_dir / "pod_temporal_overlap.png")
    plot_pod_temporal_cross_overlap_heatmap(Vh=Vh_dummy, lambdas=lambdas, out_path=out_dir / "pod_temporal_cross_overlap.png")
    plot_pod_eigen_alignment(pod_modes_physical, torch.stack(psi_true, dim=0).T, dx=dx, lambdas=lambdas, out_path=out_dir / "pod_eigen_alignment.png")

    psi_true_stacked = torch.stack(psi_true, dim=1)
    plot_hilbert_phase_portrait(learned_wavefunctions=psi_matrix, true_wavefunctions=psi_true_stacked, x=x, out_path=out_dir / "hilbert_portrait.png")
    plot_spectral_energy_cascade(learned_wavefunctions=psi_matrix, energies=E_true, out_path=out_dir / "spectral_cascade.png")
    plot_partition_function_spectrum(learned_wavefunctions=psi_matrix, out_path=out_dir / "partition_spectrum.png")

    print(f"\n[OK] Smoke test complete. All {len(list(out_dir.iterdir()))} figures written to {out_dir.resolve()}\n")


def main() -> None:
    """Entry point for ``python -m src.visualizations``."""
    _smoke_test()


if __name__ == "__main__":
    main()
