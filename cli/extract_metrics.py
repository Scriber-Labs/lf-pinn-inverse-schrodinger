# src/extract_metrics.py
"""
extract_metrics.py

Standalone script to extract POD and training analysis metrics from PIML artifact files.

NO EXTERNAL DEPENDENCIES beyond: argparse, json, numpy, pandas, torch, pathlib

Usage:
    python extract_metrics.py --artifacts-dir /path/to/artifacts --output-dir /path/to/output

This script reads:
    - config.json
    - diagnostics.npz
    - ground_truth.pt
    - history.json
    - model.pt (optional)

And outputs:
    - pod_metrics.csv
    - training_analysis.csv

Author: Eigenscribe
Date: 02-2026
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Dict, Any
import sys


# ============================================================================
# POD UTILITIES (from src/pod.py)
# ============================================================================

def mode_overlap_matrix(psi_matrix: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the discrete overlap matrix <psi_m | psi_n> on a uniform grid.
    Uses trapezoidal rule for integration.
    """
    N = psi_matrix.shape[0]
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    weighted = psi_matrix * weights.reshape(-1, 1)
    return weighted.T @ psi_matrix * dx


def cross_overlap_matrix(psi_A: np.ndarray, psi_B: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute <psi_A_m | psi_B_n>.
    """
    N = psi_A.shape[0]
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    weighted_A = psi_A * weights.reshape(-1, 1)
    return weighted_A.T @ psi_B * dx


# ============================================================================
# LOGGING UTILITIES (simple, no external deps)
# ============================================================================

class SimpleLogger:
    """Minimal logger with emoji output."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def info(self, msg: str):
        if self.verbose:
            print(f"ℹ️  {msg}")

    def success(self, msg: str):
        if self.verbose:
            print(f"✅ {msg}")

    def warning(self, msg: str):
        if self.verbose:
            print(f"⚠️  {msg}")

    def error(self, msg: str):
        print(f"❌ {msg}", file=sys.stderr)


# ============================================================================
# METRIC EXTRACTION FUNCTIONS
# ============================================================================

def extract_pod_metrics(
        config: Dict[str, Any],
        diagnostics: Dict[str, np.ndarray],
        ground_truth: Dict[str, torch.Tensor],
        logger: SimpleLogger,
) -> pd.DataFrame:
    """
    Extract POD analysis metrics and return as DataFrame.
    """
    data = []

    # Get spatial grid spacing
    dx = float(config.get("dx"))

    # Get learned wavefunctions
    psi_learned = diagnostics["psi_learned"]
    if isinstance(psi_learned, list):
        psi_learned = np.array(psi_learned)

    # Convert to torch for SVD
    psi_learned_torch = torch.from_numpy(psi_learned).float()

    # ===== POD DECOMPOSITION =====
    U, S, Vh = torch.linalg.svd(psi_learned_torch, full_matrices=False)
    U_np = U.numpy()
    S_np = S.numpy()
    Vh_np = Vh.numpy()

    # POD shapes and statistics
    data.append({"Category": "POD Decomposition", "Parameter": "Spatial Modes (U) Shape", "Value": str(U_np.shape),
                 "Type": "array"})
    data.append({"Category": "POD Decomposition", "Parameter": "Singular Values (S) Shape", "Value": str(S_np.shape),
                 "Type": "array"})
    data.append(
        {"Category": "POD Decomposition", "Parameter": "Modal Coefficients (Vh) Shape", "Value": str(Vh_np.shape),
         "Type": "array"})

    # Singular value statistics
    data.append(
        {"Category": "POD Decomposition", "Parameter": "Total Energy (sum S²)", "Value": float(np.sum(S_np ** 2)),
         "Type": "float"})
    data.append({"Category": "POD Decomposition", "Parameter": "Max Singular Value", "Value": float(np.max(S_np)),
                 "Type": "float"})
    data.append({"Category": "POD Decomposition", "Parameter": "Min Singular Value", "Value": float(np.min(S_np)),
                 "Type": "float"})
    data.append({"Category": "POD Decomposition", "Parameter": "Mean Singular Value", "Value": float(np.mean(S_np)),
                 "Type": "float"})
    data.append({"Category": "POD Decomposition", "Parameter": "Std Singular Value", "Value": float(np.std(S_np)),
                 "Type": "float"})

    # Energy distribution
    total_energy = np.sum(S_np ** 2)
    for i in range(len(S_np)):
        cumsum_energy = np.sum(S_np[:i + 1] ** 2) / total_energy * 100
        data.append({"Category": "POD Decomposition", "Parameter": f"Cumulative Energy (First {i + 1} Mode(s))",
                     "Value": f"{cumsum_energy:.2f}%", "Type": "percentage"})
        data.append(
            {"Category": "POD Decomposition", "Parameter": f"σ_{i + 1}", "Value": float(S_np[i]), "Type": "float"})

    # ===== ORTHONORMALITY CHECK =====
    ortho_check = U_np.T @ U_np
    ortho_error = np.sum(np.abs(ortho_check - np.eye(U_np.shape[1])))
    data.append(
        {"Category": "POD Orthonormality", "Parameter": "U^T U Orthonormality Error", "Value": float(ortho_error),
         "Type": "float"})

    # ===== MODE OVERLAP MATRIX =====
    overlap_learned = mode_overlap_matrix(psi_learned, dx)
    data.append(
        {"Category": "Mode Overlap", "Parameter": "Learned Overlap Matrix Shape", "Value": str(overlap_learned.shape),
         "Type": "array"})

    overlap_error = np.sum(np.abs(overlap_learned - np.eye(overlap_learned.shape[0])))
    data.append({"Category": "Mode Overlap", "Parameter": "Learned Orthonormality Error (Overlap)",
                 "Value": float(overlap_error), "Type": "float"})

    # Diagonal and off-diagonal statistics
    diag_vals = np.diag(overlap_learned)
    off_diag_vals = overlap_learned[np.triu_indices_from(overlap_learned, k=1)]

    data.append(
        {"Category": "Mode Overlap", "Parameter": "Diagonal Mean (should be ~1)", "Value": float(np.mean(diag_vals)),
         "Type": "float"})
    data.append(
        {"Category": "Mode Overlap", "Parameter": "Diagonal Std", "Value": float(np.std(diag_vals)), "Type": "float"})
    data.append({"Category": "Mode Overlap", "Parameter": "Off-Diagonal Mean (should be ~0)",
                 "Value": float(np.mean(off_diag_vals)), "Type": "float"})
    data.append(
        {"Category": "Mode Overlap", "Parameter": "Off-Diagonal Max Abs", "Value": float(np.max(np.abs(off_diag_vals))),
         "Type": "float"})

    # ===== CROSS OVERLAP =====
    cross_overlap = cross_overlap_matrix(U_np, psi_learned, dx)
    data.append({"Category": "Cross Overlap", "Parameter": "POD Modes vs Learned Wavefunctions Shape",
                 "Value": str(cross_overlap.shape), "Type": "array"})

    cross_overlap_abs = np.abs(cross_overlap)
    data.append({"Category": "Cross Overlap", "Parameter": "Max Absolute Cross Overlap",
                 "Value": float(np.max(cross_overlap_abs)), "Type": "float"})
    data.append({"Category": "Cross Overlap", "Parameter": "Mean Absolute Cross Overlap",
                 "Value": float(np.mean(cross_overlap_abs)), "Type": "float"})

    # Dominant alignments
    for i in range(cross_overlap.shape[0]):
        max_idx = np.argmax(np.abs(cross_overlap[i, :]))
        max_val = cross_overlap[i, max_idx]
        data.append({"Category": "Cross Overlap", "Parameter": f"POD Mode {i} Best Alignment",
                     "Value": f"ψ_learned[{max_idx}] = {max_val:.4f}", "Type": "str"})

    # ===== GROUND TRUTH COMPARISON =====
    psi_true = ground_truth["psi_true"]
    if isinstance(psi_true, torch.Tensor):
        psi_true = psi_true.numpy()
    elif isinstance(psi_true, list):
        psi_true = np.array(psi_true)

    overlap_true_learned = cross_overlap_matrix(psi_true, psi_learned, dx)
    data.append({"Category": "Ground Truth Comparison", "Parameter": "True vs Learned Overlap Shape",
                 "Value": str(overlap_true_learned.shape), "Type": "array"})

    overlap_true_learned_abs = np.abs(overlap_true_learned)
    data.append({"Category": "Ground Truth Comparison", "Parameter": "Max True-Learned Overlap",
                 "Value": float(np.max(overlap_true_learned_abs)), "Type": "float"})
    data.append({"Category": "Ground Truth Comparison", "Parameter": "Mean True-Learned Overlap",
                 "Value": float(np.mean(overlap_true_learned_abs)), "Type": "float"})

    for i in range(overlap_true_learned.shape[0]):
        max_idx = np.argmax(np.abs(overlap_true_learned[i, :]))
        max_val = overlap_true_learned[i, max_idx]
        data.append({"Category": "Ground Truth Comparison", "Parameter": f"ψ_true[{i}] Best Match",
                     "Value": f"ψ_learned[{max_idx}] = {max_val:.4f}", "Type": "str"})

    # ===== FAILURE MODE INDICATORS =====
    sv_ratio = S_np[0] / S_np[-1] if len(S_np) > 1 else 1.0
    data.append({"Category": "Failure Modes", "Parameter": "Singular Value Ratio (σ_1/σ_n)", "Value": float(sv_ratio),
                 "Type": "float"})

    if sv_ratio > 10:
        data.append({"Category": "Failure Modes", "Parameter": "Mode Collapse Risk", "Value": "HIGH (ratio > 10)",
                     "Type": "str"})
    elif sv_ratio > 3:
        data.append({"Category": "Failure Modes", "Parameter": "Mode Collapse Risk", "Value": "MODERATE (ratio 3-10)",
                     "Type": "str"})
    else:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Mode Collapse Risk", "Value": "LOW (ratio < 3)", "Type": "str"})

    if overlap_error > 0.5:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Orthonormality Violation", "Value": "SEVERE", "Type": "str"})
    elif overlap_error > 0.1:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Orthonormality Violation", "Value": "MODERATE", "Type": "str"})
    else:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Orthonormality Violation", "Value": "MINIMAL", "Type": "str"})

    max_cross = np.max(cross_overlap_abs)
    if max_cross < 0.7:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Mode Mixing Risk", "Value": "HIGH (max cross-overlap < 0.7)",
             "Type": "str"})
    elif max_cross < 0.9:
        data.append({"Category": "Failure Modes", "Parameter": "Mode Mixing Risk",
                     "Value": "MODERATE (max cross-overlap 0.7-0.9)", "Type": "str"})
    else:
        data.append(
            {"Category": "Failure Modes", "Parameter": "Mode Mixing Risk", "Value": "LOW (max cross-overlap > 0.9)",
             "Type": "str"})

    return pd.DataFrame(data)


def extract_training_analysis(
        config: Dict[str, Any],
        diagnostics: Dict[str, np.ndarray],
        ground_truth: Dict[str, torch.Tensor],
        history: list,
        model_state: Dict[str, torch.Tensor] = None,
        logger: SimpleLogger = None,
) -> pd.DataFrame:
    """
    Extract training and analysis metrics and return as DataFrame.
    """
    data = []

    # ===== METADATA =====
    data.append({"Category": "Metadata", "Parameter": "Device", "Value": config.get("device"), "Type": "str"})
    data.append({"Category": "Metadata", "Parameter": "Random Seed", "Value": config.get("seed"), "Type": "int"})

    # ===== TRAINING HYPERPARAMETERS =====
    data.append({"Category": "Training Hyperparameters", "Parameter": "Learning Rate", "Value": config.get("lr"),
                 "Type": "float"})
    data.append(
        {"Category": "Training Hyperparameters", "Parameter": "Epochs", "Value": config.get("epochs"), "Type": "int"})
    data.append(
        {"Category": "Training Hyperparameters", "Parameter": "Log Every N Epochs", "Value": config.get("log_every"),
         "Type": "int"})

    # ===== MODEL ARCHITECTURE =====
    data.append({"Category": "Model Architecture", "Parameter": "Number of Modes", "Value": config.get("n_modes"),
                 "Type": "int"})
    data.append({"Category": "Model Architecture", "Parameter": "Hidden Layer Size", "Value": config.get("hidden"),
                 "Type": "int"})

    # ===== NUMERICAL GRID =====
    data.append(
        {"Category": "Numerical Grid", "Parameter": "Grid Points", "Value": config.get("n_points"), "Type": "int"})
    data.append(
        {"Category": "Numerical Grid", "Parameter": "Grid Spacing (dx)", "Value": config.get("dx"), "Type": "float"})

    # ===== LOSS WEIGHTS =====
    lambdas = config.get("lambdas", {})
    for loss_name, weight in lambdas.items():
        data.append({"Category": "Loss Weights", "Parameter": f"λ_{loss_name}", "Value": weight, "Type": "float"})

    # ===== GROUND TRUTH DATA =====
    x = ground_truth["x"]
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    elif isinstance(x, list):
        x = np.array(x)
    data.append({"Category": "Ground Truth", "Parameter": "Spatial Grid Shape", "Value": str(x.shape), "Type": "array"})
    data.append({"Category": "Ground Truth", "Parameter": "Domain Min", "Value": float(np.min(x)), "Type": "float"})
    data.append({"Category": "Ground Truth", "Parameter": "Domain Max", "Value": float(np.max(x)), "Type": "float"})

    V_true = ground_truth["V_true"]
    if isinstance(V_true, torch.Tensor):
        V_true = V_true.numpy()
    elif isinstance(V_true, list):
        V_true = np.array(V_true)
    data.append({"Category": "Ground Truth", "Parameter": "V_true Shape", "Value": str(V_true.shape), "Type": "array"})
    data.append(
        {"Category": "Ground Truth", "Parameter": "V_true Min", "Value": float(np.min(V_true)), "Type": "float"})
    data.append(
        {"Category": "Ground Truth", "Parameter": "V_true Max", "Value": float(np.max(V_true)), "Type": "float"})
    data.append(
        {"Category": "Ground Truth", "Parameter": "V_true Mean", "Value": float(np.mean(V_true)), "Type": "float"})

    psi_true = ground_truth["psi_true"]
    if isinstance(psi_true, torch.Tensor):
        psi_true = psi_true.numpy()
    elif isinstance(psi_true, list):
        psi_true = np.array(psi_true)
    data.append(
        {"Category": "Ground Truth", "Parameter": "ψ_true Shape", "Value": str(psi_true.shape), "Type": "array"})

    E_true = ground_truth["E_true"]
    if isinstance(E_true, torch.Tensor):
        E_true = E_true.numpy()
    elif isinstance(E_true, list):
        E_true = np.array(E_true)
    data.append({"Category": "Ground Truth", "Parameter": "E_true Shape", "Value": str(E_true.shape), "Type": "array"})
    for i, E in enumerate(E_true):
        data.append({"Category": "Ground Truth", "Parameter": f"E_true[{i}]", "Value": float(E), "Type": "float"})

    # ===== LEARNED OUTPUTS =====
    E_learned = diagnostics["E_learned"]
    if isinstance(E_learned, list):
        E_learned = np.array(E_learned)
    data.append(
        {"Category": "Learned Outputs", "Parameter": "E_learned Shape", "Value": str(E_learned.shape), "Type": "array"})
    for i, E in enumerate(E_learned):
        data.append({"Category": "Learned Outputs", "Parameter": f"E_learned[{i}]", "Value": float(E), "Type": "float"})

    V_learned = diagnostics["V_learned"]
    if isinstance(V_learned, list):
        V_learned = np.array(V_learned)
    data.append(
        {"Category": "Learned Outputs", "Parameter": "V_learned Shape", "Value": str(V_learned.shape), "Type": "array"})
    data.append({"Category": "Learned Outputs", "Parameter": "V_learned Min", "Value": float(np.min(V_learned)),
                 "Type": "float"})
    data.append({"Category": "Learned Outputs", "Parameter": "V_learned Max", "Value": float(np.max(V_learned)),
                 "Type": "float"})
    data.append({"Category": "Learned Outputs", "Parameter": "V_learned Mean", "Value": float(np.mean(V_learned)),
                 "Type": "float"})

    psi_learned = diagnostics["psi_learned"]
    if isinstance(psi_learned, list):
        psi_learned = np.array(psi_learned)
    data.append({"Category": "Learned Outputs", "Parameter": "ψ_learned Shape", "Value": str(psi_learned.shape),
                 "Type": "array"})

    # ===== ERROR METRICS =====
    abs_errors = np.abs(E_learned - E_true)
    rel_errors = np.abs(E_learned - E_true) / np.abs(E_true) * 100

    data.append({"Category": "Error Metrics", "Parameter": "Energy Errors (Absolute)",
                 "Value": str([f"{e:.6f}" for e in abs_errors]), "Type": "array"})
    data.append({"Category": "Error Metrics", "Parameter": "Energy Errors (Relative %)",
                 "Value": str([f"{e:.4f}%" for e in rel_errors]), "Type": "array"})
    data.append(
        {"Category": "Error Metrics", "Parameter": "Mean Absolute Energy Error", "Value": float(np.mean(abs_errors)),
         "Type": "float"})

    mse_V = float(np.mean((V_learned - V_true) ** 2))
    mae_V = float(np.mean(np.abs(V_learned - V_true)))
    data.append({"Category": "Error Metrics", "Parameter": "Potential MSE", "Value": mse_V, "Type": "float"})
    data.append({"Category": "Error Metrics", "Parameter": "Potential MAE", "Value": mae_V, "Type": "float"})

    # ===== TRAINING HISTORY =====
    history_df = pd.DataFrame(history)
    data.append(
        {"Category": "Training History", "Parameter": "Total Epochs Logged", "Value": len(history_df), "Type": "int"})

    # Loss statistics at different training stages
    for stage_name, stage_epochs in [("Early (0-10%)", slice(0, int(len(history_df) * 0.1))),
                                     ("Mid (40-50%)", slice(int(len(history_df) * 0.4), int(len(history_df) * 0.5))),
                                     ("Late (90-100%)", slice(int(len(history_df) * 0.9), len(history_df)))]:
        stage_data = history_df.iloc[stage_epochs]

        if "total_loss" in stage_data.columns:
            data.append({"Category": "Training History", "Parameter": f"Total Loss {stage_name} (Mean)",
                         "Value": float(stage_data["total_loss"].mean()), "Type": "float"})
        if "physics_loss" in stage_data.columns:
            data.append({"Category": "Training History", "Parameter": f"Physics Loss {stage_name} (Mean)",
                         "Value": float(stage_data["physics_loss"].mean()), "Type": "float"})
        if "data_loss" in stage_data.columns:
            data.append({"Category": "Training History", "Parameter": f"Data Loss {stage_name} (Mean)",
                         "Value": float(stage_data["data_loss"].mean()), "Type": "float"})
        if "smooth_loss" in stage_data.columns:
            data.append({"Category": "Training History", "Parameter": f"Smoothness Loss {stage_name} (Mean)",
                         "Value": float(stage_data["smooth_loss"].mean()), "Type": "float"})

    # Final loss values
    final_epoch = history_df.iloc[-1]
    data.append({"Category": "Training History", "Parameter": "Final Total Loss",
                 "Value": float(final_epoch.get("total_loss", np.nan)), "Type": "float"})
    data.append({"Category": "Training History", "Parameter": "Final Physics Loss",
                 "Value": float(final_epoch.get("physics_loss", np.nan)), "Type": "float"})
    data.append({"Category": "Training History", "Parameter": "Final Data Loss",
                 "Value": float(final_epoch.get("data_loss", np.nan)), "Type": "float"})
    data.append({"Category": "Training History", "Parameter": "Final Smoothness Loss",
                 "Value": float(final_epoch.get("smooth_loss", np.nan)), "Type": "float"})

    # Loss reduction
    if "total_loss" in history_df.columns:
        initial_loss = history_df["total_loss"].iloc[0]
        final_loss = history_df["total_loss"].iloc[-1]
        reduction = (initial_loss - final_loss) / initial_loss * 100
        data.append({"Category": "Training History", "Parameter": "Total Loss Reduction (%)", "Value": float(reduction),
                     "Type": "float"})

    # ===== MODEL PARAMETERS (optional) =====
    if model_state is not None:
        data.append({"Category": "Model Parameters", "Parameter": "Total Parameter Tensors", "Value": len(model_state),
                     "Type": "int"})

        total_params = 0
        for param_name, param_tensor in model_state.items():
            if isinstance(param_tensor, torch.Tensor):
                param_size = param_tensor.numel()
                total_params += param_size
                data.append({"Category": "Model Parameters", "Parameter": f"{param_name} Shape",
                             "Value": str(tuple(param_tensor.shape)), "Type": "array"})
                data.append({"Category": "Model Parameters", "Parameter": f"{param_name} Count", "Value": param_size,
                             "Type": "int"})
                data.append({"Category": "Model Parameters", "Parameter": f"{param_name} Mean",
                             "Value": float(param_tensor.mean()), "Type": "float"})

        data.append(
            {"Category": "Model Parameters", "Parameter": "Total Parameters", "Value": total_params, "Type": "int"})

    return pd.DataFrame(data)


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_all_metrics(
        artifacts_dir: str,
        output_dir: str = None,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all artifacts from a directory and extract metrics.

    Parameters
    ----------
    artifacts_dir : str
        Path to directory containing artifact files.
    output_dir : str, optional
        Path to save CSV files. If None, uses artifacts_dir.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pod_df : pd.DataFrame
        POD metrics.
    training_df : pd.DataFrame
        Training analysis metrics.
    """
    logger = SimpleLogger(verbose=verbose)

    artifacts_dir = Path(artifacts_dir)
    if output_dir is None:
        output_dir = artifacts_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading artifacts from: {artifacts_dir}")

    # Load config
    config_path = artifacts_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"Loaded config.json")

    # Load diagnostics
    diagnostics_path = artifacts_dir / "diagnostics.npz"
    diagnostics = np.load(diagnostics_path, allow_pickle=True)
    logger.info(f"Loaded diagnostics.npz")

    # Load ground truth
    ground_truth_path = artifacts_dir / "ground_truth.pt"
    ground_truth = torch.load(ground_truth_path, map_location="cpu")
    logger.info(f"Loaded ground_truth.pt")

    # Load history
    history_path = artifacts_dir / "history.json"
    with open(history_path, "r") as f:
        history = json.load(f)
    logger.info(f"Loaded history.json ({len(history)} epochs)")

    # Load model (optional)
    model_state = None
    model_path = artifacts_dir / "model.pt"
    if model_path.exists():
        model_state = torch.load(model_path, map_location="cpu")
        logger.info(f"Loaded model.pt ({len(model_state)} parameters)")

    # Extract metrics
    logger.info(f"Extracting POD metrics...")
    pod_df = extract_pod_metrics(config, diagnostics, ground_truth, logger)

    logger.info(f"Extracting training analysis metrics...")
    training_df = extract_training_analysis(config, diagnostics, ground_truth, history, model_state, logger)

    # Save to CSV
    pod_csv = output_dir / "pod_metrics.csv"
    training_csv = output_dir / "training_analysis.csv"

    pod_df.to_csv(pod_csv, index=False)
    training_df.to_csv(training_csv, index=False)

    logger.success(f"Extraction complete!")
    logger.info(f"POD metrics: {pod_csv} ({len(pod_df)} rows)")
    logger.info(f"Training analysis: {training_csv} ({len(training_df)} rows)")

    return pod_df, training_df


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract POD and training metrics from PIML artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_metrics.py --artifacts-dir ./artifacts
  python extract_metrics.py --artifacts-dir ./run_001 --output-dir ./metrics
  python extract_metrics.py --artifacts-dir ./artifacts --quiet
        """
    )

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Path to directory containing artifact files (config.json, diagnostics.npz, etc.)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save CSV files. Defaults to artifacts-dir.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )

    args = parser.parse_args()

    try:
        extract_all_metrics(
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger = SimpleLogger(verbose=True)
        logger.error(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def l2_inner_product(
    f: torch.Tensor,
    g: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """
    Compute the discrete L2 inner product ``<f|g>`` using the trapezoidal rule.

    Parameters
    ----------
    f, g : torch.Tensor, shape ``(N, 1)``
        Function evaluated on the same grid.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    torch.Tensor, shape ``(N, 1)``
        Approximation of ``int(f(x)*g(x)*dx)``.=
    """
    # Trapezoidal rule reduces to a simple sum since the grid is uniform.
    return torch.sum(f * g) * dx


if __name__ == "__main__":
    main()