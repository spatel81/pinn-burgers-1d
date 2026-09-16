#!/usr/bin/env python3
"""
PINN Validation and Plotting — Post-Processing for C++ Inference
=================================================================

Validates the output of the C++ / LibTorch inference binary against a
high-resolution finite-difference reference solution, and produces the
plots that the main branch's pinn_burgers.py generates in-process.

This script is the final stage of the C++ inference pipeline:

    [1] Python (export_model.py):  train → export TorchScript .pt
    [2] C++ (pinn_inference):      load .pt → inference on (x, t) grid
    [3] Python (this script):      reference solution → error → plots

The important distinction from the main branch: the predictions checked
here are read from inference_results.csv, i.e. they are what LibTorch
actually computed in C++ — not what the Python model would have
predicted.  That exercises the whole chain (training → torch.jit.trace
→ torch::jit::load → forward), so a faulty trace or a mistake in the
C++ grid construction shows up as a bad L2 error.

Reference solution:
    Method of lines — central finite differences in x, then scipy's
    Radau implicit integrator in t.  Identical to the main branch.

Inputs (defaults relative to --output-dir):
    inference_results.csv   written by the C++ binary   (required)
    training_log.csv        written by export_model.py  (optional)

Outputs:
    pinn_vs_reference.png   PINN vs reference at each time slice
    training_loss.png       loss curves (PDE, IC, BC) — needs the log
    validation_errors.csv   per-time-slice and overall error metrics

Usage:
    python export/validate_and_plot.py --output-dir build/output
    python export/validate_and_plot.py --results-csv run1/inference_results.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

# Use the non-interactive "Agg" backend so matplotlib writes directly
# to image files without needing a display server.  This is essential
# on Aurora compute nodes (no screen) and harmless on a laptop.
# Must be set BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Physics constants come from the training script rather than being
# redefined here, so the reference solution cannot drift out of sync
# with the equation the network was actually trained on.
#
# That import pulls in torch, which this script does not otherwise need
# — it works in numpy/scipy alone.  Inside the pipeline torch is always
# present, but validating an existing CSV on a machine without it is a
# perfectly reasonable thing to do, so fall back to local copies of the
# constants rather than failing.  Keep the two in sync if the physics
# ever changes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from export_model import NU, X_MIN, X_MAX, T_MIN, T_MAX  # noqa: E402
except ImportError:
    NU = 0.01 / np.pi             # Kinematic viscosity (≈ 0.00318)
    X_MIN, X_MAX = -1.0, 1.0      # Spatial domain
    T_MIN, T_MAX =  0.0, 1.0      # Time domain
    print("  NOTE: torch unavailable — using local copies of the "
          "physics constants.")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ── Reference solution ───────────────────────────────────────────────
REF_NX = 1024                 # Spatial grid points for the FD reference solver

# ── Plotting ─────────────────────────────────────────────────────────
MAX_COLS = 5                  # Subplots per row in the comparison figure
DPI = 150


# ═══════════════════════════════════════════════════════════════════════
# COMMAND-LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields: output_dir, results_csv,
        training_log, ref_nx.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate C++ PINN inference output against a finite-difference "
            "reference solution and generate plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="build/output",
        help=(
            "Directory holding the inference results and receiving the "
            "plots (default: build/output)"
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to inference_results.csv (default: <output-dir>/inference_results.csv)",
    )
    parser.add_argument(
        "--training-log",
        type=str,
        default=None,
        help="Path to training_log.csv (default: <output-dir>/training_log.csv)",
    )
    parser.add_argument(
        "--ref-nx",
        type=int,
        default=REF_NX,
        help=f"Spatial grid points for the reference solver (default: {REF_NX})",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# READING THE C++ INFERENCE OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def read_inference_csv(path):
    """Read the C++ inference CSV into a rectangular (t, x) grid.

    The file has columns x, t, u_pred with one row per evaluation point,
    written time-slice by time-slice by run_inference() in src/main.cpp.

    Parameters
    ----------
    path : str
        Path to inference_results.csv.

    Returns
    -------
    x_eval : np.ndarray, shape (nx,)
        Unique spatial coordinates, ascending.
    t_eval : np.ndarray, shape (nt,)
        Unique time values, ascending.
    u_pinn : np.ndarray, shape (nt, nx)
        Predicted u at each (t, x) pair.

    Raises
    ------
    SystemExit
        If the file is missing, malformed, or not a complete rectangular
        grid (every x must appear at every t, exactly once).
    """
    if not os.path.isfile(path):
        sys.exit(
            f"  ERROR: Inference results not found: {path}\n"
            f"         Run the C++ binary first (./build/pinn_inference)."
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        sys.exit(f"  ERROR: {path} contains no data rows.")
    if data.shape[1] != 3:
        sys.exit(
            f"  ERROR: {path} has {data.shape[1]} columns, expected 3 "
            f"(x, t, u_pred)."
        )

    x_all, t_all, u_all = data[:, 0], data[:, 1], data[:, 2]

    # np.unique returns sorted values, which is what the reference
    # solver and the plots both want.
    x_eval = np.unique(x_all)
    t_eval = np.unique(t_all)

    if x_eval.size * t_eval.size != x_all.size:
        sys.exit(
            f"  ERROR: {path} is not a rectangular grid — "
            f"{x_eval.size} unique x × {t_eval.size} unique t "
            f"≠ {x_all.size} rows."
        )

    # Scatter the rows into the (t, x) grid.  NaN-filled first so a
    # duplicated point (which would leave a hole elsewhere) is caught.
    x_index = {v: i for i, v in enumerate(x_eval)}
    t_index = {v: j for j, v in enumerate(t_eval)}

    u_pinn = np.full((t_eval.size, x_eval.size), np.nan)
    for x_val, t_val, u_val in zip(x_all, t_all, u_all):
        u_pinn[t_index[t_val], x_index[x_val]] = u_val

    if np.isnan(u_pinn).any():
        sys.exit(
            f"  ERROR: {path} has duplicate or missing (x, t) points — "
            f"the grid could not be filled."
        )

    # The reference solver integrates over [T_MIN, T_MAX]; anything
    # outside that window cannot be validated.
    if t_eval[0] < T_MIN or t_eval[-1] > T_MAX:
        sys.exit(
            f"  ERROR: time values in {path} span "
            f"[{t_eval[0]:.4f}, {t_eval[-1]:.4f}], outside the "
            f"training domain [{T_MIN}, {T_MAX}]."
        )

    print(f"  Read {x_all.size:,} points from {path}")
    print(f"  Grid: {x_eval.size} x-points × {t_eval.size} t-slices")

    return x_eval, t_eval, u_pinn


# ═══════════════════════════════════════════════════════════════════════
# REFERENCE SOLUTION (FINITE DIFFERENCES)
# ═══════════════════════════════════════════════════════════════════════
#
# To validate the PINN, we need a trusted "ground truth."  We compute
# one using the method of lines (MOL):
#
#   1. Discretize the spatial domain into a fine grid of nx points.
#   2. Approximate spatial derivatives with central finite differences:
#        ∂u/∂x  ≈ (u[i+1] − u[i−1]) / (2·Δx)
#        ∂²u/∂x² ≈ (u[i+1] − 2·u[i] + u[i−1]) / Δx²
#   3. This converts the PDE into a system of nx coupled ODEs in time.
#   4. Hand that ODE system to scipy's solve_ivp to integrate in time.
#
# We use the Radau solver (an implicit Runge-Kutta method) because the
# diffusion term makes the ODE system mildly stiff — explicit solvers
# (like RK45) would need impractically small time steps.
#
# Stability note — grid Reynolds number:
#
#   Central differences for the convection term (u·∂u/∂x) are stable
#   when the grid Reynolds number
#       Re_h = |u|·Δx / (2ν)
#   is less than 1.  With our parameters:
#       |u|_max ≈ 1,  Δx = 2/1024 ≈ 0.002,  ν ≈ 0.00318
#       Re_h ≈ 1 × 0.002 / (2 × 0.00318) ≈ 0.31
#   Safely below 1, so central differences are fine at this resolution.

def compute_reference_solution(x_eval, t_eval, nu, nx=REF_NX):
    """Compute a reference solution via the method of lines.

    Parameters
    ----------
    x_eval : np.ndarray, shape (M,)
        Spatial points where we want the solution evaluated.
    t_eval : array-like of float
        Time values to evaluate (e.g. [0.0, 0.25, 0.5, 0.75, 1.0]).
    nu : float
        Kinematic viscosity.
    nx : int
        Number of interior spatial grid points.  Higher = more
        accurate reference (but slower).

    Returns
    -------
    np.ndarray, shape (len(t_eval), len(x_eval))
        Reference solution u(x, t) at the requested points.
    """
    # ── Spatial grid ─────────────────────────────────────────────────
    # nx+2 total points: nx interior + 2 boundary (x=−1 and x=+1).
    x_grid = np.linspace(X_MIN, X_MAX, nx + 2)
    dx = x_grid[1] - x_grid[0]

    # Initial condition at interior points only.
    # Boundary values (u=0 at x=±1) are enforced separately.
    u0_interior = -np.sin(np.pi * x_grid[1:-1])

    def rhs(t_unused, u_interior):
        """Right-hand side of the MOL ODE system.

        Parameters
        ----------
        t_unused : float
            Current time.  Not used in the RHS (the PDE coefficients
            don't depend on t), but solve_ivp requires this signature.
        u_interior : np.ndarray, shape (nx,)
            Solution at interior grid points at the current time.

        Returns
        -------
        np.ndarray, shape (nx,)
            Time derivative du/dt at each interior point.
        """
        # Reconstruct the full solution including fixed boundary values.
        u_full = np.zeros(nx + 2)
        u_full[0]    = 0.0           # u(−1, t) = 0
        u_full[-1]   = 0.0           # u(+1, t) = 0
        u_full[1:-1] = u_interior    # Interior values from ODE solver

        # Central-difference approximations for spatial derivatives.
        #
        # ∂u/∂x at interior point i:
        #   (u[i+1] − u[i−1]) / (2·Δx)
        u_x = (u_full[2:] - u_full[:-2]) / (2.0 * dx)

        # ∂²u/∂x² at interior point i:
        #   (u[i+1] − 2·u[i] + u[i−1]) / Δx²
        u_xx = (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / (dx**2)

        # Burgers' equation in MOL form:
        #   du[i]/dt = −u[i]·(∂u/∂x)[i] + ν·(∂²u/∂x²)[i]
        dudt = -u_interior * u_x + nu * u_xx

        return dudt

    # ── Integrate the ODE system in time ─────────────────────────────
    print(f"\n  Computing reference solution (nx={nx}, solver=Radau)...")

    t_eval_arr = np.array(t_eval, dtype=np.float64)
    sol = solve_ivp(
        rhs,
        t_span=[float(T_MIN), float(T_MAX)],
        y0=u0_interior,
        method="Radau",        # Implicit solver — handles stiffness
        t_eval=t_eval_arr,
        rtol=1e-8,             # Tight tolerance for an accurate reference
        atol=1e-10,
    )

    if not sol.success:
        print(f"  WARNING: Reference solver failed: {sol.message}")
    else:
        print(f"  Reference solution computed successfully.")

    # sol.y has shape (nx, len(t_eval)).
    # Reconstruct full solution including boundary points.
    u_ref_full = np.zeros((nx + 2, len(t_eval_arr)))
    u_ref_full[1:-1, :] = sol.y    # Interior from solve_ivp
    # Boundaries stay at 0 (already initialized to zeros).

    # ── Interpolate to the requested evaluation points ───────────────
    # The FD grid may not match the exact x_eval points, so we
    # interpolate with a cubic spline.
    u_at_eval = np.zeros((len(t_eval_arr), len(x_eval)))
    for j in range(len(t_eval_arr)):
        interpolator = interp1d(x_grid, u_ref_full[:, j], kind="cubic")
        u_at_eval[j, :] = interpolator(x_eval)

    return u_at_eval


# ═══════════════════════════════════════════════════════════════════════
# ERROR METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_errors(u_pinn, u_ref):
    """Compute L2 relative and max absolute errors.

    Parameters
    ----------
    u_pinn : np.ndarray, shape (nt, nx)
    u_ref  : np.ndarray, shape (nt, nx)

    Returns
    -------
    l2_overall : float
        ‖u_pinn − u_ref‖₂ / ‖u_ref‖₂ over all points.
    per_slice : list of dict
        One entry per time slice: {l2, max_abs}.
    """
    l2_overall = np.linalg.norm(u_pinn - u_ref) / np.linalg.norm(u_ref)

    per_slice = []
    for j in range(u_ref.shape[0]):
        diff = u_pinn[j, :] - u_ref[j, :]
        ref_norm = np.linalg.norm(u_ref[j, :])
        # At t where the reference is identically zero the relative
        # error is undefined; report NaN rather than dividing by zero.
        l2 = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else np.nan
        per_slice.append({
            "l2": l2,
            "max_abs": np.abs(diff).max(),
        })

    return l2_overall, per_slice


def write_validation_csv(t_eval, per_slice, l2_overall, output_dir):
    """Write per-time-slice error metrics to CSV.

    Columns: t, l2_rel_error, max_abs_error.  A final row with
    t = "overall" carries the aggregate L2 relative error.

    Parameters
    ----------
    t_eval     : np.ndarray, shape (nt,)
    per_slice  : list of dict
    l2_overall : float
    output_dir : str

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    csv_path = os.path.join(output_dir, "validation_errors.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "l2_rel_error", "max_abs_error"])
        for t_val, errs in zip(t_eval, per_slice):
            writer.writerow([
                f"{t_val:.6f}",
                f"{errs['l2']:.6e}",
                f"{errs['max_abs']:.6e}",
            ])
        writer.writerow(["overall", f"{l2_overall:.6e}", ""])

    print(f"  Saved: {csv_path}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: PINN VS REFERENCE
# ═══════════════════════════════════════════════════════════════════════

def plot_pinn_vs_reference(x_eval, t_eval, u_pinn, u_ref,
                           l2_overall, output_dir):
    """Overlay the PINN prediction and the reference at each time slice.

    Parameters
    ----------
    x_eval     : np.ndarray, shape (nx,)
    t_eval     : np.ndarray, shape (nt,)
    u_pinn     : np.ndarray, shape (nt, nx)
    u_ref      : np.ndarray, shape (nt, nx)
    l2_overall : float
    output_dir : str

    Returns
    -------
    str
        Path to the saved PNG.
    """
    # The C++ binary's --nt is free, so the slice count can be 1 or 20,
    # not just the 5 the main branch hard-codes.  Wrap into rows of at
    # most MAX_COLS; squeeze=False keeps `axes` 2-D even for a single
    # subplot, so the indexing below works in every case.
    n_slices = len(t_eval)
    n_cols = min(n_slices, MAX_COLS)
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for j, t_val in enumerate(t_eval):
        ax = flat_axes[j]
        ax.plot(x_eval, u_ref[j, :],  "b-",  linewidth=2, label="Reference")
        ax.plot(x_eval, u_pinn[j, :], "r--", linewidth=2, label="PINN (C++)")
        ax.set_xlabel("x")
        ax.set_title(f"t = {t_val:.2f}")
        if j == 0:
            ax.set_ylabel("u(x, t)")
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide any unused panels in the last row.
    for ax in flat_axes[n_slices:]:
        ax.set_visible(False)

    fig.suptitle(
        f"PINN (LibTorch C++) vs Reference — 1D Burgers' Equation  "
        f"(L2 error = {l2_overall:.2e})",
        fontsize=13,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "pinn_vs_reference.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return path


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: TRAINING LOSS CURVES
# ═══════════════════════════════════════════════════════════════════════

def read_training_log(path):
    """Read training_log.csv into Adam and L-BFGS record lists.

    Parameters
    ----------
    path : str

    Returns
    -------
    adam_history, lbfgs_history : list of dict
        Each record: {step, total, pde, ic, bc} with float values.
        Both lists are empty if the file cannot be read.
    """
    adam_history, lbfgs_history = [], []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            record = {
                "step":  int(float(row["step"])),
                "total": float(row["total"]),
                "pde":   float(row["pde"]),
                "ic":    float(row["ic"]),
                "bc":    float(row["bc"]),
            }
            if row["phase"] == "adam":
                adam_history.append(record)
            elif row["phase"] == "lbfgs":
                lbfgs_history.append(record)

    return adam_history, lbfgs_history


def plot_training_loss(log_path, output_dir):
    """Plot the loss history on a log scale.

    Skipped without error when the log is absent — that is the normal
    case for a --skip-training run, where no training happened in this
    invocation of the pipeline.

    Parameters
    ----------
    log_path   : str
        Path to training_log.csv.
    output_dir : str

    Returns
    -------
    str or None
        Path to the saved PNG, or None if the plot was skipped.
    """
    if not os.path.isfile(log_path):
        print(f"  Training log not found ({log_path}) — "
              f"skipping training_loss.png")
        return None

    adam_history, lbfgs_history = read_training_log(log_path)
    if not adam_history:
        print(f"  Training log has no Adam records — "
              f"skipping training_loss.png")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [r["step"] for r in adam_history]
    ax.semilogy(epochs, [r["total"] for r in adam_history],
                "k-", linewidth=2, label="Total loss")
    ax.semilogy(epochs, [r["pde"] for r in adam_history],
                linewidth=1, alpha=0.8, label="PDE residual")
    ax.semilogy(epochs, [r["ic"] for r in adam_history],
                linewidth=1, alpha=0.8, label="Initial condition")
    ax.semilogy(epochs, [r["bc"] for r in adam_history],
                linewidth=1, alpha=0.8, label="Boundary condition")

    # If L-BFGS was used, show a dividing line and the L-BFGS loss.
    if lbfgs_history:
        adam_end = epochs[-1]
        ax.axvline(x=adam_end, color="gray", linestyle=":",
                   alpha=0.5, label="Adam → L-BFGS")
        lbfgs_x = [adam_end + r["step"] for r in lbfgs_history]
        ax.semilogy(lbfgs_x, [r["total"] for r in lbfgs_history],
                    "g-", linewidth=2, label="L-BFGS total")

    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    return path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Validate the C++ inference output and write plots."""
    args = parse_args()

    results_csv = args.results_csv or os.path.join(
        args.output_dir, "inference_results.csv"
    )
    training_log = args.training_log or os.path.join(
        args.output_dir, "training_log.csv"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  VALIDATION AND PLOTTING")
    print(f"{'='*65}")
    print(f"  Results:    {results_csv}")
    print(f"  Output dir: {args.output_dir}/")
    print(f"  Viscosity:  ν = 0.01/π ≈ {NU:.6f}")

    # ── Read what the C++ binary predicted ───────────────────────────
    x_eval, t_eval, u_pinn = read_inference_csv(results_csv)

    # ── Reference solution on the same points ────────────────────────
    u_ref = compute_reference_solution(x_eval, t_eval, NU, nx=args.ref_nx)

    # ── Error metrics ────────────────────────────────────────────────
    l2_overall, per_slice = compute_errors(u_pinn, u_ref)

    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*65}")
    print(f"  L2 relative error:  {l2_overall:.6e}")
    print(f"  (Raissi et al. report ~1e-3 to 1e-2 for this setup)")
    print()
    print(f"  {'t':>8}  {'L2 rel':>12}  {'max abs':>12}")
    print(f"  {'-'*36}")
    for t_val, errs in zip(t_eval, per_slice):
        print(f"  {t_val:>8.3f}  {errs['l2']:>12.4e}  "
              f"{errs['max_abs']:>12.4e}")
    print()

    # ── Outputs ──────────────────────────────────────────────────────
    write_validation_csv(t_eval, per_slice, l2_overall, args.output_dir)
    plot_pinn_vs_reference(
        x_eval, t_eval, u_pinn, u_ref, l2_overall, args.output_dir
    )
    plot_training_loss(training_log, args.output_dir)

    print()


if __name__ == "__main__":
    main()
