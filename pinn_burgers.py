#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) for the 1D Viscous Burgers' Equation
============================================================================

Solves the PDE:

    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²      x ∈ [−1, 1],  t ∈ [0, 1]

    Initial condition:    u(x, 0) = −sin(πx)
    Boundary conditions:  u(−1, t) = u(1, t) = 0
    Viscosity:            ν = 0.01/π  (≈ 0.00318)

This is the standard PINN benchmark from:
    Raissi, Perdikaris & Karniadakis (2019),
    "Physics-Informed Neural Networks: A Deep Learning Framework for
     Solving Forward and Inverse Problems Involving Nonlinear PDEs"

The script trains a neural network that learns the solution u(x,t) by
embedding the PDE directly into the loss function — no simulation data
is needed for the interior of the domain.  It then validates the result
against a high-resolution finite-difference reference solution.

Portable across devices:
    - CPU          (any machine)
    - MPS          (Apple Silicon Mac)
    - XPU          (Intel GPUs on Aurora supercomputer)
    - CUDA         (NVIDIA GPUs)

Usage examples:

    # Auto-detect best available device
    python pinn_burgers.py

    # Force CPU
    python pinn_burgers.py --device cpu

    # Quick test run (fewer epochs, no L-BFGS fine-tuning)
    python pinn_burgers.py --device cpu --epochs 1000 --no-lbfgs

    # Custom output directory
    python pinn_burgers.py --output-dir results/run1
"""


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════════
#
# Each library serves a specific role in the pipeline:
#   - torch:       builds and trains the neural network
#   - numpy:       array math for the reference solution and plotting
#   - matplotlib:  generates the validation and loss-curve plots
#   - scipy:       provides the ODE solver for the reference solution
#   - argparse:    parses command-line flags (--device, --epochs, etc.)
#   - csv:         writes the training log to a simple CSV file
#   - os, time:    file I/O and timing

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

# Use the non-interactive "Agg" backend so matplotlib writes directly
# to image files without needing a display server.  This is essential
# on Aurora (no screen) and harmless on a laptop (we save PNGs anyway).
# Must be set BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
#
# Every tunable parameter lives here in one place.  When you want to
# experiment — different network size, more collocation points, lower
# learning rate — you change it here, not buried in a function body.

# ── Physics ──────────────────────────────────────────────────────────
NU = 0.01 / np.pi             # Kinematic viscosity (≈ 0.00318).
                               # Small enough that the solution develops
                               # a steep gradient (shock-like) near x=0
                               # as t increases.
X_MIN, X_MAX = -1.0, 1.0      # Spatial domain: x ∈ [−1, 1]
T_MIN, T_MAX =  0.0, 1.0      # Time domain:    t ∈ [0, 1]

# ── Network architecture ────────────────────────────────────────────
#    Input layer:  2 neurons  (x, t)
#    Hidden layers: 8 layers × 20 neurons each
#    Output layer: 1 neuron   (u)
#
# This matches the architecture in Raissi et al. (2019), Section 3.1.
# It is deliberately modest — big enough to represent the solution,
# small enough to train in minutes on a laptop CPU.
LAYER_SIZES = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# ── Training ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS  = 15000        # Adam optimizer epochs (more = better,
                               # but diminishing returns past ~15k)
LEARNING_RATE   = 1e-3         # Adam learning rate (1e-3 is a safe default)
N_COLLOCATION   = 10000        # Interior points where we enforce the PDE
N_IC            = 200          # Points along t=0 for the initial condition
N_BC            = 100          # Points along x=±1 for boundary conditions
                               # (50 per side)

# ── L-BFGS fine-tuning ──────────────────────────────────────────────
LBFGS_MAX_ITER       = 50000   # Maximum L-BFGS iterations
LBFGS_LR             = 1.0    # Step size (1.0 is standard for L-BFGS)
LBFGS_TOLERANCE_GRAD = 1e-7   # Stop when gradients are this small
LBFGS_TOLERANCE_CHANGE = 1e-9 # Stop when loss changes less than this
LBFGS_HISTORY_SIZE   = 50     # Past updates kept for Hessian approximation

# ── Logging ──────────────────────────────────────────────────────────
LOG_EVERY = 500                # Print a status line every N Adam epochs

# ── Reference solution ───────────────────────────────────────────────
REF_NX = 1024                  # Spatial grid points for FD reference solver
EVAL_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]  # Time slices to plot & compare


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: COMMAND-LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════
#
# Minimal CLI: only the knobs you actually turn between runs.
# Everything else is controlled by the constants above.

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields: device, epochs, no_lbfgs, output_dir.
    """
    parser = argparse.ArgumentParser(
        description="Train a PINN on the 1D viscous Burgers' equation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "xpu", "cuda"],
        help=(
            "Compute device.  'auto' picks the best available: "
            "XPU (Intel GPU) → MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.  "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of Adam training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--no-lbfgs",
        action="store_true",
        help="Skip the L-BFGS fine-tuning phase after Adam",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saved plots, model, and logs (default: outputs)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: DEVICE SELECTION
# ═══════════════════════════════════════════════════════════════════════
#
# This function makes the script portable across three environments:
#
#   Laptop (macOS / Apple Silicon)  →  MPS (Metal Performance Shaders)
#   Aurora supercomputer            →  XPU (Intel Data Center GPU Max)
#   Any machine without a GPU       →  CPU
#
# The --device flag lets you override auto-detection.  This is useful
# for benchmarking CPU vs GPU on the same machine, or when you want to
# force CPU even though a GPU is available (e.g., to compare runtimes).

def select_device(requested="auto"):
    """Choose the compute device, auto-detecting if requested='auto'.

    Priority order for auto-detection:
        1. XPU  — Intel GPUs (Aurora)
        2. MPS  — Apple Silicon (Mac)
        3. CUDA — NVIDIA GPUs
        4. CPU  — fallback

    Parameters
    ----------
    requested : str
        'auto', 'cpu', 'mps', 'xpu', or 'cuda'.

    Returns
    -------
    torch.device
    """
    # ── Explicit device request ──────────────────────────────────────
    if requested != "auto":
        if requested == "xpu":
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                print("  WARNING: XPU requested but not available → CPU.")
                return torch.device("cpu")
        elif requested == "mps":
            if not (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()):
                print("  WARNING: MPS requested but not available → CPU.")
                return torch.device("cpu")
        elif requested == "cuda":
            if not torch.cuda.is_available():
                print("  WARNING: CUDA requested but not available → CPU.")
                return torch.device("cpu")
        return torch.device(requested)

    # ── Auto-detection: try accelerators in priority order ───────────
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
#
# The network is a fully-connected multi-layer perceptron (MLP):
#
#     (x, t) → [Linear → tanh] × 8 → Linear → u(x, t)
#
# It takes two scalar inputs — a spatial coordinate x and a time
# coordinate t — and predicts the velocity u at that point.
#
# Why tanh activations?
#
#   The Burgers' equation solution is smooth (for ν > 0), and the PINN
#   loss requires computing ∂²u/∂x² through the network via autodiff.
#   tanh is infinitely differentiable (class C∞), so its first and
#   second derivatives are well-defined everywhere — the network can
#   represent smooth gradients cleanly.
#
#   ReLU, by contrast, has zero second derivative everywhere (and is
#   undefined at the kink).  A ReLU network's ∂²u/∂x² would be zero
#   almost everywhere, making the diffusion term ν·∂²u/∂x² in the PDE
#   loss useless — the network could never learn the viscous behavior.
#
# Why Xavier (Glorot) initialization?
#
#   In a deep network (8 layers), if weights are initialized too large,
#   activations explode; too small, and gradients vanish.  Xavier
#   initialization scales each layer's weights by 1/√(fan_in + fan_out),
#   keeping the variance of activations and gradients roughly constant
#   across layers.  This gives the optimizer a reasonable starting point.

class PINN(nn.Module):
    """Physics-Informed Neural Network for the Burgers equation.

    A fully-connected MLP with tanh activations in hidden layers
    and a linear (unactivated) output layer.

    Parameters
    ----------
    layer_sizes : list of int
        Neuron count per layer.  First element = 2 (inputs: x, t).
        Last element = 1 (output: u).  Example: [2, 20, 20, ..., 1].
    """

    def __init__(self, layer_sizes):
        super().__init__()

        # Build a list of Linear layers from the size specification.
        # For layer_sizes = [2, 20, 20, ..., 20, 1] this creates:
        #   Linear(2→20), Linear(20→20), ..., Linear(20→20), Linear(20→1)
        # That's 9 Linear layers total: 8 hidden + 1 output.
        self.linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])

            # Xavier initialization for weights
            nn.init.xavier_normal_(layer.weight)
            # Zero-initialize biases (simple, standard choice)
            nn.init.zeros_(layer.bias)

            self.linears.append(layer)

    def forward(self, inputs):
        """Forward pass: map (x, t) to predicted u(x, t).

        Parameters
        ----------
        inputs : torch.Tensor, shape (N, 2)
            Column 0 = spatial coordinate x.
            Column 1 = time coordinate t.

        Returns
        -------
        torch.Tensor, shape (N, 1)
            Predicted velocity u at each input point.
        """
        # Pass through each hidden layer: Linear → tanh.
        out = inputs
        for layer in self.linears[:-1]:
            out = torch.tanh(layer(out))

        # Final layer: linear output (no activation), so the network
        # can predict any real value of u — positive or negative.
        out = self.linears[-1](out)
        return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: PDE RESIDUAL — THE HEART OF THE PINN
# ═══════════════════════════════════════════════════════════════════════
#
# A PINN encodes the governing PDE directly into its loss function.
# Instead of training on labeled data (x, t, u_true), we ask:
#
#   "At randomly sampled interior points, does the network's output
#    satisfy the Burgers' equation?"
#
# Concretely, we compute the PDE residual:
#
#   r(x, t) = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²
#
# If the network perfectly satisfies the PDE, r = 0 everywhere.
# We penalize |r|² in the loss to push the network toward solutions
# that obey the physics.
#
# The derivatives ∂u/∂t, ∂u/∂x, and ∂²u/∂x² are computed via
# automatic differentiation (autodiff) — PyTorch's torch.autograd.grad.
# This is exact (to machine precision), not a finite-difference
# approximation.
#
# ── Key PyTorch autodiff concepts ────────────────────────────────────
#
#   torch.autograd.grad(outputs, inputs, grad_outputs, create_graph,
#                       retain_graph)
#
#   • grad_outputs = torch.ones_like(u)
#       Tells PyTorch to compute ∂(sum of u)/∂input.  For a batched
#       u of shape (N,1), this effectively gives the element-wise
#       derivative ∂u/∂input at each of the N points.
#
#   • create_graph = True
#       By default, grad() computes the derivative but does NOT build
#       a computational graph for the derivative itself.  Setting this
#       to True says: "I want to differentiate this derivative later."
#       We need this for two reasons:
#         (a) To compute the second derivative ∂²u/∂x² (differentiating
#             the first derivative ∂u/∂x).
#         (b) To let loss.backward() propagate gradients through the
#             PDE residual back to the network weights.
#
#   • retain_graph = True
#       After computing one gradient, PyTorch normally frees the
#       computational graph to save memory.  We set this to True
#       because we compute multiple gradients (∂u/∂t, ∂u/∂x, ∂²u/∂x²)
#       from the same forward pass — if the graph were freed after the
#       first grad() call, the second one would fail.

def compute_pde_residual(model, x, t, nu):
    """Compute the Burgers' equation PDE residual via autodiff.

    residual = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²

    Parameters
    ----------
    model : PINN
        The neural network.
    x : torch.Tensor, shape (N, 1)
        Spatial coordinates.  Must have requires_grad=True.
    t : torch.Tensor, shape (N, 1)
        Time coordinates.  Must have requires_grad=True.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    torch.Tensor, shape (N, 1)
        PDE residual at each point.  Should be ≈ 0 if the network
        satisfies the equation.
    """
    # ── Forward pass ─────────────────────────────────────────────────
    # Concatenate x and t into a single (N, 2) tensor and predict u.
    u = model(torch.cat([x, t], dim=1))

    # ── First-order derivatives ──────────────────────────────────────

    # ∂u/∂x — spatial gradient: how u changes as we move in space.
    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,     # We will differentiate u_x → u_xx next
        retain_graph=True,     # Keep the graph for computing u_t
    )[0]

    # ∂u/∂t — temporal gradient: how u changes over time.
    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,     # Needed for backprop through the loss
        retain_graph=True,     # Keep the graph for computing u_xx
    )[0]

    # ── Second-order derivative ──────────────────────────────────────

    # ∂²u/∂x² — spatial curvature: how ∂u/∂x itself changes in space.
    # This is why we set create_graph=True when computing u_x above:
    # we are now differentiating the derivative.
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,     # Needed for backprop through the loss
        retain_graph=True,     # Keep the graph for loss.backward()
    )[0]

    # ── Assemble the PDE residual ────────────────────────────────────
    #
    # Burgers' equation:  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    #
    # Rearranged:          ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x² = 0
    #                      \_____________________________/
    #                                residual
    #
    # The first term (u·∂u/∂x) is the nonlinear advection — the fluid
    # carries itself.  The second term (ν·∂²u/∂x²) is viscous diffusion
    # — it smooths gradients.  When ν is small (as here), advection
    # dominates and a shock-like steep gradient forms near x=0.
    residual = u_t + u * u_x - nu * u_xx

    return residual


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════
#
# A PINN needs three sets of points (no labeled simulation data):
#
#   1. Collocation points — random (x, t) in the domain interior.
#      The PDE residual is evaluated here.  More points → better
#      coverage of the domain, but more computation per epoch.
#
#   2. Initial condition (IC) points — (x, 0) along the bottom edge.
#      We know u(x, 0) = −sin(πx) and enforce it as a soft constraint.
#
#   3. Boundary condition (BC) points — (±1, t) along the side edges.
#      We know u(±1, t) = 0 and enforce it similarly.
#
# All points are sampled once (uniform random) and fixed for the entire
# training run.  Resampling each epoch can improve coverage but adds
# complexity; fixed points are standard for small benchmarks like this.

def generate_training_data(device, seed=42):
    """Sample collocation, IC, and BC points and move them to `device`.

    Parameters
    ----------
    device : torch.device
        Target device (cpu, mps, xpu, or cuda).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'x_col', 't_col', 'x_ic', 't_ic', 'u_ic',
              'x_bc', 't_bc', 'u_bc'.
    """
    torch.manual_seed(seed)

    # ── Collocation points (domain interior) ─────────────────────────
    # Random (x, t) pairs uniformly distributed in [−1, 1] × [0, 1].
    x_col = (X_MIN + (X_MAX - X_MIN) * torch.rand(N_COLLOCATION, 1))
    t_col = (T_MIN + (T_MAX - T_MIN) * torch.rand(N_COLLOCATION, 1))

    # Move to device, then enable gradient tracking.
    # requires_grad=True tells PyTorch to record operations on these
    # tensors so that torch.autograd.grad can compute ∂u/∂x and ∂u/∂t.
    x_col = x_col.to(device).requires_grad_(True)
    t_col = t_col.to(device).requires_grad_(True)

    # ── Initial condition points (t = 0) ─────────────────────────────
    # u(x, 0) = −sin(πx)
    x_ic = (X_MIN + (X_MAX - X_MIN) * torch.rand(N_IC, 1)).to(device)
    t_ic = torch.zeros(N_IC, 1, device=device)

    # The known IC values — these are the "labels" for the IC loss.
    u_ic = -torch.sin(torch.pi * x_ic)

    # ── Boundary condition points (x = ±1) ───────────────────────────
    # u(−1, t) = u(+1, t) = 0
    n_per_side = N_BC // 2  # 50 points on each boundary

    # Left boundary: x = −1, random times
    t_bc_left = (T_MIN + (T_MAX - T_MIN) * torch.rand(n_per_side, 1))
    x_bc_left = torch.full((n_per_side, 1), X_MIN)

    # Right boundary: x = +1, random times
    t_bc_right = (T_MIN + (T_MAX - T_MIN) * torch.rand(n_per_side, 1))
    x_bc_right = torch.full((n_per_side, 1), X_MAX)

    # Stack both sides into single tensors
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0).to(device)
    t_bc = torch.cat([t_bc_left, t_bc_right], dim=0).to(device)

    # BC value is zero at both boundaries
    u_bc = torch.zeros(N_BC, 1, device=device)

    return {
        "x_col": x_col,  "t_col": t_col,
        "x_ic":  x_ic,   "t_ic":  t_ic,   "u_ic": u_ic,
        "x_bc":  x_bc,   "t_bc":  t_bc,   "u_bc": u_bc,
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════
#
# The total PINN loss is the sum of three mean-squared-error terms:
#
#   L_total = L_pde + L_ic + L_bc
#
#   L_pde = (1/N_col) Σ (residual_i)²      PDE must be satisfied
#   L_ic  = (1/N_ic)  Σ (u_pred − u_ic)²   IC must be matched
#   L_bc  = (1/N_bc)  Σ (u_pred − u_bc)²   BCs must be matched
#
# We use equal weights (1 : 1 : 1) as a starting point.  In practice,
# tuning these weights — or using adaptive schemes like the "learning
# rate annealing" method from Wang, Teng & Perdikaris (2021) — can
# improve training.  But equal weights work well for this benchmark
# and keep the code straightforward.

def compute_loss(model, data, nu):
    """Compute total PINN loss and its three components.

    Parameters
    ----------
    model : PINN
    data  : dict from generate_training_data()
    nu    : float, kinematic viscosity

    Returns
    -------
    loss_total, loss_pde, loss_ic, loss_bc : torch.Tensor (scalars)
    """
    # ── PDE residual loss ────────────────────────────────────────────
    # "Does the network satisfy the Burgers' equation in the interior?"
    residual = compute_pde_residual(model, data["x_col"], data["t_col"], nu)
    loss_pde = torch.mean(residual ** 2)

    # ── Initial condition loss ───────────────────────────────────────
    # "Does the network match u(x, 0) = −sin(πx)?"
    u_pred_ic = model(torch.cat([data["x_ic"], data["t_ic"]], dim=1))
    loss_ic = torch.mean((u_pred_ic - data["u_ic"]) ** 2)

    # ── Boundary condition loss ──────────────────────────────────────
    # "Does the network give u = 0 at x = ±1?"
    u_pred_bc = model(torch.cat([data["x_bc"], data["t_bc"]], dim=1))
    loss_bc = torch.mean((u_pred_bc - data["u_bc"]) ** 2)

    # ── Total (equal weighting) ──────────────────────────────────────
    loss_total = loss_pde + loss_ic + loss_bc

    return loss_total, loss_pde, loss_ic, loss_bc


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: TRAINING — ADAM OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════
#
# Phase 1 of training uses the Adam optimizer.  Adam is an adaptive
# gradient method that maintains per-parameter learning rates — it is
# robust, general-purpose, and reliably makes good progress from a
# random starting point.
#
# Its weakness: Adam tends to plateau before reaching high precision.
# That's why we optionally follow it with L-BFGS (Phase 2).
#
# We log the three loss components separately.  This lets you diagnose
# training behavior:  "PDE loss is dropping but BC loss is stuck" would
# mean the network is learning interior physics but ignoring boundaries
# — you'd increase the BC weight.  Logging now is cheap and saves you
# from having to re-run later when writing things up.

def train_adam(model, data, nu, epochs, lr, log_every):
    """Train the PINN with the Adam optimizer.

    Parameters
    ----------
    model     : PINN
    data      : dict, training data
    nu        : float, viscosity
    epochs    : int, number of epochs
    lr        : float, learning rate
    log_every : int, print status every N epochs

    Returns
    -------
    list of dict
        One entry per epoch: {epoch, total, pde, ic, bc}.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    print(f"\n{'='*65}")
    print(f"  PHASE 1: ADAM TRAINING — {epochs} epochs, lr = {lr}")
    print(f"{'='*65}")
    print(f"  {'Epoch':>7}  {'Total':>12}  {'PDE':>12}  "
          f"{'IC':>12}  {'BC':>12}")
    print(f"  {'-'*61}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # (a) Zero out gradients accumulated from the previous step.
        #     Without this, gradients would accumulate across epochs
        #     and the updates would be wrong.
        optimizer.zero_grad()

        # (b) Forward pass + loss computation.
        loss_total, loss_pde, loss_ic, loss_bc = compute_loss(
            model, data, nu
        )

        # (c) Backward pass: compute ∂(loss)/∂(weights) for every
        #     weight in the network.  This is the "learning signal."
        loss_total.backward()

        # (d) Update step: Adam adjusts each weight using its gradient,
        #     its history of past gradients, and its per-parameter
        #     adaptive learning rate.
        optimizer.step()

        # (e) Record this epoch's losses for later plotting.
        record = {
            "epoch": epoch,
            "total": loss_total.item(),   # .item() converts a 0-dim
            "pde":   loss_pde.item(),     # tensor to a plain Python
            "ic":    loss_ic.item(),      # float — avoids accumulating
            "bc":    loss_bc.item(),      # GPU memory.
        }
        history.append(record)

        # Print a status line periodically (and always on epoch 1).
        if epoch == 1 or epoch % log_every == 0:
            print(f"  {epoch:>7}  {record['total']:>12.4e}  "
                  f"{record['pde']:>12.4e}  {record['ic']:>12.4e}  "
                  f"{record['bc']:>12.4e}")

    elapsed = time.time() - t_start
    print(f"  {'-'*61}")
    print(f"  Adam complete — {elapsed:.1f}s, "
          f"final loss = {history[-1]['total']:.4e}")

    return history


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: TRAINING — L-BFGS FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════
#
# Phase 2 (optional) uses L-BFGS — a quasi-Newton method that
# approximates the Hessian (matrix of second derivatives of the loss
# with respect to weights) to take more informed steps than Adam.
#
# Why Adam first, then L-BFGS?
#
#   Adam is gradient-descent-based: it sees only the slope (first
#   derivative) of the loss landscape.  It makes steady progress from
#   any starting point but converges slowly near the optimum.
#
#   L-BFGS sees curvature (second-derivative information), so it can
#   take "Newton steps" that jump much closer to the minimum.  But it
#   is sensitive to initialization — started far from the optimum, it
#   can overshoot or diverge.
#
#   Using Adam to get close, then L-BFGS to sharpen, combines the
#   strengths of both.  This is the standard recipe in Raissi et al.
#
# The "closure" pattern:
#
#   Unlike Adam (which always takes exactly one step per .step() call),
#   L-BFGS may evaluate the loss multiple times per step during its
#   internal line search (finding the right step size).  So it needs a
#   callable — the "closure" — that can recompute the loss on demand.

def train_lbfgs(model, data, nu):
    """Fine-tune the PINN with the L-BFGS optimizer.

    Should be called AFTER Adam pre-training so the network starts
    near a good solution.

    Parameters
    ----------
    model : PINN
    data  : dict, training data
    nu    : float, viscosity

    Returns
    -------
    list of dict
        One entry per L-BFGS function evaluation: {step, total, pde, ic, bc}.
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",  # Ensures each step satisfies
                                         # the strong Wolfe conditions
                                         # (sufficient decrease + curvature)
    )

    history = []
    # We use a list [0] instead of a plain int because Python closures
    # can read but not reassign variables from the enclosing scope.
    # A list is mutable, so closure can do step_count[0] += 1.
    step_count = [0]

    print(f"\n{'='*65}")
    print(f"  PHASE 2: L-BFGS FINE-TUNING")
    print(f"{'='*65}")

    t_start = time.time()

    def closure():
        """Re-compute loss and gradients.  Called by L-BFGS internally.

        Each call to this function:
          1. Zeros the gradients
          2. Runs the forward pass and computes the loss
          3. Runs the backward pass (fills in gradients)
          4. Returns the scalar loss (L-BFGS uses it for line search)
        """
        optimizer.zero_grad()
        loss_total, loss_pde, loss_ic, loss_bc = compute_loss(
            model, data, nu
        )
        loss_total.backward()

        step_count[0] += 1
        history.append({
            "step":  step_count[0],
            "total": loss_total.item(),
            "pde":   loss_pde.item(),
            "ic":    loss_ic.item(),
            "bc":    loss_bc.item(),
        })

        if step_count[0] % 500 == 0:
            print(f"    step {step_count[0]:>6}:  "
                  f"loss = {loss_total.item():.4e}")

        return loss_total

    # A single optimizer.step(closure) call runs L-BFGS's internal
    # loop — potentially thousands of closure evaluations — until it
    # converges or hits max_iter.
    optimizer.step(closure)

    elapsed = time.time() - t_start
    print(f"  L-BFGS complete — {elapsed:.1f}s, "
          f"{step_count[0]} function evaluations")
    if history:
        print(f"  Final loss = {history[-1]['total']:.4e}")

    return history


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: REFERENCE SOLUTION (FINITE DIFFERENCES)
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
    u_ref_full = np.zeros((nx + 2, len(t_eval)))
    u_ref_full[1:-1, :] = sol.y    # Interior from solve_ivp
    # Boundaries stay at 0 (already initialized to zeros).

    # ── Interpolate to the requested evaluation points ───────────────
    # The FD grid may not match the exact x_eval points, so we
    # interpolate with a cubic spline.
    u_at_eval = np.zeros((len(t_eval), len(x_eval)))
    for j in range(len(t_eval)):
        interpolator = interp1d(x_grid, u_ref_full[:, j], kind="cubic")
        u_at_eval[j, :] = interpolator(x_eval)

    return u_at_eval


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12: VALIDATION AND PLOTTING
# ═══════════════════════════════════════════════════════════════════════
#
# After training, we:
#   1. Evaluate the PINN on a fine grid at several time slices.
#   2. Compute the reference solution on the same grid.
#   3. Calculate the L2 relative error.
#   4. Plot PINN vs reference side-by-side at each time slice.
#   5. Plot the training loss curves (PDE, IC, BC components).

def validate_and_plot(model, device, output_dir,
                      adam_history, lbfgs_history=None):
    """Compare PINN predictions to the reference and save plots.

    Produces:
      • pinn_vs_reference.png — overlay at t = 0, 0.25, 0.5, 0.75, 1
      • training_loss.png     — log-scale loss curves

    Parameters
    ----------
    model         : PINN (trained)
    device        : torch.device
    output_dir    : str, path to save plots
    adam_history   : list of dict, Adam loss records
    lbfgs_history : list of dict or None, L-BFGS loss records

    Returns
    -------
    float
        L2 relative error over all evaluation points.
    """
    # Switch network to evaluation mode.  (Disables dropout / batchnorm
    # if present.  We don't use them here, but it's good practice and
    # costs nothing.)
    model.eval()

    # ── Build evaluation grid ────────────────────────────────────────
    nx_eval = 256
    x_eval = np.linspace(X_MIN, X_MAX, nx_eval)
    t_eval = EVAL_TIMES  # [0.0, 0.25, 0.5, 0.75, 1.0]

    # ── Reference solution ───────────────────────────────────────────
    u_ref = compute_reference_solution(x_eval, t_eval, NU)

    # ── PINN predictions ─────────────────────────────────────────────
    u_pinn = np.zeros((len(t_eval), nx_eval))

    # torch.no_grad() disables gradient tracking, saving memory and
    # time.  We don't need gradients during inference — only during
    # training.
    t_infer_start = time.time()
    with torch.no_grad():
        for j, t_val in enumerate(t_eval):
            # Build a (nx_eval, 2) input tensor for this time slice.
            x_tensor = torch.tensor(
                x_eval, dtype=torch.float32
            ).unsqueeze(1).to(device)           # shape: (256, 1)
            t_tensor = torch.full_like(x_tensor, t_val)  # (256, 1)

            u_pred = model(torch.cat([x_tensor, t_tensor], dim=1))
            u_pinn[j, :] = u_pred.cpu().numpy().flatten()
    t_infer_elapsed = time.time() - t_infer_start
    print(f"  PINN inference: {t_infer_elapsed:.3f}s "
          f"({len(t_eval)} time slices × {nx_eval} points)")

    # ── L2 relative error ────────────────────────────────────────────
    # ‖u_pinn − u_ref‖₂ / ‖u_ref‖₂
    # Computed over ALL points (all time slices, all spatial locations).
    l2_error = np.linalg.norm(u_pinn - u_ref) / np.linalg.norm(u_ref)

    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*65}")
    print(f"  L2 relative error:  {l2_error:.6e}")
    print(f"  (Raissi et al. report ~1e-3 to 1e-2 for this setup)")

    # ── Plot 1: PINN vs Reference at each time slice ─────────────────
    n_slices = len(t_eval)
    fig, axes = plt.subplots(
        1, n_slices,
        figsize=(4 * n_slices, 4),
        sharey=True,
    )

    for j, (t_val, ax) in enumerate(zip(t_eval, axes)):
        ax.plot(x_eval, u_ref[j, :],  "b-",  linewidth=2, label="Reference")
        ax.plot(x_eval, u_pinn[j, :], "r--", linewidth=2, label="PINN")
        ax.set_xlabel("x")
        ax.set_title(f"t = {t_val:.2f}")
        if j == 0:
            ax.set_ylabel("u(x, t)")
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PINN vs Reference — 1D Burgers' Equation  "
        f"[{device}]  (L2 error = {l2_error:.2e})",
        fontsize=13,
    )
    fig.tight_layout()

    path1 = os.path.join(output_dir, "pinn_vs_reference.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")
    plt.close(fig)

    # ── Plot 2: Training loss curves ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [r["epoch"] for r in adam_history]
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

    path2 = os.path.join(output_dir, "training_loss.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")
    plt.close(fig)

    return l2_error


def save_training_log(adam_history, lbfgs_history, output_dir):
    """Write the loss history to a CSV file.

    Columns: phase, step, total, pde, ic, bc.
    One row per epoch (Adam) or per function evaluation (L-BFGS).

    Parameters
    ----------
    adam_history   : list of dict
    lbfgs_history : list of dict or None
    output_dir    : str
    """
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase", "step", "total", "pde", "ic", "bc"]
        )
        writer.writeheader()

        for r in adam_history:
            writer.writerow({
                "phase": "adam",  "step": r["epoch"],
                "total": r["total"], "pde": r["pde"],
                "ic": r["ic"], "bc": r["bc"],
            })

        if lbfgs_history:
            for r in lbfgs_history:
                writer.writerow({
                    "phase": "lbfgs", "step": r["step"],
                    "total": r["total"], "pde": r["pde"],
                    "ic": r["ic"], "bc": r["bc"],
                })

    print(f"  Saved: {log_path}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN — ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════
#
# Ties everything together:
#   1. Parse CLI arguments
#   2. Select compute device
#   3. Build the network
#   4. Generate training data
#   5. Train (Adam → optional L-BFGS)
#   6. Save training log
#   7. Validate against reference solution
#   8. Save the trained model

def main():
    """Main entry point."""
    t_total_start = time.time()
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────
    device = select_device(args.device)
    print(f"\n  Device:     {device}")
    print(f"  PyTorch:    {torch.__version__}")

    # ── Output directory ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Output dir: {args.output_dir}/")

    # ── Build network ────────────────────────────────────────────────
    model = PINN(LAYER_SIZES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Network:    {LAYER_SIZES}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Viscosity:  ν = 0.01/π ≈ {NU:.6f}")

    # ── Training data ────────────────────────────────────────────────
    data = generate_training_data(device)
    print(f"  Collocation pts: {N_COLLOCATION:,}")
    print(f"  IC pts:          {N_IC}")
    print(f"  BC pts:          {N_BC}")

    # ── Phase 1: Adam ────────────────────────────────────────────────
    adam_history = train_adam(
        model, data, NU, args.epochs, LEARNING_RATE, LOG_EVERY
    )

    # ── Phase 2: L-BFGS (optional) ──────────────────────────────────
    lbfgs_history = None
    if not args.no_lbfgs:
        lbfgs_history = train_lbfgs(model, data, NU)
    else:
        print(f"\n  L-BFGS skipped (--no-lbfgs flag).")

    # ── Save training log ────────────────────────────────────────────
    save_training_log(adam_history, lbfgs_history, args.output_dir)

    # ── Validate and plot ────────────────────────────────────────────
    l2_error = validate_and_plot(
        model, device, args.output_dir, adam_history, lbfgs_history
    )

    # ── Save model checkpoint ────────────────────────────────────────
    # We save the architecture info alongside the weights so the model
    # can be loaded without hard-coding the layer sizes.
    model_path = os.path.join(args.output_dir, "pinn_burgers_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "layer_sizes": LAYER_SIZES,
        "nu": NU,
        "l2_error": l2_error,
        "epochs_adam": args.epochs,
        "lbfgs_used": not args.no_lbfgs,
        "device": str(device),
    }, model_path)
    print(f"  Saved: {model_path}")

    # ── Summary ──────────────────────────────────────────────────────
    t_total_elapsed = time.time() - t_total_start
    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")
    print(f"  L2 relative error:   {l2_error:.6e}")
    print(f"  Total wall-clock:    {t_total_elapsed:.1f}s")
    print(f"  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"    • pinn_vs_reference.png   — PINN vs reference at 5 times")
    print(f"    • training_loss.png       — loss curves (PDE, IC, BC)")
    print(f"    • training_log.csv        — per-epoch loss values")
    print(f"    • pinn_burgers_model.pt   — trained model weights")
    print()


# ── Script entry point ───────────────────────────────────────────────
# This guard ensures main() runs only when the script is executed
# directly (python pinn_burgers.py), not when imported as a module.
if __name__ == "__main__":
    main()
