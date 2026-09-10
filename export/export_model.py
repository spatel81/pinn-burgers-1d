#!/usr/bin/env python3
"""
PINN Training and TorchScript Export
=====================================

Trains a Physics-Informed Neural Network (PINN) for the 1D viscous
Burgers' equation and exports the trained model as a TorchScript file
(.pt) for inference from C++ via LibTorch.

This script is the Python half of the C++ inference pipeline:

    [1] Python (this script):  train → export TorchScript .pt
    [2] C++ (pinn_inference):  load .pt → inference on (x, t) grid

The PINN architecture, training procedure, and physics are identical
to the original pinn_burgers.py on the main branch.  Validation,
plotting, and the reference solution are omitted — this script's
sole job is to produce a portable TorchScript model file.

Solves:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²      x ∈ [−1, 1],  t ∈ [0, 1]
    u(x, 0) = −sin(πx)                 initial condition
    u(−1, t) = u(1, t) = 0             boundary conditions
    ν = 0.01/π ≈ 0.00318               viscosity

Usage:
    python export/export_model.py --device xpu --output-dir build/output
    python export/export_model.py --device cpu --epochs 1000 --no-lbfgs
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ── Physics ──────────────────────────────────────────────────────────
NU = 0.01 / np.pi             # Kinematic viscosity (≈ 0.00318)
X_MIN, X_MAX = -1.0, 1.0      # Spatial domain
T_MIN, T_MAX =  0.0, 1.0      # Time domain

# ── Network architecture ────────────────────────────────────────────
LAYER_SIZES = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# ── Training ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS  = 15000
LEARNING_RATE   = 1e-3
N_COLLOCATION   = 10000
N_IC            = 200
N_BC            = 100

# ── L-BFGS fine-tuning ──────────────────────────────────────────────
LBFGS_MAX_ITER       = 50000
LBFGS_LR             = 1.0
LBFGS_TOLERANCE_GRAD = 1e-7
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE   = 50

# ── Logging ──────────────────────────────────────────────────────────
LOG_EVERY = 500


# ═══════════════════════════════════════════════════════════════════════
# COMMAND-LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields: device, epochs, no_lbfgs, output_dir.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a PINN on the 1D Burgers' equation and export "
            "as TorchScript for C++ inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "xpu", "cuda"],
        help=(
            "Compute device for training.  'auto' picks the best "
            "available: XPU → MPS → CUDA → CPU.  Default: auto"
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
        default="build/output",
        help="Directory for the exported TorchScript model (default: build/output)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# DEVICE SELECTION
# ═══════════════════════════════════════════════════════════════════════

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

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class PINN(nn.Module):
    """Physics-Informed Neural Network for the Burgers equation.

    A fully-connected MLP with tanh activations in hidden layers
    and a linear (unactivated) output layer.

    Parameters
    ----------
    layer_sizes : list of int
        Neuron count per layer.  First element = 2 (inputs: x, t).
        Last element = 1 (output: u).
    """

    def __init__(self, layer_sizes):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

    def forward(self, inputs):
        """Forward pass: map (x, t) to predicted u(x, t).

        Parameters
        ----------
        inputs : torch.Tensor, shape (N, 2)
            Column 0 = x, Column 1 = t.

        Returns
        -------
        torch.Tensor, shape (N, 1)
            Predicted velocity u.
        """
        out = inputs
        for layer in self.linears[:-1]:
            out = torch.tanh(layer(out))
        out = self.linears[-1](out)
        return out


# ═══════════════════════════════════════════════════════════════════════
# PDE RESIDUAL
# ═══════════════════════════════════════════════════════════════════════

def compute_pde_residual(model, x, t, nu):
    """Compute the Burgers' equation PDE residual via autodiff.

    residual = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²

    Parameters
    ----------
    model : PINN
    x : torch.Tensor, shape (N, 1), requires_grad=True
    t : torch.Tensor, shape (N, 1), requires_grad=True
    nu : float

    Returns
    -------
    torch.Tensor, shape (N, 1)
    """
    u = model(torch.cat([x, t], dim=1))

    u_x = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
    )[0]

    u_t = torch.autograd.grad(
        outputs=u, inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True,
    )[0]

    residual = u_t + u * u_x - nu * u_xx
    return residual


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_training_data(device, seed=42):
    """Sample collocation, IC, and BC points and move them to `device`.

    Parameters
    ----------
    device : torch.device
    seed : int

    Returns
    -------
    dict
    """
    torch.manual_seed(seed)

    # Collocation points (domain interior)
    x_col = (X_MIN + (X_MAX - X_MIN) * torch.rand(N_COLLOCATION, 1))
    t_col = (T_MIN + (T_MAX - T_MIN) * torch.rand(N_COLLOCATION, 1))
    x_col = x_col.to(device).requires_grad_(True)
    t_col = t_col.to(device).requires_grad_(True)

    # Initial condition points (t = 0)
    x_ic = (X_MIN + (X_MAX - X_MIN) * torch.rand(N_IC, 1)).to(device)
    t_ic = torch.zeros(N_IC, 1, device=device)
    u_ic = -torch.sin(torch.pi * x_ic)

    # Boundary condition points (x = ±1)
    n_per_side = N_BC // 2
    t_bc_left = (T_MIN + (T_MAX - T_MIN) * torch.rand(n_per_side, 1))
    x_bc_left = torch.full((n_per_side, 1), X_MIN)
    t_bc_right = (T_MIN + (T_MAX - T_MIN) * torch.rand(n_per_side, 1))
    x_bc_right = torch.full((n_per_side, 1), X_MAX)

    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0).to(device)
    t_bc = torch.cat([t_bc_left, t_bc_right], dim=0).to(device)
    u_bc = torch.zeros(N_BC, 1, device=device)

    return {
        "x_col": x_col,  "t_col": t_col,
        "x_ic":  x_ic,   "t_ic":  t_ic,   "u_ic": u_ic,
        "x_bc":  x_bc,   "t_bc":  t_bc,   "u_bc": u_bc,
    }


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def compute_loss(model, data, nu):
    """Compute total PINN loss (PDE + IC + BC).

    Parameters
    ----------
    model : PINN
    data  : dict
    nu    : float

    Returns
    -------
    loss_total, loss_pde, loss_ic, loss_bc : torch.Tensor (scalars)
    """
    residual = compute_pde_residual(model, data["x_col"], data["t_col"], nu)
    loss_pde = torch.mean(residual ** 2)

    u_pred_ic = model(torch.cat([data["x_ic"], data["t_ic"]], dim=1))
    loss_ic = torch.mean((u_pred_ic - data["u_ic"]) ** 2)

    u_pred_bc = model(torch.cat([data["x_bc"], data["t_bc"]], dim=1))
    loss_bc = torch.mean((u_pred_bc - data["u_bc"]) ** 2)

    loss_total = loss_pde + loss_ic + loss_bc
    return loss_total, loss_pde, loss_ic, loss_bc


# ═══════════════════════════════════════════════════════════════════════
# TRAINING — ADAM
# ═══════════════════════════════════════════════════════════════════════

def train_adam(model, data, nu, epochs, lr, log_every):
    """Train the PINN with the Adam optimizer.

    Parameters
    ----------
    model     : PINN
    data      : dict
    nu        : float
    epochs    : int
    lr        : float
    log_every : int

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
        optimizer.zero_grad()
        loss_total, loss_pde, loss_ic, loss_bc = compute_loss(
            model, data, nu
        )
        loss_total.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "total": loss_total.item(),
            "pde":   loss_pde.item(),
            "ic":    loss_ic.item(),
            "bc":    loss_bc.item(),
        }
        history.append(record)

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
# TRAINING — L-BFGS FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════

def train_lbfgs(model, data, nu):
    """Fine-tune the PINN with the L-BFGS optimizer.

    Parameters
    ----------
    model : PINN
    data  : dict
    nu    : float

    Returns
    -------
    list of dict
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    history = []
    step_count = [0]

    print(f"\n{'='*65}")
    print(f"  PHASE 2: L-BFGS FINE-TUNING")
    print(f"{'='*65}")

    t_start = time.time()

    def closure():
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

    optimizer.step(closure)

    elapsed = time.time() - t_start
    print(f"  L-BFGS complete — {elapsed:.1f}s, "
          f"{step_count[0]} function evaluations")
    if history:
        print(f"  Final loss = {history[-1]['total']:.4e}")

    return history


# ═══════════════════════════════════════════════════════════════════════
# TORCHSCRIPT EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_torchscript(model, output_dir):
    """Export the trained model as a TorchScript file.

    The model is moved to CPU before tracing so the resulting .pt file
    is device-agnostic — it can be loaded on any device (CPU, XPU, CUDA)
    by the C++ inference binary.

    Parameters
    ----------
    model : PINN (trained)
    output_dir : str

    Returns
    -------
    str
        Path to the saved TorchScript file.
    """
    model.eval()
    model_cpu = model.cpu()

    # Trace the forward pass with an example input.
    # The PINN forward() has no data-dependent control flow, so
    # tracing captures the full computation graph in one pass.
    example_input = torch.randn(1, 2)
    traced_model = torch.jit.trace(model_cpu, example_input)

    # Verify the traced model matches the original
    with torch.no_grad():
        test_input = torch.randn(10, 2)
        orig_out = model_cpu(test_input)
        traced_out = traced_model(test_input)
        max_diff = (orig_out - traced_out).abs().max().item()
        if max_diff > 1e-6:
            print(f"  WARNING: Trace mismatch — max diff = {max_diff:.2e}")
        else:
            print(f"  Trace verified: max difference = {max_diff:.2e}")

    output_path = os.path.join(output_dir, "pinn_burgers_traced.pt")
    traced_model.save(output_path)
    print(f"  TorchScript model saved: {output_path}")

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Train the PINN and export as TorchScript."""
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
    train_adam(model, data, NU, args.epochs, LEARNING_RATE, LOG_EVERY)

    # ── Phase 2: L-BFGS (optional) ──────────────────────────────────
    if not args.no_lbfgs:
        train_lbfgs(model, data, NU)
    else:
        print(f"\n  L-BFGS skipped (--no-lbfgs flag).")

    # ── Export TorchScript ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  EXPORTING TORCHSCRIPT MODEL")
    print(f"{'='*65}")
    output_path = export_torchscript(model, args.output_dir)

    # ── Summary ──────────────────────────────────────────────────────
    t_total_elapsed = time.time() - t_total_start
    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")
    print(f"  Total wall-clock:     {t_total_elapsed:.1f}s")
    print(f"  TorchScript model:    {output_path}")
    print(f"  Ready for C++ inference via LibTorch.")
    print()


if __name__ == "__main__":
    main()
