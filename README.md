# PINN for the 1D Viscous Burgers' Equation

A from-scratch PyTorch implementation of a physics-informed neural network
(PINN) that solves the 1D viscous Burgers' equation by embedding the PDE
directly into the training loss — no simulation training data required.
The result is validated against an independent finite-difference reference
solution.

Runs unmodified on CPU, Apple Silicon (MPS), NVIDIA GPUs (CUDA), and Intel
GPUs (XPU) — including Intel Data Center GPU Max nodes on the [Aurora
supercomputer](https://www.alcf.anl.gov/aurora) at Argonne National
Laboratory.

## What this solves

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²      x ∈ [−1, 1],  t ∈ [0, 1]

u(x, 0) = −sin(πx)                 initial condition
u(−1, t) = u(1, t) = 0             boundary conditions
ν = 0.01/π ≈ 0.00318                viscosity
```

This is the standard PINN benchmark problem: a small viscosity drives the
solution toward a steep, shock-like gradient near x = 0 as t → 1, which
makes it a reasonable stress test for whether a PINN is resolving the
underlying physics or just fitting smooth regions.

**Reference:** M. Raissi, P. Perdikaris, and G. E. Karniadakis,
"Physics-informed neural networks: A deep learning framework for solving
forward and inverse problems involving nonlinear partial differential
equations," *Journal of Computational Physics*, vol. 378, pp. 686–707,
2019. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

```bibtex
@article{raissi2019physics,
  title   = {Physics-informed neural networks: A deep learning framework for
             solving forward and inverse problems involving nonlinear
             partial differential equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal = {Journal of Computational Physics},
  volume  = {378},
  pages   = {686--707},
  year    = {2019},
  publisher = {Elsevier},
  doi     = {10.1016/j.jcp.2018.10.045}
}
```

## How it works, in brief

A PINN doesn't train on labeled `(x, t, u)` data. Instead, it asks: *at
randomly sampled points in the domain, does the network's output satisfy
the governing PDE?* The training loss has three terms:

- **PDE residual loss** — at ~10,000 interior collocation points, compute
  `r = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²` via automatic differentiation and
  penalize `r²`. If the network exactly satisfies the PDE, `r = 0`
  everywhere.
- **Initial condition loss** — enforces `u(x, 0) = −sin(πx)`.
- **Boundary condition loss** — enforces `u(±1, t) = 0`.

Training runs in two phases: Adam (first-order, robust from a random
start, ~15,000 epochs) followed by L-BFGS (quasi-Newton, uses curvature
information to sharpen convergence near the optimum) — the standard recipe
from the original paper. The network itself is a small fully-connected MLP
(8 hidden layers × 20 neurons, tanh activations — tanh is used because it's
infinitely differentiable, which matters since the PDE loss requires a
clean second derivative through the network).

Validation runs a finite-difference method-of-lines solver (central
differences in space, an implicit Radau integrator in time) as an
independent ground truth, and reports the L2 relative error between the
PINN and that reference at five time slices.

## Results

Across CPU, Apple Silicon (MPS), and Intel GPU (XPU on Aurora), the model
consistently lands in the 1e-3–1e-2 L2 relative error range reported in
the original paper, and visibly resolves the developing shock near x = 0
at every evaluated time slice.

| Device        | Adam epochs | Wall time (Adam) | L2 relative error |
|---------------|-------------|-------------------|--------------------|
| Apple Silicon (MPS) | 15,000 | ~167s | 1.70e-03 |
| Intel GPU (XPU, Aurora) | 15,000 | ~184s | 6.10e-03 |
| CPU | 15,000 | ~420s–1,130s | 1.45e-02–2.78e-03 |

*(Exact numbers vary run to run — collocation points are randomly sampled
each run without a fixed benchmark seed across devices.)*

## Quick start — local (CPU / Apple Silicon / NVIDIA)

```bash
git clone <this-repo-url>
cd pinn-burgers-1d

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full run — auto-detects the best available device
# (XPU > MPS > CUDA > CPU)
python pinn_burgers.py

# Quick smoke test (~30s): fewer epochs, skip L-BFGS
python pinn_burgers.py --device cpu --epochs 1000 --no-lbfgs

# Force a specific device
python pinn_burgers.py --device mps      # Apple Silicon
python pinn_burgers.py --device cuda     # NVIDIA GPU
python pinn_burgers.py --device cpu
```

Outputs land in `outputs/` by default: `pinn_vs_reference.png`,
`training_loss.png`, `training_log.csv`, and `pinn_burgers_model.pt`.

### CLI options

| Flag           | Default   | Description                                    |
|----------------|-----------|-------------------------------------------------|
| `--device`     | `auto`    | `auto`, `cpu`, `mps`, `xpu`, or `cuda`          |
| `--epochs`     | `15000`   | Number of Adam training epochs                  |
| `--no-lbfgs`   | *(off)*   | Skip L-BFGS fine-tuning after Adam              |
| `--output-dir` | `outputs` | Where to save plots, model, and logs            |

## Running on an Intel GPU (Aurora or similar Intel Data Center GPU Max systems)

No manual PyTorch install is needed on Aurora — the system's `frameworks`
module ships a PyTorch build with Intel XPU support already wired up.

```bash
module load frameworks
pip install --user matplotlib scipy   # one-time; torch/numpy come from the module

python pinn_burgers.py --device xpu
```

**Batch submission** via PBS: edit the `#PBS -A <YOUR_PROJECT>` line in
`submit_aurora.sh` with your allocation name, then:

```bash
qsub submit_aurora.sh
qstat -u $USER          # check status
```

**Interactive session** on a compute node:

```bash
qsub -I -A <YOUR_PROJECT> -q debug -l select=1 -l walltime=00:30:00
module load frameworks
cd /path/to/pinn-burgers-1d
python pinn_burgers.py --device xpu
```

The `--device auto` flag will also pick up `xpu` automatically on a system
where it's available, so `--device xpu` above is just being explicit.

## Code walkthrough

`pinn_burgers.py` is a single, heavily-commented script organized into 13
labeled sections:

1. **Imports** — torch (model + training), numpy/scipy (reference
   solution), matplotlib (`Agg` backend, so it runs headless on Aurora),
   argparse, csv.
2. **Configuration** — every tunable constant (physics parameters,
   network architecture, training hyperparameters) lives in one block at
   the top of the file, rather than scattered through the code.
3. **Command-line arguments** — the four knobs actually worth changing
   between runs (`--device`, `--epochs`, `--no-lbfgs`, `--output-dir`).
4. **Device selection** — auto-detects XPU → MPS → CUDA → CPU, or honors
   an explicit `--device` override; this is what makes the same script
   portable between a laptop and Aurora.
5. **Network architecture** — the `PINN` class: a fully-connected MLP
   `(x, t) → [Linear → tanh] × 8 → Linear → u(x, t)`, with Xavier
   initialization and a discussion of why tanh (not ReLU) is required for
   a PDE loss that needs a well-behaved second derivative.
6. **PDE residual** — the core of the method: computes
   `∂u/∂t`, `∂u/∂x`, and `∂²u/∂x²` via `torch.autograd.grad` (exact
   autodiff, not finite differences) and assembles the Burgers' residual.
7. **Training data generation** — samples collocation, initial-condition,
   and boundary-condition points once per run (fixed, not resampled per
   epoch).
8. **Loss function** — sums the three MSE terms (PDE + IC + BC) with
   equal weighting.
9. **Adam training loop** — phase 1: standard Adam optimization, logging
   the three loss components separately every 500 epochs.
10. **L-BFGS fine-tuning** — phase 2 (optional): a quasi-Newton pass
    using PyTorch's closure-based `LBFGS` optimizer to sharpen convergence
    after Adam plateaus.
11. **Reference solution** — an independent finite-difference
    method-of-lines solver (central differences + `scipy.integrate.solve_ivp`
    with the implicit Radau method) used purely for validation, with a
    grid-Reynolds-number stability check documented inline.
12. **Validation and plotting** — evaluates the PINN and the reference
    solution on a shared grid, computes the L2 relative error, and
    produces the two output plots.
13. **Main / orchestration** — wires the above together: parse args →
    pick device → build network → generate data → train (Adam → L-BFGS)
    → validate → save model checkpoint and logs.

## Requirements

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
```

On Aurora, `module load frameworks` already provides `torch` and `numpy`
with XPU support — you only need `pip install --user matplotlib scipy`.

## License

MIT — see [LICENSE](LICENSE).
