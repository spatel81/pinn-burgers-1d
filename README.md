# PINN Burgers 1D — C++ Inference via LibTorch

A C++ inference pipeline for the Physics-Informed Neural Network (PINN)
that solves the 1D viscous Burgers' equation.  A single executable
handles the entire workflow: trains the model via Python, exports it as
TorchScript, then loads it in C++ via LibTorch for inference — all in
one command.

This is the **`cpp` branch**.  The original Python-only implementation
lives on the [`main` branch](https://github.com/spatel81/pinn-burgers-1d/tree/main).

## What this solves

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²      x ∈ [−1, 1],  t ∈ [0, 1]

u(x, 0) = −sin(πx)                 initial condition
u(−1, t) = u(1, t) = 0             boundary conditions
ν = 0.01/π ≈ 0.00318               viscosity
```

## How it works

```
./build/pinn_inference --device xpu
  │
  ├── [Step 1] Train & Export (Python)
  │     ├── Train PINN: Adam (15,000 epochs) + L-BFGS
  │     ├── torch.jit.trace(model, example_input)
  │     └── Save pinn_burgers_traced.pt
  │
  ├── [Step 2] Load Model (LibTorch / C++)
  │     ├── torch::jit::load("pinn_burgers_traced.pt")
  │     └── model.to(device)  →  CPU or XPU
  │
  └── [Step 3] Inference (LibTorch / C++)
        ├── Evaluate u(x, t) on 256 × 5 grid
        └── Write inference_results.csv
```

## Prerequisites

**On Aurora (Intel Data Center GPU Max / Sapphire Rapids):**
- `module load frameworks` — provides PyTorch (with XPU), LibTorch, numpy
- `module load cmake`
- Intel `icpx` compiler (included in the frameworks module)

**On a workstation:**
- Python 3.8+ with PyTorch ≥ 2.0, numpy
- CMake ≥ 3.18
- LibTorch (download from [pytorch.org](https://pytorch.org/get-started/locally/))
- A C++17-compatible compiler (g++, clang++, or icpx)

## Build

### On Aurora

```bash
module load frameworks cmake

mkdir -p build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j 4
cd ..
```

### On a workstation (with LibTorch)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j 4
cd ..
```

## Usage

### Full pipeline (train + infer) — single command

```bash
# On Aurora (Intel GPU)
./build/pinn_inference --device xpu --output-dir build/output

# On CPU
./build/pinn_inference --device cpu --output-dir build/output

# Quick test (fewer epochs, no L-BFGS)
./build/pinn_inference --device cpu --epochs 1000 --no-lbfgs --output-dir build/output
```

### Inference only (skip training, reuse a model)

```bash
# Use a previously exported model
./build/pinn_inference --skip-training --model-path build/output/pinn_burgers_traced.pt --device xpu

# Or export first, then infer separately
python export/export_model.py --device xpu --output-dir build/output
./build/pinn_inference --skip-training --output-dir build/output --device xpu
```

### PBS submission on Aurora

Edit `submit_aurora.sh` — replace `<YOUR_PROJECT>` with your allocation
name, then:

```bash
qsub submit_aurora.sh
qstat -u $USER          # check status
```

Or use an interactive session:

```bash
qsub -I -A <YOUR_PROJECT> -q debug -l select=1 -l walltime=00:30:00
module load frameworks cmake
cd /path/to/pinn-burgers-1d
# Build and run as above
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | Inference device: `cpu` or `xpu` |
| `--output-dir` | `build/output` | Output directory for model and results |
| `--model-path` | *(auto)* | Path to existing TorchScript model (skips training) |
| `--epochs` | `15000` | Adam training epochs (forwarded to Python) |
| `--no-lbfgs` | *(off)* | Skip L-BFGS fine-tuning (forwarded to Python) |
| `--skip-training` | *(off)* | Skip training entirely, use existing model |
| `--nx` | `256` | Spatial grid points for inference |
| `--nt` | `5` | Time slices for inference |
| `--help` | | Show usage message |

## Output

The inference writes `inference_results.csv` to the output directory
with columns:

```csv
x,t,u_pred
-1.000000,0.000000,-0.000012
-0.992157,0.000000,-0.024541
...
```

Each row is one evaluation point: `x` (spatial coordinate), `t` (time),
`u_pred` (predicted velocity from the PINN).  With default settings,
the file has 1,280 rows (256 x-points × 5 time slices at
t = 0.0, 0.25, 0.5, 0.75, 1.0).

## Project structure

```
pinn-burgers-1d/          (cpp branch)
├── CMakeLists.txt         LibTorch build configuration
├── README.md              This file
├── LICENSE                MIT license
├── .gitignore
├── src/
│   └── main.cpp           C++ driver: train → load → infer
├── export/
│   └── export_model.py    Python: train PINN + export TorchScript
└── submit_aurora.sh       PBS job script for Aurora
```

## License

MIT — see [LICENSE](LICENSE).
