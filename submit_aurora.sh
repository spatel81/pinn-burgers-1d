#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PBS job script — Build and run PINN C++ inference on Aurora
# ═══════════════════════════════════════════════════════════════════════
#
# This script does everything in one job:
#   1. Builds the C++ inference binary (CMake + make)
#   2. Trains the PINN and exports a TorchScript model (via Python)
#   3. Loads the model in C++ and runs inference on a (x, t) grid
#
# Submit with:    qsub submit_aurora.sh
# Check status:   qstat -u $USER
# View output:    cat pinn_cpp.o<JOB_ID>
#
# Before first use, edit the #PBS -A line below with your project
# allocation name.

# ── PBS directives ───────────────────────────────────────────────────
#PBS -A <YOUR_PROJECT>
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -N pinn_cpp
#PBS -j oe

# ── Environment setup ───────────────────────────────────────────────
# Load the frameworks module — provides PyTorch (with XPU support),
# LibTorch, numpy, and the Intel compilers.
module load frameworks
module load cmake

# ── Navigate to the script directory ─────────────────────────────────
cd ${PBS_O_WORKDIR}

echo "============================================"
echo "  Job:      ${PBS_JOBID}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "  Dir:      $(pwd)"
echo "  Python:   $(python --version)"
echo "  Compiler: $(icpx --version 2>&1 | head -1)"
echo "============================================"

# ── Build ────────────────────────────────────────────────────────────
echo ""
echo ">>> Building C++ inference binary..."
mkdir -p build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j 4
cd ..

if [ ! -f build/pinn_inference ]; then
    echo "ERROR: Build failed — pinn_inference not found"
    exit 1
fi

# ── Run (train + infer, all in one shot) ─────────────────────────────
echo ""
echo ">>> Running pinn_inference on XPU..."
./build/pinn_inference \
    --device xpu \
    --output-dir build/output \
    --epochs 15000

echo ""
echo ">>> Job complete: $(date) <<<"
