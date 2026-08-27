#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PBS job script for running pinn_burgers.py on Aurora
# ═══════════════════════════════════════════════════════════════════════
#
# Submit with:    qsub submit_aurora.sh
# Check status:   qstat -u $USER
# View output:    cat pinn_burgers.o<JOB_ID>
#
# Before first use, edit the #PBS -A line below with your project
# allocation name.

# ── PBS directives ───────────────────────────────────────────────────
#PBS -A <YOUR_PROJECT>
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -N pinn_burgers
#PBS -j oe

# ── Environment setup ───────────────────────────────────────────────
# Load the frameworks module — this provides PyTorch with XPU (Intel
# GPU) support, plus numpy.  No need to install torch separately.
module load frameworks

# Install plotting/solver libraries if not already present.
# --user installs to ~/.local (no admin privileges needed).
# These are fast no-ops if already installed.
pip install --user matplotlib scipy

# ── Navigate to the script directory ─────────────────────────────────
cd ${PBS_O_WORKDIR}

echo "============================================"
echo "  Job:    ${PBS_JOBID}"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Dir:    $(pwd)"
echo "  Python: $(python --version)"
echo "============================================"

# ── Run on Intel GPU (XPU) ───────────────────────────────────────────
echo ""
echo ">>> Running on XPU (Intel GPU) <<<"
python pinn_burgers.py --device xpu --output-dir outputs_xpu

# ── (Optional) Run on CPU for comparison ─────────────────────────────
# Uncomment the lines below to also run a CPU-only version.
# This is useful for comparing runtimes.
#
# echo ""
# echo ">>> Running on CPU <<<"
# python pinn_burgers.py --device cpu --output-dir outputs_cpu

echo ""
echo ">>> Job complete: $(date) <<<"
