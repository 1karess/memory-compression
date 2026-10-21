#!/bin/bash
#SBATCH --job-name=mz_pycompile
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
# Syntax check of all scripts (compute node only).
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
source "$ROOT/venv/bin/activate"
python -m py_compile "$ROOT"/scripts/*.py && echo "py_compile OK"
