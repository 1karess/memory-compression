#!/bin/bash
#SBATCH --job-name=mz_pseudo_ctrl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Pseudo-label regression control (CPU only).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
python scripts/pseudo_label_control.py
