#!/bin/bash
#SBATCH --job-name=mz_robust_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Robustness summary (CPU only); run after jobs/15, 16, 17.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
python scripts/robust_report.py
