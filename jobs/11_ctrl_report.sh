#!/bin/bash
#SBATCH --job-name=mz_ctrl_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Summary of the information-source controls (CPU only); run after jobs/10_controls.sh.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
python scripts/ctrl_report.py
