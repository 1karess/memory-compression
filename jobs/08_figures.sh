#!/bin/bash
#SBATCH --job-name=mz_figures
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Meeting figures (CPU only).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export MPLCONFIGDIR="$ROOT/.mplconfig"
python scripts/make_figures.py
