#!/bin/bash
#SBATCH --job-name=mz_big_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Scale-up gate: teacher B vs B-shuffle with memories up to ~2.5k rows (~110k tokens, YaRN). GPU.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/big_gate.py "$@"
echo "done $(date -Is)"
