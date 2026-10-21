#!/bin/bash
#SBATCH --job-name=mz_teacher_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Step 2: teacher gate (A / B / B-shuffle over memory lengths) + prefix-KV equivalence. GPU.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/teacher_gate.py "$@"
echo "done $(date -Is)"
