#!/bin/bash
#SBATCH --job-name=mz_cart_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Step 3: patched Cartridges unit tests + Flex-vs-HF / dtype / save-load checks with Qwen3-4B. GPU.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1 UV_CACHE_DIR="$ROOT/.uv-cache"
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git -C "$CARTRIDGES_DIR" diff > "$ROOT/logs/cartridges_patch.diff"

$HOME/.local/bin/uv pip install -q pytest
echo "== upstream Qwen3 equivalence tests =="
(cd "$CARTRIDGES_DIR" && python -m pytest -q cartridges/tests/models/test_qwen.py 2>&1 | tail -20) || true

echo "== MZ checks =="
python scripts/cartridges_checks.py
echo "done $(date -Is)"
