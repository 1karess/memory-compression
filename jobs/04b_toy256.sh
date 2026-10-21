#!/bin/bash
#SBATCH --job-name=mz_toy256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# y = 3x + 7 numeric smoke test with a 256-row memory (64 rows made the teacher explain instead of answer).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
echo "host=$(hostname) date=$(date -Is)"
T=outputs/toy256
python scripts/make_teacher_data.py --task toy --toy_mem 256 --toy_queries 250 --out_dir $T/teacher
python scripts/train_cartridge.py --data $T/teacher/train.parquet --memory $T/teacher/memory.txt \
  --k 32 --lr 2e-2 --epochs 15 --batch 4 --save_every 20 --run_dir $T/run_k32
python scripts/eval_cartridge.py select --run_dir $T/run_k32 --memory $T/teacher/memory.txt --dev_parquet $T/teacher/dev.parquet
echo "done $(date -Is)"
