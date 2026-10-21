#!/bin/bash
#SBATCH --job-name=mz_k_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Cartridge size sweep on the main memory r0_n256 and the main teacher data: k = 16 and 256, seeds 42, 43, 44
# (k = 64 is the main run). Same lr / updates / selection as the main run.
# Test table outputs/robust/k_sweep/final: A, B, B-shuffle, C_k16 x 3, C_k256 x 3.
# Each stage skips itself if its outputs already exist, so the job can be resubmitted after a failure.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
SUBSET=r0_n256
LR=2e-2
TD=outputs/main/teacher
O=outputs/robust/k_sweep
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

CARTS=()
for k in 16 256; do
  for s in 42 43 44; do
    dir=$O/runs/k${k}_s$s
    stage "run k${k}_s$s"
    if [ ! -f $dir/dev_selection.json ]; then
      if [ ! -f $dir/TRAIN_DONE ]; then
        python scripts/train_cartridge.py --data $TD/distill.parquet --memory $TD/memory.txt --k $k --lr $LR --epochs 20 \
          --batch 8 --seed $s --save_every 32 --run_dir $dir
      fi
      python scripts/eval_cartridge.py select --run_dir $dir --memory $TD/memory.txt --dev_parquet $TD/dev.parquet
    fi
    cat $dir/dev_selection.json
    CARTS+=("C_k${k}_s$s=$dir/best_cache.pt")
  done
done

stage "test table"
python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $O/final --cartridges "${CARTS[@]}"
echo "done $(date -Is)"
