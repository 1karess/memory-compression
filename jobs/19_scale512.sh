#!/bin/bash
#SBATCH --job-name=mz_scale512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Scale-up: the main experiment on a 512-row memory (SUBSET=r0_n512, r1_n512 or r2_n512; about 22k tokens, each one
# contains the 256-row memory of the same draw), settings unchanged (lr 2e-2, 640 updates, best checkpoint by dev KL,
# same 1024 distill / 256 dev / 2000 test queries). KS = cartridge sizes to train, seeds 42, 43, 44 each.
# Test table outputs/scale/$SUBSET/final: A, B, B-shuffle, C_k{k}_s{seed}.
# Each stage skips itself if its outputs already exist, so the job can be resubmitted after a failure.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
SUBSET="${SUBSET:?set SUBSET=r0_n512, r1_n512 or r2_n512}"
KS="${KS:-64}"
LR=2e-2
O=outputs/scale/$SUBSET
TD=$O/teacher
echo "host=$(hostname) date=$(date -Is) subset=$SUBSET ks=$KS"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

stage "teacher data: B"
[ -f $TD/summary.json ] || python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --splits distill,dev --out_dir $TD
cat $TD/summary.json

CARTS=()
for k in $KS; do
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
