#!/bin/bash
#SBATCH --job-name=mz_shift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Counterfactual memory M' = M with every PE + SHIFT (same rows, same x). Does the teacher B' move by SHIFT,
# and does a cartridge distilled from B' move with it?
#   shift       : teacher B', Z initialized from M'
#   shift_initM : teacher B', Z initialized from the original M (labels and init disagree)
# Same setting as the main run (r0_n256, k = 64, lr = 2e-2, seed 42, 640 updates, best checkpoint by dev KL under B').
# Test table: B, B-shuffle, B_shift, the main C (seed 42) and both shift cartridges.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
SUBSET=r0_n256
SHIFT="${SHIFT:-10}"
K=64
LR=2e-2
SEED=42
TD=outputs/main/teacher
C=outputs/ctrl
TS=$C/teacher_shift
echo "host=$(hostname) date=$(date -Is) shift=$SHIFT"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

stage "teacher data: B' (PE + $SHIFT)"
[ -f $TS/summary.json ] || \
  python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --pe_shift $SHIFT --splits distill,dev --out_dir $TS
cat $TS/summary.json

run() {  # name init_text
  local dir=$C/runs/$1
  stage "run $1"
  if [ ! -f $dir/dev_selection.json ]; then
    if [ ! -f $dir/TRAIN_DONE ]; then
      python scripts/train_cartridge.py --data $TS/distill.parquet --memory $2 --k $K --lr $LR --epochs 20 --batch 8 \
        --seed $SEED --save_every 32 --run_dir $dir
    fi
    python scripts/eval_cartridge.py select --run_dir $dir --memory $TS/memory.txt --dev_parquet $TS/dev.parquet
  fi
  cat $dir/dev_selection.json
}

run shift       $TS/memory.txt
run shift_initM $TD/memory.txt

stage "test table"
python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $C/final_shift --pe_shift $SHIFT \
  --cartridges C_main_s42=outputs/main/runs/lr${LR}_s42/best_cache.pt \
               C_shift=$C/runs/shift/best_cache.pt \
               C_shift_initM=$C/runs/shift_initM/best_cache.pt
echo "done $(date -Is)"
