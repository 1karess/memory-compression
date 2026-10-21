#!/bin/bash
#SBATCH --job-name=mz_other_mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Repeat the main experiment and the region counterfactual on another 256-row memory (SUBSET=r1_n256 or r2_n256),
# with the r0 settings unchanged (k = 64, lr = 2e-2, 640 updates, best checkpoint by dev KL):
#   C seeds 42, 43, 44 from teacher B; C_rshift seed 42 from teacher B'' (PE + 10 where AT < this memory's median AT)
# Test table outputs/robust/$SUBSET/final: A, B, B-shuffle, B_shift (= B''), C x 3, C_rshift.
# Each stage skips itself if its outputs already exist, so the job can be resubmitted after a failure.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
SUBSET="${SUBSET:?set SUBSET=r1_n256 or r2_n256}"
K=64
LR=2e-2
O=outputs/robust/$SUBSET
TD=$O/teacher
TS=$O/teacher_rshift
echo "host=$(hostname) date=$(date -Is) subset=$SUBSET"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

stage "teacher data: B"
[ -f $TD/summary.json ] || python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --splits distill,dev --out_dir $TD
cat $TD/summary.json
stage "teacher data: B'' (PE + 10 where AT < median)"
[ -f $TS/summary.json ] || \
  python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --pe_shift 10 --low_at_only --splits distill,dev --out_dir $TS
cat $TS/summary.json

run() {  # name seed teacher_dir
  local dir=$O/runs/$1_s$2 td=$3
  stage "run $1_s$2"
  if [ ! -f $dir/dev_selection.json ]; then
    if [ ! -f $dir/TRAIN_DONE ]; then
      python scripts/train_cartridge.py --data $td/distill.parquet --memory $td/memory.txt --k $K --lr $LR --epochs 20 \
        --batch 8 --seed $2 --save_every 32 --run_dir $dir
    fi
    python scripts/eval_cartridge.py select --run_dir $dir --memory $td/memory.txt --dev_parquet $td/dev.parquet
  fi
  cat $dir/dev_selection.json
}

for s in 42 43 44; do run main $s $TD; done
run rshift 42 $TS

stage "test table"
python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $O/final --pe_shift 10 --low_at_only \
  --cartridges C_s42=$O/runs/main_s42/best_cache.pt C_s43=$O/runs/main_s43/best_cache.pt \
               C_s44=$O/runs/main_s44/best_cache.pt C_rshift_s42=$O/runs/rshift_s42/best_cache.pt
echo "done $(date -Is)"
