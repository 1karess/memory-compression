#!/bin/bash
#SBATCH --job-name=mz_main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Main first-round experiment on CCPP with the pre-registered memory r0_n256 and k = 64:
#   teacher data (distill 1024 + dev 256) -> lr choice {5e-3, 2e-2} on seed 42 by dev KL
#   -> seeds 43, 44 at that lr -> test table (B, B-shuffle, C-init, C x 3 seeds, KL(B||A)) + A_long on tune/test
# Each stage skips itself if its outputs already exist, so the job can be resubmitted after a failure.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1  # patch P4: fp32 master cache parameters (see results/cartridges_checks.json adam_update)
SUBSET=r0_n256
K=64
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
M=outputs/main
TD=$M/teacher
echo "host=$(hostname) date=$(date -Is) subset=$SUBSET k=$K epochs=$EPOCHS batch=$BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

stage "P4 check (fp32 master cache)"
[ -f results/p4_check.json ] || python scripts/p4_check.py

stage "teacher data"
if [ ! -f $TD/summary.json ]; then
  python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --splits distill,dev --out_dir $TD
fi
cat $TD/summary.json

train_and_select() {  # lr seed
  local run=$M/runs/lr$1_s$2
  if [ ! -f $run/dev_selection.json ]; then
    if [ ! -f $run/TRAIN_DONE ]; then
      python scripts/train_cartridge.py --data $TD/distill.parquet --memory $TD/memory.txt \
        --k $K --lr $1 --epochs $EPOCHS --batch $BATCH --seed $2 --run_dir $run
    fi
    python scripts/eval_cartridge.py select --run_dir $run --memory $TD/memory.txt --dev_parquet $TD/dev.parquet
  fi
  cat $run/dev_selection.json
}

stage "lr choice (seed 42)"
train_and_select 5e-3 42
train_and_select 2e-2 42
LR=$(python - <<EOF
import json
d = {lr: json.load(open(f"$M/runs/lr{lr}_s42/dev_selection.json"))["best_dev_kl"] for lr in ["5e-3", "2e-2"]}
print(min(d, key=d.get))
EOF
)
echo "chosen lr=$LR"
echo "$LR" > $M/chosen_lr.txt

stage "seeds 43, 44 at lr=$LR"
train_and_select $LR 43
train_and_select $LR 44

stage "A with long answers (tune, test)"
[ -f $M/A_long_tune.csv ] || python scripts/eval_a_long.py --split tune --out $M
[ -f $M/A_long_test.csv ] || python scripts/eval_a_long.py --split test --out $M

stage "test table"
python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $M/final \
  --cartridges C_init=$M/runs/lr${LR}_s42/cache_init.pt \
               C_s42=$M/runs/lr${LR}_s42/best_cache.pt \
               C_s43=$M/runs/lr${LR}_s43/best_cache.pt \
               C_s44=$M/runs/lr${LR}_s44/best_cache.pt
echo "done $(date -Is)"
