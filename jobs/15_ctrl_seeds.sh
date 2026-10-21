#!/bin/bash
#SBATCH --job-name=mz_ctrl_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Seeds 43 and 44 for the key controls of jobs/10 and jobs/13 (seed 42 already done), same teacher data and settings:
#   GROUP=ctrl   : shufT, n64, synth             -> test table outputs/ctrl/final_seeds
#   GROUP=rshift : rshift (low-AT PE + 10)       -> test table outputs/ctrl/final_rshift_seeds (with the main C at
#                                                   seeds 43, 44, so each seed's shift is measured against its own C)
# Each stage skips itself if its outputs already exist, so the job can be resubmitted after a failure.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1
GROUP="${GROUP:?set GROUP=ctrl or GROUP=rshift}"
SUBSET=r0_n256
K=64
LR=2e-2
TD=outputs/main/teacher
C=outputs/ctrl
echo "host=$(hostname) date=$(date -Is) group=$GROUP"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

run() {  # name seed data init_text epochs sel_memory sel_dev [extra train args]
  local name=$1 seed=$2 data=$3 init=$4 epochs=$5 sel_mem=$6 sel_dev=$7; shift 7
  local dir=$C/runs/${name}_s$seed
  stage "run ${name}_s$seed"
  if [ ! -f $dir/dev_selection.json ]; then
    if [ ! -f $dir/TRAIN_DONE ]; then
      python scripts/train_cartridge.py --data $data --memory $init --k $K --lr $LR --epochs $epochs --batch 8 \
        --seed $seed --save_every 32 --run_dir $dir "$@"
    fi
    python scripts/eval_cartridge.py select --run_dir $dir --memory $sel_mem --dev_parquet $sel_dev
  fi
  cat $dir/dev_selection.json
}

if [ "$GROUP" = ctrl ]; then
  for s in 43 44; do
    run shufT $s $C/teacher_shuf/distill.parquet $C/teacher_shuf/memory.txt 20  $C/teacher_shuf/memory.txt $C/teacher_shuf/dev.parquet
    run n64   $s $TD/distill.parquet             $TD/memory.txt              320 $TD/memory.txt $TD/dev.parquet --limit 64
    run synth $s $C/teacher_synth/synth.parquet  $TD/memory.txt              20  $TD/memory.txt $TD/dev.parquet
  done
  stage "test table"
  python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $C/final_seeds \
    --cartridges C_shufT_s43=$C/runs/shufT_s43/best_cache.pt C_shufT_s44=$C/runs/shufT_s44/best_cache.pt \
                 C_n64_s43=$C/runs/n64_s43/best_cache.pt     C_n64_s44=$C/runs/n64_s44/best_cache.pt \
                 C_synth_s43=$C/runs/synth_s43/best_cache.pt C_synth_s44=$C/runs/synth_s44/best_cache.pt
elif [ "$GROUP" = rshift ]; then
  TS=$C/teacher_rshift
  for s in 43 44; do
    run rshift $s $TS/distill.parquet $TS/memory.txt 20 $TS/memory.txt $TS/dev.parquet
  done
  stage "test table"
  python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $C/final_rshift_seeds --pe_shift 10 --low_at_only \
    --cartridges C_main_s43=outputs/main/runs/lr${LR}_s43/best_cache.pt C_main_s44=outputs/main/runs/lr${LR}_s44/best_cache.pt \
                 C_rshift_s43=$C/runs/rshift_s43/best_cache.pt        C_rshift_s44=$C/runs/rshift_s44/best_cache.pt
else
  echo "unknown GROUP=$GROUP" >&2; exit 1
fi
echo "done $(date -Is)"
