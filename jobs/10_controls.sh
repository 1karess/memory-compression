#!/bin/bash
#SBATCH --job-name=mz_controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Information-source controls for stage 0: is C compressing M, or regressing on the teacher's answers?
# Same setting as the main run (r0_n256, k = 64, lr = 2e-2, seed 42, 640 updates, best checkpoint by dev KL):
#   shufT    : teacher = B-shuffle (distill + dev answers from the PE-permuted memory); Z init from that memory
#   randinit : main teacher data, Z initialized from unrelated text instead of M
#   n64/n256 : first 64 / 256 main distillation conversations, epochs scaled to keep 640 updates
#   synth    : 1024 query-independent inputs (features uniform over the memory pool's range) answered by B
# then one test table with all of them (B and B-shuffle are re-run as references).
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
K=64
LR=2e-2
SEED=42
TD=outputs/main/teacher
C=outputs/ctrl
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv

stage() { echo; echo "== $1 == $(date -Is)"; }

stage "teacher data: B-shuffle (distill, dev)"
[ -f $C/teacher_shuf/summary.json ] || \
  python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --shuffle_pe --splits distill,dev --out_dir $C/teacher_shuf
cat $C/teacher_shuf/summary.json

stage "teacher data: B on synthetic inputs"
[ -f $C/teacher_synth/summary.json ] || \
  python scripts/make_teacher_data.py --task ccpp --subset $SUBSET --splits synth --out_dir $C/teacher_synth
cat $C/teacher_synth/summary.json

run() {  # name data init_text epochs sel_memory sel_dev [extra train args]
  local name=$1 data=$2 init=$3 epochs=$4 sel_mem=$5 sel_dev=$6; shift 6
  local dir=$C/runs/$name
  stage "run $name"
  if [ ! -f $dir/dev_selection.json ]; then
    if [ ! -f $dir/TRAIN_DONE ]; then
      python scripts/train_cartridge.py --data $data --memory $init --k $K --lr $LR --epochs $epochs --batch 8 \
        --seed $SEED --save_every 32 --run_dir $dir "$@"
    fi
    python scripts/eval_cartridge.py select --run_dir $dir --memory $sel_mem --dev_parquet $sel_dev
  fi
  cat $dir/dev_selection.json
}

run shufT    $C/teacher_shuf/distill.parquet $C/teacher_shuf/memory.txt          20  $C/teacher_shuf/memory.txt $C/teacher_shuf/dev.parquet
run randinit $TD/distill.parquet             data/processed/unrelated_text.txt   20  $TD/memory.txt $TD/dev.parquet
run n256     $TD/distill.parquet             $TD/memory.txt                      80  $TD/memory.txt $TD/dev.parquet --limit 256
run n64      $TD/distill.parquet             $TD/memory.txt                      320 $TD/memory.txt $TD/dev.parquet --limit 64
run synth    $C/teacher_synth/synth.parquet  $TD/memory.txt                      20  $TD/memory.txt $TD/dev.parquet

stage "test table"
python scripts/eval_cartridge.py final --subset $SUBSET --split test --out $C/final \
  --cartridges C_main_s42=outputs/main/runs/lr${LR}_s42/best_cache.pt \
               C_shufT=$C/runs/shufT/best_cache.pt \
               C_randinit_init=$C/runs/randinit/cache_init.pt \
               C_randinit=$C/runs/randinit/best_cache.pt \
               C_n256=$C/runs/n256/best_cache.pt \
               C_n64=$C/runs/n64/best_cache.pt \
               C_synth=$C/runs/synth/best_cache.pt
echo "done $(date -Is)"
