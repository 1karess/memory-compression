#!/bin/bash
#SBATCH --job-name=mz_e2e_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Small end-to-end tests of the full teacher-data -> Cartridges training -> HF evaluation pipeline:
#   toy    : y = 3x + 7, 64-row memory, 200 train / 50 dev queries, k = 32
#   ccpp32 : 32 real distill queries, k = 64, overfit check (selection on the same 32 queries)
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
export CARTRIDGES_DIR="$ROOT/third_party/cartridges" CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/cartridges" WANDB_MODE=offline
export TRITON_CACHE_DIR="$ROOT/.triton-cache" TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor-cache"
export MZ_FP32_CACHE=1  # patch P4: fp32 master cache parameters (see results/cartridges_checks.json adam_update)
SUBSET="${SUBSET:-r0_n256}"
echo "host=$(hostname) date=$(date -Is) subset=$SUBSET"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== P4 check (fp32 master cache) =="
python scripts/p4_check.py

echo "== toy: teacher data =="
T=outputs/toy
python scripts/make_teacher_data.py --task toy --out_dir $T/teacher
echo "== toy: train k=32 =="
python scripts/train_cartridge.py --data $T/teacher/train.parquet --memory $T/teacher/memory.txt \
  --k 32 --lr 2e-2 --epochs 15 --batch 4 --save_every 20 --run_dir $T/run_k32
echo "== toy: dev KL per checkpoint =="
python scripts/eval_cartridge.py select --run_dir $T/run_k32 --memory $T/teacher/memory.txt --dev_parquet $T/teacher/dev.parquet

echo "== ccpp32: teacher data =="
C=outputs/ccpp32
python scripts/make_teacher_data.py --task ccpp --subset "$SUBSET" --splits distill --limit 32 --out_dir $C/teacher
echo "== ccpp32: train k=64 (overfit) =="
python scripts/train_cartridge.py --data $C/teacher/distill.parquet --memory $C/teacher/memory.txt \
  --k 64 --lr 2e-2 --epochs 40 --batch 2 --save_every 20 --run_dir $C/run_k64
echo "== ccpp32: KL on the same 32 training queries =="
python scripts/eval_cartridge.py select --run_dir $C/run_k64 --memory $C/teacher/memory.txt --dev_parquet $C/teacher/distill.parquet
echo "done $(date -Is)"
