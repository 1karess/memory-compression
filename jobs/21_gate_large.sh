#!/bin/bash
#SBATCH --job-name=mz_gate_large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Scale-up feasibility: (1) large memories of 1024 / 2048 / 2560 rows (up to ~110k tokens) from memory_pool + reserve,
# (2) teacher gate with Qwen3 YaRN (x4) on the tune split: A, B, B-shuffle at 256 ... 2560 rows x 3 resamples,
# plus cached-vs-full and chunked-prefill checks under YaRN. Compare with the no-YaRN gate in results/gate.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home" HF_HUB_OFFLINE=1
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f data/processed/memory_subsets_large.json ] || python scripts/make_large_subsets.py
python scripts/teacher_gate.py --yarn --skip_fp32 --lengths 256,512,1024,2048,2560 --out results/gate_yarn
echo "done $(date -Is)"
