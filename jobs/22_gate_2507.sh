#!/bin/bash
#SBATCH --job-name=mz_gate_2507
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Scale-up feasibility with Qwen3-4B-Instruct-2507 (native 262k context, no thinking mode, no YaRN needed):
# download it to models/, then the teacher gate on the tune split: A, B, B-shuffle at 256 ... 2560 rows x 3 resamples
# (large memories from jobs/21), plus the cached-vs-full and chunked-prefill checks. Compare with results/gate (Qwen3-4B).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_home"
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv

M=Qwen3-4B-Instruct-2507
if [ ! -f models/$M.REVISION ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python - <<EOF
from huggingface_hub import snapshot_download, HfApi
repo = "Qwen/$M"
sha = HfApi().model_info(repo).sha
print("model revision:", sha)
snapshot_download(repo, revision=sha, local_dir="$ROOT/models/$M")
open("$ROOT/models/$M.REVISION", "w").write(sha + "\n")
EOF
fi
cat models/$M.REVISION
grep -E "max_position_embeddings|rope_theta|rope_scaling" models/$M/config.json

export HF_HUB_OFFLINE=1 MZ_MODEL=$M
[ -f data/processed/memory_subsets_large.json ] || python scripts/make_large_subsets.py
python scripts/teacher_gate.py --skip_fp32 --lengths 256,512,1024,2048,2560 --out results/gate_2507
echo "done $(date -Is)"
