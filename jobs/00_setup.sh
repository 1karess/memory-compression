#!/bin/bash
#SBATCH --job-name=mz_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Environment + downloads. Runs on a compute node only.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export ROOT HF_HOME="$ROOT/hf_home"
UV=$HOME/.local/bin/uv

echo "host=$(hostname) date=$(date -Is)"

echo "== internet check =="
curl -sS -o /dev/null -w "huggingface.co %{http_code}\n" https://huggingface.co || echo "huggingface.co unreachable"
curl -sS -o /dev/null -w "archive.ics.uci.edu %{http_code}\n" https://archive.ics.uci.edu || echo "uci unreachable"

echo "== venv =="
if [ ! -x "$ROOT/venv/bin/python" ]; then
  "$UV" venv --python /opt/apps/miniconda3/3.12/bin/python3.12 "$ROOT/venv"
fi
source "$ROOT/venv/bin/activate"
python --version

echo "== packages =="
"$UV" pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$UV" pip install transformers==4.55.0 accelerate huggingface_hub hf_transfer \
  numpy pandas scipy scikit-learn openpyxl pyarrow
# Cartridges itself (deps resolve against the already-pinned torch/transformers)
"$UV" pip install -e "$ROOT/third_party/cartridges"
"$UV" pip freeze > "$ROOT/logs/pip_freeze_setup.txt"

echo "== model =="
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'EOF'
import os
from huggingface_hub import snapshot_download, HfApi
repo = "Qwen/Qwen3-4B"
sha = HfApi().model_info(repo).sha
print("model revision:", sha)
path = snapshot_download(repo, revision=sha, local_dir=os.environ["ROOT"] + "/models/Qwen3-4B")
open(os.environ["ROOT"] + "/models/Qwen3-4B.REVISION", "w").write(sha + "\n")
print("downloaded to", path)
EOF
ls -la "$ROOT/models/Qwen3-4B"

echo "== data =="
mkdir -p "$ROOT/data/raw"
if [ ! -f "$ROOT/data/raw/ccpp.zip" ]; then
  wget -q -O "$ROOT/data/raw/ccpp.zip" "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
fi
cd "$ROOT/data/raw" && unzip -o -q ccpp.zip && find . -maxdepth 3 -type f | head -20
sha256sum "$ROOT/data/raw/ccpp.zip"

echo "== import check =="
python -c "import torch, transformers, sklearn, openpyxl; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__)"
echo "done $(date -Is)"
