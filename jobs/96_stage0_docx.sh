#!/bin/bash
#SBATCH --job-name=mz_docx
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Stage-0 result report as Word (CPU only): docx-js in tools/node, then a pandoc read-back for checking.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
mkdir -p tools/node
[ -d tools/node/node_modules/docx ] || npm install --prefix tools/node docx@9 --no-audit --no-fund
OUT="$ROOT/docs/记忆压缩实验_阶段0_结果报告.docx"
NODE_PATH=tools/node/node_modules node scripts/make_stage0_docx.js "$OUT"
pandoc -t plain "$OUT" | head -80
