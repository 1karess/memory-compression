#!/bin/bash
#SBATCH --job-name=mz_fill_record
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Fill docs/记忆压缩实验_阶段0_执行跟踪_Qwen3-4B.xlsx from the result files, then dump it for checking. CPU only.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
source "$ROOT/venv/bin/activate"
python scripts/fill_record.py docs/input_record.xlsx "docs/记忆压缩实验_阶段0_执行跟踪_Qwen3-4B.xlsx"
python scripts/dump_xlsx.py "docs/记忆压缩实验_阶段0_执行跟踪_Qwen3-4B.xlsx" > logs/record_dump.txt 2>&1
echo dumped
