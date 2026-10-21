#!/bin/bash
#SBATCH --job-name=mz_dump_xlsx
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; source venv/bin/activate
python scripts/dump_xlsx.py "$1"
which soffice libreoffice 2>&1; ls /opt/apps 2>/dev/null | grep -i office
