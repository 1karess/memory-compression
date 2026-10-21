#!/bin/bash
# Submit a job script to the primary partition if it can take it now, otherwise to primary + fallback
# (whichever frees first). Partition names come from jobs/cluster.local.sh (not in git):
#   MZ_PARTITION=<primary>  MZ_FALLBACK=<fallback>
# Usage: jobs/submit.sh [--gpu] <job.sh> [sbatch args...] [-- script args...]
set -euo pipefail
[ -f "$(dirname "$0")/cluster.local.sh" ] && source "$(dirname "$0")/cluster.local.sh"
PRIMARY="${MZ_PARTITION:?set MZ_PARTITION in jobs/cluster.local.sh}"
FALLBACK="${MZ_FALLBACK:-}"
NEED_GPU=0
if [ "${1:-}" = "--gpu" ]; then NEED_GPU=1; shift; fi
JOB="$1"; shift

SB=(); SA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; SA=("$@"); break; fi
  SB+=("$1"); shift
done

PART=$PRIMARY
if [ "$NEED_GPU" = 1 ] && [ -n "$FALLBACK" ]; then
  total=$(sinfo -h -p "$PRIMARY" -N -o "%G" | { grep -o 'gpu:[^:]*:[0-9]*' || true; } | awk -F: '{s+=$3} END{print s+0}')
  used=$(squeue -h -p "$PRIMARY" -t RUNNING -o "%b" | { grep -o 'gpu[^,]*' || true; } | awk -F: '{s+=$NF} END{print s+0}')
  echo "$PRIMARY GPUs: total=$total running=$used"
  if [ "$total" -eq 0 ]; then echo "warning: could not parse $PRIMARY GPU count" >&2; fi
  if [ "$used" -ge "$total" ]; then PART=$PRIMARY,$FALLBACK; fi
fi
echo "partition=$PART"
sbatch --partition="$PART" ${SB[@]+"${SB[@]}"} "$JOB" ${SA[@]+"${SA[@]}"}
