#!/bin/bash
#SBATCH --job-name=mz_git_push
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Commit code, jobs, docs and results (see .gitignore) and push to github.com/1karess/memory-compression.
# The token is read from ~/.mz_gh_token by a credential helper and never written to the repo, the log or the URL.
# Commits are authored as 1karess with the GitHub no-reply e-mail. Before pushing, the staged files (and the text
# inside .docx files) are checked for personal / server details; the push is aborted if any are found.
# MSG = commit message (default: "Update"). RESET=1 starts the history over and force-pushes (replaces old commits).
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"  # submit from the project root
cd "$ROOT"
REMOTE=https://github.com/1karess/memory-compression.git
MSG="${MSG:-Update}"
[ -s ~/.mz_gh_token ] || { echo "missing ~/.mz_gh_token" >&2; exit 1; }
helper='!f() { echo username=x-access-token; echo "password=$(cat ~/.mz_gh_token)"; }; f'
source jobs/cluster.local.sh  # not in git; MZ_PRIVATE = regex of personal / server details to keep out of the repo
PRIVATE="${MZ_PRIVATE:?set MZ_PRIVATE in jobs/cluster.local.sh}"

if [ "${RESET:-0}" = 1 ]; then rm -rf .git; fi
[ -d .git ] || { git init -q -b main; git remote add origin "$REMOTE"; }
gh_id=$(curl -s https://api.github.com/users/1karess | sed -n 's/^  "id": \([0-9]*\),$/\1/p')
[ -n "$gh_id" ] || { echo "could not look up the GitHub user id" >&2; exit 1; }
git config user.name "1karess"
git config user.email "${gh_id}+1karess@users.noreply.github.com"

mkdir -p patches
git -C third_party/cartridges diff > patches/cartridges.diff
git -C third_party/cartridges rev-parse HEAD > patches/cartridges.COMMIT

git add -A
git status --short | head -100
leaks=$(git grep --cached -I -n -i -E "$PRIVATE" || true)
while IFS= read -r f; do
  hits=$(unzip -p "$f" | grep -o -i -E "$PRIVATE" || true)
  [ -z "$hits" ] || leaks+=$'\n'"$f: $(echo $hits)"
done < <(git -c core.quotePath=false diff --cached --name-only --diff-filter=AM -- '*.docx')
if [ -n "$leaks" ]; then echo "personal / server details found, not pushing:"; echo "$leaks" | head -30; exit 1; fi

if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -F - <<EOF
$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Avr46LzxbAizYMhw2XyQgw
EOF
fi
git log --format='%h %an <%ae> %s' -3
if [ "${RESET:-0}" = 1 ]; then
  git -c credential.helper= -c credential.helper="$helper" push --force -u origin main
else
  git -c credential.helper= -c credential.helper="$helper" push -u origin main
fi
echo "done $(date -Is)"
