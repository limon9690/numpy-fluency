#!/usr/bin/env bash
# Put problem stubs back to the version they were first added in.
#
#   scripts/reset.sh 012        reset one problem
#   scripts/reset.sh 001-008    reset a range
#   scripts/reset.sh all        reset every problem
#
# Only problem.py is reset. Your old attempts stay in git history, and
# TRACKER.md is left alone (update it yourself).
# Needs a full clone (a normal `git clone`), not a shallow one.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ $# -ne 1 ]; then
  echo "usage: scripts/reset.sh 012 | 001-008 | all" >&2
  exit 1
fi

if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
  echo "This is a shallow clone; run 'git fetch --unshallow' first." >&2
  exit 1
fi

case "$1" in
  all) numbers=$(ls -d problems/*/[0-9][0-9][0-9]_* | sed -E 's#.*/([0-9]{3})_.*#\1#') ;;
  *-*) a=${1%-*}; b=${1#*-}; numbers=$(seq -f "%03g" "$((10#$a))" "$((10#$b))") ;;
  *)   numbers=$(printf "%03d" "$((10#$1))") ;;
esac

for n in $numbers; do
  for dir in problems/*/"${n}"_*; do
    [ -d "$dir" ] || continue
    file="$dir/problem.py"
    first=$(git log --diff-filter=A --format=%H -- "$file" | tail -n 1)
    if [ -z "$first" ]; then
      echo "skip $n: $file isn't committed yet"
      continue
    fi
    git show "$first:$file" > "$file"
    echo "reset $file"
  done
done
