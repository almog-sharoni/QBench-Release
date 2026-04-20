#!/usr/bin/env bash
set -u
SCRIPT="/home/spark1/yarden/QBench2/QBench-Release/custom/helpers/download-scannet.py"
OUT=/data/scannet/ref
LOG=/data/scannet/ref/_download.log
SCENE_LIST="${1:-/tmp/scannet_test_scenes.txt}"
echo "=== start $(date -Iseconds) list=$SCENE_LIST ===" >> "$LOG"
while IFS= read -r scene; do
  [ -z "$scene" ] && continue
  echo "--- $scene $(date -Iseconds) ---" >> "$LOG"
  # Script prompts once for TOS confirmation; feed one empty line.
  printf '\n' | python3 "$SCRIPT" -o "$OUT" --id "$scene" --type .sens --skip_existing >> "$LOG" 2>&1
done < "$SCENE_LIST"
echo "=== done $(date -Iseconds) ===" >> "$LOG"
