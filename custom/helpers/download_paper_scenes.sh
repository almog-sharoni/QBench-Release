#!/usr/bin/env bash
set -u

PAIRS_SRC=/home/spark1/yarden/QBench2/QBench-Release/custom/SuperGluePretrainedNetwork/assets/scannet_test_pairs_with_gt.txt
SCANNET_ROOT=/data/scannet
NUM_SCENES=100

DIR="$(cd "$(dirname "$0")" && pwd)"
SCENES_LIST=/tmp/scannet_test_scenes.txt
LOG="$SCANNET_ROOT/_download.log"

mkdir -p "$SCANNET_ROOT"

python3 - "$PAIRS_SRC" "$NUM_SCENES" "$SCENES_LIST" <<'PY'
import sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seen = []
seen_set = set()
with open(src) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 2:
            continue
        for p in (parts[0], parts[1]):
            s = p.split('/')[1]
            if s not in seen_set:
                seen_set.add(s)
                seen.append(s)
        if n > 0 and len(seen) >= n:
            break
if n > 0:
    seen = seen[:n]
with open(out, 'w') as g:
    g.write('\n'.join(seen) + '\n')
print(f'selected {len(seen)} scenes -> {out}')
PY

echo "=== start $(date -Iseconds) ===" >> "$LOG"
while IFS= read -r scene; do
  [ -z "$scene" ] && continue
  echo "--- $scene $(date -Iseconds) ---" >> "$LOG"
  printf '\n' | python3 "$DIR/download-scannet.py" -o "$SCANNET_ROOT" --id "$scene" --type .sens --skip_existing >> "$LOG" 2>&1
done < "$SCENES_LIST"
echo "=== done $(date -Iseconds) ===" >> "$LOG"

echo "downloaded $NUM_SCENES scenes to $SCANNET_ROOT/scans_test (log: $LOG)"
