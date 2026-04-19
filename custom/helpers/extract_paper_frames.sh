#!/usr/bin/env bash
set -u

PAIRS_SRC=/home/spark1/yarden/QBench2/QBench-Release/custom/SuperGluePretrainedNetwork/assets/scannet_test_pairs_with_gt.txt
SCANNET_ROOT=/data/scannet/ref
NUM_SCENES=100

DIR="$(cd "$(dirname "$0")" && pwd)"
NEEDED_JSON=/tmp/scannet_extract/needed.json
mkdir -p "$(dirname "$NEEDED_JSON")"

python3 - "$PAIRS_SRC" "$NUM_SCENES" "$NEEDED_JSON" <<'PY'
import sys, json
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
order = []
order_set = set()
pair_rows = []
with open(src) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 2:
            continue
        s0 = parts[0].split('/')[1]
        s1 = parts[1].split('/')[1]
        for s in (s0, s1):
            if s not in order_set:
                order_set.add(s)
                order.append(s)
        pair_rows.append((s0, s1, parts[0], parts[1]))
selected = set(order[:n] if n > 0 else order)
needed = {}
for s0, s1, p0, p1 in pair_rows:
    if s0 in selected and s1 in selected:
        for scene, path in ((s0, p0), (s1, p1)):
            idx = int(path.split('/')[-1].split('-')[1].split('.')[0])
            needed.setdefault(scene, set()).add(idx)
needed = {k: sorted(v) for k, v in needed.items()}
with open(out, 'w') as g:
    json.dump(needed, g)
total = sum(len(v) for v in needed.values())
print(f'selected {len(needed)} scenes, {total} unique frames -> {out}')
PY

python3 "$DIR/extract.py"

echo "extracted frames for $NUM_SCENES scenes under $SCANNET_ROOT/scans_test"
