#!/usr/bin/env bash
set -u

PAIRS_SRC=/home/spark1/yarden/QBench2/QBench-Release/custom/SuperGluePretrainedNetwork/assets/scannet_test_pairs_with_gt.txt
NUM_SCENES=100
PAIRS_OUT=/home/spark1/yarden/QBench2/QBench-Release/runspace/inputs/scannet/pairs_test_paper_100scenes.txt

mkdir -p "$(dirname "$PAIRS_OUT")"

python3 - "$PAIRS_SRC" "$NUM_SCENES" "$PAIRS_OUT" <<'PY'
import sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
order = []
order_set = set()
rows = []
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
        rows.append((s0, s1, line))
selected = set(order[:n] if n > 0 else order)
kept = [ln for s0, s1, ln in rows if s0 in selected and s1 in selected]
with open(out, 'w') as g:
    g.writelines(kept)
print(f'wrote {len(kept)} pairs across {len(selected)} scenes -> {out}')
PY
