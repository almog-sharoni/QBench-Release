#!/bin/bash
# Generate Design C training pairs via depth-based overlap filtering.
# Wraps custom/helpers/generate_scannet_pairs.py:
#   - --root /data/scannet/posed_images — color + depth + pose layout
#   - --scenes-file /data/scannet/scannetv2_train.txt — TRAIN SPLIT ONLY
#     (test and val scenes are excluded by construction)
#   - --overlap 0.2..0.8 — SuperGlue paper regime
#   - --pairs-per-scene 50 — stratified across 3 overlap buckets (0.2-0.4,
#     0.4-0.6, 0.6-0.8)
#   - --cache-dir writable location under runspace/outputs
#   - --out runspace/inputs/scannet/pairs_train.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/make_train_pairs_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/make_train_pairs_${TIMESTAMP}.log"
CACHE_DIR="runspace/outputs/design_c_t3/overlap_cache"

mkdir -p "$LOG_DIR" "$SCRIPT_DIR/$CACHE_DIR"

docker exec qbench bash -c "\
  python custom/helpers/generate_scannet_pairs.py \
    --root /data/scannet/posed_images \
    --scenes-file /data/scannet/scannetv2_train.txt \
    --cache-dir ${CACHE_DIR} \
    --pairs-per-scene 200 \
    --workers 16 \
    --overlap-min 0.2 \
    --overlap-max 0.8 \
    --out runspace/inputs/scannet/pairs_train.txt \
    2>&1 | tee ${LOG_CONTAINER}"

echo "Pair-generation log: ${LOG_HOST}"
echo "Cache: $SCRIPT_DIR/$CACHE_DIR"
echo "Output: $SCRIPT_DIR/runspace/inputs/scannet/pairs_train.txt"
