#!/bin/bash
# Phase 2 training smoke test — 2 epochs, 50 train pairs, 100 val pairs.
# Confirms the training loop builds, loss decreases, and checkpoints land.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase2_smoke_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase2_smoke_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_scannet.yaml \
    --epochs 2 \
    --max-train-pairs 50 \
    --max-val-pairs 100 \
    2>&1 | tee ${LOG_CONTAINER}"

echo "Phase 2 smoke log: ${LOG_HOST}"
