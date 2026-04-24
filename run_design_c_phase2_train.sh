#!/bin/bash
# Design C Phase 2 — full head-only training.
#
# Config:   runspace/inputs/train_configs/design_c_t3_scannet.yaml
# Data:     238K overlap-filtered train pairs (pairs_train.txt, depth-based 0.2-0.8 overlap)
# Val:      ScanNet-paper-100 scenes, 300 pairs, MatchingMetrics pose_auc_10
# Schedule: 30 epochs, batch=1, AdamW lr=1e-4, cosine schedule
# Output:   runspace/outputs/design_c_t3/checkpoints/{ckpt_epNNN.pt, best.pt, history.json}
#
# Runs DETACHED inside the qbench container so the terminal is freed.
# Monitor progress with:   tail -f runspace/outputs/logs/phase2_train_<ts>.log
# Stop with:               docker exec qbench pkill -f 'runspace/train.py'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase2_train_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase2_train_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec -d qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_scannet.yaml \
    > ${LOG_CONTAINER} 2>&1"

echo "Started Design C Phase 2 training, detached in container qbench."
echo "Log: ${LOG_HOST}"
echo ""
echo "Monitor:  tail -f ${LOG_HOST}"
echo "Stop:     docker exec qbench pkill -f 'runspace/train.py'"
