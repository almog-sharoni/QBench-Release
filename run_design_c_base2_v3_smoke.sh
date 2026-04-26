#!/bin/bash
# Smoke test for the v3 base-2 cold-start training pipeline.
# Runs ATTACHED so the caller sees pass/fail immediately.
# 1 epoch, 200 train pairs (sampled from a 1000-pair pool), 20 val pairs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/design_c_base2_v3_smoke_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/design_c_base2_v3_smoke_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_base2_v3_smoke.yaml \
    2>&1 | tee ${LOG_CONTAINER}"

echo ""
echo "Smoke log: ${LOG_HOST}"
