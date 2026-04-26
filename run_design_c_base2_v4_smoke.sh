#!/bin/bash
# Smoke variant of run_design_c_base2_v4_train.sh: 1 epoch, 200 train pairs,
# 30 val pairs. Verifies the quantized-backbone training path before kicking
# off the full 30-epoch run. Foreground (not detached) so logs stream live.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/design_c_base2_v4_smoke_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/design_c_base2_v4_smoke_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec qbench bash -c "\
  PYTHONPATH=/app python runspace/train.py \
    --config runspace/inputs/train_configs/design_c_t3_base2_v4_smoke.yaml \
    2>&1 | tee ${LOG_CONTAINER}"

echo ""
echo "Smoke run complete."
echo "Log: ${LOG_HOST}"
