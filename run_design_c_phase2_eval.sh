#!/bin/bash
# Phase 2 best.pt eval — run the QBench matching pipeline with the trained
# head loaded from best.pt. Confirms the training → eval roundtrip works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase2_eval_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase2_eval_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec qbench bash -c "\
  PYTHONPATH=/app python runspace/run_all.py \
    --base-configs runspace/inputs/base_configs/sp_sg_learned_scannet_eval_best.yaml \
    --task feature_matching \
    --stop-on-error \
    2>&1 | tee ${LOG_CONTAINER}"

echo "Phase 2 eval log: ${LOG_HOST}"
