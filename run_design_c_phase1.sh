#!/bin/bash
# Phase 1 adapter-integration verification for Design C.
# Runs the two learned-Sinkhorn base configs (FP32 + FP8 E4M3 on Conv1d) through
# QBench's feature-matching pipeline with the random-init learned head.
# Head is random-init so pose AUC will be near-zero — the purpose is to confirm
# the new pipeline runs end-to-end without Python errors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/runspace/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOST="$LOG_DIR/phase1_learned_${TIMESTAMP}.log"
LOG_CONTAINER="runspace/outputs/logs/phase1_learned_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

docker exec qbench bash -c "\
  PYTHONPATH=/app python runspace/run_all.py \
    --base-configs \
      runspace/inputs/base_configs/sp_sg_learned_scannet_fp32.yaml \
      runspace/inputs/base_configs/sp_sg_learned_scannet_fp8e4m3.yaml \
    --task feature_matching \
    --stop-on-error \
    2>&1 | tee ${LOG_CONTAINER}"

echo "Phase 1 integration log: ${LOG_HOST}"
